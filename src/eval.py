"""Evaluation entrypoint for trained cold-protein DTI checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.config import ConfigError, load_and_resolve_config, save_config
from src.data.dataloader import build_dataloaders, describe_graph_cache, describe_splits
from src.metrics import build_metrics_payload
from src.utils import (
    build_logger,
    choose_device,
    deep_merge,
    environment_snapshot,
    read_json,
    round_nested,
    try_import_torch,
    write_json,
)


def _move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _validate_binary_loss_inputs(torch_module: Any, logits: Any, labels: Any, *, stage: str) -> None:
    """Validate BCEWithLogitsLoss inputs before computing evaluation loss."""
    if not bool(torch_module.is_floating_point(logits)):
        raise TypeError(f"{stage} logits must be floating-point tensors. Received {logits.dtype}.")
    if not bool(torch_module.is_floating_point(labels)):
        raise TypeError(f"{stage} labels must be floating-point tensors. Received {labels.dtype}.")
    if tuple(logits.shape) != tuple(labels.shape):
        raise ValueError(
            f"{stage} logits and labels must share the same shape. "
            f"Received {tuple(logits.shape)} and {tuple(labels.shape)}."
        )
    if not bool(torch_module.isfinite(logits).all().item()):
        raise ValueError(f"{stage} logits contain NaN or Inf values.")
    if not bool(torch_module.isfinite(labels).all().item()):
        raise ValueError(f"{stage} labels contain NaN or Inf values.")

    min_label = float(labels.detach().amin().item())
    max_label = float(labels.detach().amax().item())
    if min_label < 0.0 or max_label > 1.0:
        raise ValueError(
            f"{stage} labels must stay within [0, 1] for BCEWithLogitsLoss. "
            f"Observed min={min_label:.6f}, max={max_label:.6f}."
        )


def _resolve_checkpoint_path(run_dir: Path, explicit_checkpoint: str | None) -> Path:
    if explicit_checkpoint:
        return Path(explicit_checkpoint)
    best_checkpoint = run_dir / "checkpoints" / "best.pt"
    latest_checkpoint = run_dir / "checkpoints" / "latest.pt"
    if best_checkpoint.exists():
        return best_checkpoint
    if latest_checkpoint.exists():
        return latest_checkpoint
    raise FileNotFoundError(
        f"No checkpoint found under {run_dir / 'checkpoints'}. "
        "Pass --checkpoint explicitly or train the run first."
    )


def write_evaluation_artifacts(
    *,
    config: dict[str, Any],
    data_summary: dict[str, Any] | None,
    test_metrics: dict[str, Any],
    checkpoint_path: str | None,
    source: str,
    status: str = "completed",
) -> dict[str, Any]:
    run_dir = Path(config["output"]["run_dir"])
    logger = build_logger(run_dir / "eval.log", logger_name=f"eval:{config['run_name']}")

    eval_payload = {
        "status": status,
        "mode": "eval",
        "run_name": config["run_name"],
        "checkpoint": checkpoint_path,
        "source": source,
        "environment": environment_snapshot(),
        "data": data_summary,
        "test": test_metrics,
    }
    rounded_payload = round_nested(eval_payload)
    write_json(rounded_payload, run_dir / "eval_metrics.json")

    metrics_json = run_dir / "metrics.json"
    if metrics_json.exists():
        merged = deep_merge(read_json(metrics_json), {"eval": rounded_payload})
        write_json(merged, metrics_json)

    if checkpoint_path:
        logger.info("Evaluation source checkpoint: %s", checkpoint_path)
    else:
        logger.info("Evaluation source: %s", source)
    logger.info("Evaluation finished. Metrics written to %s", run_dir / "eval_metrics.json")
    return rounded_payload


def run_evaluation(config: dict[str, Any], checkpoint_path: str | None, dry_run: bool) -> Path:
    run_dir = Path(config["output"]["root_dir"]) / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    config["output"]["run_dir"] = str(run_dir)
    config["output"]["checkpoint_dir"] = str(run_dir / "checkpoints")
    save_config(config, run_dir / "config.resolved.yaml")

    logger = build_logger(run_dir / "eval.log", logger_name=f"eval:{config['run_name']}")
    split_summary = describe_splits(config["data"])
    graph_cache_summary = describe_graph_cache(config["data"])
    resolved_checkpoint = _resolve_checkpoint_path(run_dir, checkpoint_path)
    logger.info("Evaluating checkpoint: %s", resolved_checkpoint)
    logger.info("Split summary: %s", round_nested(split_summary))
    logger.info("Graph cache: %s", round_nested(graph_cache_summary))

    if dry_run:
        payload = {
            "status": "dry_run",
            "mode": "eval",
            "run_name": config["run_name"],
            "checkpoint": str(resolved_checkpoint),
            "split_summary": round_nested(split_summary),
            "graph_cache": round_nested(graph_cache_summary),
            "environment": environment_snapshot(),
        }
        write_json(payload, run_dir / "eval_metrics.json")
        logger.info("Dry-run evaluation completed successfully.")
        return run_dir

    torch = try_import_torch()
    from src.model.dti_model import build_model

    device = choose_device(str(config["training"]["device"]), torch)
    model = build_model(config).to(device)
    checkpoint = torch.load(resolved_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])

    loaders, loader_metadata = build_dataloaders(config)
    criterion = torch.nn.BCEWithLogitsLoss()

    logits_list = []
    labels_list = []
    target_ids: list[str] = []
    total_loss = 0.0
    n_batches = 0
    max_eval_batches = config["training"]["max_eval_batches"]
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None

    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loaders["test"]):
            if max_eval_batches is not None and batch_index >= max_eval_batches:
                break
            batch = _move_batch_to_device(batch, device)
            outputs = model(batch)
            logits = outputs["logits"]
            labels = batch["labels"]
            _validate_binary_loss_inputs(torch, logits, labels, stage="eval")
            loss = criterion(logits, labels)

            total_loss += float(loss.item())
            n_batches += 1
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())
            target_ids.extend(batch["target_uniprot_ids"])

    if not logits_list:
        raise RuntimeError("Evaluation loader produced zero batches.")

    logits_np = np.concatenate(logits_list)
    labels_np = np.concatenate(labels_list)
    metrics_payload = build_metrics_payload(
        labels=labels_np,
        logits=logits_np,
        group_ids=target_ids,
        threshold=float(config["metrics"]["threshold"]),
        ks=[int(item) for item in config["metrics"]["ks"]],
        ef_fractions=[float(item) for item in config["metrics"]["ef_fractions"]],
        loss=total_loss / max(n_batches, 1),
    )

    write_evaluation_artifacts(
        config=config,
        data_summary=loader_metadata["summary"],
        test_metrics=metrics_payload,
        checkpoint_path=str(resolved_checkpoint),
        source="checkpoint",
    )
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a cold-protein DTI checkpoint from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the experiment config file.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--set", action="append", default=[], help="Override a config value with key=value.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and checkpoint paths only.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config = load_and_resolve_config(
            config_path=args.config,
            seed=args.seed,
            mode="eval",
            overrides=args.set,
        )
        run_evaluation(config=config, checkpoint_path=args.checkpoint, dry_run=bool(args.dry_run))
    except ConfigError as exc:
        raise SystemExit(f"[config-error] {exc}") from exc


if __name__ == "__main__":
    main()
