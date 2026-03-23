"""Training entrypoint for config-driven cold-protein DTI experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.config import ConfigError, load_and_resolve_config, save_config
from src.data.dataloader import build_dataloaders, describe_graph_cache, describe_splits
from src.metrics import build_metrics_payload, sigmoid
from src.utils import (
    build_logger,
    choose_device,
    environment_snapshot,
    prepare_run_directory,
    round_nested,
    set_global_seed,
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


def _evaluate_model(
    model: Any,
    loader: Any,
    criterion: Any,
    device: str,
    threshold: float,
    ks: list[int],
    ef_fractions: list[float],
    max_batches: int | None = None,
) -> dict[str, Any]:
    torch = try_import_torch()
    model.eval()

    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    target_ids: list[str] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = _move_batch_to_device(batch, device)
            outputs = model(batch)
            logits = outputs["logits"]
            labels = batch["labels"]
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
    scores_np = sigmoid(logits_np)
    avg_loss = total_loss / max(n_batches, 1)
    return build_metrics_payload(
        labels=labels_np,
        scores=scores_np,
        group_ids=target_ids,
        threshold=threshold,
        ks=ks,
        ef_fractions=ef_fractions,
        loss=avg_loss,
    )


def _save_checkpoint(
    torch_module: Any,
    checkpoint_path: Path,
    model: Any,
    optimizer: Any,
    epoch_index: int,
    best_metric: float,
    config: dict[str, Any],
    val_metrics: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch_index,
        "best_metric": best_metric,
        "config": config,
        "val_metrics": val_metrics,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(checkpoint, checkpoint_path)


def _load_checkpoint(torch_module: Any, checkpoint_path: Path, model: Any, optimizer: Any | None = None) -> dict[str, Any]:
    checkpoint = torch_module.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def run_training(config: dict[str, Any], dry_run: bool, max_steps: int | None = None) -> Path:
    output_cfg = config["output"]
    training_cfg = config["training"]

    run_dir, actual_run_name = prepare_run_directory(
        root_dir=output_cfg["root_dir"],
        run_name=config["run_name"],
        allow_rerun_suffix=bool(output_cfg["allow_rerun_suffix"]),
        resume=bool(training_cfg["resume"]),
    )
    config["run_name"] = actual_run_name
    config["output"]["run_dir"] = str(run_dir)
    config["output"]["checkpoint_dir"] = str(run_dir / "checkpoints")

    logger = build_logger(run_dir / "train.log", logger_name=f"train:{actual_run_name}")
    save_config(config, run_dir / "config.resolved.yaml")

    split_summary = describe_splits(config["data"])
    graph_cache_summary = describe_graph_cache(config["data"])
    logger.info("Resolved config from %s", config["config_path"])
    logger.info("Run directory: %s", run_dir)
    logger.info("Split summary: %s", round_nested(split_summary))
    logger.info("Graph cache: %s", round_nested(graph_cache_summary))

    if dry_run:
        payload = {
            "status": "dry_run",
            "mode": config["mode"],
            "run_name": config["run_name"],
            "split_summary": round_nested(split_summary),
            "graph_cache": round_nested(graph_cache_summary),
            "environment": environment_snapshot(),
        }
        write_json(payload, run_dir / "metrics.json")
        logger.info("Dry-run completed successfully.")
        return run_dir

    torch = try_import_torch()
    from src.model.dti_model import build_model

    device = choose_device(str(training_cfg["device"]), torch)
    set_global_seed(int(config["seed"]), torch_module=torch)

    model = build_model(config)
    model = model.to(device)

    loaders, loader_metadata = build_dataloaders(config)
    logger.info("Loader metadata: %s", round_nested(loader_metadata))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(training_cfg["amp"]) and device.startswith("cuda"))

    best_ckpt = run_dir / "checkpoints" / "best.pt"
    legacy_latest_ckpt = run_dir / "checkpoints" / "latest.pt"
    start_epoch = 0
    best_metric = float("-inf")
    best_state_payload: dict[str, Any] | None = None
    epochs_without_improvement = 0

    if bool(training_cfg["resume"]):
        resume_checkpoint = best_ckpt if best_ckpt.exists() else legacy_latest_ckpt if legacy_latest_ckpt.exists() else None
        if resume_checkpoint is not None:
            checkpoint = _load_checkpoint(torch, resume_checkpoint, model, optimizer)
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_metric = float(checkpoint.get("best_metric", float("-inf")))
            best_val_metrics = checkpoint.get("val_metrics")
            if isinstance(best_val_metrics, dict):
                best_state_payload = {
                    "epoch": int(checkpoint.get("epoch", start_epoch - 1)),
                    "val": best_val_metrics,
                }
            if resume_checkpoint == legacy_latest_ckpt and not best_ckpt.exists():
                legacy_latest_ckpt.replace(best_ckpt)
                logger.info("Migrated legacy checkpoint %s to %s", legacy_latest_ckpt, best_ckpt)
            logger.info("Resumed from checkpoint %s at epoch %s", resume_checkpoint, start_epoch)
        else:
            logger.info("Resume requested, but no checkpoint was found under %s", run_dir / "checkpoints")

    history: list[dict[str, Any]] = []
    max_train_batches = (
        int(training_cfg["max_train_batches"]) if training_cfg["max_train_batches"] is not None else max_steps
    )
    max_eval_batches = int(training_cfg["max_eval_batches"]) if training_cfg["max_eval_batches"] is not None else None
    primary_metric = str(config["metrics"]["primary"])

    for epoch_index in range(start_epoch, int(training_cfg["epochs"])):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_index, batch in enumerate(loaders["train"]):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(training_cfg["amp"]) and device.startswith("cuda")):
                outputs = model(batch)
                loss = criterion(outputs["logits"], batch["labels"])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            batch_count += 1

        if batch_count == 0:
            raise RuntimeError("Training loader produced zero batches.")

        train_loss = epoch_loss / batch_count
        val_metrics = _evaluate_model(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            threshold=float(config["metrics"]["threshold"]),
            ks=[int(item) for item in config["metrics"]["ks"]],
            ef_fractions=[float(item) for item in config["metrics"]["ef_fractions"]],
            max_batches=max_eval_batches,
        )

        metric_value = val_metrics["classification"].get(primary_metric)
        metric_score = float("-inf") if metric_value is None else float(metric_value)
        history.append(
            {
                "epoch": epoch_index,
                "train_loss": train_loss,
                "val": val_metrics,
            }
        )
        logger.info(
            "Epoch %s | train_loss=%.4f | val_%s=%s",
            epoch_index,
            train_loss,
            primary_metric,
            "nan" if metric_value is None else f"{metric_value:.4f}",
        )

        should_update_best = metric_score > best_metric or not best_ckpt.exists()
        if should_update_best:
            best_metric = max(best_metric, metric_score)
            epochs_without_improvement = 0
            _save_checkpoint(
                torch_module=torch,
                checkpoint_path=best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch_index=epoch_index,
                best_metric=best_metric,
                config=config,
                val_metrics=val_metrics,
            )
            best_state_payload = {"epoch": epoch_index, "val": val_metrics}
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(training_cfg["early_stopping_patience"]):
            logger.info("Early stopping triggered after %s epochs without improvement.", epochs_without_improvement)
            break

    if best_ckpt.exists():
        _load_checkpoint(torch, best_ckpt, model)
    if legacy_latest_ckpt.exists() and best_ckpt.exists():
        legacy_latest_ckpt.unlink()
        logger.info("Removed stale legacy checkpoint %s", legacy_latest_ckpt)

    test_metrics = _evaluate_model(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
        threshold=float(config["metrics"]["threshold"]),
        ks=[int(item) for item in config["metrics"]["ks"]],
        ef_fractions=[float(item) for item in config["metrics"]["ef_fractions"]],
        max_batches=max_eval_batches,
    )

    payload = {
        "status": "completed",
        "mode": config["mode"],
        "run_name": config["run_name"],
        "environment": environment_snapshot(),
        "data": loader_metadata["summary"],
        "best": best_state_payload,
        "test": test_metrics,
        "history": history,
    }
    write_json(round_nested(payload), run_dir / "metrics.json")
    logger.info("Training finished. Metrics written to %s", run_dir / "metrics.json")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a cold-protein DTI model from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the experiment config file.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--set", action="append", default=[], help="Override a config value with key=value.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and outputs without importing torch.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional training batch limit for smoke tests.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config = load_and_resolve_config(
            config_path=args.config,
            seed=args.seed,
            mode="train",
            overrides=args.set,
        )
        run_training(config=config, dry_run=bool(args.dry_run), max_steps=args.max_steps)
    except ConfigError as exc:
        raise SystemExit(f"[config-error] {exc}") from exc


if __name__ == "__main__":
    main()
