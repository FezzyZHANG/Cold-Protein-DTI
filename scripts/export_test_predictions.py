#!/usr/bin/env python
"""Export per-row test-set predictions for trained runs under a results root."""

from __future__ import annotations

import argparse
from copy import deepcopy
import logging
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from src.config import load_config, resolve_config
from src.data.dataloader import _filter_frame_by_graph_availability, read_split_table, resolve_graph_cache_path
from src.data.mol_graph import load_graph_store
from src.utils import choose_device, set_global_seed, try_import_torch


LOGGER = logging.getLogger("export-test-predictions")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _load_checkpoint(torch_module: Any, checkpoint_path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch_module.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch_module.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint payload type at {checkpoint_path}: {type(checkpoint)!r}")
    return checkpoint


def _discover_best_checkpoints(results_root: Path) -> list[Path]:
    checkpoints = [
        path
        for path in results_root.rglob("best.pt")
        if path.parent.name == "checkpoints" and path.is_file()
    ]
    return sorted(checkpoints)


def _load_run_config(run_dir: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    config_path = run_dir / "config.resolved.yaml"
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        if not config_path.exists():
            raise FileNotFoundError(
                f"Run config was not embedded in the checkpoint and {config_path} is missing."
            )
        config = load_config(config_path)

    seed = config.get("seed")
    resolved = resolve_config(
        deepcopy(config),
        config_path=config_path,
        cli_mode=str(config.get("mode") or "train"),
        cli_seed=int(seed) if seed is not None else None,
    )
    resolved.setdefault("output", {})
    resolved["run_name"] = str(resolved.get("run_name") or run_dir.name)
    resolved["output"]["root_dir"] = str(run_dir.parent)
    resolved["output"]["run_dir"] = str(run_dir)
    resolved["output"]["checkpoint_dir"] = str(run_dir / "checkpoints")
    return resolved


def _build_test_loader(
    *,
    config: dict[str, Any],
    batch_size_override: int | None,
    num_workers_override: int | None,
    device: str,
) -> tuple[Any, pl.DataFrame, dict[str, Any]]:
    from torch.utils.data import DataLoader

    from src.data.dataset import DTIDataset, collate_dti_batch

    graph_cache_path = resolve_graph_cache_path(config["data"])
    if graph_cache_path is None:
        raise FileNotFoundError(
            "A graph cache path is required for test export. "
            "Set `data.graph_cache_path`, provide `data.raw_path`, or stage a split-local cache."
        )

    graph_store, graph_metadata = load_graph_store(graph_cache_path)
    raw_test_frame = read_split_table(config["data"]["test_path"]).with_row_index("__export_row_id")
    filtered_test_frame, filter_stats = _filter_frame_by_graph_availability(raw_test_frame, graph_store)
    if filtered_test_frame.height == 0:
        raise RuntimeError("Test split has zero usable rows after graph filtering.")

    dataset = DTIDataset(
        frame=filtered_test_frame,
        graph_store=graph_store,
        max_protein_length=int(config["data"]["max_protein_length"]),
    )
    batch_size = int(batch_size_override or config["training"]["batch_size"])
    num_workers = int(
        config["training"]["num_workers"] if num_workers_override is None else num_workers_override
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_dti_batch,
        pin_memory=device.startswith("cuda"),
    )

    metadata = {
        "rows_total": int(raw_test_frame.height),
        "rows_usable": int(filtered_test_frame.height),
        "rows_dropped": int(raw_test_frame.height - filtered_test_frame.height),
        "graph_cache_path": str(graph_cache_path),
        "graph_cache_metadata": graph_metadata,
        "filter_stats": filter_stats,
    }
    return loader, filtered_test_frame, metadata


def _predict_test_rows(
    *,
    model: Any,
    loader: Any,
    device: str,
    torch_module: Any,
) -> tuple[list[float], list[float]]:
    logits_values: list[float] = []
    sigmoid_values: list[float] = []

    model.eval()
    with torch_module.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(batch)
            logits = outputs["logits"].detach().cpu().reshape(-1)
            sigmoid_scores = torch_module.sigmoid(logits)

            logits_values.extend(float(item) for item in logits.tolist())
            sigmoid_values.extend(float(item) for item in sigmoid_scores.tolist())

    return logits_values, sigmoid_values


def export_run_predictions(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    output_name: str,
    device_override: str | None,
    batch_size_override: int | None,
    num_workers_override: int | None,
) -> Path:
    torch = try_import_torch()
    checkpoint = _load_checkpoint(torch, checkpoint_path)
    config = _load_run_config(run_dir, checkpoint)
    requested_device = str(device_override or config["training"]["device"])
    device = choose_device(requested_device, torch)

    if "seed" in config and config["seed"] is not None:
        set_global_seed(int(config["seed"]), torch_module=torch)

    from src.model.dti_model import build_model

    model = build_model(config).to(device)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint weights are incompatible with the current model code. "
            "This usually means the run was trained with an older architecture revision. "
            f"checkpoint={checkpoint_path}"
        ) from exc

    loader, filtered_test_frame, metadata = _build_test_loader(
        config=config,
        batch_size_override=batch_size_override,
        num_workers_override=num_workers_override,
        device=device,
    )

    logits_values, sigmoid_values = _predict_test_rows(
        model=model,
        loader=loader,
        device=device,
        torch_module=torch,
    )

    if len(logits_values) != filtered_test_frame.height:
        raise RuntimeError(
            "Prediction count does not match the filtered test split size. "
            f"predictions={len(logits_values)}, rows={filtered_test_frame.height}"
        )

    prediction_frame = pl.DataFrame(
        {
            "__export_row_id": filtered_test_frame["__export_row_id"].to_list(),
            "pred_logit": logits_values,
            "pred_sigmoid": sigmoid_values,
        }
    )
    export_frame = (
        read_split_table(config["data"]["test_path"])
        .with_row_index("__export_row_id")
        .join(prediction_frame, on="__export_row_id", how="left")
        .with_columns(pl.col("pred_sigmoid").is_not_null().alias("prediction_available"))
        .sort(
            by=["prediction_available", "pred_sigmoid", "target_uniprot_id", "inchi_key"],
            descending=[True, True, False, False],
            nulls_last=True,
        )
        .drop("__export_row_id")
    )

    output_path = run_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_frame.write_csv(output_path)

    LOGGER.info(
        "Exported %s | device=%s | total_rows=%s | predicted_rows=%s | dropped_rows=%s | output=%s",
        run_dir.name,
        device,
        metadata["rows_total"],
        metadata["rows_usable"],
        metadata["rows_dropped"],
        output_path,
    )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a results root for checkpoints/best.pt and export per-row test-set predictions "
            "into a CSV under each run directory."
        )
    )
    parser.add_argument("--results-root", required=True, help="Results root to scan recursively.")
    parser.add_argument(
        "--output-name",
        default="test_inference.csv",
        help="CSV filename to write inside each run directory.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, for example `cpu`, `cuda`, or `cuda:0`.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch-size override for export inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional dataloader worker override for export inference.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs where the output CSV already exists.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one run fails.",
    )
    return parser


def main() -> None:
    _configure_logging()
    args = build_arg_parser().parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise SystemExit(f"Results root does not exist: {results_root}")

    checkpoints = _discover_best_checkpoints(results_root)
    if not checkpoints:
        raise SystemExit(f"No checkpoints/best.pt files were found under {results_root}")

    LOGGER.info("Discovered %s run(s) with best checkpoints under %s", len(checkpoints), results_root)

    exported = 0
    skipped = 0
    failures: list[tuple[Path, str]] = []

    for checkpoint_path in checkpoints:
        run_dir = checkpoint_path.parent.parent
        output_path = run_dir / args.output_name
        if args.skip_existing and output_path.exists():
            skipped += 1
            LOGGER.info("Skipping %s because %s already exists.", run_dir.name, output_path)
            continue

        try:
            export_run_predictions(
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                output_name=args.output_name,
                device_override=args.device,
                batch_size_override=args.batch_size,
                num_workers_override=args.num_workers,
            )
            exported += 1
        except Exception as exc:  # pragma: no cover - exercised through CLI usage
            failures.append((run_dir, str(exc)))
            LOGGER.exception("Failed to export predictions for %s", run_dir)
            if args.fail_fast:
                break

    LOGGER.info(
        "Finished export scan | discovered=%s | exported=%s | skipped=%s | failed=%s",
        len(checkpoints),
        exported,
        skipped,
        len(failures),
    )

    if failures:
        for run_dir, message in failures:
            LOGGER.error("Failure summary | run=%s | error=%s", run_dir, message)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
