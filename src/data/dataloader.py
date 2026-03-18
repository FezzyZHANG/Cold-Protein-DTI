"""Split resolution and DataLoader construction helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from src.data.mol_graph import default_graph_cache_path, default_split_graph_cache_path, load_graph_store


def read_split_table(path: str | Path) -> pl.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Split file does not exist: {table_path}")

    if table_path.suffix.lower() == ".csv":
        return pl.read_csv(table_path)
    if table_path.suffix.lower() == ".parquet":
        return pl.read_parquet(table_path)
    raise ValueError(f"Unsupported split file format: {table_path.suffix}")


def resolve_split_paths(data_cfg: dict[str, Any]) -> dict[str, Path]:
    return {
        "train": Path(data_cfg["train_path"]),
        "val": Path(data_cfg["val_path"]),
        "test": Path(data_cfg["test_path"]),
    }


def resolve_graph_cache_path(data_cfg: dict[str, Any]) -> Path | None:
    if data_cfg.get("graph_cache_path"):
        return Path(data_cfg["graph_cache_path"])
    if data_cfg.get("split_dir"):
        split_cache_path = default_split_graph_cache_path(data_cfg["split_dir"])
        if split_cache_path.exists():
            return split_cache_path
    if data_cfg.get("raw_path"):
        return default_graph_cache_path(data_cfg["raw_path"])
    if data_cfg.get("split_dir"):
        return default_split_graph_cache_path(data_cfg["split_dir"])
    return None


def describe_splits(data_cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for split_name, path in resolve_split_paths(data_cfg).items():
        frame = read_split_table(path)
        summary[split_name] = {
            "n_rows": float(frame.height),
            "n_drugs": float(frame.select("inchi_key").n_unique()),
            "n_proteins": float(frame.select("target_uniprot_id").n_unique()),
            "positive_ratio": 0.0 if frame.height == 0 else float(frame["label"].mean()),
        }
    return summary


def describe_graph_cache(data_cfg: dict[str, Any]) -> dict[str, Any]:
    cache_path = resolve_graph_cache_path(data_cfg)
    if cache_path is None:
        return {"path": None, "exists": False}
    summary: dict[str, Any] = {"path": str(cache_path), "exists": cache_path.exists()}
    if cache_path.exists():
        manifest_path = cache_path.with_suffix(".json")
        if manifest_path.exists():
            summary["metadata"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return summary


def build_dataloaders(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from torch.utils.data import DataLoader

    from src.data.dataset import DTIDataset, collate_dti_batch

    data_cfg = config["data"]
    training_cfg = config["training"]
    split_paths = resolve_split_paths(data_cfg)
    graph_cache_path = resolve_graph_cache_path(data_cfg)
    if graph_cache_path is None:
        raise FileNotFoundError(
            "A graph cache path is required for the GVP drug encoder. "
            "Set `data.graph_cache_path`, provide `data.raw_path`, or build a split-local cache under `data.split_dir`."
        )
    graph_store, graph_metadata = load_graph_store(graph_cache_path)
    frames = {name: read_split_table(path) for name, path in split_paths.items()}

    datasets = {
        name: DTIDataset(
            frame=frame,
            graph_store=graph_store,
            max_protein_length=int(data_cfg["max_protein_length"]),
        )
        for name, frame in frames.items()
    }

    common_loader_kwargs = {
        "batch_size": int(training_cfg["batch_size"]),
        "num_workers": int(training_cfg["num_workers"]),
        "collate_fn": collate_dti_batch,
        "pin_memory": str(training_cfg["device"]).startswith("cuda"),
    }

    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, **common_loader_kwargs),
        "val": DataLoader(datasets["val"], shuffle=False, **common_loader_kwargs),
        "test": DataLoader(datasets["test"], shuffle=False, **common_loader_kwargs),
    }

    metadata = {
        "paths": {name: str(path) for name, path in split_paths.items()},
        "summary": describe_splits(data_cfg),
        "graph_cache": {
            "path": str(graph_cache_path),
            "num_graphs": len(graph_store),
            "metadata": graph_metadata,
        },
    }
    return loaders, metadata
