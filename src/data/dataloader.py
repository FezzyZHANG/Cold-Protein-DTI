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


def _summarize_frame(frame: pl.DataFrame) -> dict[str, float]:
    return {
        "n_rows": float(frame.height),
        "n_drugs": float(frame.select("inchi_key").n_unique()),
        "n_proteins": float(frame.select("target_uniprot_id").n_unique()),
        "positive_ratio": 0.0 if frame.height == 0 else float(frame["label"].mean()),
    }


def describe_splits(data_cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for split_name, path in resolve_split_paths(data_cfg).items():
        frame = read_split_table(path)
        summary[split_name] = _summarize_frame(frame)
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


def _is_non_empty_graph(graph: Any) -> bool:
    if graph is None:
        return False
    node_scalar = getattr(graph, "node_s", None)
    node_vector = getattr(graph, "node_v", None)
    edge_index = getattr(graph, "edge_index", None)
    edge_scalar = getattr(graph, "edge_s", None)
    edge_vector = getattr(graph, "edge_v", None)
    if node_scalar is None or node_vector is None or edge_index is None or edge_scalar is None or edge_vector is None:
        return False
    if getattr(node_scalar, "ndim", None) != 2 or int(node_scalar.shape[0]) == 0:
        return False
    if getattr(node_vector, "ndim", None) != 3 or int(node_vector.shape[0]) == 0:
        return False
    if getattr(edge_index, "ndim", None) != 2 or int(edge_index.shape[0]) != 2 or int(edge_index.shape[1]) == 0:
        return False
    if getattr(edge_scalar, "ndim", None) != 2 or int(edge_scalar.shape[0]) == 0:
        return False
    if getattr(edge_vector, "ndim", None) != 3 or int(edge_vector.shape[0]) == 0:
        return False
    num_nodes = getattr(graph, "num_nodes", None)
    if num_nodes is None:
        num_nodes = int(node_scalar.shape[0])
    return int(num_nodes) > 0


def _filter_frame_by_graph_availability(
    frame: pl.DataFrame,
    graph_store: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    normalized = frame.with_columns(pl.col("inchi_key").cast(pl.Utf8))
    unique_keys = normalized["inchi_key"].unique().to_list()

    valid_keys: list[str] = []
    missing_keys: list[str] = []
    empty_graph_keys: list[str] = []
    for inchi_key in unique_keys:
        if inchi_key is None:
            missing_keys.append("<null>")
            continue
        graph = graph_store.get(inchi_key)
        if graph is None:
            missing_keys.append(str(inchi_key))
            continue
        if not _is_non_empty_graph(graph):
            empty_graph_keys.append(str(inchi_key))
            continue
        valid_keys.append(str(inchi_key))

    filtered = normalized.filter(pl.col("inchi_key").is_in(valid_keys))
    stats = {
        "rows_before": float(normalized.height),
        "rows_after": float(filtered.height),
        "dropped_rows": float(normalized.height - filtered.height),
        "missing_graph_keys": float(len(missing_keys)),
        "empty_graph_keys": float(len(empty_graph_keys)),
    }
    if missing_keys:
        stats["missing_graph_examples"] = missing_keys[:5]
    if empty_graph_keys:
        stats["empty_graph_examples"] = empty_graph_keys[:5]
    return filtered, stats


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
    filtered_frames: dict[str, pl.DataFrame] = {}
    filtering_summary: dict[str, dict[str, Any]] = {}
    for name, frame in frames.items():
        filtered_frame, filter_stats = _filter_frame_by_graph_availability(frame, graph_store)
        if filtered_frame.height == 0:
            raise RuntimeError(
                f"Split `{name}` has zero usable rows after graph filtering. "
                "Check graph cache coverage and molecule preprocessing outputs."
            )
        filtered_frames[name] = filtered_frame
        filtering_summary[name] = filter_stats

    datasets = {
        name: DTIDataset(
            frame=filtered_frames[name],
            graph_store=graph_store,
            max_protein_length=int(data_cfg["max_protein_length"]),
        )
        for name in filtered_frames
    }
    batch_size = int(training_cfg["batch_size"])
    undersized_splits = [name for name, dataset in datasets.items() if len(dataset) < batch_size]
    if undersized_splits:
        raise RuntimeError(
            "drop_last=True requires every split to contain at least one full batch. "
            f"batch_size={batch_size}, undersized_splits={undersized_splits}."
        )

    common_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": int(training_cfg["num_workers"]),
        "collate_fn": collate_dti_batch,
        "pin_memory": str(training_cfg["device"]).startswith("cuda"),
    }

    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, drop_last=True, **common_loader_kwargs),
        "val": DataLoader(datasets["val"], shuffle=False, drop_last=True, **common_loader_kwargs),
        "test": DataLoader(datasets["test"], shuffle=False, drop_last=True, **common_loader_kwargs),
    }

    metadata = {
        "paths": {name: str(path) for name, path in split_paths.items()},
        "summary": {name: _summarize_frame(frame) for name, frame in filtered_frames.items()},
        "filtering": filtering_summary,
        "graph_cache": {
            "path": str(graph_cache_path),
            "num_graphs": len(graph_store),
            "metadata": graph_metadata,
        },
    }
    return loaders, metadata
