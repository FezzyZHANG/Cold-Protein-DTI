"""Split resolution and DataLoader construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


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


def build_dataloaders(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from torch.utils.data import DataLoader

    from src.data.dataset import DTIDataset, collate_dti_batch

    data_cfg = config["data"]
    training_cfg = config["training"]
    split_paths = resolve_split_paths(data_cfg)
    frames = {name: read_split_table(path) for name, path in split_paths.items()}

    datasets = {
        name: DTIDataset(
            frame=frame,
            max_smiles_length=int(data_cfg["max_smiles_length"]),
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
    }
    return loaders, metadata
