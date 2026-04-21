"""Prepare and run Scope-DTI benchmarks on the CP-easy/CP-hard splits.

Upstream Scope-DTI source and generated benchmark inputs live under
``artifacts/``; run logs, checkpoints, and metrics live under ``runs/``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from src.config import apply_overrides, load_config, save_config
from src.utils import (
    build_logger,
    environment_snapshot,
    prepare_run_directory,
    round_nested,
    write_json,
)


SCOPE_REQUIRED_MODULES = [
    "dgl",
    "dgllife",
    "pandas",
    "prettytable",
    "pyarrow",
    "rdkit",
    "sklearn",
    "tensorboard",
    "torch",
    "torch_cluster",
    "torch_geometric",
    "tqdm",
    "yacs",
]


class ScopeInputError(ValueError):
    """Raised when Scope-DTI benchmark input preparation cannot continue."""


def _resolve_path(value: str | Path | None, *, base: Path = PROJECT_ROOT) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else base / path


def _path_for_yaml(path: Path) -> str:
    return path.resolve().as_posix()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_split_table(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split file does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    raise ValueError(f"Unsupported split file format for Scope-DTI export: {path.suffix}")


def _input_paths(split_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        candidate = split_dir / f"{split}.parquet"
        if not candidate.exists():
            candidate = split_dir / f"{split}.csv"
        paths[split] = candidate
    return paths


def _frame_summary(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": int(frame.height),
        "n_drugs": int(frame.select("drug_chembl_id").n_unique()) if frame.height else 0,
        "n_proteins": int(frame.select("target_uniprot_id").n_unique()) if frame.height else 0,
        "positive_ratio": None if frame.height == 0 else float(frame["label"].mean()),
    }


def _conflict_summary(frame: pl.DataFrame, key: str, value: str) -> dict[str, Any]:
    conflicts = (
        frame.select([key, value])
        .unique()
        .group_by(key)
        .agg(pl.col(value).n_unique().alias("n_values"))
        .filter(pl.col("n_values") > 1)
    )
    return {
        "count": int(conflicts.height),
        "examples": conflicts.head(10).select(key).to_series().to_list() if conflicts.height else [],
    }


def _format_scope_frame(frame: pl.DataFrame, split_name: str, data_cfg: dict[str, Any]) -> pl.DataFrame:
    drug_id_col = str(data_cfg.get("drug_id_column", "inchi_key"))
    protein_id_col = str(data_cfg.get("protein_id_column", "target_uniprot_id"))
    smiles_col = str(data_cfg.get("smiles_column", "smiles"))
    sequence_col = str(data_cfg.get("sequence_column", "sequence"))
    label_col = str(data_cfg.get("label_column", "label"))
    required = {drug_id_col, protein_id_col, smiles_col, sequence_col, label_col}
    missing = required - set(frame.columns)
    if missing:
        raise ScopeInputError(f"Split `{split_name}` is missing required columns: {sorted(missing)}")

    return frame.with_columns(
        [
            pl.col(drug_id_col).cast(pl.Utf8).alias("drug_chembl_id"),
            pl.col(drug_id_col).cast(pl.Utf8).alias("source_drug_id"),
            pl.col(protein_id_col).cast(pl.Utf8).alias("target_uniprot_id"),
            pl.col(smiles_col).cast(pl.Utf8).alias("smiles"),
            pl.col(sequence_col).cast(pl.Utf8).alias("sequence"),
            pl.col(label_col).cast(pl.Int64).alias("label"),
            pl.lit(split_name).alias("split"),
        ]
    )



def _stable_smiles_hash(smiles: str) -> str:
    return hashlib.blake2s(smiles.encode("utf-8"), digest_size=5).hexdigest()


def _disambiguate_conflicting_drug_ids(frames: dict[str, pl.DataFrame]) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    all_frame = pl.concat(frames.values(), how="vertical")
    conflict_ids = (
        all_frame.select(["source_drug_id", "smiles"])
        .unique()
        .group_by("source_drug_id")
        .agg(pl.col("smiles").n_unique().alias("n_smiles"))
        .filter(pl.col("n_smiles") > 1)
        .select("source_drug_id")
        .to_series()
        .to_list()
    )
    if not conflict_ids:
        return frames, {"count": 0, "examples": []}

    conflict_set = set(str(item) for item in conflict_ids)
    updated: dict[str, pl.DataFrame] = {}
    for split, frame in frames.items():
        updated[split] = frame.with_columns(
            pl.when(pl.col("source_drug_id").is_in(conflict_set))
            .then(
                pl.concat_str(
                    [
                        pl.col("source_drug_id"),
                        pl.lit("__smi"),
                        pl.col("smiles").map_elements(_stable_smiles_hash, return_dtype=pl.Utf8),
                    ]
                )
            )
            .otherwise(pl.col("source_drug_id"))
            .alias("drug_chembl_id")
        )
    return updated, {"count": len(conflict_set), "examples": sorted(conflict_set)[:20]}


def _molblock_from_smiles(smiles: str, *, seed: int) -> str:
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except ImportError as exc:
        raise RuntimeError("RDKit is required to generate Scope-DTI SDF/MolBlock values.") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES.")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise ValueError("RDKit conformer embedding failed.")

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=200)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        # The conformer is still usable for Scope-DTI graph construction.
        pass

    block = Chem.MolToMolBlock(mol)
    if not block.strip():
        raise ValueError("RDKit produced an empty MolBlock.")
    return block


def _build_sdf_cache(
    drugs: pl.DataFrame,
    cache_path: Path,
    *,
    force: bool,
    seed: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        existing = pl.read_parquet(cache_path)
        expected = {"drug_chembl_id", "smiles", "sdf"}
        if expected.issubset(existing.columns):
            existing = existing.select(["drug_chembl_id", "smiles", "sdf"]).unique(
                subset=["drug_chembl_id"], keep="first"
            )
        else:
            existing = pl.DataFrame(schema={"drug_chembl_id": pl.Utf8, "smiles": pl.Utf8, "sdf": pl.Utf8})
    else:
        existing = pl.DataFrame(schema={"drug_chembl_id": pl.Utf8, "smiles": pl.Utf8, "sdf": pl.Utf8})

    existing_ids = set(existing.get_column("drug_chembl_id").to_list()) if existing.height else set()
    missing_drugs = drugs.filter(~pl.col("drug_chembl_id").is_in(existing_ids))
    generated_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    for row in missing_drugs.iter_rows(named=True):
        drug_id = str(row["drug_chembl_id"])
        smiles = str(row["smiles"])
        try:
            generated_rows.append(
                {
                    "drug_chembl_id": drug_id,
                    "smiles": smiles,
                    "sdf": _molblock_from_smiles(smiles, seed=seed),
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-compound failures for manifest/debugging.
            failures.append({"drug_chembl_id": drug_id, "smiles": smiles, "reason": str(exc)})

    generated = (
        pl.DataFrame(generated_rows, schema={"drug_chembl_id": pl.Utf8, "smiles": pl.Utf8, "sdf": pl.Utf8})
        if generated_rows
        else pl.DataFrame(schema={"drug_chembl_id": pl.Utf8, "smiles": pl.Utf8, "sdf": pl.Utf8})
    )
    cache = pl.concat([existing, generated], how="vertical").unique(subset=["drug_chembl_id"], keep="first")
    cache.write_parquet(cache_path)
    summary = {
        "cache_path": str(cache_path),
        "n_required_drugs": int(drugs.height),
        "n_existing": int(existing.height),
        "n_generated": len(generated_rows),
        "n_failed": len(failures),
        "failure_examples": failures[:20],
    }
    return cache, summary


def _load_coord_index(coord_path: Path | None) -> tuple[set[str], dict[str, Any]]:
    if coord_path is None:
        return set(), {"path": None, "exists": False}
    if not coord_path.exists():
        return set(), {"path": str(coord_path), "exists": False}

    try:
        with coord_path.open("rb") as handle:
            coords_df = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001 - file may come from upstream pandas pickle.
        raise ScopeInputError(f"Could not read Scope-DTI coordinate PKL at {coord_path}: {exc}") from exc

    index_values = [str(item) for item in getattr(coords_df, "index", [])]
    columns = [str(item) for item in getattr(coords_df, "columns", [])]
    if not index_values or "sequence" not in columns or "crod" not in columns:
        raise ScopeInputError(
            "Scope-DTI coordinate PKL must be a DataFrame indexed by UniProt ID with `sequence` and `crod` columns."
        )
    return set(index_values), {
        "path": str(coord_path),
        "exists": True,
        "n_proteins": len(index_values),
        "columns": columns,
    }


def _write_missing_proteins(
    frames: dict[str, pl.DataFrame],
    coord_index: set[str],
    output_path: Path,
) -> dict[str, Any]:
    missing_rows: list[dict[str, Any]] = []
    by_split: dict[str, Any] = {}
    for split, frame in frames.items():
        missing = frame.filter(~pl.col("target_uniprot_id").is_in(coord_index))
        protein_ids = sorted(missing.select("target_uniprot_id").unique().to_series().to_list())
        by_split[split] = {
            "n_rows": int(missing.height),
            "n_proteins": len(protein_ids),
            "examples": protein_ids[:20],
        }
        for row in missing.select(["target_uniprot_id", "sequence"]).unique().iter_rows(named=True):
            missing_rows.append(
                {
                    "split": split,
                    "target_uniprot_id": str(row["target_uniprot_id"]),
                    "sequence_length": len(str(row["sequence"])),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if missing_rows:
        pl.DataFrame(missing_rows).write_csv(output_path, separator="\t")
    elif output_path.exists():
        output_path.unlink()
    return {
        "missing_path": str(output_path) if missing_rows else None,
        "by_split": by_split,
        "total_missing_rows": int(sum(item["n_rows"] for item in by_split.values())),
        "total_missing_proteins": len({row["target_uniprot_id"] for row in missing_rows}),
    }


def _safe_write_scope_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def _write_asset_readme(artifact_dir: Path, split_name: str) -> None:
    readme_path = artifact_dir / "README.md"
    if readme_path.exists():
        return
    readme_path.write_text(
        "\n".join(
            [
                f"# Scope-DTI {split_name} Benchmark Inputs",
                "",
                "Generated from this repository's cold-protein split parquet files.",
                "",
                "Files here:",
                "- `train.parquet`, `val.parquet`, `test.parquet`: Scope-DTI rows with embedded `sdf` MolBlocks.",
                "- `drug_sdf_cache.parquet`: per-compound generated SDF/MolBlock cache.",
                "- `manifest.json`: source paths, coverage checks, and split summaries.",
                "- `missing_coord_proteins.tsv`: written only when graph-mode coordinate coverage is incomplete.",
                "",
                "Scope-DTI graph mode also requires the upstream source tree and `unip_cords.pkl`,",
                "usually staged under `artifacts/ScopeDTI/source`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def prepare_scopedti_inputs(
    config: dict[str, Any],
    *,
    force_sdf: bool = False,
    allow_missing_coords: bool | None = None,
) -> dict[str, Any]:
    data_cfg = config["data"]
    export_cfg = config.get("export", {})
    scopedti_cfg = config.get("scopedti", {})
    split_name = str(data_cfg["split_name"])
    split_dir = _resolve_path(data_cfg["split_dir"])
    if split_dir is None:
        raise ScopeInputError("data.split_dir is required.")

    artifact_dir = _resolve_path(scopedti_cfg.get("artifact_dir") or export_cfg.get("artifact_dir"))
    if artifact_dir is None:
        artifact_dir = PROJECT_ROOT / "artifacts" / "ScopeDTI" / "benchmarks" / split_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    input_paths = _input_paths(split_dir)
    frames: dict[str, pl.DataFrame] = {
        split: _format_scope_frame(_read_split_table(path), split, data_cfg)
        for split, path in input_paths.items()
    }
    frames, drug_id_disambiguation = _disambiguate_conflicting_drug_ids(frames)
    all_frame = pl.concat(frames.values(), how="vertical")
    drugs = all_frame.select(["drug_chembl_id", "smiles"]).unique(subset=["drug_chembl_id"], keep="first").sort(
        "drug_chembl_id"
    )
    sdf_cache_path = _resolve_path(scopedti_cfg.get("sdf_cache_path")) or (artifact_dir / "drug_sdf_cache.parquet")
    cache, sdf_summary = _build_sdf_cache(
        drugs,
        sdf_cache_path,
        force=bool(force_sdf or export_cfg.get("force_sdf", False)),
        seed=int(config.get("seed", 13)),
    )
    if sdf_summary["n_failed"] > 0:
        raise ScopeInputError(f"Failed to generate Scope-DTI SDF/MolBlock values: {sdf_summary['failure_examples']}")

    graph_mode = str(scopedti_cfg.get("protein_mode", "graph")).lower() == "graph"
    coord_path = _resolve_path(scopedti_cfg.get("coord_path"))
    coord_index: set[str] = set()
    coord_summary: dict[str, Any] = {"path": str(coord_path) if coord_path is not None else None, "exists": False}
    coord_coverage: dict[str, Any] = {}
    should_allow_missing_coords = (
        bool(scopedti_cfg.get("allow_missing_coords", False))
        if allow_missing_coords is None
        else bool(allow_missing_coords)
    )
    if graph_mode:
        coord_index, coord_summary = _load_coord_index(coord_path)
        coord_coverage = _write_missing_proteins(frames, coord_index, artifact_dir / "missing_coord_proteins.tsv")
        if coord_coverage["total_missing_rows"] > 0 and should_allow_missing_coords:
            frames = {
                split: frame.filter(pl.col("target_uniprot_id").is_in(coord_index))
                for split, frame in frames.items()
            }

    written_frames: dict[str, pl.DataFrame] = {}
    for split, frame in frames.items():
        adapted = frame.join(cache.select(["drug_chembl_id", "sdf"]), on="drug_chembl_id", how="left")
        missing_sdf = adapted.filter(pl.col("sdf").is_null()).height
        if missing_sdf:
            raise ScopeInputError(f"Split `{split}` has {missing_sdf} rows without Scope-DTI `sdf` values.")
        output_path = artifact_dir / f"{split}.parquet"
        _safe_write_scope_parquet(adapted, output_path)
        written_frames[split] = adapted

    _write_asset_readme(artifact_dir, split_name)
    manifest = {
        "generated_at": _now(),
        "split_name": split_name,
        "source_split_dir": str(split_dir),
        "artifact_dir": str(artifact_dir),
        "files": {
            "train": str(artifact_dir / "train.parquet"),
            "val": str(artifact_dir / "val.parquet"),
            "test": str(artifact_dir / "test.parquet"),
            "sdf_cache": str(sdf_cache_path),
        },
        "summary": {split: _frame_summary(frame) for split, frame in written_frames.items()},
        "total": _frame_summary(pl.concat(written_frames.values(), how="vertical")),
        "conflicts": {
            "source_drug_smiles": _conflict_summary(all_frame, "source_drug_id", "smiles"),
            "scopedti_drug_id_smiles": _conflict_summary(all_frame, "drug_chembl_id", "smiles"),
            "protein_sequence": _conflict_summary(all_frame, "target_uniprot_id", "sequence"),
        },
        "drug_id_disambiguation": drug_id_disambiguation,
        "sdf": sdf_summary,
        "protein_mode": str(scopedti_cfg.get("protein_mode", "graph")).lower(),
        "coordinate_asset": coord_summary,
        "coordinate_coverage": coord_coverage,
        "allow_missing_coords": should_allow_missing_coords,
        "ok": True,
    }
    if graph_mode and coord_coverage.get("total_missing_rows", 0) > 0 and not should_allow_missing_coords:
        manifest["ok"] = False
        manifest["reason"] = "missing_scope_coordinate_rows"

    write_json(round_nested(manifest), artifact_dir / "manifest.json")
    return manifest


def _missing_python_modules() -> list[str]:
    return [name for name in SCOPE_REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def validate_scopedti_assets(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    scopedti_cfg = config.get("scopedti", {})
    scopedti_home = _resolve_path(scopedti_cfg.get("home", "artifacts/ScopeDTI/source"))
    coord_path = _resolve_path(scopedti_cfg.get("coord_path"))
    graph_path = _resolve_path(scopedti_cfg.get("graph_path"))
    missing_paths: list[dict[str, str]] = []

    required_paths = {
        "scopedti_home": scopedti_home,
        "coord_path": coord_path if str(scopedti_cfg.get("protein_mode", "graph")).lower() == "graph" else None,
    }
    for name, path in required_paths.items():
        if path is not None and not path.exists():
            missing_paths.append({"name": name, "path": str(path)})
    if scopedti_home is not None:
        for relative in ("main.py", "dataloader.py", "models.py", "configs.py"):
            path = scopedti_home / relative
            if not path.exists():
                missing_paths.append({"name": f"scopedti_source:{relative}", "path": str(path)})

    return {
        "ok": not missing_paths and bool(manifest.get("ok", False)),
        "paths": {
            "scopedti_home": str(scopedti_home) if scopedti_home is not None else None,
            "coord_path": str(coord_path) if coord_path is not None else None,
            "graph_path": str(graph_path) if graph_path is not None else None,
        },
        "missing_paths": missing_paths,
        "manifest_ok": bool(manifest.get("ok", False)),
        "manifest_reason": manifest.get("reason"),
    }


def _build_upstream_config(config: dict[str, Any], manifest: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    scopedti_cfg = config.get("scopedti", {})
    training_cfg = config.get("training", {})
    artifact_dir = Path(manifest["artifact_dir"])
    graph_path = _resolve_path(scopedti_cfg.get("graph_path")) or (artifact_dir / "protein_graph.pkl")
    coord_path = _resolve_path(scopedti_cfg.get("coord_path"))
    protein_mode = str(scopedti_cfg.get("protein_mode", "graph")).lower()

    upstream = {
        "DRUG": {
            "ATOM_IN_DIM": [74, 1],
            "ATOM_HIDDEN_DIM": [int(item) for item in scopedti_cfg.get("drug_atom_hidden_dim", [320, 64])],
            "EDGE_IN_DIM": [16, 1],
            "EDGE_HIDDEN_DIM": [32, 1],
            "NUM_LAYERS": int(scopedti_cfg.get("drug_num_layers", 3)),
            "DROP_RATE": float(scopedti_cfg.get("dropout", 0.1)),
            "MAX_NODES": int(scopedti_cfg.get("max_drug_nodes", 300)),
            "EDGE_CUTOFF": float(scopedti_cfg.get("drug_edge_cutoff", 4.5)),
            "NUM_RDF": int(scopedti_cfg.get("drug_num_rdf", 16)),
        },
        "PROTEIN": {
            "MODE": protein_mode,
            "EMBEDDING_DIM": int(scopedti_cfg.get("protein_embedding_dim", 320)),
            "PADDING": bool(scopedti_cfg.get("protein_padding", True)),
            "MAX_LENGTH": int(scopedti_cfg.get("max_protein_length", 1024)),
            "GRAPH": {
                "COORD_PATH": _path_for_yaml(coord_path) if coord_path is not None else "",
                "PATH": _path_for_yaml(graph_path),
                "EDGE_CUTOFF": float(scopedti_cfg.get("protein_edge_cutoff", 10.0)),
                "NUM_KNN": int(scopedti_cfg.get("protein_num_knn", 5)),
                "NUM_LAYER": int(scopedti_cfg.get("protein_num_layers", 4)),
                "FC_BIAS": bool(scopedti_cfg.get("protein_fc_bias", True)),
            },
            "SEQUENCE": {
                "CHAR_DIM": int(scopedti_cfg.get("sequence_char_dim", 128)),
                "NUM_FILTERS": [int(item) for item in scopedti_cfg.get("sequence_num_filters", [128, 128, 128])],
                "KERNEL_SIZE": [int(item) for item in scopedti_cfg.get("sequence_kernel_size", [3, 6, 9])],
            },
        },
        "SOLVER": {
            "MAX_EPOCH": int(training_cfg.get("epochs", 70)),
            "BATCH_SIZE": int(training_cfg.get("batch_size", 16)),
            "NUM_WORKERS": int(training_cfg.get("num_workers", 0)),
            "LR": float(training_cfg.get("lr", 5.0e-5)),
            "SEED": int(config.get("seed", 3076)),
        },
        "RESULT": {
            "OUTPUT_DIR": _path_for_yaml(run_dir),
            "SAVE_MODEL": bool(training_cfg.get("save_model", True)),
        },
        "DECODER": {
            "NAME": "MLP",
            "IN_DIM": int(scopedti_cfg.get("decoder_in_dim", 256)),
            "HIDDEN_DIM": int(scopedti_cfg.get("decoder_hidden_dim", 128)),
            "OUT_DIM": int(scopedti_cfg.get("decoder_out_dim", 64)),
            "BINARY": 1,
        },
        "BCN": {"HEADS": int(scopedti_cfg.get("ban_heads", 2))},
        "DATA": {
            "TRAIN": _path_for_yaml(Path(manifest["files"]["train"])),
            "VAL": _path_for_yaml(Path(manifest["files"]["val"])),
            "TEST": _path_for_yaml(Path(manifest["files"]["test"])),
        },
    }
    return upstream


def _run_upstream_script(
    *,
    scopedti_home: Path,
    script_name: str,
    config_path: Path,
    log_path: Path,
) -> int:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{_now()}] Running upstream Scope-DTI: {script_name} {config_path}\n")
        log_file.flush()
        completed = subprocess.run(
            [sys.executable, script_name, str(config_path)],
            cwd=scopedti_home,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_file.write(f"\n[{_now()}] Upstream Scope-DTI exited with code {completed.returncode}\n")
        return int(completed.returncode)


def _move_scope_checkpoints(run_dir: Path) -> list[str]:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for path in sorted(run_dir.glob("*.pth")):
        destination = checkpoint_dir / path.name
        if destination.exists():
            destination.unlink()
        shutil.move(str(path), str(destination))
        moved.append(str(destination))
    return moved


def _normalize_upstream_metrics(upstream_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "classification": {
            "auroc": upstream_metrics.get("auroc"),
            "auprc": upstream_metrics.get("auprc"),
            "f1": upstream_metrics.get("F1") or upstream_metrics.get("f1"),
            "precision": upstream_metrics.get("Precision") or upstream_metrics.get("precision"),
            "sensitivity": upstream_metrics.get("sensitivity"),
            "specificity": upstream_metrics.get("specificity"),
            "accuracy": upstream_metrics.get("accuracy"),
            "threshold": upstream_metrics.get("thred_optim"),
        },
        "loss": upstream_metrics.get("test_loss"),
        "best_epoch": upstream_metrics.get("best_epoch"),
    }


def run_scopedti_benchmark(
    config: dict[str, Any],
    *,
    prepare_only: bool,
    dry_run: bool,
    force_sdf: bool,
    allow_missing_coords: bool | None,
    preprocess_graphs: bool | None,
) -> Path:
    output_cfg = config.setdefault("output", {})
    training_cfg = config.setdefault("training", {})
    scopedti_cfg = config.setdefault("scopedti", {})
    output_cfg.setdefault("root_dir", "runs/scopedti")
    output_cfg.setdefault("allow_rerun_suffix", True)
    config.setdefault("seed", 42)
    config.setdefault("run_name", f"scopedti_{config['data']['split_name']}_graph_s{config['seed']}")

    run_dir, actual_run_name = prepare_run_directory(
        root_dir=output_cfg["root_dir"],
        run_name=str(config["run_name"]),
        allow_rerun_suffix=bool(output_cfg["allow_rerun_suffix"]),
        resume=bool(training_cfg.get("resume", False)),
    )
    config["run_name"] = actual_run_name
    config["output"]["run_dir"] = str(run_dir)
    config["output"]["checkpoint_dir"] = str(run_dir / "checkpoints")
    logger = build_logger(run_dir / "train.log", logger_name=f"scopedti:{actual_run_name}")
    save_config(config, run_dir / "config.resolved.yaml")

    try:
        manifest = prepare_scopedti_inputs(
            config,
            force_sdf=force_sdf,
            allow_missing_coords=allow_missing_coords,
        )
    except Exception as exc:  # noqa: BLE001 - write machine-readable blocked payload.
        payload = {
            "status": "blocked",
            "reason": "prepare_failed",
            "run_name": actual_run_name,
            "benchmark_model": "Scope-DTI",
            "environment": environment_snapshot(),
            "error": str(exc),
        }
        write_json(round_nested(payload), run_dir / "metrics.json")
        logger.error("Scope-DTI benchmark preparation failed: %s", exc)
        raise SystemExit(2) from exc

    missing_modules = _missing_python_modules()
    assets = validate_scopedti_assets(config, manifest)
    base_payload = {
        "status": "prepared" if prepare_only else "dry_run" if dry_run else "pending",
        "run_name": actual_run_name,
        "benchmark_model": "Scope-DTI",
        "environment": environment_snapshot(),
        "artifacts": manifest,
        "asset_validation": assets,
        "missing_python_modules": missing_modules,
    }
    if prepare_only:
        write_json(round_nested(base_payload), run_dir / "metrics.json")
        logger.info("Prepare-only run complete. Outputs written under %s", run_dir)
        return run_dir

    if dry_run or missing_modules or not assets["ok"]:
        reason = None
        if missing_modules:
            reason = "missing_python_modules"
        elif not assets["ok"]:
            reason = str(assets.get("manifest_reason") or "missing_scopedti_assets")
        payload = {
            **base_payload,
            "status": "dry_run" if dry_run and reason is None else "blocked",
            "reason": reason,
        }
        write_json(round_nested(payload), run_dir / "metrics.json")
        if payload["status"] == "blocked":
            logger.error("Scope-DTI benchmark blocked: %s", reason)
            raise SystemExit(2)
        logger.info("Dry-run validation complete.")
        return run_dir

    scopedti_home = _resolve_path(scopedti_cfg.get("home", "artifacts/ScopeDTI/source"))
    if scopedti_home is None:
        raise ValueError("scopedti.home is required.")

    upstream_config = _build_upstream_config(config, manifest, run_dir)
    upstream_config_path = run_dir / "scopedti.upstream.yaml"
    save_config(upstream_config, upstream_config_path)

    should_preprocess = (
        bool(scopedti_cfg.get("preprocess_graphs", False))
        if preprocess_graphs is None
        else bool(preprocess_graphs)
    )
    if should_preprocess and upstream_config["PROTEIN"]["MODE"] == "graph":
        preprocess_code = _run_upstream_script(
            scopedti_home=scopedti_home,
            script_name="dataloader.py",
            config_path=upstream_config_path,
            log_path=run_dir / "train.log",
        )
        if preprocess_code != 0:
            payload = {
                **base_payload,
                "status": "failed",
                "reason": "scope_preprocess_failed",
                "return_code": preprocess_code,
                "upstream_config": str(upstream_config_path),
            }
            write_json(round_nested(payload), run_dir / "metrics.json")
            raise SystemExit(preprocess_code)

    return_code = _run_upstream_script(
        scopedti_home=scopedti_home,
        script_name="main.py",
        config_path=upstream_config_path,
        log_path=run_dir / "train.log",
    )
    if return_code != 0:
        payload = {
            **base_payload,
            "status": "failed",
            "reason": "scope_training_failed",
            "return_code": return_code,
            "upstream_config": str(upstream_config_path),
        }
        write_json(round_nested(payload), run_dir / "metrics.json")
        raise SystemExit(return_code)

    upstream_metrics_path = run_dir / "metrics.json"
    upstream_metrics = json.loads(upstream_metrics_path.read_text(encoding="utf-8"))
    moved_checkpoints = _move_scope_checkpoints(run_dir)
    payload = {
        **base_payload,
        "status": "completed",
        "upstream_config": str(upstream_config_path),
        "upstream_metrics": upstream_metrics,
        "test": _normalize_upstream_metrics(upstream_metrics),
        "checkpoints": moved_checkpoints,
    }
    write_json(round_nested(payload), upstream_metrics_path)
    logger.info("Scope-DTI benchmark finished. Metrics written to %s", upstream_metrics_path)
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare/run Scope-DTI benchmarks for CP-easy or CP-hard splits.")
    parser.add_argument("--config", required=True, help="Path to a Scope-DTI benchmark YAML config.")
    parser.add_argument("--set", action="append", default=[], help="Override a config value with key=value.")
    parser.add_argument("--prepare-only", action="store_true", help="Only export Scope-DTI input files and metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare inputs and validate assets without training.")
    parser.add_argument("--force-sdf", action="store_true", help="Regenerate the compound SDF/MolBlock cache.")
    parser.add_argument(
        "--allow-missing-coords",
        action="store_true",
        help="Explicitly drop graph-mode rows whose proteins are absent from the coordinate PKL.",
    )
    parser.add_argument("--preprocess-graphs", action="store_true", help="Run upstream dataloader.py before training.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    run_scopedti_benchmark(
        config,
        prepare_only=bool(args.prepare_only),
        dry_run=bool(args.dry_run),
        force_sdf=bool(args.force_sdf),
        allow_missing_coords=True if args.allow_missing_coords else None,
        preprocess_graphs=True if args.preprocess_graphs else None,
    )


if __name__ == "__main__":
    main()
