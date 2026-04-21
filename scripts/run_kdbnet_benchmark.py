"""Prepare and run KDBNet benchmarks on the CP-easy/CP-hard splits.

Upstream KDBNet code and generated benchmark inputs live under ``artifacts/``;
run logs, checkpoints, predictions, and metrics live under ``runs/``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl

from src.config import apply_overrides, load_config, save_config
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


SPLIT_FILE_NAMES = {"train": "train.tsv", "val": "valid.tsv", "test": "test.tsv"}
KDBNET_REQUIRED_MODULES = ["pandas", "rdkit", "scipy", "torch", "torch_cluster", "torch_geometric", "yaml"]


def _resolve_path(value: str | Path | None, *, base: Path = PROJECT_ROOT) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else base / path


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_split_table(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split file does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    raise ValueError(f"Unsupported split file format for KDBNet export: {path.suffix}")


def _format_kdbnet_frame(frame: pl.DataFrame, split_name: str) -> pl.DataFrame:
    required = {"inchi_key", "target_uniprot_id", "smiles", "sequence", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Split `{split_name}` is missing required columns: {sorted(missing)}")
    return frame.select(
        [
            pl.col("inchi_key").cast(pl.Utf8).alias("drug"),
            pl.col("target_uniprot_id").cast(pl.Utf8).alias("protein"),
            pl.col("smiles").cast(pl.Utf8),
            pl.col("sequence").cast(pl.Utf8),
            pl.col("label").cast(pl.Float32).alias("y"),
            pl.lit(split_name).alias("split"),
        ]
    )


def _summarize_frame(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": float(frame.height),
        "n_drugs": float(frame.select("drug").n_unique()),
        "n_proteins": float(frame.select("protein").n_unique()),
        "positive_ratio": 0.0 if frame.height == 0 else float(frame["y"].mean()),
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
        "examples": conflicts.head(5).select(key).to_series().to_list() if conflicts.height else [],
    }


def _safe_write_tsv(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path, separator="\t")


def _write_asset_readme(artifact_dir: Path, split_name: str) -> None:
    readme_path = artifact_dir / "README.md"
    if readme_path.exists():
        return
    readme_path.write_text(
        "\n".join(
            [
                f"# KDBNet {split_name} Benchmark Inputs",
                "",
                "Generated from this repository's cold-protein split parquet files.",
                "",
                "Files here:",
                "- `train.tsv`, `valid.tsv`, `test.tsv`: KDBNet rows with `drug`, `protein`, `y`, and `split`.",
                "- `all.tsv`: concatenated split file.",
                "- `drugs.tsv`: unique InChIKey/SMILES pairs.",
                "- `proteins.tsv`: unique UniProt/sequence pairs.",
                "- `sdf/`: optional generated compound 3D SDF files, one per `drug` key.",
                "",
                "KDBNet also requires protein assets that are not derivable from these splits:",
                "- protein-to-PDB mapping YAML;",
                "- `pockets_structure.json` in the upstream KDBNet format;",
                "- matching ESM-1b residue embedding `.pt` files.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_protein_mapping_template(proteins: pl.DataFrame, artifact_dir: Path) -> None:
    template_path = artifact_dir / "protein2pdb.template.yaml"
    if template_path.exists():
        return
    protein_ids = proteins.select("protein").to_series().to_list()
    preview = protein_ids[:50]
    lines = [
        "# Fill this file with KDBNet protein-structure mappings, then point",
        "# `kdbnet.protein2pdb_path` at the completed YAML.",
        "#",
        "# Format:",
        "#   <target_uniprot_id>: <pdb_structure_key_in_pockets_structure_json>",
        "#",
    ]
    lines.extend(f"# {protein_id}: null" for protein_id in preview)
    if len(protein_ids) > len(preview):
        lines.append(f"# ... {len(protein_ids) - len(preview)} additional proteins omitted")
    template_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sdf_files(drugs: pl.DataFrame, sdf_dir: Path, *, force: bool = False, seed: int = 13) -> dict[str, Any]:
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except ImportError as exc:
        raise RuntimeError("RDKit is required to generate KDBNet SDF files. Rerun with --skip-sdf.") from exc

    sdf_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_existing = 0
    failures: list[dict[str, str]] = []
    for row in drugs.iter_rows(named=True):
        drug_id = str(row["drug"])
        smiles = str(row["smiles"])
        output_path = sdf_dir / f"{drug_id}.sdf"
        if output_path.exists() and not force:
            skipped_existing += 1
            continue
        try:
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
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=200)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            mol.SetProp("_Name", drug_id)
            writer = Chem.SDWriter(str(output_path))
            try:
                writer.write(mol)
            finally:
                writer.close()
            written += 1
        except Exception as exc:  # noqa: BLE001 - keep per-compound failures in metadata.
            failures.append({"drug": drug_id, "reason": str(exc)})
    return {
        "sdf_dir": str(sdf_dir),
        "written": written,
        "skipped_existing": skipped_existing,
        "failed": len(failures),
        "failure_examples": failures[:20],
    }


def prepare_kdbnet_inputs(
    config: dict[str, Any],
    *,
    write_sdf: bool | None = None,
    force_sdf: bool = False,
) -> dict[str, Any]:
    data_cfg = config["data"]
    export_cfg = config.get("export", {})
    kdbnet_cfg = config.get("kdbnet", {})
    split_name = str(data_cfg["split_name"])
    split_dir = _resolve_path(data_cfg["split_dir"])
    if split_dir is None:
        raise ValueError("data.split_dir is required.")

    artifact_dir = _resolve_path(kdbnet_cfg.get("artifact_dir") or export_cfg.get("artifact_dir"))
    if artifact_dir is None:
        artifact_dir = PROJECT_ROOT / "artifacts" / "kdbnet" / "benchmarks" / split_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pl.DataFrame] = {}
    for split in ("train", "val", "test"):
        input_path = split_dir / f"{split}.parquet"
        if not input_path.exists():
            input_path = split_dir / f"{split}.csv"
        frames[split] = _format_kdbnet_frame(_read_split_table(input_path), split)

    for split, frame in frames.items():
        _safe_write_tsv(frame.select(["drug", "protein", "y", "split"]), artifact_dir / SPLIT_FILE_NAMES[split])

    all_frame = pl.concat([frames["train"], frames["val"], frames["test"]], how="vertical")
    _safe_write_tsv(all_frame.select(["drug", "protein", "y", "split"]), artifact_dir / "all.tsv")
    drugs = all_frame.select(["drug", "smiles"]).unique(subset=["drug"], keep="first").sort("drug")
    proteins = all_frame.select(["protein", "sequence"]).unique(subset=["protein"], keep="first").sort("protein")
    _safe_write_tsv(drugs, artifact_dir / "drugs.tsv")
    _safe_write_tsv(proteins, artifact_dir / "proteins.tsv")
    _write_asset_readme(artifact_dir, split_name)
    _write_protein_mapping_template(proteins, artifact_dir)

    should_write_sdf = bool(export_cfg.get("write_sdf", False)) if write_sdf is None else bool(write_sdf)
    sdf_summary = None
    if should_write_sdf:
        sdf_dir = _resolve_path(kdbnet_cfg.get("drug_sdf_dir")) or (artifact_dir / "sdf")
        sdf_summary = _write_sdf_files(drugs, sdf_dir, force=force_sdf, seed=int(config.get("seed", 13)))

    manifest = {
        "generated_at": _now(),
        "split_name": split_name,
        "source_split_dir": str(split_dir),
        "artifact_dir": str(artifact_dir),
        "files": {
            "train": str(artifact_dir / "train.tsv"),
            "valid": str(artifact_dir / "valid.tsv"),
            "test": str(artifact_dir / "test.tsv"),
            "all": str(artifact_dir / "all.tsv"),
            "drugs": str(artifact_dir / "drugs.tsv"),
            "proteins": str(artifact_dir / "proteins.tsv"),
            "protein2pdb_template": str(artifact_dir / "protein2pdb.template.yaml"),
        },
        "summary": {split: _summarize_frame(frame) for split, frame in frames.items()},
        "total": _summarize_frame(all_frame),
        "conflicts": {
            "drug_smiles": _conflict_summary(all_frame, "drug", "smiles"),
            "protein_sequence": _conflict_summary(all_frame, "protein", "sequence"),
        },
        "sdf": sdf_summary,
    }
    write_json(round_nested(manifest), artifact_dir / "manifest.json")
    return manifest


def _missing_python_modules() -> list[str]:
    return [name for name in KDBNET_REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for KDBNet protein mapping files.") from exc
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {} if loaded is None else loaded


def _read_tsv_column(path: Path, column: str) -> list[str]:
    frame = pl.read_csv(path, separator="\t")
    return [str(item) for item in frame[column].cast(pl.Utf8).to_list()]


def _discover_checkpoints(config: dict[str, Any], explicit_checkpoints: list[str] | None = None) -> list[Path]:
    kdbnet_cfg = config.get("kdbnet", {})
    checkpoint_values: list[str] = []
    checkpoint_values.extend(explicit_checkpoints or [])
    configured = kdbnet_cfg.get("checkpoints") or []
    if isinstance(configured, str):
        checkpoint_values.append(configured)
    else:
        checkpoint_values.extend(str(item) for item in configured)

    checkpoints: list[Path] = []
    for value in checkpoint_values:
        path = _resolve_path(value)
        if path is not None and path.exists() and path.is_file():
            checkpoints.append(path)

    weights_dir = _resolve_path(kdbnet_cfg.get("weights_dir"))
    if weights_dir is not None and weights_dir.exists():
        for pattern in ("*.pt", "*.pth", "*.ckpt"):
            checkpoints.extend(path for path in sorted(weights_dir.glob(pattern)) if path.is_file())

    deduped: list[Path] = []
    seen: set[Path] = set()
    for checkpoint in checkpoints:
        resolved = checkpoint.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(checkpoint)
    return deduped


def validate_kdbnet_assets(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    kdbnet_cfg = config.get("kdbnet", {})
    artifact_dir = Path(manifest["artifact_dir"])
    kdbnet_home = _resolve_path(kdbnet_cfg.get("home", "artifacts/kdbnet/source"))
    protein2pdb_path = _resolve_path(kdbnet_cfg.get("protein2pdb_path"))
    pdb_json_path = _resolve_path(kdbnet_cfg.get("pdb_json_path"))
    esm_dir = _resolve_path(kdbnet_cfg.get("esm_dir"))
    drug_sdf_dir = _resolve_path(kdbnet_cfg.get("drug_sdf_dir")) or (artifact_dir / "sdf")

    missing_paths: list[dict[str, str]] = []
    required_paths = {
        "kdbnet_home": kdbnet_home,
        "protein2pdb_path": protein2pdb_path,
        "pdb_json_path": pdb_json_path,
        "esm_dir": esm_dir,
        "drug_sdf_dir": drug_sdf_dir,
    }
    for name, path in required_paths.items():
        if path is None or not path.exists():
            missing_paths.append({"name": name, "path": "" if path is None else str(path)})
    if kdbnet_home is not None and not (kdbnet_home / "kdbnet" / "model.py").exists():
        missing_paths.append({"name": "kdbnet_source", "path": str(kdbnet_home / "kdbnet" / "model.py")})

    coverage: dict[str, Any] = {}
    missing_entities: dict[str, Any] = {}
    if not missing_paths and protein2pdb_path and pdb_json_path and esm_dir and drug_sdf_dir:
        protein2pdb = {str(k): str(v) for k, v in _load_yaml(protein2pdb_path).items()}
        pdb_data = json.loads(pdb_json_path.read_text(encoding="utf-8"))
        pdb_data = {str(key): value for key, value in pdb_data.items()}
        proteins = _read_tsv_column(artifact_dir / "proteins.tsv", "protein")
        drugs = _read_tsv_column(artifact_dir / "drugs.tsv", "drug")

        missing_protein_mapping = sorted(set(proteins) - set(protein2pdb))
        mapped_pdb_ids = {protein2pdb[protein] for protein in proteins if protein in protein2pdb}
        missing_pdb_entries = sorted(mapped_pdb_ids - set(pdb_data))
        missing_embeddings: list[str] = []
        for protein in proteins:
            pdb_key = protein2pdb.get(protein)
            if pdb_key is None or pdb_key not in pdb_data:
                continue
            entry = pdb_data[pdb_key]
            embed_name = f"{entry['PDB_id']}.{entry['chain']}.pt"
            if not (esm_dir / embed_name).exists():
                missing_embeddings.append(embed_name)
        missing_sdf = [drug for drug in drugs if not (drug_sdf_dir / f"{drug}.sdf").exists()]
        missing_entities = {
            "protein_mapping": {"count": len(missing_protein_mapping), "examples": missing_protein_mapping[:20]},
            "pdb_entries": {"count": len(missing_pdb_entries), "examples": missing_pdb_entries[:20]},
            "esm_embeddings": {"count": len(missing_embeddings), "examples": missing_embeddings[:20]},
            "drug_sdf": {"count": len(missing_sdf), "examples": missing_sdf[:20]},
        }
        coverage = {
            "n_proteins": len(proteins),
            "n_drugs": len(drugs),
            "n_mapped_proteins": len(proteins) - len(missing_protein_mapping),
            "n_available_drug_sdf": len(drugs) - len(missing_sdf),
        }

    has_missing_entities = any(item.get("count", 0) > 0 for item in missing_entities.values())
    return {
        "ok": not missing_paths and not has_missing_entities,
        "paths": {name: str(path) if path is not None else None for name, path in required_paths.items()},
        "missing_paths": missing_paths,
        "missing_entities": missing_entities,
        "coverage": coverage,
    }


def _add_kdbnet_to_path(kdbnet_home: Path) -> None:
    home = str(kdbnet_home)
    if home not in sys.path:
        sys.path.insert(0, home)


def _load_state_dict(torch_module: Any, checkpoint_path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch_module.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch_module.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model_state_dict"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return state
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    raise ValueError(f"Unable to find a model state dict in checkpoint: {checkpoint_path}")


def _load_model_checkpoint(model: Any, torch_module: Any, checkpoint_path: Path) -> None:
    state = _load_state_dict(torch_module, checkpoint_path)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        stripped = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(stripped)


def _save_model_checkpoint(
    torch_module: Any,
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    epoch: int,
    metric_name: str,
    metric_value: float,
    config: dict[str, Any],
    val_metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric_name": metric_name,
            "best_metric": metric_value,
            "config": config,
            "val_metrics": val_metrics,
        },
        path,
    )


def _metric_at_path(metrics: dict[str, Any], dotted_path: str) -> float:
    value: Any = metrics
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Metric path `{dotted_path}` not found.")
        value = value[part]
    if value is None:
        return float("-inf")
    scalar = float(value)
    return scalar if math.isfinite(scalar) else float("-inf")


def _build_kdbnet_model(config: dict[str, Any]) -> Any:
    from kdbnet.model import DTAModel  # type: ignore

    model_cfg = config.get("model", {})
    return DTAModel(
        prot_emb_dim=int(model_cfg.get("prot_emb_dim", 1280)),
        prot_gcn_dims=[int(item) for item in model_cfg.get("prot_gcn_dims", [128, 256, 256])],
        prot_fc_dims=[int(item) for item in model_cfg.get("prot_fc_dims", [1024, 128])],
        drug_node_in_dim=[int(item) for item in model_cfg.get("drug_node_in_dim", [66, 1])],
        drug_node_h_dims=[int(item) for item in model_cfg.get("drug_gcn_dims", [128, 64])],
        drug_edge_in_dim=[int(item) for item in model_cfg.get("drug_edge_in_dim", [16, 1])],
        drug_edge_h_dims=[int(item) for item in model_cfg.get("drug_edge_h_dims", [32, 1])],
        drug_fc_dims=[int(item) for item in model_cfg.get("drug_fc_dims", [1024, 128])],
        mlp_dims=[int(item) for item in model_cfg.get("mlp_dims", [1024, 512])],
        mlp_dropout=float(model_cfg.get("mlp_dropout", 0.25)),
    )


def _build_kdbnet_data(
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import pandas as pd  # type: ignore
    import yaml  # type: ignore
    from torch_geometric.loader import DataLoader  # type: ignore

    from kdbnet.dta import DTATask  # type: ignore

    kdbnet_cfg = config.get("kdbnet", {})
    training_cfg = config.get("training", {})
    artifact_dir = Path(manifest["artifact_dir"])
    protein2pdb_path = _resolve_path(kdbnet_cfg.get("protein2pdb_path"))
    pdb_json_path = _resolve_path(kdbnet_cfg.get("pdb_json_path"))
    esm_dir = _resolve_path(kdbnet_cfg.get("esm_dir"))
    drug_sdf_dir = _resolve_path(kdbnet_cfg.get("drug_sdf_dir")) or (artifact_dir / "sdf")
    if protein2pdb_path is None or pdb_json_path is None or esm_dir is None:
        raise ValueError("KDBNet protein asset paths are required before building data.")

    split_df = {
        "train": pd.read_table(artifact_dir / "train.tsv"),
        "valid": pd.read_table(artifact_dir / "valid.tsv"),
        "test": pd.read_table(artifact_dir / "test.tsv"),
    }
    for frame in split_df.values():
        frame["drug"] = frame["drug"].astype(str)
        frame["protein"] = frame["protein"].astype(str)
        frame["y"] = frame["y"].astype(float)

    all_df = pd.concat(split_df.values(), ignore_index=True)
    task = DTATask(
        task_name="cold_protein_dti",
        df=all_df,
        prot_pdb_id=yaml.safe_load(protein2pdb_path.read_text(encoding="utf-8")),
        pdb_data=json.loads(pdb_json_path.read_text(encoding="utf-8")),
        emb_dir=str(esm_dir),
        drug_sdf_dir=str(drug_sdf_dir),
        num_pos_emb=int(kdbnet_cfg.get("num_pos_emb", 16)),
        num_rbf=int(kdbnet_cfg.get("num_rbf", 16)),
        contact_cutoff=float(kdbnet_cfg.get("contact_cutoff", 8.0)),
        seed=int(config.get("seed", 42)),
        onthefly=bool(kdbnet_cfg.get("onthefly", False)),
    )
    data = {
        split: task.build_data(frame[["drug", "protein", "y"]], onthefly=bool(kdbnet_cfg.get("onthefly", False)))
        for split, frame in split_df.items()
    }
    loaders = {
        split: DataLoader(
            dataset=dataset,
            batch_size=int(training_cfg.get("batch_size", 128)),
            shuffle=(split == "train"),
            pin_memory=False,
            num_workers=int(training_cfg.get("num_workers", 0)),
        )
        for split, dataset in data.items()
    }
    return loaders, split_df


def _evaluate_model(
    *,
    model: Any,
    loader: Any,
    split_df: Any,
    torch_module: Any,
    criterion: Any,
    device: str,
    config: dict[str, Any],
    output_kind: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    outputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0
    with torch_module.no_grad():
        for batch in loader:
            drug_graph = batch["drug"].to(device)
            protein_graph = batch["protein"].to(device)
            y = batch["y"].to(device).view(-1)
            raw = model(drug_graph, protein_graph).view(-1)
            if not bool(torch_module.isfinite(raw).all().item()):
                raise ValueError("KDBNet produced NaN or Inf outputs during evaluation.")
            loss = criterion(raw, y)
            total_loss += float(loss.item())
            n_batches += 1
            outputs.append(raw.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
    if not outputs:
        raise RuntimeError("KDBNet evaluation produced zero batches.")

    raw_np = np.concatenate(outputs)
    labels_np = np.concatenate(labels)
    target_ids = [str(item) for item in split_df["protein"].tolist()]
    metrics_cfg = config.get("metrics", {})
    kwargs: dict[str, Any] = {
        "labels": labels_np,
        "group_ids": target_ids,
        "threshold": float(metrics_cfg.get("threshold", 0.5)),
        "ks": [int(item) for item in metrics_cfg.get("ks", [10, 50, 100])],
        "ef_fractions": [float(item) for item in metrics_cfg.get("ef_fractions", [0.01, 0.05])],
        "loss": total_loss / max(n_batches, 1),
    }
    if output_kind == "scores":
        kwargs["scores"] = raw_np
    else:
        kwargs["logits"] = raw_np
    return build_metrics_payload(**kwargs), raw_np, labels_np


def _write_predictions(path: Path, split_df: Any, raw_outputs: np.ndarray, output_kind: str) -> None:
    output = split_df.copy()
    output["y_true"] = output["y"].astype(float)
    output["y_raw"] = raw_outputs
    output["y_score"] = raw_outputs if output_kind == "scores" else sigmoid(raw_outputs)
    columns = ["drug", "protein", "y_true", "y_raw", "y_score", "split"]
    path.parent.mkdir(parents=True, exist_ok=True)
    output[columns].to_csv(path, sep="\t", index=False)


def _train_one_model(
    *,
    model_index: int,
    config: dict[str, Any],
    loaders: dict[str, Any],
    split_df: dict[str, Any],
    torch_module: Any,
    device: str,
    run_dir: Path,
    logger: Any,
) -> tuple[Any, dict[str, Any]]:
    training_cfg = config.get("training", {})
    metrics_cfg = config.get("metrics", {})
    output_kind = str(config.get("kdbnet", {}).get("output_kind", "logits"))
    model = _build_kdbnet_model(config).to(device)
    optimizer = torch_module.optim.Adam(
        model.parameters(),
        lr=float(training_cfg.get("lr", 0.0005)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    criterion = torch_module.nn.BCEWithLogitsLoss()
    primary_metric = str(metrics_cfg.get("primary", "classification.auprc"))
    best_metric = float("-inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    epochs = int(training_cfg.get("epochs", 100))
    patience = int(training_cfg.get("early_stopping_patience", 10))
    max_train_batches = training_cfg.get("max_train_batches")
    max_train_batches = int(max_train_batches) if max_train_batches is not None else None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_index, batch in enumerate(loaders["train"]):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            drug_graph = batch["drug"].to(device)
            protein_graph = batch["protein"].to(device)
            y = batch["y"].to(device).view(-1)
            optimizer.zero_grad(set_to_none=True)
            raw = model(drug_graph, protein_graph).view(-1)
            loss = criterion(raw, y)
            if not bool(torch_module.isfinite(loss).all().item()):
                raise ValueError(f"Non-finite KDBNet training loss at epoch {epoch}, batch {batch_index}.")
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        val_metrics, _, _ = _evaluate_model(
            model=model,
            loader=loaders["valid"],
            split_df=split_df["valid"],
            torch_module=torch_module,
            criterion=criterion,
            device=device,
            config=config,
            output_kind=output_kind,
        )
        val_score = _metric_at_path(val_metrics, primary_metric)
        train_loss = epoch_loss / max(n_batches, 1)
        history.append(round_nested({"epoch": epoch, "model_index": model_index, "train_loss": train_loss, "val": val_metrics}))
        logger.info(
            "KDBNet model %s epoch %s train_loss=%.6f %s=%.6f",
            model_index,
            epoch,
            train_loss,
            primary_metric,
            val_score,
        )

        _save_model_checkpoint(
            torch_module,
            run_dir / "checkpoints" / f"model_{model_index}.latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metric_name=primary_metric,
            metric_value=val_score,
            config=config,
            val_metrics=val_metrics,
        )
        if val_score > best_metric:
            best_metric = val_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            _save_model_checkpoint(
                torch_module,
                run_dir / "checkpoints" / f"model_{model_index}.best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metric_name=primary_metric,
                metric_value=val_score,
                config=config,
                val_metrics=val_metrics,
            )
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                logger.info("KDBNet model %s early stopped at epoch %s.", model_index, epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "model_index": model_index,
        "best_epoch": best_epoch,
        "best_metric_name": primary_metric,
        "best_metric": best_metric,
        "history": history,
    }


def _average_model_outputs(
    *,
    models: list[Any],
    loader: Any,
    split_df: Any,
    torch_module: Any,
    criterion: Any,
    device: str,
    config: dict[str, Any],
    output_kind: str,
) -> tuple[dict[str, Any], np.ndarray]:
    raw_outputs: list[np.ndarray] = []
    labels_np: np.ndarray | None = None
    for model in models:
        _, raw_np, current_labels = _evaluate_model(
            model=model,
            loader=loader,
            split_df=split_df,
            torch_module=torch_module,
            criterion=criterion,
            device=device,
            config=config,
            output_kind=output_kind,
        )
        raw_outputs.append(raw_np)
        labels_np = current_labels if labels_np is None else labels_np

    averaged_raw = np.mean(np.vstack(raw_outputs), axis=0)
    target_ids = [str(item) for item in split_df["protein"].tolist()]
    metrics_cfg = config.get("metrics", {})
    kwargs: dict[str, Any] = {
        "labels": labels_np,
        "group_ids": target_ids,
        "threshold": float(metrics_cfg.get("threshold", 0.5)),
        "ks": [int(item) for item in metrics_cfg.get("ks", [10, 50, 100])],
        "ef_fractions": [float(item) for item in metrics_cfg.get("ef_fractions", [0.01, 0.05])],
    }
    if output_kind == "scores":
        kwargs["scores"] = averaged_raw
    else:
        kwargs["logits"] = averaged_raw
    return build_metrics_payload(**kwargs), averaged_raw


def run_kdbnet_benchmark(
    config: dict[str, Any],
    *,
    prepare_only: bool,
    dry_run: bool,
    write_sdf: bool | None,
    force_sdf: bool,
    force_train: bool,
    infer_only: bool,
    explicit_checkpoints: list[str] | None,
) -> Path:
    output_cfg = config.setdefault("output", {})
    training_cfg = config.setdefault("training", {})
    output_cfg.setdefault("root_dir", "runs/kdbnet")
    output_cfg.setdefault("allow_rerun_suffix", True)
    config.setdefault("seed", 42)
    config.setdefault("run_name", f"kdbnet_{config['data']['split_name']}_s{config['seed']}")

    run_dir, actual_run_name = prepare_run_directory(
        root_dir=output_cfg["root_dir"],
        run_name=str(config["run_name"]),
        allow_rerun_suffix=bool(output_cfg["allow_rerun_suffix"]),
        resume=bool(training_cfg.get("resume", False)),
    )
    config["run_name"] = actual_run_name
    config["output"]["run_dir"] = str(run_dir)
    config["output"]["checkpoint_dir"] = str(run_dir / "checkpoints")
    logger = build_logger(run_dir / "kdbnet.log", logger_name=f"kdbnet:{actual_run_name}")
    save_config(config, run_dir / "config.resolved.yaml")

    logger.info("Preparing KDBNet benchmark inputs for %s", config["data"]["split_name"])
    manifest = prepare_kdbnet_inputs(config, write_sdf=write_sdf, force_sdf=force_sdf)
    checkpoints = [] if force_train else _discover_checkpoints(config, explicit_checkpoints)
    weights_available = len(checkpoints) > 0
    execution_mode = "inference_only" if weights_available and not force_train else "train_then_eval"
    if infer_only and not weights_available:
        execution_mode = "blocked_missing_checkpoint"

    base_payload = {
        "status": "prepared" if prepare_only else "dry_run" if dry_run else "pending",
        "mode": execution_mode,
        "run_name": actual_run_name,
        "benchmark_model": "KDBNet",
        "environment": environment_snapshot(),
        "artifacts": manifest,
        "checkpoints": [str(path) for path in checkpoints],
    }
    if prepare_only:
        write_json(round_nested(base_payload), run_dir / "metrics.json")
        logger.info("Prepare-only run complete. Outputs written under %s", run_dir)
        return run_dir

    missing_modules = _missing_python_modules()
    assets = validate_kdbnet_assets(config, manifest)
    if dry_run or missing_modules or not assets["ok"] or (infer_only and not weights_available):
        status = "dry_run" if dry_run and not missing_modules and assets["ok"] else "blocked"
        reason = None
        if infer_only and not weights_available:
            reason = "missing_checkpoint"
        elif missing_modules:
            reason = "missing_python_modules"
        elif not assets["ok"]:
            reason = "missing_kdbnet_assets"
        payload = {
            **base_payload,
            "status": status,
            "reason": reason,
            "missing_python_modules": missing_modules,
            "asset_validation": assets,
        }
        write_json(round_nested(payload), run_dir / "metrics.json")
        if status == "blocked":
            logger.error("KDBNet benchmark blocked: %s", reason)
            raise SystemExit(2)
        logger.info("Dry-run validation complete.")
        return run_dir

    kdbnet_home = _resolve_path(config.get("kdbnet", {}).get("home", "artifacts/kdbnet/source"))
    if kdbnet_home is None:
        raise ValueError("kdbnet.home is required.")
    _add_kdbnet_to_path(kdbnet_home)

    torch = try_import_torch()
    device = choose_device(str(training_cfg.get("device", "cuda")), torch)
    set_global_seed(int(config["seed"]), torch_module=torch)
    loaders, split_df = _build_kdbnet_data(config, manifest)
    criterion = torch.nn.BCEWithLogitsLoss()
    output_kind = str(config.get("kdbnet", {}).get("output_kind", "logits"))
    if output_kind not in {"logits", "scores"}:
        raise ValueError("kdbnet.output_kind must be `logits` or `scores`.")

    models: list[Any] = []
    training_summaries: list[dict[str, Any]] = []
    if weights_available and not force_train:
        logger.info("KDBNet checkpoint weights found; running inference only.")
        for checkpoint in checkpoints:
            model = _build_kdbnet_model(config).to(device)
            _load_model_checkpoint(model, torch, checkpoint)
            models.append(model)
    else:
        n_ensembles = int(config.get("kdbnet", {}).get("n_ensembles", 1))
        for index in range(1, n_ensembles + 1):
            set_global_seed(int(config["seed"]) + index - 1, torch_module=torch)
            model, summary = _train_one_model(
                model_index=index,
                config=config,
                loaders=loaders,
                split_df=split_df,
                torch_module=torch,
                device=device,
                run_dir=run_dir,
                logger=logger,
            )
            models.append(model)
            training_summaries.append(summary)

    test_metrics, raw_outputs = _average_model_outputs(
        models=models,
        loader=loaders["test"],
        split_df=split_df["test"],
        torch_module=torch,
        criterion=criterion,
        device=device,
        config=config,
        output_kind=output_kind,
    )
    predictions_path = run_dir / "predictions.tsv"
    _write_predictions(predictions_path, split_df["test"], raw_outputs, output_kind)

    payload = {
        **base_payload,
        "status": "completed",
        "mode": "inference_only" if weights_available and not force_train else "train_then_eval",
        "device": device,
        "asset_validation": assets,
        "training": training_summaries if training_summaries else None,
        "test": test_metrics,
        "predictions": str(predictions_path),
    }
    write_json(round_nested(payload), run_dir / "metrics.json")
    logger.info("KDBNet benchmark finished. Metrics written to %s", run_dir / "metrics.json")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare/run KDBNet benchmarks for CP-easy or CP-hard splits.")
    parser.add_argument("--config", required=True, help="Path to a KDBNet benchmark YAML config.")
    parser.add_argument("--set", action="append", default=[], help="Override a config value with key=value.")
    parser.add_argument("--checkpoint", action="append", default=[], help="Checkpoint to use for inference-only mode.")
    parser.add_argument("--prepare-only", action="store_true", help="Only export KDBNet input files and metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare inputs and validate required assets without running.")
    parser.add_argument("--write-sdf", dest="write_sdf", action="store_true", help="Generate compound SDF files.")
    parser.add_argument("--skip-sdf", dest="write_sdf", action="store_false", help="Do not generate compound SDF files.")
    parser.set_defaults(write_sdf=None)
    parser.add_argument("--force-sdf", action="store_true", help="Regenerate existing SDF files.")
    parser.add_argument("--force-train", action="store_true", help="Ignore discovered weights and train.")
    parser.add_argument("--infer-only", action="store_true", help="Require checkpoint weights; do not train if absent.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    run_kdbnet_benchmark(
        config,
        prepare_only=bool(args.prepare_only),
        dry_run=bool(args.dry_run),
        write_sdf=args.write_sdf,
        force_sdf=bool(args.force_sdf),
        force_train=bool(args.force_train),
        infer_only=bool(args.infer_only),
        explicit_checkpoints=args.checkpoint,
    )


if __name__ == "__main__":
    main()
