#!/usr/bin/env python
"""Collect experiment metrics from ``results/`` into a Polars DataFrame."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import polars as pl

_RERUN_SUFFIX_RE = re.compile(r"__(\d{8}_\d{6})$")
_SEED_RE = re.compile(r"^s(\d+)$")
_DATASET_ALIASES = {
    "cp-easy": "cp-easy",
    "cp-hard": "cp-hard",
    "cp_easy": "cp-easy",
    "cp_hard": "cp-hard",
}
_KNOWN_FUSIONS = {"concat", "ban"}
_PROT_ENCODER_ALIASES = {
    "cnn": "cnn",
    "esm": "esm",
    "esm_frozen": "esm_frozen",
    "esm_part_frozen": "esm_part_frozen",
    "esm_finetuned": "esm_fine_tune",
    "esm_fine_tune": "esm_fine_tune",
    "esm_finetune": "esm_fine_tune",
}
_EMPTY_SCHEMA: dict[str, pl.DataType] = {
    "id": pl.Int64,
    "results_group": pl.String,
    "relative_run_dir": pl.String,
    "run_dir_name": pl.String,
    "run_name": pl.String,
    "base_run_name": pl.String,
    "rerun_suffix": pl.String,
    "experiment_tag": pl.String,
    "dataset": pl.String,
    "prot_enc": pl.String,
    "fusion_type": pl.String,
    "seed": pl.Int64,
    "metrics_path": pl.String,
}


def _strip_rerun_suffix(run_name: str) -> tuple[str, str | None]:
    """Split ``run_name`` into the base name and optional rerun timestamp suffix."""

    match = _RERUN_SUFFIX_RE.search(run_name)
    if match is None:
        return run_name, None
    return run_name[: match.start()], match.group(1)


def _find_dataset_tokens(tokens: list[str]) -> tuple[int | None, int, str | None]:
    """Locate the dataset token span inside a tokenized run name."""

    for index, token in enumerate(tokens):
        normalized = _DATASET_ALIASES.get(token)
        if normalized is not None:
            return index, 1, normalized
        if index + 1 >= len(tokens):
            continue
        pair = f"{token}_{tokens[index + 1]}"
        normalized = _DATASET_ALIASES.get(pair)
        if normalized is not None:
            return index, 2, normalized
    return None, 0, None


def parse_run_name(run_name: str) -> dict[str, Any]:
    """Parse metadata fields from a run name.

    Supported examples:
    - ``cp-easy_cnn_concat_s42``
    - ``exprS1_cp-hard_esm_frozen_concat_s42``
    - ``exprS2_cp-easy_esm_fine_tune_ban``
    - ``exprS1_cp-easy_cnn_concat_s42__20260324_154012``
    """

    base_run_name, rerun_suffix = _strip_rerun_suffix(run_name)
    tokens = base_run_name.split("_")
    dataset_index, dataset_width, dataset = _find_dataset_tokens(tokens)

    parsed: dict[str, Any] = {
        "run_name": run_name,
        "base_run_name": base_run_name,
        "rerun_suffix": rerun_suffix,
        "experiment_tag": None,
        "dataset": dataset,
        "prot_enc": None,
        "fusion_type": None,
        "seed": None,
    }
    if dataset_index is None:
        return parsed

    experiment_tokens = tokens[:dataset_index]
    parsed["experiment_tag"] = "_".join(experiment_tokens) if experiment_tokens else None

    suffix_tokens = tokens[dataset_index + dataset_width :]
    if suffix_tokens:
        seed_match = _SEED_RE.match(suffix_tokens[-1])
        if seed_match is not None:
            parsed["seed"] = int(seed_match.group(1))
            suffix_tokens = suffix_tokens[:-1]

    if suffix_tokens and suffix_tokens[-1] in _KNOWN_FUSIONS:
        parsed["fusion_type"] = suffix_tokens[-1]
        suffix_tokens = suffix_tokens[:-1]

    if suffix_tokens:
        prot_key = "_".join(suffix_tokens)
        parsed["prot_enc"] = _PROT_ENCODER_ALIASES.get(prot_key, prot_key)

    return parsed


def _flatten_scalars(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict values and summarize lists by their length."""

    flat: dict[str, Any] = {}
    for key, value in payload.items():
        column_name = f"{prefix}__{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_scalars(value, prefix=column_name))
            continue
        if isinstance(value, list):
            flat[f"{column_name}__len"] = len(value)
            continue
        flat[column_name] = value
    return flat


def _load_metrics_row(metrics_path: Path, results_root: Path, row_id: int) -> dict[str, Any]:
    """Load one metrics file and convert it into a flat row dict."""

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    relative_parts = metrics_path.relative_to(results_root).parts
    run_dir = metrics_path.parent
    run_dir_name = run_dir.name
    run_name = str(payload.get("run_name") or run_dir_name)

    if len(relative_parts) > 2:
        results_group = relative_parts[0]
    else:
        results_group = None

    row: dict[str, Any] = {
        "id": row_id,
        "results_group": results_group,
        "relative_run_dir": str(run_dir.relative_to(results_root)),
        "run_dir_name": run_dir_name,
        "metrics_path": str(metrics_path),
    }
    row.update(parse_run_name(run_name))
    row.update(_flatten_scalars(payload))
    return row


def collect_results_metrics(results_root: str | Path = "results") -> pl.DataFrame:
    """Collect all ``metrics.json`` files under ``results_root``.

    The returned DataFrame includes parsed metadata columns such as:
    ``dataset``, ``prot_enc``, ``fusion_type``, ``experiment_tag``, ``seed``,
    ``results_group``, ``rerun_suffix``, and a stable integer ``id``.
    """

    root = Path(results_root)
    metrics_paths = sorted(root.rglob("metrics.json"))
    if not metrics_paths:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    rows = [_load_metrics_row(path, root, row_id=index) for index, path in enumerate(metrics_paths, start=1)]
    return pl.from_dicts(rows)


def main() -> None:
    """Print a compact preview for ad-hoc inspection from the CLI."""

    parser = argparse.ArgumentParser(description="Collect metrics.json files from results/ into a Polars DataFrame.")
    parser.add_argument("--results-root", default="results", help="Root directory to scan for metrics.json files.")
    parser.add_argument("--head", type=int, default=10, help="Number of rows to print.")
    args = parser.parse_args()

    frame = collect_results_metrics(args.results_root)
    with pl.Config(tbl_rows=args.head):
        print(frame)


if __name__ == "__main__":
    main()
