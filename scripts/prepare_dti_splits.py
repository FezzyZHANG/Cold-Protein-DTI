#!/usr/bin/env python
"""Prepare train/val/test split files directly from a raw DTI parquet or CSV table."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from split_utils import cold_protein_split, dti_set_stats, naive_random_split


def read_table(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    raise ValueError(f"Unsupported input file format: {path.suffix}")


def write_manifest(
    output_dir: Path,
    input_path: Path,
    mode: str,
    seed: int,
    subsample_n: int | None,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    dropped_df: pl.DataFrame,
) -> None:
    summary = {
        "input_path": str(input_path),
        "mode": mode,
        "seed": seed,
        "subsample_n": subsample_n,
        "train": {
            "n_rows": train_df.height,
            "n_drugs": int(train_df.select("inchi_key").n_unique()),
            "n_proteins": int(train_df.select("target_uniprot_id").n_unique()),
        },
        "val": {
            "n_rows": val_df.height,
            "n_drugs": int(val_df.select("inchi_key").n_unique()),
            "n_proteins": int(val_df.select("target_uniprot_id").n_unique()),
        },
        "test": {
            "n_rows": test_df.height,
            "n_drugs": int(test_df.select("inchi_key").n_unique()),
            "n_proteins": int(test_df.select("target_uniprot_id").n_unique()),
        },
        "dropped_rows": dropped_df.height,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DTI split parquet files from a raw table.")
    parser.add_argument("--input-path", required=True, help="Raw DTI table path (.parquet or .csv).")
    parser.add_argument("--output-dir", required=True, help="Directory to write train/val/test outputs.")
    parser.add_argument("--mode", default="cp-easy", choices=["naive", "cp-easy", "cp-hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample-n", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--min-dti-per-chem", type=int, default=5)
    parser.add_argument("--min-dti-per-protein", type=int, default=5)
    parser.add_argument("--max-extreme-ratio", type=float, default=10.0)
    parser.add_argument("--filter-max-iters", type=int, default=20)
    parser.add_argument("--cp-hard-verbose", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = read_table(input_path)

    split_kwargs = {
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "apply_scope_filter": True,
        "min_dti_per_chem": args.min_dti_per_chem,
        "min_dti_per_protein": args.min_dti_per_protein,
        "max_extrame_ratio": args.max_extreme_ratio,
        "filter_max_iters": args.filter_max_iters,
        "subsample_n": args.subsample_n,
    }

    try:
        if args.mode == "naive":
            train_df, val_df, test_df, dropped_df = naive_random_split(table, **split_kwargs)
        else:
            train_df, val_df, test_df, dropped_df = cold_protein_split(
                table,
                similarity_based=args.mode == "cp-hard",
                similarity_mode_config={"verbose": True} if args.cp_hard_verbose else None,
                **split_kwargs,
            )
    except ImportError as exc:
        raise SystemExit(
            "Split preparation requires optional dependencies that are not installed. "
            "For `cp-hard`, install the ESM extra with `uv sync --extra esm` and make sure "
            "the required pretrained assets are staged locally."
        ) from exc

    train_df.write_parquet(output_dir / "train.parquet")
    val_df.write_parquet(output_dir / "val.parquet")
    test_df.write_parquet(output_dir / "test.parquet")

    stats_df = pl.concat(
        [
            dti_set_stats(train_df).with_columns(pl.lit("train").alias("split")),
            dti_set_stats(val_df).with_columns(pl.lit("val").alias("split")),
            dti_set_stats(test_df).with_columns(pl.lit("test").alias("split")),
        ],
        how="vertical",
    )
    stats_df.write_csv(output_dir / "stats.csv")

    if dropped_df.height > 0:
        dropped_df.write_parquet(output_dir / "dropped.parquet")

    write_manifest(
        output_dir=output_dir,
        input_path=input_path,
        mode=args.mode,
        seed=args.seed,
        subsample_n=args.subsample_n,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        dropped_df=dropped_df,
    )

    print(f"[split] wrote split files to: {output_dir}")


if __name__ == "__main__":
    main()
