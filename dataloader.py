#!/usr/bin/env python
"""Precalculate RDKit/PyG molecule graphs keyed by `inchi_key`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from src.data.mol_graph import build_graph_store_from_table, default_graph_cache_path, save_graph_store


def read_table(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    raise ValueError(f"Unsupported input file format: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a graph cache from a full DTI dataset. Usage: python dataloader.py path_to_full_dataset"
    )
    parser.add_argument("input_path", help="Full dataset path (.parquet or .csv).")
    parser.add_argument("--output-path", default=None, help="Optional output cache path.")
    parser.add_argument("--id-column", default="inchi_key")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--distance-cutoff", type=float, default=4.5)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick debugging.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else default_graph_cache_path(input_path)
    manifest_path = output_path.with_suffix(".json")

    frame = read_table(input_path)
    graph_store, failures = build_graph_store_from_table(
        frame=frame,
        id_column=args.id_column,
        smiles_column=args.smiles_column,
        distance_cutoff=float(args.distance_cutoff),
        limit=args.limit,
        progress_every=25000,
    )

    save_graph_store(
        graph_store=graph_store,
        output_path=output_path,
        source_path=input_path,
        distance_cutoff=float(args.distance_cutoff),
        graph_id_column=args.id_column,
        failures=failures,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "source_path": str(input_path),
                "output_path": str(output_path),
                "graph_id_column": args.id_column,
                "smiles_column": args.smiles_column,
                "distance_cutoff": args.distance_cutoff,
                "num_graphs": len(graph_store),
                "num_failed_graphs": len(failures),
                "failed_graph_ids": [item.get(args.id_column, item.get("inchi_key")) for item in failures[:100]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if failures:
        print(
            "[graph-cache] skipped compounds with failed graph construction: "
            f"{len(failures)} compounds. Examples: {json.dumps(failures[:5], indent=2)}"
        )
    print(f"[graph-cache] wrote {len(graph_store)} graphs to {output_path}")


if __name__ == "__main__":
    main()
