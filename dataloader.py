#!/usr/bin/env python
"""Precalculate RDKit/PyG molecule graphs keyed by `inchi_key`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from src.data.mol_graph import build_graph_from_smiles, default_graph_cache_path, save_graph_store


def read_table(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    raise ValueError(f"Unsupported input file format: {path.suffix}")


def iter_unique_compounds(frame: pl.DataFrame, id_column: str, smiles_column: str) -> list[tuple[str, str]]:
    pairs = frame.select([id_column, smiles_column]).unique()
    conflicts = (
        pairs.group_by(id_column)
        .agg(pl.col(smiles_column).n_unique().alias("n_smiles"))
        .filter(pl.col("n_smiles") > 1)
    )
    if conflicts.height > 0:
        example_ids = conflicts.head(5).select(id_column).to_series().to_list()
        raise ValueError(
            "Each graph id must map to exactly one SMILES string. "
            f"Found {conflicts.height} conflicting ids, for example: {example_ids}"
        )

    return list(zip(pairs[id_column].cast(pl.Utf8).to_list(), pairs[smiles_column].cast(pl.Utf8).to_list()))


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
    compounds = iter_unique_compounds(frame, id_column=args.id_column, smiles_column=args.smiles_column)
    if args.limit is not None:
        compounds = compounds[: args.limit]

    graph_store: dict[str, object] = {}
    failures: list[dict[str, str]] = []

    total = len(compounds)
    for index, (graph_id, smiles) in enumerate(compounds, start=1):
        try:
            graph_store[graph_id] = build_graph_from_smiles(
                smiles=smiles,
                graph_id=graph_id,
                distance_cutoff=float(args.distance_cutoff),
            )
        except Exception as exc:  # pragma: no cover - surfaced through manifest
            failures.append({"inchi_key": graph_id, "smiles": smiles, "error": str(exc)})

        if index == total or index % 250 == 0:
            print(f"[graph-cache] processed {index}/{total}")

    if failures:
        example = failures[:5]
        raise SystemExit(
            "Graph precalculation failed for one or more molecules. "
            f"Examples: {json.dumps(example, indent=2)}"
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[graph-cache] wrote {len(graph_store)} graphs to {output_path}")


if __name__ == "__main__":
    main()
