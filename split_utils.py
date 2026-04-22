"""
Utilities for filtering DTI tables, building dataset splits, and (optionally)
computing ESM-2 protein embeddings for similarity-aware cold-protein splitting.

This script supports three split modes:
- naive: random row-level split
- cp-easy: cold-protein split by deterministic protein partitioning
- cp-hard: cold-protein split with iterative embedding-similarity optimization
"""

from __future__ import annotations
import polars as pl
import os
import argparse
import importlib
from typing import Any

np = importlib.import_module("numpy")

def chem_stats(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("inchi_key")
          .agg(
              pl.len().alias("n"),
              pl.col("label").sum().alias("n_pos"),
              (pl.col("label").mean()).alias("pos_ratio"),
              pl.col("target_uniprot_id").n_unique().alias("n_unique_prot"),
          )
          .with_columns(
              (pl.col("n") - pl.col("n_pos")).alias("n_neg")
          )
          .sort("n", descending=True)
    )

def dti_set_stats(df: pl.DataFrame) -> pl.DataFrame:
    # provide overall DTI set statistics: n_dti, n_chem, n_prot, pos_ratio
    pos_ratio = 0.0 if df.height == 0 else float(df["label"].mean())
    return pl.DataFrame(
        {
            "n_dti": df.height,
            "n_chem": df.select("inchi_key").n_unique(),
            "n_prot": df.select("target_uniprot_id").n_unique(),
            "pos_ratio": pos_ratio,
        },
        schema={"n_dti": pl.Int64, "n_chem": pl.Int64, "n_prot": pl.Int64, "pos_ratio": pl.Float64},
    )


def _resolve_split_counts(n_items: int, train_frac: float, val_frac: float, test_frac: float) -> tuple[int, int, int]:
    n_train = int(n_items * train_frac)
    n_val = int(n_items * val_frac)
    n_test = n_items - n_train - n_val

    if n_items >= 3:
        if n_val == 0:
            n_val = 1
            n_train = max(1, n_train - 1)
        if n_test == 0:
            n_test = 1
            n_train = max(1, n_train - 1)
        if n_train + n_val + n_test != n_items:
            n_train = n_items - n_val - n_test
    return n_train, n_val, n_test


def _deterministic_subsample_rows(df: pl.DataFrame, n_rows: int, seed: int) -> pl.DataFrame:
    """Deterministically subsample rows using hash order."""
    if n_rows >= df.height:
        return df
    if n_rows < 1:
        raise ValueError("subsample_n must be >= 1 when provided")

    return (
        df.with_row_index("_row_id")
          .with_columns(pl.col("_row_id").hash(seed=seed).alias("_ord"))
          .sort("_ord")
          .slice(0, n_rows)
          .drop(["_row_id", "_ord"])
    )


def _downsample_extreme_ratio_by_group(
    df: pl.DataFrame,
    group_col: str,
    max_extrame_ratio: float,
    seed: int,
) -> pl.DataFrame:
    """Randomly drop majority-label rows within groups until both label ratios are below the threshold."""
    if df.height == 0:
        return df

    stats = (
        df.group_by(group_col)
          .agg(
              pl.len().alias("_n"),
              pl.col("label").sum().cast(pl.Int64).alias("_n_pos"),
          )
          .with_columns((pl.col("_n") - pl.col("_n_pos")).alias("_n_neg"))
    )

    d = (
        df.with_columns(pl.col("_row_id").hash(seed=seed).alias("_balance_ord"))
          .sort([group_col, "label", "_balance_ord"])
          .with_columns((pl.col("_row_id").cum_count().over([group_col, "label"]) - 1).alias("_class_rank"))
          .join(stats, on=group_col, how="left")
          .with_columns(
              (pl.col("_n_neg") > pl.col("_n_pos")).alias("_neg_is_majority"),
              (pl.col("_n_pos") > pl.col("_n_neg")).alias("_pos_is_majority"),
              ((pl.col("_n_pos") * max_extrame_ratio).ceil().cast(pl.Int64) - 1).alias("_max_neg"),
              ((pl.col("_n_neg") * max_extrame_ratio).ceil().cast(pl.Int64) - 1).alias("_max_pos"),
          )
          .with_columns(
              pl.when((pl.col("_n_pos") == 0) | (pl.col("_n_neg") == 0))
                .then(False)
                .when(
                    (pl.col("_n_pos") == pl.col("_n_neg"))
                    & ((pl.col("_n_neg") / pl.col("_n_pos")) >= max_extrame_ratio)
                    & ((pl.col("_n_pos") / pl.col("_n_neg")) >= max_extrame_ratio)
                )
                .then(False)
                .when(
                    (pl.col("label") == 0)
                    & pl.col("_neg_is_majority")
                    & ((pl.col("_n_neg") / pl.col("_n_pos")) >= max_extrame_ratio)
                )
                .then(pl.col("_class_rank") < pl.col("_max_neg"))
                .when(
                    (pl.col("label") == 1)
                    & pl.col("_pos_is_majority")
                    & ((pl.col("_n_pos") / pl.col("_n_neg")) >= max_extrame_ratio)
                )
                .then(pl.col("_class_rank") < pl.col("_max_pos"))
                .otherwise(True)
                .alias("_keep_balance")
          )
          .filter(pl.col("_keep_balance"))
          .drop(
              [
                  "_balance_ord",
                  "_class_rank",
                  "_n",
                  "_n_pos",
                  "_n_neg",
                  "_neg_is_majority",
                  "_pos_is_majority",
                  "_max_neg",
                  "_max_pos",
                  "_keep_balance",
              ]
          )
    )
    return d


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance in [0, 2] with NumPy acceleration."""
    a_np = np.asarray(a, dtype=np.float32)
    b_np = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a_np))
    nb = float(np.linalg.norm(b_np))
    if na == 0.0 or nb == 0.0:
        return 1.0
    cos_sim = float(np.dot(a_np, b_np) / (na * nb))
    return float(1.0 - cos_sim)


def _normalize_rows(x: Any) -> Any:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def _similarity_mode_preset(n_proteins: int) -> dict[str, Any]:
    """
    Internal preset for similarity-based split.

    Tuned preset for medium-scale datasets (around 633 unique proteins):
    - moderate threshold for train-vs-non-train boundary
    - bounded candidate search for runtime control
    - larger ESM batch size for embedding throughput
    """
    cpu = os.cpu_count() or 4

    if n_proteins >= 600:
        return {
            "threshold": 0.12,
            "max_iters": 12,
            "esm_model_name": "esm2_t33_650M_UR50D",
            "esm_repr_layer": None,
            "esm_batch_size": 16,
            "esm_device": None,
            "esm_normalize": True,
            "verbose": False,
        }

    # default small/medium fallback
    return {
        "threshold": 0.15,
        "max_iters": 15,
        "esm_model_name": "esm2_t33_650M_UR50D",
        "esm_repr_layer": None,
        "esm_batch_size": 8,
        "esm_device": None,
        "esm_normalize": True,
        "verbose": False,
    }


def iterative_similarity_cold_protein_settings(
    protein_meta: pl.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    similarity_mode_config: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """
    Iterative k-means-like cold-protein split settings (2 means).

    Steps:
    1) Input protein metadata (`target_uniprot_id`, `sequence`)
    2) Compute ESM-2 embeddings
     3) Build initial split (train vs non-train)
     4) Iteratively update 2 centroids (train/non-train) and reassign proteins
         (with fixed train count) until threshold is reached or convergence.
    5) Return protein split settings DataFrame: (`target_uniprot_id`, `split`)
    """
    required = {"target_uniprot_id", "sequence"}
    missing = required - set(protein_meta.columns)
    if missing:
        raise ValueError(f"Missing required columns for similarity split: {sorted(missing)}")

    cfg = _similarity_mode_preset(protein_meta.select("target_uniprot_id").n_unique())
    if similarity_mode_config:
        cfg.update(similarity_mode_config)

    proteins = protein_meta.select(["target_uniprot_id", "sequence"]).unique()
    emb_df = build_esm2_embedding_table(
        proteins,
        model_name=cfg["esm_model_name"],
        repr_layer=cfg["esm_repr_layer"],
        batch_size=cfg["esm_batch_size"],
        device=cfg["esm_device"],
        normalize=cfg["esm_normalize"],
    )

    if cfg["verbose"]:
        print(f"Computed embeddings for {emb_df.height} proteins. Starting 2-means optimization...")

    ordered = (
        emb_df.select(["target_uniprot_id", "embedding"])
              .with_columns(pl.col("target_uniprot_id").hash(seed=seed).alias("_ord"))
              .sort("_ord")
    )

    pids = ordered["target_uniprot_id"].to_list()
    emb = np.asarray(ordered["embedding"].to_list(), dtype=np.float32)
    emb = _normalize_rows(emb)

    n_prot = len(pids)
    n_train, n_val, n_test = _resolve_split_counts(n_prot, train_frac, val_frac, test_frac)
    n_nontrain = n_prot - n_train

    # Initial assignment from deterministic order.
    is_train = np.zeros(n_prot, dtype=bool)
    is_train[:n_train] = True

    threshold = float(cfg["threshold"])
    max_iters = int(cfg["max_iters"])

    for i in range(max_iters):
        train_emb = emb[is_train]
        nontrain_emb = emb[~is_train]
        if train_emb.shape[0] == 0 or nontrain_emb.shape[0] == 0:
            break

        c_train = train_emb.mean(axis=0, keepdims=True)
        c_non = nontrain_emb.mean(axis=0, keepdims=True)
        c_train = _normalize_rows(c_train)[0]
        c_non = _normalize_rows(c_non)[0]

        # boundary quality: centroid cosine distance
        centroid_sep = float(1.0 - np.dot(c_train, c_non))

        if cfg["verbose"] and i == 0:
            print(f"Initial centroid separation: {centroid_sep:.4f}")

        # Reassign with fixed train size using relative score.
        d_train = 1.0 - (emb @ c_train)
        d_non = 1.0 - (emb @ c_non)
        score = d_train - d_non  # smaller => closer to train centroid
        order = np.argsort(score, kind="mergesort")
        new_is_train = np.zeros(n_prot, dtype=bool)
        new_is_train[order[:n_train]] = True

        changed = int(np.sum(new_is_train != is_train))
        is_train = new_is_train

        if cfg["verbose"]:
            print(f"Iter {i+1}: centroid_sep={centroid_sep:.4f}, changed={changed}")

        if centroid_sep >= threshold or changed == 0:
            break

    train_ids = [pids[idx] for idx in np.where(is_train)[0]]
    nontrain_ids = [pids[idx] for idx in np.where(~is_train)[0]]

    # Deterministic val/test split from non-train pool.
    non_df = pl.DataFrame({"target_uniprot_id": nontrain_ids})
    non_df = non_df.with_columns(pl.col("target_uniprot_id").hash(seed=seed).alias("_ord")).sort("_ord")
    val_ids = non_df.slice(0, n_val)["target_uniprot_id"].to_list()
    test_ids = non_df.slice(n_val, n_nontrain - n_val)["target_uniprot_id"].to_list()

    if cfg["verbose"]:
        # Optional debug export: project embeddings to 2D with UMAP and save split labels.
        try:
            umap = importlib.import_module("umap")
            reducer = umap.UMAP(n_components=2, random_state=seed)
            emb_2d = reducer.fit_transform(emb)

            split_map = {pid: "train" for pid in train_ids}
            split_map.update({pid: "val" for pid in val_ids})
            split_map.update({pid: "test" for pid in test_ids})

            debug_df = pl.DataFrame(
                {
                    "target_uniprot_id": pids,
                    "is_train": is_train.tolist(),
                    "split": [split_map.get(pid, "test") for pid in pids],
                    "umap_x": emb_2d[:, 0].astype(np.float32).tolist(),
                    "umap_y": emb_2d[:, 1].astype(np.float32).tolist(),
                }
            )

            os.makedirs("data", exist_ok=True)
            debug_path = os.path.join("data", f"cp_hard_umap_seed{seed}.parquet")
            debug_df.write_parquet(debug_path)
            print(f"Saved UMAP split debug table to: {debug_path}")
        except ImportError:
            print("Verbose UMAP export skipped: package `umap-learn` is not installed.")
        except Exception as e:
            print(f"Verbose UMAP export failed: {e}")

    return pl.DataFrame(
        {
            "target_uniprot_id": train_ids + val_ids + test_ids,
            "split": ["train"] * len(train_ids) + ["val"] * len(val_ids) + ["test"] * len(test_ids),
        }
    )


def scopeDTI_chem_filter(
    table: pl.DataFrame,
    min_dti_per_chem: int = 1,
    min_dti_per_protein: int = 1,
    max_extrame_ratio: float | None = None,
    max_iters: int = 20,
    subsample_n: int | None = None,
    subsample_seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Filter DTI table by chemistry/protein support and label imbalance.

    Required input columns:
    - `inchi_key`, `target_uniprot_id`, `smiles`, `sequence`, `label`

    Rules:
    1) keep entities with at least min DTIs (chem and protein)
    2) if provided, try to satisfy label balance per chem/protein by randomly
       dropping majority-label rows until both n_neg / n_pos and n_pos / n_neg
       are below max_extrame_ratio

    Returns:
    - (filtered_table, dropped_table)
    """
    required_cols = {"inchi_key", "target_uniprot_id", "smiles", "sequence", "label"}
    missing = required_cols - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if min_dti_per_chem < 1 or min_dti_per_protein < 1:
        raise ValueError("min_dti_per_chem and min_dti_per_protein must be >= 1")
    if max_extrame_ratio is not None and max_extrame_ratio <= 0:
        raise ValueError("max_extrame_ratio must be > 0 when provided")
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1")
    if subsample_n is not None and subsample_n < 1:
        raise ValueError("subsample_n must be >= 1 when provided")

    if subsample_n is not None:
        table = _deterministic_subsample_rows(table, subsample_n, subsample_seed)

    table0 = table.with_row_index("_row_id")
    current = table0

    for iter_idx in range(max_iters):
        prev_height = current.height

        chem_s = (
            current.group_by("inchi_key")
                   .agg(
                       pl.len().alias("_n"),
                       pl.col("label").sum().cast(pl.Int64).alias("_n_pos"),
                   )
                   .with_columns((pl.col("_n") - pl.col("_n_pos")).alias("_n_neg"))
        )
        chem_keep_expr = pl.col("_n") >= min_dti_per_chem
        keep_chems = chem_s.filter(chem_keep_expr).select("inchi_key")
        current = current.join(keep_chems, on="inchi_key", how="inner")

        if max_extrame_ratio is not None:
            current = _downsample_extreme_ratio_by_group(
                current,
                group_col="inchi_key",
                max_extrame_ratio=max_extrame_ratio,
                seed=subsample_seed + iter_idx * 2,
            )

        prot_s = (
            current.group_by("target_uniprot_id")
                   .agg(
                       pl.len().alias("_n"),
                       pl.col("label").sum().cast(pl.Int64).alias("_n_pos"),
                   )
                   .with_columns((pl.col("_n") - pl.col("_n_pos")).alias("_n_neg"))
        )
        prot_keep_expr = pl.col("_n") >= min_dti_per_protein
        keep_prots = prot_s.filter(prot_keep_expr).select("target_uniprot_id")
        current = current.join(keep_prots, on="target_uniprot_id", how="inner")

        if max_extrame_ratio is not None:
            current = _downsample_extreme_ratio_by_group(
                current,
                group_col="target_uniprot_id",
                max_extrame_ratio=max_extrame_ratio,
                seed=subsample_seed + iter_idx * 2 + 1,
            )

        if current.height == prev_height:
            # print(f"Filtering converged after {_+1} iterations. Remaining rows: {current.height}")
            break

    dropped = table0.join(current.select("_row_id"), on="_row_id", how="anti").drop("_row_id")
    return current.drop("_row_id"), dropped

def cold_protein_split(
    df: pl.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    apply_scope_filter: bool = True,
    min_dti_per_chem: int = 1,
    min_dti_per_protein: int = 1,
    max_extrame_ratio: float | None = None,
    filter_max_iters: int = 20,
    subsample_n: int | None = None,
    subsample_seed: int = 0,
    similarity_based: bool = False,
    similarity_mode_config: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Cold-protein split.

    Returns: (train, val, test, dropped)
    Constraint: `target_uniprot_id` is disjoint across train / val / test.
    """

    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    if "target_uniprot_id" not in df.columns:
        raise ValueError("Input DataFrame must contain column: target_uniprot_id")
    if subsample_n is not None and subsample_n < 1:
        raise ValueError("subsample_n must be >= 1 when provided")

    if subsample_n is not None and not apply_scope_filter:
        df = _deterministic_subsample_rows(df, subsample_n, subsample_seed)

    dropped = pl.DataFrame(schema=df.schema)
    if apply_scope_filter:
        df, dropped = scopeDTI_chem_filter(
            table=df,
            min_dti_per_chem=min_dti_per_chem,
            min_dti_per_protein=min_dti_per_protein,
            max_extrame_ratio=max_extrame_ratio,
            max_iters=filter_max_iters,
            subsample_n=subsample_n,
            subsample_seed=subsample_seed,
        )

    # Similarity-based iterative mode returns split settings directly.
    if similarity_based:
        if "sequence" not in df.columns:
            raise ValueError("similarity_based=True requires `sequence` column")
        protein_meta = df.select(["target_uniprot_id", "sequence"]).unique()
        settings = iterative_similarity_cold_protein_settings(
            protein_meta=protein_meta,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
            similarity_mode_config=similarity_mode_config,
        )

        train_ids = settings.filter(pl.col("split") == "train").select("target_uniprot_id")
        val_ids = settings.filter(pl.col("split") == "val").select("target_uniprot_id")
        test_ids = settings.filter(pl.col("split") == "test").select("target_uniprot_id")

        train = df.join(train_ids, on="target_uniprot_id", how="inner")
        val = df.join(val_ids, on="target_uniprot_id", how="inner")
        test = df.join(test_ids, on="target_uniprot_id", how="inner")
    else:
        prot_counts = (
            df.group_by("target_uniprot_id")
              .agg(pl.len().alias("n_rows"))
              .with_columns(pl.col("target_uniprot_id").hash(seed=seed).alias("_ord"))
              .sort("_ord")
        )

        n_prot = prot_counts.height
        n_train, n_val, n_test = _resolve_split_counts(n_prot, train_frac, val_frac, test_frac)

        train_ids = prot_counts.slice(0, n_train).select("target_uniprot_id")
        val_ids = prot_counts.slice(n_train, n_val).select("target_uniprot_id")
        test_ids = prot_counts.slice(n_train + n_val, n_test).select("target_uniprot_id")

        train = df.join(train_ids, on="target_uniprot_id", how="inner")
        val = df.join(val_ids, on="target_uniprot_id", how="inner")
        test = df.join(test_ids, on="target_uniprot_id", how="inner")

    # Sanity check: disjoint proteins across splits.
    tr_set = set(train.select("target_uniprot_id").unique().to_series().to_list())
    va_set = set(val.select("target_uniprot_id").unique().to_series().to_list())
    te_set = set(test.select("target_uniprot_id").unique().to_series().to_list())
    if (tr_set & va_set) or (tr_set & te_set) or (va_set & te_set):
        raise RuntimeError("Cold-protein split failed: protein overlap detected between splits.")
    
    return train, val, test, dropped


def naive_random_split(
    df: pl.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    apply_scope_filter: bool = True,
    min_dti_per_chem: int = 1,
    min_dti_per_protein: int = 1,
    max_extrame_ratio: float | None = None,
    filter_max_iters: int = 20,
    subsample_n: int | None = None,
    subsample_seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Naive random row-level split with optional scopeDTI filter.

    Returns: (train, val, test, dropped)
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    if subsample_n is not None and subsample_n < 1:
        raise ValueError("subsample_n must be >= 1 when provided")

    dropped = pl.DataFrame(schema=df.schema)
    if apply_scope_filter:
        df, dropped = scopeDTI_chem_filter(
            table=df,
            min_dti_per_chem=min_dti_per_chem,
            min_dti_per_protein=min_dti_per_protein,
            max_extrame_ratio=max_extrame_ratio,
            max_iters=filter_max_iters,
            subsample_n=subsample_n,
            subsample_seed=subsample_seed,
        )
    elif subsample_n is not None:
        df = _deterministic_subsample_rows(df, subsample_n, subsample_seed)

    d = (
        df.with_row_index("_row_id")
          .with_columns(pl.col("_row_id").hash(seed=seed).alias("_ord"))
          .sort("_ord")
    )

    n_rows = d.height
    n_train, n_val, n_test = _resolve_split_counts(n_rows, train_frac, val_frac, test_frac)

    train = d.slice(0, n_train).drop(["_row_id", "_ord"])
    val = d.slice(n_train, n_val).drop(["_row_id", "_ord"])
    test = d.slice(n_train + n_val, n_test).drop(["_row_id", "_ord"])
    return train, val, test, dropped


def build_esm2_embedding_table(
    protein_df: pl.DataFrame,
    model_name: str = "esm2_t33_650M_UR50D",
    repr_layer: int | None = None,
    batch_size: int = 8,
    device: str | None = None,
    normalize: bool = False,
) -> pl.DataFrame:
    
    MAX_SEQ_LEN = 256  # ESM-2 max sequence length (including BOS/EOS)

    """
    Build an ESM-2 embedding table from a protein table.

    Input:
    - DataFrame with columns: `target_uniprot_id`, `sequence`

    Output:
    - DataFrame with columns: `target_uniprot_id`, `sequence`, `embedding`
      where `embedding` is a list[float] (sequence-level mean pooled embedding).
    """
    required_cols = {"target_uniprot_id", "sequence"}
    missing = required_cols - set(protein_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Make unique pair table and enforce one sequence per protein id.
    pairs = protein_df.select(["target_uniprot_id", "sequence"]).unique()
    conflicts = (
        pairs.group_by("target_uniprot_id")
             .agg(pl.col("sequence").n_unique().alias("n_seq"))
             .filter(pl.col("n_seq") > 1)
    )
    if conflicts.height > 0:
        raise ValueError(
            "Each target_uniprot_id must map to a single unique sequence. "
            f"Found {conflicts.height} conflicting IDs."
        )

    if pairs.filter(pl.col("sequence").is_null()).height > 0:
        raise ValueError("Null sequence values found.")

    pairs = pairs.with_columns(pl.col("sequence").cast(pl.Utf8))
    if pairs.filter(pl.col("sequence").str.len_chars() == 0).height > 0:
        raise ValueError("Empty sequence values found.")

    # Lazy import to avoid hard dependency for users who only need splitting.
    try:
        torch = importlib.import_module("torch")
    except ImportError as e:
        raise ImportError(
            "ESM-2 embedding requires `torch`. Please install the ESM extra "
            "with `uv sync --extra esm`."
        ) from e

    def _rows_from_huggingface_backbone() -> pl.DataFrame:
        from src.model.esm_support import load_esm_backbone

        loaded = load_esm_backbone(model_name=model_name, prefer_staged_artifacts=True, backend="huggingface")
        model = loaded.backbone.eval()

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(resolved_device)

        resolved_layer = int(loaded.num_layers if repr_layer is None else repr_layer)
        if resolved_layer < 0 or resolved_layer > int(loaded.num_layers):
            raise ValueError(
                f"repr_layer must be between 0 and {loaded.num_layers} for {model_name}; got {resolved_layer}."
            )

        rows: list[dict[str, Any]] = []
        ids = pairs["target_uniprot_id"].to_list()
        seqs = pairs["sequence"].to_list()

        print(
            "Computing ESM-2 embeddings for "
            f"{len(ids)} proteins using staged model {model_name} on device {resolved_device}..."
        )

        try:
            tqdm = importlib.import_module("tqdm")
            iter_range = tqdm.tqdm(range(0, len(ids), batch_size), desc="Computing ESM-2 embeddings")
        except ImportError:
            iter_range = range(0, len(ids), batch_size)

        tokenizer = loaded.tokenizer
        for start in iter_range:
            end = min(start + batch_size, len(ids))
            batch_ids = ids[start:end]
            batch_seqs = seqs[start:end]

            encoded = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_special_tokens_mask=True,
            )
            input_ids = encoded["input_ids"].to(resolved_device)
            attention_mask = encoded["attention_mask"].to(resolved_device)
            special_tokens_mask = encoded["special_tokens_mask"].to(resolved_device).bool()

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                token_reps = out.hidden_states[resolved_layer]

            valid_mask = attention_mask.bool() & ~special_tokens_mask
            for i, pid in enumerate(batch_ids):
                residue_reps = token_reps[i][valid_mask[i]]
                if residue_reps.shape[0] == 0:
                    raise ValueError(f"Sequence for protein {pid} produced no residue tokens after tokenization.")
                emb = residue_reps.mean(0)
                if normalize:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=0)

                rows.append(
                    {
                        "target_uniprot_id": pid,
                        "sequence": batch_seqs[i],
                        "embedding": emb.detach().cpu().to(torch.float32).tolist(),
                    }
                )

        return pl.DataFrame(rows)

    def _rows_from_fair_esm() -> pl.DataFrame:
        try:
            esm = importlib.import_module("esm")
        except ImportError as e:
            raise ImportError(
                "ESM-2 embedding requires either staged Hugging Face ESM assets "
                "or the legacy `fair-esm` package."
            ) from e

        pretrained = getattr(esm, "pretrained", None)
        if pretrained is None:
            raise AttributeError("The installed `esm` package does not provide `esm.pretrained`.")

        load_fn = getattr(pretrained, model_name, None)
        if load_fn is None:
            raise ValueError(f"Unknown ESM-2 model for legacy fair-esm loader: {model_name}")

        model, alphabet = load_fn()
        model.eval()

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(resolved_device)

        resolved_layer = int(getattr(model, "num_layers", 33) if repr_layer is None else repr_layer)
        batch_converter = alphabet.get_batch_converter()

        rows: list[dict[str, Any]] = []
        ids = pairs["target_uniprot_id"].to_list()
        seqs = pairs["sequence"].to_list()
        seqs = [s if len(s) <= MAX_SEQ_LEN - 2 else s[:MAX_SEQ_LEN - 2] for s in seqs]

        print(f"Computing ESM-2 embeddings for {len(ids)} proteins using model {model_name} on device {resolved_device}...")

        try:
            tqdm = importlib.import_module("tqdm")
            iter_range = tqdm.tqdm(range(0, len(ids), batch_size), desc="Computing ESM-2 embeddings")
        except ImportError:
            iter_range = range(0, len(ids), batch_size)

        for start in iter_range:
            end = min(start + batch_size, len(ids))
            batch_data = [(ids[i], seqs[i]) for i in range(start, end)]

            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(resolved_device)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            with torch.no_grad():
                out = model(batch_tokens, repr_layers=[resolved_layer], return_contacts=False)
                token_reps = out["representations"][resolved_layer]

            for i, (pid, seq) in enumerate(batch_data):
                seq_len = int(batch_lens[i].item())
                # Exclude BOS/EOS tokens.
                residue_reps = token_reps[i, 1:seq_len - 1]
                emb = residue_reps.mean(0)
                if normalize:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=0)

                rows.append(
                    {
                        "target_uniprot_id": pid,
                        "sequence": seq,
                        "embedding": emb.detach().cpu().to(torch.float32).tolist(),
                    }
                )

        return pl.DataFrame(rows)

    try:
        return _rows_from_huggingface_backbone()
    except (FileNotFoundError, RuntimeError, ImportError, ValueError) as hf_exc:
        try:
            return _rows_from_fair_esm()
        except (ImportError, AttributeError, ValueError) as fair_exc:
            raise RuntimeError(
                "Unable to load ESM-2 embeddings. Tried the staged Hugging Face artifact loader "
                f"for `{model_name}` and the legacy fair-esm `esm.pretrained` loader. "
                f"Staged-loader error: {hf_exc}. Legacy-loader error: {fair_exc}."
            ) from fair_exc


def split_dataset(
    input_dir: str,
    output_dir: str,
    seed: int = 42,
    subsample_n: int | None = None,
    mode: str = "cp-easy",
    cp_hard_verbose: bool = False,
) -> None:
    """
    Load parquet file from {dataset_name}/{dataset_name}.parquet structure,
    split with cold-protein strategy, and save splits + stats.
    """
    # Infer dataset name from input_dir
    dataset_name = os.path.basename(input_dir.rstrip(os.sep))
    fp = os.path.join(input_dir, dataset_name, f"{dataset_name}.parquet")
    
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Dataset not found at {fp}")
    
    data = pl.read_parquet(fp)

    mode = mode.lower()
    if mode not in {"naive", "cp-easy", "cp-hard"}:
        raise ValueError("mode must be one of: naive, cp-easy, cp-hard")

    split_kwargs = dict(
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        seed=seed,
        apply_scope_filter=True,
        min_dti_per_chem=5,
        min_dti_per_protein=5,
        max_extrame_ratio=10,
        filter_max_iters=20,
        subsample_n=subsample_n,
    )

    if mode == "naive":
        train_df, val_df, test_df, dropped_df = naive_random_split(data, **split_kwargs)
    else:
        # cp-hard enables sequence-similarity-aware ordering proxy.
        similarity_based = mode == "cp-hard"
        similarity_mode_config = {"verbose": True} if (mode == "cp-hard" and cp_hard_verbose) else None
        train_df, val_df, test_df, dropped_df = cold_protein_split(
            data,
            similarity_based=similarity_based,
            similarity_mode_config=similarity_mode_config,
            **split_kwargs,
        )

    # Compute statistics
    stats_df = pl.concat(
        [
            dti_set_stats(train_df).with_columns(pl.lit("train").alias("split")),
            dti_set_stats(val_df).with_columns(pl.lit("val").alias("split")),
            dti_set_stats(test_df).with_columns(pl.lit("test").alias("split")),
        ],
        how="vertical",
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save splits
    train_df.write_parquet(os.path.join(output_dir, "train.parquet"))
    val_df.write_parquet(os.path.join(output_dir, "val.parquet"))
    test_df.write_parquet(os.path.join(output_dir, "test.parquet"))
    
    # Save statistics
    stats_df.write_csv(os.path.join(output_dir, "stats.csv"))
    
    # Optionally save dropped rows
    if dropped_df.height > 0:
        dropped_df.write_parquet(os.path.join(output_dir, "dropped.parquet"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split chemical-protein dataset with cold-protein strategy.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing data.parquet")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for splits and stats")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--subsample_n", type=int, default=None, help="Optional number of rows to subsample for quick testing")
    parser.add_argument(
        "--mode",
        type=str,
        default="cp-easy",
        choices=["naive", "cp-easy", "cp-hard"],
        help="Split mode: naive (random rows), cp-easy (cold protein), cp-hard (cold protein with similarity_based=True)",
    )
    parser.add_argument(
        "--cp_hard_verbose",
        action="store_true",
        help="Enable cp-hard verbose mode (includes UMAP debug parquet export to data/)",
    )
    args = parser.parse_args()

    split_dataset(
        args.input_dir,
        args.output_dir,
        args.seed,
        args.subsample_n,
        args.mode,
        args.cp_hard_verbose,
    )
