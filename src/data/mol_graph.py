"""RDKit + PyG utilities for molecule 3D graph construction keyed by `inchi_key`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch


ATOM_TYPES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
DEGREE_VALUES = [0, 1, 2, 3, 4, 5]
TOTAL_H_VALUES = [0, 1, 2, 3, 4]
DISTANCE_CUTOFF = 4.5
NODE_SCALAR_DIM = 41
NODE_VECTOR_DIM = 1
EDGE_SCALAR_DIM = 16
EDGE_VECTOR_DIM = 1


def _load_rdkit() -> tuple[Any, Any]:
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for molecule graph construction. "
            "Install it in the active environment before running `python dataloader.py ...`."
        ) from exc
    return Chem, AllChem


def _load_pyg_data() -> Any:
    try:
        from torch_geometric.data import Data  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "torch-geometric is required for molecule graph construction and loading. "
            "Install it in the active environment before training or precalculating graphs."
        ) from exc
    return Data


def _one_hot_with_other(value: Any, choices: list[Any]) -> list[float]:
    output = [0.0] * (len(choices) + 1)
    index = len(choices)
    if value in choices:
        index = choices.index(value)
    output[index] = 1.0
    return output


def _build_atom_features(atom: Any, Chem: Any) -> list[float]:
    chiral_tags = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]

    features = []
    features.extend(_one_hot_with_other(atom.GetAtomicNum(), ATOM_TYPES))
    features.extend(_one_hot_with_other(atom.GetTotalDegree(), DEGREE_VALUES))
    features.extend(_one_hot_with_other(atom.GetTotalNumHs(), TOTAL_H_VALUES))
    features.extend(_one_hot_with_other(atom.GetChiralTag(), chiral_tags))
    features.extend(_one_hot_with_other(atom.GetHybridization(), hybridizations))
    features.append(float(atom.GetIsAromatic()))
    features.append(float(atom.IsInRing()))
    features.append(float(atom.GetFormalCharge()) / 5.0)
    features.append(float(atom.GetMass()) / 200.0)
    return features


def _build_bond_features(bond: Any, Chem: Any, distance: float, is_spatial: bool) -> list[float]:
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    stereo_types = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]

    if bond is None:
        bond_type = [0.0] * (len(bond_types) + 1)
        stereo = [0.0] * (len(stereo_types) + 1)
        conjugated = 0.0
        in_ring = 0.0
    else:
        bond_type = _one_hot_with_other(bond.GetBondType(), bond_types)
        stereo = _one_hot_with_other(bond.GetStereo(), stereo_types)
        conjugated = float(bond.GetIsConjugated())
        in_ring = float(bond.IsInRing())

    return bond_type + stereo + [conjugated, in_ring, float(is_spatial), float(distance) / 10.0]


def build_molecule_3d(smiles: str, random_seed: int = 13) -> tuple[Any, np.ndarray]:
    """Generate a deterministic RDKit conformer and return heavy-atom coordinates."""
    Chem, AllChem = _load_rdkit()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    # params.maxAttempts = 10; decraped

    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise ValueError("RDKit conformer embedding failed after deterministic retries.")

    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=200)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)

    mol = Chem.RemoveHs(mol)
    if mol.GetNumConformers() == 0:
        raise ValueError("RDKit conformer generation produced no conformers after hydrogen removal.")

    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    coords = coords - coords.mean(axis=0, keepdims=True)
    return mol, coords


def build_graph_from_smiles(
    smiles: str,
    graph_id: str | None = None,
    distance_cutoff: float = DISTANCE_CUTOFF,
) -> Any:
    """Build a PyG `Data` graph using bond edges plus spatial proximity edges."""
    Chem, _ = _load_rdkit()
    Data = _load_pyg_data()
    mol, coords = build_molecule_3d(smiles)

    node_scalar = torch.tensor(
        [_build_atom_features(atom, Chem) for atom in mol.GetAtoms()],
        dtype=torch.float32,
    )
    node_vector = torch.from_numpy(coords).to(torch.float32).unsqueeze(1)

    bond_lookup: dict[tuple[int, int], Any] = {}
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        key = (min(begin, end), max(begin, end))
        bond_lookup[key] = bond

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_scalar: list[list[float]] = []
    edge_vector: list[np.ndarray] = []

    num_atoms = mol.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            displacement = coords[j] - coords[i]
            distance = float(np.linalg.norm(displacement))
            bond = bond_lookup.get((i, j))
            is_spatial = bond is None and distance <= distance_cutoff
            if bond is None and not is_spatial:
                continue

            feature = _build_bond_features(bond=bond, Chem=Chem, distance=distance, is_spatial=is_spatial)
            unit_vector = displacement / max(distance, 1.0e-8)

            edge_src.extend([i, j])
            edge_dst.extend([j, i])
            edge_scalar.extend([feature, feature])
            edge_vector.extend([unit_vector, -unit_vector])

    if not edge_src:
        zero_feature = _build_bond_features(bond=None, Chem=Chem, distance=0.0, is_spatial=False)
        for atom_index in range(num_atoms):
            edge_src.append(atom_index)
            edge_dst.append(atom_index)
            edge_scalar.append(zero_feature)
            edge_vector.append(np.zeros(3, dtype=np.float32))

    graph = Data(
        node_s=node_scalar,
        node_v=node_vector,
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_s=torch.tensor(edge_scalar, dtype=torch.float32),
        edge_v=torch.tensor(np.asarray(edge_vector).reshape(-1, 1, 3), dtype=torch.float32),
        num_nodes=num_atoms,
    )
    return graph


def default_graph_cache_path(raw_path: str | Path) -> Path:
    raw = Path(raw_path)
    return raw.parent / "graphs" / f"{raw.stem}_graphs.pt"


def default_split_graph_cache_path(split_dir: str | Path) -> Path:
    split_path = Path(split_dir)
    return split_path / "graph_cache.pt"


def iter_unique_compounds(frame: Any, id_column: str, smiles_column: str) -> list[tuple[str, str]]:
    pairs = frame.select([id_column, smiles_column]).unique()
    conflicts = (
        pairs.group_by(id_column)
        .agg(pl.col(smiles_column).n_unique().alias("n_smiles"))
        .filter(pl.col("n_smiles") > 1)
    )
    if conflicts.height > 0:
        example_ids = conflicts.head(5).select(id_column).to_series().to_list()
        print(
            f"[graph-cache] WARNING: Found {conflicts.height} graph ids mapping to multiple SMILES strings. "
            f"Keeping only the first SMILES for each id. Examples of conflicting ids: {example_ids}"
        )
        pairs = pairs.unique(subset=id_column, keep="first")

    return list(
        zip(
            pairs[id_column].cast(pl.Utf8).to_list(),
            pairs[smiles_column].cast(pl.Utf8).to_list(),
        )
    )


def build_graph_store_from_table(
    frame: Any,
    id_column: str = "inchi_key",
    smiles_column: str = "smiles",
    distance_cutoff: float = DISTANCE_CUTOFF,
    limit: int | None = None,
    progress_every: int = 25000,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Build a graph store for the unique compounds referenced by a table."""
    compounds = iter_unique_compounds(frame, id_column=id_column, smiles_column=smiles_column)
    if limit is not None:
        compounds = compounds[:limit]

    graph_store: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    total = len(compounds)

    for index, (graph_id, smiles) in enumerate(compounds, start=1):
        try:
            graph_store[graph_id] = build_graph_from_smiles(
                smiles=smiles,
                graph_id=graph_id,
                distance_cutoff=float(distance_cutoff),
            )
        except Exception as exc:  # pragma: no cover - surfaced through caller manifest
            failures.append({"inchi_key": graph_id, "smiles": smiles, "error": str(exc)})

        if index == total or index % progress_every == 0:
            print(f"[graph-cache] processed {index}/{total}")

    return graph_store, failures


def save_graph_store(
    graph_store: dict[str, Any],
    output_path: str | Path,
    source_path: str | Path,
    distance_cutoff: float,
    graph_id_column: str = "inchi_key",
    failures: list[dict[str, str]] | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "graphs": graph_store,
        "metadata": {
            "source_path": str(source_path),
            "graph_id_column": graph_id_column,
            "node_scalar_dim": graph_store[next(iter(graph_store))].node_s.shape[-1] if graph_store else 0,
            "node_vector_dim": NODE_VECTOR_DIM,
            "edge_scalar_dim": graph_store[next(iter(graph_store))].edge_s.shape[-1] if graph_store else 0,
            "edge_vector_dim": EDGE_VECTOR_DIM,
            "distance_cutoff": distance_cutoff,
            "num_graphs": len(graph_store),
            "failures": failures or [],
        },
    }
    torch.save(payload, output)


def load_graph_store(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    graph_path = Path(path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph cache not found: {graph_path}")
    try:
        payload = torch.load(graph_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(graph_path, map_location="cpu")

    if isinstance(payload, dict) and "graphs" in payload:
        return payload["graphs"], payload.get("metadata", {})
    if isinstance(payload, dict):
        return payload, {}
    raise TypeError(f"Unexpected graph cache payload type: {type(payload)!r}")
