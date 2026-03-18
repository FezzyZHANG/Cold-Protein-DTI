"""Dataset and collate definitions for graph-backed cold-protein DTI experiments."""

from __future__ import annotations

from typing import Any

import polars as pl
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
AMINO_ACID_VOCAB = {
    char: idx
    for idx, char in enumerate(list("ACDEFGHIKLMNPQRSTVWYBXZJUO"), start=2)
}


def _encode_text(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    encoded = [vocab.get(char, UNK_TOKEN_ID) for char in text[:max_length]]
    if len(encoded) < max_length:
        encoded.extend([PAD_TOKEN_ID] * (max_length - len(encoded)))
    return encoded


class DTIDataset(Dataset):
    """A split dataset that joins protein sequences with precomputed molecule graphs."""

    def __init__(self, frame: pl.DataFrame, graph_store: dict[str, Any], max_protein_length: int) -> None:
        required = {"smiles", "sequence", "label", "inchi_key", "target_uniprot_id"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        self.sequences = frame["sequence"].cast(pl.Utf8).to_list()
        self.labels = frame["label"].cast(pl.Float32).to_list()
        self.inchi_keys = frame["inchi_key"].cast(pl.Utf8).to_list()
        self.target_ids = frame["target_uniprot_id"].cast(pl.Utf8).to_list()
        self.graph_store = graph_store
        self.max_protein_length = int(max_protein_length)

        missing_graphs = sorted(set(self.inchi_keys) - set(self.graph_store))
        if missing_graphs:
            preview = missing_graphs[:5]
            raise KeyError(
                "The graph cache is missing one or more compounds referenced by the split files. "
                f"Examples: {preview}"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence = self.sequences[index]
        inchi_key = self.inchi_keys[index]
        graph = self.graph_store[inchi_key].clone()
        return {
            "drug_graph": graph,
            "protein_tokens": torch.tensor(
                _encode_text(sequence, AMINO_ACID_VOCAB, self.max_protein_length),
                dtype=torch.long,
            ),
            "protein_sequence": sequence,
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
            "inchi_key": inchi_key,
            "target_uniprot_id": self.target_ids[index],
        }


def collate_dti_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    protein_tokens = torch.stack([item["protein_tokens"] for item in items], dim=0)
    graph_batch = Batch.from_data_list([item["drug_graph"] for item in items])

    return {
        "drug_graph_batch": graph_batch,
        "protein_tokens": protein_tokens,
        "protein_mask": protein_tokens.ne(PAD_TOKEN_ID),
        "protein_sequences": [item["protein_sequence"] for item in items],
        "labels": torch.stack([item["label"] for item in items], dim=0),
        "inchi_keys": [item["inchi_key"] for item in items],
        "target_uniprot_ids": [item["target_uniprot_id"] for item in items],
    }
