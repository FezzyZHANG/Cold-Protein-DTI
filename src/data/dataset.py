"""Dataset and collate definitions for cold-protein DTI experiments."""

from __future__ import annotations

from typing import Any

import polars as pl
import torch
from torch.utils.data import Dataset


PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
SMILES_VOCAB = {
    char: idx
    for idx, char in enumerate(
        list("#%()+-./0123456789:=@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]\\abcdefghijklmnopqrstuvwxyz"),
        start=2,
    )
}
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
    """A lightweight dataset over split parquet/csv files."""

    def __init__(self, frame: pl.DataFrame, max_smiles_length: int, max_protein_length: int) -> None:
        required = {"smiles", "sequence", "label", "inchi_key", "target_uniprot_id"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        self.smiles = frame["smiles"].cast(pl.Utf8).to_list()
        self.sequences = frame["sequence"].cast(pl.Utf8).to_list()
        self.labels = frame["label"].cast(pl.Float32).to_list()
        self.inchi_keys = frame["inchi_key"].cast(pl.Utf8).to_list()
        self.target_ids = frame["target_uniprot_id"].cast(pl.Utf8).to_list()
        self.max_smiles_length = int(max_smiles_length)
        self.max_protein_length = int(max_protein_length)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        smiles = self.smiles[index]
        sequence = self.sequences[index]
        return {
            "smiles_tokens": torch.tensor(
                _encode_text(smiles, SMILES_VOCAB, self.max_smiles_length),
                dtype=torch.long,
            ),
            "protein_tokens": torch.tensor(
                _encode_text(sequence, AMINO_ACID_VOCAB, self.max_protein_length),
                dtype=torch.long,
            ),
            "smiles_text": smiles,
            "protein_sequence": sequence,
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
            "inchi_key": self.inchi_keys[index],
            "target_uniprot_id": self.target_ids[index],
        }


def collate_dti_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    smiles_tokens = torch.stack([item["smiles_tokens"] for item in items], dim=0)
    protein_tokens = torch.stack([item["protein_tokens"] for item in items], dim=0)

    return {
        "smiles_tokens": smiles_tokens,
        "smiles_mask": smiles_tokens.ne(PAD_TOKEN_ID),
        "protein_tokens": protein_tokens,
        "protein_mask": protein_tokens.ne(PAD_TOKEN_ID),
        "protein_sequences": [item["protein_sequence"] for item in items],
        "labels": torch.stack([item["label"] for item in items], dim=0),
        "inchi_keys": [item["inchi_key"] for item in items],
        "target_uniprot_ids": [item["target_uniprot_id"] for item in items],
    }

