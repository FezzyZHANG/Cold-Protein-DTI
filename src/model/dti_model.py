"""Unified DTI model assembly helpers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from src.data.dataset import AMINO_ACID_VOCAB
from src.model.drug_encoder import GVPDrugEncoder
from src.model.fusion import BANFusion, ConcatFusion
from src.model.protein_encoder import CNNProteinEncoder, ESMProteinEncoder


class DTIModel(nn.Module):
    """Compose a drug encoder, protein encoder, and fusion head behind one interface."""

    def __init__(self, drug_encoder: nn.Module, protein_encoder: nn.Module, fusion_head: nn.Module) -> None:
        super().__init__()
        self.drug_encoder = drug_encoder
        self.protein_encoder = protein_encoder
        self.fusion_head = fusion_head

    @staticmethod
    def _disabled_autocast_context(device_type: str) -> Any:
        """Run numerically sensitive branches outside AMP when requested."""
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type=device_type, enabled=False)
        if device_type == "cuda":
            return torch.cuda.amp.autocast(enabled=False)
        return nullcontext()

    def _infer_device_type(self, batch: dict[str, object]) -> str:
        try:
            return next(self.parameters()).device.type
        except StopIteration:
            for value in batch.values():
                if hasattr(value, "device"):
                    return str(value.device.type)
        return "cpu"

    def forward(self, batch: dict[str, object]) -> dict[str, object]:
        device_type = self._infer_device_type(batch)
        with self._disabled_autocast_context(device_type):
            drug_output = self.drug_encoder(batch)
        protein_output = self.protein_encoder(batch)
        logits = self.fusion_head(drug_output, protein_output)
        return {
            "logits": logits,
            "drug": drug_output,
            "protein": protein_output,
        }


def build_model(config: dict[str, object]) -> DTIModel:
    model_cfg = config["model"]
    drug_cfg = model_cfg["drug_encoder"]
    protein_cfg = model_cfg["protein_encoder"]
    fusion_cfg = model_cfg["fusion"]

    drug_encoder = GVPDrugEncoder(
        node_hidden_scalar=int(drug_cfg["node_hidden_scalar"]),
        node_hidden_vector=int(drug_cfg["node_hidden_vector"]),
        edge_hidden_scalar=int(drug_cfg["edge_hidden_scalar"]),
        edge_hidden_vector=int(drug_cfg["edge_hidden_vector"]),
        num_layers=int(drug_cfg["num_layers"]),
        dropout=float(drug_cfg["dropout"]),
    )

    if protein_cfg["name"] == "cnn":
        protein_encoder = CNNProteinEncoder(
            vocab_size=len(AMINO_ACID_VOCAB) + 2,
            embed_dim=int(protein_cfg["embed_dim"]),
            hidden_dim=int(protein_cfg["hidden_dim"]),
            kernel_sizes=[int(kernel_size) for kernel_size in protein_cfg["kernel_sizes"]],
            dropout=float(protein_cfg["dropout"]),
        )
    else:
        protein_encoder = ESMProteinEncoder(
            mode=str(protein_cfg["mode"]),
            hidden_dim=int(protein_cfg["hidden_dim"]),
            dropout=float(protein_cfg["dropout"]),
            model_name=str(protein_cfg["model_name"]),
            repr_layer=protein_cfg["repr_layer"],
            local_checkpoint_path=protein_cfg["local_checkpoint_path"],
            max_input_length=int(protein_cfg["max_input_length"]),
            freeze_n_layers=int(protein_cfg["freeze_n_layers"]),
            prefer_staged_artifacts=bool(protein_cfg["prefer_staged_artifacts"]),
            backend=protein_cfg["backend"],
            base_model_name=protein_cfg["base_model_name"],
            base_checkpoint_path=protein_cfg["base_checkpoint_path"],
        )

    if fusion_cfg["name"] == "concat":
        fusion_head = ConcatFusion(
            drug_input_dim=int(drug_encoder.output_dim),
            protein_input_dim=int(protein_encoder.output_dim),
            hidden_dim=int(fusion_cfg["hidden_dim"]),
            dropout=float(fusion_cfg["dropout"]),
            input_norm=str(fusion_cfg["input_norm"]),
        )
    else:
        fusion_head = BANFusion(
            drug_input_dim=int(drug_encoder.output_dim),
            protein_input_dim=int(protein_encoder.output_dim),
            joint_dim=int(fusion_cfg["joint_dim"]),
            classifier_hidden_dim=int(fusion_cfg["classifier_hidden_dim"]),
            glimpses=int(fusion_cfg["glimpses"]),
            dropout=float(fusion_cfg["dropout"]),
            drug_feature_mode=str(fusion_cfg["drug_feature_mode"]),
            protein_feature_mode=str(fusion_cfg["protein_feature_mode"]),
            use_global_features=bool(fusion_cfg["use_global_features"]),
            attention_softmax=bool(fusion_cfg["attention_softmax"]),
            norm=str(fusion_cfg["norm"]),
            input_norm=str(fusion_cfg["input_norm"]),
            classifier_num_blocks=int(fusion_cfg["classifier_num_blocks"]),
            classifier_expansion=float(fusion_cfg["classifier_expansion"]),
        )

    return DTIModel(drug_encoder=drug_encoder, protein_encoder=protein_encoder, fusion_head=fusion_head)
