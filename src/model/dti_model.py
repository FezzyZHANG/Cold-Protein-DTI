"""Unified DTI model assembly helpers."""

from __future__ import annotations

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

    def forward(self, batch: dict[str, object]) -> dict[str, object]:
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
        )

    fusion_input_dim = int(model_cfg["hidden_dim"])
    if fusion_cfg["name"] == "concat":
        fusion_head = ConcatFusion(
            input_dim=fusion_input_dim,
            hidden_dim=int(fusion_cfg["hidden_dim"]),
            dropout=float(fusion_cfg["dropout"]),
        )
    else:
        fusion_head = BANFusion(
            input_dim=fusion_input_dim,
            hidden_dim=int(fusion_cfg["hidden_dim"]),
            glimpses=int(fusion_cfg["glimpses"]),
            dropout=float(fusion_cfg["dropout"]),
        )

    return DTIModel(drug_encoder=drug_encoder, protein_encoder=protein_encoder, fusion_head=fusion_head)
