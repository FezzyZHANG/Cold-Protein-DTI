"""Drug encoder scaffold using a simple chain-graph GCN over tokenized SMILES."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.unsqueeze(-1).to(features.dtype)
    denom = mask_float.sum(dim=1).clamp_min(1.0)
    return (features * mask_float).sum(dim=1) / denom


class ChainGraphConv(nn.Module):
    """A conservative chain-neighbor message passing block for scaffold experiments."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        left = F.pad(node_features[:, :-1, :], (0, 0, 1, 0))
        right = F.pad(node_features[:, 1:, :], (0, 0, 0, 1))
        left_mask = F.pad(mask[:, :-1], (1, 0))
        right_mask = F.pad(mask[:, 1:], (0, 1))

        degree = left_mask.to(node_features.dtype) + right_mask.to(node_features.dtype)
        neighbor_sum = left * left_mask.unsqueeze(-1) + right * right_mask.unsqueeze(-1)
        neighbor_mean = neighbor_sum / degree.unsqueeze(-1).clamp_min(1.0)

        updated = self.self_proj(node_features) + self.neighbor_proj(neighbor_mean)
        updated = F.gelu(self.norm(updated))
        return updated * mask.unsqueeze(-1)


class GCNDrugEncoder(nn.Module):
    """Token-level scaffold drug encoder with chain-graph message passing."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.layers = nn.ModuleList([ChainGraphConv(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        token_ids = batch["smiles_tokens"]
        mask = batch["smiles_mask"]

        token_features = self.embedding(token_ids)
        token_features = self.dropout(self.input_proj(token_features))
        token_features = token_features * mask.unsqueeze(-1)

        for layer in self.layers:
            token_features = layer(token_features, mask)

        pooled = masked_mean(token_features, mask)
        return {
            "token_features": token_features,
            "mask": mask,
            "pooled": pooled,
        }

