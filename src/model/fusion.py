"""Fusion heads for pooled concatenation and BAN-style token attention."""

from __future__ import annotations

import torch
from torch import nn


class ConcatFusion(nn.Module):
    """Baseline concatenation + MLP fusion head."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, drug_output: dict[str, torch.Tensor], protein_output: dict[str, torch.Tensor]) -> torch.Tensor:
        fused = torch.cat([drug_output["pooled"], protein_output["pooled"]], dim=-1)
        return self.classifier(fused).squeeze(-1)


class BANFusion(nn.Module):
    """A compact BAN-style fusion block over token-level drug/protein features."""

    def __init__(self, input_dim: int, hidden_dim: int, glimpses: int, dropout: float) -> None:
        super().__init__()
        self.glimpses = int(glimpses)
        self.drug_proj = nn.Linear(input_dim, hidden_dim)
        self.protein_proj = nn.Linear(input_dim, hidden_dim)
        self.attention_proj = nn.Linear(hidden_dim, self.glimpses, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.glimpses * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, drug_output: dict[str, torch.Tensor], protein_output: dict[str, torch.Tensor]) -> torch.Tensor:
        drug_tokens = self.drug_proj(drug_output["token_features"])
        protein_tokens = self.protein_proj(protein_output["token_features"])

        joint = torch.tanh(drug_tokens.unsqueeze(2) + protein_tokens.unsqueeze(1))
        attn_logits = self.attention_proj(joint).permute(0, 3, 1, 2)

        valid_pairs = drug_output["mask"].unsqueeze(1).unsqueeze(3) & protein_output["mask"].unsqueeze(1).unsqueeze(2)
        attn_logits = attn_logits.masked_fill(~valid_pairs, -1e9)
        attn = torch.softmax(attn_logits.flatten(start_dim=2), dim=-1).view_as(attn_logits)

        drug_weights = attn.sum(dim=-1)
        protein_weights = attn.sum(dim=-2)
        drug_context = torch.einsum("bgd,bdh->bgh", drug_weights, drug_tokens)
        protein_context = torch.einsum("bgp,bph->bgh", protein_weights, protein_tokens)
        fused = torch.cat([drug_context, protein_context], dim=-1).reshape(drug_context.shape[0], -1)
        return self.classifier(fused).squeeze(-1)

