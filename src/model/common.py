"""Shared model helpers."""

from __future__ import annotations

import torch
from torch import nn


def masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average token features over valid positions only."""

    mask_float = mask.unsqueeze(-1).to(features.dtype)
    denom = mask_float.sum(dim=1).clamp_min(1.0)
    return (features * mask_float).sum(dim=1) / denom


def init_embedding(embedding: nn.Embedding) -> None:
    """Initialize token embeddings while keeping the padding row fixed at zero."""

    nn.init.normal_(embedding.weight, mean=0.0, std=embedding.embedding_dim**-0.5)
    if embedding.padding_idx is not None:
        with torch.no_grad():
            embedding.weight[embedding.padding_idx].zero_()


def init_linear(linear: nn.Linear, gain: float = 1.0) -> None:
    """Apply Xavier initialization to a linear layer and zero its bias."""

    nn.init.xavier_uniform_(linear.weight, gain=gain)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def init_conv1d(conv: nn.Conv1d, nonlinearity: str = "relu") -> None:
    """Apply Kaiming initialization to a 1D convolution followed by a ReLU-like activation."""

    nn.init.kaiming_normal_(conv.weight, nonlinearity=nonlinearity)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def init_batch_norm(batch_norm: nn.BatchNorm1d) -> None:
    """Initialize batch normalization as an identity transform."""

    batch_norm.reset_running_stats()
    if batch_norm.weight is not None:
        nn.init.ones_(batch_norm.weight)
    if batch_norm.bias is not None:
        nn.init.zeros_(batch_norm.bias)


def init_layer_norm(layer_norm: nn.LayerNorm) -> None:
    """Initialize layer normalization as an identity transform."""

    if layer_norm.weight is not None:
        nn.init.ones_(layer_norm.weight)
    if layer_norm.bias is not None:
        nn.init.zeros_(layer_norm.bias)


def activation_gain(name: str | None) -> float:
    """Return a conservative Xavier gain for the requested activation."""

    if not name:
        return 1.0
    normalized = name.lower()
    if normalized == "gelu":
        return 2.0**0.5
    try:
        return nn.init.calculate_gain(normalized)
    except ValueError:
        return 1.0
