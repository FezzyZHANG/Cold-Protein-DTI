"""Shared model helpers."""

from __future__ import annotations

import torch


def masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average token features over valid positions only."""

    mask_float = mask.unsqueeze(-1).to(features.dtype)
    denom = mask_float.sum(dim=1).clamp_min(1.0)
    return (features * mask_float).sum(dim=1) / denom
