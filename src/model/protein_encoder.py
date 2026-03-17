"""Protein encoder implementations for CNN and staged local ESM backbones."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from src.model.drug_encoder import masked_mean


class CNNProteinEncoder(nn.Module):
    """1D CNN protein encoder trained from scratch."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, kernel_sizes: list[int], dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, batch: dict[str, torch.Tensor | list[str]]) -> dict[str, torch.Tensor]:
        token_ids = batch["protein_tokens"]
        mask = batch["protein_mask"]
        embedded = self.embedding(token_ids).transpose(1, 2)

        conv_outputs = [F.gelu(conv(embedded)).transpose(1, 2) for conv in self.convs]
        token_features = torch.stack(conv_outputs, dim=0).mean(dim=0)
        token_features = self.dropout(self.output_proj(token_features))
        token_features = token_features * mask.unsqueeze(-1)
        pooled = masked_mean(token_features, mask)
        return {
            "token_features": token_features,
            "mask": mask,
            "pooled": pooled,
        }


class ESMProteinEncoder(nn.Module):
    """Wrapper around local or cached ESM checkpoints for frozen or finetuned runs."""

    def __init__(
        self,
        mode: str,
        hidden_dim: int,
        dropout: float,
        model_name: str,
        max_input_length: int,
        repr_layer: int | None = None,
        local_checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()

        try:
            import esm  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The ESM protein encoder requires the optional `fair-esm` dependency. "
                "Install it with `uv sync --extra esm` and stage the checkpoint locally."
            ) from exc

        if local_checkpoint_path:
            checkpoint = Path(local_checkpoint_path)
            if not checkpoint.exists():
                raise FileNotFoundError(f"Local ESM checkpoint not found: {checkpoint}")
            backbone, alphabet = esm.pretrained.load_model_and_alphabet_local(str(checkpoint))
        else:
            load_fn = getattr(esm.pretrained, model_name, None)
            if load_fn is None:
                raise ValueError(f"Unknown ESM model name: {model_name}")
            backbone, alphabet = load_fn()

        self.backbone = backbone
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.mode = mode
        self.max_input_length = int(max_input_length)
        self.repr_layer = int(repr_layer) if repr_layer is not None else int(getattr(backbone, "num_layers", 33))
        backbone_dim = getattr(backbone, "embed_dim", None)
        if backbone_dim is None and hasattr(backbone, "embed_tokens"):
            backbone_dim = int(backbone.embed_tokens.embedding_dim)
        if backbone_dim is None:
            raise RuntimeError("Unable to determine the ESM embedding dimension from the loaded checkpoint.")

        self.output_proj = nn.Linear(int(backbone_dim), hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

        if self.mode == "frozen":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()

    def forward(self, batch: dict[str, torch.Tensor | list[str]]) -> dict[str, torch.Tensor]:
        sequences = [sequence[: self.max_input_length] for sequence in batch["protein_sequences"]]
        entries = [(str(index), sequence) for index, sequence in enumerate(sequences)]
        _, _, tokens = self.batch_converter(entries)
        device = next(self.parameters()).device
        tokens = tokens.to(device)

        context = torch.no_grad() if self.mode == "frozen" else nullcontext()
        with context:
            outputs = self.backbone(tokens, repr_layers=[self.repr_layer], return_contacts=False)
            token_features = outputs["representations"][self.repr_layer]

        token_features = self.dropout(self.output_proj(token_features))
        mask = tokens.ne(self.alphabet.padding_idx)
        mask[:, 0] = False
        lengths = mask.sum(dim=1)
        eos_positions = (lengths - 1).clamp_min(0)
        mask[torch.arange(mask.shape[0], device=device), eos_positions] = False

        token_features = token_features * mask.unsqueeze(-1)
        pooled = masked_mean(token_features, mask)
        return {
            "token_features": token_features,
            "mask": mask,
            "pooled": pooled,
        }

