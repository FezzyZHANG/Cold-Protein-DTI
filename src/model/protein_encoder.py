"""Protein encoders that preserve residue-level features for downstream fusion."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F

from src.model.esm_support import load_esm_backbone


def _lengths_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """Convert per-sample valid lengths into a dense boolean mask."""

    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


class CNNProteinEncoder(nn.Module):
    """SCOPE-style 1D CNN protein encoder that keeps sequence positions explicit."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, kernel_sizes: list[int], dropout: float) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one convolution width.")

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        in_channels = embed_dim
        for kernel_size in kernel_sizes:
            kernel_size = int(kernel_size)
            if kernel_size < 1:
                raise ValueError(f"kernel_sizes must be >= 1. Received {kernel_size}.")
            self.convs.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            in_channels = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, batch: dict[str, torch.Tensor | list[str]]) -> dict[str, torch.Tensor]:
        token_ids = batch["protein_tokens"]
        mask = batch["protein_mask"]
        sequence_features = self.embedding(token_ids).transpose(1, 2)
        valid_lengths = mask.sum(dim=1)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            kernel_size = int(conv.kernel_size[0])
            if sequence_features.size(-1) < kernel_size:
                raise ValueError(
                    "Protein sequence length after convolution became shorter than the next kernel size. "
                    f"Current length: {sequence_features.size(-1)}, kernel size: {kernel_size}."
                )
            sequence_features = batch_norm(F.relu(conv(sequence_features)))
            valid_lengths = (valid_lengths - kernel_size + 1).clamp_min(0)

        token_features = self.dropout(sequence_features.transpose(1, 2))
        sequence_mask = _lengths_to_mask(valid_lengths, token_features.size(1))
        token_features = token_features * sequence_mask.unsqueeze(-1).to(token_features.dtype)
        return {
            "token_features": token_features,
            "mask": sequence_mask,
        }


class ESMProteinEncoder(nn.Module):
    """Wrapper around local or cached ESM checkpoints for sequence-resolved outputs."""

    def __init__(
        self,
        mode: str,
        hidden_dim: int,
        dropout: float,
        model_name: str,
        max_input_length: int,
        repr_layer: int | None = None,
        local_checkpoint_path: str | None = None,
        freeze_n_layers: int = 0,
    ) -> None:
        super().__init__()

        loaded_backbone = load_esm_backbone(model_name=model_name, local_checkpoint_path=local_checkpoint_path)
        self.backbone = loaded_backbone.backbone
        self.tokenizer = loaded_backbone.tokenizer
        self.backend = loaded_backbone.backend
        self.mode = mode
        self.max_input_length = int(max_input_length)
        self.num_layers = int(loaded_backbone.num_layers)
        self.repr_layer = int(repr_layer) if repr_layer is not None else self.num_layers
        if self.repr_layer < 0 or self.repr_layer > self.num_layers:
            raise ValueError(
                f"repr_layer must be between 0 and {self.num_layers} for {model_name}. "
                f"Received {self.repr_layer}."
            )
        self.freeze_n_layers = int(freeze_n_layers)
        if self.freeze_n_layers < 0:
            raise ValueError(f"freeze_n_layers must be >= 0. Received {self.freeze_n_layers}.")
        if self.freeze_n_layers > self.num_layers:
            raise ValueError(
                f"freeze_n_layers cannot exceed the number of ESM layers ({self.num_layers}). "
                f"Received {self.freeze_n_layers}."
            )
        self.special_token_ids = {
            int(token_id)
            for token_id in getattr(self.tokenizer, "all_special_ids", [])
            if token_id is not None
        }

        self.output_proj = nn.Linear(int(loaded_backbone.hidden_size), hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim
        self._modules_kept_in_eval: list[nn.Module] = []

        if self.mode == "frozen":
            self._freeze_module(self.backbone)
            self._modules_kept_in_eval.append(self.backbone)
        elif self.freeze_n_layers > 0:
            for layer in self._get_encoder_layers()[: self.freeze_n_layers]:
                self._freeze_module(layer)
                self._modules_kept_in_eval.append(layer)

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = False
        module.eval()

    def _get_encoder_layers(self) -> list[nn.Module]:
        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            raise RuntimeError("Unable to locate ESM encoder layers for partial freezing.")
        return list(layers)

    def train(self, mode: bool = True) -> "ESMProteinEncoder":
        super().train(mode)
        for module in self._modules_kept_in_eval:
            module.eval()
        return self

    def _forward_huggingface(
        self,
        sequences: list[str],
        device: torch.device,
        context: nullcontext | torch.no_grad,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            sequences,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_input_length + 2,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        use_hidden_states = self.repr_layer != self.num_layers

        with context:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=use_hidden_states,
                return_dict=True,
            )
            if use_hidden_states:
                token_features = outputs.hidden_states[self.repr_layer]
            else:
                token_features = outputs.last_hidden_state

        mask = attention_mask.to(dtype=torch.bool)
        for token_id in self.special_token_ids:
            mask &= input_ids.ne(token_id)
        return token_features, mask

    def forward(self, batch: dict[str, torch.Tensor | list[str]]) -> dict[str, torch.Tensor]:
        sequences = [sequence[: self.max_input_length] for sequence in batch["protein_sequences"]]
        device = next(self.parameters()).device

        context = torch.no_grad() if self.mode == "frozen" else nullcontext()
        token_features, mask = self._forward_huggingface(sequences, device, context)

        token_features = self.dropout(self.output_proj(token_features))
        token_features = token_features * mask.unsqueeze(-1)
        return {
            "token_features": token_features,
            "mask": mask,
        }
