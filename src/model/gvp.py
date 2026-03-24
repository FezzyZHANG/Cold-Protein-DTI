"""Minimal GVP layers for scalar/vector message passing on PyG graphs."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from src.model.common import init_layer_norm, init_linear


ScalarVector = tuple[torch.Tensor, torch.Tensor]


def _norm_no_nan(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1.0e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp(torch.sum(tensor * tensor, dim=dim, keepdim=keepdim), min=eps))


def _merge(scalar: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if vector.shape[-2] == 0:
        return scalar
    return torch.cat([scalar, vector.reshape(*vector.shape[:-2], vector.shape[-2] * 3)], dim=-1)


def _split(merged: torch.Tensor, vector_channels: int) -> ScalarVector:
    if vector_channels == 0:
        return merged, merged.new_zeros(*merged.shape[:-1], 0, 3)
    scalar = merged[..., :-3 * vector_channels]
    vector = merged[..., -3 * vector_channels :].reshape(*merged.shape[:-1], vector_channels, 3)
    return scalar, vector


def tuple_sum(left: ScalarVector, right: ScalarVector) -> ScalarVector:
    return left[0] + right[0], left[1] + right[1]


def tuple_cat(*items: ScalarVector) -> ScalarVector:
    scalar = torch.cat([item[0] for item in items], dim=-1)
    vector = torch.cat([item[1] for item in items], dim=-2)
    return scalar, vector


class VectorDropout(nn.Module):
    def __init__(self, drop_rate: float) -> None:
        super().__init__()
        self.drop_rate = float(drop_rate)

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_rate == 0.0 or vector.shape[-2] == 0:
            return vector
        keep_rate = 1.0 - self.drop_rate
        mask = torch.empty(vector.shape[:-1], device=vector.device, dtype=vector.dtype).bernoulli_(keep_rate)
        return vector * (mask.unsqueeze(-1) / keep_rate)


class GVPTupleDropout(nn.Module):
    def __init__(self, drop_rate: float) -> None:
        super().__init__()
        self.scalar_dropout = nn.Dropout(drop_rate)
        self.vector_dropout = VectorDropout(drop_rate)

    def forward(self, item: ScalarVector) -> ScalarVector:
        return self.scalar_dropout(item[0]), self.vector_dropout(item[1])


class GVPLayerNorm(nn.Module):
    def __init__(self, dims: tuple[int, int]) -> None:
        super().__init__()
        scalar_dim, vector_dim = dims
        self.scalar_norm = nn.LayerNorm(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.vector_scale = nn.Parameter(torch.ones(self.vector_dim)) if self.vector_dim > 0 else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_layer_norm(self.scalar_norm)
        if self.vector_scale is not None:
            nn.init.ones_(self.vector_scale)

    def forward(self, item: ScalarVector) -> ScalarVector:
        scalar, vector = item
        scalar = self.scalar_norm(scalar)
        if self.vector_dim == 0:
            return scalar, vector
        vector_norm = _norm_no_nan(vector, dim=-1)
        scale = vector_norm.mean(dim=-1, keepdim=True).clamp_min(1.0e-8)
        vector = vector / scale.unsqueeze(-1)
        vector = vector * self.vector_scale.view(*([1] * (vector.ndim - 2)), self.vector_dim, 1)
        return scalar, vector


class GVP(nn.Module):
    """A geometric vector perceptron mapping `(scalar, vector)` to `(scalar, vector)`."""

    def __init__(
        self,
        in_dims: tuple[int, int],
        out_dims: tuple[int, int],
        h_dim: int | None = None,
        activations: tuple[Callable[[torch.Tensor], torch.Tensor] | None, Callable[[torch.Tensor], torch.Tensor] | None] = (
            torch.nn.functional.silu,
            torch.sigmoid,
        ),
        vector_gate: bool = True,
    ) -> None:
        super().__init__()
        self.si, self.vi = int(in_dims[0]), int(in_dims[1])
        self.so, self.vo = int(out_dims[0]), int(out_dims[1])
        self.vector_gate = vector_gate
        self.scalar_act, self.vector_act = activations

        if self.vi > 0:
            hidden_vector = int(h_dim or max(self.vi, self.vo, 1))
            self.wh = nn.Linear(self.vi, hidden_vector, bias=False)
            self.ws = nn.Linear(self.si + hidden_vector, self.so)
            self.wv = nn.Linear(hidden_vector, self.vo, bias=False) if self.vo > 0 else None
        else:
            self.wh = None
            self.ws = nn.Linear(self.si, self.so)
            self.wv = None

        self.wg = nn.Linear(self.so, self.vo) if self.vo > 0 and self.vector_gate else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.wh is not None:
            init_linear(self.wh)
        init_linear(self.ws)
        if self.wv is not None:
            init_linear(self.wv)
        if self.wg is not None:
            init_linear(self.wg)

    def forward(self, item: ScalarVector) -> ScalarVector:
        scalar, vector = item

        if self.vi > 0:
            vector_hidden = self.wh(vector.transpose(-1, -2))
            vector_norm = _norm_no_nan(vector_hidden, dim=-2)
            scalar = self.ws(torch.cat([scalar, vector_norm], dim=-1))
            if self.vo > 0 and self.wv is not None:
                vector = self.wv(vector_hidden).transpose(-1, -2)
                if self.wg is not None:
                    gate = torch.sigmoid(self.wg(scalar)).unsqueeze(-1)
                    vector = vector * gate
                elif self.vector_act is not None:
                    vector = vector * self.vector_act(_norm_no_nan(vector, dim=-1, keepdim=True))
            else:
                vector = vector.new_zeros(*vector.shape[:-2], 0, 3)
        else:
            scalar = self.ws(scalar)
            if self.vo > 0:
                vector = scalar.new_zeros(*scalar.shape[:-1], self.vo, 3)
            else:
                vector = scalar.new_zeros(*scalar.shape[:-1], 0, 3)

        if self.scalar_act is not None:
            scalar = self.scalar_act(scalar)
        return scalar, vector


class GVPConv(MessagePassing):
    """Message passing over scalar/vector node states."""

    def __init__(
        self,
        node_dims: tuple[int, int],
        edge_dims: tuple[int, int],
        out_dims: tuple[int, int] | None = None,
        message_layers: int = 2,
        aggr: str = "mean",
    ) -> None:
        super().__init__(aggr=aggr, node_dim=0)
        self.node_dims = (int(node_dims[0]), int(node_dims[1]))
        self.edge_dims = (int(edge_dims[0]), int(edge_dims[1]))
        self.out_dims = out_dims or self.node_dims
        self.out_dims = (int(self.out_dims[0]), int(self.out_dims[1]))

        layers: list[nn.Module] = []
        current_dims = (
            (self.node_dims[0] * 2) + self.edge_dims[0],
            (self.node_dims[1] * 2) + self.edge_dims[1],
        )
        for _ in range(max(message_layers - 1, 0)):
            layers.append(GVP(current_dims, self.node_dims))
            current_dims = self.node_dims
        layers.append(GVP(current_dims, self.out_dims, activations=(None, None)))
        self.message_layers = nn.ModuleList(layers)

    def forward(self, node_state: ScalarVector, edge_index: torch.Tensor, edge_state: ScalarVector) -> ScalarVector:
        out = self.propagate(
            edge_index,
            x_s=node_state[0],
            x_v=node_state[1],
            edge_s=edge_state[0],
            edge_v=edge_state[1],
        )
        return _split(out, self.out_dims[1])

    def message(
        self,
        x_s_i: torch.Tensor,
        x_v_i: torch.Tensor,
        x_s_j: torch.Tensor,
        x_v_j: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
    ) -> torch.Tensor:
        message = tuple_cat((x_s_i, x_v_i), (edge_s, edge_v), (x_s_j, x_v_j))
        for layer in self.message_layers:
            message = layer(message)
        return _merge(*message)


class GVPConvLayer(nn.Module):
    """Residual GVP message passing block with tuple layer norm and tuple dropout."""

    def __init__(self, node_dims: tuple[int, int], edge_dims: tuple[int, int], drop_rate: float = 0.1) -> None:
        super().__init__()
        node_dims = (int(node_dims[0]), int(node_dims[1]))
        edge_dims = (int(edge_dims[0]), int(edge_dims[1]))

        self.conv = GVPConv(node_dims=node_dims, edge_dims=edge_dims)
        hidden_dims = (node_dims[0] * 2, max(node_dims[1] * 2, 1))
        self.ff_first = GVP(node_dims, hidden_dims)
        self.ff_second = GVP(hidden_dims, node_dims, activations=(None, None))
        self.norm_first = GVPLayerNorm(node_dims)
        self.norm_second = GVPLayerNorm(node_dims)
        self.dropout_first = GVPTupleDropout(drop_rate)
        self.dropout_second = GVPTupleDropout(drop_rate)

    def forward(self, node_state: ScalarVector, edge_index: torch.Tensor, edge_state: ScalarVector) -> ScalarVector:
        update = self.conv(node_state=node_state, edge_index=edge_index, edge_state=edge_state)
        node_state = self.norm_first(tuple_sum(node_state, self.dropout_first(update)))

        ff_update = self.ff_second(self.ff_first(node_state))
        node_state = self.norm_second(tuple_sum(node_state, self.dropout_second(ff_update)))
        return node_state
