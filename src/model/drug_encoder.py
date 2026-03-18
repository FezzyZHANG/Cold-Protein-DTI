"""PyG-based drug encoder using geometric vector perceptron message passing."""

from __future__ import annotations

from torch import nn
from torch_geometric.utils import to_dense_batch

from src.data.mol_graph import EDGE_SCALAR_DIM, EDGE_VECTOR_DIM, NODE_SCALAR_DIM, NODE_VECTOR_DIM
from src.model.common import masked_mean
from src.model.gvp import GVP, GVPConvLayer, GVPLayerNorm


class GVPDrugEncoder(nn.Module):
    """Encode precalculated molecular graphs with a compact GVP stack."""

    def __init__(
        self,
        node_hidden_scalar: int,
        node_hidden_vector: int,
        edge_hidden_scalar: int,
        edge_hidden_vector: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_input = GVP(
            in_dims=(NODE_SCALAR_DIM, NODE_VECTOR_DIM),
            out_dims=(node_hidden_scalar, node_hidden_vector),
        )
        self.edge_input = GVP(
            in_dims=(EDGE_SCALAR_DIM, EDGE_VECTOR_DIM),
            out_dims=(edge_hidden_scalar, edge_hidden_vector),
        )
        self.layers = nn.ModuleList(
            [
                GVPConvLayer(
                    node_dims=(node_hidden_scalar, node_hidden_vector),
                    edge_dims=(edge_hidden_scalar, edge_hidden_vector),
                    drop_rate=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = GVPLayerNorm((node_hidden_scalar, node_hidden_vector))
        self.output_proj = GVP(
            in_dims=(node_hidden_scalar, node_hidden_vector),
            out_dims=(node_hidden_scalar, 0),
            activations=(None, None),
        )
        self.output_dim = node_hidden_scalar

    def forward(self, batch: dict[str, object]) -> dict[str, object]:
        graph_batch = batch["drug_graph_batch"]
        node_state = self.node_input((graph_batch.node_s.float(), graph_batch.node_v.float()))
        edge_state = self.edge_input((graph_batch.edge_s.float(), graph_batch.edge_v.float()))

        for layer in self.layers:
            node_state = layer(node_state=node_state, edge_index=graph_batch.edge_index, edge_state=edge_state)

        node_state = self.output_norm(node_state)
        node_scalar, _ = self.output_proj(node_state)
        dense_nodes, mask = to_dense_batch(node_scalar, graph_batch.batch)
        pooled = masked_mean(dense_nodes, mask)
        return {
            "token_features": dense_nodes,
            "mask": mask,
            "pooled": pooled,
        }
