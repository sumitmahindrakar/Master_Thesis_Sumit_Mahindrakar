"""
=================================================================
layers.py — BUILDING BLOCKS: MLP + Message Passing Layer
=================================================================

Design choices for strong-form PIGNN:
  - SiLU activation: smooth (C∞), safe for higher-order autodiff
  - Residual connections: gradient flow for d³u/dx³ computation
=================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with SiLU activation.

    SiLU (not ReLU) because we need C∞ smoothness for
    up to 3rd-order autodiff: d³u/dx³ for shear force.
    """

    def __init__(self, input_dim, hidden_dims, output_dim,
                 act=True, dropout=False, p=0.5):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.act = act
        self.dropout = dropout
        self.p = p

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.act and i != len(self.layers) - 1:
                x = F.silu(x)
            if self.dropout and self.training:
                x = F.dropout(x, p=self.p, training=self.training)
        return x


class GraphNetworkLayer(MessagePassing):
    """
    Message passing layer with residual connection.

    Message:  m_ij = MLP([h_i || h_j || e_ij])
    Update:   h_i' = LayerNorm(h_i + MLP([h_i || aggr(m_ij)]))
                                 ↑
                           residual skip
    
    Residual is critical: autodiff computes up to 3rd derivatives
    through the entire network. Without skip connections,
    gradients vanish → d³u/dx³ becomes zero/noise.
    """

    def __init__(self, hidden_dim, edge_attr_dim, aggr='add'):
        super().__init__(aggr=aggr)

        self.message_mlp = MLP(
            hidden_dim * 2 + edge_attr_dim,
            [hidden_dim],
            hidden_dim,
            act=True,
        )

        self.update_mlp = MLP(
            hidden_dim * 2,
            [hidden_dim],
            hidden_dim,
            act=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.norm(x + out)    # ← residual connection

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(
            torch.cat([x_i, x_j, edge_attr], dim=-1)
        )

    def update(self, aggr_out, x):
        return self.update_mlp(
            torch.cat([x, aggr_out], dim=-1)
        )