# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import MessagePassing


# class MLP(nn.Module):
#     """Simple MLP with optional activation and dropout."""

#     def __init__(self, input_dim, hidden_dims, output_dim,
#                  act=True, dropout=False, p=0.5):
#         super().__init__()
#         dims = [input_dim] + list(hidden_dims) + [output_dim]
#         self.layers = nn.ModuleList()
#         for i in range(len(dims) - 1):
#             self.layers.append(nn.Linear(dims[i], dims[i + 1]))
#         self.act = act
#         self.dropout = dropout
#         self.p = p

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if self.act and i != len(self.layers) - 1:
#                 x = F.silu(x)  # SiLU works better than ReLU for physics
#             if self.dropout:
#                 x = F.dropout(x, p=self.p, training=self.training)
#         return x


# class GraphNetworkLayer(MessagePassing):
#     """
#     Message passing layer with edge attributes.
    
#     Message:  m_ij = MLP([h_i || h_j || e_ij])
#     Update:   h_i' = MLP([h_i || aggr(m_ij)])
#     """

#     def __init__(self, hidden_dim, edge_attr_dim, aggr='add'):
#         super().__init__(aggr=aggr)

#         # Message MLP: [h_src || h_dst || edge_attr] → message
#         self.message_mlp = MLP(
#             hidden_dim * 2 + edge_attr_dim,
#             [hidden_dim],
#             hidden_dim,
#             act=True
#         )

#         # Update MLP: [h_node || aggregated_messages] → updated h
#         self.update_mlp = MLP(
#             hidden_dim * 2,
#             [hidden_dim],
#             hidden_dim,
#             act=True
#         )

#         self.norm = nn.LayerNorm(hidden_dim)

#     def forward(self, x, edge_index, edge_attr):
#         out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
#         return self.norm(out)  # normalize for stability

#     def message(self, x_i, x_j, edge_attr):
#         return self.message_mlp(
#             torch.cat([x_i, x_j, edge_attr], dim=-1)
#         )

#     def update(self, aggr_out, x):
#         return self.update_mlp(
#             torch.cat([x, aggr_out], dim=-1)
#         )

"""
=================================================================
layers.py — BUILDING BLOCKS: MLP + Message Passing Layer
=================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with optional activation and dropout.

    Args:
        input_dim:   input feature dimension
        hidden_dims: list of hidden layer dimensions (e.g. [128] or [128, 64])
        output_dim:  output dimension
        act:         apply SiLU activation between layers (not on last)
        dropout:     apply dropout between layers
        p:           dropout probability
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
    Message passing layer with edge attributes.

    Message:  m_ij = MLP([h_i || h_j || e_ij])
    Update:   h_i' = MLP([h_i || aggr(m_ij)])

    Args:
        hidden_dim:    node embedding dimension
        edge_attr_dim: edge feature dimension
        aggr:          aggregation method ('add', 'mean', 'max')
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
        return self.norm(out)

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(
            torch.cat([x_i, x_j, edge_attr], dim=-1)
        )

    def update(self, aggr_out, x):
        return self.update_mlp(
            torch.cat([x, aggr_out], dim=-1)
        )