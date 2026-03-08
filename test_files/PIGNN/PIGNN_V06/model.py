"""
model_v2.py — Fixed: Coords flow through GNN for real spatial derivatives
"""

import torch
import torch.nn as nn
from layers import MLP, GraphNetworkLayer
from torch.utils.checkpoint import checkpoint


class GraphNetworkLayerWithCoords(nn.Module):
    """
    Message passing where coordinates participate in messages.
    
    This ensures ∂h_i/∂coords_j ≠ 0 for neighbors j,
    which means autograd spatial derivatives capture
    ACTUAL spatial variation across the structure.
    """

    def __init__(self, hidden_dim, edge_feat_dim, aggr='add'):
        super().__init__()
        
        # Message: uses relative coordinates + node states + edge features
        # coord_dim=3 for relative coords (x_j - x_i)
        self.message_mlp = MLP(
            hidden_dim * 2 + edge_feat_dim + 3,  # +3 for relative coords
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

    def forward(self, x, edge_index, edge_attr, coords):
        """
        Args:
            x:          (N, H) node embeddings
            edge_index: (2, num_edges) 
            edge_attr:  (num_edges, edge_feat_dim)
            coords:     (N, 3) coordinates WITH grad enabled
        """
        src, dst = edge_index  # src → dst
        
        # Relative coordinates: this creates the gradient path!
        # ∂(coords[dst] - coords[src])/∂coords[src] = -I
        # ∂(coords[dst] - coords[src])/∂coords[dst] = +I
        rel_coords = coords[dst] - coords[src]  # (num_edges, 3)
        
        # Messages include relative coordinates
        msg_input = torch.cat([
            x[src], x[dst], edge_attr, rel_coords
        ], dim=-1)
        messages = self.message_mlp(msg_input)  # (num_edges, H)
        
        # Aggregate messages to destination nodes
        aggr = torch.zeros_like(x)
        aggr.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        
        # Update
        update_input = torch.cat([x, aggr], dim=-1)
        out = self.update_mlp(update_input)
        
        return self.norm(x + out)  # residual


class PIGNN_Hybrid(nn.Module):
    """
    Fixed architecture: coordinates flow through BOTH
    the GNN AND the decoder.
    
    Key difference from V1:
      V1: h = GNN(data.x) → pred = decoder(coords, h)
          ∂pred_i/∂coords_j = 0 for i≠j  ← BROKEN
      
      V2: h = GNN(data.x, coords) → pred = decoder(coords, h)
          ∂pred_i/∂coords_j ≠ 0 for neighbors  ← CORRECT
    
    Now du/dx at node i captures how displacement changes
    when you move node i AND how that affects its neighbors'
    contributions — i.e., real spatial gradients.
    """

    def __init__(self,
                 node_in_dim=9,
                 edge_in_dim=10,
                 hidden_dim=128,
                 n_layers=6,
                 out_dim=3):
        super().__init__()

        H = hidden_dim
        self.out_dim = out_dim

        # Node encoder: includes raw coordinates
        self.node_encoder = MLP(node_in_dim, [H], H, act=True)
        self.edge_encoder = MLP(edge_in_dim, [H], H, act=True)

        # GNN layers now take coordinates
        self.mp_layers = nn.ModuleList([
            GraphNetworkLayerWithCoords(H, H, aggr='add')
            for _ in range(n_layers)
        ])

        # Decoder: coords + GNN context → displacement
        self.decoder = MLP(3 + H, [H, H, 64], out_dim, act=True)

        # Small initialization for correct force scale
        with torch.no_grad():
            last_layer = self.decoder.layers[-1]
            nn.init.normal_(last_layer.weight, std=1e-4)
            nn.init.zeros_(last_layer.bias)

    def forward(self, data):
        # Coordinates with gradient tracking
        coords_physics = data.coords.clone().requires_grad_(True)
        
        # ── Phase 1: GNN with coordinate coupling ──
        h = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)

        for mp in self.mp_layers:
            h = checkpoint(mp, h, data.edge_index, e, coords_physics,
                   use_reentrant=False)
            h = mp(h, data.edge_index, e, coords_physics)
            #                                ^^^^^^^^^^^^^^
            # This is the critical fix: coords flow through GNN
            # so h_i depends on coords_j for all neighbors j

        # ── Phase 2: Pointwise decoder ──
        with torch.no_grad():
            c_min = coords_physics.min(dim=0)[0]
            c_max = coords_physics.max(dim=0)[0]
            c_range = (c_max - c_min).clamp(min=1e-8)

        coords_norm = (coords_physics - c_min) / c_range

        decoder_input = torch.cat([coords_norm, h], dim=-1)
        pred = self.decoder(decoder_input)

        # Hard BCs
        pred = self._apply_bc(pred, data)

        self._coords = coords_physics
        return pred

    def _apply_bc(self, pred, data):
        pred = pred.clone()
        pred[:, 0:2] = pred[:, 0:2] * (1.0 - data.bc_disp)
        pred[:, 2:3] = pred[:, 2:3] * (1.0 - data.bc_rot)
        return pred

    def get_coords(self):
        return self._coords

    def count_params(self):
        return sum(p.numel() for p in self.parameters())