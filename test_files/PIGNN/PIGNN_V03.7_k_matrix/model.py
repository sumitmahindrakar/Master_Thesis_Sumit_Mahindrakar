"""
model.py — PIGNN for 2D Euler-Bernoulli frames
Output: (N, 3) [u_x, u_z, φ]
"""

import torch
import torch.nn as nn
from layers import MLP, GraphNetworkLayer


class PIGNN(nn.Module):

    def __init__(self,
                 node_in_dim=9,
                 edge_in_dim=10,
                 hidden_dim=128,
                 n_layers=6,
                 out_dim=3):
        super().__init__()
        H = hidden_dim

        self.node_encoder = MLP(node_in_dim, [H], H, act=True)
        self.edge_encoder = MLP(edge_in_dim, [H], H, act=True)

        self.mp_layers = nn.ModuleList([
            GraphNetworkLayer(H, H, aggr='add')
            for _ in range(n_layers)
        ])

        self.decoder = MLP(H, [H, 64], out_dim, act=True)

    def forward(self, data):
        h = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)

        for mp in self.mp_layers:
            h = h + mp(h, data.edge_index, e)

        pred = self.decoder(h)
        pred = self._apply_bc(pred, data)
        return pred

    def _apply_bc(self, pred, data):
        """Hard BC — out-of-place, autograd-safe."""
        disp_mask = (1.0 - data.bc_disp)     # (N, 1)
        rot_mask  = (1.0 - data.bc_rot)       # (N, 1)
        return torch.cat([
            pred[:, 0:2] * disp_mask,         # u_x, u_z
            pred[:, 2:3] * rot_mask,          # φ
        ], dim=1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())