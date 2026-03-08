"""
=================================================================
model.py — Hybrid GNN-PINN with Output Scaling
=================================================================

Two fixes for correct force balance:
1. Normalize coords in decoder input → [0, 1]
2. Initialize last layer very small → pred starts near zero

Without these: N = EA × du/dx ≈ 4.3e9 × 0.01 = 43,000,000 N
With these:    N = EA × du/dx ≈ 4.3e9 × 1e-8 = 43 N  ← correct!
=================================================================
"""

import torch
import torch.nn as nn
from layers import MLP, GraphNetworkLayer


class PIGNN_Hybrid(nn.Module):

    def __init__(self,
                 node_in_dim=9,
                 edge_in_dim=10,
                 hidden_dim=128,
                 n_layers=6,
                 out_dim=3):
        super().__init__()

        H = hidden_dim
        self.out_dim = out_dim

        # ── Phase 1: GNN ──
        self.node_encoder = MLP(node_in_dim, [H], H, act=True)
        self.edge_encoder = MLP(edge_in_dim, [H], H, act=True)

        self.mp_layers = nn.ModuleList([
            GraphNetworkLayer(H, H, aggr='add')
            for _ in range(n_layers)
        ])

        # ── Phase 2: Pointwise Decoder ──
        self.decoder = MLP(3 + H, [H, H, 64], out_dim, act=True)

        # ══════════════════════════════════════
        # FIX 1: Initialize last layer VERY small
        # ══════════════════════════════════════
        # Without this: initial pred ≈ 0.01 m
        #   → du/dx ≈ 0.01/18 ≈ 5e-4
        #   → N = EA × 5e-4 = 4.3e9 × 5e-4 = 2.1e6 N (way too big!)
        #
        # With this (std=1e-6): initial pred ≈ 1e-6 m
        #   → du/dx ≈ 1e-6/18 ≈ 5e-8
        #   → N = EA × 5e-8 = 4.3e9 × 5e-8 = 215 N (reasonable!)
        #
        with torch.no_grad():
            last_layer = self.decoder.layers[-1]
            nn.init.normal_(last_layer.weight, std=1e-4)#std=1e-6
            nn.init.zeros_(last_layer.bias)

    def forward(self, data):
        # ── Phase 1: GNN context ──
        h = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)

        for mp in self.mp_layers:
            h = mp(h, data.edge_index, e)

        # ── Phase 2: Pointwise decoder ──
        coords_physics = data.coords.clone().requires_grad_(True)

        # ══════════════════════════════════════
        # FIX 2: Normalize coords to [0, 1]
        # ══════════════════════════════════════
        # Without: coords in [0, 18] → decoder gradient ∂u/∂x̄ gets
        #          divided by 18 in chain rule → poorly conditioned
        #
        # With:    coords in [0, 1] → decoder gradient O(1)
        #          autograd handles chain rule automatically:
        #          ∂u/∂x_physical = (1/range) × ∂u/∂x_normalized
        #
        with torch.no_grad():
            c_min = coords_physics.min(dim=0)[0]
            c_max = coords_physics.max(dim=0)[0]
            c_range = (c_max - c_min).clamp(min=1e-8)

        coords_norm = (coords_physics - c_min) / c_range  # [0, 1]

        decoder_input = torch.cat([coords_norm, h], dim=-1)
        pred = self.decoder(decoder_input)

        # ── Hard BCs ──
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