"""
model.py — PIGNN with SEPARATED coordinate path

Architecture:
  Branch 1 (physics-free): other features → encoder → MP → latent h_i
  Branch 2 (spatial):      coords (x, z) passed directly to decoder

  Decoder: u_i = MLP(h_i, x_i, z_i)

  ∂u/∂x is now a clean autograd derivative through the decoder only.
  No message passing in the autograd path.
"""

import torch
import torch.nn as nn
from layers import MLP, GraphNetworkLayer


class PIGNN_SeparatedCoords(nn.Module):
    """
    Coordinate-separated PIGNN.

    Forward path:
      1. Non-coord features → encoder → h
      2. h → message passing → h'  (coords NOT in this path)
      3. [h', coords] → decoder → [u_x, u_z, φ]

    Autograd path:
      ∂u/∂coords goes through decoder ONLY (no MP layers)
    """

    def __init__(self,
                 node_in_dim=9,
                 edge_in_dim=10,
                 hidden_dim=128,
                 n_layers=6,
                 out_dim=3,
                 coord_dim=2):        # x, z for 2D frame
        super().__init__()
        H = hidden_dim
        self.out_dim = out_dim
        self.coord_dim = coord_dim

        # Node features WITHOUT coords: 9 - 3 = 6
        # [bc_disp, bc_rot, wl_x, wl_y, wl_z, response]
        non_coord_dim = node_in_dim - 3  # = 6

        # ── Branch 1: Feature encoder (no coords) ──
        self.node_encoder = MLP(non_coord_dim, [H], H, act=True)
        self.edge_encoder = MLP(edge_in_dim, [H], H, act=True)

        # ── Message Passing (coords NOT involved) ──
        self.mp_layers = nn.ModuleList([
            GraphNetworkLayer(H, H, aggr='add')
            for _ in range(n_layers)
        ])

        # ── Decoder: takes [latent h, coords] ──
        # Input: H (from MP) + coord_dim (x, z)
        self.decoder = MLP(H + coord_dim, [H, 64], out_dim, act=True)

        # Storage for coords in autograd graph
        self._coords_2d = None

    def forward(self, data):
        """
        Args:
            data: PyG Data with .x (N, 9), .edge_index, .edge_attr

        Returns:
            pred: (N, 3) [u_x, u_z, φ]
        """
        # ── Extract and SEPARATE coords from features ──
        # data.x columns: [x, y, z, bc_d, bc_r, wl_x, wl_y, wl_z, resp]
        #                  [0  1  2   3     4     5     6     7      8  ]

        # Coords: use x and z (2D frame in XZ plane)
        self._coords_2d = data.coords[:, [0, 2]].clone().requires_grad_(True)
        # shape: (N, 2) — [x, z]

        # Non-coord features: columns 3-8
        non_coord_features = data.x[:, 3:]    # (N, 6)

        # ── Branch 1: Encode non-coord features ──
        h = self.node_encoder(non_coord_features)    # (N, H)
        e = self.edge_encoder(data.edge_attr)        # (2E, H)

        # ── Message Passing (NO coords in this path) ──
        for mp in self.mp_layers:
            h = h + mp(h, data.edge_index, e)        # (N, H)

        # ── Decoder: combine latent + coords ──
        # Coords enter HERE — only through the decoder
        decoder_input = torch.cat([h, self._coords_2d], dim=-1)
        # shape: (N, H + 2)

        pred = self.decoder(decoder_input)           # (N, 3)

        # ── Hard BC ──
        pred = self._apply_bc(pred, data)

        return pred

    def get_coords(self):
        """Return coords tensor in autograd graph."""
        return self._coords_2d    # (N, 2): [x, z]

    def _apply_bc(self, pred, data):
        pred = pred.clone()
        disp_mask = (1.0 - data.bc_disp)     # (N, 1)
        rot_mask  = (1.0 - data.bc_rot)       # (N, 1)
        pred[:, 0:2] *= disp_mask             # u_x, u_z
        pred[:, 2:3] *= rot_mask              # φ
        return pred

    def count_params(self):
        return sum(p.numel() for p in self.parameters())