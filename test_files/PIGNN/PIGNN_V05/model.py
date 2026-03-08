"""
=================================================================
model.py — Hybrid GNN-PINN for Correct Autodiff
=================================================================

Architecture:
    Phase 1 (GNN):     data.x → Encoder → Message Passing → h_i
                        (learns structural context per node)
    
    Phase 2 (PINN):    Decoder(coords_i, h_i) → pred_i
                        (pointwise, differentiable w.r.t. coords)

The two phases use SEPARATE coordinate tensors:
    data.x[:, 0:3]     → goes through GNN (spatial learning)
    data.coords.clone() → goes through Decoder (autodiff target)

Since h_i has no grad connection to the decoder's coords,
the Jacobian ∂pred/∂coords is BLOCK-DIAGONAL.
This means autograd.grad gives EXACT spatial derivatives.
=================================================================
"""

import torch
import torch.nn as nn
from layers import MLP, GraphNetworkLayer


class PIGNN_Hybrid(nn.Module):
    """
    Two-phase architecture for physics-informed GNN
    with correct autodiff spatial derivatives.
    
    Phase 1 - GNN Context:
        h_i = GNN(data.x, edge_index, edge_attr)
        This captures: what loads act on this node?
                       what are the boundary conditions?
                       what material properties surround it?
    
    Phase 2 - Pointwise Decoder:
        u_i = Decoder(x_i, y_i, z_i, h_i)
        This produces: displacement field value at location (x,y,z)
                       conditioned on structural context h
    
    Key property:
        ∂u_i/∂x_j = 0  for i ≠ j
        → autograd.grad gives TRUE spatial derivatives
        → all physics PDEs can be enforced via autodiff
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

        # ══════════════════════════════════
        # PHASE 1: GNN (context extraction)
        # ══════════════════════════════════
        self.node_encoder = MLP(node_in_dim, [H], H, act=True)
        self.edge_encoder = MLP(edge_in_dim, [H], H, act=True)

        self.mp_layers = nn.ModuleList([
            GraphNetworkLayer(H, H, aggr='add')
            for _ in range(n_layers)
        ])

        # ══════════════════════════════════
        # PHASE 2: Pointwise Decoder
        # ══════════════════════════════════
        # Input: [x, y, z (3) | h_context (H)] = 3 + H
        # Output: [u_x, u_z, φ] = 3
        #
        # This is a PINN-like network:
        # same MLP applied independently to each node
        # → block-diagonal Jacobian w.r.t. coords
        self.decoder = MLP(3 + H, [H, H, 64], out_dim, act=True)

    def forward(self, data):
        """
        Forward pass with two-phase architecture.

        Returns:
            pred:   (N, 3) [u_x, u_z, φ]
            coords: (N, 3) with requires_grad (for physics loss)
        """
        # ══════════════════════════════════
        # PHASE 1: GNN context
        # ══════════════════════════════════
        # data.x includes coordinates as features (columns 0-2)
        # These coords flow through GNN for spatial awareness
        # but are NOT the coords we differentiate w.r.t.
        
        h = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)

        for mp in self.mp_layers:
            h = mp(h, data.edge_index, e)
        
        # h is now (N, H) — rich context per node
        # h depends on data.x (including coords in columns 0-2)
        # h does NOT depend on coords_physics (created below)

        # ══════════════════════════════════
        # PHASE 2: Pointwise decoder
        # ══════════════════════════════════
        # Create FRESH coordinate tensor for autodiff
        # This is a DIFFERENT tensor than data.x[:, 0:3]
        # → no gradient connection between h and coords_physics
        
        coords_physics = data.coords.clone().requires_grad_(True)

        # Concatenate: [spatial_location | structural_context]
        decoder_input = torch.cat([coords_physics, h], dim=-1)
        
        # Apply decoder POINTWISE (same MLP per node)
        # pred_i = Decoder(coords_i, h_i)
        # pred_i does NOT depend on coords_j (j≠i) ✓
        pred = self.decoder(decoder_input)   # (N, 3)

        # Hard BC enforcement
        pred = self._apply_bc(pred, data)

        # Store for physics loss
        self._coords = coords_physics

        return pred

    def _apply_bc(self, pred, data):
        """Zero displacements/rotations at constrained nodes."""
        pred = pred.clone()
        pred[:, 0:2] = pred[:, 0:2] * (1.0 - data.bc_disp)
        pred[:, 2:3] = pred[:, 2:3] * (1.0 - data.bc_rot)
        return pred

    def get_coords(self):
        """Get physics coordinates for autodiff."""
        return self._coords

    def count_params(self):
        return sum(p.numel() for p in self.parameters())