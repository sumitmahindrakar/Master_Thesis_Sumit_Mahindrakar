"""
model.py — PIGNN for displacement prediction (axial equilibrium)

Input:  RAW node features (N, 9), RAW edge features (2E, 11)
Output: displacement (N, 6)  [ux, uy, uz, rx, ry, rz]

No normalization — physics loss needs real units.
"""

import torch
import torch.nn as nn
from layers import MLP, GraphNetworkLayer


class PIGNN(nn.Module):
    """
    Predicts nodal displacements using physics loss only.
    
    Architecture:
        Node (N, 9)  → Encoder → h (N, H)
        Edge (2E,11) → Encoder → e (2E, H)
             ↓
        K × Message Passing + Residual
             ↓
        Decoder → (N, 6) → BC Mask → [ux uy uz rx ry rz]
    """

    def __init__(self,
                 node_in_dim=9,
                 edge_in_dim=11,
                 hidden_dim=128,
                 n_layers=6,
                 out_dim=6):
        super().__init__()

        H = hidden_dim
        self.out_dim = out_dim

        # ── 1. ENCODERS ──
        self.node_encoder = MLP(node_in_dim, [H], H, act=True)
        self.edge_encoder = MLP(edge_in_dim, [H], H, act=True)

        # ── 2. PROCESSOR ──
        self.mp_layers = nn.ModuleList([
            GraphNetworkLayer(H, H, aggr='add')
            for _ in range(n_layers)
        ])

        # ── 3. DECODER ──
        self.decoder = MLP(H, [H, 64], out_dim, act=True)

    def forward(self, data):
        """
        Args:
            data: PyG Data with
                .x           (N, 9)   node features
                .edge_index  (2, 2E)  directed edges
                .edge_attr   (2E, 11) edge features
                .bc_disp     (N, 1)   translation BC (0 or 1)
                .bc_rot      (N, 1)   rotation BC (0 or 1)

        Returns:
            pred: (N, 6) [ux, uy, uz, rx, ry, rz]
        """
        # Encode
        h = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)

        # Message passing with residual
        for mp in self.mp_layers:
            h = h + mp(h, data.edge_index, e)

        # Decode
        pred = self.decoder(h)

        # Hard BC enforcement
        pred = self._apply_bc(pred, data)

        return pred

    def _apply_bc(self, pred, data):
        """Zero displacement at constrained nodes."""
        pred = pred.clone()
        pred[:, 0:3] *= (1.0 - data.bc_disp)   # translations
        pred[:, 3:6] *= (1.0 - data.bc_rot)     # rotations
        return pred

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ════════════════════════════════════════════════════════════
# SMOKE TEST
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import os
    os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\PIGNN\PIGNN_V04_only_displ")

    print("=" * 50)
    print("  PIGNN — Smoke Test")
    print("=" * 50)

    # Build model
    model = PIGNN(
        node_in_dim=9,
        edge_in_dim=11,
        hidden_dim=128,
        n_layers=6,
        out_dim=6,
    )
    print(f"\n  Parameters: {model.count_params():,}")

    # Load RAW data (not normalized)
    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    sample = data_list[0]

    print(f"  Graph: {sample.num_nodes} nodes, "
          f"{sample.edge_index.shape[1]} edges, "
          f"{sample.n_elements} elements")

    # Forward pass
    model.eval()
    with torch.no_grad():
        pred = model(sample)

    print(f"\n  pred shape: {pred.shape}")
    print(f"  pred range: [{pred.min():.4e}, {pred.max():.4e}]")

    # BC check
    bc_nodes = (sample.bc_disp.squeeze() > 0.5).nonzero().squeeze()
    print(f"\n  Support nodes: {bc_nodes.tolist()}")
    print(f"  |disp| at supports: "
          f"{pred[bc_nodes, :3].abs().max():.2e}  (should be 0)")

    # FEM comparison (just to see scale)
    target = sample.y_node
    print(f"\n  FEM target range: [{target.min():.4e}, "
          f"{target.max():.4e}]")
    print(f"  MSE (untrained): {((pred - target)**2).mean():.4e}")

    # Test physics loss
    from physics_loss import PhysicsLoss
    loss_fn = PhysicsLoss()
    with torch.no_grad():
        loss = loss_fn(pred, sample)
    print(f"  Physics loss (untrained): {loss.item():.4e}")

    print(f"\n  Smoke test passed ✓")
    print(f"  Run train.py next.")
    print("=" * 50)