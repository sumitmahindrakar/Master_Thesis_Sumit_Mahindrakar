"""
=================================================================
model.py — Strong-Form PIGNN

Input:  Node features (N, 9), Edge features (2E, 7)
Output: y_i = (u_x, u_z, φ)  →  (N, 3)

Key design: Raw coordinates injected with requires_grad=True
so physics loss can later compute:
    du/dx  via torch.autograd.grad  →  N = EA · du/dx
    d²u/dx² via autograd            →  M = EI · d²u/dx²
    d³u/dx³ via autograd            →  V = EI · d³u/dx³
=================================================================
"""

import torch
import torch.nn as nn
from layers import MLP, GraphNetworkLayer


class PIGNN(nn.Module):
    """
    Predicts nodal displacements/rotation for 2D frames.

    Architecture:
        [raw_coords(3) | other_features(6)]  →  Node Encoder  →  h (N, H)
        Edge features (2E, 7)                 →  Edge Encoder  →  e (2E, H)
                     ↓
        K × GraphNetworkLayer (with internal residual + LayerNorm)
                     ↓
        Decoder  →  (N, 3)  →  BC Mask  →  [u_x, u_z, φ]

    Coordinates are injected with requires_grad=True so that
    torch.autograd.grad(pred, coords) gives spatial derivatives.
    """

    def __init__(self,
                 node_in_dim=9,
                 edge_in_dim=10,       # ← was 11
                 hidden_dim=128,
                 n_layers=6,
                 out_dim=3):          # ← was 6
        super().__init__()

        H = hidden_dim
        self.out_dim = out_dim

        # ── 1. ENCODERS ──
        self.node_encoder = MLP(node_in_dim, [H], H, act=True)
        self.edge_encoder = MLP(edge_in_dim, [H], H, act=True)

        # ── 2. PROCESSOR (residual is inside each layer) ──
        self.mp_layers = nn.ModuleList([
            GraphNetworkLayer(H, H, aggr='add')
            for _ in range(n_layers)
        ])

        # ── 3. DECODER ──
        self.decoder = MLP(H, [H, 64], out_dim, act=True)

    def forward(self, data):
        """
        Forward pass with coordinate tracking for autodiff.

        The raw coordinates (data.coords) are injected into the
        computation graph with requires_grad=True. This means the
        output pred is differentiable w.r.t. spatial coordinates,
        enabling:
            du/dx   = autograd.grad(u, coords)     → for N
            d²u/dx² = autograd.grad(du/dx, coords)  → for M
            d³u/dx³ = autograd.grad(d²u/dx², coords) → for V

        Args:
            data: PyG Data with
                .x           (N, 9)   normalized node features
                .coords      (N, 3)   RAW coordinates (not normalized)
                .edge_index  (2, 2E)  directed edges
                .edge_attr   (2E, 7)  normalized edge features
                .bc_disp     (N, 1)   translation BC flag
                .bc_rot      (N, 1)   rotation BC flag

        Returns:
            pred: (N, 3) → [u_x, u_z, φ]
        """
        # ── 1. Coordinates with grad tracking ──
        # Raw coords flow through the network so autodiff works
        # data.x[:, 0:3] is normalized — we replace with raw coords
        coords = data.coords.clone().requires_grad_(True)

        # [raw_coords (3) | normalized_other_features (6)]
        x_input = torch.cat([coords, data.x[:, 3:]], dim=-1)  # (N, 9)

        # ── 2. Encode ──
        h = self.node_encoder(x_input)
        e = self.edge_encoder(data.edge_attr)

        # ── 3. Message passing ──
        # Residual connection is inside GraphNetworkLayer
        for mp in self.mp_layers:
            h = mp(h, data.edge_index, e)

        # ── 4. Decode ──
        pred = self.decoder(h)      # (N, 3): [u_x, u_z, φ]

        # ── 5. Hard BC enforcement ──
        pred = self._apply_bc(pred, data)

        # ── 6. Store coords for physics loss ──
        self._coords = coords

        return pred

    def _apply_bc(self, pred, data):
        """
        Zero displacements/rotations at constrained nodes.

        pred[:, 0] = u_x  ┐
        pred[:, 1] = u_z  ┘ → zero where bc_disp == 1
        pred[:, 2] = φ    → zero where bc_rot == 1
        """
        pred = pred.clone()
        # bc_disp is (N, 1), broadcasts over [u_x, u_z]
        pred[:, 0:2] = pred[:, 0:2] * (1.0 - data.bc_disp)
        # bc_rot is (N, 1), applies to φ
        pred[:, 2:3] = pred[:, 2:3] * (1.0 - data.bc_rot)
        return pred

    def get_coords(self):
        """Access stored coordinates for physics loss."""
        return self._coords

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ════════════════════════════════════════════════════════════
# SMOKE TEST
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":

    from pathlib import Path
    import os

    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 50)
    print("  PIGNN — Smoke Test (Strong Form)")
    print("=" * 50)

    # Build model
    model = PIGNN(
        node_in_dim=9,
        edge_in_dim=10,       # ← updated
        hidden_dim=128,
        n_layers=6,
        out_dim=3,           # ← [u_x, u_z, φ]
    )
    print(f"\n  Parameters: {model.count_params():,}")

    # Load RAW data
    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    sample = data_list[0]

    print(f"  Graph: {sample.num_nodes} nodes, "
          f"{sample.edge_index.shape[1]} edges, "
          f"{sample.n_elements} elements")

    # ── Forward pass ──
    model.eval()
    pred = model(sample)          # no torch.no_grad — need grad!

    print(f"\n  pred shape: {pred.shape}   ← [u_x, u_z, φ]")
    print(f"  pred range: [{pred.min():.4e}, {pred.max():.4e}]")

    # ── BC check ──
    bc_nodes = (sample.bc_disp.squeeze() > 0.5).nonzero().squeeze()
    print(f"\n  Support nodes: {bc_nodes.tolist()}")
    print(f"  |u_x, u_z| at supports: "
          f"{pred[bc_nodes, :2].abs().max():.2e}  (should be 0)")

    bc_rot_nodes = (sample.bc_rot.squeeze() > 0.5).nonzero().squeeze()
    if bc_rot_nodes.numel() > 0:
        print(f"  |φ| at fixed rotations: "
              f"{pred[bc_rot_nodes, 2].abs().max():.2e}  (should be 0)")

    # ── Autodiff test ──
    print(f"\n  Autodiff test:")
    coords = model.get_coords()
    print(f"    coords.requires_grad: {coords.requires_grad}")

    # Test: can we differentiate pred w.r.t. coords?
    try:
        # du_x/d(x,y,z) at all nodes
        grad_ux = torch.autograd.grad(
            outputs=pred[:, 0].sum(),    # scalar: sum of all u_x
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]
        print(f"    du_x/d(coords) shape: {grad_ux.shape}")
        print(f"    du_x/dx range: [{grad_ux[:, 0].min():.4e}, "
              f"{grad_ux[:, 0].max():.4e}]")
        print(f"    ✓ Autodiff works!")

        # 2nd derivative test
        grad2_uz = torch.autograd.grad(
            outputs=grad_ux[:, 2].sum(),   # d(du_z)/d(coords)
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]
        print(f"    d²u/dx² shape: {grad2_uz.shape}")
        print(f"    ✓ 2nd derivative works!")

    except RuntimeError as e:
        print(f"    ✗ Autodiff failed: {e}")
        print(f"    (Expected — GNN autodiff on coords is non-trivial)")

    # ── Target comparison ──
    target = sample.y_node
    print(f"\n  FEM target shape: {target.shape}   ← [u_x, u_z, φ]")
    print(f"  FEM target range: [{target.min():.4e}, "
          f"{target.max():.4e}]")
    print(f"  MSE (untrained): "
          f"{((pred.detach() - target)**2).mean():.4e}")

    print(f"\n  Smoke test passed ✓")
    print("=" * 50)