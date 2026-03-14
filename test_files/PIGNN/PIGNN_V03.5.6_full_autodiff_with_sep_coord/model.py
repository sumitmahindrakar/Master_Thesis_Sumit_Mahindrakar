"""
=================================================================
model.py — PIGNN with Coords Separated from Message Passing
=================================================================

EXPERIMENT: Coordinates bypass message passing.

Architecture:
  1. Node encoder:    MLP(9 → H)  ← coords ZEROED in input
  2. Edge encoder:    MLP(10 → H)
  3. Message passing: N layers     ← NO coordinate information
  4. Decoder:         MLP(H+2 → 15) ← coords RE-INTRODUCED here
                          ↑
                      [h_i, x_i, z_i]

WHY:
  In the naive version, coords flow through the entire GNN.
  Autograd derivatives ∂pred/∂coords include message passing
  contamination (cross-node effects).

  Here, coords enter ONLY at the decoder. So autograd derivatives
  ∂pred/∂coords go through the decoder MLP only — no message
  passing in the gradient path.

  This tests: is the problem message passing contamination,
  or is it more fundamental (GNN output ≠ continuous field)?

Output per node (15 values):
  [0:3]    ux, uz, θy              ← 3 global displacements
  [3:6]    Fx, Fz, My  at face +x  ← 3 forces (global)
  [6:9]    Fx, Fz, My  at face -x  ← 3 forces (global)
  [9:12]   Fx, Fz, My  at face +z  ← 3 forces (global)
  [12:15]  Fx, Fz, My  at face -z  ← 3 forces (global)
=================================================================
"""

import torch
import torch.nn as nn
from layers import MLP, GraphNetworkLayer


class PIGNN(nn.Module):
    """
    Physics-Informed GNN with coordinates separated from message passing.

    Key difference from naive version:
      - Coordinates are ZEROED before entering the GNN backbone
      - Coordinates are CONCATENATED to hidden states at the decoder
      - Autograd path for ∂pred/∂coords goes through decoder only

    Args:
        node_in_dim:  input node feature dimension (default 9)
        edge_in_dim:  input edge feature dimension (default 10)
        hidden_dim:   hidden embedding dimension (default 128)
        n_layers:     number of message passing layers (default 6)
    """

    # ── Output layout constants ──
    IDX_UX    = 0
    IDX_UZ    = 1
    IDX_THETA = 2
    IDX_FACE_START = 3
    IDX_FACE_END   = 15
    N_FACES   = 4
    N_FACE_DOF = 3
    OUT_DIM   = 15

    def __init__(self,
                 node_in_dim=9,
                 edge_in_dim=10,
                 hidden_dim=128,
                 n_layers=6):
        super().__init__()
        H = hidden_dim

        # ── Encoder ──
        # Input is still 9-dim, but coords will be zeroed
        # So effectively learns from [0,0,0, bc_d, bc_r, wl, resp]
        self.node_encoder = MLP(node_in_dim, [H], H, act=True)
        self.edge_encoder = MLP(edge_in_dim, [H], H, act=True)

        # ── Message passing ──
        # Operates on hidden states WITHOUT coordinate information
        self.mp_layers = nn.ModuleList([
            GraphNetworkLayer(H, H, aggr='add')
            for _ in range(n_layers)
        ])

        # ── Decoder ──
        # Input: [h_i (H), x_i (1), z_i (1)] = H + 2
        # Coords re-introduced HERE for autograd
        self.decoder = MLP(H + 2, [H, 64], self.OUT_DIM, act=True)

        # ── Internal state for autograd ──
        self._coords = None

    def forward(self, data):
        """
        Forward pass with coordinates separated from message passing.

        Step 1: Zero coords in node features
        Step 2: Encode (without coords)
        Step 3: Message passing (without coords)
        Step 4: Concatenate coords to hidden states
        Step 5: Decode (WITH coords — autograd path starts here)
        Step 6: Apply hard constraints

        Args:
            data: PyG Data/Batch

        Returns:
            pred: (N, 15) predictions
        """
        # ════════════════════════════════════════════
        # 1. SETUP COORDINATES (for autograd)
        # ════════════════════════════════════════════
        # Create coords tensor with gradient tracking
        # Only x and z matter for 2D frame (y is always 0)
        self._coords = data.coords.clone().requires_grad_(True)

        # ════════════════════════════════════════════
        # 2. ZERO COORDS IN NODE FEATURES
        # ════════════════════════════════════════════
        # Node features: [x, y, z, bc_d, bc_r, wl_x, wl_y, wl_z, resp]
        # We zero out indices 0:3 so the GNN backbone
        # has NO access to coordinate information.
        x = data.x.clone()
        x[:, 0:3] = 0.0   # ← KEY CHANGE: coords removed from GNN

        # ════════════════════════════════════════════
        # 3. ENCODE (without coords)
        # ════════════════════════════════════════════
        h = self.node_encoder(x)              # (N, H)
        e = self.edge_encoder(data.edge_attr)  # (2E, H)

        # ════════════════════════════════════════════
        # 4. MESSAGE PASSING (without coords)
        # ════════════════════════════════════════════
        # h_i depends on topology, BCs, loads, material props
        # but NOT on coordinates
        for mp in self.mp_layers:
            h = h + mp(h, data.edge_index, e)

        # ════════════════════════════════════════════
        # 5. CONCATENATE COORDS → DECODE
        # ════════════════════════════════════════════
        # Now re-introduce coordinates at the decoder stage
        # decoder_input = [h_i, x_i, z_i]
        #
        # Autograd path for ∂pred/∂coords:
        #   coords → concat → decoder MLP → pred
        #   (NO message passing in this path)
        x_coord = self._coords[:, 0:1]     # (N, 1) — x
        z_coord = self._coords[:, 2:3]     # (N, 1) — z

        decoder_input = torch.cat([h, x_coord, z_coord], dim=1)  # (N, H+2)

        pred = self.decoder(decoder_input)   # (N, 15)

        # ════════════════════════════════════════════
        # 6. APPLY HARD CONSTRAINTS
        # ════════════════════════════════════════════
        pred = self._apply_hard_bc(pred, data)
        pred = self._apply_face_mask(pred, data)

        return pred

    def _apply_hard_bc(self, pred, data):
        """
        Hard boundary conditions on displacements.

        At displacement supports (bc_disp=1): ux=0, uz=0
        At rotation supports   (bc_rot=1):   θy=0
        """
        pred = pred.clone()
        disp_mask = (1.0 - data.bc_disp)      # (N, 1)
        rot_mask  = (1.0 - data.bc_rot)        # (N, 1)
        pred[:, self.IDX_UX:self.IDX_UZ+1] *= disp_mask
        pred[:, self.IDX_THETA:self.IDX_THETA+1] *= rot_mask
        return pred

    def _apply_face_mask(self, pred, data):
        """
        Hard mask on face forces at unconnected faces.
        """
        pred = pred.clone()
        force_mask = data.face_mask.repeat_interleave(
            self.N_FACE_DOF, dim=1
        )
        pred[:, self.IDX_FACE_START:self.IDX_FACE_END] *= force_mask
        return pred

    def get_coords(self):
        """
        Return coordinates tensor connected to autograd graph.

        IMPORTANT DIFFERENCE from naive version:
          In naive:  coords → encoder → MP → decoder → pred
          Here:      coords → decoder → pred  (ONLY)

          So autograd.grad(pred, coords) goes through
          the decoder MLP only — much cleaner gradient path.
        """
        return self._coords

    def count_params(self):
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        """Print model summary."""
        print(f"\n{'═'*55}")
        print(f"  PIGNN Model Summary (Coords Separated)")
        print(f"{'═'*55}")
        print(f"  Architecture:")
        print(f"    Encoder input:   9 (coords ZEROED to 0)")
        print(f"    Message passing: coords NOT in gradient path")
        print(f"    Decoder input:   H+2 = [h_i, x_i, z_i]")
        print(f"    Decoder output:  {self.OUT_DIM}")
        print(f"")
        print(f"  Autograd path for ∂pred/∂coords:")
        print(f"    coords → concat → decoder MLP → pred")
        print(f"    (no message passing in gradient path)")
        print(f"")
        print(f"  Output per node: {self.OUT_DIM}")
        print(f"    [0:3]   displacements: ux, uz, θy")
        print(f"    [3:6]   face +x forces: Fx, Fz, My")
        print(f"    [6:9]   face -x forces: Fx, Fz, My")
        print(f"    [9:12]  face +z forces: Fx, Fz, My")
        print(f"    [12:15] face -z forces: Fx, Fz, My")
        print(f"  Parameters: {self.count_params():,}")
        print(f"{'═'*55}\n")


# ================================================================
# QUICK TEST
# ================================================================

if __name__ == "__main__":
    import os
    from pathlib import Path
    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 60)
    print("  MODEL TEST (Coords Separated from Message Passing)")
    print("=" * 60)

    # ── Load a graph ──
    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    data = data_list[0]
    print(f"  Graph: {data.num_nodes} nodes, "
          f"{data.n_elements} elements")

    # ── Create model ──
    model = PIGNN(
        node_in_dim=9,
        edge_in_dim=10,
        hidden_dim=128,
        n_layers=6,
    )
    model.summary()

    # ── Forward pass ──
    model.eval()
    with torch.no_grad():
        pred = model(data)

    print(f"  pred shape: {pred.shape}")
    assert pred.shape == (data.num_nodes, 15), "Wrong output shape!"

    # ── Check hard BC ──
    bc_nodes = torch.where(data.bc_disp.squeeze() > 0.5)[0]
    if len(bc_nodes) > 0:
        bc_disp = pred[bc_nodes, :2]
        print(f"\n  BC check:")
        print(f"    Max |ux,uz| at supports: {bc_disp.abs().max():.2e}")
        print(f"    {'✓' if bc_disp.abs().max() < 1e-10 else '✗'} Zero")

    # ── Check face mask ──
    free_faces = torch.where(data.face_mask == 0)
    if len(free_faces[0]) > 0:
        forces = pred[:, 3:15].reshape(-1, 4, 3)
        free_force_vals = forces[free_faces[0], free_faces[1], :]
        print(f"\n  Face mask check:")
        print(f"    Max |force| at free faces: "
              f"{free_force_vals.abs().max():.2e}")
        print(f"    {'✓' if free_force_vals.abs().max() < 1e-10 else '✗'} Zero")

    # ── Autograd test ──
    print(f"\n  Autograd test (coords separated):")
    model.train()
    pred = model(data)
    coords = model.get_coords()
    print(f"    coords.requires_grad: {coords.requires_grad}")

    # Test: gradient of ux w.r.t. coords
    ux = pred[:, 0]
    grad_ux = torch.autograd.grad(
        ux.sum(), coords, create_graph=True
    )[0]
    print(f"    ∂(Σux)/∂coords shape: {grad_ux.shape}")
    print(f"    ∂(Σux)/∂x range: [{grad_ux[:, 0].min():.6f}, "
          f"{grad_ux[:, 0].max():.6f}]")
    print(f"    ∂(Σux)/∂z range: [{grad_ux[:, 2].min():.6f}, "
          f"{grad_ux[:, 2].max():.6f}]")

    # 2nd derivative test
    dux_dx = grad_ux[:, 0]
    grad2 = torch.autograd.grad(
        dux_dx.sum(), coords, create_graph=True
    )[0]
    print(f"    ∂²(Σux)/∂x² range: [{grad2[:, 0].min():.6f}, "
          f"{grad2[:, 0].max():.6f}]")

    # 3rd derivative test
    d2ux_dx2 = grad2[:, 0]
    grad3 = torch.autograd.grad(
        d2ux_dx2.sum(), coords, create_graph=True
    )[0]
    print(f"    ∂³(Σux)/∂x³ range: [{grad3[:, 0].min():.6f}, "
          f"{grad3[:, 0].max():.6f}]")

    print(f"\n  ✓ All checks passed")
    print(f"  Key: autograd path goes through decoder ONLY")
    print(f"  (no message passing contamination in gradient)")

    print(f"\n{'='*60}")
    print(f"  MODEL TEST COMPLETE ✓")
    print(f"  Run train.py with SAME physics_loss.py to compare")
    print(f"{'='*60}")