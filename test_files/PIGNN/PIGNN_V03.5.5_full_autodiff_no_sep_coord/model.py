"""
=================================================================
model.py — PIGNN for 2D Frame Structures
=================================================================

Architecture:
  1. Node encoder:    MLP(9 → H)
  2. Edge encoder:    MLP(10 → H)
  3. Message passing: N layers of GraphNetworkLayer
  4. Decoder:         MLP(H → 15)

Output per node (15 values):
  [0:3]    ux, uz, θy              ← 3 global displacements
  [3:6]    Fx, Fz, My  at face +x  ← 3 forces (global)
  [6:9]    Fx, Fz, My  at face -x  ← 3 forces (global)
  [9:12]   Fx, Fz, My  at face +z  ← 3 forces (global)
  [12:15]  Fx, Fz, My  at face -z  ← 3 forces (global)

Hard constraints applied in forward pass:
  - Displacement BCs: ux=uz=0 at supports, θy=0 at fixed supports
  - Face force mask:  forces=0 at unconnected faces

Coordinate injection:
  - data.coords injected with requires_grad_(True)
  - Replaces x[:, 0:3] in node features
  - Enables autograd for physics loss (∂pred/∂coords)
=================================================================
"""

import torch
import torch.nn as nn
from layers import MLP, GraphNetworkLayer


class PIGNN(nn.Module):

    IDX_UX    = 0
    IDX_UZ    = 1
    IDX_THETA = 2
    IDX_FACE_START = 3   # face forces start at index 3
    IDX_FACE_END   = 15  # face forces end at index 15
    N_FACES   = 4        # +x, -x, +z, -z
    N_FACE_DOF = 3       # Fx, Fz, My per face
    OUT_DIM   = 15       # total output per node

    def __init__(self,
                 node_in_dim=9,
                 edge_in_dim=10,
                 hidden_dim=128,
                 n_layers=6,
                #  out_dim=3
                 ):
        super().__init__()
        H = hidden_dim

        # ── Encoder ──
        self.node_encoder = MLP(node_in_dim, [H], H, act=True)
        self.edge_encoder = MLP(edge_in_dim, [H], H, act=True)

        # ── Message passing ──
        self.mp_layers = nn.ModuleList([
            GraphNetworkLayer(H, H, aggr='add')
            for _ in range(n_layers)
        ])

        # ── Decoder → 15 outputs ──
        self.decoder = MLP(H, [H, 64], self.OUT_DIM, act=True)

        # ── Internal state for autograd ──
        self._coords = None

    def forward(self, data):
        # ── Inject coords with grad into node features ──
        self._coords = data.coords.clone().requires_grad_(True)

        x = data.x.clone()
        x[:, 0:3] = self._coords     # replace coords in features

        # ── Encode ──
        h = self.node_encoder(x)
        e = self.edge_encoder(data.edge_attr)

        # ── Message passing with residual ──
        for mp in self.mp_layers:
            h = h + mp(h, data.edge_index, e)

        # ── Decode ──
        pred = self.decoder(h)

        # ── Hard BC ──
        # pred = self._apply_bc(pred, data)
        pred = self._apply_hard_bc(pred, data)
        pred = self._apply_face_mask(pred, data)

        return pred

    

    def _apply_hard_bc(self, pred, data):
        pred = pred.clone()
        disp_mask = (1.0 - data.bc_disp)     # (N, 1)
        rot_mask  = (1.0 - data.bc_rot)       # (N, 1)
        # pred[:, 0:2] *= disp_mask             # u_x, u_z
        # pred[:, 2:3] *= rot_mask              # φ
        pred[:, self.IDX_UX:self.IDX_UZ+1] *= disp_mask   # ux, uz
        pred[:, self.IDX_THETA:self.IDX_THETA+1] *= rot_mask  # θy

        return pred
    
    def _apply_face_mask(self, pred, data):
        """
        Hard mask on face forces at unconnected faces.

        If face_mask[node, face] == 0 (no element at this face),
        then Fx, Fz, My for that face are forced to zero.

        face_mask: (N, 4) → expand to (N, 12) by repeating ×3
        """
        pred = pred.clone()

        # face_mask (N, 4) → (N, 12): repeat each face flag 3 times
        # [m0, m1, m2, m3] → [m0,m0,m0, m1,m1,m1, m2,m2,m2, m3,m3,m3]
        force_mask = data.face_mask.repeat_interleave(
            self.N_FACE_DOF, dim=1
        )  # (N, 12)

        pred[:, self.IDX_FACE_START:self.IDX_FACE_END] *= force_mask

        return pred

    def get_coords(self):
        """Return coords tensor connected to autograd graph."""
        return self._coords
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def summary(self):
        """Print model summary."""
        print(f"\n{'═'*50}")
        print(f"  PIGNN Model Summary")
        print(f"{'═'*50}")
        print(f"  Output per node: {self.OUT_DIM}")
        print(f"    [0:3]   displacements: ux, uz, θy")
        print(f"    [3:6]   face +x forces: Fx, Fz, My")
        print(f"    [6:9]   face -x forces: Fx, Fz, My")
        print(f"    [9:12]  face +z forces: Fx, Fz, My")
        print(f"    [12:15] face -z forces: Fx, Fz, My")
        print(f"  Parameters: {self.count_params():,}")
        print(f"  Hard BCs: displacement + rotation")
        print(f"  Face mask: unconnected faces → 0")
        print(f"{'═'*50}\n")


# ================================================================
# QUICK TEST
# ================================================================

if __name__ == "__main__":
    import os
    from pathlib import Path
    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 60)
    print("  MODEL TEST")
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
    print(f"    Expected: ({data.num_nodes}, {model.OUT_DIM})")

    # ── Check displacements ──
    disp = pred[:, :3]
    print(f"\n  Displacements (ux, uz, θy):")
    print(f"    Range: [{disp.min():.6f}, {disp.max():.6f}]")

    # ── Check face forces ──
    forces = pred[:, 3:15].reshape(-1, 4, 3)
    print(f"\n  Face forces (4 faces × 3 components):")
    print(f"    Range: [{forces.min():.6f}, {forces.max():.6f}]")

    # ── Check hard BC ──
    bc_nodes = torch.where(data.bc_disp.squeeze() > 0.5)[0]
    if len(bc_nodes) > 0:
        bc_disp = pred[bc_nodes, :2]
        print(f"\n  BC check (support displacements):")
        print(f"    Max |ux,uz| at supports: {bc_disp.abs().max():.2e}")
        print(f"    {'✓' if bc_disp.abs().max() < 1e-10 else '✗'} "
              f"Should be zero")

    # ── Check face mask ──
    free_faces = torch.where(data.face_mask == 0)
    if len(free_faces[0]) > 0:
        free_force_vals = forces[free_faces[0], free_faces[1], :]
        print(f"\n  Face mask check (unconnected faces):")
        print(f"    Max |force| at free faces: "
              f"{free_force_vals.abs().max():.2e}")
        print(f"    {'✓' if free_force_vals.abs().max() < 1e-10 else '✗'} "
              f"Should be zero")

    # ── Check autograd ──
    print(f"\n  Autograd check:")
    model.train()
    pred = model(data)
    coords = model.get_coords()
    print(f"    coords.requires_grad: {coords.requires_grad}")
    print(f"    coords.shape: {coords.shape}")

    # Quick gradient test
    loss = pred[:, 0].sum()  # sum of ux
    grad = torch.autograd.grad(loss, coords, create_graph=True)[0]
    print(f"    ∂(Σux)/∂coords shape: {grad.shape}")
    print(f"    ∂(Σux)/∂coords range: "
          f"[{grad.min():.6f}, {grad.max():.6f}]")
    print(f"    ✓ Autograd works through model")

    print(f"\n{'='*60}")
    print(f"  MODEL TEST COMPLETE ✓")
    print(f"  Ready for Step 3 (physics_loss.py)")
    print(f"{'='*60}")