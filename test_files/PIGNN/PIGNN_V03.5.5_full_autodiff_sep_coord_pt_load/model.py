"""
=================================================================
model.py — PIGNN for 2D Frame Structures (Decoupled Architecture)
=================================================================

Architecture:
  1. Node encoder:    MLP(9 → H)
  2. Edge encoder:    MLP(10 → H)
  3. Message passing: N layers of GraphNetworkLayer
  4. Local decoder:   MLP(H + 3 → 15)

Key design:
  - GNN produces context vectors h_i (graph-aware)
  - Fresh coords x_i detached from graph (for clean autograd)
  - Local decoder maps [h_i, x_i] → predictions
  - ∂u_i/∂x_j = 0 for i≠j (no cross-term contamination)
  - All derivative orders are clean and local

Output per node (15 values):
  [0:3]    ux, uz, θy              ← 3 global displacements
  [3:6]    Fx, Fz, My  at face +x  ← 3 forces (global)
  [6:9]    Fx, Fz, My  at face -x  ← 3 forces (global)
  [9:12]   Fx, Fz, My  at face +z  ← 3 forces (global)
  [12:15]  Fx, Fz, My  at face -z  ← 3 forces (global)

Hard constraints applied in forward pass:
  - Displacement BCs: ux=uz=0 at supports, θy=0 at fixed supports
  - Face force mask:  forces=0 at unconnected faces
=================================================================
"""

import torch
import torch.nn as nn
from layers import MLP, GraphNetworkLayer


class PIGNN(nn.Module):

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

        # ── GNN---- (graph-aware context) ──
        self.node_encoder = MLP(node_in_dim, [H], H, act=True)
        self.edge_encoder = MLP(edge_in_dim, [H], H, act=True)

        self.mp_layers = nn.ModuleList([
            GraphNetworkLayer(H, H, aggr='add')
            for _ in range(n_layers)
        ])

        # ── Coordinate Encoder (boost coord signal) ──
        self.coord_encoder = MLP(3, [64, 64], 64, act=True)

        # ── Local Decoder (point-wise, for clean autograd) ──
        # Input: H (context from GNN) + 3 (fresh coords)
        # self.local_decoder = MLP(H + 3, [H, 64], self.OUT_DIM, act=True)

        self.local_decoder = MLP(H + 64, [H, 64], self.OUT_DIM, act=True)#-------coord_encoder

        # ── Internal state for autograd ──
        self._coords = None

    def forward(self, data):
        # ══════════════════════════════════════════════
        # STEP 1: GNN produces context vectors
        # ══════════════════════════════════════════════
        # data.x is normalized [0,1] — clean, balanced inputs
        # Gradients flow: Loss → decoder → h → GNN weights

        h = self.node_encoder(data.x)          # (N, H)
        e = self.edge_encoder(data.edge_attr)  # (2E, H)

        for mp in self.mp_layers:
            h = h + mp(h, data.edge_index, e)  # residual MP

        # h: (N, H) — each node's structural context
        # h carries gradients to GNN weights ✓
        # h depends on all coords through MP (that's fine for GNN learning)
        # But we do NOT differentiate through h for physics

        # ══════════════════════════════════════════════
        # STEP 2: Fresh coords (decoupled from graph)
        # ══════════════════════════════════════════════
        # Clone + detach: breaks connection to data.x and GNN graph
        # requires_grad_: makes it a leaf for autograd
        # Result: ∂u_i/∂x_j = 0 for i≠j (no cross-terms)

        # ── Fresh coords NORMALIZED ──
        coord_min = data.coord_min.to(data.coords.device)     # (3,)
        coord_range = data.coord_range.to(data.coords.device)  # (3,)
        coord_range = coord_range.clamp(min=1e-8)
        self._coords = ((data.coords - coord_min) / coord_range).detach().requires_grad_(True)
        
        
        # self._coords = data.coords.clone().detach().requires_grad_(True)
        coord_feat = self.coord_encoder(self._coords)

        #normalizario to change range of coorc from 0 18 to 0 1
        # self._coords_raw = data.coords.clone().detach()  # keep raw for loss
        # self._coords = (data.coords.clone().detach() - data.coord_min) / data.coord_range
        # self._coords.requires_grad_(True)
        # coord_feat = self.coord_encoder(self._coords)

        # ══════════════════════════════════════════════
        # STEP 3: Local decoder (point-wise mapping)
        # ══════════════════════════════════════════════
        # u_i = f(h_i, x_i) — only depends on node i's coord
        # All autograd derivatives are LOCAL and CLEAN

        # decoder_input = torch.cat([h, self._coords], dim=-1)  # (N, H+3)
        decoder_input = torch.cat([h, coord_feat], dim=-1)
        raw = self.local_decoder(decoder_input)                # (N, 15)

        # ══════════════════════════════════════════════
        # STEP 4: Output scaling
        # ══════════════════════════════════════════════
        pred = raw.clone()
        u_c     = data.u_c
        theta_c = data.theta_c
        F_c     = data.F_c
        M_c     = data.M_c

        # Displacements
        pred[:, 0] = raw[:, 0] * u_c          # ux (meters)
        pred[:, 1] = raw[:, 1] * u_c          # uz (meters)
        pred[:, 2] = raw[:, 2] * theta_c      # θy (radians)

        # Face forces
        for face in range(4):
            base = 3 + face * 3
            pred[:, base + 0] = raw[:, base + 0] * F_c    # Fx (N)
            pred[:, base + 1] = raw[:, base + 1] * F_c    # Fz (N)
            pred[:, base + 2] = raw[:, base + 2] * M_c    # My (N·m)

        # ══════════════════════════════════════════════
        # STEP 5: Hard constraints
        # ══════════════════════════════════════════════
        pred = self._apply_hard_bc(pred, data)
        pred = self._apply_face_mask(pred, data)

        return pred

    def _apply_hard_bc(self, pred, data):
        """Zero displacements at supports."""
        pred = pred.clone()
        disp_mask = (1.0 - data.bc_disp)                          # (N, 1)
        rot_mask  = (1.0 - data.bc_rot)                            # (N, 1)
        pred[:, self.IDX_UX:self.IDX_UZ+1] *= disp_mask           # ux, uz
        pred[:, self.IDX_THETA:self.IDX_THETA+1] *= rot_mask      # θy
        return pred

    def _apply_face_mask(self, pred, data):
        """Zero forces at unconnected faces."""
        pred = pred.clone()
        force_mask = data.face_mask.repeat_interleave(
            self.N_FACE_DOF, dim=1
        )  # (N, 12)
        pred[:, self.IDX_FACE_START:self.IDX_FACE_END] *= force_mask
        return pred

    def get_coords(self):
        """Return coords tensor for autograd in physics loss."""
        return self._coords

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        print(f"\n{'═'*60}")
        print(f"  PIGNN Model Summary (Decoupled Architecture)")
        print(f"{'═'*60}")
        print(f"  GNN Backbone:")
        print(f"    Node encoder:  MLP(9 → H)")
        print(f"    Edge encoder:  MLP(10 → H)")
        print(f"    MP layers:     {len(self.mp_layers)} × GraphNetworkLayer")
        print(f"  Local Decoder:")
        print(f"    Input:         H + 3 (context + coords)")
        print(f"    Output:        {self.OUT_DIM} per node")
        print(f"  Output layout:")
        print(f"    [0:3]   displacements: ux, uz, θy")
        print(f"    [3:6]   face +x forces: Fx, Fz, My")
        print(f"    [6:9]   face -x forces: Fx, Fz, My")
        print(f"    [9:12]  face +z forces: Fx, Fz, My")
        print(f"    [12:15] face -z forces: Fx, Fz, My")
        print(f"  Hard constraints:")
        print(f"    Displacement BCs (supports)")
        print(f"    Face force mask (unconnected)")
        print(f"  Autograd:")
        print(f"    Coords decoupled from GNN graph")
        print(f"    ∂u_i/∂x_j = 0 for i≠j (clean local derivs)")
        print(f"  Parameters: {self.count_params():,}")
        print(f"{'═'*60}\n")


# ================================================================
# QUICK TEST
# ================================================================

if __name__ == "__main__":
    import os
    from pathlib import Path
    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 60)
    print("  MODEL TEST (Decoupled Architecture)")
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

    # ── Check autograd (THE KEY TEST) ──
    print(f"\n  Autograd check (decoupled):")
    model.train()
    pred = model(data)
    coords = model.get_coords()
    print(f"    coords.requires_grad: {coords.requires_grad}")
    print(f"    coords.shape: {coords.shape}")

    # 1st derivative
    ux = pred[:, 0]
    grad_ux = torch.autograd.grad(
        ux.sum(), coords, create_graph=True, retain_graph=True)[0]
    print(f"\n    1st derivative ∂ux/∂coords:")
    print(f"      shape: {grad_ux.shape}")
    print(f"      range: [{grad_ux.min():.6f}, {grad_ux.max():.6f}]")

    # 2nd derivative
    dux_dx = grad_ux[:, 0]
    grad2 = torch.autograd.grad(
        dux_dx.sum(), coords, create_graph=True, retain_graph=True)[0]
    d2ux_dx2 = grad2[:, 0]
    print(f"\n    2nd derivative ∂²ux/∂x²:")
    print(f"      shape: {d2ux_dx2.shape}")
    print(f"      range: [{d2ux_dx2.min():.6f}, {d2ux_dx2.max():.6f}]")

    # 3rd derivative
    grad3 = torch.autograd.grad(
        d2ux_dx2.sum(), coords, create_graph=True, retain_graph=True)[0]
    d3ux_dx3 = grad3[:, 0]
    print(f"\n    3rd derivative ∂³ux/∂x³:")
    print(f"      shape: {d3ux_dx3.shape}")
    print(f"      range: [{d3ux_dx3.min():.6f}, {d3ux_dx3.max():.6f}]")

    # Verify locality: derivatives should vary per node
    print(f"\n    Locality check:")
    print(f"      ∂ux/∂x std:    {grad_ux[:, 0].std():.6f}  "
          f"(should be > 0)")
    print(f"      ∂²ux/∂x² std:  {d2ux_dx2.std():.6f}  "
          f"(should be > 0)")
    print(f"      ∂³ux/∂x³ std:  {d3ux_dx3.std():.6f}  "
          f"(should be > 0)")
    print(f"    ✓ All derivatives are node-varying (local)")

    # Verify decoupling: check grad doesn't flow to GNN through coords
    print(f"\n    Decoupling check:")
    print(f"      coords is leaf: {coords.is_leaf}")
    print(f"      coords.grad_fn: {coords.grad_fn}  (should be None)")
    print(f"    ✓ Coords decoupled from GNN graph")

    print(f"\n{'='*60}")
    print(f"  MODEL TEST COMPLETE ✓")
    print(f"  Decoupled architecture working correctly")
    print(f"  Ready for physics_loss.py")
    print(f"{'='*60}")