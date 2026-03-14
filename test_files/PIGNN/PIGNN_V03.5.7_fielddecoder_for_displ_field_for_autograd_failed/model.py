"""
=================================================================
model.py — PIGNN with Method C Field Decoder
=================================================================

Two-stage architecture:

STAGE 1: GNN Backbone
  - Node encoder → message passing → hidden states h_i
  - Node decoder → 15 values per node (ux, uz, θy, face forces)
  - Coordinates NOT in message passing

STAGE 2: Field Decoder (per element)
  - Takes element-local coordinate ξ ∈ [0, 1]
  - Concatenates [ξ, h_A, h_B, E, I, L, q_loc]
  - Shared backbone MLP with tanh activation
  - Two output heads:
      w-head: transverse displacement w(ξ)
      M-head: bending moment M(ξ)
  - Autograd on ξ gives TRUE spatial derivatives:
      w'(ξ), w''(ξ)  — slope, curvature
      M'(ξ), M''(ξ)  — shear, load

WHY tanh activation:
  - Smooth and infinitely differentiable
  - Required for reliable 2nd-order autograd
  - SiLU is smooth too, but tanh is standard for PINNs

WHY two heads (mixed formulation):
  - w-head gives displacement → w''(ξ) = curvature (2nd deriv max)
  - M-head gives moment → M''(ξ) = -q (2nd deriv max)
  - Without M-head, would need w''''(ξ) = q/EI (4th deriv!)
  - Mixed formulation: max 2nd derivative, not 4th
=================================================================
"""

import torch
import torch.nn as nn
from layers import MLP, GraphNetworkLayer


# ================================================================
# FIELD DECODER (Method C)
# ================================================================

class FieldDecoder(nn.Module):
    """
    Continuous field decoder for element-level physics.

    Maps (ξ, h_A, h_B, element_props) → (w(ξ), M(ξ))

    Architecture:
        Input: [ξ, h_A, h_B, E, I, L, q_loc]   dim = 1 + 2H + 4
        Shared backbone: MLP with tanh
        w-head: Linear → w(ξ)    (scalar)
        M-head: Linear → M(ξ)    (scalar)

    Usage:
        Called per element, at collocation points ξ ∈ {0, 0.25, 0.5, 0.75, 1}
        Vectorized: all elements × all ξ points in one forward pass

    Args:
        hidden_dim:   GNN hidden dimension H
        decoder_hidden: field decoder hidden layer dims
    """

    def __init__(self, hidden_dim, decoder_hidden=None):
        super().__init__()
        H = hidden_dim

        if decoder_hidden is None:
            decoder_hidden = [128, 64]

        # Input dimension: ξ(1) + h_A(H) + h_B(H) + E(1) + I(1) + L(1) + q(1)
        input_dim = 1 + 2 * H + 4

        # ── Shared backbone (tanh for smooth 2nd derivatives) ──
        layers = []
        dims = [input_dim] + list(decoder_hidden)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        self.backbone = nn.Sequential(*layers)

        backbone_out_dim = decoder_hidden[-1]

        # ── Output heads ──
        self.w_head = nn.Linear(backbone_out_dim, 1)   # w(ξ)
        self.M_head = nn.Linear(backbone_out_dim, 1)   # M(ξ)

    def forward(self, xi, h_A, h_B, E_val, I_val, L_val, q_val):
        """
        Evaluate displacement and moment fields at collocation points.

        Args:
            xi:    (B, 1) collocation points ξ ∈ [0, 1]
            h_A:   (B, H) hidden state at element start node
            h_B:   (B, H) hidden state at element end node
            E_val: (B, 1) Young's modulus
            I_val: (B, 1) second moment of area
            L_val: (B, 1) element length
            q_val: (B, 1) local transverse UDL

            B = n_elements × n_collocation_points

        Returns:
            w: (B, 1) transverse displacement
            M: (B, 1) bending moment
        """
        # Concatenate all inputs
        inp = torch.cat([xi, h_A, h_B, E_val, I_val, L_val, q_val],
                        dim=-1)

        # Shared backbone
        features = self.backbone(inp)

        # Two heads
        w = self.w_head(features)    # (B, 1)
        M = self.M_head(features)    # (B, 1)

        return w, M


# ================================================================
# PIGNN MODEL (with Field Decoder)
# ================================================================

class PIGNN(nn.Module):
    """
    Physics-Informed GNN with Method C field decoder.

    Stage 1: GNN backbone → h_i (hidden states) + 15 node outputs
    Stage 2: Field decoder → w(ξ), M(ξ) per element at collocation points

    The GNN predicts:
      - Node displacements: ux, uz, θy
      - Face forces: (Fx, Fz, My) × 4 faces
      - Hidden states h_i for the field decoder

    The field decoder predicts:
      - w(ξ): transverse displacement field (continuous)
      - M(ξ): bending moment field (continuous)
      - Derivatives via autograd on ξ (true spatial derivatives)

    Args:
        node_in_dim:    input node feature dimension (default 9)
        edge_in_dim:    input edge feature dimension (default 10)
        hidden_dim:     hidden embedding dimension (default 128)
        n_layers:       number of message passing layers (default 6)
        decoder_hidden: field decoder hidden dims (default [128, 64])
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

    # ── Collocation points ──
    COLLOCATION_POINTS = [0.0, 0.25, 0.5, 0.75, 1.0]
    N_COLLOC = 5
    INTERIOR_COLLOC = [0.25, 0.5, 0.75]   # for M'' = -q
    N_INTERIOR = 3

    def __init__(self,
                 node_in_dim=9,
                 edge_in_dim=10,
                 hidden_dim=128,
                 n_layers=6,
                 decoder_hidden=None):
        super().__init__()
        H = hidden_dim

        # ── Stage 1: GNN Backbone ──
        self.node_encoder = MLP(node_in_dim, [H], H, act=True)
        self.edge_encoder = MLP(edge_in_dim, [H], H, act=True)

        self.mp_layers = nn.ModuleList([
            GraphNetworkLayer(H, H, aggr='add')
            for _ in range(n_layers)
        ])

        # Node decoder: h_i → 15 node-level outputs
        # No coords here — just hidden state to outputs
        self.node_decoder = MLP(H, [H, 64], self.OUT_DIM, act=True)

        # ── Stage 2: Field Decoder ──
        if decoder_hidden is None:
            decoder_hidden = [128, 64]
        self.field_decoder = FieldDecoder(H, decoder_hidden)

        # ── Store hidden dim for external access ──
        self.hidden_dim = H

        # ── Internal state ──
        self._h = None          # hidden states (for field decoder)
        self._coords = None     # coordinates (for autograd if needed)

    def forward(self, data):
        """
        Full forward pass.

        Returns:
            pred: (N, 15) node-level predictions
                  [0:3]  = displacements (ux, uz, θy)
                  [3:15] = face forces (4 faces × 3 components)

        Also stores internally:
            self._h: (N, H) hidden states for field decoder
            self._coords: (N, 3) coordinates
        """
        # ════════════════════════════════════════════
        # 1. STORE COORDINATES
        # ════════════════════════════════════════════
        self._coords = data.coords.clone()

        # ════════════════════════════════════════════
        # 2. ZERO COORDS IN NODE FEATURES
        # ════════════════════════════════════════════
        x = data.x.clone()
        x[:, 0:3] = 0.0    # coords not in message passing

        # ════════════════════════════════════════════
        # 3. ENCODE
        # ════════════════════════════════════════════
        h = self.node_encoder(x)
        e = self.edge_encoder(data.edge_attr)

        # ════════════════════════════════════════════
        # 4. MESSAGE PASSING
        # ════════════════════════════════════════════
        for mp in self.mp_layers:
            h = h + mp(h, data.edge_index, e)

        # ════════════════════════════════════════════
        # 5. STORE HIDDEN STATES
        # ════════════════════════════════════════════
        self._h = h    # (N, H) — used by field decoder later

        # ════════════════════════════════════════════
        # 6. NODE DECODER → 15 outputs
        # ════════════════════════════════════════════
        pred = self.node_decoder(h)

        # ════════════════════════════════════════════
        # 7. HARD CONSTRAINTS
        # ════════════════════════════════════════════
        pred = self._apply_hard_bc(pred, data)
        pred = self._apply_face_mask(pred, data)

        return pred

    def evaluate_field(self, data):
        """
        Evaluate the field decoder at collocation points for ALL elements.

        Must be called AFTER forward() (needs self._h).

        For each element e with nodes (A, B):
          - Gathers h_A, h_B from stored hidden states
          - Gathers element properties E, I, L, q_loc
          - Evaluates at ξ ∈ {0, 0.25, 0.5, 0.75, 1}
          - Returns w(ξ), M(ξ) with autograd-enabled ξ

        Returns:
            field_data: dict with:
                'xi':      (E*K, 1) collocation coords (requires_grad)
                'w':       (E*K, 1) transverse displacement
                'M':       (E*K, 1) bending moment
                'E_val':   (E*K, 1) Young's modulus (for physics loss)
                'I_val':   (E*K, 1) second moment of area
                'L_val':   (E*K, 1) element length
                'q_val':   (E*K, 1) local transverse UDL
                'elem_ids':(E*K,)   element index for each row
                'xi_vals': (E*K,)   raw ξ values

            where K = N_COLLOC = 5, so total rows = E × 5
        """
        assert self._h is not None, \
            "Call forward() first to compute hidden states"

        conn = data.connectivity                # (E, 2)
        E_elems = conn.shape[0]
        K = self.N_COLLOC
        device = self._h.device

        # ── Gather hidden states at element ends ──
        h_A = self._h[conn[:, 0]]               # (E, H)
        h_B = self._h[conn[:, 1]]               # (E, H)

        # ── Element properties ──
        E_prop = data.prop_E.unsqueeze(-1)       # (E, 1)
        I_prop = data.prop_I22.unsqueeze(-1)     # (E, 1)
        L_prop = data.elem_lengths.unsqueeze(-1) # (E, 1)

        # ── Local transverse UDL ──
        # Transform global UDL to local transverse component
        # q_loc = -qx·sin(α) + qz·cos(α)
        elem_dir = data.elem_directions          # (E, 3)
        cos_a = elem_dir[:, 0]                   # cos(α)
        sin_a = elem_dir[:, 2]                   # sin(α)
        q_glob = data.elem_load                  # (E, 3)
        q_loc = (-q_glob[:, 0] * sin_a +
                  q_glob[:, 2] * cos_a).unsqueeze(-1)  # (E, 1)

        # ── Build collocation points ──
        # ξ ∈ {0, 0.25, 0.5, 0.75, 1} for each element
        # Total: E × K points
        xi_base = torch.tensor(
            self.COLLOCATION_POINTS,
            dtype=torch.float32, device=device
        )    # (K,)

        # ── Expand everything to (E*K, ...) ──
        # Each element gets K collocation points

        # xi: (E, K) → (E*K, 1) with requires_grad for autograd
        xi = xi_base.unsqueeze(0).expand(E_elems, -1)    # (E, K)
        xi = xi.reshape(-1, 1)                             # (E*K, 1)
        xi = xi.detach().requires_grad_(True)              # enable autograd

        # h_A, h_B: (E, H) → (E*K, H)
        h_A_exp = h_A.unsqueeze(1).expand(-1, K, -1).reshape(-1, self.hidden_dim)
        h_B_exp = h_B.unsqueeze(1).expand(-1, K, -1).reshape(-1, self.hidden_dim)

        # Properties: (E, 1) → (E*K, 1)
        E_exp = E_prop.unsqueeze(1).expand(-1, K, -1).reshape(-1, 1)
        I_exp = I_prop.unsqueeze(1).expand(-1, K, -1).reshape(-1, 1)
        L_exp = L_prop.unsqueeze(1).expand(-1, K, -1).reshape(-1, 1)
        q_exp = q_loc.unsqueeze(1).expand(-1, K, -1).reshape(-1, 1)

        # ── Evaluate field decoder ──
        w, M = self.field_decoder(xi, h_A_exp, h_B_exp,
                                   E_exp, I_exp, L_exp, q_exp)

        # ── Element IDs for each row ──
        elem_ids = torch.arange(E_elems, device=device)
        elem_ids = elem_ids.unsqueeze(1).expand(-1, K).reshape(-1)

        # ── Raw ξ values (for identifying endpoints vs interior) ──
        xi_vals = xi_base.unsqueeze(0).expand(E_elems, -1).reshape(-1)

        field_data = {
            'xi':       xi,           # (E*K, 1) requires_grad
            'w':        w,            # (E*K, 1)
            'M':        M,            # (E*K, 1)
            'E_val':    E_exp,        # (E*K, 1)
            'I_val':    I_exp,        # (E*K, 1)
            'L_val':    L_exp,        # (E*K, 1)
            'q_val':    q_exp,        # (E*K, 1)
            'elem_ids': elem_ids,     # (E*K,)
            'xi_vals':  xi_vals,      # (E*K,)
            'n_elems':  E_elems,
            'n_colloc': K,
        }

        return field_data

    # ── Hard constraints ──

    def _apply_hard_bc(self, pred, data):
        """Hard BC on displacements."""
        pred = pred.clone()
        disp_mask = (1.0 - data.bc_disp)
        rot_mask  = (1.0 - data.bc_rot)
        pred[:, self.IDX_UX:self.IDX_UZ+1] *= disp_mask
        pred[:, self.IDX_THETA:self.IDX_THETA+1] *= rot_mask
        return pred

    def _apply_face_mask(self, pred, data):
        """Hard mask on face forces."""
        pred = pred.clone()
        force_mask = data.face_mask.repeat_interleave(
            self.N_FACE_DOF, dim=1
        )
        pred[:, self.IDX_FACE_START:self.IDX_FACE_END] *= force_mask
        return pred

    def get_hidden_states(self):
        """Return stored hidden states (N, H)."""
        return self._h

    def get_coords(self):
        """Return stored coordinates (N, 3)."""
        return self._coords

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        gnn_params = (
            sum(p.numel() for p in self.node_encoder.parameters()) +
            sum(p.numel() for p in self.edge_encoder.parameters()) +
            sum(p.numel() for p in self.mp_layers.parameters()) +
            sum(p.numel() for p in self.node_decoder.parameters())
        )
        fd_params = sum(p.numel() for p in self.field_decoder.parameters())
        return total, gnn_params, fd_params

    def summary(self):
        total, gnn, fd = self.count_params()
        print(f"\n{'═'*55}")
        print(f"  PIGNN Model Summary (Method C)")
        print(f"{'═'*55}")
        print(f"  Stage 1: GNN Backbone")
        print(f"    Encoder:  9 → H (coords zeroed)")
        print(f"    MP:       {len(self.mp_layers)} layers, no coords")
        print(f"    Decoder:  H → 15 node outputs")
        print(f"    Params:   {gnn:,}")
        print(f"")
        print(f"  Stage 2: Field Decoder")
        print(f"    Input:    [ξ, h_A, h_B, E, I, L, q]")
        print(f"    Backbone: tanh activation")
        print(f"    Heads:    w(ξ), M(ξ)")
        print(f"    Colloc:   ξ ∈ {self.COLLOCATION_POINTS}")
        print(f"    Params:   {fd:,}")
        print(f"")
        print(f"  Total params: {total:,}")
        print(f"")
        print(f"  Node output (15):")
        print(f"    [0:3]   ux, uz, θy")
        print(f"    [3:15]  face forces (4×3)")
        print(f"")
        print(f"  Field output (per element, at {self.N_COLLOC} points):")
        print(f"    w(ξ)  — transverse displacement")
        print(f"    M(ξ)  — bending moment")
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
    print("  MODEL TEST (Method C — Field Decoder)")
    print("=" * 60)

    # ── Load graph ──
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
        decoder_hidden=[128, 64],
    )
    model.summary()

    # ── Forward pass (Stage 1) ──
    print("── Stage 1: GNN forward pass ──")
    model.train()
    pred = model(data)
    print(f"  pred shape: {pred.shape}")
    print(f"  Hidden states: {model.get_hidden_states().shape}")

    # ── Check hard BC ──
    bc_nodes = torch.where(data.bc_disp.squeeze() > 0.5)[0]
    if len(bc_nodes) > 0:
        print(f"  BC check: max |disp| at supports = "
              f"{pred[bc_nodes, :2].abs().max():.2e} "
              f"{'✓' if pred[bc_nodes, :2].abs().max() < 1e-10 else '✗'}")

    # ── Field evaluation (Stage 2) ──
    print(f"\n── Stage 2: Field decoder evaluation ──")
    field = model.evaluate_field(data)
    E_elems = data.n_elements
    K = model.N_COLLOC

    print(f"  Elements: {E_elems}")
    print(f"  Collocation points: {K}")
    print(f"  Total field points: {E_elems * K}")
    print(f"  xi shape:  {field['xi'].shape}")
    print(f"  w shape:   {field['w'].shape}")
    print(f"  M shape:   {field['M'].shape}")
    print(f"  xi requires_grad: {field['xi'].requires_grad}")

    # ── Autograd test on ξ ──
    print(f"\n── Autograd test on ξ ──")
    xi = field['xi']
    w = field['w']
    M = field['M']

    # 1st derivative: dw/dξ
    dw_dxi = torch.autograd.grad(
        w.sum(), xi, create_graph=True, retain_graph=True
    )[0]
    print(f"  dw/dξ shape: {dw_dxi.shape}")
    print(f"  dw/dξ range: [{dw_dxi.min():.6f}, {dw_dxi.max():.6f}]")

    # 2nd derivative: d²w/dξ²
    d2w_dxi2 = torch.autograd.grad(
        dw_dxi.sum(), xi, create_graph=True, retain_graph=True
    )[0]
    print(f"  d²w/dξ² shape: {d2w_dxi2.shape}")
    print(f"  d²w/dξ² range: [{d2w_dxi2.min():.6f}, {d2w_dxi2.max():.6f}]")

    # 1st derivative of M: dM/dξ
    dM_dxi = torch.autograd.grad(
        M.sum(), xi, create_graph=True, retain_graph=True
    )[0]
    print(f"  dM/dξ shape: {dM_dxi.shape}")
    print(f"  dM/dξ range: [{dM_dxi.min():.6f}, {dM_dxi.max():.6f}]")

    # 2nd derivative of M: d²M/dξ²
    d2M_dxi2 = torch.autograd.grad(
        dM_dxi.sum(), xi, create_graph=True, retain_graph=True
    )[0]
    print(f"  d²M/dξ² shape: {d2M_dxi2.shape}")
    print(f"  d²M/dξ² range: [{d2M_dxi2.min():.6f}, {d2M_dxi2.max():.6f}]")

    print(f"\n  ✓ All autograd derivatives computed successfully")
    print(f"  Key: derivatives are w.r.t. ξ (element-local)")
    print(f"  NOT w.r.t. global coords (no GNN contamination)")

    # ── Backward test ──
    print(f"\n── Backward test ──")
    loss = w.sum() + M.sum() + pred.sum()
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item()
        for p in model.parameters()
        if p.grad is not None
    )
    print(f"  Gradient norm: {grad_norm:.6e}")
    print(f"  ✓ Full backward pass works")

    print(f"\n{'='*60}")
    print(f"  MODEL TEST COMPLETE ✓")
    print(f"  Ready for physics_loss.py (Method C)")
    print(f"{'='*60}")