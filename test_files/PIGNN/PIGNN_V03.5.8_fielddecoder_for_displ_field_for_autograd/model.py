# """
# =================================================================
# model.py — PIGNN Method C with Hermite Hard BC
# =================================================================

# FIXES FROM V03.5.7:
#   1. Coords BACK in GNN (needed for spatial context in h_i)
#      - No autograd on global coords (all derivatives via ξ)
#      - Safe: we never call autograd.grad(pred, coords)
  
#   2. Hermite hard BC on w(ξ) (prevent trivial w=0 solution)
#      - w(ξ) = H(ξ) + ξ²(1-ξ)² · N_w(ξ)
#      - H(ξ) = Hermite interpolation from GNN's (w_A, w_B, θ_A, θ_B)
#      - N_w(ξ) = field decoder correction (vanishes at boundaries)
#      - Guarantees: w(0)=w_A, w(1)=w_B, w'(0)=θ_A, w'(1)=θ_B
#      - M(ξ) stays as raw MLP output (no hard BC)

# ARCHITECTURE:
#   Stage 1: GNN Backbone
#     - Node features include coords (full 9-dim input)
#     - Encoder → message passing → h_i (H-dim)
#     - Node decoder: h_i → 15 values (ux, uz, θy, face forces)

#   Stage 2: Field Decoder
#     - Input: [ξ, h_A, h_B, E, I, L, q_loc]
#     - Backbone MLP (tanh) → two heads
#     - w-head: MLP outputs N_w(ξ), then w = Hermite + bubble·N_w
#     - M-head: MLP outputs M(ξ) directly (no hard BC)

# HERMITE SHAPE FUNCTIONS:
#   H₁(ξ) = 1 - 3ξ² + 2ξ³        → H₁(0)=1, H₁(1)=0, H₁'(0)=0, H₁'(1)=0
#   H₂(ξ) = ξ - 2ξ² + ξ³         → H₂(0)=0, H₂(1)=0, H₂'(0)=1, H₂'(1)=0
#   H₃(ξ) = 3ξ² - 2ξ³            → H₃(0)=0, H₃(1)=1, H₃'(0)=0, H₃'(1)=0
#   H₄(ξ) = -ξ² + ξ³             → H₄(0)=0, H₄(1)=0, H₄'(0)=0, H₄'(1)=1

#   H(ξ) = w_A·H₁ + θ_A·L·H₂ + w_B·H₃ + θ_B·L·H₄

#   Note: θ is slope dw/dx, but Hermite needs dw/dξ = θ·L
#   So we multiply θ by L in the Hermite formula.

# BUBBLE FUNCTION:
#   B(ξ) = ξ²(1-ξ)²
#   B(0) = 0,  B(1) = 0
#   B'(0) = 0, B'(1) = 0
  
#   So w(ξ) = H(ξ) + B(ξ)·N_w(ξ) satisfies:
#     w(0) = w_A  ✓    w'(0) = θ_A·L  ✓
#     w(1) = w_B  ✓    w'(1) = θ_B·L  ✓
# =================================================================
# """

# import torch
# import torch.nn as nn
# from layers import MLP, GraphNetworkLayer


# # ================================================================
# # FIELD DECODER (Method C with Hermite)
# # ================================================================

# class FieldDecoder(nn.Module):
#     """
#     Continuous field decoder with Hermite hard BC on displacement.

#     w(ξ) = H(ξ; w_A, w_B, θ_A, θ_B) + ξ²(1-ξ)² · N_w(ξ)
#     M(ξ) = raw MLP output (no hard BC)

#     The Hermite base H(ξ) guarantees boundary consistency
#     with GNN node predictions. The MLP correction N_w(ξ)
#     can only add interior variation — it cannot violate
#     the boundary values.

#     Args:
#         hidden_dim:     GNN hidden dimension H
#         decoder_hidden: field decoder hidden layer dims
#     """

#     def __init__(self, hidden_dim, decoder_hidden=None):
#         super().__init__()
#         H = hidden_dim

#         if decoder_hidden is None:
#             decoder_hidden = [128, 64]

#         # Input: ξ(1) + h_A(H) + h_B(H) + E(1) + I(1) + L(1) + q(1)
#         input_dim = 1 + 2 * H + 4

#         # ── Shared backbone (tanh for smooth derivatives) ──
#         layers = []
#         dims = [input_dim] + list(decoder_hidden)
#         for i in range(len(dims) - 1):
#             layers.append(nn.Linear(dims[i], dims[i + 1]))
#             layers.append(nn.Tanh())
#         self.backbone = nn.Sequential(*layers)

#         backbone_out = decoder_hidden[-1]

#         # ── Output heads ──
#         self.w_head = nn.Linear(backbone_out, 1)   # N_w(ξ) correction
#         self.M_head = nn.Linear(backbone_out, 1)   # M(ξ) direct

#     def _hermite_basis(self, xi):
#         """
#         Compute Hermite shape functions at ξ.

#         Args:
#             xi: (B, 1) collocation points ξ ∈ [0, 1]

#         Returns:
#             H1, H2, H3, H4: each (B, 1)

#         Properties:
#             H1(0)=1, H1(1)=0, H1'(0)=0, H1'(1)=0  → multiplies w_A
#             H2(0)=0, H2(1)=0, H2'(0)=1, H2'(1)=0  → multiplies θ_A·L
#             H3(0)=0, H3(1)=1, H3'(0)=0, H3'(1)=0  → multiplies w_B
#             H4(0)=0, H4(1)=0, H4'(0)=0, H4'(1)=1  → multiplies θ_B·L
#         """
#         xi2 = xi * xi
#         xi3 = xi2 * xi

#         H1 = 1.0 - 3.0 * xi2 + 2.0 * xi3        # (B, 1)
#         H2 = xi - 2.0 * xi2 + xi3                  # (B, 1)
#         H3 = 3.0 * xi2 - 2.0 * xi3                 # (B, 1)
#         H4 = -xi2 + xi3                             # (B, 1)

#         return H1, H2, H3, H4

#     def _bubble(self, xi):
#         """
#         Bubble function: B(ξ) = ξ²(1-ξ)²

#         Properties:
#             B(0) = 0,  B(1) = 0
#             B'(0) = 0, B'(1) = 0

#         So w = H + B·N_w preserves all 4 boundary conditions.

#         Args:
#             xi: (B, 1)

#         Returns:
#             B: (B, 1)
#         """
#         return (xi ** 2) * ((1.0 - xi) ** 2)

#     def forward(self, xi, h_A, h_B, E_val, I_val, L_val, q_val,
#                 w_A, w_B, theta_A_L, theta_B_L):
#         """
#         Evaluate displacement and moment fields.

#         Args:
#             xi:         (B, 1) collocation points (requires_grad)
#             h_A, h_B:   (B, H) hidden states at element ends
#             E_val:      (B, 1) Young's modulus
#             I_val:      (B, 1) second moment of area
#             L_val:      (B, 1) element length
#             q_val:      (B, 1) local transverse UDL
#             w_A, w_B:   (B, 1) transverse displacement at A, B (from GNN)
#             theta_A_L:  (B, 1) θ_A · L (slope × length, for Hermite)
#             theta_B_L:  (B, 1) θ_B · L

#         Returns:
#             w: (B, 1) transverse displacement (Hermite + correction)
#             M: (B, 1) bending moment (raw MLP)
#         """
#         # ── Shared backbone ──
#         inp = torch.cat([xi, h_A, h_B, E_val, I_val, L_val, q_val],
#                         dim=-1)
#         features = self.backbone(inp)

#         # ── MLP outputs ──
#         N_w = self.w_head(features)     # (B, 1) correction for w
#         M   = self.M_head(features)     # (B, 1) moment (direct)

#         # ── Hermite base ──
#         H1, H2, H3, H4 = self._hermite_basis(xi)

#         H = (w_A * H1 +
#              theta_A_L * H2 +
#              w_B * H3 +
#              theta_B_L * H4)            # (B, 1)

#         # ── Bubble correction ──
#         B = self._bubble(xi)            # (B, 1)

#         # ── Final displacement ──
#         w = H + B * N_w                  # (B, 1)

#         return w, M


# # ================================================================
# # PIGNN MODEL
# # ================================================================

# class PIGNN(nn.Module):
#     """
#     PIGNN with Method C field decoder and Hermite hard BC.

#     Stage 1: GNN backbone (coords IN node features)
#       → h_i hidden states + 15 node outputs

#     Stage 2: Field decoder (ξ-based autograd)
#       → w(ξ) with Hermite hard BC
#       → M(ξ) raw output

#     Args:
#         node_in_dim:    node feature dim (default 9, includes coords)
#         edge_in_dim:    edge feature dim (default 10)
#         hidden_dim:     GNN hidden dim (default 128)
#         n_layers:       message passing layers (default 6)
#         decoder_hidden: field decoder hidden dims
#     """

#     # ── Output layout ──
#     IDX_UX         = 0
#     IDX_UZ         = 1
#     IDX_THETA      = 2
#     IDX_FACE_START = 3
#     IDX_FACE_END   = 15
#     N_FACES        = 4
#     N_FACE_DOF     = 3
#     OUT_DIM        = 15

#     # ── Collocation ──
#     COLLOCATION_POINTS = [0.0, 0.25, 0.5, 0.75, 1.0]
#     N_COLLOC = 5

#     def __init__(self,
#                  node_in_dim=9,
#                  edge_in_dim=10,
#                  hidden_dim=128,
#                  n_layers=6,
#                  decoder_hidden=None):
#         super().__init__()
#         H = hidden_dim

#         # ── Stage 1: GNN Backbone ──
#         self.node_encoder = MLP(node_in_dim, [H], H, act=True)
#         self.edge_encoder = MLP(edge_in_dim, [H], H, act=True)

#         self.mp_layers = nn.ModuleList([
#             GraphNetworkLayer(H, H, aggr='add')
#             for _ in range(n_layers)
#         ])

#         # Node decoder: h_i → 15 outputs
#         self.node_decoder = MLP(H, [H, 64], self.OUT_DIM, act=True)

#         # ── Stage 2: Field Decoder ──
#         if decoder_hidden is None:
#             decoder_hidden = [128, 64]
#         self.field_decoder = FieldDecoder(H, decoder_hidden)

#         self.hidden_dim = H
#         self._h = None

#     def forward(self, data):
#         """
#         Stage 1: GNN forward pass.

#         Coords ARE in the node features (full 9-dim input).
#         No autograd on global coords — all derivatives via ξ later.

#         Returns:
#             pred: (N, 15) node predictions
#         """
#         # ════════════════════════════════════════════
#         # 1. ENCODE (coords included in features)
#         # ════════════════════════════════════════════
#         # data.x = [x, y, z, bc_d, bc_r, wl_x, wl_y, wl_z, resp]
#         # Coords flow through encoder → MP → meaningful h_i
#         h = self.node_encoder(data.x)             # (N, H)
#         e = self.edge_encoder(data.edge_attr)      # (2E, H)

#         # ════════════════════════════════════════════
#         # 2. MESSAGE PASSING (with spatial context)
#         # ════════════════════════════════════════════
#         for mp in self.mp_layers:
#             h = h + mp(h, data.edge_index, e)

#         # ════════════════════════════════════════════
#         # 3. STORE HIDDEN STATES
#         # ════════════════════════════════════════════
#         self._h = h

#         # ════════════════════════════════════════════
#         # 4. NODE DECODER → 15 outputs
#         # ════════════════════════════════════════════
#         pred = self.node_decoder(h)

#         # ════════════════════════════════════════════
#         # 5. HARD CONSTRAINTS
#         # ════════════════════════════════════════════
#         pred = self._apply_hard_bc(pred, data)
#         pred = self._apply_face_mask(pred, data)

#         return pred

#     def evaluate_field(self, data, pred):
#         """
#         Stage 2: Evaluate field decoder at collocation points.

#         Must be called AFTER forward().

#         For each element, extracts boundary values from GNN predictions
#         and evaluates w(ξ) = Hermite + bubble·N_w and M(ξ).

#         Args:
#             data: PyG Data
#             pred: (N, 15) from forward() — needed for Hermite BCs

#         Returns:
#             field_data: dict with w(ξ), M(ξ), derivatives info, etc.
#         """
#         assert self._h is not None, "Call forward() first"

#         conn = data.connectivity                 # (E, 2)
#         E_elems = conn.shape[0]
#         K = self.N_COLLOC
#         device = self._h.device

#         # ── Element properties ──
#         E_prop = data.prop_E.unsqueeze(-1)        # (E, 1)
#         I_prop = data.prop_I22.unsqueeze(-1)      # (E, 1)
#         L_prop = data.elem_lengths.unsqueeze(-1)  # (E, 1)

#         # ── Local transverse UDL ──
#         elem_dir = data.elem_directions
#         cos_a = elem_dir[:, 0]
#         sin_a = elem_dir[:, 2]
#         q_glob = data.elem_load
#         q_loc = (-q_glob[:, 0] * sin_a +
#                   q_glob[:, 2] * cos_a).unsqueeze(-1)  # (E, 1)

#         # ── Hidden states at element ends ──
#         h_A = self._h[conn[:, 0]]                # (E, H)
#         h_B = self._h[conn[:, 1]]                # (E, H)

#         # ── Boundary values from GNN predictions ──
#         # Extract displacements at A and B nodes
#         disp_A = pred[conn[:, 0], 0:3]            # (E, 3) [ux, uz, θy]
#         disp_B = pred[conn[:, 1], 0:3]            # (E, 3)

#         # Transform to local coordinates
#         # w_L = -ux·sin(α) + uz·cos(α)  (transverse displacement)
#         # θ_L = θy                       (rotation unchanged in 2D)
#         cos_a_e = cos_a.unsqueeze(-1)             # (E, 1)
#         sin_a_e = sin_a.unsqueeze(-1)             # (E, 1)

#         w_A = (-disp_A[:, 0:1] * sin_a_e +
#                 disp_A[:, 1:2] * cos_a_e)         # (E, 1)
#         w_B = (-disp_B[:, 0:1] * sin_a_e +
#                 disp_B[:, 1:2] * cos_a_e)         # (E, 1)

#         theta_A = disp_A[:, 2:3]                   # (E, 1)
#         theta_B = disp_B[:, 2:3]                   # (E, 1)

#         # Hermite needs dw/dξ at boundaries = θ · L
#         # Because dw/dx = θ, and dw/dξ = dw/dx · dx/dξ = θ · L
#         theta_A_L = theta_A * L_prop               # (E, 1)
#         theta_B_L = theta_B * L_prop               # (E, 1)

#         # ── Collocation points ──
#         xi_base = torch.tensor(
#             self.COLLOCATION_POINTS,
#             dtype=torch.float32, device=device
#         )   # (K,)

#         # ── Expand to (E*K, ...) ──
#         xi = xi_base.unsqueeze(0).expand(E_elems, -1).reshape(-1, 1)
#         xi = xi.detach().requires_grad_(True)

#         def expand_E_to_EK(t):
#             """(E, D) → (E*K, D)"""
#             return t.unsqueeze(1).expand(-1, K, -1).reshape(-1, t.shape[-1])

#         h_A_exp       = expand_E_to_EK(h_A)
#         h_B_exp       = expand_E_to_EK(h_B)
#         E_exp         = expand_E_to_EK(E_prop)
#         I_exp         = expand_E_to_EK(I_prop)
#         L_exp         = expand_E_to_EK(L_prop)
#         q_exp         = expand_E_to_EK(q_loc)
#         w_A_exp       = expand_E_to_EK(w_A)
#         w_B_exp       = expand_E_to_EK(w_B)
#         theta_A_L_exp = expand_E_to_EK(theta_A_L)
#         theta_B_L_exp = expand_E_to_EK(theta_B_L)

#         # ── Evaluate field decoder ──
#         w, M = self.field_decoder(
#             xi, h_A_exp, h_B_exp,
#             E_exp, I_exp, L_exp, q_exp,
#             w_A_exp, w_B_exp, theta_A_L_exp, theta_B_L_exp
#         )

#         # ── Metadata ──
#         elem_ids = torch.arange(E_elems, device=device)
#         elem_ids = elem_ids.unsqueeze(1).expand(-1, K).reshape(-1)
#         xi_vals = xi_base.unsqueeze(0).expand(E_elems, -1).reshape(-1)

#         field_data = {
#             'xi':       xi,            # (E*K, 1) requires_grad
#             'w':        w,             # (E*K, 1)
#             'M':        M,             # (E*K, 1)
#             'E_val':    E_exp,         # (E*K, 1)
#             'I_val':    I_exp,         # (E*K, 1)
#             'L_val':    L_exp,         # (E*K, 1)
#             'q_val':    q_exp,         # (E*K, 1)
#             'elem_ids': elem_ids,      # (E*K,)
#             'xi_vals':  xi_vals,       # (E*K,)
#             'n_elems':  E_elems,
#             'n_colloc': K,
#         }

#         return field_data

#     # ── Hard constraints ──

#     def _apply_hard_bc(self, pred, data):
#         """Hard BC on displacements."""
#         pred = pred.clone()
#         disp_mask = (1.0 - data.bc_disp)
#         rot_mask  = (1.0 - data.bc_rot)
#         pred[:, self.IDX_UX:self.IDX_UZ+1] *= disp_mask
#         pred[:, self.IDX_THETA:self.IDX_THETA+1] *= rot_mask
#         return pred

#     def _apply_face_mask(self, pred, data):
#         """Hard mask on face forces."""
#         pred = pred.clone()
#         force_mask = data.face_mask.repeat_interleave(
#             self.N_FACE_DOF, dim=1
#         )
#         pred[:, self.IDX_FACE_START:self.IDX_FACE_END] *= force_mask
#         return pred

#     def get_hidden_states(self):
#         return self._h

#     def count_params(self):
#         total = sum(p.numel() for p in self.parameters())
#         gnn = (
#             sum(p.numel() for p in self.node_encoder.parameters()) +
#             sum(p.numel() for p in self.edge_encoder.parameters()) +
#             sum(p.numel() for p in self.mp_layers.parameters()) +
#             sum(p.numel() for p in self.node_decoder.parameters())
#         )
#         fd = sum(p.numel() for p in self.field_decoder.parameters())
#         return total, gnn, fd

#     def summary(self):
#         total, gnn, fd = self.count_params()
#         print(f"\n{'='*55}")
#         print(f"  PIGNN Model Summary (Method C + Hermite)")
#         print(f"{'='*55}")
#         print(f"  Stage 1: GNN Backbone")
#         print(f"    Encoder:  9 → H (coords INCLUDED)")
#         print(f"    MP:       {len(self.mp_layers)} layers")
#         print(f"    Decoder:  H → 15 node outputs")
#         print(f"    Params:   {gnn:,}")
#         print(f"")
#         print(f"  Stage 2: Field Decoder (Hermite + MLP)")
#         print(f"    w(xi) = H(xi; w_A,w_B,theta_A,theta_B)")
#         print(f"          + xi^2(1-xi)^2 * N_w(xi)")
#         print(f"    M(xi) = raw MLP output")
#         print(f"    Activation: tanh")
#         print(f"    Colloc: xi in {self.COLLOCATION_POINTS}")
#         print(f"    Params: {fd:,}")
#         print(f"")
#         print(f"  Total params: {total:,}")
#         print(f"")
#         print(f"  Key features:")
#         print(f"    - Coords in GNN (spatial context for h_i)")
#         print(f"    - No autograd on global coords")
#         print(f"    - All derivatives via xi (field decoder)")
#         print(f"    - Hermite guarantees boundary consistency")
#         print(f"    - Max derivative order: 2 (mixed formulation)")
#         print(f"{'='*55}\n")


# # ================================================================
# # QUICK TEST
# # ================================================================

# if __name__ == "__main__":
#     import os
#     from pathlib import Path
#     CURRENT_SUBFOLDER = Path(__file__).resolve().parent
#     os.chdir(CURRENT_SUBFOLDER)

#     print("=" * 60)
#     print("  MODEL TEST (Method C + Hermite)")
#     print("=" * 60)

#     # ── Load graph ──
#     data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
#     data = data_list[0]
#     print(f"  Graph: {data.num_nodes} nodes, {data.n_elements} elements")

#     # ── Create model ──
#     model = PIGNN(
#         node_in_dim=9, edge_in_dim=10,
#         hidden_dim=128, n_layers=6,
#         decoder_hidden=[128, 64],
#     )
#     model.summary()

#     # ── Forward pass ──
#     print("-- Stage 1: GNN forward --")
#     model.train()
#     pred = model(data)
#     print(f"  pred shape: {pred.shape}")
#     print(f"  h_i shape:  {model.get_hidden_states().shape}")

#     # ── BC check ──
#     bc_nodes = torch.where(data.bc_disp.squeeze() > 0.5)[0]
#     if len(bc_nodes) > 0:
#         bc_val = pred[bc_nodes, :2].abs().max()
#         print(f"  BC check: max |ux,uz| at supports = {bc_val:.2e} "
#               f"{'ok' if bc_val < 1e-10 else 'FAIL'}")

#     # ── Field decoder ──
#     print(f"\n-- Stage 2: Field decoder (Hermite) --")
#     field = model.evaluate_field(data, pred)
#     E = data.n_elements
#     K = model.N_COLLOC

#     print(f"  Points: {E}×{K} = {E*K}")
#     print(f"  w range: [{field['w'].min():.6f}, {field['w'].max():.6f}]")
#     print(f"  M range: [{field['M'].min():.6f}, {field['M'].max():.6f}]")

#     # ── Hermite BC check ──
#     print(f"\n-- Hermite boundary check --")
#     w_grid = field['w'].reshape(E, K, 1)
#     w_at_0 = w_grid[:, 0, 0]     # w(ξ=0)
#     w_at_1 = w_grid[:, -1, 0]    # w(ξ=1)

#     # Expected from GNN predictions
#     conn = data.connectivity
#     disp_A = pred[conn[:, 0], 0:3]
#     disp_B = pred[conn[:, 1], 0:3]
#     cos_a = data.elem_directions[:, 0]
#     sin_a = data.elem_directions[:, 2]
#     w_A_expected = -disp_A[:, 0] * sin_a + disp_A[:, 1] * cos_a
#     w_B_expected = -disp_B[:, 0] * sin_a + disp_B[:, 1] * cos_a

#     err_A = (w_at_0 - w_A_expected).abs().max()
#     err_B = (w_at_1 - w_B_expected).abs().max()
#     print(f"  |w(0) - w_A|_max = {err_A:.2e} "
#           f"{'ok' if err_A < 1e-5 else 'FAIL'}")
#     print(f"  |w(1) - w_B|_max = {err_B:.2e} "
#           f"{'ok' if err_B < 1e-5 else 'FAIL'}")

#     # ── Autograd on ξ ──
#     print(f"\n-- Autograd on xi --")
#     xi = field['xi']
#     w = field['w']
#     M = field['M']

#     dw = torch.autograd.grad(w.sum(), xi, create_graph=True, retain_graph=True)[0]
#     d2w = torch.autograd.grad(dw.sum(), xi, create_graph=True, retain_graph=True)[0]
#     dM = torch.autograd.grad(M.sum(), xi, create_graph=True, retain_graph=True)[0]
#     d2M = torch.autograd.grad(dM.sum(), xi, create_graph=True, retain_graph=True)[0]

#     print(f"  dw/dxi  range: [{dw.min():.4e}, {dw.max():.4e}]")
#     print(f"  d2w/dxi2 range: [{d2w.min():.4e}, {d2w.max():.4e}]")
#     print(f"  dM/dxi  range: [{dM.min():.4e}, {dM.max():.4e}]")
#     print(f"  d2M/dxi2 range: [{d2M.min():.4e}, {d2M.max():.4e}]")

#     # ── Backward ──
#     print(f"\n-- Backward test --")
#     loss = w.sum() + M.sum() + pred.sum()
#     loss.backward()
#     grad_norm = sum(p.grad.norm().item() for p in model.parameters()
#                     if p.grad is not None)
#     print(f"  Grad norm: {grad_norm:.4e}")
#     print(f"  Backward OK")

#     print(f"\n{'='*60}")
#     print(f"  MODEL TEST COMPLETE")
#     print(f"{'='*60}")

"""
=================================================================
model.py — PIGNN Method C: Hermite w(ξ) + Anchored M(ξ)
=================================================================

FIXES FROM V03.5.8:
  M(ξ) now anchored to face moments at boundaries:
    M(ξ) = M_A·(1-ξ) + M_B·ξ + ξ·(1-ξ)·N_M(ξ)
  
  This fixes:
    1. L_end trivially zero → now M(0)=M_A, M(1)=M_B by construction
    2. L_Mpp stuck → M has non-trivial shape from linear base
    3. M-head gradient flow → MLP correction changes M'' directly

ARCHITECTURE:
  w(ξ) = H(ξ; w_A,w_B,θ_A,θ_B) + ξ²(1-ξ)²·N_w(ξ)   [Hermite + bubble]
  M(ξ) = M_A·(1-ξ) + M_B·ξ + ξ·(1-ξ)·N_M(ξ)          [Linear + bubble]

  w boundary: w(0)=w_A, w(1)=w_B, w'(0)=θ_A·L, w'(1)=θ_B·L
  M boundary: M(0)=M_A, M(1)=M_B  (but M' is free)

  Note: M bubble is ξ(1-ξ), NOT ξ²(1-ξ)²
    - ξ(1-ξ) vanishes at endpoints → M(0)=M_A, M(1)=M_B  ✓
    - But its derivative does NOT vanish → M'(0), M'(1) are free ✓
    - This is correct: shear V=dM/dx should be predicted, not forced

BOUNDARY VALUES:
  w_A, w_B, θ_A, θ_B: from GNN predicted displacements (transformed to local)
  M_A, M_B: from GNN predicted face moments (transformed to local)
    Sign convention: My_A^loc = -M(0), My_B^loc = +M(1)
    So: M_A = -My_A^loc, M_B = +My_B^loc
=================================================================
"""

import torch
import torch.nn as nn
from layers import MLP, GraphNetworkLayer


# ================================================================
# FIELD DECODER
# ================================================================

class FieldDecoder(nn.Module):
    """
    Field decoder with Hermite w(ξ) and anchored M(ξ).

    w(ξ) = Hermite(ξ; w_A, w_B, θ_A·L, θ_B·L) + ξ²(1-ξ)²·N_w(ξ)
    M(ξ) = M_A·(1-ξ) + M_B·ξ + ξ·(1-ξ)·N_M(ξ)

    Args:
        hidden_dim:     GNN hidden dimension H
        decoder_hidden: MLP hidden layer dims
    """

    def __init__(self, hidden_dim, decoder_hidden=None):
        super().__init__()
        H = hidden_dim

        if decoder_hidden is None:
            decoder_hidden = [128, 64]

        # Input: ξ(1) + h_A(H) + h_B(H) + E(1) + I(1) + L(1) + q(1) = 2H+5
        input_dim = 1 + 2 * H + 4

        # ── Shared backbone (tanh) ──
        layers = []
        dims = [input_dim] + list(decoder_hidden)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        self.backbone = nn.Sequential(*layers)

        backbone_out = decoder_hidden[-1]

        # ── Output heads ──
        self.w_head = nn.Linear(backbone_out, 1)   # N_w(ξ) correction
        self.M_head = nn.Linear(backbone_out, 1)   # N_M(ξ) correction

    def _hermite_basis(self, xi):
        """
        Hermite shape functions for w(ξ).

        H1(ξ) = 1 - 3ξ² + 2ξ³      → w_A
        H2(ξ) = ξ - 2ξ² + ξ³        → θ_A·L
        H3(ξ) = 3ξ² - 2ξ³           → w_B
        H4(ξ) = -ξ² + ξ³            → θ_B·L
        """
        xi2 = xi * xi
        xi3 = xi2 * xi
        H1 = 1.0 - 3.0 * xi2 + 2.0 * xi3
        H2 = xi - 2.0 * xi2 + xi3
        H3 = 3.0 * xi2 - 2.0 * xi3
        H4 = -xi2 + xi3
        return H1, H2, H3, H4

    def forward(self, xi, h_A, h_B, E_val, I_val, L_val, q_val,
                w_A, w_B, theta_A_L, theta_B_L, M_A, M_B):
        """
        Evaluate w(ξ) and M(ξ) at collocation points.

        Args:
            xi:         (B, 1) collocation points (requires_grad)
            h_A, h_B:   (B, H) hidden states
            E_val, I_val, L_val, q_val: (B, 1) element properties
            w_A, w_B:   (B, 1) transverse displacement at A, B
            theta_A_L:  (B, 1) θ_A·L (slope × length for Hermite)
            theta_B_L:  (B, 1) θ_B·L
            M_A, M_B:   (B, 1) bending moment at A, B

        Returns:
            w: (B, 1) transverse displacement
            M: (B, 1) bending moment
        """
        # ── Shared backbone ──
        inp = torch.cat([xi, h_A, h_B, E_val, I_val, L_val, q_val],
                        dim=-1)
        features = self.backbone(inp)

        # ── MLP corrections ──
        N_w = self.w_head(features)    # (B, 1)
        N_M = self.M_head(features)    # (B, 1)

        # ══════════════════════════════════════
        # w(ξ) = Hermite + ξ²(1-ξ)²·N_w
        # ══════════════════════════════════════
        H1, H2, H3, H4 = self._hermite_basis(xi)

        w_hermite = (w_A * H1 +
                     theta_A_L * H2 +
                     w_B * H3 +
                     theta_B_L * H4)

        w_bubble = (xi ** 2) * ((1.0 - xi) ** 2)   # vanishes at 0,1 with deriv

        w = w_hermite + w_bubble * N_w

        # ══════════════════════════════════════
        # M(ξ) = M_A·(1-ξ) + M_B·ξ + ξ(1-ξ)·N_M
        # ══════════════════════════════════════
        M_linear = M_A * (1.0 - xi) + M_B * xi

        M_bubble = xi * (1.0 - xi)    # vanishes at 0,1 (but deriv is free)

        # M = M_linear + M_bubble * N_M
        q_L2 = q_val * L_val * L_val    # characteristic M'' scale
        M = M_linear + M_bubble * N_M * q_L2

        return w, M


# ================================================================
# PIGNN MODEL
# ================================================================

class PIGNN(nn.Module):
    """
    PIGNN with Method C field decoder.

    Stage 1: GNN → h_i + 15 node outputs
    Stage 2: Field decoder → w(ξ), M(ξ) with hard BCs
    """

    IDX_UX         = 0
    IDX_UZ         = 1
    IDX_THETA      = 2
    IDX_FACE_START = 3
    IDX_FACE_END   = 15
    N_FACES        = 4
    N_FACE_DOF     = 3
    OUT_DIM        = 15

    COLLOCATION_POINTS = [0.0, 0.25, 0.5, 0.75, 1.0]
    N_COLLOC = 5

    def __init__(self,
                 node_in_dim=9,
                 edge_in_dim=10,
                 hidden_dim=128,
                 n_layers=6,
                 decoder_hidden=None):
        super().__init__()
        H = hidden_dim

        # ── Stage 1: GNN ──
        self.node_encoder = MLP(node_in_dim, [H], H, act=True)
        self.edge_encoder = MLP(edge_in_dim, [H], H, act=True)
        self.mp_layers = nn.ModuleList([
            GraphNetworkLayer(H, H, aggr='add')
            for _ in range(n_layers)
        ])
        self.node_decoder = MLP(H, [H, 64], self.OUT_DIM, act=True)

        # ── Stage 2: Field Decoder ──
        if decoder_hidden is None:
            decoder_hidden = [128, 64]
        self.field_decoder = FieldDecoder(H, decoder_hidden)

        self.hidden_dim = H
        self._h = None

    def forward(self, data):
        """Stage 1: GNN forward. Coords in features."""
        h = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)
        for mp in self.mp_layers:
            h = h + mp(h, data.edge_index, e)
        self._h = h
        pred = self.node_decoder(h)
        pred = self._apply_hard_bc(pred, data)
        pred = self._apply_face_mask(pred, data)
        return pred

    def evaluate_field(self, data, pred):
        """
        Stage 2: Evaluate field decoder at collocation points.

        Extracts boundary values from GNN predictions:
          w_A, w_B, θ_A, θ_B: from predicted displacements
          M_A, M_B: from predicted face moments

        Sign convention for M at boundaries:
          Face forces (element→node):
            My_A^loc = -M(0)  →  M_A = -My_A^loc
            My_B^loc = +M(1)  →  M_B = +My_B^loc
        """
        assert self._h is not None, "Call forward() first"

        conn = data.connectivity
        E_elems = conn.shape[0]
        K = self.N_COLLOC
        device = self._h.device

        # ── Element properties ──
        E_prop = data.prop_E.unsqueeze(-1)
        I_prop = data.prop_I22.unsqueeze(-1)
        L_prop = data.elem_lengths.unsqueeze(-1)

        # ── Local UDL ──
        elem_dir = data.elem_directions
        cos_a = elem_dir[:, 0]
        sin_a = elem_dir[:, 2]
        q_glob = data.elem_load
        q_loc = (-q_glob[:, 0] * sin_a +
                  q_glob[:, 2] * cos_a).unsqueeze(-1)

        # ── Hidden states ──
        h_A = self._h[conn[:, 0]]
        h_B = self._h[conn[:, 1]]

        # ── Boundary displacements (local) ──
        disp_A = pred[conn[:, 0], 0:3]
        disp_B = pred[conn[:, 1], 0:3]

        cos_a_e = cos_a.unsqueeze(-1)
        sin_a_e = sin_a.unsqueeze(-1)

        w_A = (-disp_A[:, 0:1] * sin_a_e +
                disp_A[:, 1:2] * cos_a_e)
        w_B = (-disp_B[:, 0:1] * sin_a_e +
                disp_B[:, 1:2] * cos_a_e)

        theta_A = disp_A[:, 2:3]
        theta_B = disp_B[:, 2:3]
        theta_A_L = theta_A * L_prop
        theta_B_L = theta_B * L_prop

        # ── Boundary moments (local) ──
        # Need face moments at A-end and B-end of each element
        # From face_forces in pred[:, 3:15]
        face_forces = pred[:, 3:15].reshape(-1, 4, 3)  # (N, 4, 3)

        # Gather face moments for each element
        feid = data.face_element_id   # (N, 4)
        faa  = data.face_is_A_end     # (N, 4)
        fm   = data.face_mask         # (N, 4)

        My_A_face = torch.zeros(E_elems, 1, device=device)
        My_B_face = torch.zeros(E_elems, 1, device=device)

        for f in range(4):
            mask = fm[:, f] > 0.5
            if not mask.any():
                continue
            nodes = torch.where(mask)[0]
            elems = feid[nodes, f]
            is_A  = faa[nodes, f]

            # Face moment in global: face_forces[node, face, 2] = My
            My_global = face_forces[nodes, f, 2:3]  # (K_nodes, 1)

            # Transform to local: My_loc = My_global (unchanged for 2D)
            My_local = My_global

            a_mask = is_A == 1
            if a_mask.any():
                My_A_face[elems[a_mask]] = My_local[a_mask]
            b_mask = is_A == 0
            if b_mask.any():
                My_B_face[elems[b_mask]] = My_local[b_mask]

        # Sign convention: My_A^loc = -M(0), My_B^loc = +M(1)
        # So: M_A = -My_A^loc,  M_B = +My_B^loc
        M_A = -My_A_face     # (E, 1)
        M_B =  My_B_face     # (E, 1)

        # ── Collocation points ──
        xi_base = torch.tensor(
            self.COLLOCATION_POINTS,
            dtype=torch.float32, device=device
        )
        xi = xi_base.unsqueeze(0).expand(E_elems, -1).reshape(-1, 1)
        xi = xi.detach().requires_grad_(True)

        def expand(t):
            return t.unsqueeze(1).expand(-1, K, -1).reshape(-1, t.shape[-1])

        # ── Evaluate ──
        w, M = self.field_decoder(
            xi,
            expand(h_A), expand(h_B),
            expand(E_prop), expand(I_prop), expand(L_prop), expand(q_loc),
            expand(w_A), expand(w_B),
            expand(theta_A_L), expand(theta_B_L),
            expand(M_A), expand(M_B),
        )

        elem_ids = torch.arange(E_elems, device=device)
        elem_ids = elem_ids.unsqueeze(1).expand(-1, K).reshape(-1)
        xi_vals = xi_base.unsqueeze(0).expand(E_elems, -1).reshape(-1)

        return {
            'xi':       xi,
            'w':        w,
            'M':        M,
            'E_val':    expand(E_prop),
            'I_val':    expand(I_prop),
            'L_val':    expand(L_prop),
            'q_val':    expand(q_loc),
            'elem_ids': elem_ids,
            'xi_vals':  xi_vals,
            'n_elems':  E_elems,
            'n_colloc': K,
        }

    # ── Hard constraints ──

    def _apply_hard_bc(self, pred, data):
        pred = pred.clone()
        disp_mask = (1.0 - data.bc_disp)
        rot_mask  = (1.0 - data.bc_rot)
        pred[:, self.IDX_UX:self.IDX_UZ+1] *= disp_mask
        pred[:, self.IDX_THETA:self.IDX_THETA+1] *= rot_mask
        return pred

    def _apply_face_mask(self, pred, data):
        pred = pred.clone()
        force_mask = data.face_mask.repeat_interleave(
            self.N_FACE_DOF, dim=1
        )
        pred[:, self.IDX_FACE_START:self.IDX_FACE_END] *= force_mask
        return pred

    def get_hidden_states(self):
        return self._h

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        gnn = (
            sum(p.numel() for p in self.node_encoder.parameters()) +
            sum(p.numel() for p in self.edge_encoder.parameters()) +
            sum(p.numel() for p in self.mp_layers.parameters()) +
            sum(p.numel() for p in self.node_decoder.parameters())
        )
        fd = sum(p.numel() for p in self.field_decoder.parameters())
        return total, gnn, fd

    def summary(self):
        total, gnn, fd = self.count_params()
        print(f"\n{'='*55}")
        print(f"  PIGNN Model (Method C: Hermite w + Anchored M)")
        print(f"{'='*55}")
        print(f"  Stage 1: GNN ({gnn:,} params)")
        print(f"    Coords in features, {len(self.mp_layers)} MP layers")
        print(f"    Node decoder: H → 15")
        print(f"")
        print(f"  Stage 2: Field Decoder ({fd:,} params)")
        print(f"    w(xi) = Hermite + xi^2(1-xi)^2 * N_w(xi)")
        print(f"    M(xi) = M_A(1-xi) + M_B*xi + xi(1-xi)*N_M(xi)")
        print(f"    Colloc: {self.COLLOCATION_POINTS}")
        print(f"")
        print(f"  Total: {total:,} params")
        print(f"  Hard BCs: w endpoints, M endpoints, displ, face mask")
        print(f"{'='*55}\n")


# ================================================================
# QUICK TEST
# ================================================================

if __name__ == "__main__":
    import os
    from pathlib import Path
    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 60)
    print("  MODEL TEST (Hermite w + Anchored M)")
    print("=" * 60)

    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    data = data_list[0]
    print(f"  Graph: {data.num_nodes} nodes, {data.n_elements} elements")

    model = PIGNN(hidden_dim=128, n_layers=6, decoder_hidden=[128, 64])
    model.summary()

    # Forward
    model.train()
    pred = model(data)
    print(f"  pred: {pred.shape}")

    # Field
    field = model.evaluate_field(data, pred)
    E = data.n_elements
    K = model.N_COLLOC
    print(f"  Field points: {E*K}")
    print(f"  w range: [{field['w'].min():.4e}, {field['w'].max():.4e}]")
    print(f"  M range: [{field['M'].min():.4e}, {field['M'].max():.4e}]")

    # Check M boundary anchoring
    M_grid = field['M'].reshape(E, K, 1)
    print(f"\n  M boundary check:")
    print(f"    M(0) range: [{M_grid[:, 0, 0].min():.4e}, {M_grid[:, 0, 0].max():.4e}]")
    print(f"    M(1) range: [{M_grid[:, -1, 0].min():.4e}, {M_grid[:, -1, 0].max():.4e}]")
    print(f"    M(0.5) range: [{M_grid[:, 2, 0].min():.4e}, {M_grid[:, 2, 0].max():.4e}]")
    print(f"    M(0) != M(0.5): {(M_grid[:, 0, 0] - M_grid[:, 2, 0]).abs().mean():.4e}")

    # Autograd
    xi = field['xi']
    w = field['w']
    M = field['M']
    dw = torch.autograd.grad(w.sum(), xi, create_graph=True, retain_graph=True)[0]
    d2w = torch.autograd.grad(dw.sum(), xi, create_graph=True, retain_graph=True)[0]
    dM = torch.autograd.grad(M.sum(), xi, create_graph=True, retain_graph=True)[0]
    d2M = torch.autograd.grad(dM.sum(), xi, create_graph=True, retain_graph=True)[0]

    print(f"\n  Derivatives:")
    print(f"    dw/dxi:   [{dw.min():.4e}, {dw.max():.4e}]")
    print(f"    d2w/dxi2: [{d2w.min():.4e}, {d2w.max():.4e}]")
    print(f"    dM/dxi:   [{dM.min():.4e}, {dM.max():.4e}]")
    print(f"    d2M/dxi2: [{d2M.min():.4e}, {d2M.max():.4e}]")

    # Backward
    loss = w.sum() + M.sum() + pred.sum()
    loss.backward()
    gn = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"\n  Grad norm: {gn:.4e}")
    print(f"  Backward OK")

    print(f"\n{'='*60}")
    print(f"  TEST COMPLETE")
    print(f"{'='*60}")