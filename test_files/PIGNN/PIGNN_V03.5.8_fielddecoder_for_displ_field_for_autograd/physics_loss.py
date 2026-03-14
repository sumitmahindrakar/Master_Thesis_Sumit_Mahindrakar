# """
# =================================================================
# physics_loss.py — Method C Physics Loss (7 Terms)
# =================================================================

# PURPOSE:
#   Physics-based loss using field decoder with ξ-based autograd.
#   Derivatives are w.r.t. element-local coordinate ξ ∈ [0, 1].
#   Maximum derivative order: 2 (mixed formulation).

# ORGANISATION:
#   Section A — Helpers
#     A1. _extract_predictions()     → split 15 node values
#     A2. _get_element_end_data()    → gather A/B node data per element
#     A3. _transform_to_local()      → global → local rotation

#   Section B — Concept 1: Equilibrium (NO autograd, UNCHANGED)
#     B1. _loss_equilibrium()        → L_eq
#     B2. _loss_free_face()          → L_free
#     B3. _loss_support()            → L_sup

#   Section C — Constitutive: Axial (algebraic, UNCHANGED)
#     C1. _loss_axial()              → L_N = EA·Δu/L vs face Fx

#   Section D — Constitutive: Bending (NEW — field decoder + ξ autograd)
#     D0. _compute_field_derivatives() → dw/dξ, d²w/dξ², dM/dξ, d²M/dξ²
#     D1. _loss_moment_curvature()   → L_Mκ:  M = EI·w''  at 5 points
#     D2. _loss_moment_pde()         → L_M'': M'' = -q    at 3 interior
#     D3. _loss_end_forces()         → L_end: end M,V vs face forces

#   Section E — Total
#     forward()                      → weighted sum of 7 losses

# SIGN CONVENTION (same as before):
#   At A (ξ=0): Fx_A = -N, Fz_A = -V, My_A = -M
#   At B (ξ=1): Fx_B = +N, Fz_B = +V, My_B = +M

# COORDINATE TRANSFORM (ξ → physical):
#   x_physical = ξ · L
#   dw/dx = (1/L) · dw/dξ
#   d²w/dx² = (1/L²) · d²w/dξ²
#   d²M/dx² = (1/L²) · d²M/dξ²
# =================================================================
# """

# import torch
# import torch.nn as nn


# class MethodCPhysicsLoss(nn.Module):
#     """
#     Method C physics loss with field decoder.

#     7 loss terms:
#       L_eq   — nodal equilibrium
#       L_free — unconnected face forces = 0
#       L_sup  — support displacements = 0
#       L_N    — axial constitutive (algebraic)
#       L_Mk   — moment-curvature consistency
#       L_Mpp  — moment equilibrium PDE
#       L_end  — element end force consistency

#     Args:
#         w_eq, w_free, w_sup, w_N, w_Mk, w_Mpp, w_end: loss weights
#     """

#     def __init__(self,
#                  w_eq=1.0,
#                  w_free=1.0,
#                  w_sup=1.0,
#                  w_N=1.0,
#                  w_Mk=1.0,
#                  w_Mpp=1.0,
#                  w_end=1.0):
#         super().__init__()
#         self.w_eq   = w_eq
#         self.w_free = w_free
#         self.w_sup  = w_sup
#         self.w_N    = w_N
#         self.w_Mk   = w_Mk
#         self.w_Mpp  = w_Mpp
#         self.w_end  = w_end

#     # ════════════════════════════════════════════════════════
#     # SECTION A: HELPERS (same as naive version)
#     # ════════════════════════════════════════════════════════

#     def _extract_predictions(self, pred):
#         """
#         A1. Split 15-value prediction into displacements and face forces.

#         Returns:
#             disp:        (N, 3)    [ux, uz, θy]
#             face_forces: (N, 4, 3) [Fx, Fz, My] per face (global)
#         """
#         disp = pred[:, 0:3]
#         face_forces = pred[:, 3:15].reshape(-1, 4, 3)
#         return disp, face_forces

#     def _get_element_end_data(self, disp, face_forces, data):
#         """
#         A2. Gather displacement and face force at element ends (vectorized).

#         Returns:
#             disp_A, disp_B: (E, 3) displacements at A, B
#             ff_A, ff_B:     (E, 3) face forces at A, B (global)
#         """
#         conn = data.connectivity
#         feid = data.face_element_id
#         faa  = data.face_is_A_end
#         fm   = data.face_mask
#         E = conn.shape[0]
#         device = disp.device

#         disp_A = disp[conn[:, 0]]
#         disp_B = disp[conn[:, 1]]

#         ff_A = torch.zeros(E, 3, device=device)
#         ff_B = torch.zeros(E, 3, device=device)

#         for f in range(4):
#             mask = fm[:, f] > 0.5
#             if not mask.any():
#                 continue
#             nodes = torch.where(mask)[0]
#             elems = feid[nodes, f]
#             is_A  = faa[nodes, f]
#             forces = face_forces[nodes, f, :]

#             a_mask = is_A == 1
#             if a_mask.any():
#                 ff_A[elems[a_mask]] = forces[a_mask]
#             b_mask = is_A == 0
#             if b_mask.any():
#                 ff_B[elems[b_mask]] = forces[b_mask]

#         return disp_A, disp_B, ff_A, ff_B

#     def _transform_to_local(self, disp_A, disp_B, ff_A, ff_B,
#                              elem_directions):
#         """
#         A3. Global → Local transformation.

#         T(α) = [cos α,  sin α, 0]
#                [-sin α, cos α, 0]
#                [0,      0,     1]

#         cos α = elem_directions[:, 0]
#         sin α = elem_directions[:, 2]
#         """
#         cos_a = elem_directions[:, 0:1]
#         sin_a = elem_directions[:, 2:3]

#         def rotate(v):
#             vx = v[:, 0:1]
#             vz = v[:, 1:2]
#             vt = v[:, 2:3]
#             return torch.cat([
#                  vx * cos_a + vz * sin_a,
#                 -vx * sin_a + vz * cos_a,
#                  vt,
#             ], dim=1)

#         return rotate(disp_A), rotate(disp_B), rotate(ff_A), rotate(ff_B)

#     # ════════════════════════════════════════════════════════
#     # SECTION B: CONCEPT 1 — EQUILIBRIUM (NO AUTOGRAD)
#     #   Identical to naive version
#     # ════════════════════════════════════════════════════════

#     def _loss_equilibrium(self, face_forces, F_ext, bc_disp, bc_rot):
#         """
#         B1. Σ face forces = F_ext at free nodes.
#         """
#         sum_forces = face_forces.sum(dim=1)       # (N, 3)
#         residual = sum_forces - F_ext              # (N, 3)
#         free_mask = (bc_disp.squeeze(-1) < 0.5)   # (N,)

#         res = residual[free_mask]
#         L_eq = (res[:, 0].pow(2) +
#                 res[:, 1].pow(2) +
#                 res[:, 2].pow(2)).mean()
#         return L_eq

#     def _loss_free_face(self, face_forces, face_mask):
#         """
#         B2. Forces at unconnected faces = 0.
#         """
#         free = (face_mask < 0.5).unsqueeze(-1).expand_as(face_forces)
#         free_forces = face_forces[free]
#         if free_forces.numel() == 0:
#             return torch.tensor(0.0, device=face_forces.device)
#         return free_forces.pow(2).mean()

#     def _loss_support(self, disp, bc_disp, bc_rot):
#         """
#         B3. Support displacements = 0.
#         """
#         sup_d = (bc_disp.squeeze(-1) > 0.5)
#         sup_r = (bc_rot.squeeze(-1) > 0.5)
#         loss = torch.tensor(0.0, device=disp.device)
#         if sup_d.any():
#             loss = loss + disp[sup_d, 0].pow(2).mean()
#             loss = loss + disp[sup_d, 1].pow(2).mean()
#         if sup_r.any():
#             loss = loss + disp[sup_r, 2].pow(2).mean()
#         return loss

#     # ════════════════════════════════════════════════════════
#     # SECTION C: AXIAL CONSTITUTIVE (algebraic, NO autograd)
#     # ════════════════════════════════════════════════════════

#     def _loss_axial(self, disp_A_loc, disp_B_loc, ff_A_loc, ff_B_loc, data):
#         """
#         C1. Axial: N = EA · (u_B - u_A) / L

#         Compare with face axial forces:
#           At A: Fx_A^loc = -N  →  (Fx_A + N)² = 0
#           At B: Fx_B^loc = +N  →  (Fx_B - N)² = 0

#         No autograd — pure algebra on predicted end displacements.
#         """
#         EA = data.prop_E * data.prop_A              # (E,)
#         L  = data.elem_lengths                       # (E,)

#         u_A = disp_A_loc[:, 0]                       # (E,) axial disp at A
#         u_B = disp_B_loc[:, 0]                       # (E,) axial disp at B

#         N = EA * (u_B - u_A) / L                     # (E,)

#         Fx_A = ff_A_loc[:, 0]                         # (E,)
#         Fx_B = ff_B_loc[:, 0]                         # (E,)

#         res_A = Fx_A + N        # should be 0
#         res_B = Fx_B - N        # should be 0

#         L_N = (res_A.pow(2) + res_B.pow(2)).mean()
#         return L_N

#     # ════════════════════════════════════════════════════════
#     # SECTION D: BENDING CONSTITUTIVE (field decoder + ξ autograd)
#     #   THIS IS THE NEW PART — Method C
#     # ════════════════════════════════════════════════════════

#     def _compute_field_derivatives(self, field_data):
#         """
#         D0. Compute derivatives of w(ξ) and M(ξ) w.r.t. ξ via autograd.

#         Returns derivatives in PHYSICAL coordinates:
#           dw/dx   = (1/L) · dw/dξ
#           d²w/dx² = (1/L²) · d²w/dξ²
#           dM/dx   = (1/L) · dM/dξ
#           d²M/dx² = (1/L²) · d²M/dξ²

#         All shapes: (E*K, 1) where K = number of collocation points
#         """
#         xi = field_data['xi']      # (E*K, 1) requires_grad
#         w  = field_data['w']       # (E*K, 1)
#         M  = field_data['M']       # (E*K, 1)
#         L  = field_data['L_val']   # (E*K, 1)

#         # ── dw/dξ ──
#         dw_dxi = torch.autograd.grad(
#             w.sum(), xi,
#             create_graph=True, retain_graph=True
#         )[0]    # (E*K, 1)

#         # ── d²w/dξ² ──
#         d2w_dxi2 = torch.autograd.grad(
#             dw_dxi.sum(), xi,
#             create_graph=True, retain_graph=True
#         )[0]    # (E*K, 1)

#         # ── dM/dξ ──
#         dM_dxi = torch.autograd.grad(
#             M.sum(), xi,
#             create_graph=True, retain_graph=True
#         )[0]    # (E*K, 1)

#         # ── d²M/dξ² ──
#         d2M_dxi2 = torch.autograd.grad(
#             dM_dxi.sum(), xi,
#             create_graph=True, retain_graph=True
#         )[0]    # (E*K, 1)

#         # ── Convert ξ-derivatives to physical x-derivatives ──
#         # x = ξ · L  →  dξ = dx/L  →  d/dx = (1/L) · d/dξ
#         dw_dx    = dw_dxi / L                # (1/L) · dw/dξ
#         d2w_dx2  = d2w_dxi2 / (L * L)       # (1/L²) · d²w/dξ²
#         dM_dx    = dM_dxi / L                # (1/L) · dM/dξ
#         d2M_dx2  = d2M_dxi2 / (L * L)       # (1/L²) · d²M/dξ²

#         return {
#             # ξ-derivatives (raw)
#             'dw_dxi':    dw_dxi,
#             'd2w_dxi2':  d2w_dxi2,
#             'dM_dxi':    dM_dxi,
#             'd2M_dxi2':  d2M_dxi2,
#             # Physical derivatives
#             'dw_dx':     dw_dx,
#             'd2w_dx2':   d2w_dx2,
#             'dM_dx':     dM_dx,
#             'd2M_dx2':   d2M_dx2,
#         }

#     def _loss_moment_curvature(self, field_data, derivs):
#         """
#         D1. Moment-curvature consistency: M(ξ) = EI · d²w/dx²

#         Evaluated at ALL 5 collocation points:
#             ξ ∈ {0, 0.25, 0.5, 0.75, 1}

#         Residual: M - EI · d²w/dx² = 0

#         This is the core constitutive law for bending.
#         If this is satisfied, M and w are consistent.

#         Returns:
#             L_Mk: scalar loss
#         """
#         M   = field_data['M']             # (E*K, 1)
#         EI  = field_data['E_val'] * field_data['I_val']  # (E*K, 1)

#         d2w_dx2 = derivs['d2w_dx2']       # (E*K, 1)

#         # Residual: M - EI·κ = 0
#         residual = M - EI * d2w_dx2

#         L_Mk = residual.pow(2).mean()
#         return L_Mk

#     def _loss_moment_pde(self, field_data, derivs):
#         """
#         D2. Moment equilibrium PDE: d²M/dx² = -q

#         Evaluated at INTERIOR collocation points only:
#             ξ ∈ {0.25, 0.5, 0.75}

#         Why interior only:
#           - At endpoints (ξ=0, 1), point loads and reactions
#             create discontinuities in V = dM/dx
#           - The PDE M'' = -q holds only in the element interior

#         Residual: d²M/dx² + q = 0

#         Returns:
#             L_Mpp: scalar loss
#         """
#         d2M_dx2 = derivs['d2M_dx2']       # (E*K, 1)
#         q       = field_data['q_val']      # (E*K, 1)
#         xi_vals = field_data['xi_vals']    # (E*K,)

#         # Interior mask: ξ ∈ {0.25, 0.5, 0.75}
#         # (not 0 or 1)
#         interior = (xi_vals > 0.01) & (xi_vals < 0.99)

#         if not interior.any():
#             return torch.tensor(0.0, device=d2M_dx2.device)

#         # Residual: M'' + q = 0  →  M'' = -q
#         d2M_int = d2M_dx2[interior]
#         q_int   = q[interior]

#         residual = d2M_int + q_int

#         L_Mpp = residual.pow(2).mean()
#         return L_Mpp

#     def _loss_end_forces(self, field_data, derivs,
#                           ff_A_loc, ff_B_loc, data):
#         """
#         D3. Element end forces from field decoder must match face forces.

#         At ξ=0 (A-end):
#             M(0) from field decoder
#             V(0) = dM/dx(0) from field decoder
#             Compare with face forces:
#                 My_A^loc = -M(0)   →  (My_A + M_A)² = 0
#                 Fz_A^loc = -V(0)   →  (Fz_A + V_A)² = 0

#         At ξ=1 (B-end):
#             M(1) from field decoder
#             V(1) = dM/dx(1) from field decoder
#             Compare with face forces:
#                 My_B^loc = +M(1)   →  (My_B - M_B)² = 0
#                 Fz_B^loc = +V(1)   →  (Fz_B - V_B)² = 0

#         Returns:
#             L_end: scalar loss
#         """
#         M       = field_data['M']           # (E*K, 1)
#         dM_dx   = derivs['dM_dx']           # (E*K, 1)
#         xi_vals = field_data['xi_vals']      # (E*K,)
#         E_elems = field_data['n_elems']
#         K       = field_data['n_colloc']

#         # ── Extract values at endpoints ──
#         # Reshape to (E, K, 1) to pick ξ=0 and ξ=1
#         M_grid    = M.reshape(E_elems, K, 1)
#         dM_dx_grid = dM_dx.reshape(E_elems, K, 1)

#         # ξ=0 is index 0, ξ=1 is index K-1
#         M_A = M_grid[:, 0, 0]           # (E,) moment at A
#         M_B = M_grid[:, -1, 0]          # (E,) moment at B
#         V_A = dM_dx_grid[:, 0, 0]       # (E,) shear at A (V = dM/dx)
#         V_B = dM_dx_grid[:, -1, 0]      # (E,) shear at B

#         # ── Face forces (local) ──
#         My_A = ff_A_loc[:, 2]            # (E,) face moment at A
#         My_B = ff_B_loc[:, 2]            # (E,) face moment at B
#         Fz_A = ff_A_loc[:, 1]            # (E,) face shear at A
#         Fz_B = ff_B_loc[:, 1]            # (E,) face shear at B

#         # ── Moment residuals ──
#         # At A: My_A^loc = -M(0)  →  My_A + M_A = 0
#         # At B: My_B^loc = +M(1)  →  My_B - M_B = 0
#         res_M_A = My_A + M_A
#         res_M_B = My_B - M_B

#         # ── Shear residuals ──
#         # At A: Fz_A^loc = -V(0)  →  Fz_A + V_A = 0
#         # At B: Fz_B^loc = +V(1)  →  Fz_B - V_B = 0
#         res_V_A = Fz_A + V_A
#         res_V_B = Fz_B - V_B

#         L_end = (res_M_A.pow(2) + res_M_B.pow(2) +
#                  res_V_A.pow(2) + res_V_B.pow(2)).mean()

#         return L_end

#     # ════════════════════════════════════════════════════════
#     # SECTION E: TOTAL LOSS
#     # ════════════════════════════════════════════════════════

#     def forward(self, model, data):
#         """
#         Compute total physics loss (7 terms).

#         Flow:
#           1. Forward pass → 15 node predictions + hidden states
#           2. Extract displacements and face forces
#           3. Section B: equilibrium losses (no autograd)
#           4. Transform to local coords
#           5. Section C: axial loss (algebraic)
#           6. Evaluate field decoder at collocation points
#           7. Section D: bending losses (ξ autograd)
#           8. Weighted sum

#         Args:
#             model: PIGNN with field decoder
#             data:  PyG Data/Batch

#         Returns:
#             total_loss: scalar (differentiable)
#             loss_dict:  dict of individual loss values (detached)
#             pred:       (N, 15) node predictions
#         """
#         # ════════════════════════════════════
#         # 1. FORWARD PASS (Stage 1)
#         # ════════════════════════════════════
#         pred = model(data)                           # (N, 15)

#         # ════════════════════════════════════
#         # 2. EXTRACT PREDICTIONS
#         # ════════════════════════════════════
#         disp, face_forces = self._extract_predictions(pred)

#         # ════════════════════════════════════
#         # 3. CONCEPT 1: EQUILIBRIUM (no autograd)
#         # ════════════════════════════════════
#         L_eq   = self._loss_equilibrium(
#             face_forces, data.F_ext, data.bc_disp, data.bc_rot
#         )
#         L_free = self._loss_free_face(face_forces, data.face_mask)
#         L_sup  = self._loss_support(disp, data.bc_disp, data.bc_rot)

#         # ════════════════════════════════════
#         # 4. TRANSFORM TO LOCAL
#         # ════════════════════════════════════
#         disp_A, disp_B, ff_A, ff_B = \
#             self._get_element_end_data(disp, face_forces, data)

#         disp_A_loc, disp_B_loc, ff_A_loc, ff_B_loc = \
#             self._transform_to_local(
#                 disp_A, disp_B, ff_A, ff_B,
#                 data.elem_directions
#             )

#         # ════════════════════════════════════
#         # 5. AXIAL CONSTITUTIVE (algebraic)
#         # ════════════════════════════════════
#         L_N = self._loss_axial(
#             disp_A_loc, disp_B_loc,
#             ff_A_loc, ff_B_loc, data
#         )

#         # ════════════════════════════════════
#         # 6. FIELD DECODER (Stage 2)
#         # ════════════════════════════════════
#         # field_data = model.evaluate_field(data)
#         field_data = model.evaluate_field(data, pred)

#         # ════════════════════════════════════
#         # 7. BENDING LOSSES (ξ autograd)
#         # ════════════════════════════════════
#         derivs = self._compute_field_derivatives(field_data)

#         L_Mk  = self._loss_moment_curvature(field_data, derivs)
#         L_Mpp = self._loss_moment_pde(field_data, derivs)
#         L_end = self._loss_end_forces(
#             field_data, derivs, ff_A_loc, ff_B_loc, data
#         )

#         # ════════════════════════════════════
#         # 8. TOTAL (weighted sum)
#         # ════════════════════════════════════
#         total = (self.w_eq   * L_eq
#                + self.w_free * L_free
#                + self.w_sup  * L_sup
#                + self.w_N    * L_N
#                + self.w_Mk   * L_Mk
#                + self.w_Mpp  * L_Mpp
#                + self.w_end  * L_end)

#         loss_dict = {
#             'L_eq':    L_eq.item(),
#             'L_free':  L_free.item(),
#             'L_sup':   L_sup.item(),
#             'L_N':     L_N.item(),
#             'L_Mk':    L_Mk.item(),
#             'L_Mpp':   L_Mpp.item(),
#             'L_end':   L_end.item(),
#             'total':   total.item(),
#         }

#         return total, loss_dict, pred


# # ================================================================
# # QUICK TEST
# # ================================================================

# if __name__ == "__main__":
#     import os
#     from pathlib import Path
#     CURRENT_SUBFOLDER = Path(__file__).resolve().parent
#     os.chdir(CURRENT_SUBFOLDER)

#     from model import PIGNN

#     print("=" * 60)
#     print("  PHYSICS LOSS TEST (Method C — Field Decoder)")
#     print("=" * 60)

#     # ── Load graph ──
#     data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
#     data = data_list[0]
#     print(f"  Graph: {data.num_nodes} nodes, {data.n_elements} elements")

#     # ── Create model and loss ──
#     model = PIGNN(
#         node_in_dim=9, edge_in_dim=10,
#         hidden_dim=64, n_layers=3,
#         decoder_hidden=[64, 32],
#     )
#     loss_fn = MethodCPhysicsLoss(
#         w_eq=1.0, w_free=1.0, w_sup=1.0,
#         w_N=1.0, w_Mk=1.0, w_Mpp=1.0, w_end=1.0,
#     )

#     total_params, gnn_params, fd_params = model.count_params()
#     print(f"  Total params: {total_params:,}")
#     print(f"    GNN:   {gnn_params:,}")
#     print(f"    Field: {fd_params:,}")

#     # ── Forward + loss ──
#     model.train()
#     total_loss, loss_dict, pred = loss_fn(model, data)

#     print(f"\n  Predictions: {pred.shape}")
#     print(f"  Displacements range: [{pred[:, :3].min():.6f}, "
#           f"{pred[:, :3].max():.6f}]")

#     print(f"\n  Individual losses (7 terms):")
#     print(f"  {'─'*50}")
#     print(f"  {'Loss':<10} {'Value':>15}  {'Category'}")
#     print(f"  {'─'*50}")
#     for name, val in loss_dict.items():
#         if name == 'total':
#             continue
#         cat = {
#             'L_eq':   'Equilibrium (Concept 1)',
#             'L_free': 'Free face (Concept 1)',
#             'L_sup':  'Support BC (Concept 1)',
#             'L_N':    'Axial (algebraic)',
#             'L_Mk':   'Moment-curvature (ξ autograd)',
#             'L_Mpp':  'Moment PDE (ξ autograd)',
#             'L_end':  'End forces (ξ autograd)',
#         }.get(name, '')
#         print(f"  {name:<10} {val:>15.6e}  {cat}")
#     print(f"  {'─'*50}")
#     print(f"  {'total':<10} {loss_dict['total']:>15.6e}")

#     # ── Backward test ──
#     print(f"\n  Backward pass...")
#     total_loss.backward()
#     grad_norm = sum(
#         p.grad.norm().item()
#         for p in model.parameters()
#         if p.grad is not None
#     )
#     n_grad = sum(
#         1 for p in model.parameters()
#         if p.grad is not None
#     )
#     n_total = sum(1 for _ in model.parameters())
#     print(f"  Gradient norm: {grad_norm:.6e}")
#     print(f"  Parameters with gradients: {n_grad}/{n_total}")
#     print(f"  ✓ Backward pass successful")

#     # ── Check field decoder derivatives ──
#     print(f"\n  Field decoder derivative check:")
#     model.zero_grad()
#     pred2 = model(data)
#     field = model.evaluate_field(data)
#     derivs = loss_fn._compute_field_derivatives(field)
#     for key, val in derivs.items():
#         print(f"    {key:<12} range: [{val.min():.4e}, {val.max():.4e}]")

#     print(f"\n{'='*60}")
#     print(f"  PHYSICS LOSS TEST COMPLETE ✓ (Method C)")
#     print(f"  Ready for train.py update")
#     print(f"{'='*60}")

"""
=================================================================
physics_loss.py — Method C Physics Loss with NORMALISATION
=================================================================

FIX: Each loss term normalised by characteristic scale so all 
losses are O(1). Without this, L_N ~ 1e20 dominates and 
the optimizer ignores all other losses.

NORMALISATION STRATEGY:
  Each constitutive residual is divided by the expected magnitude
  of the quantity being compared.

  L_N:   residual = (Fx + N)        normalise by N_ref = EA_ref · u_ref / L_ref
  L_Mk:  residual = (M - EI·w'')   normalise by M_ref = EI_ref · w_ref / L_ref²
  L_Mpp: residual = (M'' + q)       normalise by q_ref
  L_end: residual = (My + M)        normalise by M_ref (moment)
         residual = (Fz + V)        normalise by V_ref = M_ref / L_ref
  L_eq:  residual = (ΣF - F_ext)    normalise by F_ref = q_ref · L_ref
  L_N algebraic:                     normalise by N_ref

CHARACTERISTIC VALUES (computed from data):
  E_ref  = mean(E)     ~ 1e11
  A_ref  = mean(A)     ~ 0.05
  I_ref  = mean(I)     ~ 1e-4
  L_ref  = mean(L)     ~ 3
  q_ref  = max(|q|)    ~ 35
  u_ref  = 1e-3        (expected displacement scale)
  w_ref  = 1e-3        (expected transverse displacement)
=================================================================
"""

import torch
import torch.nn as nn


class MethodCPhysicsLoss(nn.Module):
    """
    Method C physics loss with normalised residuals.

    7 loss terms, all normalised to be O(1) initially.
    """

    def __init__(self,
                 w_eq=1.0,
                 w_free=1.0,
                 w_sup=1.0,
                 w_N=1.0,
                 w_Mk=1.0,
                 w_Mpp=1.0,
                 w_end=1.0):
        super().__init__()
        self.w_eq   = w_eq
        self.w_free = w_free
        self.w_sup  = w_sup
        self.w_N    = w_N
        self.w_Mk   = w_Mk
        self.w_Mpp  = w_Mpp
        self.w_end  = w_end

        # Will be set on first forward call
        self._scales_computed = False
        self._N_ref  = 1.0
        self._M_ref  = 1.0
        self._V_ref  = 1.0
        self._q_ref  = 1.0
        self._F_ref  = 1.0

    def _compute_scales(self, data):
        """
        Compute characteristic scales from data.

        Called once on first forward pass.
        All subsequent calls use cached values.
        """
        if self._scales_computed:
            return

        E_ref = data.prop_E.mean().item()
        A_ref = data.prop_A.mean().item()
        I_ref = data.prop_I22.mean().item()
        L_ref = data.elem_lengths.mean().item()

        # UDL magnitude
        q_mag = data.elem_load.norm(dim=1)
        q_ref = q_mag.max().item()
        if q_ref < 1e-10:
            q_ref = 1.0

        # Characteristic force/moment scales
        u_ref = 1e-3    # expected displacement order
        w_ref = 1e-3    # expected transverse displacement

        self._N_ref = max(E_ref * A_ref * u_ref / L_ref, 1e-6)
        self._M_ref = max(E_ref * I_ref * w_ref / (L_ref ** 2), 1e-6)
        self._V_ref = max(self._M_ref / L_ref, 1e-6)
        self._q_ref = max(q_ref, 1e-6)
        self._F_ref = max(q_ref * L_ref, 1e-6)

        self._scales_computed = True

        print(f"\n  Characteristic scales:")
        print(f"    N_ref = {self._N_ref:.4e}  (axial force)")
        print(f"    M_ref = {self._M_ref:.4e}  (moment)")
        print(f"    V_ref = {self._V_ref:.4e}  (shear)")
        print(f"    q_ref = {self._q_ref:.4e}  (UDL)")
        print(f"    F_ref = {self._F_ref:.4e}  (nodal force)")

    # ════════════════════════════════════════════════════════
    # SECTION A: HELPERS
    # ════════════════════════════════════════════════════════

    def _extract_predictions(self, pred):
        disp = pred[:, 0:3]
        face_forces = pred[:, 3:15].reshape(-1, 4, 3)
        return disp, face_forces

    def _get_element_end_data(self, disp, face_forces, data):
        conn = data.connectivity
        feid = data.face_element_id
        faa  = data.face_is_A_end
        fm   = data.face_mask
        E = conn.shape[0]
        device = disp.device

        disp_A = disp[conn[:, 0]]
        disp_B = disp[conn[:, 1]]

        ff_A = torch.zeros(E, 3, device=device)
        ff_B = torch.zeros(E, 3, device=device)

        for f in range(4):
            mask = fm[:, f] > 0.5
            if not mask.any():
                continue
            nodes = torch.where(mask)[0]
            elems = feid[nodes, f]
            is_A  = faa[nodes, f]
            forces = face_forces[nodes, f, :]

            a_mask = is_A == 1
            if a_mask.any():
                ff_A[elems[a_mask]] = forces[a_mask]
            b_mask = is_A == 0
            if b_mask.any():
                ff_B[elems[b_mask]] = forces[b_mask]

        return disp_A, disp_B, ff_A, ff_B

    def _transform_to_local(self, disp_A, disp_B, ff_A, ff_B,
                             elem_directions):
        cos_a = elem_directions[:, 0:1]
        sin_a = elem_directions[:, 2:3]

        def rotate(v):
            vx = v[:, 0:1]
            vz = v[:, 1:2]
            vt = v[:, 2:3]
            return torch.cat([
                 vx * cos_a + vz * sin_a,
                -vx * sin_a + vz * cos_a,
                 vt,
            ], dim=1)

        return rotate(disp_A), rotate(disp_B), rotate(ff_A), rotate(ff_B)

    # ════════════════════════════════════════════════════════
    # SECTION B: CONCEPT 1 — EQUILIBRIUM (normalised)
    # ════════════════════════════════════════════════════════

    def _loss_equilibrium(self, face_forces, F_ext, bc_disp, bc_rot):
        """
        L_eq: Σ face forces = F_ext at free nodes.
        Normalised by F_ref = q_ref · L_ref.
        """
        sum_forces = face_forces.sum(dim=1)
        residual = (sum_forces - F_ext) / self._F_ref    # normalise
        free_mask = (bc_disp.squeeze(-1) < 0.5)

        res = residual[free_mask]
        L_eq = (res[:, 0].pow(2) +
                res[:, 1].pow(2) +
                res[:, 2].pow(2)).mean()
        return L_eq

    def _loss_free_face(self, face_forces, face_mask):
        """L_free: forces at unconnected faces = 0."""
        free = (face_mask < 0.5).unsqueeze(-1).expand_as(face_forces)
        free_forces = face_forces[free] / self._F_ref
        if free_forces.numel() == 0:
            return torch.tensor(0.0, device=face_forces.device)
        return free_forces.pow(2).mean()

    def _loss_support(self, disp, bc_disp, bc_rot):
        """L_sup: support displacements = 0."""
        sup_d = (bc_disp.squeeze(-1) > 0.5)
        sup_r = (bc_rot.squeeze(-1) > 0.5)
        loss = torch.tensor(0.0, device=disp.device)
        if sup_d.any():
            loss = loss + disp[sup_d, 0].pow(2).mean()
            loss = loss + disp[sup_d, 1].pow(2).mean()
        if sup_r.any():
            loss = loss + disp[sup_r, 2].pow(2).mean()
        return loss

    # ════════════════════════════════════════════════════════
    # SECTION C: AXIAL (algebraic, normalised)
    # ════════════════════════════════════════════════════════

    def _loss_axial(self, disp_A_loc, disp_B_loc, ff_A_loc, ff_B_loc, data):
        """
        L_N: N = EA·(u_B - u_A)/L vs face Fx.
        Normalised by N_ref.
        """
        EA = data.prop_E * data.prop_A
        L  = data.elem_lengths

        u_A = disp_A_loc[:, 0]
        u_B = disp_B_loc[:, 0]
        N = EA * (u_B - u_A) / L

        Fx_A = ff_A_loc[:, 0]
        Fx_B = ff_B_loc[:, 0]

        res_A = (Fx_A + N) / self._N_ref
        res_B = (Fx_B - N) / self._N_ref

        L_N = (res_A.pow(2) + res_B.pow(2)).mean()
        return L_N

    # ════════════════════════════════════════════════════════
    # SECTION D: BENDING (field decoder, normalised)
    # ════════════════════════════════════════════════════════

    def _compute_field_derivatives(self, field_data):
        """Compute dw/dξ, d²w/dξ², dM/dξ, d²M/dξ² and convert to physical."""
        xi = field_data['xi']
        w  = field_data['w']
        M  = field_data['M']
        L  = field_data['L_val']

        dw_dxi = torch.autograd.grad(
            w.sum(), xi, create_graph=True, retain_graph=True
        )[0]
        d2w_dxi2 = torch.autograd.grad(
            dw_dxi.sum(), xi, create_graph=True, retain_graph=True
        )[0]
        dM_dxi = torch.autograd.grad(
            M.sum(), xi, create_graph=True, retain_graph=True
        )[0]
        d2M_dxi2 = torch.autograd.grad(
            dM_dxi.sum(), xi, create_graph=True, retain_graph=True
        )[0]

        return {
            'dw_dxi':    dw_dxi,
            'd2w_dxi2':  d2w_dxi2,
            'dM_dxi':    dM_dxi,
            'd2M_dxi2':  d2M_dxi2,
            'dw_dx':     dw_dxi / L,
            'd2w_dx2':   d2w_dxi2 / (L * L),
            'dM_dx':     dM_dxi / L,
            'd2M_dx2':   d2M_dxi2 / (L * L),
        }

    def _loss_moment_curvature(self, field_data, derivs):
        """
        L_Mk: M(ξ) = EI · d²w/dx² at all 5 collocation points.
        Normalised by M_ref.
        """
        M   = field_data['M']
        EI  = field_data['E_val'] * field_data['I_val']
        d2w_dx2 = derivs['d2w_dx2']

        residual = (M - EI * d2w_dx2) / self._M_ref
        L_Mk = residual.pow(2).mean()
        return L_Mk

    def _loss_moment_pde(self, field_data, derivs):
        """
        L_Mpp: d²M/dx² = -q at interior points.
        Normalised by q_ref.
        """
        d2M_dx2 = derivs['d2M_dx2']
        q       = field_data['q_val']
        xi_vals = field_data['xi_vals']

        interior = (xi_vals > 0.01) & (xi_vals < 0.99)
        if not interior.any():
            return torch.tensor(0.0, device=d2M_dx2.device)

        residual = (d2M_dx2[interior] + q[interior]) / self._q_ref
        L_Mpp = residual.pow(2).mean()
        return L_Mpp

    def _loss_end_forces(self, field_data, derivs,
                          ff_A_loc, ff_B_loc, data):
        """
        L_end: M and V at element ends match face forces.
        Moment normalised by M_ref, shear by V_ref.
        """
        M       = field_data['M']
        dM_dx   = derivs['dM_dx']
        E_elems = field_data['n_elems']
        K       = field_data['n_colloc']

        M_grid    = M.reshape(E_elems, K, 1)
        dM_dx_grid = dM_dx.reshape(E_elems, K, 1)

        M_A = M_grid[:, 0, 0]
        M_B = M_grid[:, -1, 0]
        V_A = dM_dx_grid[:, 0, 0]
        V_B = dM_dx_grid[:, -1, 0]

        My_A = ff_A_loc[:, 2]
        My_B = ff_B_loc[:, 2]
        Fz_A = ff_A_loc[:, 1]
        Fz_B = ff_B_loc[:, 1]

        # Normalised residuals
        res_M_A = (My_A + M_A) / self._M_ref
        res_M_B = (My_B - M_B) / self._M_ref
        res_V_A = (Fz_A + V_A) / self._V_ref
        res_V_B = (Fz_B - V_B) / self._V_ref

        L_end = (res_M_A.pow(2) + res_M_B.pow(2) +
                 res_V_A.pow(2) + res_V_B.pow(2)).mean()
        return L_end

    # ════════════════════════════════════════════════════════
    # SECTION E: TOTAL
    # ════════════════════════════════════════════════════════

    def forward(self, model, data):
        """Compute total physics loss (7 terms, normalised)."""

        # Compute scales on first call
        self._compute_scales(data)

        # Stage 1: GNN forward
        pred = model(data)
        disp, face_forces = self._extract_predictions(pred)

        # Concept 1: equilibrium
        L_eq   = self._loss_equilibrium(
            face_forces, data.F_ext, data.bc_disp, data.bc_rot)
        L_free = self._loss_free_face(face_forces, data.face_mask)
        L_sup  = self._loss_support(disp, data.bc_disp, data.bc_rot)

        # Transform to local
        disp_A, disp_B, ff_A, ff_B = \
            self._get_element_end_data(disp, face_forces, data)
        disp_A_loc, disp_B_loc, ff_A_loc, ff_B_loc = \
            self._transform_to_local(
                disp_A, disp_B, ff_A, ff_B, data.elem_directions)

        # Axial
        L_N = self._loss_axial(
            disp_A_loc, disp_B_loc, ff_A_loc, ff_B_loc, data)

        # Stage 2: field decoder
        field_data = model.evaluate_field(data, pred)
        derivs = self._compute_field_derivatives(field_data)

        # Bending losses
        L_Mk  = self._loss_moment_curvature(field_data, derivs)
        L_Mpp = self._loss_moment_pde(field_data, derivs)
        L_end = self._loss_end_forces(
            field_data, derivs, ff_A_loc, ff_B_loc, data)

        # Total
        total = (self.w_eq   * L_eq
               + self.w_free * L_free
               + self.w_sup  * L_sup
               + self.w_N    * L_N
               + self.w_Mk   * L_Mk
               + self.w_Mpp  * L_Mpp
               + self.w_end  * L_end)

        loss_dict = {
            'L_eq':    L_eq.item(),
            'L_free':  L_free.item(),
            'L_sup':   L_sup.item(),
            'L_N':     L_N.item(),
            'L_Mk':    L_Mk.item(),
            'L_Mpp':   L_Mpp.item(),
            'L_end':   L_end.item(),
            'total':   total.item(),
        }

        return total, loss_dict, pred


# ================================================================
# QUICK TEST
# ================================================================

if __name__ == "__main__":
    import os
    from pathlib import Path
    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    from model import PIGNN

    print("=" * 60)
    print("  PHYSICS LOSS TEST (Method C + Normalisation)")
    print("=" * 60)

    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    data = data_list[0]
    print(f"  Graph: {data.num_nodes} nodes, {data.n_elements} elements")

    model = PIGNN(
        node_in_dim=9, edge_in_dim=10,
        hidden_dim=64, n_layers=3,
        decoder_hidden=[64, 32],
    )
    loss_fn = MethodCPhysicsLoss()

    model.train()
    total_loss, loss_dict, pred = loss_fn(model, data)

    print(f"\n  Individual losses (should all be ~O(1)):")
    print(f"  {'─'*45}")
    for name, val in loss_dict.items():
        print(f"    {name:<8} {val:>12.4e}")
    print(f"  {'─'*45}")

    total_loss.backward()
    print(f"  Backward OK")

    print(f"\n{'='*60}")
    print(f"  TEST COMPLETE")
    print(f"{'='*60}")