# """
# =================================================================
# corotational.py — Corotational EB Beam Formulation (2D)
# =================================================================

# Input:  predicted nodal displacements (u_x, u_z, θ_y) from GNN
# Output: internal forces (N, M1, M2, V) per element
#         global nodal forces assembled for equilibrium check

# All operations are differentiable (PyTorch autograd compatible).

# Sign convention:
#   - Positive N: tension
#   - Positive M: sagging (standard structural convention)
#   - Positive V: consistent with dM/dx

# Element local frame:
#   - s-axis: from node A to node B
#   - Transverse: perpendicular, right-hand rule

# =================================================================
# """

# import torch
# import torch.nn as nn


# class CorotationalBeam2D(nn.Module):
#     """
#     Corotational Euler-Bernoulli beam element for 2D frames.

#     Given predicted displacements at nodes, computes:
#       1. Local deformations (u_l, θ1_l, θ2_l)
#       2. Internal forces (N, M1, M2, V) per element
#       3. Global nodal forces (assembled for equilibrium)

#     Everything is differentiable for backprop through physics.
#     """

#     def __init__(self):
#         super().__init__()

#     def forward(self, pred_disp, data):
#         """
#         Args:
#             pred_disp: (N, 3) predicted [u_x, u_z, θ_y] per node
#             data:      PyG Data object with connectivity, properties, etc.

#         Returns:
#             result dict with:
#                 - element forces: N_e, M1_e, M2_e, V_e  (E,)
#                 - global nodal forces: F_global_A, F_global_B  (E, 3)
#                 - assembled nodal forces: nodal_forces  (N, 3)
#                 - local deformations: u_l, theta1_l, theta2_l  (E,)
#         """
#         conn = data.connectivity          # (E, 2)
#         coords = data.coords              # (N, 3)
#         prop_E = data.prop_E              # (E,)
#         prop_A = data.prop_A              # (E,)
#         prop_I = data.prop_I22            # (E,)

#         E_count = conn.shape[0]
#         N_count = pred_disp.shape[0]
#         device = pred_disp.device

#         nA = conn[:, 0]   # (E,) node indices for end A
#         nB = conn[:, 1]   # (E,) node indices for end B

#         # ════════════════════════════════════════════════
#         # STEP 1: Original geometry
#         # ════════════════════════════════════════════════

#         x_A0 = coords[nA, 0]    # (E,)
#         z_A0 = coords[nA, 2]    # (E,)
#         x_B0 = coords[nB, 0]    # (E,)
#         z_B0 = coords[nB, 2]    # (E,)

#         dx0 = x_B0 - x_A0       # (E,)
#         dz0 = z_B0 - z_A0       # (E,)

#         l0 = torch.sqrt(dx0**2 + dz0**2)              # (E,)
#         beta0 = torch.atan2(dz0, dx0)                  # (E,)

#         # ════════════════════════════════════════════════
#         # STEP 2: Deformed geometry (from predictions)
#         # ════════════════════════════════════════════════

#         ux_A = pred_disp[nA, 0]    # (E,)
#         uz_A = pred_disp[nA, 1]    # (E,)
#         th_A = pred_disp[nA, 2]    # (E,)

#         ux_B = pred_disp[nB, 0]    # (E,)
#         uz_B = pred_disp[nB, 1]    # (E,)
#         th_B = pred_disp[nB, 2]    # (E,)

#         dx = dx0 + (ux_B - ux_A)   # (E,)
#         dz = dz0 + (uz_B - uz_A)   # (E,)

#         l = torch.sqrt(dx**2 + dz**2)                  # (E,)
#         beta = torch.atan2(dz, dx)                      # (E,)

#         # ════════════════════════════════════════════════
#         # STEP 3: Corotational local deformations
#         # ════════════════════════════════════════════════

#         # Axial deformation (well-conditioned formula)
#         u_l = (l**2 - l0**2) / (l + l0)                # (E,)

#         # Rigid body rotation
#         alpha = beta - beta0                             # (E,)

#         # Local rotations
#         theta1_l = th_A - alpha                          # (E,)
#         theta2_l = th_B - alpha                          # (E,)

#         # ════════════════════════════════════════════════
#         # STEP 4: Internal forces (constitutive law)
#         # ════════════════════════════════════════════════

#         EA = prop_E * prop_A       # (E,)
#         EI = prop_E * prop_I       # (E,)

#         # Axial force
#         N_e = (EA / l0) * u_l                            # (E,)

#         # Bending moments (EB stiffness matrix)
#         # [M1]   2EI  [2  1] [θ1_l]
#         # [M2] = ─── ·[1  2]·[θ2_l]
#         #         l0
#         coeff = 2.0 * EI / l0
#         M1_e = coeff * (2.0 * theta1_l + theta2_l)      # (E,)
#         M2_e = coeff * (theta1_l + 2.0 * theta2_l)      # (E,)

#         # Shear force (equilibrium of element)
#         V_e = (M1_e + M2_e) / l0                        # (E,)

#         # ════════════════════════════════════════════════
#         # STEP 5: Transform to global frame
#         # ════════════════════════════════════════════════

#         cos_b = dx / l     # (E,) using deformed angle
#         sin_b = dz / l     # (E,)

#         # Forces that element applies to node A (reaction)
#         F_xA = -N_e * cos_b + V_e * sin_b     # (E,)
#         F_zA = -N_e * sin_b - V_e * cos_b     # (E,)
#         M_yA = M1_e                             # (E,)

#         # Forces that element applies to node B
#         F_xB =  N_e * cos_b - V_e * sin_b     # (E,)
#         F_zB =  N_e * sin_b + V_e * cos_b     # (E,)
#         M_yB = M2_e                             # (E,)

#         # Stack: (E, 3) per end
#         F_global_A = torch.stack([F_xA, F_zA, M_yA], dim=1)   # (E, 3)
#         F_global_B = torch.stack([F_xB, F_zB, M_yB], dim=1)   # (E, 3)

#         # ════════════════════════════════════════════════
#         # STEP 6: Assemble to nodes (scatter-add)
#         # ════════════════════════════════════════════════

#         nodal_forces = torch.zeros(N_count, 3, device=device)

#         # Add element A-end forces to their respective nodes
#         nodal_forces.scatter_add_(
#             0,
#             nA.unsqueeze(1).expand(-1, 3),
#             F_global_A
#         )

#         # Add element B-end forces to their respective nodes
#         nodal_forces.scatter_add_(
#             0,
#             nB.unsqueeze(1).expand(-1, 3),
#             F_global_B
#         )

#         # ════════════════════════════════════════════════
#         # RESULT
#         # ════════════════════════════════════════════════

#         result = {
#             # Element internal forces
#             'N_e':      N_e,         # (E,) axial force
#             'M1_e':     M1_e,        # (E,) BM at end A
#             'M2_e':     M2_e,        # (E,) BM at end B
#             'V_e':      V_e,         # (E,) shear force

#             # Element end forces in global frame
#             'F_global_A': F_global_A,  # (E, 3)
#             'F_global_B': F_global_B,  # (E, 3)

#             # Assembled nodal forces
#             'nodal_forces': nodal_forces,  # (N, 3)

#             # Local deformations (for debugging)
#             'u_l':       u_l,         # (E,) axial deformation
#             'theta1_l':  theta1_l,    # (E,) local rotation at A
#             'theta2_l':  theta2_l,    # (E,) local rotation at B
#             'alpha':     alpha,       # (E,) rigid body rotation

#             # Geometry
#             'l0':        l0,          # (E,) original length
#             'l':         l,           # (E,) deformed length
#             'beta0':     beta0,       # (E,) original angle
#             'beta':      beta,        # (E,) deformed angle
#         }

#         return result


# # ================================================================
# # VERIFICATION
# # ================================================================

# def verify_corotational(data):
#     """
#     Test corotational module with Kratos ground truth.

#     Uses TRUE displacements from data.y_node, computes forces,
#     and compares with data.y_element.
#     """
#     print(f"\n{'═'*60}")
#     print(f"  COROTATIONAL VERIFICATION")
#     print(f"{'═'*60}")

#     beam = CorotationalBeam2D()

#     # Use ground truth displacements
#     true_disp = data.y_node.clone()   # (N, 3): [ux, uz, θy]

#     print(f"\n  Ground truth displacements:")
#     print(f"    ux:  [{true_disp[:, 0].min():.6e}, "
#           f"{true_disp[:, 0].max():.6e}]")
#     print(f"    uz:  [{true_disp[:, 1].min():.6e}, "
#           f"{true_disp[:, 1].max():.6e}]")
#     print(f"    θy:  [{true_disp[:, 2].min():.6e}, "
#           f"{true_disp[:, 2].max():.6e}]")

#     # Compute forces
#     result = beam(true_disp, data)

#     # ── Check internal forces vs Kratos ──
#     print(f"\n  Internal forces from corotational:")
#     print(f"    N:   [{result['N_e'].min():.4e}, "
#           f"{result['N_e'].max():.4e}]")
#     print(f"    M1:  [{result['M1_e'].min():.4e}, "
#           f"{result['M1_e'].max():.4e}]")
#     print(f"    M2:  [{result['M2_e'].min():.4e}, "
#           f"{result['M2_e'].max():.4e}]")
#     print(f"    V:   [{result['V_e'].min():.4e}, "
#           f"{result['V_e'].max():.4e}]")

#     if data.y_element is not None:
#         kratos_N = data.y_element[:, 0]
#         kratos_M = data.y_element[:, 1]
#         kratos_V = data.y_element[:, 2]

#         print(f"\n  Kratos reference:")
#         print(f"    N:   [{kratos_N.min():.4e}, {kratos_N.max():.4e}]")
#         print(f"    M:   [{kratos_M.min():.4e}, {kratos_M.max():.4e}]")
#         print(f"    V:   [{kratos_V.min():.4e}, {kratos_V.max():.4e}]")

#         # Compare
#         err_N = (result['N_e'] - kratos_N).abs()
#         err_M = (result['M1_e'] - kratos_M).abs()
#         err_V = (result['V_e'] - kratos_V).abs()

#         ref_N = kratos_N.abs().max().clamp(min=1e-10)
#         ref_M = kratos_M.abs().max().clamp(min=1e-10)
#         ref_V = kratos_V.abs().max().clamp(min=1e-10)

#         print(f"\n  Relative errors:")
#         print(f"    N:  max={err_N.max()/ref_N:.6e}, "
#               f"mean={err_N.mean()/ref_N:.6e}")
#         print(f"    M:  max={err_M.max()/ref_M:.6e}, "
#               f"mean={err_M.mean()/ref_M:.6e}")
#         print(f"    V:  max={err_V.max()/ref_V:.6e}, "
#               f"mean={err_V.mean()/ref_V:.6e}")

#     # ── Check equilibrium ──
#     print(f"\n  Equilibrium check:")
#     residual = result['nodal_forces'] - data.F_ext

#     # At free nodes
#     free_mask = data.bc_disp.squeeze(-1) < 0.5
#     if free_mask.any():
#         res_free = residual[free_mask]
#         print(f"    Free node residual:")
#         print(f"      Rx:  [{res_free[:, 0].min():.6e}, "
#               f"{res_free[:, 0].max():.6e}]")
#         print(f"      Rz:  [{res_free[:, 1].min():.6e}, "
#               f"{res_free[:, 1].max():.6e}]")
#         print(f"      RM:  [{res_free[:, 2].min():.6e}, "
#               f"{res_free[:, 2].max():.6e}]")
#         print(f"      Max |R|: {res_free.abs().max():.6e}")

#     # At support nodes
#     sup_mask = data.bc_disp.squeeze(-1) > 0.5
#     if sup_mask.any():
#         reactions = result['nodal_forces'][sup_mask]
#         print(f"    Support reactions:")
#         print(f"      Rx:  {reactions[:, 0].tolist()}")
#         print(f"      Rz:  {reactions[:, 1].tolist()}")
#         print(f"      RM:  {reactions[:, 2].tolist()}")

#     # ── Local deformations ──
#     print(f"\n  Local deformations:")
#     print(f"    u_l:     [{result['u_l'].min():.6e}, "
#           f"{result['u_l'].max():.6e}]")
#     print(f"    θ1_l:    [{result['theta1_l'].min():.6e}, "
#           f"{result['theta1_l'].max():.6e}]")
#     print(f"    θ2_l:    [{result['theta2_l'].min():.6e}, "
#           f"{result['theta2_l'].max():.6e}]")
#     print(f"    α:       [{result['alpha'].min():.6e}, "
#           f"{result['alpha'].max():.6e}]")

#     print(f"\n{'═'*60}")
#     print(f"  VERIFICATION COMPLETE")
#     print(f"{'═'*60}\n")

#     return result


# # ================================================================
# # MAIN — Run verification
# # ================================================================

# if __name__ == "__main__":
#     import os
#     from pathlib import Path
#     CURRENT_SUBFOLDER = Path(__file__).resolve().parent
#     os.chdir(CURRENT_SUBFOLDER)

#     print("=" * 60)
#     print("  COROTATIONAL BEAM MODULE TEST")
#     print("=" * 60)

#     # Load graph data
#     data_list = torch.load("DATA/graph_dataset.pt",
#                            weights_only=False)
#     data = data_list[0]
#     print(f"  Loaded: {data.num_nodes} nodes, "
#           f"{data.n_elements} elements")

#     # Run verification
#     result = verify_corotational(data)

#     # Test gradient flow
#     print(f"\n  Gradient flow test:")
#     true_disp = data.y_node.clone().requires_grad_(True)
#     beam = CorotationalBeam2D()
#     result = beam(true_disp, data)

#     # Can we differentiate M w.r.t. displacements?
#     M_sum = result['M1_e'].sum()
#     grad = torch.autograd.grad(M_sum, true_disp,
#                                 create_graph=True)[0]
#     print(f"    dM/d(disp) shape: {grad.shape}")
#     print(f"    dM/d(disp) range: [{grad.min():.6e}, "
#           f"{grad.max():.6e}]")
#     print(f"    ✓ Gradients flow through corotational formulation")

#     # Can we differentiate equilibrium residual?
#     residual = result['nodal_forces'] - data.F_ext
#     R_sum = residual.pow(2).sum()
#     grad2 = torch.autograd.grad(R_sum, true_disp)[0]
#     print(f"    dR²/d(disp) shape: {grad2.shape}")
#     print(f"    dR²/d(disp) range: [{grad2.min():.6e}, "
#           f"{grad2.max():.6e}]")
#     print(f"    ✓ Equilibrium loss is differentiable")

#     print(f"\n  ALL TESTS PASSED ✓")

#-----------------------------------------------------------------------------------------
# """
# =================================================================
# corotational.py — Linear EB Beam Element for 2D Frames
# =================================================================

# Uses ORIGINAL geometry (linear, small displacement).
# Compatible with Kratos linear static analysis.

# Despite the filename, this is the LINEAR formulation
# (kept for import compatibility).
# =================================================================
# """

# import torch
# import torch.nn as nn


# class CorotationalBeam2D(nn.Module):
#     """
#     Linear 2D Euler-Bernoulli beam element.

#     Uses ORIGINAL geometry only.
#     Correct for small displacement / linear analysis.
#     """

#     def __init__(self):
#         super().__init__()

#     def forward(self, pred_disp, data):
#         """
#         Args:
#             pred_disp: (N, 3) predicted [u_x, u_z, θ_y]
#             data:      PyG Data object

#         Returns:
#             result dict with internal forces and
#             assembled nodal forces
#         """
#         conn   = data.connectivity       # (E, 2)
#         coords = data.coords             # (N, 3)
#         prop_E = data.prop_E             # (E,)
#         prop_A = data.prop_A             # (E,)
#         prop_I = data.prop_I22           # (E,)

#         E_count = conn.shape[0]
#         N_count = pred_disp.shape[0]
#         device  = pred_disp.device

#         nA = conn[:, 0]
#         nB = conn[:, 1]

#         # ════════════════════════════════════════════
#         # STEP 1: Original geometry (no grad needed)
#         # ════════════════════════════════════════════

#         x_A = coords[nA, 0]
#         z_A = coords[nA, 2]
#         x_B = coords[nB, 0]
#         z_B = coords[nB, 2]

#         dx0 = x_B - x_A
#         dz0 = z_B - z_A

#         l0 = torch.sqrt(dx0**2 + dz0**2)

#         cos0 = dx0 / l0
#         sin0 = dz0 / l0

#         # ════════════════════════════════════════════
#         # STEP 2: Gather global displacements
#         # ════════════════════════════════════════════

#         ux_A = pred_disp[nA, 0]
#         uz_A = pred_disp[nA, 1]
#         th_A = pred_disp[nA, 2]

#         ux_B = pred_disp[nB, 0]
#         uz_B = pred_disp[nB, 1]
#         th_B = pred_disp[nB, 2]

#         # ════════════════════════════════════════════
#         # STEP 3: Transform to local frame (LINEAR)
#         # ════════════════════════════════════════════

#         # Relative displacements
#         dux = ux_B - ux_A
#         duz = uz_B - uz_A

#         # Local axial displacement
#         #   u_local = Δu · cos₀ + Δw · sin₀
#         u_l = dux * cos0 + duz * sin0

#         # Local transverse displacement at B
#         #   w_local = -Δu · sin₀ + Δw · cos₀
#         w_l = -dux * sin0 + duz * cos0

#         # Rigid body rotation of chord
#         phi = w_l / l0

#         # Local rotations (subtract rigid body rotation)
#         theta1_l = th_A - phi
#         theta2_l = th_B - phi

#         # ════════════════════════════════════════════
#         # STEP 4: Internal forces (constitutive)
#         # ════════════════════════════════════════════

#         EA = prop_E * prop_A
#         EI = prop_E * prop_I

#         # Axial force
#         N_e = (EA / l0) * u_l

#         # Bending moments (standard EB stiffness)
#         coeff = 2.0 * EI / l0
#         M1_e = coeff * (2.0 * theta1_l + theta2_l)
#         M2_e = coeff * (theta1_l + 2.0 * theta2_l)

#         # Shear force
#         V_e = (M1_e + M2_e) / l0

#         # ════════════════════════════════════════════
#         # STEP 5: Local → Global forces
#         # ════════════════════════════════════════════

#         # Local force vector at node A:
#         #   f_local_A = [-N, -V, M1]
#         # Transform to global:
#         #   Fx = f_axial·cos₀ - f_shear·sin₀
#         #   Fz = f_axial·sin₀ + f_shear·cos₀

#         # Node A (element pushes back on node A)
#         F_xA = (-N_e) * cos0 - (-V_e) * sin0
#         F_zA = (-N_e) * sin0 + (-V_e) * cos0
#         M_yA = M1_e

#         # Node B
#         F_xB = (N_e) * cos0 - (V_e) * sin0
#         F_zB = (N_e) * sin0 + (V_e) * cos0
#         M_yB = M2_e

#         F_global_A = torch.stack(
#             [F_xA, F_zA, M_yA], dim=1
#         )
#         F_global_B = torch.stack(
#             [F_xB, F_zB, M_yB], dim=1
#         )

#         # ════════════════════════════════════════════
#         # STEP 6: Assemble to nodes
#         # ════════════════════════════════════════════

#         nodal_forces = torch.zeros(
#             N_count, 3, device=device
#         )

#         nodal_forces.scatter_add_(
#             0,
#             nA.unsqueeze(1).expand(-1, 3),
#             F_global_A
#         )
#         nodal_forces.scatter_add_(
#             0,
#             nB.unsqueeze(1).expand(-1, 3),
#             F_global_B
#         )

#         # ════════════════════════════════════════════
#         # RESULT
#         # ════════════════════════════════════════════

#         result = {
#             'N_e':       N_e,
#             'M1_e':      M1_e,
#             'M2_e':      M2_e,
#             'V_e':       V_e,
#             'F_global_A': F_global_A,
#             'F_global_B': F_global_B,
#             'nodal_forces': nodal_forces,
#             'u_l':        u_l,
#             'theta1_l':   theta1_l,
#             'theta2_l':   theta2_l,
#             'phi':        phi,
#             'l0':         l0,
#             'cos0':       cos0,
#             'sin0':       sin0,
#         }

#         return result


# # ================================================================
# # VERIFICATION
# # ================================================================

# def verify_beam_element(data):
#     """Verify with Kratos ground truth."""
#     print(f"\n{'═'*60}")
#     print(f"  BEAM ELEMENT VERIFICATION")
#     print(f"{'═'*60}")

#     beam = CorotationalBeam2D()
#     true_disp = data.y_node.clone()

#     print(f"\n  Ground truth displacements:")
#     print(f"    ux:  [{true_disp[:, 0].min():.6e}, "
#           f"{true_disp[:, 0].max():.6e}]")
#     print(f"    uz:  [{true_disp[:, 1].min():.6e}, "
#           f"{true_disp[:, 1].max():.6e}]")
#     print(f"    θy:  [{true_disp[:, 2].min():.6e}, "
#           f"{true_disp[:, 2].max():.6e}]")

#     result = beam(true_disp, data)

#     print(f"\n  Computed internal forces:")
#     print(f"    N:   [{result['N_e'].min():.4f}, "
#           f"{result['N_e'].max():.4f}]")
#     print(f"    M1:  [{result['M1_e'].min():.4f}, "
#           f"{result['M1_e'].max():.4f}]")
#     print(f"    M2:  [{result['M2_e'].min():.4f}, "
#           f"{result['M2_e'].max():.4f}]")
#     print(f"    V:   [{result['V_e'].min():.4f}, "
#           f"{result['V_e'].max():.4f}]")

#     if data.y_element is not None:
#         kratos_N = data.y_element[:, 0]
#         kratos_M = data.y_element[:, 1]
#         kratos_V = data.y_element[:, 2]

#         print(f"\n  Kratos reference:")
#         print(f"    N:   [{kratos_N.min():.4f}, "
#               f"{kratos_N.max():.4f}]")
#         print(f"    M:   [{kratos_M.min():.4f}, "
#               f"{kratos_M.max():.4f}]")
#         print(f"    V:   [{kratos_V.min():.4f}, "
#               f"{kratos_V.max():.4f}]")

#         ref_N = kratos_N.abs().max().clamp(min=1e-10)
#         ref_M = kratos_M.abs().max().clamp(min=1e-10)
#         ref_V = kratos_V.abs().max().clamp(min=1e-10)

#         err_N = (result['N_e'] - kratos_N).abs()
#         err_M = (result['M1_e'] - kratos_M).abs()
#         err_V = (result['V_e'] - kratos_V).abs()

#         print(f"\n  Relative errors:")
#         print(f"    N:  max={err_N.max()/ref_N:.6e}, "
#               f"mean={err_N.mean()/ref_N:.6e}")
#         print(f"    M:  max={err_M.max()/ref_M:.6e}, "
#               f"mean={err_M.mean()/ref_M:.6e}")
#         print(f"    V:  max={err_V.max()/ref_V:.6e}, "
#               f"mean={err_V.mean()/ref_V:.6e}")

#         # Per-element
#         print(f"\n  Per-element (first 10):")
#         print(f"  {'El':>3} | {'N_comp':>10} {'N_krat':>10} | "
#               f"{'M_comp':>10} {'M_krat':>10} | "
#               f"{'V_comp':>10} {'V_krat':>10}")
#         print(f"  {'-'*75}")
#         for e in range(min(10, len(kratos_N))):
#             print(
#                 f"  {e:3d} | "
#                 f"{result['N_e'][e].item():10.4f} "
#                 f"{kratos_N[e].item():10.4f} | "
#                 f"{result['M1_e'][e].item():10.4f} "
#                 f"{kratos_M[e].item():10.4f} | "
#                 f"{result['V_e'][e].item():10.4f} "
#                 f"{kratos_V[e].item():10.4f}"
#             )

#     # Equilibrium
#     residual = result['nodal_forces'] - data.F_ext
#     free = data.bc_disp.squeeze(-1) < 0.5
#     if free.any():
#         res_free = residual[free]
#         print(f"\n  Equilibrium at free nodes:")
#         print(f"    Rx:  [{res_free[:, 0].min():.6e}, "
#               f"{res_free[:, 0].max():.6e}]")
#         print(f"    Rz:  [{res_free[:, 1].min():.6e}, "
#               f"{res_free[:, 1].max():.6e}]")
#         print(f"    RM:  [{res_free[:, 2].min():.6e}, "
#               f"{res_free[:, 2].max():.6e}]")
#         print(f"    Max |R|: {res_free.abs().max():.6e}")

#     # Local deformations
#     print(f"\n  Local deformations:")
#     print(f"    u_l/l0: [{(result['u_l']/result['l0']).min():.6e},"
#           f" {(result['u_l']/result['l0']).max():.6e}]")
#     print(f"    θ1_l:   [{result['theta1_l'].min():.6e}, "
#           f"{result['theta1_l'].max():.6e}]")
#     print(f"    θ2_l:   [{result['theta2_l'].min():.6e}, "
#           f"{result['theta2_l'].max():.6e}]")
#     print(f"    φ:      [{result['phi'].min():.6e}, "
#           f"{result['phi'].max():.6e}]")

#     print(f"\n{'═'*60}\n")
#     return result


# if __name__ == "__main__":
#     import os
#     from pathlib import Path
#     CURRENT_SUBFOLDER = Path(__file__).resolve().parent
#     os.chdir(CURRENT_SUBFOLDER)

#     print("=" * 60)
#     print("  BEAM ELEMENT VERIFICATION")
#     print("=" * 60)

#     data_list = torch.load("DATA/graph_dataset.pt",
#                            weights_only=False)

#     # Test multiple cases
#     for i in range(min(3, len(data_list))):
#         data = data_list[i]
#         print(f"\n  Case {i}: {data.num_nodes} nodes, "
#               f"{data.n_elements} elements")
#         result = verify_beam_element(data)

#     # Gradient test
#     print(f"\n  Gradient flow test:")
#     data = data_list[0]
#     true_disp = data.y_node.clone().requires_grad_(True)
#     beam = CorotationalBeam2D()
#     result = beam(true_disp, data)

#     M_sum = result['M1_e'].sum()
#     grad = torch.autograd.grad(
#         M_sum, true_disp, create_graph=True
#     )[0]
#     print(f"    dM/d(disp) shape: {grad.shape}")
#     print(f"    dM/d(disp) range: [{grad.min():.6e}, "
#           f"{grad.max():.6e}]")
#     print(f"    ✓ Differentiable")

#     R = result['nodal_forces'] - data.F_ext
#     R_sum = R.pow(2).sum()
#     grad2 = torch.autograd.grad(R_sum, true_disp)[0]
#     print(f"    dR²/d(disp) range: [{grad2.min():.6e}, "
#           f"{grad2.max():.6e}]")
#     print(f"    ✓ Equilibrium loss differentiable")

#     print(f"\n  DONE ✓")

#----------------------------------------------------------------------------------------------

"""
=================================================================
corotational.py — 2D EB Beam via Full Stiffness Matrix
=================================================================

Uses the standard 6-DOF local stiffness matrix:

  k_local · d_local = f_local

Then transforms to global:

  K_global = T^T · k_local · T
  f_global = K_global · d_global

No sign convention guessing — the stiffness matrix 
encodes everything correctly.
=================================================================
"""

import torch
import torch.nn as nn


class CorotationalBeam2D(nn.Module):
    """
    2D Euler-Bernoulli beam element using full stiffness matrix.
    
    DOF ordering per element:
      d = [ux_A, uz_A, θ_A, ux_B, uz_B, θ_B]
    
    Local stiffness (6×6):
      k = [[EA/L,    0,         0,       -EA/L,   0,         0      ],
           [0,       12EI/L³,   6EI/L²,   0,     -12EI/L³,   6EI/L² ],
           [0,       6EI/L²,    4EI/L,    0,      -6EI/L²,   2EI/L  ],
           [-EA/L,   0,         0,        EA/L,    0,         0      ],
           [0,      -12EI/L³,  -6EI/L²,   0,      12EI/L³,  -6EI/L² ],
           [0,       6EI/L²,    2EI/L,    0,      -6EI/L²,   4EI/L  ]]

    Transformation matrix T (6×6):
      T = [[c,  s,  0,  0,  0,  0],
           [-s, c,  0,  0,  0,  0],
           [0,  0,  1,  0,  0,  0],
           [0,  0,  0,  c,  s,  0],
           [0,  0,  0, -s,  c,  0],
           [0,  0,  0,  0,  0,  1]]

    Global stiffness:
      K_global = T^T · k_local · T
    
    Element forces:
      f_global = K_global · d_global
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_disp, data):
        conn   = data.connectivity       # (E, 2)
        coords = data.coords             # (N, 3)
        prop_E = data.prop_E             # (E,)
        prop_A = data.prop_A             # (E,)
        prop_I = data.prop_I22           # (E,)

        E_count = conn.shape[0]
        N_count = pred_disp.shape[0]
        device  = pred_disp.device

        nA = conn[:, 0]
        nB = conn[:, 1]

        # ════════════════════════════════════════════
        # STEP 1: Geometry
        # ════════════════════════════════════════════

        dx0 = coords[nB, 0] - coords[nA, 0]
        dz0 = coords[nB, 2] - coords[nA, 2]
        l0  = torch.sqrt(dx0**2 + dz0**2)
        c   = dx0 / l0   # cos
        s   = dz0 / l0   # sin

        EA = prop_E * prop_A
        EI = prop_E * prop_I

        # ════════════════════════════════════════════
        # STEP 2: Gather global DOFs
        # ════════════════════════════════════════════

        # d_global = [ux_A, uz_A, θ_A, ux_B, uz_B, θ_B]
        d_global = torch.stack([
            pred_disp[nA, 0],   # ux_A
            pred_disp[nA, 1],   # uz_A
            pred_disp[nA, 2],   # θ_A
            pred_disp[nB, 0],   # ux_B
            pred_disp[nB, 1],   # uz_B
            pred_disp[nB, 2],   # θ_B
        ], dim=1)   # (E, 6)

        # ════════════════════════════════════════════
        # STEP 3: Transform to local: d_local = T · d_global
        # ════════════════════════════════════════════

        d_local = torch.zeros_like(d_global)

        # Node A
        d_local[:, 0] =  c * d_global[:, 0] + s * d_global[:, 1]   # u_axial_A
        d_local[:, 1] = -s * d_global[:, 0] + c * d_global[:, 1]   # w_trans_A
        d_local[:, 2] =  d_global[:, 2]                             # θ_A

        # Node B
        d_local[:, 3] =  c * d_global[:, 3] + s * d_global[:, 4]   # u_axial_B
        d_local[:, 4] = -s * d_global[:, 3] + c * d_global[:, 4]   # w_trans_B
        d_local[:, 5] =  d_global[:, 5]                             # θ_B

        # ════════════════════════════════════════════
        # STEP 4: Local stiffness × local displacements
        # ════════════════════════════════════════════

        L  = l0
        L2 = l0 * l0
        L3 = L2 * l0

        # f_local = k_local · d_local
        # Compute each component directly (vectorized)

        ua = d_local[:, 0]   # u_axial_A
        wa = d_local[:, 1]   # w_trans_A
        ta = d_local[:, 2]   # θ_A
        ub = d_local[:, 3]   # u_axial_B
        wb = d_local[:, 4]   # w_trans_B
        tb = d_local[:, 5]   # θ_B

        # f_local[0] = EA/L * (ua - ub)
        f0 = EA / L * (ua - ub)

        # f_local[1] = 12EI/L³ * (wa - wb) + 6EI/L² * (ta + tb)
        f1 = 12 * EI / L3 * (wa - wb) + 6 * EI / L2 * (ta + tb)

        # f_local[2] = 6EI/L² * (wa - wb) + EI/L * (4*ta + 2*tb)
        f2 = 6 * EI / L2 * (wa - wb) + EI / L * (4 * ta + 2 * tb)

        # f_local[3] = EA/L * (ub - ua)
        f3 = EA / L * (ub - ua)

        # f_local[4] = 12EI/L³ * (wb - wa) + 6EI/L² * (-ta - tb)
        f4 = 12 * EI / L3 * (wb - wa) - 6 * EI / L2 * (ta + tb)

        # f_local[5] = 6EI/L² * (wa - wb) + EI/L * (2*ta + 4*tb)
        f5 = 6 * EI / L2 * (wa - wb) + EI / L * (2 * ta + 4 * tb)

        f_local = torch.stack(
            [f0, f1, f2, f3, f4, f5], dim=1
        )   # (E, 6)

        # ════════════════════════════════════════════
        # STEP 5: Transform back: f_global = T^T · f_local
        # ════════════════════════════════════════════

        f_global = torch.zeros_like(f_local)

        # Node A: T^T for first 3 rows
        f_global[:, 0] = c * f_local[:, 0] - s * f_local[:, 1]
        f_global[:, 1] = s * f_local[:, 0] + c * f_local[:, 1]
        f_global[:, 2] = f_local[:, 2]

        # Node B: T^T for last 3 rows
        f_global[:, 3] = c * f_local[:, 3] - s * f_local[:, 4]
        f_global[:, 4] = s * f_local[:, 3] + c * f_local[:, 4]
        f_global[:, 5] = f_local[:, 5]

        # ════════════════════════════════════════════
        # STEP 6: Extract + Assemble
        # ════════════════════════════════════════════

        F_global_A = f_global[:, 0:3]   # (E, 3)
        F_global_B = f_global[:, 3:6]   # (E, 3)

        nodal_forces = torch.zeros(
            N_count, 3, device=device
        )
        nodal_forces.scatter_add_(
            0,
            nA.unsqueeze(1).expand(-1, 3),
            F_global_A
        )
        nodal_forces.scatter_add_(
            0,
            nB.unsqueeze(1).expand(-1, 3),
            F_global_B
        )

        # ════════════════════════════════════════════
        # Extract internal forces for monitoring
        # ════════════════════════════════════════════

        # Axial force (tension positive)
        N_e = f3   # = EA/L * (ub - ua) = tension in element

        # Shear force
        V_e = f4   # shear at B end (or -f1 = shear at A end)

        # Bending moments
        M1_e = f2   # moment at A
        M2_e = f5   # moment at B

        # Local deformations for debugging
        u_l      = ub - ua
        theta1_l = ta
        theta2_l = tb
        phi      = (wb - wa) / l0

        result = {
            'N_e':       N_e,
            'M1_e':      M1_e,
            'M2_e':      M2_e,
            'V_e':       V_e,
            'F_global_A': F_global_A,
            'F_global_B': F_global_B,
            'nodal_forces': nodal_forces,
            'f_local':    f_local,
            'f_global':   f_global,
            'd_local':    d_local,
            'u_l':        u_l,
            'theta1_l':   theta1_l,
            'theta2_l':   theta2_l,
            'phi':        phi,
            'l0':         l0,
            'cos0':       c,
            'sin0':       s,
        }

        return result


# ════════════════════════════════════════════════════════
# VERIFICATION
# ════════════════════════════════════════════════════════

def verify_beam_element(data):
    print(f"\n{'═'*70}")
    print(f"  BEAM ELEMENT VERIFICATION (Full Stiffness)")
    print(f"{'═'*70}")

    beam = CorotationalBeam2D()
    true_disp = data.y_node.clone()

    print(f"\n  Ground truth:")
    print(f"    ux:  [{true_disp[:, 0].min():.6e}, "
          f"{true_disp[:, 0].max():.6e}]")
    print(f"    uz:  [{true_disp[:, 1].min():.6e}, "
          f"{true_disp[:, 1].max():.6e}]")
    print(f"    θy:  [{true_disp[:, 2].min():.6e}, "
          f"{true_disp[:, 2].max():.6e}]")

    result = beam(true_disp, data)

    # ── Internal forces ──
    print(f"\n  Computed:")
    print(f"    N:   [{result['N_e'].min():.4f}, "
          f"{result['N_e'].max():.4f}]")
    print(f"    M1:  [{result['M1_e'].min():.4f}, "
          f"{result['M1_e'].max():.4f}]")
    print(f"    M2:  [{result['M2_e'].min():.4f}, "
          f"{result['M2_e'].max():.4f}]")
    print(f"    V:   [{result['V_e'].min():.4f}, "
          f"{result['V_e'].max():.4f}]")

    # ── Kratos comparison ──
    if data.y_element is not None:
        kratos_N = data.y_element[:, 0]
        kratos_M = data.y_element[:, 1]
        kratos_V = data.y_element[:, 2]

        print(f"\n  Kratos:")
        print(f"    N:   [{kratos_N.min():.4f}, "
              f"{kratos_N.max():.4f}]")
        print(f"    M:   [{kratos_M.min():.4f}, "
              f"{kratos_M.max():.4f}]")
        print(f"    V:   [{kratos_V.min():.4f}, "
              f"{kratos_V.max():.4f}]")

        ref_N = kratos_N.abs().max().clamp(min=1e-10)
        ref_M = kratos_M.abs().max().clamp(min=1e-10)
        ref_V = kratos_V.abs().max().clamp(min=1e-10)

        err_N = (result['N_e'] - kratos_N).abs()
        err_M = (result['M1_e'] - kratos_M).abs()
        err_V = (result['V_e'] - kratos_V).abs()

        print(f"\n  Rel errors:")
        print(f"    N:  max={err_N.max()/ref_N:.6e}")
        print(f"    M1: max={err_M.max()/ref_M:.6e}")
        print(f"    V:  max={err_V.max()/ref_V:.6e}")

        print(f"\n  Element comparison (first 15):")
        print(f"  {'El':>3} | {'N_comp':>10} {'N_krat':>10} | "
              f"{'M1_comp':>10} {'M_krat':>10} | "
              f"{'V_comp':>10} {'V_krat':>10}")
        print(f"  {'-'*80}")
        for e in range(min(15, len(kratos_N))):
            nA_id = data.connectivity[e, 0].item()
            nB_id = data.connectivity[e, 1].item()
            dx = abs(data.coords[nB_id, 0] - data.coords[nA_id, 0])
            dz = abs(data.coords[nB_id, 2] - data.coords[nA_id, 2])
            etype = "B" if dx > dz else "C"
            print(
                f"  {e:3d}{etype}| "
                f"{result['N_e'][e].item():10.4f} "
                f"{kratos_N[e].item():10.4f} | "
                f"{result['M1_e'][e].item():10.4f} "
                f"{kratos_M[e].item():10.4f} | "
                f"{result['V_e'][e].item():10.4f} "
                f"{kratos_V[e].item():10.4f}"
            )

    # ── Equilibrium ──
    residual = result['nodal_forces'] - data.F_ext
    free = data.bc_disp.squeeze(-1) < 0.5
    if free.any():
        res_free = residual[free]
        print(f"\n  Equilibrium (free nodes):")
        print(f"    Rx: [{res_free[:, 0].min():.6e}, "
              f"{res_free[:, 0].max():.6e}]")
        print(f"    Rz: [{res_free[:, 1].min():.6e}, "
              f"{res_free[:, 1].max():.6e}]")
        print(f"    RM: [{res_free[:, 2].min():.6e}, "
              f"{res_free[:, 2].max():.6e}]")
        print(f"    Max|R|: {res_free.abs().max():.6e}")

        F_max = data.F_ext.abs().max().item()
        rel_max = res_free.abs().max().item() / max(F_max, 1e-10)
        print(f"    |R|/F_max: {rel_max:.6e}")
        if rel_max < 1e-3:
            print(f"    ✓ EQUILIBRIUM VERIFIED")
        else:
            print(f"    ✗ EQUILIBRIUM NOT SATISFIED")

    print(f"\n{'═'*70}\n")
    return result


if __name__ == "__main__":
    import os
    from pathlib import Path
    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 60)
    print("  BEAM ELEMENT VERIFICATION")
    print("=" * 60)

    data_list = torch.load("DATA/graph_dataset.pt",
                           weights_only=False)

    for i in range(min(3, len(data_list))):
        data = data_list[i]
        print(f"\n  Case {i}: {data.num_nodes} nodes, "
              f"{data.n_elements} elements")
        result = verify_beam_element(data)

    # Gradient test
    print(f"  Gradient flow test:")
    data = data_list[0]
    true_disp = data.y_node.clone().requires_grad_(True)
    beam = CorotationalBeam2D()
    result = beam(true_disp, data)

    R = result['nodal_forces'] - data.F_ext
    R_sum = R.pow(2).sum()
    grad = torch.autograd.grad(R_sum, true_disp)[0]
    print(f"    dR²/d(disp): [{grad.min():.6e}, "
          f"{grad.max():.6e}]")
    print(f"    ✓ Differentiable")
    print(f"\n  DONE ✓")