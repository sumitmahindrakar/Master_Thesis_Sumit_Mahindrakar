# """
# =================================================================
# physics_loss.py — Minimum Potential Energy + Curriculum EA
# =================================================================

# Curriculum approach:
#   Start with reduced EA (EA/1000) so bending develops first.
#   Gradually ramp EA to full value over training.
#   Purely physics-based, no data needed.

# This solves the axial locking problem that traps the network
# at the zero solution when EA >> EI.
# =================================================================
# """

# import torch
# import torch.nn as nn


# class EnergyLoss(nn.Module):

#     def __init__(self, w_energy=1.0, w_bc=1.0,
#                  ea_start_factor=1e-3,
#                  ea_ramp_start=0,
#                  ea_ramp_end=1500,
#                  total_epochs=3000):
#         super().__init__()
#         self.w_energy = w_energy
#         self.w_bc = w_bc

#         # Curriculum parameters
#         self.ea_start_factor = ea_start_factor     # start at EA/1000
#         self.ea_ramp_start = ea_ramp_start         # begin ramping at this epoch
#         self.ea_ramp_end = ea_ramp_end             # reach full EA at this epoch
#         self.total_epochs = total_epochs

#         # Track current epoch (set externally)
#         self.current_epoch = 0

#     def _get_ea_factor(self):
#         """
#         Compute EA scaling factor for current epoch.

#         Returns factor ∈ [ea_start_factor, 1.0]

#         Schedule:
#           epoch < ramp_start:  factor = ea_start_factor
#           ramp_start ≤ epoch ≤ ramp_end: linear ramp
#           epoch > ramp_end:    factor = 1.0
#         """
#         epoch = self.current_epoch

#         if epoch <= self.ea_ramp_start:
#             return self.ea_start_factor

#         if epoch >= self.ea_ramp_end:
#             return 1.0

#         # Linear ramp in log space (smoother for large ratios)
#         import math
#         t = (epoch - self.ea_ramp_start) / (self.ea_ramp_end - self.ea_ramp_start)
#         log_start = math.log10(self.ea_start_factor)
#         log_end = 0.0    # log10(1.0) = 0
#         log_factor = log_start + t * (log_end - log_start)
#         return 10.0 ** log_factor

#     # ════════════════════════════════════════════════════════
#     # A. GLOBAL → LOCAL (UNCHANGED)
#     # ════════════════════════════════════════════════════════

#     def _to_local(self, disp, elem_directions, conn):
#         cos_a = elem_directions[:, 0:1]
#         sin_a = elem_directions[:, 2:3]

#         d_A = disp[conn[:, 0]]
#         d_B = disp[conn[:, 1]]

#         def rotate(d):
#             ux = d[:, 0:1]
#             uz = d[:, 1:2]
#             th = d[:, 2:3]
#             return torch.cat([
#                  ux * cos_a + uz * sin_a,
#                 -ux * sin_a + uz * cos_a,
#                  th,
#             ], dim=1)

#         return rotate(d_A), rotate(d_B)

#     # ════════════════════════════════════════════════════════
#     # B. STRAIN ENERGY (with EA factor)
#     # ════════════════════════════════════════════════════════

#     def _compute_strain_energy(self, disp_A_loc, disp_B_loc,
#                                 EA, EI, L, ea_factor):
#         """
#         Strain energy with curriculum-scaled EA.

#         U_axial uses EA × ea_factor (ramped from 0.001 to 1.0)
#         U_bend uses EI unchanged (always full stiffness)
#         """
#         u_sA = disp_A_loc[:, 0]
#         w_A  = disp_A_loc[:, 1]
#         th_A = disp_A_loc[:, 2]
#         u_sB = disp_B_loc[:, 0]
#         w_B  = disp_B_loc[:, 1]
#         th_B = disp_B_loc[:, 2]

#         # ── Axial strain energy (with curriculum factor) ──
#         EA_eff = EA * ea_factor                                # ◀◀◀ CURRICULUM
#         du = u_sB - u_sA
#         U_axial = 0.5 * EA_eff / L * du.pow(2)

#         # ── Bending strain energy (full EI always) ──
#         c = EI / L.pow(3)
#         L2 = L.pow(2)

#         U_bend = 0.5 * c * (
#               12.0 * w_A.pow(2)
#             + 12.0 * L * w_A * th_A
#             - 24.0 * w_A * w_B
#             + 12.0 * L * w_A * th_B
#             + 4.0 * L2 * th_A.pow(2)
#             - 12.0 * L * th_A * w_B
#             + 4.0 * L2 * th_A * th_B
#             + 12.0 * w_B.pow(2)
#             - 12.0 * L * w_B * th_B
#             + 4.0 * L2 * th_B.pow(2)
#         )

#         return U_axial, U_bend

#     # ════════════════════════════════════════════════════════
#     # C. EXTERNAL WORK (UNCHANGED)
#     # ════════════════════════════════════════════════════════

#     def _compute_external_work(self, pred, F_ext):
#         W = (pred * F_ext).sum()
#         return W

#     # ════════════════════════════════════════════════════════
#     # D. BC LOSS (UNCHANGED)
#     # ════════════════════════════════════════════════════════

#     def _loss_bc(self, pred, data):
#         bc_disp = data.bc_disp.squeeze(-1)
#         bc_rot  = data.bc_rot.squeeze(-1)
#         u_c     = data.u_c
#         theta_c = data.theta_c

#         sup_d = (bc_disp > 0.5)
#         sup_r = (bc_rot > 0.5)

#         loss = torch.tensor(0.0, device=pred.device)
#         if sup_d.any():
#             loss = loss + (pred[sup_d, 0] / u_c).pow(2).mean()
#             loss = loss + (pred[sup_d, 1] / u_c).pow(2).mean()
#         if sup_r.any():
#             loss = loss + (pred[sup_r, 2] / theta_c).pow(2).mean()
#         return loss

#     # ════════════════════════════════════════════════════════
#     # E. FORWARD
#     # ════════════════════════════════════════════════════════

#     def forward(self, model, data):
#         """
#         Minimize Π = U_axial(EA×factor) + U_bend(EI) - W_ext
#         """
#         # 1. Forward
#         pred = model(data)

#         # 2. Local displacements
#         conn = data.connectivity
#         dirs = data.elem_directions
#         L    = data.elem_lengths
#         EA   = data.prop_E * data.prop_A
#         EI   = data.prop_E * data.prop_I22

#         disp_A_loc, disp_B_loc = self._to_local(pred, dirs, conn)

#         # 3. Curriculum EA factor
#         ea_factor = self._get_ea_factor()

#         # 4. Strain energy
#         U_axial, U_bend = self._compute_strain_energy(
#             disp_A_loc, disp_B_loc, EA, EI, L, ea_factor
#         )
#         U_total = U_axial.sum() + U_bend.sum()

#         # 5. External work
#         W_ext = self._compute_external_work(pred, data.F_ext)

#         # 6. Potential energy
#         Pi = U_total - W_ext

#         # 7. Non-dimensionalize
#         E_c = data.F_c * data.u_c
#         Pi_nd = Pi / E_c

#         # 8. BC loss
#         L_bc = self._loss_bc(pred, data)

#         # 9. Total
#         total = self.w_energy * Pi_nd + self.w_bc * L_bc

#         loss_dict = {
#             'Pi':        Pi_nd.item(),
#             'U_axial':   (U_axial.sum() / E_c).item(),
#             'U_bend':    (U_bend.sum() / E_c).item(),
#             'W_ext':     (W_ext / E_c).item(),
#             'L_bc':      L_bc.item(),
#             'total':     total.item(),
#             'ea_factor': ea_factor,                            # ◀◀◀ TRACK
#         }

#         return total, loss_dict, pred

# """
# =================================================================
# physics_loss.py — Minimum Potential Energy Loss
# =================================================================

# Instead of ||Ku - f||² (has zero trap),
# minimize Π = U_strain - W_external (convex, no trap).

#   U_strain = ½ Σ_e [EA/L · (u_sB - u_sA)² + EI bending terms]
#   W_ext    = Σ_nodes u · F_ext

# Minimum of Π is the exact FEM solution.
# =================================================================
# """

# import torch
# import torch.nn as nn


# class EnergyLoss(nn.Module):

#     def __init__(self, w_energy=1.0, w_bc=1.0):
#         super().__init__()
#         self.w_energy = w_energy
#         self.w_bc = w_bc

#     # ════════════════════════════════════════════════════════
#     # A. GLOBAL → LOCAL
#     # ════════════════════════════════════════════════════════

#     def _to_local(self, disp, elem_directions, conn):
#         cos_a = elem_directions[:, 0:1]
#         sin_a = elem_directions[:, 2:3]

#         d_A = disp[conn[:, 0]]
#         d_B = disp[conn[:, 1]]

#         def rotate(d):
#             ux = d[:, 0:1]
#             uz = d[:, 1:2]
#             th = d[:, 2:3]
#             return torch.cat([
#                  ux * cos_a + uz * sin_a,
#                 -ux * sin_a + uz * cos_a,
#                  th,
#             ], dim=1)

#         return rotate(d_A), rotate(d_B)

#     # ════════════════════════════════════════════════════════
#     # B. STRAIN ENERGY
#     # ════════════════════════════════════════════════════════

#     def _compute_strain_energy(self, disp_A_loc, disp_B_loc,
#                                 EA, EI, L):
#         """
#         Euler-Bernoulli strain energy per element.

#         Axial:
#           U_axial = ½ · EA/L · (u_sB - u_sA)²

#         Bending (from Hermite shape functions):
#           U_bend = ½ · EI/L³ · [12(wA² + wB²) - 24·wA·wB
#                                 + 4L²(θA² + θB²)
#                                 + 12L(wA·θA - wB·θB)
#                                 - 4L²·θA·θB  (cross term from M12)
#                                 ... actually easier to use stiffness matrix]

#         Using element stiffness matrix k_e:
#           U_e = ½ · d^T · k_e · d

#         For bending DOFs [wA, θA, wB, θB]:
#           k_bend = EI/L³ × [ 12   6L  -12   6L ]
#                             [ 6L  4L²  -6L  2L² ]
#                             [-12  -6L   12  -6L ]
#                             [ 6L  2L² -6L   4L² ]

#           U_bend = ½ (EI/L³) × [12wA² + 4L²θA² + 12wB² + 4L²θB²
#                                  + 12L·wA·θA - 24wA·wB + 12L·wA·θB  (wait, let me be careful)
        
#         Actually let me just compute d^T K d directly:
#         """
#         u_sA = disp_A_loc[:, 0]
#         w_A  = disp_A_loc[:, 1]
#         th_A = disp_A_loc[:, 2]
#         u_sB = disp_B_loc[:, 0]
#         w_B  = disp_B_loc[:, 1]
#         th_B = disp_B_loc[:, 2]

#         # ── Axial strain energy ──
#         du = u_sB - u_sA
#         U_axial = 0.5 * EA / L * du.pow(2)

#         # ── Bending strain energy ──
#         # Using k_bend = EI/L³ × standard 4×4 matrix
#         # U_bend = ½ [wA θA wB θB] · k_bend · [wA θA wB θB]^T
#         #
#         # Expanding d^T K d:
#         #   12·wA² + 4L²·θA² + 12·wB² + 4L²·θB²
#         #   + 2·(6L)·wA·θA + 2·(-12)·wA·wB + 2·(6L)·wA·θB (wait, need to be careful with off-diag)
#         #
#         # k_bend[i,j] symmetic, so d^T K d = Σ_i Σ_j k_ij · d_i · d_j
#         # = k11·wA² + 2·k12·wA·θA + 2·k13·wA·wB + 2·k14·wA·θB
#         #   + k22·θA² + 2·k23·θA·wB + 2·k24·θA·θB
#         #   + k33·wB² + 2·k34·wB·θB
#         #   + k44·θB²
#         #
#         # With k = EI/L³ × [12, 6L, -12, 6L; 6L, 4L², -6L, 2L²; -12, -6L, 12, -6L; 6L, 2L², -6L, 4L²]

#         c = EI / L.pow(3)
#         L2 = L.pow(2)

#         U_bend = 0.5 * c * (
#               12.0 * w_A.pow(2)
#             + 2.0 * 6.0 * L * w_A * th_A
#             + 2.0 * (-12.0) * w_A * w_B
#             + 2.0 * 6.0 * L * w_A * th_B
#             + 4.0 * L2 * th_A.pow(2)
#             + 2.0 * (-6.0) * L * th_A * w_B
#             + 2.0 * 2.0 * L2 * th_A * th_B
#             + 12.0 * w_B.pow(2)
#             + 2.0 * (-6.0) * L * w_B * th_B
#             + 4.0 * L2 * th_B.pow(2)
#         )

#         return U_axial, U_bend

#     # ════════════════════════════════════════════════════════
#     # C. EXTERNAL WORK
#     # ════════════════════════════════════════════════════════

#     def _compute_external_work(self, pred, F_ext):
#         """
#         W_ext = Σ_nodes (ux · Fx_ext + uz · Fz_ext + θ · My_ext)
#         """
#         W = (pred * F_ext).sum()
#         return W

#     # ════════════════════════════════════════════════════════
#     # D. LOSSES
#     # ════════════════════════════════════════════════════════

#     def _loss_bc(self, pred, data):
#         bc_disp = data.bc_disp.squeeze(-1)
#         bc_rot  = data.bc_rot.squeeze(-1)
#         u_c     = data.u_c
#         theta_c = data.theta_c

#         sup_d = (bc_disp > 0.5)
#         sup_r = (bc_rot > 0.5)

#         loss = torch.tensor(0.0, device=pred.device)
#         if sup_d.any():
#             loss = loss + (pred[sup_d, 0] / u_c).pow(2).mean()
#             loss = loss + (pred[sup_d, 1] / u_c).pow(2).mean()
#         if sup_r.any():
#             loss = loss + (pred[sup_r, 2] / theta_c).pow(2).mean()
#         return loss

#     # ════════════════════════════════════════════════════════
#     # E. FORWARD
#     # ════════════════════════════════════════════════════════

#     def forward(self, model, data):
#         """
#         Minimize potential energy Π = U - W

#         Non-dimensionalized by U_c = F_c × u_c (characteristic energy)
#         """
#         # 1. Forward → (N, 3) displacements in physical units
#         pred = model(data)

#         # 2. Local displacements
#         conn = data.connectivity
#         dirs = data.elem_directions
#         L    = data.elem_lengths
#         EA   = data.prop_E * data.prop_A
#         EI   = data.prop_E * data.prop_I22

#         disp_A_loc, disp_B_loc = self._to_local(pred, dirs, conn)

#         # 3. Strain energy
#         U_axial, U_bend = self._compute_strain_energy(
#             disp_A_loc, disp_B_loc, EA, EI, L
#         )
#         U_total = U_axial.sum() + U_bend.sum()

#         # 4. External work
#         W_ext = self._compute_external_work(pred, data.F_ext)

#         # 5. Potential energy
#         Pi = U_total - W_ext

#         # 6. Non-dimensionalize
#         # Characteristic energy = F_c × u_c
#         E_c = data.F_c * data.u_c
#         Pi_nd = Pi / E_c

#         # 7. BC loss
#         L_bc = self._loss_bc(pred, data)

#         # 8. Total
#         total = self.w_energy * Pi_nd + self.w_bc * L_bc

#         loss_dict = {
#             'Pi':       Pi_nd.item(),
#             'U_axial':  (U_axial.sum() / E_c).item(),
#             'U_bend':   (U_bend.sum() / E_c).item(),
#             'W_ext':    (W_ext / E_c).item(),
#             'L_bc':     L_bc.item(),
#             'total':    total.item(),
#         }

#         return total, loss_dict, pred



"""
=================================================================
physics_loss.py — Minimum Potential Energy + Gentle Curriculum
=================================================================
"""

import torch
import torch.nn as nn
import math


class EnergyLoss(nn.Module):

    def __init__(self, w_energy=1.0, w_bc=1.0,
                 ea_start_factor=0.1,
                 ea_ramp_start=0,
                 ea_ramp_end=1000,
                 total_epochs=10000):
        super().__init__()
        self.w_energy = w_energy
        self.w_bc = w_bc
        self.ea_start_factor = ea_start_factor
        self.ea_ramp_start = ea_ramp_start
        self.ea_ramp_end = ea_ramp_end
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def _get_ea_factor(self):
        epoch = self.current_epoch
        if epoch <= self.ea_ramp_start:
            return self.ea_start_factor
        if epoch >= self.ea_ramp_end:
            return 1.0
        t = (epoch - self.ea_ramp_start) / (self.ea_ramp_end - self.ea_ramp_start)
        log_start = math.log10(self.ea_start_factor)
        log_factor = log_start + t * (0.0 - log_start)
        return 10.0 ** log_factor

    def _to_local(self, disp, elem_directions, conn):
        cos_a = elem_directions[:, 0:1]
        sin_a = elem_directions[:, 2:3]
        d_A = disp[conn[:, 0]]
        d_B = disp[conn[:, 1]]

        def rotate(d):
            ux = d[:, 0:1]
            uz = d[:, 1:2]
            th = d[:, 2:3]
            return torch.cat([
                 ux * cos_a + uz * sin_a,
                -ux * sin_a + uz * cos_a,
                 th,
            ], dim=1)

        return rotate(d_A), rotate(d_B)

    def _compute_strain_energy(self, disp_A_loc, disp_B_loc,
                                EA, EI, L, ea_factor):
        u_sA = disp_A_loc[:, 0]
        w_A  = disp_A_loc[:, 1]
        th_A = disp_A_loc[:, 2]
        u_sB = disp_B_loc[:, 0]
        w_B  = disp_B_loc[:, 1]
        th_B = disp_B_loc[:, 2]

        EA_eff = EA * ea_factor
        du = u_sB - u_sA
        U_axial = 0.5 * EA_eff / L * du.pow(2)

        c = EI / L.pow(3)
        L2 = L.pow(2)

        U_bend = 0.5 * c * (
              12.0 * w_A.pow(2)
            + 12.0 * L * w_A * th_A
            - 24.0 * w_A * w_B
            + 12.0 * L * w_A * th_B
            + 4.0 * L2 * th_A.pow(2)
            - 12.0 * L * th_A * w_B
            + 4.0 * L2 * th_A * th_B
            + 12.0 * w_B.pow(2)
            - 12.0 * L * w_B * th_B
            + 4.0 * L2 * th_B.pow(2)
        )

        return U_axial, U_bend

    def _compute_external_work(self, pred, F_ext):
        return (pred * F_ext).sum()

    def _loss_bc(self, pred, data):
        bc_disp = data.bc_disp.squeeze(-1)
        bc_rot  = data.bc_rot.squeeze(-1)
        u_c     = data.u_c
        theta_c = data.theta_c

        sup_d = (bc_disp > 0.5)
        sup_r = (bc_rot > 0.5)

        loss = torch.tensor(0.0, device=pred.device)
        if sup_d.any():
            loss = loss + (pred[sup_d, 0] / u_c).pow(2).mean()
            loss = loss + (pred[sup_d, 1] / u_c).pow(2).mean()
        if sup_r.any():
            loss = loss + (pred[sup_r, 2] / theta_c).pow(2).mean()
        return loss

    def forward(self, model, data):
        pred = model(data)

        conn = data.connectivity
        dirs = data.elem_directions
        L    = data.elem_lengths
        EA   = data.prop_E * data.prop_A
        EI   = data.prop_E * data.prop_I22

        disp_A_loc, disp_B_loc = self._to_local(pred, dirs, conn)

        ea_factor = self._get_ea_factor()

        U_axial, U_bend = self._compute_strain_energy(
            disp_A_loc, disp_B_loc, EA, EI, L, ea_factor
        )
        U_total = U_axial.sum() + U_bend.sum()

        W_ext = self._compute_external_work(pred, data.F_ext)

        Pi = U_total - W_ext

        E_c = data.F_c * data.u_c
        Pi_nd = Pi / E_c

        L_bc = self._loss_bc(pred, data)

        total = self.w_energy * Pi_nd + self.w_bc * L_bc

        loss_dict = {
            'Pi':        Pi_nd.item(),
            'U_axial':   (U_axial.sum() / E_c).item(),
            'U_bend':    (U_bend.sum() / E_c).item(),
            'W_ext':     (W_ext / E_c).item(),
            'L_bc':      L_bc.item(),
            'total':     total.item(),
            'ea_factor': ea_factor,
        }

        return total, loss_dict, pred