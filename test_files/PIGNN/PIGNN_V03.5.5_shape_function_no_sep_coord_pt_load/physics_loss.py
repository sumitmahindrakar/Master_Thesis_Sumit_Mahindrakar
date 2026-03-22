"""
=================================================================
physics_loss.py — Shape Function Physics Loss (6 Terms)
               + NON-DIMENSIONALIZED RESIDUALS
=================================================================

CHANGES FROM AUTOGRAD VERSION:
  Replaced _compute_autograd_derivs() with _compute_shape_function_forces()
  
  Constitutive losses now compare predicted face forces against
  EXACT shape-function-derived forces instead of contaminated
  autograd derivatives.

  Shape function formulas (Euler-Bernoulli, exact):
    N   = EA/L · (u_sB - u_sA)
    M_A = EI/L² · (-6·wA - 4L·θA + 6·wB - 2L·θB)
    M_B = EI/L² · ( 6·wA + 2L·θA - 6·wB + 4L·θB)
    V   = EI/L³ · (12·wA + 6L·θA - 12·wB + 6L·θB)

UNCHANGED:
  Same 15-value output, same face forces, same 6 loss terms,
  same non-dimensionalization, same equilibrium logic.
=================================================================
"""

import torch
import torch.nn as nn


class NaivePhysicsLoss(nn.Module):     #  SAME CLASS NAME — train.py unchanged

    def __init__(self,
                 w_eq=1.0,
                 w_free=1.0,
                 w_sup=1.0,
                 w_N=1.0,
                 w_M=1.0,
                 w_V=1.0):
        super().__init__()
        self.w_eq   = w_eq
        self.w_free = w_free
        self.w_sup  = w_sup
        self.w_N    = w_N
        self.w_M    = w_M
        self.w_V    = w_V

    # ════════════════════════════════════════════════════════
    # SECTION A: HELPERS (UNCHANGED)
    # ════════════════════════════════════════════════════════

    def _extract_predictions(self, pred):
        disp = pred[:, 0:3]
        face_forces = pred[:, 3:15].reshape(-1, 4, 3)
        return disp, face_forces

    def _get_element_end_data_vectorized(self, disp, face_forces, data):
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
            nodes_with_face = torch.where(mask)[0]
            elems = feid[nodes_with_face, f]
            is_A  = faa[nodes_with_face, f]
            forces = face_forces[nodes_with_face, f, :]

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
    # SECTION B: EQUILIBRIUM LOSSES (UNCHANGED)
    # ════════════════════════════════════════════════════════

    def _loss_equilibrium(self, face_forces, F_ext, bc_disp, bc_rot,
                           data):
        sum_forces = face_forces.sum(dim=1)
        residual = sum_forces - F_ext

        free_mask = (bc_disp.squeeze(-1) < 0.5)

        if not free_mask.any():
            return torch.tensor(0.0, device=face_forces.device)

        F_c = data.F_c
        M_c = data.M_c

        res_Fx = residual[free_mask, 0] / F_c
        res_Fz = residual[free_mask, 1] / F_c
        res_My = residual[free_mask, 2] / M_c

        L_eq = (res_Fx.pow(2)
              + res_Fz.pow(2)
              + res_My.pow(2)).mean()

        return L_eq

    def _loss_free_face(self, face_forces, face_mask, data):
        free = (face_mask < 0.5)

        if not free.any():
            return torch.tensor(0.0, device=face_forces.device)

        F_c = data.F_c
        M_c = data.M_c

        ff_nd = torch.stack([
            face_forces[:, :, 0] / F_c,
            face_forces[:, :, 1] / F_c,
            face_forces[:, :, 2] / M_c,
        ], dim=2)

        free_expanded = free.unsqueeze(-1).expand_as(ff_nd)
        free_vals = ff_nd[free_expanded]

        L_free = free_vals.pow(2).mean()
        return L_free

    def _loss_support(self, disp, bc_disp, bc_rot, data):
        sup_disp = (bc_disp.squeeze(-1) > 0.5)
        sup_rot  = (bc_rot.squeeze(-1) > 0.5)

        u_c     = data.u_c
        theta_c = data.theta_c

        loss = torch.tensor(0.0, device=disp.device)

        if sup_disp.any():
            loss = loss + (disp[sup_disp, 0] / u_c).pow(2).mean()
            loss = loss + (disp[sup_disp, 1] / u_c).pow(2).mean()

        if sup_rot.any():
            loss = loss + (disp[sup_rot, 2] / theta_c).pow(2).mean()

        return loss

    # ════════════════════════════════════════════════════════
    # SECTION C: CONSTITUTIVE — SHAPE FUNCTIONS (REPLACES AUTOGRAD)
    # ════════════════════════════════════════════════════════

    #  REMOVED: _compute_autograd_derivs() — entire method deleted

    def _classify_elements(self, elem_directions):            # UNCHANGED
        dx = elem_directions[:, 0]
        dz = elem_directions[:, 2]
        is_horizontal = dx.abs() > dz.abs()
        cos_a = dx.unsqueeze(-1)
        sin_a = dz.unsqueeze(-1)
        return is_horizontal, cos_a, sin_a

    def _compute_shape_function_forces(self, disp_A_loc, disp_B_loc,  #  NEW
                                        data):
        """
        Compute N, M_A, M_B, V from end-node displacements
        using EXACT Euler-Bernoulli shape functions.

        All operations are LINEAR on predicted values.
        No autograd. No approximation. Exact for EB beams.

        Local DOFs:
            disp_A_loc[:, 0] = u_sA  (axial at A)
            disp_A_loc[:, 1] = w_A   (transverse at A)
            disp_A_loc[:, 2] = θ_A   (rotation at A)

        Returns:
            N:   (E,) axial force (constant along element)
            M_A: (E,) moment at A-end
            M_B: (E,) moment at B-end
            V:   (E,) shear force (constant along element)
        """
        EA = data.prop_E * data.prop_A       # (E,)
        EI = data.prop_E * data.prop_I22     # (E,)
        L  = data.elem_lengths               # (E,)

        u_sA = disp_A_loc[:, 0]     # (E,)
        w_A  = disp_A_loc[:, 1]     # (E,)
        θ_A  = disp_A_loc[:, 2]     # (E,)

        u_sB = disp_B_loc[:, 0]     # (E,)
        w_B  = disp_B_loc[:, 1]     # (E,)
        θ_B  = disp_B_loc[:, 2]     # (E,)

        # ── Axial force (linear interpolation) ──
        N = EA * (u_sB - u_sA) / L

        # ── Moment at A-end (Hermite 2nd derivative at ξ=0) ──
        M_A = EI / L.pow(2) * (
            -6.0 * w_A - 4.0 * L * θ_A
            + 6.0 * w_B - 2.0 * L * θ_B
        )

        # ── Moment at B-end (Hermite 2nd derivative at ξ=1) ──
        M_B = EI / L.pow(2) * (
             6.0 * w_A + 2.0 * L * θ_A
            - 6.0 * w_B + 4.0 * L * θ_B
        )

        # ── Shear force (Hermite 3rd derivative, constant) ──
        V = EI / L.pow(3) * (
             12.0 * w_A + 6.0 * L * θ_A
            - 12.0 * w_B + 6.0 * L * θ_B
        )

        return N, M_A, M_B, V

    def _loss_axial(self, N_sf, ff_A_loc, ff_B_loc, data):     #  CHANGED
        """
        C1. Axial: compare predicted face Fx with shape-function N.

        Sign convention:
            At A: Fx_A^loc = -N  →  Fx_A + N = 0
            At B: Fx_B^loc = +N  →  Fx_B - N = 0

        N_sf is the shape-function-derived axial force (EXACT).
        """
        F_c = data.F_c

        Fx_A = ff_A_loc[:, 0]     # predicted face axial force at A
        Fx_B = ff_B_loc[:, 0]     # predicted face axial force at B

        res_A = (Fx_A + N_sf) / F_c        #  N_sf replaces EA*autograd_strain
        res_B = (Fx_B - N_sf) / F_c        #  N_sf replaces EA*autograd_strain

        L_N = (res_A.pow(2) + res_B.pow(2)).mean()
        return L_N

    def _loss_moment(self, M_A_sf, M_B_sf, ff_A_loc, ff_B_loc,  #  CHANGED
                      data):
        """
        C2. Moment: compare predicted face My with shape-function M.

        Sign convention:
            At A: My_A^loc = -M(0)  →  My_A + M_A = 0
            At B: My_B^loc = +M(L)  →  My_B - M_B = 0
        """
        M_c = data.M_c

        My_A = ff_A_loc[:, 2]
        My_B = ff_B_loc[:, 2]

        res_A = (My_A + M_A_sf) / M_c      #  M_A_sf replaces EI*autograd_curvature
        res_B = (My_B - M_B_sf) / M_c      #  M_B_sf replaces EI*autograd_curvature

        L_M = (res_A.pow(2) + res_B.pow(2)).mean()
        return L_M

    def _loss_shear(self, V_sf, ff_A_loc, ff_B_loc, data):     #  CHANGED
        """
        C3. Shear: compare predicted face Fz with shape-function V.

        Sign convention:
            At A: Fz_A^loc = -V  →  Fz_A + V = 0
            At B: Fz_B^loc = +V  →  Fz_B - V = 0
        """
        F_c = data.F_c

        Fz_A = ff_A_loc[:, 1]
        Fz_B = ff_B_loc[:, 1]

        res_A = (Fz_A + V_sf) / F_c        #  V_sf replaces EI*autograd_d3w
        res_B = (Fz_B - V_sf) / F_c        #  V_sf replaces EI*autograd_d3w

        L_V = (res_A.pow(2) + res_B.pow(2)).mean()
        return L_V

    # ════════════════════════════════════════════════════════
    # SECTION D: TOTAL LOSS
    # ════════════════════════════════════════════════════════

    def forward(self, model, data):                             #  CHANGED
        """
        Same 6 loss terms, same structure.
        Only difference: shape functions replace autograd.
        """
        # 1. Forward
        pred = model(data)                                      # (N, 15)
        # coords = model.get_coords()                           #  NOT NEEDED

        # 2. Extract
        disp, face_forces = self._extract_predictions(pred)

        # 3. Equilibrium (UNCHANGED)
        L_eq   = self._loss_equilibrium(
            face_forces, data.F_ext, data.bc_disp, data.bc_rot,
            data
        )
        L_free = self._loss_free_face(
            face_forces, data.face_mask, data
        )
        L_sup  = self._loss_support(
            disp, data.bc_disp, data.bc_rot, data
        )

        # 4. Transform to local (UNCHANGED)
        disp_A, disp_B, ff_A, ff_B = \
            self._get_element_end_data_vectorized(
                disp, face_forces, data
            )
        disp_A_loc, disp_B_loc, ff_A_loc, ff_B_loc = \
            self._transform_to_local(
                disp_A, disp_B, ff_A, ff_B,
                data.elem_directions
            )

        # 5. Shape function forces (REPLACES autograd)          #  CHANGED
        N_sf, M_A_sf, M_B_sf, V_sf = self._compute_shape_function_forces(
            disp_A_loc, disp_B_loc, data
        )                                                #  NEW

        # 6. Constitutive losses (use shape func, not autograd)  #  CHANGED
        L_N = self._loss_axial(N_sf, ff_A_loc, ff_B_loc, data)  #  CHANGED
        L_M = self._loss_moment(M_A_sf, M_B_sf,                 #  CHANGED
                                ff_A_loc, ff_B_loc, data)        #  CHANGED
        L_V = self._loss_shear(V_sf, ff_A_loc, ff_B_loc, data)  #  CHANGED

        # 7. Total (UNCHANGED)
        total = (self.w_eq   * L_eq
               + self.w_free * L_free
               + self.w_sup  * L_sup
               + self.w_N    * L_N
               + self.w_M    * L_M
               + self.w_V    * L_V)

        loss_dict = {
            'L_eq':    L_eq.item(),
            'L_free':  L_free.item(),
            'L_sup':   L_sup.item(),
            'L_N':     L_N.item(),
            'L_M':     L_M.item(),
            'L_V':     L_V.item(),
            'total':   total.item(),
        }

        return total, loss_dict, pred