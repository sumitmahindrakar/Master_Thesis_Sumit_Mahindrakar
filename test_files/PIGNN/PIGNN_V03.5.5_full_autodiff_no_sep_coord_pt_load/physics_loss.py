"""
=================================================================
physics_loss.py — Naive Autograd Physics Loss (6 Terms)
               + NON-DIMENSIONALIZED RESIDUALS
=================================================================

CHANGES:
  Every residual is divided by a characteristic scale:
    L_eq:   residuals / F_c (force) and / M_c (moment)
    L_free: residuals / F_c and / M_c
    L_sup:  residuals / u_c (disp) and / theta_c (rotation)
    L_N:    residuals / F_c
    L_M:    residuals / M_c
    L_V:    residuals / F_c
=================================================================
"""

import torch
import torch.nn as nn


class NaivePhysicsLoss(nn.Module):

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
    # SECTION A: HELPERS
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
    # SECTION B: EQUILIBRIUM LOSSES —  NON-DIMENSIONALIZED
    # ════════════════════════════════════════════════════════

    def _loss_equilibrium(self, face_forces, F_ext, bc_disp, bc_rot,
                           data):                              #  added data
        """
        B1. Nodal equilibrium at free nodes.
        Residuals non-dimensionalized by F_c and M_c.
        """
        sum_forces = face_forces.sum(dim=1)           # (N, 3)
        residual = sum_forces - F_ext                  # (N, 3)

        free_mask = (bc_disp.squeeze(-1) < 0.5)

        if not free_mask.any():
            return torch.tensor(0.0, device=face_forces.device)

        F_c = data.F_c                                 #  
        M_c = data.M_c                                 #  

        # Non-dim: forces by F_c, moments by M_c       #  
        res_Fx = residual[free_mask, 0] / F_c           #  
        res_Fz = residual[free_mask, 1] / F_c           #  
        res_My = residual[free_mask, 2] / M_c           #  

        L_eq = (res_Fx.pow(2)
              + res_Fz.pow(2)
              + res_My.pow(2)).mean()

        return L_eq

    def _loss_free_face(self, face_forces, face_mask,
                         data):                                #  added data
        """
        B2. Unconnected face forces = 0.
        Non-dimensionalized: Fx,Fz by F_c, My by M_c.
        """
        free = (face_mask < 0.5)                       # (N, 4)

        if not free.any():
            return torch.tensor(0.0, device=face_forces.device)

        F_c = data.F_c                                  #  
        M_c = data.M_c                                  #  

        # Non-dim the face forces                       #  
        ff_nd = torch.stack([
            face_forces[:, :, 0] / F_c,                 # Fx / F_c
            face_forces[:, :, 1] / F_c,                 # Fz / F_c
            face_forces[:, :, 2] / M_c,                 # My / M_c
        ], dim=2)                                        # (N, 4, 3)

        free_expanded = free.unsqueeze(-1).expand_as(ff_nd)
        free_vals = ff_nd[free_expanded]

        L_free = free_vals.pow(2).mean()
        return L_free

    def _loss_support(self, disp, bc_disp, bc_rot,
                       data):                                  #  added data
        """
        B3. Support displacements = 0.
        Non-dimensionalized: disp by u_c, rotation by theta_c.
        """
        sup_disp = (bc_disp.squeeze(-1) > 0.5)
        sup_rot  = (bc_rot.squeeze(-1) > 0.5)

        u_c     = data.u_c                              #  
        theta_c = data.theta_c                          #  

        loss = torch.tensor(0.0, device=disp.device)

        if sup_disp.any():
            loss = loss + (disp[sup_disp, 0] / u_c).pow(2).mean()    #  /u_c
            loss = loss + (disp[sup_disp, 1] / u_c).pow(2).mean()    #  /u_c

        if sup_rot.any():
            loss = loss + (disp[sup_rot, 2] / theta_c).pow(2).mean() #  /theta_c

        return loss

    # ════════════════════════════════════════════════════════
    # SECTION C: CONSTITUTIVE LOSSES — NON-DIMENSIONALIZED
    # ════════════════════════════════════════════════════════

    def _compute_autograd_derivs(self, pred, coords):
        """C0.  — same autograd derivatives."""
        ux = pred[:, 0]
        uz = pred[:, 1]

        grad_ux = torch.autograd.grad(
            ux.sum(), coords, create_graph=True, retain_graph=True)[0]
        grad_uz = torch.autograd.grad(
            uz.sum(), coords, create_graph=True, retain_graph=True)[0]

        dux_dx = grad_ux[:, 0]
        dux_dz = grad_ux[:, 2]
        duz_dx = grad_uz[:, 0]
        duz_dz = grad_uz[:, 2]

        grad_duz_dx = torch.autograd.grad(
            duz_dx.sum(), coords, create_graph=True, retain_graph=True)[0]
        d2uz_dx2 = grad_duz_dx[:, 0]

        grad_dux_dz = torch.autograd.grad(
            dux_dz.sum(), coords, create_graph=True, retain_graph=True)[0]
        d2ux_dz2 = grad_dux_dz[:, 2]

        grad_d2uz = torch.autograd.grad(
            d2uz_dx2.sum(), coords, create_graph=True, retain_graph=True)[0]
        d3uz_dx3 = grad_d2uz[:, 0]

        grad_d2ux = torch.autograd.grad(
            d2ux_dz2.sum(), coords, create_graph=True, retain_graph=True)[0]
        d3ux_dz3 = grad_d2ux[:, 2]

        return {
            'dux_dx': dux_dx, 'dux_dz': dux_dz,
            'duz_dx': duz_dx, 'duz_dz': duz_dz,
            'd2uz_dx2': d2uz_dx2, 'd2ux_dz2': d2ux_dz2,
            'd3uz_dx3': d3uz_dx3, 'd3ux_dz3': d3ux_dz3,
        }

    def _classify_elements(self, elem_directions):
        """."""
        dx = elem_directions[:, 0]
        dz = elem_directions[:, 2]
        is_horizontal = dx.abs() > dz.abs()
        cos_a = dx.unsqueeze(-1)
        sin_a = dz.unsqueeze(-1)
        return is_horizontal, cos_a, sin_a

    def _loss_axial(self, derivs, ff_A_loc, ff_B_loc,
                     data):                                    #   
        """
        C1. Axial: N = EA · du_L/ds
        Residual non-dimensionalized by F_c.
        """
        conn = data.connectivity
        EA = (data.prop_E * data.prop_A)
        is_horiz, _, _ = self._classify_elements(data.elem_directions)
        F_c = data.F_c                                  #  

        nA = conn[:, 0]
        nB = conn[:, 1]

        strain_A = torch.where(is_horiz,
                               derivs['dux_dx'][nA],
                               derivs['duz_dz'][nA])
        strain_B = torch.where(is_horiz,
                               derivs['dux_dx'][nB],
                               derivs['duz_dz'][nB])

        N_A = EA * strain_A
        N_B = EA * strain_B

        Fx_A = ff_A_loc[:, 0]
        Fx_B = ff_B_loc[:, 0]

        # Non-dim residuals                              #  
        res_A = (Fx_A + N_A) / F_c                       #  / F_c
        res_B = (Fx_B - N_B) / F_c                       #  / F_c

        L_N = (res_A.pow(2) + res_B.pow(2)).mean()
        return L_N

    def _loss_moment(self, derivs, ff_A_loc, ff_B_loc,
                      data):                                   #   
        """
        C2. Moment: M = EI · d²w_L/ds²
        Residual non-dimensionalized by M_c.
        """
        conn = data.connectivity
        EI = (data.prop_E * data.prop_I22)
        is_horiz, _, _ = self._classify_elements(data.elem_directions)
        M_c = data.M_c                                  #  

        nA = conn[:, 0]
        nB = conn[:, 1]

        curv_A = torch.where(is_horiz,
                             derivs['d2uz_dx2'][nA],
                             -derivs['d2ux_dz2'][nA])
        curv_B = torch.where(is_horiz,
                             derivs['d2uz_dx2'][nB],
                             -derivs['d2ux_dz2'][nB])

        M_A = EI * curv_A
        M_B = EI * curv_B

        My_A = ff_A_loc[:, 2]
        My_B = ff_B_loc[:, 2]

        # Non-dim residuals                              #  
        res_A = (My_A + M_A) / M_c                       #  / M_c
        res_B = (My_B - M_B) / M_c                       #  / M_c

        L_M = (res_A.pow(2) + res_B.pow(2)).mean()
        return L_M

    def _loss_shear(self, derivs, ff_A_loc, ff_B_loc,
                     data):                                    #   
        """
        C3. Shear: V = EI · d³w_L/ds³
        Residual non-dimensionalized by F_c.
        """
        conn = data.connectivity
        EI = (data.prop_E * data.prop_I22)
        is_horiz, _, _ = self._classify_elements(data.elem_directions)
        F_c = data.F_c                                   #  

        nA = conn[:, 0]
        nB = conn[:, 1]

        d3w_A = torch.where(is_horiz,
                            derivs['d3uz_dx3'][nA],
                            -derivs['d3ux_dz3'][nA])
        d3w_B = torch.where(is_horiz,
                            derivs['d3uz_dx3'][nB],
                            -derivs['d3ux_dz3'][nB])

        V_A = EI * d3w_A
        V_B = EI * d3w_B

        Fz_A = ff_A_loc[:, 1]
        Fz_B = ff_B_loc[:, 1]

        # Non-dim residuals                              #  
        res_A = (Fz_A + V_A) / F_c                       #  / F_c
        res_B = (Fz_B - V_B) / F_c                       #  / F_c

        L_V = (res_A.pow(2) + res_B.pow(2)).mean()
        return L_V

    # ════════════════════════════════════════════════════════
    # SECTION D: TOTAL LOSS — SIGNATURE CHANGES
    # ════════════════════════════════════════════════════════

    def forward(self, model, data):
        """
        Same flow as before.
        Only change: all sub-losses  receive `data` for scales.
        """
        # 1. Forward
        pred = model(data)
        coords = model.get_coords()

        # 2. Extract
        disp, face_forces = self._extract_predictions(pred)
        
        # ═══ DIAGNOSTIC ═══
        # print(f"\n  DIAGNOSTIC:")
        # print(f"    pred disp range:  [{disp.min():.6e}, {disp.max():.6e}]")
        # print(f"    pred disp / u_c:  [{(disp[:,:2]/data.u_c).min():.4f}, "
        #     f"{(disp[:,:2]/data.u_c).max():.4f}]")
        # print(f"    face_forces range: [{face_forces.min():.6e}, "
        #     f"{face_forces.max():.6e}]")
        # print(f"    face_forces / F_c: [{(face_forces[:,:,:2]/data.F_c).min():.4f}, "
        #     f"{(face_forces[:,:,:2]/data.F_c).max():.4f}]")
        
        # derivs = self._compute_autograd_derivs(pred, coords)
        # print(f"    dux_dx range:     [{derivs['dux_dx'].min():.6e}, "
        #     f"{derivs['dux_dx'].max():.6e}]")
        # print(f"    d2uz_dx2 range:   [{derivs['d2uz_dx2'].min():.6e}, "
        #     f"{derivs['d2uz_dx2'].max():.6e}]")
        # print(f"    d3uz_dx3 range:   [{derivs['d3uz_dx3'].min():.6e}, "
        #     f"{derivs['d3uz_dx3'].max():.6e}]")
        
        # EA = data.prop_E * data.prop_A
        # EI = data.prop_E * data.prop_I22
        # print(f"    EA range:         [{EA.min():.4e}, {EA.max():.4e}]")
        # print(f"    EI range:         [{EI.min():.4e}, {EI.max():.4e}]")
        # print(f"    EA * dux_dx:      [{(EA * derivs['dux_dx'][data.connectivity[:,0]]).min():.4e}, "
        #     f"{(EA * derivs['dux_dx'][data.connectivity[:,0]]).max():.4e}]")
        # print(f"    EI * d2uz_dx2:    [{(EI * derivs['d2uz_dx2'][data.connectivity[:,0]]).min():.4e}, "
        #     f"{(EI * derivs['d2uz_dx2'][data.connectivity[:,0]]).max():.4e}]")
        # ═══ END DIAGNOSTIC ═══


        # 3. Equilibrium (pass data for scales)           #  
        L_eq   = self._loss_equilibrium(
            face_forces, data.F_ext, data.bc_disp, data.bc_rot,
            data                                          #   
        )
        L_free = self._loss_free_face(
            face_forces, data.face_mask,
            data                                          #   
        )
        L_sup  = self._loss_support(
            disp, data.bc_disp, data.bc_rot,
            data                                          #   
        )

        # 4. Transform to local
        disp_A, disp_B, ff_A, ff_B = \
            self._get_element_end_data_vectorized(
                disp, face_forces, data
            )
        disp_A_loc, disp_B_loc, ff_A_loc, ff_B_loc = \
            self._transform_to_local(
                disp_A, disp_B, ff_A, ff_B,
                data.elem_directions
            )

        # 5. Constitutive (pass data for scales)          #  
        derivs = self._compute_autograd_derivs(pred, coords)

        L_N = self._loss_axial(derivs, ff_A_loc, ff_B_loc, data)
        L_M = self._loss_moment(derivs, ff_A_loc, ff_B_loc, data)
        L_V = self._loss_shear(derivs, ff_A_loc, ff_B_loc, data)

        # 6. Total
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