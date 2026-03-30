"""
=================================================================
energy_loss.py — Equilibrium Residual Loss (StructGNN-E style)
=================================================================

Loss = ||F_internal + F_external||² / ||F_external||²

F_internal is computed from predicted displacements using
element stiffness matrices. This is 100% unsupervised.

Advantages over TPE:
  - Loss is always >= 0
  - Target is 0 (clear convergence criterion)
  - No divergence possible
  - Same physics, better optimization landscape
=================================================================
"""

import torch
import torch.nn as nn


class FrameEnergyLoss(nn.Module):
    """
    Equilibrium residual loss for 2D frames.
    
    1. Predict displacements u
    2. Compute internal forces: F_in = K * u (assembled)
    3. Loss = ||F_in + F_ext||² / ||F_ext||²
    
    Still 100% unsupervised — no labeled data needed.
    """

    def __init__(self):
        super().__init__()

    def forward(self, model, data):
        pred_raw = model(data)  # non-dimensional ~O(1)

        u_c = data.u_c
        theta_c = data.theta_c

        u_phys = torch.zeros_like(pred_raw)
        u_phys[:, 0] = pred_raw[:, 0] * u_c
        u_phys[:, 1] = pred_raw[:, 1] * u_c
        u_phys[:, 2] = pred_raw[:, 2] * theta_c

        F_int = self._assemble_internal_forces(u_phys, data)
        F_ext = data.F_ext

        residual = F_int + F_ext

        # Normalize residual per-DOF before squaring
        F_c = data.F_c
        M_c = F_c * data.L_c

        res_nd = torch.zeros_like(residual)
        res_nd[:, 0] = residual[:, 0] / F_c
        res_nd[:, 1] = residual[:, 1] / F_c
        res_nd[:, 2] = residual[:, 2] / M_c

        # Only free DOFs (not support nodes)
        free_mask = (data.bc_disp.squeeze() < 0.5)
        
        if free_mask.any():
            loss = res_nd[free_mask].pow(2).mean()
        else:
            loss = res_nd.pow(2).mean()

        with torch.no_grad():
            U = self._strain_energy(u_phys, data)
            W = self._external_work(u_phys, data)

        loss_dict = {
            'total':      loss.item(),
            'Pi':         (U - W).item(),
            'Pi_norm':    loss.item(),
            'U_internal': U.item(),
            'W_external': W.item(),
            'U_over_W':   (U / W.abs().clamp(min=1e-30)).item(),
            'ux_range':   [u_phys[:, 0].min().item(),
                        u_phys[:, 0].max().item()],
            'uz_range':   [u_phys[:, 1].min().item(),
                        u_phys[:, 1].max().item()],
            'th_range':   [u_phys[:, 2].min().item(),
                        u_phys[:, 2].max().item()],
            'raw_range':  [pred_raw.min().item(),
                        pred_raw.max().item()],
        }

        return loss, loss_dict, pred_raw, u_phys

    def _assemble_internal_forces(self, u_phys, data):
        """
        Compute global internal force vector by assembling
        element contributions: F_int = Σ K_e * d_e
        
        Returns (N, 3): [Fx_int, Fz_int, My_int] per node
        """
        conn = data.connectivity
        nA, nB = conn[:, 0], conn[:, 1]
        n_elem = conn.shape[0]
        N = u_phys.shape[0]
        device = u_phys.device

        L  = data.elem_lengths
        EA = data.prop_E * data.prop_A
        EI = data.prop_E * data.prop_I22

        c = data.elem_directions[:, 0]
        s = data.elem_directions[:, 2]

        # ═══════════════════════════════════════
        # Global DOFs at element ends
        # ═══════════════════════════════════════
        ux_A = u_phys[nA, 0]; uz_A = u_phys[nA, 1]; th_A = u_phys[nA, 2]
        ux_B = u_phys[nB, 0]; uz_B = u_phys[nB, 1]; th_B = u_phys[nB, 2]

        # ═══════════════════════════════════════
        # Transform to local coordinates
        # ═══════════════════════════════════════
        u_A_loc =  c * ux_A + s * uz_A
        w_A_loc = -s * ux_A + c * uz_A
        u_B_loc =  c * ux_B + s * uz_B
        w_B_loc = -s * ux_B + c * uz_B

        # Negate θ for Kratos convention
        th_A_loc = -th_A
        th_B_loc = -th_B

        # Local DOF vector: [u_A, w_A, θ_A, u_B, w_B, θ_B]
        d_local = torch.stack([
            u_A_loc, w_A_loc, th_A_loc,
            u_B_loc, w_B_loc, th_B_loc
        ], dim=1)  # (E, 6)

        # ═══════════════════════════════════════
        # Local stiffness matrix K_local
        # ═══════════════════════════════════════
        K = torch.zeros(n_elem, 6, 6, device=device)

        ea_L  = EA / L
        ei_L  = EI / L
        ei_L2 = EI / L**2
        ei_L3 = EI / L**3

        # Axial
        K[:, 0, 0] =  ea_L;  K[:, 0, 3] = -ea_L
        K[:, 3, 0] = -ea_L;  K[:, 3, 3] =  ea_L

        # Bending
        K[:, 1, 1] =  12*ei_L3; K[:, 1, 2] =  6*ei_L2
        K[:, 1, 4] = -12*ei_L3; K[:, 1, 5] =  6*ei_L2
        K[:, 2, 1] =   6*ei_L2; K[:, 2, 2] =  4*ei_L
        K[:, 2, 4] =  -6*ei_L2; K[:, 2, 5] =  2*ei_L
        K[:, 4, 1] = -12*ei_L3; K[:, 4, 2] = -6*ei_L2
        K[:, 4, 4] =  12*ei_L3; K[:, 4, 5] = -6*ei_L2
        K[:, 5, 1] =   6*ei_L2; K[:, 5, 2] =  2*ei_L
        K[:, 5, 4] =  -6*ei_L2; K[:, 5, 5] =  4*ei_L

        # ═══════════════════════════════════════
        # Local forces: f_local = K_local * d_local
        # ═══════════════════════════════════════
        f_local = torch.bmm(K, d_local.unsqueeze(2)).squeeze(2)  # (E, 6)

        # f_local = [fA_u, fA_w, fA_θ, fB_u, fB_w, fB_θ] in local

        # ═══════════════════════════════════════
        # Transform local forces back to global
        # ═══════════════════════════════════════
        # Node A forces
        fA_u_loc = f_local[:, 0]
        fA_w_loc = f_local[:, 1]
        fA_th    = f_local[:, 2]

        # Inverse rotation: global = T^T * local
        fA_x =  c * fA_u_loc - s * fA_w_loc  # Fx_global
        fA_z =  s * fA_u_loc + c * fA_w_loc  # Fz_global
        fA_my = -fA_th  # Negate back for Kratos convention

        # Node B forces
        fB_u_loc = f_local[:, 3]
        fB_w_loc = f_local[:, 4]
        fB_th    = f_local[:, 5]

        fB_x =  c * fB_u_loc - s * fB_w_loc
        fB_z =  s * fB_u_loc + c * fB_w_loc
        fB_my = -fB_th

        # ═══════════════════════════════════════
        # Assemble into global force vector
        # ═══════════════════════════════════════
        F_int = torch.zeros(N, 3, device=device)

        # Scatter-add element contributions to nodes
        # Node A contributions
        F_int[:, 0].scatter_add_(0, nA, fA_x)
        F_int[:, 1].scatter_add_(0, nA, fA_z)
        F_int[:, 2].scatter_add_(0, nA, fA_my)

        # Node B contributions
        F_int[:, 0].scatter_add_(0, nB, fB_x)
        F_int[:, 1].scatter_add_(0, nB, fB_z)
        F_int[:, 2].scatter_add_(0, nB, fB_my)

        return F_int

    # ═══════════════════════════════════════════
    # Keep these for verification/logging
    # ═══════════════════════════════════════════

    def _strain_energy(self, u_phys, data):
        """Physical strain energy (for logging only)."""
        conn = data.connectivity
        nA, nB = conn[:, 0], conn[:, 1]
        n_elem = conn.shape[0]
        device = u_phys.device

        L  = data.elem_lengths
        EA = data.prop_E * data.prop_A
        EI = data.prop_E * data.prop_I22
        c = data.elem_directions[:, 0]
        s = data.elem_directions[:, 2]

        ux_A = u_phys[nA, 0]; uz_A = u_phys[nA, 1]; th_A = u_phys[nA, 2]
        ux_B = u_phys[nB, 0]; uz_B = u_phys[nB, 1]; th_B = u_phys[nB, 2]

        u_A_loc =  c * ux_A + s * uz_A
        w_A_loc = -s * ux_A + c * uz_A
        u_B_loc =  c * ux_B + s * uz_B
        w_B_loc = -s * ux_B + c * uz_B
        th_A_loc = -th_A
        th_B_loc = -th_B

        d_local = torch.stack([
            u_A_loc, w_A_loc, th_A_loc,
            u_B_loc, w_B_loc, th_B_loc
        ], dim=1)

        K = torch.zeros(n_elem, 6, 6, device=device)
        ea_L = EA/L; ei_L = EI/L; ei_L2 = EI/L**2; ei_L3 = EI/L**3

        K[:,0,0]= ea_L;  K[:,0,3]=-ea_L
        K[:,3,0]=-ea_L;  K[:,3,3]= ea_L
        K[:,1,1]= 12*ei_L3; K[:,1,2]= 6*ei_L2; K[:,1,4]=-12*ei_L3; K[:,1,5]= 6*ei_L2
        K[:,2,1]= 6*ei_L2;  K[:,2,2]= 4*ei_L;  K[:,2,4]=-6*ei_L2;  K[:,2,5]= 2*ei_L
        K[:,4,1]=-12*ei_L3; K[:,4,2]=-6*ei_L2; K[:,4,4]= 12*ei_L3; K[:,4,5]=-6*ei_L2
        K[:,5,1]= 6*ei_L2;  K[:,5,2]= 2*ei_L;  K[:,5,4]=-6*ei_L2;  K[:,5,5]= 4*ei_L

        Kd = torch.bmm(K, d_local.unsqueeze(2))
        U_per_elem = 0.5 * torch.bmm(d_local.unsqueeze(1), Kd).squeeze()
        return U_per_elem.sum()

    def _external_work(self, u_phys, data):
        """Physical external work (for logging only)."""
        W = (
            data.F_ext[:, 0] * u_phys[:, 0]
            + data.F_ext[:, 1] * u_phys[:, 1]
            + data.F_ext[:, 2] * u_phys[:, 2]
        ).sum()
        return W