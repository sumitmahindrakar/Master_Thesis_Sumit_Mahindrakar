"""
physics_loss.py — Corrected Axial + Bending equilibrium

Element forces on nodes (= -K*u from beam stiffness matrix):

AXIAL:
  N = EA/L · (u_j - u_i)·x̂

xz BENDING (w in ẑ_local, θy about ŷ_local):
  V_zi = 12·EI/L³·Δw  - 6·EI/L²·(θyi + θyj)
  M_yi =  6·EI/L²·Δw  - EI/L·(4·θyi + 2·θyj)
  M_yj =  6·EI/L²·Δw  - EI/L·(2·θyi + 4·θyj)

xy BENDING (v in ŷ_local, θz about ẑ_local):
  V_yi = 12·EI/L³·Δv  + 6·EI/L²·(θzi + θzj)     ← sign FLIPPED vs xz
  M_zi = -6·EI/L²·Δv  - EI/L·(4·θzi + 2·θzj)    ← sign FLIPPED vs xz
  M_zj = -6·EI/L²·Δv  - EI/L·(2·θzi + 4·θzj)

EQUILIBRIUM at each free node:
  Σ(element forces)  + F_external = 0
  Σ(element moments) + M_external = 0
"""

import torch
import torch.nn as nn


class PhysicsLoss(nn.Module):

    def __init__(self, w_force=1.0, w_moment=1.0, w_reg=1e-4):
        super().__init__()
        self.w_force = w_force
        self.w_moment = w_moment
        self.w_reg = w_reg       # small regularization

    def forward(self, pred, data):
        """
        pred:  (N, 6) [ux, uy, uz, rx, ry, rz]
        data:  PyG Data (RAW units)
        Returns: scalar loss
        """
        F_res, M_res = self.equilibrium_residual(pred, data)

        # Normalize by characteristic values
        EA = (data.prop_E * data.prop_A).mean().clamp(min=1.0)
        EI_L = (data.prop_E * data.prop_I22 / data.elem_lengths).mean().clamp(min=1.0)

        loss_F = (F_res / EA).pow(2).mean()
        loss_M = (M_res / EI_L).pow(2).mean()

        # Small regularization to push unconstrained DOFs toward zero
        loss_reg = pred.pow(2).mean()

        loss = self.w_force * loss_F + self.w_moment * loss_M + self.w_reg * loss_reg
        return loss

    def equilibrium_residual(self, pred, data):
        """
        Returns:
            F_res: (N_free_disp, 3)  force residual at free nodes
            M_res: (N_free_rot, 3)   moment residual at free nodes
        """
        N = pred.shape[0]
        device = pred.device

        u     = pred[:, :3]        # translations
        theta = pred[:, 3:6]       # rotations

        conn  = data.connectivity       # (E, 2)
        x_hat = data.elem_directions    # (E, 3) unit vector along element
        L     = data.elem_lengths       # (E,)
        E_mod = data.prop_E             # (E,)
        A     = data.prop_A             # (E,)
        I22   = data.prop_I22           # (E,)

        n_elem = conn.shape[0]
        i_nd = conn[:, 0]
        j_nd = conn[:, 1]

        # Local coordinate system
        y_hat, z_hat = self._local_axes(x_hat)

        # Nodal values
        u_i  = u[i_nd]          # (E, 3)
        u_j  = u[j_nd]          # (E, 3)
        th_i = theta[i_nd]      # (E, 3)
        th_j = theta[j_nd]      # (E, 3)

        du = u_j - u_i          # (E, 3)

        EI = E_mod * I22        # (E,)
        L2 = L * L
        L3 = L2 * L

        # ════════════════════════════════════════════
        # 1. AXIAL
        # ════════════════════════════════════════════
        axial_disp = (du * x_hat).sum(-1)                # (E,)
        N_axial = E_mod * A / L * axial_disp             # (E,)
        F_axial = N_axial.unsqueeze(-1) * x_hat          # (E, 3)

        # ════════════════════════════════════════════
        # 2. xz BENDING  (w in z_hat, θ about y_hat)
        # ════════════════════════════════════════════
        dw_z   = (du   * z_hat).sum(-1)                  # (E,)
        th_yi  = (th_i * y_hat).sum(-1)                  # (E,)
        th_yj  = (th_j * y_hat).sum(-1)                  # (E,)

        # Shear force on node i (in z_hat direction)
        V_z = (12.0 * EI / L3 * dw_z
               - 6.0 * EI / L2 * (th_yi + th_yj))       # (E,)

        F_shear_z = V_z.unsqueeze(-1) * z_hat            # (E, 3)

        # Moments about y_hat
        M_yi = (6.0 * EI / L2 * dw_z
                - EI / L * (4.0 * th_yi + 2.0 * th_yj))  # (E,)
        M_yj = (6.0 * EI / L2 * dw_z
                - EI / L * (2.0 * th_yi + 4.0 * th_yj))  # (E,)

        Mvec_yi = M_yi.unsqueeze(-1) * y_hat              # (E, 3)
        Mvec_yj = M_yj.unsqueeze(-1) * y_hat              # (E, 3)

        # ════════════════════════════════════════════
        # 3. xy BENDING  (v in y_hat, θ about z_hat)
        #    NOTE: coupling signs FLIPPED vs xz
        # ════════════════════════════════════════════
        dw_y   = (du   * y_hat).sum(-1)                   # (E,)
        th_zi  = (th_i * z_hat).sum(-1)                   # (E,)
        th_zj  = (th_j * z_hat).sum(-1)                   # (E,)

        # Shear force on node i (in y_hat direction)
        V_y = (12.0 * EI / L3 * dw_y
               + 6.0 * EI / L2 * (th_zi + th_zj))        # (E,)  ← + sign

        F_shear_y = V_y.unsqueeze(-1) * y_hat              # (E, 3)

        # Moments about z_hat
        M_zi = (-6.0 * EI / L2 * dw_y
                - EI / L * (4.0 * th_zi + 2.0 * th_zj))   # (E,)  ← - sign
        M_zj = (-6.0 * EI / L2 * dw_y
                - EI / L * (2.0 * th_zi + 4.0 * th_zj))   # (E,)

        Mvec_zi = M_zi.unsqueeze(-1) * z_hat               # (E, 3)
        Mvec_zj = M_zj.unsqueeze(-1) * z_hat               # (E, 3)

        # ════════════════════════════════════════════
        # 4. ASSEMBLE FORCES AT NODES
        # ════════════════════════════════════════════
        idx_i = i_nd.unsqueeze(-1).expand(n_elem, 3)
        idx_j = j_nd.unsqueeze(-1).expand(n_elem, 3)

        F_int = torch.zeros(N, 3, device=device)

        # Axial: +N on i, -N on j
        F_int.scatter_add_(0, idx_i, F_axial)
        F_int.scatter_add_(0, idx_j, -F_axial)

        # Shear z: +V on i, -V on j
        F_int.scatter_add_(0, idx_i, F_shear_z)
        F_int.scatter_add_(0, idx_j, -F_shear_z)

        # Shear y: +V on i, -V on j
        F_int.scatter_add_(0, idx_i, F_shear_y)
        F_int.scatter_add_(0, idx_j, -F_shear_y)

        # ════════════════════════════════════════════
        # 5. ASSEMBLE MOMENTS AT NODES
        # ════════════════════════════════════════════
        M_int = torch.zeros(N, 3, device=device)

        # xz bending moments (M_yi at i, M_yj at j — NOT negated)
        M_int.scatter_add_(0, idx_i, Mvec_yi)
        M_int.scatter_add_(0, idx_j, Mvec_yj)

        # xy bending moments (M_zi at i, M_zj at j — NOT negated)
        M_int.scatter_add_(0, idx_i, Mvec_zi)
        M_int.scatter_add_(0, idx_j, Mvec_zj)

        # ════════════════════════════════════════════
        # 6. RESIDUALS AT FREE NODES
        # ════════════════════════════════════════════
        F_ext = data.line_load                             # (N, 3)
        F_residual = F_int + F_ext                         # should → 0

        M_residual = M_int                                 # should → 0
        # (no external moments in this problem)

        free_d = (data.bc_disp.squeeze(-1) < 0.5)
        free_r = (data.bc_rot.squeeze(-1) < 0.5)

        return F_residual[free_d], M_residual[free_r]

    @staticmethod
    def _local_axes(x_hat):
        """
        Build orthonormal local y, z axes for each element.
        """
        E = x_hat.shape[0]
        device = x_hat.device

        ref = torch.zeros(E, 3, device=device)
        ref[:, 1] = 1.0                                   # default: global Y

        # If element is near-parallel to Y, use global Z
        parallel = (x_hat * ref).sum(-1).abs() > 0.99
        ref[parallel] = torch.tensor([0.0, 0.0, 1.0], device=device)

        z_hat = torch.cross(x_hat, ref, dim=-1)
        z_hat = z_hat / z_hat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        y_hat = torch.cross(z_hat, x_hat, dim=-1)
        y_hat = y_hat / y_hat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        return y_hat, z_hat