"""
physics_loss.py — Hybrid Loss: Strong-Form + Finite Difference
"""

import torch
import torch.nn as nn


def _grad(y, x):
    return torch.autograd.grad(
        y.sum(), x,
        create_graph=True,
        retain_graph=True,
    )[0]


class StrongFormPhysicsLoss(nn.Module):
    """
    Hybrid physics loss combining:
    
    1. AUTOGRAD constitutive (exact gradients):
       ε = ∇u · x̂,  κ = ∇φ · x̂
       N = EA·ε,  M = EI·κ
    
    2. FINITE DIFFERENCE constitutive (robust backup):
       ε_fd = (u_j - u_i) · x̂ / L
       κ_fd = (φ_j - φ_i) / L
       N_fd = EA·ε_fd,  M_fd = EI·κ_fd
    
    3. CONSISTENCY between autograd and FD:
       ε_autograd ≈ ε_fd  (penalize mismatch)
    
    The FD terms provide strong gradient signal even when
    autograd gradients are flat. The consistency term
    teaches the decoder to produce correct spatial variation.
    """

    def __init__(self,
                 w_force=1.0,
                 w_moment=1.0,
                 w_kin=0.1,
                 w_neumann=1.0,
                 w_consistency=1.0):
        super().__init__()
        self.w_force       = w_force
        self.w_moment      = w_moment
        self.w_kin         = w_kin
        self.w_neumann     = w_neumann
        self.w_consistency = w_consistency

    def forward(self, model, data):
        pred = model(data)
        coords = model.get_coords()

        N_nodes = pred.shape[0]
        device  = pred.device

        u_x = pred[:, 0]              # (N,)
        u_z = pred[:, 1]              # (N,)
        phi = pred[:, 2]              # (N,)

        conn  = data.connectivity
        E_mod = data.prop_E
        A_    = data.prop_A
        I22   = data.prop_I22
        L_    = data.elem_lengths
        x_hat = data.elem_directions

        E_elems = conn.shape[0]
        i_nd = conn[:, 0]
        j_nd = conn[:, 1]

        EA = E_mod * A_
        EI = E_mod * I22

        y_hat, z_hat = self._local_axes(x_hat)

        # ══════════════════════════════════════
        # 1. FINITE DIFFERENCE: N, M, V
        #    (uses actual predicted node values)
        #    (ALWAYS provides gradient signal)
        # ══════════════════════════════════════

        # Displacement vectors at element ends
        u_i = torch.stack([u_x[i_nd], torch.zeros_like(u_x[i_nd]),
                           u_z[i_nd]], dim=-1)
        u_j = torch.stack([u_x[j_nd], torch.zeros_like(u_x[j_nd]),
                           u_z[j_nd]], dim=-1)
        du = u_j - u_i

        # Axial strain
        du_axial_fd = (du * x_hat).sum(-1)        # (E,)
        eps_fd = du_axial_fd / L_                   # (E,)
        N_fd = EA * eps_fd                          # (E,)

        # Transverse displacement difference
        du_trans_fd = (du * z_hat).sum(-1)          # (E,)

        # Rotation at ends
        phi_i = phi[i_nd]
        phi_j = phi[j_nd]

        # Curvature (finite difference)
        kappa_fd = (phi_j - phi_i) / L_             # (E,)
        M_fd = EI * kappa_fd                        # (E,)

        # Shear from beam theory
        V_fd = (12.0 * EI / L_**3 * du_trans_fd
              - 6.0 * EI / L_**2 * (phi_i + phi_j))

        # Moments at each end (for moment equilibrium)
        M_yi = (6.0 * EI / L_**2 * du_trans_fd
              - EI / L_ * (4.0 * phi_i + 2.0 * phi_j))
        M_yj = (6.0 * EI / L_**2 * du_trans_fd
              - EI / L_ * (2.0 * phi_i + 4.0 * phi_j))

        # ══════════════════════════════════════
        # 2. AUTOGRAD: ∇u, ∇φ (for consistency)
        # ══════════════════════════════════════

        grad_ux  = _grad(pred[:, 0:1], coords)    # (N, 3)
        grad_uz  = _grad(pred[:, 1:2], coords)    # (N, 3)
        grad_phi = _grad(pred[:, 2:3], coords)    # (N, 3)

        # Autograd strain at element ends
        eps_ag_i = (
            x_hat[:, 0] * (grad_ux[i_nd] * x_hat).sum(-1) +
            x_hat[:, 2] * (grad_uz[i_nd] * x_hat).sum(-1)
        )
        eps_ag_j = (
            x_hat[:, 0] * (grad_ux[j_nd] * x_hat).sum(-1) +
            x_hat[:, 2] * (grad_uz[j_nd] * x_hat).sum(-1)
        )
        eps_ag = 0.5 * (eps_ag_i + eps_ag_j)

        # Autograd curvature at element ends
        kappa_ag_i = (grad_phi[i_nd] * x_hat).sum(-1)
        kappa_ag_j = (grad_phi[j_nd] * x_hat).sum(-1)
        kappa_ag = 0.5 * (kappa_ag_i + kappa_ag_j)

        # ══════════════════════════════════════
        # 3. EQUILIBRIUM ASSEMBLY (using FD forces)
        # ══════════════════════════════════════

        idx_i = i_nd.unsqueeze(-1).expand(E_elems, 3)
        idx_j = j_nd.unsqueeze(-1).expand(E_elems, 3)

        # Axial forces
        F_axial = N_fd.unsqueeze(-1) * x_hat
        # Shear forces
        F_shear = V_fd.unsqueeze(-1) * z_hat

        F_int = torch.zeros(N_nodes, 3, device=device)
        F_int.scatter_add_(0, idx_i,  F_axial)
        F_int.scatter_add_(0, idx_j, -F_axial)
        F_int.scatter_add_(0, idx_i,  F_shear)
        F_int.scatter_add_(0, idx_j, -F_shear)

        # Moment assembly
        Mvec_yi = M_yi.unsqueeze(-1) * y_hat
        Mvec_yj = M_yj.unsqueeze(-1) * y_hat

        M_int = torch.zeros(N_nodes, 3, device=device)
        M_int.scatter_add_(0, idx_i, Mvec_yi)
        M_int.scatter_add_(0, idx_j, Mvec_yj)

        # External forces
        elem_load = data.elem_load
        F_ext = torch.zeros(N_nodes, 3, device=device)
        F_ext_per_node = elem_load * L_.unsqueeze(-1) * 0.5
        F_ext.scatter_add_(0, idx_i, F_ext_per_node)
        F_ext.scatter_add_(0, idx_j, F_ext_per_node)

        # ══════════════════════════════════════
        # LOSSES
        # ══════════════════════════════════════

        free_d = (data.bc_disp.squeeze(-1) < 0.5)
        free_r = (data.bc_rot.squeeze(-1) < 0.5)

        F_residual = (F_int + F_ext)[free_d]
        M_residual = M_int[free_r]

        # Characteristic values
        F_char = F_ext[free_d].detach().pow(2).mean().sqrt().clamp(min=1.0)
        q_max = elem_load.detach().abs().max().clamp(min=1.0)
        L_max = L_.detach().max()
        L_total = L_.detach().sum()
        M_char = (q_max * L_max * L_total / 8.0).clamp(min=1.0)

        # ── Force equilibrium ──
        L_force = (F_residual / F_char).pow(2).mean()

        # ── Moment equilibrium ──
        L_moment = (M_residual / M_char).pow(2).mean()

        # ── Neumann: M=0 at pins ──
        bc_d = data.bc_disp.squeeze(-1)
        bc_r = data.bc_rot.squeeze(-1)
        pin_mask = (bc_d > 0.5) & (bc_r < 0.5)

        L_neumann = torch.tensor(0.0, device=device)
        if pin_mask.any():
            M_at_pins = M_int[pin_mask]
            L_neumann = (M_at_pins / M_char).pow(2).mean()

        # ── Kinematic: φ ≈ du_trans/ds ──
        dw_ds = du_trans_fd / L_
        phi_avg = 0.5 * (phi_i + phi_j)
        r_kin = phi_avg - dw_ds
        L_kin = r_kin.pow(2).mean()

        # ══════════════════════════════════════
        # CONSISTENCY: autograd ≈ finite diff
        # ══════════════════════════════════════
        # This teaches the decoder to produce
        # correct spatial variation, not just
        # correct values
        #
        # Without this: decoder outputs flat field
        #   u(x) ≈ constant → du/dx_autograd ≈ 0
        #   but (u_j - u_i)/L ≠ 0
        #
        # With this: decoder forced to make
        #   du/dx_autograd match (u_j - u_i)/L

        L_consist = (
            (eps_ag - eps_fd).pow(2).mean() +
            (kappa_ag - kappa_fd).pow(2).mean()
        )

        # ══════════════════════════════════════
        # TOTAL
        # ══════════════════════════════════════

        total = (self.w_force       * L_force
               + self.w_moment      * L_moment
               + self.w_neumann     * L_neumann
               + self.w_kin         * L_kin
               + self.w_consistency * L_consist)

        loss_dict = {
            'force':       L_force.item(),
            'moment':      L_moment.item(),
            'neumann':     L_neumann.item(),
            'kinematic':   L_kin.item(),
            'consistency': L_consist.item(),
            'total':       total.item(),
            'F_int_max':   F_int.detach().abs().max().item(),
            'F_ext_max':   F_ext.detach().abs().max().item(),
            'F_char':      F_char.item(),
            'M_char':      M_char.item(),
        }

        return total, loss_dict, pred

    @staticmethod
    def _local_axes(x_hat):
        E = x_hat.shape[0]
        device = x_hat.device
        ref = torch.zeros(E, 3, device=device)
        ref[:, 1] = 1.0
        parallel = (x_hat * ref).sum(-1).abs() > 0.99
        ref[parallel] = torch.tensor(
            [0.0, 0.0, 1.0], device=device
        )
        z_hat = torch.cross(x_hat, ref, dim=-1)
        z_hat = z_hat / z_hat.norm(
            dim=-1, keepdim=True).clamp(min=1e-8)
        y_hat = torch.cross(z_hat, x_hat, dim=-1)
        y_hat = y_hat / y_hat.norm(
            dim=-1, keepdim=True).clamp(min=1e-8)
        return y_hat, z_hat