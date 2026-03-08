# """
# =================================================================
# physics_loss.py — Element-Local Strong-Form + Neumann BCs
# =================================================================

# BCs enforced:
#     Dirichlet (HARD, in model):
#         u_x = 0, u_z = 0  at supports  (bc_disp = 1)
#         φ = 0              at fixed     (bc_rot = 1)
    
#     Neumann (SOFT, in loss):
#         M = 0  at pin supports  (bc_disp=1 AND bc_rot=0)
#         → Detected automatically, no extra flags needed
# =================================================================
# """

# import torch
# import torch.nn as nn


# def _grad(y, x):
#     """Full spatial gradient dy/d(x,y,z)."""
#     return torch.autograd.grad(
#         y.sum(), x,
#         create_graph=True,
#         retain_graph=True,
#     )[0]


# class StrongFormPhysicsLoss(nn.Module):

#     def __init__(self,
#                  w_force=1.0,
#                  w_moment=1.0,
#                  w_kin=0.1,
#                  w_neumann=1.0):       # ← NEW weight
#         super().__init__()
#         self.w_force   = w_force
#         self.w_moment  = w_moment
#         self.w_kin     = w_kin
#         self.w_neumann = w_neumann

#     def forward(self, model, data):
#         pred = model(data)
#         coords = model.get_coords()

#         N_nodes = pred.shape[0]
#         device  = pred.device

#         u_x = pred[:, 0:1]
#         u_z = pred[:, 1:2]
#         phi = pred[:, 2:3]

#         conn  = data.connectivity
#         E_mod = data.prop_E
#         A_    = data.prop_A
#         I22   = data.prop_I22
#         L_    = data.elem_lengths
#         x_hat = data.elem_directions

#         E_elems = conn.shape[0]
#         i_nd = conn[:, 0]
#         j_nd = conn[:, 1]

#         EA = E_mod * A_
#         EI = E_mod * I22

#         y_hat, z_hat = self._local_axes(x_hat)

#         # ══════════════════════════════════
#         # AUTOGRAD
#         # ══════════════════════════════════
#         grad_ux  = _grad(u_x, coords)
#         grad_uz  = _grad(u_z, coords)
#         grad_phi = _grad(phi, coords)

#         # ══════════════════════════════════
#         # ELEMENT-LOCAL: N, M, V
#         # ══════════════════════════════════

#         eps_i = (
#             x_hat[:, 0] * (grad_ux[i_nd] * x_hat).sum(-1) +
#             x_hat[:, 2] * (grad_uz[i_nd] * x_hat).sum(-1)
#         )
#         eps_j = (
#             x_hat[:, 0] * (grad_ux[j_nd] * x_hat).sum(-1) +
#             x_hat[:, 2] * (grad_uz[j_nd] * x_hat).sum(-1)
#         )

#         kappa_i = (grad_phi[i_nd] * x_hat).sum(-1)
#         kappa_j = (grad_phi[j_nd] * x_hat).sum(-1)

#         N_avg    = 0.5 * EA * (eps_i + eps_j)
#         M_i_elem = EI * kappa_i
#         M_j_elem = EI * kappa_j
#         V_elem   = (M_j_elem - M_i_elem) / L_

#         # ══════════════════════════════════
#         # EQUILIBRIUM ASSEMBLY
#         # ══════════════════════════════════

#         idx_i = i_nd.unsqueeze(-1).expand(E_elems, 3)
#         idx_j = j_nd.unsqueeze(-1).expand(E_elems, 3)

#         F_axial = N_avg.unsqueeze(-1) * x_hat
#         F_shear = V_elem.unsqueeze(-1) * z_hat

#         F_int = torch.zeros(N_nodes, 3, device=device)
#         F_int.scatter_add_(0, idx_i,  F_axial)
#         F_int.scatter_add_(0, idx_j, -F_axial)
#         F_int.scatter_add_(0, idx_i,  F_shear)
#         F_int.scatter_add_(0, idx_j, -F_shear)

#         Mvec_i = M_i_elem.unsqueeze(-1) * y_hat
#         Mvec_j = M_j_elem.unsqueeze(-1) * y_hat

#         M_int = torch.zeros(N_nodes, 3, device=device)
#         M_int.scatter_add_(0, idx_i, Mvec_i)
#         M_int.scatter_add_(0, idx_j, Mvec_j)

#         elem_load = data.elem_load
#         F_ext = torch.zeros(N_nodes, 3, device=device)
#         F_ext_per_node = elem_load * L_.unsqueeze(-1) * 0.5
#         F_ext.scatter_add_(0, idx_i, F_ext_per_node)
#         F_ext.scatter_add_(0, idx_j, F_ext_per_node)

#         # ══════════════════════════════════
#         # LOSS 1: Equilibrium at FREE nodes
#         # ══════════════════════════════════

#         free_d = (data.bc_disp.squeeze(-1) < 0.5)
#         free_r = (data.bc_rot.squeeze(-1) < 0.5)

#         F_residual = (F_int + F_ext)[free_d]
#         M_residual = M_int[free_r]

#         EA_char   = EA.detach().mean().clamp(min=1.0)
#         EI_L_char = (EI / L_).detach().mean().clamp(min=1.0)

#         L_force  = (F_residual / EA_char).pow(2).mean()
#         L_moment = (M_residual / EI_L_char).pow(2).mean()

#         # ══════════════════════════════════════════════
#         # LOSS 2: Neumann BCs — M=0 at PIN supports
#         # ══════════════════════════════════════════════
#         # Pin support = displacement fixed, rotation FREE
#         # Physically: hinge, no moment resistance
#         # Condition:  total moment at this node = 0
#         #
#         # Detection: bc_disp=1 AND bc_rot=0
#         #   → uses your EXISTING flags, no data changes!
#         #
#         # M_int already has the assembled moment at every node
#         # from all connected elements. At a pin, this must = 0.
#         # ══════════════════════════════════════════════

#         bc_d = data.bc_disp.squeeze(-1)    # (N,)
#         bc_r = data.bc_rot.squeeze(-1)     # (N,)

#         pin_mask = (bc_d > 0.5) & (bc_r < 0.5)

#         L_neumann = torch.tensor(0.0, device=device)

#         if pin_mask.any():
#             M_at_pins = M_int[pin_mask]    # (n_pins, 3)
#             L_neumann = (M_at_pins / EI_L_char).pow(2).mean()

#         # ══════════════════════════════════
#         # LOSS 3: Kinematic (Euler-Bernoulli)
#         # ══════════════════════════════════

#         du_trans_i = (
#             z_hat[:, 0] * (grad_ux[i_nd] * x_hat).sum(-1) +
#             z_hat[:, 2] * (grad_uz[i_nd] * x_hat).sum(-1)
#         )
#         du_trans_j = (
#             z_hat[:, 0] * (grad_ux[j_nd] * x_hat).sum(-1) +
#             z_hat[:, 2] * (grad_uz[j_nd] * x_hat).sum(-1)
#         )

#         r_kin_i = phi[i_nd].squeeze() - du_trans_i
#         r_kin_j = phi[j_nd].squeeze() - du_trans_j
#         L_kin = 0.5 * (
#             r_kin_i.pow(2).mean() +
#             r_kin_j.pow(2).mean()
#         )

#         # ══════════════════════════════════
#         # TOTAL
#         # ══════════════════════════════════

#         total = (self.w_force   * L_force
#                + self.w_moment  * L_moment
#                + self.w_neumann * L_neumann
#                + self.w_kin     * L_kin)

#         loss_dict = {
#             'force':     L_force.item(),
#             'moment':    L_moment.item(),
#             'neumann':   L_neumann.item(),
#             'kinematic': L_kin.item(),
#             'total':     total.item(),
#         }

#         return total, loss_dict, pred

#     @staticmethod
#     def _local_axes(x_hat):
#         E = x_hat.shape[0]
#         device = x_hat.device
#         ref = torch.zeros(E, 3, device=device)
#         ref[:, 1] = 1.0
#         parallel = (x_hat * ref).sum(-1).abs() > 0.99
#         ref[parallel] = torch.tensor(
#             [0.0, 0.0, 1.0], device=device
#         )
#         z_hat = torch.cross(x_hat, ref, dim=-1)
#         z_hat = z_hat / z_hat.norm(
#             dim=-1, keepdim=True
#         ).clamp(min=1e-8)
#         y_hat = torch.cross(z_hat, x_hat, dim=-1)
#         y_hat = y_hat / y_hat.norm(
#             dim=-1, keepdim=True
#         ).clamp(min=1e-8)
#         return y_hat, z_hat

"""
physics_loss.py — Fixed Normalization
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

    def __init__(self,
                 w_force=1.0,
                 w_moment=1.0,
                 w_kin=0.1,
                 w_neumann=1.0):
        super().__init__()
        self.w_force   = w_force
        self.w_moment  = w_moment
        self.w_kin     = w_kin
        self.w_neumann = w_neumann

    def forward(self, model, data):
        pred = model(data)
        coords = model.get_coords()

        N_nodes = pred.shape[0]
        device  = pred.device

        u_x = pred[:, 0:1]
        u_z = pred[:, 1:2]
        phi = pred[:, 2:3]

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

        # ══════════════════════════════════
        # AUTOGRAD
        # ══════════════════════════════════
        grad_ux  = _grad(u_x, coords)
        grad_uz  = _grad(u_z, coords)
        grad_phi = _grad(phi, coords)

        # ══════════════════════════════════
        # ELEMENT-LOCAL: N, M, V
        # ══════════════════════════════════
        eps_i = (
            x_hat[:, 0] * (grad_ux[i_nd] * x_hat).sum(-1) +
            x_hat[:, 2] * (grad_uz[i_nd] * x_hat).sum(-1)
        )
        eps_j = (
            x_hat[:, 0] * (grad_ux[j_nd] * x_hat).sum(-1) +
            x_hat[:, 2] * (grad_uz[j_nd] * x_hat).sum(-1)
        )

        kappa_i = (grad_phi[i_nd] * x_hat).sum(-1)
        kappa_j = (grad_phi[j_nd] * x_hat).sum(-1)

        N_avg    = 0.5 * EA * (eps_i + eps_j)
        M_i_elem = EI * kappa_i
        M_j_elem = EI * kappa_j
        V_elem   = (M_j_elem - M_i_elem) / L_

        # ══════════════════════════════════
        # EQUILIBRIUM ASSEMBLY
        # ══════════════════════════════════
        idx_i = i_nd.unsqueeze(-1).expand(E_elems, 3)
        idx_j = j_nd.unsqueeze(-1).expand(E_elems, 3)

        F_axial = N_avg.unsqueeze(-1) * x_hat
        F_shear = V_elem.unsqueeze(-1) * z_hat

        F_int = torch.zeros(N_nodes, 3, device=device)
        F_int.scatter_add_(0, idx_i,  F_axial)
        F_int.scatter_add_(0, idx_j, -F_axial)
        F_int.scatter_add_(0, idx_i,  F_shear)
        F_int.scatter_add_(0, idx_j, -F_shear)

        Mvec_i = M_i_elem.unsqueeze(-1) * y_hat
        Mvec_j = M_j_elem.unsqueeze(-1) * y_hat

        M_int = torch.zeros(N_nodes, 3, device=device)
        M_int.scatter_add_(0, idx_i, Mvec_i)
        M_int.scatter_add_(0, idx_j, Mvec_j)

        elem_load = data.elem_load
        F_ext = torch.zeros(N_nodes, 3, device=device)
        F_ext_per_node = elem_load * L_.unsqueeze(-1) * 0.5
        F_ext.scatter_add_(0, idx_i, F_ext_per_node)
        F_ext.scatter_add_(0, idx_j, F_ext_per_node)

        # ══════════════════════════════════════════════
        # LOSS 1: Equilibrium at FREE nodes
        # ══════════════════════════════════════════════
        # 
        # FIX: Normalize by actual external load magnitude
        # not by EA (which is ~10⁹ and hides everything)
        # ══════════════════════════════════════════════

        free_d = (data.bc_disp.squeeze(-1) < 0.5)
        free_r = (data.bc_rot.squeeze(-1) < 0.5)

        F_residual = (F_int + F_ext)[free_d]
        M_residual = M_int[free_r]

        # ── Characteristic force = RMS of external load ──
        # This represents "how much force SHOULD be balanced"
        F_ext_free = F_ext[free_d]
        F_char = F_ext_free.detach().pow(2).mean().sqrt().clamp(min=1.0)

        # ── Characteristic moment = q × L² / 8 ──
        # Approximate max moment in a beam under UDL
        q_max = elem_load.detach().abs().max().clamp(min=1.0)
        L_max = L_.detach().max()
        L_total = L_.detach().sum()  # rough span estimate
        M_char = (q_max * L_max * L_total / 8.0).clamp(min=1.0)

        L_force  = (F_residual / F_char).pow(2).mean()
        L_moment = (M_residual / M_char).pow(2).mean()

        # ══════════════════════════════════════════════
        # LOSS 2: Neumann BCs — M=0 at PIN supports
        # ══════════════════════════════════════════════

        bc_d = data.bc_disp.squeeze(-1)
        bc_r = data.bc_rot.squeeze(-1)
        pin_mask = (bc_d > 0.5) & (bc_r < 0.5)

        L_neumann = torch.tensor(0.0, device=device)
        if pin_mask.any():
            M_at_pins = M_int[pin_mask]
            L_neumann = (M_at_pins / M_char).pow(2).mean()

        # ══════════════════════════════════
        # LOSS 3: Kinematic (Euler-Bernoulli)
        # ══════════════════════════════════

        du_trans_i = (
            z_hat[:, 0] * (grad_ux[i_nd] * x_hat).sum(-1) +
            z_hat[:, 2] * (grad_uz[i_nd] * x_hat).sum(-1)
        )
        du_trans_j = (
            z_hat[:, 0] * (grad_ux[j_nd] * x_hat).sum(-1) +
            z_hat[:, 2] * (grad_uz[j_nd] * x_hat).sum(-1)
        )

        r_kin_i = phi[i_nd].squeeze() - du_trans_i
        r_kin_j = phi[j_nd].squeeze() - du_trans_j
        L_kin = 0.5 * (
            r_kin_i.pow(2).mean() +
            r_kin_j.pow(2).mean()
        )

        # ══════════════════════════════════
        # TOTAL
        # ══════════════════════════════════

        total = (self.w_force   * L_force
               + self.w_moment  * L_moment
               + self.w_neumann * L_neumann
               + self.w_kin     * L_kin)

        loss_dict = {
            'force':     L_force.item(),
            'moment':    L_moment.item(),
            'neumann':   L_neumann.item(),
            'kinematic': L_kin.item(),
            'total':     total.item(),
            # ── Debug info ──
            'F_int_max':  F_int.detach().abs().max().item(),
            'F_ext_max':  F_ext.detach().abs().max().item(),
            'F_char':     F_char.item(),
            'M_char':     M_char.item(),
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
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
        y_hat = torch.cross(z_hat, x_hat, dim=-1)
        y_hat = y_hat / y_hat.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
        return y_hat, z_hat