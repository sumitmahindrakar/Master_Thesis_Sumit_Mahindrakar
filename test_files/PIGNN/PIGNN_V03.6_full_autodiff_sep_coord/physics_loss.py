"""
physics_loss.py — Autograd physics loss for separated-coord model

Now coords are (N, 2): [x, z] instead of (N, 3): [x, y, z]
Directional derivatives adapted accordingly.
"""

import torch
import torch.nn as nn


def compute_grad(y, x):
    """∇y w.r.t. x via autograd."""
    return torch.autograd.grad(
        y.sum(), x,
        create_graph=True,
        retain_graph=True,
    )[0]


def directional_deriv(y, coords_2d, t_2d):
    """
    dy/ds = t · ∇y  (2D version)

    Args:
        y:          (N, 1)
        coords_2d:  (N, 2) [x, z] with requires_grad
        t_2d:       (N, 2) unit tangent [cos θ, sin θ]

    Returns:
        (N, 1)
    """
    gy = compute_grad(y, coords_2d)          # (N, 2): [∂y/∂x, ∂y/∂z]
    return (gy * t_2d).sum(dim=1, keepdim=True)


class StrongFormPhysicsLoss(nn.Module):
    """
    Autograd loss with separated coordinates.
    Coords are (N, 2): [x, z].
    """

    def __init__(self,
                 w_axial=1.0,
                 w_bending=1.0,
                 w_shear=1.0,
                 w_kinematic=0.1):
        super().__init__()
        self.w_axial = w_axial
        self.w_bending = w_bending
        self.w_shear = w_shear
        self.w_kin = w_kinematic

    def forward(self, model, data):

        # ── Forward ──
        pred = model(data)
        coords_2d = model.get_coords()     # (N, 2): [x, z]

        u_x = pred[:, 0:1]
        u_z = pred[:, 1:2]
        phi = pred[:, 2:3]

        N_nodes = coords_2d.shape[0]
        device = coords_2d.device
        conn = data.connectivity
        E_elems = conn.shape[0]

        # ── Node tangent (2D) ──
        elem_dir = data.elem_directions           # (E, 3)
        t_elem_2d = torch.stack([
            elem_dir[:, 0], elem_dir[:, 2]
        ], dim=1)                                  # (E, 2): [dx, dz]
        t_elem_2d = t_elem_2d / t_elem_2d.norm(
            dim=1, keepdim=True).clamp(min=1e-12)

        t_node_2d = torch.zeros(N_nodes, 2, device=device)
        count = torch.zeros(N_nodes, 1, device=device)
        ones = torch.ones(E_elems, 1, device=device)

        t_node_2d.scatter_add_(
            0, conn[:, 0:1].expand(-1, 2), t_elem_2d)
        t_node_2d.scatter_add_(
            0, conn[:, 1:2].expand(-1, 2), t_elem_2d)
        count.scatter_add_(0, conn[:, 0:1], ones)
        count.scatter_add_(0, conn[:, 1:2], ones)

        t_node_2d = t_node_2d / count.clamp(min=1.0)
        t_node_2d = t_node_2d / t_node_2d.norm(
            dim=1, keepdim=True).clamp(min=1e-12)

        # ── Local displacements ──
        cos_t = t_node_2d[:, 0:1]
        sin_t = t_node_2d[:, 1:2]

        u_s =  u_x * cos_t + u_z * sin_t
        u_n = -u_x * sin_t + u_z * cos_t

        # ── Material (scatter to nodes) ──
        EA_elem = (data.prop_E * data.prop_A).unsqueeze(-1)
        EI_elem = (data.prop_E * data.prop_I22).unsqueeze(-1)

        EA_node = torch.zeros(N_nodes, 1, device=device)
        EI_node = torch.zeros(N_nodes, 1, device=device)
        EA_node.scatter_add_(0, conn[:, 0:1], EA_elem)
        EA_node.scatter_add_(0, conn[:, 1:2], EA_elem)
        EI_node.scatter_add_(0, conn[:, 0:1], EI_elem)
        EI_node.scatter_add_(0, conn[:, 1:2], EI_elem)
        EA_node = EA_node / count.clamp(min=1)
        EI_node = EI_node / count.clamp(min=1)

        # ── Local loads ──
        q_global = data.elem_load
        q_s_e = (q_global[:, 0:1] * t_elem_2d[:, 0:1] +
                 q_global[:, 2:3] * t_elem_2d[:, 1:2])
        q_n_e = (-q_global[:, 0:1] * t_elem_2d[:, 1:2] +
                  q_global[:, 2:3] * t_elem_2d[:, 0:1])

        q_s = torch.zeros(N_nodes, 1, device=device)
        q_n = torch.zeros(N_nodes, 1, device=device)
        q_s.scatter_add_(0, conn[:, 0:1], q_s_e)
        q_s.scatter_add_(0, conn[:, 1:2], q_s_e)
        q_n.scatter_add_(0, conn[:, 0:1], q_n_e)
        q_n.scatter_add_(0, conn[:, 1:2], q_n_e)
        q_s = q_s / count.clamp(min=1)
        q_n = q_n / count.clamp(min=1)

        # ── Internal forces via autograd ──
        du_s_ds = directional_deriv(u_s, coords_2d, t_node_2d)
        N_int = EA_node * du_s_ds

        du_n_ds   = directional_deriv(u_n, coords_2d, t_node_2d)
        d2u_n_ds2 = directional_deriv(du_n_ds, coords_2d, t_node_2d)
        M_int = EI_node * d2u_n_ds2

        d3u_n_ds3 = directional_deriv(d2u_n_ds2, coords_2d, t_node_2d)
        V_int = EI_node * d3u_n_ds3

        # ── Equilibrium residuals ──
        dN_ds = directional_deriv(N_int, coords_2d, t_node_2d)
        r_axial = dN_ds + q_s

        dV_ds = directional_deriv(V_int, coords_2d, t_node_2d)
        r_bending = dV_ds + q_n

        r_kin = phi - du_n_ds

        # ── Loss (free nodes) ──
        free_disp = (data.bc_disp.squeeze(-1) < 0.5)
        free_rot  = (data.bc_rot.squeeze(-1) < 0.5)

        L_axial   = r_axial[free_disp].pow(2).mean()
        L_bending = r_bending[free_disp].pow(2).mean()
        L_kin     = r_kin[free_rot].pow(2).mean()

        total = (self.w_axial   * L_axial +
                 self.w_bending * L_bending +
                 self.w_kin     * L_kin)

        loss_dict = {
            'axial':     L_axial.item(),
            'bending':   L_bending.item(),
            'kinematic': L_kin.item(),
            'total':     total.item(),
        }

        return total, loss_dict, pred