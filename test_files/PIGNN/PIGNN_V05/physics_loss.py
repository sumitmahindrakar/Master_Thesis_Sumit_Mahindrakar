"""
physics_loss.py — Strong-Form Loss via Autodiff (CORRECT)

With the hybrid architecture, autograd.grad gives
TRUE spatial derivatives because the decoder is pointwise.
"""

import torch
import torch.nn as nn


def _grad(y, x):
    """dy/dx via autograd. Works correctly with hybrid architecture."""
    return torch.autograd.grad(
        y.sum(), x,
        create_graph=True,
        retain_graph=True,
    )[0]


class StrongFormPhysicsLoss(nn.Module):

    def __init__(self,
                 w_axial=1.0,
                 w_bending=1e-2,
                 w_kin=0.1):
        super().__init__()
        self.w_axial = w_axial
        self.w_bending = w_bending
        self.w_kin = w_kin

    def forward(self, model, data):
        # ── Forward (two-phase architecture) ──
        pred = model(data)               # (N, 3)
        coords = model.get_coords()      # (N, 3) requires_grad=True
        
        u_x = pred[:, 0:1]
        u_z = pred[:, 1:2]
        phi = pred[:, 2:3]

        EA = (data.prop_E * data.prop_A).mean()
        EI = (data.prop_E * data.prop_I22).mean()

        # External loads (scatter from elements)
        conn = data.connectivity
        N_nodes = pred.shape[0]
        E_elems = conn.shape[0]
        device = pred.device

        elem_load = data.elem_load
        node_load = torch.zeros(N_nodes, 3, device=device)
        count = torch.zeros(N_nodes, 1, device=device)
        idx_i = conn[:, 0].unsqueeze(-1).expand_as(elem_load)
        idx_j = conn[:, 1].unsqueeze(-1).expand_as(elem_load)
        node_load.scatter_add_(0, idx_i, elem_load)
        node_load.scatter_add_(0, idx_j, elem_load)
        ones = torch.ones(E_elems, 1, device=device)
        count.scatter_add_(0, conn[:, 0:1], ones)
        count.scatter_add_(0, conn[:, 1:2], ones)
        node_load = node_load / count.clamp(min=1)

        f_x = node_load[:, 0:1]
        q_z = node_load[:, 2:3]

        # ══════════════════════════════════
        # AUTODIFF — NOW CORRECT!
        # Because decoder is pointwise,
        # .sum() trick gives true spatial derivatives
        # ══════════════════════════════════

        # Axial: EA·d²u_x/dx² + f_x = 0
        dux = _grad(u_x, coords)[:, 0:1]
        d2ux = _grad(dux, coords)[:, 0:1]
        r_axial = EA * d2ux + f_x

        # Bending: EI·d⁴u_z/dx⁴ + q_z = 0
        duz  = _grad(u_z, coords)[:, 0:1]
        d2uz = _grad(duz, coords)[:, 0:1]
        d3uz = _grad(d2uz, coords)[:, 0:1]
        d4uz = _grad(d3uz, coords)[:, 0:1]
        r_bending = EI * d4uz + q_z

        # Kinematic: φ - du_z/dx = 0
        r_kin = phi - duz

        # Only at free nodes
        free_d = (data.bc_disp.squeeze(-1) < 0.5)
        free_r = (data.bc_rot.squeeze(-1) < 0.5)

        L_axial   = r_axial[free_d].pow(2).mean()
        L_bending = r_bending[free_d].pow(2).mean()
        L_kin     = r_kin[free_r].pow(2).mean()

        total = (self.w_axial   * L_axial
               + self.w_bending * L_bending
               + self.w_kin     * L_kin)

        loss_dict = {
            'axial':     L_axial.item(),
            'bending':   L_bending.item(),
            'kinematic': L_kin.item(),
            'total':     total.item(),
        }

        return total, loss_dict, pred