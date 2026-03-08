"""
=================================================================
physics_loss.py — Strong-Form UNSUPERVISED Physics Loss
=================================================================

Governing PDEs (Euler-Bernoulli, statics):

    Axial:      dN/ds + f_s = 0      where  N = EA · du_axial/ds
    Transverse: dV/ds + q   = 0      where  V = EI · d³u_trans/ds³
    Kinematic:  φ - du_trans/ds = 0   (Euler-Bernoulli constraint)

Expanded as derivative residuals:
    L_axial   = || d/ds(EA · du/ds) + f ||²    = || EA · d²u/ds² + f ||²
    L_bending = || d/ds(EI · d³u/ds³) + q ||²  = || EI · d⁴u/ds⁴ + q ||²
    L_kin     = || φ - du_z/ds ||²

No FEM targets. No supervision. Pure physics.

─────────────────────────────────────────────────────────────────
KNOWN LIMITATION (fix later):
    Derivatives are w.r.t. global x-coordinate only.
    Works for horizontal beams. Columns (vertical) need d/dz.
    Full fix: project onto element local axis per element.
    For now: concept exists, we iterate.
=================================================================
"""

import torch
import torch.nn as nn


def _grad(y, x):
    """
    Compute dy/dx via autograd.

    Args:
        y: (N, 1) output field
        x: (N, 3) coordinates with requires_grad=True

    Returns:
        (N, 3): [dy/dx, dy/dy, dy/dz]
    """
    return torch.autograd.grad(
        y.sum(), x,
        create_graph=True,
        retain_graph=True,
    )[0]


class StrongFormPhysicsLoss(nn.Module):
    """
    Unsupervised strong-form loss for 2D frame statics.

    No ground truth. Minimizes PDE residuals only.
    Model forward pass happens INSIDE this loss
    (ensures coords have requires_grad=True).
    """

    def __init__(self,
                 w_axial=1.0,
                 w_bending=1.0,
                 w_kin=0.1):
        super().__init__()
        self.w_axial = w_axial
        self.w_bending = w_bending
        self.w_kin = w_kin

    def forward(self, model, data):
        """
        Args:
            model: PIGNN instance (forward pass done here)
            data:  PyG Data/Batch

        Returns:
            total_loss:  scalar (differentiable)
            loss_dict:   {str: float} for logging
        """
        # ── Forward pass (coords get requires_grad inside model) ──
        pred = model(data)               # (N, 3): [u_x, u_z, φ]
        coords = model.get_coords()      # (N, 3) with requires_grad

        # ── Fields as (N, 1) for autograd ──
        u_x = pred[:, 0:1]
        u_z = pred[:, 1:2]
        phi = pred[:, 2:3]

        # ── Material properties (from data, not arguments) ──
        # Average to scalar — works when properties are uniform per case
        # For varying properties: scatter to nodes (future improvement)
        EA = (data.prop_E * data.prop_A).mean()
        EI = (data.prop_E * data.prop_I22).mean()

        # ── External load: scatter element UDL to nodes ──
        conn = data.connectivity               # (E, 2)
        N_nodes = pred.shape[0]
        E_elems = conn.shape[0]
        device = pred.device

        # Each node gets average of connected element loads
        node_load = torch.zeros(N_nodes, 3, device=device)
        count = torch.zeros(N_nodes, 1, device=device)

        idx_i = conn[:, 0].unsqueeze(-1).expand_as(data.elem_load)
        idx_j = conn[:, 1].unsqueeze(-1).expand_as(data.elem_load)
        node_load.scatter_add_(0, idx_i, data.elem_load)
        node_load.scatter_add_(0, idx_j, data.elem_load)

        ones = torch.ones(E_elems, 1, device=device)
        count.scatter_add_(0, conn[:, 0:1], ones)
        count.scatter_add_(0, conn[:, 1:2], ones)
        node_load = node_load / count.clamp(min=1)

        # ── External loads (from data, already per-node) ──
        f_x = data.x[:, 5:6]    # wl_x (axial distributed load)
        q_z = data.x[:, 7:8]    # wl_z (transverse distributed load)

        # ════════════════════════════════════════════
        # AXIAL:  EA · d²u_x/dx² + f_x = 0
        # ════════════════════════════════════════════
        dux_dx = _grad(u_x, coords)[:, 0:1]          # du_x/dx
        d2ux_dx2 = _grad(dux_dx, coords)[:, 0:1]     # d²u_x/dx²

        r_axial = EA * d2ux_dx2 + f_x                # should → 0

        # ════════════════════════════════════════════
        # BENDING:  EI · d⁴u_z/dx⁴ + q_z = 0
        # ════════════════════════════════════════════
        duz_dx   = _grad(u_z, coords)[:, 0:1]        # du_z/dx
        d2uz_dx2 = _grad(duz_dx, coords)[:, 0:1]     # d²u_z/dx²
        d3uz_dx3 = _grad(d2uz_dx2, coords)[:, 0:1]   # d³u_z/dx³
        d4uz_dx4 = _grad(d3uz_dx3, coords)[:, 0:1]   # d⁴u_z/dx⁴

        r_bending = EI * d4uz_dx4 + q_z              # should → 0

        # ════════════════════════════════════════════
        # KINEMATIC:  φ - du_z/dx = 0
        # (Euler-Bernoulli: rotation = slope)
        # ════════════════════════════════════════════
        r_kin = phi - duz_dx                          # should → 0

        # ════════════════════════════════════════════
        # LOSS: only at FREE nodes (BCs enforced in model)
        # ════════════════════════════════════════════
        free_disp = (data.bc_disp.squeeze(-1) < 0.5)
        free_rot  = (data.bc_rot.squeeze(-1) < 0.5)

        L_axial   = r_axial[free_disp].pow(2).mean()
        L_bending = r_bending[free_disp].pow(2).mean()
        L_kin     = r_kin[free_rot].pow(2).mean()

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