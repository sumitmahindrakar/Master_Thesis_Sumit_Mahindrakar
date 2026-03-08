"""
physics_loss.py — Autograd-Based Strong-Form Physics Loss
(User's proposed approach — to observe the problems)

Predict:  y_i = (u_x, u_z, φ)  at each node
Autograd: du/ds, d²u/ds², d³u/ds³ via torch.autograd.grad
Derive:   N = EA·du_s/ds,  M = EI·d²u_n/ds²,  V = EI·d³u_n/ds³
Loss:     ||F_int - F_ext|| = 0  at free nodes
"""

import torch
import torch.nn as nn


# ============================================================
# AUTOGRAD HELPERS
# ============================================================

def compute_grad(y, x):
    """
    ∇y w.r.t. x via autograd.

    Args:
        y: (N, 1) scalar field
        x: (N, 3) coordinates with requires_grad=True

    Returns:
        (N, 3): [dy/dx, dy/dy, dy/dz]
    """
    return torch.autograd.grad(
        y.sum(), x,
        create_graph=True,
        retain_graph=True,
    )[0]


def directional_deriv(y, coords, t):
    """
    dy/ds = t · ∇y  (directional derivative along tangent)

    Args:
        y:      (N, 1) field
        coords: (N, 3) requires_grad=True
        t:      (N, 3) unit tangent vector

    Returns:
        (N, 1)
    """
    gy = compute_grad(y, coords)           # (N, 3)
    return (gy * t).sum(dim=1, keepdim=True)  # (N, 1)


# ============================================================
# PHYSICS LOSS (AUTOGRAD VERSION)
# ============================================================

class StrongFormPhysicsLoss(nn.Module):
    """
    Autograd-based strong-form loss.

    User's proposed approach:
      1. Predict (u_x, u_z, φ) at nodes
      2. Transform to local (u_s, u_n) using node tangent
      3. Autograd: N = EA·du_s/ds, M = EI·d²u_n/ds², V = EI·d³u_n/ds³
      4. Equilibrium: dN/ds + q_s = 0, dV/ds + q_n = 0

    KNOWN PROBLEMS (will observe):
      - Joint nodes: tangent is averaged, wrong for both elements
      - GNN Jacobian ≠ spatial derivative
      - High-order derivatives through GNN are noisy
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
        """
        Args:
            model: PIGNN (must have get_coords() method)
            data:  PyG Data/Batch

        Returns:
            total_loss, loss_dict, pred
        """

        # ════════════════════════════════════════════════
        # 1. FORWARD PASS (coords in autograd graph)
        # ════════════════════════════════════════════════
        pred = model(data)             # (N, 3): [u_x, u_z, φ]
        coords = model.get_coords()   # (N, 3) with requires_grad

        u_x = pred[:, 0:1]    # (N, 1)
        u_z = pred[:, 1:2]    # (N, 1)
        phi = pred[:, 2:3]    # (N, 1)

        N_nodes = coords.shape[0]
        device = coords.device
        conn = data.connectivity          # (E, 2)
        E_elems = conn.shape[0]

        # ════════════════════════════════════════════════
        # 2. BUILD NODE-WISE TANGENT VECTOR
        #    (average of connected element directions)
        # ════════════════════════════════════════════════
        elem_dir = data.elem_directions   # (E, 3)
        t_elem = elem_dir / elem_dir.norm(
            dim=1, keepdim=True).clamp(min=1e-12)

        t_node = torch.zeros(N_nodes, 3, device=device)
        count = torch.zeros(N_nodes, 1, device=device)
        ones = torch.ones(E_elems, 1, device=device)

        t_node.scatter_add_(
            0, conn[:, 0:1].expand(-1, 3), t_elem)
        t_node.scatter_add_(
            0, conn[:, 1:2].expand(-1, 3), t_elem)
        count.scatter_add_(0, conn[:, 0:1], ones)
        count.scatter_add_(0, conn[:, 1:2], ones)

        t_node = t_node / count.clamp(min=1.0)
        t_node = t_node / t_node.norm(
            dim=1, keepdim=True).clamp(min=1e-12)

        # ════════════════════════════════════════════════
        # 3. LOCAL DISPLACEMENTS
        #    u_s =  u_x·cos(θ) + u_z·sin(θ)   (axial)
        #    u_n = -u_x·sin(θ) + u_z·cos(θ)   (transverse)
        # ════════════════════════════════════════════════
        cos_t = t_node[:, 0:1]   # (N, 1)
        sin_t = t_node[:, 2:3]   # (N, 1)

        u_s =  u_x * cos_t + u_z * sin_t     # (N, 1)
        u_n = -u_x * sin_t + u_z * cos_t     # (N, 1)

        # ════════════════════════════════════════════════
        # 4. MATERIAL PROPERTIES (scatter to nodes)
        # ════════════════════════════════════════════════
        EA_elem = (data.prop_E * data.prop_A).unsqueeze(-1)    # (E, 1)
        EI_elem = (data.prop_E * data.prop_I22).unsqueeze(-1)  # (E, 1)

        EA_node = torch.zeros(N_nodes, 1, device=device)
        EI_node = torch.zeros(N_nodes, 1, device=device)

        EA_node.scatter_add_(0, conn[:, 0:1], EA_elem)
        EA_node.scatter_add_(0, conn[:, 1:2], EA_elem)
        EI_node.scatter_add_(0, conn[:, 0:1], EI_elem)
        EI_node.scatter_add_(0, conn[:, 1:2], EI_elem)

        EA_node = EA_node / count.clamp(min=1)   # (N, 1)
        EI_node = EI_node / count.clamp(min=1)   # (N, 1)

        # ════════════════════════════════════════════════
        # 5. LOCAL LOADS (scatter to nodes)
        # ════════════════════════════════════════════════
        q_global = data.elem_load   # (E, 3)
        q_s_e = (q_global[:, 0:1] * t_elem[:, 0:1] +
                 q_global[:, 2:3] * t_elem[:, 2:3])   # (E, 1)
        q_n_e = (-q_global[:, 0:1] * t_elem[:, 2:3] +
                  q_global[:, 2:3] * t_elem[:, 0:1])  # (E, 1)

        q_s = torch.zeros(N_nodes, 1, device=device)
        q_n = torch.zeros(N_nodes, 1, device=device)
        q_s.scatter_add_(0, conn[:, 0:1], q_s_e)
        q_s.scatter_add_(0, conn[:, 1:2], q_s_e)
        q_n.scatter_add_(0, conn[:, 0:1], q_n_e)
        q_n.scatter_add_(0, conn[:, 1:2], q_n_e)
        q_s = q_s / count.clamp(min=1)
        q_n = q_n / count.clamp(min=1)

        # ════════════════════════════════════════════════
        # 6. INTERNAL FORCES VIA AUTOGRAD
        #
        #    N = EA · du_s/ds
        #    M = EI · d²u_n/ds²
        #    V = EI · d³u_n/ds³
        # ════════════════════════════════════════════════

        # --- Axial force ---
        du_s_ds = directional_deriv(u_s, coords, t_node)   # (N, 1)
        N_int = EA_node * du_s_ds                           # (N, 1)

        # --- Bending moment ---
        du_n_ds   = directional_deriv(u_n, coords, t_node)   # (N, 1)
        d2u_n_ds2 = directional_deriv(du_n_ds, coords, t_node)  # (N, 1)
        M_int = EI_node * d2u_n_ds2                           # (N, 1)

        # --- Shear force ---
        d3u_n_ds3 = directional_deriv(d2u_n_ds2, coords, t_node)  # (N, 1)
        V_int = EI_node * d3u_n_ds3                               # (N, 1)

        # ════════════════════════════════════════════════
        # 7. EQUILIBRIUM RESIDUALS
        #
        #    dN/ds + q_s = 0   (axial equilibrium)
        #    dV/ds + q_n = 0   (transverse equilibrium)
        # ════════════════════════════════════════════════

        # --- Axial: dN/ds + q_s = 0 ---
        dN_ds = directional_deriv(N_int, coords, t_node)  # (N, 1)
        r_axial = dN_ds + q_s

        # --- Transverse: dV/ds + q_n = 0 ---
        dV_ds = directional_deriv(V_int, coords, t_node)  # (N, 1)
        r_bending = dV_ds + q_n

        # ════════════════════════════════════════════════
        # 8. KINEMATIC RESIDUAL (Euler-Bernoulli)
        #    φ = du_n/ds
        # ════════════════════════════════════════════════
        r_kin = phi - du_n_ds

        # ════════════════════════════════════════════════
        # 9. LOSS (free nodes only)
        # ════════════════════════════════════════════════
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