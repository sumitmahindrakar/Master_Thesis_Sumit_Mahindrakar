"""
physics_loss.py — Element-Wise Strong-Form Physics Loss
Euler-Bernoulli 2D frame (beams + columns)

No supervision. No FEM targets. No autograd w.r.t. coordinates.
Pure algebraic operations using EB stiffness matrices.

For each element (LOCAL coords):
  Axial:   K_axial · [u_s_i, u_s_j] - f_axial = [F_s_i, F_s_j]
  Bending: K_bend  · [u_n_i, φ_i, u_n_j, φ_j] - f_bend = [F_n_i, M_i, F_n_j, M_j]

For each free node (GLOBAL coords):
  ΣF_x = 0,  ΣF_z = 0,  ΣM_y = 0

Loss = ||equilibrium residual||² + w_kin · ||kinematic residual||²

Why this works:
  - Each element uses its OWN local coordinate system
  - Joints handled by global force balance (physically correct)
  - EB kinematics built into Hermite shape functions
  - No autograd, no tangent averaging, no high-order derivatives
  - Exact for 2-node EB beam elements (same as FEM)
"""

import torch
import torch.nn as nn


class StrongFormPhysicsLoss(nn.Module):
    """
    Element-wise physics loss for 2D EB frames.

    Computes internal forces from predicted displacements
    using exact EB stiffness matrices, then checks equilibrium.
    """

    def __init__(self,
                 w_equilibrium=1.0,
                 w_kinematic=0.1):
        super().__init__()
        self.w_eq = w_equilibrium
        self.w_kin = w_kinematic

    def forward(self, pred, data):
        """
        Args:
            pred: (N_total, 3) predicted [u_x, u_z, φ]
            data: PyG Data/Batch

        Returns:
            total_loss:  scalar (differentiable)
            loss_dict:   {str: float} for logging
        """

        conn = data.connectivity       # (E, 2)
        L    = data.elem_lengths       # (E,)
        dirs = data.elem_directions    # (E, 3)

        E_total = conn.shape[0]
        N_total = pred.shape[0]
        device  = pred.device

        # ════════════════════════════════════════════════
        # 1. ELEMENT LOCAL FRAME
        #    s-axis: along element   → (cos θ, 0, sin θ)
        #    n-axis: perpendicular   → (-sin θ, 0, cos θ)
        # ════════════════════════════════════════════════
        cos_t = dirs[:, 0]     # (E,)
        sin_t = dirs[:, 2]     # (E,)

        EA = data.prop_E * data.prop_A       # (E,)
        EI = data.prop_E * data.prop_I22     # (E,)

        # ════════════════════════════════════════════════
        # 2. EXTRACT GLOBAL DOFs AT ELEMENT ENDPOINTS
        # ════════════════════════════════════════════════
        node_i = conn[:, 0]    # (E,)
        node_j = conn[:, 1]    # (E,)

        u_x_i = pred[node_i, 0]   # (E,)
        u_z_i = pred[node_i, 1]
        phi_i = pred[node_i, 2]

        u_x_j = pred[node_j, 0]
        u_z_j = pred[node_j, 1]
        phi_j = pred[node_j, 2]

        # ════════════════════════════════════════════════
        # 3. TRANSFORM TO LOCAL COORDINATES
        #    u_s =  u_x·cos + u_z·sin   (axial)
        #    u_n = -u_x·sin + u_z·cos   (transverse)
        #    φ stays same (rotation about y)
        # ════════════════════════════════════════════════
        u_s_i =  u_x_i * cos_t + u_z_i * sin_t
        u_n_i = -u_x_i * sin_t + u_z_i * cos_t

        u_s_j =  u_x_j * cos_t + u_z_j * sin_t
        u_n_j = -u_x_j * sin_t + u_z_j * cos_t

        # ════════════════════════════════════════════════
        # 4. LOCAL DISTRIBUTED LOADS
        #    Project global UDL to local axes
        # ════════════════════════════════════════════════
        q_global = data.elem_load     # (E, 3): [qx, qy, qz]
        q_s =  q_global[:, 0] * cos_t + q_global[:, 2] * sin_t
        q_n = -q_global[:, 0] * sin_t + q_global[:, 2] * cos_t

        # ════════════════════════════════════════════════
        # 5. AXIAL END FORCES (local)
        #
        #    K_axial = (EA/L) · [[ 1, -1],
        #                        [-1,  1]]
        #
        #    Consistent load: f = [q_s·L/2, q_s·L/2]
        #
        #    F = K·d - f
        #    F_s_i =  EA/L·(u_s_i - u_s_j) - q_s·L/2
        #    F_s_j =  EA/L·(u_s_j - u_s_i) - q_s·L/2
        # ════════════════════════════════════════════════
        F_s_i =  EA / L * (u_s_i - u_s_j) - q_s * L / 2
        F_s_j =  EA / L * (u_s_j - u_s_i) - q_s * L / 2

        # ════════════════════════════════════════════════
        # 6. BENDING END FORCES (local, Hermite cubic)
        #
        #    K_bend = (EI/L³) · [[ 12,   6L,  -12,   6L ],
        #                        [ 6L,  4L²,  -6L,  2L²],
        #                        [-12,  -6L,   12,  -6L ],
        #                        [ 6L,  2L²,  -6L,  4L²]]
        #
        #    d = [u_n_i, φ_i, u_n_j, φ_j]
        #
        #    Consistent UDL load vector:
        #      f = [q_n·L/2,  q_n·L²/12,
        #           q_n·L/2, -q_n·L²/12]
        #
        #    [F_n_i, M_i, F_n_j, M_j] = K·d - f
        # ════════════════════════════════════════════════
        a = EI / (L ** 3)
        L2 = L ** 2

        F_n_i = (a * ( 12 * u_n_i + 6*L * phi_i
                      - 12 * u_n_j + 6*L * phi_j)
                 - q_n * L / 2)

        M_i   = (a * ( 6*L * u_n_i + 4*L2 * phi_i
                      - 6*L * u_n_j + 2*L2 * phi_j)
                 - q_n * L2 / 12)

        F_n_j = (a * (-12 * u_n_i - 6*L * phi_i
                      + 12 * u_n_j - 6*L * phi_j)
                 - q_n * L / 2)

        M_j   = (a * ( 6*L * u_n_i + 2*L2 * phi_i
                      - 6*L * u_n_j + 4*L2 * phi_j)
                 + q_n * L2 / 12)

        # ════════════════════════════════════════════════
        # 7. TRANSFORM END FORCES → GLOBAL
        #
        #    F_x = F_s·cos(θ) - F_n·sin(θ)
        #    F_z = F_s·sin(θ) + F_n·cos(θ)
        #    M_y stays same
        # ════════════════════════════════════════════════
        Fx_i = F_s_i * cos_t - F_n_i * sin_t
        Fz_i = F_s_i * sin_t + F_n_i * cos_t

        Fx_j = F_s_j * cos_t - F_n_j * sin_t
        Fz_j = F_s_j * sin_t + F_n_j * cos_t

        # ════════════════════════════════════════════════
        # 8. ASSEMBLE: sum element end forces at each node
        #
        #    R_k = Σ F_element_end_at_k
        #
        #    At equilibrium: R_k = 0 for all free nodes
        #    (support nodes carry reactions → exclude)
        # ════════════════════════════════════════════════
        R_x = torch.zeros(N_total, device=device)
        R_z = torch.zeros(N_total, device=device)
        R_m = torch.zeros(N_total, device=device)

        R_x.scatter_add_(0, node_i, Fx_i)
        R_x.scatter_add_(0, node_j, Fx_j)

        R_z.scatter_add_(0, node_i, Fz_i)
        R_z.scatter_add_(0, node_j, Fz_j)

        R_m.scatter_add_(0, node_i, M_i)
        R_m.scatter_add_(0, node_j, M_j)

        # ════════════════════════════════════════════════
        # 9. EQUILIBRIUM LOSS (free nodes only)
        # ════════════════════════════════════════════════
        free_disp = (data.bc_disp.squeeze(-1) < 0.5)   # (N,)
        free_rot  = (data.bc_rot.squeeze(-1) < 0.5)     # (N,)

        n_free_disp = free_disp.sum().clamp(min=1)
        n_free_rot  = free_rot.sum().clamp(min=1)

        L_fx = R_x[free_disp].pow(2).sum() / n_free_disp
        L_fz = R_z[free_disp].pow(2).sum() / n_free_disp
        L_m  = R_m[free_rot].pow(2).sum() / n_free_rot

        L_eq = L_fx + L_fz + L_m

        # ════════════════════════════════════════════════
        # 10. KINEMATIC CONSISTENCY (optional helper)
        #
        #     EB: φ = du_n/ds
        #     At element midpoint (linear approx):
        #       slope ≈ (u_n_j - u_n_i) / L
        #       φ_mid ≈ (φ_i + φ_j) / 2
        #       residual = φ_mid - slope → 0
        #
        #     This helps the GNN learn consistent
        #     (u_n, φ) pairs faster.
        # ════════════════════════════════════════════════
        slope_mid = (u_n_j - u_n_i) / L
        phi_mid   = 0.5 * (phi_i + phi_j)
        r_kin     = phi_mid - slope_mid

        L_kin = r_kin.pow(2).mean()

        # ════════════════════════════════════════════════
        # 11. TOTAL LOSS
        # ════════════════════════════════════════════════
        total = self.w_eq * L_eq + self.w_kin * L_kin

        loss_dict = {
            'eq_fx':     L_fx.item(),
            'eq_fz':     L_fz.item(),
            'eq_m':      L_m.item(),
            'eq_total':  L_eq.item(),
            'kinematic': L_kin.item(),
            'total':     total.item(),
        }

        return total, loss_dict