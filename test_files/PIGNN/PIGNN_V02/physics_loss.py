# """
# physics_loss.py — Physics-based loss for displacement prediction.

# NO TARGETS NEEDED. Only physics:

#   1. Axial equilibrium:  At each free node, sum of internal 
#      axial forces = external load (along element directions)
     
#      F_axial = E * A * (u_j - u_i) . d_hat / L
     
#      At each free node:  sum(F_axial) + F_external = 0

#   2. BC loss is handled by hard masking in model (already done).
# """

# import torch
# import torch.nn as nn


# class PhysicsLoss(nn.Module):
#     """
#     Equilibrium loss: internal forces must balance external loads
#     at every free (unconstrained) node.
    
#     Works on SINGLE graph (not batched).
#     """

#     def __init__(self):
#         super().__init__()

#     def forward(self, pred, data):
#         """
#         Args:
#             pred: (N, 6) predicted [ux, uy, uz, rx, ry, rz]
#             data: PyG Data object (single graph, NOT normalized)
        
#         Returns:
#             loss: scalar (should → 0 when equilibrium is satisfied)
#         """
#         residual = self.equilibrium_residual(pred, data)  # (N_free, 3)
#         loss = (residual ** 2).mean()
#         return loss

#     def equilibrium_residual(self, pred, data):
#         """
#         At each free node:
#             sum of internal forces + external load = 0
        
#         Internal force in element e (from node i to node j):
#             F_axial = E * A * dot(u_j - u_i, d_hat) / L
#             Force vector on node i from element e: +F_axial * d_hat
#             Force vector on node j from element e: -F_axial * d_hat
        
#         Returns:
#             residual: (N_free, 3)  — should be zero
#         """
#         N = pred.shape[0]
#         u = pred[:, :3]                          # (N, 3) translations

#         conn = data.connectivity                  # (E, 2)
#         d_hat = data.elem_directions              # (E, 3) unit vectors
#         L = data.elem_lengths                     # (E,)
#         E_mod = data.prop_E                       # (E,)
#         A = data.prop_A                           # (E,)

#         # Node displacements at element ends
#         i_nodes = conn[:, 0]                      # (E,)
#         j_nodes = conn[:, 1]                      # (E,)

#         u_i = u[i_nodes]                          # (E, 3)
#         u_j = u[j_nodes]                          # (E, 3)

#         # Axial deformation = dot(u_j - u_i, d_hat)
#         du = u_j - u_i                            # (E, 3)
#         axial_disp = (du * d_hat).sum(dim=-1)     # (E,)

#         # Axial force (scalar, positive = tension)
#         F_axial = E_mod * A * axial_disp / L      # (E,)

#         # Force vectors: F_axial * d_hat
#         F_vec = F_axial.unsqueeze(-1) * d_hat      # (E, 3)

#         # Assemble at nodes
#         # Convention: element pulls node_i toward node_j (+d_hat)
#         #             element pulls node_j toward node_i (-d_hat)
#         F_internal = torch.zeros(N, 3, device=pred.device)

#         F_internal.scatter_add_(
#             0,
#             i_nodes.unsqueeze(-1).expand_as(F_vec),
#             F_vec                                  # +F on node i
#         )
#         F_internal.scatter_add_(
#             0,
#             j_nodes.unsqueeze(-1).expand_as(F_vec),
#             -F_vec                                 # -F on node j
#         )

#         # External load
#         F_ext = data.line_load                     # (N, 3)

#         # Total residual = internal + external (should = 0)
#         residual = F_internal + F_ext              # (N, 3)

#         # Only at FREE nodes (supports can have reactions)
#         free_mask = (data.bc_disp.squeeze(-1) < 0.5)  # (N,) bool
#         residual_free = residual[free_mask]        # (N_free, 3)

#         return residual_free


"""
physics_loss.py — Axial + Bending equilibrium (no FEM targets)

Physics:
  1. AXIAL:   F = EA * (u_j - u_i) · d̂ / L
  2. BENDING: V = 12EI/L³ * Δw + 6EI/L² * (θi + θj)
              M = 6EI/L² * Δw + EI/L * (2θi + θj)  at node i
              
  At each FREE node:  sum(forces)  + F_external = 0
                       sum(moments) + M_external = 0
"""

import torch
import torch.nn as nn


class PhysicsLoss(nn.Module):
    """
    Axial + Euler-Bernoulli bending equilibrium.
    Works on SINGLE graph (not batched).
    """

    def __init__(self, w_axial=1.0, w_bending=1.0):
        super().__init__()
        self.w_axial = w_axial
        self.w_bending = w_bending

    def forward(self, pred, data):
        """
        Args:
            pred: (N, 6) [ux, uy, uz, rx, ry, rz]
            data: PyG Data (RAW units)
        Returns:
            loss: scalar (normalized)
        """
        force_res, moment_res = self.full_residual(pred, data)

        # Normalize by characteristic force/moment
        EA = data.prop_E * data.prop_A          # (E,)
        L = data.elem_lengths                   # (E,)
        F_char = EA.mean()                      # characteristic force
        M_char = (data.prop_E * data.prop_I22 / L).mean()  # char moment

        # Clamp to avoid division by zero
        F_char = F_char.clamp(min=1.0)
        M_char = M_char.clamp(min=1.0)

        loss_force = (force_res / F_char).pow(2).mean()
        loss_moment = (moment_res / M_char).pow(2).mean()

        loss = self.w_axial * loss_force + self.w_bending * loss_moment
        return loss

    def full_residual(self, pred, data):
        """
        Compute force and moment residuals at free nodes.

        For each element (Euler-Bernoulli beam in 3D):
          Local axes:  x̂ = element direction (d_hat)
                       ŷ, ẑ = perpendicular (computed here)

          Axial (along x̂):
            N = EA/L * (u_j - u_i) · x̂

          Bending in local xz plane (transverse ẑ):
            V_z = 12EI/(L³) * Δw_z + 6EI/(L²) * (θy_i + θy_j)
            M_y_i = 6EI/(L²) * Δw_z + EI/L * (2θy_i + θy_j)
            M_y_j = 6EI/(L²) * Δw_z + EI/L * (θy_i + 2θy_j)

        Returns:
            force_residual:  (N_free, 3)
            moment_residual: (N_free, 3)
        """
        N_nodes = pred.shape[0]
        device = pred.device

        u = pred[:, :3]       # translations (N, 3)
        theta = pred[:, 3:6]  # rotations    (N, 3)

        conn = data.connectivity       # (E, 2)
        d_hat = data.elem_directions   # (E, 3) unit vector along element
        L = data.elem_lengths          # (E,)
        E_mod = data.prop_E            # (E,)
        A = data.prop_A                # (E,)
        I22 = data.prop_I22            # (E,)

        i_nodes = conn[:, 0]
        j_nodes = conn[:, 1]
        n_elem = conn.shape[0]

        # ── Local coordinate system for each element ──
        x_local = d_hat                              # (E, 3)
        y_local, z_local = self._local_axes(x_local) # (E, 3) each

        # ── Nodal values ──
        u_i = u[i_nodes]           # (E, 3)
        u_j = u[j_nodes]           # (E, 3)
        th_i = theta[i_nodes]      # (E, 3)
        th_j = theta[j_nodes]      # (E, 3)

        du = u_j - u_i             # (E, 3)

        # ── 1. AXIAL FORCE ──
        axial_disp = (du * x_local).sum(dim=-1)        # (E,)
        N_axial = E_mod * A / L * axial_disp            # (E,)

        # Force vector from axial: +N on node_i, -N on node_j
        F_axial_vec = N_axial.unsqueeze(-1) * x_local  # (E, 3)

        # ── 2. BENDING IN xz PLANE (transverse z_local) ──
        # Transverse displacement difference in z_local
        dw_z = (du * z_local).sum(dim=-1)               # (E,)
        # Rotation components about y_local
        theta_yi = (th_i * y_local).sum(dim=-1)          # (E,)
        theta_yj = (th_j * y_local).sum(dim=-1)          # (E,)

        EI = E_mod * I22                                 # (E,)
        L2 = L * L
        L3 = L2 * L

        # Shear force (acts in z_local direction)
        V_z = 12.0 * EI / L3 * dw_z + 6.0 * EI / L2 * (theta_yi + theta_yj)

        # Shear force vectors
        F_shear_vec = V_z.unsqueeze(-1) * z_local       # (E, 3)

        # Moments about y_local axis
        M_yi = 6.0 * EI / L2 * dw_z + EI / L * (2.0 * theta_yi + theta_yj)
        M_yj = 6.0 * EI / L2 * dw_z + EI / L * (theta_yi + 2.0 * theta_yj)

        # Moment vectors (about y_local)
        M_vec_i = M_yi.unsqueeze(-1) * y_local           # (E, 3)
        M_vec_j = M_yj.unsqueeze(-1) * y_local           # (E, 3)

        # ── 3. BENDING IN xy PLANE (transverse y_local) ──
        dw_y = (du * y_local).sum(dim=-1)                # (E,)
        theta_zi = (th_i * z_local).sum(dim=-1)           # (E,)
        theta_zj = (th_j * z_local).sum(dim=-1)           # (E,)

        V_y = 12.0 * EI / L3 * dw_y + 6.0 * EI / L2 * (theta_zi + theta_zj)
        F_shear_y_vec = V_y.unsqueeze(-1) * y_local      # (E, 3)

        M_zi = 6.0 * EI / L2 * dw_y + EI / L * (2.0 * theta_zi + theta_zj)
        M_zj = 6.0 * EI / L2 * dw_y + EI / L * (theta_zi + 2.0 * theta_zj)

        M_vec_zi = M_zi.unsqueeze(-1) * z_local           # (E, 3)
        M_vec_zj = M_zj.unsqueeze(-1) * z_local           # (E, 3)

        # ── ASSEMBLE FORCES AT NODES ──
        F_total = torch.zeros(N_nodes, 3, device=device)

        # Axial: +N on i, -N on j
        F_total.scatter_add_(0, i_nodes.unsqueeze(-1).expand(n_elem, 3), F_axial_vec)
        F_total.scatter_add_(0, j_nodes.unsqueeze(-1).expand(n_elem, 3), -F_axial_vec)

        # Shear z: +V on i, -V on j
        F_total.scatter_add_(0, i_nodes.unsqueeze(-1).expand(n_elem, 3), F_shear_vec)
        F_total.scatter_add_(0, j_nodes.unsqueeze(-1).expand(n_elem, 3), -F_shear_vec)

        # Shear y: +V on i, -V on j
        F_total.scatter_add_(0, i_nodes.unsqueeze(-1).expand(n_elem, 3), F_shear_y_vec)
        F_total.scatter_add_(0, j_nodes.unsqueeze(-1).expand(n_elem, 3), -F_shear_y_vec)

        # External forces
        F_ext = data.line_load                          # (N, 3)
        F_residual = F_total + F_ext                    # (N, 3)

        # ── ASSEMBLE MOMENTS AT NODES ──
        M_total = torch.zeros(N_nodes, 3, device=device)

        # Moment from xz bending
        M_total.scatter_add_(0, i_nodes.unsqueeze(-1).expand(n_elem, 3), M_vec_i)
        M_total.scatter_add_(0, j_nodes.unsqueeze(-1).expand(n_elem, 3), M_vec_j)

        # Moment from xy bending
        M_total.scatter_add_(0, i_nodes.unsqueeze(-1).expand(n_elem, 3), M_vec_zi)
        M_total.scatter_add_(0, j_nodes.unsqueeze(-1).expand(n_elem, 3), M_vec_zj)

        M_residual = M_total                            # (N, 3)
        # (no external moments in this problem)

        # ── FREE NODES ONLY ──
        free_disp = (data.bc_disp.squeeze(-1) < 0.5)   # (N,) bool
        free_rot = (data.bc_rot.squeeze(-1) < 0.5)      # (N,) bool

        F_res_free = F_residual[free_disp]               # (N_free, 3)
        M_res_free = M_residual[free_rot]                # (N_free, 3)

        return F_res_free, M_res_free

    @staticmethod
    def _local_axes(x_local):
        """
        Build local y, z axes perpendicular to element direction.

        Args:
            x_local: (E, 3) unit vector along element

        Returns:
            y_local: (E, 3)
            z_local: (E, 3)
        """
        E = x_local.shape[0]
        device = x_local.device

        # Pick a reference vector not parallel to x_local
        ref = torch.zeros(E, 3, device=device)
        ref[:, 1] = 1.0  # default: global Y

        # If element is along Y, use global Z instead
        parallel = (x_local * ref).sum(dim=-1).abs() > 0.99
        ref[parallel] = torch.tensor([0.0, 0.0, 1.0], device=device)

        # Gram-Schmidt
        z_local = torch.cross(x_local, ref, dim=-1)
        z_local = z_local / z_local.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        y_local = torch.cross(z_local, x_local, dim=-1)
        y_local = y_local / y_local.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        return y_local, z_local