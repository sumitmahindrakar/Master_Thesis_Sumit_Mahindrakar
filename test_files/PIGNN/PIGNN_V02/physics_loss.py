"""
physics_loss.py — Physics-based loss for displacement prediction.

NO TARGETS NEEDED. Only physics:

  1. Axial equilibrium:  At each free node, sum of internal 
     axial forces = external load (along element directions)
     
     F_axial = E * A * (u_j - u_i) . d_hat / L
     
     At each free node:  sum(F_axial) + F_external = 0

  2. BC loss is handled by hard masking in model (already done).
"""

import torch
import torch.nn as nn


class PhysicsLoss(nn.Module):
    """
    Equilibrium loss: internal forces must balance external loads
    at every free (unconstrained) node.
    
    Works on SINGLE graph (not batched).
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, data):
        """
        Args:
            pred: (N, 6) predicted [ux, uy, uz, rx, ry, rz]
            data: PyG Data object (single graph, NOT normalized)
        
        Returns:
            loss: scalar (should → 0 when equilibrium is satisfied)
        """
        residual = self.equilibrium_residual(pred, data)  # (N_free, 3)
        loss = (residual ** 2).mean()
        return loss

    def equilibrium_residual(self, pred, data):
        """
        At each free node:
            sum of internal forces + external load = 0
        
        Internal force in element e (from node i to node j):
            F_axial = E * A * dot(u_j - u_i, d_hat) / L
            Force vector on node i from element e: +F_axial * d_hat
            Force vector on node j from element e: -F_axial * d_hat
        
        Returns:
            residual: (N_free, 3)  — should be zero
        """
        N = pred.shape[0]
        u = pred[:, :3]                          # (N, 3) translations

        conn = data.connectivity                  # (E, 2)
        d_hat = data.elem_directions              # (E, 3) unit vectors
        L = data.elem_lengths                     # (E,)
        E_mod = data.prop_E                       # (E,)
        A = data.prop_A                           # (E,)

        # Node displacements at element ends
        i_nodes = conn[:, 0]                      # (E,)
        j_nodes = conn[:, 1]                      # (E,)

        u_i = u[i_nodes]                          # (E, 3)
        u_j = u[j_nodes]                          # (E, 3)

        # Axial deformation = dot(u_j - u_i, d_hat)
        du = u_j - u_i                            # (E, 3)
        axial_disp = (du * d_hat).sum(dim=-1)     # (E,)

        # Axial force (scalar, positive = tension)
        F_axial = E_mod * A * axial_disp / L      # (E,)

        # Force vectors: F_axial * d_hat
        F_vec = F_axial.unsqueeze(-1) * d_hat      # (E, 3)

        # Assemble at nodes
        # Convention: element pulls node_i toward node_j (+d_hat)
        #             element pulls node_j toward node_i (-d_hat)
        F_internal = torch.zeros(N, 3, device=pred.device)

        F_internal.scatter_add_(
            0,
            i_nodes.unsqueeze(-1).expand_as(F_vec),
            F_vec                                  # +F on node i
        )
        F_internal.scatter_add_(
            0,
            j_nodes.unsqueeze(-1).expand_as(F_vec),
            -F_vec                                 # -F on node j
        )

        # External load
        F_ext = data.line_load                     # (N, 3)

        # Total residual = internal + external (should = 0)
        residual = F_internal + F_ext              # (N, 3)

        # Only at FREE nodes (supports can have reactions)
        free_mask = (data.bc_disp.squeeze(-1) < 0.5)  # (N,) bool
        residual_free = residual[free_mask]        # (N_free, 3)

        return residual_free