# import torch
# import torch.nn as nn


# class FramePhysicsLoss(nn.Module):
#     """
#     Physics loss using autograd derivatives of continuous edge fields.
    
#     For each element, the EdgeFieldMLP gives:
#         u(ξ)  = axial displacement
#         w(ξ)  = transverse displacement  
#         θ(ξ)  = rotation
    
#     where ξ ∈ [0,1] is the local coordinate along the element.
    
#     Physical coordinate: x = ξ · L
#     Chain rule: d/dx = (1/L) · d/dξ
    
#     Internal forces (Euler-Bernoulli beam):
#         N(ξ) = EA · du/dx = EA/L · du/dξ
#         M(ξ) = EI · d²w/dx² = EI/L² · d²w/dξ²
#         V(ξ) = EI · d³w/dx³ = EI/L³ · d³w/dξ³
    
#     Strong form equilibrium:
#         dN/dx + w_axial = 0      → EA/L² · d²u/dξ² + w_axial = 0
#         d²M/dx² + w_trans = 0    → EI/L⁴ · d⁴w/dξ⁴ + w_trans = 0
    
#     We enforce equilibrium at quadrature points along each element.
#     """

#     def __init__(self):
#         super().__init__()

#     def compute_derivatives(self, fields, xi):
#         """
#         Compute spatial derivatives via autograd.
        
#         Args:
#             fields: (E, n_pts, 3) = [u, w, θ]
#             xi:     (E, n_pts, 1) local coordinates (requires_grad=True)
        
#         Returns:
#             dict of derivatives, each (E, n_pts)
#         """
#         u = fields[:, :, 0]   # (E, n_pts) axial
#         w = fields[:, :, 1]   # (E, n_pts) transverse
#         theta = fields[:, :, 2]  # (E, n_pts) rotation

#         # du/dξ
#         du_dxi = torch.autograd.grad(
#             u.sum(), xi, create_graph=True
#         )[0].squeeze(-1)  # (E, n_pts)

#         # dw/dξ
#         dw_dxi = torch.autograd.grad(
#             w.sum(), xi, create_graph=True
#         )[0].squeeze(-1)

#         # d²w/dξ²
#         d2w_dxi2 = torch.autograd.grad(
#             dw_dxi.sum(), xi, create_graph=True
#         )[0].squeeze(-1)

#         # d³w/dξ³
#         d3w_dxi3 = torch.autograd.grad(
#             d2w_dxi2.sum(), xi, create_graph=True
#         )[0].squeeze(-1)

#         # d²u/dξ² (for strong form axial equilibrium)
#         d2u_dxi2 = torch.autograd.grad(
#             du_dxi.sum(), xi, create_graph=True
#         )[0].squeeze(-1)

#         # d⁴w/dξ⁴ (for strong form bending equilibrium)
#         d4w_dxi4 = torch.autograd.grad(
#             d3w_dxi3.sum(), xi, create_graph=True
#         )[0].squeeze(-1)

#         return {
#             'u': u, 'w': w, 'theta': theta,
#             'du_dxi': du_dxi,
#             'dw_dxi': dw_dxi,
#             'd2w_dxi2': d2w_dxi2,
#             'd3w_dxi3': d3w_dxi3,
#             'd2u_dxi2': d2u_dxi2,
#             'd4w_dxi4': d4w_dxi4,
#         }

#     def compute_internal_forces(self, derivs, E_mod, A, I22, L):
#         """
#         Compute N, M, V from derivatives.
        
#         Chain rule: d/dx = (1/L) · d/dξ
        
#         Args:
#             derivs: dict from compute_derivatives()
#             E_mod:  (n_elem,) Young's modulus
#             A:      (n_elem,) cross-section area
#             I22:    (n_elem,) second moment of area
#             L:      (n_elem,) element length
        
#         Returns:
#             N: (E, n_pts) axial force
#             M: (E, n_pts) bending moment
#             V: (E, n_pts) shear force
#         """
#         # Expand to (E, 1) for broadcasting with (E, n_pts)
#         EA = (E_mod * A).unsqueeze(1)
#         EI = (E_mod * I22).unsqueeze(1)
#         L_ = L.unsqueeze(1)

#         # N = EA · du/dx = EA/L · du/dξ
#         N = EA / L_ * derivs['du_dxi']

#         # M = EI · d²w/dx² = EI/L² · d²w/dξ²
#         M = EI / L_.pow(2) * derivs['d2w_dxi2']

#         # V = EI · d³w/dx³ = EI/L³ · d³w/dξ³
#         # Or equivalently V = dM/dx
#         V = EI / L_.pow(3) * derivs['d3w_dxi3']

#         return N, M, V

#     def equilibrium_residual(self, derivs, E_mod, A, I22, L,
#                               w_axial, w_transverse):
#         """
#         Strong-form equilibrium residual at interior points.
        
#         Governing equations (Euler-Bernoulli):
#             EA · d²u/dx² + w_axial = 0
#             EI · d⁴w/dx⁴ + w_transverse = 0  (note: w_trans is load
#                                                 per unit length,
#                                                 sign convention matters)
        
#         In ξ coordinates:
#             EA/L² · d²u/dξ² + w_axial = 0
#             EI/L⁴ · d⁴w/dξ⁴ + w_transverse = 0
        
#         Args:
#             derivs:        dict from compute_derivatives()
#             E_mod, A, I22: (n_elem,) element properties
#             L:             (n_elem,) element lengths
#             w_axial:       (n_elem,) axial distributed load
#             w_transverse:  (n_elem,) transverse distributed load
        
#         Returns:
#             residual_axial:   (E, n_pts) should be ≈ 0
#             residual_bending: (E, n_pts) should be ≈ 0
#         """
#         EA = (E_mod * A).unsqueeze(1)
#         EI = (E_mod * I22).unsqueeze(1)
#         L_ = L.unsqueeze(1)
#         w_ax = w_axial.unsqueeze(1)
#         w_tr = w_transverse.unsqueeze(1)

#         # Axial: EA/L² · d²u/dξ² + w_axial = 0
#         res_axial = EA / L_.pow(2) * derivs['d2u_dxi2'] + w_ax

#         # Bending: EI/L⁴ · d⁴w/dξ⁴ - w_transverse = 0
#         # Sign: positive w_trans is downward load,
#         # EI·w'''' = w_trans (load = resistance)
#         res_bending = EI / L_.pow(4) * derivs['d4w_dxi4'] - w_tr

#         return res_axial, res_bending

#     def compatibility_loss(self, derivs, L):
#         """
#         Kinematic compatibility: θ = dw/dx = (1/L)·dw/dξ
        
#         The rotation θ predicted by the network should match
#         the derivative of transverse displacement.
        
#         Args:
#             derivs: dict from compute_derivatives()
#             L:      (n_elem,) element lengths
        
#         Returns:
#             scalar loss
#         """
#         L_ = L.unsqueeze(1)
#         theta_pred = derivs['theta']           # (E, n_pts)
#         theta_from_w = derivs['dw_dxi'] / L_   # (E, n_pts)

#         return ((theta_pred - theta_from_w) ** 2).mean()

#     def continuity_loss(self, fields, connectivity, n_nodes):
#         """
#         Displacement continuity at shared nodes.
        
#         If element A ends at node j (ξ=1) and element B starts 
#         at node j (ξ=0), their displacements must match.
        
#         Strategy: for each node, collect all element-end values
#         arriving at that node, compute variance → penalize.
        
#         Args:
#             fields:       (E, n_pts, 3) displacement fields
#             connectivity: (E, 2) element-to-node mapping
#             n_nodes:      int
        
#         Returns:
#             scalar loss
#         """
#         n1 = connectivity[:, 0]  # (E,) start nodes
#         n2 = connectivity[:, 1]  # (E,) end nodes

#         u_at_start = fields[:, 0, :]    # (E, 3) field at ξ=0 → node n1
#         u_at_end = fields[:, -1, :]     # (E, 3) field at ξ=1 → node n2

#         # Sum of values arriving at each node
#         node_sum = torch.zeros(n_nodes, 3,
#                             device=fields.device, dtype=fields.dtype)
#         # Sum of squares arriving at each node
#         node_sq = torch.zeros_like(node_sum)
#         # Count of contributions per node
#         node_count = torch.zeros(n_nodes, 1,
#                                 device=fields.device, dtype=fields.dtype)

#         ones = torch.ones(n1.shape[0], 1,
#                         device=fields.device, dtype=fields.dtype)

#         # ξ=0 end → node n1
#         node_sum.scatter_add_(0, n1.unsqueeze(1).expand(-1, 3), u_at_start)
#         node_sq.scatter_add_(0, n1.unsqueeze(1).expand(-1, 3), u_at_start ** 2)
#         node_count.scatter_add_(0, n1.unsqueeze(1), ones)

#         # ξ=1 end → node n2
#         node_sum.scatter_add_(0, n2.unsqueeze(1).expand(-1, 3), u_at_end)
#         node_sq.scatter_add_(0, n2.unsqueeze(1).expand(-1, 3), u_at_end ** 2)
#         node_count.scatter_add_(0, n2.unsqueeze(1), ones)

#         # Variance at each node = E[x²] - E[x]²
#         # Only nodes with count > 1 (shared nodes) matter
#         node_count = node_count.clamp(min=1)  # avoid division by zero
#         mean = node_sum / node_count
#         mean_sq = node_sq / node_count
#         variance = mean_sq - mean ** 2  # (N, 3)

#         # Only penalize nodes connected to 2+ elements
#         multi_mask = (node_count.squeeze(-1) > 1.5)  # (N,)
#         if not multi_mask.any():
#             return torch.tensor(0.0, device=fields.device)

#         return variance[multi_mask].mean()

#     def compute_sensitivity(self, derivs, E_mod, I22, L,
#                             connectivity, response_node_flag):
#         """
#         Compute dM/dI22 for each element via autograd.
        
#         M(ξ) = EI/L² · d²w/dξ²
        
#         Since I22 flows through M, autograd gives dM/dI22.
        
#         Args:
#             derivs:             dict from compute_derivatives()
#             E_mod:              (n_elem,) Young's modulus
#             I22:                (n_elem,) second moment of area (WITH grad)
#             L:                  (n_elem,) element lengths
#             connectivity:       (E, 2)
#             response_node_flag: (N, 1)
        
#         Returns:
#             dM_dI: (n_elem,) sensitivity at element midpoint
#         """
#         EI = E_mod * I22         # (E,)
#         L_ = L

#         # M at all quad points: (E, n_pts)
#         M = (EI / L_.pow(2)).unsqueeze(1) * derivs['d2w_dxi2']

#         # Find response node
#         resp_idx = torch.where(
#             response_node_flag.squeeze(-1) > 0.5)[0]

#         if len(resp_idx) == 0:
#             # No response node → use midpoint of all elements
#             n_pts = M.shape[1]
#             mid = n_pts // 2
#             M_response = M[:, mid].sum()  # scalar for autograd
#         else:
#             # Elements connected to response node
#             resp_node = resp_idx[0]
#             mask = ((connectivity[:, 0] == resp_node) |
#                     (connectivity[:, 1] == resp_node))
#             M_response = M[mask].mean()

#         # dM/dI22 via autograd
#         dM_dI = torch.autograd.grad(
#             M_response,
#             I22,
#             create_graph=True,   # allows higher-order grad if needed
#             retain_graph=True,
#         )[0]  # (n_elem,)

#         return dM_dI, M

#     def distributed_load_to_local(self, line_load, connectivity,
#                                    elem_directions):
#         """
#         Convert global LINE_LOAD to local element coordinates.
        
#         Returns:
#             w_axial:      (E,) axial component
#             w_transverse: (E,) transverse component
#         """
#         n1 = connectivity[:, 0]
#         n2 = connectivity[:, 1]

#         # Average load at element endpoints
#         w_elem = (line_load[n1] + line_load[n2]) / 2.0  # (E, 3)

#         cos_t = elem_directions[:, 0]
#         sin_t = elem_directions[:, 2]

#         wx = w_elem[:, 0]
#         wz = w_elem[:, 2]

#         w_axial = wx * cos_t + wz * sin_t
#         w_transverse = -wx * sin_t + wz * cos_t

#         return w_axial, w_transverse

#     def forward(self, fields, xi, data, I22_grad=None):
#         """
#         Args:
#             fields:    (E, n_pts, 3) from model
#             xi:        (E, n_pts, 1) local coords (requires_grad)
#             data:      PyG Data object
#             I22_grad:  (n_elem,) I22 with requires_grad=True
#                     (passed separately to enable autograd)
        
#         Returns:
#             total_loss, loss_dict
#         """
#         # ── 1. Compute derivatives ──
#         derivs = self.compute_derivatives(fields, xi)

#         # ── 2. Get distributed loads in local coords ──
#         w_ax, w_tr = self.distributed_load_to_local(
#             data.line_load, data.connectivity,
#             data.elem_directions)

#         # ── 3. Equilibrium residual ──
#         res_axial, res_bending = self.equilibrium_residual(
#             derivs,
#             data.prop_E, data.prop_A, data.prop_I22,
#             data.elem_lengths,
#             w_ax, w_tr)

#         loss_axial = (res_axial ** 2).mean()
#         loss_bending = (res_bending ** 2).mean()

#         # ── 4. Compatibility: θ = dw/dx ──
#         loss_compat = self.compatibility_loss(
#             derivs, data.elem_lengths)

#         # ── 5. Continuity at shared nodes ──
#         loss_cont = self.continuity_loss(
#             fields, data.connectivity, data.num_nodes)

#         # ── 6. Sensitivity dM/dI22 ──
#         loss_sens = torch.tensor(0.0, device=fields.device)
#         dM_dI = None

#         if I22_grad is not None:
#             dM_dI, M_field = self.compute_sensitivity(
#                 derivs,
#                 data.prop_E,
#                 I22_grad,           # ← has requires_grad=True
#                 data.elem_lengths,
#                 data.connectivity,
#                 data.response_node_flag,
#             )

#             # If model predicts sensitivity, compare:
#             if hasattr(data, 'y_element') and data.y_element is not None:
#                 sens_target = data.y_element[:, 6]  # Kratos dBM/dI
#                 loss_sens = ((dM_dI - sens_target) ** 2).mean()
#             # OR if you want pure physics (no data):
#             # Just store dM_dI for output, no loss needed

#         # ── 7. Total ──
#         w_sens = 0.1  # weight for sensitivity loss
#         total = (loss_axial + loss_bending + loss_compat
#                 + loss_cont + w_sens * loss_sens)

#         return total, {
#             'axial_equilibrium': loss_axial.item(),
#             'bending_equilibrium': loss_bending.item(),
#             'compatibility': loss_compat.item(),
#             'continuity': loss_cont.item(),
#             'sensitivity': loss_sens.item(),
#             'total': total.item(),
#         }, dM_dI  # ← return sensitivity as output


"""
=================================================================
losses.py — PHYSICS-INFORMED LOSS FUNCTIONS
=================================================================
All losses computed from autograd derivatives of continuous edge
fields. NO data targets used.

Loss components:
    1. Axial equilibrium:   EA/L² · d²u/dξ² + w_axial = 0
    2. Bending equilibrium: EI/L⁴ · d⁴w/dξ⁴ - w_trans = 0
    3. Compatibility:       θ = (1/L) · dw/dξ
    4. Continuity:          u must match at shared nodes
    5. Sensitivity:         dM/dI22 via autograd (output, not loss)

Internal forces (computed for output):
    N = EA/L · du/dξ
    M = EI/L² · d²w/dξ²
    V = EI/L³ · d³w/dξ³
=================================================================
"""

import torch
import torch.nn as nn


class FramePhysicsLoss(nn.Module):
    """
    Physics-informed loss using autograd derivatives of
    continuous edge fields from FramePIGNN.
    """

    def __init__(self):
        super().__init__()

    # ─────────────────────────────────────────────
    # DERIVATIVES VIA AUTOGRAD
    # ─────────────────────────────────────────────

    def compute_derivatives(self, fields, xi):
        """
        Compute spatial derivatives via autograd.

        Args:
            fields: (E, n_pts, 3) = [u, w, θ]
            xi:     (E, n_pts, 1) local coordinates (requires_grad=True)

        Returns:
            dict of derivatives, each (E, n_pts)
        """
        u = fields[:, :, 0]      # axial
        w = fields[:, :, 1]      # transverse
        theta = fields[:, :, 2]  # rotation

        # du/dξ
        du_dxi = torch.autograd.grad(
            u.sum(), xi, create_graph=True
        )[0].squeeze(-1)

        # dw/dξ
        dw_dxi = torch.autograd.grad(
            w.sum(), xi, create_graph=True
        )[0].squeeze(-1)

        # d²u/dξ²
        d2u_dxi2 = torch.autograd.grad(
            du_dxi.sum(), xi, create_graph=True
        )[0].squeeze(-1)

        # d²w/dξ²
        d2w_dxi2 = torch.autograd.grad(
            dw_dxi.sum(), xi, create_graph=True
        )[0].squeeze(-1)

        # d³w/dξ³
        d3w_dxi3 = torch.autograd.grad(
            d2w_dxi2.sum(), xi, create_graph=True
        )[0].squeeze(-1)

        # d⁴w/dξ⁴
        d4w_dxi4 = torch.autograd.grad(
            d3w_dxi3.sum(), xi, create_graph=True
        )[0].squeeze(-1)

        return {
            'u': u, 'w': w, 'theta': theta,
            'du_dxi': du_dxi,
            'dw_dxi': dw_dxi,
            'd2u_dxi2': d2u_dxi2,
            'd2w_dxi2': d2w_dxi2,
            'd3w_dxi3': d3w_dxi3,
            'd4w_dxi4': d4w_dxi4,
        }

    # ─────────────────────────────────────────────
    # INTERNAL FORCES FROM DERIVATIVES
    # ─────────────────────────────────────────────

    def compute_internal_forces(self, derivs, E_mod, A, I22, L):
        """
        Compute N, M, V from derivatives.

        Chain rule: d/dx = (1/L) · d/dξ

        Args:
            derivs: dict from compute_derivatives()
            E_mod:  (n_elem,) Young's modulus
            A:      (n_elem,) cross-section area
            I22:    (n_elem,) second moment of area
            L:      (n_elem,) element length

        Returns:
            N: (E, n_pts) axial force
            M: (E, n_pts) bending moment
            V: (E, n_pts) shear force
        """
        EA = (E_mod * A).unsqueeze(1)
        EI = (E_mod * I22).unsqueeze(1)
        L_ = L.unsqueeze(1)

        N = EA / L_ * derivs['du_dxi']
        M = EI / L_.pow(2) * derivs['d2w_dxi2']
        V = EI / L_.pow(3) * derivs['d3w_dxi3']

        return N, M, V

    # ─────────────────────────────────────────────
    # DISTRIBUTED LOAD → LOCAL COORDS
    # ─────────────────────────────────────────────

    def distributed_load_to_local(self, line_load, connectivity,
                                   elem_directions):
        """
        Convert global LINE_LOAD to local element coordinates.

        Returns:
            w_axial:      (E,) axial component
            w_transverse: (E,) transverse component
        """
        n1 = connectivity[:, 0]
        n2 = connectivity[:, 1]

        w_elem = (line_load[n1] + line_load[n2]) / 2.0

        cos_t = elem_directions[:, 0]
        sin_t = elem_directions[:, 2]

        wx = w_elem[:, 0]
        wz = w_elem[:, 2]

        w_axial = wx * cos_t + wz * sin_t
        w_transverse = -wx * sin_t + wz * cos_t

        return w_axial, w_transverse

    # ─────────────────────────────────────────────
    # LOSS 1 & 2: EQUILIBRIUM (STRONG FORM PDE)
    # ─────────────────────────────────────────────

    def equilibrium_residual(self, derivs, E_mod, A, I22, L,
                              w_axial, w_transverse):
        """
        Strong-form equilibrium residual at interior points.

        EA/L² · d²u/dξ² + w_axial = 0
        EI/L⁴ · d⁴w/dξ⁴ - w_transverse = 0

        Returns:
            residual_axial:   (E, n_pts)
            residual_bending: (E, n_pts)
        """
        EA = (E_mod * A).unsqueeze(1)
        EI = (E_mod * I22).unsqueeze(1)
        L_ = L.unsqueeze(1)
        w_ax = w_axial.unsqueeze(1)
        w_tr = w_transverse.unsqueeze(1)

        res_axial = EA / L_.pow(2) * derivs['d2u_dxi2'] + w_ax
        res_bending = EI / L_.pow(4) * derivs['d4w_dxi4'] - w_tr

        return res_axial, res_bending

    # ─────────────────────────────────────────────
    # LOSS 3: COMPATIBILITY θ = dw/dx
    # ─────────────────────────────────────────────

    def compatibility_loss(self, derivs, L):
        """
        Kinematic compatibility: θ = dw/dx = (1/L)·dw/dξ

        Returns:
            scalar loss
        """
        L_ = L.unsqueeze(1)
        theta_pred = derivs['theta']
        theta_from_w = derivs['dw_dxi'] / L_

        return ((theta_pred - theta_from_w) ** 2).mean()

    # ─────────────────────────────────────────────
    # LOSS 4: CONTINUITY AT SHARED NODES
    # ─────────────────────────────────────────────

    def continuity_loss(self, fields, connectivity, n_nodes):
        """
        Displacement continuity at shared nodes.

        If element A ends at node j (ξ=1) and element B starts
        at node j (ξ=0), their displacements must match.

        Strategy: collect all element-end values at each node,
        compute variance, penalize.

        Returns:
            scalar loss
        """
        n1 = connectivity[:, 0]
        n2 = connectivity[:, 1]

        u_at_start = fields[:, 0, :]   # (E, 3) at ξ=0 → node n1
        u_at_end = fields[:, -1, :]    # (E, 3) at ξ=1 → node n2

        device = fields.device
        dtype = fields.dtype

        node_sum = torch.zeros(n_nodes, 3, device=device, dtype=dtype)
        node_sq = torch.zeros(n_nodes, 3, device=device, dtype=dtype)
        node_count = torch.zeros(n_nodes, 1, device=device, dtype=dtype)

        ones = torch.ones(n1.shape[0], 1, device=device, dtype=dtype)

        # ξ=0 end → node n1
        node_sum.scatter_add_(0, n1.unsqueeze(1).expand(-1, 3), u_at_start)
        node_sq.scatter_add_(0, n1.unsqueeze(1).expand(-1, 3), u_at_start ** 2)
        node_count.scatter_add_(0, n1.unsqueeze(1), ones)

        # ξ=1 end → node n2
        node_sum.scatter_add_(0, n2.unsqueeze(1).expand(-1, 3), u_at_end)
        node_sq.scatter_add_(0, n2.unsqueeze(1).expand(-1, 3), u_at_end ** 2)
        node_count.scatter_add_(0, n2.unsqueeze(1), ones)

        # Variance = E[x²] - E[x]²
        node_count = node_count.clamp(min=1)
        mean = node_sum / node_count
        mean_sq = node_sq / node_count
        variance = mean_sq - mean ** 2

        # Only penalize nodes with 2+ element connections
        multi_mask = (node_count.squeeze(-1) > 1.5)
        if not multi_mask.any():
            return torch.tensor(0.0, device=device)

        return variance[multi_mask].mean()

    # ─────────────────────────────────────────────
    # SENSITIVITY: dM/dI22
    # ─────────────────────────────────────────────

    def compute_sensitivity(self, derivs, E_mod, I22, L,
                             connectivity, response_node_flag):
        """
        Compute dM/dI22 for each element via autograd.

        M(ξ) = EI/L² · d²w/dξ²

        Since I22 flows through M, autograd gives dM/dI22.

        Args:
            derivs:             dict from compute_derivatives()
            E_mod:              (n_elem,) Young's modulus
            I22:                (n_elem,) second moment of area (WITH grad)
            L:                  (n_elem,) element lengths
            connectivity:       (E, 2)
            response_node_flag: (N, 1)

        Returns:
            dM_dI: (n_elem,) sensitivity per element
            M:     (E, n_pts) moment field
        """
        EI = E_mod * I22
        L_ = L

        # M at all quad points
        M = (EI / L_.pow(2)).unsqueeze(1) * derivs['d2w_dxi2']

        # Find response node
        resp_idx = torch.where(
            response_node_flag.squeeze(-1) > 0.5)[0]

        if len(resp_idx) == 0:
            n_pts = M.shape[1]
            mid = n_pts // 2
            M_response = M[:, mid].sum()
        else:
            resp_node = resp_idx[0]
            mask = ((connectivity[:, 0] == resp_node) |
                    (connectivity[:, 1] == resp_node))
            M_response = M[mask].mean()

        # dM/dI22 via autograd
        try:
            dM_dI = torch.autograd.grad(
                M_response,
                I22,
                create_graph=True,
                retain_graph=True,
            )[0]
        except RuntimeError:
            dM_dI = torch.zeros_like(I22)

        return dM_dI, M

    # ─────────────────────────────────────────────
    # TOTAL LOSS
    # ─────────────────────────────────────────────

    # def forward(self, fields, xi, data, I22_grad=None):
    #     """
    #     Compute total physics-informed loss.

    #     Args:
    #         fields:   (E, n_pts, 3) from model
    #         xi:       (E, n_pts, 1) local coords (requires_grad)
    #         data:     PyG Data object
    #         I22_grad: (n_elem,) I22 with requires_grad=True

    #     Returns:
    #         total_loss: scalar
    #         loss_dict:  breakdown of loss components
    #         dM_dI:      (n_elem,) or None — sensitivity output

    #     """
    #     # 1. Compute derivatives
    #     derivs = self.compute_derivatives(fields, xi)

    #     # 2. Distributed loads in local coords
    #     w_ax, w_tr = self.distributed_load_to_local(
    #         data.line_load, data.connectivity,
    #         data.elem_directions)

    #     # 3. Equilibrium residual (strong form PDE)
    #     res_axial, res_bending = self.equilibrium_residual(
    #         derivs,
    #         data.prop_E, data.prop_A, data.prop_I22,
    #         data.elem_lengths,
    #         w_ax, w_tr)

    #     loss_axial = (res_axial ** 2).mean()
    #     loss_bending = (res_bending ** 2).mean()

    #     # 4. Compatibility: θ = dw/dx
    #     loss_compat = self.compatibility_loss(
    #         derivs, data.elem_lengths)

    #     # 5. Continuity at shared nodes
    #     loss_cont = self.continuity_loss(
    #         fields, data.connectivity, data.num_nodes)

    #     # 6. Sensitivity dM/dI22 (computed as output, not as loss)
    #     dM_dI = None
    #     loss_sens = torch.tensor(0.0, device=fields.device)

    #     if I22_grad is not None:
    #         dM_dI, M_field = self.compute_sensitivity(
    #             derivs,
    #             data.prop_E,
    #             I22_grad,
    #             data.elem_lengths,
    #             data.connectivity,
    #             data.response_node_flag,
    #         )

    #     # 7. Total loss (sensitivity is output, not penalized)
    #     total = loss_axial + loss_bending + loss_compat + loss_cont

    #     loss_dict = {
    #         'axial_equilibrium': loss_axial.item(),
    #         'bending_equilibrium': loss_bending.item(),
    #         'compatibility': loss_compat.item(),
    #         'continuity': loss_cont.item(),
    #         'sensitivity': loss_sens.item(),
    #         'total': total.item(),
    #     }

    #     return total, loss_dict, dM_dI

    def forward(self, fields, xi, data, I22_grad=None):
        """
        Compute total physics-informed loss.
        Each component is normalized by its own scale to prevent
        one term from dominating.
        """
        # 1. Compute derivatives
        derivs = self.compute_derivatives(fields, xi)

        # 2. Distributed loads in local coords
        w_ax, w_tr = self.distributed_load_to_local(
            data.line_load, data.connectivity,
            data.elem_directions)

        # 3. Equilibrium residuals
        res_axial, res_bending = self.equilibrium_residual(
            derivs,
            data.prop_E, data.prop_A, data.prop_I22,
            data.elem_lengths,
            w_ax, w_tr)

        # Normalize by EA/L² and EI/L⁴ scales respectively
        EA_L2 = (data.prop_E * data.prop_A / data.elem_lengths.pow(2))
        EI_L4 = (data.prop_E * data.prop_I22 / data.elem_lengths.pow(4))

        # Relative residuals (dimensionless)
        EA_scale = EA_L2.mean().detach().clamp(min=1e-10)
        EI_scale = EI_L4.mean().detach().clamp(min=1e-10)
        w_ax_scale = w_ax.abs().mean().detach().clamp(min=1e-10)
        w_tr_scale = w_tr.abs().mean().detach().clamp(min=1e-10)

        # Normalize: divide residual by characteristic scale
        loss_axial = (res_axial / (EA_scale + w_ax_scale)).pow(2).mean()
        loss_bending = (res_bending / (EI_scale + w_tr_scale)).pow(2).mean()

        # 4. Compatibility
        loss_compat = self.compatibility_loss(derivs, data.elem_lengths)

        # 5. Continuity
        loss_cont = self.continuity_loss(
            fields, data.connectivity, data.num_nodes)

        # 6. Sensitivity
        dM_dI = None
        loss_sens = torch.tensor(0.0, device=fields.device)

        if I22_grad is not None:
            dM_dI, M_field = self.compute_sensitivity(
                derivs,
                data.prop_E,
                I22_grad,
                data.elem_lengths,
                data.connectivity,
                data.response_node_flag,
            )

        # 7. Total — all terms now O(1)
        total = loss_axial + loss_bending + 10.0 * loss_compat + loss_cont

        loss_dict = {
            'axial_equilibrium': loss_axial.item(),
            'bending_equilibrium': loss_bending.item(),
            'compatibility': loss_compat.item(),
            'continuity': loss_cont.item(),
            'sensitivity': loss_sens.item(),
            'total': total.item(),
        }

        return total, loss_dict, dM_dI