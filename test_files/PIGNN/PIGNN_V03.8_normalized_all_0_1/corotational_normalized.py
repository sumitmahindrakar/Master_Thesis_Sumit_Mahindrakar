"""
=================================================================
corotational_normalized.py — Physics in [0,1] Space
=================================================================
"""

import torch
import torch.nn as nn


class CorotationalBeam2DNormalized(nn.Module):
    """
    Corotational beam with ALL quantities in [0, 1].
    
    Input:  pred_norm (N, 3) in [0, 1]
    Output: forces_norm in [0, 1]
    
    No scaling issues - everything is O(1)!
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_norm, data):
        """
        Args:
            pred_norm: (N, 3) network output in [0, 1]
                       [ux_norm, uz_norm, θ_norm]
            data: PyG Data with normalized properties
        
        Returns:
            result dict with normalized forces
        """
        conn = data.connectivity
        coords_norm = data.coords_norm
        E_norm = data.prop_E_norm
        A_norm = data.prop_A_norm
        I_norm = data.prop_I22_norm
        L_norm = data.elem_lengths_norm
        
        E_count = conn.shape[0]
        N_count = pred_norm.shape[0]
        device = pred_norm.device
        
        nA = conn[:, 0]
        nB = conn[:, 1]
        
        # ════════════════════════════════════════
        # Geometry (in normalized space)
        # ════════════════════════════════════════
        
        dx0 = coords_norm[nB, 0] - coords_norm[nA, 0]
        dz0 = coords_norm[nB, 2] - coords_norm[nA, 2]
        l0_norm = torch.sqrt(dx0**2 + dz0**2 + 1e-10)
        c = dx0 / (l0_norm + 1e-10)
        s = dz0 / (l0_norm + 1e-10)
        
        # ════════════════════════════════════════
        # Normalized stiffness coefficients
        # ════════════════════════════════════════
        # 
        # Key insight: In normalized space, the stiffness
        # becomes a dimensionless ratio!
        #
        # k_norm = (E_norm * A_norm) / L_norm
        #
        # Since E, A, L are all in [0,1], k_norm ∈ [0, 1/L_min]
        # which is much better behaved!
        
        EA_norm = E_norm * A_norm
        EI_norm = E_norm * I_norm
        
        # Axial stiffness (normalized)
        k_ax = EA_norm / (l0_norm + 1e-10)
        
        # Bending stiffness (normalized)
        k_bend = EI_norm / (l0_norm + 1e-10)
        
        # Shear-rotation coupling
        k_sw = EI_norm / (l0_norm.pow(2) + 1e-10)
        
        # Transverse stiffness
        k_tr = EI_norm / (l0_norm.pow(3) + 1e-10)
        
        # ════════════════════════════════════════
        # Gather normalized DOFs
        # ════════════════════════════════════════
        
        d_norm = torch.stack([
            pred_norm[nA, 0],
            pred_norm[nA, 1],
            -pred_norm[nA, 2],
            pred_norm[nB, 0],
            pred_norm[nB, 1],
            -pred_norm[nB, 2],
        ], dim=1)
        
        # ════════════════════════════════════════
        # Transform to local coordinates
        # ════════════════════════════════════════
        
        dl = torch.zeros_like(d_norm)
        
        dl[:, 0] = c * d_norm[:, 0] + s * d_norm[:, 1]
        dl[:, 1] = -s * d_norm[:, 0] + c * d_norm[:, 1]
        dl[:, 2] = d_norm[:, 2]
        
        dl[:, 3] = c * d_norm[:, 3] + s * d_norm[:, 4]
        dl[:, 4] = -s * d_norm[:, 3] + c * d_norm[:, 4]
        dl[:, 5] = d_norm[:, 5]
        
        ua = dl[:, 0]
        wa = dl[:, 1]
        ta = dl[:, 2]
        ub = dl[:, 3]
        wb = dl[:, 4]
        tb = dl[:, 5]
        
        # ════════════════════════════════════════
        # Compute normalized local forces
        # ════════════════════════════════════════
        
        # Axial
        f0 = k_ax * (ua - ub)
        f3 = k_ax * (ub - ua)
        
        # Shear
        f1 = 12 * k_tr * (wa - wb) + 6 * k_sw * (ta + tb)
        f4 = 12 * k_tr * (wb - wa) - 6 * k_sw * (ta + tb)
        
        # Moment
        f2 = 6 * k_sw * (wa - wb) + k_bend * (4 * ta + 2 * tb)
        f5 = 6 * k_sw * (wa - wb) + k_bend * (2 * ta + 4 * tb)
        
        # ════════════════════════════════════════
        # Transform to global (normalized)
        # ════════════════════════════════════════
        
        fg = torch.zeros(E_count, 6, device=device)
        
        fg[:, 0] = c * f0 - s * f1
        fg[:, 1] = s * f0 + c * f1
        fg[:, 2] = f2
        
        fg[:, 3] = c * f3 - s * f4
        fg[:, 4] = s * f3 + c * f4
        fg[:, 5] = f5
        
        # ════════════════════════════════════════
        # Assemble (normalized)
        # ════════════════════════════════════════
        
        nodal_forces_norm = torch.zeros(N_count, 3, device=device)
        nodal_forces_norm.scatter_add_(
            0,
            nA.unsqueeze(1).expand(-1, 3),
            fg[:, 0:3]
        )
        nodal_forces_norm.scatter_add_(
            0,
            nB.unsqueeze(1).expand(-1, 3),
            fg[:, 3:6]
        )
        
        # ════════════════════════════════════════
        # Normalized external forces
        # ════════════════════════════════════════
        
        F_ext_norm = data.F_ext_norm
        
        # ════════════════════════════════════════
        # Denormalize for output (if needed)
        # ════════════════════════════════════════
        
        phys_disp = torch.zeros_like(pred_norm)
        phys_disp[:, 0] = pred_norm[:, 0] * data.u_scale
        phys_disp[:, 1] = pred_norm[:, 1] * data.u_scale
        phys_disp[:, 2] = pred_norm[:, 2] * data.theta_scale
        
        result = {
            # Normalized (for loss)
            'nodal_forces_norm': nodal_forces_norm,
            'F_ext_norm': F_ext_norm,
            
            # Physical (for validation)
            'phys_disp': phys_disp,
            
            # Debug
            'k_ax_range': [k_ax.min().item(), k_ax.max().item()],
            'k_bend_range': [k_bend.min().item(), k_bend.max().item()],
        }
        
        return result