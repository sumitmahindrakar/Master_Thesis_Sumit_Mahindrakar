"""
=================================================================
step_4_physics_element.py — Corotational Element in [0,1] Space
=================================================================
All quantities in normalized [0, 1] range:
  - Geometry: coords_norm, L_norm
  - Material: E_norm, A_norm, I_norm
  - Displacements: u_norm, θ_norm
  - Forces: F_norm, M_norm

Stiffness coefficients naturally O(1)!
=================================================================
"""

import torch
import torch.nn as nn


class CorotationalBeam2DNormalized(nn.Module):
    """
    2D corotational beam element with normalized quantities.
    
    Key insight: In [0,1] space, stiffness becomes:
        k_norm = (E_norm * A_norm) / L_norm
    
    Since E, A, L ∈ [0,1], k_norm ∈ [0, ∞) but typically O(0.1-10)
    which is MUCH better than O(10⁷)!
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
            result dict with:
              - nodal_forces_norm (N, 3): internal forces in normalized space
              - F_ext_norm (N, 3): external forces in normalized space
              - phys_disp (N, 3): denormalized displacements (for validation)
        """
        
        conn = data.connectivity
        coords_norm = data.coords_norm
        E_norm = data.prop_E_norm
        A_norm = data.prop_A_norm
        I_norm = data.prop_I22_norm
        
        E_count = conn.shape[0]
        N_count = pred_norm.shape[0]
        device = pred_norm.device
        
        nA = conn[:, 0]
        nB = conn[:, 1]
        
        # ════════════════════════════════════════════════
        # 1. Geometry (in normalized space)
        # ════════════════════════════════════════════════
        
        dx0 = coords_norm[nB, 0] - coords_norm[nA, 0]
        dz0 = coords_norm[nB, 2] - coords_norm[nA, 2]
        l0_norm = torch.sqrt(dx0**2 + dz0**2 + 1e-10)
        
        c = dx0 / (l0_norm + 1e-10)
        s = dz0 / (l0_norm + 1e-10)
        
        # ════════════════════════════════════════════════
        # 2. Normalized stiffness coefficients
        # ════════════════════════════════════════════════
        #
        # In [0,1] space:
        #   k = (E_norm * A_norm) / L_norm
        #
        # Since all quantities ∈ [0,1], k is naturally well-scaled!
        
        EA_norm = E_norm * A_norm
        EI_norm = E_norm * I_norm
        
        # Axial stiffness
        k_ax = EA_norm / (l0_norm + 1e-10)
        
        # Bending stiffness
        k_bend = EI_norm / (l0_norm + 1e-10)
        
        # Shear-rotation coupling
        k_sw = EI_norm / (l0_norm.pow(2) + 1e-10)
        
        # Transverse stiffness
        k_tr = EI_norm / (l0_norm.pow(3) + 1e-10)
        
        # ════════════════════════════════════════════════
        # 3. Gather normalized DOFs with -θ
        # ════════════════════════════════════════════════
        
        d_norm = torch.stack([
            pred_norm[nA, 0],         # ux_A (norm)
            pred_norm[nA, 1],         # uz_A (norm)
            -pred_norm[nA, 2],        # -θ_A (norm)
            pred_norm[nB, 0],         # ux_B (norm)
            pred_norm[nB, 1],         # uz_B (norm)
            -pred_norm[nB, 2],        # -θ_B (norm)
        ], dim=1)  # (E, 6)
        
        # ════════════════════════════════════════════════
        # 4. Transform to local coordinates
        # ════════════════════════════════════════════════
        
        dl = torch.zeros_like(d_norm)
        
        dl[:, 0] =  c * d_norm[:, 0] + s * d_norm[:, 1]
        dl[:, 1] = -s * d_norm[:, 0] + c * d_norm[:, 1]
        dl[:, 2] =  d_norm[:, 2]
        
        dl[:, 3] =  c * d_norm[:, 3] + s * d_norm[:, 4]
        dl[:, 4] = -s * d_norm[:, 3] + c * d_norm[:, 4]
        dl[:, 5] =  d_norm[:, 5]
        
        ua = dl[:, 0]
        wa = dl[:, 1]
        ta = dl[:, 2]
        ub = dl[:, 3]
        wb = dl[:, 4]
        tb = dl[:, 5]
        
        # ════════════════════════════════════════════════
        # 5. Compute normalized local forces
        # ════════════════════════════════════════════════
        
        # Axial
        f0 = k_ax * (ua - ub)
        f3 = k_ax * (ub - ua)
        
        # Shear
        f1 = 12.0 * k_tr * (wa - wb) + 6.0 * k_sw * (ta + tb)
        f4 = 12.0 * k_tr * (wb - wa) - 6.0 * k_sw * (ta + tb)
        
        # Moment
        f2 = 6.0 * k_sw * (wa - wb) + k_bend * (4.0 * ta + 2.0 * tb)
        f5 = 6.0 * k_sw * (wa - wb) + k_bend * (2.0 * ta + 4.0 * tb)
        
        # ════════════════════════════════════════════════
        # 6. Transform to global (normalized)
        # ════════════════════════════════════════════════
        
        fg = torch.zeros(E_count, 6, device=device)
        
        fg[:, 0] = c * f0 - s * f1
        fg[:, 1] = s * f0 + c * f1
        fg[:, 2] = f2
        
        fg[:, 3] = c * f3 - s * f4
        fg[:, 4] = s * f3 + c * f4
        fg[:, 5] = f5
        
        # ════════════════════════════════════════════════
        # 7. Assemble (normalized)
        # ════════════════════════════════════════════════
        
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
        
        # ════════════════════════════════════════════════
        # 8. External forces (already normalized)
        # ════════════════════════════════════════════════
        
        # F_ext_norm = data.F_ext_norm  # (N, 3): [Fx, My, Fz] in [0,1]
        
        # # Rearrange to match nodal_forces_norm: [Fx, Fz, My]
        # F_ext_norm_ordered = torch.zeros(N_count, 3, device=device)
        # F_ext_norm_ordered[:, 0] = F_ext_norm[:, 0]  # Fx
        # F_ext_norm_ordered[:, 1] = F_ext_norm[:, 2]  # Fz
        # F_ext_norm_ordered[:, 2] = F_ext_norm[:, 1]  # My
        F_ext_norm_ordered = data.F_ext_norm.clone()
        
        # ════════════════════════════════════════════════
        # 9. Denormalize for output (validation only)
        # ════════════════════════════════════════════════
        
        phys_disp = torch.zeros_like(pred_norm)
        phys_disp[:, 0] = pred_norm[:, 0] * data.u_scale
        phys_disp[:, 1] = pred_norm[:, 1] * data.u_scale
        phys_disp[:, 2] = pred_norm[:, 2] * data.theta_scale
        
        # ════════════════════════════════════════════════
        # Result
        # ════════════════════════════════════════════════
        
        result = {
            # Normalized (for loss) - all O(1)
            'nodal_forces_norm': nodal_forces_norm,      # (N, 3)
            'F_ext_norm': F_ext_norm_ordered,             # (N, 3)
            
            # Physical (for validation)
            'phys_disp': phys_disp,                       # (N, 3)
            
            # Debug info
            'k_ax_range': [k_ax.min().item(), k_ax.max().item()],
            'k_bend_range': [k_bend.min().item(), k_bend.max().item()],
            'l0_norm_range': [l0_norm.min().item(), l0_norm.max().item()],
        }
        
        return result


# ════════════════════════════════════════════════════════════
# VERIFICATION
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    from pathlib import Path
    
    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)
    
    print("="*70)
    print("  NORMALIZED COROTATIONAL ELEMENT VERIFICATION")
    print("="*70)
    
    # Load normalized data
    data_list = torch.load("DATA/graph_dataset_norm.pt", weights_only=False)
    
    beam = CorotationalBeam2DNormalized()
    
    for i in range(min(3, len(data_list))):
        data = data_list[i]
        
        print(f"\n  Case {i}:")
        
        # Use normalized ground truth as test input
        if hasattr(data, 'y_node_norm') and data.y_node_norm is not None:
            pred_norm = data.y_node_norm.clone()
            
            print(f"    Input range:  [{pred_norm.min():.4f}, {pred_norm.max():.4f}]")
            
            result = beam(pred_norm, data)
            
            # Check stiffness ranges
            print(f"    k_ax:   {result['k_ax_range']}")
            print(f"    k_bend: {result['k_bend_range']}")
            
            # Equilibrium check
            res_norm = result['nodal_forces_norm'] - result['F_ext_norm']
            free = data.bc_disp.squeeze(-1) < 0.5
            
            if free.any():
                max_res = res_norm[free].abs().max().item()
                print(f"    Max|R_norm|: {max_res:.6e}")
                
                status = "✓ GOOD" if max_res < 0.1 else "~ OK" if max_res < 1.0 else "✗ CHECK"
                print(f"    Status: {status}")
            
            # Roundtrip check
            if data.y_node is not None:
                phys_err = (result['phys_disp'] - data.y_node).abs().max().item()
                print(f"    Phys roundtrip: {phys_err:.6e}")
        else:
            print(f"    ⚠ No ground truth available")
    
    # Gradient test
    print(f"\n  Gradient test:")
    data = data_list[0]
    if hasattr(data, 'y_node_norm') and data.y_node_norm is not None:
        pred_norm = data.y_node_norm.clone()
        pred_norm.requires_grad_(True)
        
        result = beam(pred_norm, data)
        res = result['nodal_forces_norm'] - result['F_ext_norm']
        free = data.bc_disp.squeeze(-1) < 0.5
        loss = res[free].pow(2).mean()
        
        grad = torch.autograd.grad(loss, pred_norm)[0]
        print(f"    Loss:       {loss:.6e}")
        print(f"    Grad range: [{grad.min():.4e}, {grad.max():.4e}]")
        print(f"    Grad norm:  {grad.norm():.4e}")
        print(f"    ✓ Differentiable")
    
    print(f"\n{'='*70}")
    print(f"  VERIFICATION COMPLETE ✓")
    print(f"{'='*70}\n")