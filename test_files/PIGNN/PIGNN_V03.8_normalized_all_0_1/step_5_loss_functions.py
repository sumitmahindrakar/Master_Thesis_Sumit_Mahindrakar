"""
=================================================================
step_5_loss_functions.py — PURE Physics Loss (No Data)
=================================================================
"""

import torch
import torch.nn as nn
from step_4_physics_element import CorotationalBeam2DNormalized


class NormalizedPhysicsLoss(nn.Module):
    """
    Pure physics loss - NO supervised learning.
    Network must satisfy equilibrium from scratch.
    """
    
    def __init__(self, 
                 force_weight=1.0,
                 moment_weight=1.0):
        super().__init__()
        self.beam = CorotationalBeam2DNormalized()
        self.current_epoch = 0
        self.force_weight = force_weight
        self.moment_weight = moment_weight
        self._debug_printed = False
    
    def set_epoch(self, epoch):
        """Update epoch counter"""
        self.current_epoch = epoch
    
    def forward(self, model, data):
        """
        Pure physics loss with enhanced anti-collapse penalties.
        """
        
        # ════════════════════════════════════════════════
        # 1. Network prediction
        # ════════════════════════════════════════════════
        
        pred_norm = model(data)  # (N, 3)
        
        # ════════════════════════════════════════════════
        # 2. Physics computation
        # ════════════════════════════════════════════════
        
        beam_result = self.beam(pred_norm, data)
        
        nodal_forces = beam_result['nodal_forces_norm']
        F_ext = beam_result['F_ext_norm']
        
        residual = nodal_forces - F_ext
        
        # ════════════════════════════════════════════════
        # 3. Auto-scale residuals
        # ════════════════════════════════════════════════
        
        with torch.no_grad():
            res_force_rms = residual[:, :2].pow(2).mean().sqrt().clamp(min=1e-10)
            res_moment_rms = residual[:, 2].pow(2).mean().sqrt().clamp(min=1e-10)
        
        residual_scaled = residual.clone()
        residual_scaled[:, 0] = residual[:, 0] / res_force_rms
        residual_scaled[:, 1] = residual[:, 1] / res_force_rms
        residual_scaled[:, 2] = residual[:, 2] / res_moment_rms
        
        if not self._debug_printed:
            print(f"\n  ═══ PURE PHYSICS MODE (Enhanced) ═══")
            print(f"    No data loss - equilibrium only")
            print(f"    Force weight: {self.force_weight}, Moment weight: {self.moment_weight}")
            print(f"    Residual RMS: Force={res_force_rms:.2e}, Moment={res_moment_rms:.2e}")
            print(f"    Scaled residuals: [{residual_scaled.min():.2f}, {residual_scaled.max():.2f}]")
            print(f"  ═════════════════════════════════════\n")
            self._debug_printed = True
        
        # ════════════════════════════════════════════════
        # 4. Free DOFs
        # ════════════════════════════════════════════════
        
        free_disp = (data.bc_disp.squeeze(-1) < 0.5)
        free_rot = (data.bc_rot.squeeze(-1) < 0.5)
        
        # ════════════════════════════════════════════════
        # 5. Equilibrium loss
        # ════════════════════════════════════════════════
        
        if free_disp.any():
            res_force = residual_scaled[free_disp, :2]
            L_force = res_force.pow(2).mean()
        else:
            L_force = torch.tensor(0.0, device=pred_norm.device)
        
        if free_rot.any():
            res_moment = residual_scaled[free_rot, 2]
            L_moment = res_moment.pow(2).mean()
        else:
            L_moment = torch.tensor(0.0, device=pred_norm.device)
        
        L_physics = self.force_weight * L_force + self.moment_weight * L_moment
        
        # ════════════════════════════════════════════════
        # 6. ENHANCED Anti-Collapse Penalties
        # ════════════════════════════════════════════════
        
        # Penalty 1: Magnitude penalty (prevent small outputs)
        target_magnitude = 0.5
        current_magnitude = pred_norm.abs().mean()
        magnitude_gap = (target_magnitude - current_magnitude).clamp(min=0.0)
        magnitude_penalty = magnitude_gap.pow(2)
        
        # Penalty 2: Variance penalty (prevent constant outputs)
        # Encourages network to use full range of outputs
        output_variance = pred_norm.var()
        target_variance = 0.1
        variance_gap = (target_variance - output_variance).clamp(min=0.0)
        variance_penalty = variance_gap.pow(2)
        
        # Penalty 3: Minimum value penalty (ensure ALL outputs contribute)
        # Prevents network from having some large, some zero
        min_output = pred_norm.abs().min()
        target_min = 0.1
        min_gap = (target_min - min_output).clamp(min=0.0)
        min_penalty = min_gap.pow(2)
        
        # Combined penalty with annealing
        penalty_weight = max(0.0, 1.0 - self.current_epoch / 2000.0)
        
        total_penalty = (
            magnitude_penalty + 
            0.5 * variance_penalty + 
            0.5 * min_penalty
        )
        
        # ════════════════════════════════════════════════
        # 7. Total loss (STRONGER PENALTIES)
        # ════════════════════════════════════════════════
        
        L_total = L_physics + 20.0 * penalty_weight * total_penalty
        
        # ════════════════════════════════════════════════
        # 8. Diagnostics
        # ════════════════════════════════════════════════
        
        with torch.no_grad():
            if free_disp.any():
                max_res = residual[free_disp].abs().max().item()
            else:
                max_res = 0.0
            
            pred_min = pred_norm.min().item()
            pred_max = pred_norm.max().item()
            
            phys_disp = beam_result['phys_disp']
            ux_min = phys_disp[:, 0].min().item()
            ux_max = phys_disp[:, 0].max().item()
            theta_min = phys_disp[:, 2].min().item()
            theta_max = phys_disp[:, 2].max().item()
            
            k_ax_range = beam_result['k_ax_range']
            k_bend_range = beam_result['k_bend_range']
        
        loss_dict = {
            'L_eq': L_total.item(),
            'L_force': L_force.item(),
            'L_moment': L_moment.item(),
            'L_data': 0.0,
            'alpha': 0.0,
            'magnitude_penalty': magnitude_penalty.item(),
            'variance_penalty': variance_penalty.item(),
            'min_penalty': min_penalty.item(),
            'penalty_weight': penalty_weight,
            'current_magnitude': current_magnitude.item(),
            'output_variance': output_variance.item(),
            'min_output': min_output.item(),
            'max_res': max_res,
            'pred_range': [pred_min, pred_max],
            'ux_range': [ux_min, ux_max],
            'theta_range': [theta_min, theta_max],
            'k_ax_range': k_ax_range,
            'k_bend_range': k_bend_range,
        }
        
        
        return L_total, loss_dict, pred_norm, beam_result