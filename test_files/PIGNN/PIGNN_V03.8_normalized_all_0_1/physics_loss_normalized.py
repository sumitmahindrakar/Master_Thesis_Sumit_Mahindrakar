"""
=================================================================
physics_loss_normalized.py — Loss in [0,1] Space
=================================================================
"""

import torch
import torch.nn as nn
from corotational_normalized import CorotationalBeam2DNormalized


class NormalizedPhysicsLoss(nn.Module):
    
    def __init__(self, data_warmup_epochs=100):
        super().__init__()
        self.beam = CorotationalBeam2DNormalized()
        self.data_warmup_epochs = data_warmup_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def forward(self, model, data):
        
        pred_norm = model(data)  # Network outputs [0, 1]
        beam_result = self.beam(pred_norm, data)
        
        # Residual in normalized space (all O(1)!)
        res_norm = (beam_result['nodal_forces_norm'] 
                   - beam_result['F_ext_norm'])
        
        free_disp = (data.bc_disp.squeeze(-1) < 0.5)
        free_rot = (data.bc_rot.squeeze(-1) < 0.5)
        
        # ════════════════════════════════════════
        # 1. Equilibrium loss (all O(1))
        # ════════════════════════════════════════
        
        if free_disp.any():
            res_F = res_norm[free_disp, :2]
            L_force = res_F.pow(2).mean()
        else:
            L_force = torch.tensor(0.0, device=pred_norm.device)
        
        if free_rot.any():
            res_M = res_norm[free_rot, 2]
            L_moment = res_M.pow(2).mean()
        else:
            L_moment = torch.tensor(0.0, device=pred_norm.device)
        
        # ════════════════════════════════════════
        # 2. Data warmup (optional)
        # ════════════════════════════════════════
        
        L_data = torch.tensor(0.0, device=pred_norm.device)
        alpha = 0.0
        
        if (self.current_epoch < self.data_warmup_epochs 
                and hasattr(data, 'y_node_norm')
                and data.y_node_norm is not None):
            
            target_norm = data.y_node_norm
            err = (pred_norm - target_norm)[free_disp | free_rot.unsqueeze(1).expand(-1, 3)]
            
            if err.numel() > 0:
                L_data = err.pow(2).mean()
            
            alpha = max(0.0, 1.0 - self.current_epoch / self.data_warmup_epochs)
        
        # ════════════════════════════════════════
        # 3. Combined loss
        # ════════════════════════════════════════
        
        L_physics = L_force + L_moment
        L_total = alpha * L_data + (1.0 - alpha) * L_physics
        
        # ════════════════════════════════════════
        # Diagnostics
        # ════════════════════════════════════════
        
        loss_dict = {
            'L_eq': L_total.item(),
            'L_force': L_force.item(),
            'L_moment': L_moment.item(),
            'L_data': L_data.item(),
            'alpha': alpha,
            'pred_range': [pred_norm.min().item(), pred_norm.max().item()],
            'k_ax_range': beam_result['k_ax_range'],
            'k_bend_range': beam_result['k_bend_range'],
            'max_res': res_norm[free_disp].abs().max().item() if free_disp.any() else 0.0,
        }
        
        return L_total, loss_dict, pred_norm, beam_result