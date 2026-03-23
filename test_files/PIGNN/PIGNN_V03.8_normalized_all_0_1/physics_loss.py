"""
=================================================================
physics_loss.py — Physics Loss for Normalized [0,1] Data
=================================================================
Works with data normalized by CompleteNormalizer.
All computations in [0,1] space for numerical stability.
=================================================================
"""

import torch
import torch.nn as nn
from corotational import CorotationalBeam2D


class CorotationalPhysicsLoss(nn.Module):
    """
    Physics-informed loss for normalized data.
    
    Expects:
      - Network outputs in [0, 1] range
      - Data normalized by CompleteNormalizer
      - Physics scales (u_scale, F_scale, etc.) stored in data
    """
    
    def __init__(self, 
                 data_warmup_epochs=0,
                 force_weight=1.0,
                 moment_weight=1.0):
        super().__init__()
        self.beam = CorotationalBeam2D()
        self.data_warmup_epochs = data_warmup_epochs
        self.current_epoch = 0
        self.force_weight = force_weight
        self.moment_weight = moment_weight
    
    def set_epoch(self, epoch):
        """Set current epoch for warmup scheduling"""
        self.current_epoch = epoch
    
    def forward(self, model, data):
        """
        Args:
            model: PIGNN model
            data: PyG Data object with normalized features
        
        Returns:
            total_loss: scalar loss
            loss_dict: dict with components
            pred_raw: network output
            beam_result: physics computation result
        """
        
        # ════════════════════════════════════════════════
        # 1. Get network prediction (should be ~[0,1])
        # ════════════════════════════════════════════════
        
        pred_raw = model(data)  # (N, 3): [ux, uz, θ]
        
        # ════════════════════════════════════════════════
        # 2. Ensure data has required scales
        # ════════════════════════════════════════════════
        
        if not hasattr(data, 'u_c'):
            # Add scales from denormalization parameters
            if hasattr(data, 'u_scale'):
                data.u_c = data.u_scale
                data.theta_c = data.theta_scale
                data.F_c = data.F_scale
                data.M_c = data.M_scale
            else:
                # Fallback defaults (assumes already O(1))
                data.u_c = torch.tensor(1.0, device=pred_raw.device)
                data.theta_c = torch.tensor(1.0, device=pred_raw.device)
                data.F_c = torch.tensor(1.0, device=pred_raw.device)
                data.M_c = torch.tensor(1.0, device=pred_raw.device)
        
        # ════════════════════════════════════════════════
        # 3. Compute physics (corotational element)
        # ════════════════════════════════════════════════
        
        beam_result = self.beam(pred_raw, data)
        
        # Extract forces (already scaled appropriately)
        nodal_forces = beam_result['nodal_forces_nd']  # (N, 3)
        F_ext = beam_result['F_ext_nd']  # (N, 3)
        
        # Residual: internal forces - external forces
        residual = nodal_forces - F_ext  # (N, 3)
        
        # ════════════════════════════════════════════════
        # 4. Get free DOFs
        # ════════════════════════════════════════════════
        
        free_disp = (data.bc_disp.squeeze(-1) < 0.5)  # (N,)
        free_rot = (data.bc_rot.squeeze(-1) < 0.5)    # (N,)
        
        # ════════════════════════════════════════════════
        # 5. Compute equilibrium loss
        # ════════════════════════════════════════════════
        
        # Force equilibrium (Fx, Fz)
        if free_disp.any():
            res_force = residual[free_disp, :2]  # (M, 2)
            L_force = res_force.pow(2).mean()
        else:
            L_force = torch.tensor(0.0, device=pred_raw.device)
        
        # Moment equilibrium (My)
        if free_rot.any():
            res_moment = residual[free_rot, 2]  # (M,)
            L_moment = res_moment.pow(2).mean()
        else:
            L_moment = torch.tensor(0.0, device=pred_raw.device)
        
        # ════════════════════════════════════════════════
        # 6. Optional: Data warmup loss
        # ════════════════════════════════════════════════
        
        L_data = torch.tensor(0.0, device=pred_raw.device)
        alpha = 0.0
        
        if (self.current_epoch < self.data_warmup_epochs 
                and hasattr(data, 'y_node')
                and data.y_node is not None):
            
            # Convert ground truth to normalized form
            target_norm = torch.zeros_like(pred_raw)
            target_norm[:, 0] = data.y_node[:, 0] / data.u_c
            target_norm[:, 1] = data.y_node[:, 1] / data.u_c
            target_norm[:, 2] = data.y_node[:, 2] / data.theta_c
            
            # MSE on free DOFs
            free_mask = free_disp.unsqueeze(1).expand(-1, 3) | free_rot.unsqueeze(1).expand(-1, 3)
            err = (pred_raw - target_norm)[free_mask]
            
            if err.numel() > 0:
                L_data = err.pow(2).mean()
            
            # Warmup schedule: linear decay from 1.0 to 0.0
            alpha = max(0.0, 1.0 - self.current_epoch / self.data_warmup_epochs)
        
        # ════════════════════════════════════════════════
        # 7. Combined loss
        # ════════════════════════════════════════════════
        
        L_physics = self.force_weight * L_force + self.moment_weight * L_moment
        L_total = alpha * L_data + (1.0 - alpha) * L_physics
        
        # ════════════════════════════════════════════════
        # 8. Diagnostics
        # ════════════════════════════════════════════════
        
        with torch.no_grad():
            # Max residual for monitoring
            if free_disp.any():
                max_res_nd = residual[free_disp].abs().max().item()
            else:
                max_res_nd = 0.0
            
            # Prediction ranges
            raw_min = pred_raw.min().item()
            raw_max = pred_raw.max().item()
            
            ux_min = (pred_raw[:, 0] * data.u_c).min().item()
            ux_max = (pred_raw[:, 0] * data.u_c).max().item()
            
            th_min = (pred_raw[:, 2] * data.theta_c).min().item()
            th_max = (pred_raw[:, 2] * data.theta_c).max().item()
        
        loss_dict = {
            'L_eq': L_total.item(),
            'L_force': L_force.item(),
            'L_moment': L_moment.item(),
            'L_data': L_data.item(),
            'alpha': alpha,
            'max_res_nd': max_res_nd,
            'raw_range': [raw_min, raw_max],
            'ux_range': [ux_min, ux_max],
            'th_range': [th_min, th_max],
        }
        
        return L_total, loss_dict, pred_raw, beam_result


# ════════════════════════════════════════════════════════════
# VERIFICATION
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    from pathlib import Path
    
    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)
    
    print("="*70)
    print("  PHYSICS LOSS VERIFICATION")
    print("="*70)
    
    # Load data
    data_list = torch.load("DATA/graph_dataset_norm.pt", weights_only=False)
    
    # Check if scales present
    d = data_list[0]
    print(f"\nData attributes:")
    print(f"  Has u_scale: {hasattr(d, 'u_scale')}")
    print(f"  Has F_scale: {hasattr(d, 'F_scale')}")
    print(f"  Has u_c: {hasattr(d, 'u_c')}")
    print(f"  Has F_c: {hasattr(d, 'F_c')}")
    
    if hasattr(d, 'u_scale'):
        print(f"\nDenormalization scales:")
        print(f"  u_scale:     {d.u_scale:.4e}")
        print(f"  theta_scale: {d.theta_scale:.4e}")
        print(f"  F_scale:     {d.F_scale:.4e}")
    
    # Create model and loss
    from model import PIGNN
    model = PIGNN()
    loss_fn = CorotationalPhysicsLoss(data_warmup_epochs=100)
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    data = data_list[0]
    
    try:
        loss_fn.set_epoch(0)  # Warmup epoch
        loss, loss_dict, pred, beam_result = loss_fn(model, data)
        
        print(f"✓ Forward pass successful")
        print(f"\nLoss components:")
        print(f"  L_total:  {loss_dict['L_eq']:.4e}")
        print(f"  L_force:  {loss_dict['L_force']:.4e}")
        print(f"  L_moment: {loss_dict['L_moment']:.4e}")
        print(f"  L_data:   {loss_dict['L_data']:.4e}")
        print(f"  alpha:    {loss_dict['alpha']:.2f}")
        
        print(f"\nPrediction ranges:")
        print(f"  raw:   {loss_dict['raw_range']}")
        print(f"  ux:    {loss_dict['ux_range']}")
        print(f"  theta: {loss_dict['th_range']}")
        
        # Test gradient
        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"\n✓ Gradient norm: {grad_norm:.4e}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"  VERIFICATION COMPLETE")
    print(f"{'='*70}\n")