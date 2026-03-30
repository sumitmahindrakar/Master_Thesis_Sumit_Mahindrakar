"""
direct_energy_test.py — Verify energy formulation
Optimizes displacement field directly for ONE case.
If this doesn't converge, the energy formula is wrong.
"""
import torch
from energy_loss import FrameEnergyLoss

import os
import time
import torch
import numpy as np
from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)
from step_2_grapg_constr import FrameData

# Load one case
data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
from normalizer import PhysicsScaler
data_list = PhysicsScaler.compute_and_store_list(data_list)
data = data_list[0]

N = data.num_nodes
loss_fn = FrameEnergyLoss()

# True solution for reference
u_true = data.y_node
U_true = loss_fn._strain_energy(u_true, data)
W_true = loss_fn._external_work(u_true, data)
Pi_true = U_true - W_true
print(f"TRUE: U={U_true.item():.6e}, W={W_true.item():.6e}, "
      f"Π={Pi_true.item():.6e}")
print(f"TRUE: U/W = {(U_true/W_true).item():.4f}")

# Direct optimization: optimize raw displacement field
pred_raw = torch.zeros(N, 3, requires_grad=True)
optimizer = torch.optim.Adam([pred_raw], lr=0.01)

# Need a fake "model" that returns pred_raw with BCs
class DirectPredictor(torch.nn.Module):
    def __init__(self, pred_raw, bc_disp, bc_rot):
        super().__init__()
        self.pred_raw = pred_raw
        self.bc_disp = bc_disp
        self.bc_rot = bc_rot
    
    def forward(self, data):
        pred = self.pred_raw.clone()
        pred[:, 0:2] *= (1.0 - self.bc_disp)
        pred[:, 2:3] *= (1.0 - self.bc_rot)
        return pred

fake_model = DirectPredictor(pred_raw, data.bc_disp, data.bc_rot)

for step in range(2000):
    optimizer.zero_grad()
    loss, loss_dict, pr, u_phys = loss_fn(fake_model, data)
    loss.backward()
    optimizer.step()
    
    if step % 200 == 0 or step == 1999:
        print(f"  Step {step:4d}: Π={loss.item():10.6e}, "
              f"U/W={loss_dict['U_over_W']:.4f}, "
              f"raw=[{pr.min().item():.4f}, {pr.max().item():.4f}]")

# Compare with true
print(f"\nFinal pred vs true:")
u_final = u_phys.detach()
for dof, name in enumerate(['ux', 'uz', 'θ']):
    err = (u_final[:, dof] - u_true[:, dof]).pow(2).sum().sqrt()
    ref = u_true[:, dof].pow(2).sum().sqrt().clamp(min=1e-10)
    print(f"  {name}: rel_err = {(err/ref).item():.4e}")