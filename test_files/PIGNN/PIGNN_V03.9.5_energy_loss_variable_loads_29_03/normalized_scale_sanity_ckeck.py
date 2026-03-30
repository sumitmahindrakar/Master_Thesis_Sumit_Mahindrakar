# import torch
# from energy_loss import FrameEnergyLoss
# import os
# from pathlib import Path

# from step_2_grapg_constr import FrameData

# CURRENT_SUBFOLDER = Path(__file__).resolve().parent
# os.chdir(CURRENT_SUBFOLDER)


# data_list = torch.load('DATA/graph_dataset_norm.pt', weights_only=False)
# loss_fn = FrameEnergyLoss()
# g = data_list[0]

# # Check non-dim energy with TRUE displacements
# pred_true = torch.zeros(g.num_nodes, 3)
# pred_true[:, 0] = g.y_node[:, 0] / g.ux_c
# pred_true[:, 1] = g.y_node[:, 1] / g.uz_c
# pred_true[:, 2] = g.y_node[:, 2] / g.theta_c

# Pi_true = loss_fn._nondim_energy(pred_true, g)
# U = loss_fn._strain_energy(g.y_node, g)
# W = loss_fn._external_work(g.y_node, g)
# E_c = g.F_c * g.ux_c
# Pi_check = (U - W) / E_c

# print(f'Pi_nondim (direct):    {Pi_true:.6e}')
# print(f'Pi_nondim (U-W)/E_c:   {Pi_check:.6e}')
# print(f'Match: {abs(Pi_true - Pi_check) / abs(Pi_check) < 0.01}')
# print(f'pred_true range: [{pred_true.min():.3f}, {pred_true.max():.3f}] (should be ~O(1))')
# print(f'BC enforced: pred at supports = {pred_true[g.bc_mask].abs().max():.2e}')

import torch
import os
from pathlib import Path
CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

from step_2_grapg_constr import FrameData


data_list = torch.load('DATA/graph_dataset_norm.pt', weights_only=False)
g = data_list[0]

# Which nodes does bc_mask select?
bc_nodes = g.bc_mask.nonzero(as_tuple=True)[0].tolist()
print(f'bc_mask nodes: {bc_nodes}')
print(f'bc_mask sum: {g.bc_mask.sum()}')

# What are their coordinates?
for n in bc_nodes:
    coords = g.coords[n]
    y_node = g.y_node[n]
    print(f'  Node {n}: pos=({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})')
    print(f'    y_node = [{y_node[0]:.6e}, {y_node[1]:.6e}, {y_node[2]:.6e}]')
    print(f'    normalized = [{y_node[0]/g.ux_c:.4f}, {y_node[1]/g.uz_c:.4f}, {y_node[2]/g.theta_c:.4f}]')

# What about bc_disp raw values?
print(f'\nbc_disp shape: {g.bc_disp.shape}')
print(f'bc_disp nonzero: {(g.bc_disp.squeeze() > 0.5).nonzero(as_tuple=True)[0].tolist()}')

# Find actual zero-displacement nodes
disp_mag = g.y_node.abs().sum(dim=1)
zero_nodes = (disp_mag < 1e-10).nonzero(as_tuple=True)[0].tolist()
small_nodes = (disp_mag < 1e-6).nonzero(as_tuple=True)[0].tolist()
print(f'\nNodes with |u| < 1e-10: {zero_nodes}')
print(f'Nodes with |u| < 1e-6:  {small_nodes}')

# Show smallest displacement nodes
_, sorted_idx = disp_mag.sort()
print(f'\n5 smallest displacement nodes:')
for i in range(min(5, len(sorted_idx))):
    n = sorted_idx[i].item()
    print(f'  Node {n}: |u|={disp_mag[n]:.6e}  pos=({g.coords[n,0]:.1f}, {g.coords[n,2]:.1f})  y={g.y_node[n].tolist()}')

# Check F_ext at support nodes
print(f'\nF_ext at bc_mask nodes:')
for n in bc_nodes:
    print(f'  Node {n}: F_ext = {g.F_ext[n].tolist()}')



