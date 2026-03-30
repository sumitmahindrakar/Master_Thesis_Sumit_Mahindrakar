import torch
from energy_loss import FrameEnergyLoss

import os
from pathlib import Path

from step_2_grapg_constr import FrameData

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)


data_list = torch.load('DATA/graph_dataset_norm.pt', weights_only=False)
loss_fn = FrameEnergyLoss()
g = data_list[0]
u_phys = torch.zeros(g.num_nodes, 3)
u_phys[:, 0] = g.y_node[:, 0]  # true ux
u_phys[:, 1] = g.y_node[:, 1]  # true uz
u_phys[:, 2] = g.y_node[:, 2]  # true θ
U = loss_fn._strain_energy(u_phys, g)
W = loss_fn._external_work(u_phys, g)
print(f'2U/W = {2*U/W:.6f} (should be ~1.0)')
# Boundary condition mask
if hasattr(g, "bc_mask"):
    print(f'bc_mask: True, sum={g.bc_mask.sum().item()}')
else:
    print('bc_mask: False, sum=N/A')
print(f'ux_c={g.ux_c:.4e}, uz_c={g.uz_c:.4e}, ratio={g.ux_c/g.uz_c:.1f}x')