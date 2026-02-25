"""
QUICK DIAGNOSTIC — run this to see what GNN is predicting
"""

import os

os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\PIGNN\PIGNN_V01")
print(f"Working directory: {os.getcwd()}")

import torch
from step_3_model import create_model

# Load data and model
data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
data = data_list[0]

model = create_model('medium')
model.load_state_dict(
    torch.load("DATA/training_logs/best_model.pt", weights_only=False))
model.eval()

with torch.no_grad():
    node_pred, elem_pred = model(data)

print("="*60)
print("PREDICTION vs KRATOS COMPARISON")
print("="*60)

# Node predictions
print("\nDISPLACEMENT (active DOFs: ux=col0, uz=col2):")
print(f"{'Node':>5} {'GNN_ux':>14} {'KRA_ux':>14} {'GNN_uz':>14} {'KRA_uz':>14}")
for i in range(data.num_nodes):
    gnn_ux = node_pred[i, 0].item()
    kra_ux = data.y_node[i, 0].item()
    gnn_uz = node_pred[i, 2].item()
    kra_uz = data.y_node[i, 2].item()
    print(f"{i:>5} {gnn_ux:>14.4e} {kra_ux:>14.4e} "
          f"{gnn_uz:>14.4e} {kra_uz:>14.4e}")

# Element predictions  
print("\nMOMENT My (col 1) and FORCE Fx (col 3):")
print(f"{'Elem':>5} {'GNN_My':>14} {'KRA_My':>14} "
      f"{'GNN_Fx':>14} {'KRA_Fx':>14}")
for e in range(data.n_elements):
    gnn_my = elem_pred[e, 1].item()
    kra_my = data.y_element[e, 1].item()
    gnn_fx = elem_pred[e, 3].item()
    kra_fx = data.y_element[e, 3].item()
    print(f"{e:>5} {gnn_my:>14.4e} {kra_my:>14.4e} "
          f"{gnn_fx:>14.4e} {kra_fx:>14.4e}")

print("\nSENSITIVITY dBM/dI22:")
print(f"{'Elem':>5} {'GNN_sens':>14} {'KRA_sens':>14}")
for e in range(data.n_elements):
    gnn_s = elem_pred[e, 6].item()
    kra_s = data.y_element[e, 6].item()
    print(f"{e:>5} {gnn_s:>14.4e} {kra_s:>14.4e}")