import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from model import PIGNN

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)


# =====================================
# CONFIG
# =====================================

MODEL_PATH = "RESULTS/best.pt"   # or final.pt
DATA_PATH = "DATA/graph_dataset_norm.pt"
RAW_PATH  = "DATA/graph_dataset.pt"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =====================================
# LOAD DATA
# =====================================

data_list = torch.load(DATA_PATH, weights_only=False)
raw_data  = torch.load(RAW_PATH,  weights_only=False)

device = torch.device(DEVICE)


# =====================================
# LOAD MODEL
# =====================================

model = PIGNN(
    node_in_dim=10,
    edge_in_dim=7,
    hidden_dim=128,
    n_layers=6,
).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']}")


# =====================================
# PREDICTION FUNCTION
# =====================================

def predict_physical(model, data):
    data = data.to(device)

    with torch.no_grad():
        pred_raw = model(data)

    # Convert to physical (same as training!)
    pred_phys = torch.zeros_like(pred_raw)
    pred_phys[:, 0] = pred_raw[:, 0] * data.u_c
    pred_phys[:, 1] = pred_raw[:, 1] * data.u_c
    pred_phys[:, 2] = pred_raw[:, 2] * data.theta_c

    return pred_phys.cpu().numpy()


# =====================================
# PLOT FUNCTION
# =====================================

def plot_sample(i):
    data = data_list[i]
    true = raw_data[i].y_node.numpy()

    pred = predict_physical(model, data)

    nodes = np.arange(len(true))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    labels = ['ux', 'uy', 'theta']

    for d in range(3):
        axes[d].plot(nodes, true[:, d], 'k-', label='Kratos')
        axes[d].plot(nodes, pred[:, d], 'r--', label='PIGNN')

        axes[d].set_ylabel(labels[d])
        axes[d].grid(True)

        if d == 0:
            axes[d].legend()

    axes[-1].set_xlabel("Node index")

    plt.suptitle(f"Sample {i}")
    plt.tight_layout()
    plt.show()


# =====================================
# ERROR FUNCTION
# =====================================

def compute_error(i):
    data = data_list[i]
    true = raw_data[i].y_node.numpy()
    pred = predict_physical(model, data)

    err = np.linalg.norm(pred - true)
    ref = np.linalg.norm(true) + 1e-12

    return err / ref


# =====================================
# RUN ON MULTIPLE SAMPLES
# =====================================

if __name__ == "__main__":

    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)
    # Pick samples to inspect
    test_ids = [0, 1, 2, 5, 10]

    for i in test_ids:
        e = compute_error(i)
        print(f"Sample {i} → Rel error: {e:.4e}")

        plot_sample(i)