# test_physics_loss.py
import torch
from model import FramePIGNN
from physics_loss import PhysicsLoss

import os

print(f"Working directory: {os.getcwd()}")
os.chdir(r"test_files/PIGNN/PIGNN_V02")#test_files\PIGNN\PIGNN_V02"
print(f"Working directory: {os.getcwd()}")

# Load RAW (not normalized) graphs — physics needs real units
data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
sample = data_list[0]

model = FramePIGNN()
loss_fn = PhysicsLoss()

model.eval()
pred = model(sample)
loss = loss_fn(pred, sample)

print(f"pred shape: {pred.shape}")
print(f"physics loss: {loss.item():.4e}")