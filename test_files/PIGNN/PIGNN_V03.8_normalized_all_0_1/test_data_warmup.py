# test_data_warmup.py
import torch
from model import PIGNN
from step_5_loss_functions import NormalizedPhysicsLoss

import os
import time
import torch
import numpy as np
from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)


# Load data
data_list = torch.load("DATA/graph_dataset_norm.pt", weights_only=False)
data = data_list[0]

print("="*70)
print("  DATA WARMUP TEST")
print("="*70)

# Check if ground truth exists
print(f"\n1. Ground truth check:")
print(f"   Has y_node_norm: {hasattr(data, 'y_node_norm')}")
if hasattr(data, 'y_node_norm'):
    print(f"   Is None: {data.y_node_norm is None}")
    if data.y_node_norm is not None:
        print(f"   Shape: {data.y_node_norm.shape}")
        print(f"   Range: [{data.y_node_norm.min():.4f}, {data.y_node_norm.max():.4f}]")
        print(f"   ✓ Ground truth available")
    else:
        print(f"   ✗ y_node_norm is None!")
else:
    print(f"   ✗ y_node_norm attribute missing!")

# Test loss function
print(f"\n2. Loss function test:")
model = PIGNN()
loss_fn = NormalizedPhysicsLoss(data_warmup_epochs=200)

# Epoch 0 (should use data)
loss_fn.set_epoch(0)
loss, loss_dict, pred, result = loss_fn(model, data)

print(f"   Epoch 0 (warmup):")
print(f"     alpha:    {loss_dict['alpha']:.3f} (should be 1.0)")
print(f"     L_data:   {loss_dict['L_data']:.4e} (should be > 0)")
print(f"     L_force:  {loss_dict['L_force']:.4e}")
print(f"     L_moment: {loss_dict['L_moment']:.4e}")
print(f"     L_eq:     {loss_dict['L_eq']:.4e}")

# Epoch 200 (should use physics)
loss_fn.set_epoch(200)
loss, loss_dict, pred, result = loss_fn(model, data)

print(f"\n   Epoch 200 (physics):")
print(f"     alpha:    {loss_dict['alpha']:.3f} (should be 0.0)")
print(f"     L_data:   {loss_dict['L_data']:.4e} (should be 0.0)")
print(f"     L_force:  {loss_dict['L_force']:.4e}")
print(f"     L_moment: {loss_dict['L_moment']:.4e}")

print(f"\n3. Physics element test:")
print(f"   F_ext_norm shape: {data.F_ext_norm.shape}")
print(f"   F_ext_norm range: [{data.F_ext_norm.min():.4f}, {data.F_ext_norm.max():.4f}]")
print(f"   Forces ordered shape: {result['F_ext_norm'].shape}")

print(f"\n{'='*70}\n")