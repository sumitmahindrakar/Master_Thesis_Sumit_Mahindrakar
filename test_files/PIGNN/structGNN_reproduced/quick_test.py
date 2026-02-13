"""
Quick test to verify everything works before training.
"""
import torch
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print("=" * 60)
print("QUICK TEST - VERIFYING SETUP")
print("=" * 60)

# Test 1: Load data
print("\n[1] Loading data...")
from Utils import get_dataset, normalize_dataset, split_dataset

dataset = get_dataset(dataset_name='Static_Linear_Analysis', structure_num=10)
print(f"    ✓ Loaded {len(dataset)} structures")
print(f"    Node features: {dataset[0].x.shape}")
print(f"    Edge features: {dataset[0].edge_attr.shape}")
print(f"    Output features: {dataset[0].y.shape}")

# Test 2: Normalize
print("\n[2] Normalizing data...")
dataset, norm_dict = normalize_dataset(dataset)
print(f"    x range after: [{dataset[0].x.min():.4f}, {dataset[0].x.max():.4f}]")
print(f"    y range after: [{dataset[0].y.min():.4f}, {dataset[0].y.max():.4f}]")

# Test 3: Split
print("\n[3] Splitting data...")
train_data, valid_data, _ = split_dataset(dataset, train_ratio=0.8)
print(f"    Train: {len(train_data)}, Valid: {len(valid_data)}")

# Test 4: Create model
print("\n[4] Creating model...")
from GNN import Structure_GraphNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"
input_dim = dataset[0].x.shape[1]  # 11
edge_attr_dim = dataset[0].edge_attr.shape[1]  # 3

model = Structure_GraphNetwork(
    layer_num=3,
    input_dim=input_dim,
    hidden_dim=64,
    edge_attr_dim=edge_attr_dim,
    aggr='mean',
    device=device
).to(device)

print(f"    ✓ Model created on {device}")
print(f"    Input dim: {input_dim}, Edge dim: {edge_attr_dim}")
print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test 5: Forward pass
print("\n[5] Testing forward pass...")
sample = dataset[0].to(device)
with torch.no_grad():
    output = model(sample.x, sample.edge_index, sample.edge_attr)
print(f"    Input: {sample.x.shape}")
print(f"    Output: {output.shape}")
print(f"    ✓ Forward pass successful!")

# Test 6: Loss and accuracy
print("\n[6] Testing loss and accuracy...")
from GNN import L1_Loss
from Utils import node_accuracy, get_target_index

criterion = L1_Loss()
y_start, y_end = get_target_index('all')

loss = criterion(output[:, y_start:y_end], sample.y[:, y_start:y_end], 1e-4)
correct, count = node_accuracy(output[:, y_start:y_end], sample.y[:, y_start:y_end], 1e-4)

print(f"    Loss: {loss.item():.4f}")
print(f"    Accuracy elements: {count}")
print(f"    ✓ Loss and accuracy work!")

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print("\nYou can now run training:")
print("  python train.py --data_num 100 --epoch_num 10")
print("=" * 60)