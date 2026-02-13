"""
TEST STEP 11: Verify train.py is created correctly and can run
"""
import os
import sys

print("=" * 70)
print("TEST STEP 11: Main Training Script (train.py) Check")
print("=" * 70)

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print(f"\nProject directory: {script_dir}")

all_passed = True

# Check if train.py exists
print("\n" + "-" * 70)
print("Checking train.py exists:")
print("-" * 70)

train_path = os.path.join(script_dir, 'train.py')

if os.path.exists(train_path):
    size = os.path.getsize(train_path)
    print(f"  ✓ train.py exists ({size} bytes)")
    if size < 5000:
        print(f"    ⚠ Warning: File seems too small. Did you add the full code?")
        all_passed = False
else:
    print(f"  ✗ train.py MISSING!")
    all_passed = False
    print("\n" + "=" * 70)
    print("STEP 11 FAILED! ✗")
    print("Please create train.py with the provided code.")
    print("=" * 70)
    sys.exit(1)

# Check syntax by importing
print("\n" + "-" * 70)
print("Checking syntax:")
print("-" * 70)

try:
    import ast
    with open(train_path, 'r') as f:
        source = f.read()
    ast.parse(source)
    print("  ✓ train.py has valid Python syntax")
except SyntaxError as e:
    print(f"  ✗ Syntax error in train.py: {e}")
    all_passed = False

# Check all required imports work
print("\n" + "-" * 70)
print("Checking required imports:")
print("-" * 70)

import_checks = [
    ("import torch", "torch"),
    ("import torch.nn as nn", "torch.nn"),
    ("from torch_geometric.loader import DataLoader", "DataLoader"),
    ("import numpy as np", "numpy"),
    ("from argparse import ArgumentParser", "ArgumentParser"),
    ("from GNN import Structure_GraphNetwork", "Structure_GraphNetwork"),
    ("from GNN import L1_Loss", "L1_Loss"),
    ("from Utils import get_dataset", "get_dataset"),
    ("from Utils import normalize_dataset", "normalize_dataset"),
    ("from Utils import node_accuracy", "node_accuracy"),
    ("from Utils import plot_learningCurve", "plot_learningCurve"),
]

for import_statement, name in import_checks:
    try:
        exec(import_statement)
        print(f"  ✓ {import_statement}")
    except ImportError as e:
        print(f"  ✗ {import_statement}: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ {import_statement}: {e}")
        all_passed = False

# Check all components work together
print("\n" + "-" * 70)
print("Testing component integration:")
print("-" * 70)

try:
    import torch
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data
    
    from GNN import Structure_GraphNetwork, L1_Loss
    from Utils import get_target_index, node_accuracy
    
    # Create mock data
    def create_mock_graph(num_nodes=10, seed=42):
        torch.manual_seed(seed)
        x = torch.randn(num_nodes, 15)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_attr = torch.randn(num_nodes * 2, 3)
        y = torch.randn(num_nodes, 38)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # Create mock dataset
    mock_dataset = [create_mock_graph(seed=i) for i in range(10)]
    print(f"  ✓ Created mock dataset with {len(mock_dataset)} samples")
    
    # Create data loader
    loader = DataLoader(mock_dataset, batch_size=2, shuffle=True)
    print(f"  ✓ Created DataLoader")
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Structure_GraphNetwork(
        layer_num=3,
        input_dim=15,
        hidden_dim=64,
        edge_attr_dim=3,
        aggr='mean',
        device=device
    ).to(device)
    print(f"  ✓ Created model on {device}")
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = L1_Loss()
    print(f"  ✓ Created optimizer and loss function")
    
    # Get target indices
    y_start, y_end = get_target_index('all')
    print(f"  ✓ Target indices: [{y_start}:{y_end}]")
    
    # Run one training step
    model.train()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        output = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(output[:, y_start:y_end], batch.y[:, y_start:y_end], 1e-4)
        loss.backward()
        optimizer.step()
        
        print(f"  ✓ Training step successful")
        print(f"    Batch size: {batch.x.shape[0]} nodes")
        print(f"    Output shape: {output.shape}")
        print(f"    Loss: {loss.item():.6f}")
        break
    
    # Run evaluation
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.edge_attr)
            
            correct, count = node_accuracy(
                output[:, y_start:y_end],
                batch.y[:, y_start:y_end],
                1e-4
            )
            accuracy = (correct / count).item() if count > 0 else 0
            
            print(f"  ✓ Evaluation step successful")
            print(f"    Accuracy: {accuracy:.4f}")
            break
    
    print(f"\n  ✓ All components work together!")
    
except Exception as e:
    print(f"  ✗ Integration test failed: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# Check Data directory
print("\n" + "-" * 70)
print("Checking Data directory:")
print("-" * 70)

data_dir = os.path.join(script_dir, 'Data')
if os.path.exists(data_dir):
    print(f"  ✓ Data/ directory exists")
    
    # Look for dataset folders
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if subdirs:
        print(f"  ✓ Found dataset folders: {subdirs}")
        
        # Check for structure folders
        for dataset_name in subdirs[:1]:  # Check first dataset
            dataset_path = os.path.join(data_dir, dataset_name)
            structures = [d for d in os.listdir(dataset_path) if d.startswith('structure_')]
            if structures:
                print(f"  ✓ Found {len(structures)} structure folders in {dataset_name}/")
                
                # Check for .pt files
                first_struct = os.path.join(dataset_path, structures[0])
                pt_files = [f for f in os.listdir(first_struct) if f.endswith('.pt')]
                if pt_files:
                    print(f"  ✓ Found .pt files: {pt_files}")
                else:
                    print(f"  ⚠ No .pt files found in {structures[0]}/")
            else:
                print(f"  ⚠ No structure_* folders found in {dataset_name}/")
    else:
        print(f"  ⚠ No dataset folders found in Data/")
        print(f"    You need to add your data before training!")
else:
    print(f"  ⚠ Data/ directory does not exist")
    print(f"    You need to add your data before training!")

# Check Results directory
print("\n" + "-" * 70)
print("Checking Results directory:")
print("-" * 70)

results_dir = os.path.join(script_dir, 'Results')
if os.path.exists(results_dir):
    print(f"  ✓ Results/ directory exists")
else:
    os.makedirs(results_dir, exist_ok=True)
    print(f"  ✓ Created Results/ directory")

# Display usage instructions
print("\n" + "-" * 70)
print("How to Use train.py:")
print("-" * 70)
print("""
  BASIC USAGE:
  ────────────────────────────────────────────────────────────────
  
  # Train with default settings
  python train.py
  
  # Train with more epochs
  python train.py --epoch_num 100
  
  # Train with different model
  python train.py --model Structure_GCN
  
  # Train with custom settings
  python train.py --epoch_num 50 --hidden_dim 128 --lr 1e-4 --layer_num 5
  
  ────────────────────────────────────────────────────────────────
  
  AVAILABLE ARGUMENTS:
  ────────────────────────────────────────────────────────────────
  
  Dataset:
    --dataset_name      Name of dataset folder (default: Static_Linear_Analysis)
    --data_num          Number of structures to load (default: 100)
    --train_ratio       Train/validation split ratio (default: 0.9)
  
  Model:
    --model             Model type: Structure_GraphNetwork, Structure_GCN,
                        Structure_GAT, Structure_GIN
    --hidden_dim        Hidden layer dimension (default: 256)
    --layer_num         Number of GNN layers (default: 9)
    --aggr              Aggregation: mean, sum, max (default: mean)
  
  Training:
    --epoch_num         Number of epochs (default: 10)
    --batch_size        Batch size (default: 1)
    --lr                Learning rate (default: 5e-5)
    --loss_function     Loss: L1_Loss, L2_Loss (default: L1_Loss)
    --target            Output target: all, disp, moment, shear (default: all)
""")

# Summary
print("\n" + "=" * 70)
if all_passed:
    print("STEP 11 PASSED! ✓")
    print("=" * 70)
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║   🎉 CONGRATULATIONS! PROJECT SETUP COMPLETE! 🎉                ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    Your complete project structure:
    
    structGNN_reproduced/
    ├── GNN/
    │   ├── __init__.py         ✓
    │   ├── layers.py           ✓  (MLP, GraphNetwork_layer)
    │   ├── losses.py           ✓  (L1_Loss, L2_Loss)
    │   └── models.py           ✓  (Structure_GraphNetwork, GCN, GAT, GIN)
    │
    ├── Utils/
    │   ├── __init__.py         ✓
    │   ├── accuracy.py         ✓  (node_accuracy)
    │   ├── datasets.py         ✓  (get_dataset, split_dataset)
    │   ├── normalization.py    ✓  (normalize_dataset, denormalize)
    │   └── plot.py             ✓  (plot_learningCurve, plot_lossCurve)
    │
    ├── Data/                        (add your .pt files here)
    │   └── Static_Linear_Analysis/
    │       ├── structure_1/
    │       │   └── structure_graph_NodeAsNode.pt
    │       └── ...
    │
    ├── Results/                     (training results saved here)
    │
    └── train.py                ✓  (main training script)
    
    ══════════════════════════════════════════════════════════════════
    
    NEXT STEPS:
    
    1. Add your data files to Data/Static_Linear_Analysis/
    
    2. Run training:
       python train.py --epoch_num 10
    
    3. Check results in Results/ folder
    
    ══════════════════════════════════════════════════════════════════
    """)
else:
    print("STEP 11 FAILED! ✗")
    print("=" * 70)
    print("\nPlease fix the errors above and run this test again.")
print("=" * 70)