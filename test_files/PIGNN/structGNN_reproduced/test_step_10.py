"""
TEST STEP 10: Verify __init__.py files are updated correctly
"""
import os
import sys

print("=" * 60)
print("TEST STEP 10: Package Initialization (__init__.py) Check")
print("=" * 60)

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print(f"\nProject directory: {script_dir}")

all_passed = True

# Check if __init__.py files exist and have content
print("\n" + "-" * 60)
print("Checking __init__.py files:")
print("-" * 60)

init_files = [
    os.path.join(script_dir, 'GNN', '__init__.py'),
    os.path.join(script_dir, 'Utils', '__init__.py')
]

for init_file in init_files:
    if os.path.exists(init_file):
        size = os.path.getsize(init_file)
        if size > 500:  # Should have substantial content now
            print(f"  ✓ {init_file.replace(script_dir, '.')} ({size} bytes)")
        else:
            print(f"  ⚠ {init_file.replace(script_dir, '.')} ({size} bytes) - seems too small!")
            all_passed = False
    else:
        print(f"  ✗ {init_file.replace(script_dir, '.')} MISSING!")
        all_passed = False

# Test 1: Import GNN package
print("\n" + "-" * 60)
print("Test 1: Import GNN package")
print("-" * 60)

try:
    import GNN
    print(f"  ✓ import GNN")
    print(f"    Version: {getattr(GNN, '__version__', 'N/A')}")
except ImportError as e:
    print(f"  ✗ import GNN failed: {e}")
    all_passed = False

# Test direct imports from GNN
print("\n  Testing direct imports from GNN:")

gnn_imports = [
    ('MLP', 'layers'),
    ('GraphNetwork_layer', 'layers'),
    ('L1_Loss', 'losses'),
    ('L2_Loss', 'losses'),
    ('Structure_GraphNetwork', 'models'),
    ('Structure_GCN', 'models'),
    ('Structure_GAT', 'models'),
    ('Structure_GIN', 'models'),
]

for name, source in gnn_imports:
    try:
        obj = getattr(GNN, name)
        print(f"    ✓ from GNN import {name}  (from {source})")
    except AttributeError:
        print(f"    ✗ from GNN import {name}  - NOT FOUND!")
        all_passed = False

# Test 2: Import Utils package
print("\n" + "-" * 60)
print("Test 2: Import Utils package")
print("-" * 60)

try:
    import Utils
    print(f"  ✓ import Utils")
    print(f"    Version: {getattr(Utils, '__version__', 'N/A')}")
except ImportError as e:
    print(f"  ✗ import Utils failed: {e}")
    all_passed = False

# Test direct imports from Utils
print("\n  Testing direct imports from Utils:")

utils_imports = [
    ('node_accuracy', 'accuracy'),
    ('get_dataset', 'datasets'),
    ('split_dataset', 'datasets'),
    ('get_target_index', 'datasets'),
    ('normalize_dataset', 'normalization'),
    ('denormalize_y_linear', 'normalization'),
    ('print_space', 'plot'),
    ('plot_learningCurve', 'plot'),
    ('plot_lossCurve', 'plot'),
]

for name, source in utils_imports:
    try:
        obj = getattr(Utils, name)
        print(f"    ✓ from Utils import {name}  (from {source})")
    except AttributeError:
        print(f"    ✗ from Utils import {name}  - NOT FOUND!")
        all_passed = False

# Test 3: Test various import styles
print("\n" + "-" * 60)
print("Test 3: Different import styles")
print("-" * 60)

# Style 1: Direct package import
print("\n  Style 1: from PACKAGE import ITEM")
try:
    from GNN import Structure_GraphNetwork
    from GNN import L1_Loss
    from Utils import get_dataset
    from Utils import normalize_dataset
    print("    ✓ All direct imports work")
except ImportError as e:
    print(f"    ✗ Direct import failed: {e}")
    all_passed = False

# Style 2: Module-level import
print("\n  Style 2: from PACKAGE.MODULE import ITEM")
try:
    from GNN.models import Structure_GraphNetwork
    from GNN.losses import L1_Loss
    from GNN.layers import MLP
    from Utils.datasets import get_dataset
    from Utils.normalization import normalize_dataset
    from Utils.plot import plot_learningCurve
    print("    ✓ All module-level imports work")
except ImportError as e:
    print(f"    ✗ Module-level import failed: {e}")
    all_passed = False

# Style 3: Import entire module
print("\n  Style 3: from PACKAGE import MODULE")
try:
    from GNN import models
    from GNN import losses
    from GNN import layers
    from Utils import datasets
    from Utils import normalization
    from Utils import plot
    print("    ✓ All module imports work")
except ImportError as e:
    print(f"    ✗ Module import failed: {e}")
    all_passed = False

# Test 4: Check __all__ exports
print("\n" + "-" * 60)
print("Test 4: Check __all__ exports")
print("-" * 60)

try:
    gnn_all = getattr(GNN, '__all__', [])
    print(f"  GNN.__all__ has {len(gnn_all)} items:")
    for item in gnn_all[:5]:
        print(f"    - {item}")
    if len(gnn_all) > 5:
        print(f"    ... and {len(gnn_all) - 5} more")
    
    utils_all = getattr(Utils, '__all__', [])
    print(f"\n  Utils.__all__ has {len(utils_all)} items:")
    for item in utils_all[:5]:
        print(f"    - {item}")
    if len(utils_all) > 5:
        print(f"    ... and {len(utils_all) - 5} more")
        
    if len(gnn_all) >= 8 and len(utils_all) >= 15:
        print(f"\n  ✓ Both __all__ lists have expected items")
    else:
        print(f"\n  ⚠ __all__ lists may be incomplete")
        
except Exception as e:
    print(f"  ✗ __all__ check failed: {e}")

# Test 5: Functional test - create a model
print("\n" + "-" * 60)
print("Test 5: Functional test - create and use components")
print("-" * 60)

try:
    import torch
    
    # Import using the new clean syntax
    from GNN import Structure_GraphNetwork, L1_Loss, MLP
    from Utils import get_target_index, print_space
    
    # Create a model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Structure_GraphNetwork(
        layer_num=3,
        input_dim=15,
        hidden_dim=64,
        edge_attr_dim=3,
        aggr='mean',
        device=device
    ).to(device)
    
    print(f"  ✓ Created Structure_GraphNetwork model")
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss
    criterion = L1_Loss()
    print(f"  ✓ Created L1_Loss criterion")
    
    # Get target index
    y_start, y_end = get_target_index('all')
    print(f"  ✓ get_target_index('all') → [{y_start}:{y_end}]")
    
    # Test forward pass
    x = torch.randn(10, 15).to(device)
    edge_index = torch.randint(0, 10, (2, 20)).to(device)
    edge_attr = torch.randn(20, 3).to(device)
    
    output = model(x, edge_index, edge_attr)
    print(f"  ✓ Forward pass successful, output shape: {output.shape}")
    
except Exception as e:
    print(f"  ✗ Functional test failed: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# Test 6: Show the clean import structure
print("\n" + "-" * 60)
print("Test 6: Clean import examples")
print("-" * 60)

print("""
  Your code can now use these clean imports:
  
  ┌────────────────────────────────────────────────────────────┐
  │  # Models                                                  │
  │  from GNN import Structure_GraphNetwork                    │
  │  from GNN import Structure_GCN, Structure_GAT              │
  │                                                            │
  │  # Losses                                                  │
  │  from GNN import L1_Loss, L2_Loss                          │
  │                                                            │
  │  # Layers (for custom models)                              │
  │  from GNN import MLP, GraphNetwork_layer                   │
  │                                                            │
  │  # Data handling                                           │
  │  from Utils import get_dataset, split_dataset              │
  │  from Utils import normalize_dataset, denormalize_y_linear │
  │                                                            │
  │  # Training utilities                                      │
  │  from Utils import node_accuracy, get_target_index         │
  │                                                            │
  │  # Visualization                                           │
  │  from Utils import plot_learningCurve, plot_lossCurve      │
  │  from Utils import print_space                             │
  └────────────────────────────────────────────────────────────┘
""")

# Display final project structure
print("\n" + "-" * 60)
print("Current Project Structure:")
print("-" * 60)

def print_tree(path, prefix=""):
    """Print directory tree."""
    items = sorted(os.listdir(path))
    dirs = [i for i in items if os.path.isdir(os.path.join(path, i)) and not i.startswith('.') and not i.startswith('__')]
    files = [i for i in items if os.path.isfile(os.path.join(path, i)) and i.endswith('.py')]
    
    for i, f in enumerate(files):
        is_last = (i == len(files) - 1) and len(dirs) == 0
        print(f"  {prefix}{'└── ' if is_last else '├── '}{f}")
    
    for i, d in enumerate(dirs):
        is_last = i == len(dirs) - 1
        print(f"  {prefix}{'└── ' if is_last else '├── '}{d}/")
        print_tree(os.path.join(path, d), prefix + ('    ' if is_last else '│   '))

print(f"\n  {os.path.basename(script_dir)}/")
print_tree(script_dir)

# Summary
print("\n" + "=" * 60)
if all_passed:
    print("STEP 10 PASSED! ✓")
    print("=" * 60)
    print("\nPackage initialization is complete!")
    print("""
    What you updated:
    ├── GNN/
    │   └── __init__.py  ✓  (exports models, losses, layers)
    └── Utils/
        └── __init__.py  ✓  (exports all utility functions)
    
    Benefits:
    • Cleaner imports: from GNN import Structure_GraphNetwork
    • Better organization: all exports defined in one place
    • Documentation: package docstrings explain usage
    • Version tracking: __version__ = '1.0.0'
    
    Complete project structure:
    ┌──────────────────────────────────────────────────────────┐
    │  GNN/                                                    │
    │  ├── __init__.py      (package exports)                  │
    │  ├── layers.py        (MLP, GraphNetwork_layer)          │
    │  ├── losses.py        (L1_Loss, L2_Loss)                 │
    │  └── models.py        (Structure_GraphNetwork, etc.)     │
    │                                                          │
    │  Utils/                                                  │
    │  ├── __init__.py      (package exports)                  │
    │  ├── accuracy.py      (node_accuracy)                    │
    │  ├── datasets.py      (get_dataset, split_dataset)       │
    │  ├── normalization.py (normalize_dataset, denormalize)   │
    │  └── plot.py          (plot_learningCurve, etc.)         │
    └──────────────────────────────────────────────────────────┘
    """)
    print("You can proceed to Step 11 (Final Step: Main Training Script).")
else:
    print("STEP 10 FAILED! ✗")
    print("=" * 60)
    print("\nPlease fix the errors above and run this test again.")
print("=" * 60)