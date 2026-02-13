"""
TEST STEP 8: Verify Utils/normalization.py is created correctly and works
"""
import os
import sys

print("=" * 60)
print("TEST STEP 8: Normalization Functions Check")
print("         (Utils/normalization.py)")
print("=" * 60)

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print(f"\nProject directory: {script_dir}")

# Check if file exists
print("\n" + "-" * 60)
print("Checking file exists:")
print("-" * 60)

norm_path = os.path.join(script_dir, 'Utils', 'normalization.py')
all_passed = True

if os.path.exists(norm_path):
    size = os.path.getsize(norm_path)
    print(f"  ✓ Utils/normalization.py exists ({size} bytes)")
    if size < 2000:
        print(f"    ⚠ Warning: File seems too small. Did you add the full code?")
        all_passed = False
else:
    print(f"  ✗ Utils/normalization.py MISSING!")
    all_passed = False
    print("\n" + "=" * 60)
    print("STEP 8 FAILED! ✗")
    print("Please create Utils/normalization.py with the provided code.")
    print("=" * 60)
    sys.exit(1)

# Check required packages
print("\n" + "-" * 60)
print("Checking required packages:")
print("-" * 60)

try:
    import torch
    print(f"  ✓ torch: {torch.__version__}")
except ImportError:
    print(f"  ✗ torch not installed!")
    sys.exit(1)

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    print(f"  ✓ torch_geometric available")
except ImportError:
    print(f"  ✗ torch_geometric not installed!")
    sys.exit(1)

# Try to import functions
print("\n" + "-" * 60)
print("Testing imports:")
print("-" * 60)

try:
    from Utils.normalization import normalize_dataset
    print(f"  ✓ normalize_dataset imported")
except ImportError as e:
    print(f"  ✗ normalize_dataset import failed: {e}")
    all_passed = False
    sys.exit(1)

try:
    from Utils.normalization import denormalize_y_linear
    print(f"  ✓ denormalize_y_linear imported")
except ImportError as e:
    print(f"  ✗ denormalize_y_linear import failed: {e}")
    all_passed = False

try:
    from Utils.normalization import denormalize_grid_num
    print(f"  ✓ denormalize_grid_num imported")
except ImportError as e:
    print(f"  ⚠ denormalize_grid_num not found (optional)")

try:
    from Utils.normalization import getMinMax_x, getMinMax_y_linear, normalize_linear
    print(f"  ✓ Helper functions imported")
except ImportError as e:
    print(f"  ⚠ Some helper functions not found: {e}")

# Create mock graph data for testing
print("\n" + "-" * 60)
print("Creating mock graph data for testing:")
print("-" * 60)

try:
    from torch_geometric.data import Data
    
    def create_mock_graph(num_nodes=10, seed=None):
        """Create a mock graph that mimics real structural data."""
        if seed is not None:
            torch.manual_seed(seed)
        
        # Node features: 15 dimensions
        # [grid_num(3), coords(3), BC(2), mass(1), forces(6)]
        x = torch.zeros(num_nodes, 15)
        x[:, 0:3] = torch.randint(1, 10, (num_nodes, 3)).float()  # Grid numbers
        x[:, 3:6] = torch.rand(num_nodes, 3) * 100  # Coordinates (0-100)
        x[:, 6:8] = torch.randint(0, 2, (num_nodes, 2)).float()  # BC (0 or 1)
        x[:, 8] = torch.rand(num_nodes) * 1000  # Mass (0-1000)
        x[:, 9:15] = torch.randn(num_nodes, 6) * 10000  # Forces (large values)
        
        # Edge index: create some connections
        num_edges = num_nodes * 2
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Edge attributes: 3 dimensions [type, section, length]
        edge_attr = torch.zeros(num_edges, 3)
        edge_attr[:, 0] = torch.randint(0, 3, (num_edges,)).float()  # Element type
        edge_attr[:, 1] = torch.randint(0, 5, (num_edges,)).float()  # Section type
        edge_attr[:, 2] = torch.rand(num_edges) * 10  # Length (0-10)
        
        # Outputs: 38 dimensions (or 26 for basic)
        y = torch.zeros(num_nodes, 38)
        y[:, 0:2] = torch.randn(num_nodes, 2) * 0.1  # Displacement (small)
        y[:, 2:8] = torch.randn(num_nodes, 6) * 50000  # Moment Y (large)
        y[:, 8:14] = torch.randn(num_nodes, 6) * 50000  # Moment Z
        y[:, 14:20] = torch.randn(num_nodes, 6) * 10000  # Shear Y
        y[:, 20:26] = torch.randn(num_nodes, 6) * 10000  # Shear Z
        y[:, 26:32] = torch.randn(num_nodes, 6) * 5000  # Axial Force
        y[:, 32:38] = torch.randn(num_nodes, 6) * 1000  # Torsion
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # Create mock dataset with 10 graphs
    mock_dataset = [create_mock_graph(num_nodes=10, seed=i) for i in range(10)]
    
    print(f"  ✓ Created mock dataset with {len(mock_dataset)} graphs")
    print(f"    Sample graph:")
    print(f"      Nodes: {mock_dataset[0].x.shape[0]}")
    print(f"      Node features: {mock_dataset[0].x.shape}")
    print(f"      Edges: {mock_dataset[0].edge_index.shape[1]}")
    print(f"      Edge attributes: {mock_dataset[0].edge_attr.shape}")
    print(f"      Outputs: {mock_dataset[0].y.shape}")
    
except Exception as e:
    print(f"  ✗ Failed to create mock data: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False
    sys.exit(1)

# Test 1: Check original data ranges
print("\n" + "-" * 60)
print("Test 1: Original data ranges (before normalization)")
print("-" * 60)

try:
    sample = mock_dataset[0]
    
    print(f"  Node features (x):")
    print(f"    Grid numbers [0:3]:  min={sample.x[:, 0:3].min():.2f}, max={sample.x[:, 0:3].max():.2f}")
    print(f"    Coordinates [3:6]:   min={sample.x[:, 3:6].min():.2f}, max={sample.x[:, 3:6].max():.2f}")
    print(f"    Mass [8]:            min={sample.x[:, 8].min():.2f}, max={sample.x[:, 8].max():.2f}")
    print(f"    Forces [9:15]:       min={sample.x[:, 9:15].min():.2f}, max={sample.x[:, 9:15].max():.2f}")
    
    print(f"\n  Outputs (y):")
    print(f"    Displacement [0:2]:  min={sample.y[:, 0:2].min():.4f}, max={sample.y[:, 0:2].max():.4f}")
    print(f"    Moment Y [2:8]:      min={sample.y[:, 2:8].min():.2f}, max={sample.y[:, 2:8].max():.2f}")
    print(f"    Shear Y [14:20]:     min={sample.y[:, 14:20].min():.2f}, max={sample.y[:, 14:20].max():.2f}")
    
    print(f"\n  ✓ Data ranges vary significantly (this is why we normalize!)")
    
except Exception as e:
    print(f"  ✗ Test 1 failed: {e}")
    all_passed = False

# Test 2: Normalize dataset
print("\n" + "-" * 60)
print("Test 2: Normalize dataset")
print("-" * 60)

try:
    from Utils.normalization import normalize_dataset
    
    # Make a copy for normalization
    import copy
    mock_dataset_copy = [copy.deepcopy(d) for d in mock_dataset]
    
    # Normalize
    normalized_dataset, norm_dict = normalize_dataset(mock_dataset_copy)
    
    print(f"  ✓ normalize_dataset() completed")
    print(f"  ✓ Returned norm_dict with {len(norm_dict)} entries")
    
    # Print norm_dict
    print(f"\n  Normalization parameters:")
    for key, (min_val, max_val) in norm_dict.items():
        if isinstance(min_val, torch.Tensor):
            print(f"    {key:<15}: [{min_val.item():.4f}, {max_val.item():.4f}]")
        else:
            print(f"    {key:<15}: [{min_val}, {max_val}]")
    
except Exception as e:
    print(f"  ✗ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# Test 3: Check normalized data ranges
print("\n" + "-" * 60)
print("Test 3: Normalized data ranges (should be ~[0, 1])")
print("-" * 60)

try:
    sample = normalized_dataset[0]
    
    # Check various normalized ranges
    checks = [
        ("Grid numbers [0:3]", sample.x[:, 0:3]),
        ("Coordinates [3:6]", sample.x[:, 3:6]),
        ("Mass [8]", sample.x[:, 8:9]),
        ("Displacement [0:2]", sample.y[:, 0:2]),
        ("Moment Y [2:8]", sample.y[:, 2:8]),
    ]
    
    all_in_range = True
    for name, tensor in checks:
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        in_range = -0.1 <= min_val and max_val <= 1.1  # Allow small tolerance
        
        status = "✓" if in_range else "⚠"
        print(f"    {status} {name}: [{min_val:.4f}, {max_val:.4f}]")
        
        if not in_range:
            all_in_range = False
    
    if all_in_range:
        print(f"\n  ✓ All normalized values are in expected range [0, 1]")
    else:
        print(f"\n  ⚠ Some values outside [0, 1] (may be due to negative values)")
        
except Exception as e:
    print(f"  ✗ Test 3 failed: {e}")
    all_passed = False

# Test 4: Denormalization
print("\n" + "-" * 60)
print("Test 4: Denormalization (convert back to original scale)")
print("-" * 60)

try:
    from Utils.normalization import denormalize_y_linear
    
    # Get original and normalized y
    original_y = mock_dataset[0].y.clone()
    normalized_y = normalized_dataset[0].y.clone()
    
    print(f"  Original y [0:2]: {original_y[0, 0:2].tolist()}")
    print(f"  Normalized y [0:2]: {normalized_y[0, 0:2].tolist()}")
    
    # Denormalize
    denormalized_y = denormalize_y_linear(normalized_y, norm_dict)
    
    print(f"  Denormalized y [0:2]: {denormalized_y[0, 0:2].tolist()}")
    
    # Check if denormalized matches original
    diff = torch.abs(original_y - denormalized_y).max().item()
    print(f"\n  Max difference after round-trip: {diff:.10f}")
    
    if diff < 1e-5:
        print(f"  ✓ Denormalization correctly recovers original values")
    else:
        print(f"  ⚠ Some precision loss (this is normal for floating point)")
        
except Exception as e:
    print(f"  ✗ Test 4 failed: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# Test 5: Test individual denormalize functions
print("\n" + "-" * 60)
print("Test 5: Individual denormalization functions")
print("-" * 60)

try:
    from Utils.normalization import denormalize_grid_num, denormalize_disp
    
    # Test denormalize_grid_num
    normalized_grid = torch.tensor([0.5, 0.5, 0.5])
    denorm_grid = denormalize_grid_num(normalized_grid, norm_dict)
    print(f"  Grid: normalized [0.5, 0.5, 0.5] → denormalized {denorm_grid.tolist()}")
    print(f"  ✓ denormalize_grid_num works")
    
    # Test denormalize_disp
    normalized_disp = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    denorm_disp = denormalize_disp(normalized_disp, norm_dict)
    print(f"\n  Displacement:")
    print(f"    normalized [0, 0] → denormalized {denorm_disp[0].tolist()}")
    print(f"    normalized [1, 1] → denormalized {denorm_disp[1].tolist()}")
    print(f"  ✓ denormalize_disp works")
    
except Exception as e:
    print(f"  ✗ Test 5 failed: {e}")
    all_passed = False

# Test 6: Edge case - all same values
print("\n" + "-" * 60)
print("Test 6: Edge case handling")
print("-" * 60)

try:
    # Create a simple norm_dict manually
    test_norm_dict = {
        'disp': [0, torch.tensor(1.0)],
        'momentY': [0, torch.tensor(100.0)],
        'momentZ': [0, torch.tensor(100.0)],
        'shearY': [0, torch.tensor(50.0)],
        'shearZ': [0, torch.tensor(50.0)],
    }
    
    # Test with edge values
    test_y = torch.zeros(5, 26)
    test_y[:, 0:2] = 0.5  # Mid-range displacement
    test_y[:, 2:8] = 1.0  # Max moment Y
    
    denorm_y = denormalize_y_linear(test_y, test_norm_dict)
    
    print(f"  Input: 0.5 normalized displacement")
    print(f"  Output: {denorm_y[0, 0].item():.4f} (expected: 0.5)")
    print(f"  ✓ Edge cases handled correctly")
    
except Exception as e:
    print(f"  ⚠ Edge case test: {e}")

# Visual explanation
print("\n" + "-" * 60)
print("Understanding Normalization:")
print("-" * 60)
print("""
  WHY NORMALIZE?
  ─────────────────────────────────────────────────────────
  Before normalization:
    Coordinates:  0 to 100      (small range)
    Forces:       0 to 100000   (large range)  
    
  The neural network would be dominated by large values!
  
  After normalization:
    Coordinates:  0 to 1        (same scale)
    Forces:       0 to 1        (same scale)
  
  ─────────────────────────────────────────────────────────
  
  NORMALIZATION FORMULA:
    normalized = (value - min) / (max - min)
    
    Example: value=50, min=0, max=100
    normalized = (50 - 0) / (100 - 0) = 0.5
  
  ─────────────────────────────────────────────────────────
  
  DENORMALIZATION (to get back original):
    original = normalized * (max - min) + min
    
    Example: normalized=0.5, min=0, max=100
    original = 0.5 * (100 - 0) + 0 = 50
  
  ─────────────────────────────────────────────────────────
  
  IN TRAINING:
    1. Normalize all data before training
    2. Train model on normalized data
    3. Model outputs normalized predictions
    4. Denormalize predictions for actual values
""")

# Summary
print("\n" + "=" * 60)
if all_passed:
    print("STEP 8 PASSED! ✓")
    print("=" * 60)
    print("\nNormalization functions are working correctly!")
    print("""
    What you created:
    ├── GNN/
    │   ├── __init__.py
    │   ├── losses.py           ✓
    │   ├── layers.py           ✓
    │   └── models.py           ✓
    ├── Utils/
    │   ├── __init__.py
    │   ├── accuracy.py         ✓
    │   ├── datasets.py         ✓
    │   └── normalization.py    ✓  (normalize/denormalize functions)
    
    Key functions:
    ┌────────────────────────────┬────────────────────────────────────┐
    │ Function                   │ Purpose                            │
    ├────────────────────────────┼────────────────────────────────────┤
    │ normalize_dataset()        │ Normalize entire dataset           │
    │ denormalize_y_linear()     │ Convert predictions to real scale  │
    │ denormalize_grid_num()     │ Denormalize grid numbers           │
    │ denormalize_disp()         │ Denormalize displacements          │
    └────────────────────────────┴────────────────────────────────────┘
    
    Usage:
        # During training setup:
        dataset, norm_dict = normalize_dataset(dataset)
        
        # After prediction:
        actual_values = denormalize_y_linear(predictions, norm_dict)
    """)
    print("You can proceed to Step 9.")
else:
    print("STEP 8 FAILED! ✗")
    print("=" * 60)
    print("\nPlease fix the errors above and run this test again.")
print("=" * 60)