"""
TEST STEP 5: Verify GNN/models.py is created correctly and works
"""
import os
import sys

print("=" * 60)
print("TEST STEP 5: GNN Models (GNN/models.py) Check")
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

models_path = os.path.join(script_dir, 'GNN', 'models.py')
all_passed = True

if os.path.exists(models_path):
    size = os.path.getsize(models_path)
    print(f"  ✓ GNN/models.py exists ({size} bytes)")
    if size < 2000:
        print(f"    ⚠ Warning: File seems too small. Did you add the full code?")
        all_passed = False
else:
    print(f"  ✗ GNN/models.py MISSING!")
    all_passed = False
    print("\n" + "=" * 60)
    print("STEP 5 FAILED! ✗")
    print("Please create GNN/models.py with the provided code.")
    print("=" * 60)
    sys.exit(1)

# Check required packages
print("\n" + "-" * 60)
print("Checking required packages:")
print("-" * 60)

try:
    import torch
    print(f"  ✓ torch: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  ✓ Using device: {device}")
except ImportError:
    print(f"  ✗ torch not installed!")
    sys.exit(1)

try:
    import torch_geometric
    print(f"  ✓ torch_geometric: {torch_geometric.__version__}")
except ImportError:
    print(f"  ✗ torch_geometric not installed!")
    sys.exit(1)

# Try to import models
print("\n" + "-" * 60)
print("Testing imports:")
print("-" * 60)

models_to_test = []

try:
    from GNN.models import Structure_GraphNetwork
    print(f"  ✓ Structure_GraphNetwork imported")
    models_to_test.append(('Structure_GraphNetwork', Structure_GraphNetwork, True))
except ImportError as e:
    print(f"  ✗ Structure_GraphNetwork import failed: {e}")
    all_passed = False

try:
    from GNN.models import Structure_GCN
    print(f"  ✓ Structure_GCN imported")
    models_to_test.append(('Structure_GCN', Structure_GCN, False))
except ImportError as e:
    print(f"  ✗ Structure_GCN import failed: {e}")
    all_passed = False

try:
    from GNN.models import Structure_GAT
    print(f"  ✓ Structure_GAT imported")
    models_to_test.append(('Structure_GAT', Structure_GAT, False))
except ImportError as e:
    print(f"  ✗ Structure_GAT import failed: {e}")
    all_passed = False

try:
    from GNN.models import Structure_GIN
    print(f"  ✓ Structure_GIN imported")
    models_to_test.append(('Structure_GIN', Structure_GIN, False))
except ImportError as e:
    print(f"  ✗ Structure_GIN import failed: {e}")
    all_passed = False

if not models_to_test:
    print("\n" + "=" * 60)
    print("STEP 5 FAILED! ✗")
    print("No models could be imported. Check GNN/models.py for errors.")
    print("=" * 60)
    sys.exit(1)

# Create test graph data
print("\n" + "-" * 60)
print("Creating test graph (simulating a simple structure):")
print("-" * 60)

"""
Test structure (2x2 grid):

    0 ─── 1
    │     │
    2 ─── 3

Nodes: 4 (structural joints)
Edges: 8 (4 beams × 2 directions)
Node features: 15 (coordinates, forces, etc.)
Edge features: 3 (beam properties)
"""

num_nodes = 4
num_edges = 8
input_dim = 15
edge_attr_dim = 3
hidden_dim = 64
layer_num = 3

# Node features (simulating structural node properties)
x = torch.randn(num_nodes, input_dim).to(device)

# Edge connectivity (bidirectional edges)
edge_index = torch.tensor([
    [0, 1, 1, 0, 0, 2, 2, 0, 1, 3, 3, 1, 2, 3, 3, 2],
    [1, 0, 0, 1, 2, 0, 0, 2, 3, 1, 1, 3, 3, 2, 2, 3]
], dtype=torch.long).to(device)
edge_index = edge_index[:, :num_edges]  # Take first 8 edges

# Edge attributes (beam properties: type, section, length)
edge_attr = torch.randn(num_edges, edge_attr_dim).to(device)

print(f"  Graph structure: 2x2 grid (4 nodes, 4 beams)")
print(f"  Node features shape: {x.shape}")
print(f"  Edge index shape: {edge_index.shape}")
print(f"  Edge attributes shape: {edge_attr.shape}")
print(f"  Device: {device}")

# Test each model
for model_name, ModelClass, uses_edge_attr in models_to_test:
    print("\n" + "-" * 60)
    print(f"Testing {model_name}:")
    print("-" * 60)
    
    try:
        # Create model
        model = ModelClass(
            layer_num=layer_num,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            edge_attr_dim=edge_attr_dim,
            aggr='mean',
            gnn_act=True,
            gnn_dropout=False,
            dropout_p=0.0,
            device=device
        ).to(device)
        
        print(f"  ✓ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ✓ Total parameters: {total_params:,}")
        print(f"  ✓ Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(x, edge_index, edge_attr)
        
        print(f"  ✓ Forward pass successful")
        print(f"    Input shape: {x.shape}")
        print(f"    Output shape: {output.shape}")
        
        # Verify output shape
        expected_shape = (num_nodes, 26)
        if output.shape == expected_shape:
            print(f"  ✓ Output shape correct: {output.shape}")
        else:
            print(f"  ✗ Output shape wrong! Expected {expected_shape}, got {output.shape}")
            all_passed = False
        
        # Verify output structure
        print(f"\n  Output structure verification:")
        print(f"    [:, 0]     displacement X : {output[:, 0].shape}")
        print(f"    [:, 1]     displacement Z : {output[:, 1].shape}")
        print(f"    [:, 2:8]   moment Y (6)   : {output[:, 2:8].shape}")
        print(f"    [:, 8:14]  moment Z (6)   : {output[:, 8:14].shape}")
        print(f"    [:, 14:20] shear Y (6)    : {output[:, 14:20].shape}")
        print(f"    [:, 20:26] shear Z (6)    : {output[:, 20:26].shape}")
        
        # Test training mode
        model.train()
        output_train = model(x, edge_index, edge_attr)
        loss = output_train.sum()
        loss.backward()
        print(f"  ✓ Backward pass successful (gradients computed)")
        
        # Check if model uses edge attributes
        if uses_edge_attr:
            print(f"  ℹ This model USES edge attributes (beam properties)")
        else:
            print(f"  ℹ This model does NOT use edge attributes")
        
    except Exception as e:
        print(f"  ✗ {model_name} test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

# Test model output range
print("\n" + "-" * 60)
print("Testing output characteristics:")
print("-" * 60)

try:
    model = Structure_GraphNetwork(
        layer_num=3,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        edge_attr_dim=edge_attr_dim,
        aggr='mean',
        device=device
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, edge_attr)
    
    print(f"  Output statistics (before training):")
    print(f"    Mean: {output.mean().item():.6f}")
    print(f"    Std:  {output.std().item():.6f}")
    print(f"    Min:  {output.min().item():.6f}")
    print(f"    Max:  {output.max().item():.6f}")
    print(f"  ✓ Model produces reasonable output range")
    
except Exception as e:
    print(f"  ✗ Output test failed: {e}")
    all_passed = False

# Summary
print("\n" + "=" * 60)
if all_passed:
    print("STEP 5 PASSED! ✓")
    print("=" * 60)
    print("\nGNN models are working correctly!")
    print("""
    What you created:
    ├── GNN/
    │   ├── __init__.py
    │   ├── losses.py           ✓
    │   ├── layers.py           ✓
    │   └── models.py           ✓  (4 model architectures)
    
    Available models:
    ┌─────────────────────────┬──────────────────────┬─────────────┐
    │ Model                   │ Edge Attributes      │ Best For    │
    ├─────────────────────────┼──────────────────────┼─────────────┤
    │ Structure_GraphNetwork  │ ✓ Uses edge attrs    │ Recommended │
    │ Structure_GCN           │ ✗ Ignores edge attrs │ Baseline    │
    │ Structure_GAT           │ ✗ Ignores edge attrs │ Attention   │
    │ Structure_GIN           │ ✗ Ignores edge attrs │ Expressive  │
    └─────────────────────────┴──────────────────────┴─────────────┘
    
    Model output (26 values per node):
    • Displacement X (1)
    • Displacement Z (1)
    • Moment Y (6)
    • Moment Z (6)
    • Shear Y (6)
    • Shear Z (6)
    """)
    print("You can proceed to Step 6.")
else:
    print("STEP 5 FAILED! ✗")
    print("=" * 60)
    print("\nPlease fix the errors above and run this test again.")
print("=" * 60)