"""
TEST STEP 4: Verify GNN/layers.py is created correctly and works
"""
import os
import sys

print("=" * 60)
print("TEST STEP 4: GNN Layers (GNN/layers.py) Check")
print("=" * 60)

# Get the directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print(f"\nProject directory: {script_dir}")

# Check if file exists
print("\n" + "-" * 60)
print("Checking file exists:")
print("-" * 60)

layers_path = os.path.join(script_dir, 'GNN', 'layers.py')
all_passed = True

if os.path.exists(layers_path):
    size = os.path.getsize(layers_path)
    print(f"  ✓ GNN/layers.py exists ({size} bytes)")
    if size < 500:
        print(f"    ⚠ Warning: File seems too small. Did you add the full code?")
        all_passed = False
else:
    print(f"  ✗ GNN/layers.py MISSING!")
    print(f"    Expected at: {layers_path}")
    all_passed = False
    print("\n" + "=" * 60)
    print("STEP 4 FAILED! ✗")
    print("=" * 60)
    print("\nPlease create GNN/layers.py with the provided code.")
    sys.exit(1)

# Check required packages
print("\n" + "-" * 60)
print("Checking required packages:")
print("-" * 60)

try:
    import torch
    print(f"  ✓ torch: {torch.__version__}")
except ImportError:
    print(f"  ✗ torch not installed! Run: pip install torch")
    sys.exit(1)

try:
    import torch_geometric
    print(f"  ✓ torch_geometric: {torch_geometric.__version__}")
except ImportError:
    print(f"  ✗ torch_geometric not installed!")
    print(f"    Run: pip install torch-geometric")
    sys.exit(1)

try:
    from torch_geometric.nn import MessagePassing
    print(f"  ✓ MessagePassing class available")
except ImportError as e:
    print(f"  ✗ MessagePassing import failed: {e}")
    sys.exit(1)

# Try to import layers
print("\n" + "-" * 60)
print("Testing import:")
print("-" * 60)

try:
    from GNN.layers import MLP, GraphNetwork_layer
    print(f"  ✓ 'from GNN.layers import MLP, GraphNetwork_layer' works")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    print("\nPossible causes:")
    print("  1. Syntax error in layers.py")
    print("  2. Missing import statements")
    all_passed = False
    sys.exit(1)

# Test MLP
print("\n" + "-" * 60)
print("Testing MLP:")
print("-" * 60)

try:
    # Test 1: Simple MLP without hidden layers
    mlp_simple = MLP(input_dim=10, hidden_dim=[], output_dim=5, act=False, dropout=False)
    print(f"  ✓ MLP(10 → 5) created")
    
    # Count parameters
    params = sum(p.numel() for p in mlp_simple.parameters())
    print(f"    Parameters: {params} (expected: 10*5 + 5 = 55)")
    
    # Test forward pass
    x = torch.randn(4, 10)  # batch of 4, 10 features each
    output = mlp_simple(x)
    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {output.shape}")
    assert output.shape == (4, 5), "Output shape mismatch!"
    print(f"  ✓ Forward pass successful")
    
except Exception as e:
    print(f"  ✗ MLP test failed: {e}")
    all_passed = False

try:
    # Test 2: MLP with hidden layers and activation
    mlp_deep = MLP(input_dim=10, hidden_dim=[64, 32], output_dim=5, act=True, dropout=False)
    print(f"\n  ✓ MLP(10 → 64 → 32 → 5, with ReLU) created")
    
    # Count parameters
    params = sum(p.numel() for p in mlp_deep.parameters())
    # 10*64 + 64 + 64*32 + 32 + 32*5 + 5 = 640 + 64 + 2048 + 32 + 160 + 5 = 2949
    print(f"    Parameters: {params}")
    
    # Test forward pass
    x = torch.randn(4, 10)
    output = mlp_deep(x)
    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {output.shape}")
    assert output.shape == (4, 5), "Output shape mismatch!"
    print(f"  ✓ Forward pass successful")
    
except Exception as e:
    print(f"  ✗ Deep MLP test failed: {e}")
    all_passed = False

# Test GraphNetwork_layer
print("\n" + "-" * 60)
print("Testing GraphNetwork_layer:")
print("-" * 60)

try:
    # Create a simple graph:
    #   0 ←→ 1 ←→ 2
    #         ↓
    #         3
    #
    # Edges: 0-1, 1-0, 1-2, 2-1, 1-3, 3-1
    
    print("  Creating test graph:")
    print("      0 ←→ 1 ←→ 2")
    print("           ↓")
    print("           3")
    
    edge_index = torch.tensor([
        [0, 1, 1, 2, 1, 3],  # source nodes
        [1, 0, 2, 1, 3, 1]   # target nodes
    ], dtype=torch.long)
    
    num_nodes = 4
    num_edges = 6
    node_features = 8
    edge_features = 3
    output_features = 16
    
    # Node features: 4 nodes, 8 features each
    x = torch.randn(num_nodes, node_features)
    
    # Edge attributes: 6 edges, 3 features each
    edge_attr = torch.randn(num_edges, edge_features)
    
    print(f"\n  Graph statistics:")
    print(f"    Nodes: {num_nodes}")
    print(f"    Edges: {num_edges}")
    print(f"    Node features: {node_features}")
    print(f"    Edge features: {edge_features}")
    
    # Create GraphNetwork layer
    gn_layer = GraphNetwork_layer(
        input_dim=node_features,
        output_dim=output_features,
        edge_attr_dim=edge_features,
        aggr='mean'
    )
    print(f"\n  ✓ GraphNetwork_layer created")
    print(f"    Input dim: {node_features}")
    print(f"    Output dim: {output_features}")
    print(f"    Edge attr dim: {edge_features}")
    print(f"    Aggregation: mean")
    
    # Test forward pass
    output = gn_layer(x, edge_index, edge_attr)
    
    print(f"\n  Forward pass:")
    print(f"    Input x shape: {x.shape}")
    print(f"    Output shape: {output.shape}")
    
    assert output.shape == (num_nodes, output_features), "Output shape mismatch!"
    print(f"  ✓ Forward pass successful!")
    
except Exception as e:
    print(f"  ✗ GraphNetwork_layer test failed: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# Test with different aggregation methods
print("\n" + "-" * 60)
print("Testing different aggregation methods:")
print("-" * 60)

for aggr_method in ['mean', 'sum', 'max']:
    try:
        gn_layer = GraphNetwork_layer(
            input_dim=8,
            output_dim=16,
            edge_attr_dim=3,
            aggr=aggr_method
        )
        output = gn_layer(x, edge_index, edge_attr)
        print(f"  ✓ Aggregation '{aggr_method}' works, output shape: {output.shape}")
    except Exception as e:
        print(f"  ✗ Aggregation '{aggr_method}' failed: {e}")
        all_passed = False

# Test gradient flow
print("\n" + "-" * 60)
print("Testing gradient flow (important for training):")
print("-" * 60)

try:
    gn_layer = GraphNetwork_layer(input_dim=8, output_dim=16, edge_attr_dim=3, aggr='mean')
    x = torch.randn(4, 8, requires_grad=True)
    edge_attr = torch.randn(6, 3)
    
    output = gn_layer(x, edge_index, edge_attr)
    loss = output.sum()
    loss.backward()
    
    if x.grad is not None:
        print(f"  ✓ Gradients flow correctly through the layer")
        print(f"    x.grad shape: {x.grad.shape}")
    else:
        print(f"  ✗ No gradients!")
        all_passed = False
        
except Exception as e:
    print(f"  ✗ Gradient test failed: {e}")
    all_passed = False

# Summary
print("\n" + "=" * 60)
if all_passed:
    print("STEP 4 PASSED! ✓")
    print("=" * 60)
    print("\nGNN layers are working correctly!")
    print("""
    What you created:
    ├── GNN/
    │   ├── __init__.py
    │   ├── losses.py        ✓
    │   └── layers.py        ✓  (MLP, GraphNetwork_layer)
    
    MLP:
    • Basic feedforward neural network
    • Used for encoding/decoding features
    
    GraphNetwork_layer:
    • Message passing between connected nodes
    • Each node collects info from neighbors
    • Updates its features based on neighborhood
    
    Visual of message passing:
    
        Node A ──edge──→ Node B
           │                │
           └── message ────→│
                            ↓
                    B updates itself
    """)
    print("You can proceed to Step 5.")
else:
    print("STEP 4 FAILED! ✗")
    print("=" * 60)
    print("\nPlease fix the errors above and run this test again.")
print("=" * 60)