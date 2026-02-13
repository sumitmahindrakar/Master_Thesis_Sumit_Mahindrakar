"""
TEST STEP 9: Verify Utils/plot.py is created correctly and works
"""
import os
import sys

print("=" * 60)
print("TEST STEP 9: Plotting Functions (Utils/plot.py) Check")
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

plot_path = os.path.join(script_dir, 'Utils', 'plot.py')
all_passed = True

if os.path.exists(plot_path):
    size = os.path.getsize(plot_path)
    print(f"  ✓ Utils/plot.py exists ({size} bytes)")
    if size < 2000:
        print(f"    ⚠ Warning: File seems too small. Did you add the full code?")
        all_passed = False
else:
    print(f"  ✗ Utils/plot.py MISSING!")
    all_passed = False
    print("\n" + "=" * 60)
    print("STEP 9 FAILED! ✗")
    print("Please create Utils/plot.py with the provided code.")
    print("=" * 60)
    sys.exit(1)

# Check required packages
print("\n" + "-" * 60)
print("Checking required packages:")
print("-" * 60)

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    print(f"  ✓ matplotlib: {matplotlib.__version__}")
except ImportError:
    print(f"  ✗ matplotlib not installed!")
    print(f"    Run: pip install matplotlib")
    sys.exit(1)

try:
    import numpy as np
    print(f"  ✓ numpy: {np.__version__}")
except ImportError:
    print(f"  ✗ numpy not installed!")
    sys.exit(1)

try:
    import networkx as nx
    print(f"  ✓ networkx: {nx.__version__}")
except ImportError:
    print(f"  ⚠ networkx not installed (optional, for graph visualization)")
    print(f"    Run: pip install networkx")

# Try to import functions
print("\n" + "-" * 60)
print("Testing imports:")
print("-" * 60)

try:
    from Utils.plot import print_space
    print(f"  ✓ print_space imported")
except ImportError as e:
    print(f"  ✗ print_space import failed: {e}")
    all_passed = False

try:
    from Utils.plot import plot_learningCurve
    print(f"  ✓ plot_learningCurve imported")
except ImportError as e:
    print(f"  ✗ plot_learningCurve import failed: {e}")
    all_passed = False

try:
    from Utils.plot import plot_lossCurve
    print(f"  ✓ plot_lossCurve imported")
except ImportError as e:
    print(f"  ✗ plot_lossCurve import failed: {e}")
    all_passed = False

try:
    from Utils.plot import visualize_graph
    print(f"  ✓ visualize_graph imported")
except ImportError as e:
    print(f"  ⚠ visualize_graph import failed: {e}")

try:
    from Utils.plot import plot_prediction_comparison, plot_error_distribution
    print(f"  ✓ Additional plotting functions imported")
except ImportError as e:
    print(f"  ⚠ Some additional functions not found: {e}")

# Create test output directory
test_output_dir = os.path.join(script_dir, 'test_plots')
os.makedirs(test_output_dir, exist_ok=True)
print(f"\n  Test plots will be saved to: {test_output_dir}")

# Test 1: print_space function
print("\n" + "-" * 60)
print("Test 1: print_space function")
print("-" * 60)

try:
    from Utils.plot import print_space
    
    space = print_space()
    print(f"  ✓ print_space() returns a string")
    print(f"    Length: {len(space)} characters")
    print(f"    Contains separator: {'=' * 10 in space}")
    
except Exception as e:
    print(f"  ✗ Test 1 failed: {e}")
    all_passed = False

# Test 2: plot_learningCurve
print("\n" + "-" * 60)
print("Test 2: plot_learningCurve function")
print("-" * 60)

try:
    from Utils.plot import plot_learningCurve
    
    # Create mock accuracy data
    num_epochs = 50
    epochs = np.arange(num_epochs)
    
    # Simulate typical learning curve
    train_acc = 0.5 + 0.4 * (1 - np.exp(-epochs / 10)) + np.random.randn(num_epochs) * 0.02
    valid_acc = 0.4 + 0.35 * (1 - np.exp(-epochs / 15)) + np.random.randn(num_epochs) * 0.03
    test_acc = np.zeros(num_epochs)  # Not used
    
    # Clip to valid range
    train_acc = np.clip(train_acc, 0, 1)
    valid_acc = np.clip(valid_acc, 0, 1)
    
    accuracy_record = np.array([train_acc, valid_acc, test_acc])
    
    print(f"  Created mock accuracy data:")
    print(f"    Shape: {accuracy_record.shape}")
    print(f"    Final train accuracy: {train_acc[-1]:.3f}")
    print(f"    Final valid accuracy: {valid_acc[-1]:.3f}")
    print(f"    Best valid accuracy: {valid_acc.max():.3f}")
    
    # Generate plot
    plot_learningCurve(
        accuracy_record,
        test_output_dir,
        title="Test Learning Curve",
        target="test"
    )
    
    # Check if file was created
    expected_file = os.path.join(test_output_dir, "LearningCurve_test.png")
    if os.path.exists(expected_file):
        print(f"  ✓ Learning curve plot created successfully")
    else:
        print(f"  ✗ Plot file not found!")
        all_passed = False
        
except Exception as e:
    print(f"  ✗ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# Test 3: plot_lossCurve
print("\n" + "-" * 60)
print("Test 3: plot_lossCurve function")
print("-" * 60)

try:
    from Utils.plot import plot_lossCurve
    
    # Create mock loss data (decreasing over time)
    num_epochs = 50
    epochs = np.arange(num_epochs)
    
    # Simulate typical loss curve
    train_loss = 10 * np.exp(-epochs / 15) + 0.5 + np.random.randn(num_epochs) * 0.1
    valid_loss = 12 * np.exp(-epochs / 20) + 0.8 + np.random.randn(num_epochs) * 0.15
    test_loss = np.zeros(num_epochs)
    
    # Ensure positive
    train_loss = np.maximum(train_loss, 0.1)
    valid_loss = np.maximum(valid_loss, 0.1)
    
    loss_record = np.array([train_loss, valid_loss, test_loss])
    
    print(f"  Created mock loss data:")
    print(f"    Shape: {loss_record.shape}")
    print(f"    Initial train loss: {train_loss[0]:.3f}")
    print(f"    Final train loss: {train_loss[-1]:.3f}")
    print(f"    Final valid loss: {valid_loss[-1]:.3f}")
    
    # Generate plot
    plot_lossCurve(
        loss_record,
        test_output_dir,
        title="Test Loss Curve",
        target="test"
    )
    
    # Check if file was created
    expected_file = os.path.join(test_output_dir, "LossCurve_test.png")
    if os.path.exists(expected_file):
        print(f"  ✓ Loss curve plot created successfully")
    else:
        print(f"  ✗ Plot file not found!")
        all_passed = False
        
except Exception as e:
    print(f"  ✗ Test 3 failed: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# Test 4: plot_prediction_comparison
print("\n" + "-" * 60)
print("Test 4: plot_prediction_comparison function")
print("-" * 60)

try:
    from Utils.plot import plot_prediction_comparison
    
    # Create mock prediction data
    y_true = np.random.randn(100) * 10
    y_pred = y_true + np.random.randn(100) * 2  # Add some noise
    
    print(f"  Created mock prediction data:")
    print(f"    Samples: {len(y_true)}")
    print(f"    True values range: [{y_true.min():.2f}, {y_true.max():.2f}]")
    print(f"    Pred values range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    
    # Generate plot
    plot_prediction_comparison(
        y_pred,
        y_true,
        test_output_dir,
        target_name="test"
    )
    
    # Check if file was created
    expected_file = os.path.join(test_output_dir, "Prediction_test.png")
    if os.path.exists(expected_file):
        print(f"  ✓ Prediction comparison plot created successfully")
    else:
        print(f"  ✗ Plot file not found!")
        all_passed = False
        
except Exception as e:
    print(f"  ✗ Test 4 failed: {e}")
    all_passed = False

# Test 5: plot_error_distribution
print("\n" + "-" * 60)
print("Test 5: plot_error_distribution function")
print("-" * 60)

try:
    from Utils.plot import plot_error_distribution
    
    # Use same data as Test 4
    plot_error_distribution(
        y_pred,
        y_true,
        test_output_dir,
        target_name="test"
    )
    
    # Check if file was created
    expected_file = os.path.join(test_output_dir, "Error_test.png")
    if os.path.exists(expected_file):
        print(f"  ✓ Error distribution plot created successfully")
    else:
        print(f"  ✗ Plot file not found!")
        all_passed = False
        
except Exception as e:
    print(f"  ✗ Test 5 failed: {e}")
    all_passed = False

# Test 6: visualize_graph (if networkx available)
print("\n" + "-" * 60)
print("Test 6: visualize_graph function")
print("-" * 60)

try:
    from Utils.plot import visualize_graph
    import torch
    from torch_geometric.data import Data
    
    # Create mock graph
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0, 0, 2],
        [1, 0, 2, 1, 3, 2, 0, 3, 2, 0]
    ])
    x = torch.randn(4, 5)
    
    mock_graph = Data(x=x, edge_index=edge_index)
    
    print(f"  Created mock graph:")
    print(f"    Nodes: {mock_graph.x.shape[0]}")
    print(f"    Edges: {mock_graph.edge_index.shape[1]}")
    
    # Generate plot
    visualize_graph(
        mock_graph,
        test_output_dir,
        name="test_graph"
    )
    
    # Check if file was created
    expected_file = os.path.join(test_output_dir, "test_graph.png")
    if os.path.exists(expected_file):
        print(f"  ✓ Graph visualization created successfully")
    else:
        print(f"  ⚠ Graph visualization skipped (networkx may not be available)")
        
except Exception as e:
    print(f"  ⚠ Test 6 skipped: {e}")

# Test 7: Check all created files
print("\n" + "-" * 60)
print("Test 7: Checking created files")
print("-" * 60)

created_files = os.listdir(test_output_dir)
print(f"  Files in {test_output_dir}:")
for f in created_files:
    filepath = os.path.join(test_output_dir, f)
    size = os.path.getsize(filepath)
    print(f"    ✓ {f} ({size:,} bytes)")

if len(created_files) >= 4:
    print(f"\n  ✓ All expected plot files created!")
else:
    print(f"\n  ⚠ Some plot files may be missing")

# Visual explanation
print("\n" + "-" * 60)
print("Understanding the Plots:")
print("-" * 60)
print("""
  LEARNING CURVE (Accuracy vs Epochs):
  ────────────────────────────────────
  • Shows how accuracy improves during training
  • Train line should increase over time
  • Valid line shows generalization performance
  • Gap between train/valid indicates overfitting
  
  Good training:
    Accuracy
    1.0 ─┐     ═══════════ Train
         │   ══
         │ ══  ─────────── Valid
    0.5 ─┤══
         └────────────────
           Epochs
  
  ────────────────────────────────────
  
  LOSS CURVE (Loss vs Epochs):
  ────────────────────────────────────
  • Shows how loss decreases during training
  • Both curves should decrease
  • Flattening indicates convergence
  
  Good training:
    Loss
    High─┐══
         │  ══
         │    ════════════ Valid
         │      ══
    Low ─┤        ════════ Train
         └────────────────
           Epochs
  
  ────────────────────────────────────
  
  PREDICTION VS ACTUAL:
  ────────────────────────────────────
  • Perfect predictions lie on diagonal
  • Scatter around diagonal shows error
  • R² score measures fit quality
  
      Predicted
         │    ●  /
         │   ● ●/
         │  ●●/●
         │ ●/●
         │/
         └──────── Actual
""")

# Summary
print("\n" + "=" * 60)
if all_passed:
    print("STEP 9 PASSED! ✓")
    print("=" * 60)
    print("\nPlotting functions are working correctly!")
    print(f"""
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
    │   ├── normalization.py    ✓
    │   └── plot.py             ✓  (visualization functions)
    
    Key functions:
    ┌────────────────────────────┬─────────────────────────────────┐
    │ Function                   │ Purpose                         │
    ├────────────────────────────┼─────────────────────────────────┤
    │ plot_learningCurve()       │ Plot accuracy over epochs       │
    │ plot_lossCurve()           │ Plot loss over epochs           │
    │ visualize_graph()          │ Visualize graph structure       │
    │ plot_prediction_comparison()│ Scatter plot: pred vs actual   │
    │ plot_error_distribution()  │ Histogram of errors             │
    │ print_space()              │ Formatted console separator     │
    └────────────────────────────┴─────────────────────────────────┘
    
    Test plots saved in: {test_output_dir}
    """)
    print("You can proceed to Step 10.")
else:
    print("STEP 9 FAILED! ✗")
    print("=" * 60)
    print("\nPlease fix the errors above and run this test again.")
print("=" * 60)

# Clean up option
print(f"\nNote: Test plots were saved to '{test_output_dir}'")
print("You can delete this folder after reviewing the plots.")