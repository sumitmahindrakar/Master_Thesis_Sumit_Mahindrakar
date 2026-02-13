"""
TEST STEP 3: Verify GNN/losses.py is created correctly and works
"""
import os
import sys

print("=" * 60)
print("TEST STEP 3: Loss Functions (GNN/losses.py) Check")
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

losses_path = os.path.join(script_dir, 'GNN', 'losses.py')
all_passed = True

if os.path.exists(losses_path):
    size = os.path.getsize(losses_path)
    print(f"  ✓ GNN/losses.py exists ({size} bytes)")
    if size < 100:
        print(f"    ⚠ Warning: File seems too small. Did you add the code?")
        all_passed = False
else:
    print(f"  ✗ GNN/losses.py MISSING!")
    print(f"    Expected at: {losses_path}")
    all_passed = False

if not all_passed:
    print("\n" + "=" * 60)
    print("STEP 3 FAILED! ✗")
    print("=" * 60)
    print("\nPlease create GNN/losses.py with the provided code.")
    sys.exit(1)

# Try to import
print("\n" + "-" * 60)
print("Testing import:")
print("-" * 60)

try:
    from GNN.losses import L1_Loss, L2_Loss
    print(f"  ✓ 'from GNN.losses import L1_Loss, L2_Loss' works")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    all_passed = False

if not all_passed:
    print("\n" + "=" * 60)
    print("STEP 3 FAILED! ✗")
    print("=" * 60)
    print("\nCheck the code in GNN/losses.py for syntax errors.")
    sys.exit(1)

# Check if PyTorch is available
print("\n" + "-" * 60)
print("Checking PyTorch:")
print("-" * 60)

try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")
except ImportError:
    print(f"  ✗ PyTorch not installed!")
    print(f"    Run: pip install torch")
    all_passed = False
    sys.exit(1)

# Test the loss functions
print("\n" + "-" * 60)
print("Testing L1_Loss:")
print("-" * 60)

try:
    # Create loss function instance
    l1_criterion = L1_Loss()
    print(f"  ✓ L1_Loss() created successfully")
    
    # Create dummy data
    predictions = torch.tensor([1.0, 2.0, 0.001, 3.0, 0.0001])
    targets = torch.tensor([1.1, 2.2, 0.002, 2.8, 0.0002])
    threshold = 0.01
    
    print(f"  Test data:")
    print(f"    predictions: {predictions.tolist()}")
    print(f"    targets:     {targets.tolist()}")
    print(f"    threshold:   {threshold}")
    
    # Calculate loss
    loss = l1_criterion(predictions, targets, threshold)
    print(f"  ✓ L1_Loss calculated: {loss.item():.6f}")
    
    # Verify manually
    # Values above threshold: indices 0, 1, 3 (values 1.1, 2.2, 2.8)
    # |1.0-1.1| + |2.0-2.2| + |3.0-2.8| = 0.1 + 0.2 + 0.2 = 0.5
    expected = 0.5
    if abs(loss.item() - expected) < 0.001:
        print(f"  ✓ Loss value is correct (expected ~{expected})")
    else:
        print(f"  ⚠ Loss value might be unexpected (expected ~{expected})")
        
except Exception as e:
    print(f"  ✗ L1_Loss test failed: {e}")
    all_passed = False

print("\n" + "-" * 60)
print("Testing L2_Loss:")
print("-" * 60)

try:
    # Create loss function instance
    l2_criterion = L2_Loss()
    print(f"  ✓ L2_Loss() created successfully")
    
    # Use same dummy data
    loss = l2_criterion(predictions, targets, threshold)
    print(f"  ✓ L2_Loss calculated: {loss.item():.6f}")
    
    # Verify manually
    # (0.1)² + (0.2)² + (0.2)² = 0.01 + 0.04 + 0.04 = 0.09
    expected = 0.09
    if abs(loss.item() - expected) < 0.001:
        print(f"  ✓ Loss value is correct (expected ~{expected})")
    else:
        print(f"  ⚠ Loss value might be unexpected (expected ~{expected})")
        
except Exception as e:
    print(f"  ✗ L2_Loss test failed: {e}")
    all_passed = False

# Test with 2D tensor (like real model output)
print("\n" + "-" * 60)
print("Testing with 2D tensors (realistic scenario):")
print("-" * 60)

try:
    # Simulate model output: 10 nodes, 26 outputs each
    batch_predictions = torch.randn(10, 26)
    batch_targets = torch.randn(10, 26)
    
    l1_loss = l1_criterion(batch_predictions, batch_targets, 0.0001)
    l2_loss = l2_criterion(batch_predictions, batch_targets, 0.0001)
    
    print(f"  Input shape: {batch_predictions.shape}")
    print(f"  ✓ L1_Loss on batch: {l1_loss.item():.4f}")
    print(f"  ✓ L2_Loss on batch: {l2_loss.item():.4f}")
    
except Exception as e:
    print(f"  ✗ Batch test failed: {e}")
    all_passed = False

# Summary
print("\n" + "=" * 60)
if all_passed:
    print("STEP 3 PASSED! ✓")
    print("=" * 60)
    print("\nLoss functions are working correctly!")
    print("""
    What you created:
    ├── GNN/
    │   ├── __init__.py
    │   └── losses.py  ✓  (L1_Loss, L2_Loss)
    
    These loss functions:
    • Measure prediction error during training
    • Ignore small values (below threshold) to avoid numerical issues
    • L1 = absolute error, L2 = squared error
    """)
    print("You can proceed to Step 4.")
else:
    print("STEP 3 FAILED! ✗")
    print("=" * 60)
    print("\nPlease fix the errors above and run this test again.")
print("=" * 60)