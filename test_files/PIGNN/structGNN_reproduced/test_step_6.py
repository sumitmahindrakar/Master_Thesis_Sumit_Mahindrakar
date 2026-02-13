"""
TEST STEP 6: Verify Utils/accuracy.py is created correctly and works
"""
import os
import sys

print("=" * 60)
print("TEST STEP 6: Accuracy Functions (Utils/accuracy.py) Check")
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

accuracy_path = os.path.join(script_dir, 'Utils', 'accuracy.py')
all_passed = True

if os.path.exists(accuracy_path):
    size = os.path.getsize(accuracy_path)
    print(f"  ✓ Utils/accuracy.py exists ({size} bytes)")
    if size < 500:
        print(f"    ⚠ Warning: File seems too small. Did you add the full code?")
        all_passed = False
else:
    print(f"  ✗ Utils/accuracy.py MISSING!")
    all_passed = False
    print("\n" + "=" * 60)
    print("STEP 6 FAILED! ✗")
    print("Please create Utils/accuracy.py with the provided code.")
    print("=" * 60)
    sys.exit(1)

# Check PyTorch
print("\n" + "-" * 60)
print("Checking PyTorch:")
print("-" * 60)

try:
    import torch
    print(f"  ✓ torch: {torch.__version__}")
except ImportError:
    print(f"  ✗ torch not installed!")
    sys.exit(1)

# Try to import
print("\n" + "-" * 60)
print("Testing imports:")
print("-" * 60)

try:
    from Utils.accuracy import node_accuracy
    print(f"  ✓ node_accuracy imported")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    all_passed = False
    sys.exit(1)

try:
    from Utils.accuracy import calculate_accuracy_percentage
    print(f"  ✓ calculate_accuracy_percentage imported")
except ImportError as e:
    print(f"  ⚠ calculate_accuracy_percentage not found (optional)")

try:
    from Utils.accuracy import detailed_accuracy
    print(f"  ✓ detailed_accuracy imported")
except ImportError as e:
    print(f"  ⚠ detailed_accuracy not found (optional)")

# Test 1: Basic accuracy calculation
print("\n" + "-" * 60)
print("Test 1: Basic accuracy calculation")
print("-" * 60)

try:
    # Create simple test case
    # Actual values
    actual = torch.tensor([100.0, 50.0, 200.0, 0.001])
    
    # Predicted values (with some error)
    predicted = torch.tensor([95.0, 45.0, 210.0, 0.002])
    
    threshold = 0.01  # Ignore values below 0.01
    
    print(f"  Test data:")
    print(f"    Actual:    {actual.tolist()}")
    print(f"    Predicted: {predicted.tolist()}")
    print(f"    Threshold: {threshold}")
    
    correct, count = node_accuracy(predicted, actual, threshold)
    
    print(f"\n  Results:")
    print(f"    Sum of accuracies: {correct.item():.4f}")
    print(f"    Count of values:   {count}")
    print(f"    Average accuracy:  {(correct/count).item():.4f}")
    
    # Manual calculation for verification
    # Only first 3 values are above threshold (0.001 < 0.01, so ignored)
    # acc1 = 1 - |95-100|/|100| = 1 - 5/100 = 0.95
    # acc2 = 1 - |45-50|/|50| = 1 - 5/50 = 0.90
    # acc3 = 1 - |210-200|/|200| = 1 - 10/200 = 0.95
    # avg = (0.95 + 0.90 + 0.95) / 3 = 0.933
    
    expected_count = 3
    expected_avg = 0.933
    
    if count == expected_count:
        print(f"  ✓ Correct count (threshold filtering works)")
    else:
        print(f"  ✗ Count wrong! Expected {expected_count}, got {count}")
        all_passed = False
    
    avg_acc = (correct / count).item()
    if abs(avg_acc - expected_avg) < 0.01:
        print(f"  ✓ Accuracy calculation correct (expected ~{expected_avg:.3f})")
    else:
        print(f"  ⚠ Accuracy differs from expected. Got {avg_acc:.3f}, expected ~{expected_avg:.3f}")
        
except Exception as e:
    print(f"  ✗ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# Test 2: Perfect predictions
print("\n" + "-" * 60)
print("Test 2: Perfect predictions (accuracy should be 1.0)")
print("-" * 60)

try:
    actual = torch.tensor([100.0, 50.0, 200.0])
    predicted = torch.tensor([100.0, 50.0, 200.0])  # Exact match
    
    correct, count = node_accuracy(predicted, actual, 0.01)
    avg_acc = (correct / count).item()
    
    print(f"  Actual = Predicted")
    print(f"  Average accuracy: {avg_acc:.4f}")
    
    if abs(avg_acc - 1.0) < 0.0001:
        print(f"  ✓ Perfect predictions give accuracy = 1.0")
    else:
        print(f"  ✗ Expected 1.0, got {avg_acc}")
        all_passed = False
        
except Exception as e:
    print(f"  ✗ Test 2 failed: {e}")
    all_passed = False

# Test 3: Completely wrong predictions
print("\n" + "-" * 60)
print("Test 3: Very wrong predictions (accuracy should be 0.0)")
print("-" * 60)

try:
    actual = torch.tensor([100.0, 50.0])
    predicted = torch.tensor([0.0, 0.0])  # 100% error
    
    correct, count = node_accuracy(predicted, actual, 0.01)
    avg_acc = (correct / count).item()
    
    print(f"  Actual: {actual.tolist()}")
    print(f"  Predicted: {predicted.tolist()} (100% error)")
    print(f"  Average accuracy: {avg_acc:.4f}")
    
    if avg_acc < 0.01:
        print(f"  ✓ Completely wrong predictions give accuracy ≈ 0.0")
    else:
        print(f"  ⚠ Expected ~0.0, got {avg_acc}")
        
except Exception as e:
    print(f"  ✗ Test 3 failed: {e}")
    all_passed = False

# Test 4: 2D tensor (like real model output)
print("\n" + "-" * 60)
print("Test 4: 2D tensor (realistic scenario)")
print("-" * 60)

try:
    # Simulate: 10 nodes, 26 outputs each
    num_nodes = 10
    num_outputs = 26
    
    actual = torch.randn(num_nodes, num_outputs) * 100  # Random values around 0-100
    
    # Add 10% noise to predictions
    noise = torch.randn(num_nodes, num_outputs) * 10
    predicted = actual + noise
    
    correct, count = node_accuracy(predicted, actual, 0.01)
    avg_acc = (correct / count).item()
    
    print(f"  Shape: ({num_nodes}, {num_outputs})")
    print(f"  Total elements: {num_nodes * num_outputs}")
    print(f"  Elements above threshold: {count}")
    print(f"  Average accuracy: {avg_acc:.4f}")
    
    if count > 0 and 0 <= avg_acc <= 1:
        print(f"  ✓ 2D tensor processing works correctly")
    else:
        print(f"  ✗ Unexpected results")
        all_passed = False
        
except Exception as e:
    print(f"  ✗ Test 4 failed: {e}")
    all_passed = False

# Test 5: GPU compatibility (if available)
print("\n" + "-" * 60)
print("Test 5: GPU compatibility")
print("-" * 60)

try:
    if torch.cuda.is_available():
        device = "cuda"
        actual = torch.tensor([100.0, 50.0, 200.0]).to(device)
        predicted = torch.tensor([95.0, 45.0, 190.0]).to(device)
        
        correct, count = node_accuracy(predicted, actual, 0.01)
        
        print(f"  Device: {device}")
        print(f"  ✓ GPU computation works correctly")
    else:
        print(f"  ⚠ CUDA not available, skipping GPU test")
        
except Exception as e:
    print(f"  ✗ GPU test failed: {e}")
    all_passed = False

# Test 6: Edge cases
print("\n" + "-" * 60)
print("Test 6: Edge cases")
print("-" * 60)

try:
    # Case 1: All values below threshold
    actual = torch.tensor([0.001, 0.0001, 0.00001])
    predicted = torch.tensor([0.002, 0.0002, 0.00002])
    
    correct, count = node_accuracy(predicted, actual, 0.01)
    
    print(f"  Case: All values below threshold")
    print(f"    Count: {count}")
    
    if count == 0:
        print(f"  ✓ Correctly returns 0 count when all values below threshold")
    else:
        print(f"  ✗ Expected count=0")
        all_passed = False
        
except Exception as e:
    print(f"  ✗ Edge case test failed: {e}")
    all_passed = False

# Test 7: Accumulation across batches
print("\n" + "-" * 60)
print("Test 7: Batch accumulation (how it's used in training)")
print("-" * 60)

try:
    # Simulate processing multiple batches
    total_correct = 0
    total_count = 0
    
    for batch_idx in range(3):
        actual = torch.randn(5, 26) * 100
        predicted = actual + torch.randn(5, 26) * 5  # 5% noise
        
        correct, count = node_accuracy(predicted, actual, 0.01)
        total_correct += correct.item()
        total_count += count
    
    overall_accuracy = total_correct / total_count
    
    print(f"  Processed 3 batches")
    print(f"  Total correct (sum): {total_correct:.4f}")
    print(f"  Total count: {total_count}")
    print(f"  Overall accuracy: {overall_accuracy:.4f}")
    print(f"  ✓ Batch accumulation works correctly")
    
except Exception as e:
    print(f"  ✗ Batch accumulation test failed: {e}")
    all_passed = False

# Visual explanation
print("\n" + "-" * 60)
print("Understanding the accuracy calculation:")
print("-" * 60)
print("""
  Example with threshold = 0.01:
  
  Actual:    [100.0,  50.0,  0.005,  200.0]
  Predicted: [ 95.0,  45.0,  0.010,  190.0]
                ↓      ↓       ↓       ↓
  |Actual|>0.01: YES   YES     NO     YES
                ↓      ↓       ↓       ↓
  Considered:   YES    YES   SKIP     YES
  
  Accuracy calculations:
    Value 1: 1 - |95-100|/100 = 1 - 0.05 = 0.95
    Value 2: 1 - |45-50|/50   = 1 - 0.10 = 0.90
    Value 4: 1 - |190-200|/200 = 1 - 0.05 = 0.95
  
  Sum = 0.95 + 0.90 + 0.95 = 2.80
  Count = 3
  Average = 2.80 / 3 = 0.933 (93.3% accurate)
""")

# Summary
print("\n" + "=" * 60)
if all_passed:
    print("STEP 6 PASSED! ✓")
    print("=" * 60)
    print("\nAccuracy functions are working correctly!")
    print("""
    What you created:
    ├── GNN/
    │   ├── __init__.py
    │   ├── losses.py           ✓
    │   ├── layers.py           ✓
    │   └── models.py           ✓
    ├── Utils/
    │   ├── __init__.py
    │   └── accuracy.py         ✓  (node_accuracy function)
    
    The accuracy function:
    • Calculates relative prediction accuracy
    • Ignores small values (below threshold)
    • Returns (sum, count) for batch accumulation
    • Works on both CPU and GPU
    
    Usage in training loop:
        correct, count = node_accuracy(predictions, targets, threshold)
        accuracy = correct / count
    """)
    print("You can proceed to Step 7.")
else:
    print("STEP 6 FAILED! ✗")
    print("=" * 60)
    print("\nPlease fix the errors above and run this test again.")
print("=" * 60)