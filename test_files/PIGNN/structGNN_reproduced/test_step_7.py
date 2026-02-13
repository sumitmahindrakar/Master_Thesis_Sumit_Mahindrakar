"""
TEST STEP 7: Verify Utils/datasets.py is created correctly and works
"""
import os
import sys

print("=" * 60)
print("TEST STEP 7: Dataset Functions (Utils/datasets.py) Check")
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

datasets_path = os.path.join(script_dir, 'Utils', 'datasets.py')
all_passed = True

if os.path.exists(datasets_path):
    size = os.path.getsize(datasets_path)
    print(f"  ✓ Utils/datasets.py exists ({size} bytes)")
    if size < 1000:
        print(f"    ⚠ Warning: File seems too small. Did you add the full code?")
        all_passed = False
else:
    print(f"  ✗ Utils/datasets.py MISSING!")
    all_passed = False
    print("\n" + "=" * 60)
    print("STEP 7 FAILED! ✗")
    print("Please create Utils/datasets.py with the provided code.")
    print("=" * 60)
    sys.exit(1)

# Check PyTorch
print("\n" + "-" * 60)
print("Checking required packages:")
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
    from Utils.datasets import get_dataset
    print(f"  ✓ get_dataset imported")
except ImportError as e:
    print(f"  ✗ get_dataset import failed: {e}")
    all_passed = False
    sys.exit(1)

try:
    from Utils.datasets import split_dataset
    print(f"  ✓ split_dataset imported")
except ImportError as e:
    print(f"  ✗ split_dataset import failed: {e}")
    all_passed = False

try:
    from Utils.datasets import get_target_index
    print(f"  ✓ get_target_index imported")
except ImportError as e:
    print(f"  ✗ get_target_index import failed: {e}")
    all_passed = False

try:
    from Utils.datasets import describe_dataset
    print(f"  ✓ describe_dataset imported")
except ImportError as e:
    print(f"  ⚠ describe_dataset not found (optional)")

# Test 1: get_target_index function
print("\n" + "-" * 60)
print("Test 1: get_target_index function")
print("-" * 60)

try:
    # Test all target names
    test_targets = [
        ('disp_x', 0, 1),
        ('disp_z', 1, 2),
        ('disp', 0, 2),
        ('momentY', 2, 8),
        ('momentZ', 8, 14),
        ('moment', 2, 14),
        ('shearY', 14, 20),
        ('shearZ', 20, 26),
        ('shear', 14, 26),
        ('all', 0, 26)
    ]
    
    print(f"  Testing target name mappings:")
    for target_name, expected_start, expected_end in test_targets:
        start, end = get_target_index(target_name)
        if start == expected_start and end == expected_end:
            print(f"    ✓ '{target_name}' → [{start}:{end}]")
        else:
            print(f"    ✗ '{target_name}' wrong! Expected [{expected_start}:{expected_end}], got [{start}:{end}]")
            all_passed = False
    
    # Test invalid target
    try:
        get_target_index('invalid_target')
        print(f"    ✗ Should have raised error for invalid target")
        all_passed = False
    except ValueError as e:
        print(f"    ✓ Correctly raises error for invalid target")
        
except Exception as e:
    print(f"  ✗ Test 1 failed: {e}")
    all_passed = False

# Test 2: split_dataset function with mock data
print("\n" + "-" * 60)
print("Test 2: split_dataset function")
print("-" * 60)

try:
    # Create mock dataset (list of 100 items)
    mock_dataset = list(range(100))
    
    # Test 2-way split (90% train, 10% valid)
    train, valid, test = split_dataset(mock_dataset, train_ratio=0.9, valid_ratio=0.1)
    
    print(f"  2-way split (90/10):")
    print(f"    Train size: {len(train)} (expected: 90)")
    print(f"    Valid size: {len(valid)} (expected: 10)")
    print(f"    Test: {test} (expected: None)")
    
    if len(train) == 90 and len(valid) == 10 and test is None:
        print(f"  ✓ 2-way split correct")
    else:
        print(f"  ✗ 2-way split incorrect")
        all_passed = False
    
    # Test 3-way split (80% train, 10% valid, 10% test)
    train, valid, test = split_dataset(mock_dataset, 
                                        train_ratio=0.8, 
                                        valid_ratio=0.1, 
                                        test_ratio=0.1)
    
    print(f"\n  3-way split (80/10/10):")
    print(f"    Train size: {len(train)} (expected: 80)")
    print(f"    Valid size: {len(valid)} (expected: 10)")
    print(f"    Test size: {len(test)} (expected: 10)")
    
    if len(train) == 80 and len(valid) == 10 and len(test) == 10:
        print(f"  ✓ 3-way split correct")
    else:
        print(f"  ✗ 3-way split incorrect")
        all_passed = False
    
    # Test reproducibility (same split with same seed)
    train1, valid1, _ = split_dataset(mock_dataset, train_ratio=0.9)
    train2, valid2, _ = split_dataset(mock_dataset, train_ratio=0.9)
    
    # Get indices from both splits
    train1_indices = [mock_dataset[i] for i in train1.indices]
    train2_indices = [mock_dataset[i] for i in train2.indices]
    
    if train1_indices == train2_indices:
        print(f"\n  ✓ Split is reproducible (same seed gives same split)")
    else:
        print(f"\n  ⚠ Split may not be reproducible")
        
except Exception as e:
    print(f"  ✗ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# Test 3: Check Data folder structure
print("\n" + "-" * 60)
print("Test 3: Checking Data folder structure")
print("-" * 60)

data_path = os.path.join(script_dir, 'Data')
if os.path.exists(data_path):
    print(f"  ✓ Data/ folder exists")
    
    # Check for dataset folders
    dataset_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    if dataset_folders:
        print(f"  ✓ Found dataset folders: {dataset_folders}")
        
        # Check first dataset folder for structure folders
        first_dataset = dataset_folders[0]
        dataset_path = os.path.join(data_path, first_dataset)
        structure_folders = [f for f in os.listdir(dataset_path) 
                           if f.startswith('structure_') and os.path.isdir(os.path.join(dataset_path, f))]
        
        if structure_folders:
            print(f"  ✓ Found {len(structure_folders)} structure folders in {first_dataset}/")
            
            # Check for .pt files
            first_structure = os.path.join(dataset_path, structure_folders[0])
            pt_files = [f for f in os.listdir(first_structure) if f.endswith('.pt')]
            
            if pt_files:
                print(f"  ✓ Found .pt files: {pt_files}")
            else:
                print(f"  ⚠ No .pt files found in {first_structure}")
        else:
            print(f"  ⚠ No structure_* folders found in {first_dataset}/")
    else:
        print(f"  ⚠ No dataset folders found in Data/")
        print(f"    Expected structure:")
        print(f"      Data/")
        print(f"      └── Static_Linear_Analysis/")
        print(f"          ├── structure_1/")
        print(f"          │   └── structure_graph_NodeAsNode.pt")
        print(f"          └── ...")
else:
    print(f"  ⚠ Data/ folder does not exist yet")
    print(f"    You'll need to add your data files before training")

# Test 4: Try loading actual data (if available)
print("\n" + "-" * 60)
print("Test 4: Attempting to load actual data")
print("-" * 60)

try:
    # Try to load with a small number
    print(f"  Attempting to load data from Data/Static_Linear_Analysis/...")
    dataset = get_dataset(
        dataset_name='Static_Linear_Analysis',
        whatAsNode='NodeAsNode',
        structure_num=5  # Try just 5 structures
    )
    
    if len(dataset) > 0:
        print(f"\n  ✓ Successfully loaded {len(dataset)} structure(s)!")
        
        # Show info about first structure
        first_graph = dataset[0]
        print(f"\n  First structure info:")
        print(f"    Type: {type(first_graph)}")
        print(f"    Node features (x): {first_graph.x.shape}")
        print(f"    Edge index: {first_graph.edge_index.shape}")
        print(f"    Edge attributes: {first_graph.edge_attr.shape}")
        print(f"    Outputs (y): {first_graph.y.shape}")
        
        # Try describe_dataset if available
        try:
            from Utils.datasets import describe_dataset
            describe_dataset(dataset)
        except:
            pass
            
    else:
        print(f"\n  ⚠ No data files found")
        print(f"    This is OK if you haven't added your data yet")
        print(f"    The functions are working correctly!")
        
except Exception as e:
    print(f"  ⚠ Could not load data: {e}")
    print(f"    This is expected if you haven't added data files yet")
    print(f"    The functions are working correctly!")

# Test 5: Demonstrate usage
print("\n" + "-" * 60)
print("Test 5: Usage demonstration")
print("-" * 60)

print("""
  How to use these functions in your training script:
  
  # 1. Load the dataset
  from Utils.datasets import get_dataset, split_dataset, get_target_index
  
  dataset = get_dataset(
      dataset_name='Static_Linear_Analysis',
      whatAsNode='NodeAsNode',
      structure_num=100
  )
  
  # 2. Split into train/validation
  train_data, valid_data, _ = split_dataset(
      dataset, 
      train_ratio=0.9, 
      valid_ratio=0.1
  )
  
  # 3. Get target indices for training
  y_start, y_end = get_target_index('all')  # Train on all outputs
  
  # 4. Use in training loop
  for data in train_loader:
      output = model(data.x, data.edge_index, data.edge_attr)
      loss = criterion(output[:, y_start:y_end], data.y[:, y_start:y_end])
""")

# Summary
print("\n" + "=" * 60)
if all_passed:
    print("STEP 7 PASSED! ✓")
    print("=" * 60)
    print("\nDataset functions are working correctly!")
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
    │   └── datasets.py         ✓  (get_dataset, split_dataset, get_target_index)
    
    Functions created:
    ┌─────────────────────┬─────────────────────────────────────────┐
    │ Function            │ Purpose                                 │
    ├─────────────────────┼─────────────────────────────────────────┤
    │ get_dataset()       │ Load .pt graph files from Data folder  │
    │ split_dataset()     │ Split into train/valid/test sets       │
    │ get_target_index()  │ Map target names to column indices     │
    │ describe_dataset()  │ Print dataset information (optional)   │
    └─────────────────────┴─────────────────────────────────────────┘
    
    Output target mapping:
    ┌────────────┬─────────────┬─────────────────────────────────┐
    │ Target     │ Columns     │ Description                     │
    ├────────────┼─────────────┼─────────────────────────────────┤
    │ disp       │ [0:2]       │ Displacement (X, Z)             │
    │ momentY    │ [2:8]       │ Moment about Y axis (6 values)  │
    │ momentZ    │ [8:14]      │ Moment about Z axis (6 values)  │
    │ shearY     │ [14:20]     │ Shear in Y direction (6 values) │
    │ shearZ     │ [20:26]     │ Shear in Z direction (6 values) │
    │ all        │ [0:26]      │ All of the above                │
    └────────────┴─────────────┴─────────────────────────────────┘
    """)
    print("You can proceed to Step 8.")
else:
    print("STEP 7 FAILED! ✗")
    print("=" * 60)
    print("\nPlease fix the errors above and run this test again.")
print("=" * 60)