"""
analyze_data.py

Analyze the available structural data and understand its organization.
"""
import torch
import os

print("=" * 70)
print("ANALYZING YOUR STRUCTURAL DATA")
print("=" * 70)

# Path to data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "Data", "Static_Linear_Analysis")

print(f"\nData directory: {data_dir}")

# Find all structure folders
if not os.path.exists(data_dir):
    print(f"\nERROR: Data directory not found!")
    exit(1)

structure_folders = []
for item in os.listdir(data_dir):
    item_path = os.path.join(data_dir, item)
    if os.path.isdir(item_path) and item.startswith("structure_"):
        try:
            # Extract number from folder name
            num = int(item.replace("structure_", ""))
            structure_folders.append((num, item))
        except ValueError:
            pass

# Sort by number
structure_folders.sort(key=lambda x: x[0])

print(f"\nFound {len(structure_folders)} structure folders")

# Show distribution
print("\n" + "-" * 70)
print("Structure folder distribution:")
print("-" * 70)

numbers = [num for num, name in structure_folders]
print(f"  Minimum: structure_{min(numbers)}")
print(f"  Maximum: structure_{max(numbers)}")

# Check for gaps
expected = set(range(min(numbers), max(numbers) + 1))
actual = set(numbers)
missing = expected - actual

if missing:
    print(f"\n  Missing structures: {len(missing)}")
    if len(missing) <= 20:
        print(f"  Missing numbers: {sorted(missing)}")
    else:
        print(f"  First 10 missing: {sorted(missing)[:10]}")
else:
    print(f"\n  ✓ All structures from {min(numbers)} to {max(numbers)} are present")

# Group by ranges
print("\n  Structure ranges:")
ranges = {}
for num in numbers:
    if num < 10:
        key = "1-9"
    elif num < 100:
        key = "10-99"
    elif num < 1000:
        key = "100-999"
    else:
        key = "1000+"
    ranges[key] = ranges.get(key, 0) + 1

for key in ["1-9", "10-99", "100-999", "1000+"]:
    if key in ranges:
        print(f"    {key}: {ranges[key]} structures")

# Check file types in first few structures
print("\n" + "-" * 70)
print("Available file types (checking first 5 structures):")
print("-" * 70)

file_types = set()
for num, name in structure_folders[:5]:
    folder_path = os.path.join(data_dir, name)
    for f in os.listdir(folder_path):
        if f.endswith('.pt'):
            file_types.add(f)

for ft in sorted(file_types):
    print(f"  ✓ {ft}")

# Load and analyze one sample
print("\n" + "-" * 70)
print("Sample data analysis (first structure):")
print("-" * 70)

sample_folder = os.path.join(data_dir, structure_folders[0][1])
sample_file = os.path.join(sample_folder, "structure_graph_NodeAsNode.pt")

if os.path.exists(sample_file):
    data = torch.load(sample_file)
    print(f"\n  Structure: {structure_folders[0][1]}")
    print(f"  File: structure_graph_NodeAsNode.pt")
    print(f"\n  Data attributes: {data.keys() if hasattr(data, 'keys') else dir(data)}")
    
    if hasattr(data, 'x'):
        print(f"\n  Node features (x):")
        print(f"    Shape: {data.x.shape}")
        print(f"    Dtype: {data.x.dtype}")
        print(f"    Sample values (first node): {data.x[0, :5].tolist()}...")
    
    if hasattr(data, 'edge_index'):
        print(f"\n  Edge index:")
        print(f"    Shape: {data.edge_index.shape}")
        print(f"    Num edges: {data.edge_index.shape[1]}")
    
    if hasattr(data, 'edge_attr'):
        print(f"\n  Edge attributes:")
        print(f"    Shape: {data.edge_attr.shape}")
    
    if hasattr(data, 'y'):
        print(f"\n  Targets (y):")
        print(f"    Shape: {data.y.shape}")
        print(f"    Dtype: {data.y.dtype}")
else:
    print(f"  Could not find: {sample_file}")

# Provide recommendations
print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

# Get list of valid structure numbers
valid_structures = [num for num, name in structure_folders]

print(f"""
  You have {len(structure_folders)} structures available.
  
  To train on your data, you can:
  
  1. Train on the first N structures (by sorted order):
     python train.py --data_num {min(len(structure_folders), 100)}
  
  2. Or I can update the dataset loading code to handle your 
     specific structure numbering.
  
  Structure numbers available: {valid_structures[:10]}... (and more)
""")

# Save the list of available structures
available_file = os.path.join(script_dir, "available_structures.txt")
with open(available_file, 'w') as f:
    for num, name in structure_folders:
        f.write(f"{num}\n")
print(f"  Saved list of available structures to: available_structures.txt")

print("=" * 70)