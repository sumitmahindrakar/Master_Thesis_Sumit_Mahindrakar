"""
check_data_format.py

Analyze the actual data format to update normalization code.
"""
import torch
import os

print("=" * 70)
print("CHECKING DATA FORMAT")
print("=" * 70)

# Load sample data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(
    script_dir, "Data", "Static_Linear_Analysis", 
    "structure_1", "structure_graph_NodeAsNode.pt"
)

data = torch.load(data_path)

print("\n" + "-" * 70)
print("NODE FEATURES (x):")
print("-" * 70)
print(f"Shape: {data.x.shape}")
print(f"Number of nodes: {data.x.shape[0]}")
print(f"Number of features: {data.x.shape[1]}")

print("\nFeature statistics:")
print(f"{'Index':<8} {'Min':<15} {'Max':<15} {'Mean':<15} {'Likely content'}")
print("-" * 70)

for i in range(data.x.shape[1]):
    col = data.x[:, i]
    min_val = col.min().item()
    max_val = col.max().item()
    mean_val = col.mean().item()
    
    # Guess what the feature might be
    if min_val >= 0 and max_val <= 20 and min_val == int(min_val) and max_val == int(max_val):
        guess = "Grid number / Index"
    elif min_val >= 0 and max_val <= 2 and len(col.unique()) <= 3:
        guess = "Binary / Categorical"
    elif abs(max_val) > 100:
        guess = "Force / Large value"
    elif 0 < max_val < 100:
        guess = "Coordinate / Position"
    else:
        guess = "Unknown"
    
    print(f"{i:<8} {min_val:<15.4f} {max_val:<15.4f} {mean_val:<15.4f} {guess}")

print("\nSample values (first 3 nodes, all features):")
print(data.x[:3, :])

print("\n" + "-" * 70)
print("EDGE ATTRIBUTES:")
print("-" * 70)
print(f"Shape: {data.edge_attr.shape}")

print("\nEdge feature statistics:")
print(f"{'Index':<8} {'Min':<15} {'Max':<15} {'Mean':<15}")
print("-" * 50)

for i in range(data.edge_attr.shape[1]):
    col = data.edge_attr[:, i]
    min_val = col.min().item()
    max_val = col.max().item()
    mean_val = col.mean().item()
    print(f"{i:<8} {min_val:<15.4f} {max_val:<15.4f} {mean_val:<15.4f}")

print("\n" + "-" * 70)
print("OUTPUT TARGETS (y):")
print("-" * 70)
print(f"Shape: {data.y.shape}")
print(f"Number of output features: {data.y.shape[1]}")

print("\nOutput ranges (first 26 - the ones we predict):")
print(f"{'Columns':<15} {'Name':<15} {'Min':<15} {'Max':<15}")
print("-" * 60)

output_names = [
    ("0:2", "Displacement"),
    ("2:8", "Moment Y"),
    ("8:14", "Moment Z"),
    ("14:20", "Shear Y"),
    ("20:26", "Shear Z"),
    ("26:32", "Axial Force"),
    ("32:38", "Torsion"),
]

for cols, name in output_names:
    start, end = map(int, cols.split(":"))
    if end <= data.y.shape[1]:
        min_val = data.y[:, start:end].min().item()
        max_val = data.y[:, start:end].max().item()
        print(f"{cols:<15} {name:<15} {min_val:<15.4f} {max_val:<15.4f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
  Node features (x): {data.x.shape[1]} dimensions
  Edge attributes:   {data.edge_attr.shape[1]} dimensions  
  Output targets:    {data.y.shape[1]} dimensions
  
  This data format differs from what the normalization expects.
  I will provide updated code to match your data format.
""")
print("=" * 70)