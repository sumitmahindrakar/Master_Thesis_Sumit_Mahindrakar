"""
column_layout_check.py — Run this to determine actual column mapping
"""
import torch
import os, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

# Make FrameData available for unpickling
from step_2_grapg_constr import FrameData
import __main__
__main__.FrameData = FrameData

print("=" * 70)
print("  COLUMN LAYOUT VERIFICATION")
print("=" * 70)

raw_data = torch.load("DATA/graph_dataset.pt", weights_only=False)
g = raw_data[0]

# ── Node features ──
print("\n  ══ NODE FEATURES ══")
print(f"  Shape: {list(g.x.shape)}")
print(f"\n  First 5 nodes (all columns):")
print(f"  {'idx':>4} | {'c0':>10} {'c1':>10} {'c2':>10} {'c3':>10} "
      f"{'c4':>10} {'c5':>10} {'c6':>10} {'c7':>10} {'c8':>10} {'c9':>10}")
print(f"  {'-'*115}")
for i in range(min(10, g.x.shape[0])):
    row = g.x[i]
    print(f"  {i:4d} |", end="")
    for v in row:
        print(f" {v.item():10.4e}", end="")
    print()

# ── Identify columns by value patterns ──
print("\n  ══ COLUMN IDENTIFICATION ══")
for c in range(g.x.shape[1]):
    vals = g.x[:, c]
    unique = vals.unique()
    n_unique = len(unique)
    vmin, vmax = vals.min().item(), vals.max().item()

    # Heuristic identification
    if n_unique <= 2 and set(unique.tolist()).issubset({0.0, 1.0}):
        guess = "bc_flag or binary indicator"
    elif vmin >= 0 and vmax <= 1 and n_unique > 5:
        guess = "normalised coordinate or feature"
    elif abs(vmax) > 1e8:
        guess = f"likely E (Young's modulus) = {vmax:.1e}"
    elif 0.05 < vmax < 1.0 and n_unique <= 3:
        guess = f"likely A (cross-area) = {vmax}"
    elif vmax < 1e-3 and vmax > 0:
        guess = f"likely I (inertia) = {vmax:.1e}"
    elif vmin < -1 and vmax > 1:
        guess = f"likely loads range [{vmin:.1f}, {vmax:.1f}]"
    elif vmax > 1 and n_unique > 5:
        guess = f"coordinate or count [0, {vmax:.1f}]"
    else:
        guess = "unknown"

    print(f"    col {c:2d}: [{vmin:12.4e}, {vmax:12.4e}]  "
          f"n_unique={n_unique:4d}  → {guess}")

# ── Edge features ──
print("\n  ══ EDGE FEATURES ══")
print(f"  Shape: {list(g.edge_attr.shape)}")

# Show first few edges
src = g.edge_index[0]
dst = g.edge_index[1]
mask = src < dst  # one direction only
src_m = src[mask]
dst_m = dst[mask]

print(f"\n  First 5 elements (undirected):")
print(f"  {'elem':>4} {'i→j':>6} | {'c0':>10} {'c1':>10} {'c2':>10} "
      f"{'c3':>10} {'c4':>10} {'c5':>10} {'c6':>10}")
print(f"  {'-'*90}")
for idx in range(min(10, mask.sum().item())):
    # Find the idx-th True in mask
    all_mask_indices = mask.nonzero(as_tuple=True)[0]
    eidx = all_mask_indices[idx].item()
    i_n, j_n = src[eidx].item(), dst[eidx].item()
    row = g.edge_attr[eidx]
    print(f"  {idx:4d} {i_n:2d}→{j_n:2d} |", end="")
    for v in row:
        print(f" {v.item():10.4e}", end="")
    print()

print("\n  ══ EDGE COLUMN IDENTIFICATION ══")
for c in range(g.edge_attr.shape[1]):
    vals = g.edge_attr[:, c]  # all directed edges
    vmin, vmax = vals.min().item(), vals.max().item()
    vmean = vals.mean().item()

    if abs(vmax) > 1e8:
        guess = f"E (Young's modulus) = {vmax:.1e}"
    elif vmin >= -1.01 and vmax <= 1.01 and abs(vmin + vmax) < 0.1:
        guess = "direction cosine (cos_α or dx/L)"
    elif vmin >= 0 and vmax <= 1.0 and abs(vmax - 0.1) < 0.01:
        guess = f"A (cross-area) = {vmax:.4f}"
    elif vmax < 1e-3 and vmax > 0:
        guess = f"I (inertia) = {vmax:.1e}"
    elif vmin >= 0 and vmax > 0 and vmax < 10:
        guess = f"length or L/L_ref = {vmax:.4f}"
    else:
        guess = "unknown"

    print(f"    col {c:2d}: [{vmin:12.4e}, {vmax:12.4e}]  "
          f"mean={vmean:12.4e}  → {guess}")

# ── Cross-check with node coordinates ──
print("\n  ══ NODE COORDINATE CROSS-CHECK ══")
# Try to find which columns are coordinates by checking
# if edge dx matches node position differences
for node_col in range(min(4, g.x.shape[1])):
    coords = g.x[:, node_col]
    # Check first edge
    i0, j0 = src_m[0].item(), dst_m[0].item()
    dc = (coords[j0] - coords[i0]).item()
    print(f"    Node col {node_col}: node[{j0}]-node[{i0}] = {dc:.4f}")

# ── Compare with y_node (FEM displacements) ──
print("\n  ══ FEM SOLUTION CHECK ══")
if hasattr(g, 'y_node'):
    y = g.y_node
    print(f"  y_node shape: {list(y.shape)}")
    print(f"  ux:  [{y[:,0].min():.6e}, {y[:,0].max():.6e}]")
    print(f"  uz:  [{y[:,1].min():.6e}, {y[:,1].max():.6e}]")
    print(f"  θy:  [{y[:,2].min():.6e}, {y[:,2].max():.6e}]")

    # Which nodes have largest displacement?
    max_ux_node = y[:, 0].abs().argmax().item()
    max_uz_node = y[:, 1].abs().argmax().item()
    print(f"\n  Node with max |ux|: {max_ux_node}  "
          f"(ux={y[max_ux_node, 0]:.6e})")
    print(f"    node features: {g.x[max_ux_node].tolist()}")
    print(f"  Node with max |uz|: {max_uz_node}  "
          f"(uz={y[max_uz_node, 1]:.6e})")
    print(f"    node features: {g.x[max_uz_node].tolist()}")

# ── Check bc_mask ──
print("\n  ══ BC CHECK ══")
if hasattr(g, 'bc_mask'):
    print(f"  bc_mask exists: {g.bc_mask.sum().item()} fixed nodes")
else:
    print("  ⚠ No bc_mask attribute!")
    # Try to identify fixed nodes from displacements
    zero_disp = (g.y_node.abs().sum(dim=1) < 1e-12)
    print(f"  Nodes with zero displacement: "
          f"{zero_disp.nonzero(as_tuple=True)[0].tolist()}")
    print(f"  Their features:")
    for idx in zero_disp.nonzero(as_tuple=True)[0][:4]:
        print(f"    node {idx.item()}: {g.x[idx].tolist()}")

# ── Check normalised version too ──
print("\n  ══ NORMALISED DATA COMPARISON ══")
norm_data = torch.load("DATA/graph_dataset_norm.pt", weights_only=False)
gn = norm_data[0]

print("  Node features comparison (node 0):")
print(f"    Raw:  {g.x[0].tolist()}")
print(f"    Norm: {gn.x[0].tolist()}")

print("\n  Edge features comparison (edge 0):")
print(f"    Raw:  {g.edge_attr[0].tolist()}")
print(f"    Norm: {gn.edge_attr[0].tolist()}")