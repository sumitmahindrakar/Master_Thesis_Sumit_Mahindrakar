"""
debug_junction.py — Check equilibrium at beam-column junctions
"""
import torch
import os
from pathlib import Path
from collections import Counter

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

data_list = torch.load("DATA/graph_dataset.pt",
                       weights_only=False)
data = data_list[0]

coords = data.coords
conn = data.connectivity
true_disp = data.y_node
E_count = conn.shape[0]
N_count = data.num_nodes

print(f"{'═'*70}")
print(f"  JUNCTION NODE EQUILIBRIUM CHECK")
print(f"{'═'*70}")

# ── Classify elements ──
elem_type = []
for e in range(E_count):
    nA = conn[e, 0].item()
    nB = conn[e, 1].item()
    dx = abs(coords[nB, 0] - coords[nA, 0])
    dz = abs(coords[nB, 2] - coords[nA, 2])
    elem_type.append('B' if dx > dz else 'C')

print(f"\n  Element types:")
n_beam = sum(1 for t in elem_type if t == 'B')
n_col = sum(1 for t in elem_type if t == 'C')
print(f"    Beams: {n_beam}, Columns: {n_col}")

# ── Find junction nodes (connected to both beams and columns) ──
node_has_beam = set()
node_has_col = set()
node_elements = {}

for e in range(E_count):
    nA = conn[e, 0].item()
    nB = conn[e, 1].item()
    
    for n in [nA, nB]:
        if n not in node_elements:
            node_elements[n] = []
        node_elements[n].append(e)
    
    if elem_type[e] == 'B':
        node_has_beam.add(nA)
        node_has_beam.add(nB)
    else:
        node_has_col.add(nA)
        node_has_col.add(nB)

junctions = node_has_beam & node_has_col
sup = set(torch.where(
    data.bc_disp.squeeze() > 0.5
)[0].tolist())
free_junctions = junctions - sup

print(f"  Junction nodes (beam+column): {len(junctions)}")
print(f"  Free junction nodes: {len(free_junctions)}")
print(f"  Junction nodes: {sorted(free_junctions)}")

# ── Compute forces ──
from corotational import CorotationalBeam2D
beam_module = CorotationalBeam2D()
result = beam_module(true_disp, data)

# ── Check equilibrium at each junction ──
print(f"\n  Per-junction equilibrium:")
print(f"  {'Node':>4} {'Coords':>12} | "
      f"{'Rx':>12} {'Rz':>12} {'RM':>12} | "
      f"{'F_ext_x':>10} {'#Elem':>5}")
print(f"  {'-'*85}")

max_res = 0
for node in sorted(free_junctions):
    total_f = torch.zeros(3)
    elems_at_node = node_elements[node]
    
    for e in elems_at_node:
        if conn[e, 0].item() == node:
            total_f += result['F_global_A'][e]
        else:
            total_f += result['F_global_B'][e]
    
    fext = data.F_ext[node]
    res = total_f - fext
    max_res = max(max_res, res.abs().max().item())
    
    coord_str = (f"({coords[node,0]:.1f},"
                 f"{coords[node,2]:.1f})")
    print(f"  {node:4d} {coord_str:>12} | "
          f"{res[0]:12.4f} {res[1]:12.4f} "
          f"{res[2]:12.4f} | "
          f"{fext[0]:10.4f} {len(elems_at_node):5d}")

print(f"\n  Max |R| at junctions: {max_res:.4f}")

# ── Also check mid-column nodes ──
mid_col_nodes = (node_has_col - node_has_beam) - sup
print(f"\n  Mid-column nodes (no beam): "
      f"{len(mid_col_nodes)}")

if mid_col_nodes:
    print(f"\n  Per mid-column equilibrium:")
    print(f"  {'Node':>4} {'Coords':>12} | "
          f"{'Rx':>12} {'Rz':>12} {'RM':>12} | "
          f"{'F_ext_x':>10}")
    print(f"  {'-'*75}")
    
    max_res_mid = 0
    for node in sorted(mid_col_nodes)[:10]:
        total_f = torch.zeros(3)
        for e in node_elements[node]:
            if conn[e, 0].item() == node:
                total_f += result['F_global_A'][e]
            else:
                total_f += result['F_global_B'][e]
        
        fext = data.F_ext[node]
        res = total_f - fext
        max_res_mid = max(max_res_mid,
                         res.abs().max().item())
        
        coord_str = (f"({coords[node,0]:.1f},"
                     f"{coords[node,2]:.1f})")
        print(f"  {node:4d} {coord_str:>12} | "
              f"{res[0]:12.4f} {res[1]:12.4f} "
              f"{res[2]:12.4f} | "
              f"{fext[0]:10.4f}")
    
    print(f"\n  Max |R| at mid-column: "
          f"{max_res_mid:.4f}")

# ── Print element info around a junction ──
if free_junctions:
    test_node = sorted(free_junctions)[0]
    print(f"\n{'═'*70}")
    print(f"  DETAILED: Node {test_node} "
          f"({coords[test_node,0]:.1f}, "
          f"{coords[test_node,2]:.1f})")
    print(f"{'═'*70}")
    
    for e in node_elements[test_node]:
        nA = conn[e, 0].item()
        nB = conn[e, 1].item()
        is_A = (nA == test_node)
        end = "A" if is_A else "B"
        
        f = (result['F_global_A'][e] if is_A 
             else result['F_global_B'][e])
        f_loc = result['f_local'][e]
        
        print(f"\n  Elem {e} ({elem_type[e]}): "
              f"node {nA}→{nB}, this is end {end}")
        print(f"    f_local:  [{', '.join(f'{v:.4f}' for v in f_loc.tolist())}]")
        print(f"    f_global: Fx={f[0]:.4f}, "
              f"Fz={f[1]:.4f}, My={f[2]:.4f}")

print(f"\n{'═'*70}")