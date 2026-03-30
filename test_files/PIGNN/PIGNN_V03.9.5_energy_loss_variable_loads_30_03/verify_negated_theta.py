"""
verify_negated_theta.py — Full verification with -θ
"""
import torch
import os
from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

data_list = torch.load("DATA/graph_dataset.pt",
                       weights_only=False)
data = data_list[0]

coords = data.coords
conn = data.connectivity
true_disp = data.y_node.clone()

# ── NEGATE θ ──
true_disp_fixed = true_disp.clone()
true_disp_fixed[:, 2] = -true_disp[:, 2]   # negate θ_y

print(f"{'═'*70}")
print(f"  VERIFICATION WITH NEGATED θ")
print(f"{'═'*70}")

print(f"\n  Original θ range: [{true_disp[:, 2].min():.6e}, "
      f"{true_disp[:, 2].max():.6e}]")
print(f"  Negated θ range:  [{true_disp_fixed[:, 2].min():.6e}, "
      f"{true_disp_fixed[:, 2].max():.6e}]")

from corotational import CorotationalBeam2D
beam = CorotationalBeam2D()
result = beam(true_disp_fixed, data)

# ── Equilibrium check ──
residual = result['nodal_forces'] - data.F_ext
free = data.bc_disp.squeeze(-1) < 0.5

print(f"\n  Equilibrium at FREE nodes:")
res_free = residual[free]
print(f"    Rx:  [{res_free[:, 0].min():.6e}, "
      f"{res_free[:, 0].max():.6e}]")
print(f"    Rz:  [{res_free[:, 1].min():.6e}, "
      f"{res_free[:, 1].max():.6e}]")
print(f"    RM:  [{res_free[:, 2].min():.6e}, "
      f"{res_free[:, 2].max():.6e}]")
print(f"    Max |R|: {res_free.abs().max():.6e}")

F_max = data.F_ext.abs().max().item()
print(f"    |R|/F_max: {res_free.abs().max().item()/F_max:.6e}")

# ── Internal forces comparison ──
print(f"\n  Internal forces:")
print(f"    N:   [{result['N_e'].min():.4f}, "
      f"{result['N_e'].max():.4f}]")
print(f"    M1:  [{result['M1_e'].min():.4f}, "
      f"{result['M1_e'].max():.4f}]")
print(f"    V:   [{result['V_e'].min():.4f}, "
      f"{result['V_e'].max():.4f}]")

print(f"\n  Kratos reference:")
print(f"    N:   [{data.y_element[:, 0].min():.4f}, "
      f"{data.y_element[:, 0].max():.4f}]")
print(f"    M:   [{data.y_element[:, 1].min():.4f}, "
      f"{data.y_element[:, 1].max():.4f}]")
print(f"    V:   [{data.y_element[:, 2].min():.4f}, "
      f"{data.y_element[:, 2].max():.4f}]")

# ── Per-element comparison ──
print(f"\n  Per-element (first 15 + beams):")
print(f"  {'El':>3} {'T':>1} | {'N_comp':>10} {'N_krat':>10} | "
      f"{'M1_comp':>10} {'M_krat':>10} | "
      f"{'V_comp':>10} {'V_krat':>10}")
print(f"  {'-'*80}")

# Show first 10 columns + first 5 beams
shown = 0
for e in range(conn.shape[0]):
    nA = conn[e, 0].item()
    nB = conn[e, 1].item()
    dx = abs(coords[nB, 0] - coords[nA, 0])
    dz = abs(coords[nB, 2] - coords[nA, 2])
    etype = 'B' if dx > dz else 'C'

    if (etype == 'C' and shown < 10) or \
       (etype == 'B' and shown < 15):
        print(
            f"  {e:3d} {etype} | "
            f"{result['N_e'][e].item():10.4f} "
            f"{data.y_element[e, 0].item():10.4f} | "
            f"{result['M1_e'][e].item():10.4f} "
            f"{data.y_element[e, 1].item():10.4f} | "
            f"{result['V_e'][e].item():10.4f} "
            f"{data.y_element[e, 2].item():10.4f}"
        )
        shown += 1

# ── Check beam elements specifically ──
print(f"\n  ALL beam elements:")
print(f"  {'El':>3} | {'N_comp':>10} {'N_krat':>10} | "
      f"{'M1_comp':>10} {'M_krat':>10} | "
      f"{'V_comp':>10} {'V_krat':>10}")
print(f"  {'-'*75}")
for e in range(conn.shape[0]):
    nA = conn[e, 0].item()
    nB = conn[e, 1].item()
    dx = abs(coords[nB, 0] - coords[nA, 0])
    dz = abs(coords[nB, 2] - coords[nA, 2])
    if dx > dz:  # beam
        print(
            f"  {e:3d} | "
            f"{result['N_e'][e].item():10.4f} "
            f"{data.y_element[e, 0].item():10.4f} | "
            f"{result['M1_e'][e].item():10.4f} "
            f"{data.y_element[e, 1].item():10.4f} | "
            f"{result['V_e'][e].item():10.4f} "
            f"{data.y_element[e, 2].item():10.4f}"
        )

print(f"\n{'═'*70}")