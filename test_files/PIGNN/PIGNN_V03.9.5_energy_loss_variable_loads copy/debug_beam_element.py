"""
debug_beam_element.py — Check beam element 60 in detail
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
true_disp = data.y_node

# Element 60: beam from node 5 to node 62
e = 60
nA = conn[e, 0].item()
nB = conn[e, 1].item()

print(f"{'═'*70}")
print(f"  BEAM ELEMENT {e} DETAILED DEBUG")
print(f"{'═'*70}")

print(f"\n  Node {nA}: ({coords[nA,0]:.2f}, "
      f"{coords[nA,2]:.2f})")
print(f"  Node {nB}: ({coords[nB,0]:.2f}, "
      f"{coords[nB,2]:.2f})")

dx0 = coords[nB, 0] - coords[nA, 0]
dz0 = coords[nB, 2] - coords[nA, 2]
l0 = (dx0**2 + dz0**2).sqrt()
c = dx0 / l0
s = dz0 / l0

print(f"\n  dx0={dx0:.4f}, dz0={dz0:.4f}")
print(f"  l0={l0:.4f}")
print(f"  cos={c:.6f}, sin={s:.6f}")

EA = data.prop_E[e] * data.prop_A[e]
EI = data.prop_E[e] * data.prop_I22[e]
print(f"  EA={EA:.4e}, EI={EI:.4e}")

# Displacements
print(f"\n  y_node (as stored):")
print(f"    Node {nA}: [{true_disp[nA,0]:.6e}, "
      f"{true_disp[nA,1]:.6e}, {true_disp[nA,2]:.6e}]")
print(f"    Node {nB}: [{true_disp[nB,0]:.6e}, "
      f"{true_disp[nB,1]:.6e}, {true_disp[nB,2]:.6e}]")

# Transform to local
ux_A = true_disp[nA, 0].item()
uz_A = true_disp[nA, 1].item()
th_A = true_disp[nA, 2].item()
ux_B = true_disp[nB, 0].item()
uz_B = true_disp[nB, 1].item()
th_B = true_disp[nB, 2].item()

c_val = c.item()
s_val = s.item()

ua_loc = c_val * ux_A + s_val * uz_A
wa_loc = -s_val * ux_A + c_val * uz_A
ta_loc = th_A

ub_loc = c_val * ux_B + s_val * uz_B
wb_loc = -s_val * ux_B + c_val * uz_B
tb_loc = th_B

print(f"\n  Local displacements:")
print(f"    ua={ua_loc:.6e}, wa={wa_loc:.6e}, θa={ta_loc:.6e}")
print(f"    ub={ub_loc:.6e}, wb={wb_loc:.6e}, θb={tb_loc:.6e}")

# Compute local forces manually
L = l0.item()
L2 = L * L
L3 = L2 * L
EA_val = EA.item()
EI_val = EI.item()

f0 = EA_val / L * (ua_loc - ub_loc)
f1 = 12 * EI_val / L3 * (wa_loc - wb_loc) + \
     6 * EI_val / L2 * (ta_loc + tb_loc)
f2 = 6 * EI_val / L2 * (wa_loc - wb_loc) + \
     EI_val / L * (4 * ta_loc + 2 * tb_loc)
f3 = EA_val / L * (ub_loc - ua_loc)
f4 = 12 * EI_val / L3 * (wb_loc - wa_loc) - \
     6 * EI_val / L2 * (ta_loc + tb_loc)
f5 = 6 * EI_val / L2 * (wa_loc - wb_loc) + \
     EI_val / L * (2 * ta_loc + 4 * tb_loc)

print(f"\n  Local forces:")
print(f"    f0 (N_A):  {f0:.4f}")
print(f"    f1 (V_A):  {f1:.4f}")
print(f"    f2 (M_A):  {f2:.4f}")
print(f"    f3 (N_B):  {f3:.4f}")
print(f"    f4 (V_B):  {f4:.4f}")
print(f"    f5 (M_B):  {f5:.4f}")

# Global forces
fx_A = c_val * f0 - s_val * f1
fz_A = s_val * f0 + c_val * f1
my_A = f2

fx_B = c_val * f3 - s_val * f4
fz_B = s_val * f3 + c_val * f4
my_B = f5

print(f"\n  Global forces:")
print(f"    Node A: Fx={fx_A:.4f}, Fz={fz_A:.4f}, "
      f"My={my_A:.4f}")
print(f"    Node B: Fx={fx_B:.4f}, Fz={fz_B:.4f}, "
      f"My={my_B:.4f}")

# ── Now try with SWAPPED DOFs ──
# What if y_node is actually [uz, ux, θy]?
print(f"\n{'═'*70}")
print(f"  TRY SWAPPED: y_node = [uz, ux, θy]")
print(f"{'═'*70}")

ux_A_swap = true_disp[nA, 1].item()  # was uz
uz_A_swap = true_disp[nA, 0].item()  # was ux
th_A_swap = th_A

ux_B_swap = true_disp[nB, 1].item()
uz_B_swap = true_disp[nB, 0].item()
th_B_swap = th_B

ua_s = c_val * ux_A_swap + s_val * uz_A_swap
wa_s = -s_val * ux_A_swap + c_val * uz_A_swap
ub_s = c_val * ux_B_swap + s_val * uz_B_swap
wb_s = -s_val * ux_B_swap + c_val * uz_B_swap

print(f"  Local displacements (swapped):")
print(f"    ua={ua_s:.6e}, wa={wa_s:.6e}")
print(f"    ub={ub_s:.6e}, wb={wb_s:.6e}")

f0s = EA_val / L * (ua_s - ub_s)
f1s = 12 * EI_val / L3 * (wa_s - wb_s) + \
      6 * EI_val / L2 * (th_A_swap + th_B_swap)
f2s = 6 * EI_val / L2 * (wa_s - wb_s) + \
      EI_val / L * (4 * th_A_swap + 2 * th_B_swap)

print(f"\n  Local forces (swapped):")
print(f"    f0 (N_A):  {f0s:.4f}")
print(f"    f1 (V_A):  {f1s:.4f}")
print(f"    f2 (M_A):  {f2s:.4f}")

# ── Also try negating θ ──
print(f"\n{'═'*70}")
print(f"  TRY NEGATED θ: y_node = [ux, uz, -θy]")
print(f"{'═'*70}")

th_A_neg = -th_A
th_B_neg = -th_B

f0n = EA_val / L * (ua_loc - ub_loc)
f1n = 12 * EI_val / L3 * (wa_loc - wb_loc) + \
      6 * EI_val / L2 * (th_A_neg + th_B_neg)
f2n = 6 * EI_val / L2 * (wa_loc - wb_loc) + \
      EI_val / L * (4 * th_A_neg + 2 * th_B_neg)

print(f"  Local forces (negated θ):")
print(f"    f0 (N_A):  {f0n:.4f}")
print(f"    f1 (V_A):  {f1n:.4f}")
print(f"    f2 (M_A):  {f2n:.4f}")

# ── What Kratos says ──
print(f"\n{'═'*70}")
print(f"  KRATOS REFERENCE")
print(f"{'═'*70}")
print(f"  Element {e}:")
print(f"    N:  {data.y_element[e, 0]:.4f}")
print(f"    M:  {data.y_element[e, 1]:.4f}")
print(f"    V:  {data.y_element[e, 2]:.4f}")

# ── Check what Kratos N means ──
# If Kratos N = 218.86 and our f0 (swapped) matches,
# then DOFs are swapped
print(f"\n  Comparison:")
print(f"    Original:  N={f0:.4f}")
print(f"    Swapped:   N={f0s:.4f}")
print(f"    Kratos:    N={data.y_element[e, 0]:.4f}")