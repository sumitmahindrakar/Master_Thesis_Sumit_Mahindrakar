# """
# debug_corotational.py — Find the sign convention error
# """
# import torch
# import os
# from pathlib import Path

# CURRENT_SUBFOLDER = Path(__file__).resolve().parent
# os.chdir(CURRENT_SUBFOLDER)

# data_list = torch.load("DATA/graph_dataset.pt",
#                        weights_only=False)
# data = data_list[0]
# true_disp = data.y_node.clone()

# conn   = data.connectivity
# coords = data.coords
# prop_E = data.prop_E
# prop_A = data.prop_A
# prop_I = data.prop_I22

# print(f"Nodes: {data.num_nodes}, Elements: {data.n_elements}")
# print(f"\n{'═'*80}")
# print(f"  ELEMENT-BY-ELEMENT DEBUG")
# print(f"{'═'*80}")

# # Check a few elements
# for e_idx in range(min(10, conn.shape[0])):
#     nA = conn[e_idx, 0].item()
#     nB = conn[e_idx, 1].item()

#     # Original geometry
#     xA, zA = coords[nA, 0].item(), coords[nA, 2].item()
#     xB, zB = coords[nB, 0].item(), coords[nB, 2].item()
#     dx0 = xB - xA
#     dz0 = zB - zA
#     l0 = (dx0**2 + dz0**2)**0.5
#     cos0 = dx0 / l0
#     sin0 = dz0 / l0

#     # Classify
#     is_horiz = abs(dx0) > abs(dz0)
#     etype = "BEAM" if is_horiz else "COLUMN"

#     # Displacements
#     uxA = true_disp[nA, 0].item()
#     uzA = true_disp[nA, 1].item()
#     thA = true_disp[nA, 2].item()
#     uxB = true_disp[nB, 0].item()
#     uzB = true_disp[nB, 1].item()
#     thB = true_disp[nB, 2].item()

#     # Properties
#     E_val = prop_E[e_idx].item()
#     A_val = prop_A[e_idx].item()
#     I_val = prop_I[e_idx].item()
#     EA = E_val * A_val
#     EI = E_val * I_val

#     print(f"\n  Element {e_idx} ({etype}): "
#           f"node {nA}→{nB}")
#     print(f"    Coords: ({xA:.2f},{zA:.2f}) → "
#           f"({xB:.2f},{zB:.2f})")
#     print(f"    l0={l0:.4f}, cos0={cos0:.4f}, "
#           f"sin0={sin0:.4f}")
#     print(f"    EA={EA:.4e}, EI={EI:.4e}")
#     print(f"    Disp A: ux={uxA:.6e}, uz={uzA:.6e}, "
#           f"θ={thA:.6e}")
#     print(f"    Disp B: ux={uxB:.6e}, uz={uzB:.6e}, "
#           f"θ={thB:.6e}")

#     # ── Corotational computation ──
#     import math
#     beta0 = math.atan2(dz0, dx0)

#     dx = dx0 + (uxB - uxA)
#     dz = dz0 + (uzB - uzA)
#     l = (dx**2 + dz**2)**0.5
#     beta = math.atan2(dz, dx)

#     u_l = (l**2 - l0**2) / (l + l0)
#     alpha = beta - beta0
#     theta1_l = thA - alpha
#     theta2_l = thB - alpha

#     N_e = (EA / l0) * u_l
#     coeff = 2.0 * EI / l0
#     M1 = coeff * (2 * theta1_l + theta2_l)
#     M2 = coeff * (theta1_l + 2 * theta2_l)
#     V_e = (M1 + M2) / l0

#     print(f"    beta0={beta0:.6f}, beta={beta:.6f}, "
#           f"alpha={alpha:.6e}")
#     print(f"    u_l={u_l:.6e}")
#     print(f"    θ1_l={theta1_l:.6e}, θ2_l={theta2_l:.6e}")
#     print(f"    N={N_e:.4f}, M1={M1:.4f}, "
#           f"M2={M2:.4f}, V={V_e:.4f}")

#     # ── Kratos reference ──
#     if data.y_element is not None:
#         k_N = data.y_element[e_idx, 0].item()
#         k_M = data.y_element[e_idx, 1].item()
#         k_V = data.y_element[e_idx, 2].item()
#         print(f"    Kratos: N={k_N:.4f}, M={k_M:.4f}, "
#               f"V={k_V:.4f}")
#         print(f"    Error:  N={abs(N_e-k_N):.4f}, "
#               f"M={abs(M1-k_M):.4f}, "
#               f"V={abs(V_e-k_V):.4f}")

# # ── Check displacement convention ──
# print(f"\n{'═'*80}")
# print(f"  DISPLACEMENT CONVENTION CHECK")
# print(f"{'═'*80}")

# print(f"\n  y_node columns: [0]=?, [1]=?, [2]=?")
# print(f"  y_node[:, 0] range: "
#       f"[{true_disp[:, 0].min():.6e}, "
#       f"{true_disp[:, 0].max():.6e}]")
# print(f"  y_node[:, 1] range: "
#       f"[{true_disp[:, 1].min():.6e}, "
#       f"{true_disp[:, 1].max():.6e}]")
# print(f"  y_node[:, 2] range: "
#       f"[{true_disp[:, 2].min():.6e}, "
#       f"{true_disp[:, 2].max():.6e}]")

# # Check raw displacement field
# if 'displacement' in dir(data) or hasattr(data, 'displacement'):
#     print(f"\n  Hmm, no raw 3D displacement stored in graph")

# # Check: is y_node[:, 0] really u_x or is it u_z?
# # For a frame with vertical load, u_z should be largest
# print(f"\n  Which column has largest magnitude?")
# for col in range(3):
#     mag = true_disp[:, col].abs().max().item()
#     print(f"    col {col}: max|val| = {mag:.6e}")

# print(f"\n  If col 0 has largest magnitude AND you expect")
# print(f"  vertical deflection to dominate, then columns")
# print(f"  might be swapped!")

# # Check coords
# print(f"\n  Coordinate ranges:")
# print(f"    x: [{coords[:, 0].min():.2f}, "
#       f"{coords[:, 0].max():.2f}]")
# print(f"    y: [{coords[:, 1].min():.2f}, "
#       f"{coords[:, 1].max():.2f}]")
# print(f"    z: [{coords[:, 2].min():.2f}, "
#       f"{coords[:, 2].max():.2f}]")

# # Check F_ext
# print(f"\n  F_ext:")
# print(f"    col 0 (Fx): [{data.F_ext[:, 0].min():.4f}, "
#       f"{data.F_ext[:, 0].max():.4f}]")
# print(f"    col 1 (Fz): [{data.F_ext[:, 1].min():.4f}, "
#       f"{data.F_ext[:, 1].max():.4f}]")
# print(f"    col 2 (My): [{data.F_ext[:, 2].min():.4f}, "
#       f"{data.F_ext[:, 2].max():.4f}]")

# # ── Check support nodes ──
# print(f"\n  Support nodes:")
# sup = torch.where(data.bc_disp.squeeze() > 0.5)[0]
# for s in sup:
#     i = s.item()
#     print(f"    Node {i}: coords=({coords[i, 0]:.2f}, "
#           f"{coords[i, 2]:.2f}), "
#           f"disp=({true_disp[i, 0]:.6e}, "
#           f"{true_disp[i, 1]:.6e}, "
#           f"{true_disp[i, 2]:.6e})")

"""
debug_convention.py — Find the exact DOF convention
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

print(f"{'═'*70}")
print(f"  DOF CONVENTION DIAGNOSTIC")
print(f"{'═'*70}")

# ── What does step 1 data loader store? ──
# y_node is built from:
#   nodal_disp_2d = [displacement[:,0], displacement[:,2], rotation[:,1]]
#   i.e. [ux_kratos, uz_kratos, rot_y_kratos]

# BUT: what is Kratos DISPLACEMENT and ROTATION?
# Kratos 3D: DISPLACEMENT = [Dx, Dy, Dz]
# Kratos 3D: ROTATION = [Rx, Ry, Rz]
# 
# For 2D frame in XZ plane:
#   Dx = horizontal displacement (x-direction)
#   Dz = vertical displacement (z-direction)
#   Ry = rotation about y-axis

print(f"\n  y_node columns from data loader:")
print(f"    col 0 = displacement[:,0] = Dx (x-disp)")
print(f"    col 1 = displacement[:,2] = Dz (z-disp)")  
print(f"    col 2 = rotation[:,1]     = Ry (y-rot)")

# ── Look at a column element ──
# Element 0: vertical column at x=0
e = 0
nA = conn[e, 0].item()
nB = conn[e, 1].item()

print(f"\n  Element {e}: node {nA} → {nB}")
print(f"    A: ({coords[nA,0]:.2f}, {coords[nA,2]:.2f})")
print(f"    B: ({coords[nB,0]:.2f}, {coords[nB,2]:.2f})")

# Column goes UP: from z=0 to z=0.6
# Under horizontal loading, top sways RIGHT
# The column bends → rotation at base

print(f"\n  Displacements:")
print(f"    Node {nA}: Dx={true_disp[nA,0]:.6e}, "
      f"Dz={true_disp[nA,1]:.6e}, "
      f"Ry={true_disp[nA,2]:.6e}")
print(f"    Node {nB}: Dx={true_disp[nB,0]:.6e}, "
      f"Dz={true_disp[nB,1]:.6e}, "
      f"Ry={true_disp[nB,2]:.6e}")

# ── Physical reasoning ──
# Node 0 (base): fixed → Dx=0, Dz=0, Ry=0.012
# Wait... Ry is NOT zero at the fixed base?!

print(f"\n  ⚠ CHECK: Node {nA} is a SUPPORT node")
print(f"    bc_disp = {data.bc_disp[nA].item()}")
print(f"    bc_rot  = {data.bc_rot[nA].item()}")
print(f"    Dx = {true_disp[nA,0]:.6e}")
print(f"    Dz = {true_disp[nA,1]:.6e}")
print(f"    Ry = {true_disp[nA,2]:.6e}")

if abs(true_disp[nA, 2].item()) > 1e-6:
    print(f"\n  ❗ ROTATION IS NON-ZERO AT SUPPORT!")
    print(f"     This means the support is PINNED "
          f"(not fixed)!")
    print(f"     bc_rot should be 0 (free rotation)")

# ── Check ALL support nodes ──
print(f"\n  All support nodes:")
sup = torch.where(data.bc_disp.squeeze() > 0.5)[0]
for s in sup:
    i = s.item()
    print(f"    Node {i}: "
          f"coords=({coords[i,0]:.2f}, {coords[i,2]:.2f}), "
          f"Dx={true_disp[i,0]:.6e}, "
          f"Dz={true_disp[i,1]:.6e}, "
          f"Ry={true_disp[i,2]:.6e}, "
          f"bc_rot={data.bc_rot[i].item():.0f}")

# ── Try SIMPLE stiffness test ──
# For element 0 (vertical column, cos=0, sin=1):
# Local frame: axial=z, transverse=x
# T·d_global:
#   u_axial_A  = cos*Dx_A + sin*Dz_A = 0*0 + 1*0 = 0
#   w_trans_A  = -sin*Dx_A + cos*Dz_A = -1*0 + 0*0 = 0
#   θ_A        = Ry_A = 0.01216
#   u_axial_B  = cos*Dx_B + sin*Dz_B = 0*0.00724 + 1*0 = 0
#   w_trans_B  = -sin*Dx_B + cos*Dz_B = -1*0.00724 + 0 = -0.00724
#   θ_B        = Ry_B = 0.01186

# So in local frame:
#   w_A = 0, w_B = -0.00724
#   θ_A = 0.01216, θ_B = 0.01186
#
# Standard stiffness f = k·d:
#   f_shear_A = 12EI/L³*(w_A-w_B) + 6EI/L²*(θ_A+θ_B)
#             = 12*1e5/0.216*(0.00724) + 6*1e5/0.36*(0.02402)
#             = 40222 + 40033 = 80255
#
# This is HUGE because w_B = -0.00724 is interpreted as 
# a large transverse displacement!
#
# BUT: in structural analysis, the column sway is a 
# RIGID BODY MOTION, not a deformation!
# The shear force should come from the RELATIVE rotation
# between the two ends, not from the absolute sway.

print(f"\n{'═'*70}")
print(f"  KEY INSIGHT")
print(f"{'═'*70}")
print(f"""
  The 6-DOF stiffness matrix k·d gives TOTAL nodal forces
  including rigid body contributions. For a cantilevered
  column that sways sideways, the transverse displacement
  w_B = -Dx (for vertical element) creates enormous
  fictitious shear and moment — but these cancel when
  assembled at joints where beams connect.

  The EQUILIBRIUM should still be correct after assembly!
  
  The fact that Max|R| = 47,622 means either:
  1. The assembly is wrong
  2. The F_ext doesn't match the Kratos loading
  3. There's a DOF convention mismatch
  
  Let's check assembly at a specific interior node.
""")

# ── Check assembly at an interior node ──
# Find a node connected to multiple elements
from collections import Counter
node_elem_count = Counter()
for e in range(conn.shape[0]):
    node_elem_count[conn[e, 0].item()] += 1
    node_elem_count[conn[e, 1].item()] += 1

# Find a non-support node with multiple elements
interior = [n for n in node_elem_count
            if node_elem_count[n] >= 2
            and data.bc_disp[n].item() < 0.5]

if interior:
    test_node = interior[0]
    print(f"\n  Assembly check at node {test_node}:")
    print(f"    coords: ({coords[test_node,0]:.2f}, "
          f"{coords[test_node,2]:.2f})")
    print(f"    F_ext: {data.F_ext[test_node].tolist()}")
    print(f"    Connected elements:")

    from corotational import CorotationalBeam2D
    beam = CorotationalBeam2D()
    result = beam(true_disp, data)

    total_force = torch.zeros(3)
    for e in range(conn.shape[0]):
        if conn[e, 0].item() == test_node:
            f = result['F_global_A'][e]
            print(f"      Elem {e} (as A): "
                  f"Fx={f[0]:.4f}, "
                  f"Fz={f[1]:.4f}, "
                  f"My={f[2]:.4f}")
            total_force += f
        elif conn[e, 1].item() == test_node:
            f = result['F_global_B'][e]
            print(f"      Elem {e} (as B): "
                  f"Fx={f[0]:.4f}, "
                  f"Fz={f[1]:.4f}, "
                  f"My={f[2]:.4f}")
            total_force += f

    print(f"    Total assembled: "
          f"Fx={total_force[0]:.4f}, "
          f"Fz={total_force[1]:.4f}, "
          f"My={total_force[2]:.4f}")
    print(f"    F_ext:           "
          f"Fx={data.F_ext[test_node, 0]:.4f}, "
          f"Fz={data.F_ext[test_node, 1]:.4f}, "
          f"My={data.F_ext[test_node, 2]:.4f}")
    residual = total_force - data.F_ext[test_node]
    print(f"    Residual:        "
          f"Rx={residual[0]:.4f}, "
          f"Rz={residual[1]:.4f}, "
          f"RM={residual[2]:.4f}")

print(f"\n{'═'*70}")