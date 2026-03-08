"""
debug_physics.py — Find root cause of FEM verification failure

Key question: WHY does the FEM solution NOT satisfy our equilibrium?
"""

import os
os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\PIGNN\PIGNN_V03.7_k_matrix")

import torch
import numpy as np


def main():
    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    data = data_list[0]

    # ════════════════════════════════════════════════════════
    # DEBUG 1: WHAT PLANE IS THE FRAME IN?
    # ════════════════════════════════════════════════════════
    print("=" * 70)
    print("  DEBUG 1: COORDINATE SYSTEM")
    print("=" * 70)

    coords = data.coords
    for j, name in enumerate(['X', 'Y', 'Z']):
        col = coords[:, j]
        extent = col.max() - col.min()
        print(f"  {name}: [{col.min():.4f}, {col.max():.4f}]  "
              f"extent={extent:.4f}")

    x_ext = (coords[:, 0].max() - coords[:, 0].min()).item()
    y_ext = (coords[:, 1].max() - coords[:, 1].min()).item()
    z_ext = (coords[:, 2].max() - coords[:, 2].min()).item()

    print(f"\n  Verdict:")
    if y_ext < 0.01 * max(x_ext, z_ext):
        print(f"  → Frame in XZ plane (Y≈const)")
        print(f"    Horizontal axis: X")
        print(f"    Vertical axis:   Z")
        vert_idx = 2
    elif z_ext < 0.01 * max(x_ext, y_ext):
        print(f"  → Frame in XY plane (Z≈const)")
        print(f"    Horizontal axis: X")
        print(f"    Vertical axis:   Y")
        vert_idx = 1
    else:
        print(f"  → 3D frame or oblique")
        vert_idx = -1

    # ════════════════════════════════════════════════════════
    # DEBUG 2: WHICH COMPONENT HAS THE LOAD?
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  DEBUG 2: LOAD DIRECTION")
    print(f"{'='*70}")

    print(f"\n  elem_load (E, 3) per component:")
    for j in range(3):
        col = data.elem_load[:, j]
        nz = (col.abs() > 1e-10).sum().item()
        total = col.abs().sum().item()
        print(f"    [{['qx','qy','qz'][j]}]  "
              f"min={col.min():+.4e}  max={col.max():+.4e}  "
              f"nonzero={nz}/{data.elem_load.shape[0]}  "
              f"|total|={total:.4e}")

    print(f"\n  node line_load x[:, 5:8] per component:")
    for j in range(3):
        col = data.x[:, 5 + j]
        nz = (col.abs() > 1e-10).sum().item()
        total = col.abs().sum().item()
        print(f"    [{['wl_x','wl_y','wl_z'][j]}]  "
              f"min={col.min():+.4e}  max={col.max():+.4e}  "
              f"nonzero={nz}/{data.x.shape[0]}  "
              f"|total|={total:.4e}")

    # Identify load component
    load_totals = [data.elem_load[:, j].abs().sum().item()
                   for j in range(3)]
    load_comp = int(np.argmax(load_totals))
    print(f"\n  Dominant load component: {load_comp} "
          f"({['X','Y','Z'][load_comp]})")
    print(f"  Physics loss uses components: 0 (X) and 2 (Z)")
    if load_comp == 1:
        print(f"  ⚠ PROBLEM: Load is in Y but code ignores Y!")

    # First few elem_loads
    print(f"\n  First 5 elem_load values:")
    for e in range(min(5, data.elem_load.shape[0])):
        q = data.elem_load[e]
        loaded = "← LOADED" if q.abs().sum() > 1e-10 else ""
        print(f"    Elem {e}: [{q[0]:+.4e}, {q[1]:+.4e}, "
              f"{q[2]:+.4e}]  {loaded}")

    # ════════════════════════════════════════════════════════
    # DEBUG 3: ELEMENT DIRECTIONS
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  DEBUG 3: ELEMENT DIRECTIONS")
    print(f"{'='*70}")

    dirs = data.elem_directions
    print(f"\n  elem_directions per component:")
    for j in range(3):
        col = dirs[:, j]
        nz = (col.abs() > 1e-10).sum().item()
        print(f"    [{['dx','dy','dz'][j]}]  "
              f"min={col.min():+.4e}  max={col.max():+.4e}  "
              f"nonzero={nz}/{dirs.shape[0]}")

    # Classify elements by direction
    cos_x = dirs[:, 0].abs()
    sin_y = dirs[:, 1].abs()
    sin_z = dirs[:, 2].abs()

    n_horiz = (cos_x > 0.9).sum().item()
    n_vert_y = ((sin_y > 0.9) & (cos_x < 0.1)).sum().item()
    n_vert_z = ((sin_z > 0.9) & (cos_x < 0.1)).sum().item()
    n_other = dirs.shape[0] - n_horiz - n_vert_y - n_vert_z

    print(f"\n  Element classification:")
    print(f"    Horizontal (|dx|>0.9):   {n_horiz}")
    print(f"    Vertical Y (|dy|>0.9):   {n_vert_y}")
    print(f"    Vertical Z (|dz|>0.9):   {n_vert_z}")
    print(f"    Other/diagonal:          {n_other}")

    print(f"\n  Code uses: cos_t = dirs[:, 0],  sin_t = dirs[:, 2]")
    if n_vert_y > 0 and n_vert_z == 0:
        print(f"  ⚠ PROBLEM: Columns are in Y-direction but "
              f"code uses Z for sin_t!")
        print(f"    sin_t would be 0 for ALL columns → "
              f"wrong local coordinates!")

    # First few directions
    print(f"\n  First 10 element directions:")
    for e in range(min(10, dirs.shape[0])):
        d = dirs[e]
        L = data.elem_lengths[e].item()
        elem_type = "BEAM" if abs(d[0]) > 0.9 else (
            "COL_Y" if abs(d[1]) > 0.9 else (
            "COL_Z" if abs(d[2]) > 0.9 else "OTHER"))
        print(f"    Elem {e}: ({d[0]:+.4f}, {d[1]:+.4f}, "
              f"{d[2]:+.4f})  L={L:.4f}  {elem_type}")

    # ════════════════════════════════════════════════════════
    # DEBUG 4: BC FLAGS
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  DEBUG 4: BOUNDARY CONDITIONS")
    print(f"{'='*70}")

    bc_d = data.bc_disp.squeeze()
    bc_r = data.bc_rot.squeeze()
    n_bc_d = (bc_d > 0.5).sum().item()
    n_bc_r = (bc_r > 0.5).sum().item()

    print(f"  bc_disp constrained: {n_bc_d} / {bc_d.shape[0]}")
    print(f"  bc_rot  constrained: {n_bc_r} / {bc_r.shape[0]}")

    if n_bc_r == 0:
        print(f"  ⚠ PROBLEM: bc_rot is ALL ZERO!")
        print(f"    Moment equilibrium checked at supports too!")

    sup = (bc_d > 0.5).nonzero().squeeze().tolist()
    if isinstance(sup, int):
        sup = [sup]
    print(f"\n  Support nodes: {sup}")
    for n in sup:
        c = coords[n]
        print(f"    Node {n}: ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})  "
              f"bc_d={bc_d[n]:.0f}  bc_r={bc_r[n]:.0f}")

    # ════════════════════════════════════════════════════════
    # DEBUG 5: FEM DISPLACEMENT CONVENTION
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  DEBUG 5: FEM DISPLACEMENTS")
    print(f"{'='*70}")

    y = data.y_node
    print(f"  y_node (N, 3) = [u_x, u_z, φ]:")
    for j, name in enumerate(['u_x (col0)', 'u_z (col1)', 'φ (col2)']):
        col = y[:, j]
        nz = (col.abs() > 1e-10).sum().item()
        print(f"    {name}:  min={col.min():+.4e}  "
              f"max={col.max():+.4e}  nonzero={nz}")

    # ════════════════════════════════════════════════════════
    # DEBUG 6: SINGLE ELEMENT STIFFNESS TEST
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  DEBUG 6: SINGLE ELEMENT STIFFNESS TEST")
    print(f"{'='*70}")

    conn = data.connectivity
    loaded_mask = data.elem_load.abs().sum(dim=1) > 1e-10
    horiz_mask = dirs[:, 0].abs() > 0.9

    # Find loaded beam
    candidates = (loaded_mask & horiz_mask).nonzero().squeeze()
    if candidates.dim() == 0:
        candidates = candidates.unsqueeze(0)

    if len(candidates) == 0:
        # Try any loaded element
        candidates = loaded_mask.nonzero().squeeze()
        if candidates.dim() == 0:
            candidates = candidates.unsqueeze(0)

    if len(candidates) > 0:
        e = candidates[len(candidates) // 2].item()  # middle element
        test_single_element(data, e, vert_idx)
    else:
        print("  No loaded elements found!")

    # ════════════════════════════════════════════════════════
    # DEBUG 7: ASSEMBLY TEST AT ONE FREE NODE
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  DEBUG 7: ASSEMBLY AT ONE FREE NODE")
    print(f"{'='*70}")

    free_nodes = (bc_d < 0.5).nonzero().squeeze().tolist()
    if isinstance(free_nodes, int):
        free_nodes = [free_nodes]

    # Pick a free node connected to loaded elements
    node_to_test = None
    for fn in free_nodes:
        connected = ((conn[:, 0] == fn) | (conn[:, 1] == fn))
        if connected.any() and loaded_mask[connected].any():
            node_to_test = fn
            break

    if node_to_test is None and len(free_nodes) > 0:
        node_to_test = free_nodes[len(free_nodes) // 2]

    if node_to_test is not None:
        test_assembly_at_node(data, node_to_test, vert_idx)

    # ════════════════════════════════════════════════════════
    # DEBUG 8: SUMMARY & FIX RECOMMENDATIONS
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  DEBUG 8: DIAGNOSIS SUMMARY")
    print(f"{'='*70}")

    issues = []

    # Load component
    if load_comp == 1:
        issues.append(
            "LOAD IN Y-COMPONENT: elem_load[:, 1] has the UDL,\n"
            "    but physics_loss uses components 0 and 2.\n"
            "    FIX: Use correct vertical axis index.")

    # Direction component
    if n_vert_y > 0 and n_vert_z == 0:
        issues.append(
            "COLUMNS IN Y-DIRECTION: dirs[:, 1] is vertical,\n"
            "    but physics_loss uses dirs[:, 2] for sin_t.\n"
            "    FIX: sin_t = dirs[:, vert_idx]")

    # BC rot
    if n_bc_r == 0:
        issues.append(
            "bc_rot ALL ZERO: Moment equilibrium checked at\n"
            "    supports where reactions exist.\n"
            "    FIX: Set bc_rot=1 at clamped supports.")

    # Displacement convention
    if vert_idx == 1:
        y_uz = data.y_node[:, 1]
        if y_uz.abs().max() < 1e-10:
            issues.append(
                "DISPLACEMENT CONVENTION: y_node[:, 1] is labeled u_z\n"
                "    but may actually be 0 (out-of-plane).\n"
                "    True vertical disp might be in different index.")

    if len(issues) == 0:
        print("\n  No obvious coordinate/load issues found.")
        print("  Check element numbering vs Kratos convention.")
    else:
        print(f"\n  FOUND {len(issues)} ISSUE(S):\n")
        for i, issue in enumerate(issues):
            print(f"  {i+1}. {issue}\n")

    print(f"{'='*70}")


def test_single_element(data, e, vert_idx):
    """Test one element with FEM displacements."""
    conn = data.connectivity
    dirs = data.elem_directions
    coords = data.coords

    i, j = conn[e, 0].item(), conn[e, 1].item()
    L = data.elem_lengths[e].item()
    d = dirs[e]

    EA = (data.prop_E[e] * data.prop_A[e]).item()
    EI = (data.prop_E[e] * data.prop_I22[e]).item()
    q = data.elem_load[e]

    u_fem = data.y_node

    print(f"\n  Element {e}: node {i} → {j}")
    print(f"  Coords i: ({coords[i,0]:.4f}, {coords[i,1]:.4f}, "
          f"{coords[i,2]:.4f})")
    print(f"  Coords j: ({coords[j,0]:.4f}, {coords[j,1]:.4f}, "
          f"{coords[j,2]:.4f})")
    print(f"  L={L:.6f}  dir=({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f})")
    print(f"  EA={EA:.4e}  EI={EI:.4e}")
    print(f"  Load: ({q[0]:+.4e}, {q[1]:+.4e}, {q[2]:+.4e})")

    # FEM DOFs
    uxi, uzi, phi_i = (u_fem[i, 0].item(), u_fem[i, 1].item(),
                        u_fem[i, 2].item())
    uxj, uzj, phi_j = (u_fem[j, 0].item(), u_fem[j, 1].item(),
                        u_fem[j, 2].item())

    print(f"\n  FEM DOFs:")
    print(f"    i: u_x={uxi:+.6e}  u_z={uzi:+.6e}  φ={phi_i:+.6e}")
    print(f"    j: u_x={uxj:+.6e}  u_z={uzj:+.6e}  φ={phi_j:+.6e}")

    # ── Test BOTH conventions ──
    for label, sin_idx in [("Z (code uses)", 2), ("Y (alternative)", 1)]:
        ct = d[0].item()
        st = d[sin_idx].item()

        # Load projection
        q_s = q[0].item() * ct + q[sin_idx].item() * st
        q_n = -q[0].item() * st + q[sin_idx].item() * ct

        # Local DOFs
        u_s_i =  uxi * ct + uzi * st
        u_n_i = -uxi * st + uzi * ct
        u_s_j =  uxj * ct + uzj * st
        u_n_j = -uxj * st + uzj * ct

        # Axial forces
        F_s_i = EA/L * (u_s_i - u_s_j) - q_s * L / 2
        F_s_j = EA/L * (u_s_j - u_s_i) - q_s * L / 2

        # Bending forces
        a = EI / L**3
        L2 = L**2

        F_n_i = (a * (12*u_n_i + 6*L*phi_i - 12*u_n_j + 6*L*phi_j)
                 - q_n * L / 2)
        M_i = (a * (6*L*u_n_i + 4*L2*phi_i - 6*L*u_n_j + 2*L2*phi_j)
               - q_n * L2 / 12)
        F_n_j = (a * (-12*u_n_i - 6*L*phi_i + 12*u_n_j - 6*L*phi_j)
                 - q_n * L / 2)
        M_j = (a * (6*L*u_n_i + 2*L2*phi_i - 6*L*u_n_j + 4*L2*phi_j)
               + q_n * L2 / 12)

        # Global forces
        Fx_i = F_s_i * ct - F_n_i * st
        Fz_i = F_s_i * st + F_n_i * ct
        Fx_j = F_s_j * ct - F_n_j * st
        Fz_j = F_s_j * st + F_n_j * ct

        print(f"\n  ── Convention: sin_t from {label} ──")
        print(f"    cos_t={ct:+.4f}  sin_t={st:+.4f}")
        print(f"    q_s={q_s:+.4e}  q_n={q_n:+.4e}")
        print(f"    Local: u_s_i={u_s_i:+.4e} u_n_i={u_n_i:+.4e}")
        print(f"    Local: u_s_j={u_s_j:+.4e} u_n_j={u_n_j:+.4e}")
        print(f"    End forces (local):")
        print(f"      F_s_i={F_s_i:+.4e}  F_n_i={F_n_i:+.4e}  "
              f"M_i={M_i:+.4e}")
        print(f"      F_s_j={F_s_j:+.4e}  F_n_j={F_n_j:+.4e}  "
              f"M_j={M_j:+.4e}")
        print(f"    End forces (global):")
        print(f"      Fx_i={Fx_i:+.4e}  Fz_i={Fz_i:+.4e}  M_i={M_i:+.4e}")
        print(f"      Fx_j={Fx_j:+.4e}  Fz_j={Fz_j:+.4e}  M_j={M_j:+.4e}")

    # FEM element targets
    if e < data.y_element.shape[0]:
        fem_f = data.y_element[e]
        print(f"\n  FEM element forces: "
              f"N={fem_f[0]:+.4e}  M={fem_f[1]:+.4e}  "
              f"V={fem_f[2]:+.4e}")


def test_assembly_at_node(data, node, vert_idx):
    """Sum element end forces at one free node."""
    conn = data.connectivity
    dirs = data.elem_directions
    u_fem = data.y_node

    connected = ((conn[:, 0] == node) | (conn[:, 1] == node))
    elem_ids = connected.nonzero().squeeze().tolist()
    if isinstance(elem_ids, int):
        elem_ids = [elem_ids]

    print(f"\n  Node {node}: coords=({data.coords[node, 0]:.4f}, "
          f"{data.coords[node, 1]:.4f}, {data.coords[node, 2]:.4f})")
    print(f"  Connected elements: {elem_ids}")

    # Sum forces using BOTH conventions
    for label, sin_idx in [("Z (code)", 2), ("Y (alt)", 1)]:
        sum_Fx = 0.0
        sum_Fz = 0.0
        sum_M = 0.0

        for e in elem_ids:
            i_n, j_n = conn[e, 0].item(), conn[e, 1].item()
            is_i = (i_n == node)
            L = data.elem_lengths[e].item()
            d = dirs[e]
            ct = d[0].item()
            st = d[sin_idx].item()

            EA = (data.prop_E[e] * data.prop_A[e]).item()
            EI = (data.prop_E[e] * data.prop_I22[e]).item()
            q = data.elem_load[e]
            q_s = q[0].item() * ct + q[sin_idx].item() * st
            q_n = -q[0].item() * st + q[sin_idx].item() * ct

            # DOFs
            uxi = u_fem[i_n, 0].item()
            uzi = u_fem[i_n, 1].item()
            phi_i = u_fem[i_n, 2].item()
            uxj = u_fem[j_n, 0].item()
            uzj = u_fem[j_n, 1].item()
            phi_j = u_fem[j_n, 2].item()

            u_s_i = uxi*ct + uzi*st
            u_n_i = -uxi*st + uzi*ct
            u_s_j = uxj*ct + uzj*st
            u_n_j = -uxj*st + uzj*ct

            a = EI / L**3

            if is_i:
                F_s = EA/L*(u_s_i - u_s_j) - q_s*L/2
                F_n = (a*(12*u_n_i + 6*L*phi_i - 12*u_n_j + 6*L*phi_j)
                       - q_n*L/2)
                M = (a*(6*L*u_n_i + 4*L**2*phi_i
                        - 6*L*u_n_j + 2*L**2*phi_j)
                     - q_n*L**2/12)
            else:
                F_s = EA/L*(u_s_j - u_s_i) - q_s*L/2
                F_n = (a*(-12*u_n_i - 6*L*phi_i
                          + 12*u_n_j - 6*L*phi_j)
                       - q_n*L/2)
                M = (a*(6*L*u_n_i + 2*L**2*phi_i
                        - 6*L*u_n_j + 4*L**2*phi_j)
                     + q_n*L**2/12)

            Fx = F_s*ct - F_n*st
            Fz = F_s*st + F_n*ct

            sum_Fx += Fx
            sum_Fz += Fz
            sum_M += M

            side = "node_i" if is_i else "node_j"
            print(f"    Elem {e} ({side}): "
                  f"Fx={Fx:+.4e}  Fz={Fz:+.4e}  M={M:+.4e}")

        print(f"  ── Sum at node {node} [{label}]: "
              f"ΣFx={sum_Fx:+.4e}  ΣFz={sum_Fz:+.4e}  "
              f"ΣM={sum_M:+.4e}")
        print(f"     (should be ≈ 0 for FEM solution)")


if __name__ == "__main__":
    main()