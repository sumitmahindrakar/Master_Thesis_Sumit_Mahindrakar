"""
diagnose_pignn.py — Comprehensive Diagnostic Suite for PIGNN Training
=====================================================================
Run this BEFORE training to identify data, scaling, energy, gradient,
and numerical issues.

Usage:
    python diagnose_pignn.py

Requires:
    - DATA/graph_dataset.pt       (raw graphs)
    - DATA/graph_dataset_norm.pt  (normalised graphs)
    - model.py, energy_loss.py, normalizer.py in same folder
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from collections import Counter, OrderedDict
from step_2_grapg_constr import FrameData

# CURRENT_SUBFOLDER = Path(__file__).resolve().parent
# os.chdir(CURRENT_SUBFOLDER)

# ── make sure we import from the script's directory ──
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

from model import PIGNN
from energy_loss import FrameEnergyLoss
from normalizer import PhysicsScaler

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION  (match your TrainConfig)
# ═══════════════════════════════════════════════════════════
RAW_DATA_PATH  = "DATA/graph_dataset.pt"
NORM_DATA_PATH = "DATA/graph_dataset_norm.pt"
RESULTS_DIR    = "RESULTS"

HIDDEN_DIM  = 128
N_LAYERS    = 6
NODE_IN_DIM = 10
EDGE_IN_DIM = 7
LR          = 2e-4
GRAD_CLIP   = 3.0
TRAIN_RATIO = 0.875

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ═══════════════════════════════════════════════════════════
#  UTILITY — pretty section headers
# ═══════════════════════════════════════════════════════════
def section(title):
    w = 70
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")

def subsection(title):
    print(f"\n  ── {title} ──")

def warn(msg):
    print(f"  ⚠  {msg}")

def ok(msg):
    print(f"  ✓  {msg}")

def fail(msg):
    print(f"  ✗  {msg}")

def info(msg):
    print(f"     {msg}")


# ═══════════════════════════════════════════════════════════
#  DIAGNOSTIC 1 — RAW DATA SANITY
# ═══════════════════════════════════════════════════════════
def diagnose_raw_data(raw_data):
    section("DIAGNOSTIC 1: RAW DATA SANITY")

    n = len(raw_data)
    print(f"  Number of graphs: {n}")
    if n == 0:
        fail("No graphs loaded!"); return

    g0 = raw_data[0]
    N  = g0.num_nodes
    E  = g0.n_elements
    print(f"  Nodes per graph:    {N}")
    print(f"  Elements per graph: {E}")
    print(f"  Edge_index shape:   {list(g0.edge_index.shape)}")
    print(f"  Node feat shape:    {list(g0.x.shape)}")
    print(f"  Edge feat shape:    {list(g0.edge_attr.shape)}")
    print(f"  Target shape:       {list(g0.y_node.shape)}")

    # ─── 1a. node feature columns ───
    subsection("Node Feature Statistics (raw)")
    col_names = ['x/L', 'z/L', 'bc_flag',
                 'Fx', 'Fz', 'My',
                 'E', 'A', 'I', 'degree']
    all_x = torch.cat([g.x for g in raw_data], dim=0)
    for c in range(min(all_x.shape[1], len(col_names))):
        vals = all_x[:, c]
        print(f"    col {c:2d} ({col_names[c]:>8s}): "
              f"min={vals.min():.4e}  max={vals.max():.4e}  "
              f"mean={vals.mean():.4e}  std={vals.std():.4e}")

    # ─── 1b. edge feature columns ───
    subsection("Edge Feature Statistics (raw)")
    ecol_names = ['dx/L', 'dz/L', 'L_e/L_ref',
                  'EA', 'EI', 'cos_a', 'L_elem']
    all_e = torch.cat([g.edge_attr for g in raw_data], dim=0)
    for c in range(min(all_e.shape[1], len(ecol_names))):
        vals = all_e[:, c]
        print(f"    col {c:2d} ({ecol_names[c]:>8s}): "
              f"min={vals.min():.4e}  max={vals.max():.4e}  "
              f"mean={vals.mean():.4e}  std={vals.std():.4e}")

    # ─── 1c. load statistics ───
    subsection("Load Statistics Across All Cases")
    all_Fx, all_Fz, all_My = [], [], []
    total_Fx_per_case = []
    n_loaded_per_case = []
    for g in raw_data:
        Fx = g.x[:, 3]
        Fz = g.x[:, 4]
        My = g.x[:, 5]
        all_Fx.append(Fx)
        all_Fz.append(Fz)
        all_My.append(My)
        total_Fx_per_case.append(Fx.sum().item())
        n_loaded_per_case.append((Fx.abs() > 1e-12).sum().item())

    all_Fx = torch.cat(all_Fx)
    all_Fz = torch.cat(all_Fz)
    all_My = torch.cat(all_My)

    print(f"    Fx: range=[{all_Fx.min():.4f}, {all_Fx.max():.4f}]  "
          f"mean={all_Fx.mean():.4f}  std={all_Fx.std():.4f}")
    print(f"    Fz: range=[{all_Fz.min():.4f}, {all_Fz.max():.4f}]  "
          f"mean={all_Fz.mean():.4f}  std={all_Fz.std():.4f}")
    print(f"    My: range=[{all_My.min():.4f}, {all_My.max():.4f}]  "
          f"mean={all_My.mean():.4f}  std={all_My.std():.4f}")

    total_Fx_arr = np.array(total_Fx_per_case)
    print(f"\n    Total Fx per case: "
          f"min={total_Fx_arr.min():.4f}  "
          f"max={total_Fx_arr.max():.4f}  "
          f"mean={total_Fx_arr.mean():.4f}  "
          f"std={total_Fx_arr.std():.4f}")

    n_near_zero = np.sum(np.abs(total_Fx_arr) < 1.0)
    if n_near_zero > 0:
        warn(f"{n_near_zero}/{n} cases have near-zero total Fx (< 1.0) "
             "→ opposing loads cancel, tiny displacements")

    n_loaded_arr = np.array(n_loaded_per_case)
    print(f"    Loaded nodes per case: "
          f"min={n_loaded_arr.min()}  max={n_loaded_arr.max()}")

    # ─── 1d. displacement statistics ───
    subsection("FEM Displacement Statistics")
    all_ux, all_uz, all_th = [], [], []
    for g in raw_data:
        all_ux.append(g.y_node[:, 0])
        all_uz.append(g.y_node[:, 1])
        all_th.append(g.y_node[:, 2])
    all_ux = torch.cat(all_ux)
    all_uz = torch.cat(all_uz)
    all_th = torch.cat(all_th)

    print(f"    ux: range=[{all_ux.min():.6e}, {all_ux.max():.6e}]  "
          f"std={all_ux.std():.6e}")
    print(f"    uz: range=[{all_uz.min():.6e}, {all_uz.max():.6e}]  "
          f"std={all_uz.std():.6e}")
    print(f"    θy: range=[{all_th.min():.6e}, {all_th.max():.6e}]  "
          f"std={all_th.std():.6e}")

    # check for any zero-displacement cases
    for i, g in enumerate(raw_data):
        u_max = g.y_node.abs().max().item()
        if u_max < 1e-15:
            warn(f"Case {i}: ALL displacements are zero! Check FEM solution.")

    # ─── 1e. BC mask ───
    subsection("Boundary Conditions")
    bc_count = raw_data[0].bc_mask.sum().item()
    free_count = N - bc_count
    print(f"    Fixed nodes: {bc_count}")
    print(f"    Free  nodes: {free_count}")
    fixed_ids = raw_data[0].bc_mask.nonzero(as_tuple=True)[0].tolist()
    print(f"    Fixed node indices: {fixed_ids}")

    # Check: are fixed node displacements zero?
    for i, g in enumerate(raw_data[:5]):
        bc_disp = g.y_node[g.bc_mask].abs().max().item()
        if bc_disp > 1e-10:
            warn(f"Case {i}: Fixed-node displacement = {bc_disp:.4e} (should be ~0)")

    # ─── 1f. energy from FEM solutions ───
    subsection("Energy from FEM Solutions (Π = U − W)")
    loss_fn = FrameEnergyLoss()
    energies = {'Pi': [], 'U': [], 'W': []}
    for i, g in enumerate(raw_data):
        u_true = g.y_node
        U = loss_fn._strain_energy(u_true, g).item()
        W = loss_fn._external_work(u_true, g).item()
        Pi = U - W
        energies['Pi'].append(Pi)
        energies['U'].append(U)
        energies['W'].append(W)
        if i < 5:
            print(f"    Case {i:3d}: U={U:12.6e}  W={W:12.6e}  "
                  f"Π={Pi:12.6e}  U/W={abs(U)/max(abs(W),1e-30):.4f}")

    Pi_arr = np.array(energies['Pi'])
    U_arr  = np.array(energies['U'])
    W_arr  = np.array(energies['W'])

    print(f"\n    Summary across all {n} cases:")
    print(f"      Π:  min={Pi_arr.min():.4e}  max={Pi_arr.max():.4e}  "
          f"mean={Pi_arr.mean():.4e}  std={Pi_arr.std():.4e}")
    print(f"      U:  min={U_arr.min():.4e}  max={U_arr.max():.4e}")
    print(f"      W:  min={W_arr.min():.4e}  max={W_arr.max():.4e}")

    # Check U/W ≈ 0.5 (linear elasticity: U = W/2)
    UoW = np.abs(U_arr) / np.maximum(np.abs(W_arr), 1e-30)
    print(f"      U/|W|: min={UoW.min():.4f}  max={UoW.max():.4f}  "
          f"mean={UoW.mean():.4f}")
    if np.mean(np.abs(UoW - 0.5)) > 0.05:
        warn("U/W deviates from 0.5 → check strain energy formula or BCs")
    else:
        ok("U/W ≈ 0.5 (consistent with linear elasticity)")

    # Check: Π should be negative
    n_positive_Pi = np.sum(Pi_arr > 0)
    if n_positive_Pi > 0:
        warn(f"{n_positive_Pi}/{n} cases have Π > 0 "
             "(should be negative for stable solution)")

    # Energy spread
    if Pi_arr.std() > 0 and Pi_arr.mean() != 0:
        coeff_var = abs(Pi_arr.std() / Pi_arr.mean())
        print(f"      Energy CoV: {coeff_var:.4f}")
        if coeff_var > 2.0:
            warn(f"Large energy variation (CoV={coeff_var:.2f}) "
                 "→ consider per-case normalisation")

    return energies


# ═══════════════════════════════════════════════════════════
#  DIAGNOSTIC 2 — NORMALISATION CHECK
# ═══════════════════════════════════════════════════════════
def diagnose_normalisation(raw_data, norm_data):
    section("DIAGNOSTIC 2: NORMALISATION CHECK")

    if len(raw_data) != len(norm_data):
        fail(f"Dataset size mismatch: raw={len(raw_data)}, "
             f"norm={len(norm_data)}")

    subsection("Characteristic Scales (per graph)")
    F_cs, u_cs, th_cs = [], [], []
    for i, g in enumerate(norm_data):
        if not hasattr(g, 'F_c'):
            fail(f"Graph {i} has no F_c attribute!"); continue
        F_cs.append(g.F_c.item())
        u_cs.append(g.u_c.item())
        th_cs.append(g.theta_c.item())
        if i < 5:
            E_c = (g.F_c * g.u_c).item()
            print(f"    Case {i}: F_c={g.F_c.item():.4e}  "
                  f"u_c={g.u_c.item():.4e}  "
                  f"θ_c={g.theta_c.item():.4e}  "
                  f"E_c={E_c:.4e}")

    F_cs = np.array(F_cs)
    u_cs = np.array(u_cs)
    th_cs = np.array(th_cs)

    print(f"\n    F_c:  range=[{F_cs.min():.4e}, {F_cs.max():.4e}]  "
          f"ratio={F_cs.max()/max(F_cs.min(),1e-30):.2f}")
    print(f"    u_c:  range=[{u_cs.min():.4e}, {u_cs.max():.4e}]  "
          f"ratio={u_cs.max()/max(u_cs.min(),1e-30):.2f}")
    print(f"    θ_c:  range=[{th_cs.min():.4e}, {th_cs.max():.4e}]  "
          f"ratio={th_cs.max()/max(th_cs.min(),1e-30):.2f}")

    # Issue: per-graph u_c varies wildly with opposing loads
    if u_cs.max() / max(u_cs.min(), 1e-30) > 100:
        warn("u_c varies by >100x across cases! "
             "Cases with small net load get tiny u_c, "
             "making normalised predictions huge.")
        warn("Consider using GLOBAL u_c = max(u_c) across all cases.")

    if F_cs.max() / max(F_cs.min(), 1e-30) > 10:
        warn("F_c varies by >10x across cases.")

    # ─── Check normalised features are O(1) ───
    subsection("Normalised Node Feature Statistics")
    col_names = ['x/L', 'z/L', 'bc_flag',
                 'Fx_n', 'Fz_n', 'My_n',
                 'log(E)', 'A_n', 'I_n', 'degree']
    all_x = torch.cat([g.x for g in norm_data], dim=0)
    issues = []
    for c in range(min(all_x.shape[1], len(col_names))):
        vals = all_x[:, c]
        rng = vals.max().item() - vals.min().item()
        mx  = vals.abs().max().item()
        status = "OK" if mx < 10 else "⚠ LARGE"
        if mx > 10:
            issues.append(c)
        print(f"    col {c:2d} ({col_names[c]:>8s}): "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]  "
              f"|max|={mx:.4f}  {status}")

    if issues:
        warn(f"Columns {issues} have |max| > 10 "
             "→ network inputs not well-scaled")
    else:
        ok("All normalised node features are O(1)")

    # ─── Check normalised edge features ───
    subsection("Normalised Edge Feature Statistics")
    ecol_names = ['dx/L', 'dz/L', 'L_e/L_ref',
                  'EA', 'EI', 'cos_a', 'L_elem']
    all_e = torch.cat([g.edge_attr for g in norm_data], dim=0)
    for c in range(min(all_e.shape[1], len(ecol_names))):
        vals = all_e[:, c]
        mx = vals.abs().max().item()
        status = "OK" if mx < 100 else "⚠ LARGE"
        print(f"    col {c:2d} ({ecol_names[c]:>8s}): "
              f"range=[{vals.min():.4e}, {vals.max():.4e}]  "
              f"|max|={mx:.4e}  {status}")

    # Specifically check EA, EI columns — are they normalised?
    EA_vals = all_e[:, 3]
    EI_vals = all_e[:, 4]
    if EA_vals.abs().max() > 1e6:
        warn(f"EA in edge features = {EA_vals.abs().max():.4e} (NOT normalised!) "
             "→ This flows through the energy computation as E*A*L values")
    if EI_vals.abs().max() > 1e3:
        warn(f"EI in edge features = {EI_vals.abs().max():.4e} "
             "→ Large values cause stiff gradients in energy loss")

    # ─── Cross-check: forces normalised consistently? ───
    subsection("Force Normalisation Cross-check")
    for i in range(min(5, len(raw_data))):
        raw_Fx = raw_data[i].x[:, 3]
        nrm_Fx = norm_data[i].x[:, 3]
        F_c    = norm_data[i].F_c.item()
        expected = raw_Fx / F_c
        err = (nrm_Fx - expected).abs().max().item()
        status = "✓" if err < 1e-5 else "✗"
        print(f"    Case {i}: |Fx_norm - Fx_raw/F_c|_max = {err:.6e}  {status}")

    # ─── Energy normalisation cross-check ───
    subsection("Energy Scale (E_c = F_c × u_c)")
    E_cs = F_cs * u_cs
    print(f"    E_c: range=[{E_cs.min():.4e}, {E_cs.max():.4e}]  "
          f"ratio={E_cs.max()/max(E_cs.min(),1e-30):.2f}")
    if E_cs.max() / max(E_cs.min(), 1e-30) > 100:
        warn("E_c varies by >100x → normalised Π has very different scales "
             "across cases, confusing the optimiser")


# ═══════════════════════════════════════════════════════════
#  DIAGNOSTIC 3 — ENERGY COMPUTATION VERIFICATION
# ═══════════════════════════════════════════════════════════
def diagnose_energy_computation(raw_data):
    section("DIAGNOSTIC 3: ENERGY COMPUTATION VERIFICATION")

    loss_fn = FrameEnergyLoss()

    subsection("Strain Energy Formula Check")
    info("For linear elasticity with EB beam:")
    info("  U = ½·EA/L·(Δux)² + EI/L³·[6(Δuz)² − 6L·Δuz·(θi+θj) + L²·(2θi²+2θj²+2θiθj)]")
    info("At equilibrium: U = W/2, so Π = U − W = −U = −W/2")

    for i in range(min(5, len(raw_data))):
        g = raw_data[i]
        u_true = g.y_node

        U = loss_fn._strain_energy(u_true, g).item()
        W = loss_fn._external_work(u_true, g).item()

        # For linear problem: 2U should equal W
        ratio_2U_W = abs(2*U) / max(abs(W), 1e-30)
        err_pct = abs(ratio_2U_W - 1.0) * 100
        status = "✓" if err_pct < 1.0 else ("~" if err_pct < 5 else "✗")

        print(f"    Case {i}: 2U={2*U:.6e}  W={W:.6e}  "
              f"2U/W={ratio_2U_W:.6f}  err={err_pct:.2f}%  {status}")

        if err_pct > 5.0:
            warn(f"Case {i}: 2U/W = {ratio_2U_W:.4f} ≠ 1.0 "
                 "→ strain energy formula may be wrong or BCs not handled")

    # ─── Element-level check ───
    subsection("Element-Level Energy Breakdown (Case 0)")
    g = raw_data[0]
    u_true = g.y_node
    ei = g.edge_index
    src, dst = ei[0], ei[1]
    mask = src < dst
    src_m, dst_m = src[mask], dst[mask]

    EA = g.edge_attr[mask, 3]
    EI = g.edge_attr[mask, 4]
    L  = g.edge_attr[mask, 6]

    n_elem = mask.sum().item()
    print(f"    Elements: {n_elem}")

    for eidx in range(min(5, n_elem)):
        i_n, j_n = src_m[eidx].item(), dst_m[eidx].item()
        ea, ei_v, le = EA[eidx].item(), EI[eidx].item(), L[eidx].item()

        dux = (u_true[j_n, 0] - u_true[i_n, 0]).item()
        duz = (u_true[j_n, 1] - u_true[i_n, 1]).item()
        ti  = u_true[i_n, 2].item()
        tj  = u_true[j_n, 2].item()

        U_ax = 0.5 * ea / le * dux**2
        U_b  = (ei_v / le**3) * (6*duz**2 - 6*le*duz*(ti+tj)
                + le**2*(2*ti**2 + 2*tj**2 + 2*ti*tj))

        print(f"      Elem {eidx} (n{i_n}→n{j_n}): "
              f"L={le:.4f}  EA={ea:.4e}  EI={ei_v:.4e}")
        print(f"        Δux={dux:.6e}  Δuz={duz:.6e}  "
              f"θi={ti:.6e}  θj={tj:.6e}")
        print(f"        U_ax={U_ax:.6e}  U_bend={U_b:.6e}  "
              f"ratio={abs(U_b)/max(abs(U_ax),1e-30):.2f}")

    # ─── Numerical precision test ───
    subsection("Numerical Precision Test (float32 vs float64)")
    g = raw_data[0]
    u32 = g.y_node.float()
    u64 = g.y_node.double()

    # Need to convert edge_attr too for the double computation
    g64 = g.clone()
    g64.edge_attr = g64.edge_attr.double()
    g64.x = g64.x.double()

    U32 = loss_fn._strain_energy(u32, g).item()

    # Manual double computation
    ei = g64.edge_index
    src, dst = ei[0], ei[1]
    mask = src < dst
    src_m, dst_m = src[mask], dst[mask]
    EA_d = g64.edge_attr[mask, 3]
    EI_d = g64.edge_attr[mask, 4]
    L_d  = g64.edge_attr[mask, 6]
    dux_d = u64[dst_m, 0] - u64[src_m, 0]
    duz_d = u64[dst_m, 1] - u64[src_m, 1]
    ti_d  = u64[src_m, 2]
    tj_d  = u64[dst_m, 2]
    U_ax_d = 0.5 * EA_d / L_d * dux_d**2
    U_b_d  = (EI_d / L_d**3) * (6*duz_d**2 - 6*L_d*duz_d*(ti_d+tj_d)
              + L_d**2*(2*ti_d**2 + 2*tj_d**2 + 2*ti_d*tj_d))
    U64 = (U_ax_d + U_b_d).sum().item()

    rel_err = abs(U32 - U64) / max(abs(U64), 1e-30)
    print(f"    U(float32) = {U32:.10e}")
    print(f"    U(float64) = {U64:.10e}")
    print(f"    Relative error: {rel_err:.4e}")
    if rel_err > 1e-4:
        warn(f"Float32 precision error = {rel_err:.2e} > 1e-4 "
             "→ consider using float64 for energy computation")
    else:
        ok(f"Float32 precision adequate (err={rel_err:.2e})")


# ═══════════════════════════════════════════════════════════
#  DIAGNOSTIC 4 — SCALE MISMATCH IN ENERGY
# ═══════════════════════════════════════════════════════════
def diagnose_energy_scales(raw_data, norm_data):
    section("DIAGNOSTIC 4: ENERGY SCALE MISMATCH")

    loss_fn = FrameEnergyLoss()

    subsection("Sensitivity of U and W to small displacements")
    info("If ∂U/∂u >> ∂W/∂u at u≈0, the network cannot escape zero.")

    for i in range(min(3, len(raw_data))):
        g = raw_data[i]
        N = g.num_nodes

        # Compute gradient of energy w.r.t. uniform displacement
        u_test = torch.zeros(N, 3, dtype=torch.float32, requires_grad=True)
        U = loss_fn._strain_energy(u_test, g)
        W = loss_fn._external_work(u_test, g)
        Pi = U - W

        dPi_du = torch.autograd.grad(Pi, u_test, create_graph=False)[0]

        print(f"\n    Case {i} at u=0:")
        print(f"      |dΠ/d(ux)|_max = {dPi_du[:,0].abs().max():.4e}")
        print(f"      |dΠ/d(uz)|_max = {dPi_du[:,1].abs().max():.4e}")
        print(f"      |dΠ/d(θ)| _max = {dPi_du[:,2].abs().max():.4e}")

        # At u=0, ∂U/∂u = 0, so ∂Π/∂u = -∂W/∂u = -F
        F_max = g.x[:, 3:6].abs().max().item()
        print(f"      F_max = {F_max:.4e}")
        print(f"      → At u=0: dΠ/du should equal -F (since U=0 at u=0)")

    # Now check at a small displacement
    subsection("Energy Sensitivity at Small Displacement")
    for i in range(min(3, len(raw_data))):
        g = raw_data[i]
        N = g.num_nodes
        u_true = g.y_node.clone()
        u_scale = u_true.abs().max().item()
        if u_scale < 1e-15:
            continue

        for frac in [0.01, 0.1, 0.5, 1.0]:
            u_test = (frac * u_true).detach().requires_grad_(True)
            U = loss_fn._strain_energy(u_test, g)
            W = loss_fn._external_work(u_test, g)

            dU_du = torch.autograd.grad(U, u_test, create_graph=False)[0]
            dW_du = torch.autograd.grad(
                loss_fn._external_work(u_test.detach().requires_grad_(True), g),
                u_test if u_test.requires_grad else u_test.detach().requires_grad_(True),
                create_graph=False
            )

            print(f"    Case {i}, u = {frac:.0%} × u_true: "
                  f"U={U.item():.4e}  W={W.item():.4e}  "
                  f"|∂U/∂u|_max={dU_du.abs().max().item():.4e}")

    # ─── Scale mismatch in normalised energy ───
    subsection("Normalised Energy Scale Check")
    info("Checking if Π_norm = Π / E_c produces values of O(1)")
    for i in range(min(5, len(raw_data))):
        g_raw  = raw_data[i]
        g_norm = norm_data[i] if i < len(norm_data) else None
        if g_norm is None: continue

        u_true = g_raw.y_node
        U = loss_fn._strain_energy(u_true, g_raw).item()
        W = loss_fn._external_work(u_true, g_raw).item()
        Pi = U - W
        E_c = (g_norm.F_c * g_norm.u_c).item()
        Pi_norm = Pi / max(E_c, 1e-30)

        print(f"    Case {i}: Π={Pi:.4e}  E_c={E_c:.4e}  "
              f"Π/E_c={Pi_norm:.4e}")
        if abs(Pi_norm) > 100:
            warn(f"Case {i}: Π_norm = {Pi_norm:.2f} >> O(1)")
        if abs(Pi_norm) < 0.01:
            warn(f"Case {i}: Π_norm = {Pi_norm:.4f} << O(1)")


# ═══════════════════════════════════════════════════════════
#  DIAGNOSTIC 5 — MODEL INITIALISATION
# ═══════════════════════════════════════════════════════════
def diagnose_model_init(norm_data, raw_data):
    section("DIAGNOSTIC 5: MODEL INITIALISATION")

    device = torch.device(DEVICE)
    model = PIGNN(
        node_in_dim=NODE_IN_DIM,
        edge_in_dim=EDGE_IN_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
    ).to(device)

    loss_fn = FrameEnergyLoss()

    subsection("Forward Pass at Initialisation")
    g = norm_data[0].clone().to(device)
    with torch.no_grad():
        pred = model(g)
    print(f"    pred shape: {list(pred.shape)}")
    print(f"    pred max:   {pred.abs().max().item():.6e}")
    print(f"    pred mean:  {pred.abs().mean().item():.6e}")
    ok("Zero-init decoder → predictions start at 0") if pred.abs().max() < 1e-6 else \
        warn(f"Non-zero initial predictions: {pred.abs().max():.4e}")

    subsection("Gradient at Initialisation (Case 0)")
    model.train()
    model.zero_grad()
    g = norm_data[0].clone().to(device)

    Pi_norm, loss_dict, pred_raw, u_phys = loss_fn(model, g)
    Pi_norm.backward()

    total_norm = 0
    max_grad_name = ""
    max_grad_val = 0
    print(f"\n    Per-parameter gradient norms:")
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            total_norm += gn**2
            if gn > max_grad_val:
                max_grad_val = gn
                max_grad_name = name
            if 'dec' in name or 'enc' in name.split('.')[0]:
                print(f"      {name:45s}: |∇|={gn:.4e}  shape={list(p.shape)}")
    total_norm = total_norm**0.5

    print(f"\n    Total gradient norm: {total_norm:.4e}")
    print(f"    Largest gradient:   {max_grad_name} ({max_grad_val:.4e})")

    if total_norm > 1e6:
        fail(f"Gradient norm = {total_norm:.1e} >> 1 "
             "→ extreme stiffness, optimiser will struggle")
    elif total_norm > 1e4:
        warn(f"Gradient norm = {total_norm:.1e} is large "
             "→ gradient clipping will be aggressive")
    elif total_norm < 1e-3:
        warn(f"Gradient norm = {total_norm:.1e} is very small "
             "→ learning may be extremely slow")
    else:
        ok(f"Gradient norm = {total_norm:.1e} looks reasonable")

    # ─── Effective learning step ───
    subsection("Effective Learning Step Analysis")
    effective_clip = min(GRAD_CLIP, total_norm)
    scale = effective_clip / max(total_norm, 1e-30)
    effective_lr = LR * scale

    print(f"    Configured LR:      {LR:.1e}")
    print(f"    Gradient norm:      {total_norm:.1e}")
    print(f"    Grad clip:          {GRAD_CLIP:.1e}")
    print(f"    Clip ratio:         {scale:.6f}")
    print(f"    Effective LR:       {effective_lr:.4e}")
    print(f"    (= LR × clip/|∇|)")

    if scale < 0.001:
        fail(f"Clipping removes {(1-scale)*100:.1f}% of gradient! "
             "Network can barely learn.")
    elif scale < 0.1:
        warn(f"Clipping removes {(1-scale)*100:.1f}% of gradient.")
    else:
        ok(f"Gradient clipping is mild ({(1-scale)*100:.1f}%)")

    # ─── Energy at u=0 ───
    subsection("Energy Landscape at u=0")
    info("At zero displacement: U=0, W=0, Π=0")
    info("The loss should push Π negative → force u away from 0")
    print(f"    Π_norm at init: {loss_dict['Pi_norm']:.6e}")
    print(f"    U at init:      {loss_dict['U_internal']:.6e}")
    print(f"    W at init:      {loss_dict['W_external']:.6e}")

    return model


# ═══════════════════════════════════════════════════════════
#  DIAGNOSTIC 6 — GRADIENT CONFLICT BETWEEN CASES
# ═══════════════════════════════════════════════════════════
def diagnose_gradient_conflict(norm_data, raw_data, model=None):
    section("DIAGNOSTIC 6: GRADIENT CONFLICT BETWEEN CASES")

    device = torch.device(DEVICE)
    if model is None:
        model = PIGNN(
            node_in_dim=NODE_IN_DIM,
            edge_in_dim=EDGE_IN_DIM,
            hidden_dim=HIDDEN_DIM,
            n_layers=N_LAYERS,
        ).to(device)

    loss_fn = FrameEnergyLoss()
    model.train()

    n_check = min(20, len(norm_data))
    per_case_grads = []
    per_case_Pi = []

    subsection(f"Computing per-case gradients ({n_check} cases)")
    for i in range(n_check):
        g = norm_data[i].clone().to(device)
        model.zero_grad()

        Pi_norm, loss_dict, pred_raw, u_phys = loss_fn(model, g)
        Pi_norm.backward()

        grad_vec = torch.cat([
            p.grad.flatten() for p in model.parameters()
            if p.grad is not None
        ]).detach().cpu()

        per_case_grads.append(grad_vec)
        per_case_Pi.append(Pi_norm.item())

    grads = torch.stack(per_case_grads)  # [n_check, n_params]
    mean_grad = grads.mean(dim=0)

    # ─── Cosine similarity ───
    subsection("Gradient Agreement (cosine similarity)")
    cos_sims = []
    for i in range(n_check):
        cos = torch.nn.functional.cosine_similarity(
            grads[i].unsqueeze(0), mean_grad.unsqueeze(0)
        ).item()
        cos_sims.append(cos)

    cos_arr = np.array(cos_sims)
    print(f"    Mean cosine sim:  {cos_arr.mean():.4f}")
    print(f"    Std:              {cos_arr.std():.4f}")
    print(f"    Min:              {cos_arr.min():.4f}")
    print(f"    Max:              {cos_arr.max():.4f}")

    n_negative = np.sum(cos_arr < 0)
    n_low      = np.sum(cos_arr < 0.3)

    if n_negative > 0:
        fail(f"{n_negative}/{n_check} cases have NEGATIVE cosine sim "
             "→ gradients point in opposite directions!")
    if n_low > n_check * 0.3:
        warn(f"{n_low}/{n_check} cases have cos < 0.3 "
             "→ significant gradient conflict")
    elif n_low > 0:
        warn(f"{n_low}/{n_check} cases have cos < 0.3")
    else:
        ok("All cases have cos > 0.3 (gradients roughly aligned)")

    # ─── Pairwise cosine ───
    subsection("Pairwise Gradient Similarity (first 10)")
    n_pw = min(10, n_check)
    pw_cos = torch.nn.functional.cosine_similarity(
        grads[:n_pw].unsqueeze(1),
        grads[:n_pw].unsqueeze(0),
        dim=2
    )  # [n_pw, n_pw]
    off_diag = pw_cos[~torch.eye(n_pw, dtype=bool)]
    print(f"    Pairwise cos mean: {off_diag.mean():.4f}")
    print(f"    Pairwise cos min:  {off_diag.min():.4f}")

    if off_diag.min() < -0.5:
        fail("Strongly opposing gradients between cases!")

    # ─── Gradient magnitude variation ───
    subsection("Per-case Gradient Magnitude")
    grad_norms = grads.norm(dim=1)
    print(f"    Mean: {grad_norms.mean():.4e}")
    print(f"    Std:  {grad_norms.std():.4e}")
    print(f"    Min:  {grad_norms.min():.4e}")
    print(f"    Max:  {grad_norms.max():.4e}")
    print(f"    Max/Min: {(grad_norms.max()/grad_norms.min()).item():.2f}")

    if (grad_norms.max()/grad_norms.min()).item() > 100:
        warn("Gradient magnitudes vary by >100x "
             "→ some cases dominate the optimisation")

    # ─── Correlation with load pattern ───
    subsection("Gradient Magnitude vs Load Pattern")
    for i in range(min(5, n_check)):
        g = norm_data[i]
        Fx = g.x[:, 3]
        total_F = g.x[:, 3:6].abs().sum().item()
        net_Fx = Fx.sum().item()
        print(f"    Case {i}: |∇|={grad_norms[i]:.4e}  "
              f"Σ|F|={total_F:.2f}  net_Fx={net_Fx:.2f}  "
              f"cos={cos_sims[i]:.4f}")


# ═══════════════════════════════════════════════════════════
#  DIAGNOSTIC 7 — TRAINING STEP SIMULATION
# ═══════════════════════════════════════════════════════════
def diagnose_training_steps(norm_data, raw_data, n_steps=50):
    section("DIAGNOSTIC 7: TRAINING STEP SIMULATION")
    info(f"Running {n_steps} training steps to check learning dynamics")

    device = torch.device(DEVICE)
    model = PIGNN(
        node_in_dim=NODE_IN_DIM,
        edge_in_dim=EDGE_IN_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
    ).to(device)

    loss_fn = FrameEnergyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Use a single case first
    g = norm_data[0].clone().to(device)
    g_raw = raw_data[0]

    print(f"\n    {'Step':>5} | {'Π_norm':>12} | {'U':>12} | {'W':>12} | "
          f"{'|∇|':>10} | {'|u|_max':>10} | {'|u_true|':>10}")
    print(f"    {'-'*85}")

    u_true_max = g_raw.y_node.abs().max().item()
    Pi_history = []

    for step in range(1, n_steps + 1):
        model.train()
        optimizer.zero_grad()

        Pi_norm, loss_dict, pred_raw, u_phys = loss_fn(model, g)

        if torch.isnan(Pi_norm):
            fail(f"NaN at step {step}!")
            break

        Pi_norm.backward()

        gn = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=GRAD_CLIP
        ).item()

        optimizer.step()

        u_max = u_phys.abs().max().item()
        Pi_history.append(loss_dict['Pi_norm'])

        if step <= 10 or step % 10 == 0:
            print(f"    {step:5d} | {loss_dict['Pi_norm']:12.4e} | "
                  f"{loss_dict['U_internal']:12.4e} | "
                  f"{loss_dict['W_external']:12.4e} | "
                  f"{gn:10.2e} | {u_max:10.4e} | {u_true_max:10.4e}")

    # Analyse the trajectory
    subsection("Training Trajectory Analysis")
    Pi_arr = np.array(Pi_history)
    if len(Pi_arr) > 10:
        first10 = Pi_arr[:10].mean()
        last10  = Pi_arr[-10:].mean()
        improvement = (first10 - last10) / max(abs(first10), 1e-30)
        print(f"    Π_norm first 10: {first10:.4e}")
        print(f"    Π_norm last  10: {last10:.4e}")
        print(f"    Improvement:     {improvement*100:.2f}%")

        if improvement < 0.01:
            fail("Less than 1% improvement in 50 steps "
                 "→ learning is stuck")
        elif improvement < 0.1:
            warn(f"Only {improvement*100:.1f}% improvement → slow learning")
        else:
            ok(f"{improvement*100:.1f}% improvement → learning is progressing")

    # ─── Multi-case simulation ───
    subsection(f"Multi-case Training ({min(10, len(norm_data))} cases, 20 epochs)")
    model2 = PIGNN(
        node_in_dim=NODE_IN_DIM,
        edge_in_dim=EDGE_IN_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
    ).to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=LR)

    n_cases = min(10, len(norm_data))
    for epoch in range(1, 21):
        epoch_Pi = 0
        epoch_gn = 0
        for ci in range(n_cases):
            g = norm_data[ci].clone().to(device)
            optimizer2.zero_grad()
            Pi_norm, loss_dict, _, u_phys = loss_fn(model2, g)
            if torch.isnan(Pi_norm): continue
            Pi_norm.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                model2.parameters(), max_norm=GRAD_CLIP
            ).item()
            optimizer2.step()
            epoch_Pi += loss_dict['Pi_norm']
            epoch_gn += gn

        avg_Pi = epoch_Pi / n_cases
        avg_gn = epoch_gn / n_cases

        if epoch <= 5 or epoch % 5 == 0:
            # Check max displacement
            with torch.no_grad():
                g_test = norm_data[0].clone().to(device)
                pred = model2(g_test)
                u_max = (pred * torch.tensor([g_test.u_c, g_test.u_c, g_test.theta_c],
                         device=device)).abs().max().item()
            print(f"    Epoch {epoch:3d}: avg_Π={avg_Pi:.4e}  "
                  f"avg_|∇|={avg_gn:.2e}  |u|_max={u_max:.4e}")


# ═══════════════════════════════════════════════════════════
#  DIAGNOSTIC 8 — PER-GRAPH SCALE ANALYSIS
# ═══════════════════════════════════════════════════════════
def diagnose_per_graph_scales(raw_data, norm_data):
    section("DIAGNOSTIC 8: PER-GRAPH SCALE VARIABILITY")

    info("When F_c, u_c are computed per-graph, cases with")
    info("opposing loads get tiny u_c → hugely amplified normalised outputs")

    subsection("Scale Statistics Across Cases")
    records = []
    for i, (gr, gn) in enumerate(zip(raw_data, norm_data)):
        F_c = gn.F_c.item()
        u_c = gn.u_c.item()
        th_c = gn.theta_c.item()
        E_c = F_c * u_c

        u_max = gr.y_node[:, :2].abs().max().item()
        th_max = gr.y_node[:, 2].abs().max().item()

        # Net load
        net_Fx = gr.x[:, 3].sum().item()
        total_F = gr.x[:, 3:6].abs().sum().item()
        load_cancellation = 1.0 - abs(net_Fx) / max(total_F, 1e-30)

        records.append({
            'case': i, 'F_c': F_c, 'u_c': u_c, 'th_c': th_c,
            'E_c': E_c, 'u_max': u_max, 'th_max': th_max,
            'net_Fx': net_Fx, 'total_F': total_F,
            'cancellation': load_cancellation,
        })

    # Sort by u_c to find extreme cases
    records.sort(key=lambda r: r['u_c'])

    print(f"\n    Cases with SMALLEST u_c (most cancelled loads):")
    for r in records[:5]:
        print(f"      Case {r['case']:3d}: u_c={r['u_c']:.4e}  "
              f"net_Fx={r['net_Fx']:.2f}  cancel={r['cancellation']:.2%}  "
              f"E_c={r['E_c']:.4e}")

    print(f"\n    Cases with LARGEST u_c:")
    for r in records[-5:]:
        print(f"      Case {r['case']:3d}: u_c={r['u_c']:.4e}  "
              f"net_Fx={r['net_Fx']:.2f}  cancel={r['cancellation']:.2%}  "
              f"E_c={r['E_c']:.4e}")

    # Ratio of scales
    u_c_arr = np.array([r['u_c'] for r in records])
    E_c_arr = np.array([r['E_c'] for r in records])
    ratio_uc = u_c_arr.max() / max(u_c_arr.min(), 1e-30)
    ratio_Ec = E_c_arr.max() / max(E_c_arr.min(), 1e-30)

    print(f"\n    u_c max/min ratio: {ratio_uc:.2f}")
    print(f"    E_c max/min ratio: {ratio_Ec:.2f}")

    if ratio_uc > 100:
        fail(f"u_c varies by {ratio_uc:.0f}x across cases! "
             "Use global scales instead of per-graph.")
        subsection("RECOMMENDATION: Global Scales")
        global_F_c = max(r['F_c'] for r in records)
        global_u_c = max(r['u_c'] for r in records)
        global_th_c = max(r['th_c'] for r in records)
        print(f"    Recommended global scales:")
        print(f"      F_c  = {global_F_c:.4e}")
        print(f"      u_c  = {global_u_c:.4e}")
        print(f"      θ_c  = {global_th_c:.4e}")
        print(f"      E_c  = {global_F_c * global_u_c:.4e}")
    elif ratio_uc > 10:
        warn(f"u_c varies by {ratio_uc:.0f}x — may cause issues")
    else:
        ok(f"u_c variation is moderate ({ratio_uc:.1f}x)")


# ═══════════════════════════════════════════════════════════
#  DIAGNOSTIC 9 — EDGE FEATURE (EA, EI) IN ENERGY
# ═══════════════════════════════════════════════════════════
def diagnose_stiffness_in_energy(raw_data, norm_data):
    section("DIAGNOSTIC 9: STIFFNESS VALUES IN ENERGY LOSS")

    info("The energy loss uses edge_attr[:, 3]=EA and edge_attr[:, 4]=EI")
    info("These come from the NORMALISED graph but should be PHYSICAL values.")

    g_raw  = raw_data[0]
    g_norm = norm_data[0]

    EA_raw  = g_raw.edge_attr[:, 3]
    EI_raw  = g_raw.edge_attr[:, 4]
    L_raw   = g_raw.edge_attr[:, 6]

    EA_norm = g_norm.edge_attr[:, 3]
    EI_norm = g_norm.edge_attr[:, 4]
    L_norm  = g_norm.edge_attr[:, 6]

    subsection("Raw vs Normalised Edge Features")
    print(f"    EA raw:  [{EA_raw.min():.4e}, {EA_raw.max():.4e}]")
    print(f"    EA norm: [{EA_norm.min():.4e}, {EA_norm.max():.4e}]")
    print(f"    → Same? {torch.allclose(EA_raw, EA_norm)}")

    print(f"    EI raw:  [{EI_raw.min():.4e}, {EI_raw.max():.4e}]")
    print(f"    EI norm: [{EI_norm.min():.4e}, {EI_norm.max():.4e}]")
    print(f"    → Same? {torch.allclose(EI_raw, EI_norm)}")

    print(f"    L raw:   [{L_raw.min():.4e}, {L_raw.max():.4e}]")
    print(f"    L norm:  [{L_norm.min():.4e}, {L_norm.max():.4e}]")
    print(f"    → Same? {torch.allclose(L_raw, L_norm)}")

    if torch.allclose(EA_raw, EA_norm):
        info("Edge features (EA, EI, L) are NOT changed by normaliser.")
        info("This means the energy loss uses physical EA, EI values directly.")
        info("→ Energy has magnitude O(EA·u²/L) or O(EI·θ²/L)")
        EA_v = EA_raw.abs().max().item()
        EI_v = EI_raw.abs().max().item()
        print(f"\n    EA = {EA_v:.4e}")
        print(f"    EI = {EI_v:.4e}")

        # Expected energy scale
        u_c = norm_data[0].u_c.item()
        th_c = norm_data[0].theta_c.item()
        L_e = L_raw.mean().item()

        U_ax_scale = 0.5 * EA_v / L_e * u_c**2
        U_b_scale  = EI_v / L_e * th_c**2
        F_c = norm_data[0].F_c.item()
        W_scale = F_c * u_c

        print(f"\n    Expected scale for 1 element:")
        print(f"      U_axial  ~ 0.5·EA/L·u_c² = {U_ax_scale:.4e}")
        print(f"      U_bending ~ EI/L·θ_c²     = {U_b_scale:.4e}")
        print(f"      W_ext    ~ F_c·u_c         = {W_scale:.4e}")
        print(f"      U_ax/W  = {U_ax_scale/max(W_scale,1e-30):.4e}")
        print(f"      U_b/W   = {U_b_scale/max(W_scale,1e-30):.4e}")

        if U_ax_scale / max(W_scale, 1e-30) > 10 or U_b_scale / max(W_scale, 1e-30) > 10:
            warn("Stiffness energy >> Work energy at characteristic scales!")
            warn("The network will be penalised for any non-zero displacement.")
    else:
        warn("Edge features were modified by normaliser — check if energy_loss "
             "uses the correct (physical) values!")

    # ─── Check: forces used in W ───
    subsection("Forces Used in External Work")
    info("energy_loss uses data.x[:, 3:6] for forces")
    info("If normalised graph is passed, these are F/F_c, not physical F!")

    F_in_norm = g_norm.x[:, 3:6]
    F_in_raw  = g_raw.x[:, 3:6]

    print(f"    Forces in normalised graph: [{F_in_norm.min():.4f}, {F_in_norm.max():.4f}]")
    print(f"    Forces in raw graph:        [{F_in_raw.min():.4f}, {F_in_raw.max():.4f}]")

    if not torch.allclose(F_in_norm, F_in_raw, atol=1e-6):
        info("Forces DIFFER between raw and norm → normaliser changed them.")
        info("But energy_loss._external_work uses data.x[:, 3:6] directly!")
        info("→ W is computed with normalised forces, but U uses physical EA, EI")
        warn("THIS IS THE CRITICAL BUG: W and U have incompatible scales!")
        print(f"\n    W uses F_normalized = F/F_c")
        print(f"    U uses physical EA, EI")
        print(f"    The loss Π = U(physical) - W(normalised) mixes scales!")
    else:
        ok("Forces are the same in raw and norm graphs.")


# ═══════════════════════════════════════════════════════════
#  DIAGNOSTIC 10 — CHECKPOINT ANALYSIS
# ═══════════════════════════════════════════════════════════
def diagnose_checkpoint(checkpoint_path=None):
    section("DIAGNOSTIC 10: CHECKPOINT ANALYSIS")

    if checkpoint_path is None:
        # Try common paths
        candidates = [
            os.path.join(RESULTS_DIR, 'best.pt'),
            os.path.join(RESULTS_DIR, 'final.pt'),
            os.path.join(RESULTS_DIR, 'epoch_2000.pt'),
        ]
        for c in candidates:
            if os.path.exists(c):
                checkpoint_path = c
                break

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        info("No checkpoint found. Skipping.")
        return

    print(f"  Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

    print(f"  Keys: {list(ckpt.keys())}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Best Π: {ckpt.get('best_Pi', '?')}")

    losses = ckpt.get('losses', {})
    if losses:
        subsection("Loss Dictionary")
        for k, v in losses.items():
            if isinstance(v, dict):
                print(f"    {k}:")
                for kk, vv in v.items():
                    print(f"      {kk}: {vv}")
            else:
                print(f"    {k}: {v}")

    # ─── Analyse model weights ───
    subsection("Model Weight Statistics")
    state = ckpt.get('model_state', {})
    for name, tensor in state.items():
        if tensor.numel() > 1:
            print(f"    {name:45s}: shape={list(tensor.shape)}  "
                  f"|mean|={tensor.abs().mean():.4e}  "
                  f"|max|={tensor.abs().max():.4e}  "
                  f"std={tensor.std():.4e}")

    # Check decoder weights specifically
    subsection("Decoder Weight Analysis")
    for dec_name in ['dec_ux', 'dec_uz', 'dec_th']:
        for suffix in ['.0.weight', '.0.bias', '.2.weight', '.2.bias']:
            key = f"{dec_name}{suffix}"
            if key in state:
                t = state[key]
                print(f"    {key:30s}: |max|={t.abs().max():.6e}  "
                      f"mean={t.mean():.6e}")
                if t.abs().max() < 1e-6:
                    warn(f"{key} is still near zero → decoder not learning")


# ═══════════════════════════════════════════════════════════
#  DIAGNOSTIC 11 — RECOMMENDED FIXES
# ═══════════════════════════════════════════════════════════
def generate_recommendations(raw_data, norm_data):
    section("DIAGNOSTIC 11: RECOMMENDATIONS")

    issues = []

    # 1. Check scale consistency in energy
    g_norm = norm_data[0]
    F_in_norm = g_norm.x[:, 3:6]
    EA_norm = g_norm.edge_attr[:, 3]

    if F_in_norm.abs().max() <= 1.1 and EA_norm.abs().max() > 1e3:
        issues.append(("CRITICAL", "Scale mismatch in energy",
            "Forces are normalised (O(1)) but EA/EI are physical (O(1e9)).\n"
            "     Fix: Either pass raw graph to energy_loss, or normalise EA/EI too.\n"
            "     Best: Use raw data for energy computation, normalised data only for\n"
            "     node/edge features fed to the GNN encoder."))

    # 2. Check per-graph scale variation
    u_cs = [g.u_c.item() for g in norm_data]
    if max(u_cs) / max(min(u_cs), 1e-30) > 100:
        issues.append(("HIGH", "Per-graph u_c varies by >100x",
            "Use global u_c = max(u_c across all cases).\n"
            "     This ensures consistent normalisation."))

    # 3. Check gradient magnitude
    device = torch.device(DEVICE)
    model = PIGNN(NODE_IN_DIM, EDGE_IN_DIM, HIDDEN_DIM, N_LAYERS).to(device)
    loss_fn = FrameEnergyLoss()
    model.train()
    model.zero_grad()
    g = norm_data[0].clone().to(device)
    Pi, _, _, _ = loss_fn(model, g)
    Pi.backward()
    gn = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5

    if gn > 1e5:
        issues.append(("HIGH", f"Initial gradient norm = {gn:.1e}",
            "Reduce by:\n"
            "     1. Non-dimensionalising the energy\n"
            "     2. Increasing grad_clip\n"
            "     3. Reducing LR\n"
            "     4. Normalising EA, EI in edge features"))

    if gn / GRAD_CLIP > 100:
        issues.append(("HIGH", f"Grad clip ratio = {gn/GRAD_CLIP:.0f}x",
            f"Clip at {GRAD_CLIP} removes {(1-GRAD_CLIP/gn)*100:.1f}% of gradient.\n"
            f"     Consider increasing grad_clip to {gn*0.1:.0f} or fixing root cause."))

    # 4. Check for load cancellation
    n_cancel = 0
    for g in raw_data:
        net_F = g.x[:, 3:6].sum(dim=0).abs().max().item()
        total_F = g.x[:, 3:6].abs().sum().item()
        if net_F < 0.1 * total_F:
            n_cancel += 1
    if n_cancel > len(raw_data) * 0.3:
        issues.append(("MEDIUM", f"{n_cancel}/{len(raw_data)} cases have >90% load cancellation",
            "Many cases have opposing loads that nearly cancel.\n"
            "     These cases have tiny displacements and dominate u_c.\n"
            "     Fix: Use global scales or filter out near-zero cases."))

    # 5. Material stiffness
    E_val = raw_data[0].x[0, 6].item()
    if E_val > 1e8:
        issues.append(("MEDIUM", f"High stiffness E = {E_val:.1e}",
            "High E → tiny displacements → poor float32 precision.\n"
            "     Consider non-dimensionalising: use E_ref as reference.\n"
            "     Or work in different units (kN, mm, MPa instead of N, m, Pa)."))

    # Print recommendations
    print()
    if not issues:
        ok("No major issues detected!")
    else:
        for severity, title, fix in issues:
            marker = "🔴" if severity == "CRITICAL" else "🟡" if severity == "HIGH" else "🟠"
            print(f"  {marker} [{severity}] {title}")
            print(f"     {fix}")
            print()


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  PIGNN COMPREHENSIVE DIAGNOSTIC SUITE")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Raw data:  {RAW_DATA_PATH}")
    print(f"  Norm data: {NORM_DATA_PATH}")

    # ── Load data ──
    print(f"\n  Loading data...")
    if not os.path.exists(RAW_DATA_PATH):
        fail(f"Raw data not found: {RAW_DATA_PATH}")
        return
    if not os.path.exists(NORM_DATA_PATH):
        fail(f"Normalised data not found: {NORM_DATA_PATH}")
        return

    raw_data  = torch.load(RAW_DATA_PATH, weights_only=False)
    norm_data = torch.load(NORM_DATA_PATH, weights_only=False)

    # Force to CPU
    raw_data  = [g.cpu() for g in raw_data]
    norm_data = [g.cpu() for g in norm_data]

    # Ensure scales exist
    if not hasattr(norm_data[0], 'F_c'):
        print("  Computing scales for normalised data...")
        norm_data = PhysicsScaler.compute_and_store_list(norm_data, clone=False)
    if not hasattr(raw_data[0], 'F_c'):
        print("  Computing scales for raw data...")
        raw_data = PhysicsScaler.compute_and_store_list(raw_data, clone=True)

    print(f"  Raw graphs:  {len(raw_data)}")
    print(f"  Norm graphs: {len(norm_data)}")

    # ── Run all diagnostics ──
    try:
        energies = diagnose_raw_data(raw_data)
    except Exception as e:
        fail(f"DIAGNOSTIC 1 failed: {e}")
        import traceback; traceback.print_exc()

    try:
        diagnose_normalisation(raw_data, norm_data)
    except Exception as e:
        fail(f"DIAGNOSTIC 2 failed: {e}")
        import traceback; traceback.print_exc()

    try:
        diagnose_energy_computation(raw_data)
    except Exception as e:
        fail(f"DIAGNOSTIC 3 failed: {e}")
        import traceback; traceback.print_exc()

    try:
        diagnose_energy_scales(raw_data, norm_data)
    except Exception as e:
        fail(f"DIAGNOSTIC 4 failed: {e}")
        import traceback; traceback.print_exc()

    try:
        model = diagnose_model_init(norm_data, raw_data)
    except Exception as e:
        fail(f"DIAGNOSTIC 5 failed: {e}")
        import traceback; traceback.print_exc()
        model = None

    try:
        diagnose_gradient_conflict(norm_data, raw_data, model)
    except Exception as e:
        fail(f"DIAGNOSTIC 6 failed: {e}")
        import traceback; traceback.print_exc()

    try:
        diagnose_training_steps(norm_data, raw_data, n_steps=50)
    except Exception as e:
        fail(f"DIAGNOSTIC 7 failed: {e}")
        import traceback; traceback.print_exc()

    try:
        diagnose_per_graph_scales(raw_data, norm_data)
    except Exception as e:
        fail(f"DIAGNOSTIC 8 failed: {e}")
        import traceback; traceback.print_exc()

    try:
        diagnose_stiffness_in_energy(raw_data, norm_data)
    except Exception as e:
        fail(f"DIAGNOSTIC 9 failed: {e}")
        import traceback; traceback.print_exc()

    try:
        diagnose_checkpoint()
    except Exception as e:
        fail(f"DIAGNOSTIC 10 failed: {e}")
        import traceback; traceback.print_exc()

    try:
        generate_recommendations(raw_data, norm_data)
    except Exception as e:
        fail(f"DIAGNOSTIC 11 failed: {e}")
        import traceback; traceback.print_exc()

    # ── Summary ──
    section("DIAGNOSTIC COMPLETE")
    print(f"  Review the output above to identify issues.")
    print(f"  Focus on 🔴 CRITICAL and 🟡 HIGH items first.")
    print(f"  The most common root cause is scale mismatch between")
    print(f"  normalised forces (in W) and physical stiffness (in U).")


if __name__ == "__main__":
    main()