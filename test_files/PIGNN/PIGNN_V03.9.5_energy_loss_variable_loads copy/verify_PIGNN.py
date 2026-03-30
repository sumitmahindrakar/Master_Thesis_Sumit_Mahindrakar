"""
=================================================================
verify_pignn.py — Cross-Verification: PIGNN Predictions vs Kratos
=================================================================

Compares PIGNN predictions against Kratos FEA ground truth at
3 specific nodes:
  1. Top corner node
  2. Middle of top beam 1
  3. Middle of top beam 2

Parameters compared:
  - Displacements: ux, uz, θy
  - Forces: N (axial), V (shear) at connected elements
  - Moments: M (bending) at connected elements

Also reports:
  - Initial, best, and final loss values (all 6 terms + total)
  - Material/section properties: E, I, A, UDL, EA, EI

INPUTS (already saved — no Kratos re-run needed):
  DATA/graph_dataset.pt    — raw PyG graphs with Kratos ground truth
  RESULTS/best.pt          — trained model checkpoint
  RESULTS/history.pt       — training loss history

OUTPUT:
  RESULTS/verification_report.txt
  RESULTS/verification_plot.png

USAGE:
  python verify_pignn.py
=================================================================
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Setup path ──
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

from model import PIGNN


# ================================================================
# CONFIGURATION  (modify these if your paths / architecture differ)
# ================================================================

DATA_PATH       = "DATA/graph_dataset.pt"
CHECKPOINT_PATH = "RESULTS/best.pt"          # or "RESULTS/final.pt"
HISTORY_PATH    = "RESULTS/history.pt"
REPORT_PATH     = "RESULTS/verification_report.txt"
PLOT_PATH       = "RESULTS/verification_plot.png"

CASE_INDEX = 0          # which case to verify (0-indexed)

# Set to dict to override auto-detection, e.g.:
#   MANUAL_NODES = {'top_corner': 10, 'mid_top_beam_1': 8, 'mid_top_beam_2': 6}
MANUAL_NODES = None

# Must match the architecture used during training
MODEL_CONFIG = dict(
    node_in_dim = 9,
    edge_in_dim = 10,
    hidden_dim  = 128,
    n_layers    = 6,
)

FACE_NAMES = ['+x', '-x', '+z', '-z']


# ================================================================
# 1.  NODE IDENTIFICATION
# ================================================================

def find_verification_nodes(data, manual=None):
    """
    Automatically locate three nodes at the top of the frame:
      top_corner      — rightmost node at max z
      mid_top_beam_1  — midpoint of 1st top beam segment
      mid_top_beam_2  — midpoint of 2nd top beam segment

    Falls back gracefully for single-bay frames.
    """
    if manual is not None:
        print(f"  Using manual nodes: {manual}")
        return manual

    coords = data.coords.numpy()
    conn   = data.connectivity.numpy()
    N, E   = len(coords), len(conn)

    # --- top level ---
    z_unique = np.unique(np.round(coords[:, 2], 2))
    z_top    = z_unique[-1]
    tol      = max(0.1, 0.05 * (z_unique[-1] - z_unique[0]))
    top_mask = np.abs(coords[:, 2] - z_top) < tol
    top_nodes = np.where(top_mask)[0]

    x_at_top   = coords[top_nodes, 0]
    order      = np.argsort(x_at_top)
    top_sorted = top_nodes[order]

    if len(top_sorted) < 2:
        return {
            'top_corner':      int(top_sorted[0]) if len(top_sorted) else 0,
            'mid_top_beam_1':  1,
            'mid_top_beam_2':  2,
        }

    # --- column junctions (top nodes that also connect downward) ---
    junctions = {int(top_sorted[0]), int(top_sorted[-1])}
    for n in top_sorted:
        for e_idx in range(E):
            nA, nB = conn[e_idx]
            if nA == n or nB == n:
                other = nB if nA == n else nA
                dz = abs(coords[other, 2] - coords[n, 2])
                dx = abs(coords[other, 0] - coords[n, 0])
                if dz > dx and dz > 0.5:
                    junctions.add(int(n))
    junctions = sorted(junctions, key=lambda n: coords[n, 0])

    # --- beam-segment midpoints ---
    beam_mids = []
    for i in range(len(junctions) - 1):
        j1, j2 = junctions[i], junctions[i + 1]
        x_mid  = 0.5 * (coords[j1, 0] + coords[j2, 0])
        dists  = np.abs(coords[top_sorted, 0] - x_mid)
        # avoid picking a junction itself
        for j in junctions:
            idx_j = np.where(top_sorted == j)[0]
            if len(idx_j):
                dists[idx_j[0]] = 1e10
        best = top_sorted[np.argmin(dists)]
        # if all candidates were junctions, allow picking one
        if dists.min() > 1e9:
            dists2 = np.abs(coords[top_sorted, 0] - x_mid)
            best = top_sorted[np.argmin(dists2)]
        beam_mids.append(int(best))

    result = {'top_corner': int(top_sorted[-1])}
    if len(beam_mids) >= 2:
        result['mid_top_beam_1'] = beam_mids[0]
        result['mid_top_beam_2'] = beam_mids[1]
    elif len(beam_mids) == 1:
        result['mid_top_beam_1'] = beam_mids[0]
        n_top = len(top_sorted)
        result['mid_top_beam_2'] = int(top_sorted[max(0, n_top // 3)])
    else:
        n_top = len(top_sorted)
        result['mid_top_beam_1'] = int(top_sorted[n_top // 3])
        result['mid_top_beam_2'] = int(top_sorted[2 * n_top // 3])

    print(f"  Top level z = {z_top:.4f} m   ({len(top_sorted)} nodes)")
    print(f"  Column junctions: {junctions}")
    for lbl, idx in result.items():
        c = coords[idx]
        print(f"    {lbl:20s}: node {idx:3d}  "
              f"({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})")
    return result


# ================================================================
# 2.  ELEMENT CONNECTIVITY AT A NODE
# ================================================================

def get_connected_elements(node_idx, data):
    """Return list of dicts describing every element meeting this node."""
    conn = data.connectivity.numpy()
    fm   = data.face_mask.numpy()
    feid = data.face_element_id.numpy()
    faa  = data.face_is_A_end.numpy()

    elems = []
    for f in range(4):
        if fm[node_idx, f] < 0.5:
            continue
        e    = int(feid[node_idx, f])
        is_A = int(faa[node_idx, f]) == 1
        nA, nB = conn[e]
        elems.append(dict(
            elem_idx   = e,
            is_A_end   = is_A,
            other_node = int(nB if is_A else nA),
            face_idx   = f,
        ))
    return elems


# ================================================================
# 3.  DISPLACEMENT COMPARISON
# ================================================================

def compare_displacement(node_idx, pred, data):
    p = pred[node_idx, 0:3].cpu().numpy()
    k = data.y_node[node_idx].cpu().numpy()
    err = p - k
    rel = np.abs(err) / np.where(np.abs(k) > 1e-15, np.abs(k), 1.0)
    return dict(pignn=p, kratos=k, error=err, rel_error=rel)


# ================================================================
# 4.  FORCE / MOMENT COMPARISON
# ================================================================

def compare_forces(node_idx, pred, data):
    """
    For every element connected to *node_idx*:
      1. grab PIGNN face force (global)
      2. rotate to element-local frame
      3. apply sign convention  → internal N, V, M
      4. compare with Kratos y_element
    """
    connected   = get_connected_elements(node_idx, data)
    face_forces = pred[node_idx, 3:15].reshape(4, 3).cpu().numpy()

    results = []
    for info in connected:
        e    = info['elem_idx']
        is_A = info['is_A_end']
        f    = info['face_idx']

        # Kratos (element-level, local coords)
        kN = data.y_element[e, 0].item()
        kM = data.y_element[e, 1].item()
        kV = data.y_element[e, 2].item()

        # PIGNN face force (global)
        ff_g = face_forces[f]                     # [Fx, Fz, My]

        # global → local rotation
        d   = data.elem_directions[e].numpy()
        c_a = d[0];  s_a = d[2]
        ff_l = np.array([
             ff_g[0]*c_a + ff_g[1]*s_a,          # Fx_loc
            -ff_g[0]*s_a + ff_g[1]*c_a,          # Fz_loc
             ff_g[2],                             # My_loc
        ])

        # sign convention → internal forces
        if is_A:
            pN, pV, pM = -ff_l[0], -ff_l[1], -ff_l[2]
        else:
            pN, pV, pM =  ff_l[0],  ff_l[1],  ff_l[2]

        def _rel(a, b):
            return abs(a - b) / max(abs(b), 1e-15)

        results.append(dict(
            elem_idx   = e,
            is_A_end   = is_A,
            end_label  = 'A-end' if is_A else 'B-end',
            face_idx   = f,
            face_name  = FACE_NAMES[f],
            other_node = info['other_node'],
            direction  = d,
            length     = data.elem_lengths[e].item(),
            E          = data.prop_E[e].item(),
            A          = data.prop_A[e].item(),
            I22        = data.prop_I22[e].item(),
            ff_global  = ff_g,
            ff_local   = ff_l,
            pignn      = dict(N=pN, V=pV, M=pM),
            kratos     = dict(N=kN, V=kV, M=kM),
            error      = dict(N=pN-kN, V=pV-kV, M=pM-kM),
            rel_error  = dict(N=_rel(pN,kN), V=_rel(pV,kV), M=_rel(pM,kM)),
        ))
    return results


# ================================================================
# 5.  PROPERTY EXTRACTION
# ================================================================

def extract_properties(data):
    coords = data.coords.numpy()
    E  = data.prop_E.numpy()
    A  = data.prop_A.numpy()
    I  = data.prop_I22.numpy()
    el = data.elem_load.numpy()

    loaded = np.where(np.linalg.norm(el, axis=1) > 1e-10)[0]
    udl = el[loaded[0]] if len(loaded) else np.zeros(3)

    ll = data.x[:, 5:8].numpy()
    n_loaded_nodes = int((np.linalg.norm(ll, axis=1) > 1e-10).sum())

    bc_d = data.bc_disp.squeeze(-1).numpy()
    bc_r = data.bc_rot.squeeze(-1).numpy()
    n_sup   = int((bc_d > 0.5).sum())
    n_fixed = int((bc_r > 0.5).sum())

    return dict(
        E=E, A=A, I22=I, EA=E*A, EI=E*I,
        UDL=udl,
        n_loaded_elements  = len(loaded),
        n_loaded_nodes     = n_loaded_nodes,
        x_span   = coords[:,0].max() - coords[:,0].min(),
        z_height = coords[:,2].max() - coords[:,2].min(),
        n_nodes    = int(data.num_nodes),
        n_elements = int(data.n_elements),
        n_supports = n_sup,
        n_fixed    = n_fixed,
        n_pinned   = n_sup - n_fixed,
        case_id          = int(data.case_id)          if hasattr(data,'case_id')          else -1,
        nearest_node_id  = int(data.nearest_node_id)  if hasattr(data,'nearest_node_id')  else -1,
        traced_element_id= int(data.traced_element_id) if hasattr(data,'traced_element_id') else -1,
    )


# ================================================================
# 6.  LOSS SUMMARY
# ================================================================

def extract_loss_summary(history):
    if history is None:
        return None
    totals = history.get('total', [])
    if not totals:
        return None

    best_idx = int(np.argmin(totals))
    out = dict(
        total_epochs = len(totals),
        best_epoch   = history['epoch'][best_idx] if 'epoch' in history else best_idx+1,
    )
    for k in ['L_eq','L_free','L_sup','L_N','L_M','L_V','total']:
        v = history.get(k, [])
        if v:
            out[k] = dict(initial=v[0], best=v[best_idx], final=v[-1])

    de = history.get('val_disp_error', [])
    if de:
        out['disp_error'] = dict(initial=de[0], best=min(de), final=de[-1])
    return out


# ================================================================
# 7.  REPORT GENERATION
# ================================================================

def generate_report(props, loss_sum, node_res, path):
    L = []
    def add(s=""):
        L.append(s)
    W = 80

    add("=" * W)
    add("  PIGNN vs KRATOS — CROSS-VERIFICATION REPORT")
    add(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add("=" * W)

    # ── SECTION 1: PROPERTIES ──
    add("\n" + "─"*W)
    add("  SECTION 1: CASE PROPERTIES")
    add("─"*W)
    add(f"  Case ID:              {props['case_id']}")
    add(f"  Nodes / Elements:     {props['n_nodes']} / {props['n_elements']}")
    add(f"  Supports:             {props['n_supports']} "
        f"({props['n_fixed']} fixed, {props['n_pinned']} pinned)")
    add(f"  Frame span (X):       {props['x_span']:.4f} m")
    add(f"  Frame height (Z):     {props['z_height']:.4f} m")
    add(f"  Loaded elements:      {props['n_loaded_elements']}")
    add(f"  Response node:        {props['nearest_node_id']}")
    add(f"  Traced element:       {props['traced_element_id']}")
    add(f"")
    add(f"  Material & Section:")
    add(f"  {'Property':<16} {'Min':>14}  {'Max':>14}  {'Unit':<6}")
    add(f"  {'─'*56}")
    for tag, arr, u in [
        ('E (Young)',   props['E'],  'Pa'),
        ('A (Area)',    props['A'],  'm²'),
        ('I22 (Inertia)',props['I22'],'m⁴'),
        ('EA',          props['EA'], 'N'),
        ('EI',          props['EI'], 'N·m²'),
    ]:
        add(f"  {tag:<16} {arr.min():>14.4e}  {arr.max():>14.4e}  {u}")
    add(f"")
    add(f"  Applied UDL (global):  qx={props['UDL'][0]:.4f}  "
        f"qy={props['UDL'][1]:.4f}  qz={props['UDL'][2]:.4f}  N/m")

    # ── SECTION 2: LOSSES ──
    add("\n" + "─"*W)
    add("  SECTION 2: TRAINING LOSS SUMMARY")
    add("─"*W)
    if loss_sum:
        add(f"  Total epochs: {loss_sum['total_epochs']}   "
            f"Best epoch: {loss_sum['best_epoch']}")
        add(f"")
        add(f"  {'Loss':<10} {'Initial':>14} {'Best':>14} {'Final':>14}")
        add(f"  {'─'*56}")
        for k in ['L_eq','L_free','L_sup','L_N','L_M','L_V','total']:
            if k in loss_sum:
                v = loss_sum[k]
                note = ""
                if k == 'L_M': note = "  ← moment"
                if k == 'L_V': note = "  ← shear"
                add(f"  {k:<10} {v['initial']:>14.6e} {v['best']:>14.6e} "
                    f"{v['final']:>14.6e}{note}")
        if 'disp_error' in loss_sum:
            v = loss_sum['disp_error']
            add(f"")
            add(f"  Disp error vs Kratos (relative L2):")
            add(f"    Initial {v['initial']:.6e}  "
                f"Best {v['best']:.6e}  Final {v['final']:.6e}")
    else:
        add("  (no training history found)")

    # ── SECTION 3: NODE COMPARISON ──
    add("\n" + "─"*W)
    add("  SECTION 3: NODE-LEVEL COMPARISON")
    add("─"*W)
    add("  Note: PIGNN forces are element-end face forces;")
    add("        Kratos values are element-centroid. Minor mismatch expected.")

    for label, nd in node_res.items():
        c = nd['coords']
        add(f"\n  ╔{'═'*62}╗")
        add(f"  ║  {label:<26} node {nd['node_idx']:<5}"
            f"  ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})  ║")
        add(f"  ╚{'═'*62}╝")

        # displacements
        d = nd['displacement']
        add(f"    {'DOF':<6} {'PIGNN':>14} {'Kratos':>14} "
            f"{'Abs Err':>14} {'Rel Err':>10} {'Unit':<4}")
        add(f"    {'─'*66}")
        for i, (nm, un) in enumerate(
                [('ux','m'),('uz','m'),('θy','rad')]):
            add(f"    {nm:<6} {d['pignn'][i]:>14.6e} {d['kratos'][i]:>14.6e} "
                f"{d['error'][i]:>14.6e} {d['rel_error'][i]:>10.4f} {un}")

        # forces
        for fc in nd['force_comparisons']:
            e = fc['elem_idx']
            add(f"")
            add(f"    Element {e} ({fc['end_label']}, face {fc['face_name']})"
                f"  L={fc['length']:.4f}m")
            add(f"      E={fc['E']:.4e}  A={fc['A']:.4e}  "
                f"I22={fc['I22']:.4e}")
            add(f"      dir=({fc['direction'][0]:.4f}, "
                f"{fc['direction'][1]:.4f}, {fc['direction'][2]:.4f})  "
                f"other node={fc['other_node']}")
            add(f"      face_global: Fx={fc['ff_global'][0]:.4e}  "
                f"Fz={fc['ff_global'][1]:.4e}  My={fc['ff_global'][2]:.4e}")
            add(f"      face_local:  Fx={fc['ff_local'][0]:.4e}  "
                f"Fz={fc['ff_local'][1]:.4e}  My={fc['ff_local'][2]:.4e}")
            add(f"      {'Qty':<6} {'PIGNN':>14} {'Kratos':>14} "
                f"{'Abs Err':>14} {'Rel':>10} {'Unit':<4}")
            add(f"      {'─'*62}")
            for q, u in [('N','N'),('V','N'),('M','N·m')]:
                add(f"      {q:<6} {fc['pignn'][q]:>14.6e} "
                    f"{fc['kratos'][q]:>14.6e} "
                    f"{fc['error'][q]:>14.6e} "
                    f"{fc['rel_error'][q]:>10.4f} {u}")

    # ── SECTION 4: SUMMARY ──
    add("\n" + "─"*W)
    add("  SECTION 4: OVERALL ACCURACY")
    add("─"*W)

    all_dr, all_fr = [], []
    for nd in node_res.values():
        all_dr.extend(nd['displacement']['rel_error'].tolist())
        for fc in nd['force_comparisons']:
            for q in ('N','V','M'):
                all_fr.append(fc['rel_error'][q])

    if all_dr:
        add(f"  Displacement rel-error:  mean {np.mean(all_dr):.6e}  "
            f"max {np.max(all_dr):.6e}")
    if all_fr:
        add(f"  Force/moment rel-error:  mean {np.mean(all_fr):.6e}  "
            f"max {np.max(all_fr):.6e}")

    avg = np.mean(all_dr) if all_dr else 1.0
    add(f"")
    if   avg < 0.01:  add("  ✅  Displacement accuracy: EXCELLENT (< 1 %)")
    elif avg < 0.05:  add("  ⚠️   Displacement accuracy: GOOD (< 5 %)")
    elif avg < 0.20:  add("  ⚠️   Displacement accuracy: MODERATE (< 20 %)")
    else:
        add(f"  ❌  Displacement accuracy: POOR ({avg*100:.0f} %)")
        add("       Expected for naive-autograd physics-only training.")

    add("\n" + "="*W)
    add("  END OF REPORT")
    add("="*W)

    report = "\n".join(L)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    return report


# ================================================================
# 8.  VISUALISATION
# ================================================================

def plot_verification(data, pred, ver_nodes, props, save_path):
    try:
        import matplotlib
        matplotlib.use('Agg')          # non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("  ⚠  matplotlib not found — skipping plot")
        return

    coords = data.coords.numpy()
    conn   = data.connectivity.numpy()
    E_num  = data.n_elements

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # ---- panel 1: frame with verification nodes ----
    ax = axes[0]
    for e in range(E_num):
        n1, n2 = conn[e]
        xs = [coords[n1,0], coords[n2,0]]
        zs = [coords[n1,2], coords[n2,2]]
        dx = abs(coords[n2,0]-coords[n1,0])
        dz = abs(coords[n2,2]-coords[n1,2])
        ax.plot(xs, zs, '-', color='steelblue' if dz>dx else 'coral',
                lw=2.5)
    for i in range(data.num_nodes):
        is_sup = data.bc_disp[i].item() > 0.5
        ax.plot(coords[i,0], coords[i,2],
                '^' if is_sup else 'o',
                ms=10 if is_sup else 3,
                color='green' if is_sup else 'gray',
                mec='black', zorder=5)
        ax.text(coords[i,0]+0.12, coords[i,2]+0.12, str(i),
                fontsize=5, color='gray')
    vc = ['red','blue','orange']
    for idx,(lbl,ni) in enumerate(ver_nodes.items()):
        ax.plot(coords[ni,0], coords[ni,2], '*', ms=18,
                color=vc[idx], mec='black', zorder=10)
        ax.text(coords[ni,0]+0.25, coords[ni,2]+0.25, lbl,
                fontsize=7, color=vc[idx], fontweight='bold')
    ax.set_title(f"Frame (case {props['case_id']})")
    ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # ---- panel 2: displacement bar chart ----
    ax = axes[1]
    labels_list = list(ver_nodes.keys())
    dof_names = ['ux','uz','θy']
    x_pos = np.arange(len(labels_list))
    bw = 0.12
    for d_i, dof in enumerate(dof_names):
        pv = [pred[ver_nodes[l], d_i].item() for l in labels_list]
        kv = [data.y_node[ver_nodes[l], d_i].item() for l in labels_list]
        off = (d_i - 1) * bw * 2.5
        ax.bar(x_pos+off-bw/2, pv, bw, label=f'PIGNN {dof}', alpha=0.7)
        ax.bar(x_pos+off+bw/2, kv, bw, label=f'Kratos {dof}',
               alpha=0.7, hatch='//')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"n{ver_nodes[l]}" for l in labels_list], fontsize=8)
    ax.set_title('Displacement comparison')
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3, axis='y')

    # ---- panel 3: loss history ----
    ax = axes[2]
    hist_path = HISTORY_PATH
    if os.path.exists(hist_path):
        h = torch.load(hist_path, weights_only=False)
        epochs = h.get('epoch', [])
        for k, c in [('L_eq','tab:blue'),('L_free','tab:cyan'),
                      ('L_sup','tab:green'),('L_N','tab:orange'),
                      ('L_M','tab:red'),('L_V','tab:purple'),
                      ('total','black')]:
            v = h.get(k,[])
            if v:
                ax.semilogy(epochs, v, color=c, lw=1.5 if k!='total' else 2.5,
                            label=k)
        ax.set_title('Training loss history')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5,0.5,'No history', ha='center', va='center',
                transform=ax.transAxes)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {save_path}")
    try:
        plt.show()
    except Exception:
        pass


# ================================================================
# 9.  MAIN PIPELINE
# ================================================================

def run_verification():
    print("=" * 60)
    print("  PIGNN vs KRATOS — Cross-Verification")
    print("=" * 60)

    # ── data ──
    print("\n── Loading graph dataset ──")
    if not os.path.exists(DATA_PATH):
        print(f"  ✗  Not found: {DATA_PATH}\n"
              f"     Run step1 + step2 first."); return
    data_list = torch.load(DATA_PATH, weights_only=False)
    print(f"  {len(data_list)} graphs loaded (Kratos truth embedded)")
    if CASE_INDEX >= len(data_list):
        print(f"  ✗  case index {CASE_INDEX} out of range"); return
    data = data_list[CASE_INDEX]
    print(f"  Case {CASE_INDEX}: {data.num_nodes} nodes, "
          f"{data.n_elements} elements")

    # ── model ──
    print("\n── Loading trained model ──")
    model = PIGNN(**MODEL_CONFIG)
    ckpt_loaded = False
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, weights_only=False,
                          map_location='cpu')
        model.load_state_dict(ckpt['model_state'])
        ckpt_loaded = True
        print(f"  Checkpoint: {CHECKPOINT_PATH}")
        print(f"  Epoch {ckpt.get('epoch','?')}   "
              f"Losses: {ckpt.get('losses',{})}")
    else:
        print(f"  ⚠  No checkpoint — using random weights")
    print(f"  Parameters: {model.count_params():,}")

    # ── history ──
    print("\n── Loading loss history ──")
    history = None
    if os.path.exists(HISTORY_PATH):
        history = torch.load(HISTORY_PATH, weights_only=False)
        print(f"  {len(history.get('epoch',[]))} epochs recorded")
    else:
        print("  ⚠  No history file")

    # ── predict ──
    print("\n── Running forward pass ──")
    model.eval()
    with torch.no_grad():
        pred = model(data)
    print(f"  pred shape: {pred.shape}  "
          f"(expected ({data.num_nodes}, 15))")
    print(f"  disp  range: [{pred[:,:3].min():.6e}, {pred[:,:3].max():.6e}]")
    print(f"  force range: [{pred[:,3:].min():.6e}, {pred[:,3:].max():.6e}]")

    # ── identify nodes ──
    print("\n── Identifying verification nodes ──")
    ver_nodes = find_verification_nodes(data, MANUAL_NODES)

    # ── properties ──
    props = extract_properties(data)

    # ── losses ──
    loss_sum = extract_loss_summary(history)

    # ── compare at each node ──
    print("\n── Node-level comparison ──")
    node_res = {}
    for label, ni in ver_nodes.items():
        dc = compare_displacement(ni, pred, data)
        fc = compare_forces(ni, pred, data)
        node_res[label] = dict(
            node_idx = ni,
            coords   = data.coords[ni].numpy(),
            displacement     = dc,
            force_comparisons = fc,
        )
        print(f"  {label} (node {ni}):  "
              f"ux_err={dc['error'][0]:.4e}  "
              f"uz_err={dc['error'][1]:.4e}  "
              f"θy_err={dc['error'][2]:.4e}  "
              f"#elems={len(fc)}")

    # ── report ──
    print("\n── Generating report ──")
    report = generate_report(props, loss_sum, node_res, REPORT_PATH)
    print(report)
    print(f"\n  ✓  Report saved: {REPORT_PATH}")

    # ── plot ──
    print("\n── Generating plot ──")
    try:
        plot_verification(data, pred, ver_nodes, props, PLOT_PATH)
    except Exception as exc:
        print(f"  ⚠  Plot failed: {exc}")

    return node_res


# ================================================================
if __name__ == "__main__":
    run_verification()