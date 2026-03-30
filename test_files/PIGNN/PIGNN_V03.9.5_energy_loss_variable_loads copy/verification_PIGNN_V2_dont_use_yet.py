"""
=================================================================
verify_pignn.py — Cross-Verification: PIGNN Predictions vs Kratos
                  v2: + hyperparameters + versioned saving
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
  - ALL training hyperparameters (model, optimizer, loss, scheduler)
  - Initial, best, and final loss values (all 6 terms + total)
  - Material/section properties: E, I, A, UDL, EA, EI

Versioned saving:
  - After each run, option to save as new version
  - Never overwrites previous reports
  - Format: verification_report_v01.txt, _v02.txt, ...

INPUTS (already saved — no Kratos re-run needed):
  DATA/graph_dataset.pt    — raw PyG graphs with Kratos ground truth
  RESULTS/best.pt          — trained model checkpoint
  RESULTS/history.pt       — training loss history

OUTPUT:
  RESULTS/verification_report.txt       (latest, overwritten)
  RESULTS/verification_report_v01.txt   (permanent version)
  RESULTS/verification_plot.png         (latest)
  RESULTS/verification_plot_v01.png     (permanent version)

USAGE:
  python verify_pignn.py
=================================================================
"""

import os
import sys
import glob
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
# CONFIGURATION
# ================================================================

DATA_PATH       = "DATA/graph_dataset.pt"
CHECKPOINT_PATH = "RESULTS/best.pt"
HISTORY_PATH    = "RESULTS/history.pt"
REPORT_DIR      = "RESULTS"
REPORT_BASE     = "verification_report"
PLOT_BASE       = "verification_plot"

CASE_INDEX = 0

# Set to dict to override auto-detection:
#   MANUAL_NODES = {'top_corner': 10, 'mid_top_beam_1': 8, 'mid_top_beam_2': 6}
MANUAL_NODES = None

# Must match architecture used during training
MODEL_CONFIG = dict(
    node_in_dim = 9,
    edge_in_dim = 10,
    hidden_dim  = 128,
    n_layers    = 6,
)

# ── Replicate ALL training hyperparameters here ──
# (should match train.py TrainConfig exactly)
TRAIN_HYPERPARAMS = dict(
    # ── Model architecture ──
    node_in_dim     = 9,
    edge_in_dim     = 10,
    hidden_dim      = 128,
    n_layers        = 6,
    output_dim      = 15,
    activation      = "SiLU",
    aggregation     = "add",
    normalization   = "LayerNorm",
    residual        = True,
    decoder_layers  = "[128, 64, 15]",

    # ── Optimizer ──
    optimizer       = "Adam",
    learning_rate   = 1e-3,
    weight_decay    = 1e-5,
    grad_clip_norm  = 10.0,

    # ── Scheduler ──
    scheduler       = "StepLR",
    scheduler_step  = 100,
    scheduler_gamma = 0.5,

    # ── Training ──
    epochs          = 500,
    batch_size      = "full dataset (no batching)",
    data_normalization = "None (raw physical units for physics loss)",
    train_val_split = "all graphs used for training",

    # ── Loss function ──
    loss_type       = "NaivePhysicsLoss (autograd)",
    w_eq            = 1.0,
    w_free          = 1.0,
    w_sup           = 1.0,
    w_N             = 1.0,
    w_M             = 1.0,
    w_V             = 1.0,

    # ── Loss terms description ──
    L_eq_desc       = "Nodal equilibrium: Σ face_forces = F_ext",
    L_free_desc     = "Free face forces = 0 (soft backup to hard mask)",
    L_sup_desc      = "Support displacements = 0 (soft backup to hard BC)",
    L_N_desc        = "Axial: N = EA · ∂u_L/∂s  (1st derivative)",
    L_M_desc        = "Moment: M = EI · ∂²w_L/∂s² (2nd derivative)",
    L_V_desc        = "Shear: V = EI · ∂³w_L/∂s³ (3rd derivative)",

    # ── Hard constraints ──
    hard_bc_disp    = "ux=uz=0 at support nodes",
    hard_bc_rot     = "θy=0 at fixed support nodes",
    hard_face_mask  = "forces=0 at unconnected faces",

    # ── Coordinate injection ──
    coord_injection = "data.coords → requires_grad_(True) → replaces x[:,0:3]",

    # ── Physics ──
    beam_theory     = "Euler-Bernoulli (2D, XZ plane)",
    sign_convention = "A-end: Fx=-N, Fz=-V, My=-M | B-end: Fx=+N, Fz=+V, My=+M",
    face_convention = "0=+x, 1=-x, 2=+z, 3=-z",
    local_transform = "T(α) rotation: cos α = dx, sin α = dz",

    # ── Data ──
    node_features   = "[x,y,z, bc_disp, bc_rot, wl_x,wl_y,wl_z, response_flag]",
    edge_features   = "[L, dx,dy,dz, E, A, I22, qx,qy,qz]",
    node_targets    = "[ux, uz, θy]",
    elem_targets    = "[N, M, V, dBM/dI22]",
)

FACE_NAMES = ['+x', '-x', '+z', '-z']


# ================================================================
# 1.  VERSION MANAGEMENT
# ================================================================

def get_next_version(base_dir, base_name, extension=".txt"):
    """
    Find next available version number.
    Scans for existing files: base_name_v01.ext, _v02.ext, ...
    Returns (next_version_int, next_filepath).
    """
    existing = glob.glob(
        os.path.join(base_dir, f"{base_name}_v[0-9][0-9]*{extension}")
    )
    if not existing:
        return 1, os.path.join(base_dir, f"{base_name}_v01{extension}")

    versions = []
    for f in existing:
        fname = os.path.basename(f)
        # extract version number from filename
        try:
            part = fname.replace(base_name + "_v", "").replace(extension, "")
            versions.append(int(part))
        except ValueError:
            continue

    next_v = max(versions) + 1 if versions else 1
    path = os.path.join(base_dir, f"{base_name}_v{next_v:02d}{extension}")
    return next_v, path


def list_existing_versions(base_dir, base_name, extension=".txt"):
    """List all existing version files."""
    pattern = os.path.join(base_dir, f"{base_name}_v[0-9][0-9]*{extension}")
    files = sorted(glob.glob(pattern))
    return files


def prompt_save_version(report_text, plot_path_latest):
    """
    After run completes, ask user whether to save a versioned copy.
    Options:
      y  — save new version
      n  — skip (only latest files exist)
      q  — quit
      l  — list existing versions
    """
    print(f"\n{'═'*60}")
    print(f"  VERSION SAVE OPTIONS")
    print(f"{'═'*60}")

    existing_reports = list_existing_versions(REPORT_DIR, REPORT_BASE, ".txt")
    existing_plots   = list_existing_versions(REPORT_DIR, PLOT_BASE,   ".png")

    if existing_reports:
        print(f"  Existing report versions:")
        for f in existing_reports:
            size = os.path.getsize(f) / 1024
            mtime = datetime.fromtimestamp(os.path.getmtime(f))
            print(f"    {os.path.basename(f):40s} "
                  f"{size:6.1f} KB   {mtime:%Y-%m-%d %H:%M}")
    else:
        print(f"  No previous versions found.")

    while True:
        print(f"\n  Options:")
        print(f"    [y] Save as new version (never overwrites)")
        print(f"    [n] Skip (keep only latest)")
        print(f"    [l] List existing versions")
        print(f"    [q] Quit")

        try:
            choice = input("\n  Your choice [y/n/l/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Skipping save.")
            return None

        if choice == 'q':
            print("  Done.")
            return None

        elif choice == 'l':
            all_files = list_existing_versions(REPORT_DIR, REPORT_BASE, ".txt") + \
                        list_existing_versions(REPORT_DIR, PLOT_BASE, ".png")
            if all_files:
                print(f"\n  All versioned files:")
                for f in sorted(all_files):
                    size = os.path.getsize(f) / 1024
                    print(f"    {os.path.basename(f):45s}  {size:6.1f} KB")
            else:
                print("  No versioned files yet.")

        elif choice == 'n':
            print("  Skipped. Only latest files saved.")
            return None

        elif choice == 'y':
            # ── Save report version ──
            v_num, v_report_path = get_next_version(
                REPORT_DIR, REPORT_BASE, ".txt"
            )
            os.makedirs(REPORT_DIR, exist_ok=True)
            with open(v_report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"  ✓ Report saved: {v_report_path}")

            # ── Save plot version ──
            if plot_path_latest and os.path.exists(plot_path_latest):
                _, v_plot_path = get_next_version(
                    REPORT_DIR, PLOT_BASE, ".png"
                )
                import shutil
                shutil.copy2(plot_path_latest, v_plot_path)
                print(f"  ✓ Plot   saved: {v_plot_path}")

            # ── Save checkpoint version ──
            if os.path.exists(CHECKPOINT_PATH):
                _, v_ckpt_path = get_next_version(
                    REPORT_DIR, "checkpoint", ".pt"
                )
                import shutil
                shutil.copy2(CHECKPOINT_PATH, v_ckpt_path)
                print(f"  ✓ Checkpoint saved: {v_ckpt_path}")

            print(f"\n  Version {v_num:02d} saved successfully.")
            print(f"  Files will never be overwritten.")
            return v_num

        else:
            print(f"  Invalid choice '{choice}'. Try y/n/l/q.")


# ================================================================
# 2.  NODE IDENTIFICATION
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

    # --- column junctions ---
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
        for j in junctions:
            idx_j = np.where(top_sorted == j)[0]
            if len(idx_j):
                dists[idx_j[0]] = 1e10
        best = top_sorted[np.argmin(dists)]
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
# 3.  ELEMENT CONNECTIVITY AT A NODE
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
# 4.  DISPLACEMENT COMPARISON
# ================================================================

def compare_displacement(node_idx, pred, data):
    p = pred[node_idx, 0:3].cpu().numpy()
    k = data.y_node[node_idx].cpu().numpy()
    err = p - k
    rel = np.abs(err) / np.where(np.abs(k) > 1e-15, np.abs(k), 1.0)
    return dict(pignn=p, kratos=k, error=err, rel_error=rel)


# ================================================================
# 5.  FORCE / MOMENT COMPARISON
# ================================================================

def compare_forces(node_idx, pred, data):
    """
    For every element connected to *node_idx*:
      1. grab PIGNN face force (global)
      2. rotate to element-local frame
      3. apply sign convention → internal N, V, M
      4. compare with Kratos y_element
    """
    connected   = get_connected_elements(node_idx, data)
    face_forces = pred[node_idx, 3:15].reshape(4, 3).cpu().numpy()

    results = []
    for info in connected:
        e    = info['elem_idx']
        is_A = info['is_A_end']
        f    = info['face_idx']

        kN = data.y_element[e, 0].item()
        kM = data.y_element[e, 1].item()
        kV = data.y_element[e, 2].item()

        ff_g = face_forces[f]

        d   = data.elem_directions[e].numpy()
        c_a = d[0];  s_a = d[2]
        ff_l = np.array([
             ff_g[0]*c_a + ff_g[1]*s_a,
            -ff_g[0]*s_a + ff_g[1]*c_a,
             ff_g[2],
        ])

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
# 6.  PROPERTY EXTRACTION
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
        case_id          = int(data.case_id)           if hasattr(data,'case_id')           else -1,
        nearest_node_id  = int(data.nearest_node_id)   if hasattr(data,'nearest_node_id')   else -1,
        traced_element_id= int(data.traced_element_id) if hasattr(data,'traced_element_id') else -1,
    )


# ================================================================
# 7.  EXTRACT TRAINING HYPERPARAMETERS FROM CHECKPOINT
# ================================================================

def extract_checkpoint_info(ckpt_path):
    """
    Extract whatever information is stored in the checkpoint.
    Returns dict with epoch, losses, optimizer state summary.
    """
    info = dict(
        checkpoint_file = os.path.basename(ckpt_path),
        checkpoint_exists = False,
    )

    if not os.path.exists(ckpt_path):
        return info

    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    info['checkpoint_exists'] = True
    info['epoch'] = ckpt.get('epoch', '?')
    info['losses'] = ckpt.get('losses', {})

    # Optimizer state → extract LR if available
    opt_state = ckpt.get('optimizer_state', {})
    if 'param_groups' in opt_state:
        pg = opt_state['param_groups']
        if pg:
            info['final_lr'] = pg[0].get('lr', '?')
            info['final_weight_decay'] = pg[0].get('weight_decay', '?')
            info['final_betas'] = pg[0].get('betas', '?')
            info['final_eps'] = pg[0].get('eps', '?')

    # File metadata
    info['file_size_KB'] = os.path.getsize(ckpt_path) / 1024
    info['file_modified'] = datetime.fromtimestamp(
        os.path.getmtime(ckpt_path)
    ).strftime('%Y-%m-%d %H:%M:%S')

    return info


# ================================================================
# 8.  LOSS SUMMARY
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

    # Convergence ratios
    for k in ['L_eq','L_free','L_sup','L_N','L_M','L_V','total']:
        v = history.get(k, [])
        if len(v) >= 20:
            first = np.mean(v[:10])
            last  = np.mean(v[-10:])
            ratio = last / max(first, 1e-30)
            if   ratio < 0.01:  trend = "CONVERGED (>100x)"
            elif ratio < 0.1:   trend = "PARTIAL (10-100x)"
            elif ratio < 1.0:   trend = f"SLOW ({1/max(ratio,1e-30):.1f}x)"
            else:               trend = "NOT CONVERGING"
            out[f'{k}_trend'] = trend
            out[f'{k}_ratio'] = ratio

    return out


# ================================================================
# 9.  REPORT GENERATION
# ================================================================

def generate_report(props, loss_sum, node_res, ckpt_info, path):
    L = []
    def add(s=""):
        L.append(s)
    W = 85

    add("=" * W)
    add("  PIGNN vs KRATOS — CROSS-VERIFICATION REPORT")
    add(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add("=" * W)

    # ══════════════════════════════════════════════════════
    # SECTION 1: TRAINING HYPERPARAMETERS (COMPLETE)
    # ══════════════════════════════════════════════════════
    add("\n" + "─"*W)
    add("  SECTION 1: TRAINING HYPERPARAMETERS (COMPLETE)")
    add("─"*W)

    add(f"\n  ┌─── Model Architecture ────────────────────────────────────┐")
    for k in ['node_in_dim','edge_in_dim','hidden_dim','n_layers',
              'output_dim','activation','aggregation','normalization',
              'residual','decoder_layers']:
        add(f"  │  {k:<25} {str(TRAIN_HYPERPARAMS.get(k,'-')):<35}│")
    n_params = sum(p.numel() for p in PIGNN(**MODEL_CONFIG).parameters())
    add(f"  │  {'total_parameters':<25} {n_params:<35,}│")
    add(f"  └──────────────────────────────────────────────────────────┘")

    add(f"\n  ┌─── Optimizer ──────────────────────────────────────────────┐")
    for k in ['optimizer','learning_rate','weight_decay','grad_clip_norm']:
        add(f"  │  {k:<25} {str(TRAIN_HYPERPARAMS.get(k,'-')):<35}│")
    if ckpt_info.get('checkpoint_exists'):
        add(f"  │  {'final_lr (from ckpt)':<25} "
            f"{str(ckpt_info.get('final_lr','-')):<35}│")
        add(f"  │  {'final_betas':<25} "
            f"{str(ckpt_info.get('final_betas','-')):<35}│")
    add(f"  └──────────────────────────────────────────────────────────┘")

    add(f"\n  ┌─── Scheduler ──────────────────────────────────────────────┐")
    for k in ['scheduler','scheduler_step','scheduler_gamma']:
        add(f"  │  {k:<25} {str(TRAIN_HYPERPARAMS.get(k,'-')):<35}│")
    add(f"  └──────────────────────────────────────────────────────────┘")

    add(f"\n  ┌─── Training ──────────────────────────────────────────────┐")
    for k in ['epochs','batch_size','data_normalization','train_val_split']:
        add(f"  │  {k:<25} {str(TRAIN_HYPERPARAMS.get(k,'-')):<35}│")
    add(f"  └──────────────────────────────────────────────────────────┘")

    add(f"\n  ┌─── Loss Weights ──────────────────────────────────────────┐")
    for k in ['loss_type','w_eq','w_free','w_sup','w_N','w_M','w_V']:
        add(f"  │  {k:<25} {str(TRAIN_HYPERPARAMS.get(k,'-')):<35}│")
    add(f"  └──────────────────────────────────────────────────────────┘")

    add(f"\n  ┌─── Loss Term Descriptions ───────────────────────────────┐")
    for k in ['L_eq_desc','L_free_desc','L_sup_desc',
              'L_N_desc','L_M_desc','L_V_desc']:
        add(f"  │  {TRAIN_HYPERPARAMS.get(k,'-'):<61}│")
    add(f"  └──────────────────────────────────────────────────────────┘")

    add(f"\n  ┌─── Hard Constraints ──────────────────────────────────────┐")
    for k in ['hard_bc_disp','hard_bc_rot','hard_face_mask']:
        add(f"  │  {k:<18} {str(TRAIN_HYPERPARAMS.get(k,'-')):<42}│")
    add(f"  └──────────────────────────────────────────────────────────┘")

    add(f"\n  ┌─── Physics Settings ──────────────────────────────────────┐")
    for k in ['beam_theory','sign_convention','face_convention',
              'local_transform','coord_injection']:
        v = str(TRAIN_HYPERPARAMS.get(k, '-'))
        # wrap long values
        while len(v) > 58:
            add(f"  │  {k:<18} {v[:58]}│")
            v = v[58:]
            k = ''
        add(f"  │  {k:<18} {v:<42}│")
    add(f"  └──────────────────────────────────────────────────────────┘")

    add(f"\n  ┌─── Feature Dimensions ───────────────────────────────────┐")
    for k in ['node_features','edge_features','node_targets','elem_targets']:
        add(f"  │  {k:<18} {str(TRAIN_HYPERPARAMS.get(k,'-')):<42}│")
    add(f"  └──────────────────────────────────────────────────────────┘")

    add(f"\n  ┌─── Checkpoint Info ───────────────────────────────────────┐")
    add(f"  │  {'file':<25} {ckpt_info.get('checkpoint_file','-'):<35}│")
    add(f"  │  {'exists':<25} {str(ckpt_info.get('checkpoint_exists')):<35}│")
    add(f"  │  {'epoch':<25} {str(ckpt_info.get('epoch','-')):<35}│")
    add(f"  │  {'file_size':<25} "
        f"{ckpt_info.get('file_size_KB',0):.1f} KB{'':<28}│")
    add(f"  │  {'modified':<25} "
        f"{ckpt_info.get('file_modified','-'):<35}│")
    add(f"  └──────────────────────────────────────────────────────────┘")

    # ══════════════════════════════════════════════════════
    # SECTION 2: CASE PROPERTIES
    # ══════════════════════════════════════════════════════
    add("\n" + "─"*W)
    add("  SECTION 2: CASE PROPERTIES")
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
    add(f"  Material & Section Properties:")
    add(f"  {'Property':<16} {'Min':>14}  {'Max':>14}  {'Unit':<6}")
    add(f"  {'─'*56}")
    for tag, arr, u in [
        ('E (Young)',     props['E'],  'Pa'),
        ('A (Area)',      props['A'],  'm²'),
        ('I22 (Inertia)', props['I22'],'m⁴'),
        ('EA',            props['EA'], 'N'),
        ('EI',            props['EI'], 'N·m²'),
    ]:
        add(f"  {tag:<16} {arr.min():>14.4e}  {arr.max():>14.4e}  {u}")
    add(f"")
    add(f"  Applied UDL (global):  qx={props['UDL'][0]:.4f}  "
        f"qy={props['UDL'][1]:.4f}  qz={props['UDL'][2]:.4f}  N/m")

    # ══════════════════════════════════════════════════════
    # SECTION 3: TRAINING LOSS SUMMARY
    # ══════════════════════════════════════════════════════
    add("\n" + "─"*W)
    add("  SECTION 3: TRAINING LOSS SUMMARY")
    add("─"*W)
    if loss_sum:
        add(f"  Total epochs: {loss_sum['total_epochs']}   "
            f"Best epoch: {loss_sum['best_epoch']}")
        add(f"")
        add(f"  {'Loss':<10} {'Initial':>14} {'Best':>14} "
            f"{'Final':>14} {'Trend':<22}")
        add(f"  {'─'*78}")
        for k in ['L_eq','L_free','L_sup','L_N','L_M','L_V','total']:
            if k in loss_sum:
                v = loss_sum[k]
                trend = loss_sum.get(f'{k}_trend', '')
                add(f"  {k:<10} {v['initial']:>14.6e} {v['best']:>14.6e} "
                    f"{v['final']:>14.6e} {trend:<22}")

        if 'disp_error' in loss_sum:
            v = loss_sum['disp_error']
            add(f"")
            add(f"  Displacement error vs Kratos (relative L2):")
            add(f"    Initial {v['initial']:.6e}  "
                f"Best {v['best']:.6e}  Final {v['final']:.6e}")

        # Convergence classification
        add(f"")
        add(f"  Convergence Classification:")
        add(f"  {'─'*60}")
        for k in ['L_eq','L_free','L_sup','L_N','L_M','L_V']:
            trend = loss_sum.get(f'{k}_trend', 'unknown')
            ratio = loss_sum.get(f'{k}_ratio', 1.0)
            if 'CONVERGED' in trend:   icon = '✅'
            elif 'PARTIAL' in trend:   icon = '⚠️ '
            elif 'SLOW' in trend:      icon = '⚠️ '
            else:                      icon = '❌'
            add(f"    {icon} {k:<10} {trend:<25} "
                f"(ratio: {ratio:.4e})")
    else:
        add("  (no training history found)")

    # ══════════════════════════════════════════════════════
    # SECTION 4: NODE-LEVEL COMPARISON
    # ══════════════════════════════════════════════════════
    add("\n" + "─"*W)
    add("  SECTION 4: NODE-LEVEL COMPARISON  (PIGNN vs Kratos)")
    add("─"*W)
    add("  Note: PIGNN forces are element-end face forces;")
    add("        Kratos values are element-centroid. Minor mismatch expected.")

    for label, nd in node_res.items():
        c = nd['coords']
        add(f"\n  ╔{'═'*68}╗")
        add(f"  ║  {label:<26} node {nd['node_idx']:<5}"
            f"  ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})        ║")
        add(f"  ╚{'═'*68}╝")

        # displacements
        d = nd['displacement']
        add(f"    Displacements:")
        add(f"    {'DOF':<6} {'PIGNN':>14} {'Kratos':>14} "
            f"{'Abs Err':>14} {'Rel Err':>10} {'Unit':<4}")
        add(f"    {'─'*66}")
        for i, (nm, un) in enumerate(
                [('ux','m'), ('uz','m'), ('θy','rad')]):
            add(f"    {nm:<6} {d['pignn'][i]:>14.6e} "
                f"{d['kratos'][i]:>14.6e} "
                f"{d['error'][i]:>14.6e} "
                f"{d['rel_error'][i]:>10.4f} {un}")

        # forces for each connected element
        for fc in nd['force_comparisons']:
            e = fc['elem_idx']
            add(f"")
            add(f"    Element {e} ({fc['end_label']}, face {fc['face_name']})"
                f"  L={fc['length']:.4f}m")
            add(f"      Properties: E={fc['E']:.4e} Pa  "
                f"A={fc['A']:.4e} m²  I22={fc['I22']:.4e} m⁴")
            add(f"      EA={fc['E']*fc['A']:.4e} N   "
                f"EI={fc['E']*fc['I22']:.4e} N·m²")
            add(f"      dir=({fc['direction'][0]:.4f}, "
                f"{fc['direction'][1]:.4f}, "
                f"{fc['direction'][2]:.4f})  "
                f"other_node={fc['other_node']}")
            add(f"      face_global: Fx={fc['ff_global'][0]:+.4e}  "
                f"Fz={fc['ff_global'][1]:+.4e}  "
                f"My={fc['ff_global'][2]:+.4e}")
            add(f"      face_local:  Fx={fc['ff_local'][0]:+.4e}  "
                f"Fz={fc['ff_local'][1]:+.4e}  "
                f"My={fc['ff_local'][2]:+.4e}")
            add(f"")
            add(f"      Internal Forces (from sign convention):")
            add(f"      {'Qty':<6} {'PIGNN':>14} {'Kratos':>14} "
                f"{'Abs Err':>14} {'Rel':>10} {'Unit':<4}")
            add(f"      {'─'*66}")
            for q, u in [('N','N'), ('V','N'), ('M','N·m')]:
                add(f"      {q:<6} {fc['pignn'][q]:>14.6e} "
                    f"{fc['kratos'][q]:>14.6e} "
                    f"{fc['error'][q]:>14.6e} "
                    f"{fc['rel_error'][q]:>10.4f} {u}")

    # ══════════════════════════════════════════════════════
    # SECTION 5: OVERALL ACCURACY SUMMARY
    # ══════════════════════════════════════════════════════
    add("\n" + "─"*W)
    add("  SECTION 5: OVERALL ACCURACY SUMMARY")
    add("─"*W)

    all_dr, all_fr = [], []
    for nd in node_res.values():
        all_dr.extend(nd['displacement']['rel_error'].tolist())
        for fc in nd['force_comparisons']:
            for q in ('N','V','M'):
                all_fr.append(fc['rel_error'][q])

    if all_dr:
        add(f"  Displacement relative errors at verification nodes:")
        add(f"    Mean:  {np.mean(all_dr):.6e}")
        add(f"    Max:   {np.max(all_dr):.6e}")
        add(f"    Min:   {np.min(all_dr):.6e}")
    if all_fr:
        add(f"")
        add(f"  Force/moment relative errors at verification nodes:")
        add(f"    Mean:  {np.mean(all_fr):.6e}")
        add(f"    Max:   {np.max(all_fr):.6e}")
        add(f"    Min:   {np.min(all_fr):.6e}")

    avg_d = np.mean(all_dr) if all_dr else 1.0
    avg_f = np.mean(all_fr) if all_fr else 1.0

    add(f"")
    add(f"  VERDICT:")
    if   avg_d < 0.01:  add("    ✅  Displacement: EXCELLENT (< 1 %)")
    elif avg_d < 0.05:  add("    ⚠️   Displacement: GOOD (< 5 %)")
    elif avg_d < 0.20:  add("    ⚠️   Displacement: MODERATE (< 20 %)")
    else:               add(f"    ❌  Displacement: POOR ({avg_d*100:.0f} %)")

    if   avg_f < 0.05:  add("    ✅  Forces/Moments: EXCELLENT (< 5 %)")
    elif avg_f < 0.15:  add("    ⚠️   Forces/Moments: GOOD (< 15 %)")
    elif avg_f < 0.30:  add("    ⚠️   Forces/Moments: MODERATE (< 30 %)")
    else:               add(f"    ❌  Forces/Moments: POOR ({avg_f*100:.0f} %)")

    if avg_d > 0.5:
        add(f"")
        add(f"    Note: Poor accuracy expected for naive-autograd physics-only")
        add(f"    training. Known issues with L_M (2nd deriv) and L_V (3rd deriv)")
        add(f"    through GNN message passing. See physics_loss.py docstrings.")

    add("\n" + "="*W)
    add("  END OF REPORT")
    add("="*W)

    report = "\n".join(L)

    # ── Save latest (always overwrite) ──
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)

    return report


# ================================================================
# 10.  VISUALISATION
# ================================================================

def plot_verification(data, pred, ver_nodes, props, save_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("  ⚠  matplotlib not found — skipping plot")
        return

    coords = data.coords.numpy()
    conn   = data.connectivity.numpy()
    E_num  = data.n_elements

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # ──── Panel 1: Frame with verification nodes ────
    ax = axes[0, 0]
    for e in range(E_num):
        n1, n2 = conn[e]
        xs = [coords[n1,0], coords[n2,0]]
        zs = [coords[n1,2], coords[n2,2]]
        dx = abs(coords[n2,0] - coords[n1,0])
        dz = abs(coords[n2,2] - coords[n1,2])
        ax.plot(xs, zs, '-',
                color='steelblue' if dz > dx else 'coral', lw=2.5)
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
    for idx, (lbl, ni) in enumerate(ver_nodes.items()):
        ax.plot(coords[ni,0], coords[ni,2], '*', ms=18,
                color=vc[idx % 3], mec='black', zorder=10)
        ax.text(coords[ni,0]+0.25, coords[ni,2]+0.25, lbl,
                fontsize=7, color=vc[idx % 3], fontweight='bold')
    ax.set_title(f"Frame (case {props['case_id']})\n"
                 f"{props['n_nodes']} nodes, {props['n_elements']} elems",
                 fontsize=10)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Z [m]')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # ──── Panel 2: Displacement bar chart ────
    ax = axes[0, 1]
    labels_list = list(ver_nodes.keys())
    dof_names = ['ux', 'uz', 'θy']
    x_pos = np.arange(len(labels_list))
    bw = 0.12
    colors_p = ['#2196F3', '#4CAF50', '#FF9800']
    colors_k = ['#1565C0', '#2E7D32', '#E65100']
    for d_i, dof in enumerate(dof_names):
        pv = [pred[ver_nodes[l], d_i].item() for l in labels_list]
        kv = [data.y_node[ver_nodes[l], d_i].item() for l in labels_list]
        off = (d_i - 1) * bw * 2.5
        ax.bar(x_pos + off - bw/2, pv, bw,
               color=colors_p[d_i], alpha=0.8,
               label=f'PIGNN {dof}')
        ax.bar(x_pos + off + bw/2, kv, bw,
               color=colors_k[d_i], alpha=0.5, hatch='//',
               label=f'Kratos {dof}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"n{ver_nodes[l]}\n{l[:12]}"
                        for l in labels_list], fontsize=7)
    ax.set_title('Displacement Comparison\n(PIGNN vs Kratos)', fontsize=10)
    ax.set_ylabel('Value')
    ax.legend(fontsize=6, ncol=2, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    # ──── Panel 3: Loss history ────
    ax = axes[1, 0]
    if os.path.exists(HISTORY_PATH):
        h = torch.load(HISTORY_PATH, weights_only=False)
        epochs = h.get('epoch', [])
        for k, c, ls in [
            ('L_eq',   'tab:blue',   '-'),
            ('L_free', 'tab:cyan',   '--'),
            ('L_sup',  'tab:green',  '--'),
            ('L_N',    'tab:orange', '-'),
            ('L_M',    'tab:red',    '-'),
            ('L_V',    'tab:purple', '-'),
            ('total',  'black',      '-'),
        ]:
            v = h.get(k, [])
            if v:
                lw = 2.5 if k == 'total' else 1.5
                ax.semilogy(epochs, v, color=c, lw=lw,
                            linestyle=ls, label=k)
        ax.set_title('Training Loss History (log scale)', fontsize=10)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No history file found',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')

    # ──── Panel 4: Force/moment comparison ────
    ax = axes[1, 1]
    bar_data = []
    bar_labels = []
    for label, ni in ver_nodes.items():
        short = label[:10]
        fcs = compare_forces(ni, pred, data)
        for fc in fcs[:2]:   # max 2 elements per node for readability
            for q in ['N', 'V', 'M']:
                bar_data.append((
                    f"n{ni} e{fc['elem_idx']} {q}",
                    fc['pignn'][q],
                    fc['kratos'][q],
                ))
    if bar_data:
        xlabels = [b[0] for b in bar_data]
        pvals   = [b[1] for b in bar_data]
        kvals   = [b[2] for b in bar_data]
        x_b = np.arange(len(bar_data))
        ax.bar(x_b - 0.15, pvals, 0.3, color='#2196F3',
               alpha=0.8, label='PIGNN')
        ax.bar(x_b + 0.15, kvals, 0.3, color='#E65100',
               alpha=0.5, hatch='//', label='Kratos')
        ax.set_xticks(x_b)
        ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=6)
        ax.set_title('Force/Moment Comparison', fontsize=10)
        ax.set_ylabel('Value [N or N·m]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('PIGNN vs Kratos — Cross-Verification',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {save_path}")
    try:
        plt.show()
    except Exception:
        pass


# ================================================================
# 11.  MAIN PIPELINE
# ================================================================

def run_verification():
    print("=" * 70)
    print("  PIGNN vs KRATOS — Cross-Verification (with hyperparams + versioning)")
    print("=" * 70)

    # ── data ──
    print("\n── Loading graph dataset ──")
    if not os.path.exists(DATA_PATH):
        print(f"  ✗  Not found: {DATA_PATH}")
        print(f"     Run step1 + step2 first.")
        return
    data_list = torch.load(DATA_PATH, weights_only=False)
    print(f"  {len(data_list)} graphs loaded (Kratos truth embedded)")
    if CASE_INDEX >= len(data_list):
        print(f"  ✗  case index {CASE_INDEX} out of range")
        return
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
        print(f"  ⚠  No checkpoint found at {CHECKPOINT_PATH}")
        print(f"     Using randomly initialized weights")
    print(f"  Parameters: {model.count_params():,}")

    # ── checkpoint info ──
    ckpt_info = extract_checkpoint_info(CHECKPOINT_PATH)

    # ── history ──
    print("\n── Loading loss history ──")
    history = None
    if os.path.exists(HISTORY_PATH):
        history = torch.load(HISTORY_PATH, weights_only=False)
        n_ep = len(history.get('epoch', []))
        print(f"  {n_ep} epochs recorded")
        # Quick summary
        if 'total' in history and history['total']:
            print(f"    Initial total: {history['total'][0]:.6e}")
            print(f"    Final total:   {history['total'][-1]:.6e}")
            print(f"    Best total:    {min(history['total']):.6e}")
    else:
        print("  ⚠  No history file found")

    # ── predict ──
    print("\n── Running forward pass ──")
    model.eval()
    with torch.no_grad():
        pred = model(data)
    print(f"  pred shape: {pred.shape}  "
          f"(expected ({data.num_nodes}, 15))")
    print(f"  disp  range: [{pred[:,:3].min():.6e}, "
          f"{pred[:,:3].max():.6e}]")
    print(f"  force range: [{pred[:,3:].min():.6e}, "
          f"{pred[:,3:].max():.6e}]")

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
            node_idx          = ni,
            coords            = data.coords[ni].numpy(),
            displacement      = dc,
            force_comparisons = fc,
        )
        print(f"  {label} (node {ni}):  "
              f"ux_err={dc['error'][0]:.4e}  "
              f"uz_err={dc['error'][1]:.4e}  "
              f"θy_err={dc['error'][2]:.4e}  "
              f"#connected_elems={len(fc)}")

    # ── generate report ──
    print("\n── Generating report ──")
    latest_report_path = os.path.join(REPORT_DIR,
                                       f"{REPORT_BASE}.txt")
    report = generate_report(
        props, loss_sum, node_res, ckpt_info, latest_report_path
    )

    # ── print report ──
    print("\n" + report)
    print(f"\n  ✓  Latest report saved: {latest_report_path}")

    # ── generate plot ──
    print("\n── Generating plot ──")
    latest_plot_path = os.path.join(REPORT_DIR, f"{PLOT_BASE}.png")
    try:
        plot_verification(data, pred, ver_nodes, props, latest_plot_path)
    except Exception as exc:
        print(f"  ⚠  Plot failed: {exc}")
        latest_plot_path = None

    # ── version save prompt ──
    prompt_save_version(report, latest_plot_path)

    print(f"\n{'='*70}")
    print(f"  VERIFICATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Files:")
    print(f"    Latest report: {latest_report_path}")
    if latest_plot_path:
        print(f"    Latest plot:   {latest_plot_path}")
    print(f"    Versioned:     {REPORT_DIR}/{REPORT_BASE}_vXX.txt")
    print(f"{'='*70}")

    return node_res


# ================================================================
if __name__ == "__main__":
    run_verification()