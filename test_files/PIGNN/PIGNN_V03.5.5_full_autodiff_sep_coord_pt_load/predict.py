"""
predict.py — Visualize PIGNN Predictions vs Ground Truth
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from model import PIGNN


def load_trained_model(checkpoint_path, data_sample, device='cpu'):
    """Load trained model from checkpoint."""
    model = PIGNN(
        node_in_dim=9,
        edge_in_dim=10,
        hidden_dim=128,
        n_layers=6,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f"  Loaded checkpoint: epoch {ckpt['epoch']}")
    print(f"  Training losses: {ckpt['losses']}")
    return model


def predict_case(model, data, device='cpu'):
    """Run model on a single graph, return predictions."""
    model.eval()
    data = data.to(device)

    # Need train mode for coords.requires_grad_ to work properly
    # but we don't need it for pure prediction
    with torch.no_grad():
        # Manual forward without autograd
        h = model.node_encoder(data.x)
        e = model.edge_encoder(data.edge_attr)
        for mp in model.mp_layers:
            h = h + mp(h, data.edge_index, e)

        coords = data.coords.clone()
        decoder_input = torch.cat([h, coords], dim=-1)
        raw = model.local_decoder(decoder_input)

        # Output scaling
        pred = raw.clone()
        pred[:, 0] = raw[:, 0] * data.u_c
        pred[:, 1] = raw[:, 1] * data.u_c
        pred[:, 2] = raw[:, 2] * data.theta_c
        for face in range(4):
            base = 3 + face * 3
            pred[:, base + 0] = raw[:, base + 0] * data.F_c
            pred[:, base + 1] = raw[:, base + 1] * data.F_c
            pred[:, base + 2] = raw[:, base + 2] * data.M_c

        # Hard constraints
        disp_mask = (1.0 - data.bc_disp)
        rot_mask = (1.0 - data.bc_rot)
        pred[:, 0:2] *= disp_mask
        pred[:, 2:3] *= rot_mask
        force_mask = data.face_mask.repeat_interleave(3, dim=1)
        pred[:, 3:15] *= force_mask

    return pred.cpu(), raw.cpu()


def print_prediction_summary(pred, raw, data, case_idx=0):
    """Print numerical comparison of predictions vs ground truth."""
    print(f"\n{'='*70}")
    print(f"  PREDICTION SUMMARY — Case {case_idx}")
    print(f"{'='*70}")

    true_disp = data.y_node  # (N, 3): ux, uz, θy
    pred_disp = pred[:, 0:3]
    pred_faces = pred[:, 3:15].reshape(-1, 4, 3)

    # ── Displacement comparison ──
    print(f"\n  DISPLACEMENTS (pred vs true):")
    print(f"  {'':>8} {'Pred min':>12} {'Pred max':>12} "
          f"{'True min':>12} {'True max':>12} {'Ratio max':>12}")
    print(f"  {'-'*72}")

    labels = ['ux (m)', 'uz (m)', 'θy (rad)']
    for i, label in enumerate(labels):
        p_min = pred_disp[:, i].min().item()
        p_max = pred_disp[:, i].max().item()
        t_min = true_disp[:, i].min().item()
        t_max = true_disp[:, i].max().item()
        t_absmax = max(abs(t_min), abs(t_max), 1e-30)
        p_absmax = max(abs(p_min), abs(p_max))
        ratio = p_absmax / t_absmax
        print(f"  {label:>8} {p_min:>12.4e} {p_max:>12.4e} "
              f"{t_min:>12.4e} {t_max:>12.4e} {ratio:>12.4f}")

    # ── Raw network output ──
    print(f"\n  RAW NETWORK OUTPUT (before scaling):")
    print(f"  {'Channel':>12} {'Min':>12} {'Max':>12} {'Std':>12}")
    print(f"  {'-'*52}")
    channel_names = ['raw_ux', 'raw_uz', 'raw_θy'] + \
                    [f'face{f}_{c}' for f in range(4) for c in ['Fx','Fz','My']]
    for i, name in enumerate(channel_names):
        vals = raw[:, i]
        print(f"  {name:>12} {vals.min().item():>12.4f} "
              f"{vals.max().item():>12.4f} {vals.std().item():>12.4f}")

    # ── Physics scales ──
    print(f"\n  PHYSICS SCALES:")
    print(f"    u_c     = {data.u_c.item():.4e} m")
    print(f"    theta_c = {data.theta_c.item():.4e} rad")
    print(f"    F_c     = {data.F_c.item():.4e} N")
    print(f"    M_c     = {data.M_c.item():.4e} N·m")

    # ── Error at specific nodes ──
    print(f"\n  NODE-LEVEL ERRORS (first 10 + response + supports):")
    print(f"  {'Node':>6} {'pred_ux':>12} {'true_ux':>12} "
          f"{'pred_uz':>12} {'true_uz':>12} {'|err_uz|':>12} {'Type':>10}")
    print(f"  {'-'*80}")

    # Find special nodes
    bc_nodes = torch.where(data.bc_disp.squeeze() > 0.5)[0].tolist()
    resp_nodes = torch.where(data.x[:, 8] > 0.5)[0].tolist()
    show_nodes = list(range(min(10, len(pred_disp)))) + resp_nodes + bc_nodes
    show_nodes = sorted(set(show_nodes))

    for n in show_nodes[:20]:
        node_type = ''
        if n in bc_nodes:
            node_type = 'SUPPORT'
        if n in resp_nodes:
            node_type = 'RESPONSE'

        p_ux = pred_disp[n, 0].item()
        t_ux = true_disp[n, 0].item()
        p_uz = pred_disp[n, 1].item()
        t_uz = true_disp[n, 1].item()
        err_uz = abs(p_uz - t_uz)

        print(f"  {n:>6} {p_ux:>12.4e} {t_ux:>12.4e} "
              f"{p_uz:>12.4e} {t_uz:>12.4e} {err_uz:>12.4e} {node_type:>10}")

    # ── Face forces ──
    print(f"\n  FACE FORCES (pred, first 5 nodes with connections):")
    face_names = ['+x', '-x', '+z', '-z']
    connected = torch.where(data.face_mask.sum(dim=1) > 0)[0][:5]
    for n in connected:
        print(f"  Node {n.item():>3}: ", end='')
        for f in range(4):
            if data.face_mask[n, f] > 0.5:
                Fx = pred_faces[n, f, 0].item()
                Fz = pred_faces[n, f, 1].item()
                My = pred_faces[n, f, 2].item()
                print(f"  {face_names[f]}:[{Fx:>8.2f}, {Fz:>8.2f}, {My:>8.3f}]", end='')
        print()

    # ── Element forces comparison ──
    print(f"\n  ELEMENT FORCES (first 10):")
    true_elem = data.y_element  # (E, 4): N, M, V, sens
    print(f"  {'Elem':>6} {'true_N':>12} {'true_M':>12} "
          f"{'true_V':>12} {'type':>8}")
    print(f"  {'-'*56}")
    for e in range(min(10, len(true_elem))):
        dx = abs(data.elem_directions[e, 0].item())
        dz = abs(data.elem_directions[e, 2].item())
        etype = 'beam' if dx > dz else 'column'
        print(f"  {e:>6} {true_elem[e,0].item():>12.2f} "
              f"{true_elem[e,1].item():>12.4f} "
              f"{true_elem[e,2].item():>12.2f} {etype:>8}")

    # ── Overall error ──
    err = (pred_disp - true_disp).pow(2).sum().sqrt()
    ref = true_disp.pow(2).sum().sqrt().clamp(min=1e-10)
    rel_err = (err / ref).item()
    print(f"\n  OVERALL DISPLACEMENT ERROR: {rel_err:.6f} "
          f"({'≈ 1.0 means TRIVIAL SOLUTION' if rel_err > 0.9 else 'OK'})")


def plot_predictions(pred, data, case_idx=0, save_dir='RESULTS'):
    """Visualize predictions vs ground truth."""
    os.makedirs(save_dir, exist_ok=True)

    coords = data.coords.numpy()
    conn = data.connectivity.numpy()
    true_disp = data.y_node.numpy()
    pred_disp = pred[:, 0:3].numpy()
    N = len(coords)
    E = len(conn)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'PIGNN Predictions vs Ground Truth — Case {case_idx}',
                 fontsize=14, fontweight='bold')

    # ── 1. Deformed shape comparison ──
    ax = axes[0, 0]
    scale = 500  # amplification factor
    for e in range(E):
        n1, n2 = conn[e]
        # Undeformed
        ax.plot([coords[n1, 0], coords[n2, 0]],
                [coords[n1, 2], coords[n2, 2]],
                'k-', alpha=0.2, linewidth=0.5)
        # True deformed
        ax.plot([coords[n1, 0] + scale*true_disp[n1, 0],
                 coords[n2, 0] + scale*true_disp[n2, 0]],
                [coords[n1, 2] + scale*true_disp[n1, 1],
                 coords[n2, 2] + scale*true_disp[n2, 1]],
                'b-', linewidth=1.5, alpha=0.7)
        # Predicted deformed
        ax.plot([coords[n1, 0] + scale*pred_disp[n1, 0],
                 coords[n2, 0] + scale*pred_disp[n2, 0]],
                [coords[n1, 2] + scale*pred_disp[n1, 1],
                 coords[n2, 2] + scale*pred_disp[n2, 1]],
                'r--', linewidth=1.5, alpha=0.7)
    ax.set_title(f'Deformed Shape (×{scale})')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.legend(['Undeformed', 'True', 'Predicted'])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # ── 2-4. Displacement profiles ──
    dof_labels = ['ux (m)', 'uz (m)', 'θy (rad)']
    for i, label in enumerate(dof_labels):
        ax = axes[0, 1] if i == 0 else axes[1, i-1] if i > 0 else None
        if i == 0:
            ax = axes[0, 1]
        elif i == 1:
            ax = axes[0, 2]
        else:
            ax = axes[1, 0]

        ax.plot(range(N), true_disp[:, i], 'b.-', ms=2,
                label='True (Kratos)', alpha=0.7)
        ax.plot(range(N), pred_disp[:, i], 'r.--', ms=2,
                label='Predicted', alpha=0.7)
        ax.set_title(f'{label} per Node')
        ax.set_xlabel('Node ID')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ── 5. Scatter: pred vs true ──
    ax = axes[1, 1]
    for i, (label, color) in enumerate(
            zip(['ux', 'uz', 'θy'], ['red', 'blue', 'green'])):
        ax.scatter(true_disp[:, i], pred_disp[:, i],
                   s=5, alpha=0.5, color=color, label=label)
    lims = [min(true_disp.min(), pred_disp.min()),
            max(true_disp.max(), pred_disp.max())]
    if lims[0] == lims[1]:
        lims = [-1e-6, 1e-6]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title('Predicted vs True (all DOFs)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # ── 6. Error distribution ──
    ax = axes[1, 2]
    errors = np.abs(pred_disp - true_disp)
    for i, (label, color) in enumerate(
            zip(['|Δux|', '|Δuz|', '|Δθy|'], ['red', 'blue', 'green'])):
        ax.semilogy(range(N), errors[:, i] + 1e-20,
                    '.', ms=3, alpha=0.5, color=color, label=label)
    ax.set_title('Absolute Errors per Node')
    ax.set_xlabel('Node ID')
    ax.set_ylabel('|Error|')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'predictions_case_{case_idx}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {path}")


def plot_face_forces_on_frame(pred, data, case_idx=0, save_dir='RESULTS'):
    """Visualize predicted face forces on the frame."""
    os.makedirs(save_dir, exist_ok=True)

    coords = data.coords.numpy()
    conn = data.connectivity.numpy()
    face_forces = pred[:, 3:15].reshape(-1, 4, 3).numpy()
    face_mask = data.face_mask.numpy()
    N = len(coords)
    E = len(conn)

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle(f'Predicted Face Forces — Case {case_idx}',
                 fontsize=14, fontweight='bold')
    comp_names = ['Fx (N)', 'Fz (N)', 'My (N·m)']

    for comp in range(3):
        ax = axes[comp]
        # Draw frame
        for e in range(E):
            n1, n2 = conn[e]
            ax.plot([coords[n1, 0], coords[n2, 0]],
                    [coords[n1, 2], coords[n2, 2]],
                    'k-', alpha=0.3, linewidth=1)

        # Color nodes by net face force
        net_force = np.zeros(N)
        for n in range(N):
            for f in range(4):
                if face_mask[n, f] > 0.5:
                    net_force[n] += face_forces[n, f, comp]

        vmax = max(abs(net_force.min()), abs(net_force.max()), 1e-10)
        scatter = ax.scatter(coords[:, 0], coords[:, 2],
                            c=net_force, cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, s=30, zorder=5)
        plt.colorbar(scatter, ax=ax, label=comp_names[comp])
        ax.set_title(f'Net {comp_names[comp]}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'face_forces_case_{case_idx}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {path}")


def diagnose_gradients(model, data, device='cpu'):
    """Check what the autograd derivatives look like."""
    model.train()
    data = data.to(device)
    pred = model(data)
    coords = model.get_coords()

    print(f"\n{'='*60}")
    print(f"  AUTOGRAD GRADIENT DIAGNOSIS")
    print(f"{'='*60}")

    ux = pred[:, 0]
    uz = pred[:, 1]

    grad_ux = torch.autograd.grad(
        ux.sum(), coords, create_graph=True, retain_graph=True)[0]
    grad_uz = torch.autograd.grad(
        uz.sum(), coords, create_graph=True, retain_graph=True)[0]

    dux_dx = grad_ux[:, 0]
    duz_dx = grad_uz[:, 0]
    duz_dz = grad_uz[:, 2]

    grad2 = torch.autograd.grad(
        duz_dx.sum(), coords, create_graph=True, retain_graph=True)[0]
    d2uz_dx2 = grad2[:, 0]

    grad3 = torch.autograd.grad(
        d2uz_dx2.sum(), coords, create_graph=True, retain_graph=True)[0]
    d3uz_dx3 = grad3[:, 0]

    EA = data.prop_E * data.prop_A
    EI = data.prop_E * data.prop_I22

    print(f"\n  Displacement predictions:")
    print(f"    ux:  [{ux.min():.4e}, {ux.max():.4e}]  "
          f"(u_c = {data.u_c.item():.4e})")
    print(f"    uz:  [{uz.min():.4e}, {uz.max():.4e}]")

    print(f"\n  Derivatives (should be non-trivial):")
    print(f"    dux/dx:    [{dux_dx.min():.4e}, {dux_dx.max():.4e}]  "
          f"std={dux_dx.std():.4e}")
    print(f"    duz/dx:    [{duz_dx.min():.4e}, {duz_dx.max():.4e}]  "
          f"std={duz_dx.std():.4e}")
    print(f"    d²uz/dx²:  [{d2uz_dx2.min():.4e}, {d2uz_dx2.max():.4e}]  "
          f"std={d2uz_dx2.std():.4e}")
    print(f"    d³uz/dx³:  [{d3uz_dx3.min():.4e}, {d3uz_dx3.max():.4e}]  "
          f"std={d3uz_dx3.std():.4e}")

    print(f"\n  Constitutive forces (from autograd):")
    conn = data.connectivity
    nA = conn[:, 0]
    strain_A = dux_dx[nA]
    curv_A = d2uz_dx2[nA]
    d3w_A = d3uz_dx3[nA]
    N_auto = (EA * strain_A)
    M_auto = (EI * curv_A)
    V_auto = (EI * d3w_A)
    print(f"    N = EA·ε:    [{N_auto.min():.4e}, {N_auto.max():.4e}]  "
          f"(F_c = {data.F_c.item():.4e})")
    print(f"    M = EI·κ:    [{M_auto.min():.4e}, {M_auto.max():.4e}]  "
          f"(M_c = {data.M_c.item():.4e})")
    print(f"    V = EI·d³w:  [{V_auto.min():.4e}, {V_auto.max():.4e}]")

    print(f"\n  Face forces (from network):")
    face_forces = pred[:, 3:15].reshape(-1, 4, 3)
    active = data.face_mask.sum(dim=1) > 0
    ff_active = face_forces[active]
    print(f"    Fx: [{ff_active[:,:,0].min():.4e}, "
          f"{ff_active[:,:,0].max():.4e}]")
    print(f"    Fz: [{ff_active[:,:,1].min():.4e}, "
          f"{ff_active[:,:,1].max():.4e}]")
    print(f"    My: [{ff_active[:,:,2].min():.4e}, "
          f"{ff_active[:,:,2].max():.4e}]")

    print(f"\n  KEY INSIGHT:")
    if dux_dx.abs().max() < 1e-5:
        print(f"    ⚠ Derivatives are NEAR ZERO — coords input is being ignored!")
        print(f"    ⚠ The 128-dim GNN context dominates the 3-dim coords")
        print(f"    ⚠ Model found TRIVIAL SOLUTION: u ≈ 0 everywhere")
    else:
        print(f"    ✓ Derivatives are non-trivial")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 60)
    print("  PIGNN PREDICTION & DIAGNOSIS")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Load data ──
    data_list = torch.load("DATA/graph_dataset_norm.pt", weights_only=False)
    print(f"  Loaded {len(data_list)} graphs")

    # ── Load model ──
    model = load_trained_model("RESULTS/best.pt", data_list[0], device)

    # ── Predict & analyze each case ──
    for case_idx in range(min(3, len(data_list))):
        data = data_list[case_idx]
        pred, raw = predict_case(model, data, device)

        print_prediction_summary(pred, raw, data, case_idx)
        plot_predictions(pred, data, case_idx)
        plot_face_forces_on_frame(pred, data, case_idx)

    # ── Gradient diagnosis ──
    data0 = data_list[0].to(device)
    model = model.to(device)
    diagnose_gradients(model, data0, device)

    print(f"\n{'='*60}")
    print(f"  DIAGNOSIS COMPLETE")
    print(f"{'='*60}")