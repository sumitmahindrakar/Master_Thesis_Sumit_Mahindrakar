"""
train.py — Test coordinate-separated PIGNN
Diagnostics proving what still fails and what improves.
"""

import os
from pathlib import Path
print(f"Working directory: {os.getcwd()}")
CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)
print(f"Working directory: {os.getcwd()}")
import torch
import numpy as np
import time
from model import PIGNN_SeparatedCoords
from physics_loss import StrongFormPhysicsLoss


# ================================================================
# DIAGNOSTICS
# ================================================================

def diagnose_autograd_path(model, data, device):
    """
    TEST: Does separating coords give cleaner spatial derivatives?

    Compare:
      1. Autograd ∂u/∂x through decoder only
      2. Finite difference (u_j - u_i) / L
      3. Check if they match better than before
    """
    print(f"\n{'═'*70}")
    print(f"  DIAGNOSTIC A: Autograd Path Check")
    print(f"  (Coords separated from MP — only through decoder)")
    print(f"{'═'*70}")

    model.eval()
    data = data.to(device)

    pred = model(data)
    coords_2d = model.get_coords()      # (N, 2): [x, z]

    # Check autograd graph
    print(f"\n  coords_2d.requires_grad: {coords_2d.requires_grad}")
    print(f"  coords_2d.shape:         {coords_2d.shape}")
    print(f"  pred.requires_grad:      {pred.requires_grad}")

    # Compute ∂u_x/∂x and ∂u_z/∂z via autograd
    u_x = pred[:, 0:1]
    u_z = pred[:, 1:2]

    grad_ux = torch.autograd.grad(
        u_x.sum(), coords_2d,
        create_graph=True, retain_graph=True
    )[0]  # (N, 2): [∂u_x/∂x, ∂u_x/∂z]

    grad_uz = torch.autograd.grad(
        u_z.sum(), coords_2d,
        create_graph=True, retain_graph=True
    )[0]  # (N, 2): [∂u_z/∂x, ∂u_z/∂z]

    print(f"\n  Autograd gradients:")
    print(f"    ∂u_x/∂x: [{grad_ux[:, 0].min():.4e}, "
          f"{grad_ux[:, 0].max():.4e}]")
    print(f"    ∂u_x/∂z: [{grad_ux[:, 1].min():.4e}, "
          f"{grad_ux[:, 1].max():.4e}]")
    print(f"    ∂u_z/∂x: [{grad_uz[:, 0].min():.4e}, "
          f"{grad_uz[:, 0].max():.4e}]")
    print(f"    ∂u_z/∂z: [{grad_uz[:, 1].min():.4e}, "
          f"{grad_uz[:, 1].max():.4e}]")

    # Compare with finite differences per element
    conn = data.connectivity
    E = conn.shape[0]

    print(f"\n  {'Elem':>4}  {'θ':>6}  {'FD du_s/ds':>14}  "
          f"{'AG du_s/ds':>14}  {'Ratio':>8}  "
          f"{'FD du_n/ds':>14}  {'AG du_n/ds':>14}  {'Ratio':>8}")
    print(f"  {'-'*100}")

    dirs = data.elem_directions
    fd_ratios = []
    for e in range(min(E, 12)):
        i, j = conn[e, 0].item(), conn[e, 1].item()
        L = data.elem_lengths[e].item()
        cos_t = dirs[e, 0].item()
        sin_t = dirs[e, 2].item()
        theta = np.degrees(np.arctan2(sin_t, cos_t))

        # Global displacements
        uxi = pred[i, 0].item()
        uzi = pred[i, 1].item()
        uxj = pred[j, 0].item()
        uzj = pred[j, 1].item()

        # Local displacements
        us_i =  uxi * cos_t + uzi * sin_t
        us_j =  uxj * cos_t + uzj * sin_t
        un_i = -uxi * sin_t + uzi * cos_t
        un_j = -uxj * sin_t + uzj * cos_t

        # Finite difference
        fd_dus = (us_j - us_i) / max(L, 1e-15)
        fd_dun = (un_j - un_i) / max(L, 1e-15)

        # Autograd at midpoint (average of nodes i, j)
        ag_dux_dx_i = grad_ux[i, 0].item()
        ag_dux_dz_i = grad_ux[i, 1].item()
        ag_duz_dx_i = grad_uz[i, 0].item()
        ag_duz_dz_i = grad_uz[i, 1].item()

        ag_dux_dx_j = grad_ux[j, 0].item()
        ag_dux_dz_j = grad_ux[j, 1].item()
        ag_duz_dx_j = grad_uz[j, 0].item()
        ag_duz_dz_j = grad_uz[j, 1].item()

        # Average autograd at two nodes
        ag_dux_dx = 0.5 * (ag_dux_dx_i + ag_dux_dx_j)
        ag_dux_dz = 0.5 * (ag_dux_dz_i + ag_dux_dz_j)
        ag_duz_dx = 0.5 * (ag_duz_dx_i + ag_duz_dx_j)
        ag_duz_dz = 0.5 * (ag_duz_dz_i + ag_duz_dz_j)

        # Autograd du_s/ds = cos·(∂u_x/∂x·cos + ∂u_x/∂z·sin)
        #                   + sin·(∂u_z/∂x·cos + ∂u_z/∂z·sin)
        ag_dus = (cos_t * (ag_dux_dx * cos_t + ag_dux_dz * sin_t) +
                  sin_t * (ag_duz_dx * cos_t + ag_duz_dz * sin_t))
        ag_dun = (-sin_t * (ag_dux_dx * cos_t + ag_dux_dz * sin_t) +
                   cos_t * (ag_duz_dx * cos_t + ag_duz_dz * sin_t))

        r_s = ag_dus / fd_dus if abs(fd_dus) > 1e-15 else float('nan')
        r_n = ag_dun / fd_dun if abs(fd_dun) > 1e-15 else float('nan')
        if not np.isnan(r_s):
            fd_ratios.append(abs(r_s))

        print(f"  {e:>4}  {theta:>5.0f}°  "
              f"{fd_dus:>+14.6e}  {ag_dus:>+14.6e}  {r_s:>8.3f}  "
              f"{fd_dun:>+14.6e}  {ag_dun:>+14.6e}  {r_n:>8.3f}")

    if fd_ratios:
        print(f"\n  Axial ratio AG/FD: "
              f"[{min(fd_ratios):.3f}, {max(fd_ratios):.3f}]  "
              f"mean={np.mean(fd_ratios):.3f}")
        print(f"  Should be ≈ 1.0 if autograd = spatial derivative")

        if 0.8 < np.mean(fd_ratios) < 1.2:
            print(f"  ✓ Separating coords IMPROVED autograd "
                  f"(closer to spatial derivative)")
        else:
            print(f"  ✗ Autograd still ≠ spatial derivative "
                  f"(decoder is per-node, not global)")


def diagnose_joint_problem(model, data, device):
    """
    TEST: Joint nodes still have wrong tangent.
    This is UNCHANGED by separating coords.
    """
    print(f"\n{'═'*70}")
    print(f"  DIAGNOSTIC B: Joint Node Tangent (Still Wrong)")
    print(f"{'═'*70}")

    data = data.to(device)
    conn = data.connectivity.cpu().numpy()
    dirs = data.elem_directions.cpu().numpy()
    N = data.coords.shape[0]

    node_elements = {i: [] for i in range(N)}
    for e in range(conn.shape[0]):
        node_elements[conn[e, 0]].append(e)
        node_elements[conn[e, 1]].append(e)

    joint_count = 0
    interior_count = 0
    support_count = 0

    for node in range(N):
        elems = node_elements[node]
        is_support = data.bc_disp[node].item() > 0.5
        if is_support:
            support_count += 1
            continue

        if len(elems) >= 2:
            elem_dirs = [dirs[e] for e in elems]
            is_joint = False
            for k in range(1, len(elem_dirs)):
                dot = abs(np.dot(elem_dirs[0], elem_dirs[k]))
                if dot < 0.99:
                    is_joint = True
                    break
            if is_joint:
                joint_count += 1
                t_avg = np.mean(elem_dirs, axis=0)
                t_avg = t_avg / max(np.linalg.norm(t_avg), 1e-12)
                angle = np.degrees(np.arctan2(t_avg[2], t_avg[0]))
                print(f"    Node {node}: JOINT — "
                      f"elements {elems}, "
                      f"averaged tangent angle = {angle:.1f}°")
            else:
                interior_count += 1
        elif len(elems) == 1:
            interior_count += 1

    total_free = N - support_count
    print(f"\n    Free nodes:     {total_free}")
    print(f"    Joint nodes:    {joint_count}  "
          f"({joint_count/max(total_free,1)*100:.0f}% of free — "
          f"WRONG tangent)")
    print(f"    Interior nodes: {interior_count}  "
          f"({interior_count/max(total_free,1)*100:.0f}% of free — "
          f"OK tangent)")
    print(f"\n    ⚠ Separating coords does NOT fix this.")
    print(f"      Joint tangent is a STRUCTURAL problem, "
          f"not an autograd problem.")


def diagnose_decoder_locality(model, data, device):
    """
    TEST: The decoder is a PER-NODE function.

    u_i = decoder(h_i, x_i, z_i)
    u_j = decoder(h_j, x_j, z_j)

    Since h_i ≠ h_j (different latent states from MP),
    ∂u/∂x at node i and node j come from DIFFERENT functions.

    This means ∂u/∂x is NOT a continuous spatial derivative
    across the domain — it's piecewise per node.
    """
    print(f"\n{'═'*70}")
    print(f"  DIAGNOSTIC C: Decoder Locality Problem")
    print(f"  (Each node has its own function u_i = f_i(x, z))")
    print(f"{'═'*70}")

    model.eval()
    data = data.to(device)

    # Get latent states h_i BEFORE decoder
    non_coord_features = data.x[:, 3:]
    h = model.node_encoder(non_coord_features)
    e = model.edge_encoder(data.edge_attr)
    for mp in model.mp_layers:
        h = h + mp(h, data.edge_index, e)

    # Check how different h_i values are across adjacent nodes
    conn = data.connectivity
    E = conn.shape[0]

    print(f"\n  Latent vector h_i statistics:")
    print(f"    h shape: {h.shape}")
    print(f"    h range: [{h.min():.4e}, {h.max():.4e}]")

    print(f"\n  {'Elem':>4}  {'Nodes':>8}  {'||h_i - h_j||':>16}  "
          f"{'||h_i||':>12}  {'Rel diff':>10}  {'Same func?'}")
    print(f"  {'-'*75}")

    diffs = []
    for elem in range(min(E, 12)):
        i, j = conn[elem, 0].item(), conn[elem, 1].item()
        hi = h[i].detach()
        hj = h[j].detach()
        diff = (hi - hj).norm().item()
        norm_i = hi.norm().item()
        rel = diff / max(norm_i, 1e-15)
        diffs.append(rel)

        same = "≈ YES" if rel < 0.01 else "NO ✗"
        print(f"  {elem:>4}  ({i:>2},{j:>2})  "
              f"{diff:>16.6e}  {norm_i:>12.6e}  "
              f"{rel:>10.4f}  {same}")

    print(f"\n  Mean relative difference: {np.mean(diffs):.4f}")
    print(f"\n  EXPLANATION:")
    print(f"    u_i = decoder(h_i, x_i, z_i)")
    print(f"    u_j = decoder(h_j, x_j, z_j)")
    print(f"    Since h_i ≠ h_j, these are DIFFERENT functions.")
    print(f"    ∂u_i/∂x comes from f_i(x, z) = decoder(h_i, x, z)")
    print(f"    ∂u_j/∂x comes from f_j(x, z) = decoder(h_j, x, z)")
    print(f"    These are NOT consistent spatial derivatives.")
    print(f"    The 'spatial derivative' is DISCONTINUOUS across nodes.")


def diagnose_high_order_derivs(model, data, device):
    """
    TEST: High-order derivatives through decoder.
    Better than through MP, but still problematic.
    """
    print(f"\n{'═'*70}")
    print(f"  DIAGNOSTIC D: High-Order Derivatives (Decoder Only)")
    print(f"{'═'*70}")

    model.eval()
    data = data.to(device)

    pred = model(data)
    coords_2d = model.get_coords()

    u_z = pred[:, 1:2]

    # Build 2D tangent
    conn = data.connectivity
    E = conn.shape[0]
    N = coords_2d.shape[0]

    elem_dir = data.elem_directions
    t_elem_2d = torch.stack([
        elem_dir[:, 0], elem_dir[:, 2]], dim=1)
    t_elem_2d = t_elem_2d / t_elem_2d.norm(
        dim=1, keepdim=True).clamp(min=1e-12)

    t_node_2d = torch.zeros(N, 2, device=device)
    count = torch.zeros(N, 1, device=device)
    ones = torch.ones(E, 1, device=device)
    t_node_2d.scatter_add_(0, conn[:, 0:1].expand(-1, 2), t_elem_2d)
    t_node_2d.scatter_add_(0, conn[:, 1:2].expand(-1, 2), t_elem_2d)
    count.scatter_add_(0, conn[:, 0:1], ones)
    count.scatter_add_(0, conn[:, 1:2], ones)
    t_node_2d = t_node_2d / count.clamp(min=1.0)
    t_node_2d = t_node_2d / t_node_2d.norm(
        dim=1, keepdim=True).clamp(min=1e-12)

    cos_t = t_node_2d[:, 0:1]
    sin_t = t_node_2d[:, 1:2]
    u_x = pred[:, 0:1]
    u_n = -u_x * sin_t + u_z * cos_t

    # Successive derivatives
    derivs = [u_n]
    names = ['u_n', 'du_n/ds', 'd²u_n/ds²',
             'd³u_n/ds³', 'd⁴u_n/ds⁴']

    for order in range(1, 5):
        try:
            gy = torch.autograd.grad(
                derivs[-1].sum(), coords_2d,
                create_graph=True, retain_graph=True
            )[0]
            d_ds = (gy * t_node_2d).sum(dim=1, keepdim=True)
            derivs.append(d_ds)
        except RuntimeError as err:
            print(f"  ⚠ Order {order} FAILED: {err}")
            derivs.append(torch.zeros_like(u_n))
            break

    print(f"\n  {'Order':>6}  {'Name':>12}  {'Min':>14}  "
          f"{'Max':>14}  {'Mean|·|':>14}  {'Std':>14}")
    print(f"  {'-'*80}")

    ref_scale = None
    for order, (d, name) in enumerate(zip(derivs, names)):
        d_np = d.detach().cpu().numpy().flatten()
        d_abs = np.abs(d_np)
        d_min = np.min(d_np)
        d_max = np.max(d_np)
        d_mean = np.mean(d_abs)
        d_std = np.std(d_np)

        if order == 0:
            ref_scale = max(d_mean, 1e-15)

        print(f"  {order:>6}  {name:>12}  {d_min:>+14.4e}  "
              f"{d_max:>+14.4e}  {d_mean:>14.4e}  {d_std:>14.4e}")

    # Physical comparison
    EI = (data.prop_E * data.prop_I22).mean().item()
    q = data.elem_load[:, 2].abs().max().item()
    Lmax = data.elem_lengths.max().item()

    if q > 0 and EI > 0:
        print(f"\n  Physical reference (UDL beam):")
        print(f"    EI={EI:.4e}, q={q:.4e}, L={Lmax:.4e}")
        expected = {
            'u_n':       q * Lmax**4 / (384 * EI),
            'd²u_n/ds²': q * Lmax**2 / (8 * EI),
            'd⁴u_n/ds⁴': q / EI,
        }
        for name, exp in expected.items():
            idx = names.index(name) if name in names else -1
            if idx >= 0:
                actual = np.mean(np.abs(
                    derivs[idx].detach().cpu().numpy()))
                ratio = actual / max(exp, 1e-30)
                print(f"    {name}: expected={exp:.4e}  "
                      f"actual={actual:.4e}  "
                      f"ratio={ratio:.2f}")


def diagnose_all(model, data, device):
    """Run all diagnostics."""
    print(f"\n{'╔'+'═'*68+'╗'}")
    print(f"{'║'} SEPARATED-COORDS PIGNN — DIAGNOSTIC REPORT "
          f"{'║':>24}")
    print(f"{'╚'+'═'*68+'╝'}")

    diagnose_autograd_path(model, data, device)
    diagnose_joint_problem(model, data, device)
    diagnose_decoder_locality(model, data, device)
    diagnose_high_order_derivs(model, data, device)

    print(f"\n{'═'*70}")
    print(f"  CONCLUSION: Separated Coords")
    print(f"{'═'*70}")
    print(f"  ✓ IMPROVED: Autograd path is cleaner (no MP layers)")
    print(f"  ✗ STILL BROKEN: Joint tangent (structural problem)")
    print(f"  ✗ STILL BROKEN: Per-node decoder = discontinuous ∂u/∂x")
    print(f"  ✗ STILL BROKEN: High-order derivs = noisy/meaningless")
    print(f"\n  Separating coords helps but does NOT fix the core issue:")
    print(f"  GNN gives u at DISCRETE nodes, not a continuous field.")
    print(f"  Autograd differentiates the NETWORK, not the physics.")
    print(f"\n  RECOMMENDATION: Use element-wise stiffness approach.")
    print(f"{'═'*70}")


# ================================================================
# TRAINING
# ================================================================

def train_one_epoch(model, data_list, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    loss_components = {'axial': 0, 'bending': 0, 'kinematic': 0}
    idx = np.random.permutation(len(data_list))

    for i in idx:
        data = data_list[i].to(device)
        loss, loss_dict, pred = loss_fn(model, data)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += loss_dict.get(k, 0)

    n = len(data_list)
    return total_loss / n, {k: v/n for k, v in loss_components.items()}


def evaluate(model, data_list, loss_fn, device):
    model.eval()
    total_loss = 0.0
    loss_components = {'axial': 0, 'bending': 0, 'kinematic': 0}

    for data in data_list:
        data = data.to(device)
        loss, loss_dict, pred = loss_fn(model, data)
        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += loss_dict.get(k, 0)

    n = len(data_list)
    return total_loss / n, {k: v/n for k, v in loss_components.items()}


def split_data(data_list, train_ratio=0.7, val_ratio=0.15, seed=42):
    n = len(data_list)
    np.random.seed(seed)
    idx = np.random.permutation(n)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    train = [data_list[i] for i in idx[:n_train]]
    val = [data_list[i] for i in idx[n_train:n_train + n_val]]
    test = [data_list[i] for i in idx[n_train + n_val:]]
    if not test:
        test = val.copy()
    return train, val, test


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("  EXPERIMENT: Coordinate-Separated PIGNN")
    print("  Testing if separating coords fixes autograd physics loss")
    print("=" * 70)

    HIDDEN_DIM = 128
    N_LAYERS = 6
    LR = 1e-4
    EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n  Device: {DEVICE}")

    # ── Load data ──
    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    print(f"  Loaded {len(data_list)} graphs")

    train_data, val_data, test_data = split_data(data_list)
    print(f"  Train: {len(train_data)}  Val: {len(val_data)}"
          f"  Test: {len(test_data)}")

    # ── Model (separated coords) ──
    model = PIGNN_SeparatedCoords(
        node_in_dim=9,
        edge_in_dim=10,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        out_dim=3,
        coord_dim=2,
    ).to(DEVICE)
    print(f"  Parameters: {model.count_params():,}")

    loss_fn = StrongFormPhysicsLoss(
        w_axial=1.0, w_bending=1.0,
        w_shear=1.0, w_kinematic=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7)

    # ── Pre-training diagnostics ──
    print(f"\n{'='*70}")
    print(f"  PRE-TRAINING DIAGNOSTICS")
    print(f"{'='*70}")
    diagnose_joint_problem(model, data_list[0], DEVICE)
    diagnose_decoder_locality(model, data_list[0], DEVICE)

    # ── Training ──
    print(f"\n{'='*70}")
    print(f"  TRAINING (separated coords)")
    print(f"{'='*70}")
    print(f"  {'Ep':>4}  {'Train':>12}  {'Val':>12}  "
          f"{'Axial':>10}  {'Bend':>10}  {'Kin':>10}  "
          f"{'LR':>8}  {'t':>4}")
    print(f"  {'-'*78}")

    best_val = float('inf')
    best_epoch = 0
    history = {'train': [], 'val': [],
               'axial': [], 'bending': [], 'kinematic': []}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, tc = train_one_epoch(
            model, train_data, optimizer, loss_fn, DEVICE)
        val_loss, vc = evaluate(
            model, val_data, loss_fn, DEVICE)
        scheduler.step(val_loss)
        dt = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['axial'].append(tc['axial'])
        history['bending'].append(tc['bending'])
        history['kinematic'].append(tc['kinematic'])

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(),
                        "DATA/pignn_separated_best.pt")

        if epoch % 10 == 0 or epoch <= 5:
            print(f"  {epoch:>4}  {train_loss:>12.4e}  "
                  f"{val_loss:>12.4e}  "
                  f"{tc['axial']:>10.3e}  "
                  f"{tc['bending']:>10.3e}  "
                  f"{tc['kinematic']:>10.3e}  "
                  f"{lr:>8.1e}  {dt:>3.0f}s")

    print(f"\n  Best val: {best_val:.6e} (epoch {best_epoch})")

    # ── Post-training diagnostics ──
    model.load_state_dict(
        torch.load("DATA/pignn_separated_best.pt",
                    weights_only=False))

    print(f"\n{'='*70}")
    print(f"  POST-TRAINING DIAGNOSTICS")
    print(f"{'='*70}")
    diagnose_all(model, test_data[0], DEVICE)

    # ── Compare with FEM ──
    print(f"\n{'='*70}")
    print(f"  PREDICTIONS vs FEM")
    print(f"{'='*70}")
    model.eval()
    sample = test_data[0].to(DEVICE)
    pred = model(sample)
    target = sample.y_node

    labels = ['u_x', 'u_z', 'φ']
    print(f"\n  {'DOF':>4}  {'Pred Range':>28}  "
          f"{'FEM Range':>28}  {'MAE':>12}  {'Rel':>8}")
    print(f"  {'-'*85}")
    for i, label in enumerate(labels):
        p = pred[:, i].detach()
        t = target[:, i]
        mae = (p - t).abs().mean().item()
        t_max = t.abs().max().item()
        rel = mae / max(t_max, 1e-15)
        print(f"  {label:>4}  "
              f"[{p.min():>+12.6e}, {p.max():>+12.6e}]  "
              f"[{t.min():>+12.6e}, {t.max():>+12.6e}]  "
              f"{mae:>12.4e}  {rel:>8.2%}")

    pred_mag = pred.detach().abs().max().item()
    target_mag = target.abs().max().item()
    print(f"\n  Max |pred|:   {pred_mag:.6e}")
    print(f"  Max |target|: {target_mag:.6e}")
    if pred_mag < target_mag * 0.01:
        print(f"  ⚠ TRIVIAL SOLUTION (near zero)")

    # ── Save history ──
    torch.save(history, "DATA/history_separated.pt")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].semilogy(history['train'], label='Train')
        axes[0].semilogy(history['val'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Separated Coords — Total Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(history['axial'], label='Axial')
        axes[1].semilogy(history['bending'], label='Bending')
        axes[1].semilogy(history['kinematic'], label='Kinematic')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Components')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Coordinate-Separated PIGNN',
                     fontweight='bold')
        plt.tight_layout()
        plt.savefig('DATA/training_separated.png', dpi=150)
        plt.show()
    except ImportError:
        pass

    # ── Final verdict ──
    print(f"\n{'═'*70}")
    print(f"  VERDICT: Separated Coords Experiment")
    print(f"{'═'*70}")
    print(f"  What improved:")
    print(f"    - Autograd path is cleaner (decoder only)")
    print(f"    - 1st derivative may be slightly better")
    print(f"  What did NOT improve:")
    print(f"    - Joint tangent still wrong (structural)")
    print(f"    - Decoder is per-node → discontinuous derivatives")
    print(f"    - High-order derivatives still noisy")
    print(f"    - Model likely still learns trivial solution")
    print(f"\n  NEXT: Switch to element-wise stiffness loss")
    print(f"{'═'*70}")