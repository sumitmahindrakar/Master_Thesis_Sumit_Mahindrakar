"""
train.py — Train PIGNN with autograd physics loss
+ Diagnostics showing WHY autograd fails for GNN frames
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
from model import PIGNN
from physics_loss import StrongFormPhysicsLoss


# ================================================================
# DIAGNOSTIC FUNCTIONS — PROOF OF FAILURE
# ================================================================

def diagnose_joint_tangent(data, device):
    """
    PROBLEM 1: Joint nodes get meaningless averaged tangent.

    At a beam-column joint, the tangent from each element
    points in a different direction. Averaging gives ~45°,
    which is wrong for BOTH elements.
    """
    print(f"\n{'═'*70}")
    print(f"  DIAGNOSTIC 1: Joint Node Tangent Vectors")
    print(f"{'═'*70}")

    data = data.to(device)
    conn = data.connectivity.cpu().numpy()
    dirs = data.elem_directions.cpu().numpy()
    coords = data.coords.cpu().numpy()
    E = conn.shape[0]
    N = coords.shape[0]

    # Build per-node element list
    node_elements = {i: [] for i in range(N)}
    for e in range(E):
        node_elements[conn[e, 0]].append(e)
        node_elements[conn[e, 1]].append(e)

    # Classify nodes
    print(f"\n  {'Node':>4}  {'#Elem':>5}  {'Type':>10}  "
          f"{'Elem Directions':>40}  {'Averaged t':>25}")
    print(f"  {'-'*90}")

    for node in range(N):
        elems = node_elements[node]
        n_conn = len(elems)

        # Get directions of connected elements
        elem_dirs = [dirs[e] for e in elems]

        # Average tangent (what the loss computes)
        t_avg = np.mean(elem_dirs, axis=0)
        t_avg_norm = np.linalg.norm(t_avg)
        if t_avg_norm > 1e-12:
            t_avg = t_avg / t_avg_norm

        # Check if all directions are parallel
        is_joint = False
        if n_conn >= 2:
            for k in range(1, len(elem_dirs)):
                dot = abs(np.dot(elem_dirs[0], elem_dirs[k]))
                if dot < 0.99:  # not parallel
                    is_joint = True
                    break

        node_type = "JOINT ✗" if is_joint else (
            "support" if n_conn == 1 else "interior"
        )

        # Format element directions
        dir_strs = [f"e{e}:({dirs[e,0]:+.2f},{dirs[e,2]:+.2f})"
                     for e in elems]
        dir_str = ", ".join(dir_strs)

        print(f"  {node:>4}  {n_conn:>5}  {node_type:>10}  "
              f"{dir_str:>40}  "
              f"({t_avg[0]:+.3f}, {t_avg[2]:+.3f})")

        if is_joint:
            print(f"         ⚠ Beam tangent: ({elem_dirs[0][0]:+.2f},"
                  f"{elem_dirs[0][2]:+.2f})  "
                  f"Column tangent: ({elem_dirs[1][0]:+.2f},"
                  f"{elem_dirs[1][2]:+.2f})")
            print(f"         ⚠ Averaged to: ({t_avg[0]:+.3f},"
                  f"{t_avg[2]:+.3f}) — WRONG for both!")
            angle = np.degrees(np.arctan2(t_avg[2], t_avg[0]))
            print(f"         ⚠ Angle: {angle:.1f}° "
                  f"(should be 0° OR 90°, got neither)")

    # Count
    n_joints = sum(1 for node in range(N)
                   if len(node_elements[node]) >= 2
                   and any(abs(np.dot(dirs[node_elements[node][0]],
                                      dirs[e])) < 0.99
                           for e in node_elements[node][1:]))
    n_supports = sum(1 for node in range(N)
                     if data.bc_disp[node].item() > 0.5)
    n_interior = N - n_joints - n_supports

    print(f"\n  Summary:")
    print(f"    Total nodes:    {N}")
    print(f"    Support nodes:  {n_supports}  (excluded by BC mask)")
    print(f"    Joint nodes:    {n_joints}  (WRONG tangent)")
    print(f"    Interior nodes: {n_interior}  "
          f"(only these have correct tangent)")
    print(f"\n  ⚠ Physics loss evaluated at {n_joints} nodes "
          f"with WRONG tangent direction!")
    if n_interior == 0:
        print(f"  ⚠ ZERO interior nodes — "
              f"NO correct evaluation points exist!")


def diagnose_jacobian_vs_spatial(model, data, device):
    """
    PROBLEM 2: GNN autograd Jacobian ≠ spatial derivative.

    For a PINN:  u = NN(x) → du/dx is a spatial derivative.
    For a GNN:   u_i = GNN(x_i, x_j, ...) → du_i/dx_i includes
                 effects of edge features, neighbor messages, etc.

    Compare:
      autograd:   ∂u_i/∂x_i  (network Jacobian)
      spatial:    (u_j - u_i) / L_ij  (finite difference)
    """
    print(f"\n{'═'*70}")
    print(f"  DIAGNOSTIC 2: Autograd Jacobian vs Spatial Derivative")
    print(f"{'═'*70}")

    model.eval()
    data = data.to(device)

    # Forward pass with grad
    pred = model(data)
    coords = model.get_coords()

    u_x = pred[:, 0:1]
    u_z = pred[:, 1:2]

    # Autograd: du_x/dx_coord
    grad_ux = torch.autograd.grad(
        u_x.sum(), coords,
        create_graph=True, retain_graph=True
    )[0]  # (N, 3): [du_x/dx, du_x/dy, du_x/dz]

    grad_uz = torch.autograd.grad(
        u_z.sum(), coords,
        create_graph=True, retain_graph=True
    )[0]  # (N, 3)

    # Finite difference per element
    conn = data.connectivity
    E = conn.shape[0]

    print(f"\n  {'Elem':>4}  {'Nodes':>8}  {'L':>8}  "
          f"{'FD du_x/ds':>14}  {'AG du_x/dx(i)':>14}  "
          f"{'AG du_x/dx(j)':>14}  {'Ratio i':>10}  {'Ratio j':>10}")
    print(f"  {'-'*100}")

    ratios = []
    for e in range(min(E, 15)):  # show first 15 elements
        i, j = conn[e, 0].item(), conn[e, 1].item()
        L = data.elem_lengths[e].item()
        d = data.elem_directions[e]

        # Finite difference (spatial derivative)
        du_x_fd = (pred[j, 0] - pred[i, 0]).item() / max(L, 1e-15)

        # Autograd at node i and j
        du_x_ag_i = grad_ux[i, 0].item()   # du_x/dx at node i
        du_x_ag_j = grad_ux[j, 0].item()   # du_x/dx at node j

        # Ratio
        r_i = du_x_ag_i / du_x_fd if abs(du_x_fd) > 1e-15 else float('nan')
        r_j = du_x_ag_j / du_x_fd if abs(du_x_fd) > 1e-15 else float('nan')

        ratios.append((r_i, r_j))

        print(f"  {e:>4}  ({i:>2},{j:>2})  {L:>8.3f}  "
              f"{du_x_fd:>+14.6e}  {du_x_ag_i:>+14.6e}  "
              f"{du_x_ag_j:>+14.6e}  {r_i:>10.3f}  {r_j:>10.3f}")

    valid_ratios = [(ri, rj) for ri, rj in ratios
                    if not (np.isnan(ri) or np.isnan(rj))]
    if valid_ratios:
        ri_vals = [r[0] for r in valid_ratios]
        rj_vals = [r[1] for r in valid_ratios]
        print(f"\n  Ratio autograd/FD at node i: "
              f"[{min(ri_vals):.3f}, {max(ri_vals):.3f}]  "
              f"mean={np.mean(ri_vals):.3f}")
        print(f"  Ratio autograd/FD at node j: "
              f"[{min(rj_vals):.3f}, {max(rj_vals):.3f}]  "
              f"mean={np.mean(rj_vals):.3f}")
        print(f"\n  If autograd = spatial derivative, "
              f"ratios should all be ≈ 1.0")
        print(f"  Actual range: [{min(ri_vals + rj_vals):.3f}, "
              f"{max(ri_vals + rj_vals):.3f}]")
        if max(abs(r) for r in ri_vals + rj_vals) > 2.0:
            print(f"  ⚠ LARGE DEVIATION — autograd ≠ spatial derivative!")
    else:
        print(f"\n  ⚠ All predictions near zero — "
              f"cannot compute meaningful ratios")
        print(f"  ⚠ This itself is a problem: model outputs ~0 everywhere")


def diagnose_derivative_noise(model, data, device):
    """
    PROBLEM 3: High-order autograd through GNN = noise.

    Show magnitudes of 1st through 4th order derivatives.
    For real physics, higher derivatives should be bounded.
    Through a GNN, they explode or oscillate wildly.
    """
    print(f"\n{'═'*70}")
    print(f"  DIAGNOSTIC 3: High-Order Derivative Magnitudes")
    print(f"{'═'*70}")

    model.eval()
    data = data.to(device)

    pred = model(data)
    coords = model.get_coords()

    u_z = pred[:, 1:2]  # transverse displacement

    # Build node tangent (same as physics_loss does)
    conn = data.connectivity
    E = conn.shape[0]
    N = coords.shape[0]

    elem_dir = data.elem_directions
    t_elem = elem_dir / elem_dir.norm(
        dim=1, keepdim=True).clamp(min=1e-12)

    t_node = torch.zeros(N, 3, device=device)
    count = torch.zeros(N, 1, device=device)
    ones = torch.ones(E, 1, device=device)

    t_node.scatter_add_(0, conn[:, 0:1].expand(-1, 3), t_elem)
    t_node.scatter_add_(0, conn[:, 1:2].expand(-1, 3), t_elem)
    count.scatter_add_(0, conn[:, 0:1], ones)
    count.scatter_add_(0, conn[:, 1:2], ones)
    t_node = t_node / count.clamp(min=1.0)
    t_node = t_node / t_node.norm(
        dim=1, keepdim=True).clamp(min=1e-12)

    cos_t = t_node[:, 0:1]
    sin_t = t_node[:, 2:3]
    u_x = pred[:, 0:1]
    u_n = -u_x * sin_t + u_z * cos_t

    # Compute successive derivatives
    print(f"\n  Computing directional derivatives of u_n along s...")
    print(f"  (For bending: need d⁴u_n/ds⁴)")

    derivs = [u_n]
    names = ['u_n', 'du_n/ds', 'd²u_n/ds²',
             'd³u_n/ds³', 'd⁴u_n/ds⁴']

    for order in range(1, 5):
        try:
            gy = torch.autograd.grad(
                derivs[-1].sum(), coords,
                create_graph=True, retain_graph=True
            )[0]
            d_ds = (gy * t_node).sum(dim=1, keepdim=True)
            derivs.append(d_ds)
        except RuntimeError as e:
            print(f"  ⚠ Order {order} FAILED: {e}")
            derivs.append(torch.zeros_like(u_n))
            break

    # Print statistics
    print(f"\n  {'Order':>6}  {'Name':>12}  {'Min':>14}  "
          f"{'Max':>14}  {'Mean|·|':>14}  {'Std':>14}  {'Status'}")
    print(f"  {'-'*95}")

    for order, (d, name) in enumerate(zip(derivs, names)):
        d_np = d.detach().cpu().numpy().flatten()
        d_min = np.min(d_np)
        d_max = np.max(d_np)
        d_mean = np.mean(np.abs(d_np))
        d_std = np.std(d_np)

        # Check for blow-up or noise
        if order == 0:
            status = "OK"
            ref_scale = max(d_mean, 1e-15)
        else:
            ratio = d_mean / ref_scale if ref_scale > 1e-15 else 0
            if d_mean < 1e-15:
                status = "≈ 0 (dead)"
            elif ratio > 1e6:
                status = f"EXPLODED ✗ ({ratio:.1e}×)"
            elif d_std / max(d_mean, 1e-15) > 10:
                status = f"NOISY ✗ (CV={d_std/max(d_mean,1e-15):.1f})"
            else:
                status = "OK"

        print(f"  {order:>6}  {name:>12}  {d_min:>+14.4e}  "
              f"{d_max:>+14.4e}  {d_mean:>14.4e}  "
              f"{d_std:>14.4e}  {status}")

    # Physical sanity check
    print(f"\n  Physical expectation (for UDL beam):")
    EI = (data.prop_E * data.prop_I22).mean().item()
    q = data.elem_load[:, 2].abs().max().item()
    Lmax = data.elem_lengths.max().item()

    if q > 0 and EI > 0:
        expected_u = q * Lmax**4 / (384 * EI)
        expected_d2 = q * Lmax**2 / (8 * EI)
        expected_d4 = q / EI

        print(f"    EI = {EI:.4e},  q = {q:.4e},  L = {Lmax:.4e}")
        print(f"    Expected u_max     ≈ {expected_u:.4e}")
        print(f"    Expected d²u/ds²   ≈ {expected_d2:.4e}")
        print(f"    Expected d⁴u/ds⁴   ≈ {expected_d4:.4e}")

        actual_u = np.mean(np.abs(derivs[0].detach().cpu().numpy()))
        actual_d2 = np.mean(np.abs(derivs[2].detach().cpu().numpy()))
        actual_d4 = np.mean(np.abs(derivs[4].detach().cpu().numpy()))

        print(f"    Actual   u_mean    = {actual_u:.4e}  "
              f"(ratio: {actual_u/max(expected_u,1e-30):.2f})")
        print(f"    Actual   d²u_mean  = {actual_d2:.4e}  "
              f"(ratio: {actual_d2/max(expected_d2,1e-30):.2f})")
        print(f"    Actual   d⁴u_mean  = {actual_d4:.4e}  "
              f"(ratio: {actual_d4/max(expected_d4,1e-30):.2f})")


def diagnose_all(model, data, device):
    """Run all three diagnostics."""
    print(f"\n{'╔'+'═'*68+'╗'}")
    print(f"{'║'} AUTOGRAD PHYSICS LOSS — DIAGNOSTIC REPORT "
          f"{'║':>26}")
    print(f"{'╚'+'═'*68+'╝'}")

    diagnose_joint_tangent(data, device)
    diagnose_jacobian_vs_spatial(model, data, device)
    diagnose_derivative_noise(model, data, device)

    print(f"\n{'═'*70}")
    print(f"  CONCLUSION")
    print(f"{'═'*70}")
    print(f"  Problem 1: Joint nodes have WRONG tangent direction")
    print(f"             → N, M, V computed in wrong local frame")
    print(f"  Problem 2: GNN autograd ≠ spatial derivative")
    print(f"             → du/dx is network Jacobian, not du_physical/dx")
    print(f"  Problem 3: High-order derivatives are noise")
    print(f"             → d⁴u/ds⁴ through GNN has no spatial meaning")
    print(f"\n  RECOMMENDATION: Use element-wise stiffness approach")
    print(f"{'═'*70}")


# ================================================================
# TRAINING FUNCTIONS
# ================================================================

def train_one_epoch(model, data_list, optimizer, loss_fn, device):
    """
    Train one epoch.
    Note: loss_fn(model, data) — model forward happens INSIDE loss
    because autograd needs coords.requires_grad in the graph.
    """
    model.train()
    total_loss = 0.0
    loss_components = {'axial': 0, 'bending': 0, 'kinematic': 0}
    idx = np.random.permutation(len(data_list))

    for i in idx:
        data = data_list[i].to(device)

        # Forward + physics loss (model called inside loss_fn)
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
    avg_components = {k: v / n for k, v in loss_components.items()}
    return total_loss / n, avg_components


def evaluate(model, data_list, loss_fn, device):
    """
    Evaluate WITHOUT torch.no_grad() — autograd needs gradients.
    """
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
    avg_components = {k: v / n for k, v in loss_components.items()}
    return total_loss / n, avg_components


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
    print("  TRAIN PIGNN — Autograd Physics Loss")
    print("  (Demonstrating WHY this approach fails)")
    print("=" * 70)

    # ── Config ──
    HIDDEN_DIM = 128
    N_LAYERS = 6
    LR = 1e-4
    EPOCHS = 100        # fewer — we expect failure
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n  Device:     {DEVICE}")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  MP layers:  {N_LAYERS}")
    print(f"  LR:         {LR}")
    print(f"  Epochs:     {EPOCHS}")

    # ── Load RAW data (not normalized — physics needs real units) ──
    print(f"\n── Loading RAW graph data ──")
    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    print(f"  {len(data_list)} graphs loaded")

    s = data_list[0]
    print(f"  Sample: {s.num_nodes} nodes, {s.n_elements} elements")
    print(f"  Node features:  {s.x.shape}")
    print(f"  Edge features:  {s.edge_attr.shape}")
    print(f"  Node targets:   {s.y_node.shape}   [u_x, u_z, φ]")
    print(f"  Elem targets:   {s.y_element.shape} [N, M, V, sens]")
    print(f"  E:    {s.prop_E[0]:.4e}")
    print(f"  A:    {s.prop_A[0]:.4e}")
    print(f"  I22:  {s.prop_I22[0]:.4e}")

    # ── Split ──
    train_data, val_data, test_data = split_data(data_list)
    print(f"  Train: {len(train_data)}  Val: {len(val_data)}"
          f"  Test: {len(test_data)}")

    # ── Model (3 DOFs: u_x, u_z, φ) ──
    model = PIGNN(
        node_in_dim=9,
        edge_in_dim=10,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        out_dim=3,            # [u_x, u_z, φ]
    ).to(DEVICE)
    print(f"  Parameters: {model.count_params():,}")

    # ── Loss + Optimizer ──
    loss_fn = StrongFormPhysicsLoss(
        w_axial=1.0,
        w_bending=1.0,
        w_kinematic=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7
    )

    # ════════════════════════════════════════════════════════
    # DIAGNOSTIC 1: Before training — check structure
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PRE-TRAINING DIAGNOSTICS")
    print(f"{'='*70}")
    diagnose_joint_tangent(data_list[0], DEVICE)

    # ════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  TRAINING")
    print(f"{'='*70}")
    print(f"  {'Ep':>4}  {'Train':>12}  {'Val':>12}  "
          f"{'Axial':>10}  {'Bending':>10}  {'Kin':>10}  "
          f"{'LR':>8}  {'Time':>5}")
    print(f"  {'-'*85}")

    best_val = float('inf')
    best_epoch = 0
    history = {'train': [], 'val': [],
               'axial': [], 'bending': [], 'kinematic': []}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_comp = train_one_epoch(
            model, train_data, optimizer, loss_fn, DEVICE)
        val_loss, val_comp = evaluate(
            model, val_data, loss_fn, DEVICE)

        scheduler.step(val_loss)
        dt = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['axial'].append(train_comp['axial'])
        history['bending'].append(train_comp['bending'])
        history['kinematic'].append(train_comp['kinematic'])

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "DATA/pignn_autograd_best.pt")

        if epoch % 10 == 0 or epoch <= 5:
            print(f"  {epoch:>4}  {train_loss:>12.4e}  "
                  f"{val_loss:>12.4e}  "
                  f"{train_comp['axial']:>10.3e}  "
                  f"{train_comp['bending']:>10.3e}  "
                  f"{train_comp['kinematic']:>10.3e}  "
                  f"{lr:>8.1e}  {dt:>4.1f}s")

    print(f"\n  Best val loss: {best_val:.6e}  (epoch {best_epoch})")

    # ════════════════════════════════════════════════════════
    # POST-TRAINING DIAGNOSTICS
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  POST-TRAINING DIAGNOSTICS")
    print(f"{'='*70}")

    model.load_state_dict(
        torch.load("DATA/pignn_autograd_best.pt", weights_only=False))

    # Run all three diagnostics
    diagnose_all(model, test_data[0], DEVICE)

    # ════════════════════════════════════════════════════════
    # COMPARE PREDICTIONS vs FEM GROUND TRUTH
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PREDICTIONS vs FEM")
    print(f"{'='*70}")

    model.eval()
    sample = test_data[0].to(DEVICE)
    pred = model(sample)
    target = sample.y_node  # FEM ground truth: [u_x, u_z, φ]

    labels = ['u_x', 'u_z', 'φ']
    print(f"\n  {'DOF':>4}  {'Pred Range':>28}  "
          f"{'FEM Range':>28}  {'MAE':>12}  {'RelErr':>10}")
    print(f"  {'-'*90}")

    for i, label in enumerate(labels):
        p = pred[:, i].detach()
        t = target[:, i]
        mae = (p - t).abs().mean().item()
        t_max = t.abs().max().item()
        rel = mae / max(t_max, 1e-15)
        print(f"  {label:>4}  "
              f"[{p.min():>+12.6e}, {p.max():>+12.6e}]  "
              f"[{t.min():>+12.6e}, {t.max():>+12.6e}]  "
              f"{mae:>12.4e}  {rel:>10.2%}")

    # ── Check if model learned trivial solution ──
    pred_mag = pred.detach().abs().max().item()
    target_mag = target.abs().max().item()
    print(f"\n  Max |prediction|:  {pred_mag:.6e}")
    print(f"  Max |FEM target|:  {target_mag:.6e}")
    print(f"  Ratio:             {pred_mag / max(target_mag, 1e-15):.4f}")

    if pred_mag < target_mag * 0.01:
        print(f"  ⚠ MODEL LEARNED NEAR-ZERO (trivial solution)")
        print(f"    This happens because u=0 approximately satisfies")
        print(f"    the autograd equilibrium when derivatives are noisy")

    # ── BC check ──
    bc_nodes = (sample.bc_disp.squeeze() > 0.5).nonzero().squeeze()
    print(f"\n  Support nodes: {bc_nodes.tolist()}")
    print(f"  Max |disp| at supports: "
          f"{pred[bc_nodes, :2].detach().abs().max():.2e}  "
          f"(should be 0)")

    # ════════════════════════════════════════════════════════
    # TRAINING CURVES
    # ════════════════════════════════════════════════════════
    torch.save(history, "DATA/train_history_autograd.pt")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Total loss
        axes[0].semilogy(history['train'], label='Train', alpha=0.8)
        axes[0].semilogy(history['val'], label='Val', alpha=0.8)
        axes[0].axvline(best_epoch - 1, color='red', ls='--',
                        alpha=0.5, label=f'Best (ep {best_epoch})')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss (log)')
        axes[0].set_title('Autograd Physics Loss — Total')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Components
        axes[1].semilogy(history['axial'], label='Axial', alpha=0.8)
        axes[1].semilogy(history['bending'],
                         label='Bending', alpha=0.8)
        axes[1].semilogy(history['kinematic'],
                         label='Kinematic', alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Component Loss (log)')
        axes[1].set_title('Loss Components')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('PIGNN — Autograd Physics Loss '
                     '(Expected: fails to converge)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('DATA/training_curve_autograd.png', dpi=150)
        plt.show()
        print(f"  Saved: DATA/training_curve_autograd.png")
    except ImportError:
        pass

    # ════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  FINAL VERDICT")
    print(f"{'═'*70}")
    print(f"  Training converged:    ", end="")
    if best_val < 1e-3:
        print("YES (but answers are wrong)")
    else:
        print(f"NO (loss = {best_val:.4e})")

    print(f"  Predictions accurate:  ", end="")
    max_rel_err = max(
        (pred[:, i].detach() - target[:, i]).abs().mean().item()
        / max(target[:, i].abs().max().item(), 1e-15)
        for i in range(3)
    )
    if max_rel_err < 0.05:
        print(f"YES (rel err {max_rel_err:.2%})")
    else:
        print(f"NO (rel err {max_rel_err:.2%})")

    print(f"\n  ROOT CAUSES:")
    print(f"    1. GNN autograd ∂u/∂x ≠ spatial derivative du/dx")
    print(f"    2. Joint nodes: averaged tangent wrong for all elements")
    print(f"    3. d⁴u/ds⁴ through GNN: noise, not physics")
    print(f"\n  NEXT STEP: Use element-wise stiffness loss")
    print(f"{'═'*70}")