"""
train.py — Train PIGNN with element-wise stiffness physics loss

No FEM targets. Pure physics (equilibrium at free nodes).
Each element uses its own local frame → joints handled correctly.
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
# TRAINING FUNCTIONS
# ================================================================

def train_one_epoch(model, data_list, optimizer, loss_fn, device):
    """Train one epoch — graph-by-graph."""
    model.train()
    total_loss = 0.0
    comp = {'eq_fx': 0, 'eq_fz': 0, 'eq_m': 0,
            'kinematic': 0, 'eq_total': 0}
    idx = np.random.permutation(len(data_list))

    for i in idx:
        data = data_list[i].to(device)

        # Forward
        pred = model(data)

        # Physics loss (no targets needed)
        loss, loss_dict = loss_fn(pred, data)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k in comp:
            comp[k] += loss_dict.get(k, 0)

    n = len(data_list)
    return total_loss / n, {k: v / n for k, v in comp.items()}


@torch.no_grad()
def evaluate(model, data_list, loss_fn, device):
    """Evaluate — no gradients needed (no autograd w.r.t. coords)."""
    model.eval()
    total_loss = 0.0
    comp = {'eq_fx': 0, 'eq_fz': 0, 'eq_m': 0,
            'kinematic': 0, 'eq_total': 0}

    for data in data_list:
        data = data.to(device)
        pred = model(data)
        loss, loss_dict = loss_fn(pred, data)
        total_loss += loss.item()
        for k in comp:
            comp[k] += loss_dict.get(k, 0)

    n = len(data_list)
    return total_loss / n, {k: v / n for k, v in comp.items()}


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
# VERIFICATION UTILITIES
# ================================================================

def verify_physics_loss_with_fem(loss_fn, data, device):
    """
    Verify: if we plug in FEM displacements, equilibrium
    residual should be ≈ 0.
    """
    print(f"\n{'═'*70}")
    print(f"  VERIFICATION: FEM solution → physics loss ≈ 0?")
    print(f"{'═'*70}")

    data = data.to(device)
    fem_pred = data.y_node.to(device)    # [u_x, u_z, φ] from FEM

    loss, loss_dict = loss_fn(fem_pred, data)

    print(f"  FEM displacement range:")
    for i, name in enumerate(['u_x', 'u_z', 'φ']):
        col = fem_pred[:, i]
        print(f"    {name}: [{col.min():.6e}, {col.max():.6e}]")

    print(f"\n  Physics loss with FEM solution:")
    for k, v in loss_dict.items():
        status = "✓" if v < 1e-4 else "⚠" if v < 1e-2 else "✗"
        print(f"    {status} {k:>12}: {v:.6e}")

    print(f"\n  Total: {loss.item():.6e}")
    if loss.item() < 1e-4:
        print(f"  ✓ PASSED — FEM solution satisfies element-wise "
              f"equilibrium")
    else:
        print(f"  ⚠ Non-zero residual — check sign conventions "
              f"or load directions")

    return loss.item()


def compute_element_forces(pred, data, device):
    """
    Extract N, V, M per element from predicted displacements.
    Includes consistent UDL correction.
    """
    data = data.to(device)
    conn = data.connectivity
    L = data.elem_lengths
    dirs = data.elem_directions
    cos_t = dirs[:, 0]
    sin_t = dirs[:, 2]

    EA = data.prop_E * data.prop_A
    EI = data.prop_E * data.prop_I22

    node_i = conn[:, 0]
    node_j = conn[:, 1]

    u_x_i = pred[node_i, 0]
    u_z_i = pred[node_i, 1]
    phi_i = pred[node_i, 2]
    u_x_j = pred[node_j, 0]
    u_z_j = pred[node_j, 1]
    phi_j = pred[node_j, 2]

    u_s_i =  u_x_i * cos_t + u_z_i * sin_t
    u_n_i = -u_x_i * sin_t + u_z_i * cos_t
    u_s_j =  u_x_j * cos_t + u_z_j * sin_t
    u_n_j = -u_x_j * sin_t + u_z_j * cos_t

    # Local loads
    q_global = data.elem_load
    q_s =  q_global[:, 0] * cos_t + q_global[:, 2] * sin_t
    q_n = -q_global[:, 0] * sin_t + q_global[:, 2] * cos_t

    # Axial force (internal, tension positive)
    N_elem = EA / L * (u_s_j - u_s_i)

    # Shear and moment at node i (includes UDL)
    a = EI / (L ** 3)
    V_elem = (a * ( 12*u_n_i + 6*L*phi_i
                   - 12*u_n_j + 6*L*phi_j)
              - q_n * L / 2)

    M_elem = (a * L * ( 6*u_n_i + 4*L*phi_i
                       - 6*u_n_j + 2*L*phi_j)
              - q_n * L**2 / 12)

    return N_elem, V_elem, M_elem



# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("  TRAIN PIGNN — Element-Wise Stiffness Physics Loss")
    print("  No FEM targets. Pure equilibrium.")
    print("=" * 70)

    # ── Config ──
    HIDDEN_DIM = 128
    N_LAYERS = 6
    LR = 1e-4
    EPOCHS = 500
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
    print(f"  E:    [{s.prop_E.min():.4e}, {s.prop_E.max():.4e}]")
    print(f"  A:    [{s.prop_A.min():.4e}, {s.prop_A.max():.4e}]")
    print(f"  I22:  [{s.prop_I22.min():.4e}, {s.prop_I22.max():.4e}]")
    print(f"  Load: [{s.elem_load.min():.4e}, "
          f"{s.elem_load.max():.4e}]")

    # ── Split ──
    train_data, val_data, test_data = split_data(data_list)
    print(f"  Train: {len(train_data)}  Val: {len(val_data)}"
          f"  Test: {len(test_data)}")

    # ── Model ──
    model = PIGNN(
        node_in_dim=9,
        edge_in_dim=10,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        out_dim=3,
    ).to(DEVICE)
    print(f"  Parameters: {model.count_params():,}")

    # ── Loss + Optimizer ──
    loss_fn = StrongFormPhysicsLoss(
        w_equilibrium=1.0,
        w_kinematic=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-7
    )

    # ════════════════════════════════════════════════════════
    # STEP 0: VERIFY — FEM solution gives ~0 loss
    # ════════════════════════════════════════════════════════
    fem_loss = verify_physics_loss_with_fem(
        loss_fn, data_list[0], DEVICE)

    # ════════════════════════════════════════════════════════
    # TRAINING
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  TRAINING")
    print(f"{'='*70}")
    print(f"  {'Ep':>4}  {'Train':>12}  {'Val':>12}  "
          f"{'Fx':>10}  {'Fz':>10}  {'M':>10}  {'Kin':>10}  "
          f"{'LR':>8}  {'t':>4}")
    print(f"  {'-'*90}")

    best_val = float('inf')
    best_epoch = 0
    history = {
        'train': [], 'val': [],
        'eq_fx': [], 'eq_fz': [], 'eq_m': [],
        'kinematic': [], 'eq_total': [],
    }

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
        for k in ['eq_fx', 'eq_fz', 'eq_m', 'kinematic', 'eq_total']:
            history[k].append(tc.get(k, 0))

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "DATA/pignn_elemwise_best.pt")

        if epoch % 20 == 0 or epoch <= 5 or epoch == EPOCHS:
            print(f"  {epoch:>4}  {train_loss:>12.4e}  "
                  f"{val_loss:>12.4e}  "
                  f"{tc['eq_fx']:>10.3e}  "
                  f"{tc['eq_fz']:>10.3e}  "
                  f"{tc['eq_m']:>10.3e}  "
                  f"{tc['kinematic']:>10.3e}  "
                  f"{lr:>8.1e}  {dt:>3.0f}s")

    print(f"\n  Best val loss: {best_val:.6e}  (epoch {best_epoch})")

    # ════════════════════════════════════════════════════════
    # EVALUATION
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  EVALUATION")
    print(f"{'='*70}")

    model.load_state_dict(
        torch.load("DATA/pignn_elemwise_best.pt", weights_only=False))
    model.eval()

    # ── Test loss ──
    test_loss, test_comp = evaluate(
        model, test_data, loss_fn, DEVICE)
    print(f"  Test loss: {test_loss:.6e}")
    for k, v in test_comp.items():
        print(f"    {k:>12}: {v:.6e}")

    # ── Compare predictions vs FEM ──
    print(f"\n── Predictions vs FEM Ground Truth ──")
    sample = test_data[0].to(DEVICE)
    with torch.no_grad():
        pred = model(sample)
    target = sample.y_node

    labels = ['u_x', 'u_z', 'φ']
    print(f"\n  {'DOF':>4}  {'Pred Range':>28}  "
          f"{'FEM Range':>28}  {'MAE':>12}  {'Rel':>8}")
    print(f"  {'-'*85}")
    for i, label in enumerate(labels):
        p = pred[:, i]
        t = target[:, i]
        mae = (p - t).abs().mean().item()
        t_max = t.abs().max().item()
        rel = mae / max(t_max, 1e-15)
        print(f"  {label:>4}  "
              f"[{p.min():>+12.6e}, {p.max():>+12.6e}]  "
              f"[{t.min():>+12.6e}, {t.max():>+12.6e}]  "
              f"{mae:>12.4e}  {rel:>8.2%}")

    # ── Check trivial solution ──
    pred_mag = pred.abs().max().item()
    target_mag = target.abs().max().item()
    print(f"\n  Max |prediction|:  {pred_mag:.6e}")
    print(f"  Max |FEM target|:  {target_mag:.6e}")
    print(f"  Ratio:             "
          f"{pred_mag / max(target_mag, 1e-15):.4f}")

    if pred_mag < target_mag * 0.01:
        print(f"  ⚠ Near-zero prediction — "
              f"may need loss weighting adjustment")

    # ── Element internal forces ──
    print(f"\n── Element Forces: Predicted vs FEM ──")
    with torch.no_grad():
        N_pred, V_pred, M_pred = compute_element_forces(
            pred, sample, DEVICE)

    # FEM element targets: [N, M, V, sens]
    N_fem = sample.y_element[:, 0]
    M_fem = sample.y_element[:, 1]
    V_fem = sample.y_element[:, 2]

    print(f"\n  {'Force':>6}  {'Pred Range':>28}  "
          f"{'FEM Range':>28}  {'MAE':>12}")
    print(f"  {'-'*78}")
    for name, p, f in [('N', N_pred, N_fem),
                        ('V', V_pred, V_fem),
                        ('M', M_pred, M_fem)]:
        mae = (p - f).abs().mean().item()
        print(f"  {name:>6}  "
              f"[{p.min():>+12.4e}, {p.max():>+12.4e}]  "
              f"[{f.min():>+12.4e}, {f.max():>+12.4e}]  "
              f"{mae:>12.4e}")

    # ── BC check ──
    bc_nodes = (sample.bc_disp.squeeze() > 0.5).nonzero().squeeze()
    print(f"\n  Support nodes: {bc_nodes.tolist()}")
    print(f"  Max |disp| at supports: "
          f"{pred[bc_nodes, :2].abs().max():.2e}  (should be 0)")

    # ── Equilibrium check ──
    print(f"\n── Equilibrium Residual (should → 0) ──")
    loss_final, ld = loss_fn(pred, sample)
    free_disp = (sample.bc_disp.squeeze() < 0.5)
    free_rot  = (sample.bc_rot.squeeze() < 0.5)
    print(f"  Free nodes (disp): {free_disp.sum().item()}")
    print(f"  Free nodes (rot):  {free_rot.sum().item()}")
    print(f"  Residual Fx: {ld['eq_fx']:.6e}")
    print(f"  Residual Fz: {ld['eq_fz']:.6e}")
    print(f"  Residual M:  {ld['eq_m']:.6e}")

    # ════════════════════════════════════════════════════════
    # SAVE & PLOT
    # ════════════════════════════════════════════════════════
    torch.save(history, "DATA/train_history_elemwise.pt")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Total loss
        axes[0].semilogy(history['train'], label='Train', alpha=0.8)
        axes[0].semilogy(history['val'], label='Val', alpha=0.8)
        axes[0].axvline(best_epoch - 1, color='red', ls='--',
                        alpha=0.5, label=f'Best (ep {best_epoch})')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Physics Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Equilibrium components
        axes[1].semilogy(history['eq_fx'], label='ΣFx', alpha=0.8)
        axes[1].semilogy(history['eq_fz'], label='ΣFz', alpha=0.8)
        axes[1].semilogy(history['eq_m'], label='ΣM', alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Residual')
        axes[1].set_title('Equilibrium Components')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Kinematic
        axes[2].semilogy(history['kinematic'],
                         label='Kinematic', alpha=0.8, color='purple')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Residual')
        axes[2].set_title('Kinematic (φ = du_n/ds)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.suptitle('PIGNN — Element-Wise Stiffness Loss',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('DATA/training_elemwise.png', dpi=150)
        plt.show()
        print(f"  Saved: DATA/training_elemwise.png")
    except ImportError:
        pass

    # ════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  SUMMARY")
    print(f"{'═'*70}")
    print(f"  Method:     Element-wise EB stiffness")
    print(f"  Best val:   {best_val:.6e}  (epoch {best_epoch})")
    print(f"  Test loss:  {test_loss:.6e}")
    print(f"  FEM verify: {fem_loss:.6e}  "
          f"(should be ≈ 0)")
    print(f"\n  Properties:")
    print(f"    ✓ Joints handled correctly (force balance)")
    print(f"    ✓ Columns handled correctly (local coords)")
    print(f"    ✓ No autograd w.r.t. coordinates")
    print(f"    ✓ Exact for EB beam elements")
    print(f"    ✓ @torch.no_grad() works for evaluation")
    print(f"{'═'*70}")