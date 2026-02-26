"""
train.py — Train PIGNN with physics loss (axial + bending)
No FEM targets used.
"""

import os
os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\PIGNN\PIGNN_V04_displ")

import torch
import numpy as np
import time
from model import PIGNN
from physics_loss import PhysicsLoss


def train_one_epoch(model, data_list, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    idx = np.random.permutation(len(data_list))

    for i in idx:
        data = data_list[i].to(device)
        pred = model(data)
        loss = loss_fn(pred, data)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_list)


@torch.no_grad()
def evaluate(model, data_list, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for data in data_list:
        data = data.to(device)
        pred = model(data)
        loss = loss_fn(pred, data)
        total_loss += loss.item()
    return total_loss / len(data_list)


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


if __name__ == "__main__":

    print("=" * 60)
    print("  TRAIN PIGNN — Axial + Bending Equilibrium")
    print("  No FEM targets used.")
    print("=" * 60)

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

    # ── Load RAW data ──
    print(f"\n── Loading RAW graph data ──")
    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    print(f"  {len(data_list)} graphs loaded")

    s = data_list[0]
    print(f"  Sample: {s.num_nodes} nodes, {s.n_elements} elements")
    print(f"  E:    {s.prop_E[0]:.4e}")
    print(f"  A:    {s.prop_A[0]:.4e}")
    print(f"  I22:  {s.prop_I22[0]:.4e}")
    print(f"  Load: [{s.line_load.min():.4e}, {s.line_load.max():.4e}]")

    # ── Split ──
    train_data, val_data, test_data = split_data(data_list)
    print(f"  Train: {len(train_data)}  Val: {len(val_data)}"
          f"  Test: {len(test_data)}")

    # ── Model ──
    model = PIGNN(
        node_in_dim=9, edge_in_dim=11,
        hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, out_dim=6,
    ).to(DEVICE)
    print(f"  Parameters: {model.count_params():,}")

    # ── Loss + Optimizer ──
    loss_fn = PhysicsLoss(w_force=1.0, w_moment=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-7
    )

    # ── Training ──
    print(f"\n── Training ──")
    print(f"  {'Epoch':>6}  {'Train Loss':>14}  {'Val Loss':>14}"
          f"  {'LR':>10}  {'Time':>6}")
    print(f"  {'-'*58}")

    best_val = float('inf')
    best_epoch = 0
    history = {'train': [], 'val': []}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_data, optimizer,
                                     loss_fn, DEVICE)
        val_loss = evaluate(model, val_data, loss_fn, DEVICE)
        scheduler.step(val_loss)
        dt = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "DATA/pignn_best.pt")

        if epoch % 20 == 0 or epoch <= 5:
            print(f"  {epoch:>6}  {train_loss:>14.6e}  "
                  f"{val_loss:>14.6e}  {lr:>10.1e}  {dt:>5.1f}s")

    print(f"\n  Best val loss: {best_val:.6e}  (epoch {best_epoch})")

    # ── Test ──
    print(f"\n── Test Set ──")
    model.load_state_dict(
        torch.load("DATA/pignn_best.pt", weights_only=False)
    )
    test_loss = evaluate(model, test_data, loss_fn, DEVICE)
    print(f"  Test physics loss: {test_loss:.6e}")

    # ── Sample prediction vs FEM ──
    print(f"\n── Sample Prediction vs FEM ──")
    model.eval()
    sample = test_data[0].to(DEVICE)
    with torch.no_grad():
        pred = model(sample)
    target = sample.y_node

    labels = ['ux', 'uy', 'uz', 'rx', 'ry', 'rz']
    print(f"\n  {'DOF':>4}  {'Pred Range':>28}  {'FEM Range':>28}"
          f"  {'MAE':>12}")
    print(f"  {'-'*78}")
    for i, label in enumerate(labels):
        p = pred[:, i]
        t = target[:, i]
        mae = (p - t).abs().mean().item()
        print(f"  {label:>4}  [{p.min():>+12.6e}, {p.max():>+12.6e}]"
              f"  [{t.min():>+12.6e}, {t.max():>+12.6e}]"
              f"  {mae:>12.6e}")

    # Residual check
    F_res, M_res = loss_fn.equilibrium_residual(pred, sample)
    print(f"\n  Force  residual norm: {F_res.norm():.6e}")
    print(f"  Moment residual norm: {M_res.norm():.6e}")

    # BC check
    bc_nodes = (sample.bc_disp.squeeze() > 0.5).nonzero().squeeze()
    print(f"\n  Support nodes: {bc_nodes.tolist()}")
    print(f"  max|disp| at supports: "
          f"{pred[bc_nodes, :3].abs().max():.2e}  (should be 0)")

    # ── Save ──
    torch.save(history, "DATA/train_history.pt")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(history['train'], label='Train', alpha=0.8)
        ax.semilogy(history['val'], label='Val', alpha=0.8)
        ax.axvline(best_epoch - 1, color='red', ls='--', alpha=0.5,
                   label=f'Best (epoch {best_epoch})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Physics Loss (log)')
        ax.set_title('PIGNN — Axial + Bending Equilibrium')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('DATA/training_curve.png', dpi=150)
        plt.show()
        print(f"  Saved: DATA/training_curve.png")
    except ImportError:
        pass

    print(f"\n{'='*60}")
    print(f"  DONE ✓")
    print(f"{'='*60}")