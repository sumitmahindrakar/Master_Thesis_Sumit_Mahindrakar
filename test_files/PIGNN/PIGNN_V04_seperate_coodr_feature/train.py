"""
=================================================================
train.py — Train PIGNN with Strong-Form Physics Loss
=================================================================

UNSUPERVISED — No FEM targets used during training.

Loss = PDE residuals only:
    L_axial   = || EA·d²u_x/dx² + f_x ||²
    L_bending = || EI·d⁴u_z/dx⁴ + q_z ||²
    L_kin     = || φ - du_z/dx ||²

FEM data used ONLY for post-training comparison (not for loss).
=================================================================
"""

import os
print(f"Working directory: {os.getcwd()}")

from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)
print(f"Working directory: {os.getcwd()}")

import torch
import time
from torch_geometric.loader import DataLoader
from model import PIGNN
from physics_loss import StrongFormPhysicsLoss


# ================================================================
# TRAINING FUNCTIONS
# ================================================================

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    One training epoch.

    Note: model forward pass happens INSIDE loss_fn
    because autograd.grad needs coords in the computation graph.
    """
    model.train()
    total_loss = 0.0
    comp_sums = {}
    n = 0

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()
        loss, loss_dict, _ = loss_fn(model, data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + v
        n += 1

    avg = total_loss / n
    avg_comp = {k: v / n for k, v in comp_sums.items()}
    return avg, avg_comp


def evaluate(model, loader, loss_fn, device):
    """
    Validation/test evaluation.

    NO torch.no_grad() — autograd.grad inside loss_fn
    requires the computation graph to exist.
    We just don't call .backward() so no weight updates.
    """
    model.eval()
    total_loss = 0.0
    comp_sums = {}
    n = 0

    for data in loader:
        data = data.to(device)
        loss, loss_dict, _ = loss_fn(model, data)
        # No backward — just measure loss
        total_loss += loss.item()
        for k, v in loss_dict.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + v
        n += 1

    avg = total_loss / n
    avg_comp = {k: v / n for k, v in comp_sums.items()}
    return avg, avg_comp


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  TRAIN PIGNN — Strong-Form Physics Loss")
    print("  No FEM targets. Pure PDE residuals.")
    print("=" * 60)

    # ═══════════════════════════════════════════
    # CONFIG
    # ═══════════════════════════════════════════
    HIDDEN_DIM = 128
    N_LAYERS   = 6
    LR         = 1e-4
    EPOCHS     = 50
    BATCH_SIZE = 1        # 1 graph per batch (safest for physics loss)
    DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Physics loss weights
    W_AXIAL   = 1.0
    W_BENDING = 1e-2      # start small — 4th derivative is stiff
    W_KIN     = 0.1

    print(f"\n  Device:      {DEVICE}")
    print(f"  Hidden dim:  {HIDDEN_DIM}")
    print(f"  MP layers:   {N_LAYERS}")
    print(f"  LR:          {LR}")
    print(f"  Epochs:      {EPOCHS}")
    print(f"  Batch size:  {BATCH_SIZE}")
    print(f"  Weights:     axial={W_AXIAL}, "
          f"bending={W_BENDING}, kin={W_KIN}")

    # ═══════════════════════════════════════════
    # LOAD DATA (RAW — not normalized)
    # ═══════════════════════════════════════════
    print(f"\n── Loading RAW graph data ──")
    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    print(f"  {len(data_list)} graphs loaded")

    s = data_list[0]
    print(f"  Nodes:  {s.num_nodes}")
    print(f"  Elems:  {s.n_elements}")
    print(f"  E:      {s.prop_E[0]:.4e}")
    print(f"  A:      {s.prop_A[0]:.4e}")
    print(f"  I22:    {s.prop_I22[0]:.4e}")
    print(f"  Load:   [{s.x[:, 5:8].min():.4e}, "
      f"{s.x[:, 5:8].max():.4e}]")

    # ═══════════════════════════════════════════
    # SPLIT & DATALOADERS
    # ═══════════════════════════════════════════
    n = len(data_list)
    gen = torch.Generator().manual_seed(42)
    idx = torch.randperm(n, generator=gen).tolist()

    n_train = max(1, int(n * 0.7))
    n_val   = max(1, int(n * 0.15))

    train_data = [data_list[i] for i in idx[:n_train]]
    val_data   = [data_list[i] for i in idx[n_train:n_train + n_val]]
    test_data  = [data_list[i] for i in idx[n_train + n_val:]]
    if not test_data:
        test_data = val_data.copy()

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False
    )

    print(f"  Train: {len(train_data)}  "
          f"Val: {len(val_data)}  "
          f"Test: {len(test_data)}")

    # ═══════════════════════════════════════════
    # MODEL
    # ═══════════════════════════════════════════
    model = PIGNN(
        node_in_dim=9,
        edge_in_dim=10,         # ← was 11
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        out_dim=3,             # ← [u_x, u_z, φ]
    ).to(DEVICE)

    print(f"  Parameters: {model.count_params():,}")

    # ═══════════════════════════════════════════
    # LOSS & OPTIMIZER
    # ═══════════════════════════════════════════
    loss_fn = StrongFormPhysicsLoss(
        w_axial=W_AXIAL,
        w_bending=W_BENDING,
        w_kin=W_KIN,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-7
    )

    # ═══════════════════════════════════════════
    # TRAINING LOOP
    # ═══════════════════════════════════════════
    print(f"\n── Training ──")
    header = (f"  {'Ep':>5}  {'Train':>12}  {'Val':>12}  "
              f"{'Axial':>10}  {'Bend':>10}  {'Kin':>10}  "
              f"{'LR':>9}  {'t':>5}")
    print(header)
    print(f"  {'-' * len(header)}")

    best_val = float('inf')
    best_epoch = 0
    history = {
        'train': [], 'val': [],
        'axial': [], 'bending': [], 'kinematic': [],
    }

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_comp = train_one_epoch(
            model, train_loader, optimizer, loss_fn, DEVICE
        )
        val_loss, val_comp = evaluate(
            model, val_loader, loss_fn, DEVICE
        )

        scheduler.step(val_loss)
        dt = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        # History
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['axial'].append(train_comp.get('axial', 0))
        history['bending'].append(train_comp.get('bending', 0))
        history['kinematic'].append(train_comp.get('kinematic', 0))

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "DATA/pignn_best.pt")

        # Print
        if epoch % 20 == 0 or epoch <= 5 or epoch == EPOCHS:
            print(f"  {epoch:>5}  {train_loss:>12.4e}  "
                  f"{val_loss:>12.4e}  "
                  f"{train_comp.get('axial', 0):>10.3e}  "
                  f"{train_comp.get('bending', 0):>10.3e}  "
                  f"{train_comp.get('kinematic', 0):>10.3e}  "
                  f"{lr:>9.1e}  {dt:>4.1f}s")

        # Early NaN detection
        if torch.isnan(torch.tensor(train_loss)):
            print(f"\n  ⚠ NaN at epoch {epoch}! "
                  f"Try reducing LR or w_bending.")
            break

    print(f"\n  Best val: {best_val:.4e} (epoch {best_epoch})")

    # ═══════════════════════════════════════════
    # TEST SET
    # ═══════════════════════════════════════════
    print(f"\n── Test Set ──")
    model.load_state_dict(
        torch.load("DATA/pignn_best.pt", weights_only=False)
    )
    test_loss, test_comp = evaluate(
        model, test_loader, loss_fn, DEVICE
    )
    print(f"  Test physics loss: {test_loss:.4e}")
    for k, v in test_comp.items():
        print(f"    {k:>10}: {v:.4e}")

    # ═══════════════════════════════════════════
    # COMPARE WITH FEM (post-training only)
    # ═══════════════════════════════════════════
    print(f"\n── Post-Training: Compare with FEM ──")
    print(f"  (FEM data NOT used during training)")

    model.eval()
    sample = test_data[0].to(DEVICE)
    pred = model(sample)              # (N, 3)
    target = sample.y_node            # (N, 3) FEM ground truth

    labels = ['u_x', 'u_z', 'φ']
    print(f"\n  {'DOF':>4}  {'Predicted':>28}  "
          f"{'FEM':>28}  {'MAE':>12}")
    print(f"  {'-' * 78}")
    for i, label in enumerate(labels):
        p = pred[:, i].detach()
        t = target[:, i]
        mae = (p - t).abs().mean().item()
        print(f"  {label:>4}  "
              f"[{p.min():>+12.6e}, {p.max():>+12.6e}]  "
              f"[{t.min():>+12.6e}, {t.max():>+12.6e}]  "
              f"{mae:>12.4e}")

    # BC check
    bc_nodes = (sample.bc_disp.squeeze() > 0.5).nonzero().squeeze()
    print(f"\n  Support nodes: {bc_nodes.tolist()}")
    print(f"  max|disp| at supports: "
          f"{pred[bc_nodes, :2].detach().abs().max():.2e}"
          f"  (should be 0)")

    # ═══════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════
    torch.save(history, "DATA/train_history.pt")

    # ═══════════════════════════════════════════
    # PLOT
    # ═══════════════════════════════════════════
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: total loss
        ax = axes[0]
        ax.semilogy(history['train'], label='Train', alpha=0.8)
        ax.semilogy(history['val'], label='Val', alpha=0.8)
        ax.axvline(best_epoch - 1, color='red', ls='--',
                   alpha=0.5, label=f'Best (ep {best_epoch})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Physics Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: component losses
        ax = axes[1]
        ax.semilogy(history['axial'], label='Axial', alpha=0.8)
        ax.semilogy(history['bending'], label='Bending', alpha=0.8)
        ax.semilogy(history['kinematic'], label='Kinematic',
                    alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Component Loss')
        ax.set_title('Physics Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('PIGNN — Strong-Form Physics Loss '
                     '(Unsupervised)', fontsize=13)
        plt.tight_layout()
        plt.savefig('DATA/training_curve.png', dpi=150)
        plt.show()
        print(f"  Saved: DATA/training_curve.png")
    except ImportError:
        pass

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE ✓")
    print(f"  Best val loss: {best_val:.4e} (epoch {best_epoch})")
    print(f"  Model saved:   DATA/pignn_best.pt")
    print(f"{'=' * 60}")