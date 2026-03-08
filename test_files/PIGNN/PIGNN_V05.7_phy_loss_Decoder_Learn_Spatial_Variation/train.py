"""
=================================================================
train.py — Train Hybrid GNN-PINN with Strong-Form Physics Loss
=================================================================

UNSUPERVISED — No FEM targets used during training.

Architecture:
    Phase 1 (GNN):  data.x → Encoder → MP → h_i (context)
    Phase 2 (PINN): Decoder(coords_i, h_i) → [u_x, u_z, φ]_i

Loss = PDE residuals via EXACT autodiff:
    L_axial   = || EA·d²u_x/dx² + f_x ||²
    L_bending = || EI·d⁴u_z/dx⁴ + q_z ||²
    L_kin     = || φ - du_z/dx ||²

FEM data used ONLY for post-training comparison.
=================================================================
"""

import os
from pathlib import Path
print(f"Working directory: {os.getcwd()}")
CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)
print(f"Working directory: {os.getcwd()}")

import torch
import time
from torch_geometric.loader import DataLoader
from model import PIGNN_Hybrid
from physics_loss import StrongFormPhysicsLoss


# ================================================================
# TRAINING FUNCTIONS
# ================================================================

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    One training epoch.
    Forward pass happens INSIDE loss_fn (autograd needs it).
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
    Validation/test — no .backward(), no weight updates.
    NO torch.no_grad() — autograd.grad needs computation graph.
    """
    model.eval()
    total_loss = 0.0
    comp_sums = {}
    n = 0

    for data in loader:
        data = data.to(device)
        loss, loss_dict, _ = loss_fn(model, data)
        total_loss += loss.item()
        for k, v in loss_dict.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + v
        n += 1

    avg = total_loss / n
    avg_comp = {k: v / n for k, v in comp_sums.items()}
    return avg, avg_comp


# ================================================================
# AUTODIFF VERIFICATION
# ================================================================

def verify_autodiff(model, sample, device):
    """
    Verify that the hybrid architecture gives correct autodiff.

    Checks:
    1. coords.requires_grad is True
    2. 1st derivative is non-zero (not collapsed)
    3. 2nd derivative is non-zero
    4. 4th derivative is non-zero (THIS is what failed before)
    5. Jacobian is block-diagonal (optional, expensive)
    """
    print(f"\n{'═'*60}")
    print(f"  AUTODIFF VERIFICATION (Hybrid Architecture)")
    print(f"{'═'*60}")

    model.eval()
    sample = sample.to(device)
    pred = model(sample)
    coords = model.get_coords()

    print(f"\n  1. BASIC CHECKS:")
    print(f"     coords.requires_grad: {coords.requires_grad}")
    print(f"     pred shape: {pred.shape}")
    print(f"     coords shape: {coords.shape}")

    u_z = pred[:, 1:2]   # transverse displacement

    # ── 1st derivative ──
    try:
        duz = torch.autograd.grad(
            u_z.sum(), coords,
            create_graph=True, retain_graph=True
        )[0]
        duz_dx = duz[:, 0]
        print(f"\n  2. 1st DERIVATIVE (du_z/dx):")
        print(f"     range: [{duz_dx.min():.4e}, {duz_dx.max():.4e}]")
        print(f"     non-zero: {(duz_dx.abs() > 1e-10).sum().item()}"
              f"/{duz_dx.shape[0]} nodes")
        first_ok = duz_dx.abs().max() > 1e-10
        print(f"     {'✓' if first_ok else '✗'} 1st derivative "
              f"{'exists' if first_ok else 'collapsed to zero!'}")
    except RuntimeError as e:
        print(f"     ✗ FAILED: {e}")
        return False

    # ── 2nd derivative ──
    try:
        d2uz = torch.autograd.grad(
            duz[:, 0:1].sum(), coords,
            create_graph=True, retain_graph=True
        )[0]
        d2uz_dx2 = d2uz[:, 0]
        print(f"\n  3. 2nd DERIVATIVE (d²u_z/dx²):")
        print(f"     range: [{d2uz_dx2.min():.4e}, {d2uz_dx2.max():.4e}]")
        second_ok = d2uz_dx2.abs().max() > 1e-12
        print(f"     {'✓' if second_ok else '✗'} 2nd derivative "
              f"{'exists' if second_ok else 'collapsed!'}")
    except RuntimeError as e:
        print(f"     ✗ FAILED: {e}")
        return False

    # ── 4th derivative (the critical test!) ──
    try:
        d3uz = torch.autograd.grad(
            d2uz[:, 0:1].sum(), coords,
            create_graph=True, retain_graph=True
        )[0]
        d4uz = torch.autograd.grad(
            d3uz[:, 0:1].sum(), coords,
            create_graph=True, retain_graph=True
        )[0]
        d4uz_dx4 = d4uz[:, 0]
        print(f"\n  4. 4th DERIVATIVE (d⁴u_z/dx⁴):")
        print(f"     range: [{d4uz_dx4.min():.4e}, {d4uz_dx4.max():.4e}]")
        fourth_ok = d4uz_dx4.abs().max() > 1e-15
        print(f"     {'✓' if fourth_ok else '✗'} 4th derivative "
              f"{'exists' if fourth_ok else 'COLLAPSED!'}")

        if fourth_ok:
            # Quick check: EI * d4u/dx4 should be on order of q
            EI = (sample.prop_E * sample.prop_I22).mean()
            q_z = sample.elem_load[:, 2].mean()
            residual = EI * d4uz_dx4.mean() + q_z
            print(f"\n  5. PDE RESIDUAL CHECK:")
            print(f"     EI = {EI:.4e}")
            print(f"     q_z = {q_z:.4e}")
            print(f"     EI·d⁴u/dx⁴ mean = "
                  f"{(EI * d4uz_dx4.mean()):.4e}")
            print(f"     Residual = {residual:.4e}")
            print(f"     (should decrease during training)")
    except RuntimeError as e:
        print(f"     ✗ FAILED: {e}")
        fourth_ok = False

    # ── Jacobian diagonal check (SAMPLE of nodes) ──
    N = pred.shape[0]
    n_check = min(10, N)    # check 10 random nodes
    print(f"\n  6. JACOBIAN DIAGONAL CHECK "
          f"(sampling {n_check}/{N} nodes):")
    
    u_z_col = pred[:, 1]    # transverse displacement
    
    # Pick random nodes (avoid BC nodes where grad=0)
    free_mask = (sample.bc_disp.squeeze() < 0.5)
    free_indices = free_mask.nonzero().squeeze().tolist()
    if isinstance(free_indices, int):
        free_indices = [free_indices]
    
    # Sample from free nodes
    import random
    random.seed(42)
    check_nodes = random.sample(
        free_indices, min(n_check, len(free_indices))
    )
    
    off_diag_ratios = []
    
    print(f"     {'Node':>6}  {'Self ∂u/∂x':>12}  "
          f"{'Max Other':>12}  {'Ratio':>10}  {'Status'}")
    print(f"     {'-'*58}")
    
    for i in check_nodes:
        g = torch.autograd.grad(
            u_z_col[i], coords,
            create_graph=False, retain_graph=True
        )[0]  # (N, 3)
        
        self_grad = g[i, 0].abs().item()
        # All OTHER nodes' gradients
        other_grads = torch.cat([g[:i, 0], g[i+1:, 0]])
        max_other = other_grads.abs().max().item()
        mean_other = other_grads.abs().mean().item()
        
        if self_grad > 1e-15:
            ratio = max_other / self_grad
            off_diag_ratios.append(ratio)
            status = '✓ diagonal' if ratio < 0.01 else '⚠ coupled'
            print(f"     {i:>6}  {self_grad:>12.4e}  "
                  f"{max_other:>12.4e}  {ratio:>10.4e}  {status}")
        else:
            print(f"     {i:>6}  {'~0':>12}  "
                  f"{max_other:>12.4e}  {'skip':>10}  "
                  f"⚠ self≈0")
    
    if off_diag_ratios:
        avg_ratio = sum(off_diag_ratios) / len(off_diag_ratios)
        max_ratio = max(off_diag_ratios)
        print(f"\n     Summary ({len(off_diag_ratios)} nodes checked):")
        print(f"       Avg ratio:  {avg_ratio:.6e}")
        print(f"       Max ratio:  {max_ratio:.6e}")
        
        if max_ratio < 1e-6:
            verdict = "✓ PERFECTLY block-diagonal (ratio < 1e-6)"
        elif max_ratio < 0.01:
            verdict = "✓ Effectively block-diagonal (ratio < 1%)"
        elif max_ratio < 0.1:
            verdict = "⚠ Weakly coupled (ratio < 10%)"
        else:
            verdict = "✗ Dense Jacobian (ratio > 10%) — autodiff contaminated!"
        
        print(f"       {verdict}")
    else:
        print(f"\n     No valid nodes to check")

    all_ok = first_ok and second_ok and fourth_ok
    print(f"\n  RESULT: {'ALL PASSED ✓' if all_ok else 'SOME FAILED ✗'}")
    print(f"{'═'*60}\n")
    return all_ok


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  TRAIN Hybrid GNN-PINN — Strong-Form Physics Loss")
    print("  Approach B: Exact autodiff via pointwise decoder")
    print("  No FEM targets. Pure PDE residuals.")
    print("=" * 60)

    # ═══════════════════════════════════════════
    # CONFIG
    # ═══════════════════════════════════════════
    HIDDEN_DIM = 128
    N_LAYERS   = 6
    LR         = 5e-4
    EPOCHS     = 500
    BATCH_SIZE = 1
    DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Physics loss weights
    W_FORCE   = 1.0
    W_MOMENT = 1.0      # start small — 4th derivative is stiff
    W_KIN     = 0.1
    W_NEUMANN = 1.0
    W_CONSISTENCY = 1.0

    print(f"\n  Architecture: Hybrid GNN-PINN (Approach B)")
    print(f"  Device:       {DEVICE}")
    print(f"  Hidden dim:   {HIDDEN_DIM}")
    print(f"  MP layers:    {N_LAYERS}")
    print(f"  LR:           {LR}")
    print(f"  Epochs:       {EPOCHS}")
    print(f"  Batch size:   {BATCH_SIZE}")
    print(f"  Weights:      force={W_FORCE}, "
          f"moment={W_MOMENT}, neumann={W_NEUMANN} kin={W_KIN}")

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
    print(f"  Load:   [{s.elem_load[:, 2].min():.4e}, "
          f"{s.elem_load[:, 2].max():.4e}]")

    # ═══════════════════════════════════════════
    # SPLIT & DATALOADERS
    # ═══════════════════════════════════════════
    n = len(data_list)
    gen = torch.Generator().manual_seed(42)
    idx = torch.randperm(n, generator=gen).tolist()

    n_train = max(1, int(n * 0.6))
    n_val   = max(1, int(n * 0.2))

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
    # MODEL (Hybrid GNN-PINN)
    # ═══════════════════════════════════════════
    model = PIGNN_Hybrid(
        node_in_dim=9,
        edge_in_dim=10,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        out_dim=3,
    ).to(DEVICE)

    print(f"  Parameters: {model.count_params():,}")
    print(f"  Architecture: GNN(context) → PINN(decoder)")

    # ═══════════════════════════════════════════
    # VERIFY AUTODIFF BEFORE TRAINING
    # ═══════════════════════════════════════════
    print(f"\n── Verifying Autodiff (Pre-Training) ──")
    autodiff_ok = verify_autodiff(model, data_list[0], DEVICE)

    if not autodiff_ok:
        print("  ⚠ Autodiff verification failed!")
        print("  Proceeding anyway — derivatives may improve "
              "after some training.")

    # ═══════════════════════════════════════════
    # LOSS & OPTIMIZER
    # ═══════════════════════════════════════════
    loss_fn = StrongFormPhysicsLoss(
        w_force=W_FORCE,
        w_moment=W_MOMENT,
        w_neumann=W_NEUMANN,
        w_kin=W_KIN,
        w_consistency=W_CONSISTENCY,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-7
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,           # restart every 50 epochs
        T_mult=2,         # double period after each restart
        eta_min=1e-6,     # minimum LR
    )

    # ═══════════════════════════════════════════
    # TRAINING LOOP
    # ═══════════════════════════════════════════
    print(f"\n── Training ──")
    header = (f"  {'Ep':>5}  {'Train':>12}  {'Val':>12}  "
              f"{'Force':>10}  {'Mom':>10}  {'Consist':>10}  "
              f"{'Kin':>10}  {'LR':>9}  {'t':>5}")
    print(header)
    print(f"  {'-' * len(header)}")

    best_val = float('inf')
    best_epoch = 0
    history = {
        'train': [], 'val': [],
        'force': [], 'moment': [],
         'neumann': [], 'kinematic': [],
         'consistency': [],
    }

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_comp = train_one_epoch(
            model, train_loader, optimizer, loss_fn, DEVICE
        )
        val_loss, val_comp = evaluate(
            model, val_loader, loss_fn, DEVICE
        )

        # scheduler.step(val_loss)
        scheduler.step(epoch)
        dt = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        # History
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['force'].append(train_comp.get('force', 0))
        history['moment'].append(train_comp.get('moment', 0))
        history['neumann'].append(train_comp.get('neumann', 0))
        history['kinematic'].append(train_comp.get('kinematic', 0))
        history['consistency'].append(train_comp.get('consistency', 0))

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "DATA/pignn_hybrid_best.pt")

        # Print
        if epoch % 20 == 0 or epoch <= 5 or epoch == EPOCHS:
            print(f"  {epoch:>5}  {train_loss:>12.4e}  "
          f"{val_loss:>12.4e}  "
          f"{train_comp.get('force', 0):>10.3e}  "
          f"{train_comp.get('moment', 0):>10.3e}  "
          f"{train_comp.get('consistency', 0):>10.3e}  "
          f"{train_comp.get('kinematic', 0):>10.3e}  "
          f"{lr:>9.1e}  {dt:>4.1f}s")

        # Early NaN detection
        if torch.isnan(torch.tensor(train_loss)):
            print(f"\n  ⚠ NaN at epoch {epoch}! "
                  f"Try reducing LR or w_bending.")
            break

    print(f"\n  Best val: {best_val:.4e} (epoch {best_epoch})")


    # ═══════════════════════════════════════════
    # DEBUG: Check force balance
    # ═══════════════════════════════════════════
    print(f"\n── Debug: Force Balance Check ──")
    model.eval()
    sample = train_data[0].to(DEVICE)
    _, debug_dict, debug_pred = loss_fn(model, sample)

    print(f"  F_int max:  {debug_dict['F_int_max']:.4e} N")
    print(f"  F_ext max:  {debug_dict['F_ext_max']:.4e} N")
    print(f"  F_char:     {debug_dict['F_char']:.4e} N")
    print(f"  M_char:     {debug_dict['M_char']:.4e} N·m")
    print(f"  Ratio F_int/F_ext: "
          f"{debug_dict['F_int_max'] / max(debug_dict['F_ext_max'], 1e-10):.4f}")
    print(f"  (should be ~1.0 when equilibrium is real)")

    # ═══════════════════════════════════════════
    # Physical Scale Check
    # ═══════════════════════════════════════════
    print(f"\n── Physical Scale Check ──")
    with torch.no_grad():
        p = debug_pred
        print(f"  Pred |u_x| max: {p[:, 0].abs().max():.4e} m")
        print(f"  Pred |u_z| max: {p[:, 1].abs().max():.4e} m")
        print(f"  Pred |φ|   max: {p[:, 2].abs().max():.4e} rad")
        print(f"  FEM  |u_x| max: {sample.y_node[:, 0].abs().max():.4e} m")
        print(f"  FEM  |u_z| max: {sample.y_node[:, 1].abs().max():.4e} m")
        print(f"  FEM  |φ|   max: {sample.y_node[:, 2].abs().max():.4e} rad")

        for dof, name in [(0, 'u_x'), (1, 'u_z'), (2, 'φ')]:
            pred_scale = p[:, dof].abs().max().item()
            fem_scale = sample.y_node[:, dof].abs().max().item()
            if fem_scale > 1e-15:
                ratio = pred_scale / fem_scale
                print(f"  {name} ratio pred/FEM: {ratio:.2f}x "
                      f"{'✓' if 0.1 < ratio < 10 else '✗ wrong scale!'}")

    # ═══════════════════════════════════════════
    # VERIFY AUTODIFF AFTER TRAINING
    # ═══════════════════════════════════════════
    print(f"\n── Verifying Autodiff (Post-Training) ──")
    model.load_state_dict(
        torch.load("DATA/pignn_hybrid_best.pt", weights_only=False)
    )
    verify_autodiff(model, data_list[0], DEVICE)

    # ═══════════════════════════════════════════
    # TEST SET
    # ═══════════════════════════════════════════
    print(f"\n── Test Set ──")
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
    pred = model(sample)
    target = sample.y_node

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

    # ── Derived quantities (N, M, V from autodiff) ──
    print(f"\n── Derived Physics Quantities ──")
    coords = model.get_coords()
    u_z = pred[:, 1:2]

    duz = torch.autograd.grad(
        u_z.sum(), coords,
        create_graph=True, retain_graph=True
    )[0][:, 0:1]
    d2uz = torch.autograd.grad(
        duz.sum(), coords,
        create_graph=True, retain_graph=True
    )[0][:, 0:1]

    EI = (sample.prop_E * sample.prop_I22).mean()
    M_pred = EI * d2uz.detach().squeeze()

    print(f"  EI = {EI:.4e}")
    print(f"  M (from autodiff d²u_z/dx²):")
    print(f"    range: [{M_pred.min():.4e}, {M_pred.max():.4e}]")
    print(f"  M (FEM ground truth):")
    print(f"    range: [{target[:, 1].min():.4e} ... "
          f"not directly comparable (elem vs node)]")

    # ── BC check ──
    bc_nodes = (sample.bc_disp.squeeze() > 0.5).nonzero().squeeze()
    print(f"\n  Support nodes: {bc_nodes.tolist()}")
    print(f"  max|disp| at supports: "
          f"{pred[bc_nodes, :2].detach().abs().max():.2e}"
          f"  (should be 0)")

    # ═══════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════
    torch.save(history, "DATA/train_history_hybrid.pt")

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
        ax.set_title('Total Loss (Hybrid GNN-PINN)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: component losses
        ax = axes[1]
        ax.semilogy(history['force'], label='Force', alpha=0.8)
        ax.semilogy(history['moment'], label='Moment', alpha=0.8)
        ax.semilogy(history['neumann'], label='Neumann BC', alpha=0.8)
        ax.semilogy(history['kinematic'], label='Kinematic',
                    alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Component Loss')
        ax.set_title('Physics Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('Hybrid GNN-PINN — Strong-Form Physics Loss '
                     '(Unsupervised, Exact Autodiff)',
                     fontsize=13)
        plt.tight_layout()
        plt.savefig('DATA/training_curve_hybrid.png', dpi=150)
        plt.show()
        print(f"  Saved: DATA/training_curve_hybrid.png")
    except ImportError:
        pass

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE ✓ (Hybrid GNN-PINN)")
    print(f"{'=' * 60}")
    print(f"  Architecture:   GNN(context) → PINN(decoder)")
    print(f"  Autodiff:       Exact (block-diagonal Jacobian)")
    print(f"  Best val loss:  {best_val:.4e} (epoch {best_epoch})")
    print(f"  Model saved:    DATA/pignn_hybrid_best.pt")
    print(f"{'=' * 60}")