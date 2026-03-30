"""
=================================================================
train.py — L-BFGS + Adam Hybrid Training for PIGNN v2
=================================================================
Phase 0: Gradient diagnosis (verify init fix works)
Phase 1: L-BFGS with reduced axial weight (find basin)
Phase 2: Adam fine-tune with axial weight ramp (polish)
=================================================================
"""

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Batch
import numpy as np
import time
import os
import pickle

import torch
from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

from model import PIGNN
from energy_loss import FrameEnergyLoss
from step_2_grapg_constr import FrameData


# ════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════

class Config:
    # Model
    hidden_dim   = 128
    n_layers     = 10          # was 6 — need full graph coverage
    init_gain    = 0.01        # was 0 — critical fix

    # Phase 1: L-BFGS
    lbfgs_steps   = 2000
    lbfgs_lr      = 1.0        # line search handles actual step
    lbfgs_history = 100
    lbfgs_max_iter = 20        # inner CG iterations per step

    # Phase 2: Adam
    adam_epochs   = 3000
    adam_lr       = 1e-5
    adam_lr_min   = 1e-7
    ramp_epochs   = 0       # aw ramp duration
    grad_clip     = 10.0

    # General
    train_ratio  = 0.85
    log_every    = 20          # L-BFGS
    log_every_adam = 200       # Adam
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir     = 'RESULTS'


# ════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════

def get_scalar(attr):
    """Extract scalar from tensor or float."""
    if isinstance(attr, torch.Tensor):
        return attr.item() if attr.numel() == 1 else attr[0].item()
    return float(attr)


def compute_disp_error(pred_raw, batch):
    """
    Relative displacement error: ||pred - true|| / ||true||.
    Returns: (err_total, err_ux, err_uz, err_th)
    """
    with torch.no_grad():
        if hasattr(batch, 'batch') and batch.batch is not None:
            ux_c  = batch.ux_c[batch.batch]
            uz_c  = batch.uz_c[batch.batch]
            th_c  = batch.theta_c[batch.batch]
        else:
            ux_c  = batch.ux_c
            uz_c  = batch.uz_c
            th_c  = batch.theta_c

        pred_phys = torch.stack([
            pred_raw[:, 0] * ux_c,
            pred_raw[:, 1] * uz_c,
            pred_raw[:, 2] * th_c,
        ], dim=1)

        true_phys = batch.y_node
        eps = 1e-12

        diff = pred_phys - true_phys
        err_total = diff.norm() / true_phys.norm().clamp(min=eps)

        err_ux = diff[:, 0].norm() / true_phys[:, 0].norm().clamp(min=eps)
        err_uz = diff[:, 1].norm() / true_phys[:, 1].norm().clamp(min=eps)
        err_th = diff[:, 2].norm() / true_phys[:, 2].norm().clamp(min=eps)

    return err_total.item(), err_ux.item(), err_uz.item(), err_th.item()


def compute_optimal_aw(data):
    """Compute optimal axial weight from element properties."""
    EA = data.prop_E * data.prop_A
    EI = data.prop_E * data.prop_I22
    L  = data.elem_lengths
    aw = (12.0 * EI / (EA * L**2)).mean().item()
    return aw


def ensure_forward_mask(data_list):
    """Add edge_forward_mask if missing."""
    for d in data_list:
        if not hasattr(d, 'edge_forward_mask'):
            E = d.n_elements
            d.edge_forward_mask = torch.cat([
                torch.ones(E, dtype=torch.bool),
                torch.zeros(E, dtype=torch.bool)
            ])


# ════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════

def load_data():
    """
    Load graph data. Tries in order:
      1. DATA/graph_dataset_norm.pt (pre-processed)
      2. DATA/graph_dataset.pt (unnormalized)
      3. DATA/frame_dataset.pkl (raw → build)
    """
    paths = [
        "DATA/graph_dataset_norm.pt",
        "DATA/graph_dataset.pt",
    ]

    for p in paths:
        if os.path.exists(p):
            data_list = torch.load(p, weights_only=False)
            print(f"  Loaded {len(data_list)} graphs from {p}")
            return data_list

    # Build from pickle
    pkl_path = "DATA/frame_dataset.pkl"
    if os.path.exists(pkl_path):
        from step_2_graph_constr import FrameGraphBuilder
        from normalizer import PhysicsScaler, MinMaxNormalizer

        with open(pkl_path, "rb") as f:
            dataset = pickle.load(f)

        builder = FrameGraphBuilder()
        data_list = builder.build_dataset(dataset)
        data_list = PhysicsScaler.compute_and_store_list(data_list)

        normalizer = MinMaxNormalizer()
        normalizer.fit(data_list)
        data_list = normalizer.transform_list(data_list)

        print(f"  Built {len(data_list)} graphs from {pkl_path}")
        return data_list

    raise FileNotFoundError(
        "No data found! Run step_1 and step_2 first."
    )


# ════════════════════════════════════════════════
# Main training
# ════════════════════════════════════════════════

def main():
    cfg = Config()
    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("=" * 70)
    print("  PIGNN TRAINING — L-BFGS + Adam Hybrid v2")
    print("=" * 70)

    # ────────────────────────────────────────
    # Load data
    # ────────────────────────────────────────
    print("\n── Loading data ──")
    data_list = load_data()

    # Verify required attributes
    d0 = data_list[0]
    for attr in ['ux_c', 'uz_c', 'theta_c', 'F_c']:
        assert hasattr(d0, attr), \
            f"Missing {attr}! Run PhysicsScaler first."

    ensure_forward_mask(data_list)

    # Extract scales
    ux_c    = get_scalar(d0.ux_c)
    uz_c    = get_scalar(d0.uz_c)
    theta_c = get_scalar(d0.theta_c)
    F_c     = get_scalar(d0.F_c)
    E_c     = F_c * ux_c
    N       = d0.num_nodes
    E       = d0.n_elements

    print(f"  Nodes: {N}, Elements: {E}")
    print(f"  ux_c   = {ux_c:.4e}")
    print(f"  uz_c   = {uz_c:.4e}")
    print(f"  θ_c    = {theta_c:.4e}")
    print(f"  F_c    = {F_c:.4e}")
    print(f"  E_c    = {E_c:.4e}")
    print(f"  ux_c/uz_c = {ux_c/uz_c:.1f}×")

    # Optimal axial weight
    optimal_aw = compute_optimal_aw(d0)
    EA_L_avg  = (d0.prop_E * d0.prop_A / d0.elem_lengths).mean().item()
    EI_L3_avg = (d0.prop_E * d0.prop_I22 / d0.elem_lengths**3).mean().item()
    cond_full = EA_L_avg / (12 * EI_L3_avg)

    print(f"  Optimal aw   = {optimal_aw:.4e}")
    print(f"  Cond(full)   = {cond_full:.0f}")
    print(f"  Cond(aw_opt) ≈ {max(1, cond_full * optimal_aw):.0f}")

    # Train/test split
    n_total = len(data_list)
    n_train = max(1, int(cfg.train_ratio * n_total))
    train_graphs = data_list[:n_train]
    test_graphs  = (data_list[n_train:]
                    if n_train < n_total
                    else data_list[-2:])
    print(f"  Split: {len(train_graphs)} train, "
          f"{len(test_graphs)} test")

    # Full batch (required for L-BFGS, fine for Adam too)
    train_batch = Batch.from_data_list(train_graphs).to(device)
    test_batch  = Batch.from_data_list(test_graphs).to(device)

    # ────────────────────────────────────────
    # Create model
    # ────────────────────────────────────────
    print("\n── Creating model ──")
    model = PIGNN(
        node_in_dim=10,
        edge_in_dim=7,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        decoder_init_gain=cfg.init_gain,
    ).to(device)
    model.summary()

    loss_fn = FrameEnergyLoss().to(device)

    # ────────────────────────────────────────
    # Phase 0: Diagnostics
    # ────────────────────────────────────────
    print("── Phase 0: Diagnostics ──")

    # Check initial predictions
    model.eval()
    with torch.no_grad():
        init_pred = model(train_batch)
        p_min = init_pred.min().item()
        p_max = init_pred.max().item()
    print(f"  Initial pred range: [{p_min:.6e}, {p_max:.6e}]")
    assert abs(p_max) > 1e-8 or abs(p_min) > 1e-8, \
        "Predictions are still zero — init fix failed!"
    print(f"  ✓ Non-zero initial predictions")

    # Check gradient coverage
    model.train()
    pred = model(train_batch)
    loss, _, _, _ = loss_fn.compute(
        pred, train_batch, axial_weight=optimal_aw
    )
    loss.backward()

    total_p = 0
    grad_p  = 0
    for p in model.parameters():
        total_p += 1
        if p.grad is not None and p.grad.abs().max() > 0:
            grad_p += 1
    pct = 100 * grad_p / total_p
    print(f"  Gradient coverage: {grad_p}/{total_p} "
          f"({pct:.0f}%)")
    assert pct > 80, \
        f"Only {pct:.0f}% of params have gradient!"
    print(f"  ✓ Good gradient flow")
    model.zero_grad()

    # Target energy from true displacements
    with torch.no_grad():
        true_d = train_batch.y_node.to(device)
        if hasattr(train_batch, 'batch') \
                and train_batch.batch is not None:
            ux_c_t = train_batch.ux_c[train_batch.batch]
            uz_c_t = train_batch.uz_c[train_batch.batch]
            th_c_t = train_batch.theta_c[train_batch.batch]
        else:
            ux_c_t = train_batch.ux_c
            uz_c_t = train_batch.uz_c
            th_c_t = train_batch.theta_c

        true_raw = torch.stack([
            true_d[:, 0] / ux_c_t,
            true_d[:, 1] / uz_c_t,
            true_d[:, 2] / th_c_t,
        ], dim=1)
        # Handle 0/0
        true_raw = torch.nan_to_num(true_raw, 0.0)

        _, tgt_dict, _, _ = loss_fn.compute(
            true_raw, train_batch, axial_weight=1.0
        )
        target_Pi = tgt_dict['Pi']
        print(f"  Target Π/E_c  = {target_Pi:.4e}")
        print(f"  Target U/W    = {tgt_dict['U_over_W']:.4f}"
              f" (should ≈ 0.50)")

    # ════════════════════════════════════════
    # Phase 1: L-BFGS
    # ════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print(f"  PHASE 1: L-BFGS  |  aw={optimal_aw:.2e}  |"
          f"  {cfg.lbfgs_steps} steps")
    print(f"{'═' * 70}")

    history = {
        'step': [], 'loss': [], 'err': [],
        'grad_norm': [], 'phase': [], 'aw': [],
    }
    best_err   = 999.
    best_state = None
    t0 = time.time()

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=cfg.lbfgs_lr,
        max_iter=cfg.lbfgs_max_iter,
        history_size=cfg.lbfgs_history,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    model.train()
    lbfgs_failed = False

    for step in range(cfg.lbfgs_steps):
        loss_val = [0.0]

        def closure():
            optimizer.zero_grad()
            pred = model(train_batch)
            loss, _, _, _ = loss_fn.compute(
                pred, train_batch,
                axial_weight=optimal_aw
            )
            loss.backward()
            loss_val[0] = loss.item()
            return loss

        try:
            optimizer.step(closure)
        except Exception as e:
            print(f"  Step {step}: L-BFGS failed — {e}")
            lbfgs_failed = True
            break

        # NaN check
        if not np.isfinite(loss_val[0]):
            print(f"  Step {step}: NaN/Inf — stopping L-BFGS")
            lbfgs_failed = True
            break

        # Logging
        if step % cfg.log_every == 0 or step < 5:
            with torch.no_grad():
                pred = model(train_batch)
            err = compute_disp_error(pred, train_batch)
            gn = sum(
                p.grad.norm()**2
                for p in model.parameters()
                if p.grad is not None
            ).sqrt().item()

            history['step'].append(step)
            history['loss'].append(loss_val[0])
            history['err'].append(err[0])
            history['grad_norm'].append(gn)
            history['phase'].append('lbfgs')
            history['aw'].append(optimal_aw)

            if err[0] < best_err:
                best_err = err[0]
                best_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }

            print(
                f"  {step:4d} | "
                f"Π={loss_val[0]:+.4e} | "
                f"err=[{err[1]:.4f},{err[2]:.4f},{err[3]:.4f}]"
                f" tot={err[0]:.4f} | "
                f"|∇|={gn:.2e} | "
                f"pred=[{pred.min().item():.3f},"
                f"{pred.max().item():.3f}]"
            )

        # Early stopping
        if best_err < 0.01:
            print(f"  ✓ Phase 1 converged (err < 1%)")
            break

    t_phase1 = time.time() - t0

    if lbfgs_failed and best_state is not None:
        print("  Restoring best state before failure")
        model.load_state_dict(best_state)

    print(f"\n  Phase 1: {t_phase1:.1f}s, "
          f"best_err={best_err:.4f}")

    # # ════════════════════════════════════════
    # # Phase 2: Adam with axial weight ramp
    # # ════════════════════════════════════════
    # aw_start = optimal_aw
    # aw_end   = min(optimal_aw * 100, 1.0)

    # print(f"\n{'═' * 70}")
    # print(f"  PHASE 2: Adam  |  aw ramp "
    #       f"{aw_start:.1e}→{aw_end:.1e}  |"
    #       f"  {cfg.adam_epochs} epochs")
    # print(f"{'═' * 70}")

    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=cfg.adam_lr
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=cfg.adam_epochs,
    #     eta_min=cfg.adam_lr_min
    # )

    # t0 = time.time()
    # model.train()

    # for epoch in range(cfg.adam_epochs):
    #     # Axial weight ramp (log-linear)
    #     t = min(epoch / max(cfg.ramp_epochs, 1), 1.0)
    #     aw = aw_start * ((aw_end / aw_start) ** t)

    #     optimizer.zero_grad()
    #     pred = model(train_batch)
    #     loss, ld, _, _ = loss_fn.compute(
    #         pred, train_batch, axial_weight=aw
    #     )
    #     loss.backward()
    #     clip_grad_norm_(model.parameters(), cfg.grad_clip)
    #     optimizer.step()
    #     scheduler.step()

    #     # Logging
    #     if (epoch % cfg.log_every_adam == 0
    #             or epoch == cfg.adam_epochs - 1):
    #         with torch.no_grad():
    #             pred = model(train_batch)
    #         err = compute_disp_error(pred, train_batch)
    #         gn = sum(
    #             p.grad.norm()**2
    #             for p in model.parameters()
    #             if p.grad is not None
    #         ).sqrt().item()
    #         lr_now = optimizer.param_groups[0]['lr']

    #         step_global = cfg.lbfgs_steps + epoch
    #         history['step'].append(step_global)
    #         history['loss'].append(loss.item())
    #         history['err'].append(err[0])
    #         history['grad_norm'].append(gn)
    #         history['phase'].append('adam')
    #         history['aw'].append(aw)

    #         if err[0] < best_err:
    #             best_err = err[0]
    #             best_state = {
    #                 k: v.cpu().clone()
    #                 for k, v in model.state_dict().items()
    #             }

    #         print(
    #             f"  {epoch:5d} | "
    #             f"Π={loss.item():+.4e} | "
    #             f"aw={aw:.2e} | "
    #             f"err=[{err[1]:.4f},{err[2]:.4f},{err[3]:.4f}]"
    #             f" tot={err[0]:.4f} | "
    #             f"|∇|={gn:.2e} | lr={lr_now:.1e} | "
    #             f"U/W={ld['U_over_W']:.3f}"
    #         )

    #     # Early stopping
    #     if best_err < 0.005:
    #         print(f"  ✓ Phase 2 converged (err < 0.5%)")
    #         break

    # t_phase2 = time.time() - t0

    # ════════════════════════════════════════
    # Phase 2: Adam fine-tune (GENTLE)
    # ════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print(f"  PHASE 2: Adam fine-tune  |  aw={optimal_aw:.2e}"
          f" (constant)  |  {cfg.adam_epochs} epochs")
    print(f"{'═' * 70}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.adam_lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.adam_epochs,
        eta_min=cfg.adam_lr_min
    )

    t0 = time.time()
    model.train()

    for epoch in range(cfg.adam_epochs):
        aw = optimal_aw  # CONSTANT — no ramp

        optimizer.zero_grad()
        pred = model(train_batch)
        loss, ld, _, _ = loss_fn.compute(
            pred, train_batch, axial_weight=aw
        )
        loss.backward()

        # Check if gradient is finite before stepping
        gn_raw = sum(
            p.grad.norm()**2
            for p in model.parameters()
            if p.grad is not None
        ).sqrt().item()

        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        # Logging
        if (epoch % cfg.log_every_adam == 0
                or epoch == cfg.adam_epochs - 1):
            with torch.no_grad():
                pred = model(train_batch)
            err = compute_disp_error(pred, train_batch)
            lr_now = optimizer.param_groups[0]['lr']

            step_global = cfg.lbfgs_steps + epoch
            history['step'].append(step_global)
            history['loss'].append(loss.item())
            history['err'].append(err[0])
            history['grad_norm'].append(gn_raw)
            history['phase'].append('adam')
            history['aw'].append(aw)

            if err[0] < best_err:
                best_err = err[0]
                best_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }

            clipped = "CLIP" if gn_raw > cfg.grad_clip else "    "
            print(
                f"  {epoch:5d} | "
                f"Π={loss.item():+.4e} | "
                f"err=[{err[1]:.4f},{err[2]:.4f},{err[3]:.4f}]"
                f" tot={err[0]:.4f} | "
                f"|∇|={gn_raw:.2e} {clipped} | "
                f"lr={lr_now:.1e} | "
                f"U/W={ld['U_over_W']:.3f}"
            )

        if best_err < 0.005:
            print(f"  ✓ Converged (err < 0.5%)")
            break

    t_phase2 = time.time() - t0

    # ════════════════════════════════════════
    # Final evaluation
    # ════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print(f"  RESULTS")
    print(f"{'═' * 70}")
    print(f"  Phase 1 (L-BFGS): {t_phase1:.1f}s")
    print(f"  Phase 2 (Adam):   {t_phase2:.1f}s")
    print(f"  Best total error: {best_err:.6f}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()

    # Train error
    with torch.no_grad():
        pred = model(train_batch)
    err_train = compute_disp_error(pred, train_batch)
    _, ld_train, _, u_train = loss_fn.compute(
        pred, train_batch, axial_weight=1.0
    )

    print(f"\n  Train set (best model):")
    print(f"    ux:  {err_train[1]:.6f}")
    print(f"    uz:  {err_train[2]:.6f}")
    print(f"    θ:   {err_train[3]:.6f}")
    print(f"    all: {err_train[0]:.6f}")
    print(f"    Π/Π_target: "
          f"{ld_train['Pi'] / target_Pi:.4f}")
    print(f"    U/W: {ld_train['U_over_W']:.4f}")
    print(f"    pred range: [{pred.min().item():.4f}, "
          f"{pred.max().item():.4f}]")
    print(f"    ux range: [{u_train[:, 0].min().item():.4e}, "
          f"{u_train[:, 0].max().item():.4e}]")

    # Test error
    if test_batch is not None:
        with torch.no_grad():
            pred_t = model(test_batch)
        err_test = compute_disp_error(pred_t, test_batch)
        print(f"\n  Test set:")
        print(f"    ux:  {err_test[1]:.6f}")
        print(f"    uz:  {err_test[2]:.6f}")
        print(f"    θ:   {err_test[3]:.6f}")
        print(f"    all: {err_test[0]:.6f}")

    # Quality assessment
    if best_err < 0.02:
        quality = "✓ EXCELLENT (<2%)"
    elif best_err < 0.05:
        quality = "✓ GOOD (<5%)"
    elif best_err < 0.20:
        quality = "~ MODERATE (<20%)"
    else:
        quality = "✗ NEEDS WORK (>20%)"
    print(f"\n  Quality: {quality}")

    # ── Save ──
    save_path = os.path.join(cfg.save_dir, 'checkpoint.pt')
    torch.save({
        'model_state': model.state_dict(),
        'history':     history,
        'config':      vars(cfg),
        'best_err':    best_err,
        'target_Pi':   target_Pi,
        'scales': {
            'ux_c': ux_c, 'uz_c': uz_c,
            'theta_c': theta_c, 'F_c': F_c,
        },
    }, save_path)
    print(f"\n  Saved: {save_path}")

    # ── Loss curves ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        axes[0, 0].semilogy(
            history['step'],
            [abs(l) + 1e-30 for l in history['loss']]
        )
        axes[0, 0].set_title('|Π/E_c|')
        axes[0, 0].set_xlabel('Step')

        # Error
        axes[0, 1].semilogy(history['step'], history['err'])
        axes[0, 1].axhline(0.05, color='g', ls='--',
                           label='5%')
        axes[0, 1].set_title('Displacement Error')
        axes[0, 1].legend()

        # Gradient norm
        axes[1, 0].semilogy(
            history['step'], history['grad_norm']
        )
        axes[1, 0].set_title('Gradient Norm')

        # Axial weight
        axes[1, 1].semilogy(history['step'], history['aw'])
        axes[1, 1].set_title('Axial Weight')

        # Phase boundaries
        for ax in axes.flat:
            ax.axvline(cfg.lbfgs_steps, color='r',
                       ls=':', alpha=0.5)
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"PIGNN Training — best err={best_err:.4f}",
            fontsize=14
        )
        plt.tight_layout()
        fig_path = os.path.join(cfg.save_dir, 'loss_curves.png')
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Curves: {fig_path}")
    except Exception as e:
        print(f"  (Plot skipped: {e})")

    print(f"\n{'═' * 70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'═' * 70}")


if __name__ == '__main__':
    main()