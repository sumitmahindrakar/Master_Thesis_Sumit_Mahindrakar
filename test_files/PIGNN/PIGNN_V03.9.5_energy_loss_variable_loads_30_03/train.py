"""
=================================================================
train.py — L-BFGS with Auto-Restart for PIGNN v2
=================================================================
Key fixes:
  1. Auto-restart L-BFGS when stalled (detects zero-step)
  2. Smaller model option (hidden=64 → 327K params)
  3. NO Adam phase (L-BFGS only — Adam destroys progress)
  4. Constant axial weight (no ramp)
  5. Multiple random seeds (pick best run)
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
import copy

from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

from model import PIGNN
from energy_loss import FrameEnergyLoss
from step_2_grapg_constr import FrameData
from torch_geometric.loader import DataLoader


# ════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════

class Config:
    # Model — SMALLER for better L-BFGS convergence
    hidden_dim   = 64          # was 128 → 4× fewer params
    n_layers     = 10          # keep for full graph coverage
    init_gain    = 0.01

    # Phase 1: Adam mini-batch
    adam_phase1_epochs = 5000
    adam_phase1_lr     = 1e-3
    batch_size         = 16       # ~5 mini-batches per epoch

    # Phase 2: L-BFGS
    lbfgs_steps     = 0     # lbfgs long run
    lbfgs_lr        = 1.0
    lbfgs_history   = 500
    lbfgs_max_iter  = 20
    stall_patience  = 10       # restart after this many zero-steps
    # stall_tol       = 1e-7     # loss change threshold for "stall"

    # Multiple seeds
    n_seeds         = 1        # try 3 random initializations

    # Phase 3: Optional gentle Adam polish (after L-BFGS)
    adam_epochs     = 3000
    adam_lr         = 1e-5     # very small
    adam_lr_min     = 1e-8
    grad_clip       = 5.0      # tight clip

    # Anti-overfitting (ADD THESE)
    weight_decay    = 0#1e-4          # L2 regularization in Adam- Causing Oscillation

    # General
    train_ratio  = 0.85
    log_every    = 50
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir     = 'RESULTS'


# ════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════

def get_scalar(attr):
    if isinstance(attr, torch.Tensor):
        return attr.item() if attr.numel() == 1 else attr[0].item()
    return float(attr)


def compute_disp_error(pred_raw, batch):
    """Relative displacement error per DOF."""
    with torch.no_grad():
        if hasattr(batch, 'batch') and batch.batch is not None:
            ux_c = batch.ux_c[batch.batch]
            # uz_c = batch.uz_c[batch.batch]
            uz_c = batch.ux_c[batch.batch]    # ← CHANGED: use ux_c
            th_c = batch.theta_c[batch.batch]
        else:
            ux_c = batch.ux_c
            # uz_c = batch.uz_c
            uz_c = batch.ux_c    # ← CHANGED: use ux_c
            th_c = batch.theta_c

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
    EA = data.prop_E * data.prop_A
    EI = data.prop_E * data.prop_I22
    L  = data.elem_lengths
    return (12.0 * EI / (EA * L**2)).mean().item()


def ensure_forward_mask(data_list):
    for d in data_list:
        if not hasattr(d, 'edge_forward_mask'):
            E = d.n_elements
            d.edge_forward_mask = torch.cat([
                torch.ones(E, dtype=torch.bool),
                torch.zeros(E, dtype=torch.bool)
            ])


def load_data():
    paths = [
        "DATA/graph_dataset_norm.pt",
        "DATA/graph_dataset.pt",
    ]
    for p in paths:
        if os.path.exists(p):
            data_list = torch.load(p, weights_only=False)
            print(f"  Loaded {len(data_list)} graphs from {p}")
            return data_list

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

    raise FileNotFoundError("No data found!")


# ════════════════════════════════════════════════
# Adam optimizer Phase 1
# ════════════════════════════════════════════════

def train_adam_minibatch(model, train_graphs, loss_fn, optimal_aw, cfg, device, test_batch=None):
    """Phase 1: Adam with mini-batches to handle gradient conflict."""
    model.to(device).train()
    
    loader = DataLoader(train_graphs, batch_size=cfg.batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.adam_phase1_lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.adam_phase1_epochs, eta_min=cfg.adam_phase1_lr / 100
    )
    
    best_err = 999.
    best_state = None
    
    for epoch in range(cfg.adam_phase1_epochs):
        epoch_loss = 0.
        n_batches = 0
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss, _, _, _ = loss_fn.compute(pred, batch, axial_weight=optimal_aw)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Evaluate on full training set every log_every epochs
        if epoch % cfg.log_every == 0 or epoch < 5:
            full_batch = Batch.from_data_list(train_graphs).to(device)
            with torch.no_grad():
                pred_full = model(full_batch)
            err = compute_disp_error(pred_full, full_batch)
            lr = optimizer.param_groups[0]['lr']
            
            if err[0] < best_err:
                best_err = err[0]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            _, ld, _, _ = loss_fn.compute(pred_full, full_batch, axial_weight=optimal_aw)
            
            # ADD: test eval
            if test_batch is not None:
                with torch.no_grad():
                    pred_test = model(test_batch)
                err_test = compute_disp_error(pred_test, test_batch)
                test_str = f" | test={err_test[0]:.4f}"
            else:
                test_str = ""

            print(
                f"    {epoch:5d} | "
                f"Π={ld['Pi']:+.4e} | "
                f"err=[{err[1]:.4f},{err[2]:.4f},{err[3]:.4f}] "
                f"tot={err[0]:.4f} | "
                f"{test_str} | " 
                f"lr={lr:.2e} | "
                f"pred=[{pred_full.min().item():.3f},{pred_full.max().item():.3f}]"
            )
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print(f"\n    Adam Phase 1 done: best_err={best_err:.4f}")
    return best_err, best_state
# ════════════════════════════════════════════════
# L-BFGS training with auto-restart
# ════════════════════════════════════════════════
def train_lbfgs_with_restarts(
    model, train_batch, loss_fn, optimal_aw, cfg, device,
    seed_id=0
):
    """
    L-BFGS with window-based stall detection and
    perturbation restarts.
    """
    model.to(device).train()

    history = {
        'step': [], 'loss': [], 'err': [],
        'grad_norm': [], 'restarts': [],
    }
    best_err   = 999.
    best_loss  = 999.
    best_state = None
    n_restarts = 0

    # ── Window-based stall detection ──
    loss_window = []
    err_window  = []
    WINDOW_SIZE = 300           # check over last 50 steps
    STALL_REL_TOL = 1e-4       # <0.01% improvement = stalled
    PERTURB_SCALE = 0.003      # noise scale for perturbation
    PERTURB_GROWTH = 1.3         # was 1.5
    MAX_PERTURB   = 0.05         # was uncapped
    MAX_RESTARTS  = 20    

    def make_optimizer():
        return torch.optim.LBFGS(
            model.parameters(),
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_max_iter,
            history_size=cfg.lbfgs_history,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-10,
            tolerance_change=1e-14,
        )

    def perturb_decoder(scale):
        """Add noise to decoder weights to escape plateau."""
        with torch.no_grad():
            for name, p in model.named_parameters():
                if 'decoder' in name:
                    noise = torch.randn_like(p) * scale
                    # Scale noise by parameter magnitude
                    noise *= p.abs().mean().clamp(min=1e-6)
                    p.add_(noise)

    optimizer = make_optimizer()
    t0 = time.time()

    for step in range(cfg.lbfgs_steps):
        # ── L-BFGS step ──
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
            print(f"    Step {step}: exception — {e}")
            if best_state is not None:
                model.load_state_dict(best_state)
            optimizer = make_optimizer()
            n_restarts += 1
            loss_window.clear()
            err_window.clear()
            continue

        if not np.isfinite(loss_val[0]):
            print(f"    Step {step}: NaN — restoring best")
            if best_state is not None:
                model.load_state_dict(best_state)
            optimizer = make_optimizer()
            n_restarts += 1
            loss_window.clear()
            err_window.clear()
            continue

        # ── Track windows ──
        loss_window.append(loss_val[0])
        if len(loss_window) > WINDOW_SIZE:
            loss_window.pop(0)

        # ── Window-based stall detection ──
        stalled = False
        if len(loss_window) >= WINDOW_SIZE:
            # Compare first half vs second half of window
            first_half = np.mean(loss_window[:WINDOW_SIZE//2])
            second_half = np.mean(loss_window[WINDOW_SIZE//2:])
            rel_improve = (first_half - second_half) / (
                abs(first_half) + 1e-30
            )
            # Stall if improvement < 0.01% over window
            if rel_improve < STALL_REL_TOL:
                stalled = True

        # ── Perturbation restart when stalled ──
        if stalled and n_restarts < MAX_RESTARTS:
            n_restarts += 1
            loss_window.clear()
            err_window.clear()

            # Increasing perturbation with each restart
            # scale = PERTURB_SCALE * (1.5 ** (n_restarts - 1))
            scale = min(
                PERTURB_SCALE * (PERTURB_GROWTH ** (n_restarts - 1)),
                MAX_PERTURB
            )

            with torch.no_grad():
                pred = model(train_batch)
            err = compute_disp_error(pred, train_batch)

            print(
                f"    {step:5d} | *** RESTART #{n_restarts} "
                f"(perturb={scale:.4f}) *** | "
                f"Π={loss_val[0]:+.4e} | "
                f"err={err[0]:.4f}"
            )

            # Perturb decoder weights
            perturb_decoder(scale)

            # Reset optimizer (clears stale history)
            optimizer = make_optimizer()
            continue

        # ── Track best ──
        if loss_val[0] < best_loss - 1e-8:
            best_loss = loss_val[0]

        with torch.no_grad():
            pred = model(train_batch)
        err = compute_disp_error(pred, train_batch)

        if err[0] < best_err:
            best_err = err[0]
            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

        # ── Logging ──
        if step % cfg.log_every == 0 or step < 5:
            gn = sum(
                p.grad.norm()**2
                for p in model.parameters()
                if p.grad is not None
            ).sqrt().item()

            history['step'].append(step)
            history['loss'].append(loss_val[0])
            history['err'].append(err[0])
            history['grad_norm'].append(gn)
            history['restarts'].append(n_restarts)

            # Show progress rate
            rate_str = ""
            if len(history['err']) >= 3:
                prev = history['err'][-3]
                curr = history['err'][-1]
                rate = (prev - curr) / (
                    2 * cfg.log_every + 1e-10
                )
                rate_str = f" Δ/step={rate:.2e}"

            print(
                f"    {step:5d} | "
                f"Π={loss_val[0]:+.4e} | "
                f"err=[{err[1]:.4f},{err[2]:.4f},"
                f"{err[3]:.4f}]"
                f" tot={err[0]:.4f} | "
                f"|∇|={gn:.2e} | "
                f"pred=[{pred.min().item():.3f},"
                f"{pred.max().item():.3f}] | "
                f"R={n_restarts}"
                f"{rate_str}"
            )

        # Early stopping
        if best_err < 0.01:
            print(f"    ✓ Converged (err < 1%)")
            break

    elapsed = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n    Seed {seed_id}: {elapsed:.0f}s, "
          f"{n_restarts} restarts, "
          f"best_err={best_err:.4f}, "
          f"best_loss={best_loss:.4e}")

    return best_err, best_loss, history

# ════════════════════════════════════════════════
# Optional Adam polish
# ════════════════════════════════════════════════

def adam_polish(model, train_batch, loss_fn, optimal_aw,
               cfg, device):
    """Very gentle Adam fine-tuning after L-BFGS."""
    model.to(device).train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.adam_lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.adam_epochs,
        eta_min=cfg.adam_lr_min
    )

    best_err = 999.
    best_state = None
    initial_err = None

    for epoch in range(cfg.adam_epochs):
        optimizer.zero_grad()
        pred = model(train_batch)
        loss, ld, _, _ = loss_fn.compute(
            pred, train_batch, axial_weight=optimal_aw
        )
        loss.backward()

        gn = sum(
            p.grad.norm()**2
            for p in model.parameters()
            if p.grad is not None
        ).sqrt().item()

        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if epoch % 200 == 0 or epoch == cfg.adam_epochs - 1:
            with torch.no_grad():
                pred = model(train_batch)
            err = compute_disp_error(pred, train_batch)
            lr = optimizer.param_groups[0]['lr']

            if initial_err is None:
                initial_err = err[0]

            if err[0] < best_err:
                best_err = err[0]
                best_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }

            clp = "CLIP" if gn > cfg.grad_clip else "    "
            print(
                f"    {epoch:5d} | "
                f"Π={loss.item():+.4e} | "
                f"err={err[0]:.4f} | "
                f"|∇|={gn:.2e} {clp} | "
                f"lr={lr:.1e} | "
                f"U/W={ld['U_over_W']:.3f}"
            )

            # If Adam is making things worse, stop
            if err[0] > initial_err * 1.1 and epoch > 100:
                print(f"    Adam making things worse "
                      f"({err[0]:.4f} > {initial_err:.4f})"
                      f" — stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_err


# ════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════

def main():
    cfg = Config()
    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("=" * 70)
    print("  PIGNN TRAINING — L-BFGS with Auto-Restart v3")
    print("=" * 70)

    # ── Load data ──
    print("\n── Loading data ──")
    data_list = load_data()
    d0 = data_list[0]

    for attr in ['ux_c', 'uz_c', 'theta_c', 'F_c']:
        assert hasattr(d0, attr), f"Missing {attr}!"

    ensure_forward_mask(data_list)

    ux_c    = get_scalar(d0.ux_c)
    uz_c    = get_scalar(d0.uz_c)
    theta_c = get_scalar(d0.theta_c)
    F_c     = get_scalar(d0.F_c)
    E_c     = F_c * ux_c
    N       = d0.num_nodes
    E       = d0.n_elements
    optimal_aw = compute_optimal_aw(d0)

    print(f"  Nodes: {N}, Elements: {E}")
    print(f"  ux_c = {ux_c:.4e}, uz_c = {uz_c:.4e}, "
          f"θ_c = {theta_c:.4e}")
    print(f"  F_c = {F_c:.4e}, E_c = {E_c:.4e}")
    print(f"  Optimal aw = {optimal_aw:.4e}")
    print(f"  Config: hidden={cfg.hidden_dim}, "
          f"layers={cfg.n_layers}, "
          f"seeds={cfg.n_seeds}")

    # ── Split ──
    n_total = len(data_list)
    n_train = max(1, int(cfg.train_ratio * n_total))
    train_graphs = data_list[:n_train]
    test_graphs  = (data_list[n_train:]
                    if n_train < n_total
                    else data_list[-2:])
    print(f"  Split: {len(train_graphs)} train, "
          f"{len(test_graphs)} test")

    train_batch = Batch.from_data_list(
        train_graphs
    ).to(device)
    test_batch = Batch.from_data_list(
        test_graphs
    ).to(device)

    loss_fn = FrameEnergyLoss().to(device)

    # ── Target energy ──
    with torch.no_grad():
        true_d = train_batch.y_node.to(device)
        if hasattr(train_batch, 'batch') \
                and train_batch.batch is not None:
            ux_c_t = train_batch.ux_c[train_batch.batch]
            # uz_c_t = train_batch.uz_c[train_batch.batch]
            uz_c_t = train_batch.ux_c[train_batch.batch]
            th_c_t = train_batch.theta_c[train_batch.batch]
        else:
            ux_c_t = train_batch.ux_c
            # uz_c_t = train_batch.uz_c
            uz_c_t = train_batch.ux_c
            th_c_t = train_batch.theta_c

        true_raw = torch.stack([
            true_d[:, 0] / ux_c_t,
            # true_d[:, 1] / uz_c_t,
            true_d[:, 1] / ux_c_t,            # ← CHANGED: use ux_c
            true_d[:, 2] / th_c_t,
        ], dim=1)
        true_raw = torch.nan_to_num(true_raw, 0.0)

        _, tgt_dict, _, _ = loss_fn.compute(
            true_raw, train_batch, axial_weight=optimal_aw
        )
        target_Pi = tgt_dict['Pi']
        print(f"  Target Π/E_c = {target_Pi:.4e} "
              f"(aw={optimal_aw:.2e})")
        print(f"  Target U/W   = {tgt_dict['U_over_W']:.4f}")

    # ════════════════════════════════════════
    # Multi-seed L-BFGS training
    # ════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print(f"  L-BFGS TRAINING: {cfg.n_seeds} seeds × "
          f"{cfg.lbfgs_steps} steps")
    print(f"  aw={optimal_aw:.2e} (constant), "
          f"history={cfg.lbfgs_history}, "
          f"stall_patience={cfg.stall_patience}")
    print(f"{'═' * 70}")

    best_overall_err = 999.
    best_overall_state = None
    best_seed = 0
    all_histories = []

    for seed in range(cfg.n_seeds):
        print(f"\n── Seed {seed} "
              f"{'─' * 50}")

        torch.manual_seed(42 + seed * 1000)
        np.random.seed(42 + seed * 1000)

        model = PIGNN(
            node_in_dim=10,
            edge_in_dim=7,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
            decoder_init_gain=cfg.init_gain,
        ).to(device)

        n_params = model.count_params()
        if seed == 0:
            model.summary()
            print(f"  L-BFGS history coverage: "
                  f"{cfg.lbfgs_history}/{n_params} = "
                  f"{100*cfg.lbfgs_history/n_params:.3f}%")

        # Verify init
        with torch.no_grad():
            p = model(train_batch)
            assert p.abs().max() > 1e-8, "Zero init!"

        # Train
        # ── Phase 1: Adam mini-batch ──
        print(f"\n    Phase 1: Adam mini-batch (bs={cfg.batch_size}, "
              f"lr={cfg.adam_phase1_lr:.0e}, {cfg.adam_phase1_epochs} epochs)")
        
        err_adam, state_adam = train_adam_minibatch(
            model, train_graphs, loss_fn, optimal_aw, cfg, device
        )
        
        # ── Phase 2: L-BFGS polish from Adam result ──
        print(f"\n    Phase 2: L-BFGS polish from err={err_adam:.4f}")

        err, loss, hist = train_lbfgs_with_restarts(
            model, train_batch, loss_fn, optimal_aw,
            cfg, device, seed_id=seed
        )
        all_histories.append(hist)

        if err < best_overall_err:
            best_overall_err = err
            best_overall_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_seed = seed

    print(f"\n{'═' * 70}")
    print(f"  Best seed: {best_seed}, "
          f"err={best_overall_err:.4f}")
    print(f"{'═' * 70}")

    # ════════════════════════════════════════
    # Optional Adam polish
    # ════════════════════════════════════════
    if best_overall_err > 0.01:
        print(f"\n── Adam polish (lr={cfg.adam_lr:.0e}) ──")
        model = PIGNN(
            node_in_dim=10, edge_in_dim=7,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
            decoder_init_gain=cfg.init_gain,
        ).to(device)
        model.load_state_dict(best_overall_state)

        err_after_adam = adam_polish(
            model, train_batch, loss_fn, optimal_aw,
            cfg, device
        )

        if err_after_adam < best_overall_err:
            best_overall_err = err_after_adam
            best_overall_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
            print(f"  Adam improved: {err_after_adam:.4f}")
        else:
            # Reload L-BFGS result
            model.load_state_dict(best_overall_state)
            print(f"  Adam did NOT improve, "
                  f"keeping L-BFGS result")

    # ════════════════════════════════════════
    # Final evaluation
    # ════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print(f"  RESULTS")
    print(f"{'═' * 70}")

    model = PIGNN(
        node_in_dim=10, edge_in_dim=7,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        decoder_init_gain=cfg.init_gain,
    ).to(device)
    model.load_state_dict(best_overall_state)
    model.eval()

    with torch.no_grad():
        pred = model(train_batch)
    err_train = compute_disp_error(pred, train_batch)

    # Compute energy with OPTIMAL aw (not full aw=1)
    _, ld, _, u_train = loss_fn.compute(
        pred, train_batch, axial_weight=optimal_aw
    )

    print(f"\n  Train set (aw={optimal_aw:.2e}):")
    print(f"    ux:  {err_train[1]:.6f}")
    print(f"    uz:  {err_train[2]:.6f}")
    print(f"    θ:   {err_train[3]:.6f}")
    print(f"    all: {err_train[0]:.6f}")
    print(f"    Π/Π_target: "
          f"{ld['Pi'] / target_Pi:.4f}")
    print(f"    U/W: {ld['U_over_W']:.4f} "
          f"(target: 0.5000)")
    print(f"    pred range: [{pred.min().item():.4f}, "
          f"{pred.max().item():.4f}]")

    with torch.no_grad():
        pred_t = model(test_batch)
    err_test = compute_disp_error(pred_t, test_batch)

    print(f"\n  Test set:")
    print(f"    ux:  {err_test[1]:.6f}")
    print(f"    uz:  {err_test[2]:.6f}")
    print(f"    θ:   {err_test[3]:.6f}")
    print(f"    all: {err_test[0]:.6f}")

    if best_overall_err < 0.02:
        quality = "✓ EXCELLENT (<2%)"
    elif best_overall_err < 0.05:
        quality = "✓ GOOD (<5%)"
    elif best_overall_err < 0.20:
        quality = "~ MODERATE (<20%)"
    else:
        quality = "✗ NEEDS WORK (>20%)"
    print(f"\n  Quality: {quality}")

    # ── Save ──
    save_path = os.path.join(cfg.save_dir, 'checkpoint.pt')
    torch.save({
        'model_state':  best_overall_state,
        'histories':    all_histories,
        'config':       vars(cfg),
        'best_err':     best_overall_err,
        'best_seed':    best_seed,
        'target_Pi':    target_Pi,
        'optimal_aw':   optimal_aw,
        'scales': {
            'ux_c': ux_c, 'uz_c': uz_c,
            'theta_c': theta_c, 'F_c': F_c,
        },
    }, save_path)
    print(f"\n  Saved: {save_path}")

    # ── Curves ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        colors = ['tab:blue', 'tab:orange', 'tab:green',
                  'tab:red', 'tab:purple']

        for i, hist in enumerate(all_histories):
            c = colors[i % len(colors)]
            lbl = f"Seed {i}"
            if i == best_seed:
                lbl += " ★"

            axes[0].semilogy(
                hist['step'],
                [abs(l) + 1e-30 for l in hist['loss']],
                color=c, label=lbl, alpha=0.8
            )
            axes[1].semilogy(
                hist['step'], hist['err'],
                color=c, label=lbl, alpha=0.8
            )
            axes[2].semilogy(
                hist['step'], hist['grad_norm'],
                color=c, label=lbl, alpha=0.8
            )

        axes[0].set_title('|Π/E_c|')
        axes[1].set_title('Displacement Error')
        axes[1].axhline(0.05, color='g', ls='--',
                        alpha=0.5, label='5%')
        axes[2].set_title('Gradient Norm')

        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Step')

        plt.suptitle(
            f"L-BFGS+Restart — best err={best_overall_err:.4f}"
            f" (seed {best_seed})",
            fontsize=14
        )
        plt.tight_layout()
        fig_path = os.path.join(
            cfg.save_dir, 'loss_curves.png'
        )
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