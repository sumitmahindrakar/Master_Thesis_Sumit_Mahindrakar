"""
=================================================================
train.py — Full Pipeline PIGNN Training v5
=================================================================
  1. Unified TrainingHistory across ALL phases (Adam→uz→LBFGS→Polish)
  2. Test evaluation at every logging step
  3. Save both best AND final model states
  4. Complete checkpoint (config, scales, split, timings, history)
  5. Per-sample error analysis + worst-case identification
  6. Comprehensive results report + JSON summary
  7. Centralized uz_c scaling flag
  8. Prediction scatter plots + pred-vs-true curves
  9. Safe zero-epoch/step handling for all phases
  10. uz-focused phase (freeze all except decoder_uz)
  11. Residual loss mode: loss = ||∂Π/∂u||² (flag-controlled)
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
import json
from datetime import datetime
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
    # Model
    hidden_dim   = 32
    n_layers     = 10
    init_gain    = 0.01

    # Loss mode
    use_residual_loss = True   # True = ||∂Π/∂u||²,  False = Π (energy)

    # Phase 1: Adam mini-batch
    adam_phase1_epochs = 1000
    adam_phase1_lr     = 1e-3
    batch_size         = 16

    # Phase 2: uz-focused (decoder_uz only)
    uz_epochs    = 0
    uz_lr        = 1e-3

    # Phase 3: L-BFGS
    lbfgs_steps     = 0
    lbfgs_lr        = 1.0
    lbfgs_history   = 500
    lbfgs_max_iter  = 20
    stall_patience  = 10

    # Multiple seeds
    n_seeds         = 1

    # Phase 4: Adam polish
    adam_epochs     = 1000
    adam_lr         = 1e-5
    adam_lr_min     = 1e-8
    grad_clip       = 5.0

    # Regularization
    weight_decay    = 0

    # General
    train_ratio  = 0.85
    log_every    = 50
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir     = 'RESULTS'

    # Physics scaling flag
    use_ux_for_uz = False


# ════════════════════════════════════════════════
# Unified History Tracker
# ════════════════════════════════════════════════

class TrainingHistory:
    def __init__(self):
        self.records = []
        self.phase_boundaries = []
        self.global_step = 0
        self._phase_name = None
        self._phase_start = 0

    def begin_phase(self, name):
        if self._phase_name is not None:
            self.phase_boundaries.append(
                (self._phase_name, self._phase_start, self.global_step)
            )
        self._phase_name = name
        self._phase_start = self.global_step

    def end_current_phase(self):
        if self._phase_name is not None:
            self.phase_boundaries.append(
                (self._phase_name, self._phase_start, self.global_step)
            )
            self._phase_name = None

    def log(self, local_step, **metrics):
        self.records.append({
            'global_step': self.global_step,
            'local_step':  local_step,
            'phase':       self._phase_name,
            **metrics,
        })
        self.global_step += 1

    def series(self, key):
        xs, ys = [], []
        for r in self.records:
            if key in r and r[key] is not None:
                xs.append(r['global_step'])
                ys.append(r[key])
        return xs, ys

    def phase_records(self, phase_name):
        return [r for r in self.records if r.get('phase') == phase_name]

    def to_dict(self):
        return {
            'records':          self.records,
            'phase_boundaries': self.phase_boundaries,
            'final_global_step': self.global_step,
        }


# ════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════

def get_scalar(attr):
    if isinstance(attr, torch.Tensor):
        return attr.item() if attr.numel() == 1 else attr[0].item()
    return float(attr)


def _broadcast(per_graph_attr, batch_vec):
    if batch_vec is not None:
        return per_graph_attr[batch_vec]
    return per_graph_attr


def uz_scale(batch, cfg):
    src = batch.ux_c if cfg.use_ux_for_uz else batch.uz_c
    bv  = batch.batch if hasattr(batch, 'batch') else None
    return _broadcast(src, bv)


def compute_disp_error(pred_raw, batch, cfg):
    with torch.no_grad():
        bv = batch.batch if hasattr(batch, 'batch') else None
        ux_c = _broadcast(batch.ux_c, bv)
        uz_c = uz_scale(batch, cfg)
        th_c = _broadcast(batch.theta_c, bv)

        pred_phys = torch.stack([
            pred_raw[:, 0] * ux_c,
            pred_raw[:, 1] * uz_c,
            pred_raw[:, 2] * th_c,
        ], dim=1)

        true_phys = batch.y_node
        eps  = 1e-12
        diff = pred_phys - true_phys

        err_total = diff.norm()        / true_phys.norm().clamp(min=eps)
        err_ux    = diff[:, 0].norm()  / true_phys[:, 0].norm().clamp(min=eps)
        err_uz    = diff[:, 1].norm()  / true_phys[:, 1].norm().clamp(min=eps)
        err_th    = diff[:, 2].norm()  / true_phys[:, 2].norm().clamp(min=eps)

    return err_total.item(), err_ux.item(), err_uz.item(), err_th.item()


def _compute_loss(pred, batch, loss_fn, aw, cfg):
    """
    Compute training loss based on config flag.
    Returns (loss_tensor, res_info_or_None).
    
    Energy mode:   loss = Π,              res_info = None
    Residual mode: loss = ||∂Π/∂u||²,    res_info = dict with R_ux, R_uz, R_th, total, Pi
    """
    if cfg.use_residual_loss:
        loss, res_info = loss_fn.compute_residual_loss(
            pred, batch, axial_weight=aw
        )
        return loss, res_info
    else:
        loss, _, _, _ = loss_fn.compute(pred, batch, axial_weight=aw)
        return loss, None


def evaluate_full(model, batch, loss_fn, aw, cfg):
    """Energy-based evaluation (always, regardless of loss mode)."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred = model(batch)
        err  = compute_disp_error(pred, batch, cfg)
        loss, ld, _, _ = loss_fn.compute(pred, batch, axial_weight=aw)
    if was_training:
        model.train()
    return pred, err, loss.item(), ld


def evaluate_residual(model, batch, loss_fn, aw):
    """
    Compute full-set residual R² (requires grad — NOT inside no_grad).
    Used for final report only.
    """
    was_training = model.training
    model.eval()
    pred = model(batch)
    _, res_info = loss_fn.compute_residual_loss(pred, batch, axial_weight=aw)
    if was_training:
        model.train()
    return res_info


def per_sample_errors(model, data_list, loss_fn, aw, cfg, device):
    model.eval()
    results = []
    with torch.no_grad():
        for i, data in enumerate(data_list):
            b = Batch.from_data_list([data]).to(device)
            pred = model(b)
            err  = compute_disp_error(pred, b, cfg)
            _, ld, _, _ = loss_fn.compute(pred, b, axial_weight=aw)
            results.append({
                'idx':       i,
                'err_total': err[0],
                'err_ux':    err[1],
                'err_uz':    err[2],
                'err_theta': err[3],
                'Pi':        float(ld['Pi']),
                'U_over_W':  float(ld['U_over_W']),
                'n_nodes':   data.num_nodes,
                'n_elem':    int(data.n_elements) if hasattr(data, 'n_elements') else None,
            })
    model.train()
    return results


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
                torch.zeros(E, dtype=torch.bool),
            ])


def load_data():
    paths = ["DATA/graph_dataset_norm.pt", "DATA/graph_dataset.pt"]
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


def _grad_norm(model):
    gn_sq = sum(
        p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None
    )
    return gn_sq.sqrt().item() if isinstance(gn_sq, torch.Tensor) else 0.0


# ════════════════════════════════════════════════
# Logging helpers
# ════════════════════════════════════════════════

def _log_line(local_step, ld_tr, err_tr, err_te,
              gn, lr, pred, res_info=None, extra=""):
    """
    Unified log line.
    res_info: dict with R_ux, R_uz, R_th, total (when using residual loss)
    """
    parts = [f"    {local_step:5d} |"]

    if res_info is not None:
        parts.append(
            f" R²=[{res_info['R_ux']:.2e},{res_info['R_uz']:.2e},"
            f"{res_info['R_th']:.2e}]={res_info['total']:.2e} |"
        )

    parts.append(f" Π={ld_tr['Pi']:+.4e} |")
    parts.append(
        f" err=[{err_tr[1]:.4f},{err_tr[2]:.4f},{err_tr[3]:.4f}]"
        f" tot={err_tr[0]:.4f} |"
    )
    parts.append(f" test={err_te[0]:.4f} |")
    parts.append(f" U/W={ld_tr['U_over_W']:.3f} |")
    parts.append(f" |∇|={gn:.2e} |")
    parts.append(f" lr={lr:.2e} |")
    parts.append(f" pred=[{pred.min().item():.3f},{pred.max().item():.3f}]")
    if extra:
        parts.append(extra)

    print("".join(parts))


def _record_metrics(history, local_step, loss_tr, loss_te,
                    err_tr, err_te, ld_tr, ld_te,
                    gn, lr, pred, best_err,
                    res_info=None, **extra):
    """
    Push a full record into unified history.
    loss_tr/loss_te are always energy-based (from evaluate_full).
    res_info contains residual metrics when using residual loss.
    """
    d = dict(
        train_loss=loss_tr,
        test_loss=loss_te,
        train_err=err_tr[0], test_err=err_te[0],
        train_err_ux=err_tr[1],    test_err_ux=err_te[1],
        train_err_uz=err_tr[2],    test_err_uz=err_te[2],
        train_err_theta=err_tr[3], test_err_theta=err_te[3],
        Pi_train=float(ld_tr['Pi']),       Pi_test=float(ld_te['Pi']),
        U_over_W_train=float(ld_tr['U_over_W']),
        U_over_W_test=float(ld_te['U_over_W']),
        grad_norm=gn, lr=lr,
        pred_min=pred.min().item(), pred_max=pred.max().item(),
        best_err=best_err,
    )
    if res_info is not None:
        d['R_total'] = res_info['total']
        d['R_ux']    = res_info['R_ux']
        d['R_uz']    = res_info['R_uz']
        d['R_th']    = res_info['R_th']
    d.update(extra)
    history.log(local_step, **d)


def _accumulate_residual(accum, res_info):
    """Accumulate mini-batch residual info into epoch-level dict."""
    if res_info is None:
        return accum
    if accum is None:
        return {k: v for k, v in res_info.items()}
    for k in res_info:
        accum[k] += res_info[k]
    return accum


def _average_residual(accum, n):
    """Average accumulated residual over n mini-batches."""
    if accum is None or n == 0:
        return None
    return {k: v / n for k, v in accum.items()}


# ════════════════════════════════════════════════
# Phase 1 — Adam mini-batch
# ════════════════════════════════════════════════

def train_adam_minibatch(model, train_graphs, train_batch, test_batch,
                         loss_fn, aw, cfg, device, history):
    history.begin_phase('adam_phase1')

    if cfg.adam_phase1_epochs <= 0:
        print("    Phase 1: SKIPPED (0 epochs)")
        return 999., None, 0.

    model.to(device).train()

    loader = DataLoader(train_graphs, batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.adam_phase1_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.adam_phase1_epochs, 1),
        eta_min=cfg.adam_phase1_lr / 100,
    )

    best_err   = 999.
    best_state = None
    t0 = time.time()

    for epoch in range(cfg.adam_phase1_epochs):
        epoch_loss = 0.
        epoch_res  = None
        n_batches  = 0

        for mb in loader:
            mb = mb.to(device)
            optimizer.zero_grad()
            pred = model(mb)
            loss, res_info = _compute_loss(pred, mb, loss_fn, aw, cfg)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_res = _accumulate_residual(epoch_res, res_info)
            n_batches += 1

        scheduler.step()
        avg_res = _average_residual(epoch_res, n_batches)

        do_log = (epoch % cfg.log_every == 0 or epoch < 5
                  or epoch == cfg.adam_phase1_epochs - 1)
        if not do_log:
            continue

        pred_tr, err_tr, loss_tr, ld_tr = evaluate_full(
            model, train_batch, loss_fn, aw, cfg)
        pred_te, err_te, loss_te, ld_te = evaluate_full(
            model, test_batch,  loss_fn, aw, cfg)
        gn = _grad_norm(model)
        lr = optimizer.param_groups[0]['lr']

        if err_tr[0] < best_err:
            best_err   = err_tr[0]
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        _record_metrics(history, epoch, loss_tr, loss_te,
                        err_tr, err_te, ld_tr, ld_te,
                        gn, lr, pred_tr, best_err,
                        res_info=avg_res)
        _log_line(epoch, ld_tr, err_tr, err_te,
                  gn, lr, pred_tr, res_info=avg_res)

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n    Phase 1 done: {elapsed:.0f}s, best_err={best_err:.4f}")
    return best_err, best_state, elapsed


# ════════════════════════════════════════════════
# Phase 2 — uz-focused (freeze all except decoder_uz)
# ════════════════════════════════════════════════

def train_uz_focused(model, train_graphs, train_batch, test_batch,
                     loss_fn, aw, cfg, device, history):
    history.begin_phase('uz_focused')

    if cfg.uz_epochs <= 0:
        print("    Phase 2: SKIPPED (0 epochs)")
        return 999., None, 0.

    model.to(device).train()

    # ── Freeze everything except decoder_uz ──
    for name, param in model.named_parameters():
        param.requires_grad = ('decoder_uz' in name)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_p   = sum(p.numel() for p in model.parameters())
    print(f"    Trainable: {n_trainable}/{n_total_p} params (decoder_uz only)")

    if n_trainable == 0:
        print("    ⚠ No trainable params with 'decoder_uz' — check model!")
        for param in model.parameters():
            param.requires_grad = True
        return 999., None, 0.

    loader = DataLoader(train_graphs, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.uz_lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.uz_epochs, 1),
        eta_min=cfg.uz_lr / 100,
    )

    best_err   = 999.
    best_state = None
    t0 = time.time()

    for epoch in range(cfg.uz_epochs):
        epoch_loss = 0.
        epoch_res  = None
        n_batches  = 0

        for mb in loader:
            mb = mb.to(device)
            optimizer.zero_grad()
            pred = model(mb)
            loss, res_info = _compute_loss(pred, mb, loss_fn, aw, cfg)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_res = _accumulate_residual(epoch_res, res_info)
            n_batches += 1

        scheduler.step()
        avg_res = _average_residual(epoch_res, n_batches)

        do_log = (epoch % cfg.log_every == 0 or epoch < 5
                  or epoch == cfg.uz_epochs - 1)
        if not do_log:
            continue

        pred_tr, err_tr, loss_tr, ld_tr = evaluate_full(
            model, train_batch, loss_fn, aw, cfg)
        pred_te, err_te, loss_te, ld_te = evaluate_full(
            model, test_batch,  loss_fn, aw, cfg)
        gn = _grad_norm(model)
        lr = optimizer.param_groups[0]['lr']

        if err_tr[0] < best_err:
            best_err   = err_tr[0]
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        _record_metrics(history, epoch, loss_tr, loss_te,
                        err_tr, err_te, ld_tr, ld_te,
                        gn, lr, pred_tr, best_err,
                        res_info=avg_res)
        _log_line(epoch, ld_tr, err_tr, err_te,
                  gn, lr, pred_tr, res_info=avg_res)

    # ── Unfreeze all parameters ──
    for param in model.parameters():
        param.requires_grad = True

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
        for param in model.parameters():
            param.requires_grad = True

    print(f"\n    Phase 2 done: {elapsed:.0f}s, best_err={best_err:.4f}")
    return best_err, best_state, elapsed


# ════════════════════════════════════════════════
# Phase 3 — L-BFGS with auto-restart
# ════════════════════════════════════════════════

def train_lbfgs_with_restarts(model, train_batch, test_batch,
                               loss_fn, aw, cfg, device, history,
                               seed_id=0):
    history.begin_phase('lbfgs')

    if cfg.lbfgs_steps <= 0:
        print("    Phase 3: SKIPPED (0 steps)")
        with torch.no_grad():
            p = model(train_batch)
        err = compute_disp_error(p, train_batch, cfg)
        return err[0], 999., 0.

    model.to(device).train()

    best_err   = 999.
    best_loss  = 999.
    best_state = None
    n_restarts = 0

    loss_window = []
    WIN       = 300
    STALL_TOL = 1e-4
    P_SCALE   = 0.003
    P_GROW    = 1.3
    P_MAX     = 0.05
    R_MAX     = 20

    def make_opt():
        return torch.optim.LBFGS(
            model.parameters(), lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_max_iter,
            history_size=cfg.lbfgs_history,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-10, tolerance_change=1e-14,
        )

    def perturb(scale):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if 'decoder' in n:
                    p.add_(torch.randn_like(p) * scale
                           * p.abs().mean().clamp(min=1e-6))

    opt = make_opt()
    t0  = time.time()
    res_holder = [None]   # capture residual info from closure

    for step in range(cfg.lbfgs_steps):
        lv = [0.0]

        def closure():
            opt.zero_grad()
            pred = model(train_batch)
            loss, res_holder[0] = _compute_loss(
                pred, train_batch, loss_fn, aw, cfg
            )
            loss.backward()
            lv[0] = loss.item()
            return loss

        try:
            opt.step(closure)
        except Exception as e:
            print(f"    Step {step}: exception — {e}")
            if best_state:
                model.load_state_dict(best_state)
            opt = make_opt(); n_restarts += 1; loss_window.clear()
            continue

        if not np.isfinite(lv[0]):
            print(f"    Step {step}: NaN — restoring best")
            if best_state:
                model.load_state_dict(best_state)
            opt = make_opt(); n_restarts += 1; loss_window.clear()
            continue

        loss_window.append(lv[0])
        if len(loss_window) > WIN:
            loss_window.pop(0)

        stalled = False
        if len(loss_window) >= WIN:
            h1 = np.mean(loss_window[:WIN // 2])
            h2 = np.mean(loss_window[WIN // 2:])
            if (h1 - h2) / (abs(h1) + 1e-30) < STALL_TOL:
                stalled = True

        if stalled and n_restarts < R_MAX:
            n_restarts += 1
            loss_window.clear()
            sc = min(P_SCALE * P_GROW ** (n_restarts - 1), P_MAX)
            with torch.no_grad():
                pred = model(train_batch)
            err = compute_disp_error(pred, train_batch, cfg)
            print(f"    {step:5d} | *** RESTART #{n_restarts} "
                  f"(perturb={sc:.4f}) *** | "
                  f"loss={lv[0]:+.4e} | err={err[0]:.4f}")
            perturb(sc)
            opt = make_opt()
            continue

        if lv[0] < best_loss - 1e-8:
            best_loss = lv[0]

        with torch.no_grad():
            pred = model(train_batch)
        err_tr = compute_disp_error(pred, train_batch, cfg)
        if err_tr[0] < best_err:
            best_err   = err_tr[0]
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        if step % cfg.log_every == 0 or step < 5:
            gn = _grad_norm(model)
            pred_tr2, err_tr2, loss_tr2, ld_tr2 = evaluate_full(
                model, train_batch, loss_fn, aw, cfg)
            _, err_te, loss_te, ld_te = evaluate_full(
                model, test_batch, loss_fn, aw, cfg)

            _record_metrics(history, step, loss_tr2, loss_te,
                            err_tr2, err_te, ld_tr2, ld_te,
                            gn, cfg.lbfgs_lr, pred_tr2, best_err,
                            res_info=res_holder[0],
                            n_restarts=n_restarts)
            _log_line(step, ld_tr2, err_tr2, err_te,
                      gn, cfg.lbfgs_lr, pred_tr2,
                      res_info=res_holder[0],
                      extra=f" | R={n_restarts}")

        if best_err < 0.01:
            print(f"    ✓ Converged (err < 1%)")
            break

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n    Phase 3 done: {elapsed:.0f}s, {n_restarts} restarts, "
          f"best_err={best_err:.4f}")
    return best_err, best_loss, elapsed


# ════════════════════════════════════════════════
# Phase 4 — Adam polish
# ════════════════════════════════════════════════

def adam_polish(model, train_batch, test_batch,
               loss_fn, aw, cfg, device, history):
    history.begin_phase('adam_polish')

    if cfg.adam_epochs <= 0:
        print("    Phase 4: SKIPPED (0 epochs)")
        with torch.no_grad():
            p = model(train_batch)
        err = compute_disp_error(p, train_batch, cfg)
        return err[0], 0.

    model.to(device).train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.adam_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.adam_epochs, 1),
        eta_min=cfg.adam_lr_min,
    )

    best_err    = 999.
    best_state  = None
    initial_err = None
    t0 = time.time()

    for epoch in range(cfg.adam_epochs):
        optimizer.zero_grad()
        pred = model(train_batch)
        # Enable grad tracking for residual computation
        # pred = pred.detach().requires_grad_(True)
        loss, res_info = _compute_loss(pred, train_batch, loss_fn, aw, cfg)
        loss.backward()
        gn = _grad_norm(model)
        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        do_log = (epoch % 200 == 0 or epoch == cfg.adam_epochs - 1)
        if not do_log:
            continue

        pred_tr, err_tr, loss_tr, ld_tr = evaluate_full(
            model, train_batch, loss_fn, aw, cfg)
        pred_te, err_te, loss_te, ld_te = evaluate_full(
            model, test_batch,  loss_fn, aw, cfg)
        lr = optimizer.param_groups[0]['lr']

        if initial_err is None:
            initial_err = err_tr[0]
        if err_tr[0] < best_err:
            best_err   = err_tr[0]
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        _record_metrics(history, epoch, loss_tr, loss_te,
                        err_tr, err_te, ld_tr, ld_te,
                        gn, lr, pred_tr, best_err,
                        res_info=res_info)
        _log_line(epoch, ld_tr, err_tr, err_te,
                  gn, lr, pred_tr, res_info=res_info)

        if err_tr[0] > initial_err * 1.1 and epoch > 100:
            print(f"    Adam degrading ({err_tr[0]:.4f} > "
                  f"{initial_err:.4f}) — stopping")
            break

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n    Phase 4 done: {elapsed:.0f}s, best_err={best_err:.4f}")
    return best_err, elapsed


# ════════════════════════════════════════════════
# Results Report
# ════════════════════════════════════════════════

def print_report(model, train_graphs, test_graphs,
                 train_batch, test_batch,
                 loss_fn, aw, cfg, device,
                 history, target_Pi, timings):
    model.eval()

    print(f"\n{'═' * 70}")
    print(f"  RESULTS REPORT   {datetime.now():%Y-%m-%d %H:%M:%S}")
    loss_tag = "Residual R²" if cfg.use_residual_loss else "Energy Π"
    print(f"  Loss mode: {loss_tag}")
    print(f"{'═' * 70}")

    # 1 ── Timing ──
    total_t = sum(timings.values())
    print(f"\n  ┌─ Training Time {'─' * 50}")
    for ph, t in timings.items():
        pct = 100 * t / total_t if total_t else 0
        print(f"  │  {ph:20s}  {t:8.1f}s  ({pct:4.0f}%)")
    print(f"  │  {'TOTAL':20s}  {total_t:8.1f}s")
    if history.phase_boundaries:
        phases_str = ' → '.join(p[0] for p in history.phase_boundaries)
        print(f"  │  Phases: {phases_str}")
    print(f"  └{'─' * 67}")

    # 2 ── Aggregate errors ──
    _, err_tr, _, ld_tr = evaluate_full(model, train_batch, loss_fn, aw, cfg)
    _, err_te, _, ld_te = evaluate_full(model, test_batch,  loss_fn, aw, cfg)

    print(f"\n  ┌─ Displacement Errors {'─' * 44}")
    print(f"  │  {'DOF':>8s}  {'Train':>10s}  {'Test':>10s}  {'Test/Tr':>10s}")
    print(f"  │  {'─' * 44}")
    for name, j in [('ux', 1), ('uz', 2), ('θ', 3)]:
        r = err_te[j] / max(err_tr[j], 1e-12)
        flag = " ⚠" if r > 2.0 else ""
        print(f"  │  {name:>8s}  {err_tr[j]:10.6f}  {err_te[j]:10.6f}  {r:10.2f}{flag}")
    r_all = err_te[0] / max(err_tr[0], 1e-12)
    flag  = " ⚠" if r_all > 2.0 else ""
    print(f"  │  {'─' * 44}")
    print(f"  │  {'ALL':>8s}  {err_tr[0]:10.6f}  {err_te[0]:10.6f}  {r_all:10.2f}{flag}")
    print(f"  └{'─' * 67}")
    if r_all > 2.0:
        print(f"  ⚠ Test/Train ratio > 2 — possible overfitting")

    # 3 ── Energy decomposition ──
    print(f"\n  ┌─ Energy {'─' * 57}")
    print(f"  │  {'':>20s}  {'Train':>12s}  {'Test':>12s}  {'Target':>12s}")
    print(f"  │  {'─' * 58}")
    print(f"  │  {'Π/E_c':>20s}  {ld_tr['Pi']:+12.4e}  "
          f"{ld_te['Pi']:+12.4e}  {target_Pi:+12.4e}")
    print(f"  │  {'U/W':>20s}  {ld_tr['U_over_W']:12.4f}  "
          f"{ld_te['U_over_W']:12.4f}  {'0.5000':>12s}")
    print(f"  │  {'Π/Π_tgt':>20s}  "
          f"{ld_tr['Pi']/target_Pi:12.4f}  "
          f"{ld_te['Pi']/target_Pi:12.4f}  {'1.0000':>12s}")
    for key in ('U_axial', 'U_bend', 'W_ext', 'U_total'):
        if key in ld_tr:
            print(f"  │  {key:>20s}  {float(ld_tr[key]):+12.4e}  "
                  f"{float(ld_te[key]):+12.4e}")
    print(f"  └{'─' * 67}")

    # 3b ── Residual equilibrium (only when using residual loss) ──
    if cfg.use_residual_loss:
        print(f"\n  ┌─ Residual R² (equilibrium check) {'─' * 31}")
        try:
            res_tr = evaluate_residual(model, train_batch, loss_fn, aw)
            res_te = evaluate_residual(model, test_batch,  loss_fn, aw)
            print(f"  │  {'DOF':>8s}  {'Train':>12s}  {'Test':>12s}")
            print(f"  │  {'─' * 36}")
            for name, k in [('ux', 'R_ux'), ('uz', 'R_uz'), ('θ', 'R_th')]:
                print(f"  │  {name:>8s}  {res_tr[k]:12.4e}  {res_te[k]:12.4e}")
            print(f"  │  {'─' * 36}")
            print(f"  │  {'TOTAL':>8s}  {res_tr['total']:12.4e}  {res_te['total']:12.4e}")
        except Exception as e:
            print(f"  │  (Residual eval failed: {e})")
            res_tr, res_te = None, None
        print(f"  └{'─' * 67}")

    # 4 ── Per-sample analysis ──
    ps_tr = per_sample_errors(model, train_graphs, loss_fn, aw, cfg, device)
    ps_te = per_sample_errors(model, test_graphs,  loss_fn, aw, cfg, device)
    tr_errs = [s['err_total'] for s in ps_tr]
    te_errs = [s['err_total'] for s in ps_te]

    print(f"\n  ┌─ Per-Sample Statistics {'─' * 43}")
    for tag, errs in [('Train', tr_errs), ('Test', te_errs)]:
        a = np.array(errs)
        print(f"  │  {tag} ({len(a)} samples):")
        print(f"  │    mean={a.mean():.6f}  std={a.std():.6f}  "
              f"median={np.median(a):.6f}")
        print(f"  │    min ={a.min():.6f}  max={a.max():.6f}")

    def _worst(ps, n=5):
        return sorted(ps, key=lambda x: -x['err_total'])[:n]

    print(f"  │")
    print(f"  │  Worst 5 TRAIN:")
    for s in _worst(ps_tr):
        print(f"  │    #{s['idx']:3d}  err={s['err_total']:.6f}  "
              f"[ux={s['err_ux']:.4f} uz={s['err_uz']:.4f} "
              f"θ={s['err_theta']:.4f}]  Π={s['Pi']:+.3e}  "
              f"U/W={s['U_over_W']:.3f}")
    print(f"  │  Worst 5 TEST:")
    for s in _worst(ps_te):
        print(f"  │    #{s['idx']:3d}  err={s['err_total']:.6f}  "
              f"[ux={s['err_ux']:.4f} uz={s['err_uz']:.4f} "
              f"θ={s['err_theta']:.4f}]  Π={s['Pi']:+.3e}  "
              f"U/W={s['U_over_W']:.3f}")
    print(f"  └{'─' * 67}")

    # 5 ── Quality ──
    if err_tr[0] < 0.02:    q = "✓ EXCELLENT (<2%)"
    elif err_tr[0] < 0.05:  q = "✓ GOOD (<5%)"
    elif err_tr[0] < 0.20:  q = "~ MODERATE (<20%)"
    else:                    q = "✗ NEEDS WORK (>20%)"

    gap = abs(err_te[0] - err_tr[0]) / max(err_tr[0], 1e-12)
    if   gap < 0.10: g = "✓ Excellent"
    elif gap < 0.50: g = "✓ Good"
    elif gap < 1.00: g = "~ Moderate gap"
    else:            g = "✗ Overfitting"

    print(f"\n  ┌─ Quality {'─' * 56}")
    print(f"  │  Accuracy:        {q}")
    print(f"  │  Generalization:  {g}  (gap={gap:.1%})")
    print(f"  │  Energy balance:  U/W = {ld_tr['U_over_W']:.4f}  "
          f"(target 0.5)")
    print(f"  └{'─' * 67}")

    return {
        'err_train': err_tr, 'err_test': err_te,
        'ld_train': {k: float(v) if hasattr(v, 'item') else v
                     for k, v in ld_tr.items()},
        'ld_test':  {k: float(v) if hasattr(v, 'item') else v
                     for k, v in ld_te.items()},
        'per_sample_train': ps_tr,
        'per_sample_test':  ps_te,
    }


# ════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════

def plot_curves(history, save_dir, best_err, target_Pi):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plots)")
        return

    recs = history.records
    if len(recs) < 2:
        print("  (too few records to plot)")
        return

    has_residual = any('R_total' in r for r in recs)

    phase_clr = {
        'adam_phase1': 'tab:blue',
        'uz_focused':  'tab:purple',
        'lbfgs':       'tab:orange',
        'adam_polish':  'tab:green',
    }

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    def _plot(ax, key, ylabel, log=True, test_key=None):
        for ph, clr in phase_clr.items():
            pr = [r for r in recs if r.get('phase') == ph and key in r]
            if not pr:
                continue
            x = [r['global_step'] for r in pr]
            y = [abs(r[key]) + 1e-30 for r in pr] if log else \
                [r[key] for r in pr]
            plot_fn = ax.semilogy if log else ax.plot
            plot_fn(x, y, color=clr, label=f"{ph} (train)", lw=1.5, alpha=.8)

            if test_key:
                pr_t = [r for r in pr if test_key in r]
                xt = [r['global_step'] for r in pr_t]
                yt = [abs(r[test_key]) + 1e-30 for r in pr_t] if log else \
                     [r[test_key] for r in pr_t]
                if xt:
                    plot_fn(xt, yt, color=clr, ls='--', alpha=.4,
                            label=f"{ph} (test)")

        ax.set_ylabel(ylabel); ax.set_xlabel('log-step')
        ax.legend(fontsize=7); ax.grid(True, alpha=.3)
        for _, s, _ in history.phase_boundaries:
            ax.axvline(s, color='grey', ls=':', alpha=.3)

    _plot(axes[0, 0], 'train_loss', '|Π/E_c|', test_key='test_loss')
    axes[0, 0].set_title('Potential Energy  (solid=train, dash=test)')

    _plot(axes[0, 1], 'train_err', 'Error', test_key='test_err')
    axes[0, 1].axhline(.05, c='g', ls=':', alpha=.5, label='5%')
    axes[0, 1].axhline(.02, c='r', ls=':', alpha=.5, label='2%')
    axes[0, 1].set_title('Displacement Error')
    axes[0, 1].legend(fontsize=7)

    _plot(axes[0, 2], 'U_over_W_train', 'U/W', log=False,
          test_key='U_over_W_test')
    axes[0, 2].axhline(.5, c='r', ls='--', alpha=.5, label='0.5')
    axes[0, 2].set_title('Energy Balance U/W')
    axes[0, 2].legend(fontsize=7)

    # Per-DOF displacement error
    for dof, key in [('ux', 'train_err_ux'),
                     ('uz', 'train_err_uz'),
                     ('θ',  'train_err_theta')]:
        xs = [r['global_step'] for r in recs if key in r]
        ys = [r[key] for r in recs if key in r]
        if xs:
            axes[1, 0].semilogy(xs, ys, label=dof, alpha=.8)
    axes[1, 0].set_title('Per-DOF Train Error')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=.3)
    axes[1, 0].set_xlabel('log-step')

    _plot(axes[1, 1], 'grad_norm', '|∇|')
    axes[1, 1].set_title('Gradient Norm')

    # axes[1, 2]: Residual per-DOF if available, else LR
    if has_residual:
        for dof, key in [('R_ux', 'R_ux'), ('R_uz', 'R_uz'), ('R_θ', 'R_th')]:
            xs = [r['global_step'] for r in recs if key in r]
            ys = [r[key] + 1e-30 for r in recs if key in r]
            if xs:
                axes[1, 2].semilogy(xs, ys, label=dof, alpha=.8)
        # total
        xs = [r['global_step'] for r in recs if 'R_total' in r]
        ys = [r['R_total'] + 1e-30 for r in recs if 'R_total' in r]
        if xs:
            axes[1, 2].semilogy(xs, ys, 'k-', lw=2, alpha=.6, label='total')
        axes[1, 2].set_title('Residual R² per DOF')
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].grid(True, alpha=.3)
        axes[1, 2].set_xlabel('log-step')
        for _, s, _ in history.phase_boundaries:
            axes[1, 2].axvline(s, color='grey', ls=':', alpha=.3)
    else:
        _plot(axes[1, 2], 'lr', 'LR', log=True)
        axes[1, 2].set_title('Learning Rate')

    loss_tag = "(Residual)" if has_residual else "(Energy)"
    plt.suptitle(f"Training Curves {loss_tag} — best err={best_err:.4f}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Curves saved: {p}")


def plot_predictions(model, test_graphs, loss_fn, aw, cfg, device,
                     save_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    model.eval()
    n_show = min(4, len(test_graphs))
    dof_names = ['ux', 'uz', 'θ']

    fig, axes = plt.subplots(n_show, 3, figsize=(15, 4 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_show):
        b = Batch.from_data_list([test_graphs[i]]).to(device)
        with torch.no_grad():
            pr = model(b)
        bv = b.batch if hasattr(b, 'batch') else None
        ux_c = _broadcast(b.ux_c, bv)
        uz_c_ = uz_scale(b, cfg)
        th_c = _broadcast(b.theta_c, bv)
        pred_phys = torch.stack([
            pr[:, 0] * ux_c, pr[:, 1] * uz_c_, pr[:, 2] * th_c
        ], dim=1).cpu().numpy()
        true_phys = b.y_node.cpu().numpy()

        for j in range(3):
            ax = axes[i, j]
            nids = np.arange(len(true_phys))
            ax.plot(nids, true_phys[:, j], 'k-o', ms=3, label='True')
            ax.plot(nids, pred_phys[:, j], 'r--s', ms=3, label='Pred')
            ax.set_title(f'Sample {i} — {dof_names[j]}')
            ax.legend(fontsize=7); ax.grid(True, alpha=.3)
            if i == n_show - 1:
                ax.set_xlabel('Node')

    plt.suptitle('Pred vs True (Test)', fontsize=14)
    plt.tight_layout()
    p = os.path.join(save_dir, 'pred_vs_true.png')
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Line plots: {p}")

    # ── scatter ──
    all_pred, all_true = [], []
    for data in test_graphs:
        b = Batch.from_data_list([data]).to(device)
        with torch.no_grad():
            pr = model(b)
        bv = b.batch if hasattr(b, 'batch') else None
        ux_c = _broadcast(b.ux_c, bv)
        uz_c_ = uz_scale(b, cfg)
        th_c = _broadcast(b.theta_c, bv)
        pp = torch.stack([
            pr[:, 0] * ux_c, pr[:, 1] * uz_c_, pr[:, 2] * th_c
        ], dim=1).cpu().numpy()
        all_pred.append(pp)
        all_true.append(b.y_node.cpu().numpy())
    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for j in range(3):
        ax = axes[j]
        ax.scatter(all_true[:, j], all_pred[:, j], s=5, alpha=.3)
        lo = min(all_true[:, j].min(), all_pred[:, j].min())
        hi = max(all_true[:, j].max(), all_pred[:, j].max())
        ax.plot([lo, hi], [lo, hi], 'r--', alpha=.5)
        ax.set_xlabel(f'True {dof_names[j]}')
        ax.set_ylabel(f'Pred {dof_names[j]}')
        ax.set_title(dof_names[j])
        ax.set_aspect('equal'); ax.grid(True, alpha=.3)
    plt.suptitle('Scatter (Test)', fontsize=14)
    plt.tight_layout()
    p = os.path.join(save_dir, 'scatter.png')
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Scatter:    {p}")


# ════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════

def main():
    cfg    = Config()
    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    loss_tag = "Residual R²" if cfg.use_residual_loss else "Energy Π"

    print("=" * 70)
    print("  PIGNN TRAINING — Full Pipeline v5")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Loss mode: {loss_tag}")
    print("=" * 70)

    # ── data ──
    print("\n── Loading data ──")
    data_list = load_data()
    d0 = data_list[0]
    for a in ('ux_c', 'uz_c', 'theta_c', 'F_c'):
        assert hasattr(d0, a), f"Missing {a}!"
    ensure_forward_mask(data_list)

    ux_c    = get_scalar(d0.ux_c)
    uz_c    = get_scalar(d0.uz_c)
    theta_c = get_scalar(d0.theta_c)
    F_c     = get_scalar(d0.F_c)
    E_c     = F_c * ux_c
    optimal_aw = compute_optimal_aw(d0)

    print(f"  Nodes={d0.num_nodes}  Elems={d0.n_elements}")
    print(f"  ux_c={ux_c:.4e}  uz_c={uz_c:.4e}  θ_c={theta_c:.4e}")
    print(f"  F_c={F_c:.4e}  E_c={E_c:.4e}")
    print(f"  optimal_aw={optimal_aw:.4e}")
    print(f"  use_ux_for_uz={cfg.use_ux_for_uz}")

    # ── split ──
    n_total = len(data_list)
    n_train = max(1, int(cfg.train_ratio * n_total))
    train_idx = list(range(n_train))
    test_idx  = (list(range(n_train, n_total))
                 if n_train < n_total
                 else list(range(n_total - 2, n_total)))
    train_graphs = [data_list[i] for i in train_idx]
    test_graphs  = [data_list[i] for i in test_idx]
    print(f"  Split: {len(train_graphs)} train, {len(test_graphs)} test")

    train_batch = Batch.from_data_list(train_graphs).to(device)
    test_batch  = Batch.from_data_list(test_graphs).to(device)
    loss_fn = FrameEnergyLoss().to(device)

    # ── target Π ──
    with torch.no_grad():
        true_d = train_batch.y_node.to(device)
        bv = train_batch.batch if hasattr(train_batch, 'batch') else None
        ux_ct = _broadcast(train_batch.ux_c, bv)
        uz_ct = uz_scale(train_batch, cfg)
        th_ct = _broadcast(train_batch.theta_c, bv)
        true_raw = torch.nan_to_num(torch.stack([
            true_d[:, 0] / ux_ct,
            true_d[:, 1] / uz_ct,
            true_d[:, 2] / th_ct,
        ], dim=1), 0.0)
        _, tgt = loss_fn.compute(true_raw, train_batch,
                                 axial_weight=optimal_aw)[:2]
        target_Pi = tgt['Pi']
    print(f"  Target Π/E_c={target_Pi:.4e}  U/W={tgt['U_over_W']:.4f}")

    # ── Active phases summary ──
    active_phases = []
    if cfg.adam_phase1_epochs > 0:
        active_phases.append(f"Adam({cfg.adam_phase1_epochs}ep)")
    if cfg.uz_epochs > 0:
        active_phases.append(f"uz-focus({cfg.uz_epochs}ep)")
    if cfg.lbfgs_steps > 0:
        active_phases.append(f"LBFGS({cfg.lbfgs_steps}st)")
    if cfg.adam_epochs > 0:
        active_phases.append(f"Polish({cfg.adam_epochs}ep)")
    if not active_phases:
        print("\n  ⚠ WARNING: All phases have 0 epochs/steps — "
              "model will NOT be trained!")
        active_phases.append("(none)")

    print(f"\n{'═' * 70}")
    print(f"  TRAINING  ({cfg.n_seeds} seed(s))")
    print(f"    Loss mode:     {loss_tag}")
    print(f"    Active phases: {' → '.join(active_phases)}")
    print(f"{'═' * 70}")

    best_overall_err   = 999.
    best_overall_state = None
    best_final_state   = None
    best_seed   = 0
    best_hist   = None
    best_times  = {}
    all_seed_summaries = []

    for seed in range(cfg.n_seeds):
        print(f"\n{'─' * 70}")
        print(f"  Seed {seed}")
        print(f"{'─' * 70}")

        torch.manual_seed(42 + seed * 1000)
        np.random.seed(42 + seed * 1000)

        model = PIGNN(
            node_in_dim=10, edge_in_dim=7,
            hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers,
            decoder_init_gain=cfg.init_gain,
        ).to(device)

        if seed == 0:
            model.summary()
            np_ = model.count_params()
            if cfg.lbfgs_steps > 0:
                print(f"  LBFGS history/params = "
                      f"{cfg.lbfgs_history}/{np_} = "
                      f"{100 * cfg.lbfgs_history / np_:.3f}%")

        with torch.no_grad():
            p = model(train_batch)
            assert p.abs().max() > 1e-8, "Zero init!"

        history = TrainingHistory()
        timings = {}

        # ── Phase 1: Adam mini-batch ──
        print(f"\n    ── Phase 1: Adam ({cfg.adam_phase1_epochs} ep) ──")
        e1, s1, t1 = train_adam_minibatch(
            model, train_graphs, train_batch, test_batch,
            loss_fn, optimal_aw, cfg, device, history)
        timings['adam_phase1'] = t1

        # ── Phase 2: uz-focused ──
        print(f"\n    ── Phase 2: uz-focused ({cfg.uz_epochs} ep) ──")
        e2_uz, s2_uz, t2_uz = train_uz_focused(
            model, train_graphs, train_batch, test_batch,
            loss_fn, optimal_aw, cfg, device, history)
        timings['uz_focused'] = t2_uz

        # ── Phase 3: L-BFGS ──
        print(f"\n    ── Phase 3: L-BFGS ({cfg.lbfgs_steps} steps) ──")
        e3, l3, t3 = train_lbfgs_with_restarts(
            model, train_batch, test_batch, loss_fn,
            optimal_aw, cfg, device, history, seed_id=seed)
        timings['lbfgs'] = t3

        # ── Phase 4: Adam polish ──
        with torch.no_grad():
            pc = model(train_batch)
        cur_err = compute_disp_error(pc, train_batch, cfg)[0]
        if cur_err > 0.01 and cfg.adam_epochs > 0:
            print(f"\n    ── Phase 4: Adam polish "
                  f"(err={cur_err:.4f}) ──")
            e4, t4 = adam_polish(
                model, train_batch, test_batch,
                loss_fn, optimal_aw, cfg, device, history)
            timings['adam_polish'] = t4
        else:
            reason = ("err < 1%" if cur_err <= 0.01
                      else "0 epochs configured")
            print(f"\n    ── Phase 4: SKIPPED ({reason}) ──")
            timings['adam_polish'] = 0.

        history.end_current_phase()

        # ── seed result ──
        with torch.no_grad():
            pf = model(train_batch)
        err_final = compute_disp_error(pf, train_batch, cfg)
        final_state = {k: v.cpu().clone()
                       for k, v in model.state_dict().items()}

        all_seed_summaries.append({
            'seed': seed,
            'err': err_final[0],
            'time': sum(timings.values()),
        })

        if err_final[0] < best_overall_err:
            best_overall_err   = err_final[0]
            best_overall_state = final_state
            best_final_state   = final_state
            best_seed  = seed
            best_hist  = history
            best_times = timings.copy()

        print(f"\n    Seed {seed}: err={err_final[0]:.4f}  "
              f"time={sum(timings.values()):.0f}s")

    if cfg.n_seeds > 1:
        print(f"\n  Seed summary:")
        for s in all_seed_summaries:
            star = " ★" if s['seed'] == best_seed else ""
            print(f"    seed={s['seed']}  err={s['err']:.4f}  "
                  f"t={s['time']:.0f}s{star}")

    print(f"\n{'═' * 70}")
    print(f"  Best: seed={best_seed}  err={best_overall_err:.4f}")
    print(f"{'═' * 70}")

    # ── Handle no training ──
    if best_overall_state is None:
        print("\n  ⚠ No training occurred — saving initial model")
        model = PIGNN(
            node_in_dim=10, edge_in_dim=7,
            hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers,
            decoder_init_gain=cfg.init_gain,
        ).to(device)
        best_overall_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
        best_final_state = best_overall_state
        best_hist = TrainingHistory()
        best_times = {
            'adam_phase1': 0., 'uz_focused': 0.,
            'lbfgs': 0., 'adam_polish': 0.,
        }

    # ── Load best model ──
    model = PIGNN(
        node_in_dim=10, edge_in_dim=7,
        hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers,
        decoder_init_gain=cfg.init_gain,
    ).to(device)
    model.load_state_dict(best_overall_state)
    model.eval()

    # ── Report ──
    report = print_report(
        model, train_graphs, test_graphs,
        train_batch, test_batch,
        loss_fn, optimal_aw, cfg, device,
        best_hist, target_Pi, best_times)

    # ── Save checkpoint ──
    ckpt_path = os.path.join(cfg.save_dir, 'checkpoint.pt')
    torch.save({
        'best_model_state':  best_overall_state,
        'final_model_state': best_final_state,
        'model_config': dict(
            node_in_dim=10, edge_in_dim=7,
            hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers,
            decoder_init_gain=cfg.init_gain),

        'history': best_hist.to_dict(),

        'config': {k: v for k, v in vars(cfg).items()
                   if not k.startswith('_')},

        'best_err':   best_overall_err,
        'best_seed':  best_seed,
        'target_Pi':  float(target_Pi),
        'optimal_aw': optimal_aw,
        'err_train':  report['err_train'],
        'err_test':   report['err_test'],
        'energy_train': report['ld_train'],
        'energy_test':  report['ld_test'],
        'per_sample_train': report['per_sample_train'],
        'per_sample_test':  report['per_sample_test'],

        'scales': dict(ux_c=ux_c, uz_c=uz_c, theta_c=theta_c,
                       F_c=F_c, E_c=E_c),

        'data_split': dict(train_indices=train_idx,
                           test_indices=test_idx,
                           n_total=n_total,
                           train_ratio=cfg.train_ratio),

        'timestamp':  ts,
        'timings':    best_times,
        'device':     str(device),
        'loss_mode':  'residual' if cfg.use_residual_loss else 'energy',
        'seed_summaries': all_seed_summaries,
    }, ckpt_path)
    print(f"\n  Checkpoint: {ckpt_path}")
    print(f"    ► best_model_state, final_model_state")
    print(f"    ► history (all phases), config, results")
    print(f"    ► scales, data_split, timings, loss_mode")

    # ── JSON summary ──
    summary = {
        'timestamp': ts,
        'loss_mode': 'residual' if cfg.use_residual_loss else 'energy',
        'best_seed': best_seed,
        'train_err': {
            'total': report['err_train'][0],
            'ux':    report['err_train'][1],
            'uz':    report['err_train'][2],
            'theta': report['err_train'][3],
        },
        'test_err': {
            'total': report['err_test'][0],
            'ux':    report['err_test'][1],
            'uz':    report['err_test'][2],
            'theta': report['err_test'][3],
        },
        'energy': {
            'target_Pi': float(target_Pi),
            'train_Pi':  report['ld_train']['Pi'],
            'test_Pi':   report['ld_test']['Pi'],
            'train_UW':  report['ld_train']['U_over_W'],
            'test_UW':   report['ld_test']['U_over_W'],
        },
        'data': {
            'n_train': len(train_graphs),
            'n_test':  len(test_graphs),
            'n_total': n_total,
        },
        'timing_s': best_times,
        'total_time_s': sum(best_times.values()),
        'config': {k: v for k, v in vars(cfg).items()
                   if not k.startswith('_')},
    }
    json_path = os.path.join(cfg.save_dir, 'summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary:    {json_path}")

    # ── Plots ──
    plot_curves(best_hist, cfg.save_dir, best_overall_err, target_Pi)
    plot_predictions(model, test_graphs, loss_fn, optimal_aw,
                     cfg, device, cfg.save_dir)

    print(f"\n{'═' * 70}")
    print(f"  DONE  ({loss_tag})")
    print(f"  Train err = {report['err_train'][0]:.6f}")
    print(f"  Test  err = {report['err_test'][0]:.6f}")
    print(f"  Time      = {sum(best_times.values()):.0f}s")
    print(f"{'═' * 70}")


if __name__ == '__main__':
    main()