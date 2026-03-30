"""
=================================================================
train.py — Energy-Based Training v3
=================================================================

Fixes from v2 analysis:
  1. Lower lr (2e-4 instead of 5e-4)
  2. Gradient clipping (prevents spikes)
  3. Longer fine-tune phase
  4. Three-phase schedule: warmup → main → fine-tune
  5. Better early stopping (based on smoothed loss)
=================================================================
"""

import os
import time
import torch
import numpy as np
from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

from model import PIGNN
from energy_loss import FrameEnergyLoss
from step_2_grapg_constr import FrameData


# ════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════

class TrainConfig:

    data_path       = "DATA/graph_dataset_norm.pt"
    save_dir        = "RESULTS"

    hidden_dim      = 128
    n_layers        = 6
    node_in_dim     = 10
    edge_in_dim     = 7

    # ── Training schedule ──
    epochs          = 8000

    # Three-phase LR:
    #   Phase 1 (warm-up):   1-200,      ramp 0→lr
    #   Phase 2 (main):      201-3000,   lr
    #   Phase 3 (fine-tune): 3001-8000,  lr/10
    lr              = 1e-4 #2e-4
    warmup_epochs   = 500 #200
    decay_epoch     = 3000
    lr_decay_factor = 0.1
    weight_decay    = 0.0

    # ── Gradient clipping ──
    grad_clip       = 5.0 #500.0    # clip gradient norm
    batch_size      = 1 #4 #32     # 32 (total 400- training 350- 350/16=22 batches per epoch)

    # ── Convergence ──
    patience        = 1500
    min_delta       = 1e-3
    smooth_window   = 50       # smooth loss for stopping

    print_every     = 200
    save_every      = 2000
    validate_every  = 200

    train_ratio = 0.875  # 350 train, 50 test

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ════════════════════════════════════════════════
# TRAINER
# ════════════════════════════════════════════════

class Trainer:

    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        os.makedirs(config.save_dir, exist_ok=True)

        self._load_data()
        self._create_model()
        self._create_loss()
        self._create_optimizer()

        self.history = {
            'epoch':      [],
            'Pi':         [],
            'U':          [],
            'W':          [],
            'U_over_W':   [],
            'grad_norm':  [],
            'disp_err':   [],
            'lr':         [],
        }

        # Early stopping with smoothed loss
        self.best_Pi = float('inf')
        self.best_smooth_Pi = float('inf')
        self.best_epoch = 0
        self.no_improve_count = 0
        self._recent_Pi = []

    def _load_data(self):
        print(f"\n── Loading data ──")
        self.data_list = torch.load(
            self.cfg.data_path, weights_only=False
        )
        print(f"  Loaded {len(self.data_list)} graphs")

        # ═══════════════════════════════════════
        # Force ALL data to CPU
        # ═══════════════════════════════════════
        for i in range(len(self.data_list)):
            self.data_list[i] = self.data_list[i].cpu()

        from normalizer import PhysicsScaler
        if not hasattr(self.data_list[0], 'F_c'):
            self.data_list = (
                PhysicsScaler.compute_and_store_list(
                    self.data_list
                )
            )

        d = self.data_list[0]
        print(f"\n  Verification:")
        print(f"    Nodes: {d.num_nodes}, "
              f"Elements: {d.n_elements}")
        print(f"    F_c  = {d.F_c.item():.4e}")
        print(f"    u_c  = {d.u_c.item():.4e}")
        print(f"    θ_c  = {d.theta_c.item():.4e}")

        self.train_data = self.data_list

        self.raw_data = torch.load(
            "DATA/graph_dataset.pt", weights_only=False
        )
        self.raw_data = [d.cpu() for d in self.raw_data]

        from normalizer import PhysicsScaler
        if not hasattr(self.raw_data[0], 'u_c'):
            self.raw_data = (
                PhysicsScaler.compute_and_store_list(
                    self.raw_data
                )
            )

        # Target energies
        print(f"\n  Target energies:")
        from energy_loss import FrameEnergyLoss
        loss_fn = FrameEnergyLoss()
        self._target_energies = []
        for i, rd in enumerate(self.raw_data):
            u_true = rd.y_node
            U_t = loss_fn._strain_energy(u_true, rd)
            W_t = loss_fn._external_work(u_true, rd)
            Pi_t = U_t - W_t
            E_c = (rd.F_c * rd.u_c).clamp(min=1e-30)
            Pi_t_norm = (Pi_t / E_c).item()
            self._target_energies.append(Pi_t_norm)
            if i < 3:
                print(f"    Case {i}: Π = {Pi_t_norm:.4e}")

        self._avg_target = np.mean(self._target_energies)
        print(f"    Average: {self._avg_target:.4e}")

        # ── Train/Test split ──
        n = len(self.data_list)
        n_train = int(n * self.cfg.train_ratio)
        self.train_data = self.data_list[:n_train]
        self.test_data = self.data_list[n_train:]
        print(f"  Split: {len(self.train_data)} train, "
            f"{len(self.test_data)} test")

        # ── DataLoader for batching ──
        from torch_geometric.loader import DataLoader

        # self.train_loader = DataLoader(
        #     self.train_data,
        #     batch_size=32,
        #     shuffle=True
        # )
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True
        )

    def _create_model(self):
        print(f"\n── Creating model ──")
        self.model = PIGNN(
            node_in_dim=self.cfg.node_in_dim,
            edge_in_dim=self.cfg.edge_in_dim,
            hidden_dim=self.cfg.hidden_dim,
            n_layers=self.cfg.n_layers,
        ).to(self.device)
        self.model.summary()

        with torch.no_grad():
            d = self.data_list[0].clone().to(self.device)
            pred = self.model(d)
            print(f"  Initial pred max: "
                  f"{pred.abs().max().item():.6e}")

    def _create_loss(self):
        self.loss_fn = FrameEnergyLoss()

    def _create_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    def _get_lr(self, epoch):
        if epoch <= self.cfg.warmup_epochs:
            return (self.cfg.lr * epoch
                    / self.cfg.warmup_epochs)
        elif epoch <= self.cfg.decay_epoch:
            return self.cfg.lr
        else:
            return self.cfg.lr * self.cfg.lr_decay_factor

    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    # ════════════════════════════════════════
    # TRAINING
    # ════════════════════════════════════════

    # def train_one_epoch(self, epoch):
    #     self.model.train()

    #     lr = self._get_lr(epoch)
    #     self._set_lr(lr)

    #     # ═══════════════════════════════════════
    #     # USE DATALOADER FOR BATCHING
    #     # ═══════════════════════════════════════
    #     # from torch_geometric.loader import DataLoader

    #     # loader = DataLoader(
    #     #     self.train_data,
    #     #     batch_size=32,       # 400/32 = 13 batches
    #     #     shuffle=True
    #     # )

    #     epoch_Pi = 0.0
    #     epoch_U = 0.0
    #     epoch_W = 0.0
    #     epoch_grad_norm = 0.0
    #     last_dict = None
    #     n_graphs = len(self.train_data)

    #     for data in self.train_data:
    #         data = data.to(self.device)
    #         self.optimizer.zero_grad()

    #         Pi_norm, loss_dict, pred_raw, u_phys = \
    #             self.loss_fn(self.model, data)

    #         if torch.isnan(Pi_norm):
    #             print(f"  ⚠ NaN at epoch {epoch}")
    #             continue

    #         Pi_norm.backward()

    #         # ── Gradient clipping ──
    #         if self.cfg.grad_clip > 0:
    #             gn = torch.nn.utils.clip_grad_norm_(
    #                 self.model.parameters(),
    #                 max_norm=self.cfg.grad_clip
    #             ).item()
    #         else:
    #             gn = 0.0
    #             for p in self.model.parameters():
    #                 if p.grad is not None:
    #                     gn += p.grad.norm().item()**2
    #             gn = gn**0.5

    #         epoch_grad_norm += gn

    #         self.optimizer.step()

    #         epoch_Pi += loss_dict['Pi_norm']
    #         epoch_U  += loss_dict['U_internal']
    #         epoch_W  += loss_dict['W_external']
    #         last_dict = loss_dict

    #     avg_W = epoch_W / n_graphs
    #     avg_U = epoch_U / n_graphs

    #     avg = {
    #         'Pi':        epoch_Pi / n_graphs,
    #         'U':         avg_U,
    #         'W':         avg_W,
    #         'U_over_W':  (abs(avg_U)
    #                      / max(abs(avg_W), 1e-30)),
    #         'grad_norm': epoch_grad_norm / n_graphs,
    #         'lr':        lr,
    #         'last':      last_dict,
    #     }
    #     return avg

    # In train_one_epoch, replace the per-graph loop:

    def train_one_epoch(self, epoch):
        self.model.train()

        lr = self._get_lr(epoch)
        self._set_lr(lr)

        # ═══════════════════════════════════════
        # USE DATALOADER FOR BATCHING
        # ═══════════════════════════════════════
        from torch_geometric.loader import DataLoader

        loader = DataLoader(
            self.train_data,
            batch_size=32,       # 400/32 = 13 batches
            shuffle=True
        )

        epoch_Pi = 0.0
        epoch_U = 0.0
        epoch_W = 0.0
        epoch_grad_norm = 0.0
        last_dict = None
        n_graphs = len(self.train_data)

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            Pi_norm, loss_dict, pred_raw, u_phys = \
                self.loss_fn(self.model, batch)

            if torch.isnan(Pi_norm):
                continue

            Pi_norm.backward()

            if self.cfg.grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.cfg.grad_clip
                ).item()
            else:
                gn = 0.0

            epoch_grad_norm += gn
            self.optimizer.step()

            bs = batch.num_graphs
            epoch_Pi += loss_dict['Pi_norm'] * bs
            epoch_U  += loss_dict['U_internal'] * bs
            epoch_W  += loss_dict['W_external'] * bs
            last_dict = loss_dict

        avg_W = epoch_W / n_graphs
        avg_U = epoch_U / n_graphs

        avg = {
            'Pi':        epoch_Pi / n_graphs,
            'U':         avg_U,
            'W':         avg_W,
            'U_over_W':  (abs(avg_U)
                        / max(abs(avg_W), 1e-30)),
            'grad_norm': epoch_grad_norm / len(loader),
            'lr':        lr,
            'last':      last_dict,
        }
        return avg

    # ════════════════════════════════════════
    # VALIDATION
    # ════════════════════════════════════════

    # def validate(self):
    #     self.model.eval()
    #     total_err = 0.0
    #     n_graphs = 0

    #     with torch.no_grad():
    #         for i, data in enumerate(self.train_data):
    #             data = data.to(self.device)
    #             pred_raw = self.model(data)

    #             pred_phys = torch.zeros_like(pred_raw)
    #             pred_phys[:, 0] = (
    #                 pred_raw[:, 0] * data.u_c
    #             )
    #             pred_phys[:, 1] = (
    #                 pred_raw[:, 1] * data.u_c
    #             )
    #             pred_phys[:, 2] = (
    #                 pred_raw[:, 2] * data.theta_c
    #             )

    #             if i < len(self.raw_data):
    #                 true_disp = (
    #                     self.raw_data[i].y_node
    #                     .to(self.device)
    #                 )
    #                 err = ((pred_phys - true_disp)
    #                        .pow(2).sum().sqrt())
    #                 ref = (true_disp.pow(2).sum()
    #                        .sqrt().clamp(min=1e-10))
    #                 total_err += (err / ref).item()
    #                 n_graphs += 1

    #     return {
    #         'disp_err': total_err / max(n_graphs, 1)
    #     }

    def validate(self):
        self.model.eval()
        total_err = 0.0
        n_graphs = 0

        n_train = len(self.train_data)

        with torch.no_grad():
            for i, data in enumerate(self.test_data):
                data = data.to(self.device)
                pred_raw = self.model(data)

                pred_phys = torch.zeros_like(pred_raw)
                pred_phys[:, 0] = (
                    pred_raw[:, 0] * data.u_c
                )
                pred_phys[:, 1] = (
                    pred_raw[:, 1] * data.u_c
                )
                pred_phys[:, 2] = (
                    pred_raw[:, 2] * data.theta_c
                )

                raw_idx = n_train + i
                if raw_idx < len(self.raw_data):
                    true_disp = (
                        self.raw_data[raw_idx].y_node
                        .to(self.device)
                    )
                    err = ((pred_phys - true_disp)
                        .pow(2).sum().sqrt())
                    ref = (true_disp.pow(2).sum()
                        .sqrt().clamp(min=1e-10))
                    total_err += (err / ref).item()
                    n_graphs += 1

        return {
            'disp_err': total_err / max(n_graphs, 1)
        }

    # ════════════════════════════════════════
    # SMOOTHED EARLY STOPPING
    # ════════════════════════════════════════

    def _check_convergence(self, epoch, Pi):
        self._recent_Pi.append(Pi)
        if len(self._recent_Pi) > self.cfg.smooth_window:
            self._recent_Pi.pop(0)

        if len(self._recent_Pi) >= self.cfg.smooth_window:
            smooth_Pi = np.mean(self._recent_Pi)

            if smooth_Pi < self.best_smooth_Pi - self.cfg.min_delta:
                self.best_smooth_Pi = smooth_Pi
                self.best_epoch = epoch
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1

        # Always track absolute best
        if Pi < self.best_Pi:
            self.best_Pi = Pi
            self._save_checkpoint('best.pt', epoch,
                                  {'Pi': Pi})

        return (self.no_improve_count >=
                self.cfg.patience
                and epoch > self.cfg.decay_epoch)

    # ════════════════════════════════════════
    # MAIN LOOP
    # ════════════════════════════════════════

    def train(self):
        print(f"\n{'═'*95}")
        print(f"  TRAINING — Energy-Based PIGNN v3")
        print(f"  {self.cfg.epochs} epochs, "
              f"lr={self.cfg.lr}, "
              f"grad_clip={self.cfg.grad_clip}, "
              f"device={self.device}")
        print(f"  Warmup:    1-{self.cfg.warmup_epochs}")
        print(f"  Main:      "
              f"{self.cfg.warmup_epochs+1}-"
              f"{self.cfg.decay_epoch}")
        print(f"  Fine-tune: "
              f"{self.cfg.decay_epoch+1}-"
              f"{self.cfg.epochs}")
        print(f"  Target Π:  {self._avg_target:.4e}")
        print(f"{'═'*95}")

        header = (
            f"  {'Ep':>5} | {'Π_norm':>11} | "
            f"{'Π/Π_t':>7} | "
            f"{'U/W':>7} | "
            f"{'|∇|':>9} | "
            f"{'DispErr':>10} | "
            f"{'LR':>9}"
        )
        print(f"\n{header}")
        print(f"  {'-'*85}")

        best_disp_err = float('inf')
        t_start = time.time()

        for epoch in range(1, self.cfg.epochs + 1):

            avg = self.train_one_epoch(epoch)

            self.history['epoch'].append(epoch)
            self.history['Pi'].append(avg['Pi'])
            self.history['U'].append(avg['U'])
            self.history['W'].append(avg['W'])
            self.history['U_over_W'].append(
                avg['U_over_W']
            )
            self.history['grad_norm'].append(
                avg['grad_norm']
            )
            self.history['lr'].append(avg['lr'])

            # ── Validation ──
            disp_err = float('nan')
            if (epoch % self.cfg.validate_every == 0
                    or epoch == 1
                    or epoch == self.cfg.epochs):
                val = self.validate()
                disp_err = val['disp_err']
                if not np.isnan(disp_err):
                    best_disp_err = min(
                        best_disp_err, disp_err
                    )
            self.history['disp_err'].append(disp_err)

            # ── Convergence check ──
            should_stop = self._check_convergence(
                epoch, avg['Pi']
            )

            # ── Print ──
            if (epoch % self.cfg.print_every == 0
                    or epoch == 1
                    or epoch == self.cfg.epochs):
                de = (f"{disp_err:.4e}"
                      if not np.isnan(disp_err)
                      else "    ---   ")

                ratio = (avg['Pi']
                        / self._avg_target
                        if abs(self._avg_target) > 1e-30
                        else 0)

                print(
                    f"  {epoch:5d} | "
                    f"{avg['Pi']:11.4e} | "
                    f"{ratio:7.4f} | "
                    f"{avg['U_over_W']:7.4f} | "
                    f"{avg['grad_norm']:9.1e} | "
                    f"{de} | "
                    f"{avg['lr']:9.1e}"
                )

                d = avg['last']
                if d and (epoch % (self.cfg.print_every*2)
                         == 0 or epoch == 1):
                    print(
                        f"        raw: "
                        f"[{d['raw_range'][0]:.4f}, "
                        f"{d['raw_range'][1]:.4f}]  "
                        f"ux: "
                        f"[{d['ux_range'][0]:.4e}, "
                        f"{d['ux_range'][1]:.4e}]  "
                        f"θ: "
                        f"[{d['th_range'][0]:.4e}, "
                        f"{d['th_range'][1]:.4e}]"
                    )

            # ── Periodic save ──
            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(
                    f'epoch_{epoch}.pt', epoch, avg
                )

            # ── Phase transition log ──
            if epoch == self.cfg.decay_epoch + 1:
                print(f"\n  >>> Fine-tune phase: "
                      f"LR → {self.cfg.lr * self.cfg.lr_decay_factor:.1e} "
                      f"<<<\n")

            # ── Early stopping ──
            if should_stop:
                print(f"\n  Early stopping at epoch "
                      f"{epoch}")
                break

        elapsed = time.time() - t_start
        print(f"\n  {'-'*85}")
        print(f"  Done: {elapsed:.1f}s "
              f"({elapsed/epoch:.2f}s/epoch)")
        print(f"  Best Π_norm:   {self.best_Pi:.6e} "
              f"(epoch {self.best_epoch})")
        print(f"  Target Π_norm: {self._avg_target:.6e}")
        print(f"  Ratio:         "
              f"{self.best_Pi/self._avg_target:.4f}")
        print(f"  Best disp_err: {best_disp_err:.6e}")

        self._save_checkpoint(
            'final.pt', epoch, avg
        )
        self._save_history()
        self._plot_losses()
        self._analyze_results()

    # ════════════════════════════════════════
    # UTILITIES
    # ════════════════════════════════════════

    def _save_checkpoint(self, filename, epoch, losses):
        path = os.path.join(self.cfg.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state':
                self.optimizer.state_dict(),
            'losses': losses,
            'best_Pi': self.best_Pi,
        }, path)

    def _save_history(self):
        path = os.path.join(
            self.cfg.save_dir, 'history.pt'
        )
        torch.save(self.history, path)
        print(f"  History saved: {path}")

    def _plot_losses(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        epochs = self.history['epoch']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            'Energy-Based PIGNN Training v3',
            fontsize=14, fontweight='bold'
        )

        # 1. Π_norm
        ax = axes[0, 0]
        ax.plot(epochs, self.history['Pi'],
                'b-', linewidth=0.5, alpha=0.5,
                label='Raw')
        # Smoothed
        w = self.cfg.smooth_window
        if len(epochs) > w:
            smooth = np.convolve(
                self.history['Pi'],
                np.ones(w)/w, mode='valid'
            )
            ax.plot(epochs[w-1:], smooth,
                    'b-', linewidth=2, label='Smoothed')
        ax.axhline(y=self._avg_target, color='r',
                   linestyle='--', linewidth=2,
                   label=f'Target: {self._avg_target:.2f}')
        ax.axhline(y=0, color='gray', linestyle=':')
        ax.set_title('Π_norm')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Π/Π_target ratio
        ax = axes[0, 1]
        ratios = [
            p / self._avg_target
            if abs(self._avg_target) > 1e-30 else 0
            for p in self.history['Pi']
        ]
        ax.plot(epochs, ratios,
                'g-', linewidth=0.5, alpha=0.5)
        if len(epochs) > w:
            smooth_r = np.convolve(
                ratios, np.ones(w)/w, mode='valid'
            )
            ax.plot(epochs[w-1:], smooth_r,
                    'g-', linewidth=2)
        ax.axhline(y=1.0, color='r', linestyle='--',
                   label='Target (1.0)')
        ax.set_title('Π/Π_target (→ 1.0)')
        ax.set_ylim(0, 1.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. U/W ratio
        ax = axes[0, 2]
        ax.plot(epochs, self.history['U_over_W'],
                'k-', linewidth=0.5, alpha=0.5)
        if len(epochs) > w:
            smooth_uw = np.convolve(
                self.history['U_over_W'],
                np.ones(w)/w, mode='valid'
            )
            ax.plot(epochs[w-1:], smooth_uw,
                    'k-', linewidth=2)
        ax.axhline(y=0.5, color='r', linestyle='--',
                   label='Optimal (0.5)')
        ax.set_title('U/|W| (→ 0.5)')
        ax.set_ylim(0, 2)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Gradient norm
        ax = axes[1, 0]
        ax.semilogy(epochs,
                    self.history['grad_norm'],
                    'purple', linewidth=0.5, alpha=0.7)
        ax.axhline(y=self.cfg.grad_clip, color='r',
                   linestyle='--', alpha=0.5,
                   label=f'Clip: {self.cfg.grad_clip}')
        ax.set_title('Gradient Norm (clipped)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Displacement error
        ax = axes[1, 1]
        valid_de = [
            (e, d) for e, d in
            zip(epochs, self.history['disp_err'])
            if not np.isnan(d)
        ]
        if valid_de:
            de_e, de_v = zip(*valid_de)
            ax.semilogy(de_e, de_v,
                       'mo-', linewidth=1.5, ms=3)
        ax.set_title('Displacement Error')
        ax.grid(True, alpha=0.3)

        # 6. Learning rate
        ax = axes[1, 2]
        ax.semilogy(epochs, self.history['lr'],
                    'orange', linewidth=2)
        ax.set_title('Learning Rate')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(
            self.cfg.save_dir, 'loss_curves.png'
        )
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Loss curves saved: {path}")

    def _analyze_results(self):
        print(f"\n{'='*60}")
        print(f"  RESULT ANALYSIS")
        print(f"{'='*60}")

        final_Pi = self.history['Pi'][-1]
        print(f"\n  Final Π_norm:  {final_Pi:.6e}")
        print(f"  Best  Π_norm:  {self.best_Pi:.6e}")
        print(f"  Target Π_norm: {self._avg_target:.6e}")

        ratio = self.best_Pi / self._avg_target
        print(f"  Ratio: {ratio:.4f}")
        if abs(ratio - 1.0) < 0.01:
            print(f"  ✓ CONVERGED (<1% error)")
        elif abs(ratio - 1.0) < 0.05:
            print(f"  ~ Close (<5%)")
        elif abs(ratio - 1.0) < 0.1:
            print(f"  ~ Fair (<10%)")
        else:
            print(f"  ✗ Not converged")

        # Gradient analysis
        gn = self.history['grad_norm']
        if len(gn) > 100:
            gn_start = np.mean(gn[:100])
            gn_end = np.mean(gn[-100:])
            print(f"\n  Gradient norm:")
            print(f"    Start: {gn_start:.1e}")
            print(f"    End:   {gn_end:.1e}")
            if gn_end < gn_start:
                print(f"    ✓ Decreasing (converging)")
            else:
                print(f"    ⚠ Not decreasing")

        # Displacement error
        valid_de = [
            d for d in self.history['disp_err']
            if not np.isnan(d)
        ]
        if valid_de:
            best_de = min(valid_de)
            print(f"\n  Best displacement error: "
                  f"{best_de:.4e}")
            if best_de < 0.01:
                print(f"  ACCURACY: EXCELLENT (<1%)")
            elif best_de < 0.05:
                print(f"  ACCURACY: GOOD (<5%)")
            elif best_de < 0.1:
                print(f"  ACCURACY: FAIR (<10%)")
            elif best_de < 0.2:
                print(f"  ACCURACY: MODERATE (<20%)")
            else:
                print(f"  ACCURACY: NEEDS MORE TRAINING")


# ════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("  PIGNN TRAINING — Energy-Based v3")
    print("=" * 60)

    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")