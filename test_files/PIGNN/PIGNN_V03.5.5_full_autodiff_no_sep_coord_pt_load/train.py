"""
train.py — Train PIGNN with autograd physics loss
"""

import os
from pathlib import Path
print(f"Working directory: {os.getcwd()}")
CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)
print(f"Working directory: {os.getcwd()}")

"""
=================================================================
train.py — Training Loop for Naive Autograd PIGNN
=================================================================
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from model import PIGNN
from physics_loss import NaivePhysicsLoss


# ================================================================
# A. CONFIGURATION
# ================================================================

class TrainConfig:

    # ── Data ──
    data_path       = "DATA_15_case/graph_dataset_norm.pt"   # ◀◀◀ CHANGED (was graph_dataset.pt)
    save_dir        = "RESULTS"

    # ── Model ──
    hidden_dim      = 128
    n_layers        = 6
    node_in_dim     = 10
    edge_in_dim     = 7

    # ── Training ──
    epochs          = 300
    lr              = 1e-3
    weight_decay    = 1e-5
    # scheduler_step  = 10 # delete for cosine scheluler
    # scheduler_gamma = 0.5 # delete for cosine scheluler

    # ── Loss weights ──
    w_eq            = 1.0
    w_free          = 1.0
    w_sup           = 1.0
    w_N             = 1.0
    w_M             = 1.0
    w_V             = 1.0

    # ── Logging ──
    print_every     = 10
    save_every      = 100
    validate_every  = 50

    # ── Device ──
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================================================================
# B. TRAINING LOOP
# ================================================================

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
            'epoch':  [],
            'L_eq':   [], 'L_free': [], 'L_sup': [],
            'L_N':    [], 'L_M':    [], 'L_V':   [],
            'total':  [],
            'val_disp_error': [],
        }

    # ════════════════════════════════════════════════════════
    # ◀◀◀ CHANGED: _load_data() — adds PhysicsScaler + verification
    # ════════════════════════════════════════════════════════

    def _load_data(self):
        """Load graph dataset."""
        print(f"\n── Loading data ──")
        print(f"  Path: {self.cfg.data_path}")
        self.data_list = torch.load(
            self.cfg.data_path, weights_only=False
        )
        print(f"  Loaded {len(self.data_list)} graphs")

        # ── Add physics scales if missing ──                # ◀◀◀ NEW
        from normalizer import PhysicsScaler                  # ◀◀◀ NEW
        if not hasattr(self.data_list[0], 'F_c'):             # ◀◀◀ NEW
            print(f"  Physics scales NOT found — computing...")# ◀◀◀ NEW
            self.data_list = (                                 # ◀◀◀ NEW
                PhysicsScaler.compute_and_store_list(          # ◀◀◀ NEW
                    self.data_list                             # ◀◀◀ NEW
                )                                              # ◀◀◀ NEW
            )                                                  # ◀◀◀ NEW
        else:                                                  # ◀◀◀ NEW
            print(f"  Physics scales present ✓")               # ◀◀◀ NEW

        # ── Verify data state ──                             # ◀◀◀ NEW
        d = self.data_list[0]                                  # ◀◀◀ NEW
        print(f"\n  Data verification:")                       # ◀◀◀ NEW
        print(f"    x range:        [{d.x.min():.4f}, "        # ◀◀◀ NEW
              f"{d.x.max():.4f}]  "                            # ◀◀◀ NEW
              f"({'OK [0,1]' if d.x.max() < 2.0 else 'NOT NORMALIZED!'})")  # ◀◀◀ NEW
        print(f"    edge_attr range: [{d.edge_attr.min():.4f}, "  # ◀◀◀ NEW
              f"{d.edge_attr.max():.4f}]  "                    # ◀◀◀ NEW
              f"({'OK [0,1]' if d.edge_attr.max() < 2.0 else 'NOT NORMALIZED!'})")  # ◀◀◀ NEW
        print(f"    coords range:   [{d.coords.min():.2f}, "  # ◀◀◀ NEW
              f"{d.coords.max():.2f}]  (raw m)")               # ◀◀◀ NEW
        print(f"    prop_E range:   [{d.prop_E.min():.2e}, "   # ◀◀◀ NEW
              f"{d.prop_E.max():.2e}]  (raw Pa)")              # ◀◀◀ NEW
        print(f"    F_ext range:    [{d.F_ext.min():.2f}, "    # ◀◀◀ NEW
              f"{d.F_ext.max():.2f}]  (raw N)")                # ◀◀◀ NEW
        print(f"    F_c  = {d.F_c.item():.4e}  (char. force)")# ◀◀◀ NEW
        print(f"    M_c  = {d.M_c.item():.4e}  (char. moment)")# ◀◀◀ NEW
        print(f"    u_c  = {d.u_c.item():.4e}  (char. disp)") # ◀◀◀ NEW
        print(f"    θ_c  = {d.theta_c.item():.4e}  (char. rot)")# ◀◀◀ NEW

        self.train_data = self.data_list
        print(f"  Nodes: {d.num_nodes}, Elements: {d.n_elements}")

    # ════════════════════════════════════════════════════════
    # UNCHANGED from here
    # ════════════════════════════════════════════════════════

    def _create_model(self):
        print(f"\n── Creating model ──")
        self.model = PIGNN(
            node_in_dim=self.cfg.node_in_dim,
            edge_in_dim=self.cfg.edge_in_dim,
            hidden_dim=self.cfg.hidden_dim,
            n_layers=self.cfg.n_layers,
        ).to(self.device)
        self.model.summary()

    def _create_loss(self):
        self.loss_fn = NaivePhysicsLoss(
            w_eq=self.cfg.w_eq,
            w_free=self.cfg.w_free,
            w_sup=self.cfg.w_sup,
            w_N=self.cfg.w_N,
            w_M=self.cfg.w_M,
            w_V=self.cfg.w_V,
        )

    def _create_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        # self.scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer,
        #     step_size=self.cfg.scheduler_step,
        #     gamma=self.cfg.scheduler_gamma,
        # )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=1e-6
        )
        

    def train_one_epoch(self, epoch):
        self.model.train()

        epoch_losses = {
            'L_eq': 0, 'L_free': 0, 'L_sup': 0,
            'L_N': 0,  'L_M': 0,    'L_V': 0,
            'total': 0,
        }
        n_graphs = len(self.train_data)

        for data in self.train_data:
            data = data.to(self.device)

            self.optimizer.zero_grad()
            total_loss, loss_dict, pred = self.loss_fn(self.model, data)

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=10.0
            )

            self.optimizer.step()

            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]

        avg_losses = {k: v / n_graphs for k, v in epoch_losses.items()}
        return avg_losses

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        total_disp_error = 0.0
        n_graphs = 0

        for data in self.train_data:
            data = data.to(self.device)

            self.model.train()
            pred = self.model(data)
            self.model.eval()

            pred_disp = pred[:, 0:3]
            true_disp = data.y_node.to(self.device)

            err = (pred_disp - true_disp).pow(2).sum().sqrt()
            ref = true_disp.pow(2).sum().sqrt().clamp(min=1e-10)
            total_disp_error += (err / ref).item()

            n_graphs += 1

        val_metrics = {
            'disp_error': total_disp_error / max(n_graphs, 1),
        }
        return val_metrics

    def train(self):
        print(f"\n{'═'*60}")
        print(f"  TRAINING START")
        print(f"  {self.cfg.epochs} epochs, lr={self.cfg.lr}, "
              f"device={self.device}")
        print(f"{'═'*60}")

        header = (f"  {'Epoch':>5} | {'Total':>10} | "
                  f"{'L_eq':>10} {'L_free':>10} {'L_sup':>10} | "
                  f"{'L_N':>10} {'L_M':>10} {'L_V':>10} | "
                  f"{'Disp Err':>10}")
        print(f"\n{header}")
        print(f"  {'-'*105}")

        best_loss = float('inf')
        t_start = time.time()

        for epoch in range(1, self.cfg.epochs + 1):

            avg = self.train_one_epoch(epoch)

            self.history['epoch'].append(epoch)
            for key in ['L_eq', 'L_free', 'L_sup', 'L_N', 'L_M', 'L_V', 'total']:
                self.history[key].append(avg[key])

            disp_err = float('nan')
            if epoch % self.cfg.validate_every == 0 or epoch == 1:
                val = self.validate()
                disp_err = val['disp_error']
                self.history['val_disp_error'].append(disp_err)

            if epoch % self.cfg.print_every == 0 or epoch == 1:
                disp_str = f"{disp_err:.4e}" if not np.isnan(disp_err) else "    ---   "
                print(f"  {epoch:5d} | {avg['total']:10.4e} | "
                      f"{avg['L_eq']:10.4e} {avg['L_free']:10.4e} "
                      f"{avg['L_sup']:10.4e} | "
                      f"{avg['L_N']:10.4e} {avg['L_M']:10.4e} "
                      f"{avg['L_V']:10.4e} | "
                      f"{disp_str}")

            self.scheduler.step()

            if avg['total'] < best_loss:
                best_loss = avg['total']
                self._save_checkpoint('best.pt', epoch, avg)

            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(f'epoch_{epoch}.pt', epoch, avg)

        elapsed = time.time() - t_start
        print(f"\n  {'-'*105}")
        print(f"  Training complete: {elapsed:.1f}s "
              f"({elapsed/self.cfg.epochs:.2f}s/epoch)")
        print(f"  Best total loss: {best_loss:.6e}")

        self._save_checkpoint('final.pt', self.cfg.epochs, avg)
        self._save_history()
        self._plot_losses()
        self._analyze_results()

    def _save_checkpoint(self, filename, epoch, losses):
        path = os.path.join(self.cfg.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'losses': losses,
        }, path)

    def _save_history(self):
        path = os.path.join(self.cfg.save_dir, 'history.pt')
        torch.save(self.history, path)
        print(f"  History saved: {path}")

    def _plot_losses(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("  matplotlib not found, skipping plots")
            return

        epochs = self.history['epoch']

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle('Physics Loss Components (Autograd + Non-dim)',
                     fontsize=14, fontweight='bold')

        loss_info = [
            ('L_eq',   'Equilibrium',           'tab:blue',   'algebraic'),
            ('L_free', 'Free Face',             'tab:cyan',   'hard mask'),
            ('L_sup',  'Support BC',            'tab:green',  'hard BC'),
            ('L_N',    'Axial N=EA*du/ds',      'tab:orange', '1st deriv'),
            ('L_M',    'Moment M=EI*d2w/ds2',   'tab:red',    '2nd deriv'),
            ('L_V',    'Shear V=EI*d3w/ds3',    'tab:purple', '3rd deriv'),
        ]

        for idx, (key, title, color, note) in enumerate(loss_info):
            ax = axes[idx // 2, idx % 2]
            values = self.history[key]
            if any(v > 0 for v in values):
                ax.semilogy(epochs, values, color=color, linewidth=1.5)
            else:
                ax.plot(epochs, values, color=color, linewidth=1.5)
            ax.set_title(f'{title} ({note})', fontsize=10)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.cfg.save_dir, 'loss_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Loss curves saved: {path}")

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.semilogy(epochs, self.history['total'],
                     'k-', linewidth=2, label='Total')
        ax2.set_title('Total Physics Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        path2 = os.path.join(self.cfg.save_dir, 'total_loss.png')
        plt.savefig(path2, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Total loss saved: {path2}")

    def _analyze_results(self):
        print(f"\n{'='*60}")
        print(f"  RESULT ANALYSIS")
        print(f"{'='*60}")

        final = {k: self.history[k][-1] for k in
                 ['L_eq', 'L_free', 'L_sup', 'L_N', 'L_M', 'L_V', 'total']}

        best_idx = np.argmin(self.history['total'])
        best = {k: self.history[k][best_idx] for k in
                ['L_eq', 'L_free', 'L_sup', 'L_N', 'L_M', 'L_V', 'total']}
        best_epoch = self.history['epoch'][best_idx]

        print(f"\n  Final losses (epoch {self.cfg.epochs}):")
        print(f"  {'-'*50}")
        for key, val in final.items():
            status = self._classify_convergence(key, val)
            print(f"    {key:<10} {val:>12.6e}  {status}")

        print(f"\n  Best losses (epoch {best_epoch}):")
        print(f"  {'-'*50}")
        for key, val in best.items():
            status = self._classify_convergence(key, val)
            print(f"    {key:<10} {val:>12.6e}  {status}")

        print(f"\n  Convergence analysis:")
        print(f"  {'-'*50}")
        for key in ['L_eq', 'L_free', 'L_sup', 'L_N', 'L_M', 'L_V']:
            vals = self.history[key]
            if len(vals) >= 10:
                first_10 = np.mean(vals[:10])
                last_10 = np.mean(vals[-10:])
                ratio = last_10 / max(first_10, 1e-30)
                if ratio < 0.01:
                    trend = "CONVERGED (>100x reduction)"
                elif ratio < 0.1:
                    trend = "PARTIAL (10-100x reduction)"
                elif ratio < 1.0:
                    trend = "SLOW (<10x reduction)"
                else:
                    trend = "NOT CONVERGING"
                print(f"    {key:<10} first10={first_10:.4e} "
                      f"last10={last_10:.4e}  {trend}")

        if self.history['val_disp_error']:
            last_disp_err = self.history['val_disp_error'][-1]
            print(f"\n  Displacement error vs Kratos: {last_disp_err:.4e}")
            if last_disp_err > 0.5:
                print(f"  Poor accuracy")
            elif last_disp_err > 0.1:
                print(f"  Moderate accuracy")
            else:
                print(f"  Good accuracy")

    def _classify_convergence(self, key, val):
        if key in ['L_free', 'L_sup'] and val < 1e-10:
            return "(hard constraint)"
        elif val < 1e-6:
            return "converged"
        elif val < 1e-2:
            return "partial"
        else:
            return "not converged"


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 60)
    print("  PIGNN TRAINING — Autograd + Non-dim")
    print("=" * 60)
    print(f"  Device: {TrainConfig.device}")

    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Check {config.save_dir}/ for results")
    print(f"{'='*60}")