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

"""
=================================================================
train.py — Training Loop for Naive Autograd PIGNN
=================================================================

PURPOSE:
  Train PIGNN using ONLY physics losses (no labelled data).
  Observe which losses converge and which fail.

EXPECTED OBSERVATIONS:
  ✅ L_eq   — should converge (pure algebra on predictions)
  ✅ L_free — should converge (trivial with hard mask)
  ✅ L_sup  — should converge (trivial with hard BC)
  ⚠️ L_N    — may partially work (1st derivative, least noisy)
  ❌ L_M    — problematic (2nd derivative through GNN)
  ❌ L_V    — will fail (3rd derivative = noise)

USAGE:
  python train.py
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
    """
    All training hyperparameters in one place.
    Easy to modify, easy to reproduce.
    """

    # ── Data ──
    data_path       = "DATA/graph_dataset.pt"
    save_dir        = "RESULTS"

    # ── Model ──
    hidden_dim      = 128
    n_layers        = 6
    node_in_dim     = 9
    edge_in_dim     = 10

    # ── Training ──
    epochs          = 500
    lr              = 1e-3
    weight_decay    = 1e-5
    scheduler_step  = 100
    scheduler_gamma = 0.5

    # ── Loss weights ──
    # Start equal, observe, then adjust
    w_eq            = 1.0
    w_free          = 1.0
    w_sup           = 1.0
    w_N             = 1.0
    w_M             = 1.0
    w_V             = 1.0

    # ── Logging ──
    print_every     = 10      # print loss every N epochs
    save_every      = 100     # save checkpoint every N epochs
    validate_every  = 50      # compare with ground truth every N epochs

    # ── Device ──
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================================================================
# B. TRAINING LOOP
# ================================================================

class Trainer:
    """
    Training manager.

    Handles:
      - Model creation
      - Loss computation
      - Optimization
      - Logging of all 6 individual losses
      - Validation against ground truth
      - Checkpointing
    """

    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        # ── Create save directory ──
        os.makedirs(config.save_dir, exist_ok=True)

        # ── Load data ──
        self._load_data()

        # ── Create model ──
        self._create_model()

        # ── Create loss ──
        self._create_loss()

        # ── Create optimizer ──
        self._create_optimizer()

        # ── History for plotting ──
        self.history = {
            'epoch':  [],
            'L_eq':   [], 'L_free': [], 'L_sup': [],
            'L_N':    [], 'L_M':    [], 'L_V':   [],
            'total':  [],
            'val_disp_error': [],
        }

    def _load_data(self):
        """Load graph dataset (unnormalized — physics needs raw units)."""
        print(f"\n── Loading data ──")
        self.data_list = torch.load(
            self.cfg.data_path, weights_only=False
        )
        print(f"  Loaded {len(self.data_list)} graphs")
        print(f"  Using UNNORMALIZED data (physics loss needs raw units)")

        # For now: train on all graphs
        # Later: add train/val split
        self.train_data = self.data_list
        d = self.data_list[0]
        print(f"  Nodes: {d.num_nodes}, Elements: {d.n_elements}")

    def _create_model(self):
        """Create PIGNN model."""
        print(f"\n── Creating model ──")
        self.model = PIGNN(
            node_in_dim=self.cfg.node_in_dim,
            edge_in_dim=self.cfg.edge_in_dim,
            hidden_dim=self.cfg.hidden_dim,
            n_layers=self.cfg.n_layers,
        ).to(self.device)
        self.model.summary()

    def _create_loss(self):
        """Create physics loss function."""
        self.loss_fn = NaivePhysicsLoss(
            w_eq=self.cfg.w_eq,
            w_free=self.cfg.w_free,
            w_sup=self.cfg.w_sup,
            w_N=self.cfg.w_N,
            w_M=self.cfg.w_M,
            w_V=self.cfg.w_V,
        )

    def _create_optimizer(self):
        """Create optimizer and scheduler."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.cfg.scheduler_step,
            gamma=self.cfg.scheduler_gamma,
        )

    # ────────────────────────────────────────

    def train_one_epoch(self, epoch):
        """
        Train one epoch over all graphs.

        Returns:
            avg_losses: dict of average losses over all graphs
        """
        self.model.train()

        epoch_losses = {
            'L_eq': 0, 'L_free': 0, 'L_sup': 0,
            'L_N': 0,  'L_M': 0,    'L_V': 0,
            'total': 0,
        }
        n_graphs = len(self.train_data)

        for data in self.train_data:
            data = data.to(self.device)

            # ── Forward + physics loss ──
            self.optimizer.zero_grad()
            total_loss, loss_dict, pred = self.loss_fn(self.model, data)

            # ── Backward ──
            total_loss.backward()

            # ── Gradient clipping (safety for noisy 3rd derivatives) ──
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=10.0
            )

            # ── Update ──
            self.optimizer.step()

            # ── Accumulate ──
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]

        # ── Average over graphs ──
        avg_losses = {k: v / n_graphs for k, v in epoch_losses.items()}
        return avg_losses

    @torch.no_grad()
    def validate(self):
        """
        Compare predictions with Kratos ground truth.

        Returns:
            val_metrics: dict with error metrics
        """
        self.model.eval()

        total_disp_error = 0.0
        total_force_error = 0.0
        n_graphs = 0

        for data in self.train_data:
            data = data.to(self.device)

            # Need coords with grad for forward pass
            # But we don't need loss gradients
            self.model.train()  # needed for coord injection
            pred = self.model(data)
            self.model.eval()

            # ── Displacement error ──
            pred_disp = pred[:, 0:3]                     # (N, 3)
            true_disp = data.y_node.to(self.device)       # (N, 3)

            # Relative L2 error
            err = (pred_disp - true_disp).pow(2).sum().sqrt()
            ref = true_disp.pow(2).sum().sqrt().clamp(min=1e-10)
            total_disp_error += (err / ref).item()

            n_graphs += 1

        val_metrics = {
            'disp_error': total_disp_error / max(n_graphs, 1),
        }
        return val_metrics

    # ────────────────────────────────────────

    def train(self):
        """
        Full training loop.

        Logs all 6 losses separately per epoch.
        Validates against ground truth periodically.
        """
        print(f"\n{'═'*60}")
        print(f"  TRAINING START")
        print(f"  {self.cfg.epochs} epochs, lr={self.cfg.lr}, "
              f"device={self.device}")
        print(f"{'═'*60}")

        # ── Header ──
        header = (f"  {'Epoch':>5} │ {'Total':>10} │ "
                  f"{'L_eq':>10} {'L_free':>10} {'L_sup':>10} │ "
                  f"{'L_N':>10} {'L_M':>10} {'L_V':>10} │ "
                  f"{'Disp Err':>10}")
        print(f"\n{header}")
        print(f"  {'─'*105}")

        best_loss = float('inf')
        t_start = time.time()

        for epoch in range(1, self.cfg.epochs + 1):

            # ── Train ──
            avg = self.train_one_epoch(epoch)

            # ── Record ──
            self.history['epoch'].append(epoch)
            for key in ['L_eq', 'L_free', 'L_sup', 'L_N', 'L_M', 'L_V', 'total']:
                self.history[key].append(avg[key])

            # ── Validate ──
            disp_err = float('nan')
            if epoch % self.cfg.validate_every == 0 or epoch == 1:
                val = self.validate()
                disp_err = val['disp_error']
                self.history['val_disp_error'].append(disp_err)

            # ── Print ──
            if epoch % self.cfg.print_every == 0 or epoch == 1:
                disp_str = f"{disp_err:.4e}" if not np.isnan(disp_err) else "    —     "
                print(f"  {epoch:5d} │ {avg['total']:10.4e} │ "
                      f"{avg['L_eq']:10.4e} {avg['L_free']:10.4e} "
                      f"{avg['L_sup']:10.4e} │ "
                      f"{avg['L_N']:10.4e} {avg['L_M']:10.4e} "
                      f"{avg['L_V']:10.4e} │ "
                      f"{disp_str}")

            # ── Scheduler ──
            self.scheduler.step()

            # ── Save best ──
            if avg['total'] < best_loss:
                best_loss = avg['total']
                self._save_checkpoint('best.pt', epoch, avg)

            # ── Periodic save ──
            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(f'epoch_{epoch}.pt', epoch, avg)

        # ── Training complete ──
        elapsed = time.time() - t_start
        print(f"\n  {'─'*105}")
        print(f"  Training complete: {elapsed:.1f}s "
              f"({elapsed/self.cfg.epochs:.2f}s/epoch)")
        print(f"  Best total loss: {best_loss:.6e}")

        # ── Save final ──
        self._save_checkpoint('final.pt', self.cfg.epochs, avg)
        self._save_history()
        self._plot_losses()

        # ── Final analysis ──
        self._analyze_results()

    # ────────────────────────────────────────

    def _save_checkpoint(self, filename, epoch, losses):
        """Save model checkpoint."""
        path = os.path.join(self.cfg.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'losses': losses,
        }, path)

    def _save_history(self):
        """Save loss history for later analysis."""
        path = os.path.join(self.cfg.save_dir, 'history.pt')
        torch.save(self.history, path)
        print(f"  History saved: {path}")

    # ────────────────────────────────────────

    def _plot_losses(self):
        """
        Plot all 6 losses on separate subplots.

        This is the KEY visualization — shows which losses
        converge and which don't.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("  ⚠ matplotlib not found, skipping plots")
            return

        epochs = self.history['epoch']

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle('Physics Loss Components (Naive Autograd)',
                     fontsize=14, fontweight='bold')

        loss_info = [
            ('L_eq',   'Equilibrium (Concept 1)',     'tab:blue',   '✅ Should converge'),
            ('L_free', 'Free Face (Concept 1)',       'tab:cyan',   '✅ Trivial with hard mask'),
            ('L_sup',  'Support BC (Concept 1)',      'tab:green',  '✅ Trivial with hard BC'),
            ('L_N',    'Axial N=EA·du/ds (Concept 3)','tab:orange', '⚠️ 1st derivative'),
            ('L_M',    'Moment M=EI·d²w/ds²',        'tab:red',    '❌ 2nd derivative problem'),
            ('L_V',    'Shear V=EI·d³w/ds³',         'tab:purple', '❌ 3rd derivative noise'),
        ]

        for idx, (key, title, color, note) in enumerate(loss_info):
            ax = axes[idx // 2, idx % 2]
            values = self.history[key]
            ax.semilogy(epochs, values, color=color, linewidth=1.5)
            ax.set_title(f'{title}\n{note}', fontsize=10)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.cfg.save_dir, 'loss_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Loss curves saved: {path}")

        # ── Also plot total loss ──
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

    # ────────────────────────────────────────

    def _analyze_results(self):
        """
        Print analysis of final loss values.

        This is where we document the observed problems
        to motivate Method C.
        """
        print(f"\n{'═'*60}")
        print(f"  RESULT ANALYSIS")
        print(f"{'═'*60}")

        # Get final epoch values
        final = {k: self.history[k][-1] for k in
                 ['L_eq', 'L_free', 'L_sup', 'L_N', 'L_M', 'L_V', 'total']}

        # Get best epoch values
        best_idx = np.argmin(self.history['total'])
        best = {k: self.history[k][best_idx] for k in
                ['L_eq', 'L_free', 'L_sup', 'L_N', 'L_M', 'L_V', 'total']}
        best_epoch = self.history['epoch'][best_idx]

        print(f"\n  Final losses (epoch {self.cfg.epochs}):")
        print(f"  {'─'*50}")
        for key, val in final.items():
            status = self._classify_convergence(key, val)
            print(f"    {key:<10} {val:>12.6e}  {status}")

        print(f"\n  Best losses (epoch {best_epoch}):")
        print(f"  {'─'*50}")
        for key, val in best.items():
            status = self._classify_convergence(key, val)
            print(f"    {key:<10} {val:>12.6e}  {status}")

        # ── Convergence analysis ──
        print(f"\n  Convergence analysis:")
        print(f"  {'─'*50}")
        for key in ['L_eq', 'L_free', 'L_sup', 'L_N', 'L_M', 'L_V']:
            vals = self.history[key]
            if len(vals) >= 10:
                first_10 = np.mean(vals[:10])
                last_10 = np.mean(vals[-10:])
                ratio = last_10 / max(first_10, 1e-30)
                if ratio < 0.01:
                    trend = "✅ CONVERGED (>100x reduction)"
                elif ratio < 0.1:
                    trend = "⚠️ PARTIAL (10-100x reduction)"
                elif ratio < 1.0:
                    trend = "⚠️ SLOW (<10x reduction)"
                else:
                    trend = "❌ NOT CONVERGING"
                print(f"    {key:<10} first10={first_10:.4e} "
                      f"last10={last_10:.4e}  {trend}")

        # ── Displacement accuracy ──
        if self.history['val_disp_error']:
            last_disp_err = self.history['val_disp_error'][-1]
            print(f"\n  Displacement error vs Kratos: {last_disp_err:.4e}")
            if last_disp_err > 0.5:
                print(f"  ❌ Poor accuracy — physics loss alone insufficient")
            elif last_disp_err > 0.1:
                print(f"  ⚠️ Moderate accuracy")
            else:
                print(f"  ✅ Good accuracy")

        # ── Expected problems ──
        print(f"\n  {'═'*50}")
        print(f"  EXPECTED PROBLEMS TO OBSERVE:")
        print(f"  {'─'*50}")
        print(f"  1. L_M (moment) does NOT converge:")
        print(f"     → ∂²uz/∂x² through GNN ≠ physical curvature")
        print(f"     → autograd measures GNN sensitivity, not beam bending")
        print(f"")
        print(f"  2. L_V (shear) is NOISY or DIVERGES:")
        print(f"     → 3rd derivative through GNN = amplified noise")
        print(f"     → chain rule through 3 levels of message passing")
        print(f"")
        print(f"  3. L_N (axial) may PARTIALLY work:")
        print(f"     → 1st derivative is less noisy")
        print(f"     → but still contaminated by .sum() in autograd")
        print(f"")
        print(f"  4. Displacement accuracy is POOR:")
        print(f"     → constitutive losses don't provide correct gradients")
        print(f"     → model has no reliable bending/shear supervision")
        print(f"")
        print(f"  CONCLUSION: Need Method C (field decoder) to fix this.")
        print(f"  {'═'*50}")

    def _classify_convergence(self, key, val):
        """Simple convergence status label."""
        if key in ['L_free', 'L_sup'] and val < 1e-10:
            return "✅ (hard constraint)"
        elif val < 1e-6:
            return "✅ converged"
        elif val < 1e-2:
            return "⚠️ partial"
        else:
            return "❌ not converged"


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 60)
    print("  PIGNN TRAINING — Naive Autograd Physics Loss")
    print("=" * 60)
    print(f"  Device: {TrainConfig.device}")
    print(f"  Purpose: Observe autograd problems → motivate Method C")

    # ── Train ──
    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Check {config.save_dir}/ for:")
    print(f"    loss_curves.png   — 6 individual loss plots")
    print(f"    total_loss.png    — total loss curve")
    print(f"    history.pt        — raw loss values")
    print(f"    best.pt           — best model checkpoint")
    print(f"{'='*60}")