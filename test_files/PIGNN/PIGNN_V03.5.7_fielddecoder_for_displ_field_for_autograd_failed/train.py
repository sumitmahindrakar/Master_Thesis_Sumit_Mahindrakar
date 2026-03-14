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
train.py — Training Loop for Method C PIGNN
=================================================================

PURPOSE:
  Train PIGNN with Method C field decoder using physics losses.
  7 loss terms, derivatives via ξ-autograd through field decoder.

LOSSES:
  L_eq   — nodal equilibrium (Concept 1)
  L_free — unconnected face forces = 0
  L_sup  — support displacements = 0
  L_N    — axial constitutive (algebraic)
  L_Mk   — moment-curvature: M = EI·w''     (ξ autograd)
  L_Mpp  — moment PDE: M'' = -q              (ξ autograd)
  L_end  — end forces: M,V at ends vs faces  (ξ autograd)

EXPECTED IMPROVEMENTS OVER NAIVE:
  - L_Mk should converge (ξ derivative = true spatial deriv)
  - L_Mpp should converge (2nd deriv max, not 4th)
  - L_end should converge (1st deriv of M for shear, not 3rd of w)
  - L_eq should improve (consistent constitutive → better face forces)

USAGE:
  python train.py
=================================================================
"""

import os
import time
import torch
import numpy as np
from pathlib import Path

from model import PIGNN
from physics_loss import MethodCPhysicsLoss


# ================================================================
# A. CONFIGURATION
# ================================================================

class TrainConfig:
    """All training hyperparameters in one place."""

    # ── Data ──
    data_path       = "DATA/graph_dataset.pt"
    save_dir        = "RESULTS"

    # ── Model ──
    hidden_dim      = 128
    n_layers        = 6
    node_in_dim     = 9
    edge_in_dim     = 10
    decoder_hidden  = [128, 64]   # field decoder hidden dims

    # ── Training ──
    epochs          = 500
    lr              = 1e-3
    weight_decay    = 1e-5
    scheduler_step  = 100
    scheduler_gamma = 0.5

    # ── Loss weights (7 terms) ──
    w_eq            = 1.0
    w_free          = 1.0
    w_sup           = 1.0
    w_N             = 1.0
    w_Mk            = 1.0
    w_Mpp           = 1.0
    w_end           = 1.0

    # ── Logging ──
    print_every     = 10
    save_every      = 100
    validate_every  = 50

    # ── Device ──
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================================================================
# B. TRAINER
# ================================================================

class Trainer:
    """
    Training manager for Method C PIGNN.

    Handles model creation, physics loss, optimization,
    logging of all 7 losses, validation, and checkpointing.
    """

    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        os.makedirs(config.save_dir, exist_ok=True)

        self._load_data()
        self._create_model()
        self._create_loss()
        self._create_optimizer()

        # History for all 7 losses
        self.history = {
            'epoch':  [],
            'L_eq':   [], 'L_free': [], 'L_sup': [],
            'L_N':    [], 'L_Mk':   [], 'L_Mpp': [], 'L_end': [],
            'total':  [],
            'val_disp_error': [],
        }

    def _load_data(self):
        """Load graph dataset (unnormalized for physics)."""
        print(f"\n-- Loading data --")
        self.data_list = torch.load(
            self.cfg.data_path, weights_only=False
        )
        print(f"  Loaded {len(self.data_list)} graphs")
        print(f"  Using UNNORMALIZED data (physics needs raw units)")
        self.train_data = self.data_list
        d = self.data_list[0]
        print(f"  Nodes: {d.num_nodes}, Elements: {d.n_elements}")

    def _create_model(self):
        """Create PIGNN with field decoder."""
        print(f"\n-- Creating model --")
        self.model = PIGNN(
            node_in_dim=self.cfg.node_in_dim,
            edge_in_dim=self.cfg.edge_in_dim,
            hidden_dim=self.cfg.hidden_dim,
            n_layers=self.cfg.n_layers,
            decoder_hidden=self.cfg.decoder_hidden,
        ).to(self.device)
        self.model.summary()

    def _create_loss(self):
        """Create Method C physics loss."""
        self.loss_fn = MethodCPhysicsLoss(
            w_eq=self.cfg.w_eq,
            w_free=self.cfg.w_free,
            w_sup=self.cfg.w_sup,
            w_N=self.cfg.w_N,
            w_Mk=self.cfg.w_Mk,
            w_Mpp=self.cfg.w_Mpp,
            w_end=self.cfg.w_end,
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
        """Train one epoch over all graphs."""
        self.model.train()

        epoch_losses = {
            'L_eq': 0, 'L_free': 0, 'L_sup': 0,
            'L_N': 0,  'L_Mk': 0,   'L_Mpp': 0, 'L_end': 0,
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

        avg = {k: v / n_graphs for k, v in epoch_losses.items()}
        return avg

    @torch.no_grad()
    def validate(self):
        """Compare predictions with Kratos ground truth."""
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

        return {'disp_error': total_disp_error / max(n_graphs, 1)}

    # ────────────────────────────────────────

    def train(self):
        """Full training loop with 7-loss logging."""
        print(f"\n{'='*70}")
        print(f"  TRAINING START (Method C — Field Decoder)")
        print(f"  {self.cfg.epochs} epochs, lr={self.cfg.lr}, "
              f"device={self.device}")
        print(f"{'='*70}")

        # Header for 7 losses
        header = (
            f"  {'Ep':>5} | {'Total':>10} | "
            f"{'L_eq':>9} {'L_free':>9} {'L_sup':>9} | "
            f"{'L_N':>10} | "
            f"{'L_Mk':>10} {'L_Mpp':>10} {'L_end':>10} | "
            f"{'DspErr':>9}"
        )
        print(f"\n{header}")
        print(f"  {'─'*115}")

        best_loss = float('inf')
        t_start = time.time()

        for epoch in range(1, self.cfg.epochs + 1):

            avg = self.train_one_epoch(epoch)

            # Record
            self.history['epoch'].append(epoch)
            for key in ['L_eq', 'L_free', 'L_sup', 'L_N',
                        'L_Mk', 'L_Mpp', 'L_end', 'total']:
                self.history[key].append(avg[key])

            # Validate
            disp_err = float('nan')
            if epoch % self.cfg.validate_every == 0 or epoch == 1:
                val = self.validate()
                disp_err = val['disp_error']
                self.history['val_disp_error'].append(disp_err)

            # Print
            if epoch % self.cfg.print_every == 0 or epoch == 1:
                de = f"{disp_err:.3e}" if not np.isnan(disp_err) else "   ---   "
                print(
                    f"  {epoch:5d} | {avg['total']:10.3e} | "
                    f"{avg['L_eq']:9.3e} {avg['L_free']:9.3e} "
                    f"{avg['L_sup']:9.3e} | "
                    f"{avg['L_N']:10.3e} | "
                    f"{avg['L_Mk']:10.3e} {avg['L_Mpp']:10.3e} "
                    f"{avg['L_end']:10.3e} | "
                    f"{de}"
                )

            # Scheduler
            self.scheduler.step()

            # Save best
            if avg['total'] < best_loss:
                best_loss = avg['total']
                self._save_checkpoint('best.pt', epoch, avg)

            # Periodic save
            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(f'epoch_{epoch}.pt', epoch, avg)

        elapsed = time.time() - t_start
        print(f"\n  {'─'*115}")
        print(f"  Training complete: {elapsed:.1f}s "
              f"({elapsed/self.cfg.epochs:.2f}s/epoch)")
        print(f"  Best total loss: {best_loss:.6e}")

        self._save_checkpoint('final.pt', self.cfg.epochs, avg)
        self._save_history()
        self._plot_losses()
        self._analyze_results()

    # ────────────────────────────────────────

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

    # ────────────────────────────────────────

    def _plot_losses(self):
        """Plot all 7 losses + total on separate subplots."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("  matplotlib not found, skipping plots")
            return

        epochs = self.history['epoch']

        # ── 7 individual losses ──
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        fig.suptitle('Method C Physics Loss Components',
                     fontsize=14, fontweight='bold')

        loss_info = [
            # (key, title, color)
            ('L_eq',   'L_eq: Equilibrium',              'tab:blue'),
            ('L_free', 'L_free: Free Face',              'tab:cyan'),
            ('L_sup',  'L_sup: Support BC',              'tab:green'),
            ('L_N',    'L_N: Axial (algebraic)',         'tab:orange'),
            ('L_Mk',   'L_Mk: Moment-Curvature (xi)',   'tab:red'),
            ('L_Mpp',  'L_Mpp: Moment PDE M\'\'=-q',    'tab:purple'),
            ('L_end',  'L_end: End Forces',              'tab:brown'),
        ]

        for idx, (key, title, color) in enumerate(loss_info):
            ax = axes[idx // 2, idx % 2]
            values = self.history[key]
            # Filter out zeros for log scale
            vals_plot = [max(v, 1e-30) for v in values]
            ax.semilogy(epochs, vals_plot, color=color, linewidth=1.5)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)

        # Last subplot: total
        ax = axes[3, 1]
        ax.semilogy(epochs, self.history['total'], 'k-', linewidth=2)
        ax.set_title('Total Loss', fontsize=10)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.cfg.save_dir, 'loss_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Loss curves saved: {path}")

        # ── Comparison: Concept 1 vs Concept 3 ──
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle('Concept 1 (Equilibrium) vs Concept 3 (Constitutive)',
                      fontsize=12, fontweight='bold')

        # Concept 1
        ax1.semilogy(epochs, [max(v, 1e-30) for v in self.history['L_eq']],
                     'b-', label='L_eq', linewidth=1.5)
        ax1.set_title('Concept 1: Equilibrium')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Concept 3
        ax2.semilogy(epochs, [max(v, 1e-30) for v in self.history['L_N']],
                     'orange', label='L_N (axial)', linewidth=1.5)
        ax2.semilogy(epochs, [max(v, 1e-30) for v in self.history['L_Mk']],
                     'r-', label='L_Mk (moment-curv)', linewidth=1.5)
        ax2.semilogy(epochs, [max(v, 1e-30) for v in self.history['L_Mpp']],
                     'm-', label='L_Mpp (M\'\'=-q)', linewidth=1.5)
        ax2.semilogy(epochs, [max(v, 1e-30) for v in self.history['L_end']],
                     'brown', label='L_end (end forces)', linewidth=1.5)
        ax2.set_title('Concept 3: Constitutive (Method C)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path2 = os.path.join(self.cfg.save_dir, 'concept_comparison.png')
        plt.savefig(path2, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Comparison saved: {path2}")

    # ────────────────────────────────────────

    def _analyze_results(self):
        """Analyze convergence of all 7 losses."""
        print(f"\n{'='*70}")
        print(f"  RESULT ANALYSIS (Method C)")
        print(f"{'='*70}")

        # Final values
        final = {k: self.history[k][-1] for k in
                 ['L_eq', 'L_free', 'L_sup', 'L_N',
                  'L_Mk', 'L_Mpp', 'L_end', 'total']}

        # Best values
        best_idx = np.argmin(self.history['total'])
        best = {k: self.history[k][best_idx] for k in
                ['L_eq', 'L_free', 'L_sup', 'L_N',
                 'L_Mk', 'L_Mpp', 'L_end', 'total']}
        best_epoch = self.history['epoch'][best_idx]

        print(f"\n  Final losses (epoch {self.cfg.epochs}):")
        print(f"  {'─'*55}")
        categories = {
            'L_eq':   'Concept 1 (equilibrium)',
            'L_free': 'Concept 1 (free face)',
            'L_sup':  'Concept 1 (support)',
            'L_N':    'Concept 3 (axial, algebraic)',
            'L_Mk':   'Concept 3 (moment-curvature, xi)',
            'L_Mpp':  'Concept 3 (moment PDE, xi)',
            'L_end':  'Concept 3 (end forces, xi)',
            'total':  'TOTAL',
        }
        for key, val in final.items():
            status = self._classify(key, val)
            print(f"    {key:<8} {val:>12.4e}  {status}")

        print(f"\n  Best losses (epoch {best_epoch}):")
        print(f"  {'─'*55}")
        for key, val in best.items():
            status = self._classify(key, val)
            print(f"    {key:<8} {val:>12.4e}  {status}")

        # Convergence trends
        print(f"\n  Convergence analysis:")
        print(f"  {'─'*55}")
        for key in ['L_eq', 'L_N', 'L_Mk', 'L_Mpp', 'L_end']:
            vals = self.history[key]
            if len(vals) >= 20:
                first = np.mean(vals[:10])
                last  = np.mean(vals[-10:])
                ratio = last / max(first, 1e-30)
                if ratio < 0.01:
                    trend = "CONVERGED (>100x reduction)"
                elif ratio < 0.1:
                    trend = "PARTIAL (10-100x reduction)"
                elif ratio < 1.0:
                    trend = "SLOW (<10x reduction)"
                else:
                    trend = "NOT CONVERGING"
                print(f"    {key:<8} first={first:.3e} "
                      f"last={last:.3e}  {trend}")

        # Displacement accuracy
        if self.history['val_disp_error']:
            de = self.history['val_disp_error'][-1]
            print(f"\n  Displacement error vs Kratos: {de:.4e}")
            if de > 0.5:
                print(f"  Poor accuracy")
            elif de > 0.1:
                print(f"  Moderate accuracy")
            else:
                print(f"  Good accuracy")

        # Method C specific analysis
        print(f"\n  {'='*55}")
        print(f"  METHOD C ANALYSIS:")
        print(f"  {'─'*55}")
        print(f"  Field decoder derivatives are w.r.t. xi (element-local)")
        print(f"  Max derivative order: 2 (mixed formulation)")
        print(f"")
        print(f"  Compare with naive autograd:")
        print(f"    Naive L_N  ~1e6-1e9  vs  Method C L_N  = {final['L_N']:.3e}")
        print(f"    Naive L_M  ~1e2-1e6  vs  Method C L_Mk = {final['L_Mk']:.3e}")
        print(f"    Naive L_V  ~50-9000  vs  Method C L_end= {final['L_end']:.3e}")
        print(f"  {'='*55}")

    def _classify(self, key, val):
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

    print("=" * 70)
    print("  PIGNN TRAINING — Method C (Field Decoder)")
    print("=" * 70)
    print(f"  Device: {TrainConfig.device}")
    print(f"  7 losses: L_eq, L_free, L_sup, L_N, L_Mk, L_Mpp, L_end")
    print(f"  Derivatives: autograd on xi (element-local)")
    print(f"  Max derivative order: 2 (mixed formulation)")

    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"  Check {config.save_dir}/ for:")
    print(f"    loss_curves.png       — 7 individual loss plots")
    print(f"    concept_comparison.png — Concept 1 vs 3")
    print(f"    history.pt            — raw loss values")
    print(f"    best.pt               — best model checkpoint")
    print(f"{'='*70}")