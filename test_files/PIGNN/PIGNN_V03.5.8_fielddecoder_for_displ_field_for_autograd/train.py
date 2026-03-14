"""
=================================================================
train.py — Method C Training with Adaptive Loss Weighting
=================================================================

PURPOSE:
  Train PIGNN with Method C field decoder using physics losses.
  Adaptive loss weighting ensures all 7 losses contribute equally.

ADAPTIVE WEIGHTING:
  Each loss weight λ_i is updated every epoch:
    λ_i = 1 / (EMA(L_i) + ε)
  
  This means:
    Large loss → small weight (don't let it dominate)
    Small loss → large weight (give it more gradient signal)
    All weighted losses ≈ O(1) → balanced optimisation

  One-step lag: weights computed from current losses,
  applied to NEXT iteration. Acceptable because losses
  change slowly between epochs.

LOSSES:
  L_eq   — nodal equilibrium
  L_free — unconnected face forces = 0
  L_sup  — support displacements = 0
  L_N    — axial constitutive (algebraic)
  L_Mk   — moment-curvature: M = EI·w''
  L_Mpp  — moment PDE: M'' = -q
  L_end  — end forces: M, V at ends vs face forces

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
# A. ADAPTIVE LOSS WEIGHTER
# ================================================================

class AdaptiveLossWeighter:
    """
    Automatically balances loss weights using exponential moving average.

    Each loss weight = 1 / (EMA of that loss + epsilon).
    This ensures all losses contribute roughly equally to the total,
    regardless of their absolute magnitude.

    Example:
      L_Mk ~ 10000, L_Mpp ~ 1.0
      w_Mk = 1/10000 = 0.0001,  w_Mpp = 1/1.0 = 1.0
      weighted: w_Mk·L_Mk = 1.0,  w_Mpp·L_Mpp = 1.0  → balanced!

    Args:
        loss_keys:  list of loss names to balance
        alpha:      EMA smoothing factor (0.05 = moderate)
        epsilon:    prevent division by zero
    """

    def __init__(self, loss_keys, alpha=0.05, epsilon=1e-8):
        self.alpha = alpha
        self.epsilon = epsilon
        self.ema = {k: None for k in loss_keys}
        self.weights = {k: 1.0 for k in loss_keys}

    def update(self, loss_dict):
        """
        Update EMA and recompute weights.

        Args:
            loss_dict: dict of current loss values (detached floats)

        Returns:
            weights: dict of updated weights
        """
        for k in self.ema:
            if k not in loss_dict:
                continue
            val = loss_dict[k]
            if val <= 0:
                continue

            if self.ema[k] is None:
                # First call: initialise EMA with current value
                self.ema[k] = val
            else:
                # Exponential moving average
                self.ema[k] = (1 - self.alpha) * self.ema[k] + self.alpha * val

            # Weight = inverse of EMA
            self.weights[k] = 1.0 / (self.ema[k] + self.epsilon)

        return self.weights

    def print_status(self):
        """Print current EMA and weights."""
        print(f"\n  Adaptive weights:")
        print(f"  {'Loss':<8} {'EMA':>12} {'Weight':>12}")
        print(f"  {'─'*36}")
        for k in self.ema:
            ema_val = self.ema[k] if self.ema[k] is not None else 0.0
            w_val = self.weights[k]
            print(f"  {k:<8} {ema_val:>12.4e} {w_val:>12.4e}")


# ================================================================
# B. CONFIGURATION
# ================================================================

class TrainConfig:
    """
    All training hyperparameters in one place.

    Changes from previous version:
      - 1000 epochs (more time for L_Mpp to learn)
      - Adaptive weighting enabled
      - Scheduler step = 200 (slower LR decay)
    """

    # ── Data ──
    data_path       = "DATA/graph_dataset.pt"
    save_dir        = "RESULTS"

    # ── Model ──
    hidden_dim      = 128
    n_layers        = 6
    node_in_dim     = 9
    edge_in_dim     = 10
    decoder_hidden  = [128, 64]

    # ── Training ──
    epochs          = 1000
    lr              = 1e-3
    weight_decay    = 1e-5
    scheduler_step  = 200
    scheduler_gamma = 0.5

    # ── Loss weights (initial — overridden by adaptive) ──
    w_eq            = 1.0
    w_free          = 1.0
    w_sup           = 1.0
    w_N             = 1.0
    w_Mk            = 1.0
    w_Mpp           = 1.0
    w_end           = 1.0

    # ── Adaptive weighting ──
    use_adaptive    = True
    adaptive_alpha  = 0.05    # EMA smoothing (0.01=slow, 0.1=fast)

    # ── Logging ──
    print_every     = 20
    save_every      = 200
    validate_every  = 50

    # ── Device ──
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================================================================
# C. TRAINER
# ================================================================

class Trainer:
    """
    Training manager for Method C PIGNN with adaptive loss weighting.

    Handles:
      - Model and loss creation
      - Adaptive weight updates per epoch
      - Logging of all 7 losses + weights
      - Validation against Kratos ground truth
      - Checkpointing and visualisation
    """

    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        os.makedirs(config.save_dir, exist_ok=True)

        # ── Setup ──
        self._load_data()
        self._create_model()
        self._create_loss()
        self._create_optimizer()

        # ── Adaptive weighter ──
        self.loss_keys = ['L_eq', 'L_free', 'L_sup', 'L_N',
                          'L_Mk', 'L_Mpp', 'L_end']

        if config.use_adaptive:
            self.weighter = AdaptiveLossWeighter(
                self.loss_keys, alpha=config.adaptive_alpha
            )
        else:
            self.weighter = None

        # ── History ──
        self.history = {
            'epoch': [],
            # Losses
            'L_eq': [], 'L_free': [], 'L_sup': [],
            'L_N': [], 'L_Mk': [], 'L_Mpp': [], 'L_end': [],
            'total': [],
            # Validation
            'val_disp_error': [],
            # Adaptive weights
            'w_eq': [], 'w_N': [], 'w_Mk': [], 'w_Mpp': [], 'w_end': [],
        }

    # ────────────────────────────────────────
    # SETUP
    # ────────────────────────────────────────

    def _load_data(self):
        """Load unnormalised graph dataset."""
        print(f"\n-- Loading data --")
        self.data_list = torch.load(
            self.cfg.data_path, weights_only=False
        )
        self.train_data = self.data_list
        d = self.data_list[0]
        print(f"  {len(self.data_list)} graphs, "
              f"{d.num_nodes} nodes, {d.n_elements} elements")

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
        """Create Adam optimiser with step LR scheduler."""
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
    # TRAINING
    # ────────────────────────────────────────

    def train_one_epoch(self, epoch):
        """
        Train one epoch over all graphs.

        If adaptive weighting is enabled:
          1. Compute losses with current weights
          2. Backward + update parameters
          3. Update adaptive weights for NEXT epoch

        Returns:
            avg_losses: dict of average losses over all graphs
        """
        self.model.train()

        epoch_losses = {k: 0.0 for k in self.loss_keys + ['total']}
        n_graphs = len(self.train_data)

        for data in self.train_data:
            data = data.to(self.device)

            # ── Forward + loss ──
            self.optimizer.zero_grad()
            total_loss, loss_dict, pred = self.loss_fn(
                self.model, data
            )

            # ── Backward ──
            total_loss.backward()

            # ── Gradient clipping ──
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=10.0
            )

            # ── Update parameters ──
            self.optimizer.step()

            # ── Accumulate losses ──
            for key in epoch_losses:
                epoch_losses[key] += loss_dict.get(key, 0.0)

        # ── Average ──
        avg = {k: v / n_graphs for k, v in epoch_losses.items()}

        # ── Update adaptive weights for NEXT epoch ──
        if self.weighter is not None:
            weights = self.weighter.update(avg)

            # Apply to loss function
            self.loss_fn.w_eq   = weights.get('L_eq', 1.0)
            self.loss_fn.w_free = weights.get('L_free', 1.0)
            self.loss_fn.w_sup  = weights.get('L_sup', 1.0)
            self.loss_fn.w_N    = weights.get('L_N', 1.0)
            self.loss_fn.w_Mk   = weights.get('L_Mk', 1.0)
            self.loss_fn.w_Mpp  = weights.get('L_Mpp', 1.0)
            self.loss_fn.w_end  = weights.get('L_end', 1.0)

        return avg

    @torch.no_grad()
    def validate(self):
        """
        Compare predictions with Kratos ground truth.

        Returns relative L2 displacement error.
        """
        self.model.eval()
        total_err = 0.0
        n = 0

        for data in self.train_data:
            data = data.to(self.device)

            # Need model.train() for forward pass (h_i storage)
            self.model.train()
            pred = self.model(data)
            self.model.eval()

            pred_disp = pred[:, 0:3]
            true_disp = data.y_node.to(self.device)

            err = (pred_disp - true_disp).pow(2).sum().sqrt()
            ref = true_disp.pow(2).sum().sqrt().clamp(min=1e-10)
            total_err += (err / ref).item()
            n += 1

        return {'disp_error': total_err / max(n, 1)}

    # ────────────────────────────────────────
    # MAIN TRAINING LOOP
    # ────────────────────────────────────────

    def train(self):
        """Full training loop with adaptive weighting."""
        print(f"\n{'='*75}")
        print(f"  TRAINING START (Method C + Adaptive Weights)")
        print(f"  {self.cfg.epochs} epochs, lr={self.cfg.lr}, "
              f"device={self.device}")
        if self.weighter:
            print(f"  Adaptive weighting: alpha={self.cfg.adaptive_alpha}")
        print(f"{'='*75}")

        # ── Header ──
        header = (
            f"  {'Ep':>5} | {'Total':>10} | "
            f"{'L_eq':>8} {'L_N':>8} | "
            f"{'L_Mk':>10} {'L_Mpp':>8} {'L_end':>9} | "
            f"{'DspErr':>9} | "
            f"{'w_Mk':>8} {'w_Mpp':>8}"
        )
        print(f"\n{header}")
        print(f"  {'─'*110}")

        best_loss = float('inf')
        t_start = time.time()

        for epoch in range(1, self.cfg.epochs + 1):

            # ── Train one epoch ──
            avg = self.train_one_epoch(epoch)

            # ── Record losses ──
            self.history['epoch'].append(epoch)
            for k in self.loss_keys + ['total']:
                self.history[k].append(avg.get(k, 0.0))

            # ── Record adaptive weights ──
            if self.weighter:
                for wk in ['w_eq', 'w_N', 'w_Mk', 'w_Mpp', 'w_end']:
                    loss_key = 'L_' + wk[2:]
                    self.history[wk].append(
                        self.weighter.weights.get(loss_key, 1.0)
                    )
            else:
                for wk in ['w_eq', 'w_N', 'w_Mk', 'w_Mpp', 'w_end']:
                    self.history[wk].append(1.0)

            # ── Validate ──
            disp_err = float('nan')
            if epoch % self.cfg.validate_every == 0 or epoch == 1:
                val = self.validate()
                disp_err = val['disp_error']
                self.history['val_disp_error'].append(disp_err)

            # ── Print ──
            if epoch % self.cfg.print_every == 0 or epoch == 1:
                de = f"{disp_err:.3e}" if not np.isnan(disp_err) else "   ---   "
                w_mk = (self.weighter.weights.get('L_Mk', 1.0)
                        if self.weighter else 1.0)
                w_mpp = (self.weighter.weights.get('L_Mpp', 1.0)
                         if self.weighter else 1.0)
                print(
                    f"  {epoch:5d} | {avg['total']:10.3e} | "
                    f"{avg['L_eq']:8.3e} {avg['L_N']:8.3e} | "
                    f"{avg['L_Mk']:10.3e} {avg['L_Mpp']:8.3e} "
                    f"{avg['L_end']:9.3e} | "
                    f"{de} | "
                    f"{w_mk:8.3e} {w_mpp:8.3e}"
                )

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
        print(f"\n  {'─'*110}")
        print(f"  Training complete: {elapsed:.1f}s "
              f"({elapsed/self.cfg.epochs:.2f}s/epoch)")
        print(f"  Best total loss: {best_loss:.6e}")

        # ── Save and visualise ──
        self._save_checkpoint('final.pt', self.cfg.epochs, avg)
        self._save_history()

        if self.weighter:
            self.weighter.print_status()

        self._plot_losses()
        self._analyze_results()

    # ────────────────────────────────────────
    # CHECKPOINTING
    # ────────────────────────────────────────

    def _save_checkpoint(self, filename, epoch, losses):
        """Save model checkpoint."""
        path = os.path.join(self.cfg.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'losses': losses,
            'adaptive_weights': (self.weighter.weights
                                 if self.weighter else None),
        }, path)

    def _save_history(self):
        """Save full training history."""
        path = os.path.join(self.cfg.save_dir, 'history.pt')
        torch.save(self.history, path)
        print(f"  History saved: {path}")

    # ────────────────────────────────────────
    # VISUALISATION
    # ────────────────────────────────────────

    def _plot_losses(self):
        """Plot all 7 losses + total + weight evolution."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("  matplotlib not found, skipping plots")
            return

        epochs = self.history['epoch']

        # ════════════════════════════════════
        # Plot 1: Individual losses (4×2 grid)
        # ════════════════════════════════════
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        fig.suptitle('Method C + Adaptive Weights — Loss Components',
                     fontsize=14, fontweight='bold')

        loss_plots = [
            ('L_eq',   'L_eq: Equilibrium',        'tab:blue'),
            ('L_free', 'L_free: Free Face',         'tab:cyan'),
            ('L_sup',  'L_sup: Support BC',          'tab:green'),
            ('L_N',    'L_N: Axial (algebraic)',     'tab:orange'),
            ('L_Mk',   'L_Mk: Moment-Curvature',    'tab:red'),
            ('L_Mpp',  'L_Mpp: M\'\'=-q PDE',       'tab:purple'),
            ('L_end',  'L_end: End Forces',          'tab:brown'),
            ('total',  'Total Loss',                 'black'),
        ]

        for idx, (key, title, color) in enumerate(loss_plots):
            ax = axes[idx // 2, idx % 2]
            vals = [max(v, 1e-30) for v in self.history[key]]
            ax.semilogy(epochs, vals, color=color, linewidth=1.5)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.cfg.save_dir, 'loss_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Loss curves: {path}")

        # ════════════════════════════════════
        # Plot 2: Concept 1 vs Concept 3
        # ════════════════════════════════════
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle('Equilibrium vs Constitutive Losses',
                      fontsize=12, fontweight='bold')

        # Concept 1
        ax1.semilogy(epochs,
                     [max(v, 1e-30) for v in self.history['L_eq']],
                     'b-', label='L_eq', lw=1.5)
        ax1.set_title('Concept 1: Equilibrium')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Concept 3
        for key, color, label in [
            ('L_N',   'tab:orange', 'L_N (axial)'),
            ('L_Mk',  'tab:red',    'L_Mk (M=EI*w\'\')'),
            ('L_Mpp', 'tab:purple', 'L_Mpp (M\'\'=-q)'),
            ('L_end', 'tab:brown',  'L_end (end forces)'),
        ]:
            ax2.semilogy(epochs,
                         [max(v, 1e-30) for v in self.history[key]],
                         color=color, label=label, lw=1.5)
        ax2.set_title('Concept 3: Constitutive (Method C)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path2 = os.path.join(self.cfg.save_dir, 'concept_comparison.png')
        plt.savefig(path2, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Comparison: {path2}")

        # ════════════════════════════════════
        # Plot 3: Adaptive weight evolution
        # ════════════════════════════════════
        if self.weighter:
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            fig3.suptitle('Adaptive Weight Evolution',
                          fontsize=12, fontweight='bold')

            weight_plots = [
                ('w_eq',  'tab:blue',   'w_eq'),
                ('w_N',   'tab:orange', 'w_N'),
                ('w_Mk',  'tab:red',    'w_Mk'),
                ('w_Mpp', 'tab:purple', 'w_Mpp'),
                ('w_end', 'tab:brown',  'w_end'),
            ]

            for key, color, label in weight_plots:
                if key in self.history and self.history[key]:
                    vals = self.history[key]
                    ax3.semilogy(epochs[:len(vals)], vals,
                                color=color, label=label, lw=1.5)

            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Weight (log scale)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            path3 = os.path.join(self.cfg.save_dir, 'weight_evolution.png')
            plt.savefig(path3, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"  Weights: {path3}")

        # ════════════════════════════════════
        # Plot 4: Displacement error
        # ════════════════════════════════════
        if self.history['val_disp_error']:
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            val_epochs = [
                e for i, e in enumerate(epochs)
                if i < len(self.history['val_disp_error'])
                # This approximation works for validate_every
            ]
            # More robust: just use indices
            n_val = len(self.history['val_disp_error'])
            val_x = np.linspace(1, epochs[-1], n_val)

            ax4.semilogy(val_x, self.history['val_disp_error'],
                         'ko-', lw=1.5, ms=4)
            ax4.set_title('Displacement Error vs Kratos')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Relative L2 Error')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            path4 = os.path.join(self.cfg.save_dir, 'disp_error.png')
            plt.savefig(path4, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"  Disp error: {path4}")

    # ────────────────────────────────────────
    # ANALYSIS
    # ────────────────────────────────────────

    def _analyze_results(self):
        """Print convergence analysis of all losses."""
        print(f"\n{'='*65}")
        print(f"  RESULT ANALYSIS (Method C + Adaptive Weights)")
        print(f"{'='*65}")

        # ── Final losses ──
        final = {k: self.history[k][-1]
                 for k in self.loss_keys + ['total']}

        print(f"\n  Final losses (epoch {self.cfg.epochs}):")
        print(f"  {'─'*50}")
        for k, v in final.items():
            status = self._classify(k, v)
            print(f"    {k:<8} {v:>12.4e}  {status}")

        # ── Best losses ──
        best_idx = np.argmin(self.history['total'])
        best = {k: self.history[k][best_idx]
                for k in self.loss_keys + ['total']}
        best_epoch = self.history['epoch'][best_idx]

        print(f"\n  Best losses (epoch {best_epoch}):")
        print(f"  {'─'*50}")
        for k, v in best.items():
            status = self._classify(k, v)
            print(f"    {k:<8} {v:>12.4e}  {status}")

        # ── Convergence trends ──
        print(f"\n  Convergence analysis:")
        print(f"  {'─'*60}")
        for k in ['L_eq', 'L_N', 'L_Mk', 'L_Mpp', 'L_end']:
            vals = self.history[k]
            if len(vals) >= 20:
                first = np.mean(vals[:10])
                last  = np.mean(vals[-10:])
                ratio = last / max(first, 1e-30)
                if ratio < 0.01:
                    trend = "CONVERGED (>100x reduction)"
                elif ratio < 0.1:
                    trend = "PARTIAL (10-100x)"
                elif ratio < 1.0:
                    trend = "SLOW (<10x)"
                else:
                    trend = "NOT CONVERGING"
                print(f"    {k:<8} {first:.3e} -> {last:.3e}  {trend}")

        # ── Displacement error ──
        if self.history['val_disp_error']:
            first_de = self.history['val_disp_error'][0]
            last_de = self.history['val_disp_error'][-1]
            best_de = min(self.history['val_disp_error'])
            print(f"\n  Displacement error vs Kratos:")
            print(f"    First:  {first_de:.4e}")
            print(f"    Last:   {last_de:.4e}")
            print(f"    Best:   {best_de:.4e}")
            if last_de > 0.5:
                print(f"    Status: Poor accuracy")
            elif last_de > 0.1:
                print(f"    Status: Moderate accuracy")
            elif last_de > 0.01:
                print(f"    Status: Good accuracy")
            else:
                print(f"    Status: Excellent accuracy")

        # ── Final adaptive weights ──
        if self.weighter:
            print(f"\n  Final adaptive weights:")
            print(f"  {'─'*40}")
            for k in self.loss_keys:
                w = self.weighter.weights.get(k, 1.0)
                ema = self.weighter.ema.get(k, 0.0)
                if ema is not None:
                    print(f"    {k:<8} weight={w:>10.4e}  "
                          f"ema={ema:>10.4e}")

        print(f"\n{'='*65}")

    def _classify(self, key, val):
        """Simple convergence status."""
        if key in ['L_free', 'L_sup'] and val < 1e-10:
            return "(hard constraint)"
        elif key == 'L_end' and val < 1e-6:
            return "(hard BC)"
        elif val < 1e-4:
            return "converged"
        elif val < 1e-1:
            return "partial"
        elif val < 10.0:
            return "improving"
        else:
            return "not converged"


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 75)
    print("  PIGNN TRAINING — Method C + Adaptive Loss Weighting")
    print("=" * 75)
    print(f"  Device: {TrainConfig.device}")
    print(f"  Losses: L_eq, L_free, L_sup, L_N, L_Mk, L_Mpp, L_end")
    print(f"  Adaptive weighting: {TrainConfig.use_adaptive}")
    print(f"  Epochs: {TrainConfig.epochs}")

    # ── Train ──
    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()

    print(f"\n{'='*75}")
    print(f"  TRAINING COMPLETE")
    print(f"  Check {config.save_dir}/ for:")
    print(f"    loss_curves.png        — 7 individual losses")
    print(f"    concept_comparison.png — equilibrium vs constitutive")
    print(f"    weight_evolution.png   — adaptive weight history")
    print(f"    disp_error.png         — displacement accuracy")
    print(f"    history.pt             — raw data")
    print(f"    best.pt                — best checkpoint")
    print(f"{'='*75}")