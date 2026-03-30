"""
=================================================================
train.py — Energy-Based Training v2
=================================================================

Fixes based on training analysis:
  1. Three-phase LR schedule (warm-up → train → fine-tune)
  2. More epochs (or early stopping when converged)
  3. Better convergence monitoring
  4. Gradient norm tracking for diagnostics
=================================================================
"""

import os
import time
import torch
import numpy as np
from pathlib import Path

from model import PIGNN
from energy_loss import FrameEnergyLoss


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
    epochs          = 5000
    
    # Three-phase LR:
    #   Phase 1 (warm-up):    epochs 1-100,     lr ramps 0→lr
    #   Phase 2 (main):       epochs 101-2500,  lr
    #   Phase 3 (fine-tune):  epochs 2501-5000, lr/10
    lr              = 5e-4     # lower than before
    warmup_epochs   = 100
    decay_epoch     = 2500     # when to switch to lr/10
    lr_decay_factor = 0.1
    weight_decay    = 0.0

    # ── Convergence monitoring ──
    patience        = 500      # early stop if no improvement
    min_delta       = 1e-4     # minimum improvement threshold

    print_every     = 100
    save_every      = 1000
    validate_every  = 100

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

        # Early stopping
        self.best_Pi = float('inf')
        self.best_epoch = 0
        self.no_improve_count = 0

    def _load_data(self):
        print(f"\n── Loading data ──")
        self.data_list = torch.load(
            self.cfg.data_path, weights_only=False
        )
        print(f"  Loaded {len(self.data_list)} graphs")

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

        # Expected target energy
        from energy_loss import FrameEnergyLoss
        loss_fn = FrameEnergyLoss()
        for i in range(min(3, len(self.data_list))):
            dd = self.data_list[i]
            # Need raw data for true displacement
            pass

        self.train_data = self.data_list

        self.raw_data = torch.load(
            "DATA/graph_dataset.pt", weights_only=False
        )
        from normalizer import PhysicsScaler
        if not hasattr(self.raw_data[0], 'u_c'):
            self.raw_data = (
                PhysicsScaler.compute_and_store_list(
                    self.raw_data
                )
            )

        # Compute target energy for reference
        print(f"\n  Target energies (from Kratos solution):")
        self._target_energies = []
        for i, rd in enumerate(self.raw_data):
            u_true = rd.y_node
            U_t = loss_fn._strain_energy(u_true, rd)
            W_t = loss_fn._external_work(u_true, rd)
            Pi_t = U_t - W_t
            E_c = (rd.F_c * rd.u_c).clamp(min=1e-30)
            Pi_t_norm = Pi_t / E_c
            self._target_energies.append(
                Pi_t_norm.item()
            )
            if i < 3:
                print(
                    f"    Case {i}: Π_target = "
                    f"{Pi_t_norm.item():.4e}, "
                    f"U/|W| = "
                    f"{(U_t/W_t.abs().clamp(min=1e-30)).item():.4f}"
                )

        avg_target = np.mean(self._target_energies)
        print(f"    Average Π_target = {avg_target:.4e}")
        print(f"    (network should converge to this)")

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
            d = self.data_list[0].to(self.device)
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
        """Three-phase learning rate schedule."""
        if epoch <= self.cfg.warmup_epochs:
            # Linear warmup
            return self.cfg.lr * epoch / self.cfg.warmup_epochs
        elif epoch <= self.cfg.decay_epoch:
            # Main training
            return self.cfg.lr
        else:
            # Fine-tuning
            return self.cfg.lr * self.cfg.lr_decay_factor

    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    # ════════════════════════════════════════
    # TRAINING
    # ════════════════════════════════════════

    def train_one_epoch(self, epoch):
        self.model.train()

        lr = self._get_lr(epoch)
        self._set_lr(lr)

        epoch_Pi = 0.0
        epoch_U = 0.0
        epoch_W = 0.0
        epoch_grad_norm = 0.0
        last_dict = None
        n_graphs = len(self.train_data)

        for data in self.train_data:
            data = data.to(self.device)

            self.optimizer.zero_grad()
            Pi_norm, loss_dict, pred_raw, u_phys = \
                self.loss_fn(self.model, data)

            if torch.isnan(Pi_norm):
                print(f"  ⚠ NaN at epoch {epoch}")
                continue

            Pi_norm.backward()

            # Track gradient norm
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item()**2
            total_norm = total_norm**0.5
            epoch_grad_norm += total_norm

            self.optimizer.step()

            epoch_Pi += loss_dict['Pi_norm']
            epoch_U += loss_dict['U_internal']
            epoch_W += loss_dict['W_external']
            last_dict = loss_dict

        avg_W = epoch_W / n_graphs
        avg_U = epoch_U / n_graphs

        avg = {
            'Pi':        epoch_Pi / n_graphs,
            'U':         avg_U,
            'W':         avg_W,
            'U_over_W':  abs(avg_U) / max(abs(avg_W), 1e-30),
            'grad_norm': epoch_grad_norm / n_graphs,
            'lr':        lr,
            'last':      last_dict,
        }
        return avg

    # ════════════════════════════════════════
    # VALIDATION
    # ════════════════════════════════════════

    def validate(self):
        self.model.eval()

        total_err = 0.0
        total_Pi_err = 0.0
        n_graphs = 0

        with torch.no_grad():
            for i, data in enumerate(self.train_data):
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

                if i < len(self.raw_data):
                    true_disp = (
                        self.raw_data[i].y_node
                        .to(self.device)
                    )
                    err = ((pred_phys - true_disp)
                           .pow(2).sum().sqrt())
                    ref = (true_disp.pow(2).sum()
                           .sqrt().clamp(min=1e-10))
                    total_err += (err / ref).item()
                    n_graphs += 1

        return {
            'disp_err': total_err / max(n_graphs, 1),
        }

    # ════════════════════════════════════════
    # MAIN LOOP
    # ════════════════════════════════════════

    def train(self):
        print(f"\n{'═'*90}")
        print(f"  TRAINING — Energy-Based PIGNN v2")
        print(f"  {self.cfg.epochs} epochs, "
              f"lr={self.cfg.lr}, "
              f"device={self.device}")
        print(f"  Warmup:    1-{self.cfg.warmup_epochs}")
        print(f"  Main:      "
              f"{self.cfg.warmup_epochs+1}-"
              f"{self.cfg.decay_epoch}")
        print(f"  Fine-tune: "
              f"{self.cfg.decay_epoch+1}-"
              f"{self.cfg.epochs}")
        print(f"  Patience:  {self.cfg.patience}")
        print(f"{'═'*90}")

        header = (
            f"  {'Ep':>5} | {'Π_norm':>12} | "
            f"{'U':>12} {'W':>12} {'U/W':>7} | "
            f"{'|∇|':>10} | "
            f"{'DispErr':>10} | "
            f"{'LR':>9}"
        )
        print(f"\n{header}")
        print(f"  {'-'*100}")

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

            # ── Early stopping check ──
            if avg['Pi'] < self.best_Pi - self.cfg.min_delta:
                self.best_Pi = avg['Pi']
                self.best_epoch = epoch
                self.no_improve_count = 0
                self._save_checkpoint(
                    'best.pt', epoch, avg
                )
            else:
                self.no_improve_count += 1

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

            # ── Print ──
            if (epoch % self.cfg.print_every == 0
                    or epoch == 1
                    or epoch == self.cfg.epochs):
                de = (f"{disp_err:.4e}"
                      if not np.isnan(disp_err)
                      else "    ---   ")

                print(
                    f"  {epoch:5d} | "
                    f"{avg['Pi']:12.4e} | "
                    f"{avg['U']:12.4e} "
                    f"{avg['W']:12.4e} "
                    f"{avg['U_over_W']:7.4f} | "
                    f"{avg['grad_norm']:10.2e} | "
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

            # ── Early stopping ──
            if (self.no_improve_count >=
                    self.cfg.patience
                    and epoch > self.cfg.decay_epoch):
                print(f"\n  Early stopping at epoch "
                      f"{epoch} (no improvement for "
                      f"{self.cfg.patience} epochs)")
                break

        elapsed = time.time() - t_start
        print(f"\n  {'-'*100}")
        print(f"  Done: {elapsed:.1f}s "
              f"({elapsed/epoch:.2f}s/epoch)")
        print(f"  Best Π_norm:   {self.best_Pi:.6e} "
              f"(epoch {self.best_epoch})")
        print(f"  Best disp_err: {best_disp_err:.6e}")

        if self._target_energies:
            avg_target = np.mean(self._target_energies)
            ratio = self.best_Pi / avg_target
            print(f"  Target Π_norm: {avg_target:.6e}")
            print(f"  Achieved/Target: {ratio:.4f}")

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
            'Energy-Based PIGNN Training v2',
            fontsize=14, fontweight='bold'
        )

        # 1. Π_norm
        ax = axes[0, 0]
        ax.plot(epochs, self.history['Pi'],
                'b-', linewidth=0.8, alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='--',
                   alpha=0.5)
        if self._target_energies:
            avg_t = np.mean(self._target_energies)
            ax.axhline(y=avg_t, color='r',
                      linestyle='--', alpha=0.7,
                      label=f'Target: {avg_t:.2f}')
            ax.legend()
        ax.set_title('Π_norm (→ target)')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)

        # 2. U and W
        ax = axes[0, 1]
        ax.plot(epochs, self.history['U'],
                'r-', alpha=0.7, label='U')
        ax.plot(epochs,
                [abs(w) for w in self.history['W']],
                'g-', alpha=0.7, label='|W|')
        ax.set_title('U vs |W|')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. U/W ratio
        ax = axes[0, 2]
        ax.plot(epochs, self.history['U_over_W'],
                'k-', linewidth=0.8, alpha=0.7)
        ax.axhline(y=0.5, color='r', linestyle='--',
                   alpha=0.7, label='Optimal (0.5)')
        ax.set_title('U/|W| ratio (→ 0.5)')
        ax.set_ylim(0, 2)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Gradient norm
        ax = axes[1, 0]
        ax.semilogy(epochs,
                    self.history['grad_norm'],
                    'purple', linewidth=0.8, alpha=0.7)
        ax.set_title('Gradient Norm')
        ax.set_xlabel('Epoch')
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
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)

        # 6. Learning rate
        ax = axes[1, 2]
        ax.semilogy(epochs, self.history['lr'],
                    'orange', linewidth=2)
        ax.set_title('Learning Rate')
        ax.set_xlabel('Epoch')
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
        print(f"  Best  Π_norm:  {self.best_Pi:.6e} "
              f"(epoch {self.best_epoch})")

        if self._target_energies:
            avg_t = np.mean(self._target_energies)
            ratio = self.best_Pi / avg_t
            print(f"  Target Π_norm: {avg_t:.6e}")
            print(f"  Ratio (best/target): {ratio:.4f}")
            if abs(ratio - 1.0) < 0.01:
                print(f"  ✓ CONVERGED to target energy!")
            elif abs(ratio - 1.0) < 0.05:
                print(f"  ~ Close to target "
                      f"(within 5%)")
            elif abs(ratio - 1.0) < 0.1:
                print(f"  ~ Fair (within 10%)")
            else:
                print(f"  ✗ Not converged to target")

        # U/W analysis
        final_UW = self.history['U_over_W'][-1]
        print(f"\n  Final U/|W|: {final_UW:.4f}")
        if abs(final_UW - 0.5) < 0.02:
            print(f"  ✓ Optimal ratio achieved")
        elif abs(final_UW - 0.5) < 0.1:
            print(f"  ~ Close to optimal")

        # Convergence rate
        Pi_vals = self.history['Pi']
        if len(Pi_vals) >= 200:
            early = np.mean(Pi_vals[100:200])
            late = np.mean(Pi_vals[-100:])
            print(f"\n  Convergence:")
            print(f"    Epochs 100-200: "
                  f"Π = {early:.4e}")
            print(f"    Last 100:       "
                  f"Π = {late:.4e}")

        # Gradient norm analysis
        gn = self.history['grad_norm']
        if gn:
            print(f"\n  Gradient norm:")
            print(f"    Start: {gn[0]:.4e}")
            print(f"    End:   {gn[-1]:.4e}")
            if gn[-1] < gn[0] * 0.01:
                print(f"  ✓ Gradients decayed "
                      f"(converging)")

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

    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 60)
    print("  PIGNN TRAINING — Energy-Based v2")
    print("=" * 60)

    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")