"""
=================================================================
train.py — Energy-Based Training for PIGNN
=================================================================

Following Dalton et al.:
  1. Loss = Total Potential Energy (Π = U - W)
  2. Zero-init decoder (start at u=0)
  3. Two-phase LR: lr for first half, lr/10 for second
  4. No gradient clipping needed (energy is smooth)
  5. Per-sample gradient (not batched)
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

    epochs          = 2000
    lr              = 1e-3
    lr_decay_factor = 0.1    # lr → lr/10 at halfway
    weight_decay    = 0.0

    print_every     = 50
    save_every      = 500
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
            'epoch':     [],
            'Pi':        [],
            'U':         [],
            'W':         [],
            'disp_err':  [],
        }

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
        print(f"\n  Data verification:")
        print(f"    Nodes: {d.num_nodes}, "
              f"Elements: {d.n_elements}")
        print(f"    F_c  = {d.F_c.item():.4e}")
        print(f"    u_c  = {d.u_c.item():.4e}")
        print(f"    θ_c  = {d.theta_c.item():.4e}")

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

    def _create_model(self):
        print(f"\n── Creating model ──")
        self.model = PIGNN(
            node_in_dim=self.cfg.node_in_dim,
            edge_in_dim=self.cfg.edge_in_dim,
            hidden_dim=self.cfg.hidden_dim,
            n_layers=self.cfg.n_layers,
        ).to(self.device)
        self.model.summary()

        # Verify zero init
        with torch.no_grad():
            d = self.data_list[0].to(self.device)
            pred = self.model(d)
            print(f"  Initial pred max: "
                  f"{pred.abs().max().item():.6e} "
                  f"(should be ~0)")

    def _create_loss(self):
        self.loss_fn = FrameEnergyLoss()

    def _create_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    # ════════════════════════════════════════
    # TRAINING
    # ════════════════════════════════════════

    def train_one_epoch(self, epoch):
        self.model.train()

        epoch_Pi = 0.0
        epoch_U = 0.0
        epoch_W = 0.0
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
            self.optimizer.step()

            epoch_Pi += loss_dict['Pi_norm']
            epoch_U  += loss_dict['U_internal']
            epoch_W  += loss_dict['W_external']
            last_dict = loss_dict

        avg = {
            'Pi':     epoch_Pi / n_graphs,
            'U':      epoch_U / n_graphs,
            'W':      epoch_W / n_graphs,
            'last':   last_dict,
        }
        return avg

    # ════════════════════════════════════════
    # VALIDATION
    # ════════════════════════════════════════

    def validate(self):
        self.model.eval()

        total_err = 0.0
        n_graphs = 0

        with torch.no_grad():
            for i, data in enumerate(self.train_data):
                data = data.to(self.device)
                pred_raw = self.model(data)

                # Convert to physical
                pred_phys = torch.zeros_like(pred_raw)
                pred_phys[:, 0] = pred_raw[:, 0] * data.u_c
                pred_phys[:, 1] = pred_raw[:, 1] * data.u_c
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
                    ref = (true_disp.pow(2).sum().sqrt()
                           .clamp(min=1e-10))
                    total_err += (err / ref).item()
                    n_graphs += 1

        return {
            'disp_err': total_err / max(n_graphs, 1)
        }

    # ════════════════════════════════════════
    # MAIN LOOP
    # ════════════════════════════════════════

    def train(self):
        print(f"\n{'═'*80}")
        print(f"  TRAINING — Energy-Based PIGNN")
        print(f"  {self.cfg.epochs} epochs, "
              f"lr={self.cfg.lr}, "
              f"device={self.device}")
        print(f"  Phase 1: epochs 1-{self.cfg.epochs//2} "
              f"at lr={self.cfg.lr}")
        print(f"  Phase 2: epochs "
              f"{self.cfg.epochs//2+1}-{self.cfg.epochs} "
              f"at lr={self.cfg.lr * self.cfg.lr_decay_factor}")
        print(f"{'═'*80}")

        header = (
            f"  {'Ep':>5} | {'Π_norm':>12} | "
            f"{'U':>12} {'W':>12} {'U/|W|':>8} | "
            f"{'DispErr':>10} | {'LR':>10}"
        )
        print(f"\n{header}")
        print(f"  {'-'*90}")

        best_loss = float('inf')
        best_disp_err = float('inf')
        t_start = time.time()

        for epoch in range(1, self.cfg.epochs + 1):

            # ── Phase 2: reduce LR at halfway ──
            if epoch == self.cfg.epochs // 2 + 1:
                new_lr = (self.cfg.lr
                         * self.cfg.lr_decay_factor)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = new_lr
                print(f"\n  >>> Phase 2: "
                      f"LR → {new_lr:.1e} <<<\n")

            avg = self.train_one_epoch(epoch)

            self.history['epoch'].append(epoch)
            self.history['Pi'].append(avg['Pi'])
            self.history['U'].append(avg['U'])
            self.history['W'].append(avg['W'])

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
                lr = self.optimizer.param_groups[0]['lr']
                de = (f"{disp_err:.4e}"
                      if not np.isnan(disp_err)
                      else "    ---   ")

                U_over_W = (
                    abs(avg['U'])
                    / max(abs(avg['W']), 1e-30)
                )

                print(
                    f"  {epoch:5d} | "
                    f"{avg['Pi']:12.4e} | "
                    f"{avg['U']:12.4e} "
                    f"{avg['W']:12.4e} "
                    f"{U_over_W:8.4f} | "
                    f"{de} | "
                    f"{lr:10.2e}"
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

            # ── Save best ──
            if avg['Pi'] < best_loss:
                best_loss = avg['Pi']
                self._save_checkpoint(
                    'best.pt', epoch, avg
                )

            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(
                    f'epoch_{epoch}.pt', epoch, avg
                )

        elapsed = time.time() - t_start
        print(f"\n  {'-'*90}")
        print(f"  Done: {elapsed:.1f}s")
        print(f"  Best Π_norm:   {best_loss:.6e}")
        print(f"  Best disp_err: {best_disp_err:.6e}")

        self._save_checkpoint(
            'final.pt', self.cfg.epochs, avg
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
            'optimizer_state': (
                self.optimizer.state_dict()
            ),
            'losses': losses,
        }, path)

    def _save_history(self):
        path = os.path.join(
            self.cfg.save_dir, 'history.pt'
        )
        torch.save(self.history, path)

    def _plot_losses(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        epochs = self.history['epoch']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            'Energy-Based PIGNN Training',
            fontsize=14, fontweight='bold'
        )

        # Π vs epoch
        ax = axes[0, 0]
        ax.plot(epochs, self.history['Pi'],
                'b-', linewidth=1.0, alpha=0.7)
        ax.set_title('Π_norm (should go negative)')
        ax.set_xlabel('Epoch')
        ax.axhline(y=0, color='r', linestyle='--',
                   alpha=0.5)
        ax.grid(True, alpha=0.3)

        # U and W
        ax = axes[0, 1]
        ax.plot(epochs, self.history['U'],
                'r-', alpha=0.7, label='U (strain)')
        ax.plot(epochs,
                [abs(w) for w in self.history['W']],
                'g-', alpha=0.7, label='|W| (work)')
        ax.set_title('U_strain vs |W_external|')
        ax.legend()
        ax.set_yscale('symlog', linthresh=1e-6)
        ax.grid(True, alpha=0.3)

        # U/W ratio
        ax = axes[1, 0]
        ratios = [
            abs(u) / max(abs(w), 1e-30)
            for u, w in
            zip(self.history['U'], self.history['W'])
        ]
        ax.plot(epochs, ratios,
                'k-', linewidth=1.0, alpha=0.7)
        ax.axhline(y=0.5, color='r', linestyle='--',
                   alpha=0.5, label='Optimal (0.5)')
        ax.set_title('U/|W| ratio (→ 0.5)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Displacement error
        ax = axes[1, 1]
        valid_de = [
            (e, d) for e, d in
            zip(epochs, self.history['disp_err'])
            if not np.isnan(d)
        ]
        if valid_de:
            de_e, de_v = zip(*valid_de)
            ax.semilogy(de_e, de_v, 'mo-', linewidth=1.5)
        ax.set_title('Displacement Error vs Kratos')
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

        # Energy should be negative at convergence
        final_Pi = self.history['Pi'][-1]
        min_Pi = min(self.history['Pi'])
        min_idx = np.argmin(self.history['Pi'])

        print(f"\n  Final Π_norm:  {final_Pi:.6e}")
        print(f"  Min   Π_norm:  {min_Pi:.6e} "
              f"(epoch {self.history['epoch'][min_idx]})")

        if min_Pi < 0:
            print(f"  ✓ Energy went negative "
                  f"(non-trivial solution found!)")
        else:
            print(f"  ✗ Energy never went negative "
                  f"(stuck near zero)")

        # U/W ratio
        if self.history['W']:
            final_U = abs(self.history['U'][-1])
            final_W = abs(self.history['W'][-1])
            ratio = final_U / max(final_W, 1e-30)
            print(f"\n  Final U/|W|: {ratio:.4f}")
            if abs(ratio - 0.5) < 0.1:
                print(f"  ✓ Close to optimal "
                      f"(0.5 for linear)")
            elif ratio < 0.1:
                print(f"  ~ Solution still small")

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
            else:
                print(f"  ACCURACY: POOR")


# ════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════

if __name__ == "__main__":

    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 60)
    print("  PIGNN TRAINING — Energy-Based")
    print("=" * 60)

    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")