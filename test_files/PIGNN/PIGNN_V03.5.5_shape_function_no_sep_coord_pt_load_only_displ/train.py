"""
train.py — Train PIGNN with Shape Function Physics Loss
"""

import os
from pathlib import Path
print(f"Working directory: {os.getcwd()}")
CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)
print(f"Working directory: {os.getcwd()}")

"""
=================================================================
train.py — Training Loop for Shape Function PIGNN
=================================================================

3-output model: [ux, uz, θy] per node
Forces derived via Euler-Bernoulli shape functions (exact).
Equilibrium enforced at free nodes.

Loss terms:
  L_eq   — nodal equilibrium (non-dim by F_c, M_c)
  L_bc   — support BCs (non-dim by u_c, theta_c)

No autograd. No data. Pure physics.
=================================================================
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from model import PIGNN
from physics_loss import EnergyLoss


# ================================================================
# A. CONFIGURATION
# ================================================================

class TrainConfig:

    # ── Data ──
    data_path       = "DATA/graph_dataset_norm.pt"
    save_dir        = "RESULTS"

    # ── Model ──
    hidden_dim      = 128
    n_layers        = 15#6
    node_in_dim     = 9
    edge_in_dim     = 10

    # ── Training ──
    epochs          = 10000
    lr              = 3e-3
    weight_decay    = 0#1e-5

    scheduler_type  = 'cosine' 
    # scheduler_step  = 150
    # scheduler_gamma = 0.5
    scheduler_T_max = 3000
    scheduler_eta_min = 1e-5

    grad_clip       = 1.0 

    # ── Loss weights ──
    w_eq            = 1.0
    w_bc            = 1.0

    # ── Logging ──
    print_every     = 200
    save_every      = 1000
    validate_every  = 200

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
            'epoch': [],
            'Pi': [], 'U_axial': [], 'U_bend': [],
            'W_ext': [], 'L_bc': [], 'total': [],
            'ea_factor': [],  
            'val_disp_error': [],
        }

    def _load_data(self):
        """Load graph dataset."""
        print(f"\n── Loading data ──")
        print(f"  Path: {self.cfg.data_path}")
        self.data_list = torch.load(
            self.cfg.data_path, weights_only=False
        )
        print(f"  Loaded {len(self.data_list)} graphs")

        # ── Add physics scales if missing ──
        from normalizer import PhysicsScaler
        if not hasattr(self.data_list[0], 'F_c'):
            print(f"  Physics scales NOT found — computing...")
            self.data_list = (
                PhysicsScaler.compute_and_store_list(
                    self.data_list
                )
            )
        else:
            print(f"  Physics scales present ✓")

        # ── Verify data state ──
        d = self.data_list[0]
        print(f"\n  Data verification:")
        print(f"    x range:        [{d.x.min():.4f}, "
              f"{d.x.max():.4f}]  "
              f"({'OK [0,1]' if d.x.max() < 2.0 else 'NOT NORMALIZED!'})")
        print(f"    edge_attr range: [{d.edge_attr.min():.4f}, "
              f"{d.edge_attr.max():.4f}]  "
              f"({'OK [0,1]' if d.edge_attr.max() < 2.0 else 'NOT NORMALIZED!'})")
        print(f"    coords range:   [{d.coords.min():.2f}, "
              f"{d.coords.max():.2f}]  (raw m)")
        print(f"    prop_E range:   [{d.prop_E.min():.2e}, "
              f"{d.prop_E.max():.2e}]  (raw Pa)")
        print(f"    F_ext range:    [{d.F_ext.min():.2f}, "
              f"{d.F_ext.max():.2f}]  (raw N)")
        print(f"    F_c  = {d.F_c.item():.4e}  (char. force)")
        print(f"    M_c  = {d.M_c.item():.4e}  (char. moment)")
        print(f"    u_c  = {d.u_c.item():.4e}  (char. disp)")
        print(f"    theta_c = {d.theta_c.item():.4e}  (char. rot)")

        self.train_data = self.data_list
        print(f"  Nodes: {d.num_nodes}, Elements: {d.n_elements}")

    def _create_model(self):
        """Create PIGNN model (3 outputs)."""
        print(f"\n── Creating model ──")
        self.model = PIGNN(
            node_in_dim=self.cfg.node_in_dim,
            edge_in_dim=self.cfg.edge_in_dim,
            hidden_dim=self.cfg.hidden_dim,
            n_layers=self.cfg.n_layers,
        ).to(self.device)
        self.model.summary()

    # def _create_loss(self):
    #     from physics_loss import EnergyLoss
    #     self.loss_fn = EnergyLoss(
    #         w_energy=self.cfg.w_eq,    # reuse w_eq config
    #         w_bc=self.cfg.w_bc,
    #     )
    def _create_loss(self): #curriculum
        from physics_loss import EnergyLoss
        self.loss_fn = EnergyLoss(
            w_energy=self.cfg.w_eq,
            w_bc=self.cfg.w_bc,
            ea_start_factor=1e-3,          # start at EA/1000
            ea_ramp_start=0,               # begin ramping immediately
            ea_ramp_end=5000,              # reach full EA at epoch 1500
            total_epochs=self.cfg.epochs,
        )

    def _create_optimizer(self):
        """Create optimizer and scheduler."""
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
            self.optimizer,
            T_max=self.cfg.epochs,
            eta_min=1e-5,
        )

    # ────────────────────────────────────────

    def train_one_epoch(self, epoch):
        self.model.train()
        self.loss_fn.current_epoch = epoch
        epoch_losses = {
            'Pi': 0, 'U_axial': 0, 'U_bend': 0,
            'W_ext': 0, 'L_bc': 0, 'total': 0,
            'ea_factor': 0,
        }
        n_graphs = len(self.train_data)

        for data in self.train_data:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            total_loss, loss_dict, pred = self.loss_fn(self.model, data)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.cfg.grad_clip)
            self.optimizer.step()
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]

        avg = {k: v / n_graphs for k, v in epoch_losses.items()}
        avg['ea_factor'] = self.loss_fn._get_ea_factor()    # ◀◀◀ exact value
        return avg

    @torch.no_grad()
    def validate(self):
        """Compare predictions with Kratos ground truth."""
        self.model.eval()

        total_disp_error = 0.0
        total_ux_error = 0.0
        total_uz_error = 0.0
        total_th_error = 0.0
        n_graphs = 0

        for data in self.train_data:
            data = data.to(self.device)

            self.model.train()    # needed for forward pass mechanics
            pred = self.model(data)
            self.model.eval()

            pred_disp = pred[:, 0:3]
            true_disp = data.y_node.to(self.device)

            # Relative L2 error (overall)
            err = (pred_disp - true_disp).pow(2).sum().sqrt()
            ref = true_disp.pow(2).sum().sqrt().clamp(min=1e-10)
            total_disp_error += (err / ref).item()

            # Per-DOF relative error
            for i, acc in enumerate([total_ux_error, total_uz_error, total_th_error]):
                e = (pred_disp[:, i] - true_disp[:, i]).pow(2).sum().sqrt()
                r = true_disp[:, i].pow(2).sum().sqrt().clamp(min=1e-10)
                if i == 0:
                    total_ux_error += (e / r).item()
                elif i == 1:
                    total_uz_error += (e / r).item()
                else:
                    total_th_error += (e / r).item()

            n_graphs += 1

        n = max(n_graphs, 1)
        val_metrics = {
            'disp_error': total_disp_error / n,
            'ux_error':   total_ux_error / n,
            'uz_error':   total_uz_error / n,
            'th_error':   total_th_error / n,
        }
        return val_metrics

    # ────────────────────────────────────────

    def train(self):
        """Full training loop."""
        print(f"\n{'='*80}")
        print(f"  TRAINING START (Shape Function PIGNN)")
        print(f"  {self.cfg.epochs} epochs, lr={self.cfg.lr}, "
              f"device={self.device}")
        print(f"  Losses: L_eq (equilibrium) + L_bc (boundary)")
        print(f"  Forces: derived from shape functions (exact)")
        print(f"{'='*80}")

        header = (f"  {'Epoch':>5} | {'Total':>10} | "
                  f"{'Pi':>10} {'U_ax':>10} {'U_bend':>10} "
                  f"{'W_ext':>10} {'L_bc':>10} | "
                  f"{'Disp Err':>10}")
        print(f"\n{header}")
        print(f"  {'-'*115}")

        best_loss = float('inf')
        best_disp = float('inf')
        t_start = time.time()

        for epoch in range(1, self.cfg.epochs + 1):

            # ── Train ──
            avg = self.train_one_epoch(epoch)

            # ── Record ──
            self.history['epoch'].append(epoch)
            for key in ['Pi', 'U_axial', 'U_bend', 'W_ext', 'L_bc', 'total', 'ea_factor']:
                self.history[key].append(avg[key])
            # for key in ['Pi', 'U_axial', 'U_bend', 'W_ext', 'L_bc', 'total']:
            #     self.history[key].append(avg[key])

            # ── Validate ──
            disp_err = float('nan')
            uz_err   = float('nan')
            if epoch % self.cfg.validate_every == 0 or epoch == 1:
                val = self.validate()
                disp_err = val['disp_error']
                uz_err   = val['uz_error']
                self.history['val_disp_error'].append(disp_err)

                # Track best displacement error
                if disp_err < best_disp:
                    best_disp = disp_err
                    self._save_checkpoint('best_disp.pt', epoch, avg)

            # ── Print ──
            if epoch % self.cfg.print_every == 0 or epoch == 1:
                disp_str = f"{disp_err:.4e}" if not np.isnan(disp_err) else "    ---   "
                uz_str   = f"{uz_err:.4e}"   if not np.isnan(uz_err)   else "    ---   "
                print(f"  {epoch:5d} | {avg['total']:10.4e} | "
                    f"{avg['Pi']:10.4e} {avg['U_axial']:10.4e} "
                    f"{avg['U_bend']:10.4e} {avg['W_ext']:10.4e} "
                    f"{avg['L_bc']:10.4e} | "
                    f"{disp_str} "
                    f"| EA×{avg['ea_factor']:.1e}"
                    )
            # ── Scheduler ──
            self.scheduler.step()

            # ── Save best loss ──
            if avg['total'] < best_loss:
                best_loss = avg['total']
                self._save_checkpoint('best.pt', epoch, avg)

            # ── Periodic save ──
            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(f'epoch_{epoch}.pt', epoch, avg)

        # ── Training complete ──
        elapsed = time.time() - t_start
        print(f"\n  {'-'*115}")
        print(f"  Training complete: {elapsed:.1f}s "
              f"({elapsed/self.cfg.epochs:.3f}s/epoch)")
        print(f"  Best total loss:    {best_loss:.6e}")
        print(f"  Best disp error:    {best_disp:.6e}")

        # ── Save final ──
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
            'config': {
                'hidden_dim': self.cfg.hidden_dim,
                'n_layers':   self.cfg.n_layers,
                'node_in_dim': self.cfg.node_in_dim,
                'edge_in_dim': self.cfg.edge_in_dim,
            },
        }, path)

    def _save_history(self):
        path = os.path.join(self.cfg.save_dir, 'history.pt')
        torch.save(self.history, path)
        print(f"  History saved: {path}")

    # ────────────────────────────────────────

    def _plot_losses(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("  matplotlib not found, skipping plots")
            return

        epochs = self.history['epoch']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Energy PIGNN — Training', fontsize=14, fontweight='bold')

        # Pi (total potential energy)
        ax = axes[0, 0]
        ax.plot(epochs, self.history['Pi'], 'k-', lw=1.5)
        ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
        ax.set_title('Pi (potential energy) — should go negative')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Pi / E_c')
        ax.grid(True, alpha=0.3)

        # U_axial and U_bend
        ax = axes[0, 1]
        ax.semilogy(epochs, [max(v, 1e-15) for v in self.history['U_axial']],
                    'tab:orange', lw=1.5, label='U_axial')
        ax.semilogy(epochs, [max(v, 1e-15) for v in self.history['U_bend']],
                    'tab:red', lw=1.5, label='U_bend')
        ax.set_title('Strain Energy Components')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # W_ext
        ax = axes[1, 0]
        ax.plot(epochs, self.history['W_ext'], 'tab:blue', lw=1.5)
        ax.set_title('External Work W_ext')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)

        # # EA factor
        # ax = axes[1, 1]
        ax.semilogy(epochs, self.history['ea_factor'], 'tab:green', lw=1.5)
        ax.set_title('EA Curriculum Factor')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('EA factor')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.cfg.save_dir, 'loss_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Loss curves saved: {path}")

        # Displacement error
        if self.history['val_disp_error']:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            n_val = len(self.history['val_disp_error'])
            val_epochs = [
                self.history['epoch'][i]
                for i in range(len(self.history['epoch']))
                if i == 0 or (self.history['epoch'][i] % self.cfg.validate_every == 0)
            ][:n_val]

            ax2.plot(val_epochs, self.history['val_disp_error'],
                    'ro-', ms=4, lw=1.5)
            ax2.axhline(y=1.0, color='gray', ls='--', alpha=0.5, label='zero solution')
            ax2.axhline(y=0.1, color='green', ls='--', alpha=0.5, label='10% error')
            ax2.set_title('Displacement Error vs Kratos')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Relative L2 Error')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            path2 = os.path.join(self.cfg.save_dir, 'disp_error.png')
            plt.savefig(path2, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"  Disp error saved: {path2}")

    # ────────────────────────────────────────

    def _analyze_results(self):
        print(f"\n{'='*80}")
        print(f"  RESULT ANALYSIS (Shape Function PIGNN)")
        print(f"{'='*80}")

        # Final values
        final = {k: self.history[k][-1] for k in
             ['Pi', 'U_axial', 'U_bend', 'W_ext', 'L_bc', 'total']}

        # Best by total loss
        best_idx = int(np.argmin(self.history['total']))
        best = {k: self.history[k][best_idx] for k in
                ['Pi', 'L_bc', 'total']}
        best_epoch = self.history['epoch'][best_idx]

        print(f"\n  Final losses (epoch {self.cfg.epochs}):")
        print(f"  {'-'*50}")
        for key, val in final.items():
            status = self._classify(val)
            print(f"    {key:<10} {val:>12.6e}  {status}")

        print(f"\n  Best losses (epoch {best_epoch}):")
        print(f"  {'-'*50}")
        for key, val in best.items():
            status = self._classify(val)
            print(f"    {key:<10} {val:>12.6e}  {status}")

        # Convergence
        print(f"\n  Convergence analysis:")
        print(f"  {'-'*60}")
        for key in ['Pi', 'L_bc', 'total']:
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
                print(f"    {key:<10} first10={first:.4e} "
                      f"last10={last:.4e}  {trend}")

        # Displacement accuracy
        if self.history['val_disp_error']:
            de = self.history['val_disp_error']
            best_de = min(de)
            final_de = de[-1]
            print(f"\n  Displacement error:")
            print(f"    Best:   {best_de:.6e}")
            print(f"    Final:  {final_de:.6e}")
            if best_de < 0.01:
                print(f"    EXCELLENT (< 1%)")
            elif best_de < 0.05:
                print(f"    GOOD (< 5%)")
            elif best_de < 0.20:
                print(f"    MODERATE (< 20%)")
            elif best_de < 1.0:
                print(f"    POOR but improving (< 100%)")
            else:
                print(f"    VERY POOR (predicting ~zero)")

        # Force magnitudes
        # print(f"\n  Derived force magnitudes (final epoch):")
        # print(f"    |N|_max = {self.history['N_max'][-1]:.2e} N")
        # print(f"    |M|_max = {self.history['M_max'][-1]:.2e} N-m")
        # print(f"    |V|_max = {self.history['V_max'][-1]:.2e} N")

        # Comparison with expected values
        # d = self.train_data[0]
        # if d.y_element is not None:
        #     print(f"\n  Kratos reference forces (case 0):")
        #     print(f"    |N|_max = {d.y_element[:, 0].abs().max():.2e} N")
        #     print(f"    |M|_max = {d.y_element[:, 1].abs().max():.2e} N-m")
        #     print(f"    |V|_max = {d.y_element[:, 2].abs().max():.2e} N")

        # print(f"\n{'='*80}")

    def _classify(self, val):
        if val < 1e-6:
            return "converged"
        elif val < 1e-2:
            return "partial"
        elif val < 1.0:
            return "slow"
        else:
            return "not converged"


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 80)
    print("  PIGNN TRAINING — Shape Function Physics Loss")
    print("=" * 80)
    print(f"  Device: {TrainConfig.device}")
    print(f"  Output: 3 per node [ux, uz, theta]")
    print(f"  Forces: derived from EB shape functions (exact)")
    print(f"  Loss:   equilibrium + boundary conditions")
    print(f"  No autograd. No data. Pure physics.")

    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()

    print(f"\n{'='*80}")
    print(f"  TRAINING COMPLETE")
    print(f"  Check {config.save_dir}/ for:")
    print(f"    loss_curves.png   — L_eq, L_bc, force magnitudes")
    print(f"    disp_error.png    — displacement error vs Kratos")
    print(f"    history.pt        — raw values")
    print(f"    best.pt           — best total loss checkpoint")
    print(f"    best_disp.pt      — best displacement error checkpoint")
    print(f"{'='*80}")