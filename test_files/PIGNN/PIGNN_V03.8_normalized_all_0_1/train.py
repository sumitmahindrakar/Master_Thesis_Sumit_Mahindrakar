"""
=================================================================
train.py — Training Loop for Corotational PIGNN
=================================================================

Pure physics training. No data loss.
Network outputs ~O(1), corotational handles scaling.
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
# from physics_loss import CorotationalPhysicsLoss
from step_5_loss_functions import NormalizedPhysicsLoss
# from step_2_grapg_constr import FrameData


# ================================================================
# A. CONFIGURATION
# ================================================================

class TrainConfig:

    data_path       = "DATA/graph_dataset_norm.pt"
    save_dir        = "RESULTS"

    hidden_dim      = 128
    n_layers        = 6
    node_in_dim     = 10
    edge_in_dim     = 7

    epochs          = 5000
    lr              = 1e-4 #5e-4
    weight_decay    = 0.0 #1e-5
    warmup_epochs   = 0
    grad_clip       = 1.0 #1.0

    print_every     = 50
    save_every      = 500
    validate_every  = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================================================================
# B. TRAINER
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
            'epoch':     [],
            'L_eq':      [],
            'L_force':   [],
            'L_moment':  [],
            'max_res':   [],
            'disp_err':  [],
        }

    def _load_data(self):
        print(f"\n── Loading data ──")
        
        # Load normalized dataset
        self.data_list = torch.load(
            self.cfg.data_path,  # "DATA/graph_dataset_norm.pt"
            weights_only=False
        )
        print(f"  Loaded {len(self.data_list)} normalized graphs")
        
        # Load normalizer for denormalization during validation
        from step_3_normalizer import CompleteNormalizer
        self.normalizer = CompleteNormalizer.load("DATA/normalizer.pt")
        
        # Verify normalization
        d = self.data_list[0]
        print(f"\n  Data verification:")
        print(f"    x range:        [{d.x.min():.4f}, {d.x.max():.4f}]")
        print(f"    edge_attr:      [{d.edge_attr.min():.4f}, {d.edge_attr.max():.4f}]")
        print(f"    u_scale:        {d.u_scale:.4e}")
        print(f"    theta_scale:    {d.theta_scale:.4e}")
        
        self.train_data = self.data_list

        # from normalizer import PhysicsScaler
        # if not hasattr(self.raw_data[0], 'u_c'):
        #     self.raw_data = (
        #         PhysicsScaler.compute_and_store_list(
        #             self.raw_data
        #         )
        #     )

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
        # self.loss_fn = NormalizedPhysicsLoss(data_warmup_epochs=200)
        self.loss_fn = NormalizedPhysicsLoss(force_weight=1.0,
                                                moment_weight=0.10)

    def _create_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        # self.scheduler = (
        #     torch.optim.lr_scheduler
        #     .CosineAnnealingWarmRestarts(
        #         self.optimizer,
        #         T_0=500,
        #         T_mult=1,
        #         eta_min=1e-6,
        #     )
        # )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.epochs,
            eta_min=1e-6,
        )

    def _get_warmup_lr(self, epoch):
        if epoch <= self.cfg.warmup_epochs:
            return (self.cfg.lr
                    * epoch / self.cfg.warmup_epochs)
        return None

    # ════════════════════════════════════════════════
    # TRAINING
    # ════════════════════════════════════════════════

    def train_one_epoch(self, epoch):
        self.model.train()
        
        # ← ADD THIS LINE HERE
        self.loss_fn.set_epoch(epoch)
        
        warmup_lr = self._get_warmup_lr(epoch)
        if warmup_lr is not None:
            for pg in self.optimizer.param_groups:
                pg['lr'] = warmup_lr

        epoch_loss = 0.0
        epoch_force = 0.0
        epoch_moment = 0.0
        epoch_maxres = 0.0
        last_loss_dict = None
        n_graphs = len(self.train_data)

        for data in self.train_data:
            data = data.to(self.device)

            self.optimizer.zero_grad()
            total_loss, loss_dict, pred_raw, beam_result = \
                self.loss_fn(self.model, data)

            if torch.isnan(total_loss):
                print(f"  ⚠ NaN at epoch {epoch}")
                continue

            total_loss.backward()

            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.cfg.grad_clip
                )

            self.optimizer.step()

            epoch_loss   += loss_dict['L_eq']
            epoch_force  += loss_dict['L_force']
            epoch_moment += loss_dict['L_moment']
            
            # ← ALSO FIX THIS: Change 'max_res_nd' to 'max_res'
            epoch_maxres  = max(epoch_maxres, loss_dict['max_res'])
            
            last_loss_dict = loss_dict

        avg = {
            'L_eq':     epoch_loss / n_graphs,
            'L_force':  epoch_force / n_graphs,
            'L_moment': epoch_moment / n_graphs,
            'max_res':  epoch_maxres,
            'last_dict': last_loss_dict,
        }
        return avg

    # ════════════════════════════════════════════════
    # VALIDATION
    # ════════════════════════════════════════════════

    def validate(self):
        self.model.eval()
        
        from step_4_physics_element import CorotationalBeam2DNormalized
        beam = CorotationalBeam2DNormalized()
        
        total_err = 0.0
        n_graphs = 0
        
        with torch.no_grad():
            for data in self.train_data:
                data = data.to(self.device)
                pred_norm = self.model(data)  # Normalized output [0,1]
                
                beam_result = beam(pred_norm, data)
                pred_phys = beam_result['phys_disp']  # Already denormalized
                
                if data.y_node is not None:
                    true_disp = data.y_node.to(self.device)
                    
                    err = (pred_phys - true_disp).pow(2).sum().sqrt()
                    ref = true_disp.pow(2).sum().sqrt().clamp(min=1e-10)
                    total_err += (err / ref).item()
                    n_graphs += 1
        
        return {'disp_err': total_err / max(n_graphs, 1)}

    # ════════════════════════════════════════════════
    # MAIN LOOP
    # ════════════════════════════════════════════════

    def train(self):
        print(f"\n{'═'*80}")
        print(f"  TRAINING — Corotational PIGNN")
        print(f"  {self.cfg.epochs} epochs, lr={self.cfg.lr}, "
              f"warmup={self.cfg.warmup_epochs}, "
              f"device={self.device}")
        print(f"{'═'*80}")

        header = (
            f"  {'Ep':>5} | {'L_eq':>10} | "
            f"{'L_force':>10} {'L_moment':>10} | "
            f"{'Max|R|':>10} | {'DispErr':>10} | "
            f"{'LR':>10}"
        )
        print(f"\n{header}")
        print(f"  {'-'*85}")

        best_loss = float('inf')
        best_disp_err = float('inf')
        t_start = time.time()

        for epoch in range(1, self.cfg.epochs + 1):

            avg = self.train_one_epoch(epoch)

            self.history['epoch'].append(epoch)
            self.history['L_eq'].append(avg['L_eq'])
            self.history['L_force'].append(avg['L_force'])
            self.history['L_moment'].append(
                avg['L_moment']
            )
            self.history['max_res'].append(avg['max_res'])

            if epoch > self.cfg.warmup_epochs:
                self.scheduler.step()

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

            if (epoch % self.cfg.print_every == 0
                    or epoch == 1
                    or epoch == self.cfg.epochs):
                lr = self.optimizer.param_groups[0]['lr']
                de_str = (
                    f"{disp_err:.4e}"
                    if not np.isnan(disp_err)
                    else "    ---   "
                )
                print(
                    f"  {epoch:5d} | "
                    f"{avg['L_eq']:10.4e} | "
                    f"{avg['L_force']:10.4e} "
                    f"{avg['L_moment']:10.4e} | "
                    f"{avg['max_res']:10.4e} | "
                    f"{de_str} | "
                    f"{lr:10.2e}"
                )

                # Extra diagnostics
                d = avg['last_dict']
                if (d is not None and
                        (epoch % (self.cfg.print_every * 2) == 0 or epoch == 1)):
                    # ← REMOVE DUPLICATE, KEEP ONLY THIS:
                    print(
                        f"        pred: [{d['pred_range'][0]:.4f}, {d['pred_range'][1]:.4f}]  "
                        f"ux: [{d['ux_range'][0]:.4e}, {d['ux_range'][1]:.4e}]  "
                        f"θ: [{d['theta_range'][0]:.4e}, {d['theta_range'][1]:.4e}]"
                    )
                    print(
                        f"        mag: {d['current_magnitude']:.3f} "
                        f"var: {d['output_variance']:.3f} "
                        f"min: {d['min_output']:.3f} "
                        f"(pen_wt: {d['penalty_weight']:.3f})"
                    )

            if avg['L_eq'] < best_loss:
                best_loss = avg['L_eq']
                self._save_checkpoint('best.pt', epoch, avg)

            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(f'epoch_{epoch}.pt', epoch, avg)

        elapsed = time.time() - t_start
        print(f"\n  {'-'*85}")
        print(f"  Done: {elapsed:.1f}s "
              f"({elapsed/self.cfg.epochs:.2f}s/epoch)")
        print(f"  Best L_eq:     {best_loss:.6e}")
        print(f"  Best disp_err: {best_disp_err:.6e}")

        self._save_checkpoint(
            'final.pt', self.cfg.epochs, avg
        )
        self._save_history()
        self._plot_losses()
        self._analyze_results()

    # ════════════════════════════════════════════════
    # UTILITIES
    # ════════════════════════════════════════════════

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
            print("  matplotlib not found")
            return

        epochs = self.history['epoch']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            'Corotational PIGNN Training',
            fontsize=14, fontweight='bold'
        )

        ax = axes[0, 0]
        ax.semilogy(epochs, self.history['L_eq'],
                    'b-', linewidth=1.0, alpha=0.7)
        ax.set_title('Equilibrium Loss (L_eq)')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.semilogy(epochs, self.history['L_force'],
                    'r-', linewidth=1.0, alpha=0.7,
                    label='L_force')
        ax.semilogy(epochs, self.history['L_moment'],
                    'g-', linewidth=1.0, alpha=0.7,
                    label='L_moment')
        ax.set_title('Force vs Moment')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.semilogy(epochs, self.history['max_res'],
                    'k-', linewidth=1.0, alpha=0.7)
        ax.set_title('Max |Residual|')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        valid_de = [
            (e, d) for e, d in
            zip(epochs, self.history['disp_err'])
            if not np.isnan(d)
        ]
        if valid_de:
            de_epochs, de_vals = zip(*valid_de)
            ax.semilogy(de_epochs, de_vals,
                       'mo-', linewidth=1.5)
        ax.set_title('Displacement Error vs Kratos')
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

        final_eq = self.history['L_eq'][-1]
        best_idx = np.argmin(self.history['L_eq'])
        best_eq = self.history['L_eq'][best_idx]
        best_epoch = self.history['epoch'][best_idx]

        print(f"\n  Final L_eq:  {final_eq:.6e}")
        print(f"  Best  L_eq:  {best_eq:.6e} "
              f"(epoch {best_epoch})")

        vals = self.history['L_eq']
        if len(vals) >= 20:
            first = np.mean(vals[:10])
            last = np.mean(vals[-10:])
            ratio = last / max(first, 1e-30)
            print(f"\n  Convergence: "
                  f"{first:.4e} → {last:.4e}")

            if ratio < 1e-6:
                print(f"  STATUS: FULLY CONVERGED ✓")
            elif ratio < 1e-4:
                print(f"  STATUS: CONVERGED ✓")
            elif ratio < 0.01:
                print(f"  STATUS: GOOD PROGRESS")
            elif ratio < 0.1:
                print(f"  STATUS: PARTIAL")
            else:
                print(f"  STATUS: SLOW")

        valid_de = [d for d in self.history['disp_err']
                    if not np.isnan(d)]
        if valid_de:
            last_de = valid_de[-1]
            best_de = min(valid_de)
            print(f"\n  Displacement error:")
            print(f"    Final: {last_de:.4e}")
            print(f"    Best:  {best_de:.4e}")
            if best_de < 0.01:
                print(f"  ACCURACY: EXCELLENT (<1%)")
            elif best_de < 0.05:
                print(f"  ACCURACY: GOOD (<5%)")
            elif best_de < 0.1:
                print(f"  ACCURACY: FAIR (<10%)")
            else:
                print(f"  ACCURACY: POOR")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  PIGNN TRAINING — Corotational")
    print("=" * 60)

    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")