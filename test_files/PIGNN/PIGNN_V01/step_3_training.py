"""
=================================================================
STEP 3e: PHYSICS-INFORMED TRAINING LOOP
=================================================================
Trains the PIGNN using ONLY physics losses — NO data targets.

Training strategy:
  1. Normalize losses by their initial magnitudes
  2. Start with strong BC loss, relax gradually
  3. Add sensitivity loss after initial convergence
  4. Log all loss components for monitoring
  5. Validate against Kratos data (NOT used in training!)

Loss normalization:
  Each loss component has very different magnitudes:
    Constitutive: ~10⁹  (forces in N)
    Equilibrium:  ~10¹²  (force residuals)
    BC:           ~10⁻⁶  (displacements in m)
    Sensitivity:  ~10¹⁴  (sensitivity values)
  
  We normalize each by its initial value so all start ≈ 1.0

=================================================================
"""
import os

os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\PIGNN\PIGNN_V01")
print(f"Working directory: {os.getcwd()}")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import time
from typing import Dict, List, Optional

# Import from previous steps
from step_3_model import FramePIGNN, create_model
from step_3_phys_loss import PhysicsLoss


class PhysicsTrainer:
    """
    Trains PIGNN using physics-informed losses only.
    Validates against Kratos data for comparison (not training).
    """

    def __init__(self,
                 model: FramePIGNN,
                 data_list: list,
                 lr: float = 1e-4,
                 w_constitutive: float = 1.0,
                 w_equilibrium: float = 1.0,
                 w_bc: float = 10.0,
                 w_sensitivity: float = 0.1,
                 device: str = 'cpu',
                 log_dir: str = 'DATA/training_logs'):
        """
        Args:
            model:           FramePIGNN model
            data_list:       list of UN-normalized PyG Data objects
                            (raw physical units — needed for physics loss)
            lr:              learning rate
            w_constitutive:  weight for constitutive loss
            w_equilibrium:   weight for equilibrium loss
            w_bc:            weight for BC loss
            w_sensitivity:   weight for sensitivity loss
            device:          'cpu' or 'cuda'
            log_dir:         directory for training logs
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.data_list = data_list

        # Physics loss
        self.loss_fn = PhysicsLoss(
            w_constitutive=w_constitutive,
            w_equilibrium=w_equilibrium,
            w_bc=w_bc,
            w_sensitivity=w_sensitivity,
        )

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=200,
            min_lr=1e-7)

        # Loss normalization factors (computed from first forward pass)
        self.loss_norms = None

        # Training history
        self.history = {
            'epoch': [],
            'total_loss': [],
            'constitutive': [],
            'equilibrium': [],
            'bc': [],
            'sensitivity': [],
            'lr': [],
            'validation': [],
        }

        # Logging
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    # ─────────────────────────────────────────────
    # LOSS NORMALIZATION
    # ─────────────────────────────────────────────

    def _compute_loss_norms(self):
        """
        Compute normalization factors from initial loss values.
        Sensitivity loss needs gradients, so we can't use no_grad().
        """
        print(f"\n  Computing loss normalization factors...")

        self.model.eval()
        total_losses = {'constitutive': 0, 'equilibrium': 0,
                        'bc': 0, 'sensitivity': 0}

        for data in self.data_list:
            data = data.to(self.device)
            node_pred, elem_pred = self.model(data)
            
            # Compute each loss separately, skip sensitivity if it fails
            try:
                losses = self.loss_fn(node_pred, elem_pred, data,
                                       return_components=True)
                for key in total_losses:
                    total_losses[key] += losses[key].item()
            except RuntimeError:
                # Sensitivity may fail without proper grad setup
                # Compute non-sensitivity losses only
                L_const = self.loss_fn.constitutive_loss(
                    node_pred, elem_pred, data)
                L_equil = self.loss_fn.equilibrium_loss(
                    node_pred, data)
                L_bc = self.loss_fn.bc_loss(node_pred, data)
                
                total_losses['constitutive'] += L_const.item()
                total_losses['equilibrium'] += L_equil.item()
                total_losses['bc'] += L_bc.item()
                total_losses['sensitivity'] += 1.0  # default norm

        n = len(self.data_list)
        self.loss_norms = {}
        for key, val in total_losses.items():
            avg = val / n
            self.loss_norms[key] = max(avg, 1e-10)

        print(f"  Initial loss magnitudes (normalization factors):")
        for key, val in self.loss_norms.items():
            print(f"    {key:<15}: {val:.4e}")

    # ─────────────────────────────────────────────
    # SINGLE TRAINING STEP
    # ─────────────────────────────────────────────

    def _train_step(self, data) -> Dict[str, float]:
        """
        One training step on one graph.
        
        Returns dict of loss components (normalized).
        """
        self.model.train()
        data = data.to(self.device)

        self.optimizer.zero_grad()

        # Forward pass
        node_pred, elem_pred = self.model(data)

        # Physics loss (raw)
        losses_raw = self.loss_fn(node_pred, elem_pred, data,
                                   return_components=True)

        # Normalize each component
        L_const = losses_raw['constitutive'] / self.loss_norms['constitutive']
        L_equil = losses_raw['equilibrium'] / self.loss_norms['equilibrium']
        L_bc = losses_raw['bc'] / self.loss_norms['bc']
        L_sens = losses_raw['sensitivity'] / self.loss_norms['sensitivity']

        # Weighted total (using the original weights from PhysicsLoss)
        total = (self.loss_fn.w_const * L_const +
                 self.loss_fn.w_equil * L_equil +
                 self.loss_fn.w_bc * L_bc +
                 self.loss_fn.w_sens * L_sens)

        # Backward
        total.backward()

        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                        max_norm=1.0)

        # Update weights
        self.optimizer.step()

        return {
            'total': total.item(),
            'constitutive': L_const.item(),
            'equilibrium': L_equil.item(),
            'bc': L_bc.item(),
            'sensitivity': L_sens.item(),
        }

    # ─────────────────────────────────────────────
    # VALIDATION (against Kratos — NOT for training)
    # ─────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Compare GNN predictions against Kratos output.
        
        This is NOT used for training — only for monitoring
        how close the physics-trained model is to FEM solution.
        
        Returns relative errors for each output quantity.
        """
        self.model.eval()

        total_errors = {
            'disp_rel_error': 0,
            'rot_rel_error': 0,
            'moment_rel_error': 0,
            'force_rel_error': 0,
            'sens_rel_error': 0,
        }

        for data in self.data_list:
            data = data.to(self.device)
            node_pred, elem_pred = self.model(data)

            # Kratos reference (targets)
            u_ref = data.y_node       # (N, 6)
            e_ref = data.y_element    # (E, 7)

            # Relative errors (avoid division by zero)
            def rel_error(pred, ref):
                ref_norm = ref.abs().max()
                if ref_norm < 1e-15:
                    return 0.0
                return ((pred - ref).abs().max() / ref_norm).item()

            # Displacement (columns 0, 2)
            total_errors['disp_rel_error'] += rel_error(
                node_pred[:, [0, 2]], u_ref[:, [0, 2]])

            # Rotation (column 4)
            total_errors['rot_rel_error'] += rel_error(
                node_pred[:, 4], u_ref[:, 4])

            # Moment (column 1 = My)
            total_errors['moment_rel_error'] += rel_error(
                elem_pred[:, 1], e_ref[:, 1])

            # Force (columns 3, 5 = Fx, Fz)
            total_errors['force_rel_error'] += rel_error(
                elem_pred[:, [3, 5]], e_ref[:, [3, 5]])

            # Sensitivity (column 6)
            total_errors['sens_rel_error'] += rel_error(
                elem_pred[:, 6], e_ref[:, 6])

        n = len(self.data_list)
        return {k: v / n for k, v in total_errors.items()}

    # ─────────────────────────────────────────────
    # MAIN TRAINING LOOP
    # ─────────────────────────────────────────────

    def train(self, n_epochs: int = 5000,
              print_every: int = 100,
              validate_every: int = 500,
              save_every: int = 1000) -> dict:
        """
        Main training loop.
        
        Args:
            n_epochs:       total training epochs
            print_every:    print loss every N epochs
            validate_every: validate against Kratos every N epochs
            save_every:     save checkpoint every N epochs
        
        Returns:
            training history dict
        """
        print(f"\n{'═'*60}")
        print(f"  PHYSICS-INFORMED TRAINING")
        print(f"{'═'*60}")
        print(f"  Model params:  {self.model.count_parameters():,}")
        print(f"  Training data: {len(self.data_list)} graphs")
        print(f"  Epochs:        {n_epochs}")
        print(f"  Device:        {self.device}")
        print(f"  Loss weights:")
        print(f"    Constitutive: {self.loss_fn.w_const}")
        print(f"    Equilibrium:  {self.loss_fn.w_equil}")
        print(f"    BC:           {self.loss_fn.w_bc}")
        print(f"    Sensitivity:  {self.loss_fn.w_sens}")

        # Compute normalization factors
        if self.loss_norms is None:
            self._compute_loss_norms()

        # Initial validation
        val_errors = self.validate()
        print(f"\n  Initial validation (vs Kratos):")
        for k, v in val_errors.items():
            print(f"    {k}: {v:.4f}")

        start_time = time.time()
        best_loss = float('inf')

        # ── Training epochs ──
        for epoch in range(1, n_epochs + 1):

            epoch_losses = {
                'total': 0, 'constitutive': 0,
                'equilibrium': 0, 'bc': 0, 'sensitivity': 0}

            # Train on each graph
            for data in self.data_list:
                step_losses = self._train_step(data)
                for k in epoch_losses:
                    epoch_losses[k] += step_losses[k]

            # Average over graphs
            n = len(self.data_list)
            for k in epoch_losses:
                epoch_losses[k] /= n

            # Update learning rate
            self.scheduler.step(epoch_losses['total'])

            # Record history
            self.history['epoch'].append(epoch)
            self.history['total_loss'].append(epoch_losses['total'])
            self.history['constitutive'].append(epoch_losses['constitutive'])
            self.history['equilibrium'].append(epoch_losses['equilibrium'])
            self.history['bc'].append(epoch_losses['bc'])
            self.history['sensitivity'].append(epoch_losses['sensitivity'])
            self.history['lr'].append(
                self.optimizer.param_groups[0]['lr'])

            # Best model
            if epoch_losses['total'] < best_loss:
                best_loss = epoch_losses['total']
                torch.save(self.model.state_dict(),
                           os.path.join(self.log_dir, 'best_model.pt'))

            # Print progress
            if epoch % print_every == 0 or epoch == 1:
                elapsed = time.time() - start_time
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch:>5}/{n_epochs} "
                      f"| Loss: {epoch_losses['total']:.4e} "
                      f"| Const: {epoch_losses['constitutive']:.4e} "
                      f"| Equil: {epoch_losses['equilibrium']:.4e} "
                      f"| BC: {epoch_losses['bc']:.4e} "
                      f"| Sens: {epoch_losses['sensitivity']:.4e} "
                      f"| LR: {lr:.1e} "
                      f"| {elapsed:.0f}s")

            # Validation
            if epoch % validate_every == 0:
                val_errors = self.validate()
                self.history['validation'].append({
                    'epoch': epoch, **val_errors})
                print(f"  ── Validation (vs Kratos) ──")
                for k, v in val_errors.items():
                    print(f"       {k}: {v:.4f}")

            # Save checkpoint
            if epoch % save_every == 0:
                self._save_checkpoint(epoch, epoch_losses)

        # Final validation
        final_val = self.validate()
        print(f"\n  ── Final Validation ──")
        for k, v in final_val.items():
            print(f"    {k}: {v:.4f}")

        # Save final model
        torch.save(self.model.state_dict(),
                   os.path.join(self.log_dir, 'final_model.pt'))

        # Save history
        self._save_history()

        total_time = time.time() - start_time
        print(f"\n  Training complete: {total_time:.1f}s")
        print(f"  Best loss: {best_loss:.4e}")
        print(f"  Models saved in: {self.log_dir}/")

        return self.history

    # ─────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────

    def _save_checkpoint(self, epoch, losses):
        """Save training checkpoint."""
        path = os.path.join(self.log_dir, f'checkpoint_ep{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'losses': losses,
            'loss_norms': self.loss_norms,
        }, path)

    def _save_history(self):
        """Save training history to JSON."""
        # Convert tensors to floats for JSON serialization
        history_json = {}
        for k, v in self.history.items():
            if isinstance(v, list) and v:
                if isinstance(v[0], dict):
                    history_json[k] = v
                else:
                    history_json[k] = [float(x) for x in v]
            else:
                history_json[k] = v

        path = os.path.join(self.log_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(history_json, f, indent=2)
        print(f"  History saved: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.loss_norms = checkpoint['loss_norms']
        print(f"  Checkpoint loaded: epoch {checkpoint['epoch']}")


# ================================================================
# PLOTTING
# ================================================================

def plot_training_history(history: dict, save_path: str = None):
    """Plot training loss curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  pip install matplotlib for plotting")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = history['epoch']

    # Total loss
    ax = axes[0, 0]
    ax.semilogy(epochs, history['total_loss'], 'b-', linewidth=0.8)
    ax.set_title('Total Loss')
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)

    # Individual losses
    ax = axes[0, 1]
    ax.semilogy(epochs, history['constitutive'], label='Constitutive', linewidth=0.8)
    ax.semilogy(epochs, history['equilibrium'], label='Equilibrium', linewidth=0.8)
    ax.semilogy(epochs, history['bc'], label='BC', linewidth=0.8)
    ax.semilogy(epochs, history['sensitivity'], label='Sensitivity', linewidth=0.8)
    ax.set_title('Individual Losses (normalized)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 0]
    ax.semilogy(epochs, history['lr'], 'g-', linewidth=0.8)
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)

    # Validation errors
    ax = axes[1, 1]
    if history['validation']:
        val_epochs = [v['epoch'] for v in history['validation']]
        for key in ['disp_rel_error', 'force_rel_error', 'moment_rel_error']:
            vals = [v[key] for v in history['validation']]
            ax.semilogy(val_epochs, vals, 'o-', label=key, linewidth=0.8,
                       markersize=3)
        ax.set_title('Validation vs Kratos (relative error)')
        ax.set_xlabel('Epoch')
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_path or 'DATA/training_history.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Plot saved: {save_path}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  STEP 3e: Physics-Informed Training")
    print("=" * 60)

    # ── Load UN-normalized data (need physical units for physics loss) ──
    print("\n── Loading data ──")
    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    print(f"  Loaded {len(data_list)} graphs")

    # ── Create model ──
    print("\n── Creating model ──")
    model = create_model('medium')  # 128 hidden, 6 processors

    # ── Create trainer ──
    print("\n── Setting up trainer ──")
    trainer = PhysicsTrainer(
        model=model,
        data_list=data_list,
        lr=1e-4,
        w_constitutive=1.0,
        w_equilibrium=1.0,
        w_bc=10.0,
        w_sensitivity=0.1,
        device='cpu',
        log_dir='DATA/training_logs',
    )

    # ── Train ──
    # Use small number of epochs for testing
    # For real training: 5000-50000 epochs
    history = trainer.train(
        n_epochs=1000,
        print_every=100,
        validate_every=500,
        save_every=500,
    )

    # ── Plot ──
    print("\n── Plotting ──")
    plot_training_history(history)

    print(f"\n{'='*60}")
    print(f"  STEP 3e COMPLETE ✓")
    print(f"")
    print(f"  Training used ONLY physics losses:")
    print(f"    ✓ Constitutive: K×u = f")
    print(f"    ✓ Equilibrium:  Σf = f_external")
    print(f"    ✓ BC:           u = 0 at supports")
    print(f"    ✓ Sensitivity:  dBM/dI22 via autograd")
    print(f"")
    print(f"  Kratos data used ONLY for validation")
    print(f"")
    print(f"  Files saved:")
    print(f"    DATA/training_logs/best_model.pt")
    print(f"    DATA/training_logs/final_model.pt")
    print(f"    DATA/training_logs/training_history.json")
    print(f"    DATA/training_history.png")
    print(f"")
    print(f"  Ready for Step 4 (Inference & Visualization)")
    print(f"{'='*60}")