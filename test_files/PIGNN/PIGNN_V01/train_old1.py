"""
=================================================================
PHYSICS-INFORMED TRAINING LOOP
=================================================================
Trains FramePIGNN using ONLY physics loss — NO data targets.

Model outputs: [ux, uz, θy] per node (3 DOFs, hard BCs applied)
Loss:          ||F_internal + F_external||² at free nodes
Validation:    Compare against Kratos FEM (NOT used in training)
=================================================================
"""
import os

os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\PIGNN\PIGNN_V01")
print(f"Working directory: {os.getcwd()}")

import torch
import torch.optim as optim
import numpy as np
import os
import json
import time
from typing import Dict

from model import FramePIGNN
from losses import FramePhysicsLoss

class PhysicsTrainer:
    """
    Trains PIGNN using equilibrium loss only.
    Hard BCs are enforced inside the model (no BC loss needed).
    """

    def __init__(self,
                 model: FramePIGNN,
                 data_list: list,
                 lr: float = 1e-4,
                 device: str = 'cpu',
                 log_dir: str = 'DATA/training_logs'):

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.data_list = data_list

        # Single physics loss
        self.loss_fn = FramePhysicsLoss()

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5,
            patience=200, min_lr=1e-7
        )

        # Loss normalization (computed from first forward pass)
        self.loss_norm = None

        # History
        self.history = {
            'epoch': [],
            'loss': [],
            'lr': [],
            'validation': [],
        }

        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    # ─────────────────────────────────────────────
    # LOSS NORMALIZATION
    # ─────────────────────────────────────────────

    # def _compute_loss_norm(self):
    #     """
    #     Compute normalization factor from initial loss.
    #     Makes loss start near 1.0 regardless of physical units.
    #     """
    #     print(f"\n  Computing loss normalization factor...")
    #     self.model.eval()
    #     total = 0.0

    #     for data in self.data_list:
    #         data = data.to(self.device)
    #         with torch.no_grad():
    #             u_pred = self.model(data)
    #         # Need grad for loss computation (K·u)
    #         u_pred_grad = u_pred.detach().requires_grad_(False)
    #         loss = self.loss_fn(u_pred_grad, data)
    #         total += loss.item()

    #     self.loss_norm = max(total / len(self.data_list), 1e-10)
    #     print(f"  Initial loss magnitude: {self.loss_norm:.4e}")
        # print(f"  (All losses will be divided by this)")

    def _compute_loss_norm(self):
        print(f"\n  Computing loss normalization factor...")
        self.model.eval()
        total = 0.0

        for data in self.data_list:
            data = data.to(self.device)
            h, xi, fields, I22_grad = self.model(data)
            loss_raw, _, _ = self.loss_fn(
                fields, xi, data, I22_grad=I22_grad)
            total += loss_raw.item()

        self.loss_norm = max(total / len(self.data_list), 1e-10)
        print(f"  Initial loss magnitude: {self.loss_norm:.4e}")
    
    # ─────────────────────────────────────────────
    # SINGLE TRAINING STEP
    # ─────────────────────────────────────────────

    # def _train_step(self, data) -> float:
    #     """One training step on one graph."""
    #     self.model.train()
    #     data = data.to(self.device)

    #     self.optimizer.zero_grad()

    #     # Forward: model outputs (N, 3) with hard BCs
    #     u_pred = self.model(data)

    #     # Physics loss: equilibrium residual
    #     loss_raw = self.loss_fn(u_pred, data)

    #     # Normalize
    #     loss = loss_raw / self.loss_norm

    #     # Backward
    #     loss.backward()

    #     # Gradient clipping
    #     torch.nn.utils.clip_grad_norm_(
    #         self.model.parameters(), max_norm=1.0)

    #     # Update
    #     self.optimizer.step()

    #     return loss.item()
    def _train_step(self, data) -> dict:
        self.model.train()
        data = data.to(self.device)

        self.optimizer.zero_grad()

        # Forward
        h, xi, fields, I22_grad = self.model(data)

        # Physics loss (with sensitivity)
        loss_raw, loss_dict, dM_dI = self.loss_fn(
            fields, xi, data, I22_grad=I22_grad)

        loss = loss_raw / self.loss_norm
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss_dict

    # ─────────────────────────────────────────────
    # VALIDATION (vs Kratos — NOT for training)
    # ─────────────────────────────────────────────

    @torch.no_grad()
    # def validate(self) -> Dict[str, float]:
    #     """
    #     Compare predictions against Kratos FEM solution.
    #     NOT used for training — monitoring only.
        
    #     Model outputs [ux, uz, θy] (3 DOFs).
    #     Kratos y_node has [ux, uy, uz, rx, ry, rz] (6 DOFs).
    #     Compare active DOFs: ux(col0), uz(col2), θy=ry(col4).
    #     """
    #     self.model.eval()
    #     errors = {'ux_rel': 0, 'uz_rel': 0, 'ry_rel': 0}

    #     for data in self.data_list:
    #         data = data.to(self.device)
    #         u_pred = self.model(data)  # (N, 3) = [ux, uz, θy]

    #         # Kratos reference
    #         u_ref = data.y_node  # (N, 6)
    #         ux_ref = u_ref[:, 0]
    #         uz_ref = u_ref[:, 2]
    #         ry_ref = u_ref[:, 4]

    #         def rel_err(pred, ref):
    #             denom = ref.abs().max()
    #             if denom < 1e-15:
    #                 return 0.0
    #             return ((pred - ref).abs().max() / denom).item()

    #         errors['ux_rel'] += rel_err(u_pred[:, 0], ux_ref)
    #         errors['uz_rel'] += rel_err(u_pred[:, 1], uz_ref)
    #         errors['ry_rel'] += rel_err(u_pred[:, 2], ry_ref)

    #     n = len(self.data_list)
    #     return {k: v / n for k, v in errors.items()}
    def validate(self) -> Dict[str, float]:
        """
        Compare against Kratos by evaluating fields at ξ=0 and ξ=1.
        
        fields at ξ=0 → displacement at node n1
        fields at ξ=1 → displacement at node n2
        Average if node appears in multiple elements.
        """
        self.model.eval()
        errors = {'ux_rel': 0, 'uz_rel': 0, 'ry_rel': 0}

        for data in self.data_list:
            data = data.to(self.device)
            # h, xi, fields = self.model(data)
            h, xi, fields, I22_grad = self.model(data)

            N = data.num_nodes
            conn = data.connectivity
            n1 = conn[:, 0]
            n2 = conn[:, 1]

            # Collect nodal displacements from element ends
            # fields: (E, n_pts, 3) — columns are [u_local, w_local, θ]
            u_at_0 = fields[:, 0, :]     # (E, 3) at ξ=0 (node n1)
            u_at_1 = fields[:, -1, :]    # (E, 3) at ξ=1 (node n2)

            # Average at each node
            node_sum = torch.zeros(N, 3, device=data.x.device)
            node_cnt = torch.zeros(N, 1, device=data.x.device)
            ones = torch.ones(n1.shape[0], 1, device=data.x.device)

            node_sum.scatter_add_(0, n1.unsqueeze(1).expand(-1, 3), u_at_0)
            node_cnt.scatter_add_(0, n1.unsqueeze(1), ones)
            node_sum.scatter_add_(0, n2.unsqueeze(1).expand(-1, 3), u_at_1)
            node_cnt.scatter_add_(0, n2.unsqueeze(1), ones)

            node_avg = node_sum / node_cnt.clamp(min=1)
            # node_avg: (N, 3) = [u_local, w_local, θ] averaged

            # NOTE: fields are in LOCAL coords per element
            # For validation we need GLOBAL coords
            # At nodes connected to elements of SAME orientation,
            # local ≈ global mapping is known:
            #   Beam (c=1,s=0):  u_local=ux, w_local=uz
            #   Column (c=0,s=1): u_local=uz, w_local=-ux
            # 
            # Simplified: compare magnitudes
            # (proper validation needs T^T transform per element)

            u_ref = data.y_node  # (N, 6)

            def rel_err(pred, ref):
                denom = ref.abs().max()
                if denom < 1e-15:
                    return 0.0
                return ((pred - ref).abs().max() / denom).item()

            # Approximate comparison (magnitude-based)
            errors['ux_rel'] += rel_err(node_avg[:, 0], u_ref[:, 0])
            errors['uz_rel'] += rel_err(node_avg[:, 1], u_ref[:, 2])
            errors['ry_rel'] += rel_err(node_avg[:, 2], u_ref[:, 4])

        n = len(self.data_list)
        return {k: v / n for k, v in errors.items()}

    # ─────────────────────────────────────────────
    # MAIN TRAINING LOOP
    # ─────────────────────────────────────────────

    def train(self, n_epochs: int = 5000,
              print_every: int = 100,
              validate_every: int = 500,
              save_every: int = 1000) -> dict:

        print(f"\n{'═'*60}")
        print(f"  PHYSICS-INFORMED TRAINING")
        print(f"{'═'*60}")
        print(f"  Model params:  {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"  Training data: {len(self.data_list)} graphs")
        print(f"  Epochs:        {n_epochs}")
        print(f"  Device:        {self.device}")
        print(f"  Loss:          Equilibrium ||F_int + F_ext||²")
        print(f"  BCs:           Hard (zeroed in model)")

        # Compute normalization
        if self.loss_norm is None:
            self._compute_loss_norm()

        # Initial validation
        val_errors = self.validate()
        print(f"\n  Initial validation (vs Kratos):")
        for k, v in val_errors.items():
            print(f"    {k}: {v:.4f}")

        start_time = time.time()
        best_loss = float('inf')

        # for epoch in range(1, n_epochs + 1):

        #     epoch_loss = 0.0

        #     for data in self.data_list:
        #         epoch_loss += self._train_step(data)

        #     epoch_loss /= len(self.data_list)

        #     # Scheduler
        #     self.scheduler.step(epoch_loss)

        #     # Record
        #     self.history['epoch'].append(epoch)
        #     self.history['loss'].append(epoch_loss)
        #     self.history['lr'].append(
        #         self.optimizer.param_groups[0]['lr'])

        #     # Best model
        #     if epoch_loss < best_loss:
        #         best_loss = epoch_loss
        #         torch.save(self.model.state_dict(),
        #                    os.path.join(self.log_dir, 'best_model.pt'))

        #     # Print
        #     if epoch % print_every == 0 or epoch == 1:
        #         elapsed = time.time() - start_time
        #         lr = self.optimizer.param_groups[0]['lr']
        #         print(f"  Epoch {epoch:>5}/{n_epochs} "
        #               f"| Loss: {epoch_loss:.6e} "
        #               f"| LR: {lr:.1e} "
        #               f"| {elapsed:.0f}s")

        for epoch in range(1, n_epochs + 1):

            epoch_losses = {}

            for data in self.data_list:
                step_dict = self._train_step(data)
                for k, v in step_dict.items():
                    epoch_losses[k] = epoch_losses.get(k, 0) + v

            n = len(self.data_list)
            for k in epoch_losses:
                epoch_losses[k] /= n

            # Scheduler uses total
            self.scheduler.step(epoch_losses['total'])

            # Record
            self.history['epoch'].append(epoch)
            self.history['loss'].append(epoch_losses['total'])
            self.history['lr'].append(
                self.optimizer.param_groups[0]['lr'])

            # Best model
            if epoch_losses['total'] < best_loss:
                best_loss = epoch_losses['total']
                torch.save(self.model.state_dict(),
                        os.path.join(self.log_dir, 'best_model.pt'))

            # Print
            if epoch % print_every == 0 or epoch == 1:
                elapsed = time.time() - start_time
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch:>5}/{n_epochs} "
                    f"| Total: {epoch_losses['total']:.4e} "
                    f"| Axial: {epoch_losses['axial_equilibrium']:.4e} "
                    f"| Bend: {epoch_losses['bending_equilibrium']:.4e} "
                    f"| Compat: {epoch_losses['compatibility']:.4e} "
                    f"| Contin: {epoch_losses['continuity']:.4e} "
                    f"| Sens: {epoch_losses['sensitivity']:.4e} "
                    f"| LR: {lr:.1e} "
                    f"| {elapsed:.0f}s")

            # Validate
            if epoch % validate_every == 0:
                val_errors = self.validate()
                self.history['validation'].append(
                    {'epoch': epoch, **val_errors})
                print(f"  ── Validation (vs Kratos) ──")
                for k, v in val_errors.items():
                    print(f"       {k}: {v:.4f}")

            # Checkpoint
            if epoch % save_every == 0:
                self._save_checkpoint(epoch, epoch_losses)

        # Final
        final_val = self.validate()
        print(f"\n  ── Final Validation ──")
        for k, v in final_val.items():
            print(f"    {k}: {v:.4f}")

        torch.save(self.model.state_dict(),
                   os.path.join(self.log_dir, 'final_model.pt'))
        self._save_history()

        total_time = time.time() - start_time
        print(f"\n  Training complete: {total_time:.1f}s")
        print(f"  Best loss: {best_loss:.6e}")

        return self.history

    # ─────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────

    def _save_checkpoint(self, epoch, loss):
        path = os.path.join(self.log_dir, f'checkpoint_ep{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'loss': loss,
            'loss_norm': self.loss_norm,
        }, path)

    def _save_history(self):
        history_json = {}
        for k, v in self.history.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                history_json[k] = v
            elif isinstance(v, list):
                history_json[k] = [float(x) for x in v]
            else:
                history_json[k] = v

        path = os.path.join(self.log_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(history_json, f, indent=2)
        print(f"  History saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, weights_only=False)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.loss_norm = ckpt['loss_norm']
        print(f"  Checkpoint loaded: epoch {ckpt['epoch']}")


# ================================================================
# PLOTTING
# ================================================================

def plot_training_history(history: dict, save_path: str = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  pip install matplotlib for plotting")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    epochs = history['epoch']

    # Loss
    axes[0].semilogy(epochs, history['loss'], 'b-', linewidth=0.8)
    axes[0].set_title('Equilibrium Loss (normalized)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)

    # Learning rate
    axes[1].semilogy(epochs, history['lr'], 'g-', linewidth=0.8)
    axes[1].set_title('Learning Rate')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(True, alpha=0.3)

    # Validation
    if history['validation']:
        val_epochs = [v['epoch'] for v in history['validation']]
        for key in ['ux_rel', 'uz_rel', 'ry_rel']:
            vals = [v[key] for v in history['validation']]
            axes[2].semilogy(val_epochs, vals, 'o-',
                            label=key, markersize=3)
        axes[2].set_title('Validation vs Kratos')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Relative Error')
        axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_path or 'DATA/training_history.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Plot saved: {save_path}")

# ================================================================
# INFERENCE — After training
# ================================================================

def predict_with_sensitivity(model, loss_fn, data, device='cpu'):
    """
    Run trained model and compute dM/dI22.
    Called AFTER training is complete.
    
    NOTE: No @torch.no_grad() because autograd is needed
    for dM/dI22 computation. But model weights are NOT updated.
    
    Args:
        model:    trained FramePIGNN
        loss_fn:  FramePhysicsLoss instance
        data:     single PyG Data object
        device:   'cpu' or 'cuda'
    
    Returns:
        dict with fields, moments, sensitivity
    """
    model.eval()  # disable dropout etc, but keep autograd alive
    data = data.to(device)

    # Forward pass (no optimizer, no backward on weights)
    h, xi, fields, I22_grad = model(data)

    # Compute derivatives (needs autograd for d/dξ)
    derivs = loss_fn.compute_derivatives(fields, xi)

    # Compute sensitivity (needs autograd for dM/dI22)
    dM_dI, M_field = loss_fn.compute_sensitivity(
        derivs,
        data.prop_E,
        I22_grad,
        data.elem_lengths,
        data.connectivity,
        data.response_node_flag,
    )

    # Detach everything — no more grad needed
    return {
        'fields': fields.detach(),        # (E, n_pts, 3)
        'M_field': M_field.detach(),       # (E, n_pts)
        'dM_dI22': dM_dI.detach(),         # (E,)
        'xi': xi.detach(),                 # (E, n_pts, 1)
    }
# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  PHYSICS-INFORMED TRAINING")
    print("=" * 60)

    # ── Load UN-normalized data (physical units needed) ──
    print("\n── Loading data ──")
    data_list = torch.load("DATA/graph_dataset.pt",
                           weights_only=False)
    print(f"  Loaded {len(data_list)} graphs")

    # ── Create model ──
    print("\n── Creating model ──")
    model = FramePIGNN(
        node_in_dim=9,
        edge_attr_dim=11,
        hidden_dim=128,
        n_layers=6,
        # node_out_dim=3,  # [ux, uz, θy]
    )
    n_params = sum(p.numel() for p in model.parameters()
                   if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ── Train ──
    trainer = PhysicsTrainer(
        model=model,
        data_list=data_list,
        lr=1e-4,
        device='cpu',
        log_dir='DATA/training_logs',
    )

    history = trainer.train(
        n_epochs=200,#2000
        print_every=10,#100
        validate_every=50,#500
        save_every=100,#1000
    )

    # ── Plot ──
    plot_training_history(history)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Loss: Single equilibrium ||F_int + F_ext||²")
    print(f"  BCs:  Hard (enforced in model)")
    print(f"  Data: NOT used in training (physics only)")
    print(f"{'='*60}")

    # ── Inference with sensitivity ──
    print("\n── Inference ──")

    # Load best model
    model.load_state_dict(
        torch.load('DATA/training_logs/best_model.pt',
                    weights_only=False))

    loss_fn = FramePhysicsLoss()

    # Run on first graph
    data = data_list[0]
    result = predict_with_sensitivity(model, loss_fn, data)

    print(f"  Fields shape:  {result['fields'].shape}")
    print(f"  Moment shape:  {result['M_field'].shape}")
    print(f"  dM/dI22 shape: {result['dM_dI22'].shape}")
    print(f"  dM/dI22 values (first 5 elements):")
    for e in range(min(5, result['dM_dI22'].shape[0])):
        print(f"    Element {e}: {result['dM_dI22'][e].item():.6e}")

    # Compare with Kratos sensitivity
    if hasattr(data, 'y_element'):
        kratos_sens = data.y_element[:, 6]
        print(f"\n  Comparison with Kratos dBM/dI22:")
        print(f"  {'Elem':>4} {'Predicted':>14} {'Kratos':>14} {'Error%':>10}")
        for e in range(min(5, kratos_sens.shape[0])):
            pred = result['dM_dI22'][e].item()
            ref = kratos_sens[e].item()
            err = abs(pred - ref) / max(abs(ref), 1e-15) * 100
            print(f"  {e:>4} {pred:>14.4e} {ref:>14.4e} {err:>10.2f}%")