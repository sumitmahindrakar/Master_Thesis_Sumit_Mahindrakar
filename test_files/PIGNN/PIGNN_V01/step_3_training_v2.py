"""
=================================================================
STEP 3 REVISED: SCALED OUTPUT PIGNN
=================================================================
Problem: GNN collapses to trivial solution (all outputs ≈ 0)
Fix:     Scale GNN output to known physical ranges
         so physics equations are meaningful at the correct scale.

Physical ranges from your data:
  Displacement: ~1e-4 m
  Rotation:     ~1e-5 rad
  Moment:       ~1e3 N·m  
  Force:        ~1e4 N
  Sensitivity:  ~1e5 (varies wildly)
=================================================================
"""
import os

os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\PIGNN\PIGNN_V01")
print(f"Working directory: {os.getcwd()}")

import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from typing import Dict

from step_3_model import FramePIGNN, create_model, MLP
from step_3_phys_loss import PhysicsLoss


# ================================================================
# SCALED OUTPUT MODEL
# ================================================================

class ScaledFramePIGNN(nn.Module):
    """
    FramePIGNN with learnable output scaling.
    
    Output = base_model_output × scale_factor
    
    Scale factors are set from data statistics or physics:
      node scales: [ux, uy, uz, rx, ry, rz]
      elem scales: [Mx, My, Mz, Fx, Fy, Fz, dBM/dI22]
    
    This prevents trivial zero solution by anchoring outputs
    to physically meaningful magnitudes.
    """

    def __init__(self, base_model: FramePIGNN,
                 node_scales: torch.Tensor,
                 elem_scales: torch.Tensor):
        """
        Args:
            base_model:   FramePIGNN
            node_scales:  (6,) scale for [ux,uy,uz,rx,ry,rz]
            elem_scales:  (7,) scale for [Mx,My,Mz,Fx,Fy,Fz,sens]
        """
        super().__init__()
        self.base_model = base_model

        # Register as buffers (not parameters — not trained)
        self.register_buffer('node_scales', node_scales.float())
        self.register_buffer('elem_scales', elem_scales.float())

    def forward(self, data):
        node_raw, elem_raw = self.base_model(data)

        # Scale outputs to physical range
        node_pred = node_raw * self.node_scales
        elem_pred = elem_raw * self.elem_scales

        return node_pred, elem_pred

    def predict_components(self, data):
        node_pred, elem_pred = self.forward(data)
        return {
            'displacement': node_pred[:, 0:3],
            'rotation':     node_pred[:, 3:6],
            'moment':       elem_pred[:, 0:3],
            'force':        elem_pred[:, 3:6],
            'sensitivity':  elem_pred[:, 6:7],
        }

    def count_parameters(self):
        return self.base_model.count_parameters()


def compute_output_scales(data_list: list) -> tuple:
    """
    Compute output scale factors from dataset statistics.
    
    Scale = RMS of each output quantity across all cases.
    If RMS = 0, use 1.0 (no scaling).
    """
    all_node = torch.cat([d.y_node for d in data_list], dim=0)  # (N*cases, 6)
    all_elem = torch.cat([d.y_element for d in data_list], dim=0)  # (E*cases, 7)

    # RMS per output dimension
    node_scales = all_node.pow(2).mean(dim=0).sqrt()
    elem_scales = all_elem.pow(2).mean(dim=0).sqrt()

    # Replace zeros with 1.0
    node_scales = torch.where(node_scales < 1e-15,
                               torch.ones_like(node_scales), node_scales)
    elem_scales = torch.where(elem_scales < 1e-15,
                               torch.ones_like(elem_scales), elem_scales)

    print(f"\n  Output scales (RMS from data):")
    node_names = ['ux', 'uy', 'uz', 'rx', 'ry', 'rz']
    elem_names = ['Mx', 'My', 'Mz', 'Fx', 'Fy', 'Fz', 'sens']
    for name, scale in zip(node_names, node_scales):
        print(f"    {name:<6}: {scale:.4e}")
    for name, scale in zip(elem_names, elem_scales):
        print(f"    {name:<6}: {scale:.4e}")

    return node_scales, elem_scales


# ================================================================
# EQUILIBRIUM-FOCUSED TRAINER
# ================================================================

class PhysicsTrainerV2:
    """
    Improved trainer that avoids trivial solution.
    
    Key changes from V1:
    1. Use ScaledFramePIGNN — prevents collapse to zero
    2. Stronger equilibrium loss — this is the hardest constraint
    3. Normalize losses by PHYSICAL scale not initial model output
    4. Add energy loss — ensures positive strain energy
    """

    def __init__(self,
                 model,
                 data_list: list,
                 lr: float = 1e-4,
                 w_constitutive: float = 1.0,
                 w_equilibrium: float = 10.0,
                 w_bc: float = 50.0,
                 w_sensitivity: float = 0.01,
                 device: str = 'cpu',
                 log_dir: str = 'DATA/training_logs_v2'):

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.data_list = [d.to(self.device) for d in data_list]

        # Higher equilibrium weight — hardest physical constraint
        self.loss_fn = PhysicsLoss(
            w_constitutive=w_constitutive,
            w_equilibrium=w_equilibrium,
            w_bc=w_bc,
            w_sensitivity=w_sensitivity,
        )

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=300,
            min_lr=1e-8)

        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Physics scale factors for loss normalization
        # Computed from physical understanding (not model output)
        self._compute_physics_norms()

        self.history = {k: [] for k in [
            'epoch', 'total', 'constitutive', 'equilibrium',
            'bc', 'sensitivity', 'energy', 'lr',
            'disp_err', 'moment_err', 'force_err', 'sens_err']}

    def _compute_physics_norms(self):
        """
        Compute normalization from physical data — NOT from model output.
        This avoids the trivial solution problem.
        """
        print(f"\n  Computing physics-based normalization...")

        # Compute from actual Kratos outputs (only for normalization)
        all_moments = torch.cat([d.y_element[:, 1] for d in self.data_list])
        all_forces  = torch.cat([d.y_element[:, 3] for d in self.data_list])
        all_disp    = torch.cat([d.y_node[:, 2] for d in self.data_list])

        # RMS values
        rms_M = all_moments.pow(2).mean().sqrt().item()
        rms_F = all_forces.pow(2).mean().sqrt().item()
        rms_u = all_disp.pow(2).mean().sqrt().item()

        # Expected force residual at equilibrium
        # Σf_internal = f_external → residual scale = scale of f_external
        all_loads = torch.cat([d.line_load[:, 2] for d in self.data_list])
        rms_load = all_loads.pow(2).mean().sqrt().item()
        expected_reaction = rms_load * 3.0  # × typical tributary length

        # BC: displacement at supports should be ~0, scale by expected disp
        bc_scale = max(rms_u, 1e-10)

        # Constitutive: K×u - f → scale by typical force magnitude
        constitutive_scale = max(rms_F ** 2, 1.0)

        # Equilibrium: residual scale ≈ (rms_load * L)²
        equil_scale = max((expected_reaction) ** 2, 1.0)

        self.phys_norms = {
            'constitutive': constitutive_scale,
            'equilibrium':  equil_scale,
            'bc':           bc_scale ** 2,
            'sensitivity':  1.0,  # Will update after first pass
        }

        print(f"    rms_moment = {rms_M:.4e}")
        print(f"    rms_force  = {rms_F:.4e}")
        print(f"    rms_disp   = {rms_u:.4e}")
        print(f"    Physics norms:")
        for k, v in self.phys_norms.items():
            print(f"      {k:<15}: {v:.4e}")

    def _energy_loss(self, node_pred: torch.Tensor, data) -> torch.Tensor:
        """
        Strain energy must be positive.
        
        U = 0.5 × u^T × K × u ≥ 0
        
        If GNN predicts wrong-sign displacements, this penalizes it.
        """
        from step_3_ele_stiff_matrx import FramePhysicsXZ
        physics = FramePhysicsXZ()

        result = physics.compute_from_data(node_pred, data)
        u_elem = result['u_global']    # (E, 6)
        f_elem = result['f_global']    # (E, 6)

        # Element strain energy = 0.5 × u^T × f (for each element)
        energy_per_elem = 0.5 * (u_elem * f_elem).sum(dim=1)  # (E,)

        # Penalty: negative strain energy elements
        # Loss = ReLU(-energy)  ← penalize negative energy
        neg_energy = torch.relu(-energy_per_elem)

        return neg_energy.mean()

    def _train_step(self, data) -> Dict[str, float]:
        """One training step."""
        self.model.train()

        self.optimizer.zero_grad()
        node_pred, elem_pred = self.model(data)

        # Physics losses
        L_const = self.loss_fn.constitutive_loss(
            node_pred, elem_pred, data)
        L_equil = self.loss_fn.equilibrium_loss(node_pred, data)
        L_bc    = self.loss_fn.bc_loss(node_pred, data)
        L_sens  = self.loss_fn.sensitivity_loss(
            node_pred, elem_pred, data)
        L_energy = self._energy_loss(node_pred, data)

        # Normalize by physics scales
        L_const_n  = L_const  / self.phys_norms['constitutive']
        L_equil_n  = L_equil  / self.phys_norms['equilibrium']
        L_bc_n     = L_bc     / self.phys_norms['bc']
        L_sens_n   = L_sens   / max(self.phys_norms['sensitivity'], 1e-10)
        L_energy_n = L_energy  # Already normalized

        # Weighted total
        total = (self.loss_fn.w_const * L_const_n +
                 self.loss_fn.w_equil * L_equil_n +
                 self.loss_fn.w_bc    * L_bc_n +
                 self.loss_fn.w_sens  * L_sens_n +
                 1.0 * L_energy_n)

        total.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'total':        total.item(),
            'constitutive': L_const_n.item(),
            'equilibrium':  L_equil_n.item(),
            'bc':           L_bc_n.item(),
            'sensitivity':  L_sens_n.item(),
            'energy':       L_energy_n.item(),
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Compare against Kratos."""
        self.model.eval()
        errors = {k: 0 for k in [
            'disp_err', 'moment_err', 'force_err', 'sens_err']}

        for data in self.data_list:
            node_pred, elem_pred = self.model(data)
            y_node = data.y_node
            y_elem = data.y_element

            def rel_err(pred, ref):
                denom = ref.abs().max()
                if denom < 1e-15:
                    return 0.0
                return ((pred - ref).abs().max() / denom).item()

            errors['disp_err']   += rel_err(node_pred[:, 2], y_node[:, 2])
            errors['moment_err'] += rel_err(elem_pred[:, 1], y_elem[:, 1])
            errors['force_err']  += rel_err(elem_pred[:, 3], y_elem[:, 3])
            errors['sens_err']   += rel_err(elem_pred[:, 6], y_elem[:, 6])

        n = len(self.data_list)
        return {k: v / n for k, v in errors.items()}

    def train(self, n_epochs: int = 10000,
              print_every: int = 200,
              validate_every: int = 1000,
              save_every: int = 2000) -> dict:

        print(f"\n{'═'*60}")
        print(f"  PHYSICS TRAINING V2 (Scaled Output)")
        print(f"{'═'*60}")
        print(f"  Params: {self.model.count_parameters():,}")
        print(f"  Graphs: {len(self.data_list)}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Loss weights:")
        print(f"    Constitutive: {self.loss_fn.w_const}")
        print(f"    Equilibrium:  {self.loss_fn.w_equil}")
        print(f"    BC:           {self.loss_fn.w_bc}")
        print(f"    Sensitivity:  {self.loss_fn.w_sens}")
        print(f"    Energy:       1.0")

        # Initial validation
        val = self.validate()
        print(f"\n  Initial errors vs Kratos:")
        for k, v in val.items():
            print(f"    {k}: {v:.4f}")

        best_loss = float('inf')
        start = time.time()

        for epoch in range(1, n_epochs + 1):

            ep_losses = {k: 0 for k in [
                'total', 'constitutive', 'equilibrium',
                'bc', 'sensitivity', 'energy']}

            for data in self.data_list:
                step = self._train_step(data)
                for k in ep_losses:
                    ep_losses[k] += step[k]

            n = len(self.data_list)
            for k in ep_losses:
                ep_losses[k] /= n

            self.scheduler.step(ep_losses['total'])

            # Record
            self.history['epoch'].append(epoch)
            self.history['total'].append(ep_losses['total'])
            for k in ['constitutive', 'equilibrium', 'bc',
                      'sensitivity', 'energy']:
                self.history[k].append(ep_losses[k])
            self.history['lr'].append(
                self.optimizer.param_groups[0]['lr'])

            # Best model
            if ep_losses['total'] < best_loss:
                best_loss = ep_losses['total']
                torch.save(self.model.state_dict(),
                           f'{self.log_dir}/best_model.pt')

            # Print
            if epoch % print_every == 0 or epoch == 1:
                elapsed = time.time() - start
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Ep {epoch:>6} "
                      f"| L={ep_losses['total']:.3e} "
                      f"| C={ep_losses['constitutive']:.3e} "
                      f"| E={ep_losses['equilibrium']:.3e} "
                      f"| BC={ep_losses['bc']:.3e} "
                      f"| S={ep_losses['sensitivity']:.3e} "
                      f"| En={ep_losses['energy']:.3e} "
                      f"| lr={lr:.1e} "
                      f"| {elapsed:.0f}s")

            # Validate
            if epoch % validate_every == 0:
                val = self.validate()
                for k, v in val.items():
                    self.history[k].append(v)
                print(f"\n  ── Validation epoch {epoch} ──")
                for k, v in val.items():
                    print(f"       {k}: {v:.4f}")
                print()

            # Checkpoint
            if epoch % save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'opt': self.optimizer.state_dict(),
                }, f'{self.log_dir}/checkpoint_{epoch}.pt')

        # Final
        torch.save(self.model.state_dict(),
                   f'{self.log_dir}/final_model.pt')
        val = self.validate()
        print(f"\n  ── Final Validation ──")
        for k, v in val.items():
            print(f"    {k}: {v:.4f}")
        print(f"\n  Best loss: {best_loss:.4e}")
        print(f"  Time: {time.time()-start:.1f}s")

        return self.history


# ================================================================
# MAIN
# ================================================================

import torch.optim as optim

if __name__ == "__main__":

    print("=" * 60)
    print("  STEP 3 REVISED: Scaled Output Training")
    print("=" * 60)

    # Load UN-normalized data
    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    print(f"  Loaded {len(data_list)} graphs")

    # Compute output scales from data
    node_scales, elem_scales = compute_output_scales(data_list)

    # Create base model
    base_model = create_model('medium')

    # Wrap with output scaling
    model = ScaledFramePIGNN(base_model, node_scales, elem_scales)
    print(f"  Scaled model: {model.count_parameters():,} params")

    # Train
    trainer = PhysicsTrainerV2(
        model=model,
        data_list=data_list,
        lr=1e-4,
        w_constitutive=1.0,
        w_equilibrium=10.0,
        w_bc=50.0,
        w_sensitivity=0.01,
        device='cpu',
        log_dir='DATA/training_logs_v2',
    )

    history = trainer.train(
        n_epochs=100,#5000
        print_every=4,#200
        validate_every=20,#1000
        save_every=50,#2000
    )

    print(f"\n{'='*60}")
    print(f"  REVISED TRAINING COMPLETE")
    print(f"  Key changes:")
    print(f"    ✓ ScaledFramePIGNN — outputs anchored to physics scale")
    print(f"    ✓ Physics-based loss normalization (not model output)")
    print(f"    ✓ Energy loss — prevents wrong-sign solutions")
    print(f"    ✓ Higher equilibrium weight (10x)")
    print(f"    ✓ Higher BC weight (50x)")
    print(f"{'='*60}")