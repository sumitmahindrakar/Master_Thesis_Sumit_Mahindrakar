"""
=================================================================
step_3_training_v3.py — FIXED TRAINING
=================================================================
Changes:
  1. Sensitivity DISABLED (was dominating 99.99% of loss)
  2. Subsample 50 graphs per epoch (1000 is too slow)
  3. Phased training (BC → Equilibrium → Constitutive)
  4. Proper loss normalization per-graph
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
import time
from typing import Dict, List
import random

from step_3_model import FramePIGNN, create_model
from step_3_phys_loss import PhysicsLoss
from step_3_ele_stiff_matrx import FramePhysicsXZ


# ================================================================
# SCALED MODEL (same as before)
# ================================================================

class ScaledFramePIGNN(nn.Module):
    def __init__(self, base_model, node_scales, elem_scales):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('node_scales', node_scales.float())
        self.register_buffer('elem_scales', elem_scales.float())

    def forward(self, data):
        node_raw, elem_raw = self.base_model(data)
        return node_raw * self.node_scales, elem_raw * self.elem_scales

    def count_parameters(self):
        return self.base_model.count_parameters()


def compute_output_scales(data_list):
    all_node = torch.cat([d.y_node for d in data_list], dim=0)
    all_elem = torch.cat([d.y_element for d in data_list], dim=0)
    node_scales = all_node.pow(2).mean(dim=0).sqrt()
    elem_scales = all_elem.pow(2).mean(dim=0).sqrt()
    node_scales = torch.where(node_scales < 1e-15,
                               torch.ones_like(node_scales), node_scales)
    elem_scales = torch.where(elem_scales < 1e-15,
                               torch.ones_like(elem_scales), elem_scales)

    print(f"\n  Output scales:")
    for name, s in zip(['ux','uy','uz','rx','ry','rz'], node_scales):
        print(f"    {name}: {s:.4e}")
    for name, s in zip(['Mx','My','Mz','Fx','Fy','Fz','sens'], elem_scales):
        print(f"    {name}: {s:.4e}")
    return node_scales, elem_scales


# ================================================================
# PHASED TRAINER
# ================================================================

class PhasedTrainer:
    """
    Three-phase physics training with subsampling.
    
    Phase 1: BC only (1000 epochs)
    Phase 2: BC + Equilibrium (2000 epochs)
    Phase 3: BC + Equilibrium + Constitutive (3000 epochs)
    
    Each epoch uses a RANDOM SUBSET of graphs (faster).
    """

    def __init__(self, model, data_list,
                 graphs_per_epoch=50, device='cpu',
                 log_dir='DATA/training_logs_v3'):

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.data_list = [d.to(self.device) for d in data_list]
        self.graphs_per_epoch = min(graphs_per_epoch, len(data_list))

        self.physics = FramePhysicsXZ()
        self.loss_fn = PhysicsLoss()

        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.history = {k: [] for k in [
            'epoch', 'phase', 'total', 'bc', 'equilibrium',
            'constitutive', 'energy', 'lr']}

    def _sample_graphs(self):
        """Random subset of graphs for this epoch."""
        indices = random.sample(range(len(self.data_list)),
                                 self.graphs_per_epoch)
        return [self.data_list[i] for i in indices]

    def _bc_loss(self, node_pred, data):
        """Displacement at supports must be zero."""
        bc_mask = data.bc_disp.squeeze(-1) > 0.5
        if not bc_mask.any():
            return torch.tensor(0.0, device=self.device)
        u_support = node_pred[bc_mask]
        return (u_support[:, 0] ** 2 +
                u_support[:, 2] ** 2 +
                u_support[:, 4] ** 2).mean()

    def _equil_loss(self, node_pred, data):
        """Force balance at free nodes."""
        from step_3_equill import EquivalentNodalLoads
        enl = EquivalentNodalLoads()

        N = node_pred.shape[0]
        result = self.physics.compute_from_data(node_pred, data)
        f_global = result['f_global']

        # Assemble internal forces at nodes
        conn = data.connectivity
        f_int = torch.zeros(N, 3, device=self.device)
        f_int.scatter_add_(0, conn[:, 0].unsqueeze(1).expand(-1, 3),
                            f_global[:, 0:3])
        f_int.scatter_add_(0, conn[:, 1].unsqueeze(1).expand(-1, 3),
                            f_global[:, 3:6])

        # External loads
        load_result = enl.compute_from_data(data)
        F_ext = load_result['F_ext']

        # Residual at free nodes
        residual = f_int + F_ext
        free_mask = data.bc_disp.squeeze(-1) < 0.5
        residual_free = residual[free_mask]

        # Normalize by force scale for this graph
        force_scale = (data.prop_E * data.prop_A).mean()
        return (residual_free ** 2).mean() / (force_scale ** 2 + 1e-10)

    def _const_loss(self, node_pred, elem_pred, data):
        """K×u must be consistent with predicted forces."""
        result = self.physics.compute_from_data(node_pred, data)
        f_local = result['f_local']

        N_mid = (f_local[:, 0] + f_local[:, 3]) / 2
        V_mid = (f_local[:, 1] + f_local[:, 4]) / 2
        M_mid = (f_local[:, 2] + f_local[:, 5]) / 2

        Fx_pred = elem_pred[:, 3]
        Fz_pred = elem_pred[:, 5]
        My_pred = elem_pred[:, 1]

        force_scale = (data.prop_E * data.prop_A).mean()
        moment_scale = (data.prop_E * data.prop_I22).mean()

        loss_N = ((N_mid - Fx_pred) ** 2).mean() / (force_scale ** 2 + 1e-10)
        loss_V = ((V_mid - Fz_pred) ** 2).mean() / (force_scale ** 2 + 1e-10)
        loss_M = ((M_mid - My_pred) ** 2).mean() / (moment_scale ** 2 + 1e-10)

        return loss_N + loss_V + loss_M

    def _energy_loss(self, node_pred, data):
        """Strain energy must be positive."""
        result = self.physics.compute_from_data(node_pred, data)
        u = result['u_global']
        f = result['f_global']
        energy = 0.5 * (u * f).sum(dim=1)
        return torch.relu(-energy).mean()

    @torch.no_grad()
    def validate(self, n_samples=100):
        """Compare with Kratos on subset."""
        self.model.eval()
        n = min(n_samples, len(self.data_list))
        indices = random.sample(range(len(self.data_list)), n)
        err = {'disp': 0, 'moment': 0, 'force': 0}

        for i in indices:
            data = self.data_list[i]
            node_pred, elem_pred = self.model(data)

            def rel(p, r):
                d = r.abs().max()
                return ((p-r).abs().max()/d).item() if d > 1e-15 else 0

            err['disp']   += rel(node_pred[:, 2], data.y_node[:, 2])
            err['moment'] += rel(elem_pred[:, 1], data.y_element[:, 1])
            err['force']  += rel(elem_pred[:, 3], data.y_element[:, 3])

        return {k: v/n for k, v in err.items()}

    def _run_phase(self, phase_name, loss_fn, n_epochs, lr,
                    print_every=100, validate_every=500):
        """Run one training phase."""

        print(f"\n{'─'*60}")
        print(f"  PHASE: {phase_name}")
        print(f"  Epochs: {n_epochs}, LR: {lr}, "
              f"Graphs/epoch: {self.graphs_per_epoch}")
        print(f"{'─'*60}")

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01)

        best_loss = float('inf')
        start = time.time()

        for epoch in range(1, n_epochs + 1):
            self.model.train()
            epoch_loss = 0
            batch = self._sample_graphs()

            for data in batch:
                optimizer.zero_grad()
                node_pred, elem_pred = self.model(data)
                loss = loss_fn(node_pred, elem_pred, data)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(batch)
            scheduler.step()

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.model.state_dict(),
                           f'{self.log_dir}/best_{phase_name}.pt')

            self.history['epoch'].append(
                len(self.history['epoch']))
            self.history['phase'].append(phase_name)
            self.history['total'].append(epoch_loss)
            self.history['lr'].append(
                optimizer.param_groups[0]['lr'])

            if epoch % print_every == 0 or epoch == 1:
                lr_now = optimizer.param_groups[0]['lr']
                elapsed = time.time() - start
                print(f"    Ep {epoch:>5}/{n_epochs} "
                      f"| Loss: {epoch_loss:.4e} "
                      f"| Best: {best_loss:.4e} "
                      f"| LR: {lr_now:.1e} "
                      f"| {elapsed:.0f}s")

            if epoch % validate_every == 0:
                val = self.validate()
                print(f"    ── Val: disp={val['disp']:.3f}, "
                      f"moment={val['moment']:.3f}, "
                      f"force={val['force']:.3f}")

        # Load best and validate
        self.model.load_state_dict(
            torch.load(f'{self.log_dir}/best_{phase_name}.pt',
                       weights_only=False))
        val = self.validate()
        print(f"  Phase {phase_name} done. Best loss: {best_loss:.4e}")
        print(f"  Val: disp={val['disp']:.3f}, "
              f"moment={val['moment']:.3f}, "
              f"force={val['force']:.3f}")

        return best_loss

    def train(self):
        """Full three-phase training."""

        print(f"\n{'═'*60}")
        print(f"  PHASED PHYSICS TRAINING")
        print(f"{'═'*60}")
        print(f"  Model:       {self.model.count_parameters():,} params")
        print(f"  Total graphs:{len(self.data_list)}")
        print(f"  Per epoch:   {self.graphs_per_epoch}")

        val = self.validate()
        print(f"\n  Initial: disp={val['disp']:.3f}, "
              f"moment={val['moment']:.3f}, "
              f"force={val['force']:.3f}")

        total_start = time.time()

        # ── PHASE 1: BC ONLY ──
        def phase1(node_pred, elem_pred, data):
            L_bc = self._bc_loss(node_pred, data)
            # Normalize BC by displacement scale
            disp_scale = data.y_node[:, 2].pow(2).mean().sqrt()
            disp_scale = max(disp_scale.item(), 1e-6)
            return L_bc / (disp_scale ** 2)

        self._run_phase("P1_BC", phase1,
                         n_epochs=1000, lr=1e-3,
                         print_every=200, validate_every=500)

        # ── PHASE 2: BC + EQUILIBRIUM ──
        def phase2(node_pred, elem_pred, data):
            L_bc = self._bc_loss(node_pred, data)
            L_eq = self._equil_loss(node_pred, data)
            L_en = self._energy_loss(node_pred, data)

            disp_scale = data.y_node[:, 2].pow(2).mean().sqrt()
            disp_scale = max(disp_scale.item(), 1e-6)

            return 50.0 * L_bc / (disp_scale ** 2) + 10.0 * L_eq + L_en

        self._run_phase("P2_BC_EQ", phase2,
                         n_epochs=3000, lr=5e-4,
                         print_every=500, validate_every=1000)

        # ── PHASE 3: BC + EQUILIBRIUM + CONSTITUTIVE ──
        def phase3(node_pred, elem_pred, data):
            L_bc = self._bc_loss(node_pred, data)
            L_eq = self._equil_loss(node_pred, data)
            L_co = self._const_loss(node_pred, elem_pred, data)
            L_en = self._energy_loss(node_pred, data)

            disp_scale = data.y_node[:, 2].pow(2).mean().sqrt()
            disp_scale = max(disp_scale.item(), 1e-6)

            return (50.0 * L_bc / (disp_scale ** 2) +
                    10.0 * L_eq +
                    1.0 * L_co +
                    L_en)

        self._run_phase("P3_ALL", phase3,
                         n_epochs=6000, lr=1e-4,
                         print_every=500, validate_every=1000)

        # Save final
        torch.save(self.model.state_dict(),
                   f'{self.log_dir}/final_model.pt')

        total_time = time.time() - total_start
        val = self.validate()

        print(f"\n{'═'*60}")
        print(f"  TRAINING COMPLETE")
        print(f"{'═'*60}")
        print(f"  Time: {total_time/60:.1f} minutes")
        print(f"  Final: disp={val['disp']:.4f}, "
              f"moment={val['moment']:.4f}, "
              f"force={val['force']:.4f}")
        print(f"{'═'*60}")

        return val, self.history


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  STEP 3 V3: Phased Training (No Sensitivity)")
    print("=" * 60)

    # Load data
    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    print(f"  Loaded {len(data_list)} graphs")

    # Scales
    node_scales, elem_scales = compute_output_scales(data_list)

    # Model
    base_model = create_model('medium')
    model = ScaledFramePIGNN(base_model, node_scales, elem_scales)

    # Train
    trainer = PhasedTrainer(
        model=model,
        data_list=data_list,
        graphs_per_epoch=50,  # Only 50 random graphs per epoch
        device='cpu',
        log_dir='DATA/training_logs_v3',
    )

    results, history = trainer.train()