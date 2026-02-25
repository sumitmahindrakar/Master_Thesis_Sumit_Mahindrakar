"""
=================================================================
STEP 3d: PHYSICS LOSS FUNCTIONS FOR PIGNN
=================================================================
NO DATA USED FOR TRAINING — only physics equations!

LOSS COMPONENTS:
  L1: CONSTITUTIVE   — f_internal = K × u_predicted
  L2: EQUILIBRIUM    — Σ f_internal = f_external at free nodes
  L3: BOUNDARY COND  — u = 0 at supports
  L4: SENSITIVITY    — dBM/dI22 via autograd through K(I22)

TOTAL LOSS:
  L = w1*L1 + w2*L2 + w3*L3 + w4*L4

All losses are computed from GNN PREDICTIONS + KNOWN INPUTS.
Kratos output is NEVER used in training.

This file brings together:
  - FramePhysicsXZ       (from step_3b)
  - EquivalentNodalLoads (from step_3c)
  - FramePIGNN model     (from step_3)
=================================================================
"""
import os

os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\PIGNN\PIGNN_V01")
print(f"Working directory: {os.getcwd()}")

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Tuple, Optional

# Import from previous steps
from step_3_ele_stiff_matrx import FramePhysicsXZ
from step_3_equill import EquivalentNodalLoads


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss for structural frame analysis.
    
    All losses computed from:
      - GNN predictions (displacement, rotation, forces, moments)
      - Known inputs (geometry, BCs, loads, material properties)
      - Physics equations (stiffness relation, equilibrium, BCs)
    
    NO Kratos output data is used.
    """

    def __init__(self,
                 w_constitutive: float = 1.0,
                 w_equilibrium: float = 1.0,
                 w_bc: float = 10.0,
                 w_sensitivity: float = 0.1):
        """
        Args:
            w_constitutive: weight for constitutive law loss
            w_equilibrium:  weight for equilibrium loss
            w_bc:           weight for boundary condition loss
                           (higher = stronger BC enforcement)
            w_sensitivity:  weight for sensitivity loss
        """
        super().__init__()

        self.w_const = w_constitutive
        self.w_equil = w_equilibrium
        self.w_bc = w_bc
        self.w_sens = w_sensitivity

        self.physics = FramePhysicsXZ()
        self.loads = EquivalentNodalLoads()

    # ─────────────────────────────────────────────
    # LOSS 1: CONSTITUTIVE LAW
    # ─────────────────────────────────────────────

    def constitutive_loss(self,
                           node_pred: torch.Tensor,
                           elem_pred: torch.Tensor,
                           data) -> torch.Tensor:
        """
        The GNN predicts BOTH displacements and forces.
        They must be consistent through the stiffness matrix:
        
            f_from_stiffness = K × u_predicted
            f_from_gnn       = elem_pred (GNN's force/moment output)
            
            Loss = |f_from_stiffness - f_from_gnn|²
        
        This ensures the GNN learns the constitutive relationship
        between displacements and internal forces.
        
        Args:
            node_pred: (N, 6) [ux, uy, uz, rx, ry, rz]
            elem_pred: (E, 7) [Mx, My, Mz, Fx, Fy, Fz, dBM/dI22]
            data:      PyG Data with structural properties
        
        Returns:
            scalar loss
        """
        # Compute f = K × u from predicted displacements
        result = self.physics.compute_from_data(node_pred, data)
        f_local_from_K = result['f_local']  # (E, 6) [N1,V1,M1,N2,V2,M2]

        # Extract GNN's predicted forces in local coords
        # elem_pred columns: [Mx, My, Mz, Fx, Fy, Fz, sens]
        # We need to compare with local forces
        # 
        # GNN predicts in a format we need to match to local:
        #   Kratos FORCE = [Fx, Fy, Fz] ≈ [axial, 0, shear] in local
        #   Kratos MOMENT = [Mx, My, Mz] ≈ [0, bending, 0] in local
        #
        # Local stiffness gives: [N1, V1, M1, N2, V2, M2]
        #   N = axial force
        #   V = shear force  
        #   M = bending moment
        #
        # For XZ plane frame (Y=0):
        #   Kratos Fx → local N (axial)
        #   Kratos Fz → local V (shear)
        #   Kratos My → local M (bending)
        #
        # So we compare:
        #   f_local_from_K[N1, V1, M1] with [Fx_pred, Fz_pred, My_pred]
        #
        # But elem_pred is per-element (averaged over element),
        # while f_local_from_K has values at BOTH ends.
        # Use midpoint average: (end1 + end2) / 2 for forces
        # For moment: compare both ends

        # Average forces at element midpoint
        N_mid = (f_local_from_K[:, 0] + f_local_from_K[:, 3]) / 2  # axial
        V_mid = (f_local_from_K[:, 1] + f_local_from_K[:, 4]) / 2  # shear

        # Moments at midpoint (average of end moments)
        M_mid = (f_local_from_K[:, 2] + f_local_from_K[:, 5]) / 2  # bending

        # GNN predictions (element level)
        Fx_pred = elem_pred[:, 3]   # axial ≈ N
        Fz_pred = elem_pred[:, 5]   # shear ≈ V
        My_pred = elem_pred[:, 1]   # bending moment

        # Loss: MSE between stiffness-derived and GNN-predicted
        loss_N = ((N_mid - Fx_pred) ** 2).mean()
        loss_V = ((V_mid - Fz_pred) ** 2).mean()
        loss_M = ((M_mid - My_pred) ** 2).mean()

        return loss_N + loss_V + loss_M

    # ─────────────────────────────────────────────
    # LOSS 2: EQUILIBRIUM AT FREE NODES
    # ─────────────────────────────────────────────

    def equilibrium_loss(self,
                          node_pred: torch.Tensor,
                          data) -> torch.Tensor:
        """
        At every FREE node, the sum of internal forces from all
        connected elements must equal the external applied load.
        
            Σ f_internal_from_elements + f_external = 0
            
            or equivalently:
            
            Σ f_internal = -f_external (= F_ext from equiv nodal loads)
        
        Sign convention:
            f_internal points INTO the element
            f_external points INTO the node (applied load)
            At equilibrium they balance
        
        Args:
            node_pred: (N, 6) predicted displacements
            data:      PyG Data
        
        Returns:
            scalar loss (MSE of residual at free nodes)
        """
        N = node_pred.shape[0]

        # 1. Compute internal forces from predicted displacements
        result = self.physics.compute_from_data(node_pred, data)
        f_global = result['f_global']  # (E, 6) [Fx1,Fz1,My1,Fx2,Fz2,My2]

        # 2. Assemble internal forces at each node
        #    Sum contributions from all connected elements
        conn = data.connectivity
        n1 = conn[:, 0]
        n2 = conn[:, 1]

        f_internal = torch.zeros(N, 3, device=node_pred.device,
                                  dtype=node_pred.dtype)

        # Node 1 contributions: [Fx1, Fz1, My1]
        f_internal.scatter_add_(
            0, n1.unsqueeze(1).expand(-1, 3),
            f_global[:, 0:3])

        # Node 2 contributions: [Fx2, Fz2, My2]
        f_internal.scatter_add_(
            0, n2.unsqueeze(1).expand(-1, 3),
            f_global[:, 3:6])

        # 3. Compute external forces (equivalent nodal loads)
        load_result = self.loads.compute_from_data(data)
        F_ext = load_result['F_ext']  # (N, 3) [Fx, Fz, My]

        # 4. Equilibrium residual at each node
        #    f_internal + F_ext = 0  (internal forces balance external)
        residual = f_internal + F_ext  # (N, 3)

        # 5. Apply ONLY at free nodes (not at supports)
        #    At supports, reactions are unknown, so equilibrium
        #    is automatically satisfied by the support forces
        bc_disp = data.bc_disp  # (N, 1) — 1 at supports
        free_mask = (bc_disp.squeeze(-1) < 0.5)  # True at free nodes

        # Mask: only free nodes contribute to loss
        residual_free = residual[free_mask]  # (N_free, 3)

        # MSE of residual
        loss = (residual_free ** 2).mean()

        return loss

    # ─────────────────────────────────────────────
    # LOSS 3: BOUNDARY CONDITIONS
    # ─────────────────────────────────────────────

    def bc_loss(self, node_pred: torch.Tensor,
                data) -> torch.Tensor:
        """
        At support nodes, all active DOFs must be zero.
        
        For your fixed supports (nodes 0, 19):
            ux = 0
            uz = 0
            θy = 0
        
        Args:
            node_pred: (N, 6) [ux, uy, uz, rx, ry, rz]
            data:      PyG Data with bc_disp
        
        Returns:
            scalar loss
        """
        bc_disp = data.bc_disp  # (N, 1) — 1 at supports
        support_mask = (bc_disp.squeeze(-1) > 0.5)  # True at supports

        if not support_mask.any():
            return torch.tensor(0.0, device=node_pred.device)

        # Active DOFs: ux (col 0), uz (col 2), θy (col 4)
        u_at_supports = node_pred[support_mask]  # (S, 6)

        # All DOFs should be zero at supports
        # Active DOFs only:
        ux = u_at_supports[:, 0]
        uz = u_at_supports[:, 2]
        ry = u_at_supports[:, 4]  # θy stored as ry

        loss = (ux ** 2).mean() + (uz ** 2).mean() + (ry ** 2).mean()

        return loss

    # ─────────────────────────────────────────────
    # LOSS 4: SENSITIVITY dBM/dI22
    # ─────────────────────────────────────────────

    def sensitivity_loss(self,
                          node_pred: torch.Tensor,
                          elem_pred: torch.Tensor,
                          data) -> torch.Tensor:
        """
        Sensitivity of bending moment at response node w.r.t. I22.
        Uses autograd through K(I22).
        """
        # Make I22 a leaf tensor with gradient tracking
        I22_values = data.prop_I22.detach().clone().requires_grad_(True)

        dirs = data.elem_directions
        cos_theta = dirs[:, 0]
        sin_theta = dirs[:, 2]

        # Use DETACHED node_pred — we only want dM/dI22
        node_pred_detached = node_pred.detach()

        # Compute internal forces with gradient-tracked I22
        result = self.physics.compute_internal_forces(
            node_pred=node_pred_detached,
            connectivity=data.connectivity,
            E=data.prop_E.detach(),
            A=data.prop_A.detach(),
            I22=I22_values,
            L=data.elem_lengths.detach(),
            cos_theta=cos_theta.detach(),
            sin_theta=sin_theta.detach(),
        )

        f_local = result['f_local']  # (E, 6) [N1,V1,M1,N2,V2,M2]

        # Bending moment at element midpoints
        M_mid = (f_local[:, 2] + f_local[:, 5]) / 2  # (E,)

        # Find response node
        resp_flag = data.response_node_flag  # (N, 1)
        resp_node_idx = torch.where(
            resp_flag.squeeze(-1) > 0.5)[0]

        if len(resp_node_idx) == 0:
            return torch.tensor(0.0, device=node_pred.device)

        resp_node = resp_node_idx[0].item()

        # Find elements connected to response node
        conn = data.connectivity
        resp_elem_mask = ((conn[:, 0] == resp_node) |
                           (conn[:, 1] == resp_node))

        if not resp_elem_mask.any():
            return torch.tensor(0.0, device=node_pred.device)

        # BM at response location
        M_response = M_mid[resp_elem_mask].mean()

        # Compute dM/dI22 via autograd
        try:
            dM_dI22 = torch.autograd.grad(
                M_response,
                I22_values,
                create_graph=True,
                retain_graph=True,
            )[0]  # (E,)
        except RuntimeError:
            return torch.tensor(0.0, device=node_pred.device)

        # GNN's predicted sensitivity
        sens_pred = elem_pred[:, 6]  # (E,)

        # Loss: MSE
        loss = ((dM_dI22 - sens_pred) ** 2).mean()

        return loss

    # ─────────────────────────────────────────────
    # TOTAL LOSS
    # ─────────────────────────────────────────────

    def forward(self,
                node_pred: torch.Tensor,
                elem_pred: torch.Tensor,
                data,
                return_components: bool = False
                ) -> torch.Tensor:
        """
        Compute total physics-informed loss.
        
        NO DATA TARGETS USED — only physics equations.
        
        Args:
            node_pred:         (N, 6) GNN predicted displacements
            elem_pred:         (E, 7) GNN predicted forces/moments
            data:              PyG Data with inputs
            return_components: if True, return dict of individual losses
        
        Returns:
            total loss (scalar)
            OR dict with individual losses if return_components=True
        """
        # L1: Constitutive law
        L_const = self.constitutive_loss(node_pred, elem_pred, data)

        # L2: Equilibrium at free nodes
        L_equil = self.equilibrium_loss(node_pred, data)

        # L3: Boundary conditions
        L_bc = self.bc_loss(node_pred, data)

        # L4: Sensitivity (optional — can be expensive)
        L_sens = self.sensitivity_loss(node_pred, elem_pred, data)

        # Weighted sum
        total = (self.w_const * L_const +
                 self.w_equil * L_equil +
                 self.w_bc * L_bc +
                 self.w_sens * L_sens)

        if return_components:
            return {
                'total':        total,
                'constitutive': L_const,
                'equilibrium':  L_equil,
                'bc':           L_bc,
                'sensitivity':  L_sens,
                'w_const':      self.w_const,
                'w_equil':      self.w_equil,
                'w_bc':         self.w_bc,
                'w_sens':       self.w_sens,
            }

        return total


# ================================================================
# VERIFICATION
# ================================================================

def verify_physics_loss():
    """
    Test physics loss with your frame data.
    
    Uses:
      - Random GNN predictions (should give HIGH loss)
      - Kratos displacements as predictions (should give LOW loss)
    """
    print(f"\n{'═'*60}")
    print(f"  PHYSICS LOSS VERIFICATION")
    print(f"{'═'*60}")

    # Load graph
    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    data = data_list[0]
    N = data.num_nodes
    E = data.n_elements

    print(f"  Graph: {N} nodes, {E} elements")

    loss_fn = PhysicsLoss(
        w_constitutive=1.0,
        w_equilibrium=1.0,
        w_bc=10.0,
        w_sensitivity=0.1,
    )

    # ─── Test 1: Random predictions (should give HIGH loss) ───
    print(f"\n  Test 1: Random predictions")
    node_pred_rand = torch.randn(N, 6) * 0.001
    elem_pred_rand = torch.randn(E, 7) * 100.0

    losses_rand = loss_fn(node_pred_rand, elem_pred_rand, data,
                           return_components=True)

    print(f"    Constitutive: {losses_rand['constitutive']:.4e}")
    print(f"    Equilibrium:  {losses_rand['equilibrium']:.4e}")
    print(f"    BC:           {losses_rand['bc']:.4e}")
    print(f"    Sensitivity:  {losses_rand['sensitivity']:.4e}")
    print(f"    TOTAL:        {losses_rand['total']:.4e}")

    # ─── Test 2: Kratos solution (should give LOWER loss) ───
    print(f"\n  Test 2: Kratos solution as prediction")
    node_pred_kratos = data.y_node.clone()  # Kratos displacement+rotation
    elem_pred_kratos = data.y_element.clone()  # Kratos forces+moments

    losses_kratos = loss_fn(node_pred_kratos, elem_pred_kratos, data,
                             return_components=True)

    print(f"    Constitutive: {losses_kratos['constitutive']:.4e}")
    print(f"    Equilibrium:  {losses_kratos['equilibrium']:.4e}")
    print(f"    BC:           {losses_kratos['bc']:.4e}")
    print(f"    Sensitivity:  {losses_kratos['sensitivity']:.4e}")
    print(f"    TOTAL:        {losses_kratos['total']:.4e}")

    # ─── Test 3: Zero predictions (all zeros) ───
    print(f"\n  Test 3: Zero predictions")
    node_pred_zero = torch.zeros(N, 6)
    elem_pred_zero = torch.zeros(E, 7)

    losses_zero = loss_fn(node_pred_zero, elem_pred_zero, data,
                           return_components=True)

    print(f"    Constitutive: {losses_zero['constitutive']:.4e}")
    print(f"    Equilibrium:  {losses_zero['equilibrium']:.4e}")
    print(f"    BC:           {losses_zero['bc']:.4e}")
    print(f"    Sensitivity:  {losses_zero['sensitivity']:.4e}")
    print(f"    TOTAL:        {losses_zero['total']:.4e}")

    # ─── Comparison ───
    print(f"\n  Comparison:")
    print(f"  {'Prediction':<20} {'Total Loss':>15}")
    print(f"  {'-'*35}")
    print(f"  {'Random':<20} {losses_rand['total']:>15.4e}")
    print(f"  {'Kratos solution':<20} {losses_kratos['total']:>15.4e}")
    print(f"  {'Zeros':<20} {losses_zero['total']:>15.4e}")
    print(f"\n  Kratos solution should have LOWEST constitutive")
    print(f"  and equilibrium loss (but not exactly zero due to")
    print(f"  numerical precision and load conversion)")

    # # ─── Test 4: Gradient check ───
    # print(f"\n  Test 4: Gradient flows through loss")
    # node_pred_test = torch.randn(N, 6, requires_grad=True) * 0.001
    # elem_pred_test = torch.randn(E, 7, requires_grad=True) * 100.0

    # total_loss = loss_fn(node_pred_test, elem_pred_test, data)
    # total_loss.backward()

    # node_grad = node_pred_test.grad
    # elem_grad = elem_pred_test.grad

    # print(f"    node_pred grad: shape={node_grad.shape}, "
    #       f"|grad|={node_grad.abs().mean():.4e}")
    # print(f"    elem_pred grad: shape={elem_grad.shape}, "
    #       f"|grad|={elem_grad.abs().mean():.4e}")
    # print(f"    Gradients exist: "
    #       f"{'✓' if node_grad.abs().sum() > 0 else '✗'}")

    # print(f"\n{'═'*60}\n")
    # ─── Test 4: Gradient check ───
    print(f"\n  Test 4: Gradient flows through loss")
    node_pred_test = torch.randn(N, 6) * 0.001
    elem_pred_test = torch.randn(E, 7) * 100.0
    
    # Make them leaf tensors with grad
    node_pred_test = node_pred_test.detach().requires_grad_(True)
    elem_pred_test = elem_pred_test.detach().requires_grad_(True)

    total_loss = loss_fn(node_pred_test, elem_pred_test, data)
    total_loss.backward()

    node_grad = node_pred_test.grad
    elem_grad = elem_pred_test.grad

    has_node_grad = node_grad is not None and node_grad.abs().sum() > 0
    has_elem_grad = elem_grad is not None and elem_grad.abs().sum() > 0

    if has_node_grad:
        print(f"    node_pred grad: shape={node_grad.shape}, "
              f"|grad|={node_grad.abs().mean():.4e} ✓")
    else:
        print(f"    node_pred grad: None ✗")

    if has_elem_grad:
        print(f"    elem_pred grad: shape={elem_grad.shape}, "
              f"|grad|={elem_grad.abs().mean():.4e} ✓")
    else:
        print(f"    elem_pred grad: None ✗")

    print(f"    Gradients flow: "
          f"{'✓' if has_node_grad and has_elem_grad else '✗'}")



# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  STEP 3d: Physics Loss Functions")
    print("=" * 60)

    if os.path.exists("DATA/graph_dataset.pt"):
        verify_physics_loss()
    else:
        print("  Run Step 2 first to create DATA/graph_dataset.pt")

    print(f"{'='*60}")
    print(f"  STEP 3d COMPLETE ✓")
    print(f"")
    print(f"  PhysicsLoss provides:")
    print(f"    constitutive_loss() — K×u must equal predicted forces")
    print(f"    equilibrium_loss()  — Σf_internal = f_external")
    print(f"    bc_loss()           — u = 0 at supports")
    print(f"    sensitivity_loss()  — dBM/dI22 via autograd")
    print(f"    forward()           — weighted total loss")
    print(f"")
    print(f"  NO DATA TARGETS USED — pure physics!")
    print(f"")
    print(f"  Ready for Step 3e (Training Loop)")
    print(f"{'='*60}")