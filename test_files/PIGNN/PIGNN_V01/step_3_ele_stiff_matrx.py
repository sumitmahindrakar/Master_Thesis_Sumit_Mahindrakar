"""
=================================================================
STEP 3b: ELEMENT STIFFNESS MATRIX FOR XZ PLANE FRAME
=================================================================
All operations in differentiable PyTorch so gradients flow
through I22 for sensitivity computation.

YOUR FRAME:
  Plane: XZ (Y=0)
  Active DOFs per node: Ux, Uz, θy (3 DOFs)
  Element DOFs: [Ux1, Uz1, θy1, Ux2, Uz2, θy2] (6 DOFs)

  Columns (vertical): cos=0, sin=1
  Beams (horizontal): cos=1, sin=0

EULER-BERNOULLI BEAM LOCAL STIFFNESS (6×6):

  DOF order: [u1, w1, θ1, u2, w2, θ2]
    u = axial (along element)
    w = transverse (perpendicular, in XZ plane)
    θ = rotation about Y

    ┌                                                    ┐
    │  EA/L      0        0     -EA/L     0        0     │
    │   0     12EI/L³   6EI/L²    0    -12EI/L³  6EI/L²  │
    │   0      6EI/L²   4EI/L     0     -6EI/L²  2EI/L   │
    │ -EA/L     0        0      EA/L     0        0      │
    │   0    -12EI/L³  -6EI/L²    0     12EI/L³ -6EI/L²  │
    │   0      6EI/L²   2EI/L     0     -6EI/L²  4EI/L   │
    └                                                    ┘

TRANSFORMATION (local ↔ global):

    ┌            ┐   ┌             ┐   ┌            ┐
    │ u_local    │   │  c   s   0  │   │ Ux_global  │
    │ w_local    │ = │ -s   c   0  │ × │ Uz_global  │
    │ θ_local    │   │  0   0   1  │   │ θy_global  │
    └            ┘   └             ┘   └            ┘

    c = cos(θ) = dx/L    (X component of direction)
    s = sin(θ) = dz/L    (Z component of direction)

    Column (vertical):   c=0, s=1 → u_local=Uz, w_local=-Ux
    Beam (horizontal):   c=1, s=0 → u_local=Ux, w_local=Uz

K_global = T^T × K_local × T
=================================================================
"""
import os

os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\PIGNN\PIGNN_V01")
print(f"Working directory: {os.getcwd()}")

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class FramePhysicsXZ:
    """
    Physics of 2D frame in XZ plane.
    
    All operations are differentiable PyTorch tensors.
    I22 MUST flow through the computation graph so that
    torch.autograd.grad can compute dBM/dI22.
    
    GNN output format: [ux, uy, uz, rx, ry, rz] per node
    Active DOFs:        ux(col 0), uz(col 2), θy(col 4)
    """

    # Indices of active DOFs in GNN's (N, 6) output
    # node_pred[:, 0] = ux
    # node_pred[:, 2] = uz
    # node_pred[:, 4] = θy (ry in rotation)
    ACTIVE_DOF_INDICES = [0, 2, 4]

    # ─────────────────────────────────────────────
    # LOCAL STIFFNESS MATRIX
    # ─────────────────────────────────────────────

    @staticmethod
    def local_stiffness_matrix(E: torch.Tensor,
                                A: torch.Tensor,
                                I22: torch.Tensor,
                                L: torch.Tensor
                                ) -> torch.Tensor:
        """
        Build Euler-Bernoulli beam local stiffness matrix.
        
        IMPORTANT: I22 must have requires_grad=True for
        sensitivity computation via autograd.
        
        Args:
            E:   (n_elem,) Young's modulus
            A:   (n_elem,) cross-sectional area
            I22: (n_elem,) second moment of area
            L:   (n_elem,) element length
        
        Returns:
            K_local: (n_elem, 6, 6)
        """
        n = E.shape[0]

        # Stiffness coefficients (all differentiable w.r.t. I22)
        ea_l = E * A / L              # EA/L
        ei_l3 = E * I22 / L.pow(3)    # EI/L³
        ei_l2 = E * I22 / L.pow(2)    # EI/L²
        ei_l  = E * I22 / L           # EI/L

        # Build matrix
        K = torch.zeros(n, 6, 6, device=E.device, dtype=E.dtype)

        # ── Axial stiffness (row/col 0,3) ──
        K[:, 0, 0] =  ea_l
        K[:, 0, 3] = -ea_l
        K[:, 3, 0] = -ea_l
        K[:, 3, 3] =  ea_l

        # ── Bending stiffness (rows/cols 1,2,4,5) ──
        # Row 1 (w1)
        K[:, 1, 1] =  12.0 * ei_l3
        K[:, 1, 2] =   6.0 * ei_l2
        K[:, 1, 4] = -12.0 * ei_l3
        K[:, 1, 5] =   6.0 * ei_l2

        # Row 2 (θ1)
        K[:, 2, 1] =   6.0 * ei_l2
        K[:, 2, 2] =   4.0 * ei_l
        K[:, 2, 4] =  -6.0 * ei_l2
        K[:, 2, 5] =   2.0 * ei_l

        # Row 4 (w2)
        K[:, 4, 1] = -12.0 * ei_l3
        K[:, 4, 2] =  -6.0 * ei_l2
        K[:, 4, 4] =  12.0 * ei_l3
        K[:, 4, 5] =  -6.0 * ei_l2

        # Row 5 (θ2)
        K[:, 5, 1] =   6.0 * ei_l2
        K[:, 5, 2] =   2.0 * ei_l
        K[:, 5, 4] =  -6.0 * ei_l2
        K[:, 5, 5] =   4.0 * ei_l

        return K

    # ─────────────────────────────────────────────
    # TRANSFORMATION MATRIX
    # ─────────────────────────────────────────────

    @staticmethod
    def transformation_matrix(cos_theta: torch.Tensor,
                               sin_theta: torch.Tensor
                               ) -> torch.Tensor:
        """
        Build local↔global transformation matrix T for XZ plane.
        
        d_local = T × d_global
        f_global = T^T × f_local
        K_global = T^T × K_local × T
        
        Args:
            cos_theta: (n_elem,) dx/L
            sin_theta: (n_elem,) dz/L
        
        Returns:
            T: (n_elem, 6, 6)
        
        Verification:
            Column (c=0, s=1):
              u_local = 0*Ux + 1*Uz = Uz (axial = vertical)
              w_local = -1*Ux + 0*Uz = -Ux (transverse = horizontal)
              ✓ Makes sense: column's axis is vertical
            
            Beam (c=1, s=0):
              u_local = 1*Ux + 0*Uz = Ux (axial = horizontal)
              w_local = 0*Ux + 1*Uz = Uz (transverse = vertical)
              ✓ Makes sense: beam's axis is horizontal
        """
        n = cos_theta.shape[0]
        c = cos_theta
        s = sin_theta

        T = torch.zeros(n, 6, 6, device=c.device, dtype=c.dtype)

        # Node 1 block (rows 0-2, cols 0-2)
        T[:, 0, 0] =  c
        T[:, 0, 1] =  s
        T[:, 1, 0] = -s
        T[:, 1, 1] =  c
        T[:, 2, 2] =  1.0

        # Node 2 block (rows 3-5, cols 3-5)
        T[:, 3, 3] =  c
        T[:, 3, 4] =  s
        T[:, 4, 3] = -s
        T[:, 4, 4] =  c
        T[:, 5, 5] =  1.0

        return T

    # ─────────────────────────────────────────────
    # GLOBAL STIFFNESS MATRIX
    # ─────────────────────────────────────────────

    @staticmethod
    def global_stiffness_matrix(K_local: torch.Tensor,
                                 T: torch.Tensor
                                 ) -> torch.Tensor:
        """
        K_global = T^T × K_local × T
        
        Args:
            K_local: (n_elem, 6, 6)
            T:       (n_elem, 6, 6)
        
        Returns:
            K_global: (n_elem, 6, 6)
        """
        T_t = T.transpose(1, 2)
        return torch.bmm(torch.bmm(T_t, K_local), T)

    # ─────────────────────────────────────────────
    # EXTRACT ELEMENT DOFs FROM NODE PREDICTIONS
    # ─────────────────────────────────────────────

    @staticmethod
    def extract_element_dofs(node_pred: torch.Tensor,
                              connectivity: torch.Tensor
                              ) -> torch.Tensor:
        """
        Extract active DOFs for each element from node predictions.
        
        From GNN output (N, 6) = [ux, uy, uz, rx, ry, rz]
        Extract: [Ux_n1, Uz_n1, θy_n1, Ux_n2, Uz_n2, θy_n2]
                 [col0,  col2,  col4,   col0,  col2,  col4 ]
        
        Args:
            node_pred:    (N, 6) GNN node predictions
            connectivity: (E, 2) element-to-node mapping
        
        Returns:
            u_elem: (E, 6) element DOF vectors in global coords
        """
        n1 = connectivity[:, 0]  # (E,) start node indices
        n2 = connectivity[:, 1]  # (E,) end node indices

        u_elem = torch.stack([
            node_pred[n1, 0],   # Ux at node 1
            node_pred[n1, 2],   # Uz at node 1
            node_pred[n1, 4],   # θy at node 1
            node_pred[n2, 0],   # Ux at node 2
            node_pred[n2, 2],   # Uz at node 2
            node_pred[n2, 4],   # θy at node 2
        ], dim=1)  # (E, 6)

        return u_elem

    # ─────────────────────────────────────────────
    # COMPUTE INTERNAL FORCES FROM DISPLACEMENTS
    # ─────────────────────────────────────────────

    def compute_internal_forces(self,
                                 node_pred: torch.Tensor,
                                 connectivity: torch.Tensor,
                                 E: torch.Tensor,
                                 A: torch.Tensor,
                                 I22: torch.Tensor,
                                 L: torch.Tensor,
                                 cos_theta: torch.Tensor,
                                 sin_theta: torch.Tensor
                                 ) -> dict:
        """
        Compute internal forces from predicted displacements.
        
        Pipeline:
            u_global → (T) → u_local → (K_local) → f_local → (T^T) → f_global
        
        Args:
            node_pred:    (N, 6) predicted displacements/rotations
            connectivity: (E, 2) element-to-node mapping
            E, A, I22, L: (n_elem,) element properties
            cos_theta, sin_theta: (n_elem,) element orientation
        
        Returns:
            dict with:
                'f_global':   (E, 6) global internal forces per element
                              [Fx_n1, Fz_n1, My_n1, Fx_n2, Fz_n2, My_n2]
                'f_local':    (E, 6) local internal forces per element
                              [N1, V1, M1, N2, V2, M2]
                'u_global':   (E, 6) element DOFs in global coords
                'u_local':    (E, 6) element DOFs in local coords
                'K_local':    (E, 6, 6) local stiffness matrices
                'K_global':   (E, 6, 6) global stiffness matrices
                'T':          (E, 6, 6) transformation matrices
        """
        # Build matrices
        K_local = self.local_stiffness_matrix(E, A, I22, L)
        T = self.transformation_matrix(cos_theta, sin_theta)
        K_global = self.global_stiffness_matrix(K_local, T)

        # Extract element DOFs from node predictions
        u_global = self.extract_element_dofs(node_pred, connectivity)

        # Transform to local
        u_local = torch.bmm(
            T, u_global.unsqueeze(2)).squeeze(2)  # (E, 6)

        # Internal forces in local coords: f_local = K_local × u_local
        f_local = torch.bmm(
            K_local, u_local.unsqueeze(2)).squeeze(2)  # (E, 6)

        # Internal forces in global coords: f_global = T^T × f_local
        T_t = T.transpose(1, 2)
        f_global = torch.bmm(
            T_t, f_local.unsqueeze(2)).squeeze(2)  # (E, 6)

        return {
            'f_global':  f_global,
            'f_local':   f_local,
            'u_global':  u_global,
            'u_local':   u_local,
            'K_local':   K_local,
            'K_global':  K_global,
            'T':         T,
        }

    # ─────────────────────────────────────────────
    # CONVENIENCE: from PyG Data object
    # ─────────────────────────────────────────────

    def compute_from_data(self, node_pred: torch.Tensor,
                           data) -> dict:
        """
        Compute internal forces using properties from PyG Data.
        
        Args:
            node_pred: (N, 6) from GNN
            data:      PyG Data with prop_E, prop_A, prop_I22,
                       elem_lengths, elem_directions, connectivity
        
        Returns:
            Same dict as compute_internal_forces()
        """
        dirs = data.elem_directions  # (E, 3)
        cos_theta = dirs[:, 0]       # X component
        sin_theta = dirs[:, 2]       # Z component

        return self.compute_internal_forces(
            node_pred=node_pred,
            connectivity=data.connectivity,
            E=data.prop_E,
            A=data.prop_A,
            I22=data.prop_I22,
            L=data.elem_lengths,
            cos_theta=cos_theta,
            sin_theta=sin_theta,
        )


# ================================================================
# VERIFICATION
# ================================================================

def verify_stiffness_matrix():
    """
    Verify stiffness computation against known analytical results.
    
    Test 1: Horizontal cantilever beam
        Fixed at left, point load P at right (downward = -Z)
        Length L, EI
        
        Exact: tip deflection = PL³/(3EI)
               tip rotation = PL²/(2EI)
               root moment = -PL
               root shear = P
    
    Test 2: Vertical column
        Fixed at bottom, axial load P at top (compression = -Z)
        
        Exact: top displacement = -PL/(EA)
    """
    print(f"\n{'═'*60}")
    print(f"  STIFFNESS MATRIX VERIFICATION")
    print(f"{'═'*60}")

    physics = FramePhysicsXZ()

    # ─── Test 1: Horizontal beam K_global = K_local ───
    print(f"\n  Test 1: Horizontal beam (c=1, s=0)")
    print(f"  K_global should equal K_local (no rotation needed)")

    E_val = torch.tensor([200e9])      # 200 GPa
    A_val = torch.tensor([0.01])       # 0.01 m²
    I_val = torch.tensor([8.33e-6])    # I = bh³/12
    L_val = torch.tensor([3.0])        # 3 m

    K_local = physics.local_stiffness_matrix(E_val, A_val, I_val, L_val)
    T = physics.transformation_matrix(
        torch.tensor([1.0]),   # cos = 1 (horizontal)
        torch.tensor([0.0])    # sin = 0
    )
    K_global = physics.global_stiffness_matrix(K_local, T)

    # For horizontal beam: T = I → K_global = K_local
    diff = (K_global - K_local).abs().max()
    print(f"  |K_global - K_local| = {diff:.2e} "
          f"(should be 0) {'✓' if diff < 1e-10 else '✗'}")

    # Check specific entries
    EA_L = (200e9 * 0.01 / 3.0)
    EI_L3 = (200e9 * 8.33e-6 / 27.0)
    EI_L = (200e9 * 8.33e-6 / 3.0)
    
    print(f"  K[0,0] = EA/L = {K_local[0,0,0].item():.4e} "
          f"(expected {EA_L:.4e}) "
          f"{'✓' if abs(K_local[0,0,0].item() - EA_L) < 1 else '✗'}")
    print(f"  K[1,1] = 12EI/L³ = {K_local[0,1,1].item():.4e} "
          f"(expected {12*EI_L3:.4e}) "
          f"{'✓' if abs(K_local[0,1,1].item() - 12*EI_L3) < 1 else '✗'}")
    print(f"  K[2,2] = 4EI/L = {K_local[0,2,2].item():.4e} "
          f"(expected {4*EI_L:.4e}) "
          f"{'✓' if abs(K_local[0,2,2].item() - 4*EI_L) < 1 else '✗'}")

    # Symmetry check
    sym_diff = (K_local - K_local.transpose(1, 2)).abs().max()
    print(f"  Symmetry: |K - K^T| = {sym_diff:.2e} "
          f"{'✓' if sym_diff < 1e-10 else '✗'}")

    # ─── Test 2: Vertical column (c=0, s=1) ───
    print(f"\n  Test 2: Vertical column (c=0, s=1)")
    print(f"  Axial = global Z direction")

    T_col = physics.transformation_matrix(
        torch.tensor([0.0]),   # cos = 0 (vertical)
        torch.tensor([1.0])    # sin = 1
    )
    K_global_col = physics.global_stiffness_matrix(K_local, T_col)

    # For vertical column: axial is Z → K_global[1,1] should be EA/L
    # (global index 1 = Uz, which maps to local index 0 = u_axial)
    print(f"  K_global[1,1] = {K_global_col[0,1,1].item():.4e} "
          f"(should be EA/L = {EA_L:.4e}) "
          f"{'✓' if abs(K_global_col[0,1,1].item() - EA_L) < 1 else '✗'}")

    # K_global[0,0] should be 12EI/L³ (bending stiffness in X)
    print(f"  K_global[0,0] = {K_global_col[0,0,0].item():.4e} "
          f"(should be 12EI/L³ = {12*EI_L3:.4e}) "
          f"{'✓' if abs(K_global_col[0,0,0].item() - 12*EI_L3) < 1 else '✗'}")

    # ─── Test 3: Cantilever tip deflection ───
    print(f"\n  Test 3: Cantilever beam — tip deflection")
    print(f"  Fixed at left (node 0), point load P at right (node 1)")

    P = 1000.0  # N downward
    L = 3.0
    E_v = 200e9
    I_v = 8.33e-6

    # Analytical solutions
    delta_exact = P * L**3 / (3 * E_v * I_v)
    theta_exact = P * L**2 / (2 * E_v * I_v)
    M_root_exact = -P * L

    print(f"  Analytical:")
    print(f"    Tip deflection: {delta_exact:.6e} m")
    print(f"    Tip rotation:   {theta_exact:.6e} rad")
    print(f"    Root moment:    {M_root_exact:.4e} N·m")

    # Solve using K: K_reduced × u_free = F_free
    K_g = K_global[0]  # (6, 6) for single horizontal element

    # Fixed at node 0 (DOFs 0,1,2 = Ux1, Uz1, θy1)
    # Free at node 1 (DOFs 3,4,5 = Ux2, Uz2, θy2)
    K_ff = K_g[3:6, 3:6]  # Free-free partition
    F_free = torch.tensor([0.0, -P, 0.0])  # [Fx2=0, Fz2=-P, My2=0]

    u_free = torch.linalg.solve(K_ff, F_free)
    uz_tip = u_free[1].item()
    ry_tip = u_free[2].item()

    print(f"  Computed from K:")
    print(f"    Tip deflection: {uz_tip:.6e} m "
          f"(error: {abs(uz_tip + delta_exact)/delta_exact*100:.4f}%)")
    print(f"    Tip rotation:   {ry_tip:.6e} rad "
          f"(error: {abs(ry_tip - theta_exact)/theta_exact*100:.4f}%)")

    # Check internal forces
    u_full = torch.tensor([0, 0, 0, u_free[0], u_free[1], u_free[2]])
    f_int = K_g @ u_full
    print(f"    Root Fz:     {f_int[1].item():.4e} (should be {P:.4e})")
    print(f"    Root My:     {f_int[2].item():.4e} (should be {M_root_exact:.4e})")

    # ─── Test 4: Gradient check (dM/dI22) ───
    print(f"\n  Test 4: Sensitivity dM/dI22 via autograd")

    I22_var = torch.tensor([I_v], requires_grad=True)
    K_local_v = physics.local_stiffness_matrix(
        torch.tensor([E_v]), torch.tensor([0.01]), I22_var, torch.tensor([L]))
    T_v = physics.transformation_matrix(
        torch.tensor([1.0]), torch.tensor([0.0]))
    K_g_v = physics.global_stiffness_matrix(K_local_v, T_v)

    K_ff_v = K_g_v[0, 3:6, 3:6]
    F_free_v = torch.tensor([0.0, -P, 0.0])
    u_free_v = torch.linalg.solve(K_ff_v, F_free_v)

    # Root moment = K[2,:] @ u_full
    u_full_v = torch.cat([torch.zeros(3), u_free_v])
    f_int_v = K_g_v[0] @ u_full_v
    M_root = f_int_v[2]  # Root bending moment

    # Compute dM/dI22
    dM_dI = torch.autograd.grad(M_root, I22_var, retain_graph=True)[0]

    # Analytical: M = -PL (constant, independent of I22!)
    # So dM/dI22 = 0 for moment at root of cantilever
    print(f"    M_root = {M_root.item():.4e}")
    print(f"    dM_root/dI22 = {dM_dI.item():.4e} "
          f"(should be ≈0 for fixed-load problem)")

    # For deflection: δ = PL³/(3EI) → dδ/dI = -PL³/(3EI²)
    delta_tip = u_free_v[1]
    d_delta_dI = torch.autograd.grad(
        delta_tip, I22_var, retain_graph=True)[0]
    d_delta_exact = -P * L**3 / (3 * E_v * I_v**2)
    print(f"    dδ/dI22 = {d_delta_dI.item():.4e} "
          f"(exact: {d_delta_exact:.4e}) "
          f"{'✓' if abs(d_delta_dI.item() - d_delta_exact)/abs(d_delta_exact) < 0.01 else '✗'}")

    print(f"\n{'═'*60}")
    print(f"  VERIFICATION COMPLETE")
    print(f"{'═'*60}\n")


# ================================================================
# QUICK TEST WITH YOUR FRAME DATA
# ================================================================

def test_with_frame_data():
    """
    Test stiffness computation using your Kratos frame data.
    """
    print(f"\n{'═'*60}")
    print(f"  TEST WITH YOUR FRAME DATA")
    print(f"{'═'*60}")

    # Load your graph
    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    data = data_list[0]
    print(f"  Case: {data.num_nodes} nodes, {data.n_elements} elements")

    physics = FramePhysicsXZ()

    # Use Kratos displacement as "node_pred" to check forces
    # Build fake node_pred from actual displacement + rotation
    node_pred = torch.zeros(data.num_nodes, 6)
    node_pred[:, 0:3] = data.y_node[:, 0:3]  # displacement
    node_pred[:, 3:6] = data.y_node[:, 3:6]  # rotation

    print(f"\n  Using Kratos displacements as input to K×u:")
    print(f"    Max |ux|: {node_pred[:, 0].abs().max():.4e}")
    print(f"    Max |uz|: {node_pred[:, 2].abs().max():.4e}")
    print(f"    Max |θy|: {node_pred[:, 4].abs().max():.4e}")

    # Compute internal forces
    result = physics.compute_from_data(node_pred, data)

    f_local = result['f_local']    # (E, 6) [N1, V1, M1, N2, V2, M2]
    f_global = result['f_global']  # (E, 6) [Fx1, Fz1, My1, Fx2, Fz2, My2]

    print(f"\n  Internal forces (local) — first 3 elements:")
    print(f"  {'Elem':<5} {'N1':>12} {'V1':>12} {'M1':>12} "
          f"{'N2':>12} {'V2':>12} {'M2':>12}")
    for e in range(3):
        f = f_local[e]
        print(f"  {e:<5} {f[0]:>12.2f} {f[1]:>12.2f} {f[2]:>12.2f} "
              f"{f[3]:>12.2f} {f[4]:>12.2f} {f[5]:>12.2f}")

    # Compare with Kratos FORCE/MOMENT (cell data)
    kratos_force = data.y_element[:, 3:6]   # [Fx, Fy, Fz]
    kratos_moment = data.y_element[:, 0:3]  # [Mx, My, Mz]

    print(f"\n  Kratos FORCE (local coords) — first 3 elements:")
    print(f"  {'Elem':<5} {'Fx':>12} {'Fy':>12} {'Fz':>12}")
    for e in range(3):
        f = kratos_force[e]
        print(f"  {e:<5} {f[0]:>12.2f} {f[1]:>12.2f} {f[2]:>12.2f}")

    print(f"\n  Kratos MOMENT (local coords) — first 3 elements:")
    print(f"  {'Elem':<5} {'Mx':>12} {'My':>12} {'Mz':>12}")
    for e in range(3):
        m = kratos_moment[e]
        print(f"  {e:<5} {m[0]:>12.2f} {m[1]:>12.2f} {m[2]:>12.2f}")

    # Check equilibrium at a free node
    print(f"\n  Equilibrium check at free nodes:")
    E_count = data.n_elements
    conn = data.connectivity

    for node in [1, 5, 10]:  # Sample free nodes
        total_f = torch.zeros(3)  # [Fx, Fz, My]
        connected_elems = []

        for e in range(E_count):
            n1, n2 = conn[e]
            if n1 == node:
                # This node is node 1 of element e
                total_f[0] += f_global[e, 0]  # Fx
                total_f[1] += f_global[e, 1]  # Fz
                total_f[2] += f_global[e, 2]  # My
                connected_elems.append(f"e{e}(n1)")
            elif n2 == node:
                # This node is node 2 of element e
                total_f[0] += f_global[e, 3]  # Fx
                total_f[1] += f_global[e, 4]  # Fz
                total_f[2] += f_global[e, 5]  # My
                connected_elems.append(f"e{e}(n2)")

        # External load at this node
        ext_load = data.line_load[node]  # [wl_x, wl_y, wl_z]

        print(f"    Node {node}: connected to {connected_elems}")
        print(f"      Σ internal: Fx={total_f[0]:.2f}, "
              f"Fz={total_f[1]:.2f}, My={total_f[2]:.2f}")
        print(f"      External:   wl_x={ext_load[0]:.2f}, "
              f"wl_z={ext_load[2]:.2f}")
        print(f"      (Note: LINE_LOAD needs conversion to "
              f"equivalent nodal loads — done in Step 3c)")

    print(f"\n{'═'*60}\n")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  STEP 3b: Element Stiffness Matrix")
    print("=" * 60)

    # Run verification
    verify_stiffness_matrix()

    # Test with your data (if available)
    if os.path.exists("DATA/graph_dataset.pt"):
        test_with_frame_data()
    else:
        print("  DATA/graph_dataset.pt not found — skip frame test")
        print("  Run Step 2 first")

    print(f"{'='*60}")
    print(f"  STEP 3b COMPLETE ✓")
    print(f"")
    print(f"  FramePhysicsXZ provides:")
    print(f"    local_stiffness_matrix()   → (E, 6, 6)")
    print(f"    transformation_matrix()    → (E, 6, 6)")
    print(f"    global_stiffness_matrix()  → (E, 6, 6)")
    print(f"    extract_element_dofs()     → (E, 6)")
    print(f"    compute_internal_forces()  → f_global, f_local")
    print(f"    compute_from_data()        → all above from PyG Data")
    print(f"")
    print(f"  Key: I22 flows through computation graph")
    print(f"       → autograd can compute dBM/dI22")
    print(f"")
    print(f"  Ready for Step 3c (Equivalent Nodal Loads)")
    print(f"{'='*60}")