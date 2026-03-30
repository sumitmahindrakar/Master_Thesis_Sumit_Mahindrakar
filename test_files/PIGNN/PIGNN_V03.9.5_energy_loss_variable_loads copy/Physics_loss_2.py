"""
=================================================================
physics_loss.py — Naive Autograd Physics Loss (6 Terms, Per-Node)
=================================================================

PURPOSE:
  Compute physics-based loss for 2D frame structures using ONLY
  the governing equations of Euler-Bernoulli beam theory.
  No labelled data required.

  This is the NAIVE version using autograd through the GNN.
  We expect problems with L_M and L_V (see Section C).

ORGANISATION:
  Section A — Helpers
    A1. _extract_predictions()    → split 15 values
    A2. (removed — no longer needed)
    A3. (removed — merged into per-node constitutive)

  Section B — Concept 1: Equilibrium (NO autograd)
    B1. _loss_equilibrium()       → L_eq
    B2. _loss_free_face()         → L_free
    B3. _loss_support()           → L_sup

  Section C — Concept 2+3: Constitutive (AUTOGRAD, per-node)
    C0. _compute_autograd_derivs()       → spatial derivatives via autograd
    C1. _loss_constitutive_per_node()    → L_N, L_M, L_V (unified)

  Section D — Total
    forward()                     → weighted sum

SIGN CONVENTION (from document Section 2.2):
  Element end forces (element → node):
    At A (x=0): Fx_A^loc = -N,  Fz_A^loc = -V,  My_A^loc = -M
    At B (x=L): Fx_B^loc = +N,  Fz_B^loc = +V,  My_B^loc = +M

  Unified sign rule:
    A-end (sign=+1): ff_loc + N = 0,  ff_loc + V = 0,  ff_loc + M = 0
    B-end (sign=-1): ff_loc - N = 0,  ff_loc - V = 0,  ff_loc - M = 0
=================================================================
"""

import torch
import torch.nn as nn


class NaivePhysicsLoss(nn.Module):
    """
    Naive autograd-based physics loss for 2D frames.

    6 loss terms enforcing Euler-Bernoulli beam theory:
      L_eq   — nodal equilibrium (global)
      L_free — unconnected face forces = 0
      L_sup  — support displacements = 0
      L_N    — axial constitutive (1st derivative)
      L_M    — moment constitutive (2nd derivative)
      L_V    — shear constitutive (3rd derivative)

    Args:
        w_eq:   weight for equilibrium loss
        w_free: weight for free face loss
        w_sup:  weight for support loss
        w_N:    weight for axial constitutive loss
        w_M:    weight for moment constitutive loss
        w_V:    weight for shear constitutive loss
    """

    def __init__(self,
                 w_eq=1.0,
                 w_free=1.0,
                 w_sup=1.0,
                 w_N=1.0,
                 w_M=1.0,
                 w_V=1.0):
        super().__init__()
        self.w_eq   = w_eq
        self.w_free = w_free
        self.w_sup  = w_sup
        self.w_N    = w_N
        self.w_M    = w_M
        self.w_V    = w_V

    # ════════════════════════════════════════════════════════
    # SECTION A: HELPERS
    # ════════════════════════════════════════════════════════

    def _extract_predictions(self, pred):
        """
        A1. Split the 15-value prediction into displacements and face forces.

        Args:
            pred: (N, 15) raw model output

        Returns:
            disp:        (N, 3)    → [ux, uz, θy]
            face_forces: (N, 4, 3) → [Fx, Fz, My] per face (global)

        Face ordering:
            face 0 = +x,  face 1 = -x,  face 2 = +z,  face 3 = -z
        """
        disp = pred[:, 0:3]                                # (N, 3)
        face_forces = pred[:, 3:15].reshape(-1, 4, 3)      # (N, 4, 3)
        return disp, face_forces

    # ════════════════════════════════════════════════════════
    # SECTION B: CONCEPT 1 — EQUILIBRIUM LOSSES (NO AUTOGRAD)
    # ════════════════════════════════════════════════════════

    def _loss_equilibrium(self, face_forces, F_ext, bc_disp, bc_rot):
        """
        B1. Nodal equilibrium: sum of face forces = external load.

        At each FREE node n:
          Σ_{f=0}^{3} Fx^(f) = Fx_ext
          Σ_{f=0}^{3} Fz^(f) = Fz_ext
          Σ_{f=0}^{3} My^(f) = My_ext

        Support nodes EXCLUDED (unknown reactions).

        Args:
            face_forces: (N, 4, 3) predicted face forces (global)
            F_ext:       (N, 3) precomputed equivalent nodal loads
            bc_disp:     (N, 1) displacement BC flag
            bc_rot:      (N, 1) rotation BC flag

        Returns:
            L_eq: scalar loss
        """
        # Sum face forces over 4 faces → (N, 3)
        sum_forces = face_forces.sum(dim=1)          # (N, 3)

        # Residuals
        residual = sum_forces - F_ext                 # (N, 3)

        # Free node mask (not a support)
        free_mask = (bc_disp.squeeze(-1) < 0.5)      # (N,) bool

        res_x = residual[free_mask, 0]
        res_z = residual[free_mask, 1]
        res_m = residual[free_mask, 2]

        L_eq = (res_x.pow(2) + res_z.pow(2) + res_m.pow(2)).mean()

        return L_eq

    def _loss_free_face(self, face_forces, face_mask):
        """
        B2. Unconnected face forces must be zero.

        If face_mask[n, f] == 0, then face_forces[n, f, :] should be zero.
        Already enforced by hard mask in model.py — soft backup.

        Args:
            face_forces: (N, 4, 3) predicted face forces
            face_mask:   (N, 4) connectivity mask

        Returns:
            L_free: scalar loss
        """
        free = (face_mask < 0.5).unsqueeze(-1)        # (N, 4, 1)
        free = free.expand_as(face_forces)             # (N, 4, 3)

        free_forces = face_forces[free]

        if free_forces.numel() == 0:
            return torch.tensor(0.0, device=face_forces.device)

        L_free = free_forces.pow(2).mean()
        return L_free

    def _loss_support(self, disp, bc_disp, bc_rot):
        """
        B3. Support displacements must be zero.

        Already enforced by hard BC in model.py — soft backup.

        Args:
            disp:    (N, 3) predicted [ux, uz, θy]
            bc_disp: (N, 1) displacement BC flag
            bc_rot:  (N, 1) rotation BC flag

        Returns:
            L_sup: scalar loss
        """
        sup_disp = (bc_disp.squeeze(-1) > 0.5)       # (N,)
        sup_rot  = (bc_rot.squeeze(-1) > 0.5)         # (N,)

        loss = torch.tensor(0.0, device=disp.device)

        if sup_disp.any():
            loss = loss + disp[sup_disp, 0].pow(2).mean()   # ux
            loss = loss + disp[sup_disp, 1].pow(2).mean()   # uz

        if sup_rot.any():
            loss = loss + disp[sup_rot, 2].pow(2).mean()    # θy

        return loss

    # ════════════════════════════════════════════════════════
    # SECTION C: CONCEPT 2+3 — CONSTITUTIVE LOSSES (AUTOGRAD)
    #
    # Per-node iteration: for each (node, face) pair, look up
    # the element, transform to local, compute N/M/V residual.
    # Mathematically identical to per-element with A/B ends.
    # ════════════════════════════════════════════════════════

    def _compute_autograd_derivs(self, pred, coords):
        """
        C0. Compute spatial derivatives of predicted displacements
            w.r.t. input coordinates using torch.autograd.

        Computes:
            1st order: ∂ux/∂x, ∂uz/∂z
            2nd order: ∂²uz/∂x², ∂²ux/∂z²
            3rd order: ∂³uz/∂x³, ∂³ux/∂z³

        Derivatives needed per element orientation:
            Horizontal beam (α=0°):
                N = EA · ∂ux/∂x
                M = EI · ∂²uz/∂x²
                V = EI · ∂³uz/∂x³

            Vertical column (α=90°):
                N = EA · ∂uz/∂z
                M = EI · (-∂²ux/∂z²)    [w_L = -ux]
                V = EI · (-∂³ux/∂z³)

        KNOWN PROBLEM:
            autograd.grad computes ∂(Σ_j pred_j)/∂coords_i.
            Through GNN message passing, pred_j depends on coords_i
            for ALL neighbours j. The gradient includes cross-node
            contamination — NOT the physical strain/curvature.

        Args:
            pred:   (N, 15) model predictions
            coords: (N, 3) coordinates with requires_grad=True

        Returns:
            dict with keys:
                'dux_dx':    (N,) ∂ux/∂x   → axial strain (horizontal)
                'duz_dz':    (N,) ∂uz/∂z   → axial strain (vertical)
                'd2uz_dx2':  (N,) ∂²uz/∂x² → curvature (horizontal)
                'd2ux_dz2':  (N,) ∂²ux/∂z² → curvature (vertical)
                'd3uz_dx3':  (N,) ∂³uz/∂x³ → shear (horizontal)
                'd3ux_dz3':  (N,) ∂³ux/∂z³ → shear (vertical)
        """
        ux = pred[:, 0]     # (N,)
        uz = pred[:, 1]     # (N,)

        # ── 1st ORDER ──
        grad_ux = torch.autograd.grad(
            ux.sum(), coords,
            create_graph=True, retain_graph=True
        )[0]    # (N, 3)

        grad_uz = torch.autograd.grad(
            uz.sum(), coords,
            create_graph=True, retain_graph=True
        )[0]    # (N, 3)

        dux_dx = grad_ux[:, 0]    # ∂ux/∂x
        dux_dz = grad_ux[:, 2]    # ∂ux/∂z
        duz_dx = grad_uz[:, 0]    # ∂uz/∂x
        duz_dz = grad_uz[:, 2]    # ∂uz/∂z

        # ── 2nd ORDER ──
        grad_duz_dx = torch.autograd.grad(
            duz_dx.sum(), coords,
            create_graph=True, retain_graph=True
        )[0]
        d2uz_dx2 = grad_duz_dx[:, 0]    # ∂²uz/∂x²

        grad_dux_dz = torch.autograd.grad(
            dux_dz.sum(), coords,
            create_graph=True, retain_graph=True
        )[0]
        d2ux_dz2 = grad_dux_dz[:, 2]    # ∂²ux/∂z²

        # ── 3rd ORDER ──
        grad_d2uz_dx2 = torch.autograd.grad(
            d2uz_dx2.sum(), coords,
            create_graph=True, retain_graph=True
        )[0]
        d3uz_dx3 = grad_d2uz_dx2[:, 0]   # ∂³uz/∂x³

        grad_d2ux_dz2 = torch.autograd.grad(
            d2ux_dz2.sum(), coords,
            create_graph=True, retain_graph=True
        )[0]
        d3ux_dz3 = grad_d2ux_dz2[:, 2]   # ∂³ux/∂z³

        return {
            'dux_dx': dux_dx,
            'duz_dz': duz_dz,
            'd2uz_dx2': d2uz_dx2,
            'd2ux_dz2': d2ux_dz2,
            'd3uz_dx3': d3uz_dx3,
            'd3ux_dz3': d3ux_dz3,
        }

    def _loss_constitutive_per_node(self, derivs, face_forces, data):
        """
        C1. All three constitutive losses computed per-node.

        Iterates over all active (node, face) pairs. For each pair:
          1. Look up element index, properties, direction
          2. Determine if node is A-end or B-end of that element
          3. Transform face force to local coords of that element
          4. Compute N, M, V from autograd derivatives at this node
          5. Apply sign convention (A: +sign, B: -sign)
          6. Compute residual

        Total checks = 2E (each element end visited exactly once).
        Mathematically identical to per-element A/B approach.

        Sign convention (unified):
          A-end (sign=+1): ff_loc + N = 0,  ff_loc + V = 0,  ff_loc + M = 0
          B-end (sign=-1): ff_loc - N = 0,  ff_loc - V = 0,  ff_loc - M = 0

        Args:
            derivs:      dict from _compute_autograd_derivs
            face_forces: (N_nodes, 4, 3) predicted face forces (global)
            data:        PyG Data

        Returns:
            L_N: scalar axial constitutive loss
            L_M: scalar moment constitutive loss
            L_V: scalar shear constitutive loss
        """
        feid = data.face_element_id       # (N_nodes, 4)
        faa  = data.face_is_A_end         # (N_nodes, 4)
        fm   = data.face_mask             # (N_nodes, 4)

        device = face_forces.device

        # ──────────────────────────────────────────────
        # Step 1: Find all active (node, face) pairs
        # ──────────────────────────────────────────────
        active = fm > 0.5                             # (N_nodes, 4) bool
        node_idx, face_idx = torch.where(active)      # (K,), (K,)
        # K = total active pairs = 2E

        if node_idx.numel() == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero

        # ──────────────────────────────────────────────
        # Step 2: Look up element info for each pair
        # ──────────────────────────────────────────────
        elem_idx = feid[node_idx, face_idx].long()    # (K,) element index
        is_A     = faa[node_idx, face_idx]             # (K,) 1=A, 0=B

        # Element properties
        EA = (data.prop_E * data.prop_A)[elem_idx]    # (K,)
        EI = (data.prop_E * data.prop_I22)[elem_idx]  # (K,)

        # Element direction
        elem_dir = data.elem_directions[elem_idx]     # (K, 3)
        cos_a = elem_dir[:, 0:1]                      # (K, 1)
        sin_a = elem_dir[:, 2:3]                      # (K, 1)

        # ──────────────────────────────────────────────
        # Step 3: Get face forces → transform to local
        # ──────────────────────────────────────────────
        # Global face force at this (node, face) pair
        ff_g = face_forces[node_idx, face_idx, :]     # (K, 3) [Fx, Fz, My]

        # Rotate global → local of the element
        #   Fx_loc =  Fx·cos(α) + Fz·sin(α)
        #   Fz_loc = -Fx·sin(α) + Fz·cos(α)
        #   My_loc =  My  (unchanged)
        ff_loc = torch.cat([
             ff_g[:, 0:1] * cos_a + ff_g[:, 1:2] * sin_a,   # Fx_loc
            -ff_g[:, 0:1] * sin_a + ff_g[:, 1:2] * cos_a,   # Fz_loc
             ff_g[:, 2:3],                                     # My_loc
        ], dim=1)   # (K, 3)

        # ──────────────────────────────────────────────
        # Step 4: Compute N, M, V from autograd derivs
        # ──────────────────────────────────────────────
        dx = elem_dir[:, 0]
        dz = elem_dir[:, 2]
        is_horiz = dx.abs() > dz.abs()   # (K,)

        # Axial: N = EA · du_L/ds
        #   Horizontal: ∂ux/∂x
        #   Vertical:   ∂uz/∂z
        strain = torch.where(
            is_horiz,
            derivs['dux_dx'][node_idx],
            derivs['duz_dz'][node_idx]
        )   # (K,)
        N = EA * strain   # (K,)

        # Moment: M = EI · d²w_L/ds²
        #   Horizontal: ∂²uz/∂x²
        #   Vertical:  -∂²ux/∂z²  (w_L = -ux)
        curv = torch.where(
            is_horiz,
            derivs['d2uz_dx2'][node_idx],
            -derivs['d2ux_dz2'][node_idx]
        )   # (K,)
        M = EI * curv   # (K,)

        # Shear: V = EI · d³w_L/ds³
        #   Horizontal: ∂³uz/∂x³
        #   Vertical:  -∂³ux/∂z³
        d3w = torch.where(
            is_horiz,
            derivs['d3uz_dx3'][node_idx],
            -derivs['d3ux_dz3'][node_idx]
        )   # (K,)
        V = EI * d3w   # (K,)

        # ──────────────────────────────────────────────
        # Step 5: Sign convention
        # ──────────────────────────────────────────────
        #   A-end: face = -internal → ff + internal = 0 → sign = +1
        #   B-end: face = +internal → ff - internal = 0 → sign = -1
        sign = torch.where(
            is_A > 0.5,
            torch.ones_like(N),
            -torch.ones_like(N)
        )   # (K,)

        # ──────────────────────────────────────────────
        # Step 6: Residuals
        # ──────────────────────────────────────────────
        res_N = ff_loc[:, 0] + sign * N     # Fx_loc ± N = 0
        res_V = ff_loc[:, 1] + sign * V     # Fz_loc ± V = 0
        res_M = ff_loc[:, 2] + sign * M     # My_loc ± M = 0

        L_N = res_N.pow(2).mean()
        L_M = res_M.pow(2).mean()
        L_V = res_V.pow(2).mean()

        return L_N, L_M, L_V

    # ════════════════════════════════════════════════════════
    # SECTION D: TOTAL LOSS
    # ════════════════════════════════════════════════════════

    def forward(self, model, data):
        """
        Compute total physics loss.

        Flow:
          1. Forward pass → 15 predictions per node
          2. Extract displacements and face forces
          3. Concept 1: equilibrium losses (no autograd)
          4. Concept 2+3: constitutive losses (per-node, autograd)
          5. Weighted sum

        Args:
            model: PIGNN model (must have get_coords() method)
            data:  PyG Data/Batch

        Returns:
            total_loss: scalar (differentiable)
            loss_dict:  dict of individual loss values (detached)
            pred:       (N, 15) predictions
        """
        # ════════════════════════════════════
        # 1. FORWARD PASS
        # ════════════════════════════════════
        pred = model(data)                           # (N, 15)
        coords = model.get_coords()                  # (N, 3) requires_grad

        # ════════════════════════════════════
        # 2. EXTRACT PREDICTIONS
        # ════════════════════════════════════
        disp, face_forces = self._extract_predictions(pred)
        # disp:        (N, 3)    [ux, uz, θy]
        # face_forces: (N, 4, 3) [Fx, Fz, My] × 4 faces

        # ════════════════════════════════════
        # 3. CONCEPT 1: EQUILIBRIUM (no autograd)
        # ════════════════════════════════════
        L_eq   = self._loss_equilibrium(
            face_forces, data.F_ext, data.bc_disp, data.bc_rot
        )
        L_free = self._loss_free_face(face_forces, data.face_mask)
        L_sup  = self._loss_support(disp, data.bc_disp, data.bc_rot)

        # ════════════════════════════════════
        # 4. CONCEPT 2+3: CONSTITUTIVE (per-node, autograd)
        #    Transform to local + compute N/M/V in one step
        # ════════════════════════════════════
        derivs = self._compute_autograd_derivs(pred, coords)
        L_N, L_M, L_V = self._loss_constitutive_per_node(
            derivs, face_forces, data
        )

        # ════════════════════════════════════
        # 5. TOTAL LOSS (weighted sum)
        # ════════════════════════════════════
        total = (self.w_eq   * L_eq
               + self.w_free * L_free
               + self.w_sup  * L_sup
               + self.w_N    * L_N
               + self.w_M    * L_M
               + self.w_V    * L_V)

        loss_dict = {
            'L_eq':    L_eq.item(),
            'L_free':  L_free.item(),
            'L_sup':   L_sup.item(),
            'L_N':     L_N.item(),
            'L_M':     L_M.item(),
            'L_V':     L_V.item(),
            'total':   total.item(),
        }

        return total, loss_dict, pred


# ================================================================
# QUICK TEST
# ================================================================

if __name__ == "__main__":
    import os
    from pathlib import Path
    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    from model import PIGNN

    print("=" * 60)
    print("  PHYSICS LOSS TEST (Naive Autograd, Per-Node)")
    print("=" * 60)

    # ── Load graph ──
    data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
    data = data_list[0]
    print(f"  Graph: {data.num_nodes} nodes, {data.n_elements} elements")

    # ── Create model and loss ──
    model = PIGNN(node_in_dim=9, edge_in_dim=10, hidden_dim=64, n_layers=3)
    loss_fn = NaivePhysicsLoss(
        w_eq=1.0, w_free=1.0, w_sup=1.0,
        w_N=1.0, w_M=1.0, w_V=1.0
    )

    print(f"  Model params: {model.count_params():,}")

    # ── Forward + loss ──
    model.train()
    total_loss, loss_dict, pred = loss_fn(model, data)

    print(f"\n  Predictions: {pred.shape}")
    print(f"  Displacements range: [{pred[:, :3].min():.6f}, "
          f"{pred[:, :3].max():.6f}]")
    print(f"  Face forces range:   [{pred[:, 3:].min():.6f}, "
          f"{pred[:, 3:].max():.6f}]")

    print(f"\n  Individual losses:")
    print(f"  {'─'*45}")
    for name, val in loss_dict.items():
        marker = ""
        if name in ['L_M', 'L_V']:
            marker = "  ← EXPECTED PROBLEM"
        print(f"    {name:<10} {val:>15.6e}{marker}")
    print(f"  {'─'*45}")
    print(f"    {'total':<10} {loss_dict['total']:>15.6e}")

    # ── Backward test ──
    print(f"\n  Backward pass...")
    total_loss.backward()
    grad_norm = sum(
        p.grad.norm().item()
        for p in model.parameters()
        if p.grad is not None
    )
    print(f"  Gradient norm: {grad_norm:.6e}")
    print(f"  ✓ Backward pass successful")

    # ── Verify check count ──
    fm = data.face_mask
    n_active = (fm > 0.5).sum().item()
    n_elements = data.n_elements
    print(f"\n  Verification:")
    print(f"    Active (node, face) pairs: {n_active}")
    print(f"    Expected (2 × E):          {2 * n_elements}")
    print(f"    Match: {'✓' if n_active == 2 * n_elements else '✗'}")

    # ── Autograd derivative inspection ──
    print(f"\n  Autograd derivative magnitudes:")
    model.zero_grad()
    pred2 = model(data)
    coords = model.get_coords()
    derivs = loss_fn._compute_autograd_derivs(pred2, coords)
    for key, val in derivs.items():
        print(f"    {key:<12} range: [{val.min():.4e}, {val.max():.4e}]  "
              f"mean: {val.abs().mean():.4e}")

    print(f"\n{'='*60}")
    print(f"  PHYSICS LOSS TEST COMPLETE ✓")
    print(f"  Ready for Step 4 (train.py)")
    print(f"{'='*60}")