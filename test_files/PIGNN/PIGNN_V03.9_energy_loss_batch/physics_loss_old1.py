"""
=================================================================
physics_loss.py — Naive Autograd Physics Loss (6 Terms)
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
    A2. _get_element_end_data()   → gather A/B node data per element
    A3. _transform_to_local()     → Concept 2: global → local

  Section B — Concept 1: Equilibrium (NO autograd)
    B1. _loss_equilibrium()       → L_eq
    B2. _loss_free_face()         → L_free
    B3. _loss_support()           → L_sup

  Section C — Concept 3: Constitutive (AUTOGRAD)
    C0. _compute_autograd_derivs()→ spatial derivatives via autograd
    C1. _loss_axial()             → L_N
    C2. _loss_moment()            → L_M
    C3. _loss_shear()             → L_V

  Section D — Total
    forward()                     → weighted sum

SIGN CONVENTION (from document Section 2.2):
  Element end forces (element → node):
    At A (x=0): Fx_A^loc = -N,  Fz_A^loc = -V,  My_A^loc = -M
    At B (x=L): Fx_B^loc = +N,  Fz_B^loc = +V,  My_B^loc = +M

  So the constitutive loss checks:
    (face_force_A^loc + N) = 0   →  (Fx_A^loc + N)² 
    (face_force_B^loc - N) = 0   →  (Fx_B^loc - N)²
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

    def _get_element_end_data(self, disp, face_forces, data):
        """
        A2. Gather displacement and face force data at element ends.

        For each element e (connecting node A → node B):
          - Look up which face of node A connects to element e
          - Look up which face of node B connects to element e
          - Gather the displacements and face forces at those faces

        Args:
            disp:        (N, 3)    predicted displacements
            face_forces: (N, 4, 3) predicted face forces (global)
            data:        PyG Data with face_element_id, face_is_A_end, connectivity

        Returns:
            disp_A:  (E, 3) displacement at A-end nodes
            disp_B:  (E, 3) displacement at B-end nodes
            ff_A:    (E, 3) face force at A-end [Fx, Fz, My] (global)
            ff_B:    (E, 3) face force at B-end [Fx, Fz, My] (global)
        """
        conn = data.connectivity               # (E, 2)
        feid = data.face_element_id             # (N, 4)
        faa  = data.face_is_A_end               # (N, 4)
        E = conn.shape[0]

        # ── Displacements at element ends ──
        # Simple: just index into disp using connectivity
        disp_A = disp[conn[:, 0]]               # (E, 3)
        disp_B = disp[conn[:, 1]]               # (E, 3)

        # ── Face forces at element ends ──
        # Need to find which face of node A/B connects to element e
        # face_element_id[node, face] = element index
        # face_is_A_end[node, face] = 1 if A-end

        # Build lookup: for each element, which (node, face) is its A-end and B-end
        ff_A = torch.zeros(E, 3, device=disp.device)
        ff_B = torch.zeros(E, 3, device=disp.device)

        N = disp.shape[0]
        for f in range(4):
            # All nodes that have an element at face f
            elem_at_face = feid[:, f]               # (N,)
            is_A_at_face = faa[:, f]                 # (N,)

            for n in range(N):
                e = elem_at_face[n].item()
                if e < 0:
                    continue   # no element at this face

                if is_A_at_face[n].item() == 1:
                    ff_A[e] = face_forces[n, f, :]
                else:
                    ff_B[e] = face_forces[n, f, :]

        return disp_A, disp_B, ff_A, ff_B

    def _get_element_end_data_vectorized(self, disp, face_forces, data):
        """
        A2 (vectorized). Same as _get_element_end_data but without Python loops.

        Uses scatter/gather operations for GPU efficiency.
        """
        conn = data.connectivity               # (E, 2)
        feid = data.face_element_id             # (N, 4)
        faa  = data.face_is_A_end               # (N, 4)
        fm   = data.face_mask                    # (N, 4)
        E = conn.shape[0]
        device = disp.device

        # ── Displacements at element ends ──
        disp_A = disp[conn[:, 0]]               # (E, 3)
        disp_B = disp[conn[:, 1]]               # (E, 3)

        # ── Face forces at element ends (vectorized) ──
        ff_A = torch.zeros(E, 3, device=device)
        ff_B = torch.zeros(E, 3, device=device)

        # Process each face index (only 4 iterations, not N)
        for f in range(4):
            # Which nodes have an element at face f?
            mask = fm[:, f] > 0.5                # (N,) bool
            if not mask.any():
                continue

            nodes_with_face = torch.where(mask)[0]           # (K,)
            elems = feid[nodes_with_face, f]                  # (K,)
            is_A  = faa[nodes_with_face, f]                   # (K,)
            forces = face_forces[nodes_with_face, f, :]       # (K, 3)

            # A-end nodes
            a_mask = is_A == 1
            if a_mask.any():
                a_elems = elems[a_mask]                        # (Ka,)
                a_forces = forces[a_mask]                      # (Ka, 3)
                ff_A[a_elems] = a_forces

            # B-end nodes
            b_mask = is_A == 0
            if b_mask.any():
                b_elems = elems[b_mask]                        # (Kb,)
                b_forces = forces[b_mask]                      # (Kb, 3)
                ff_B[b_elems] = b_forces

        return disp_A, disp_B, ff_A, ff_B

    def _transform_to_local(self, disp_A, disp_B, ff_A, ff_B,
                             elem_directions):
        """
        A3. Concept 2: Global → Local transformation.

        Rotation matrix T(α):
            [u_L]     [cos α   sin α   0] [ux]
            [w_L]  =  [-sin α  cos α   0] [uz]
            [θ_L]     [0       0       1] [θy]

        Applied to both displacements and face forces.

        For 2D in XZ plane:
            cos α = elem_directions[:, 0]  (dx component)
            sin α = elem_directions[:, 2]  (dz component)

        Horizontal beam (α=0°): cos=1, sin=0 → local = global
        Vertical column (α=90°): cos=0, sin=1 → u_L=uz, w_L=-ux

        Args:
            disp_A, disp_B: (E, 3) global displacements [ux, uz, θy]
            ff_A, ff_B:     (E, 3) global face forces [Fx, Fz, My]
            elem_directions:(E, 3) unit direction vectors

        Returns:
            disp_A_loc, disp_B_loc: (E, 3) local [u_L, w_L, θ_L]
            ff_A_loc, ff_B_loc:     (E, 3) local [Fx_loc, Fz_loc, My_loc]
        """
        cos_a = elem_directions[:, 0:1]    # (E, 1)
        sin_a = elem_directions[:, 2:3]    # (E, 1)

        def rotate(v):
            """Apply T(α) to (E, 3) vector [vx, vz, vθ]."""
            vx = v[:, 0:1]
            vz = v[:, 1:2]
            vt = v[:, 2:3]
            return torch.cat([
                 vx * cos_a + vz * sin_a,    # u_L or Fx_loc
                -vx * sin_a + vz * cos_a,    # w_L or Fz_loc
                 vt,                          # θ_L or My_loc
            ], dim=1)

        disp_A_loc = rotate(disp_A)
        disp_B_loc = rotate(disp_B)
        ff_A_loc   = rotate(ff_A)
        ff_B_loc   = rotate(ff_B)

        return disp_A_loc, disp_B_loc, ff_A_loc, ff_B_loc

    # ════════════════════════════════════════════════════════
    # SECTION B: CONCEPT 1 — EQUILIBRIUM LOSSES (NO AUTOGRAD)
    # ════════════════════════════════════════════════════════

    def _loss_equilibrium(self, face_forces, F_ext, bc_disp, bc_rot):
        """
        B1. Nodal equilibrium: sum of face forces = external load.

        At each FREE node n:
          Σ_{f=0}^{3} Fx^(f) = Fx_ext     (typically 0)
          Σ_{f=0}^{3} Fz^(f) = Fz_ext     (from UDL)
          Σ_{f=0}^{3} My^(f) = My_ext     (from UDL fixed-end moments)

        Support nodes are EXCLUDED because reaction forces are unknown.

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

        # Residuals: sum_forces - F_ext
        residual = sum_forces - F_ext                 # (N, 3)

        # Free node mask
        # A node is "free" for force equilibrium if it is NOT a support
        # (support nodes have unknown reactions)
        free_mask = (bc_disp.squeeze(-1) < 0.5)      # (N,) bool

        # Also: moment equilibrium only at nodes free in rotation
        # But for simplicity, use same mask (support nodes excluded entirely)
        # More refined: separate masks for force vs moment
        res_x = residual[free_mask, 0]                # Fx residual
        res_z = residual[free_mask, 1]                # Fz residual
        res_m = residual[free_mask, 2]                # My residual

        L_eq = (res_x.pow(2) + res_z.pow(2) + res_m.pow(2)).mean()

        return L_eq

    def _loss_free_face(self, face_forces, face_mask):# hard enforceed free face force to zero
        """
        B2. Unconnected face forces must be zero.

        If face_mask[n, f] == 0, then face_forces[n, f, :] should be zero.

        This is already enforced by hard mask in model.py,
        but we keep a soft penalty as backup/regularisation.

        Args:
            face_forces: (N, 4, 3) predicted face forces
            face_mask:   (N, 4) connectivity mask

        Returns:
            L_free: scalar loss
        """
        # Free faces: where mask == 0
        free = (face_mask < 0.5).unsqueeze(-1)        # (N, 4, 1)
        free = free.expand_as(face_forces)             # (N, 4, 3)

        # Forces at free faces (should all be zero)
        free_forces = face_forces[free]

        if free_forces.numel() == 0:
            return torch.tensor(0.0, device=face_forces.device)

        L_free = free_forces.pow(2).mean()
        return L_free

    def _loss_support(self, disp, bc_disp, bc_rot):
        """
        B3. Support displacements must be zero.

        Already enforced by hard BC in model.py.
        Soft penalty as backup.

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
    # SECTION C: CONCEPT 3 — CONSTITUTIVE LOSSES (AUTOGRAD)
    # ════════════════════════════════════════════════════════

    def _compute_autograd_derivs(self, pred, coords):
        """
        C0. Compute spatial derivatives of predicted displacements
            w.r.t. input coordinates using torch.autograd.

        Computes:
            1st order: ∂ux/∂x, ∂ux/∂z, ∂uz/∂x, ∂uz/∂z
            2nd order: ∂²uz/∂x², ∂²ux/∂z²
            3rd order: ∂³uz/∂x³, ∂³ux/∂z³

        WHY we compute these specific derivatives:
            Horizontal beam (α=0°):
                Axial:  u_L = ux  → N = EA · ∂ux/∂x
                Transverse: w_L = uz → M = EI · ∂²uz/∂x²
                                       V = EI · ∂³uz/∂x³
            Vertical column (α=90°):
                Axial:  u_L = uz  → N = EA · ∂uz/∂z
                Transverse: w_L = -ux → M = EI · (-∂²ux/∂z²)
                                         V = EI · (-∂³ux/∂z³)

        KNOWN PROBLEM:
            autograd.grad(pred.sum(), coords) computes ∂(Σ_j pred_j)/∂coords_i.
            Through the GNN message passing, pred_j depends on coords_i
            for ALL neighbours j, not just node i itself. This means:
                ∂(Σ_j ux_j)/∂x_i  ≠  ∂ux_i/∂x_i
            The gradient includes "cross-node" contamination from
            message passing. This is NOT the physical strain.

        Args:
            pred:   (N, 15) model predictions
            coords: (N, 3) coordinates with requires_grad=True

        Returns:
            dict with keys:
                'dux_dx', 'dux_dz': (N,) 1st derivatives of ux
                'duz_dx', 'duz_dz': (N,) 1st derivatives of uz
                'd2uz_dx2':         (N,) 2nd derivative (horizontal M)
                'd2ux_dz2':         (N,) 2nd derivative (vertical M)
                'd3uz_dx3':         (N,) 3rd derivative (horizontal V)
                'd3ux_dz3':         (N,) 3rd derivative (vertical V)
        """
        ux = pred[:, 0]     # (N,)
        uz = pred[:, 1]     # (N,)

        # ────────────────────────────────────────
        # 1st ORDER: ∂ux/∂coords, ∂uz/∂coords
        # ────────────────────────────────────────
        grad_ux = torch.autograd.grad(
            ux.sum(), coords,
            create_graph=True, retain_graph=True
        )[0]    # (N, 3): [∂ux/∂x, ∂ux/∂y, ∂ux/∂z]

        grad_uz = torch.autograd.grad(
            uz.sum(), coords,
            create_graph=True, retain_graph=True
        )[0]    # (N, 3): [∂uz/∂x, ∂uz/∂y, ∂uz/∂z]

        dux_dx = grad_ux[:, 0]    # ∂ux/∂x → axial strain (horizontal)
        dux_dz = grad_ux[:, 2]    # ∂ux/∂z → needed for vertical
        duz_dx = grad_uz[:, 0]    # ∂uz/∂x → curvature (horizontal)
        duz_dz = grad_uz[:, 2]    # ∂uz/∂z → axial strain (vertical)

        # ────────────────────────────────────────
        # 2nd ORDER: ∂²uz/∂x², ∂²ux/∂z²
        # ────────────────────────────────────────
        grad_duz_dx = torch.autograd.grad(
            duz_dx.sum(), coords,
            create_graph=True, retain_graph=True
        )[0]    # (N, 3)
        d2uz_dx2 = grad_duz_dx[:, 0]    # ∂²uz/∂x² → curvature (horiz)

        grad_dux_dz = torch.autograd.grad(
            dux_dz.sum(), coords,
            create_graph=True, retain_graph=True
        )[0]    # (N, 3)
        d2ux_dz2 = grad_dux_dz[:, 2]    # ∂²ux/∂z² → curvature (vert)

        # ────────────────────────────────────────
        # 3rd ORDER: ∂³uz/∂x³, ∂³ux/∂z³
        # ────────────────────────────────────────
        grad_d2uz_dx2 = torch.autograd.grad(
            d2uz_dx2.sum(), coords,
            create_graph=True, retain_graph=True
        )[0]    # (N, 3)
        d3uz_dx3 = grad_d2uz_dx2[:, 0]   # ∂³uz/∂x³ → shear (horiz)

        grad_d2ux_dz2 = torch.autograd.grad(
            d2ux_dz2.sum(), coords,
            create_graph=True, retain_graph=True
        )[0]    # (N, 3)
        d3ux_dz3 = grad_d2ux_dz2[:, 2]   # ∂³ux/∂z³ → shear (vert)

        return {
            # 1st order
            'dux_dx': dux_dx,    'dux_dz': dux_dz,
            'duz_dx': duz_dx,    'duz_dz': duz_dz,
            # 2nd order
            'd2uz_dx2': d2uz_dx2,
            'd2ux_dz2': d2ux_dz2,
            # 3rd order
            'd3uz_dx3': d3uz_dx3,
            'd3ux_dz3': d3ux_dz3,
        }

    def _classify_elements(self, elem_directions):
        """
        Classify each element as horizontal or vertical.

        Args:
            elem_directions: (E, 3) unit direction vectors

        Returns:
            is_horizontal: (E,) bool — True if |dx| > |dz|
            cos_a: (E, 1) — cos(α)
            sin_a: (E, 1) — sin(α)
        """
        dx = elem_directions[:, 0]
        dz = elem_directions[:, 2]
        is_horizontal = dx.abs() > dz.abs()
        cos_a = dx.unsqueeze(-1)    # (E, 1)
        sin_a = dz.unsqueeze(-1)    # (E, 1)
        return is_horizontal, cos_a, sin_a

    def _loss_axial(self, derivs, ff_A_loc, ff_B_loc, data):
        """
        C1. Axial constitutive loss: N = EA · du_L/ds

        For each element e:
            Horizontal: N = EA · ∂ux/∂x
            Vertical:   N = EA · ∂uz/∂z

        Compare with face forces (already in local coords):
            At A: (Fx_A^loc + N_A)² = 0   [because Fx_A^loc = -N]
            At B: (Fx_B^loc - N_B)² = 0   [because Fx_B^loc = +N]

        KNOWN PROBLEM:
            The autograd derivative ∂ux/∂x at node A is NOT the same
            as du_L/ds for element e. It's a global sensitivity
            contaminated by message passing.

        Args:
            derivs:    dict from _compute_autograd_derivs
            ff_A_loc:  (E, 3) face forces at A in local coords
            ff_B_loc:  (E, 3) face forces at B in local coords
            data:      PyG Data

        Returns:
            L_N: scalar loss
        """
        conn = data.connectivity                    # (E, 2)
        EA = (data.prop_E * data.prop_A)             # (E,)
        is_horiz, _, _ = self._classify_elements(data.elem_directions)

        # Gather derivatives at element end nodes
        nA = conn[:, 0]    # (E,)
        nB = conn[:, 1]    # (E,)

        # Axial strain du_L/ds:
        #   Horizontal → ∂ux/∂x
        #   Vertical   → ∂uz/∂z
        strain_A = torch.where(
            is_horiz,
            derivs['dux_dx'][nA],
            derivs['duz_dz'][nA]
        )   # (E,)

        strain_B = torch.where(
            is_horiz,
            derivs['dux_dx'][nB],
            derivs['duz_dz'][nB]
        )   # (E,)

        # N = EA · strain
        N_A = EA * strain_A    # (E,)
        N_B = EA * strain_B    # (E,)

        # Face axial force (local): ff_loc[:, 0] = Fx_loc
        Fx_A = ff_A_loc[:, 0]  # (E,)
        Fx_B = ff_B_loc[:, 0]  # (E,)

        # Residuals (from sign convention):
        #   At A: Fx_A^loc = -N  →  Fx_A + N = 0
        #   At B: Fx_B^loc = +N  →  Fx_B - N = 0
        res_A = Fx_A + N_A
        res_B = Fx_B - N_B

        L_N = (res_A.pow(2) + res_B.pow(2)).mean()
        return L_N

    def _loss_moment(self, derivs, ff_A_loc, ff_B_loc, data):
        """
        C2. Moment constitutive loss: M = EI · d²w_L/ds²

        For each element e:
            Horizontal: M = EI · ∂²uz/∂x²
            Vertical:   M = EI · (-∂²ux/∂z²)
                         (negative because w_L = -ux for vertical)

        Compare with face moments (already in local coords):
            At A: (My_A^loc + M_A)² = 0   [because My_A = -M(0)]
            At B: (My_B^loc - M_B)² = 0   [because My_B = +M(L)]

        KNOWN PROBLEM:
            ∂²uz/∂x² via autograd through GNN is NOT the physical
            curvature. It measures how the GNN output changes
            when input coordinates shift — not the beam curvature.

        Args:
            derivs:    dict from _compute_autograd_derivs
            ff_A_loc:  (E, 3) face forces at A in local coords
            ff_B_loc:  (E, 3) face forces at B in local coords
            data:      PyG Data

        Returns:
            L_M: scalar loss
        """
        conn = data.connectivity
        EI = (data.prop_E * data.prop_I22)           # (E,)
        is_horiz, _, _ = self._classify_elements(data.elem_directions)

        nA = conn[:, 0]
        nB = conn[:, 1]

        # Curvature d²w_L/ds²:
        #   Horizontal → ∂²uz/∂x²
        #   Vertical   → -∂²ux/∂z²  (because w_L = -ux)
        curv_A = torch.where(
            is_horiz,
            derivs['d2uz_dx2'][nA],
            -derivs['d2ux_dz2'][nA]
        )   # (E,)

        curv_B = torch.where(
            is_horiz,
            derivs['d2uz_dx2'][nB],
            -derivs['d2ux_dz2'][nB]
        )   # (E,)

        # M = EI · curvature
        M_A = EI * curv_A
        M_B = EI * curv_B

        # Face moment (local): ff_loc[:, 2] = My_loc
        My_A = ff_A_loc[:, 2]
        My_B = ff_B_loc[:, 2]

        # Residuals:
        #   At A: My_A^loc = -M(0)   →  My_A + M_A = 0
        #   At B: My_B^loc = +M(L)   →  My_B - M_B = 0
        res_A = My_A + M_A
        res_B = My_B - M_B

        L_M = (res_A.pow(2) + res_B.pow(2)).mean()
        return L_M

    def _loss_shear(self, derivs, ff_A_loc, ff_B_loc, data):
        """
        C3. Shear constitutive loss: V = EI · d³w_L/ds³

        For each element e:
            Horizontal: V = EI · ∂³uz/∂x³
            Vertical:   V = EI · (-∂³ux/∂z³)

        Compare with face shear forces (already in local coords):
            At A: (Fz_A^loc + V_A)² = 0
            At B: (Fz_B^loc - V_B)² = 0

        KNOWN PROBLEM:
            3rd derivative through GNN via autograd = pure noise.
            Three levels of chain rule through message passing
            amplify numerical errors to meaningless values.

        Args:
            derivs:    dict from _compute_autograd_derivs
            ff_A_loc:  (E, 3)
            ff_B_loc:  (E, 3)
            data:      PyG Data

        Returns:
            L_V: scalar loss
        """
        conn = data.connectivity
        EI = (data.prop_E * data.prop_I22)
        is_horiz, _, _ = self._classify_elements(data.elem_directions)

        nA = conn[:, 0]
        nB = conn[:, 1]

        # d³w_L/ds³:
        #   Horizontal → ∂³uz/∂x³
        #   Vertical   → -∂³ux/∂z³
        d3w_A = torch.where(
            is_horiz,
            derivs['d3uz_dx3'][nA],
            -derivs['d3ux_dz3'][nA]
        )

        d3w_B = torch.where(
            is_horiz,
            derivs['d3uz_dx3'][nB],
            -derivs['d3ux_dz3'][nB]
        )

        # V = EI · d³w/ds³
        V_A = EI * d3w_A
        V_B = EI * d3w_B

        # Face shear (local): ff_loc[:, 1] = Fz_loc
        Fz_A = ff_A_loc[:, 1]
        Fz_B = ff_B_loc[:, 1]

        # Residuals:
        #   At A: Fz_A^loc = -V(0)  →  Fz_A + V_A = 0
        #   At B: Fz_B^loc = +V(L)  →  Fz_B - V_B = 0
        res_A = Fz_A + V_A
        res_B = Fz_B - V_B

        L_V = (res_A.pow(2) + res_B.pow(2)).mean()
        return L_V

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
          4. Concept 2: transform to local coords
          5. Concept 3: constitutive losses (autograd)
          6. Weighted sum

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
        # 4. CONCEPT 2: TRANSFORM TO LOCAL
        # ════════════════════════════════════
        disp_A, disp_B, ff_A, ff_B = \
            self._get_element_end_data_vectorized(
                disp, face_forces, data
            )

        disp_A_loc, disp_B_loc, ff_A_loc, ff_B_loc = \
            self._transform_to_local(
                disp_A, disp_B, ff_A, ff_B,
                data.elem_directions
            )

        # ════════════════════════════════════
        # 5. CONCEPT 3: CONSTITUTIVE (autograd)
        # ════════════════════════════════════
        derivs = self._compute_autograd_derivs(pred, coords)

        L_N = self._loss_axial(derivs, ff_A_loc, ff_B_loc, data)
        L_M = self._loss_moment(derivs, ff_A_loc, ff_B_loc, data)
        L_V = self._loss_shear(derivs, ff_A_loc, ff_B_loc, data)

        # ════════════════════════════════════
        # 6. TOTAL LOSS (weighted sum)
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
    print("  PHYSICS LOSS TEST (Naive Autograd)")
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