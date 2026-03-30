"""
=================================================================
energy_loss.py — FIXED Energy Loss with Correct Rotation Transform
=================================================================

The bug: θ_y (global rotation about Y-axis) was NOT being 
transformed when converting from global to local coordinates.

For a beam element at angle α:
  - Local axial:      u = c·ux + s·uz
  - Local transverse: w = -s·ux + c·uz  
  - Local rotation:   θ_local = θ_global  (SAME for 2D!)

BUT the sign convention in the stiffness matrix assumes
θ_local is positive when rotating from axial toward 
transverse (right-hand rule about the local z-axis).

For a VERTICAL element (α=90°), the global θ_y causes
the beam to curve in the OPPOSITE local sense compared
to a horizontal element. This is automatically handled
by the stiffness matrix IF we use the full 6x6 
transformation matrix T (including rotation DOFs).

The full transformation is:
  d_local = T · d_global

where T is block diagonal:
  T = [R  0]
      [0  R]

  R = [ c  s  0]
      [-s  c  0]
      [ 0  0  1]

This means θ_local = θ_global (no sign change needed).

The ACTUAL bug is that K_local uses the convention where
positive w is "upward" relative to the beam axis, and
positive θ is counterclockwise. For the stiffness matrix
to work correctly, we need the FULL d^T K d formulation
where K is the GLOBAL stiffness matrix:

  K_global = T^T · K_local · T

This automatically handles all sign conventions.
=================================================================
"""

import torch
import torch.nn as nn


class FrameEnergyLoss(nn.Module):
    """
    Uses K_global = T^T K_local T for correctness.
    
    U = (1/2) Σ_e  d_global_e^T · K_global_e · d_global_e
    W = Σ_i F_ext_i · u_i
    Π = U - W
    """

    def __init__(self):
        super().__init__()

    def forward(self, model, data):
        pred_raw = model(data)

        u_phys = torch.zeros_like(pred_raw)
        u_phys[:, 0] = pred_raw[:, 0] * data.u_c
        u_phys[:, 1] = pred_raw[:, 1] * data.u_c
        u_phys[:, 2] = pred_raw[:, 2] * data.theta_c

        U_internal = self._strain_energy(u_phys, data)
        W_external = self._external_work(u_phys, data)

        Pi = U_internal - W_external

        E_c = (data.F_c * data.u_c).clamp(min=1e-30)
        Pi_norm = Pi / E_c

        loss_dict = {
            'total':      Pi_norm.item(),
            'Pi':         Pi.item(),
            'Pi_norm':    Pi_norm.item(),
            'U_internal': U_internal.item(),
            'W_external': W_external.item(),
            'U_over_W':   (
                U_internal
                / W_external.abs().clamp(min=1e-30)
            ).item(),
            'ux_range':   [u_phys[:, 0].min().item(),
                          u_phys[:, 0].max().item()],
            'uz_range':   [u_phys[:, 1].min().item(),
                          u_phys[:, 1].max().item()],
            'th_range':   [u_phys[:, 2].min().item(),
                          u_phys[:, 2].max().item()],
            'raw_range':  [pred_raw.min().item(),
                          pred_raw.max().item()],
        }

        return Pi_norm, loss_dict, pred_raw, u_phys

    def _strain_energy(self, u_phys, data):
        """
        U = (1/2) Σ_e d_e^T K_global_e d_e
        
        where K_global = T^T K_local T
        
        Instead of building K_global explicitly, we:
          1. Transform d_global → d_local using T
          2. Compute d_local^T K_local d_local
        
        This is equivalent and more efficient.
        
        The key: we must use the FULL transformation 
        matrix T, including the rotation DOF row.
        """
        conn = data.connectivity
        nA, nB = conn[:, 0], conn[:, 1]
        n_elem = conn.shape[0]
        device = u_phys.device

        L  = data.elem_lengths
        EA = data.prop_E * data.prop_A
        EI = data.prop_E * data.prop_I22

        c = data.elem_directions[:, 0]
        s = data.elem_directions[:, 2]

        # ════════════════════════════════════════
        # Build T matrix (6x6) for each element
        # ════════════════════════════════════════
        #
        # T = [R 0]    R = [ c  s  0]
        #     [0 R]        [-s  c  0]
        #                  [ 0  0  1]
        #
        # d_local = T · d_global

        T = torch.zeros(n_elem, 6, 6, device=device)

        # Top-left R (node A)
        T[:, 0, 0] = c     # u_loc_A = c*ux_A + s*uz_A
        T[:, 0, 1] = s
        T[:, 1, 0] = -s    # w_loc_A = -s*ux_A + c*uz_A
        T[:, 1, 1] = c
        T[:, 2, 2] = 1     # θ_loc_A = θ_A

        # Bottom-right R (node B)
        T[:, 3, 3] = c
        T[:, 3, 4] = s
        T[:, 4, 3] = -s
        T[:, 4, 4] = c
        T[:, 5, 5] = 1

        # ════════════════════════════════════════
        # Global DOF vector for each element
        # ════════════════════════════════════════

        d_global = torch.stack([
            u_phys[nA, 0],   # ux_A
            u_phys[nA, 1],   # uz_A
            u_phys[nA, 2],   # θ_A
            u_phys[nB, 0],   # ux_B
            u_phys[nB, 1],   # uz_B
            u_phys[nB, 2],   # θ_B
        ], dim=1)  # (E, 6)

        # ════════════════════════════════════════
        # Transform: d_local = T · d_global
        # ════════════════════════════════════════

        d_local = torch.bmm(
            T, d_global.unsqueeze(2)
        ).squeeze(2)  # (E, 6)

        # ════════════════════════════════════════
        # Build K_local (6x6) for each element
        # ════════════════════════════════════════

        K = torch.zeros(
            n_elem, 6, 6, device=device
        )

        ea_L = EA / L
        ei_L = EI / L
        ei_L2 = EI / L**2
        ei_L3 = EI / L**3

        # Axial stiffness
        K[:, 0, 0] = ea_L
        K[:, 0, 3] = -ea_L
        K[:, 3, 0] = -ea_L
        K[:, 3, 3] = ea_L

        # Bending stiffness
        K[:, 1, 1] = 12 * ei_L3
        K[:, 1, 2] = 6 * ei_L2
        K[:, 1, 4] = -12 * ei_L3
        K[:, 1, 5] = 6 * ei_L2

        K[:, 2, 1] = 6 * ei_L2
        K[:, 2, 2] = 4 * ei_L
        K[:, 2, 4] = -6 * ei_L2
        K[:, 2, 5] = 2 * ei_L

        K[:, 4, 1] = -12 * ei_L3
        K[:, 4, 2] = -6 * ei_L2
        K[:, 4, 4] = 12 * ei_L3
        K[:, 4, 5] = -6 * ei_L2

        K[:, 5, 1] = 6 * ei_L2
        K[:, 5, 2] = 2 * ei_L
        K[:, 5, 4] = -6 * ei_L2
        K[:, 5, 5] = 4 * ei_L

        # ════════════════════════════════════════
        # U_e = (1/2) d_local^T K_local d_local
        # ════════════════════════════════════════

        Kd = torch.bmm(
            K, d_local.unsqueeze(2)
        )  # (E, 6, 1)
        U_per_elem = 0.5 * torch.bmm(
            d_local.unsqueeze(1), Kd
        ).squeeze()  # (E,)

        return U_per_elem.sum()

    def _external_work(self, u_phys, data):
        """W = Σ_i (Fx·ux + Fz·uz + My·θ)"""
        W = (
            data.F_ext[:, 0] * u_phys[:, 0]
            + data.F_ext[:, 1] * u_phys[:, 1]
            + data.F_ext[:, 2] * u_phys[:, 2]
        ).sum()
        return W


# ════════════════════════════════════════════════
# VERIFICATION
# ════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    from pathlib import Path

    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 70)
    print("  ENERGY LOSS VERIFICATION (T^T K T Form)")
    print("=" * 70)

    data_list = torch.load(
        "DATA/graph_dataset.pt", weights_only=False
    )

    from normalizer import PhysicsScaler
    if not hasattr(data_list[0], 'u_c'):
        data_list = PhysicsScaler.compute_and_store_list(
            data_list
        )

    loss_fn = FrameEnergyLoss()

    # ══════════════════════════════════════════
    # TEST 1: Analytical cantilever (horizontal)
    # ══════════════════════════════════════════
    print(f"\n  TEST 1a: Horizontal cantilever")
    from torch_geometric.data import Data

    L_t = 10.0
    E_t = 2.1e11
    A_t = 0.01
    I_t = 8.333e-6
    P_t = -1000.0

    w_tip = P_t * L_t**3 / (3 * E_t * I_t)
    th_tip = P_t * L_t**2 / (2 * E_t * I_t)
    U_exact = abs(P_t) * abs(w_tip) / 2
    W_exact = P_t * w_tip  # = P * (P*L³/3EI) = P²L³/3EI > 0

    test_h = Data(
        connectivity=torch.tensor([[0, 1]]),
        coords=torch.tensor(
            [[0, 0, 0], [L_t, 0, 0]],
            dtype=torch.float32
        ),
        elem_lengths=torch.tensor([L_t],
                                   dtype=torch.float32),
        elem_directions=torch.tensor(
            [[1, 0, 0]], dtype=torch.float32
        ),
        prop_E=torch.tensor([E_t],
                            dtype=torch.float32),
        prop_A=torch.tensor([A_t],
                            dtype=torch.float32),
        prop_I22=torch.tensor([I_t],
                              dtype=torch.float32),
        F_ext=torch.tensor(
            [[0, 0, 0], [0, P_t, 0]],
            dtype=torch.float32
        ),
        F_c=torch.tensor(abs(P_t)),
        u_c=torch.tensor(abs(w_tip)),
        theta_c=torch.tensor(abs(th_tip)),
    )

    u_h = torch.tensor(
        [[0, 0, 0], [0, w_tip, th_tip]],
        dtype=torch.float32
    )

    U_h = loss_fn._strain_energy(u_h, test_h)
    W_h = loss_fn._external_work(u_h, test_h)
    Pi_h = U_h - W_h

    print(f"    Exact:    U={U_exact:.6e}, "
          f"W={W_exact:.6e}")
    print(f"    Computed: U={U_h.item():.6e}, "
          f"W={W_h.item():.6e}")
    print(f"    U/|W| = {(U_h/abs(W_h)).item():.6f} "
          f"(should be 0.5)")
    U_err_h = abs(U_h.item() - U_exact) / U_exact
    print(f"    U error: {U_err_h:.2e} "
          f"{'✓' if U_err_h < 1e-4 else '✗'}")

    # ══════════════════════════════════════════
    # TEST 1b: VERTICAL cantilever (the critical test)
    # ══════════════════════════════════════════
    print(f"\n  TEST 1b: Vertical cantilever")
    print(f"    (Column with horizontal tip load)")

    # Vertical beam: base at (0,0,0), tip at (0,0,L)
    # Horizontal load Fx at tip
    P_v = 1000.0  # horizontal
    w_tip_v = P_v * L_t**3 / (3 * E_t * I_t)
    th_tip_v = P_v * L_t**2 / (2 * E_t * I_t)
    U_exact_v = abs(P_v) * abs(w_tip_v) / 2

    test_v = Data(
        connectivity=torch.tensor([[0, 1]]),
        coords=torch.tensor(
            [[0, 0, 0], [0, 0, L_t]],
            dtype=torch.float32
        ),
        elem_lengths=torch.tensor([L_t],
                                   dtype=torch.float32),
        elem_directions=torch.tensor(
            [[0, 0, 1]], dtype=torch.float32
        ),
        prop_E=torch.tensor([E_t],
                            dtype=torch.float32),
        prop_A=torch.tensor([A_t],
                            dtype=torch.float32),
        prop_I22=torch.tensor([I_t],
                              dtype=torch.float32),
        F_ext=torch.tensor(
            [[0, 0, 0], [P_v, 0, 0]],
            dtype=torch.float32
        ),
        F_c=torch.tensor(abs(P_v)),
        u_c=torch.tensor(abs(w_tip_v)),
        theta_c=torch.tensor(abs(th_tip_v)),
    )

    # For vertical column with Fx at tip:
    # Global: ux = w_tip_v (horizontal deflection)
    #         uz = 0 (no vertical displacement for 
    #              inextensible)
    #         θ = -th_tip_v (rotation, sign depends 
    #             on convention)
    # Let's try both signs

    for th_sign, label in [(+1, "θ positive"),
                           (-1, "θ negative")]:
        u_v = torch.tensor(
            [[0, 0, 0],
             [w_tip_v, 0, th_sign * th_tip_v]],
            dtype=torch.float32
        )

        U_v = loss_fn._strain_energy(u_v, test_v)
        W_v = loss_fn._external_work(u_v, test_v)
        Pi_v = U_v - W_v

        U_err_v = abs(U_v.item() - U_exact_v) / U_exact_v
        UW_ratio = (U_v / abs(W_v)).item()

        print(f"\n    {label}:")
        print(f"      u = [ux={w_tip_v:.4e}, "
              f"uz=0, θ={th_sign*th_tip_v:.4e}]")
        print(f"      U = {U_v.item():.6e} "
              f"(exact: {U_exact_v:.6e})")
        print(f"      W = {W_v.item():.6e}")
        print(f"      Π = {Pi_v.item():.6e}")
        print(f"      U/|W| = {UW_ratio:.6f}")
        print(f"      U error: {U_err_v:.2e} "
              f"{'✓' if U_err_v < 1e-3 else '✗'}")

        # Check d_local
        c_v = test_v.elem_directions[0, 0]
        s_v = test_v.elem_directions[0, 2]
        ux_B = u_v[1, 0]
        uz_B = u_v[1, 1]
        th_B = u_v[1, 2]

        u_loc = c_v * ux_B + s_v * uz_B
        w_loc = -s_v * ux_B + c_v * uz_B

        print(f"      d_local: u={u_loc.item():.4e}, "
              f"w={w_loc.item():.4e}, "
              f"θ={th_B.item():.4e}")

    # ══════════════════════════════════════════
    # TEST 2: Kratos data with new formulation
    # ══════════════════════════════════════════
    print(f"\n\n  TEST 2: Kratos frame data")

    for i in range(min(3, len(data_list))):
        data = data_list[i]
        u_true = data.y_node.clone()

        U_true = loss_fn._strain_energy(u_true, data)
        W_true = loss_fn._external_work(u_true, data)
        Pi_true = U_true - W_true

        u_zero = torch.zeros_like(u_true)
        Pi_zero = (loss_fn._strain_energy(u_zero, data)
                 - loss_fn._external_work(u_zero, data))

        UW = (U_true / W_true.abs().clamp(
            min=1e-30
        )).item()

        print(f"\n  Case {i}:")
        print(f"    U = {U_true.item():.6e}")
        print(f"    W = {W_true.item():.6e}")
        print(f"    Π = {Pi_true.item():.6e}")
        print(f"    U/|W| = {UW:.6f} "
              f"(should be ~0.5)")
        print(f"    Π(0) = {Pi_zero.item():.6e}")

        if Pi_true < Pi_zero:
            print(f"    ✓ True minimizes energy")
        else:
            print(f"    ✗ True has HIGHER energy!")
            print(f"    → rotation sign issue likely")

    # ══════════════════════════════════════════
    # TEST 3: Energy landscape
    # ══════════════════════════════════════════
    print(f"\n\n  TEST 3: Energy along u_true")
    data = data_list[0]
    u_true = data.y_node.clone()

    print(f"    α      Π              U/|W|")
    alphas = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    Pi_vals = []
    for alpha in alphas:
        u_a = alpha * u_true
        U_a = loss_fn._strain_energy(u_a, data)
        W_a = loss_fn._external_work(u_a, data)
        Pi_a = (U_a - W_a).item()
        UW_a = (U_a / W_a.abs().clamp(
            min=1e-30
        )).item()
        Pi_vals.append(Pi_a)
        marker = " ← should be min" if alpha == 1.0 else ""
        print(f"    {alpha:.2f}   {Pi_a:14.6e}   "
              f"{UW_a:.4f}{marker}")

    # Find minimum
    min_idx = Pi_vals.index(min(Pi_vals))
    print(f"\n    Minimum at α={alphas[min_idx]:.2f} "
          f"({'✓ CORRECT' if min_idx == alphas.index(1.0) else '✗ WRONG'})")

    print(f"\n  DONE ✓")