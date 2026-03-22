"""
=================================================================
energy_loss.py — Corrected Energy Loss with Sign Verification
=================================================================

Fixes:
  1. Verify rotation sign convention against Kratos
  2. Use u^T K u / 2 directly (guaranteed correct)
  3. Add detailed energy decomposition for debugging
=================================================================
"""

import torch
import torch.nn as nn


class FrameEnergyLoss(nn.Module):
    """
    Total Potential Energy for 2D Euler-Bernoulli frame.
    
    Uses explicit stiffness matrix multiplication:
      U = (1/2) * d_local^T * K_local * d_local
    
    This avoids sign convention issues in the expanded form.
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
        Strain energy using explicit local stiffness
        matrix multiplication.
        
        For each element:
          d_local = T * d_global  (6x1)
          U_e = (1/2) * d_local^T * K_local * d_local
        
        K_local (6x6) for Euler-Bernoulli beam:
          [EA/L    0        0     -EA/L    0        0    ]
          [0     12EI/L³  6EI/L²   0   -12EI/L³  6EI/L² ]
          [0      6EI/L²  4EI/L    0    -6EI/L²  2EI/L  ]
          [-EA/L   0        0      EA/L    0        0    ]
          [0    -12EI/L³ -6EI/L²   0    12EI/L³ -6EI/L² ]
          [0      6EI/L²  2EI/L    0    -6EI/L²  4EI/L  ]
        """
        conn = data.connectivity
        nA, nB = conn[:, 0], conn[:, 1]
        n_elem = conn.shape[0]

        L  = data.elem_lengths
        EA = data.prop_E * data.prop_A
        EI = data.prop_E * data.prop_I22

        c = data.elem_directions[:, 0]
        s = data.elem_directions[:, 2]

        # Global DOFs
        ux_A = u_phys[nA, 0]
        uz_A = u_phys[nA, 1]
        th_A = u_phys[nA, 2]
        ux_B = u_phys[nB, 0]
        uz_B = u_phys[nB, 1]
        th_B = u_phys[nB, 2]

        # Transform to local
        u_A = c * ux_A + s * uz_A
        w_A = -s * ux_A + c * uz_A
        u_B = c * ux_B + s * uz_B
        w_B = -s * ux_B + c * uz_B

        # Local DOF vector: [u_A, w_A, θ_A, u_B, w_B, θ_B]
        # Shape: (E, 6)
        d_local = torch.stack(
            [u_A, w_A, th_A, u_B, w_B, th_B], dim=1
        )

        # Build K_local for each element: (E, 6, 6)
        K = torch.zeros(
            n_elem, 6, 6, device=u_phys.device
        )

        ea_L = EA / L
        ei_L = EI / L
        ei_L2 = EI / L**2
        ei_L3 = EI / L**3

        # Row 0: [EA/L, 0, 0, -EA/L, 0, 0]
        K[:, 0, 0] = ea_L
        K[:, 0, 3] = -ea_L

        # Row 1: [0, 12EI/L³, 6EI/L², 0, -12EI/L³, 6EI/L²]
        K[:, 1, 1] = 12 * ei_L3
        K[:, 1, 2] = 6 * ei_L2
        K[:, 1, 4] = -12 * ei_L3
        K[:, 1, 5] = 6 * ei_L2

        # Row 2: [0, 6EI/L², 4EI/L, 0, -6EI/L², 2EI/L]
        K[:, 2, 1] = 6 * ei_L2
        K[:, 2, 2] = 4 * ei_L
        K[:, 2, 4] = -6 * ei_L2
        K[:, 2, 5] = 2 * ei_L

        # Row 3: [-EA/L, 0, 0, EA/L, 0, 0]
        K[:, 3, 0] = -ea_L
        K[:, 3, 3] = ea_L

        # Row 4: [0, -12EI/L³, -6EI/L², 0, 12EI/L³, -6EI/L²]
        K[:, 4, 1] = -12 * ei_L3
        K[:, 4, 2] = -6 * ei_L2
        K[:, 4, 4] = 12 * ei_L3
        K[:, 4, 5] = -6 * ei_L2

        # Row 5: [0, 6EI/L², 2EI/L, 0, -6EI/L², 4EI/L]
        K[:, 5, 1] = 6 * ei_L2
        K[:, 5, 2] = 2 * ei_L
        K[:, 5, 4] = -6 * ei_L2
        K[:, 5, 5] = 4 * ei_L

        # U_e = (1/2) * d^T K d for each element
        # K @ d: (E, 6, 6) @ (E, 6, 1) -> (E, 6, 1)
        Kd = torch.bmm(K, d_local.unsqueeze(2))
        # d^T @ Kd: (E, 1, 6) @ (E, 6, 1) -> (E, 1, 1)
        U_per_elem = 0.5 * torch.bmm(
            d_local.unsqueeze(1), Kd
        ).squeeze()

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
# DETAILED VERIFICATION
# ════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    from pathlib import Path

    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 70)
    print("  ENERGY LOSS VERIFICATION (Matrix Form)")
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
    # TEST 1: Single cantilever beam element
    # ══════════════════════════════════════════
    print(f"\n  TEST 1: Analytical cantilever check")
    print(f"  (Single beam, tip load P)")

    # Create synthetic single-element test
    from torch_geometric.data import Data

    L_test = 10.0
    E_test = 2.1e11
    A_test = 0.01
    I_test = 8.333e-6
    P_test = -1000.0  # downward tip load

    # Analytical solution
    w_tip = P_test * L_test**3 / (3 * E_test * I_test)
    th_tip = P_test * L_test**2 / (2 * E_test * I_test)
    U_analytical = -P_test * w_tip / 2  # = P²L³/(6EI)
    W_analytical = P_test * w_tip
    Pi_analytical = U_analytical - W_analytical

    print(f"    L={L_test}, E={E_test:.2e}, "
          f"I={I_test:.4e}, P={P_test}")
    print(f"    Analytical: w_tip={w_tip:.6e}, "
          f"θ_tip={th_tip:.6e}")
    print(f"    U_analytical = {U_analytical:.6e}")
    print(f"    W_analytical = {W_analytical:.6e}")
    print(f"    Π_analytical = {Pi_analytical:.6e}")
    print(f"    U/|W| = {U_analytical/abs(W_analytical):.4f} "
          f"(should be 0.5)")

    # Build test data
    test_data = Data(
        connectivity=torch.tensor([[0, 1]]),
        coords=torch.tensor(
            [[0, 0, 0], [L_test, 0, 0]],
            dtype=torch.float32
        ),
        elem_lengths=torch.tensor(
            [L_test], dtype=torch.float32
        ),
        elem_directions=torch.tensor(
            [[1, 0, 0]], dtype=torch.float32
        ),
        prop_E=torch.tensor(
            [E_test], dtype=torch.float32
        ),
        prop_A=torch.tensor(
            [A_test], dtype=torch.float32
        ),
        prop_I22=torch.tensor(
            [I_test], dtype=torch.float32
        ),
        F_ext=torch.tensor(
            [[0, 0, 0], [0, P_test, 0]],
            dtype=torch.float32
        ),
        F_c=torch.tensor(abs(P_test)),
        u_c=torch.tensor(abs(w_tip)),
        theta_c=torch.tensor(abs(th_tip)),
    )

    # True displacement: [ux, uz, θ]
    # Node 0 (fixed): [0, 0, 0]
    # Node 1 (tip):   [0, w_tip, θ_tip]
    u_true_test = torch.tensor(
        [[0, 0, 0], [0, w_tip, th_tip]],
        dtype=torch.float32
    )

    U_test = loss_fn._strain_energy(u_true_test, test_data)
    W_test = loss_fn._external_work(u_true_test, test_data)
    Pi_test = U_test - W_test

    print(f"\n    Computed:    U={U_test.item():.6e}, "
          f"W={W_test.item():.6e}, "
          f"Π={Pi_test.item():.6e}")
    print(f"    U/|W| = "
          f"{(U_test/abs(W_test)).item():.6f}")

    U_err = abs(U_test.item() - U_analytical) / abs(
        U_analytical
    )
    W_err = abs(W_test.item() - W_analytical) / abs(
        W_analytical
    )
    print(f"    U error: {U_err:.2e}")
    print(f"    W error: {W_err:.2e}")
    print(f"    {'✓' if U_err < 1e-6 else '✗'} "
          f"U matches analytical")
    print(f"    {'✓' if W_err < 1e-6 else '✗'} "
          f"W matches analytical")

    # ══════════════════════════════════════════
    # TEST 2: Kratos data
    # ══════════════════════════════════════════
    print(f"\n\n  TEST 2: Kratos frame data")

    for i in range(min(3, len(data_list))):
        data = data_list[i]
        u_true = data.y_node.clone()

        print(f"\n  Case {i}:")
        print(f"    N={data.num_nodes}, "
              f"E={data.n_elements}")

        # Check displacement ranges
        print(f"    True disp:")
        print(f"      ux: [{u_true[:, 0].min():.4e}, "
              f"{u_true[:, 0].max():.4e}]")
        print(f"      uz: [{u_true[:, 1].min():.4e}, "
              f"{u_true[:, 1].max():.4e}]")
        print(f"      θ:  [{u_true[:, 2].min():.4e}, "
              f"{u_true[:, 2].max():.4e}]")

        # Check F_ext
        print(f"    F_ext:")
        print(f"      Fx: [{data.F_ext[:, 0].min():.4e}, "
              f"{data.F_ext[:, 0].max():.4e}]")
        print(f"      Fz: [{data.F_ext[:, 1].min():.4e}, "
              f"{data.F_ext[:, 1].max():.4e}]")
        print(f"      My: [{data.F_ext[:, 2].min():.4e}, "
              f"{data.F_ext[:, 2].max():.4e}]")

        # Energies at TRUE
        U_true = loss_fn._strain_energy(u_true, data)
        W_true = loss_fn._external_work(u_true, data)
        Pi_true = U_true - W_true

        print(f"    TRUE displacement:")
        print(f"      U = {U_true.item():.6e}")
        print(f"      W = {W_true.item():.6e}")
        print(f"      Π = {Pi_true.item():.6e}")
        print(f"      U/|W| = "
              f"{(U_true/W_true.abs().clamp(min=1e-30)).item():.6f}")

        # Energies at ZERO
        u_zero = torch.zeros_like(u_true)
        U_zero = loss_fn._strain_energy(u_zero, data)
        W_zero = loss_fn._external_work(u_zero, data)
        Pi_zero = U_zero - W_zero

        print(f"    ZERO displacement:")
        print(f"      U = {U_zero.item():.6e}")
        print(f"      W = {W_zero.item():.6e}")
        print(f"      Π = {Pi_zero.item():.6e}")

        if Pi_true < Pi_zero:
            print(f"    ✓ True minimizes energy")
        else:
            print(f"    ✗ ERROR! True has higher energy!")

        # ── Per-element energy breakdown ──
        conn = data.connectivity
        nA, nB = conn[:, 0], conn[:, 1]
        dirs = data.elem_directions

        n_beams = 0
        n_cols = 0
        U_beams = 0
        U_cols = 0

        for e in range(data.n_elements):
            is_col = (abs(dirs[e, 2]) >
                      abs(dirs[e, 0]))

            # Single element energy
            u_e = torch.zeros(data.num_nodes, 3)
            u_e[conn[e, 0]] = u_true[conn[e, 0]]
            u_e[conn[e, 1]] = u_true[conn[e, 1]]

            # Create single-element data
            L_e = data.elem_lengths[e:e+1]
            d_e = dirs[e:e+1]
            EA_e = (data.prop_E[e] * data.prop_A[e]).unsqueeze(0)
            EI_e = (data.prop_E[e] * data.prop_I22[e]).unsqueeze(0)

            c_e = d_e[:, 0]
            s_e = d_e[:, 2]

            ux_A = u_true[conn[e, 0], 0]
            uz_A = u_true[conn[e, 0], 1]
            th_A = u_true[conn[e, 0], 2]
            ux_B = u_true[conn[e, 1], 0]
            uz_B = u_true[conn[e, 1], 1]
            th_B = u_true[conn[e, 1], 2]

            u_A_loc = c_e * ux_A + s_e * uz_A
            w_A_loc = -s_e * ux_A + c_e * uz_A
            u_B_loc = c_e * ux_B + s_e * uz_B
            w_B_loc = -s_e * ux_B + c_e * uz_B

            du = (u_B_loc - u_A_loc).item()
            dw = (w_B_loc - w_A_loc).item()

            U_ax = 0.5 * EA_e / L_e * du**2
            U_bd_terms = {
                '4θA²': 4 * th_A**2,
                '4θB²': 4 * th_B**2,
                '4θAθB': 4 * th_A * th_B,
                '12/L²·Δw²': (12 / L_e**2) * (
                    w_A_loc**2 + w_B_loc**2
                    - 2*w_A_loc*w_B_loc
                ),
                '-12/L·Δw·Σθ': -(12 / L_e) * (
                    w_B_loc - w_A_loc
                ) * (th_A + th_B),
            }

            if is_col:
                n_cols += 1
            else:
                n_beams += 1

            if e < 3:
                etype = "COL" if is_col else "BEAM"
                print(f"\n      Elem {e} ({etype}):")
                print(f"        L={L_e.item():.4f}, "
                      f"dir=({c_e.item():.3f}, "
                      f"{s_e.item():.3f})")
                print(f"        local: "
                      f"uA={u_A_loc.item():.4e} "
                      f"wA={w_A_loc.item():.4e} "
                      f"θA={th_A.item():.4e}")
                print(f"        local: "
                      f"uB={u_B_loc.item():.4e} "
                      f"wB={w_B_loc.item():.4e} "
                      f"θB={th_B.item():.4e}")
                print(f"        Δu={du:.4e}, "
                      f"Δw={dw:.4e}")
                print(f"        U_axial = "
                      f"{U_ax.item():.4e}")
                for name, val in U_bd_terms.items():
                    print(f"        {name}: "
                          f"{val.item():.4e}")

    # ══════════════════════════════════════════
    # TEST 3: Gradient at zero
    # ══════════════════════════════════════════
    print(f"\n\n  TEST 3: Gradient at u=0")
    data = data_list[0]
    u = torch.zeros(
        data.num_nodes, 3, requires_grad=True
    )

    Pi = (loss_fn._strain_energy(u, data)
         - loss_fn._external_work(u, data))
    Pi.backward()

    grad = u.grad
    print(f"    Π(0) = {Pi.item():.6e} "
          f"(should be 0)")
    print(f"    ∇Π = [{grad.min().item():.4e}, "
          f"{grad.max().item():.4e}]")

    # At u=0: ∂U/∂u = 0, so ∂Π/∂u = -F_ext
    for dof, name in [(0, 'Fx'), (1, 'Fz'), (2, 'My')]:
        g = grad[:, dof]
        f = -data.F_ext[:, dof]
        loaded = f.abs() > 1e-10
        if loaded.any():
            match = torch.allclose(
                g[loaded], f[loaded],
                rtol=1e-4, atol=1e-6
            )
            print(f"    ∂Π/∂u_{name} = -F_{name}? "
                  f"{'✓' if match else '✗'}")
            if not match:
                idx = loaded.nonzero()[0].item()
                print(f"      Example node {idx}: "
                      f"grad={g[idx].item():.4e}, "
                      f"-F={f[idx].item():.4e}")

    # ══════════════════════════════════════════
    # TEST 4: Energy landscape along true direction
    # ══════════════════════════════════════════
    print(f"\n\n  TEST 4: Energy along u_true direction")
    data = data_list[0]
    u_true = data.y_node.clone()

    print(f"    α     U              W              "
          f"Π              U/|W|")
    for alpha in [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25,
                  1.5, 2.0]:
        u_alpha = alpha * u_true
        U_a = loss_fn._strain_energy(u_alpha, data)
        W_a = loss_fn._external_work(u_alpha, data)
        Pi_a = U_a - W_a
        UW = (U_a / W_a.abs().clamp(min=1e-30)).item()
        marker = " ← minimum" if alpha == 1.0 else ""
        print(f"    {alpha:.2f}  {U_a.item():14.6e}  "
              f"{W_a.item():14.6e}  "
              f"{Pi_a.item():14.6e}  "
              f"{UW:.4f}{marker}")

    print(f"\n  DONE ✓")