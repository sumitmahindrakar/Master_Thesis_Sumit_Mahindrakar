"""
=================================================================
corotational.py — Fully Non-dimensionalized Beam Element
=================================================================

Everything computed in non-dimensional form:
  - Displacements: u/u_c, θ/θ_c
  - Forces: F/F_c, M/M_c  
  - Stiffness: (EA·u_c)/(F_c·L), (EI·θ_c)/(M_c·L)

This keeps all intermediate values ~O(1) and prevents
gradient explosion.

Residual output: (ΣF - F_ext) / F_c  → O(1) directly
=================================================================
"""

import torch
import torch.nn as nn


class CorotationalBeam2D(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred_raw, data):
        """
        Args:
            pred_raw: (N, 3) network output ~O(1)
                      [ux/u_c, uz/u_c, θ/θ_c]
            data:     PyG Data object

        Returns:
            result dict with:
              - nodal_forces_nd (N, 3): [Fx/F_c, Fz/F_c, My/M_c]
              - F_ext_nd (N, 3): [Fx_ext/F_c, Fz_ext/F_c, My_ext/M_c]
              - phys_disp (N, 3): physical displacements
              - internal forces in physical units
        """
        conn   = data.connectivity
        coords = data.coords
        prop_E = data.prop_E
        prop_A = data.prop_A
        prop_I = data.prop_I22

        E_count = conn.shape[0]
        N_count = pred_raw.shape[0]
        device  = pred_raw.device

        nA = conn[:, 0]
        nB = conn[:, 1]

        # ════════════════════════════════════════════
        # Scales
        # ════════════════════════════════════════════

        u_c     = data.u_c
        theta_c = data.theta_c
        F_c     = data.F_c
        M_c     = data.M_c

        # ════════════════════════════════════════════
        # Physical displacements (for validation)
        # ════════════════════════════════════════════

        phys_disp = torch.zeros_like(pred_raw)
        phys_disp[:, 0] = pred_raw[:, 0] * u_c
        phys_disp[:, 1] = pred_raw[:, 1] * u_c
        phys_disp[:, 2] = pred_raw[:, 2] * theta_c

        # ════════════════════════════════════════════
        # STEP 1: Geometry
        # ════════════════════════════════════════════

        dx0 = coords[nB, 0] - coords[nA, 0]
        dz0 = coords[nB, 2] - coords[nA, 2]
        l0  = torch.sqrt(dx0**2 + dz0**2)
        c   = dx0 / l0
        s   = dz0 / l0

        EA = prop_E * prop_A
        EI = prop_E * prop_I

        # ════════════════════════════════════════════
        # STEP 2: Non-dim stiffness coefficients
        # ════════════════════════════════════════════
        #
        # Standard: f = (EA/L) * u
        # Non-dim:  f/F_c = (EA·u_c)/(F_c·L) * (u/u_c)
        #
        # So the non-dim axial stiffness is:
        #   k_axial_nd = EA * u_c / (F_c * L)
        #
        # For bending: M = (EI/L) * θ
        # Non-dim: M/M_c = (EI·θ_c)/(M_c·L) * (θ/θ_c)
        #
        # Shear couples: f = (EI/L²) * θ
        # Non-dim: f/F_c = (EI·θ_c)/(F_c·L²) * (θ/θ_c)
        #
        # Transverse: f = (EI/L³) * w
        # Non-dim: f/F_c = (EI·u_c)/(F_c·L³) * (w/u_c)

        k_ax   = EA * u_c / (F_c * l0)          # axial
        k_bend = EI * theta_c / (M_c * l0)      # moment-rotation
        k_sw   = EI * theta_c / (F_c * l0**2)   # shear-rotation coupling
        k_tr   = EI * u_c / (F_c * l0**3)       # shear-transverse

        # ════════════════════════════════════════════
        # STEP 3: Gather non-dim DOFs with -θ
        # ════════════════════════════════════════════

        d_nd = torch.stack([
            pred_raw[nA, 0],         # ux_A / u_c
            pred_raw[nA, 1],         # uz_A / u_c
            -pred_raw[nA, 2],        # -θ_A / θ_c
            pred_raw[nB, 0],         # ux_B / u_c
            pred_raw[nB, 1],         # uz_B / u_c
            -pred_raw[nB, 2],        # -θ_B / θ_c
        ], dim=1)

        # ════════════════════════════════════════════
        # STEP 4: Transform to local
        # ════════════════════════════════════════════

        dl = torch.zeros_like(d_nd)

        dl[:, 0] =  c * d_nd[:, 0] + s * d_nd[:, 1]
        dl[:, 1] = -s * d_nd[:, 0] + c * d_nd[:, 1]
        dl[:, 2] =  d_nd[:, 2]

        dl[:, 3] =  c * d_nd[:, 3] + s * d_nd[:, 4]
        dl[:, 4] = -s * d_nd[:, 3] + c * d_nd[:, 4]
        dl[:, 5] =  d_nd[:, 5]

        ua = dl[:, 0]
        wa = dl[:, 1]
        ta = dl[:, 2]
        ub = dl[:, 3]
        wb = dl[:, 4]
        tb = dl[:, 5]

        # ════════════════════════════════════════════
        # STEP 5: Non-dim local forces
        # f_nd = K_nd · d_nd
        #
        # f0, f3: axial (F/F_c)
        # f1, f4: shear (F/F_c)
        # f2, f5: moment (M/M_c)
        # ════════════════════════════════════════════

        # Axial: f/F_c = k_ax * (u/u_c)
        f0 = k_ax * (ua - ub)
        f3 = k_ax * (ub - ua)

        # Shear: f/F_c = 12*k_tr*(w/u_c) + 6*k_sw*(θ/θ_c)
        f1 = (12 * k_tr * (wa - wb)
            + 6 * k_sw * (ta + tb))
        f4 = (12 * k_tr * (wb - wa)
            - 6 * k_sw * (ta + tb))

        # Moment: M/M_c = 6*k_sw*(F_c/M_c)*(w/u_c) 
        #                + k_bend*(4ta + 2tb)
        # But 6*EI*u_c/(F_c*L²) * (F_c/M_c)
        #   = 6*EI*u_c/(M_c*L²)
        k_mw = EI * u_c / (M_c * l0**2)   # moment-transverse

        f2 = (6 * k_mw * (wa - wb)
            + k_bend * (4 * ta + 2 * tb))
        f5 = (6 * k_mw * (wa - wb)
            + k_bend * (2 * ta + 4 * tb))

        # ════════════════════════════════════════════
        # STEP 6: Transform to global (non-dim)
        #
        # f_global[0,1]: F/F_c
        # f_global[2]:   M/M_c
        # f_global[3,4]: F/F_c
        # f_global[5]:   M/M_c
        # ════════════════════════════════════════════

        fg = torch.zeros(E_count, 6, device=device)

        fg[:, 0] = c * f0 - s * f1
        fg[:, 1] = s * f0 + c * f1
        fg[:, 2] = f2

        fg[:, 3] = c * f3 - s * f4
        fg[:, 4] = s * f3 + c * f4
        fg[:, 5] = f5

        # ════════════════════════════════════════════
        # STEP 7: Assemble (non-dim)
        # ════════════════════════════════════════════

        nodal_forces_nd = torch.zeros(
            N_count, 3, device=device
        )
        nodal_forces_nd.scatter_add_(
            0,
            nA.unsqueeze(1).expand(-1, 3),
            fg[:, 0:3]
        )
        nodal_forces_nd.scatter_add_(
            0,
            nB.unsqueeze(1).expand(-1, 3),
            fg[:, 3:6]
        )

        # ════════════════════════════════════════════
        # Non-dim external forces
        # ════════════════════════════════════════════

        F_ext_nd = torch.zeros(
            N_count, 3, device=device
        )
        F_ext_nd[:, 0] = data.F_ext[:, 0] / F_c
        F_ext_nd[:, 1] = data.F_ext[:, 1] / F_c
        F_ext_nd[:, 2] = data.F_ext[:, 2] / M_c

        # ════════════════════════════════════════════
        # Physical internal forces (for monitoring)
        # ════════════════════════════════════════════

        N_e  = f3 * F_c
        M1_e = f2 * M_c
        M2_e = f5 * M_c
        V_e  = f4 * F_c

        result = {
            # Non-dim (for loss)
            'nodal_forces_nd': nodal_forces_nd,
            'F_ext_nd':        F_ext_nd,

            # Physical (for monitoring)
            'N_e':        N_e,
            'M1_e':       M1_e,
            'M2_e':       M2_e,
            'V_e':        V_e,
            'phys_disp':  phys_disp,

            # Debug
            'l0':         l0,
            'cos0':       c,
            'sin0':       s,
        }

        return result


# ════════════════════════════════════════════════════════
# VERIFICATION
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    from pathlib import Path
    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 70)
    print("  NON-DIM COROTATIONAL VERIFICATION")
    print("=" * 70)

    data_list = torch.load(
        "DATA/graph_dataset.pt", weights_only=False
    )

    from normalizer import PhysicsScaler
    if not hasattr(data_list[0], 'u_c'):
        data_list = (
            PhysicsScaler.compute_and_store_list(data_list)
        )

    beam = CorotationalBeam2D()

    for i in range(min(3, len(data_list))):
        data = data_list[i]
        true_disp = data.y_node.clone()

        # Convert to non-dim
        pred_raw = torch.zeros_like(true_disp)
        pred_raw[:, 0] = true_disp[:, 0] / data.u_c
        pred_raw[:, 1] = true_disp[:, 1] / data.u_c
        pred_raw[:, 2] = true_disp[:, 2] / data.theta_c

        print(f"\n  Case {i}:")
        print(f"    Non-dim range: "
              f"[{pred_raw.min():.4f}, "
              f"{pred_raw.max():.4f}]")

        result = beam(pred_raw, data)

        # Equilibrium in non-dim
        res_nd = (result['nodal_forces_nd']
                - result['F_ext_nd'])
        free = data.bc_disp.squeeze(-1) < 0.5

        if free.any():
            res_free = res_nd[free]
            max_res = res_free.abs().max().item()

            print(f"    Max|R_nd|  = {max_res:.6e}")
            print(f"    (should be ~0 for correct "
                  f"displacements)")

            status = ("✓ GOOD" if max_res < 0.1 else
                      "~ OK" if max_res < 1.0 else
                      "✗ CHECK")
            print(f"    Status: {status}")

        # Roundtrip check
        phys_err = (result['phys_disp'] - true_disp
                    ).abs().max().item()
        print(f"    Phys roundtrip: {phys_err:.6e}")

    # Gradient test
    print(f"\n  Gradient test:")
    data = data_list[0]
    true_disp = data.y_node.clone()
    pred_raw = torch.zeros_like(true_disp)
    pred_raw[:, 0] = true_disp[:, 0] / data.u_c
    pred_raw[:, 1] = true_disp[:, 1] / data.u_c
    pred_raw[:, 2] = true_disp[:, 2] / data.theta_c
    pred_raw.requires_grad_(True)

    result = beam(pred_raw, data)
    res = (result['nodal_forces_nd']
         - result['F_ext_nd'])
    free = data.bc_disp.squeeze(-1) < 0.5
    loss = res[free].pow(2).mean()
    grad = torch.autograd.grad(loss, pred_raw)[0]
    print(f"    Loss value: {loss:.6e}")
    print(f"    Grad range: [{grad.min():.4e}, "
          f"{grad.max():.4e}]")
    print(f"    Grad norm:  {grad.norm():.4e}")
    print(f"    (should be ~O(1))")
    print(f"    ✓ Differentiable")

    print(f"\n  DONE ✓")