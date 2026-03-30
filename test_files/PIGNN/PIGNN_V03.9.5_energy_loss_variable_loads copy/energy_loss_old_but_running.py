"""
=================================================================
energy_loss.py — Total Potential Energy Loss for 2D Frames
=================================================================

Adapted from Dalton et al. PI-GNN approach:
  Loss = Π = U_internal - W_external

At equilibrium: ∂Π/∂u = 0 (minimum of energy)
At u=0: Π = 0, but ∂Π/∂u = -F_ext ≠ 0 → gradient pushes away!

No zero attractor. No normalization needed.
=================================================================
"""

import torch
import torch.nn as nn


class FrameEnergyLoss(nn.Module):
    """
    Total Potential Energy loss for 2D Euler-Bernoulli frame.
    
    Π = U_strain - W_external
    
    U_strain = Σ_e [U_axial_e + U_bending_e]
    W_external = Σ_i [Fx_i·ux_i + Fz_i·uz_i + My_i·θ_i]
    """

    def __init__(self):
        super().__init__()

    def forward(self, model, data):
        """
        Args:
            model: PIGNN that outputs (N, 3) raw predictions
            data: PyG Data object
        
        Returns:
            Pi_norm: normalized potential energy (scalar loss)
            loss_dict: diagnostics
            pred_raw: raw network output
            u_phys: physical displacements
        """
        # 1. Network prediction (raw, ~O(1) if scaled)
        pred_raw = model(data)  # (N, 3)

        # 2. Convert to physical displacements
        u_phys = torch.zeros_like(pred_raw)
        u_phys[:, 0] = pred_raw[:, 0] * data.u_c   # ux
        u_phys[:, 1] = pred_raw[:, 1] * data.u_c   # uz
        u_phys[:, 2] = pred_raw[:, 2] * data.theta_c  # θy

        # 3. Strain energy
        U_internal = self._strain_energy(u_phys, data)

        # 4. External work
        W_external = self._external_work(u_phys, data)

        # 5. Total potential energy
        Pi = U_internal - W_external

        # 6. Normalize by characteristic energy scale
        E_c = (data.F_c * data.u_c).clamp(min=1e-30)
        Pi_norm = Pi / E_c

        # 7. Diagnostics
        loss_dict = {
            'total':      Pi_norm.item(),
            'Pi':         Pi.item(),
            'Pi_norm':    Pi_norm.item(),
            'U_internal': U_internal.item(),
            'W_external': W_external.item(),
            'U_over_W':   (U_internal / W_external.abs().clamp(min=1e-30)).item(),
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
        Internal strain energy for all beam elements.
        
        Axial:   U_ax = (EA/2L) · (u_B - u_A)²
        Bending: U_bd = (EI/2L) · [4θA² + 4θB² + 4θAθB
                        + (12/L²)(wA² + wB² - 2wAwB)
                        - (12/L)(wB-wA)(θA+θB)]
        """
        conn = data.connectivity
        nA, nB = conn[:, 0], conn[:, 1]

        L  = data.elem_lengths
        EA = data.prop_E * data.prop_A
        EI = data.prop_E * data.prop_I22

        # Direction cosines (element local → global)
        c = data.elem_directions[:, 0]   # cos α
        s = data.elem_directions[:, 2]   # sin α

        # Global DOFs at element ends
        ux_A = u_phys[nA, 0]
        uz_A = u_phys[nA, 1]
        th_A = u_phys[nA, 2]
        ux_B = u_phys[nB, 0]
        uz_B = u_phys[nB, 1]
        th_B = u_phys[nB, 2]

        # Global → Local transformation
        # Local axial:      u_loc =  c·ux + s·uz
        # Local transverse: w_loc = -s·ux + c·uz
        u_A_loc =  c * ux_A + s * uz_A
        w_A_loc = -s * ux_A + c * uz_A
        u_B_loc =  c * ux_B + s * uz_B
        w_B_loc = -s * ux_B + c * uz_B

        # ── Axial strain energy ──
        du = u_B_loc - u_A_loc
        U_axial = 0.5 * (EA / L) * du**2

        # ── Bending strain energy (Hermite) ──
        dw = w_B_loc - w_A_loc

        U_bend = (EI / (2 * L)) * (
            4 * th_A**2
            + 4 * th_B**2
            + 4 * th_A * th_B
            + (12 / L**2) * (w_A_loc**2 + w_B_loc**2
                             - 2 * w_A_loc * w_B_loc)
            - (12 / L) * dw * (th_A + th_B)
        )

        return (U_axial + U_bend).sum()

    def _external_work(self, u_phys, data):
        """
        Work done by external forces.
        
        W = Σ_i (Fx_i·ux_i + Fz_i·uz_i + My_i·θy_i)
        
        F_ext layout: (N, 3) = [Fx, Fz, My]
        u_phys layout: (N, 3) = [ux, uz, θy]
        """
        W = (
            data.F_ext[:, 0] * u_phys[:, 0]    # Fx · ux
            + data.F_ext[:, 1] * u_phys[:, 1]  # Fz · uz
            + data.F_ext[:, 2] * u_phys[:, 2]  # My · θy
        ).sum()

        return W


# ════════════════════════════════════════════════════════
# VERIFICATION
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    from pathlib import Path

    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)

    print("=" * 70)
    print("  ENERGY LOSS VERIFICATION")
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

    for i in range(min(3, len(data_list))):
        data = data_list[i]
        u_true = data.y_node.clone()

        # ── Energy at TRUE solution ──
        U_true = loss_fn._strain_energy(u_true, data)
        W_true = loss_fn._external_work(u_true, data)
        Pi_true = U_true - W_true

        # ── Energy at ZERO ──
        u_zero = torch.zeros_like(u_true)
        U_zero = loss_fn._strain_energy(u_zero, data)
        W_zero = loss_fn._external_work(u_zero, data)
        Pi_zero = U_zero - W_zero

        # ── Energy at 2x TRUE (overshoot) ──
        u_2x = 2.0 * u_true
        U_2x = loss_fn._strain_energy(u_2x, data)
        W_2x = loss_fn._external_work(u_2x, data)
        Pi_2x = U_2x - W_2x

        print(f"\n  Case {i}:")
        print(f"    {'':15} {'U_strain':>14} {'W_external':>14} "
              f"{'Π=U-W':>14}")
        print(f"    {'ZERO':15} {U_zero.item():14.6e} "
              f"{W_zero.item():14.6e} {Pi_zero.item():14.6e}")
        print(f"    {'TRUE':15} {U_true.item():14.6e} "
              f"{W_true.item():14.6e} {Pi_true.item():14.6e}")
        print(f"    {'2x TRUE':15} {U_2x.item():14.6e} "
              f"{W_2x.item():14.6e} {Pi_2x.item():14.6e}")

        if Pi_true < Pi_zero and Pi_true < Pi_2x:
            print(f"    ✓ True solution MINIMIZES energy")
        elif Pi_true < Pi_zero:
            print(f"    ~ True < Zero but True > 2x "
                  f"(check bending signs)")
        else:
            print(f"    ✗ ERROR: energy not minimized "
                  f"at true solution!")

        # ── Theorem check: at minimum, U = W/2 ──
        # For linear elasticity: Π_min = -U = -W/2
        ratio = U_true / W_true.abs().clamp(min=1e-30)
        print(f"    U/|W| = {ratio.item():.4f} "
              f"(should be ~0.5 for linear)")

    # ── Gradient check at u=0 ──
    print(f"\n  Gradient at u=0:")
    data = data_list[0]
    u = torch.zeros(data.num_nodes, 3, requires_grad=True)

    U = loss_fn._strain_energy(u, data)
    W = loss_fn._external_work(u, data)
    Pi = U - W
    Pi.backward()

    grad = u.grad
    print(f"    Π(0) = {Pi.item():.6e}")
    print(f"    U(0) = {U.item():.6e} (should be 0)")
    print(f"    W(0) = {W.item():.6e} (should be 0)")
    print(f"    ∇Π = [{grad.min().item():.4e}, "
          f"{grad.max().item():.4e}]")
    print(f"    |∇Π| = {grad.norm().item():.4e}")

    # At u=0: U=0, W=0, so ∂Π/∂u = ∂U/∂u - F_ext = -F_ext
    # Gradient should equal -F_ext
    F_ext_norm = data.F_ext.norm().item()
    print(f"    |F_ext| = {F_ext_norm:.4e}")
    print(f"    |∇Π|/|F_ext| = "
          f"{grad.norm().item()/max(F_ext_norm,1e-30):.4f} "
          f"(should be ~1.0)")

    # Check gradient direction matches -F_ext
    grad_Fx = grad[:, 0]
    Fext_x = -data.F_ext[:, 0]
    loaded = data.F_ext[:, 0].abs() > 1e-10
    if loaded.any():
        match = torch.allclose(
            grad_Fx[loaded], Fext_x[loaded],
            rtol=1e-4, atol=1e-6
        )
        print(f"    ∂Π/∂ux = -Fx? {match}")

    # ── Step test: gradient descent reduces energy ──
    print(f"\n  Gradient descent test:")
    for step in [1e-6, 1e-4, 1e-2, 1e-1]:
        u_step = (-step * grad / grad.norm()).detach()
        Pi_step = loss_fn._strain_energy(u_step, data) \
                - loss_fn._external_work(u_step, data)
        print(f"    step={step:.0e}: "
              f"Π = {Pi_step.item():12.6e}  "
              f"{'↓' if Pi_step < Pi else '↑'}")

    print(f"\n  DONE ✓")