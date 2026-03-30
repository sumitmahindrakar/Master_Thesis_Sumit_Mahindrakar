"""
=================================================================
energy_loss.py — FIXED: Negate θ for Kratos Sign Convention
=================================================================

Kratos θ_y uses opposite sign to standard E-B stiffness matrix.
Fix: θ_local = -θ_global in strain energy computation.
=================================================================
"""

import torch
import torch.nn as nn


class FrameEnergyLoss(nn.Module):
    """
    Total Potential Energy with Kratos rotation fix.
    
    Π = U_strain - W_external
    
    U uses -θ (corrected sign convention)
    W uses +θ (external moment · rotation, sign cancels)
    """

    def __init__(self):
        super().__init__()

    # def forward(self, model, data):
    #     pred_raw = model(data)

    #     u_phys = torch.zeros_like(pred_raw)
    #     u_phys[:, 0] = pred_raw[:, 0] * data.u_c
    #     u_phys[:, 1] = pred_raw[:, 1] * data.u_c
    #     u_phys[:, 2] = pred_raw[:, 2] * data.theta_c

    #     U_internal = self._strain_energy(u_phys, data)
    #     W_external = self._external_work(u_phys, data)

    #     Pi = U_internal - W_external

    #     E_c = (data.F_c * data.u_c).clamp(min=1e-30)
    #     Pi_norm = Pi / E_c

    #     loss_dict = {
    #         'total':      Pi_norm.item(),
    #         'Pi':         Pi.item(),
    #         'Pi_norm':    Pi_norm.item(),
    #         'U_internal': U_internal.item(),
    #         'W_external': W_external.item(),
    #         'U_over_W':   (
    #             U_internal
    #             / W_external.abs().clamp(min=1e-30)
    #         ).item(),
    #         'ux_range':   [u_phys[:, 0].min().item(),
    #                       u_phys[:, 0].max().item()],
    #         'uz_range':   [u_phys[:, 1].min().item(),
    #                       u_phys[:, 1].max().item()],
    #         'th_range':   [u_phys[:, 2].min().item(),
    #                       u_phys[:, 2].max().item()],
    #         'raw_range':  [pred_raw.min().item(),
    #                       pred_raw.max().item()],
    #     }

    #     return Pi_norm, loss_dict, pred_raw, u_phys
#---------------------------
    # def forward(self, model, data):
    #     pred_raw = model(data)

    #     # ═══════════════════════════════════════
    #     # Handle batched data: expand scales per node
    #     # ═══════════════════════════════════════
    #     if hasattr(data, 'batch') and data.batch is not None:
    #         # Batched: u_c is (B,), need (N_total,)
    #         u_c = data.u_c[data.batch]           # (N,)
    #         theta_c = data.theta_c[data.batch]   # (N,)
    #     else:
    #         # Single graph
    #         u_c = data.u_c
    #         theta_c = data.theta_c

    #     u_phys = torch.zeros_like(pred_raw)
    #     u_phys[:, 0] = pred_raw[:, 0] * u_c
    #     u_phys[:, 1] = pred_raw[:, 1] * u_c
    #     u_phys[:, 2] = pred_raw[:, 2] * theta_c

    #     U_internal = self._strain_energy(u_phys, data)
    #     W_external = self._external_work(u_phys, data)

    #     Pi = U_internal - W_external

    #     # ═══════════════════════════════════════
    #     # Normalize: handle batched F_c, u_c
    #     # ═══════════════════════════════════════
    #     if hasattr(data, 'batch') and data.batch is not None:
    #         # Per-graph normalization, then average
    #         F_c = data.F_c  # (B,)
    #         u_c_graph = data.u_c  # (B,)
    #         E_c = (F_c * u_c_graph).clamp(min=1e-30)
    #         Pi_norm = Pi / E_c.mean()  # average across batch
    #     else:
    #         E_c = (data.F_c * data.u_c).clamp(min=1e-30)
    #         Pi_norm = Pi / E_c

    #     loss_dict = {
    #         'total':      Pi_norm.item(),
    #         'Pi':         Pi.item(),
    #         'Pi_norm':    Pi_norm.item(),
    #         'U_internal': U_internal.item(),
    #         'W_external': W_external.item(),
    #         'U_over_W':   (U_internal
    #                     / W_external.abs().clamp(min=1e-30)
    #                     ).item(),
    #         'ux_range':   [u_phys[:, 0].min().item(),
    #                     u_phys[:, 0].max().item()],
    #         'uz_range':   [u_phys[:, 1].min().item(),
    #                     u_phys[:, 1].max().item()],
    #         'th_range':   [u_phys[:, 2].min().item(),
    #                     u_phys[:, 2].max().item()],
    #         'raw_range':  [pred_raw.min().item(),
    #                     pred_raw.max().item()],
    #     }

    #     return Pi_norm, loss_dict, pred_raw, u_phys
#---------------------------------------------

    # def forward(self, model, data):
    #     pred_raw = model(data)

    #     # ═══════════════════════════════════════
    #     # Handle batched data: expand scales per node
    #     # ═══════════════════════════════════════
    #     if hasattr(data, 'batch') and data.batch is not None:
    #         u_c = data.u_c[data.batch]
    #         theta_c = data.theta_c[data.batch]
    #         # Per-element scales
    #         batch_elem = data.batch[data.connectivity[:, 0]]
    #         F_c_elem = data.F_c[batch_elem]
    #         u_c_elem = data.u_c[batch_elem]
    #         E_c = (F_c_elem * u_c_elem).clamp(min=1e-30)
    #     else:
    #         u_c = data.u_c
    #         theta_c = data.theta_c
    #         E_c = (data.F_c * data.u_c).clamp(min=1e-30)

    #     u_phys = torch.zeros_like(pred_raw)
    #     u_phys[:, 0] = pred_raw[:, 0] * u_c
    #     u_phys[:, 1] = pred_raw[:, 1] * u_c
    #     u_phys[:, 2] = pred_raw[:, 2] * theta_c

    #     # ═══════════════════════════════════════
    #     # Compute per-element energies and normalize
    #     # EACH element's energy by its own scale
    #     # ═══════════════════════════════════════
    #     U_per_elem = self._strain_energy_per_elem(u_phys, data)
    #     W_external = self._external_work(u_phys, data)

    #     U_internal = U_per_elem.sum()
    #     Pi = U_internal - W_external

    #     # Normalize
    #     if hasattr(data, 'batch') and data.batch is not None:
    #         F_c_graph = data.F_c
    #         u_c_graph = data.u_c
    #         # E_c_total = (F_c_graph * u_c_graph).clamp(min=1e-30)
    #         # Pi_norm = Pi / E_c_total.mean()
    #         E_c_per_graph = (F_c_graph * u_c_graph).clamp(min=1e-30)
    #         E_c_total = E_c_per_graph.sum()  # ← sum, not mean!
    #         Pi_norm = Pi / E_c_total
    #     else:
    #         E_c_total = (data.F_c * data.u_c).clamp(min=1e-30)
    #         Pi_norm = Pi / E_c_total

    #     loss_dict = {
    #         'total':      Pi_norm.item(),
    #         'Pi':         Pi.item(),
    #         'Pi_norm':    Pi_norm.item(),
    #         'U_internal': U_internal.item(),
    #         'W_external': W_external.item(),
    #         'U_over_W':   (U_internal
    #                     / W_external.abs().clamp(min=1e-30)
    #                     ).item(),
    #         'ux_range':   [u_phys[:, 0].min().item(),
    #                     u_phys[:, 0].max().item()],
    #         'uz_range':   [u_phys[:, 1].min().item(),
    #                     u_phys[:, 1].max().item()],
    #         'th_range':   [u_phys[:, 2].min().item(),
    #                     u_phys[:, 2].max().item()],
    #         'raw_range':  [pred_raw.min().item(),
    #                     pred_raw.max().item()],
    #     }

    #     return Pi_norm, loss_dict, pred_raw, u_phys



    def forward(self, model, data):
        pred_raw = model(data)  # (N, 3) non-dimensional

        # ═══════════════════════════════════════
        # DON'T convert to physical units!
        # Work directly with non-dimensional predictions.
        # Scale the stiffness and loads instead.
        # ═══════════════════════════════════════
        if hasattr(data, 'batch') and data.batch is not None:
            u_c = data.u_c[data.batch]
            theta_c = data.theta_c[data.batch]
        else:
            u_c = data.u_c
            theta_c = data.theta_c

        # Physical displacements (needed for energy)
        u_phys = torch.zeros_like(pred_raw)
        u_phys[:, 0] = pred_raw[:, 0] * u_c
        u_phys[:, 1] = pred_raw[:, 1] * u_c
        u_phys[:, 2] = pred_raw[:, 2] * theta_c

        U_per_elem = self._strain_energy_per_elem(u_phys, data)
        W_external = self._external_work(u_phys, data)

        U_internal = U_per_elem.sum()
        Pi = U_internal - W_external

        # ═══════════════════════════════════════
        # Normalize by TOTAL energy scale across batch
        # ═══════════════════════════════════════
        if hasattr(data, 'batch') and data.batch is not None:
            n_graphs = data.num_graphs
            E_c_per_graph = (data.F_c * data.u_c).clamp(min=1e-30)
            E_c_total = E_c_per_graph.sum()
        else:
            n_graphs = 1
            E_c_total = (data.F_c * data.u_c).clamp(min=1e-30)

        Pi_norm = Pi / E_c_total

        # ═══════════════════════════════════════
        # SCALE the loss to prevent huge gradients
        # ═══════════════════════════════════════
        loss = Pi_norm / n_graphs  # average over graphs

        loss_dict = {
            'total':      loss.item(),
            'Pi':         Pi.item(),
            'Pi_norm':    Pi_norm.item() / n_graphs,
            'U_internal': U_internal.item() / n_graphs,
            'W_external': W_external.item() / n_graphs,
            'U_over_W':   (U_internal
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

        return loss, loss_dict, pred_raw, u_phys


    def _strain_energy_per_elem(self, u_phys, data):
        """
        Returns per-element strain energy (E,) tensor.
        Same math as _strain_energy but returns before .sum()
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

        ux_A = u_phys[nA, 0]
        uz_A = u_phys[nA, 1]
        th_A = u_phys[nA, 2]
        ux_B = u_phys[nB, 0]
        uz_B = u_phys[nB, 1]
        th_B = u_phys[nB, 2]

        u_A_loc =  c * ux_A + s * uz_A
        w_A_loc = -s * ux_A + c * uz_A
        u_B_loc =  c * ux_B + s * uz_B
        w_B_loc = -s * ux_B + c * uz_B

        th_A_loc = -th_A
        th_B_loc = -th_B

        d_local = torch.stack([
            u_A_loc, w_A_loc, th_A_loc,
            u_B_loc, w_B_loc, th_B_loc
        ], dim=1)

        K = torch.zeros(n_elem, 6, 6, device=device)

        ea_L  = EA / L
        ei_L  = EI / L
        ei_L2 = EI / L**2
        ei_L3 = EI / L**3

        K[:, 0, 0] =  ea_L;  K[:, 0, 3] = -ea_L
        K[:, 3, 0] = -ea_L;  K[:, 3, 3] =  ea_L

        K[:, 1, 1] =  12*ei_L3; K[:, 1, 2] =  6*ei_L2
        K[:, 1, 4] = -12*ei_L3; K[:, 1, 5] =  6*ei_L2
        K[:, 2, 1] =   6*ei_L2; K[:, 2, 2] =  4*ei_L
        K[:, 2, 4] =  -6*ei_L2; K[:, 2, 5] =  2*ei_L
        K[:, 4, 1] = -12*ei_L3; K[:, 4, 2] = -6*ei_L2
        K[:, 4, 4] =  12*ei_L3; K[:, 4, 5] = -6*ei_L2
        K[:, 5, 1] =   6*ei_L2; K[:, 5, 2] =  2*ei_L
        K[:, 5, 4] =  -6*ei_L2; K[:, 5, 5] =  4*ei_L

        Kd = torch.bmm(K, d_local.unsqueeze(2))
        U_per_elem = 0.5 * torch.bmm(
            d_local.unsqueeze(1), Kd
        ).squeeze()

        return U_per_elem

    def _strain_energy(self, u_phys, data):
        """
        U = (1/2) Σ_e d_local^T K_local d_local
        
        With NEGATED θ to match Kratos convention.
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

        # Global DOFs
        ux_A = u_phys[nA, 0]
        uz_A = u_phys[nA, 1]
        th_A = u_phys[nA, 2]
        ux_B = u_phys[nB, 0]
        uz_B = u_phys[nB, 1]
        th_B = u_phys[nB, 2]

        # Transform to local
        u_A_loc =  c * ux_A + s * uz_A
        w_A_loc = -s * ux_A + c * uz_A
        u_B_loc =  c * ux_B + s * uz_B
        w_B_loc = -s * ux_B + c * uz_B

        # ═══════════════════════════════════════
        # NEGATE θ for Kratos sign convention
        # ═══════════════════════════════════════
        th_A_loc = -th_A
        th_B_loc = -th_B

        # Local DOF vector
        d_local = torch.stack([
            u_A_loc, w_A_loc, th_A_loc,
            u_B_loc, w_B_loc, th_B_loc
        ], dim=1)  # (E, 6)

        # Build K_local
        K = torch.zeros(n_elem, 6, 6, device=device)

        ea_L  = EA / L
        ei_L  = EI / L
        ei_L2 = EI / L**2
        ei_L3 = EI / L**3

        # Axial
        K[:, 0, 0] =  ea_L
        K[:, 0, 3] = -ea_L
        K[:, 3, 0] = -ea_L
        K[:, 3, 3] =  ea_L

        # Bending
        K[:, 1, 1] =  12 * ei_L3
        K[:, 1, 2] =   6 * ei_L2
        K[:, 1, 4] = -12 * ei_L3
        K[:, 1, 5] =   6 * ei_L2

        K[:, 2, 1] =   6 * ei_L2
        K[:, 2, 2] =   4 * ei_L
        K[:, 2, 4] =  -6 * ei_L2
        K[:, 2, 5] =   2 * ei_L

        K[:, 4, 1] = -12 * ei_L3
        K[:, 4, 2] =  -6 * ei_L2
        K[:, 4, 4] =  12 * ei_L3
        K[:, 4, 5] =  -6 * ei_L2

        K[:, 5, 1] =   6 * ei_L2
        K[:, 5, 2] =   2 * ei_L
        K[:, 5, 4] =  -6 * ei_L2
        K[:, 5, 5] =   4 * ei_L

        # U = (1/2) d^T K d
        Kd = torch.bmm(K, d_local.unsqueeze(2))
        U_per_elem = 0.5 * torch.bmm(
            d_local.unsqueeze(1), Kd
        ).squeeze()

        return U_per_elem.sum()

    def _external_work(self, u_phys, data):
        """
        W = Σ_i (Fx·ux + Fz·uz + My·θ)
        
        NOTE: We do NOT negate θ here!
        The external moment My is defined in Kratos 
        convention, so My·θ_kratos is correct as-is.
        
        (The negation only applies to the INTERNAL
        stiffness matrix sign convention.)
        """
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
    print("  ENERGY LOSS VERIFICATION (θ-Negated)")
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
    # TEST 1: Cantilever checks
    # ══════════════════════════════════════════
    from torch_geometric.data import Data

    L_t = 10.0
    E_t = 2.1e11
    A_t = 0.01
    I_t = 8.333e-6

    # 1a: Horizontal
    print(f"\n  TEST 1a: Horizontal cantilever")
    P_h = -1000.0
    w_h = P_h * L_t**3 / (3 * E_t * I_t)
    th_h = P_h * L_t**2 / (2 * E_t * I_t)
    U_exact_h = abs(P_h) * abs(w_h) / 2

    test_h = Data(
        connectivity=torch.tensor([[0, 1]]),
        elem_lengths=torch.tensor([L_t],
                                   dtype=torch.float32),
        elem_directions=torch.tensor(
            [[1, 0, 0]], dtype=torch.float32),
        prop_E=torch.tensor([E_t],
                            dtype=torch.float32),
        prop_A=torch.tensor([A_t],
                            dtype=torch.float32),
        prop_I22=torch.tensor([I_t],
                              dtype=torch.float32),
        F_ext=torch.tensor(
            [[0, 0, 0], [0, P_h, 0]],
            dtype=torch.float32),
        F_c=torch.tensor(abs(P_h)),
        u_c=torch.tensor(abs(w_h)),
        theta_c=torch.tensor(abs(th_h)),
    )

    # Kratos convention: θ has opposite sign
    # So tip rotation in Kratos = -th_h
    u_h = torch.tensor(
        [[0, 0, 0], [0, w_h, -th_h]],
        dtype=torch.float32
    )

    U_h = loss_fn._strain_energy(u_h, test_h)
    W_h = loss_fn._external_work(u_h, test_h)
    UW_h = (U_h / abs(W_h)).item()
    print(f"    Kratos θ (negated): {-th_h:.4e}")
    print(f"    U = {U_h.item():.6e} "
          f"(exact: {U_exact_h:.6e})")
    print(f"    U/|W| = {UW_h:.6f} "
          f"{'✓' if abs(UW_h - 0.5) < 0.001 else '✗'}")

    # 1b: Vertical
    print(f"\n  TEST 1b: Vertical cantilever")
    P_v = 1000.0
    w_v = P_v * L_t**3 / (3 * E_t * I_t)
    th_v = P_v * L_t**2 / (2 * E_t * I_t)
    U_exact_v = abs(P_v) * abs(w_v) / 2

    test_v = Data(
        connectivity=torch.tensor([[0, 1]]),
        elem_lengths=torch.tensor([L_t],
                                   dtype=torch.float32),
        elem_directions=torch.tensor(
            [[0, 0, 1]], dtype=torch.float32),
        prop_E=torch.tensor([E_t],
                            dtype=torch.float32),
        prop_A=torch.tensor([A_t],
                            dtype=torch.float32),
        prop_I22=torch.tensor([I_t],
                              dtype=torch.float32),
        F_ext=torch.tensor(
            [[0, 0, 0], [P_v, 0, 0]],
            dtype=torch.float32),
        F_c=torch.tensor(abs(P_v)),
        u_c=torch.tensor(abs(w_v)),
        theta_c=torch.tensor(abs(th_v)),
    )

    # Kratos: positive θ for vertical column with 
    # positive Fx → test both
    for th_sign_label, th_val in [
        ("Kratos +θ", th_v),
        ("Kratos -θ", -th_v),
    ]:
        u_v = torch.tensor(
            [[0, 0, 0], [w_v, 0, th_val]],
            dtype=torch.float32
        )
        U_v = loss_fn._strain_energy(u_v, test_v)
        W_v = loss_fn._external_work(u_v, test_v)
        UW_v = (U_v / abs(W_v)).item()
        err = abs(U_v.item() - U_exact_v) / U_exact_v
        print(f"    {th_sign_label}: "
              f"U/|W|={UW_v:.4f}, "
              f"err={err:.2e} "
              f"{'✓' if err < 1e-3 else '✗'}")

    # ══════════════════════════════════════════
    # TEST 2: Kratos frame data
    # ══════════════════════════════════════════
    print(f"\n\n  TEST 2: Kratos frame data")

    for i in range(min(5, len(data_list))):
        data = data_list[i]
        u_true = data.y_node.clone()

        U_true = loss_fn._strain_energy(u_true, data)
        W_true = loss_fn._external_work(u_true, data)
        Pi_true = U_true - W_true

        Pi_zero = torch.tensor(0.0)

        UW = (U_true / W_true.abs().clamp(
            min=1e-30
        )).item()

        status = ("✓" if Pi_true < Pi_zero
                  else "✗")
        print(f"    Case {i}: U={U_true.item():.4e}, "
              f"W={W_true.item():.4e}, "
              f"Π={Pi_true.item():.4e}, "
              f"U/|W|={UW:.4f} {status}")

    # ══════════════════════════════════════════
    # TEST 3: Energy landscape
    # ══════════════════════════════════════════
    print(f"\n\n  TEST 3: Energy along u_true")
    data = data_list[0]
    u_true = data.y_node.clone()

    print(f"    α      Π              U/|W|")
    alphas = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
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
        marker = (" ← should be min"
                  if alpha == 1.0 else "")
        print(f"    {alpha:.2f}   {Pi_a:14.6e}   "
              f"{UW_a:.4f}{marker}")

    min_idx = Pi_vals.index(min(Pi_vals))
    correct = (min_idx == alphas.index(1.0))
    print(f"\n    Minimum at α={alphas[min_idx]:.2f} "
          f"({'✓ CORRECT' if correct else '✗ WRONG'})")

    # ══════════════════════════════════════════
    # TEST 4: Gradient at u=0
    # ══════════════════════════════════════════
    print(f"\n\n  TEST 4: Gradient at u=0")
    data = data_list[0]
    u = torch.zeros(
        data.num_nodes, 3, requires_grad=True
    )

    Pi = (loss_fn._strain_energy(u, data)
        - loss_fn._external_work(u, data))
    Pi.backward()

    grad = u.grad
    print(f"    Π(0) = {Pi.item():.6e}")
    print(f"    ∇Π = [{grad.min().item():.4e}, "
          f"{grad.max().item():.4e}]")
    print(f"    |∇Π| = {grad.norm().item():.4e}")

    # Verify ∂Π/∂u = -F_ext at u=0
    for dof, name in [(0, 'Fx'), (1, 'Fz'), (2, 'My')]:
        g = grad[:, dof]
        f = -data.F_ext[:, dof]
        loaded = f.abs() > 1e-10
        if loaded.any():
            match = torch.allclose(
                g[loaded], f[loaded],
                rtol=1e-3, atol=1e-6
            )
            print(f"    ∂Π/∂u_{name} = -F_{name}? "
                  f"{'✓' if match else '✗'}")
            if not match:
                idx = loaded.nonzero()[0].item()
                print(f"      node {idx}: "
                      f"grad={g[idx].item():.4e}, "
                      f"-F={f[idx].item():.4e}")

    # ══════════════════════════════════════════
    # TEST 5: Target energy for training
    # ══════════════════════════════════════════
    print(f"\n\n  TEST 5: Target energies")
    for i in range(min(5, len(data_list))):
        data = data_list[i]
        u_true = data.y_node.clone()
        U = loss_fn._strain_energy(u_true, data)
        W = loss_fn._external_work(u_true, data)
        Pi = U - W
        E_c = (data.F_c * data.u_c).clamp(min=1e-30)
        Pi_norm = (Pi / E_c).item()
        print(f"    Case {i}: Π_norm = {Pi_norm:.4e}")

    print(f"\n  DONE ✓")