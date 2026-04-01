"""
=================================================================
energy_loss.py — Non-Dimensionalized Energy Loss (v2 clean)
=================================================================
Two interfaces:
  forward(model, data, aw)   — calls model internally
  compute(pred_raw, data, aw) — takes predictions directly
                                (needed for L-BFGS closure)
=================================================================
"""

import torch
import torch.nn as nn


class FrameEnergyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    # ────────────────────────────────────────
    # Public interfaces
    # ────────────────────────────────────────

    def forward(self, model, data, axial_weight=1.0):
        """Original interface: model + data → loss."""
        pred_raw = model(data)
        return self.compute(pred_raw, data, axial_weight)

    def compute(self, pred_raw, data, axial_weight=1.0):
        """
        Direct interface: pred_raw + data → loss.

        Args:
            pred_raw:     (N, 3) non-dim [ux/ux_c, uz/uz_c, θ/θ_c]
            data:         PyG Data/Batch
            axial_weight: curriculum weight for axial stiffness

        Returns:
            loss, loss_dict, pred_raw, u_phys
        """
        # ── Physics scales ──
        if hasattr(data, 'batch') and data.batch is not None:
            ux_c    = data.ux_c[data.batch]
            # uz_c    = data.uz_c[data.batch]
            uz_c    = data.ux_c[data.batch] # same scale as ux
            theta_c = data.theta_c[data.batch]
        else:
            ux_c    = data.ux_c
            # uz_c    = data.uz_c
            uz_c    = data.ux_c # same scale as ux
            theta_c = data.theta_c

        # ── Physical displacements ──
        u_phys = torch.zeros_like(pred_raw)
        u_phys[:, 0] = pred_raw[:, 0] * ux_c
        u_phys[:, 1] = pred_raw[:, 1] * uz_c
        u_phys[:, 2] = pred_raw[:, 2] * theta_c

        n_graphs = (data.num_graphs
                    if hasattr(data, 'batch')
                    and data.batch is not None
                    else 1)

        # ── Non-dimensional energy (the loss) ──
        loss = self._nondim_energy(
            pred_raw, data, axial_weight=axial_weight
        )

        # ── Logging dict ──
        loss_dict = {
            'total':      loss.item(),
            'Pi':         loss.item(),
            'Pi_norm':    loss.item(),
            'U_internal': 0.0,
            'W_external': 0.0,
            'U_over_W':   0.0,
            'ux_range':   [u_phys[:, 0].min().item(),
                           u_phys[:, 0].max().item()],
            'uz_range':   [u_phys[:, 1].min().item(),
                           u_phys[:, 1].max().item()],
            'th_range':   [u_phys[:, 2].min().item(),
                           u_phys[:, 2].max().item()],
            'raw_range':  [pred_raw.min().item(),
                           pred_raw.max().item()],
        }

        with torch.no_grad():
            U = self._strain_energy(u_phys, data)
            W = self._external_work(u_phys, data)
            loss_dict['U_internal'] = U.item() / n_graphs
            loss_dict['W_external'] = W.item() / n_graphs
            loss_dict['U_over_W'] = (
                U / W.abs().clamp(min=1e-30)
            ).item()

        return loss, loss_dict, pred_raw, u_phys

    # ────────────────────────────────────────
    # Core energy computation
    # ────────────────────────────────────────

    def _nondim_energy(self, pred_raw, data, axial_weight=1.0):
        """Π / E_c in non-dimensional form."""
        conn = data.connectivity
        nA, nB = conn[:, 0], conn[:, 1]

        L  = data.elem_lengths
        EA = data.prop_E * data.prop_A
        EI = data.prop_E * data.prop_I22
        c  = data.elem_directions[:, 0]
        s  = data.elem_directions[:, 2]

        # ── Per-element / per-node scales ──
        if hasattr(data, 'batch') and data.batch is not None:
            bn = data.batch
            be = bn[nA]
            ux_c_A    = data.ux_c[bn[nA]]
            ux_c_B    = data.ux_c[bn[nB]]
            # uz_c_A    = data.uz_c[bn[nA]]
            # uz_c_B    = data.uz_c[bn[nB]]
            uz_c_A    = data.ux_c[bn[nA]] # same scale as ux
            uz_c_B    = data.ux_c[bn[nB]] # same scale as ux
            theta_c_A = data.theta_c[bn[nA]]
            theta_c_B = data.theta_c[bn[nB]]
            E_c_e     = (data.F_c[be] * data.ux_c[be]).clamp(min=1e-30)
            E_c_n     = (data.F_c[bn] * data.ux_c[bn]).clamp(min=1e-30)
            ux_c_n    = data.ux_c[bn]
            # uz_c_n    = data.uz_c[bn]
            uz_c_n    = data.ux_c[bn] # same scale as ux
            theta_c_n = data.theta_c[bn]
        else:
            ux_c_A = ux_c_B = data.ux_c
            # uz_c_A = uz_c_B = data.uz_c
            uz_c_A = uz_c_B = data.ux_c # same scale as ux
            theta_c_A = theta_c_B = data.theta_c
            E_c_e = (data.F_c * data.ux_c).clamp(min=1e-30)
            E_c_n = E_c_e
            ux_c_n = data.ux_c
            # uz_c_n = data.uz_c
            uz_c_n = data.uz_c # same scale as ux
            theta_c_n = data.theta_c

        # ── Physical local DOFs ──
        ux_A = pred_raw[nA, 0] * ux_c_A
        uz_A = pred_raw[nA, 1] * uz_c_A
        th_A = pred_raw[nA, 2] * theta_c_A
        ux_B = pred_raw[nB, 0] * ux_c_B
        uz_B = pred_raw[nB, 1] * uz_c_B
        th_B = pred_raw[nB, 2] * theta_c_B

        # Global → local coordinate transform
        u_A_loc =  c * ux_A + s * uz_A
        w_A_loc = -s * ux_A + c * uz_A
        u_B_loc =  c * ux_B + s * uz_B
        w_B_loc = -s * ux_B + c * uz_B
        th_A_loc = -th_A
        th_B_loc = -th_B

        # ── Strain energy ──
        ea_L  = EA / L
        ei_L  = EI / L
        ei_L2 = EI / L**2
        ei_L3 = EI / L**3

        du_axial = u_B_loc - u_A_loc
        U_axial = (axial_weight * 0.5 * ea_L
                   * du_axial**2 / E_c_e)

        U_bend = (
            12 * ei_L3 * (w_A_loc**2 + w_B_loc**2
                          - 2 * w_A_loc * w_B_loc)
            + 12 * ei_L2 * (w_A_loc * th_A_loc
                            - w_B_loc * th_A_loc
                            + w_A_loc * th_B_loc
                            - w_B_loc * th_B_loc)
            + 4 * ei_L * (th_A_loc**2 + th_B_loc**2)
            + 4 * ei_L * th_A_loc * th_B_loc
        ) * 0.5 / E_c_e

        # ── External work ──
        W_per_node = (
            data.F_ext[:, 0] * pred_raw[:, 0] * ux_c_n
            + data.F_ext[:, 1] * pred_raw[:, 1] * uz_c_n
            + data.F_ext[:, 2] * pred_raw[:, 2] * theta_c_n
        ) / E_c_n

        # ── Π / E_c averaged over graphs ──
        n_graphs = (data.num_graphs
                    if hasattr(data, 'batch')
                    and data.batch is not None
                    else 1)

        Pi_hat = ((U_axial + U_bend).sum()
                  - W_per_node.sum()) / n_graphs
        return Pi_hat

    # ────────────────────────────────────────
    # Physical energies (logging / verification)
    # ────────────────────────────────────────

    def _strain_energy(self, u_phys, data):
        """Physical U = ½ d^T K d (full, no axial weight)."""
        conn = data.connectivity
        nA, nB = conn[:, 0], conn[:, 1]
        n_elem = conn.shape[0]
        device = u_phys.device

        L  = data.elem_lengths
        EA = data.prop_E * data.prop_A
        EI = data.prop_E * data.prop_I22
        c  = data.elem_directions[:, 0]
        s  = data.elem_directions[:, 2]

        ux_A = u_phys[nA, 0]; uz_A = u_phys[nA, 1]
        th_A = u_phys[nA, 2]
        ux_B = u_phys[nB, 0]; uz_B = u_phys[nB, 1]
        th_B = u_phys[nB, 2]

        d = torch.stack([
             c*ux_A + s*uz_A,
            -s*ux_A + c*uz_A,
            -th_A,
             c*ux_B + s*uz_B,
            -s*ux_B + c*uz_B,
            -th_B,
        ], dim=1)

        K = torch.zeros(n_elem, 6, 6, device=device)
        ea_L = EA/L; ei_L = EI/L
        ei_L2 = EI/L**2; ei_L3 = EI/L**3

        K[:,0,0]= ea_L;  K[:,0,3]=-ea_L
        K[:,3,0]=-ea_L;  K[:,3,3]= ea_L
        K[:,1,1]= 12*ei_L3; K[:,1,2]= 6*ei_L2
        K[:,1,4]=-12*ei_L3; K[:,1,5]= 6*ei_L2
        K[:,2,1]= 6*ei_L2;  K[:,2,2]= 4*ei_L
        K[:,2,4]=-6*ei_L2;  K[:,2,5]= 2*ei_L
        K[:,4,1]=-12*ei_L3; K[:,4,2]=-6*ei_L2
        K[:,4,4]= 12*ei_L3; K[:,4,5]=-6*ei_L2
        K[:,5,1]= 6*ei_L2;  K[:,5,2]= 2*ei_L
        K[:,5,4]=-6*ei_L2;  K[:,5,5]= 4*ei_L

        Kd = torch.bmm(K, d.unsqueeze(2))
        U = 0.5 * torch.bmm(d.unsqueeze(1), Kd).squeeze()
        return U.sum()

    def _external_work(self, u_phys, data):
        """Physical W = F · u."""
        return (
            data.F_ext[:, 0] * u_phys[:, 0]
            + data.F_ext[:, 1] * u_phys[:, 1]
            + data.F_ext[:, 2] * u_phys[:, 2]
        ).sum()
    
    def compute_residual_loss(self, model_pred, data, axial_weight=1.0):
        """
        Residual form: loss = ||∂Π/∂u||²
        model_pred must be direct model output (connected to computation graph).
        Same equilibrium as energy minimization, but all DOFs get equal gradient.
        """
        # Compute energy (scalar) — model_pred is in the graph
        Pi = self._nondim_energy(model_pred, data, axial_weight=axial_weight)
        
        # Residual R = ∂Π/∂pred  [N_total_nodes, 3]
        R = torch.autograd.grad(
            Pi, model_pred,
            create_graph=True,   # need 2nd order for loss.backward()
        )[0]
        
        n_graphs = data.num_graphs if hasattr(data, 'num_graphs') else 1
        
        # Per-DOF squared residual, averaged over graphs
        R_ux = (R[:, 0] ** 2).sum() / n_graphs
        R_uz = (R[:, 1] ** 2).sum() / n_graphs
        R_th = (R[:, 2] ** 2).sum() / n_graphs
        
        loss = R_ux + R_uz + R_th
        
        info = {
            'total': loss.item(),
            'R_ux': R_ux.item(),
            'R_uz': R_uz.item(),
            'R_th': R_th.item(),
            'Pi': Pi.item(),
        }
        
        return loss, info