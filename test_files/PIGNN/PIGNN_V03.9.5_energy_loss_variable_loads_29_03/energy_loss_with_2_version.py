"""
=================================================================
energy_loss.py — Non-Dimensionalized Energy Loss
=================================================================

Solves the K/E_c = 6.7e7 conditioning problem by working
in non-dimensional coordinates throughout.

All energies are computed as Π/(F_c * u_c) directly,
avoiding intermediate huge numbers.
=================================================================
"""

import torch
import torch.nn as nn


class FrameEnergyLoss(nn.Module):
    """
    Total Potential Energy in non-dimensional form.
    
    Π_hat = U_hat - W_hat
    
    Where:
      U_hat = (1/2) Σ_e d_hat^T K_hat d_hat
      W_hat = Σ_i F_hat · u_hat
      
    All quantities are O(1), preventing gradient explosion.
    """

    def __init__(self):
        super().__init__()

    def forward(self, model, data, axial_weight=1.0):
        pred_raw = model(data)  # (N, 3) non-dimensional

        # ═══ FIX 3: Separate scales per DOF ═══
        if hasattr(data, 'batch') and data.batch is not None:
            ux_c    = data.ux_c[data.batch]
            uz_c    = data.uz_c[data.batch]
            theta_c = data.theta_c[data.batch]
            F_c     = data.F_c[data.batch]
            # n_graphs = data.num_graphs
        else:
            ux_c    = data.ux_c
            uz_c    = data.uz_c
            theta_c = data.theta_c
            F_c     = data.F_c
            # n_graphs = 1

        u_phys = torch.zeros_like(pred_raw)
        u_phys[:, 0] = pred_raw[:, 0] * ux_c      # was: u_c
        u_phys[:, 1] = pred_raw[:, 1] * uz_c      # was: u_c (THE KEY FIX)
        u_phys[:, 2] = pred_raw[:, 2] * theta_c
        
        # ═══ Graph count ═══
        if hasattr(data, 'batch') and data.batch is not None:
            n_graphs = data.num_graphs
        else:
            n_graphs = 1

        # ═══════════════════════════════════════
        # Non-dimensional energy computation
        # ═══════════════════════════════════════
        Pi_nd = self._nondim_energy(pred_raw, data, axial_weight=axial_weight)

        # Pi_nd is already Π / E_c (per graph average)
        loss = Pi_nd

        loss_dict = {
            'total':      loss.item(),
            'Pi':         loss.item(),
            'Pi_norm':    loss.item(),
            'U_internal': 0.0,  # filled below
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

        # Compute physical energies for logging only
        with torch.no_grad():
            U = self._strain_energy(u_phys, data)
            W = self._external_work(u_phys, data)
            loss_dict['U_internal'] = U.item() / n_graphs
            loss_dict['W_external'] = W.item() / n_graphs
            loss_dict['U_over_W'] = (
                U / W.abs().clamp(min=1e-30)
            ).item()

        return loss, loss_dict, pred_raw, u_phys

    def _nondim_energy(self, pred_raw, data, axial_weight=1.0):
        """
        Compute Π / E_c directly in non-dimensional form.
        
        pred_raw[:, 0] = ux / ux_c  ~O(1)
        pred_raw[:, 1] = uz / uz_c  ~O(1)  
        pred_raw[:, 2] = θ / θ_c   ~O(1)
        
        Non-dim stiffness:
          K_hat_axial = (EA/L) * u_c² / E_c  ~O(1) if u_c² ~ E_c/(EA/L)
          K_hat_bend  = (EI/L³) * u_c² / E_c ~O(1)
        
        Non-dim force:
          F_hat = F / F_c  ~O(1)
        """
        # conn = data.connectivity
        # nA, nB = conn[:, 0], conn[:, 1]
        # n_elem = conn.shape[0]
        # device = pred_raw.device

        # L  = data.elem_lengths
        # EA = data.prop_E * data.prop_A
        # EI = data.prop_E * data.prop_I22

        # c = data.elem_directions[:, 0]
        # s = data.elem_directions[:, 2]

        # # ═══════════════════════════════════════
        # # Per-element scales
        # # ═══════════════════════════════════════
        # if hasattr(data, 'batch') and data.batch is not None:
        #     batch_node = data.batch
        #     batch_elem = batch_node[nA]
        #     u_c_e = data.u_c[batch_elem]
        #     theta_c_e = data.theta_c[batch_elem]
        #     F_c_e = data.F_c[batch_elem]
        #     E_c_e = (F_c_e * u_c_e).clamp(min=1e-30)
            
        #     u_c_A = data.u_c[batch_node[nA]]
        #     u_c_B = data.u_c[batch_node[nB]]
        #     theta_c_A = data.theta_c[batch_node[nA]]
        #     theta_c_B = data.theta_c[batch_node[nB]]
        #     F_c_node = data.F_c[batch_node]
        #     E_c_node = (F_c_node * data.u_c[batch_node]).clamp(min=1e-30)
        # else:
        #     u_c_e = data.u_c
        #     theta_c_e = data.theta_c
        #     F_c_e = data.F_c
        #     E_c_e = (F_c_e * u_c_e).clamp(min=1e-30)
            
        #     u_c_A = data.u_c
        #     u_c_B = data.u_c
        #     theta_c_A = data.theta_c
        #     theta_c_B = data.theta_c
        #     F_c_node = data.F_c
        #     E_c_node = (F_c_node * data.u_c).clamp(min=1e-30)

        # # ═══════════════════════════════════════
        # # Non-dimensional local DOFs
        # # ═══════════════════════════════════════
        # # Raw predictions are already non-dim
        # ux_hat_A = pred_raw[nA, 0]  # ux/u_c
        # uz_hat_A = pred_raw[nA, 1]  # uz/u_c
        # th_hat_A = pred_raw[nA, 2]  # θ/θ_c
        # ux_hat_B = pred_raw[nB, 0]
        # uz_hat_B = pred_raw[nB, 1]
        # th_hat_B = pred_raw[nB, 2]

        # # Physical local displacements
        # ux_A = ux_hat_A * u_c_A
        # uz_A = uz_hat_A * u_c_A
        # th_A = th_hat_A * theta_c_A
        # ux_B = ux_hat_B * u_c_B
        # uz_B = uz_hat_B * u_c_B
        # th_B = th_hat_B * theta_c_B

        # u_A_loc =  c * ux_A + s * uz_A
        # w_A_loc = -s * ux_A + c * uz_A
        # u_B_loc =  c * ux_B + s * uz_B
        # w_B_loc = -s * ux_B + c * uz_B
        # th_A_loc = -th_A
        # th_B_loc = -th_B

        # # ═══════════════════════════════════════
        # # Strain energy per element, normalized by E_c
        # # U_hat_e = (1/2) d^T K d / E_c
        # # ═══════════════════════════════════════
        # ea_L  = EA / L
        # ei_L  = EI / L
        # ei_L2 = EI / L**2
        # ei_L3 = EI / L**3

        # # Axial contribution: (EA/L)(u_B - u_A)² / (2 E_c)
        # du_axial = u_B_loc - u_A_loc
        # U_axial = 0.5 * ea_L * du_axial**2 / E_c_e

        # # Bending contribution using Hermitian shape functions
        # # U_bend = (EI/2) ∫ (w'')² dx
        # # For cubic Hermite: 
        # #   U = (1/2)[12EI/L³(wA-wB)² + 12EI/L³(wA-wB)(L·θA+L·θB)/2 
        # #        + 4EI/L·θA² + 4EI/L·θB² + 2·2EI/L·θA·θB
        # #        + ...]
        # # Easier: just do d^T K_bend d directly

        # dw = w_A_loc - w_B_loc
        # sum_th = th_A_loc + th_B_loc
        # diff_th = th_A_loc - th_B_loc

        # U_bend = (
        #     12 * ei_L3 * (w_A_loc**2 + w_B_loc**2 - 2*w_A_loc*w_B_loc)
        #     + 12 * ei_L2 * (w_A_loc * th_A_loc - w_B_loc * th_A_loc 
        #                    + w_A_loc * th_B_loc - w_B_loc * th_B_loc)
        #     + 4 * ei_L * (th_A_loc**2 + th_B_loc**2)
        #     + 2 * 2 * ei_L * th_A_loc * th_B_loc
        # ) * 0.5 / E_c_e

        # U_hat_per_elem = U_axial + U_bend

        # # ═══════════════════════════════════════
        # # External work, normalized by E_c
        # # W_hat = Σ F·u / E_c
        # # ═══════════════════════════════════════
        # if hasattr(data, 'batch') and data.batch is not None:
        #     u_phys_0 = pred_raw[:, 0] * data.u_c[data.batch]
        #     u_phys_1 = pred_raw[:, 1] * data.u_c[data.batch]
        #     u_phys_2 = pred_raw[:, 2] * data.theta_c[data.batch]
        # else:
        #     u_phys_0 = pred_raw[:, 0] * data.u_c
        #     u_phys_1 = pred_raw[:, 1] * data.u_c
        #     u_phys_2 = pred_raw[:, 2] * data.theta_c

        # W_per_node = (
        #     data.F_ext[:, 0] * u_phys_0
        #     + data.F_ext[:, 1] * u_phys_1
        #     + data.F_ext[:, 2] * u_phys_2
        # ) / E_c_node

        # # ═══════════════════════════════════════
        # # Total: Π_hat = U_hat - W_hat (averaged)
        # # ═══════════════════════════════════════
        # if hasattr(data, 'batch') and data.batch is not None:
        #     n_graphs = data.num_graphs
        # else:
        #     n_graphs = 1

        # Pi_hat = U_hat_per_elem.sum() - W_per_node.sum()
        # Pi_hat = Pi_hat / n_graphs

        # return Pi_hat
        conn = data.connectivity
        nA, nB = conn[:, 0], conn[:, 1]
        device = pred_raw.device

        L  = data.elem_lengths
        EA = data.prop_E * data.prop_A
        EI = data.prop_E * data.prop_I22
        c  = data.elem_directions[:, 0]
        s  = data.elem_directions[:, 2]

        # ═══ Per-element scales ═══
        if hasattr(data, 'batch') and data.batch is not None:
            batch_node = data.batch
            batch_elem = batch_node[nA]
            ux_c_A = data.ux_c[batch_node[nA]]
            ux_c_B = data.ux_c[batch_node[nB]]
            uz_c_A = data.uz_c[batch_node[nA]]
            uz_c_B = data.uz_c[batch_node[nB]]
            theta_c_A = data.theta_c[batch_node[nA]]
            theta_c_B = data.theta_c[batch_node[nB]]
            F_c_e = data.F_c[batch_elem]
            ux_c_e = data.ux_c[batch_elem]
            E_c_e = (F_c_e * ux_c_e).clamp(min=1e-30)
            
            F_c_node = data.F_c[batch_node]
            ux_c_node = data.ux_c[batch_node]
            uz_c_node = data.uz_c[batch_node]
            theta_c_node = data.theta_c[batch_node]
            E_c_node = (F_c_node * ux_c_node).clamp(min=1e-30)
        else:
            ux_c_A = ux_c_B = data.ux_c
            uz_c_A = uz_c_B = data.uz_c
            theta_c_A = theta_c_B = data.theta_c
            F_c_e = data.F_c
            ux_c_e = data.ux_c
            E_c_e = (F_c_e * ux_c_e).clamp(min=1e-30)
            
            F_c_node = data.F_c
            ux_c_node = data.ux_c
            uz_c_node = data.uz_c
            theta_c_node = data.theta_c
            E_c_node = (F_c_node * ux_c_node).clamp(min=1e-30)

        # ═══ Physical local displacements with separate scales ═══
        ux_A = pred_raw[nA, 0] * ux_c_A
        uz_A = pred_raw[nA, 1] * uz_c_A      # ← uses uz_c, not u_c
        th_A = pred_raw[nA, 2] * theta_c_A
        ux_B = pred_raw[nB, 0] * ux_c_B
        uz_B = pred_raw[nB, 1] * uz_c_B      # ← uses uz_c, not u_c
        th_B = pred_raw[nB, 2] * theta_c_B

        u_A_loc =  c * ux_A + s * uz_A
        w_A_loc = -s * ux_A + c * uz_A
        u_B_loc =  c * ux_B + s * uz_B
        w_B_loc = -s * ux_B + c * uz_B
        th_A_loc = -th_A
        th_B_loc = -th_B

        # ═══ Strain energy (unchanged formula) ═══
        ea_L  = EA / L
        ei_L  = EI / L
        ei_L2 = EI / L**2
        ei_L3 = EI / L**3

        du_axial = u_B_loc - u_A_loc
        # U_axial = 0.5 * ea_L * du_axial**2 / E_c_e
        # ══════════════════════════════════════════════
        # AXIAL with curriculum weight
        # ══════════════════════════════════════════════
        U_axial = axial_weight * 0.5 * ea_L * du_axial**2 / E_c_e

        U_bend = (
            12 * ei_L3 * (w_A_loc**2 + w_B_loc**2 - 2*w_A_loc*w_B_loc)
            + 12 * ei_L2 * (w_A_loc * th_A_loc - w_B_loc * th_A_loc
                        + w_A_loc * th_B_loc - w_B_loc * th_B_loc)
            + 4 * ei_L * (th_A_loc**2 + th_B_loc**2)
            + 4 * ei_L * th_A_loc * th_B_loc
        ) * 0.5 / E_c_e

        U_hat_per_elem = U_axial + U_bend

        # ═══ External work with separate scales ═══
        u_phys_0 = pred_raw[:, 0] * ux_c_node       # ← ux_c
        u_phys_1 = pred_raw[:, 1] * uz_c_node       # ← uz_c
        u_phys_2 = pred_raw[:, 2] * theta_c_node

        W_per_node = (
            data.F_ext[:, 0] * u_phys_0
            + data.F_ext[:, 1] * u_phys_1
            + data.F_ext[:, 2] * u_phys_2
        ) / E_c_node

        # ═══ Total ═══
        n_graphs = data.num_graphs if (hasattr(data, 'batch') and data.batch is not None) else 1
        Pi_hat = (U_hat_per_elem.sum() - W_per_node.sum()) / n_graphs

        return Pi_hat

    def _strain_energy(self, u_phys, data):
        """Physical strain energy (for verification/logging)."""
        conn = data.connectivity
        nA, nB = conn[:, 0], conn[:, 1]
        n_elem = conn.shape[0]
        device = u_phys.device

        L  = data.elem_lengths
        EA = data.prop_E * data.prop_A
        EI = data.prop_E * data.prop_I22

        c = data.elem_directions[:, 0]
        s = data.elem_directions[:, 2]

        ux_A = u_phys[nA, 0]; uz_A = u_phys[nA, 1]; th_A = u_phys[nA, 2]
        ux_B = u_phys[nB, 0]; uz_B = u_phys[nB, 1]; th_B = u_phys[nB, 2]

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
        ea_L = EA/L; ei_L = EI/L; ei_L2 = EI/L**2; ei_L3 = EI/L**3

        K[:,0,0]= ea_L;  K[:,0,3]=-ea_L
        K[:,3,0]=-ea_L;  K[:,3,3]= ea_L
        K[:,1,1]= 12*ei_L3; K[:,1,2]= 6*ei_L2; K[:,1,4]=-12*ei_L3; K[:,1,5]= 6*ei_L2
        K[:,2,1]= 6*ei_L2;  K[:,2,2]= 4*ei_L;  K[:,2,4]=-6*ei_L2;  K[:,2,5]= 2*ei_L
        K[:,4,1]=-12*ei_L3; K[:,4,2]=-6*ei_L2; K[:,4,4]= 12*ei_L3; K[:,4,5]=-6*ei_L2
        K[:,5,1]= 6*ei_L2;  K[:,5,2]= 2*ei_L;  K[:,5,4]=-6*ei_L2;  K[:,5,5]= 4*ei_L

        Kd = torch.bmm(K, d_local.unsqueeze(2))
        U_per_elem = 0.5 * torch.bmm(d_local.unsqueeze(1), Kd).squeeze()
        return U_per_elem.sum()

    def _strain_energy_per_elem(self, u_phys, data):
        """Physical strain energy per element (for verification)."""
        conn = data.connectivity
        nA, nB = conn[:, 0], conn[:, 1]
        n_elem = conn.shape[0]
        device = u_phys.device

        L  = data.elem_lengths
        EA = data.prop_E * data.prop_A
        EI = data.prop_E * data.prop_I22
        c = data.elem_directions[:, 0]
        s = data.elem_directions[:, 2]

        ux_A = u_phys[nA, 0]; uz_A = u_phys[nA, 1]; th_A = u_phys[nA, 2]
        ux_B = u_phys[nB, 0]; uz_B = u_phys[nB, 1]; th_B = u_phys[nB, 2]

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
        ea_L = EA/L; ei_L = EI/L; ei_L2 = EI/L**2; ei_L3 = EI/L**3

        K[:,0,0]= ea_L;  K[:,0,3]=-ea_L
        K[:,3,0]=-ea_L;  K[:,3,3]= ea_L
        K[:,1,1]= 12*ei_L3; K[:,1,2]= 6*ei_L2; K[:,1,4]=-12*ei_L3; K[:,1,5]= 6*ei_L2
        K[:,2,1]= 6*ei_L2;  K[:,2,2]= 4*ei_L;  K[:,2,4]=-6*ei_L2;  K[:,2,5]= 2*ei_L
        K[:,4,1]=-12*ei_L3; K[:,4,2]=-6*ei_L2; K[:,4,4]= 12*ei_L3; K[:,4,5]=-6*ei_L2
        K[:,5,1]= 6*ei_L2;  K[:,5,2]= 2*ei_L;  K[:,5,4]=-6*ei_L2;  K[:,5,5]= 4*ei_L

        Kd = torch.bmm(K, d_local.unsqueeze(2))
        U_per_elem = 0.5 * torch.bmm(d_local.unsqueeze(1), Kd).squeeze()
        return U_per_elem

    def _external_work(self, u_phys, data):
        """Physical external work (for verification/logging)."""
        W = (
            data.F_ext[:, 0] * u_phys[:, 0]
            + data.F_ext[:, 1] * u_phys[:, 1]
            + data.F_ext[:, 2] * u_phys[:, 2]
        ).sum()
        return W