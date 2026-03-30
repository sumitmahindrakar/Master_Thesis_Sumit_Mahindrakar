# """
# =================================================================
# physics_loss.py — Non-dim Corotational Physics Loss
# =================================================================

# Everything in non-dimensional units.
# Residual is already non-dim from corotational module.
# No additional scaling needed in loss.
# =================================================================
# """

# import torch
# import torch.nn as nn
# from corotational import CorotationalBeam2D


# class CorotationalPhysicsLoss(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.beam = CorotationalBeam2D()

#     def forward(self, model, data):

#         # 1. Predict ~O(1)
#         pred_raw = model(data)

#         # 2. Corotational (all non-dim inside)
#         beam_result = self.beam(pred_raw, data)

#         # 3. Non-dim residual (already scaled)
#         res_nd = (beam_result['nodal_forces_nd']
#                 - beam_result['F_ext_nd'])

#         # 4. Mask free nodes
#         free_disp = (data.bc_disp.squeeze(-1) < 0.5)
#         free_rot  = (data.bc_rot.squeeze(-1) < 0.5)

#         # 5. Loss (already non-dim, no extra scaling)
#         if free_disp.any():
#             L_force = (res_nd[free_disp, 0].pow(2)
#                      + res_nd[free_disp, 1].pow(2)).mean()
#         else:
#             L_force = torch.tensor(
#                 0.0, device=pred_raw.device
#             )

#         if free_rot.any():
#             L_moment = res_nd[free_rot, 2].pow(2).mean()
#         else:
#             L_moment = torch.tensor(
#                 0.0, device=pred_raw.device
#             )

#         L_eq = L_force + L_moment

#         # 6. Diagnostics
#         phys_disp = beam_result['phys_disp']

#         loss_dict = {
#             'L_eq':       L_eq.item(),
#             'L_force':    L_force.item(),
#             'L_moment':   L_moment.item(),
#             'total':      L_eq.item(),
#             'N_range':    [beam_result['N_e'].min().item(),
#                           beam_result['N_e'].max().item()],
#             'M_range':    [beam_result['M1_e'].min().item(),
#                           beam_result['M1_e'].max().item()],
#             'V_range':    [beam_result['V_e'].min().item(),
#                           beam_result['V_e'].max().item()],
#             'ux_range':   [phys_disp[:, 0].min().item(),
#                           phys_disp[:, 0].max().item()],
#             'uz_range':   [phys_disp[:, 1].min().item(),
#                           phys_disp[:, 1].max().item()],
#             'th_range':   [phys_disp[:, 2].min().item(),
#                           phys_disp[:, 2].max().item()],
#             'raw_range':  [pred_raw.min().item(),
#                           pred_raw.max().item()],
#             'max_res_nd': res_nd[free_disp].abs().max()
#                           .item()
#                           if free_disp.any() else 0.0,
#         }

#         return L_eq, loss_dict, pred_raw, beam_result

"""
=================================================================
physics_loss.py — Enhanced Corotational Physics Loss
=================================================================

Additions:
  1. Load-weighted equilibrium (loaded nodes matter more)
  2. Global equilibrium (reactions balance applied loads)
  3. Non-zero initialization bias in model
=================================================================
"""

import torch
import torch.nn as nn
from corotational import CorotationalBeam2D


class CorotationalPhysicsLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.beam = CorotationalBeam2D()

    def forward(self, model, data):

        pred_raw = model(data)
        beam_result = self.beam(pred_raw, data)

        res_nd = (beam_result['nodal_forces_nd']
                - beam_result['F_ext_nd'])

        free_disp = (data.bc_disp.squeeze(-1) < 0.5)
        free_rot  = (data.bc_rot.squeeze(-1) < 0.5)
        sup_mask  = (data.bc_disp.squeeze(-1) > 0.5)

        F_c = data.F_c
        M_c = data.M_c

        # ════════════════════════════════════════
        # 1. Load-weighted nodal equilibrium
        # ════════════════════════════════════════

        # Nodes with load get higher weight
        load_mag = data.F_ext.abs().sum(dim=1)    # (N,)
        load_max = load_mag.max().clamp(min=1e-10)
        node_weight = (load_mag / load_max) + 0.1  # min 0.1
        node_weight = node_weight / node_weight.mean()  # normalize

        if free_disp.any():
            w_free = node_weight[free_disp]
            res_Fx = res_nd[free_disp, 0]
            res_Fz = res_nd[free_disp, 1]
            L_force = ((res_Fx.pow(2) + res_Fz.pow(2))
                       * w_free).mean()
        else:
            L_force = torch.tensor(
                0.0, device=pred_raw.device
            )

        if free_rot.any():
            w_rot = node_weight[free_rot]
            res_My = res_nd[free_rot, 2]
            L_moment = (res_My.pow(2) * w_rot).mean()
        else:
            L_moment = torch.tensor(
                0.0, device=pred_raw.device
            )

        # ════════════════════════════════════════
        # 2. Global equilibrium (reactions)
        # ════════════════════════════════════════

        # Total external force
        total_Fx_ext = data.F_ext[:, 0].sum()
        total_Fz_ext = data.F_ext[:, 1].sum()

        # Total reaction at supports
        nodal_f = beam_result['nodal_forces_nd']
        react_Fx = nodal_f[sup_mask, 0].sum() * F_c
        react_Fz = nodal_f[sup_mask, 1].sum() * F_c

        # Global: reactions + external = 0
        L_global = (
            ((react_Fx + total_Fx_ext) / F_c).pow(2)
          + ((react_Fz + total_Fz_ext) / F_c).pow(2)
        )

        # ════════════════════════════════════════
        # 3. Total loss
        # ════════════════════════════════════════

        L_eq = L_force + L_moment + 0.1 * L_global

        # ════════════════════════════════════════
        # Diagnostics
        # ════════════════════════════════════════

        phys_disp = beam_result['phys_disp']

        loss_dict = {
            'L_eq':       L_eq.item(),
            'L_force':    L_force.item(),
            'L_moment':   L_moment.item(),
            'L_global':   L_global.item(),
            'total':      L_eq.item(),
            'N_range':    [beam_result['N_e'].min().item(),
                          beam_result['N_e'].max().item()],
            'M_range':    [beam_result['M1_e'].min().item(),
                          beam_result['M1_e'].max().item()],
            'V_range':    [beam_result['V_e'].min().item(),
                          beam_result['V_e'].max().item()],
            'ux_range':   [phys_disp[:, 0].min().item(),
                          phys_disp[:, 0].max().item()],
            'uz_range':   [phys_disp[:, 1].min().item(),
                          phys_disp[:, 1].max().item()],
            'th_range':   [phys_disp[:, 2].min().item(),
                          phys_disp[:, 2].max().item()],
            'raw_range':  [pred_raw.min().item(),
                          pred_raw.max().item()],
            'max_res_nd': res_nd[free_disp].abs().max()
                          .item()
                          if free_disp.any() else 0.0,
        }

        return L_eq, loss_dict, pred_raw, beam_result