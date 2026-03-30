# """
# =================================================================
# normalizer.py — Per-Case Physics Scales
# =================================================================
# """

# import torch
# import numpy as np
# from typing import List

# try:
#     from torch_geometric.data import Data
# except ImportError:
#     raise ImportError("pip install torch-geometric")


# # ================================================================
# # A. MIN-MAX INPUT NORMALIZATION (unchanged)
# # ================================================================

# class MinMaxNormalizer:

#     def __init__(self):
#         self.x_min    = None
#         self.x_range  = None
#         self.ea_min   = None
#         self.ea_range = None
#         self.node_skip_cols = [3, 4, 9]
#         self.is_fitted = False

#     def fit(self, data_list: List[Data]):
#         all_x  = torch.cat(
#             [d.x for d in data_list], dim=0
#         )
#         all_ea = torch.cat(
#             [d.edge_attr for d in data_list], dim=0
#         )

#         self.x_min, _  = all_x.min(dim=0)
#         self.x_max, _  = all_x.max(dim=0)
#         self.ea_min, _ = all_ea.min(dim=0)
#         self.ea_max, _ = all_ea.max(dim=0)

#         self.x_range  = self.x_max - self.x_min
#         self.ea_range = self.ea_max - self.ea_min

#         self.x_range[self.x_range < 1e-10]   = 1.0
#         self.ea_range[self.ea_range < 1e-10]  = 1.0

#         for c in self.node_skip_cols:
#             self.x_min[c]   = 0.0
#             self.x_range[c] = 1.0

#         self.is_fitted = True
#         self._print_stats()

#     def _print_stats(self):
#         print(f"\n  Min-Max Normalization:")
#         node_labels = [
#             'x', 'y', 'z', 'bc_d*', 'bc_r*',
#             'pl_x', 'pl_y', 'pl_z', 'My', 'resp*',
#         ]
#         for i, label in enumerate(node_labels):
#             mn = self.x_min[i].item()
#             rn = self.x_range[i].item()
#             skip = (' (skip)'
#                     if i in self.node_skip_cols else '')
#             print(f"    node {label:<8} "
#                   f"[{mn:.4e}, {mn+rn:.4e}]{skip}")

#         edge_labels = [
#             'L', 'dx', 'dy', 'dz', 'E', 'A', 'I22'
#         ]
#         for i, label in enumerate(edge_labels):
#             mn = self.ea_min[i].item()
#             rn = self.ea_range[i].item()
#             print(f"    edge {label:<8} "
#                   f"[{mn:.4e}, {mn+rn:.4e}]")

#     def transform(self, data: Data) -> Data:
#         assert self.is_fitted
#         data = data.clone()
#         data.x = (data.x - self.x_min) / self.x_range
#         data.edge_attr = (
#             (data.edge_attr - self.ea_min) / self.ea_range
#         )
#         data.coord_min   = self.x_min[0:3].clone()
#         data.coord_range = self.x_range[0:3].clone()
#         return data

#     def transform_list(self, data_list):
#         return [self.transform(d) for d in data_list]

#     def save(self, filepath: str):
#         torch.save({
#             'x_min': self.x_min,
#             'x_range': self.x_range,
#             'ea_min': self.ea_min,
#             'ea_range': self.ea_range,
#             'node_skip_cols': self.node_skip_cols,
#             'is_fitted': self.is_fitted,
#         }, filepath)
#         print(f"  MinMaxNormalizer saved: {filepath}")

#     @classmethod
#     def load(cls, filepath):
#         state = torch.load(filepath, weights_only=False)
#         norm = cls()
#         for k, v in state.items():
#             setattr(norm, k, v)
#         return norm


# # ================================================================
# # B. PHYSICS SCALER — PER-CASE, DATA-DRIVEN
# # ================================================================

# class PhysicsScaler:
#     """
#     Characteristic scales computed PER CASE.

#     u_c and theta_c from EACH case's own displacements.
#     This ensures non-dim output ~O(1) for EVERY case.

#     F_c and M_c from external loads (per case).
#     """

#     @staticmethod
#     def compute_and_store(data: Data) -> Data:
#         data = data.clone()

#         L_c  = data.elem_lengths.max().item()
#         EI   = data.prop_E * data.prop_I22
#         EA   = data.prop_E * data.prop_A
#         EI_c = EI.max().item()
#         EA_c = EA.max().item()

#         # ── Force scales from F_ext ──
#         F_ext_max = data.F_ext.abs().max().item()
#         if F_ext_max < 1e-15:
#             F_ext_max = 1.0

#         F_c = F_ext_max
#         M_c = F_ext_max * L_c

#         if hasattr(data, 'point_moment_My'):
#             My_max = (
#                 data.point_moment_My.abs().max().item()
#             )
#             if My_max > 1e-15:
#                 M_c = max(M_c, My_max)

#         # ── Displacement scales from THIS case ──
#         if data.y_node is not None:
#             u_max = (
#                 data.y_node[:, 0:2].abs().max().item()
#             )
#             th_max = data.y_node[:, 2].abs().max().item()
#             u_c = max(u_max, 1e-15)
#             theta_c = max(th_max, 1e-15)
#         else:
#             # Fallback to beam theory
#             u_c = F_c * L_c**3 / EI_c
#             theta_c = F_c * L_c**2 / EI_c

#         # Safety
#         F_c     = max(F_c, 1e-15)
#         M_c     = max(M_c, 1e-15)
#         u_c     = max(u_c, 1e-15)
#         theta_c = max(theta_c, 1e-15)

#         data.L_c     = torch.tensor(
#             L_c, dtype=torch.float32
#         )
#         data.EI_c    = torch.tensor(
#             EI_c, dtype=torch.float32
#         )
#         data.EA_c    = torch.tensor(
#             EA_c, dtype=torch.float32
#         )
#         data.q_c     = torch.tensor(
#             F_c / L_c, dtype=torch.float32
#         )
#         data.F_c     = torch.tensor(
#             F_c, dtype=torch.float32
#         )
#         data.M_c     = torch.tensor(
#             M_c, dtype=torch.float32
#         )
#         data.u_c     = torch.tensor(
#             u_c, dtype=torch.float32
#         )
#         data.theta_c = torch.tensor(
#             theta_c, dtype=torch.float32
#         )

#         return data

#     @staticmethod
#     def compute_and_store_list(
#         data_list: List[Data]
#     ) -> List[Data]:

#         result = [
#             PhysicsScaler.compute_and_store(d)
#             for d in data_list
#         ]

#         print(f"\n  Physics Scales (per-case):")
#         print(f"  {'Scale':<12} {'Min':<16} {'Max':<16} "
#               f"{'Meaning'}")
#         print(f"  {'-'*70}")
#         for name, meaning in [
#             ('L_c',     'max element length'),
#             ('EI_c',    'max bending stiffness'),
#             ('EA_c',    'max axial stiffness'),
#             ('F_c',     'max external force'),
#             ('M_c',     'char. moment'),
#             ('u_c',     'max |disp| this case'),
#             ('theta_c', 'max |rot| this case'),
#         ]:
#             vals = [
#                 getattr(d, name).item() for d in result
#             ]
#             print(f"    {name:<10} [{min(vals):>12.4e}, "
#                   f"{max(vals):>12.4e}]  {meaning}")

#         # Verify non-dim ~O(1)
#         print(f"\n  Non-dim check (all cases):")
#         for i, d in enumerate(result):
#             if d.y_node is not None:
#                 ux_nd = (d.y_node[:, 0].abs().max().item()
#                          / d.u_c.item())
#                 uz_nd = (d.y_node[:, 1].abs().max().item()
#                          / d.u_c.item())
#                 th_nd = (d.y_node[:, 2].abs().max().item()
#                          / d.theta_c.item())
#                 ok = (ux_nd <= 1.01
#                       and uz_nd <= 1.01
#                       and th_nd <= 1.01)
#                 if i < 5 or not ok:
#                     print(
#                         f"    Case {i}: "
#                         f"ux/u_c={ux_nd:.3f}, "
#                         f"uz/u_c={uz_nd:.3f}, "
#                         f"θ/θ_c={th_nd:.3f}  "
#                         f"{'✅' if ok else '⚠️'}"
#                     )

#         return result

"""
=================================================================
normalizer.py — Per-Case Physics Scales (FIXED device placement)
=================================================================
"""

import torch
import numpy as np
from typing import List

try:
    from torch_geometric.data import Data
except ImportError:
    raise ImportError("pip install torch-geometric")


# ================================================================
# A. MIN-MAX INPUT NORMALIZATION (unchanged)
# ================================================================

class MinMaxNormalizer:

    def __init__(self):
        self.x_min    = None
        self.x_range  = None
        self.ea_min   = None
        self.ea_range = None
        self.node_skip_cols = [3, 4, 9]
        self.is_fitted = False

    def fit(self, data_list: List[Data]):
        all_x  = torch.cat(
            [d.x for d in data_list], dim=0
        )
        all_ea = torch.cat(
            [d.edge_attr for d in data_list], dim=0
        )

        self.x_min, _  = all_x.min(dim=0)
        self.x_max, _  = all_x.max(dim=0)
        self.ea_min, _ = all_ea.min(dim=0)
        self.ea_max, _ = all_ea.max(dim=0)

        self.x_range  = self.x_max - self.x_min
        self.ea_range = self.ea_max - self.ea_min

        self.x_range[self.x_range < 1e-10]   = 1.0
        self.ea_range[self.ea_range < 1e-10]  = 1.0

        for c in self.node_skip_cols:
            self.x_min[c]   = 0.0
            self.x_range[c] = 1.0

        self.is_fitted = True
        self._print_stats()

    def _print_stats(self):
        print(f"\n  Min-Max Normalization:")
        node_labels = [
            'x', 'y', 'z', 'bc_d*', 'bc_r*',
            'pl_x', 'pl_y', 'pl_z', 'My', 'resp*',
        ]
        for i, label in enumerate(node_labels):
            mn = self.x_min[i].item()
            rn = self.x_range[i].item()
            skip = (' (skip)'
                    if i in self.node_skip_cols else '')
            print(f"    node {label:<8} "
                  f"[{mn:.4e}, {mn+rn:.4e}]{skip}")

        edge_labels = [
            'L', 'dx', 'dy', 'dz', 'E', 'A', 'I22'
        ]
        for i, label in enumerate(edge_labels):
            mn = self.ea_min[i].item()
            rn = self.ea_range[i].item()
            print(f"    edge {label:<8} "
                  f"[{mn:.4e}, {mn+rn:.4e}]")

    def transform(self, data: Data) -> Data:
        assert self.is_fitted
        data = data.clone()
        data.x = (data.x - self.x_min) / self.x_range
        data.edge_attr = (
            (data.edge_attr - self.ea_min) / self.ea_range
        )
        data.coord_min   = self.x_min[0:3].clone()
        data.coord_range = self.x_range[0:3].clone()
        return data

    def transform_list(self, data_list):
        return [self.transform(d) for d in data_list]

    def save(self, filepath: str):
        torch.save({
            'x_min': self.x_min,
            'x_range': self.x_range,
            'ea_min': self.ea_min,
            'ea_range': self.ea_range,
            'node_skip_cols': self.node_skip_cols,
            'is_fitted': self.is_fitted,
        }, filepath)
        print(f"  MinMaxNormalizer saved: {filepath}")

    @classmethod
    def load(cls, filepath):
        state = torch.load(filepath, weights_only=False)
        norm = cls()
        for k, v in state.items():
            setattr(norm, k, v)
        return norm


# ================================================================
# B. PHYSICS SCALER — PER-CASE, DATA-DRIVEN (FIXED)
# ================================================================

class PhysicsScaler:
    """
    Characteristic scales computed PER CASE.

    u_c and theta_c from EACH case's own displacements.
    This ensures non-dim output ~O(1) for EVERY case.

    F_c and M_c from external loads (per case).
    
    FIX: All new tensors are created on the SAME device
    as the input data to prevent CPU/GPU mismatch during
    DataLoader collation.
    """

    @staticmethod
    def compute_and_store(data: Data) -> Data:
        data = data.clone()

        # ═══════════════════════════════════════
        # Get device from existing data tensors
        # ═══════════════════════════════════════
        device = data.x.device

        L_c  = data.elem_lengths.max().item()
        EI   = data.prop_E * data.prop_I22
        EA   = data.prop_E * data.prop_A
        EI_c = EI.max().item()
        EA_c = EA.max().item()

        # ── Force scales from F_ext ──
        F_ext_max = data.F_ext.abs().max().item()
        if F_ext_max < 1e-15:
            F_ext_max = 1.0

        F_c = F_ext_max
        M_c = F_ext_max * L_c

        if hasattr(data, 'point_moment_My'):
            My_max = (
                data.point_moment_My.abs().max().item()
            )
            if My_max > 1e-15:
                M_c = max(M_c, My_max)

        # ── Displacement scales from THIS case ──
        if data.y_node is not None:
            u_max = (
                data.y_node[:, 0:2].abs().max().item()
            )
            th_max = data.y_node[:, 2].abs().max().item()
            u_c = max(u_max, 1e-15)
            theta_c = max(th_max, 1e-15)
        else:
            # Fallback to beam theory
            u_c = F_c * L_c**3 / EI_c
            theta_c = F_c * L_c**2 / EI_c

        # Safety
        F_c     = max(F_c, 1e-15)
        M_c     = max(M_c, 1e-15)
        u_c     = max(u_c, 1e-15)
        theta_c = max(theta_c, 1e-15)

        # ═══════════════════════════════════════
        # Create tensors on the SAME device
        # ═══════════════════════════════════════
        data.L_c     = torch.tensor(
            L_c, dtype=torch.float32, device=device
        )
        data.EI_c    = torch.tensor(
            EI_c, dtype=torch.float32, device=device
        )
        data.EA_c    = torch.tensor(
            EA_c, dtype=torch.float32, device=device
        )
        data.q_c     = torch.tensor(
            F_c / L_c, dtype=torch.float32, device=device
        )
        data.F_c     = torch.tensor(
            F_c, dtype=torch.float32, device=device
        )
        data.M_c     = torch.tensor(
            M_c, dtype=torch.float32, device=device
        )
        data.u_c     = torch.tensor(
            u_c, dtype=torch.float32, device=device
        )
        data.theta_c = torch.tensor(
            theta_c, dtype=torch.float32, device=device
        )

        return data

    @staticmethod
    def compute_and_store_list(
        data_list: List[Data]
    ) -> List[Data]:

        result = [
            PhysicsScaler.compute_and_store(d)
            for d in data_list
        ]

        print(f"\n  Physics Scales (per-case):")
        print(f"  {'Scale':<12} {'Min':<16} {'Max':<16} "
              f"{'Meaning'}")
        print(f"  {'-'*70}")
        for name, meaning in [
            ('L_c',     'max element length'),
            ('EI_c',    'max bending stiffness'),
            ('EA_c',    'max axial stiffness'),
            ('F_c',     'max external force'),
            ('M_c',     'char. moment'),
            ('u_c',     'max |disp| this case'),
            ('theta_c', 'max |rot| this case'),
        ]:
            vals = [
                getattr(d, name).item() for d in result
            ]
            print(f"    {name:<10} [{min(vals):>12.4e}, "
                  f"{max(vals):>12.4e}]  {meaning}")

        # Verify non-dim ~O(1)
        print(f"\n  Non-dim check (all cases):")
        for i, d in enumerate(result):
            if d.y_node is not None:
                ux_nd = (d.y_node[:, 0].abs().max().item()
                         / d.u_c.item())
                uz_nd = (d.y_node[:, 1].abs().max().item()
                         / d.u_c.item())
                th_nd = (d.y_node[:, 2].abs().max().item()
                         / d.theta_c.item())
                ok = (ux_nd <= 1.01
                      and uz_nd <= 1.01
                      and th_nd <= 1.01)
                if i < 5 or not ok:
                    print(
                        f"    Case {i}: "
                        f"ux/u_c={ux_nd:.3f}, "
                        f"uz/u_c={uz_nd:.3f}, "
                        f"θ/θ_c={th_nd:.3f}  "
                        f"{'✅' if ok else '⚠️'}"
                    )

        return result