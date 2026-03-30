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
# B. PHYSICS SCALER — GLOBAL SCALE NOT PER CASE SCALES
# ================================================================

class PhysicsScaler:
    """
    Characteristic scales computed GLOBALLY across all cases.
    
    For varying loads, per-case normalization creates
    inconsistent gradient scales. Global normalization
    ensures all cases use the same scales.
    """

    @staticmethod
    def compute_and_store(data: Data, 
                          global_scales: dict = None) -> Data:
        data = data.clone()
        device = data.x.device

        L_c  = data.elem_lengths.max().item()
        EI   = data.prop_E * data.prop_I22
        EA   = data.prop_E * data.prop_A
        EI_c = EI.max().item()
        EA_c = EA.max().item()

        if global_scales is not None:
            # ═══════════════════════════════════════
            # USE GLOBAL SCALES (same for all cases)
            # ═══════════════════════════════════════
            F_c     = global_scales['F_c']
            M_c     = global_scales['M_c']
            u_c     = global_scales['u_c']
            theta_c = global_scales['theta_c']
        else:
            # ── Per-case fallback ──
            F_ext_max = data.F_ext.abs().max().item()
            if F_ext_max < 1e-15:
                F_ext_max = 1.0
            F_c = F_ext_max
            M_c = F_ext_max * L_c

            if hasattr(data, 'point_moment_My'):
                My_max = data.point_moment_My.abs().max().item()
                if My_max > 1e-15:
                    M_c = max(M_c, My_max)

            if data.y_node is not None:
                u_c = max(data.y_node[:, 0:2].abs().max().item(), 1e-15)
                theta_c = max(data.y_node[:, 2].abs().max().item(), 1e-15)
            else:
                u_c = F_c * L_c**3 / EI_c
                theta_c = F_c * L_c**2 / EI_c

        # Safety
        F_c     = max(F_c, 1e-15)
        M_c     = max(M_c, 1e-15)
        u_c     = max(u_c, 1e-15)
        theta_c = max(theta_c, 1e-15)

        data.L_c     = torch.tensor(L_c, dtype=torch.float32, device=device)
        data.EI_c    = torch.tensor(EI_c, dtype=torch.float32, device=device)
        data.EA_c    = torch.tensor(EA_c, dtype=torch.float32, device=device)
        data.q_c     = torch.tensor(F_c / L_c, dtype=torch.float32, device=device)
        data.F_c     = torch.tensor(F_c, dtype=torch.float32, device=device)
        data.M_c     = torch.tensor(M_c, dtype=torch.float32, device=device)
        data.u_c     = torch.tensor(u_c, dtype=torch.float32, device=device)
        data.theta_c = torch.tensor(theta_c, dtype=torch.float32, device=device)

        return data

    @staticmethod
    def compute_global_scales(data_list: list) -> dict:
        """
        Compute scales as MAX across ALL cases.
        This ensures consistent normalization.
        """
        F_c = max(d.F_ext.abs().max().item() for d in data_list)
        L_c = max(d.elem_lengths.max().item() for d in data_list)
        M_c = F_c * L_c

        # Check for moments
        for d in data_list:
            if hasattr(d, 'point_moment_My'):
                My_max = d.point_moment_My.abs().max().item()
                if My_max > 1e-15:
                    M_c = max(M_c, My_max)

        u_c = max(
            d.y_node[:, 0:2].abs().max().item()
            for d in data_list
            if d.y_node is not None
        )
        theta_c = max(
            d.y_node[:, 2].abs().max().item()
            for d in data_list
            if d.y_node is not None
        )

        scales = {
            'F_c':     max(F_c, 1e-15),
            'M_c':     max(M_c, 1e-15),
            'u_c':     max(u_c, 1e-15),
            'theta_c': max(theta_c, 1e-15),
        }

        print(f"\n  Global Physics Scales:")
        for k, v in scales.items():
            print(f"    {k:<10} = {v:.4e}")

        return scales

    @staticmethod
    def compute_and_store_list(data_list: list) -> list:
        """
        Compute GLOBAL scales, then apply to ALL cases.
        """
        # ═══════════════════════════════════════
        # Step 1: Find global max scales
        # ═══════════════════════════════════════
        global_scales = PhysicsScaler.compute_global_scales(data_list)

        # ═══════════════════════════════════════
        # Step 2: Apply same scales to every case
        # ═══════════════════════════════════════
        result = [
            PhysicsScaler.compute_and_store(d, global_scales)
            for d in data_list
        ]

        # ── Print verification ──
        print(f"\n  Non-dim check (global scales):")
        for i, d in enumerate(result):
            if d.y_node is not None:
                ux_nd = d.y_node[:, 0].abs().max().item() / d.u_c.item()
                uz_nd = d.y_node[:, 1].abs().max().item() / d.u_c.item()
                th_nd = d.y_node[:, 2].abs().max().item() / d.theta_c.item()
                if i < 5:
                    print(
                        f"    Case {i}: "
                        f"ux/u_c={ux_nd:.3f}, "
                        f"uz/u_c={uz_nd:.3f}, "
                        f"θ/θ_c={th_nd:.3f}"
                    )

        # Verify all cases have SAME scales
        F_cs = [d.F_c.item() for d in result]
        u_cs = [d.u_c.item() for d in result]
        all_same = (max(F_cs) == min(F_cs) and max(u_cs) == min(u_cs))
        print(f"\n  {'✓' if all_same else '✗'} All cases use same scales")
        print(f"    F_c = {result[0].F_c.item():.4e} (all cases)")
        print(f"    u_c = {result[0].u_c.item():.4e} (all cases)")
        print(f"    θ_c = {result[0].theta_c.item():.4e} (all cases)")

        return result