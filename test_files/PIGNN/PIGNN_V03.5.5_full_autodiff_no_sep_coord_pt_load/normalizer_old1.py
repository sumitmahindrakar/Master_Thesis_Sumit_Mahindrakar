"""
=================================================================
normalizer.py — Min-Max Normalization + Physics Scales
=================================================================

Two independent concerns:
  A. MinMaxNormalizer  — inputs to [0, 1]  (statistical)
  B. PhysicsScaler     — characteristic scales for loss (physics)
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
# A. MIN-MAX INPUT NORMALIZATION
# ================================================================

class MinMaxNormalizer:
    """
    Min-Max normalization for INPUT features.
    Maps to [0, 1]. Binary flags skipped.
    """

    def __init__(self):
        self.x_min    = None
        self.x_range  = None
        self.ea_min   = None
        self.ea_range = None
        self.node_skip_cols = [3, 4, 8]   # bc_disp, bc_rot, response
        self.is_fitted = False

    def fit(self, data_list: List[Data]):
        all_x  = torch.cat([d.x for d in data_list], dim=0)
        all_ea = torch.cat([d.edge_attr for d in data_list], dim=0)

        self.x_min, _  = all_x.min(dim=0)
        self.x_max, _  = all_x.max(dim=0)
        self.ea_min, _ = all_ea.min(dim=0)
        self.ea_max, _ = all_ea.max(dim=0)

        self.x_range  = self.x_max - self.x_min
        self.ea_range = self.ea_max - self.ea_min

        # Prevent div by zero for constant features
        self.x_range[self.x_range < 1e-10]   = 1.0
        self.ea_range[self.ea_range < 1e-10]  = 1.0

        # Skip binary flags
        for c in self.node_skip_cols:
            self.x_min[c]   = 0.0
            self.x_range[c] = 1.0

        self.is_fitted = True
        self._print_stats()

    def _print_stats(self):
        print(f"\n  Min-Max Normalization (INPUTS only):")
        print(f"  {'Feature':<20} {'Min':>12} {'Max':>12} {'Range':>12}")
        print(f"  {'-'*60}")

        node_labels = ['x','y','z','bc_d*','bc_r*',
                       'pl_x','pl_y','pl_z','resp*']
        for i, label in enumerate(node_labels):
            mn = self.x_min[i].item()
            rn = self.x_range[i].item()
            skip = '  (skip)' if i in self.node_skip_cols else ''
            print(f"    node {label:<8} {mn:>12.4e} "
                  f"{mn+rn:>12.4e} {rn:>12.4e}{skip}")

        edge_labels = ['L','dx','dy','dz','E','A','I22','qx','qy','qz']
        for i, label in enumerate(edge_labels):
            mn = self.ea_min[i].item()
            rn = self.ea_range[i].item()
            print(f"    edge {label:<8} {mn:>12.4e} "
                  f"{mn+rn:>12.4e} {rn:>12.4e}")

    def transform(self, data: Data) -> Data:
        assert self.is_fitted
        data = data.clone()
        data.x         = (data.x - self.x_min) / self.x_range
        data.edge_attr = (data.edge_attr - self.ea_min) / self.ea_range
        data.coord_min   = self.x_min[0:3].clone()
        data.coord_range = self.x_range[0:3].clone()
        return data

    def transform_list(self, data_list: List[Data]) -> List[Data]:
        return [self.transform(d) for d in data_list]

    def save(self, filepath: str):
        torch.save({
            'x_min': self.x_min, 'x_range': self.x_range,
            'ea_min': self.ea_min, 'ea_range': self.ea_range,
            'node_skip_cols': self.node_skip_cols,
            'is_fitted': self.is_fitted,
        }, filepath)
        print(f"  MinMaxNormalizer saved: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'MinMaxNormalizer':
        state = torch.load(filepath, weights_only=False)
        norm = cls()
        for k, v in state.items():
            setattr(norm, k, v)
        return norm


# ================================================================
# B. PHYSICS SCALER
# ================================================================

class PhysicsScaler:
    """
    Characteristic scales from structural mechanics.

    NOT statistical. Derived from beam theory:
      δ ~ FL³/EI     → u_c
      θ ~ FL²/EI     → theta_c
      F ~ qL         → F_c
      M ~ qL²        → M_c

    Stored PER CASE in the Data object.
    Analogous to blood flow paper's U_nor, A_nor, p_nor per artery.
    """

    @staticmethod
    def compute_and_store(data: Data) -> Data:
        data = data.clone()

        # ── Raw scales ──
        L_c  = data.elem_lengths.max().item()
        EI   = data.prop_E * data.prop_I22
        EA   = data.prop_E * data.prop_A
        EI_c = EI.max().item()
        EA_c = EA.max().item()

        # ── Load scale ──
        q_mag = data.elem_load.abs().max().item()
        if q_mag < 1e-15:
            q_mag = 1.0
        q_c = q_mag

        # ── Derived scales ──
        F_c     = q_c * L_c
        M_c     = q_c * L_c ** 2

        # Refine from F_ext if available
        if hasattr(data, 'F_ext') and data.F_ext is not None:
            F_ext_max = data.F_ext.abs().max().item()
            if F_ext_max > 1e-15:
                F_c = max(F_c, F_ext_max)
                M_c = max(M_c, F_ext_max * L_c)

        u_c     = F_c * L_c ** 3 / EI_c
        theta_c = F_c * L_c ** 2 / EI_c

        # ── Safety ──
        u_c     = max(u_c, 1e-15)
        theta_c = max(theta_c, 1e-15)
        F_c     = max(F_c, 1e-15)
        M_c     = max(M_c, 1e-15)

        # ── Store ──
        data.L_c     = torch.tensor(L_c,     dtype=torch.float32)
        data.EI_c    = torch.tensor(EI_c,    dtype=torch.float32)
        data.EA_c    = torch.tensor(EA_c,    dtype=torch.float32)
        data.q_c     = torch.tensor(q_c,     dtype=torch.float32)
        data.F_c     = torch.tensor(F_c,     dtype=torch.float32)
        data.M_c     = torch.tensor(M_c,     dtype=torch.float32)
        data.u_c     = torch.tensor(u_c,     dtype=torch.float32)
        data.theta_c = torch.tensor(theta_c, dtype=torch.float32)

        return data

    @staticmethod
    def compute_and_store_list(data_list: List[Data]) -> List[Data]:
        result = [PhysicsScaler.compute_and_store(d) for d in data_list]

        print(f"\n  Physics Scales:")
        print(f"  {'Scale':<12} {'Min':<16} {'Max':<16} {'Meaning'}")
        print(f"  {'-'*70}")
        for name, meaning in [
            ('L_c',     'max element length'),
            ('EI_c',    'max bending stiffness'),
            ('EA_c',    'max axial stiffness'),
            ('q_c',     'max load intensity'),
            ('F_c',     'char. force = q·L'),
            ('M_c',     'char. moment = q·L²'),
            ('u_c',     'char. disp = F·L³/EI'),
            ('theta_c', 'char. rot = F·L²/EI'),
        ]:
            vals = [getattr(d, name).item() for d in result]
            print(f"    {name:<10} [{min(vals):>12.4e}, "
                  f"{max(vals):>12.4e}]  {meaning}")

        # ── Verify O(1) ──
        d0 = result[0]
        if d0.y_node is not None:
            disp = d0.y_node
            ux_nd = disp[:, 0].abs().max().item() / d0.u_c.item()
            uz_nd = disp[:, 1].abs().max().item() / d0.u_c.item()
            th_nd = disp[:, 2].abs().max().item() / d0.theta_c.item()
            print(f"\n  Non-dim check (case 0):")
            print(f"    |ux|/u_c   = {ux_nd:.4f}  "
                  f"{'✅' if 0.001 < ux_nd < 100 else '⚠️'}")
            print(f"    |uz|/u_c   = {uz_nd:.4f}  "
                  f"{'✅' if 0.001 < uz_nd < 100 else '⚠️'}")
            print(f"    |θy|/θ_c   = {th_nd:.4f}  "
                  f"{'✅' if 0.001 < th_nd < 100 else '⚠️'}")

        return result