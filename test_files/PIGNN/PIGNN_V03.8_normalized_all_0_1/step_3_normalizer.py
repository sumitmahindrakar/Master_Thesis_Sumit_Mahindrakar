# # """
# # =================================================================
# # normalizer.py — Complete [0,1] Normalization
# # =================================================================
# # """

# # import torch
# # import numpy as np
# # from typing import List

# # try:
# #     from torch_geometric.data import Data
# # except ImportError:
# #     raise ImportError("pip install torch-geometric")


# # class CompleteNormalizer:
# #     """
# #     Normalize EVERYTHING to [0, 1]:
# #     - Geometry (coords, lengths)
# #     - Material properties (E, A, I)
# #     - Loads (F_ext, moments)
# #     - Displacements (u, θ) - if available
    
# #     Physics equations computed in normalized space.
# #     Only denormalize at final prediction.
# #     """
    
# #     def __init__(self):
# #         self.is_fitted = False
        
# #         # Geometry scales
# #         self.coord_min = None
# #         self.coord_max = None
# #         self.L_min = None
# #         self.L_max = None
        
# #         # Material scales
# #         self.E_min = None
# #         self.E_max = None
# #         self.A_min = None
# #         self.A_max = None
# #         self.I_min = None
# #         self.I_max = None
        
# #         # Load scales
# #         self.F_min = None
# #         self.F_max = None
# #         self.M_min = None
# #         self.M_max = None
        
# #         # Displacement scales (for denormalization)
# #         self.u_min = None
# #         self.u_max = None
# #         self.theta_min = None
# #         self.theta_max = None
    
# #     def fit(self, data_list: List[Data]):
# #         """Learn normalization parameters from dataset"""
        
# #         # Collect all values
# #         all_coords = torch.cat([d.coords for d in data_list], dim=0)
# #         all_L = torch.cat([d.elem_lengths for d in data_list], dim=0)
# #         all_E = torch.cat([d.prop_E for d in data_list], dim=0)
# #         all_A = torch.cat([d.prop_A for d in data_list], dim=0)
# #         all_I = torch.cat([d.prop_I22 for d in data_list], dim=0)
# #         all_F = torch.cat([d.F_ext for d in data_list], dim=0)
        
# #         # Geometry
# #         self.coord_min = all_coords.min(dim=0)[0]
# #         self.coord_max = all_coords.max(dim=0)[0]
# #         self.coord_range = (self.coord_max - self.coord_min).clamp(min=1e-10)
        
# #         self.L_min = all_L.min()
# #         self.L_max = all_L.max()
# #         self.L_range = (self.L_max - self.L_min).clamp(min=1e-10)
        
# #         # Material properties
# #         self.E_min = all_E.min()
# #         self.E_max = all_E.max()
# #         self.E_range = (self.E_max - self.E_min).clamp(min=1e-10)
        
# #         self.A_min = all_A.min()
# #         self.A_max = all_A.max()
# #         self.A_range = (self.A_max - self.A_min).clamp(min=1e-10)
        
# #         self.I_min = all_I.min()
# #         self.I_max = all_I.max()
# #         self.I_range = (self.I_max - self.I_min).clamp(min=1e-10)
        
# #         # Loads
# #         F_abs = all_F.abs()
# #         self.F_min = F_abs.min()
# #         self.F_max = F_abs.max()
# #         self.F_range = (self.F_max - self.F_min).clamp(min=1e-10)
        
# #         # Moments (if present)
# #         all_M = []
# #         for d in data_list:
# #             if hasattr(d, 'point_moment_My'):
# #                 all_M.append(d.point_moment_My)
# #         if all_M:
# #             all_M = torch.cat(all_M, dim=0)
# #             M_abs = all_M.abs()
# #             self.M_min = M_abs.min()
# #             self.M_max = M_abs.max()
# #             self.M_range = (self.M_max - self.M_min).clamp(min=1e-10)
# #         else:
# #             self.M_min = 0.0
# #             self.M_max = 1.0
# #             self.M_range = 1.0
        
# #         # Displacements (for output denormalization)
# #         all_u = []
# #         all_theta = []
# #         for d in data_list:
# #             if hasattr(d, 'y_node') and d.y_node is not None:
# #                 all_u.append(d.y_node[:, :2].abs())
# #                 all_theta.append(d.y_node[:, 2].abs())
        
# #         if all_u:
# #             all_u = torch.cat(all_u, dim=0)
# #             all_theta = torch.cat(all_theta, dim=0)
            
# #             self.u_min = all_u.min()
# #             self.u_max = all_u.max()
# #             self.u_range = (self.u_max - self.u_min).clamp(min=1e-10)
            
# #             self.theta_min = all_theta.min()
# #             self.theta_max = all_theta.max()
# #             self.theta_range = (self.theta_max - self.theta_min).clamp(min=1e-10)
# #         else:
# #             # Fallback: estimate from beam theory
# #             self.u_range = self.L_max * 0.1  # Assume ~10% deflection
# #             self.theta_range = 0.1  # Assume ~0.1 rad
# #             self.u_min = 0.0
# #             self.u_max = self.u_range
# #             self.theta_min = 0.0
# #             self.theta_max = self.theta_range
        
# #         self.is_fitted = True
# #         self._print_stats()
    
# #     def _print_stats(self):
# #         print(f"\n{'='*70}")
# #         print(f"  Complete Normalization to [0, 1]")
# #         print(f"{'='*70}")
        
# #         print(f"\n  Geometry:")
# #         print(f"    Coords:  [{self.coord_min[0]:.3e}, {self.coord_max[0]:.3e}]")
# #         print(f"    Lengths: [{self.L_min:.3e}, {self.L_max:.3e}]")
        
# #         print(f"\n  Material:")
# #         print(f"    E:       [{self.E_min:.3e}, {self.E_max:.3e}]")
# #         print(f"    A:       [{self.A_min:.3e}, {self.A_max:.3e}]")
# #         print(f"    I:       [{self.I_min:.3e}, {self.I_max:.3e}]")
        
# #         print(f"\n  Loads:")
# #         print(f"    F_ext:   [{self.F_min:.3e}, {self.F_max:.3e}]")
# #         print(f"    Moments: [{self.M_min:.3e}, {self.M_max:.3e}]")
        
# #         print(f"\n  Displacements (output scale):")
# #         print(f"    u:       [{self.u_min:.3e}, {self.u_max:.3e}]")
# #         print(f"    θ:       [{self.theta_min:.3e}, {self.theta_max:.3e}]")
# #         print(f"{'='*70}\n")
    
# #     def transform(self, data: Data) -> Data:
# #         """Normalize data to [0, 1]"""
# #         assert self.is_fitted, "Call fit() first"
        
# #         data = data.clone()
        
# #         # Normalize coordinates
# #         data.coords_norm = (data.coords - self.coord_min) / self.coord_range
        
# #         # Normalize element lengths
# #         data.elem_lengths_norm = (data.elem_lengths - self.L_min) / self.L_range
        
# #         # Normalize material properties
# #         data.prop_E_norm = (data.prop_E - self.E_min) / self.E_range
# #         data.prop_A_norm = (data.prop_A - self.A_min) / self.A_range
# #         data.prop_I22_norm = (data.prop_I22 - self.I_min) / self.I_range
        
# #         # Normalize loads (preserve sign!)
# #         data.F_ext_norm = torch.zeros_like(data.F_ext)
# #         data.F_ext_norm[:, 0] = data.F_ext[:, 0] / self.F_max
# #         data.F_ext_norm[:, 1] = data.F_ext[:, 1] / self.F_max
# #         data.F_ext_norm[:, 2] = data.F_ext[:, 2] / (self.M_max + 1e-10)
        
# #         # Normalize target displacements (if available)
# #         if hasattr(data, 'y_node') and data.y_node is not None:
# #             data.y_node_norm = torch.zeros_like(data.y_node)
# #             data.y_node_norm[:, 0] = data.y_node[:, 0] / self.u_max
# #             data.y_node_norm[:, 1] = data.y_node[:, 1] / self.u_max
# #             data.y_node_norm[:, 2] = data.y_node[:, 2] / self.theta_max
        
# #         # Store denormalization scales in data
# #         data.u_scale = self.u_max
# #         data.theta_scale = self.theta_max
# #         data.F_scale = self.F_max
# #         data.M_scale = self.M_max
# #         data.E_scale = self.E_max
# #         data.A_scale = self.A_max
# #         data.I_scale = self.I_max
# #         data.L_scale = self.L_max
        
# #         # Keep original data for validation
# #         data.coords_raw = data.coords.clone()
# #         data.F_ext_raw = data.F_ext.clone()
        
# #         return data
    
# #     def transform_list(self, data_list):
# #         return [self.transform(d) for d in data_list]
    
# #     def denormalize_displacement(self, u_norm, theta_norm):
# #         """Convert normalized output back to physical units"""
# #         u_phys = u_norm * self.u_max
# #         theta_phys = theta_norm * self.theta_max
# #         return u_phys, theta_phys
    
# #     def save(self, filepath: str):
# #         torch.save({
# #             'coord_min': self.coord_min,
# #             'coord_max': self.coord_max,
# #             'L_min': self.L_min,
# #             'L_max': self.L_max,
# #             'E_min': self.E_min,
# #             'E_max': self.E_max,
# #             'A_min': self.A_min,
# #             'A_max': self.A_max,
# #             'I_min': self.I_min,
# #             'I_max': self.I_max,
# #             'F_min': self.F_min,
# #             'F_max': self.F_max,
# #             'M_min': self.M_min,
# #             'M_max': self.M_max,
# #             'u_min': self.u_min,
# #             'u_max': self.u_max,
# #             'theta_min': self.theta_min,
# #             'theta_max': self.theta_max,
# #             'is_fitted': self.is_fitted,
# #         }, filepath)
# #         print(f"  CompleteNormalizer saved: {filepath}")
    
# #     @classmethod
# #     def load(cls, filepath):
# #         state = torch.load(filepath, weights_only=False)
# #         norm = cls()
# #         for k, v in state.items():
# #             setattr(norm, k, v)
# #         # Recompute ranges
# #         norm.coord_range = (norm.coord_max - norm.coord_min).clamp(min=1e-10)
# #         norm.L_range = (norm.L_max - norm.L_min).clamp(min=1e-10)
# #         norm.E_range = (norm.E_max - norm.E_min).clamp(min=1e-10)
# #         norm.A_range = (norm.A_max - norm.A_min).clamp(min=1e-10)
# #         norm.I_range = (norm.I_max - norm.I_min).clamp(min=1e-10)
# #         norm.F_range = (norm.F_max - norm.F_min).clamp(min=1e-10)
# #         norm.M_range = (norm.M_max - norm.M_min).clamp(min=1e-10)
# #         norm.u_range = (norm.u_max - norm.u_min).clamp(min=1e-10)
# #         norm.theta_range = (norm.theta_max - norm.theta_min).clamp(min=1e-10)
# #         return norm

# """
# =================================================================
# step_3_normalizer.py — Complete [0,1] Normalization
# =================================================================
# Normalizes ALL quantities to [0, 1]:
#   - Geometry (coords, lengths)
#   - Material properties (E, A, I)
#   - Loads (F_ext, moments)
#   - Displacements (u, θ) - if available

# Physics equations computed in normalized space.
# Only denormalize at final prediction.
# =================================================================
# """

# import torch
# import numpy as np
# from typing import List

# try:
#     from torch_geometric.data import Data
# except ImportError:
#     raise ImportError("pip install torch-geometric")


# class CompleteNormalizer:
#     """
#     Normalize EVERYTHING to [0, 1].
    
#     Strategy:
#       1. Fit: Learn min/max from entire dataset
#       2. Transform: Apply (x - min) / (max - min)
#       3. Store scales in each Data object for denormalization
#     """
    
#     def __init__(self):
#         self.is_fitted = False
        
#         # Geometry scales
#         self.coord_min = None
#         self.coord_max = None
#         self.L_min = None
#         self.L_max = None
        
#         # Material scales
#         self.E_min = None
#         self.E_max = None
#         self.A_min = None
#         self.A_max = None
#         self.I_min = None
#         self.I_max = None
        
#         # Load scales
#         self.F_min = None
#         self.F_max = None
#         self.M_min = None
#         self.M_max = None
        
#         # Displacement scales (for denormalization)
#         self.u_min = None
#         self.u_max = None
#         self.theta_min = None
#         self.theta_max = None
    
#     def fit(self, data_list: List[Data]):
#         """Learn normalization parameters from dataset"""
        
#         print(f"\n{'='*70}")
#         print(f"  Fitting CompleteNormalizer on {len(data_list)} cases")
#         print(f"{'='*70}")
        
#         # ════════════════════════════════════════════════
#         # Collect all values
#         # ════════════════════════════════════════════════
        
#         all_coords = torch.cat([d.coords for d in data_list], dim=0)
#         all_L = torch.cat([d.elem_lengths for d in data_list], dim=0)
#         all_E = torch.cat([d.prop_E for d in data_list], dim=0)
#         all_A = torch.cat([d.prop_A for d in data_list], dim=0)
#         all_I = torch.cat([d.prop_I22 for d in data_list], dim=0)
        
#         # F_ext: (N, 3) per case → flatten
#         all_F = torch.cat([d.F_ext for d in data_list], dim=0)
        
#         # ════════════════════════════════════════════════
#         # Geometry
#         # ════════════════════════════════════════════════
        
#         self.coord_min = all_coords.min(dim=0)[0]
#         self.coord_max = all_coords.max(dim=0)[0]
#         self.coord_range = (self.coord_max - self.coord_min).clamp(min=1e-10)
        
#         self.L_min = all_L.min()
#         self.L_max = all_L.max()
#         self.L_range = (self.L_max - self.L_min).clamp(min=1e-10)
        
#         # ════════════════════════════════════════════════
#         # Material properties
#         # ════════════════════════════════════════════════
        
#         self.E_min = all_E.min()
#         self.E_max = all_E.max()
#         self.E_range = (self.E_max - self.E_min).clamp(min=1e-10)
        
#         self.A_min = all_A.min()
#         self.A_max = all_A.max()
#         self.A_range = (self.A_max - self.A_min).clamp(min=1e-10)
        
#         self.I_min = all_I.min()
#         self.I_max = all_I.max()
#         self.I_range = (self.I_max - self.I_min).clamp(min=1e-10)
        
#         # ════════════════════════════════════════════════
#         # Loads (preserve sign, normalize magnitude)
#         # ════════════════════════════════════════════════
        
#         # Forces: Fx, Fz (columns 0, 2)
#         F_forces = all_F[:, [0, 2]]
#         F_abs = F_forces.abs()
#         self.F_max = F_abs.max().clamp(min=1e-10)
#         self.F_min = torch.tensor(0.0)
        
#         # Moments: My (column 1)
#         M_abs = all_F[:, 1].abs()
#         self.M_max = M_abs.max().clamp(min=1e-10)
#         self.M_min = torch.tensor(0.0)
        
#         # ════════════════════════════════════════════════
#         # Displacements (for output denormalization)
#         # ════════════════════════════════════════════════
        
#         all_u = []
#         all_theta = []
        
#         for d in data_list:
#             if hasattr(d, 'y_node') and d.y_node is not None:
#                 all_u.append(d.y_node[:, :2].abs())  # ux, uz
#                 all_theta.append(d.y_node[:, 2].abs())  # θy
        
#         if all_u:
#             all_u = torch.cat(all_u, dim=0)
#             all_theta = torch.cat(all_theta, dim=0)
            
#             self.u_max = all_u.max().clamp(min=1e-10)
#             self.u_min = torch.tensor(0.0)
            
#             self.theta_max = all_theta.max().clamp(min=1e-10)
#             self.theta_min = torch.tensor(0.0)
#         else:
#             # Fallback: estimate from beam theory
#             # Assume max deflection ~ L/100
#             self.u_max = self.L_max * 0.01
#             self.u_min = torch.tensor(0.0)
            
#             # Assume max rotation ~ 0.1 rad
#             self.theta_max = torch.tensor(0.1)
#             self.theta_min = torch.tensor(0.0)
        
#         self.is_fitted = True
#         self._print_stats()
    
#     def _print_stats(self):
#         print(f"\n  Normalization ranges:")
#         print(f"  {'Parameter':<15} {'Min':<12} {'Max':<12} {'Range':<12}")
#         print(f"  {'-'*55}")
        
#         # Geometry
#         print(f"  {'coord_x':<15} {self.coord_min[0]:>12.4e} {self.coord_max[0]:>12.4e} {self.coord_range[0]:>12.4e}")
#         print(f"  {'coord_z':<15} {self.coord_min[2]:>12.4e} {self.coord_max[2]:>12.4e} {self.coord_range[2]:>12.4e}")
#         print(f"  {'Length':<15} {self.L_min:>12.4e} {self.L_max:>12.4e} {self.L_range:>12.4e}")
        
#         # Material
#         print(f"  {'E':<15} {self.E_min:>12.4e} {self.E_max:>12.4e} {self.E_range:>12.4e}")
#         print(f"  {'A':<15} {self.A_min:>12.4e} {self.A_max:>12.4e} {self.A_range:>12.4e}")
#         print(f"  {'I22':<15} {self.I_min:>12.4e} {self.I_max:>12.4e} {self.I_range:>12.4e}")
        
#         # Loads
#         print(f"  {'F_ext':<15} {self.F_min:>12.4e} {self.F_max:>12.4e} {self.F_max:>12.4e}")
#         print(f"  {'Moment':<15} {self.M_min:>12.4e} {self.M_max:>12.4e} {self.M_max:>12.4e}")
        
#         # Displacements
#         print(f"  {'u':<15} {self.u_min:>12.4e} {self.u_max:>12.4e} {self.u_max:>12.4e}")
#         print(f"  {'theta':<15} {self.theta_min:>12.4e} {self.theta_max:>12.4e} {self.theta_max:>12.4e}")
        
#         print(f"{'='*70}\n")
    
#     def transform(self, data: Data) -> Data:
#         """Normalize data to [0, 1]"""
#         assert self.is_fitted, "Call fit() first"
        
#         data = data.clone()
        
#         # ════════════════════════════════════════════════
#         # Normalize geometry
#         # ════════════════════════════════════════════════
        
#         data.coords_norm = (data.coords - self.coord_min) / self.coord_range
#         data.elem_lengths_norm = (data.elem_lengths - self.L_min) / self.L_range
        
#         # ════════════════════════════════════════════════
#         # Normalize material properties
#         # ════════════════════════════════════════════════
        
#         data.prop_E_norm = (data.prop_E - self.E_min) / self.E_range
#         data.prop_A_norm = (data.prop_A - self.A_min) / self.A_range
#         data.prop_I22_norm = (data.prop_I22 - self.I_min) / self.I_range
        
#         # ════════════════════════════════════════════════
#         # Normalize loads (preserve sign!)
#         # ════════════════════════════════════════════════
                
#         data.F_ext_norm = torch.zeros_like(data.F_ext)
#         data.F_ext_norm[:, 0] = data.F_ext[:, 0] / self.F_max  # Fx / F_max
#         data.F_ext_norm[:, 1] = data.F_ext[:, 1] / self.F_max  # Fz / F_max
#         data.F_ext_norm[:, 2] = data.F_ext[:, 2] / self.M_max  # My / M_max
        
#         # ════════════════════════════════════════════════
#         # Normalize target displacements (if available)
#         # ════════════════════════════════════════════════
        
#         if hasattr(data, 'y_node') and data.y_node is not None:
#             data.y_node_norm = torch.zeros_like(data.y_node)
#             data.y_node_norm[:, 0] = data.y_node[:, 0] / self.u_max  # ux
#             data.y_node_norm[:, 1] = data.y_node[:, 1] / self.u_max  # uz
#             data.y_node_norm[:, 2] = data.y_node[:, 2] / self.theta_max  # θy
        
#         # ════════════════════════════════════════════════
#         # Store denormalization scales
#         # ════════════════════════════════════════════════
        
#         data.u_scale = self.u_max
#         data.theta_scale = self.theta_max
#         data.F_scale = self.F_max
#         data.M_scale = self.M_max
#         data.E_scale = self.E_max
#         data.A_scale = self.A_max
#         data.I_scale = self.I_max
#         data.L_scale = self.L_max
        
#         # Keep original data for validation
#         data.coords_raw = data.coords.clone()
#         data.F_ext_raw = data.F_ext.clone()
        
#         return data
    
#     def transform_list(self, data_list):
#         """Transform entire dataset"""
#         return [self.transform(d) for d in data_list]
    
#     def denormalize_displacement(self, u_norm, theta_norm):
#         """Convert normalized output back to physical units"""
#         u_phys = u_norm * self.u_max
#         theta_phys = theta_norm * self.theta_max
#         return u_phys, theta_phys
    
#     def save(self, filepath: str):
#         """Save normalizer state"""
#         torch.save({
#             'coord_min': self.coord_min,
#             'coord_max': self.coord_max,
#             'coord_range': self.coord_range,
#             'L_min': self.L_min,
#             'L_max': self.L_max,
#             'L_range': self.L_range,
#             'E_min': self.E_min,
#             'E_max': self.E_max,
#             'E_range': self.E_range,
#             'A_min': self.A_min,
#             'A_max': self.A_max,
#             'A_range': self.A_range,
#             'I_min': self.I_min,
#             'I_max': self.I_max,
#             'I_range': self.I_range,
#             'F_min': self.F_min,
#             'F_max': self.F_max,
#             'M_min': self.M_min,
#             'M_max': self.M_max,
#             'u_min': self.u_min,
#             'u_max': self.u_max,
#             'theta_min': self.theta_min,
#             'theta_max': self.theta_max,
#             'is_fitted': self.is_fitted,
#         }, filepath)
#         print(f"  ✓ Normalizer saved: {filepath}")
    
#     @classmethod
#     def load(cls, filepath):
#         """Load normalizer state"""
#         state = torch.load(filepath, weights_only=False)
#         norm = cls()
#         for k, v in state.items():
#             setattr(norm, k, v)
#         return norm


# # ════════════════════════════════════════════════════════════
# # VERIFICATION
# # ════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     import os
#     from pathlib import Path
    
#     CURRENT_SUBFOLDER = Path(__file__).resolve().parent
#     os.chdir(CURRENT_SUBFOLDER)
    
#     print("="*70)
#     print("  NORMALIZER VERIFICATION")
#     print("="*70)
    
#     # Load raw graph dataset
#     data_list = torch.load("DATA/graph_dataset.pt", weights_only=False)
#     print(f"\nLoaded {len(data_list)} graphs")
    
#     # Fit normalizer
#     normalizer = CompleteNormalizer()
#     normalizer.fit(data_list)
    
#     # Transform
#     data_norm = normalizer.transform(data_list[0])
    
#     # Verify ranges
#     print(f"\nVerification (case 0):")
#     print(f"  coords_norm:   [{data_norm.coords_norm.min():.4f}, {data_norm.coords_norm.max():.4f}]")
#     print(f"  E_norm:        [{data_norm.prop_E_norm.min():.4f}, {data_norm.prop_E_norm.max():.4f}]")
#     print(f"  F_ext_norm:    [{data_norm.F_ext_norm.min():.4f}, {data_norm.F_ext_norm.max():.4f}]")
    
#     if hasattr(data_norm, 'y_node_norm'):
#         print(f"  y_node_norm:   [{data_norm.y_node_norm.min():.4f}, {data_norm.y_node_norm.max():.4f}]")
    
#     # Check denormalization
#     if data_norm.y_node_norm is not None:
#         u_back, theta_back = normalizer.denormalize_displacement(
#             data_norm.y_node_norm[:, :2],
#             data_norm.y_node_norm[:, 2]
#         )
#         err_u = (u_back - data_list[0].y_node[:, :2]).abs().max()
#         err_theta = (theta_back - data_list[0].y_node[:, 2]).abs().max()
#         print(f"\n  Roundtrip error:")
#         print(f"    u:     {err_u:.4e}")
#         print(f"    theta: {err_theta:.4e}")
#         print(f"    {'✓ PASS' if err_u < 1e-6 and err_theta < 1e-6 else '✗ FAIL'}")
    
#     print(f"\n{'='*70}\n")

"""
=================================================================
step_3_normalizer.py — Hybrid Normalization
=================================================================
Handles datasets where material properties are constant.
Only normalizes VARYING quantities to [0,1].
=================================================================
"""

import torch
import numpy as np
from typing import List

try:
    from torch_geometric.data import Data
except ImportError:
    raise ImportError("pip install torch-geometric")


class CompleteNormalizer:
    """
    Hybrid normalizer for datasets with constant material properties.
    
    Normalizes to [0,1]:
      - Geometry (coords, lengths) - if they vary
      - Loads (forces, moments) - if they vary
      - Displacements (u, θ) - if they vary
    
    Keeps at physical scale:
      - Material properties (E, A, I) if constant
    """
    
    def __init__(self):
        self.is_fitted = False
        
        # Geometry scales
        self.coord_min = None
        self.coord_max = None
        self.L_min = None
        self.L_max = None
        
        # Material scales (may be constant)
        self.E_min = None
        self.E_max = None
        self.A_min = None
        self.A_max = None
        self.I_min = None
        self.I_max = None
        
        # Flags for constant properties
        self.E_is_constant = False
        self.A_is_constant = False
        self.I_is_constant = False
        
        # Material reference values (for constant case)
        self.E_ref = None
        self.A_ref = None
        self.I_ref = None
        
        # Load scales
        self.F_max = None
        self.M_max = None
        
        # Displacement scales
        self.u_max = None
        self.theta_max = None
    
    def fit(self, data_list: List[Data]):
        """Learn normalization parameters from dataset"""
        
        print(f"\n{'='*70}")
        print(f"  Fitting Hybrid Normalizer on {len(data_list)} cases")
        print(f"{'='*70}")
        
        # Collect all values
        all_coords = torch.cat([d.coords for d in data_list], dim=0)
        all_L = torch.cat([d.elem_lengths for d in data_list], dim=0)
        all_E = torch.cat([d.prop_E for d in data_list], dim=0)
        all_A = torch.cat([d.prop_A for d in data_list], dim=0)
        all_I = torch.cat([d.prop_I22 for d in data_list], dim=0)
        all_F = torch.cat([d.F_ext for d in data_list], dim=0)
        
        # ════════════════════════════════════════════════
        # Geometry (always normalize)
        # ════════════════════════════════════════════════
        
        self.coord_min = all_coords.min(dim=0)[0]
        self.coord_max = all_coords.max(dim=0)[0]
        self.coord_range = (self.coord_max - self.coord_min).clamp(min=1e-10)
        
        self.L_min = all_L.min()
        self.L_max = all_L.max()
        self.L_range = (self.L_max - self.L_min).clamp(min=1e-10)
        
        # ════════════════════════════════════════════════
        # Material properties (check if constant)
        # ════════════════════════════════════════════════
        
        # Young's modulus
        self.E_min = all_E.min()
        self.E_max = all_E.max()
        E_variation = (self.E_max - self.E_min) / self.E_max.clamp(min=1e-10)
        self.E_is_constant = (E_variation < 1e-6)
        
        if self.E_is_constant:
            self.E_ref = all_E.mean()
            print(f"  E is CONSTANT: {self.E_ref:.4e}")
        else:
            self.E_range = (self.E_max - self.E_min).clamp(min=1e-10)
            print(f"  E varies: [{self.E_min:.4e}, {self.E_max:.4e}]")
        
        # Cross-section area
        self.A_min = all_A.min()
        self.A_max = all_A.max()
        A_variation = (self.A_max - self.A_min) / self.A_max.clamp(min=1e-10)
        self.A_is_constant = (A_variation < 1e-6)
        
        if self.A_is_constant:
            self.A_ref = all_A.mean()
            print(f"  A is CONSTANT: {self.A_ref:.4e}")
        else:
            self.A_range = (self.A_max - self.A_min).clamp(min=1e-10)
            print(f"  A varies: [{self.A_min:.4e}, {self.A_max:.4e}]")
        
        # Second moment of area
        self.I_min = all_I.min()
        self.I_max = all_I.max()
        I_variation = (self.I_max - self.I_min) / self.I_max.clamp(min=1e-10)
        self.I_is_constant = (I_variation < 1e-6)
        
        if self.I_is_constant:
            self.I_ref = all_I.mean()
            print(f"  I22 is CONSTANT: {self.I_ref:.4e}")
        else:
            self.I_range = (self.I_max - self.I_min).clamp(min=1e-10)
            print(f"  I22 varies: [{self.I_min:.4e}, {self.I_max:.4e}]")
        
        # ════════════════════════════════════════════════
        # Characteristic scales for physics
        # ════════════════════════════════════════════════
        
        # If properties are constant, derive physics scales
        if self.E_is_constant and self.A_is_constant and self.I_is_constant:
            print(f"\n  Using physics-based scaling (constant properties)")
            
            # Characteristic stiffnesses
            EA_char = self.E_ref * self.A_ref
            EI_char = self.E_ref * self.I_ref
            L_char = self.L_max
            
            # Force scale from stiffness
            self.F_char = EA_char / L_char
            self.M_char = EI_char / L_char
            
            print(f"    EA = {EA_char:.4e}")
            print(f"    EI = {EI_char:.4e}")
            print(f"    L  = {L_char:.4e}")
            print(f"    F_char = EA/L = {self.F_char:.4e}")
            print(f"    M_char = EI/L = {self.M_char:.4e}")
        
        # ════════════════════════════════════════════════
        # Loads (normalize by max or by physics scale)
        # ════════════════════════════════════════════════
        
        F_forces = all_F[:, [0, 2]].abs()  # Fx, Fz
        F_abs_max = F_forces.max().clamp(min=1e-10)
        
        M_moments = all_F[:, 1].abs()  # My
        M_abs_max = M_moments.max().clamp(min=1e-10)
        
        # Use physics scale if available, otherwise data scale
        # if hasattr(self, 'F_char'):
        #     self.F_max = max(F_abs_max, self.F_char)
        #     self.M_max = max(M_abs_max, self.M_char)
        # else:
        #     self.F_max = F_abs_max
        #     self.M_max = M_abs_max
        # ALWAYS use data scale for loads (not physics scale)
        self.F_max = F_abs_max
        self.M_max = M_abs_max if M_abs_max > 1e-10 else F_abs_max

        print(f"\n  Load scales (from data):")
        print(f"    F_max = {self.F_max:.4e}")
        print(f"    M_max = {self.M_max:.4e}")
        
        print(f"\n  Load scales:")
        print(f"    F_max = {self.F_max:.4e}")
        print(f"    M_max = {self.M_max:.4e}")
        
        # ════════════════════════════════════════════════
        # Displacements (for denormalization)
        # ════════════════════════════════════════════════
        
        all_u = []
        all_theta = []
        
        for d in data_list:
            if hasattr(d, 'y_node') and d.y_node is not None:
                all_u.append(d.y_node[:, :2].abs())
                all_theta.append(d.y_node[:, 2].abs())
        
        if all_u:
            all_u = torch.cat(all_u, dim=0)
            all_theta = torch.cat(all_theta, dim=0)
            
            self.u_max = all_u.max().clamp(min=1e-10)
            self.theta_max = all_theta.max().clamp(min=1e-10)
        else:
            # Estimate from beam theory
            if hasattr(self, 'F_char'):
                self.u_max = self.F_max * self.L_max**3 / (EI_char * 48)  # Simply supported
                self.theta_max = self.F_max * self.L_max**2 / (EI_char * 16)
            else:
                self.u_max = self.L_max * 0.01
                self.theta_max = torch.tensor(0.1)
        
        print(f"\n  Displacement scales:")
        print(f"    u_max = {self.u_max:.4e}")
        print(f"    θ_max = {self.theta_max:.4e}")
        
        self.is_fitted = True
        print(f"{'='*70}\n")
    
    def transform(self, data: Data) -> Data:
        """Normalize data"""
        assert self.is_fitted, "Call fit() first"
        
        data = data.clone()
        
        # ════════════════════════════════════════════════
        # Normalize geometry
        # ════════════════════════════════════════════════
        
        data.coords_norm = (data.coords - self.coord_min) / self.coord_range
        data.elem_lengths_norm = (data.elem_lengths - self.L_min) / self.L_range
        
        # ════════════════════════════════════════════════
        # Normalize/scale material properties
        # ════════════════════════════════════════════════
        
        if self.E_is_constant:
            # Keep at reference value (don't normalize to [0,1])
            data.prop_E_norm = data.prop_E / self.E_ref
        else:
            data.prop_E_norm = (data.prop_E - self.E_min) / self.E_range
        
        if self.A_is_constant:
            data.prop_A_norm = data.prop_A / self.A_ref
        else:
            data.prop_A_norm = (data.prop_A - self.A_min) / self.A_range
        
        if self.I_is_constant:
            data.prop_I22_norm = data.prop_I22 / self.I_ref
        else:
            data.prop_I22_norm = (data.prop_I22 - self.I_min) / self.I_range
        
        # ════════════════════════════════════════════════
        # Normalize loads
        # ════════════════════════════════════════════════
        
        data.F_ext_norm = torch.zeros_like(data.F_ext)
        data.F_ext_norm[:, 0] = data.F_ext[:, 0] / self.F_max  # Fx
        data.F_ext_norm[:, 1] = data.F_ext[:, 1] / self.F_max  # Fz
        data.F_ext_norm[:, 2] = data.F_ext[:, 2] / self.M_max  # My
        
        # ════════════════════════════════════════════════
        # Normalize displacements
        # ════════════════════════════════════════════════
        
        if hasattr(data, 'y_node') and data.y_node is not None:
            data.y_node_norm = torch.zeros_like(data.y_node)
            data.y_node_norm[:, 0] = data.y_node[:, 0] / self.u_max
            data.y_node_norm[:, 1] = data.y_node[:, 1] / self.u_max
            data.y_node_norm[:, 2] = data.y_node[:, 2] / self.theta_max
        
        # ════════════════════════════════════════════════
        # Store scales for denormalization
        # ════════════════════════════════════════════════
        
        data.u_scale = self.u_max
        data.theta_scale = self.theta_max
        data.F_scale = self.F_max
        data.M_scale = self.M_max

        # ADD THESE LINES:
        data.E_scale = self.E_ref if self.E_is_constant else self.E_max
        data.A_scale = self.A_ref if self.A_is_constant else self.A_max
        data.I_scale = self.I_ref if self.I_is_constant else self.I_max
        data.L_scale = self.L_max

        # Store reference values
        data.E_ref = self.E_ref if self.E_is_constant else self.E_max
        data.A_ref = self.A_ref if self.A_is_constant else self.A_max
        data.I_ref = self.I_ref if self.I_is_constant else self.I_max

        # Flags
        data.E_is_constant = self.E_is_constant
        data.A_is_constant = self.A_is_constant
        data.I_is_constant = self.I_is_constant
        
        data.coords_raw = data.coords.clone()
        data.F_ext_raw = data.F_ext.clone()

        return data
    
    def transform_list(self, data_list):
        return [self.transform(d) for d in data_list]
    
    def save(self, filepath: str):
        torch.save({
            'coord_min': self.coord_min,
            'coord_max': self.coord_max,
            'coord_range': self.coord_range,
            'L_min': self.L_min,
            'L_max': self.L_max,
            'L_range': self.L_range,
            'E_min': self.E_min,
            'E_max': self.E_max,
            'A_min': self.A_min,
            'A_max': self.A_max,
            'I_min': self.I_min,
            'I_max': self.I_max,
            'E_is_constant': self.E_is_constant,
            'A_is_constant': self.A_is_constant,
            'I_is_constant': self.I_is_constant,
            'E_ref': self.E_ref,
            'A_ref': self.A_ref,
            'I_ref': self.I_ref,
            'F_max': self.F_max,
            'M_max': self.M_max,
            'u_max': self.u_max,
            'theta_max': self.theta_max,
            'is_fitted': self.is_fitted,
        }, filepath)
        print(f"  ✓ Normalizer saved: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        state = torch.load(filepath, weights_only=False)
        norm = cls()
        for k, v in state.items():
            setattr(norm, k, v)
        return norm