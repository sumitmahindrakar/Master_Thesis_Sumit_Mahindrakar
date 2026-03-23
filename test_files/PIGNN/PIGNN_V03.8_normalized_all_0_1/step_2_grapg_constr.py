# """
# =================================================================
# step_2_graph_constr.py — Build Graph Dataset with Complete Normalization
# =================================================================
# """

# import torch
# import numpy as np
# from torch_geometric.data import Data
# from pathlib import Path
# import os

# CURRENT_SUBFOLDER = Path(__file__).resolve().parent
# os.chdir(CURRENT_SUBFOLDER)


# class FrameData:
#     """Your existing FrameData class - keep as is"""
#     pass


# def build_graph_from_frame(frame_data: FrameData) -> Data:
#     """
#     Convert FrameData to PyTorch Geometric Data object.
    
#     This version builds features WITHOUT normalization.
#     Normalization happens later via CompleteNormalizer.
#     """
    
#     n_nodes = len(frame_data.nodes)
#     n_elements = len(frame_data.elements)
    
#     # ════════════════════════════════════════════════
#     # 1. Node coordinates (raw physical units)
#     # ════════════════════════════════════════════════
    
#     coords = torch.zeros(n_nodes, 3)
#     for i, node in enumerate(frame_data.nodes):
#         coords[i, 0] = node.x
#         coords[i, 1] = node.y  
#         coords[i, 2] = node.z
    
#     # ════════════════════════════════════════════════
#     # 2. Connectivity
#     # ════════════════════════════════════════════════
    
#     connectivity = torch.zeros(n_elements, 2, dtype=torch.long)
#     for i, elem in enumerate(frame_data.elements):
#         connectivity[i, 0] = elem.node_i
#         connectivity[i, 1] = elem.node_j
    
#     # ════════════════════════════════════════════════
#     # 3. Element properties (raw physical units)
#     # ════════════════════════════════════════════════
    
#     elem_lengths = torch.zeros(n_elements)
#     prop_E = torch.zeros(n_elements)
#     prop_A = torch.zeros(n_elements)
#     prop_I22 = torch.zeros(n_elements)
    
#     for i, elem in enumerate(frame_data.elements):
#         dx = coords[elem.node_j] - coords[elem.node_i]
#         elem_lengths[i] = torch.sqrt((dx**2).sum())
        
#         prop_E[i] = elem.E
#         prop_A[i] = elem.A
#         prop_I22[i] = elem.I22
    
#     # ════════════════════════════════════════════════
#     # 4. Boundary conditions (already 0/1)
#     # ════════════════════════════════════════════════
    
#     bc_disp = torch.zeros(n_nodes, 1)
#     bc_rot = torch.zeros(n_nodes, 1)
    
#     for i, node in enumerate(frame_data.nodes):
#         if node.bc_x or node.bc_y or node.bc_z:
#             bc_disp[i] = 1.0
#         if node.bc_rx or node.bc_ry or node.bc_rz:
#             bc_rot[i] = 1.0
    
#     # ════════════════════════════════════════════════
#     # 5. External loads (raw physical units)
#     # ════════════════════════════════════════════════
    
#     F_ext = torch.zeros(n_nodes, 3)
#     point_moment_My = torch.zeros(n_elements)
    
#     for i, node in enumerate(frame_data.nodes):
#         F_ext[i, 0] = node.F_x
#         F_ext[i, 1] = node.F_y
#         F_ext[i, 2] = node.F_z
    
#     for i, elem in enumerate(frame_data.elements):
#         if hasattr(elem, 'point_moment_y'):
#             point_moment_My[i] = elem.point_moment_y
    
#     # ════════════════════════════════════════════════
#     # 6. Ground truth displacements (if available)
#     # ════════════════════════════════════════════════
    
#     if hasattr(frame_data, 'solution'):
#         y_node = torch.zeros(n_nodes, 3)
#         for i, node in enumerate(frame_data.nodes):
#             if hasattr(node, 'disp_x'):
#                 y_node[i, 0] = node.disp_x
#                 y_node[i, 1] = node.disp_z  
#                 y_node[i, 2] = node.rot_y
#     else:
#         y_node = None
    
#     # ════════════════════════════════════════════════
#     # 7. Build edge_index (bidirectional)
#     # ════════════════════════════════════════════════
    
#     edge_index = torch.cat([
#         connectivity.t(),
#         connectivity.flip(1).t()
#     ], dim=1)
    
#     # ════════════════════════════════════════════════
#     # 8. Build RAW node features (before normalization)
#     # ════════════════════════════════════════════════
    
#     x = torch.cat([
#         coords,              # (N, 3) - will be normalized later
#         bc_disp,             # (N, 1) - already 0/1
#         bc_rot,              # (N, 1) - already 0/1
#         F_ext[:, :2],        # (N, 2) - will be normalized later
#         F_ext[:, 2:3],       # (N, 1) - will be normalized later
#         torch.zeros(n_nodes, 1),  # Placeholder for response type
#         torch.zeros(n_nodes, 1),  # Placeholder for other features
#     ], dim=1)  # Total: (N, 10)
    
#     # ════════════════════════════════════════════════
#     # 9. Build RAW edge features (before normalization)
#     # ════════════════════════════════════════════════
    
#     edge_attr_list = []
    
#     for i in range(connectivity.shape[0]):
#         node_i = connectivity[i, 0]
#         node_j = connectivity[i, 1]
        
#         dx = coords[node_j] - coords[node_i]
#         L = elem_lengths[i]
        
#         # Direction vector (will be normalized later, but store raw)
#         dx_norm = dx / (L + 1e-10)
        
#         edge_feat = torch.tensor([
#             L,                  # Length (raw)
#             dx_norm[0],         # dx/L (already dimensionless)
#             dx_norm[1],         # dy/L
#             dx_norm[2],         # dz/L
#             prop_E[i],          # E (raw)
#             prop_A[i],          # A (raw)
#             prop_I22[i],        # I (raw)
#         ])
        
#         edge_attr_list.append(edge_feat)
    
#     # Each edge appears twice (bidirectional)
#     edge_attr = torch.stack(edge_attr_list + edge_attr_list, dim=0)
    
#     # ════════════════════════════════════════════════
#     # 10. Create Data object
#     # ════════════════════════════════════════════════
    
#     data = Data(
#         x=x,
#         edge_index=edge_index,
#         edge_attr=edge_attr,
        
#         # Store raw geometric/material data
#         coords=coords,
#         connectivity=connectivity,
#         elem_lengths=elem_lengths,
#         prop_E=prop_E,
#         prop_A=prop_A,
#         prop_I22=prop_I22,
        
#         # BCs and loads
#         bc_disp=bc_disp,
#         bc_rot=bc_rot,
#         F_ext=F_ext,
#         point_moment_My=point_moment_My,
        
#         # Ground truth (if available)
#         y_node=y_node,
        
#         # Metadata
#         n_elements=n_elements,
#     )
    
#     return data


# def normalize_dataset(data_list):
#     """
#     Apply complete normalization to dataset.
    
#     This creates normalized versions of all quantities
#     and updates x and edge_attr accordingly.
#     """
#     from normalizer import CompleteNormalizer
    
#     print("\n" + "="*70)
#     print("  COMPLETE NORMALIZATION TO [0, 1]")
#     print("="*70)
    
#     # Fit normalizer
#     normalizer = CompleteNormalizer()
#     normalizer.fit(data_list)
    
#     # Transform all data
#     data_list_norm = normalizer.transform_list(data_list)
    
#     # Update x and edge_attr with normalized values
#     for data in data_list_norm:
#         # Rebuild node features with normalized quantities
#         data.x = torch.cat([
#             data.coords_norm,              # (N, 3) normalized coords
#             data.bc_disp,                  # (N, 1) flags (unchanged)
#             data.bc_rot,                   # (N, 1) flags (unchanged)
#             data.F_ext_norm[:, :2],        # (N, 2) normalized forces
#             data.F_ext_norm[:, 2:3],       # (N, 1) normalized moment
#             torch.zeros(data.num_nodes, 1), # response type
#             torch.zeros(data.num_nodes, 1), # other
#         ], dim=1)
        
#         # Rebuild edge features with normalized quantities
#         edge_attr_list = []
#         connectivity = data.connectivity
        
#         for i in range(connectivity.shape[0]):
#             node_i = connectivity[i, 0]
#             node_j = connectivity[i, 1]
            
#             # Direction vector (compute from normalized coords)
#             dx_norm = data.coords_norm[node_j] - data.coords_norm[node_i]
#             L_norm = torch.sqrt((dx_norm**2).sum())
#             dx_unit = dx_norm / (L_norm + 1e-10)
            
#             edge_feat = torch.tensor([
#                 data.elem_lengths_norm[i],  # Normalized length
#                 dx_unit[0],                  # dx/L (dimensionless)
#                 dx_unit[1],                  # dy/L
#                 dx_unit[2],                  # dz/L
#                 data.prop_E_norm[i],        # Normalized E
#                 data.prop_A_norm[i],        # Normalized A
#                 data.prop_I22_norm[i],      # Normalized I
#             ])
            
#             edge_attr_list.append(edge_feat)
        
#         # Bidirectional edges
#         data.edge_attr = torch.stack(edge_attr_list + edge_attr_list, dim=0)
    
#     # Save normalizer
#     normalizer.save("DATA/normalizer.pt")
    
#     return data_list_norm, normalizer


# # ════════════════════════════════════════════════════════════
# # MAIN: Build dataset
# # ════════════════════════════════════════════════════════════

# if __name__ == "__main__":
    
#     print("="*70)
#     print("  GRAPH DATASET CONSTRUCTION")
#     print("="*70)
    
#     # ────────────────────────────────────────────────
#     # 1. Load raw Kratos data
#     # ────────────────────────────────────────────────
    
#     raw_data_path = "DATA/frame_dataset.pkl"  # Your raw data file
#     print(f"\nLoading raw data from: {raw_data_path}")
    
#     frame_data_list = torch.load(raw_data_path, weights_only=False)
#     print(f"  Loaded {len(frame_data_list)} frames")
    
#     # ────────────────────────────────────────────────
#     # 2. Convert to PyG graphs (unnormalized)
#     # ────────────────────────────────────────────────
    
#     print(f"\nBuilding graphs...")
#     data_list = []
#     for i, frame_data in enumerate(frame_data_list):
#         data = build_graph_from_frame(frame_data)
#         data_list.append(data)
#         if (i+1) % 10 == 0:
#             print(f"  Built {i+1}/{len(frame_data_list)}")
    
#     # Save unnormalized version
#     torch.save(data_list, "DATA/graph_dataset.pt")
#     print(f"\n✓ Saved unnormalized dataset: DATA/graph_dataset.pt")
    
#     # ────────────────────────────────────────────────
#     # 3. Normalize dataset
#     # ────────────────────────────────────────────────
    
#     data_list_norm, normalizer = normalize_dataset(data_list)
    
#     # Save normalized version
#     torch.save(data_list_norm, "DATA/graph_dataset_norm.pt")
#     print(f"\n✓ Saved normalized dataset: DATA/graph_dataset_norm.pt")
    
#     # ────────────────────────────────────────────────
#     # 4. Verification
#     # ────────────────────────────────────────────────
    
#     print(f"\n" + "="*70)
#     print(f"  VERIFICATION")
#     print(f"="*70)
    
#     data = data_list_norm[0]
    
#     print(f"\n  Graph structure:")
#     print(f"    Nodes:     {data.num_nodes}")
#     print(f"    Edges:     {data.num_edges}")
#     print(f"    x.shape:   {data.x.shape}")
#     print(f"    edge_attr: {data.edge_attr.shape}")
    
#     print(f"\n  Normalized ranges:")
#     print(f"    x:              [{data.x.min():.4f}, {data.x.max():.4f}]")
#     print(f"    edge_attr:      [{data.edge_attr.min():.4f}, {data.edge_attr.max():.4f}]")
#     print(f"    coords_norm:    [{data.coords_norm.min():.4f}, {data.coords_norm.max():.4f}]")
#     print(f"    E_norm:         [{data.prop_E_norm.min():.4f}, {data.prop_E_norm.max():.4f}]")
#     print(f"    F_ext_norm:     [{data.F_ext_norm.min():.4f}, {data.F_ext_norm.max():.4f}]")
    
#     if data.y_node_norm is not None:
#         print(f"    y_node_norm:    [{data.y_node_norm.min():.4f}, {data.y_node_norm.max():.4f}]")
    
#     print(f"\n  Denormalization scales:")
#     print(f"    u_scale:     {data.u_scale:.4e}")
#     print(f"    theta_scale: {data.theta_scale:.4e}")
#     print(f"    F_scale:     {data.F_scale:.4e}")
#     print(f"    E_scale:     {data.E_scale:.4e}")
    
#     print(f"\n{'='*70}")
#     print(f"  DATASET READY ✓")
#     print(f"{'='*70}\n")

"""
=================================================================
step_2_graph_constr.py — Build Graph Dataset with Complete Normalization
=================================================================
Reads:  DATA/frame_dataset.pkl    (from step_1_data_loading.py)
Writes: DATA/graph_dataset.pt     (unnormalized)
        DATA/graph_dataset_norm.pt (normalized)
        DATA/normalizer.pt         (normalization parameters)
=================================================================
"""

import torch
import numpy as np
from torch_geometric.data import Data
from pathlib import Path
import os
import pickle

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)


def build_graph_from_frame(frame_case: dict) -> Data:
    """
    Convert frame case dict (from step_1) to PyTorch Geometric Data object.
    
    Args:
        frame_case: dict from step_1_data_loading.py with keys:
                    - coords (N, 3)
                    - point_load (N, 3)
                    - point_moment_My (N, 1)
                    - connectivity (E, 2)
                    - bc_disp (N, 1)
                    - bc_rot (N, 1)
                    - elem_lengths (E,)
                    - young_modulus (E,)
                    - cross_area (E,)
                    - I22 (E,)
                    - etc.
    
    Returns:
        PyG Data object with raw (unnormalized) features
    """
    
    N = frame_case['n_nodes']
    E = frame_case['n_elements']
    
    # ════════════════════════════════════════════════
    # 1. Extract data from frame_case dict
    # ════════════════════════════════════════════════
    
    coords = torch.from_numpy(frame_case['coords']).float()
    connectivity = torch.from_numpy(frame_case['connectivity']).long()
    
    # Material properties (element-level)
    elem_lengths = torch.from_numpy(frame_case['elem_lengths']).float()
    prop_E = torch.from_numpy(frame_case['young_modulus']).float()
    prop_A = torch.from_numpy(frame_case['cross_area']).float()
    prop_I22 = torch.from_numpy(frame_case['I22']).float()
    
    # Boundary conditions (node-level)
    bc_disp = torch.from_numpy(frame_case['bc_disp']).float()
    bc_rot = torch.from_numpy(frame_case['bc_rot']).float()
    
    # Loads (node-level) - PER-NODE VARYING
    point_load = torch.from_numpy(frame_case['point_load']).float()  # (N, 3)
    point_moment_My = torch.from_numpy(frame_case['point_moment_My']).float()  # (N, 1)
    
    # Response node flag
    response_node_flag = torch.from_numpy(frame_case['response_node_flag']).float()
    
    # Ground truth displacements (if available)
    if frame_case['nodal_disp_2d'] is not None:
        y_node = torch.from_numpy(frame_case['nodal_disp_2d']).float()  # (N, 3): [u_x, u_z, φ_y]
    else:
        y_node = None
    
    # ════════════════════════════════════════════════
    # 2. Build edge_index (bidirectional)
    # ════════════════════════════════════════════════
    
    edge_index = torch.cat([
        connectivity.t(),           # (2, E) forward
        connectivity.flip(1).t()    # (2, E) backward
    ], dim=1)  # (2, 2E)
    
    # ════════════════════════════════════════════════
    # 3. Build RAW node features (before normalization)
    # ════════════════════════════════════════════════
    # Total: 10 features per node
    
    x = torch.cat([
        coords,                     # (N, 3) - x, y, z
        bc_disp,                    # (N, 1) - displacement BC flag
        bc_rot,                     # (N, 1) - rotation BC flag
        point_load[:, [0, 2]],      # (N, 2) - Fx, Fz (skip Fy for 2D)
        point_moment_My,            # (N, 1) - My
        response_node_flag,         # (N, 1) - response location
        torch.zeros(N, 1),          # (N, 1) - placeholder/padding
    ], dim=1)  # (N, 10)
    
    # ════════════════════════════════════════════════
    # 4. Build RAW edge features (before normalization)
    # ════════════════════════════════════════════════
    # Total: 7 features per edge
    
    edge_attr_list = []
    
    for i in range(E):
        node_i = connectivity[i, 0]
        node_j = connectivity[i, 1]
        
        dx = coords[node_j] - coords[node_i]
        L = elem_lengths[i]
        
        # Direction vector (dimensionless)
        dx_norm = dx / (L + 1e-10)
        
        edge_feat = torch.tensor([
            L.item(),                   # Length (raw)
            dx_norm[0].item(),          # dx/L
            dx_norm[1].item(),          # dy/L
            dx_norm[2].item(),          # dz/L
            prop_E[i].item(),           # Young's modulus (raw)
            prop_A[i].item(),           # Cross-section area (raw)
            prop_I22[i].item(),         # Second moment of inertia (raw)
        ], dtype=torch.float32)
        
        edge_attr_list.append(edge_feat)
    
    # Each edge appears twice (bidirectional)
    edge_attr = torch.stack(edge_attr_list + edge_attr_list, dim=0)  # (2E, 7)
    
    # ════════════════════════════════════════════════
    # 5. External forces for physics (combined nodal representation)
    # ════════════════════════════════════════════════
    
        
    F_ext = torch.cat([
        point_load[:, [0]],         # Fx
        point_load[:, [2]],         # Fz
        point_moment_My,            # My
    ], dim=1)  # (N, 3)
    
    # ════════════════════════════════════════════════
    # 6. Create Data object
    # ════════════════════════════════════════════════
    
    data = Data(
        x=x,                        # (N, 10) node features
        edge_index=edge_index,      # (2, 2E)
        edge_attr=edge_attr,        # (2E, 7)
        
        # Store raw geometric/material data for physics
        coords=coords,              # (N, 3)
        connectivity=connectivity,  # (E, 2)
        elem_lengths=elem_lengths,  # (E,)
        prop_E=prop_E,              # (E,)
        prop_A=prop_A,              # (E,)
        prop_I22=prop_I22,          # (E,)
        
        # Boundary conditions and loads
        bc_disp=bc_disp,            # (N, 1)
        bc_rot=bc_rot,              # (N, 1)
        F_ext=F_ext,                # (N, 3) - combined [Fx, Fz, My]
        point_moment_My=point_moment_My,  # (N, 1) - keep separate for reference
        
        # Ground truth (if available)
        y_node=y_node,              # (N, 3) or None
        
        # Metadata
        n_elements=E,
        case_num=frame_case['case_num'],
    )
    
    return data


def normalize_dataset(data_list):
    """
    Apply complete normalization to dataset.
    
    This creates normalized versions of all quantities
    and updates x and edge_attr accordingly.
    """
    from step_3_normalizer import CompleteNormalizer
    
    print("\n" + "="*70)
    print("  COMPLETE NORMALIZATION TO [0, 1]")
    print("="*70)
    
    # Fit normalizer
    normalizer = CompleteNormalizer()
    normalizer.fit(data_list)
    
    # Transform all data
    data_list_norm = normalizer.transform_list(data_list)
    
    # Update x and edge_attr with normalized values
    for data in data_list_norm:
        # Rebuild node features with normalized quantities
        data.x = torch.cat([
            data.coords_norm,              # (N, 3) normalized coords
            data.bc_disp,                  # (N, 1) flags (unchanged)
            data.bc_rot,                   # (N, 1) flags (unchanged)
            data.F_ext_norm[:, :2],        # (N, 2) normalized Fx, Fz
            data.F_ext_norm[:, 2:3],       # (N, 1) normalized My
            torch.zeros(data.num_nodes, 1), # response flag or padding
            torch.zeros(data.num_nodes, 1), # padding
        ], dim=1)
        
        # Rebuild edge features with normalized quantities
        edge_attr_list = []
        connectivity = data.connectivity
        
        for i in range(connectivity.shape[0]):
            node_i = connectivity[i, 0]
            node_j = connectivity[i, 1]
            
            # Direction vector (compute from normalized coords)
            dx_norm = data.coords_norm[node_j] - data.coords_norm[node_i]
            L_norm = torch.sqrt((dx_norm**2).sum())
            dx_unit = dx_norm / (L_norm + 1e-10)
            
            edge_feat = torch.tensor([
                data.elem_lengths_norm[i].item(),  # Normalized length
                dx_unit[0].item(),                  # dx/L (dimensionless)
                dx_unit[1].item(),                  # dy/L
                dx_unit[2].item(),                  # dz/L
                data.prop_E_norm[i].item(),        # Normalized E
                data.prop_A_norm[i].item(),        # Normalized A
                data.prop_I22_norm[i].item(),      # Normalized I
            ], dtype=torch.float32)
            
            edge_attr_list.append(edge_feat)
        
        # Bidirectional edges
        data.edge_attr = torch.stack(edge_attr_list + edge_attr_list, dim=0)
    
    # Save normalizer
    normalizer.save("DATA/normalizer.pt")
    
    return data_list_norm, normalizer


# ════════════════════════════════════════════════════════════
# MAIN: Build dataset
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("="*70)
    print("  STEP 2: GRAPH DATASET CONSTRUCTION")
    print("="*70)
    
    # ────────────────────────────────────────────────
    # 1. Load frame data from step_1
    # ────────────────────────────────────────────────
    
    raw_data_path = "DATA/frame_dataset.pkl"  # ✅ MATCHES YOUR STEP 1
    print(f"\nLoading frame data from: {raw_data_path}")
    
    # Load using pickle (matches your step_1 save format)
    with open(raw_data_path, 'rb') as f:
        frame_data_list = pickle.load(f)
    
    print(f"  ✓ Loaded {len(frame_data_list)} frame cases")
    
    # Verify structure
    case0 = frame_data_list[0]
    print(f"\n  Verification (case 0):")
    print(f"    Nodes:     {case0['n_nodes']}")
    print(f"    Elements:  {case0['n_elements']}")
    print(f"    Keys:      {len(case0.keys())} fields")
    
    # ────────────────────────────────────────────────
    # 2. Convert to PyG graphs (unnormalized)
    # ────────────────────────────────────────────────
    
    print(f"\n  Building PyG graphs...")
    data_list = []
    for i, frame_case in enumerate(frame_data_list):
        data = build_graph_from_frame(frame_case)
        data_list.append(data)
        if (i+1) % 10 == 0:
            print(f"    Built {i+1}/{len(frame_data_list)}")
    
    # Save unnormalized version
    torch.save(data_list, "DATA/graph_dataset.pt")
    print(f"\n  ✓ Saved unnormalized: DATA/graph_dataset.pt")
    
    # ────────────────────────────────────────────────
    # 3. Normalize dataset
    # ────────────────────────────────────────────────
    
    data_list_norm, normalizer = normalize_dataset(data_list)
    
    # Save normalized version
    torch.save(data_list_norm, "DATA/graph_dataset_norm.pt")
    print(f"\n  ✓ Saved normalized: DATA/graph_dataset_norm.pt")
    
    # ────────────────────────────────────────────────
    # 4. Verification
    # ────────────────────────────────────────────────
    
    print(f"\n" + "="*70)
    print(f"  VERIFICATION")
    print(f"="*70)
    
    data = data_list_norm[0]
    
    print(f"\n  Graph structure:")
    print(f"    Nodes:     {data.num_nodes}")
    print(f"    Edges:     {data.num_edges}")
    print(f"    x.shape:   {data.x.shape}")
    print(f"    edge_attr: {data.edge_attr.shape}")
    
    print(f"\n  Normalized ranges:")
    print(f"    x:              [{data.x.min():.4f}, {data.x.max():.4f}]")
    print(f"    edge_attr:      [{data.edge_attr.min():.4f}, {data.edge_attr.max():.4f}]")
    print(f"    coords_norm:    [{data.coords_norm.min():.4f}, {data.coords_norm.max():.4f}]")
    print(f"    E_norm:         [{data.prop_E_norm.min():.4f}, {data.prop_E_norm.max():.4f}]")
    print(f"    F_ext_norm:     [{data.F_ext_norm.min():.4f}, {data.F_ext_norm.max():.4f}]")
    
    if data.y_node_norm is not None:
        print(f"    y_node_norm:    [{data.y_node_norm.min():.4f}, {data.y_node_norm.max():.4f}]")
    
    print(f"\n  Denormalization scales:")
    print(f"    u_scale:     {data.u_scale:.4e}")
    print(f"    theta_scale: {data.theta_scale:.4e}")
    print(f"    F_scale:     {data.F_scale:.4e}")
    print(f"    E_scale:     {data.E_scale:.4e}")
    
    print(f"\n{'='*70}")
    print(f"  STEP 2 COMPLETE ✓")
    print(f"  Next: Run step_7_train.py")
    print(f"{'='*70}\n")