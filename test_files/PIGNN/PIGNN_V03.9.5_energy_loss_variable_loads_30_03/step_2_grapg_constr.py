"""
=================================================================
STEP 2: GRAPH CONSTRUCTION — Strong-Form PIGNN  (v2)
=================================================================
Changes from v1:
  - Added edge_forward_mask for correct batched MP
  - Updated __cat_dim__ for the new attribute
  Everything else is identical.
=================================================================
"""

import torch
import numpy as np
import pickle
import os
from typing import List, Tuple

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    print(f"[OK] torch {torch.__version__}")
    print(f"[OK] torch_geometric loaded")
except ImportError:
    raise ImportError("pip install torch-geometric")


# ================================================================
# CUSTOM DATA CLASS FOR PROPER BATCHING
# ================================================================

class FrameData(Data):
    """
    Custom Data class that tells PyG how to increment
    custom tensor attributes during batching.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'connectivity':
            return self.num_nodes
        if key == 'face_element_id':
            return self.n_elements
        if key == 'element_map':
            return self.n_elements
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'connectivity':
            return 0
        if key == 'face_element_id':
            return 0
        if key == 'face_mask':
            return 0
        if key == 'face_is_A_end':
            return 0
        if key == 'edge_forward_mask':          # ◄◄◄ NEW
            return 0                             # ◄◄◄ NEW
        return super().__cat_dim__(key, value, *args, **kwargs)


# ================================================================
# 2A. GRAPH BUILDER
# ================================================================

class FrameGraphBuilder:
    """
    Converts frame case dicts → PyG Data objects.
    Per-node loads (Fx, Fz) and moments (My).
    """

    NODE_INPUT_DIM  = 10
    EDGE_INPUT_DIM  = 7
    NODE_TARGET_DIM = 3
    ELEM_TARGET_DIM = 4

    # ─── EDGE INDEX ───

    def build_edge_index(self, connectivity: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        E = len(connectivity)
        src = np.concatenate([
            connectivity[:, 0], connectivity[:, 1]
        ])
        dst = np.concatenate([
            connectivity[:, 1], connectivity[:, 0]
        ])
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)
        element_map = np.tile(np.arange(E), 2).astype(np.int64)
        return edge_index, element_map

    # ─── NODE FEATURES ───

    def build_node_features(self, case: dict) -> np.ndarray:
        N = case['n_nodes']
        feat = np.zeros((N, self.NODE_INPUT_DIM),
                        dtype=np.float64)
        feat[:, 0:3] = case['coords'][:, :3]
        feat[:, 3:4] = case['bc_disp']
        feat[:, 4:5] = case['bc_rot']
        feat[:, 5:8] = case['point_load'][:, :3]
        feat[:, 8:9] = case['point_moment_My']
        feat[:, 9:10] = case['response_node_flag']
        return feat

    # ─── EDGE FEATURES ───

    def build_edge_features(self, case, element_map):
        E = case['n_elements']
        elem_feat = np.column_stack([
            case['elem_lengths'],
            case['elem_directions'],
            case['young_modulus'],
            case['cross_area'],
            case['I22'],
        ])

        edge_feat = np.zeros((2 * E, self.EDGE_INPUT_DIM),
                            dtype=np.float64)
        edge_feat[:E]  = elem_feat
        edge_feat[E:]  = elem_feat
        edge_feat[E:, 1:4] *= -1.0
        return edge_feat

    # ─── NODE TARGETS ───

    def build_node_targets(self, case: dict) -> np.ndarray:
        N = case['n_nodes']
        tgt = np.zeros((N, self.NODE_TARGET_DIM),
                        dtype=np.float64)
        if ('nodal_disp_2d' in case and
                case['nodal_disp_2d'] is not None):
            return case['nodal_disp_2d'].astype(np.float64)
        if case.get('displacement') is not None:
            tgt[:, 0] = case['displacement'][:, 0]
            tgt[:, 1] = case['displacement'][:, 2]
        if case.get('rotation') is not None:
            tgt[:, 2] = case['rotation'][:, 1]
        return tgt

    # ─── ELEMENT TARGETS ───

    def build_element_targets(self, case: dict) -> np.ndarray:
        E = case['n_elements']
        tgt = np.zeros((E, self.ELEM_TARGET_DIM),
                        dtype=np.float64)

        if ('elem_N' in case and
                case['elem_N'] is not None):
            tgt[:, 0] = case['elem_N']
        elif case.get('force') is not None:
            tgt[:, 0] = case['force'][:, 0]

        if ('elem_M' in case and
                case['elem_M'] is not None):
            tgt[:, 1] = case['elem_M']
        elif case.get('moment') is not None:
            tgt[:, 1] = case['moment'][:, 1]

        if ('elem_V' in case and
                case['elem_V'] is not None):
            tgt[:, 2] = case['elem_V']
        elif case.get('force') is not None:
            tgt[:, 2] = case['force'][:, 2]

        if case.get('I22_sensitivity') is not None:
            tgt[:, 3] = case['I22_sensitivity']

        return tgt

    # ─── FACE ASSIGNMENT ───

    def build_face_assignment(
        self,
        connectivity: np.ndarray,
        elem_directions: np.ndarray,
        n_nodes: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        E = len(connectivity)
        N = n_nodes

        face_mask = np.zeros((N, 4), dtype=np.float64)
        face_element_id = np.full((N, 4), -1, dtype=np.int64)
        face_is_A_end = np.full((N, 4), -1, dtype=np.int64)

        for e in range(E):
            nA, nB = connectivity[e]
            dx = elem_directions[e, 0]
            dz = elem_directions[e, 2]

            if abs(dx) > abs(dz):
                if dx > 0:
                    face_A, face_B = 0, 1
                else:
                    face_A, face_B = 1, 0
            else:
                if dz > 0:
                    face_A, face_B = 2, 3
                else:
                    face_A, face_B = 3, 2

            face_mask[nA, face_A] = 1.0
            face_element_id[nA, face_A] = e
            face_is_A_end[nA, face_A] = 1

            face_mask[nB, face_B] = 1.0
            face_element_id[nB, face_B] = e
            face_is_A_end[nB, face_B] = 0

        max_faces = face_mask.sum(axis=1).max()
        assert max_faces <= 4, \
            f"Node has {max_faces} faces — exceeds max 4!"

        n_connected = int(face_mask.sum())
        print(f"    Face assignment: {n_connected} connections "
              f"({E} elements × 2 ends)")
        print(f"    Max faces per node: {int(max_faces)}")

        return face_mask, face_element_id, face_is_A_end

    # ─── EQUIVALENT NODAL LOADS ───

    def build_equivalent_nodal_loads(
        self,
        point_load: np.ndarray,
        point_moment_My: np.ndarray,
        n_nodes: int
    ) -> np.ndarray:
        N = n_nodes
        F_ext = np.zeros((N, 3), dtype=np.float64)

        F_ext[:, 0] = point_load[:, 0]
        F_ext[:, 1] = point_load[:, 2]
        F_ext[:, 2] = point_moment_My.flatten()

        loaded = np.where(
            np.linalg.norm(F_ext, axis=1) > 1e-10
        )[0]
        print(f"    F_ext: {len(loaded)} nodes loaded")
        print(f"      Fx: [{F_ext[:, 0].min():.2f}, "
              f"{F_ext[:, 0].max():.2f}]")
        print(f"      Fz: [{F_ext[:, 1].min():.2f}, "
              f"{F_ext[:, 1].max():.2f}]")
        print(f"      My: [{F_ext[:, 2].min():.4f}, "
              f"{F_ext[:, 2].max():.4f}]")

        return F_ext

    # ─── BUILD SINGLE GRAPH ───

    def case_to_graph(self, case: dict,
                       case_id: int = 0) -> FrameData:
        conn = case['connectivity']
        N = case['n_nodes']
        E = case['n_elements']

        edge_index, element_map = self.build_edge_index(conn)
        node_feat = self.build_node_features(case)
        edge_feat = self.build_edge_features(case, element_map)
        node_tgt = self.build_node_targets(case)
        elem_tgt = self.build_element_targets(case)

        face_mask, face_element_id, face_is_A_end = \
            self.build_face_assignment(
                conn, case['elem_directions'], N
            )

        F_ext = self.build_equivalent_nodal_loads(
            case['point_load'],
            case['point_moment_My'],
            N
        )

        # ◄◄◄ NEW: Forward mask for batched message passing
        # First E edges are forward (A→B), next E are backward (B→A)
        # This mask survives PyG batching correctly:
        #   Graph 0: [True*E0 | False*E0]
        #   Graph 1: [True*E1 | False*E1]
        #   Batched: [True*E0 | False*E0 | True*E1 | False*E1]
        edge_forward_mask = torch.cat([
            torch.ones(E, dtype=torch.bool),
            torch.zeros(E, dtype=torch.bool)
        ])

        data = FrameData(
            # ── Graph structure ──
            x=torch.tensor(
                node_feat, dtype=torch.float32),
            edge_index=torch.tensor(
                edge_index, dtype=torch.long),
            edge_attr=torch.tensor(
                edge_feat, dtype=torch.float32),

            # ── Forward mask for momentum-conserving MP ──
            edge_forward_mask=edge_forward_mask,    # ◄◄◄ NEW

            # ── Targets ──
            y_node=torch.tensor(
                node_tgt, dtype=torch.float32),
            y_element=torch.tensor(
                elem_tgt, dtype=torch.float32),

            # ── Topology ──
            element_map=torch.tensor(
                element_map, dtype=torch.long),
            connectivity=torch.tensor(
                conn, dtype=torch.long),

            # ── Geometry ──
            coords=torch.tensor(
                case['coords'], dtype=torch.float32),
            elem_lengths=torch.tensor(
                case['elem_lengths'], dtype=torch.float32),
            elem_directions=torch.tensor(
                case['elem_directions'], dtype=torch.float32),

            # ── Material properties ──
            prop_E=torch.tensor(
                case['young_modulus'], dtype=torch.float32),
            prop_A=torch.tensor(
                case['cross_area'], dtype=torch.float32),
            prop_I22=torch.tensor(
                case['I22'], dtype=torch.float32),

            # ── Boundary conditions ──
            bc_disp=torch.tensor(
                case['bc_disp'], dtype=torch.float32),
            bc_rot=torch.tensor(
                case['bc_rot'], dtype=torch.float32),

            # ── Loads ──
            F_ext=torch.tensor(
                F_ext, dtype=torch.float32),
            point_moment_My=torch.tensor(
                case['point_moment_My'],
                dtype=torch.float32),

            # ── Face assignment ──
            face_mask=torch.tensor(
                face_mask, dtype=torch.float32),
            face_element_id=torch.tensor(
                face_element_id, dtype=torch.long),
            face_is_A_end=torch.tensor(
                face_is_A_end, dtype=torch.long),

            # ── Metadata ──
            num_nodes_val=N,
            n_elements=E,
            case_id=case_id,
            nearest_node_id=case['nearest_node_id'],
            traced_element_id=case['traced_element_id'],
        )

        return data

    # ─── BUILD FULL DATASET ───

    def build_dataset(self, cases: List[dict]) -> List[Data]:
        data_list = [
            self.case_to_graph(case, case_id=i)
            for i, case in enumerate(cases)
        ]

        d = data_list[0]
        print(f"\n  Built {len(data_list)} graphs")
        print(f"    Nodes per graph:    {d.num_nodes}")
        print(f"    Directed edges:     {d.edge_index.shape[1]}")
        print(f"    Node features:      {d.x.shape[1]}")
        print(f"    Edge features:      {d.edge_attr.shape[1]}")
        print(f"    Node targets:       {d.y_node.shape[1]}")
        print(f"    Element targets:    {d.y_element.shape[1]}")
        print(f"    Face mask:          {d.face_mask.shape}")
        print(f"    F_ext:              {d.F_ext.shape}")
        print(f"    point_moment_My:    {d.point_moment_My.shape}")
        print(f"    edge_forward_mask:  {d.edge_forward_mask.shape}"  # ◄◄◄ NEW
              f"  ({d.edge_forward_mask.sum()} fwd)")                 # ◄◄◄ NEW

        return data_list


# ================================================================
# REST OF FILE — UNCHANGED
# ================================================================
# split_dataset, create_dataloaders, verify_graph,
# visualize_frame_graph, and __main__ block remain identical.
# Just copy them from your existing file.
# ================================================================

def split_dataset(
    data_list: List[Data],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[list, list, list]:
    n = len(data_list)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=gen).tolist()

    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    train = [data_list[i] for i in idx[:n_train]]
    val = [data_list[i] for i in idx[n_train:n_train + n_val]]
    test = [data_list[i] for i in idx[n_train + n_val:]]
    if not test:
        test = val.copy()

    print(f"  Split: {len(train)} train, "
          f"{len(val)} val, {len(test)} test")
    return train, val, test


def create_dataloaders(train, val, test,
                        batch_size: int = 16) -> dict:
    loaders = {
        'train': DataLoader(
            train, batch_size=batch_size, shuffle=True),
        'val': DataLoader(
            val, batch_size=batch_size, shuffle=False),
        'test': DataLoader(
            test, batch_size=batch_size, shuffle=False),
    }
    for name, loader in loaders.items():
        print(f"    {name}: {len(loader.dataset)} graphs, "
              f"{len(loader)} batches")
    return loaders


def verify_graph(data: Data) -> bool:
    print(f"\n{'═'*60}")
    print(f"  GRAPH VERIFICATION")
    print(f"{'═'*60}")

    N = data.num_nodes
    E = data.n_elements
    all_ok = True

    print(f"\n  1. SHAPES:")
    checks = [
        ('x',               data.x.shape,         (N, 10)),
        ('edge_index',      data.edge_index.shape, (2, 2*E)),
        ('edge_attr',       data.edge_attr.shape,  (2*E, 7)),
        ('edge_forward_mask', data.edge_forward_mask.shape,  # ◄◄◄ NEW
         (2*E,)),                                             # ◄◄◄ NEW
        ('y_node',          data.y_node.shape,     (N, 3)),
        ('y_element',       data.y_element.shape,  (E, 4)),
        ('element_map',     data.element_map.shape,(2*E,)),
        ('connectivity',    data.connectivity.shape,(E, 2)),
        ('prop_E',          data.prop_E.shape,     (E,)),
        ('prop_A',          data.prop_A.shape,     (E,)),
        ('prop_I22',        data.prop_I22.shape,   (E,)),
        ('face_mask',       data.face_mask.shape,  (N, 4)),
        ('face_element_id', data.face_element_id.shape, (N, 4)),
        ('face_is_A_end',   data.face_is_A_end.shape,   (N, 4)),
        ('F_ext',           data.F_ext.shape,      (N, 3)),
        ('point_moment_My', data.point_moment_My.shape, (N, 1)),
    ]
    for name, actual, expected in checks:
        ok = actual == expected
        if not ok:
            all_ok = False
        print(f"    {'✓' if ok else '✗'} {name:<22} "
              f"{str(actual):<15} (expected {expected})")

    # ◄◄◄ NEW: Verify forward mask content
    print(f"\n  1b. FORWARD MASK:")                            # ◄◄◄ NEW
    fwd = data.edge_forward_mask                               # ◄◄◄ NEW
    n_fwd = fwd.sum().item()                                   # ◄◄◄ NEW
    n_bwd = (~fwd).sum().item()                                # ◄◄◄ NEW
    ok_mask = (n_fwd == E) and (n_bwd == E)                    # ◄◄◄ NEW
    if not ok_mask:                                            # ◄◄◄ NEW
        all_ok = False                                         # ◄◄◄ NEW
    print(f"    {'✓' if ok_mask else '✗'} "                    # ◄◄◄ NEW
          f"fwd={n_fwd}, bwd={n_bwd} (expected {E}, {E})")    # ◄◄◄ NEW

    # ── 2. Face assignment ──
    print(f"\n  2. FACE ASSIGNMENT:")
    fm = data.face_mask
    faces_per_node = fm.sum(dim=1)
    print(f"    Faces/node: min={int(faces_per_node.min())}, "
          f"max={int(faces_per_node.max())}, "
          f"mean={faces_per_node.mean():.1f}")
    print(f"    Total connections: {int(fm.sum())} "
          f"(expected: {2*E})")

    ok = int(fm.sum()) == 2 * E
    if not ok:
        all_ok = False
    print(f"    {'✓' if ok else '✗'} Connection count = 2E")

    face_names = ['+x', '-x', '+z', '-z']
    for f in range(4):
        count = int(fm[:, f].sum())
        print(f"    Face {face_names[f]}: {count}")

    # ── 3. Face consistency ──
    print(f"\n  3. FACE CONSISTENCY:")
    feid = data.face_element_id
    faa = data.face_is_A_end
    conn = data.connectivity

    n_checked = 0
    n_ok = 0
    for n in range(N):
        for f in range(4):
            if fm[n, f] > 0.5:
                e = feid[n, f].item()
                is_A = faa[n, f].item()
                nA, nB = conn[e].tolist()
                if is_A == 1:
                    ok = (nA == n)
                else:
                    ok = (nB == n)
                n_checked += 1
                if ok:
                    n_ok += 1

    print(f"    {n_ok}/{n_checked} correct")
    if n_ok != n_checked:
        all_ok = False
    print(f"    {'✓' if n_ok == n_checked else '✗'} "
          f"All faces consistent")

    # ── 4-10: identical to your existing code ──
    print(f"\n  4. EQUIVALENT NODAL LOADS (F_ext):")
    fext = data.F_ext
    print(f"    Fx: [{fext[:, 0].min():.4f}, "
          f"{fext[:, 0].max():.4f}]")
    print(f"    Fz: [{fext[:, 1].min():.4f}, "
          f"{fext[:, 1].max():.4f}]")
    print(f"    My: [{fext[:, 2].min():.6f}, "
          f"{fext[:, 2].max():.6f}]")

    loaded = torch.where(
        fext.abs().sum(dim=1) > 1e-10
    )[0]
    print(f"    Loaded nodes: {len(loaded)}/{N}")

    print(f"\n  5. NODE FEATURES (x):")
    print(f"    x[:, 0:3]  coords:     "
          f"[{data.x[:, :3].min():.2f}, "
          f"{data.x[:, :3].max():.2f}]")
    print(f"    x[:, 3]    bc_disp:    "
          f"{int(data.x[:, 3].sum())} nodes fixed")
    print(f"    x[:, 4]    bc_rot:     "
          f"{int(data.x[:, 4].sum())} nodes fixed")

    pl = data.x[:, 5:8]
    force_nodes = torch.where(
        pl.abs().sum(dim=1) > 1e-10
    )[0]
    print(f"    x[:, 5:8]  point_load: "
          f"{len(force_nodes)} nodes loaded")
    print(f"      Fx: [{pl[:, 0].min():.2f}, "
          f"{pl[:, 0].max():.2f}]")
    print(f"      Fy: [{pl[:, 1].min():.2f}, "
          f"{pl[:, 1].max():.2f}] (should ~0)")
    print(f"      Fz: [{pl[:, 2].min():.2f}, "
          f"{pl[:, 2].max():.2f}]")

    pm = data.x[:, 8]
    moment_nodes = torch.where(pm.abs() > 1e-10)[0]
    print(f"    x[:, 8]    My:         "
          f"{len(moment_nodes)} nodes")
    if len(moment_nodes) > 0:
        print(f"      My: [{pm.min():.4f}, "
              f"{pm.max():.4f}]")
    else:
        print(f"      My: all zeros (disabled)")

    resp = data.x[:, 9]
    resp_nodes = torch.where(resp > 0.5)[0]
    print(f"    x[:, 9]    response:   "
          f"node(s) {resp_nodes.tolist()}")

    fy_ok = torch.allclose(
        pl[:, 1], torch.zeros_like(pl[:, 1]), atol=1e-10
    )
    print(f"    {'✓' if fy_ok else '⚠'} Fy all zero (2D frame)")

    print(f"\n  6. EDGE FEATURES (edge_attr):")
    ea = data.edge_attr
    print(f"    [0]   length:  [{ea[:, 0].min():.4f}, "
          f"{ea[:, 0].max():.4f}]")
    print(f"    [1:4] dir:     [{ea[:, 1:4].min():.4f}, "
          f"{ea[:, 1:4].max():.4f}]")
    print(f"    [4]   E:       [{ea[:, 4].min():.4e}, "
          f"{ea[:, 4].max():.4e}]")
    print(f"    [5]   A:       [{ea[:, 5].min():.6f}, "
          f"{ea[:, 5].max():.6f}]")
    print(f"    [6]   I22:     [{ea[:, 6].min():.4e}, "
          f"{ea[:, 6].max():.4e}]")

    print(f"\n  7. SPECIAL NODES:")
    bc = data.bc_disp.squeeze(-1)
    supports = torch.where(bc > 0.5)[0].tolist()
    print(f"    Supports: {supports}")
    print(f"    Response: {resp_nodes.tolist()}")

    print(f"\n  8. PHYSICS:")
    EA = data.prop_E * data.prop_A
    EI = data.prop_E * data.prop_I22
    print(f"    EA: [{EA.min():.4e}, {EA.max():.4e}]")
    print(f"    EI: [{EI.min():.4e}, {EI.max():.4e}]")

    if hasattr(data, 'F_c'):
        print(f"\n  9. PHYSICS SCALES:")
        print(f"    L_c     = {data.L_c.item():.4e}")
        print(f"    EI_c    = {data.EI_c.item():.4e}")
        print(f"    EA_c    = {data.EA_c.item():.4e}")
        print(f"    q_c     = {data.q_c.item():.4e}")
        print(f"    F_c     = {data.F_c.item():.4e}")
        print(f"    M_c     = {data.M_c.item():.4e}")
        print(f"    u_c     = {data.u_c.item():.4e}")
        print(f"    theta_c = {data.theta_c.item():.4e}")
    else:
        print(f"\n  9. PHYSICS SCALES: not yet computed")

    print(f"\n  10. F_EXT CONSISTENCY:")
    fext_fx = data.F_ext[:, 0]
    node_fx = data.x[:, 5]
    fx_match = torch.allclose(fext_fx, node_fx, atol=1e-10)
    print(f"    {'✓' if fx_match else '✗'} "
          f"F_ext[:,0] == x[:,5] (Fx)")

    fext_fz = data.F_ext[:, 1]
    node_fz = data.x[:, 7]
    fz_match = torch.allclose(fext_fz, node_fz, atol=1e-10)
    print(f"    {'✓' if fz_match else '✗'} "
          f"F_ext[:,1] == x[:,7] (Fz)")

    fext_my = data.F_ext[:, 2]
    node_my = data.x[:, 8]
    my_match = torch.allclose(fext_my, node_my, atol=1e-10)
    print(f"    {'✓' if my_match else '✗'} "
          f"F_ext[:,2] == x[:,8] (My)")

    if not (fx_match and fz_match and my_match):
        all_ok = False

    status = "ALL PASSED ✓" if all_ok else "SOME FAILED ✗"
    print(f"\n  RESULT: {status}")
    print(f"{'═'*60}\n")
    return all_ok


def visualize_frame_graph(data: Data,
                           title: str = "Frame Graph"):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Arc
    except ImportError:
        print("  pip install matplotlib")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    coords = data.coords.numpy()
    conn = data.connectivity.numpy()
    x_feat = data.x.numpy()
    E = data.n_elements

    for e in range(E):
        n1, n2 = conn[e]
        xs = [coords[n1, 0], coords[n2, 0]]
        zs = [coords[n1, 2], coords[n2, 2]]
        dx = abs(coords[n2, 0] - coords[n1, 0])
        dz = abs(coords[n2, 2] - coords[n1, 2])
        color = 'steelblue' if dz > dx else 'coral'
        ax.plot(xs, zs, '-', color=color, linewidth=2.5,
                solid_capstyle='round')

    for i in range(data.num_nodes):
        xp, zp = coords[i, 0], coords[i, 2]
        is_support = x_feat[i, 3] > 0.5
        is_response = x_feat[i, 9] > 0.5

        if is_response:
            ax.plot(xp, zp, '*', ms=18, color='gold',
                    mec='black', zorder=6)
        elif is_support:
            ax.plot(xp, zp, '^', ms=12, color='green',
                    mec='black', zorder=5)
        else:
            ax.plot(xp, zp, 'o', ms=5, color='white',
                    mec='black', zorder=5)

        ax.text(xp + 0.15, zp + 0.15, f'{i}',
                fontsize=5, color='gray')

        fx = x_feat[i, 5]
        if abs(fx) > 1e-10:
            scale = 0.02
            ax.annotate(
                '', xy=(xp, zp),
                xytext=(xp - fx * scale, zp),
                arrowprops=dict(
                    arrowstyle='->', color='red',
                    lw=max(0.5, min(2.0, abs(fx) * 0.02))
                )
            )

        fz = x_feat[i, 7]
        if abs(fz) > 1e-10:
            scale = 0.02
            ax.annotate(
                '', xy=(xp, zp),
                xytext=(xp, zp - fz * scale),
                arrowprops=dict(
                    arrowstyle='->', color='red',
                    lw=max(0.5, min(2.0, abs(fz) * 0.02))
                )
            )

        my = x_feat[i, 8]
        if abs(my) > 1e-10:
            arc_size = max(0.3, abs(my) * 0.01)
            arc_color = 'green' if my > 0 else 'darkgreen'
            theta1 = 0 if my > 0 else 90
            theta2 = 270 if my > 0 else 360
            arc = Arc(
                (xp, zp), arc_size, arc_size,
                angle=0, theta1=theta1, theta2=theta2,
                color=arc_color, lw=1.5
            )
            ax.add_patch(arc)

    legend = [
        Line2D([0], [0], color='steelblue', lw=3,
               label='Column'),
        Line2D([0], [0], color='coral', lw=3,
               label='Beam'),
        Line2D([0], [0], marker='^', color='green', lw=0,
               ms=10, label='Support'),
        Line2D([0], [0], marker='*', color='gold', lw=0,
               ms=15, label='Response node'),
        Line2D([0], [0], marker='>', color='red', lw=0,
               ms=8, label='Force (Fx/Fz)'),
        Line2D([0], [0], color='green', lw=2,
               label='Moment (My)'),
    ]
    ax.legend(handles=legend, loc='upper right')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m] (height)')
    ax.set_title(
        f"{title}\n{data.num_nodes} nodes, "
        f"{E} elements, {2*E} directed edges"
    )
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("DATA", exist_ok=True)
    plt.savefig('DATA/frame_graph.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    print("  Saved: DATA/frame_graph.png")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    from pathlib import Path
    print(f"Working directory: {os.getcwd()}")
    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)
    print(f"Working directory: {os.getcwd()}")

    print("=" * 60)
    print("  STEP 2: Graph Construction (Per-Node Loads)")
    print("=" * 60)

    print("\n── Loading dataset from Step 1 ──")
    with open("DATA/frame_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    print(f"  Loaded {len(dataset)} cases")

    print("\n── Building PyG graphs ──")
    builder = FrameGraphBuilder()
    data_list = builder.build_dataset(dataset)

    print("\n── Verifying first graph ──")
    verify_graph(data_list[0])

    print("\n── Visualizing ──")
    visualize_frame_graph(
        data_list[0],
        title=f"Case {dataset[0]['case_num']}"
    )

    from normalizer import MinMaxNormalizer, PhysicsScaler

    print("\n── Computing physics scales ──")
    data_list = PhysicsScaler.compute_and_store_list(data_list)

    print("\n── Normalizing inputs (Min-Max → [0,1]) ──")
    normalizer = MinMaxNormalizer()
    normalizer.fit(data_list)
    data_list_norm = normalizer.transform_list(data_list)

    print("\n── Splitting ──")
    train, val, test = split_dataset(data_list_norm)

    print("\n── DataLoaders ──")
    loaders = create_dataloaders(train, val, test,
                                  batch_size=4)

    print("\n── Test batch ──")
    batch = next(iter(loaders['train']))
    print(f"  batch.x:                 {batch.x.shape}")
    print(f"  batch.edge_index:        {batch.edge_index.shape}")
    print(f"  batch.edge_attr:         {batch.edge_attr.shape}")
    print(f"  batch.edge_forward_mask: {batch.edge_forward_mask.shape}"   # ◄◄◄ NEW
          f"  (fwd={batch.edge_forward_mask.sum().item()})")               # ◄◄◄ NEW
    print(f"  batch.y_node:            {batch.y_node.shape}")
    print(f"  batch.y_element:         {batch.y_element.shape}")
    print(f"  batch.face_mask:         {batch.face_mask.shape}")
    print(f"  batch.face_element_id:   {batch.face_element_id.shape}")
    print(f"  batch.F_ext:             {batch.F_ext.shape}")
    print(f"  batch.point_moment_My:   {batch.point_moment_My.shape}")

    print(f"\n── Physics scales in batch ──")
    if hasattr(batch, 'F_c'):
        print(f"  batch.F_c:     {batch.F_c}")
        print(f"  batch.M_c:     {batch.M_c}")
        print(f"  batch.u_c:     {batch.u_c}")
        print(f"  batch.theta_c: {batch.theta_c}")
        print(f"  ✓ Physics scales accessible")
    else:
        print(f"  ✗ Physics scales NOT in batch!")

    print("\n── Saving ──")
    os.makedirs("DATA", exist_ok=True)
    torch.save(data_list, "DATA/graph_dataset.pt")
    torch.save(data_list_norm, "DATA/graph_dataset_norm.pt")
    normalizer.save("DATA/normalizer_minmax.pt")

    d_norm = data_list_norm[0]
    print(f"\n── Verifying normalized graph ──")
    print(f"  x range: [{d_norm.x.min():.4f}, "
          f"{d_norm.x.max():.4f}]  (should ~[0, 1])")
    print(f"  F_c present: {hasattr(d_norm, 'F_c')}")
    print(f"  edge_forward_mask present: "                     # ◄◄◄ NEW
          f"{hasattr(d_norm, 'edge_forward_mask')}")           # ◄◄◄ NEW

    print(f"\n{'='*60}")
    print(f"  STEP 2 COMPLETE ✓")
    print(f"{'='*60}")
    print(f"  ┌──────────────────────────────────────────────┐")
    print(f"  │ Node features:   (N, 10)                     │")
    print(f"  │  [x,y,z, bc_d, bc_r, Fx,Fy,Fz, My, resp]    │")
    print(f"  │ Edge features:   (2E, 7)                     │")
    print(f"  │  [L, dx,dy,dz, E, A, I22]                    │")
    print(f"  │ edge_forward_mask: (2E,) bool                │")  # ◄◄◄ NEW
    print(f"  │ F_ext:           (N, 3)  [Fx, Fz, My]        │")
    print(f"  │ Node targets:    (N, 3)  [u_x, u_z, φ_y]     │")
    print(f"  │ Elem targets:    (E, 4)  [N, M, V, sens]     │")
    print(f"  └──────────────────────────────────────────────┘")
    print(f"{'='*60}")