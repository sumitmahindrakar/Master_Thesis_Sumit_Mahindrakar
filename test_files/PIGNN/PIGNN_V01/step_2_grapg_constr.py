"""
=================================================================
STEP 2: GRAPH CONSTRUCTION FOR PIGNN
=================================================================
Converts Step 1 dataset (list of dicts) → PyG Data objects.

YOUR FRAME (32 nodes, 36 elements):
  Graph: 32 nodes, 72 directed edges (36 elements × 2 directions)

NODE INPUT FEATURES (9 per node):
  [x, y, z,              ← geometry (3)
   bc_disp,              ← translation BC (1)
   bc_rot,               ← rotation BC (1)
   wl_x, wl_y, wl_z,    ← distributed load (3)
   response_node_flag]   ← response location (1)

EDGE INPUT FEATURES (11 per directed edge):
  [L,                    ← element length (1)
   dir_x, dir_y, dir_z, ← direction vector (3)
   E, A, I22, I33,      ← section/material (4)
   ν, density, J]       ← section/material (3)

NODE TARGETS (6 per node):
  [ux, uy, uz, rx, ry, rz]

ELEMENT TARGETS (7 per element):
  [Mx, My, Mz, Fx, Fy, Fz, dBM/dI22]

Prerequisites: pip install torch torch-geometric
=================================================================
"""

import os

os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\PIGNN\PIGNN_V01")
print(f"Working directory: {os.getcwd()}")

import torch
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    print(f"[OK] torch {torch.__version__}")
    print(f"[OK] torch_geometric loaded")
except ImportError:
    raise ImportError(
        "pip install torch-geometric\n"
        "See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
    )


# ================================================================
# 2A. GRAPH BUILDER
# ================================================================

class FrameGraphBuilder:
    """
    Converts structural frame case dicts → PyG Data objects.

    Each structural element becomes TWO directed edges:
      Forward:  node_i → node_j  direction = +(j-i)/L
      Backward: node_j → node_i  direction = -(j-i)/L

    This lets GNN pass messages in BOTH directions along each beam.
    """

    NODE_INPUT_DIM = 9
    EDGE_INPUT_DIM = 11
    NODE_TARGET_DIM = 6
    ELEM_TARGET_DIM = 7

    # ─── EDGE INDEX ───

    def build_edge_index(self, connectivity: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build bidirectional edge_index from element connectivity.

        Args:
            connectivity: (E, 2) element node pairs

        Returns:
            edge_index:  (2, 2E) — [source_nodes; target_nodes]
            element_map: (2E,) — maps each directed edge back to
                         its structural element index

        For your frame (36 elements → 72 directed edges):
            Edges 0-35:  forward  (as in connectivity)
            Edges 36-71: backward (reversed)
        """
        E = len(connectivity)

        # Forward: node_i → node_j
        fwd_src = connectivity[:, 0]
        fwd_dst = connectivity[:, 1]

        # Backward: node_j → node_i
        bwd_src = connectivity[:, 1]
        bwd_dst = connectivity[:, 0]

        src = np.concatenate([fwd_src, bwd_src])
        dst = np.concatenate([fwd_dst, bwd_dst])
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)

        # Edge i belongs to element (i % E)
        element_map = np.concatenate([
            np.arange(E), np.arange(E)
        ]).astype(np.int64)

        return edge_index, element_map

    # ─── NODE FEATURES ───

    def build_node_features(self, case: dict) -> np.ndarray:
        """
        Assemble node input features (N, 9).

        Columns:
          0-2:  x, y, z         (coords)
          3:    bc_disp          (1=fixed translations)
          4:    bc_rot           (1=fixed rotations)
          5-7:  wl_x, wl_y, wl_z (line_load)
          8:    response_node    (1=response measured here)
        """
        N = case['n_nodes']
        feat = np.zeros((N, self.NODE_INPUT_DIM), dtype=np.float64)

        feat[:, 0:3] = case['coords'][:, :3]
        feat[:, 3:4] = case['bc_disp']
        feat[:, 4:5] = case['bc_rot']
        feat[:, 5:8] = case['line_load'][:, :3]
        feat[:, 8:9] = case['response_node_flag']

        return feat

    # ─── EDGE FEATURES ───

    def build_edge_features(self, case: dict,
                             element_map: np.ndarray) -> np.ndarray:
        """
        Assemble edge input features (2E, 11).

        For each structural element, creates TWO edge feature rows:
          Forward:  [L, +dir_x, +dir_y, +dir_z, E, A, I22, I33, ν, ρ, J]
          Backward: [L, -dir_x, -dir_y, -dir_z, E, A, I22, I33, ν, ρ, J]

        Direction is NEGATED for backward edges — all other features same.
        """
        E = case['n_elements']

        # Element-level properties (E,)
        L = case['elem_lengths']
        directions = case['elem_directions']        # (E, 3)
        young = case['young_modulus']
        area = case['cross_area']
        I22 = case['I22']
        I33 = case['I33']
        nu = case['poisson_ratio']
        rho = case['density']
        J = case['torsional_inertia']

        # Build per-element feature matrix (E, 11)
        elem_feat = np.column_stack([
            L,                  # col 0
            directions,         # col 1-3
            young,              # col 4
            area,               # col 5
            I22,                # col 6
            I33,                # col 7
            nu,                 # col 8
            rho,                # col 9
            J,                  # col 10
        ])  # (E, 11)

        # Duplicate for bidirectional edges (2E, 11)
        n_edges = 2 * E
        edge_feat = np.zeros((n_edges, self.EDGE_INPUT_DIM),
                              dtype=np.float64)

        # Forward edges [0:E] — positive direction
        edge_feat[:E, :] = elem_feat

        # Backward edges [E:2E] — negate direction only
        edge_feat[E:, :] = elem_feat
        edge_feat[E:, 1:4] *= -1.0  # Flip direction vector

        return edge_feat

    # ─── NODE TARGETS ───

    def build_node_targets(self, case: dict) -> np.ndarray:
        """
        Node target matrix (N, 6).
        Columns: [ux, uy, uz, rx, ry, rz]
        """
        N = case['n_nodes']
        tgt = np.zeros((N, self.NODE_TARGET_DIM), dtype=np.float64)

        if case['displacement'] is not None:
            tgt[:, 0:3] = case['displacement'][:, :3]
        if case['rotation'] is not None:
            tgt[:, 3:6] = case['rotation'][:, :3]

        return tgt

    # ─── ELEMENT TARGETS ───

    def build_element_targets(self, case: dict) -> np.ndarray:
        """
        Element target matrix (E, 7).
        Columns: [Mx, My, Mz, Fx, Fy, Fz, dBM/dI22]
        """
        E = case['n_elements']
        tgt = np.zeros((E, self.ELEM_TARGET_DIM), dtype=np.float64)

        if case['moment'] is not None:
            tgt[:, 0:3] = case['moment'][:, :3]
        if case['force'] is not None:
            tgt[:, 3:6] = case['force'][:, :3]
        if case['I22_sensitivity'] is not None:
            tgt[:, 6] = case['I22_sensitivity']

        return tgt

    # ─── CASE → GRAPH ───

    def case_to_graph(self, case: dict, case_id: int = 0) -> Data:
        """
        Convert one Step 1 case dict → PyG Data object.

        Returns PyG Data with:
          .x              (32, 9)   node input features
          .edge_index     (2, 72)   directed edge connectivity
          .edge_attr      (72, 11)  edge input features
          .y_node         (32, 6)   node targets
          .y_element      (36, 7)   element targets
          + structural metadata for physics loss
        """
        conn = case['connectivity']
        N = case['n_nodes']
        E = case['n_elements']

        # 1. Topology
        edge_index, element_map = self.build_edge_index(conn)

        # 2. Features
        node_feat = self.build_node_features(case)
        edge_feat = self.build_edge_features(case, element_map)

        # 3. Targets
        node_tgt = self.build_node_targets(case)
        elem_tgt = self.build_element_targets(case)

        # 4. PyG Data
        data = Data(
            # ── Graph ──
            x=torch.tensor(node_feat, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_feat, dtype=torch.float32),

            # ── Targets ──
            y_node=torch.tensor(node_tgt, dtype=torch.float32),
            y_element=torch.tensor(elem_tgt, dtype=torch.float32),

            # ── Structural metadata ──
            element_map=torch.tensor(element_map, dtype=torch.long),
            connectivity=torch.tensor(conn, dtype=torch.long),
            coords=torch.tensor(case['coords'], dtype=torch.float32),
            bc_disp=torch.tensor(
                case['bc_disp'], dtype=torch.float32),
            bc_rot=torch.tensor(
                case['bc_rot'], dtype=torch.float32),
            elem_lengths=torch.tensor(
                case['elem_lengths'], dtype=torch.float32),
            elem_directions=torch.tensor(
                case['elem_directions'], dtype=torch.float32),
            line_load=torch.tensor(
                case['line_load'], dtype=torch.float32),
            prop_E=torch.tensor(
                case['young_modulus'], dtype=torch.float32),
            prop_A=torch.tensor(
                case['cross_area'], dtype=torch.float32),
            prop_I22=torch.tensor(
                case['I22'], dtype=torch.float32),
            response_node_flag=torch.tensor(
                case['response_node_flag'], dtype=torch.float32),

            # ── Scalars ──
            num_nodes_val=N,
            n_elements=E,
            case_id=case_id,
            nearest_node_id=case['nearest_node_id'],
            traced_element_id=case['traced_element_id'],
        )

        return data

    # ─── BUILD FULL DATASET ───

    def build_dataset(self, cases: List[dict]) -> List[Data]:
        """
        Convert all cases to PyG Data objects.
        """
        data_list = []
        for i, case in enumerate(cases):
            data = self.case_to_graph(case, case_id=i)
            data_list.append(data)

        print(f"\n  Built {len(data_list)} graphs")
        print(f"    Nodes:          {data_list[0].num_nodes}")
        print(f"    Directed edges: {data_list[0].edge_index.shape[1]}")
        print(f"    Node features:  {data_list[0].x.shape[1]}")
        print(f"    Edge features:  {data_list[0].edge_attr.shape[1]}")
        print(f"    Node targets:   {data_list[0].y_node.shape[1]}")
        print(f"    Element targets:{data_list[0].y_element.shape[1]}")

        return data_list


# ================================================================
# 2B. NORMALIZER
# ================================================================

class DataNormalizer:
    """
    Z-score normalization for graph features and targets.

    x_norm = (x - mean) / std

    Skips binary flag columns:
      Node col 3 (bc_disp), col 4 (bc_rot), col 8 (response_node)
    """

    def __init__(self):
        self.x_mean = None
        self.x_std = None
        self.ea_mean = None
        self.ea_std = None
        self.yn_mean = None
        self.yn_std = None
        self.ye_mean = None
        self.ye_std = None

        # Binary flag columns — DO NOT normalize
        self.node_skip_cols = [3, 4, 8]    # bc_disp, bc_rot, response
        self.edge_skip_cols = []            # none for edges

        self.is_fitted = False

    def fit(self, data_list: List[Data]):
        """
        Compute mean/std from training data.
        Call on TRAINING set only.
        """
        all_x = torch.cat([d.x for d in data_list], dim=0)
        all_ea = torch.cat([d.edge_attr for d in data_list], dim=0)
        all_yn = torch.cat([d.y_node for d in data_list], dim=0)
        all_ye = torch.cat([d.y_element for d in data_list], dim=0)

        self.x_mean = all_x.mean(dim=0)
        self.x_std = all_x.std(dim=0).clamp(min=1e-8)
        self.ea_mean = all_ea.mean(dim=0)
        self.ea_std = all_ea.std(dim=0).clamp(min=1e-8)
        self.yn_mean = all_yn.mean(dim=0)
        self.yn_std = all_yn.std(dim=0).clamp(min=1e-8)
        self.ye_mean = all_ye.mean(dim=0)
        self.ye_std = all_ye.std(dim=0).clamp(min=1e-8)

        # Don't normalize binary flags
        for c in self.node_skip_cols:
            self.x_mean[c] = 0.0
            self.x_std[c] = 1.0
        for c in self.edge_skip_cols:
            self.ea_mean[c] = 0.0
            self.ea_std[c] = 1.0

        self.is_fitted = True
        self._print_stats()

    def _print_stats(self):
        print(f"\n  Normalization fitted:")
        print(f"  {'Feature':<20} {'Mean range':<30} {'Std range':<30}")
        print(f"  {'-'*80}")
        for name, mean, std in [
            ('Node features', self.x_mean, self.x_std),
            ('Edge features', self.ea_mean, self.ea_std),
            ('Node targets', self.yn_mean, self.yn_std),
            ('Elem targets', self.ye_mean, self.ye_std),
        ]:
            print(f"  {name:<20} [{mean.min():.4e}, {mean.max():.4e}]  "
                  f"[{std.min():.4e}, {std.max():.4e}]")

    def transform(self, data: Data) -> Data:
        """Normalize one graph."""
        assert self.is_fitted
        data = data.clone()
        data.x = (data.x - self.x_mean) / self.x_std
        data.edge_attr = (data.edge_attr - self.ea_mean) / self.ea_std
        data.y_node = (data.y_node - self.yn_mean) / self.yn_std
        data.y_element = (data.y_element - self.ye_mean) / self.ye_std
        return data

    def transform_list(self, data_list: List[Data]) -> List[Data]:
        return [self.transform(d) for d in data_list]

    def inverse_node_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize node predictions back to physical units."""
        return y * self.yn_std.to(y.device) + self.yn_mean.to(y.device)

    def inverse_elem_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize element predictions back to physical units."""
        return y * self.ye_std.to(y.device) + self.ye_mean.to(y.device)

    def save(self, filepath: str):
        torch.save({
            'x_mean': self.x_mean, 'x_std': self.x_std,
            'ea_mean': self.ea_mean, 'ea_std': self.ea_std,
            'yn_mean': self.yn_mean, 'yn_std': self.yn_std,
            'ye_mean': self.ye_mean, 'ye_std': self.ye_std,
            'node_skip_cols': self.node_skip_cols,
            'edge_skip_cols': self.edge_skip_cols,
            'is_fitted': self.is_fitted,
        }, filepath)
        print(f"  Normalizer saved: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'DataNormalizer':
        state = torch.load(filepath, weights_only=False)
        norm = cls()
        for k, v in state.items():
            setattr(norm, k, v)
        return norm


# ================================================================
# 2C. DATASET SPLIT & DATALOADERS
# ================================================================

def split_dataset(data_list: List[Data],
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   seed: int = 42) -> Tuple[list, list, list]:
    """Split into train/val/test."""
    n = len(data_list)
    np.random.seed(seed)
    idx = np.random.permutation(n)

    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    train = [data_list[i] for i in idx[:n_train]]
    val = [data_list[i] for i in idx[n_train:n_train + n_val]]
    test = [data_list[i] for i in idx[n_train + n_val:]]

    # Ensure test is not empty
    if not test:
        test = val.copy()

    print(f"  Split: {len(train)} train, {len(val)} val, {len(test)} test")
    return train, val, test


def create_dataloaders(train: list, val: list, test: list,
                        batch_size: int = 16) -> dict:
    """Create PyG DataLoaders."""
    loaders = {
        'train': DataLoader(train, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test, batch_size=batch_size, shuffle=False),
    }
    for name, loader in loaders.items():
        print(f"    {name}: {len(loader.dataset)} graphs, "
              f"{len(loader)} batches")
    return loaders


# ================================================================
# 2D. GRAPH VERIFICATION
# ================================================================

def verify_graph(data: Data) -> bool:
    """Run verification checks on a graph."""
    print(f"\n{'═'*60}")
    print(f"  GRAPH VERIFICATION")
    print(f"{'═'*60}")

    N = data.num_nodes
    E = data.n_elements
    n_edges = data.edge_index.shape[1]
    all_ok = True

    # 1. Shape checks
    print(f"\n  1. SHAPES:")
    checks = [
        ('x',         data.x.shape,         (N, 9)),
        ('edge_index', data.edge_index.shape, (2, 2*E)),
        ('edge_attr', data.edge_attr.shape,  (2*E, 11)),
        ('y_node',    data.y_node.shape,     (N, 6)),
        ('y_element', data.y_element.shape,  (E, 7)),
        ('element_map', data.element_map.shape, (2*E,)),
        ('connectivity', data.connectivity.shape, (E, 2)),
    ]
    for name, actual, expected in checks:
        ok = actual == expected
        if not ok:
            all_ok = False
        print(f"    {'✓' if ok else '✗'} {name:<15} "
              f"{str(actual):<15} (expected {expected})")

    # 2. Edge index validity
    print(f"\n  2. EDGE INDEX:")
    ei = data.edge_index
    print(f"    Range: [{ei.min()}, {ei.max()}] (should be [0, {N-1}])")
    print(f"    No self-loops: {bool((ei[0] != ei[1]).all())}")

    # Bidirectionality check
    fwd = set(zip(ei[0, :E].tolist(), ei[1, :E].tolist()))
    bwd = set(zip(ei[0, E:].tolist(), ei[1, E:].tolist()))
    fwd_rev = set((b, a) for a, b in fwd)
    print(f"    Bidirectional: {fwd_rev == bwd}")

    # 3. Feature values
    print(f"\n  3. NODE FEATURES RANGE [MIN , MAX]:")
    x = data.x
    labels = ['x', 'y', 'z', 'bc_d', 'bc_r',
              'wl_x', 'wl_y', 'wl_z', 'resp']
    for i, label in enumerate(labels):
        col = x[:, i]
        print(f"    {label:>5}: [{col.min():.4e}, {col.max():.4e}]")

    print(f"\n  4. EDGE FEATURES (forward edges only) [MIN , MAX]:")
    ea = data.edge_attr[:E]
    e_labels = ['L', 'dx', 'dy', 'dz', 'E', 'A',
                'I22', 'I33', 'nu', 'rho', 'J']
    for i, label in enumerate(e_labels):
        col = ea[:, i]
        print(f"    {label:>4}: [{col.min():.4e}, {col.max():.4e}]")

    # 5. Target values
    print(f"\n  5. NODE TARGETS [MIN , MAX]:")
    yn = data.y_node
    t_labels = ['ux', 'uy', 'uz', 'rx', 'ry', 'rz']
    for i, label in enumerate(t_labels):
        col = yn[:, i]
        print(f"    {label:>3}: [{col.min():.4e}, {col.max():.4e}]")

    print(f"\n  6. ELEMENT TARGETS [MIN , MAX]:")
    ye = data.y_element
    et_labels = ['Mx', 'My', 'Mz', 'Fx', 'Fy', 'Fz', 'dBM/dI']
    for i, label in enumerate(et_labels):
        col = ye[:, i]
        print(f"    {label:>7}: [{col.min():.4e}, {col.max():.4e}]")

    # 6. Response node
    resp = data.x[:, 8]
    resp_nodes = torch.where(resp > 0.5)[0].tolist()
    print(f"\n  7. RESPONSE NODE: {resp_nodes} "
          f"(config: {data.nearest_node_id})")

    # 7. Support nodes
    bc = data.x[:, 3]
    support_nodes = torch.where(bc > 0.5)[0].tolist()
    print(f"  8. SUPPORT NODES: {support_nodes}")

    status = "ALL PASSED ✓" if all_ok else "SOME FAILED ✗"
    print(f"\n  RESULT: {status}")
    print(f"{'═'*60}\n")
    return all_ok


# ================================================================
# 2E. VISUALIZATION
# ================================================================

def visualize_frame_graph(data: Data, title: str = "Frame Graph"):
    """Visualize the frame graph in XZ plane."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  pip install matplotlib for visualization")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    coords = data.coords.numpy()
    conn = data.connectivity.numpy()
    x_feat = data.x.numpy()
    E = data.n_elements

    # Draw elements
    for e in range(E):
        n1, n2 = conn[e]
        xs = [coords[n1, 0], coords[n2, 0]]
        zs = [coords[n1, 2], coords[n2, 2]]

        # Color by type
        dx = abs(coords[n2, 0] - coords[n1, 0])
        dz = abs(coords[n2, 2] - coords[n1, 2])
        if dz > dx:
            color, lw = 'steelblue', 2.5
        else:
            color, lw = 'coral', 2.5

        ax.plot(xs, zs, '-', color=color, linewidth=lw,
                solid_capstyle='round')

    # Draw nodes
    for i in range(data.num_nodes):
        xp, zp = coords[i, 0], coords[i, 2]

        is_support = x_feat[i, 3] > 0.5
        is_response = x_feat[i, 8] > 0.5

        if is_response:
            ax.plot(xp, zp, '*', markersize=18, color='gold',
                    markeredgecolor='black', zorder=6)
        elif is_support:
            ax.plot(xp, zp, '^', markersize=12, color='green',
                    markeredgecolor='black', zorder=5)
        else:
            ax.plot(xp, zp, 'o', markersize=5, color='white',
                    markeredgecolor='black', zorder=5)

        ax.text(xp + 0.2, zp + 0.2, f'{i}', fontsize=6,
                color='gray')

        # Load arrows
        wl_z = x_feat[i, 7]  # line_load Z component
        if abs(wl_z) > 1e-10:
            scale = 0.02
            ax.annotate('', xy=(xp, zp),
                       xytext=(xp, zp - wl_z * scale),
                       arrowprops=dict(arrowstyle='->', color='red',
                                     lw=1.2))

    # Legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color='steelblue', lw=3, label='Column'),
        Line2D([0], [0], color='coral', lw=3, label='Beam'),
        Line2D([0], [0], marker='^', color='green', lw=0,
               markersize=10, label='Support'),
        Line2D([0], [0], marker='*', color='gold', lw=0,
               markersize=15, label='Response node'),
        Line2D([0], [0], marker='>', color='red', lw=0,
               markersize=8, label='Load'),
    ]
    ax.legend(handles=legend, loc='upper right')

    ax.set_xlabel('X')
    ax.set_ylabel('Z (height)')
    ax.set_title(f"{title}\n{data.num_nodes} nodes, "
                 f"{E} elements, {2*E} directed edges")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("DATA", exist_ok=True)
    plt.savefig('DATA/frame_graph.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: DATA/frame_graph.png")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  STEP 2: Graph Construction")
    print("=" * 60)

    # ─── Load Step 1 dataset ───
    print("\n── Loading dataset from Step 1 ──")
    with open("DATA/frame_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    print(f"  Loaded {len(dataset)} cases")

    # ─── Build graphs ───
    print("\n── Building PyG graphs ──")
    builder = FrameGraphBuilder()
    data_list = builder.build_dataset(dataset)

    # ─── Verify first graph ───
    print("\n── Verifying first graph ──")
    verify_graph(data_list[0])

    # ─── Visualize ───
    print("\n── Visualizing ──")
    visualize_frame_graph(data_list[0],
                           title=f"Case {dataset[0]['case_num']}")

    # ─── Normalize ───
    print("\n── Normalizing ──")
    normalizer = DataNormalizer()
    normalizer.fit(data_list)
    data_list_norm = normalizer.transform_list(data_list)

    # ─── Split ───
    print("\n── Splitting ──")
    train, val, test = split_dataset(data_list_norm)

    # ─── DataLoaders ───
    print("\n── DataLoaders ──")
    loaders = create_dataloaders(train, val, test, batch_size=4)

    # ─── Test batch ───
    print("\n── Test batch ──")
    batch = next(iter(loaders['train']))
    print(f"  batch.x:          {batch.x.shape}")
    print(f"  batch.edge_index: {batch.edge_index.shape}")
    print(f"  batch.edge_attr:  {batch.edge_attr.shape}")
    print(f"  batch.y_node:     {batch.y_node.shape}")
    print(f"  batch.y_element:  {batch.y_element.shape}")
    print(f"  batch.batch:      {batch.batch.shape}")

    # ─── Save ───
    print("\n── Saving ──")
    os.makedirs("DATA", exist_ok=True)
    torch.save(data_list, "DATA/graph_dataset.pt")
    torch.save(data_list_norm, "DATA/graph_dataset_norm.pt")
    normalizer.save("DATA/normalizer.pt")
    print(f"  Saved graph_dataset.pt ({len(data_list)} graphs)")
    print(f"  Saved graph_dataset_norm.pt (normalized)")
    print(f"  Saved normalizer.pt")

    print(f"\n{'='*60}")
    print(f"  STEP 2 COMPLETE ✓")
    print(f"")
    print(f"  Each graph:")
    d = data_list[0]
    print(f"    x:         ({d.num_nodes}, 9)   node inputs")
    print(f"    edge_attr: ({d.edge_index.shape[1]}, 11) edge inputs")
    print(f"    y_node:    ({d.num_nodes}, 6)   targets")
    print(f"    y_element: ({d.n_elements}, 7)  targets")
    print(f"")
    print(f"  Ready for Step 3 (PIGNN Model Architecture)")
    print(f"{'='*60}")