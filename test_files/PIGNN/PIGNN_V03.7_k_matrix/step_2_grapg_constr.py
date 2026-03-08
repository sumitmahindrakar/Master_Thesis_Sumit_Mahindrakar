"""
=================================================================
STEP 2: GRAPH CONSTRUCTION — Strong-Form PIGNN
=================================================================

Network output:  y_i = (u_x, u_z, φ)        → node targets (N, 3)
Physics derives: N = EA·du/dx                 ┐
                 M = EI·d²u/dx²               ├→ element targets (E, 4)
                 V = EI·d³u/dx³               ┘

Edge features reduced: 11 → 10
  Removed: I33, ν, ρ, J (not in 2D Euler-Bernoulli strong form)
  Added:   elem_load (qx, qy, qz) for equilibrium residuals

Pre-computed: prop_EA, prop_EI for physics loss
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
# 2A. GRAPH BUILDER
# ================================================================

class FrameGraphBuilder:
    """
    Converts frame case dicts → PyG Data objects for strong-form PIGNN.

    Network predicts:  y_i = (u_x, u_z, φ) at each node
    Physics derives:   N = EA·du/dx, M = EI·d²u/dx², V = EI·d³u/dx³

    Each structural element → TWO directed edges (bidirectional).
    """

    NODE_INPUT_DIM  = 9   # [x,y,z, bc_d,bc_r, wl_x,wl_y,wl_z, resp]
    EDGE_INPUT_DIM  = 10   # [L, dx,dy,dz, E, A, I22, lineload x, lineload y,lineload z]
    NODE_TARGET_DIM = 3   # [u_x, u_z, φ]                 ← was 6
    ELEM_TARGET_DIM = 4   # [N, M, V, dBM/dI22]           ← was 7

    # ─── EDGE INDEX ───

    def build_edge_index(self, connectivity: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bidirectional edges from element connectivity.

        Returns:
            edge_index:  (2, 2E)
            element_map: (2E,) — maps each directed edge to its element
        """
        E = len(connectivity)

        src = np.concatenate([connectivity[:, 0], connectivity[:, 1]])
        dst = np.concatenate([connectivity[:, 1], connectivity[:, 0]])
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)

        # np.tile is cleaner than concatenate([arange, arange])
        element_map = np.tile(np.arange(E), 2).astype(np.int64)

        return edge_index, element_map

    # ─── NODE FEATURES ───

    def build_node_features(self, case: dict) -> np.ndarray:
        """
        (N, 9): [x, y, z, bc_disp, bc_rot, wl_x, wl_y, wl_z, response]

        Unchanged — all 9 features are needed:
          coords    → GNN learns spatial relationships
          bc flags  → boundary condition awareness
          loads     → input forcing
          response  → which node is the sensitivity location
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

    def build_edge_features(self, case, element_map):
        """
        (2E, 10): [L, dx,dy,dz, E, A, I22, qx,qy,qz]
        """
        E = case['n_elements']

        elem_feat = np.column_stack([
            case['elem_lengths'],       # 0:   L
            case['elem_directions'],    # 1-3: direction
            case['young_modulus'],      # 4:   E
            case['cross_area'],         # 5:   A
            case['I22'],                # 6:   I22
            case['elem_load'],          # 7-9: UDL [qx, qy, qz]  ← NEW
        ])  # (E, 10)

        edge_feat = np.zeros((2 * E, self.EDGE_INPUT_DIM),
                            dtype=np.float64)
        edge_feat[:E]  = elem_feat
        edge_feat[E:]  = elem_feat
        edge_feat[E:, 1:4] *= -1.0     # negate direction only
        # Load direction stays same for both edge directions

        return edge_feat

    # ─── NODE TARGETS ───

    def build_node_targets(self, case: dict) -> np.ndarray:
        """
        (N, 3): [u_x, u_z, φ]

        This IS your network output vector y_i.

        For 2D frame in XZ plane:
          u_x = displacement[:, 0]  (horizontal)
          u_z = displacement[:, 2]  (vertical)
          φ   = rotation[:, 1]      (about Y-axis)
        """
        N = case['n_nodes']
        tgt = np.zeros((N, self.NODE_TARGET_DIM), dtype=np.float64)

        # Use pre-extracted 2D DOFs if available (from Step 1 update)
        if 'nodal_disp_2d' in case and case['nodal_disp_2d'] is not None:
            return case['nodal_disp_2d'].astype(np.float64)

        # Otherwise extract from full 3D fields
        if case.get('displacement') is not None:
            tgt[:, 0] = case['displacement'][:, 0]    # u_x
            tgt[:, 1] = case['displacement'][:, 2]    # u_z
        if case.get('rotation') is not None:
            tgt[:, 2] = case['rotation'][:, 1]         # φ_y

        return tgt

    # ─── ELEMENT TARGETS ───

    def build_element_targets(self, case: dict) -> np.ndarray:
        """
        (E, 4): [N, M, V, dBM/dI22]

        These are the strong-form ground truths:
          col 0: N = axial force        (compare with EA·du_axial/dx)
          col 1: M = bending moment     (compare with EI·d²u_trans/dx²)
          col 2: V = shear force        (compare with EI·d³u_trans/dx³)
          col 3: dBM/dI22 = sensitivity (from adjoint, optional)

        Note: N, M, V from Kratos are in ELEMENT LOCAL coordinates.
        Your physics derivatives must also be in local coords.
        """
        E = case['n_elements']
        tgt = np.zeros((E, self.ELEM_TARGET_DIM), dtype=np.float64)

        # Axial force N — FORCE[:, 0]
        if 'elem_N' in case and case['elem_N'] is not None:
            tgt[:, 0] = case['elem_N']
        elif case.get('force') is not None:
            tgt[:, 0] = case['force'][:, 0]

        # Bending moment M — MOMENT[:, 1] (about Y)
        if 'elem_M' in case and case['elem_M'] is not None:
            tgt[:, 1] = case['elem_M']
        elif case.get('moment') is not None:
            tgt[:, 1] = case['moment'][:, 1]

        # Shear force V — FORCE[:, 2]
        if 'elem_V' in case and case['elem_V'] is not None:
            tgt[:, 2] = case['elem_V']
        elif case.get('force') is not None:
            tgt[:, 2] = case['force'][:, 2]

        # Sensitivity (optional)
        if case.get('I22_sensitivity') is not None:
            tgt[:, 3] = case['I22_sensitivity']

        return tgt

    # ─── CASE → GRAPH ───

    def case_to_graph(self, case: dict, case_id: int = 0) -> Data:
        """
        Convert one case dict → PyG Data object.

        Returns Data with:
          .x              (N, 9)    node inputs
          .edge_index     (2, 2E)   bidirectional edges
          .edge_attr      (2E, 7)   edge inputs
          .y_node         (N, 3)    [u_x, u_z, φ]
          .y_element      (E, 4)    [N, M, V, sens]
          + physics metadata for strong-form loss
        """
        conn = case['connectivity']
        N = case['n_nodes']
        E = case['n_elements']

        edge_index, element_map = self.build_edge_index(conn)
        node_feat = self.build_node_features(case)
        edge_feat = self.build_edge_features(case, element_map)
        node_tgt  = self.build_node_targets(case)
        elem_tgt  = self.build_element_targets(case)

        data = Data(
            # ── Graph structure & features ──
            x          = torch.tensor(node_feat, dtype=torch.float32),
            edge_index = torch.tensor(edge_index, dtype=torch.long),
            edge_attr  = torch.tensor(edge_feat,  dtype=torch.float32),

            # ── Element load (raw, for physics loss) ──
            elem_load = torch.tensor(
                case['elem_load'], dtype=torch.float32
            ),  # (E, 3)

            # ── Targets ──
            y_node    = torch.tensor(node_tgt,  dtype=torch.float32),
            y_element = torch.tensor(elem_tgt,  dtype=torch.float32),

            # ── Physics metadata (separate for autodiff) ──
            element_map     = torch.tensor(element_map, dtype=torch.long),
            connectivity    = torch.tensor(conn, dtype=torch.long),
            coords          = torch.tensor(
                case['coords'], dtype=torch.float32),
            elem_lengths    = torch.tensor(
                case['elem_lengths'], dtype=torch.float32),
            elem_directions = torch.tensor(
                case['elem_directions'], dtype=torch.float32),

            # ── Keep E, A, I22 SEPARATE ──
            # So later: EI = prop_E * prop_I22  (I22 stays in graph)
            #           dM/dI22 via autodiff works!
            prop_E    = torch.tensor(
                case['young_modulus'], dtype=torch.float32),
            prop_A    = torch.tensor(
                case['cross_area'], dtype=torch.float32),
            prop_I22  = torch.tensor(
                case['I22'], dtype=torch.float32),

            bc_disp   = torch.tensor(
                case['bc_disp'], dtype=torch.float32),
            bc_rot    = torch.tensor(
                case['bc_rot'], dtype=torch.float32),

            # ── Scalars ──
            num_nodes_val     = N,
            n_elements        = E,
            case_id           = case_id,
            nearest_node_id   = case['nearest_node_id'],
            traced_element_id = case['traced_element_id'],
        )

        return data

    # ─── BUILD FULL DATASET ───

    def build_dataset(self, cases: List[dict]) -> List[Data]:
        """Convert all cases to PyG Data objects."""
        data_list = [
            self.case_to_graph(case, case_id=i)
            for i, case in enumerate(cases)
        ]

        d = data_list[0]
        print(f"\n  Built {len(data_list)} graphs")
        print(f"    Nodes per graph:  {d.num_nodes}")
        print(f"    Directed edges:   {d.edge_index.shape[1]}")
        print(f"    Node features:    {d.x.shape[1]}   "
              f"[x,y,z, bc_d,bc_r, wl_x,wl_y,wl_z, resp]")
        print(f"    Edge features:    {d.edge_attr.shape[1]}   "
              f"[L, dx,dy,dz, E, A, I22, qx,qy,qz]")
        print(f"    Node targets:     {d.y_node.shape[1]}   "
              f"[u_x, u_z, φ]")
        print(f"    Element targets:  {d.y_element.shape[1]}   "
              f"[N, M, V, dBM/dI22]")

        return data_list


# ================================================================
# 2B. NORMALIZER (simplified — inputs only)
# ================================================================

class DataNormalizer:
    """
    Z-score normalization for INPUT features only.
    Targets stay in raw physical units (for physics loss).
    Physics metadata (coords, EA, EI, etc.) also NOT normalized.
    """

    def __init__(self):
        self.x_mean  = None
        self.x_std   = None
        self.ea_mean = None
        self.ea_std  = None
        # Binary flags — skip normalization
        self.node_skip_cols = [3, 4, 8]  # bc_disp, bc_rot, response
        self.is_fitted = False

    def fit(self, data_list: List[Data]):
        """Compute stats from training data only."""
        all_x  = torch.cat([d.x for d in data_list], dim=0)
        all_ea = torch.cat([d.edge_attr for d in data_list], dim=0)

        # torch.std_mean → single pass, more efficient
        self.x_std,  self.x_mean  = torch.std_mean(all_x,  dim=0)
        self.ea_std, self.ea_mean = torch.std_mean(all_ea, dim=0)

        # Prevent division by zero
        self.x_std  = self.x_std.clamp(min=1e-8)
        self.ea_std = self.ea_std.clamp(min=1e-8)

        # Don't normalize binary flags
        for c in self.node_skip_cols:
            self.x_mean[c] = 0.0
            self.x_std[c]  = 1.0

        self.is_fitted = True
        self._print_stats()

    def _print_stats(self):
        print(f"\n  Normalization fitted (INPUTS only):")
        print(f"  {'Feature':<20} {'Mean range':<30} {'Std range'}")
        print(f"  {'-'*75}")
        for name, mean, std in [
            ('Node features (9)',  self.x_mean,  self.x_std),
            ('Edge features (7)',  self.ea_mean, self.ea_std),
        ]:
            print(f"  {name:<20} [{mean.min():.4e}, {mean.max():.4e}]  "
                  f"[{std.min():.4e}, {std.max():.4e}]")
        print(f"  Targets:          NOT normalized (raw physical units)")
        print(f"  Physics metadata: NOT normalized (coords, EA, EI, etc.)")

    def transform(self, data: Data) -> Data:
        """Normalize one graph's input features."""
        assert self.is_fitted, "Call .fit() first"
        data = data.clone()
        data.x         = (data.x - self.x_mean) / self.x_std
        data.edge_attr = (data.edge_attr - self.ea_mean) / self.ea_std
        return data

    def transform_list(self, data_list: List[Data]) -> List[Data]:
        return [self.transform(d) for d in data_list]

    def save(self, filepath: str):
        torch.save({
            'x_mean': self.x_mean,   'x_std': self.x_std,
            'ea_mean': self.ea_mean, 'ea_std': self.ea_std,
            'node_skip_cols': self.node_skip_cols,
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
# 2C. SPLIT & DATALOADERS
# ================================================================

def split_dataset(data_list: List[Data],
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   seed: int = 42) -> Tuple[list, list, list]:
    """
    Split into train/val/test.
    Uses torch.Generator for PyTorch-native reproducibility.
    """
    n = len(data_list)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=gen).tolist()

    n_train = max(1, int(n * train_ratio))
    n_val   = max(1, int(n * val_ratio))

    train = [data_list[i] for i in idx[:n_train]]
    val   = [data_list[i] for i in idx[n_train:n_train + n_val]]
    test  = [data_list[i] for i in idx[n_train + n_val:]]
    if not test:
        test = val.copy()

    print(f"  Split: {len(train)} train, "
          f"{len(val)} val, {len(test)} test")
    return train, val, test


def create_dataloaders(train, val, test,
                        batch_size: int = 16) -> dict:
    """Create PyG DataLoaders."""
    loaders = {
        'train': DataLoader(train, batch_size=batch_size, shuffle=True),
        'val':   DataLoader(val,   batch_size=batch_size, shuffle=False),
        'test':  DataLoader(test,  batch_size=batch_size, shuffle=False),
    }
    for name, loader in loaders.items():
        print(f"    {name}: {len(loader.dataset)} graphs, "
              f"{len(loader)} batches")
    return loaders


# ================================================================
# 2D. VERIFICATION (updated for new dimensions)
# ================================================================

def verify_graph(data: Data) -> bool:
    """Verify a graph matches strong-form PIGNN spec."""
    print(f"\n{'═'*60}")
    print(f"  GRAPH VERIFICATION (Strong-Form PIGNN)")
    print(f"{'═'*60}")

    N = data.num_nodes
    E = data.n_elements
    all_ok = True

    # 1. Shape checks
    print(f"\n  1. SHAPES:")
    checks = [
        ('x',            data.x.shape,            (N, 9)),
        ('edge_index',   data.edge_index.shape,   (2, 2*E)),
        ('edge_attr',    data.edge_attr.shape,     (2*E, 10)),
        ('y_node',       data.y_node.shape,        (N, 3)),
        ('y_element',    data.y_element.shape,     (E, 4)),
        ('element_map',  data.element_map.shape,   (2*E,)),
        ('connectivity', data.connectivity.shape,  (E, 2)),
        ('prop_E',       data.prop_E.shape,       (E,)),
        ('prop_A',       data.prop_A.shape,       (E,)),
        ('prop_I22',     data.prop_I22.shape,     (E,)),
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
    print(f"    Range: [{ei.min()}, {ei.max()}] "
          f"(should be [0, {N-1}])")
    print(f"    No self-loops: "
          f"{bool((ei[0] != ei[1]).all())}")

    fwd = set(zip(ei[0, :E].tolist(), ei[1, :E].tolist()))
    bwd = set(zip(ei[0, E:].tolist(), ei[1, E:].tolist()))
    fwd_rev = set((b, a) for a, b in fwd)
    print(f"    Bidirectional: {fwd_rev == bwd}")

    # 3. Node features
    print(f"\n  3. NODE FEATURES (9):")
    labels = ['x', 'y', 'z', 'bc_d', 'bc_r',
              'wl_x', 'wl_y', 'wl_z', 'resp']
    for i, label in enumerate(labels):
        col = data.x[:, i]
        print(f"    {label:>5}: [{col.min():.4e}, {col.max():.4e}]")

    # 4. Edge features (forward only)
    print(f"\n  4. EDGE FEATURES (10, forward only):")
    ea = data.edge_attr[:E]
    e_labels = ['L', 'dx', 'dy', 'dz', 'E', 'A', 'I22', 'qx', 'qy', 'qz']
    for i, label in enumerate(e_labels):
        col = ea[:, i]
        print(f"    {label:>4}: [{col.min():.4e}, {col.max():.4e}]")

    # 5. Node targets
    print(f"\n  5. NODE TARGETS [u_x, u_z, φ]:")
    t_labels = ['u_x', 'u_z', 'φ']
    for i, label in enumerate(t_labels):
        col = data.y_node[:, i]
        print(f"    {label:>4}: [{col.min():.4e}, {col.max():.4e}]")

    # 6. Element targets
    print(f"\n  6. ELEMENT TARGETS [N, M, V, sens]:")
    et_labels = ['N', 'M', 'V', 'dBM/dI']
    for i, label in enumerate(et_labels):
        col = data.y_element[:, i]
        print(f"    {label:>7}: [{col.min():.4e}, {col.max():.4e}]")

    # 7. Physics constants
    print(f"\n  7. PHYSICS CONSTANTS:")
    print(f"    E:   [{data.prop_E.min():.4e}, "
          f"{data.prop_E.max():.4e}]")
    print(f"    A:   [{data.prop_A.min():.4e}, "
          f"{data.prop_A.max():.4e}]")
    print(f"    I22: [{data.prop_I22.min():.4e}, "
          f"{data.prop_I22.max():.4e}]")
    EA = data.prop_E * data.prop_A
    EI = data.prop_E * data.prop_I22
    print(f"    EA:  [{EA.min():.4e}, {EA.max():.4e}]  (computed)")
    print(f"    EI:  [{EI.min():.4e}, {EI.max():.4e}]  (computed)")

    # 8. Special nodes
    resp = data.x[:, 8]
    resp_nodes = torch.where(resp > 0.5)[0].tolist()
    bc = data.x[:, 3]
    support_nodes = torch.where(bc > 0.5)[0].tolist()
    print(f"\n  8. RESPONSE NODE:  {resp_nodes} "
          f"(config: {data.nearest_node_id})")
    print(f"     SUPPORT NODES:  {support_nodes}")

    status = "ALL PASSED ✓" if all_ok else "SOME FAILED ✗"
    print(f"\n  RESULT: {status}")
    print(f"{'═'*60}\n")
    return all_ok


# ================================================================
# 2E. VISUALIZATION
# ================================================================

def visualize_frame_graph(data: Data,
                           title: str = "Frame Graph"):
    """Visualize frame graph in XZ plane."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
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
        is_support  = x_feat[i, 3] > 0.5
        is_response = x_feat[i, 8] > 0.5

        if is_response:
            ax.plot(xp, zp, '*', ms=18, color='gold',
                    mec='black', zorder=6)
        elif is_support:
            ax.plot(xp, zp, '^', ms=12, color='green',
                    mec='black', zorder=5)
        else:
            ax.plot(xp, zp, 'o', ms=5, color='white',
                    mec='black', zorder=5)

        ax.text(xp + 0.2, zp + 0.2, f'{i}', fontsize=6,
                color='gray')

        wl_z = x_feat[i, 7]
        if abs(wl_z) > 1e-10:
            ax.annotate('', xy=(xp, zp),
                       xytext=(xp, zp - wl_z * 0.02),
                       arrowprops=dict(arrowstyle='->',
                                     color='red', lw=1.2))

    legend = [
        Line2D([0], [0], color='steelblue', lw=3, label='Column'),
        Line2D([0], [0], color='coral', lw=3, label='Beam'),
        Line2D([0], [0], marker='^', color='green', lw=0,
               ms=10, label='Support'),
        Line2D([0], [0], marker='*', color='gold', lw=0,
               ms=15, label='Response node'),
        Line2D([0], [0], marker='>', color='red', lw=0,
               ms=8, label='Load'),
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
    plt.savefig('DATA/frame_graph.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    print("  Saved: DATA/frame_graph.png")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    
    import os
    from pathlib import Path
    print(f"Working directory: {os.getcwd()}")
    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)
    print(f"Working directory: {os.getcwd()}")

    print("=" * 60)
    print("  STEP 2: Graph Construction (Strong-Form PIGNN)")
    print("=" * 60)

    # ─── Load Step 1 data ───
    print("\n── Loading dataset from Step 1 ──")
    with open("DATA/frame_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    print(f"  Loaded {len(dataset)} cases")

    # ─── Build graphs ───
    print("\n── Building PyG graphs ──")
    builder = FrameGraphBuilder()
    data_list = builder.build_dataset(dataset)

    # ─── Verify ───
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
    print(f"  batch.y_node:     {batch.y_node.shape}   "
          f"← [u_x, u_z, φ]")
    print(f"  batch.y_element:  {batch.y_element.shape}   "
          f"← [N, M, V, sens]")
    print(f"  batch.prop_E:     {batch.prop_E.shape}")
    print(f"  batch.prop_A:     {batch.prop_A.shape}")
    print(f"  batch.prop_I22:   {batch.prop_I22.shape}")
    print(f"  batch.batch:      {batch.batch.shape}")

    # ─── Save ───
    print("\n── Saving ──")
    os.makedirs("DATA", exist_ok=True)
    torch.save(data_list, "DATA/graph_dataset.pt")
    torch.save(data_list_norm, "DATA/graph_dataset_norm.pt")
    normalizer.save("DATA/normalizer.pt")

    # ─── Final summary ───
    print(f"\n{'='*60}")
    print(f"  STEP 2 COMPLETE ✓ (Strong-Form PIGNN)")
    print(f"{'='*60}")
    d = data_list[0]
    print(f"  ┌──────────────────────────────────────────┐")
    print(f"  │ Node inputs:   (N, 9)                    │")
    print(f"  │ Edge inputs:   (2E, 10)                  │")
    print(f"  │   [L, dx,dy,dz, E, A, I22, qx,qy,qz]   │")
    print(f"  │ Node targets:  (N, 3)    [u_x, u_z, φ]  │")
    print(f"  │ Elem targets:  (E, 4)    [N, M, V, sens] │")
    print(f"  │ Physics:       prop_EA, prop_EI          │")
    print(f"  ├──────────────────────────────────────────┤")
    print(f"  │ Network output → y_i = (u_x, u_z, φ)   │")
    print(f"  │ Physics loss  → N=EA·du/dx              │")
    print(f"  │                 M=EI·d²u/dx²            │")
    print(f"  │                 V=EI·d³u/dx³            │")
    print(f"  └──────────────────────────────────────────┘")
    print(f"  │ Physics (computed in Step 3):             │")
    print(f"  │  Constitutive:                           │")
    print(f"  │    N  = EA·∂u_s/∂s                       │")
    print(f"  │    M  = EI·∂²u_n/∂s²                     │")
    print(f"  │  Kinematic (EB):                         │")
    print(f"  │    φ  = ∂u_n/∂s                          │")
    print(f"  │  Equilibrium:                            │")
    print(f"  │    ∂N/∂s + q_s = 0                       │")
    print(f"  │    ∂V/∂s + q_n = 0                       │")
    print(f"  │    V  = ∂M/∂s                            │")
    print(f"  │  (s,n) = local element coordinates       │")
    print(f"  Ready for Step 3 (Model Architecture)")
    print(f"{'='*60}")