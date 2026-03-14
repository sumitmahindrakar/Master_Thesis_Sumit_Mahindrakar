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

ADDITIONAL INFORMATION for Physics Loss:
  A. Face assignment    → which element connects to which node face
  B. Equivalent loads   → F_ext from UDL for equilibrium RHS
  C. Updated graph      → new tensors stored in Data object
  D. Updated verify     → checks for new tensors

Face convention:
  Face 0 = +x  (beam going right)
  Face 1 = -x  (beam going left)
  Face 2 = +z  (column going up)
  Face 3 = -z  (column going down)

CHANGES FROM ORIGINAL:                                    # ◀◀◀ NEW DOCSTRING
  1. Removed DataNormalizer class (moved to normalizer.py) # ◀◀◀
  2. Import MinMaxNormalizer + PhysicsScaler               # ◀◀◀
  3. Main block uses PhysicsScaler THEN MinMaxNormalizer    # ◀◀◀
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
# 2A. GRAPH BUILDER (UNCHANGED)
# ================================================================

class FrameGraphBuilder:
    """
    Converts frame case dicts → PyG Data objects for strong-form PIGNN.

    Network predicts 15 values per node:
      - 3 global displacements: ux, uz, θy
      - 12 face forces: (Fx, Fz, My) × 4 faces

    Each structural element → TWO directed edges (bidirectional).
    """

    NODE_INPUT_DIM  = 9
    EDGE_INPUT_DIM  = 10
    NODE_TARGET_DIM = 3
    ELEM_TARGET_DIM = 4

    # ─── EDGE INDEX (UNCHANGED) ───

    def build_edge_index(self, connectivity: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        E = len(connectivity)
        src = np.concatenate([connectivity[:, 0], connectivity[:, 1]])
        dst = np.concatenate([connectivity[:, 1], connectivity[:, 0]])
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)
        element_map = np.tile(np.arange(E), 2).astype(np.int64)
        return edge_index, element_map

    # ─── NODE FEATURES (UNCHANGED) ───

    def build_node_features(self, case: dict) -> np.ndarray:
        N = case['n_nodes']
        feat = np.zeros((N, self.NODE_INPUT_DIM), dtype=np.float64)
        feat[:, 0:3] = case['coords'][:, :3]
        feat[:, 3:4] = case['bc_disp']
        feat[:, 4:5] = case['bc_rot']
        feat[:, 5:8] = case['line_load'][:, :3]
        feat[:, 8:9] = case['response_node_flag']
        return feat

    # ─── EDGE FEATURES (UNCHANGED) ───

    def build_edge_features(self, case, element_map):
        E = case['n_elements']
        elem_feat = np.column_stack([
            case['elem_lengths'],
            case['elem_directions'],
            case['young_modulus'],
            case['cross_area'],
            case['I22'],
            case['elem_load'],
        ])

        edge_feat = np.zeros((2 * E, self.EDGE_INPUT_DIM),
                            dtype=np.float64)
        edge_feat[:E]  = elem_feat
        edge_feat[E:]  = elem_feat
        edge_feat[E:, 1:4] *= -1.0
        return edge_feat

    # ─── NODE TARGETS (UNCHANGED) ───

    def build_node_targets(self, case: dict) -> np.ndarray:
        N = case['n_nodes']
        tgt = np.zeros((N, self.NODE_TARGET_DIM), dtype=np.float64)
        if 'nodal_disp_2d' in case and case['nodal_disp_2d'] is not None:
            return case['nodal_disp_2d'].astype(np.float64)
        if case.get('displacement') is not None:
            tgt[:, 0] = case['displacement'][:, 0]
            tgt[:, 1] = case['displacement'][:, 2]
        if case.get('rotation') is not None:
            tgt[:, 2] = case['rotation'][:, 1]
        return tgt

    # ─── ELEMENT TARGETS (UNCHANGED) ───

    def build_element_targets(self, case: dict) -> np.ndarray:
        E = case['n_elements']
        tgt = np.zeros((E, self.ELEM_TARGET_DIM), dtype=np.float64)
        if 'elem_N' in case and case['elem_N'] is not None:
            tgt[:, 0] = case['elem_N']
        elif case.get('force') is not None:
            tgt[:, 0] = case['force'][:, 0]
        if 'elem_M' in case and case['elem_M'] is not None:
            tgt[:, 1] = case['elem_M']
        elif case.get('moment') is not None:
            tgt[:, 1] = case['moment'][:, 1]
        if 'elem_V' in case and case['elem_V'] is not None:
            tgt[:, 2] = case['elem_V']
        elif case.get('force') is not None:
            tgt[:, 2] = case['force'][:, 2]
        if case.get('I22_sensitivity') is not None:
            tgt[:, 3] = case['I22_sensitivity']
        return tgt

    # ─── FACE ASSIGNMENT (UNCHANGED) ───

    def build_face_assignment(self, connectivity: np.ndarray,
                               elem_directions: np.ndarray,
                               n_nodes: int
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        E = len(connectivity)
        N = n_nodes

        face_mask       = np.zeros((N, 4), dtype=np.float64)
        face_element_id = np.full((N, 4), -1, dtype=np.int64)
        face_is_A_end   = np.full((N, 4), -1, dtype=np.int64)

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

            face_mask[nA, face_A]       = 1.0
            face_element_id[nA, face_A] = e
            face_is_A_end[nA, face_A]   = 1

            face_mask[nB, face_B]       = 1.0
            face_element_id[nB, face_B] = e
            face_is_A_end[nB, face_B]   = 0

        max_faces = face_mask.sum(axis=1).max()
        assert max_faces <= 4, \
            f"Node has {max_faces} faces — exceeds max 4!"

        n_connected = int(face_mask.sum())
        print(f"    Face assignment: {n_connected} connections "
              f"({E} elements × 2 ends)")
        print(f"    Max faces per node: {int(max_faces)}")

        return face_mask, face_element_id, face_is_A_end

    # ─── EQUIVALENT NODAL LOADS (UNCHANGED) ───

    def build_equivalent_nodal_loads(self,
                                      connectivity: np.ndarray,
                                      elem_directions: np.ndarray,
                                      elem_load: np.ndarray,
                                      elem_lengths: np.ndarray,
                                      n_nodes: int
                                      ) -> np.ndarray:
        E = len(connectivity)
        N = n_nodes
        F_ext = np.zeros((N, 3), dtype=np.float64)

        for e in range(E):
            nA, nB = connectivity[e]
            L  = elem_lengths[e]
            qx = elem_load[e, 0]
            qz = elem_load[e, 2]
            dx = elem_directions[e, 0]
            dz = elem_directions[e, 2]

            q_mag = np.sqrt(qx**2 + qz**2)
            if q_mag < 1e-15:
                continue

            Fx_half = qx * L / 2.0
            Fz_half = qz * L / 2.0

            F_ext[nA, 0] += Fx_half
            F_ext[nA, 1] += Fz_half
            F_ext[nB, 0] += Fx_half
            F_ext[nB, 1] += Fz_half

            cos_a = dx
            sin_a = dz
            q_loc = -qx * sin_a + qz * cos_a

            M_fixed = q_loc * L**2 / 12.0

            F_ext[nA, 2] += +M_fixed
            F_ext[nB, 2] += -M_fixed

        loaded_nodes = np.where(
            np.linalg.norm(F_ext, axis=1) > 1e-10
        )[0]
        print(f"    Equivalent nodal loads: {len(loaded_nodes)} "
              f"nodes have non-zero F_ext")
        print(f"      Fz range: [{F_ext[:, 1].min():.2f}, "
              f"{F_ext[:, 1].max():.2f}]")
        print(f"      My range: [{F_ext[:, 2].min():.4f}, "
              f"{F_ext[:, 2].max():.4f}]")

        return F_ext

    # ─── CASE → GRAPH (UNCHANGED) ───

    def case_to_graph(self, case: dict, case_id: int = 0) -> Data:
        conn = case['connectivity']
        N = case['n_nodes']
        E = case['n_elements']

        edge_index, element_map = self.build_edge_index(conn)
        node_feat = self.build_node_features(case)
        edge_feat = self.build_edge_features(case, element_map)
        node_tgt  = self.build_node_targets(case)
        elem_tgt  = self.build_element_targets(case)

        face_mask, face_element_id, face_is_A_end = \
            self.build_face_assignment(
                conn, case['elem_directions'], N
            )

        F_ext = self.build_equivalent_nodal_loads(
            conn, case['elem_directions'],
            case['elem_load'], case['elem_lengths'], N
        )

        data = Data(
            x          = torch.tensor(node_feat, dtype=torch.float32),
            edge_index = torch.tensor(edge_index, dtype=torch.long),
            edge_attr  = torch.tensor(edge_feat,  dtype=torch.float32),

            elem_load = torch.tensor(
                case['elem_load'], dtype=torch.float32),

            y_node    = torch.tensor(node_tgt,  dtype=torch.float32),
            y_element = torch.tensor(elem_tgt,  dtype=torch.float32),

            element_map     = torch.tensor(element_map, dtype=torch.long),
            connectivity    = torch.tensor(conn, dtype=torch.long),
            coords          = torch.tensor(
                case['coords'], dtype=torch.float32),
            elem_lengths    = torch.tensor(
                case['elem_lengths'], dtype=torch.float32),
            elem_directions = torch.tensor(
                case['elem_directions'], dtype=torch.float32),

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

            face_mask = torch.tensor(
                face_mask, dtype=torch.float32),
            face_element_id = torch.tensor(
                face_element_id, dtype=torch.long),
            face_is_A_end = torch.tensor(
                face_is_A_end, dtype=torch.long),

            F_ext = torch.tensor(F_ext, dtype=torch.float32),

            num_nodes_val     = N,
            n_elements        = E,
            case_id           = case_id,
            nearest_node_id   = case['nearest_node_id'],
            traced_element_id = case['traced_element_id'],
        )

        return data

    # ─── BUILD FULL DATASET (UNCHANGED) ───

    def build_dataset(self, cases: List[dict]) -> List[Data]:
        data_list = [
            self.case_to_graph(case, case_id=i)
            for i, case in enumerate(cases)
        ]

        d = data_list[0]
        print(f"\n  Built {len(data_list)} graphs")
        print(f"    Nodes per graph:  {d.num_nodes}")
        print(f"    Directed edges:   {d.edge_index.shape[1]}")
        print(f"    Node features:    {d.x.shape[1]}")
        print(f"    Edge features:    {d.edge_attr.shape[1]}")
        print(f"    Node targets:     {d.y_node.shape[1]}")
        print(f"    Element targets:  {d.y_element.shape[1]}")
        print(f"    Face mask:        {d.face_mask.shape}")
        print(f"    F_ext:            {d.F_ext.shape}")

        return data_list


# ════════════════════════════════════════════════════════════════
# REMOVED: DataNormalizer class                            # ◀◀◀ REMOVED
# Was here. Now lives in normalizer.py as MinMaxNormalizer # ◀◀◀ REMOVED
# ════════════════════════════════════════════════════════════════


# ================================================================
# 2C. SPLIT & DATALOADERS (UNCHANGED)
# ================================================================

def split_dataset(data_list: List[Data],
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   seed: int = 42) -> Tuple[list, list, list]:
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
# 2D. VERIFICATION (UNCHANGED)
# ================================================================

def verify_graph(data: Data) -> bool:
    print(f"\n{'═'*60}")
    print(f"  GRAPH VERIFICATION (with Face Assignment)")
    print(f"{'═'*60}")

    N = data.num_nodes
    E = data.n_elements
    all_ok = True

    print(f"\n  1. SHAPES:")
    checks = [
        ('x',              data.x.shape,              (N, 9)),
        ('edge_index',     data.edge_index.shape,     (2, 2*E)),
        ('edge_attr',      data.edge_attr.shape,      (2*E, 10)),
        ('y_node',         data.y_node.shape,         (N, 3)),
        ('y_element',      data.y_element.shape,      (E, 4)),
        ('element_map',    data.element_map.shape,     (2*E,)),
        ('connectivity',   data.connectivity.shape,    (E, 2)),
        ('prop_E',         data.prop_E.shape,         (E,)),
        ('prop_A',         data.prop_A.shape,         (E,)),
        ('prop_I22',       data.prop_I22.shape,       (E,)),
        ('face_mask',      data.face_mask.shape,      (N, 4)),
        ('face_element_id',data.face_element_id.shape,(N, 4)),
        ('face_is_A_end',  data.face_is_A_end.shape,  (N, 4)),
        ('F_ext',          data.F_ext.shape,          (N, 3)),
    ]
    for name, actual, expected in checks:
        ok = actual == expected
        if not ok:
            all_ok = False
        print(f"    {'✓' if ok else '✗'} {name:<18} "
              f"{str(actual):<15} (expected {expected})")

    print(f"\n  2. FACE ASSIGNMENT:")
    fm = data.face_mask
    faces_per_node = fm.sum(dim=1)
    print(f"    Faces per node: min={int(faces_per_node.min())}, "
          f"max={int(faces_per_node.max())}, "
          f"mean={faces_per_node.mean():.1f}")
    print(f"    Total connections: {int(fm.sum())} "
          f"(expected: {2*E})")

    ok = int(fm.sum()) == 2 * E
    if not ok:
        all_ok = False
    print(f"    {'✓' if ok else '✗'} Connection count matches 2E")

    face_names = ['+x', '-x', '+z', '-z']
    for f in range(4):
        count = int(fm[:, f].sum())
        print(f"    Face {face_names[f]}: {count} connections")

    print(f"\n  3. FACE CONSISTENCY:")
    feid = data.face_element_id
    faa  = data.face_is_A_end
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

    print(f"    Checked {n_checked} connections: "
          f"{n_ok}/{n_checked} correct")
    if n_ok != n_checked:
        all_ok = False

    print(f"\n  4. EQUIVALENT NODAL LOADS (F_ext):")
    fext = data.F_ext
    print(f"    Fx_ext: [{fext[:, 0].min():.4f}, "
          f"{fext[:, 0].max():.4f}]")
    print(f"    Fz_ext: [{fext[:, 1].min():.4f}, "
          f"{fext[:, 1].max():.4f}]")
    print(f"    My_ext: [{fext[:, 2].min():.6f}, "
          f"{fext[:, 2].max():.6f}]")

    loaded = torch.where(fext.abs().sum(dim=1) > 1e-10)[0]
    print(f"    Loaded nodes: {len(loaded)}/{N}")

    print(f"\n  5. SPECIAL NODES:")
    bc = data.bc_disp.squeeze(-1)
    support_nodes = torch.where(bc > 0.5)[0].tolist()
    resp = data.x[:, 8]
    resp_nodes = torch.where(resp > 0.5)[0].tolist()
    print(f"    Supports: {support_nodes}")
    print(f"    Response: {resp_nodes}")

    print(f"\n  6. PHYSICS CONSTANTS:")
    EA = data.prop_E * data.prop_A
    EI = data.prop_E * data.prop_I22
    print(f"    EA:  [{EA.min():.4e}, {EA.max():.4e}]")
    print(f"    EI:  [{EI.min():.4e}, {EI.max():.4e}]")

    # ── NEW: Check physics scales if present ──             # ◀◀◀ NEW SECTION
    if hasattr(data, 'F_c'):                                  # ◀◀◀
        print(f"\n  7. PHYSICS SCALES (non-dimensionalization):")  # ◀◀◀
        print(f"    L_c     = {data.L_c.item():.4e}  "        # ◀◀◀
              f"(char. length)")                                # ◀◀◀
        print(f"    EI_c    = {data.EI_c.item():.4e}  "       # ◀◀◀
              f"(char. bending stiffness)")                     # ◀◀◀
        print(f"    EA_c    = {data.EA_c.item():.4e}  "       # ◀◀◀
              f"(char. axial stiffness)")                       # ◀◀◀
        print(f"    q_c     = {data.q_c.item():.4e}  "        # ◀◀◀
              f"(char. load intensity)")                        # ◀◀◀
        print(f"    F_c     = {data.F_c.item():.4e}  "        # ◀◀◀
              f"(char. force)")                                 # ◀◀◀
        print(f"    M_c     = {data.M_c.item():.4e}  "        # ◀◀◀
              f"(char. moment)")                                # ◀◀◀
        print(f"    u_c     = {data.u_c.item():.4e}  "        # ◀◀◀
              f"(char. displacement)")                          # ◀◀◀
        print(f"    theta_c = {data.theta_c.item():.4e}  "    # ◀◀◀
              f"(char. rotation)")                              # ◀◀◀
    else:                                                      # ◀◀◀
        print(f"\n  7. PHYSICS SCALES: not yet computed")      # ◀◀◀
        print(f"     Run PhysicsScaler.compute_and_store_list()")  # ◀◀◀

    status = "ALL PASSED ✓" if all_ok else "SOME FAILED ✗"
    print(f"\n  RESULT: {status}")
    print(f"{'═'*60}\n")
    return all_ok


# ================================================================
# 2E. VISUALIZATION (UNCHANGED)
# ================================================================

def visualize_frame_graph(data: Data,
                           title: str = "Frame Graph"):
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

    # ─── Load Step 1 data (UNCHANGED) ───
    print("\n── Loading dataset from Step 1 ──")
    with open("DATA/frame_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    print(f"  Loaded {len(dataset)} cases")

    # ─── Build graphs (UNCHANGED) ───
    print("\n── Building PyG graphs ──")
    builder = FrameGraphBuilder()
    data_list = builder.build_dataset(dataset)

    # ─── Verify (UNCHANGED) ───
    print("\n── Verifying first graph ──")
    verify_graph(data_list[0])

    # ─── Visualize (UNCHANGED) ───
    print("\n── Visualizing ──")
    visualize_frame_graph(data_list[0],
                           title=f"Case {dataset[0]['case_num']}")

    # ════════════════════════════════════════════════════════
    # ◀◀◀ CHANGED SECTION START — Was DataNormalizer, now:
    #   1. PhysicsScaler (adds characteristic scales)
    #   2. MinMaxNormalizer (replaces z-score with [0,1])
    # ════════════════════════════════════════════════════════

    from normalizer import MinMaxNormalizer, PhysicsScaler    # ◀◀◀ NEW IMPORT

    # ── Step A: Physics scales (BEFORE normalization) ──    # ◀◀◀ NEW
    print("\n── Computing physics scales ──")                 # ◀◀◀ NEW
    data_list = PhysicsScaler.compute_and_store_list(          # ◀◀◀ NEW
        data_list                                              # ◀◀◀ NEW
    )                                                          # ◀◀◀ NEW

    # ── Step B: Min-Max normalize inputs ──                  # ◀◀◀ CHANGED
    print("\n── Normalizing inputs (Min-Max → [0,1]) ──")     # ◀◀◀ CHANGED
    normalizer = MinMaxNormalizer()                            # ◀◀◀ CHANGED (was DataNormalizer)
    normalizer.fit(data_list)                                  # ◀◀◀ same API
    data_list_norm = normalizer.transform_list(data_list)      # ◀◀◀ same API

    # ════════════════════════════════════════════════════════
    # ◀◀◀ CHANGED SECTION END
    # ════════════════════════════════════════════════════════

    # ─── Split (UNCHANGED) ───
    print("\n── Splitting ──")
    train, val, test = split_dataset(data_list_norm)

    # ─── DataLoaders (UNCHANGED) ───
    print("\n── DataLoaders ──")
    loaders = create_dataloaders(train, val, test, batch_size=4)

    # ─── Test batch (UNCHANGED) ───
    print("\n── Test batch ──")
    batch = next(iter(loaders['train']))
    print(f"  batch.x:              {batch.x.shape}")
    print(f"  batch.edge_index:     {batch.edge_index.shape}")
    print(f"  batch.edge_attr:      {batch.edge_attr.shape}")
    print(f"  batch.y_node:         {batch.y_node.shape}")
    print(f"  batch.y_element:      {batch.y_element.shape}")
    print(f"  batch.face_mask:      {batch.face_mask.shape}")
    print(f"  batch.face_element_id:{batch.face_element_id.shape}")
    print(f"  batch.F_ext:          {batch.F_ext.shape}")

    # ── NEW: Check physics scales survived batching ──       # ◀◀◀ NEW
    print(f"\n── Physics scales in batch ──")                  # ◀◀◀ NEW
    if hasattr(batch, 'F_c'):                                  # ◀◀◀ NEW
        print(f"  batch.F_c:     {batch.F_c}")                # ◀◀◀ NEW
        print(f"  batch.M_c:     {batch.M_c}")                # ◀◀◀ NEW
        print(f"  batch.u_c:     {batch.u_c}")                # ◀◀◀ NEW
        print(f"  batch.theta_c: {batch.theta_c}")            # ◀◀◀ NEW
        print(f"  ✓ Physics scales accessible in batch")       # ◀◀◀ NEW
    else:                                                      # ◀◀◀ NEW
        print(f"  ✗ Physics scales NOT in batch!")             # ◀◀◀ NEW
        print(f"    Check PhysicsScaler ran before batching")  # ◀◀◀ NEW

    # ─── Save ───
    print("\n── Saving ──")
    os.makedirs("DATA", exist_ok=True)
    torch.save(data_list, "DATA/graph_dataset.pt")
    torch.save(data_list_norm, "DATA/graph_dataset_norm.pt")
    normalizer.save("DATA/normalizer_minmax.pt")               # ◀◀◀ CHANGED filename

    # ─── Verify normalized graph has scales ──                # ◀◀◀ NEW
    print("\n── Verifying normalized graph has physics scales ──")  # ◀◀◀ NEW
    d_norm = data_list_norm[0]                                  # ◀◀◀ NEW
    print(f"  Input x range: [{d_norm.x.min():.4f}, "          # ◀◀◀ NEW
          f"{d_norm.x.max():.4f}]  "                            # ◀◀◀ NEW
          f"(should be ~[0, 1])")                               # ◀◀◀ NEW
    print(f"  F_c still present: {hasattr(d_norm, 'F_c')}")    # ◀◀◀ NEW
    print(f"  F_ext unchanged:   "                              # ◀◀◀ NEW
          f"[{d_norm.F_ext.min():.2f}, "                        # ◀◀◀ NEW
          f"{d_norm.F_ext.max():.2f}]  "                        # ◀◀◀ NEW
          f"(should be raw units, NOT [0,1])")                  # ◀◀◀ NEW

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
    print(f"  │ CHANGES:                                 │")    # ◀◀◀ NEW
    print(f"  │  Input norm:  Min-Max [0,1] (was z-score)│")    # ◀◀◀ NEW
    print(f"  │  Physics:     F_c, M_c, u_c, θ_c added  │")    # ◀◀◀ NEW
    print(f"  │  Residuals:   Will be /F_c, /M_c in loss │")    # ◀◀◀ NEW
    print(f"  └──────────────────────────────────────────┘")
    print(f"{'='*60}")