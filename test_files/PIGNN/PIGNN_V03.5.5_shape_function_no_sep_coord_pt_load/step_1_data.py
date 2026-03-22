"""
=================================================================
STEP 1: DATA LOADING — Response at NODE level
=================================================================

Node inputs (9 per node):
  coords           (3)  x, y, z
  bc_disp          (1)  1=translations fixed
  bc_rot           (1)  1=rotations fixed
  point_load       (3)  point load at node
  response_node    (1)  1=response measured here, 0=not

Element inputs (11 per element):
  length           (1)
  direction        (3)
  E, A, I22, I33   (4)
  ν, density, J    (3)

=================================================================
"""

import numpy as np
import json
import os
import pickle
from pathlib import Path
from typing import Optional, List, Tuple

try:
    import pyvista as pv
    print(f"[OK] pyvista {pv.__version__}")
except ImportError:
    raise ImportError("pip install pyvista")


class FrameDataLoader:
    """
    Loads Kratos frame data.
    Response location is tracked at NODE level (nearest_node_id).
    Supports POINT_LOAD (point loads on beam nodes).
    """

    def __init__(self,
                 primal_base_dir: str,
                 adjoint_base_dir: str,
                 primal_folder_prefix: str = "case_primal_",
                 adjoint_folder_prefix: str = "case_adjoint_",
                 primal_vtk_subdir: str = "vtk_output_primal",
                 adjoint_vtk_subdir: str = "vtk_output_adjoint",
                 vtk_filename: str = "Structure_1_.vtk"):

        self.primal_base = Path(primal_base_dir)
        self.adjoint_base = Path(adjoint_base_dir)
        self.p_prefix = primal_folder_prefix
        self.a_prefix = adjoint_folder_prefix
        self.p_vtk_sub = primal_vtk_subdir
        self.a_vtk_sub = adjoint_vtk_subdir
        self.vtk_file = vtk_filename

    # ─── PATHS ───

    def _primal_vtk_path(self, case_num: int) -> Path:
        return (self.primal_base /
                f"{self.p_prefix}{case_num}" /
                self.p_vtk_sub / self.vtk_file)

    def _adjoint_vtk_path(self, case_num: int) -> Path:
        return (self.adjoint_base /
                f"{self.a_prefix}{case_num}" /
                self.a_vtk_sub / self.vtk_file)

    def _config_path(self, case_num: int) -> Path:
        return (self.primal_base /
                f"{self.p_prefix}{case_num}" /
                "case_config.json")

    # ─── DISCOVER ───

    def discover_cases(self) -> List[int]:
        case_nums = []
        for folder in self.primal_base.iterdir():
            if folder.is_dir() and folder.name.startswith(self.p_prefix):
                try:
                    num = int(folder.name[len(self.p_prefix):])
                    if (self._primal_vtk_path(num).exists() and
                            self._config_path(num).exists()):
                        case_nums.append(num)
                except ValueError:
                    continue
        case_nums.sort()
        print(f"Found {len(case_nums)} cases: "
              f"[{case_nums[0]}...{case_nums[-1]}]")
        return case_nums

    # ─── CONFIG ───

    def _read_config(self, case_num: int) -> dict:
        with open(self._config_path(case_num), 'r') as f:
            raw = json.load(f)
        config = {}
        config['case_id'] = raw.get('case_id', case_num)
        config['timestamp'] = raw.get('timestamp', '')
        params = raw.get('parameters', {})
        config.update(params)
        return config

    # ─── VTK HELPERS ───

    def _read_vtk(self, filepath: Path) -> pv.DataSet:
        if not filepath.exists():
            raise FileNotFoundError(f"Not found: {filepath}")
        return pv.read(str(filepath))

    def _get_point_field(self, mesh, name: str) -> Optional[np.ndarray]:
        if name in mesh.point_data:
            return np.array(mesh.point_data[name], dtype=np.float64)
        return None

    def _get_cell_field(self, mesh, name: str) -> Optional[np.ndarray]:
        if name in mesh.cell_data:
            arr = np.array(mesh.cell_data[name], dtype=np.float64)
            if arr.ndim > 1 and arr.shape[1] == 1:
                arr = arr.flatten()
            return arr
        return None

    def _extract_connectivity(self, mesh) -> np.ndarray:
        conn = []
        for i in range(mesh.n_cells):
            conn.append(list(mesh.get_cell(i).point_ids))
        return np.array(conn, dtype=np.int64)

    # ─── BC FLAGS ───

    def _build_bc_flags(self, coords, displacement, rotation):
        N = len(coords)
        bc_disp = np.zeros((N, 1), dtype=np.float64)
        bc_rot = np.zeros((N, 1), dtype=np.float64)

        if displacement is not None:
            disp_mag = np.linalg.norm(displacement, axis=1)
            max_disp = np.max(disp_mag)
            threshold = max_disp * 1e-8 if max_disp > 0 else 1e-15
            support_nodes = np.where(disp_mag < threshold)[0]
            bc_disp[support_nodes] = 1.0
            if rotation is not None:
                rot_mag = np.linalg.norm(rotation, axis=1)
                max_rot = np.max(rot_mag)
                rot_thresh = max_rot * 1e-8 if max_rot > 0 else 1e-15
                for node in support_nodes:
                    if rot_mag[node] < rot_thresh:
                        bc_rot[node] = 1.0

        return bc_disp, bc_rot

    # ─── RESPONSE NODE FLAG ───

    def _build_response_node_flag(self, n_nodes: int,
                                    nearest_node_id: int
                                    ) -> np.ndarray:
        resp_flag = np.zeros((n_nodes, 1), dtype=np.float64)
        if 0 <= nearest_node_id < n_nodes:
            resp_flag[nearest_node_id] = 1.0
        return resp_flag

    # ─── ELEMENT GEOMETRY ───

    def _compute_element_geometry(self, coords, connectivity):
        n1 = coords[connectivity[:, 0]]
        n2 = coords[connectivity[:, 1]]
        diff = n2 - n1
        lengths = np.linalg.norm(diff, axis=1)
        safe_L = np.where(lengths > 1e-15, lengths, 1.0)
        directions = diff / safe_L[:, np.newaxis]
        return {'lengths': lengths, 'directions': directions}

    # ─── ELEMENT LOAD (from nodal POINT_LOAD) ───

    def _compute_element_load(self, point_load_nodes, connectivity):
        """
        Convert node-level POINT_LOAD to element-level load.
        Average of both end nodes' point loads.
        """
        n1_load = point_load_nodes[connectivity[:, 0]]  # (E, 3)
        n2_load = point_load_nodes[connectivity[:, 1]]  # (E, 3)
        elem_load = 0.5 * (n1_load + n2_load)
        return elem_load

    # ─── FIX: ZERO OUT COLUMN LOADS ───

    @staticmethod
    def fix_elem_load(elem_load, elem_directions):
        """
        Zero out elem_load for column (vertical) elements.

        POINT_LOAD is applied on beam nodes only. However, junction
        nodes (where beams meet columns) carry the load value, so
        when averaging to elements, column elements that share a
        junction node incorrectly inherit a non-zero load.

        For vertical elements (|dz| > |dx|): set load to zero.
        """
        elem_load_fixed = elem_load.copy()

        for e in range(len(elem_directions)):
            dx = abs(elem_directions[e, 0])
            dz = abs(elem_directions[e, 2])

            if dz > dx:  # Vertical element (column)
                elem_load_fixed[e] = [0.0, 0.0, 0.0]

        n_zeroed = np.sum(np.any(elem_load != elem_load_fixed, axis=1))
        print(f"    Fixed elem_load: zeroed {n_zeroed} column elements")

        return elem_load_fixed

    # ─── LOAD ONE CASE ───

    def load_case(self, case_num: int) -> dict:
        print(f"\n  Case {case_num}:")

        # ═══ 1. CONFIG ═══
        config = self._read_config(case_num)
        nearest_node = config['nearest_node_id']
        traced_elem = config['traced_element_id']
        print(f"    Response: node={nearest_node} "
              f"(traced_elem={traced_elem})")
        print(f"    Params: load={config.get('load_modulus', '?'):.2f}, "
              f"E={config.get('youngs_modulus', '?'):.2e}, "
              f"I22={config.get('I22', '?'):.4e}")

        # ═══ 2. PRIMAL VTK ═══
        primal = self._read_vtk(self._primal_vtk_path(case_num))
        coords = np.array(primal.points, dtype=np.float64)
        connectivity = self._extract_connectivity(primal)
        N = len(coords)
        E = len(connectivity)

        # INPUT: Point load (per node)
        point_load = self._get_point_field(primal, 'POINT_LOAD')
        if point_load is None:
            point_load = np.zeros((N, 3))
            print(f"    ⚠ POINT_LOAD not found in VTK")

        # INPUT: Element properties
        prop_map = {
            'young_modulus':     'YOUNG_MODULUS',
            'cross_area':        'CROSS_AREA',
            'I22':               'I22',
            'I33':               'I33',
            'poisson_ratio':     'POISSON_RATIO',
            'density':           'DENSITY',
            'torsional_inertia': 'TORSIONAL_INERTIA',
        }
        props = {}
        for key, vtk_name in prop_map.items():
            arr = self._get_cell_field(primal, vtk_name)
            props[key] = arr if arr is not None else np.ones(E)

        # OUTPUT: Displacement, Rotation
        displacement = self._get_point_field(primal, 'DISPLACEMENT')
        rotation = self._get_point_field(primal, 'ROTATION')

        # ═══ EXTRACT 2D FRAME DOFs ═══
        if displacement is not None and rotation is not None:
            nodal_disp_2d = np.stack([
                displacement[:, 0],   # u_x
                displacement[:, 2],   # u_z
                rotation[:, 1]        # phi_y
            ], axis=1)
        else:
            nodal_disp_2d = None

        # OUTPUT: Moment, Force
        moment = self._get_cell_field(primal, 'MOMENT')
        force = self._get_cell_field(primal, 'FORCE')

        # ═══ EXTRACT 2D INTERNAL FORCES ═══
        if force is not None:
            elem_N = force[:, 0]
            elem_V = force[:, 2]
        else:
            elem_N = None
            elem_V = None

        if moment is not None:
            elem_M = moment[:, 1]
        else:
            elem_M = None

        # ═══ 3. ADJOINT VTK ═══
        adjoint_path = self._adjoint_vtk_path(case_num)
        I22_sensitivity = None
        if adjoint_path.exists():
            adjoint = self._read_vtk(adjoint_path)
            I22_sensitivity = self._get_cell_field(
                adjoint, 'I22_SENSITIVITY')
            if I22_sensitivity is not None:
                print(f"    Sensitivity: [{I22_sensitivity.min():.4e}, "
                      f"{I22_sensitivity.max():.4e}]")
        else:
            print(f"    ⚠ Adjoint not found")

        # ═══ 4. DERIVED INPUTS ═══
        bc_disp, bc_rot = self._build_bc_flags(
            coords, displacement, rotation)
        response_node_flag = self._build_response_node_flag(
            N, nearest_node)

        geom = self._compute_element_geometry(coords, connectivity)

        # Element-level load (from nodal POINT_LOAD)
        elem_load = self._compute_element_load(point_load, connectivity)

        # Fix: zero out column elements
        elem_load = self.fix_elem_load(elem_load, geom['directions'])

        loaded_nodes = np.where(
            np.linalg.norm(point_load, axis=1) > 1e-10
        )[0]
        loaded_elems = np.where(
            np.linalg.norm(elem_load, axis=1) > 1e-10
        )[0]
        print(f"    Loaded nodes: {len(loaded_nodes)}/{N}")
        print(f"    Loaded elements: {len(loaded_elems)}/{E} "
              f"(beams only)")

        # ═══ 5. ASSEMBLE ═══
        case = {
            # ── INPUTS ──
            'coords':              coords,              # (N, 3)
            'connectivity':        connectivity,        # (E, 2)
            'bc_disp':             bc_disp,             # (N, 1)
            'bc_rot':              bc_rot,              # (N, 1)
            'point_load':          point_load,          # (N, 3)
            'elem_load':           elem_load,           # (E, 3)
            'response_node_flag':  response_node_flag,  # (N, 1)
            'young_modulus':       props['young_modulus'],
            'cross_area':          props['cross_area'],
            'I22':                 props['I22'],
            'I33':                 props['I33'],
            'poisson_ratio':       props['poisson_ratio'],
            'density':             props['density'],
            'torsional_inertia':   props['torsional_inertia'],
            'elem_lengths':        geom['lengths'],
            'elem_directions':     geom['directions'],

            # ── PRIMARY OUTPUTS ──
            'displacement':        displacement,        # (N, 3)
            'rotation':            rotation,            # (N, 3)
            'nodal_disp_2d':       nodal_disp_2d,       # (N, 3)

            # ── PHYSICS OUTPUTS ──
            'moment':              moment,              # (E, 3)
            'force':               force,               # (E, 3)
            'elem_N':              elem_N,              # (E,)
            'elem_V':              elem_V,              # (E,)
            'elem_M':              elem_M,              # (E,)
            'I22_sensitivity':     I22_sensitivity,

            # ── METADATA ──
            'n_nodes':             N,
            'n_elements':          E,
            'case_num':            case_num,
            'nearest_node_id':     nearest_node,
            'traced_element_id':   traced_elem,
            'config':              config,
        }

        support_nodes = np.where(bc_disp.flatten() > 0.5)[0]
        resp_node = np.where(
            response_node_flag.flatten() > 0.5)[0]
        print(f"    Supports: {support_nodes.tolist()}")
        print(f"    Response node: {resp_node.tolist()} ✓")
        print(f"    ✓ Case {case_num} loaded")

        return case

    # ─── LOAD ALL ───

    def load_all(self) -> List[dict]:
        case_nums = self.discover_cases()
        print(f"\n{'═'*60}")
        print(f"  Loading {len(case_nums)} cases")
        print(f"{'═'*60}")

        dataset, failed = [], []
        for num in case_nums:
            try:
                dataset.append(self.load_case(num))
            except Exception as e:
                print(f"    ✗ Case {num} FAILED: {e}")
                failed.append(num)

        print(f"\n{'═'*60}")
        print(f"  LOADED: {len(dataset)}  |  FAILED: {len(failed)}")
        print(f"{'═'*60}")
        self._print_summary(dataset)
        return dataset

    def load_range(self, start: int, end: int) -> List[dict]:
        dataset = []
        for num in range(start, end + 1):
            try:
                dataset.append(self.load_case(num))
            except Exception as e:
                print(f"    ✗ Case {num}: {e}")
        self._print_summary(dataset)
        return dataset

    # ─── SUMMARY ───

    def _print_summary(self, dataset):
        if not dataset:
            print("  No data loaded!")
            return

        n = len(dataset)
        N = dataset[0]['n_nodes']
        E = dataset[0]['n_elements']

        print(f"\n{'═'*60}")
        print(f"  DATASET SUMMARY")
        print(f"{'═'*60}")
        print(f"  Cases: {n}  |  Nodes: {N}  |  Elements: {E}")

        loads = [c['config'].get('load_modulus', 0) for c in dataset]
        Es = [c['config'].get('youngs_modulus', 0) for c in dataset]
        I22s = [c['config'].get('I22', 0) for c in dataset]
        resp_nodes = [c['nearest_node_id'] for c in dataset]

        print(f"\n  Varying parameters:")
        print(f"    Point load:     [{min(loads):.2f}, {max(loads):.2f}]")
        print(f"    E:              [{min(Es):.2e}, {max(Es):.2e}]")
        print(f"    I22:            [{min(I22s):.4e}, {max(I22s):.4e}]")
        print(f"    Response nodes: {sorted(set(resp_nodes))}")

        # ── Verify elem_load fix ──
        c0 = dataset[0]
        unique_fz = np.unique(c0['elem_load'][:, 2].round(4))
        n_loaded = np.sum(np.abs(c0['elem_load'][:, 2]) > 1e-5)
        n_beams = np.sum(
            np.abs(c0['elem_directions'][:, 0]) >
            np.abs(c0['elem_directions'][:, 2])
        )
        n_columns = E - n_beams
        print(f"\n  elem_load verification (case 0):")
        print(f"    Unique Fz values: {unique_fz}")
        print(f"    Loaded elements:  {n_loaded} "
              f"(beams={n_beams}, columns={n_columns})")

        # Check loaded nodes
        n_loaded_nodes = np.sum(
            np.linalg.norm(c0['point_load'], axis=1) > 1e-10
        )
        print(f"    Loaded nodes:     {n_loaded_nodes}/{N}")

        if n_loaded <= n_beams:
            print(f"    ✓ No columns carry load (fix working)")
        else:
            print(f"    ✗ WARNING: columns still carry load!")

        print(f"\n  Feature dimensions:")
        print(f"  ┌─────────────────────────────────────────┐")
        print(f"  │ NODE INPUTS (per node):          9      │")
        print(f"  │   coords              (3) x,y,z         │")
        print(f"  │   bc_disp             (1) constraint    │")
        print(f"  │   bc_rot              (1) constraint    │")
        print(f"  │   point_load          (3) load          │")
        print(f"  │   response_node_flag  (1) response loc  │")
        print(f"  ├─────────────────────────────────────────┤")
        print(f"  │ ELEMENT INPUTS (per element):    11     │")
        print(f"  │   length              (1) geometry      │")
        print(f"  │   direction           (3) geometry      │")
        print(f"  │   E, A, I22, I33      (4) properties    │")
        print(f"  │   ν, density, J       (3) properties    │")
        print(f"  ├─────────────────────────────────────────┤")
        print(f"  │ NODE OUTPUTS (network target):          │")
        print(f"  │   nodal_disp_2d       (3) [u_x,u_z,φ]  │")
        print(f"  ├─────────────────────────────────────────┤")
        print(f"  │ ELEMENT OUTPUTS (physics target):       │")
        print(f"  │   elem_N              (1) axial force   │")
        print(f"  │   elem_M              (1) moment        │")
        print(f"  │   elem_V              (1) shear force   │")
        print(f"  │   I22_sensitivity     (1) dBM/dI22      │")
        print(f"  └─────────────────────────────────────────┘")
        print(f"{'═'*60}\n")

    # ─── SAVE / LOAD ───

    @staticmethod
    def save(dataset, filepath):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        mb = os.path.getsize(filepath) / 1e6
        print(f"Saved: {filepath} ({mb:.1f} MB, {len(dataset)} cases)")

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Loaded: {filepath} ({len(dataset)} cases)")
        return dataset


# ================================================================
if __name__ == "__main__":

    print("=" * 60)
    print("  STEP 1: Load Kratos Data")
    print("=" * 60)

    loader = FrameDataLoader(
        primal_base_dir="test_files/Kratos_data_creation/primal",
        adjoint_base_dir="test_files/Kratos_data_creation/adjoint",
        primal_folder_prefix="case_primal_",
        adjoint_folder_prefix="case_adjoint_",
        primal_vtk_subdir="vtk_output_primal",
        adjoint_vtk_subdir="vtk_output_adjoint",
        vtk_filename="Structure_0_1.vtk"
    )

    dataset = loader.load_all()

    import os
    from pathlib import Path
    print(f"Working directory: {os.getcwd()}")
    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)
    print(f"Working directory: {os.getcwd()}")

    FrameDataLoader.save(dataset, "DATA/frame_dataset.pkl")

    if dataset:
        c = dataset[0]
        print(f"\nVerification case {c['case_num']}:")
        print(f"  response_node_flag: "
              f"node={np.where(c['response_node_flag'].flatten()>0.5)[0]}")
        print(f"  (nearest_node_id from config: {c['nearest_node_id']})")

        # ── Verify point load distribution ──
        loaded_nodes = np.where(
            np.linalg.norm(c['point_load'], axis=1) > 1e-10
        )[0]
        print(f"\n  Loaded nodes: {loaded_nodes.tolist()}")
        print(f"  Point load values (unique Fz): "
              f"{np.unique(c['point_load'][:, 2].round(4))}")

        # ── Verify elem_load fix ──
        print(f"  elem_load unique Fz: "
              f"{np.unique(c['elem_load'][:, 2].round(4))}")

    print("\n  Step 1 COMPLETE → proceed to Step 2")