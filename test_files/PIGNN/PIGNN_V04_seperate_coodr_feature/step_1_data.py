"""
=================================================================
STEP 1: DATA LOADING — Response at NODE level (CORRECTED)
=================================================================

Node inputs (9 per node):
  coords           (3)  x, y, z
  bc_disp          (1)  1=translations fixed
  bc_rot           (1)  1=rotations fixed
  line_load        (3)  distributed load
  response_node    (1)  1=response measured here, 0=not  ← NEW

Element inputs (11 per element):
  length           (1)
  direction        (3)
  E, A, I22, I33   (4)
  ν, density, J    (3)
  (no response_flag here anymore)

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
        """
        Read case_config.json, flatten nested 'parameters'.
        
        After flattening:
            config['traced_element_id']  = 8
            config['nearest_node_id']    = 10
            config['stress_location']    = 1
            config['udl']                = 37.85
            config['youngs_modulus']      = 1.07e11
            config['I22']                = 0.000225
            config['response_coords']    = [3.84, 0, 7.13]
        """
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
        """
        bc_disp (N,1): 1=all translations fixed
        bc_rot  (N,1): 1=all rotations fixed
        """
        N = len(coords)
        bc_disp = np.zeros((N, 1), dtype=np.float64)
        bc_rot = np.zeros((N, 1), dtype=np.float64)

        if displacement is not None:
            disp_mag = np.linalg.norm(displacement, axis=1)
            max_disp = np.max(disp_mag)
            threshold = max_disp * 1e-8 if max_disp > 0 else 1e-15
            support_nodes = np.where(disp_mag < threshold)[0]
            bc_disp[support_nodes] = 1.0
            bc_rot[support_nodes] = 0.0

        return bc_disp, bc_rot

    # ─── RESPONSE NODE FLAG ───

    def _build_response_node_flag(self, n_nodes: int,
                                    nearest_node_id: int
                                    ) -> np.ndarray:
        """
        Build response node flag.
        
        (N, 1): 1.0 at the response node, 0.0 elsewhere.
        
        This tells the GNN: "The bending moment sensitivity
        is measured at THIS node's location."
        """
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

    def _compute_element_load(self, line_load_nodes, connectivity):
        """
        Convert node-level LINE_LOAD to element-level UDL.
        
        Kratos applies LINE_LOAD per condition (≈ per element),
        then we transfer to nodes for VTK. Now reverse that:
        
        Logic: If BOTH end nodes have the same non-zero load,
        the element carries that UDL. If either is zero,
        check if the other has load (partial). Average is safe.
        
        Args:
            line_load_nodes: (N, 3) from VTK point_data
            connectivity:    (E, 2) element node pairs
        
        Returns:
            elem_load: (E, 3) distributed load per element
        """
        n1_load = line_load_nodes[connectivity[:, 0]]  # (E, 3)
        n2_load = line_load_nodes[connectivity[:, 1]]  # (E, 3)
        
        # For UDL: both nodes should have the same value
        # Average handles edge cases (partial loading, interpolation)
        elem_load = 0.5 * (n1_load + n2_load)
        
        return elem_load

    # ─── LOAD ONE CASE ───

    def load_case(self, case_num: int) -> dict:
        """
        Load one case.
        
        Returns dict:
        
        INPUTS:
          coords              (N, 3)   node coordinates
          connectivity        (E, 2)   element connectivity
          bc_disp             (N, 1)   displacement BC flag
          bc_rot              (N, 1)   rotation BC flag
          line_load           (N, 3)   distributed load
          response_node_flag  (N, 1)   1 at response node  ← NODE LEVEL
          young_modulus       (E,)     from VTK
          cross_area          (E,)     from VTK
          I22                 (E,)     from VTK
          I33                 (E,)     from VTK
          poisson_ratio       (E,)     from VTK
          density             (E,)     from VTK
          torsional_inertia   (E,)     from VTK
          elem_lengths        (E,)     computed
          elem_directions     (E, 3)   computed
        
        OUTPUTS:
          displacement        (N, 3)
          rotation            (N, 3)
          moment              (E, 3)
          force               (E, 3)
          I22_sensitivity     (E,)
        
        METADATA:
          case_num, nearest_node_id, traced_element_id, config
        """
        print(f"\n  Case {case_num}:")

        # ═══ 1. CONFIG ═══
        config = self._read_config(case_num)
        nearest_node = config['nearest_node_id']
        traced_elem = config['traced_element_id']
        print(f"    Response: node={nearest_node} "
              f"(traced_elem={traced_elem})")
        print(f"    Params: udl={config.get('udl', '?'):.2f}, "
              f"E={config.get('youngs_modulus', '?'):.2e}, "
              f"I22={config.get('I22', '?'):.4e}")

        # ═══ 2. PRIMAL VTK ═══
        primal = self._read_vtk(self._primal_vtk_path(case_num))
        coords = np.array(primal.points, dtype=np.float64)
        connectivity = self._extract_connectivity(primal)
        N = len(coords)
        E = len(connectivity)

        # INPUT: Line load
        line_load = self._get_point_field(primal, 'LINE_LOAD')
        if line_load is None:
            line_load = np.zeros((N, 3))
            print(f"    ⚠ LINE_LOAD not found")

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
        # For 2D frames in XZ plane:
        #   u_x = displacement[:, 0]   (axial)
        #   u_z = displacement[:, 2]   (transverse)
        #   phi = rotation[:, 1]       (rotation about Y-axis)
        
        if displacement is not None:
            nodal_disp_2d = np.stack([
                displacement[:, 0],   # u_x
                displacement[:, 2],   # u_z
                rotation[:, 1]        # phi_y
            ], axis=1)                # (N, 3) → [u_x, u_z, φ]
        else:
            nodal_disp_2d = None

        # OUTPUT: Moment, Force
        moment = self._get_cell_field(primal, 'MOMENT')
        force = self._get_cell_field(primal, 'FORCE')
        # ═══ EXTRACT 2D INTERNAL FORCES ═══
        # For 2D frames: N (axial), V (shear in z), M (moment about y)
        if force is not None:
            elem_N = force[:, 0]       # Axial force (along x)
            elem_V = force[:, 2]       # Shear force (along z)
        else:
            elem_N = None
            elem_V = None

        if moment is not None:
            elem_M = moment[:, 1]      # Bending moment (about y)
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

        # Element-level UDL (from node-level LINE_LOAD)
        elem_load = self._compute_element_load(line_load, connectivity)

        # Print which elements carry load
        loaded_elems = np.where(
            np.linalg.norm(elem_load, axis=1) > 1e-10
        )[0]
        print(f"    Loaded elements: {len(loaded_elems)}/{E} "
            f"(UDL on beams only)")

        # RESPONSE AT NODE LEVEL
        response_node_flag = self._build_response_node_flag(
            N, nearest_node)

        geom = self._compute_element_geometry(coords, connectivity)

        # ═══ 5. ASSEMBLE ═══
        case = {
            # ── INPUTS ──
            'coords':              coords,              # (N, 3)
            'connectivity':        connectivity,        # (E, 2)
            'bc_disp':             bc_disp,             # (N, 1)
            'bc_rot':              bc_rot,              # (N, 1)
            'line_load':           line_load,           # (N, 3)
            'elem_load':           elem_load,
            'response_node_flag':  response_node_flag,  # (N, 1) ← NODE
            'young_modulus':       props['young_modulus'],
            'cross_area':          props['cross_area'],
            'I22':                 props['I22'],
            'I33':                 props['I33'],
            'poisson_ratio':       props['poisson_ratio'],
            'density':             props['density'],
            'torsional_inertia':   props['torsional_inertia'],
            'elem_lengths':        geom['lengths'],
            'elem_directions':     geom['directions'],
           # ── PRIMARY OUTPUTS (network targets) ──
            'displacement':        displacement,        # (N, 3) full 3D
            'rotation':            rotation,            # (N, 3) full 3D
            'nodal_disp_2d':       nodal_disp_2d,       # (N, 3) [u_x, u_z, φ]
            
            # ── PHYSICS OUTPUTS (strong-form targets) ──
            'moment':              moment,              # (E, 3) full 3D
            'force':               force,               # (E, 3) full 3D
            'elem_N':              elem_N,              # (E,) axial force
            'elem_V':              elem_V,              # (E,) shear force
            'elem_M':              elem_M,              # (E,) bending moment
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

        udls = [c['config'].get('udl', 0) for c in dataset]
        Es = [c['config'].get('youngs_modulus', 0) for c in dataset]
        I22s = [c['config'].get('I22', 0) for c in dataset]
        resp_nodes = [c['nearest_node_id'] for c in dataset]

        print(f"\n  Varying parameters:")
        print(f"    UDL:            [{min(udls):.2f}, {max(udls):.2f}]")
        print(f"    E:              [{min(Es):.2e}, {max(Es):.2e}]")
        print(f"    I22:            [{min(I22s):.4e}, {max(I22s):.4e}]")
        print(f"    Response nodes: {sorted(set(resp_nodes))}")

        print(f"\n  Feature dimensions:")
        print(f"  ┌─────────────────────────────────────────┐")
        print(f"  │ NODE INPUTS (per node):          9      │")
        print(f"  │   coords              (3) x,y,z         │")
        print(f"  │   bc_disp             (1) constraint    │")
        print(f"  │   bc_rot              (1) constraint    │")
        print(f"  │   line_load           (3) load          │")
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
        primal_folder_prefix="case_primal_",         # ← FIXED
        adjoint_folder_prefix="case_adjoint_",
        primal_vtk_subdir="vtk_output_primal",       # primal has subdir
        adjoint_vtk_subdir="vtk_output_adjoint",                        # ← FIXED: no subdir
        vtk_filename="Structure_0_1.vtk"              # ← FIXED
    )

    dataset = loader.load_all()
    FrameDataLoader.save(dataset, "test_files/PIGNN/PIGNN_V04_only_displ/DATA/frame_dataset.pkl")

    if dataset:
        c = dataset[0]
        print(f"\nVerification case {c['case_num']}:")
        print(f"  response_node_flag: "
              f"node={np.where(c['response_node_flag'].flatten()>0.5)[0]}")
        print(f"  (nearest_node_id from config: {c['nearest_node_id']})")

    print("\n  Step 1 COMPLETE → proceed to Step 2")