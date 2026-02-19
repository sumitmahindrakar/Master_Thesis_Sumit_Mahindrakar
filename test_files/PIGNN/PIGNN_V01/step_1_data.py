"""
=================================================================
STEP 1: DATA LOADING FROM KRATOS VTK + case_config.json
=================================================================
Tailored to YOUR exact folder structure and requirements.

FOLDER STRUCTURE:
─────────────────
primal/
  case_primary_1/
    vtk_output_primal/Structure_1_.vtk
    case_config.json
  case_primary_2/
    vtk_output_primal/Structure_1_.vtk
    case_config.json
  ...

adjoint/
  case_adjoint_1/
    vtk_output_adjoint/Structure_1_.vtk
  case_adjoint_2/
    vtk_output_adjoint/Structure_1_.vtk
  ...

NODE INPUTS (per node, 8 features):
  coords      (N,3)  → x, y, z
  bc_disp     (N,1)  → 1=displacement fixed, 0=free
  bc_rot      (N,1)  → 1=rotation fixed, 0=free
  line_load   (N,3)  → distributed load (from VTK)

ELEMENT INPUTS (per element, from VTK cell data + config):
  E, A, I22, I33, ν, density, torsional_inertia
  response_flag  → 1 if traced_element_id, 0 otherwise
  length, direction (computed from geometry)

NODE OUTPUTS:
  displacement (N,3)
  rotation     (N,3)

ELEMENT OUTPUTS:
  moment          (E,3)
  force           (E,3)
  I22_sensitivity (E,)

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
    Loads Kratos frame data from your exact folder structure.
    Reads primal VTK, adjoint VTK, and case_config.json.
    """

    def __init__(self,
                 primal_base_dir: str,
                 adjoint_base_dir: str,
                 primal_folder_prefix: str = "case_primary_",
                 adjoint_folder_prefix: str = "case_adjoint_",
                 primal_vtk_subdir: str = "vtk_output_primal",
                 adjoint_vtk_subdir: str = "vtk_output_adjoint",
                 vtk_filename: str = "Structure_1_.vtk"):
        """
        Args:
            primal_base_dir:      "path/to/primal/"
            adjoint_base_dir:     "path/to/adjoint/"
            primal_folder_prefix: "case_primary_"
            adjoint_folder_prefix:"case_adjoint_"
            primal_vtk_subdir:    "vtk_output_primal"
            adjoint_vtk_subdir:   "vtk_output_adjoint"
            vtk_filename:         "Structure_1_.vtk"
        """
        self.primal_base = Path(primal_base_dir)
        self.adjoint_base = Path(adjoint_base_dir)
        self.p_prefix = primal_folder_prefix
        self.a_prefix = adjoint_folder_prefix
        self.p_vtk_sub = primal_vtk_subdir
        self.a_vtk_sub = adjoint_vtk_subdir
        self.vtk_file = vtk_filename

    # ─────────────────────────────────────────────
    # PATH BUILDERS
    # ─────────────────────────────────────────────

    def _primal_vtk_path(self, case_num: int) -> Path:
        """primal/case_primary_1/vtk_output_primal/Structure_1_.vtk"""
        return (self.primal_base /
                f"{self.p_prefix}{case_num}" /
                self.p_vtk_sub /
                self.vtk_file)

    def _adjoint_vtk_path(self, case_num: int) -> Path:
        """adjoint/case_adjoint_1/vtk_output_adjoint/Structure_1_.vtk"""
        return (self.adjoint_base /
                f"{self.a_prefix}{case_num}" /
                self.a_vtk_sub /
                self.vtk_file)

    def _config_path(self, case_num: int) -> Path:
        """primal/case_primary_1/case_config.json"""
        return (self.primal_base /
                f"{self.p_prefix}{case_num}" /
                "case_config.json")

    # ─────────────────────────────────────────────
    # DISCOVER AVAILABLE CASES
    # ─────────────────────────────────────────────

    def discover_cases(self) -> List[int]:
        """
        Find all case numbers by scanning primal directory.
        Returns sorted list of case numbers.
        """
        case_nums = []
        for folder in self.primal_base.iterdir():
            if folder.is_dir() and folder.name.startswith(self.p_prefix):
                try:
                    num = int(folder.name[len(self.p_prefix):])
                    # Verify VTK and config exist
                    vtk_path = self._primal_vtk_path(num)
                    cfg_path = self._config_path(num)
                    if vtk_path.exists() and cfg_path.exists():
                        case_nums.append(num)
                except ValueError:
                    continue

        case_nums.sort()
        print(f"Found {len(case_nums)} cases: "
              f"[{case_nums[0]}...{case_nums[-1]}]")
        return case_nums

    # ─────────────────────────────────────────────
    # READ case_config.json
    # ─────────────────────────────────────────────

    def _read_config(self, case_num: int) -> dict:
        """
        Read case_config.json.

        Expected fields:
            traced_element_id:       int  → response element
            stress_location:         int  → response location
            nearest_node_id:         int
            nearest_node_distance:   float
            + any other parameters you have (UDL, E, I22, etc.)
        """
        cfg_path = self._config_path(case_num)
        with open(cfg_path, 'r') as f:
            config = json.load(f)
        return config

    # ─────────────────────────────────────────────
    # EXTRACT VTK DATA
    # ─────────────────────────────────────────────

    def _read_vtk(self, filepath: Path) -> pv.DataSet:
        """Read VTK file."""
        if not filepath.exists():
            raise FileNotFoundError(f"Not found: {filepath}")
        return pv.read(str(filepath))

    def _get_point_field(self, mesh, name: str) -> Optional[np.ndarray]:
        """Extract point data field."""
        if name in mesh.point_data:
            return np.array(mesh.point_data[name], dtype=np.float64)
        return None

    def _get_cell_field(self, mesh, name: str) -> Optional[np.ndarray]:
        """Extract cell data field."""
        if name in mesh.cell_data:
            arr = np.array(mesh.cell_data[name], dtype=np.float64)
            if arr.ndim > 1 and arr.shape[1] == 1:
                arr = arr.flatten()
            return arr
        return None

    def _extract_connectivity(self, mesh) -> np.ndarray:
        """Extract (E, 2) connectivity for beam elements."""
        conn = []
        for i in range(mesh.n_cells):
            cell = mesh.get_cell(i)
            conn.append(list(cell.point_ids))
        return np.array(conn, dtype=np.int64)

    # ─────────────────────────────────────────────
    # BUILD SIMPLIFIED BC FLAGS
    # ─────────────────────────────────────────────

    def _build_bc_flags(self, coords: np.ndarray,
                         displacement: np.ndarray,
                         rotation: np.ndarray) -> Tuple[np.ndarray,
                                                         np.ndarray]:
        """
        Build simplified BC flags.

        For YOUR fixed-base frame:
          Nodes 0, 7 (z=0) → bc_disp=1, bc_rot=1
          All others        → bc_disp=0, bc_rot=0

        Returns:
            bc_disp: (N, 1)  1=all translations fixed, 0=free
            bc_rot:  (N, 1)  1=all rotations fixed, 0=free
        """
        N = len(coords)
        bc_disp = np.zeros((N, 1), dtype=np.float64)
        bc_rot = np.zeros((N, 1), dtype=np.float64)

        # Detect support nodes from zero displacement
        disp_mag = np.linalg.norm(displacement, axis=1)
        max_disp = np.max(disp_mag)
        threshold = max_disp * 1e-8 if max_disp > 0 else 1e-15

        support_nodes = np.where(disp_mag < threshold)[0]

        # Set flags
        bc_disp[support_nodes] = 1.0
        bc_rot[support_nodes] = 1.0  # fixed supports → rotation also fixed

        return bc_disp, bc_rot

    # ─────────────────────────────────────────────
    # BUILD RESPONSE FLAG
    # ─────────────────────────────────────────────

    def _build_response_flag(self, n_elements: int,
                              traced_element_id: int) -> np.ndarray:
        """
        Build response flag array.

        Args:
            n_elements:        total number of elements
            traced_element_id: from case_config.json

        Returns:
            (E,) array: 1.0 at traced element, 0.0 elsewhere
        """
        resp_flag = np.zeros(n_elements, dtype=np.float64)
        resp_flag[traced_element_id] = 1.0
        return resp_flag

    # ─────────────────────────────────────────────
    # COMPUTE ELEMENT GEOMETRY
    # ─────────────────────────────────────────────

    def _compute_element_geometry(self, coords: np.ndarray,
                                   connectivity: np.ndarray) -> dict:
        """
        Compute element lengths and direction vectors.

        Returns dict with:
            lengths:    (E,)   element lengths
            directions: (E,3)  unit direction vectors
        """
        n1 = coords[connectivity[:, 0]]
        n2 = coords[connectivity[:, 1]]
        diff = n2 - n1
        lengths = np.linalg.norm(diff, axis=1)
        safe_L = np.where(lengths > 1e-15, lengths, 1.0)
        directions = diff / safe_L[:, np.newaxis]
        return {'lengths': lengths, 'directions': directions}

    # ─────────────────────────────────────────────
    # LOAD ONE CASE
    # ─────────────────────────────────────────────

    def load_case(self, case_num: int) -> dict:
        """
        Load one complete case (primal + adjoint + config).

        Returns dict with:
        ─── INPUTS ───
          coords          (N, 3)    node coordinates
          connectivity    (E, 2)    element connectivity
          bc_disp         (N, 1)    displacement BC flag
          bc_rot          (N, 1)    rotation BC flag
          line_load       (N, 3)    distributed load
          young_modulus   (E,)      from VTK cell data
          cross_area      (E,)      from VTK cell data
          I22             (E,)      from VTK cell data
          I33             (E,)      from VTK cell data
          poisson_ratio   (E,)      from VTK cell data
          density         (E,)      from VTK cell data
          torsional_inertia (E,)    from VTK cell data
          response_flag   (E,)      1 at traced element, 0 elsewhere
          elem_lengths    (E,)      computed from geometry
          elem_directions (E, 3)    computed from geometry

        ─── OUTPUTS ───
          displacement    (N, 3)    from primal VTK
          rotation        (N, 3)    from primal VTK
          moment          (E, 3)    from primal VTK
          force           (E, 3)    from primal VTK
          I22_sensitivity (E,)      from adjoint VTK

        ─── METADATA ───
          case_num, traced_element_id, config (full json)
        """

        print(f"\n  Case {case_num}:")

        # ══════════════════════════════════════════
        # 1. READ case_config.json
        # ══════════════════════════════════════════
        config = self._read_config(case_num)
        traced_elem = config['traced_element_id']
        print(f"    Config: traced_element={traced_elem}, "
              f"stress_loc={config.get('stress_location', '?')}")

        # ══════════════════════════════════════════
        # 2. READ PRIMAL VTK
        # ══════════════════════════════════════════
        primal_path = self._primal_vtk_path(case_num)
        primal = self._read_vtk(primal_path)

        coords = np.array(primal.points, dtype=np.float64)
        connectivity = self._extract_connectivity(primal)
        N = len(coords)
        E = len(connectivity)

        print(f"    Primal: {N} nodes, {E} elements")

        # ── INPUT: Line load (point data) ──
        line_load = self._get_point_field(primal, 'LINE_LOAD')
        if line_load is None:
            print(f"    ⚠ LINE_LOAD not found, using zeros")
            line_load = np.zeros((N, 3))

        # ── INPUT: Element properties (cell data) ──
        prop_names = {
            'young_modulus':     'YOUNG_MODULUS',
            'cross_area':        'CROSS_AREA',
            'I22':               'I22',
            'I33':               'I33',
            'poisson_ratio':     'POISSON_RATIO',
            'density':           'DENSITY',
            'torsional_inertia': 'TORSIONAL_INERTIA',
        }
        props = {}
        for key, vtk_name in prop_names.items():
            arr = self._get_cell_field(primal, vtk_name)
            if arr is not None:
                props[key] = arr
            else:
                print(f"    ⚠ {vtk_name} not found")
                props[key] = np.ones(E)

        # ── OUTPUT: Displacement, Rotation (point data) ──
        displacement = self._get_point_field(primal, 'DISPLACEMENT')
        rotation = self._get_point_field(primal, 'ROTATION')

        # ── OUTPUT: Moment, Force (cell data) ──
        moment = self._get_cell_field(primal, 'MOMENT')
        force = self._get_cell_field(primal, 'FORCE')

        # ══════════════════════════════════════════
        # 3. READ ADJOINT VTK
        # ══════════════════════════════════════════
        adjoint_path = self._adjoint_vtk_path(case_num)
        I22_sensitivity = None

        if adjoint_path.exists():
            adjoint = self._read_vtk(adjoint_path)
            I22_sensitivity = self._get_cell_field(
                adjoint, 'I22_SENSITIVITY')
            if I22_sensitivity is not None:
                print(f"    Adjoint: I22_SENSITIVITY loaded, "
                      f"range=[{I22_sensitivity.min():.4e}, "
                      f"{I22_sensitivity.max():.4e}]")
            else:
                print(f"    ⚠ I22_SENSITIVITY not found in adjoint VTK")
        else:
            print(f"    ⚠ Adjoint VTK not found: {adjoint_path}")

        # ══════════════════════════════════════════
        # 4. BUILD DERIVED INPUTS
        # ══════════════════════════════════════════

        # Simplified BC flags
        bc_disp, bc_rot = self._build_bc_flags(
            coords, displacement, rotation)

        # Response flag from config
        response_flag = self._build_response_flag(E, traced_elem)

        # Element geometry
        geom = self._compute_element_geometry(coords, connectivity)

        # ══════════════════════════════════════════
        # 5. ASSEMBLE CASE DICT
        # ══════════════════════════════════════════
        case = {
            # ── INPUTS ──
            'coords':            coords,                # (N, 3)
            'connectivity':      connectivity,          # (E, 2)
            'bc_disp':           bc_disp,               # (N, 1)
            'bc_rot':            bc_rot,                # (N, 1)
            'line_load':         line_load,             # (N, 3)
            'young_modulus':     props['young_modulus'],     # (E,)
            'cross_area':        props['cross_area'],        # (E,)
            'I22':               props['I22'],               # (E,)
            'I33':               props['I33'],               # (E,)
            'poisson_ratio':     props['poisson_ratio'],     # (E,)
            'density':           props['density'],           # (E,)
            'torsional_inertia': props['torsional_inertia'], # (E,)
            'response_flag':     response_flag,         # (E,)
            'elem_lengths':      geom['lengths'],       # (E,)
            'elem_directions':   geom['directions'],    # (E, 3)

            # ── OUTPUTS ──
            'displacement':      displacement,          # (N, 3)
            'rotation':          rotation,              # (N, 3)
            'moment':            moment,                # (E, 3)
            'force':             force,                 # (E, 3)
            'I22_sensitivity':   I22_sensitivity,       # (E,)

            # ── METADATA ──
            'n_nodes':           N,
            'n_elements':        E,
            'case_num':          case_num,
            'traced_element_id': traced_elem,
            'config':            config,
        }

        # Print summary
        support_nodes = np.where(bc_disp.flatten() > 0.5)[0]
        print(f"    Supports: nodes {support_nodes.tolist()}")
        print(f"    UDL (z): {line_load[:, 2].min():.1f}")
        print(f"    E={props['young_modulus'][0]:.2e}, "
              f"I22={props['I22'][0]:.2e}")
        print(f"    Response element: {traced_elem} "
              f"(flag sum={response_flag.sum():.0f})")
        if displacement is not None:
            print(f"    Max |disp|: {np.max(np.abs(displacement)):.4e}")

        return case

    # ─────────────────────────────────────────────
    # LOAD ALL CASES
    # ─────────────────────────────────────────────

    def load_all(self) -> List[dict]:
        """
        Discover and load all available cases.

        Returns:
            List of case dicts, sorted by case number
        """
        case_nums = self.discover_cases()

        print(f"\n{'═'*60}")
        print(f"  Loading {len(case_nums)} cases")
        print(f"{'═'*60}")

        dataset = []
        failed = []
        for num in case_nums:
            try:
                case = self.load_case(num)
                dataset.append(case)
            except Exception as e:
                print(f"    ✗ Case {num} FAILED: {e}")
                failed.append(num)

        print(f"\n{'═'*60}")
        print(f"  LOADED: {len(dataset)} cases")
        if failed:
            print(f"  FAILED: {len(failed)} cases: {failed}")
        print(f"{'═'*60}")

        self._print_dataset_summary(dataset)
        return dataset

    def load_range(self, start: int, end: int) -> List[dict]:
        """Load cases from start to end (inclusive)."""
        dataset = []
        for num in range(start, end + 1):
            try:
                case = self.load_case(num)
                dataset.append(case)
            except Exception as e:
                print(f"    ✗ Case {num}: {e}")

        self._print_dataset_summary(dataset)
        return dataset

    # ─────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────

    def _print_dataset_summary(self, dataset: List[dict]):
        """Print summary of loaded dataset."""
        if not dataset:
            print("  No data loaded!")
            return

        N = dataset[0]['n_nodes']
        E = dataset[0]['n_elements']
        n_cases = len(dataset)

        print(f"\n{'═'*60}")
        print(f"  DATASET SUMMARY")
        print(f"{'═'*60}")
        print(f"  Cases:    {n_cases}")
        print(f"  Nodes:    {N} (per case)")
        print(f"  Elements: {E} (per case)")

        # Collect parameter ranges
        udls = [c['line_load'][:, 2].min() for c in dataset]
        Es = [c['young_modulus'][0] for c in dataset]
        I22s = [c['I22'][0] for c in dataset]
        resps = [c['traced_element_id'] for c in dataset]

        print(f"\n  Parameter ranges:")
        print(f"    UDL (z):    [{min(udls):.1f}, {max(udls):.1f}]")
        print(f"    E:          [{min(Es):.4e}, {max(Es):.4e}]")
        print(f"    I22:        [{min(I22s):.4e}, {max(I22s):.4e}]")
        print(f"    Response elements: {sorted(set(resps))}")

        # Output ranges
        if dataset[0]['displacement'] is not None:
            all_disp = np.concatenate(
                [c['displacement'] for c in dataset])
            print(f"\n  Output ranges:")
            print(f"    Displacement: [{all_disp.min():.4e}, "
                  f"{all_disp.max():.4e}]")

        if dataset[0]['I22_sensitivity'] is not None:
            all_sens = np.concatenate(
                [c['I22_sensitivity'] for c in dataset])
            print(f"    I22 Sensitivity: [{all_sens.min():.4e}, "
                  f"{all_sens.max():.4e}]")

        # Node input dimensions
        print(f"\n  NODE INPUT FEATURES (per node): 8")
        print(f"    coords     (3)  x, y, z")
        print(f"    bc_disp    (1)  displacement constraint")
        print(f"    bc_rot     (1)  rotation constraint")
        print(f"    line_load  (3)  distributed load")

        # Element input dimensions
        print(f"\n  ELEMENT INPUT FEATURES (per element): 13")
        print(f"    E, A, I22, I33, ν, ρ, J    (7) from VTK")
        print(f"    response_flag              (1) from config")
        print(f"    length                     (1) computed")
        print(f"    direction                  (3) computed")
        print(f"    TOTAL edge features:       12  (dir has 3)")

        print(f"\n  OUTPUTS:")
        print(f"    displacement  (N, 3)  node target")
        print(f"    rotation      (N, 3)  node target")
        print(f"    moment        (E, 3)  element target")
        print(f"    force         (E, 3)  element target")
        print(f"    I22_sens      (E,)    element target")
        print(f"{'═'*60}\n")

    # ─────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────

    @staticmethod
    def save(dataset: list, filepath: str):
        """Save dataset to pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        mb = os.path.getsize(filepath) / 1e6
        print(f"Saved: {filepath} ({mb:.1f} MB, {len(dataset)} cases)")

    @staticmethod
    def load(filepath: str) -> list:
        """Load dataset from pickle."""
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Loaded: {filepath} ({len(dataset)} cases)")
        return dataset


# ================================================================
# USAGE
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  STEP 1: Load Kratos Data")
    print("=" * 60)

    # ── Point to YOUR directories ──
    loader = FrameDataLoader(
        primal_base_dir="test_files/Kratos_data_creation/primal",
        adjoint_base_dir="test_files/Kratos_data_creation/adjoint",
        primal_folder_prefix="case_primary_",
        adjoint_folder_prefix="case_adjoint_",
        primal_vtk_subdir="vtk_output_primal",
        adjoint_vtk_subdir="vtk_output_adjoint",
        vtk_filename="Structure_0_1.vtk"
    )

    # ── Option A: Load all cases ──
    dataset = loader.load_all()

    # ── Option B: Load specific range ──
    # dataset = loader.load_range(1, 50)

    # ── Option C: Load single case ──
    # case = loader.load_case(1)

    # ── Save for Step 2 ──
    FrameDataLoader.save(dataset, "test_files/PIGNN/PIGNN_V01/frame_dataset.pkl")

    # ── Verify one case ──
    if dataset:
        c = dataset[0]
        print(f"\nCase {c['case_num']} verification:")
        print(f"  coords:         {c['coords'].shape}")
        print(f"  bc_disp:        {c['bc_disp'].shape}  "
              f"sum={c['bc_disp'].sum():.0f}")
        print(f"  bc_rot:         {c['bc_rot'].shape}  "
              f"sum={c['bc_rot'].sum():.0f}")
        print(f"  line_load:      {c['line_load'].shape}")
        print(f"  response_flag:  {c['response_flag'].shape}  "
              f"elem={np.where(c['response_flag']>0.5)[0]}")
        print(f"  displacement:   {c['displacement'].shape}")
        print(f"  I22_sensitivity:{c['I22_sensitivity'].shape}")

    print("\n  Step 1 COMPLETE → proceed to Step 2")