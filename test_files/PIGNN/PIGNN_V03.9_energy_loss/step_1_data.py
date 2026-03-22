# """
# =================================================================
# STEP 1: DATA LOADING — Per-Node Loads + Moments, NODE-level response
# =================================================================

# Node inputs (10 per node):
#   coords           (3)  x, y, z
#   bc_disp          (1)  1=translations fixed
#   bc_rot           (1)  1=rotations fixed
#   point_load       (3)  Fx, Fy, Fz per node (VARYING)
#   point_moment_My  (1)  My per node (VARYING, 0 if disabled)
#   response_node    (1)  1=response measured here, 0=not

# Element inputs (11 per element):
#   length           (1)
#   direction        (3)
#   E, A, I22, I33   (4)
#   ν, density, J    (3)

# =================================================================
# """

# import numpy as np
# import json
# import os
# import pickle
# from pathlib import Path
# from typing import Optional, List, Tuple

# try:
#     import pyvista as pv
#     print(f"[OK] pyvista {pv.__version__}")
# except ImportError:
#     raise ImportError("pip install pyvista")


# class FrameDataLoader:
#     """
#     Loads Kratos frame data with per-node varying loads and moments.
#     Supports POINT_LOAD (Fx, Fz) and POINT_MOMENT (My) at each node.
#     Response location tracked at NODE level (nearest_node_id).
#     """

#     def __init__(self,
#                  primal_base_dir: str,
#                  adjoint_base_dir: str,
#                  primal_folder_prefix: str = "case_primal_",
#                  adjoint_folder_prefix: str = "case_adjoint_",
#                  primal_vtk_subdir: str = "vtk_output_primal",
#                  adjoint_vtk_subdir: str = "vtk_output_adjoint",
#                  vtk_filename: str = "Structure_1_.vtk"):

#         self.primal_base = Path(primal_base_dir)
#         self.adjoint_base = Path(adjoint_base_dir)
#         self.p_prefix = primal_folder_prefix
#         self.a_prefix = adjoint_folder_prefix
#         self.p_vtk_sub = primal_vtk_subdir
#         self.a_vtk_sub = adjoint_vtk_subdir
#         self.vtk_file = vtk_filename

#     # ─── PATHS ───

#     def _primal_vtk_path(self, case_num: int) -> Path:
#         return (self.primal_base /
#                 f"{self.p_prefix}{case_num}" /
#                 self.p_vtk_sub / self.vtk_file)

#     def _adjoint_vtk_path(self, case_num: int) -> Path:
#         return (self.adjoint_base /
#                 f"{self.a_prefix}{case_num}" /
#                 self.a_vtk_sub / self.vtk_file)

#     def _config_path(self, case_num: int) -> Path:
#         return (self.primal_base /
#                 f"{self.p_prefix}{case_num}" /
#                 "case_config.json")

#     # ─── DISCOVER ───

#     def discover_cases(self) -> List[int]:
#         case_nums = []
#         for folder in self.primal_base.iterdir():
#             if folder.is_dir() and folder.name.startswith(self.p_prefix):
#                 try:
#                     num = int(folder.name[len(self.p_prefix):])
#                     if (self._primal_vtk_path(num).exists() and
#                             self._config_path(num).exists()):
#                         case_nums.append(num)
#                 except ValueError:
#                     continue
#         case_nums.sort()
#         if case_nums:
#             print(f"Found {len(case_nums)} cases: "
#                   f"[{case_nums[0]}...{case_nums[-1]}]")
#         else:
#             print("No cases found!")
#         return case_nums

#     # ─── CONFIG ───

#     def _read_config(self, case_num: int) -> dict:
#         with open(self._config_path(case_num), 'r') as f:
#             raw = json.load(f)
#         config = {}
#         config['case_id'] = raw.get('case_id', case_num)
#         config['timestamp'] = raw.get('timestamp', '')
#         params = raw.get('parameters', {})
#         config.update(params)

#         # Per-node load info
#         config['node_loads'] = params.get('node_loads', {})
#         config['load_summary'] = raw.get('load_summary', {})

#         # Merge info (if merged dataset)
#         config['merge_info'] = raw.get('merge_info', {})

#         return config

#     # ─── VTK HELPERS ───

#     def _read_vtk(self, filepath: Path) -> pv.DataSet:
#         if not filepath.exists():
#             raise FileNotFoundError(f"Not found: {filepath}")
#         return pv.read(str(filepath))

#     def _get_point_field(self, mesh, name: str) -> Optional[np.ndarray]:
#         if name in mesh.point_data:
#             return np.array(mesh.point_data[name], dtype=np.float64)
#         return None

#     def _get_cell_field(self, mesh, name: str) -> Optional[np.ndarray]:
#         if name in mesh.cell_data:
#             arr = np.array(mesh.cell_data[name], dtype=np.float64)
#             if arr.ndim > 1 and arr.shape[1] == 1:
#                 arr = arr.flatten()
#             return arr
#         return None

#     def _extract_connectivity(self, mesh) -> np.ndarray:
#         conn = []
#         for i in range(mesh.n_cells):
#             conn.append(list(mesh.get_cell(i).point_ids))
#         return np.array(conn, dtype=np.int64)

#     # ─── BC FLAGS ───

#     def _build_bc_flags(self, coords, displacement, rotation):
#         N = len(coords)
#         bc_disp = np.zeros((N, 1), dtype=np.float64)
#         bc_rot = np.zeros((N, 1), dtype=np.float64)

#         if displacement is not None:
#             disp_mag = np.linalg.norm(displacement, axis=1)
#             max_disp = np.max(disp_mag)
#             threshold = max_disp * 1e-8 if max_disp > 0 else 1e-15
#             support_nodes = np.where(disp_mag < threshold)[0]
#             bc_disp[support_nodes] = 1.0
#             if rotation is not None:
#                 rot_mag = np.linalg.norm(rotation, axis=1)
#                 max_rot = np.max(rot_mag)
#                 rot_thresh = max_rot * 1e-8 if max_rot > 0 else 1e-15
#                 for node in support_nodes:
#                     if rot_mag[node] < rot_thresh:
#                         bc_rot[node] = 1.0

#         return bc_disp, bc_rot

#     # ─── RESPONSE NODE FLAG ───

#     def _build_response_node_flag(self, n_nodes: int,
#                                     nearest_node_id: int
#                                     ) -> np.ndarray:
#         resp_flag = np.zeros((n_nodes, 1), dtype=np.float64)
#         if 0 <= nearest_node_id < n_nodes:
#             resp_flag[nearest_node_id] = 1.0
#         return resp_flag

#     # ─── ELEMENT GEOMETRY ───

#     def _compute_element_geometry(self, coords, connectivity):
#         n1 = coords[connectivity[:, 0]]
#         n2 = coords[connectivity[:, 1]]
#         diff = n2 - n1
#         lengths = np.linalg.norm(diff, axis=1)
#         safe_L = np.where(lengths > 1e-15, lengths, 1.0)
#         directions = diff / safe_L[:, np.newaxis]
#         return {'lengths': lengths, 'directions': directions}

#     # ─── ELEMENT LOAD (from nodal POINT_LOAD) ───

#     def _compute_element_load(self, point_load_nodes, connectivity):
#         """
#         Convert node-level POINT_LOAD to element-level load.
#         Average of both end nodes' point loads.
#         """
#         n1_load = point_load_nodes[connectivity[:, 0]]
#         n2_load = point_load_nodes[connectivity[:, 1]]
#         elem_load = 0.5 * (n1_load + n2_load)
#         return elem_load

#     # ─── ELEMENT MOMENT (from nodal POINT_MOMENT My) ───

#     def _compute_element_moment(self, point_moment_My, connectivity):
#         """
#         Convert node-level My to element-level moment.
#         Average of both end nodes' My values.
#         """
#         n1_mom = point_moment_My[connectivity[:, 0]]  # (E, 1)
#         n2_mom = point_moment_My[connectivity[:, 1]]  # (E, 1)
#         elem_moment = 0.5 * (n1_mom + n2_mom)
#         return elem_moment

#     # ─── FIX: ZERO OUT COLUMN LOADS ───

#     @staticmethod
#     def fix_elem_load(elem_load, elem_directions):
#         """
#         Zero out elem_load for column (vertical) elements.
#         For vertical elements (|dz| > |dx|): set load to zero.
#         """
#         elem_load_fixed = elem_load.copy()

#         for e in range(len(elem_directions)):
#             dx = abs(elem_directions[e, 0])
#             dz = abs(elem_directions[e, 2])

#             if dz > dx:
#                 elem_load_fixed[e] = [0.0, 0.0, 0.0]

#         n_zeroed = np.sum(np.any(elem_load != elem_load_fixed, axis=1))
#         print(f"    Fixed elem_load: zeroed {n_zeroed} column elements")

#         return elem_load_fixed

#     @staticmethod
#     def fix_elem_moment(elem_moment, elem_directions):
#         """
#         Zero out elem_moment for column (vertical) elements.
#         Same logic as fix_elem_load.
#         """
#         elem_moment_fixed = elem_moment.copy()

#         for e in range(len(elem_directions)):
#             dx = abs(elem_directions[e, 0])
#             dz = abs(elem_directions[e, 2])

#             if dz > dx:
#                 elem_moment_fixed[e] = 0.0

#         n_zeroed = np.sum(elem_moment.flatten() != elem_moment_fixed.flatten())
#         if n_zeroed > 0:
#             print(f"    Fixed elem_moment: zeroed {n_zeroed} column elements")

#         return elem_moment_fixed

#     # ─── PER-NODE LOAD SUMMARY ───

#     @staticmethod
#     def _summarize_node_loads(point_load, point_moment_My, config):
#         """Print per-node load and moment statistics."""
#         load_mags = np.linalg.norm(point_load, axis=1)
#         n_loaded = np.sum(load_mags > 1e-10)
#         N = len(point_load)

#         moment_mags = np.abs(point_moment_My.flatten())
#         n_moment_loaded = np.sum(moment_mags > 1e-10)

#         print(f"    Per-node loads: {n_loaded}/{N} nodes loaded")

#         if n_loaded > 0:
#             loaded_mask = load_mags > 1e-10
#             Fx_vals = point_load[loaded_mask, 0]
#             Fz_vals = point_load[loaded_mask, 2]

#             print(f"      Fx range: [{Fx_vals.min():.2f}, "
#                   f"{Fx_vals.max():.2f}]")
#             print(f"      Fz range: [{Fz_vals.min():.2f}, "
#                   f"{Fz_vals.max():.2f}]")
#             print(f"      Max |F|:  {load_mags.max():.2f}")

#         if n_moment_loaded > 0:
#             My_vals = point_moment_My.flatten()[moment_mags > 1e-10]
#             print(f"    Per-node moments: {n_moment_loaded}/{N} nodes")
#             print(f"      My range: [{My_vals.min():.2f}, "
#                   f"{My_vals.max():.2f}]")
#         else:
#             print(f"    Per-node moments: 0/{N} (disabled)")

#         # Cross-check with config
#         cfg_summary = config.get('load_summary', {})
#         if cfg_summary:
#             cfg_loaded = cfg_summary.get('loaded_nodes', '?')
#             print(f"      Config says: {cfg_loaded} nodes loaded")

#     # ─── LOAD ONE CASE ───

#     def load_case(self, case_num: int) -> dict:
#         print(f"\n  Case {case_num}:")

#         # ═══ 1. CONFIG ═══
#         config = self._read_config(case_num)
#         nearest_node = config['nearest_node_id']
#         traced_elem = config['traced_element_id']

#         # Print source info if merged
#         merge_info = config.get('merge_info', {})
#         if merge_info:
#             print(f"    Source: {merge_info.get('source_folder', '?')} "
#                   f"(orig ID: {merge_info.get('original_case_id', '?')})")

#         print(f"    Response: node={nearest_node} "
#               f"(traced_elem={traced_elem})")
#         print(f"    Params: E={config.get('youngs_modulus', '?'):.2e}, "
#               f"I22={config.get('I22', '?'):.4e}")

#         # ═══ 2. PRIMAL VTK ═══
#         primal = self._read_vtk(self._primal_vtk_path(case_num))
#         coords = np.array(primal.points, dtype=np.float64)
#         connectivity = self._extract_connectivity(primal)
#         N = len(coords)
#         E = len(connectivity)

#         # INPUT: Point load (per node — Fx, Fy, Fz VARYING)
#         point_load = self._get_point_field(primal, 'POINT_LOAD')
#         if point_load is None:
#             point_load = np.zeros((N, 3))
#             print(f"    ⚠ POINT_LOAD not found in VTK")

#         # INPUT: Point moment — extract My only (per node)
#         point_moment_raw = self._get_point_field(primal, 'POINT_MOMENT')
#         if point_moment_raw is not None:
#             point_moment_My = point_moment_raw[:, 1:2]  # (N, 1)
#         else:
#             point_moment_My = np.zeros((N, 1))

#         # Print per-node load + moment summary
#         self._summarize_node_loads(point_load, point_moment_My, config)

#         # INPUT: Element properties
#         prop_map = {
#             'young_modulus':     'YOUNG_MODULUS',
#             'cross_area':        'CROSS_AREA',
#             'I22':               'I22',
#             'I33':               'I33',
#             'poisson_ratio':     'POISSON_RATIO',
#             'density':           'DENSITY',
#             'torsional_inertia': 'TORSIONAL_INERTIA',
#         }
#         props = {}
#         for key, vtk_name in prop_map.items():
#             arr = self._get_cell_field(primal, vtk_name)
#             props[key] = arr if arr is not None else np.ones(E)

#         # OUTPUT: Displacement, Rotation
#         displacement = self._get_point_field(primal, 'DISPLACEMENT')
#         rotation = self._get_point_field(primal, 'ROTATION')

#         # ═══ EXTRACT 2D FRAME DOFs ═══
#         if displacement is not None and rotation is not None:
#             nodal_disp_2d = np.stack([
#                 displacement[:, 0],   # u_x
#                 displacement[:, 2],   # u_z
#                 rotation[:, 1]        # phi_y
#             ], axis=1)
#         else:
#             nodal_disp_2d = None

#         # OUTPUT: Moment, Force (internal)
#         moment = self._get_cell_field(primal, 'MOMENT')
#         force = self._get_cell_field(primal, 'FORCE')

#         # ═══ EXTRACT 2D INTERNAL FORCES ═══
#         if force is not None:
#             elem_N = force[:, 0]
#             elem_V = force[:, 2]
#         else:
#             elem_N = None
#             elem_V = None

#         if moment is not None:
#             elem_M = moment[:, 1]
#         else:
#             elem_M = None

#         # ═══ 3. ADJOINT VTK ═══
#         adjoint_path = self._adjoint_vtk_path(case_num)
#         I22_sensitivity = None
#         if adjoint_path.exists():
#             adjoint = self._read_vtk(adjoint_path)
#             I22_sensitivity = self._get_cell_field(
#                 adjoint, 'I22_SENSITIVITY')
#             if I22_sensitivity is not None:
#                 print(f"    Sensitivity: [{I22_sensitivity.min():.4e}, "
#                       f"{I22_sensitivity.max():.4e}]")
#         else:
#             print(f"    ⚠ Adjoint not found")

#         # ═══ 4. DERIVED INPUTS ═══
#         bc_disp, bc_rot = self._build_bc_flags(
#             coords, displacement, rotation)
#         response_node_flag = self._build_response_node_flag(
#             N, nearest_node)

#         geom = self._compute_element_geometry(coords, connectivity)

#         # Element-level load (from per-node POINT_LOAD)
#         elem_load = self._compute_element_load(point_load, connectivity)
#         elem_load = self.fix_elem_load(elem_load, geom['directions'])

#         # Element-level moment (from per-node My)
#         elem_applied_moment = self._compute_element_moment(
#             point_moment_My, connectivity
#         )
#         elem_applied_moment = self.fix_elem_moment(
#             elem_applied_moment, geom['directions']
#         )

#         loaded_nodes = np.where(
#             np.linalg.norm(point_load, axis=1) > 1e-10
#         )[0]
#         loaded_elems = np.where(
#             np.linalg.norm(elem_load, axis=1) > 1e-10
#         )[0]
#         moment_loaded_nodes = np.where(
#             np.abs(point_moment_My.flatten()) > 1e-10
#         )[0]
#         print(f"    Loaded nodes (force): {len(loaded_nodes)}/{N}")
#         print(f"    Loaded nodes (moment): {len(moment_loaded_nodes)}/{N}")
#         print(f"    Loaded elements: {len(loaded_elems)}/{E} "
#               f"(beams only)")

#         # ═══ 5. ASSEMBLE ═══
#         case = {
#             # ── INPUTS ──
#             'coords':              coords,              # (N, 3)
#             'connectivity':        connectivity,        # (E, 2)
#             'bc_disp':             bc_disp,             # (N, 1)
#             'bc_rot':              bc_rot,              # (N, 1)
#             'point_load':          point_load,          # (N, 3)
#             'point_moment_My':     point_moment_My,     # (N, 1)
#             'elem_load':           elem_load,           # (E, 3)
#             'elem_applied_moment': elem_applied_moment, # (E, 1)
#             'response_node_flag':  response_node_flag,  # (N, 1)
#             'young_modulus':       props['young_modulus'],
#             'cross_area':          props['cross_area'],
#             'I22':                 props['I22'],
#             'I33':                 props['I33'],
#             'poisson_ratio':       props['poisson_ratio'],
#             'density':             props['density'],
#             'torsional_inertia':   props['torsional_inertia'],
#             'elem_lengths':        geom['lengths'],
#             'elem_directions':     geom['directions'],

#             # ── PRIMARY OUTPUTS ──
#             'displacement':        displacement,        # (N, 3)
#             'rotation':            rotation,            # (N, 3)
#             'nodal_disp_2d':       nodal_disp_2d,       # (N, 3)

#             # ── PHYSICS OUTPUTS ──
#             'moment':              moment,              # (E, 3)
#             'force':               force,               # (E, 3)
#             'elem_N':              elem_N,              # (E,)
#             'elem_V':              elem_V,              # (E,)
#             'elem_M':              elem_M,              # (E,)
#             'I22_sensitivity':     I22_sensitivity,

#             # ── METADATA ──
#             'n_nodes':             N,
#             'n_elements':          E,
#             'case_num':            case_num,
#             'nearest_node_id':     nearest_node,
#             'traced_element_id':   traced_elem,
#             'config':              config,
#         }

#         support_nodes = np.where(bc_disp.flatten() > 0.5)[0]
#         resp_node = np.where(
#             response_node_flag.flatten() > 0.5)[0]
#         print(f"    Supports: {support_nodes.tolist()}")
#         print(f"    Response node: {resp_node.tolist()} ✓")
#         print(f"    ✓ Case {case_num} loaded")

#         return case

#     # ─── LOAD ALL ───

#     def load_all(self) -> List[dict]:
#         case_nums = self.discover_cases()
#         print(f"\n{'═'*60}")
#         print(f"  Loading {len(case_nums)} cases")
#         print(f"{'═'*60}")

#         dataset, failed = [], []
#         for num in case_nums:
#             try:
#                 dataset.append(self.load_case(num))
#             except Exception as e:
#                 print(f"    ✗ Case {num} FAILED: {e}")
#                 failed.append(num)

#         print(f"\n{'═'*60}")
#         print(f"  LOADED: {len(dataset)}  |  FAILED: {len(failed)}")
#         print(f"{'═'*60}")
#         self._print_summary(dataset)
#         return dataset

#     def load_range(self, start: int, end: int) -> List[dict]:
#         dataset = []
#         for num in range(start, end + 1):
#             try:
#                 dataset.append(self.load_case(num))
#             except Exception as e:
#                 print(f"    ✗ Case {num}: {e}")
#         self._print_summary(dataset)
#         return dataset

#     # ─── SUMMARY ───

#     def _print_summary(self, dataset):
#         if not dataset:
#             print("  No data loaded!")
#             return

#         n = len(dataset)
#         N = dataset[0]['n_nodes']
#         E = dataset[0]['n_elements']

#         print(f"\n{'═'*60}")
#         print(f"  DATASET SUMMARY")
#         print(f"{'═'*60}")
#         print(f"  Cases: {n}  |  Nodes: {N}  |  Elements: {E}")

#         # ── Per-node load statistics across ALL cases ──
#         all_Fx = []
#         all_Fz = []
#         all_My = []
#         all_loaded_counts = []
#         all_moment_counts = []

#         for c in dataset:
#             pl = c['point_load']
#             mags = np.linalg.norm(pl, axis=1)
#             loaded = mags > 1e-10
#             all_loaded_counts.append(np.sum(loaded))
#             if np.any(loaded):
#                 all_Fx.extend(pl[loaded, 0].tolist())
#                 all_Fz.extend(pl[loaded, 2].tolist())

#             my = c['point_moment_My'].flatten()
#             my_loaded = np.abs(my) > 1e-10
#             all_moment_counts.append(np.sum(my_loaded))
#             if np.any(my_loaded):
#                 all_My.extend(my[my_loaded].tolist())

#         all_Fx = np.array(all_Fx) if all_Fx else np.array([0.0])
#         all_Fz = np.array(all_Fz) if all_Fz else np.array([0.0])
#         all_My = np.array(all_My) if all_My else np.array([0.0])

#         print(f"\n  Per-node load statistics (across {n} cases):")
#         print(f"    Fx range:       [{all_Fx.min():.2f}, "
#               f"{all_Fx.max():.2f}]")
#         print(f"    Fz range:       [{all_Fz.min():.2f}, "
#               f"{all_Fz.max():.2f}]")
#         print(f"    Loaded nodes:   [{min(all_loaded_counts)}, "
#               f"{max(all_loaded_counts)}] per case")
#         print(f"    Mean loaded:    {np.mean(all_loaded_counts):.1f}")

#         if np.any(np.abs(all_My) > 1e-10):
#             print(f"\n  Per-node moment statistics:")
#             print(f"    My range:       [{all_My.min():.2f}, "
#                   f"{all_My.max():.2f}]")
#             print(f"    Moment nodes:   [{min(all_moment_counts)}, "
#                   f"{max(all_moment_counts)}] per case")
#         else:
#             print(f"\n  Per-node moments: DISABLED (all zeros)")

#         # ── Material parameters ──
#         Es = [c['config'].get('youngs_modulus', 0) for c in dataset]
#         I22s = [c['config'].get('I22', 0) for c in dataset]
#         resp_nodes = [c['nearest_node_id'] for c in dataset]

#         print(f"\n  Material parameters:")
#         print(f"    E:              [{min(Es):.2e}, {max(Es):.2e}]")
#         print(f"    I22:            [{min(I22s):.4e}, {max(I22s):.4e}]")
#         print(f"    Response nodes: {len(set(resp_nodes))} unique")

#         # ── Source datasets (if merged) ──
#         sources = set()
#         for c in dataset:
#             mi = c['config'].get('merge_info', {})
#             if mi:
#                 sources.add(mi.get('source_folder', 'unknown'))
#         if sources:
#             print(f"\n  Merged from: {sorted(sources)}")

#         # ── Verify elem_load fix ──
#         c0 = dataset[0]
#         n_beams = np.sum(
#             np.abs(c0['elem_directions'][:, 0]) >
#             np.abs(c0['elem_directions'][:, 2])
#         )
#         n_columns = E - n_beams
#         n_loaded_elems = np.sum(
#             np.linalg.norm(c0['elem_load'], axis=1) > 1e-10
#         )

#         print(f"\n  elem_load verification (case 0):")
#         print(f"    Beams: {n_beams}, Columns: {n_columns}")
#         print(f"    Loaded elements: {n_loaded_elems}")

#         if n_loaded_elems <= n_beams:
#             print(f"    ✓ No columns carry load")
#         else:
#             print(f"    ✗ WARNING: columns carry load!")

#         # ── Feature dimensions table ──
#         print(f"\n  Feature dimensions:")
#         print(f"  ┌──────────────────────────────────────────────┐")
#         print(f"  │ NODE INPUTS (per node):             10       │")
#         print(f"  │   coords              (3)  x, y, z           │")
#         print(f"  │   bc_disp             (1)  constraint        │")
#         print(f"  │   bc_rot              (1)  constraint        │")
#         print(f"  │   point_load          (3)  Fx, Fy, Fz *VARY* │")
#         print(f"  │   point_moment_My     (1)  My *VARY*         │")
#         print(f"  │   response_node_flag  (1)  response loc      │")
#         print(f"  ├──────────────────────────────────────────────┤")
#         print(f"  │ ELEMENT INPUTS (per element):        11      │")
#         print(f"  │   length              (1)  geometry          │")
#         print(f"  │   direction           (3)  geometry          │")
#         print(f"  │   E, A, I22, I33      (4)  properties       │")
#         print(f"  │   ν, density, J       (3)  properties       │")
#         print(f"  ├──────────────────────────────────────────────┤")
#         print(f"  │ NODE OUTPUTS:                                │")
#         print(f"  │   nodal_disp_2d       (3)  [u_x, u_z, φ_y]  │")
#         print(f"  ├──────────────────────────────────────────────┤")
#         print(f"  │ ELEMENT OUTPUTS:                             │")
#         print(f"  │   elem_N              (1)  axial force       │")
#         print(f"  │   elem_M              (1)  bending moment    │")
#         print(f"  │   elem_V              (1)  shear force       │")
#         print(f"  │   I22_sensitivity     (1)  dBM/dI22          │")
#         print(f"  └──────────────────────────────────────────────┘")
#         print(f"{'═'*60}\n")

#     # ─── SAVE / LOAD ───

#     @staticmethod
#     def save(dataset, filepath):
#         os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
#         with open(filepath, 'wb') as f:
#             pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
#         mb = os.path.getsize(filepath) / 1e6
#         print(f"Saved: {filepath} ({mb:.1f} MB, {len(dataset)} cases)")

#     @staticmethod
#     def load(filepath):
#         with open(filepath, 'rb') as f:
#             dataset = pickle.load(f)
#         print(f"Loaded: {filepath} ({len(dataset)} cases)")
#         return dataset


# # ================================================================
# if __name__ == "__main__":

#     print("=" * 60)
#     print("  STEP 1: Load Kratos Data (Per-Node Loads + Moments)")
#     print("=" * 60)

#     loader = FrameDataLoader(
#         primal_base_dir="test_files/Kratos_data_creation/primal",
#         adjoint_base_dir="test_files/Kratos_data_creation/adjoint",
#         primal_folder_prefix="case_primal_",
#         adjoint_folder_prefix="case_adjoint_",
#         primal_vtk_subdir="vtk_output_primal",
#         adjoint_vtk_subdir="vtk_output_adjoint",
#         vtk_filename="Structure_0_1.vtk"
#     )

#     dataset = loader.load_all()

#     from pathlib import Path
#     print(f"Working directory: {os.getcwd()}")
#     CURRENT_SUBFOLDER = Path(__file__).resolve().parent
#     os.chdir(CURRENT_SUBFOLDER)
#     print(f"Working directory: {os.getcwd()}")

#     FrameDataLoader.save(dataset, "DATA/frame_dataset.pkl")

#     if dataset:
#         c = dataset[0]
#         print(f"\nVerification case {c['case_num']}:")
#         print(f"  response_node_flag: "
#               f"node={np.where(c['response_node_flag'].flatten()>0.5)[0]}")
#         print(f"  (nearest_node_id from config: {c['nearest_node_id']})")

#         # ── Verify per-node load distribution ──
#         loaded_nodes = np.where(
#             np.linalg.norm(c['point_load'], axis=1) > 1e-10
#         )[0]
#         print(f"\n  Force-loaded nodes: {loaded_nodes.tolist()}")

#         for i, nid in enumerate(loaded_nodes[:5]):
#             fx, fy, fz = c['point_load'][nid]
#             print(f"    Node {nid}: Fx={fx:.2f} Fy={fy:.2f} Fz={fz:.2f}")
#         if len(loaded_nodes) > 5:
#             print(f"    ... ({len(loaded_nodes) - 5} more)")

#         # ── Verify moment distribution ──
#         my_loaded = np.where(
#             np.abs(c['point_moment_My'].flatten()) > 1e-10
#         )[0]
#         print(f"\n  Moment-loaded nodes: {len(my_loaded)}")
#         if len(my_loaded) > 0:
#             for nid in my_loaded[:5]:
#                 print(f"    Node {nid}: "
#                       f"My={c['point_moment_My'][nid, 0]:.2f}")
#             if len(my_loaded) > 5:
#                 print(f"    ... ({len(my_loaded) - 5} more)")
#         else:
#             print(f"    (moments disabled — all zeros)")

#         # ── Verify elem_load fix ──
#         loaded_elems = np.where(
#             np.linalg.norm(c['elem_load'], axis=1) > 1e-10
#         )[0]
#         print(f"\n  Loaded elements: {loaded_elems.tolist()}")
#         print(f"  elem_load unique Fx: "
#               f"{np.unique(c['elem_load'][:, 0].round(2))}")
#         print(f"  elem_load unique Fz: "
#               f"{np.unique(c['elem_load'][:, 2].round(2))}")

#         # ── Verify elem_applied_moment ──
#         mom_elems = np.where(
#             np.abs(c['elem_applied_moment'].flatten()) > 1e-10
#         )[0]
#         print(f"  Moment-loaded elements: {len(mom_elems)}")

#     print("\n  Step 1 COMPLETE → proceed to Step 2")

"""
=================================================================
STEP 1: DATA LOADING — Per-Node Loads + Moments
=================================================================

Node-level data loaded from VTK:
  coords           (N, 3)   x, y, z
  point_load       (N, 3)   Fx, Fy, Fz per node (VARYING)
  point_moment_My  (N, 1)   My per node (VARYING or zero)
  displacement     (N, 3)   from primal solution
  rotation         (N, 3)   from primal solution

Derived node data:
  bc_disp          (N, 1)   1=translations fixed
  bc_rot           (N, 1)   1=rotations fixed
  response_node    (N, 1)   1=response measured here
  nodal_disp_2d    (N, 3)   [u_x, u_z, phi_y]

Element-level data:
  elem_lengths     (E,)     from geometry
  elem_directions  (E, 3)   unit vectors
  E, A, I22, ...   (E,)     material properties
  elem_load        (E, 3)   averaged from end nodes (verification)
  elem_applied_mom (E, 1)   averaged from end nodes (verification)

Adjoint:
  I22_sensitivity  (E,)     from adjoint VTK

=================================================================
"""

import numpy as np
import json
import os
import pickle
from pathlib import Path
from typing import Optional, List

try:
    import pyvista as pv
    print(f"[OK] pyvista {pv.__version__}")
except ImportError:
    raise ImportError("pip install pyvista")


class FrameDataLoader:
    """
    Loads Kratos frame data with per-node varying loads and moments.
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

    # ───────────────────────────────────────
    # PATHS
    # ───────────────────────────────────────

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

    # ───────────────────────────────────────
    # DISCOVER CASES
    # ───────────────────────────────────────

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
        if case_nums:
            print(f"Found {len(case_nums)} cases: "
                  f"[{case_nums[0]}...{case_nums[-1]}]")
        else:
            print("No cases found!")
        return case_nums

    # ───────────────────────────────────────
    # CONFIG
    # ───────────────────────────────────────

    def _read_config(self, case_num: int) -> dict:
        with open(self._config_path(case_num), 'r') as f:
            raw = json.load(f)

        config = {}
        config['case_id'] = raw.get('case_id', case_num)
        config['timestamp'] = raw.get('timestamp', '')

        params = raw.get('parameters', {})
        config.update(params)

        config['node_loads'] = params.get('node_loads', {})
        config['load_summary'] = raw.get('load_summary', {})
        config['merge_info'] = raw.get('merge_info', {})

        return config

    # ───────────────────────────────────────
    # VTK HELPERS
    # ───────────────────────────────────────

    def _read_vtk(self, filepath: Path) -> pv.DataSet:
        if not filepath.exists():
            raise FileNotFoundError(f"Not found: {filepath}")
        return pv.read(str(filepath))

    def _get_point_field(self, mesh, name: str
                         ) -> Optional[np.ndarray]:
        if name in mesh.point_data:
            return np.array(mesh.point_data[name],
                           dtype=np.float64)
        return None

    def _get_cell_field(self, mesh, name: str
                        ) -> Optional[np.ndarray]:
        if name in mesh.cell_data:
            arr = np.array(mesh.cell_data[name],
                          dtype=np.float64)
            if arr.ndim > 1 and arr.shape[1] == 1:
                arr = arr.flatten()
            return arr
        return None

    def _extract_connectivity(self, mesh) -> np.ndarray:
        conn = []
        for i in range(mesh.n_cells):
            conn.append(list(mesh.get_cell(i).point_ids))
        return np.array(conn, dtype=np.int64)

    # ───────────────────────────────────────
    # BOUNDARY CONDITION FLAGS
    # ───────────────────────────────────────

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
                rot_thresh = (max_rot * 1e-8
                             if max_rot > 0 else 1e-15)
                for node in support_nodes:
                    if rot_mag[node] < rot_thresh:
                        bc_rot[node] = 1.0

        return bc_disp, bc_rot

    # ───────────────────────────────────────
    # RESPONSE NODE FLAG
    # ───────────────────────────────────────

    def _build_response_node_flag(self, n_nodes: int,
                                    nearest_node_id: int
                                    ) -> np.ndarray:
        resp_flag = np.zeros((n_nodes, 1), dtype=np.float64)
        if 0 <= nearest_node_id < n_nodes:
            resp_flag[nearest_node_id] = 1.0
        return resp_flag

    # ───────────────────────────────────────
    # ELEMENT GEOMETRY
    # ───────────────────────────────────────

    def _compute_element_geometry(self, coords, connectivity):
        n1 = coords[connectivity[:, 0]]
        n2 = coords[connectivity[:, 1]]
        diff = n2 - n1
        lengths = np.linalg.norm(diff, axis=1)
        safe_L = np.where(lengths > 1e-15, lengths, 1.0)
        directions = diff / safe_L[:, np.newaxis]
        return {'lengths': lengths, 'directions': directions}

    # ───────────────────────────────────────
    # ELEMENT LOAD (for verification only)
    # ───────────────────────────────────────

    def _compute_element_load(self, point_load, connectivity):
        """Average of both end nodes' point loads."""
        n1_load = point_load[connectivity[:, 0]]
        n2_load = point_load[connectivity[:, 1]]
        return 0.5 * (n1_load + n2_load)

    def _compute_element_moment(self, point_moment_My,
                                  connectivity):
        """Average of both end nodes' My values."""
        n1_mom = point_moment_My[connectivity[:, 0]]
        n2_mom = point_moment_My[connectivity[:, 1]]
        return 0.5 * (n1_mom + n2_mom)

    # ───────────────────────────────────────
    # FIX: ZERO OUT COLUMN LOADS
    # ───────────────────────────────────────

    @staticmethod
    def fix_elem_load(elem_load, elem_directions):
        """Zero out elem_load for vertical (column) elements."""
        fixed = elem_load.copy()
        for e in range(len(elem_directions)):
            if abs(elem_directions[e, 2]) > abs(elem_directions[e, 0]):
                fixed[e] = [0.0, 0.0, 0.0]

        n_zeroed = np.sum(np.any(elem_load != fixed, axis=1))
        if n_zeroed > 0:
            print(f"    Fixed elem_load: zeroed {n_zeroed} "
                  f"column elements")
        return fixed

    @staticmethod
    def fix_elem_moment(elem_moment, elem_directions):
        """Zero out elem_moment for vertical (column) elements."""
        fixed = elem_moment.copy()
        for e in range(len(elem_directions)):
            if abs(elem_directions[e, 2]) > abs(elem_directions[e, 0]):
                fixed[e] = 0.0

        n_zeroed = np.sum(
            elem_moment.flatten() != fixed.flatten()
        )
        if n_zeroed > 0:
            print(f"    Fixed elem_moment: zeroed {n_zeroed} "
                  f"column elements")
        return fixed

    # ───────────────────────────────────────
    # LOAD ONE CASE
    # ───────────────────────────────────────

    def load_case(self, case_num: int) -> dict:
        print(f"\n  Case {case_num}:")

        # ═══ 1. CONFIG ═══
        config = self._read_config(case_num)
        nearest_node = config['nearest_node_id']
        traced_elem = config['traced_element_id']

        merge_info = config.get('merge_info', {})
        if merge_info:
            print(f"    Source: {merge_info.get('source_folder', '?')} "
                  f"(orig ID: {merge_info.get('original_case_id', '?')})")

        print(f"    Response: node={nearest_node} "
              f"(traced_elem={traced_elem})")
        print(f"    Params: E={config.get('youngs_modulus', '?'):.2e}, "
              f"I22={config.get('I22', '?'):.4e}")

        # ═══ 2. PRIMAL VTK ═══
        primal = self._read_vtk(self._primal_vtk_path(case_num))
        coords = np.array(primal.points, dtype=np.float64)
        connectivity = self._extract_connectivity(primal)
        N = len(coords)
        E = len(connectivity)

        # ── Point load (Fx, Fy, Fz per node) ──
        point_load = self._get_point_field(primal, 'POINT_LOAD')
        if point_load is None:
            point_load = np.zeros((N, 3))
            print(f"    ⚠ POINT_LOAD not found in VTK")

        # ── Point moment (extract My only) ──
        point_moment_raw = self._get_point_field(
            primal, 'POINT_MOMENT'
        )
        if point_moment_raw is not None:
            point_moment_My = point_moment_raw[:, 1:2]  # (N, 1)
        else:
            point_moment_My = np.zeros((N, 1))

        # ── Print load summary ──
        load_mags = np.linalg.norm(point_load, axis=1)
        n_force_loaded = np.sum(load_mags > 1e-10)
        moment_mags = np.abs(point_moment_My.flatten())
        n_moment_loaded = np.sum(moment_mags > 1e-10)

        print(f"    Forces: {n_force_loaded}/{N} nodes loaded")
        if n_force_loaded > 0:
            loaded = load_mags > 1e-10
            print(f"      Fx: [{point_load[loaded, 0].min():.2f}, "
                  f"{point_load[loaded, 0].max():.2f}]")
            print(f"      Fz: [{point_load[loaded, 2].min():.2f}, "
                  f"{point_load[loaded, 2].max():.2f}]")

        if n_moment_loaded > 0:
            My_vals = point_moment_My.flatten()[moment_mags > 1e-10]
            print(f"    Moments: {n_moment_loaded}/{N} nodes")
            print(f"      My: [{My_vals.min():.2f}, "
                  f"{My_vals.max():.2f}]")
        else:
            print(f"    Moments: disabled (all zeros)")

        # Cross-check with config
        cfg_summary = config.get('load_summary', {})
        if cfg_summary:
            print(f"    Config loaded_nodes: "
                  f"{cfg_summary.get('loaded_nodes', '?')}")

        # ── Element properties ──
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

        # ── Displacement, Rotation ──
        displacement = self._get_point_field(primal, 'DISPLACEMENT')
        rotation = self._get_point_field(primal, 'ROTATION')

        # ── 2D DOFs ──
        if displacement is not None and rotation is not None:
            nodal_disp_2d = np.stack([
                displacement[:, 0],   # u_x
                displacement[:, 2],   # u_z
                rotation[:, 1]        # phi_y
            ], axis=1)
        else:
            nodal_disp_2d = None

        # ── Internal forces ──
        moment = self._get_cell_field(primal, 'MOMENT')
        force = self._get_cell_field(primal, 'FORCE')

        if force is not None:
            elem_N = force[:, 0]
            elem_V = force[:, 2]
        else:
            elem_N = None
            elem_V = None

        elem_M = moment[:, 1] if moment is not None else None

        # ═══ 3. ADJOINT VTK ═══
        adjoint_path = self._adjoint_vtk_path(case_num)
        I22_sensitivity = None
        if adjoint_path.exists():
            adjoint = self._read_vtk(adjoint_path)
            I22_sensitivity = self._get_cell_field(
                adjoint, 'I22_SENSITIVITY'
            )
            if I22_sensitivity is not None:
                print(f"    Sensitivity: "
                      f"[{I22_sensitivity.min():.4e}, "
                      f"{I22_sensitivity.max():.4e}]")
        else:
            print(f"    ⚠ Adjoint not found")

        # ═══ 4. DERIVED DATA ═══
        bc_disp, bc_rot = self._build_bc_flags(
            coords, displacement, rotation
        )
        response_node_flag = self._build_response_node_flag(
            N, nearest_node
        )
        geom = self._compute_element_geometry(
            coords, connectivity
        )

        # Element-level load (verification only)
        elem_load = self._compute_element_load(
            point_load, connectivity
        )
        elem_load = self.fix_elem_load(
            elem_load, geom['directions']
        )

        elem_applied_moment = self._compute_element_moment(
            point_moment_My, connectivity
        )
        elem_applied_moment = self.fix_elem_moment(
            elem_applied_moment, geom['directions']
        )

        # ═══ 5. ASSEMBLE ═══
        case = {
            # ── Node-level inputs ──
            'coords':              coords,              # (N, 3)
            'bc_disp':             bc_disp,             # (N, 1)
            'bc_rot':              bc_rot,              # (N, 1)
            'point_load':          point_load,          # (N, 3)
            'point_moment_My':     point_moment_My,     # (N, 1)
            'response_node_flag':  response_node_flag,  # (N, 1)

            # ── Element-level inputs ──
            'connectivity':        connectivity,        # (E, 2)
            'elem_lengths':        geom['lengths'],     # (E,)
            'elem_directions':     geom['directions'],  # (E, 3)
            'young_modulus':       props['young_modulus'],
            'cross_area':          props['cross_area'],
            'I22':                 props['I22'],
            'I33':                 props['I33'],
            'poisson_ratio':       props['poisson_ratio'],
            'density':             props['density'],
            'torsional_inertia':   props['torsional_inertia'],

            # ── Element-level derived (verification) ──
            'elem_load':           elem_load,           # (E, 3)
            'elem_applied_moment': elem_applied_moment, # (E, 1)

            # ── Node-level outputs ──
            'displacement':        displacement,        # (N, 3)
            'rotation':            rotation,            # (N, 3)
            'nodal_disp_2d':       nodal_disp_2d,       # (N, 3)

            # ── Element-level outputs ──
            'moment':              moment,              # (E, 3)
            'force':               force,               # (E, 3)
            'elem_N':              elem_N,              # (E,)
            'elem_V':              elem_V,              # (E,)
            'elem_M':              elem_M,              # (E,)
            'I22_sensitivity':     I22_sensitivity,     # (E,)

            # ── Metadata ──
            'n_nodes':             N,
            'n_elements':          E,
            'case_num':            case_num,
            'nearest_node_id':     nearest_node,
            'traced_element_id':   traced_elem,
            'config':              config,
        }

        # ── Print verification ──
        support_nodes = np.where(bc_disp.flatten() > 0.5)[0]
        resp_node = np.where(
            response_node_flag.flatten() > 0.5
        )[0]
        n_loaded_elems = np.sum(
            np.linalg.norm(elem_load, axis=1) > 1e-10
        )
        print(f"    Supports: {support_nodes.tolist()}")
        print(f"    Response node: {resp_node.tolist()}")
        print(f"    Loaded elements: {n_loaded_elems}/{E}")
        print(f"    ✓ Case {case_num} loaded")

        return case

    # ───────────────────────────────────────
    # LOAD ALL / RANGE
    # ───────────────────────────────────────

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

    # ───────────────────────────────────────
    # SUMMARY + VERIFICATION
    # ───────────────────────────────────────

    def _print_summary(self, dataset):
        if not dataset:
            print("  No data loaded!")
            return

        n = len(dataset)
        c0 = dataset[0]
        N = c0['n_nodes']
        E = c0['n_elements']

        print(f"\n{'═'*60}")
        print(f"  DATASET SUMMARY")
        print(f"{'═'*60}")
        print(f"  Cases: {n}  |  Nodes: {N}  |  Elements: {E}")

        # ── 1. Per-node force statistics ──
        all_Fx, all_Fz = [], []
        all_loaded_counts = []

        for c in dataset:
            pl = c['point_load']
            mags = np.linalg.norm(pl, axis=1)
            loaded = mags > 1e-10
            all_loaded_counts.append(np.sum(loaded))
            if np.any(loaded):
                all_Fx.extend(pl[loaded, 0].tolist())
                all_Fz.extend(pl[loaded, 2].tolist())

        all_Fx = np.array(all_Fx) if all_Fx else np.array([0.0])
        all_Fz = np.array(all_Fz) if all_Fz else np.array([0.0])

        print(f"\n  Force statistics (across {n} cases):")
        print(f"    Fx range:     [{all_Fx.min():.2f}, "
              f"{all_Fx.max():.2f}]")
        print(f"    Fz range:     [{all_Fz.min():.2f}, "
              f"{all_Fz.max():.2f}]")
        print(f"    Loaded nodes: [{min(all_loaded_counts)}, "
              f"{max(all_loaded_counts)}] per case")
        print(f"    Mean loaded:  {np.mean(all_loaded_counts):.1f}")

        # ── 2. Per-node moment statistics ──
        all_My = []
        all_moment_counts = []

        for c in dataset:
            my = c['point_moment_My'].flatten()
            my_loaded = np.abs(my) > 1e-10
            all_moment_counts.append(np.sum(my_loaded))
            if np.any(my_loaded):
                all_My.extend(my[my_loaded].tolist())

        if all_My:
            all_My = np.array(all_My)
            print(f"\n  Moment statistics:")
            print(f"    My range:      [{all_My.min():.2f}, "
                  f"{all_My.max():.2f}]")
            print(f"    Moment nodes:  [{min(all_moment_counts)}, "
                  f"{max(all_moment_counts)}] per case")
        else:
            print(f"\n  Moments: DISABLED (all zeros)")

        # ── 3. Material parameters ──
        Es = [c['config'].get('youngs_modulus', 0)
              for c in dataset]
        I22s = [c['config'].get('I22', 0) for c in dataset]
        resp_nodes = [c['nearest_node_id'] for c in dataset]

        print(f"\n  Material parameters:")
        print(f"    E:             [{min(Es):.2e}, {max(Es):.2e}]")
        print(f"    I22:           [{min(I22s):.4e}, "
              f"{max(I22s):.4e}]")
        print(f"    Response nodes: {len(set(resp_nodes))} unique")

        # ── 4. Source datasets ──
        sources = set()
        for c in dataset:
            mi = c['config'].get('merge_info', {})
            if mi:
                sources.add(mi.get('source_folder', 'unknown'))
        if sources:
            print(f"\n  Merged from: {sorted(sources)}")

        # ── 5. Verify elem_load fix ──
        n_beams = np.sum(
            np.abs(c0['elem_directions'][:, 0]) >
            np.abs(c0['elem_directions'][:, 2])
        )
        n_columns = E - n_beams
        n_loaded_elems = np.sum(
            np.linalg.norm(c0['elem_load'], axis=1) > 1e-10
        )

        print(f"\n  Verification (case 0):")
        print(f"    Beams: {n_beams}, Columns: {n_columns}")
        print(f"    Loaded elements: {n_loaded_elems}")

        if n_loaded_elems <= n_beams:
            print(f"    ✓ No columns carry load")
        else:
            print(f"    ✗ WARNING: columns carry load!")

        # ── 6. Verify consistency across cases ──
        print(f"\n  Consistency checks:")

        # Same mesh?
        all_same_N = all(c['n_nodes'] == N for c in dataset)
        all_same_E = all(c['n_elements'] == E for c in dataset)
        print(f"    {'✓' if all_same_N else '✗'} "
              f"Same node count: {N}")
        print(f"    {'✓' if all_same_E else '✗'} "
              f"Same element count: {E}")

        # All have displacement?
        all_have_disp = all(
            c['displacement'] is not None for c in dataset
        )
        print(f"    {'✓' if all_have_disp else '✗'} "
              f"All cases have displacement")

        # All have internal forces?
        all_have_force = all(
            c['force'] is not None for c in dataset
        )
        print(f"    {'✓' if all_have_force else '✗'} "
              f"All cases have internal forces")

        # All have sensitivity?
        n_with_sens = sum(
            1 for c in dataset
            if c['I22_sensitivity'] is not None
        )
        print(f"    {'✓' if n_with_sens == n else '⚠'} "
              f"Sensitivity: {n_with_sens}/{n} cases")

        # Support nodes consistent?
        support_sets = []
        for c in dataset:
            s = set(np.where(
                c['bc_disp'].flatten() > 0.5
            )[0].tolist())
            support_sets.append(s)
        all_same_supports = all(
            s == support_sets[0] for s in support_sets
        )
        print(f"    {'✓' if all_same_supports else '✗'} "
              f"Same supports: {sorted(support_sets[0])}")

        # Loads actually vary?
        if n > 1:
            loads_vary = not np.allclose(
                dataset[0]['point_load'],
                dataset[1]['point_load'],
                atol=1e-10
            )
            print(f"    {'✓' if loads_vary else '✗'} "
                  f"Loads vary between cases")

        # ── 7. Feature dimensions table ──
        print(f"\n  Data dimensions:")
        print(f"  ┌──────────────────────────────────────────────────┐")
        print(f"  │ NODE-LEVEL DATA:                                 │")
        print(f"  │   coords              (N, 3)  x, y, z           │")
        print(f"  │   bc_disp             (N, 1)  constraint        │")
        print(f"  │   bc_rot              (N, 1)  constraint        │")
        print(f"  │   point_load          (N, 3)  Fx, Fy, Fz *VARY* │")
        print(f"  │   point_moment_My     (N, 1)  My *VARY/ZERO*    │")
        print(f"  │   response_node_flag  (N, 1)  response loc      │")
        print(f"  │   displacement        (N, 3)  solution          │")
        print(f"  │   rotation            (N, 3)  solution          │")
        print(f"  │   nodal_disp_2d       (N, 3)  [u_x, u_z, φ_y]  │")
        print(f"  ├──────────────────────────────────────────────────┤")
        print(f"  │ ELEMENT-LEVEL DATA:                              │")
        print(f"  │   connectivity        (E, 2)  topology          │")
        print(f"  │   elem_lengths        (E,)    geometry          │")
        print(f"  │   elem_directions     (E, 3)  geometry          │")
        print(f"  │   E, A, I22, I33      (E,)    properties        │")
        print(f"  │   ν, density, J       (E,)    properties        │")
        print(f"  │   elem_load           (E, 3)  verification      │")
        print(f"  │   elem_applied_moment (E, 1)  verification      │")
        print(f"  │   force               (E, 3)  internal N,V      │")
        print(f"  │   moment              (E, 3)  internal M        │")
        print(f"  │   I22_sensitivity     (E,)    adjoint            │")
        print(f"  └──────────────────────────────────────────────────┘")
        print(f"{'═'*60}\n")

    # ───────────────────────────────────────
    # SAVE / LOAD
    # ───────────────────────────────────────

    @staticmethod
    def save(dataset, filepath):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f,
                       protocol=pickle.HIGHEST_PROTOCOL)
        mb = os.path.getsize(filepath) / 1e6
        print(f"Saved: {filepath} ({mb:.1f} MB, "
              f"{len(dataset)} cases)")

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Loaded: {filepath} ({len(dataset)} cases)")
        return dataset


# ================================================================
# STANDALONE VERIFICATION
# ================================================================

def verify_case(case: dict) -> bool:
    """Detailed verification of one loaded case."""
    print(f"\n{'═'*60}")
    print(f"  CASE {case['case_num']} VERIFICATION")
    print(f"{'═'*60}")

    N = case['n_nodes']
    E = case['n_elements']
    all_ok = True

    # ── 1. Shape checks ──
    print(f"\n  1. SHAPES:")
    checks = [
        ('coords',            case['coords'].shape,            (N, 3)),
        ('connectivity',      case['connectivity'].shape,      (E, 2)),
        ('bc_disp',           case['bc_disp'].shape,           (N, 1)),
        ('bc_rot',            case['bc_rot'].shape,            (N, 1)),
        ('point_load',        case['point_load'].shape,        (N, 3)),
        ('point_moment_My',   case['point_moment_My'].shape,   (N, 1)),
        ('response_node_flag',
         case['response_node_flag'].shape,                     (N, 1)),
        ('elem_lengths',      case['elem_lengths'].shape,      (E,)),
        ('elem_directions',   case['elem_directions'].shape,   (E, 3)),
        ('young_modulus',     case['young_modulus'].shape,      (E,)),
        ('cross_area',        case['cross_area'].shape,        (E,)),
        ('I22',               case['I22'].shape,               (E,)),
        ('elem_load',         case['elem_load'].shape,         (E, 3)),
        ('elem_applied_moment',
         case['elem_applied_moment'].shape,                    (E, 1)),
    ]

    for name, actual, expected in checks:
        ok = (actual == expected)
        if not ok:
            all_ok = False
        print(f"    {'✓' if ok else '✗'} {name:<22} "
              f"{str(actual):<12} (expected {expected})")

    # ── 2. Solution shapes ──
    print(f"\n  2. SOLUTION:")
    if case['displacement'] is not None:
        print(f"    ✓ displacement: {case['displacement'].shape}")
    else:
        print(f"    ✗ displacement: None")
        all_ok = False

    if case['rotation'] is not None:
        print(f"    ✓ rotation: {case['rotation'].shape}")
    else:
        print(f"    ✗ rotation: None")
        all_ok = False

    if case['nodal_disp_2d'] is not None:
        print(f"    ✓ nodal_disp_2d: {case['nodal_disp_2d'].shape}")
    else:
        print(f"    ✗ nodal_disp_2d: None")

    if case['force'] is not None:
        print(f"    ✓ force: {case['force'].shape}")
    else:
        print(f"    ✗ force: None")

    if case['I22_sensitivity'] is not None:
        print(f"    ✓ I22_sensitivity: "
              f"{case['I22_sensitivity'].shape}")
    else:
        print(f"    ⚠ I22_sensitivity: None")

    # ── 3. Boundary conditions ──
    print(f"\n  3. BOUNDARY CONDITIONS:")
    supports = np.where(case['bc_disp'].flatten() > 0.5)[0]
    rot_fixed = np.where(case['bc_rot'].flatten() > 0.5)[0]
    print(f"    Displacement fixed: {supports.tolist()}")
    print(f"    Rotation fixed: {rot_fixed.tolist()}")

    ok = len(supports) >= 2
    print(f"    {'✓' if ok else '✗'} At least 2 supports")
    if not ok:
        all_ok = False

    # ── 4. Response node ──
    print(f"\n  4. RESPONSE NODE:")
    resp = np.where(
        case['response_node_flag'].flatten() > 0.5
    )[0]
    print(f"    Flag node(s): {resp.tolist()}")
    print(f"    Config nearest_node_id: "
          f"{case['nearest_node_id']}")

    ok = (len(resp) == 1 and
          resp[0] == case['nearest_node_id'])
    print(f"    {'✓' if ok else '✗'} Match")
    if not ok:
        all_ok = False

    # ── 5. Per-node loads ──
    print(f"\n  5. PER-NODE LOADS:")
    pl = case['point_load']
    load_mags = np.linalg.norm(pl, axis=1)
    n_loaded = np.sum(load_mags > 1e-10)
    print(f"    Force-loaded: {n_loaded}/{N}")

    if n_loaded > 0:
        loaded = load_mags > 1e-10
        print(f"    Fx: [{pl[loaded, 0].min():.2f}, "
              f"{pl[loaded, 0].max():.2f}]")
        print(f"    Fy: [{pl[loaded, 1].min():.2f}, "
              f"{pl[loaded, 1].max():.2f}] (should be ~0)")
        print(f"    Fz: [{pl[loaded, 2].min():.2f}, "
              f"{pl[loaded, 2].max():.2f}]")

        # Check Fy is zero
        fy_ok = np.allclose(pl[:, 1], 0.0, atol=1e-10)
        print(f"    {'✓' if fy_ok else '⚠'} Fy all zero "
              f"(2D frame)")

        # Check supports have no load
        for s in supports:
            if load_mags[s] > 1e-10:
                print(f"    ⚠ Support node {s} has load!")

    # Per-node moments
    my = case['point_moment_My'].flatten()
    n_my = np.sum(np.abs(my) > 1e-10)
    print(f"    Moment-loaded: {n_my}/{N}")
    if n_my > 0:
        my_loaded = np.abs(my) > 1e-10
        print(f"    My: [{my[my_loaded].min():.2f}, "
              f"{my[my_loaded].max():.2f}]")

    # ── 6. Element load fix ──
    print(f"\n  6. ELEMENT LOAD FIX:")
    dirs = case['elem_directions']
    el = case['elem_load']

    n_beams = 0
    n_cols = 0
    col_with_load = 0

    for e in range(E):
        is_col = abs(dirs[e, 2]) > abs(dirs[e, 0])
        if is_col:
            n_cols += 1
            if np.linalg.norm(el[e]) > 1e-10:
                col_with_load += 1
        else:
            n_beams += 1

    print(f"    Beams: {n_beams}, Columns: {n_cols}")
    ok = col_with_load == 0
    print(f"    {'✓' if ok else '✗'} Columns with load: "
          f"{col_with_load} (should be 0)")
    if not ok:
        all_ok = False

    # ── 7. Direction vectors ──
    print(f"\n  7. DIRECTION VECTORS:")
    dir_mags = np.linalg.norm(dirs, axis=1)
    ok = np.allclose(dir_mags, 1.0, atol=1e-10)
    print(f"    {'✓' if ok else '✗'} All unit vectors "
          f"(mag range: [{dir_mags.min():.6f}, "
          f"{dir_mags.max():.6f}])")
    if not ok:
        all_ok = False

    # ── 8. Connectivity ──
    print(f"\n  8. CONNECTIVITY:")
    conn = case['connectivity']
    ok = (conn.min() >= 0 and conn.max() < N)
    print(f"    {'✓' if ok else '✗'} Node indices in range "
          f"[0, {N-1}]")
    if not ok:
        all_ok = False

    # ── Result ──
    status = "ALL PASSED ✓" if all_ok else "SOME FAILED ✗"
    print(f"\n  RESULT: {status}")
    print(f"{'═'*60}\n")
    return all_ok


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  STEP 1: Load Kratos Data (Per-Node Loads + Moments)")
    print("=" * 60)

    loader = FrameDataLoader(
        primal_base_dir=(
            "test_files/Kratos_data_creation/primal"
        ),
        adjoint_base_dir=(
            "test_files/Kratos_data_creation/adjoint"
        ),
        primal_folder_prefix="case_primal_",
        adjoint_folder_prefix="case_adjoint_",
        primal_vtk_subdir="vtk_output_primal",
        adjoint_vtk_subdir="vtk_output_adjoint",
        vtk_filename="Structure_0_1.vtk"
    )

    dataset = loader.load_all()

    from pathlib import Path
    print(f"Working directory: {os.getcwd()}")
    CURRENT_SUBFOLDER = Path(__file__).resolve().parent
    os.chdir(CURRENT_SUBFOLDER)
    print(f"Working directory: {os.getcwd()}")

    FrameDataLoader.save(dataset, "DATA/frame_dataset.pkl")

    # ── Detailed verification of first case ──
    if dataset:
        print("\n" + "=" * 60)
        print("  DETAILED VERIFICATION")
        print("=" * 60)

        verify_case(dataset[0])

        # Quick spot-check of a few cases
        if len(dataset) > 1:
            c0 = dataset[0]
            c1 = dataset[1]

            print(f"\n  Cross-case check:")
            loads_differ = not np.allclose(
                c0['point_load'], c1['point_load'], atol=1e-10
            )
            print(f"    {'✓' if loads_differ else '✗'} "
                  f"Loads differ between case 0 and 1")

            disp_differ = not np.allclose(
                c0['displacement'], c1['displacement'],
                atol=1e-10
            )
            print(f"    {'✓' if disp_differ else '✗'} "
                  f"Displacements differ")

            # Show sample per-node values
            print(f"\n  Sample node loads (case 0):")
            loaded = np.where(
                np.linalg.norm(c0['point_load'], axis=1) > 1e-10
            )[0]
            for nid in loaded[:5]:
                fx, fy, fz = c0['point_load'][nid]
                my = c0['point_moment_My'][nid, 0]
                print(f"    Node {nid}: Fx={fx:8.2f} "
                      f"Fz={fz:8.2f} My={my:8.2f}")
            if len(loaded) > 5:
                print(f"    ... ({len(loaded) - 5} more)")

    print("\n  Step 1 COMPLETE → proceed to Step 2")