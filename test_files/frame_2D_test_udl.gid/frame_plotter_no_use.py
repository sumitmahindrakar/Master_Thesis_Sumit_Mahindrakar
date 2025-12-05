"""
Extended Frame Analysis Results Plotter

Adds:
- Column diagrams (Moment, Shear, Axial) plotting
- FEM vs analytical comparisons for columns (M, V, N)
- Column deflected shapes in FEM vs Analytical plots
- Improved member recognition for multi-bay frames (clusters columns & beams)

Drop-in replacement / extension for your original script. Save as a .py and run.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import os


@dataclass
class VTKData:
    points: np.ndarray = None
    cells: List[List[int]] = field(default_factory=list)
    point_data: Dict[str, np.ndarray] = field(default_factory=dict)
    cell_data: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Member:
    id: int
    node_i: int
    node_j: int
    start_coords: np.ndarray
    end_coords: np.ndarray
    length: float
    angle: float
    orientation: str
    member_type: str = ""
    # link back to higher-level member group if needed
    group_id: Optional[int] = None


@dataclass
class ColumnGroup:
    id: int
    x_coord: float
    node_indices: List[int] = field(default_factory=list)
    member_ids: List[int] = field(default_factory=list)
    y_min: float = 0.0
    y_max: float = 0.0


@dataclass
class FrameParameters:
    L: float
    H: float
    w: float
    P: float
    E: float
    I_beam: float
    I_col: float
    A_beam: float
    A_col: float
    frame_type: str
    load_type: str


# ---------- Parsing (unchanged) ----------

def parse_vtk_file(filepath: str) -> VTKData:
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = VTKData()
    i = 0
    current_section = None

    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue

        if line.startswith('POINTS'):
            parts = line.split()
            n_points = int(parts[1])
            points = []
            i += 1
            while len(points) < n_points * 3 and i < len(lines):
                values = lines[i].strip().split()
                points.extend([float(v) for v in values if v])
                i += 1
            data.points = np.array(points).reshape(n_points, 3)
            continue

        if line.startswith('CELLS'):
            parts = line.split()
            n_cells = int(parts[1])
            cells = []
            i += 1
            for _ in range(n_cells):
                if i < len(lines):
                    cell_values = [int(v) for v in lines[i].strip().split()]
                    cells.append(cell_values[1:])
                    i += 1
            data.cells = cells
            continue

        if line.startswith('POINT_DATA'):
            current_section = 'point'
            i += 1
            continue

        if line.startswith('CELL_DATA'):
            current_section = 'cell'
            i += 1
            continue

        if line.startswith('FIELD'):
            n_arrays = int(line.split()[-1])
            i += 1

            for _ in range(n_arrays):
                if i >= len(lines):
                    break

                field_line = lines[i].strip().split()
                if len(field_line) < 4:
                    i += 1
                    continue

                field_name = field_line[0]
                n_components = int(field_line[1])
                n_tuples = int(field_line[2])

                i += 1
                values = []
                while len(values) < n_components * n_tuples and i < len(lines):
                    val_line = lines[i].strip()
                    if val_line and not any(val_line.startswith(k) for k in 
                                           ['FIELD', 'POINT_DATA', 'CELL_DATA', 'CELL_TYPES']):
                        try:
                            values.extend([float(v) for v in val_line.split() if v])
                            i += 1
                        except ValueError:
                            break
                    else:
                        break

                if len(values) == n_components * n_tuples:
                    arr = np.array(values).reshape(n_tuples, n_components)
                    if current_section == 'point':
                        data.point_data[field_name] = arr
                    elif current_section == 'cell':
                        data.cell_data[field_name] = arr
            continue

        i += 1

    return data


# ---------- Improved geometry analysis ----------

def cluster_unique(values: np.ndarray, tol: float = 1e-3) -> List[float]:
    """Cluster nearly-equal scalar values and return cluster centers."""
    if len(values) == 0:
        return []
    vals = np.sort(np.array(values))
    clusters = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    centers = [np.mean(c) for c in clusters]
    return centers


def analyze_frame_geometry(vtk_data: VTKData, tol: float = 1e-6) -> Tuple[List[Member], Dict, List[ColumnGroup]]:
    members = []

    x_coords = vtk_data.points[:, 0]
    y_coords = vtk_data.points[:, 1]

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    frame_info = {
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'span': x_max - x_min,
        'height': y_max - y_min,
        'n_columns': 0,
        'n_beams': 0
    }

    # Create raw members and primary orientation classification
    for idx, cell in enumerate(vtk_data.cells):
        node_i, node_j = cell[0], cell[1]
        start = vtk_data.points[node_i]
        end = vtk_data.points[node_j]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        angle = np.arctan2(dy, dx)

        # fuzzy orientation test
        if abs(dx) < 1e-6 and abs(dy) > 1e-6:
            orientation = 'vertical'
            member_type = 'column'
            frame_info['n_columns'] += 1
        elif abs(dy) < 1e-6 and abs(dx) > 1e-6:
            orientation = 'horizontal'
            member_type = 'beam'
            frame_info['n_beams'] += 1
        else:
            orientation = 'inclined'
            member_type = 'inclined'

        members.append(Member(
            id=idx + 1,
            node_i=node_i,
            node_j=node_j,
            start_coords=start.copy(),
            end_coords=end.copy(),
            length=length,
            angle=angle,
            orientation=orientation,
            member_type=member_type
        ))

    # ---- Group vertical members into column groups by x coordinate clustering ----
    # Collect x positions of all vertical members (use midpoints)
    vert_mid_x = []
    vert_member_idxs = []
    for m in members:
        if m.orientation == 'vertical':
            midx = 0.5 * (m.start_coords[0] + m.end_coords[0])
            vert_mid_x.append(midx)
            vert_member_idxs.append(m.id)

    column_groups: List[ColumnGroup] = []
    if vert_mid_x:
        centers = cluster_unique(np.array(vert_mid_x), tol=max(tol, (x_max-x_min)*1e-3))
        # build groups
        for gid, cx in enumerate(centers, start=1):
            cg = ColumnGroup(id=gid, x_coord=cx)
            column_groups.append(cg)

        # assign vertical members to nearest column center
        for m in members:
            if m.orientation != 'vertical':
                continue
            midx = 0.5 * (m.start_coords[0] + m.end_coords[0])
            # find nearest center
            dists = [abs(midx - c.x_coord) for c in column_groups]
            idx_min = int(np.argmin(dists))
            column_groups[idx_min].member_ids.append(m.id)
            m.group_id = column_groups[idx_min].id

    # For each column group collect node indices and y extents
    for cg in column_groups:
        nodes = set()
        ys = []
        for mid in cg.member_ids:
            m = next(mm for mm in members if mm.id == mid)
            nodes.add(m.node_i)
            nodes.add(m.node_j)
            ys.append(m.start_coords[1])
            ys.append(m.end_coords[1])
        cg.node_indices = sorted(list(nodes))
        if ys:
            cg.y_min = float(np.min(ys))
            cg.y_max = float(np.max(ys))

    # Identify beam(s) by clustering y positions of horizontal members
    # (use for multi-bay: there could be several beams at different levels)
    horiz_mid_y = []
    for m in members:
        if m.orientation == 'horizontal':
            midy = 0.5 * (m.start_coords[1] + m.end_coords[1])
            horiz_mid_y.append(midy)
    beam_levels = cluster_unique(np.array(horiz_mid_y), tol=max(tol, (y_max-y_min)*1e-3)) if horiz_mid_y else []

    # store beam level info
    frame_info['beam_levels'] = beam_levels

    return members, frame_info, column_groups


# ---------- Analytical (unchanged except small helpers) ----------

class PortalFrameAnalytical:

    @staticmethod
    def pinned_base_udl(params: FrameParameters) -> Dict:
        L = params.L
        H = params.H
        w = params.w
        E = params.E
        I = params.I_beam

        Ry = w * L / 2
        M_max_beam = w * L**2 / 8
        V_max_beam = w * L / 2
        delta_max_beam = 5 * w * L**4 / (384 * E * I)
        theta_beam_end = w * L**3 / (24 * E * I)
        N_column = Ry

        return {
            'reactions': {'R1x': 0, 'R1y': Ry, 'R4x': 0, 'R4y': Ry},
            'beam': {
                'M_max': M_max_beam,
                'M_left': 0,
                'M_right': 0,
                'V_left': V_max_beam,
                'V_right': -V_max_beam,
                'V_max': V_max_beam,
                'delta_max': delta_max_beam,
                'theta_end': theta_beam_end
            },
            'columns': {
                'M_top': 0,
                'M_bottom': 0,
                'N': N_column,
                'V': 0
            },
            'description': 'Portal frame with pinned bases, UDL on beam'
        }

    @staticmethod
    def fixed_base_udl(params: FrameParameters) -> Dict:
        L = params.L
        H = params.H
        w = params.w
        E = params.E
        I_beam = params.I_beam
        I_col = params.I_col

        k_beam = I_beam / L
        k_col = I_col / H
        alpha = k_col / (k_beam + k_col)
        FEM_beam = w * L**2 / 12
        M_joint = FEM_beam * (1 - alpha)
        M_col_top = FEM_beam * alpha
        M_col_bottom = M_col_top / 2
        M_center = w * L**2 / 8 - M_joint
        Ry = w * L / 2
        V_col = (M_col_top + M_col_bottom) / H
        delta_max_beam = w * L**4 / (384 * E * I_beam)

        return {
            'reactions': {'R1x': V_col, 'R1y': Ry, 'M1': M_col_bottom, 'R4x': -V_col, 'R4y': Ry, 'M4': M_col_bottom},
            'beam': {
                'M_max': M_center,
                'M_left': -M_joint,
                'M_right': -M_joint,
                'V_left': w * L / 2,
                'V_right': -w * L / 2,
                'V_max': w * L / 2,
                'delta_max': delta_max_beam
            },
            'columns': {
                'M_top': M_col_top,
                'M_bottom': M_col_bottom,
                'N': Ry,
                'V': V_col
            },
            'description': 'Portal frame with fixed bases, UDL on beam'
        }

    @staticmethod
    def get_beam_distributions(x: np.ndarray, params: FrameParameters, frame_type: str = 'pinned') -> Dict:
        L = params.L
        w = params.w
        E = params.E
        I = params.I_beam

        if frame_type == 'pinned':
            moment = w * L * x / 2 - w * x**2 / 2
            shear = w * L / 2 - w * x
            deflection = w * x * (L**3 - 2*L*x**2 + x**3) / (24 * E * I)
            rotation = w * (L**3 - 6*L*x**2 + 4*x**3) / (24 * E * I)
        else:
            moment = -w * L**2 / 12 + w * L * x / 2 - w * x**2 / 2
            shear = w * L / 2 - w * x
            deflection = w * x**2 * (L - x)**2 / (24 * E * I)
            rotation = w * x * (L - x) * (L - 2*x) / (12 * E * I)

        return {'x': x, 'moment': moment, 'shear': shear, 'deflection': deflection, 'rotation': rotation}

    @staticmethod
    def get_column_distributions(y: np.ndarray, params: FrameParameters, analytical: Dict, column: str = 'left') -> Dict:
        H = params.H
        M_top = analytical['columns']['M_top']
        M_bottom = analytical['columns']['M_bottom']
        N = analytical['columns']['N']
        V = analytical['columns']['V']
        moment = M_bottom + (M_top - M_bottom) * y / H
        shear = np.full_like(y, V)
        axial = np.full_like(y, -N)
        return {'y': y, 'moment': moment, 'shear': shear, 'axial': axial}


# ---------- Error calculation extended for columns ----------

def calculate_frame_errors(vtk_data: VTKData, members: List[Member], params: FrameParameters, frame_info: Dict, column_groups: List[ColumnGroup]) -> Dict:
    if params.frame_type == 'portal_pinned':
        analytical = PortalFrameAnalytical.pinned_base_udl(params)
    else:
        analytical = PortalFrameAnalytical.fixed_base_udl(params)

    errors = {'analytical': analytical, 'frame_info': frame_info, 'params': params}

    # Reactions (same as before)
    if 'REACTION' in vtk_data.point_data:
        reactions = vtk_data.point_data['REACTION']
        support_nodes = []
        for i, point in enumerate(vtk_data.points):
            if abs(point[1] - frame_info['y_min']) < 1e-6:
                support_nodes.append(i)
        if len(support_nodes) >= 2:
            support_nodes.sort(key=lambda n: vtk_data.points[n, 0])
            node_left = support_nodes[0]
            node_right = support_nodes[-1]
            fem_R1y = reactions[node_left, 1]
            fem_R4y = reactions[node_right, 1]
            anal_R1y = analytical['reactions']['R1y']
            anal_R4y = analytical['reactions']['R4y']
            errors['reactions'] = {
                'R1y': {'fem': fem_R1y / 1000, 'analytical': anal_R1y / 1000, 'error_percent': abs(fem_R1y - anal_R1y) / anal_R1y * 100 if anal_R1y != 0 else 0},
                'R4y': {'fem': fem_R4y / 1000, 'analytical': anal_R4y / 1000, 'error_percent': abs(fem_R4y - anal_R4y) / anal_R4y * 100 if anal_R4y != 0 else 0}
            }

    # Beam deflection (same as before)
    if 'DISPLACEMENT' in vtk_data.point_data:
        disp = vtk_data.point_data['DISPLACEMENT']
        beam_nodes = []
        for i, point in enumerate(vtk_data.points):
            if abs(point[1] - frame_info['y_max']) < 1e-6:
                beam_nodes.append(i)
        if beam_nodes:
            max_disp_idx = max(beam_nodes, key=lambda n: abs(disp[n, 1]))
            fem_delta_max = abs(disp[max_disp_idx, 1])
            anal_delta_max = analytical['beam']['delta_max']
            errors['deflection'] = {'beam_max': {'fem': fem_delta_max * 1000, 'analytical': anal_delta_max * 1000, 'error_percent': abs(fem_delta_max - anal_delta_max) / anal_delta_max * 100 if anal_delta_max != 0 else 0}}

    # Beam moment (same as before)
    if 'MOMENT' in vtk_data.point_data:
        moments = vtk_data.point_data['MOMENT']
        beam_center_nodes = []
        for i, point in enumerate(vtk_data.points):
            if abs(point[1] - frame_info['y_max']) < 1e-6:
                if abs(point[0] - (frame_info['x_min'] + frame_info['x_max'])/2) < 0.1:
                    beam_center_nodes.append(i)
        if beam_center_nodes:
            center_node = beam_center_nodes[0]
            fem_M_center = abs(moments[center_node, 2])
            anal_M_max = analytical['beam']['M_max']
            errors['moment'] = {'beam_center': {'fem': fem_M_center / 1000, 'analytical': anal_M_max / 1000, 'error_percent': abs(fem_M_center - anal_M_max) / anal_M_max * 100 if anal_M_max != 0 else 0}}

    # Member summary
    errors['members'] = {}
    for m in members:
        errors['members'][m.id] = {'type': m.member_type, 'orientation': m.orientation, 'length': m.length, 'group_id': m.group_id}

    # ----- Column comparisons -----
    errors['columns'] = {}
    # If FEM has MOMENT, FORCE or other data at points, we can compare
    fem_moment = vtk_data.point_data.get('MOMENT', None)
    fem_force = vtk_data.point_data.get('FORCE', None)
    fem_disp = vtk_data.point_data.get('DISPLACEMENT', None)

    for cg in column_groups:
        # Build a y-grid from node positions
        ys = [vtk_data.points[n, 1] for n in cg.node_indices]
        if len(ys) == 0:
            continue
        y_sorted_idx = sorted(range(len(ys)), key=lambda k: ys[k])
        y_sorted = [ys[i] for i in y_sorted_idx]

        # Analytical column distribution on these y positions
        y_local = np.array(y_sorted) - frame_info['y_min']  # local from base
        col_dist = PortalFrameAnalytical.get_column_distributions(y_local, params, analytical)

        # Prepare FEM arrays if available
        fem_M = None
        fem_V = None
        fem_N = None

        M_vals = []
        V_vals = []
        N_vals = []
        coords_x = cg.x_coord

        for node in cg.node_indices:
            # moment about z (index 2), force in y (shear) index 1, axial typically index 0
            if fem_moment is not None:
                M_vals.append(abs(fem_moment[node, 2]) / 1000)  # kN·m
            else:
                M_vals.append(None)
            if fem_force is not None:
                # axial = FORCE[node,0], shear-y = FORCE[node,1]
                N_vals.append(-fem_force[node, 0] / 1000)
                V_vals.append(-fem_force[node, 1] / 1000)
            else:
                N_vals.append(None)
                V_vals.append(None)

        # Convert analytical to same units: moments in N·m -> kN·m, forces N -> kN
        anal_M = (col_dist['moment'] / 1000).tolist()
        anal_V = (col_dist['shear'] / 1000).tolist()
        anal_N = (col_dist['axial'] / 1000).tolist()

        # Interpolate analytical to node y positions
        y_anal_grid = np.array(col_dist['y']) if 'y' in col_dist else y_local
        inter_M = np.interp(y_local, y_anal_grid, col_dist['moment']) / 1000
        inter_V = np.interp(y_local, y_anal_grid, col_dist['shear']) / 1000
        inter_N = np.interp(y_local, y_anal_grid, col_dist['axial']) / 1000

        # compute simple pointwise errors where FEM data exists
        M_errors = []
        V_errors = []
        N_errors = []
        for i in range(len(y_local)):
            fem_m_val = M_vals[i]
            fem_v_val = V_vals[i]
            fem_n_val = N_vals[i]
            if fem_m_val is not None and abs(inter_M[i]) > 1e-12:
                M_errors.append(abs(fem_m_val - inter_M[i]) / abs(inter_M[i]) * 100)
            else:
                M_errors.append(None)
            if fem_v_val is not None and abs(inter_V[i]) > 1e-12:
                V_errors.append(abs(fem_v_val - inter_V[i]) / abs(inter_V[i]) * 100)
            else:
                V_errors.append(None)
            if fem_n_val is not None and abs(inter_N[i]) > 1e-12:
                N_errors.append(abs(fem_n_val - inter_N[i]) / abs(inter_N[i]) * 100)
            else:
                N_errors.append(None)

        errors['columns'][cg.id] = {
            'x': cg.x_coord,
            'nodes': cg.node_indices,
            'y_local': y_local.tolist(),
            'analytical': {'M_kNm': inter_M.tolist(), 'V_kN': inter_V.tolist(), 'N_kN': inter_N.tolist()},
            'fem': {'M_kNm': M_vals, 'V_kN': V_vals, 'N_kN': N_vals},
            'errors_percent': {'M': M_errors, 'V': V_errors, 'N': N_errors}
        }

    return errors


# ---------- Plotting: column diagrams ----------

def plot_column_diagrams(params: FrameParameters, errors: Dict, vtk_data: VTKData, column_groups: List[ColumnGroup], output_prefix: str = 'frame'):
    analytical = errors['analytical']
    figs = []
    for cg in column_groups:
        col_data = errors['columns'].get(cg.id, None)
        if col_data is None:
            continue

        y_local = np.array(col_data['y_local'])
        anal_M = np.array(col_data['analytical']['M_kNm'])
        anal_V = np.array(col_data['analytical']['V_kN'])
        anal_N = np.array(col_data['analytical']['N_kN'])

        fem_M = np.array([v if v is not None else np.nan for v in col_data['fem']['M_kNm']])
        fem_V = np.array([v if v is not None else np.nan for v in col_data['fem']['V_kN']])
        fem_N = np.array([v if v is not None else np.nan for v in col_data['fem']['N_kN']])

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'Column {cg.id} (x={cg.x_coord:.3f} m) - Analytical vs FEM', fontsize=14, fontweight='bold')

        axes[0].plot(anal_M, y_local, 'b-', linewidth=2, label='Analytical (M)')
        axes[0].plot(fem_M, y_local, 'ro', label='FEM (M)')
        axes[0].axvline(0, color='k', linewidth=0.5)
        axes[0].set_xlabel('Moment (kN·m)')
        axes[0].set_ylabel('y from base (m)')
        axes[0].invert_yaxis()
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)

        axes[1].plot(anal_V, y_local, 'b-', linewidth=2, label='Analytical (V)')
        axes[1].plot(fem_V, y_local, 'ro', label='FEM (V)')
        axes[1].axvline(0, color='k', linewidth=0.5)
        axes[1].set_xlabel('Shear (kN)')
        axes[1].invert_yaxis()
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.5)

        axes[2].plot(anal_N, y_local, 'b-', linewidth=2, label='Analytical (N)')
        axes[2].plot(fem_N, y_local, 'ro', label='FEM (N)')
        axes[2].axvline(0, color='k', linewidth=0.5)
        axes[2].set_xlabel('Axial (kN)')
        axes[2].invert_yaxis()
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        output_file = f'{output_prefix}_column_{cg.id}_diagrams.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        figs.append(fig)
    return figs


# ---------- Update FEM vs Analytical plot to include column deflected shapes ----------

def plot_fem_vs_analytical_extended(vtk_data: VTKData, members: List[Member], errors: Dict, params: FrameParameters, column_groups: List[ColumnGroup], output_prefix: str = 'frame'):
    analytical = errors['analytical']
    L = params.L
    H = params.H

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FEM vs Analytical Comparison (Extended)', fontsize=14, fontweight='bold')

    # Deformed shape with column deflections
    ax1 = axes[0, 0]
    ax1.plot([0, 0], [0, H], 'k--', linewidth=1, alpha=0.5)
    ax1.plot([L, L], [0, H], 'k--', linewidth=1, alpha=0.5)
    ax1.plot([0, L], [H, H], 'k--', linewidth=1, alpha=0.5)

    x_beam = np.linspace(0, L, 50)
    beam_dist = PortalFrameAnalytical.get_beam_distributions(x_beam, params, params.frame_type.split('_')[1])

    scale = 1000
    anal_y = H - beam_dist['deflection'] * scale
    ax1.plot(x_beam, anal_y, 'b-', linewidth=2, label='Analytical Beam')
    ax1.plot([0, 0], [0, H], 'b-', linewidth=2)
    ax1.plot([L, L], [0, H], 'b-', linewidth=2)

    # plot analytic column as vertical lines
    for cg in column_groups:
        ax1.plot([cg.x_coord, cg.x_coord], [0, H], 'b-', linewidth=2)

    # FEM deflected shape using DISPLACEMENT (scale)
    if 'DISPLACEMENT' in vtk_data.point_data:
        disp = vtk_data.point_data['DISPLACEMENT']
        # draw members deformed
        for m in members:
            x_def = [vtk_data.points[m.node_i, 0] + disp[m.node_i, 0] * scale,
                     vtk_data.points[m.node_j, 0] + disp[m.node_j, 0] * scale]
            y_def = [vtk_data.points[m.node_i, 1] + disp[m.node_i, 1] * scale,
                     vtk_data.points[m.node_j, 1] + disp[m.node_j, 1] * scale]
            ax1.plot(x_def, y_def, 'ro-', linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=1)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Deformed Shape (Scale: {scale}x)', fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Beam BMD comparison (same as before)
    ax2 = axes[0, 1]
    ax2.plot(x_beam, beam_dist['moment'] / 1000, 'b-', linewidth=2, label='Analytical')
    ax2.fill_between(x_beam, 0, beam_dist['moment'] / 1000, alpha=0.3, color='blue')

    if 'MOMENT' in vtk_data.point_data:
        moments = vtk_data.point_data['MOMENT']
        beam_nodes = []
        for i, point in enumerate(vtk_data.points):
            if abs(point[1] - H) < 1e-6:
                beam_nodes.append((point[0], i))
        beam_nodes.sort(key=lambda x: x[0])
        if beam_nodes:
            x_fem = [n[0] for n in beam_nodes]
            m_fem = [-moments[n[1], 2] / 1000 for n in beam_nodes]
            ax2.plot(x_fem, m_fem, 'ro-', linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=1, label='FEM')

    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Position along beam (m)')
    ax2.set_ylabel('Bending Moment (kN·m)')
    ax2.set_title('Beam Bending Moment', fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Column moment example: pick first column group (if present)
    ax3 = axes[1, 0]
    if column_groups and errors['columns']:
        cg = column_groups[0]
        col = errors['columns'][cg.id]
        y_local = np.array(col['y_local'])
        anal_M = np.array(col['analytical']['M_kNm'])
        fem_M = np.array([v if v is not None else np.nan for v in col['fem']['M_kNm']])
        ax3.plot(anal_M, y_local, 'b-', linewidth=2, label='Analytical M')
        ax3.plot(fem_M, y_local, 'ro', label='FEM M')
        ax3.invert_yaxis()
        ax3.set_xlabel('Moment (kN·m)')
        ax3.set_ylabel('y from base (m)')
        ax3.set_title(f'Column {cg.id} Moment', fontweight='bold')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.5)
    else:
        ax3.text(0.5, 0.5, 'No column data available', ha='center', va='center')
        ax3.set_title('Column Moment', fontweight='bold')
        ax3.axis('off')

    # Error summary (reuse previous bar chart)
    ax4 = axes[1, 1]
    error_names = []
    error_values = []
    if 'reactions' in errors:
        error_names.append('Reaction Ry')
        error_values.append(errors['reactions']['R1y']['error_percent'])
    if 'deflection' in errors:
        error_names.append('Max Deflection')
        error_values.append(errors['deflection']['beam_max']['error_percent'])
    if 'moment' in errors:
        error_names.append('Beam Moment')
        error_values.append(errors['moment']['beam_center']['error_percent'])

    if error_names:
        bars = ax4.bar(error_names, error_values, alpha=0.8, edgecolor='black')
        for bar, val in zip(bars, error_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Error (%)')
        ax4.set_title('Error Summary', fontweight='bold')
        ax4.grid(True, linestyle='--', alpha=0.5, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No error data available', transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        ax4.set_title('Error Summary', fontweight='bold')
        ax4.axis('off')

    plt.tight_layout()
    output_file = f'{output_prefix}_comparison_extended.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    return fig


# ---------- Main (integrates everything) ----------

def main():
    VTK_FILE = "test_files/frame_2D_test_udl.gid/vtk_output/Parts_Beam_Beams_0_1.vtk"
    OUTPUT_PREFIX = "test_files/frame_2D_test_udl.gid/plots/frame_analysis"

    params = FrameParameters(
        L = 2.0,
        H = 2.0,
        w = 10000.0,
        P = 0.0,
        E = 210e9,
        I_beam = 5e-6,
        I_col = 5e-6,
        A_beam = 0.00287,
        A_col = 0.00287,
        frame_type = 'portal_pinned',
        load_type = 'udl_beam'
    )

    if not os.path.exists(VTK_FILE):
        print(f"Error: File '{VTK_FILE}' not found!")
        vtk_files = []
        for root, dirs, files in os.walk('.'):
            for f in files:
                if f.endswith('.vtk'):
                    vtk_files.append(os.path.join(root, f))
        if vtk_files:
            print('Available VTK files:')
            for f in vtk_files:
                print(' -', f)
        return

    print(f"Reading: {VTK_FILE}")
    vtk_data = parse_vtk_file(VTK_FILE)
    print(f" Nodes: {len(vtk_data.points)} | Elements: {len(vtk_data.cells)} | PointData: {list(vtk_data.point_data.keys())}")

    members, frame_info, column_groups = analyze_frame_geometry(vtk_data)

    # Update params L,H from geometry if they differ
    if abs(params.L - frame_info['span']) > 0.01:
        print(f"Updating span from {params.L} to {frame_info['span']}")
        params.L = frame_info['span']
    if abs(params.H - frame_info['height']) > 0.01:
        print(f"Updating height from {params.H} to {frame_info['height']}")
        params.H = frame_info['height']

    errors = calculate_frame_errors(vtk_data, members, params, frame_info, column_groups)

    # Print brief summary
    print('\nMembers:')
    for m in members:
        print(f" {m.id:>3}: type={m.member_type:<8} orient={m.orientation:<8} len={m.length:.3f} group={m.group_id}")

    print('\nColumn groups:')
    for cg in column_groups:
        print(f" Column {cg.id}: x={cg.x_coord:.3f}, nodes={cg.node_indices}, members={cg.member_ids}")

    # Generate plots
    print('\nGenerating extended plots...')
    plot_fem_vs_analytical_extended(vtk_data, members, errors, params, column_groups, OUTPUT_PREFIX)
    plot_column_diagrams(params, errors, vtk_data, column_groups, OUTPUT_PREFIX)

    print('\nDone.')


if __name__ == '__main__':
    main()


# === Combined Internal Force Diagram for Beam and Columns ===

def plot_combined_internal_forces(vtk_data, members, column_groups, params, output_prefix):
    """
    Plot a single comprehensive diagram showing:
      - Beam M, V, N
      - Column M, V, N (each column group)
    in a vertically stacked format similar to existing analytical diagrams.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # =============== BEAM DATA ===============
    beam_members = [m for m in members if m['type'] == 'beam']
    if beam_members:
        bm = beam_members[0]
        x = bm['fem']['x']
        M = bm['fem']['Mz']
        V = bm['fem']['Vy']
        N = bm['fem']['Nx']
        axes[0].plot(x, M, label='Beam M')
        axes[1].plot(x, V, label='Beam V')
        axes[2].plot(x, N, label='Beam N')

    # =============== COLUMN DATA (ALL GROUPS) ===============
    for g_idx, group in enumerate(column_groups):
        for col in group:
            cx = col['fem']['x']
            CM = col['fem']['Mz']
            CV = col['fem']['Vy']
            CN = col['fem']['Nx']

            axes[0].plot(cx, CM, '--', label=f'Column {g_idx+1} M')
            axes[1].plot(cx, CV, '--', label=f'Column {g_idx+1} V')
            axes[2].plot(cx, CN, '--', label=f'Column {g_idx+1} N')

    axes[0].set_ylabel("Moment M [Nm]")
    axes[1].set_ylabel("Shear V [N]")
    axes[2].set_ylabel("Axial N [N]")
    axes[2].set_xlabel("Local Member Coordinate [m]")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    out = output_prefix + "_combined_internal_forces.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Combined internal force diagram saved: {out}")
