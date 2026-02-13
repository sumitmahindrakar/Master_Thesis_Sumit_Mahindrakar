import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

VTK_FILE_PATH = "test_files/SA_Kratos_adj_V2.5/vtk_output_adjoint/Structure_0_1.vtk"  # Update this to your VTK file path

# Scale factors for visualization
DEFLECTION_SCALE = 10          # Scale factor for primal displacement
ROTATION_SCALE = 10             # Scale factor for primal rotation
ADJ_DEFLECTION_SCALE = 1        # Scale factor for adjoint displacement
ADJ_ROTATION_SCALE = 0.1        # Scale factor for adjoint rotation
SENSITIVITY_SCALE = 10           # Scale factor for shape sensitivity
CELL_SENSITIVITY_SCALE = 10     # Scale factor for cell sensitivity diagrams

# Colors for plotting
COLOR_STRUCTURE = 'black'
COLOR_DEFLECTION = 'blue'
COLOR_ROTATION = 'green'
COLOR_ADJ_DEFLECTION = 'purple'
COLOR_ADJ_ROTATION = 'darkorange'
COLOR_SENSITIVITY = 'red'
COLOR_I33_SENS = 'teal'
COLOR_YM_SENS = 'brown'

COLOR_FILL_DEFLECTION = 'lightblue'
COLOR_FILL_ROTATION = 'lightgreen'
COLOR_FILL_ADJ_DEFLECTION = 'plum'
COLOR_FILL_ADJ_ROTATION = 'moccasin'
COLOR_FILL_SENSITIVITY = 'lightcoral'
COLOR_FILL_I33_SENS = 'paleturquoise'
COLOR_FILL_YM_SENS = 'bisque'

# Line widths
LINEWIDTH_STRUCTURE = 2.0
LINEWIDTH_DIAGRAM = 1.5

# Figure settings
FIGURE_DPI = 100
SAVE_FIGURES = True
OUTPUT_FOLDER = "test_files/SA_Kratos_adj_V2.5/plots_adjoint_output_"


# =============================================================================
# VTK PARSER
# =============================================================================

def parse_vtk_file(filename):
    """
    Parse a VTK file and extract points, cells, and field data.
    Handles both multi-component and single-component fields.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"VTK file not found: {filename}")

    print(f"Reading VTK file: {filename}")

    with open(filename, 'r') as f:
        lines = f.readlines()

    data = {
        'points': None,
        'cells': None,
        'cell_types': None,
        'point_data': {},
        'cell_data': {}
    }

    i = 0
    current_section = None
    num_points = 0
    num_cells = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith('#'):
            i += 1
            continue

        # Parse header lines
        if line.startswith('vtk') or line.startswith('ASCII') or line.startswith('DATASET'):
            i += 1
            continue

        # ----- POINTS -----
        if line.startswith('POINTS'):
            parts = line.split()
            num_points = int(parts[1])
            points = []
            i += 1

            while len(points) < num_points and i < len(lines):
                values = lines[i].strip().split()
                for j in range(0, len(values), 3):
                    if j + 2 < len(values):
                        points.append([
                            float(values[j]),
                            float(values[j + 1]),
                            float(values[j + 2])
                        ])
                i += 1

            data['points'] = np.array(points[:num_points])
            print(f"  - Loaded {num_points} points")
            continue

        # ----- CELLS -----
        elif line.startswith('CELLS'):
            parts = line.split()
            num_cells = int(parts[1])
            cells = []
            i += 1

            for _ in range(num_cells):
                if i >= len(lines):
                    break
                values = lines[i].strip().split()
                cell_points = [int(v) for v in values[1:]]
                cells.append(cell_points)
                i += 1

            data['cells'] = cells
            print(f"  - Loaded {num_cells} cells")
            continue

        # ----- CELL_TYPES -----
        elif line.startswith('CELL_TYPES'):
            parts = line.split()
            num_types = int(parts[1])
            cell_types = []
            i += 1

            while len(cell_types) < num_types and i < len(lines):
                values = lines[i].strip().split()
                cell_types.extend([int(v) for v in values])
                i += 1

            data['cell_types'] = cell_types[:num_types]
            continue

        # ----- POINT_DATA / CELL_DATA section headers -----
        elif line.startswith('POINT_DATA'):
            current_section = 'point_data'
            i += 1
            continue

        elif line.startswith('CELL_DATA'):
            current_section = 'cell_data'
            i += 1
            continue

        # ----- FIELD declaration -----
        elif line.startswith('FIELD'):
            i += 1
            continue

        # ----- Parse field arrays -----
        elif current_section is not None:
            parts = line.split()

            if len(parts) >= 4:
                try:
                    field_name = parts[0]
                    num_components = int(parts[1])
                    num_tuples = int(parts[2])
                    # parts[3] is the data type (float, double, etc.)

                    field_data = []
                    i += 1

                    while len(field_data) < num_tuples and i < len(lines):
                        raw_line = lines[i].strip()
                        if not raw_line:
                            i += 1
                            continue

                        values = raw_line.split()

                        # Check if we've hit a new section header
                        if values[0] in ['POINT_DATA', 'CELL_DATA', 'FIELD',
                                         'SCALARS', 'VECTORS', 'CELLS',
                                         'CELL_TYPES']:
                            break

                        # For single-component data, each line has 1 value
                        # For multi-component data, each line has num_components values
                        if num_components == 1:
                            for v in values:
                                try:
                                    field_data.append([float(v)])
                                except ValueError:
                                    break
                                if len(field_data) >= num_tuples:
                                    break
                        else:
                            if len(values) >= num_components:
                                try:
                                    field_data.append(
                                        [float(v) for v in values[:num_components]]
                                    )
                                except ValueError:
                                    break

                        i += 1

                    if len(field_data) == num_tuples:
                        data[current_section][field_name] = np.array(field_data)
                        print(f"  - Loaded {current_section[:-5]} field: "
                              f"{field_name} ({num_tuples} tuples, "
                              f"{num_components} components)")
                    else:
                        print(f"  - WARNING: Field {field_name} expected "
                              f"{num_tuples} tuples but got {len(field_data)}")
                    continue

                except (ValueError, IndexError):
                    pass

        i += 1

    return data


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def get_element_direction(p1, p2):
    direction = np.array(p2) - np.array(p1)
    length = np.linalg.norm(direction)
    if length > 0:
        return direction / length
    return np.array([1, 0, 0])


def get_perpendicular_direction(p1, p2):
    """
    Get the perpendicular direction (rotated 90° CCW in the XY plane).
    For a horizontal beam: perpendicular points upward (+Y).
    """
    direction = get_element_direction(p1, p2)
    perpendicular = np.array([-direction[1], direction[0], 0])
    return perpendicular


def find_support_indices(points):
    x_coords = points[:, 0]
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    tolerance = (max_x - min_x) * 0.01 if max_x > min_x else 0.01

    left_indices = np.where(np.abs(x_coords - min_x) < tolerance)[0].tolist()
    right_indices = np.where(np.abs(x_coords - max_x) < tolerance)[0].tolist()

    return left_indices, right_indices


# =============================================================================
# PLOTTING: STRUCTURE & SUPPORTS
# =============================================================================

def plot_structure(ax, points, cells, color=COLOR_STRUCTURE,
                   linewidth=LINEWIDTH_STRUCTURE, label='Structure',
                   show_node_labels=False):
    for idx, cell in enumerate(cells):
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = points[p1_idx]
        p2 = points[p2_idx]

        if idx == 0:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=color, linewidth=linewidth, label=label)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=color, linewidth=linewidth)

    ax.scatter(points[:, 0], points[:, 1], color=color, s=30, zorder=5)

    if show_node_labels:
        for i, point in enumerate(points):
            ax.annotate(f'{i}', (point[0], point[1]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color='gray')


def add_supports_l(ax, points, support_indices=None):
    if support_indices is None:
        support_indices, _ = find_support_indices(points)
    if isinstance(support_indices, (int, np.integer)):
        support_indices = [support_indices]

    size = 0.03  # Scaled down for beam of length 1m

    for idx in support_indices:
        if idx >= len(points):
            continue
        x, y = points[idx][0], points[idx][1]
        ax.plot([x, x], [y - size, y + size], color='black', linewidth=1.5)
        for hy in np.linspace(y - size, y + size, 6):
            ax.plot([x, x - size * 0.5], [hy, hy - size * 0.5], 'k', lw=0.8)


def add_supports_r(ax, points, support_indices=None):
    if support_indices is None:
        _, support_indices = find_support_indices(points)
    if isinstance(support_indices, (int, np.integer)):
        support_indices = [support_indices]

    size = 0.03

    for idx in support_indices:
        if idx >= len(points):
            continue
        x, y = points[idx][0], points[idx][1]
        ax.plot([x, x], [y - size, y + size], color='black', linewidth=1.5)
        for hy in np.linspace(y - size, y + size, 6):
            ax.plot([x, x + size * 0.5], [hy, hy - size * 0.5], 'k', lw=0.8)


# =============================================================================
# PLOTTING: POINT-DATA DIAGRAMS (perpendicular to element)
# =============================================================================

def plot_perpendicular_point_diagram(ax, points, cells, field_data,
                                     component='y', scale=1.0,
                                     color='blue', fill_color='lightblue',
                                     linewidth=LINEWIDTH_DIAGRAM,
                                     show_fill=True,
                                     show_connecting_lines=True,
                                     show_values=False,
                                     value_fmt='.4e',
                                     label=None):
    """
    Generic function to plot a point-data field perpendicular to each element.

    component : 'y' → project Y-displacement onto perp direction
                'z_rot' → use the Z-component directly as scalar along perp
                'x' → project X-displacement onto perp direction
                'perp' → project full 2D vector onto perp direction
    """
    first = True
    for cell in cells:
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = points[p1_idx][:2]
        p2 = points[p2_idx][:2]
        perp = get_perpendicular_direction(p1, p2)[:2]

        if component == 'perp':
            # Project the 2D displacement vector onto perpendicular direction
            v1 = field_data[p1_idx][:2]
            v2 = field_data[p2_idx][:2]
            val1 = np.dot(v1, perp)
            val2 = np.dot(v2, perp)
        elif component == 'y':
            val1 = field_data[p1_idx][1]
            val2 = field_data[p2_idx][1]
        elif component == 'x':
            val1 = field_data[p1_idx][0]
            val2 = field_data[p2_idx][0]
        elif component == 'z_rot':
            val1 = field_data[p1_idx][2]
            val2 = field_data[p2_idx][2]
        else:
            val1 = field_data[p1_idx][1]
            val2 = field_data[p2_idx][1]

        offset_p1 = p1 + perp * val1 * scale
        offset_p2 = p2 + perp * val2 * scale

        lbl = label if first else None
        first = False

        ax.plot([offset_p1[0], offset_p2[0]],
                [offset_p1[1], offset_p2[1]],
                color=color, linewidth=linewidth, label=lbl)

        if show_connecting_lines:
            ax.plot([p1[0], offset_p1[0]], [p1[1], offset_p1[1]],
                    color=color, linewidth=0.5, linestyle='--', alpha=0.5)
            ax.plot([p2[0], offset_p2[0]], [p2[1], offset_p2[1]],
                    color=color, linewidth=0.5, linestyle='--', alpha=0.5)

        if show_fill:
            polygon_x = [p1[0], offset_p1[0], offset_p2[0], p2[0]]
            polygon_y = [p1[1], offset_p1[1], offset_p2[1], p2[1]]
            ax.fill(polygon_x, polygon_y, color=fill_color, alpha=0.3)

    # Annotate extreme values at nodes (only once per node)
    if show_values:
        already_annotated = set()
        for cell in cells:
            for nid in [cell[0], cell[1]]:
                if nid in already_annotated:
                    continue
                already_annotated.add(nid)

                p = points[nid][:2]
                perp_dir = np.array([0, 1])  # default up for horizontal beam

                if component in ('y', 'perp'):
                    val = field_data[nid][1]
                elif component == 'x':
                    val = field_data[nid][0]
                elif component == 'z_rot':
                    val = field_data[nid][2]
                else:
                    val = field_data[nid][1]

                if abs(val) > 1e-10:
                    offset_pt = p + perp_dir * val * scale
                    ax.annotate(f'{val:{value_fmt}}',
                                xy=(offset_pt[0], offset_pt[1]),
                                fontsize=7, color=color, ha='center',
                                va='bottom' if val * scale > 0 else 'top',
                                bbox=dict(boxstyle='round,pad=0.15',
                                          facecolor='white', alpha=0.7,
                                          edgecolor='none'))


# =============================================================================
# PLOTTING: CELL-DATA SCALAR DIAGRAMS
# =============================================================================

def plot_cell_scalar_diagram(ax, points, cells, cell_data,
                             scale=1.0, color='teal',
                             fill_color='paleturquoise',
                             linewidth=LINEWIDTH_DIAGRAM,
                             show_fill=True,
                             show_values=True,
                             show_connecting_lines=True,
                             value_fmt='.4e',
                             label=None):
    """
    Plot a single-component cell-data field as a constant value
    per element, drawn perpendicular to the element.
    """
    first = True
    for idx, cell in enumerate(cells):
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = points[p1_idx][:2]
        p2 = points[p2_idx][:2]
        perp = get_perpendicular_direction(p1, p2)[:2]

        val = cell_data[idx][0]  # single component

        offset_p1 = p1 + perp * val * scale
        offset_p2 = p2 + perp * val * scale

        lbl = label if first else None
        first = False

        # Diagram line (constant along element)
        ax.plot([offset_p1[0], offset_p2[0]],
                [offset_p1[1], offset_p2[1]],
                color=color, linewidth=linewidth, label=lbl)

        if show_connecting_lines:
            ax.plot([p1[0], offset_p1[0]], [p1[1], offset_p1[1]],
                    color=color, linewidth=0.5, linestyle='-', alpha=0.5)
            ax.plot([p2[0], offset_p2[0]], [p2[1], offset_p2[1]],
                    color=color, linewidth=0.5, linestyle='-', alpha=0.5)

        if show_fill:
            polygon_x = [p1[0], offset_p1[0], offset_p2[0], p2[0]]
            polygon_y = [p1[1], offset_p1[1], offset_p2[1], p2[1]]
            ax.fill(polygon_x, polygon_y, color=fill_color, alpha=0.3)

        if show_values:
            mid = (p1 + p2) / 2
            mid_offset = mid + perp * val * scale
            ax.annotate(f'{val:{value_fmt}}',
                        xy=(mid_offset[0], mid_offset[1]),
                        fontsize=7, color=color, ha='center',
                        va='bottom' if val * scale > 0 else 'top',
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', alpha=0.7,
                                  edgecolor='none'))


# =============================================================================
# HELPER: SET UP AXES
# =============================================================================

def setup_axes(ax, points, title, x_margin=0.15, y_margin=0.15):
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    y_range = max(y_max - y_min, 0.01)
    ax.set_ylim(y_min - y_range * 3 - y_margin, y_max + y_range * 3 + y_margin)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


# =============================================================================
# MAIN PLOT CREATION
# =============================================================================

def create_all_plots(vtk_data, save_figures=SAVE_FIGURES,
                     output_folder=OUTPUT_FOLDER):
    points = vtk_data['points']
    cells = vtk_data['cells']
    left_supports, right_supports = find_support_indices(points)

    print(f"\n  Detected left supports at nodes:  {left_supports}")
    print(f"  Detected right supports at nodes: {right_supports}")

    if save_figures and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"  Created output folder: {output_folder}")

    # Extract fields (with safe defaults)
    n_pts = len(points)
    n_cells = len(cells)

    displacement = vtk_data['point_data'].get(
        'DISPLACEMENT', np.zeros((n_pts, 3)))
    rotation = vtk_data['point_data'].get(
        'ROTATION', np.zeros((n_pts, 3)))
    adj_displacement = vtk_data['point_data'].get(
        'ADJOINT_DISPLACEMENT', np.zeros((n_pts, 3)))
    adj_rotation = vtk_data['point_data'].get(
        'ADJOINT_ROTATION', np.zeros((n_pts, 3)))
    shape_sens = vtk_data['point_data'].get(
        'SHAPE_SENSITIVITY', np.zeros((n_pts, 3)))

    i33_sens = vtk_data['cell_data'].get(
        'I33_SENSITIVITY', np.zeros((n_cells, 1)))
    ym_sens = vtk_data['cell_data'].get(
        'YOUNG_MODULUS_SENSITIVITY', np.zeros((n_cells, 1)))

    # =====================================================================
    # PLOT 1: OVERVIEW – 6 subplots
    # =====================================================================
    fig, axes = plt.subplots(3, 2, figsize=(18, 18), dpi=FIGURE_DPI)

    # ---- (0,0) Structure ----
    ax = axes[0, 0]
    setup_axes(ax, points, 'Frame Structure')
    plot_structure(ax, points, cells, show_node_labels=True)
    add_supports_l(ax, points, left_supports)
    add_supports_r(ax, points, right_supports)
    ax.legend(loc='upper right')

    # ---- (0,1) Displacement ----
    ax = axes[0, 1]
    setup_axes(ax, points,
               f'Displacement (scale {DEFLECTION_SCALE}x)')
    plot_structure(ax, points, cells, color='gray', linewidth=1.0,
                   label='Structure')
    plot_perpendicular_point_diagram(
        ax, points, cells, displacement,
        component='perp', scale=DEFLECTION_SCALE,
        color=COLOR_DEFLECTION, fill_color=COLOR_FILL_DEFLECTION,
        label='Displacement')
    add_supports_l(ax, points, left_supports)
    add_supports_r(ax, points, right_supports)
    ax.legend(loc='upper right', fontsize=9)

    # ---- (1,0) Rotation ----
    ax = axes[1, 0]
    setup_axes(ax, points,
               f'Rotation (scale {ROTATION_SCALE}x)')
    plot_structure(ax, points, cells, color='gray', linewidth=1.0,
                   label='Structure')
    plot_perpendicular_point_diagram(
        ax, points, cells, rotation,
        component='z_rot', scale=ROTATION_SCALE,
        color=COLOR_ROTATION, fill_color=COLOR_FILL_ROTATION,
        label='Rotation')
    add_supports_l(ax, points, left_supports)
    add_supports_r(ax, points, right_supports)
    ax.legend(loc='upper right', fontsize=9)

    # ---- (1,1) Adjoint Displacement ----
    ax = axes[1, 1]
    setup_axes(ax, points,
               f'Adjoint Displacement (scale {ADJ_DEFLECTION_SCALE}x)')
    plot_structure(ax, points, cells, color='gray', linewidth=1.0,
                   label='Structure')
    plot_perpendicular_point_diagram(
        ax, points, cells, adj_displacement,
        component='perp', scale=ADJ_DEFLECTION_SCALE,
        color=COLOR_ADJ_DEFLECTION, fill_color=COLOR_FILL_ADJ_DEFLECTION,
        label='Adj. Displacement')
    add_supports_l(ax, points, left_supports)
    add_supports_r(ax, points, right_supports)
    ax.legend(loc='upper right', fontsize=9)

    # ---- (2,0) Adjoint Rotation ----
    ax = axes[2, 0]
    setup_axes(ax, points,
               f'Adjoint Rotation (scale {ADJ_ROTATION_SCALE}x)')
    plot_structure(ax, points, cells, color='gray', linewidth=1.0,
                   label='Structure')
    plot_perpendicular_point_diagram(
        ax, points, cells, adj_rotation,
        component='z_rot', scale=ADJ_ROTATION_SCALE,
        color=COLOR_ADJ_ROTATION, fill_color=COLOR_FILL_ADJ_ROTATION,
        label='Adj. Rotation')
    add_supports_l(ax, points, left_supports)
    add_supports_r(ax, points, right_supports)
    ax.legend(loc='upper right', fontsize=9)

    # ---- (2,1) I33 Sensitivity (cell scalar) ----
    ax = axes[2, 1]
    setup_axes(ax, points,
               f'I33 Sensitivity (scale {CELL_SENSITIVITY_SCALE}x)')
    plot_structure(ax, points, cells, color='gray', linewidth=1.0,
                   label='Structure')
    plot_cell_scalar_diagram(
        ax, points, cells, i33_sens,
        scale=CELL_SENSITIVITY_SCALE,
        color=COLOR_I33_SENS, fill_color=COLOR_FILL_I33_SENS,
        show_values=False, label='I33 Sensitivity')
    add_supports_l(ax, points, left_supports)
    add_supports_r(ax, points, right_supports)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    if save_figures:
        fp = os.path.join(output_folder, 'overview_6panel.png')
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fp}")
    plt.show()

    # =====================================================================
    # PLOT 2: Detailed Displacement
    # =====================================================================
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)
    setup_axes(ax, points,
               f'Displacement Diagram (scale {DEFLECTION_SCALE}x)')
    plot_structure(ax, points, cells, color='gray', linewidth=1.0)
    plot_perpendicular_point_diagram(
        ax, points, cells, displacement,
        component='perp', scale=DEFLECTION_SCALE,
        color=COLOR_DEFLECTION, fill_color=COLOR_FILL_DEFLECTION,
        show_values=True, value_fmt='.4e',
        label='Displacement')
    add_supports_l(ax, points, left_supports)
    add_supports_r(ax, points, right_supports)
    max_d = np.max(np.abs(displacement[:, :2]))
    ax.text(0.02, 0.95,
            f'Max |disp| = {max_d:.6e} m\nScale = {DEFLECTION_SCALE}x',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save_figures:
        fp = os.path.join(output_folder, 'displacement_detail.png')
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fp}")
    plt.show()

    # =====================================================================
    # PLOT 3: Detailed Rotation
    # =====================================================================
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)
    setup_axes(ax, points,
               f'Rotation Diagram (scale {ROTATION_SCALE}x)')
    plot_structure(ax, points, cells, color='gray', linewidth=1.0)
    plot_perpendicular_point_diagram(
        ax, points, cells, rotation,
        component='z_rot', scale=ROTATION_SCALE,
        color=COLOR_ROTATION, fill_color=COLOR_FILL_ROTATION,
        show_values=True, value_fmt='.4e',
        label='Rotation')
    add_supports_l(ax, points, left_supports)
    add_supports_r(ax, points, right_supports)
    max_r = np.max(np.abs(rotation[:, 2]))
    ax.text(0.02, 0.95,
            f'Max |rot_z| = {max_r:.6e} rad\nScale = {ROTATION_SCALE}x',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save_figures:
        fp = os.path.join(output_folder, 'rotation_detail.png')
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fp}")
    plt.show()

    # =====================================================================
    # PLOT 4: Detailed Adjoint Displacement
    # =====================================================================
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)
    setup_axes(ax, points,
               f'Adjoint Displacement (scale {ADJ_DEFLECTION_SCALE}x)')
    plot_structure(ax, points, cells, color='gray', linewidth=1.0)
    plot_perpendicular_point_diagram(
        ax, points, cells, adj_displacement,
        component='perp', scale=ADJ_DEFLECTION_SCALE,
        color=COLOR_ADJ_DEFLECTION, fill_color=COLOR_FILL_ADJ_DEFLECTION,
        show_values=True, value_fmt='.4e',
        label='Adj. Displacement')
    add_supports_l(ax, points, left_supports)
    add_supports_r(ax, points, right_supports)
    max_ad = np.max(np.abs(adj_displacement[:, :2]))
    ax.text(0.02, 0.95,
            f'Max |adj_disp| = {max_ad:.6e}\nScale = {ADJ_DEFLECTION_SCALE}x',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save_figures:
        fp = os.path.join(output_folder, 'adjoint_displacement_detail.png')
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fp}")
    plt.show()

    # =====================================================================
    # PLOT 5: Detailed Adjoint Rotation
    # =====================================================================
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)
    setup_axes(ax, points,
               f'Adjoint Rotation (scale {ADJ_ROTATION_SCALE}x)')
    plot_structure(ax, points, cells, color='gray', linewidth=1.0)
    plot_perpendicular_point_diagram(
        ax, points, cells, adj_rotation,
        component='z_rot', scale=ADJ_ROTATION_SCALE,
        color=COLOR_ADJ_ROTATION, fill_color=COLOR_FILL_ADJ_ROTATION,
        show_values=True, value_fmt='.4e',
        label='Adj. Rotation')
    add_supports_l(ax, points, left_supports)
    add_supports_r(ax, points, right_supports)
    max_ar = np.max(np.abs(adj_rotation[:, 2]))
    ax.text(0.02, 0.95,
            f'Max |adj_rot_z| = {max_ar:.6e}\nScale = {ADJ_ROTATION_SCALE}x',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save_figures:
        fp = os.path.join(output_folder, 'adjoint_rotation_detail.png')
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fp}")
    plt.show()

    # =====================================================================
    # PLOT 6: Shape Sensitivity (X component – dominant at supports)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)
    setup_axes(ax, points,
               f'Shape Sensitivity – X component (scale {SENSITIVITY_SCALE}x)')
    plot_structure(ax, points, cells, color='gray', linewidth=1.0)
    plot_perpendicular_point_diagram(
        ax, points, cells, shape_sens,
        component='x', scale=SENSITIVITY_SCALE,
        color=COLOR_SENSITIVITY, fill_color=COLOR_FILL_SENSITIVITY,
        show_values=True, value_fmt='.4e',
        label='Shape Sens. (X)')
    add_supports_l(ax, points, left_supports)
    add_supports_r(ax, points, right_supports)
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save_figures:
        fp = os.path.join(output_folder, 'shape_sensitivity_X_detail.png')
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fp}")
    plt.show()

    # =====================================================================
    # PLOT 7: I33 Sensitivity (cell scalar)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)
    setup_axes(ax, points,
               f'I33 Sensitivity (scale {CELL_SENSITIVITY_SCALE}x)')
    plot_structure(ax, points, cells, color='gray', linewidth=1.0)
    plot_cell_scalar_diagram(
        ax, points, cells, i33_sens,
        scale=CELL_SENSITIVITY_SCALE,
        color=COLOR_I33_SENS, fill_color=COLOR_FILL_I33_SENS,
        show_values=True, value_fmt='.4e',
        label='I33 Sensitivity')
    add_supports_l(ax, points, left_supports)
    add_supports_r(ax, points, right_supports)
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save_figures:
        fp = os.path.join(output_folder, 'I33_sensitivity_detail.png')
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fp}")
    plt.show()

    # =====================================================================
    # PLOT 8: Young's Modulus Sensitivity (cell scalar)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)
    setup_axes(ax, points,
               f"Young's Modulus Sensitivity (scale {CELL_SENSITIVITY_SCALE}x)")
    plot_structure(ax, points, cells, color='gray', linewidth=1.0)
    plot_cell_scalar_diagram(
        ax, points, cells, ym_sens,
        scale=CELL_SENSITIVITY_SCALE,
        color=COLOR_YM_SENS, fill_color=COLOR_FILL_YM_SENS,
        show_values=True, value_fmt='.4e',
        label='E Sensitivity')
    add_supports_l(ax, points, left_supports)
    add_supports_r(ax, points, right_supports)
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save_figures:
        fp = os.path.join(output_folder, 'YM_sensitivity_detail.png')
        plt.savefig(fp, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fp}")
    plt.show()


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(vtk_data):
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    points = vtk_data['points']
    cells = vtk_data['cells']

    print(f"\nGeometry:")
    print(f"  Nodes:    {len(points)}")
    print(f"  Elements: {len(cells)}")
    print(f"  X range:  [{points[:, 0].min():.4f}, {points[:, 0].max():.4f}]")
    print(f"  Y range:  [{points[:, 1].min():.4f}, {points[:, 1].max():.4f}]")

    print(f"\nPoint Data Fields:")
    for name, data in vtk_data['point_data'].items():
        print(f"  {name:30s}  shape {str(data.shape):>12s}"
              f"  range [{data.min():.4e}, {data.max():.4e}]")

    print(f"\nCell Data Fields:")
    for name, data in vtk_data['cell_data'].items():
        print(f"  {name:30s}  shape {str(data.shape):>12s}"
              f"  range [{data.min():.4e}, {data.max():.4e}]")

    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("VTK ADJOINT STRUCTURE PLOTTER")
    print("=" * 60)

    if len(sys.argv) > 1:
        vtk_file = sys.argv[1]
    else:
        vtk_file = VTK_FILE_PATH

    try:
        print(f"\n1. Loading VTK file: {vtk_file}")
        vtk_data = parse_vtk_file(vtk_file)

        print_summary(vtk_data)

        print("\n2. Creating plots...")
        create_all_plots(vtk_data,
                         save_figures=SAVE_FIGURES,
                         output_folder=OUTPUT_FOLDER)

        print("\nDone!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Set VTK_FILE_PATH at the top of the script, or run:")
        print("  python plot_adjoint.py  your_file.vtk")
        sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)