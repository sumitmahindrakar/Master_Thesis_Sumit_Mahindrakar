"""
VTK Beam Plotter - For simply supported / fixed beams along X-axis
Plots: Structure, Deformed Shape, BMD (discrete bars), Sensitivity (discrete bars)
Auto-detects supports from boundary displacements.
Reads traced_element_id and stress_location from adjoint JSON settings.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon, Circle, FancyArrowPatch
import matplotlib.patches as mpatches
import os
import json
import glob


def load_adjoint_settings(json_path):
    """
    Load adjoint JSON and extract traced_element_id and stress_location
    from solver_settings.response_function_settings.
    Returns (traced_element_id, stress_location) as integers.
    """
    with open(json_path, 'r') as f:
        settings = json.load(f)

    response_settings = (
        settings
        .get("solver_settings", {})
        .get("response_function_settings", {})
    )

    traced_element_id = response_settings.get("traced_element_id", None)
    stress_location = response_settings.get("stress_location", None)
    stress_type = response_settings.get("stress_type", None)

    if traced_element_id is None:
        raise KeyError(
            f"'traced_element_id' not found in "
            f"solver_settings.response_function_settings of {json_path}"
        )
    if stress_location is None:
        raise KeyError(
            f"'stress_location' not found in "
            f"solver_settings.response_function_settings of {json_path}"
        )

    print(f"[JSON] Loaded from: {json_path}")
    print(f"       traced_element_id = {traced_element_id}")
    print(f"       stress_location   = {stress_location}")
    if stress_type:
        print(f"       stress_type       = {stress_type}")

    return int(traced_element_id), int(stress_location), stress_type


def find_adjoint_json(search_dir="."):
    """
    Auto-detect the adjoint settings JSON file.
    Looks for files matching common naming patterns.
    Returns the path to the first match, or None.
    """
    # Common naming patterns for adjoint parameter files
    patterns = [
        "ProjectParametersAdjoint*.json",
        "project_parameters_adjoint*.json",
        "*adjoint*.json",
        "*Adjoint*.json",
    ]

    for pattern in patterns:
        matches = glob.glob(os.path.join(search_dir, pattern))
        if matches:
            return matches[0]

    return None


def read_vtk_file(filename):
    """Read VTK ASCII file and extract all data."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = {'points': [], 'cells': [], 'point_data': {}, 'cell_data': {}}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('POINTS'):
            n_points = int(line.split()[1])
            i += 1
            coords = []
            while len(coords) < n_points * 3:
                coords.extend([float(v) for v in lines[i].strip().split()])
                i += 1
            data['points'] = np.array(coords).reshape(n_points, 3)
            continue
        elif line.startswith('CELLS'):
            parts = line.split()
            n_cells = int(parts[1])
            i += 1
            for _ in range(n_cells):
                values = lines[i].strip().split()
                n_nodes = int(values[0])
                data['cells'].append([int(v) for v in values[1:n_nodes + 1]])
                i += 1
            continue
        elif line.startswith('POINT_DATA'):
            i += 1
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('CELL_DATA'):
                    break
                if line.startswith('FIELD'):
                    n_fields = int(line.split()[-1])
                    i += 1
                    for _ in range(n_fields):
                        parts = lines[i].strip().split()
                        name, n_comp, n_tuples = parts[0], int(parts[1]), int(parts[2])
                        i += 1
                        vals = []
                        while len(vals) < n_tuples * n_comp:
                            vals.extend([float(v) for v in lines[i].strip().split()])
                            i += 1
                        arr = np.array(vals)
                        data['point_data'][name] = arr.reshape(n_tuples, n_comp) if n_comp > 1 else arr
                    continue
                i += 1
            continue
        elif line.startswith('CELL_DATA'):
            i += 1
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('POINT_DATA'):
                    break
                if line.startswith('FIELD'):
                    n_fields = int(line.split()[-1])
                    i += 1
                    for _ in range(n_fields):
                        parts = lines[i].strip().split()
                        name, n_comp, n_tuples = parts[0], int(parts[1]), int(parts[2])
                        i += 1
                        vals = []
                        while len(vals) < n_tuples * n_comp:
                            vals.extend([float(v) for v in lines[i].strip().split()])
                            i += 1
                        arr = np.array(vals)
                        data['cell_data'][name] = arr.reshape(n_tuples, n_comp) if n_comp > 1 else arr
                    continue
                i += 1
            continue
        i += 1
    return data


def detect_support_nodes_beam(points):
    """Detect supports for a beam: nodes at min X and max X."""
    x_vals = points[:, 0]
    x_min, x_max = x_vals.min(), x_vals.max()
    supports = []
    for i in range(len(points)):
        if abs(x_vals[i] - x_min) < 1e-6 or abs(x_vals[i] - x_max) < 1e-6:
            supports.append(i)
    return supports


def detect_support_type(data, support_nodes):
    """
    Detect if a support is pinned, roller, or fixed based on reaction moments.
    Returns dict: {node_index: 'fixed' | 'pinned' | 'roller'}
    """
    support_types = {}
    reaction_moment = data['point_data'].get('REACTION_MOMENT', None)

    x_vals = data['points'][:, 0]
    x_min, x_max = x_vals.min(), x_vals.max()

    for sn in support_nodes:
        has_moment = False
        if reaction_moment is not None:
            mom = reaction_moment[sn]
            if np.max(np.abs(mom)) > 1e-3:
                has_moment = True

        if has_moment:
            support_types[sn] = 'fixed'
        else:
            if abs(x_vals[sn] - x_min) < 1e-6:
                support_types[sn] = 'pinned'
            else:
                support_types[sn] = 'roller'

    return support_types


def format_value(val):
    abs_val = abs(val)
    if abs_val >= 1e6:
        return f'{val / 1e6:.1f}M'
    elif abs_val >= 1e3:
        return f'{val / 1e3:.1f}K'
    elif abs_val >= 1:
        return f'{val:.2f}'
    elif abs_val >= 1e-3:
        return f'{val:.4f}'
    else:
        return f'{val:.2e}'


def draw_support_pinned(ax, x, y, size=0.03):
    """Draw pinned (triangle) support."""
    s = size
    tri = Polygon(
        [[x - s, y - 1.5 * s], [x + s, y - 1.5 * s], [x, y]],
        closed=True, facecolor='gray', edgecolor='black',
        linewidth=1.5, zorder=6)
    ax.add_patch(tri)
    ax.plot([x - 1.3 * s, x + 1.3 * s], [y - 1.5 * s, y - 1.5 * s],
            'k-', linewidth=2)
    for hx in np.linspace(x - 1.2 * s, x + 1.2 * s, 5):
        ax.plot([hx, hx - 0.4 * s], [y - 1.5 * s, y - 2.1 * s],
                'k-', linewidth=0.7)


def draw_support_roller(ax, x, y, size=0.03):
    """Draw roller (triangle + circle) support."""
    s = size
    tri = Polygon(
        [[x - s, y - 1.5 * s], [x + s, y - 1.5 * s], [x, y]],
        closed=True, facecolor='lightgray', edgecolor='black',
        linewidth=1.5, zorder=6)
    ax.add_patch(tri)
    circle_r = 0.3 * s
    ax.add_patch(Circle((x - 0.6 * s, y - 1.5 * s - circle_r), circle_r,
                         facecolor='white', edgecolor='black', linewidth=1, zorder=6))
    ax.add_patch(Circle((x + 0.6 * s, y - 1.5 * s - circle_r), circle_r,
                         facecolor='white', edgecolor='black', linewidth=1, zorder=6))
    ax.add_patch(Circle((x, y - 1.5 * s - circle_r), circle_r,
                         facecolor='white', edgecolor='black', linewidth=1, zorder=6))
    ground_y = y - 1.5 * s - 2 * circle_r
    ax.plot([x - 1.3 * s, x + 1.3 * s], [ground_y, ground_y],
            'k-', linewidth=2)
    for hx in np.linspace(x - 1.2 * s, x + 1.2 * s, 5):
        ax.plot([hx, hx - 0.4 * s], [ground_y, ground_y - 0.6 * s],
                'k-', linewidth=0.7)


def draw_support_fixed(ax, x, y, size=0.03, side='left'):
    """Draw fixed (wall) support."""
    s = size
    wall_x = x - 0.3 * s if side == 'left' else x + 0.3 * s
    ax.plot([wall_x, wall_x], [y - 2 * s, y + 2 * s],
            'k-', linewidth=3, zorder=6)
    for hz in np.linspace(y - 1.8 * s, y + 1.8 * s, 6):
        dx = -0.5 * s if side == 'left' else 0.5 * s
        ax.plot([wall_x, wall_x + dx], [hz, hz - 0.4 * s],
                'k-', linewidth=0.7)


def draw_supports(ax, points, support_nodes, support_types, size=0.03):
    """Draw support symbols based on detected types."""
    x_vals = points[:, 0]
    x_min, x_max = x_vals.min(), x_vals.max()

    for sn in support_nodes:
        x = points[sn][0]
        y = 0.0

        stype = support_types.get(sn, 'pinned')

        if stype == 'fixed':
            side = 'left' if abs(x - x_min) < 1e-6 else 'right'
            draw_support_fixed(ax, x, y, size, side)
        elif stype == 'roller':
            draw_support_roller(ax, x, y, size)
        else:
            draw_support_pinned(ax, x, y, size)


def draw_beam(ax, points, cells, lw=4, color='black'):
    """Draw beam as a line along X-axis."""
    for cell in cells:
        p1, p2 = points[cell[0]], points[cell[1]]
        ax.plot([p1[0], p2[0]], [0, 0], '-', color=color,
                linewidth=lw, zorder=4, solid_capstyle='round')
    for pt in points:
        ax.plot(pt[0], 0, 'ko', markersize=3, zorder=5)


def draw_udl_on_beam(ax, points, arrow_len=0.04, n_arrows=15):
    """Draw UDL arrows on beam."""
    x_min = points[:, 0].min()
    x_max = points[:, 0].max()
    span = x_max - x_min

    udl_top = arrow_len * 1.2
    ax.plot([x_min, x_max], [udl_top, udl_top], 'b-', linewidth=1.5, alpha=0.7)

    for xx in np.linspace(x_min + span * 0.02, x_max - span * 0.02, n_arrows):
        ax.annotate('', xy=(xx, 0), xytext=(xx, udl_top),
                    arrowprops=dict(arrowstyle='->', color='blue',
                                    lw=1.0, alpha=0.6))


def draw_discrete_bar_beam(ax, x1, x2, value, scale,
                           pos_color='#3366cc', neg_color='#cc3333', alpha=0.4):
    """Draw constant-value bar perpendicular to beam (vertical)."""
    off = value * scale
    color = pos_color if value >= 0 else neg_color

    poly_x = [x1, x2, x2, x1]
    poly_y = [0, 0, off, off]
    ax.add_patch(Polygon(list(zip(poly_x, poly_y)), closed=True,
                         facecolor=color, edgecolor='none', alpha=alpha, zorder=2))
    ax.plot([x1, x2], [off, off], color=color, linewidth=1.2, alpha=0.7, zorder=3)
    ax.plot([x1, x1], [0, off], color='gray', linewidth=0.6, alpha=0.5, zorder=3)
    ax.plot([x2, x2], [0, off], color='gray', linewidth=0.6, alpha=0.5, zorder=3)


def draw_response_location_beam(ax, points, cells, traced_idx, stress_location,
                                ring_radius=0.02):
    """Draw response location ring for beam."""
    cell = cells[traced_idx]
    resp_node = cell[1] if stress_location == 2 else cell[0]
    rx = points[resp_node][0]
    ry = 0.0

    ax.add_patch(Circle((rx, ry), ring_radius, fill=False,
                        edgecolor='darkorange', linewidth=2.5, zorder=8))
    ax.add_patch(Circle((rx, ry), ring_radius * 0.55, fill=False,
                        edgecolor='darkorange', linewidth=1.5,
                        linestyle='--', zorder=8))
    ax.plot(rx, ry, 'o', color='darkorange', markersize=2, zorder=9)

    ax.annotate(f'Response\n(Node {resp_node + 1})',
                xy=(rx, ry),
                xytext=(rx + 0.05, ry + 0.06),
                fontsize=8, fontweight='bold', color='darkorange', ha='center',
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5,
                                connectionstyle='arc3,rad=0.2'),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                          edgecolor='darkorange', linewidth=1.2, alpha=0.95),
                zorder=10)


def setup_beam_ax(ax, points, support_nodes, support_types, beam_height=0.08):
    """Set axis for beam plot."""
    x_min = points[:, 0].min()
    x_max = points[:, 0].max()
    span = x_max - x_min
    pad_x = span * 0.15

    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)

    support_size = span * 0.04
    draw_supports(ax, points, support_nodes, support_types, size=support_size)


# =====================================================================
#  PLOT 1: Structure with Loads
# =====================================================================
def plot_structure(data, save_name=None):
    points = data['points']
    cells = data['cells']
    support_nodes = detect_support_nodes_beam(points)
    support_types = detect_support_type(data, support_nodes)

    span = points[:, 0].max() - points[:, 0].min()

    fig, ax = plt.subplots(figsize=(12, 4))

    draw_beam(ax, points, cells, lw=5)
    draw_udl_on_beam(ax, points, arrow_len=span * 0.06, n_arrows=20)

    for idx, pt in enumerate(points):
        if idx % 2 == 0 or idx == len(points) - 1:
            ax.annotate(f'{idx + 1}', xy=(pt[0], 0),
                        xytext=(0, -span * 0.08), textcoords='offset points',
                        fontsize=6, color='blue', fontweight='bold', ha='center')

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    dim_y = -span * 0.12
    ax.annotate('', xy=(x_max, dim_y), xytext=(x_min, dim_y),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2))
    ax.text((x_min + x_max) / 2, dim_y - span * 0.02,
            f'L = {x_max - x_min:.1f} m', ha='center', fontsize=9, color='gray')

    ax.annotate('q = 40 N/m ↓', xy=((x_min + x_max) / 2, span * 0.1),
                fontsize=10, ha='center', color='blue', fontweight='bold')

    setup_beam_ax(ax, points, support_nodes, support_types)
    y_range = span * 0.25
    ax.set_ylim(-y_range, y_range)
    ax.set_ylabel('')

    for sn in support_nodes:
        stype = support_types.get(sn, 'pinned')
        ax.annotate(stype.capitalize(),
                    xy=(points[sn][0], -span * 0.06),
                    fontsize=7, ha='center', color='gray', fontstyle='italic')

    ax.set_title('Simply Supported Beam with UDL',
                 fontsize=14, fontweight='bold', pad=12)

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


# =====================================================================
#  PLOT 2: Deformed Shape
# =====================================================================
def plot_deformed(data, traced_elem_id=6, stress_location=2, save_name=None):
    points = data['points']
    cells = data['cells']
    disp = data['point_data']['DISPLACEMENT']
    disp_Z = disp[:, 2]
    support_nodes = detect_support_nodes_beam(points)
    support_types = detect_support_type(data, support_nodes)
    span = points[:, 0].max() - points[:, 0].min()

    fig, ax = plt.subplots(figsize=(12, 5))

    draw_beam(ax, points, cells, lw=3, color='gray')

    max_disp = np.max(np.abs(disp_Z))
    amp = span * 0.15 / max_disp if max_disp > 0 else 1

    x_def = points[:, 0]
    z_def = disp_Z * amp

    for cell in cells:
        i1, i2 = cell[0], cell[1]
        ax.plot([x_def[i1], x_def[i2]], [z_def[i1], z_def[i2]],
                'r-', linewidth=3, alpha=0.8, zorder=4)

    vmax = max(abs(disp_Z.max()), abs(disp_Z.min()))
    if vmax == 0:
        vmax = 1
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    scatter = ax.scatter(x_def, z_def, c=disp_Z,
                         cmap=cm.coolwarm, norm=norm, s=40,
                         edgecolors='black', linewidth=0.8, zorder=5)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02, aspect=20)
    cbar.set_label('Displacement Z (m)', fontsize=10)

    for i in range(len(points)):
        if abs(disp_Z[i]) > max_disp * 0.05:
            ax.plot([x_def[i], x_def[i]], [0, z_def[i]],
                    'k:', linewidth=0.5, alpha=0.3)

    max_idx = np.argmin(disp_Z)
    ax.annotate(f'Max: {disp_Z[max_idx]:.4e} m\nNode {max_idx + 1}',
                xy=(x_def[max_idx], z_def[max_idx]),
                xytext=(x_def[max_idx] + span * 0.1, z_def[max_idx] - span * 0.05),
                fontsize=9, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose',
                          edgecolor='red', alpha=0.9))

    draw_response_location_beam(ax, points, cells, traced_elem_id - 1,
                                stress_location, ring_radius=span * 0.015)

    orig = mpatches.Patch(facecolor='gray', alpha=0.5, label='Original')
    defd = mpatches.Patch(facecolor='red', alpha=0.8, label=f'Deformed (×{amp:.0f})')
    ax.legend(handles=[orig, defd], loc='upper right', fontsize=9)

    setup_beam_ax(ax, points, support_nodes, support_types)
    y_range = span * 0.3
    ax.set_ylim(-y_range, y_range * 0.6)
    ax.set_ylabel('Displacement (amplified)', fontsize=10)

    ax.set_title(f'Deformed Shape (Amplified ×{amp:.0f})',
                 fontsize=14, fontweight='bold', pad=12)

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


# =====================================================================
#  PLOT 3: BMD — Discrete Bars (FIXED)
# =====================================================================
def plot_bmd(data, traced_elem_id=6, stress_location=2,
             stress_type="MY", save_name=None):
    points = data['points']
    cells = data['cells']

    moment_data = data['cell_data']['MOMENT']

    if moment_data.ndim == 2:
        # Find the column with actual non-zero moment data
        # Map stress_type to preferred column index
        moment_component_map = {"MX": 0, "MY": 1, "MZ": 2}
        col = moment_component_map.get(stress_type, 1)
        moment_values = moment_data[:, col]

        # AUTO-DETECT: if selected column is all zeros, find the non-zero one
        if np.max(np.abs(moment_values)) < 1e-10:
            print(f"  [WARNING] {stress_type} (column {col}) is all zeros.")
            col_magnitudes = [np.max(np.abs(moment_data[:, c]))
                              for c in range(moment_data.shape[1])]
            best_col = int(np.argmax(col_magnitudes))
            col_names = {0: "MX", 1: "MY", 2: "MZ"}

            if col_magnitudes[best_col] > 1e-10:
                print(f"  [AUTO-FIX] Using {col_names[best_col]} (column {best_col}) "
                      f"which has max = {col_magnitudes[best_col]:.4f}")
                moment_values = moment_data[:, best_col]
                # Update label for plot (keep original stress_type for title context)
                active_moment_label = col_names[best_col]
            else:
                print(f"  [WARNING] All moment columns are zero!")
                active_moment_label = stress_type
        else:
            active_moment_label = stress_type
    else:
        moment_values = moment_data
        active_moment_label = stress_type

    support_nodes = detect_support_nodes_beam(points)
    support_types = detect_support_type(data, support_nodes)
    span = points[:, 0].max() - points[:, 0].min()

    fig, ax = plt.subplots(figsize=(12, 6))

    draw_beam(ax, points, cells, lw=3)

    # Scale bars
    max_moment = np.max(np.abs(moment_values))
    bm_scale = span * 0.15 / max_moment if max_moment > 0 else 1

    for idx, cell in enumerate(cells):
        x1 = points[cell[0]][0]
        x2 = points[cell[1]][0]
        draw_discrete_bar_beam(ax, x1, x2, moment_values[idx], bm_scale)

    # Value labels on bars
    for idx, cell in enumerate(cells):
        x1 = points[cell[0]][0]
        x2 = points[cell[1]][0]
        mx = (x1 + x2) / 2
        off = moment_values[idx] * bm_scale
        label_y = off + np.sign(off) * span * 0.015
        ax.annotate(f'{moment_values[idx]:.2f}',
                    xy=(mx, label_y), fontsize=6, ha='center',
                    va='bottom' if off >= 0 else 'top',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.08', facecolor='white',
                              alpha=0.9, edgecolor='gray', linewidth=0.4))

    # Response location (NO UDL on BMD plot)
    draw_response_location_beam(ax, points, cells, traced_elem_id - 1,
                                stress_location, ring_radius=span * 0.015)

    # Legend
    pos_p = mpatches.Patch(facecolor='#3366cc', alpha=0.4,
                           label=f'Positive {active_moment_label} (sagging)')
    neg_p = mpatches.Patch(facecolor='#cc3333', alpha=0.4,
                           label=f'Negative {active_moment_label} (hogging)')
    resp_p = mpatches.Patch(facecolor='lightyellow', edgecolor='darkorange',
                             linewidth=1.5, label='Response location')
    ax.legend(handles=[pos_p, neg_p, resp_p], loc='upper left', fontsize=8)

    setup_beam_ax(ax, points, support_nodes, support_types)
    max_bar = max_moment * bm_scale
    if max_bar > 0:
        ax.set_ylim(-max_bar * 1.8, max_bar * 1.8)
    else:
        ax.set_ylim(-1, 1)
    ax.set_ylabel(f'Bending Moment {active_moment_label}', fontsize=10)

    # Title shows both JSON stress_type and actual plotted component
    if active_moment_label != stress_type:
        title = (f'Bending Moment Diagram\n'
                 f'(JSON: {stress_type} → Plotted: {active_moment_label})')
    else:
        title = f'Bending Moment {stress_type} Diagram'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


# =====================================================================
#  PLOT 4: Sensitivity dMY/dI22 — Discrete Bars
# =====================================================================
def plot_sensitivity(data, primal_data=None, traced_elem_id=6,
                     stress_location=2, stress_type="MY", save_name=None):
    points = data['points']
    cells = data['cells']

    sensitivity = data['cell_data']['I22_SENSITIVITY']
    if sensitivity.ndim == 2:
        sensitivity = sensitivity[:, 0]

    traced_idx = traced_elem_id - 1
    support_nodes = detect_support_nodes_beam(points)

    if primal_data is not None:
        support_types = detect_support_type(primal_data, support_nodes)
    else:
        x_vals = points[:, 0]
        x_min = x_vals.min()
        support_types = {}
        for sn in support_nodes:
            if abs(x_vals[sn] - x_min) < 1e-6:
                support_types[sn] = 'pinned'
            else:
                support_types[sn] = 'roller'

    span = points[:, 0].max() - points[:, 0].min()

    fig, ax = plt.subplots(figsize=(12, 6))

    draw_beam(ax, points, cells, lw=3)

    max_sens = np.max(np.abs(sensitivity))
    sens_scale = span * 0.15 / max_sens if max_sens > 0 else 1

    for idx, cell in enumerate(cells):
        x1 = points[cell[0]][0]
        x2 = points[cell[1]][0]
        draw_discrete_bar_beam(ax, x1, x2, sensitivity[idx], sens_scale)

    for idx, cell in enumerate(cells):
        x1 = points[cell[0]][0]
        x2 = points[cell[1]][0]
        mx = (x1 + x2) / 2
        val = sensitivity[idx]
        off = val * sens_scale
        label_y = off + np.sign(off) * span * 0.015
        face = '#ccddff' if val >= 0 else '#ffcccc'
        ax.annotate(f'E{idx + 1}\n{format_value(val)}',
                    xy=(mx, label_y), fontsize=5.5, ha='center',
                    va='bottom' if off >= 0 else 'top',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.08', facecolor=face,
                              alpha=0.9, edgecolor='gray', linewidth=0.4))

    x1t = points[cells[traced_idx][0]][0]
    x2t = points[cells[traced_idx][1]][0]
    ax.plot([x1t, x2t], [0, 0], color='gold', linewidth=12,
            alpha=0.5, zorder=3, solid_capstyle='round')

    draw_response_location_beam(ax, points, cells, traced_idx,
                                stress_location, ring_radius=span * 0.015)

    pos_p = mpatches.Patch(facecolor='#3366cc', alpha=0.4,
                           label=f'Positive: stiffening INCREASES {stress_type}')
    neg_p = mpatches.Patch(facecolor='#cc3333', alpha=0.4,
                           label=f'Negative: stiffening DECREASES {stress_type}')
    traced_p = mpatches.Patch(facecolor='gold', edgecolor='darkorange',
                               alpha=0.5, label=f'Traced element (E{traced_elem_id})')
    resp_p = mpatches.Patch(facecolor='lightyellow', edgecolor='darkorange',
                             linewidth=1.5, label='Response location')
    ax.legend(handles=[pos_p, neg_p, traced_p, resp_p],
              loc='upper left', fontsize=7, framealpha=0.9)

    resp_node = cells[traced_idx][1] if stress_location == 2 else cells[traced_idx][0]
    max_abs_idx = np.argmax(np.abs(sensitivity))
    info = (f'Response: {stress_type} at Node {resp_node + 1} (E{traced_elem_id})\n'
            f'Max |d{stress_type}/dI22|: E{max_abs_idx + 1} = '
            f'{format_value(sensitivity[max_abs_idx])}')
    ax.text(0.98, 0.02, info, transform=ax.transAxes,
            fontsize=8, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='darkorange', linewidth=1.2, alpha=0.95))

    setup_beam_ax(ax, points, support_nodes, support_types)
    max_bar = max_sens * sens_scale
    ax.set_ylim(-max_bar * 2.0, max_bar * 2.0)
    ax.set_ylabel(f'Sensitivity d{stress_type}/dI22', fontsize=10)

    ax.set_title(f'I22 Sensitivity Diagram (d{stress_type}/dI22)\n'
                 f'Traced: Element {traced_elem_id}, '
                 f'Location: {"end" if stress_location == 2 else "start"} node',
                 fontsize=14, fontweight='bold', pad=12)

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


# =====================================================================
#  MAIN
# =====================================================================
if __name__ == '__main__':

    os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\SA_Kratos_adj_V3_copy")

    # ---- Load settings from adjoint JSON ----
    ADJOINT_JSON = "ProjectParametersAdjoint.json"

    # Try explicit path first, then auto-detect
    if not os.path.exists(ADJOINT_JSON):
        print(f"'{ADJOINT_JSON}' not found, searching for adjoint JSON...")
        found = find_adjoint_json(".")
        if found:
            ADJOINT_JSON = found
            print(f"Found: {ADJOINT_JSON}")
        else:
            raise FileNotFoundError(
                "No adjoint JSON file found. Please specify the path manually."
            )

    TRACED_ELEMENT_ID, STRESS_LOCATION, STRESS_TYPE = load_adjoint_settings(
        ADJOINT_JSON
    )

    # Fallback if stress_type is None
    if STRESS_TYPE is None:
        STRESS_TYPE = "MY"

    print(f"\n{'='*60}")
    print(f"  TRACED_ELEMENT_ID = {TRACED_ELEMENT_ID}")
    print(f"  STRESS_LOCATION   = {STRESS_LOCATION} "
          f"({'start node' if STRESS_LOCATION == 1 else 'end node'})")
    print(f"  STRESS_TYPE       = {STRESS_TYPE}")
    print(f"{'='*60}\n")

    # ---- PRIMAL ----
    primal_file = "vtk_output_primal/Structure_0_1.vtk"
    primal = None
    if os.path.exists(primal_file):
        primal = read_vtk_file(primal_file)

        print("Point data fields:", list(primal['point_data'].keys()))
        print("Cell data fields:", list(primal['cell_data'].keys()))
        print(f"Points: {len(primal['points'])}, Cells: {len(primal['cells'])}")

        print("\n1. Plotting structure...")
        plot_structure(primal, save_name='Beam_plots/plot_01_structure.png')

        print("2. Plotting deformed shape...")
        plot_deformed(primal, TRACED_ELEMENT_ID, STRESS_LOCATION,
                      save_name='Beam_plots/plot_02_deformed.png')

        print("3. Plotting BMD...")
        plot_bmd(primal, TRACED_ELEMENT_ID, STRESS_LOCATION,
                 stress_type=STRESS_TYPE,
                 save_name='Beam_plots/plot_03_BMD.png')
    else:
        print(f"Not found: {primal_file}")

    # ---- ADJOINT ----
    adjoint_file = "vtk_output_adjoint/Structure_0_1.vtk"
    if os.path.exists(adjoint_file):
        adjoint = read_vtk_file(adjoint_file)

        print("\nAdjoint point data fields:", list(adjoint['point_data'].keys()))
        print("Adjoint cell data fields:", list(adjoint['cell_data'].keys()))

        print("\n4. Plotting I22 sensitivity...")
        plot_sensitivity(adjoint, primal_data=primal,
                         traced_elem_id=TRACED_ELEMENT_ID,
                         stress_location=STRESS_LOCATION,
                         stress_type=STRESS_TYPE,
                         save_name='Beam_plots/plot_04_sensitivity.png')
    else:
        print(f"Not found: {adjoint_file}")

    print("\nDone!")