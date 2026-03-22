"""
VTK Frame Plotter - Auto-detects structure from VTK data
BMD (discrete bars), Displacement, Sensitivity (discrete bars)
Supports detected from MDPA, works for any frame geometry.
Reads traced_element_id, stress_location, stress_type from adjoint JSON.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon, Circle
import matplotlib.patches as mpatches
import os
import json
import glob


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
            n_cells = int(line.split()[1])
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


def load_adjoint_settings(json_path):
    """
    Load adjoint JSON and extract traced_element_id, stress_location,
    and stress_type from solver_settings.response_function_settings.
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
    """Auto-detect adjoint JSON by common naming patterns."""
    patterns = [
        "ProjectParametersAdjoint*.json",
        "*adjoint*.json",
        "*Adjoint*.json",
    ]
    for pattern in patterns:
        matches = glob.glob(os.path.join(search_dir, pattern))
        if matches:
            return matches[0]
    return None


def detect_support_nodes(points):
    """Auto-detect support nodes as nodes at minimum Z elevation."""
    z_vals = points[:, 2]
    z_min = z_vals.min()
    return [i for i in range(len(points)) if abs(z_vals[i] - z_min) < 1e-6]


def classify_elements(cells, points):
    """Classify elements into columns (vertical) and beams (horizontal)."""
    columns = []
    beams = []
    for idx, cell in enumerate(cells):
        p1, p2 = points[cell[0]], points[cell[1]]
        dx = abs(p1[0] - p2[0])
        dz = abs(p1[2] - p2[2])
        if dz > dx:
            columns.append(idx)
        else:
            beams.append(idx)
    return columns, beams


def detect_floors(points, cells, beams):
    """Detect floor levels from beam element Z-coordinates."""
    floor_z = set()
    for idx in beams:
        p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
        z_avg = (p1[2] + p2[2]) / 2
        floor_z.add(round(z_avg, 2))
    return sorted(floor_z)


def get_moment_column(moment_data, stress_type):
    """
    Select correct moment column based on stress_type.
    Auto-detects non-zero column if requested column is all zeros.
    Returns (moment_values, active_label).
    """
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
            active_label = col_names[best_col]
        else:
            print(f"  [WARNING] All moment columns are zero!")
            active_label = stress_type
    else:
        active_label = stress_type

    return moment_values, active_label


def format_value(val):
    abs_val = abs(val)
    if abs_val >= 1e6:
        return f'{val / 1e6:.1f}M'
    elif abs_val >= 1e3:
        return f'{val / 1e3:.1f}K'
    elif abs_val >= 1:
        return f'{val:.1f}'
    elif abs_val >= 1e-3:
        return f'{val:.4f}'
    else:
        return f'{val:.2e}'


def draw_supports(ax, points, support_indices):
    """Draw fixed support symbols at detected support nodes."""
    for sn in support_indices:
        x, z = points[sn][0], points[sn][2]
        tri = plt.Polygon(
            [[x - 0.25, z - 0.4], [x + 0.25, z - 0.4], [x, z]],
            closed=True, facecolor='gray', edgecolor='black',
            linewidth=1.5, zorder=6)
        ax.add_patch(tri)
        ax.plot([x - 0.4, x + 0.4], [z - 0.4, z - 0.4], 'k-', linewidth=2)
        for hx in np.linspace(x - 0.35, x + 0.35, 5):
            ax.plot([hx, hx - 0.12], [z - 0.4, z - 0.6], 'k-', linewidth=0.7)


def draw_floor_labels(ax, floor_z_list, x_pos):
    """Draw floor labels at detected floor levels."""
    ax.annotate('Ground', xy=(x_pos, 0), fontsize=7, color='gray',
                va='center', ha='right', fontstyle='italic')
    for i, z in enumerate(floor_z_list):
        ax.annotate(f'Floor {i + 1}', xy=(x_pos, z), fontsize=7, color='gray',
                    va='center', ha='right', fontstyle='italic')


def draw_udl_arrows(ax, cells, points, beam_indices):
    """Draw UDL arrows on beam elements."""
    for idx in beam_indices:
        p1 = points[cells[idx][0]]
        p2 = points[cells[idx][1]]
        x_min = min(p1[0], p2[0])
        x_max = max(p1[0], p2[0])
        z = (p1[2] + p2[2]) / 2
        span = x_max - x_min
        n_arrows = max(3, int(span / 0.6))
        for xx in np.linspace(x_min + span * 0.05, x_max - span * 0.05, n_arrows):
            ax.annotate('', xy=(xx, z), xytext=(xx, z + 0.25),
                        arrowprops=dict(arrowstyle='->', color='blue',
                                        lw=0.8, alpha=0.3))


def draw_response_location(ax, points, cells, traced_idx, stress_location,
                            ring_radius=0.3):
    """Draw response location ring."""
    cell = cells[traced_idx]
    resp_node = cell[1] if stress_location == 2 else cell[0]
    rx, rz = points[resp_node][0], points[resp_node][2]

    ax.add_patch(Circle((rx, rz), ring_radius, fill=False,
                        edgecolor='darkorange', linewidth=2.5, zorder=8))
    ax.add_patch(Circle((rx, rz), ring_radius * 0.55, fill=False,
                        edgecolor='darkorange', linewidth=1.5,
                        linestyle='--', zorder=8))
    ax.plot(rx, rz, 'o', color='darkorange', markersize=2, zorder=9,
            markeredgecolor='black', markeredgewidth=0.7)
    ax.annotate(f'Response\n(Node {resp_node + 1})',
                xy=(rx, rz),
                xytext=(rx + 0.6, rz - 0.6),
                fontsize=7, fontweight='bold', color='darkorange', ha='center',
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5,
                                connectionstyle='arc3,rad=0.2'),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                          edgecolor='darkorange', linewidth=1.2, alpha=0.95),
                zorder=10)


def draw_frame(ax, points, cells, lw=2.5):
    """Draw frame structure."""
    for cell in cells:
        p1, p2 = points[cell[0]], points[cell[1]]
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], 'k-', linewidth=lw, zorder=4)
    for pt in points:
        ax.plot(pt[0], pt[2], 'ko', markersize=1, zorder=5)


def draw_discrete_bar(ax, p1_3d, p2_3d, value, scale,
                      pos_color='#3366cc', neg_color='#cc3333', alpha=0.35):
    """Draw constant-value bar perpendicular to element."""
    x1, z1 = p1_3d[0], p1_3d[2]
    x2, z2 = p2_3d[0], p2_3d[2]
    dx, dz = x2 - x1, z2 - z1
    L = np.sqrt(dx ** 2 + dz ** 2)
    if L < 1e-10:
        return
    nx, nz = -dz / L, dx / L
    off = value * scale
    poly_x = [x1, x2, x2 + nx * off, x1 + nx * off]
    poly_z = [z1, z2, z2 + nz * off, z1 + nz * off]
    color = pos_color if value >= 0 else neg_color
    ax.add_patch(Polygon(list(zip(poly_x, poly_z)), closed=True,
                         facecolor=color, edgecolor='none', alpha=alpha, zorder=2))
    ax.plot([x1 + nx * off, x2 + nx * off], [z1 + nz * off, z2 + nz * off],
            color=color, linewidth=1.2, alpha=0.7, zorder=3)
    ax.plot([x1, x1 + nx * off], [z1, z1 + nz * off],
            color='gray', linewidth=0.6, alpha=0.5, zorder=3)
    ax.plot([x2, x2 + nx * off], [z2, z2 + nz * off],
            color='gray', linewidth=0.6, alpha=0.5, zorder=3)


def setup_ax(ax, points, support_indices, floor_z_list):
    """Set axis properties centered on structure."""
    x_all = points[:, 0]
    z_all = points[:, 2]
    x_range = x_all.max() - x_all.min()
    z_range = z_all.max() - z_all.min()
    pad_x = x_range * 0.3
    pad_z_top = z_range * 0.12
    pad_z_bot = z_range * 0.08

    ax.set_xlim(x_all.min() - pad_x, x_all.max() + pad_x)
    ax.set_ylim(z_all.min() - pad_z_bot - 0.8, z_all.max() + pad_z_top)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Z (m)', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)

    draw_supports(ax, points, support_indices)
    draw_floor_labels(ax, floor_z_list, x_pos=x_all.min() - pad_x * 0.3)


# =====================================================================
#  PLOT 1: Structure with Loads
# =====================================================================
def plot_structure(data, save_name=None):
    points = data['points']
    cells = data['cells']
    support_nodes = detect_support_nodes(points)
    col_idx, beam_idx = classify_elements(cells, points)
    floor_z = detect_floors(points, cells, beam_idx)

    fig, ax = plt.subplots(figsize=(10, 14))

    draw_frame(ax, points, cells, lw=3)
    draw_udl_arrows(ax, cells, points, beam_idx)

    # UDL label
    z_top = points[:, 2].max()
    x_mid = (points[:, 0].min() + points[:, 0].max()) / 2
    ax.annotate('q = 40 N/m ↓Z', xy=(x_mid, z_top + 0.5), fontsize=9,
                ha='center', color='blue', fontweight='bold')

    setup_ax(ax, points, support_nodes, floor_z)
    ax.set_title('Frame Structure\nFixed Supports, UDL on All Floors',
                 fontsize=14, fontweight='bold', pad=12)

    plt.tight_layout()
    if save_name:
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


# =====================================================================
#  PLOT 2: Deformed Shape
# =====================================================================
def plot_deformed(data, traced_elem_id=14, stress_location=2, save_name=None):
    points = data['points']
    cells = data['cells']
    disp = data['point_data']['DISPLACEMENT']
    disp_Z = disp[:, 2]
    support_nodes = detect_support_nodes(points)
    _, beam_idx = classify_elements(cells, points)
    floor_z = detect_floors(points, cells, beam_idx)

    fig, ax = plt.subplots(figsize=(10, 14))

    # Original
    for cell in cells:
        p1, p2 = points[cell[0]], points[cell[1]]
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], 'k--', linewidth=1, alpha=0.3)

    # Auto scale deformation
    max_disp = np.max(np.abs(disp))
    x_range = points[:, 0].max() - points[:, 0].min()
    amp = x_range * 0.05 / max_disp if max_disp > 0 else 1

    for cell in cells:
        i1, i2 = cell[0], cell[1]
        p1d = points[i1] + disp[i1] * amp
        p2d = points[i2] + disp[i2] * amp
        ax.plot([p1d[0], p2d[0]], [p1d[2], p2d[2]], 'r-', linewidth=2.5, alpha=0.8)

    # Node colors
    vmax = max(abs(disp_Z.max()), abs(disp_Z.min()))
    if vmax == 0:
        vmax = 1
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    scatter = ax.scatter(points[:, 0], points[:, 2], c=disp_Z,
                         cmap=cm.coolwarm, norm=norm, s=10,
                         edgecolors='black', linewidth=0.5, zorder=5)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.3, pad=0.02, aspect=20)
    cbar.set_label('Displacement Z (m)', fontsize=10)

    # Max displacement
    max_idx = np.argmin(disp_Z)
    ax.annotate(f'Max: {disp_Z[max_idx]:.2e} m\nNode {max_idx + 1}',
                xy=(points[max_idx][0], points[max_idx][2]),
                xytext=(20, -20), textcoords='offset points',
                fontsize=8, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose',
                          edgecolor='red', alpha=0.9))

    draw_response_location(ax, points, cells, traced_elem_id - 1, stress_location)

    orig = mpatches.Patch(facecolor='black', alpha=0.3, label='Original')
    defd = mpatches.Patch(facecolor='red', alpha=0.8, label=f'Deformed (×{amp:.0f})')
    ax.legend(handles=[orig, defd], loc='upper right', fontsize=9)

    setup_ax(ax, points, support_nodes, floor_z)
    ax.set_title(f'Deformed Shape (Amplified ×{amp:.0f})',
                 fontsize=14, fontweight='bold', pad=12)

    plt.tight_layout()
    if save_name:
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


# =====================================================================
#  PLOT 3: BMD — Discrete Bars
# =====================================================================
def plot_bmd(data, traced_elem_id=14, stress_location=2,
             stress_type="MY", save_name=None):
    points = data['points']
    cells = data['cells']

    moment_data = data['cell_data']['MOMENT']
    if moment_data.ndim == 2:
        moment_values, active_label = get_moment_column(moment_data, stress_type)
    else:
        moment_values = moment_data
        active_label = stress_type

    support_nodes = detect_support_nodes(points)
    col_idx, beam_idx = classify_elements(cells, points)
    floor_z = detect_floors(points, cells, beam_idx)

    fig, ax = plt.subplots(figsize=(10, 14))

    draw_frame(ax, points, cells)

    # Auto scale for bars
    max_moment = np.max(np.abs(moment_values))
    x_range = points[:, 0].max() - points[:, 0].min()
    bm_scale = x_range * 0.12 / max_moment if max_moment > 0 else 1

    for idx, cell in enumerate(cells):
        draw_discrete_bar(ax, points[cell[0]], points[cell[1]],
                          moment_values[idx], bm_scale)

    draw_response_location(ax, points, cells, traced_elem_id - 1, stress_location)

    # Legend
    pos_p = mpatches.Patch(facecolor='#3366cc', alpha=0.35,
                           label=f'Positive {active_label} (sagging)')
    neg_p = mpatches.Patch(facecolor='#cc3333', alpha=0.35,
                           label=f'Negative {active_label} (hogging)')
    resp_p = mpatches.Patch(facecolor='lightyellow', edgecolor='darkorange',
                            linewidth=1.5, label='Response location')
    ax.legend(handles=[pos_p, neg_p, resp_p], loc='upper left', fontsize=8)

    setup_ax(ax, points, support_nodes, floor_z)

    # Title shows both JSON stress_type and actual plotted component
    if active_label != stress_type:
        title = (f'Bending Moment Diagram\n'
                 f'(JSON: {stress_type} → Plotted: {active_label})')
    else:
        title = f'Bending Moment {active_label} Diagram'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)

    plt.tight_layout()
    if save_name:
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


# =====================================================================
#  PLOT 4: Sensitivity dMY/dI22 — Discrete Bars
# =====================================================================
def plot_sensitivity(data, traced_elem_id=14, stress_location=2,
                     stress_type="MY", save_name=None):
    points = data['points']
    cells = data['cells']
    sensitivity = data['cell_data']['I22_SENSITIVITY']
    if sensitivity.ndim == 2:
        sensitivity = sensitivity[:, 0]

    traced_idx = traced_elem_id - 1
    support_nodes = detect_support_nodes(points)
    col_idx, beam_idx = classify_elements(cells, points)
    floor_z = detect_floors(points, cells, beam_idx)

    fig, ax = plt.subplots(figsize=(10, 14))

    draw_frame(ax, points, cells)

    # Auto scale
    max_sens = np.max(np.abs(sensitivity))
    x_range = points[:, 0].max() - points[:, 0].min()
    sens_scale = x_range * 0.12 / max_sens if max_sens > 0 else 1

    for idx, cell in enumerate(cells):
        draw_discrete_bar(ax, points[cell[0]], points[cell[1]],
                          sensitivity[idx], sens_scale)

    # Highlight traced element
    p1t = points[cells[traced_idx][0]]
    p2t = points[cells[traced_idx][1]]
    ax.plot([p1t[0], p2t[0]], [p1t[2], p2t[2]],
            color='gold', linewidth=10, alpha=0.5, zorder=3)

    draw_response_location(ax, points, cells, traced_idx, stress_location)

    # Legend
    pos_p = mpatches.Patch(facecolor='#3366cc', alpha=0.35,
                           label=f'Positive: stiffening INCREASES {stress_type}')
    neg_p = mpatches.Patch(facecolor='#cc3333', alpha=0.35,
                           label=f'Negative: stiffening DECREASES {stress_type}')
    traced_p = mpatches.Patch(facecolor='gold', edgecolor='darkorange',
                              alpha=0.5, label=f'Traced element (E{traced_elem_id})')
    resp_p = mpatches.Patch(facecolor='lightyellow', edgecolor='darkorange',
                            linewidth=1.5, label='Response location')
    ax.legend(handles=[pos_p, neg_p, traced_p, resp_p],
              loc='upper left', fontsize=7, framealpha=0.9)

    # Info box
    resp_node = cells[traced_idx][1] if stress_location == 2 else cells[traced_idx][0]
    max_abs_idx = np.argmax(np.abs(sensitivity))
    info = (f'Response: {stress_type} at Node {resp_node + 1} (E{traced_elem_id})\n'
            f'Max |d{stress_type}/dI22|: E{max_abs_idx + 1} = '
            f'{format_value(sensitivity[max_abs_idx])}')
    ax.text(0.98, 0.02, info, transform=ax.transAxes,
            fontsize=7, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='darkorange', linewidth=1.2, alpha=0.95))

    setup_ax(ax, points, support_nodes, floor_z)
    ax.set_title(f'I22 Sensitivity Diagram (d{stress_type}/dI22)\n'
                 f'Traced: Element {traced_elem_id}, '
                 f'Location: {"end" if stress_location == 2 else "start"} node',
                 fontsize=14, fontweight='bold', pad=12)

    plt.tight_layout()
    if save_name:
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


# =====================================================================
#  MAIN
# =====================================================================
if __name__ == '__main__':

    os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\SA_Kratos_adj_V3")

    # ---- Load settings from adjoint JSON ----
    ADJOINT_JSON = "beam_test_local_stress_adjoint_parameters.json"

    # Auto-detect if file not found at explicit path
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

    print(f"\n{'=' * 60}")
    print(f"  TRACED_ELEMENT_ID = {TRACED_ELEMENT_ID}")
    print(f"  STRESS_LOCATION   = {STRESS_LOCATION} "
          f"({'start node' if STRESS_LOCATION == 1 else 'end node'})")
    print(f"  STRESS_TYPE       = {STRESS_TYPE}")
    print(f"{'=' * 60}\n")

    # ---- PRIMAL ----
    primal_file = "vtk_output_primal/Structure_0_1.vtk"
    if os.path.exists(primal_file):
        primal = read_vtk_file(primal_file)

        print("Point data fields:", list(primal['point_data'].keys()))
        print("Cell data fields:", list(primal['cell_data'].keys()))
        print(f"Points: {len(primal['points'])}, Cells: {len(primal['cells'])}")

        print("\n1. Plotting structure...")
        plot_structure(primal, save_name='Frame_plots/plot_01_structure.png')

        print("2. Plotting deformed shape...")
        plot_deformed(primal, TRACED_ELEMENT_ID, STRESS_LOCATION,
                      save_name='Frame_plots/plot_02_deformed.png')

        print("3. Plotting BMD...")
        plot_bmd(primal, TRACED_ELEMENT_ID, STRESS_LOCATION,
                 stress_type=STRESS_TYPE,
                 save_name='Frame_plots/plot_03_BMD.png')
    else:
        print(f"Not found: {primal_file}")

    # ---- ADJOINT ----
    adjoint_file = "vtk_output_adjoint/Structure_0_1.vtk"
    if os.path.exists(adjoint_file):
        adjoint = read_vtk_file(adjoint_file)

        print("\nAdjoint point data fields:", list(adjoint['point_data'].keys()))
        print("Adjoint cell data fields:", list(adjoint['cell_data'].keys()))

        print("\n4. Plotting I22 sensitivity...")
        plot_sensitivity(adjoint, TRACED_ELEMENT_ID, STRESS_LOCATION,
                         stress_type=STRESS_TYPE,
                         save_name='Frame_plots/plot_04_sensitivity.png')
    else:
        print(f"Not found: {adjoint_file}")

    print("\nDone!")