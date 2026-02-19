"""
VTK Frame Plotter - Separate Clean Plots
BMD with bars, Sensitivity with bars, Response location highlight.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon, Circle
import matplotlib.patches as mpatches
import os


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


def format_value(val):
    """Format large numbers readably."""
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


def draw_supports(ax, points, support_indices=[0, 1]):
    """Draw fixed support symbols."""
    for sn in support_indices:
        x, z = points[sn][0], points[sn][2]
        tri = plt.Polygon(
            [[x - 0.3, z - 0.5], [x + 0.3, z - 0.5], [x, z]],
            closed=True, facecolor='gray', edgecolor='black', linewidth=1.5, zorder=6)
        ax.add_patch(tri)
        ax.plot([x - 0.5, x + 0.5], [z - 0.5, z - 0.5], 'k-', linewidth=2)
        for hx in np.linspace(x - 0.4, x + 0.4, 5):
            ax.plot([hx, hx - 0.15], [z - 0.5, z - 0.75], 'k-', linewidth=0.8)


def draw_floor_labels(ax, x_right=6.0):
    """Draw floor level labels."""
    floors = {0: 'Ground', 3: 'Floor 1', 6: 'Floor 2', 9: 'Floor 3',
              12: 'Floor 4', 15: 'Floor 5', 18: 'Floor 6'}
    for z, label in floors.items():
        ax.annotate(label, xy=(x_right + 0.8, z), fontsize=8,
                    color='gray', va='center', fontstyle='italic')


def draw_udl_arrows(ax, cells, points, beam_indices, color='blue', alpha=0.3):
    """Draw UDL arrows on beam elements."""
    for idx in beam_indices:
        p1 = points[cells[idx][0]]
        p2 = points[cells[idx][1]]
        z = p1[2]
        for xx in np.linspace(p1[0] + 0.15, p2[0] - 0.15, 4):
            ax.annotate('', xy=(xx, z), xytext=(xx, z + 0.4),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.2, alpha=alpha))


def draw_response_location(ax, points, cells, traced_elem_idx, stress_location,
                           ring_radius=0.35, ring_color='darkorange', ring_lw=3,
                           label_offset_x=1.2, label_offset_z=-1.2):
    """Draw ring circle at response location."""
    cell = cells[traced_elem_idx]
    resp_node = cell[1] if stress_location == 2 else cell[0]
    resp_x = points[resp_node][0]
    resp_z = points[resp_node][2]

    ring_outer = Circle((resp_x, resp_z), ring_radius,
                        fill=False, edgecolor=ring_color, linewidth=ring_lw,
                        linestyle='-', zorder=8)
    ax.add_patch(ring_outer)
    ring_inner = Circle((resp_x, resp_z), ring_radius * 0.6,
                        fill=False, edgecolor=ring_color, linewidth=ring_lw * 0.6,
                        linestyle='--', zorder=8)
    ax.add_patch(ring_inner)
    ax.plot(resp_x, resp_z, 'o', color=ring_color, markersize=6, zorder=9,
            markeredgecolor='black', markeredgewidth=0.8)

    ax.annotate(f'Response\nLocation\n(Node {resp_node + 1})',
                xy=(resp_x, resp_z),
                xytext=(resp_x + label_offset_x, resp_z + label_offset_z),
                fontsize=8, fontweight='bold', color=ring_color, ha='center',
                arrowprops=dict(arrowstyle='->', color=ring_color, lw=2,
                                connectionstyle='arc3,rad=0.2'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor=ring_color, linewidth=1.5, alpha=0.95),
                zorder=10)
    return resp_x, resp_z


def draw_frame(ax, points, cells, linewidth=2.5):
    """Draw the frame structure as black lines."""
    for cell in cells:
        p1, p2 = points[cell[0]], points[cell[1]]
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], 'k-', linewidth=linewidth, zorder=4)
    for idx, pt in enumerate(points):
        ax.plot(pt[0], pt[2], 'ko', markersize=3, zorder=5)


def get_element_end_moments(cells, points, moment_mid):
    """Estimate end moments from midpoint values using node averaging."""
    node_moments = {}
    for idx, cell in enumerate(cells):
        node_moments.setdefault(cell[0], []).append(moment_mid[idx])
        node_moments.setdefault(cell[1], []).append(moment_mid[idx])
    node_avg = {nid: np.mean(vals) for nid, vals in node_moments.items()}
    end_moments = []
    for idx, cell in enumerate(cells):
        end_moments.append((node_avg[cell[0]], moment_mid[idx], node_avg[cell[1]]))
    return end_moments


def get_element_end_values(cells, points, cell_values):
    """Estimate end values from cell-center values using node averaging for sensitivity."""
    node_vals = {}
    for idx, cell in enumerate(cells):
        node_vals.setdefault(cell[0], []).append(cell_values[idx])
        node_vals.setdefault(cell[1], []).append(cell_values[idx])
    node_avg = {nid: np.mean(vals) for nid, vals in node_vals.items()}
    end_vals = []
    for idx, cell in enumerate(cells):
        end_vals.append((node_avg[cell[0]], cell_values[idx], node_avg[cell[1]]))
    return end_vals


def draw_bars_on_element(ax, p1_3d, p2_3d, v_start, v_mid, v_end, scale,
                         pos_color='#3366cc', neg_color='#cc3333', alpha=0.35,
                         n_segments=20):
    """Draw filled diagram bars perpendicular to element with parabolic interpolation."""
    x1, z1 = p1_3d[0], p1_3d[2]
    x2, z2 = p2_3d[0], p2_3d[2]
    dx, dz = x2 - x1, z2 - z1
    L = np.sqrt(dx**2 + dz**2)
    if L < 1e-10:
        return

    nx, nz = -dz / L, dx / L
    t = np.linspace(0, 1, n_segments + 1)

    c = v_start
    a = 2 * v_end + 2 * v_start - 4 * v_mid
    b = v_end - v_start - a
    v_vals = a * t**2 + b * t + c

    base_x = x1 + t * dx
    base_z = z1 + t * dz
    bar_x = base_x + v_vals * scale * nx
    bar_z = base_z + v_vals * scale * nz

    for seg in range(n_segments):
        v_avg = (v_vals[seg] + v_vals[seg + 1]) / 2
        color = pos_color if v_avg >= 0 else neg_color
        seg_poly = list(zip(
            [base_x[seg], base_x[seg + 1], bar_x[seg + 1], bar_x[seg]],
            [base_z[seg], base_z[seg + 1], bar_z[seg + 1], bar_z[seg]]
        ))
        ax.add_patch(Polygon(seg_poly, closed=True, facecolor=color,
                             edgecolor='none', alpha=alpha, zorder=2))

    dominant = pos_color if np.mean(v_vals) >= 0 else neg_color
    ax.plot(bar_x, bar_z, color=dominant, linewidth=1.5, alpha=0.8, zorder=3)
    ax.plot([base_x[0], bar_x[0]], [base_z[0], bar_z[0]],
            color='gray', linewidth=0.8, alpha=0.6, zorder=3)
    ax.plot([base_x[-1], bar_x[-1]], [base_z[-1], bar_z[-1]],
            color='gray', linewidth=0.8, alpha=0.6, zorder=3)


def add_value_labels(ax, cells, points, values, scale, beam_indices, col_indices,
                     fontsize_beam=6.5, fontsize_col=6, format_fn=None):
    """Add value labels next to diagram bars."""
    if format_fn is None:
        format_fn = lambda v: f'{v:.1f}'

    for idx in beam_indices:
        p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
        mid_x = (p1[0] + p2[0]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        val = values[idx]
        dx, dz = p2[0] - p1[0], p2[2] - p1[2]
        L = np.sqrt(dx**2 + dz**2)
        nx, nz = -dz / L, dx / L
        off = val * scale * 1.4
        ax.annotate(format_fn(val),
                    xy=(mid_x + nx * off, mid_z + nz * off),
                    fontsize=fontsize_beam, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                              alpha=0.9, edgecolor='gray', linewidth=0.5))

    for idx in col_indices:
        p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
        mid_x = (p1[0] + p2[0]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        val = values[idx]
        offset_x = -0.9 if mid_x < 3 else 0.9
        face = '#ccddff' if val >= 0 else '#ffcccc'
        ax.annotate(format_fn(val),
                    xy=(mid_x + offset_x, mid_z),
                    fontsize=fontsize_col, ha='center', color='dimgray',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor=face,
                              alpha=0.8, edgecolor='gray', linewidth=0.3))


def setup_ax(ax, points, title):
    """Common axis setup."""
    draw_supports(ax, points)
    draw_floor_labels(ax)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    ax.margins(0.15)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)


# =====================================================================
#  PLOT 1: Bending Moment Diagram
# =====================================================================
def plot_bmd(data, traced_elem_id=14, stress_location=2, save_name=None):
    """Bending moment diagram with perpendicular bars."""

    points = data['points']
    cells = data['cells']
    moment_Y = data['cell_data']['MOMENT'][:, 1]
    traced_idx = traced_elem_id - 1

    fig, ax = plt.subplots(figsize=(12, 16))

    draw_frame(ax, points, cells)

    # BMD bars
    bm_scale = 0.015
    end_moments = get_element_end_moments(cells, points, moment_Y)
    for idx, cell in enumerate(cells):
        draw_bars_on_element(ax, points[cell[0]], points[cell[1]],
                             *end_moments[idx], bm_scale,
                             pos_color='#3366cc', neg_color='#cc3333')

    # Labels
    beam_idx = list(range(12, 30))
    col_idx = list(range(0, 12))
    add_value_labels(ax, cells, points, moment_Y, bm_scale, beam_idx, col_idx)

    # UDL arrows
    draw_udl_arrows(ax, cells, points, beam_idx)

    # Response location
    draw_response_location(ax, points, cells, traced_idx, stress_location)

    # Legend
    pos_p = mpatches.Patch(facecolor='#3366cc', alpha=0.35, label='Positive MY (sagging)')
    neg_p = mpatches.Patch(facecolor='#cc3333', alpha=0.35, label='Negative MY (hogging)')
    resp_p = mpatches.Patch(facecolor='lightyellow', edgecolor='darkorange',
                             linewidth=2, label='Response location')
    ax.legend(handles=[pos_p, neg_p, resp_p], loc='upper left', fontsize=10, framealpha=0.9)

    setup_ax(ax, points, 'Bending Moment MY Diagram\nFixed-Fixed 6-Floor Frame, UDL q = 40 N/m ↓Z')

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


# =====================================================================
#  PLOT 2: Deformed Shape
# =====================================================================
def plot_deformed(data, traced_elem_id=14, stress_location=2, save_name=None):
    """Deformed shape with displacement coloring."""

    points = data['points']
    cells = data['cells']
    disp = data['point_data']['DISPLACEMENT']
    disp_Z = disp[:, 2]
    traced_idx = traced_elem_id - 1

    fig, ax = plt.subplots(figsize=(12, 16))

    # Original (dashed)
    for cell in cells:
        p1, p2 = points[cell[0]], points[cell[1]]
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], 'k--', linewidth=1, alpha=0.25)

    # Deformed
    scale_disp = 5000
    for cell in cells:
        i1, i2 = cell[0], cell[1]
        p1d = points[i1] + disp[i1] * scale_disp
        p2d = points[i2] + disp[i2] * scale_disp
        ax.plot([p1d[0], p2d[0]], [p1d[2], p2d[2]], 'r-', linewidth=2.5, alpha=0.8)

    # Node colors
    vmax_d = max(abs(disp_Z.max()), abs(disp_Z.min()))
    if vmax_d == 0:
        vmax_d = 1
    norm_d = mcolors.TwoSlopeNorm(vmin=-vmax_d, vcenter=0, vmax=vmax_d)
    scatter = ax.scatter(points[:, 0], points[:, 2], c=disp_Z,
                         cmap=cm.coolwarm, norm=norm_d, s=100,
                         edgecolors='black', linewidth=0.5, zorder=5)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.4, pad=0.02, aspect=25)
    cbar.set_label('Displacement Z (m)', fontsize=11)

    # Max displacement
    max_idx = np.argmin(disp_Z)
    ax.annotate(f'Max: {disp_Z[max_idx]:.2e} m\nNode {max_idx + 1}',
                xy=(points[max_idx][0], points[max_idx][2]),
                xytext=(30, -30), textcoords='offset points',
                fontsize=10, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose',
                          edgecolor='red', alpha=0.9))

    # Response location
    draw_response_location(ax, points, cells, traced_idx, stress_location)

    # Legend
    orig = mpatches.Patch(facecolor='black', alpha=0.25, label='Original shape')
    defd = mpatches.Patch(facecolor='red', alpha=0.8, label=f'Deformed (×{scale_disp})')
    resp = mpatches.Patch(facecolor='lightyellow', edgecolor='darkorange',
                           linewidth=2, label='Response location')
    ax.legend(handles=[orig, defd, resp], loc='upper right', fontsize=10)

    setup_ax(ax, points, 'Deformed Shape & Vertical Displacement\nAmplification ×5000')

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


# =====================================================================
#  PLOT 3: I22 Sensitivity Diagram with bars (like BMD)
# =====================================================================
def plot_sensitivity(data, traced_elem_id=14, stress_location=2, save_name=None):
    """I22 Sensitivity diagram with perpendicular bars like BMD."""

    points = data['points']
    cells = data['cells']
    sensitivity = data['cell_data']['I22_SENSITIVITY']
    traced_idx = traced_elem_id - 1

    fig, ax = plt.subplots(figsize=(12, 16))

    draw_frame(ax, points, cells)

    # Sensitivity bars — interpolated like BMD for smooth connected diagram
    sens_scale = 2e-6
    end_vals = get_element_end_values(cells, points, sensitivity)
    for idx, cell in enumerate(cells):
        draw_bars_on_element(ax, points[cell[0]], points[cell[1]],
                             *end_vals[idx], sens_scale,
                             pos_color='#3366cc', neg_color='#cc3333')

    # Highlight traced element
    p1t = points[cells[traced_idx][0]]
    p2t = points[cells[traced_idx][1]]
    ax.plot([p1t[0], p2t[0]], [p1t[2], p2t[2]], color='gold',
            linewidth=12, alpha=0.5, zorder=3)

    # Response location
    draw_response_location(ax, points, cells, traced_idx, stress_location,
                           ring_radius=0.4)

    # Labels
    beam_idx = list(range(12, 30))
    col_idx = list(range(0, 12))

    for idx in beam_idx:
        p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
        mid_x = (p1[0] + p2[0]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        val = sensitivity[idx]
        dx, dz = p2[0] - p1[0], p2[2] - p1[2]
        L = np.sqrt(dx**2 + dz**2)
        nx, nz = -dz / L, dx / L
        off = val * sens_scale * 1.3
        face = '#ccddff' if val >= 0 else '#ffcccc'
        ax.annotate(f'E{idx + 1}\n{format_value(val)}',
                    xy=(mid_x + nx * off, mid_z + nz * off + 0.15),
                    fontsize=6, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.12', facecolor=face,
                              alpha=0.9, edgecolor='gray', linewidth=0.5))

    for idx in col_idx:
        p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
        mid_x = (p1[0] + p2[0]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        val = sensitivity[idx]
        offset_x = -1.0 if mid_x < 3 else 1.0
        face = '#ccddff' if val >= 0 else '#ffcccc'
        ax.annotate(f'E{idx + 1}\n{format_value(val)}',
                    xy=(mid_x + offset_x, mid_z), fontsize=5.5, ha='center',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor=face,
                              alpha=0.85, edgecolor='gray', linewidth=0.3))

    # Legend
    pos_p = mpatches.Patch(facecolor='#3366cc', alpha=0.35,
                           label='Positive: stiffening INCREASES MY at response')
    neg_p = mpatches.Patch(facecolor='#cc3333', alpha=0.35,
                           label='Negative: stiffening DECREASES MY at response')
    traced_p = mpatches.Patch(facecolor='gold', edgecolor='darkorange',
                               alpha=0.5, label=f'Traced element (E{traced_elem_id})')
    resp_p = mpatches.Patch(facecolor='lightyellow', edgecolor='darkorange',
                             linewidth=2, label='Response location')
    ax.legend(handles=[pos_p, neg_p, traced_p, resp_p],
              loc='upper left', fontsize=9, framealpha=0.9)

    # Info box
    resp_node = cells[traced_idx][1] if stress_location == 2 else cells[traced_idx][0]
    info_text = (f'Response: MY at Node {resp_node + 1}\n'
                 f'Element {traced_elem_id}, stress_location={stress_location}\n'
                 f'Max |sensitivity|: E{np.argmax(np.abs(sensitivity)) + 1} '
                 f'= {format_value(sensitivity[np.argmax(np.abs(sensitivity))])}')
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
            fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='darkorange', linewidth=1.5, alpha=0.95))

    setup_ax(ax, points,
             f'I22 Sensitivity Diagram (dMY/dI22)\n'
             f'Traced: Element {traced_elem_id}, Node {resp_node + 1}')

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


# =====================================================================
#  PLOT 4: Adjoint Displacement
# =====================================================================
def plot_adjoint_displacement(data, component=0, traced_elem_id=14,
                              stress_location=2, save_name=None):
    """Adjoint displacement field on frame."""

    points = data['points']
    cells = data['cells']
    adj_disp = data['point_data']['ADJOINT_DISPLACEMENT']
    traced_idx = traced_elem_id - 1
    comp_labels = ['X', 'Y', 'Z']

    values = adj_disp[:, component]

    fig, ax = plt.subplots(figsize=(12, 16))

    # Frame lines
    for cell in cells:
        p1, p2 = points[cell[0]], points[cell[1]]
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], 'k-', linewidth=1.5, alpha=0.3)

    # Node colors
    vmax = max(abs(values.max()), abs(values.min()))
    if vmax == 0:
        vmax = 1
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    scatter = ax.scatter(points[:, 0], points[:, 2], c=values,
                         cmap=cm.RdBu_r, norm=norm, s=120,
                         edgecolors='black', linewidth=0.5, zorder=5)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.4, pad=0.02, aspect=25)
    cbar.set_label(f'Adjoint Displacement {comp_labels[component]}', fontsize=11)

    # Node value labels
    for idx, pt in enumerate(points):
        ax.annotate(f'N{idx + 1}\n{values[idx]:.3e}',
                    xy=(pt[0], pt[2]),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=5, ha='left',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              alpha=0.85, edgecolor='gray', linewidth=0.3))

    # Response location
    draw_response_location(ax, points, cells, traced_idx, stress_location)

    setup_ax(ax, points,
             f'Adjoint Displacement ({comp_labels[component]}-component)')

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


# =====================================================================
#  MAIN
# =====================================================================
if __name__ == '__main__':

    os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\SA_Kratos_adj_V3")

    # Configuration
    TRACED_ELEMENT_ID = 14
    STRESS_LOCATION = 2

    # ---- PRIMAL ----
    primal_file = "vtk_output_primal/Structure_0_1.vtk"
    if os.path.exists(primal_file):
        primal = read_vtk_file(primal_file)

        print("1. Plotting BMD...")
        plot_bmd(primal, TRACED_ELEMENT_ID, STRESS_LOCATION,
                 save_name='plot_01_BMD.png')

        print("2. Plotting deformed shape...")
        plot_deformed(primal, TRACED_ELEMENT_ID, STRESS_LOCATION,
                      save_name='plot_02_deformed.png')
    else:
        print(f"Primal not found: {primal_file}")

    # ---- ADJOINT ----
    adjoint_file = "vtk_output_adjoint/Structure_0_1.vtk"
    if os.path.exists(adjoint_file):
        adjoint = read_vtk_file(adjoint_file)

        print("3. Plotting I22 sensitivity diagram...")
        plot_sensitivity(adjoint, TRACED_ELEMENT_ID, STRESS_LOCATION,
                         save_name='plot_03_I22_sensitivity.png')

        # print("4. Plotting adjoint displacement X...")
        # plot_adjoint_displacement(adjoint, component=0,
        #                           traced_elem_id=TRACED_ELEMENT_ID,
        #                           stress_location=STRESS_LOCATION,
        #                           save_name='plot_04_adjoint_disp_X.png')

        # print("5. Plotting adjoint displacement Z...")
        # plot_adjoint_displacement(adjoint, component=2,
        #                           traced_elem_id=TRACED_ELEMENT_ID,
        #                           stress_location=STRESS_LOCATION,
        #                           save_name='plot_05_adjoint_disp_Z.png')
    else:
        print(f"Adjoint not found: {adjoint_file}")

    print("\nDone! All plots generated.")