"""
Clear and Visual VTK Frame Plotter
With BMD bars, Sensitivity bars, and Response location highlight.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon, Circle, FancyBboxPatch
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
                           ring_radius=0.35, ring_color='darkorange', ring_lw=3):
    """
    Draw a ring circle at the response location on the traced element.
    
    Parameters:
        traced_elem_idx: 0-indexed element index
        stress_location: 1 = first node, 2 = second node of element
    """
    cell = cells[traced_elem_idx]
    if stress_location == 2:
        resp_node = cell[1]
    else:
        resp_node = cell[0]
    
    resp_x = points[resp_node][0]
    resp_z = points[resp_node][2]
    
    # Outer ring
    ring_outer = Circle((resp_x, resp_z), ring_radius,
                        fill=False, edgecolor=ring_color, linewidth=ring_lw,
                        linestyle='-', zorder=8)
    ax.add_patch(ring_outer)
    
    # Inner ring
    ring_inner = Circle((resp_x, resp_z), ring_radius * 0.6,
                        fill=False, edgecolor=ring_color, linewidth=ring_lw * 0.6,
                        linestyle='--', zorder=8)
    ax.add_patch(ring_inner)
    
    # Center dot
    ax.plot(resp_x, resp_z, 'o', color=ring_color, markersize=6, zorder=9,
            markeredgecolor='black', markeredgewidth=0.8)
    
    # Label
    ax.annotate(f'Response\nLocation\n(Node {resp_node + 1})',
                xy=(resp_x, resp_z),
                xytext=(resp_x + 1.2, resp_z - 1.2),
                fontsize=8, fontweight='bold', color=ring_color,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=ring_color, lw=2,
                                connectionstyle='arc3,rad=0.2'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor=ring_color, linewidth=1.5, alpha=0.95),
                zorder=10)
    
    return resp_x, resp_z


def get_element_end_moments(cells, points, moment_mid, n_points):
    """Estimate end moments from midpoint values using node averaging."""
    node_moments = {}
    for idx, cell in enumerate(cells):
        n1, n2 = cell[0], cell[1]
        m_mid = moment_mid[idx]
        node_moments.setdefault(n1, []).append(m_mid)
        node_moments.setdefault(n2, []).append(m_mid)
    
    node_avg = {nid: np.mean(vals) for nid, vals in node_moments.items()}
    
    end_moments = []
    for idx, cell in enumerate(cells):
        m1 = node_avg[cell[0]]
        m2 = node_avg[cell[1]]
        end_moments.append((m1, moment_mid[idx], m2))
    return end_moments


def draw_bars_on_element(ax, p1_3d, p2_3d, m_start, m_mid, m_end, scale,
                         pos_color='#3366cc', neg_color='#cc3333', alpha=0.3,
                         n_segments=20):
    """Draw filled BMD/sensitivity bars perpendicular to element with parabolic interpolation."""
    x1, z1 = p1_3d[0], p1_3d[2]
    x2, z2 = p2_3d[0], p2_3d[2]
    dx, dz = x2 - x1, z2 - z1
    L = np.sqrt(dx**2 + dz**2)
    if L < 1e-10:
        return
    
    nx, nz = -dz / L, dx / L
    t = np.linspace(0, 1, n_segments + 1)
    
    # Parabolic: M(0)=m_start, M(0.5)=m_mid, M(1)=m_end
    c = m_start
    a = 2 * m_end + 2 * m_start - 4 * m_mid
    b = m_end - m_start - a
    m_vals = a * t**2 + b * t + c
    
    base_x = x1 + t * dx
    base_z = z1 + t * dz
    bmd_x = base_x + m_vals * scale * nx
    bmd_z = base_z + m_vals * scale * nz
    
    # Split into positive/negative segments for proper coloring
    for seg in range(n_segments):
        m_avg = (m_vals[seg] + m_vals[seg + 1]) / 2
        color = pos_color if m_avg >= 0 else neg_color
        
        seg_poly_x = [base_x[seg], base_x[seg + 1],
                      bmd_x[seg + 1], bmd_x[seg]]
        seg_poly_z = [base_z[seg], base_z[seg + 1],
                      bmd_z[seg + 1], bmd_z[seg]]
        
        poly = Polygon(list(zip(seg_poly_x, seg_poly_z)), closed=True,
                       facecolor=color, edgecolor='none', alpha=alpha, zorder=2)
        ax.add_patch(poly)
    
    # Outline
    dominant_color = pos_color if np.mean(m_vals) >= 0 else neg_color
    ax.plot(bmd_x, bmd_z, color=dominant_color, linewidth=1.5, alpha=0.8, zorder=3)
    
    # End lines
    ax.plot([base_x[0], bmd_x[0]], [base_z[0], bmd_z[0]],
            color='gray', linewidth=0.8, alpha=0.6, zorder=3)
    ax.plot([base_x[-1], bmd_x[-1]], [base_z[-1], bmd_z[-1]],
            color='gray', linewidth=0.8, alpha=0.6, zorder=3)


def draw_constant_bar_on_element(ax, p1_3d, p2_3d, value, scale,
                                  pos_color='#3366cc', neg_color='#cc3333',
                                  alpha=0.3):
    """Draw constant-value bar perpendicular to element (for cell data like sensitivity)."""
    x1, z1 = p1_3d[0], p1_3d[2]
    x2, z2 = p2_3d[0], p2_3d[2]
    dx, dz = x2 - x1, z2 - z1
    L = np.sqrt(dx**2 + dz**2)
    if L < 1e-10:
        return
    
    nx, nz = -dz / L, dx / L
    offset = value * scale
    
    poly_x = [x1, x2, x2 + nx * offset, x1 + nx * offset]
    poly_z = [z1, z2, z2 + nz * offset, z1 + nz * offset]
    
    color = pos_color if value >= 0 else neg_color
    
    poly = Polygon(list(zip(poly_x, poly_z)), closed=True,
                   facecolor=color, edgecolor='none', alpha=alpha, zorder=2)
    ax.add_patch(poly)
    
    # Outline on offset side
    ax.plot([x1 + nx * offset, x2 + nx * offset],
            [z1 + nz * offset, z2 + nz * offset],
            color=color, linewidth=1.5, alpha=0.7, zorder=3)
    
    # End closing lines
    ax.plot([x1, x1 + nx * offset], [z1, z1 + nz * offset],
            color='gray', linewidth=0.6, alpha=0.5, zorder=3)
    ax.plot([x2, x2 + nx * offset], [z2, z2 + nz * offset],
            color='gray', linewidth=0.6, alpha=0.5, zorder=3)


# =====================================================================
#  PLOT 1: PRIMAL with BMD bars + Response Location
# =====================================================================
def plot_primal(data, traced_elem_id=14, stress_location=2, save_name=None):
    """Primal results with connected BMD bars and response location."""

    points = data['points']
    cells = data['cells']
    moment = data['cell_data']['MOMENT']
    disp = data['point_data']['DISPLACEMENT']
    moment_Y = moment[:, 1]
    disp_Z = disp[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(22, 14))
    fig.suptitle('PRIMAL ANALYSIS — Fixed-Fixed 6-Floor Frame with UDL (q = 40 N/m ↓Z)',
                 fontsize=16, fontweight='bold', y=0.98)

    # ---- LEFT: BMD ----
    ax = axes[0]
    ax.set_title('Bending Moment MY Diagram', fontsize=13, fontweight='bold', pad=15)

    # Draw frame
    for cell in cells:
        p1, p2 = points[cell[0]], points[cell[1]]
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], 'k-', linewidth=2.5, zorder=4)

    # BMD bars
    bm_scale = 0.015
    end_moments = get_element_end_moments(cells, points, moment_Y, len(points))
    
    for idx, cell in enumerate(cells):
        p1, p2 = points[cell[0]], points[cell[1]]
        m_start, m_mid, m_end = end_moments[idx]
        draw_bars_on_element(ax, p1, p2, m_start, m_mid, m_end, bm_scale,
                             pos_color='#3366cc', neg_color='#cc3333', alpha=0.35)

    # Value labels — beams
    beam_indices = list(range(12, 30))
    for idx in beam_indices:
        p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
        mid_x = (p1[0] + p2[0]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        m_val = moment_Y[idx]
        dx, dz = p2[0] - p1[0], p2[2] - p1[2]
        L = np.sqrt(dx**2 + dz**2)
        nx, nz = -dz / L, dx / L
        label_off = m_val * bm_scale * 1.4
        ax.annotate(f'{m_val:.1f}',
                    xy=(mid_x + nx * label_off, mid_z + nz * label_off),
                    fontsize=6.5, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                              alpha=0.9, edgecolor='gray', linewidth=0.5))

    # Value labels — columns
    col_indices = list(range(0, 12))
    for idx in col_indices:
        p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
        mid_x = (p1[0] + p2[0]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        m_val = moment_Y[idx]
        offset_x = -0.9 if mid_x < 3 else 0.9
        ax.annotate(f'{m_val:.1f}',
                    xy=(mid_x + offset_x, mid_z), fontsize=6, ha='center',
                    color='dimgray',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='lightyellow',
                              alpha=0.8, edgecolor='gray', linewidth=0.3))

    # Response location
    traced_idx = traced_elem_id - 1  # Convert to 0-indexed
    draw_response_location(ax, points, cells, traced_idx, stress_location,
                           ring_radius=0.35, ring_color='darkorange')

    # UDL arrows
    draw_udl_arrows(ax, cells, points, beam_indices)

    # Nodes
    for idx, pt in enumerate(points):
        ax.plot(pt[0], pt[2], 'ko', markersize=3, zorder=5)

    # Legend
    pos_patch = mpatches.Patch(facecolor='#3366cc', alpha=0.35, label='Positive MY (sagging)')
    neg_patch = mpatches.Patch(facecolor='#cc3333', alpha=0.35, label='Negative MY (hogging)')
    frame_line = mpatches.Patch(facecolor='black', label='Frame structure')
    resp_circle = mpatches.Patch(facecolor='lightyellow', edgecolor='darkorange',
                                  linewidth=2, label='Response location')
    ax.legend(handles=[pos_patch, neg_patch, frame_line, resp_circle],
              loc='upper left', fontsize=9, framealpha=0.9)

    draw_supports(ax, points)
    draw_floor_labels(ax)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Z (m)', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    ax.margins(0.15)

    # ---- RIGHT: Deformed shape ----
    ax2 = axes[1]
    ax2.set_title('Deformed Shape & Vertical Displacement', fontsize=13,
                  fontweight='bold', pad=15)

    # Original
    for cell in cells:
        p1, p2 = points[cell[0]], points[cell[1]]
        ax2.plot([p1[0], p2[0]], [p1[2], p2[2]], 'k--', linewidth=1, alpha=0.25)

    # Deformed
    scale_disp = 5000
    for cell in cells:
        i1, i2 = cell[0], cell[1]
        p1d = points[i1] + disp[i1] * scale_disp
        p2d = points[i2] + disp[i2] * scale_disp
        ax2.plot([p1d[0], p2d[0]], [p1d[2], p2d[2]], 'r-', linewidth=2.5, alpha=0.8)

    # Node colors
    vmax_d = max(abs(disp_Z.max()), abs(disp_Z.min()))
    if vmax_d == 0:
        vmax_d = 1
    norm_d = mcolors.TwoSlopeNorm(vmin=-vmax_d, vcenter=0, vmax=vmax_d)
    scatter = ax2.scatter(points[:, 0], points[:, 2], c=disp_Z,
                          cmap=cm.coolwarm, norm=norm_d, s=80,
                          edgecolors='black', linewidth=0.5, zorder=5)
    cbar2 = plt.colorbar(scatter, ax=ax2, shrink=0.5, pad=0.03, aspect=30)
    cbar2.set_label('Displacement Z (m)', fontsize=11)

    # Max displacement
    max_idx = np.argmin(disp_Z)
    ax2.annotate(f'Max: {disp_Z[max_idx]:.2e} m\nNode {max_idx + 1}',
                 xy=(points[max_idx][0], points[max_idx][2]),
                 xytext=(30, -30), textcoords='offset points',
                 fontsize=9, fontweight='bold', color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose',
                           edgecolor='red', alpha=0.9))

    # Response location on deformed plot too
    draw_response_location(ax2, points, cells, traced_idx, stress_location,
                           ring_radius=0.35, ring_color='darkorange')

    orig_line = mpatches.Patch(facecolor='black', alpha=0.25, label='Original shape')
    def_line = mpatches.Patch(facecolor='red', alpha=0.8, label=f'Deformed (×{scale_disp})')
    resp_circle2 = mpatches.Patch(facecolor='lightyellow', edgecolor='darkorange',
                                   linewidth=2, label='Response location')
    ax2.legend(handles=[orig_line, def_line, resp_circle2],
               loc='upper right', fontsize=10)

    draw_supports(ax2, points)
    draw_floor_labels(ax2)
    ax2.set_xlabel('X (m)', fontsize=11)
    ax2.set_ylabel('Z (m)', fontsize=11)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.15)
    ax2.margins(0.12)

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


# =====================================================================
#  PLOT 2: ADJOINT — I22_SENSITIVITY with bars + Response Location
# =====================================================================
def plot_adjoint(data, traced_elem_id=14, stress_location=2, save_name=None):
    """Adjoint sensitivity with perpendicular bars and response location."""

    points = data['points']
    cells = data['cells']
    sensitivity = data['cell_data']['I22_SENSITIVITY']
    traced_idx = traced_elem_id - 1

    fig, axes = plt.subplots(1, 2, figsize=(22, 14))
    fig.suptitle('ADJOINT SENSITIVITY ANALYSIS — dMY/dI22\n'
                 f'Traced Element: {traced_elem_id} (Floor 1, Middle Beam)',
                 fontsize=16, fontweight='bold', y=0.99)

    # ---- LEFT: Frame with sensitivity bars ----
    ax = axes[0]
    ax.set_title('I22 Sensitivity Diagram on Frame', fontsize=13,
                 fontweight='bold', pad=15)

    # Draw frame baseline
    for cell in cells:
        p1, p2 = points[cell[0]], points[cell[1]]
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], 'k-', linewidth=2.5, zorder=4)

    # Draw sensitivity bars perpendicular to each element
    sens_scale = 2e-7  # Adjust for visual size
    for idx, cell in enumerate(cells):
        p1, p2 = points[cell[0]], points[cell[1]]
        draw_constant_bar_on_element(ax, p1, p2, sensitivity[idx], sens_scale,
                                      pos_color='#3366cc', neg_color='#cc3333',
                                      alpha=0.35)

    # Highlight traced element
    p1t = points[cells[traced_idx][0]]
    p2t = points[cells[traced_idx][1]]
    ax.plot([p1t[0], p2t[0]], [p1t[2], p2t[2]], color='gold',
            linewidth=12, alpha=0.5, zorder=3)

    # Response location ring
    draw_response_location(ax, points, cells, traced_idx, stress_location,
                           ring_radius=0.4, ring_color='darkorange')

    # Value labels — beams
    beam_indices = list(range(12, 30))
    for idx in beam_indices:
        p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
        mid_x = (p1[0] + p2[0]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        val = sensitivity[idx]
        dx, dz = p2[0] - p1[0], p2[2] - p1[2]
        L = np.sqrt(dx**2 + dz**2)
        nx, nz = -dz / L, dx / L
        label_off = val * sens_scale * 1.3
        face_color = '#ccddff' if val >= 0 else '#ffcccc'
        ax.annotate(f'E{idx + 1}\n{format_value(val)}',
                    xy=(mid_x + nx * label_off,
                        mid_z + nz * label_off + 0.15),
                    fontsize=6, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.12', facecolor=face_color,
                              alpha=0.9, edgecolor='gray', linewidth=0.5))

    # Value labels — columns
    col_indices = list(range(0, 12))
    for idx in col_indices:
        p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
        mid_x = (p1[0] + p2[0]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        val = sensitivity[idx]
        offset_x = -1.0 if mid_x < 3 else 1.0
        face_color = '#ccddff' if val >= 0 else '#ffcccc'
        ax.annotate(f'E{idx + 1}\n{format_value(val)}',
                    xy=(mid_x + offset_x, mid_z), fontsize=5.5, ha='center',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor=face_color,
                              alpha=0.85, edgecolor='gray', linewidth=0.3))

    # Nodes
    for idx, pt in enumerate(points):
        ax.plot(pt[0], pt[2], 'ko', markersize=3, zorder=5)

    # Legend
    pos_patch = mpatches.Patch(facecolor='#3366cc', alpha=0.35,
                               label='Positive: stiffening INCREASES MY')
    neg_patch = mpatches.Patch(facecolor='#cc3333', alpha=0.35,
                               label='Negative: stiffening DECREASES MY')
    traced_patch = mpatches.Patch(facecolor='gold', edgecolor='darkorange',
                                  alpha=0.5, label=f'Traced element (E{traced_elem_id})')
    resp_patch = mpatches.Patch(facecolor='lightyellow', edgecolor='darkorange',
                                 linewidth=2, label='Response location')
    ax.legend(handles=[pos_patch, neg_patch, traced_patch, resp_patch],
              loc='upper left', fontsize=9, framealpha=0.9)

    draw_supports(ax, points)
    draw_floor_labels(ax)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Z (m)', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    ax.margins(0.2)

    # ---- RIGHT: Bar chart ----
    ax2 = axes[1]
    ax2.set_title('I22 Sensitivity by Structural Member', fontsize=13,
                  fontweight='bold', pad=15)

    groups = {
        'Left\nColumn': (list(range(0, 6)), '#4A90D9'),
        'Right\nColumn': (list(range(6, 12)), '#6BA3D6'),
        'Floor 1\nBeams': (list(range(12, 15)), '#E74C3C'),
        'Floor 2\nBeams': (list(range(15, 18)), '#E67E22'),
        'Floor 3\nBeams': (list(range(18, 21)), '#F1C40F'),
        'Floor 4\nBeams': (list(range(21, 24)), '#2ECC71'),
        'Floor 5\nBeams': (list(range(24, 27)), '#1ABC9C'),
        'Floor 6\nBeams': (list(range(27, 30)), '#9B59B6'),
    }

    x_pos = 0
    x_ticks = []
    x_labels_list = []
    group_mids = {}

    for group_name, (indices, color) in groups.items():
        start_pos = x_pos
        for idx in indices:
            val = sensitivity[idx]
            edge_color = 'darkorange' if idx == traced_idx else 'black'
            edge_width = 3 if idx == traced_idx else 0.5
            hatch = '///' if idx == traced_idx else ''

            ax2.bar(x_pos, val, color=color, edgecolor=edge_color,
                    linewidth=edge_width, width=0.8, alpha=0.8, hatch=hatch)

            # Value on bar
            va = 'bottom' if val >= 0 else 'top'
            offset = abs(val) * 0.08
            y_text = val + offset if val >= 0 else val - offset
            ax2.text(x_pos, y_text, format_value(val),
                     ha='center', va=va, fontsize=5.5, fontweight='bold',
                     rotation=90)

            x_ticks.append(x_pos)
            x_labels_list.append(f'E{idx + 1}')
            x_pos += 1

        group_mids[group_name] = (start_pos + x_pos - 1) / 2
        ax2.axvline(x=x_pos - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        x_pos += 0.5

    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels_list, fontsize=6, rotation=45, ha='right')
    ax2.set_ylabel('I22_SENSITIVITY (dMY/dI22)', fontsize=11)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.grid(axis='y', alpha=0.2)

    # Group labels at bottom
    for group_name, (indices, color) in groups.items():
        mid = group_mids[group_name]
        ax2.text(mid, ax2.get_ylim()[0] * 0.92, group_name,
                 ha='center', fontsize=7, fontweight='bold',
                 bbox=dict(facecolor=color, alpha=0.3, pad=3,
                           boxstyle='round,pad=0.3'))

    # Traced element annotation
    ax2.annotate(f'◄ TRACED (E{traced_elem_id})',
                 xy=(x_ticks[traced_idx], sensitivity[traced_idx]),
                 xytext=(x_ticks[traced_idx] + 4, sensitivity[traced_idx] * 0.5),
                 fontsize=10, fontweight='bold', color='darkorange',
                 arrowprops=dict(arrowstyle='->', color='darkorange', lw=2))

    # Response location annotation on bar chart
    resp_node = cells[traced_idx][1] if stress_location == 2 else cells[traced_idx][0]
    ax2.text(0.98, 0.02, f'Response: MY at Node {resp_node + 1}\n'
             f'(Element {traced_elem_id}, location {stress_location})',
             transform=ax2.transAxes, fontsize=9, va='bottom', ha='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                       edgecolor='darkorange', linewidth=1.5, alpha=0.95))

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

    # Configuration — change these to match your adjoint JSON
    TRACED_ELEMENT_ID = 14     # 1-indexed element ID
    STRESS_LOCATION = 2        # 1 = first node, 2 = second node

    # ---- PRIMAL ----
    primal_file = "vtk_output_primal/Structure_0_1.vtk"
    if os.path.exists(primal_file):
        primal = read_vtk_file(primal_file)
        print("Plotting primal results with BMD bars + response location...")
        plot_primal(primal,
                    traced_elem_id=TRACED_ELEMENT_ID,
                    stress_location=STRESS_LOCATION,
                    save_name='primal_BMD_with_response.png')
    else:
        print(f"Primal file not found: {primal_file}")

    # ---- ADJOINT ----
    adjoint_file = "vtk_output_adjoint/Structure_0_1.vtk"
    if os.path.exists(adjoint_file):
        adjoint = read_vtk_file(adjoint_file)
        print("Plotting adjoint sensitivity with bars + response location...")
        plot_adjoint(adjoint,
                     traced_elem_id=TRACED_ELEMENT_ID,
                     stress_location=STRESS_LOCATION,
                     save_name='adjoint_sensitivity_with_response.png')
    else:
        print(f"Adjoint file not found: {adjoint_file}")

    print("\nDone! All plots generated.")