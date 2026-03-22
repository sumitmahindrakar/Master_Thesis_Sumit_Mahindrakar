"""
Clear and Visual VTK Frame Plotter
With Bending Moment Diagram bars perpendicular to elements.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
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
                        values_list = []
                        while len(values_list) < n_tuples * n_comp:
                            values_list.extend([float(v) for v in lines[i].strip().split()])
                            i += 1
                        arr = np.array(values_list)
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
                        values_list = []
                        while len(values_list) < n_tuples * n_comp:
                            values_list.extend([float(v) for v in lines[i].strip().split()])
                            i += 1
                        arr = np.array(values_list)
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
        triangle = plt.Polygon(
            [[x - 0.3, z - 0.5], [x + 0.3, z - 0.5], [x, z]],
            closed=True, facecolor='gray', edgecolor='black', linewidth=1.5, zorder=6
        )
        ax.add_patch(triangle)
        ax.plot([x - 0.5, x + 0.5], [z - 0.5, z - 0.5], 'k-', linewidth=2)
        for hx in np.linspace(x - 0.4, x + 0.4, 5):
            ax.plot([hx, hx - 0.15], [z - 0.5, z - 0.75], 'k-', linewidth=0.8)


def draw_floor_labels(ax, x_right=6.0):
    """Draw floor level labels."""
    floors = {
        0: 'Ground', 3: 'Floor 1', 6: 'Floor 2', 9: 'Floor 3',
        12: 'Floor 4', 15: 'Floor 5', 18: 'Floor 6'
    }
    for z, label in floors.items():
        ax.annotate(label, xy=(x_right + 0.8, z), fontsize=8, color='gray',
                    va='center', fontstyle='italic')


def draw_udl_arrows(ax, cells, points, beam_indices, color='blue', alpha=0.3):
    """Draw UDL arrows on beam elements."""
    for idx in beam_indices:
        p1 = points[cells[idx][0]]
        p2 = points[cells[idx][1]]
        z = p1[2]
        for xx in np.linspace(p1[0] + 0.15, p2[0] - 0.15, 4):
            ax.annotate('', xy=(xx, z), xytext=(xx, z + 0.4),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.2, alpha=alpha))


def get_element_end_moments(cells, points, moment_mid, n_points):
    """
    Estimate bending moment at element ends from midpoint values.
    For a beam with UDL: M varies parabolically.
    For columns: M varies linearly.
    
    Uses averaging at shared nodes for continuity.
    """
    # For each element, store [M_start, M_end]
    # Simple approach: use midpoint value at both ends initially,
    # then average at shared nodes
    
    node_moments = {}  # node_id -> list of (element_idx, moment_value)
    
    for idx, cell in enumerate(cells):
        n1, n2 = cell[0], cell[1]
        m_mid = moment_mid[idx]
        
        if n1 not in node_moments:
            node_moments[n1] = []
        if n2 not in node_moments:
            node_moments[n2] = []
        
        node_moments[n1].append(m_mid)
        node_moments[n2].append(m_mid)
    
    # Average at each node
    node_avg = {}
    for nid, vals in node_moments.items():
        node_avg[nid] = np.mean(vals)
    
    # Build end moments for each element
    end_moments = []
    for idx, cell in enumerate(cells):
        n1, n2 = cell[0], cell[1]
        m1 = node_avg[n1]
        m2 = node_avg[n2]
        m_mid = moment_mid[idx]
        end_moments.append((m1, m_mid, m2))
    
    return end_moments


def draw_bmd_on_element(ax, p1_3d, p2_3d, m_start, m_mid, m_end, scale,
                        pos_color='#4444ff', neg_color='#ff4444', alpha=0.25,
                        n_segments=20):
    """
    Draw bending moment diagram as filled polygon perpendicular to element.
    Uses parabolic interpolation through (m_start, m_mid, m_end).
    
    Convention: positive moment drawn on tension side.
    """
    # Element in X-Z plane
    x1, z1 = p1_3d[0], p1_3d[2]
    x2, z2 = p2_3d[0], p2_3d[2]
    
    # Element direction and normal
    dx = x2 - x1
    dz = z2 - z1
    L = np.sqrt(dx**2 + dz**2)
    if L < 1e-10:
        return
    
    # Unit tangent and normal (perpendicular, rotated 90° CCW)
    tx, tz = dx / L, dz / L
    nx, nz = -tz, tx  # Normal pointing "left" of element direction
    
    # Parametric position along element: t in [0, 1]
    t_vals = np.linspace(0, 1, n_segments + 1)
    
    # Parabolic interpolation: M(t) = a*t^2 + b*t + c
    # M(0) = m_start, M(0.5) = m_mid, M(1) = m_end
    c = m_start
    a = 2 * m_end + 2 * m_start - 4 * m_mid
    b = m_end - m_start - a
    
    m_vals = a * t_vals**2 + b * t_vals + c
    
    # Build polygon points: element baseline + offset by moment
    base_x = x1 + t_vals * dx
    base_z = z1 + t_vals * dz
    
    offset_x = m_vals * scale * nx
    offset_z = m_vals * scale * nz
    
    bmd_x = base_x + offset_x
    bmd_z = base_z + offset_z
    
    # Split into positive and negative regions for coloring
    # Draw as single filled polygon between baseline and BMD curve
    
    # Forward path (BMD curve)
    poly_x = np.concatenate([base_x, bmd_x[::-1]])
    poly_z = np.concatenate([base_z, bmd_z[::-1]])
    
    polygon_pts = list(zip(poly_x, poly_z))
    
    # Determine dominant color
    if np.mean(m_vals) >= 0:
        color = pos_color
    else:
        color = neg_color
    
    poly = Polygon(polygon_pts, closed=True, facecolor=color, edgecolor='none',
                   alpha=alpha, zorder=2)
    ax.add_patch(poly)
    
    # Draw BMD outline
    ax.plot(bmd_x, bmd_z, color=color.replace('44', '00'), linewidth=1.5,
            alpha=0.8, zorder=3)
    
    # Draw closing lines at ends
    ax.plot([base_x[0], bmd_x[0]], [base_z[0], bmd_z[0]],
            color='gray', linewidth=0.8, alpha=0.6, zorder=3)
    ax.plot([base_x[-1], bmd_x[-1]], [base_z[-1], bmd_z[-1]],
            color='gray', linewidth=0.8, alpha=0.6, zorder=3)


def draw_bmd_connected(ax, data, component=1, scale=0.02,
                       pos_color='#4444ff', neg_color='#ff4444',
                       alpha=0.3, show_values=True, value_fontsize=6):
    """
    Draw connected bending moment diagram on all elements.
    
    Parameters:
        data: dict from read_vtk_file
        component: moment component (0=MX, 1=MY, 2=MZ)
        scale: visual scaling factor for moment bars
        pos_color: color for positive moment
        neg_color: color for negative moment
    """
    points = data['points']
    cells = data['cells']
    moment = data['cell_data']['MOMENT']
    moment_comp = moment[:, component]
    
    # Get end moments for continuity
    end_moments = get_element_end_moments(cells, points, moment_comp, len(points))
    
    for idx, cell in enumerate(cells):
        p1 = points[cell[0]]
        p2 = points[cell[1]]
        m_start, m_mid, m_end = end_moments[idx]
        
        draw_bmd_on_element(ax, p1, p2, m_start, m_mid, m_end, scale,
                           pos_color=pos_color, neg_color=neg_color,
                           alpha=alpha)
    
    # Add value labels at midpoints
    if show_values:
        beam_indices = list(range(12, 30))
        col_indices = list(range(0, 12))
        
        for idx in beam_indices:
            p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
            mid_x = (p1[0] + p2[0]) / 2
            mid_z = (p1[2] + p2[2]) / 2
            
            # Offset label away from BMD
            m_val = moment_comp[idx]
            dx = p2[0] - p1[0]
            dz = p2[2] - p1[2]
            L = np.sqrt(dx**2 + dz**2)
            nx, nz = -dz / L, dx / L
            label_offset = m_val * scale * 1.3
            
            ax.annotate(f'{m_val:.1f}',
                       xy=(mid_x + nx * label_offset, mid_z + nz * label_offset),
                       fontsize=value_fontsize, ha='center', va='center',
                       fontweight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                                 alpha=0.9, edgecolor='gray', linewidth=0.5))
        
        for idx in col_indices:
            p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
            mid_x = (p1[0] + p2[0]) / 2
            mid_z = (p1[2] + p2[2]) / 2
            m_val = moment_comp[idx]
            offset_x = -0.8 if mid_x < 3 else 0.8
            
            ax.annotate(f'{m_val:.1f}',
                       xy=(mid_x + offset_x, mid_z),
                       fontsize=value_fontsize - 0.5, ha='center', va='center',
                       color='dimgray',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='lightyellow',
                                 alpha=0.8, edgecolor='gray', linewidth=0.3))


def draw_sfd_on_element(ax, p1_3d, p2_3d, v_val, scale,
                        pos_color='#22aa22', neg_color='#aa2222', alpha=0.2):
    """Draw shear force diagram as constant rectangle on element."""
    x1, z1 = p1_3d[0], p1_3d[2]
    x2, z2 = p2_3d[0], p2_3d[2]
    
    dx = x2 - x1
    dz = z2 - z1
    L = np.sqrt(dx**2 + dz**2)
    if L < 1e-10:
        return
    
    tx, tz = dx / L, dz / L
    nx, nz = -tz, tx
    
    offset = v_val * scale
    
    poly_x = [x1, x2, x2 + nx * offset, x1 + nx * offset]
    poly_z = [z1, z2, z2 + nz * offset, z1 + nz * offset]
    
    color = pos_color if v_val >= 0 else neg_color
    
    poly = Polygon(list(zip(poly_x, poly_z)), closed=True,
                   facecolor=color, edgecolor=color.replace('22', '00'),
                   alpha=alpha, linewidth=1, zorder=2)
    ax.add_patch(poly)


# =====================================================================
#  PLOT 1: PRIMAL with BMD bars
# =====================================================================
def plot_primal(data, save_name=None):
    """Primal results with connected BMD bars and displacement."""

    points = data['points']
    cells = data['cells']
    moment = data['cell_data']['MOMENT']
    disp = data['point_data']['DISPLACEMENT']
    
    moment_Y = moment[:, 1]
    disp_Z = disp[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(22, 14))
    fig.suptitle('PRIMAL ANALYSIS — Fixed-Fixed 6-Floor Frame with UDL (q = 40 N/m ↓Z)',
                 fontsize=16, fontweight='bold', y=0.98)

    # ---- LEFT: Bending Moment Diagram with bars ----
    ax = axes[0]
    ax.set_title('Bending Moment MY Diagram', fontsize=13, fontweight='bold', pad=15)

    # Draw frame structure (baseline)
    for cell in cells:
        p1, p2 = points[cell[0]], points[cell[1]]
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], 'k-', linewidth=2.5, zorder=4)

    # Draw BMD with connected bars
    bm_scale = 0.015  # Adjust this to make diagram bigger/smaller
    draw_bmd_connected(ax, data, component=1, scale=bm_scale,
                       pos_color='#3366cc', neg_color='#cc3333',
                       alpha=0.3, show_values=True, value_fontsize=6.5)

    # UDL arrows
    beam_indices = list(range(12, 30))
    draw_udl_arrows(ax, cells, points, beam_indices)

    # Nodes
    for idx, pt in enumerate(points):
        ax.plot(pt[0], pt[2], 'ko', markersize=4, zorder=5)

    # Legend
    pos_patch = mpatches.Patch(facecolor='#3366cc', alpha=0.3,
                               label='Positive MY (sagging)')
    neg_patch = mpatches.Patch(facecolor='#cc3333', alpha=0.3,
                               label='Negative MY (hogging)')
    frame_line = mpatches.Patch(facecolor='black', label='Frame structure')
    ax.legend(handles=[pos_patch, neg_patch, frame_line],
              loc='upper left', fontsize=9, framealpha=0.9)

    draw_supports(ax, points)
    draw_floor_labels(ax)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Z (m)', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    ax.margins(0.15)

    # ---- RIGHT: Deformed shape with displacement ----
    ax2 = axes[1]
    ax2.set_title('Deformed Shape & Vertical Displacement', fontsize=13,
                  fontweight='bold', pad=15)

    # Original frame (gray dashed)
    for cell in cells:
        p1, p2 = points[cell[0]], points[cell[1]]
        ax2.plot([p1[0], p2[0]], [p1[2], p2[2]], 'k--', linewidth=1, alpha=0.25)

    # Deformed shape
    scale_disp = 5000
    for cell in cells:
        i1, i2 = cell[0], cell[1]
        p1_def = points[i1] + disp[i1] * scale_disp
        p2_def = points[i2] + disp[i2] * scale_disp
        ax2.plot([p1_def[0], p2_def[0]], [p1_def[2], p2_def[2]],
                 'r-', linewidth=2.5, alpha=0.8)

    # Node colors by displacement
    vmax_d = max(abs(disp_Z.max()), abs(disp_Z.min()))
    if vmax_d == 0:
        vmax_d = 1
    norm_d = mcolors.TwoSlopeNorm(vmin=-vmax_d, vcenter=0, vmax=vmax_d)

    scatter = ax2.scatter(points[:, 0], points[:, 2], c=disp_Z,
                          cmap=cm.coolwarm, norm=norm_d, s=80,
                          edgecolors='black', linewidth=0.5, zorder=5)

    cbar2 = plt.colorbar(scatter, ax=ax2, shrink=0.5, pad=0.03, aspect=30)
    cbar2.set_label('Displacement Z (m)', fontsize=11)

    # Max displacement annotation
    max_idx = np.argmin(disp_Z)
    ax2.annotate(f'Max: {disp_Z[max_idx]:.2e} m\nNode {max_idx + 1}',
                 xy=(points[max_idx][0], points[max_idx][2]),
                 xytext=(30, -30), textcoords='offset points',
                 fontsize=9, fontweight='bold', color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose',
                           edgecolor='red', alpha=0.9))

    original_line = mpatches.Patch(facecolor='black', alpha=0.25, label='Original shape')
    deformed_line = mpatches.Patch(facecolor='red', alpha=0.8, label=f'Deformed (×{scale_disp})')
    ax2.legend(handles=[original_line, deformed_line], loc='upper right', fontsize=10)

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
#  PLOT 2: ADJOINT — I22_SENSITIVITY
# =====================================================================
def plot_adjoint(data, save_name=None):
    """Adjoint sensitivity plot with frame and bar chart."""

    points = data['points']
    cells = data['cells']
    sensitivity = data['cell_data']['I22_SENSITIVITY']

    fig, axes = plt.subplots(1, 2, figsize=(22, 14))
    fig.suptitle('ADJOINT SENSITIVITY ANALYSIS — dMY/dI22\n'
                 'Traced Element: 14 (Floor 1, Middle Beam)',
                 fontsize=16, fontweight='bold', y=0.99)

    # ---- LEFT: Frame with sensitivity ----
    ax = axes[0]
    ax.set_title('I22 Sensitivity on Frame Structure', fontsize=13,
                 fontweight='bold', pad=15)

    # Draw sensitivity as bars perpendicular to elements
    sens_scale = 2e-7  # Adjust for visual size
    
    for idx, cell in enumerate(cells):
        p1 = points[cell[0]]
        p2 = points[cell[1]]
        val = sensitivity[idx]
        
        # Draw baseline element
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], 'k-', linewidth=2, zorder=4)
        
        # Draw sensitivity bar perpendicular to element
        x1, z1 = p1[0], p1[2]
        x2, z2 = p2[0], p2[2]
        dx = x2 - x1
        dz = z2 - z1
        L = np.sqrt(dx**2 + dz**2)
        if L < 1e-10:
            continue
        
        nx, nz = -dz / L, dx / L
        offset = val * sens_scale
        
        # Rectangle from element to offset
        poly_x = [x1, x2, x2 + nx * offset, x1 + nx * offset]
        poly_z = [z1, z2, z2 + nz * offset, z1 + nz * offset]
        
        color = '#3366cc' if val >= 0 else '#cc3333'
        poly = Polygon(list(zip(poly_x, poly_z)), closed=True,
                       facecolor=color, edgecolor=color, alpha=0.35,
                       linewidth=1, zorder=2)
        ax.add_patch(poly)
        
        # Outline
        ax.plot([x1 + nx * offset, x2 + nx * offset],
                [z1 + nz * offset, z2 + nz * offset],
                color=color, linewidth=1.5, alpha=0.7, zorder=3)
        ax.plot([x1, x1 + nx * offset], [z1, z1 + nz * offset],
                color='gray', linewidth=0.5, alpha=0.5, zorder=3)
        ax.plot([x2, x2 + nx * offset], [z2, z2 + nz * offset],
                color='gray', linewidth=0.5, alpha=0.5, zorder=3)

    # Highlight traced element
    traced_idx = 13
    p1, p2 = points[cells[traced_idx][0]], points[cells[traced_idx][1]]
    ax.plot([p1[0], p2[0]], [p1[2], p2[2]], color='gold', linewidth=12,
            alpha=0.5, zorder=3)
    mid_x = (p1[0] + p2[0]) / 2
    mid_z = (p1[2] + p2[2]) / 2
    ax.annotate('TRACED\nE14', xy=(mid_x, mid_z - 0.7),
                fontsize=9, ha='center', fontweight='bold', color='darkorange',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor='darkorange', linewidth=2))

    # Value labels
    beam_indices = list(range(12, 30))
    col_indices = list(range(0, 12))

    for idx in beam_indices:
        p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
        mid_x = (p1[0] + p2[0]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        val = sensitivity[idx]
        dx = p2[0] - p1[0]
        dz = p2[2] - p1[2]
        L = np.sqrt(dx**2 + dz**2)
        nx, nz = -dz / L, dx / L
        label_offset = val * sens_scale * 1.2
        
        face_color = '#ccddff' if val >= 0 else '#ffcccc'
        ax.annotate(f'E{idx + 1}\n{format_value(val)}',
                    xy=(mid_x + nx * label_offset, mid_z + nz * label_offset + 0.2),
                    fontsize=6, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.12', facecolor=face_color,
                              alpha=0.9, edgecolor='gray', linewidth=0.5))

    for idx in col_indices:
        p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
        mid_x = (p1[0] + p2[0]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        val = sensitivity[idx]
        offset_x = -1.0 if mid_x < 3 else 1.0
        face_color = '#ccddff' if val >= 0 else '#ffcccc'
        
        ax.annotate(f'E{idx + 1}\n{format_value(val)}',
                    xy=(mid_x + offset_x, mid_z),
                    fontsize=5.5, ha='center',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor=face_color,
                              alpha=0.85, edgecolor='gray', linewidth=0.3))

    # Nodes
    for idx, pt in enumerate(points):
        ax.plot(pt[0], pt[2], 'ko', markersize=4, zorder=5)

    # Legend
    pos_patch = mpatches.Patch(facecolor='#3366cc', alpha=0.35,
                               label='Positive: stiffening INCREASES MY at E14')
    neg_patch = mpatches.Patch(facecolor='#cc3333', alpha=0.35,
                               label='Negative: stiffening DECREASES MY at E14')
    traced_patch = mpatches.Patch(facecolor='gold', edgecolor='darkorange',
                                  alpha=0.5, label='Traced element (E14)')
    ax.legend(handles=[pos_patch, neg_patch, traced_patch],
              loc='upper left', fontsize=9, framealpha=0.9)

    draw_supports(ax, points)
    draw_floor_labels(ax)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Z (m)', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    ax.margins(0.2)

    # ---- RIGHT: Bar chart grouped ----
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

    for group_name, (indices, color) in groups.items():
        for idx in indices:
            val = sensitivity[idx]
            edge_color = 'gold' if idx == 13 else 'black'
            edge_width = 3 if idx == 13 else 0.5

            ax2.bar(x_pos, val, color=color, edgecolor=edge_color,
                    linewidth=edge_width, width=0.8, alpha=0.8)

            va = 'bottom' if val >= 0 else 'top'
            offset = abs(val) * 0.08
            y_text = val + offset if val >= 0 else val - offset
            ax2.text(x_pos, y_text, format_value(val),
                     ha='center', va=va, fontsize=5.5, fontweight='bold', rotation=90)

            x_ticks.append(x_pos)
            x_labels_list.append(f'E{idx + 1}')
            x_pos += 1

        ax2.axvline(x=x_pos - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        x_pos += 0.5

    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels_list, fontsize=6, rotation=45, ha='right')
    ax2.set_ylabel('I22_SENSITIVITY (dMY/dI22)', fontsize=11)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.grid(axis='y', alpha=0.2)

    # Group labels
    x_pos_reset = 0
    for group_name, (indices, color) in groups.items():
        mid = x_pos_reset + len(indices) / 2 - 0.5
        ax2.text(mid, ax2.get_ylim()[0] * 0.9, group_name,
                 ha='center', fontsize=7, fontweight='bold',
                 bbox=dict(facecolor=color, alpha=0.3, pad=3,
                           boxstyle='round,pad=0.3'))
        x_pos_reset += len(indices) + 0.5

    # Traced element annotation
    ax2.annotate('◄ TRACED', xy=(x_ticks[13], sensitivity[13]),
                 xytext=(x_ticks[13] + 4, sensitivity[13] * 0.6),
                 fontsize=10, fontweight='bold', color='darkorange',
                 arrowprops=dict(arrowstyle='->', color='darkorange', lw=2))

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

    # ---- PRIMAL ----
    primal_file = "vtk_output_primal/Structure_0_1.vtk"
    if os.path.exists(primal_file):
        primal = read_vtk_file(primal_file)
        print("Plotting primal results with BMD bars...")
        plot_primal(primal, save_name='primal_BMD_results.png')
    else:
        print(f"Primal file not found: {primal_file}")

    # ---- ADJOINT ----
    adjoint_file = "vtk_output_adjoint/Structure_0_1.vtk"
    if os.path.exists(adjoint_file):
        adjoint = read_vtk_file(adjoint_file)
        print("Plotting adjoint sensitivity with bars...")
        plot_adjoint(adjoint, save_name='adjoint_sensitivity_bars.png')
    else:
        print(f"Adjoint file not found: {adjoint_file}")

    print("\nDone! All plots generated.")