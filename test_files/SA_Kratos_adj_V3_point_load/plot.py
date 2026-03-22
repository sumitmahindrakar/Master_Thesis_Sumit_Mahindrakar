"""
Clear and Visual VTK Frame Plotter
Easy-to-understand plots for I22_SENSITIVITY and Primal results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Rectangle
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
                        values = []
                        while len(values) < n_tuples * n_comp:
                            values.extend([float(v) for v in lines[i].strip().split()])
                            i += 1
                        arr = np.array(values)
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
                        values = []
                        while len(values) < n_tuples * n_comp:
                            values.extend([float(v) for v in lines[i].strip().split()])
                            i += 1
                        arr = np.array(values)
                        data['cell_data'][name] = arr.reshape(n_tuples, n_comp) if n_comp > 1 else arr
                    continue
                i += 1
            continue
        
        i += 1
    return data


def format_value(val):
    """Format large numbers in readable scientific notation."""
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
    """Draw fixed support triangles and ground hatching."""
    for sn in support_indices:
        x, z = points[sn][0], points[sn][2]
        # Triangle
        triangle = plt.Polygon(
            [[x - 0.3, z - 0.5], [x + 0.3, z - 0.5], [x, z]],
            closed=True, facecolor='gray', edgecolor='black', linewidth=1.5, zorder=6
        )
        ax.add_patch(triangle)
        # Ground line
        ax.plot([x - 0.5, x + 0.5], [z - 0.5, z - 0.5], 'k-', linewidth=2)
        # Hatching
        for hx in np.linspace(x - 0.4, x + 0.4, 5):
            ax.plot([hx, hx - 0.15], [z - 0.5, z - 0.75], 'k-', linewidth=0.8)


def draw_floor_labels(ax, x_right=6.0):
    """Draw floor level labels on the right side."""
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
        mid_x = (p1[0] + p2[0]) / 2
        z = p1[2]
        for xx in np.linspace(p1[0] + 0.15, p2[0] - 0.15, 4):
            ax.annotate('', xy=(xx, z), xytext=(xx, z + 0.4),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.2, alpha=alpha))


# =====================================================================
#  PLOT 1: PRIMAL — Frame with BM and Displacement
# =====================================================================
def plot_primal(data, save_name=None):
    """Beautiful primal results plot with BM diagram and displacement."""

    points = data['points']
    cells = data['cells']
    moment = data['cell_data']['MOMENT']
    disp = data['point_data']['DISPLACEMENT']
    
    moment_Y = moment[:, 1]  # MY component
    disp_Z = disp[:, 2]      # Vertical displacement

    fig, axes = plt.subplots(1, 2, figsize=(20, 14))
    fig.suptitle('PRIMAL ANALYSIS — Fixed-Fixed 6-Floor Frame with UDL (q = 40 N/m)',
                 fontsize=16, fontweight='bold', y=0.98)

    # ---- LEFT: Bending Moment Diagram ----
    ax = axes[0]
    ax.set_title('Bending Moment MY Distribution', fontsize=13, fontweight='bold', pad=15)

    # Color scale
    vmax = max(abs(moment_Y.max()), abs(moment_Y.min()))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = cm.RdBu_r

    # Draw elements with color
    segments = []
    for cell in cells:
        p1, p2 = points[cell[0]], points[cell[1]]
        segments.append([[p1[0], p1[2]], [p2[0], p2[2]]])

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=8)
    lc.set_array(moment_Y)
    ax.add_collection(lc)

    cbar = plt.colorbar(lc, ax=ax, shrink=0.5, pad=0.03, aspect=30)
    cbar.set_label('Bending Moment MY (Nm)', fontsize=11)

    # Element value labels — only beams (large values)
    beam_indices = list(range(12, 30))
    for idx in beam_indices:
        p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
        mid_x = (p1[0] + p2[0]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        ax.annotate(f'{moment_Y[idx]:.1f}',
                    xy=(mid_x, mid_z + 0.35), fontsize=7, ha='center',
                    fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              alpha=0.9, edgecolor='gray', linewidth=0.5))

    # Column labels (smaller font)
    col_indices = list(range(0, 12))
    for idx in col_indices:
        p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
        mid_x = (p1[0] + p2[0]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        offset_x = -0.7 if mid_x < 3 else 0.7
        ax.annotate(f'{moment_Y[idx]:.1f}',
                    xy=(mid_x + offset_x, mid_z), fontsize=6, ha='center',
                    color='dimgray',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='lightyellow',
                              alpha=0.8, edgecolor='gray', linewidth=0.3))

    # UDL arrows
    draw_udl_arrows(ax, cells, points, beam_indices)

    # Nodes
    for idx, pt in enumerate(points):
        ax.plot(pt[0], pt[2], 'ko', markersize=3, zorder=5)

    draw_supports(ax, points)
    draw_floor_labels(ax)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Z (m)', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    ax.margins(0.12)

    # ---- RIGHT: Displacement ----
    ax2 = axes[1]
    ax2.set_title('Vertical Displacement (Z-component)', fontsize=13, fontweight='bold', pad=15)

    # Frame outline
    for cell in cells:
        p1, p2 = points[cell[0]], points[cell[1]]
        ax2.plot([p1[0], p2[0]], [p1[2], p2[2]], 'k-', linewidth=1, alpha=0.2)

    # Deformed shape (amplified)
    scale = 5000  # Amplification factor
    for cell in cells:
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = points[p1_idx] + disp[p1_idx] * scale
        p2 = points[p2_idx] + disp[p2_idx] * scale
        ax2.plot([p1[0], p2[0]], [p1[2], p2[2]], 'r-', linewidth=2.5, alpha=0.7)

    # Node displacement values
    vmax_d = max(abs(disp_Z.max()), abs(disp_Z.min()))
    if vmax_d == 0:
        vmax_d = 1
    norm_d = mcolors.TwoSlopeNorm(vmin=-vmax_d, vcenter=0, vmax=vmax_d)

    scatter = ax2.scatter(points[:, 0], points[:, 2], c=disp_Z,
                          cmap=cm.coolwarm, norm=norm_d, s=80,
                          edgecolors='black', linewidth=0.5, zorder=5)

    cbar2 = plt.colorbar(scatter, ax=ax2, shrink=0.5, pad=0.03, aspect=30)
    cbar2.set_label('Displacement Z (m)', fontsize=11)

    # Show max displacement
    max_disp_idx = np.argmin(disp_Z)
    ax2.annotate(f'Max: {disp_Z[max_disp_idx]:.2e} m\nNode {max_disp_idx + 1}',
                 xy=(points[max_disp_idx][0], points[max_disp_idx][2]),
                 xytext=(30, -30), textcoords='offset points',
                 fontsize=9, fontweight='bold', color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose',
                           edgecolor='red', alpha=0.9))

    # Legend
    original_line = mpatches.Patch(color='black', alpha=0.2, label='Original shape')
    deformed_line = mpatches.Patch(color='red', alpha=0.7, label=f'Deformed (×{scale})')
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
    """Beautiful adjoint sensitivity plot."""

    points = data['points']
    cells = data['cells']
    sensitivity = data['cell_data']['I22_SENSITIVITY']

    fig, axes = plt.subplots(1, 2, figsize=(20, 14))
    fig.suptitle('ADJOINT SENSITIVITY ANALYSIS — dMY/dI22\n'
                 'Traced Element: 14 (Floor 1, Middle Beam)',
                 fontsize=16, fontweight='bold', y=0.99)

    # ---- LEFT: Frame with sensitivity colors ----
    ax = axes[0]
    ax.set_title('I22 Sensitivity on Frame Structure', fontsize=13, fontweight='bold', pad=15)

    # Use log-like scaling for better visualization
    abs_vals = np.abs(sensitivity)
    signs = np.sign(sensitivity)

    # Symmetric log norm for huge range of values
    vmax = abs_vals.max()
    norm = mcolors.SymLogNorm(linthresh=1000, vmin=-vmax, vmax=vmax)
    cmap = cm.RdBu_r

    # Draw elements as thick colored lines
    segments = []
    for cell in cells:
        p1, p2 = points[cell[0]], points[cell[1]]
        segments.append([[p1[0], p1[2]], [p2[0], p2[2]]])

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=8)
    lc.set_array(sensitivity)
    ax.add_collection(lc)

    cbar = plt.colorbar(lc, ax=ax, shrink=0.5, pad=0.03, aspect=30)
    cbar.set_label('I22_SENSITIVITY (dMY/dI22)', fontsize=11)

    # Highlight traced element
    traced_idx = 13
    p1, p2 = points[cells[traced_idx][0]], points[cells[traced_idx][1]]
    ax.plot([p1[0], p2[0]], [p1[2], p2[2]], color='gold', linewidth=14, alpha=0.5, zorder=3)
    mid_x = (p1[0] + p2[0]) / 2
    mid_z = (p1[2] + p2[2]) / 2
    ax.annotate('TRACED\nElement 14', xy=(mid_x, mid_z - 0.6),
                fontsize=9, ha='center', fontweight='bold', color='darkorange',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor='darkorange', linewidth=2))

    # Element labels with formatted values
    beam_indices = list(range(12, 30))
    col_indices = list(range(0, 12))

    for idx in beam_indices:
        p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
        mid_x = (p1[0] + p2[0]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        val_str = format_value(sensitivity[idx])
        face_color = '#ffcccc' if sensitivity[idx] < 0 else '#ccddff'
        ax.annotate(f'E{idx + 1}\n{val_str}',
                    xy=(mid_x, mid_z + 0.4), fontsize=6, ha='center',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=face_color,
                              alpha=0.9, edgecolor='gray', linewidth=0.5))

    for idx in col_indices:
        p1, p2 = points[cells[idx][0]], points[cells[idx][1]]
        mid_x = (p1[0] + p2[0]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        offset_x = -0.8 if mid_x < 3 else 0.8
        val_str = format_value(sensitivity[idx])
        face_color = '#ffcccc' if sensitivity[idx] < 0 else '#ccddff'
        ax.annotate(f'E{idx + 1}\n{val_str}',
                    xy=(mid_x + offset_x, mid_z), fontsize=5.5, ha='center',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor=face_color,
                              alpha=0.85, edgecolor='gray', linewidth=0.3))

    # Nodes
    for idx, pt in enumerate(points):
        ax.plot(pt[0], pt[2], 'ko', markersize=3, zorder=5)

    # Legend for colors
    pos_patch = mpatches.Patch(facecolor='#ccddff', edgecolor='gray',
                               label='Positive: stiffening INCREASES MY')
    neg_patch = mpatches.Patch(facecolor='#ffcccc', edgecolor='gray',
                               label='Negative: stiffening DECREASES MY')
    traced_patch = mpatches.Patch(facecolor='gold', edgecolor='darkorange',
                                  alpha=0.5, label='Traced element')
    ax.legend(handles=[pos_patch, neg_patch, traced_patch],
              loc='upper left', fontsize=9, framealpha=0.9)

    draw_supports(ax, points)
    draw_floor_labels(ax)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Z (m)', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    ax.margins(0.12)

    # ---- RIGHT: Bar chart by floor ----
    ax2 = axes[1]
    ax2.set_title('I22 Sensitivity by Structural Member', fontsize=13, fontweight='bold', pad=15)

    # Group data
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
            bar_color = color if val >= 0 else '#ff9999'
            edge_color = 'gold' if idx == 13 else 'black'
            edge_width = 3 if idx == 13 else 0.5

            ax2.bar(x_pos, val, color=bar_color, edgecolor=edge_color,
                    linewidth=edge_width, width=0.8)

            # Value label on bar
            va = 'bottom' if val >= 0 else 'top'
            offset = val * 0.05 if abs(val) > 1000 else (500 if val >= 0 else -500)
            ax2.text(x_pos, val + offset, format_value(val),
                     ha='center', va=va, fontsize=6, fontweight='bold', rotation=90)

            x_ticks.append(x_pos)
            x_labels_list.append(f'E{idx + 1}')
            x_pos += 1

        # Group separator
        ax2.axvline(x=x_pos - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        x_pos += 0.5

    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels_list, fontsize=6, rotation=45, ha='right')
    ax2.set_ylabel('I22_SENSITIVITY (dMY/dI22)', fontsize=11)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.grid(axis='y', alpha=0.2)

    # Group name labels at bottom
    x_pos_reset = 0
    for group_name, (indices, color) in groups.items():
        mid = x_pos_reset + len(indices) / 2 - 0.5
        ax2.text(mid, ax2.get_ylim()[0] * 0.85, group_name,
                 ha='center', fontsize=7, fontweight='bold',
                 bbox=dict(facecolor=color, alpha=0.3, pad=3,
                           boxstyle='round,pad=0.3'))
        x_pos_reset += len(indices) + 0.5

    # Annotation for traced element
    ax2.annotate('◄ TRACED', xy=(x_ticks[13], sensitivity[13]),
                 xytext=(x_ticks[13] + 3, sensitivity[13] * 0.7),
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
        print("Plotting primal results...")
        plot_primal(primal, save_name='primal_results.png')
    else:
        print(f"Primal file not found: {primal_file}")

    # ---- ADJOINT ----
    adjoint_file = "vtk_output_adjoint/Structure_0_1.vtk"
    if os.path.exists(adjoint_file):
        adjoint = read_vtk_file(adjoint_file)
        print("Plotting adjoint results...")
        plot_adjoint(adjoint, save_name='adjoint_sensitivity_results.png')
    else:
        print(f"Adjoint file not found: {adjoint_file}")

    print("\nDone! All plots generated.")