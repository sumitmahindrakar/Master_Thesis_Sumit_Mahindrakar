"""
Adjoint VTK Plotter for 6-Floor Frame
Plots I22_SENSITIVITY and ADJOINT_DISPLACEMENT from adjoint VTK file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
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
                data['cells'].append([int(v) for v in values[1:n_nodes+1]])
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


def get_element_groups():
    """Define element groups for 30-element frame."""
    return {
        'Left Column':   list(range(0, 6)),
        'Right Column':  list(range(6, 12)),
        'Floor 1 Beams': list(range(12, 15)),
        'Floor 2 Beams': list(range(15, 18)),
        'Floor 3 Beams': list(range(18, 21)),
        'Floor 4 Beams': list(range(21, 24)),
        'Floor 5 Beams': list(range(24, 27)),
        'Floor 6 Beams': list(range(27, 30)),
    }


def plot_cell_data(data, field_name, component=None, title=None, save_name=None,
                   figsize=(12, 16), label_fontsize=6, show_values=True):
    """Plot frame with cell data color-mapped on elements."""
    
    points = data['points']
    cells = data['cells']
    field = data['cell_data'][field_name]
    
    # Handle vector vs scalar
    if len(field.shape) > 1 and component is not None:
        values = field[:, component]
        comp_label = ['X', 'Y', 'Z'][component]
    elif len(field.shape) > 1:
        values = np.linalg.norm(field, axis=1)
        comp_label = 'magnitude'
    else:
        values = field
        comp_label = ''
    
    # Create line segments in X-Z plane
    segments = []
    for cell in cells:
        p1, p2 = points[cell[0]], points[cell[1]]
        segments.append([[p1[0], p1[2]], [p2[0], p2[2]]])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colormap
    vmax = max(abs(values.max()), abs(values.min()))
    if vmax == 0:
        vmax = 1
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    lc = LineCollection(segments, cmap=cm.RdBu_r, norm=norm, linewidths=5)
    lc.set_array(values)
    ax.add_collection(lc)
    
    # Colorbar
    cbar = plt.colorbar(lc, ax=ax, shrink=0.6, pad=0.02)
    label = f'{field_name} ({comp_label})' if comp_label else field_name
    cbar.set_label(label, fontsize=12)
    
    # Element labels
    if show_values:
        for idx, cell in enumerate(cells):
            p1, p2 = points[cell[0]], points[cell[1]]
            mid_x = (p1[0] + p2[0]) / 2
            mid_z = (p1[2] + p2[2]) / 2
            is_column = abs(p1[0] - p2[0]) < 0.01
            
            if is_column:
                offset_x = -0.6 if mid_x < 3 else 0.6
                offset_z = 0
            else:
                offset_x = 0
                offset_z = 0.4
            
            ax.annotate(f'E{idx+1}\n{values[idx]:.1f}',
                       xy=(mid_x + offset_x, mid_z + offset_z),
                       fontsize=label_fontsize, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                alpha=0.85, edgecolor='gray'))
    
    # Nodes
    for idx, pt in enumerate(points):
        ax.plot(pt[0], pt[2], 'ko', markersize=4)
        ax.annotate(f'N{idx+1}', xy=(pt[0], pt[2]),
                   xytext=(3, 3), textcoords='offset points',
                   fontsize=6, color='blue')
    
    # Supports
    for sn in [0, 1]:
        ax.plot(points[sn][0], points[sn][2], 'g^', markersize=15,
               markeredgecolor='black', markeredgewidth=1.5)
    
    # Traced element highlight
    traced_idx = 13  # Element 14 (0-indexed)
    p1, p2 = points[cells[traced_idx][0]], points[cells[traced_idx][1]]
    ax.plot([p1[0], p2[0]], [p1[2], p2[2]], 'y-', linewidth=10, alpha=0.4, 
            label=f'Traced Element {traced_idx+1}')
    ax.legend(loc='upper right', fontsize=10)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.margins(0.15)
    ax.set_title(title or f'{field_name} Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


def plot_point_data(data, field_name, component=None, title=None, save_name=None,
                    figsize=(12, 16)):
    """Plot frame with point data on nodes."""
    
    points = data['points']
    cells = data['cells']
    field = data['point_data'][field_name]
    
    if len(field.shape) > 1 and component is not None:
        values = field[:, component]
        comp_label = ['X', 'Y', 'Z'][component]
    elif len(field.shape) > 1:
        values = np.linalg.norm(field, axis=1)
        comp_label = 'magnitude'
    else:
        values = field
        comp_label = ''
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Frame lines
    for cell in cells:
        p1, p2 = points[cell[0]], points[cell[1]]
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], 'k-', linewidth=1.5, alpha=0.3)
    
    # Color nodes
    vmax = max(abs(values.max()), abs(values.min()))
    if vmax == 0:
        vmax = 1
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    scatter = ax.scatter(points[:, 0], points[:, 2], c=values,
                        cmap=cm.RdBu_r, norm=norm, s=120,
                        edgecolors='black', zorder=5)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    label = f'{field_name} ({comp_label})' if comp_label else field_name
    cbar.set_label(label, fontsize=12)
    
    # Node labels
    for idx, pt in enumerate(points):
        ax.annotate(f'N{idx+1}\n{values[idx]:.4e}',
                   xy=(pt[0], pt[2]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=5, ha='left',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                            alpha=0.85, edgecolor='gray'))
    
    # Supports
    for sn in [0, 1]:
        ax.plot(points[sn][0], points[sn][2], 'g^', markersize=15,
               markeredgecolor='black', markeredgewidth=1.5, zorder=6)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.margins(0.15)
    ax.set_title(title or f'{field_name} Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


def print_sensitivity_table(data):
    """Print I22_SENSITIVITY in organized table."""
    
    values = data['cell_data']['I22_SENSITIVITY']
    groups = get_element_groups()
    
    print("\n" + "=" * 60)
    print("  I22_SENSITIVITY (dMY/dI22) — Traced Element: 14")
    print("=" * 60)
    
    for group_name, indices in groups.items():
        print(f"\n  --- {group_name} ---")
        for idx in indices:
            marker = " ◄◄◄ TRACED" if idx == 13 else ""
            print(f"    Element {idx+1:>3}: {values[idx]:>14.2f}{marker}")
    
    print(f"\n  Max sensitivity:  Element {np.argmax(np.abs(values))+1} = {values[np.argmax(np.abs(values))]:.2f}")
    print(f"  Min sensitivity:  Element {np.argmin(values)+1} = {values[np.argmin(values)]:.2f}")


def plot_sensitivity_bar_chart(data, save_name=None):
    """Bar chart of I22_SENSITIVITY grouped by structural member."""
    
    values = data['cell_data']['I22_SENSITIVITY']
    groups = get_element_groups()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # --- Plot 1: All elements ---
    ax = axes[0]
    colors = ['red' if v < 0 else 'steelblue' for v in values]
    elem_ids = np.arange(1, 31)
    ax.bar(elem_ids, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvspan(13.5, 14.5, alpha=0.3, color='yellow', label='Traced Element 14')
    ax.set_xlabel('Element ID', fontsize=11)
    ax.set_ylabel('I22_SENSITIVITY', fontsize=11)
    ax.set_title('I22_SENSITIVITY (dMY/dI22) — All Elements', fontsize=13, fontweight='bold')
    ax.set_xticks(elem_ids)
    ax.set_xticklabels(elem_ids, fontsize=7)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # --- Plot 2: Grouped by structural member ---
    ax2 = axes[1]
    group_names = []
    group_values = []
    group_colors_map = {
        'Left Column': 'steelblue',
        'Right Column': 'cornflowerblue',
        'Floor 1 Beams': 'orangered',
        'Floor 2 Beams': 'darkorange',
        'Floor 3 Beams': 'gold',
        'Floor 4 Beams': 'yellowgreen',
        'Floor 5 Beams': 'mediumseagreen',
        'Floor 6 Beams': 'mediumpurple',
    }
    
    x_pos = 0
    x_positions = []
    x_labels = []
    bar_colors = []
    all_values = []
    separators = []
    
    for group_name, indices in groups.items():
        separators.append(x_pos - 0.5)
        for idx in indices:
            x_positions.append(x_pos)
            x_labels.append(f'E{idx+1}')
            all_values.append(values[idx])
            bar_colors.append(group_colors_map[group_name])
            x_pos += 1
        x_pos += 0.5
    
    ax2.bar(x_positions, all_values, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    
    # Group labels
    for group_name, indices in groups.items():
        start = x_positions[indices[0]] if indices[0] < len(x_positions) else 0
        end = x_positions[min(indices[-1], len(x_positions)-1)]
        mid = (start + end) / 2
        ax2.text(mid, ax2.get_ylim()[1] * 0.95, group_name,
                ha='center', fontsize=7, fontweight='bold',
                bbox=dict(facecolor=group_colors_map[group_name], alpha=0.3, pad=2))
    
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels, fontsize=6, rotation=45)
    ax2.set_ylabel('I22_SENSITIVITY', fontsize=11)
    ax2.set_title('I22_SENSITIVITY Grouped by Structural Member', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.show()


# ============================================================
#  MAIN
# ============================================================
if __name__ == '__main__':
    
    os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\SA_Kratos_adj_V3")
    
    adjoint_file = "vtk_output_adjoint/Structure_0_1.vtk"
    
    if not os.path.exists(adjoint_file):
        print(f"File not found: {adjoint_file}")
        exit()
    
    # Read data
    adjoint = read_vtk_file(adjoint_file)
    
    # Print available data
    print("\n" + "=" * 50)
    print("AVAILABLE DATA")
    print("=" * 50)
    for name, arr in adjoint['point_data'].items():
        print(f"  Point: {name} {arr.shape}")
    for name, arr in adjoint['cell_data'].items():
        print(f"  Cell:  {name} {arr.shape}")
    
    # ---- 1. I22_SENSITIVITY Table ----
    print_sensitivity_table(adjoint)
    
    # ---- 2. I22_SENSITIVITY on Frame ----
    plot_cell_data(adjoint, 'I22_SENSITIVITY',
                   title='I22 Sensitivity (dMY/dI22)\nTraced: Element 14 (Floor 1 Middle Beam)',
                   save_name='adjoint_I22_sensitivity_frame.png')
    
    # ---- 3. I22_SENSITIVITY Bar Chart ----
    plot_sensitivity_bar_chart(adjoint,
                              save_name='adjoint_I22_sensitivity_bars.png')
    
    # ---- 4. Adjoint Displacement X ----
    plot_point_data(adjoint, 'ADJOINT_DISPLACEMENT', component=0,
                    title='Adjoint Displacement X',
                    save_name='adjoint_disp_X.png')
    
    # ---- 5. Adjoint Displacement Z ----
    plot_point_data(adjoint, 'ADJOINT_DISPLACEMENT', component=2,
                    title='Adjoint Displacement Z',
                    save_name='adjoint_disp_Z.png')
    
    # ---- 6. Adjoint Rotation Y ----
    plot_point_data(adjoint, 'ADJOINT_ROTATION', component=1,
                    title='Adjoint Rotation Y',
                    save_name='adjoint_rotation_Y.png')
    
    print("\nAll plots generated!")