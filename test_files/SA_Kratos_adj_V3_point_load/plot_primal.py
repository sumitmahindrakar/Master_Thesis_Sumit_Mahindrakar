"""
VTK Frame Structure Plotter
Reads VTK files and plots frame geometry with color-mapped data.
Works for any VTK output (primal, adjoint, sensitivity).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colorbar import ColorbarBase
import matplotlib.colors as mcolors
import sys
import os


def read_vtk_file(filename):
    """Read VTK ASCII file and extract all data."""
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = {
        'points': [],
        'cells': [],
        'point_data': {},
        'cell_data': {}
    }
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Read points
        if line.startswith('POINTS'):
            parts = line.split()
            n_points = int(parts[1])
            i += 1
            coords = []
            while len(coords) < n_points * 3:
                values = lines[i].strip().split()
                coords.extend([float(v) for v in values])
                i += 1
            data['points'] = np.array(coords).reshape(n_points, 3)
            continue
        
        # Read cells
        elif line.startswith('CELLS'):
            parts = line.split()
            n_cells = int(parts[1])
            i += 1
            for c in range(n_cells):
                values = lines[i].strip().split()
                n_nodes = int(values[0])
                node_ids = [int(v) for v in values[1:n_nodes+1]]
                data['cells'].append(node_ids)
                i += 1
            continue
        
        # Read point data
        elif line.startswith('POINT_DATA'):
            n_points = int(line.split()[1])
            i += 1
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('CELL_DATA'):
                    break
                if line.startswith('FIELD'):
                    n_fields = int(line.split()[-1])
                    i += 1
                    for f_idx in range(n_fields):
                        field_line = lines[i].strip().split()
                        field_name = field_line[0]
                        n_components = int(field_line[1])
                        n_tuples = int(field_line[2])
                        i += 1
                        values = []
                        while len(values) < n_tuples * n_components:
                            vals = lines[i].strip().split()
                            values.extend([float(v) for v in vals])
                            i += 1
                        if n_components > 1:
                            data['point_data'][field_name] = np.array(values).reshape(n_tuples, n_components)
                        else:
                            data['point_data'][field_name] = np.array(values)
                    continue
                i += 1
            continue
        
        # Read cell data
        elif line.startswith('CELL_DATA'):
            n_cells = int(line.split()[1])
            i += 1
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('FIELD'):
                    n_fields = int(line.split()[-1])
                    i += 1
                    for f_idx in range(n_fields):
                        field_line = lines[i].strip().split()
                        field_name = field_line[0]
                        n_components = int(field_line[1])
                        n_tuples = int(field_line[2])
                        i += 1
                        values = []
                        while len(values) < n_tuples * n_components:
                            vals = lines[i].strip().split()
                            values.extend([float(v) for v in vals])
                            i += 1
                        if n_components > 1:
                            data['cell_data'][field_name] = np.array(values).reshape(n_tuples, n_components)
                        else:
                            data['cell_data'][field_name] = np.array(values)
                    continue
                i += 1
            continue
        
        i += 1
    
    return data


def get_element_info(n_cells):
    """Define element groups for the frame."""
    info = {}
    if n_cells == 30:
        info['Left Column'] = list(range(0, 6))
        info['Right Column'] = list(range(6, 12))
        info['Floor 1 Beams'] = list(range(12, 15))
        info['Floor 2 Beams'] = list(range(15, 18))
        info['Floor 3 Beams'] = list(range(18, 21))
        info['Floor 4 Beams'] = list(range(21, 24))
        info['Floor 5 Beams'] = list(range(24, 27))
        info['Floor 6 Beams'] = list(range(27, 30))
    elif n_cells == 10:
        info['Beam Elements'] = list(range(0, 10))
    else:
        info['All Elements'] = list(range(0, n_cells))
    return info


def plot_frame_with_cell_data(data, field_name, component=None, title=None, save_name=None):
    """
    Plot frame with cell data color-mapped on elements.
    
    Parameters:
        data: dict from read_vtk_file
        field_name: str, name of cell data field (e.g., 'MOMENT', 'I22_SENSITIVITY')
        component: int or None, component index for vector data (0,1,2)
        title: str, plot title
        save_name: str, filename to save
    """
    points = data['points']
    cells = data['cells']
    cell_field = data['cell_data'][field_name]
    
    # Extract component if vector
    if len(cell_field.shape) > 1 and component is not None:
        values = cell_field[:, component]
        comp_labels = ['X', 'Y', 'Z']
        comp_label = comp_labels[component]
    elif len(cell_field.shape) > 1:
        values = np.linalg.norm(cell_field, axis=1)
        comp_label = 'magnitude'
    else:
        values = cell_field
        comp_label = ''
    
    # Create line segments
    segments = []
    for cell in cells:
        p1 = points[cell[0]]
        p2 = points[cell[1]]
        segments.append([[p1[0], p1[2]], [p2[0], p2[2]]])  # X-Z plane
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    
    # Symmetric colormap for values with positive and negative
    vmax = max(abs(values.max()), abs(values.min()))
    if vmax == 0:
        vmax = 1
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = cm.RdBu_r
    
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=4)
    lc.set_array(values)
    ax.add_collection(lc)
    
    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax, shrink=0.6, pad=0.02)
    if comp_label:
        cbar.set_label(f'{field_name} ({comp_label})', fontsize=12)
    else:
        cbar.set_label(field_name, fontsize=12)
    
    # Add element labels with values
    for idx, cell in enumerate(cells):
        p1 = points[cell[0]]
        p2 = points[cell[1]]
        mid_x = (p1[0] + p2[0]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        
        # Offset label slightly
        is_column = abs(p1[0] - p2[0]) < 0.01
        if is_column:
            offset_x = -0.4 if mid_x < 3 else 0.4
            offset_z = 0
        else:
            offset_x = 0
            offset_z = 0.3
        
        ax.annotate(f'E{idx+1}\n{values[idx]:.2f}',
                    xy=(mid_x + offset_x, mid_z + offset_z),
                    fontsize=6, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                             alpha=0.8, edgecolor='gray'))
    
    # Add node labels
    for idx, point in enumerate(points):
        ax.plot(point[0], point[2], 'ko', markersize=5)
        ax.annotate(f'N{idx+1}', xy=(point[0], point[2]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, color='blue')
    
    # Add support symbols
    support_nodes = [0, 1]  # Ground nodes (0-indexed)
    for sn in support_nodes:
        if sn < len(points):
            ax.plot(points[sn][0], points[sn][2], 'g^', markersize=15, 
                    markeredgecolor='black', markeredgewidth=1.5)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.margins(0.15)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'{field_name} Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    
    plt.show()


def plot_frame_with_point_data(data, field_name, component=None, title=None, save_name=None):
    """
    Plot frame with point data color-mapped on nodes.
    
    Parameters:
        data: dict from read_vtk_file
        field_name: str, name of point data field (e.g., 'DISPLACEMENT')
        component: int or None, component index for vector data (0,1,2)
        title: str, plot title
        save_name: str, filename to save
    """
    points = data['points']
    cells = data['cells']
    point_field = data['point_data'][field_name]
    
    # Extract component if vector
    if len(point_field.shape) > 1 and component is not None:
        values = point_field[:, component]
        comp_labels = ['X', 'Y', 'Z']
        comp_label = comp_labels[component]
    elif len(point_field.shape) > 1:
        values = np.linalg.norm(point_field, axis=1)
        comp_label = 'magnitude'
    else:
        values = point_field
        comp_label = ''
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    
    # Draw frame lines in gray
    for cell in cells:
        p1 = points[cell[0]]
        p2 = points[cell[1]]
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], 'k-', linewidth=1.5, alpha=0.3)
    
    # Color nodes
    vmax = max(abs(values.max()), abs(values.min()))
    if vmax == 0:
        vmax = 1
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    scatter = ax.scatter(points[:, 0], points[:, 2], c=values,
                        cmap=cm.RdBu_r, norm=norm, s=100, 
                        edgecolors='black', zorder=5)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    if comp_label:
        cbar.set_label(f'{field_name} ({comp_label})', fontsize=12)
    else:
        cbar.set_label(field_name, fontsize=12)
    
    # Add node labels with values
    for idx, point in enumerate(points):
        ax.annotate(f'N{idx+1}\n{values[idx]:.4e}',
                    xy=(point[0], point[2]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=6, ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                             alpha=0.8, edgecolor='gray'))
    
    # Support symbols
    for sn in [0, 1]:
        if sn < len(points):
            ax.plot(points[sn][0], points[sn][2], 'g^', markersize=15,
                    markeredgecolor='black', markeredgewidth=1.5, zorder=6)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.margins(0.15)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'{field_name} Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
    
    plt.show()


def print_available_data(data):
    """Print all available fields in the VTK file."""
    print("\n" + "="*50)
    print("AVAILABLE DATA IN VTK FILE")
    print("="*50)
    print(f"\nPoints: {len(data['points'])} nodes")
    print(f"Cells: {len(data['cells'])} elements")
    
    print("\nPoint Data (nodal):")
    for name, arr in data['point_data'].items():
        shape = arr.shape
        print(f"  {name}: shape={shape}")
    
    print("\nCell Data (element):")
    for name, arr in data['cell_data'].items():
        shape = arr.shape
        print(f"  {name}: shape={shape}")
    print("="*50)


def print_data_table(data, field_name, data_type='cell', component=None):
    """Print data in tabular format."""
    if data_type == 'cell':
        field = data['cell_data'][field_name]
        elem_info = get_element_info(len(data['cells']))
    else:
        field = data['point_data'][field_name]
    
    print(f"\n{'='*60}")
    print(f"  {field_name} Values")
    print(f"{'='*60}")
    
    if data_type == 'cell':
        if len(field.shape) > 1:
            print(f"  {'Elem':>6}  {'X':>12}  {'Y':>12}  {'Z':>12}")
            print(f"  {'-'*48}")
            for group_name, indices in elem_info.items():
                print(f"\n  --- {group_name} ---")
                for idx in indices:
                    print(f"  {idx+1:>6}  {field[idx,0]:>12.4f}  {field[idx,1]:>12.4f}  {field[idx,2]:>12.4f}")
        else:
            print(f"  {'Elem':>6}  {'Value':>12}")
            print(f"  {'-'*20}")
            for group_name, indices in elem_info.items():
                print(f"\n  --- {group_name} ---")
                for idx in indices:
                    print(f"  {idx+1:>6}  {field[idx]:>12.4f}")
    else:
        if len(field.shape) > 1 and component is not None:
            comp_labels = ['X', 'Y', 'Z']
            print(f"  Component: {comp_labels[component]}")
            print(f"  {'Node':>6}  {'Value':>12}")
            print(f"  {'-'*20}")
            for idx in range(len(field)):
                print(f"  {idx+1:>6}  {field[idx, component]:>12.6e}")
        elif len(field.shape) > 1:
            print(f"  {'Node':>6}  {'X':>14}  {'Y':>14}  {'Z':>14}")
            print(f"  {'-'*50}")
            for idx in range(len(field)):
                print(f"  {idx+1:>6}  {field[idx,0]:>14.6e}  {field[idx,1]:>14.6e}  {field[idx,2]:>14.6e}")


# ============================================================
#  MAIN - Example usage
# ============================================================
if __name__ == '__main__':
    
    # Set your working directory
    os.chdir(r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\SA_Kratos_adj_V3")
    
    # ---- PRIMAL RESULTS ----
    primal_file = "vtk_output_primal/Structure_0_1.vtk"
    if os.path.exists(primal_file):
        primal = read_vtk_file(primal_file)
        print_available_data(primal)
        
        # Plot Bending Moment MY (component 1)
        print_data_table(primal, 'MOMENT', 'cell')
        plot_frame_with_cell_data(primal, 'MOMENT', component=1,
            title='Bending Moment MY Distribution (Primal)',
            save_name='Beam_plots/primal_moment_MY.png')
        
        # Plot Displacement Z (component 2)
        plot_frame_with_point_data(primal, 'DISPLACEMENT', component=2,
            title='Displacement Z (Primal)',
            save_name='Beam_plots/primal_displacement_Z.png')
        
        # Plot Reactions Z
        plot_frame_with_point_data(primal, 'REACTION', component=2,
            title='Reaction Forces Z (Primal)',
            save_name='Beam_plots/primal_reaction_Z.png')
    
    # ---- ADJOINT RESULTS ----
    adjoint_file = "vtk_output_adjoint/Structure_0_1.vtk"
    if os.path.exists(adjoint_file):
        adjoint = read_vtk_file(adjoint_file)
        print_available_data(adjoint)
        
        # Plot I22 Sensitivity
        if 'I22_SENSITIVITY' in adjoint['cell_data']:
            print_data_table(adjoint, 'I22_SENSITIVITY', 'cell')
            plot_frame_with_cell_data(adjoint, 'I22_SENSITIVITY',
                title='I22 Sensitivity (dMY/dI22)',
                save_name='Beam_plots/adjoint_I22_sensitivity.png')
        
        # Plot I33 Sensitivity
        if 'I33_SENSITIVITY' in adjoint['cell_data']:
            print_data_table(adjoint, 'I33_SENSITIVITY', 'cell')
            plot_frame_with_cell_data(adjoint, 'I33_SENSITIVITY',
                title='I33 Sensitivity (dMZ/dI33)',
                save_name='Beam_plots/adjoint_I33_sensitivity.png')
        
        # Plot Adjoint Displacement
        plot_frame_with_point_data(adjoint, 'ADJOINT_DISPLACEMENT', component=1,
            title='Adjoint Displacement Y',
            save_name='Beam_plots/adjoint_displacement_Y.png')
        
        # Plot Shape Sensitivity
        if 'SHAPE_SENSITIVITY' in adjoint['point_data']:
            plot_frame_with_point_data(adjoint, 'SHAPE_SENSITIVITY', component=0,
                title='Shape Sensitivity X',
                save_name='Beam_plots/adjoint_shape_sensitivity_X.png')
    
    print("\nDone! All plots generated.")