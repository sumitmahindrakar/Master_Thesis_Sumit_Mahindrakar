"""
Moment Sensitivity Calculator: ∂M/∂(EI)
Using General Influence Method with VTK File Parsing and Visualization

Description:
    Computes the sensitivity of bending moment at a response location
    with respect to flexural rigidity (EI) of beam elements using the
    adjoint/dual method.
    
Formula:
    ∂M_response/∂(EI)_k = -(1/(EI)²) × ∫ M_k(x) × M̄_k(x) dx
    
    Approximated as:
    ∂M_response/∂(EI)_k = -(M_k × M̄_k × L_k) / (EI)²
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch


# =============================================================================
# CONFIGURATION - Easy to modify parameters
# =============================================================================

# Scale factors for visualization
PRIMARY_MOMENT_SCALE = 0.0003     # Scale for primary moment diagram
DUAL_MOMENT_SCALE = 0.000001      # Scale for dual moment diagram  
SENSITIVITY_SCALE = 1e4         # Scale for sensitivity diagram

# Colors
COLOR_STRUCTURE = 'black'
COLOR_PRIMARY_MOMENT = 'blue'
COLOR_DUAL_MOMENT = 'green'
COLOR_SENSITIVITY = 'red'
COLOR_FILL_PRIMARY = 'lightblue'
COLOR_FILL_DUAL = 'lightgreen'
COLOR_FILL_SENSITIVITY_POS = '#90EE90'  # Light green for positive
COLOR_FILL_SENSITIVITY_NEG = '#FFB6C1'  # Light pink for negative

# Line widths
LINEWIDTH_STRUCTURE = 2.0
LINEWIDTH_DIAGRAM = 1.5

# Figure settings
FIGURE_DPI = 100
SAVE_FIGURES = True
OUTPUT_FOLDER = "test_files/SA_beam_2D_udl_kink.gid/plots"


# =============================================================================
# VTK FILE PARSER
# =============================================================================

def parse_vtk_file(filename):
    """
    Parse a VTK file and extract points, cells, and field data.
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
        
        # Parse POINTS section
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
                            float(values[j+1]), 
                            float(values[j+2])
                        ])
                i += 1
            
            data['points'] = np.array(points[:num_points])
            print(f"  - Loaded {num_points} points")
            continue
        
        # Parse CELLS section
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
        
        # Parse CELL_TYPES section
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
        
        # Track data section
        elif line.startswith('POINT_DATA'):
            current_section = 'point_data'
            i += 1
            continue
        
        elif line.startswith('CELL_DATA'):
            current_section = 'cell_data'
            i += 1
            continue
        
        elif line.startswith('FIELD'):
            i += 1
            continue
        
        # Parse field arrays
        elif current_section is not None:
            parts = line.split()
            
            if len(parts) >= 4:
                try:
                    field_name = parts[0]
                    num_components = int(parts[1])
                    num_tuples = int(parts[2])
                    
                    field_data = []
                    i += 1
                    
                    while len(field_data) < num_tuples and i < len(lines):
                        values = lines[i].strip().split()
                        
                        if values and (values[0] in ['POINT_DATA', 'CELL_DATA', 
                                                       'FIELD', 'SCALARS', 'VECTORS']):
                            break
                        
                        if len(values) >= num_components:
                            try:
                                field_data.append([float(v) for v in values[:num_components]])
                                i += 1
                            except ValueError:
                                break
                        else:
                            i += 1
                    
                    if len(field_data) == num_tuples:
                        data[current_section][field_name] = np.array(field_data)
                        print(f"  - Loaded {current_section[:-5]} field: {field_name} "
                              f"({num_tuples} tuples, {num_components} components)")
                    continue
                    
                except (ValueError, IndexError):
                    pass
        
        i += 1
    
    return data


def parse_vtk_cell_moments(vtk_file_path):
    """
    Parse bending moments (Mz) from VTK file CELL_DATA section
    """
    moments = {}
    
    if not os.path.exists(vtk_file_path):
        print(f"Error: VTK file not found: {vtk_file_path}")
        return moments
    
    with open(vtk_file_path, 'r') as f:
        lines = f.readlines()
    
    in_cell_data = False
    reading_moment = False
    elem_idx = 1
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        if line_stripped.startswith('CELL_DATA'):
            in_cell_data = True
            continue
        
        if line_stripped.startswith('POINT_DATA'):
            in_cell_data = False
            reading_moment = False
            continue
        
        if in_cell_data and 'MOMENT' in line_stripped and 'FIELD' not in line_stripped:
            reading_moment = True
            continue
        
        if reading_moment and line_stripped:
            parts = line_stripped.split()
            if len(parts) >= 3:
                try:
                    Mz = float(parts[2])
                    moments[elem_idx] = Mz
                    elem_idx += 1
                except ValueError:
                    reading_moment = False
    
    return moments


# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================

def get_element_direction(p1, p2):
    """Get the unit direction vector of an element."""
    direction = np.array(p2) - np.array(p1)
    length = np.linalg.norm(direction)
    if length > 0:
        return direction / length
    return np.array([1, 0, 0])


def get_element_length(p1, p2):
    """Get the length of an element."""
    return np.linalg.norm(np.array(p2) - np.array(p1))


def get_perpendicular_direction(p1, p2):
    """Get the perpendicular direction to an element (in 2D plane)."""
    direction = get_element_direction(p1, p2)
    perpendicular = np.array([-direction[1], direction[0], 0])
    return perpendicular


def find_support_indices(points):
    """Automatically find the left-most and right-most points for supports."""
    x_coords = points[:, 0]
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    
    tolerance = (max_x - min_x) * 0.01 if max_x > min_x else 0.01
    
    left_indices = np.where(np.abs(x_coords - min_x) < tolerance)[0].tolist()
    right_indices = np.where(np.abs(x_coords - max_x) < tolerance)[0].tolist()
    
    return left_indices, right_indices


# =============================================================================
# PLOTTING FUNCTIONS FOR STRUCTURE
# =============================================================================

def plot_structure(ax, points, cells, color=COLOR_STRUCTURE, 
                   linewidth=LINEWIDTH_STRUCTURE, label='Structure',
                   show_node_labels=False, show_element_labels=False):
    """Plot the frame structure."""
    
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
        
        # Element labels
        if show_element_labels:
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            ax.annotate(f'E{idx+1}', (mid_x, mid_y), 
                       textcoords="offset points", xytext=(0, -15),
                       fontsize=8, color='gray', ha='center')
    
    # Plot nodes
    ax.scatter(points[:, 0], points[:, 1], color=color, s=30, zorder=5)
    
    # Node labels
    if show_node_labels:
        for i, point in enumerate(points):
            ax.annotate(f'{i}', (point[0], point[1]), 
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=8, color='gray')


def add_supports_l(ax, points, support_indices=None):
    """Add left support symbols at fixed nodes."""
    if support_indices is None:
        support_indices, _ = find_support_indices(points)
    
    if isinstance(support_indices, (int, np.integer)):
        support_indices = [support_indices]
    
    size = 0.08
    
    for idx in support_indices:
        if idx >= len(points):
            continue
            
        x, y = points[idx][0], points[idx][1]
        ax.plot([x, x], [y - size, y + size], color='black', linewidth=1.5)
        for hy in np.linspace(y - size, y + size, 6):
            ax.plot([x, x - size*0.3], [hy, hy - size*0.3], 'k', lw=0.8)


def add_supports_r(ax, points, support_indices=None):
    """Add right support symbols at fixed nodes."""
    if support_indices is None:
        _, support_indices = find_support_indices(points)
    
    if isinstance(support_indices, (int, np.integer)):
        support_indices = [support_indices]
    
    size = 0.08
    
    for idx in support_indices:
        if idx >= len(points):
            continue
            
        x, y = points[idx][0], points[idx][1]
        ax.plot([x, x], [y - size, y + size], color='black', linewidth=1.5)
        for hy in np.linspace(y - size, y + size, 6):
            ax.plot([x, x + size*0.3], [hy, hy - size*0.3], 'k', lw=0.8)


# =============================================================================
# DIAGRAM PLOTTING FUNCTIONS
# =============================================================================

def plot_moment_diagram_on_structure(ax, points, cells, moment_data, 
                                      scale, color, fill_color,
                                      linewidth=LINEWIDTH_DIAGRAM, 
                                      show_fill=True,
                                      show_values=True, 
                                      use_cell_data=True,
                                      show_connecting_lines=True,
                                      value_fontsize=9):
    """
    Plot moment diagram perpendicular to each element.
    Works for beams and frame structures.
    """
    
    for idx, cell in enumerate(cells):
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = points[p1_idx][:2]
        p2 = points[p2_idx][:2]
        
        perp = get_perpendicular_direction(p1, p2)[:2]
        
        if use_cell_data:
            if idx < len(moment_data):
                moment_value = moment_data[idx][2] if hasattr(moment_data[idx], '__len__') else moment_data[idx]
            else:
                moment_value = 0
            
            offset_p1 = p1 + perp * moment_value * scale
            offset_p2 = p2 + perp * moment_value * scale
            
            if show_values and abs(moment_value) > 1:
                mid_struct = (p1 + p2) / 2
                mid_offset = (offset_p1 + offset_p2) / 2
                
                # Position text at midpoint of the diagram
                text_pos = mid_offset + perp * np.sign(moment_value) * 0.02
                
                ax.annotate(f'{moment_value:.0f}', 
                           xy=(text_pos[0], text_pos[1]),
                           fontsize=value_fontsize, color=color,
                           ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                    facecolor='white', alpha=0.8, 
                                    edgecolor='none'))
        else:
            moment1 = moment_data[p1_idx][2]
            moment2 = moment_data[p2_idx][2]
            
            offset_p1 = p1 + perp * moment1 * scale
            offset_p2 = p2 + perp * moment2 * scale
        
        # Plot moment diagram line
        ax.plot([offset_p1[0], offset_p2[0]], [offset_p1[1], offset_p2[1]], 
               color=color, linewidth=linewidth)
        
        # Connecting lines
        if show_connecting_lines:
            ax.plot([p1[0], offset_p1[0]], [p1[1], offset_p1[1]], 
                   color=color, linewidth=0.5, linestyle='-', alpha=0.7)
            ax.plot([p2[0], offset_p2[0]], [p2[1], offset_p2[1]], 
                   color=color, linewidth=0.5, linestyle='-', alpha=0.7)
        
        # Fill area
        if show_fill:
            polygon_x = [p1[0], offset_p1[0], offset_p2[0], p2[0]]
            polygon_y = [p1[1], offset_p1[1], offset_p2[1], p2[1]]
            ax.fill(polygon_x, polygon_y, color=fill_color, alpha=0.4)


def plot_sensitivity_diagram_on_structure(ax, points, cells, sensitivity_values, 
                                          scale, linewidth=LINEWIDTH_DIAGRAM, 
                                          show_fill=True, show_values=True,
                                          show_connecting_lines=True,
                                          value_fontsize=8):
    """
    Plot sensitivity diagram perpendicular to each element.
    Uses different colors for positive and negative values.
    """
    
    elem_ids = sorted(sensitivity_values.keys())
    
    for idx, cell in enumerate(cells):
        elem_id = idx + 1
        
        if elem_id not in sensitivity_values:
            continue
        
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = points[p1_idx][:2]
        p2 = points[p2_idx][:2]
        
        perp = get_perpendicular_direction(p1, p2)[:2]
        
        sens_value = sensitivity_values[elem_id]['dM_dEI']
        
        offset_p1 = p1 + perp * sens_value * scale
        offset_p2 = p2 + perp * sens_value * scale
        
        # Choose color based on sign
        if sens_value >= 0:
            color = '#228B22'  # Forest green for positive
            fill_color = COLOR_FILL_SENSITIVITY_POS
        else:
            color = '#DC143C'  # Crimson for negative
            fill_color = COLOR_FILL_SENSITIVITY_NEG
        
        # Plot sensitivity diagram line
        ax.plot([offset_p1[0], offset_p2[0]], [offset_p1[1], offset_p2[1]], 
               color=color, linewidth=linewidth)
        
        # Connecting lines
        if show_connecting_lines:
            ax.plot([p1[0], offset_p1[0]], [p1[1], offset_p1[1]], 
                   color=color, linewidth=0.5, linestyle='-', alpha=0.7)
            ax.plot([p2[0], offset_p2[0]], [p2[1], offset_p2[1]], 
                   color=color, linewidth=0.5, linestyle='-', alpha=0.7)
        
        # Fill area
        if show_fill:
            polygon_x = [p1[0], offset_p1[0], offset_p2[0], p2[0]]
            polygon_y = [p1[1], offset_p1[1], offset_p2[1], p2[1]]
            ax.fill(polygon_x, polygon_y, color=fill_color, alpha=0.4)
        
        # Show values (selectively for readability)
        if show_values and (len(cells) <= 10 or idx % (len(cells) // 8 + 1) == 0):
            mid_struct = (p1 + p2) / 2
            mid_offset = (offset_p1 + offset_p2) / 2
            text_pos = mid_offset + perp * np.sign(sens_value) * 0.015
            
            ax.annotate(f'{sens_value:.2e}', 
                       xy=(text_pos[0], text_pos[1]),
                       fontsize=value_fontsize, color=color,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.15', 
                                facecolor='white', alpha=0.8, 
                                edgecolor='none'),
                       rotation=0)


def plot_product_diagram_on_structure(ax, points, cells, M_primary, M_dual,
                                       scale, linewidth=LINEWIDTH_DIAGRAM, 
                                       show_fill=True, show_values=False,
                                       show_connecting_lines=True):
    """
    Plot M × M̄ product diagram (integrand of sensitivity).
    """
    
    for idx, cell in enumerate(cells):
        elem_id = idx + 1
        
        if elem_id not in M_primary or elem_id not in M_dual:
            continue
        
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = points[p1_idx][:2]
        p2 = points[p2_idx][:2]
        
        perp = get_perpendicular_direction(p1, p2)[:2]
        
        # Product M × M̄
        product = M_primary[elem_id] * M_dual[elem_id]
        
        offset_p1 = p1 + perp * product * scale
        offset_p2 = p2 + perp * product * scale
        
        # Choose color based on sign
        if product >= 0:
            color = '#FF8C00'  # Dark orange for positive
            fill_color = '#FFDAB9'  # Peach
        else:
            color = '#8B008B'  # Dark magenta for negative
            fill_color = '#DDA0DD'  # Plum
        
        ax.plot([offset_p1[0], offset_p2[0]], [offset_p1[1], offset_p2[1]], 
               color=color, linewidth=linewidth)
        
        if show_connecting_lines:
            ax.plot([p1[0], offset_p1[0]], [p1[1], offset_p1[1]], 
                   color=color, linewidth=0.5, linestyle='-', alpha=0.7)
            ax.plot([p2[0], offset_p2[0]], [p2[1], offset_p2[1]], 
                   color=color, linewidth=0.5, linestyle='-', alpha=0.7)
        
        if show_fill:
            polygon_x = [p1[0], offset_p1[0], offset_p2[0], p2[0]]
            polygon_y = [p1[1], offset_p1[1], offset_p2[1], p2[1]]
            ax.fill(polygon_x, polygon_y, color=fill_color, alpha=0.4)


# =============================================================================
# SENSITIVITY COMPUTATION
# =============================================================================

def compute_moment_sensitivity(E, I, L_elements, M_primary, M_dual):
    """
    Compute ∂M/∂(EI) for all elements using the adjoint method.
    
    Formula: ∂M_response/∂(EI)_k = -(M_k · M̄_k · L_k) / (EI)²
    """
    EI = E * I
    EI_squared = EI ** 2
    
    sensitivities = {}
    total_sensitivity = 0.0
    
    common_elem_ids = set(M_primary.keys()) & set(M_dual.keys())
    
    for eid in sorted(common_elem_ids):
        M_p = M_primary[eid]
        M_d = M_dual.get(eid, 0.0)
        L = L_elements.get(eid, 1.0)
        
        integral = M_p * M_d * L
        dM_dEI = -integral / EI_squared
        
        sensitivities[eid] = {
            'M_primary': M_p,
            'M_dual': M_d,
            'length': L,
            'integral': integral,
            'dM_dEI': dM_dEI
        }
        
        total_sensitivity += dM_dEI
    
    return sensitivities, total_sensitivity


def print_results(E, I, sensitivities, total_sensitivity, response_element=None):
    """Print formatted sensitivity results."""
    EI = E * I
    
    print("\n" + "=" * 75)
    print("MOMENT SENSITIVITY ANALYSIS: ∂M/∂(EI)")
    print("Using General Influence (Adjoint) Method")
    print("=" * 75)
    
    print(f"\nMaterial/Section Properties:")
    print(f"  E  = {E:.4e} Pa")
    print(f"  I  = {I:.4e} m⁴")
    print(f"  EI = {EI:.4e} N·m²")
    
    if response_element:
        print(f"\nResponse: Bending moment in Element {response_element}")
    
    print("\n" + "-" * 75)
    print(f"{'Elem k':^8} {'M_k [N·m]':^14} {'M̄_k [N·m]':^14} "
          f"{'L_k [m]':^10} {'∂M/∂(EI)_k':^20}")
    print("-" * 75)
    
    for eid, data in sorted(sensitivities.items()):
        print(f"{eid:^8} {data['M_primary']:^+14.4f} {data['M_dual']:^+14.6f} "
              f"{data['length']:^10.4f} {data['dM_dEI']:^+20.6e}")
    
    print("-" * 75)
    print(f"{'TOTAL':^8} {' ':^14} {' ':^14} "
          f"{' ':^10} {total_sensitivity:^+20.6e}")
    print("-" * 75)
    
    delta_EI_percent = 10.0
    delta_EI = (delta_EI_percent / 100.0) * EI
    delta_M = total_sensitivity * delta_EI
    
    print(f"\nLinear Approximation:")
    print(f"  If EI increases by {delta_EI_percent}%: ΔM ≈ {delta_M:+.6f} N·m")


# =============================================================================
# COMPREHENSIVE VISUALIZATION
# =============================================================================

def create_sensitivity_visualization(vtk_primary, vtk_dual, E, I, 
                                     sensitivities, total_sensitivity,
                                     M_primary_dict, M_dual_dict,
                                     save_figures=True, output_folder="."):
    """
    Create comprehensive sensitivity visualization with diagrams plotted over the structure.
    """
    
    points = vtk_primary['points']
    cells = vtk_primary['cells']
    
    # Get support locations
    left_supports, right_supports = find_support_indices(points)
    
    # Get moment data as arrays
    cell_moment_primary = vtk_primary['cell_data'].get('MOMENT', np.zeros((len(cells), 3)))
    cell_moment_dual = vtk_dual['cell_data'].get('MOMENT', np.zeros((len(cells), 3)))
    
    # Calculate plot limits
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    x_margin = (x_max - x_min) * 0.15 if x_max > x_min else 0.5
    y_margin = max((y_max - y_min) * 0.5, 0.3)
    
    EI = E * I
    
    # Create output folder
    if save_figures and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"\nCreated output folder: {output_folder}")
    
    # =========================================================================
    # PLOT 1: Combined Overview (4 subplots)
    # =========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=FIGURE_DPI)
    fig.suptitle('Moment Sensitivity Analysis: ∂M/∂(EI)\nUsing Adjoint Method', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # --- Subplot 1: Structure with Primary Moment ---
    ax1 = axes[0, 0]
    ax1.set_title(f'Primary Bending Moment M(x)\n[From Applied Load] (Scale: {PRIMARY_MOMENT_SCALE})', 
                  fontsize=11, fontweight='bold')
    plot_structure(ax1, points, cells, color='gray', linewidth=1.5, 
                  label='Structure', show_element_labels=True)
    plot_moment_diagram_on_structure(ax1, points, cells, cell_moment_primary,
                                     scale=PRIMARY_MOMENT_SCALE,
                                     color=COLOR_PRIMARY_MOMENT,
                                     fill_color=COLOR_FILL_PRIMARY,
                                     show_values=True, use_cell_data=True)
    add_supports_l(ax1, points, support_indices=left_supports)
    add_supports_r(ax1, points, support_indices=right_supports)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_xlim(x_min - x_margin, x_max + x_margin)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Legend
    struct_line = plt.Line2D([0], [0], color='gray', linewidth=1.5)
    moment_line = plt.Line2D([0], [0], color=COLOR_PRIMARY_MOMENT, linewidth=1.5)
    ax1.legend([struct_line, moment_line], ['Structure', 'Primary Moment M'], loc='best')
    
    # --- Subplot 2: Structure with Dual Moment ---
    ax2 = axes[0, 1]
    ax2.set_title(f'Dual/Adjoint Moment M̄(x)\n[From Unit Virtual Load] (Scale: {DUAL_MOMENT_SCALE})', 
                  fontsize=11, fontweight='bold')
    plot_structure(ax2, points, cells, color='gray', linewidth=1.5, label='Structure')
    plot_moment_diagram_on_structure(ax2, points, cells, cell_moment_dual,
                                     scale=DUAL_MOMENT_SCALE,
                                     color=COLOR_DUAL_MOMENT,
                                     fill_color=COLOR_FILL_DUAL,
                                     show_values=True, use_cell_data=True,
                                     value_fontsize=8)
    add_supports_l(ax2, points, support_indices=left_supports)
    add_supports_r(ax2, points, support_indices=right_supports)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_xlim(x_min - x_margin, x_max + x_margin)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    dual_line = plt.Line2D([0], [0], color=COLOR_DUAL_MOMENT, linewidth=1.5)
    ax2.legend([struct_line, dual_line], ['Structure', 'Dual Moment M̄'], loc='best')
    
    # --- Subplot 3: Sensitivity Diagram ---
    ax3 = axes[1, 0]
    ax3.set_title(f'Sensitivity ∂M/∂(EI)\n(Scale: {SENSITIVITY_SCALE:.0e})', 
                  fontsize=11, fontweight='bold')
    plot_structure(ax3, points, cells, color='gray', linewidth=1.5, label='Structure')
    plot_sensitivity_diagram_on_structure(ax3, points, cells, sensitivities,
                                          scale=SENSITIVITY_SCALE,
                                          show_values=True, value_fontsize=7)
    add_supports_l(ax3, points, support_indices=left_supports)
    add_supports_r(ax3, points, support_indices=right_supports)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_xlim(x_min - x_margin, x_max + x_margin)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Custom legend for sensitivity
    pos_patch = mpatches.Patch(color=COLOR_FILL_SENSITIVITY_POS, alpha=0.4, 
                               label='Positive (↑ stiffness → ↑ moment)')
    neg_patch = mpatches.Patch(color=COLOR_FILL_SENSITIVITY_NEG, alpha=0.4, 
                               label='Negative (↑ stiffness → ↓ moment)')
    ax3.legend(handles=[struct_line, pos_patch, neg_patch], loc='best', fontsize=8)
    
    # --- Subplot 4: Summary Bar Chart ---
    ax4 = axes[1, 1]
    ax4.set_title('Sensitivity by Element', fontsize=11, fontweight='bold')
    
    elem_ids = sorted(sensitivities.keys())
    sens_values = [sensitivities[eid]['dM_dEI'] for eid in elem_ids]
    
    x_pos = np.arange(len(elem_ids))
    colors = ['#228B22' if v >= 0 else '#DC143C' for v in sens_values]
    
    bars = ax4.bar(x_pos, sens_values, color=colors, edgecolor='black', alpha=0.7)
    
    # Add total bar
    ax4.bar(len(elem_ids) + 0.5, total_sensitivity, color='#2C3E50', 
           edgecolor='black', alpha=0.9, width=0.8)
    ax4.annotate(f'{total_sensitivity:.2e}', 
                xy=(len(elem_ids) + 0.5, total_sensitivity),
                xytext=(0, 5 if total_sensitivity >= 0 else -15),
                textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold')
    
    ax4.axhline(y=0, color='black', linewidth=1)
    
    # X-axis labels
    if len(elem_ids) <= 20:
        ax4.set_xticks(list(x_pos) + [len(elem_ids) + 0.5])
        ax4.set_xticklabels([f'E{eid}' for eid in elem_ids] + ['TOTAL'], 
                           rotation=45, fontsize=8)
    else:
        step = len(elem_ids) // 10 + 1
        tick_pos = list(x_pos[::step]) + [len(elem_ids) + 0.5]
        tick_labels = [f'E{elem_ids[i]}' for i in range(0, len(elem_ids), step)] + ['TOTAL']
        ax4.set_xticks(tick_pos)
        ax4.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    
    ax4.set_ylabel('∂M/∂(EI)')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # Info box
    info_text = (f'E = {E:.2e} Pa\n'
                f'I = {I:.2e} m⁴\n'
                f'EI = {EI:.2e} N·m²\n'
                f'Total ∂M/∂(EI) = {total_sensitivity:.4e}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_figures:
        filepath = os.path.join(output_folder, 'sensitivity_combined.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    
    # =========================================================================
    # PLOT 2: Detailed Primary Moment Diagram
    # =========================================================================
    
    fig2, ax = plt.subplots(figsize=(14, 6), dpi=FIGURE_DPI)
    ax.set_title('Primary Bending Moment Diagram M(x)\n[From Applied Loading]', 
                 fontsize=14, fontweight='bold')
    plot_structure(ax, points, cells, color='gray', linewidth=1.5, 
                  show_element_labels=True)
    plot_moment_diagram_on_structure(ax, points, cells, cell_moment_primary,
                                     scale=PRIMARY_MOMENT_SCALE,
                                     color=COLOR_PRIMARY_MOMENT,
                                     fill_color=COLOR_FILL_PRIMARY,
                                     show_values=True, use_cell_data=True)
    add_supports_l(ax, points, support_indices=left_supports)
    add_supports_r(ax, points, support_indices=right_supports)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Info box
    max_M = np.max(cell_moment_primary[:, 2])
    min_M = np.min(cell_moment_primary[:, 2])
    info_text = f'Max M: {max_M:.1f} N·m\nMin M: {min_M:.1f} N·m\nScale: {PRIMARY_MOMENT_SCALE}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if save_figures:
        filepath = os.path.join(output_folder, 'primary_moment_diagram.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    
    # =========================================================================
    # PLOT 3: Detailed Dual Moment Diagram
    # =========================================================================
    
    fig3, ax = plt.subplots(figsize=(14, 6), dpi=FIGURE_DPI)
    ax.set_title('Dual/Adjoint Bending Moment Diagram M̄(x)\n[From Unit Virtual Load at Response Location]', 
                 fontsize=14, fontweight='bold')
    plot_structure(ax, points, cells, color='gray', linewidth=1.5,
                  show_element_labels=True)
    plot_moment_diagram_on_structure(ax, points, cells, cell_moment_dual,
                                     scale=DUAL_MOMENT_SCALE,
                                     color=COLOR_DUAL_MOMENT,
                                     fill_color=COLOR_FILL_DUAL,
                                     show_values=True, use_cell_data=True)
    add_supports_l(ax, points, support_indices=left_supports)
    add_supports_r(ax, points, support_indices=right_supports)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Info box
    max_M_dual = np.max(cell_moment_dual[:, 2])
    min_M_dual = np.min(cell_moment_dual[:, 2])
    info_text = f'Max M̄: {max_M_dual:.1f} N·m\nMin M̄: {min_M_dual:.1f} N·m\nScale: {DUAL_MOMENT_SCALE}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if save_figures:
        filepath = os.path.join(output_folder, 'dual_moment_diagram.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    
    # =========================================================================
    # PLOT 4: Detailed Sensitivity Diagram
    # =========================================================================
    
    fig4, ax = plt.subplots(figsize=(14, 6), dpi=FIGURE_DPI)
    ax.set_title('Sensitivity Diagram: ∂M/∂(EI)\n[How Response Moment Changes with Element Stiffness]', 
                 fontsize=14, fontweight='bold')
    plot_structure(ax, points, cells, color='gray', linewidth=1.5,
                  show_element_labels=True)
    plot_sensitivity_diagram_on_structure(ax, points, cells, sensitivities,
                                          scale=SENSITIVITY_SCALE,
                                          show_values=True)
    add_supports_l(ax, points, support_indices=left_supports)
    add_supports_r(ax, points, support_indices=right_supports)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Custom legend
    pos_patch = mpatches.Patch(color=COLOR_FILL_SENSITIVITY_POS, alpha=0.4, 
                               label='Positive: ↑EI → ↑M')
    neg_patch = mpatches.Patch(color=COLOR_FILL_SENSITIVITY_NEG, alpha=0.4, 
                               label='Negative: ↑EI → ↓M')
    ax.legend(handles=[pos_patch, neg_patch], loc='best', fontsize=10)
    
    # Info box
    max_sens = max(s['dM_dEI'] for s in sensitivities.values())
    min_sens = min(s['dM_dEI'] for s in sensitivities.values())
    info_text = (f'Max ∂M/∂(EI): {max_sens:.4e}\n'
                f'Min ∂M/∂(EI): {min_sens:.4e}\n'
                f'Total: {total_sensitivity:.4e}\n'
                f'Scale: {SENSITIVITY_SCALE:.0e}')
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if save_figures:
        filepath = os.path.join(output_folder, 'sensitivity_diagram.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    
    # =========================================================================
    # PLOT 5: All Three Diagrams Stacked Vertically
    # =========================================================================
    
    fig5, axes = plt.subplots(3, 1, figsize=(14, 14), dpi=FIGURE_DPI)
    fig5.suptitle('Sensitivity Analysis: Complete Diagram Set', fontsize=14, fontweight='bold')
    
    # Primary Moment
    ax = axes[0]
    ax.set_title('Primary Moment M(x)', fontsize=12, fontweight='bold')
    plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    plot_moment_diagram_on_structure(ax, points, cells, cell_moment_primary,
                                     scale=PRIMARY_MOMENT_SCALE,
                                     color=COLOR_PRIMARY_MOMENT,
                                     fill_color=COLOR_FILL_PRIMARY,
                                     show_values=True, use_cell_data=True)
    add_supports_l(ax, points, support_indices=left_supports)
    add_supports_r(ax, points, support_indices=right_supports)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Y (m)')
    
    # Dual Moment
    ax = axes[1]
    ax.set_title('Dual Moment M̄(x)', fontsize=12, fontweight='bold')
    plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    plot_moment_diagram_on_structure(ax, points, cells, cell_moment_dual,
                                     scale=DUAL_MOMENT_SCALE,
                                     color=COLOR_DUAL_MOMENT,
                                     fill_color=COLOR_FILL_DUAL,
                                     show_values=True, use_cell_data=True)
    add_supports_l(ax, points, support_indices=left_supports)
    add_supports_r(ax, points, support_indices=right_supports)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Y (m)')
    
    # Sensitivity
    ax = axes[2]
    ax.set_title('Sensitivity ∂M/∂(EI)', fontsize=12, fontweight='bold')
    plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    plot_sensitivity_diagram_on_structure(ax, points, cells, sensitivities,
                                          scale=SENSITIVITY_SCALE,
                                          show_values=True)
    add_supports_l(ax, points, support_indices=left_supports)
    add_supports_r(ax, points, support_indices=right_supports)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_figures:
        filepath = os.path.join(output_folder, 'sensitivity_stacked.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    
    return


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(primary_vtk_path, dual_vtk_path, E, I, beam_length, n_elements,
         response_element=None, create_plots=True, save_plots=True, output_dir="."):
    """
    Main function - compute sensitivity from VTK files with visualization.
    """
    
    print("=" * 75)
    print("MOMENT SENSITIVITY ANALYSIS: ∂M/∂(EI)")
    print("Using Adjoint Method with VTK File Parsing")
    print("=" * 75)
    
    # =========================================
    # Parse Primary VTK File
    # =========================================
    print(f"\n1. Loading Primary VTK: {primary_vtk_path}")
    vtk_primary = parse_vtk_file(primary_vtk_path)
    M_primary = parse_vtk_cell_moments(primary_vtk_path)
    
    if not M_primary:
        print("Error: Could not parse primary moments from VTK file.")
        return None, None
    
    print(f"   Found {len(M_primary)} elements with primary moments")
    
    # =========================================
    # Parse Dual VTK File
    # =========================================
    print(f"\n2. Loading Dual VTK: {dual_vtk_path}")
    vtk_dual = parse_vtk_file(dual_vtk_path)
    M_dual = parse_vtk_cell_moments(dual_vtk_path)
    
    if not M_dual:
        print("Error: Could not parse dual moments from VTK file.")
        return None, None
    
    print(f"   Found {len(M_dual)} elements with dual moments")
    
    # =========================================
    # Reconcile element data
    # =========================================
    print("\n3. Reconciling element data...")
    
    primary_elem_ids = set(M_primary.keys())
    dual_elem_ids = set(M_dual.keys())
    common_elem_ids = primary_elem_ids & dual_elem_ids
    
    print(f"   Primary elements: {len(primary_elem_ids)}")
    print(f"   Dual elements: {len(dual_elem_ids)}")
    print(f"   Common elements: {len(common_elem_ids)}")
    
    if len(common_elem_ids) < len(primary_elem_ids):
        print(f"   Warning: Using only {len(common_elem_ids)} common elements")
    
    # Filter to common elements
    M_primary_common = {k: v for k, v in M_primary.items() if k in common_elem_ids}
    M_dual_common = {k: v for k, v in M_dual.items() if k in common_elem_ids}
    
    # Calculate element lengths from VTK
    points = vtk_primary['points']
    cells = vtk_primary['cells']
    
    L_elements = {}
    for idx, cell in enumerate(cells):
        elem_id = idx + 1
        if elem_id in common_elem_ids:
            p1 = points[cell[0]]
            p2 = points[cell[1]]
            L_elements[elem_id] = get_element_length(p1, p2)
    
    actual_n_elements = len(common_elem_ids)
    
    # =========================================
    # Compute Sensitivities
    # =========================================
    print("\n4. Computing sensitivities...")
    
    sensitivities, total_sensitivity = compute_moment_sensitivity(
        E, I, L_elements, M_primary_common, M_dual_common
    )
    
    # =========================================
    # Print Results
    # =========================================
    print_results(E, I, sensitivities, total_sensitivity, response_element)
    
    # =========================================
    # Create Visualizations
    # =========================================
    if create_plots:
        print("\n5. Creating visualizations...")
        
        create_sensitivity_visualization(
            vtk_primary, vtk_dual, E, I,
            sensitivities, total_sensitivity,
            M_primary_common, M_dual_common,
            save_figures=save_plots,
            output_folder=output_dir
        )
    
    return sensitivities, total_sensitivity


# =============================================================================
# RUN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    
    # =========================================
    # USER INPUT - MODIFY THESE VALUES
    # =========================================
    
    # VTK file paths
    PRIMARY_VTK = "test_files/SA_beam_2D_udl_kink.gid/vtk_output/Parts_Beam_Beams_0_1.vtk"
    DUAL_VTK = "test_files/SA_beam_2D_udl_kink.gid/vtk_output_dual/Parts_Beam_Beams_0_1.vtk"
    
    # Material properties
    E = 2.1e11      # Young's modulus [Pa]
    I = 5.0e-6      # Second moment of area [m^4]
    
    # Geometry
    BEAM_LENGTH = 2.0   # Total beam length [m]
    N_ELEMENTS = 20     # Number of elements
    
    # Response element (optional)
    RESPONSE_ELEMENT = None
    
    # Plotting options
    CREATE_PLOTS = True
    SAVE_PLOTS = True
    OUTPUT_DIR = OUTPUT_FOLDER
    
    # =========================================
    # Run Analysis
    # =========================================
    sensitivities, total = main(
        primary_vtk_path=PRIMARY_VTK,
        dual_vtk_path=DUAL_VTK,
        E=E,
        I=I,
        beam_length=BEAM_LENGTH,
        n_elements=N_ELEMENTS,
        response_element=RESPONSE_ELEMENT,
        create_plots=CREATE_PLOTS,
        save_plots=SAVE_PLOTS,
        output_dir=OUTPUT_DIR
    )