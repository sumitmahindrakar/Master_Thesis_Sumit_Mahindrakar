"""
VTK Frame Structure Plotter - Complete Version
===============================================
This script reads a VTK file containing frame structure data and plots:
1. Original frame structure
2. Deformed shape overlay
3. Deflection diagram (perpendicular to elements)
4. Bending moment diagram
5. Shear force diagram
6. Axial force diagram

Usage:
------
    python vtk_plotter.py your_structure.vtk

Or set VTK_FILE_PATH variable and run directly.

Author: Your Name
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
import os
import sys

# =============================================================================
# USER INPUT - SET YOUR VTK FILE PATH HERE
# =============================================================================

VTK_FILE_PATH = "test_files/frame_2D_test_udl_2_floor_fine_refined_high_A.gid/vtk_output/Parts_Beam_Beams_0_1.vtk"  # <-- Change this to your VTK file path

# =============================================================================
# CONFIGURATION - Easy to modify parameters
# =============================================================================

# --- Scale Factors ---
DEFLECTION_SCALE = 1000          # Scale for deflection diagram
MOMENT_SCALE = 0.0005            # Scale for bending moment diagram
SHEAR_SCALE = 0.0001             # Scale for shear force diagram
AXIAL_SCALE = 0.00005            # Scale for axial force diagram
DEFORMED_SCALE = 500             # Scale for deformed shape

# --- Colors ---
COLOR_STRUCTURE = 'black'              # Original structure
COLOR_DEFORMED = 'green'               # Deformed shape
COLOR_DEFLECTION = 'blue'              # Deflection diagram
COLOR_MOMENT = 'red'                   # Bending moment diagram
COLOR_SHEAR = 'purple'                 # Shear force diagram
COLOR_AXIAL_TENSION = 'orange'         # Axial tension
COLOR_AXIAL_COMPRESSION = 'brown'      # Axial compression

# --- Fill Colors (with transparency) ---
COLOR_FILL_DEFLECTION = 'lightblue'
COLOR_FILL_MOMENT = 'lightcoral'
COLOR_FILL_SHEAR = 'plum'
COLOR_FILL_AXIAL = 'moccasin'

# --- Line Widths ---
LINEWIDTH_STRUCTURE = 2.5
LINEWIDTH_DEFORMED = 2.0
LINEWIDTH_DIAGRAM = 1.5
LINEWIDTH_THIN = 0.5

# --- Figure Settings ---
FIGURE_DPI = 100
SAVE_FIGURES = True
OUTPUT_FOLDER = "test_files/frame_2D_test_udl_2_floor_fine_refined_high_A.gid/plots"
# --- Support Settings ---
SUPPORT_SIZE = 0.15              # Size of support symbols
SHOW_SUPPORTS = True             # Whether to show support symbols

# --- Annotation Settings ---
SHOW_VALUES = True               # Show values on diagrams
FONT_SIZE_VALUES = 10             # Font size for value annotations
FONT_SIZE_TITLE = 14             # Font size for titles
FONT_SIZE_LABELS = 10            # Font size for axis labels

# --- What to Plot ---
PLOT_STRUCTURE = False
PLOT_DEFORMED = True
PLOT_DEFLECTION = True
PLOT_MOMENT = True
PLOT_SHEAR = False
PLOT_AXIAL = False


# =============================================================================
# VTK FILE PARSER
# =============================================================================

def parse_vtk_file(filename):
    """
    Parse a VTK file and extract points, cells, and field data.
    
    Parameters:
    -----------
    filename : str
        Path to the VTK file
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'points': numpy array of point coordinates (Nx3)
        - 'cells': list of cell connectivity
        - 'point_data': dict of point-based field data
        - 'cell_data': dict of cell-based field data
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
        
        # Parse POINTS
        if line.startswith('POINTS'):
            parts = line.split()
            num_points = int(parts[1])
            points = []
            i += 1
            
            while len(points) < num_points and i < len(lines):
                values = lines[i].strip().split()
                j = 0
                while j + 2 < len(values) and len(points) < num_points:
                    points.append([float(values[j]), float(values[j+1]), float(values[j+2])])
                    j += 3
                if j < len(values) and len(points) < num_points:
                    remaining = values[j:]
                    i += 1
                    if i < len(lines):
                        next_values = lines[i].strip().split()
                        combined = remaining + next_values
                        if len(combined) >= 3:
                            points.append([float(combined[0]), float(combined[1]), float(combined[2])])
                i += 1
            
            data['points'] = np.array(points[:num_points])
            print(f"  - Loaded {num_points} points")
            continue
        
        # Parse CELLS
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
        
        # Parse CELL_TYPES
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
                        
                        if values and values[0] in ['POINT_DATA', 'CELL_DATA', 
                                                     'FIELD', 'SCALARS', 'VECTORS']:
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
                        print(f"  - Loaded {current_section[:-5]} field: {field_name}")
                    continue
                    
                except (ValueError, IndexError):
                    pass
        
        i += 1
    
    return data


# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================

def get_element_direction(p1, p2):
    """Get unit direction vector from p1 to p2."""
    direction = np.array(p2) - np.array(p1)
    length = np.linalg.norm(direction)
    if length > 0:
        return direction / length
    return np.array([1, 0, 0])


def get_element_length(p1, p2):
    """Get length of element."""
    return np.linalg.norm(np.array(p2) - np.array(p1))


def get_perpendicular_direction(p1, p2):
    """
    Get perpendicular direction (90° counterclockwise rotation).
    
    For vertical elements: perpendicular points left (negative X)
    For horizontal elements: perpendicular points up (positive Y)
    """
    direction = get_element_direction(p1, p2)
    # Rotate 90° counterclockwise: (x, y) -> (-y, x)
    perpendicular = np.array([-direction[1], direction[0], 0])
    return perpendicular


def get_element_type(p1, p2, tolerance=0.1):
    """
    Determine element type based on orientation.
    
    Returns: 'column' (vertical), 'beam' (horizontal), or 'inclined'
    """
    direction = get_element_direction(p1, p2)
    
    if abs(direction[1]) > 1 - tolerance:
        return 'column'
    elif abs(direction[0]) > 1 - tolerance:
        return 'beam'
    else:
        return 'inclined'


def get_local_coordinates(p1, p2):
    """
    Get local coordinate system for an element.
    
    Returns:
    --------
    tuple: (local_x, local_y) unit vectors
        local_x: along element direction
        local_y: perpendicular to element (counterclockwise)
    """
    local_x = get_element_direction(p1, p2)[:2]
    local_y = np.array([-local_x[1], local_x[0]])
    return local_x, local_y


# =============================================================================
# SUPPORT SYMBOLS
# =============================================================================

class SupportType:
    """Enumeration for support types."""
    FIXED = 'fixed'
    PINNED = 'pinned'
    ROLLER_X = 'roller_x'  # Roller allowing X movement
    ROLLER_Y = 'roller_y'  # Roller allowing Y movement


def detect_supports(points, displacement, tolerance=1e-10):
    """
    Automatically detect support types based on displacement constraints.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Point coordinates
    displacement : numpy.ndarray
        Displacement at each point
        
    Returns:
    --------
    dict : {node_index: support_type}
    """
    supports = {}
    
    for i in range(len(points)):
        dx = abs(displacement[i, 0]) < tolerance
        dy = abs(displacement[i, 1]) < tolerance
        
        if dx and dy:
            # Both directions fixed - could be fixed or pinned
            # Check if at bottom of structure
            if points[i, 1] <= np.min(points[:, 1]) + 0.01:
                supports[i] = SupportType.FIXED
        elif dx and not dy:
            supports[i] = SupportType.ROLLER_Y
        elif dy and not dx:
            supports[i] = SupportType.ROLLER_X
    
    return supports


def draw_fixed_support(ax, x, y, size=SUPPORT_SIZE, angle=0):
    """
    Draw a fixed support symbol (triangle with ground hatching).
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
    x, y : float
        Position of support
    size : float
        Size of support symbol
    angle : float
        Rotation angle in degrees
    """
    # Triangle vertices
    triangle = plt.Polygon([
        [x, y],
        [x - size, y - size],
        [x + size, y - size]
    ], fill=False, edgecolor='black', linewidth=1.5, zorder=10)
    ax.add_patch(triangle)
    
    # Ground line
    ax.plot([x - size*1.3, x + size*1.3], [y - size, y - size], 
           'k-', linewidth=2, zorder=10)
    
    # Hatching lines
    num_hatches = 5
    for hx in np.linspace(x - size*1.1, x + size*0.9, num_hatches):
        ax.plot([hx, hx - size*0.3], [y - size, y - size*1.4], 
               'k-', linewidth=0.8, zorder=10)


def draw_pinned_support(ax, x, y, size=SUPPORT_SIZE):
    """
    Draw a pinned support symbol (triangle with circle at pin).
    """
    # Triangle
    triangle = plt.Polygon([
        [x, y],
        [x - size, y - size],
        [x + size, y - size]
    ], fill=False, edgecolor='black', linewidth=1.5, zorder=10)
    ax.add_patch(triangle)
    
    # Pin circle
    circle = Circle((x, y), size*0.15, fill=False, edgecolor='black', 
                    linewidth=1.5, zorder=11)
    ax.add_patch(circle)
    
    # Ground line
    ax.plot([x - size*1.3, x + size*1.3], [y - size, y - size], 
           'k-', linewidth=2, zorder=10)
    
    # Hatching
    for hx in np.linspace(x - size*1.1, x + size*0.9, 5):
        ax.plot([hx, hx - size*0.3], [y - size, y - size*1.4], 
               'k-', linewidth=0.8, zorder=10)


def draw_roller_support(ax, x, y, size=SUPPORT_SIZE, direction='y'):
    """
    Draw a roller support symbol.
    
    Parameters:
    -----------
    direction : str
        'y' for roller on horizontal surface (allows X movement)
        'x' for roller on vertical surface (allows Y movement)
    """
    if direction == 'y':
        # Roller on ground (horizontal)
        # Triangle
        triangle = plt.Polygon([
            [x, y],
            [x - size*0.8, y - size*0.6],
            [x + size*0.8, y - size*0.6]
        ], fill=False, edgecolor='black', linewidth=1.5, zorder=10)
        ax.add_patch(triangle)
        
        # Rollers (circles)
        circle1 = Circle((x - size*0.4, y - size*0.75), size*0.15, 
                         fill=False, edgecolor='black', linewidth=1.5, zorder=10)
        circle2 = Circle((x + size*0.4, y - size*0.75), size*0.15, 
                         fill=False, edgecolor='black', linewidth=1.5, zorder=10)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        
        # Ground line
        ax.plot([x - size*1.3, x + size*1.3], [y - size*0.9, y - size*0.9], 
               'k-', linewidth=2, zorder=10)
    else:
        # Roller on wall (vertical)
        triangle = plt.Polygon([
            [x, y],
            [x - size*0.6, y - size*0.8],
            [x - size*0.6, y + size*0.8]
        ], fill=False, edgecolor='black', linewidth=1.5, zorder=10)
        ax.add_patch(triangle)
        
        # Rollers
        circle1 = Circle((x - size*0.75, y - size*0.4), size*0.15, 
                         fill=False, edgecolor='black', linewidth=1.5, zorder=10)
        circle2 = Circle((x - size*0.75, y + size*0.4), size*0.15, 
                         fill=False, edgecolor='black', linewidth=1.5, zorder=10)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        
        # Wall line
        ax.plot([x - size*0.9, x - size*0.9], [y - size*1.3, y + size*1.3], 
               'k-', linewidth=2, zorder=10)


# def add_supports(ax, points, supports=None, displacement=None, size=SUPPORT_SIZE):
#     """
#     Add support symbols to the plot.
    
#     Parameters:
#     -----------
#     ax : matplotlib.axes.Axes
#     points : numpy.ndarray
#         Point coordinates
#     supports : dict
#         {node_index: support_type}. If None, auto-detect from displacement.
#     displacement : numpy.ndarray
#         Displacement data for auto-detection
#     size : float
#         Size of support symbols
#     """
#     if not SHOW_SUPPORTS:
#         return
    
#     if supports is None:
#         if displacement is not None:
#             supports = detect_supports(points, displacement)
#         else:
#             # Default: assume bottom nodes are fixed
#             min_y = np.min(points[:, 1])
#             supports = {}
#             for i in range(len(points)):
#                 if abs(points[i, 1] - min_y) < 0.01:
#                     supports[i] = SupportType.FIXED
    
    # for node_idx, support_type in supports.items():
    #     x, y = points[node_idx, 0], points[node_idx, 1]
        
    #     if support_type == SupportType.FIXED:
    #         draw_fixed_support(ax, x, y, size)
    #     elif support_type == SupportType.PINNED:
    #         draw_pinned_support(ax, x, y, size)
    #     elif support_type == SupportType.ROLLER_X:
    #         draw_roller_support(ax, x, y, size, direction='y')
    #     elif support_type == SupportType.ROLLER_Y:
    #         draw_roller_support(ax, x, y, size, direction='x')

def add_supports(ax, points, support_indices=None):
    """
    Add support symbols at fixed nodes.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    points : numpy.ndarray
        Array of point coordinates
    support_indices : list
        Indices of nodes with supports. If None, assumes bottom nodes are fixed.
    """
    
    if support_indices is None:
        # Auto-detect: assume nodes at minimum y are supports
        min_y = np.min(points[:, 1])
        support_indices = np.where(np.abs(points[:, 1] - min_y) < 0.01)[0]
    
    # Triangle size for support symbol
    size = 0.15
    
    for idx in support_indices:
        x, y = points[idx][0], points[idx][1]
        
        # Draw fixed support (triangle)
        triangle = plt.Polygon([
            [x, y],
            [x - size, y - size],
            [x + size, y - size]
        ], fill=False, edgecolor='black', linewidth=1.5)
        ax.add_patch(triangle)
        
        # Draw ground line
        ax.plot([x - size*1.2, x + size*1.2], [y - size, y - size], 
               'k-', linewidth=1.5)
        
        # Draw hatching lines
        for hx in np.linspace(x - size, x + size, 4):
            ax.plot([hx, hx - size*0.3], [y - size, y - size*1.3], 
                   'k-', linewidth=0.8)

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_structure(ax, points, cells, color=COLOR_STRUCTURE, 
                   linewidth=LINEWIDTH_STRUCTURE, label='Structure',
                   show_node_labels=False, show_element_labels=False):
    """
    Plot the frame structure.
    """
    for idx, cell in enumerate(cells):
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        
        if idx == 0:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                   color=color, linewidth=linewidth, label=label, zorder=5)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                   color=color, linewidth=linewidth, zorder=5)
        
        # Element labels
        if show_element_labels:
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            ax.annotate(f'E{idx}', (mid_x, mid_y), fontsize=7, color='gray',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                alpha=0.7, edgecolor='none'))
    
    # Nodes
    ax.scatter(points[:, 0], points[:, 1], color=color, s=40, zorder=6)
    
    # Node labels
    if show_node_labels:
        for i, point in enumerate(points):
            ax.annotate(f'{i}', (point[0], point[1]), 
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=8, color='darkgray')


def plot_deformed_shape(ax, points, cells, displacement, 
                        scale=DEFORMED_SCALE,
                        color=COLOR_DEFORMED, 
                        linewidth=LINEWIDTH_DEFORMED,
                        linestyle='-',
                        label='Deformed Shape'):
    """
    Plot the deformed shape of the structure.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
    points : numpy.ndarray
        Original point coordinates
    cells : list
        Cell connectivity
    displacement : numpy.ndarray
        Displacement at each point
    scale : float
        Scale factor for deformation visualization
    """
    # Calculate deformed positions
    deformed_points = points.copy()
    deformed_points[:, 0] += displacement[:, 0] * scale
    deformed_points[:, 1] += displacement[:, 1] * scale
    
    for idx, cell in enumerate(cells):
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = deformed_points[p1_idx]
        p2 = deformed_points[p2_idx]
        
        if idx == 0:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                   color=color, linewidth=linewidth, linestyle=linestyle,
                   label=label, zorder=4)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                   color=color, linewidth=linewidth, linestyle=linestyle, zorder=4)
    
    # Deformed nodes
    ax.scatter(deformed_points[:, 0], deformed_points[:, 1], 
              color=color, s=30, zorder=5, alpha=0.7)


def plot_deflection_diagram(ax, points, cells, displacement, 
                            scale=DEFLECTION_SCALE,
                            color=COLOR_DEFLECTION, 
                            fill_color=COLOR_FILL_DEFLECTION,
                            linewidth=LINEWIDTH_DIAGRAM, 
                            show_fill=True,
                            show_values=False):
    """
    Plot deflection diagram perpendicular to each element.
    """
    for idx, cell in enumerate(cells):
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = points[p1_idx][:2]
        p2 = points[p2_idx][:2]
        
        perp = get_perpendicular_direction(p1, p2)[:2]
        
        disp1 = displacement[p1_idx][:2]
        disp2 = displacement[p2_idx][:2]
        
        # Project displacement onto perpendicular direction
        disp1_perp = np.dot(disp1, perp)
        disp2_perp = np.dot(disp2, perp)
        
        offset_p1 = p1 + perp * disp1_perp * scale
        offset_p2 = p2 + perp * disp2_perp * scale
        
        # Diagram line
        ax.plot([offset_p1[0], offset_p2[0]], [offset_p1[1], offset_p2[1]], 
               color=color, linewidth=linewidth, zorder=3)
        
        # Connecting lines
        ax.plot([p1[0], offset_p1[0]], [p1[1], offset_p1[1]], 
               color=color, linewidth=LINEWIDTH_THIN, linestyle='--', alpha=0.7)
        ax.plot([p2[0], offset_p2[0]], [p2[1], offset_p2[1]], 
               color=color, linewidth=LINEWIDTH_THIN, linestyle='--', alpha=0.7)
        
        # Fill
        if show_fill:
            polygon_x = [p1[0], offset_p1[0], offset_p2[0], p2[0]]
            polygon_y = [p1[1], offset_p1[1], offset_p2[1], p2[1]]
            ax.fill(polygon_x, polygon_y, color=fill_color, alpha=0.3, zorder=2)
        
        # Values
        if show_values:
            for point, disp_val, offset_pt in [(p1, disp1_perp, offset_p1), 
                                                (p2, disp2_perp, offset_p2)]:
                if abs(disp_val) > 1e-8:
                    ax.annotate(f'{disp_val:.2e}', 
                               xy=offset_pt, fontsize=FONT_SIZE_VALUES,
                               color=color, ha='center', va='bottom')


def plot_moment_diagram(ax, points, cells, moment_data, 
                        scale=MOMENT_SCALE,
                        color=COLOR_MOMENT, 
                        fill_color=COLOR_FILL_MOMENT,
                        linewidth=LINEWIDTH_DIAGRAM, 
                        show_fill=True,
                        show_values=True, 
                        use_cell_data=True):
    """
    Plot bending moment diagram perpendicular to each element.
    """
    for idx, cell in enumerate(cells):
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = points[p1_idx][:2]
        p2 = points[p2_idx][:2]
        
        perp = get_perpendicular_direction(p1, p2)[:2]
        
        if use_cell_data:
            moment_value = moment_data[idx][2]
            offset_p1 = p1 + perp * moment_value * scale
            offset_p2 = p2 + perp * moment_value * scale
        else:
            moment1 = moment_data[p1_idx][2]
            moment2 = moment_data[p2_idx][2]
            offset_p1 = p1 + perp * moment1 * scale
            offset_p2 = p2 + perp * moment2 * scale
        
        # Diagram line
        ax.plot([offset_p1[0], offset_p2[0]], [offset_p1[1], offset_p2[1]], 
               color=color, linewidth=linewidth, zorder=3)
        
        # Connecting lines
        ax.plot([p1[0], offset_p1[0]], [p1[1], offset_p1[1]], 
               color=color, linewidth=LINEWIDTH_THIN, zorder=3)
        ax.plot([p2[0], offset_p2[0]], [p2[1], offset_p2[1]], 
               color=color, linewidth=LINEWIDTH_THIN, zorder=3)
        
        # Fill
        if show_fill:
            polygon_x = [p1[0], offset_p1[0], offset_p2[0], p2[0]]
            polygon_y = [p1[1], offset_p1[1], offset_p2[1], p2[1]]
            ax.fill(polygon_x, polygon_y, color=fill_color, alpha=0.3, zorder=2)
        
        # Values
        if show_values and use_cell_data and abs(moment_value) > 1:
            mid_offset = (offset_p1 + offset_p2) / 2
            ax.annotate(f'{moment_value:.0f}', 
                       xy=(mid_offset[0], mid_offset[1]),
                       fontsize=FONT_SIZE_VALUES, color=color,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.15', 
                                facecolor='white', alpha=0.8, 
                                edgecolor='none'))


def plot_shear_diagram(ax, points, cells, force_data, 
                       scale=SHEAR_SCALE,
                       color=COLOR_SHEAR, 
                       fill_color=COLOR_FILL_SHEAR,
                       linewidth=LINEWIDTH_DIAGRAM, 
                       show_fill=True,
                       show_values=True, 
                       use_cell_data=True):
    """
    Plot shear force diagram perpendicular to each element.
    
    Shear is the force component perpendicular to the element axis.
    """
    for idx, cell in enumerate(cells):
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = points[p1_idx][:2]
        p2 = points[p2_idx][:2]
        
        perp = get_perpendicular_direction(p1, p2)[:2]
        local_x, local_y = get_local_coordinates(p1, p2)
        
        if use_cell_data:
            # Extract shear force (perpendicular component)
            force = force_data[idx][:2]
            # Shear is the force component perpendicular to element
            shear_value = np.dot(force, local_y)
            
            offset_p1 = p1 + perp * shear_value * scale
            offset_p2 = p2 + perp * shear_value * scale
        else:
            force1 = force_data[p1_idx][:2]
            force2 = force_data[p2_idx][:2]
            shear1 = np.dot(force1, local_y)
            shear2 = np.dot(force2, local_y)
            
            offset_p1 = p1 + perp * shear1 * scale
            offset_p2 = p2 + perp * shear2 * scale
            shear_value = (shear1 + shear2) / 2
        
        # Diagram line
        ax.plot([offset_p1[0], offset_p2[0]], [offset_p1[1], offset_p2[1]], 
               color=color, linewidth=linewidth, zorder=3)
        
        # Connecting lines
        ax.plot([p1[0], offset_p1[0]], [p1[1], offset_p1[1]], 
               color=color, linewidth=LINEWIDTH_THIN, zorder=3)
        ax.plot([p2[0], offset_p2[0]], [p2[1], offset_p2[1]], 
               color=color, linewidth=LINEWIDTH_THIN, zorder=3)
        
        # Fill
        if show_fill:
            polygon_x = [p1[0], offset_p1[0], offset_p2[0], p2[0]]
            polygon_y = [p1[1], offset_p1[1], offset_p2[1], p2[1]]
            ax.fill(polygon_x, polygon_y, color=fill_color, alpha=0.3, zorder=2)
        
        # Values
        if show_values and abs(shear_value) > 1:
            mid_offset = (offset_p1 + offset_p2) / 2
            ax.annotate(f'{shear_value:.0f}', 
                       xy=(mid_offset[0], mid_offset[1]),
                       fontsize=FONT_SIZE_VALUES, color=color,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.15', 
                                facecolor='white', alpha=0.8, 
                                edgecolor='none'))


def plot_axial_diagram(ax, points, cells, force_data, 
                       scale=AXIAL_SCALE,
                       color_tension=COLOR_AXIAL_TENSION,
                       color_compression=COLOR_AXIAL_COMPRESSION,
                       fill_color=COLOR_FILL_AXIAL,
                       linewidth=LINEWIDTH_DIAGRAM, 
                       show_fill=True,
                       show_values=True, 
                       use_cell_data=True):
    """
    Plot axial force diagram along each element.
    
    Axial force is the force component along the element axis.
    Positive = tension, Negative = compression
    """
    for idx, cell in enumerate(cells):
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = points[p1_idx][:2]
        p2 = points[p2_idx][:2]
        
        perp = get_perpendicular_direction(p1, p2)[:2]
        local_x, local_y = get_local_coordinates(p1, p2)
        
        if use_cell_data:
            force = force_data[idx][:2]
            # Axial is the force component along element axis
            axial_value = np.dot(force, local_x)
            
            # Offset perpendicular to show diagram
            offset_p1 = p1 + perp * axial_value * scale
            offset_p2 = p2 + perp * axial_value * scale
        else:
            force1 = force_data[p1_idx][:2]
            force2 = force_data[p2_idx][:2]
            axial1 = np.dot(force1, local_x)
            axial2 = np.dot(force2, local_x)
            
            offset_p1 = p1 + perp * axial1 * scale
            offset_p2 = p2 + perp * axial2 * scale
            axial_value = (axial1 + axial2) / 2
        
        # Choose color based on tension/compression
        if axial_value > 0:
            color = color_tension
            force_type = 'T'  # Tension
        else:
            color = color_compression
            force_type = 'C'  # Compression
        
        # Diagram line
        ax.plot([offset_p1[0], offset_p2[0]], [offset_p1[1], offset_p2[1]], 
               color=color, linewidth=linewidth, zorder=3)
        
        # Connecting lines
        ax.plot([p1[0], offset_p1[0]], [p1[1], offset_p1[1]], 
               color=color, linewidth=LINEWIDTH_THIN, zorder=3)
        ax.plot([p2[0], offset_p2[0]], [p2[1], offset_p2[1]], 
               color=color, linewidth=LINEWIDTH_THIN, zorder=3)
        
        # Fill
        if show_fill:
            polygon_x = [p1[0], offset_p1[0], offset_p2[0], p2[0]]
            polygon_y = [p1[1], offset_p1[1], offset_p2[1], p2[1]]
            ax.fill(polygon_x, polygon_y, color=fill_color, alpha=0.3, zorder=2)
        
        # Values
        if show_values and abs(axial_value) > 1:
            mid_offset = (offset_p1 + offset_p2) / 2
            ax.annotate(f'{abs(axial_value):.0f} ({force_type})', 
                       xy=(mid_offset[0], mid_offset[1]),
                       fontsize=FONT_SIZE_VALUES, color=color,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.15', 
                                facecolor='white', alpha=0.8, 
                                edgecolor='none'))


# =============================================================================
# MAIN PLOTTING FUNCTIONS
# =============================================================================

def setup_axes(ax, title, points):
    """Configure axes with common settings."""
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABELS)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add some padding to axis limits
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    padding = max(x_range, y_range) * 0.15
    
    ax.set_xlim(points[:, 0].min() - padding, points[:, 0].max() + padding)
    ax.set_ylim(points[:, 1].min() - padding*1.5, points[:, 1].max() + padding)


def create_combined_plot(vtk_data, save_path=None):
    """Create a combined plot with structure, deformed shape, and all diagrams."""
    
    points = vtk_data['points']
    cells = vtk_data['cells']
    
    displacement = vtk_data['point_data'].get('DISPLACEMENT', np.zeros((len(points), 3)))
    cell_moment = vtk_data['cell_data'].get('MOMENT', np.zeros((len(cells), 3)))
    cell_force = vtk_data['cell_data'].get('FORCE', np.zeros((len(cells), 3)))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=FIGURE_DPI)
    
    # --- Plot 1: Structure with node labels ---
    ax = axes[0, 0]
    setup_axes(ax, 'Frame Structure', points)
    plot_structure(ax, points, cells, show_node_labels=True, show_element_labels=True)
    add_supports(ax, points)
    ax.legend(loc='upper right')
    
    # --- Plot 2: Deformed Shape ---
    ax = axes[0, 1]
    setup_axes(ax, f'Deformed Shape (Scale: {DEFORMED_SCALE}x)', points)
    plot_structure(ax, points, cells, color='lightgray', linewidth=1.5, label='Original')
    plot_deformed_shape(ax, points, cells, displacement, scale=DEFORMED_SCALE)
    add_supports(ax, points)
    ax.legend(loc='upper right')
    
    # --- Plot 3: Deflection Diagram ---
    ax = axes[0, 2]
    setup_axes(ax, f'Deflection Diagram (Scale: {DEFLECTION_SCALE}x)', points)
    plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    plot_deflection_diagram(ax, points, cells, displacement, scale=DEFLECTION_SCALE)
    add_supports(ax, points)
    
    # --- Plot 4: Bending Moment Diagram ---
    ax = axes[1, 0]
    setup_axes(ax, f'Bending Moment Diagram (Scale: {MOMENT_SCALE}x)', points)
    plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    plot_moment_diagram(ax, points, cells, cell_moment, scale=MOMENT_SCALE,
                       show_values=SHOW_VALUES)
    add_supports(ax, points)
    
    # --- Plot 5: Shear Force Diagram ---
    ax = axes[1, 1]
    setup_axes(ax, f'Shear Force Diagram (Scale: {SHEAR_SCALE}x)', points)
    plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    plot_shear_diagram(ax, points, cells, cell_force, scale=SHEAR_SCALE,
                      show_values=SHOW_VALUES)
    add_supports(ax, points)
    
    # --- Plot 6: Axial Force Diagram ---
    ax = axes[1, 2]
    setup_axes(ax, f'Axial Force Diagram (Scale: {AXIAL_SCALE}x)', points)
    plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    plot_axial_diagram(ax, points, cells, cell_force, scale=AXIAL_SCALE,
                      show_values=SHOW_VALUES)
    add_supports(ax, points)
    
    # Add tension/compression legend
    tension_line = Line2D([0], [0], color=COLOR_AXIAL_TENSION, linewidth=2)
    compression_line = Line2D([0], [0], color=COLOR_AXIAL_COMPRESSION, linewidth=2)
    ax.legend([tension_line, compression_line], ['Tension', 'Compression'], 
             loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    # plt.show()


def create_individual_plots(vtk_data, output_folder=OUTPUT_FOLDER):
    """Create individual detailed plots for each diagram type."""
    
    points = vtk_data['points']
    cells = vtk_data['cells']
    
    displacement = vtk_data['point_data'].get('DISPLACEMENT', np.zeros((len(points), 3)))
    cell_moment = vtk_data['cell_data'].get('MOMENT', np.zeros((len(cells), 3)))
    cell_force = vtk_data['cell_data'].get('FORCE', np.zeros((len(cells), 3)))
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # --- 1. Structure Only ---
    # fig, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
    # setup_axes(ax, 'Frame Structure', points)
    # plot_structure(ax, points, cells, show_node_labels=True, show_element_labels=True)
    # add_supports(ax, points)
    # ax.legend(loc='upper right')
    # plt.savefig(os.path.join(output_folder, '1_structure.png'), dpi=300, bbox_inches='tight')
    # print(f"Saved: {output_folder}/1_structure.png")
    # plt.show()
    
    # --- 2. Deformed Shape ---
    fig, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
    setup_axes(ax, f'Deformed Shape (Scale: {DEFORMED_SCALE}x)', points)
    plot_structure(ax, points, cells, color='lightgray', linewidth=1.5, label='Original')
    plot_deformed_shape(ax, points, cells, displacement, scale=DEFORMED_SCALE)
    add_supports(ax, points)
    ax.legend(loc='best')
    
    # # Add max displacement info
    max_disp = np.max(np.linalg.norm(displacement[:, :2], axis=1))
    # # ax.text(0.2, 0.2, f'Max displacement: {max_disp:.6e} m\nScale: {DEFORMED_SCALE}x',
    # #        transform=ax.transAxes, fontsize=10, va='top',
    # #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.savefig(os.path.join(output_folder, '2_deformed_shape.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_folder}/2_deformed_shape.png")
    plt.show()
    
    # --- 3. Deflection Diagram ---
    fig, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
    setup_axes(ax, f'Deflection Diagram (Scale: {DEFLECTION_SCALE}x)', points)
    
    plot_structure(ax, points, cells, color='lightgray', linewidth=1.5)
    plot_deflection_diagram(ax, points, cells, displacement, scale=DEFLECTION_SCALE,
                           show_values=False)
    add_supports(ax, points)

    ax.text(0.15, 0.1, f'Max displacement: {max_disp:.6e} m\nScale: {DEFORMED_SCALE}x',
           transform=ax.transAxes, fontsize=10, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.savefig(os.path.join(output_folder, '3_deflection.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_folder}/3_deflection.png")
    plt.show()
    
    # --- 4. Bending Moment Diagram ---
    fig, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
    setup_axes(ax, f'Bending Moment Diagram (Scale: {MOMENT_SCALE}x)', points)
    plt.xlim(-1.5,3.5)
    plt.ylim(-0.5,5)
    plot_structure(ax, points, cells, color='lightgray', linewidth=1.5)
    plot_moment_diagram(ax, points, cells, cell_moment, scale=MOMENT_SCALE,
                       show_values=False)
    add_supports(ax, points)
    
    # Add moment info
    max_moment = np.max(cell_moment[:, 2])
    min_moment = np.min(cell_moment[:, 2])

    ax.text(0.38, 0.1, f'Max BM: {max_moment:.1f}\nMin BM: {min_moment:.1f}',
           transform=ax.transAxes, fontsize=10, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.savefig(os.path.join(output_folder, '4_bending_moment.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_folder}/4_bending_moment.png")
    plt.show()
    
    # --- 5. Shear Force Diagram ---
    # fig, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
    # setup_axes(ax, f'Shear Force Diagram (Scale: {SHEAR_SCALE}x)', points)
    # plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    # plot_shear_diagram(ax, points, cells, cell_force, scale=SHEAR_SCALE,
    #                   show_values=True)
    # add_supports(ax, points)
    # plt.savefig(os.path.join(output_folder, '5_shear_force.png'), dpi=300, bbox_inches='tight')
    # print(f"Saved: {output_folder}/5_shear_force.png")
    # plt.show()
    
    # --- 6. Axial Force Diagram ---
    # fig, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
    # setup_axes(ax, f'Axial Force Diagram (Scale: {AXIAL_SCALE}x)', points)
    # plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    # plot_axial_diagram(ax, points, cells, cell_force, scale=AXIAL_SCALE,
    #                   show_values=True)
    # add_supports(ax, points)
    
    # # Legend
    # tension_line = Line2D([0], [0], color=COLOR_AXIAL_TENSION, linewidth=2)
    # compression_line = Line2D([0], [0], color=COLOR_AXIAL_COMPRESSION, linewidth=2)
    # ax.legend([tension_line, compression_line], ['Tension (T)', 'Compression (C)'], 
    #          loc='upper right')
    
    # plt.savefig(os.path.join(output_folder, '6_axial_force.png'), dpi=300, bbox_inches='tight')
    # print(f"Saved: {output_folder}/6_axial_force.png")
    # plt.show()


def print_summary(vtk_data):
    """Print detailed summary of VTK data."""
    
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    
    points = vtk_data['points']
    cells = vtk_data['cells']
    
    print(f"\n{'GEOMETRY':-^70}")
    print(f"  Number of nodes:    {len(points)}")
    print(f"  Number of elements: {len(cells)}")
    print(f"  X range: [{points[:, 0].min():.4f}, {points[:, 0].max():.4f}] m")
    print(f"  Y range: [{points[:, 1].min():.4f}, {points[:, 1].max():.4f}] m")
    
    # Element types
    print(f"\n{'ELEMENT TYPES':-^70}")
    for idx, cell in enumerate(cells):
        p1 = points[cell[0]][:2]
        p2 = points[cell[1]][:2]
        elem_type = get_element_type(p1, p2)
        length = get_element_length(p1, p2)
        print(f"  Element {idx:2d}: {elem_type:8s} (L = {length:.3f} m)")
    
    print(f"\n{'POINT DATA FIELDS':-^70}")
    for name, data in vtk_data['point_data'].items():
        print(f"  {name}: {data.shape}")
    
    print(f"\n{'CELL DATA FIELDS':-^70}")
    for name, data in vtk_data['cell_data'].items():
        print(f"  {name}: {data.shape}")
    
    # Displacement
    if 'DISPLACEMENT' in vtk_data['point_data']:
        disp = vtk_data['point_data']['DISPLACEMENT']
        print(f"\n{'DISPLACEMENT SUMMARY':-^70}")
        print(f"  Max X displacement: {np.max(np.abs(disp[:, 0])):.6e} m")
        print(f"  Max Y displacement: {np.max(np.abs(disp[:, 1])):.6e} m")
        max_total = np.max(np.linalg.norm(disp[:, :2], axis=1))
        max_idx = np.argmax(np.linalg.norm(disp[:, :2], axis=1))
        print(f"  Max total displacement: {max_total:.6e} m (at node {max_idx})")
    
    # Moment
    if 'MOMENT' in vtk_data['cell_data']:
        moment = vtk_data['cell_data']['MOMENT']
        print(f"\n{'BENDING MOMENT SUMMARY':-^70}")
        print(f"  Max moment: {np.max(moment[:, 2]):.2f}")
        print(f"  Min moment: {np.min(moment[:, 2]):.2f}")
        max_idx = np.argmax(np.abs(moment[:, 2]))
        print(f"  Max |M| at element {max_idx}: {moment[max_idx, 2]:.2f}")
    
    # Force
    if 'FORCE' in vtk_data['cell_data']:
        force = vtk_data['cell_data']['FORCE']
        print(f"\n{'FORCE SUMMARY':-^70}")
        print(f"  Max X force: {np.max(np.abs(force[:, 0])):.2f}")
        print(f"  Max Y force: {np.max(np.abs(force[:, 1])):.2f}")
    
    print("\n" + "=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the plotter."""
    
    print("=" * 70)
    print("VTK FRAME STRUCTURE PLOTTER - COMPLETE VERSION")
    print("=" * 70)
    
    # Get file path
    if len(sys.argv) > 1:
        vtk_file = sys.argv[1]
    else:
        vtk_file = VTK_FILE_PATH
    
    # Parse VTK file
    try:
        print(f"\n[1] Loading VTK file: {vtk_file}")
        vtk_data = parse_vtk_file(vtk_file)
        
        # Print summary
        print_summary(vtk_data)
        
        # Create output folder
        if SAVE_FIGURES and not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
            print(f"\nCreated output folder: {OUTPUT_FOLDER}")
        
        # Create plots
        print(f"\n[2] Creating combined plot...")
        combined_path = os.path.join(OUTPUT_FOLDER, 'combined_all_diagrams.png') if SAVE_FIGURES else None
        create_combined_plot(vtk_data, save_path=combined_path)
        
        print(f"\n[3] Creating individual detailed plots...")
        create_individual_plots(vtk_data, output_folder=OUTPUT_FOLDER)
        
        print("\n" + "=" * 70)
        print("COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        if SAVE_FIGURES:
            print(f"\nOutput files saved to: {os.path.abspath(OUTPUT_FOLDER)}/")
            print("  - combined_all_diagrams.png")
            print("  - 1_structure.png")
            print("  - 2_deformed_shape.png")
            print("  - 3_deflection.png")
            print("  - 4_bending_moment.png")
            print("  - 5_shear_force.png")
            print("  - 6_axial_force.png")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease provide a valid VTK file path:")
        print("  Option 1: Edit VTK_FILE_PATH at the top of the script")
        print("  Option 2: Run: python vtk_plotter.py your_file.vtk")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[ERROR] Failed to process VTK file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()