"""
VTK Frame Structure Plotter
===========================
This script reads a VTK file containing frame structure data and plots:
1. The original frame structure
2. Deflection diagram (perpendicular to elements)
3. Bending moment diagram (perpendicular to elements)

Usage:
------
1. Set the VTK_FILE_PATH variable to your file path
2. Run the script

Author: Your Name
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

# =============================================================================
# USER INPUT - SET YOUR VTK FILE PATH HERE
# =============================================================================
FOLDER = os.path.dirname(os.path.abspath(__file__))
VTK_FILE_PATH = os.path.join(FOLDER, "vtk_output/Parts_Beam_Beams_0_1.vtk")  # <-- Change this to your VTK file path

# =============================================================================
# CONFIGURATION - Easy to modify parameters
# =============================================================================

# Scale factors for visualization (adjust these to make diagrams visible)
DEFLECTION_SCALE = 1000      # Scale factor for deflection diagram
MOMENT_SCALE = 0.0005        # Scale factor for bending moment diagram
ROTATION_SCALE = 1000

# Colors for plotting
COLOR_STRUCTURE = 'black'           # Original structure color
COLOR_DEFLECTION = 'blue'           # Deflection diagram color
COLOR_MOMENT = 'red'                # Bending moment diagram color
COLOR_FILL_DEFLECTION = 'lightblue' # Fill color for deflection
COLOR_FILL_MOMENT = 'lightcoral'    # Fill color for moment

# Line widths
LINEWIDTH_STRUCTURE = 2.0
LINEWIDTH_DIAGRAM = 1.5

# Figure settings
FIGURE_DPI = 100
SAVE_FIGURES = True          # Set to True to save figures as PNG
OUTPUT_FOLDER = os.path.join(FOLDER, "plots")     # Folder to save output figures

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
    dict : Dictionary containing all parsed data
        - 'points': numpy array of point coordinates (Nx3)
        - 'cells': list of cell connectivity
        - 'point_data': dict of point-based field data
        - 'cell_data': dict of cell-based field data
    """
    
    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"VTK file not found: {filename}")
    
    print(f"Reading VTK file: {filename}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Initialize data structure
    data = {
        'points': None,
        'cells': None,
        'cell_types': None,
        'point_data': {},
        'cell_data': {}
    }
    
    i = 0
    current_section = None  # 'POINT_DATA' or 'CELL_DATA'
    num_points = 0
    num_cells = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            i += 1
            continue
        
        # -----------------------------------------------------------------
        # Parse POINTS section
        # -----------------------------------------------------------------
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
                    elif len(points) < num_points:
                        # Handle case where coordinates span multiple lines
                        remaining = values[j:]
                        i += 1
                        next_values = lines[i].strip().split()
                        combined = remaining + next_values
                        points.append([
                            float(combined[0]), 
                            float(combined[1]), 
                            float(combined[2])
                        ])
                i += 1
            
            data['points'] = np.array(points[:num_points])
            print(f"  - Loaded {num_points} points")
            continue
        
        # -----------------------------------------------------------------
        # Parse CELLS section
        # -----------------------------------------------------------------
        elif line.startswith('CELLS'):
            parts = line.split()
            num_cells = int(parts[1])
            cells = []
            i += 1
            
            for _ in range(num_cells):
                if i >= len(lines):
                    break
                values = lines[i].strip().split()
                # First value is count, rest are point indices
                cell_points = [int(v) for v in values[1:]]
                cells.append(cell_points)
                i += 1
            
            data['cells'] = cells
            print(f"  - Loaded {num_cells} cells")
            continue
        
        # -----------------------------------------------------------------
        # Parse CELL_TYPES section
        # -----------------------------------------------------------------
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
        
        # -----------------------------------------------------------------
        # Track data section (POINT_DATA or CELL_DATA)
        # -----------------------------------------------------------------
        elif line.startswith('POINT_DATA'):
            current_section = 'point_data'
            i += 1
            continue
        
        elif line.startswith('CELL_DATA'):
            current_section = 'cell_data'
            i += 1
            continue
        
        # -----------------------------------------------------------------
        # Parse FIELD declaration
        # -----------------------------------------------------------------
        elif line.startswith('FIELD'):
            i += 1
            continue
        
        # -----------------------------------------------------------------
        # Parse field arrays (DISPLACEMENT, MOMENT, etc.)
        # -----------------------------------------------------------------
        elif current_section is not None:
            parts = line.split()
            
            # Check if this is a field header (name, components, tuples, type)
            if len(parts) >= 4:
                try:
                    field_name = parts[0]
                    num_components = int(parts[1])
                    num_tuples = int(parts[2])
                    # parts[3] is the data type (float, int, etc.)
                    
                    field_data = []
                    i += 1
                    
                    while len(field_data) < num_tuples and i < len(lines):
                        values = lines[i].strip().split()
                        
                        # Check if we hit another section
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


# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================

def get_element_direction(p1, p2):
    """
    Get the unit direction vector of an element.
    
    Parameters:
    -----------
    p1, p2 : array-like
        Start and end points of the element
        
    Returns:
    --------
    numpy.ndarray : Unit direction vector
    """
    direction = np.array(p2) - np.array(p1)
    length = np.linalg.norm(direction)
    if length > 0:
        return direction / length
    return np.array([1, 0, 0])


def get_element_length(p1, p2):
    """
    Get the length of an element.
    """
    return np.linalg.norm(np.array(p2) - np.array(p1))


def get_perpendicular_direction(p1, p2):
    """
    Get the perpendicular direction to an element (in 2D plane).
    
    For vertical elements (columns): perpendicular is horizontal (left/right)
    For horizontal elements (beams): perpendicular is vertical (up/down)
    
    Parameters:
    -----------
    p1, p2 : array-like
        Start and end points of the element
        
    Returns:
    --------
    numpy.ndarray : Unit perpendicular vector (rotated 90° counterclockwise)
    """
    direction = get_element_direction(p1, p2)
    # Rotate 90 degrees counterclockwise in 2D: (x, y) -> (-y, x)
    perpendicular = np.array([-direction[1], direction[0], 0])
    return perpendicular


def is_vertical_element(p1, p2, tolerance=0.1):
    """Check if an element is vertical (column)."""
    direction = get_element_direction(p1, p2)
    return abs(direction[1]) > abs(direction[0])


def is_horizontal_element(p1, p2, tolerance=0.1):
    """Check if an element is horizontal (beam)."""
    direction = get_element_direction(p1, p2)
    return abs(direction[0]) > abs(direction[1])


def get_element_type(p1, p2):
    """
    Determine if element is a column, beam, or inclined member.
    
    Returns:
    --------
    str : 'column', 'beam', or 'inclined'
    """
    direction = get_element_direction(p1, p2)
    
    if abs(direction[1]) > 0.9:  # Mostly vertical
        return 'column'
    elif abs(direction[0]) > 0.9:  # Mostly horizontal
        return 'beam'
    else:
        return 'inclined'


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_structure(ax, points, cells, color=COLOR_STRUCTURE, 
                   linewidth=LINEWIDTH_STRUCTURE, label='Structure',
                   show_node_labels=False):
    """
    Plot the frame structure.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    points : numpy.ndarray
        Array of point coordinates
    cells : list
        List of cell connectivity
    color : str
        Line color
    linewidth : float
        Line width
    label : str
        Label for legend
    show_node_labels : bool
        Whether to show node number labels
    """
    
    # Plot elements
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
    
    # Plot nodes
    ax.scatter(points[:, 0], points[:, 1], color=color, s=30, zorder=5)
    
    # Add node labels if requested
    if show_node_labels:
        for i, point in enumerate(points):
            ax.annotate(f'{i}', (point[0], point[1]), 
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=8, color='gray')


def plot_deflection_diagram(ax, points, cells, displacement, 
                            scale=DEFLECTION_SCALE,
                            color=COLOR_DEFLECTION, 
                            fill_color=COLOR_FILL_DEFLECTION,
                            linewidth=LINEWIDTH_DIAGRAM, 
                            show_fill=True,
                            show_connecting_lines=True):
    """
    Plot deflection diagram perpendicular to each element.
    
    For columns (vertical): deflection shown left/right
    For beams (horizontal): deflection shown above/below
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    points : numpy.ndarray
        Array of point coordinates
    cells : list
        List of cell connectivity
    displacement : numpy.ndarray
        Displacement values at each point (Nx3 array)
    scale : float
        Scale factor for visualization
    color : str
        Line color
    fill_color : str
        Fill color
    linewidth : float
        Line width
    show_fill : bool
        Whether to show filled area
    show_connecting_lines : bool
        Whether to show dashed lines connecting structure to diagram
    """
    
    for cell in cells:
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = points[p1_idx][:2]  # Only x, y
        p2 = points[p2_idx][:2]
        
        # Get perpendicular direction for this element
        perp = get_perpendicular_direction(p1, p2)[:2]
        
        # Get displacement at nodes
        disp1 = displacement[p1_idx][:2]
        disp2 = displacement[p2_idx][:2]
        
        # Project displacement onto perpendicular direction
        # This gives the component of displacement perpendicular to the element
        disp1_perp = np.dot(disp1, perp)
        disp2_perp = np.dot(disp2, perp)
        
        # Calculate offset points for the diagram
        offset_p1 = p1 + perp * disp1_perp * scale
        offset_p2 = p2 + perp * disp2_perp * scale
        
        # Plot deflected shape line
        ax.plot([offset_p1[0], offset_p2[0]], [offset_p1[1], offset_p2[1]], 
               color=color, linewidth=linewidth)
        
        # Plot connecting lines from structure to deflection diagram
        if show_connecting_lines:
            ax.plot([p1[0], offset_p1[0]], [p1[1], offset_p1[1]], 
                   color=color, linewidth=0.5, linestyle='--', alpha=0.7)
            ax.plot([p2[0], offset_p2[0]], [p2[1], offset_p2[1]], 
                   color=color, linewidth=0.5, linestyle='--', alpha=0.7)
        
        # Fill area between structure and diagram
        if show_fill:
            polygon_x = [p1[0], offset_p1[0], offset_p2[0], p2[0]]
            polygon_y = [p1[1], offset_p1[1], offset_p2[1], p2[1]]
            ax.fill(polygon_x, polygon_y, color=fill_color, alpha=0.3)


def plot_moment_diagram(ax, points, cells, moment_data, 
                        scale=MOMENT_SCALE,
                        color=COLOR_MOMENT, 
                        fill_color=COLOR_FILL_MOMENT,
                        linewidth=LINEWIDTH_DIAGRAM, 
                        show_fill=True,
                        show_values=True, 
                        use_cell_data=True,
                        show_connecting_lines=True):
    """
    Plot bending moment diagram perpendicular to each element.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    points : numpy.ndarray
        Array of point coordinates
    cells : list
        List of cell connectivity
    moment_data : numpy.ndarray
        Moment values (point data or cell data)
    scale : float
        Scale factor for visualization
    color : str
        Line color
    fill_color : str
        Fill color
    linewidth : float
        Line width
    show_fill : bool
        Whether to show filled area
    show_values : bool
        Whether to annotate moment values
    use_cell_data : bool
        If True, use cell-based moment data; if False, use point-based
    show_connecting_lines : bool
        Whether to show lines connecting structure to diagram
    """
    
    for idx, cell in enumerate(cells):
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = points[p1_idx][:2]
        p2 = points[p2_idx][:2]
        
        # Get perpendicular direction
        perp = get_perpendicular_direction(p1, p2)[:2]
        
        if use_cell_data:
            # Use cell moment (single value per element)
            # Moment is typically in Z component for 2D frame analysis
            moment_value = moment_data[idx][2]  # Z component
            
            # For a constant moment along element, offset both ends equally
            offset_p1 = p1 + perp * moment_value * scale
            offset_p2 = p2 + perp * moment_value * scale
            
            # Annotate moment value at midpoint
            if show_values and abs(moment_value) > 1:
                mid_offset = (offset_p1 + offset_p2) / 2
                # Offset text slightly for readability
                text_offset = perp * moment_value * scale * 0.3
                ax.annotate(f'{moment_value:.0f}', 
                           xy=(mid_offset[0], mid_offset[1]),
                           fontsize=7, color=color,
                           ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                    facecolor='white', alpha=0.7, 
                                    edgecolor='none'))
        else:
            # Use point moment data (varies along element)
            moment1 = moment_data[p1_idx][2]  # Z component at node 1
            moment2 = moment_data[p2_idx][2]  # Z component at node 2
            
            offset_p1 = p1 + perp * moment1 * scale
            offset_p2 = p2 + perp * moment2 * scale
        
        # Plot moment diagram line
        ax.plot([offset_p1[0], offset_p2[0]], [offset_p1[1], offset_p2[1]], 
               color=color, linewidth=linewidth)
        
        # Plot connecting lines from structure to moment diagram
        if show_connecting_lines:
            ax.plot([p1[0], offset_p1[0]], [p1[1], offset_p1[1]], 
                   color=color, linewidth=0.5, linestyle='-', alpha=0.7)
            ax.plot([p2[0], offset_p2[0]], [p2[1], offset_p2[1]], 
                   color=color, linewidth=0.5, linestyle='-', alpha=0.7)
        
        # Fill area between structure and diagram
        if show_fill:
            polygon_x = [p1[0], offset_p1[0], offset_p2[0], p2[0]]
            polygon_y = [p1[1], offset_p1[1], offset_p2[1], p2[1]]
            ax.fill(polygon_x, polygon_y, color=fill_color, alpha=0.3)

def plot_rotation_diagram(ax, points, cells, rotation_data, 
                          scale=ROTATION_SCALE,  # Adjust scale as needed
                          color='green', 
                          fill_color='lightgreen',
                          linewidth=1.5, 
                          show_fill=True,
                          show_values=True,
                          show_connecting_lines=True):
    """
    Plot rotation diagram perpendicular to each element.
    
    Parameters:
    -----------
    rotation_data : numpy.ndarray
        Rotation values at each point (Nx3 array, Z component used for 2D)
    """
    
    for idx, cell in enumerate(cells):
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = points[p1_idx][:2]
        p2 = points[p2_idx][:2]
        
        # Get perpendicular direction
        perp = get_perpendicular_direction(p1, p2)[:2]
        
        # Get rotation at nodes (Z component for 2D rotation)
        rot1 = rotation_data[p1_idx][2]  # Rotation about Z-axis
        rot2 = rotation_data[p2_idx][2]
        
        # Calculate offset points
        offset_p1 = p1 + perp * rot1 * scale
        offset_p2 = p2 + perp * rot2 * scale
        
        # Plot rotation diagram line
        ax.plot([offset_p1[0], offset_p2[0]], [offset_p1[1], offset_p2[1]], 
               color=color, linewidth=linewidth)
        
        # Plot connecting lines
        if show_connecting_lines:
            ax.plot([p1[0], offset_p1[0]], [p1[1], offset_p1[1]], 
                   color=color, linewidth=0.5, linestyle='--', alpha=0.7)
            ax.plot([p2[0], offset_p2[0]], [p2[1], offset_p2[1]], 
                   color=color, linewidth=0.5, linestyle='--', alpha=0.7)
        
        # Fill area
        if show_fill:
            polygon_x = [p1[0], offset_p1[0], offset_p2[0], p2[0]]
            polygon_y = [p1[1], offset_p1[1], offset_p2[1], p2[1]]
            ax.fill(polygon_x, polygon_y, color=fill_color, alpha=0.3)
        
        # Show values at nodes
        if show_values:
            if abs(rot1) > 1e-6:
                ax.annotate(f'{rot1:.4f}', (offset_p1[0], offset_p1[1]),
                           fontsize=8, color=color, ha='center')
            if abs(rot2) > 1e-6:
                ax.annotate(f'{rot2:.4f}', (offset_p2[0], offset_p2[1]),
                           fontsize=8, color=color, ha='center')
 

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
# MAIN PLOTTING FUNCTION
# =============================================================================

def create_frame_plots(vtk_data, save_figures=SAVE_FIGURES, 
                       output_folder=OUTPUT_FOLDER):
    """
    Create all plots for the frame structure.
    
    Parameters:
    -----------
    vtk_data : dict
        Parsed VTK data
    save_figures : bool
        Whether to save figures to files
    output_folder : str
        Folder to save output figures
    """
    
    # Extract data
    points = vtk_data['points']
    cells = vtk_data['cells']
    
    # Get field data with defaults
    displacement = vtk_data['point_data'].get('DISPLACEMENT', 
                                               np.zeros((len(points), 3)))
    point_moment = vtk_data['point_data'].get('MOMENT', 
                                               np.zeros((len(points), 3)))
    cell_moment = vtk_data['cell_data'].get('MOMENT', 
                                             np.zeros((len(cells), 3)))
    
    # Create output folder if saving figures
    if save_figures and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"\nCreated output folder: {output_folder}")
    
    # =========================================================================
    # PLOT 1: Combined view (Structure + Deflection + Moment)
    # =========================================================================
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), dpi=FIGURE_DPI)
    
    # --- Subplot 1: Structure Only ---
    ax1 = axes[0]
    ax1.set_title('Frame Structure', fontsize=14, fontweight='bold')
    plot_structure(ax1, points, cells, show_node_labels=True)
    add_supports(ax1, points)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # --- Subplot 2: Structure + Deflection ---
    ax2 = axes[1]
    ax2.set_title(f'Deflection Diagram\n(Scale: {DEFLECTION_SCALE}x)', 
                  fontsize=14, fontweight='bold')
    plot_structure(ax2, points, cells, color='gray', linewidth=1.5, 
                  label='Structure')
    plot_deflection_diagram(ax2, points, cells, displacement, 
                           scale=DEFLECTION_SCALE, show_fill=True)
    add_supports(ax2, points)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Custom legend
    struct_line = plt.Line2D([0], [0], color='gray', linewidth=1.5)
    defl_line = plt.Line2D([0], [0], color=COLOR_DEFLECTION, linewidth=1.5)
    ax2.legend([struct_line, defl_line], ['Structure', 'Deflection'], 
              loc='upper right')
    
    # --- Subplot 3: Structure + Bending Moment ---
    ax3 = axes[2]
    ax3.set_title(f'Bending Moment Diagram\n(Scale: {MOMENT_SCALE}x)', 
                  fontsize=14, fontweight='bold')
    plot_structure(ax3, points, cells, color='gray', linewidth=1.5, 
                  label='Structure')
    plot_moment_diagram(ax3, points, cells, cell_moment,
                       scale=MOMENT_SCALE, show_fill=True,
                       show_values=True, use_cell_data=True)
    add_supports(ax3, points)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Custom legend
    moment_line = plt.Line2D([0], [0], color=COLOR_MOMENT, linewidth=1.5)
    ax3.legend([struct_line, moment_line], ['Structure', 'Bending Moment'], 
              loc='upper right')
    
    plt.tight_layout()
    
    if save_figures:
        filepath = os.path.join(output_folder, 'combined_plots.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    
    # =========================================================================
    # PLOT 2: Detailed Deflection Diagram
    # =========================================================================
    
    fig2, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
    ax.set_title('Deflection Diagram', fontsize=14, fontweight='bold')
    plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    plot_deflection_diagram(ax, points, cells, displacement, 
                           scale=DEFLECTION_SCALE, show_fill=True)
    add_supports(ax, points)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add info text
    max_disp = np.max(np.abs(displacement[:, :2]))
    ax.text(0.35, 0.2, f'Max displacement: {max_disp:.6f} m\nScale factor: {DEFLECTION_SCALE}x',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_figures:
        filepath = os.path.join(output_folder, 'deflection_diagram.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    
    # =========================================================================
    # PLOT 3: Detailed Bending Moment Diagram
    # =========================================================================
    
    fig3, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
    ax.set_title('Bending Moment Diagram (Detailed)', fontsize=14, fontweight='bold')
    plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    plot_moment_diagram(ax, points, cells, cell_moment,
                       scale=MOMENT_SCALE, show_fill=True,
                       show_values=True, use_cell_data=True)
    add_supports(ax, points)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add info text
    max_moment = np.max(np.abs(cell_moment[:, 2]))
    min_moment = np.min(cell_moment[:, 2])
    max_moment_val = np.max(cell_moment[:, 2])
    ax.text(0.4, 0.2, f'Max moment: {max_moment_val:.1f}\nMin moment: {min_moment:.1f}\nScale factor: {MOMENT_SCALE}x',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_figures:
        filepath = os.path.join(output_folder, 'moment_diagram.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()

    # =========================================================================
    # PLOT 4: Detailed Rotation Diagram
    # =========================================================================
    # Extract rotation data from VTK
    rotation = vtk_data['point_data'].get('ROTATION', np.zeros((len(points), 3)))


    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
    ax.set_title(f'Rotation Diagram (Detailed)\n(Scale: {ROTATION_SCALE}x)', fontsize=14, fontweight='bold')
    plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    add_supports(ax, points)
    plot_rotation_diagram(ax, points, cells, rotation, scale=1000.0)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if save_figures:
        filepath = os.path.join(output_folder, 'rotation_diagram.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")

    plt.show()
def print_summary(vtk_data):
    """Print a summary of the VTK data."""
    
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    points = vtk_data['points']
    cells = vtk_data['cells']
    
    print(f"\nGeometry:")
    print(f"  - Number of nodes: {len(points)}")
    print(f"  - Number of elements: {len(cells)}")
    print(f"  - X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  - Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    
    print(f"\nPoint Data Fields:")
    for name, data in vtk_data['point_data'].items():
        print(f"  - {name}: shape {data.shape}")
    
    print(f"\nCell Data Fields:")
    for name, data in vtk_data['cell_data'].items():
        print(f"  - {name}: shape {data.shape}")
    
    # Displacement summary
    if 'DISPLACEMENT' in vtk_data['point_data']:
        disp = vtk_data['point_data']['DISPLACEMENT']
        print(f"\nDisplacement Summary:")
        print(f"  - Max X displacement: {np.max(np.abs(disp[:, 0])):.6e}")
        print(f"  - Max Y displacement: {np.max(np.abs(disp[:, 1])):.6e}")
    
    # Moment summary
    if 'MOMENT' in vtk_data['cell_data']:
        moment = vtk_data['cell_data']['MOMENT']
        print(f"\nBending Moment Summary (Cell Data):")
        print(f"  - Max moment: {np.max(moment[:, 2]):.2f}")
        print(f"  - Min moment: {np.min(moment[:, 2]):.2f}")
    
    print("=" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("VTK FRAME STRUCTURE PLOTTER")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Option 1: Use command line argument
    # -------------------------------------------------------------------------
    if len(sys.argv) > 1:
        vtk_file = sys.argv[1]
    else:
        vtk_file = VTK_FILE_PATH
    
    # -------------------------------------------------------------------------
    # Parse the VTK file
    # -------------------------------------------------------------------------
    try:
        print(f"\n1. Loading VTK file: {vtk_file}")
        vtk_data = parse_vtk_file(vtk_file)
        
        # Print summary
        # print_summary(vtk_data)
        
        # Create plots
        print("\n2. Creating plots...")
        create_frame_plots(vtk_data, save_figures=SAVE_FIGURES, 
                          output_folder=OUTPUT_FOLDER)
        
        print("\nDone!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease check the file path and try again.")
        print("You can either:")
        print("  1. Set VTK_FILE_PATH at the top of the script")
        print("  2. Run: python script.py your_file.vtk")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError parsing VTK file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)