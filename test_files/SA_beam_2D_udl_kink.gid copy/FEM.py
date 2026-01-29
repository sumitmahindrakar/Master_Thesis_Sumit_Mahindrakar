"""
VTK Frame Structure Plotter
===========================
This script reads a VTK file containing frame structure data and plots:
1. The original frame structure
2. Deflection diagram (perpendicular to elements)
3. Bending moment diagram (perpendicular to elements)

Usage:
------
1. Set the VTK_FILE_PATH and MDPA_FILE_PATH variables to your file paths
2. Run the script

Author: Your Name
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import re

# =============================================================================
# USER INPUT - SET YOUR FILE PATHS HERE
# =============================================================================

VTK_FILE_PATH = "test_files/SA_beam_2D_udl.gid/vtk_output/Parts_Beam_Beams_0_1.vtk"
MDPA_FILE_PATH = "test_files/SA_beam_2D_udl.gid/SA_beam_2D_udl_refined.mdpa"  # <-- MDPA file for support info

# =============================================================================
# CONFIGURATION - Easy to modify parameters
# =============================================================================

# Scale factors for visualization (adjust these to make diagrams visible)
DEFLECTION_SCALE = 1000      # Scale factor for deflection diagram
MOMENT_SCALE = 0.0005        # Scale factor for bending moment diagram

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
OUTPUT_FOLDER = "test_files/SA_beam_2D_udl.gid/plots"     # Folder to save output figures

# =============================================================================
# MDPA FILE PARSER
# =============================================================================

def parse_mdpa_file(filename):
    """
    Parse an MDPA file and extract nodes and support information.
    
    Parameters:
    -----------
    filename : str
        Path to the MDPA file
        
    Returns:
    --------
    dict : Dictionary containing parsed data
        - 'nodes': dict of node_id -> (x, y, z)
        - 'elements': list of element connectivity
        - 'supports': dict with support information
            - 'fixed': list of node IDs with fixed supports
            - 'pinned': list of node IDs with pinned supports
            - 'roller_x': list of node IDs with roller supports (free in X)
            - 'roller_y': list of node IDs with roller supports (free in Y)
            - 'displacement_x': set of node IDs with X displacement constraint
            - 'displacement_y': set of node IDs with Y displacement constraint
            - 'rotation': set of node IDs with rotation constraint
    """
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"MDPA file not found: {filename}")
    
    print(f"Reading MDPA file: {filename}")
    
    with open(filename, 'r') as f:
        content = f.read()
    
    data = {
        'nodes': {},
        'elements': [],
        'supports': {
            'displacement_x': set(),
            'displacement_y': set(),
            'displacement_z': set(),
            'rotation': set(),
            'rotation_x': set(),
            'rotation_y': set(),
            'rotation_z': set(),
        }
    }
    
    # Parse Nodes section
    nodes_match = re.search(r'Begin Nodes\s*\n(.*?)End Nodes', content, re.DOTALL)
    if nodes_match:
        nodes_text = nodes_match.group(1)
        for line in nodes_text.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('//'):
                parts = line.split()
                if len(parts) >= 4:
                    node_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    data['nodes'][node_id] = (x, y, z)
        print(f"  - Loaded {len(data['nodes'])} nodes")
    
    # Parse Elements section
    elements_match = re.search(r'Begin Elements\s+\w+\s*//.*?\n(.*?)End Elements', content, re.DOTALL)
    if elements_match:
        elements_text = elements_match.group(1)
        for line in elements_text.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('//'):
                parts = line.split()
                if len(parts) >= 4:
                    elem_id = int(parts[0])
                    prop_id = int(parts[1])
                    node1 = int(parts[2])
                    node2 = int(parts[3])
                    data['elements'].append({
                        'id': elem_id,
                        'property': prop_id,
                        'nodes': [node1, node2]
                    })
        print(f"  - Loaded {len(data['elements'])} elements")
    
    # Parse SubModelPart sections for boundary conditions
    submodelpart_pattern = r'Begin SubModelPart\s+(\w+)\s*\n(.*?)End SubModelPart'
    submodelpart_matches = re.findall(submodelpart_pattern, content, re.DOTALL)
    
    for name, body in submodelpart_matches:
        name_upper = name.upper()
        
        # Extract nodes from this SubModelPart
        nodes_in_part = []
        nodes_section = re.search(r'Begin SubModelPartNodes\s*\n(.*?)End SubModelPartNodes', body, re.DOTALL)
        if nodes_section:
            for line in nodes_section.group(1).strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('//'):
                    try:
                        node_id = int(line)
                        nodes_in_part.append(node_id)
                    except ValueError:
                        pass
        
        # Categorize based on SubModelPart name
        if 'DISPLACEMENT' in name_upper:
            if '_X' in name_upper or 'NODE1_X' in name_upper:
                data['supports']['displacement_x'].update(nodes_in_part)
                print(f"  - Found X displacement constraint at nodes: {nodes_in_part}")
            elif '_Y' in name_upper:
                data['supports']['displacement_y'].update(nodes_in_part)
                print(f"  - Found Y displacement constraint at nodes: {nodes_in_part}")
            elif '_Z' in name_upper:
                data['supports']['displacement_z'].update(nodes_in_part)
                print(f"  - Found Z displacement constraint at nodes: {nodes_in_part}")
            else:
                # Generic displacement - assume all directions
                data['supports']['displacement_x'].update(nodes_in_part)
                data['supports']['displacement_y'].update(nodes_in_part)
                data['supports']['displacement_z'].update(nodes_in_part)
                print(f"  - Found displacement constraint (all) at nodes: {nodes_in_part}")
        
        if 'ROTATION' in name_upper:
            if '_X' in name_upper:
                data['supports']['rotation_x'].update(nodes_in_part)
            elif '_Y' in name_upper:
                data['supports']['rotation_y'].update(nodes_in_part)
            elif '_Z' in name_upper:
                data['supports']['rotation_z'].update(nodes_in_part)
            else:
                # Generic rotation constraint
                data['supports']['rotation'].update(nodes_in_part)
            print(f"  - Found rotation constraint at nodes: {nodes_in_part}")
    
    # Determine support types based on constraints
    data['support_types'] = determine_support_types(data['supports'])
    
    return data


def determine_support_types(supports):
    """
    Determine the type of support at each node based on constraints.
    
    Returns:
    --------
    dict : Dictionary with node_id -> support_type
        Support types: 'fixed', 'pinned', 'roller_x', 'roller_y', 'hinge'
    """
    
    support_types = {}
    
    # Get all constrained nodes
    all_constrained_nodes = (
        supports['displacement_x'] | 
        supports['displacement_y'] | 
        supports['displacement_z'] |
        supports['rotation'] |
        supports['rotation_x'] |
        supports['rotation_y'] |
        supports['rotation_z']
    )
    
    for node_id in all_constrained_nodes:
        has_disp_x = node_id in supports['displacement_x']
        has_disp_y = node_id in supports['displacement_y']
        has_rotation = (node_id in supports['rotation'] or 
                       node_id in supports['rotation_z'])  # For 2D, Z rotation is the one
        
        if has_disp_x and has_disp_y and has_rotation:
            support_types[node_id] = 'fixed'
        elif has_disp_x and has_disp_y and not has_rotation:
            support_types[node_id] = 'pinned'
        elif has_disp_y and not has_disp_x:
            if has_rotation:
                support_types[node_id] = 'roller_y_fixed_rot'  # Roller in Y with fixed rotation
            else:
                support_types[node_id] = 'roller_y'  # Pure roller (free in X)
        elif has_disp_x and not has_disp_y:
            support_types[node_id] = 'roller_x'  # Roller (free in Y)
        elif has_rotation and not has_disp_x and not has_disp_y:
            support_types[node_id] = 'rotation_only'
        else:
            support_types[node_id] = 'partial'
    
    print(f"\n  Support types identified:")
    for node_id, stype in sorted(support_types.items()):
        print(f"    Node {node_id}: {stype}")
    
    return support_types


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
    """
    direction = get_element_direction(p1, p2)
    # Rotate 90 degrees counterclockwise in 2D: (x, y) -> (-y, x)
    perpendicular = np.array([-direction[1], direction[0], 0])
    return perpendicular


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_structure(ax, points, cells, color=COLOR_STRUCTURE, 
                   linewidth=LINEWIDTH_STRUCTURE, label='Structure',
                   show_node_labels=False):
    """
    Plot the frame structure.
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
    """
    
    for idx, cell in enumerate(cells):
        p1_idx, p2_idx = cell[0], cell[1]
        p1 = points[p1_idx][:2]
        p2 = points[p2_idx][:2]
        
        # Get perpendicular direction
        perp = get_perpendicular_direction(p1, p2)[:2]
        
        if use_cell_data:
            # Use cell moment (single value per element)
            moment_value = moment_data[idx][2]  # Z component
            
            offset_p1 = p1 + perp * moment_value * scale
            offset_p2 = p2 + perp * moment_value * scale
            
            # Annotate moment value at midpoint
            if show_values and abs(moment_value) > 1:
                mid_offset = (offset_p1 + offset_p2) / 2
                ax.annotate(f'{moment_value:.0f}', 
                           xy=(mid_offset[0], mid_offset[1]),
                           fontsize=7, color=color,
                           ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                    facecolor='white', alpha=0.7, 
                                    edgecolor='none'))
        else:
            # Use point moment data (varies along element)
            moment1 = moment_data[p1_idx][2]
            moment2 = moment_data[p2_idx][2]
            
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


def add_supports(ax, points, mdpa_data, vtk_to_mdpa_mapping=None):
    """
    Add support symbols at constrained nodes based on MDPA data.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    points : numpy.ndarray
        Array of point coordinates from VTK
    mdpa_data : dict
        Parsed MDPA data containing support information
    vtk_to_mdpa_mapping : dict
        Mapping from VTK point index to MDPA node ID (if None, auto-detect)
    """
    
    support_types = mdpa_data.get('support_types', {})
    mdpa_nodes = mdpa_data.get('nodes', {})
    
    if not support_types:
        print("  Warning: No support information found in MDPA file")
        return
    
    # Create mapping from VTK indices to MDPA node IDs based on coordinates
    if vtk_to_mdpa_mapping is None:
        vtk_to_mdpa_mapping = {}
        mdpa_to_vtk_mapping = {}
        
        for vtk_idx, vtk_point in enumerate(points):
            for mdpa_id, mdpa_coords in mdpa_nodes.items():
                # Check if coordinates match (within tolerance)
                if (abs(vtk_point[0] - mdpa_coords[0]) < 1e-6 and
                    abs(vtk_point[1] - mdpa_coords[1]) < 1e-6 and
                    abs(vtk_point[2] - mdpa_coords[2]) < 1e-6):
                    vtk_to_mdpa_mapping[vtk_idx] = mdpa_id
                    mdpa_to_vtk_mapping[mdpa_id] = vtk_idx
                    break
    
    # Symbol size (relative to structure size)
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    size = max(x_range, y_range) * 0.03
    if size == 0:
        size = 0.15
    
    for mdpa_node_id, support_type in support_types.items():
        # Find VTK index for this MDPA node
        vtk_idx = None
        for vidx, mid in vtk_to_mdpa_mapping.items():
            if mid == mdpa_node_id:
                vtk_idx = vidx
                break
        
        if vtk_idx is None:
            # Try direct mapping (if VTK index = MDPA ID - 1)
            if mdpa_node_id - 1 < len(points):
                vtk_idx = mdpa_node_id - 1
            else:
                continue
        
        x, y = points[vtk_idx][0], points[vtk_idx][1]
        
        if support_type == 'fixed':
            # Fixed support - filled rectangle with hatching
            draw_fixed_support(ax, x, y, size)
        elif support_type == 'pinned':
            # Pinned support - triangle
            draw_pinned_support(ax, x, y, size)
        elif support_type in ['roller_y', 'roller_y_fixed_rot']:
            # Roller support (can move in X) - triangle with circle
            draw_roller_support(ax, x, y, size, direction='horizontal')
        elif support_type == 'roller_x':
            # Roller support (can move in Y) - triangle with circle, rotated
            draw_roller_support(ax, x, y, size, direction='vertical')
        else:
            # Generic partial constraint - small square
            draw_partial_support(ax, x, y, size)


def draw_fixed_support(ax, x, y, size):
    """Draw a fixed support symbol (rectangle with hatching)."""
    
    # Draw ground rectangle
    rect = plt.Rectangle((x - size, y - size * 0.5), size * 2, size * 0.5,
                         fill=True, facecolor='lightgray', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    
    # Draw hatching lines
    for hx in np.linspace(x - size * 0.8, x + size * 0.8, 5):
        ax.plot([hx, hx - size * 0.3], [y - size * 0.5, y - size * 0.8], 
               'k-', linewidth=0.8)


def draw_pinned_support(ax, x, y, size):
    """Draw a pinned support symbol (triangle)."""
    
    # Triangle
    triangle = plt.Polygon([
        [x, y],
        [x - size, y - size],
        [x + size, y - size]
    ], fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(triangle)
    
    # Ground line
    ax.plot([x - size * 1.2, x + size * 1.2], [y - size, y - size], 
           'k-', linewidth=1.5)
    
    # Hatching lines
    for hx in np.linspace(x - size, x + size, 4):
        ax.plot([hx, hx - size * 0.3], [y - size, y - size * 1.3], 
               'k-', linewidth=0.8)


def draw_roller_support(ax, x, y, size, direction='horizontal'):
    """Draw a roller support symbol (triangle with circle)."""
    
    # Triangle
    triangle = plt.Polygon([
        [x, y],
        [x - size, y - size * 0.7],
        [x + size, y - size * 0.7]
    ], fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(triangle)
    
    # Roller circle
    circle = plt.Circle((x, y - size * 0.85), size * 0.15, 
                        fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    
    # Ground line
    ax.plot([x - size * 1.2, x + size * 1.2], [y - size, y - size], 
           'k-', linewidth=1.5)
    
    # Hatching lines
    for hx in np.linspace(x - size, x + size, 4):
        ax.plot([hx, hx - size * 0.3], [y - size, y - size * 1.3], 
               'k-', linewidth=0.8)


def draw_partial_support(ax, x, y, size):
    """Draw a partial constraint symbol (small square)."""
    
    rect = plt.Rectangle((x - size * 0.3, y - size * 0.3), size * 0.6, size * 0.6,
                         fill=True, facecolor='yellow', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)


# =============================================================================
# MAIN PLOTTING FUNCTION
# =============================================================================

def create_frame_plots(vtk_data, mdpa_data, save_figures=SAVE_FIGURES, 
                       output_folder=OUTPUT_FOLDER):
    """
    Create all plots for the frame structure.
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
    add_supports(ax1, points, mdpa_data)
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
    add_supports(ax2, points, mdpa_data)
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
    add_supports(ax3, points, mdpa_data)
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
    ax.set_title('Deflection Diagram (Detailed)', fontsize=14, fontweight='bold')
    plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    plot_deflection_diagram(ax, points, cells, displacement, 
                           scale=DEFLECTION_SCALE, show_fill=True)
    add_supports(ax, points, mdpa_data)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add info text
    max_disp = np.max(np.abs(displacement[:, :2]))
    ax.text(0.02, 0.98, f'Max displacement: {max_disp:.6f} m\nScale factor: {DEFLECTION_SCALE}x',
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
    add_supports(ax, points, mdpa_data)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add info text
    max_moment = np.max(np.abs(cell_moment[:, 2]))
    min_moment = np.min(cell_moment[:, 2])
    max_moment_val = np.max(cell_moment[:, 2])
    ax.text(0.02, 0.98, f'Max moment: {max_moment_val:.1f}\nMin moment: {min_moment:.1f}\nScale factor: {MOMENT_SCALE}x',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_figures:
        filepath = os.path.join(output_folder, 'moment_diagram.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()


def print_summary(vtk_data, mdpa_data):
    """Print a summary of the data."""
    
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    points = vtk_data['points']
    cells = vtk_data['cells']
    
    print(f"\nGeometry (from VTK):")
    print(f"  - Number of nodes: {len(points)}")
    print(f"  - Number of elements: {len(cells)}")
    print(f"  - X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  - Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    
    print(f"\nSupport Information (from MDPA):")
    support_types = mdpa_data.get('support_types', {})
    for node_id, stype in sorted(support_types.items()):
        print(f"  - Node {node_id}: {stype}")
    
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
    # Handle command line arguments
    # -------------------------------------------------------------------------
    if len(sys.argv) > 2:
        vtk_file = sys.argv[1]
        mdpa_file = sys.argv[2]
    elif len(sys.argv) > 1:
        vtk_file = sys.argv[1]
        mdpa_file = MDPA_FILE_PATH
    else:
        vtk_file = VTK_FILE_PATH
        mdpa_file = MDPA_FILE_PATH
    
    # -------------------------------------------------------------------------
    # Parse the files
    # -------------------------------------------------------------------------
    try:
        print(f"\n1. Loading MDPA file: {mdpa_file}")
        mdpa_data = parse_mdpa_file(mdpa_file)
        
        print(f"\n2. Loading VTK file: {vtk_file}")
        vtk_data = parse_vtk_file(vtk_file)
        
        # Print summary
        print_summary(vtk_data, mdpa_data)
        
        # Create plots
        print("\n3. Creating plots...")
        create_frame_plots(vtk_data, mdpa_data, save_figures=SAVE_FIGURES, 
                          output_folder=OUTPUT_FOLDER)
        
        print("\nDone!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease check the file paths and try again.")
        print("You can either:")
        print("  1. Set VTK_FILE_PATH and MDPA_FILE_PATH at the top of the script")
        print("  2. Run: python script.py vtk_file.vtk mdpa_file.mdpa")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)