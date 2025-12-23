"""
Moment Sensitivity Calculator: ∂M/∂(EI)
Using General Influence Method with VTK File Parsing and Visualization

Description:
    Computes the sensitivity of bending moment at a response location
    with respect to flexural rigidity (EI) of beam elements using the
    adjoint/dual method.
    
Formula:
    ∂M_response/∂(EI)_k = -(1/(EI)²) × ∫ M_k(x) × M̄_k(x) dx
    
Features:
    - Shows only max/min values per member type (columns vs beams)
    - Circles repeated/duplicate nodes on dual and sensitivity plots
"""

import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle

FOLDER = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CONFIGURATION
# =============================================================================

# Scale factors for visualization
PRIMARY_MOMENT_SCALE = 0.0001
DUAL_MOMENT_SCALE = 0.000001
SENSITIVITY_SCALE = 1e3 * 5

# Colors
COLOR_STRUCTURE = 'black'
COLOR_PRIMARY_MOMENT = 'blue'
COLOR_DUAL_MOMENT = 'green'
COLOR_FILL_PRIMARY = 'lightblue'
COLOR_FILL_DUAL = 'lightgreen'
COLOR_FILL_SENSITIVITY_POS = '#90EE90'
COLOR_FILL_SENSITIVITY_NEG = '#FFB6C1'
COLOR_REPEATED_NODE = 'red'  # Color for repeated node circles

# Line widths
LINEWIDTH_STRUCTURE = 2.0
LINEWIDTH_DIAGRAM = 1.5

# Figure settings
FIGURE_DPI = 100
SAVE_FIGURES = True

# Repeated node circle settings
REPEATED_NODE_CIRCLE_RADIUS = 0.12
REPEATED_NODE_CIRCLE_LINEWIDTH = 2.5

OUTPUT_FOLDER = os.path.join(FOLDER, "SA_plots")


# =============================================================================
# JSON PARSER FOR MATERIAL PROPERTIES
# =============================================================================

def load_material_properties(json_path, model_part_name=None):
    """Load material properties (E, I) from StructuralMaterials.json"""
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Material JSON file not found: {json_path}")
    
    print(f"\nLoading material properties from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    properties_list = data.get('properties', [])
    
    if not properties_list:
        raise ValueError("No properties found in JSON file")
    
    selected_prop = None
    
    if model_part_name:
        for prop in properties_list:
            if prop.get('model_part_name') == model_part_name:
                selected_prop = prop
                break
        if not selected_prop:
            print(f"  Warning: model_part_name '{model_part_name}' not found. Using first property.")
            selected_prop = properties_list[0]
    else:
        selected_prop = properties_list[0]
    
    material = selected_prop.get('Material', {})
    variables = material.get('Variables', {})
    
    E = variables.get('YOUNG_MODULUS', None)
    I33 = variables.get('I33', None)
    I22 = variables.get('I22', None)
    
    I = I33 if I33 is not None else I22
    
    if E is None:
        raise ValueError("YOUNG_MODULUS not found in material properties")
    if I is None:
        raise ValueError("I33 or I22 not found in material properties")
    
    result = {
        'E': E,
        'I': I,
        'I33': I33,
        'I22': I22,
        'DENSITY': variables.get('DENSITY', None),
        'POISSON_RATIO': variables.get('POISSON_RATIO', None),
        'CROSS_AREA': variables.get('CROSS_AREA', None),
        'TORSIONAL_INERTIA': variables.get('TORSIONAL_INERTIA', None),
        'model_part_name': selected_prop.get('model_part_name', 'Unknown'),
        'properties_id': selected_prop.get('properties_id', None)
    }
    
    print(f"  Model Part: {result['model_part_name']}")
    print(f"  E (Young's Modulus): {E:.4e} Pa")
    print(f"  I (Second Moment of Area): {I:.4e} m⁴")
    print(f"  EI (Flexural Rigidity): {E*I:.4e} N·m²")
    
    return result


def find_material_json(vtk_path):
    """Try to find StructuralMaterials.json in the same directory or parent directories."""
    
    vtk_dir = os.path.dirname(os.path.abspath(vtk_path))
    
    search_paths = [
        os.path.join(vtk_dir, 'StructuralMaterials.json'),
        os.path.join(vtk_dir, '..', 'StructuralMaterials.json'),
        os.path.join(vtk_dir, '..', '..', 'StructuralMaterials.json'),
    ]
    
    current_dir = vtk_dir
    for _ in range(4):
        check_path = os.path.join(current_dir, 'StructuralMaterials.json')
        if check_path not in search_paths:
            search_paths.append(check_path)
        current_dir = os.path.dirname(current_dir)
    
    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return None


# =============================================================================
# VTK FILE PARSER
# =============================================================================

def parse_vtk_file(filename):
    """Parse a VTK file and extract points, cells, and field data."""
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
        
        if line.startswith('POINTS'):
            parts = line.split()
            num_points = int(parts[1])
            points = []
            i += 1
            
            while len(points) < num_points and i < len(lines):
                values = lines[i].strip().split()
                for j in range(0, len(values), 3):
                    if j + 2 < len(values):
                        points.append([float(values[j]), float(values[j+1]), float(values[j+2])])
                i += 1
            
            data['points'] = np.array(points[:num_points])
            print(f"  - Loaded {num_points} points")
            continue
        
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
            print(f"  - Loaded {num_cells} cells (elements)")
            continue
        
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
                        
                        if values and (values[0] in ['POINT_DATA', 'CELL_DATA', 'FIELD', 'SCALARS', 'VECTORS']):
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


def parse_vtk_cell_moments(vtk_file_path):
    """Parse bending moments (Mz) from VTK file CELL_DATA section."""
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
# REPEATED NODE DETECTION
# =============================================================================

def find_repeated_nodes(points, tolerance=1e-6):
    """
    Find nodes that have the same coordinates (repeated/duplicate nodes).
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of point coordinates (N x 3)
    tolerance : float
        Distance tolerance to consider nodes as duplicates
        
    Returns:
    --------
    dict : {
        'repeated_positions': list of (x, y, z) tuples for unique repeated positions,
        'node_groups': list of lists, each containing node indices at same position,
        'all_repeated_indices': set of all node indices that are repeated
    }
    """
    
    n_points = len(points)
    
    # Dictionary to group nodes by position
    position_to_nodes = {}
    
    for i, point in enumerate(points):
        # Round to tolerance for grouping
        key = tuple(np.round(point / tolerance) * tolerance)
        
        if key not in position_to_nodes:
            position_to_nodes[key] = []
        position_to_nodes[key].append(i)
    
    # Find positions with more than one node
    repeated_positions = []
    node_groups = []
    all_repeated_indices = set()
    
    for pos, indices in position_to_nodes.items():
        if len(indices) > 1:
            repeated_positions.append(pos)
            node_groups.append(indices)
            all_repeated_indices.update(indices)
    
    result = {
        'repeated_positions': repeated_positions,
        'node_groups': node_groups,
        'all_repeated_indices': all_repeated_indices
    }
    
    if repeated_positions:
        print(f"\n  Found {len(repeated_positions)} repeated node location(s):")
        for i, (pos, group) in enumerate(zip(repeated_positions, node_groups)):
            print(f"    Position {i+1}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
            print(f"      Node indices: {group}")
    else:
        print("\n  No repeated nodes found.")
    
    return result


def add_repeated_node_circles(ax, repeated_node_info, 
                               radius=REPEATED_NODE_CIRCLE_RADIUS,
                               color=COLOR_REPEATED_NODE,
                               linewidth=REPEATED_NODE_CIRCLE_LINEWIDTH,
                               label='Response Location'):
    """
    Add circles around repeated node positions on the plot.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    repeated_node_info : dict
        Output from find_repeated_nodes()
    radius : float
        Radius of the circles
    color : str
        Color of the circles
    linewidth : float
        Line width of the circles
    label : str
        Label for legend
    """
    
    positions = repeated_node_info['repeated_positions']
    
    for i, pos in enumerate(positions):
        x, y = pos[0], pos[1]
        
        circle = Circle((x, y), radius, 
                        fill=False, 
                        edgecolor=color, 
                        linewidth=linewidth,
                        linestyle='-',
                        zorder=10)
        ax.add_patch(circle)
        
        # Add label only for first circle (for legend)
        if i == 0 and label:
            # Create a dummy line for legend
            ax.plot([], [], color=color, linewidth=linewidth, 
                   label=label, linestyle='-')


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


def calculate_geometry_from_vtk(vtk_data):
    """Calculate beam geometry from VTK data."""
    
    points = vtk_data['points']
    cells = vtk_data['cells']
    
    n_elements = len(cells)
    
    element_lengths = {}
    total_length = 0.0
    
    for idx, cell in enumerate(cells):
        elem_id = idx + 1
        p1 = points[cell[0]]
        p2 = points[cell[1]]
        length = get_element_length(p1, p2)
        element_lengths[elem_id] = length
        total_length += length
    
    x_coords = points[:, 0]
    beam_span = np.max(x_coords) - np.min(x_coords)
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    
    result = {
        'n_elements': n_elements,
        'n_nodes': len(points),
        'total_length': total_length,
        'beam_span': beam_span,
        'element_lengths': element_lengths,
        'avg_element_length': total_length / n_elements if n_elements > 0 else 0,
        'bounding_box': {
            'x': (x_min, x_max),
            'y': (y_min, y_max),
            'z': (z_min, z_max)
        }
    }
    
    print(f"\nGeometry extracted from VTK:")
    print(f"  Number of elements: {n_elements}")
    print(f"  Number of nodes: {len(points)}")
    print(f"  Total element length: {total_length:.6f} m")
    print(f"  Beam span (x-direction): {beam_span:.6f} m")
    print(f"  Average element length: {result['avg_element_length']:.6f} m")
    
    return result


# =============================================================================
# MEMBER CLASSIFICATION AND FILTERING
# =============================================================================

def classify_members(points, cells, angle_threshold=45.0):
    """
    Classify elements as 'column' (vertical) or 'beam' (horizontal).
    """
    
    classifications = {}
    
    for idx, cell in enumerate(cells):
        elem_id = idx + 1
        p1 = points[cell[0]]
        p2 = points[cell[1]]
        
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        
        if dy < 1e-10:
            angle_from_vertical = 90.0
        elif dx < 1e-10:
            angle_from_vertical = 0.0
        else:
            angle_from_vertical = np.degrees(np.arctan(dx / dy))
        
        if angle_from_vertical < angle_threshold:
            classifications[elem_id] = 'column'
        else:
            classifications[elem_id] = 'beam'
    
    return classifications


def get_extreme_element_ids(moment_data, classifications, use_cell_data=True):
    """
    Get element IDs that have max/min moments for each member type.
    """
    
    column_values = {}
    beam_values = {}
    
    for elem_id, member_type in classifications.items():
        idx = elem_id - 1
        
        if use_cell_data and hasattr(moment_data, '__len__'):
            if idx < len(moment_data):
                if hasattr(moment_data[idx], '__len__'):
                    moment_val = moment_data[idx][2]
                else:
                    moment_val = moment_data[idx]
            else:
                continue
        elif isinstance(moment_data, dict):
            if elem_id in moment_data:
                moment_val = moment_data[elem_id]
            else:
                continue
        else:
            continue
        
        if member_type == 'column':
            column_values[elem_id] = moment_val
        else:
            beam_values[elem_id] = moment_val
    
    show_elements = set()
    tolerance = 1e-6
    
    if column_values:
        max_val = max(column_values.values())
        min_val = min(column_values.values())
        for elem_id, val in column_values.items():
            if abs(val - max_val) < tolerance * max(abs(max_val), 1):
                show_elements.add(elem_id)
            if abs(val - min_val) < tolerance * max(abs(min_val), 1):
                show_elements.add(elem_id)
    
    if beam_values:
        max_val = max(beam_values.values())
        min_val = min(beam_values.values())
        for elem_id, val in beam_values.items():
            if abs(val - max_val) < tolerance * max(abs(max_val), 1):
                show_elements.add(elem_id)
            if abs(val - min_val) < tolerance * max(abs(min_val), 1):
                show_elements.add(elem_id)
    
    return show_elements


def get_extreme_sensitivity_ids(sensitivities, classifications):
    """
    Get element IDs that have max/min sensitivities for each member type.
    """
    
    column_values = {}
    beam_values = {}
    
    for elem_id, data in sensitivities.items():
        if elem_id not in classifications:
            continue
        
        sens_val = data['dM_dEI']
        
        if classifications[elem_id] == 'column':
            column_values[elem_id] = sens_val
        else:
            beam_values[elem_id] = sens_val
    
    show_elements = set()
    tolerance = 1e-10
    
    if column_values:
        max_val = max(column_values.values())
        min_val = min(column_values.values())
        for elem_id, val in column_values.items():
            if abs(val - max_val) < tolerance * max(abs(max_val), 1e-20):
                show_elements.add(elem_id)
            if abs(val - min_val) < tolerance * max(abs(min_val), 1e-20):
                show_elements.add(elem_id)
    
    if beam_values:
        max_val = max(beam_values.values())
        min_val = min(beam_values.values())
        for elem_id, val in beam_values.items():
            if abs(val - max_val) < tolerance * max(abs(max_val), 1e-20):
                show_elements.add(elem_id)
            if abs(val - min_val) < tolerance * max(abs(min_val), 1e-20):
                show_elements.add(elem_id)
    
    return show_elements


# =============================================================================
# PLOTTING FUNCTIONS
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
        
        if show_element_labels:
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            ax.annotate(f'E{idx+1}', (mid_x, mid_y), 
                       textcoords="offset points", xytext=(0, -15),
                       fontsize=8, color='gray', ha='center')
    
    ax.scatter(points[:, 0], points[:, 1], color=color, s=30, zorder=5)
    
    if show_node_labels:
        for i, point in enumerate(points):
            ax.annotate(f'{i}', (point[0], point[1]), 
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=8, color='gray')


def add_supports(ax, points, support_indices=None):
    """Add support symbols at fixed nodes."""
    
    if support_indices is None:
        min_y = np.min(points[:, 1])
        support_indices = np.where(np.abs(points[:, 1] - min_y) < 0.01)[0]
    
    size = 0.15
    
    for idx in support_indices:
        x, y = points[idx][0], points[idx][1]
        
        triangle = plt.Polygon([
            [x, y],
            [x - size, y - size],
            [x + size, y - size]
        ], fill=False, edgecolor='black', linewidth=1.5)
        ax.add_patch(triangle)
        
        ax.plot([x - size*1.2, x + size*1.2], [y - size, y - size], 
               'k-', linewidth=1.5)
        
        for hx in np.linspace(x - size, x + size, 4):
            ax.plot([hx, hx - size*0.3], [y - size, y - size*1.3], 
                   'k-', linewidth=0.8)


def plot_moment_diagram_on_structure(ax, points, cells, moment_data, 
                                      scale, color, fill_color,
                                      show_element_ids=None,
                                      linewidth=LINEWIDTH_DIAGRAM, 
                                      show_fill=True, 
                                      use_cell_data=True, 
                                      show_connecting_lines=True,
                                      value_fontsize=9):
    """
    Plot moment diagram perpendicular to each element.
    Only shows values for elements in show_element_ids.
    """
    
    for idx, cell in enumerate(cells):
        elem_id = idx + 1
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
            
            if show_element_ids is not None and elem_id in show_element_ids and abs(moment_value) > 1:
                mid_offset = (offset_p1 + offset_p2) / 2
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


def plot_sensitivity_diagram_on_structure(ax, points, cells, sensitivity_values, 
                                          scale, 
                                          show_element_ids=None,
                                          linewidth=LINEWIDTH_DIAGRAM, 
                                          show_fill=True,
                                          show_connecting_lines=True, 
                                          value_fontsize=8):
    """
    Plot sensitivity diagram perpendicular to each element.
    Only shows values for elements in show_element_ids.
    """
    
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
        
        if sens_value >= 0:
            color = '#228B22'
            fill_color = COLOR_FILL_SENSITIVITY_POS
        else:
            color = '#DC143C'
            fill_color = COLOR_FILL_SENSITIVITY_NEG
        
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
        
        if show_element_ids is not None and elem_id in show_element_ids:
            mid_offset = (offset_p1 + offset_p2) / 2
            text_pos = mid_offset + perp * np.sign(sens_value) * 0.015
            
            ax.annotate(f'{sens_value:.2e}', 
                       xy=(text_pos[0], text_pos[1]),
                       fontsize=value_fontsize, color=color,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.15', 
                                facecolor='white', alpha=0.8, 
                                edgecolor='none'))


# =============================================================================
# SENSITIVITY COMPUTATION
# =============================================================================

def compute_moment_sensitivity(E, I, L_elements, M_primary, M_dual):
    """Compute ∂M/∂(EI) for all elements using the adjoint method."""
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


def print_results(E, I, sensitivities, total_sensitivity, geometry_info, 
                  classifications=None, response_element=None):
    """Print formatted sensitivity results."""
    EI = E * I
    
    print("\n" + "=" * 80)
    print("MOMENT SENSITIVITY ANALYSIS: ∂M/∂(EI)")
    print("Using General Influence (Adjoint) Method")
    print("=" * 80)
    
    print(f"\nMaterial/Section Properties (from JSON):")
    print(f"  E  = {E:.4e} Pa")
    print(f"  I  = {I:.4e} m⁴")
    print(f"  EI = {EI:.4e} N·m²")
    
    print(f"\nGeometry (from VTK):")
    print(f"  Number of elements: {geometry_info['n_elements']}")
    print(f"  Total length: {geometry_info['total_length']:.6f} m")
    print(f"  Beam span: {geometry_info['beam_span']:.6f} m")
    
    if classifications:
        n_columns = sum(1 for v in classifications.values() if v == 'column')
        n_beams = sum(1 for v in classifications.values() if v == 'beam')
        print(f"  Columns: {n_columns}, Beams: {n_beams}")
    
    if response_element:
        print(f"\nResponse: Bending moment in Element {response_element}")
    
    print("\n" + "-" * 80)
    print(f"{'Elem k':^8} {'Type':^8} {'M_k [N·m]':^14} {'M̄_k [N·m]':^14} "
          f"{'L_k [m]':^10} {'∂M/∂(EI)_k':^20}")
    print("-" * 80)
    
    for eid, data in sorted(sensitivities.items()):
        member_type = classifications.get(eid, 'N/A')[:3].upper() if classifications else 'N/A'
        print(f"{eid:^8} {member_type:^8} {data['M_primary']:^+14.4f} {data['M_dual']:^+14.6f} "
              f"{data['length']:^10.4f} {data['dM_dEI']:^+20.6e}")
    
    print("-" * 80)
    print(f"{'TOTAL':^8} {' ':^8} {' ':^14} {' ':^14} "
          f"{' ':^10} {total_sensitivity:^+20.6e}")
    print("-" * 80)
    
    delta_EI_percent = 10.0
    delta_EI = (delta_EI_percent / 100.0) * EI
    delta_M = total_sensitivity * delta_EI
    
    print(f"\nLinear Approximation:")
    print(f"  If EI increases by {delta_EI_percent}%: ΔM ≈ {delta_M:+.6f} N·m")


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_sensitivity_visualization(vtk_primary, vtk_dual, E, I, 
                                     sensitivities, total_sensitivity,
                                     M_primary_dict, M_dual_dict,
                                     geometry_info,
                                     save_figures=True, output_folder="."):
    """Create sensitivity visualization showing only max/min values per member type."""
    
    points = vtk_primary['points']
    cells = vtk_primary['cells']
    
    # Get points from dual VTK for repeated node detection
    points_dual = vtk_dual['points']
    
    cell_moment_primary = vtk_primary['cell_data'].get('MOMENT', np.zeros((len(cells), 3)))
    cell_moment_dual = vtk_dual['cell_data'].get('MOMENT', np.zeros((len(cells), 3)))
    
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    x_margin = (x_max - x_min) * 0.15 if x_max > x_min else 0.5
    y_margin = max((y_max - y_min) * 0.5, 0.3)
    
    EI = E * I
    
    if save_figures and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"\nCreated output folder: {output_folder}")
    
    # =========================================================================
    # FIND REPEATED NODES FROM DUAL VTK
    # =========================================================================
    print("\n  Detecting repeated nodes from dual VTK...")
    repeated_node_info = find_repeated_nodes(points_dual)
    
    # =========================================================================
    # CLASSIFY MEMBERS AND GET EXTREME ELEMENT IDS
    # =========================================================================
    print("\n  Classifying members (columns vs beams)...")
    classifications = classify_members(points, cells)
    
    n_columns = sum(1 for v in classifications.values() if v == 'column')
    n_beams = sum(1 for v in classifications.values() if v == 'beam')
    print(f"    Columns: {n_columns}, Beams: {n_beams}")
    
    show_primary_ids = get_extreme_element_ids(cell_moment_primary, classifications)
    show_dual_ids = get_extreme_element_ids(cell_moment_dual, classifications)
    show_sensitivity_ids = get_extreme_sensitivity_ids(sensitivities, classifications)
    
    print(f"    Showing primary moment values for elements: {sorted(show_primary_ids)}")
    print(f"    Showing dual moment values for elements: {sorted(show_dual_ids)}")
    print(f"    Showing sensitivity values for elements: {sorted(show_sensitivity_ids)}")
    
    max_M_dual = np.max(np.abs(cell_moment_dual[:, 2]))
    
    # =========================================================================
    # PLOT 1: Primary Moment Diagram (NO circles)
    # =========================================================================
    
    fig1, ax = plt.subplots(figsize=(14, 6), dpi=FIGURE_DPI)
    ax.set_title('Primary Bending Moment Diagram M(x)\n[From Applied Loading]', 
                 fontsize=14, fontweight='bold')
    plot_structure(ax, points, cells, color='gray', linewidth=1.5, 
                  show_element_labels=True)
    plot_moment_diagram_on_structure(ax, points, cells, cell_moment_primary,
                                     scale=PRIMARY_MOMENT_SCALE,
                                     color=COLOR_PRIMARY_MOMENT,
                                     fill_color=COLOR_FILL_PRIMARY,
                                     show_element_ids=show_primary_ids,
                                     use_cell_data=True)
    add_supports(ax, points)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    max_M = np.max(cell_moment_primary[:, 2])
    min_M = np.min(cell_moment_primary[:, 2])
    info_text = (f'Max M: {max_M:.1f} N·m\n'
                f'Min M: {min_M:.1f} N·m\n'
                f'Scale: {PRIMARY_MOMENT_SCALE}\n'
                f'─────────────\n'
                f'Elements: {geometry_info["n_elements"]}\n')
    ax.text(0.5, 0.25, info_text, transform=ax.transAxes, fontsize=10,
           horizontalalignment='center', verticalalignment='center', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if save_figures:
        filepath = os.path.join(output_folder, 'primary_moment_diagram.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    
    # =========================================================================
    # PLOT 2: Dual Moment Diagram (WITH circles for repeated nodes)
    # =========================================================================
    
    max_M_dual_val = np.max(cell_moment_dual[:, 2])
    min_M_dual_val = np.min(cell_moment_dual[:, 2])

    fig2, ax = plt.subplots(figsize=(14, 6), dpi=FIGURE_DPI)
    ax.set_title('Dual/Adjoint Bending Moment Diagram M̄(x)\n[From Unit Virtual Load at Response Location]', 
                 fontsize=14, fontweight='bold')
    plot_structure(ax, points, cells, color='gray', linewidth=1.5,
                  show_element_labels=True)
    plot_moment_diagram_on_structure(ax, points, cells, cell_moment_dual,
                                     scale=DUAL_MOMENT_SCALE,
                                     color=COLOR_DUAL_MOMENT,
                                     fill_color=COLOR_FILL_DUAL,
                                     show_element_ids=show_dual_ids,
                                     use_cell_data=True)
    add_supports(ax, points)
    
    # Add circles for repeated nodes
    add_repeated_node_circles(ax, repeated_node_info, label='Response Location')
    
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_xlim(x_min - x_margin - max_M_dual*DUAL_MOMENT_SCALE, 
                x_max + x_margin + max_M_dual*DUAL_MOMENT_SCALE)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    info_text = (f'Max M̄: {max_M_dual_val:.1f} N·m\n'
                f'Min M̄: {min_M_dual_val:.1f} N·m\n'
                f'Scale: {DUAL_MOMENT_SCALE}\n'
                f'─────────────────\n'
                f'Elements: {geometry_info["n_elements"]}\n')
    ax.text(0.5, 0.25, info_text, transform=ax.transAxes, fontsize=10,
           horizontalalignment='center', verticalalignment='center', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if save_figures:
        filepath = os.path.join(output_folder, 'dual_moment_diagram.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    
    # =========================================================================
    # PLOT 3: Sensitivity Diagram (WITH circles for repeated nodes)
    # =========================================================================
    
    fig3, ax = plt.subplots(figsize=(14, 6), dpi=FIGURE_DPI)
    ax.set_title('Sensitivity Diagram: ∂M/∂(EI)\n[How Response Moment Changes with Element Stiffness]', 
                 fontsize=14, fontweight='bold')
    plot_structure(ax, points, cells, color='gray', linewidth=1.5,
                  show_element_labels=True)
    plot_sensitivity_diagram_on_structure(ax, points, cells, sensitivities,
                                          scale=SENSITIVITY_SCALE,
                                          show_element_ids=show_sensitivity_ids)
    add_supports(ax, points)
    
    # Add circles for repeated nodes
    add_repeated_node_circles(ax, repeated_node_info, label='Response Location')
    
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_xlim(x_min - x_margin, x_max + x_margin + 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Legend with sensitivity colors and response location
    pos_patch = mpatches.Patch(color=COLOR_FILL_SENSITIVITY_POS, alpha=0.4, 
                               label='Positive: ↑EI → ↑M')
    neg_patch = mpatches.Patch(color=COLOR_FILL_SENSITIVITY_NEG, alpha=0.4, 
                               label='Negative: ↑EI → ↓M')
    response_line = plt.Line2D([0], [0], color=COLOR_REPEATED_NODE, 
                                linewidth=REPEATED_NODE_CIRCLE_LINEWIDTH,
                                label='Response Location')
    ax.legend(handles=[pos_patch, neg_patch, response_line], loc='upper right', fontsize=9)
    
    max_sens = max(s['dM_dEI'] for s in sensitivities.values())
    min_sens = min(s['dM_dEI'] for s in sensitivities.values())
    info_text = (f'E = {E:.2e} Pa\n'
                f'I = {I:.2e} m⁴\n'
                f'EI = {EI:.2e} N·m²\n'
                f'─────────────────\n'
                f'Elements: {geometry_info["n_elements"]}\n'
                f'Scale: {SENSITIVITY_SCALE:.0e}\n'
                f'─────────────────\n'
                f'Max: {max_sens:.4e}\n'
                f'Min: {min_sens:.4e}\n'
                f'─────────────────\n'
                f'TOTAL ∂M/∂(EI):\n'
                f'{total_sensitivity:.4e}')
    
    ax.text(0.75, 0.5, info_text, transform=ax.transAxes, fontsize=9,
           horizontalalignment='center', verticalalignment='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black'))
    
    if save_figures:
        filepath = os.path.join(output_folder, 'sensitivity_diagram.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    
    # =========================================================================
    # PLOT 4: All Three Diagrams Side by Side (Horizontal)
    # =========================================================================

    fig4, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=FIGURE_DPI)
    fig4.suptitle('Sensitivity Analysis: Complete Diagram Set', fontsize=14, fontweight='bold')

    # Primary Moment (no circles)
    ax = axes[0]
    ax.set_title('Primary Moment M(x)', fontsize=12, fontweight='bold')
    plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    plot_moment_diagram_on_structure(ax, points, cells, cell_moment_primary,
                                     scale=PRIMARY_MOMENT_SCALE,
                                     color=COLOR_PRIMARY_MOMENT,
                                     fill_color=COLOR_FILL_PRIMARY,
                                     show_element_ids=show_primary_ids,
                                     use_cell_data=True)
    add_supports(ax, points)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    # Dual Moment (with circles)
    ax = axes[1]
    ax.set_title('Dual Moment M̄(x)', fontsize=12, fontweight='bold')
    plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    plot_moment_diagram_on_structure(ax, points, cells, cell_moment_dual,
                                     scale=DUAL_MOMENT_SCALE,
                                     color=COLOR_DUAL_MOMENT,
                                     fill_color=COLOR_FILL_DUAL,
                                     show_element_ids=show_dual_ids,
                                     use_cell_data=True)
    add_supports(ax, points)
    add_repeated_node_circles(ax, repeated_node_info, label=None)  # No label for combined plot
    ax.set_xlim(x_min - x_margin - max_M_dual*DUAL_MOMENT_SCALE, 
                x_max + x_margin + max_M_dual*DUAL_MOMENT_SCALE)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    # Sensitivity (with circles)
    ax = axes[2]
    ax.set_title(f'Sensitivity ∂M/∂(EI)  |  TOTAL = {total_sensitivity:.4e}', 
                fontsize=12, fontweight='bold', color='#8B0000')
    plot_structure(ax, points, cells, color='gray', linewidth=1.5)
    plot_sensitivity_diagram_on_structure(ax, points, cells, sensitivities,
                                          scale=SENSITIVITY_SCALE,
                                          show_element_ids=show_sensitivity_ids)
    add_supports(ax, points)
    add_repeated_node_circles(ax, repeated_node_info, label=None)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    ax.text(0.5, 0.02, f'Σ = {total_sensitivity:.4e}', 
        transform=ax.transAxes, fontsize=11, fontweight='bold',
        ha='center', va='bottom',
        bbox=dict(boxstyle='round', facecolor='#FFD700', alpha=0.9))

    # Add a common legend for the combined plot
    response_circle = plt.Line2D([0], [0], marker='o', color='w', 
                                  markeredgecolor=COLOR_REPEATED_NODE,
                                  markerfacecolor='none',
                                  markersize=12, markeredgewidth=2.5,
                                  label='Response Location')
    fig4.legend(handles=[response_circle], loc='upper right', 
                bbox_to_anchor=(0.99, 0.95), fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_figures:
        filepath = os.path.join(output_folder, 'sensitivity_horizontal.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")

    plt.show()
    
    return classifications, repeated_node_info


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(primary_vtk_path, dual_vtk_path, material_json_path=None,
         E=None, I=None, response_element=None, 
         create_plots=True, save_plots=True, output_dir="."):
    """
    Main function - compute sensitivity from VTK files with visualization.
    """
    
    print("=" * 80)
    print("MOMENT SENSITIVITY ANALYSIS: ∂M/∂(EI)")
    print("Using Adjoint Method with VTK File Parsing")
    print("=" * 80)
    
    # Step 1: Parse Primary VTK File
    print(f"\n1. Loading Primary VTK: {primary_vtk_path}")
    vtk_primary = parse_vtk_file(primary_vtk_path)
    M_primary = parse_vtk_cell_moments(primary_vtk_path)
    
    if not M_primary:
        print("Error: Could not parse primary moments from VTK file.")
        return None, None
    
    print(f"   Found {len(M_primary)} elements with primary moments")
    
    # Step 2: Parse Dual VTK File
    print(f"\n2. Loading Dual VTK: {dual_vtk_path}")
    vtk_dual = parse_vtk_file(dual_vtk_path)
    M_dual = parse_vtk_cell_moments(dual_vtk_path)
    
    if not M_dual:
        print("Error: Could not parse dual moments from VTK file.")
        return None, None
    
    print(f"   Found {len(M_dual)} elements with dual moments")
    
    # Step 3: Extract Geometry from VTK
    print("\n3. Extracting geometry from VTK...")
    geometry_info = calculate_geometry_from_vtk(vtk_primary)
    
    # Step 4: Load Material Properties
    print("\n4. Loading material properties...")
    
    if E is not None and I is not None:
        print(f"   Using provided values: E = {E:.4e} Pa, I = {I:.4e} m⁴")
    else:
        if material_json_path is None:
            material_json_path = find_material_json(primary_vtk_path)
        
        if material_json_path:
            material_props = load_material_properties(material_json_path)
            E = material_props['E']
            I = material_props['I']
        else:
            raise ValueError("Material properties not provided and StructuralMaterials.json not found.")
    
    # Step 5: Classify members
    print("\n5. Classifying members...")
    points = vtk_primary['points']
    cells = vtk_primary['cells']
    classifications = classify_members(points, cells)
    
    n_columns = sum(1 for v in classifications.values() if v == 'column')
    n_beams = sum(1 for v in classifications.values() if v == 'beam')
    print(f"   Columns: {n_columns}, Beams: {n_beams}")
    
    # Step 6: Reconcile element data
    print("\n6. Reconciling element data...")
    
    primary_elem_ids = set(M_primary.keys())
    dual_elem_ids = set(M_dual.keys())
    common_elem_ids = primary_elem_ids & dual_elem_ids
    
    print(f"   Primary elements: {len(primary_elem_ids)}")
    print(f"   Dual elements: {len(dual_elem_ids)}")
    print(f"   Common elements: {len(common_elem_ids)}")
    
    M_primary_common = {k: v for k, v in M_primary.items() if k in common_elem_ids}
    M_dual_common = {k: v for k, v in M_dual.items() if k in common_elem_ids}
    
    L_elements = {k: v for k, v in geometry_info['element_lengths'].items() if k in common_elem_ids}
    
    # Step 7: Compute Sensitivities
    print("\n7. Computing sensitivities...")
    
    sensitivities, total_sensitivity = compute_moment_sensitivity(
        E, I, L_elements, M_primary_common, M_dual_common
    )
    
    # Step 8: Print Results
    print_results(E, I, sensitivities, total_sensitivity, geometry_info, 
                  classifications, response_element)
    
    # Step 9: Create Visualizations
    if create_plots:
        print("\n8. Creating visualizations...")
        
        create_sensitivity_visualization(
            vtk_primary, vtk_dual, E, I,
            sensitivities, total_sensitivity,
            M_primary_common, M_dual_common,
            geometry_info,
            save_figures=save_plots,
            output_folder=output_dir
        )
    
    return sensitivities, total_sensitivity


# =============================================================================
# RUN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    
    # VTK file paths (required)
    PRIMARY_VTK = os.path.join(FOLDER, "vtk_output/Parts_Beam_Beams_0_1.vtk")  
    DUAL_VTK = os.path.join(FOLDER, "vtk_output_dual/Parts_Beam_Beams_0_1.vtk")
    
    # Material JSON path (optional - will auto-search if None)
    MATERIAL_JSON = os.path.join(FOLDER, "StructuralMaterials.json")
    
    # Optional manual overrides (set to None to use JSON values)
    E_OVERRIDE = None
    I_OVERRIDE = None
    
    # Response element (optional)
    RESPONSE_ELEMENT = None
    
    # Plotting options
    CREATE_PLOTS = True
    SAVE_PLOTS = True
    OUTPUT_DIR = OUTPUT_FOLDER
    
    # Run Analysis
    sensitivities, total = main(
        primary_vtk_path=PRIMARY_VTK,
        dual_vtk_path=DUAL_VTK,
        material_json_path=MATERIAL_JSON,
        E=E_OVERRIDE,
        I=I_OVERRIDE,
        response_element=RESPONSE_ELEMENT,
        create_plots=CREATE_PLOTS,
        save_plots=SAVE_PLOTS,
        output_dir=OUTPUT_DIR
    )