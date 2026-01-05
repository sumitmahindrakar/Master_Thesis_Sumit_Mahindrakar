"""
Unified Plotter for Sensitivity Analysis Pipeline
=================================================
Generates all visualization plots from VTK results.

Plots Generated:
1. Structure diagram
2. Deflection diagram (primary)
3. Deflection diagram (dual / influence line)
4. Bending moment diagram (primary)
5. Bending moment diagram (dual)
6. Rotation diagram (primary)
7. Rotation diagram (dual)
8. Sensitivity diagram
9. Combined comparison plots

Features:
- Auto-detects support types (beam vs frame)
- Auto-calculates scales if not provided
- Highlights max/min values per member type
- Marks response location from hinge_info.json
- Properly handles dual VTK with duplicate kink nodes

Author: SA Pipeline
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, Polygon
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.config_loader import load_config, Config
except ImportError:
    print("Warning: Could not import config_loader.")
    Config = None


# =============================================================================
# CONFIGURATION
# =============================================================================

COLORS = {
    'structure': 'black',
    'deflection': 'blue',
    'deflection_dual': 'cyan',
    'moment_primary': 'red',
    'moment_dual': 'green',
    'rotation': 'purple',
    'rotation_dual': 'magenta',
    'sensitivity_pos': '#228B22',
    'sensitivity_neg': '#DC143C',
    'fill_deflection': 'lightblue',
    'fill_deflection_dual': 'lightcyan',
    'fill_moment_primary': 'lightcoral',
    'fill_moment_dual': 'lightgreen',
    'fill_rotation': 'plum',
    'fill_rotation_dual': 'thistle',
    'fill_sensitivity_pos': '#90EE90',
    'fill_sensitivity_neg': '#FFB6C1',
    'response_location': 'red',
    'support': 'black'
}

LINEWIDTH_STRUCTURE = 2.0
LINEWIDTH_DIAGRAM = 1.5
LINEWIDTH_RESPONSE = 2.0
FIGURE_DPI = 100
RESPONSE_CIRCLE_RADIUS = 0.05  # Smaller circle
SUPPORT_SIZE = 0.15


# =============================================================================
# VTK PARSER
# =============================================================================

def parse_vtk_file(filename: str) -> Dict[str, Any]:
    """Parse a VTK file and extract all data."""
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"VTK file not found: {filename}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = {
        'points': None,
        'cells': None,
        'point_data': {},
        'cell_data': {}
    }
    
    i = 0
    current_section = None
    
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
                        if values and values[0] in ['POINT_DATA', 'CELL_DATA', 'FIELD']:
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
                    continue
                except (ValueError, IndexError):
                    pass
        
        i += 1
    
    return data


# =============================================================================
# RESPONSE LOCATION LOADER
# =============================================================================

def load_response_location(config: Config) -> Optional[Tuple[float, float, float]]:
    """Load response location from hinge_info.json."""
    hinge_info_path = os.path.join(config.paths.input_dir, 'hinge_info.json')
    
    if os.path.exists(hinge_info_path):
        with open(hinge_info_path, 'r') as f:
            hinge_info = json.load(f)
        
        response_loc = hinge_info.get('response_location', None)
        if response_loc:
            return tuple(response_loc)
    
    return (config.response.x, config.response.y, 0.0)


def get_response_location_info(config: Config) -> Dict[str, Any]:
    """Get response location information for plotting."""
    response_loc = load_response_location(config)
    return {'position': response_loc, 'found': response_loc is not None}


# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================

def get_element_direction(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Get unit direction vector of an element."""
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length > 0:
        return direction / length
    return np.array([1, 0, 0])


def get_perpendicular_direction(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Get perpendicular direction (rotated 90° CCW in 2D)."""
    direction = get_element_direction(p1, p2)
    return np.array([-direction[1], direction[0], 0])


def classify_element(p1: np.ndarray, p2: np.ndarray, angle_threshold: float = 45.0) -> str:
    """Classify element as 'beam' or 'column'."""
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    
    if dy < 1e-10:
        return 'beam'
    elif dx < 1e-10:
        return 'column'
    else:
        angle_from_vertical = np.degrees(np.arctan(dx / dy))
        return 'column' if angle_from_vertical < angle_threshold else 'beam'


def detect_structure_type(points: np.ndarray) -> str:
    """Detect if structure is a beam or frame based on geometry."""
    y_coords = points[:, 1]
    y_range = y_coords.max() - y_coords.min()
    x_range = points[:, 0].max() - points[:, 0].min()
    
    if y_range < x_range * 0.1:
        return 'beam'
    else:
        return 'frame'


def detect_support_type(config: Config, structure_type: str) -> str:
    """Detect support type from template name and structure type."""
    template = config.problem.template.lower()
    
    if structure_type == 'frame':
        return 'pinned_base'
    elif 'beam' in template:
        return 'fixed_fixed'
    else:
        return 'simply_supported'


def find_support_nodes(points: np.ndarray, structure_type: str) -> Dict[str, List[int]]:
    """Find support node indices based on structure type."""
    supports = {'left': [], 'right': [], 'bottom': [], 'all': []}
    
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y = y_coords.min()
    
    tolerance_x = (max_x - min_x) * 0.01 if max_x > min_x else 0.01
    tolerance_y = 0.01
    
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        if abs(x - min_x) < tolerance_x:
            supports['left'].append(i)
        if abs(x - max_x) < tolerance_x:
            supports['right'].append(i)
        if abs(y - min_y) < tolerance_y:
            supports['bottom'].append(i)
    
    if structure_type == 'beam':
        supports['all'] = list(set(supports['left'] + supports['right']))
    else:
        supports['all'] = supports['bottom']
    
    return supports


def get_extreme_elements(values: Dict[int, float], classifications: Dict[int, str]) -> set:
    """Get element IDs with max/min values per member type."""
    
    column_values = {k: v for k, v in values.items() if classifications.get(k) == 'column'}
    beam_values = {k: v for k, v in values.items() if classifications.get(k) == 'beam'}
    
    show_elements = set()
    
    for member_values in [column_values, beam_values]:
        if member_values:
            max_val = max(member_values.values())
            min_val = min(member_values.values())
            for elem_id, val in member_values.items():
                if abs(val - max_val) < 1e-10 * max(abs(max_val), 1):
                    show_elements.add(elem_id)
                if abs(val - min_val) < 1e-10 * max(abs(min_val), 1):
                    show_elements.add(elem_id)
    
    return show_elements


# =============================================================================
# AUTO SCALE CALCULATION
# =============================================================================

def calculate_scales(vtk_primary: Dict, vtk_dual: Optional[Dict],
                     sensitivity_values: Optional[Dict],
                     config: Optional[Config] = None) -> Dict[str, float]:
    """Calculate appropriate scale factors for all diagrams."""
    
    points = vtk_primary['points']
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    structure_size = max(x_range, y_range)
    target_size = structure_size * 0.2
    
    scales = {}
    
    if config and config.plotting.deflection_scale:
        scales['deflection'] = config.plotting.deflection_scale
    else:
        if 'DISPLACEMENT' in vtk_primary['point_data']:
            disp = vtk_primary['point_data']['DISPLACEMENT']
            max_disp = np.max(np.abs(disp[:, :2]))
            scales['deflection'] = target_size / max_disp if max_disp > 1e-12 else 1000.0
        else:
            scales['deflection'] = 1000.0
    
    if vtk_dual and 'DISPLACEMENT' in vtk_dual['point_data']:
        disp = vtk_dual['point_data']['DISPLACEMENT']
        max_disp = np.max(np.abs(disp[:, :2]))
        scales['deflection_dual'] = target_size / max_disp if max_disp > 1e-12 else 1000.0
    else:
        scales['deflection_dual'] = 1000.0
    
    if config and config.plotting.moment_scale:
        scales['moment_primary'] = config.plotting.moment_scale
    else:
        if 'MOMENT' in vtk_primary['cell_data']:
            moment = vtk_primary['cell_data']['MOMENT']
            max_moment = np.max(np.abs(moment[:, 2]))
            scales['moment_primary'] = target_size / max_moment if max_moment > 1e-6 else 0.0001
        else:
            scales['moment_primary'] = 0.0001
    
    if vtk_dual and 'MOMENT' in vtk_dual['cell_data']:
        moment = vtk_dual['cell_data']['MOMENT']
        max_moment = np.max(np.abs(moment[:, 2]))
        scales['moment_dual'] = target_size / max_moment if max_moment > 1e-6 else 0.000001
    else:
        scales['moment_dual'] = 0.000001
    
    if config and config.plotting.rotation_scale:
        scales['rotation'] = config.plotting.rotation_scale
    else:
        if 'ROTATION' in vtk_primary['point_data']:
            rot = vtk_primary['point_data']['ROTATION']
            max_rot = np.max(np.abs(rot[:, 2]))
            scales['rotation'] = target_size / max_rot if max_rot > 1e-12 else 1000.0
        else:
            scales['rotation'] = 1000.0
    
    if vtk_dual and 'ROTATION' in vtk_dual['point_data']:
        rot = vtk_dual['point_data']['ROTATION']
        max_rot = np.max(np.abs(rot[:, 2]))
        scales['rotation_dual'] = target_size / max_rot if max_rot > 1e-12 else 1000.0
    else:
        scales['rotation_dual'] = 1000.0
    
    if config and config.plotting.sensitivity_scale:
        scales['sensitivity'] = config.plotting.sensitivity_scale
    else:
        if sensitivity_values:
            max_sens = max(abs(v) for v in sensitivity_values.values())
            scales['sensitivity'] = target_size / max_sens if max_sens > 1e-20 else 1e6
        else:
            scales['sensitivity'] = 1e6
    
    return scales


# =============================================================================
# SUPPORT DRAWING FUNCTIONS
# =============================================================================

def draw_fixed_support_horizontal(ax, x: float, y: float, 
                                   side: str = 'bottom',
                                   size: float = SUPPORT_SIZE):
    """Draw fixed support symbol for horizontal beam."""
    
    if side == 'left':
        ax.plot([x, x], [y - size, y + size], color=COLORS['support'], linewidth=2)
        for hy in np.linspace(y - size, y + size, 6):
            ax.plot([x - size*0.4, x], [hy - size*0.2, hy], color=COLORS['support'], linewidth=0.8)
    else:
        ax.plot([x, x], [y - size, y + size], color=COLORS['support'], linewidth=2)
        for hy in np.linspace(y - size, y + size, 6):
            ax.plot([x, x + size*0.4], [hy, hy - size*0.2], color=COLORS['support'], linewidth=0.8)


def draw_pinned_support(ax, x: float, y: float, size: float = SUPPORT_SIZE):
    """Draw pinned support symbol."""
    
    triangle = plt.Polygon([
        [x, y],
        [x - size, y - size],
        [x + size, y - size]
    ], fill=False, edgecolor=COLORS['support'], linewidth=1.5)
    ax.add_patch(triangle)
    
    ax.plot([x - size*1.2, x + size*1.2], [y - size, y - size], color=COLORS['support'], linewidth=1.5)
    
    for hx in np.linspace(x - size, x + size, 4):
        ax.plot([hx, hx - size*0.3], [y - size, y - size*1.3], color=COLORS['support'], linewidth=0.8)


def draw_roller_support(ax, x: float, y: float, size: float = SUPPORT_SIZE):
    """Draw roller support symbol."""
    
    triangle = plt.Polygon([
        [x, y],
        [x - size, y - size*0.7],
        [x + size, y - size*0.7]
    ], fill=False, edgecolor=COLORS['support'], linewidth=1.5)
    ax.add_patch(triangle)
    
    circle = Circle((x, y - size*0.9), size*0.2, fill=False, edgecolor=COLORS['support'], linewidth=1.5)
    ax.add_patch(circle)
    
    ax.plot([x - size*1.2, x + size*1.2], [y - size*1.1, y - size*1.1], color=COLORS['support'], linewidth=1.5)


def add_supports(ax, points: np.ndarray, supports: Dict[str, List[int]], 
                 support_type: str, structure_type: str):
    """Add appropriate support symbols."""
    
    if structure_type == 'beam':
        if support_type == 'fixed_fixed':
            for idx in supports.get('left', []):
                x, y = points[idx][0], points[idx][1]
                draw_fixed_support_horizontal(ax, x, y, side='left')
            for idx in supports.get('right', []):
                x, y = points[idx][0], points[idx][1]
                draw_fixed_support_horizontal(ax, x, y, side='right')
        elif support_type == 'simply_supported':
            for idx in supports.get('left', []):
                x, y = points[idx][0], points[idx][1]
                draw_pinned_support(ax, x, y)
            for idx in supports.get('right', []):
                x, y = points[idx][0], points[idx][1]
                draw_roller_support(ax, x, y)
        else:
            for idx in supports.get('all', []):
                x, y = points[idx][0], points[idx][1]
                draw_pinned_support(ax, x, y)
    else:
        for idx in supports.get('bottom', []):
            x, y = points[idx][0], points[idx][1]
            draw_pinned_support(ax, x, y)


def add_response_circle(ax, position: Tuple[float, float, float],
                        radius: float = RESPONSE_CIRCLE_RADIUS,
                        color: str = COLORS['response_location'],
                        label: str = 'Response Location'):
    """Add circle at response location."""
    
    if position is None:
        return
    
    x, y = position[0], position[1]
    circle = Circle((x, y), radius, fill=False, edgecolor=color, linewidth=LINEWIDTH_RESPONSE, zorder=10)
    ax.add_patch(circle)
    
    if label:
        ax.plot([], [], color=color, linewidth=LINEWIDTH_RESPONSE, label=label)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_structure(ax, points: np.ndarray, cells: List, 
                   color: str = COLORS['structure'],
                   linewidth: float = LINEWIDTH_STRUCTURE,
                   label: str = 'Structure',
                   show_node_labels: bool = False,
                   show_element_labels: bool = False):
    """Plot the structure."""
    
    for idx, cell in enumerate(cells):
        p1 = points[cell[0]]
        p2 = points[cell[1]]
        
        if idx == 0:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth, label=label)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth)
        
        if show_element_labels:
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            ax.annotate(f'E{idx+1}', (mid_x, mid_y), textcoords="offset points", 
                       xytext=(0, -12), fontsize=7, color='gray', ha='center')
    
    unique_positions = {}
    for i, point in enumerate(points):
        key = (round(point[0], 6), round(point[1], 6))
        if key not in unique_positions:
            unique_positions[key] = i
    
    unique_indices = list(unique_positions.values())
    ax.scatter(points[unique_indices, 0], points[unique_indices, 1], color=color, s=30, zorder=5)
    
    if show_node_labels:
        for i, point in enumerate(points):
            ax.annotate(f'{i}', (point[0], point[1]), textcoords="offset points", 
                       xytext=(5, 5), fontsize=7, color='gray')


def plot_deflection_diagram(ax, points: np.ndarray, cells: List,
                            displacement: np.ndarray, scale: float,
                            color: str = COLORS['deflection'],
                            fill_color: str = COLORS['fill_deflection'],
                            show_fill: bool = True):
    """Plot deflection diagram."""
    
    for cell in cells:
        n1, n2 = cell[0], cell[1]
        p1 = points[n1][:2]
        p2 = points[n2][:2]
        perp = get_perpendicular_direction(p1, p2)[:2]
        
        disp1 = displacement[n1][:2]
        disp2 = displacement[n2][:2]
        disp1_perp = np.dot(disp1, perp)
        disp2_perp = np.dot(disp2, perp)
        
        offset_p1 = p1 + perp * disp1_perp * scale
        offset_p2 = p2 + perp * disp2_perp * scale
        
        ax.plot([offset_p1[0], offset_p2[0]], [offset_p1[1], offset_p2[1]], color=color, linewidth=LINEWIDTH_DIAGRAM)
        ax.plot([p1[0], offset_p1[0]], [p1[1], offset_p1[1]], color=color, linewidth=0.5, linestyle='--', alpha=0.7)
        ax.plot([p2[0], offset_p2[0]], [p2[1], offset_p2[1]], color=color, linewidth=0.5, linestyle='--', alpha=0.7)
        
        if show_fill:
            polygon_x = [p1[0], offset_p1[0], offset_p2[0], p2[0]]
            polygon_y = [p1[1], offset_p1[1], offset_p2[1], p2[1]]
            ax.fill(polygon_x, polygon_y, color=fill_color, alpha=0.3)


def plot_moment_diagram(ax, points: np.ndarray, cells: List,
                        moment_data: np.ndarray, scale: float,
                        color: str, fill_color: str,
                        show_fill: bool = True,
                        show_values: bool = True,
                        show_element_ids: Optional[set] = None,
                        classifications: Optional[Dict] = None):
    """Plot bending moment diagram."""
    
    for idx, cell in enumerate(cells):
        elem_id = idx + 1
        n1, n2 = cell[0], cell[1]
        p1 = points[n1][:2]
        p2 = points[n2][:2]
        perp = get_perpendicular_direction(p1, p2)[:2]
        
        moment_value = moment_data[idx][2] if idx < len(moment_data) else 0
        
        offset_p1 = p1 + perp * moment_value * scale
        offset_p2 = p2 + perp * moment_value * scale
        
        ax.plot([offset_p1[0], offset_p2[0]], [offset_p1[1], offset_p2[1]], color=color, linewidth=LINEWIDTH_DIAGRAM)
        ax.plot([p1[0], offset_p1[0]], [p1[1], offset_p1[1]], color=color, linewidth=0.5, linestyle='-', alpha=0.7)
        ax.plot([p2[0], offset_p2[0]], [p2[1], offset_p2[1]], color=color, linewidth=0.5, linestyle='-', alpha=0.7)
        
        if show_fill:
            polygon_x = [p1[0], offset_p1[0], offset_p2[0], p2[0]]
            polygon_y = [p1[1], offset_p1[1], offset_p2[1], p2[1]]
            ax.fill(polygon_x, polygon_y, color=fill_color, alpha=0.3)
        
        if show_values and show_element_ids and elem_id in show_element_ids:
            if abs(moment_value) > 1:
                mid_offset = (offset_p1 + offset_p2) / 2
                ax.annotate(f'{moment_value:.2e} N·m', xy=(mid_offset[0], mid_offset[1]),
                           fontsize=8, color=color, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))


def plot_rotation_diagram(ax, points: np.ndarray, cells: List,
                          rotation_data: np.ndarray, scale: float,
                          color: str = COLORS['rotation'],
                          fill_color: str = COLORS['fill_rotation'],
                          show_fill: bool = True):
    """Plot rotation diagram."""
    
    for cell in cells:
        n1, n2 = cell[0], cell[1]
        p1 = points[n1][:2]
        p2 = points[n2][:2]
        perp = get_perpendicular_direction(p1, p2)[:2]
        
        rot1 = rotation_data[n1][2]
        rot2 = rotation_data[n2][2]
        
        offset_p1 = p1 + perp * rot1 * scale
        offset_p2 = p2 + perp * rot2 * scale
        
        ax.plot([offset_p1[0], offset_p2[0]], [offset_p1[1], offset_p2[1]], color=color, linewidth=LINEWIDTH_DIAGRAM)
        ax.plot([p1[0], offset_p1[0]], [p1[1], offset_p1[1]], color=color, linewidth=0.5, linestyle='--', alpha=0.7)
        ax.plot([p2[0], offset_p2[0]], [p2[1], offset_p2[1]], color=color, linewidth=0.5, linestyle='--', alpha=0.7)
        
        if show_fill:
            polygon_x = [p1[0], offset_p1[0], offset_p2[0], p2[0]]
            polygon_y = [p1[1], offset_p1[1], offset_p2[1], p2[1]]
            ax.fill(polygon_x, polygon_y, color=fill_color, alpha=0.3)


def plot_sensitivity_diagram(ax, points: np.ndarray, cells: List,
                             sensitivity_values: Dict[int, float], scale: float,
                             show_fill: bool = True,
                             show_values: bool = True,
                             show_element_ids: Optional[set] = None):
    """Plot sensitivity diagram."""
    
    for idx, cell in enumerate(cells):
        elem_id = idx + 1
        
        if elem_id not in sensitivity_values:
            continue
        
        n1, n2 = cell[0], cell[1]
        p1 = points[n1][:2]
        p2 = points[n2][:2]
        perp = get_perpendicular_direction(p1, p2)[:2]
        
        sens_value = sensitivity_values[elem_id]
        
        offset_p1 = p1 + perp * sens_value * scale
        offset_p2 = p2 + perp * sens_value * scale
        
        if sens_value >= 0:
            color = COLORS['sensitivity_pos']
            fill_color = COLORS['fill_sensitivity_pos']
        else:
            color = COLORS['sensitivity_neg']
            fill_color = COLORS['fill_sensitivity_neg']
        
        ax.plot([offset_p1[0], offset_p2[0]], [offset_p1[1], offset_p2[1]], color=color, linewidth=LINEWIDTH_DIAGRAM)
        ax.plot([p1[0], offset_p1[0]], [p1[1], offset_p1[1]], color=color, linewidth=0.5, linestyle='-', alpha=0.7)
        ax.plot([p2[0], offset_p2[0]], [p2[1], offset_p2[1]], color=color, linewidth=0.5, linestyle='-', alpha=0.7)
        
        if show_fill:
            polygon_x = [p1[0], offset_p1[0], offset_p2[0], p2[0]]
            polygon_y = [p1[1], offset_p1[1], offset_p2[1], p2[1]]
            ax.fill(polygon_x, polygon_y, color=fill_color, alpha=0.4)
        
        if show_values and show_element_ids and elem_id in show_element_ids:
            mid_offset = (offset_p1 + offset_p2) / 2
            ax.annotate(f'{sens_value:.2e}', xy=(mid_offset[0], mid_offset[1]),
                       fontsize=7, color=color, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8, edgecolor='none'))


# =============================================================================
# MAIN PLOTTING FUNCTION
# =============================================================================

def create_all_plots(config: Config,
                     vtk_primary: Dict, vtk_dual: Dict,
                     sensitivity_values: Optional[Dict[int, float]] = None,
                     total_sensitivity: Optional[float] = None) -> None:
    """Create all visualization plots."""
    
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    points_primary = vtk_primary['points']
    cells_primary = vtk_primary['cells']
    points_dual = vtk_dual['points']
    cells_dual = vtk_dual['cells']
    
    print(f"  Primary: {len(points_primary)} nodes, {len(cells_primary)} elements")
    print(f"  Dual: {len(points_dual)} nodes, {len(cells_dual)} elements")
    
    structure_type = detect_structure_type(points_primary)
    support_type = detect_support_type(config, structure_type)
    supports_primary = find_support_nodes(points_primary, structure_type)
    supports_dual = find_support_nodes(points_dual, structure_type)
    
    print(f"  Structure type: {structure_type}")
    print(f"  Support type: {support_type}")
    
    response_info = get_response_location_info(config)
    response_location = response_info['position']
    print(f"  Response location: {response_location}")
    
    classifications = {}
    for idx, cell in enumerate(cells_primary):
        p1 = points_primary[cell[0]]
        p2 = points_primary[cell[1]]
        classifications[idx + 1] = classify_element(p1, p2)
    
    scales = calculate_scales(vtk_primary, vtk_dual, sensitivity_values, config)
    print(f"  Scales: deflection={scales['deflection']:.2f}, moment={scales['moment_primary']:.6f}")
    
    os.makedirs(config.paths.plots_dir, exist_ok=True)
    
    displacement_primary = vtk_primary['point_data'].get('DISPLACEMENT', np.zeros((len(points_primary), 3)))
    rotation_primary = vtk_primary['point_data'].get('ROTATION', np.zeros((len(points_primary), 3)))
    moment_primary = vtk_primary['cell_data'].get('MOMENT', np.zeros((len(cells_primary), 3)))
    
    displacement_dual = vtk_dual['point_data'].get('DISPLACEMENT', np.zeros((len(points_dual), 3)))
    rotation_dual = vtk_dual['point_data'].get('ROTATION', np.zeros((len(points_dual), 3)))
    moment_dual = vtk_dual['cell_data'].get('MOMENT', np.zeros((len(cells_dual), 3)))
    
    moment_values_primary = {i+1: moment_primary[i][2] for i in range(len(moment_primary))}
    moment_values_dual = {i+1: moment_dual[i][2] for i in range(len(moment_dual))}
    
    show_primary_ids = get_extreme_elements(moment_values_primary, classifications)
    show_dual_ids = get_extreme_elements(moment_values_dual, classifications)
    show_sens_ids = get_extreme_elements(sensitivity_values, classifications) if sensitivity_values else set()
    
    x_min, x_max = points_primary[:, 0].min(), points_primary[:, 0].max()
    y_min, y_max = points_primary[:, 1].min(), points_primary[:, 1].max()
    x_margin = (x_max - x_min) * 0.25 if x_max > x_min else 0.5
    y_margin = max((y_max - y_min) * 0.3, 0.3)
    
    # Get actual max/min values for annotations
    max_disp_primary = np.max(np.abs(displacement_primary[:, 1]))
    max_disp_dual = np.max(np.abs(displacement_dual[:, 1]))
    max_moment_primary = np.max(np.abs(moment_primary[:, 2]))
    max_moment_dual = np.max(np.abs(moment_dual[:, 2]))
    max_rot_primary = np.max(np.abs(rotation_primary[:, 2]))
    max_rot_dual = np.max(rotation_dual[:, 2])
    min_rot_dual = np.min(rotation_dual[:, 2])
    
    # PLOT 1: Structure
    print("\n  1. Plotting structure...")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)
    ax.set_title('Structure Diagram', fontsize=14, fontweight='bold')
    plot_structure(ax, points_primary, cells_primary, show_node_labels=True, show_element_labels=True)
    add_supports(ax, points_primary, supports_primary, support_type, structure_type)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    if config.plotting.save_figures:
        plt.savefig(os.path.join(config.paths.plots_dir, 'structure.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # PLOT 2: Deflection Primary
    print("  2. Plotting deflection (primary)...")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)
    ax.set_title(f'Deflection Diagram - Primary Analysis\n(Scale: {scales["deflection"]:.0f}x)', fontsize=14, fontweight='bold')
    plot_structure(ax, points_primary, cells_primary, color='gray', linewidth=1.5)
    plot_deflection_diagram(ax, points_primary, cells_primary, displacement_primary, scales['deflection'])
    add_supports(ax, points_primary, supports_primary, support_type, structure_type)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f'Max Deflection: {max_disp_primary:.4e} m', transform=ax.transAxes, fontsize=10, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    if config.plotting.save_figures:
        plt.savefig(os.path.join(config.paths.plots_dir, 'deflection_primary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # PLOT 3: Deflection Dual
    print("  3. Plotting deflection (dual)...")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)
    ax.set_title(f'Deflection Influence Line - Dual Analysis\n(Scale: {scales["deflection_dual"]:.0f}x)', fontsize=14, fontweight='bold')
    plot_structure(ax, points_dual, cells_dual, color='gray', linewidth=1.5)
    plot_deflection_diagram(ax, points_dual, cells_dual, displacement_dual, scales['deflection_dual'],
                           color=COLORS['deflection_dual'], fill_color=COLORS['fill_deflection_dual'])
    add_supports(ax, points_dual, supports_dual, support_type, structure_type)
    add_response_circle(ax, response_location, label='Response Location')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.text(0.02, 0.98, f'Max Deflection: {max_disp_dual:.4e} m', transform=ax.transAxes, fontsize=10, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    if config.plotting.save_figures:
        plt.savefig(os.path.join(config.paths.plots_dir, 'deflection_dual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # PLOT 4: Moment Primary
    print("  4. Plotting bending moment (primary)...")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)
    ax.set_title(f'Bending Moment Diagram - Primary Analysis\n(Scale: {scales["moment_primary"]:.2e}x)', fontsize=14, fontweight='bold')
    plot_structure(ax, points_primary, cells_primary, color='gray', linewidth=1.5)
    plot_moment_diagram(ax, points_primary, cells_primary, moment_primary, scales['moment_primary'],
                       COLORS['moment_primary'], COLORS['fill_moment_primary'],
                       show_values=True, show_element_ids=show_primary_ids, classifications=classifications)
    add_supports(ax, points_primary, supports_primary, support_type, structure_type)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f'Max Moment: {max_moment_primary:.4e} N·m', transform=ax.transAxes, fontsize=10, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    if config.plotting.save_figures:
        plt.savefig(os.path.join(config.paths.plots_dir, 'moment_primary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # PLOT 5: Moment Dual
    print("  5. Plotting bending moment (dual)...")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)
    ax.set_title(f'Bending Moment Diagram - Dual Analysis\n(Scale: {scales["moment_dual"]:.2e}x)', fontsize=14, fontweight='bold')
    plot_structure(ax, points_dual, cells_dual, color='gray', linewidth=1.5)
    plot_moment_diagram(ax, points_dual, cells_dual, moment_dual, scales['moment_dual'],
                       COLORS['moment_dual'], COLORS['fill_moment_dual'],
                       show_values=True, show_element_ids=show_dual_ids, classifications=classifications)
    add_supports(ax, points_dual, supports_dual, support_type, structure_type)
    add_response_circle(ax, response_location, label='Response Location')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.text(0.02, 0.98, f'Max Moment: {max_moment_dual:.4e} N·m', transform=ax.transAxes, fontsize=10, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    if config.plotting.save_figures:
        plt.savefig(os.path.join(config.paths.plots_dir, 'moment_dual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # PLOT 6: Rotation Primary
    print("  6. Plotting rotation (primary)...")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)
    ax.set_title(f'Rotation Diagram - Primary Analysis\n(Scale: {scales["rotation"]:.0f}x)', fontsize=14, fontweight='bold')
    plot_structure(ax, points_primary, cells_primary, color='gray', linewidth=1.5)
    plot_rotation_diagram(ax, points_primary, cells_primary, rotation_primary, scales['rotation'])
    add_supports(ax, points_primary, supports_primary, support_type, structure_type)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f'Max Rotation: {max_rot_primary:.4e} rad', transform=ax.transAxes, fontsize=10, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    if config.plotting.save_figures:
        plt.savefig(os.path.join(config.paths.plots_dir, 'rotation_primary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # PLOT 7: Rotation Dual
    print("  7. Plotting rotation (dual)...")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)
    ax.set_title(f'Rotation Diagram - Dual Analysis (Unit Kink)\n(Scale: {scales["rotation_dual"]:.0f}x)', fontsize=14, fontweight='bold')
    plot_structure(ax, points_dual, cells_dual, color='gray', linewidth=1.5)
    plot_rotation_diagram(ax, points_dual, cells_dual, rotation_dual, scales['rotation_dual'],
                         color=COLORS['rotation_dual'], fill_color=COLORS['fill_rotation_dual'])
    add_supports(ax, points_dual, supports_dual, support_type, structure_type)
    add_response_circle(ax, response_location, label='Kink Location')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.text(0.02, 0.98, f'θ_left: {min_rot_dual:.4f} rad\nθ_right: {max_rot_dual:.4f} rad\nΔθ: {max_rot_dual - min_rot_dual:.4f} rad', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    if config.plotting.save_figures:
        plt.savefig(os.path.join(config.paths.plots_dir, 'rotation_dual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # PLOT 8: Sensitivity
    if sensitivity_values:
        print("  8. Plotting sensitivity...")
        max_sens = max(sensitivity_values.values())
        min_sens = min(sensitivity_values.values())
        total_sens = sum(sensitivity_values.values())
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)
        title = f'Sensitivity Diagram: ∂M_response/∂(EI)\n(Scale: {scales["sensitivity"]:.2e}x)'
        if total_sensitivity is not None:
            title += f'  |  TOTAL = {total_sensitivity:.4e}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        plot_structure(ax, points_primary, cells_primary, color='gray', linewidth=1.5)
        plot_sensitivity_diagram(ax, points_primary, cells_primary, sensitivity_values, scales['sensitivity'],
                                show_values=True, show_element_ids=show_sens_ids)
        add_supports(ax, points_primary, supports_primary, support_type, structure_type)
        add_response_circle(ax, response_location, label='Response Location')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        pos_patch = mpatches.Patch(color=COLORS['fill_sensitivity_pos'], alpha=0.4, label='Positive: ↑EI → ↑M')
        neg_patch = mpatches.Patch(color=COLORS['fill_sensitivity_neg'], alpha=0.4, label='Negative: ↑EI → ↓M')
        ax.legend(handles=[pos_patch, neg_patch], loc='upper right')
        ax.text(0.02, 0.98, f'Max: {max_sens:.4e}\nMin: {min_sens:.4e}\nTotal: {total_sens:.4e}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        if config.plotting.save_figures:
            plt.savefig(os.path.join(config.paths.plots_dir, 'sensitivity.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # PLOT 9: Combined Primary
    print("  9. Plotting combined primary...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=FIGURE_DPI)
    fig.suptitle('Primary Analysis Results', fontsize=14, fontweight='bold')
    
    ax = axes[0, 0]
    ax.set_title('Structure')
    plot_structure(ax, points_primary, cells_primary, show_element_labels=True)
    add_supports(ax, points_primary, supports_primary, support_type, structure_type)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.set_title(f'Deflection (Max: {max_disp_primary:.4e} m)')
    plot_structure(ax, points_primary, cells_primary, color='gray', linewidth=1.5)
    plot_deflection_diagram(ax, points_primary, cells_primary, displacement_primary, scales['deflection'])
    add_supports(ax, points_primary, supports_primary, support_type, structure_type)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.set_title(f'Bending Moment (Max: {max_moment_primary:.4e} N·m)')
    plot_structure(ax, points_primary, cells_primary, color='gray', linewidth=1.5)
    plot_moment_diagram(ax, points_primary, cells_primary, moment_primary, scales['moment_primary'],
                       COLORS['moment_primary'], COLORS['fill_moment_primary'],
                       show_values=True, show_element_ids=show_primary_ids, classifications=classifications)
    add_supports(ax, points_primary, supports_primary, support_type, structure_type)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.set_title(f'Rotation (Max: {max_rot_primary:.4e} rad)')
    plot_structure(ax, points_primary, cells_primary, color='gray', linewidth=1.5)
    plot_rotation_diagram(ax, points_primary, cells_primary, rotation_primary, scales['rotation'])
    add_supports(ax, points_primary, supports_primary, support_type, structure_type)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if config.plotting.save_figures:
        plt.savefig(os.path.join(config.paths.plots_dir, 'combined_primary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # PLOT 10: Combined Dual
    print("  10. Plotting combined dual...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=FIGURE_DPI)
    fig.suptitle('Dual Analysis Results', fontsize=14, fontweight='bold')
    
    ax = axes[0, 0]
    ax.set_title('Structure with Response Location')
    plot_structure(ax, points_dual, cells_dual, show_element_labels=True)
    add_supports(ax, points_dual, supports_dual, support_type, structure_type)
    add_response_circle(ax, response_location, label='Response Location')
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    
    ax = axes[0, 1]
    ax.set_title(f'Deflection (Max: {max_disp_dual:.4e} m)')
    plot_structure(ax, points_dual, cells_dual, color='gray', linewidth=1.5)
    plot_deflection_diagram(ax, points_dual, cells_dual, displacement_dual, scales['deflection_dual'],
                           color=COLORS['deflection_dual'], fill_color=COLORS['fill_deflection_dual'])
    add_supports(ax, points_dual, supports_dual, support_type, structure_type)
    add_response_circle(ax, response_location, label=None)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.set_title(f'Bending Moment (Max: {max_moment_dual:.4e} N·m)')
    plot_structure(ax, points_dual, cells_dual, color='gray', linewidth=1.5)
    plot_moment_diagram(ax, points_dual, cells_dual, moment_dual, scales['moment_dual'],
                       COLORS['moment_dual'], COLORS['fill_moment_dual'],
                       show_values=True, show_element_ids=show_dual_ids, classifications=classifications)
    add_supports(ax, points_dual, supports_dual, support_type, structure_type)
    add_response_circle(ax, response_location, label=None)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.set_title(f'Rotation (Δθ: {max_rot_dual - min_rot_dual:.4f} rad)')
    plot_structure(ax, points_dual, cells_dual, color='gray', linewidth=1.5)
    plot_rotation_diagram(ax, points_dual, cells_dual, rotation_dual, scales['rotation_dual'],
                         color=COLORS['rotation_dual'], fill_color=COLORS['fill_rotation_dual'])
    add_supports(ax, points_dual, supports_dual, support_type, structure_type)
    add_response_circle(ax, response_location, label=None)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if config.plotting.save_figures:
        plt.savefig(os.path.join(config.paths.plots_dir, 'combined_dual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # PLOT 11: Combined Sensitivity
    if sensitivity_values:
        print("  11. Plotting combined sensitivity...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=FIGURE_DPI)
        fig.suptitle('Sensitivity Analysis: ∂M_response/∂(EI)', fontsize=14, fontweight='bold')
        
        ax = axes[0]
        ax.set_title(f'Primary Moment M(x)\n(Max: {max_moment_primary:.4e} N·m)')
        plot_structure(ax, points_primary, cells_primary, color='gray', linewidth=1.5)
        plot_moment_diagram(ax, points_primary, cells_primary, moment_primary, scales['moment_primary'],
                           COLORS['moment_primary'], COLORS['fill_moment_primary'],
                           show_values=True, show_element_ids=show_primary_ids, classifications=classifications)
        add_supports(ax, points_primary, supports_primary, support_type, structure_type)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        ax.set_title(f'Dual Moment M̄(x)\n(Max: {max_moment_dual:.4e} N·m)')
        plot_structure(ax, points_dual, cells_dual, color='gray', linewidth=1.5)
        plot_moment_diagram(ax, points_dual, cells_dual, moment_dual, scales['moment_dual'],
                           COLORS['moment_dual'], COLORS['fill_moment_dual'],
                           show_values=True, show_element_ids=show_dual_ids, classifications=classifications)
        add_supports(ax, points_dual, supports_dual, support_type, structure_type)
        add_response_circle(ax, response_location, label=None)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        ax = axes[2]
        total_sens = sum(sensitivity_values.values())
        ax.set_title(f'Sensitivity ∂M_response/∂(EI)\n(Total: {total_sens:.4e})', color='#8B0000', fontweight='bold')
        plot_structure(ax, points_primary, cells_primary, color='gray', linewidth=1.5)
        plot_sensitivity_diagram(ax, points_primary, cells_primary, sensitivity_values, scales['sensitivity'],
                                show_values=True, show_element_ids=show_sens_ids)
        add_supports(ax, points_primary, supports_primary, support_type, structure_type)
        add_response_circle(ax, response_location, label=None)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if config.plotting.save_figures:
            plt.savefig(os.path.join(config.paths.plots_dir, 'combined_sensitivity.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n  Plots saved to: {config.paths.plots_dir}")
    print("=" * 70)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def generate_plots(config: Config) -> None:
    """Main function: Generate all plots from VTK results."""
    
    if not config.plotting.enabled:
        print("Plotting disabled in config. Skipping.")
        return
    
    print("\nLoading VTK files for plotting...")
    
    if not os.path.exists(config.paths.vtk_primary):
        print(f"Error: Primary VTK not found: {config.paths.vtk_primary}")
        return
    
    if not os.path.exists(config.paths.vtk_dual):
        print(f"Error: Dual VTK not found: {config.paths.vtk_dual}")
        return
    
    vtk_primary = parse_vtk_file(config.paths.vtk_primary)
    vtk_dual = parse_vtk_file(config.paths.vtk_dual)
    
    sensitivity_values = None
    total_sensitivity = None
    
    if os.path.exists(config.paths.sa_results_json):
        print(f"Loading sensitivity results from: {config.paths.sa_results_json}")
        with open(config.paths.sa_results_json, 'r') as f:
            sa_data = json.load(f)
        
        sensitivity_values = {
            elem['element_id']: elem['dM_dEI']
            for elem in sa_data.get('elements', [])
        }
        total_sensitivity = sa_data.get('summary', {}).get('total_sensitivity')
    
    create_all_plots(config, vtk_primary, vtk_dual, sensitivity_values, total_sensitivity)


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    try:
        config = load_config()
        generate_plots(config)
        print("\nPlotting completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)