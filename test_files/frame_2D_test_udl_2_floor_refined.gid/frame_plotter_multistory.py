"""
General Frame Analysis Results Plotter with Analytical Verification
Handles single-story, multi-story, and braced frames
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import os


@dataclass
class VTKData:
    """Container for VTK file data"""
    points: np.ndarray = None
    cells: List[List[int]] = field(default_factory=list)
    point_data: Dict[str, np.ndarray] = field(default_factory=dict)
    cell_data: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Member:
    """Represents a frame member"""
    id: int
    node_i: int
    node_j: int
    start_coords: np.ndarray
    end_coords: np.ndarray
    length: float
    angle: float  # radians from horizontal
    orientation: str  # 'horizontal', 'vertical', 'inclined'
    member_type: str  # 'column', 'beam', 'brace'
    level: int = 0  # Floor level (0 = ground, 1 = first floor, etc.)


@dataclass
class FrameParameters:
    """Frame properties for analysis"""
    E: float = 210e9          # Young's modulus [Pa]
    I: float = 5e-6           # Moment of inertia [m^4]
    A: float = 0.00287        # Cross-sectional area [m^2]
    w: float = 10000.0        # UDL [N/m]
    n_stories: int = 1        # Number of stories
    story_heights: List[float] = None  # Height of each story
    bay_widths: List[float] = None     # Width of each bay


@dataclass
class FrameGeometry:
    """Detected frame geometry"""
    n_nodes: int
    n_members: int
    n_stories: int
    n_bays: int
    story_levels: List[float]  # Y-coordinates of each floor
    column_lines: List[float]  # X-coordinates of column lines
    span: float
    total_height: float
    members_by_type: Dict[str, List[int]]


def parse_vtk_file(filepath: str) -> VTKData:
    """Parse a VTK ASCII file"""
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data = VTKData()
    i = 0
    current_section = None
    
    while i < len(lines):
        line = lines[i].strip()
        
        if not line or line.startswith('#'):
            i += 1
            continue
        
        if line.startswith('POINTS'):
            parts = line.split()
            n_points = int(parts[1])
            points = []
            i += 1
            while len(points) < n_points * 3 and i < len(lines):
                values = lines[i].strip().split()
                points.extend([float(v) for v in values if v])
                i += 1
            data.points = np.array(points).reshape(n_points, 3)
            continue
        
        if line.startswith('CELLS'):
            parts = line.split()
            n_cells = int(parts[1])
            cells = []
            i += 1
            for _ in range(n_cells):
                if i < len(lines):
                    cell_values = [int(v) for v in lines[i].strip().split()]
                    cells.append(cell_values[1:])
                    i += 1
            data.cells = cells
            continue
        
        if line.startswith('POINT_DATA'):
            current_section = 'point'
            i += 1
            continue
        
        if line.startswith('CELL_DATA'):
            current_section = 'cell'
            i += 1
            continue
        
        if line.startswith('FIELD'):
            n_arrays = int(line.split()[-1])
            i += 1
            
            for _ in range(n_arrays):
                if i >= len(lines):
                    break
                
                field_line = lines[i].strip().split()
                if len(field_line) < 4:
                    i += 1
                    continue
                
                field_name = field_line[0]
                n_components = int(field_line[1])
                n_tuples = int(field_line[2])
                
                i += 1
                values = []
                while len(values) < n_components * n_tuples and i < len(lines):
                    val_line = lines[i].strip()
                    if val_line and not any(val_line.startswith(k) for k in 
                                           ['FIELD', 'POINT_DATA', 'CELL_DATA', 'CELL_TYPES']):
                        try:
                            values.extend([float(v) for v in val_line.split() if v])
                            i += 1
                        except ValueError:
                            break
                    else:
                        break
                
                if len(values) == n_components * n_tuples:
                    arr = np.array(values).reshape(n_tuples, n_components)
                    if current_section == 'point':
                        data.point_data[field_name] = arr
                    elif current_section == 'cell':
                        data.cell_data[field_name] = arr
            continue
        
        i += 1
    
    return data


def analyze_frame_geometry(vtk_data: VTKData) -> Tuple[List[Member], FrameGeometry]:
    """Analyze frame geometry and identify members, stories, bays"""
    
    points = vtk_data.points
    cells = vtk_data.cells
    
    # Find unique Y-levels (story levels)
    y_coords = np.unique(np.round(points[:, 1], decimals=6))
    story_levels = sorted(y_coords.tolist())
    n_stories = len(story_levels) - 1  # Number of stories = levels - 1
    
    # Find unique X-coordinates (column lines)
    x_coords = np.unique(np.round(points[:, 0], decimals=6))
    column_lines = sorted(x_coords.tolist())
    n_bays = len(column_lines) - 1
    
    # Frame bounds
    x_min, x_max = min(column_lines), max(column_lines)
    y_min, y_max = min(story_levels), max(story_levels)
    
    members = []
    members_by_type = {'column': [], 'beam': [], 'brace': []}
    
    for idx, cell in enumerate(cells):
        node_i, node_j = cell[0], cell[1]
        
        start = points[node_i].copy()
        end = points[node_j].copy()
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        angle = np.arctan2(dy, dx)
        
        # Determine member type
        if abs(dx) < 1e-6 and abs(dy) > 1e-6:
            orientation = 'vertical'
            member_type = 'column'
            
            # Determine which story this column belongs to
            y_mid = (start[1] + end[1]) / 2
            level = 0
            for i, (y_low, y_high) in enumerate(zip(story_levels[:-1], story_levels[1:])):
                if y_low <= y_mid <= y_high:
                    level = i + 1
                    break
                    
        elif abs(dy) < 1e-6 and abs(dx) > 1e-6:
            orientation = 'horizontal'
            member_type = 'beam'
            
            # Determine which level this beam is at
            level = 0
            for i, y_level in enumerate(story_levels):
                if abs(start[1] - y_level) < 1e-6:
                    level = i
                    break
        else:
            orientation = 'inclined'
            member_type = 'brace'
            level = 0
        
        member = Member(
            id=idx + 1,
            node_i=node_i,
            node_j=node_j,
            start_coords=start,
            end_coords=end,
            length=length,
            angle=angle,
            orientation=orientation,
            member_type=member_type,
            level=level
        )
        
        members.append(member)
        members_by_type[member_type].append(idx + 1)
    
    geometry = FrameGeometry(
        n_nodes=len(points),
        n_members=len(members),
        n_stories=n_stories,
        n_bays=n_bays,
        story_levels=story_levels,
        column_lines=column_lines,
        span=x_max - x_min,
        total_height=y_max - y_min,
        members_by_type=members_by_type
    )
    
    return members, geometry


class MultiStoryFrameAnalytical:
    """Analytical solutions for multi-story frames"""
    
    @staticmethod
    def simply_supported_beam_udl(L: float, w: float, E: float, I: float, x: np.ndarray = None) -> Dict:
        """Simply supported beam with UDL"""
        
        if x is None:
            x = np.linspace(0, L, 100)
        
        moment = w * L * x / 2 - w * x**2 / 2
        shear = w * L / 2 - w * x
        deflection = w * x * (L**3 - 2*L*x**2 + x**3) / (24 * E * I)
        
        return {
            'x': x,
            'moment': moment,
            'shear': shear,
            'deflection': deflection,
            'M_max': w * L**2 / 8,
            'V_max': w * L / 2,
            'delta_max': 5 * w * L**4 / (384 * E * I),
            'reaction': w * L / 2
        }
    
    @staticmethod
    def get_frame_reactions(geometry: FrameGeometry, w: float) -> Dict:
        """Calculate approximate reactions for multi-story frame"""
        
        # For symmetric frame with symmetric loading
        # Each column line carries load from tributary width
        
        n_bays = geometry.n_bays
        n_stories = geometry.n_stories
        span = geometry.span / n_bays if n_bays > 0 else geometry.span
        
        # Total load per floor = w * span * n_bays
        total_load_per_floor = w * geometry.span
        
        # Total load on frame
        total_load = total_load_per_floor * n_stories
        
        # For pinned base, reactions at each support
        n_supports = len(geometry.column_lines)
        
        # Interior columns take more load than exterior
        if n_supports == 2:
            reactions = [total_load / 2, total_load / 2]
        else:
            # Exterior columns take half the tributary width
            reactions = []
            for i in range(n_supports):
                if i == 0 or i == n_supports - 1:
                    reactions.append(total_load / (2 * (n_supports - 1)))
                else:
                    reactions.append(total_load / (n_supports - 1))
        
        return {
            'reactions': reactions,
            'total_load': total_load,
            'load_per_floor': total_load_per_floor
        }


def print_frame_results(vtk_data: VTKData, members: List[Member], 
                        geometry: FrameGeometry, params: FrameParameters):
    """Print comprehensive frame analysis results"""
    
    print("\n" + "=" * 120)
    print("MULTI-STORY FRAME ANALYSIS RESULTS")
    print("=" * 120)
    
    # Frame geometry summary
    print(f"\n{'FRAME GEOMETRY SUMMARY':^120}")
    print("-" * 120)
    print(f"  Total Nodes:        {geometry.n_nodes}")
    print(f"  Total Members:      {geometry.n_members}")
    print(f"  Number of Stories:  {geometry.n_stories}")
    print(f"  Number of Bays:     {geometry.n_bays}")
    print(f"  Total Span:         {geometry.span:.4f} m")
    print(f"  Total Height:       {geometry.total_height:.4f} m")
    print(f"\n  Story Levels (Y):   {[f'{y:.2f}' for y in geometry.story_levels]}")
    print(f"  Column Lines (X):   {[f'{x:.2f}' for x in geometry.column_lines]}")
    
    # Member counts
    print(f"\n  Columns: {len(geometry.members_by_type['column'])}")
    print(f"  Beams:   {len(geometry.members_by_type['beam'])}")
    print(f"  Braces:  {len(geometry.members_by_type['brace'])}")
    
    # Member details
    print(f"\n{'MEMBER DETAILS':^120}")
    print("-" * 120)
    print(f"{'Member':<8} {'Type':<12} {'Node i':<8} {'Node j':<8} {'Length(m)':<12} {'Level':<8} {'Angle(°)':<12}")
    print("-" * 120)
    
    for m in members:
        angle_deg = np.degrees(m.angle)
        print(f"{m.id:<8} {m.member_type:<12} {m.node_i + 1:<8} {m.node_j + 1:<8} "
              f"{m.length:<12.4f} {m.level:<8} {angle_deg:<12.2f}")
    
    print("=" * 120)
    
    # Nodal coordinates
    print(f"\n{'NODAL COORDINATES':^120}")
    print("-" * 120)
    print(f"{'Node':<8} {'X (m)':<15} {'Y (m)':<15} {'Z (m)':<15} {'Level':<10}")
    print("-" * 120)
    
    for i, point in enumerate(vtk_data.points):
        # Determine level
        level = 0
        for j, y_level in enumerate(geometry.story_levels):
            if abs(point[1] - y_level) < 1e-6:
                level = j
                break
        print(f"{i + 1:<8} {point[0]:<15.6f} {point[1]:<15.6f} {point[2]:<15.6f} {level:<10}")
    
    print("=" * 120)
    
    # Displacements
    if 'DISPLACEMENT' in vtk_data.point_data:
        print(f"\n{'NODAL DISPLACEMENTS':^120}")
        print("-" * 120)
        print(f"{'Node':<8} {'Ux (mm)':<18} {'Uy (mm)':<18} {'Uz (mm)':<18} {'|U| (mm)':<18}")
        print("-" * 120)
        
        disp = vtk_data.point_data['DISPLACEMENT']
        for i in range(len(vtk_data.points)):
            ux, uy, uz = disp[i] * 1000
            u_mag = np.sqrt(ux**2 + uy**2 + uz**2)
            print(f"{i + 1:<8} {ux:<18.6f} {uy:<18.6f} {uz:<18.6f} {u_mag:<18.6f}")
        
        # Max displacement
        max_disp_idx = np.argmax(np.abs(disp[:, 1]))
        max_disp = np.abs(disp[max_disp_idx, 1]) * 1000
        print(f"\n  Maximum vertical displacement: {max_disp:.6f} mm at Node {max_disp_idx + 1}")
        
    print("=" * 120)
    
    # Rotations
    if 'ROTATION' in vtk_data.point_data:
        print(f"\n{'NODAL ROTATIONS':^120}")
        print("-" * 120)
        print(f"{'Node':<8} {'Rx (mrad)':<18} {'Ry (mrad)':<18} {'Rz (mrad)':<18}")
        print("-" * 120)
        
        rot = vtk_data.point_data['ROTATION']
        for i in range(len(vtk_data.points)):
            rx, ry, rz = rot[i] * 1000
            print(f"{i + 1:<8} {rx:<18.6f} {ry:<18.6f} {rz:<18.6f}")
    
    print("=" * 120)
    
    # Support reactions
    if 'REACTION' in vtk_data.point_data:
        print(f"\n{'SUPPORT REACTIONS':^120}")
        print("-" * 120)
        print(f"{'Node':<8} {'X (m)':<12} {'Y (m)':<12} {'Rx (kN)':<15} {'Ry (kN)':<15} {'Rz (kN)':<15}")
        print("-" * 120)
        
        reactions = vtk_data.point_data['REACTION']
        total_ry = 0
        for i in range(len(vtk_data.points)):
            rx, ry, rz = reactions[i] / 1000
            if abs(rx) > 0.001 or abs(ry) > 0.001 or abs(rz) > 0.001:
                point = vtk_data.points[i]
                print(f"{i + 1:<8} {point[0]:<12.4f} {point[1]:<12.4f} {rx:<15.4f} {ry:<15.4f} {rz:<15.4f}")
                total_ry += ry
        
        print(f"\n  Total vertical reaction: {total_ry:.4f} kN")
        
        # Expected total load
        n_loaded_beams = len(geometry.members_by_type['beam'])
        expected_load = params.w * geometry.span * n_loaded_beams / 1000
        print(f"  Expected total load: {expected_load:.4f} kN")
        
    print("=" * 120)
    
    # Member forces
    if 'FORCE' in vtk_data.point_data:
        print(f"\n{'MEMBER END FORCES (at nodes)':^120}")
        print("-" * 120)
        print(f"{'Node':<8} {'Fx (kN)':<18} {'Fy (kN)':<18} {'Fz (kN)':<18}")
        print("-" * 120)
        
        forces = vtk_data.point_data['FORCE']
        for i in range(len(vtk_data.points)):
            fx, fy, fz = forces[i] / 1000
            print(f"{i + 1:<8} {fx:<18.4f} {fy:<18.4f} {fz:<18.4f}")
    
    print("=" * 120)
    
    # Member moments
    if 'MOMENT' in vtk_data.point_data:
        print(f"\n{'MEMBER END MOMENTS (at nodes)':^120}")
        print("-" * 120)
        print(f"{'Node':<8} {'Mx (kN·m)':<18} {'My (kN·m)':<18} {'Mz (kN·m)':<18}")
        print("-" * 120)
        
        moments = vtk_data.point_data['MOMENT']
        for i in range(len(vtk_data.points)):
            mx, my, mz = moments[i] / 1000
            print(f"{i + 1:<8} {mx:<18.6f} {my:<18.6f} {mz:<18.6f}")
    
    print("=" * 120)
    
    # Analytical comparison for beams
    print(f"\n{'ANALYTICAL COMPARISON (Simply Supported Beams)':^120}")
    print("-" * 120)
    
    beam_span = geometry.span
    anal = MultiStoryFrameAnalytical.simply_supported_beam_udl(
        beam_span, params.w, params.E, params.I
    )
    
    print(f"  For each beam with UDL = {params.w/1000:.2f} kN/m, Span = {beam_span:.2f} m:")
    print(f"    Max Moment (analytical):     {anal['M_max']/1000:.4f} kN·m")
    print(f"    Max Shear (analytical):      {anal['V_max']/1000:.4f} kN")
    print(f"    Max Deflection (analytical): {anal['delta_max']*1000:.6f} mm")
    print(f"    End Reactions (analytical):  {anal['reaction']/1000:.4f} kN each")
    
    print("=" * 120)


def plot_frame_geometry(vtk_data: VTKData, members: List[Member], 
                        geometry: FrameGeometry, ax=None, 
                        show_labels: bool = True):
    """Plot frame geometry with member and node labels"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = {
        'column': '#1D3557',
        'beam': '#2E86AB',
        'brace': '#457B9D'
    }
    
    # Plot members
    for m in members:
        x = [m.start_coords[0], m.end_coords[0]]
        y = [m.start_coords[1], m.end_coords[1]]
        
        color = colors.get(m.member_type, 'gray')
        lw = 4 if m.member_type in ['column', 'beam'] else 2
        ls = '-' if m.member_type != 'brace' else '--'
        
        ax.plot(x, y, color=color, linewidth=lw, linestyle=ls)
        
        if show_labels:
            mid_x = (m.start_coords[0] + m.end_coords[0]) / 2
            mid_y = (m.start_coords[1] + m.end_coords[1]) / 2
            
            # Offset label based on orientation
            if m.orientation == 'vertical':
                offset = (-0.2, 0)
            elif m.orientation == 'horizontal':
                offset = (0, 0.15)
            else:
                offset = (0.1, 0.1)
            
            ax.annotate(f'M{m.id}', (mid_x + offset[0], mid_y + offset[1]), 
                       fontsize=9, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))
    
    # Plot nodes
    for i, point in enumerate(vtk_data.points):
        circle = Circle((point[0], point[1]), 0.08, facecolor='white', 
                        edgecolor='black', linewidth=2, zorder=5)
        ax.add_patch(circle)
        
        if show_labels:
            ax.annotate(f'{i + 1}', (point[0] + 0.15, point[1] + 0.15),
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='circle,pad=0.2', facecolor='yellow', alpha=0.9))
    
    # Plot supports (at ground level)
    support_size = 0.12
    for i, point in enumerate(vtk_data.points):
        if abs(point[1] - geometry.story_levels[0]) < 1e-6:
            # Draw pinned support
            triangle = plt.Polygon([
                (point[0], point[1]),
                (point[0] - support_size, point[1] - support_size * 1.5),
                (point[0] + support_size, point[1] - support_size * 1.5)
            ], facecolor='#1D3557', edgecolor='black', linewidth=2, zorder=4)
            ax.add_patch(triangle)
            
            # Ground line
            ax.plot([point[0] - 0.2, point[0] + 0.2], 
                   [point[1] - support_size * 1.5 - 0.02] * 2, 'k-', linewidth=2)
    
    # Story level lines (dashed)
    for y_level in geometry.story_levels[1:]:
        ax.axhline(y=y_level, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors['column'], linewidth=4, label='Column'),
        Line2D([0], [0], color=colors['beam'], linewidth=4, label='Beam'),
        Line2D([0], [0], color=colors['brace'], linewidth=2, linestyle='--', label='Brace'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    return ax


def plot_deformed_shape(vtk_data: VTKData, members: List[Member], 
                        geometry: FrameGeometry, scale: float = 1000, ax=None):
    """Plot deformed shape with amplification"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    if 'DISPLACEMENT' not in vtk_data.point_data:
        ax.text(0.5, 0.5, 'No displacement data', transform=ax.transAxes,
               ha='center', va='center', fontsize=14)
        return ax
    
    disp = vtk_data.point_data['DISPLACEMENT']
    
    # Plot undeformed shape
    for m in members:
        x = [m.start_coords[0], m.end_coords[0]]
        y = [m.start_coords[1], m.end_coords[1]]
        ax.plot(x, y, 'k--', linewidth=1, alpha=0.4)
    
    # Calculate deformed coordinates
    deformed = vtk_data.points.copy()
    deformed[:, 0] += disp[:, 0] * scale
    deformed[:, 1] += disp[:, 1] * scale
    
    # Plot deformed shape
    for m in members:
        x = [deformed[m.node_i, 0], deformed[m.node_j, 0]]
        y = [deformed[m.node_i, 1], deformed[m.node_j, 1]]
        
        ax.plot(x, y, 'r-', linewidth=2.5)
    
    # Plot deformed nodes
    for i in range(len(vtk_data.points)):
        ax.plot(deformed[i, 0], deformed[i, 1], 'ro', markersize=8,
               markerfacecolor='white', markeredgewidth=2)
    
    # Max displacement annotation
    max_disp_idx = np.argmax(np.abs(disp[:, 1]))
    max_disp = disp[max_disp_idx, 1] * 1000
    
    ax.text(0.02, 0.98, f'Scale: {scale}x\nMax Uy: {max_disp:.4f} mm',
           transform=ax.transAxes, fontsize=10, va='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('Deformed Shape', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return ax


def plot_axial_force_diagram(vtk_data: VTKData, members: List[Member],
                              geometry: FrameGeometry, scale: float = 0.00005, ax=None):
    """Plot axial force diagram"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    if 'FORCE' not in vtk_data.point_data:
        ax.text(0.5, 0.5, 'No force data', transform=ax.transAxes,
               ha='center', va='center', fontsize=14)
        return ax
    
    forces = vtk_data.point_data['FORCE']
    
    # Plot frame outline
    for m in members:
        x = [m.start_coords[0], m.end_coords[0]]
        y = [m.start_coords[1], m.end_coords[1]]
        ax.plot(x, y, 'k-', linewidth=2)
    
    # Plot axial force diagrams
    for m in members:
        fi = forces[m.node_i]
        fj = forces[m.node_j]
        
        # Calculate axial force (along member axis)
        cos_a = np.cos(m.angle)
        sin_a = np.sin(m.angle)
        
        axial_i = fi[0] * cos_a + fi[1] * sin_a
        axial_j = fj[0] * cos_a + fj[1] * sin_a
        
        # Normal direction
        nx, ny = -sin_a, cos_a
        
        x1, y1 = m.start_coords[0], m.start_coords[1]
        x2, y2 = m.end_coords[0], m.end_coords[1]
        
        x1_off = x1 + axial_i * scale * nx
        y1_off = y1 + axial_i * scale * ny
        x2_off = x2 + axial_j * scale * nx
        y2_off = y2 + axial_j * scale * ny
        
        # Color based on tension/compression
        avg_axial = (axial_i + axial_j) / 2
        color = '#E63946' if avg_axial < 0 else '#2E86AB'
        
        polygon_x = [x1, x1_off, x2_off, x2, x1]
        polygon_y = [y1, y1_off, y2_off, y2, y1]
        ax.fill(polygon_x, polygon_y, color=color, alpha=0.3, edgecolor=color, linewidth=1)
        
        # Value label
        if abs(avg_axial) > 100:
            mid_x = (x1_off + x2_off) / 2
            mid_y = (y1_off + y2_off) / 2
            ax.text(mid_x, mid_y, f'{avg_axial/1000:.1f}', fontsize=8, ha='center', va='center')
    
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('Axial Force Diagram (kN)\nRed: Compression, Blue: Tension', 
                fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return ax


def plot_bending_moment_diagram(vtk_data: VTKData, members: List[Member],
                                 geometry: FrameGeometry, scale: float = 0.0005, ax=None):
    """Plot bending moment diagram"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    if 'MOMENT' not in vtk_data.point_data:
        ax.text(0.5, 0.5, 'No moment data', transform=ax.transAxes,
               ha='center', va='center', fontsize=14)
        return ax
    
    moments = vtk_data.point_data['MOMENT']
    
    # Plot frame outline
    for m in members:
        x = [m.start_coords[0], m.end_coords[0]]
        y = [m.start_coords[1], m.end_coords[1]]
        ax.plot(x, y, 'k-', linewidth=2)
    
    # Plot moment diagrams
    for m in members:
        mz_i = moments[m.node_i, 2]
        mz_j = moments[m.node_j, 2]
        
        # Normal direction (perpendicular to member)
        cos_a = np.cos(m.angle)
        sin_a = np.sin(m.angle)
        nx, ny = -sin_a, cos_a
        
        x1, y1 = m.start_coords[0], m.start_coords[1]
        x2, y2 = m.end_coords[0], m.end_coords[1]
        
        x1_off = x1 + mz_i * scale * nx
        y1_off = y1 + mz_i * scale * ny
        x2_off = x2 + mz_j * scale * nx
        y2_off = y2 + mz_j * scale * ny
        
        color = '#F18F01'
        
        polygon_x = [x1, x1_off, x2_off, x2, x1]
        polygon_y = [y1, y1_off, y2_off, y2, y1]
        ax.fill(polygon_x, polygon_y, color=color, alpha=0.4, edgecolor=color, linewidth=1)
        
        # Value labels
        if abs(mz_i) > 0.1:
            ax.text(x1_off, y1_off, f'{mz_i/1000:.2f}', fontsize=8, ha='center', va='center')
        if abs(mz_j) > 0.1:
            ax.text(x2_off, y2_off, f'{mz_j/1000:.2f}', fontsize=8, ha='center', va='center')
    
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('Bending Moment Diagram (kN·m)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return ax


def plot_shear_force_diagram(vtk_data: VTKData, members: List[Member],
                              geometry: FrameGeometry, scale: float = 0.0001, ax=None):
    """Plot shear force diagram"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    if 'FORCE' not in vtk_data.point_data:
        ax.text(0.5, 0.5, 'No force data', transform=ax.transAxes,
               ha='center', va='center', fontsize=14)
        return ax
    
    forces = vtk_data.point_data['FORCE']
    
    # Plot frame outline
    for m in members:
        x = [m.start_coords[0], m.end_coords[0]]
        y = [m.start_coords[1], m.end_coords[1]]
        ax.plot(x, y, 'k-', linewidth=2)
    
    # Plot shear diagrams
    for m in members:
        fi = forces[m.node_i]
        fj = forces[m.node_j]
        
        cos_a = np.cos(m.angle)
        sin_a = np.sin(m.angle)
        
        # Shear = perpendicular component
        shear_i = -fi[0] * sin_a + fi[1] * cos_a
        shear_j = -fj[0] * sin_a + fj[1] * cos_a
        
        nx, ny = -sin_a, cos_a
        
        x1, y1 = m.start_coords[0], m.start_coords[1]
        x2, y2 = m.end_coords[0], m.end_coords[1]
        
        x1_off = x1 + shear_i * scale * nx
        y1_off = y1 + shear_i * scale * ny
        x2_off = x2 + shear_j * scale * nx
        y2_off = y2 + shear_j * scale * ny
        
        color = '#A23B72'
        
        polygon_x = [x1, x1_off, x2_off, x2, x1]
        polygon_y = [y1, y1_off, y2_off, y2, y1]
        ax.fill(polygon_x, polygon_y, color=color, alpha=0.4, edgecolor=color, linewidth=1)
    
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('Shear Force Diagram (kN)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return ax


def plot_frame_complete(vtk_data: VTKData, members: List[Member], 
                        geometry: FrameGeometry, output_prefix: str = "frame"):
    """Create complete frame analysis plot"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Frame Analysis Results\n({geometry.n_stories} Story, {geometry.n_bays} Bay Frame)', 
                 fontsize=14, fontweight='bold')
    
    # 1. Deformed Shape
    plot_deformed_shape(vtk_data, members, geometry, scale=500, ax=axes[0, 0])
    
    # 2. Axial Force
    plot_axial_force_diagram(vtk_data, members, geometry, scale=0.00003, ax=axes[0, 1])
    
    # 3. Bending Moment
    plot_bending_moment_diagram(vtk_data, members, geometry, scale=0.0003, ax=axes[1, 0])
    
    # 4. Shear Force
    plot_shear_force_diagram(vtk_data, members, geometry, scale=0.00005, ax=axes[1, 1])
    
    plt.tight_layout()
    
    output_file = f'{output_prefix}_diagrams.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return fig


def plot_frame_with_loads(vtk_data: VTKData, members: List[Member], 
                          geometry: FrameGeometry, params: FrameParameters,
                          output_prefix: str = "frame"):
    """Plot frame system diagram with loads and reactions"""
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot frame geometry
    plot_frame_geometry(vtk_data, members, geometry, ax=ax)
    
    # Add UDL arrows on beams
    for m in members:
        if m.member_type == 'beam':
            n_arrows = 8
            x_start, x_end = m.start_coords[0], m.end_coords[0]
            y = m.start_coords[1]
            
            arrow_height = 0.25
            
            for i in range(n_arrows + 1):
                x = x_start + i * (x_end - x_start) / n_arrows
                ax.annotate('', xy=(x, y + 0.05), xytext=(x, y + arrow_height),
                           arrowprops=dict(arrowstyle='->', color='#E63946', lw=1))
            
            # UDL line
            ax.plot([x_start, x_end], [y + arrow_height, y + arrow_height], 
                   color='#E63946', linewidth=2)
            
            # Label
            ax.text((x_start + x_end) / 2, y + arrow_height + 0.1,
                   f'w = {params.w/1000:.1f} kN/m', fontsize=10, ha='center',
                   color='#E63946', fontweight='bold')
    
    # Reactions
    if 'REACTION' in vtk_data.point_data:
        reactions = vtk_data.point_data['REACTION']
        
        for i, point in enumerate(vtk_data.points):
            ry = reactions[i, 1]
            if abs(ry) > 100:  # Significant reaction
                ax.annotate('', xy=(point[0], point[1] - 0.1),
                           xytext=(point[0], point[1] - 0.5),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2))
                ax.text(point[0] + 0.15, point[1] - 0.4,
                       f'Ry={ry/1000:.1f}kN', fontsize=9, color='green')
    
    ax.set_title(f'{geometry.n_stories}-Story Frame with UDL Loading', 
                fontsize=14, fontweight='bold')
    
    # Adjust limits
    margin = 0.8
    ax.set_xlim(geometry.column_lines[0] - margin, geometry.column_lines[-1] + margin)
    ax.set_ylim(geometry.story_levels[0] - margin, geometry.story_levels[-1] + margin)
    
    plt.tight_layout()
    
    output_file = f'{output_prefix}_system.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return fig


def plot_analytical_comparison(vtk_data: VTKData, members: List[Member],
                                geometry: FrameGeometry, params: FrameParameters,
                                output_prefix: str = "frame"):
    """Plot FEM vs analytical comparison for beams"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Beam Analysis: FEM vs Analytical (Simply Supported)', 
                fontsize=14, fontweight='bold')
    
    # Get analytical solution for beam
    beam_span = geometry.span
    x_anal = np.linspace(0, beam_span, 100)
    anal = MultiStoryFrameAnalytical.simply_supported_beam_udl(
        beam_span, params.w, params.E, params.I, x_anal
    )
    
    # Get beam members and their node data
    beam_members = [m for m in members if m.member_type == 'beam']
    
    colors = {'analytical': '#1D3557', 'fem': '#E63946'}
    
    # 1. Deflection comparison
    ax1 = axes[0, 0]
    ax1.plot(x_anal, -anal['deflection'] * 1000, '-', color=colors['analytical'],
            linewidth=2, label='Analytical')
    
    if 'DISPLACEMENT' in vtk_data.point_data:
        disp = vtk_data.point_data['DISPLACEMENT']
        
        for bm in beam_members:
            # Get nodes on this beam
            beam_nodes = []
            y_level = bm.start_coords[1]
            
            for i, point in enumerate(vtk_data.points):
                if abs(point[1] - y_level) < 1e-6:
                    beam_nodes.append((point[0], i))
            
            beam_nodes.sort(key=lambda x: x[0])
            
            if beam_nodes:
                x_fem = [n[0] for n in beam_nodes]
                d_fem = [disp[n[1], 1] * 1000 for n in beam_nodes]
                ax1.plot(x_fem, d_fem, 'o', color=colors['fem'], markersize=10,
                        markerfacecolor='white', markeredgewidth=2,
                        label=f'FEM (Level {y_level:.1f}m)')
    
    ax1.set_xlabel('Position along beam (m)')
    ax1.set_ylabel('Deflection (mm)')
    ax1.set_title('Beam Deflection')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Moment comparison
    ax2 = axes[0, 1]
    ax2.plot(x_anal, anal['moment'] / 1000, '-', color=colors['analytical'],
            linewidth=2, label='Analytical')
    ax2.fill_between(x_anal, 0, anal['moment'] / 1000, alpha=0.2, color=colors['analytical'])
    
    if 'MOMENT' in vtk_data.point_data:
        moments = vtk_data.point_data['MOMENT']
        
        for bm in beam_members:
            beam_nodes = []
            y_level = bm.start_coords[1]
            
            for i, point in enumerate(vtk_data.points):
                if abs(point[1] - y_level) < 1e-6:
                    beam_nodes.append((point[0], i))
            
            beam_nodes.sort(key=lambda x: x[0])
            
            if beam_nodes:
                x_fem = [n[0] for n in beam_nodes]
                m_fem = [-moments[n[1], 2] / 1000 for n in beam_nodes]
                ax2.plot(x_fem, m_fem, 'o', color=colors['fem'], markersize=10,
                        markerfacecolor='white', markeredgewidth=2,
                        label=f'FEM (Level {y_level:.1f}m)')
    
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Position along beam (m)')
    ax2.set_ylabel('Bending Moment (kN·m)')
    ax2.set_title('Beam Bending Moment')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Shear comparison
    ax3 = axes[1, 0]
    ax3.plot(x_anal, anal['shear'] / 1000, '-', color=colors['analytical'],
            linewidth=2, label='Analytical')
    ax3.fill_between(x_anal, 0, anal['shear'] / 1000, alpha=0.2, color=colors['analytical'])
    
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_xlabel('Position along beam (m)')
    ax3.set_ylabel('Shear Force (kN)')
    ax3.set_title('Beam Shear Force')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # 4. Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate errors
    summary_text = "ANALYTICAL SUMMARY\n"
    summary_text += "═" * 40 + "\n\n"
    summary_text += f"Beam Span: {beam_span:.2f} m\n"
    summary_text += f"UDL: {params.w/1000:.2f} kN/m\n\n"
    summary_text += "ANALYTICAL VALUES:\n"
    summary_text += f"  Max Moment:     {anal['M_max']/1000:.4f} kN·m\n"
    summary_text += f"  Max Shear:      {anal['V_max']/1000:.4f} kN\n"
    summary_text += f"  Max Deflection: {anal['delta_max']*1000:.6f} mm\n"
    summary_text += f"  Reaction:       {anal['reaction']/1000:.4f} kN\n\n"
    
    # FEM values
    if 'DISPLACEMENT' in vtk_data.point_data:
        disp = vtk_data.point_data['DISPLACEMENT']
        max_disp = np.max(np.abs(disp[:, 1])) * 1000
        
        error = abs(max_disp - anal['delta_max']*1000) / (anal['delta_max']*1000) * 100
        
        summary_text += "FEM VALUES:\n"
        summary_text += f"  Max Deflection: {max_disp:.6f} mm\n"
        summary_text += f"  Deflection Error: {error:.2f}%\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=11, fontfamily='monospace', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    
    output_file = f'{output_prefix}_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return fig


def main():
    """Main function"""
    
    # =============== CONFIGURATION ===============
    VTK_FILE = "test_files/frame_2D_test_udl_2_floor_refined.gid/vtk_output/Parts_Beam_Beams_0_1.vtk"
    OUTPUT_PREFIX = "test_files/frame_2D_test_udl_2_floor_refined.gid/plots/frame_2story"
    
    params = FrameParameters(
        E = 210e9,
        I = 5e-6,
        A = 0.00287,
        w = 10000.0,  # 10 kN/m
    )
    # =============================================
    
    print("=" * 120)
    print("MULTI-STORY FRAME ANALYSIS RESULTS PLOTTER")
    print("=" * 120)
    
    # Find VTK file
    if not os.path.exists(VTK_FILE):
        print(f"\nError: File '{VTK_FILE}' not found!")
        
        vtk_files = []
        for root, dirs, files in os.walk('.'):
            for f in files:
                if f.endswith('.vtk'):
                    vtk_files.append(os.path.join(root, f))
        
        if vtk_files:
            print(f"\nAvailable VTK files:")
            for f in vtk_files:
                print(f"  - {f}")
        return
    
    # Parse VTK
    print(f"\nReading: {VTK_FILE}")
    vtk_data = parse_vtk_file(VTK_FILE)
    
    print(f"  Nodes: {len(vtk_data.points)}")
    print(f"  Elements: {len(vtk_data.cells)}")
    print(f"  Point data: {list(vtk_data.point_data.keys())}")
    
    # Analyze geometry
    members, geometry = analyze_frame_geometry(vtk_data)
    
    # Print results
    print_frame_results(vtk_data, members, geometry, params)
    
    # Generate plots
    print("\n" + "-" * 120)
    print("Generating plots...")
    
    fig1 = plot_frame_with_loads(vtk_data, members, geometry, params, OUTPUT_PREFIX)
    fig2 = plot_frame_complete(vtk_data, members, geometry, OUTPUT_PREFIX)
    fig3 = plot_analytical_comparison(vtk_data, members, geometry, params, OUTPUT_PREFIX)
    
    plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()