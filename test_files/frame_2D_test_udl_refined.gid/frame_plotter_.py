"""
Frame Analysis Results Plotter
Reads VTK output from Kratos for frame structures
Plots deformed shape, member forces, and moments
"""

import numpy as np
import matplotlib.pyplot as plt
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
    node_i: int  # Start node index
    node_j: int  # End node index
    start_coords: np.ndarray
    end_coords: np.ndarray
    length: float
    angle: float  # Angle from horizontal (radians)
    orientation: str  # 'horizontal', 'vertical', 'inclined'


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


def analyze_frame_geometry(vtk_data: VTKData) -> List[Member]:
    """Analyze frame geometry and identify members"""
    
    members = []
    
    for idx, cell in enumerate(vtk_data.cells):
        node_i, node_j = cell[0], cell[1]
        
        start = vtk_data.points[node_i]
        end = vtk_data.points[node_j]
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        angle = np.arctan2(dy, dx)
        
        # Determine orientation
        if abs(dx) < 1e-6 and abs(dy) > 1e-6:
            orientation = 'vertical'
        elif abs(dy) < 1e-6 and abs(dx) > 1e-6:
            orientation = 'horizontal'
        else:
            orientation = 'inclined'
        
        members.append(Member(
            id=idx + 1,
            node_i=node_i,
            node_j=node_j,
            start_coords=start,
            end_coords=end,
            length=length,
            angle=angle,
            orientation=orientation
        ))
    
    return members


def print_frame_results(vtk_data: VTKData, members: List[Member]):
    """Print frame analysis results"""
    
    print("\n" + "=" * 100)
    print("FRAME ANALYSIS RESULTS")
    print("=" * 100)
    
    # Frame geometry summary
    print(f"\n{'FRAME GEOMETRY':^100}")
    print("-" * 100)
    print(f"  Total Nodes: {len(vtk_data.points)}")
    print(f"  Total Members: {len(members)}")
    
    n_vert = sum(1 for m in members if m.orientation == 'vertical')
    n_horz = sum(1 for m in members if m.orientation == 'horizontal')
    n_incl = sum(1 for m in members if m.orientation == 'inclined')
    
    print(f"  Vertical Members: {n_vert}")
    print(f"  Horizontal Members: {n_horz}")
    print(f"  Inclined Members: {n_incl}")
    
    # Member details
    print(f"\n{'MEMBER DETAILS':^100}")
    print("-" * 100)
    print(f"{'Member':<8} {'Node i':<8} {'Node j':<8} {'Length(m)':<12} {'Orientation':<15} {'Angle(°)':<12}")
    print("-" * 100)
    
    for m in members:
        angle_deg = np.degrees(m.angle)
        print(f"{m.id:<8} {m.node_i + 1:<8} {m.node_j + 1:<8} {m.length:<12.4f} {m.orientation:<15} {angle_deg:<12.2f}")
    
    print("=" * 100)
    
    # Nodal coordinates
    print(f"\n{'NODAL COORDINATES':^100}")
    print("-" * 100)
    print(f"{'Node':<8} {'X (m)':<15} {'Y (m)':<15} {'Z (m)':<15}")
    print("-" * 100)
    
    for i, point in enumerate(vtk_data.points):
        print(f"{i + 1:<8} {point[0]:<15.6f} {point[1]:<15.6f} {point[2]:<15.6f}")
    
    print("=" * 100)
    
    # Nodal displacements
    if 'DISPLACEMENT' in vtk_data.point_data:
        print(f"\n{'NODAL DISPLACEMENTS':^100}")
        print("-" * 100)
        print(f"{'Node':<8} {'Ux (mm)':<18} {'Uy (mm)':<18} {'Uz (mm)':<18}")
        print("-" * 100)
        
        disp = vtk_data.point_data['DISPLACEMENT']
        for i in range(len(vtk_data.points)):
            ux, uy, uz = disp[i] * 1000  # Convert to mm
            print(f"{i + 1:<8} {ux:<18.6f} {uy:<18.6f} {uz:<18.6f}")
        
        print("=" * 100)
    
    # Nodal rotations
    if 'ROTATION' in vtk_data.point_data:
        print(f"\n{'NODAL ROTATIONS':^100}")
        print("-" * 100)
        print(f"{'Node':<8} {'Rx (mrad)':<18} {'Ry (mrad)':<18} {'Rz (mrad)':<18}")
        print("-" * 100)
        
        rot = vtk_data.point_data['ROTATION']
        for i in range(len(vtk_data.points)):
            rx, ry, rz = rot[i] * 1000  # Convert to mrad
            print(f"{i + 1:<8} {rx:<18.6f} {ry:<18.6f} {rz:<18.6f}")
        
        print("=" * 100)
    
    # Support reactions
    if 'REACTION' in vtk_data.point_data:
        print(f"\n{'SUPPORT REACTIONS':^100}")
        print("-" * 100)
        print(f"{'Node':<8} {'Rx (kN)':<18} {'Ry (kN)':<18} {'Rz (kN)':<18}")
        print("-" * 100)
        
        reactions = vtk_data.point_data['REACTION']
        for i in range(len(vtk_data.points)):
            rx, ry, rz = reactions[i] / 1000  # Convert to kN
            if abs(rx) > 0.001 or abs(ry) > 0.001 or abs(rz) > 0.001:
                print(f"{i + 1:<8} {rx:<18.6f} {ry:<18.6f} {rz:<18.6f}")
        
        print("=" * 100)
    
    # Member end forces
    if 'FORCE' in vtk_data.point_data:
        print(f"\n{'MEMBER END FORCES (at nodes)':^100}")
        print("-" * 100)
        print(f"{'Node':<8} {'Fx (kN)':<18} {'Fy (kN)':<18} {'Fz (kN)':<18}")
        print("-" * 100)
        
        forces = vtk_data.point_data['FORCE']
        for i in range(len(vtk_data.points)):
            fx, fy, fz = forces[i] / 1000
            print(f"{i + 1:<8} {fx:<18.6f} {fy:<18.6f} {fz:<18.6f}")
        
        print("=" * 100)
    
    # Member end moments
    if 'MOMENT' in vtk_data.point_data:
        print(f"\n{'MEMBER END MOMENTS (at nodes)':^100}")
        print("-" * 100)
        print(f"{'Node':<8} {'Mx (kN·m)':<18} {'My (kN·m)':<18} {'Mz (kN·m)':<18}")
        print("-" * 100)
        
        moments = vtk_data.point_data['MOMENT']
        for i in range(len(vtk_data.points)):
            mx, my, mz = moments[i] / 1000
            print(f"{i + 1:<8} {mx:<18.6f} {my:<18.6f} {mz:<18.6f}")
        
        print("=" * 100)


def plot_frame_geometry(vtk_data: VTKData, members: List[Member], ax=None):
    """Plot frame geometry (undeformed)"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot members
    for m in members:
        x = [m.start_coords[0], m.end_coords[0]]
        y = [m.start_coords[1], m.end_coords[1]]
        
        color = '#1D3557' if m.orientation == 'vertical' else '#2E86AB'
        ax.plot(x, y, 'o-', color=color, linewidth=3, markersize=8,
               markerfacecolor='white', markeredgewidth=2)
        
        # Member label
        mid_x = (m.start_coords[0] + m.end_coords[0]) / 2
        mid_y = (m.start_coords[1] + m.end_coords[1]) / 2
        ax.text(mid_x, mid_y, f'M{m.id}', fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Node labels
    for i, point in enumerate(vtk_data.points):
        ax.annotate(f'{i + 1}', (point[0], point[1]), 
                   textcoords="offset points", xytext=(10, 10),
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.8))
    
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return ax


def plot_deformed_shape(vtk_data: VTKData, members: List[Member], 
                        scale: float = 100, ax=None, show_undeformed: bool = True):
    """Plot deformed shape of frame"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    if 'DISPLACEMENT' not in vtk_data.point_data:
        ax.text(0.5, 0.5, 'No displacement data', transform=ax.transAxes,
               ha='center', va='center', fontsize=14)
        return ax
    
    disp = vtk_data.point_data['DISPLACEMENT']
    
    # Plot undeformed shape
    if show_undeformed:
        for m in members:
            x = [m.start_coords[0], m.end_coords[0]]
            y = [m.start_coords[1], m.end_coords[1]]
            ax.plot(x, y, '--', color='gray', linewidth=1.5, alpha=0.5, label='Undeformed' if m.id == 1 else '')
    
    # Calculate deformed coordinates
    deformed_points = vtk_data.points.copy()
    deformed_points[:, 0] += disp[:, 0] * scale
    deformed_points[:, 1] += disp[:, 1] * scale
    deformed_points[:, 2] += disp[:, 2] * scale
    
    # Plot deformed shape
    for m in members:
        x = [deformed_points[m.node_i, 0], deformed_points[m.node_j, 0]]
        y = [deformed_points[m.node_i, 1], deformed_points[m.node_j, 1]]
        
        ax.plot(x, y, 'o-', color='#E63946', linewidth=2.5, markersize=8,
               markerfacecolor='white', markeredgewidth=2,
               label='Deformed' if m.id == 1 else '')
    
    # Find max displacement for annotation
    max_disp = np.max(np.abs(disp)) * 1000
    ax.text(0.02, 0.98, f'Scale: {scale}x\nMax Disp: {max_disp:.4f} mm', 
           transform=ax.transAxes, fontsize=10, va='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('Deformed Shape', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    
    return ax


def plot_axial_force_diagram(vtk_data: VTKData, members: List[Member], 
                              scale: float = 0.0001, ax=None):
    """Plot axial force diagram"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    if 'FORCE' not in vtk_data.point_data:
        ax.text(0.5, 0.5, 'No force data', transform=ax.transAxes,
               ha='center', va='center', fontsize=14)
        return ax
    
    forces = vtk_data.point_data['FORCE']
    
    # Plot frame
    for m in members:
        x = [m.start_coords[0], m.end_coords[0]]
        y = [m.start_coords[1], m.end_coords[1]]
        ax.plot(x, y, 'k-', linewidth=2)
    
    # Plot axial force diagram for each member
    for m in members:
        # Get forces at member ends
        fi = forces[m.node_i]
        fj = forces[m.node_j]
        
        # Calculate axial force (along member axis)
        cos_a = np.cos(m.angle)
        sin_a = np.sin(m.angle)
        
        # Project force onto member axis
        axial_i = fi[0] * cos_a + fi[1] * sin_a
        axial_j = fj[0] * cos_a + fj[1] * sin_a
        
        # Normal direction for plotting
        nx = -sin_a
        ny = cos_a
        
        # Create polygon for axial force diagram
        x1 = m.start_coords[0]
        y1 = m.start_coords[1]
        x2 = m.end_coords[0]
        y2 = m.end_coords[1]
        
        x1_off = x1 + axial_i * scale * nx
        y1_off = y1 + axial_i * scale * ny
        x2_off = x2 + axial_j * scale * nx
        y2_off = y2 + axial_j * scale * ny
        
        # Determine color based on tension/compression
        avg_axial = (axial_i + axial_j) / 2
        color = '#E63946' if avg_axial < 0 else '#2E86AB'
        
        polygon_x = [x1, x1_off, x2_off, x2, x1]
        polygon_y = [y1, y1_off, y2_off, y2, y1]
        ax.fill(polygon_x, polygon_y, color=color, alpha=0.3, edgecolor=color, linewidth=1)
        
        # Add value label
        mid_x = (x1_off + x2_off) / 2
        mid_y = (y1_off + y2_off) / 2
        ax.text(mid_x, mid_y, f'{avg_axial/1000:.2f}', fontsize=9, ha='center', va='center')
    
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('Axial Force Diagram (kN)\nRed: Compression, Blue: Tension', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return ax


def plot_bending_moment_diagram(vtk_data: VTKData, members: List[Member], 
                                 scale: float = 0.0002, ax=None):
    """Plot bending moment diagram"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    if 'MOMENT' not in vtk_data.point_data:
        ax.text(0.5, 0.5, 'No moment data', transform=ax.transAxes,
               ha='center', va='center', fontsize=14)
        return ax
    
    moments = vtk_data.point_data['MOMENT']
    
    # Plot frame
    for m in members:
        x = [m.start_coords[0], m.end_coords[0]]
        y = [m.start_coords[1], m.end_coords[1]]
        ax.plot(x, y, 'k-', linewidth=2)
    
    # Plot bending moment diagram for each member
    for m in members:
        # Get moments at member ends (Mz for 2D frame)
        mz_i = moments[m.node_i, 2]
        mz_j = moments[m.node_j, 2]
        
        # Normal direction for plotting (perpendicular to member)
        cos_a = np.cos(m.angle)
        sin_a = np.sin(m.angle)
        nx = -sin_a
        ny = cos_a
        
        # Create polygon for moment diagram
        x1 = m.start_coords[0]
        y1 = m.start_coords[1]
        x2 = m.end_coords[0]
        y2 = m.end_coords[1]
        
        x1_off = x1 + mz_i * scale * nx
        y1_off = y1 + mz_i * scale * ny
        x2_off = x2 + mz_j * scale * nx
        y2_off = y2 + mz_j * scale * ny
        
        # Determine color
        color = '#F18F01'
        
        polygon_x = [x1, x1_off, x2_off, x2, x1]
        polygon_y = [y1, y1_off, y2_off, y2, y1]
        ax.fill(polygon_x, polygon_y, color=color, alpha=0.4, edgecolor=color, linewidth=1)
        
        # Add value labels
        if abs(mz_i) > 0.1:
            ax.text(x1_off, y1_off, f'{mz_i/1000:.3f}', fontsize=9, ha='center', va='center')
        if abs(mz_j) > 0.1:
            ax.text(x2_off, y2_off, f'{mz_j/1000:.3f}', fontsize=9, ha='center', va='center')
    
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('Bending Moment Diagram (kN·m)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return ax


def plot_shear_force_diagram(vtk_data: VTKData, members: List[Member], 
                              scale: float = 0.0001, ax=None):
    """Plot shear force diagram"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    if 'FORCE' not in vtk_data.point_data:
        ax.text(0.5, 0.5, 'No force data', transform=ax.transAxes,
               ha='center', va='center', fontsize=14)
        return ax
    
    forces = vtk_data.point_data['FORCE']
    
    # Plot frame
    for m in members:
        x = [m.start_coords[0], m.end_coords[0]]
        y = [m.start_coords[1], m.end_coords[1]]
        ax.plot(x, y, 'k-', linewidth=2)
    
    # Plot shear force diagram for each member
    for m in members:
        fi = forces[m.node_i]
        fj = forces[m.node_j]
        
        # Calculate shear force (perpendicular to member axis)
        cos_a = np.cos(m.angle)
        sin_a = np.sin(m.angle)
        
        # Shear = force perpendicular to member
        shear_i = -fi[0] * sin_a + fi[1] * cos_a
        shear_j = -fj[0] * sin_a + fj[1] * cos_a
        
        # Normal direction for plotting
        nx = -sin_a
        ny = cos_a
        
        x1 = m.start_coords[0]
        y1 = m.start_coords[1]
        x2 = m.end_coords[0]
        y2 = m.end_coords[1]
        
        x1_off = x1 + shear_i * scale * nx
        y1_off = y1 + shear_i * scale * ny
        x2_off = x2 + shear_j * scale * nx
        y2_off = y2 + shear_j * scale * ny
        
        color = '#A23B72'
        
        polygon_x = [x1, x1_off, x2_off, x2, x1]
        polygon_y = [y1, y1_off, y2_off, y2, y1]
        ax.fill(polygon_x, polygon_y, color=color, alpha=0.4, edgecolor=color, linewidth=1)
        
        # Add value labels
        mid_x = (x1_off + x2_off) / 2
        mid_y = (y1_off + y2_off) / 2
        avg_shear = (shear_i + shear_j) / 2
        if abs(avg_shear) > 0.1:
            ax.text(mid_x, mid_y, f'{avg_shear/1000:.2f}', fontsize=9, ha='center', va='center')
    
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('Shear Force Diagram (kN)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return ax


def plot_frame_complete(vtk_data: VTKData, members: List[Member], 
                         output_prefix: str = "frame"):
    """Create complete frame analysis plot"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Frame Analysis Results', fontsize=14, fontweight='bold')
    
    # 1. Deformed Shape
    plot_deformed_shape(vtk_data, members, scale=1000, ax=axes[0, 0])
    
    # 2. Axial Force Diagram
    plot_axial_force_diagram(vtk_data, members, scale=0.00005, ax=axes[0, 1])
    
    # 3. Bending Moment Diagram  
    plot_bending_moment_diagram(vtk_data, members, scale=0.0005, ax=axes[1, 0])
    
    # 4. Shear Force Diagram
    plot_shear_force_diagram(vtk_data, members, scale=0.00005, ax=axes[1, 1])
    
    plt.tight_layout()
    
    output_file = f'{output_prefix}_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return fig


def plot_frame_with_loads(vtk_data: VTKData, members: List[Member], 
                          output_prefix: str = "frame"):
    """Plot frame with support symbols and reactions"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot members
    for m in members:
        x = [m.start_coords[0], m.end_coords[0]]
        y = [m.start_coords[1], m.end_coords[1]]
        
        ax.plot(x, y, 'o-', color='#2E86AB', linewidth=4, markersize=10,
               markerfacecolor='white', markeredgewidth=2)
        
        # Member label
        mid_x = (m.start_coords[0] + m.end_coords[0]) / 2
        mid_y = (m.start_coords[1] + m.end_coords[1]) / 2
        ax.text(mid_x + 0.1, mid_y, f'M{m.id}', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Node labels
    for i, point in enumerate(vtk_data.points):
        ax.annotate(f'{i + 1}', (point[0], point[1]), 
                   textcoords="offset points", xytext=(15, 15),
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.9))
    
    # Plot supports (detect from reactions)
    if 'REACTION' in vtk_data.point_data:
        reactions = vtk_data.point_data['REACTION']
        
        for i, point in enumerate(vtk_data.points):
            rx, ry, rz = reactions[i]
            
            # If node has reactions, draw support
            if abs(ry) > 1:
                # Fixed support triangle
                support_size = 0.15
                triangle = plt.Polygon([
                    (point[0], point[1]),
                    (point[0] - support_size, point[1] - support_size * 1.5),
                    (point[0] + support_size, point[1] - support_size * 1.5)
                ], facecolor='#1D3557', edgecolor='black', linewidth=2)
                ax.add_patch(triangle)
                
                # Ground hatching
                ax.plot([point[0] - 0.2, point[0] + 0.2], 
                       [point[1] - support_size * 1.5 - 0.02] * 2,
                       'k-', linewidth=2)
                
                # Reaction arrows
                arrow_scale = 0.00003
                if abs(ry) > 1:
                    ax.annotate('', xy=(point[0], point[1] - 0.1),
                               xytext=(point[0], point[1] - 0.5),
                               arrowprops=dict(arrowstyle='->', color='green', lw=2))
                    ax.text(point[0] + 0.15, point[1] - 0.4, f'Ry={ry/1000:.2f}kN', 
                           fontsize=10, color='green')
    
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('Frame Geometry with Supports and Reactions', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits with padding
    x_min = min(p[0] for p in vtk_data.points) - 0.5
    x_max = max(p[0] for p in vtk_data.points) + 0.5
    y_min = min(p[1] for p in vtk_data.points) - 0.8
    y_max = max(p[1] for p in vtk_data.points) + 0.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    output_file = f'{output_prefix}_geometry.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return fig


def main():
    """Main function"""
    
    # =============== CONFIGURATION ===============
    VTK_FILE = "test_files/frame_2D_test_udl.gid/vtk_output/Parts_Beam_Beams_0_1.vtk"  # Your VTK file
    OUTPUT_PREFIX = "frame_results"
    # =============================================
    
    print("=" * 100)
    print("FRAME ANALYSIS RESULTS PLOTTER")
    print("=" * 100)
    
    # Check if file exists
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
    
    # Parse VTK file
    print(f"\nReading: {VTK_FILE}")
    vtk_data = parse_vtk_file(VTK_FILE)
    
    print(f"  Nodes: {len(vtk_data.points)}")
    print(f"  Elements/Members: {len(vtk_data.cells)}")
    print(f"  Point data: {list(vtk_data.point_data.keys())}")
    
    # Analyze frame geometry
    members = analyze_frame_geometry(vtk_data)
    
    # Print results
    print_frame_results(vtk_data, members)
    
    # Generate plots
    print("\n" + "-" * 100)
    print("Generating plots...")
    
    fig1 = plot_frame_with_loads(vtk_data, members, OUTPUT_PREFIX)
    fig2 = plot_frame_complete(vtk_data, members, OUTPUT_PREFIX)
    
    plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()