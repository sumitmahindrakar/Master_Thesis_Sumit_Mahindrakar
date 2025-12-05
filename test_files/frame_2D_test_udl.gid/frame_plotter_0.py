"""
Frame Analysis Results Plotter with Analytical Verification
Reads VTK output from Kratos for frame structures
Compares with analytical solutions and calculates errors
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, Circle
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
    angle: float
    orientation: str
    member_type: str = ""  # 'column_left', 'column_right', 'beam'


@dataclass 
class FrameParameters:
    """Frame properties for analytical solution"""
    L: float          # Beam span [m]
    A: float          #crossArea m^2
    H: float          # Column height [m]
    w: float          # UDL on beam [N/m]
    P: float          # Point load [N]
    E: float          # Young's modulus [Pa]
    I_beam: float     # Beam moment of inertia [m^4]
    I_col: float      # Column moment of inertia [m^4]
    A_beam: float     # Beam cross-sectional area [m^2]
    A_col: float      # Column cross-sectional area [m^2]
    frame_type: str   # 'portal_pinned', 'portal_fixed'
    load_type: str    # 'udl_beam', 'point_beam_center'


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


def analyze_frame_geometry(vtk_data: VTKData) -> Tuple[List[Member], Dict]:
    """Analyze frame geometry and identify members"""
    
    members = []
    
    # Find frame bounds
    x_coords = vtk_data.points[:, 0]
    y_coords = vtk_data.points[:, 1]
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    frame_info = {
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'span': x_max - x_min,
        'height': y_max - y_min,
        'n_columns': 0,
        'n_beams': 0
    }
    
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
            frame_info['n_columns'] += 1
            
            # Determine if left or right column
            if abs(start[0] - x_min) < 1e-6 or abs(end[0] - x_min) < 1e-6:
                member_type = 'column_left'
            else:
                member_type = 'column_right'
                
        elif abs(dy) < 1e-6 and abs(dx) > 1e-6:
            orientation = 'horizontal'
            member_type = 'beam'
            frame_info['n_beams'] += 1
        else:
            orientation = 'inclined'
            member_type = 'inclined'
        
        members.append(Member(
            id=idx + 1,
            node_i=node_i,
            node_j=node_j,
            start_coords=start.copy(),
            end_coords=end.copy(),
            length=length,
            angle=angle,
            orientation=orientation,
            member_type=member_type
        ))
    
    return members, frame_info


class PortalFrameAnalytical:
    """Analytical solutions for portal frames"""
    
    @staticmethod
    def pinned_base_udl(params: FrameParameters) -> Dict:
        """
        Portal frame with pinned bases and UDL on beam
        
        Structure:
            2 ----w---- 3
            |          |
            |          |
            1          4
            △          △
        
        For symmetric portal frame with pinned bases:
        - Vertical reactions: R1y = R4y = wL/2
        - Horizontal reactions: R1x = R4x = 0 (for vertical load only)
        - Beam moment at center: M_max = wL²/8 (simply supported)
        - Column moments: 0 (pinned base, no horizontal load)
        """
        L = params.L
        H = params.H
        w = params.w
        E = params.E
        I = params.I_beam
        
        # Reactions
        Ry = w * L / 2
        Rx = 0
        
        # Beam analysis (simply supported)
        # Moment: M(x) = Ry*x - w*x²/2 = wL*x/2 - w*x²/2
        # Max moment at center: M_max = wL²/8
        M_max_beam = w * L**2 / 8
        
        # Shear: V(x) = Ry - w*x = wL/2 - w*x
        V_max_beam = w * L / 2
        
        # Deflection at center: δ = 5wL⁴/(384EI)
        delta_max_beam = 5 * w * L**4 / (384 * E * I)
        
        # End rotation of beam: θ = wL³/(24EI)
        theta_beam_end = w * L**3 / (24 * E * I)
        
        # Column analysis (pinned base, no moment transfer)
        # Axial force = Ry = wL/2
        N_column = Ry
        
        # Column shortening: δ_col = N*H/(EA)
        # delta_column = N_column * H / (E * params.A_col)
        
        return {
            'reactions': {
                'R1x': 0, 'R1y': Ry,
                'R4x': 0, 'R4y': Ry
            },
            'beam': {
                'M_max': M_max_beam,
                'M_left': 0,
                'M_right': 0,
                'V_left': V_max_beam,
                'V_right': -V_max_beam,
                'V_max': V_max_beam,
                'delta_max': delta_max_beam,
                'theta_end': theta_beam_end
            },
            'columns': {
                'M_top': 0,
                'M_bottom': 0,
                'N': N_column,
                'V': 0
            },
            'description': 'Portal frame with pinned bases, UDL on beam'
        }
    
    @staticmethod
    def fixed_base_udl(params: FrameParameters) -> Dict:
        """
        Portal frame with fixed bases and UDL on beam
        
        For symmetric loading on symmetric frame:
        - No sway occurs
        - Can analyze as frame with rotational springs
        
        Using stiffness method / moment distribution:
        For equal I in beam and columns:
        """
        L = params.L
        H = params.H
        w = params.w
        E = params.E
        I_beam = params.I_beam
        I_col = params.I_col
        
        # Stiffness factors
        k_beam = I_beam / L
        k_col = I_col / H
        
        # Distribution factors at joints 2 and 3
        DF_beam = k_beam / (k_beam + k_col)
        DF_col = k_col / (k_beam + k_col)
        
        # Fixed end moments for UDL on beam
        FEM_beam = w * L**2 / 12
        
        # For symmetric frame with symmetric loading
        # Joint moments after distribution
        # M_joint ≈ FEM * DF_col (simplified)
        
        # More accurate: use moment distribution or stiffness method
        # For equal stiffness: M_joint = wL²/12 * k_col/(k_beam + k_col)
        
        # Simplified solution for equal I:
        alpha = k_col / (k_beam + k_col)
        
        M_joint = FEM_beam * (1 - alpha)  # Moment at beam ends
        M_col_top = FEM_beam * alpha  # Moment at column tops
        M_col_bottom = M_col_top / 2  # Carry-over to fixed base
        
        # Beam center moment
        M_center = w * L**2 / 8 - M_joint
        
        # Reactions
        Ry = w * L / 2
        
        # Column shear from moments
        V_col = (M_col_top + M_col_bottom) / H
        
        # Beam deflection (approximate, reduced due to end fixity)
        delta_max_beam = w * L**4 / (384 * E * I_beam)  # Fixed-fixed formula
        
        return {
            'reactions': {
                'R1x': V_col, 'R1y': Ry, 'M1': M_col_bottom,
                'R4x': -V_col, 'R4y': Ry, 'M4': M_col_bottom
            },
            'beam': {
                'M_max': M_center,
                'M_left': -M_joint,
                'M_right': -M_joint,
                'V_left': w * L / 2,
                'V_right': -w * L / 2,
                'V_max': w * L / 2,
                'delta_max': delta_max_beam
            },
            'columns': {
                'M_top': M_col_top,
                'M_bottom': M_col_bottom,
                'N': Ry,
                'V': V_col
            },
            'description': 'Portal frame with fixed bases, UDL on beam'
        }
    
    @staticmethod
    def get_beam_distributions(x: np.ndarray, params: FrameParameters, 
                                frame_type: str = 'pinned') -> Dict:
        """Get moment, shear, deflection distributions along beam"""
        
        L = params.L
        w = params.w
        E = params.E
        I = params.I_beam
        
        if frame_type == 'pinned':
            # Simply supported beam formulas
            moment = w * L * x / 2 - w * x**2 / 2
            shear = w * L / 2 - w * x
            deflection = w * x * (L**3 - 2*L*x**2 + x**3) / (24 * E * I)
            rotation = w * (L**3 - 6*L*x**2 + 4*x**3) / (24 * E * I)
            
        else:  # fixed
            # Fixed-end beam formulas (approximate for frame)
            M_end = w * L**2 / 12
            moment = M_end - w * L * x / 2 + w * x**2 / 2 + M_end
            # More accurate: M(x) = -wL²/12 + wLx/2 - wx²/2
            moment = -w * L**2 / 12 + w * L * x / 2 - w * x**2 / 2
            shear = w * L / 2 - w * x
            deflection = w * x**2 * (L - x)**2 / (24 * E * I)
            rotation = w * x * (L - x) * (L - 2*x) / (12 * E * I)
        
        return {
            'x': x,
            'moment': moment,
            'shear': shear,
            'deflection': deflection,
            'rotation': rotation
        }
    
    @staticmethod
    def get_column_distributions(y: np.ndarray, params: FrameParameters,
                                  analytical: Dict, column: str = 'left') -> Dict:
        """Get moment, shear, axial distributions along column"""
        
        H = params.H
        
        M_top = analytical['columns']['M_top']
        M_bottom = analytical['columns']['M_bottom']
        N = analytical['columns']['N']
        V = analytical['columns']['V']
        
        # Linear moment distribution for columns
        moment = M_bottom + (M_top - M_bottom) * y / H
        
        # Constant shear and axial
        shear = np.full_like(y, V)
        axial = np.full_like(y, -N)  # Compression is negative
        
        return {
            'y': y,
            'moment': moment,
            'shear': shear,
            'axial': axial
        }


def calculate_frame_errors(vtk_data: VTKData, members: List[Member], 
                           params: FrameParameters, frame_info: Dict) -> Dict:
    """Calculate errors between FEM and analytical solutions"""
    
    # Get analytical solution
    if params.frame_type == 'portal_pinned':
        analytical = PortalFrameAnalytical.pinned_base_udl(params)
    else:
        analytical = PortalFrameAnalytical.fixed_base_udl(params)
    
    errors = {
        'analytical': analytical,
        'frame_info': frame_info,
        'params': params
    }
    
    # Extract FEM results at key points
    if 'REACTION' in vtk_data.point_data:
        reactions = vtk_data.point_data['REACTION']
        
        # Find support nodes (nodes at y=0)
        support_nodes = []
        for i, point in enumerate(vtk_data.points):
            if abs(point[1] - frame_info['y_min']) < 1e-6:
                support_nodes.append(i)
        
        if len(support_nodes) >= 2:
            # Sort by x-coordinate
            support_nodes.sort(key=lambda n: vtk_data.points[n, 0])
            
            node_left = support_nodes[0]
            node_right = support_nodes[-1]
            
            fem_R1y = reactions[node_left, 1]
            fem_R4y = reactions[node_right, 1]
            
            anal_R1y = analytical['reactions']['R1y']
            anal_R4y = analytical['reactions']['R4y']
            
            errors['reactions'] = {
                'R1y': {
                    'fem': fem_R1y / 1000,
                    'analytical': anal_R1y / 1000,
                    'error_percent': abs(fem_R1y - anal_R1y) / anal_R1y * 100 if anal_R1y != 0 else 0
                },
                'R4y': {
                    'fem': fem_R4y / 1000,
                    'analytical': anal_R4y / 1000,
                    'error_percent': abs(fem_R4y - anal_R4y) / anal_R4y * 100 if anal_R4y != 0 else 0
                }
            }
    
    # Displacement errors
    if 'DISPLACEMENT' in vtk_data.point_data:
        disp = vtk_data.point_data['DISPLACEMENT']
        
        # Find beam nodes (nodes at y = y_max)
        beam_nodes = []
        for i, point in enumerate(vtk_data.points):
            if abs(point[1] - frame_info['y_max']) < 1e-6:
                beam_nodes.append(i)
        
        if beam_nodes:
            # Find max vertical displacement on beam
            max_disp_idx = max(beam_nodes, key=lambda n: abs(disp[n, 1]))
            fem_delta_max = abs(disp[max_disp_idx, 1])
            
            anal_delta_max = analytical['beam']['delta_max']
            
            errors['deflection'] = {
                'beam_max': {
                    'fem': fem_delta_max * 1000,
                    'analytical': anal_delta_max * 1000,
                    'error_percent': abs(fem_delta_max - anal_delta_max) / anal_delta_max * 100 if anal_delta_max != 0 else 0
                }
            }
    
    # Moment errors (if available)
    if 'MOMENT' in vtk_data.point_data:
        moments = vtk_data.point_data['MOMENT']
        
        # Find beam center node if exists
        beam_center_nodes = []
        for i, point in enumerate(vtk_data.points):
            if abs(point[1] - frame_info['y_max']) < 1e-6:
                if abs(point[0] - (frame_info['x_min'] + frame_info['x_max'])/2) < 0.1:
                    beam_center_nodes.append(i)
        
        if beam_center_nodes:
            center_node = beam_center_nodes[0]
            fem_M_center = abs(moments[center_node, 2])
            anal_M_max = analytical['beam']['M_max']
            
            errors['moment'] = {
                'beam_center': {
                    'fem': fem_M_center / 1000,
                    'analytical': anal_M_max / 1000,
                    'error_percent': abs(fem_M_center - anal_M_max) / anal_M_max * 100 if anal_M_max != 0 else 0
                }
            }
    
    # Member-wise results
    errors['members'] = {}
    for m in members:
        errors['members'][m.id] = {
            'type': m.member_type,
            'orientation': m.orientation,
            'length': m.length
        }
    
    return errors


def print_frame_results(vtk_data: VTKData, members: List[Member], 
                        errors: Dict, params: FrameParameters):
    """Print frame analysis results with analytical comparison"""
    
    analytical = errors['analytical']
    frame_info = errors['frame_info']
    
    print("\n" + "=" * 110)
    print("FRAME ANALYSIS RESULTS WITH ANALYTICAL VERIFICATION")
    print("=" * 110)
    
    # Frame properties
    print(f"\n{'FRAME PROPERTIES':^110}")
    print("-" * 110)
    print(f"  Span (L):               {params.L:.4f} m")
    print(f"  Cross-Area (A):         {params.A:.4f} m")
    print(f"  Height (H):             {params.H:.4f} m")
    print(f"  UDL on beam (w):        {params.w/1000:.4f} kN/m")
    print(f"  Young's Modulus (E):    {params.E/1e9:.2f} GPa")
    print(f"  Beam I:                 {params.I_beam*1e6:.4f} × 10⁻⁶ m⁴")
    print(f"  Column I:               {params.I_col*1e6:.4f} × 10⁻⁶ m⁴")
    print(f"  Frame Type:             {params.frame_type}")
    print(f"  Number of Members:      {len(members)}")
    print(f"  Number of Nodes:        {len(vtk_data.points)}")
    
    # Member summary
    print(f"\n{'MEMBER SUMMARY':^110}")
    print("-" * 110)
    print(f"{'Member':<10} {'Type':<20} {'Length (m)':<15} {'Node i':<10} {'Node j':<10}")
    print("-" * 110)
    for m in members:
        print(f"{m.id:<10} {m.member_type:<20} {m.length:<15.4f} {m.node_i + 1:<10} {m.node_j + 1:<10}")
    
    # Nodal displacements
    if 'DISPLACEMENT' in vtk_data.point_data:
        print(f"\n{'NODAL DISPLACEMENTS (FEM)':^110}")
        print("-" * 110)
        print(f"{'Node':<8} {'X (m)':<12} {'Y (m)':<12} {'Ux (mm)':<15} {'Uy (mm)':<15} {'Uz (mm)':<15}")
        print("-" * 110)
        
        disp = vtk_data.point_data['DISPLACEMENT']
        for i, point in enumerate(vtk_data.points):
            ux, uy, uz = disp[i] * 1000
            print(f"{i + 1:<8} {point[0]:<12.4f} {point[1]:<12.4f} {ux:<15.6f} {uy:<15.6f} {uz:<15.6f}")
    
    print("=" * 110)
    
    # Analytical vs FEM comparison
    print(f"\n{'ANALYTICAL vs FEM COMPARISON':^110}")
    print("=" * 110)
    print(f"\n{analytical['description']}")
    print("-" * 110)
    
    print(f"\n{'Parameter':<35} {'Analytical':<25} {'FEM':<25} {'Error (%)':<20}")
    print("-" * 110)
    
    # Reactions
    if 'reactions' in errors:
        r = errors['reactions']
        print(f"{'Left Support Ry (kN)':<35} {r['R1y']['analytical']:<25.4f} {r['R1y']['fem']:<25.4f} {r['R1y']['error_percent']:<20.4f}")
        print(f"{'Right Support Ry (kN)':<35} {r['R4y']['analytical']:<25.4f} {r['R4y']['fem']:<25.4f} {r['R4y']['error_percent']:<20.4f}")
    
    # Deflection
    if 'deflection' in errors:
        d = errors['deflection']['beam_max']
        print(f"{'Max Beam Deflection (mm)':<35} {d['analytical']:<25.6f} {d['fem']:<25.6f} {d['error_percent']:<20.2f}")
    
    # Moments
    if 'moment' in errors:
        m = errors['moment']['beam_center']
        print(f"{'Beam Center Moment (kN·m)':<35} {m['analytical']:<25.4f} {m['fem']:<25.4f} {m['error_percent']:<20.2f}")
    
    # Analytical key values
    print(f"\n{'ANALYTICAL KEY VALUES':^110}")
    print("-" * 110)
    print(f"  Beam Max Moment:        {analytical['beam']['M_max']/1000:.4f} kN·m")
    print(f"  Beam Max Shear:         {analytical['beam']['V_max']/1000:.4f} kN")
    print(f"  Beam Max Deflection:    {analytical['beam']['delta_max']*1000:.6f} mm")
    print(f"  Column Axial Force:     {analytical['columns']['N']/1000:.4f} kN")
    
    print("=" * 110)
    
    # Support reactions from VTK
    if 'REACTION' in vtk_data.point_data:
        print(f"\n{'SUPPORT REACTIONS (FEM)':^110}")
        print("-" * 110)
        reactions = vtk_data.point_data['REACTION']
        for i, point in enumerate(vtk_data.points):
            rx, ry, rz = reactions[i]
            if abs(rx) > 0.1 or abs(ry) > 0.1 or abs(rz) > 0.1:
                print(f"  Node {i + 1} (x={point[0]:.2f}, y={point[1]:.2f}): "
                      f"Rx = {rx/1000:>10.4f} kN, Ry = {ry/1000:>10.4f} kN")
        print("=" * 110)


def plot_frame_system_diagram(params: FrameParameters, output_prefix: str = "frame"):
    """Plot frame system diagram with loads and dimensions"""
    
    fig, ax = plt.subplots(figsize=(13, 7))
    
    L = params.L
    A = params.A
    H = params.H
    w = params.w
    
    # Draw frame
    frame_color = '#2E86AB'
    lw = 4
    
    # Left column
    ax.plot([0, 0], [0, H], color=frame_color, linewidth=lw)
    # Right column
    ax.plot([L, L], [0, H], color=frame_color, linewidth=lw)
    # Beam
    ax.plot([0, L], [H, H], color=frame_color, linewidth=lw)
    
    # Nodes
    nodes = [(0, 0), (0, H), (L, H), (L, 0)]
    node_labels = ['1', '2', '3', '4']
    
    for (x, y), label in zip(nodes, node_labels):
        circle = Circle((x, y), 0.02, facecolor='black', edgecolor='black', linewidth=2, zorder=5)
        ax.add_patch(circle)
        # ax.text(x - 0.25, y + 0.15, label, fontsize=12, fontweight='bold',
        #        bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.9))
    
    # UDL on beam
    n_arrows = 12
    arrow_y_start = H + 0.3
    for i in range(n_arrows + 1):
        x = i * L / n_arrows
        ax.annotate('', xy=(x, H + 0.05), xytext=(x, arrow_y_start),
                   arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.5))
    
    # UDL line
    ax.plot([0, L], [arrow_y_start, arrow_y_start], color='#E63946', linewidth=2)
    ax.text(L/2, arrow_y_start + 0.15, f'w = {w/1000:.1f} kN/m', fontsize=12, 
           ha='center', fontweight='bold', color='#E63946')
    
    # Supports
    support_size = 0.12
    
    # Left support (pinned)
    triangle_left = plt.Polygon([
        (0, 0), (-support_size, -support_size*1.5), (support_size, -support_size*1.5)
    ], facecolor='#1D3557', edgecolor='black', linewidth=2)
    ax.add_patch(triangle_left)
    ax.plot([-0.2, 0.2], [-support_size*1.5 - 0.02]*2, 'k-', linewidth=2)
    
    # Right support (pinned)
    triangle_right = plt.Polygon([
        (L, 0), (L-support_size, -support_size*1.5), (L+support_size, -support_size*1.5)
    ], facecolor='#1D3557', edgecolor='black', linewidth=2)
    ax.add_patch(triangle_right)
    ax.plot([L-0.2, L+0.2], [-support_size*1.5 - 0.02]*2, 'k-', linewidth=2)
    
    # Dimensions
    dim_offset = -0.6
    
    # Span dimension
    ax.annotate('', xy=(0, dim_offset+0.2), xytext=(L, dim_offset+0.2),
               arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(L/2, dim_offset + 0.05, f'L = {L:.1f} m', fontsize=11, ha='center', color='gray')

    ax.annotate('', xy=(0, -0.25), xytext=(0, -0.6),
               arrowprops=dict(arrowstyle='-', color='gray', lw=1.5))
    # ax.text(0.15, -0.4, f'Ry = {w*L/2/1000:.1f} kN', fontsize=10, color='green')
    
    ax.annotate('', xy=(L, -0.25), xytext=(L, -0.6),
               arrowprops=dict(arrowstyle='-', color='gray', lw=1.5))
    
    # Height dimension
    ax.annotate('', xy=(-0.4, 0), xytext=(-0.4, H),
               arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(-0.5, H/2, f'H = {H:.1f} m', fontsize=11, ha='center', va='center', 
           rotation=90, color='gray')
    
    # line
    ax.annotate('', xy=(-0.1, 0), xytext=(-0.5, 0),
               arrowprops=dict(arrowstyle='-', color='gray', lw=1.5))
    # ax.text(0.15, -0.4, f'Ry = {w*L/2/1000:.1f} kN', fontsize=10, color='green')
    
    ax.annotate('', xy=(-0.1, H), xytext=(-0.5, H),
               arrowprops=dict(arrowstyle='-', color='gray', lw=1.5))
    # ax.text(L-0.4, -0.4, f'Ry = {w*L/2/1000:.1f} kN', fontsize=10, color='green')
    
    # Member labels
    ax.text(0.25, H/2, 'M1\n(Column)', fontsize=10, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.text(L/2, H - 0.15, 'M2 (Beam)', fontsize=10, ha='center', va='bottom',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.text(L - 0.25, H/2, 'M3\n(Column)', fontsize=10, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Properties box
    props_text = (f"Frame Properties:\n"
                  f"─────────────────\n"
                  f"E = {params.E/1e9:.0f} GPa\n"
                  f"I = {params.I_beam*1e6:.2f}×10⁻⁶ m⁴\n"
                  f"L = {L:.1f} m\n"
                  f"A = {A:.1f} m\n"
                  f"H = {H:.1f} m\n"
                  f"w = {w/1000:.1f} kN/m")
    ax.text(L + 0.6, H, props_text, fontsize=10, va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax.set_xlim(-1, L + 1.5)
    ax.set_ylim(-1, H + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Portal Frame with UDL - System Diagram', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_file = f'{output_prefix}_system.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    
    return fig


def plot_analytical_diagrams(params: FrameParameters, errors: Dict, 
                              output_prefix: str = "frame"):
    """Plot analytical BMD, SFD, and deflection diagrams"""
    
    L = params.L
    H = params.H
    analytical = errors['analytical']
    
    # Create fine grids for plotting
    x_beam = np.linspace(0, L, 100)
    y_col = np.linspace(0, H, 50)
    
    beam_dist = PortalFrameAnalytical.get_beam_distributions(x_beam, params, 
                                                             params.frame_type.split('_')[1])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analytical Solution - Internal Force Diagrams', fontsize=14, fontweight='bold')
    
    # 1. Bending Moment Diagram
    ax1 = axes[0, 0]
    
    # Draw frame outline
    ax1.plot([0, 0], [0, H], 'k-', linewidth=2)
    ax1.plot([L, L], [0, H], 'k-', linewidth=2)
    ax1.plot([0, L], [H, H], 'k-', linewidth=2)
    
    # BMD for beam (plot perpendicular to member)
    moment_scale = 0.3 / (analytical['beam']['M_max'] / 1000) if analytical['beam']['M_max'] != 0 else 1
    
    bmd_x = x_beam
    bmd_y = H - beam_dist['moment'] / 1000 * moment_scale
    
    ax1.fill_between(bmd_x, H, bmd_y, alpha=0.4, color='#F18F01')
    ax1.plot(bmd_x, bmd_y, color='#F18F01', linewidth=2)
    
    # Max moment label
    max_idx = np.argmax(beam_dist['moment'])
    ax1.annotate(f'M_max = {beam_dist["moment"][max_idx]/1000:.2f} kN·m',
                xy=(x_beam[max_idx], bmd_y[max_idx]),
                xytext=(x_beam[max_idx], bmd_y[max_idx] - 0.3),
                fontsize=10, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#F18F01'))
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Bending Moment Diagram (BMD)', fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Shear Force Diagram
    ax2 = axes[0, 1]
    
    ax2.plot([0, 0], [0, H], 'k-', linewidth=2)
    ax2.plot([L, L], [0, H], 'k-', linewidth=2)
    ax2.plot([0, L], [H, H], 'k-', linewidth=2)
    
    shear_scale = 0.3 / (analytical['beam']['V_max'] / 1000) if analytical['beam']['V_max'] != 0 else 1
    
    sfd_x = x_beam
    sfd_y = H - beam_dist['shear'] / 1000 * shear_scale
    
    ax2.fill_between(sfd_x, H, sfd_y, alpha=0.4, color='#A23B72')
    ax2.plot(sfd_x, sfd_y, color='#A23B72', linewidth=2)
    
    # Shear values at ends
    ax2.text(0.1, H - 0.4, f'+{analytical["beam"]["V_left"]/1000:.1f} kN', 
            fontsize=10, color='#A23B72', fontweight='bold')
    ax2.text(L - 0.3, H + 0.3, f'{analytical["beam"]["V_right"]/1000:.1f} kN', 
            fontsize=10, color='#A23B72', fontweight='bold')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Shear Force Diagram (SFD)', fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Deflection Diagram
    ax3 = axes[1, 0]
    
    ax3.plot([0, 0], [0, H], 'k--', linewidth=1, alpha=0.5)
    ax3.plot([L, L], [0, H], 'k--', linewidth=1, alpha=0.5)
    ax3.plot([0, L], [H, H], 'k--', linewidth=1, alpha=0.5)
    
    defl_scale = 500  # Amplification factor
    
    defl_x = x_beam
    defl_y = H - beam_dist['deflection'] * defl_scale
    
    ax3.plot(defl_x, defl_y, color='#2E86AB', linewidth=2.5)
    ax3.plot([0, 0], [0, H], color='#2E86AB', linewidth=2.5)
    ax3.plot([L, L], [0, H], color='#2E86AB', linewidth=2.5)
    
    # Max deflection
    max_defl_idx = np.argmax(beam_dist['deflection'])
    max_defl = beam_dist['deflection'][max_defl_idx] * 1000
    ax3.annotate(f'δ_max = {max_defl:.4f} mm',
                xy=(x_beam[max_defl_idx], defl_y[max_defl_idx]),
                xytext=(x_beam[max_defl_idx] + 0.3, defl_y[max_defl_idx] - 0.3),
                fontsize=10, ha='left', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2E86AB'))
    
    ax3.text(0.02, 0.02, f'Scale: {defl_scale}x', transform=ax3.transAxes,
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Deflected Shape', fontweight='bold')
    ax3.set_aspect('equal')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # 4. Summary of Key Values
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = (
        "ANALYTICAL SOLUTION SUMMARY\n"
        "═══════════════════════════════════════\n\n"
        f"Frame Type: {params.frame_type.replace('_', ' ').title()}\n"
        f"Load Type:  UDL on Beam\n\n"
        "REACTIONS:\n"
        f"  Ry (left)  = {analytical['reactions']['R1y']/1000:.4f} kN\n"
        f"  Ry (right) = {analytical['reactions']['R4y']/1000:.4f} kN\n\n"
        "BEAM:\n"
        f"  Max Moment     = {analytical['beam']['M_max']/1000:.4f} kN·m\n"
        f"  Max Shear      = {analytical['beam']['V_max']/1000:.4f} kN\n"
        f"  Max Deflection = {analytical['beam']['delta_max']*1000:.6f} mm\n\n"
        "COLUMNS:\n"
        f"  Axial Force = {analytical['columns']['N']/1000:.4f} kN (compression)\n"
        f"  Moment (top) = {analytical['columns']['M_top']/1000:.4f} kN·m\n"
        f"  Moment (btm) = {analytical['columns']['M_bottom']/1000:.4f} kN·m\n"
    )
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=11, fontfamily='monospace', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    output_file = f'{output_prefix}_analytical.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return fig


def plot_fem_vs_analytical(vtk_data: VTKData, members: List[Member], 
                            errors: Dict, params: FrameParameters,
                            output_prefix: str = "frame"):
    """Plot FEM vs Analytical comparison"""
    
    analytical = errors['analytical']
    L = params.L
    H = params.H
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FEM vs Analytical Comparison', fontsize=14, fontweight='bold')
    
    # 1. Deformed Shape Comparison
    ax1 = axes[0, 0]
    
    # Undeformed
    ax1.plot([0, 0], [0, H], 'k--', linewidth=1, alpha=0.5, label='Undeformed')
    ax1.plot([L, L], [0, H], 'k--', linewidth=1, alpha=0.5)
    ax1.plot([0, L], [H, H], 'k--', linewidth=1, alpha=0.5)
    
    # Analytical deflected shape
    x_beam = np.linspace(0, L, 50)
    beam_dist = PortalFrameAnalytical.get_beam_distributions(x_beam, params, 
                                                             params.frame_type.split('_')[1])
    
    scale = 1000
    anal_y = H - beam_dist['deflection'] * scale
    ax1.plot(x_beam, anal_y, 'b-', linewidth=2, label='Analytical')
    ax1.plot([0, 0], [0, H], 'b-', linewidth=2)
    ax1.plot([L, L], [0, H], 'b-', linewidth=2)
    
    # FEM deflected shape
    if 'DISPLACEMENT' in vtk_data.point_data:
        disp = vtk_data.point_data['DISPLACEMENT']
        
        for m in members:
            x_def = [vtk_data.points[m.node_i, 0] + disp[m.node_i, 0] * scale,
                    vtk_data.points[m.node_j, 0] + disp[m.node_j, 0] * scale]
            y_def = [vtk_data.points[m.node_i, 1] + disp[m.node_i, 1] * scale,
                    vtk_data.points[m.node_j, 1] + disp[m.node_j, 1] * scale]
            
            label = 'FEM' if m.id == 1 else ''
            ax1.plot(x_def, y_def, 'ro-', linewidth=2, markersize=8,
                    markerfacecolor='white', markeredgewidth=2, label=label)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Deformed Shape (Scale: {scale}x)', fontweight='bold')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Beam BMD Comparison
    ax2 = axes[0, 1]
    
    # Analytical
    ax2.plot(x_beam, beam_dist['moment'] / 1000, 'b-', linewidth=2, label='Analytical')
    ax2.fill_between(x_beam, 0, beam_dist['moment'] / 1000, alpha=0.3, color='blue')
    
    # FEM (if moment data available)
    if 'MOMENT' in vtk_data.point_data:
        moments = vtk_data.point_data['MOMENT']
        
        # Get beam nodes
        beam_nodes = []
        for i, point in enumerate(vtk_data.points):
            if abs(point[1] - H) < 1e-6:
                beam_nodes.append((point[0], i))
        
        beam_nodes.sort(key=lambda x: x[0])
        
        if beam_nodes:
            x_fem = [n[0] for n in beam_nodes]
            m_fem = [-moments[n[1], 2] / 1000 for n in beam_nodes]  # Sign convention
            ax2.plot(x_fem, m_fem, 'ro-', linewidth=2, markersize=10,
                    markerfacecolor='white', markeredgewidth=2, label='FEM')
    
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Position along beam (m)')
    ax2.set_ylabel('Bending Moment (kN·m)')
    ax2.set_title('Beam Bending Moment', fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Beam SFD Comparison
    ax3 = axes[1, 0]
    
    # Analytical
    ax3.plot(x_beam, beam_dist['shear'] / 1000, 'b-', linewidth=2, label='Analytical')
    ax3.fill_between(x_beam, 0, beam_dist['shear'] / 1000, alpha=0.3, color='blue')
    
    # FEM shear
    if 'FORCE' in vtk_data.point_data:
        forces = vtk_data.point_data['FORCE']
        
        beam_nodes = []
        for i, point in enumerate(vtk_data.points):
            if abs(point[1] - H) < 1e-6:
                beam_nodes.append((point[0], i))
        
        beam_nodes.sort(key=lambda x: x[0])
        
        if beam_nodes:
            x_fem = [n[0] for n in beam_nodes]
            v_fem = [-forces[n[1], 1] / 1000 for n in beam_nodes]  # Shear in Y direction
            ax3.plot(x_fem, v_fem, 'ro-', linewidth=2, markersize=10,
                    markerfacecolor='white', markeredgewidth=2, label='FEM')
    
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_xlabel('Position along beam (m)')
    ax3.set_ylabel('Shear Force (kN)')
    ax3.set_title('Beam Shear Force', fontweight='bold')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # 4. Error Summary Bar Chart
    ax4 = axes[1, 1]
    
    error_names = []
    error_values = []
    colors = []
    
    if 'reactions' in errors:
        error_names.append('Reaction Ry')
        error_values.append(errors['reactions']['R1y']['error_percent'])
        colors.append('#2E86AB')
    
    if 'deflection' in errors:
        error_names.append('Max Deflection')
        error_values.append(errors['deflection']['beam_max']['error_percent'])
        colors.append('#F18F01')
    
    if 'moment' in errors:
        error_names.append('Beam Moment')
        error_values.append(errors['moment']['beam_center']['error_percent'])
        colors.append('#A23B72')
    
    if error_names:
        bars = ax4.bar(error_names, error_values, color=colors, alpha=0.8, edgecolor='black')
        
        for bar, val in zip(bars, error_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax4.set_ylabel('Error (%)')
        ax4.set_title('Error Summary', fontweight='bold')
        ax4.grid(True, linestyle='--', alpha=0.5, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No error data available', transform=ax4.transAxes,
                ha='center', va='center', fontsize=12)
        ax4.set_title('Error Summary', fontweight='bold')
    
    plt.tight_layout()
    output_file = f'{output_prefix}_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return fig


def plot_error_distribution(vtk_data: VTKData, errors: Dict, params: FrameParameters,
                            output_prefix: str = "frame"):
    """Plot error distribution along frame members"""
    
    L = params.L
    H = params.H
    analytical = errors['analytical']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Error Distribution Along Frame Members', fontsize=14, fontweight='bold')
    
    # Get analytical distributions
    x_beam = np.linspace(0, L, 100)
    beam_dist = PortalFrameAnalytical.get_beam_distributions(x_beam, params, 
                                                             params.frame_type.split('_')[1])
    
    # 1. Deflection along beam
    ax1 = axes[0, 0]
    
    # Analytical
    ax1.plot(x_beam, -beam_dist['deflection'] * 1000, 'b-', linewidth=2, label='Analytical')
    
    # FEM points
    if 'DISPLACEMENT' in vtk_data.point_data:
        disp = vtk_data.point_data['DISPLACEMENT']
        
        beam_nodes = []
        for i, point in enumerate(vtk_data.points):
            if abs(point[1] - H) < 1e-6:
                beam_nodes.append((point[0], i))
        
        beam_nodes.sort(key=lambda x: x[0])
        
        if beam_nodes:
            x_fem = [n[0] for n in beam_nodes]
            d_fem = [disp[n[1], 1] * 1000 for n in beam_nodes]
            ax1.plot(x_fem, d_fem, 'ro', markersize=12, markerfacecolor='white',
                    markeredgewidth=2, label='FEM')
            
            # Calculate pointwise error
            for xi, di in zip(x_fem, d_fem):
                # Interpolate analytical value
                anal_val = np.interp(xi, x_beam, -beam_dist['deflection'] * 1000)
                if abs(anal_val) > 1e-10:
                    err = abs(di - anal_val) / abs(anal_val) * 100
                    ax1.annotate(f'{err:.1f}%', (xi, di), textcoords='offset points',
                               xytext=(5, 10), fontsize=9, color='red')
    
    ax1.set_xlabel('Position along beam (m)')
    ax1.set_ylabel('Deflection (mm)')
    ax1.set_title('Beam Deflection', fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Moment along beam
    ax2 = axes[0, 1]
    
    ax2.plot(x_beam, beam_dist['moment'] / 1000, 'b-', linewidth=2, label='Analytical')
    ax2.fill_between(x_beam, 0, beam_dist['moment'] / 1000, alpha=0.2, color='blue')
    
    if 'MOMENT' in vtk_data.point_data:
        moments = vtk_data.point_data['MOMENT']
        
        beam_nodes = []
        for i, point in enumerate(vtk_data.points):
            if abs(point[1] - H) < 1e-6:
                beam_nodes.append((point[0], i))
        
        beam_nodes.sort(key=lambda x: x[0])
        
        if beam_nodes:
            x_fem = [n[0] for n in beam_nodes]
            m_fem = [-moments[n[1], 2] / 1000 for n in beam_nodes]
            ax2.plot(x_fem, m_fem, 'ro', markersize=12, markerfacecolor='white',
                    markeredgewidth=2, label='FEM')
    
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Position along beam (m)')
    ax2.set_ylabel('Bending Moment (kN·m)')
    ax2.set_title('Beam Bending Moment', fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Pointwise error chart
    ax3 = axes[1, 0]
    
    if 'DISPLACEMENT' in vtk_data.point_data:
        disp = vtk_data.point_data['DISPLACEMENT']
        
        beam_nodes = []
        for i, point in enumerate(vtk_data.points):
            if abs(point[1] - H) < 1e-6:
                beam_nodes.append((point[0], i))
        
        beam_nodes.sort(key=lambda x: x[0])
        
        if len(beam_nodes) > 0:
            x_fem = [n[0] for n in beam_nodes]
            errors_disp = []
            
            for xi, (_, node_idx) in zip(x_fem, beam_nodes):
                fem_val = disp[node_idx, 1] * 1000
                anal_val = np.interp(xi, x_beam, -beam_dist['deflection'] * 1000)
                if abs(anal_val) > 1e-10:
                    errors_disp.append(abs(fem_val - anal_val))
                else:
                    errors_disp.append(abs(fem_val))
            
            ax3.bar(x_fem, errors_disp, width=0.1, color='#E63946', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Position along beam (m)')
            ax3.set_ylabel('Absolute Error (mm)')
            ax3.set_title('Deflection Absolute Error', fontweight='bold')
            ax3.grid(True, linestyle='--', alpha=0.5)
    
    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "ERROR SUMMARY\n"
    summary_text += "═" * 50 + "\n\n"
    
    if 'reactions' in errors:
        r = errors['reactions']
        summary_text += f"REACTIONS:\n"
        summary_text += f"  Left Ry:  Anal={r['R1y']['analytical']:.4f} kN, FEM={r['R1y']['fem']:.4f} kN, Error={r['R1y']['error_percent']:.4f}%\n"
        summary_text += f"  Right Ry: Anal={r['R4y']['analytical']:.4f} kN, FEM={r['R4y']['fem']:.4f} kN, Error={r['R4y']['error_percent']:.4f}%\n\n"
    
    if 'deflection' in errors:
        d = errors['deflection']['beam_max']
        summary_text += f"DEFLECTION:\n"
        summary_text += f"  Max: Anal={d['analytical']:.6f} mm, FEM={d['fem']:.6f} mm, Error={d['error_percent']:.2f}%\n\n"
    
    if 'moment' in errors:
        m = errors['moment']['beam_center']
        summary_text += f"MOMENT:\n"
        summary_text += f"  Center: Anal={m['analytical']:.4f} kN·m, FEM={m['fem']:.4f} kN·m, Error={m['error_percent']:.2f}%\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, fontfamily='monospace', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    output_file = f'{output_prefix}_error_distribution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return fig


def main():
    """Main function"""
    
    # =============== CONFIGURATION ===============
    VTK_FILE = "test_files/frame_2D_test_udl.gid/vtk_output/Parts_Beam_Beams_0_1.vtk"  # Your VTK file path
    OUTPUT_PREFIX = "test_files/frame_2D_test_udl.gid/plots/frame_analysis"
    
    # Frame parameters (modify according to your model)
    params = FrameParameters(
        L = 2.0,              # Beam span [m]
        A = 0.00287,          # Beam cross-sectional area [m^2]
        H = 2.0,              # Column height [m]
        w = 10000.0,          # UDL on beam [N/m] = 10 kN/m
        P = 0.0,              # Point load [N]
        E = 210e9,            # Young's modulus [Pa]
        I_beam = 5e-6,        # Beam moment of inertia [m^4]
        I_col = 5e-6,         # Column moment of inertia [m^4]
        A_beam = 0.00287,     # Beam cross-sectional area [m^2]
        A_col = 0.00287,      # Column cross-sectional area [m^2]
        frame_type = 'portal_pinned',  # 'portal_pinned' or 'portal_fixed'
        load_type = 'udl_beam'
    )
    # =============================================
    
    print("=" * 110)
    print("FRAME ANALYSIS RESULTS PLOTTER WITH ANALYTICAL VERIFICATION")
    print("=" * 110)
    
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
    members, frame_info = analyze_frame_geometry(vtk_data)
    
    # Update params with detected dimensions if needed
    if abs(params.L - frame_info['span']) > 0.01:
        print(f"  Note: Updating span from {params.L} to {frame_info['span']}")
        params.L = frame_info['span']
    if abs(params.H - frame_info['height']) > 0.01:
        print(f"  Note: Updating height from {params.H} to {frame_info['height']}")
        params.H = frame_info['height']
    
    # Calculate errors
    errors = calculate_frame_errors(vtk_data, members, params, frame_info)
    
    # Print results
    print_frame_results(vtk_data, members, errors, params)
    
    # Generate plots
    print("\n" + "-" * 110)
    print("Generating plots...")
    
    fig1 = plot_frame_system_diagram(params, OUTPUT_PREFIX)
    # fig2 = plot_analytical_diagrams(params, errors, OUTPUT_PREFIX)
    # fig3 = plot_fem_vs_analytical(vtk_data, members, errors, params, OUTPUT_PREFIX)
    # fig4 = plot_error_distribution(vtk_data, errors, params, OUTPUT_PREFIX)
    
    plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()