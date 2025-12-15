"""
Beam Analysis Results Plotter with Analytical Verification
Reads VTK output from Kratos and plots deflection/bending moment diagrams
Compares with analytical solutions and calculates errors
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import os


@dataclass
class VTKData:
    """Container for VTK file data"""
    points: np.ndarray = None
    cells: List[List[int]] = field(default_factory=list)
    point_data: Dict[str, np.ndarray] = field(default_factory=dict)
    cell_data: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class BeamParameters:
    """Beam properties for analytical solution"""
    L: float          # Length [m]
    w: float          # UDL [N/m] (for UDL case)
    P: float          # Point load [N] (for point load case)
    E: float          # Young's modulus [Pa]
    I: float          # Moment of inertia [m^4]
    load_type: str    # 'udl', 'point_center', 'point_any'
    load_position: float = None  # For point load at specific position


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
        
        # Parse POINTS
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
        
        # Parse CELLS
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
        
        # Parse FIELD data
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


def get_beam_axis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str, int]:
    """Determine beam axis and return sorted coordinates"""
    # Use np.ptp() instead of .ptp() for NumPy 2.0 compatibility
    ranges = np.ptp(points, axis=0)
    main_axis = np.argmax(ranges)
    axis_names = ['X', 'Y', 'Z']
    
    coords = points[:, main_axis]
    sort_idx = np.argsort(coords)
    
    return coords[sort_idx], sort_idx, axis_names[main_axis], main_axis


def get_transverse_axis(points: np.ndarray, main_axis: int) -> int:
    """Get the transverse (deflection) axis"""
    ranges = np.ptp(points, axis=0)
    ranges[main_axis] = -1
    return np.argmax(ranges) if np.max(ranges) > 0 else (1 if main_axis != 1 else 2)


def is_2d_beam(points: np.ndarray) -> bool:
    """Check if beam is 2D (Z variation negligible)"""
    z_range = np.ptp(points[:, 2])
    return z_range < 1e-10


class AnalyticalSolutions:
    """Analytical solutions for simply supported beams"""
    
    @staticmethod
    def simply_supported_udl(x: np.ndarray, params: BeamParameters) -> Dict[str, np.ndarray]:
        """
        Simply supported beam with uniformly distributed load
        """
        L, w, E, I = params.L, params.w, params.E, params.I
        
        deflection = w * x * (L**3 - 2*L*x**2 + x**3) / (24 * E * I)
        rotation = w * (L**3 - 6*L*x**2 + 4*x**3) / (24 * E * I)
        moment = w * L * x / 2 - w * x**2 / 2
        shear = w * L / 2 - w * x
        
        return {
            'deflection': deflection,
            'rotation': rotation,
            'moment': moment,
            'shear': shear,
            'max_deflection': 5 * w * L**4 / (384 * E * I),
            'max_moment': w * L**2 / 8,
            'max_shear': w * L / 2,
            'end_rotation': w * L**3 / (24 * E * I),
            'reaction': w * L / 2
        }
    
    @staticmethod
    def simply_supported_point_center(x: np.ndarray, params: BeamParameters) -> Dict[str, np.ndarray]:
        """
        Simply supported beam with point load at center
        """
        L, P, E, I = params.L, params.P, params.E, params.I
        
        deflection = np.zeros_like(x)
        moment = np.zeros_like(x)
        shear = np.zeros_like(x)
        rotation = np.zeros_like(x)
        
        for i, xi in enumerate(x):
            if xi <= L/2:
                deflection[i] = P * xi * (3*L**2 - 4*xi**2) / (48 * E * I)
                moment[i] = P * xi / 2
                shear[i] = P / 2
                rotation[i] = P * (L**2 - 4*xi**2) / (16 * E * I)
            else:
                xi_sym = L - xi
                deflection[i] = P * xi_sym * (3*L**2 - 4*xi_sym**2) / (48 * E * I)
                moment[i] = P * xi_sym / 2
                shear[i] = -P / 2
                rotation[i] = -P * (L**2 - 4*xi_sym**2) / (16 * E * I)
        
        return {
            'deflection': deflection,
            'rotation': rotation,
            'moment': moment,
            'shear': shear,
            'max_deflection': P * L**3 / (48 * E * I),
            'max_moment': P * L / 4,
            'max_shear': P / 2,
            'end_rotation': P * L**2 / (16 * E * I),
            'reaction': P / 2
        }
    
    @staticmethod
    def cantilever_udl(x: np.ndarray, params: BeamParameters) -> Dict[str, np.ndarray]:
        """
        Cantilever beam with UDL (fixed at x=0, free at x=L)
        """
        L, w, E, I = params.L, params.w, params.E, params.I
        
        deflection = w * x**2 * (6*L**2 - 4*L*x + x**2) / (24 * E * I)
        rotation = w * x * (3*L**2 - 3*L*x + x**2) / (6 * E * I)
        moment = -w * (L - x)**2 / 2
        shear = w * (L - x)
        
        return {
            'deflection': deflection,
            'rotation': rotation,
            'moment': moment,
            'shear': shear,
            'max_deflection': w * L**4 / (8 * E * I),
            'max_moment': w * L**2 / 2,
            'max_shear': w * L,
            'end_rotation': w * L**3 / (6 * E * I),
            'reaction': w * L
        }
    
    @staticmethod
    def cantilever_point_end(x: np.ndarray, params: BeamParameters) -> Dict[str, np.ndarray]:
        """
        Cantilever beam with point load at free end
        """
        L, P, E, I = params.L, params.P, params.E, params.I
        
        deflection = P * x**2 * (3*L - x) / (6 * E * I)
        rotation = P * x * (2*L - x) / (2 * E * I)
        moment = -P * (L - x)
        shear = np.full_like(x, P)
        
        return {
            'deflection': deflection,
            'rotation': rotation,
            'moment': moment,
            'shear': shear,
            'max_deflection': P * L**3 / (3 * E * I),
            'max_moment': P * L,
            'max_shear': P,
            'end_rotation': P * L**2 / (2 * E * I),
            'reaction': P
        }


def calculate_errors(vtk_data: VTKData, params: BeamParameters, 
                     beam_type: str = 'simply_supported') -> Dict[str, Dict]:
    """
    Calculate errors between FEM and analytical solutions
    """
    
    x_sorted, sort_idx, _, main_axis_idx = get_beam_axis(vtk_data.points)
    trans_axis = get_transverse_axis(vtk_data.points, main_axis_idx)
    is_2d = is_2d_beam(vtk_data.points)
    
    rot_axis = 2 if is_2d else (0 if main_axis_idx == 1 else 1)
    moment_axis = rot_axis
    
    # Get analytical solution
    if beam_type == 'simply_supported':
        if params.load_type == 'udl':
            analytical = AnalyticalSolutions.simply_supported_udl(x_sorted, params)
        elif params.load_type == 'point_center':
            analytical = AnalyticalSolutions.simply_supported_point_center(x_sorted, params)
        else:
            analytical = AnalyticalSolutions.simply_supported_udl(x_sorted, params)
    elif beam_type == 'cantilever':
        if params.load_type == 'udl':
            analytical = AnalyticalSolutions.cantilever_udl(x_sorted, params)
        elif params.load_type == 'point_end':
            analytical = AnalyticalSolutions.cantilever_point_end(x_sorted, params)
        else:
            analytical = AnalyticalSolutions.cantilever_udl(x_sorted, params)
    else:
        analytical = AnalyticalSolutions.simply_supported_udl(x_sorted, params)
    
    errors = {}
    
    # Deflection error
    if 'DISPLACEMENT' in vtk_data.point_data:
        disp_fem = vtk_data.point_data['DISPLACEMENT'][sort_idx, trans_axis]
        disp_anal = -analytical['deflection']
        
        max_anal = np.max(np.abs(disp_anal))
        if max_anal > 1e-15:
            pointwise_error = np.abs(disp_fem - disp_anal)
            
            errors['deflection'] = {
                'fem_values': disp_fem * 1000,
                'analytical_values': disp_anal * 1000,
                'fem_max': np.min(disp_fem) * 1000,
                'analytical_max': analytical['max_deflection'] * 1000,
                'max_error_percent': np.abs(np.min(disp_fem) + analytical['max_deflection']) / analytical['max_deflection'] * 100,
                'pointwise_errors': pointwise_error * 1000,
                'rms_error': np.sqrt(np.mean((disp_fem - disp_anal)**2)) * 1000,
                'positions': x_sorted
            }
    
    # Rotation error
    if 'ROTATION' in vtk_data.point_data:
        rot_fem = vtk_data.point_data['ROTATION'][sort_idx, rot_axis]
        rot_anal = analytical['rotation']
        
        max_anal = np.max(np.abs(rot_anal))
        if max_anal > 1e-15:
            end_rot_fem = np.abs(rot_fem[0])
            end_rot_anal = analytical['end_rotation']
            
            errors['rotation'] = {
                'fem_values': rot_fem * 1000,
                'analytical_values': rot_anal * 1000,
                'fem_end': end_rot_fem * 1000,
                'analytical_end': end_rot_anal * 1000,
                'end_error_percent': np.abs(end_rot_fem - end_rot_anal) / end_rot_anal * 100 if end_rot_anal > 1e-15 else 0,
                'rms_error': np.sqrt(np.mean((rot_fem - rot_anal)**2)) * 1000,
                'positions': x_sorted
            }
    
    # Moment error
    if 'MOMENT' in vtk_data.point_data:
        moment_fem = -vtk_data.point_data['MOMENT'][sort_idx, moment_axis]
        moment_anal = analytical['moment']
        
        max_anal = np.max(np.abs(moment_anal))
        if max_anal > 1e-15:
            center_idx = len(x_sorted) // 2
            moment_fem_max = moment_fem[center_idx]
            moment_anal_max = analytical['max_moment']
            
            errors['moment'] = {
                'fem_values': moment_fem / 1000,
                'analytical_values': moment_anal / 1000,
                'fem_max': np.max(np.abs(moment_fem)) / 1000,
                'analytical_max': moment_anal_max / 1000,
                'max_error_percent': np.abs(np.max(np.abs(moment_fem)) - moment_anal_max) / moment_anal_max * 100,
                'rms_error': np.sqrt(np.mean((moment_fem - moment_anal)**2)) / 1000,
                'positions': x_sorted
            }
    
    # Shear error
    if 'FORCE' in vtk_data.point_data:
        shear_fem = -vtk_data.point_data['FORCE'][sort_idx, trans_axis]
        shear_anal = analytical['shear']
        
        max_anal = np.max(np.abs(shear_anal))
        if max_anal > 1e-15:
            shear_fem_max = np.max(np.abs(shear_fem))
            shear_anal_max = analytical['max_shear']
            
            errors['shear'] = {
                'fem_values': shear_fem / 1000,
                'analytical_values': shear_anal / 1000,
                'fem_max': shear_fem_max / 1000,
                'analytical_max': shear_anal_max / 1000,
                'max_error_percent': np.abs(shear_fem_max - shear_anal_max) / shear_anal_max * 100,
                'rms_error': np.sqrt(np.mean((shear_fem - shear_anal)**2)) / 1000,
                'positions': x_sorted
            }
    
    # Reaction error
    if 'REACTION' in vtk_data.point_data:
        reactions = vtk_data.point_data['REACTION']
        reaction_fem = np.abs(reactions[sort_idx[0], trans_axis])
        reaction_anal = analytical['reaction']
        
        if reaction_anal > 1e-15:
            errors['reaction'] = {
                'fem_value': reaction_fem / 1000,
                'analytical_value': reaction_anal / 1000,
                'error_percent': np.abs(reaction_fem - reaction_anal) / reaction_anal * 100
            }
    
    # Store analytical for plotting
    errors['analytical'] = analytical
    errors['x_fine'] = np.linspace(0, params.L, 200)
    
    # Calculate fine analytical solution for plotting
    if beam_type == 'simply_supported':
        if params.load_type == 'udl':
            errors['analytical_fine'] = AnalyticalSolutions.simply_supported_udl(errors['x_fine'], params)
        elif params.load_type == 'point_center':
            errors['analytical_fine'] = AnalyticalSolutions.simply_supported_point_center(errors['x_fine'], params)
        else:
            errors['analytical_fine'] = AnalyticalSolutions.simply_supported_udl(errors['x_fine'], params)
    elif beam_type == 'cantilever':
        if params.load_type == 'udl':
            errors['analytical_fine'] = AnalyticalSolutions.cantilever_udl(errors['x_fine'], params)
        elif params.load_type == 'point_end':
            errors['analytical_fine'] = AnalyticalSolutions.cantilever_point_end(errors['x_fine'], params)
        else:
            errors['analytical_fine'] = AnalyticalSolutions.cantilever_udl(errors['x_fine'], params)
    else:
        errors['analytical_fine'] = AnalyticalSolutions.simply_supported_udl(errors['x_fine'], params)
    
    return errors


def print_results_with_errors(vtk_data: VTKData, errors: Dict, params: BeamParameters):
    """Print formatted results table with analytical comparison and errors"""
    
    x_sorted, sort_idx, axis_name, main_axis_idx = get_beam_axis(vtk_data.points)
    
    print("\n" + "=" * 100)
    print("BEAM ANALYSIS RESULTS WITH ANALYTICAL VERIFICATION")
    print("=" * 100)
    
    # Input parameters
    print(f"\n{'INPUT PARAMETERS':^100}")
    print("-" * 100)
    print(f"  Beam Length (L):        {params.L} m")
    if params.load_type == 'udl':
        print(f"  UDL (w):                {params.w/1000} kN/m")
    else:
        print(f"  Point Load (P):         {params.P/1000} kN")
    print(f"  Young's Modulus (E):    {params.E/1e9} GPa")
    print(f"  Moment of Inertia (I):  {params.I*1e6:.4f} × 10⁻⁶ m⁴")
    print(f"  Number of Elements:     {len(vtk_data.cells)}")
    print(f"  Number of Nodes:        {len(vtk_data.points)}")
    
    # Nodal results table
    print(f"\n{'NODAL RESULTS':^100}")
    print("-" * 100)
    print(f"{'Node':<6} {axis_name+'(m)':<10} {'Disp(mm)':<14} {'Rot(mrad)':<14} "
          f"{'Moment(kN·m)':<16} {'Shear(kN)':<14}")
    print("-" * 100)
    
    n_nodes = len(vtk_data.points)
    
    for i, idx in enumerate(sort_idx):
        node_id = idx + 1
        x = vtk_data.points[idx, main_axis_idx]
        
        disp_val = errors.get('deflection', {}).get('fem_values', np.zeros(n_nodes))[i] if 'deflection' in errors else 0
        rot_val = errors.get('rotation', {}).get('fem_values', np.zeros(n_nodes))[i] if 'rotation' in errors else 0
        moment_val = errors.get('moment', {}).get('fem_values', np.zeros(n_nodes))[i] if 'moment' in errors else 0
        shear_val = errors.get('shear', {}).get('fem_values', np.zeros(n_nodes))[i] if 'shear' in errors else 0
        
        print(f"{node_id:<6} {x:<10.4f} {disp_val:<14.6f} {rot_val:<14.6f} "
              f"{moment_val:<16.4f} {shear_val:<14.4f}")
    
    print("=" * 100)
    
    # Comparison with analytical solution
    print(f"\n{'COMPARISON WITH ANALYTICAL SOLUTION':^100}")
    print("=" * 100)
    
    print(f"\n{'Parameter':<25} {'Analytical':<20} {'FEM':<20} {'Error (%)':<15}")
    print("-" * 80)
    
    if 'deflection' in errors:
        d = errors['deflection']
        print(f"{'Max Deflection (mm)':<25} {d['analytical_max']:<20.6f} {abs(d['fem_max']):<20.6f} {d['max_error_percent']:<15.2f}")
    
    if 'rotation' in errors:
        r = errors['rotation']
        print(f"{'End Rotation (mrad)':<25} {r['analytical_end']:<20.6f} {r['fem_end']:<20.6f} {r['end_error_percent']:<15.2f}")
    
    if 'moment' in errors:
        m = errors['moment']
        print(f"{'Max Moment (kN·m)':<25} {m['analytical_max']:<20.4f} {m['fem_max']:<20.4f} {m['max_error_percent']:<15.2f}")
    
    if 'shear' in errors:
        s = errors['shear']
        print(f"{'Max Shear (kN)':<25} {s['analytical_max']:<20.4f} {s['fem_max']:<20.4f} {s['max_error_percent']:<15.2f}")
    
    if 'reaction' in errors:
        rx = errors['reaction']
        print(f"{'Support Reaction (kN)':<25} {rx['analytical_value']:<20.4f} {rx['fem_value']:<20.4f} {rx['error_percent']:<15.4f}")
    
    print("=" * 100)
    
    # RMS Errors
    print(f"\n{'RMS ERRORS':^100}")
    print("-" * 100)
    
    if 'deflection' in errors:
        print(f"  Deflection RMS Error:   {errors['deflection']['rms_error']:.6f} mm")
    if 'rotation' in errors:
        print(f"  Rotation RMS Error:     {errors['rotation']['rms_error']:.6f} mrad")
    if 'moment' in errors:
        print(f"  Moment RMS Error:       {errors['moment']['rms_error']:.4f} kN·m")
    if 'shear' in errors:
        print(f"  Shear RMS Error:        {errors['shear']['rms_error']:.4f} kN")
    
    print("=" * 100)
    
    # Support reactions
    if 'REACTION' in vtk_data.point_data:
        print(f"\n{'SUPPORT REACTIONS':^100}")
        print("-" * 100)
        reactions = vtk_data.point_data['REACTION']
        for idx in sort_idx:
            rx, ry, rz = reactions[idx]
            if abs(rx) > 1 or abs(ry) > 1 or abs(rz) > 1:
                print(f"  Node {idx+1}: Rx = {rx/1000:>10.4f} kN,  Ry = {ry/1000:>10.4f} kN,  Rz = {rz/1000:>10.4f} kN")
        print("=" * 100)


def plot_results_with_analytical(vtk_data: VTKData, errors: Dict, 
                                  output_prefix: str = "beam"):
    """Plot FEM results with analytical comparison"""
    
    x_sorted, sort_idx, axis_name, _ = get_beam_axis(vtk_data.points)
    x_fine = errors['x_fine']
    anal_fine = errors['analytical_fine']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Beam Analysis: FEM vs Analytical Solution', fontsize=14, fontweight='bold')
    
    colors = {'fem': '#E63946', 'analytical': '#1D3557'}
    
    # 1. DEFLECTION
    ax1 = axes[0, 0]
    ax1.plot(x_fine, -anal_fine['deflection'] * 1000, '-', color=colors['analytical'], 
             linewidth=2, label='Analytical')
    if 'deflection' in errors:
        ax1.plot(x_sorted, errors['deflection']['fem_values'], 'o', color=colors['fem'],
                markersize=10, markerfacecolor='white', markeredgewidth=2, label='FEM')
        
        err = errors['deflection']['max_error_percent']
        ax1.text(0.98, 0.02, f'Max Error: {err:.2f}%', transform=ax1.transAxes,
                ha='right', va='bottom', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel(f'Position [{axis_name}] (m)', fontsize=11)
    ax1.set_ylabel('Deflection (mm)', fontsize=11)
    ax1.set_title('Deflection Diagram', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.axhline(y=0, color='black', linewidth=0.8)
    
    # 2. ROTATION
    ax2 = axes[0, 1]
    ax2.plot(x_fine, anal_fine['rotation'] * 1000, '-', color=colors['analytical'],
             linewidth=2, label='Analytical')
    if 'rotation' in errors:
        ax2.plot(x_sorted, -errors['rotation']['fem_values'], 'o', color=colors['fem'],
                markersize=10, markerfacecolor='white', markeredgewidth=2, label='FEM')
        
        err = errors['rotation']['end_error_percent']
        ax2.text(0.98, 0.02, f'End Error: {err:.2f}%', transform=ax2.transAxes,
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_xlabel(f'Position [{axis_name}] (m)', fontsize=11)
    ax2.set_ylabel('Rotation (mrad)', fontsize=11)
    ax2.set_title('Rotation Diagram', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    
    # 3. BENDING MOMENT
    ax3 = axes[1, 0]
    ax3.plot(x_fine, anal_fine['moment'] / 1000, '-', color=colors['analytical'],
             linewidth=2, label='Analytical')
    if 'moment' in errors:
        ax3.plot(x_sorted, errors['moment']['fem_values'], 'o', color=colors['fem'],
                markersize=10, markerfacecolor='white', markeredgewidth=2, label='FEM')
        
        err = errors['moment']['max_error_percent']
        ax3.text(0.98, 0.02, f'Max Error: {err:.2f}%', transform=ax3.transAxes,
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax3.set_xlabel(f'Position [{axis_name}] (m)', fontsize=11)
    ax3.set_ylabel('Bending Moment (kN·m)', fontsize=11)
    ax3.set_title('Bending Moment Diagram (BMD)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.axhline(y=0, color='black', linewidth=0.8)
    
    # 4. SHEAR FORCE
    ax4 = axes[1, 1]
    ax4.plot(x_fine, anal_fine['shear'] / 1000, '-', color=colors['analytical'],
             linewidth=2, label='Analytical')
    if 'shear' in errors:
        ax4.plot(x_sorted, errors['shear']['fem_values'], 'o', color=colors['fem'],
                markersize=10, markerfacecolor='white', markeredgewidth=2, label='FEM')
        
        err = errors['shear']['max_error_percent']
        ax4.text(0.98, 0.02, f'Max Error: {err:.2f}%', transform=ax4.transAxes,
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax4.set_xlabel(f'Position [{axis_name}] (m)', fontsize=11)
    ax4.set_ylabel('Shear Force (kN)', fontsize=11)
    ax4.set_title('Shear Force Diagram (SFD)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.axhline(y=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    
    output_file = f'{output_prefix}_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return fig


def plot_error_distribution(errors: Dict, output_prefix: str = "beam"):
    """Plot error distribution along beam"""
    
    if 'deflection' not in errors:
        return None
    
    x = errors['deflection']['positions']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Error Distribution Along Beam', fontsize=14, fontweight='bold')
    
    color = '#E63946'
    
    # 1. Deflection Error
    ax1 = axes[0, 0]
    if 'deflection' in errors:
        err = errors['deflection']['pointwise_errors']
        ax1.bar(x, err, width=0.05, color=color, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Absolute Error (mm)')
        ax1.set_title('Deflection Error')
    ax1.set_xlabel('Position (m)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Moment comparison
    ax2 = axes[0, 1]
    if 'moment' in errors:
        fem_vals = errors['moment']['fem_values']
        anal_vals = errors['moment']['analytical_values']
        
        width = 0.03
        ax2.bar(x - width/2, anal_vals, width, label='Analytical', color='#1D3557', alpha=0.7)
        ax2.bar(x + width/2, fem_vals, width, label='FEM', color='#E63946', alpha=0.7)
        ax2.set_ylabel('Moment (kN·m)')
        ax2.set_title('Moment Comparison')
        ax2.legend()
    ax2.set_xlabel('Position (m)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Shear comparison
    ax3 = axes[1, 0]
    if 'shear' in errors:
        fem_vals = errors['shear']['fem_values']
        anal_vals = errors['shear']['analytical_values']
        
        width = 0.03
        ax3.bar(x - width/2, anal_vals, width, label='Analytical', color='#1D3557', alpha=0.7)
        ax3.bar(x + width/2, fem_vals, width, label='FEM', color='#E63946', alpha=0.7)
        ax3.set_ylabel('Shear (kN)')
        ax3.set_title('Shear Force Comparison')
        ax3.legend()
    ax3.set_xlabel('Position (m)')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Summary error bar chart
    ax4 = axes[1, 1]
    error_names = []
    error_values = []
    
    if 'deflection' in errors:
        error_names.append('Deflection')
        error_values.append(errors['deflection']['max_error_percent'])
    if 'rotation' in errors:
        error_names.append('Rotation')
        error_values.append(errors['rotation']['end_error_percent'])
    if 'moment' in errors:
        error_names.append('Moment')
        error_values.append(errors['moment']['max_error_percent'])
    if 'shear' in errors:
        error_names.append('Shear')
        error_values.append(errors['shear']['max_error_percent'])
    
    bar_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(error_names)]
    bars = ax4.bar(error_names, error_values, color=bar_colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Error (%)')
    ax4.set_title('Maximum Errors Summary')
    ax4.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    for bar, val in zip(bars, error_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = f'{output_prefix}_errors.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return fig


def main():
    """Main function"""
    
    # =============== CONFIGURATION ===============
    VTK_FILE = "test_files/beam_2D_test_udl.gid/vtk_output/Parts_Beam_Beams_0_1.vtk"
    OUTPUT_PREFIX = "test_files/beam_2D_test_udl.gid/plots/beam_results"
    
    BEAM_TYPE = 'simply_supported'  # 'simply_supported' or 'cantilever'
    LOAD_TYPE = 'udl'  # 'udl', 'point_center', 'point_end'
    
    params = BeamParameters(
        L = 2.0,
        w = 10000.0,
        P = 0.0,
        E = 210000000000.0,
        I = 5e-6,
        load_type = LOAD_TYPE
    )
    # =============================================
    
    print("=" * 100)
    print("BEAM ANALYSIS RESULTS PLOTTER WITH ANALYTICAL VERIFICATION")
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
    print(f"  Elements: {len(vtk_data.cells)}")
    print(f"  Point data: {list(vtk_data.point_data.keys())}")
    
    # Calculate errors
    errors = calculate_errors(vtk_data, params, BEAM_TYPE)
    
    # Print results
    print_results_with_errors(vtk_data, errors, params)
    
    # Generate plots
    print("\n" + "-" * 100)
    print("Generating plots...")
    
    fig1 = plot_results_with_analytical(vtk_data, errors, OUTPUT_PREFIX)
    fig2 = plot_error_distribution(errors, OUTPUT_PREFIX)
    
    plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()