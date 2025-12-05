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
    L: float
    w: float
    P: float
    E: float
    I: float
    load_type: str
    load_position: float = None


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


def get_beam_axis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str, int]:
    """Determine beam axis and return sorted coordinates"""
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
    """Analytical solutions for beams"""
    
    @staticmethod
    def simply_supported_udl(x: np.ndarray, params: BeamParameters) -> Dict[str, np.ndarray]:
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


def check_mesh_quality(vtk_data: VTKData, params: BeamParameters) -> Dict:
    """Check mesh quality and provide warnings"""
    
    n_nodes = len(vtk_data.points)
    n_elements = len(vtk_data.cells)
    
    warnings = []
    
    if n_elements < 2:
        warnings.append(f"CRITICAL: Only {n_elements} element(s)! Need at least 2 elements for meaningful results.")
    elif n_elements < 4:
        warnings.append(f"WARNING: Only {n_elements} elements. Results may be inaccurate. Recommend 10+ elements.")
    elif n_elements < 10:
        warnings.append(f"NOTE: {n_elements} elements. Consider using 10+ elements for higher accuracy.")
    
    if n_nodes <= 2:
        warnings.append("CRITICAL: Only boundary nodes present. Cannot capture interior deflection/moment.")
    
    all_zero = {}
    for field_name in ['DISPLACEMENT', 'ROTATION', 'MOMENT', 'FORCE']:
        if field_name in vtk_data.point_data:
            data = vtk_data.point_data[field_name]
            if np.allclose(data, 0, atol=1e-15):
                all_zero[field_name] = True
                warnings.append(f"WARNING: {field_name} values are all zero at nodes.")
    
    return {
        'n_nodes': n_nodes,
        'n_elements': n_elements,
        'warnings': warnings,
        'all_zero': all_zero,
        'has_interior_nodes': n_nodes > 2
    }


def calculate_errors(vtk_data: VTKData, params: BeamParameters, 
                     beam_type: str = 'simply_supported') -> Dict[str, Dict]:
    """Calculate errors between FEM and analytical solutions"""
    
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
    
    # Mesh quality check
    mesh_info = check_mesh_quality(vtk_data, params)
    errors['mesh_info'] = mesh_info
    
    # Deflection
    if 'DISPLACEMENT' in vtk_data.point_data:
        disp_fem = vtk_data.point_data['DISPLACEMENT'][sort_idx, trans_axis]
        disp_anal = -analytical['deflection']
        
        fem_max = np.min(disp_fem) if np.any(disp_fem != 0) else 0
        anal_max = analytical['max_deflection']
        pointwise_error = np.abs(disp_fem - disp_anal)
        
        errors['deflection'] = {
            'fem_values': disp_fem * 1000,
            'analytical_values': disp_anal * 1000,
            'fem_max': fem_max * 1000,
            'analytical_max': anal_max * 1000,
            'max_error_percent': abs(abs(fem_max) - anal_max) / anal_max * 100 if anal_max > 1e-15 else 0,
            'pointwise_errors': pointwise_error * 1000,
            'rms_error': np.sqrt(np.mean((disp_fem - disp_anal)**2)) * 1000,
            'positions': x_sorted,
            'is_all_zero': np.allclose(disp_fem, 0, atol=1e-15)
        }
    
    # Rotation
    if 'ROTATION' in vtk_data.point_data:
        rot_fem = vtk_data.point_data['ROTATION'][sort_idx, rot_axis]
        rot_anal = analytical['rotation']
        
        end_rot_fem = np.abs(rot_fem[0]) if len(rot_fem) > 0 else 0
        end_rot_anal = analytical['end_rotation']
        pointwise_error = np.abs(rot_fem - rot_anal)
        
        errors['rotation'] = {
            'fem_values': rot_fem * 1000,
            'analytical_values': rot_anal * 1000,
            'fem_end': end_rot_fem * 1000,
            'analytical_end': end_rot_anal * 1000,
            'end_error_percent': abs(end_rot_fem - end_rot_anal) / end_rot_anal * 100 if end_rot_anal > 1e-15 else 0,
            'pointwise_errors': pointwise_error * 1000,
            'rms_error': np.sqrt(np.mean((rot_fem - rot_anal)**2)) * 1000,
            'positions': x_sorted,
            'is_all_zero': np.allclose(rot_fem, 0, atol=1e-15)
        }
    
    # Moment
    if 'MOMENT' in vtk_data.point_data:
        moment_fem = -vtk_data.point_data['MOMENT'][sort_idx, moment_axis]
        moment_anal = analytical['moment']
        
        moment_anal_max = analytical['max_moment']
        moment_fem_max = np.max(np.abs(moment_fem)) if np.any(moment_fem != 0) else 0
        pointwise_error = np.abs(moment_fem - moment_anal)
        
        errors['moment'] = {
            'fem_values': moment_fem / 1000,
            'analytical_values': moment_anal / 1000,
            'fem_max': moment_fem_max / 1000,
            'analytical_max': moment_anal_max / 1000,
            'max_error_percent': abs(moment_fem_max - moment_anal_max) / moment_anal_max * 100 if moment_anal_max > 1e-15 else 0,
            'pointwise_errors': pointwise_error / 1000,
            'rms_error': np.sqrt(np.mean((moment_fem - moment_anal)**2)) / 1000,
            'positions': x_sorted,
            'is_all_zero': np.allclose(moment_fem, 0, atol=1e-15)
        }
    
    # Shear
    if 'FORCE' in vtk_data.point_data:
        shear_fem = -vtk_data.point_data['FORCE'][sort_idx, trans_axis]
        shear_anal = analytical['shear']
        
        shear_anal_max = analytical['max_shear']
        shear_fem_max = np.max(np.abs(shear_fem)) if np.any(shear_fem != 0) else 0
        pointwise_error = np.abs(shear_fem - shear_anal)
        
        errors['shear'] = {
            'fem_values': shear_fem / 1000,
            'analytical_values': shear_anal / 1000,
            'fem_max': shear_fem_max / 1000,
            'analytical_max': shear_anal_max / 1000,
            'max_error_percent': abs(shear_fem_max - shear_anal_max) / shear_anal_max * 100 if shear_anal_max > 1e-15 else 0,
            'pointwise_errors': pointwise_error / 1000,
            'rms_error': np.sqrt(np.mean((shear_fem - shear_anal)**2)) / 1000,
            'positions': x_sorted,
            'is_all_zero': np.allclose(shear_fem, 0, atol=1e-15)
        }
    
    # Reaction
    if 'REACTION' in vtk_data.point_data:
        reactions = vtk_data.point_data['REACTION']
        reaction_fem = np.abs(reactions[sort_idx[0], trans_axis])
        reaction_anal = analytical['reaction']
        
        errors['reaction'] = {
            'fem_value': reaction_fem / 1000,
            'analytical_value': reaction_anal / 1000,
            'error_percent': abs(reaction_fem - reaction_anal) / reaction_anal * 100 if reaction_anal > 1e-15 else 0
        }
    
    # Fine grid for plotting
    errors['analytical'] = analytical
    errors['x_fine'] = np.linspace(0, params.L, 200)
    
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
    
    # Mesh quality warnings
    if 'mesh_info' in errors:
        mesh_info = errors['mesh_info']
        if mesh_info['warnings']:
            print(f"\n{'⚠️  MESH QUALITY WARNINGS ⚠️':^100}")
            print("-" * 100)
            for warning in mesh_info['warnings']:
                print(f"  • {warning}")
            print("-" * 100)
    
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
    print(f"\n{'NODAL RESULTS (FEM)':^100}")
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
    
    # Analytical values at node positions
    print(f"\n{'ANALYTICAL VALUES AT NODE POSITIONS':^100}")
    print("-" * 100)
    print(f"{'Node':<6} {axis_name+'(m)':<10} {'Disp(mm)':<14} {'Rot(mrad)':<14} "
          f"{'Moment(kN·m)':<16} {'Shear(kN)':<14}")
    print("-" * 100)
    
    for i, idx in enumerate(sort_idx):
        node_id = idx + 1
        x = vtk_data.points[idx, main_axis_idx]
        
        disp_val = errors.get('deflection', {}).get('analytical_values', np.zeros(n_nodes))[i] if 'deflection' in errors else 0
        rot_val = errors.get('rotation', {}).get('analytical_values', np.zeros(n_nodes))[i] if 'rotation' in errors else 0
        moment_val = errors.get('moment', {}).get('analytical_values', np.zeros(n_nodes))[i] if 'moment' in errors else 0
        shear_val = errors.get('shear', {}).get('analytical_values', np.zeros(n_nodes))[i] if 'shear' in errors else 0
        
        print(f"{node_id:<6} {x:<10.4f} {disp_val:<14.6f} {rot_val:<14.6f} "
              f"{moment_val:<16.4f} {shear_val:<14.4f}")
    
    print("=" * 100)
    
    # Comparison
    print(f"\n{'COMPARISON: MAXIMUM VALUES':^100}")
    print("=" * 100)
    
    print(f"\n{'Parameter':<25} {'Analytical':<20} {'FEM':<20} {'Error (%)':<15} {'Status':<15}")
    print("-" * 95)
    
    if 'deflection' in errors:
        d = errors['deflection']
        status = "⚠️ ALL ZERO" if d.get('is_all_zero', False) else "✓"
        print(f"{'Max Deflection (mm)':<25} {d['analytical_max']:<20.6f} {abs(d['fem_max']):<20.6f} {d['max_error_percent']:<15.2f} {status:<15}")
    
    if 'rotation' in errors:
        r = errors['rotation']
        status = "⚠️ ALL ZERO" if r.get('is_all_zero', False) else "✓"
        print(f"{'End Rotation (mrad)':<25} {r['analytical_end']:<20.6f} {r['fem_end']:<20.6f} {r['end_error_percent']:<15.2f} {status:<15}")
    
    if 'moment' in errors:
        m = errors['moment']
        status = "⚠️ ALL ZERO" if m.get('is_all_zero', False) else "✓"
        print(f"{'Max Moment (kN·m)':<25} {m['analytical_max']:<20.4f} {m['fem_max']:<20.4f} {m['max_error_percent']:<15.2f} {status:<15}")
    
    if 'shear' in errors:
        s = errors['shear']
        status = "⚠️ ALL ZERO" if s.get('is_all_zero', False) else "✓"
        print(f"{'Max Shear (kN)':<25} {s['analytical_max']:<20.4f} {s['fem_max']:<20.4f} {s['max_error_percent']:<15.2f} {status:<15}")
    
    if 'reaction' in errors:
        rx = errors['reaction']
        print(f"{'Support Reaction (kN)':<25} {rx['analytical_value']:<20.4f} {rx['fem_value']:<20.4f} {rx['error_percent']:<15.4f} {'✓':<15}")
    
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
    
    mesh_info = errors.get('mesh_info', {})
    n_elements = mesh_info.get('n_elements', len(vtk_data.cells))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    title = 'Beam Analysis: FEM vs Analytical Solution'
    if n_elements < 2:
        title += f'\n⚠️ WARNING: Only {n_elements} element(s) - Insufficient mesh!'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    colors = {'fem': '#E63946', 'analytical': '#1D3557'}
    
    # Helper function for plotting each diagram
    def plot_diagram(ax, x_fine, anal_fine_data, fem_data, errors_data, 
                     ylabel, title, is_zero_key='is_all_zero', 
                     error_key='max_error_percent', error_label='Max Error'):
        
        ax.plot(x_fine, anal_fine_data, '-', color=colors['analytical'], 
                linewidth=2, label='Analytical')
        
        if errors_data is not None:
            is_zero = errors_data.get(is_zero_key, False)
            ax.plot(x_sorted, fem_data, 'o', color=colors['fem'],
                    markersize=10, markerfacecolor='white', markeredgewidth=2, 
                    label=f'FEM {"(all zero)" if is_zero else ""}')
            
            if is_zero:
                ax.text(0.5, 0.5, 'FEM: All values zero\n(Need more elements)', 
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=8, color='red', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
            else:
                err = errors_data.get(error_key, 0)
                ax.text(0.98, 0.02, f'{error_label}: {err:.2f}%', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=8, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel(f'Position [{axis_name}] (m)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=0.8)
        x_min=np.min(x_fine)
        x_max=np.max(x_fine)
        y_min=np.min([np.min(anal_fine_data), np.min(fem_data)])
        y_max=np.max([np.max(anal_fine_data), np.max(fem_data)])
        ax.set_xlim(x_min-0.5, x_max + 0.5)#-1, 3.5
        ax.set_ylim(y_min-0.5, y_max + 0.5)#0.0, 3.2
    
    # 1. Deflection
    plot_diagram(
        axes[0], x_fine, -anal_fine['deflection'] * 1000,# axes[0,0]
        errors.get('deflection', {}).get('fem_values', []),
        errors.get('deflection'),
        'Deflection (mm)', 'Deflection Diagram'
    )
    
    # 2. Rotation
    # plot_diagram(
    #     axes[1,0], x_fine, anal_fine['rotation'] * 1000,#axes[0,1]
    #     -errors.get('rotation', {}).get('fem_values', []),
    #     errors.get('rotation'),
    #     'Rotation (mrad)', 'Rotation Diagram',
    #     error_key='end_error_percent', error_label='End Error'
    # )
    
    #3. Bending Moment
    plot_diagram(
        axes[1], x_fine, anal_fine['moment'] / 1000,
        errors.get('moment', {}).get('fem_values', []),
        errors.get('moment'),
        'Bending Moment (kN·m)', 'Bending Moment Diagram (BMD)'
    )
    
    # 4. Shear Force
    # plot_diagram(
    #     axes[1, 1], x_fine, anal_fine['shear'] / 1000,
    #     errors.get('shear', {}).get('fem_values', []),
    #     errors.get('shear'),
    #     'Shear Force (kN)', 'Shear Force Diagram (SFD)'
    # )
    # fig.delaxes(axes[0,1])
    # fig.delaxes(axes[1,1])
    
    plt.tight_layout()
    
    output_file = f'{output_prefix}_comparison_FEM_vs_Analy.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return fig


def plot_error_distribution(errors: Dict, output_prefix: str = "beam"):
    """Plot error distribution along beam"""
    
    # Check if we have valid data
    has_data = False
    for key in ['deflection', 'rotation', 'moment', 'shear']:
        if key in errors and not errors[key].get('is_all_zero', True):
            has_data = True
            break
    
    if not has_data:
        print("Warning: All FEM values are zero. Skipping error distribution plot.")
        print("         (This typically happens with insufficient mesh - only 1 element)")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Error Distribution Along Beam', fontsize=14, fontweight='bold')
    
    # Get positions
    x = errors.get('deflection', errors.get('moment', {})).get('positions', np.array([]))
    
    if len(x) == 0:
        print("Warning: No position data available for error distribution plot.")
        return None
    
    # Calculate bar width based on beam length
    L = x[-1] - x[0] if len(x) > 1 else 1.0
    bar_width = L / (len(x) * 3) if len(x) > 1 else 0.1
    
    # 1. Deflection Error
    ax1 = axes[0, 0]
    if 'deflection' in errors and not errors['deflection'].get('is_all_zero', True):
        err = errors['deflection']['pointwise_errors']
        ax1.bar(x, err, width=bar_width, color='#E63946', alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Absolute Error (mm)')
        ax1.set_title('Deflection Error Distribution')
    else:
        ax1.text(0.5, 0.5, 'No valid FEM data\n(All values zero)', 
                transform=ax1.transAxes, ha='center', va='center',
                fontsize=12, color='gray')
        ax1.set_title('Deflection Error Distribution')
    ax1.set_xlabel('Position (m)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Moment Comparison
    ax2 = axes[0, 1]
    if 'moment' in errors:
        fem_vals = errors['moment']['fem_values']
        anal_vals = errors['moment']['analytical_values']
        is_zero = errors['moment'].get('is_all_zero', False)
        
        if not is_zero or not np.allclose(anal_vals, 0):
            width = bar_width * 0.8
            ax2.bar(x - width/2, anal_vals, width, label='Analytical', color='#1D3557', alpha=0.7)
            ax2.bar(x + width/2, fem_vals, width, label='FEM', color='#E63946', alpha=0.7)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No valid data', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=12, color='gray')
        ax2.set_ylabel('Moment (kN·m)')
        ax2.set_title('Moment Comparison at Nodes')
    ax2.set_xlabel('Position (m)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Shear Comparison
    ax3 = axes[1, 0]
    if 'shear' in errors:
        fem_vals = errors['shear']['fem_values']
        anal_vals = errors['shear']['analytical_values']
        is_zero = errors['shear'].get('is_all_zero', False)
        
        if not is_zero or not np.allclose(anal_vals, 0):
            width = bar_width * 0.8
            ax3.bar(x - width/2, anal_vals, width, label='Analytical', color='#1D3557', alpha=0.7)
            ax3.bar(x + width/2, fem_vals, width, label='FEM', color='#E63946', alpha=0.7)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No valid data', transform=ax3.transAxes, 
                    ha='center', va='center', fontsize=12, color='gray')
        ax3.set_ylabel('Shear (kN)')
        ax3.set_title('Shear Force Comparison at Nodes')
    ax3.set_xlabel('Position (m)')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Summary Error Bar Chart
    ax4 = axes[1, 1]
    error_names = []
    error_values = []
    bar_colors = []
    color_map = {
        'Deflection': '#2E86AB',
        'Rotation': '#A23B72',
        'Moment': '#F18F01',
        'Shear': '#C73E1D'
    }
    
    if 'deflection' in errors and not errors['deflection'].get('is_all_zero', True):
        error_names.append('Deflection')
        error_values.append(errors['deflection']['max_error_percent'])
        bar_colors.append(color_map['Deflection'])
    
    if 'rotation' in errors and not errors['rotation'].get('is_all_zero', True):
        error_names.append('Rotation')
        error_values.append(errors['rotation']['end_error_percent'])
        bar_colors.append(color_map['Rotation'])
    
    if 'moment' in errors and not errors['moment'].get('is_all_zero', True):
        error_names.append('Moment')
        error_values.append(errors['moment']['max_error_percent'])
        bar_colors.append(color_map['Moment'])
    
    if 'shear' in errors and not errors['shear'].get('is_all_zero', True):
        error_names.append('Shear')
        error_values.append(errors['shear']['max_error_percent'])
        bar_colors.append(color_map['Shear'])
    
    if error_names:
        bars = ax4.bar(error_names, error_values, color=bar_colors, alpha=0.8, edgecolor='black')
        ax4.set_ylabel('Error (%)')
        ax4.set_title('Maximum Errors Summary')
        ax4.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, error_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No valid FEM data\nfor error calculation', 
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=12, color='gray')
        ax4.set_title('Maximum Errors Summary')
    
    plt.tight_layout()
    
    output_file = f'{output_prefix}_errors.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    return fig


def main():
    """Main function"""
    
    # =============== CONFIGURATION ===============
    VTK_FILE = "test_files/beam_2D_test_udl_1e.gid/vtk_output/Parts_Beam_Beams_0_1.vtk"
    OUTPUT_PREFIX = "test_files/beam_2D_test_udl_1e.gid/plots/beam_results"
    
    BEAM_TYPE = 'simply_supported'
    LOAD_TYPE = 'udl'
    
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