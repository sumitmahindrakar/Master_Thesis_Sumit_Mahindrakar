"""
Sensitivity Analysis Calculator with Export
============================================
Computes ∂M/∂(EI) using the adjoint method and exports results.

Features:
- Reads VTK outputs from primary and dual analyses
- Computes element-wise sensitivity
- Exports results to CSV and/or JSON
- Auto-calculates plot scales
- Generates all visualization plots

Formula:
    ∂M_response/∂(EI)_k = -(1/(EI)²) × ∫ M_k(x) × M̄_k(x) dx

Author: SA Pipeline
"""

import os
import sys
import json
import csv
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.config_loader import load_config, Config, create_directories
except ImportError:
    print("Warning: Could not import config_loader.")
    Config = None


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ElementSensitivity:
    """Sensitivity data for a single element."""
    element_id: int
    member_type: str  # 'beam' or 'column'
    length: float
    M_primary: float
    M_dual: float
    integral: float
    dM_dEI: float
    x_center: float
    y_center: float


@dataclass
class SensitivityResults:
    """Complete sensitivity analysis results."""
    # Metadata
    timestamp: str
    problem_name: str
    template: str
    response_location: Tuple[float, float, float]
    
    # Material properties
    E: float
    I: float
    EI: float
    
    # Geometry
    n_elements: int
    n_nodes_primary: int
    n_nodes_dual: int
    total_length: float
    
    # Results
    elements: List[ElementSensitivity]
    total_sensitivity: float
    
    # Statistics
    max_sensitivity: float
    min_sensitivity: float
    max_element_id: int
    min_element_id: int
    
    # Member type breakdown
    n_beams: int
    n_columns: int
    beam_sensitivity_sum: float
    column_sensitivity_sum: float


# =============================================================================
# VTK PARSER
# =============================================================================

def parse_vtk_file(filename: str) -> Dict[str, Any]:
    """Parse a VTK file and extract points, cells, and field data."""
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"VTK file not found: {filename}")
    
    print(f"  Reading: {filename}")
    
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
        
        # POINTS section
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
                i += 1
            
            data['points'] = np.array(points[:num_points])
            continue
        
        # CELLS section
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
        
        # CELL_TYPES section
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
        
        # Data sections
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
        
        # Field arrays
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
                    continue
                    
                except (ValueError, IndexError):
                    pass
        
        i += 1
    
    return data


def parse_vtk_cell_moments(vtk_data: Dict) -> Dict[int, float]:
    """Extract bending moments (Mz) from VTK cell data."""
    
    moments = {}
    
    if 'cell_data' in vtk_data and 'MOMENT' in vtk_data['cell_data']:
        moment_data = vtk_data['cell_data']['MOMENT']
        for idx, moment_vec in enumerate(moment_data):
            elem_id = idx + 1
            moments[elem_id] = moment_vec[2]  # Z component
    
    return moments


# =============================================================================
# MATERIAL PROPERTIES LOADER
# =============================================================================

def load_material_properties(json_path: str) -> Dict[str, float]:
    """Load E and I from StructuralMaterials.json."""
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Material file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    properties = data.get('properties', [])
    
    if not properties:
        raise ValueError("No properties found in material file")
    
    prop = properties[0]
    variables = prop.get('Material', {}).get('Variables', {})
    
    E = variables.get('YOUNG_MODULUS')
    I33 = variables.get('I33')
    I22 = variables.get('I22')
    
    I = I33 if I33 is not None else I22
    
    if E is None:
        raise ValueError("YOUNG_MODULUS not found in material properties")
    if I is None:
        raise ValueError("I33 or I22 not found in material properties")
    
    return {'E': E, 'I': I, 'EI': E * I}


# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================

def get_element_length(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate element length."""
    return np.linalg.norm(p2 - p1)


def get_element_center(p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float]:
    """Get element center coordinates."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def classify_element(p1: np.ndarray, p2: np.ndarray, 
                     angle_threshold: float = 45.0) -> str:
    """
    Classify element as 'beam' (horizontal) or 'column' (vertical).
    """
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    
    if dy < 1e-10:
        return 'beam'
    elif dx < 1e-10:
        return 'column'
    else:
        angle_from_vertical = np.degrees(np.arctan(dx / dy))
        return 'column' if angle_from_vertical < angle_threshold else 'beam'


def calculate_geometry(vtk_data: Dict) -> Dict[str, Any]:
    """Extract geometry information from VTK data."""
    
    points = vtk_data['points']
    cells = vtk_data['cells']
    
    element_data = {}
    total_length = 0.0
    
    for idx, cell in enumerate(cells):
        elem_id = idx + 1
        p1 = points[cell[0]]
        p2 = points[cell[1]]
        
        length = get_element_length(p1, p2)
        center = get_element_center(p1, p2)
        member_type = classify_element(p1, p2)
        
        element_data[elem_id] = {
            'length': length,
            'x_center': center[0],
            'y_center': center[1],
            'member_type': member_type,
            'node1': cell[0],
            'node2': cell[1]
        }
        
        total_length += length
    
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    return {
        'n_elements': len(cells),
        'n_nodes': len(points),
        'total_length': total_length,
        'element_data': element_data,
        'bounds': {
            'x_min': float(x_coords.min()),
            'x_max': float(x_coords.max()),
            'y_min': float(y_coords.min()),
            'y_max': float(y_coords.max())
        }
    }


# =============================================================================
# SENSITIVITY COMPUTATION
# =============================================================================

def compute_sensitivities(E: float, I: float,
                          geometry: Dict[str, Any],
                          M_primary: Dict[int, float],
                          M_dual: Dict[int, float]) -> Tuple[List[ElementSensitivity], float]:
    """
    Compute ∂M/∂(EI) for all elements.
    
    Formula: ∂M/∂(EI)_k = -(1/(EI)²) × M_k × M̄_k × L_k
    
    Returns
    -------
    Tuple of (element_sensitivities, total_sensitivity)
    """
    
    EI = E * I
    EI_squared = EI ** 2
    
    sensitivities = []
    total = 0.0
    
    element_data = geometry['element_data']
    
    # Get common elements
    common_ids = set(M_primary.keys()) & set(M_dual.keys()) & set(element_data.keys())
    
    for elem_id in sorted(common_ids):
        M_p = M_primary[elem_id]
        M_d = M_dual[elem_id]
        elem_info = element_data[elem_id]
        L = elem_info['length']
        
        # Compute integral and sensitivity
        integral = M_p * M_d * L
        dM_dEI = -integral / EI_squared
        
        sens = ElementSensitivity(
            element_id=elem_id,
            member_type=elem_info['member_type'],
            length=L,
            M_primary=M_p,
            M_dual=M_d,
            integral=integral,
            dM_dEI=dM_dEI,
            x_center=elem_info['x_center'],
            y_center=elem_info['y_center']
        )
        
        sensitivities.append(sens)
        total += dM_dEI
    
    return sensitivities, total


# =============================================================================
# RESULTS ASSEMBLY
# =============================================================================

def assemble_results(config: Config,
                     material: Dict[str, float],
                     geometry: Dict[str, Any],
                     sensitivities: List[ElementSensitivity],
                     total_sensitivity: float,
                     n_nodes_dual: int) -> SensitivityResults:
    """Assemble complete results object."""
    
    # Find extremes
    sens_values = [s.dM_dEI for s in sensitivities]
    max_sens = max(sens_values)
    min_sens = min(sens_values)
    max_elem = next(s.element_id for s in sensitivities if s.dM_dEI == max_sens)
    min_elem = next(s.element_id for s in sensitivities if s.dM_dEI == min_sens)
    
    # Member type breakdown
    beams = [s for s in sensitivities if s.member_type == 'beam']
    columns = [s for s in sensitivities if s.member_type == 'column']
    
    results = SensitivityResults(
        timestamp=datetime.now().isoformat(),
        problem_name=config.problem.name,
        template=config.problem.template,
        response_location=(config.response.x, config.response.y, 0.0),
        E=material['E'],
        I=material['I'],
        EI=material['EI'],
        n_elements=geometry['n_elements'],
        n_nodes_primary=geometry['n_nodes'],
        n_nodes_dual=n_nodes_dual,
        total_length=geometry['total_length'],
        elements=sensitivities,
        total_sensitivity=total_sensitivity,
        max_sensitivity=max_sens,
        min_sensitivity=min_sens,
        max_element_id=max_elem,
        min_element_id=min_elem,
        n_beams=len(beams),
        n_columns=len(columns),
        beam_sensitivity_sum=sum(s.dM_dEI for s in beams),
        column_sensitivity_sum=sum(s.dM_dEI for s in columns)
    )
    
    return results


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_to_csv(results: SensitivityResults, filepath: str) -> None:
    """Export sensitivity results to CSV file."""
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header section
        writer.writerow(['# Sensitivity Analysis Results'])
        writer.writerow(['# ' + '=' * 60])
        writer.writerow(['# Timestamp', results.timestamp])
        writer.writerow(['# Problem', results.problem_name])
        writer.writerow(['# Template', results.template])
        writer.writerow(['# Response Location', f"({results.response_location[0]}, {results.response_location[1]})"])
        writer.writerow(['#'])
        writer.writerow(['# Material Properties'])
        writer.writerow(['# E [Pa]', f"{results.E:.6e}"])
        writer.writerow(['# I [m^4]', f"{results.I:.6e}"])
        writer.writerow(['# EI [N.m^2]', f"{results.EI:.6e}"])
        writer.writerow(['#'])
        writer.writerow(['# Geometry'])
        writer.writerow(['# Elements', results.n_elements])
        writer.writerow(['# Nodes (primary)', results.n_nodes_primary])
        writer.writerow(['# Nodes (dual)', results.n_nodes_dual])
        writer.writerow(['# Total Length [m]', f"{results.total_length:.6f}"])
        writer.writerow(['#'])
        writer.writerow(['# Summary'])
        writer.writerow(['# Total Sensitivity', f"{results.total_sensitivity:.6e}"])
        writer.writerow(['# Max Sensitivity', f"{results.max_sensitivity:.6e}", f"Element {results.max_element_id}"])
        writer.writerow(['# Min Sensitivity', f"{results.min_sensitivity:.6e}", f"Element {results.min_element_id}"])
        writer.writerow(['# Beams', results.n_beams, f"Sum: {results.beam_sensitivity_sum:.6e}"])
        writer.writerow(['# Columns', results.n_columns, f"Sum: {results.column_sensitivity_sum:.6e}"])
        writer.writerow(['#'])
        writer.writerow(['# ' + '=' * 60])
        writer.writerow([])
        
        # Data header
        writer.writerow([
            'Element_ID',
            'Member_Type',
            'Length_m',
            'X_Center_m',
            'Y_Center_m',
            'M_Primary_Nm',
            'M_Dual_Nm',
            'Integral',
            'dM_dEI'
        ])
        
        # Data rows
        for elem in results.elements:
            writer.writerow([
                elem.element_id,
                elem.member_type,
                f"{elem.length:.6f}",
                f"{elem.x_center:.6f}",
                f"{elem.y_center:.6f}",
                f"{elem.M_primary:.6f}",
                f"{elem.M_dual:.6f}",
                f"{elem.integral:.6e}",
                f"{elem.dM_dEI:.6e}"
            ])
        
        # Footer
        writer.writerow([])
        writer.writerow(['TOTAL', '', '', '', '', '', '', '', f"{results.total_sensitivity:.6e}"])
    
    print(f"  Exported CSV: {filepath}")


def export_to_json(results: SensitivityResults, filepath: str) -> None:
    """Export sensitivity results to JSON file."""
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert to dictionary
    data = {
        'metadata': {
            'timestamp': results.timestamp,
            'problem_name': results.problem_name,
            'template': results.template,
            'response_location': list(results.response_location)
        },
        'material': {
            'E': results.E,
            'I': results.I,
            'EI': results.EI
        },
        'geometry': {
            'n_elements': results.n_elements,
            'n_nodes_primary': results.n_nodes_primary,
            'n_nodes_dual': results.n_nodes_dual,
            'total_length': results.total_length
        },
        'summary': {
            'total_sensitivity': results.total_sensitivity,
            'max_sensitivity': results.max_sensitivity,
            'min_sensitivity': results.min_sensitivity,
            'max_element_id': results.max_element_id,
            'min_element_id': results.min_element_id
        },
        'member_breakdown': {
            'n_beams': results.n_beams,
            'n_columns': results.n_columns,
            'beam_sensitivity_sum': results.beam_sensitivity_sum,
            'column_sensitivity_sum': results.column_sensitivity_sum
        },
        'elements': [asdict(elem) for elem in results.elements]
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Exported JSON: {filepath}")


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

def print_results(results: SensitivityResults) -> None:
    """Print formatted results to console."""
    
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS RESULTS: ∂M/∂(EI)")
    print("=" * 80)
    
    print(f"\nProblem: {results.problem_name} ({results.template})")
    print(f"Response Location: ({results.response_location[0]:.4f}, {results.response_location[1]:.4f})")
    
    print(f"\nMaterial Properties:")
    print(f"  E  = {results.E:.4e} Pa")
    print(f"  I  = {results.I:.4e} m⁴")
    print(f"  EI = {results.EI:.4e} N·m²")
    
    print(f"\nGeometry:")
    print(f"  Elements: {results.n_elements}")
    print(f"  Nodes (primary): {results.n_nodes_primary}")
    print(f"  Nodes (dual): {results.n_nodes_dual}")
    print(f"  Total length: {results.total_length:.4f} m")
    print(f"  Beams: {results.n_beams}, Columns: {results.n_columns}")
    
    print("\n" + "-" * 80)
    print(f"{'Elem':^6} {'Type':^8} {'M_primary':^14} {'M_dual':^14} "
          f"{'Length':^10} {'∂M/∂(EI)':^18}")
    print("-" * 80)
    
    for elem in results.elements:
        print(f"{elem.element_id:^6} {elem.member_type:^8} "
              f"{elem.M_primary:^+14.4f} {elem.M_dual:^+14.6f} "
              f"{elem.length:^10.4f} {elem.dM_dEI:^+18.6e}")
    
    print("-" * 80)
    print(f"{'TOTAL':^6} {' ':^8} {' ':^14} {' ':^14} "
          f"{' ':^10} {results.total_sensitivity:^+18.6e}")
    print("-" * 80)
    
    print(f"\nStatistics:")
    print(f"  Max sensitivity: {results.max_sensitivity:+.6e} (Element {results.max_element_id})")
    print(f"  Min sensitivity: {results.min_sensitivity:+.6e} (Element {results.min_element_id})")
    print(f"  Beam contribution: {results.beam_sensitivity_sum:+.6e}")
    print(f"  Column contribution: {results.column_sensitivity_sum:+.6e}")
    
    # Linear approximation
    delta_EI_percent = 10.0
    delta_EI = (delta_EI_percent / 100.0) * results.EI
    delta_M = results.total_sensitivity * delta_EI
    
    print(f"\nLinear Approximation:")
    print(f"  If EI increases by {delta_EI_percent}%: ΔM ≈ {delta_M:+.6f} N·m")
    
    print("=" * 80)


# =============================================================================
# AUTO SCALE CALCULATION
# =============================================================================

def calculate_auto_scales(vtk_primary: Dict, vtk_dual: Dict,
                          sensitivities: List[ElementSensitivity],
                          geometry: Dict) -> Dict[str, float]:
    """
    Calculate appropriate scale factors for plotting.
    
    Returns scales that make diagrams ~20% of structure size.
    """
    
    bounds = geometry['bounds']
    structure_size = max(
        bounds['x_max'] - bounds['x_min'],
        bounds['y_max'] - bounds['y_min']
    )
    
    target_diagram_size = structure_size * 0.2
    
    scales = {}
    
    # Deflection scale
    if 'DISPLACEMENT' in vtk_primary['point_data']:
        disp = vtk_primary['point_data']['DISPLACEMENT']
        max_disp = np.max(np.abs(disp[:, :2]))
        if max_disp > 1e-12:
            scales['deflection'] = target_diagram_size / max_disp
        else:
            scales['deflection'] = 1000.0
    else:
        scales['deflection'] = 1000.0
    
    # Primary moment scale
    if 'MOMENT' in vtk_primary['cell_data']:
        moment = vtk_primary['cell_data']['MOMENT']
        max_moment = np.max(np.abs(moment[:, 2]))
        if max_moment > 1e-6:
            scales['moment_primary'] = target_diagram_size / max_moment
        else:
            scales['moment_primary'] = 0.0001
    else:
        scales['moment_primary'] = 0.0001
    
    # Dual moment scale
    if 'MOMENT' in vtk_dual['cell_data']:
        moment = vtk_dual['cell_data']['MOMENT']
        max_moment = np.max(np.abs(moment[:, 2]))
        if max_moment > 1e-6:
            scales['moment_dual'] = target_diagram_size / max_moment
        else:
            scales['moment_dual'] = 0.000001
    else:
        scales['moment_dual'] = 0.000001
    
    # Rotation scale
    if 'ROTATION' in vtk_primary['point_data']:
        rot = vtk_primary['point_data']['ROTATION']
        max_rot = np.max(np.abs(rot[:, 2]))
        if max_rot > 1e-12:
            scales['rotation'] = target_diagram_size / max_rot
        else:
            scales['rotation'] = 1000.0
    else:
        scales['rotation'] = 1000.0
    
    # Sensitivity scale
    if sensitivities:
        max_sens = max(abs(s.dM_dEI) for s in sensitivities)
        if max_sens > 1e-20:
            scales['sensitivity'] = target_diagram_size / max_sens
        else:
            scales['sensitivity'] = 1e6
    else:
        scales['sensitivity'] = 1e6
    
    return scales


# =============================================================================
# MAIN COMPUTATION FUNCTION
# =============================================================================

def compute_sensitivity_analysis(config: Config) -> Optional[SensitivityResults]:
    """
    Main function: Compute sensitivity analysis from VTK files.
    
    Parameters
    ----------
    config : Config
        Configuration object from config.yaml
        
    Returns
    -------
    SensitivityResults or None if failed
    """
    
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS COMPUTATION")
    print("=" * 70)
    
    # Check VTK files exist
    if not os.path.exists(config.paths.vtk_primary):
        print(f"Error: Primary VTK not found: {config.paths.vtk_primary}")
        print("Run analyses first (run_analysis.py)")
        return None
    
    if not os.path.exists(config.paths.vtk_dual):
        print(f"Error: Dual VTK not found: {config.paths.vtk_dual}")
        print("Run analyses first (run_analysis.py)")
        return None
    
    # Load VTK files
    print("\n1. Loading VTK files...")
    vtk_primary = parse_vtk_file(config.paths.vtk_primary)
    vtk_dual = parse_vtk_file(config.paths.vtk_dual)
    
    # Extract moments
    print("\n2. Extracting moments...")
    M_primary = parse_vtk_cell_moments(vtk_primary)
    M_dual = parse_vtk_cell_moments(vtk_dual)
    
    print(f"  Primary: {len(M_primary)} elements")
    print(f"  Dual: {len(M_dual)} elements")
    
    if not M_primary or not M_dual:
        print("Error: Could not extract moments from VTK files")
        return None
    
    # Load material properties
    print("\n3. Loading material properties...")
    if config.material.E is not None and config.material.I is not None:
        material = {
            'E': config.material.E,
            'I': config.material.I,
            'EI': config.material.E * config.material.I
        }
        print(f"  Using config override: E={material['E']:.4e}, I={material['I']:.4e}")
    else:
        material = load_material_properties(config.paths.input_materials)
        print(f"  From JSON: E={material['E']:.4e}, I={material['I']:.4e}")
    
    # Calculate geometry
    print("\n4. Calculating geometry...")
    geometry = calculate_geometry(vtk_primary)
    print(f"  Elements: {geometry['n_elements']}")
    print(f"  Total length: {geometry['total_length']:.4f} m")
    
    # Compute sensitivities
    print("\n5. Computing sensitivities...")
    sensitivities, total = compute_sensitivities(
        material['E'], material['I'],
        geometry, M_primary, M_dual
    )
    print(f"  Computed for {len(sensitivities)} elements")
    print(f"  Total ∂M/∂(EI) = {total:.6e}")
    
    # Assemble results
    print("\n6. Assembling results...")
    n_nodes_dual = len(vtk_dual['points'])
    results = assemble_results(
        config, material, geometry,
        sensitivities, total, n_nodes_dual
    )
    
    # Calculate auto scales
    print("\n7. Calculating plot scales...")
    scales = calculate_auto_scales(vtk_primary, vtk_dual, sensitivities, geometry)
    
    # Store scales in config for plotting
    if config.plotting.deflection_scale is None:
        config.plotting.deflection_scale = scales['deflection']
    if config.plotting.moment_scale is None:
        config.plotting.moment_scale = scales['moment_primary']
    if config.plotting.sensitivity_scale is None:
        config.plotting.sensitivity_scale = scales['sensitivity']
    
    print(f"  Deflection scale: {config.plotting.deflection_scale:.2f}")
    print(f"  Moment scale: {config.plotting.moment_scale:.6f}")
    print(f"  Sensitivity scale: {config.plotting.sensitivity_scale:.2e}")
    
    # Print results
    print_results(results)
    
    # Export results
    if config.output.export_SA:
        print("\n8. Exporting results...")
        
        # Create output directory
        os.makedirs(config.paths.output_dir, exist_ok=True)
        
        if config.output.format in ['csv', 'both']:
            export_to_csv(results, config.paths.sa_results_csv)
        
        if config.output.format in ['json', 'both']:
            export_to_json(results, config.paths.sa_results_json)
    
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 70)
    
    return results


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    """Run sensitivity analysis using config.yaml."""
    
    try:
        config = load_config()
        
        if not config.analysis.compute_SA:
            print("Sensitivity analysis disabled in config. Exiting.")
            sys.exit(0)
        
        results = compute_sensitivity_analysis(config)
        
        if results is None:
            print("\nSensitivity analysis failed.")
            sys.exit(1)
        
        print("\nSensitivity analysis completed successfully!")
        print(f"  Results CSV: {config.paths.sa_results_csv}")
        print(f"  Results JSON: {config.paths.sa_results_json}")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)