"""
Diagnostic script to compare SA results
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json

# FOLDER = os.path.dirname(os.path.abspath(__file__))
# PRIMARY_VTK = os.path.join(FOLDER, "vtk_output/Parts_Beam_Beams_0_1.vtk")
# DUAL_VTK = os.path.join(FOLDER, "vtk_output_dual/Parts_Beam_Beams_0_1.vtk")
# MATERIAL_JSON = os.path.join(FOLDER, "StructuralMaterials.json")

PRIMARY_VTK = "vtk_output/frame_2_example_vtk/primary/Parts_Beam_Beams_0_1.vtk"
DUAL_VTK =  "vtk_output/frame_2_example_vtk/dual/Parts_Beam_Beams_0_1.vtk"
MATERIAL_JSON = "input_files/frame_2_example_input/StructuralMaterials.json"

# PRIMARY_VTK = "test_files/SA_frame_2/vtk_output/Parts_Beam_Beams_0_1.vtk"
# DUAL_VTK =  "test_files/SA_frame_2/vtk_output_dual/Parts_Beam_Beams_0_1.vtk"
# MATERIAL_JSON = "test_files/SA_frame_2/StructuralMaterials.json"

# method 1 compute_sensitivity
# method 2 sensitivity_analysis_03_07

# ============ Method 1: From Code 1 ============
def parse_vtk_file_method1(filename: str) -> Dict[str, Any]:
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


def parse_moments_method1(vtk_data):
    """Code 1's moment extraction"""
    moments = {}
    if 'cell_data' in vtk_data and 'MOMENT' in vtk_data['cell_data']:
        moment_data = vtk_data['cell_data']['MOMENT']
        for idx, moment_vec in enumerate(moment_data):
            elem_id = idx + 1
            moments[elem_id] = moment_vec[2]
    return moments

# ============ Method 2: From Code 2 ============
def parse_vtk_cell_moments_method2(vtk_file_path):
    """Code 2's moment extraction (line-by-line)"""
    moments = {}
    with open(vtk_file_path, 'r') as f:
        lines = f.readlines()
    
    in_cell_data = False
    reading_moment = False
    elem_idx = 1
    
    for line in lines:
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

# ============ Compare ============
print("=" * 60)
print("COMPARING MOMENT EXTRACTION METHODS")
print("=" * 60)

# Method 1
vtk_data = parse_vtk_file_method1(PRIMARY_VTK)
M_primary_1 = parse_moments_method1(vtk_data)

# Method 2
M_primary_2 = parse_vtk_cell_moments_method2(PRIMARY_VTK)

print(f"\nMethod 1 found {len(M_primary_1)} elements")
print(f"Method 2 found {len(M_primary_2)} elements")

# Compare values
print("\nElement-by-element comparison:")
print(f"{'Elem':>6} {'Method1':>15} {'Method2':>15} {'Diff':>15}")
print("-" * 55)

for elem_id in sorted(set(M_primary_1.keys()) | set(M_primary_2.keys())):
    m1 = M_primary_1.get(elem_id, float('nan'))
    m2 = M_primary_2.get(elem_id, float('nan'))
    diff = m1 - m2 if not (np.isnan(m1) or np.isnan(m2)) else float('nan')
    print(f"{elem_id:>6} {m1:>15.6f} {m2:>15.6f} {diff:>15.6e}")

    """
Direct comparison of both SA calculation methods
"""


# FOLDER = os.path.dirname(os.path.abspath(__file__))
# PRIMARY_VTK = os.path.join(FOLDER, "vtk_output/Parts_Beam_Beams_0_1.vtk")
# DUAL_VTK = os.path.join(FOLDER, "vtk_output_dual/Parts_Beam_Beams_0_1.vtk")
# MATERIAL_JSON = os.path.join(FOLDER, "StructuralMaterials.json")


def parse_vtk_file(filename):
    """Parse VTK file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = {'points': None, 'cells': None, 'cell_data': {}}
    i = 0
    current_section = None
    
    while i < len(lines):
        line = lines[i].strip()
        
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
                cells.append([int(v) for v in values[1:]])
                i += 1
            data['cells'] = cells
            continue
        
        elif line.startswith('CELL_DATA'):
            current_section = 'cell_data'
            i += 1
            continue
        
        elif line.startswith('POINT_DATA'):
            current_section = None
            i += 1
            continue
        
        elif current_section == 'cell_data':
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


def load_material(json_path):
    """Load E and I from JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    prop = data['properties'][0]
    variables = prop['Material']['Variables']
    
    E = variables['YOUNG_MODULUS']
    I = variables.get('I33') or variables.get('I22')
    
    return E, I


def get_element_length(p1, p2):
    return np.linalg.norm(np.array(p2) - np.array(p1))


# ============================================================
# MAIN COMPARISON
# ============================================================

print("=" * 80)
print("SENSITIVITY ANALYSIS - DIRECT COMPARISON")
print("=" * 80)

# Load data
vtk_primary = parse_vtk_file(PRIMARY_VTK)
vtk_dual = parse_vtk_file(DUAL_VTK)

points = vtk_primary['points']
cells = vtk_primary['cells']

# Load material
E, I = load_material(MATERIAL_JSON)
EI = E * I
EI_squared = EI ** 2

print(f"\nMaterial Properties:")
print(f"  E  = {E:.15e}")
print(f"  I  = {I:.15e}")
print(f"  EI = {EI:.15e}")
print(f"  EI² = {EI_squared:.15e}")

# Extract moments
M_primary_arr = vtk_primary['cell_data']['MOMENT']
M_dual_arr = vtk_dual['cell_data']['MOMENT']

print(f"\nElements: {len(cells)}")
print(f"Primary moments shape: {M_primary_arr.shape}")
print(f"Dual moments shape: {M_dual_arr.shape}")

# Compute sensitivities
print("\n" + "-" * 100)
print(f"{'Elem':>5} {'M_p':>14} {'M_d':>14} {'L':>12} {'M_p*M_d*L':>18} {'dM/dEI':>20}")
print("-" * 100)

total_sensitivity = 0.0

for idx, cell in enumerate(cells):
    elem_id = idx + 1
    
    # Get points
    p1 = points[cell[0]]
    p2 = points[cell[1]]
    
    # Get length
    L = get_element_length(p1, p2)
    
    # Get moments (Mz = index 2)
    M_p = M_primary_arr[idx][2]
    M_d = M_dual_arr[idx][2]
    
    # Compute integral and sensitivity
    integral = M_p * M_d * L
    dM_dEI = -integral / EI_squared
    
    total_sensitivity += dM_dEI
    
    print(f"{elem_id:>5} {M_p:>+14.6f} {M_d:>+14.8f} {L:>12.8f} "
          f"{integral:>+18.10e} {dM_dEI:>+20.10e}")

print("-" * 100)
print(f"{'TOTAL':>5} {' ':>14} {' ':>14} {' ':>12} {' ':>18} {total_sensitivity:>+20.10e}")
print("=" * 100)

# Additional checks
print("\n=== VERIFICATION ===")
print(f"Sum of all lengths: {sum(get_element_length(points[c[0]], points[c[1]]) for c in cells):.10f}")
print(f"Number of elements: {len(cells)}")
print(f"Total sensitivity: {total_sensitivity:.15e}")