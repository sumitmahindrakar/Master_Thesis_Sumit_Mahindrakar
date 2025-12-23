"""
Step-by-Step Finite Difference Verification
============================================

Run each step sequentially to verify adjoint sensitivity.

Usage:
    python fd_verification_stepwise.py --step 1
    python fd_verification_stepwise.py --step 2
    ... (run FE analyses manually between steps 2 and 4)
    python fd_verification_stepwise.py --step 4
    python fd_verification_stepwise.py --step 5
    python fd_verification_stepwise.py --step 6

Or run all preparation steps:
    python fd_verification_stepwise.py --step all

Author: FD Verification
"""

import os
import sys
import json
import numpy as np
import argparse
from datetime import datetime
from typing import Dict, Tuple, Any
import shutil


# =============================================================================
# CONFIGURATION - MODIFY THESE PATHS
# =============================================================================

class Config:
    """Configuration - UPDATE THESE FOR YOUR SETUP"""
    
    # Base directory containing your analysis
    BASE_DIR = "test_files/SA_frame_2"
    
    # Input files
    MATERIAL_JSON = os.path.join(BASE_DIR, "StructuralMaterials.json")
    PROJECT_PARAMS = os.path.join(BASE_DIR, "ProjectParameters.json")
    
    # Base analysis VTK outputs (already computed)
    VTK_PRIMARY = os.path.join(BASE_DIR, "vtk_output/Parts_Beam_Beams_0_1.vtk")
    VTK_DUAL = os.path.join(BASE_DIR, "vtk_output_dual/Parts_Beam_Beams_0_1.vtk")
    
    # Working directory for FD verification
    WORK_DIR = os.path.join(BASE_DIR, "fd_verification")
    
    # Perturbation percentage (start with 1%, can try smaller values)
    PERTURBATION_PERCENT = 1.0
    
    # Response element ID (None = use total/sum of all elements)
    RESPONSE_ELEMENT_ID = None  # Or set to specific element, e.g., 25
    
    # Output file for results
    RESULTS_FILE = os.path.join(BASE_DIR, "fd_verification_results.json")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_vtk_file(filename: str) -> Dict[str, Any]:
    """Parse VTK file and extract points, cells, and moment data."""
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"VTK file not found: {filename}")
    
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


def get_element_length(p1, p2) -> float:
    """Calculate element length."""
    return np.linalg.norm(np.array(p2) - np.array(p1))


def load_results(filepath: str) -> Dict:
    """Load results from JSON file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}


def save_results(filepath: str, results: Dict) -> None:
    """Save results to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {filepath}")


# =============================================================================
# STEP 1: COMPUTE ADJOINT SENSITIVITY (BASELINE)
# =============================================================================

def step1_compute_adjoint_sensitivity(config: Config) -> Dict:
    """
    STEP 1: Compute adjoint sensitivity from existing VTK files.
    
    This is your reference value that FD should match.
    """
    
    print("\n" + "=" * 70)
    print("STEP 1: COMPUTE ADJOINT SENSITIVITY")
    print("=" * 70)
    
    # Check files exist
    if not os.path.exists(config.VTK_PRIMARY):
        print(f"  ERROR: Primary VTK not found: {config.VTK_PRIMARY}")
        print("  Please run the primary analysis first.")
        return None
    
    if not os.path.exists(config.VTK_DUAL):
        print(f"  ERROR: Dual VTK not found: {config.VTK_DUAL}")
        print("  Please run the dual analysis first.")
        return None
    
    if not os.path.exists(config.MATERIAL_JSON):
        print(f"  ERROR: Material file not found: {config.MATERIAL_JSON}")
        return None
    
    # Load material properties
    print("\n  Loading material properties...")
    with open(config.MATERIAL_JSON, 'r') as f:
        mat_data = json.load(f)
    
    variables = mat_data['properties'][0]['Material']['Variables']
    E = variables['YOUNG_MODULUS']
    I = variables.get('I33') or variables.get('I22')
    EI = E * I
    EI_squared = EI ** 2
    
    print(f"    E  = {E:.6e} Pa")
    print(f"    I  = {I:.6e} m^4")
    print(f"    EI = {EI:.6e} N.m^2")
    
    # Parse VTK files
    print("\n  Parsing VTK files...")
    vtk_primary = parse_vtk_file(config.VTK_PRIMARY)
    vtk_dual = parse_vtk_file(config.VTK_DUAL)
    
    points = vtk_primary['points']
    cells = vtk_primary['cells']
    M_primary = vtk_primary['cell_data']['MOMENT']
    M_dual = vtk_dual['cell_data']['MOMENT']
    
    n_elements = len(cells)
    print(f"    Elements: {n_elements}")
    
    # Compute adjoint sensitivity
    print("\n  Computing adjoint sensitivity...")
    print(f"    Formula: dM/d(EI)_k = -(M_k × M̄_k × L_k) / (EI)^2")
    
    sensitivities = {}
    total_sensitivity = 0.0
    
    for idx, cell in enumerate(cells):
        elem_id = idx + 1
        
        p1 = points[cell[0]]
        p2 = points[cell[1]]
        L = get_element_length(p1, p2)
        
        M_p = M_primary[idx][2]  # Mz
        M_d = M_dual[idx][2]     # Mz
        
        integral = M_p * M_d * L
        dM_dEI = -integral / EI_squared
        
        sensitivities[elem_id] = {
            'M_primary': M_p,
            'M_dual': M_d,
            'length': L,
            'integral': integral,
            'dM_dEI': dM_dEI
        }
        
        total_sensitivity += dM_dEI
    
    # Print results
    print("\n  " + "-" * 65)
    print(f"  {'Elem':>5} {'M_primary':>12} {'M_dual':>12} {'L':>8} {'dM/dEI':>18}")
    print("  " + "-" * 65)
    
    for elem_id in list(sensitivities.keys())[:10]:
        s = sensitivities[elem_id]
        print(f"  {elem_id:>5} {s['M_primary']:>+12.2f} {s['M_dual']:>+12.2f} "
              f"{s['length']:>8.4f} {s['dM_dEI']:>+18.6e}")
    
    if n_elements > 10:
        print(f"  ... ({n_elements - 10} more elements)")
    
    print("  " + "-" * 65)
    print(f"  {'TOTAL':>5} {' ':>12} {' ':>12} {' ':>8} {total_sensitivity:>+18.6e}")
    print("  " + "-" * 65)
    
    # Save results
    results = {
        'step1_completed': datetime.now().isoformat(),
        'material': {
            'E': E,
            'I': I,
            'EI': EI
        },
        'adjoint': {
            'total_sensitivity': total_sensitivity,
            'n_elements': n_elements,
            'element_sensitivities': {str(k): v for k, v in sensitivities.items()}
        }
    }
    
    save_results(config.RESULTS_FILE, results)
    
    print(f"\n  ✅ STEP 1 COMPLETE")
    print(f"     Adjoint sensitivity: {total_sensitivity:+.10e}")
    
    return results


# =============================================================================
# STEP 2: CREATE PERTURBED MATERIAL FILES
# =============================================================================

def step2_create_perturbed_files(config: Config) -> Dict:
    """
    STEP 2: Create material files with perturbed EI values.
    
    Creates:
    - forward/  : EI + ΔEI (E increased)
    - backward/ : EI - ΔEI (E decreased)
    """
    
    print("\n" + "=" * 70)
    print("STEP 2: CREATE PERTURBED MATERIAL FILES")
    print("=" * 70)
    
    # Load previous results
    results = load_results(config.RESULTS_FILE)
    if 'material' not in results:
        print("  ERROR: Run Step 1 first!")
        return None
    
    E = results['material']['E']
    I = results['material']['I']
    EI = results['material']['EI']
    
    # Calculate perturbation
    delta_EI = (config.PERTURBATION_PERCENT / 100.0) * EI
    
    print(f"\n  Base EI: {EI:.6e} N.m^2")
    print(f"  Perturbation: {config.PERTURBATION_PERCENT}%")
    print(f"  ΔEI: {delta_EI:.6e} N.m^2")
    
    # Create working directory
    os.makedirs(config.WORK_DIR, exist_ok=True)
    
    # Load original material file
    with open(config.MATERIAL_JSON, 'r') as f:
        mat_data = json.load(f)
    
    # -------------------------------------------------------------------------
    # FORWARD PERTURBATION (EI + ΔEI)
    # -------------------------------------------------------------------------
    print("\n  Creating FORWARD perturbation (EI + ΔEI)...")
    
    forward_dir = os.path.join(config.WORK_DIR, "forward")
    os.makedirs(forward_dir, exist_ok=True)
    
    EI_forward = EI + delta_EI
    E_forward = EI_forward / I  # Keep I constant, change E
    
    mat_forward = json.loads(json.dumps(mat_data))  # Deep copy
    mat_forward['properties'][0]['Material']['Variables']['YOUNG_MODULUS'] = E_forward
    
    forward_material_file = os.path.join(forward_dir, "StructuralMaterials.json")
    with open(forward_material_file, 'w') as f:
        json.dump(mat_forward, f, indent=4)
    
    print(f"    E_forward = {E_forward:.6e} Pa")
    print(f"    EI_forward = {EI_forward:.6e} N.m^2")
    print(f"    Saved: {forward_material_file}")
    
    # -------------------------------------------------------------------------
    # BACKWARD PERTURBATION (EI - ΔEI)
    # -------------------------------------------------------------------------
    print("\n  Creating BACKWARD perturbation (EI - ΔEI)...")
    
    backward_dir = os.path.join(config.WORK_DIR, "backward")
    os.makedirs(backward_dir, exist_ok=True)
    
    EI_backward = EI - delta_EI
    E_backward = EI_backward / I
    
    mat_backward = json.loads(json.dumps(mat_data))
    mat_backward['properties'][0]['Material']['Variables']['YOUNG_MODULUS'] = E_backward
    
    backward_material_file = os.path.join(backward_dir, "StructuralMaterials.json")
    with open(backward_material_file, 'w') as f:
        json.dump(mat_backward, f, indent=4)
    
    print(f"    E_backward = {E_backward:.6e} Pa")
    print(f"    EI_backward = {EI_backward:.6e} N.m^2")
    print(f"    Saved: {backward_material_file}")
    
    # -------------------------------------------------------------------------
    # COPY OTHER REQUIRED FILES
    # -------------------------------------------------------------------------
    print("\n  Copying other required files...")
    
    # List of files to copy (adjust based on your setup)
    files_to_copy = [
        "ProjectParameters.json",
        "MainKratos.py",
        "StructuralAnalysis.json",
        # Add other files your analysis needs
    ]
    
    for filename in files_to_copy:
        src = os.path.join(config.BASE_DIR, filename)
        if os.path.exists(src):
            # Copy to forward
            dst_forward = os.path.join(forward_dir, filename)
            shutil.copy2(src, dst_forward)
            
            # Copy to backward
            dst_backward = os.path.join(backward_dir, filename)
            shutil.copy2(src, dst_backward)
            
            print(f"    Copied: {filename}")
    
    # Copy mesh/mdpa files if present
    for filename in os.listdir(config.BASE_DIR):
        if filename.endswith('.mdpa'):
            src = os.path.join(config.BASE_DIR, filename)
            shutil.copy2(src, os.path.join(forward_dir, filename))
            shutil.copy2(src, os.path.join(backward_dir, filename))
            print(f"    Copied: {filename}")
    
    # Update results
    results['step2_completed'] = datetime.now().isoformat()
    results['perturbation'] = {
        'percent': config.PERTURBATION_PERCENT,
        'delta_EI': delta_EI,
        'forward': {
            'E': E_forward,
            'EI': EI_forward,
            'directory': forward_dir
        },
        'backward': {
            'E': E_backward,
            'EI': EI_backward,
            'directory': backward_dir
        }
    }
    
    save_results(config.RESULTS_FILE, results)
    
    print(f"\n  ✅ STEP 2 COMPLETE")
    print(f"\n  " + "=" * 60)
    print("  NEXT: Run FE analyses manually!")
    print("  " + "=" * 60)
    print(f"""
  Run the following analyses:

  1. FORWARD analysis:
     cd {forward_dir}
     python MainKratos.py
     (or your analysis command)

  2. BACKWARD analysis:
     cd {backward_dir}
     python MainKratos.py
     (or your analysis command)

  After both analyses complete, run Step 3 to specify VTK paths,
  then Step 4 to extract moments, and Step 5-6 for comparison.
  """)
    
    return results


# =============================================================================
# STEP 3: SPECIFY VTK OUTPUT PATHS
# =============================================================================

def step3_specify_vtk_paths(config: Config, 
                            vtk_forward: str = None, 
                            vtk_backward: str = None) -> Dict:
    """
    STEP 3: Specify paths to VTK outputs from perturbed analyses.
    
    Call with paths after running FE analyses:
        step3_specify_vtk_paths(config, 
                                vtk_forward="path/to/forward/vtk",
                                vtk_backward="path/to/backward/vtk")
    """
    
    print("\n" + "=" * 70)
    print("STEP 3: SPECIFY VTK OUTPUT PATHS")
    print("=" * 70)
    
    # Load previous results
    results = load_results(config.RESULTS_FILE)
    if 'perturbation' not in results:
        print("  ERROR: Run Step 2 first!")
        return None
    
    # Default paths (guess based on typical Kratos output structure)
    forward_dir = results['perturbation']['forward']['directory']
    backward_dir = results['perturbation']['backward']['directory']
    
    if vtk_forward is None:
        # Try to find VTK file
        vtk_forward_dir = os.path.join(forward_dir, "vtk_output")
        if os.path.exists(vtk_forward_dir):
            for f in os.listdir(vtk_forward_dir):
                if f.endswith('.vtk') and 'Beam' in f:
                    vtk_forward = os.path.join(vtk_forward_dir, f)
                    break
    
    if vtk_backward is None:
        vtk_backward_dir = os.path.join(backward_dir, "vtk_output")
        if os.path.exists(vtk_backward_dir):
            for f in os.listdir(vtk_backward_dir):
                if f.endswith('.vtk') and 'Beam' in f:
                    vtk_backward = os.path.join(vtk_backward_dir, f)
                    break
    
    print(f"\n  Forward VTK:  {vtk_forward}")
    print(f"  Backward VTK: {vtk_backward}")
    
    # Verify files exist
    forward_exists = vtk_forward and os.path.exists(vtk_forward)
    backward_exists = vtk_backward and os.path.exists(vtk_backward)
    
    if not forward_exists:
        print(f"\n  ⚠️  Forward VTK not found!")
        print(f"      Expected: {vtk_forward}")
        print(f"      Run the forward FE analysis first.")
    
    if not backward_exists:
        print(f"\n  ⚠️  Backward VTK not found!")
        print(f"      Expected: {vtk_backward}")
        print(f"      Run the backward FE analysis first.")
    
    if not forward_exists or not backward_exists:
        print("\n  Please run the FE analyses and try again.")
        print("  Or specify paths manually:")
        print("""
    from fd_verification_stepwise import step3_specify_vtk_paths, Config
    config = Config()
    step3_specify_vtk_paths(
        config,
        vtk_forward="path/to/your/forward/output.vtk",
        vtk_backward="path/to/your/backward/output.vtk"
    )
        """)
        return None
    
    # Update results
    results['step3_completed'] = datetime.now().isoformat()
    results['vtk_paths'] = {
        'base': config.VTK_PRIMARY,
        'forward': vtk_forward,
        'backward': vtk_backward
    }
    
    save_results(config.RESULTS_FILE, results)
    
    print(f"\n  ✅ STEP 3 COMPLETE")
    print(f"     VTK paths saved.")
    
    return results


# =============================================================================
# STEP 4: EXTRACT MOMENTS FROM VTK FILES
# =============================================================================

def step4_extract_moments(config: Config) -> Dict:
    """
    STEP 4: Extract moments from base, forward, and backward VTK files.
    """
    
    print("\n" + "=" * 70)
    print("STEP 4: EXTRACT MOMENTS FROM VTK FILES")
    print("=" * 70)
    
    # Load previous results
    results = load_results(config.RESULTS_FILE)
    if 'vtk_paths' not in results:
        print("  ERROR: Run Step 3 first!")
        return None
    
    vtk_base = results['vtk_paths']['base']
    vtk_forward = results['vtk_paths']['forward']
    vtk_backward = results['vtk_paths']['backward']
    
    # Extract moments from each VTK
    print("\n  Extracting moments from BASE analysis...")
    data_base = parse_vtk_file(vtk_base)
    M_base = data_base['cell_data']['MOMENT']
    print(f"    Found {len(M_base)} elements")
    
    print("\n  Extracting moments from FORWARD analysis (EI + ΔEI)...")
    data_forward = parse_vtk_file(vtk_forward)
    M_forward = data_forward['cell_data']['MOMENT']
    print(f"    Found {len(M_forward)} elements")
    
    print("\n  Extracting moments from BACKWARD analysis (EI - ΔEI)...")
    data_backward = parse_vtk_file(vtk_backward)
    M_backward = data_backward['cell_data']['MOMENT']
    print(f"    Found {len(M_backward)} elements")
    
    # Store moments (Mz component)
    moments = {
        'base': {},
        'forward': {},
        'backward': {}
    }
    
    n_elements = len(M_base)
    
    print("\n  " + "-" * 70)
    print(f"  {'Elem':>5} {'M_base':>14} {'M_forward':>14} {'M_backward':>14}")
    print("  " + "-" * 70)
    
    for idx in range(n_elements):
        elem_id = idx + 1
        
        m_b = M_base[idx][2]
        m_f = M_forward[idx][2]
        m_bk = M_backward[idx][2]
        
        moments['base'][str(elem_id)] = m_b
        moments['forward'][str(elem_id)] = m_f
        moments['backward'][str(elem_id)] = m_bk
        
        if idx < 10:
            print(f"  {elem_id:>5} {m_b:>+14.4f} {m_f:>+14.4f} {m_bk:>+14.4f}")
    
    if n_elements > 10:
        print(f"  ... ({n_elements - 10} more elements)")
    
    print("  " + "-" * 70)
    
    # Calculate totals
    total_base = sum(moments['base'].values())
    total_forward = sum(moments['forward'].values())
    total_backward = sum(moments['backward'].values())
    
    print(f"  {'TOTAL':>5} {total_base:>+14.4f} {total_forward:>+14.4f} {total_backward:>+14.4f}")
    
    # Update results
    results['step4_completed'] = datetime.now().isoformat()
    results['moments'] = moments
    results['moment_totals'] = {
        'base': total_base,
        'forward': total_forward,
        'backward': total_backward
    }
    
    save_results(config.RESULTS_FILE, results)
    
    print(f"\n  ✅ STEP 4 COMPLETE")
    print(f"     Moments extracted and saved.")
    
    return results


# =============================================================================
# STEP 5: COMPUTE FINITE DIFFERENCE SENSITIVITIES
# =============================================================================

def step5_compute_fd_sensitivity(config: Config) -> Dict:
    """
    STEP 5: Compute finite difference sensitivities.
    
    Forward:  dM/dEI ≈ [M(EI+ΔEI) - M(EI)] / ΔEI
    Backward: dM/dEI ≈ [M(EI) - M(EI-ΔEI)] / ΔEI
    Central:  dM/dEI ≈ [M(EI+ΔEI) - M(EI-ΔEI)] / (2·ΔEI)
    """
    
    print("\n" + "=" * 70)
    print("STEP 5: COMPUTE FINITE DIFFERENCE SENSITIVITIES")
    print("=" * 70)
    
    # Load previous results
    results = load_results(config.RESULTS_FILE)
    if 'moments' not in results:
        print("  ERROR: Run Step 4 first!")
        return None
    
    delta_EI = results['perturbation']['delta_EI']
    moments = results['moments']
    
    print(f"\n  ΔEI = {delta_EI:.6e} N.m^2")
    print(f"  Perturbation = {results['perturbation']['percent']}%")
    
    # Compute FD for each element
    fd_sensitivities = {}
    n_elements = len(moments['base'])
    
    print("\n  Computing element-wise FD sensitivities...")
    print("\n  " + "-" * 80)
    print(f"  {'Elem':>5} {'dM/dEI_fwd':>18} {'dM/dEI_bwd':>18} {'dM/dEI_central':>18}")
    print("  " + "-" * 80)
    
    total_fd_forward = 0.0
    total_fd_backward = 0.0
    total_fd_central = 0.0
    
    for elem_id_str in moments['base'].keys():
        m_b = moments['base'][elem_id_str]
        m_f = moments['forward'][elem_id_str]
        m_bk = moments['backward'][elem_id_str]
        
        # Finite differences
        dM_dEI_forward = (m_f - m_b) / delta_EI
        dM_dEI_backward = (m_b - m_bk) / delta_EI
        dM_dEI_central = (m_f - m_bk) / (2 * delta_EI)
        
        fd_sensitivities[elem_id_str] = {
            'M_base': m_b,
            'M_forward': m_f,
            'M_backward': m_bk,
            'dM_dEI_forward': dM_dEI_forward,
            'dM_dEI_backward': dM_dEI_backward,
            'dM_dEI_central': dM_dEI_central
        }
        
        total_fd_forward += dM_dEI_forward
        total_fd_backward += dM_dEI_backward
        total_fd_central += dM_dEI_central
        
        elem_id = int(elem_id_str)
        if elem_id <= 10:
            print(f"  {elem_id:>5} {dM_dEI_forward:>+18.6e} {dM_dEI_backward:>+18.6e} "
                  f"{dM_dEI_central:>+18.6e}")
    
    if n_elements > 10:
        print(f"  ... ({n_elements - 10} more elements)")
    
    print("  " + "-" * 80)
    print(f"  {'TOTAL':>5} {total_fd_forward:>+18.6e} {total_fd_backward:>+18.6e} "
          f"{total_fd_central:>+18.6e}")
    print("  " + "-" * 80)
    
    # Update results
    results['step5_completed'] = datetime.now().isoformat()
    results['fd_sensitivities'] = {
        'elements': fd_sensitivities,
        'totals': {
            'forward': total_fd_forward,
            'backward': total_fd_backward,
            'central': total_fd_central
        }
    }
    
    save_results(config.RESULTS_FILE, results)
    
    print(f"\n  ✅ STEP 5 COMPLETE")
    print(f"     FD sensitivities computed.")
    
    return results


# =============================================================================
# STEP 6: COMPARE ADJOINT VS FINITE DIFFERENCE
# =============================================================================

def step6_compare_results(config: Config) -> Dict:
    """
    STEP 6: Compare adjoint and finite difference sensitivities.
    """
    
    print("\n" + "=" * 70)
    print("STEP 6: COMPARE ADJOINT VS FINITE DIFFERENCE")
    print("=" * 70)
    
    # Load results
    results = load_results(config.RESULTS_FILE)
    if 'fd_sensitivities' not in results:
        print("  ERROR: Run Step 5 first!")
        return None
    
    adjoint_total = results['adjoint']['total_sensitivity']
    fd_forward = results['fd_sensitivities']['totals']['forward']
    fd_backward = results['fd_sensitivities']['totals']['backward']
    fd_central = results['fd_sensitivities']['totals']['central']
    
    # Calculate errors
    def calc_error(fd_val, adj_val):
        if abs(adj_val) > 1e-20:
            return abs(fd_val - adj_val) / abs(adj_val) * 100
        else:
            return float('inf')
    
    err_forward = calc_error(fd_forward, adjoint_total)
    err_backward = calc_error(fd_backward, adjoint_total)
    err_central = calc_error(fd_central, adjoint_total)
    
    # Print comparison
    print(f"\n  Perturbation: {results['perturbation']['percent']}%")
    print(f"  ΔEI = {results['perturbation']['delta_EI']:.6e}")
    
    print("\n  " + "=" * 65)
    print(f"  {'Method':<20} {'Sensitivity':>20} {'Rel. Error':>15}")
    print("  " + "=" * 65)
    print(f"  {'Adjoint':<20} {adjoint_total:>+20.10e} {'(reference)':>15}")
    print("  " + "-" * 65)
    print(f"  {'FD Forward':<20} {fd_forward:>+20.10e} {err_forward:>14.6f}%")
    print(f"  {'FD Backward':<20} {fd_backward:>+20.10e} {err_backward:>14.6f}%")
    print(f"  {'FD Central':<20} {fd_central:>+20.10e} {err_central:>14.6f}%")
    print("  " + "=" * 65)
    
    # Assessment
    print("\n  ASSESSMENT:")
    if err_central < 0.1:
        print("  ✅ EXCELLENT: Central FD matches adjoint within 0.1%")
        assessment = "EXCELLENT"
    elif err_central < 1.0:
        print("  ✅ VERY GOOD: Central FD matches adjoint within 1%")
        assessment = "VERY_GOOD"
    elif err_central < 5.0:
        print("  ✅ GOOD: Central FD matches adjoint within 5%")
        assessment = "GOOD"
    elif err_central < 10.0:
        print("  ⚠️  MODERATE: Central FD matches adjoint within 10%")
        print("      Consider using smaller perturbation")
        assessment = "MODERATE"
    else:
        print("  ❌ POOR: Central FD differs from adjoint by more than 10%")
        print("      Check:")
        print("        - Response location consistency")
        print("        - Boundary conditions")
        print("        - Load application")
        print("        - Dual analysis setup")
        assessment = "POOR"
    
    # Element-wise comparison (for a few elements)
    print("\n  Element-wise comparison (first 5 elements):")
    print("  " + "-" * 75)
    print(f"  {'Elem':>5} {'Adjoint':>18} {'FD Central':>18} {'Error %':>12}")
    print("  " + "-" * 75)
    
    adj_elements = results['adjoint']['element_sensitivities']
    fd_elements = results['fd_sensitivities']['elements']
    
    for elem_id_str in list(adj_elements.keys())[:5]:
        adj_val = adj_elements[elem_id_str]['dM_dEI']
        fd_val = fd_elements[elem_id_str]['dM_dEI_central']
        err = calc_error(fd_val, adj_val)
        
        print(f"  {int(elem_id_str):>5} {adj_val:>+18.6e} {fd_val:>+18.6e} {err:>11.4f}%")
    
    print("  " + "-" * 75)
    
    # Update results
    results['step6_completed'] = datetime.now().isoformat()
    results['comparison'] = {
        'adjoint_total': adjoint_total,
        'fd_forward': fd_forward,
        'fd_backward': fd_backward,
        'fd_central': fd_central,
        'error_forward_pct': err_forward,
        'error_backward_pct': err_backward,
        'error_central_pct': err_central,
        'assessment': assessment
    }
    
    save_results(config.RESULTS_FILE, results)
    
    print(f"\n  ✅ STEP 6 COMPLETE")
    print(f"\n  " + "=" * 65)
    print("  VERIFICATION COMPLETE!")
    print("  " + "=" * 65)
    print(f"\n  Full results saved to: {config.RESULTS_FILE}")
    
    return results


# =============================================================================
# MAIN - COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(
        description="Step-by-step Finite Difference Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1 - Compute adjoint sensitivity (baseline)
  2 - Create perturbed material files
  3 - Specify VTK paths (after running FE analyses)
  4 - Extract moments from VTK files
  5 - Compute FD sensitivities
  6 - Compare adjoint vs FD

Example:
  python fd_verification_stepwise.py --step 1
  python fd_verification_stepwise.py --step 2
  (run FE analyses manually)
  python fd_verification_stepwise.py --step 3
  python fd_verification_stepwise.py --step 4
  python fd_verification_stepwise.py --step 5
  python fd_verification_stepwise.py --step 6
        """
    )
    
    parser.add_argument('--step', type=str, required=True,
                        help='Step number (1-6) or "all" for steps 1-2')
    parser.add_argument('--vtk-forward', type=str, default=None,
                        help='Path to forward VTK (for step 3)')
    parser.add_argument('--vtk-backward', type=str, default=None,
                        help='Path to backward VTK (for step 3)')
    
    args = parser.parse_args()
    
    config = Config()
    
    print("\n" + "=" * 70)
    print("FINITE DIFFERENCE VERIFICATION FOR ∂M/∂(EI)")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Base directory: {config.BASE_DIR}")
    print(f"Results file: {config.RESULTS_FILE}")
    
    if args.step == '1':
        step1_compute_adjoint_sensitivity(config)
    
    elif args.step == '2':
        step2_create_perturbed_files(config)
    
    elif args.step == '3':
        step3_specify_vtk_paths(config, args.vtk_forward, args.vtk_backward)
    
    elif args.step == '4':
        step4_extract_moments(config)
    
    elif args.step == '5':
        step5_compute_fd_sensitivity(config)
    
    elif args.step == '6':
        step6_compare_results(config)
    
    elif args.step == 'all':
        # Run steps 1-2 (preparation)
        step1_compute_adjoint_sensitivity(config)
        step2_create_perturbed_files(config)
        print("\n" + "=" * 70)
        print("Steps 1-2 complete. Now run FE analyses, then continue with steps 3-6.")
        print("=" * 70)
    
    elif args.step == 'post':
        # Run steps 3-6 (post-processing after FE analyses)
        step3_specify_vtk_paths(config, args.vtk_forward, args.vtk_backward)
        step4_extract_moments(config)
        step5_compute_fd_sensitivity(config)
        step6_compare_results(config)
    
    else:
        print(f"Unknown step: {args.step}")
        print("Valid options: 1, 2, 3, 4, 5, 6, all, post")


if __name__ == "__main__":
    main()