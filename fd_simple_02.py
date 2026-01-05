"""
Simple Finite Difference Validation - FIXED
============================================
Handles VTK parsing correctly.
"""

import numpy as np
import os
import json


# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

VTK_BASE = "vtk_base.vtk"
VTK_FORWARD = "vtk_forward.vtk"
VTK_BACKWARD = "vtk_backward.vtk"
VTK_DUAL = "vtk_dual.vtk"

# Material properties
E_BASE = 2.1e11
I_BASE = 5.0e-6
PERTURBATION_PCT = 1.0

# Response location
RESPONSE_X = 1.0
RESPONSE_Y = 4.0


# =============================================================================
# VTK PARSER - FIXED
# =============================================================================

def parse_vtk_file(filename):
    """Parse VTK file - FIXED version."""
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    print(f"  Reading: {filename}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = {
        'points': None,
        'cells': None,
        'moments': None
    }
    
    i = 0
    current_section = None
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
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
                if len(values) > 1:
                    cells.append([int(v) for v in values[1:]])
                i += 1
            
            data['cells'] = cells
            continue
        
        # CELL_DATA section
        elif line.startswith('CELL_DATA'):
            current_section = 'cell_data'
            num_cells = int(line.split()[1])
            i += 1
            continue
        
        # POINT_DATA section
        elif line.startswith('POINT_DATA'):
            current_section = 'point_data'
            i += 1
            continue
        
        # FIELD declaration
        elif line.startswith('FIELD'):
            i += 1
            continue
        
        # Parse field arrays in CELL_DATA
        elif current_section == 'cell_data':
            parts = line.split()
            
            # Check if this is a field header (name num_components num_tuples type)
            if len(parts) >= 4 and parts[0] == 'MOMENT':
                field_name = parts[0]
                num_components = int(parts[1])
                num_tuples = int(parts[2])
                
                moments = []
                i += 1
                
                while len(moments) < num_tuples and i < len(lines):
                    values = lines[i].strip().split()
                    
                    # Stop if we hit another section
                    if not values or values[0] in ['POINT_DATA', 'CELL_DATA', 'FIELD', 'SCALARS', 'VECTORS']:
                        break
                    
                    if len(values) >= num_components:
                        try:
                            # Get Mz (index 2 for 3-component moment)
                            Mz = float(values[2]) if num_components >= 3 else float(values[0])
                            moments.append(Mz)
                            i += 1
                        except (ValueError, IndexError):
                            break
                    else:
                        i += 1
                
                data['moments'] = np.array(moments)
                continue
        
        i += 1
    
    # Verify data
    n_elements = len(data['cells']) if data['cells'] else 0
    n_nodes = len(data['points']) if data['points'] is not None else 0
    n_moments = len(data['moments']) if data['moments'] is not None else 0
    
    print(f"    Elements: {n_elements}, Nodes: {n_nodes}, Moments: {n_moments}")
    
    # Check consistency
    if n_moments != n_elements:
        print(f"    ⚠️  Warning: Moments ({n_moments}) != Elements ({n_elements})")
        # Truncate moments to match elements if needed
        if n_moments > n_elements and data['moments'] is not None:
            data['moments'] = data['moments'][:n_elements]
            print(f"    Truncated moments to {n_elements}")
    
    return data


def get_element_centers(points, cells):
    """Calculate center coordinates of each element."""
    centers = []
    for cell in cells:
        if len(cell) >= 2:
            p1 = points[cell[0]]
            p2 = points[cell[1]]
            center = (p1 + p2) / 2
            centers.append(center)
    return np.array(centers)


def get_element_lengths(points, cells):
    """Calculate length of each element."""
    lengths = []
    for cell in cells:
        if len(cell) >= 2:
            p1 = points[cell[0]]
            p2 = points[cell[1]]
            length = np.linalg.norm(p2 - p1)
            lengths.append(length)
    return np.array(lengths)


def find_element_at_location(centers, x, y):
    """Find element closest to given (x, y) location."""
    distances = np.sqrt((centers[:, 0] - x)**2 + (centers[:, 1] - y)**2)
    closest_idx = np.argmin(distances)
    return closest_idx


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def run_validation():
    """Run the FD validation."""
    
    print("\n" + "=" * 70)
    print("FINITE DIFFERENCE VALIDATION")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Load VTK files
    # -------------------------------------------------------------------------
    print("\n[1] Loading VTK files...")
    
    data_base = parse_vtk_file(VTK_BASE)
    data_forward = parse_vtk_file(VTK_FORWARD)
    data_backward = parse_vtk_file(VTK_BACKWARD)
    data_dual = parse_vtk_file(VTK_DUAL)
    
    # Get element counts
    n_base = len(data_base['moments'])
    n_forward = len(data_forward['moments'])
    n_backward = len(data_backward['moments'])
    n_dual = len(data_dual['moments'])
    
    print(f"\n  Moment arrays:")
    print(f"    Base:     {n_base}")
    print(f"    Forward:  {n_forward}")
    print(f"    Backward: {n_backward}")
    print(f"    Dual:     {n_dual}")
    
    # -------------------------------------------------------------------------
    # Material properties
    # -------------------------------------------------------------------------
    print("\n[2] Material properties...")
    
    EI = E_BASE * I_BASE
    delta_EI = (PERTURBATION_PCT / 100.0) * EI
    EI_squared = EI ** 2
    
    print(f"  E  = {E_BASE:.6e} Pa")
    print(f"  I  = {I_BASE:.6e} m^4")
    print(f"  EI = {EI:.6e} N.m^2")
    print(f"  ΔEI = {delta_EI:.6e} ({PERTURBATION_PCT}%)")
    
    # -------------------------------------------------------------------------
    # Check mesh compatibility
    # -------------------------------------------------------------------------
    print("\n[3] Checking mesh compatibility...")
    
    if n_base == n_forward == n_backward:
        print(f"  ✅ Base/Forward/Backward match: {n_base} elements")
    else:
        print(f"  ❌ Mismatch in analysis meshes!")
        return None
    
    if n_base == n_dual:
        print(f"  ✅ Dual mesh matches: {n_dual} elements")
        meshes_match = True
    else:
        print(f"  ⚠️  Dual mesh differs: {n_dual} elements")
        meshes_match = False
    
    # -------------------------------------------------------------------------
    # Compute geometry
    # -------------------------------------------------------------------------
    print("\n[4] Computing geometry...")
    
    centers = get_element_centers(data_base['points'], data_base['cells'])
    lengths = get_element_lengths(data_base['points'], data_base['cells'])
    
    print(f"  Total length: {np.sum(lengths):.4f} m")
    
    # -------------------------------------------------------------------------
    # Find response element
    # -------------------------------------------------------------------------
    print("\n[5] Finding response element...")
    print(f"  Response location: ({RESPONSE_X}, {RESPONSE_Y})")
    
    resp_idx = find_element_at_location(centers, RESPONSE_X, RESPONSE_Y)
    
    print(f"  Closest element: {resp_idx + 1}")
    print(f"  Element center: ({centers[resp_idx, 0]:.4f}, {centers[resp_idx, 1]:.4f})")
    
    # -------------------------------------------------------------------------
    # Extract moments
    # -------------------------------------------------------------------------
    print("\n[6] Extracting moments...")
    
    M_base = data_base['moments']
    M_forward = data_forward['moments']
    M_backward = data_backward['moments']
    M_dual = data_dual['moments']
    
    # If dual has different size, find corresponding element
    if meshes_match:
        M_dual_matched = M_dual
    else:
        # Find response element in dual mesh
        centers_dual = get_element_centers(data_dual['points'], data_dual['cells'])
        resp_idx_dual = find_element_at_location(centers_dual, RESPONSE_X, RESPONSE_Y)
        print(f"  Dual response element: {resp_idx_dual + 1}")
        
        # For now, use only response element comparison
        M_dual_matched = None
    
    print(f"\n  Response element {resp_idx + 1} moments:")
    print(f"    M_base     = {M_base[resp_idx]:+.4f} N.m")
    print(f"    M_forward  = {M_forward[resp_idx]:+.4f} N.m")
    print(f"    M_backward = {M_backward[resp_idx]:+.4f} N.m")
    if meshes_match:
        print(f"    M_dual     = {M_dual[resp_idx]:+.4f} N.m")
    
    # -------------------------------------------------------------------------
    # Compute sensitivities
    # -------------------------------------------------------------------------
    print("\n[7] Computing sensitivities...")
    
    # ADJOINT (only if meshes match)
    if meshes_match:
        adjoint_elem = -(M_base * M_dual * lengths) / EI_squared
        adjoint_total = np.sum(adjoint_elem)
        print(f"  Adjoint total: {adjoint_total:+.10e}")
    else:
        adjoint_total = None
        print(f"  Adjoint: Cannot compute (mesh mismatch)")
    
    # FINITE DIFFERENCE - Element-wise
    fd_elem_forward = (M_forward - M_base) / delta_EI
    fd_elem_backward = (M_base - M_backward) / delta_EI
    fd_elem_central = (M_forward - M_backward) / (2 * delta_EI)
    
    fd_total_forward = np.sum(fd_elem_forward)
    fd_total_backward = np.sum(fd_elem_backward)
    fd_total_central = np.sum(fd_elem_central)
    
    print(f"\n  FD Total Forward:  {fd_total_forward:+.10e}")
    print(f"  FD Total Backward: {fd_total_backward:+.10e}")
    print(f"  FD Total Central:  {fd_total_central:+.10e}")
    
    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n  {'Method':<25} {'Sensitivity':>20}")
    print("  " + "-" * 50)
    
    if adjoint_total is not None:
        print(f"  {'Adjoint':<25} {adjoint_total:>+20.6e}")
    
    print(f"  {'FD Forward':<25} {fd_total_forward:>+20.6e}")
    print(f"  {'FD Backward':<25} {fd_total_backward:>+20.6e}")
    print(f"  {'FD Central':<25} {fd_total_central:>+20.6e}")
    print("  " + "-" * 50)
    
    # Error analysis
    if adjoint_total is not None and abs(adjoint_total) > 1e-20:
        err_forward = abs(fd_total_forward - adjoint_total) / abs(adjoint_total) * 100
        err_backward = abs(fd_total_backward - adjoint_total) / abs(adjoint_total) * 100
        err_central = abs(fd_total_central - adjoint_total) / abs(adjoint_total) * 100
        
        print(f"\n  Error vs Adjoint:")
        print(f"    FD Forward:  {err_forward:.6f}%")
        print(f"    FD Backward: {err_backward:.6f}%")
        print(f"    FD Central:  {err_central:.6f}%")
        
        print("\n  Assessment:")
        if err_central < 0.1:
            print("  ✅ EXCELLENT: Error < 0.1%")
        elif err_central < 1.0:
            print("  ✅ VERY GOOD: Error < 1%")
        elif err_central < 5.0:
            print("  ✅ GOOD: Error < 5%")
        elif err_central < 10.0:
            print("  ⚠️  MODERATE: Error < 10%")
        else:
            print("  ❌ POOR: Error > 10%")
    
    # -------------------------------------------------------------------------
    # Element breakdown
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ELEMENT BREAKDOWN (First 15)")
    print("=" * 70)
    
    print(f"\n  {'Elem':>5} {'M_base':>12} {'M_fwd':>12} {'M_bwd':>12} {'ΔM':>12} {'FD_cent':>15}")
    print("  " + "-" * 75)
    
    for i in range(min(15, n_base)):
        delta_M = M_forward[i] - M_backward[i]
        print(f"  {i+1:>5} {M_base[i]:>+12.2f} {M_forward[i]:>+12.2f} "
              f"{M_backward[i]:>+12.2f} {delta_M:>+12.4f} {fd_elem_central[i]:>+15.6e}")
    
    print("  " + "-" * 75)
    
    # Check if moments are changing
    total_delta = np.sum(np.abs(M_forward - M_backward))
    print(f"\n  Total |ΔM| = {total_delta:.6f}")
    
    if total_delta < 1e-6:
        print("\n  ⚠️  WARNING: Moments are NOT changing between analyses!")
        print("      This means EI perturbation had no effect.")
        print("      Check that you're modifying the correct material file.")
    
    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    results = {
        'material': {'E': E_BASE, 'I': I_BASE, 'EI': EI, 'delta_EI': delta_EI},
        'adjoint_total': float(adjoint_total) if adjoint_total else None,
        'fd_forward': float(fd_total_forward),
        'fd_backward': float(fd_total_backward),
        'fd_central': float(fd_total_central),
        'total_delta_M': float(total_delta)
    }
    
    with open("fd_validation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: fd_validation_results.json")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return results


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    run_validation()

# base EI
# material:
#   E: 210000000000.0
#   I: 0.000005

# python scripts/FEM_SA/run_analysis.py
# cp vtk_output/frame_2_example_vtk/primary/Parts_Beam_Beams_0_1.vtk vtk_base.vtk
# cp vtk_output/frame_2_example_vtk/dual/Parts_Beam_Beams_0_1.vtk vtk_dual.vtk

# cp vtk_output/beam_example_E2100_vtk/primary/Parts_Beam_Beams_0_1.vtk vtk_base.vtk
# cp vtk_output/beam_example_E2100_vtk/dual/Parts_Beam_Beams_0_1.vtk vtk_dual.vtk

# edit confiig E by +1.0 %
# then copy the forward vtk
# material:
#   E: 212100000000.0
#   I: 0.000005
# python scripts/FEM_SA/run_analysis.py
# cp vtk_output/frame_2_example_E2121_vtk/primary/Parts_Beam_Beams_0_1.vtk vtk_forward.vtk

# cp vtk_output/beam_example_E2121_vtk/primary/Parts_Beam_Beams_0_1.vtk vtk_forward.vtk


# edit confiig E by -1.0 %
# material:
#   E: 207900000000.0
#   I: 0.000005
# python scripts/FEM_SA/run_analysis.py
# cp vtk_output/frame_2_example_E2079_vtk/primary/Parts_Beam_Beams_0_1.vtk vtk_backward.vtk

#cp vtk_output/beam_example_E2079_vtk/primary/Parts_Beam_Beams_0_1.vtk vtk_backward.vtk

# run this scripts
