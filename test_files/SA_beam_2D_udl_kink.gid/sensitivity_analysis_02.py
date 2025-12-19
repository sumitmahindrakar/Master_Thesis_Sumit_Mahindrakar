"""
Moment Sensitivity Calculator: ∂M/∂(EI)
Using General Influence Method with VTK File Parsing
"""

import numpy as np
import os


def parse_vtk_cell_moments(vtk_file_path):
    """
    Parse bending moments (Mz) from VTK file CELL_DATA section
    
    Args:
        vtk_file_path: Path to VTK file
        
    Returns:
        dict: {element_id: Mz_value}
    """
    moments = {}
    
    if not os.path.exists(vtk_file_path):
        print(f"Error: VTK file not found: {vtk_file_path}")
        return moments
    
    with open(vtk_file_path, 'r') as f:
        lines = f.readlines()
    
    in_cell_data = False
    reading_moment = False
    elem_idx = 1
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Detect CELL_DATA section
        if line_stripped.startswith('CELL_DATA'):
            in_cell_data = True
            continue
        
        # Reset if we hit POINT_DATA (we've passed CELL_DATA)
        if line_stripped.startswith('POINT_DATA'):
            in_cell_data = False
            reading_moment = False
            continue
        
        # Look for MOMENT field in CELL_DATA
        if in_cell_data and 'MOMENT' in line_stripped and 'FIELD' not in line_stripped:
            reading_moment = True
            continue
        
        # Read moment values
        if reading_moment and line_stripped:
            parts = line_stripped.split()
            if len(parts) >= 3:
                try:
                    # Moment is [Mx, My, Mz] - we want Mz (3rd component)
                    Mz = float(parts[2])
                    moments[elem_idx] = Mz
                    elem_idx += 1
                except ValueError:
                    # Not a number line, stop reading
                    reading_moment = False
    
    return moments


def parse_vtk_node_data(vtk_file_path, variable_name):
    """
    Parse nodal data from VTK file POINT_DATA section
    
    Args:
        vtk_file_path: Path to VTK file
        variable_name: Name of variable to extract (e.g., 'DISPLACEMENT', 'ROTATION')
        
    Returns:
        dict: {node_id: [x, y, z] values}
    """
    data = {}
    
    if not os.path.exists(vtk_file_path):
        print(f"Error: VTK file not found: {vtk_file_path}")
        return data
    
    with open(vtk_file_path, 'r') as f:
        lines = f.readlines()
    
    in_point_data = False
    reading_variable = False
    node_idx = 1
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        if line_stripped.startswith('POINT_DATA'):
            in_point_data = True
            continue
        
        if line_stripped.startswith('CELL_DATA'):
            in_point_data = False
            reading_variable = False
            continue
        
        if in_point_data and variable_name in line_stripped:
            reading_variable = True
            continue
        
        if reading_variable and line_stripped:
            parts = line_stripped.split()
            if len(parts) >= 3:
                try:
                    values = [float(parts[0]), float(parts[1]), float(parts[2])]
                    data[node_idx] = values
                    node_idx += 1
                except ValueError:
                    reading_variable = False
    
    return data


def compute_moment_sensitivity(E, I, L_elements, M_primary, M_dual):
    """
    Compute ∂M/∂(EI) for all elements
    
    Formula: ∂M/∂(EI)_k = -(M_k · M̄_k · L_k) / (EI)²
    
    Args:
        E: Young's modulus [Pa]
        I: Second moment of area [m^4]
        L_elements: Dict of element lengths {elem_id: length}
        M_primary: Dict of primary moments {elem_id: moment}
        M_dual: Dict of dual moments {elem_id: moment}
        
    Returns:
        tuple: (sensitivities dict, total sensitivity)
    """
    EI = E * I
    EI_squared = EI ** 2
    
    sensitivities = {}
    total_sensitivity = 0.0
    
    for eid in sorted(M_primary.keys()):
        M_p = M_primary[eid]
        M_d = M_dual.get(eid, 0.0)
        L = L_elements.get(eid, 0.6667)
        
        # Sensitivity formula
        integral = M_p * M_d * L
        dM_dEI = -integral / EI_squared
        
        sensitivities[eid] = {
            'M_primary': M_p,
            'M_dual': M_d,
            'length': L,
            'integral': integral,
            'dM_dEI': dM_dEI
        }
        
        total_sensitivity += dM_dEI
    
    return sensitivities, total_sensitivity


def print_results(E, I, sensitivities, total_sensitivity):
    """
    Print formatted sensitivity results
    """
    EI = E * I
    
    print("=" * 70)
    print("MOMENT SENSITIVITY ANALYSIS: ∂M/∂(EI)")
    print("Using General Influence Method")
    print("=" * 70)
    
    print(f"\nMaterial/Section Properties:")
    print(f"  E  = {E:.4e} Pa")
    print(f"  I  = {I:.4e} m⁴")
    print(f"  EI = {EI:.4e} N·m²")
    
    print("\n" + "-" * 70)
    print(f"{'Elem':^8} {'M [N·m]':^14} {'M̄ [N·m]':^14} "
          f"{'L [m]':^10} {'∂M/∂(EI)':^18}")
    print("-" * 70)
    
    for eid, data in sorted(sensitivities.items()):
        print(f"{eid:^8} {data['M_primary']:^+14.4f} {data['M_dual']:^+14.6f} "
              f"{data['length']:^10.4f} {data['dM_dEI']:^+18.6e}")
    
    print("-" * 70)
    print(f"{'TOTAL':^8} {' ':^14} {' ':^14} "
          f"{' ':^10} {total_sensitivity:^+18.6e}")
    print("-" * 70)
    
    # Example calculation
    delta_EI_percent = 10.0
    delta_EI = (delta_EI_percent / 100.0) * EI
    delta_M = total_sensitivity * delta_EI
    
    print(f"\n{delta_EI_percent}% increase in EI → ΔM₂ ≈ {delta_M:+.4f} N·m")


def verify_calculation(E, I, M_primary, M_dual, L, expected_total):
    """
    Verify the sensitivity calculation step by step
    """
    EI = E * I
    
    print("\n" + "=" * 70)
    print("VERIFICATION OF CALCULATION")
    print("=" * 70)
    
    print(f"\nGiven:")
    print(f"  E = {E:.4e} Pa")
    print(f"  I = {I:.4e} m⁴")
    print(f"  EI = E × I = {EI:.4e} N·m²")
    print(f"  EI² = {EI**2:.4e} (N·m²)²")
    print(f"  L (each element) = {L:.6f} m")
    
    print(f"\nFormula: ∂M/∂(EI)_k = -(M_k × M̄_k × L_k) / (EI)²")
    
    print("\nStep-by-step calculation:")
    print("-" * 70)
    
    total_calculated = 0.0
    
    for eid in sorted(M_primary.keys()):
        M_p = M_primary[eid]
        M_d = M_dual[eid]
        
        # Step 1: M × M̄
        product = M_p * M_d
        
        # Step 2: M × M̄ × L
        integral = product * L
        
        # Step 3: -integral / EI²
        dM_dEI = -integral / (EI ** 2)
        
        print(f"\nElement {eid}:")
        print(f"  M_{eid} = {M_p:+.4f} N·m")
        print(f"  M̄_{eid} = {M_d:+.6f} N·m")
        print(f"  M × M̄ = ({M_p:+.4f}) × ({M_d:+.6f}) = {product:+.6f}")
        print(f"  M × M̄ × L = {product:+.6f} × {L:.6f} = {integral:+.6f}")
        print(f"  ∂M/∂(EI)_{eid} = -({integral:+.6f}) / ({EI**2:.4e})")
        print(f"                 = {dM_dEI:+.6e}")
        
        total_calculated += dM_dEI
    
    print("\n" + "-" * 70)
    print(f"Total ∂M/∂(EI) = Σ(∂M/∂(EI)_k) = {total_calculated:+.6e}")
    print("-" * 70)
    
    # Compare with expected
    print(f"\nComparison with your output:")
    print(f"  Your result:  {expected_total:+.6e}")
    print(f"  Calculated:   {total_calculated:+.6e}")
    
    if abs(expected_total) > 1e-15:
        rel_error = abs(total_calculated - expected_total) / abs(expected_total) * 100
        print(f"  Relative error: {rel_error:.4f}%")
        
        if rel_error < 0.1:
            print("\n✓ RESULTS MATCH! Calculation is correct.")
        else:
            print("\n✗ Results don't match. Check input values.")
    
    return total_calculated


def main():
    """
    Main function - compute sensitivity from VTK files or direct values
    """
    
    # =========================================
    # Material Properties (UPDATE THESE)
    # =========================================
    E = 2.1e11      # Young's modulus [Pa]
    I = 5.0e-6      # Second moment of area [m^4] - YOUR UPDATED VALUE
    
    # =========================================
    # Element Geometry
    # =========================================
    beam_length = 2.0  # Total beam length [m]
    n_elements = 3
    L_elem = beam_length / n_elements  # 0.6667 m
    
    L_elements = {
        1: L_elem,
        2: L_elem,
        3: L_elem
    }
    
    # =========================================
    # Option 1: Read from VTK files
    # =========================================
    use_vtk_files = True  # Set to True if you want to read from VTK
    
    if use_vtk_files:
        # Update these paths to your VTK file locations
        primary_vtk = "test_files/SA_beam_2D_udl.gid/vtk_output/Structure_Parts_Beam_Beams_0_1.vtk"
        dual_vtk = "test_files/SA_beam_2D_udl.gid/vtk_output_dual/Structure_Parts_Beam_Beams_0_1.vtk"
        
        print("Reading moments from VTK files...")
        M_primary = parse_vtk_cell_moments(primary_vtk)
        M_dual = parse_vtk_cell_moments(dual_vtk)
        
        print(f"Primary moments: {M_primary}")
        print(f"Dual moments: {M_dual}")
    
    # =========================================
    # Option 2: Use direct values (from your VTK output)
    # =========================================
    else:
        # Primary moments (from primary VTK CELL_DATA MOMENT Z-component)
        M_primary = {
            1: 740.74237,
            2: -1481.4814,
            3: 740.74237
        }
        
        # Dual moments (from dual VTK CELL_DATA MOMENT Z-component)
        M_dual = {
            1: -0.33333334,
            2: 0.66666669,
            3: -0.33333334
        }
    
    # =========================================
    # Compute Sensitivities
    # =========================================
    sensitivities, total_sensitivity = compute_moment_sensitivity(
        E, I, L_elements, M_primary, M_dual
    )
    
    # =========================================
    # Print Results
    # =========================================
    print_results(E, I, sensitivities, total_sensitivity)
    
    # =========================================
    # Verify Against Your Output
    # =========================================
    # Your terminal output showed:
    expected_total = 8.958322e-10  # From your output
    
    verify_calculation(E, I, M_primary, M_dual, L_elem, expected_total)
    
    return sensitivities, total_sensitivity


def main_with_vtk_parsing():
    """
    Main function with automatic VTK file parsing
    """
    
    # =========================================
    # Material Properties
    # =========================================
    E = 2.1e11      # Young's modulus [Pa]
    I = 5.0e-6      # Second moment of area [m^4]
    
    # =========================================
    # VTK File Paths (UPDATE THESE)
    # =========================================
    primary_vtk = "test_files/SA_beam_2D_udl.gid/vtk_output/Structure_Parts_Beam_Beams_0_1.vtk"
    dual_vtk = "test_files/SA_beam_2D_udl.gid/vtk_output_dual/Structure_Parts_Beam_Beams_0_1.vtk"
    
    # =========================================
    # Element Geometry
    # =========================================
    L_elem = 2.0 / 3.0
    L_elements = {1: L_elem, 2: L_elem, 3: L_elem}
    
    print("=" * 70)
    print("MOMENT SENSITIVITY ANALYSIS: ∂M/∂(EI)")
    print("Reading from VTK Files")
    print("=" * 70)
    
    # =========================================
    # Parse VTK Files
    # =========================================
    print(f"\nPrimary VTK: {primary_vtk}")
    M_primary = parse_vtk_cell_moments(primary_vtk)
    
    if M_primary:
        print("Primary Moments (CELL_DATA MOMENT Mz):")
        for eid, M in sorted(M_primary.items()):
            print(f"  Element {eid}: M = {M:+.4f} N·m")
    else:
        print("Warning: Could not parse primary moments. Using default values.")
        M_primary = {1: 740.74237, 2: -1481.4814, 3: 740.74237}
    
    print(f"\nDual VTK: {dual_vtk}")
    M_dual = parse_vtk_cell_moments(dual_vtk)
    
    if M_dual:
        print("Dual Moments (CELL_DATA MOMENT Mz):")
        for eid, M in sorted(M_dual.items()):
            print(f"  Element {eid}: M̄ = {M:+.6f} N·m")
    else:
        print("Warning: Could not parse dual moments. Using default values.")
        M_dual = {1: -0.33333334, 2: 0.66666669, 3: -0.33333334}
    
    # =========================================
    # Compute and Print Results
    # =========================================
    sensitivities, total_sensitivity = compute_moment_sensitivity(
        E, I, L_elements, M_primary, M_dual
    )
    
    print_results(E, I, sensitivities, total_sensitivity)
    
    return sensitivities, total_sensitivity


if __name__ == "__main__":
    # Run main analysis
    sensitivities, total = main()