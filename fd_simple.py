"""
Finite Difference Verification using existing config.yaml
==========================================================

Simple approach: Modify E in config.yaml and re-run analysis.

Steps:
1. Run base analysis (E = E_base)
2. Run forward analysis (E = E_base * 1.01)  → EI + 1%
3. Run backward analysis (E = E_base * 0.99) → EI - 1%
4. Compare moments and compute FD sensitivity
"""

import os
import yaml
import json
import numpy as np
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_FILE = "config.yaml"
PERTURBATION_PERCENT = 1.0  # 1% perturbation

# Base material properties (from your StructuralMaterials.json)
E_BASE = 2.1e11  # Pa
I_BASE = 5.0e-6  # m^4


# =============================================================================
# STEP 1: RUN BASE ANALYSIS
# =============================================================================

def step1_run_base_analysis():
    """
    Run the base analysis with original E value.
    """
    
    print("\n" + "=" * 70)
    print("STEP 1: RUN BASE ANALYSIS")
    print("=" * 70)
    
    # Load config
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure E and I are null (use JSON values) or set to base
    config['material']['E'] = E_BASE
    config['material']['I'] = I_BASE
    
    # Save config
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"  E = {E_BASE:.6e} Pa")
    print(f"  I = {I_BASE:.6e} m^4")
    print(f"  EI = {E_BASE * I_BASE:.6e} N.m^2")
    
    print("\n  Config updated. Now run your analysis pipeline:")
    print("  $ python run_analysis.py")
    print("\n  After completion, rename/copy the VTK output:")
    print("  $ cp vtk_output/frame_2_example_vtk/primary/Parts_Beam_Beams_0_1.vtk vtk_base.vtk")
    
    return {
        'E': E_BASE,
        'I': I_BASE,
        'EI': E_BASE * I_BASE
    }


# =============================================================================
# STEP 2: RUN FORWARD ANALYSIS (EI + ΔEI)
# =============================================================================

def step2_run_forward_analysis():
    """
    Run analysis with E increased by perturbation percentage.
    """
    
    print("\n" + "=" * 70)
    print("STEP 2: RUN FORWARD ANALYSIS (EI + ΔEI)")
    print("=" * 70)
    
    # Calculate perturbed E
    EI_base = E_BASE * I_BASE
    delta_EI = (PERTURBATION_PERCENT / 100.0) * EI_base
    EI_forward = EI_base + delta_EI
    E_forward = EI_forward / I_BASE  # Keep I constant
    
    # Load and update config
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    config['material']['E'] = E_forward
    config['material']['I'] = I_BASE
    
    # Save config
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"  E_forward = {E_forward:.6e} Pa")
    print(f"  I = {I_BASE:.6e} m^4")
    print(f"  EI_forward = {EI_forward:.6e} N.m^2 (+{PERTURBATION_PERCENT}%)")
    
    print("\n  Config updated. Now run your analysis pipeline:")
    print("  $ python run_analysis.py")
    print("\n  After completion, rename/copy the VTK output:")
    print("  $ cp vtk_output/frame_2_example_vtk/primary/Parts_Beam_Beams_0_1.vtk vtk_forward.vtk")
    
    return {
        'E': E_forward,
        'I': I_BASE,
        'EI': EI_forward,
        'delta_EI': delta_EI
    }


# =============================================================================
# STEP 3: RUN BACKWARD ANALYSIS (EI - ΔEI)
# =============================================================================

def step3_run_backward_analysis():
    """
    Run analysis with E decreased by perturbation percentage.
    """
    
    print("\n" + "=" * 70)
    print("STEP 3: RUN BACKWARD ANALYSIS (EI - ΔEI)")
    print("=" * 70)
    
    # Calculate perturbed E
    EI_base = E_BASE * I_BASE
    delta_EI = (PERTURBATION_PERCENT / 100.0) * EI_base
    EI_backward = EI_base - delta_EI
    E_backward = EI_backward / I_BASE
    
    # Load and update config
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    config['material']['E'] = E_backward
    config['material']['I'] = I_BASE
    
    # Save config
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"  E_backward = {E_backward:.6e} Pa")
    print(f"  I = {I_BASE:.6e} m^4")
    print(f"  EI_backward = {EI_backward:.6e} N.m^2 (-{PERTURBATION_PERCENT}%)")
    
    print("\n  Config updated. Now run your analysis pipeline:")
    print("  $ python run_analysis.py")
    print("\n  After completion, rename/copy the VTK output:")
    print("  $ cp vtk_output/frame_2_example_vtk/primary/Parts_Beam_Beams_0_1.vtk vtk_backward.vtk")
    
    return {
        'E': E_backward,
        'I': I_BASE,
        'EI': EI_backward
    }


# =============================================================================
# STEP 4: RESET CONFIG TO BASE VALUES
# =============================================================================

def step4_reset_config():
    """
    Reset config back to base values (or null).
    """
    
    print("\n" + "=" * 70)
    print("STEP 4: RESET CONFIG TO BASE VALUES")
    print("=" * 70)
    
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    # Reset to null (use JSON values) or base values
    config['material']['E'] = None
    config['material']['I'] = None
    
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("  Config reset to use default material values from JSON.")


# =============================================================================
# STEP 5: COMPUTE FD AND COMPARE
# =============================================================================

def step5_compare_results(vtk_base: str, vtk_forward: str, vtk_backward: str,
                          vtk_dual: str):
    """
    Compute FD sensitivity and compare with adjoint.
    """
    
    print("\n" + "=" * 70)
    print("STEP 5: COMPUTE FD AND COMPARE WITH ADJOINT")
    print("=" * 70)
    
    # Parse VTK files
    def parse_vtk_moments(filename):
        """Extract Mz moments from VTK file."""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        moments = []
        in_cell_data = False
        reading_moment = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('CELL_DATA'):
                in_cell_data = True
            elif line.startswith('POINT_DATA'):
                in_cell_data = False
                reading_moment = False
            elif in_cell_data and 'MOMENT' in line and 'FIELD' not in line:
                reading_moment = True
            elif reading_moment and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        moments.append(float(parts[2]))  # Mz
                    except ValueError:
                        reading_moment = False
        
        return np.array(moments)
    
    # Load moments
    print("\n  Loading VTK files...")
    M_base = parse_vtk_moments(vtk_base)
    M_forward = parse_vtk_moments(vtk_forward)
    M_backward = parse_vtk_moments(vtk_backward)
    M_dual = parse_vtk_moments(vtk_dual)
    
    print(f"    Base: {len(M_base)} elements")
    print(f"    Forward: {len(M_forward)} elements")
    print(f"    Backward: {len(M_backward)} elements")
    print(f"    Dual: {len(M_dual)} elements")
    
    # Material properties
    EI = E_BASE * I_BASE
    delta_EI = (PERTURBATION_PERCENT / 100.0) * EI
    EI_squared = EI ** 2
    
    print(f"\n  EI = {EI:.6e}")
    print(f"  ΔEI = {delta_EI:.6e} ({PERTURBATION_PERCENT}%)")
    
    # Need element lengths for adjoint calculation
    # Parse from VTK points and cells
    def parse_vtk_geometry(filename):
        """Get points and cells from VTK."""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        points = []
        cells = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('POINTS'):
                num_points = int(line.split()[1])
                i += 1
                while len(points) < num_points and i < len(lines):
                    values = lines[i].strip().split()
                    for j in range(0, len(values), 3):
                        if j + 2 < len(values):
                            points.append([float(values[j]), float(values[j+1]), float(values[j+2])])
                    i += 1
            
            elif line.startswith('CELLS'):
                num_cells = int(line.split()[1])
                i += 1
                for _ in range(num_cells):
                    if i >= len(lines):
                        break
                    values = lines[i].strip().split()
                    cells.append([int(v) for v in values[1:]])
                    i += 1
            else:
                i += 1
        
        return np.array(points), cells
    
    points, cells = parse_vtk_geometry(vtk_base)
    
    # Compute element lengths
    lengths = []
    for cell in cells:
        p1 = points[cell[0]]
        p2 = points[cell[1]]
        L = np.linalg.norm(p2 - p1)
        lengths.append(L)
    lengths = np.array(lengths)
    
    # =========================================================================
    # ADJOINT SENSITIVITY
    # =========================================================================
    print("\n  Computing ADJOINT sensitivity...")
    
    adjoint_sens = -(M_base * M_dual * lengths) / EI_squared
    adjoint_total = np.sum(adjoint_sens)
    
    print(f"    Total adjoint: {adjoint_total:+.10e}")
    
    # =========================================================================
    # FINITE DIFFERENCE SENSITIVITY
    # =========================================================================
    print("\n  Computing FINITE DIFFERENCE sensitivity...")
    
    # For total moment response
    M_total_base = np.sum(M_base)
    M_total_forward = np.sum(M_forward)
    M_total_backward = np.sum(M_backward)
    
    fd_forward = (M_total_forward - M_total_base) / delta_EI
    fd_backward = (M_total_base - M_total_backward) / delta_EI
    fd_central = (M_total_forward - M_total_backward) / (2 * delta_EI)
    
    # Element-wise FD (for individual element moments)
    fd_elem_central = (M_forward - M_backward) / (2 * delta_EI)
    fd_elem_total = np.sum(fd_elem_central)
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n  " + "=" * 65)
    print(f"  {'Method':<25} {'Sensitivity':>20} {'Error %':>15}")
    print("  " + "=" * 65)
    print(f"  {'Adjoint (reference)':<25} {adjoint_total:>+20.10e} {'---':>15}")
    print("  " + "-" * 65)
    print(f"  {'FD Forward':<25} {fd_forward:>+20.10e} {abs(fd_forward-adjoint_total)/abs(adjoint_total)*100:>14.6f}%")
    print(f"  {'FD Backward':<25} {fd_backward:>+20.10e} {abs(fd_backward-adjoint_total)/abs(adjoint_total)*100:>14.6f}%")
    print(f"  {'FD Central':<25} {fd_central:>+20.10e} {abs(fd_central-adjoint_total)/abs(adjoint_total)*100:>14.6f}%")
    print("  " + "-" * 65)
    print(f"  {'FD Element-wise Total':<25} {fd_elem_total:>+20.10e} {abs(fd_elem_total-adjoint_total)/abs(adjoint_total)*100:>14.6f}%")
    print("  " + "=" * 65)
    
    # Assessment
    err_central = abs(fd_central - adjoint_total) / abs(adjoint_total) * 100
    
    print("\n  ASSESSMENT:")
    if err_central < 0.1:
        print("  ✅ EXCELLENT: Error < 0.1%")
    elif err_central < 1.0:
        print("  ✅ VERY GOOD: Error < 1%")
    elif err_central < 5.0:
        print("  ✅ GOOD: Error < 5%")
    elif err_central < 10.0:
        print("  ⚠️  MODERATE: Error < 10%")
    else:
        print("  ❌ POOR: Error > 10% - Check your setup!")
    
    # Element-wise comparison
    print("\n  Element-wise comparison (first 10 elements):")
    print("  " + "-" * 70)
    print(f"  {'Elem':>5} {'Adjoint':>18} {'FD Central':>18} {'Error %':>12}")
    print("  " + "-" * 70)
    
    for i in range(min(10, len(adjoint_sens))):
        adj = adjoint_sens[i]
        fd = fd_elem_central[i]
        if abs(adj) > 1e-20:
            err = abs(fd - adj) / abs(adj) * 100
        else:
            err = 0.0 if abs(fd) < 1e-20 else float('inf')
        print(f"  {i+1:>5} {adj:>+18.6e} {fd:>+18.6e} {err:>11.4f}%")
    
    print("  " + "-" * 70)
    
    return {
        'adjoint_total': adjoint_total,
        'fd_forward': fd_forward,
        'fd_backward': fd_backward,
        'fd_central': fd_central,
        'error_percent': err_central
    }


# =============================================================================
# MAIN
# =============================================================================

def print_instructions():
    """Print step-by-step instructions."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           FINITE DIFFERENCE VERIFICATION - INSTRUCTIONS              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  STEP 1: Run base analysis                                           ║
║  ─────────────────────────                                           ║
║  $ python fd_simple.py --step 1                                      ║
║  $ python run_analysis.py                                            ║
║  $ cp <vtk_output_path> vtk_base.vtk                                 ║
║                                                                      ║
║  STEP 2: Run forward analysis (EI + 1%)                              ║
║  ──────────────────────────────────────                              ║
║  $ python fd_simple.py --step 2                                      ║
║  $ python run_analysis.py                                            ║
║  $ cp <vtk_output_path> vtk_forward.vtk                              ║
║                                                                      ║
║  STEP 3: Run backward analysis (EI - 1%)                             ║
║  ───────────────────────────────────────                             ║
║  $ python fd_simple.py --step 3                                      ║
║  $ python run_analysis.py                                            ║
║  $ cp <vtk_output_path> vtk_backward.vtk                             ║
║                                                                      ║
║  STEP 4: Reset config                                                ║
║  ────────────────────                                                ║
║  $ python fd_simple.py --step 4                                      ║
║                                                                      ║
║  STEP 5: Compare results                                             ║
║  ───────────────────────                                             ║
║  $ python fd_simple.py --step 5                                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple FD Verification")
    parser.add_argument('--step', type=str, required=True,
                        help='Step number (1-5) or "help"')
    parser.add_argument('--vtk-base', type=str, default='vtk_base.vtk')
    parser.add_argument('--vtk-forward', type=str, default='vtk_forward.vtk')
    parser.add_argument('--vtk-backward', type=str, default='vtk_backward.vtk')
    parser.add_argument('--vtk-dual', type=str, 
                        default='vtk_output/frame_2_example_vtk/dual/Parts_Beam_Beams_0_1.vtk')
    
    args = parser.parse_args()
    
    if args.step == 'help':
        print_instructions()
    elif args.step == '1':
        step1_run_base_analysis()
    elif args.step == '2':
        step2_run_forward_analysis()
    elif args.step == '3':
        step3_run_backward_analysis()
    elif args.step == '4':
        step4_reset_config()
    elif args.step == '5':
        step5_compare_results(args.vtk_base, args.vtk_forward, 
                              args.vtk_backward, args.vtk_dual)
    else:
        print(f"Unknown step: {args.step}")
        print("Use --step help for instructions")