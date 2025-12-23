"""
Automated FD Verification Runner
================================
Updates material, runs analysis, copies VTK for each case.
"""

import os
import json
import shutil
import subprocess
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

MATERIAL_FILE = "input_files/frame_2_example_input/StructuralMaterials.json"
VTK_SOURCE = "vtk_output/frame_2_example_vtk/primary/Parts_Beam_Beams_0_1.vtk"

E_BASE = 210000000000.0
PERTURBATION_PCT = 1.0

# Calculate perturbed values
E_FORWARD = E_BASE * (1 + PERTURBATION_PCT / 100)   # +1%
E_BACKWARD = E_BASE * (1 - PERTURBATION_PCT / 100)  # -1%


# =============================================================================
# FUNCTIONS
# =============================================================================

def update_E(new_E):
    """Update Young's modulus in material file."""
    
    print(f"\n  Updating E to {new_E:.6e}...")
    
    with open(MATERIAL_FILE, 'r') as f:
        data = json.load(f)
    
    old_E = data['properties'][0]['Material']['Variables']['YOUNG_MODULUS']
    data['properties'][0]['Material']['Variables']['YOUNG_MODULUS'] = new_E
    
    with open(MATERIAL_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"  E: {old_E:.6e} → {new_E:.6e}")
    return True


def run_analysis():
    """Run the main analysis pipeline."""
    
    print("\n  Running analysis...")
    
    result = subprocess.run(
        [sys.executable, "main.py"],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print("  ❌ Analysis failed!")
        return False
    
    print("  ✅ Analysis complete")
    return True


def copy_vtk(destination):
    """Copy VTK output to destination."""
    
    if not os.path.exists(VTK_SOURCE):
        print(f"  ❌ VTK source not found: {VTK_SOURCE}")
        return False
    
    shutil.copy2(VTK_SOURCE, destination)
    print(f"  Copied: {VTK_SOURCE} → {destination}")
    return True


def verify_E():
    """Verify current E value in material file."""
    
    with open(MATERIAL_FILE, 'r') as f:
        data = json.load(f)
    
    current_E = data['properties'][0]['Material']['Variables']['YOUNG_MODULUS']
    print(f"  Current E in file: {current_E:.6e}")
    return current_E


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("FINITE DIFFERENCE VERIFICATION - AUTOMATED RUNNER")
    print("=" * 60)
    
    print(f"\nMaterial file: {MATERIAL_FILE}")
    print(f"VTK source: {VTK_SOURCE}")
    print(f"\nE values:")
    print(f"  Base:     {E_BASE:.6e}")
    print(f"  Forward:  {E_FORWARD:.6e} (+{PERTURBATION_PCT}%)")
    print(f"  Backward: {E_BACKWARD:.6e} (-{PERTURBATION_PCT}%)")
    
    input("\nPress Enter to start (Ctrl+C to cancel)...")
    
    # =========================================================================
    # STEP 1: BASE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: BASE ANALYSIS")
    print("=" * 60)
    
    update_E(E_BASE)
    verify_E()
    
    if not run_analysis():
        return
    
    if not copy_vtk("vtk_base.vtk"):
        return
    
    # =========================================================================
    # STEP 2: FORWARD ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: FORWARD ANALYSIS (+1%)")
    print("=" * 60)
    
    update_E(E_FORWARD)
    verify_E()
    
    if not run_analysis():
        return
    
    if not copy_vtk("vtk_forward.vtk"):
        return
    
    # =========================================================================
    # STEP 3: BACKWARD ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: BACKWARD ANALYSIS (-1%)")
    print("=" * 60)
    
    update_E(E_BACKWARD)
    verify_E()
    
    if not run_analysis():
        return
    
    if not copy_vtk("vtk_backward.vtk"):
        return
    
    # =========================================================================
    # STEP 4: RESET AND VERIFY
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: RESET E TO BASE VALUE")
    print("=" * 60)
    
    update_E(E_BASE)
    verify_E()
    
    # =========================================================================
    # STEP 5: CHECK VTK FILES
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: VERIFY VTK FILES")
    print("=" * 60)
    
    for vtk_file in ["vtk_base.vtk", "vtk_forward.vtk", "vtk_backward.vtk"]:
        if os.path.exists(vtk_file):
            size = os.path.getsize(vtk_file)
            mtime = os.path.getmtime(vtk_file)
            print(f"  {vtk_file}: {size} bytes")
        else:
            print(f"  {vtk_file}: NOT FOUND!")
    
    # =========================================================================
    # DONE
    # =========================================================================
    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE!")
    print("=" * 60)
    print("\nNow run validation:")
    print("  python fd_validation.py")


if __name__ == "__main__":
    main()