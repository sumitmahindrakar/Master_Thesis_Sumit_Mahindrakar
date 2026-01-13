"""
Unified Analysis Runner for Sensitivity Analysis Pipeline
==========================================================
Runs both primary and dual Kratos analyses.

Features:
- Loads configuration from config.yaml
- Auto-loads hinge node IDs from refinement step
- Runs primary analysis (applied loads)
- Runs dual analysis (unit kink at response location)
- Couples hinge displacements via MasterSlaveConstraints

Author: SA Pipeline
"""

import os
import sys
import json
import time
import importlib
from typing import Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.config_loader import load_config, Config
except ImportError:
    print("Warning: Could not import config_loader. Running in standalone mode.")
    Config = None

# Try to import Kratos
try:
    import KratosMultiphysics
    from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
    KRATOS_AVAILABLE = True
except ImportError:
    print("Warning: KratosMultiphysics not available. Analysis cannot run.")
    KRATOS_AVAILABLE = False


# =============================================================================
# ANALYSIS CLASSES
# =============================================================================

if KRATOS_AVAILABLE:
    
    class PrimaryAnalysis(StructuralMechanicsAnalysis):
        """
        Primary analysis with applied loading.
        Standard structural analysis - no modifications needed.
        """
        
        def __init__(self, model, project_parameters, flush_frequency=10.0):
            super().__init__(model, project_parameters)
            self.flush_frequency = flush_frequency
            self.last_flush = time.time()
            sys.stdout.flush()
        
        def Initialize(self):
            super().Initialize()
            sys.stdout.flush()
            print("\n" + "=" * 60)
            print("PRIMARY ANALYSIS INITIALIZED")
            print("=" * 60)
        
        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()
            
            if self.parallel_type == "OpenMP":
                now = time.time()
                if now - self.last_flush > self.flush_frequency:
                    sys.stdout.flush()
                    self.last_flush = now
        
        def Finalize(self):
            super().Finalize()
            self._print_summary()
        
        def _print_summary(self):
            """Print analysis summary."""
            model_part = self.model["Structure"]
            
            print("\n" + "=" * 60)
            print("PRIMARY ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"  Nodes: {model_part.NumberOfNodes()}")
            print(f"  Elements: {model_part.NumberOfElements()}")
            
            # Find max displacement
            max_disp_y = 0.0
            max_disp_node = None
            
            for node in model_part.Nodes:
                disp = node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
                if abs(disp[1]) > abs(max_disp_y):
                    max_disp_y = disp[1]
                    max_disp_node = node.Id
            
            print(f"  Max Y-displacement: {max_disp_y:.6e} at node {max_disp_node}")
            print("=" * 60)
    
    
    class DualAnalysis(StructuralMechanicsAnalysis):
        """
        Dual analysis with unit kink at response location.
        
        Implements:
        - Displacement coupling at hinge (MasterSlaveConstraints)
        - Prescribed rotations θ_left = -0.5, θ_right = +0.5 (total kink = 1.0)
        """
        
        def __init__(self, model, project_parameters, 
                     hinge_left: int, hinge_right: int,
                     delta_theta: float = 1.0,
                     flush_frequency=10.0):
            super().__init__(model, project_parameters)
            self.hinge_left = hinge_left
            self.hinge_right = hinge_right
            self.delta_theta = delta_theta
            self.flush_frequency = flush_frequency
            self.last_flush = time.time()
            self._mpc_created = False
            sys.stdout.flush()
        
        def Initialize(self):
            super().Initialize()
            
            if not self._mpc_created:
                self._mpc_created = True
                self._couple_hinge_displacements()
            
            sys.stdout.flush()
        
        def _couple_hinge_displacements(self):
            """Couple displacements at hinge nodes (rotations prescribed via JSON)."""
            model_part = self.model["Structure"]
            
            # Get hinge nodes
            try:
                node_left = model_part.GetNode(self.hinge_left)
                node_right = model_part.GetNode(self.hinge_right)
            except Exception as e:
                print(f"Error: Could not find hinge nodes {self.hinge_left}, {self.hinge_right}")
                print(f"  Available nodes: {[n.Id for n in model_part.Nodes][:20]}...")
                raise e
            
            print("\n" + "=" * 60)
            print("DUAL ANALYSIS: UNIT KINK AT RESPONSE LOCATION")
            print("=" * 60)
            print(f"  Hinge left node:  {self.hinge_left} at x={node_left.X:.4f}")
            print(f"  Hinge right node: {self.hinge_right} at x={node_right.X:.4f}")
            # print(f"  Prescribed: θ_left = -0.5 rad, θ_right = +0.5 rad")
            # print(f"  Total kink = 1.0 rad")
            # print("=" * 60)

            print(f"  Prescribed: θ_left = {-self.delta_theta/2:.4f} rad, θ_right = {+self.delta_theta/2:.4f} rad")
            print(f"  Total kink = {self.delta_theta} rad")
            print("=" * 60)

            # Couple X displacement: u_x(left) = u_x(right)
            model_part.CreateNewMasterSlaveConstraint(
                "LinearMasterSlaveConstraint", 1,
                node_left, KratosMultiphysics.DISPLACEMENT_X,
                node_right, KratosMultiphysics.DISPLACEMENT_X,
                1.0, 0.0
            )
            
            # Couple Y displacement: u_y(left) = u_y(right)
            model_part.CreateNewMasterSlaveConstraint(
                "LinearMasterSlaveConstraint", 2,
                node_left, KratosMultiphysics.DISPLACEMENT_Y,
                node_right, KratosMultiphysics.DISPLACEMENT_Y,
                1.0, 0.0
            )

            # Couple rotation: r(11) = r(12)
            # RELATIVE rotation constraint: θ(12) = θ(11) + 1.0
            # This means: θ(12) - θ(11) = 1.0 (unit kink)
            model_part.CreateNewMasterSlaveConstraint(
                "LinearMasterSlaveConstraint", 3,
                node_left, KratosMultiphysics.ROTATION_Z,
                node_right, KratosMultiphysics.ROTATION_Z,
                1.0, self.delta_theta # slave = 1.0 * master + 1.0 (self.delta_theta)
            )
            
            print("  Hinge displacement coupling applied (MasterSlaveConstraints)")
            print("=" * 60 + "\n")
        
        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()
            
            if self.parallel_type == "OpenMP":
                now = time.time()
                if now - self.last_flush > self.flush_frequency:
                    sys.stdout.flush()
                    self.last_flush = now
        
        def Finalize(self):
            super().Finalize()
            self._print_results()
        
        def _print_results(self):
            """Print final results and verify kink."""
            model_part = self.model["Structure"]
            
            print("\n" + "=" * 60)
            print("DUAL ANALYSIS RESULTS")
            print("=" * 60)
            
            # Get rotations at hinge
            node_left = model_part.GetNode(self.hinge_left)
            node_right = model_part.GetNode(self.hinge_right)
            
            rot_left = node_left.GetSolutionStepValue(KratosMultiphysics.ROTATION)[2]
            rot_right = node_right.GetSolutionStepValue(KratosMultiphysics.ROTATION)[2]
            
            disp_left = node_left.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
            disp_right = node_right.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
            
            print(f"\nHinge Verification:")
            print(f"  Left node {self.hinge_left}:")
            print(f"    Displacement: ({disp_left[0]:.6e}, {disp_left[1]:.6e})")
            print(f"    Rotation: {rot_left:+.6f} rad")
            print(f"  Right node {self.hinge_right}:")
            print(f"    Displacement: ({disp_right[0]:.6e}, {disp_right[1]:.6e})")
            print(f"    Rotation: {rot_right:+.6f} rad")
            print(f"\n  Kink (Δθ): {rot_right - rot_left:+.6f} rad (should be {self.delta_theta:+.6f})") 
            
            # Check displacement coupling
            disp_diff_x = abs(disp_left[0] - disp_right[0])
            disp_diff_y = abs(disp_left[1] - disp_right[1])
            
            if disp_diff_x < 1e-10 and disp_diff_y < 1e-10:
                print("  Displacement coupling: ✓ OK")
            else:
                print(f"  Displacement coupling: ✗ FAILED (diff: {disp_diff_x:.2e}, {disp_diff_y:.2e})")
            
            print("\n" + "-" * 60)
            print("Check VTK output for MOMENT values (M_dual)")
            print("=" * 60)


# =============================================================================
# ANALYSIS RUNNERS
# =============================================================================

def load_hinge_info(config: Config) -> Tuple[int, int]:
    """
    Load hinge node IDs from refinement step.
    
    Returns
    -------
    Tuple of (hinge_left, hinge_right)
    """
    hinge_info_path = os.path.join(config.paths.input_dir, 'hinge_info.json')
    
    if not os.path.exists(hinge_info_path):
        raise FileNotFoundError(
            f"Hinge info not found: {hinge_info_path}\n"
            f"Run mesh refinement first (refine_mesh.py)"
        )
    
    with open(hinge_info_path, 'r') as f:
        hinge_info = json.load(f)
    
    return hinge_info['hinge_node_left'], hinge_info['hinge_node_right']


def run_primary_analysis(config: Config) -> bool:
    """
    Run primary structural analysis.
    
    Returns
    -------
    bool : True if successful
    """
    
    if not KRATOS_AVAILABLE:
        print("Error: KratosMultiphysics not available.")
        return False
    
    print("\n" + "=" * 70)
    print("RUNNING PRIMARY ANALYSIS")
    print("=" * 70)
    
    params_path = config.paths.input_params_primary
    
    if not os.path.exists(params_path):
        print(f"Error: ProjectParameters.json not found: {params_path}")
        print("Run mesh refinement first (refine_mesh.py)")
        return False
    
    print(f"  Parameters: {params_path}")
    
    # Load parameters
    with open(params_path, 'r') as f:
        parameters = KratosMultiphysics.Parameters(f.read())
    
    # Create output directory
    vtk_dir = os.path.dirname(config.paths.vtk_primary)
    os.makedirs(vtk_dir, exist_ok=True)
    
    # Run analysis
    try:
        model = KratosMultiphysics.Model()
        analysis = PrimaryAnalysis(model, parameters)
        analysis.Run()
        
        print(f"\n  VTK output: {vtk_dir}")
        return True
        
    except Exception as e:
        print(f"\nError during primary analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_dual_analysis(config: Config) -> bool:
    """
    Run dual structural analysis with unit kink.
    
    Returns
    -------
    bool : True if successful
    """
    
    if not KRATOS_AVAILABLE:
        print("Error: KratosMultiphysics not available.")
        return False
    
    print("\n" + "=" * 70)
    print("RUNNING DUAL ANALYSIS")
    print("=" * 70)
    
    # Load hinge info
    try:
        hinge_left, hinge_right = load_hinge_info(config)
        print(f"  Hinge nodes: left={hinge_left}, right={hinge_right}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    
    delta_theta = getattr(config.response, 'delta_theta', 1.0)
    print(f"  Delta theta: {delta_theta}")
    print(f"  DEBUG - config.response: {config.response}")  # Debug
    print(f"  DEBUG - hasattr delta_theta: {hasattr(config.response, 'delta_theta')}")  # Debug

    params_path = config.paths.input_params_dual
    
    if not os.path.exists(params_path):
        print(f"Error: ProjectParameters_dual.json not found: {params_path}")
        print("Run mesh refinement first (refine_mesh.py)")
        return False
    
    print(f"  Parameters: {params_path}")
    
    # Load parameters
    with open(params_path, 'r') as f:
        parameters = KratosMultiphysics.Parameters(f.read())
    
    # Create output directory
    vtk_dir = os.path.dirname(config.paths.vtk_dual)
    os.makedirs(vtk_dir, exist_ok=True)
    
    # Run analysis
    try:
        model = KratosMultiphysics.Model()
        analysis = DualAnalysis(model, parameters, hinge_left, hinge_right, delta_theta)
        analysis.Run()
        
        print(f"\n  VTK output: {vtk_dir}")
        return True
        
    except Exception as e:
        print(f"\nError during dual analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_analyses(config: Config) -> Tuple[bool, bool]:
    """
    Run both primary and dual analyses based on config.
    
    Returns
    -------
    Tuple of (primary_success, dual_success)
    """
    
    primary_success = False
    dual_success = False
    
    if config.analysis.run_primary:
        primary_success = run_primary_analysis(config)
    else:
        print("\nSkipping primary analysis (disabled in config)")
        primary_success = True  # Consider as success if skipped
    
    if config.analysis.run_dual:
        dual_success = run_dual_analysis(config)
    else:
        print("\nSkipping dual analysis (disabled in config)")
        dual_success = True  # Consider as success if skipped
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Primary analysis: {'✓ SUCCESS' if primary_success else '✗ FAILED'}")
    print(f"  Dual analysis:    {'✓ SUCCESS' if dual_success else '✗ FAILED'}")
    print("=" * 70)
    
    return primary_success, dual_success


# =============================================================================
# STANDALONE RUNNERS (for backward compatibility)
# =============================================================================

def run_primary_standalone(params_path: str) -> bool:
    """
    Run primary analysis from a standalone ProjectParameters.json file.
    
    Parameters
    ----------
    params_path : str
        Path to ProjectParameters.json
        
    Returns
    -------
    bool : True if successful
    """
    
    if not KRATOS_AVAILABLE:
        print("Error: KratosMultiphysics not available.")
        return False
    
    if not os.path.exists(params_path):
        print(f"Error: File not found: {params_path}")
        return False
    
    print(f"\nRunning primary analysis from: {params_path}")
    
    with open(params_path, 'r') as f:
        parameters = KratosMultiphysics.Parameters(f.read())
    
    try:
        model = KratosMultiphysics.Model()
        analysis = PrimaryAnalysis(model, parameters)
        analysis.Run()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def run_dual_standalone(params_path: str, hinge_left: int, hinge_right: int) -> bool:
    """
    Run dual analysis from a standalone ProjectParameters_dual.json file.
    
    Parameters
    ----------
    params_path : str
        Path to ProjectParameters_dual.json
    hinge_left : int
        Left hinge node ID
    hinge_right : int
        Right hinge node ID
        
    Returns
    -------
    bool : True if successful
    """
    
    if not KRATOS_AVAILABLE:
        print("Error: KratosMultiphysics not available.")
        return False
    
    if not os.path.exists(params_path):
        print(f"Error: File not found: {params_path}")
        return False
    
    print(f"\nRunning dual analysis from: {params_path}")
    print(f"  Hinge nodes: left={hinge_left}, right={hinge_right}")
    
    with open(params_path, 'r') as f:
        parameters = KratosMultiphysics.Parameters(f.read())
    
    try:
        model = KratosMultiphysics.Model()
        delta_theta = getattr(config.response, 'delta_theta', 1.0)
        analysis = DualAnalysis(model, parameters, hinge_left, hinge_right, delta_theta)
        analysis.Run()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Run analyses using config.yaml or command line arguments.
    
    Usage:
        python run_analysis.py                    # Use config.yaml
        python run_analysis.py --primary-only     # Run only primary
        python run_analysis.py --dual-only        # Run only dual
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Kratos structural analyses")
    parser.add_argument("--primary-only", action="store_true", 
                        help="Run only primary analysis")
    parser.add_argument("--dual-only", action="store_true",
                        help="Run only dual analysis")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (optional)")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override analysis settings based on command line
        if args.primary_only:
            config.analysis.run_primary = True
            config.analysis.run_dual = False
        elif args.dual_only:
            config.analysis.run_primary = False
            config.analysis.run_dual = True
        
        # Run analyses
        primary_ok, dual_ok = run_all_analyses(config)
        
        # Exit with appropriate code
        if primary_ok and dual_ok:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)