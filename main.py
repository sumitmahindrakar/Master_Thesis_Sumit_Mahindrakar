"""
Sensitivity Analysis Pipeline - Main Runner
============================================
One-click execution of the complete sensitivity analysis workflow.

Workflow:
1. Load configuration from config.yaml
2. Refine mesh and generate primary + dual MDPA files
3. Run primary structural analysis (applied loads)
4. Run dual structural analysis (unit kink)
5. Compute sensitivity values
6. Generate all plots
7. Export results to CSV/JSON

Usage:
    python main.py                    # Run complete pipeline
    python main.py --refine-only      # Only refine mesh
    python main.py --analysis-only    # Only run analyses (skip refinement)
    python main.py --post-only        # Only compute SA and plots
    python main.py --help             # Show help

Author: SA Pipeline
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Optional
import json

# Add scripts directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'scripts'))

# Import pipeline components
try:
    from utils.config_loader import (
        load_config, Config, create_directories, print_config_summary
    )
except ImportError as e:
    print(f"Error importing config_loader: {e}")
    print("Make sure the scripts/utils/ directory exists with config_loader.py")
    sys.exit(1)

try:
    from FEM_SA.refine_mesh import refine_and_prepare, RefinementResult
except ImportError as e:
    print(f"Warning: Could not import refine_mesh: {e}")
    refine_and_prepare = None

try:
    from FEM_SA.run_analysis import run_all_analyses, run_primary_analysis, run_dual_analysis
except ImportError as e:
    print(f"Warning: Could not import run_analysis: {e}")
    run_all_analyses = None

try:
    from FEM_SA.compute_sensitivity import compute_sensitivity_analysis, SensitivityResults
except ImportError as e:
    print(f"Warning: Could not import compute_sensitivity: {e}")
    compute_sensitivity_analysis = None

try:
    from FEM_SA.plot_results import generate_plots
except ImportError as e:
    print(f"Warning: Could not import plot_results: {e}")
    generate_plots = None


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def step_refine(config: Config) -> Optional[RefinementResult]:
    """
    Step 1: Refine mesh and prepare input files.
    """
    print("\n")
    print("█" * 70)
    print("█  STEP 1: MESH REFINEMENT AND INPUT PREPARATION")
    print("█" * 70)
    
    if refine_and_prepare is None:
        print("Error: refine_mesh module not available.")
        return None
    
    try:
        result = refine_and_prepare(config)
        return result
    except Exception as e:
        print(f"Error during mesh refinement: {e}")
        import traceback
        traceback.print_exc()
        return None

def update_materials_from_config(config):
    """Update material file with config values."""
    
    print("\n" + "=" * 50)
    print("DEBUG: update_materials_from_config called")
    print(f"DEBUG: config.material.E = {config.material.E}")
    print(f"DEBUG: config.material.I = {config.material.I}")
    print("=" * 50)
    
    if config.material.E is None and config.material.I is None:
        print("DEBUG: Both E and I are None, returning early")
        return True
    
    material_path = config.paths.input_materials
    print(f"DEBUG: material_path = {material_path}")
    print(f"DEBUG: File exists? {os.path.exists(material_path)}")
    
    with open(material_path, 'r') as f:
        data = json.load(f)
    
    variables = data['properties'][0]['Material']['Variables']
    
    if config.material.E is not None:
        old_E = variables['YOUNG_MODULUS']
        variables['YOUNG_MODULUS'] = config.material.E
        print(f"DEBUG: Updating E: {old_E:.6e} → {config.material.E:.6e}")
    
    if config.material.I is not None:
        for key in ['I33', 'I22']:
            if key in variables:
                variables[key] = config.material.I
        print(f"DEBUG: Updating I: → {config.material.I:.6e}")
    
    with open(material_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"DEBUG: File saved to {material_path}")
    print("=" * 50 + "\n")
    
    return True
def step_analysis(config: Config) -> bool:
    """
    Step 2: Run primary and dual analyses.
    """
    print("\n")
    print("█" * 70)
    print("█  STEP 2: STRUCTURAL ANALYSES")
    print("█" * 70)
    
    # ===== ADD THIS CALL HERE =====
    update_materials_from_config(config)
    # ==============================
    
    if run_all_analyses is None:
        print("Error: run_analysis module not available.")
        return False
    
    try:
        primary_ok, dual_ok = run_all_analyses(config)
        return primary_ok and dual_ok
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False
def step_sensitivity(config: Config) -> Optional[SensitivityResults]:
    """
    Step 3: Compute sensitivity analysis.
    """
    print("\n")
    print("█" * 70)
    print("█  STEP 3: SENSITIVITY ANALYSIS COMPUTATION")
    print("█" * 70)
    
    if compute_sensitivity_analysis is None:
        print("Error: compute_sensitivity module not available.")
        return None
    
    if not config.analysis.compute_SA:
        print("Sensitivity analysis disabled in config. Skipping.")
        return None
    
    try:
        results = compute_sensitivity_analysis(config)
        return results
    except Exception as e:
        print(f"Error during sensitivity computation: {e}")
        import traceback
        traceback.print_exc()
        return None


def step_plotting(config: Config) -> bool:
    """
    Step 4: Generate plots.
    """
    print("\n")
    print("█" * 70)
    print("█  STEP 4: GENERATING PLOTS")
    print("█" * 70)
    
    if generate_plots is None:
        print("Error: plot_results module not available.")
        return False
    
    if not config.plotting.enabled:
        print("Plotting disabled in config. Skipping.")
        return True
    
    try:
        generate_plots(config)
        return True
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# PIPELINE RUNNER
# =============================================================================

def run_pipeline(config: Config,
                 do_refine: bool = True,
                 do_analysis: bool = True,
                 do_sensitivity: bool = True,
                 do_plotting: bool = True) -> dict:
    """
    Run the complete sensitivity analysis pipeline.
    
    Parameters
    ----------
    config : Config
        Configuration object
    do_refine : bool
        Whether to run mesh refinement step
    do_analysis : bool
        Whether to run structural analyses
    do_sensitivity : bool
        Whether to compute sensitivity
    do_plotting : bool
        Whether to generate plots
        
    Returns
    -------
    dict : Results summary with timing and status
    """
    
    start_time = time.time()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'problem_name': config.problem.name,
        'template': config.problem.template,
        'steps': {},
        'success': False,
        'total_time': 0.0
    }
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " SENSITIVITY ANALYSIS PIPELINE ".center(68) + "║")
    print("║" + f" Problem: {config.problem.name} ({config.problem.template}) ".center(68) + "║")
    print("║" + f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Print configuration summary
    print_config_summary(config)
    
    # Step 1: Mesh Refinement
    if do_refine:
        step_start = time.time()
        refinement_result = step_refine(config)
        step_time = time.time() - step_start
        
        results['steps']['refinement'] = {
            'success': refinement_result is not None,
            'time': step_time
        }
        
        if refinement_result is None:
            print("\n✗ Mesh refinement failed. Stopping pipeline.")
            results['total_time'] = time.time() - start_time
            return results
        
        results['steps']['refinement']['n_elements'] = refinement_result.n_elements
        results['steps']['refinement']['hinge_nodes'] = (
            refinement_result.hinge_node_left,
            refinement_result.hinge_node_right
        )
    else:
        print("\n⊘ Skipping mesh refinement (--analysis-only or --post-only)")
        results['steps']['refinement'] = {'success': True, 'skipped': True}
    
    # Step 2: Structural Analyses
    if do_analysis:
        step_start = time.time()
        analysis_ok = step_analysis(config)
        step_time = time.time() - step_start
        
        results['steps']['analysis'] = {
            'success': analysis_ok,
            'time': step_time
        }
        
        if not analysis_ok:
            print("\n✗ Structural analysis failed. Stopping pipeline.")
            results['total_time'] = time.time() - start_time
            return results
    else:
        print("\n⊘ Skipping structural analysis (--refine-only or --post-only)")
        results['steps']['analysis'] = {'success': True, 'skipped': True}
    
    # Step 3: Sensitivity Computation
    if do_sensitivity and config.analysis.compute_SA:
        step_start = time.time()
        sa_results = step_sensitivity(config)
        step_time = time.time() - step_start
        
        results['steps']['sensitivity'] = {
            'success': sa_results is not None,
            'time': step_time
        }
        
        if sa_results is not None:
            results['steps']['sensitivity']['total_sensitivity'] = sa_results.total_sensitivity
            results['steps']['sensitivity']['n_elements'] = sa_results.n_elements
    else:
        print("\n⊘ Skipping sensitivity computation")
        results['steps']['sensitivity'] = {'success': True, 'skipped': True}
    
    # Step 4: Plotting
    if do_plotting and config.plotting.enabled:
        step_start = time.time()
        plot_ok = step_plotting(config)
        step_time = time.time() - step_start
        
        results['steps']['plotting'] = {
            'success': plot_ok,
            'time': step_time
        }
    else:
        print("\n⊘ Skipping plotting")
        results['steps']['plotting'] = {'success': True, 'skipped': True}
    
    # Final summary
    total_time = time.time() - start_time
    results['total_time'] = total_time
    results['success'] = all(
        step.get('success', False) 
        for step in results['steps'].values()
    )
    
    print_final_summary(config, results)
    
    return results


def print_final_summary(config: Config, results: dict) -> None:
    """Print final pipeline summary."""
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " PIPELINE COMPLETE ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    print(f"\n  Problem: {config.problem.name}")
    print(f"  Template: {config.problem.template}")
    print(f"  Response: ({config.response.x}, {config.response.y})")
    
    print("\n  Step Summary:")
    print("  " + "-" * 50)
    
    step_names = {
        'refinement': 'Mesh Refinement',
        'analysis': 'Structural Analysis',
        'sensitivity': 'Sensitivity Computation',
        'plotting': 'Plot Generation'
    }
    
    for step_key, step_name in step_names.items():
        step = results['steps'].get(step_key, {})
        
        if step.get('skipped', False):
            status = "⊘ SKIPPED"
            time_str = ""
        elif step.get('success', False):
            status = "✓ SUCCESS"
            time_str = f" ({step.get('time', 0):.2f}s)"
        else:
            status = "✗ FAILED"
            time_str = f" ({step.get('time', 0):.2f}s)"
        
        print(f"    {step_name:<25} {status}{time_str}")
    
    print("  " + "-" * 50)
    print(f"    {'Total Time':<25} {results['total_time']:.2f} seconds")
    
    if results['success']:
        print("\n  ╔" + "═" * 46 + "╗")
        print("  ║" + " ✓ ALL STEPS COMPLETED SUCCESSFULLY ".center(46) + "║")
        print("  ╚" + "═" * 46 + "╝")
    else:
        print("\n  ╔" + "═" * 46 + "╗")
        print("  ║" + " ✗ PIPELINE FAILED ".center(46) + "║")
        print("  ╚" + "═" * 46 + "╝")
    
    # Output locations
    print("\n  Output Files:")
    print(f"    Input files:  {config.paths.input_dir}")
    print(f"    VTK output:   {config.paths.vtk_dir}")
    print(f"    Results:      {config.paths.output_dir}")
    print(f"    Plots:        {config.paths.plots_dir}")
    
    if 'sensitivity' in results['steps']:
        sens_step = results['steps']['sensitivity']
        if not sens_step.get('skipped', False) and sens_step.get('success', False):
            total_sa = sens_step.get('total_sensitivity')
            if total_sa is not None:
                print(f"\n  Total Sensitivity ∂M/∂(EI): {total_sa:.6e}")
    
    print("\n" + "=" * 70)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Sensitivity Analysis Pipeline - Complete workflow automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run complete pipeline
  python main.py --refine-only      Only generate mesh files
  python main.py --analysis-only    Only run Kratos analyses
  python main.py --post-only        Only compute SA and generate plots
  python main.py --config other.yaml Use different config file
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--refine-only",
        action="store_true",
        help="Only run mesh refinement step"
    )
    
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Only run structural analyses (assumes mesh already refined)"
    )
    
    parser.add_argument(
        "--post-only",
        action="store_true",
        help="Only run post-processing (SA computation and plots)"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    
    parser.add_argument(
        "--no-sa",
        action="store_true",
        help="Skip sensitivity analysis computation"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    
    args = parse_arguments()
    
    # Determine which steps to run
    if args.refine_only:
        do_refine = True
        do_analysis = False
        do_sensitivity = False
        do_plotting = False
    elif args.analysis_only:
        do_refine = False
        do_analysis = True
        do_sensitivity = False
        do_plotting = False
    elif args.post_only:
        do_refine = False
        do_analysis = False
        do_sensitivity = True
        do_plotting = True
    else:
        # Full pipeline
        do_refine = True
        do_analysis = True
        do_sensitivity = True
        do_plotting = True
    
    # Apply additional flags
    if args.no_plots:
        do_plotting = False
    if args.no_sa:
        do_sensitivity = False
    
    try:
        # Load configuration
        config = load_config(args.config)

        # ===== ADD THIS DEBUG =====
        print("\n" + "=" * 60)
        print("DEBUG: Configuration loaded")
        print(f"  config.material.E = {config.material.E}")
        print(f"  config.material.I = {config.material.I}")
        print(f"  config.paths.input_materials = {config.paths.input_materials}")
        print("=" * 60 + "\n")
        
        # Run pipeline
        results = run_pipeline(
            config,
            do_refine=do_refine,
            do_analysis=do_analysis,
            do_sensitivity=do_sensitivity,
            do_plotting=do_plotting
        )
        
        # Exit with appropriate code
        if results['success']:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure config.yaml exists in the project root.")
        print("You can copy from config_template.yaml and modify as needed.")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⊘ Pipeline interrupted by user.")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()