"""
FEM-based Sensitivity Analysis modules.
"""

from .refine_mesh import (
    refine_and_prepare,
    parse_mdpa,
    refine_mesh,
    write_mdpa,
    create_dual_mdpa,
    RefinementResult,
    MdpaData,
    SubModelPart
)

from .run_analysis import (
    run_all_analyses,
    run_primary_analysis,
    run_dual_analysis,
    run_primary_standalone,
    run_dual_standalone,
    load_hinge_info
)

from .compute_sensitivity import (
    compute_sensitivity_analysis,
    export_to_csv,
    export_to_json,
    SensitivityResults,
    ElementSensitivity
)

from .plot_results import (
    generate_plots,
    create_all_plots,
    parse_vtk_file
)

__all__ = [
    # Mesh refinement
    'refine_and_prepare',
    'parse_mdpa',
    'refine_mesh',
    'write_mdpa',
    'create_dual_mdpa',
    'RefinementResult',
    'MdpaData',
    'SubModelPart',
    
    # Analysis
    'run_all_analyses',
    'run_primary_analysis',
    'run_dual_analysis',
    'run_primary_standalone',
    'run_dual_standalone',
    'load_hinge_info',
    
    # Sensitivity
    'compute_sensitivity_analysis',
    'export_to_csv',
    'export_to_json',
    'SensitivityResults',
    'ElementSensitivity',
    
    # Plotting
    'generate_plots',
    'create_all_plots',
    'parse_vtk_file'
]