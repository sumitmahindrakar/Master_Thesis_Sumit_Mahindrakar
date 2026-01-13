"""
Configuration Loader for Sensitivity Analysis Pipeline
======================================================
Loads config.yaml and computes all paths automatically.
Provides single source of truth for all scripts.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple


@dataclass
class PathConfig:
    """All paths used in the pipeline."""
    # Root directories
    project_root: str
    templates_dir: str
    input_dir: str
    output_dir: str
    vtk_dir: str
    plots_dir: str
    scripts_dir: str
    
    # Template files (source)
    template_mdpa: str
    template_materials: str
    template_params_primary: str
    template_params_dual: str
    
    # Generated input files
    refined_mdpa_primary: str
    refined_mdpa_dual: str
    input_materials: str
    input_params_primary: str
    input_params_dual: str
    
    # Output files
    vtk_primary: str
    vtk_dual: str
    sa_results_csv: str
    sa_results_json: str


@dataclass
class ResponseConfig:
    """Response location configuration."""
    x: float
    y: float
    z: float = 0.0
    response_type: str = "moment"
    delta_theta: float = 1.0
    
    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class MeshConfig:
    """Mesh refinement configuration."""
    subdivisions: int = 10


@dataclass
class MaterialConfig:
    """Material properties (optional override)."""
    E: Optional[float] = None
    I: Optional[float] = None


@dataclass
class AnalysisConfig:
    """Analysis options."""
    run_primary: bool = True
    run_dual: bool = True
    compute_SA: bool = True


@dataclass
class PlottingConfig:
    """Plotting options."""
    enabled: bool = True
    save_figures: bool = True
    deflection_scale: Optional[float] = None
    moment_scale: Optional[float] = None
    rotation_scale: Optional[float] = None
    sensitivity_scale: Optional[float] = None
    show_values: bool = True


@dataclass
class OutputConfig:
    """Output options."""
    export_SA: bool = True
    format: str = "both"  # "csv", "json", or "both"
    vtk_output: bool = True


@dataclass
class ProblemConfig:
    """Problem selection."""
    template: str  # "beam", "frame_1story", "frame_2story"
    name: str      # descriptive name for folders


@dataclass
class Config:
    """Complete configuration container."""
    problem: ProblemConfig
    mesh: MeshConfig
    response: ResponseConfig
    material: MaterialConfig
    analysis: AnalysisConfig
    plotting: PlottingConfig
    output: OutputConfig
    paths: PathConfig
    
    # Store raw dict for access to any custom fields
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)


def find_project_root(start_path: Optional[str] = None) -> str:
    """
    Find project root by looking for config.yaml or known structure.
    
    Parameters
    ----------
    start_path : str, optional
        Starting path for search. If None, uses current file location.
        
    Returns
    -------
    str : Absolute path to project root
    """
    if start_path is None:
        # Start from this file's location and go up
        start_path = os.path.dirname(os.path.abspath(__file__))
    
    current = os.path.abspath(start_path)
    
    # Search up to 5 levels
    for _ in range(5):
        # Check for config.yaml
        if os.path.exists(os.path.join(current, "config.yaml")):
            return current
        
        # Check for known directory structure
        if (os.path.exists(os.path.join(current, "templates")) and 
            os.path.exists(os.path.join(current, "scripts"))):
            return current
        
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    
    # Fallback: assume start_path is in scripts/utils/
    fallback = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    
    if os.path.exists(os.path.join(fallback, "config.yaml")):
        return fallback
    
    raise FileNotFoundError(
        "Could not find project root. Make sure config.yaml exists in project root."
    )


def get_template_filename(template: str) -> str:
    """
    Get the MDPA filename for a template.
    
    Parameters
    ----------
    template : str
        Template name: "beam", "frame_1story", "frame_2story"
        
    Returns
    -------
    str : MDPA filename (without path)
    """
    template_files = {
        "beam": "beam.mdpa",
        "frame_1story": "frame_1story.mdpa",
        "frame_2story": "frame_2story.mdpa"
    }
    
    if template not in template_files:
        raise ValueError(
            f"Unknown template '{template}'. "
            f"Available: {list(template_files.keys())}"
        )
    
    return template_files[template]


def compute_paths(project_root: str, problem: ProblemConfig) -> PathConfig:
    """
    Compute all paths based on project root and problem configuration.
    
    Parameters
    ----------
    project_root : str
        Absolute path to project root
    problem : ProblemConfig
        Problem configuration with template and name
        
    Returns
    -------
    PathConfig : All computed paths
    """
    template = problem.template
    name = problem.name
    
    # Template filename
    template_mdpa_name = get_template_filename(template)
    template_base = template_mdpa_name.replace(".mdpa", "")
    
    # Directory paths
    templates_dir = os.path.join(project_root, "templates", template)
    input_dir = os.path.join(project_root, "input_files", f"{name}_input")
    output_dir = os.path.join(project_root, "output_files", f"{name}_output")
    vtk_dir = os.path.join(project_root, "vtk_output", f"{name}_vtk")
    plots_dir = os.path.join(output_dir, "plots")
    scripts_dir = os.path.join(project_root, "scripts")
    
    # Template files (source)
    template_mdpa = os.path.join(templates_dir, template_mdpa_name)
    template_materials = os.path.join(templates_dir, "StructuralMaterials.json")
    template_params_primary = os.path.join(templates_dir, "ProjectParameters.json")
    template_params_dual = os.path.join(templates_dir, "ProjectParameters_dual.json")
    
    # Generated input files
    refined_mdpa_primary = os.path.join(input_dir, f"{template_base}_refined.mdpa")
    refined_mdpa_dual = os.path.join(input_dir, f"{template_base}_dual_refined.mdpa")
    input_materials = os.path.join(input_dir, "StructuralMaterials.json")
    input_params_primary = os.path.join(input_dir, "ProjectParameters.json")
    input_params_dual = os.path.join(input_dir, "ProjectParameters_dual.json")
    
    # VTK output files
    vtk_primary_dir = os.path.join(vtk_dir, "primary")
    vtk_dual_dir = os.path.join(vtk_dir, "dual")
    vtk_primary = os.path.join(vtk_primary_dir, "Parts_Beam_Beams_0_1.vtk")
    vtk_dual = os.path.join(vtk_dual_dir, "Parts_Beam_Beams_0_1.vtk")
    
    # SA result files
    sa_results_csv = os.path.join(output_dir, f"{name}_SA_results.csv")
    sa_results_json = os.path.join(output_dir, f"{name}_SA_results.json")
    
    return PathConfig(
        project_root=project_root,
        templates_dir=templates_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        vtk_dir=vtk_dir,
        plots_dir=plots_dir,
        scripts_dir=scripts_dir,
        template_mdpa=template_mdpa,
        template_materials=template_materials,
        template_params_primary=template_params_primary,
        template_params_dual=template_params_dual,
        refined_mdpa_primary=refined_mdpa_primary,
        refined_mdpa_dual=refined_mdpa_dual,
        input_materials=input_materials,
        input_params_primary=input_params_primary,
        input_params_dual=input_params_dual,
        vtk_primary=vtk_primary,
        vtk_dual=vtk_dual,
        sa_results_csv=sa_results_csv,
        sa_results_json=sa_results_json
    )


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str, optional
        Path to config.yaml. If None, searches for it automatically.
        
    Returns
    -------
    Config : Complete configuration object
    """
    # Find config file
    if config_path is None:
        project_root = find_project_root()
        config_path = os.path.join(project_root, "config.yaml")
    else:
        config_path = os.path.abspath(config_path)
        project_root = os.path.dirname(config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    
    # Parse problem config
    problem_raw = raw.get('problem', {})
    problem = ProblemConfig(
        template=problem_raw.get('template', 'beam'),
        name=problem_raw.get('name', 'example')
    )
    
    # Parse mesh config
    mesh_raw = raw.get('mesh', {})
    mesh = MeshConfig(
        subdivisions=mesh_raw.get('subdivisions', 10)
    )
    
    # Parse response config
    response_raw = raw.get('response', {})
    response = ResponseConfig(
        x=response_raw.get('x', 1.0),
        y=response_raw.get('y', 0.0),
        z=response_raw.get('z', 0.0),
        response_type=response_raw.get('type', 'moment'),
        delta_theta=response_raw.get('delta_theta', 1.0)
    )
    
    # Parse material config
    material_raw = raw.get('material', {})
    material = MaterialConfig(
        E=material_raw.get('E', None),
        I=material_raw.get('I', None)
    )
    
    # Parse analysis config
    analysis_raw = raw.get('analysis', {})
    analysis = AnalysisConfig(
        run_primary=analysis_raw.get('run_primary', True),
        run_dual=analysis_raw.get('run_dual', True),
        compute_SA=analysis_raw.get('compute_SA', True)
    )
    
    # Parse plotting config
    plotting_raw = raw.get('plotting', {})
    plotting = PlottingConfig(
        enabled=plotting_raw.get('enabled', True),
        save_figures=plotting_raw.get('save_figures', True),
        deflection_scale=plotting_raw.get('deflection_scale', None),
        moment_scale=plotting_raw.get('moment_scale', None),
        rotation_scale=plotting_raw.get('rotation_scale', None),
        sensitivity_scale=plotting_raw.get('sensitivity_scale', None),
        show_values=plotting_raw.get('show_values', True)
    )
    
    # Parse output config
    output_raw = raw.get('output', {})
    output = OutputConfig(
        export_SA=output_raw.get('export_SA', True),
        format=output_raw.get('format', 'both'),
        vtk_output=output_raw.get('vtk_output', True)
    )
    
    # Compute paths
    paths = compute_paths(project_root, problem)
    
    config = Config(
        problem=problem,
        mesh=mesh,
        response=response,
        material=material,
        analysis=analysis,
        plotting=plotting,
        output=output,
        paths=paths,
        _raw=raw
    )
    
    print(f"  Template: {config.problem.template}")
    print(f"  Problem name: {config.problem.name}")
    print(f"  Subdivisions: {config.mesh.subdivisions}")
    print(f"  Response location: ({config.response.x}, {config.response.y})")
    print(f"  Delta theta:     {config.response.delta_theta}")
    
    return config


def create_directories(config: Config) -> None:
    """
    Create all necessary directories for the pipeline.
    
    Parameters
    ----------
    config : Config
        Configuration object with paths
    """
    directories = [
        config.paths.input_dir,
        config.paths.output_dir,
        config.paths.vtk_dir,
        os.path.dirname(config.paths.vtk_primary),
        os.path.dirname(config.paths.vtk_dual),
        config.paths.plots_dir
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  Created: {directory}")


def print_config_summary(config: Config) -> None:
    """Print a summary of the configuration."""
    
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    
    print(f"\nProblem:")
    print(f"  Template:        {config.problem.template}")
    print(f"  Name:            {config.problem.name}")
    
    print(f"\nMesh:")
    print(f"  Subdivisions:    {config.mesh.subdivisions}")
    
    print(f"\nResponse Location:")
    print(f"  Coordinates:     ({config.response.x}, {config.response.y}, {config.response.z})")
    print(f"  Type:            {config.response.response_type}")
    
    print(f"\nMaterial (override):")
    print(f"  E:               {config.material.E if config.material.E else 'from JSON'}")
    print(f"  I:               {config.material.I if config.material.I else 'from JSON'}")
    
    print(f"\nAnalysis:")
    print(f"  Run primary:     {config.analysis.run_primary}")
    print(f"  Run dual:        {config.analysis.run_dual}")
    print(f"  Compute SA:      {config.analysis.compute_SA}")
    
    print(f"\nPaths:")
    print(f"  Project root:    {config.paths.project_root}")
    print(f"  Template MDPA:   {config.paths.template_mdpa}")
    print(f"  Input dir:       {config.paths.input_dir}")
    print(f"  Output dir:      {config.paths.output_dir}")
    print(f"  VTK dir:         {config.paths.vtk_dir}")
    
    print("=" * 70)


# =============================================================================
# TEST / STANDALONE USAGE
# =============================================================================

if __name__ == "__main__":
    """Test the configuration loader."""
    
    try:
        config = load_config()
        print_config_summary(config)
        
        print("\nCreating directories...")
        create_directories(config)
        print("Done!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure config.yaml exists in the project root.")