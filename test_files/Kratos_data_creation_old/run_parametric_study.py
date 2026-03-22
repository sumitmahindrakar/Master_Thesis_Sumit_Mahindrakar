"""
Parametric Study Runner for Kratos Sensitivity Analysis
========================================================

Usage:
    python run_parametric_study.py
    python run_parametric_study.py --config custom_config.yaml
    python run_parametric_study.py --dry-run
    python run_parametric_study.py --setup-only
    python run_parametric_study.py --method sobol
"""

import os
import sys
import json
import shutil
import random
import math
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

try:
    from scipy.stats import qmc
    from scipy.spatial.distance import pdist
    HAS_SCIPY = True
    print("✓ scipy.stats.qmc imported")
except ImportError:
    HAS_SCIPY = False
    print("Note: scipy not installed. Only 'random' sampling will work.\n")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Note: PyYAML not installed. Use JSON config or: pip install pyyaml")

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from mdpa_refiner import parse_mdpa, refine_mesh, write_mdpa
    print("✓ mdpa_refiner imported successfully")
except ModuleNotFoundError:
    parent_dir = os.path.dirname(script_dir)
    sys.path.insert(0, parent_dir)
    try:
        from mdpa_refiner import parse_mdpa, refine_mesh, write_mdpa
        print("✓ mdpa_refiner imported from parent directory")
    except ModuleNotFoundError:
        print("✗ ERROR: mdpa_refiner.py not found!")
        sys.exit(1)

try:
    from utils.nearest_node_finder import find_nearest_element_and_location, parse_nodes_from_mdpa
    print("✓ nearest_node_finder imported successfully")
except ModuleNotFoundError:
    print("Note: Using inline nearest_node_finder.")

    def parse_nodes_from_mdpa(mdpa_path: str) -> Dict[int, Tuple[float, float, float]]:
        nodes = {}
        in_nodes_block = False
        with open(mdpa_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Begin Nodes"):
                    in_nodes_block = True
                    continue
                elif line.startswith("End Nodes"):
                    break
                elif in_nodes_block and line and not line.startswith("//"):
                    parts = line.split()
                    if len(parts) >= 4:
                        nodes[int(parts[0])] = (
                            float(parts[1]), float(parts[2]), float(parts[3])
                        )
        return nodes

    def find_nearest_element_and_location(
        target_coords: Tuple[float, float, float],
        mdpa_path: str
    ) -> Tuple[int, int, int, float]:
        nodes = parse_nodes_from_mdpa(mdpa_path)
        elements = {}
        in_elements_block = False
        with open(mdpa_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Begin Elements"):
                    in_elements_block = True
                    continue
                elif line.startswith("End Elements"):
                    break
                elif in_elements_block and line and not line.startswith("//"):
                    parts = line.split()
                    if len(parts) >= 4:
                        elements[int(parts[0])] = [int(p) for p in parts[2:]]

        min_dist = float('inf')
        nearest_node = None
        for node_id, (x, y, z) in nodes.items():
            dist = math.sqrt(
                (x - target_coords[0])**2 +
                (y - target_coords[1])**2 +
                (z - target_coords[2])**2
            )
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_id

        for elem_id, node_ids in elements.items():
            if nearest_node in node_ids:
                stress_location = node_ids.index(nearest_node)
                return elem_id, stress_location, nearest_node, min_dist

        return 1, 0, nearest_node, min_dist


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class CaseParameters:
    case_id: int
    load_modulus: float
    youngs_modulus: float
    I22: float
    I33: float
    cross_area: float
    torsional_inertia: float
    subdivisions: int
    response_coords: Tuple[float, float, float]
    traced_element_id: int = 0
    stress_location: int = 0
    nearest_node_id: int = 0
    nearest_node_distance: float = 0.0


@dataclass
class ParameterDimension:
    name: str
    min_val: float
    max_val: float
    distribution: str


@dataclass
class StudyConfig:
    num_cases: int
    random_seed: Optional[int]
    working_directory: str
    primal_folder: str
    adjoint_folder: str
    primal_prefix: str
    adjoint_prefix: str
    template_mdpa: str
    template_primal: str
    template_adjoint: str
    template_materials: str
    param_ranges: Dict[str, Any]
    fixed_params: Dict[str, Any]
    response_settings: Dict[str, Any]
    sampling_method: str = "random"
    beam_element_ids: List[int] = field(default_factory=list)


# ============================================================
# SAMPLING FUNCTIONS
# ============================================================

def to_float(val) -> float:
    return float(val)


def to_int(val) -> int:
    if isinstance(val, str):
        return int(float(val))
    return int(val)


# ============================================================
# ADVANCED SAMPLING
# ============================================================

def build_parameter_dimensions(config: StudyConfig) -> List[ParameterDimension]:
    dimensions = []
    ranges = config.param_ranges

    standard_params = [
        "point_load", "udl",
        "youngs_modulus", "I22", "I33",
        "cross_area", "torsional_inertia", "subdivisions"
    ]

    for param_name in standard_params:
        if param_name not in ranges:
            continue
        p = ranges[param_name]
        min_val = to_float(p["min"])
        max_val = to_float(p["max"])
        distribution = p.get("distribution", "uniform")
        if min_val == max_val:
            continue
        if distribution == "loguniform" and (min_val <= 0 or max_val <= 0):
            print(f"  WARNING: {param_name} loguniform with non-positive bounds. "
                  f"Using uniform.")
            distribution = "uniform"
        dimensions.append(ParameterDimension(
            name=param_name, min_val=min_val,
            max_val=max_val, distribution=distribution
        ))

    if "response_location" in ranges:
        resp = ranges["response_location"]
        resp_dist = resp.get("distribution", "uniform")
        for axis in ["x", "y", "z"]:
            if axis not in resp:
                continue
            min_val = to_float(resp[axis]["min"])
            max_val = to_float(resp[axis]["max"])
            if min_val == max_val:
                continue
            dimensions.append(ParameterDimension(
                name=f"response_{axis}", min_val=min_val,
                max_val=max_val, distribution=resp_dist
            ))

    return dimensions


def generate_unit_samples(
    n_samples: int,
    n_dimensions: int,
    method: str = "random",
    seed: Optional[int] = None
) -> np.ndarray:
    if n_dimensions == 0:
        return np.empty((n_samples, 0))

    if method == "random":
        rng = np.random.default_rng(seed)
        samples = rng.random((n_samples, n_dimensions))

    elif method == "lhs":
        if HAS_SCIPY:
            sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)
            samples = sampler.random(n=n_samples)
        else:
            samples = _lhs_fallback(n_samples, n_dimensions, seed)

    elif method == "optimal_lhs":
        if not HAS_SCIPY:
            samples = _lhs_fallback(n_samples, n_dimensions, seed)
        else:
            try:
                sampler = qmc.LatinHypercube(
                    d=n_dimensions, seed=seed, optimization="random-cd"
                )
                samples = sampler.random(n=n_samples)
            except TypeError:
                sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)
                samples = sampler.random(n=n_samples)

    elif method == "sobol":
        if not HAS_SCIPY:
            raise ImportError("scipy required for Sobol sequences.")
        if n_samples > 0 and (n_samples & (n_samples - 1)) != 0:
            next_pow2 = 2 ** int(np.ceil(np.log2(max(n_samples, 1))))
            print(f"  ⚠ Sobol works best with 2^m samples. "
                  f"Requested: {n_samples}, nearest: {next_pow2}")
        sampler = qmc.Sobol(d=n_dimensions, scramble=True, seed=seed)
        samples = sampler.random(n=n_samples)

    elif method == "maximin_lhs":
        if HAS_SCIPY:
            samples = _maximin_lhs(n_samples, n_dimensions, seed)
        else:
            samples = _lhs_fallback(n_samples, n_dimensions, seed)
    else:
        raise ValueError(f"Unknown sampling method: '{method}'")

    return samples


def _lhs_fallback(n_samples, n_dimensions, seed=None):
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, n_dimensions))
    for j in range(n_dimensions):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            low = perm[i] / n_samples
            high = (perm[i] + 1) / n_samples
            samples[i, j] = rng.uniform(low, high)
    return samples


def _maximin_lhs(n_samples, n_dimensions, seed=None,
                 n_iterations=1000, n_restarts=5):
    rng = np.random.default_rng(seed)
    best_design = None
    best_mindist = -1.0
    scaled_iterations = min(
        max(n_iterations, n_samples * n_dimensions * 5), 50000
    )

    for restart in range(n_restarts):
        design = np.zeros((n_samples, n_dimensions))
        for j in range(n_dimensions):
            perm = rng.permutation(n_samples)
            for i in range(n_samples):
                low = perm[i] / n_samples
                high = (perm[i] + 1) / n_samples
                design[i, j] = rng.uniform(low, high)

        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d2 = np.sum((design[i] - design[j]) ** 2)
                dist_matrix[i, j] = d2
                dist_matrix[j, i] = d2

        np.fill_diagonal(dist_matrix, np.inf)
        current_mindist = np.sqrt(np.min(dist_matrix))
        np.fill_diagonal(dist_matrix, 0.0)

        improvements = 0
        for iteration in range(scaled_iterations):
            col = rng.integers(0, n_dimensions)
            row1, row2 = rng.choice(n_samples, size=2, replace=False)
            old_val1 = design[row1, col]
            old_val2 = design[row2, col]
            design[row1, col] = old_val2
            design[row2, col] = old_val1

            new_min_d2 = np.inf
            old_dists_row1 = dist_matrix[row1, :].copy()
            old_dists_row2 = dist_matrix[row2, :].copy()

            for k in range(n_samples):
                if k == row1:
                    continue
                d2 = np.sum((design[row1] - design[k]) ** 2)
                dist_matrix[row1, k] = d2
                dist_matrix[k, row1] = d2
                if k != row2:
                    new_min_d2 = min(new_min_d2, d2)

            for k in range(n_samples):
                if k == row2:
                    continue
                d2 = np.sum((design[row2] - design[k]) ** 2)
                dist_matrix[row2, k] = d2
                dist_matrix[k, row2] = d2
                new_min_d2 = min(new_min_d2, d2)

            for i in range(n_samples):
                if i == row1 or i == row2:
                    continue
                for j in range(i + 1, n_samples):
                    if j == row1 or j == row2:
                        continue
                    new_min_d2 = min(new_min_d2, dist_matrix[i, j])

            new_mindist = np.sqrt(new_min_d2)
            if new_mindist > current_mindist:
                current_mindist = new_mindist
                improvements += 1
            else:
                design[row1, col] = old_val1
                design[row2, col] = old_val2
                dist_matrix[row1, :] = old_dists_row1
                dist_matrix[:, row1] = old_dists_row1
                dist_matrix[row2, :] = old_dists_row2
                dist_matrix[:, row2] = old_dists_row2

        if current_mindist > best_mindist:
            best_mindist = current_mindist
            best_design = design.copy()

        if n_restarts > 1:
            print(f"    Restart {restart + 1}/{n_restarts}: "
                  f"mindist={current_mindist:.6f}, "
                  f"improvements={improvements}/{scaled_iterations}")

    print(f"  Maximin LHS: best mindist = {best_mindist:.6f}")
    return best_design


def transform_sample(u: float, dim: ParameterDimension) -> float:
    if dim.distribution == "uniform":
        return dim.min_val + u * (dim.max_val - dim.min_val)
    elif dim.distribution == "loguniform":
        log_min = math.log10(dim.min_val)
        log_max = math.log10(dim.max_val)
        return 10 ** (log_min + u * (log_max - log_min))
    elif dim.distribution == "normal":
        mean = (dim.min_val + dim.max_val) / 2
        std = (dim.max_val - dim.min_val) / 4
        from statistics import NormalDist
        u_clipped = max(1e-10, min(1 - 1e-10, u))
        value = NormalDist(mu=mean, sigma=std).inv_cdf(u_clipped)
        return max(dim.min_val, min(dim.max_val, value))
    elif dim.distribution == "discrete":
        n_values = int(dim.max_val - dim.min_val) + 1
        idx = min(int(u * n_values), n_values - 1)
        return int(dim.min_val + idx)
    else:
        return dim.min_val + u * (dim.max_val - dim.min_val)


def generate_all_case_parameters(
    config: StudyConfig,
    start_case: int = 1
) -> Tuple[List[CaseParameters], Optional[np.ndarray], List[ParameterDimension]]:

    dimensions = build_parameter_dimensions(config)
    n_dims = len(dimensions)
    n_samples = config.num_cases
    ranges = config.param_ranges

    print(f"\n{'─' * 50}")
    print(f"SAMPLING CONFIGURATION")
    print(f"{'─' * 50}")
    print(f"  Method:             {config.sampling_method}")
    print(f"  Number of samples:  {n_samples}")
    print(f"  Varying dimensions: {n_dims}")

    for i, dim in enumerate(dimensions):
        print(f"    [{i}] {dim.name}: "
              f"[{dim.min_val:.4e}, {dim.max_val:.4e}] "
              f"({dim.distribution})")

    # Show fixed parameters
    all_scalar = ["point_load", "udl", "youngs_modulus", "I22", "I33",
                  "cross_area", "torsional_inertia", "subdivisions"]
    fixed_info = []
    for p_name in all_scalar:
        if p_name in ranges:
            if to_float(ranges[p_name]["min"]) == to_float(ranges[p_name]["max"]):
                fixed_info.append(
                    f"    {p_name}: {to_float(ranges[p_name]['min']):.4e}"
                )
    if "response_location" in ranges:
        resp = ranges["response_location"]
        for axis in ["x", "y", "z"]:
            if axis in resp:
                if to_float(resp[axis]["min"]) == to_float(resp[axis]["max"]):
                    fixed_info.append(
                        f"    response_{axis}: {to_float(resp[axis]['min']):.4e}"
                    )
    if fixed_info:
        print(f"  Fixed parameters:")
        for info in fixed_info:
            print(info)

    # Generate unit samples
    if n_dims > 0:
        print(f"\n  Generating {n_samples} × {n_dims} unit samples...")
        unit_samples = generate_unit_samples(
            n_samples=n_samples, n_dimensions=n_dims,
            method=config.sampling_method, seed=config.random_seed
        )

        if n_samples > 1 and n_dims >= 1 and HAS_SCIPY:
            disc = qmc.discrepancy(unit_samples)
            print(f"  Centered L2 Discrepancy: {disc:.6e}")
            distances = pdist(unit_samples)
            print(f"  Min distance:  {np.min(distances):.6f}")
            print(f"  Mean distance: {np.mean(distances):.6f}")

        print(f"  ✓ Unit samples generated")

        samples_path = os.path.join(
            config.working_directory,
            f"unit_samples_{config.sampling_method}.csv"
        )
        os.makedirs(config.working_directory, exist_ok=True)
        header = ",".join(dim.name for dim in dimensions)
        np.savetxt(samples_path, unit_samples, delimiter=",",
                   header=header, comments="")
        print(f"  Saved unit samples to: {samples_path}")
    else:
        unit_samples = None

    # Transform to physical parameters
    all_params = []

    for i in range(n_samples):
        case_id = start_case + i

        param_values = {}
        for p_name in all_scalar:
            if p_name in ranges:
                param_values[p_name] = to_float(ranges[p_name]["min"])

        if "response_location" in ranges:
            resp = ranges["response_location"]
            for axis in ["x", "y", "z"]:
                if axis in resp:
                    param_values[f"response_{axis}"] = to_float(resp[axis]["min"])

        if unit_samples is not None:
            for j, dim in enumerate(dimensions):
                param_values[dim.name] = transform_sample(
                    unit_samples[i, j], dim
                )

        response_coords = (
            param_values.get("response_x", 0.0),
            param_values.get("response_y", 0.0),
            param_values.get("response_z", 0.0),
        )

        load_modulus = param_values.get(
            "point_load", param_values.get("udl", 0.0)
        )

        all_params.append(CaseParameters(
            case_id=case_id,
            load_modulus=load_modulus,
            youngs_modulus=param_values["youngs_modulus"],
            I22=param_values["I22"],
            I33=param_values["I33"],
            cross_area=param_values["cross_area"],
            torsional_inertia=param_values["torsional_inertia"],
            subdivisions=int(param_values["subdivisions"]),
            response_coords=response_coords,
        ))

    return all_params, unit_samples, dimensions


# ============================================================
# FILE MODIFICATION FUNCTIONS
# ============================================================

def modify_materials_json(template_path, output_path, params, fixed):
    with open(template_path, 'r') as f:
        materials = json.load(f)

    for prop in materials["properties"]:
        variables = prop["Material"]["Variables"]
        variables["YOUNG_MODULUS"] = params.youngs_modulus
        variables["I22"] = params.I22
        variables["I33"] = params.I33
        variables["CROSS_AREA"] = params.cross_area
        variables["TORSIONAL_INERTIA"] = params.torsional_inertia
        variables["DENSITY"] = fixed.get("density", 7850.0)
        variables["POISSON_RATIO"] = fixed.get("poisson_ratio", 0.3)

    with open(output_path, 'w') as f:
        json.dump(materials, f, indent=4)


def modify_primal_json(template_path, output_path, params,
                       mdpa_filename, materials_filename,
                       vtk_output_path, fixed):
    """Modify primal JSON. Only changes modulus — everything else from template."""
    with open(template_path, 'r') as f:
        primal = json.load(f)

    primal["solver_settings"]["model_import_settings"]["input_filename"] = mdpa_filename
    primal["solver_settings"]["material_import_settings"]["materials_filename"] = materials_filename

    # Only update the modulus value
    for load_process in primal["processes"]["loads_process_list"]:
        lp = load_process["Parameters"]
        if lp.get("variable_name") in ("POINT_LOAD", "LINE_LOAD"):
            lp["modulus"] = params.load_modulus

    # Gravity
    for load_process in primal["processes"]["loads_process_list"]:
        if "VOLUME_ACCELERATION" in load_process["Parameters"].get("variable_name", ""):
            gravity = fixed.get("gravity", [0.0, 0.0, -9.81])
            if isinstance(gravity, list):
                load_process["Parameters"]["value"] = gravity

    if "output_processes" in primal:
        for vtk in primal.get("output_processes", {}).get("vtk_output", []):
            vtk["Parameters"]["output_path"] = vtk_output_path

    with open(output_path, 'w') as f:
        json.dump(primal, f, indent=4)


def modify_adjoint_json(template_path, output_path, params,
                        mdpa_filename, materials_filename,
                        vtk_output_path, response_settings):
    """Modify adjoint JSON. Only changes modulus + response — rest from template."""
    with open(template_path, 'r') as f:
        adjoint = json.load(f)

    adjoint["solver_settings"]["model_import_settings"]["input_filename"] = mdpa_filename
    adjoint["solver_settings"]["material_import_settings"]["materials_filename"] = materials_filename

    resp = adjoint["solver_settings"]["response_function_settings"]
    resp["traced_element_id"] = params.traced_element_id
    resp["stress_location"] = params.stress_location
    resp["stress_type"] = response_settings.get("stress_type", "MY")
    resp["stress_treatment"] = response_settings.get("stress_treatment", "node")

    # Only update the modulus value
    for load_process in adjoint["processes"]["loads_process_list"]:
        lp = load_process["Parameters"]
        if lp.get("variable_name") in ("POINT_LOAD", "LINE_LOAD"):
            lp["modulus"] = params.load_modulus

    if "output_processes" in adjoint:
        for vtk in adjoint.get("output_processes", {}).get("vtk_output", []):
            vtk["Parameters"]["output_path"] = vtk_output_path

    with open(output_path, 'w') as f:
        json.dump(adjoint, f, indent=4)


# ============================================================
# CASE SETUP
# ============================================================

def setup_case(case_id, params, config, dry_run=False):
    primal_folder = os.path.join(
        config.working_directory, config.primal_folder,
        f"{config.primal_prefix}_{case_id}"
    )
    adjoint_folder = os.path.join(
        config.working_directory, config.adjoint_folder,
        f"{config.adjoint_prefix}_{case_id}"
    )

    os.makedirs(primal_folder, exist_ok=True)
    os.makedirs(adjoint_folder, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Setting up Case {case_id}")
    print(f"{'='*60}")
    print(f"  Load modulus: {params.load_modulus:.2f}")
    print(f"  Young's Modulus: {params.youngs_modulus:.4e}")
    print(f"  I22: {params.I22:.4e}")
    print(f"  I33: {params.I33:.4e}")
    print(f"  Cross Area: {params.cross_area:.6f}")
    print(f"  Torsional Inertia: {params.torsional_inertia:.4e}")
    print(f"  Subdivisions: {params.subdivisions}")
    print(f"  Response coords: ({params.response_coords[0]:.2f}, "
          f"{params.response_coords[1]:.2f}, {params.response_coords[2]:.2f})")

    if dry_run:
        print("  [DRY RUN - Skipping file operations]")
        return primal_folder, adjoint_folder

    # ── Step 1: Refine MDPA ──
    template_mdpa = os.path.join(config.working_directory, config.template_mdpa)
    if not os.path.exists(template_mdpa):
        raise FileNotFoundError(f"Template MDPA not found: {template_mdpa}")

    print(f"  Reading template MDPA...")
    mdpa_data = parse_mdpa(template_mdpa)

    if params.subdivisions > 1:
        print(f"  Refining with {params.subdivisions} subdivisions...")
        beam_ids = config.beam_element_ids or None             # ← ADD
        refined_data = refine_mesh(
            mdpa_data, params.subdivisions, beam_ids            # ← ADD
        )
        print(f"  Refined: {len(mdpa_data.nodes)} -> "
              f"{len(refined_data.nodes)} nodes, "
              f"{len(refined_data.conditions)} conditions")
    else:
        refined_data = mdpa_data

    mdpa_basename = "Frame_structure_refined"
    primal_mdpa = os.path.join(primal_folder, f"{mdpa_basename}.mdpa")
    adjoint_mdpa = os.path.join(adjoint_folder, f"{mdpa_basename}.mdpa")

    write_mdpa(refined_data, primal_mdpa)
    shutil.copy(primal_mdpa, adjoint_mdpa)

    # ── Step 2: Find nearest element for response ──
    elem_id, stress_loc, node_id, dist = find_nearest_element_and_location(
        params.response_coords, primal_mdpa
    )
    params.traced_element_id = elem_id
    params.stress_location = stress_loc
    params.nearest_node_id = node_id
    params.nearest_node_distance = dist

    print(f"  Response: node {node_id} (dist: {dist:.4f}), "
          f"element {elem_id}, loc {stress_loc}")

    # ── Step 3: Materials JSON ──
    template_materials = os.path.join(
        config.working_directory, config.template_materials
    )
    if not os.path.exists(template_materials):
        raise FileNotFoundError(f"Template materials not found: {template_materials}")

    primal_materials = os.path.join(primal_folder, "materials_beam.json")
    adjoint_materials = os.path.join(adjoint_folder, "materials_beam.json")
    modify_materials_json(
        template_materials, primal_materials, params, config.fixed_params
    )
    shutil.copy(primal_materials, adjoint_materials)

    # ── Step 4: Primal JSON ──
    template_primal = os.path.join(
        config.working_directory, config.template_primal
    )
    if not os.path.exists(template_primal):
        raise FileNotFoundError(f"Template primal not found: {template_primal}")

    primal_params_json = os.path.join(primal_folder, "beam_test_parameters.json")
    modify_primal_json(
        template_primal, primal_params_json, params,
        mdpa_basename, "materials_beam.json",
        "vtk_output_primal", config.fixed_params
    )

    # ── Step 5: Adjoint JSON ──
    template_adjoint = os.path.join(
        config.working_directory, config.template_adjoint
    )
    if not os.path.exists(template_adjoint):
        raise FileNotFoundError(f"Template adjoint not found: {template_adjoint}")

    adjoint_params_json = os.path.join(
        adjoint_folder, "beam_test_local_stress_adjoint_parameters.json"
    )
    modify_adjoint_json(
        template_adjoint, adjoint_params_json, params,
        mdpa_basename, "materials_beam.json",
        "vtk_output_adjoint", config.response_settings
    )

    shutil.copy(
        primal_params_json,
        os.path.join(adjoint_folder, "beam_test_parameters.json")
    )

    # ── Step 6: Save case config ──
    case_config = {
        "case_id": case_id,
        "parameters": {
            "load_modulus": params.load_modulus,
            "youngs_modulus": params.youngs_modulus,
            "I22": params.I22,
            "I33": params.I33,
            "cross_area": params.cross_area,
            "torsional_inertia": params.torsional_inertia,
            "subdivisions": params.subdivisions,
            "response_coords": list(params.response_coords),
            "traced_element_id": params.traced_element_id,
            "stress_location": params.stress_location,
            "nearest_node_id": params.nearest_node_id,
            "nearest_node_distance": params.nearest_node_distance
        },
        "timestamp": datetime.now().isoformat()
    }

    with open(os.path.join(primal_folder, "case_config.json"), 'w') as f:
        json.dump(case_config, f, indent=4)
    with open(os.path.join(adjoint_folder, "case_config.json"), 'w') as f:
        json.dump(case_config, f, indent=4)

    print(f"  ✓ Case {case_id} setup complete")
    return primal_folder, adjoint_folder


# ============================================================
# KRATOS RUNNER
# ============================================================

def run_kratos_analysis(case_folder, analysis_type="primal"):
    original_dir = os.getcwd()

    try:
        os.chdir(case_folder)
        print(f"\n  Running {analysis_type} analysis in: {case_folder}")

        import KratosMultiphysics
        import KratosMultiphysics.StructuralMechanicsApplication as SMA
        from KratosMultiphysics.StructuralMechanicsApplication import (
            structural_mechanics_analysis
        )

        if analysis_type == "primal":
            param_file = "beam_test_parameters.json"
        else:
            param_file = "beam_test_local_stress_adjoint_parameters.json"

        with open(param_file, 'r') as f:
            parameters = KratosMultiphysics.Parameters(f.read())

        model = KratosMultiphysics.Model()

        if analysis_type == "primal":
            class CustomPrimalAnalysis(
                structural_mechanics_analysis.StructuralMechanicsAnalysis
            ):
                def Initialize(self):
                    super().Initialize()
                    mp_name = self.project_parameters[
                        "solver_settings"]["model_part_name"].GetString()
                    mp = self.model.GetModelPart(mp_name)

                    _vars = [
                        KratosMultiphysics.YOUNG_MODULUS,
                        KratosMultiphysics.DENSITY,
                        KratosMultiphysics.POISSON_RATIO,
                        SMA.CROSS_AREA,
                        SMA.TORSIONAL_INERTIA,
                        SMA.I22,
                        SMA.I33,
                    ]
                    for element in mp.Elements:
                        props = element.Properties
                        for var in _vars:
                            if props.Has(var):
                                element.SetValue(var, props[var])

                def FinalizeSolutionStep(self):
                    mp_name = self.project_parameters[
                        "solver_settings"]["model_part_name"].GetString()
                    mp = self.model.GetModelPart(mp_name)

                    POINT_LOAD = SMA.POINT_LOAD
                    POINT_MOMENT = SMA.POINT_MOMENT

                    zero = KratosMultiphysics.Array3([0.0, 0.0, 0.0])
                    for node in mp.Nodes:
                        node.SetValue(POINT_LOAD, zero)
                        node.SetValue(POINT_MOMENT, zero)

                    # Transfer POINT_LOAD from conditions to nodes
                    point_load_count = 0
                    smp_names = ["PointLoad3D_load", "LineLoad3D_load"]
                    for smp_name in smp_names:
                        try:
                            load_mp = mp.GetSubModelPart(smp_name)
                            for condition in load_mp.Conditions:
                                if condition.Has(POINT_LOAD):
                                    load_val = condition.GetValue(POINT_LOAD)
                                    for node in condition.GetGeometry():
                                        node.SetValue(POINT_LOAD, load_val)
                                        point_load_count += 1
                            if point_load_count > 0:
                                break
                        except:
                            continue

                    if point_load_count == 0:
                        for condition in mp.Conditions:
                            if condition.Has(POINT_LOAD):
                                load_val = condition.GetValue(POINT_LOAD)
                                if (abs(load_val[0]) > 1e-12 or
                                    abs(load_val[1]) > 1e-12 or
                                    abs(load_val[2]) > 1e-12):
                                    for node in condition.GetGeometry():
                                        node.SetValue(POINT_LOAD, load_val)
                                        point_load_count += 1

                    if point_load_count > 0:
                        print(f"    Transferred POINT_LOAD to "
                              f"{point_load_count} nodes")

                    # Transfer POINT_MOMENT
                    moment_count = 0
                    moment_smps = [
                        "PointMoment3D_moment", "PointMoment3D_load",
                        "PointMoment3D", "Moment3D_moment", "PointMoment"
                    ]
                    for submp_name in moment_smps:
                        try:
                            moment_mp = mp.GetSubModelPart(submp_name)
                            for condition in moment_mp.Conditions:
                                if condition.Has(POINT_MOMENT):
                                    moment_val = condition.GetValue(POINT_MOMENT)
                                    for node in condition.GetGeometry():
                                        node.SetValue(POINT_MOMENT, moment_val)
                                        moment_count += 1
                            if moment_count > 0:
                                break
                        except:
                            continue

                    if moment_count > 0:
                        print(f"    Transferred POINT_MOMENT to "
                              f"{moment_count} nodes")

                    super().FinalizeSolutionStep()

            analysis = CustomPrimalAnalysis(model, parameters)
        else:
            analysis = structural_mechanics_analysis.StructuralMechanicsAnalysis(
                model, parameters
            )

        analysis.Run()
        print(f"  ✓ {analysis_type.capitalize()} completed")
        return True

    except Exception as e:
        import traceback
        print(f"  ✗ {analysis_type.capitalize()} failed: {e}")
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_dir)


# ============================================================
# CONFIG LOADING
# ============================================================

def load_config(config_path: str) -> StudyConfig:
    if not os.path.isabs(config_path):
        config_in_script_dir = os.path.join(script_dir, config_path)
        if os.path.exists(config_in_script_dir):
            config_path = config_in_script_dir
        elif not os.path.exists(config_path):
            print(f"ERROR: Config file not found!")
            print(f"  Looked in: {os.getcwd()}")
            print(f"  Looked in: {script_dir}")
            sys.exit(1)

    print(f"Loading config from: {config_path}")

    with open(config_path, 'r') as f:
        if config_path.endswith(('.yaml', '.yml')):
            if HAS_YAML:
                cfg = yaml.safe_load(f)
            else:
                print("ERROR: PyYAML required for .yaml files.")
                sys.exit(1)
        else:
            cfg = json.load(f)

    required_keys = ["general", "output", "templates", "parameters"]
    for key in required_keys:
        if key not in cfg:
            print(f"ERROR: Missing required key '{key}' in config")
            sys.exit(1)

    return StudyConfig(
        num_cases=cfg["general"]["num_cases"],
        random_seed=cfg["general"].get("random_seed"),
        working_directory=cfg["general"]["working_directory"],
        primal_folder=cfg["output"]["primal_folder"],
        adjoint_folder=cfg["output"]["adjoint_folder"],
        primal_prefix=cfg["output"]["primal_prefix"],
        adjoint_prefix=cfg["output"]["adjoint_prefix"],
        template_mdpa=cfg["templates"]["mdpa"],
        template_primal=cfg["templates"]["primal_json"],
        template_adjoint=cfg["templates"]["adjoint_json"],
        template_materials=cfg["templates"]["materials_json"],
        param_ranges=cfg["parameters"],
        fixed_params=cfg.get("fixed", {}),
        response_settings=cfg.get("response", {}),
        sampling_method=cfg["general"].get("sampling_method", "random"),
        beam_element_ids=cfg.get("beam_element_ids", []),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run parametric study for Kratos"
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Config file path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--start-case", type=int, default=1)
    parser.add_argument("--run-existing", type=int, nargs='+')
    parser.add_argument("--method",
                        choices=["random", "lhs", "optimal_lhs",
                                 "sobol", "maximin_lhs"],
                        default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("KRATOS PARAMETRIC STUDY RUNNER")
    print("=" * 60)

    config = load_config(args.config)

    if args.method:
        config.sampling_method = args.method

    if config.random_seed is not None:
        random.seed(config.random_seed)
        print(f"Random seed: {config.random_seed}")

    print(f"Working directory: {config.working_directory}")
    print(f"Number of cases: {config.num_cases}")
    print(f"Sampling method: {config.sampling_method}")

    os.makedirs(
        os.path.join(config.working_directory, config.primal_folder),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(config.working_directory, config.adjoint_folder),
        exist_ok=True
    )

    # Generate ALL parameters at once
    all_case_params, unit_samples, dimensions = \
        generate_all_case_parameters(config, args.start_case)

    all_cases: List[Tuple[int, str, str, CaseParameters]] = []

    # Setup cases
    for params in all_case_params:
        try:
            primal_folder, adjoint_folder = setup_case(
                params.case_id, params, config, dry_run=args.dry_run
            )
            all_cases.append((
                params.case_id, primal_folder, adjoint_folder, params
            ))
        except Exception as e:
            print(f"  ERROR setting up case {params.case_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save study summary
    if not args.dry_run:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config_file": args.config,
            "num_cases": config.num_cases,
            "random_seed": config.random_seed,
            "sampling_method": config.sampling_method,
            "varying_dimensions": [
                {"name": d.name, "min": d.min_val, "max": d.max_val,
                 "distribution": d.distribution}
                for d in dimensions
            ],
            "cases": [
                {
                    "id": c[0], "primal": c[1], "adjoint": c[2],
                    "parameters": {
                        "load_modulus": c[3].load_modulus,
                        "youngs_modulus": c[3].youngs_modulus,
                        "I22": c[3].I22, "I33": c[3].I33,
                        "cross_area": c[3].cross_area,
                        "torsional_inertia": c[3].torsional_inertia,
                        "subdivisions": c[3].subdivisions,
                        "response_coords": list(c[3].response_coords),
                        "traced_element_id": c[3].traced_element_id,
                        "stress_location": c[3].stress_location
                    }
                }
                for c in all_cases
            ]
        }

        summary_path = os.path.join(
            config.working_directory, "study_summary.json"
        )
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"\nStudy summary saved to: {summary_path}")

    # Run analyses
    if not args.dry_run and not args.setup_only:
        print("\n" + "=" * 60)
        print("RUNNING ANALYSES")
        print("=" * 60)

        results = []

        for case_id, primal_folder, adjoint_folder, params in all_cases:
            print(f"\n{'>' * 20} Case {case_id} {'<' * 20}")

            primal_success = run_kratos_analysis(primal_folder, "primal")

            if primal_success:
                h5_copied = 0
                for f_name in os.listdir(primal_folder):
                    if f_name.endswith('.h5'):
                        shutil.copy(
                            os.path.join(primal_folder, f_name),
                            os.path.join(adjoint_folder, f_name)
                        )
                        h5_copied += 1
                if h5_copied > 0:
                    print(f"  Copied {h5_copied} HDF5 file(s)")

                adjoint_success = run_kratos_analysis(
                    adjoint_folder, "adjoint"
                )
            else:
                adjoint_success = False
                print("  Skipping adjoint (primal failed)")

            results.append({
                "case_id": case_id,
                "primal_success": primal_success,
                "adjoint_success": adjoint_success
            })

        results_path = os.path.join(
            config.working_directory, "results_summary.json"
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        successful = sum(
            1 for r in results
            if r["primal_success"] and r["adjoint_success"]
        )
        primal_only = sum(
            1 for r in results
            if r["primal_success"] and not r["adjoint_success"]
        )
        failed = sum(1 for r in results if not r["primal_success"])

        print(f"  Fully successful: {successful}/{len(results)}")
        print(f"  Primal only: {primal_only}/{len(results)}")
        print(f"  Failed: {failed}/{len(results)}")
        print()

        for r in results:
            if r["primal_success"] and r["adjoint_success"]:
                status = "✓ PASS"
            elif r["primal_success"]:
                status = "◐ PRIMAL OK"
            else:
                status = "✗ FAIL"
            print(f"  Case {r['case_id']}: {status}")

    print("\n" + "=" * 60)
    print("PARAMETRIC STUDY COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()