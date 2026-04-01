"""
Parametric Study Runner for Kratos Sensitivity Analysis
========================================================
Supports per-node loads (Fx, Fz, My) with LHS sampling.

Usage:
    python run_parametric_study.py --config config_dataset1_Fx.yaml --setup-only
    python run_parametric_study.py --config config_dataset2.yaml
    python run_parametric_study.py --dry-run
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
    print("Note: scipy not installed. Only fallback LHS will work.\n")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Note: PyYAML not installed. pip install pyyaml")

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

# ── Inline nearest_node_finder fallback ──
try:
    from utils.nearest_node_finder import (
        find_nearest_element_and_location, parse_nodes_from_mdpa
    )
    print("✓ nearest_node_finder imported")
except ModuleNotFoundError:
    print("Note: Using inline nearest_node_finder.")

    def parse_nodes_from_mdpa(mdpa_path):
        nodes = {}
        in_block = False
        with open(mdpa_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Begin Nodes"):
                    in_block = True
                    continue
                elif line.startswith("End Nodes"):
                    break
                elif in_block and line and not line.startswith("//"):
                    parts = line.split()
                    if len(parts) >= 4:
                        nodes[int(parts[0])] = (
                            float(parts[1]), float(parts[2]), float(parts[3])
                        )
        return nodes

    def find_nearest_element_and_location(target_coords, mdpa_path):
        nodes = parse_nodes_from_mdpa(mdpa_path)
        elements = {}
        in_block = False
        with open(mdpa_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Begin Elements"):
                    in_block = True
                    continue
                elif line.startswith("End Elements"):
                    break
                elif in_block and line and not line.startswith("//"):
                    parts = line.split()
                    if len(parts) >= 4:
                        elements[int(parts[0])] = [int(p) for p in parts[2:]]
        min_dist = float('inf')
        nearest_node = None
        for nid, (x, y, z) in nodes.items():
            dist = math.sqrt(
                (x - target_coords[0])**2 +
                (y - target_coords[1])**2 +
                (z - target_coords[2])**2
            )
            if dist < min_dist:
                min_dist = dist
                nearest_node = nid
        for eid, node_ids in elements.items():
            if nearest_node in node_ids:
                return eid, node_ids.index(nearest_node), nearest_node, min_dist
        return 1, 0, nearest_node, min_dist


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class CaseParameters:
    case_id: int
    youngs_modulus: float
    I22: float
    I33: float
    cross_area: float
    torsional_inertia: float
    subdivisions: int
    response_coords: Tuple[float, float, float]
    # ── Per-node loads: {node_id: (Fx, Fz, My)} ──
    node_loads: Dict[int, Tuple[float, float, float]] = field(
        default_factory=dict
    )
    # ── Legacy single load (backward compat) ──
    load_modulus: float = 0.0
    load_direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    # ── Response tracking ──
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
    sampling_method: str = "lhs"
    beam_element_ids: List[int] = field(default_factory=list)


# ============================================================
# HELPERS
# ============================================================

def to_float(val) -> float:
    return float(val)

def to_int(val) -> int:
    return int(float(val)) if isinstance(val, str) else int(val)


# ============================================================
# SAMPLING FUNCTIONS
# ============================================================

def build_parameter_dimensions(config: StudyConfig) -> List[ParameterDimension]:
    """Build LHS dimensions for GLOBAL parameters (not per-node)."""
    dimensions = []
    ranges = config.param_ranges
    has_per_node = "per_node_load" in ranges

    # If per_node_load active, skip legacy point_load/udl
    standard_params = [
        "youngs_modulus", "I22", "I33",
        "cross_area", "torsional_inertia", "subdivisions"
    ]
    if not has_per_node:
        standard_params = ["point_load", "udl"] + standard_params

    for pname in standard_params:
        if pname not in ranges:
            continue
        p = ranges[pname]
        lo = to_float(p["min"])
        hi = to_float(p["max"])
        dist = p.get("distribution", "uniform")
        if lo == hi:
            continue
        if dist == "loguniform" and (lo <= 0 or hi <= 0):
            dist = "uniform"
        dimensions.append(ParameterDimension(pname, lo, hi, dist))

    if "response_location" in ranges:
        resp = ranges["response_location"]
        rdist = resp.get("distribution", "uniform")
        for axis in ["x", "y", "z"]:
            if axis not in resp:
                continue
            lo = to_float(resp[axis]["min"])
            hi = to_float(resp[axis]["max"])
            if lo == hi:
                continue
            dimensions.append(
                ParameterDimension(f"response_{axis}", lo, hi, rdist)
            )

    return dimensions


def build_per_node_dimensions(
    config: StudyConfig,
    eligible_nodes: List[int]
) -> List[ParameterDimension]:
    """Build LHS dimensions for per-node loads (Fx, Fz, My × each node)."""
    pn_cfg = config.param_ranges.get("per_node_load", {})
    if not pn_cfg:
        return []

    dims = []
    for comp in ["Fx", "Fz", "My"]:
        if comp not in pn_cfg:
            continue
        cc = pn_cfg[comp]
        lo = to_float(cc["min"])
        hi = to_float(cc["max"])
        if lo == hi:
            continue  # Fixed → no dimension needed
        dist = cc.get("distribution", "uniform")
        for nid in eligible_nodes:
            dims.append(ParameterDimension(
                f"node_{nid}_{comp}", lo, hi, dist
            ))

    return dims


def generate_unit_samples(n_samples, n_dimensions, method="lhs", seed=None):
    """Generate unit hypercube samples [0,1]^d."""
    if n_dimensions == 0:
        return np.empty((n_samples, 0))

    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.random((n_samples, n_dimensions))

    elif method == "lhs":
        if HAS_SCIPY:
            sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)
            return sampler.random(n=n_samples)
        return _lhs_fallback(n_samples, n_dimensions, seed)

    elif method == "optimal_lhs":
        if not HAS_SCIPY:
            return _lhs_fallback(n_samples, n_dimensions, seed)
        try:
            sampler = qmc.LatinHypercube(
                d=n_dimensions, seed=seed, optimization="random-cd"
            )
            return sampler.random(n=n_samples)
        except TypeError:
            sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)
            return sampler.random(n=n_samples)

    elif method == "sobol":
        if not HAS_SCIPY:
            raise ImportError("scipy required for Sobol.")
        sampler = qmc.Sobol(d=n_dimensions, scramble=True, seed=seed)
        return sampler.random(n=n_samples)

    raise ValueError(f"Unknown sampling method: '{method}'")


def _lhs_fallback(n_samples, n_dimensions, seed=None):
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, n_dimensions))
    for j in range(n_dimensions):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            lo = perm[i] / n_samples
            hi = (perm[i] + 1) / n_samples
            samples[i, j] = rng.uniform(lo, hi)
    return samples


def transform_sample(u, dim):
    """Map unit sample [0,1] → physical value."""
    if dim.distribution == "uniform":
        return dim.min_val + u * (dim.max_val - dim.min_val)
    elif dim.distribution == "loguniform":
        log_lo = math.log10(dim.min_val)
        log_hi = math.log10(dim.max_val)
        return 10 ** (log_lo + u * (log_hi - log_lo))
    elif dim.distribution == "discrete":
        n_vals = int(dim.max_val - dim.min_val) + 1
        idx = min(int(u * n_vals), n_vals - 1)
        return int(dim.min_val + idx)
    return dim.min_val + u * (dim.max_val - dim.min_val)


# ============================================================
# PRE-REFINEMENT (for node counting)
# ============================================================

def _pre_refine_for_nodes(config: StudyConfig) -> List[int]:
    """Refine mesh once (no disk write) to get eligible node list."""
    template = os.path.join(config.working_directory, config.template_mdpa)
    if not os.path.exists(template):
        raise FileNotFoundError(f"Template MDPA not found: {template}")

    mdpa_data = parse_mdpa(template)
    subdivisions = int(to_float(
        config.param_ranges["subdivisions"]["min"]
    ))
    beam_ids = config.beam_element_ids or None
    

    print(f"\n  Pre-refining mesh (subdivisions={subdivisions})...")
    refined = refine_mesh(mdpa_data, subdivisions, beam_ids,
        generate_moment_conditions=False )
    nodes = refined.condition_node_list
    print(f"    Eligible load nodes: {len(nodes)}")
    return nodes


# ============================================================
# MASTER SAMPLING FUNCTION
# ============================================================

def generate_all_case_parameters(config, start_case=1):
    """
    Generate all case parameters using combined LHS.
    Returns: (case_params_list, unit_samples, all_dims, eligible_nodes)
    """
    ranges = config.param_ranges
    n_samples = config.num_cases
    has_per_node = "per_node_load" in ranges

    # ── 1. Get eligible nodes ──
    eligible_nodes = _pre_refine_for_nodes(config) if has_per_node else []

    # ── 2. Build combined dimension list ──
    global_dims = build_parameter_dimensions(config)
    per_node_dims = build_per_node_dimensions(config, eligible_nodes)
    all_dims = global_dims + per_node_dims

    n_global = len(global_dims)
    n_per_node = len(per_node_dims)
    n_total = len(all_dims)

    print(f"\n{'─' * 60}")
    print(f"SAMPLING CONFIGURATION")
    print(f"{'─' * 60}")
    print(f"  Method:              {config.sampling_method}")
    print(f"  Cases:               {n_samples}")
    print(f"  Global dims:         {n_global}")
    print(f"  Per-node dims:       {n_per_node}")
    print(f"  Total dims:          {n_total}")

    if has_per_node:
        pn_cfg = ranges["per_node_load"]
        active = [c for c in ["Fx", "Fz", "My"]
                  if c in pn_cfg and
                  to_float(pn_cfg[c]["min"]) != to_float(pn_cfg[c]["max"])]
        fixed = [c for c in ["Fx", "Fz", "My"]
                 if c in pn_cfg and
                 to_float(pn_cfg[c]["min"]) == to_float(pn_cfg[c]["max"])]
        print(f"  Active components:   {active}")
        print(f"  Fixed components:    {fixed}")
        print(f"  Sparsity:            {pn_cfg.get('sparsity', 0.0)}")

    for i, dim in enumerate(global_dims):
        print(f"    [G{i}] {dim.name}: "
              f"[{dim.min_val:.4e}, {dim.max_val:.4e}] ({dim.distribution})")
    if n_per_node > 10:
        print(f"    [N0..N{n_per_node-1}] {n_per_node} per-node dims")
    else:
        for i, dim in enumerate(per_node_dims):
            print(f"    [N{i}] {dim.name}: "
                  f"[{dim.min_val:.4e}, {dim.max_val:.4e}]")

    # ── 3. Generate LHS ──
    if n_total > 0:
        print(f"\n  Generating {n_samples} × {n_total} LHS samples...")
        unit_samples = generate_unit_samples(
            n_samples, n_total, config.sampling_method, config.random_seed
        )
        # if n_samples > 1 and HAS_SCIPY and n_total >= 1:
        #     disc = qmc.discrepancy(unit_samples)
        #     print(f"  L2 Discrepancy: {disc:.6e}")
        if n_samples > 1 and HAS_SCIPY and n_total >= 1:
            if n_total <= 20:
                disc = qmc.discrepancy(unit_samples)
                print(f"  L2 Discrepancy: {disc:.6e}")
            else:
                print(f"  (Discrepancy skipped — {n_total}D too high "
                      f"for meaningful metric)")

        samples_path = os.path.join(
            config.working_directory,
            f"unit_samples_{config.sampling_method}.csv"
        )
        os.makedirs(config.working_directory, exist_ok=True)
        header = ",".join(d.name for d in all_dims)
        np.savetxt(samples_path, unit_samples, delimiter=",",
                   header=header, comments="")
        print(f"  Saved: {samples_path}")
    else:
        unit_samples = None

    # ── 4. Transform to physical parameters ──
    all_params = []
    pn_cfg = ranges.get("per_node_load", {})
    sparsity = pn_cfg.get("sparsity", 0.0)

    # Fixed values for components not in LHS
    fix_Fx = to_float(pn_cfg.get("Fx", {}).get("min", 0.0)) \
        if has_per_node else 0.0
    fix_Fz = to_float(pn_cfg.get("Fz", {}).get("min", 0.0)) \
        if has_per_node else 0.0
    fix_My = to_float(pn_cfg.get("My", {}).get("min", 0.0)) \
        if has_per_node else 0.0

    for i in range(n_samples):
        case_id = start_case + i

        # Transform all dims for this sample
        pv = {}  # param_values
        if unit_samples is not None:
            for j, dim in enumerate(all_dims):
                pv[dim.name] = transform_sample(unit_samples[i, j], dim)

        # Fill fixed global params
        scalar_list = ["youngs_modulus", "I22", "I33",
                       "cross_area", "torsional_inertia", "subdivisions"]
        if not has_per_node:
            scalar_list = ["point_load", "udl"] + scalar_list

        for pname in scalar_list:
            if pname in ranges and pname not in pv:
                pv[pname] = to_float(ranges[pname]["min"])

        # Response coords
        if "response_location" in ranges:
            resp = ranges["response_location"]
            for axis in ["x", "y", "z"]:
                key = f"response_{axis}"
                if key not in pv and axis in resp:
                    pv[key] = to_float(resp[axis]["min"])

        response_coords = (
            pv.get("response_x", 0.0),
            pv.get("response_y", 0.0),
            pv.get("response_z", 0.0),
        )

        # ── Per-node loads ──
        node_loads = {}
        if has_per_node and eligible_nodes:
            sp_seed = (config.random_seed * 1000 + i) \
                if config.random_seed else None
            sp_rng = np.random.default_rng(sp_seed)

            for nid in eligible_nodes:
                Fx = pv.get(f"node_{nid}_Fx", fix_Fx)
                Fz = pv.get(f"node_{nid}_Fz", fix_Fz)
                My = pv.get(f"node_{nid}_My", fix_My)

                # Sparsity: zero out this node
                if sparsity > 0 and sp_rng.random() < sparsity:
                    Fx, Fz, My = 0.0, 0.0, 0.0

                node_loads[nid] = (Fx, Fz, My)

        # Legacy single load
        load_mod = pv.get("point_load", pv.get("udl", 0.0)) \
            if not has_per_node else 0.0

        all_params.append(CaseParameters(
            case_id=case_id,
            youngs_modulus=pv["youngs_modulus"],
            I22=pv["I22"],
            I33=pv["I33"],
            cross_area=pv["cross_area"],
            torsional_inertia=pv["torsional_inertia"],
            subdivisions=int(pv["subdivisions"]),
            response_coords=response_coords,
            node_loads=node_loads,
            load_modulus=load_mod,
        ))

    return all_params, unit_samples, all_dims, eligible_nodes


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


# def _build_per_node_load_processes(params, refined_data):
#     force_processes = []
#     moment_processes = []

#     for nid, (Fx, Fz, My) in params.node_loads.items():
#         # ── Use separate SMP names ──
#         smp_info = refined_data.per_node_smps[nid]
#         force_smp = smp_info['force']     # "AutoForce_node_3"
#         moment_smp = smp_info['moment']   # "AutoMoment_node_3"

#         # ── Force ──
#         force_mag = math.sqrt(Fx**2 + Fz**2)
#         if force_mag > 1e-15:
#             direction = [Fx / force_mag, 0.0, Fz / force_mag]
#         else:
#             direction = [0.0, 0.0, -1.0]
#             force_mag = 0.0

#         force_processes.append({
#             "python_module": "assign_vector_by_direction_to_condition_process",
#             "kratos_module": "KratosMultiphysics",
#             "Parameters": {
#                 "model_part_name": f"Structure.{force_smp}",
#                 "variable_name": "POINT_LOAD",
#                 "modulus": force_mag,
#                 "direction": direction,
#                 "interval": [0.0, "End"]
#             }
#         })

#         # ── Moment ──
#         moment_mag = abs(My)
#         if moment_mag > 1e-15:
#             moment_dir = [0.0, 1.0 if My > 0 else -1.0, 0.0]
#         else:
#             moment_dir = [0.0, 1.0, 0.0]
#             moment_mag = 0.0

#         moment_processes.append({
#             "python_module": "assign_vector_by_direction_to_condition_process",
#             "kratos_module": "KratosMultiphysics",
#             "Parameters": {
#                 "model_part_name": f"Structure.{moment_smp}",
#                 "variable_name": "POINT_MOMENT",
#                 "modulus": moment_mag,
#                 "direction": moment_dir,
#                 "interval": [0.0, "End"]
#             }
#         })

#     return force_processes, moment_processes

def _build_per_node_load_processes(params, refined_data):
    force_processes = []
    moment_processes = []

    for nid, (Fx, Fz, My) in params.node_loads.items():
        smp_info = refined_data.per_node_smps[nid]

        # ── Force ──
        force_smp = smp_info['force']
        force_mag = math.sqrt(Fx**2 + Fz**2)
        if force_mag > 1e-15:
            direction = [Fx / force_mag, 0.0, Fz / force_mag]
        else:
            direction = [0.0, 0.0, -1.0]
            force_mag = 0.0

        force_processes.append({
            "python_module": "assign_vector_by_direction_to_condition_process",
            "kratos_module": "KratosMultiphysics",
            "Parameters": {
                "model_part_name": f"Structure.{force_smp}",
                "variable_name": "POINT_LOAD",
                "modulus": force_mag,
                "direction": direction,
                "interval": [0.0, "End"]
            }
        })

        # ── Moment (only if SMP exists) ──
        if 'moment' in smp_info:
            moment_smp = smp_info['moment']
            moment_mag = abs(My)
            if moment_mag > 1e-15:
                moment_dir = [0.0, 1.0 if My > 0 else -1.0, 0.0]
            else:
                moment_dir = [0.0, 1.0, 0.0]
                moment_mag = 0.0

            moment_processes.append({
                "python_module": "assign_vector_by_direction_to_condition_process",
                "kratos_module": "KratosMultiphysics",
                "Parameters": {
                    "model_part_name": f"Structure.{moment_smp}",
                    "variable_name": "POINT_MOMENT",
                    "modulus": moment_mag,
                    "direction": moment_dir,
                    "interval": [0.0, "End"]
                }
            })

    return force_processes, moment_processes


def modify_primal_json(template_path, output_path, params,
                       mdpa_filename, materials_filename,
                       vtk_output_path, fixed, refined_data=None):
    """Modify primal JSON with per-node loads."""
    with open(template_path, 'r') as f:
        primal = json.load(f)

    primal["solver_settings"]["model_import_settings"][
        "input_filename"] = mdpa_filename
    primal["solver_settings"]["material_import_settings"][
        "materials_filename"] = materials_filename

    # ── Replace load processes ──
    if params.node_loads and refined_data is not None:
        # Remove old POINT_LOAD / LINE_LOAD processes
        old_loads = primal["processes"]["loads_process_list"]
        non_load = [p for p in old_loads
                    if p["Parameters"].get("variable_name")
                    not in ("POINT_LOAD", "LINE_LOAD", "POINT_MOMENT")]

        force_procs, moment_procs = _build_per_node_load_processes(
            params, refined_data
        )
        primal["processes"]["loads_process_list"] = (
            non_load + force_procs + moment_procs
        )
    else:
        # Legacy: single load
        for lp in primal["processes"]["loads_process_list"]:
            p = lp["Parameters"]
            if p.get("variable_name") in ("POINT_LOAD", "LINE_LOAD"):
                p["modulus"] = params.load_modulus
                p["direction"] = list(params.load_direction)

    # Gravity
    for lp in primal["processes"]["loads_process_list"]:
        if "VOLUME_ACCELERATION" in lp["Parameters"].get(
                "variable_name", ""):
            gravity = fixed.get("gravity", [0.0, 0.0, -9.81])
            if isinstance(gravity, list):
                lp["Parameters"]["value"] = gravity

    # VTK output path
    if "output_processes" in primal:
        for vtk in primal.get("output_processes", {}).get("vtk_output", []):
            vtk["Parameters"]["output_path"] = vtk_output_path

    with open(output_path, 'w') as f:
        json.dump(primal, f, indent=4)


def modify_adjoint_json(template_path, output_path, params,
                        mdpa_filename, materials_filename,
                        vtk_output_path, response_settings,
                        refined_data=None):
    """Modify adjoint JSON with per-node loads."""
    with open(template_path, 'r') as f:
        adjoint = json.load(f)

    adjoint["solver_settings"]["model_import_settings"][
        "input_filename"] = mdpa_filename
    adjoint["solver_settings"]["material_import_settings"][
        "materials_filename"] = materials_filename

    # Response settings
    resp = adjoint["solver_settings"]["response_function_settings"]
    resp["traced_element_id"] = params.traced_element_id
    resp["stress_location"] = params.stress_location
    resp["stress_type"] = response_settings.get("stress_type", "MY")
    resp["stress_treatment"] = response_settings.get(
        "stress_treatment", "node"
    )

    # ── Replace load processes ──
    if params.node_loads and refined_data is not None:
        old_loads = adjoint["processes"]["loads_process_list"]
        non_load = [p for p in old_loads
                    if p["Parameters"].get("variable_name")
                    not in ("POINT_LOAD", "LINE_LOAD", "POINT_MOMENT")]

        force_procs, moment_procs = _build_per_node_load_processes(
            params, refined_data
        )
        adjoint["processes"]["loads_process_list"] = (
            non_load + force_procs + moment_procs
        )
    else:
        for lp in adjoint["processes"]["loads_process_list"]:
            p = lp["Parameters"]
            if p.get("variable_name") in ("POINT_LOAD", "LINE_LOAD"):
                p["modulus"] = params.load_modulus
                p["direction"] = list(params.load_direction)

    # VTK
    if "output_processes" in adjoint:
        for vtk in adjoint.get("output_processes", {}).get(
                "vtk_output", []):
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
    print(f"  Young's Modulus:     {params.youngs_modulus:.4e}")
    print(f"  I22:                 {params.I22:.4e}")
    print(f"  I33:                 {params.I33:.4e}")
    print(f"  Cross Area:          {params.cross_area:.6f}")
    print(f"  Torsional Inertia:   {params.torsional_inertia:.4e}")
    print(f"  Subdivisions:        {params.subdivisions}")
    print(f"  Response coords:     ({params.response_coords[0]:.2f}, "
          f"{params.response_coords[1]:.2f}, "
          f"{params.response_coords[2]:.2f})")

    # ── Per-node load summary ──
    if params.node_loads:
        n_loaded = sum(
            1 for (Fx, Fz, My) in params.node_loads.values()
            if abs(Fx) > 1e-15 or abs(Fz) > 1e-15 or abs(My) > 1e-15
        )
        n_total = len(params.node_loads)
        print(f"  Per-node loads:      {n_loaded}/{n_total} nodes loaded")

        # Print first 5 as sample
        for idx, (nid, (Fx, Fz, My)) in enumerate(
                params.node_loads.items()):
            if idx >= 5:
                print(f"    ... ({n_total - 5} more)")
                break
            print(f"    Node {nid:3d}: "
                  f"Fx={Fx:8.2f}  Fz={Fz:8.2f}  My={My:8.2f}")
    else:
        print(f"  Load modulus:        {params.load_modulus:.2f}")

    if dry_run:
        print("  [DRY RUN — skipping file ops]")
        return primal_folder, adjoint_folder

    # ── Step 1: Refine MDPA ──
    template_mdpa = os.path.join(
        config.working_directory, config.template_mdpa
    )
    if not os.path.exists(template_mdpa):
        raise FileNotFoundError(
            f"Template MDPA not found: {template_mdpa}"
        )

    print(f"  Reading template MDPA...")
    mdpa_data = parse_mdpa(template_mdpa)

    beam_ids = config.beam_element_ids or None

    # ── Determine if moments are active ──                    # ← ADD
    pn_cfg = config.param_ranges.get("per_node_load", {})      # ← ADD
    has_moments = False                                         # ← ADD
    if "My" in pn_cfg:                                          # ← ADD
        my_min = to_float(pn_cfg["My"]["min"])                  # ← ADD
        my_max = to_float(pn_cfg["My"]["max"])                  # ← ADD
        has_moments = (my_min != my_max) or (abs(my_min) > 1e-15)  # ← ADD


    if params.subdivisions > 1:
        print(f"  Refining ({params.subdivisions} subdivisions)...")
        refined_data = refine_mesh(
            mdpa_data, params.subdivisions, beam_ids,
            generate_moment_conditions=has_moments #
        )
        print(f"  Refined: {len(mdpa_data.nodes)} → "
              f"{len(refined_data.nodes)} nodes, "
              f"{len(refined_data.conditions_force)} force conds, "
              f"{len(refined_data.conditions_moment)} moment conds, "
              f"{len(refined_data.per_node_smps)} per-node SMPs")
    else:
        refined_data = refine_mesh(mdpa_data, 1, beam_ids
                                   ,generate_moment_conditions=has_moments
                                   )

    mdpa_basename = "Frame_structure_refined"
    primal_mdpa = os.path.join(primal_folder, f"{mdpa_basename}.mdpa")
    adjoint_mdpa = os.path.join(adjoint_folder, f"{mdpa_basename}.mdpa")

    write_mdpa(refined_data, primal_mdpa)
    shutil.copy(primal_mdpa, adjoint_mdpa)

    # ── Step 2: Find nearest element for response ──
    elem_id, stress_loc, node_id, dist = \
        find_nearest_element_and_location(
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
        raise FileNotFoundError(
            f"Template materials not found: {template_materials}"
        )

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
        raise FileNotFoundError(
            f"Template primal not found: {template_primal}"
        )

    primal_params_json = os.path.join(
        primal_folder, "beam_test_parameters.json"
    )
    modify_primal_json(
        template_primal, primal_params_json, params,
        mdpa_basename, "materials_beam.json",
        "vtk_output_primal", config.fixed_params,
        refined_data=refined_data
    )

    # ── Step 5: Adjoint JSON ──
    template_adjoint = os.path.join(
        config.working_directory, config.template_adjoint
    )
    if not os.path.exists(template_adjoint):
        raise FileNotFoundError(
            f"Template adjoint not found: {template_adjoint}"
        )

    adjoint_params_json = os.path.join(
        adjoint_folder,
        "beam_test_local_stress_adjoint_parameters.json"
    )
    modify_adjoint_json(
        template_adjoint, adjoint_params_json, params,
        mdpa_basename, "materials_beam.json",
        "vtk_output_adjoint", config.response_settings,
        refined_data=refined_data
    )

    # Copy primal params to adjoint folder too
    shutil.copy(
        primal_params_json,
        os.path.join(adjoint_folder, "beam_test_parameters.json")
    )

    # ── Step 6: Save case config ──
    node_loads_serializable = {
        str(nid): {"Fx": Fx, "Fz": Fz, "My": My}
        for nid, (Fx, Fz, My) in params.node_loads.items()
    }

    case_config = {
        "case_id": case_id,
        "parameters": {
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
            "nearest_node_distance": params.nearest_node_distance,
            "node_loads": node_loads_serializable,
        },
        "load_summary": {
            "total_nodes": len(params.node_loads),
            "loaded_nodes": sum(
                1 for (Fx, Fz, My) in params.node_loads.values()
                if abs(Fx) > 1e-15 or abs(Fz) > 1e-15 or abs(My) > 1e-15
            ),
            "Fx_range": [
                min(v[0] for v in params.node_loads.values()),
                max(v[0] for v in params.node_loads.values())
            ] if params.node_loads else [0, 0],
            "Fz_range": [
                min(v[1] for v in params.node_loads.values()),
                max(v[1] for v in params.node_loads.values())
            ] if params.node_loads else [0, 0],
            "My_range": [
                min(v[2] for v in params.node_loads.values()),
                max(v[2] for v in params.node_loads.values())
            ] if params.node_loads else [0, 0],
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
            param_file = \
                "beam_test_local_stress_adjoint_parameters.json"

        with open(param_file, 'r') as f:
            parameters = KratosMultiphysics.Parameters(f.read())

        model = KratosMultiphysics.Model()

        if analysis_type == "primal":
            class CustomPrimalAnalysis(
                structural_mechanics_analysis
                .StructuralMechanicsAnalysis
            ):
                def Initialize(self):
                    super().Initialize()
                    mp_name = self.project_parameters[
                        "solver_settings"][
                        "model_part_name"].GetString()
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
                        "solver_settings"][
                        "model_part_name"].GetString()
                    mp = self.model.GetModelPart(mp_name)

                    POINT_LOAD = SMA.POINT_LOAD
                    POINT_MOMENT = SMA.POINT_MOMENT

                    zero = KratosMultiphysics.Array3(
                        [0.0, 0.0, 0.0]
                    )
                    for node in mp.Nodes:
                        node.SetValue(POINT_LOAD, zero)
                        node.SetValue(POINT_MOMENT, zero)

                    # Transfer loads from conditions to nodes
                    point_load_count = 0
                    moment_count = 0

                    # ── Scan ALL conditions for loads ──
                    for condition in mp.Conditions:
                        # Point loads
                        if condition.Has(POINT_LOAD):
                            load_val = condition.GetValue(
                                POINT_LOAD
                            )
                            if (abs(load_val[0]) > 1e-12 or
                                abs(load_val[1]) > 1e-12 or
                                abs(load_val[2]) > 1e-12):
                                for node in \
                                        condition.GetGeometry():
                                    existing = node.GetValue(
                                        POINT_LOAD
                                    )
                                    combined = \
                                        KratosMultiphysics.Array3([
                                            existing[0] + load_val[0],
                                            existing[1] + load_val[1],
                                            existing[2] + load_val[2]
                                        ])
                                    node.SetValue(
                                        POINT_LOAD, combined
                                    )
                                    point_load_count += 1

                        # Point moments
                        if condition.Has(POINT_MOMENT):
                            mom_val = condition.GetValue(
                                POINT_MOMENT
                            )
                            if (abs(mom_val[0]) > 1e-12 or
                                abs(mom_val[1]) > 1e-12 or
                                abs(mom_val[2]) > 1e-12):
                                for node in \
                                        condition.GetGeometry():
                                    existing = node.GetValue(
                                        POINT_MOMENT
                                    )
                                    combined = \
                                        KratosMultiphysics.Array3([
                                            existing[0] + mom_val[0],
                                            existing[1] + mom_val[1],
                                            existing[2] + mom_val[2]
                                        ])
                                    node.SetValue(
                                        POINT_MOMENT, combined
                                    )
                                    moment_count += 1

                    if point_load_count > 0:
                        print(f"    Transferred POINT_LOAD to "
                              f"{point_load_count} nodes")
                    if moment_count > 0:
                        print(f"    Transferred POINT_MOMENT to "
                              f"{moment_count} nodes")

                    super().FinalizeSolutionStep()

            analysis = CustomPrimalAnalysis(model, parameters)
        else:
            analysis = (
                structural_mechanics_analysis
                .StructuralMechanicsAnalysis(model, parameters)
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
        in_script = os.path.join(script_dir, config_path)
        if os.path.exists(in_script):
            config_path = in_script
        elif not os.path.exists(config_path):
            print(f"ERROR: Config not found!")
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

    required = ["general", "output", "templates", "parameters"]
    for key in required:
        if key not in cfg:
            print(f"ERROR: Missing key '{key}' in config")
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
        sampling_method=cfg["general"].get(
            "sampling_method", "lhs"
        ),
        beam_element_ids=cfg.get("beam_element_ids", []),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parametric study for Kratos — per-node loads"
    )
    parser.add_argument(
        "--config", default="config_dataset2.yaml",#config.yaml config_test.yaml config_test_no_moment.yaml
        help="Config file path" #config_dataset1_Fx.yaml config_dataset1_Fz.yaml
    ) #config_dataset2.yaml
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--setup-only", action="store_true",
        default=False)#True for test run
    parser.add_argument("--start-case", type=int, default=1)
    parser.add_argument(
        "--method",
        choices=["random", "lhs", "optimal_lhs", "sobol"],
        default=None
    )
    args = parser.parse_args()

    print("=" * 60)
    print("KRATOS PARAMETRIC STUDY — PER-NODE LOADS")
    print("=" * 60)

    config = load_config(args.config)

    if args.method:
        config.sampling_method = args.method

    if config.random_seed is not None:
        random.seed(config.random_seed)
        print(f"Random seed: {config.random_seed}")

    print(f"Working directory: {config.working_directory}")
    print(f"Number of cases:  {config.num_cases}")
    print(f"Sampling method:  {config.sampling_method}")

    os.makedirs(
        os.path.join(
            config.working_directory, config.primal_folder
        ), exist_ok=True
    )
    os.makedirs(
        os.path.join(
            config.working_directory, config.adjoint_folder
        ), exist_ok=True
    )

    # ── Generate ALL parameters at once ──
    all_case_params, unit_samples, all_dims, eligible_nodes = \
        generate_all_case_parameters(config, args.start_case)

    all_cases: List[Tuple[int, str, str, CaseParameters]] = []

    # ── Setup cases ──
    for params in all_case_params:
        try:
            primal_folder, adjoint_folder = setup_case(
                params.case_id, params, config,
                dry_run=args.dry_run
            )
            all_cases.append((
                params.case_id, primal_folder,
                adjoint_folder, params
            ))
        except Exception as e:
            print(f"  ERROR case {params.case_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── Save study summary ──
    if not args.dry_run:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config_file": args.config,
            "num_cases": config.num_cases,
            "random_seed": config.random_seed,
            "sampling_method": config.sampling_method,
            "eligible_load_nodes": eligible_nodes,
            "num_eligible_nodes": len(eligible_nodes),
            "varying_dimensions": [
                {
                    "name": d.name,
                    "min": d.min_val,
                    "max": d.max_val,
                    "distribution": d.distribution
                }
                for d in all_dims
            ],
            "cases": [
                {
                    "id": c[0],
                    "primal": c[1],
                    "adjoint": c[2],
                    "parameters": {
                        "youngs_modulus": c[3].youngs_modulus,
                        "I22": c[3].I22,
                        "I33": c[3].I33,
                        "cross_area": c[3].cross_area,
                        "torsional_inertia": c[3].torsional_inertia,
                        "subdivisions": c[3].subdivisions,
                        "response_coords": list(
                            c[3].response_coords
                        ),
                        "traced_element_id": (
                            c[3].traced_element_id
                        ),
                        "stress_location": c[3].stress_location,
                        "num_loaded_nodes": sum(
                            1 for (Fx, Fz, My)
                            in c[3].node_loads.values()
                            if abs(Fx) > 1e-15
                            or abs(Fz) > 1e-15
                            or abs(My) > 1e-15
                        ),
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
        print(f"\nStudy summary: {summary_path}")

    # ── Run analyses ──
    if not args.dry_run and not args.setup_only:
        print("\n" + "=" * 60)
        print("RUNNING ANALYSES")
        print("=" * 60)

        results = []

        for case_id, primal_folder, adjoint_folder, params \
                in all_cases:
            print(
                f"\n{'>' * 20} Case {case_id} {'<' * 20}"
            )

            primal_ok = run_kratos_analysis(
                primal_folder, "primal"
            )

            if primal_ok:
                h5_copied = 0
                for fname in os.listdir(primal_folder):
                    if fname.endswith('.h5'):
                        shutil.copy(
                            os.path.join(
                                primal_folder, fname
                            ),
                            os.path.join(
                                adjoint_folder, fname
                            )
                        )
                        h5_copied += 1
                if h5_copied > 0:
                    print(
                        f"  Copied {h5_copied} HDF5 file(s)"
                    )

                adjoint_ok = run_kratos_analysis(
                    adjoint_folder, "adjoint"
                )
            else:
                adjoint_ok = False
                print("  Skipping adjoint (primal failed)")

            results.append({
                "case_id": case_id,
                "primal_success": primal_ok,
                "adjoint_success": adjoint_ok
            })

        # Save results
        results_path = os.path.join(
            config.working_directory, "results_summary.json"
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        # Print summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        successful = sum(
            1 for r in results
            if r["primal_success"] and r["adjoint_success"]
        )
        primal_only = sum(
            1 for r in results
            if r["primal_success"]
            and not r["adjoint_success"]
        )
        failed = sum(
            1 for r in results
            if not r["primal_success"]
        )

        print(
            f"  Fully successful: "
            f"{successful}/{len(results)}"
        )
        print(
            f"  Primal only:      "
            f"{primal_only}/{len(results)}"
        )
        print(
            f"  Failed:           "
            f"{failed}/{len(results)}"
        )
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