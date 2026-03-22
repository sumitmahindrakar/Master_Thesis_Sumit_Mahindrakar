"""
Parametric Study Runner for Kratos Sensitivity Analysis
========================================================

This script:
1. Reads configuration from YAML or JSON
2. Generates random parameter samples
3. Creates case folders with modified input files
4. Runs Kratos analyses
5. Organizes outputs

Usage:
    python run_parametric_study.py
    python run_parametric_study.py --config custom_config.yaml
    python run_parametric_study.py --config config.json
    python run_parametric_study.py --dry-run
    python run_parametric_study.py --setup-only
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
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np

try:
    from scipy.stats import qmc
    from scipy.spatial.distance import pdist
    HAS_SCIPY = True
    print("✓ scipy.stats.qmc imported")
except ImportError:
    HAS_SCIPY = False
    print("Note: scipy not installed. Only 'ramdom' sampling will work.\n")
    print(" Install with: pip install scipy\n")

# Try to import yaml
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Note: PyYAML not installed. Use JSON config or install with: pip install pyyaml")

# ============================================================
# IMPORT LOCAL MODULES
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import mdpa_refiner
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
        print(f"  Please copy mdpa_refiner.py to: {script_dir}")
        sys.exit(1)

# Import or create nearest_node_finder
try:
    from utils.nearest_node_finder import find_nearest_element_and_location, parse_nodes_from_mdpa
    print("✓ nearest_node_finder imported successfully")
except ModuleNotFoundError:
    print("Note: utils/nearest_node_finder.py not found. Using inline version.")
    
    def parse_nodes_from_mdpa(mdpa_path: str) -> Dict[int, Tuple[float, float, float]]:
        """Parse nodes from MDPA file."""
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
                        node_id = int(parts[0])
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        nodes[node_id] = (x, y, z)
        return nodes
    
    def find_nearest_element_and_location(
        target_coords: Tuple[float, float, float],
        mdpa_path: str
    ) -> Tuple[int, int, int, float]:
        """Find nearest node, element, and stress location."""
        nodes = parse_nodes_from_mdpa(mdpa_path)
        
        # Parse elements
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
                        elem_id = int(parts[0])
                        node_ids = [int(p) for p in parts[2:]]
                        elements[elem_id] = node_ids
        
        # Find nearest node
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
        
        # Find element containing node
        for elem_id, node_ids in elements.items():
            if nearest_node in node_ids:
                stress_location = node_ids.index(nearest_node)
                return elem_id, stress_location, nearest_node, min_dist
        
        # Fallback
        return 1, 0, nearest_node, min_dist


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class CaseParameters:
    """Parameters for a single analysis case."""
    case_id: int
    udl: float
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
    """
    Describes one dimension in the sampling space.
    
    CONCEPT: Each varying parameter becomes one "dimension" of the
    design space. The sampling methods work in a unit hypercube [0,1]^D
    where D = number of varying parameters.
    
    Parameters where min == max are FIXED (not a dimension).
    
    Example with your config:
        Varying (7 dimensions):
            udl:               [20, 100]      uniform
            youngs_modulus:     [1e10, 2.5e11] loguniform
            I22:               [1e-5, 1e-3]   loguniform
            cross_area:        [0.01, 0.1]    uniform
            torsional_inertia: [1e-6, 1e-4]   loguniform
            response_x:        [0, 6]         uniform
            response_z:        [3, 18]        uniform
        
        Fixed (not dimensions):
            I33:          1e-5 (min == max)
            subdivisions: 3    (min == max)
            response_y:   0.0  (min == max)
    """
    name: str              # e.g., "udl", "youngs_modulus", "response_x"
    min_val: float         # Lower bound of the parameter range
    max_val: float         # Upper bound of the parameter range
    distribution: str      # How to transform [0,1] → actual value

@dataclass
class StudyConfig:
    """Complete study configuration."""
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


# ============================================================
# SAMPLING FUNCTIONS
# ============================================================

def to_float(val) -> float:
    """Convert value to float, handling strings and scientific notation."""
    if isinstance(val, str):
        return float(val)
    return float(val)


def to_int(val) -> int:
    """Convert value to int, handling strings."""
    if isinstance(val, str):
        return int(float(val))
    return int(val)

# ============================================================
# can delete or use das Utility
# ============================================================
def sample_uniform(min_val, max_val) -> float:
    """Sample from uniform distribution."""
    return random.uniform(to_float(min_val), to_float(max_val))


def sample_loguniform(min_val, max_val) -> float:
    """Sample from log-uniform distribution."""
    log_min = math.log10(to_float(min_val))
    log_max = math.log10(to_float(max_val))
    return 10 ** random.uniform(log_min, log_max)


def sample_normal(min_val, max_val) -> float:
    """Sample from normal distribution (min/max as mean±2σ)."""
    min_f = to_float(min_val)
    max_f = to_float(max_val)
    mean = (min_f + max_f) / 2
    std = (max_f - min_f) / 4
    value = random.gauss(mean, std)
    return max(min_f, min(max_f, value))  # Clamp


def sample_discrete(min_val, max_val) -> int:
    """Sample integer from discrete uniform distribution."""
    return random.randint(to_int(min_val), to_int(max_val))


def sample_parameter(param_config: Dict[str, Any]) -> float:
    """Sample a parameter based on its configuration."""
    distribution = param_config.get("distribution", "uniform")
    min_val = param_config["min"]
    max_val = param_config["max"]
    
    if distribution == "uniform":
        return sample_uniform(min_val, max_val)
    elif distribution == "loguniform":
        return sample_loguniform(min_val, max_val)
    elif distribution == "normal":
        return sample_normal(min_val, max_val)
    elif distribution == "discrete":
        return sample_discrete(min_val, max_val)
    else:
        return sample_uniform(min_val, max_val)
# ============================================================
# till here
# ============================================================

# ============================================================
# ADVANCED SAMPLING: LHS, OPTIMAL LHS, SOBOL
# ============================================================
# 
# ARCHITECTURE:
#   1. build_parameter_dimensions() → identify what to sample
#   2. generate_unit_samples()      → fill [0,1]^D using chosen method
#   3. transform_sample()           → map [0,1] → physical value
#   4. generate_all_case_parameters() → orchestrate the above
#
# KEY INSIGHT: All space-filling methods work in the unit hypercube
# [0,1]^D. We generate points there first, then transform each
# dimension independently using the inverse CDF of the desired
# distribution. The space-filling properties are preserved because
# the inverse CDF is a monotonic transformation.
# ============================================================


def build_parameter_dimensions(config: StudyConfig) -> List[ParameterDimension]:
    """
    Scan the config and build the list of VARYING parameter dimensions.
    
    Fixed parameters (where min == max) are excluded because they
    contribute no variability — they're constants, not dimensions.
    
    This function answers: "How many axes does our design space have,
    and what does each axis represent?"
    
    Returns:
        List of ParameterDimension objects, one per varying parameter.
        The ORDER of this list defines the column order in the sample matrix.
    """
    dimensions = []
    ranges = config.param_ranges
    
    # ── Standard scalar parameters ──
    standard_params = [
        "udl", "youngs_modulus", "I22", "I33",
        "cross_area", "torsional_inertia", "subdivisions"
    ]
    
    for param_name in standard_params:
        if param_name not in ranges:
            continue
        
        p = ranges[param_name]
        min_val = to_float(p["min"])
        max_val = to_float(p["max"])
        distribution = p.get("distribution", "uniform")
        
        # Skip fixed parameters (no variation → not a dimension)
        if min_val == max_val:
            continue
        
        # Safety check: loguniform requires positive bounds
        if distribution == "loguniform" and (min_val <= 0 or max_val <= 0):
            print(f"  WARNING: {param_name} uses loguniform but has "
                  f"non-positive bounds [{min_val}, {max_val}]. "
                  f"Switching to uniform.")
            distribution = "uniform"
        
        dimensions.append(ParameterDimension(
            name=param_name,
            min_val=min_val,
            max_val=max_val,
            distribution=distribution
        ))
    
    # ── Response location (x, y, z as separate dimensions) ──
    if "response_location" in ranges:
        resp = ranges["response_location"]
        resp_dist = resp.get("distribution", "uniform")
        
        for axis in ["x", "y", "z"]:
            if axis not in resp:
                continue
            
            min_val = to_float(resp[axis]["min"])
            max_val = to_float(resp[axis]["max"])
            
            # Skip fixed axes (e.g., y: 0.0 to 0.0)
            if min_val == max_val:
                continue
            
            dimensions.append(ParameterDimension(
                name=f"response_{axis}",
                min_val=min_val,
                max_val=max_val,
                distribution=resp_dist
            ))
    
    return dimensions


def generate_unit_samples(
    n_samples: int,
    n_dimensions: int,
    method: str = "random",
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate an N×D matrix of samples in the unit hypercube [0,1]^D.
    
    This is the CORE function where the four methods differ.
    Everything downstream (transformation, case setup) is identical.
    
    Args:
        n_samples:    Number of points (rows)
        n_dimensions: Number of parameters (columns)
        method:       "random", "lhs", "optimal_lhs", or "sobol"
        seed:         For reproducibility
    
    Returns:
        np.ndarray of shape (n_samples, n_dimensions), values in [0, 1)
    
    ────────────────────────────────────────────────────────────
    METHOD DETAILS:
    
    "random" (Monte Carlo):
        Each entry is independently drawn from Uniform(0,1).
        Fast but can leave gaps and create clusters.
        Convergence rate: O(1/√N)
    
    "lhs" (Latin Hypercube):
        Divides [0,1] into N equal strata per dimension.
        Each stratum is sampled exactly once per dimension.
        Guarantees perfect 1D projections (no gaps in any single parameter).
        Column pairings are random → 2D+ projections may still cluster.
        Convergence rate: O(1/N) for smooth functions
    
    "optimal_lhs" (Optimized Latin Hypercube):
        Starts with LHS, then iteratively swaps row entries within 
        columns to minimize the Centered L2 Discrepancy.
        Better multi-dimensional space-filling than basic LHS.
        Slower to generate (optimization loop), but same evaluation cost.
        Uses scipy's "random-cd" optimization strategy.
    
    "sobol" (Scrambled Sobol Sequence):
        Deterministic low-discrepancy sequence based on binary fractions.
        Each new point fills the largest gap in the existing sequence.
        Best theoretical convergence: O(log(N)^D / N)
        Works best when N = 2^m (power of 2).
        Scrambling adds randomization for error estimation.
    ────────────────────────────────────────────────────────────
    """
    if n_dimensions == 0:
        return np.empty((n_samples, 0))
    
    if method == "random":
        # ── Monte Carlo: fully independent random samples ──
        rng = np.random.default_rng(seed)
        samples = rng.random((n_samples, n_dimensions))
    
    elif method == "lhs":
        # ── Latin Hypercube Sampling ──
        if HAS_SCIPY:
            sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)
            samples = sampler.random(n=n_samples)
        else:
            print("  WARNING: scipy not available, using numpy LHS fallback")
            samples = _lhs_fallback(n_samples, n_dimensions, seed)
    
    elif method == "optimal_lhs":
        # ── Optimized LHS (minimize centered L2 discrepancy) ──
        if not HAS_SCIPY:
            print("  WARNING: scipy not available, using numpy LHS fallback "
                  "(no optimization)")
            samples = _lhs_fallback(n_samples, n_dimensions, seed)
        else:
            try:
                # scipy >= 1.8 supports the optimization parameter
                # "random-cd" = random column swaps, minimize Centered Discrepancy
                sampler = qmc.LatinHypercube(
                    d=n_dimensions,
                    seed=seed,
                    optimization="random-cd"
                )
                samples = sampler.random(n=n_samples)
            except TypeError:
                # Older scipy: optimization parameter doesn't exist
                print("  WARNING: scipy version too old for optimization. "
                      "Upgrade: pip install --upgrade scipy")
                print("  Falling back to basic LHS.")
                sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)
                samples = sampler.random(n=n_samples)
    
    elif method == "sobol":
        # ── Scrambled Sobol Sequence ──
        if not HAS_SCIPY:
            raise ImportError(
                "scipy is required for Sobol sequences. "
                "Install with: pip install scipy"
            )
        
        # Warn if N is not a power of 2
        if n_samples > 0 and (n_samples & (n_samples - 1)) != 0:
            next_pow2 = 2 ** int(np.ceil(np.log2(max(n_samples, 1))))
            print(f"  ⚠ Sobol works best with 2^m samples.")
            print(f"    Requested: {n_samples}, "
                  f"nearest power of 2: {next_pow2}")
            print(f"    Consider using {next_pow2} samples for "
                  f"optimal space-filling properties.")
        
        # scramble=True adds Owen scrambling for randomization
        # This preserves low-discrepancy while allowing statistical
        # error estimation across different scrambles
        sampler = qmc.Sobol(d=n_dimensions, scramble=True, seed=seed)
        samples = sampler.random(n=n_samples)
    
    elif method == "maximin_lhs":
        # ── Maximin-optimized LHS ──
        # 
        # CONCEPT: Start with basic LHS (guaranteeing 1D stratification),
        # then iteratively swap entries within columns to MAXIMIZE the
        # MINIMUM pairwise distance between points.
        #
        # ALGORITHM (simulated annealing variant):
        #   1. Generate initial LHS
        #   2. Compute current minimum pairwise distance
        #   3. Repeat for many iterations:
        #      a. Pick a random column (dimension)
        #      b. Pick two random rows
        #      c. Swap their values in that column
        #      d. Recompute minimum distance
        #      e. If improved → keep swap
        #         If worse → revert (or accept with probability e^(-ΔE/T))
        #   4. Return best design found
        #
        # WHY NOT JUST USE scipy's optimization="random-cd"?
        # Because "random-cd" optimizes DISCREPANCY, not MAXIMIN.
        # They're different objectives that produce different designs.
        # Maximin tends to push points toward a more "lattice-like"
        # arrangement, while discrepancy optimization allows more
        # irregular spacing as long as the overall uniformity is good.
        #
        if HAS_SCIPY:
            samples = _maximin_lhs(n_samples, n_dimensions, seed)
        else:
            print("  WARNING: Using numpy LHS fallback (no optimization)")
            samples = _lhs_fallback(n_samples, n_dimensions, seed)

    else:
        raise ValueError(
            f"Unknown sampling method: '{method}'. "
            f"Valid options: 'random', 'lhs', 'optimal_lhs', 'sobol'"
        )
    
    return samples


def _lhs_fallback(
    n_samples: int,
    n_dimensions: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Basic LHS implementation using only numpy (no scipy needed).
    
    This is a FALLBACK for when scipy is not installed.
    It implements the core LHS algorithm:
    
    For each dimension j:
      1. Create N strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
      2. Generate a random permutation to assign strata to samples
      3. Within each assigned stratum, sample uniformly
    
    The permutation ensures each stratum has exactly one sample.
    """
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, n_dimensions))
    
    for j in range(n_dimensions):
        # Random permutation determines which stratum each sample gets
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            # perm[i] = stratum index for sample i
            # Sample uniformly within that stratum
            low = perm[i] / n_samples
            high = (perm[i] + 1) / n_samples
            samples[i, j] = rng.uniform(low, high)
    
    return samples

def _maximin_lhs(
    n_samples: int,
    n_dimensions: int,
    seed: Optional[int] = None,
    n_iterations: int = 1000,
    n_restarts: int = 5
) -> np.ndarray:
    """
    Generate a Maximin Latin Hypercube Design.
    
    ALGORITHM:
    ──────────
    We use a multi-restart local search approach:
    
    1. Generate multiple random LHS designs (restarts)
    2. For each, run an optimization loop:
       - Randomly select a column (dimension)
       - Randomly select two rows
       - Swap their values in that column
         (This preserves the LHS property! Each column still
          has exactly one sample per stratum)
       - Accept the swap if it increases the minimum distance
    3. Return the best design across all restarts
    
    WHY SWAPS PRESERVE LHS PROPERTY:
    ────────────────────────────────
    An LHS design has each column being a permutation of the 
    strata {1, 2, ..., N}. Swapping two entries in a column 
    gives another permutation — still a valid LHS.
    
    Before swap (column 2):     After swap (column 2):
    Row 1: [0.1, 0.7, ...]     Row 1: [0.1, 0.3, ...]  ← swapped
    Row 2: [0.5, 0.3, ...]     Row 2: [0.5, 0.7, ...]  ← swapped
    Row 3: [0.9, 0.1, ...]     Row 3: [0.9, 0.1, ...]
    
    Both are valid LHS designs (each column has one value per stratum).
    But the pairwise distances between rows have changed.
    
    COMPUTATIONAL COST:
    ──────────────────
    - Initial LHS: O(N × D)
    - Each swap iteration: O(N) to recompute distances to swapped rows
    - Total: O(n_restarts × n_iterations × N)
    
    For N=100, D=7, this takes < 1 second.
    For N=1000, D=20, consider reducing n_iterations.
    
    Parameters:
        n_samples:    Number of points (N)
        n_dimensions: Number of dimensions (D)  
        seed:         Random seed
        n_iterations: Swap attempts per restart
        n_restarts:   Number of independent starting designs
    
    Returns:
        np.ndarray of shape (N, D), values in [0, 1)
    """
    rng = np.random.default_rng(seed)
    
    best_design = None
    best_mindist = -1.0
    
    # Scale iterations with problem size
    # More points/dimensions → need more iterations to explore
    scaled_iterations = max(
        n_iterations,
        n_samples * n_dimensions * 5
    )
    # But cap it to keep runtime reasonable
    scaled_iterations = min(scaled_iterations, 50000)
    
    for restart in range(n_restarts):
        # ── Generate initial LHS ──
        design = np.zeros((n_samples, n_dimensions))
        for j in range(n_dimensions):
            perm = rng.permutation(n_samples)
            for i in range(n_samples):
                low = perm[i] / n_samples
                high = (perm[i] + 1) / n_samples
                design[i, j] = rng.uniform(low, high)
        
        # ── Compute initial pairwise distances ──
        # We store the FULL distance matrix for efficient updates
        # dist_matrix[i,j] = ||x_i - x_j||²  (squared, for speed)
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d2 = np.sum((design[i] - design[j]) ** 2)
                dist_matrix[i, j] = d2
                dist_matrix[j, i] = d2
        
        # Current minimum distance (excluding diagonal zeros)
        np.fill_diagonal(dist_matrix, np.inf)
        current_mindist = np.sqrt(np.min(dist_matrix))
        np.fill_diagonal(dist_matrix, 0.0)
        
        # ── Optimization loop ──
        improvements = 0
        
        for iteration in range(scaled_iterations):
            # Pick random column and two random rows
            col = rng.integers(0, n_dimensions)
            row1, row2 = rng.choice(n_samples, size=2, replace=False)
            
            # ── Compute new distances if we swap ──
            # Only rows row1 and row2 change, so we only need to 
            # recompute distances involving these two rows.
            # This is O(N) instead of O(N²).
            
            # Save old values
            old_val1 = design[row1, col]
            old_val2 = design[row2, col]
            
            # Perform swap
            design[row1, col] = old_val2
            design[row2, col] = old_val1
            
            # Recompute distances for row1 and row2
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
            
            # Also check distances not involving row1 or row2
            # (they haven't changed, but might still be the minimum)
            for i in range(n_samples):
                if i == row1 or i == row2:
                    continue
                for j in range(i + 1, n_samples):
                    if j == row1 or j == row2:
                        continue
                    new_min_d2 = min(new_min_d2, dist_matrix[i, j])
            
            new_mindist = np.sqrt(new_min_d2)
            
            if new_mindist > current_mindist:
                # ── Accept the swap ──
                current_mindist = new_mindist
                improvements += 1
            else:
                # ── Reject: revert swap and distance matrix ──
                design[row1, col] = old_val1
                design[row2, col] = old_val2
                dist_matrix[row1, :] = old_dists_row1
                dist_matrix[:, row1] = old_dists_row1
                dist_matrix[row2, :] = old_dists_row2
                dist_matrix[:, row2] = old_dists_row2
        
        # ── Check if this restart is the best so far ──
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
    """
    Transform a [0,1] sample to the actual parameter value.
    
    This is the INVERSE CDF (quantile function) for each distribution.
    
    ────────────────────────────────────────────────────────────
    WHY THIS WORKS (Probability Integral Transform):
    
    If U ~ Uniform(0,1), then F⁻¹(U) has distribution F.
    
    This means: no matter how we generate U (random, LHS, Sobol),
    applying the inverse CDF gives us the correct distribution.
    The space-filling properties in [0,1] map to space-filling
    properties in the parameter space because inverse CDFs are
    monotonic (order-preserving) transformations.
    
    VISUAL for loguniform:
    
    u ∈ [0,1]:    |-------|-------|-------|-------|
                  0      0.25    0.5    0.75     1
                                ↓ transform
    value:        |--|---|------|------------|
                 1e-5  1e-4   1e-3         1e-2
    
    Equal spacing in u → equal spacing in log-space → concentrates
    more samples where the parameter is small (appropriate for
    parameters spanning orders of magnitude).
    ────────────────────────────────────────────────────────────
    
    Args:
        u:   Value in [0, 1) from the unit hypercube
        dim: Parameter dimension specification
    
    Returns:
        Transformed value in [dim.min_val, dim.max_val]
    """
    if dim.distribution == "uniform":
        # F⁻¹(u) = min + u × (max - min)
        return dim.min_val + u * (dim.max_val - dim.min_val)
    
    elif dim.distribution == "loguniform":
        # F⁻¹(u) = 10^(log₁₀(min) + u × (log₁₀(max) - log₁₀(min)))
        log_min = math.log10(dim.min_val)
        log_max = math.log10(dim.max_val)
        return 10 ** (log_min + u * (log_max - log_min))
    
    elif dim.distribution == "normal":
        # F⁻¹(u) = μ + σ × Φ⁻¹(u)
        # We interpret min/max as mean ± 2σ (95% interval)
        mean = (dim.min_val + dim.max_val) / 2
        std = (dim.max_val - dim.min_val) / 4
        
        # Use Python's built-in NormalDist (Python 3.8+)
        from statistics import NormalDist
        u_clipped = max(1e-10, min(1 - 1e-10, u))  # Avoid ±∞
        value = NormalDist(mu=mean, sigma=std).inv_cdf(u_clipped)
        
        # Clamp to declared range
        return max(dim.min_val, min(dim.max_val, value))
    
    elif dim.distribution == "discrete":
        # Map [0,1) uniformly to {min, min+1, ..., max}
        n_values = int(dim.max_val - dim.min_val) + 1
        idx = min(int(u * n_values), n_values - 1)  # Handle u=1.0
        return int(dim.min_val + idx)
    
    else:
        # Fallback: treat as uniform
        print(f"  WARNING: Unknown distribution '{dim.distribution}' "
              f"for {dim.name}. Using uniform.")
        return dim.min_val + u * (dim.max_val - dim.min_val)


def generate_all_case_parameters(
    config: StudyConfig,
    start_case: int = 1
) -> Tuple[List[CaseParameters], Optional[np.ndarray], List[ParameterDimension]]:
    """
    Generate ALL case parameters at once using the configured sampling method.
    
    This REPLACES the old per-case generation loop. The key difference:
    
    OLD (Monte Carlo only):
        for i in range(N):
            params[i] = sample_each_parameter_independently()
    
    NEW (any method):
        dimensions = identify_varying_parameters()          # Step 1
        unit_samples = fill_unit_hypercube(method, N, D)    # Step 2  
        for i in range(N):
            params[i] = transform_row(unit_samples[i])      # Step 3
    
    Returns:
        Tuple of:
          - List[CaseParameters]: one per case, ready for setup_case()
          - np.ndarray or None: the raw unit samples (for quality analysis)
          - List[ParameterDimension]: dimension definitions (for reporting)
    """
    # ── Step 1: Identify varying dimensions ──
    dimensions = build_parameter_dimensions(config)
    n_dims = len(dimensions)
    n_samples = config.num_cases
    ranges = config.param_ranges
    
    # ── Print sampling info ──
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
    fixed_params_info = []
    all_scalar = ["udl", "youngs_modulus", "I22", "I33",
                  "cross_area", "torsional_inertia", "subdivisions"]
    for p_name in all_scalar:
        if p_name in ranges:
            if to_float(ranges[p_name]["min"]) == to_float(ranges[p_name]["max"]):
                fixed_params_info.append(
                    f"    {p_name}: {to_float(ranges[p_name]['min']):.4e}"
                )
    if "response_location" in ranges:
        resp = ranges["response_location"]
        for axis in ["x", "y", "z"]:
            if axis in resp:
                if to_float(resp[axis]["min"]) == to_float(resp[axis]["max"]):
                    fixed_params_info.append(
                        f"    response_{axis}: "
                        f"{to_float(resp[axis]['min']):.4e}"
                    )
    if fixed_params_info:
        print(f"  Fixed parameters:")
        for info in fixed_params_info:
            print(info)
    
    # ── Step 2: Generate unit samples ──
    if n_dims > 0:
        print(f"\n  Generating {n_samples} × {n_dims} unit samples "
              f"using {config.sampling_method}...")
        
        unit_samples = generate_unit_samples(
            n_samples=n_samples,
            n_dimensions=n_dims,
            method=config.sampling_method,
            seed=config.random_seed
        )
        
        # ── Quality metrics ──
        if n_samples > 1 and n_dims >= 1:
            if HAS_SCIPY:
                disc = qmc.discrepancy(unit_samples)
                print(f"  Centered L2 Discrepancy: {disc:.6e}")
                
                distances = pdist(unit_samples)
                print(f"  Min point-to-point distance:  {np.min(distances):.6f}")
                print(f"  Mean point-to-point distance: {np.mean(distances):.6f}")
            else:
                print("  (Install scipy for quality metrics)")
        
        print(f"  ✓ Unit samples generated")
        
        # ── Save unit samples to CSV ──
        samples_path = os.path.join(
            config.working_directory,
            f"unit_samples_{config.sampling_method}.csv"
        )
        os.makedirs(config.working_directory, exist_ok=True)
        header = ",".join(dim.name for dim in dimensions)
        np.savetxt(
            samples_path, unit_samples,
            delimiter=",", header=header, comments=""
        )
        print(f"  Saved unit samples to: {samples_path}")
    else:
        unit_samples = None
        print(f"\n  No varying dimensions — "
              f"all {n_samples} cases will be identical.")
    
    # ── Step 3: Transform to physical parameters ──
    all_params = []
    
    for i in range(n_samples):
        case_id = start_case + i
        
        # Start with fixed/default values for ALL parameters
        param_values = {}
        for p_name in all_scalar:
            if p_name in ranges:
                param_values[p_name] = to_float(ranges[p_name]["min"])
        
        # Response coord defaults
        if "response_location" in ranges:
            resp = ranges["response_location"]
            for axis in ["x", "y", "z"]:
                if axis in resp:
                    param_values[f"response_{axis}"] = \
                        to_float(resp[axis]["min"])
        
        # Override varying parameters with transformed samples
        if unit_samples is not None:
            for j, dim in enumerate(dimensions):
                param_values[dim.name] = transform_sample(
                    unit_samples[i, j], dim
                )
        
        # Build response coordinates tuple
        response_coords = (
            param_values.get("response_x", 0.0),
            param_values.get("response_y", 0.0),
            param_values.get("response_z", 0.0),
        )
        
        all_params.append(CaseParameters(
            case_id=case_id,
            udl=param_values["udl"],
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

def modify_materials_json(
    template_path: str,
    output_path: str,
    params: CaseParameters,
    fixed: Dict[str, Any]
) -> None:
    """Modify materials JSON with sampled parameters."""
    with open(template_path, 'r') as f:
        materials = json.load(f)
    
    # Update material variables
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


def modify_primal_json(
    template_path: str,
    output_path: str,
    params: CaseParameters,
    mdpa_filename: str,
    materials_filename: str,
    vtk_output_path: str,
    fixed: Dict[str, Any]
) -> None:
    """Modify primal parameters JSON."""
    with open(template_path, 'r') as f:
        primal = json.load(f)
    
    # Update file references
    primal["solver_settings"]["model_import_settings"]["input_filename"] = mdpa_filename
    primal["solver_settings"]["material_import_settings"]["materials_filename"] = materials_filename
    
    # Update load (UDL)
    for load_process in primal["processes"]["loads_process_list"]:
        if "LINE_LOAD" in load_process["Parameters"].get("variable_name", ""):
            load_process["Parameters"]["modulus"] = params.udl
    
    # Update gravity if present
    for load_process in primal["processes"]["loads_process_list"]:
        if "VOLUME_ACCELERATION" in load_process["Parameters"].get("variable_name", ""):
            gravity = fixed.get("gravity", [0.0, 0.0, -9.81])
            if isinstance(gravity, list):
                load_process["Parameters"]["value"] = gravity
    
    # Update VTK output path
    if "output_processes" in primal:
        for vtk in primal.get("output_processes", {}).get("vtk_output", []):
            vtk["Parameters"]["output_path"] = vtk_output_path
    
    with open(output_path, 'w') as f:
        json.dump(primal, f, indent=4)


def modify_adjoint_json(
    template_path: str,
    output_path: str,
    params: CaseParameters,
    mdpa_filename: str,
    materials_filename: str,
    vtk_output_path: str,
    response_settings: Dict[str, Any]
) -> None:
    """Modify adjoint parameters JSON."""
    with open(template_path, 'r') as f:
        adjoint = json.load(f)
    
    # Update file references
    adjoint["solver_settings"]["model_import_settings"]["input_filename"] = mdpa_filename
    adjoint["solver_settings"]["material_import_settings"]["materials_filename"] = materials_filename
    
    # Update response function settings
    resp = adjoint["solver_settings"]["response_function_settings"]
    resp["traced_element_id"] = params.traced_element_id
    resp["stress_location"] = params.stress_location
    resp["stress_type"] = response_settings.get("stress_type", "MY")
    resp["stress_treatment"] = response_settings.get("stress_treatment", "node")
    
    # Update load (UDL) - must match primal
    for load_process in adjoint["processes"]["loads_process_list"]:
        if "LINE_LOAD" in load_process["Parameters"].get("variable_name", ""):
            load_process["Parameters"]["modulus"] = params.udl
    
    # Update VTK output path
    if "output_processes" in adjoint:
        for vtk in adjoint.get("output_processes", {}).get("vtk_output", []):
            vtk["Parameters"]["output_path"] = vtk_output_path
    
    with open(output_path, 'w') as f:
        json.dump(adjoint, f, indent=4)


# ============================================================
# CASE SETUP FUNCTION
# ============================================================

def setup_case(
    case_id: int,
    params: CaseParameters,
    config: StudyConfig,
    dry_run: bool = False
) -> Tuple[str, str]:
    """
    Setup a single case: create folders, modify files.
    
    Returns:
        (primal_case_folder, adjoint_case_folder)
    """
    # Create folder names
    primal_folder = os.path.join(
        config.working_directory,
        config.primal_folder,
        f"{config.primal_prefix}_{case_id}"
    )
    adjoint_folder = os.path.join(
        config.working_directory,
        config.adjoint_folder,
        f"{config.adjoint_prefix}_{case_id}"
    )
    
    # Create folders
    os.makedirs(primal_folder, exist_ok=True)
    os.makedirs(adjoint_folder, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Setting up Case {case_id}")
    print(f"{'='*60}")
    print(f"  UDL: {params.udl:.2f}")
    print(f"  Young's Modulus: {params.youngs_modulus:.4e}")
    print(f"  I22: {params.I22:.4e}")
    print(f"  I33: {params.I33:.4e}")
    print(f"  Cross Area: {params.cross_area:.6f}")
    print(f"  Torsional Inertia: {params.torsional_inertia:.4e}")
    print(f"  Subdivisions: {params.subdivisions}")
    print(f"  Response coords: ({params.response_coords[0]:.2f}, {params.response_coords[1]:.2f}, {params.response_coords[2]:.2f})")
    
    if dry_run:
        print("  [DRY RUN - Skipping file operations]")
        return primal_folder, adjoint_folder
    
    # --------------------------------------------------------
    # Step 1: Refine MDPA and copy to both folders
    # --------------------------------------------------------
    template_mdpa = os.path.join(config.working_directory, config.template_mdpa)
    
    if not os.path.exists(template_mdpa):
        print(f"  ERROR: Template MDPA not found: {template_mdpa}")
        raise FileNotFoundError(f"Template MDPA not found: {template_mdpa}")
    
    # Parse and refine
    print(f"  Reading template MDPA: {template_mdpa}")
    mdpa_data = parse_mdpa(template_mdpa)
    
    if params.subdivisions > 1:
        print(f"  Refining mesh with {params.subdivisions} subdivisions...")
        refined_data = refine_mesh(mdpa_data, params.subdivisions)
        print(f"  Refined: {len(mdpa_data.nodes)} -> {len(refined_data.nodes)} nodes")
    else:
        refined_data = mdpa_data
        print(f"  No refinement needed (subdivisions=1)")
    
    # Write refined MDPA to both folders
    mdpa_basename = "Frame_structure_refined"
    primal_mdpa = os.path.join(primal_folder, f"{mdpa_basename}.mdpa")
    adjoint_mdpa = os.path.join(adjoint_folder, f"{mdpa_basename}.mdpa")
    
    write_mdpa(refined_data, primal_mdpa)
    shutil.copy(primal_mdpa, adjoint_mdpa)
    
    # --------------------------------------------------------
    # Step 2: Find nearest element for response
    # --------------------------------------------------------
    elem_id, stress_loc, node_id, dist = find_nearest_element_and_location(
        params.response_coords, primal_mdpa
    )
    params.traced_element_id = elem_id
    params.stress_location = stress_loc
    params.nearest_node_id = node_id
    params.nearest_node_distance = dist
    
    print(f"  Response location:")
    print(f"    Target coords: {params.response_coords}")
    print(f"    Nearest node: {node_id} (distance: {dist:.4f})")
    print(f"    Traced element: {elem_id}")
    print(f"    Stress location: {stress_loc}")
    
    # --------------------------------------------------------
    # Step 3: Create materials JSON in both folders
    # --------------------------------------------------------
    template_materials = os.path.join(config.working_directory, config.template_materials)
    
    if not os.path.exists(template_materials):
        print(f"  ERROR: Template materials not found: {template_materials}")
        raise FileNotFoundError(f"Template materials not found: {template_materials}")
    
    primal_materials = os.path.join(primal_folder, "materials_beam.json")
    adjoint_materials = os.path.join(adjoint_folder, "materials_beam.json")
    
    modify_materials_json(template_materials, primal_materials, params, config.fixed_params)
    shutil.copy(primal_materials, adjoint_materials)
    print(f"  Created materials_beam.json")
    
    # --------------------------------------------------------
    # Step 4: Create primal parameters JSON
    # --------------------------------------------------------
    template_primal = os.path.join(config.working_directory, config.template_primal)
    
    if not os.path.exists(template_primal):
        print(f"  ERROR: Template primal not found: {template_primal}")
        raise FileNotFoundError(f"Template primal not found: {template_primal}")
    
    primal_params_json = os.path.join(primal_folder, "beam_test_parameters.json")
    
    modify_primal_json(
        template_primal,
        primal_params_json,
        params,
        mdpa_basename,
        "materials_beam.json",
        "vtk_output_primal",
        config.fixed_params
    )
    print(f"  Created beam_test_parameters.json")
    
    # --------------------------------------------------------
    # Step 5: Create adjoint parameters JSON
    # --------------------------------------------------------
    template_adjoint = os.path.join(config.working_directory, config.template_adjoint)
    
    if not os.path.exists(template_adjoint):
        print(f"  ERROR: Template adjoint not found: {template_adjoint}")
        raise FileNotFoundError(f"Template adjoint not found: {template_adjoint}")
    
    adjoint_params_json = os.path.join(adjoint_folder, "beam_test_local_stress_adjoint_parameters.json")
    
    modify_adjoint_json(
        template_adjoint,
        adjoint_params_json,
        params,
        mdpa_basename,
        "materials_beam.json",
        "vtk_output_adjoint",
        config.response_settings
    )
    print(f"  Created beam_test_local_stress_adjoint_parameters.json")
    
    # Copy primal JSON to adjoint folder (needed for HDF5 reading)
    shutil.copy(primal_params_json, os.path.join(adjoint_folder, "beam_test_parameters.json"))
    
    # --------------------------------------------------------
    # Step 6: Save case config
    # --------------------------------------------------------
    case_config = {
        "case_id": case_id,
        "parameters": {
            "udl": params.udl,
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
    print(f"  Saved case_config.json")
    
    return primal_folder, adjoint_folder


# ============================================================
# KRATOS RUNNER
# ============================================================

def run_kratos_analysis(case_folder: str, analysis_type: str = "primal") -> bool:
    """
    Run Kratos analysis in the specified folder.
    
    Returns:
        True if successful, False otherwise
    """
    original_dir = os.getcwd()
    
    try:
        os.chdir(case_folder)
        print(f"\n  Running {analysis_type} analysis in: {case_folder}")
        
        # Import Kratos
        import KratosMultiphysics
        import KratosMultiphysics.StructuralMechanicsApplication as SMA
        from KratosMultiphysics.StructuralMechanicsApplication import structural_mechanics_analysis
        
        # Select parameter file
        if analysis_type == "primal":
            param_file = "beam_test_parameters.json"
        else:
            param_file = "beam_test_local_stress_adjoint_parameters.json"
        
        # Load parameters
        with open(param_file, 'r') as f:
            parameters = KratosMultiphysics.Parameters(f.read())
        
        # Create model
        model = KratosMultiphysics.Model()
        
        if analysis_type == "primal":
            # ============================================================
            # CUSTOM PRIMAL ANALYSIS WITH LOAD TRANSFER
            # ============================================================
            class CustomPrimalAnalysis(structural_mechanics_analysis.StructuralMechanicsAnalysis):
                """Custom analysis that transfers loads to nodes for VTK output."""
                
                def Initialize(self):
                    """Initialize and transfer material properties to elements."""
                    super().Initialize()
                    
                    # Get model part
                    model_part_name = self.project_parameters["solver_settings"]["model_part_name"].GetString()
                    mp = self.model.GetModelPart(model_part_name)
                    
                    # Transfer material properties to elements
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
                    
                    print(f"    Transferred material properties to {mp.NumberOfElements()} elements")
                
                def FinalizeSolutionStep(self):
                    """Transfer loads from conditions to nodes for VTK output."""
                    
                    # Get model part
                    model_part_name = self.project_parameters["solver_settings"]["model_part_name"].GetString()
                    mp = self.model.GetModelPart(model_part_name)
                    
                    # Define load variables
                    LINE_LOAD = SMA.LINE_LOAD
                    POINT_LOAD = SMA.POINT_LOAD
                    POINT_MOMENT = SMA.POINT_MOMENT
                    
                    # Initialize all nodes to zero
                    zero = KratosMultiphysics.Array3([0.0, 0.0, 0.0])
                    for node in mp.Nodes:
                        node.SetValue(LINE_LOAD, zero)
                        node.SetValue(POINT_LOAD, zero)
                        node.SetValue(POINT_MOMENT, zero)
                    
                    # ------------------------------------------------
                    # Transfer LINE_LOAD from conditions to nodes
                    # ------------------------------------------------
                    line_load_count = 0
                    try:
                        load_mp = mp.GetSubModelPart("LineLoad3D_load")
                        for condition in load_mp.Conditions:
                            if condition.Has(LINE_LOAD):
                                load_val = condition.GetValue(LINE_LOAD)
                                for node in condition.GetGeometry():
                                    node.SetValue(LINE_LOAD, load_val)
                                    line_load_count += 1
                        print(f"    Transferred LINE_LOAD to {line_load_count} node instances")
                    except Exception as e:
                        # Fallback: check all conditions
                        for condition in mp.Conditions:
                            if condition.Has(LINE_LOAD):
                                load_val = condition.GetValue(LINE_LOAD)
                                # Check if non-zero
                                if abs(load_val[0]) > 1e-12 or abs(load_val[1]) > 1e-12 or abs(load_val[2]) > 1e-12:
                                    for node in condition.GetGeometry():
                                        node.SetValue(LINE_LOAD, load_val)
                                        line_load_count += 1
                        if line_load_count > 0:
                            print(f"    Transferred LINE_LOAD to {line_load_count} node instances (fallback)")
                    
                    # ------------------------------------------------
                    # Transfer POINT_LOAD from conditions to nodes
                    # ------------------------------------------------
                    point_load_count = 0
                    try:
                        load_mp = mp.GetSubModelPart("PointLoad3D_load")
                        for condition in load_mp.Conditions:
                            if condition.Has(POINT_LOAD):
                                load_val = condition.GetValue(POINT_LOAD)
                                for node in condition.GetGeometry():
                                    node.SetValue(POINT_LOAD, load_val)
                                    point_load_count += 1
                        if point_load_count > 0:
                            print(f"    Transferred POINT_LOAD to {point_load_count} node instances")
                    except:
                        pass
                    
                    # ------------------------------------------------
                    # Transfer POINT_MOMENT from conditions to nodes
                    # ------------------------------------------------
                    moment_count = 0
                    moment_submodelpart_names = [
                        "PointMoment3D_moment",
                        "PointMoment3D_load",
                        "PointMoment3D",
                        "Moment3D_moment",
                        "PointMoment"
                    ]
                    
                    for submp_name in moment_submodelpart_names:
                        try:
                            moment_mp = mp.GetSubModelPart(submp_name)
                            for condition in moment_mp.Conditions:
                                if condition.Has(POINT_MOMENT):
                                    moment_val = condition.GetValue(POINT_MOMENT)
                                    for node in condition.GetGeometry():
                                        node.SetValue(POINT_MOMENT, moment_val)
                                        moment_count += 1
                            if moment_count > 0:
                                print(f"    Transferred POINT_MOMENT to {moment_count} node instances")
                            break
                        except:
                            continue
                    
                    # Alternative: search all conditions if no submodelpart found
                    if moment_count == 0:
                        for condition in mp.Conditions:
                            if condition.Has(POINT_MOMENT):
                                moment_val = condition.GetValue(POINT_MOMENT)
                                if abs(moment_val[0]) > 1e-12 or abs(moment_val[1]) > 1e-12 or abs(moment_val[2]) > 1e-12:
                                    for node in condition.GetGeometry():
                                        node.SetValue(POINT_MOMENT, moment_val)
                                        moment_count += 1
                        if moment_count > 0:
                            print(f"    Transferred POINT_MOMENT to {moment_count} node instances (fallback)")
                    
                    # Call parent AFTER transfer (triggers VTK output)
                    super().FinalizeSolutionStep()
            
            # Create and run custom analysis
            analysis = CustomPrimalAnalysis(model, parameters)
        else:
            # Standard adjoint analysis
            analysis = structural_mechanics_analysis.StructuralMechanicsAnalysis(model, parameters)
        
        # Run the analysis
        analysis.Run()
        
        print(f"  ✓ {analysis_type.capitalize()} analysis completed successfully")
        return True
        
    except Exception as e:
        import traceback
        print(f"  ✗ {analysis_type.capitalize()} analysis failed: {e}")
        traceback.print_exc()
        return False
        
    finally:
        os.chdir(original_dir)


# ============================================================
# CONFIG LOADING
# ============================================================

def load_config(config_path: str) -> StudyConfig:
    """Load configuration from YAML or JSON file."""
    
    # If config_path is just a filename, look in script directory
    if not os.path.isabs(config_path):
        config_in_script_dir = os.path.join(script_dir, config_path)
        
        if os.path.exists(config_in_script_dir):
            config_path = config_in_script_dir
        elif not os.path.exists(config_path):
            print(f"ERROR: Config file not found!")
            print(f"  Looked in current directory: {os.getcwd()}")
            print(f"  Looked in script directory: {script_dir}")
            print(f"\nPlease create config.yaml or config.json, or specify path with --config")
            sys.exit(1)
    
    print(f"Loading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            if HAS_YAML:
                cfg = yaml.safe_load(f)
            else:
                print("ERROR: PyYAML required for .yaml files.")
                print("Install with: pip install pyyaml")
                print("Or use config.json instead")
                sys.exit(1)
        else:
            cfg = json.load(f)
    
    # Validate required keys
    required_keys = ["general", "output", "templates", "parameters"]
    for key in required_keys:
        if key not in cfg:
            print(f"ERROR: Missing required key '{key}' in config file")
            print(f"Available keys: {list(cfg.keys())}")
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
        sampling_method=cfg["general"].get("sampling_method", "random")
    )


# ============================================================
# PARAMETER GENERATION
# ============================================================

# ============================================================
# can delete or use as Utility
# ============================================================
def generate_case_parameters(case_id: int, config: StudyConfig) -> CaseParameters:
    """Generate random parameters for a case."""
    
    ranges = config.param_ranges
    
    # Sample response location
    resp_loc = ranges["response_location"]
    response_coords = (
        sample_parameter({
            "min": resp_loc["x"]["min"], 
            "max": resp_loc["x"]["max"], 
            "distribution": resp_loc.get("distribution", "uniform")
        }),
        sample_parameter({
            "min": resp_loc["y"]["min"], 
            "max": resp_loc["y"]["max"], 
            "distribution": resp_loc.get("distribution", "uniform")
        }),
        sample_parameter({
            "min": resp_loc["z"]["min"], 
            "max": resp_loc["z"]["max"], 
            "distribution": resp_loc.get("distribution", "uniform")
        })
    )
    
    return CaseParameters(
        case_id=case_id,
        udl=sample_parameter(ranges["udl"]),
        youngs_modulus=sample_parameter(ranges["youngs_modulus"]),
        I22=sample_parameter(ranges["I22"]),
        I33=sample_parameter(ranges["I33"]),
        cross_area=sample_parameter(ranges["cross_area"]),
        torsional_inertia=sample_parameter(ranges["torsional_inertia"]),
        subdivisions=int(sample_parameter(ranges["subdivisions"])),
        response_coords=response_coords
    )
# ============================================================
# till here
# ============================================================

# ============================================================
# MAIN FUNCTION
# ============================================================

# def main():
#     parser = argparse.ArgumentParser(description="Run parametric study for Kratos")
#     parser.add_argument("--config", default="config.yaml", help="Configuration file path")
#     parser.add_argument("--dry-run", action="store_true", help="Setup only, don't create files")
#     parser.add_argument("--setup-only", action="store_true", help="Create files but don't run Kratos")
#     parser.add_argument("--start-case", type=int, default=1, help="Starting case number")
#     parser.add_argument("--run-existing", type=int, nargs='+', help="Run specific existing cases")
#     args = parser.parse_args()
    
#     print("=" * 60)
#     print("KRATOS PARAMETRIC STUDY RUNNER")
#     print("=" * 60)
#     print(f"Config file: {args.config}")
#     print(f"Dry run: {args.dry_run}")
#     print(f"Setup only: {args.setup_only}")
#     print(f"Start case: {args.start_case}")
    
#     # Load configuration
#     config = load_config(args.config)
    
#     # Set random seed
#     if config.random_seed is not None:
#         random.seed(config.random_seed)
#         print(f"Random seed: {config.random_seed}")
#     else:
#         print("Random seed: None (truly random)")
    
#     print(f"Working directory: {config.working_directory}")
#     print(f"Number of cases: {config.num_cases}")
    
#     # Create output directories
#     os.makedirs(os.path.join(config.working_directory, config.primal_folder), exist_ok=True)
#     os.makedirs(os.path.join(config.working_directory, config.adjoint_folder), exist_ok=True)
    
#     # Track all cases
#     all_cases: List[Tuple[int, str, str, CaseParameters]] = []
    
#     # Generate and setup cases
#     for i in range(config.num_cases):
#         case_id = args.start_case + i
        
#         # Generate parameters
#         params = generate_case_parameters(case_id, config)
        
#         # Setup case (create folders and files)
#         try:
#             primal_folder, adjoint_folder = setup_case(
#                 case_id, params, config, dry_run=args.dry_run
#             )
#             all_cases.append((case_id, primal_folder, adjoint_folder, params))
#         except Exception as e:
#             print(f"  ERROR setting up case {case_id}: {e}")
#             import traceback
#             traceback.print_exc()
#             continue
    
#     # Save study summary
#     if not args.dry_run:
#         summary = {
#             "timestamp": datetime.now().isoformat(),
#             "config_file": args.config,
#             "num_cases": config.num_cases,
#             "random_seed": config.random_seed,
#             "cases": [
#                 {
#                     "id": c[0], 
#                     "primal": c[1], 
#                     "adjoint": c[2],
#                     "parameters": {
#                         "udl": c[3].udl,
#                         "youngs_modulus": c[3].youngs_modulus,
#                         "I22": c[3].I22,
#                         "I33": c[3].I33,
#                         "cross_area": c[3].cross_area,
#                         "torsional_inertia": c[3].torsional_inertia,
#                         "subdivisions": c[3].subdivisions,
#                         "response_coords": list(c[3].response_coords),
#                         "traced_element_id": c[3].traced_element_id,
#                         "stress_location": c[3].stress_location
#                     }
#                 } 
#                 for c in all_cases
#             ]
#         }
        
#         summary_path = os.path.join(config.working_directory, "study_summary.json")
#         with open(summary_path, 'w') as f:
#             json.dump(summary, f, indent=4)
#         print(f"\nStudy summary saved to: {summary_path}")
    
#     # Run analyses if not dry-run or setup-only
#     if not args.dry_run and not args.setup_only:
#         print("\n" + "=" * 60)
#         print("RUNNING ANALYSES")
#         print("=" * 60)
        
#         results = []
        
#         for case_id, primal_folder, adjoint_folder, params in all_cases:
#             print(f"\n{'>'*20} Case {case_id} {'<'*20}")
            
#             # Run primal
#             primal_success = run_kratos_analysis(primal_folder, "primal")
            
#             # Run adjoint only if primal succeeded
#             if primal_success:
#                 # Copy HDF5 files to adjoint folder
#                 h5_copied = 0
#                 for f in os.listdir(primal_folder):
#                     if f.endswith('.h5'):
#                         shutil.copy(
#                             os.path.join(primal_folder, f),
#                             os.path.join(adjoint_folder, f)
#                         )
#                         h5_copied += 1
#                 if h5_copied > 0:
#                     print(f"  Copied {h5_copied} HDF5 file(s) to adjoint folder")
                
#                 adjoint_success = run_kratos_analysis(adjoint_folder, "adjoint")
#             else:
#                 adjoint_success = False
#                 print("  Skipping adjoint (primal failed)")
            
#             results.append({
#                 "case_id": case_id,
#                 "primal_success": primal_success,
#                 "adjoint_success": adjoint_success
#             })
        
#         # Save results summary
#         results_path = os.path.join(config.working_directory, "results_summary.json")
#         with open(results_path, 'w') as f:
#             json.dump(results, f, indent=4)
        
#         # Print summary
#         print("\n" + "=" * 60)
#         print("RESULTS SUMMARY")
#         print("=" * 60)
#         successful = sum(1 for r in results if r["primal_success"] and r["adjoint_success"])
#         primal_only = sum(1 for r in results if r["primal_success"] and not r["adjoint_success"])
#         failed = sum(1 for r in results if not r["primal_success"])
        
#         print(f"  Fully successful: {successful}/{len(results)}")
#         print(f"  Primal only: {primal_only}/{len(results)}")
#         print(f"  Failed: {failed}/{len(results)}")
#         print()
        
#         for r in results:
#             if r["primal_success"] and r["adjoint_success"]:
#                 status = "✓ PASS"
#             elif r["primal_success"]:
#                 status = "◐ PRIMAL OK"
#             else:
#                 status = "✗ FAIL"
#             print(f"  Case {r['case_id']}: {status}")
    
#     print("\n" + "=" * 60)
#     print("PARAMETRIC STUDY COMPLETE")
#     print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run parametric study for Kratos"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Setup only, don't create files"
    )
    parser.add_argument(
        "--setup-only", action="store_true",
        help="Create files but don't run Kratos"
    )
    parser.add_argument(
        "--start-case", type=int, default=1,
        help="Starting case number"
    )
    parser.add_argument(
        "--run-existing", type=int, nargs='+',
        help="Run specific existing cases"
    )
    # ── NEW ARGUMENT ──
    parser.add_argument(
        "--method",
        choices=["random", "lhs", "optimal_lhs", "sobol"],
        default=None,
        help="Override sampling method from config"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("KRATOS PARAMETRIC STUDY RUNNER")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Dry run: {args.dry_run}")
    print(f"Setup only: {args.setup_only}")
    print(f"Start case: {args.start_case}")
    
    # Load configuration
    config = load_config(args.config)
    
    # ── Override sampling method if specified on command line ──  # ← NEW
    if args.method:
        config.sampling_method = args.method
        print(f"Sampling method (CLI override): {config.sampling_method}")
    else:
        print(f"Sampling method: {config.sampling_method}")
    
    # Set random seed (kept for any code still using Python's random module)
    if config.random_seed is not None:
        random.seed(config.random_seed)
        print(f"Random seed: {config.random_seed}")
    else:
        print("Random seed: None (truly random)")
    
    print(f"Working directory: {config.working_directory}")
    print(f"Number of cases: {config.num_cases}")
    
    # Create output directories
    os.makedirs(
        os.path.join(config.working_directory, config.primal_folder),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(config.working_directory, config.adjoint_folder),
        exist_ok=True
    )
    
    # ================================================================
    # CHANGED: Generate ALL parameters at once using the chosen method
    # This replaces the old per-case loop:
    #   for i in range(config.num_cases):
    #       params = generate_case_parameters(case_id, config)
    # ================================================================
    all_case_params, unit_samples, dimensions = \
        generate_all_case_parameters(config, args.start_case)
    
    # Track all cases
    all_cases: List[Tuple[int, str, str, CaseParameters]] = []
    
    # Setup cases (create folders and files) — THIS LOOP IS SIMPLIFIED
    for params in all_case_params:                          # ← CHANGED
        try:
            primal_folder, adjoint_folder = setup_case(
                params.case_id, params, config,             # ← CHANGED
                dry_run=args.dry_run
            )
            all_cases.append((
                params.case_id, primal_folder,              # ← CHANGED
                adjoint_folder, params
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
            "sampling_method": config.sampling_method,      # ← NEW
            "varying_dimensions": [                         # ← NEW
                {
                    "name": d.name,
                    "min": d.min_val,
                    "max": d.max_val,
                    "distribution": d.distribution
                }
                for d in dimensions
            ],
            "cases": [
                {
                    "id": c[0],
                    "primal": c[1],
                    "adjoint": c[2],
                    "parameters": {
                        "udl": c[3].udl,
                        "youngs_modulus": c[3].youngs_modulus,
                        "I22": c[3].I22,
                        "I33": c[3].I33,
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
    
    # ── Run analyses (UNCHANGED from your original) ──
    if not args.dry_run and not args.setup_only:
        print("\n" + "=" * 60)
        print("RUNNING ANALYSES")
        print("=" * 60)
        
        results = []
        
        for case_id, primal_folder, adjoint_folder, params in all_cases:
            print(f"\n{'>' * 20} Case {case_id} {'<' * 20}")
            
            primal_success = run_kratos_analysis(
                primal_folder, "primal"
            )
            
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
                    print(f"  Copied {h5_copied} HDF5 file(s) "
                          f"to adjoint folder")
                
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
        failed = sum(
            1 for r in results if not r["primal_success"]
        )
        
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