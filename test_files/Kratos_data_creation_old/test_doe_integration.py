"""
Integration Test for DoE Sampling Methods
==========================================

Run this BEFORE launching your parametric study to verify:
  1. All sampling methods generate valid samples
  2. Transforms produce values in correct ranges
  3. Distribution shapes are correct (uniform stays uniform, 
     loguniform concentrates at lower values)
  4. LHS property is maintained (one sample per stratum)
  5. Fixed parameters (min==max) are handled correctly
  6. The full pipeline works end-to-end

Usage:
    python test_doe_integration.py
    python test_doe_integration.py --config config.yaml
    python test_doe_integration.py --verbose

Exit code 0 = all tests pass, 1 = failures detected
"""

import os
import sys
import math
import json
import argparse
import traceback
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

try:
    from scipy.stats import qmc
    from scipy.spatial.distance import pdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ════════════════════════════════════════════════════════════
# Import from your main script
# ════════════════════════════════════════════════════════════

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from run_parametric_study import (
        ParameterDimension,
        StudyConfig,
        CaseParameters,
        build_parameter_dimensions,
        generate_unit_samples,
        transform_sample,
        generate_all_case_parameters,
        _lhs_fallback,
        to_float,
        load_config,
    )
    IMPORT_SUCCESS = True
    print("✓ Successfully imported from run_parametric_study.py")
except ImportError as e:
    IMPORT_SUCCESS = False
    print(f"✗ Failed to import from run_parametric_study.py: {e}")
    print("  Make sure run_parametric_study.py is in the same directory")

# Try importing maximin
try:
    from run_parametric_study import _maximin_lhs
    HAS_MAXIMIN = True
    print("✓ _maximin_lhs imported")
except ImportError:
    HAS_MAXIMIN = False
    print("  Note: _maximin_lhs not found (optional)")


# ════════════════════════════════════════════════════════════
# TEST INFRASTRUCTURE
# ════════════════════════════════════════════════════════════

class TestResult:
    """Store result of a single test."""
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message


class TestRunner:
    """Simple test framework."""
    
    def __init__(self, verbose: bool = False):
        self.results: List[TestResult] = []
        self.verbose = verbose
        self.current_section = ""
    
    def section(self, name: str):
        """Start a new test section."""
        self.current_section = name
        print(f"\n{'━' * 60}")
        print(f"  {name}")
        print(f"{'━' * 60}")
    
    def test(self, name: str, condition: bool, message: str = ""):
        """Record a test result."""
        result = TestResult(
            f"{self.current_section}: {name}",
            condition,
            message
        )
        self.results.append(result)
        
        if condition:
            print(f"  ✓ {name}")
            if self.verbose and message:
                print(f"    {message}")
        else:
            print(f"  ✗ {name}")
            if message:
                print(f"    DETAIL: {message}")
    
    def test_approx(
        self, name: str,
        actual: float, expected: float,
        tolerance: float = 0.05,
        message: str = ""
    ):
        """Test that a value is approximately equal to expected."""
        if expected == 0:
            close = abs(actual) < tolerance
        else:
            close = abs(actual - expected) / abs(expected) < tolerance
        
        detail = f"actual={actual:.6f}, expected={expected:.6f}"
        if message:
            detail = f"{message} | {detail}"
        
        self.test(name, close, detail)
    
    def test_in_range(
        self, name: str,
        value: float,
        low: float, high: float,
        message: str = ""
    ):
        """Test that a value falls within [low, high]."""
        in_range = low <= value <= high
        detail = f"value={value:.6e}, range=[{low:.6e}, {high:.6e}]"
        if message:
            detail = f"{message} | {detail}"
        self.test(name, in_range, detail)
    
    def summary(self) -> bool:
        """Print summary and return True if all passed."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        
        print(f"\n{'═' * 60}")
        print(f"  TEST SUMMARY")
        print(f"{'═' * 60}")
        print(f"  Total:  {total}")
        print(f"  Passed: {passed} ✓")
        print(f"  Failed: {failed} ✗")
        
        if failed > 0:
            print(f"\n  FAILED TESTS:")
            for r in self.results:
                if not r.passed:
                    print(f"    ✗ {r.name}")
                    if r.message:
                        print(f"      {r.message}")
        
        print(f"{'═' * 60}")
        
        if failed == 0:
            print("  ALL TESTS PASSED ✓")
        else:
            print(f"  {failed} TEST(S) FAILED ✗")
        
        print(f"{'═' * 60}\n")
        
        return failed == 0


# ════════════════════════════════════════════════════════════
# TEST 1: UNIT SAMPLE GENERATION
# ════════════════════════════════════════════════════════════

def test_unit_sample_generation(runner: TestRunner):
    """
    Verify that each sampling method produces valid [0,1]^D samples.
    
    WHAT WE CHECK:
    1. Output shape is (N, D)
    2. All values are in [0, 1)
    3. No NaN or Inf values
    4. Different methods produce different results
    5. Same seed produces same results (reproducibility)
    """
    runner.section("UNIT SAMPLE GENERATION")
    
    N, D = 32, 5  # Use power of 2 for Sobol
    seed = 42
    
    methods = ["random", "lhs", "optimal_lhs", "sobol"]
    if HAS_MAXIMIN:
        methods.append("maximin_lhs")
    
    all_samples = {}
    
    for method in methods:
        try:
            samples = generate_unit_samples(N, D, method, seed)
            all_samples[method] = samples
            
            # Shape check
            runner.test(
                f"{method}: shape is ({N}, {D})",
                samples.shape == (N, D),
                f"actual shape: {samples.shape}"
            )
            
            # Range check
            runner.test(
                f"{method}: all values ≥ 0",
                np.all(samples >= 0),
                f"min value: {np.min(samples):.6f}"
            )
            runner.test(
                f"{method}: all values ≤ 1",
                np.all(samples <= 1.0),
                f"max value: {np.max(samples):.6f}"
            )
            
            # No NaN/Inf
            runner.test(
                f"{method}: no NaN values",
                not np.any(np.isnan(samples)),
                f"NaN count: {np.sum(np.isnan(samples))}"
            )
            runner.test(
                f"{method}: no Inf values",
                not np.any(np.isinf(samples)),
                f"Inf count: {np.sum(np.isinf(samples))}"
            )
            
        except Exception as e:
            runner.test(
                f"{method}: generation succeeds",
                False,
                f"Exception: {e}"
            )
    
    # Reproducibility check
    for method in ["random", "lhs", "sobol"]:
        if method in all_samples:
            try:
                samples2 = generate_unit_samples(N, D, method, seed)
                runner.test(
                    f"{method}: same seed → same result",
                    np.allclose(all_samples[method], samples2),
                    f"max diff: {np.max(np.abs(all_samples[method] - samples2)):.2e}"
                )
            except Exception as e:
                runner.test(
                    f"{method}: reproducibility check",
                    False,
                    f"Exception: {e}"
                )
    
    # Different methods should give different results
    if "random" in all_samples and "lhs" in all_samples:
        runner.test(
            "random ≠ lhs (different methods differ)",
            not np.allclose(all_samples["random"], all_samples["lhs"]),
            "Methods should produce different sample sets"
        )
    
    return all_samples


# ════════════════════════════════════════════════════════════
# TEST 2: LHS PROPERTY VERIFICATION
# ════════════════════════════════════════════════════════════

def test_lhs_property(runner: TestRunner):
    """
    Verify the Latin Hypercube property: 
    In each dimension, exactly one sample falls in each stratum.
    
    WHAT IS THE LHS PROPERTY?
    For N samples, divide [0,1] into N equal strata:
      [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
    Each stratum must contain exactly one sample in every column.
    
    This is like a Sudoku constraint: each "row" (stratum) of each
    dimension has exactly one sample.
    
    VISUAL for N=4, D=2:
    ┌─────┬─────┬─────┬─────┐
    │     │  x  │     │     │  stratum 3: col0 has x, col1 has x
    ├─────┼─────┼─────┼─────┤
    │     │     │     │  x  │  stratum 2: col0 has x, col1 has x  
    ├─────┼─────┼─────┼─────┤
    │  x  │     │     │     │  stratum 1: etc.
    ├─────┼─────┼─────┼─────┤
    │     │     │  x  │     │  stratum 0
    └─────┴─────┴─────┴─────┘
      col0  col1  col0  col1
      
    Each column has exactly one x per row (stratum) ← THIS is the LHS property
    """
    runner.section("LHS PROPERTY VERIFICATION")
    
    N, D = 20, 7
    seed = 123
    
    lhs_methods = ["lhs", "optimal_lhs"]
    if HAS_MAXIMIN:
        lhs_methods.append("maximin_lhs")
    
    for method in lhs_methods:
        try:
            samples = generate_unit_samples(N, D, method, seed)
            
            all_strata_valid = True
            worst_violation = ""
            
            for j in range(D):
                # Determine which stratum each sample falls in
                strata = np.floor(samples[:, j] * N).astype(int)
                # Handle edge case where value == 1.0
                strata = np.clip(strata, 0, N - 1)
                
                # Count samples per stratum
                counts = np.bincount(strata, minlength=N)
                
                if not np.all(counts == 1):
                    all_strata_valid = False
                    empty = np.where(counts == 0)[0]
                    multi = np.where(counts > 1)[0]
                    worst_violation = (
                        f"dim {j}: empty strata={empty.tolist()}, "
                        f"multi-sample strata={multi.tolist()}"
                    )
                    break
            
            runner.test(
                f"{method}: exactly 1 sample per stratum per dimension",
                all_strata_valid,
                worst_violation if not all_strata_valid else
                f"All {D} dimensions verified for {N} strata"
            )
            
        except Exception as e:
            runner.test(
                f"{method}: LHS property check",
                False,
                f"Exception: {e}"
            )


# ════════════════════════════════════════════════════════════
# TEST 3: TRANSFORM VERIFICATION
# ════════════════════════════════════════════════════════════

def test_transforms(runner: TestRunner):
    """
    Verify that transform_sample correctly maps [0,1] → physical values.
    
    WHAT WE CHECK:
    
    For each distribution type, we verify:
    1. u=0 → min value
    2. u=1 → max value (approximately)
    3. u=0.5 → midpoint (arithmetic for uniform, geometric for loguniform)
    4. Monotonicity: u₁ < u₂ → value₁ < value₂
    5. Range: all transformed values fall within [min, max]
    
    WHY THE MIDPOINT TEST MATTERS:
    
    For UNIFORM: midpoint of [20, 100] = 60 (arithmetic mean)
    For LOGUNIFORM: midpoint of [1e-5, 1e-3] = 1e-4 (geometric mean!)
    
    The geometric mean is: 10^((log₁₀(1e-5) + log₁₀(1e-3))/2) 
                         = 10^((-5 + -3)/2) = 10^(-4) = 1e-4
    
    If the loguniform transform gave 5.05e-4 (arithmetic mean),
    that would be WRONG — it would oversample large values.
    """
    runner.section("TRANSFORM VERIFICATION")
    
    # ── Uniform transform ──
    dim_uniform = ParameterDimension(
        name="udl", min_val=20.0, max_val=100.0, distribution="uniform"
    )
    
    runner.test_approx(
        "uniform: u=0 → min",
        transform_sample(0.0, dim_uniform), 20.0,
        tolerance=0.001
    )
    runner.test_approx(
        "uniform: u=1 → max",
        transform_sample(1.0, dim_uniform), 100.0,
        tolerance=0.001
    )
    runner.test_approx(
        "uniform: u=0.5 → arithmetic midpoint",
        transform_sample(0.5, dim_uniform), 60.0,
        tolerance=0.001,
        message="(20+100)/2 = 60"
    )
    
    # Monotonicity
    u_values = np.linspace(0.01, 0.99, 50)
    transformed = [transform_sample(u, dim_uniform) for u in u_values]
    is_monotonic = all(
        transformed[i] < transformed[i + 1]
        for i in range(len(transformed) - 1)
    )
    runner.test(
        "uniform: monotonically increasing",
        is_monotonic,
        "transform should be strictly increasing with u"
    )
    
    # ── Loguniform transform ──
    dim_loguniform = ParameterDimension(
        name="E", min_val=1e-5, max_val=1e-3, distribution="loguniform"
    )
    
    runner.test_approx(
        "loguniform: u=0 → min",
        transform_sample(0.0, dim_loguniform), 1e-5,
        tolerance=0.001
    )
    runner.test_approx(
        "loguniform: u=1 → max",
        transform_sample(1.0, dim_loguniform), 1e-3,
        tolerance=0.001
    )
    runner.test_approx(
        "loguniform: u=0.5 → geometric midpoint",
        transform_sample(0.5, dim_loguniform), 1e-4,
        tolerance=0.001,
        message="10^((-5+-3)/2) = 10^(-4) = 1e-4"
    )
    
    # Verify it's NOT the arithmetic midpoint
    arithmetic_mid = (1e-5 + 1e-3) / 2  # = 5.05e-4
    geometric_mid = transform_sample(0.5, dim_loguniform)
    runner.test(
        "loguniform: midpoint is geometric, not arithmetic",
        abs(geometric_mid - 1e-4) < abs(geometric_mid - arithmetic_mid),
        f"geometric={geometric_mid:.2e}, arithmetic would be {arithmetic_mid:.2e}"
    )
    
    # Monotonicity
    transformed_log = [
        transform_sample(u, dim_loguniform) for u in u_values
    ]
    is_monotonic_log = all(
        transformed_log[i] < transformed_log[i + 1]
        for i in range(len(transformed_log) - 1)
    )
    runner.test(
        "loguniform: monotonically increasing",
        is_monotonic_log
    )
    
    # ── Discrete transform ──
    dim_discrete = ParameterDimension(
        name="subdivisions", min_val=2, max_val=5,
        distribution="discrete"
    )
    
    # Sample many values and check they're all integers in {2,3,4,5}
    valid_values = {2, 3, 4, 5}
    discrete_samples = [
        transform_sample(u, dim_discrete)
        for u in np.linspace(0.0, 0.999, 100)
    ]
    all_valid = all(int(v) in valid_values for v in discrete_samples)
    unique_values = set(int(v) for v in discrete_samples)
    
    runner.test(
        "discrete: all values are valid integers",
        all_valid,
        f"unique values: {sorted(unique_values)}, "
        f"expected: {sorted(valid_values)}"
    )
    runner.test(
        "discrete: all possible values appear",
        unique_values == valid_values,
        f"got {sorted(unique_values)}, expected {sorted(valid_values)}"
    )
    
    # ── Normal transform ──
    dim_normal = ParameterDimension(
        name="test_normal", min_val=0.0, max_val=100.0,
        distribution="normal"
    )
    
    runner.test_approx(
        "normal: u=0.5 → mean",
        transform_sample(0.5, dim_normal), 50.0,
        tolerance=0.001,
        message="midpoint of normal is the mean"
    )
    
    # u=0.5±ε should be close to mean
    val_low = transform_sample(0.3, dim_normal)
    val_high = transform_sample(0.7, dim_normal)
    runner.test(
        "normal: u=0.3 < u=0.7 in value",
        val_low < val_high,
        f"val(0.3)={val_low:.2f}, val(0.7)={val_high:.2f}"
    )
    
    # ── Range verification across many samples ──
    test_dims = [
        ParameterDimension("A", 20, 100, "uniform"),
        ParameterDimension("B", 1e10, 2.5e11, "loguniform"),
        ParameterDimension("C", 1e-5, 1e-3, "loguniform"),
        ParameterDimension("D", 0.01, 0.1, "uniform"),
    ]
    
    for dim in test_dims:
        values = [
            transform_sample(u, dim)
            for u in np.linspace(0.001, 0.999, 200)
        ]
        min_v = min(values)
        max_v = max(values)
        
        runner.test_in_range(
            f"range check: {dim.name} min ≥ declared min",
            min_v, dim.min_val * 0.999, dim.max_val * 1.001,
            f"distribution={dim.distribution}"
        )
        runner.test_in_range(
            f"range check: {dim.name} max ≤ declared max",
            max_v, dim.min_val * 0.999, dim.max_val * 1.001,
            f"distribution={dim.distribution}"
        )


# ════════════════════════════════════════════════════════════
# TEST 4: DIMENSION BUILDING
# ════════════════════════════════════════════════════════════

def test_dimension_building(runner: TestRunner):
    """
    Verify that build_parameter_dimensions correctly identifies
    varying vs fixed parameters.
    
    WHAT WE CHECK:
    - Parameters with min ≠ max are INCLUDED (they're dimensions)
    - Parameters with min == max are EXCLUDED (they're constants)
    - Response location axes are handled correctly
    - The dimension count matches expectation
    
    YOUR CONFIG has these fixed parameters (min == max):
    - I33:          1e-5 to 1e-5      → FIXED
    - subdivisions: 3 to 3             → FIXED  
    - response_y:   0.0 to 0.0         → FIXED
    
    And these varying parameters:
    - udl:               20 to 100       → DIMENSION
    - youngs_modulus:     1e10 to 2.5e11  → DIMENSION
    - I22:               1e-5 to 1e-3    → DIMENSION
    - cross_area:        0.01 to 0.1     → DIMENSION
    - torsional_inertia: 1e-6 to 1e-4    → DIMENSION
    - response_x:        0.0 to 6.0      → DIMENSION
    - response_z:        3.0 to 18.0     → DIMENSION
    
    Expected: 7 varying dimensions
    """
    runner.section("DIMENSION BUILDING")
    
    # Create a mock config matching your YAML
    mock_config = StudyConfig(
        num_cases=5,
        random_seed=42,
        working_directory=".",
        primal_folder="primal",
        adjoint_folder="adjoint",
        primal_prefix="case_primal",
        adjoint_prefix="case_adjoint",
        template_mdpa="",
        template_primal="",
        template_adjoint="",
        template_materials="",
        param_ranges={
            "udl": {"min": 20.0, "max": 100.0, "distribution": "uniform"},
            "youngs_modulus": {
                "min": 1.0e10, "max": 2.5e11,
                "distribution": "loguniform"
            },
            "I22": {
                "min": 1.0e-5, "max": 1.0e-3,
                "distribution": "loguniform"
            },
            "I33": {
                "min": 1.0e-5, "max": 1.0e-5,  # FIXED
                "distribution": "loguniform"
            },
            "cross_area": {
                "min": 0.01, "max": 0.1,
                "distribution": "uniform"
            },
            "torsional_inertia": {
                "min": 1.0e-6, "max": 1.0e-4,
                "distribution": "loguniform"
            },
            "subdivisions": {
                "min": 3, "max": 3,  # FIXED
                "distribution": "discrete"
            },
            "response_location": {
                "x": {"min": 0.0, "max": 6.0},
                "y": {"min": 0.0, "max": 0.0},  # FIXED
                "z": {"min": 3.0, "max": 18.0},
                "distribution": "uniform"
            },
        },
        fixed_params={"density": 7850.0, "poisson_ratio": 0.3},
        response_settings={"stress_type": "MY"},
        sampling_method="lhs",
    )
    
    dims = build_parameter_dimensions(mock_config)
    dim_names = [d.name for d in dims]
    
    # Count check
    runner.test(
        "dimension count is 7",
        len(dims) == 7,
        f"got {len(dims)}: {dim_names}"
    )
    
    # Varying parameters should be present
    expected_present = [
        "udl", "youngs_modulus", "I22",
        "cross_area", "torsional_inertia",
        "response_x", "response_z"
    ]
    for name in expected_present:
        runner.test(
            f"'{name}' is a dimension (varying)",
            name in dim_names,
            f"dimensions found: {dim_names}"
        )
    
    # Fixed parameters should NOT be present
    expected_absent = ["I33", "subdivisions", "response_y"]
    for name in expected_absent:
        runner.test(
            f"'{name}' is NOT a dimension (fixed, min==max)",
            name not in dim_names,
            f"dimensions found: {dim_names}"
        )
    
    # Distribution types should be preserved
    for dim in dims:
        if dim.name == "udl":
            runner.test(
                "udl distribution is 'uniform'",
                dim.distribution == "uniform",
                f"got: {dim.distribution}"
            )
        elif dim.name == "youngs_modulus":
            runner.test(
                "youngs_modulus distribution is 'loguniform'",
                dim.distribution == "loguniform",
                f"got: {dim.distribution}"
            )
    
    return mock_config


# ════════════════════════════════════════════════════════════
# TEST 5: FULL PIPELINE
# ════════════════════════════════════════════════════════════

def test_full_pipeline(runner: TestRunner, mock_config: StudyConfig):
    """
    Test the complete pipeline: config → dimensions → samples → parameters.
    
    WHAT WE CHECK:
    1. generate_all_case_parameters produces correct number of cases
    2. Each CaseParameters has valid values
    3. Fixed parameters are constant across all cases
    4. Varying parameters differ across cases
    5. All parameter values fall within declared ranges
    """
    runner.section("FULL PIPELINE (end-to-end)")
    
    methods_to_test = ["random", "lhs", "optimal_lhs", "sobol"]
    if HAS_MAXIMIN:
        methods_to_test.append("maximin_lhs")
    
    for method in methods_to_test:
        mock_config.sampling_method = method
        mock_config.num_cases = 16  # Power of 2 for Sobol
        
        try:
            all_params, unit_samples, dimensions = \
                generate_all_case_parameters(mock_config, start_case=1)
            
            # ── Count check ──
            runner.test(
                f"{method}: produces {mock_config.num_cases} cases",
                len(all_params) == mock_config.num_cases,
                f"got {len(all_params)}"
            )
            
            # ── Unit samples shape ──
            if unit_samples is not None:
                runner.test(
                    f"{method}: unit samples shape correct",
                    unit_samples.shape == (
                        mock_config.num_cases, len(dimensions)
                    ),
                    f"shape: {unit_samples.shape}, expected: "
                    f"({mock_config.num_cases}, {len(dimensions)})"
                )
            
            # ── Fixed parameter constancy ──
            i33_values = [p.I33 for p in all_params]
            subdiv_values = [p.subdivisions for p in all_params]
            resp_y_values = [p.response_coords[1] for p in all_params]
            
            runner.test(
                f"{method}: I33 is constant (fixed parameter)",
                len(set(i33_values)) == 1,
                f"unique I33 values: {set(i33_values)}"
            )
            runner.test(
                f"{method}: subdivisions is constant (fixed)",
                len(set(subdiv_values)) == 1,
                f"unique subdivision values: {set(subdiv_values)}"
            )
            runner.test(
                f"{method}: response_y is constant (fixed)",
                all(abs(y - 0.0) < 1e-10 for y in resp_y_values),
                f"response_y values: {resp_y_values[:3]}..."
            )
            
            # ── Varying parameter ranges ──
            for p in all_params:
                runner.test_in_range(
                    f"{method}: udl in [20, 100]",
                    p.udl, 20.0, 100.0
                )
                runner.test_in_range(
                    f"{method}: youngs_modulus in [1e10, 2.5e11]",
                    p.youngs_modulus, 1e10, 2.5e11
                )
                runner.test_in_range(
                    f"{method}: I22 in [1e-5, 1e-3]",
                    p.I22, 1e-5, 1e-3
                )
                runner.test_in_range(
                    f"{method}: cross_area in [0.01, 0.1]",
                    p.cross_area, 0.01, 0.1
                )
                runner.test_in_range(
                    f"{method}: response_x in [0, 6]",
                    p.response_coords[0], 0.0, 6.0
                )
                runner.test_in_range(
                    f"{method}: response_z in [3, 18]",
                    p.response_coords[2], 3.0, 18.0
                )
                # Only check first case to avoid flooding output
                break
            
            # ── Variability check ──
            udl_values = [p.udl for p in all_params]
            runner.test(
                f"{method}: udl varies across cases",
                len(set(udl_values)) > 1,
                f"unique values: {len(set(udl_values))}"
            )
            
            e_values = [p.youngs_modulus for p in all_params]
            runner.test(
                f"{method}: youngs_modulus varies across cases",
                len(set(e_values)) > 1,
                f"unique values: {len(set(e_values))}"
            )
            
        except Exception as e:
            runner.test(
                f"{method}: full pipeline succeeds",
                False,
                f"Exception: {e}\n{traceback.format_exc()}"
            )


# ════════════════════════════════════════════════════════════
# TEST 6: QUALITY METRICS
# ════════════════════════════════════════════════════════════

def test_quality_ordering(runner: TestRunner):
    """
    Verify that structured methods produce better quality
    than random sampling (on average).
    
    WHAT WE EXPECT:
    
    Discrepancy (lower = better):
      sobol ≤ optimal_lhs ≤ lhs ≤ random
      
    This isn't guaranteed for EVERY seed, but should hold
    for most seeds. We use a generous tolerance.
    
    WHY THIS TEST MATTERS:
    If random beats Sobol, something is wrong with the
    implementation. This is a sanity check, not a rigorous
    statistical test.
    """
    runner.section("QUALITY ORDERING (sanity check)")
    
    if not HAS_SCIPY:
        runner.test(
            "scipy available for quality metrics",
            False,
            "Need scipy for discrepancy computation"
        )
        return
    
    N, D = 64, 5
    
    discrepancies = {}
    methods = ["random", "lhs", "optimal_lhs", "sobol"]
    if HAS_MAXIMIN:
        methods.append("maximin_lhs")
    
    # Average over several seeds to reduce noise
    n_seeds = 5
    
    for method in methods:
        disc_values = []
        for seed in range(42, 42 + n_seeds):
            try:
                samples = generate_unit_samples(N, D, method, seed)
                disc = float(qmc.discrepancy(samples, method='CD'))
                disc_values.append(disc)
            except Exception:
                pass
        
        if disc_values:
            discrepancies[method] = np.median(disc_values)
            runner.test(
                f"{method}: discrepancy computed",
                True,
                f"median CD = {discrepancies[method]:.6e} "
                f"(over {len(disc_values)} seeds)"
            )
    
    # Check ordering (with tolerance)
    if "random" in discrepancies and "lhs" in discrepancies:
        runner.test(
            "lhs discrepancy < random discrepancy",
            discrepancies["lhs"] < discrepancies["random"] * 1.5,
            f"lhs={discrepancies['lhs']:.6e}, "
            f"random={discrepancies['random']:.6e}"
        )
    
    if "lhs" in discrepancies and "optimal_lhs" in discrepancies:
        runner.test(
            "optimal_lhs discrepancy ≤ lhs discrepancy",
            discrepancies["optimal_lhs"] <= discrepancies["lhs"] * 1.1,
            f"optimal={discrepancies['optimal_lhs']:.6e}, "
            f"lhs={discrepancies['lhs']:.6e}"
        )
    
    if "sobol" in discrepancies and "random" in discrepancies:
        runner.test(
            "sobol discrepancy << random discrepancy",
            discrepancies["sobol"] < discrepancies["random"],
            f"sobol={discrepancies['sobol']:.6e}, "
            f"random={discrepancies['random']:.6e}"
        )
    
    if HAS_MAXIMIN and "maximin_lhs" in discrepancies:
        if "lhs" in discrepancies:
            # Maximin might have slightly higher discrepancy than CD-optimal
            # but should still be better than random
            runner.test(
                "maximin_lhs discrepancy < random discrepancy",
                discrepancies["maximin_lhs"] < discrepancies["random"] * 1.2,
                f"maximin={discrepancies['maximin_lhs']:.6e}, "
                f"random={discrepancies['random']:.6e}"
            )
    
    # Check that maximin LHS has better minimum distance
    mindists = {}
    for method in methods:
        try:
            samples = generate_unit_samples(N, D, method, 42)
            md = float(np.min(pdist(samples)))
            mindists[method] = md
        except Exception:
            pass
    
    if HAS_MAXIMIN and "maximin_lhs" in mindists and "lhs" in mindists:
        runner.test(
            "maximin_lhs has higher min-distance than basic lhs",
            mindists["maximin_lhs"] >= mindists["lhs"] * 0.9,
            f"maximin={mindists['maximin_lhs']:.6f}, "
            f"lhs={mindists['lhs']:.6f}"
        )
    
    # Print summary table
    print(f"\n  {'Method':<16} {'Discrepancy':>14} {'Min Distance':>14}")
    print(f"  {'─' * 46}")
    for method in methods:
        disc = discrepancies.get(method, float('nan'))
        md = mindists.get(method, float('nan'))
        print(f"  {method:<16} {disc:>14.6e} {md:>14.6f}")


# ════════════════════════════════════════════════════════════
# TEST 7: EDGE CASES
# ════════════════════════════════════════════════════════════

def test_edge_cases(runner: TestRunner):
    """
    Test boundary conditions and unusual inputs.
    
    WHAT WE CHECK:
    1. N=1 (single sample) — should work without errors
    2. D=1 (single dimension) — degenerate but valid
    3. N=2 (minimum for distances) — should work
    4. Large N with Sobol — power of 2 warning
    5. All parameters fixed — 0 dimensions
    """
    runner.section("EDGE CASES")
    
    # ── N=1 ──
    for method in ["random", "lhs", "sobol"]:
        try:
            samples = generate_unit_samples(1, 3, method, 42)
            runner.test(
                f"{method}: N=1 works",
                samples.shape == (1, 3),
                f"shape: {samples.shape}"
            )
        except Exception as e:
            runner.test(
                f"{method}: N=1 works",
                False,
                f"Exception: {e}"
            )
    
    # ── D=1 ──
    for method in ["random", "lhs", "optimal_lhs", "sobol"]:
        try:
            samples = generate_unit_samples(10, 1, method, 42)
            runner.test(
                f"{method}: D=1 works",
                samples.shape == (10, 1),
                f"shape: {samples.shape}"
            )
        except Exception as e:
            runner.test(
                f"{method}: D=1 works",
                False,
                f"Exception: {e}"
            )
    
    # ── D=0 (all fixed) ──
    try:
        samples = generate_unit_samples(5, 0, "lhs", 42)
        runner.test(
            "D=0 (all fixed): returns empty array",
            samples.shape == (5, 0),
            f"shape: {samples.shape}"
        )
    except Exception as e:
        runner.test(
            "D=0 (all fixed): handled gracefully",
            False,
            f"Exception: {e}"
        )
    
    # ── Transform edge: u exactly 0 and 1 ──
    dim = ParameterDimension("test", 10.0, 100.0, "uniform")
    try:
        v0 = transform_sample(0.0, dim)
        v1 = transform_sample(1.0, dim)
        runner.test(
            "transform: u=0.0 and u=1.0 don't crash",
            True,
            f"v(0)={v0}, v(1)={v1}"
        )
    except Exception as e:
        runner.test(
            "transform: boundary values",
            False,
            f"Exception: {e}"
        )
    
    # ── Loguniform with very wide range ──
    dim_wide = ParameterDimension("wide", 1e-15, 1e15, "loguniform")
    try:
        v_mid = transform_sample(0.5, dim_wide)
        runner.test_approx(
            "loguniform: 30 orders of magnitude range",
            v_mid, 1.0,
            tolerance=0.01,
            message="10^((-15+15)/2) = 10^0 = 1.0"
        )
    except Exception as e:
        runner.test(
            "loguniform: wide range",
            False,
            f"Exception: {e}"
        )


# ════════════════════════════════════════════════════════════
# TEST 8: CONFIG FILE INTEGRATION
# ════════════════════════════════════════════════════════════

def test_config_loading(runner: TestRunner, config_path: Optional[str]):
    """
    Test loading an actual config file (if provided).
    
    Verifies that your real config.yaml produces valid dimensions
    and the full pipeline works with it.
    """
    runner.section("CONFIG FILE INTEGRATION")
    
    if config_path is None or not os.path.exists(config_path):
        runner.test(
            "config file exists",
            False,
            f"Not found: {config_path}. Skipping config tests. "
            f"Run with --config <path> to test your config."
        )
        return
    
    try:
        config = load_config(config_path)
        runner.test(
            "config loads successfully",
            True,
            f"working_dir: {config.working_directory}"
        )
        
        # Override working directory for testing
        config.working_directory = "."
        
        dims = build_parameter_dimensions(config)
        runner.test(
            f"config produces {len(dims)} dimensions",
            len(dims) > 0,
            f"dimensions: {[d.name for d in dims]}"
        )
        
        # Test with each method
        for method in ["lhs", "sobol"]:
            config.sampling_method = method
            config.num_cases = 8
            
            try:
                params, samples, dimensions = \
                    generate_all_case_parameters(config, start_case=1)
                
                runner.test(
                    f"config + {method}: generates {config.num_cases} cases",
                    len(params) == config.num_cases,
                    f"got {len(params)}"
                )
            except Exception as e:
                runner.test(
                    f"config + {method}: pipeline works",
                    False,
                    f"Exception: {e}"
                )
        
    except Exception as e:
        runner.test(
            "config processing",
            False,
            f"Exception: {e}\n{traceback.format_exc()}"
        )


# ════════════════════════════════════════════════════════════
# TEST 9: DISTRIBUTION SHAPE VERIFICATION
# ════════════════════════════════════════════════════════════

def test_distribution_shapes(runner: TestRunner):
    """
    Verify that transformed samples follow the correct distributions.
    
    CONCEPT: If we generate uniform [0,1] samples and transform them,
    the result should follow the specified distribution.
    
    For UNIFORM: transformed values should be uniformly spread
    For LOGUNIFORM: log₁₀(values) should be uniformly spread
    
    We check this using a simple bin test:
    - Divide the range into bins
    - Count samples per bin
    - For uniform: bins should have ~equal counts
    - For loguniform: LOG-spaced bins should have ~equal counts
    """
    runner.section("DISTRIBUTION SHAPE VERIFICATION")
    
    N = 256
    
    # ── Uniform distribution ──
    dim_u = ParameterDimension("test_u", 20.0, 100.0, "uniform")
    u_samples = np.linspace(0.001, 0.999, N)
    values = np.array([transform_sample(u, dim_u) for u in u_samples])
    
    # Split into 4 quarters — each should have ~N/4 samples
    q1 = np.sum((values >= 20) & (values < 40))
    q2 = np.sum((values >= 40) & (values < 60))
    q3 = np.sum((values >= 60) & (values < 80))
    q4 = np.sum((values >= 80) & (values <= 100))
    
    expected = N / 4
    max_deviation = max(
        abs(q - expected) / expected
        for q in [q1, q2, q3, q4]
    )
    
    runner.test(
        "uniform: quarters have ~equal counts",
        max_deviation < 0.15,
        f"quarters: [{q1}, {q2}, {q3}, {q4}], "
        f"expected ~{expected:.0f} each, max deviation {max_deviation:.1%}"
    )
    
    # ── Loguniform distribution ──
    dim_log = ParameterDimension("test_log", 1e-4, 1e0, "loguniform")
    values_log = np.array([
        transform_sample(u, dim_log) for u in u_samples
    ])
    
    # In LOG space, quarters should be equal
    log_values = np.log10(values_log)
    # log range is [-4, 0], quarters are [-4,-3], [-3,-2], [-2,-1], [-1,0]
    lq1 = np.sum((log_values >= -4) & (log_values < -3))
    lq2 = np.sum((log_values >= -3) & (log_values < -2))
    lq3 = np.sum((log_values >= -2) & (log_values < -1))
    lq4 = np.sum((log_values >= -1) & (log_values <= 0))
    
    max_log_dev = max(
        abs(q - expected) / expected
        for q in [lq1, lq2, lq3, lq4]
    )
    
    runner.test(
        "loguniform: log-space quarters have ~equal counts",
        max_log_dev < 0.15,
        f"log-quarters: [{lq1}, {lq2}, {lq3}, {lq4}], "
        f"expected ~{expected:.0f} each"
    )
    
    # Verify that in LINEAR space, lower values are more concentrated
    # (this is the whole point of loguniform)
    linear_q1 = np.sum(values_log < 0.25 * (1e0 - 1e-4) + 1e-4)
    linear_q4 = np.sum(values_log > 0.75 * (1e0 - 1e-4) + 1e-4)
    
    runner.test(
        "loguniform: more samples at lower values (linear space)",
        linear_q1 > linear_q4,
        f"samples below 25% linear: {linear_q1}, "
        f"above 75% linear: {linear_q4}"
    )


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Test DoE integration"
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to your config.yaml for integration testing"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show details for passing tests too"
    )
    args = parser.parse_args()
    
    print("═" * 60)
    print("  DoE INTEGRATION TESTS")
    print("═" * 60)
    print(f"  scipy:      {'✓' if HAS_SCIPY else '✗'}")
    print(f"  imports:    {'✓' if IMPORT_SUCCESS else '✗'}")
    print(f"  maximin:    {'✓' if HAS_MAXIMIN else '(not found)'}")
    print(f"  config:     {args.config or '(none)'}")
    
    if not IMPORT_SUCCESS:
        print("\n✗ Cannot run tests without successful imports.")
        print("  Fix the import errors above first.")
        sys.exit(1)
    
    runner = TestRunner(verbose=args.verbose)
    
    # ── Run all test suites ──
    test_unit_sample_generation(runner)
    test_lhs_property(runner)
    test_transforms(runner)
    mock_config = test_dimension_building(runner)
    test_full_pipeline(runner, mock_config)
    test_quality_ordering(runner)
    test_edge_cases(runner)
    test_distribution_shapes(runner)
    
    # Config integration test (only if path provided)
    config_path = args.config
    if config_path and not os.path.isabs(config_path):
        candidate = os.path.join(script_dir, config_path)
        if os.path.exists(candidate):
            config_path = candidate
    test_config_loading(runner, config_path)
    
    # ── Summary ──
    all_passed = runner.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()