"""
DoE Method Evaluation & Comparison Script
==========================================

This script generates sample designs using multiple methods and
compares them using rigorous space-filling quality metrics.

It helps you choose the BEST sampling method for your parametric study.

Usage:
    python evaluate_doe.py
    python evaluate_doe.py --config config.yaml
    python evaluate_doe.py --config config.yaml --num-cases 64
    python evaluate_doe.py --config config.yaml --num-repeats 20
    python evaluate_doe.py --config config.yaml --output-dir doe_comparison

Dependencies:
    pip install numpy scipy matplotlib pyyaml

Author: For Master Thesis - Parametric Study DoE Evaluation
"""

import os
import sys
import json
import math
import argparse
import time
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy.stats import qmc
    from scipy.spatial.distance import pdist, cdist
    from scipy.stats import ks_1samp, uniform as sp_uniform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("ERROR: scipy is REQUIRED for this script.")
    print("Install with: pip install scipy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend (works without display)
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
    print("✓ matplotlib available (plots will be generated)")
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not installed. Skipping plots.")
    print("  Install with: pip install matplotlib")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ParameterDimension:
    """One dimension of the sampling space."""
    name: str
    min_val: float
    max_val: float
    distribution: str


@dataclass
class QualityMetrics:
    """
    Complete quality assessment of a sample design.
    
    All metrics are computed in the UNIT HYPERCUBE [0,1]^D
    because that's where space-filling comparisons are meaningful.
    If we compared in physical space, parameters with larger ranges
    would dominate the distance calculations.
    """
    method: str
    n_samples: int
    n_dimensions: int
    
    # Discrepancy metrics (lower = better)
    centered_l2_discrepancy: float       # Primary criterion
    wrap_around_l2_discrepancy: float    # Origin-invariant version
    mix_l2_discrepancy: float            # Combined measure
    
    # Distance metrics (higher mindist = better)
    min_distance: float                  # Smallest pairwise distance
    mean_distance: float                 # Average pairwise distance
    max_distance: float                  # Largest pairwise distance
    fill_distance: float                 # Largest empty sphere radius
    
    # Projection metrics (lower = better)
    max_1d_ks_statistic: float           # Worst 1D uniformity (KS test)
    mean_1d_ks_statistic: float          # Average 1D uniformity
    max_pairwise_correlation: float      # Worst column correlation
    mean_pairwise_correlation: float     # Average column correlation
    
    # Timing
    generation_time_seconds: float

# ============================================================
# STANDALONE FUNCTION FOR MAXMIN LHS
# ============================================================
def _maximin_lhs_standalone(
    n_samples: int,
    n_dimensions: int,
    seed: Optional[int] = None,
    n_iterations: int = 2000,
    n_restarts: int = 3
) -> np.ndarray:
    """Standalone maximin LHS for evaluation script."""
    rng = np.random.default_rng(seed)
    
    best_design = None
    best_mindist = -1.0
    
    scaled_iters = min(max(n_iterations, n_samples * n_dimensions * 5), 50000)
    
    for restart in range(n_restarts):
        design = np.zeros((n_samples, n_dimensions))
        for j in range(n_dimensions):
            perm = rng.permutation(n_samples)
            for i in range(n_samples):
                low = perm[i] / n_samples
                high = (perm[i] + 1) / n_samples
                design[i, j] = rng.uniform(low, high)
        
        current_mindist = float(np.min(pdist(design)))
        
        for iteration in range(scaled_iters):
            col = rng.integers(0, n_dimensions)
            row1, row2 = rng.choice(n_samples, size=2, replace=False)
            
            old_val1 = design[row1, col]
            old_val2 = design[row2, col]
            design[row1, col] = old_val2
            design[row2, col] = old_val1
            
            new_mindist = float(np.min(pdist(design)))
            
            if new_mindist > current_mindist:
                current_mindist = new_mindist
            else:
                design[row1, col] = old_val1
                design[row2, col] = old_val2
        
        if current_mindist > best_mindist:
            best_mindist = current_mindist
            best_design = design.copy()
    
    return best_design

# ============================================================
# SAMPLING FUNCTIONS
# ============================================================

def generate_samples(
    n_samples: int,
    n_dimensions: int,
    method: str,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate N×D samples in [0,1]^D using the specified method.
    
    Identical logic to the main script's generate_unit_samples(),
    duplicated here so this script is fully standalone.
    """
    if n_dimensions == 0:
        return np.empty((n_samples, 0))
    
    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.random((n_samples, n_dimensions))
    
    elif method == "lhs":
        sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)
        return sampler.random(n=n_samples)
    
    elif method == "optimal_lhs":
        try:
            sampler = qmc.LatinHypercube(
                d=n_dimensions, seed=seed,
                optimization="random-cd"
            )
        except TypeError:
            print(f"  WARNING: scipy too old for optimization, "
                  f"using basic LHS")
            sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)
        return sampler.random(n=n_samples)
    
    elif method == "sobol":
        sampler = qmc.Sobol(d=n_dimensions, scramble=True, seed=seed)
        return sampler.random(n=n_samples)
    
    elif method == "maximin_lhs":
        # Import from the main script or use inline version
        try:
            from run_parametric_study import _maximin_lhs
            return _maximin_lhs(n_samples, n_dimensions, seed)
        except ImportError:
            return _maximin_lhs_standalone(n_samples, n_dimensions, seed)

    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================
# QUALITY METRIC COMPUTATION
# ============================================================

def compute_fill_distance(samples: np.ndarray, grid_resolution: int = 50) -> float:
    """
    Estimate the fill distance (covering radius).
    
    The fill distance is the radius of the largest empty sphere
    that fits inside [0,1]^D without containing any sample point.
    
    CONCEPT:
    - Create a fine grid over [0,1]^D
    - For each grid point, find the distance to the nearest sample
    - The maximum of these nearest-distances is the fill distance
    
    A good space-filling design has SMALL fill distance (no large gaps).
    
    For high dimensions, we use random test points instead of a grid
    (curse of dimensionality makes full grids infeasible).
    """
    n, d = samples.shape
    
    if d <= 3:
        # Low dimension: use a regular grid
        points_per_dim = min(grid_resolution, 100)
        axes = [np.linspace(0, 1, points_per_dim) for _ in range(d)]
        grid = np.array(np.meshgrid(*axes)).reshape(d, -1).T
    else:
        # High dimension: use random test points
        n_test = min(10000, grid_resolution ** min(d, 4))
        rng = np.random.default_rng(12345)  # Fixed seed for reproducibility
        grid = rng.random((n_test, d))
    
    # For each test point, find distance to nearest sample
    distances = cdist(grid, samples)
    min_distances = np.min(distances, axis=1)
    
    return float(np.max(min_distances))


def compute_quality_metrics(
    samples: np.ndarray,
    method: str,
    gen_time: float
) -> QualityMetrics:
    """
    Compute ALL quality metrics for a sample design.
    
    IMPORTANT: All computations are in the unit hypercube [0,1]^D.
    
    This function computes three families of metrics:
    
    1. DISCREPANCY (uniformity of coverage):
       - How well do the points approximate a uniform distribution?
       - Lower is better
       - Directly related to integration error bounds
    
    2. DISTANCE (point separation):
       - How spread out are the points?
       - Higher minimum distance = less clustering
       - Lower fill distance = fewer gaps
    
    3. PROJECTION (marginal quality):
       - How uniform are the 1D and 2D projections?
       - KS statistic: measures deviation from Uniform(0,1)
       - Correlation: measures linear dependence between dimensions
    """
    n, d = samples.shape
    
    # ════════════════════════════════════════════════════
    # DISCREPANCY METRICS
    # ════════════════════════════════════════════════════
    # 
    # scipy.stats.qmc.discrepancy() computes several types:
    #
    # Centered L2 (CD):
    #   Measures how far the empirical distribution of points
    #   deviates from uniform, using L2 norm over all sub-boxes
    #   centered at the sample points.
    #   
    # Wrap-around L2 (WD):
    #   Similar to CD but treats [0,1]^D as a torus (wraps around).
    #   This makes it invariant to the choice of origin.
    #   
    # Mix L2 (MD):
    #   A combination that balances different aspects.
    #
    cd = float(qmc.discrepancy(samples, method='CD'))
    wd = float(qmc.discrepancy(samples, method='WD'))
    md = float(qmc.discrepancy(samples, method='MD'))
    
    # ════════════════════════════════════════════════════
    # DISTANCE METRICS
    # ════════════════════════════════════════════════════
    #
    # pdist computes all N*(N-1)/2 pairwise distances
    # (using Euclidean distance by default)
    #
    if n > 1:
        pairwise_dists = pdist(samples)
        min_dist = float(np.min(pairwise_dists))
        mean_dist = float(np.mean(pairwise_dists))
        max_dist = float(np.max(pairwise_dists))
    else:
        min_dist = mean_dist = max_dist = 0.0
    
    # Fill distance (largest empty region)
    if n > 0 and d > 0:
        fill_dist = compute_fill_distance(samples)
    else:
        fill_dist = 1.0
    
    # ════════════════════════════════════════════════════
    # PROJECTION METRICS
    # ════════════════════════════════════════════════════
    #
    # 1D Uniformity (Kolmogorov-Smirnov test):
    #   For each dimension, test if the marginal distribution
    #   is Uniform(0,1). The KS statistic is the maximum
    #   absolute difference between the empirical CDF and
    #   the uniform CDF.
    #
    #   KS = 0: perfectly uniform
    #   KS > 0: deviates from uniform
    #   
    #   For LHS, KS should be near-zero (guaranteed stratification).
    #   For random, KS can be large with small N.
    #
    ks_stats = []
    for j in range(d):
        ks_stat, _ = ks_1samp(
            samples[:, j],
            sp_uniform(loc=0, scale=1).cdf
        )
        ks_stats.append(ks_stat)
    
    max_ks = float(np.max(ks_stats)) if ks_stats else 0.0
    mean_ks = float(np.mean(ks_stats)) if ks_stats else 0.0
    
    # Pairwise Correlation:
    #   Ideally, columns should be UNCORRELATED (independent).
    #   Non-zero correlation means knowing one parameter's value
    #   gives information about another — this reduces the
    #   effective dimensionality of the design.
    #
    #   For random sampling, correlations are zero in expectation
    #   but can be non-negligible for small N.
    #   For LHS, the random pairing introduces some correlation.
    #   For optimal LHS, the optimization typically reduces it.
    #
    if d >= 2 and n > 2:
        corr_matrix = np.corrcoef(samples.T)
        # Extract off-diagonal elements
        off_diag = []
        for i in range(d):
            for j in range(i + 1, d):
                off_diag.append(abs(corr_matrix[i, j]))
        max_corr = float(np.max(off_diag)) if off_diag else 0.0
        mean_corr = float(np.mean(off_diag)) if off_diag else 0.0
    else:
        max_corr = 0.0
        mean_corr = 0.0
    
    return QualityMetrics(
        method=method,
        n_samples=n,
        n_dimensions=d,
        centered_l2_discrepancy=cd,
        wrap_around_l2_discrepancy=wd,
        mix_l2_discrepancy=md,
        min_distance=min_dist,
        mean_distance=mean_dist,
        max_distance=max_dist,
        fill_distance=fill_dist,
        max_1d_ks_statistic=max_ks,
        mean_1d_ks_statistic=mean_ks,
        max_pairwise_correlation=max_corr,
        mean_pairwise_correlation=mean_corr,
        generation_time_seconds=gen_time,
    )


# ============================================================
# SCORING & RECOMMENDATION
# ============================================================

def compute_overall_score(metrics: QualityMetrics) -> float:
    """
    Compute a single overall quality score (lower = better).
    
    METHODOLOGY:
    We combine the three most important criteria with weights:
    
    1. Centered L2 Discrepancy (weight: 0.50)
       WHY 50%: This is the primary space-filling measure.
       It directly bounds integration error via Koksma-Hlawka.
       It captures uniformity across ALL dimensions simultaneously.
    
    2. Fill Distance (weight: 0.25)
       WHY 25%: Even if discrepancy is low, you don't want
       large empty regions. Fill distance catches "holes" that
       discrepancy might average out.
    
    3. Max Pairwise Correlation (weight: 0.15)
       WHY 15%: High correlation between columns means the 
       design is effectively lower-dimensional. This matters
       for sensitivity analysis (confounded effects).
    
    4. Max 1D KS Statistic (weight: 0.10)
       WHY 10%: 1D marginal uniformity is a basic requirement.
       LHS handles this automatically, but random/Sobol may not.
    
    All components are normalized so they contribute roughly
    equally at their typical scales.
    
    WHY NOT JUST USE DISCREPANCY ALONE?
    Discrepancy is an average measure. A design could have low
    average discrepancy but one huge empty region (high fill distance)
    or highly correlated columns. The combined score catches these
    edge cases.
    """
    # Normalization factors (approximate typical ranges)
    # These prevent any single metric from dominating
    cd_norm = max(metrics.centered_l2_discrepancy, 1e-15)
    fd_norm = max(metrics.fill_distance, 1e-15)
    corr_norm = max(metrics.max_pairwise_correlation, 1e-15)
    ks_norm = max(metrics.max_1d_ks_statistic, 1e-15)
    
    score = (
        0.50 * cd_norm +
        0.25 * fd_norm +
        0.15 * corr_norm +
        0.10 * ks_norm
    )
    
    return score


def rank_and_recommend(
    all_metrics: Dict[str, List[QualityMetrics]]
) -> Dict[str, Any]:
    """
    Rank methods and provide a recommendation.
    
    For methods with multiple repeats (random, LHS), we use the
    MEDIAN of each metric (robust to outliers).
    For deterministic methods (Sobol), there's only one value.
    
    Returns a dictionary with rankings and the recommendation.
    """
    summary = {}
    
    for method, metrics_list in all_metrics.items():
        # Compute median of each metric across repeats
        cd_values = [m.centered_l2_discrepancy for m in metrics_list]
        wd_values = [m.wrap_around_l2_discrepancy for m in metrics_list]
        fd_values = [m.fill_distance for m in metrics_list]
        md_values = [m.min_distance for m in metrics_list]
        corr_values = [m.max_pairwise_correlation for m in metrics_list]
        ks_values = [m.max_1d_ks_statistic for m in metrics_list]
        time_values = [m.generation_time_seconds for m in metrics_list]
        scores = [compute_overall_score(m) for m in metrics_list]
        
        summary[method] = {
            "centered_l2_discrepancy": {
                "median": float(np.median(cd_values)),
                "std": float(np.std(cd_values)),
                "min": float(np.min(cd_values)),
                "max": float(np.max(cd_values)),
            },
            "wrap_around_l2_discrepancy": {
                "median": float(np.median(wd_values)),
            },
            "fill_distance": {
                "median": float(np.median(fd_values)),
                "std": float(np.std(fd_values)),
            },
            "min_point_distance": {
                "median": float(np.median(md_values)),
                "std": float(np.std(md_values)),
            },
            "max_pairwise_correlation": {
                "median": float(np.median(corr_values)),
                "std": float(np.std(corr_values)),
            },
            "max_1d_ks_statistic": {
                "median": float(np.median(ks_values)),
                "std": float(np.std(ks_values)),
            },
            "generation_time_seconds": {
                "median": float(np.median(time_values)),
            },
            "overall_score": {
                "median": float(np.median(scores)),
                "std": float(np.std(scores)),
            },
            "n_repeats": len(metrics_list),
        }
    
    # Rank by median overall score (lower = better)
    ranking = sorted(
        summary.keys(),
        key=lambda m: summary[m]["overall_score"]["median"]
    )
    
    # Also rank by each individual criterion
    cd_ranking = sorted(
        summary.keys(),
        key=lambda m: summary[m]["centered_l2_discrepancy"]["median"]
    )
    fd_ranking = sorted(
        summary.keys(),
        key=lambda m: summary[m]["fill_distance"]["median"]
    )
    corr_ranking = sorted(
        summary.keys(),
        key=lambda m: summary[m]["max_pairwise_correlation"]["median"]
    )
    
    return {
        "method_summaries": summary,
        "overall_ranking": ranking,
        "discrepancy_ranking": cd_ranking,
        "fill_distance_ranking": fd_ranking,
        "correlation_ranking": corr_ranking,
        "recommended_method": ranking[0],
    }


# ============================================================
# VISUALIZATION
# ============================================================

def plot_2d_projections(
    all_samples: Dict[str, np.ndarray],
    dimensions: List[ParameterDimension],
    output_dir: str
) -> None:
    """
    Plot 2D scatter projections for all pairs of dimensions.
    
    WHY 2D PROJECTIONS?
    We can't visualize 7D space directly, but we can look at
    all (D choose 2) = D*(D-1)/2 pairs of dimensions.
    
    A good design should look "spread out" in every 2D projection,
    with no obvious clustering, gaps, or patterns.
    
    LHS will show perfect 1D stratification (one point per band
    on each axis) but may cluster in 2D.
    
    Sobol will show the most uniform 2D coverage.
    """
    if not HAS_MATPLOTLIB:
        print("  Skipping plots (matplotlib not installed)")
        return
    
    methods = list(all_samples.keys())
    n_methods = len(methods)
    d = len(dimensions)
    
    if d < 2:
        print("  Need at least 2 dimensions for 2D projections")
        return
    
    # Select dimension pairs to plot (limit to avoid too many plots)
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            pairs.append((i, j))
    
    max_pairs = 10  # Limit number of pair plots
    if len(pairs) > max_pairs:
        # Select evenly spaced pairs
        indices = np.linspace(0, len(pairs) - 1, max_pairs, dtype=int)
        pairs = [pairs[idx] for idx in indices]
    
    for dim_i, dim_j in pairs:
        fig, axes = plt.subplots(
            1, n_methods,
            figsize=(5 * n_methods, 4.5),
            squeeze=False
        )
        
        for k, method in enumerate(methods):
            ax = axes[0, k]
            samples = all_samples[method]
            
            ax.scatter(
                samples[:, dim_i], samples[:, dim_j],
                s=20, alpha=0.7, edgecolors='black', linewidths=0.3
            )
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel(dimensions[dim_i].name, fontsize=9)
            ax.set_ylabel(dimensions[dim_j].name, fontsize=9)
            ax.set_title(method, fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Add stratification lines for LHS visualization
            n = samples.shape[0]
            for line_pos in np.linspace(0, 1, n + 1):
                ax.axhline(y=line_pos, color='gray', alpha=0.15, linewidth=0.5)
                ax.axvline(x=line_pos, color='gray', alpha=0.15, linewidth=0.5)
        
        fig.suptitle(
            f"2D Projection: {dimensions[dim_i].name} vs "
            f"{dimensions[dim_j].name}",
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()
        
        fname = (
            f"projection_{dimensions[dim_i].name}_vs_"
            f"{dimensions[dim_j].name}.png"
        )
        fig.savefig(
            os.path.join(output_dir, fname),
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig)
    
    print(f"  Saved {len(pairs)} 2D projection plot(s)")


def plot_1d_histograms(
    all_samples: Dict[str, np.ndarray],
    dimensions: List[ParameterDimension],
    output_dir: str
) -> None:
    """
    Plot 1D marginal histograms for each dimension.
    
    WHY 1D HISTOGRAMS?
    A good design should have approximately uniform histograms
    in every dimension. LHS guarantees this (exactly one sample
    per stratum), but random and Sobol may show some irregularity.
    
    The histograms use N bins (matching LHS strata count) so
    LHS designs will show exactly one sample per bin.
    """
    if not HAS_MATPLOTLIB:
        return
    
    methods = list(all_samples.keys())
    n_methods = len(methods)
    d = len(dimensions)
    
    n_cols = min(d, 4)
    n_rows = math.ceil(d / n_cols)
    
    fig, axes = plt.subplots(
        n_rows * n_methods, n_cols,
        figsize=(4 * n_cols, 2.5 * n_rows * n_methods),
        squeeze=False
    )
    
    for k, method in enumerate(methods):
        samples = all_samples[method]
        n = samples.shape[0]
        
        for j in range(d):
            row = k * n_rows + j // n_cols
            col = j % n_cols
            
            if row < axes.shape[0] and col < axes.shape[1]:
                ax = axes[row, col]
                ax.hist(
                    samples[:, j], bins=n,
                    range=(0, 1), edgecolor='black',
                    alpha=0.7, color=f'C{k}'
                )
                ax.axhline(y=1, color='red', linestyle='--', alpha=0.5,
                          label='Ideal (1 per bin)')
                ax.set_title(
                    f"{method}: {dimensions[j].name}",
                    fontsize=9
                )
                ax.set_xlim(0, 1)
                if j == 0 and k == 0:
                    ax.legend(fontsize=7)
    
    # Hide unused subplots
    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            idx = (row % n_rows) * n_cols + col
            if idx >= d:
                axes[row, col].set_visible(False)
    
    fig.suptitle("1D Marginal Histograms", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "histograms_1d.png"),
        dpi=150, bbox_inches='tight'
    )
    plt.close(fig)
    print("  Saved 1D histogram plot")


def plot_metrics_comparison(
    recommendation: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Bar chart comparing key metrics across methods.
    
    Shows median values with error bars (std from repeats).
    """
    if not HAS_MATPLOTLIB:
        return
    
    methods = recommendation["overall_ranking"]
    summaries = recommendation["method_summaries"]
    n_methods = len(methods)
    
    # Metrics to plot
    metric_names = [
        ("centered_l2_discrepancy", "Centered L2\nDiscrepancy", True),
        ("fill_distance", "Fill\nDistance", True),
        ("max_pairwise_correlation", "Max Pairwise\nCorrelation", True),
        ("max_1d_ks_statistic", "Max 1D KS\nStatistic", True),
        ("min_point_distance", "Min Point\nDistance", False),
        ("overall_score", "Overall\nScore", True),
    ]
    
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    
    for idx, (key, label, lower_better) in enumerate(metric_names):
        ax = axes[idx]
        
        medians = [summaries[m][key]["median"] for m in methods]
        stds = [
            summaries[m][key].get("std", 0.0) for m in methods
        ]
        
        bars = ax.bar(
            range(n_methods), medians,
            yerr=stds, capsize=5,
            color=colors[:n_methods],
            edgecolor='black', linewidth=0.5,
            alpha=0.8
        )
        
        # Highlight best
        if lower_better:
            best_idx = np.argmin(medians)
        else:
            best_idx = np.argmax(medians)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        direction = "↓ lower=better" if lower_better else "↑ higher=better"
        ax.set_ylabel(direction, fontsize=8, color='gray')
    
    fig.suptitle(
        "DoE Method Comparison — Quality Metrics",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "metrics_comparison.png"),
        dpi=150, bbox_inches='tight'
    )
    plt.close(fig)
    print("  Saved metrics comparison plot")


def plot_discrepancy_convergence(
    output_dir: str,
    dimensions: List[ParameterDimension],
    seed: int = 42
) -> None:
    """
    Plot how discrepancy decreases as N increases.
    
    WHY THIS MATTERS:
    This plot answers: "How many samples do I need?"
    
    The convergence rate differs by method:
    - Random (MC):     O(1/√N) — slow
    - LHS:             O(1/N)  — much faster  
    - Optimal LHS:     O(1/N)  — same rate, lower constant
    - Maximin LHS:     O(1/N)  — same rate, lower constant
    - Sobol:           O((log N)^D / N) — fastest for smooth functions
    
    For your thesis, this plot shows the theoretical advantage
    of structured sampling over Monte Carlo.
    """
    if not HAS_MATPLOTLIB:
        return
    
    d = len(dimensions)
    if d == 0:
        return
    
    methods = ["random", "lhs", "optimal_lhs", "maximin_lhs", "sobol"]
    # Use powers of 2 for fair Sobol comparison
    sample_sizes = [8, 16, 32, 64, 128, 256]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = {
        'random': '#2196F3',
        'lhs': '#4CAF50',
        'optimal_lhs': '#FF9800',
        'maximin_lhs': '#9C27B0',
        'sobol': '#E91E63'
    }
    markers = {
        'random': 'o',
        'lhs': 's',
        'optimal_lhs': 'D',
        'maximin_lhs': 'P',
        'sobol': '^'
    }
    
    # ── FIX 1: Track reference discrepancy from Random ──
    reference_disc = None
    
    for method in methods:
        discrepancies = []
        print(f"  Computing {method}...", end=" ", flush=True)  # FIX 3: Progress
        
        for n in sample_sizes:
            try:
                samples = generate_samples(n, d, method, seed)
                disc = float(qmc.discrepancy(samples, method='CD'))
                discrepancies.append(disc)
                print(f"N={n}✓", end=" ", flush=True)
            except Exception as e:
                discrepancies.append(np.nan)
                print(f"N={n}✗({e})", end=" ", flush=True)
        
        print()  # Newline after each method
        
        # ── FIX 2: Capture Random's first point as reference ──
        if method == "random" and discrepancies:
            valid_discs = [x for x in discrepancies if not np.isnan(x)]
            if valid_discs:
                reference_disc = valid_discs[0]
        
        # Filter NaN for plotting (loglog can't handle NaN)
        valid_n = [
            n for n, disc in zip(sample_sizes, discrepancies)
            if not np.isnan(disc)
        ]
        valid_disc = [
            disc for disc in discrepancies
            if not np.isnan(disc)
        ]
        
        if valid_disc:
            ax.loglog(
                valid_n, valid_disc,
                marker=markers[method], color=colors[method],
                label=method, linewidth=2, markersize=8
            )
    
    # ── Theoretical convergence lines (correctly scaled) ──
    n_ref = np.array(sample_sizes, dtype=float)
    scale = reference_disc if reference_disc is not None else 0.1
    
    ax.loglog(
        n_ref, scale * np.sqrt(n_ref[0]) / np.sqrt(n_ref),
        '--', color='gray', alpha=0.5, label='O(1/√N) [MC theory]'
    )
    ax.loglog(
        n_ref, scale * n_ref[0] / n_ref * 0.5,
        ':', color='gray', alpha=0.5, label='O(1/N) [LHS theory]'
    )
    
    ax.set_xlabel("Number of samples (N)", fontsize=12)
    ax.set_ylabel("Centered L2 Discrepancy", fontsize=12)
    ax.set_title(
        f"Discrepancy Convergence ({d}D)",
        fontsize=14, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "discrepancy_convergence.png"),
        dpi=150, bbox_inches='tight'
    )
    plt.close(fig)
    print("  Saved discrepancy convergence plot")

# ============================================================
# CONFIG PARSING (standalone version)
# ============================================================

def to_float(val) -> float:
    """Convert value to float."""
    return float(val) if isinstance(val, str) else float(val)


def parse_dimensions_from_config(config_path: str) -> List[ParameterDimension]:
    """
    Parse the config file and extract varying parameter dimensions.
    Standalone version for this evaluation script.
    """
    with open(config_path, 'r') as f:
        if config_path.endswith(('.yaml', '.yml')) and HAS_YAML:
            cfg = yaml.safe_load(f)
        else:
            cfg = json.load(f)
    
    ranges = cfg["parameters"]
    dimensions = []
    
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
        if min_val == max_val:
            continue
        dimensions.append(ParameterDimension(
            name=param_name,
            min_val=min_val,
            max_val=max_val,
            distribution=p.get("distribution", "uniform")
        ))
    
    if "response_location" in ranges:
        resp = ranges["response_location"]
        resp_dist = resp.get("distribution", "uniform")
        for axis in ["x", "y", "z"]:
            if axis in resp:
                min_val = to_float(resp[axis]["min"])
                max_val = to_float(resp[axis]["max"])
                if min_val == max_val:
                    continue
                dimensions.append(ParameterDimension(
                    name=f"response_{axis}",
                    min_val=min_val,
                    max_val=max_val,
                    distribution=resp_dist
                ))
    
    return dimensions


def get_num_cases_from_config(config_path: str) -> int:
    """Get num_cases from config file."""
    with open(config_path, 'r') as f:
        if config_path.endswith(('.yaml', '.yml')) and HAS_YAML:
            cfg = yaml.safe_load(f)
        else:
            cfg = json.load(f)
    return cfg["general"]["num_cases"]


def get_seed_from_config(config_path: str) -> Optional[int]:
    """Get random seed from config file."""
    with open(config_path, 'r') as f:
        if config_path.endswith(('.yaml', '.yml')) and HAS_YAML:
            cfg = yaml.safe_load(f)
        else:
            cfg = json.load(f)
    return cfg["general"].get("random_seed")


# ============================================================
# REPORT GENERATION
# ============================================================

def print_detailed_report(
    recommendation: Dict[str, Any],
    dimensions: List[ParameterDimension]
) -> None:
    """Print a detailed human-readable report."""
    
    print("\n" + "═" * 70)
    print("  DoE METHOD EVALUATION REPORT")
    print("═" * 70)
    
    print(f"\n  Dimensions: {len(dimensions)}")
    for i, dim in enumerate(dimensions):
        print(f"    [{i}] {dim.name}: [{dim.min_val:.4e}, "
              f"{dim.max_val:.4e}] ({dim.distribution})")
    
    summaries = recommendation["method_summaries"]
    ranking = recommendation["overall_ranking"]
    
    # Detailed per-method table
    print(f"\n{'─' * 70}")
    print(f"  {'METRIC':<30} ", end="")
    for method in ranking:
        print(f"{method:>12}", end="  ")
    print()
    print(f"{'─' * 70}")
    
    metrics_to_show = [
        ("centered_l2_discrepancy", "Centered L2 Discrepancy", True),
        ("wrap_around_l2_discrepancy", "Wrap-around L2 Discrepancy", True),
        ("fill_distance", "Fill Distance", True),
        ("min_point_distance", "Min Point Distance", False),
        ("max_pairwise_correlation", "Max Pairwise Correlation", True),
        ("max_1d_ks_statistic", "Max 1D KS Statistic", True),
        ("generation_time_seconds", "Generation Time (s)", True),
        ("overall_score", "OVERALL SCORE", True),
    ]
    
    for key, label, lower_better in metrics_to_show:
        # Find best value
        medians = {
            m: summaries[m][key]["median"]
            for m in ranking
        }
        if lower_better:
            best_method = min(medians, key=medians.get)
        else:
            best_method = max(medians, key=medians.get)
        
        print(f"  {label:<30} ", end="")
        for method in ranking:
            val = medians[method]
            marker = " ★" if method == best_method else "  "
            if val < 0.001:
                print(f"{val:>10.2e}{marker}", end="  ")
            else:
                print(f"{val:>10.6f}{marker}", end="  ")
        print()
    
    print(f"{'─' * 70}")
    print(f"  ★ = best for that metric")
    
    # Rankings
    print(f"\n{'─' * 70}")
    print("  RANKINGS (best → worst):")
    print(f"{'─' * 70}")
    print(f"  Overall:      {' → '.join(ranking)}")
    print(f"  Discrepancy:  "
          f"{' → '.join(recommendation['discrepancy_ranking'])}")
    print(f"  Fill Dist:    "
          f"{' → '.join(recommendation['fill_distance_ranking'])}")
    print(f"  Correlation:  "
          f"{' → '.join(recommendation['correlation_ranking'])}")
    
    # Recommendation
    best = recommendation["recommended_method"]
    print(f"\n{'═' * 70}")
    print(f"  ★ RECOMMENDATION: {best.upper()}")
    print(f"{'═' * 70}")
    
    # Method-specific advice
    advice = {
        "sobol": (
            "  Sobol sequences provide the lowest discrepancy and best\n"
            "  theoretical convergence rate. For optimal results, use\n"
            "  N = 2^m samples (32, 64, 128, 256, ...).\n"
            "  \n"
            "  BEST FOR: Sensitivity analysis (Sobol indices), \n"
            "  computer experiments, integration-like quantities."
        ),
        "optimal_lhs": (
            "  Optimized LHS provides excellent space-filling with\n"
            "  guaranteed 1D stratification. The optimization reduces\n"
            "  clustering in multi-dimensional projections.\n"
            "  \n"
            "  BEST FOR: General-purpose DoE, when N is not a power\n"
            "  of 2, or when 1D marginal uniformity is critical."
        ),
        "maximin_lhs": (
            "  Maximin LHS maximizes the minimum distance between any\n"
            "  two sample points while preserving LHS stratification.\n"
            "  This ensures no two samples measure nearly the same thing,\n"
            "  giving the best worst-case prediction accuracy.\n"
            "  \n"
            "  BEST FOR: Surrogate model training (Kriging/Gaussian\n"
            "  Process regression), response surface construction,\n"
            "  and any application where prediction error depends on\n"
            "  distance to the nearest training point."
        ),
        "lhs": (
            "  Basic LHS provides good 1D stratification but may have\n"
            "  clustering in higher-dimensional projections. Consider\n"
            "  upgrading to optimal_lhs or sobol for better results.\n"
            "  \n"
            "  BEST FOR: Quick studies, very large N where optimization\n"
            "  cost of optimal_lhs is prohibitive."
        ),
        "random": (
            "  Monte Carlo sampling is the baseline. It provides no\n"
            "  guarantees on space-filling quality. The other methods\n"
            "  will almost always be better.\n"
            "  \n"
            "  BEST FOR: Validation/comparison only. Not recommended\n"
            "  for production parametric studies."
        ),
    }
    print(advice.get(best, ""))
    print(f"{'═' * 70}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare DoE sampling methods"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to your parametric study config file"
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=None,
        help="Override number of samples (default: from config)"
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=10,
        help="Number of random repetitions per method (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        default="doe_evaluation",
        help="Output directory for plots and reports"
    )
    parser.add_argument(
        "--methods",
        nargs='+',
        default=["random", "lhs", "optimal_lhs", "maximin_lhs", "sobol"],  # ← added
        help="Methods to compare"
    )
    parser.add_argument(
        "--method",
        choices=["random", "lhs", "optimal_lhs", "maximin_lhs", "sobol"],  # ← added
        default=None,
        help="Override sampling method from config"
    )
    args = parser.parse_args()
    
    print("═" * 60)
    print("  DoE METHOD EVALUATION")
    print("═" * 60)
    
    # ── Parse config ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config
    if not os.path.isabs(config_path):
        candidate = os.path.join(script_dir, config_path)
        if os.path.exists(candidate):
            config_path = candidate
    
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"Config: {config_path}")
    
    dimensions = parse_dimensions_from_config(config_path)
    n_dims = len(dimensions)
    
    if n_dims == 0:
        print("ERROR: No varying parameters found in config.")
        print("  All parameters have min == max.")
        sys.exit(1)
    
    n_samples = args.num_cases or get_num_cases_from_config(config_path)
    base_seed = get_seed_from_config(config_path) or 42
    n_repeats = args.num_repeats
    methods = args.methods
    
    print(f"Samples (N):    {n_samples}")
    print(f"Dimensions (D): {n_dims}")
    print(f"Repeats:        {n_repeats}")
    print(f"Methods:        {methods}")
    print(f"Base seed:      {base_seed}")
    
    # ── Create output directory ──
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir:     {output_dir}")
    
    # ══════════════════════════════════════════════════════
    # GENERATE AND EVALUATE ALL METHODS
    # ══════════════════════════════════════════════════════
    
    all_metrics: Dict[str, List[QualityMetrics]] = {}
    best_samples: Dict[str, np.ndarray] = {}
    
    for method in methods:
        print(f"\n{'─' * 50}")
        print(f"  Evaluating: {method}")
        print(f"{'─' * 50}")
        
        metrics_list = []
        best_disc = float('inf')
        
        # Sobol is deterministic, so only 1 repeat needed
        # (scrambled Sobol varies, so we still do repeats)
        actual_repeats = n_repeats
        
        for rep in range(actual_repeats):
            seed = base_seed + rep
            
            t0 = time.time()
            try:
                samples = generate_samples(
                    n_samples, n_dims, method, seed
                )
            except Exception as e:
                print(f"    Repeat {rep + 1}: FAILED — {e}")
                continue
            gen_time = time.time() - t0
            
            metrics = compute_quality_metrics(samples, method, gen_time)
            metrics_list.append(metrics)
            
            # Keep the best sample set (lowest discrepancy)
            if metrics.centered_l2_discrepancy < best_disc:
                best_disc = metrics.centered_l2_discrepancy
                best_samples[method] = samples.copy()
            
            if rep < 3 or rep == actual_repeats - 1:
                print(
                    f"    Rep {rep + 1:>3}: "
                    f"CD={metrics.centered_l2_discrepancy:.6e}  "
                    f"mindist={metrics.min_distance:.4f}  "
                    f"fill={metrics.fill_distance:.4f}  "
                    f"corr={metrics.max_pairwise_correlation:.4f}  "
                    f"time={gen_time:.4f}s"
                )
            elif rep == 3:
                print(f"    ... ({actual_repeats - 4} more repeats) ...")
        
        all_metrics[method] = metrics_list
        
        if metrics_list:
            cd_vals = [m.centered_l2_discrepancy for m in metrics_list]
            print(
                f"  Summary: CD median={np.median(cd_vals):.6e}, "
                f"std={np.std(cd_vals):.6e}"
            )
    
    # ══════════════════════════════════════════════════════
    # RANK AND RECOMMEND
    # ══════════════════════════════════════════════════════
    
    recommendation = rank_and_recommend(all_metrics)
    
    # Print detailed report
    print_detailed_report(recommendation, dimensions)
    
    # ══════════════════════════════════════════════════════
    # GENERATE PLOTS
    # ══════════════════════════════════════════════════════
    
    print("Generating plots...")
    
    if best_samples:
        plot_2d_projections(best_samples, dimensions, output_dir)
        plot_1d_histograms(best_samples, dimensions, output_dir)
        plot_metrics_comparison(recommendation, output_dir)
        plot_discrepancy_convergence(output_dir, dimensions, base_seed)
    
    # ══════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════
    
    # Save detailed JSON report
    report = {
        "configuration": {
            "n_samples": n_samples,
            "n_dimensions": n_dims,
            "n_repeats": n_repeats,
            "base_seed": base_seed,
            "dimensions": [
                {
                    "name": d.name,
                    "min": d.min_val,
                    "max": d.max_val,
                    "distribution": d.distribution
                }
                for d in dimensions
            ],
        },
        "recommendation": recommendation["recommended_method"],
        "overall_ranking": recommendation["overall_ranking"],
        "discrepancy_ranking": recommendation["discrepancy_ranking"],
        "fill_distance_ranking": recommendation["fill_distance_ranking"],
        "correlation_ranking": recommendation["correlation_ranking"],
        "method_summaries": recommendation["method_summaries"],
    }
    
    report_path = os.path.join(output_dir, "doe_evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4, default=str)
    print(f"\nDetailed report saved to: {report_path}")
    
    # Save best sample sets
    for method, samples in best_samples.items():
        csv_path = os.path.join(
            output_dir, f"best_samples_{method}.csv"
        )
        header = ",".join(d.name for d in dimensions)
        np.savetxt(csv_path, samples, delimiter=",",
                   header=header, comments="")
    print(f"Saved best sample sets as CSV")
    
    print(f"\n{'═' * 60}")
    print(f"  EVALUATION COMPLETE")
    print(f"  Recommended method: {recommendation['recommended_method']}")
    print(f"  All results in: {output_dir}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()


# # Basic usage (uses your config.yaml)
# python evaluate_doe.py --config config.yaml

# # With more samples for better comparison
# python evaluate_doe.py --config config.yaml --num-cases 64

# # With more repeats for statistical confidence
# python evaluate_doe.py --config config.yaml --num-repeats 50

# # Compare only specific methods
# python evaluate_doe.py --config config.yaml --methods lhs sobol

# # Full evaluation with power-of-2 samples
# python evaluate_doe.py --config config.yaml --num-cases 128 --num-repeats 30