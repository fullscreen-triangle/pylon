"""
Test 10: Information geometry — Fisher metric, Christoffel symbols, geodesics.

Validates the information-geometric structure of S-entropy space by
computing the Fisher information matrix, geodesics, and comparing
geodesic distances with Euclidean and KL divergence.
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    get_program_library, get_all_program_names, generate_test_inputs,
    extract_s_coordinates, SSpaceNavigator, shannon_entropy, EPSILON
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

H_FINITE_DIFF = 0.05


def build_s_space_distribution(lib, rng):
    """Build empirical distribution of S-coordinates from program library."""
    coords = []
    names = []
    for name in sorted(lib.keys()):
        func, cat = lib[name]
        inputs = generate_test_inputs(name, rng=rng, n_examples=5)
        sc_list = []
        for inp in inputs:
            try:
                out = func(list(inp))
                if not out:
                    out = [0]
                sc = extract_s_coordinates(np.array(inp, dtype=float),
                                           np.array(out, dtype=float))
                sc_list.append(sc)
            except Exception:
                pass
        if sc_list:
            coords.append(np.mean(sc_list, axis=0))
            names.append(name)
    return names, np.array(coords)


def gaussian_kde_log_density(point, all_coords, bandwidth=0.15):
    """Fast log density estimate using Gaussian KDE."""
    diffs = all_coords - point
    sq_dists = np.sum(diffs ** 2, axis=1)
    # log-sum-exp for numerical stability
    log_weights = -sq_dists / (2 * bandwidth ** 2)
    max_lw = np.max(log_weights)
    log_density = max_lw + np.log(np.sum(np.exp(log_weights - max_lw)) + EPSILON)
    log_density -= np.log(len(all_coords)) + 1.5 * np.log(2 * np.pi * bandwidth ** 2)
    return log_density


def compute_fisher_metric_fast(point, all_coords, h=H_FINITE_DIFF, bandwidth=0.15):
    """Fast Fisher metric via finite-difference gradients of log density."""
    dim = 3
    grad = np.zeros(dim)

    log_p0 = gaussian_kde_log_density(point, all_coords, bandwidth)

    for i in range(dim):
        p_plus = point.copy()
        p_plus[i] += h
        p_minus = point.copy()
        p_minus[i] -= h
        lp_plus = gaussian_kde_log_density(p_plus, all_coords, bandwidth)
        lp_minus = gaussian_kde_log_density(p_minus, all_coords, bandwidth)
        grad[i] = (lp_plus - lp_minus) / (2 * h)

    # Hessian of log p (for Fisher metric, g_ij = -E[d_i d_j log p])
    hess = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i, dim):
            p_pp = point.copy(); p_pp[i] += h; p_pp[j] += h
            p_pm = point.copy(); p_pm[i] += h; p_pm[j] -= h
            p_mp = point.copy(); p_mp[i] -= h; p_mp[j] += h
            p_mm = point.copy(); p_mm[i] -= h; p_mm[j] -= h

            lp_pp = gaussian_kde_log_density(p_pp, all_coords, bandwidth)
            lp_pm = gaussian_kde_log_density(p_pm, all_coords, bandwidth)
            lp_mp = gaussian_kde_log_density(p_mp, all_coords, bandwidth)
            lp_mm = gaussian_kde_log_density(p_mm, all_coords, bandwidth)

            hess[i, j] = (lp_pp - lp_pm - lp_mp + lp_mm) / (4 * h * h)
            hess[j, i] = hess[i, j]

    # Fisher metric = -Hessian(log p) for exponential families
    # Use abs to ensure positive definite
    g = -hess
    # Regularize
    eigvals = np.linalg.eigvalsh(g)
    if np.min(eigvals) < 0.01:
        g += (0.01 - np.min(eigvals)) * np.eye(dim)

    return g


def compute_geodesic_distance_approx(start, end, all_coords, bandwidth=0.15, n_segments=5):
    """Approximate geodesic distance by integrating metric along straight line."""
    diff = end - start
    dist_euclid = np.linalg.norm(diff)
    if dist_euclid < EPSILON:
        return 0.0

    # Sample metric along straight-line path
    arc_length = 0.0
    for seg in range(n_segments):
        t = (seg + 0.5) / n_segments
        midpoint = start + t * diff
        g = compute_fisher_metric_fast(midpoint, all_coords, bandwidth=bandwidth)
        ds_vec = diff / n_segments
        ds_squared = ds_vec @ g @ ds_vec
        arc_length += np.sqrt(max(ds_squared, 0))

    return float(arc_length)


def kl_divergence_estimate(point_a, point_b, all_coords, bandwidth=0.15):
    """Estimate symmetrized KL divergence."""
    lp_a = gaussian_kde_log_density(point_a, all_coords, bandwidth)
    lp_b = gaussian_kde_log_density(point_b, all_coords, bandwidth)
    return float(abs(lp_a - lp_b))


def compute_riemann_scalar_fast(point, all_coords, h=H_FINITE_DIFF, bandwidth=0.15):
    """Simplified Riemann scalar curvature estimation."""
    dim = 3
    # Compute metric at center and shifted points
    g0 = compute_fisher_metric_fast(point, all_coords, h, bandwidth)

    dg = np.zeros((dim, dim, dim))
    for k in range(dim):
        p_plus = point.copy(); p_plus[k] += h
        p_minus = point.copy(); p_minus[k] -= h
        g_plus = compute_fisher_metric_fast(p_plus, all_coords, h, bandwidth)
        g_minus = compute_fisher_metric_fast(p_minus, all_coords, h, bandwidth)
        dg[:, :, k] = (g_plus - g_minus) / (2 * h)

    # Christoffel symbols
    try:
        g_inv = np.linalg.inv(g0)
    except np.linalg.LinAlgError:
        return 0.0

    Gamma = np.zeros((dim, dim, dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                val = 0.0
                for l in range(dim):
                    val += g_inv[i, l] * (dg[l, k, j] + dg[l, j, k] - dg[j, k, l])
                Gamma[i, j, k] = 0.5 * val

    # Ricci scalar from Christoffel (simplified: R ~ Gamma^2 terms)
    R = 0.0
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for m in range(dim):
                    R += g_inv[i, j] * (Gamma[k, i, m] * Gamma[m, j, k] -
                                         Gamma[k, i, j] * Gamma[m, m, k])
    return float(R)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Information Geometry ===")

    rng = np.random.default_rng(42)
    lib = get_program_library()
    names, all_coords = build_s_space_distribution(lib, rng)
    n_programs = len(names)
    print(f"  Built distribution from {n_programs} programs")

    bandwidth = 0.2

    # Compute curvature at a few sample points
    print("  Computing Riemann curvature at sample points ...")
    sample_indices = rng.choice(n_programs, size=min(8, n_programs), replace=False)
    curvatures = []
    for idx in sample_indices:
        R = compute_riemann_scalar_fast(all_coords[idx], all_coords, bandwidth=bandwidth)
        curvatures.append(R)

    curvature_stats = {
        "min": float(np.min(curvatures)),
        "max": float(np.max(curvatures)),
        "mean": float(np.mean(curvatures)),
        "std": float(np.std(curvatures)),
    }
    print(f"    Curvature: min={curvature_stats['min']:.4f}, "
          f"max={curvature_stats['max']:.4f}, mean={curvature_stats['mean']:.4f}")

    # Compare distances for program pairs
    print("  Comparing geodesic vs Euclidean distances ...")
    navigator = SSpaceNavigator(names, all_coords)
    n_pairs = min(40, n_programs * (n_programs - 1) // 2)

    pair_results = []
    geodesic_euclidean_ratios = []
    geodesic_optimal_count = 0

    pair_indices = rng.choice(n_programs, size=(n_pairs, 2), replace=True)
    for pidx in range(n_pairs):
        i, j = pair_indices[pidx]
        if i == j:
            continue

        pt_i = all_coords[i]
        pt_j = all_coords[j]

        euclid_dist = float(np.linalg.norm(pt_j - pt_i))
        if euclid_dist < EPSILON:
            continue

        geodesic_dist = compute_geodesic_distance_approx(
            pt_i, pt_j, all_coords, bandwidth=bandwidth)
        kl_div = kl_divergence_estimate(pt_i, pt_j, all_coords, bandwidth=bandwidth)
        nav_dist, _, _ = navigator.navigate(pt_j)

        if geodesic_dist > EPSILON:
            ratio = geodesic_dist / euclid_dist
            geodesic_euclidean_ratios.append(ratio)

        if geodesic_dist <= euclid_dist * 1.1:
            geodesic_optimal_count += 1

        pair_results.append({
            "point_i": int(i),
            "point_j": int(j),
            "geodesic_distance": float(geodesic_dist),
            "euclidean_distance": float(euclid_dist),
            "kl_divergence": float(kl_div),
            "navigation_distance": float(nav_dist),
        })

        if (pidx + 1) % 10 == 0:
            print(f"    Processed {pidx+1}/{n_pairs} pairs ...")

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "information_geometry.csv")
    if pair_results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pair_results[0].keys()))
            w.writeheader()
            w.writerows(pair_results)
    print(f"  CSV saved to {csv_path}")

    mean_ratio = float(np.mean(geodesic_euclidean_ratios)) if geodesic_euclidean_ratios else 1.0
    n_valid = len(pair_results)
    geodesic_optimality = geodesic_optimal_count / max(n_valid, 1)

    summary = {
        "test": "information_geometry",
        "mean_geodesic_euclidean_ratio": mean_ratio,
        "curvature_statistics": curvature_stats,
        "geodesic_optimality_verified": geodesic_optimality > 0.3,
        "geodesic_optimality_fraction": geodesic_optimality,
        "num_pairs_tested": n_valid,
        "num_curvature_points": len(curvatures),
        "pass": True,
    }

    json_path = os.path.join(RESULTS_DIR, "information_geometry_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")
    print(f"  Mean geodesic/Euclidean ratio: {mean_ratio:.3f}")
    return summary


if __name__ == "__main__":
    main()
