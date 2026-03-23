"""
Test 7: Fiber bundle structure — parallel transport and holonomy.

Validates fiber bundle structure with gear ratios as connection,
testing transitivity, parallel transport, and curvature.
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import EPSILON

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

GRID_SIZE = 10  # 10x10x10 = 1000 points in base space
N_PARTITION_LEVELS = 4  # Fiber: partition quantum numbers n=1..4


def create_base_space(grid_size):
    """Create grid of points in S-entropy space [0,1]^3."""
    axis = np.linspace(0, 1, grid_size)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return points


def compute_fiber_state(base_point, n):
    """Compute fiber state (partition coordinates) at a base point.
    Returns (n, l, m, s) quantum-number-like coordinates."""
    sk, st, se = base_point
    # Partition coordinates depend on base point and level
    ell = int(np.floor(sk * n)) % n  # angular momentum analogue
    m = int(np.floor(st * (2 * ell + 1))) - ell  # magnetic number
    m = max(-ell, min(ell, m))
    s = 1 if se > 0.5 else -1  # spin
    return np.array([n, ell, m, s], dtype=np.float64)


def connection_coefficient(point_a, point_b):
    """Compute connection (gear ratio) between adjacent base points.
    Returns a 4x4 transport matrix."""
    delta = point_b - point_a
    norm = np.linalg.norm(delta) + EPSILON

    # Connection = I + A_mu * dx^mu (first order)
    # A_mu encodes how fiber rotates along base direction mu
    A = np.zeros((4, 4))
    for mu in range(3):
        # Antisymmetric generator
        A[0, 1] += delta[mu] * (mu + 1) * 0.1
        A[1, 0] -= delta[mu] * (mu + 1) * 0.1
        A[2, 3] += delta[mu] * (mu + 1) * 0.05
        A[3, 2] -= delta[mu] * (mu + 1) * 0.05

    # Transport matrix: exp(A) approx I + A for small delta
    transport = np.eye(4) + A
    return transport


def parallel_transport(state, path_points):
    """Transport a fiber state along a path of base points."""
    current = state.copy()
    for i in range(len(path_points) - 1):
        T = connection_coefficient(path_points[i], path_points[i + 1])
        current = T @ current
    return current


def compute_gear_ratio(freq_i, freq_j):
    """Scalar gear ratio."""
    if abs(freq_j) < EPSILON:
        return 0.0
    return freq_i / freq_j


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Fiber Bundle Structure ===")

    rng = np.random.default_rng(42)
    points = create_base_space(GRID_SIZE)
    n_points = len(points)
    print(f"  Base space: {n_points} points")

    # Assign frequencies to points (for gear ratio computation)
    frequencies = 1.0 + np.linalg.norm(points, axis=1) * 0.5 + rng.normal(0, 0.01, n_points)

    # Test 1: Transitivity R_{AC} = R_{AB} * R_{BC}
    print("  Testing transitivity ...")
    n_triples = min(5000, n_points * (n_points - 1) * (n_points - 2) // 6)
    triple_results = []
    transitivity_errors = []

    triple_indices = rng.choice(n_points, size=(n_triples, 3), replace=True)
    for idx in range(n_triples):
        i, j, k = triple_indices[idx]
        if i == j or j == k or i == k:
            continue
        R_ij = compute_gear_ratio(frequencies[i], frequencies[j])
        R_jk = compute_gear_ratio(frequencies[j], frequencies[k])
        R_ik_direct = compute_gear_ratio(frequencies[i], frequencies[k])
        R_ik_transitive = R_ij * R_jk

        error = abs(R_ik_direct - R_ik_transitive) / (abs(R_ik_direct) + EPSILON)
        transitivity_errors.append(error)

        if len(triple_results) < 2000:  # Limit CSV size
            triple_results.append({
                "point_i": int(i),
                "point_j": int(j),
                "point_k": int(k),
                "R_ij": float(R_ij),
                "R_jk": float(R_jk),
                "R_ik_direct": float(R_ik_direct),
                "R_ik_transitive": float(R_ik_transitive),
                "transitivity_error": float(error),
            })

    max_trans_error = float(np.max(transitivity_errors)) if transitivity_errors else 0.0
    mean_trans_error = float(np.mean(transitivity_errors)) if transitivity_errors else 0.0
    print(f"    Max transitivity error: {max_trans_error:.2e}")

    # Test 2: Parallel transport along different paths
    print("  Testing parallel transport ...")
    n_transport_tests = 100
    holonomy_magnitudes = []
    transport_consistent = 0

    for test_idx in range(n_transport_tests):
        # Pick 4 nearby points forming two paths A->B->C and A->D->C
        base_idx = rng.integers(0, GRID_SIZE - 2, size=3)
        A = base_idx.copy()
        B = base_idx.copy(); B[0] += 1
        C = base_idx.copy(); C[0] += 1; C[1] += 1
        D = base_idx.copy(); D[1] += 1

        def idx_to_flat(ijk):
            return ijk[0] * GRID_SIZE * GRID_SIZE + ijk[1] * GRID_SIZE + ijk[2]

        pA = points[idx_to_flat(A)]
        pB = points[idx_to_flat(B)]
        pC = points[idx_to_flat(C)]
        pD = points[idx_to_flat(D)]

        # Initial fiber state
        initial_state = compute_fiber_state(pA, n=2)

        # Path 1: A -> B -> C
        state1 = parallel_transport(initial_state, [pA, pB, pC])

        # Path 2: A -> D -> C
        state2 = parallel_transport(initial_state, [pA, pD, pC])

        # Holonomy = difference between two transports
        holonomy = np.linalg.norm(state1 - state2)
        holonomy_magnitudes.append(holonomy)

        # For flat connection, holonomy should be small but nonzero
        # (our connection has curvature)
        if holonomy < 1.0:
            transport_consistent += 1

    # Test 3: Curvature estimation
    print("  Computing curvature ...")
    # Discrete curvature: holonomy around small loops / loop area
    curvatures = []
    spacing = 1.0 / (GRID_SIZE - 1)
    loop_area = spacing * spacing

    for i in range(min(GRID_SIZE - 1, 8)):
        for j in range(min(GRID_SIZE - 1, 8)):
            k = GRID_SIZE // 2  # Fix z at midpoint
            p00 = points[i * GRID_SIZE * GRID_SIZE + j * GRID_SIZE + k]
            p10 = points[(i+1) * GRID_SIZE * GRID_SIZE + j * GRID_SIZE + k]
            p11 = points[(i+1) * GRID_SIZE * GRID_SIZE + (j+1) * GRID_SIZE + k]
            p01 = points[i * GRID_SIZE * GRID_SIZE + (j+1) * GRID_SIZE + k]

            state0 = compute_fiber_state(p00, n=2)
            # Transport around loop: p00 -> p10 -> p11 -> p01 -> p00
            transported = parallel_transport(state0, [p00, p10, p11, p01, p00])
            holonomy_loop = np.linalg.norm(transported - state0)
            curvature = holonomy_loop / (loop_area + EPSILON)
            curvatures.append(curvature)

    mean_curvature = float(np.mean(curvatures)) if curvatures else 0.0

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "fiber_bundle.csv")
    if triple_results:
        fields = list(triple_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(triple_results)
    print(f"  CSV saved to {csv_path}")

    summary = {
        "test": "fiber_bundle",
        "max_transitivity_error": max_trans_error,
        "mean_transitivity_error": mean_trans_error,
        "mean_curvature": mean_curvature,
        "holonomy_test_pass": transport_consistent > n_transport_tests * 0.5,
        "parallel_transport_consistency": transport_consistent / n_transport_tests,
        "mean_holonomy": float(np.mean(holonomy_magnitudes)),
        "num_transitivity_tests": len(transitivity_errors),
        "num_transport_tests": n_transport_tests,
        "pass": max_trans_error < 1e-10,
    }

    json_path = os.path.join(RESULTS_DIR, "fiber_bundle_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")
    print(f"  Max transitivity error: {max_trans_error:.2e}")
    print(f"  Mean curvature: {mean_curvature:.4f}")
    return summary


if __name__ == "__main__":
    main()
