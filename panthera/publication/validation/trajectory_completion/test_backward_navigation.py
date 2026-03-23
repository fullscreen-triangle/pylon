"""
Test 2: O(log M) backward navigation scaling.

Validates that backward navigation via KD-tree in S-entropy space
scales as O(log M) with library size M, compared to O(M) brute force.
"""

import os
import sys
import json
import csv
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    generate_synthetic_programs, generate_test_inputs,
    extract_s_coordinates, SSpaceNavigator, EPSILON
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

N_TRIALS = 50
LIBRARY_SIZES = [12, 24, 48, 96, 192, 384, 768, 1536]


def build_library_coordinates(lib, rng):
    """Compute S-coordinates for all programs in a library."""
    names = []
    coords = []
    for name, (func, cat) in lib.items():
        inputs = generate_test_inputs(name, rng=rng, n_examples=3)
        sk_vals, st_vals, se_vals = [], [], []
        for inp in inputs:
            try:
                out = func(list(inp))
                if not out:
                    out = [0]
                sc = extract_s_coordinates(np.array(inp, dtype=float),
                                           np.array(out, dtype=float))
                sk_vals.append(sc[0])
                st_vals.append(sc[1])
                se_vals.append(sc[2])
            except Exception:
                pass
        if sk_vals:
            coord = [np.mean(sk_vals), np.mean(st_vals), np.mean(se_vals)]
        else:
            coord = [0.5, 0.5, 0.5]
        names.append(name)
        coords.append(coord)
    return names, np.array(coords, dtype=np.float64)


def run_scaling_test(M, rng):
    """Run backward navigation test for library of size M."""
    lib = generate_synthetic_programs(M, rng=rng)
    names, coords = build_library_coordinates(lib, rng)
    navigator = SSpaceNavigator(names, coords)

    backward_comparisons = []
    backward_times = []
    correct_count = 0

    for trial in range(N_TRIALS):
        # Pick a random program as the target
        idx = rng.integers(0, len(names))
        target_name = names[idx]
        target_func = lib[target_name][0]

        # Generate fresh test input
        test_inp = list(rng.integers(-10, 10, size=rng.integers(3, 7)).astype(int))
        try:
            test_out = target_func(list(test_inp))
            if not test_out:
                test_out = [0]
        except Exception:
            test_out = [0]

        target_coords = extract_s_coordinates(
            np.array(test_inp, dtype=float),
            np.array(test_out, dtype=float)
        )

        # Backward navigation (KD-tree)
        t0 = time.perf_counter_ns()
        dist, found_idx, found_name = navigator.navigate(target_coords)
        t1 = time.perf_counter_ns()

        backward_times.append((t1 - t0) / 1000.0)  # microseconds
        est_comparisons = navigator.estimate_comparisons(target_coords)
        backward_comparisons.append(est_comparisons)

        if found_name == target_name:
            correct_count += 1

    accuracy = correct_count / N_TRIALS
    forward_comparisons = M  # brute force always checks all M

    return {
        "library_size": M,
        "backward_comparisons_mean": float(np.mean(backward_comparisons)),
        "backward_comparisons_std": float(np.std(backward_comparisons)),
        "backward_time_mean_us": float(np.mean(backward_times)),
        "backward_time_std_us": float(np.std(backward_times)),
        "forward_comparisons": forward_comparisons,
        "speedup_factor": float(M / max(np.mean(backward_comparisons), 1)),
        "accuracy": accuracy,
        "log2_M": float(np.log2(M)),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Backward Navigation O(log M) Scaling ===")

    rng = np.random.default_rng(42)
    results = []

    for i, M in enumerate(LIBRARY_SIZES):
        print(f"  [{i+1}/{len(LIBRARY_SIZES)}] Library size M={M} ...")
        row = run_scaling_test(M, rng)
        print(f"    comparisons={row['backward_comparisons_mean']:.1f}, "
              f"time={row['backward_time_mean_us']:.1f}us, "
              f"accuracy={row['accuracy']:.2f}, "
              f"speedup={row['speedup_factor']:.1f}x")
        results.append(row)

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "backward_navigation.csv")
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"  CSV saved to {csv_path}")

    # Log-log fit: comparisons vs log2(M)
    log2_M = np.array([r["log2_M"] for r in results])
    comp = np.array([r["backward_comparisons_mean"] for r in results])

    # Fit: comparisons = a * log2(M) + b
    coeffs = np.polyfit(log2_M, comp, 1)
    predicted = np.polyval(coeffs, log2_M)
    ss_res = np.sum((comp - predicted) ** 2)
    ss_tot = np.sum((comp - np.mean(comp)) ** 2)
    r_squared = 1.0 - ss_res / (ss_tot + EPSILON)

    # Log-log fit for scaling exponent
    log_comp = np.log(comp + EPSILON)
    log_log2M = np.log(log2_M + EPSILON)
    slope, _ = np.polyfit(log_log2M, log_comp, 1)

    accuracies = [r["accuracy"] for r in results]
    overall_accuracy = float(np.mean(accuracies))

    summary = {
        "test": "backward_navigation_scaling",
        "scaling_exponent": float(slope),
        "R_squared": float(r_squared),
        "linear_fit_slope": float(coeffs[0]),
        "linear_fit_intercept": float(coeffs[1]),
        "accuracy_overall": overall_accuracy,
        "accuracy_by_size": {str(r["library_size"]): r["accuracy"] for r in results},
        "pass": r_squared > 0.8,
    }

    json_path = os.path.join(RESULTS_DIR, "backward_navigation_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")
    print(f"  Scaling exponent={slope:.3f}, R²={r_squared:.4f}, "
          f"accuracy={overall_accuracy:.2f}")
    return summary


if __name__ == "__main__":
    main()
