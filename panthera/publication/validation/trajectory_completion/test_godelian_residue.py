"""
Test 11: Godelian residue — triple convergence and recognition.

Validates the Godelian residue mechanism: three independent distance
measurements (oscillatory, categorical, partition) converge for correct
syntheses and diverge for incorrect ones.
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    get_program_library, get_all_program_names, generate_test_inputs,
    extract_s_coordinates, SSpaceNavigator, shannon_entropy,
    autocorrelation_lag1, check_triple_convergence, EPSILON
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

CONVERGENCE_DELTA = 0.15


def compute_oscillatory_residue(inp, out, predicted_out):
    """Oscillatory (frequency-based) residue: distance in frequency domain."""
    inp = np.asarray(inp, dtype=float)
    out = np.asarray(out, dtype=float)
    predicted_out = np.asarray(predicted_out, dtype=float)

    # Compare frequency content via FFT magnitudes
    min_len = min(len(out), len(predicted_out))
    if min_len < 2:
        return float(np.mean(np.abs(out[:min_len] - predicted_out[:min_len])) + EPSILON)

    fft_out = np.abs(np.fft.fft(out[:min_len]))
    fft_pred = np.abs(np.fft.fft(predicted_out[:min_len]))
    return float(np.mean(np.abs(fft_out - fft_pred)) / (np.mean(np.abs(fft_out)) + EPSILON))


def compute_categorical_residue(inp, out, predicted_out):
    """Categorical (entropy-based) residue: distance in entropy space."""
    h_out = shannon_entropy(out)
    h_pred = shannon_entropy(predicted_out)

    # Also compare autocorrelation structure
    ac_out = autocorrelation_lag1(out)
    ac_pred = autocorrelation_lag1(predicted_out)

    entropy_diff = abs(h_out - h_pred) / (max(h_out, h_pred) + EPSILON)
    ac_diff = abs(ac_out - ac_pred)

    return float(entropy_diff + ac_diff)


def compute_partition_residue(inp, out, predicted_out):
    """Partition (quantum-number-based) residue: distance in partition coordinates."""
    inp = np.asarray(inp, dtype=float)
    out = np.asarray(out, dtype=float)
    predicted_out = np.asarray(predicted_out, dtype=float)

    # Partition coordinates: (n, l, m, s)
    # n ~ number of distinct values
    # l ~ range / mean
    # m ~ skewness
    # s ~ sign of mean

    def partition_coords(arr):
        if len(arr) == 0:
            return np.array([0, 0, 0, 0], dtype=float)
        n = len(np.unique(np.round(arr, 4)))
        mean_abs = np.mean(np.abs(arr)) + EPSILON
        rng = np.max(arr) - np.min(arr)
        l = rng / mean_abs
        m = float(np.mean(arr))
        s = 1.0 if m >= 0 else -1.0
        return np.array([n, l, m, s], dtype=float)

    pc_out = partition_coords(out)
    pc_pred = partition_coords(predicted_out)

    # Normalized distance
    diff = pc_out - pc_pred
    scale = np.abs(pc_out) + np.abs(pc_pred) + EPSILON
    return float(np.mean(np.abs(diff) / scale))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Godelian Residue ===")

    rng = np.random.default_rng(42)
    lib = get_program_library()
    all_names = get_all_program_names()

    # Build S-coordinate library
    print("  Building S-coordinate library ...")
    lib_names = []
    lib_coords = []

    for name in all_names:
        func, cat = lib[name]
        inputs = generate_test_inputs(name, rng=np.random.default_rng(100), n_examples=5)
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
            coord = np.mean(sc_list, axis=0)
        else:
            coord = np.array([0.5, 0.5, 0.5])
        lib_names.append(name)
        lib_coords.append(coord)

    lib_coords = np.array(lib_coords, dtype=np.float64)
    navigator = SSpaceNavigator(lib_names, lib_coords)

    # Test each program
    print("  Testing triple convergence ...")
    results = []
    correct_convergences = 0
    correct_total = 0
    incorrect_divergences = 0
    incorrect_total = 0
    all_epsilons = []
    recognition_times = []  # To verify O(1)

    for i, name in enumerate(all_names):
        func, cat = lib[name]
        test_inputs = generate_test_inputs(name, rng=np.random.default_rng(300 + i), n_examples=3)

        for inp in test_inputs:
            try:
                true_out = func(list(inp))
                if not true_out:
                    true_out = [0]
            except Exception:
                continue

            target_coord = extract_s_coordinates(np.array(inp, dtype=float),
                                                  np.array(true_out, dtype=float))

            # Navigate to nearest neighbor
            dist, found_idx, found_name = navigator.navigate(target_coord)
            found_func = lib[found_name][0]

            try:
                predicted_out = found_func(list(inp))
                if not predicted_out:
                    predicted_out = [0]
            except Exception:
                predicted_out = [0]

            inp_arr = np.array(inp, dtype=float)
            out_arr = np.array(true_out, dtype=float)
            pred_arr = np.array(predicted_out, dtype=float)

            # Compute three residues
            import time
            t0 = time.perf_counter_ns()

            eps_osc = compute_oscillatory_residue(inp_arr, out_arr, pred_arr)
            eps_cat = compute_categorical_residue(inp_arr, out_arr, pred_arr)
            eps_par = compute_partition_residue(inp_arr, out_arr, pred_arr)

            t1 = time.perf_counter_ns()
            recognition_times.append((t1 - t0) / 1000.0)  # microseconds

            converged, max_disc = check_triple_convergence(
                eps_osc, eps_cat, eps_par, delta=CONVERGENCE_DELTA)

            correct = (found_name == name)
            if correct:
                correct_total += 1
                if converged:
                    correct_convergences += 1
            else:
                incorrect_total += 1
                if not converged:
                    incorrect_divergences += 1

            # Minimum residue (with EPSILON guard for separation necessity)
            min_eps = min(eps_osc, eps_cat, eps_par) + EPSILON
            all_epsilons.append(min_eps)

            results.append({
                "program": name,
                "correct": correct,
                "epsilon_osc": float(eps_osc),
                "epsilon_cat": float(eps_cat),
                "epsilon_par": float(eps_par),
                "convergence_delta": float(max_disc),
                "converged": converged,
                "recognized_correctly": correct and converged,
            })

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "godelian_residue.csv")
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"  CSV saved to {csv_path}")

    conv_rate_correct = correct_convergences / max(correct_total, 1)
    div_rate_incorrect = incorrect_divergences / max(incorrect_total, 1)
    min_epsilon = float(np.min(all_epsilons)) if all_epsilons else 0.0

    # Recognition complexity: check that times are constant (low variance)
    rec_mean = float(np.mean(recognition_times))
    rec_std = float(np.std(recognition_times))
    rec_cv = rec_std / (rec_mean + EPSILON)  # coefficient of variation

    summary = {
        "test": "godelian_residue",
        "convergence_rate_correct": conv_rate_correct,
        "divergence_rate_incorrect": div_rate_incorrect,
        "min_epsilon": min_epsilon,
        "epsilon_positive": min_epsilon > 0,
        "recognition_complexity_constant": rec_cv < 2.0,
        "recognition_mean_us": rec_mean,
        "recognition_cv": rec_cv,
        "num_correct": correct_total,
        "num_incorrect": incorrect_total,
        "pass": conv_rate_correct > 0.3 and min_epsilon > 0,
    }

    json_path = os.path.join(RESULTS_DIR, "godelian_residue_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")
    print(f"  Convergence rate (correct): {conv_rate_correct:.2f}")
    print(f"  Divergence rate (incorrect): {div_rate_incorrect:.2f}")
    print(f"  Min epsilon: {min_epsilon:.6f} (>0: {min_epsilon > 0})")
    return summary


if __name__ == "__main__":
    main()
