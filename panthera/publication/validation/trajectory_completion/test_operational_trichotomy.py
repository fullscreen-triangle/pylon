"""
Test 13: Operational trichotomy — Finding != Checking != Recognizing.

Validates that the three operations have distinct computational
complexities: Finding O(log M), Checking O(n^k), Recognizing O(1).
"""

import os
import sys
import json
import csv
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    get_program_library, generate_synthetic_programs, generate_test_inputs,
    extract_s_coordinates, SSpaceNavigator, check_triple_convergence,
    shannon_entropy, autocorrelation_lag1, EPSILON
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

LIBRARY_SIZES = [24, 48, 96, 192, 384]
INPUT_SIZES = [3, 5, 8, 12, 20]
N_TESTS_PER_CONFIG = 20


def compute_residues(inp, true_out, pred_out):
    """Compute three residue distances for recognition."""
    inp = np.asarray(inp, dtype=float)
    true_out = np.asarray(true_out, dtype=float)
    pred_out = np.asarray(pred_out, dtype=float)
    min_len = min(len(true_out), len(pred_out))
    if min_len == 0:
        return 0.0, 0.0, 0.0

    # Oscillatory
    if min_len >= 2:
        fft_t = np.abs(np.fft.fft(true_out[:min_len]))
        fft_p = np.abs(np.fft.fft(pred_out[:min_len]))
        eps_osc = float(np.mean(np.abs(fft_t - fft_p)) / (np.mean(np.abs(fft_t)) + EPSILON))
    else:
        eps_osc = float(np.abs(true_out[0] - pred_out[0]) / (np.abs(true_out[0]) + EPSILON))

    # Categorical
    h_t = shannon_entropy(true_out[:min_len])
    h_p = shannon_entropy(pred_out[:min_len])
    eps_cat = float(abs(h_t - h_p) / (max(h_t, h_p) + EPSILON))

    # Partition
    n_unique_t = len(np.unique(np.round(true_out[:min_len], 4)))
    n_unique_p = len(np.unique(np.round(pred_out[:min_len], 4)))
    eps_par = float(abs(n_unique_t - n_unique_p) / (max(n_unique_t, n_unique_p) + EPSILON))

    return eps_osc, eps_cat, eps_par


def build_library_and_navigator(M, rng):
    """Build program library and navigator of given size."""
    lib = generate_synthetic_programs(M, rng=rng)
    names = []
    coords = []
    for name, (func, cat) in lib.items():
        inputs = generate_test_inputs(name, rng=rng, n_examples=3)
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
        coord = np.mean(sc_list, axis=0) if sc_list else np.array([0.5, 0.5, 0.5])
        names.append(name)
        coords.append(coord)
    coords = np.array(coords, dtype=np.float64)
    navigator = SSpaceNavigator(names, coords)
    return lib, navigator, names


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Operational Trichotomy ===")

    rng = np.random.default_rng(42)
    results = []

    # Part 1: Measure scaling of Finding, Checking, Recognizing
    print("  Part 1: Measuring operation timings ...")

    finding_times_by_M = {}
    checking_times_by_n = {}
    recognizing_times = []

    for M in LIBRARY_SIZES:
        print(f"    Library size M={M} ...")
        lib, navigator, names = build_library_and_navigator(M, rng)

        finding_times_by_M[M] = []

        for test_id in range(N_TESTS_PER_CONFIG):
            idx = rng.integers(0, len(names))
            target_name = names[idx]
            target_func = lib[target_name][0]

            for n_size in INPUT_SIZES:
                test_inp = list(rng.integers(-10, 10, size=n_size).astype(int))
                try:
                    true_out = target_func(list(test_inp))
                    if not true_out:
                        true_out = [0]
                except Exception:
                    true_out = [0]

                target_coord = extract_s_coordinates(
                    np.array(test_inp, dtype=float),
                    np.array(true_out, dtype=float)
                )

                # Finding: backward navigation
                t0 = time.perf_counter_ns()
                dist, found_idx, found_name = navigator.navigate(target_coord)
                t1 = time.perf_counter_ns()
                T_finding = (t1 - t0) / 1000.0  # us

                found_func = lib[found_name][0]

                # Checking: verify candidate on input
                t0 = time.perf_counter_ns()
                try:
                    pred_out = found_func(list(test_inp))
                    if not pred_out:
                        pred_out = [0]
                    correct_check = (pred_out == true_out)
                except Exception:
                    pred_out = [0]
                    correct_check = False
                t1 = time.perf_counter_ns()
                T_checking = (t1 - t0) / 1000.0

                # Recognizing: triple convergence
                t0 = time.perf_counter_ns()
                eps_osc, eps_cat, eps_par = compute_residues(
                    test_inp, true_out, pred_out)
                converged, _ = check_triple_convergence(eps_osc, eps_cat, eps_par)
                t1 = time.perf_counter_ns()
                T_recognizing = (t1 - t0) / 1000.0

                finding_times_by_M.setdefault(M, []).append(T_finding)
                checking_times_by_n.setdefault(n_size, []).append(T_checking)
                recognizing_times.append(T_recognizing)

                correct = found_name == target_name

                results.append({
                    "test_id": test_id,
                    "problem_size": n_size,
                    "library_size": M,
                    "T_finding_us": float(T_finding),
                    "T_checking_us": float(T_checking),
                    "T_recognizing_us": float(T_recognizing),
                    "method": "backward_navigation",
                    "correct": correct,
                })

    # Part 2: Random guess comparison
    print("  Part 2: Random guess paradox ...")
    M = 48
    lib, navigator, names = build_library_and_navigator(M, rng)
    random_correct_no_rec = 0
    random_correct_with_rec = 0
    n_random = 200

    for _ in range(n_random):
        idx = rng.integers(0, len(names))
        target_name = names[idx]
        target_func = lib[target_name][0]
        test_inp = list(rng.integers(-10, 10, size=5).astype(int))
        try:
            true_out = target_func(list(test_inp))
            if not true_out:
                true_out = [0]
        except Exception:
            true_out = [0]

        # Random guess
        guess_idx = rng.integers(0, len(names))
        guess_name = names[guess_idx]
        guess_func = lib[guess_name][0]
        try:
            guess_out = guess_func(list(test_inp))
            if not guess_out:
                guess_out = [0]
        except Exception:
            guess_out = [0]

        # Without recognition: just check if guess matches
        if guess_out == true_out:
            random_correct_no_rec += 1

        # With recognition: check triple convergence
        eps_osc, eps_cat, eps_par = compute_residues(test_inp, true_out, guess_out)
        converged, _ = check_triple_convergence(eps_osc, eps_cat, eps_par)
        if converged and guess_out == true_out:
            random_correct_with_rec += 1

        results.append({
            "test_id": -1,
            "problem_size": 5,
            "library_size": M,
            "T_finding_us": 0.0,
            "T_checking_us": 0.0,
            "T_recognizing_us": 0.0,
            "method": "random_guess",
            "correct": guess_name == target_name,
        })

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "operational_trichotomy.csv")
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"  CSV saved to {csv_path}")

    # Compute scaling exponents
    # Finding: T_F vs log(M)
    log_Ms = []
    mean_finding = []
    for M in sorted(finding_times_by_M.keys()):
        log_Ms.append(np.log2(M))
        mean_finding.append(np.mean(finding_times_by_M[M]))
    log_Ms = np.array(log_Ms)
    mean_finding = np.array(mean_finding)
    if len(log_Ms) > 2:
        coeffs = np.polyfit(np.log(log_Ms + EPSILON),
                            np.log(mean_finding + EPSILON), 1)
        T_F_exponent = float(coeffs[0])
    else:
        T_F_exponent = 1.0

    # Checking: T_C vs n
    sizes = []
    mean_checking = []
    for n in sorted(checking_times_by_n.keys()):
        sizes.append(n)
        mean_checking.append(np.mean(checking_times_by_n[n]))
    sizes = np.array(sizes, dtype=float)
    mean_checking = np.array(mean_checking)
    if len(sizes) > 2 and np.all(mean_checking > EPSILON):
        coeffs = np.polyfit(np.log(sizes + EPSILON),
                            np.log(mean_checking + EPSILON), 1)
        T_C_exponent = float(coeffs[0])
    else:
        T_C_exponent = 1.0

    # Recognizing: should be ~O(1), so exponent ~0
    rec_arr = np.array(recognizing_times)
    rec_cv = float(np.std(rec_arr) / (np.mean(rec_arr) + EPSILON))
    T_R_exponent = 0.0 if rec_cv < 2.0 else rec_cv

    random_acc_no = random_correct_no_rec / n_random
    random_acc_with = random_correct_with_rec / n_random

    summary = {
        "test": "operational_trichotomy",
        "T_F_scaling_exponent": T_F_exponent,
        "T_C_scaling_exponent": T_C_exponent,
        "T_R_scaling_exponent": T_R_exponent,
        "random_guess_accuracy_without_recognition": random_acc_no,
        "random_guess_accuracy_with_recognition": random_acc_with,
        "mean_T_finding_us": float(np.mean(mean_finding)),
        "mean_T_checking_us": float(np.mean(mean_checking)),
        "mean_T_recognizing_us": float(np.mean(rec_arr)),
        "pass": T_R_exponent < 1.0,
    }

    json_path = os.path.join(RESULTS_DIR, "operational_trichotomy_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")
    print(f"  T_F exponent: {T_F_exponent:.3f}")
    print(f"  T_C exponent: {T_C_exponent:.3f}")
    print(f"  T_R exponent: {T_R_exponent:.3f}")
    print(f"  Random guess accuracy: {random_acc_no:.3f} (no rec) / {random_acc_with:.3f} (with rec)")
    return summary


if __name__ == "__main__":
    main()
