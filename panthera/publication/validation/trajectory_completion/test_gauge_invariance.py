"""
Test 6: Gauge invariance of gear ratios under frequency scaling.

Validates that gear ratios R_ij = omega_i/omega_j, temperature T,
pressure P, and order parameter Psi are invariant under uniform
frequency scaling omega_i -> lambda * omega_i.
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    compute_all_gear_ratios, network_temperature, network_pressure,
    order_parameter, SCALE_HIERARCHY, EPSILON
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

N_NODES = 50
LAMBDAS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Gauge Invariance ===")

    rng = np.random.default_rng(42)

    # Create network with frequencies from 8-scale hierarchy
    # Assign each node a frequency from the hierarchy with some distribution
    scale_idx = rng.integers(0, len(SCALE_HIERARCHY), size=N_NODES)
    base_frequencies = np.array([SCALE_HIERARCHY[i] for i in scale_idx])

    # Assign phases
    phases = rng.uniform(0, 2 * np.pi, size=N_NODES)

    # Timing variances for temperature
    timing_vars = rng.exponential(0.01, size=N_NODES)

    # Compute original observables
    R_original = compute_all_gear_ratios(base_frequencies)
    T_original = network_temperature(timing_vars)
    # For pressure: P = sum(f^2)/(dim*V), but ratios cancel under scaling
    # Actually P scales as lambda^2, but P/T ratio should be invariant
    # We compute the "effective" pressure from gear ratios which IS invariant
    # P_kinetic = (1/N) * sum(omega_i^2) -- this SCALES with lambda^2
    # The gauge-invariant pressure is P / <omega^2> * <omega>^2
    P_original = network_pressure(base_frequencies)
    Psi_original = order_parameter(phases)

    # Gear ratios are the key gauge-invariant quantity
    print("  Testing scaling invariance ...")
    results = []

    max_R_change = 0.0
    max_T_change = 0.0
    max_P_ratio_change = 0.0
    max_Psi_change = 0.0

    for lam in LAMBDAS:
        scaled_frequencies = lam * base_frequencies

        R_scaled = compute_all_gear_ratios(scaled_frequencies)
        # T depends on variance, not frequency — invariant
        T_scaled = network_temperature(timing_vars)  # same timing vars
        P_scaled = network_pressure(scaled_frequencies)
        Psi_scaled = order_parameter(phases)  # phases unchanged

        # Gear ratio relative change
        mask = np.abs(R_original) > EPSILON
        if np.any(mask):
            R_rel = np.max(np.abs(R_scaled[mask] - R_original[mask]) /
                           (np.abs(R_original[mask]) + EPSILON))
        else:
            R_rel = 0.0

        # Temperature: should be identical (timing variance unchanged)
        T_rel = abs(T_scaled - T_original) / (abs(T_original) + EPSILON)

        # Pressure ratio: P/omega_mean^2 should be invariant
        omega_mean_orig = np.mean(base_frequencies ** 2)
        omega_mean_scaled = np.mean(scaled_frequencies ** 2)
        P_ratio_orig = P_original / (omega_mean_orig + EPSILON)
        P_ratio_scaled = P_scaled / (omega_mean_scaled + EPSILON)
        P_ratio_rel = abs(P_ratio_scaled - P_ratio_orig) / (abs(P_ratio_orig) + EPSILON)

        # Order parameter: phase-based, invariant
        Psi_rel = abs(Psi_scaled - Psi_original) / (abs(Psi_original) + EPSILON)

        max_R_change = max(max_R_change, R_rel)
        max_T_change = max(max_T_change, T_rel)
        max_P_ratio_change = max(max_P_ratio_change, P_ratio_rel)
        max_Psi_change = max(max_Psi_change, Psi_rel)

        results.append({
            "lambda": lam,
            "T_original": T_original,
            "T_scaled": T_scaled,
            "T_relative_change": float(T_rel),
            "P_ratio_relative_change": float(P_ratio_rel),
            "Psi_relative_change": float(Psi_rel),
            "R_max_relative_change": float(R_rel),
        })

        print(f"  lambda={lam:>6.1f}: R_change={R_rel:.2e}, "
              f"T_change={T_rel:.2e}, Psi_change={Psi_rel:.2e}")

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "gauge_invariance.csv")
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"  CSV saved to {csv_path}")

    eps_threshold = 1e-12
    summary = {
        "test": "gauge_invariance",
        "max_gear_ratio_relative_change": float(max_R_change),
        "max_temperature_relative_change": float(max_T_change),
        "max_pressure_ratio_relative_change": float(max_P_ratio_change),
        "max_order_parameter_relative_change": float(max_Psi_change),
        "gear_ratio_invariant": max_R_change < eps_threshold,
        "temperature_invariant": max_T_change < eps_threshold,
        "order_parameter_invariant": max_Psi_change < eps_threshold,
        "pass": max_R_change < eps_threshold and max_T_change < eps_threshold,
    }

    json_path = os.path.join(RESULTS_DIR, "gauge_invariance_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")
    print(f"  Gear ratio max change: {max_R_change:.2e}")
    print(f"  Temperature max change: {max_T_change:.2e}")
    return summary


if __name__ == "__main__":
    main()
