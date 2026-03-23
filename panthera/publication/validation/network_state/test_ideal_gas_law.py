"""
Test 1: Ideal Gas Law validation  --  PV = NkT

Runs simulations across combinations of N, V, T and verifies that
the compressibility ratio PV/(NkT) is close to unity.
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import NetworkSimulator, k_B

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_single(N, V, T_target, seed=42):
    sim = NetworkSimulator(N=N, V=V, dim=3, mode="ideal", dt=0.002, seed=seed)
    sim.init_positions_uniform()
    sim.init_velocities_mb(T_target)

    # Equilibrate
    sim.run(1000, thermostat_T=T_target)

    # Production
    sim.reset_pressure_accumulators()
    n_prod = 5000
    T_accum = 0.0
    for _ in range(n_prod):
        sim.step()
        T_accum += sim.temperature()

    elapsed = n_prod * sim.dt
    T_measured = T_accum / n_prod
    P_measured = sim.pressure_ideal(elapsed)
    ratio = (P_measured * V) / (N * k_B * T_measured) if T_measured > 0 else float("nan")
    deviation = abs(ratio - 1.0)

    return {
        "N": N,
        "V": V,
        "T_target": T_target,
        "T_measured": T_measured,
        "P_measured": P_measured,
        "PV_NkT_ratio": ratio,
        "deviation_from_unity": deviation,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Ideal Gas Law (PV = NkT) ===")

    N_values = [10, 25, 50, 100, 200]
    V_values = [1.0, 2.0, 5.0, 10.0]
    T_values = [0.5, 1.0, 2.0, 5.0]

    results = []
    total = len(N_values) * len(V_values) * len(T_values)
    done = 0

    for N in N_values:
        for V in V_values:
            for T in T_values:
                done += 1
                print(f"  [{done}/{total}] N={N}, V={V}, T={T} ...")
                row = run_single(N, V, T, seed=42 + done)
                results.append(row)

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "ideal_gas_law.csv")
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"  CSV saved to {csv_path}")

    # Summary
    ratios = [r["PV_NkT_ratio"] for r in results if np.isfinite(r["PV_NkT_ratio"])]
    mean_ratio = float(np.mean(ratios))
    std_ratio = float(np.std(ratios))
    max_dev = float(np.max([r["deviation_from_unity"] for r in results
                            if np.isfinite(r["deviation_from_unity"])]))
    passed = abs(mean_ratio - 1.0) < 0.05

    summary = {
        "test": "ideal_gas_law",
        "mean_PV_NkT_ratio": mean_ratio,
        "std_PV_NkT_ratio": std_ratio,
        "max_deviation": max_dev,
        "num_tests": len(results),
        "pass": passed,
    }
    json_path = os.path.join(RESULTS_DIR, "ideal_gas_law_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")
    print(f"  Mean ratio = {mean_ratio:.4f},  pass = {passed}")
    return summary


if __name__ == "__main__":
    main()
