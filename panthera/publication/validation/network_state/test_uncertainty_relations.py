"""
Test 7: Heisenberg-like uncertainty relation for network nodes.

Validates sigma_address * sigma_queue_depth >= hbar_network / 2.

Uses an interacting (LJ) gas so that particles exchange momentum and
both position and momentum fluctuate over time for every node.
The emergent network Planck constant hbar_net = k_B * T * tau_corr
is measured from velocity autocorrelation time.
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import NetworkSimulator, k_B, MASS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def velocity_autocorrelation_time(vel_history, dt):
    """
    Estimate velocity autocorrelation time from (n_samples,) series.
    Returns time in simulation units.
    """
    n = len(vel_history)
    if n < 20:
        return dt
    v = vel_history - np.mean(vel_history)
    var = np.var(v)
    if var < 1e-30:
        return dt

    max_lag = min(n // 4, 200)
    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        acf[lag] = np.mean(v[:n-lag] * v[lag:]) / var

    # Integrate to first zero crossing
    tau_int = 0.5  # half of lag=0
    for lag in range(1, max_lag):
        if acf[lag] < 0.0:
            break
        tau_int += acf[lag]

    return max(tau_int * dt, dt)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Uncertainty Relations ===")

    N = 80
    density = 0.3
    V = N / density
    dim = 3
    temperatures = [1.0, 2.0, 5.0]

    all_rows = []

    for T in temperatures:
        print(f"  T = {T} ...")

        sim = NetworkSimulator(N=N, V=V, dim=dim, mode="lj",
                               epsilon=1.0, sigma=1.0, dt=0.003, seed=42)
        sim.init_positions_lattice()
        sim.init_velocities_mb(T)
        sim.run(2000, thermostat_T=T)

        # Collect time series (no thermostat during production -> NVE)
        n_samples = 1000
        sample_interval = 3
        dt_sample = sample_interval * sim.dt

        pos_history = np.zeros((n_samples, N))  # x-component
        vel_history = np.zeros((n_samples, N))

        for s in range(n_samples):
            sim.run(sample_interval)
            pos_history[s] = sim.positions[:, 0]
            vel_history[s] = sim.velocities[:, 0]

        # Per-node position and momentum spread
        sigma_x = np.std(pos_history, axis=0)  # (N,)
        sigma_p = np.std(vel_history, axis=0) * MASS
        product = sigma_x * sigma_p

        # Velocity autocorrelation time (sample of nodes)
        tau_samples = []
        for node_idx in range(0, N, 10):
            tau = velocity_autocorrelation_time(vel_history[:, node_idx], dt_sample)
            tau_samples.append(tau)
        tau_corr = float(np.mean(tau_samples))

        hbar_net = k_B * T * tau_corr
        theoretical_min = hbar_net / 2.0  # analogy to hbar/2

        ratios = product / theoretical_min if theoretical_min > 0 else np.full(N, float("nan"))

        n_violating = int(np.sum(ratios < 1.0))
        print(f"    tau_corr={tau_corr:.4f}, hbar_net={hbar_net:.6f}, "
              f"min_ratio={np.min(ratios):.4f}, violating={n_violating}/{N}")

        for i in range(N):
            all_rows.append({
                "temperature": float(T),
                "node_id": i,
                "sigma_x": float(sigma_x[i]),
                "sigma_p": float(sigma_p[i]),
                "product": float(product[i]),
                "theoretical_minimum": float(theoretical_min),
                "ratio": float(ratios[i]),
            })

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "uncertainty_relations.csv")
    fields = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"  CSV saved to {csv_path}")

    all_ratios = np.array([r["ratio"] for r in all_rows if np.isfinite(r["ratio"])])
    n_total = len(all_ratios)
    n_violating_total = int(np.sum(all_ratios < 1.0))
    frac_violating = n_violating_total / n_total if n_total > 0 else 1.0

    summary = {
        "test": "uncertainty_relations",
        "temperatures_tested": temperatures,
        "min_ratio": float(np.min(all_ratios)) if len(all_ratios) > 0 else float("nan"),
        "mean_ratio": float(np.mean(all_ratios)) if len(all_ratios) > 0 else float("nan"),
        "fraction_violating": frac_violating,
        "pass": frac_violating < 0.05,
    }
    json_path = os.path.join(RESULTS_DIR, "uncertainty_relations_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Overall: min ratio = {summary['min_ratio']:.4f}, "
          f"violating = {frac_violating:.1%}")
    print(f"  pass = {summary['pass']}")
    return summary


if __name__ == "__main__":
    main()
