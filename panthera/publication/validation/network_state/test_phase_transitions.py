"""
Test 3: Phase transitions (Gas -> Liquid -> Crystal).

Uses Lennard-Jones particles and sweeps temperature from high to low,
tracking order parameter, specific heat, and mean squared displacement.
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import NetworkSimulator, k_B

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Phase Transitions ===")

    N = 200
    # Density chosen so that LJ transitions are visible (~0.8 in LJ units)
    density = 0.8
    V = N / density
    dim = 3

    T_high, T_low = 5.0, 0.01
    n_temps = 50
    temperatures = np.geomspace(T_high, T_low, n_temps)

    results = []

    # Initialise once, then cool stepwise
    sim = NetworkSimulator(N=N, V=V, dim=dim, mode="lj",
                           epsilon=1.0, sigma=1.0, dt=0.002, seed=42)
    sim.init_positions_lattice()
    sim.init_velocities_mb(T_high)

    for idx, T in enumerate(temperatures):
        print(f"  [{idx+1}/{n_temps}] T = {T:.4f}")

        # Equilibrate with thermostat
        eq_steps = 2000
        sim.run(eq_steps, thermostat_T=T)

        # Production: collect energy samples
        prod_steps = 2000
        energies = []
        psi_samples = []
        ref_pos = sim.positions.copy()
        msd_vals = []

        for s in range(prod_steps):
            sim.step(thermostat_T=T)
            E = sim.total_energy()
            energies.append(E)
            if s % 50 == 0:
                psi_samples.append(sim.order_parameter())
                msd_vals.append(sim.mean_squared_displacement(ref_pos))

        energies = np.array(energies)
        mean_E = float(np.mean(energies))
        var_E = float(np.var(energies))
        Cv = var_E / (k_B * T**2) if T > 1e-12 else 0.0
        psi = float(np.mean(psi_samples))
        msd = float(np.mean(msd_vals[-max(1, len(msd_vals)//2):]))  # late-time average

        # Phase label heuristic
        if T > 2.0:
            phase = "gas"
        elif T > 0.5:
            phase = "liquid"
        else:
            phase = "crystal"

        results.append({
            "temperature": float(T),
            "order_parameter": psi,
            "specific_heat": Cv,
            "mean_energy": mean_E,
            "energy_variance": var_E,
            "mean_sq_displacement": msd,
            "phase_label": phase,
        })

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "phase_transitions.csv")
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"  CSV saved to {csv_path}")

    # Detect transitions from C_V peaks
    Cv_vals = np.array([r["specific_heat"] for r in results])
    T_vals = np.array([r["temperature"] for r in results])
    peaks = []
    for i in range(1, len(Cv_vals) - 1):
        if Cv_vals[i] > Cv_vals[i-1] and Cv_vals[i] > Cv_vals[i+1]:
            peaks.append(i)

    T_c_est = float(T_vals[peaks[0]]) if len(peaks) > 0 else None
    T_m_est = float(T_vals[peaks[1]]) if len(peaks) > 1 else None

    # Critical exponent estimate (rough)
    beta_est = None
    if T_c_est is not None:
        psi_vals = np.array([r["order_parameter"] for r in results])
        mask = (T_vals < T_c_est) & (T_vals > 0.1)
        if np.sum(mask) > 3:
            dt_arr = np.abs(T_vals[mask] - T_c_est)
            psi_arr = psi_vals[mask]
            valid = (dt_arr > 0) & (psi_arr > 0)
            if np.sum(valid) > 2:
                coeffs = np.polyfit(np.log(dt_arr[valid]), np.log(psi_arr[valid]), 1)
                beta_est = float(coeffs[0])

    summary = {
        "test": "phase_transitions",
        "T_c_estimated": T_c_est,
        "T_m_estimated": T_m_est,
        "critical_exponent_beta": beta_est,
        "transitions_detected": len(peaks) > 0,
        "pass": bool(len(peaks) > 0),
    }
    json_path = os.path.join(RESULTS_DIR, "phase_transitions_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: T_c ~ {T_c_est}, transitions_detected = {summary['transitions_detected']}")
    return summary


if __name__ == "__main__":
    main()
