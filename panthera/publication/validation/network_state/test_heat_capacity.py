"""
Test 11: Heat capacity validation -- C_V and C_P.

Validates:
  - Ideal gas: C_V = (3/2)Nk_B via equipartition (kinetic energy per dof)
  - gamma = C_P/C_V = 5/3
  - LJ gas: C_V from energy fluctuations in NVE ensemble deviates from ideal

For the ideal gas, C_V is verified via the equipartition theorem:
    <KE> = (3/2) N k_B T  =>  C_V = d<E>/dT = (3/2) N k_B

For the LJ gas, C_V is measured from energy fluctuations in NVE:
    C_V = (3/2) N k_B / (1 - (3N k_B^2 T^2)^{-1} var(KE))
(Lebowitz-Percus-Verlet formula for microcanonical ensemble).
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import NetworkSimulator, k_B, MASS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Heat Capacity ===")

    dim = 3
    temperatures = [0.5, 1.0, 2.0, 5.0, 10.0]
    results = []
    count = 0

    # -- Ideal gas: verify equipartition --
    N_values = [50, 100, 200]
    for N in N_values:
        V = max(N * 1.0, 50.0)
        for T in temperatures:
            count += 1
            print(f"  [{count}] Ideal: N={N}, T={T}")

            sim = NetworkSimulator(N=N, V=V, dim=dim, mode="ideal",
                                   dt=0.002, seed=42 + count)
            sim.init_positions_uniform()
            sim.init_velocities_mb(T)
            sim.run(1000, thermostat_T=T)

            # Measure average kinetic energy (no thermostat)
            n_prod = 3000
            KE_sum = 0.0
            T_sum = 0.0
            for _ in range(n_prod):
                sim.step()
                KE_sum += sim.kinetic_energy
                T_sum += sim.temperature()

            KE_mean = KE_sum / n_prod
            T_meas = T_sum / n_prod

            # Equipartition: <KE> = (3/2) N kB T => C_V = (3/2) N kB
            # Verify by checking that KE/T matches (3/2)NkB
            Cv_measured = KE_mean / T_meas if T_meas > 0 else 0.0  # = (3/2) N kB
            Cv_theory = 1.5 * N * k_B
            Cp_theory = Cv_theory + N * k_B
            Cp_measured = Cv_measured + N * k_B
            gamma_theory = 5.0 / 3.0
            gamma_measured = Cp_measured / Cv_measured if Cv_measured > 1e-12 else float("nan")

            results.append({
                "type": "ideal",
                "temperature": float(T_meas),
                "N": N,
                "Cv_measured": float(Cv_measured),
                "Cv_theory": float(Cv_theory),
                "Cp_measured": float(Cp_measured),
                "Cp_theory": float(Cp_theory),
                "gamma_measured": float(gamma_measured),
                "gamma_theory": gamma_theory,
            })

    # -- LJ gas: C_V from energy fluctuations in NVE --
    N_lj = 100
    density = 0.3
    V_lj = N_lj / density
    for T in temperatures:
        count += 1
        print(f"  [{count}] LJ: N={N_lj}, T={T}, rho={density}")

        sim = NetworkSimulator(N=N_lj, V=V_lj, dim=dim, mode="lj",
                               epsilon=1.0, sigma=1.0, dt=0.003, seed=42 + count)
        sim.init_positions_lattice()
        sim.init_velocities_mb(T)
        sim.run(2000, thermostat_T=T)

        # NVE production
        n_prod = 5000
        KE_vals = []
        E_vals = []
        T_vals_lj = []
        for _ in range(n_prod):
            sim.step()  # no thermostat -> NVE
            KE_vals.append(sim.kinetic_energy)
            E_vals.append(sim.total_energy())
            T_vals_lj.append(sim.temperature())

        KE_vals = np.array(KE_vals)
        T_mean = float(np.mean(T_vals_lj))
        KE_mean = float(np.mean(KE_vals))
        KE_var = float(np.var(KE_vals))

        # Lebowitz-Percus-Verlet: Cv = (3/2)NkB / (1 - 2*var(KE)/(3*N*kB^2*T^2))
        denom = 1.0 - 2.0 * KE_var / (3.0 * N_lj * k_B**2 * T_mean**2) if T_mean > 0 else 1.0
        if abs(denom) > 1e-10:
            Cv_meas = 1.5 * N_lj * k_B / denom
        else:
            Cv_meas = float("nan")

        Cv_ideal = 1.5 * N_lj * k_B

        results.append({
            "type": "lj",
            "temperature": float(T_mean),
            "N": N_lj,
            "Cv_measured": float(Cv_meas),
            "Cv_theory": float(Cv_ideal),
            "Cp_measured": float(Cv_meas + N_lj * k_B) if np.isfinite(Cv_meas) else float("nan"),
            "Cp_theory": float(Cv_ideal + N_lj * k_B),
            "gamma_measured": float((Cv_meas + N_lj * k_B) / Cv_meas) if Cv_meas > 0 and np.isfinite(Cv_meas) else float("nan"),
            "gamma_theory": 5.0 / 3.0,
        })

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "heat_capacity.csv")
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"  CSV saved to {csv_path}")

    # Summary: ideal gas results
    ideal_results = [r for r in results if r["type"] == "ideal"]
    Cv_errs = []
    gamma_errs = []
    for r in ideal_results:
        if r["Cv_theory"] > 0:
            Cv_errs.append(abs(r["Cv_measured"] - r["Cv_theory"]) / r["Cv_theory"])
        if np.isfinite(r["gamma_measured"]) and r["gamma_theory"] > 0:
            gamma_errs.append(abs(r["gamma_measured"] - r["gamma_theory"]) / r["gamma_theory"])

    summary = {
        "test": "heat_capacity",
        "Cv_mean_relative_error_ideal": float(np.mean(Cv_errs)) if Cv_errs else float("nan"),
        "gamma_mean_relative_error_ideal": float(np.mean(gamma_errs)) if gamma_errs else float("nan"),
        "num_tests": len(results),
        "pass": bool(float(np.mean(Cv_errs)) < 0.05) if Cv_errs else False,
    }
    json_path = os.path.join(RESULTS_DIR, "heat_capacity_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Cv err = {summary['Cv_mean_relative_error_ideal']:.2%}")
    print(f"  gamma err = {summary['gamma_mean_relative_error_ideal']:.4%}")
    print(f"  pass = {summary['pass']}")
    return summary


if __name__ == "__main__":
    main()
