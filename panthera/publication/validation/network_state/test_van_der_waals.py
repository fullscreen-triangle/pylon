"""
Test 4: Van der Waals equation of state corrections.

Measures compressibility factor Z = PV/(NkT) across a range of densities
for LJ particles and fits virial and van-der-Waals parameters.
"""

import os
import sys
import json
import csv
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(__file__))
from utils import NetworkSimulator, k_B, MASS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def compute_B2_theoretical(T, epsilon=1.0, sigma=1.0):
    """Second virial coefficient from LJ potential via numerical integration.
    B2(T) = -2 pi int_0^inf [exp(-U(r)/kT) - 1] r^2 dr
    """
    def integrand(r):
        if r < 0.5 * sigma:
            return -r**2  # exp(-huge) - 1 ~ -1
        sr6 = (sigma / r) ** 6
        U = 4.0 * epsilon * (sr6**2 - sr6)
        return (np.exp(-U / (k_B * T)) - 1.0) * r**2

    result, _ = quad(integrand, 0.01 * sigma, 10.0 * sigma, limit=200)
    return -2.0 * np.pi * result


def virial_func(rho, B2, B3):
    return 1.0 + B2 * rho + B3 * rho**2


def vdw_Z(rho, a, b):
    """Van der Waals Z = PV/NkT expressed as function of rho."""
    # (P + a rho^2)(1/rho - b) = kT  =>  P = kT/(1/rho - b) - a rho^2
    # Z = P / (rho kT)
    # We fix T externally; this is parameterised for curve_fit
    # T is captured from closure
    return 1.0 / (1.0 - b * rho) - a * rho / (k_B * vdw_Z._T)

vdw_Z._T = 2.0  # will be set before fitting


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Van der Waals Equation of State ===")

    T = 2.0
    vdw_Z._T = T
    N_base = 100
    dim = 3
    n_density = 20
    densities = np.linspace(0.01, 0.8, n_density)

    B2_theory = compute_B2_theoretical(T)
    print(f"  B2 theoretical = {B2_theory:.4f}")

    results = []
    for idx, rho in enumerate(densities):
        V = N_base / rho
        L = V ** (1.0 / dim)

        # Skip if box too small for LJ cutoff
        if L < 2.5 * 2:
            # Reduce N to keep L reasonable
            N = max(int(rho * (5.5**dim)), 20)
            V = N / rho
        else:
            N = N_base

        print(f"  [{idx+1}/{n_density}] rho={rho:.3f}, N={N}, V={V:.2f}")

        sim = NetworkSimulator(N=N, V=V, dim=dim, mode="lj",
                               epsilon=1.0, sigma=1.0, dt=0.002, seed=42+idx)
        sim.init_positions_lattice()
        sim.init_velocities_mb(T)

        # Equilibrate
        sim.run(3000, thermostat_T=T)
        sim.reset_pressure_accumulators()

        # Production
        prod_steps = 5000
        T_accum = 0.0
        for _ in range(prod_steps):
            sim.step(thermostat_T=T)
            T_accum += sim.temperature()

        T_meas = T_accum / prod_steps
        P_meas = sim.pressure_virial()
        Z_meas = P_meas * V / (N * k_B * T_meas) if T_meas > 0 else float("nan")

        results.append({
            "density": float(rho),
            "N": N,
            "V": float(V),
            "pressure": float(P_meas),
            "temperature": float(T_meas),
            "Z_measured": float(Z_meas),
        })

    # Fit virial expansion
    rho_arr = np.array([r["density"] for r in results])
    Z_arr = np.array([r["Z_measured"] for r in results])
    valid = np.isfinite(Z_arr)

    try:
        popt_v, _ = curve_fit(virial_func, rho_arr[valid], Z_arr[valid],
                              p0=[B2_theory, 0.0])
        B2_fitted, B3_fitted = popt_v
    except Exception:
        B2_fitted, B3_fitted = float("nan"), float("nan")

    # Fit vdW
    try:
        popt_w, _ = curve_fit(lambda rho, a, b: 1.0/(1.0 - b*rho) - a*rho/(k_B*T),
                              rho_arr[valid], Z_arr[valid],
                              p0=[1.0, 0.5], maxfev=5000)
        a_fitted, b_fitted = popt_w
    except Exception:
        a_fitted, b_fitted = float("nan"), float("nan")

    # Add predictions to results
    for r in results:
        rho = r["density"]
        r["Z_virial_prediction"] = float(virial_func(rho, B2_fitted, B3_fitted))
        if np.isfinite(a_fitted):
            r["Z_vdw_prediction"] = float(
                1.0/(1.0 - b_fitted*rho) - a_fitted*rho/(k_B*T))
        else:
            r["Z_vdw_prediction"] = float("nan")

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "van_der_waals.csv")
    fields = ["density", "Z_measured", "Z_virial_prediction", "Z_vdw_prediction",
              "pressure", "temperature"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"  CSV saved to {csv_path}")

    rel_err_B2 = abs(B2_fitted - B2_theory) / abs(B2_theory) if B2_theory != 0 else float("nan")

    # Boyle temperature: T where B2(T) = 0
    from scipy.optimize import brentq
    try:
        T_boyle = brentq(lambda t: compute_B2_theoretical(t), 0.5, 50.0)
    except Exception:
        T_boyle = float("nan")

    summary = {
        "test": "van_der_waals",
        "B2_fitted": float(B2_fitted),
        "B2_theoretical": float(B2_theory),
        "B2_relative_error": float(rel_err_B2),
        "B3_fitted": float(B3_fitted),
        "a_fitted": float(a_fitted),
        "b_fitted": float(b_fitted),
        "Boyle_temperature": float(T_boyle),
        "pass": bool(rel_err_B2 < 0.5),  # within 50% for this simulation size
    }
    json_path = os.path.join(RESULTS_DIR, "van_der_waals_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  B2: fitted={B2_fitted:.4f} theory={B2_theory:.4f} err={rel_err_B2:.2%}")
    print(f"  pass = {summary['pass']}")
    return summary


if __name__ == "__main__":
    main()
