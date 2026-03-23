"""
Test 6: Variance restoration dynamics.

Validates exponential decay of velocity variance when nodes are coupled
to a zero-temperature reference with coupling time-constant tau.
"""

import os
import sys
import json
import csv
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(__file__))
from utils import NetworkSimulator, k_B, MASS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def exp_decay(t, sigma2_0, tau):
    return sigma2_0 * np.exp(-t / tau)


def run_single(T0, tau_input=0.5, N=100, seed=42):
    V = 50.0
    dim = 3
    dt = 0.005
    sim = NetworkSimulator(N=N, V=V, dim=dim, mode="ideal", dt=dt, seed=seed)
    sim.init_positions_uniform()
    sim.init_velocities_mb(T0)

    # No equilibration -- start hot and cool
    n_steps = 3000
    # We want variance sigma^2(t) = sigma^2_0 * exp(-t/tau).
    # Since sigma^2 ~ v^2, we damp each velocity component by
    # exp(-dt/(2*tau)) per step so that v^2 decays as exp(-dt/tau).
    coupling_factor = np.exp(-dt / (2.0 * tau_input))

    times = []
    variances = []

    for step in range(n_steps):
        sim.step()
        # Coupling: pull velocities toward zero
        sim.velocities *= coupling_factor
        sim.kinetic_energy = 0.5 * MASS * np.sum(sim.velocities ** 2)

        if step % 5 == 0:
            var = float(np.var(sim.velocities))
            times.append(step * dt)
            variances.append(var)

    times = np.array(times)
    variances = np.array(variances)

    # Fit exponential decay
    try:
        popt, _ = curve_fit(exp_decay, times, variances,
                            p0=[variances[0], tau_input], maxfev=5000)
        sigma2_0_fit, tau_fit = popt
    except Exception:
        sigma2_0_fit, tau_fit = float("nan"), float("nan")

    fitted = exp_decay(times, sigma2_0_fit, tau_fit) if np.isfinite(tau_fit) else np.full_like(times, np.nan)
    residuals = variances - fitted

    # Entropy extraction rate: dS/dt = -N*k_B/tau
    dS_dt_theory = -N * k_B / tau_input
    # Measured: from variance drop rate
    if len(variances) > 10 and variances[0] > 0:
        log_var = np.log(variances[variances > 0])
        t_valid = times[:len(log_var)]
        if len(t_valid) > 2:
            slope = np.polyfit(t_valid, log_var, 1)[0]
            # S ~ N*k_B*ln(variance)/2, so dS/dt ~ N*k_B*slope/2
            dS_dt_measured = N * k_B * slope / 2.0
        else:
            dS_dt_measured = float("nan")
    else:
        dS_dt_measured = float("nan")

    return {
        "T0": T0,
        "tau_input": tau_input,
        "tau_fitted": float(tau_fit),
        "sigma2_0_fitted": float(sigma2_0_fit),
        "dS_dt_theory": float(dS_dt_theory),
        "dS_dt_measured": float(dS_dt_measured),
        "times": times.tolist(),
        "variances": variances.tolist(),
        "fitted_variances": fitted.tolist() if np.all(np.isfinite(fitted)) else [],
        "residuals": residuals.tolist() if np.all(np.isfinite(residuals)) else [],
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Variance Restoration ===")

    tau = 0.5
    T0_values = [1.0, 5.0, 10.0, 20.0]
    all_results = []

    for i, T0 in enumerate(T0_values):
        print(f"  T0={T0} ...")
        res = run_single(T0, tau_input=tau, seed=42 + i)
        all_results.append(res)

    # Write per-T0 CSV (time series for each)
    for res in all_results:
        T0 = res["T0"]
        csv_path = os.path.join(RESULTS_DIR, f"variance_restoration_T{T0:.1f}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "variance", "fitted_variance", "residual"])
            fitted = res["fitted_variances"] if res["fitted_variances"] else [float("nan")] * len(res["times"])
            residuals = res["residuals"] if res["residuals"] else [float("nan")] * len(res["times"])
            for t, v, fv, r in zip(res["times"], res["variances"], fitted, residuals):
                w.writerow([t, v, fv, r])

    # Summary
    tau_errors = []
    for res in all_results:
        if np.isfinite(res["tau_fitted"]):
            tau_errors.append(abs(res["tau_fitted"] - tau) / tau)

    summary = {
        "test": "variance_restoration",
        "tau_theoretical": tau,
        "results_per_T0": [
            {"T0": r["T0"], "tau_fitted": r["tau_fitted"],
             "relative_error": abs(r["tau_fitted"] - tau) / tau if np.isfinite(r["tau_fitted"]) else None}
            for r in all_results
        ],
        "mean_tau_relative_error": float(np.mean(tau_errors)) if tau_errors else float("nan"),
        "pass": bool(float(np.mean(tau_errors)) < 0.2) if tau_errors else False,
    }
    json_path = os.path.join(RESULTS_DIR, "variance_restoration_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Mean tau error = {summary['mean_tau_relative_error']:.2%}")
    print(f"  pass = {summary['pass']}")
    return summary


if __name__ == "__main__":
    main()
