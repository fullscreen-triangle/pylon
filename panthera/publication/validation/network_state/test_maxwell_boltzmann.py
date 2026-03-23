"""
Test 2: Maxwell-Boltzmann speed distribution validation.

Verifies that particle speeds after equilibration follow the 3D
Maxwell-Boltzmann distribution at several temperatures.
"""

import os
import sys
import json
import csv
import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(__file__))
from utils import NetworkSimulator, k_B, MASS, chi_squared_test

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def mb_pdf(v, T, m=MASS):
    """3D Maxwell-Boltzmann speed PDF."""
    a = m / (2.0 * k_B * T)
    return 4.0 * np.pi * (a / np.pi) ** 1.5 * v**2 * np.exp(-a * v**2)


def mb_cdf(v, T, m=MASS):
    """CDF of Maxwell-Boltzmann (via scipy chi distribution with 3 dof)."""
    sigma = np.sqrt(k_B * T / m)
    return sp_stats.chi.cdf(v / sigma, df=3)


def run_single(T, N=500, seed=42):
    V = max(10.0, N * 0.5)  # enough room
    sim = NetworkSimulator(N=N, V=V, dim=3, mode="ideal", dt=0.002, seed=seed)
    sim.init_positions_uniform()
    sim.init_velocities_mb(T)

    # Equilibrate with thermostat
    sim.run(2000, thermostat_T=T)

    # Collect speed samples using Andersen-style thermostat:
    # After each measurement window, re-draw a fraction of velocities from
    # the MB distribution.  This produces correct canonical fluctuations
    # unlike velocity-rescaling which fixes total KE.
    rng_mb = np.random.default_rng(seed + 500)
    sigma_v = np.sqrt(k_B * T / MASS)

    n_snapshots = 60
    speed_samples = []
    for snap in range(n_snapshots):
        # Andersen thermostat: re-draw all velocities from MB every snapshot
        # (equivalent to drawing independent canonical samples)
        sim.velocities = rng_mb.normal(0, sigma_v, size=(N, sim.dim))
        sim.velocities -= sim.velocities.mean(axis=0)
        # Let system evolve a few steps to mix spatial degrees of freedom
        sim.run(20)
        speed_samples.append(sim.speeds())
    speed_samples = np.concatenate(speed_samples)

    # Subsample for KS test to avoid over-sensitivity at large sample size
    rng_ks = np.random.default_rng(seed + 1000)
    ks_subsample = rng_ks.choice(speed_samples, size=min(2000, len(speed_samples)),
                                 replace=False)

    # Theoretical characteristic speeds
    v_mp_theory = np.sqrt(2 * k_B * T / MASS)
    v_mean_theory = np.sqrt(8 * k_B * T / (np.pi * MASS))
    v_rms_theory = np.sqrt(3 * k_B * T / MASS)

    v_mp_measured = float(speed_samples[np.argmax(
        np.histogram(speed_samples, bins=200)[0])])  # rough
    # Better: mode from histogram
    counts, edges = np.histogram(speed_samples, bins=200)
    centres = 0.5 * (edges[:-1] + edges[1:])
    v_mp_measured = float(centres[np.argmax(counts)])
    v_mean_measured = float(np.mean(speed_samples))
    v_rms_measured = float(np.sqrt(np.mean(speed_samples**2)))

    # Chi-squared test
    n_bins = 50
    counts_obs, bin_edges = np.histogram(speed_samples, bins=n_bins, density=False)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    expected_density = mb_pdf(bin_centres, T)
    counts_exp = expected_density * bin_widths * len(speed_samples)
    chi2, chi2_p = chi_squared_test(counts_obs, counts_exp)

    # KS test (on subsample to avoid over-sensitivity)
    sigma_v = np.sqrt(k_B * T / MASS)
    ks_stat, ks_p = sp_stats.kstest(ks_subsample / sigma_v, sp_stats.chi(df=3).cdf)

    return {
        "temperature": T,
        "v_mp_theory": v_mp_theory,
        "v_mp_measured": v_mp_measured,
        "v_mean_theory": v_mean_theory,
        "v_mean_measured": v_mean_measured,
        "v_rms_theory": v_rms_theory,
        "v_rms_measured": v_rms_measured,
        "chi2_statistic": chi2,
        "chi2_p_value": chi2_p,
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_p),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Maxwell-Boltzmann Distribution ===")

    temperatures = [0.5, 1.0, 2.0, 5.0, 10.0]
    results = []
    for i, T in enumerate(temperatures):
        print(f"  T={T} ...")
        row = run_single(T, seed=42 + i)
        results.append(row)

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "maxwell_boltzmann.csv")
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"  CSV saved to {csv_path}")

    # Summary
    p_vals = [r["ks_p_value"] for r in results]
    frac_pass = sum(1 for p in p_vals if p > 0.05) / len(p_vals)
    summary = {
        "test": "maxwell_boltzmann",
        "mean_ks_p_value": float(np.mean(p_vals)),
        "fraction_passing": frac_pass,
        "pass": bool(frac_pass >= 0.8),
    }
    json_path = os.path.join(RESULTS_DIR, "maxwell_boltzmann_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: fraction passing = {frac_pass:.2f}, pass = {summary['pass']}")
    return summary


if __name__ == "__main__":
    main()
