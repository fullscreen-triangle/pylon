"""
Test 10: Phonon dispersion relation in crystal phase.

Validates omega^2(q) = omega_0^2 [1 - cos(qa)] for a 1D chain of
harmonic oscillators (linearised LJ potential).

Uses a harmonic chain (exact LJ spring constant at equilibrium) with
initial conditions exciting multiple wavevectors, then extracts
dispersion from spatiotemporal FFT.
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import k_B, MASS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Phonon Dispersion ===")

    N = 64  # power of 2 for clean FFT
    epsilon = 1.0
    sigma_lj = 1.0

    # LJ equilibrium spacing
    r_eq = 2.0 ** (1.0 / 6.0) * sigma_lj
    a = r_eq  # lattice spacing

    # LJ second derivative at equilibrium
    # U(r) = 4e[(s/r)^12 - (s/r)^6]
    # U''(r_eq) = 4e * [156*s^12/r^14 - 42*s^6/r^8] at r_eq
    sr = sigma_lj / r_eq
    sr6 = sr ** 6
    sr12 = sr6 ** 2
    K = 4.0 * epsilon * (156.0 * sr12 - 42.0 * sr6) / r_eq**2

    # Dispersion: omega^2(q) = (2K/m)(1 - cos(qa))
    omega_0_sq = 2.0 * K / MASS
    omega_max = np.sqrt(4.0 * K / MASS)
    print(f"  K = {K:.4f}, omega_0^2 = {omega_0_sq:.4f}, omega_max = {omega_max:.4f}")

    # Time step must resolve highest frequency: dt << 2*pi/omega_max
    period_min = 2.0 * np.pi / omega_max
    dt = period_min / 40.0  # 40 samples per shortest period
    print(f"  dt = {dt:.6f}, period_min = {period_min:.4f}")

    # Use harmonic potential directly (linearised LJ) for clean results
    # F_i = K * (u_{i+1} - 2*u_i + u_{i-1}) where u_i is displacement
    L = N * a

    # Initial displacement: superposition of several modes
    displacements = np.zeros(N)
    velocities = np.zeros(N)
    rng = np.random.default_rng(42)

    # Excite modes by giving random initial velocities
    for q_idx in range(1, N // 2):
        amp = 0.01 * a / np.sqrt(q_idx)
        phase = rng.uniform(0, 2 * np.pi)
        for j in range(N):
            velocities[j] += amp * omega_max * np.sin(2 * np.pi * q_idx * j / N + phase)

    # Time evolution with harmonic forces (velocity Verlet)
    # Need enough time samples to resolve frequency bins
    n_steps = 2048  # power of 2
    disp_history = np.zeros((n_steps, N))

    positions = np.arange(N, dtype=float) * a + displacements
    m = MASS

    def harmonic_forces(disp):
        """Forces from harmonic nearest-neighbour springs with PBC."""
        f = np.zeros(N)
        for i in range(N):
            left = (i - 1) % N
            right = (i + 1) % N
            f[i] = K * (disp[right] - 2.0 * disp[i] + disp[left])
        return f

    disp = displacements.copy()
    vel = velocities.copy()
    forces = harmonic_forces(disp)

    for step in range(n_steps):
        # Record
        disp_history[step] = disp

        # Velocity Verlet
        vel += 0.5 * dt * forces / m
        disp += dt * vel
        forces = harmonic_forces(disp)
        vel += 0.5 * dt * forces / m

    print(f"  Simulation complete ({n_steps} steps), computing FFT ...")

    # 2D FFT: axis 0 = time, axis 1 = space
    ft = np.fft.fft2(disp_history)
    power = np.abs(ft) ** 2

    # Axes
    freq = np.fft.fftfreq(n_steps, d=dt)
    omega_axis = 2.0 * np.pi * np.abs(freq)
    q_axis = np.fft.fftfreq(N, d=a) * 2.0 * np.pi  # wavevector

    # Extract dispersion: for each q, find omega with peak power
    n_q = N // 2
    results_rows = []

    for qi in range(1, n_q):
        q_val = abs(q_axis[qi])
        # Look at positive frequencies only
        power_slice = power[:n_steps // 2, qi]
        omega_pos = omega_axis[:n_steps // 2]

        # Find peak (exclude DC)
        peak_idx = np.argmax(power_slice[1:]) + 1
        omega_meas = omega_pos[peak_idx]

        omega_th = np.sqrt(max(omega_0_sq * (1.0 - np.cos(q_val * a)), 0))
        rel_err = abs(omega_meas - omega_th) / omega_th if omega_th > 1e-10 else float("nan")

        results_rows.append({
            "wavevector_q": float(q_val),
            "omega_measured": float(omega_meas),
            "omega_theory": float(omega_th),
            "relative_error": float(rel_err),
        })

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "phonon_dispersion.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
        w.writeheader()
        w.writerows(results_rows)
    print(f"  CSV saved to {csv_path}")

    rel_errs = np.array([r["relative_error"] for r in results_rows
                         if np.isfinite(r["relative_error"])])
    mean_err = float(np.mean(rel_errs)) if len(rel_errs) > 0 else float("nan")
    max_err = float(np.max(rel_errs)) if len(rel_errs) > 0 else float("nan")

    # Fit omega_0^2 from data
    q_arr = np.array([r["wavevector_q"] for r in results_rows])
    om_arr = np.array([r["omega_measured"] for r in results_rows])
    factor = 1.0 - np.cos(q_arr * a)
    mask = factor > 0.01
    if np.sum(mask) > 2:
        omega_0_sq_fitted = float(np.mean(om_arr[mask]**2 / factor[mask]))
    else:
        omega_0_sq_fitted = float("nan")

    summary = {
        "test": "phonon_dispersion",
        "omega_0_sq_theory": float(omega_0_sq),
        "omega_0_sq_fitted": omega_0_sq_fitted,
        "omega_0_relative_error": abs(omega_0_sq_fitted - omega_0_sq) / omega_0_sq
            if np.isfinite(omega_0_sq_fitted) and omega_0_sq > 0 else float("nan"),
        "mean_relative_error": mean_err,
        "max_relative_error": max_err,
        "pass": bool(mean_err < 0.1) if np.isfinite(mean_err) else False,
    }
    json_path = os.path.join(RESULTS_DIR, "phonon_dispersion_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Mean relative error = {mean_err:.6f}")
    print(f"  omega_0^2: theory={omega_0_sq:.4f}, fitted={omega_0_sq_fitted:.4f}")
    print(f"  pass = {summary['pass']}")
    return summary


if __name__ == "__main__":
    main()
