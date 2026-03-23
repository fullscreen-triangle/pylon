"""
Test 10: Phonon dispersion relation in crystal phase.

Validates omega^2(q) = omega_0^2 [1 - cos(qa)] for a 1D chain of
nodes interacting via Lennard-Jones potential.
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import k_B, MASS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def lj_force(r, epsilon=1.0, sigma=1.0):
    """LJ force magnitude (scalar, for 1D)."""
    sr6 = (sigma / r) ** 6
    return 24.0 * epsilon * (2.0 * sr6**2 - sr6) / r


def lj_second_derivative(r_eq, epsilon=1.0, sigma=1.0):
    """U''(r) at equilibrium for spring constant."""
    sr = sigma / r_eq
    sr6 = sr ** 6
    sr12 = sr6 ** 2
    # U(r) = 4e(sr12 - sr6)
    # U''(r) = 4e [12*13*sigma^12/r^14 - 6*7*sigma^6/r^8]
    return 4.0 * epsilon * (156.0 * sr12 / r_eq**2 - 42.0 * sr6 / r_eq**2)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Phonon Dispersion ===")

    N = 100
    epsilon = 1.0
    sigma = 1.0
    # Equilibrium spacing: LJ minimum at r_eq = 2^(1/6) * sigma
    r_eq = 2.0 ** (1.0 / 6.0) * sigma
    a = r_eq  # lattice spacing
    L = N * a
    dt = 0.001  # small dt for phonon accuracy
    m = MASS

    # Spring constant from LJ
    K = lj_second_derivative(r_eq, epsilon, sigma)
    omega_0_sq = K / m  # = 4 * K/m for the standard 1D dispersion
    # For nearest-neighbour 1D chain: omega^2(q) = (2K/m)(1 - cos(qa))
    # so omega_0^2 = 2K/m
    omega_0_sq_theory = 2.0 * K / m

    print(f"  K = {K:.4f}, omega_0^2 = {omega_0_sq_theory:.4f}")

    # Initialise 1D chain at equilibrium + small perturbation
    positions = np.arange(N, dtype=np.float64) * a + a / 2.0  # (N,)
    velocities = np.zeros(N)

    # Perturb: displace node 0 slightly
    displacements = np.zeros(N)
    displacements[N // 2] = 0.01 * a

    positions += displacements

    # Time evolution using velocity Verlet with nearest-neighbour LJ forces
    # and periodic boundary conditions
    n_steps = 4000
    n_record = n_steps
    disp_history = np.zeros((n_record, N))

    def compute_forces_1d(pos):
        forces = np.zeros(N)
        for i in range(N):
            j = (i + 1) % N
            dr = pos[j] - pos[i]
            # Minimum image
            dr -= L * np.round(dr / L)
            r = abs(dr)
            if r < 0.5 * sigma:
                r = 0.5 * sigma
            f = lj_force(r, epsilon, sigma)
            sign = 1.0 if dr > 0 else -1.0
            forces[i] += f * sign
            forces[j] -= f * sign
        return forces

    forces = compute_forces_1d(positions)

    for step in range(n_steps):
        # Velocity Verlet
        velocities += 0.5 * dt * forces / m
        positions += dt * velocities
        # PBC
        positions %= L
        forces = compute_forces_1d(positions)
        velocities += 0.5 * dt * forces / m

        # Record displacements from equilibrium
        eq_pos = np.arange(N) * a + a / 2.0
        disp = positions - eq_pos
        disp -= L * np.round(disp / L)
        disp_history[step] = disp

    print(f"  Simulation complete, computing FFT ...")

    # 2D FFT: spatial (q) and temporal (omega)
    # disp_history shape: (n_time, N)
    ft = np.fft.fft2(disp_history)
    power = np.abs(ft) ** 2

    # Frequency and wavevector axes
    freq = np.fft.fftfreq(n_record, d=dt)  # Hz
    omega_axis = 2.0 * np.pi * freq  # angular frequency
    q_axis = np.fft.fftfreq(N, d=a) * 2.0 * np.pi  # wavevector

    # Extract dispersion: for each q, find omega with peak power
    n_q = N // 2  # positive q only
    q_measured = np.zeros(n_q)
    omega_measured = np.zeros(n_q)
    omega_theory = np.zeros(n_q)

    for qi in range(1, n_q):
        q_val = q_axis[qi]
        power_slice = power[:n_record // 2, qi]  # positive frequencies only
        peak_idx = np.argmax(power_slice)
        omega_meas = abs(omega_axis[peak_idx])

        q_measured[qi] = q_val
        omega_measured[qi] = omega_meas
        omega_theory[qi] = np.sqrt(max(omega_0_sq_theory * (1.0 - np.cos(q_val * a)), 0))

    # Skip q=0 (zero frequency mode)
    valid = (q_measured != 0) & (omega_theory > 0)
    q_out = q_measured[valid]
    om_m = omega_measured[valid]
    om_t = omega_theory[valid]
    rel_err = np.abs(om_m - om_t) / om_t

    # CSV
    rows = []
    for i in range(len(q_out)):
        rows.append({
            "wavevector_q": float(q_out[i]),
            "omega_measured": float(om_m[i]),
            "omega_theory": float(om_t[i]),
            "relative_error": float(rel_err[i]),
        })

    csv_path = os.path.join(RESULTS_DIR, "phonon_dispersion.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV saved to {csv_path}")

    # Fit omega_0 from data
    # omega^2 = omega_0^2 * (1 - cos(qa))
    # Fit omega_0^2
    cos_qa = np.cos(q_out * a)
    factor = 1.0 - cos_qa
    mask = factor > 0.01  # avoid near-zero denominator
    if np.sum(mask) > 2:
        omega_0_sq_fitted = float(np.mean(om_m[mask]**2 / factor[mask]))
    else:
        omega_0_sq_fitted = float("nan")

    summary = {
        "test": "phonon_dispersion",
        "omega_0_sq_theory": float(omega_0_sq_theory),
        "omega_0_sq_fitted": omega_0_sq_fitted,
        "omega_0_relative_error": abs(omega_0_sq_fitted - omega_0_sq_theory) / omega_0_sq_theory
            if np.isfinite(omega_0_sq_fitted) and omega_0_sq_theory > 0 else float("nan"),
        "mean_relative_error": float(np.mean(rel_err)) if len(rel_err) > 0 else float("nan"),
        "max_relative_error": float(np.max(rel_err)) if len(rel_err) > 0 else float("nan"),
        "pass": bool(float(np.mean(rel_err)) < 0.5) if len(rel_err) > 0 else False,
    }
    json_path = os.path.join(RESULTS_DIR, "phonon_dispersion_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Mean relative error = {summary['mean_relative_error']:.4f}")
    print(f"  pass = {summary['pass']}")
    return summary


if __name__ == "__main__":
    main()
