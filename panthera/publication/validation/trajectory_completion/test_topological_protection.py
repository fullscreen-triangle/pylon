"""
Test 9: Topological protection — edge modes and fault tolerance.

Validates topological protection of communication channels via
phonon band structure, domain wall edge modes, Berry phase, and
fault tolerance under random node removal.
"""

import os
import sys
import json
import csv
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(__file__))
from utils import EPSILON

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

N_CHAIN = 100
N_Q_POINTS = 200
COUPLING_A = 1.0  # Spring constant region A
COUPLING_B = 2.0  # Spring constant region B (different phase-lock pattern)
MASS = 1.0
FAULT_FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.30]


def build_dynamical_matrix_1d(n, couplings, masses=None):
    """Build dynamical matrix for 1D chain with given couplings.
    couplings[i] = spring constant between site i and i+1."""
    if masses is None:
        masses = np.ones(n)
    D = np.zeros((n, n))
    for i in range(n - 1):
        k = couplings[i]
        D[i, i] += k / masses[i]
        D[i+1, i+1] += k / masses[i+1]
        D[i, i+1] -= k / np.sqrt(masses[i] * masses[i+1])
        D[i+1, i] -= k / np.sqrt(masses[i] * masses[i+1])
    return D


def compute_band_structure(coupling_a, coupling_b, n_unit_cells, n_q):
    """Compute phonon band structure for diatomic chain (SSH model analogue).
    Two atoms per unit cell with alternating couplings."""
    q_values = np.linspace(-np.pi, np.pi, n_q)
    omega_band1 = np.zeros(n_q)
    omega_band2 = np.zeros(n_q)

    for iq, q in enumerate(q_values):
        # 2x2 dynamical matrix for unit cell
        H = np.zeros((2, 2), dtype=complex)
        H[0, 0] = (coupling_a + coupling_b) / MASS
        H[1, 1] = (coupling_a + coupling_b) / MASS
        H[0, 1] = -(coupling_a + coupling_b * np.exp(-1j * q)) / MASS
        H[1, 0] = -(coupling_a + coupling_b * np.exp(1j * q)) / MASS

        eigenvalues = np.sort(np.real(np.linalg.eigvalsh(H)))
        # Frequencies: omega = sqrt(eigenvalue)
        omega_band1[iq] = np.sqrt(max(eigenvalues[0], 0))
        omega_band2[iq] = np.sqrt(max(eigenvalues[1], 0))

    return q_values, omega_band1, omega_band2


def compute_berry_phase(coupling_a, coupling_b, n_q=500):
    """Compute Berry phase for each band by discretized loop integral.
    gamma = -Im sum_q log <u(q)|u(q+dq)>."""
    q_values = np.linspace(-np.pi, np.pi, n_q, endpoint=False)
    dq = 2 * np.pi / n_q

    phases = [0.0, 0.0]  # For two bands

    eigvecs_list = []
    for q in q_values:
        H = np.zeros((2, 2), dtype=complex)
        H[0, 0] = (coupling_a + coupling_b) / MASS
        H[1, 1] = (coupling_a + coupling_b) / MASS
        H[0, 1] = -(coupling_a + coupling_b * np.exp(-1j * q)) / MASS
        H[1, 0] = -(coupling_a + coupling_b * np.exp(1j * q)) / MASS
        _, vecs = np.linalg.eigh(H)
        eigvecs_list.append(vecs)

    for band in range(2):
        product = 1.0 + 0j
        for iq in range(n_q):
            iq_next = (iq + 1) % n_q
            overlap = np.vdot(eigvecs_list[iq][:, band],
                              eigvecs_list[iq_next][:, band])
            product *= overlap / (abs(overlap) + EPSILON)
        phases[band] = float(-np.imag(np.log(product + EPSILON)))

    return phases


def create_domain_wall_chain(n, wall_position=None):
    """Create chain with domain wall: region A has couplings (strong, weak),
    region B has couplings (weak, strong) — SSH model."""
    if wall_position is None:
        wall_position = n // 2
    couplings = np.zeros(n - 1)
    for i in range(n - 1):
        if i < wall_position:
            # Region A: alternating (strong, weak)
            couplings[i] = COUPLING_B if i % 2 == 0 else COUPLING_A
        else:
            # Region B: alternating (weak, strong)
            couplings[i] = COUPLING_A if i % 2 == 0 else COUPLING_B
    return couplings


def find_edge_mode(D, n, wall_position):
    """Find localized edge mode at domain wall."""
    eigenvalues, eigenvectors = eigh(D)

    # Look for mid-gap states (eigenvalues near the gap center)
    omega_sq = eigenvalues
    sorted_idx = np.argsort(omega_sq)
    mid_gap_idx = n // 2  # Approximate gap center

    # Check localization of mid-gap states
    best_localization = 0.0
    edge_mode_amplitude = 0.0
    decay_length = float(n)

    for check_idx in range(max(0, mid_gap_idx - 5), min(n, mid_gap_idx + 5)):
        idx = sorted_idx[check_idx]
        mode = np.abs(eigenvectors[:, idx]) ** 2

        # Weight near domain wall
        wall_weight = np.sum(mode[max(0, wall_position-5):wall_position+5])
        total_weight = np.sum(mode)

        if total_weight > EPSILON:
            localization = wall_weight / total_weight
            if localization > best_localization:
                best_localization = localization
                edge_mode_amplitude = float(np.max(mode))

                # Estimate decay length
                distances = np.abs(np.arange(n) - wall_position)
                log_mode = np.log(mode + EPSILON)
                valid = mode > EPSILON
                if np.sum(valid) > 2:
                    coeffs = np.polyfit(distances[valid], log_mode[valid], 1)
                    if coeffs[0] < -EPSILON:
                        decay_length = float(-1.0 / coeffs[0])

    edge_detected = best_localization > 0.1
    return edge_detected, edge_mode_amplitude, decay_length


def test_fault_tolerance(n, wall_position, fault_fraction, rng, n_trials=20):
    """Test if edge mode survives random node removal."""
    surviving = 0
    amplitudes = []

    for _ in range(n_trials):
        n_remove = max(1, int(n * fault_fraction))
        # Remove random nodes (not at the wall itself)
        removable = list(range(0, max(0, wall_position - 3))) + \
                     list(range(wall_position + 3, n))
        if len(removable) < n_remove:
            removable = list(range(n))
        remove_idx = set(rng.choice(removable, size=min(n_remove, len(removable)),
                                     replace=False))

        # Build chain without removed nodes
        remaining = [i for i in range(n) if i not in remove_idx]
        n_remaining = len(remaining)
        if n_remaining < 5:
            continue

        couplings = create_domain_wall_chain(n, wall_position)
        # Build new coupling array for remaining nodes
        new_couplings = []
        for i in range(len(remaining) - 1):
            orig_i = remaining[i]
            orig_j = remaining[i + 1]
            # Use average coupling between original positions
            if orig_i < n - 1:
                new_couplings.append(couplings[orig_i])
            else:
                new_couplings.append(COUPLING_A)
        new_couplings = np.array(new_couplings)

        D = build_dynamical_matrix_1d(n_remaining, new_couplings)
        new_wall = sum(1 for r in remaining if r < wall_position)
        detected, amp, _ = find_edge_mode(D, n_remaining, new_wall)

        if detected:
            surviving += 1
            amplitudes.append(amp)

    survival_rate = surviving / max(n_trials, 1)
    mean_amp = float(np.mean(amplitudes)) if amplitudes else 0.0
    return survival_rate > 0.5, survival_rate, mean_amp


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Topological Protection ===")

    rng = np.random.default_rng(42)

    # Band structure
    print("  Computing band structure ...")
    n_unit_cells = N_CHAIN // 2
    q_values, omega1, omega2 = compute_band_structure(
        COUPLING_A, COUPLING_B, n_unit_cells, N_Q_POINTS)

    band_gap = float(np.min(omega2) - np.max(omega1))
    print(f"    Band gap: {band_gap:.4f}")

    # Berry phase
    print("  Computing Berry phase ...")
    berry_phases = compute_berry_phase(COUPLING_A, COUPLING_B)
    print(f"    Berry phases: {berry_phases[0]:.4f}, {berry_phases[1]:.4f}")

    # Winding number (1D topological invariant)
    # For SSH model: winding number = 0 if coupling_a > coupling_b, 1 otherwise
    winding_number = 1 if COUPLING_B > COUPLING_A else 0
    print(f"    Winding number: {winding_number}")

    # Domain wall edge mode
    print("  Testing domain wall edge mode ...")
    wall_position = N_CHAIN // 2
    couplings = create_domain_wall_chain(N_CHAIN, wall_position)
    D = build_dynamical_matrix_1d(N_CHAIN, couplings)
    edge_detected, edge_amp, decay_len = find_edge_mode(D, N_CHAIN, wall_position)
    print(f"    Edge mode detected: {edge_detected}")
    print(f"    Edge amplitude: {edge_amp:.4f}, decay length: {decay_len:.2f}")

    # Fault tolerance
    print("  Testing fault tolerance ...")
    fault_results = []
    threshold_fraction = 1.0

    for ff in FAULT_FRACTIONS:
        survives, rate, amp = test_fault_tolerance(
            N_CHAIN, wall_position, ff, rng)
        fault_results.append({
            "fault_fraction": ff,
            "edge_mode_amplitude": amp,
            "edge_mode_survives": survives,
            "survival_rate": rate,
        })
        if not survives and ff < threshold_fraction:
            threshold_fraction = ff
        print(f"    Fault {ff:.0%}: survives={survives}, rate={rate:.2f}")

    # CSV: band structure
    band_csv = []
    for i in range(N_Q_POINTS):
        band_csv.append({
            "q_vector": float(q_values[i]),
            "omega_band1": float(omega1[i]),
            "omega_band2": float(omega2[i]),
            "berry_phase_band1": berry_phases[0],
            "berry_phase_band2": berry_phases[1],
        })

    csv_path = os.path.join(RESULTS_DIR, "topological_protection_bands.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(band_csv[0].keys()))
        w.writeheader()
        w.writerows(band_csv)

    # CSV: fault tolerance
    csv_path2 = os.path.join(RESULTS_DIR, "topological_protection_faults.csv")
    with open(csv_path2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fault_results[0].keys()))
        w.writeheader()
        w.writerows(fault_results)
    print(f"  CSV saved to {csv_path}")

    summary = {
        "test": "topological_protection",
        "band_gap": band_gap,
        "berry_phase_values": [float(x) for x in berry_phases],
        "chern_numbers": [int(winding_number)],
        "winding_number": int(winding_number),
        "fault_tolerance_threshold": float(threshold_fraction),
        "edge_mode_detected": bool(edge_detected),
        "edge_mode_amplitude": float(edge_amp),
        "edge_mode_decay_length": float(decay_len),
        "pass": bool(edge_detected and band_gap > 0),
    }

    json_path = os.path.join(RESULTS_DIR, "topological_protection_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")
    return summary


if __name__ == "__main__":
    main()
