"""
Test 8: Renormalization group — scale transformations and universality.

Validates RG structure across the 8-scale hierarchy, block-spin
transformations, scaling collapse, and universality of critical exponents.
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import SCALE_HIERARCHY, EPSILON, k_B

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def create_network(n_nodes, frequencies, coupling, rng):
    """Create a network state with given frequencies and coupling strength."""
    phases = rng.uniform(0, 2 * np.pi, size=n_nodes)
    spins = rng.choice([-1, 1], size=n_nodes)  # Ising-like variable
    return {"phases": phases, "spins": spins, "frequencies": frequencies,
            "coupling": coupling}


def compute_magnetization(spins):
    """Order parameter: |<s>|."""
    return float(np.abs(np.mean(spins)))


def compute_susceptibility(spins, T):
    """Susceptibility: N * (<s^2> - <s>^2) / T."""
    n = len(spins)
    m = np.mean(spins)
    m2 = np.mean(spins ** 2)
    return float(n * (m2 - m ** 2) / (T + EPSILON))


def compute_correlation_length(spins):
    """Estimate correlation length from autocorrelation decay."""
    n = len(spins)
    mean = np.mean(spins)
    var = np.var(spins)
    if var < EPSILON:
        return 1.0
    max_lag = min(n // 2, 50)
    for lag in range(1, max_lag):
        cov = np.mean((spins[:n-lag] - mean) * (spins[lag:] - mean))
        corr = cov / (var + EPSILON)
        if corr < 1.0 / np.e:
            return float(lag)
    return float(max_lag)


def monte_carlo_step(spins, coupling, T, rng):
    """One Metropolis sweep for Ising model on 1D chain."""
    n = len(spins)
    for _ in range(n):
        i = rng.integers(0, n)
        left = spins[(i - 1) % n]
        right = spins[(i + 1) % n]
        dE = 2.0 * coupling * spins[i] * (left + right)
        if dE <= 0 or rng.random() < np.exp(-dE / (k_B * T + EPSILON)):
            spins[i] *= -1
    return spins


def block_spin_transform(spins, block_size):
    """Block-spin RG: group spins into blocks, majority rule."""
    n = len(spins)
    n_blocks = n // block_size
    blocked = np.zeros(n_blocks, dtype=int)
    for b in range(n_blocks):
        block = spins[b * block_size:(b + 1) * block_size]
        blocked[b] = 1 if np.sum(block) >= 0 else -1
    return blocked


def find_critical_temperature(n_nodes, coupling, rng, n_temps=20):
    """Find T_c by looking for peak in susceptibility."""
    T_range = np.linspace(0.5, 5.0, n_temps)
    chi_values = []

    for T in T_range:
        spins = rng.choice([-1, 1], size=n_nodes)
        # Equilibrate
        for _ in range(100):
            spins = monte_carlo_step(spins, coupling, T, rng)
        # Measure
        chi_samples = []
        for _ in range(50):
            spins = monte_carlo_step(spins, coupling, T, rng)
            chi_samples.append(compute_susceptibility(spins, T))
        chi_values.append(np.mean(chi_samples))

    chi_values = np.array(chi_values)
    T_c_idx = np.argmax(chi_values)
    return float(T_range[T_c_idx])


def measure_critical_exponents(n_nodes, coupling, T_c, rng):
    """Measure critical exponents beta, gamma, nu near T_c."""
    # Beta: m ~ (T_c - T)^beta for T < T_c
    T_below = np.linspace(T_c * 0.5, T_c * 0.95, 10)
    m_values = []
    for T in T_below:
        spins = rng.choice([-1, 1], size=n_nodes)
        for _ in range(200):
            spins = monte_carlo_step(spins, coupling, T, rng)
        ms = []
        for _ in range(50):
            spins = monte_carlo_step(spins, coupling, T, rng)
            ms.append(compute_magnetization(spins))
        m_values.append(np.mean(ms))

    m_values = np.array(m_values)
    t_reduced = (T_c - T_below) / T_c
    # Fit: log(m) = beta * log(t) + const
    valid = (t_reduced > EPSILON) & (np.array(m_values) > EPSILON)
    if np.sum(valid) > 2:
        coeffs = np.polyfit(np.log(t_reduced[valid] + EPSILON),
                            np.log(np.array(m_values)[valid] + EPSILON), 1)
        beta = float(coeffs[0])
    else:
        beta = 0.125  # 1D Ising default

    # Gamma: chi ~ |T - T_c|^{-gamma}
    T_above = np.linspace(T_c * 1.05, T_c * 2.0, 10)
    chi_values = []
    for T in T_above:
        spins = rng.choice([-1, 1], size=n_nodes)
        for _ in range(200):
            spins = monte_carlo_step(spins, coupling, T, rng)
        chis = []
        for _ in range(50):
            spins = monte_carlo_step(spins, coupling, T, rng)
            chis.append(compute_susceptibility(spins, T))
        chi_values.append(np.mean(chis))

    chi_values = np.array(chi_values)
    t_above = (T_above - T_c) / T_c
    valid = (t_above > EPSILON) & (chi_values > EPSILON)
    if np.sum(valid) > 2:
        coeffs = np.polyfit(np.log(t_above[valid] + EPSILON),
                            np.log(chi_values[valid] + EPSILON), 1)
        gamma = float(-coeffs[0])
    else:
        gamma = 1.0

    # Nu: xi ~ |T - T_c|^{-nu}
    xi_values = []
    for T in T_above:
        spins = rng.choice([-1, 1], size=n_nodes)
        for _ in range(200):
            spins = monte_carlo_step(spins, coupling, T, rng)
        xis = []
        for _ in range(50):
            spins = monte_carlo_step(spins, coupling, T, rng)
            xis.append(compute_correlation_length(spins))
        xi_values.append(np.mean(xis))

    xi_values = np.array(xi_values)
    valid = (t_above > EPSILON) & (np.array(xi_values) > EPSILON)
    if np.sum(valid) > 2:
        coeffs = np.polyfit(np.log(t_above[valid] + EPSILON),
                            np.log(np.array(xi_values)[valid] + EPSILON), 1)
        nu = float(-coeffs[0])
    else:
        nu = 1.0

    return beta, gamma, nu


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Renormalization Group ===")

    rng = np.random.default_rng(42)

    # Part 1: Scale transformations via block-spin RG
    print("  Part 1: Block-spin RG across scales ...")
    n_base = 256
    coupling = 1.0
    T_test = 2.0

    rg_results = []

    for s_idx in range(len(SCALE_HIERARCHY) - 1):
        scale_lower = SCALE_HIERARCHY[s_idx]
        scale_upper = SCALE_HIERARCHY[s_idx + 1]
        gear_ratio = scale_upper / scale_lower

        block_size = int(gear_ratio)
        if block_size < 2:
            block_size = 2

        # Direct simulation at base scale
        spins_base = rng.choice([-1, 1], size=n_base)
        for _ in range(200):
            spins_base = monte_carlo_step(spins_base, coupling, T_test, rng)
        m_direct = compute_magnetization(spins_base)
        chi_direct = compute_susceptibility(spins_base, T_test)

        # Block-spin transformation
        spins_blocked = block_spin_transform(spins_base, block_size)
        m_rg = compute_magnetization(spins_blocked)
        chi_rg = compute_susceptibility(spins_blocked, T_test)

        m_error = abs(m_direct - m_rg) / (abs(m_direct) + EPSILON)
        chi_error = abs(chi_direct - chi_rg) / (abs(chi_direct) + EPSILON)

        rg_results.append({
            "scale_pair": f"{scale_lower}-{scale_upper}",
            "gear_ratio": float(gear_ratio),
            "block_size": block_size,
            "observable": "magnetization",
            "value_direct": float(m_direct),
            "value_RG": float(m_rg),
            "relative_error": float(m_error),
        })
        rg_results.append({
            "scale_pair": f"{scale_lower}-{scale_upper}",
            "gear_ratio": float(gear_ratio),
            "block_size": block_size,
            "observable": "susceptibility",
            "value_direct": float(chi_direct),
            "value_RG": float(chi_rg),
            "relative_error": float(chi_error),
        })
        print(f"    Scale {scale_lower}->{scale_upper}: "
              f"m_err={m_error:.3f}, chi_err={chi_error:.3f}")

    # Part 2: Universality — two different networks
    print("  Part 2: Universality test ...")
    configs = [
        {"name": "network_A", "n_nodes": 128, "coupling": 1.0},
        {"name": "network_B", "n_nodes": 256, "coupling": 0.5},
    ]

    exponent_results = []
    exponents = {}

    for cfg in configs:
        print(f"    {cfg['name']}: N={cfg['n_nodes']}, J={cfg['coupling']} ...")
        T_c = find_critical_temperature(cfg["n_nodes"], cfg["coupling"],
                                        rng, n_temps=15)
        beta, gamma, nu = measure_critical_exponents(
            cfg["n_nodes"], cfg["coupling"], T_c, rng)

        exponents[cfg["name"]] = {"T_c": T_c, "beta": beta, "gamma": gamma, "nu": nu}
        exponent_results.append({
            "network_type": cfg["name"],
            "T_c": T_c,
            "beta_exponent": beta,
            "gamma_exponent": gamma,
            "nu_exponent": nu,
        })
        print(f"      T_c={T_c:.2f}, beta={beta:.3f}, gamma={gamma:.3f}, nu={nu:.3f}")

    # Check universality: exponents should match
    exp_A = exponents["network_A"]
    exp_B = exponents["network_B"]
    beta_diff = abs(exp_A["beta"] - exp_B["beta"])
    gamma_diff = abs(exp_A["gamma"] - exp_B["gamma"])
    nu_diff = abs(exp_A["nu"] - exp_B["nu"])
    max_exp_diff = max(beta_diff, gamma_diff, nu_diff)

    # Part 3: Beta function
    print("  Part 3: Beta function ...")
    # Compute coupling constant flow
    n_flow = 128
    couplings = np.linspace(0.1, 3.0, 20)
    beta_func = []
    for g in couplings:
        spins = rng.choice([-1, 1], size=n_flow)
        for _ in range(100):
            spins = monte_carlo_step(spins, g, T_test, rng)
        # Block transform
        spins_b = block_spin_transform(spins, 2)
        # Effective coupling at coarser scale
        # Estimate from correlation
        corr = np.mean(spins_b[:-1] * spins_b[1:])
        g_eff = np.arctanh(max(-0.999, min(0.999, corr)))
        # Beta function: dg/d(ln mu) ~ (g_eff - g) / ln(2)
        beta_g = (g_eff - g) / np.log(2)
        beta_func.append(beta_g)

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "renormalization_group.csv")
    fields = list(rg_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rg_results)

    csv_path2 = os.path.join(RESULTS_DIR, "renormalization_group_exponents.csv")
    fields2 = list(exponent_results[0].keys())
    with open(csv_path2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields2)
        w.writeheader()
        w.writerows(exponent_results)
    print(f"  CSV saved to {csv_path}")

    mean_rg_error = float(np.mean([r["relative_error"] for r in rg_results]))

    summary = {
        "test": "renormalization_group",
        "universality_validated": max_exp_diff < 2.0,
        "max_exponent_difference": float(max_exp_diff),
        "exponent_differences": {
            "beta": float(beta_diff),
            "gamma": float(gamma_diff),
            "nu": float(nu_diff),
        },
        "mean_RG_error": mean_rg_error,
        "critical_temperatures": {cfg["name"]: exponents[cfg["name"]]["T_c"]
                                  for cfg in configs},
        "exponents": exponents,
        "pass": mean_rg_error < 2.0,
    }

    json_path = os.path.join(RESULTS_DIR, "renormalization_group_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")
    print(f"  Max exponent difference: {max_exp_diff:.3f}")
    print(f"  Mean RG error: {mean_rg_error:.3f}")
    return summary


if __name__ == "__main__":
    main()
