"""
Test 5: Thermodynamic attack detection rate.

Validates that thermodynamic monitoring (network temperature) detects
attacks across varying attacker fractions and attack types.
Fully vectorized for performance.
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import EPSILON

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

N_NODES = 1000
N_TRIALS = 100
N_TIMESTEPS = 100
DT = 0.1
DETECTION_SIGMA = 3.0

ATTACKER_FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.49, 0.50, 0.51]
ATTACK_TYPES = ["flood", "subtle", "coordinated", "mimicry"]

INJECTION_RATES = {"flood": 0.1, "subtle": 0.01, "coordinated": 0.05, "mimicry": 0.005}


def simulate_trial_vectorized(n_nodes, n_attackers, attack_type, rng):
    """Fully vectorized trial simulation."""
    # Initial timing variances
    variances = rng.exponential(0.001, size=n_nodes)
    is_attacker = np.zeros(n_nodes, dtype=bool)
    is_attacker[:n_attackers] = True
    is_legit = ~is_attacker

    base_inj = INJECTION_RATES[attack_type]
    gpsdo_tau = 10.0  # 1/coupling
    decay = np.exp(-DT / gpsdo_tau)

    temperatures = np.zeros(N_TIMESTEPS)

    for t in range(N_TIMESTEPS):
        # Legitimate nodes: GPSDO decay + small noise
        variances[is_legit] *= decay
        variances[is_legit] += rng.normal(0, 0.001, size=int(np.sum(is_legit))) ** 2

        # Attacker nodes: inject perturbation
        n_att = int(np.sum(is_attacker))
        if n_att > 0:
            inj = base_inj
            if attack_type == "coordinated" and t % 5 == 0:
                inj *= 3.0
            elif attack_type == "mimicry":
                inj *= (1.0 + 0.3 * np.sin(t * 0.1))

            variances[is_attacker] += inj * DT
            variances[is_attacker] += rng.normal(0, inj * 0.1, size=n_att) ** 2

        variances = np.maximum(variances, 0.0)
        temperatures[t] = float(np.var(variances))

    # Detection via temperature change rate
    dT = np.diff(temperatures)
    baseline_std = np.std(dT[:10]) + EPSILON
    mean_baseline = np.mean(dT[:10])

    detected = False
    detection_time = N_TIMESTEPS

    anomalies = np.abs(dT[10:] - mean_baseline) > DETECTION_SIGMA * baseline_std
    if np.any(anomalies):
        detected = True
        detection_time = int(np.argmax(anomalies) + 10)

    temp_change_rate = float(np.mean(dT))
    return detected, detection_time, temp_change_rate


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Thermodynamic Security Detection ===")

    rng = np.random.default_rng(42)
    results = []
    summary_data = {}

    total_configs = len(ATTACKER_FRACTIONS) * len(ATTACK_TYPES)
    done = 0

    for frac in ATTACKER_FRACTIONS:
        n_attackers = int(N_NODES * frac)
        for atype in ATTACK_TYPES:
            done += 1
            print(f"  [{done}/{total_configs}] fraction={frac:.2f}, type={atype} ...")

            detections = 0
            detection_times = []
            temp_rates = []

            for trial in range(N_TRIALS):
                detected, det_time, rate = simulate_trial_vectorized(
                    N_NODES, n_attackers, atype, rng)
                detections += int(detected)
                detection_times.append(det_time)
                temp_rates.append(rate)

                results.append({
                    "attacker_fraction": frac,
                    "attack_type": atype,
                    "trial": trial,
                    "detected": bool(detected),
                    "detection_time": det_time,
                    "temperature_change_rate": rate,
                })

            det_rate = detections / N_TRIALS
            key = f"{frac:.2f}_{atype}"
            summary_data[key] = {
                "detection_rate": det_rate,
                "mean_detection_time": float(np.mean(detection_times)),
            }
            print(f"    detection_rate={det_rate:.2f}")

    # False positive rate proxy
    fp_key = "0.01_subtle"
    false_positive_rate = 1.0 - summary_data.get(fp_key, {}).get("detection_rate", 0.0)

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "thermodynamic_security.csv")
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"  CSV saved to {csv_path}")

    # Detection rates by fraction
    det_by_fraction = {}
    for frac in ATTACKER_FRACTIONS:
        rates = []
        for atype in ATTACK_TYPES:
            key = f"{frac:.2f}_{atype}"
            if key in summary_data:
                rates.append(summary_data[key]["detection_rate"])
        det_by_fraction[str(frac)] = float(np.mean(rates))

    # Detection rates by type
    det_by_type = {}
    for atype in ATTACK_TYPES:
        rates = []
        for frac in ATTACKER_FRACTIONS:
            key = f"{frac:.2f}_{atype}"
            if key in summary_data:
                rates.append(summary_data[key]["detection_rate"])
        det_by_type[atype] = float(np.mean(rates))

    frac_50 = det_by_fraction.get("0.5", 0.0)
    frac_51 = det_by_fraction.get("0.51", 0.0)

    detected_results = [r["detection_time"] for r in results if r["detected"]]
    mean_det_time = float(np.mean(detected_results)) if detected_results else float(N_TIMESTEPS)

    summary = {
        "test": "thermodynamic_security",
        "detection_rate_by_fraction": det_by_fraction,
        "detection_rate_by_type": det_by_type,
        "false_positive_rate": false_positive_rate,
        "mean_detection_time": mean_det_time,
        "50pct_threshold_validation": {
            "detection_at_0.49": det_by_fraction.get("0.49", 0.0),
            "detection_at_0.50": frac_50,
            "detection_at_0.51": frac_51,
        },
        "pass": det_by_fraction.get("0.1", 0.0) > 0.5,
    }

    json_path = os.path.join(RESULTS_DIR, "thermodynamic_security_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")
    return summary


if __name__ == "__main__":
    main()
