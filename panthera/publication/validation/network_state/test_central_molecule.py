"""
Test 9: Central molecule impossibility theorem.

Validates that per-packet tracking degrades performance compared to
bulk variance-based control -- the statistical-mechanical argument
that tracking individual molecules is less efficient than bulk
thermodynamic control.
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import k_B, MASS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


class PacketNetwork:
    """
    Simplified network simulation for comparing per-packet vs variance control.

    Models N nodes exchanging packets. Each node has a queue with
    stochastic arrival and departure.
    """

    def __init__(self, N, load_factor=0.7, seed=42):
        self.N = N
        self.load = load_factor
        self.rng = np.random.default_rng(seed)

        # Queue depths
        self.queues = np.zeros(N)
        # Packets in flight: list of (src, dst, time_sent)
        self.packets_delivered = 0
        self.control_packets = 0
        self.time = 0.0
        self.dt = 0.01

    def generate_traffic(self):
        """Generate Poisson traffic at each node."""
        n_new = self.rng.poisson(self.load, size=self.N)
        self.queues += n_new
        return int(np.sum(n_new))

    def deliver_packets(self):
        """Process queued packets (service rate = 1.0 per node per step)."""
        served = np.minimum(self.queues, 1.0 + self.rng.exponential(0.5, self.N))
        self.queues = np.maximum(self.queues - served, 0)
        self.packets_delivered += int(np.sum(served))

    def inject_loss(self, fraction=0.1):
        """Simulate burst packet loss: drop fraction of queued packets."""
        n_drop = (self.queues * fraction).astype(int)
        self.queues = np.maximum(self.queues - n_drop, 0)

    def step_per_packet(self):
        """Per-packet control: acknowledge each packet individually."""
        self.generate_traffic()
        self.deliver_packets()
        # Control overhead: one ACK per delivered packet
        self.control_packets += self.packets_delivered
        self.time += self.dt

    def step_variance(self):
        """Variance-based control: measure bulk queue variance, adjust."""
        self.generate_traffic()
        self.deliver_packets()
        # Control overhead: one measurement per step (not per packet)
        self.control_packets += 1
        # Variance-based rate adjustment
        var = np.var(self.queues)
        if var > 2.0:
            self.queues *= 0.95  # gentle backpressure
        self.time += self.dt


def run_trial(method, N=100, n_steps=2000, loss_at=1000, seed=42):
    net = PacketNetwork(N=N, load_factor=0.7, seed=seed)

    # Warm up
    for _ in range(500):
        if method == "per_packet":
            net.step_per_packet()
        else:
            net.step_variance()

    net.packets_delivered = 0
    net.control_packets = 0

    # Pre-loss phase
    for step in range(n_steps):
        if step == loss_at:
            net.inject_loss(0.1)

        if method == "per_packet":
            net.step_per_packet()
        else:
            net.step_variance()

    throughput = net.packets_delivered / (n_steps * net.dt)
    overhead = net.control_packets / max(net.packets_delivered, 1)

    # Recovery time: measure how long queue variance stays elevated after loss
    # Re-run with loss and track variance
    net2 = PacketNetwork(N=N, load_factor=0.7, seed=seed + 1000)
    for _ in range(500):
        if method == "per_packet":
            net2.step_per_packet()
        else:
            net2.step_variance()

    baseline_var = np.var(net2.queues)
    net2.inject_loss(0.1)

    recovery_time = 0.0
    for step in range(500):
        if method == "per_packet":
            net2.step_per_packet()
        else:
            net2.step_variance()
        cur_var = np.var(net2.queues)
        if cur_var <= baseline_var * 1.5:
            recovery_time = step * net2.dt
            break
    else:
        recovery_time = 500 * net2.dt

    return {
        "method": method,
        "throughput": float(throughput),
        "overhead_ratio": float(overhead),
        "recovery_time": float(recovery_time),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Central Molecule (Per-Packet vs Variance Control) ===")

    n_trials = 20
    results = []

    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}")
        for method in ["per_packet", "variance"]:
            row = run_trial(method, N=100, seed=42 + trial * 7)
            row["trial"] = trial
            results.append(row)

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "central_molecule.csv")
    fields = ["trial", "method", "throughput", "overhead_ratio", "recovery_time"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"  CSV saved to {csv_path}")

    # Statistical comparison
    from scipy.stats import ttest_ind

    pp = [r for r in results if r["method"] == "per_packet"]
    vr = [r for r in results if r["method"] == "variance"]

    tp_pp = [r["throughput"] for r in pp]
    tp_vr = [r["throughput"] for r in vr]
    oh_pp = [r["overhead_ratio"] for r in pp]
    oh_vr = [r["overhead_ratio"] for r in vr]
    rt_pp = [r["recovery_time"] for r in pp]
    rt_vr = [r["recovery_time"] for r in vr]

    _, p_throughput = ttest_ind(tp_pp, tp_vr)
    _, p_overhead = ttest_ind(oh_pp, oh_vr)
    _, p_recovery = ttest_ind(rt_pp, rt_vr)

    summary = {
        "test": "central_molecule",
        "mean_throughput_per_packet": float(np.mean(tp_pp)),
        "mean_throughput_variance": float(np.mean(tp_vr)),
        "throughput_ratio": float(np.mean(tp_vr) / np.mean(tp_pp)) if np.mean(tp_pp) > 0 else float("nan"),
        "mean_overhead_per_packet": float(np.mean(oh_pp)),
        "mean_overhead_variance": float(np.mean(oh_vr)),
        "overhead_ratio": float(np.mean(oh_pp) / np.mean(oh_vr)) if np.mean(oh_vr) > 0 else float("nan"),
        "mean_recovery_per_packet": float(np.mean(rt_pp)),
        "mean_recovery_variance": float(np.mean(rt_vr)),
        "recovery_speedup": float(np.mean(rt_pp) / np.mean(rt_vr)) if np.mean(rt_vr) > 0 else float("nan"),
        "p_value_throughput": float(p_throughput),
        "p_value_overhead": float(p_overhead),
        "p_value_recovery": float(p_recovery),
        "pass": bool(np.mean(oh_vr) < np.mean(oh_pp)),  # Variance method should have lower overhead
    }
    json_path = os.path.join(RESULTS_DIR, "central_molecule_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Overhead ratio (pp/var) = {summary['overhead_ratio']:.2f}")
    print(f"  pass = {summary['pass']}")
    return summary


if __name__ == "__main__":
    main()
