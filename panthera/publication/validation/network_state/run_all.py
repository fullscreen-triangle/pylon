"""
Master runner: executes all validation tests and produces a unified summary.

Usage:
    python run_all.py
"""

import os
import sys
import json
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Test modules in order
TEST_MODULES = [
    ("test_ideal_gas_law",        "Ideal Gas Law (PV=NkT)"),
    ("test_maxwell_boltzmann",    "Maxwell-Boltzmann Distribution"),
    ("test_phase_transitions",    "Phase Transitions"),
    ("test_van_der_waals",        "Van der Waals EoS"),
    ("test_transport_coefficients", "Transport Coefficients"),
    ("test_variance_restoration", "Variance Restoration"),
    ("test_uncertainty_relations", "Uncertainty Relations"),
    ("test_thermodynamic_potentials", "Thermodynamic Potentials"),
    ("test_central_molecule",     "Central Molecule Impossibility"),
    ("test_phonon_dispersion",    "Phonon Dispersion"),
    ("test_heat_capacity",        "Heat Capacity"),
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 60)
    print("  Network State Validation Suite")
    print("=" * 60)

    all_summaries = []
    total = len(TEST_MODULES)
    passed = 0
    failed = 0
    errored = 0

    t_start_all = time.time()

    for idx, (mod_name, description) in enumerate(TEST_MODULES):
        print(f"\n[{idx+1}/{total}] {description}")
        print("-" * 50)
        t_start = time.time()

        try:
            mod = __import__(mod_name)
            summary = mod.main()
            elapsed = time.time() - t_start

            summary["_elapsed_seconds"] = round(elapsed, 2)
            summary["_description"] = description
            all_summaries.append(summary)

            if summary.get("pass", False):
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            print(f"  --> {status} ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t_start
            errored += 1
            err_summary = {
                "test": mod_name,
                "_description": description,
                "_elapsed_seconds": round(elapsed, 2),
                "pass": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            all_summaries.append(err_summary)
            print(f"  --> ERROR ({elapsed:.1f}s): {e}")

    total_elapsed = time.time() - t_start_all

    # Master summary
    master = {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "validation_percentage": round(100.0 * passed / total, 1) if total > 0 else 0,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "individual_results": all_summaries,
    }

    json_path = os.path.join(RESULTS_DIR, "master_summary.json")
    with open(json_path, "w") as f:
        json.dump(master, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed  "
          f"({master['validation_percentage']}%)")
    print(f"  Failed: {failed}  |  Errored: {errored}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Master summary: {json_path}")
    print("=" * 60)

    return master


if __name__ == "__main__":
    main()
