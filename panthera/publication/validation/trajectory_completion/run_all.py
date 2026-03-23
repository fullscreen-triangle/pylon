"""
Master runner for all Paper 2 (Trajectory Completion) validation tests.

Imports and runs all test modules, collects summaries, and produces
master_summary.json with overall validation results.
"""

import os
import sys
import json
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# All test modules in order
TEST_MODULES = [
    ("backward_navigation", "test_backward_navigation"),
    ("program_synthesis", "test_program_synthesis"),
    ("s_space_clustering", "test_s_space_clustering"),
    ("thermodynamic_security", "test_thermodynamic_security"),
    ("gauge_invariance", "test_gauge_invariance"),
    ("fiber_bundle", "test_fiber_bundle"),
    ("renormalization_group", "test_renormalization_group"),
    ("topological_protection", "test_topological_protection"),
    ("information_geometry", "test_information_geometry"),
    ("godelian_residue", "test_godelian_residue"),
    ("entropy_computation", "test_entropy_computation"),
    ("operational_trichotomy", "test_operational_trichotomy"),
    ("byzantine_tolerance", "test_byzantine_tolerance"),
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 70)
    print("  Paper 2: Backward Trajectory Completion on Gear Ratio Manifolds")
    print("  Comprehensive Validation Test Suite")
    print("=" * 70)
    print()

    summaries = {}
    total_tests = len(TEST_MODULES)
    passed = 0
    failed = 0
    errors = []

    for i, (test_name, module_name) in enumerate(TEST_MODULES):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{total_tests}] Running: {test_name}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            mod = __import__(module_name)
            result = mod.main()
            elapsed = time.time() - t0

            result["elapsed_seconds"] = round(elapsed, 2)
            summaries[test_name] = result

            if result.get("pass", False):
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            print(f"\n  Result: {status} ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            failed += 1
            error_msg = f"{type(e).__name__}: {e}"
            errors.append((test_name, error_msg))
            summaries[test_name] = {
                "pass": False,
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(elapsed, 2),
            }
            print(f"\n  ERROR: {error_msg} ({elapsed:.1f}s)")
            traceback.print_exc()

    # Master summary
    print(f"\n{'='*70}")
    print("  MASTER SUMMARY")
    print(f"{'='*70}")

    validation_pct = (passed / total_tests * 100) if total_tests > 0 else 0.0

    master = {
        "total_tests": total_tests,
        "passed": passed,
        "failed": failed,
        "overall_validation_percentage": round(validation_pct, 1),
        "test_results": {},
        "errors": [{"test": t, "error": e} for t, e in errors],
    }

    for test_name, summary in summaries.items():
        master["test_results"][test_name] = {
            "pass": summary.get("pass", False),
            "elapsed_seconds": summary.get("elapsed_seconds", 0),
        }
        # Include key metrics
        for key in summary:
            if key not in ("pass", "elapsed_seconds", "error", "traceback", "test"):
                if isinstance(summary[key], (int, float, bool, str)):
                    master["test_results"][test_name][key] = summary[key]

    json_path = os.path.join(RESULTS_DIR, "master_summary.json")
    with open(json_path, "w") as f:
        json.dump(master, f, indent=2)

    print(f"\n  Total:  {total_tests}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Validation: {validation_pct:.1f}%")
    if errors:
        print(f"\n  Errors:")
        for t, e in errors:
            print(f"    - {t}: {e}")
    print(f"\n  Master summary saved to {json_path}")
    print(f"{'='*70}")

    return master


if __name__ == "__main__":
    main()
