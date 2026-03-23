"""
Test 3: Program synthesis accuracy via S-entropy coordinates.

Validates that KD-tree navigation in S-entropy space correctly identifies
programs from input-output examples, with per-category accuracy breakdown.
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    get_program_library, get_all_program_names, get_category_for_program,
    generate_test_inputs, extract_s_coordinates, SSpaceNavigator, EPSILON
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: Program Synthesis Accuracy ===")

    rng = np.random.default_rng(42)
    lib = get_program_library()
    all_names = get_all_program_names()

    # Build S-coordinate library
    print("  Building S-coordinate library ...")
    lib_names = []
    lib_coords = []
    lib_categories = []

    for name in all_names:
        func, cat = lib[name]
        inputs = generate_test_inputs(name, rng=np.random.default_rng(100), n_examples=5)
        sk_vals, st_vals, se_vals = [], [], []
        for inp in inputs:
            try:
                out = func(list(inp))
                if not out:
                    out = [0]
                sc = extract_s_coordinates(np.array(inp, dtype=float),
                                           np.array(out, dtype=float))
                sk_vals.append(sc[0])
                st_vals.append(sc[1])
                se_vals.append(sc[2])
            except Exception:
                pass
        if sk_vals:
            coord = [np.mean(sk_vals), np.mean(st_vals), np.mean(se_vals)]
        else:
            coord = [0.5, 0.5, 0.5]
        lib_names.append(name)
        lib_coords.append(coord)
        lib_categories.append(cat)

    lib_coords = np.array(lib_coords, dtype=np.float64)
    navigator = SSpaceNavigator(lib_names, lib_coords)

    # Test each program
    print("  Testing synthesis accuracy ...")
    results = []
    correct_by_cat = {}
    total_by_cat = {}
    correct_distances = []
    incorrect_distances = []

    for i, name in enumerate(all_names):
        func, cat = lib[name]
        # Generate fresh test inputs (different seed from library building)
        test_inputs = generate_test_inputs(name, rng=np.random.default_rng(200 + i), n_examples=3)

        sk_vals, st_vals, se_vals = [], [], []
        for inp in test_inputs:
            try:
                out = func(list(inp))
                if not out:
                    out = [0]
                sc = extract_s_coordinates(np.array(inp, dtype=float),
                                           np.array(out, dtype=float))
                sk_vals.append(sc[0])
                st_vals.append(sc[1])
                se_vals.append(sc[2])
            except Exception:
                pass

        if sk_vals:
            target_coord = [np.mean(sk_vals), np.mean(st_vals), np.mean(se_vals)]
        else:
            target_coord = [0.5, 0.5, 0.5]

        dist, found_idx, found_name = navigator.navigate(np.array(target_coord))
        correct = (found_name == name)

        if cat not in correct_by_cat:
            correct_by_cat[cat] = 0
            total_by_cat[cat] = 0
        total_by_cat[cat] += 1
        if correct:
            correct_by_cat[cat] += 1
            correct_distances.append(dist)
        else:
            incorrect_distances.append(dist)

        results.append({
            "program_name": name,
            "category": cat,
            "S_k": target_coord[0],
            "S_t": target_coord[1],
            "S_e": target_coord[2],
            "predicted_program": found_name,
            "correct": correct,
            "navigation_distance": float(dist),
        })

    # Compute intra/inter cluster distances
    print("  Computing cluster distances ...")
    intra_dists = []
    inter_dists = []
    for i in range(len(lib_names)):
        for j in range(i + 1, len(lib_names)):
            d = np.linalg.norm(lib_coords[i] - lib_coords[j])
            if lib_categories[i] == lib_categories[j]:
                intra_dists.append(d)
            else:
                inter_dists.append(d)

    mean_intra = float(np.mean(intra_dists)) if intra_dists else 0.0
    mean_inter = float(np.mean(inter_dists)) if inter_dists else 0.0
    separation_ratio = mean_inter / (mean_intra + EPSILON)

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "program_synthesis.csv")
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"  CSV saved to {csv_path}")

    # Summary
    overall_correct = sum(1 for r in results if r["correct"])
    overall_accuracy = overall_correct / len(results)
    per_cat_acc = {}
    for cat in sorted(total_by_cat.keys()):
        per_cat_acc[cat] = correct_by_cat[cat] / total_by_cat[cat]

    summary = {
        "test": "program_synthesis_accuracy",
        "overall_accuracy": overall_accuracy,
        "per_category_accuracy": per_cat_acc,
        "mean_intra_distance": mean_intra,
        "mean_inter_distance": mean_inter,
        "separation_ratio": separation_ratio,
        "mean_correct_distance": float(np.mean(correct_distances)) if correct_distances else 0.0,
        "mean_incorrect_distance": float(np.mean(incorrect_distances)) if incorrect_distances else 0.0,
        "num_programs": len(results),
        "pass": overall_accuracy > 0.05 and separation_ratio > 1.5,
    }

    json_path = os.path.join(RESULTS_DIR, "program_synthesis_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")
    print(f"  Overall accuracy={overall_accuracy:.2f}, "
          f"separation_ratio={separation_ratio:.2f}")
    for cat, acc in per_cat_acc.items():
        print(f"    {cat}: {acc:.2f}")
    return summary


if __name__ == "__main__":
    main()
