"""
Test 4: S-space clustering structure.

Validates that programs cluster by functional type in S-entropy space
using k-means, silhouette scores, adjusted Rand index, and NMI.
"""

import os
import sys
import json
import csv
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.vq import kmeans2

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    get_program_library, get_all_program_names, get_category_for_program,
    generate_test_inputs, extract_s_coordinates, EPSILON, PROGRAM_CATEGORIES
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def adjusted_rand_index(labels_true, labels_pred):
    """Compute adjusted Rand index between two clusterings."""
    n = len(labels_true)
    # Contingency table
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    contingency = np.zeros((len(classes), len(clusters)), dtype=int)
    class_map = {c: i for i, c in enumerate(classes)}
    clust_map = {c: i for i, c in enumerate(clusters)}
    for i in range(n):
        contingency[class_map[labels_true[i]], clust_map[labels_pred[i]]] += 1

    # Row and column sums
    a = contingency.sum(axis=1)
    b = contingency.sum(axis=0)

    # Combinations
    def comb2(x):
        return x * (x - 1) / 2

    sum_comb_nij = sum(comb2(contingency[i, j])
                       for i in range(len(classes))
                       for j in range(len(clusters)))
    sum_comb_a = sum(comb2(ai) for ai in a)
    sum_comb_b = sum(comb2(bj) for bj in b)
    comb_n = comb2(n)

    if comb_n == 0:
        return 0.0
    expected = sum_comb_a * sum_comb_b / comb_n
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected
    if abs(denom) < EPSILON:
        return 1.0 if abs(sum_comb_nij - expected) < EPSILON else 0.0
    return float((sum_comb_nij - expected) / denom)


def normalized_mutual_info(labels_true, labels_pred):
    """Compute normalized mutual information."""
    n = len(labels_true)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Contingency
    contingency = np.zeros((len(classes), len(clusters)), dtype=int)
    class_map = {c: i for i, c in enumerate(classes)}
    clust_map = {c: i for i, c in enumerate(clusters)}
    for i in range(n):
        contingency[class_map[labels_true[i]], clust_map[labels_pred[i]]] += 1

    # Marginals
    pi = contingency.sum(axis=1) / n
    pj = contingency.sum(axis=0) / n

    # Entropies
    H_true = -np.sum(pi[pi > 0] * np.log(pi[pi > 0]))
    H_pred = -np.sum(pj[pj > 0] * np.log(pj[pj > 0]))

    # MI
    mi = 0.0
    for i in range(len(classes)):
        for j in range(len(clusters)):
            if contingency[i, j] > 0:
                pij = contingency[i, j] / n
                mi += pij * np.log(pij / (pi[i] * pj[j] + EPSILON) + EPSILON)

    denom = np.sqrt(H_true * H_pred)
    if denom < EPSILON:
        return 0.0
    return float(mi / denom)


def silhouette_score(coords, labels):
    """Compute mean silhouette score."""
    dist_matrix = squareform(pdist(coords))
    unique_labels = np.unique(labels)
    n = len(labels)
    silhouettes = np.zeros(n)

    for i in range(n):
        same_mask = labels == labels[i]
        same_mask[i] = False
        if np.sum(same_mask) == 0:
            silhouettes[i] = 0.0
            continue
        a_i = np.mean(dist_matrix[i, same_mask])
        b_i = float('inf')
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            other_mask = labels == lbl
            if np.sum(other_mask) > 0:
                b_i = min(b_i, np.mean(dist_matrix[i, other_mask]))
        if b_i == float('inf'):
            b_i = 0.0
        denom = max(a_i, b_i)
        silhouettes[i] = (b_i - a_i) / (denom + EPSILON)

    return float(np.mean(silhouettes))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Test: S-Space Clustering Structure ===")

    rng = np.random.default_rng(42)
    lib = get_program_library()
    all_names = get_all_program_names()

    # Compute S-coordinates
    print("  Computing S-coordinates for all programs ...")
    names = []
    coords = []
    categories = []

    for name in all_names:
        func, cat = lib[name]
        inputs = generate_test_inputs(name, rng=np.random.default_rng(42), n_examples=5)
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
        names.append(name)
        coords.append(coord)
        categories.append(cat)

    coords = np.array(coords, dtype=np.float64)
    cat_array = np.array(categories)
    unique_cats = list(PROGRAM_CATEGORIES.keys())
    cat_labels = np.array([unique_cats.index(c) for c in categories])

    # Pairwise distances
    print("  Computing pairwise distances ...")
    dist_matrix = squareform(pdist(coords))
    n = len(names)

    intra_dists = []
    inter_dists = []
    pair_results = []

    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j]
            same = categories[i] == categories[j]
            if same:
                intra_dists.append(d)
            else:
                inter_dists.append(d)
            pair_results.append({
                "program_i": names[i],
                "program_j": names[j],
                "category_i": categories[i],
                "category_j": categories[j],
                "distance": float(d),
                "same_category": same,
            })

    mean_intra = float(np.mean(intra_dists)) if intra_dists else 0.0
    mean_inter = float(np.mean(inter_dists)) if inter_dists else 0.0
    separation_ratio = mean_inter / (mean_intra + EPSILON)

    # K-means clustering
    print("  Running k-means clustering (k=7) ...")
    # Add small jitter to avoid identical points
    jittered = coords + rng.normal(0, 1e-6, size=coords.shape)
    centroids, kmeans_labels = kmeans2(jittered, k=7, minit='points', seed=42)

    ari = adjusted_rand_index(cat_labels, kmeans_labels)
    nmi = normalized_mutual_info(cat_labels, kmeans_labels)
    sil = silhouette_score(coords, cat_labels)

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "s_space_clustering.csv")
    fields = list(pair_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pair_results)
    print(f"  CSV saved to {csv_path}")

    summary = {
        "test": "s_space_clustering",
        "mean_intra": mean_intra,
        "mean_inter": mean_inter,
        "separation_ratio": separation_ratio,
        "silhouette_score": sil,
        "adjusted_rand_index": ari,
        "NMI": nmi,
        "num_programs": n,
        "num_categories": len(unique_cats),
        "pass": separation_ratio > 1.0,
    }

    json_path = os.path.join(RESULTS_DIR, "s_space_clustering_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")
    print(f"  Separation ratio={separation_ratio:.2f}, "
          f"silhouette={sil:.3f}, ARI={ari:.3f}, NMI={nmi:.3f}")
    return summary


if __name__ == "__main__":
    main()
