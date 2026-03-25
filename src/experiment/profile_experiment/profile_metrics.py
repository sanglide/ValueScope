#!/usr/bin/env python
"""
Profile statistical metrics computation module.
Provides cross-model consistency, stability, distance, and other metrics.
"""

import numpy as np
from typing import Optional


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    a, b = np.array(vec_a, dtype=float), np.array(vec_b, dtype=float)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def spearman_rank_correlation(
    scores_a: list[float], scores_b: list[float]
) -> tuple[float, float]:
    """Spearman rank correlation coefficient, returns (rho, p_value)."""
    from scipy.stats import spearmanr
    rho, p = spearmanr(scores_a, scores_b)
    return float(rho), float(p)


def kendall_w(score_matrix: np.ndarray) -> float:
    """Kendall's W coefficient of concordance.

    Args:
        score_matrix: shape (M, N), scores from M raters on N items.
    Returns:
        W value in [0, 1].
    """
    m, n = score_matrix.shape
    if m < 2 or n < 2:
        return 0.0

    # Rank transformation (by row)
    from scipy.stats import rankdata
    ranks = np.array([rankdata(row) for row in score_matrix])

    # Column rank sums
    rank_sums = ranks.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    ss = np.sum((rank_sums - mean_rank_sum) ** 2)

    w = 12.0 * ss / (m ** 2 * (n ** 3 - n))
    return float(w)


def coefficient_of_variation(values: list[float]) -> float:
    """Coefficient of variation (CV = std / mean)."""
    arr = np.array(values, dtype=float)
    mean = np.mean(arr)
    if mean == 0:
        return 0.0
    return float(np.std(arr, ddof=1) / abs(mean))


def profile_distance(profile_a: dict, profile_b: dict) -> float:
    """Euclidean distance between two profiles (based on all 20 dimensions)."""
    vec_a = _profile_to_vector(profile_a)
    vec_b = _profile_to_vector(profile_b)
    return float(np.linalg.norm(np.array(vec_a) - np.array(vec_b)))


def pairwise_agreement_matrix(
    profiles: dict[str, dict],
    metric: str = "spearman",
) -> dict:
    """Compute the pairwise agreement matrix across all model pairs.

    Args:
        profiles: {model_key: profile_dict}
        metric: "spearman", "cosine", or "distance"
    Returns:
        {"keys": [...], "matrix": [[...], ...]}
    """
    keys = sorted(profiles.keys())
    n = len(keys)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0 if metric != "distance" else 0.0
            elif j > i:
                vi = _profile_to_vector(profiles[keys[i]])
                vj = _profile_to_vector(profiles[keys[j]])
                if metric == "spearman":
                    val, _ = spearman_rank_correlation(vi, vj)
                elif metric == "cosine":
                    val = cosine_similarity(vi, vj)
                else:
                    val = float(np.linalg.norm(np.array(vi) - np.array(vj)))
                matrix[i][j] = val
                matrix[j][i] = val

    return {
        "keys": keys,
        "matrix": matrix.tolist(),
    }


def compute_dimension_stats(
    profiles: list[dict],
) -> dict:
    """Compute per-dimension statistics across multiple profiles.

    Returns:
        {value_id: {"mean": ..., "std": ..., "cv": ..., "min": ..., "max": ...}}
    """
    from .profile_generator import ALL_VALUE_IDS

    stats = {}
    for vid in ALL_VALUE_IDS:
        values = []
        for p in profiles:
            if vid.startswith("HV"):
                values.append(p.get("l2_scores", {}).get(vid, 0.0))
            else:
                values.append(p.get("l3_scores", {}).get(vid, 0.0))
        arr = np.array(values, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        stats[vid] = {
            "mean": mean_val,
            "std": std_val,
            "cv": std_val / abs(mean_val) if mean_val != 0 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "values": values,
        }
    return stats


def compute_overall_consistency(profiles: dict[str, dict]) -> dict:
    """Compute an overall cross-model consistency metrics summary."""
    keys = sorted(profiles.keys())
    if len(keys) < 2:
        return {"kendall_w": 0.0, "avg_spearman": 0.0, "avg_cosine": 0.0}

    # Kendall's W
    vectors = [_profile_to_vector(profiles[k]) for k in keys]
    matrix = np.array(vectors)
    w = kendall_w(matrix)

    # Pairwise averages
    spearman_vals, cosine_vals = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            rho, _ = spearman_rank_correlation(vectors[i], vectors[j])
            cos = cosine_similarity(vectors[i], vectors[j])
            spearman_vals.append(rho)
            cosine_vals.append(cos)

    return {
        "kendall_w": w,
        "avg_spearman": float(np.mean(spearman_vals)),
        "std_spearman": float(np.std(spearman_vals)),
        "avg_cosine": float(np.mean(cosine_vals)),
        "std_cosine": float(np.std(cosine_vals)),
        "n_pairs": len(spearman_vals),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile_to_vector(profile: dict) -> list[float]:
    """Flatten a profile's l2_scores + l3_scores into a 20-dimensional vector."""
    from .profile_generator import L2_IDS, L3_IDS
    l2 = profile.get("l2_scores", {})
    l3 = profile.get("l3_scores", {})
    return [l2.get(v, 0.0) for v in L2_IDS] + [l3.get(v, 0.0) for v in L3_IDS]
