#!/usr/bin/env python3
"""
统计检验模块 — 主实验专用

提供：
  - McNemar's test (配对二分类)
  - Wilcoxon signed-rank test (配对连续值)
  - Bootstrap 95% CI
  - 效应量: Cohen's h, rank-biserial correlation
"""

import random
from typing import Optional

# scipy 是可选依赖
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# McNemar's test
# ============================================================

def mcnemar_test(
    labels_a: list[bool],
    labels_b: list[bool],
    exact: bool = True,
) -> dict:
    """McNemar's test for paired binary classifications.

    Tests whether two classifiers have the same proportion of
    disagreements (b vs c in the 2x2 table).

    Args:
        labels_a: Binary labels from classifier A (e.g., ground truth aligned)
        labels_b: Binary labels from classifier B
        exact: Use exact binomial test (recommended for small samples)

    Returns:
        {"statistic": float, "p_value": float, "b": int, "c": int}
        b = A correct & B wrong, c = A wrong & B correct
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must have the same length")

    b = 0  # A=1, B=0
    c = 0  # A=0, B=1
    for a_val, b_val in zip(labels_a, labels_b):
        if a_val and not b_val:
            b += 1
        elif not a_val and b_val:
            c += 1

    n_disagreements = b + c
    if n_disagreements == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": 0, "c": 0}

    if exact or n_disagreements < 25:
        # Exact binomial test: p = 2 * min(P(X <= min(b,c)), P(X >= min(b,c)))
        # Under H0: b ~ Binomial(n, 0.5)
        if HAS_SCIPY:
            result = scipy_stats.binomtest(min(b, c), n=n_disagreements, p=0.5)
            # two-sided: binomtest returns one-sided by default
            p_value = min(2 * result.pvalue, 1.0)
        else:
            # Manual exact computation using binomial CDF
            p_value = _binom_two_sided(min(b, c), n_disagreements, 0.5)
        statistic = float(min(b, c))
    else:
        # Continuity-corrected chi-squared approximation
        statistic = (abs(b - c) - 1.0) ** 2 / n_disagreements
        if HAS_SCIPY:
            p_value = 1.0 - scipy_stats.chi2.cdf(statistic, df=1)
        else:
            # Approximate using normal
            from math import erf, sqrt
            z = sqrt(statistic)
            p_value = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))

    return {"statistic": statistic, "p_value": p_value, "b": b, "c": c}


def _binom_two_sided(k: int, n: int, p: float) -> float:
    """Two-sided exact binomial p-value (pure Python fallback)."""
    from math import comb, floor

    # Compute P(X <= k) + P(X >= n-k) for two-sided
    prob = 0.0
    # Find the critical value: sum probabilities <= observed
    observed_prob = sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
    # Two-sided: double the smaller tail
    p_value = 2 * min(observed_prob, 1.0 - observed_prob + sum(
        comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)
    ))
    return min(p_value, 1.0)


# ============================================================
# Wilcoxon signed-rank test
# ============================================================

def wilcoxon_signed_rank_test(
    values_a: list[float],
    values_b: list[float],
) -> dict:
    """Wilcoxon signed-rank test for paired continuous values.

    Tests whether the median difference between paired observations is zero.

    Args:
        values_a: Values from method A (e.g., per-sample Jaccard for baseline)
        values_b: Values from method B (e.g., per-sample Jaccard for pipeline)

    Returns:
        {"statistic": float, "p_value": float, "n_pairs": int}
    """
    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must have the same length")

    diffs = [a - b for a, b in zip(values_a, values_b)]
    # Remove zero differences
    diffs = [d for d in diffs if abs(d) > 1e-10]
    n_pairs = len(diffs)

    if n_pairs == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": 0}

    if HAS_SCIPY:
        result = scipy_stats.wilcoxon(values_a, values_b, alternative="two-sided")
        return {
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "n_pairs": n_pairs,
        }
    else:
        # Manual Wilcoxon signed-rank
        abs_diffs = sorted(enumerate(diffs), key=lambda x: abs(x[1]))
        ranks = []
        i = 0
        while i < len(abs_diffs):
            j = i
            while j < len(abs_diffs) - 1 and abs(abs_diffs[j + 1][1]) == abs(abs_diffs[i][1]):
                j += 1
            avg_rank = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                ranks.append((abs_diffs[k][0], avg_rank, abs_diffs[k][1]))
            i = j + 1

        w_plus = sum(r for _, r, d in ranks if d > 0)
        w_minus = sum(r for _, r, d in ranks if d < 0)
        statistic = min(w_plus, w_minus)

        # Approximate p-value using normal approximation
        n = len(ranks)
        mean_w = n * (n + 1) / 4.0
        std_w = (n * (n + 1) * (2 * n + 1) / 24.0) ** 0.5
        if std_w > 0:
            z = (statistic - mean_w) / std_w
            from math import erf, sqrt
            p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        else:
            p_value = 1.0

        return {"statistic": float(statistic), "p_value": p_value, "n_pairs": n_pairs}


# ============================================================
# Bootstrap Confidence Interval
# ============================================================

def bootstrap_ci(
    values: list[float],
    statistic_fn=None,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for a statistic.

    Args:
        values: Sample values
        statistic_fn: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap iterations
        ci_level: Confidence level (default: 0.95)
        seed: Random seed

    Returns:
        {"estimate": float, "ci_lower": float, "ci_upper": float, "ci_level": float}
    """
    if statistic_fn is None:
        statistic_fn = _mean

    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "ci_level": ci_level}

    estimate = statistic_fn(values)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boot_stats.append(statistic_fn(sample))

    boot_stats.sort()
    alpha = 1.0 - ci_level
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))
    upper_idx = min(upper_idx, n_bootstrap - 1)

    return {
        "estimate": round(estimate, 6),
        "ci_lower": round(boot_stats[lower_idx], 6),
        "ci_upper": round(boot_stats[upper_idx], 6),
        "ci_level": ci_level,
    }


# ============================================================
# Effect Sizes
# ============================================================

def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for comparing two proportions.

    Args:
        p1: Proportion 1 (e.g., accuracy of method A)
        p2: Proportion 2 (e.g., accuracy of method B)

    Returns:
        Cohen's h (0.2 = small, 0.5 = medium, 0.8 = large)
    """
    from math import asin, sqrt
    return 2 * asin(sqrt(p1)) - 2 * asin(sqrt(p2))


def rank_biserial_correlation(
    values_a: list[float],
    values_b: list[float],
) -> float:
    """Rank-biserial correlation (effect size for Wilcoxon signed-rank test).

    r = Z / sqrt(N), where Z is the standardized test statistic.
    If scipy is not available, computed from the W statistic.

    Returns:
        Rank-biserial correlation r (0.1 = small, 0.3 = medium, 0.5 = large)
    """
    from math import sqrt
    n = len(values_a)
    if n == 0:
        return 0.0

    result = wilcoxon_signed_rank_test(values_a, values_b)
    w = result["statistic"]
    n_pairs = result["n_pairs"]
    if n_pairs == 0:
        return 0.0

    # r = 1 - 2W / (n_pairs * (n_pairs + 1) / 2)
    total_rank_sum = n_pairs * (n_pairs + 1) / 2.0
    if total_rank_sum == 0:
        return 0.0
    r = 1 - (2 * w) / total_rank_sum
    return r


# ============================================================
# Unified comparison function
# ============================================================

def compare_methods(
    gt_risks: list[bool],
    method_a_risks: list[bool],
    method_b_risks: list[bool],
    method_a_jaccards: list[float],
    method_b_jaccards: list[float],
    method_a_name: str = "A",
    method_b_name: str = "B",
) -> dict:
    """Unified statistical comparison between two methods.

    Args:
        gt_risks: Ground truth risk labels
        method_a_risks: Method A predicted risk labels
        method_b_risks: Method B predicted risk labels
        method_a_jaccards: Per-sample Jaccard scores for method A
        method_b_jaccards: Per-sample Jaccard scores for method B

    Returns:
        Complete comparison results dict
    """
    # Dim1: McNemar (compare which method agrees more with GT)
    # Convert to: "A correct" vs "B correct" comparison
    a_correct = [a == g for a, g in zip(method_a_risks, gt_risks)]
    b_correct = [b == g for b, g in zip(method_b_risks, gt_risks)]

    mcnemar = mcnemar_test(a_correct, b_correct)

    # Dim2: Wilcoxon on Jaccard scores
    wilcoxon = wilcoxon_signed_rank_test(method_a_jaccards, method_b_jaccards)

    # Effect sizes
    a_accuracy = _mean(a_correct) if a_correct else 0.0
    b_accuracy = _mean(b_correct) if b_correct else 0.0
    h = cohens_h(b_accuracy, a_accuracy)  # positive = B better
    r = rank_biserial_correlation(method_a_jaccards, method_b_jaccards)

    return {
        "method_a": method_a_name,
        "method_b": method_b_name,
        "dim1_mcnemar": mcnemar,
        "dim2_wilcoxon": wilcoxon,
        "effect_size_cohens_h": round(h, 4),
        "effect_size_rank_biserial": round(r, 4),
        "a_accuracy": round(a_accuracy, 4),
        "b_accuracy": round(b_accuracy, 4),
        "a_mean_jaccard": round(_mean(method_a_jaccards), 4) if method_a_jaccards else 0.0,
        "b_mean_jaccard": round(_mean(method_b_jaccards), 4) if method_b_jaccards else 0.0,
    }


# ============================================================
# Helpers
# ============================================================

def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
