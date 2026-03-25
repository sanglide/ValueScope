"""
IAA Metrics Computation Module
All functions are pure functions: they take primitive data types as input,
return numeric values, and support independent unit testing.
"""

from typing import Optional, Callable


# ============================================================
# Dimension 1: Binary Classification Metrics (Risk Detection)
# ============================================================

def percent_agreement(labels_a: list, labels_b: list) -> float:
    """Compute simple percent agreement.

    Args:
        labels_a: Label list of annotator A (bool)
        labels_b: Label list of annotator B (bool)

    Returns:
        Agreement proportion [0, 1]
    """
    if not labels_a or not labels_b:
        return 0.0
    n = min(len(labels_a), len(labels_b))
    agree = sum(1 for i in range(n) if labels_a[i] == labels_b[i])
    return agree / n


def cohen_kappa_binary(labels_a: list, labels_b: list) -> float:
    """Compute binary Cohen's Kappa.

    Args:
        labels_a: Label list of annotator A (bool)
        labels_b: Label list of annotator B (bool)

    Returns:
        Cohen's Kappa [-1, 1]
    """
    if not labels_a or not labels_b:
        return 0.0
    n = min(len(labels_a), len(labels_b))
    if n == 0:
        return 0.0

    # Build confusion matrix
    n11 = n10 = n01 = n00 = 0
    for i in range(n):
        a, b = bool(labels_a[i]), bool(labels_b[i])
        if a and b:
            n11 += 1
        elif a and not b:
            n10 += 1
        elif not a and b:
            n01 += 1
        else:
            n00 += 1

    # Observed agreement
    po = (n11 + n00) / n

    # Expected agreement
    p_a_pos = (n11 + n10) / n
    p_b_pos = (n11 + n01) / n
    pe = p_a_pos * p_b_pos + (1 - p_a_pos) * (1 - p_b_pos)

    if pe == 1.0:
        return 1.0

    return (po - pe) / (1 - pe)


def pabak(labels_a: list, labels_b: list) -> float:
    """Prevalence-Adjusted Bias-Adjusted Kappa (PABAK)

    Corrects for the prevalence and bias effects of Cohen's kappa.
    When one annotator's labels are extremely imbalanced (e.g., all positive),
    leading to kappa = 0, PABAK can reflect the true level of agreement.

    Formula: PABAK = 2 * po - 1

    Args:
        labels_a: Label list of annotator A (bool)
        labels_b: Label list of annotator B (bool)

    Returns:
        PABAK [-1, 1]
    """
    if not labels_a or not labels_b:
        return 0.0
    n = min(len(labels_a), len(labels_b))
    if n == 0:
        return 0.0
    agree = sum(1 for i in range(n) if labels_a[i] == labels_b[i])
    po = agree / n
    return 2 * po - 1


def gwet_ac1(labels_a: list, labels_b: list) -> float:
    """Gwet's AC1 -- a robust agreement coefficient against prevalence effects.

    When prevalence is extreme, Cohen's kappa becomes unstable. AC1 uses
    expected agreement based on a uniform distribution instead of the
    marginal-distribution-based pe.

    Formula:
      po = observed agreement
      pe = 2 * pi * (1 - pi), where pi = (p_a_pos + p_b_pos) / 2
      AC1 = (po - pe) / (1 - pe)

    Args:
        labels_a: Label list of annotator A (bool)
        labels_b: Label list of annotator B (bool)

    Returns:
        Gwet's AC1 [-1, 1]
    """
    if not labels_a or not labels_b:
        return 0.0
    n = min(len(labels_a), len(labels_b))
    if n == 0:
        return 0.0

    agree = sum(1 for i in range(n) if labels_a[i] == labels_b[i])
    po = agree / n

    # Estimate the marginal proportion of the positive class pi
    a_pos = sum(1 for i in range(n) if labels_a[i])
    b_pos = sum(1 for i in range(n) if labels_b[i])
    pi = (a_pos + b_pos) / (2 * n)

    # Gwet's expected agreement
    pe = 2 * pi * (1 - pi)

    if pe == 1.0:
        return 1.0

    return (po - pe) / (1 - pe)


def fleiss_kappa(matrix: list) -> float:
    """Compute Fleiss' Kappa (multi-annotator agreement).

    Args:
        matrix: N x K matrix, where matrix[i][j] = number of annotators
                who assigned sample i to category j.
                Each row sums to the number of participating annotators n.

    Returns:
        Fleiss' Kappa [-1, 1]
    """
    if not matrix:
        return 0.0

    N = len(matrix)  # Number of samples
    k = len(matrix[0])  # Number of categories

    if N == 0 or k == 0:
        return 0.0

    n = sum(matrix[0])  # Number of annotators per row
    if n <= 1:
        return 0.0

    # Marginal proportion for each category p_j
    p_j = []
    for j in range(k):
        col_sum = sum(matrix[i][j] for i in range(N))
        p_j.append(col_sum / (N * n))

    # Per-sample agreement P_i
    P_i_list = []
    for i in range(N):
        row_sum_sq = sum(matrix[i][j] ** 2 for j in range(k))
        P_i = (row_sum_sq - n) / (n * (n - 1))
        P_i_list.append(P_i)

    P_bar = sum(P_i_list) / N
    P_e = sum(pj ** 2 for pj in p_j)

    if P_e == 1.0:
        return 1.0

    return (P_bar - P_e) / (1 - P_e)


def krippendorff_alpha_nominal(reliability_data: list) -> float:
    """Compute Krippendorff's Alpha (nominal scale).

    Args:
        reliability_data: M x N matrix (M = annotators, N = samples).
                          Elements are int or None (missing).

    Returns:
        Krippendorff's Alpha [-1, 1]
    """
    if not reliability_data:
        return 0.0

    n_annotators = len(reliability_data)
    if n_annotators < 2:
        return 0.0

    n_items = len(reliability_data[0])
    if n_items == 0:
        return 0.0

    # Collect non-missing values for each item
    # and compute observed disagreement and expected disagreement
    total_pairs = 0
    disagreement_pairs = 0
    value_counts = {}  # Global value frequency
    total_values = 0

    for k in range(n_items):
        # Collect all non-None values for item k
        values = []
        for m in range(n_annotators):
            v = reliability_data[m][k]
            if v is not None:
                values.append(v)

        m_k = len(values)
        if m_k < 2:
            continue

        # Observed disagreement: all pairs within item k
        for i in range(m_k):
            for j in range(i + 1, m_k):
                total_pairs += 1
                if values[i] != values[j]:
                    disagreement_pairs += 1

        # Accumulate global value frequency
        for v in values:
            value_counts[v] = value_counts.get(v, 0) + 1
            total_values += 1

    if total_pairs == 0:
        return 0.0

    D_o = disagreement_pairs / total_pairs

    # Expected disagreement: probability of drawing two different values
    # from the marginal distribution
    # D_e = 1 - sum(p_c^2), where p_c = count_c / total
    D_e = 1.0 - sum(
        (count / total_values) ** 2 for count in value_counts.values()
    )

    if D_e == 0.0:
        # All values are identical
        return 1.0

    return 1.0 - D_o / D_e


# ============================================================
# Dimension 2: Multi-Label Metrics (Value ID Identification)
# ============================================================

def pairwise_jaccard(sets_a: list, sets_b: list) -> float:
    """Compute the mean per-sample Jaccard similarity.

    Args:
        sets_a: List of sets from annotator A
        sets_b: List of sets from annotator B

    Returns:
        Mean Jaccard [0, 1]
    """
    if not sets_a or not sets_b:
        return 0.0

    n = min(len(sets_a), len(sets_b))
    scores = []
    for i in range(n):
        a, b = sets_a[i], sets_b[i]
        if not a and not b:
            # Both empty = full agreement (both consider no value present)
            scores.append(1.0)
        elif not a or not b:
            scores.append(0.0)
        else:
            intersection = len(a & b)
            union = len(a | b)
            scores.append(intersection / union if union > 0 else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


def pairwise_symmetric_f1(sets_a: list, sets_b: list) -> float:
    """Compute the mean per-sample symmetric F1: 2|A∩B| / (|A|+|B|).

    Args:
        sets_a: List of sets from annotator A
        sets_b: List of sets from annotator B

    Returns:
        Mean Symmetric F1 [0, 1]
    """
    if not sets_a or not sets_b:
        return 0.0

    n = min(len(sets_a), len(sets_b))
    scores = []
    for i in range(n):
        a, b = sets_a[i], sets_b[i]
        if not a and not b:
            scores.append(1.0)
        elif not a or not b:
            scores.append(0.0)
        else:
            intersection = len(a & b)
            denom = len(a) + len(b)
            scores.append(2 * intersection / denom if denom > 0 else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


def per_value_fleiss_kappa(
    build_matrix_fn: Callable[[str], list],
    value_ids: list,
) -> dict:
    """Compute Fleiss' Kappa separately for each value_id.

    Args:
        build_matrix_fn: A function that returns an N x 2 Fleiss matrix
                         given a value_id.
        value_ids: List of value_ids to compute.

    Returns:
        {value_id: kappa} dictionary
    """
    result = {}
    for vid in value_ids:
        matrix = build_matrix_fn(vid)
        if not matrix:
            result[vid] = 0.0
        else:
            result[vid] = fleiss_kappa(matrix)
    return result
