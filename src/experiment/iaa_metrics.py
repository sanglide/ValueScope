"""
IAA 指标计算模块
所有函数为纯函数，输入基本数据类型，输出数值，支持独立单元测试
"""

from typing import Optional, Callable


# ============================================================
# 维度1: 二分类指标 (风险检测)
# ============================================================

def percent_agreement(labels_a: list, labels_b: list) -> float:
    """计算简单一致率

    Args:
        labels_a: 标注者A的标签列表 (bool)
        labels_b: 标注者B的标签列表 (bool)

    Returns:
        一致比例 [0, 1]
    """
    if not labels_a or not labels_b:
        return 0.0
    n = min(len(labels_a), len(labels_b))
    agree = sum(1 for i in range(n) if labels_a[i] == labels_b[i])
    return agree / n


def cohen_kappa_binary(labels_a: list, labels_b: list) -> float:
    """计算二分类 Cohen's Kappa

    Args:
        labels_a: 标注者A的标签列表 (bool)
        labels_b: 标注者B的标签列表 (bool)

    Returns:
        Cohen's Kappa [-1, 1]
    """
    if not labels_a or not labels_b:
        return 0.0
    n = min(len(labels_a), len(labels_b))
    if n == 0:
        return 0.0

    # 构建混淆矩阵
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

    # 观察一致性
    po = (n11 + n00) / n

    # 期望一致性
    p_a_pos = (n11 + n10) / n
    p_b_pos = (n11 + n01) / n
    pe = p_a_pos * p_b_pos + (1 - p_a_pos) * (1 - p_b_pos)

    if pe == 1.0:
        return 1.0

    return (po - pe) / (1 - pe)


def pabak(labels_a: list, labels_b: list) -> float:
    """Prevalence-Adjusted Bias-Adjusted Kappa (PABAK)

    修正 Cohen's κ 的 prevalence 和 bias 效应。
    当一方标注极度不平衡（如全标正类）导致 κ=0 时，
    PABAK 能反映真实的一致程度。

    公式: PABAK = 2 * po - 1

    Args:
        labels_a: 标注者A的标签列表 (bool)
        labels_b: 标注者B的标签列表 (bool)

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
    """Gwet's AC1 — 对 prevalence 效应鲁棒的一致性系数

    当 prevalence 极端时 Cohen's κ 不稳定，AC1 使用
    基于均匀分布的期望一致性来替代边际分布的 pe。

    公式:
      po = 观察一致率
      pe = 2 * π * (1 - π)，其中 π = (p_a_pos + p_b_pos) / 2
      AC1 = (po - pe) / (1 - pe)

    Args:
        labels_a: 标注者A的标签列表 (bool)
        labels_b: 标注者B的标签列表 (bool)

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

    # 估计正类的边际比例 π
    a_pos = sum(1 for i in range(n) if labels_a[i])
    b_pos = sum(1 for i in range(n) if labels_b[i])
    pi = (a_pos + b_pos) / (2 * n)

    # Gwet 的期望一致性
    pe = 2 * pi * (1 - pi)

    if pe == 1.0:
        return 1.0

    return (po - pe) / (1 - pe)


def fleiss_kappa(matrix: list) -> float:
    """计算 Fleiss' Kappa (多标注者一致性)

    Args:
        matrix: N x K 矩阵, matrix[i][j] = 样本i被标为类别j的标注者数
                每行之和 = 参与标注的标注者数 n

    Returns:
        Fleiss' Kappa [-1, 1]
    """
    if not matrix:
        return 0.0

    N = len(matrix)  # 样本数
    k = len(matrix[0])  # 类别数

    if N == 0 or k == 0:
        return 0.0

    n = sum(matrix[0])  # 每行标注者数
    if n <= 1:
        return 0.0

    # 各类别的边际比例 p_j
    p_j = []
    for j in range(k):
        col_sum = sum(matrix[i][j] for i in range(N))
        p_j.append(col_sum / (N * n))

    # 每个样本的一致性 P_i
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
    """计算 Krippendorff's Alpha (名义尺度)

    Args:
        reliability_data: M x N 矩阵 (M=标注者, N=样本)
                          元素为 int 或 None(缺失)

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

    # 收集每个 item 的非缺失值
    # 同时计算观察不一致性和期望不一致性
    total_pairs = 0
    disagreement_pairs = 0
    value_counts = {}  # 全局值频率
    total_values = 0

    for k in range(n_items):
        # 收集 item k 的所有非 None 值
        values = []
        for m in range(n_annotators):
            v = reliability_data[m][k]
            if v is not None:
                values.append(v)

        m_k = len(values)
        if m_k < 2:
            continue

        # 观察不一致性: item k 内的所有配对
        for i in range(m_k):
            for j in range(i + 1, m_k):
                total_pairs += 1
                if values[i] != values[j]:
                    disagreement_pairs += 1

        # 累积全局值频率
        for v in values:
            value_counts[v] = value_counts.get(v, 0) + 1
            total_values += 1

    if total_pairs == 0:
        return 0.0

    D_o = disagreement_pairs / total_pairs

    # 期望不一致性: 从边际分布中随机抽两个不同的概率
    # D_e = 1 - sum(p_c^2)，其中 p_c = count_c / total
    D_e = 1.0 - sum(
        (count / total_values) ** 2 for count in value_counts.values()
    )

    if D_e == 0.0:
        # 所有值相同
        return 1.0

    return 1.0 - D_o / D_e


# ============================================================
# 维度2: 多标签指标 (价值ID识别)
# ============================================================

def pairwise_jaccard(sets_a: list, sets_b: list) -> float:
    """计算逐样本 Jaccard 相似度的均值

    Args:
        sets_a: 标注者A的集合列表
        sets_b: 标注者B的集合列表

    Returns:
        平均 Jaccard [0, 1]
    """
    if not sets_a or not sets_b:
        return 0.0

    n = min(len(sets_a), len(sets_b))
    scores = []
    for i in range(n):
        a, b = sets_a[i], sets_b[i]
        if not a and not b:
            # 双空 = 完全一致（都认为无价值）
            scores.append(1.0)
        elif not a or not b:
            scores.append(0.0)
        else:
            intersection = len(a & b)
            union = len(a | b)
            scores.append(intersection / union if union > 0 else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


def pairwise_symmetric_f1(sets_a: list, sets_b: list) -> float:
    """计算逐样本对称 F1 的均值: 2|A∩B| / (|A|+|B|)

    Args:
        sets_a: 标注者A的集合列表
        sets_b: 标注者B的集合列表

    Returns:
        平均 Symmetric F1 [0, 1]
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


def recall_of_gt(gt_sets: list, pred_sets: list) -> float:
    """Recall-of-GT: 预测集对 GT 标注的召回率（方案一核心指标）

    衡量预测是否「覆盖」了 GT 标注的所有 value，不惩罚预测额外的 value。
    这对 GT 标注稀疏（只标最显著 value）的数据集更公平。

    公式: mean_i( |pred_i ∩ gt_i| / |gt_i| )，仅对 gt 非空样本计算。
    双空样本（GT 空且 pred 空）计为 1.0（正确地预测无 value）。
    GT 空但 pred 非空样本计为 0.0（惩罚 false-alarm values）。

    Args:
        gt_sets:   ground-truth 价值集合列表
        pred_sets: 预测价值集合列表

    Returns:
        平均 Recall-of-GT [0, 1]
    """
    if not gt_sets or not pred_sets:
        return 0.0

    n = min(len(gt_sets), len(pred_sets))
    scores = []
    for i in range(n):
        gt, pred = gt_sets[i], pred_sets[i]
        if not gt and not pred:
            scores.append(1.0)          # 双空：正确地都无标注
        elif not gt and pred:
            scores.append(0.0)          # GT 空但乱报：false alarm
        elif gt and not pred:
            scores.append(0.0)          # GT 有但漏报：miss
        else:
            scores.append(len(gt & pred) / len(gt))  # 覆盖率

    return sum(scores) / len(scores) if scores else 0.0


def precision_of_gt(gt_sets: list, pred_sets: list) -> float:
    """Precision-of-GT: 预测集中有多少被 GT 认可（方案一补充指标）

    衡量预测的 value 有多少是 GT 标注中存在的，
    即预测集相对于 GT 的精确率。

    公式: mean_i( |pred_i ∩ gt_i| / |pred_i| )，仅对 pred 非空样本计算。
    双空样本计为 1.0，pred 非空但 GT 空计为 0.0。

    Args:
        gt_sets:   ground-truth 价值集合列表
        pred_sets: 预测价值集合列表

    Returns:
        平均 Precision-of-GT [0, 1]
    """
    if not gt_sets or not pred_sets:
        return 0.0

    n = min(len(gt_sets), len(pred_sets))
    scores = []
    for i in range(n):
        gt, pred = gt_sets[i], pred_sets[i]
        if not gt and not pred:
            scores.append(1.0)
        elif not pred:
            scores.append(1.0)          # pred 空：无 false positive
        elif not gt and pred:
            scores.append(0.0)          # GT 空但预测非空：全错
        else:
            scores.append(len(gt & pred) / len(pred))

    return sum(scores) / len(scores) if scores else 0.0


def partial_credit_jaccard(gt_sets: list, pred_sets: list) -> float:
    """Partial-credit Jaccard: 对大集合预测更公平的 Jaccard 变体（方案一补充）

    标准 Jaccard = |A∩B| / |A∪B|，当预测集远大于 GT 时严重惩罚召回。
    本指标使用 max(|pred|, |gt|) 作为分母，相当于「取较大集合做参照」，
    比标准 Jaccard 对稀疏 GT 更宽容。

    公式: mean_i( |pred_i ∩ gt_i| / max(|pred_i|, |gt_i|) )

    Args:
        gt_sets:   ground-truth 价值集合列表
        pred_sets: 预测价值集合列表

    Returns:
        平均 Partial-credit Jaccard [0, 1]
    """
    if not gt_sets or not pred_sets:
        return 0.0

    n = min(len(gt_sets), len(pred_sets))
    scores = []
    for i in range(n):
        gt, pred = gt_sets[i], pred_sets[i]
        if not gt and not pred:
            scores.append(1.0)
        elif not gt or not pred:
            scores.append(0.0)
        else:
            intersection = len(gt & pred)
            denom = max(len(gt), len(pred))
            scores.append(intersection / denom)

    return sum(scores) / len(scores) if scores else 0.0


def per_value_fleiss_kappa(
    build_matrix_fn: Callable[[str], list],
    value_ids: list,
) -> dict:
    """对每个 value_id 分别计算 Fleiss' Kappa

    Args:
        build_matrix_fn: 给定 value_id 返回 N x 2 Fleiss 矩阵的函数
        value_ids: 要计算的 value_id 列表

    Returns:
        {value_id: kappa} 字典
    """
    result = {}
    for vid in value_ids:
        matrix = build_matrix_fn(vid)
        if not matrix:
            result[vid] = 0.0
        else:
            result[vid] = fleiss_kappa(matrix)
    return result


# ============================================================
# 主实验补充指标
# ============================================================

def precision_recall_f1_binary(
    gt_labels: list[bool],
    pred_labels: list[bool],
) -> dict:
    """计算二分类 Precision / Recall / F1 / Accuracy

    Args:
        gt_labels: Ground truth 标签
        pred_labels: 预测标签

    Returns:
        {"precision": float, "recall": float, "f1": float, "accuracy": float,
         "tp": int, "fp": int, "fn": int, "tn": int}
    """
    if not gt_labels or not pred_labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0,
                "tp": 0, "fp": 0, "fn": 0, "tn": 0}

    n = min(len(gt_labels), len(pred_labels))
    tp = fp = fn = tn = 0
    for i in range(n):
        g, p = bool(gt_labels[i]), bool(pred_labels[i])
        if g and p:
            tp += 1
        elif not g and p:
            fp += 1
        elif g and not p:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def micro_precision_recall_f1(
    gt_value_sets: list[set],
    pred_value_sets: list[set],
) -> dict:
    """计算多标签 Micro Precision / Recall / F1

    对所有样本的所有标签维度汇总 TP/FP/FN 后计算 micro-average。

    Args:
        gt_value_sets: Ground truth 价值集合列表
        pred_value_sets: 预测价值集合列表

    Returns:
        {"micro_precision": float, "micro_recall": float, "micro_f1": float}
    """
    if not gt_value_sets or not pred_value_sets:
        return {"micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0}

    n = min(len(gt_value_sets), len(pred_value_sets))
    total_tp = total_fp = total_fn = 0

    for i in range(n):
        gt_set = gt_value_sets[i] if isinstance(gt_value_sets[i], set) else set(gt_value_sets[i])
        pred_set = pred_value_sets[i] if isinstance(pred_value_sets[i], set) else set(pred_value_sets[i])
        total_tp += len(gt_set & pred_set)
        total_fp += len(pred_set - gt_set)
        total_fn += len(gt_set - pred_set)

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
    }
