#!/usr/bin/env python
"""
IAA 本地测试
用模拟数据（4 annotators x 6 samples）验证维度1和维度2的指标计算
无 API 调用，纯本地运行
"""

import sys
from pathlib import Path

# 添加路径以便直接运行
sys.path.insert(0, str(Path(__file__).parent))

from iaa_data_structures import (
    ALL_VALUE_IDS,
    VALUE_NAMES,
    AnnotatorAnnotation,
    AnnotationMatrix,
    PairwiseAgreementResult,
    MultiAnnotatorAgreementResult,
    IAAExperimentResults,
)
import iaa_metrics
from iaa_report_generator import IAAReportGenerator


# ============================================================
# 辅助函数
# ============================================================
def assert_close(actual, expected, name, tol=1e-6):
    if abs(actual - expected) > tol:
        print(f"  FAIL: {name}: expected={expected}, got={actual}")
        return False
    print(f"  PASS: {name} = {actual:.6f}")
    return True


def assert_in_range(actual, lo, hi, name):
    if not (lo <= actual <= hi):
        print(f"  FAIL: {name}: expected [{lo},{hi}], got={actual}")
        return False
    print(f"  PASS: {name} = {actual:.6f} in [{lo},{hi}]")
    return True


# ============================================================
# Mock 数据构建
# ============================================================
def build_mock_annotation_matrix() -> AnnotationMatrix:
    """构建 4 annotators x 6 samples 的模拟标注矩阵"""
    annotators = ["Human", "deepseek-chat", "gpt-4o", "claude"]
    samples = ["sample_001", "sample_002", "sample_003",
               "sample_004", "sample_005", "sample_006"]
    scenario_types = {
        "sample_001": "code", "sample_002": "code", "sample_003": "code",
        "sample_004": "text", "sample_005": "text", "sample_006": "text",
    }

    # 定义标注数据: (has_risk, value_set)
    data = {
        "sample_001": {
            "Human":          (True,  {"HV9", "HV10"}),
            "deepseek-chat":  (True,  {"HV9"}),
            "gpt-4o":         (True,  {"HV9", "HV10"}),
            "claude":         (False, set()),
        },
        "sample_002": {
            "Human":          (True,  {"HV3", "HV4"}),
            "deepseek-chat":  (True,  {"HV3"}),
            "gpt-4o":         (True,  {"HV3", "HV4"}),
            "claude":         (True,  {"HV4"}),
        },
        "sample_003": {
            "Human":          (False, set()),
            "deepseek-chat":  (False, set()),
            "gpt-4o":         (False, set()),
            "claude":         (False, set()),
        },
        "sample_004": {
            "Human":          (True,  {"SV2", "SV5"}),
            "deepseek-chat":  (True,  {"SV2"}),
            "gpt-4o":         (True,  {"SV2", "SV5"}),
            "claude":         (True,  {"SV5"}),
        },
        "sample_005": {
            "Human":          (False, set()),
            "deepseek-chat":  (True,  {"SV1"}),  # 分歧样本
            "gpt-4o":         (False, set()),
            "claude":         (False, set()),
        },
        "sample_006": {
            "Human":          (True,  {"HV9"}),
            "deepseek-chat":  (True,  {"HV9"}),
            "gpt-4o":         (True,  {"HV9"}),
            "claude":         (True,  {"HV9"}),  # 完全一致
        },
    }

    annotations = {}
    for sid in samples:
        annotations[sid] = {}
        for aid in annotators:
            has_risk, value_set = data[sid][aid]
            annotations[sid][aid] = AnnotatorAnnotation(
                annotator_id=aid,
                sample_id=sid,
                has_risk=has_risk,
                value_set=value_set,
                confidence_vector={v: 0.9 for v in value_set},
            )

    return AnnotationMatrix(
        sample_ids=samples,
        annotator_ids=annotators,
        annotations=annotations,
        scenario_types=scenario_types,
    )


# ============================================================
# 测试: 指标纯函数
# ============================================================
def test_metrics_pure_functions():
    print("\n" + "=" * 60)
    print("测试: 指标纯函数")
    print("=" * 60)
    passed = 0
    total = 0

    # --- Cohen's Kappa ---
    print("\n--- Cohen's Kappa ---")

    # 完全一致
    total += 1
    if assert_close(
        iaa_metrics.cohen_kappa_binary(
            [True, True, False, True, False, False],
            [True, True, False, True, False, False],
        ), 1.0, "完全一致 → κ=1.0"
    ):
        passed += 1

    # 完全不一致 (对称边际分布才能得到 κ=-1)
    total += 1
    if assert_close(
        iaa_metrics.cohen_kappa_binary(
            [True, False, True, False],
            [False, True, False, True],
        ), -1.0, "完全不一致(对称边际) → κ=-1.0"
    ):
        passed += 1

    # 空列表
    total += 1
    if assert_close(
        iaa_metrics.cohen_kappa_binary([], []),
        0.0, "空列表 → κ=0.0"
    ):
        passed += 1

    # 随机一致 (κ≈0)
    total += 1
    if assert_close(
        iaa_metrics.cohen_kappa_binary(
            [True, True, False, False],
            [True, False, True, False],
        ), 0.0, "[T,T,F,F] vs [T,F,T,F] → κ=0.0"
    ):
        passed += 1

    # --- Percent Agreement ---
    print("\n--- Percent Agreement ---")

    total += 1
    if assert_close(
        iaa_metrics.percent_agreement(
            [True, True, False, False],
            [True, True, False, False],
        ), 1.0, "完全一致 → 100%"
    ):
        passed += 1

    total += 1
    if assert_close(
        iaa_metrics.percent_agreement(
            [True, True, False, False],
            [True, False, True, False],
        ), 0.5, "一半一致 → 50%"
    ):
        passed += 1

    # --- Fleiss' Kappa ---
    print("\n--- Fleiss' Kappa ---")

    # 完全一致 (所有annotator分到同一类)
    total += 1
    if assert_close(
        iaa_metrics.fleiss_kappa([[4, 0], [0, 4], [4, 0]]),
        1.0, "完全一致 → κ=1.0"
    ):
        passed += 1

    # 空矩阵
    total += 1
    if assert_close(
        iaa_metrics.fleiss_kappa([]),
        0.0, "空矩阵 → κ=0.0"
    ):
        passed += 1

    # --- Krippendorff's Alpha ---
    print("\n--- Krippendorff's Alpha ---")

    # 完全一致
    total += 1
    if assert_close(
        iaa_metrics.krippendorff_alpha_nominal([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]), 1.0, "完全一致 → α=1.0"
    ):
        passed += 1

    # 含缺失值的完全一致
    total += 1
    if assert_close(
        iaa_metrics.krippendorff_alpha_nominal([
            [1, 1, None, 0],
            [1, 1, 0,    0],
            [1, None, 0, 0],
        ]), 1.0, "含缺失的完全一致 → α=1.0"
    ):
        passed += 1

    # --- PABAK ---
    print("\n--- PABAK ---")

    total += 1
    if assert_close(
        iaa_metrics.pabak(
            [True, True, False, True, False, False],
            [True, True, False, True, False, False],
        ), 1.0, "完全一致 → PABAK=1.0"
    ):
        passed += 1

    # 50% 一致 → PABAK = 2*0.5-1 = 0
    total += 1
    if assert_close(
        iaa_metrics.pabak(
            [True, True, False, False],
            [True, False, True, False],
        ), 0.0, "50%一致 → PABAK=0.0"
    ):
        passed += 1

    # 一方全标正类: po=3/6=0.5 → PABAK = 2*0.5-1 = 0.0
    total += 1
    if assert_close(
        iaa_metrics.pabak(
            [True, True, False, True, False, False],
            [True, True, True, True, True, True],
        ), 0.0, "一方全正(po=50%) → PABAK=0.0"
    ):
        passed += 1

    # --- Gwet's AC1 ---
    print("\n--- Gwet's AC1 ---")

    total += 1
    if assert_close(
        iaa_metrics.gwet_ac1(
            [True, True, False, True, False, False],
            [True, True, False, True, False, False],
        ), 1.0, "完全一致 → AC1=1.0"
    ):
        passed += 1

    # 一方全标正类时 AC1 应显著不同于 Cohen's κ
    total += 1
    kappa_val = iaa_metrics.cohen_kappa_binary(
        [True, True, False, True, False, False],
        [True, True, True, True, True, True],
    )
    ac1_val = iaa_metrics.gwet_ac1(
        [True, True, False, True, False, False],
        [True, True, True, True, True, True],
    )
    if ac1_val > kappa_val:
        print(f"  PASS: 一方全正时 AC1({ac1_val:.4f}) > κ({kappa_val:.4f})")
        passed += 1
    else:
        print(f"  FAIL: 一方全正时 AC1({ac1_val:.4f}) <= κ({kappa_val:.4f})")

    # --- Pairwise Jaccard ---
    print("\n--- Pairwise Jaccard ---")

    # 双空集
    total += 1
    if assert_close(
        iaa_metrics.pairwise_jaccard([set()], [set()]),
        1.0, "双空集 → Jaccard=1.0"
    ):
        passed += 1

    # 一空一非空
    total += 1
    if assert_close(
        iaa_metrics.pairwise_jaccard([{"HV1"}], [set()]),
        0.0, "一空一非空 → Jaccard=0.0"
    ):
        passed += 1

    # 完全相同
    total += 1
    if assert_close(
        iaa_metrics.pairwise_jaccard(
            [{"HV1", "HV2"}], [{"HV1", "HV2"}]
        ), 1.0, "完全相同 → Jaccard=1.0"
    ):
        passed += 1

    # 部分重叠: {HV1,HV2} vs {HV1,HV3} → 1/3
    total += 1
    if assert_close(
        iaa_metrics.pairwise_jaccard(
            [{"HV1", "HV2"}], [{"HV1", "HV3"}]
        ), 1.0 / 3.0, "部分重叠 → Jaccard=1/3"
    ):
        passed += 1

    # --- Pairwise Symmetric F1 ---
    print("\n--- Pairwise Symmetric F1 ---")

    total += 1
    if assert_close(
        iaa_metrics.pairwise_symmetric_f1(
            [{"HV1", "HV2"}], [{"HV1", "HV2"}]
        ), 1.0, "完全相同 → F1=1.0"
    ):
        passed += 1

    total += 1
    if assert_close(
        iaa_metrics.pairwise_symmetric_f1([set()], [set()]),
        1.0, "双空集 → F1=1.0"
    ):
        passed += 1

    # {HV1,HV2} vs {HV1,HV3} → 2*1/(2+2) = 0.5
    total += 1
    if assert_close(
        iaa_metrics.pairwise_symmetric_f1(
            [{"HV1", "HV2"}], [{"HV1", "HV3"}]
        ), 0.5, "部分重叠 → F1=0.5"
    ):
        passed += 1

    # --- Per-value Fleiss Kappa ---
    print("\n--- Per-value Fleiss Kappa ---")

    matrix = build_mock_annotation_matrix()
    per_value = iaa_metrics.per_value_fleiss_kappa(
        matrix.build_per_value_binary_matrix, ALL_VALUE_IDS
    )
    total += 1
    if len(per_value) == 20:
        print(f"  PASS: per_value_fleiss_kappa 返回 20 个条目")
        passed += 1
    else:
        print(f"  FAIL: per_value_fleiss_kappa 返回 {len(per_value)} 个条目（期望 20）")

    total += 1
    all_in_range = all(-1.0 <= v <= 1.0 for v in per_value.values())
    if all_in_range:
        print(f"  PASS: 所有 per_value kappa 在 [-1, 1] 范围内")
        passed += 1
    else:
        print(f"  FAIL: 存在 per_value kappa 超出 [-1, 1]")

    print(f"\n纯函数测试: {passed}/{total} 通过")
    return passed, total


# ============================================================
# 测试: 数据结构
# ============================================================
def test_data_structures():
    print("\n" + "=" * 60)
    print("测试: 数据结构")
    print("=" * 60)
    passed = 0
    total = 0

    matrix = build_mock_annotation_matrix()

    # slice_by_scenario
    print("\n--- slice_by_scenario ---")
    code_matrix = matrix.slice_by_scenario("code")
    total += 1
    if len(code_matrix.sample_ids) == 3:
        print(f"  PASS: code slice = 3 samples")
        passed += 1
    else:
        print(f"  FAIL: code slice = {len(code_matrix.sample_ids)} samples (expected 3)")

    text_matrix = matrix.slice_by_scenario("text")
    total += 1
    if len(text_matrix.sample_ids) == 3:
        print(f"  PASS: text slice = 3 samples")
        passed += 1
    else:
        print(f"  FAIL: text slice = {len(text_matrix.sample_ids)} samples (expected 3)")

    # get_risk_labels_for_pair
    print("\n--- get_risk_labels_for_pair ---")
    risk_a, risk_b = matrix.get_risk_labels_for_pair("Human", "gpt-4o")
    total += 1
    if len(risk_a) == len(risk_b) == 6:
        print(f"  PASS: get_risk_labels_for_pair 返回 6 对")
        passed += 1
    else:
        print(f"  FAIL: 返回 {len(risk_a)}, {len(risk_b)} (expected 6, 6)")

    # Human: [T,T,F,T,F,T], gpt-4o: [T,T,F,T,F,T] → 完全一致
    total += 1
    if risk_a == risk_b:
        print(f"  PASS: Human vs gpt-4o risk labels 完全一致")
        passed += 1
    else:
        print(f"  FAIL: Human vs gpt-4o risk labels 不一致")
        print(f"    Human:  {risk_a}")
        print(f"    gpt-4o: {risk_b}")

    # get_value_sets_for_pair
    print("\n--- get_value_sets_for_pair ---")
    sets_a, sets_b = matrix.get_value_sets_for_pair("Human", "deepseek-chat")
    total += 1
    if len(sets_a) == len(sets_b) == 6:
        print(f"  PASS: get_value_sets_for_pair 返回 6 对")
        passed += 1
    else:
        print(f"  FAIL: 返回 {len(sets_a)}, {len(sets_b)} (expected 6, 6)")

    # build_fleiss_risk_matrix
    print("\n--- build_fleiss_risk_matrix ---")
    fleiss_mat = matrix.build_fleiss_risk_matrix()
    total += 1
    if len(fleiss_mat) == 6:
        print(f"  PASS: Fleiss matrix = 6 行")
        passed += 1
    else:
        print(f"  FAIL: Fleiss matrix = {len(fleiss_mat)} 行 (expected 6)")

    total += 1
    row_sums = [sum(row) for row in fleiss_mat]
    if all(s == 4 for s in row_sums):
        print(f"  PASS: 每行之和 = 4 (4个annotator)")
        passed += 1
    else:
        print(f"  FAIL: 行和 = {row_sums} (expected all 4)")

    # build_krippendorff_risk_data
    print("\n--- build_krippendorff_risk_data ---")
    kripp = matrix.build_krippendorff_risk_data()
    total += 1
    if len(kripp) == 4 and len(kripp[0]) == 6:
        print(f"  PASS: Krippendorff data = 4 x 6")
        passed += 1
    else:
        print(f"  FAIL: Krippendorff data 尺寸不正确")

    print(f"\n数据结构测试: {passed}/{total} 通过")
    return passed, total


# ============================================================
# 测试: 集成流程
# ============================================================
def test_integration():
    print("\n" + "=" * 60)
    print("测试: 集成流程 (mock 数据)")
    print("=" * 60)
    passed = 0
    total = 0

    matrix = build_mock_annotation_matrix()

    # compute pairwise
    import itertools
    pairwise = {}
    for a, b in itertools.combinations(matrix.annotator_ids, 2):
        key = f"{a}_vs_{b}" if a < b else f"{b}_vs_{a}"
        risk_a, risk_b = matrix.get_risk_labels_for_pair(a, b)
        sets_a, sets_b = matrix.get_value_sets_for_pair(a, b)
        pairwise[key] = PairwiseAgreementResult(
            annotator_a=min(a, b),
            annotator_b=max(a, b),
            dim1_cohen_kappa=iaa_metrics.cohen_kappa_binary(risk_a, risk_b),
            dim1_pabak=iaa_metrics.pabak(risk_a, risk_b),
            dim1_gwet_ac1=iaa_metrics.gwet_ac1(risk_a, risk_b),
            dim1_percent_agreement=iaa_metrics.percent_agreement(risk_a, risk_b),
            dim1_n_samples=len(risk_a),
            dim2_jaccard=iaa_metrics.pairwise_jaccard(sets_a, sets_b),
            dim2_symmetric_f1=iaa_metrics.pairwise_symmetric_f1(sets_a, sets_b),
            dim2_n_samples=len(sets_a),
        )

    # 检查 pairwise 数量 = C(4,2) = 6
    total += 1
    if len(pairwise) == 6:
        print(f"  PASS: pairwise 结果 = 6 对")
        passed += 1
    else:
        print(f"  FAIL: pairwise 结果 = {len(pairwise)} (expected 6)")

    # 检查所有 kappa 在 [-1, 1]
    total += 1
    kappas_ok = all(-1.0 <= pr.dim1_cohen_kappa <= 1.0 for pr in pairwise.values())
    if kappas_ok:
        print(f"  PASS: 所有 Cohen's κ 在 [-1, 1]")
        passed += 1
    else:
        print(f"  FAIL: 存在 Cohen's κ 超出范围")

    # 检查所有 Jaccard 在 [0, 1]
    total += 1
    jaccard_ok = all(0.0 <= pr.dim2_jaccard <= 1.0 for pr in pairwise.values())
    if jaccard_ok:
        print(f"  PASS: 所有 Jaccard 在 [0, 1]")
        passed += 1
    else:
        print(f"  FAIL: 存在 Jaccard 超出范围")

    # 检查所有 F1 在 [0, 1]
    total += 1
    f1_ok = all(0.0 <= pr.dim2_symmetric_f1 <= 1.0 for pr in pairwise.values())
    if f1_ok:
        print(f"  PASS: 所有 Symmetric F1 在 [0, 1]")
        passed += 1
    else:
        print(f"  FAIL: 存在 Symmetric F1 超出范围")

    # compute multi-annotator
    fleiss_mat = matrix.build_fleiss_risk_matrix()
    fleiss_k = iaa_metrics.fleiss_kappa(fleiss_mat)
    total += 1
    if assert_in_range(fleiss_k, -1.0, 1.0, "Fleiss κ"):
        passed += 1

    kripp_data = matrix.build_krippendorff_risk_data()
    kripp_a = iaa_metrics.krippendorff_alpha_nominal(kripp_data)
    total += 1
    if assert_in_range(kripp_a, -1.0, 1.0, "Krippendorff α"):
        passed += 1

    # Fleiss κ 和 Krippendorff α 趋势应一致
    total += 1
    if (fleiss_k >= 0 and kripp_a >= 0) or (fleiss_k < 0 and kripp_a < 0):
        print(f"  PASS: Fleiss κ ({fleiss_k:.4f}) 和 Krippendorff α ({kripp_a:.4f}) 符号一致")
        passed += 1
    else:
        print(f"  WARN: Fleiss κ ({fleiss_k:.4f}) 和 Krippendorff α ({kripp_a:.4f}) 符号不一致")

    # 打印完整结果用于人工审查
    print("\n  --- Pairwise 详细结果 ---")
    for key, pr in sorted(pairwise.items()):
        print(f"    {key}:")
        print(f"      κ={pr.dim1_cohen_kappa:.4f}, %agree={pr.dim1_percent_agreement:.4f}")
        print(f"      Jaccard={pr.dim2_jaccard:.4f}, F1={pr.dim2_symmetric_f1:.4f}")

    print(f"\n  --- 全体指标 ---")
    print(f"    Fleiss κ = {fleiss_k:.4f}")
    print(f"    Krippendorff α = {kripp_a:.4f}")

    # Per-value kappa
    per_value = iaa_metrics.per_value_fleiss_kappa(
        matrix.build_per_value_binary_matrix, ALL_VALUE_IDS
    )
    print(f"\n  --- Per-Value Fleiss κ (非零值) ---")
    for vid, k in per_value.items():
        if k != 0.0:
            print(f"    {vid} ({VALUE_NAMES[vid]}): {k:.4f}")

    print(f"\n集成测试: {passed}/{total} 通过")
    return passed, total


# ============================================================
# 测试: 报告生成
# ============================================================
def test_report_generation():
    print("\n" + "=" * 60)
    print("测试: 报告生成")
    print("=" * 60)
    passed = 0
    total = 0

    matrix = build_mock_annotation_matrix()

    # 构建完整结果
    import itertools
    pairwise = {}
    for a, b in itertools.combinations(matrix.annotator_ids, 2):
        key = f"{a}_vs_{b}" if a < b else f"{b}_vs_{a}"
        risk_a, risk_b = matrix.get_risk_labels_for_pair(a, b)
        sets_a, sets_b = matrix.get_value_sets_for_pair(a, b)
        pairwise[key] = PairwiseAgreementResult(
            annotator_a=min(a, b),
            annotator_b=max(a, b),
            dim1_cohen_kappa=iaa_metrics.cohen_kappa_binary(risk_a, risk_b),
            dim1_pabak=iaa_metrics.pabak(risk_a, risk_b),
            dim1_gwet_ac1=iaa_metrics.gwet_ac1(risk_a, risk_b),
            dim1_percent_agreement=iaa_metrics.percent_agreement(risk_a, risk_b),
            dim1_n_samples=len(risk_a),
            dim2_jaccard=iaa_metrics.pairwise_jaccard(sets_a, sets_b),
            dim2_symmetric_f1=iaa_metrics.pairwise_symmetric_f1(sets_a, sets_b),
            dim2_n_samples=len(sets_a),
        )

    per_value = iaa_metrics.per_value_fleiss_kappa(
        matrix.build_per_value_binary_matrix, ALL_VALUE_IDS
    )
    macro_values = [v for v in per_value.values() if v != 0.0]
    macro_avg = sum(macro_values) / len(macro_values) if macro_values else 0.0

    kappas = [pr.dim1_cohen_kappa for pr in pairwise.values()]
    jaccards = [pr.dim2_jaccard for pr in pairwise.values()]
    f1s = [pr.dim2_symmetric_f1 for pr in pairwise.values()]

    multi = MultiAnnotatorAgreementResult(
        dim1_fleiss_kappa=iaa_metrics.fleiss_kappa(matrix.build_fleiss_risk_matrix()),
        dim1_krippendorff_alpha=iaa_metrics.krippendorff_alpha_nominal(
            matrix.build_krippendorff_risk_data()
        ),
        dim1_avg_pairwise_kappa=sum(kappas) / len(kappas),
        dim2_avg_pairwise_jaccard=sum(jaccards) / len(jaccards),
        dim2_avg_pairwise_f1=sum(f1s) / len(f1s),
        dim2_per_value_fleiss_kappa=per_value,
        dim2_macro_avg_value_kappa=macro_avg,
        n_annotators=4,
        n_samples=6,
    )

    results = IAAExperimentResults(
        pairwise=pairwise,
        multi_annotator=multi,
        scenario_type="overall",
        annotator_ids=matrix.annotator_ids,
        n_samples=6,
    )

    reporter = IAAReportGenerator(output_dir="/tmp/test_iaa_report")
    results_by_scenario = {"overall": results}

    # 测试各报告生成不报错
    total += 1
    try:
        md = reporter.generate_pairwise_matrix_md(results)
        assert "Cohen" in md or "κ" in md or "|" in md
        print(f"  PASS: generate_pairwise_matrix_md 成功")
        passed += 1
    except Exception as e:
        print(f"  FAIL: generate_pairwise_matrix_md 异常: {e}")

    total += 1
    try:
        md = reporter.generate_overall_table_md(results_by_scenario)
        assert "Fleiss" in md or "|" in md
        print(f"  PASS: generate_overall_table_md 成功")
        passed += 1
    except Exception as e:
        print(f"  FAIL: generate_overall_table_md 异常: {e}")

    total += 1
    try:
        md = reporter.generate_per_value_table_md(results)
        assert "HV1" in md
        print(f"  PASS: generate_per_value_table_md 成功")
        passed += 1
    except Exception as e:
        print(f"  FAIL: generate_per_value_table_md 异常: {e}")

    total += 1
    try:
        md = reporter.generate_human_vs_llm_table_md(results)
        assert "Human" in md or "vs" in md
        print(f"  PASS: generate_human_vs_llm_table_md 成功")
        passed += 1
    except Exception as e:
        print(f"  FAIL: generate_human_vs_llm_table_md 异常: {e}")

    total += 1
    try:
        latex = reporter.generate_latex_tables(results_by_scenario)
        assert "\\begin{table}" in latex
        print(f"  PASS: generate_latex_tables 成功")
        passed += 1
    except Exception as e:
        print(f"  FAIL: generate_latex_tables 异常: {e}")

    total += 1
    try:
        saved = reporter.save_all(results_by_scenario)
        assert len(saved) > 0
        print(f"  PASS: save_all 保存了 {len(saved)} 个文件")
        for name, path in saved.items():
            print(f"    {name}: {path}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: save_all 异常: {e}")

    print(f"\n报告生成测试: {passed}/{total} 通过")
    return passed, total


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    total_passed = 0
    total_tests = 0

    p, t = test_metrics_pure_functions()
    total_passed += p
    total_tests += t

    p, t = test_data_structures()
    total_passed += p
    total_tests += t

    p, t = test_integration()
    total_passed += p
    total_tests += t

    p, t = test_report_generation()
    total_passed += p
    total_tests += t

    print("\n" + "=" * 60)
    print(f"总计: {total_passed}/{total_tests} 测试通过")
    print("=" * 60)

    if total_passed == total_tests:
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print(f"FAILED: {total_tests - total_passed} 个测试未通过")
        sys.exit(1)
