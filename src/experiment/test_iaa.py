#!/usr/bin/env python
"""
IAA Local Test
Verify Dimension 1 and Dimension 2 metric calculations using mock data (4 annotators x 6 samples)
No API calls, runs entirely locally
"""

import sys
from pathlib import Path

# Add path for direct execution
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
# Helper Functions
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
# Mock Data Construction
# ============================================================
def build_mock_annotation_matrix() -> AnnotationMatrix:
    """Build a mock annotation matrix of 4 annotators x 6 samples"""
    annotators = ["Human", "deepseek-chat", "gpt-4o", "claude"]
    samples = ["sample_001", "sample_002", "sample_003",
               "sample_004", "sample_005", "sample_006"]
    scenario_types = {
        "sample_001": "code", "sample_002": "code", "sample_003": "code",
        "sample_004": "text", "sample_005": "text", "sample_006": "text",
    }

    # Define annotation data: (has_risk, value_set)
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
            "deepseek-chat":  (True,  {"SV1"}),  # Disagreement sample
            "gpt-4o":         (False, set()),
            "claude":         (False, set()),
        },
        "sample_006": {
            "Human":          (True,  {"HV9"}),
            "deepseek-chat":  (True,  {"HV9"}),
            "gpt-4o":         (True,  {"HV9"}),
            "claude":         (True,  {"HV9"}),  # Full agreement
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
# Test: Metric Pure Functions
# ============================================================
def test_metrics_pure_functions():
    print("\n" + "=" * 60)
    print("Test: Metric Pure Functions")
    print("=" * 60)
    passed = 0
    total = 0

    # --- Cohen's Kappa ---
    print("\n--- Cohen's Kappa ---")

    # Perfect agreement
    total += 1
    if assert_close(
        iaa_metrics.cohen_kappa_binary(
            [True, True, False, True, False, False],
            [True, True, False, True, False, False],
        ), 1.0, "Perfect agreement -> kappa=1.0"
    ):
        passed += 1

    # Perfect disagreement (symmetric marginal distribution yields kappa=-1)
    total += 1
    if assert_close(
        iaa_metrics.cohen_kappa_binary(
            [True, False, True, False],
            [False, True, False, True],
        ), -1.0, "Perfect disagreement (symmetric marginals) -> kappa=-1.0"
    ):
        passed += 1

    # Empty list
    total += 1
    if assert_close(
        iaa_metrics.cohen_kappa_binary([], []),
        0.0, "Empty list -> kappa=0.0"
    ):
        passed += 1

    # Random agreement (kappa ~ 0)
    total += 1
    if assert_close(
        iaa_metrics.cohen_kappa_binary(
            [True, True, False, False],
            [True, False, True, False],
        ), 0.0, "[T,T,F,F] vs [T,F,T,F] -> kappa=0.0"
    ):
        passed += 1

    # --- Percent Agreement ---
    print("\n--- Percent Agreement ---")

    total += 1
    if assert_close(
        iaa_metrics.percent_agreement(
            [True, True, False, False],
            [True, True, False, False],
        ), 1.0, "Perfect agreement -> 100%"
    ):
        passed += 1

    total += 1
    if assert_close(
        iaa_metrics.percent_agreement(
            [True, True, False, False],
            [True, False, True, False],
        ), 0.5, "Half agreement -> 50%"
    ):
        passed += 1

    # --- Fleiss' Kappa ---
    print("\n--- Fleiss' Kappa ---")

    # Perfect agreement (all annotators assigned to same category)
    total += 1
    if assert_close(
        iaa_metrics.fleiss_kappa([[4, 0], [0, 4], [4, 0]]),
        1.0, "Perfect agreement -> kappa=1.0"
    ):
        passed += 1

    # Empty matrix
    total += 1
    if assert_close(
        iaa_metrics.fleiss_kappa([]),
        0.0, "Empty matrix -> kappa=0.0"
    ):
        passed += 1

    # --- Krippendorff's Alpha ---
    print("\n--- Krippendorff's Alpha ---")

    # Perfect agreement
    total += 1
    if assert_close(
        iaa_metrics.krippendorff_alpha_nominal([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]), 1.0, "Perfect agreement -> alpha=1.0"
    ):
        passed += 1

    # Perfect agreement with missing values
    total += 1
    if assert_close(
        iaa_metrics.krippendorff_alpha_nominal([
            [1, 1, None, 0],
            [1, 1, 0,    0],
            [1, None, 0, 0],
        ]), 1.0, "Perfect agreement with missing -> alpha=1.0"
    ):
        passed += 1

    # --- PABAK ---
    print("\n--- PABAK ---")

    total += 1
    if assert_close(
        iaa_metrics.pabak(
            [True, True, False, True, False, False],
            [True, True, False, True, False, False],
        ), 1.0, "Perfect agreement -> PABAK=1.0"
    ):
        passed += 1

    # 50% agreement -> PABAK = 2*0.5-1 = 0
    total += 1
    if assert_close(
        iaa_metrics.pabak(
            [True, True, False, False],
            [True, False, True, False],
        ), 0.0, "50% agreement -> PABAK=0.0"
    ):
        passed += 1

    # One side all positive: po=3/6=0.5 -> PABAK = 2*0.5-1 = 0.0
    total += 1
    if assert_close(
        iaa_metrics.pabak(
            [True, True, False, True, False, False],
            [True, True, True, True, True, True],
        ), 0.0, "One side all positive (po=50%) -> PABAK=0.0"
    ):
        passed += 1

    # --- Gwet's AC1 ---
    print("\n--- Gwet's AC1 ---")

    total += 1
    if assert_close(
        iaa_metrics.gwet_ac1(
            [True, True, False, True, False, False],
            [True, True, False, True, False, False],
        ), 1.0, "Perfect agreement -> AC1=1.0"
    ):
        passed += 1

    # When one side labels all positive, AC1 should differ significantly from Cohen's kappa
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
        print(f"  PASS: One side all positive: AC1({ac1_val:.4f}) > kappa({kappa_val:.4f})")
        passed += 1
    else:
        print(f"  FAIL: One side all positive: AC1({ac1_val:.4f}) <= kappa({kappa_val:.4f})")

    # --- Pairwise Jaccard ---
    print("\n--- Pairwise Jaccard ---")

    # Both empty sets
    total += 1
    if assert_close(
        iaa_metrics.pairwise_jaccard([set()], [set()]),
        1.0, "Both empty sets -> Jaccard=1.0"
    ):
        passed += 1

    # One empty, one non-empty
    total += 1
    if assert_close(
        iaa_metrics.pairwise_jaccard([{"HV1"}], [set()]),
        0.0, "One empty one non-empty -> Jaccard=0.0"
    ):
        passed += 1

    # Identical sets
    total += 1
    if assert_close(
        iaa_metrics.pairwise_jaccard(
            [{"HV1", "HV2"}], [{"HV1", "HV2"}]
        ), 1.0, "Identical sets -> Jaccard=1.0"
    ):
        passed += 1

    # Partial overlap: {HV1,HV2} vs {HV1,HV3} -> 1/3
    total += 1
    if assert_close(
        iaa_metrics.pairwise_jaccard(
            [{"HV1", "HV2"}], [{"HV1", "HV3"}]
        ), 1.0 / 3.0, "Partial overlap -> Jaccard=1/3"
    ):
        passed += 1

    # --- Pairwise Symmetric F1 ---
    print("\n--- Pairwise Symmetric F1 ---")

    total += 1
    if assert_close(
        iaa_metrics.pairwise_symmetric_f1(
            [{"HV1", "HV2"}], [{"HV1", "HV2"}]
        ), 1.0, "Identical sets -> F1=1.0"
    ):
        passed += 1

    total += 1
    if assert_close(
        iaa_metrics.pairwise_symmetric_f1([set()], [set()]),
        1.0, "Both empty sets -> F1=1.0"
    ):
        passed += 1

    # {HV1,HV2} vs {HV1,HV3} -> 2*1/(2+2) = 0.5
    total += 1
    if assert_close(
        iaa_metrics.pairwise_symmetric_f1(
            [{"HV1", "HV2"}], [{"HV1", "HV3"}]
        ), 0.5, "Partial overlap -> F1=0.5"
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
        print(f"  PASS: per_value_fleiss_kappa returned 20 entries")
        passed += 1
    else:
        print(f"  FAIL: per_value_fleiss_kappa returned {len(per_value)} entries (expected 20)")

    total += 1
    all_in_range = all(-1.0 <= v <= 1.0 for v in per_value.values())
    if all_in_range:
        print(f"  PASS: All per_value kappa within [-1, 1]")
        passed += 1
    else:
        print(f"  FAIL: Some per_value kappa outside [-1, 1]")

    print(f"\nPure function tests: {passed}/{total} passed")
    return passed, total


# ============================================================
# Test: Data Structures
# ============================================================
def test_data_structures():
    print("\n" + "=" * 60)
    print("Test: Data Structures")
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
        print(f"  PASS: get_risk_labels_for_pair returned 6 pairs")
        passed += 1
    else:
        print(f"  FAIL: returned {len(risk_a)}, {len(risk_b)} (expected 6, 6)")

    # Human: [T,T,F,T,F,T], gpt-4o: [T,T,F,T,F,T] -> perfect agreement
    total += 1
    if risk_a == risk_b:
        print(f"  PASS: Human vs gpt-4o risk labels in perfect agreement")
        passed += 1
    else:
        print(f"  FAIL: Human vs gpt-4o risk labels disagree")
        print(f"    Human:  {risk_a}")
        print(f"    gpt-4o: {risk_b}")

    # get_value_sets_for_pair
    print("\n--- get_value_sets_for_pair ---")
    sets_a, sets_b = matrix.get_value_sets_for_pair("Human", "deepseek-chat")
    total += 1
    if len(sets_a) == len(sets_b) == 6:
        print(f"  PASS: get_value_sets_for_pair returned 6 pairs")
        passed += 1
    else:
        print(f"  FAIL: returned {len(sets_a)}, {len(sets_b)} (expected 6, 6)")

    # build_fleiss_risk_matrix
    print("\n--- build_fleiss_risk_matrix ---")
    fleiss_mat = matrix.build_fleiss_risk_matrix()
    total += 1
    if len(fleiss_mat) == 6:
        print(f"  PASS: Fleiss matrix = 6 rows")
        passed += 1
    else:
        print(f"  FAIL: Fleiss matrix = {len(fleiss_mat)} rows (expected 6)")

    total += 1
    row_sums = [sum(row) for row in fleiss_mat]
    if all(s == 4 for s in row_sums):
        print(f"  PASS: Each row sum = 4 (4 annotators)")
        passed += 1
    else:
        print(f"  FAIL: Row sums = {row_sums} (expected all 4)")

    # build_krippendorff_risk_data
    print("\n--- build_krippendorff_risk_data ---")
    kripp = matrix.build_krippendorff_risk_data()
    total += 1
    if len(kripp) == 4 and len(kripp[0]) == 6:
        print(f"  PASS: Krippendorff data = 4 x 6")
        passed += 1
    else:
        print(f"  FAIL: Krippendorff data dimensions incorrect")

    print(f"\nData structure tests: {passed}/{total} passed")
    return passed, total


# ============================================================
# Test: Integration Flow
# ============================================================
def test_integration():
    print("\n" + "=" * 60)
    print("Test: Integration Flow (mock data)")
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

    # Check pairwise count = C(4,2) = 6
    total += 1
    if len(pairwise) == 6:
        print(f"  PASS: pairwise results = 6 pairs")
        passed += 1
    else:
        print(f"  FAIL: pairwise results = {len(pairwise)} (expected 6)")

    # Check all kappa within [-1, 1]
    total += 1
    kappas_ok = all(-1.0 <= pr.dim1_cohen_kappa <= 1.0 for pr in pairwise.values())
    if kappas_ok:
        print(f"  PASS: All Cohen's kappa within [-1, 1]")
        passed += 1
    else:
        print(f"  FAIL: Some Cohen's kappa out of range")

    # Check all Jaccard within [0, 1]
    total += 1
    jaccard_ok = all(0.0 <= pr.dim2_jaccard <= 1.0 for pr in pairwise.values())
    if jaccard_ok:
        print(f"  PASS: All Jaccard within [0, 1]")
        passed += 1
    else:
        print(f"  FAIL: Some Jaccard out of range")

    # Check all F1 within [0, 1]
    total += 1
    f1_ok = all(0.0 <= pr.dim2_symmetric_f1 <= 1.0 for pr in pairwise.values())
    if f1_ok:
        print(f"  PASS: All Symmetric F1 within [0, 1]")
        passed += 1
    else:
        print(f"  FAIL: Some Symmetric F1 out of range")

    # compute multi-annotator
    fleiss_mat = matrix.build_fleiss_risk_matrix()
    fleiss_k = iaa_metrics.fleiss_kappa(fleiss_mat)
    total += 1
    if assert_in_range(fleiss_k, -1.0, 1.0, "Fleiss kappa"):
        passed += 1

    kripp_data = matrix.build_krippendorff_risk_data()
    kripp_a = iaa_metrics.krippendorff_alpha_nominal(kripp_data)
    total += 1
    if assert_in_range(kripp_a, -1.0, 1.0, "Krippendorff alpha"):
        passed += 1

    # Fleiss kappa and Krippendorff alpha trends should be consistent
    total += 1
    if (fleiss_k >= 0 and kripp_a >= 0) or (fleiss_k < 0 and kripp_a < 0):
        print(f"  PASS: Fleiss kappa ({fleiss_k:.4f}) and Krippendorff alpha ({kripp_a:.4f}) have consistent sign")
        passed += 1
    else:
        print(f"  WARN: Fleiss kappa ({fleiss_k:.4f}) and Krippendorff alpha ({kripp_a:.4f}) have inconsistent sign")

    # Print full results for manual review
    print("\n  --- Pairwise Detailed Results ---")
    for key, pr in sorted(pairwise.items()):
        print(f"    {key}:")
        print(f"      kappa={pr.dim1_cohen_kappa:.4f}, %agree={pr.dim1_percent_agreement:.4f}")
        print(f"      Jaccard={pr.dim2_jaccard:.4f}, F1={pr.dim2_symmetric_f1:.4f}")

    print(f"\n  --- Overall Metrics ---")
    print(f"    Fleiss kappa = {fleiss_k:.4f}")
    print(f"    Krippendorff alpha = {kripp_a:.4f}")

    # Per-value kappa
    per_value = iaa_metrics.per_value_fleiss_kappa(
        matrix.build_per_value_binary_matrix, ALL_VALUE_IDS
    )
    print(f"\n  --- Per-Value Fleiss kappa (non-zero values) ---")
    for vid, k in per_value.items():
        if k != 0.0:
            print(f"    {vid} ({VALUE_NAMES[vid]}): {k:.4f}")

    print(f"\nIntegration tests: {passed}/{total} passed")
    return passed, total


# ============================================================
# Test: Report Generation
# ============================================================
def test_report_generation():
    print("\n" + "=" * 60)
    print("Test: Report Generation")
    print("=" * 60)
    passed = 0
    total = 0

    matrix = build_mock_annotation_matrix()

    # Build complete results
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

    # Test that each report generator runs without errors
    total += 1
    try:
        md = reporter.generate_pairwise_matrix_md(results)
        assert "Cohen" in md or "κ" in md or "|" in md
        print(f"  PASS: generate_pairwise_matrix_md succeeded")
        passed += 1
    except Exception as e:
        print(f"  FAIL: generate_pairwise_matrix_md exception: {e}")

    total += 1
    try:
        md = reporter.generate_overall_table_md(results_by_scenario)
        assert "Fleiss" in md or "|" in md
        print(f"  PASS: generate_overall_table_md succeeded")
        passed += 1
    except Exception as e:
        print(f"  FAIL: generate_overall_table_md exception: {e}")

    total += 1
    try:
        md = reporter.generate_per_value_table_md(results)
        assert "HV1" in md
        print(f"  PASS: generate_per_value_table_md succeeded")
        passed += 1
    except Exception as e:
        print(f"  FAIL: generate_per_value_table_md exception: {e}")

    total += 1
    try:
        md = reporter.generate_human_vs_llm_table_md(results)
        assert "Human" in md or "vs" in md
        print(f"  PASS: generate_human_vs_llm_table_md succeeded")
        passed += 1
    except Exception as e:
        print(f"  FAIL: generate_human_vs_llm_table_md exception: {e}")

    total += 1
    try:
        latex = reporter.generate_latex_tables(results_by_scenario)
        assert "\\begin{table}" in latex
        print(f"  PASS: generate_latex_tables succeeded")
        passed += 1
    except Exception as e:
        print(f"  FAIL: generate_latex_tables exception: {e}")

    total += 1
    try:
        saved = reporter.save_all(results_by_scenario)
        assert len(saved) > 0
        print(f"  PASS: save_all saved {len(saved)} files")
        for name, path in saved.items():
            print(f"    {name}: {path}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: save_all exception: {e}")

    print(f"\nReport generation tests: {passed}/{total} passed")
    return passed, total


# ============================================================
# Main Entry
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
    print(f"Total: {total_passed}/{total_tests} tests passed")
    print("=" * 60)

    if total_passed == total_tests:
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print(f"FAILED: {total_tests - total_passed} tests did not pass")
        sys.exit(1)
