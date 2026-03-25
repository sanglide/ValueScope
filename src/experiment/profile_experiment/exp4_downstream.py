#!/usr/bin/env python
"""
Experiment 3: Profile as Bayesian Hypothesis Calibrator

Validates the weighting effect of ValueProfile as a posterior calibrator
on hypothesis confidence. Profile does not participate in hypothesis
generation directly; instead, it applies Bayesian weighting to
confidence scores after LLM output, thereby avoiding narrowing the
hypothesis space while improving identification accuracy.

Core formula:
    mu   = mean(all 20 profile scores)
    ratio[v] = profile_score[v] / mu
    weighted_conf[v] = raw_conf[v] * ratio[v]^alpha

alpha = 0 is equivalent to no calibration (pure baseline); alpha > 0
amplifies dimensions with higher Profile scores.
For projects without a Profile, a uniform prior (0.5) is used, where
ratio = 1, so any alpha has no effect.

Reuses the full dataset from the IAA experiment (68 code + 1097 issue text)
and its cached LLM outputs.
"""

import json
import logging
import sys
from pathlib import Path
from statistics import mean as _mean

project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from experiment.profile_experiment.profile_visualizer import ProfileVisualizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Value dimension IDs
# ---------------------------------------------------------------------------

ALL_VALUE_IDS = [f"HV{i}" for i in range(1, 11)] + [f"SV{i}" for i in range(1, 11)]


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def _build_uniform_profile() -> dict:
    """Return a uniform prior Profile (all dimensions = 0.5)."""
    return {
        "l2_scores": {f"HV{i}": 0.5 for i in range(1, 11)},
        "l3_scores": {f"SV{i}": 0.5 for i in range(1, 11)},
    }


def _resolve_sample_profile(sample: dict, profiles: dict) -> dict:
    """Match a sample's repo to the corresponding Profile; fall back to uniform prior on mismatch."""
    repo = sample.get("repo", "").lower().replace("-", "").replace("_", "")
    for pname, pdata in profiles.items():
        if pname.lower().replace("-", "").replace("_", "") in repo or \
           repo in pname.lower().replace("-", "").replace("_", ""):
            return pdata
    return _build_uniform_profile()


def _compute_profile_ratios(profile: dict) -> dict[str, float]:
    """Convert Profile's 20-dimension scores to normalized ratios: ratio[v] = score[v] / mu.

    For a uniform prior (all 0.5), mu=0.5, so all ratios=1.0 (no effect).
    For the degenerate case of mu=0, all ratios=1.0 (degrades to no weighting).
    """
    all_scores = []
    l2 = profile.get("l2_scores", {})
    l3 = profile.get("l3_scores", {})

    for vid in ALL_VALUE_IDS:
        if vid.startswith("HV"):
            all_scores.append(l2.get(vid, 0.5))
        else:
            all_scores.append(l3.get(vid, 0.5))

    mu = _mean(all_scores) if all_scores else 0.5
    if mu == 0:
        return {vid: 1.0 for vid in ALL_VALUE_IDS}

    ratios = {}
    for vid, score in zip(ALL_VALUE_IDS, all_scores):
        ratios[vid] = score / mu
    return ratios


# ---------------------------------------------------------------------------
# Bayesian weighting
# ---------------------------------------------------------------------------

def _apply_bayesian_weighting(
    raw_confidences: dict[str, float],
    ratios: dict[str, float],
    alpha: float,
    threshold: float = 0.5,
) -> dict:
    """Apply Bayesian posterior weighting to LLM raw prediction confidences.

    weighted_conf[v] = raw_conf[v] * ratio[v]^alpha
    Only dimensions predicted by the LLM (raw_conf > 0) have a chance to survive.

    Returns:
        {"has_value_risk": bool, "identified_values": list[str]}
    """
    surviving_values = []
    for vid in ALL_VALUE_IDS:
        raw = raw_confidences.get(vid, 0.0)
        if raw <= 0:
            continue
        ratio = ratios.get(vid, 1.0)
        weighted = raw * (ratio ** alpha)
        if weighted >= threshold:
            surviving_values.append(vid)

    return {
        "has_value_risk": len(surviving_values) > 0,
        "identified_values": surviving_values,
    }


# ---------------------------------------------------------------------------
# Metrics computation (consistent with original version)
# ---------------------------------------------------------------------------

def _compute_metrics(predictions: list[dict], ground_truths: list[dict]) -> dict:
    """Compute Risk Detection and Value Identification metrics."""
    tp = fp = fn = tn = 0
    val_tp = val_fp = val_fn = 0
    jaccard_sum = 0.0
    n_samples = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        pred_risk = pred.get("has_value_risk", False)
        gt_risk = gt.get("has_value_risk", False)
        pred_vals = set(pred.get("identified_values", []))
        gt_vals = set(gt.get("ground_truth_values", []))

        if pred_risk and gt_risk:
            tp += 1
        elif pred_risk and not gt_risk:
            fp += 1
        elif not pred_risk and gt_risk:
            fn += 1
        else:
            tn += 1

        val_tp += len(pred_vals & gt_vals)
        val_fp += len(pred_vals - gt_vals)
        val_fn += len(gt_vals - pred_vals)

        union = pred_vals | gt_vals
        if union:
            jaccard_sum += len(pred_vals & gt_vals) / len(union)
        elif not pred_vals and not gt_vals:
            jaccard_sum += 1.0

    risk_p = tp / (tp + fp) if (tp + fp) else 0.0
    risk_r = tp / (tp + fn) if (tp + fn) else 0.0
    risk_f1 = 2 * risk_p * risk_r / (risk_p + risk_r) if (risk_p + risk_r) else 0.0

    val_p = val_tp / (val_tp + val_fp) if (val_tp + val_fp) else 0.0
    val_r = val_tp / (val_tp + val_fn) if (val_tp + val_fn) else 0.0
    val_f1 = 2 * val_p * val_r / (val_p + val_r) if (val_p + val_r) else 0.0

    return {
        "Risk P": round(risk_p, 4),
        "Risk R": round(risk_r, 4),
        "Risk F1": round(risk_f1, 4),
        "Value P": round(val_p, 4),
        "Value R": round(val_r, 4),
        "Value F1": round(val_f1, 4),
        "Jaccard": round(jaccard_sum / n_samples, 4) if n_samples else 0.0,
    }


# ---------------------------------------------------------------------------
# IAA cache loading
# ---------------------------------------------------------------------------

def _load_iaa_baseline(
    iaa_cache_dir: str,
    model_key: str,
    sample_ids: list[str],
) -> list[dict]:
    """Load baseline predictions (including predicted_confidences) from IAA experiment cache.

    Returns:
        [{
            "has_value_risk": bool,
            "identified_values": list[str],
            "predicted_confidences": dict[str, float],
        }]
    """
    cache_dir = Path(iaa_cache_dir)
    results = []
    loaded = 0

    for sid in sample_ids:
        cache_file = cache_dir / f"{model_key}_{sid}_output.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                results.append({
                    "has_value_risk": data.get("predicted_has_risk", False),
                    "identified_values": data.get("predicted_values", []),
                    "predicted_confidences": data.get("predicted_confidences", {}),
                })
                loaded += 1
                continue
            except Exception:
                pass
        # Cache does not exist or parsing failed
        results.append({
            "has_value_risk": False,
            "identified_values": [],
            "predicted_confidences": {},
        })

    logger.info(f"[Exp3] Loaded {loaded}/{len(sample_ids)} baseline entries from IAA cache (with confidences)")
    return results


# ---------------------------------------------------------------------------
# Dataset loading (reuses IAA experiment loading logic)
# ---------------------------------------------------------------------------

def _load_iaa_datasets(config: dict) -> list[dict]:
    """Load all datasets used in the IAA experiment, returning a unified sample list.

    Returns:
        [{"sample_id", "scenario_content", "scenario_type", "has_value_risk",
          "ground_truth_values", "repo"}]
    """
    from experiment.data_loader import ScenarioDataLoader, IssuesDatasetLoader

    datasets_config = config.get("datasets", {})
    base_dir = Path(__file__).parent.parent  # src/experiment/
    samples = []

    for ds_key, ds_conf in datasets_config.items():
        if not ds_conf.get("enabled", True):
            continue

        ds_type = ds_conf.get("type", "json")
        scenario_type = ds_conf.get("scenario_type", "code")
        ds_path = base_dir / ds_conf.get("path", "")

        if ds_type == "json":
            loader = ScenarioDataLoader()
            loader.load_from_json(str(ds_path))
            for s in loader.get_samples():
                samples.append({
                    "sample_id": s.sample_id,
                    "scenario_content": s.scenario_content,
                    "scenario_type": scenario_type,
                    "has_value_risk": s.ground_truth_has_risk,
                    "ground_truth_values": s.ground_truth_values,
                    "repo": s.metadata.get("repo", "Signal-Android"),
                })
        elif ds_type == "issues_dataset":
            issues_loader = IssuesDatasetLoader(str(ds_path))
            for s in issues_loader.load(
                sample_per_project=ds_conf.get("sample_per_project"),
                max_text_length=ds_conf.get("max_text_length", 8000),
                seed=ds_conf.get("seed", 42),
            ):
                samples.append({
                    "sample_id": s.sample_id,
                    "scenario_content": s.scenario_content,
                    "scenario_type": "text",
                    "has_value_risk": s.ground_truth_has_risk,
                    "ground_truth_values": s.ground_truth_values,
                    "repo": s.metadata.get("project_name", "unknown"),
                })

    logger.info(f"[Exp3] Loaded IAA datasets: {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# Alpha parameter sweep
# ---------------------------------------------------------------------------

def _sweep_alpha(
    samples: list[dict],
    baselines: list[dict],
    profiles: dict[str, dict],
    alpha_values: list[float],
    threshold: float = 0.5,
) -> dict[float, dict[str, dict]]:
    """Perform Bayesian weighting for each alpha value, computing metrics grouped by scenario type.

    Returns:
        {alpha: {"code": metrics, "text": metrics, "overall": metrics}}
    """
    # Pre-compute ground truth and profile ratios for each sample
    ground_truths = []
    sample_ratios = []
    for sample in samples:
        ground_truths.append({
            "has_value_risk": sample["has_value_risk"],
            "ground_truth_values": sample["ground_truth_values"],
        })
        profile = _resolve_sample_profile(sample, profiles)
        sample_ratios.append(_compute_profile_ratios(profile))

    results = {}
    for alpha in alpha_values:
        # Apply Bayesian weighting to all samples
        predictions = []
        for i, (baseline, ratios) in enumerate(zip(baselines, sample_ratios)):
            raw_conf = baseline.get("predicted_confidences", {})
            pred = _apply_bayesian_weighting(raw_conf, ratios, alpha, threshold)
            predictions.append(pred)

        # Compute metrics grouped by scenario type
        alpha_metrics = {}
        for label, target_type in [("code", "code"), ("text", "text"), ("overall", None)]:
            if target_type:
                idxs = [i for i, s in enumerate(samples) if s["scenario_type"] == target_type]
            else:
                idxs = list(range(len(samples)))

            if not idxs:
                continue

            split_preds = [predictions[i] for i in idxs]
            split_gts = [ground_truths[i] for i in idxs]
            alpha_metrics[label] = _compute_metrics(split_preds, split_gts)
            alpha_metrics[label]["n"] = len(idxs)

        results[alpha] = alpha_metrics

    return results


def _find_optimal_alpha(
    sweep_results: dict[float, dict[str, dict]],
    target_metric: str = "Value F1",
) -> float:
    """Find the alpha with the highest target_metric on overall from sweep results."""
    best_alpha = 0.0
    best_score = -1.0
    for alpha, splits in sweep_results.items():
        score = splits.get("overall", {}).get(target_metric, 0.0)
        if score > best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha


# ---------------------------------------------------------------------------
# Main experiment entry point
# ---------------------------------------------------------------------------

def run_exp3(
    viz: ProfileVisualizer,
    profiles: dict[str, dict],
    config: dict,
    model_key: str = "qwen-plus",
    iaa_cache_dir: str = "",
    output_dir: str = "experiment_results/profile",
) -> dict:
    """Run Experiment 3: Profile as Bayesian Hypothesis Calibrator.

    A pure computation experiment requiring no LLM API calls. Loads LLM-predicted
    confidences from the IAA cache, applies Bayesian weighting using Profile as
    a prior, and sweeps over multiple alpha values.

    Args:
        viz: Visualizer
        profiles: {repo_name: profile_dict} from Exp1
        config: Full experiment config (used to load datasets identical to IAA)
        model_key: LLM used (must match the model used in the IAA experiment)
        iaa_cache_dir: IAA experiment LLM output cache directory
        output_dir: Output directory
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Read alpha sweep parameters from config
    exp3_cfg = config.get("profile_experiment", {}).get("exp3_downstream", {})
    alpha_values = exp3_cfg.get("alpha_values", [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    threshold = exp3_cfg.get("threshold", 0.5)
    target_metric = exp3_cfg.get("target_metric", "Value F1")

    # 1. Load datasets identical to the IAA experiment
    samples = _load_iaa_datasets(config)
    if not samples:
        raise RuntimeError("Failed to load IAA datasets")

    sample_ids = [s["sample_id"] for s in samples]
    code_count = sum(1 for s in samples if s["scenario_type"] == "code")
    text_count = sum(1 for s in samples if s["scenario_type"] == "text")
    logger.info(f"[Exp3] Dataset: {code_count} code + {text_count} text = {len(samples)} total")

    # 2. Load IAA cache (with predicted_confidences)
    baselines = _load_iaa_baseline(iaa_cache_dir, model_key, sample_ids)

    # 3. Collect Profile matching statistics
    profile_match_stats = {"project_profile": 0, "uniform_prior": 0}
    for s in samples:
        p = _resolve_sample_profile(s, profiles)
        if any(v != 0.5 for v in p.get("l2_scores", {}).values()):
            profile_match_stats["project_profile"] += 1
        else:
            profile_match_stats["uniform_prior"] += 1
    logger.info(f"[Exp3] Profile matching: {profile_match_stats}")

    # 4. Alpha parameter sweep
    logger.info(f"[Exp3] Starting alpha sweep: {alpha_values}")
    sweep_results = _sweep_alpha(samples, baselines, profiles, alpha_values, threshold)

    # 5. Find optimal alpha
    optimal_alpha = _find_optimal_alpha(sweep_results, target_metric)
    logger.info(f"[Exp3] Optimal alpha = {optimal_alpha} (by {target_metric})")

    # 6. Visualization — F1 vs alpha curve
    viz.plot_alpha_curve(
        sweep_results,
        metric_key=target_metric,
        title=f"Bayesian Calibration: {target_metric} vs $\\alpha$ ({model_key})",
        filename="exp3_alpha_curve.pdf",
        optimal_alpha=optimal_alpha,
    )

    # Additionally plot Risk F1 curve
    viz.plot_alpha_curve(
        sweep_results,
        metric_key="Risk F1",
        title=f"Bayesian Calibration: Risk F1 vs $\\alpha$ ({model_key})",
        filename="exp3_alpha_curve_risk.pdf",
        optimal_alpha=optimal_alpha,
    )

    # 7. Build comparison results (alpha=0 vs optimal alpha)
    baseline_metrics = sweep_results.get(0, sweep_results.get(0.0, {}))
    optimal_metrics = sweep_results.get(optimal_alpha, {})

    comparison = {}
    for label in ["code", "text", "overall"]:
        m_base = baseline_metrics.get(label, {})
        m_opt = optimal_metrics.get(label, {})
        if not m_base:
            continue
        delta = {}
        for k in m_base:
            if k == "n":
                continue
            delta[k] = round(m_opt.get(k, 0.0) - m_base.get(k, 0.0), 4)
        comparison[label] = {
            "n": m_base.get("n", 0),
            "baseline": {k: v for k, v in m_base.items() if k != "n"},
            "optimal": {k: v for k, v in m_opt.items() if k != "n"},
            "delta": delta,
        }

    # 8. LaTeX table
    headers = ["Scenario", "N", "Metric", f"$\\alpha$=0", f"$\\alpha$={optimal_alpha}", "$\\Delta$"]
    rows = []
    for label in ["code", "text", "overall"]:
        c = comparison.get(label)
        if not c:
            continue
        display = {"code": "Code", "text": "Issue Text", "overall": "\\textbf{Overall}"}[label]
        first = True
        for metric_key_name in c["baseline"]:
            v_base = c["baseline"][metric_key_name]
            v_opt = c["optimal"].get(metric_key_name, 0.0)
            d = c["delta"].get(metric_key_name, 0.0)
            d_str = f"+{d:.4f}" if d >= 0 else f"{d:.4f}"
            if d > 0:
                d_str = f"\\textbf{{{d_str}}}"
            row_label = display if first else ""
            row_n = str(c["n"]) if first else ""
            rows.append([row_label, row_n, metric_key_name, f"{v_base:.4f}", f"{v_opt:.4f}", d_str])
            first = False
        if label != "overall":
            rows.append(["\\midrule"] + [""] * 5)

    # Filter empty rows
    rows = [r for r in rows if any(cell.strip() for cell in r if cell)]
    latex = viz.generate_latex_table(
        headers, rows,
        caption=f"Bayesian Profile Calibration: $\\alpha$=0 vs $\\alpha^*$={optimal_alpha} ({model_key})",
        label="tab:exp3_bayesian",
    )
    viz.save_latex_table(latex, "exp3_bayesian_table.tex")

    # 9. Grouped bar chart — baseline vs optimal for overall only
    overall_base = comparison.get("overall", {}).get("baseline", {})
    overall_opt = comparison.get("overall", {}).get("optimal", {})
    if overall_base and overall_opt:
        viz.plot_grouped_bar(
            {
                f"$\\alpha$=0 (Baseline)": overall_base,
                f"$\\alpha$={optimal_alpha} (Optimal)": overall_opt,
            },
            title=f"Profile Bayesian Calibration ({model_key}, N={len(samples)})",
            filename="exp3_ablation_bar.pdf",
            xlabel="Metric", ylabel="Score",
        )

    # 10. Save results
    results = {
        "model": model_key,
        "total_samples": len(samples),
        "code_samples": code_count,
        "text_samples": text_count,
        "iaa_cache_dir": str(iaa_cache_dir),
        "alpha_values": alpha_values,
        "threshold": threshold,
        "target_metric": target_metric,
        "optimal_alpha": optimal_alpha,
        "profile_match_stats": profile_match_stats,
        "sweep_results": {str(k): v for k, v in sweep_results.items()},
        "comparison": comparison,
    }
    _save_json(results, out / "exp3_results.json")

    md = _generate_exp3_report(results)
    (out / "exp3_report.md").write_text(md, encoding="utf-8")

    logger.info("[Exp3] Experiment completed!")
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_json(data, path):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _generate_exp3_report(results: dict) -> str:
    lines = ["# Experiment 3: Profile as Bayesian Hypothesis Calibrator\n"]
    lines.append(f"**Model**: {results['model']}  ")
    lines.append(f"**Total Samples**: {results['total_samples']} "
                 f"({results['code_samples']} code + {results['text_samples']} text)  ")
    lines.append(f"**Optimal alpha**: {results['optimal_alpha']} "
                 f"(by {results['target_metric']})  ")
    lines.append(f"**Threshold**: {results['threshold']}  ")
    lines.append(f"**Profile Match**: {results['profile_match_stats']}\n")

    lines.append("## Method\n")
    lines.append("Profile is used as a Bayesian prior to calibrate LLM prediction "
                 "confidences post-hoc, without participating in hypothesis generation.\n")
    lines.append("Formula: `weighted_conf[v] = raw_conf[v] * (profile_score[v] / mu)^alpha`\n")

    lines.append("## Alpha Sweep Results\n")
    for alpha_str, splits in results.get("sweep_results", {}).items():
        overall = splits.get("overall", {})
        if overall:
            lines.append(f"- alpha={alpha_str}: Value F1={overall.get('Value F1', 0):.4f}, "
                         f"Risk F1={overall.get('Risk F1', 0):.4f}, "
                         f"Jaccard={overall.get('Jaccard', 0):.4f}")
    lines.append("")

    lines.append("## Comparison: Baseline vs Optimal\n")
    for label in ["code", "text", "overall"]:
        c = results.get("comparison", {}).get(label)
        if not c:
            continue
        title = {"code": "Code Scenarios", "text": "Issue Text Scenarios", "overall": "Overall"}[label]
        lines.append(f"### {title} (N={c['n']})\n")
        lines.append("| Metric | alpha=0 | alpha=optimal | Delta |")
        lines.append("|--------|---------|--------------|-------|")
        for k in c["baseline"]:
            v_base = c["baseline"][k]
            v_opt = c["optimal"].get(k, 0.0)
            d = c["delta"].get(k, 0.0)
            sign = "+" if d >= 0 else ""
            lines.append(f"| {k} | {v_base:.4f} | {v_opt:.4f} | {sign}{d:.4f} |")
        lines.append("")

    return "\n".join(lines)
