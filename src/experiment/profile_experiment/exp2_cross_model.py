#!/usr/bin/env python
"""
Experiment 2: Cross-Model Consistency.
Validates whether different LLMs produce consistent ValueProfiles for the same project.
"""

import json
import logging
import sys
from pathlib import Path

# Project paths
project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from experiment.profile_experiment.profile_generator import ProfileGenerator, ALL_VALUE_IDS
from experiment.profile_experiment.profile_metrics import (
    pairwise_agreement_matrix,
    compute_dimension_stats,
    compute_overall_consistency,
)
from experiment.profile_experiment.profile_visualizer import ProfileVisualizer
from experiment import paths as exp_paths

logger = logging.getLogger(__name__)


def run_exp2(
    generator: ProfileGenerator,
    viz: ProfileVisualizer,
    repo_configs: list[dict],
    model_keys: list[str],
    output_dir: str = None,
    force_api: bool = False,
) -> dict:
    """Run Experiment 2: Cross-Model Consistency."""
    if output_dir is None:
        output_dir = str(exp_paths.PROFILE_DIR.relative_to(exp_paths.PROJECT_ROOT))
    out = Path(output_dir)
    if not out.is_absolute():
        out = project_root / out
    out.mkdir(parents=True, exist_ok=True)
    results = {"repos": {}}

    for repo_cfg in repo_configs:
        name = repo_cfg["name"]
        path = repo_cfg["path"]
        sources = repo_cfg.get("doc_sources", [])

        logger.info(f"[Exp2] 处理项目: {name}")
        docs = generator.collect_documents(path, sources)

        # 用所有模型生成 Profile
        profiles = generator.generate_profiles_multi_model(
            name, docs, model_keys, force_api=force_api
        )

        # 保存每个 Profile
        for mk, p in profiles.items():
            _save_json(p, out / f"exp2_{name}_{mk}_profile.json")

        # --- 指标计算 ---
        # 1. Pairwise Spearman
        spearman_matrix = pairwise_agreement_matrix(profiles, metric="spearman")
        # 2. Pairwise Cosine
        cosine_matrix = pairwise_agreement_matrix(profiles, metric="cosine")
        # 3. Dimension stats
        dim_stats = compute_dimension_stats(list(profiles.values()))
        # 4. Overall consistency
        consistency = compute_overall_consistency(profiles)

        # --- 可视化 ---
        # 热力图 - Spearman
        viz.plot_heatmap(
            spearman_matrix,
            title=f"Pairwise Spearman ρ — {name}",
            filename=f"exp2_{name}_heatmap_spearman.pdf",
            vmin=0.0, vmax=1.0, cmap="YlOrRd",
        )
        # 热力图 - Cosine
        viz.plot_heatmap(
            cosine_matrix,
            title=f"Pairwise Cosine Similarity — {name}",
            filename=f"exp2_{name}_heatmap_cosine.pdf",
            vmin=0.5, vmax=1.0, cmap="YlGn",
        )
        # 箱线图 - 每个维度跨模型的分数分布
        box_data = {vid: dim_stats[vid]["values"] for vid in ALL_VALUE_IDS}
        viz.plot_box_distribution(
            box_data,
            title=f"Cross-Model Score Distribution — {name}",
            filename=f"exp2_{name}_boxplot.pdf",
        )

        # --- LaTeX 表格 ---
        headers = ["Metric", "Value"]
        std_spearman = consistency.get('std_spearman', 0.0)
        std_cosine = consistency.get('std_cosine', 0.0)
        n_pairs = consistency.get('n_pairs', 0)
        table_rows = [
            ["Kendall's W", f"{consistency['kendall_w']:.3f}"],
            ["Avg. Spearman ρ", f"{consistency['avg_spearman']:.3f} ± {std_spearman:.3f}"],
            ["Avg. Cosine Sim.", f"{consistency['avg_cosine']:.3f} ± {std_cosine:.3f}"],
            ["Num. Pairs", str(n_pairs)],
        ]
        latex = viz.generate_latex_table(
            headers, table_rows,
            caption=f"Cross-Model Consistency Metrics — {name}",
            label=f"tab:exp2_consistency_{name}",
        )
        viz.save_latex_table(latex, f"exp2_{name}_consistency_table.tex")

        results["repos"][name] = {
            "profiles": {k: _strip_raw(v) for k, v in profiles.items()},
            "consistency": consistency,
            "dimension_stats": {k: {kk: vv for kk, vv in v.items() if kk != "values"}
                                for k, v in dim_stats.items()},
        }

    # 汇总报告
    _save_json(results, out / "exp2_results.json")
    md = _generate_exp2_report(results)
    (out / "exp2_report.md").write_text(md, encoding="utf-8")

    logger.info("[Exp2] 实验完成！")
    return results


def _strip_raw(p: dict) -> dict:
    return {k: v for k, v in p.items() if k != "raw_response"}


def _save_json(data, path):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _generate_exp2_report(results):
    lines = ["# Experiment 2: Cross-Model Consistency\n"]

    for name, repo_res in results.get("repos", {}).items():
        cons = repo_res["consistency"]
        std_spearman = cons.get('std_spearman', 0.0)
        std_cosine = cons.get('std_cosine', 0.0)
        n_pairs = cons.get('n_pairs', 0)
        lines.append(f"## {name}\n")
        lines.append(f"- **Kendall's W**: {cons['kendall_w']:.3f}")
        lines.append(f"- **Avg. Spearman ρ**: {cons['avg_spearman']:.3f} ± {std_spearman:.3f}")
        lines.append(f"- **Avg. Cosine Sim.**: {cons['avg_cosine']:.3f} ± {std_cosine:.3f}")
        lines.append(f"- **Num. Model Pairs**: {n_pairs}")
        lines.append("")

        # 维度统计 - 高方差维度
        dim_stats = repo_res.get("dimension_stats", {})
        high_cv = [(k, v) for k, v in dim_stats.items() if v.get("cv", 0) > 0.15]
        if high_cv:
            lines.append("### High-Variance Dimensions (CV > 0.15)")
            for k, v in sorted(high_cv, key=lambda x: x[1]["cv"], reverse=True):
                lines.append(f"  - {k}: CV={v['cv']:.3f}, mean={v['mean']:.2f}, std={v['std']:.3f}")
            lines.append("")

    lines.append("## Interpretation\n")
    lines.append("Kendall's W above 0.7 indicates strong agreement across LLMs on the ")
    lines.append("relative ranking of value dimensions. High cosine similarity (>0.9) suggests ")
    lines.append("that different LLMs produce structurally similar value profiles.")

    return "\n".join(lines)
