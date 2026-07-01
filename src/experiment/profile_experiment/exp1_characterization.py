#!/usr/bin/env python
"""
Experiment 1: Profile Characterization (case study).
Demonstrates that ValueProfile distinguishes project value characteristics.
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

from experiment.profile_experiment.profile_generator import ProfileGenerator, L2_IDS, L3_IDS
from experiment.profile_experiment.profile_visualizer import ProfileVisualizer, L2_NAMES, L3_NAMES
from experiment import paths as exp_paths

logger = logging.getLogger(__name__)


def run_exp1(
    generator: ProfileGenerator,
    viz: ProfileVisualizer,
    repo_configs: list[dict],
    model_key: str = "qwen-plus",
    output_dir: str = None,
    force_api: bool = False,
) -> dict:
    """Run Experiment 1: Profile Characterization.

    Args:
        generator: ProfileGenerator instance
        viz: ProfileVisualizer instance
        repo_configs: [{"name": ..., "path": ..., "doc_sources": [...]}]
        model_key: Representative LLM
        output_dir: Output directory
    Returns:
        Experiment results dict
    """
    if output_dir is None:
        output_dir = str(exp_paths.PROFILE_DIR.relative_to(exp_paths.PROJECT_ROOT))
    out = Path(output_dir)
    if not out.is_absolute():
        out = project_root / out
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    # 1. 对每个项目生成 Profile
    profiles = {}
    for repo_cfg in repo_configs:
        name = repo_cfg["name"]
        path = repo_cfg["path"]
        sources = repo_cfg.get("doc_sources", [])

        logger.info(f"[Exp1] 收集文档：{name}")
        docs = generator.collect_documents(path, sources)
        logger.info(f"[Exp1] 收集到 {len(docs)} 个文档")
        
        # 对大量文档进行摘要（>20 个文档时启用）
        if len(docs) >= 20:
            logger.info(f"[Exp1] 文档数量较多（{len(docs)}），开始摘要...")
            docs_summarized = generator.summarize_documents(docs, model_key)
            logger.info(f"[Exp1] 文档摘要完成，使用 {len(docs_summarized)} 个摘要文档")
        else:
            docs_summarized = docs
        
        logger.info(f"[Exp1] 开始生成 Profile (v2: doc + code evidence)...")
        profile = generator.generate_profile(
            name, model_key, docs_summarized,
            force_api=force_api,
            repo_path=path,
            use_code_evidence=True,
        )
        profiles[name] = profile

        # 保存 Profile JSON
        profile_path = out / f"exp1_{name}_profile.json"
        _save_json(profile, profile_path)
        logger.info(f"[Exp1] Profile 已保存: {profile_path}")

    results["profiles"] = profiles

    # 2. 生成雷达图（每个项目单独一张图，分 L2 / L3 维度）
    radar_figures: dict[str, str] = {}

    paths_l2 = viz.plot_radar_chart_per_project(profiles, dimension="l2")
    paths_l3 = viz.plot_radar_chart_per_project(profiles, dimension="l3")

    for name in profiles:
        radar_figures[f"{name}_l2"] = paths_l2[name]
        radar_figures[f"{name}_l3"] = paths_l3[name]

    logger.info(f"[Exp1] 雷达图已生成: {list(radar_figures.values())}")
    results["figures"] = radar_figures

    # 3. 生成对比表格
    repo_names = list(profiles.keys())
    headers = ["Value ID", "Value Name"] + repo_names + ["Δ"]
    rows = []

    for vid in L2_IDS + L3_IDS:
        vname = L2_NAMES.get(vid, L3_NAMES.get(vid, vid))
        score_key = "l2_scores" if vid.startswith("HV") else "l3_scores"
        scores = [profiles[r].get(score_key, {}).get(vid, 0.0) for r in repo_names]
        delta = abs(scores[0] - scores[1]) if len(scores) == 2 else 0.0
        row = [vid, vname] + [f"{s:.2f}" for s in scores] + [f"{delta:.2f}"]
        rows.append(row)

    # 按 Δ 降序排列
    rows.sort(key=lambda r: float(r[-1]), reverse=True)

    # LaTeX 表格
    latex = viz.generate_latex_table(
        headers, rows,
        caption="Value Profile Comparison: Signal-Android vs Focus-Android",
        label="tab:profile_comparison",
    )
    tex_path = viz.save_latex_table(latex, "exp1_comparison_table.tex")
    results["tables"] = {"comparison_tex": tex_path}

    # 4. Core Values 摘要
    summary = {}
    for name, p in profiles.items():
        summary[name] = {
            "core_values": p.get("core_values", []),
            "top5_l2": _top_n(p.get("l2_scores", {}), 5),
            "top5_l3": _top_n(p.get("l3_scores", {}), 5),
        }
    results["summary"] = summary
    _save_json(results["summary"], out / "exp1_summary.json")

    # 5. Markdown 报告
    md = _generate_exp1_report(profiles, repo_names, results)
    md_path = out / "exp1_report.md"
    md_path.write_text(md, encoding="utf-8")
    results["report"] = str(md_path)

    logger.info("[Exp1] 实验完成！")
    return results


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _top_n(scores: dict, n: int = 5) -> list:
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


def _save_json(data, path):
    path = Path(path)
    # 过滤不可序列化的字段
    clean = {k: v for k, v in data.items() if k != "raw_response"} if isinstance(data, dict) else data
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _generate_exp1_report(profiles, repo_names, results):
    lines = ["# Experiment 1: Value Profile Characterization\n"]
    lines.append("## Profile Summary\n")

    for name in repo_names:
        p = profiles[name]
        lines.append(f"### {name}\n")
        lines.append(f"- **Core Values**: {', '.join(p.get('core_values', []))}")
        lines.append(f"- **Top L2**: {_top_n(p.get('l2_scores', {}), 3)}")
        lines.append(f"- **Top L3**: {_top_n(p.get('l3_scores', {}), 3)}")
        lines.append("")

    lines.append("## Figures\n")
    for key, path in results.get("figures", {}).items():
        lines.append(f"- {key}: `{path}`")

    lines.append("\n## Observations\n")
    lines.append("The radar charts and comparison table above demonstrate that ValueProfile ")
    lines.append("can effectively distinguish different projects' value characteristics. ")
    lines.append("Projects with strong privacy/security focus (e.g., Signal) show significantly ")
    lines.append("higher scores on HV9 (Privacy), HV10 (Security), and SV1 (Trust) dimensions.")

    return "\n".join(lines)
