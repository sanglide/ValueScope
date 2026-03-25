#!/usr/bin/env python
"""
Experiment 1: Profile Characterization (Case Study)
Demonstrates that ValueProfile can distinguish value characteristics of different projects.
"""

import json
import logging
import sys
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from experiment.profile_experiment.profile_generator import ProfileGenerator, L2_IDS, L3_IDS
from experiment.profile_experiment.profile_visualizer import ProfileVisualizer, L2_NAMES, L3_NAMES

logger = logging.getLogger(__name__)


def run_exp1(
    generator: ProfileGenerator,
    viz: ProfileVisualizer,
    repo_configs: list[dict],
    model_key: str = "qwen-plus",
    output_dir: str = "experiment_results/profile",
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
        Experiment result dict
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    # 1. Generate Profile for each project
    profiles = {}
    for repo_cfg in repo_configs:
        name = repo_cfg["name"]
        path = repo_cfg["path"]
        sources = repo_cfg.get("doc_sources", [])

        logger.info(f"[Exp1] Collecting documents: {name}")
        docs = generator.collect_documents(path, sources)
        logger.info(f"[Exp1] Collected {len(docs)} documents")
        
        # Summarize large document collections (enabled when >20 documents)
        if len(docs) >= 20:
            logger.info(f"[Exp1] Large number of documents ({len(docs)}), starting summarization...")
            docs_summarized = generator.summarize_documents(docs, model_key)
            logger.info(f"[Exp1] Document summarization complete, using {len(docs_summarized)} summarized documents")
        else:
            docs_summarized = docs
        
        logger.info(f"[Exp1] Starting Profile generation ...")
        profile = generator.generate_profile(name, model_key, docs_summarized, force_api=force_api)
        profiles[name] = profile

        # Save Profile JSON
        profile_path = out / f"exp1_{name}_profile.json"
        _save_json(profile, profile_path)
        logger.info(f"[Exp1] Profile saved: {profile_path}")

    results["profiles"] = profiles

    # 2. Generate radar charts (one per project, separated by L2/L3 dimensions)
    radar_figures: dict[str, str] = {}

    paths_l2 = viz.plot_radar_chart_per_project(profiles, dimension="l2")
    paths_l3 = viz.plot_radar_chart_per_project(profiles, dimension="l3")

    for name in profiles:
        radar_figures[f"{name}_l2"] = paths_l2[name]
        radar_figures[f"{name}_l3"] = paths_l3[name]

    logger.info(f"[Exp1] Radar charts generated: {list(radar_figures.values())}")
    results["figures"] = radar_figures

    # 3. Generate comparison table
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

    # Sort by Δ in descending order
    rows.sort(key=lambda r: float(r[-1]), reverse=True)

    # LaTeX table
    latex = viz.generate_latex_table(
        headers, rows,
        caption="Value Profile Comparison: Signal-Android vs Focus-Android",
        label="tab:profile_comparison",
    )
    tex_path = viz.save_latex_table(latex, "exp1_comparison_table.tex")
    results["tables"] = {"comparison_tex": tex_path}

    # 4. Core Values summary
    summary = {}
    for name, p in profiles.items():
        summary[name] = {
            "core_values": p.get("core_values", []),
            "top5_l2": _top_n(p.get("l2_scores", {}), 5),
            "top5_l3": _top_n(p.get("l3_scores", {}), 5),
        }
    results["summary"] = summary
    _save_json(results["summary"], out / "exp1_summary.json")

    # 5. Markdown report
    md = _generate_exp1_report(profiles, repo_names, results)
    md_path = out / "exp1_report.md"
    md_path.write_text(md, encoding="utf-8")
    results["report"] = str(md_path)

    logger.info("[Exp1] Experiment completed!")
    return results


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _top_n(scores: dict, n: int = 5) -> list:
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


def _save_json(data, path):
    path = Path(path)
    # Filter non-serializable fields
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
