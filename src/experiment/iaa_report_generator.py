"""
IAA report generation module.
Produces Markdown / LaTeX reports for inter-annotator agreement experiments.
"""

from pathlib import Path
from typing import Optional

try:
    from .iaa_data_structures import (
        ALL_VALUE_IDS,
        VALUE_NAMES,
        IAAExperimentResults,
    )
    from . import paths as exp_paths
except ImportError:
    from iaa_data_structures import (
        ALL_VALUE_IDS,
        VALUE_NAMES,
        IAAExperimentResults,
    )
    import paths as exp_paths


def _interpret_kappa(k: float) -> str:
    """Landis & Koch interpretation scale."""
    if k < 0:
        return "Poor"
    elif k < 0.20:
        return "Slight"
    elif k < 0.40:
        return "Fair"
    elif k < 0.60:
        return "Moderate"
    elif k < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


class IAAReportGenerator:
    """IAA experiment report generator."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = str(exp_paths.IAA_DIR.relative_to(exp_paths.PROJECT_ROOT))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.output_dir.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            self.output_dir = project_root / self.output_dir

    # ------------------------------------------------------------------
    # Markdown reports
    # ------------------------------------------------------------------
    def generate_pairwise_matrix_md(self, results: IAAExperimentResults) -> str:
        """Generate a pairwise agreement matrix (upper=triangular κ, lower=%agree)."""
        ids = results.annotator_ids
        n = len(ids)
        if n == 0:
            return ""

        # 建立快速查找表
        lookup = {}
        for pr in results.pairwise.values():
            a, b = pr.annotator_a, pr.annotator_b
            lookup[(a, b)] = pr
            lookup[(b, a)] = pr

        lines = [
            "## Pairwise Agreement Matrix (Risk Detection)\n",
            "Upper triangle: Cohen's κ | Lower triangle: Percent Agreement\n",
        ]

        # 表头
        header = "| | " + " | ".join(ids) + " |"
        sep = "|---|" + "|".join(["---"] * n) + "|"
        lines.append(header)
        lines.append(sep)

        for i, row_id in enumerate(ids):
            cells = [row_id]
            for j, col_id in enumerate(ids):
                if i == j:
                    cells.append("—")
                elif i < j:
                    # 上三角: Cohen's κ
                    pr = lookup.get((row_id, col_id))
                    cells.append(f"{pr.dim1_cohen_kappa:.3f}" if pr else "N/A")
                else:
                    # 下三角: Percent Agreement
                    pr = lookup.get((row_id, col_id))
                    cells.append(f"{pr.dim1_percent_agreement:.1%}" if pr else "N/A")
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def generate_overall_table_md(
        self, results_by_scenario: dict[str, IAAExperimentResults]
    ) -> str:
        """生成总体 IAA 统计表"""
        scenarios = [s for s in ["code", "text", "overall"] if s in results_by_scenario]
        if not scenarios:
            return ""

        lines = [
            "## Overall IAA Statistics\n",
            "| Dimension | Metric | " + " | ".join(scenarios) + " |",
            "|---|---| " + " | ".join(["---"] * len(scenarios)) + " |",
        ]

        # 维度1 指标
        dim1_metrics = [
            ("dim1_fleiss_kappa", "Fleiss κ"),
            ("dim1_krippendorff_alpha", "Krippendorff α"),
            ("dim1_avg_pairwise_kappa", "Avg Pairwise κ"),
            ("dim1_avg_pairwise_pabak", "Avg Pairwise PABAK"),
            ("dim1_avg_pairwise_ac1", "Avg Pairwise AC1"),
        ]
        for attr, label in dim1_metrics:
            cells = ["Risk Detection", label]
            for s in scenarios:
                m = results_by_scenario[s].multi_annotator
                val = getattr(m, attr, 0.0) if m else 0.0
                cells.append(f"{val:.4f}")
            lines.append("| " + " | ".join(cells) + " |")

        # 维度2 指标
        dim2_metrics = [
            ("dim2_macro_avg_value_kappa", "Macro Fleiss κ"),
            ("dim2_avg_pairwise_jaccard", "Avg Pairwise Jaccard"),
            ("dim2_avg_pairwise_f1", "Avg Pairwise F1"),
        ]
        for attr, label in dim2_metrics:
            cells = ["Value ID", label]
            for s in scenarios:
                m = results_by_scenario[s].multi_annotator
                val = getattr(m, attr, 0.0) if m else 0.0
                cells.append(f"{val:.4f}")
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def generate_per_value_table_md(self, results: IAAExperimentResults) -> str:
        """生成 Per-Value Fleiss' Kappa 表"""
        if not results.multi_annotator:
            return ""

        kappas = results.multi_annotator.dim2_per_value_fleiss_kappa
        lines = [
            "## Per-Value Fleiss' Kappa\n",
            "| Value ID | Name | Fleiss κ | Interpretation |",
            "|---|---|---|---|",
        ]

        for vid in ALL_VALUE_IDS:
            k = kappas.get(vid, 0.0)
            name = VALUE_NAMES.get(vid, "")
            interp = _interpret_kappa(k)
            lines.append(f"| {vid} | {name} | {k:.4f} | {interp} |")

        macro = results.multi_annotator.dim2_macro_avg_value_kappa
        lines.append(f"| **Macro Avg** | | **{macro:.4f}** | **{_interpret_kappa(macro)}** |")

        return "\n".join(lines)

    def generate_human_vs_llm_table_md(self, results: IAAExperimentResults) -> str:
        """生成 Human vs 各 LLM 专项对比表"""
        # 找到 Human 相关的 pair
        human_pairs = {}
        for key, pr in results.pairwise.items():
            if pr.annotator_a == "Human":
                human_pairs[pr.annotator_b] = pr
            elif pr.annotator_b == "Human":
                human_pairs[pr.annotator_a] = pr

        if not human_pairs:
            return ""

        llms = sorted(human_pairs.keys())
        lines = [
            "## Human vs. LLM Agreement\n",
            "| Metric | " + " | ".join([f"vs {llm}" for llm in llms]) + " |",
            "|---| " + " | ".join(["---"] * len(llms)) + " |",
        ]

        # Risk κ
        row = ["Risk κ"]
        for llm in llms:
            row.append(f"{human_pairs[llm].dim1_cohen_kappa:.4f}")
        lines.append("| " + " | ".join(row) + " |")

        # Risk PABAK
        row = ["Risk PABAK"]
        for llm in llms:
            row.append(f"{human_pairs[llm].dim1_pabak:.4f}")
        lines.append("| " + " | ".join(row) + " |")

        # Risk AC1
        row = ["Risk AC1"]
        for llm in llms:
            row.append(f"{human_pairs[llm].dim1_gwet_ac1:.4f}")
        lines.append("| " + " | ".join(row) + " |")

        # Risk %agree
        row = ["Risk %Agree"]
        for llm in llms:
            row.append(f"{human_pairs[llm].dim1_percent_agreement:.1%}")
        lines.append("| " + " | ".join(row) + " |")

        # Value Jaccard
        row = ["Value Jaccard"]
        for llm in llms:
            row.append(f"{human_pairs[llm].dim2_jaccard:.4f}")
        lines.append("| " + " | ".join(row) + " |")

        # Value F1
        row = ["Value F1"]
        for llm in llms:
            row.append(f"{human_pairs[llm].dim2_symmetric_f1:.4f}")
        lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LaTeX 报告
    # ------------------------------------------------------------------
    def generate_latex_tables(
        self, results_by_scenario: dict[str, IAAExperimentResults]
    ) -> str:
        """生成 LaTeX booktabs 格式表格"""
        sections = []

        # Overall 表
        scenarios = [s for s in ["code", "text", "overall"] if s in results_by_scenario]
        if scenarios:
            cols = "l l " + " ".join(["c"] * len(scenarios))
            lines = [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Inter-Annotator Agreement Statistics}",
                "\\label{tab:iaa-overall}",
                f"\\begin{{tabular}}{{{cols}}}",
                "\\toprule",
                "Dimension & Metric & " + " & ".join(s.capitalize() for s in scenarios) + " \\\\",
                "\\midrule",
            ]

            metrics_map = [
                ("Risk Detection", [
                    ("dim1_fleiss_kappa", "Fleiss $\\kappa$"),
                    ("dim1_krippendorff_alpha", "Krippendorff $\\alpha$"),
                    ("dim1_avg_pairwise_kappa", "Avg Pairwise $\\kappa$"),
                    ("dim1_avg_pairwise_pabak", "Avg Pairwise PABAK"),
                    ("dim1_avg_pairwise_ac1", "Avg Pairwise AC1"),
                ]),
                ("Value ID", [
                    ("dim2_macro_avg_value_kappa", "Macro Fleiss $\\kappa$"),
                    ("dim2_avg_pairwise_jaccard", "Avg Pairwise Jaccard"),
                    ("dim2_avg_pairwise_f1", "Avg Pairwise F1"),
                ]),
            ]

            for dim_name, metrics in metrics_map:
                for idx, (attr, label) in enumerate(metrics):
                    dim_cell = dim_name if idx == 0 else ""
                    vals = []
                    for s in scenarios:
                        m = results_by_scenario[s].multi_annotator
                        v = getattr(m, attr, 0.0) if m else 0.0
                        vals.append(f"{v:.3f}")
                    lines.append(f"{dim_cell} & {label} & " + " & ".join(vals) + " \\\\")
                if dim_name != "Value ID":
                    lines.append("\\midrule")

            lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ])
            sections.append("\n".join(lines))

        # Per-value κ 表
        overall = results_by_scenario.get("overall")
        if overall and overall.multi_annotator:
            kappas = overall.multi_annotator.dim2_per_value_fleiss_kappa
            lines = [
                "",
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Per-Value Fleiss' $\\kappa$}",
                "\\label{tab:per-value-kappa}",
                "\\begin{tabular}{l l c l}",
                "\\toprule",
                "Value ID & Name & Fleiss $\\kappa$ & Interpretation \\\\",
                "\\midrule",
            ]
            for vid in ALL_VALUE_IDS:
                k = kappas.get(vid, 0.0)
                name = VALUE_NAMES.get(vid, "")
                interp = _interpret_kappa(k)
                lines.append(f"{vid} & {name} & {k:.3f} & {interp} \\\\")
            lines.append("\\midrule")
            macro = overall.multi_annotator.dim2_macro_avg_value_kappa
            lines.append(
                f"\\textbf{{Macro Avg}} & & \\textbf{{{macro:.3f}}} & "
                f"\\textbf{{{_interpret_kappa(macro)}}} \\\\"
            )
            lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ])
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------
    def save_all(
        self,
        results_by_scenario: dict[str, IAAExperimentResults],
        output_dir: Optional[str] = None,
    ) -> dict[str, str]:
        """保存所有报告文件

        Returns:
            {报告名: 文件路径}
        """
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        saved = {}

        full_report_parts = ["# IAA Experiment Report\n"]

        # Pairwise matrix (one per scenario)
        for scenario, res in results_by_scenario.items():
            md = self.generate_pairwise_matrix_md(res)
            if md:
                fname = f"iaa_pairwise_matrix_{scenario}.md"
                fpath = out / fname
                fpath.write_text(md, encoding='utf-8')
                saved[fname] = str(fpath)
                full_report_parts.append(f"\n### Scenario: {scenario}\n")
                full_report_parts.append(md)

        # Overall table
        md = self.generate_overall_table_md(results_by_scenario)
        if md:
            fpath = out / "iaa_overall_table.md"
            fpath.write_text(md, encoding='utf-8')
            saved["iaa_overall_table.md"] = str(fpath)
            full_report_parts.append(f"\n{md}")

        # Per-value kappa (用 overall 结果)
        overall = results_by_scenario.get("overall")
        if overall:
            md = self.generate_per_value_table_md(overall)
            if md:
                fpath = out / "iaa_per_value_kappa.md"
                fpath.write_text(md, encoding='utf-8')
                saved["iaa_per_value_kappa.md"] = str(fpath)
                full_report_parts.append(f"\n{md}")

            md = self.generate_human_vs_llm_table_md(overall)
            if md:
                fpath = out / "iaa_human_vs_llm.md"
                fpath.write_text(md, encoding='utf-8')
                saved["iaa_human_vs_llm.md"] = str(fpath)
                full_report_parts.append(f"\n{md}")

        # Full report
        full_md = "\n".join(full_report_parts)
        fpath = out / "iaa_full_report.md"
        fpath.write_text(full_md, encoding='utf-8')
        saved["iaa_full_report.md"] = str(fpath)

        # LaTeX
        latex = self.generate_latex_tables(results_by_scenario)
        if latex:
            fpath = out / "iaa_tables.tex"
            fpath.write_text(latex, encoding='utf-8')
            saved["iaa_tables.tex"] = str(fpath)

        return saved
