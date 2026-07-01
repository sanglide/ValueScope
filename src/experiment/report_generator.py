"""
Report generation module.
Produces tables in Markdown, LaTeX, and CSV formats.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict

try:
    from .evaluator import EvaluationMetrics
    from . import paths as exp_paths
except ImportError:
    from evaluator import EvaluationMetrics
    import paths as exp_paths


class ReportGenerator:
    """Generic report generator."""
    
    # Default table columns
    DEFAULT_COLUMNS = [
        ("model_name", "Model"),
        ("risk_precision", "Risk P"),
        ("risk_recall", "Risk R"),
        ("risk_f1", "Risk F1"),
        ("value_precision", "Value P"),
        ("value_recall", "Value R"),
        ("value_f1", "Value F1"),
        ("jaccard_index", "Jaccard"),
        ("value_precision_loose", "Value P*"),
        ("value_recall_loose", "Value R*"),
        ("value_f1_loose", "Value F1*"),
        ("exact_match", "EM"),
        ("cohen_kappa", "Kappa"),
    ]
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = str(exp_paths.RESULTS_DIR.relative_to(exp_paths.PROJECT_ROOT))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Resolve relative paths against the project root when used as a default
        if not self.output_dir.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            self.output_dir = project_root / self.output_dir
    
    def generate_markdown_table(
        self,
        metrics_list: list[EvaluationMetrics],
        columns: Optional[list[tuple[str, str]]] = None,
        title: str = "Experiment Results"
    ) -> str:
        """Generate a Markdown table."""
        columns = columns or self.DEFAULT_COLUMNS
        
        lines = [f"## {title}\n"]
        
        # 表头
        header = "| " + " | ".join([col[1] for col in columns]) + " |"
        separator = "|" + "|".join(["---" for _ in columns]) + "|"
        lines.append(header)
        lines.append(separator)
        
        # 数据行
        for metrics in metrics_list:
            metrics_dict = metrics.to_dict()
            row_values = []
            for col_key, _ in columns:
                value = metrics_dict.get(col_key, "N/A")
                if isinstance(value, float):
                    row_values.append(f"{value:.4f}")
                else:
                    row_values.append(str(value))
            lines.append("| " + " | ".join(row_values) + " |")
        
        return "\n".join(lines)
    
    def generate_latex_table(
        self,
        metrics_list: list[EvaluationMetrics],
        columns: Optional[list[tuple[str, str]]] = None,
        caption: str = "Evaluation Results",
        label: str = "tab:results"
    ) -> str:
        """生成LaTeX格式的表格"""
        columns = columns or self.DEFAULT_COLUMNS
        
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{" + "l" + "c" * (len(columns) - 1) + "}",
            "\\toprule"
        ]
        
        # 表头
        header = " & ".join([col[1] for col in columns]) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")
        
        # 数据行
        for metrics in metrics_list:
            metrics_dict = metrics.to_dict()
            row_values = []
            for col_key, _ in columns:
                value = metrics_dict.get(col_key, "N/A")
                if isinstance(value, float):
                    row_values.append(f"{value:.4f}")
                else:
                    row_values.append(str(value))
            lines.append(" & ".join(row_values) + " \\\\")
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)
    
    def generate_csv(
        self,
        metrics_list: list[EvaluationMetrics],
        columns: Optional[list[tuple[str, str]]] = None
    ) -> str:
        """生成CSV格式的表格"""
        columns = columns or self.DEFAULT_COLUMNS
        
        lines = []
        # 表头
        lines.append(",".join([col[1] for col in columns]))
        
        # 数据行
        for metrics in metrics_list:
            metrics_dict = metrics.to_dict()
            row_values = []
            for col_key, _ in columns:
                value = metrics_dict.get(col_key, "N/A")
                if isinstance(value, float):
                    row_values.append(f"{value:.4f}")
                else:
                    row_values.append(str(value))
            lines.append(",".join(row_values))
        
        return "\n".join(lines)
    
    def save_results(
        self,
        metrics_list: list[EvaluationMetrics],
        experiment_name: str = "experiment",
        formats: list[str] = None
    ) -> dict[str, str]:
        """保存结果到多种格式的文件"""
        formats = formats or ["markdown", "latex", "csv"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        if "markdown" in formats:
            md_content = self.generate_markdown_table(metrics_list)
            md_file = self.output_dir / f"{experiment_name}_{timestamp}.md"
            md_file.write_text(md_content, encoding='utf-8')
            saved_files["markdown"] = str(md_file)
        
        if "latex" in formats:
            latex_content = self.generate_latex_table(metrics_list)
            latex_file = self.output_dir / f"{experiment_name}_{timestamp}.tex"
            latex_file.write_text(latex_content, encoding='utf-8')
            saved_files["latex"] = str(latex_file)
        
        if "csv" in formats:
            csv_content = self.generate_csv(metrics_list)
            csv_file = self.output_dir / f"{experiment_name}_{timestamp}.csv"
            csv_file.write_text(csv_content, encoding='utf-8')
            saved_files["csv"] = str(csv_file)
        
        # 保存完整的JSON结果
        json_data = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "results": [m.to_dict() for m in metrics_list]
        }
        json_file = self.output_dir / f"{experiment_name}_{timestamp}.json"
        json_file.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding='utf-8')
        saved_files["json"] = str(json_file)
        
        return saved_files
    
    def print_summary(self, metrics_list: list[EvaluationMetrics]) -> None:
        """打印结果摘要到控制台"""
        print("\n" + "=" * 80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 80)
        
        # 使用简洁的格式打印
        header = f"{'Model':<25} {'Risk F1':>10} {'Value F1':>10} {'Jaccard':>10} {'Kappa':>10}"
        print(header)
        print("-" * 80)
        
        for metrics in metrics_list:
            row = f"{metrics.model_name:<25} {metrics.risk_f1:>10.4f} {metrics.value_f1:>10.4f} {metrics.jaccard_index:>10.4f} {metrics.cohen_kappa:>10.4f}"
            print(row)
        
        print("=" * 80 + "\n")


class DetailedReportGenerator(ReportGenerator):
    """详细报告生成器，包含更多分析信息"""
    
    def generate_detailed_markdown(
        self,
        metrics_list: list[EvaluationMetrics],
        raw_predictions: Optional[dict] = None,
        experiment_config: Optional[dict] = None
    ) -> str:
        """生成详细的Markdown报告"""
        lines = ["# Value Risk Identification Experiment Report\n"]
        lines.append(f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 实验配置
        if experiment_config:
            lines.append("## Experiment Configuration\n")
            lines.append("```yaml")
            for key, value in experiment_config.items():
                lines.append(f"{key}: {value}")
            lines.append("```\n")
        
        # 主结果表格
        lines.append("## Main Results\n")
        lines.append(self.generate_markdown_table(metrics_list))
        lines.append("\n")
        
        # 风险检测结果表格
        lines.append("## Risk Detection Results\n")
        risk_columns = [
            ("model_name", "Model"),
            ("risk_precision", "Precision"),
            ("risk_recall", "Recall"),
            ("risk_f1", "F1"),
            ("risk_accuracy", "Accuracy"),
        ]
        lines.append(self.generate_markdown_table(metrics_list, columns=risk_columns, title=""))
        lines.append("\n")
        
        # 价值识别结果表格
        lines.append("## Value Identification Results\n")
        value_columns = [
            ("model_name", "Model"),
            ("value_precision", "Precision"),
            ("value_recall", "Recall"),
            ("value_f1", "F1"),
            ("jaccard_index", "Jaccard"),
            ("exact_match", "Exact Match"),
        ]
        lines.append(self.generate_markdown_table(metrics_list, columns=value_columns, title=""))
        lines.append("\n")
        
        # 统计信息
        lines.append("## Statistics\n")
        for metrics in metrics_list:
            lines.append(f"- **{metrics.model_name}**: {metrics.valid_predictions}/{metrics.total_samples} valid predictions")
        
        return "\n".join(lines)
