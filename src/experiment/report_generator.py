"""
Report Generation Module
Generate tables for experiment results (supports Markdown, LaTeX, CSV formats)
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict

try:
    from .evaluator import EvaluationMetrics
except ImportError:
    from evaluator import EvaluationMetrics


class ReportGenerator:
    """Report Generator"""
    
    # Default table column configuration
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
    
    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_markdown_table(
        self,
        metrics_list: list[EvaluationMetrics],
        columns: Optional[list[tuple[str, str]]] = None,
        title: str = "Experiment Results"
    ) -> str:
        """Generate a Markdown format table"""
        columns = columns or self.DEFAULT_COLUMNS
        
        lines = [f"## {title}\n"]
        
        # Header
        header = "| " + " | ".join([col[1] for col in columns]) + " |"
        separator = "|" + "|".join(["---" for _ in columns]) + "|"
        lines.append(header)
        lines.append(separator)
        
        # Data rows
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
        """Generate a LaTeX format table"""
        columns = columns or self.DEFAULT_COLUMNS
        
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{" + "l" + "c" * (len(columns) - 1) + "}",
            "\\toprule"
        ]
        
        # Header
        header = " & ".join([col[1] for col in columns]) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")
        
        # Data rows
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
        """Generate a CSV format table"""
        columns = columns or self.DEFAULT_COLUMNS
        
        lines = []
        # Header
        lines.append(",".join([col[1] for col in columns]))
        
        # Data rows
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
        """Save results to files in multiple formats"""
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
        
        # Save complete JSON results
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
        """Print result summary to console"""
        print("\n" + "=" * 80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 80)
        
        # Print in concise format
        header = f"{'Model':<25} {'Risk F1':>10} {'Value F1':>10} {'Jaccard':>10} {'Kappa':>10}"
        print(header)
        print("-" * 80)
        
        for metrics in metrics_list:
            row = f"{metrics.model_name:<25} {metrics.risk_f1:>10.4f} {metrics.value_f1:>10.4f} {metrics.jaccard_index:>10.4f} {metrics.cohen_kappa:>10.4f}"
            print(row)
        
        print("=" * 80 + "\n")


class DetailedReportGenerator(ReportGenerator):
    """Detailed report generator with additional analysis information"""
    
    def generate_detailed_markdown(
        self,
        metrics_list: list[EvaluationMetrics],
        raw_predictions: Optional[dict] = None,
        experiment_config: Optional[dict] = None
    ) -> str:
        """Generate a detailed Markdown report"""
        lines = ["# Value Risk Identification Experiment Report\n"]
        lines.append(f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Experiment configuration
        if experiment_config:
            lines.append("## Experiment Configuration\n")
            lines.append("```yaml")
            for key, value in experiment_config.items():
                lines.append(f"{key}: {value}")
            lines.append("```\n")
        
        # Main results table
        lines.append("## Main Results\n")
        lines.append(self.generate_markdown_table(metrics_list))
        lines.append("\n")
        
        # Risk detection results table
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
        
        # Value identification results table
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
        
        # Statistics
        lines.append("## Statistics\n")
        for metrics in metrics_list:
            lines.append(f"- **{metrics.model_name}**: {metrics.valid_predictions}/{metrics.total_samples} valid predictions")
        
        return "\n".join(lines)
