#!/usr/bin/env python
"""
Publication-quality visualization module.
All charts are output following IEEE/ACM paper standards (300 DPI, serif font).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Global Style
# ---------------------------------------------------------------------------

def setup_style():
    """Set up publication-quality global style."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (7, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

setup_style()

# Color scheme
COLORS = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#795548",  # brown
    "#607D8B",  # blue-grey
    "#E91E63",  # pink
    "#CDDC39",  # lime
]

# L2 / L3 short name mapping
L2_NAMES = {
    "HV1": "Conformity", "HV2": "Pleasure", "HV3": "Dignity",
    "HV4": "Inclusiveness", "HV5": "Belonging", "HV6": "Freedom",
    "HV7": "Independence", "HV8": "Wealth", "HV9": "Privacy",
    "HV10": "Security",
}
L3_NAMES = {
    "SV1": "Trust", "SV2": "Correctness", "SV3": "Compatibility",
    "SV4": "Portability", "SV5": "Reliability", "SV6": "Efficiency",
    "SV7": "Energy Pres.", "SV8": "Usability", "SV9": "Accessibility",
    "SV10": "Longevity",
}


class ProfileVisualizer:
    """Publication-quality ValueProfile visualization."""

    def __init__(self, output_dir: str = "experiment_results/profile"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Radar Chart
    # ------------------------------------------------------------------

    def plot_radar_chart(
        self,
        profiles: dict[str, dict],
        dimension: str = "l2",
        title: str = "",
        filename: str = "radar.pdf",
    ) -> str:
        """Plot a radar chart comparing multiple profiles on L2/L3 dimensions.

        Args:
            profiles: {label: profile_dict}
            dimension: "l2" or "l3"
        """
        if dimension == "l2":
            ids = list(L2_NAMES.keys())
            names = [L2_NAMES[v] for v in ids]
            score_key = "l2_scores"
        else:
            ids = list(L3_NAMES.keys())
            names = [L3_NAMES[v] for v in ids]
            score_key = "l3_scores"

        n = len(ids)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        for idx, (label, profile) in enumerate(profiles.items()):
            scores = [profile.get(score_key, {}).get(v, 0.0) for v in ids]
            scores += scores[:1]
            color = COLORS[idx % len(COLORS)]
            ax.plot(angles, scores, "o-", linewidth=2, label=label, color=color, markersize=4)
            ax.fill(angles, scores, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(names, fontsize=20)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
        ax.set_title(title or f"Value Profile Comparison ({dimension.upper()})", pad=20, fontsize=20,fontweight='bold')
        # ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        out_path = str(self.output_dir / filename)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_radar_chart_per_project(
        self,
        profiles: dict[str, dict],
        dimension: str = "l2",
    ) -> dict[str, str]:
        """Plot a separate radar chart for each project, returning {project_name: saved_path}.

        Args:
            profiles: {project_name: profile_dict}
            dimension: "l2" or "l3"
        """
        paths = {}
        for name, profile in profiles.items():
            safe_name = name.replace("/", "_").replace(" ", "_")
            filename = f"exp1_{safe_name}_radar_{dimension}.pdf"
            path = self.plot_radar_chart(
                {name: profile},
                dimension=dimension,
                title=f"{name} — Value Profile ({dimension.upper()})",
                filename=filename,
            )
            paths[name] = path
        return paths

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------

    def plot_heatmap(
        self,
        matrix_data: dict,
        title: str = "",
        filename: str = "heatmap.pdf",
        vmin: float = 0.0,
        vmax: float = 1.0,
        cmap: str = "YlOrRd",
        fmt: str = ".2f",
    ) -> str:
        """Plot a heatmap with annotated values.

        Args:
            matrix_data: {"keys": [...], "matrix": [[...]]}
        """
        keys = matrix_data["keys"]
        matrix = np.array(matrix_data["matrix"])
        n = len(keys)

        fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)))
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(range(n))
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(keys, fontsize=9)

        # Annotate values
        for i in range(n):
            for j in range(n):
                text_color = "white" if matrix[i, j] > (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{matrix[i, j]:{fmt}}", ha="center", va="center",
                        color=text_color, fontsize=8)

        ax.set_title(title, fontsize=13, pad=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        out_path = str(self.output_dir / filename)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # Box Plot
    # ------------------------------------------------------------------

    def plot_box_distribution(
        self,
        data: dict[str, list[float]],
        title: str = "",
        filename: str = "boxplot.pdf",
        xlabel: str = "Value Dimension",
        ylabel: str = "Score",
    ) -> str:
        """Box plot for each dimension.

        Args:
            data: {value_id: [score_from_model_1, score_from_model_2, ...]}
        """
        labels = list(data.keys())
        values = [data[k] for k in labels]
        display_names = [L2_NAMES.get(k, L3_NAMES.get(k, k)) for k in labels]

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
        bp = ax.boxplot(values, patch_artist=True, labels=display_names, widths=0.6)

        # Coloring
        for i, box in enumerate(bp["boxes"]):
            color = COLORS[0] if labels[i].startswith("HV") else COLORS[1]
            box.set_facecolor(color)
            box.set_alpha(0.6)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(-0.05, 1.05)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS[0], alpha=0.6, label="L2 Human Values"),
            Patch(facecolor=COLORS[1], alpha=0.6, label="L3 System Values"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        out_path = str(self.output_dir / filename)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # Grouped Bar Chart
    # ------------------------------------------------------------------

    def plot_grouped_bar(
        self,
        groups: dict[str, dict[str, float]],
        title: str = "",
        filename: str = "grouped_bar.pdf",
        xlabel: str = "Metric",
        ylabel: str = "Value",
    ) -> str:
        """Grouped bar chart.

        Args:
            groups: {group_label: {metric_name: value}}
        """
        group_labels = list(groups.keys())
        metric_names = list(groups[group_labels[0]].keys())
        n_groups = len(group_labels)
        n_metrics = len(metric_names)

        x = np.arange(n_metrics)
        width = 0.8 / n_groups

        fig, ax = plt.subplots(figsize=(max(7, n_metrics * 1.2), 5))

        for i, gl in enumerate(group_labels):
            vals = [groups[gl].get(m, 0) for m in metric_names]
            offset = (i - n_groups / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=gl, color=COLORS[i % len(COLORS)], alpha=0.85)
            # Annotate values
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend()

        out_path = str(self.output_dir / filename)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # Bar Chart with Error Bars
    # ------------------------------------------------------------------

    def plot_stability_errorbar(
        self,
        data: dict[str, dict],
        title: str = "",
        filename: str = "stability_errorbar.pdf",
    ) -> str:
        """Bar chart with error bars to illustrate stability.

        Args:
            data: {model_key: {value_id: {"mean": ..., "std": ...}}}
        """
        from .profile_generator import ALL_VALUE_IDS

        models = list(data.keys())
        n_models = len(models)
        n_dims = len(ALL_VALUE_IDS)
        x = np.arange(n_dims)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(max(12, n_dims * 0.7), 5))

        for i, model in enumerate(models):
            means = [data[model].get(v, {}).get("mean", 0) for v in ALL_VALUE_IDS]
            stds = [data[model].get(v, {}).get("std", 0) for v in ALL_VALUE_IDS]
            offset = (i - n_models / 2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   label=model, color=COLORS[i % len(COLORS)], alpha=0.8,
                   capsize=2, error_kw={"linewidth": 0.8})

        display_names = [L2_NAMES.get(v, L3_NAMES.get(v, v)) for v in ALL_VALUE_IDS]
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)

        out_path = str(self.output_dir / filename)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # Line Chart — Convergence Trend
    # ------------------------------------------------------------------

    def plot_convergence_lines(
        self,
        data: dict[str, list[dict]],
        dimension: str = "l2",
        title: str = "",
        filename: str = "convergence.pdf",
        step_labels: Optional[list[str]] = None,
    ) -> str:
        """Line chart: Profile score convergence trend over steps.

        Args:
            data: {model_key: [profile_step0, profile_step1, ...]}
            dimension: "l2" or "l3"
        """
        if dimension == "l2":
            ids = list(L2_NAMES.keys())
            names = L2_NAMES
            score_key = "l2_scores"
        else:
            ids = list(L3_NAMES.keys())
            names = L3_NAMES
            score_key = "l3_scores"

        models = list(data.keys())
        n_steps = max(len(v) for v in data.values())
        steps_x = list(range(1, n_steps + 1))
        if step_labels is None:
            step_labels = [f"Step {i}" for i in steps_x]

        # Select the top-5 dimensions with the largest variation
        all_ranges = {}
        for vid in ids:
            scores_all = []
            for model, profiles in data.items():
                for p in profiles:
                    scores_all.append(p.get(score_key, {}).get(vid, 0.0))
            if scores_all:
                all_ranges[vid] = max(scores_all) - min(scores_all)
        top_dims = sorted(all_ranges, key=all_ranges.get, reverse=True)[:5]

        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
        if len(models) == 1:
            axes = [axes]

        for ax, model in zip(axes, models):
            profiles = data[model]
            for idx, vid in enumerate(top_dims):
                scores = [p.get(score_key, {}).get(vid, 0.0) for p in profiles]
                ax.plot(steps_x[:len(scores)], scores, "o-",
                        label=f"{vid} ({names[vid]})",
                        color=COLORS[idx % len(COLORS)],
                        linewidth=2, markersize=5)
            ax.set_xlabel("Document Increment Step", fontsize=11)
            ax.set_title(model, fontsize=12)
            ax.set_xticks(steps_x)
            ax.set_xticklabels(step_labels[:n_steps], rotation=30, ha="right", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8, loc="lower right")

        axes[0].set_ylabel("Score", fontsize=11)
        fig.suptitle(title or f"Profile Convergence ({dimension.upper()})", fontsize=14, y=1.02)

        out_path = str(self.output_dir / filename)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # Distance Decay Bar Chart
    # ------------------------------------------------------------------

    def plot_distance_decay(
        self,
        data: dict[str, list[float]],
        title: str = "",
        filename: str = "distance_decay.pdf",
        step_labels: Optional[list[str]] = None,
    ) -> str:
        """Step-wise distance decay bar chart.

        Args:
            data: {model_key: [dist_step1_to_2, dist_step2_to_3, ...]}
        """
        models = list(data.keys())
        n_steps = max(len(v) for v in data.values())
        x = np.arange(n_steps)
        width = 0.8 / len(models)

        if step_labels is None:
            step_labels = [f"Step {i+1}→{i+2}" for i in range(n_steps)]

        fig, ax = plt.subplots(figsize=(max(7, n_steps * 1.5), 5))

        for i, model in enumerate(models):
            dists = data[model]
            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x[:len(dists)] + offset, dists, width,
                   label=model, color=COLORS[i % len(COLORS)], alpha=0.85)

        ax.set_xlabel("Document Increment", fontsize=11)
        ax.set_ylabel("Profile Distance (L2 norm)", fontsize=11)
        ax.set_title(title or "Profile Convergence: Step-wise Distance Decay", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(step_labels[:n_steps], rotation=30, ha="right")
        ax.legend()

        out_path = str(self.output_dir / filename)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # Alpha Parameter Curve — Bayesian Calibration Experiment
    # ------------------------------------------------------------------

    def plot_alpha_curve(
        self,
        sweep_data: dict[float, dict[str, dict]],
        metric_key: str = "Value F1",
        title: str = "Bayesian Calibration: F1 vs Weighting Strength",
        filename: str = "exp3_alpha_curve.pdf",
        optimal_alpha: float | None = None,
    ) -> str:
        """Plot F1 vs. alpha parameter curve, showing the effect of Bayesian calibration strength on detection performance.

        Args:
            sweep_data: {alpha: {"code": metrics, "text": metrics, "overall": metrics}}
            metric_key: Name of the metric to plot (e.g., "Value F1", "Risk F1")
            optimal_alpha: Optimal alpha value, annotated as a vertical dashed line
        """
        alphas = sorted(sweep_data.keys())
        splits = ["code", "text", "overall"]
        split_styles = {
            "code": {"color": COLORS[0], "marker": "s", "label": "Code Scenarios"},
            "text": {"color": COLORS[1], "marker": "^", "label": "Issue Text Scenarios"},
            "overall": {"color": "#333333", "marker": "o", "label": "Overall", "linewidth": 2.5},
        }

        fig, ax = plt.subplots(figsize=(7, 4.5))

        for split in splits:
            style = split_styles[split]
            values = []
            for a in alphas:
                metrics = sweep_data[a].get(split, {})
                values.append(metrics.get(metric_key, 0.0))
            lw = style.get("linewidth", 1.8)
            ax.plot(alphas, values, marker=style["marker"], color=style["color"],
                    label=style["label"], linewidth=lw, markersize=6)

        # Mark the optimal alpha
        if optimal_alpha is not None:
            ax.axvline(x=optimal_alpha, color="#888888", linestyle="--",
                       linewidth=1.2, alpha=0.7)
            # Annotation text
            y_range = ax.get_ylim()
            ax.text(optimal_alpha + 0.03, y_range[0] + (y_range[1] - y_range[0]) * 0.05,
                    f"$\\alpha^*$={optimal_alpha}", fontsize=9, color="#555555")

        ax.set_xlabel(r"Calibration Strength $\alpha$", fontsize=11)
        ax.set_ylabel(metric_key, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(alphas)
        ax.legend(loc="best")

        out_path = str(self.output_dir / filename)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # LaTeX Table
    # ------------------------------------------------------------------

    @staticmethod
    def generate_latex_table(
        headers: list[str],
        rows: list[list[str]],
        caption: str = "",
        label: str = "",
    ) -> str:
        """Generate a LaTeX table in booktabs format."""
        cols = "l" + "c" * (len(headers) - 1)
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}" if caption else "",
            f"\\label{{{label}}}" if label else "",
            f"\\begin{{tabular}}{{{cols}}}",
            r"\toprule",
            " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\",
            r"\midrule",
        ]
        for row in rows:
            lines.append(" & ".join(row) + r" \\")
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(line for line in lines if line)

    def save_latex_table(self, content: str, filename: str) -> str:
        out_path = str(self.output_dir / filename)
        Path(out_path).write_text(content, encoding="utf-8")
        return out_path
