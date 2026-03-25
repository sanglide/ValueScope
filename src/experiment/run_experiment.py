#!/usr/bin/env python
"""
Main script for value hypothesis identification experiments.
Evaluates the ability of different LLMs to generate value hypotheses in value scenarios.

Supports two scenario types:
  - code: Code scenarios (sample_scenarios.json)
  - text: Issue text scenarios (values-issues-dataset-master)

Experiment results are reported separately along three dimensions: code / text / overall.
"""

import argparse
import json
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import os

try:
    from .llm_client import LLMClientFactory, LLMResponse, BaseLLMClient
    from .data_loader import (
        ValueModelLoader,
        ScenarioDataLoader,
        ValueScenarioSample,
        IssuesDatasetLoader,
        create_sample_dataset,
        save_sample_dataset
    )
    from .evaluator import (
        PredictionResult,
        EvaluationMetrics,
        MetricsCalculator,
        create_ground_truth_metrics
    )
    from .report_generator import ReportGenerator, DetailedReportGenerator
except ImportError:
    from llm_client import LLMClientFactory, LLMResponse, BaseLLMClient
    from data_loader import (
        ValueModelLoader,
        ScenarioDataLoader,
        ValueScenarioSample,
        IssuesDatasetLoader,
        create_sample_dataset,
        save_sample_dataset
    )
    from evaluator import (
        PredictionResult,
        EvaluationMetrics,
        MetricsCalculator,
        create_ground_truth_metrics
    )
    from report_generator import ReportGenerator, DetailedReportGenerator


class ValueRiskExperiment:
    """Value risk identification experiment class"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.llm_clients: dict[str, BaseLLMClient] = {}
        self.value_model_loader: Optional[ValueModelLoader] = None
        self.metrics_calculator = MetricsCalculator(
            confidence_threshold=self.config.get("evaluation", {}).get("confidence_threshold", 0.5)
        )
        # Store samples by scenario type
        self.datasets: dict[str, list[ValueScenarioSample]] = {}  # "code" / "text"
        # Store prediction results by (model_key, scenario_type)
        self.results: dict[str, dict[str, list[PredictionResult]]] = {}
        
        # Set up logging
        self.log_dir = Path(__file__).parent / self.config.get("output", {}).get("logs_dir", "experiment_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"experiment_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create storage directories for aggregated data and LLM outputs
        self.aggregated_data_dir = self.log_dir / "aggregated_data"
        self.llm_outputs_dir = self.log_dir / "llm_outputs"
        self.aggregated_data_dir.mkdir(exist_ok=True)
        self.llm_outputs_dir.mkdir(exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        config_file = Path(__file__).parent / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def setup(self, tables_dir: str = None, data_file: str = None) -> None:
        """Initialize experiment environment"""
        print("=" * 60)
        print("Initializing experiment environment")
        print("=" * 60)

        # Load value model
        if tables_dir is None:
            tables_dir = Path(__file__).parent.parent.parent / "tables"
        self.value_model_loader = ValueModelLoader(str(tables_dir))
        self.value_model_loader.load()
        print(f"Loaded {len(self.value_model_loader.l2_values)} L2 values, "
              f"{len(self.value_model_loader.l3_values)} L3 values")

        # Load datasets -- prefer command-line argument; otherwise load from config.yaml
        if data_file:
            self._load_single_data_file(data_file)
        else:
            self._load_datasets_from_config()

        # Create LLM clients
        llm_configs = self.config.get("llm_models", {})
        self.llm_clients = LLMClientFactory.create_all_enabled(llm_configs)
        print(f"Created {len(self.llm_clients)} LLM client(s): {list(self.llm_clients.keys())}")
        print()

    def _load_single_data_file(self, data_file: str) -> None:
        """Load from a single file (backward compatible)"""
        loader = ScenarioDataLoader(data_file)
        if data_file.endswith('.json'):
            loader.load_from_json(data_file)
        elif data_file.endswith('.csv'):
            loader.load_from_csv(data_file)
        self.datasets["code"] = loader.get_samples()
        print(f"Loaded {len(self.datasets['code'])} scenario samples (from file)")

    def _load_datasets_from_config(self) -> None:
        """Load multiple datasets based on the datasets configuration in config.yaml"""
        datasets_config = self.config.get("datasets", {})
        base_dir = Path(__file__).parent

        for ds_key, ds_conf in datasets_config.items():
            if not ds_conf.get("enabled", True):
                continue

            ds_type = ds_conf.get("type", "json")
            scenario_type = ds_conf.get("scenario_type", "code")
            ds_path = base_dir / ds_conf.get("path", "")

            if ds_type == "json":
                loader = ScenarioDataLoader()
                loader.load_from_json(str(ds_path))
                samples = loader.get_samples()
                # Tag each sample with scenario_type
                for s in samples:
                    s.metadata["scenario_type"] = scenario_type

            elif ds_type == "issues_dataset":
                issues_loader = IssuesDatasetLoader(str(ds_path))
                samples = issues_loader.load(
                    sample_per_project=ds_conf.get("sample_per_project"),
                    max_text_length=ds_conf.get("max_text_length", 8000),
                    seed=ds_conf.get("seed", 42)
                )
            else:
                print(f"Warning: Unknown dataset type '{ds_type}', skipping {ds_key}")
                continue

            if scenario_type not in self.datasets:
                self.datasets[scenario_type] = []
            self.datasets[scenario_type].extend(samples)
            print(f"[{ds_key}] Loaded {len(samples)} {scenario_type} sample(s)")

        if not self.datasets:
            print("No dataset configuration found, using built-in sample dataset")
            for sample in create_sample_dataset():
                sample.metadata["scenario_type"] = "code"
            self.datasets["code"] = create_sample_dataset()
            print(f"Loaded {len(self.datasets['code'])} built-in sample(s)")

    # ------------------------------------------------------------------
    # prompt
    # ------------------------------------------------------------------
    def _get_prompt_name_for_scenario_type(self, scenario_type: str) -> str:
        """Select the corresponding prompt based on scenario type"""
        prompts_config = self.config.get("prompts", {})
        mapping = {
            "code": "code_value_risk",
            "text": "text_value_risk",
        }
        preferred = mapping.get(scenario_type, "value_risk_identification")
        if preferred in prompts_config:
            return preferred
        return "value_risk_identification"

    def _build_prompt(self, sample: ValueScenarioSample, prompt_name: str) -> tuple:
        """Build prompt"""
        prompts_config = self.config.get("prompts", {})
        prompt_config = prompts_config.get(prompt_name, prompts_config.get("value_risk_identification"))

        system_prompt = prompt_config.get("system_prompt", "")
        user_prompt_template = prompt_config.get("user_prompt_template", "")

        value_model_text = self.value_model_loader.format_value_model_for_prompt()

        user_prompt = user_prompt_template.replace("{value_scenario}", sample.scenario_content)
        user_prompt = user_prompt.replace("{value_model}", value_model_text)
        
        # Log aggregated data
        sample_log = {
            "sample_id": sample.sample_id,
            "scenario_content": sample.scenario_content,
            "ground_truth_has_risk": sample.ground_truth_has_risk,
            "ground_truth_values": sample.ground_truth_values,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "metadata": sample.metadata
        }
        
        sample_log_file = self.aggregated_data_dir / f"{sample.sample_id}_input.json"
        with open(sample_log_file, 'w', encoding='utf-8') as f:
            json.dump(sample_log, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Recorded aggregated data: {sample.sample_id}")

        return system_prompt, user_prompt

    # ------------------------------------------------------------------
    # LLM invocation & parsing
    # ------------------------------------------------------------------
    def _parse_llm_result(self, response: LLMResponse) -> tuple:
        """Parse LLM response"""
        if response.error or not response.parsed_result:
            return False, [], {}

        result = response.parsed_result
        has_risk = result.get("has_value_risk", False)

        identified_values = []
        confidences = {}
        for item in result.get("identified_values", []):
            value_id = item.get("value_id", "")
            if value_id:
                identified_values.append(value_id)
                confidences[value_id] = item.get("confidence", 1.0)

        return has_risk, identified_values, confidences

    def _run_model_on_samples(
        self,
        model_key: str,
        client: BaseLLMClient,
        samples: list[ValueScenarioSample],
        scenario_type: str,
        verbose: bool = True
    ) -> list[PredictionResult]:
        """Run a single model on a set of samples"""
        prompt_name = self._get_prompt_name_for_scenario_type(scenario_type)
        predictions = []

        for i, sample in enumerate(samples):
            if verbose:
                print(f"  [{model_key}][{scenario_type}] {i+1}/{len(samples)}: {sample.sample_id}")

            system_prompt, user_prompt = self._build_prompt(sample, prompt_name)
            response = client.call(system_prompt, user_prompt)
            has_risk, values, confidences = self._parse_llm_result(response)
            
            # Log LLM output
            llm_output_log = {
                "sample_id": sample.sample_id,
                "model": model_key,
                "scenario_type": scenario_type,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": response.raw_response,
                "parsed_result": response.parsed_result,
                "predicted_has_risk": has_risk,
                "predicted_values": values,
                "predicted_confidences": confidences,
                "ground_truth_has_risk": sample.ground_truth_has_risk,
                "ground_truth_values": sample.ground_truth_values,
                "error": response.error,
                "latency_ms": response.latency_ms
            }
            
            llm_output_file = self.llm_outputs_dir / f"{model_key}_{sample.sample_id}_output.json"
            with open(llm_output_file, 'w', encoding='utf-8') as f:
                json.dump(llm_output_log, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Recorded LLM output: {model_key} - {sample.sample_id}")

            pred = PredictionResult(
                sample_id=sample.sample_id,
                predicted_has_risk=has_risk,
                predicted_values=values,
                predicted_confidences=confidences,
                ground_truth_has_risk=sample.ground_truth_has_risk,
                ground_truth_values=sample.ground_truth_values
            )
            predictions.append(pred)

            if verbose and response.error:
                print(f"    Warning: {response.error}")

        return predictions

    # ------------------------------------------------------------------
    # Main experiment logic
    # ------------------------------------------------------------------
    def run_experiment(
        self,
        parallel: bool = False,
        verbose: bool = True
    ) -> dict[str, dict[str, EvaluationMetrics]]:
        """Run the full experiment

        Returns:
            Nested dict: {scenario_type: {model_key: EvaluationMetrics}}
            where scenario_type includes "code", "text", "overall"
        """
        total_samples = sum(len(v) for v in self.datasets.values())
        if total_samples == 0:
            raise ValueError("No available scenario samples")

        print(f"Starting experiment:")
        for stype, samples in self.datasets.items():
            n_risk = sum(1 for s in samples if s.ground_truth_has_risk)
            print(f"  {stype}: {len(samples)} samples (with risk={n_risk}, no risk={len(samples)-n_risk})")
        print(f"  Number of models: {len(self.llm_clients)}")
        print()

        # Collect all (model_key, scenario_type) -> predictions
        all_predictions: dict[str, dict[str, list[PredictionResult]]] = {}

        for model_key, client in self.llm_clients.items():
            all_predictions[model_key] = {}
            for scenario_type, samples in self.datasets.items():
                print(f"Running: {model_key} on {scenario_type} ({len(samples)} samples)")
                preds = self._run_model_on_samples(
                    model_key, client, samples, scenario_type, verbose
                )
                all_predictions[model_key][scenario_type] = preds

        self.results = all_predictions

        # Calculate metrics: per scenario_type + overall
        metrics_by_type: dict[str, dict[str, EvaluationMetrics]] = {}

        for scenario_type in list(self.datasets.keys()) + ["overall"]:
            metrics_by_type[scenario_type] = {}

            for model_key in self.llm_clients:
                if scenario_type == "overall":
                    # Merge predictions from all types
                    combined = []
                    for st_preds in all_predictions[model_key].values():
                        combined.extend(st_preds)
                else:
                    combined = all_predictions[model_key].get(scenario_type, [])

                m = self.metrics_calculator.calculate(combined, model_key)
                metrics_by_type[scenario_type][model_key] = m

            # Add human annotation baseline
            if scenario_type == "overall":
                gt_samples = []
                for samples in self.datasets.values():
                    gt_samples.extend(samples)
            else:
                gt_samples = self.datasets.get(scenario_type, [])

            gt_preds = [
                PredictionResult(
                    sample_id=s.sample_id,
                    predicted_has_risk=s.ground_truth_has_risk,
                    predicted_values=s.ground_truth_values,
                    predicted_confidences={v: 1.0 for v in s.ground_truth_values},
                    ground_truth_has_risk=s.ground_truth_has_risk,
                    ground_truth_values=s.ground_truth_values
                )
                for s in gt_samples
            ]
            metrics_by_type[scenario_type]["Human (Ground Truth)"] = \
                create_ground_truth_metrics(gt_preds)

        return metrics_by_type

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def generate_report(
        self,
        metrics_by_type: dict[str, dict[str, EvaluationMetrics]],
        experiment_name: str = "value_risk_experiment",
        output_formats: list[str] = None
    ) -> dict[str, str]:
        """Generate experiment report"""
        output_dir = str(Path(__file__).parent / self.config.get("output", {}).get("results_dir", "experiment_results"))
        output_formats = output_formats or ["markdown", "latex", "csv"]

        generator = DetailedReportGenerator(output_dir)
        saved_files = {}

        for scenario_type, metrics_dict in metrics_by_type.items():
            metrics_list = []
            for key, m in metrics_dict.items():
                if key != "Human (Ground Truth)":
                    metrics_list.append(m)
            if "Human (Ground Truth)" in metrics_dict:
                metrics_list.append(metrics_dict["Human (Ground Truth)"])

            tag = f"{experiment_name}_{scenario_type}"

            print(f"\n{'='*60}")
            print(f"  Results: {scenario_type.upper()}")
            print(f"{'='*60}")
            generator.print_summary(metrics_list)

            files = generator.save_results(metrics_list, experiment_name=tag, formats=output_formats)
            for fmt, path in files.items():
                saved_files[f"{scenario_type}_{fmt}"] = path

        # Generate comprehensive detailed report
        all_md_parts = [
            "# Value Risk Identification Experiment Report\n",
            f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        ]

        # Dataset statistics
        all_md_parts.append("## Dataset Summary\n")
        for stype, samples in self.datasets.items():
            n_risk = sum(1 for s in samples if s.ground_truth_has_risk)
            all_md_parts.append(f"- **{stype}**: {len(samples)} samples "
                                f"(with risk: {n_risk}, no risk: {len(samples)-n_risk})")
        all_md_parts.append("")

        for scenario_type, metrics_dict in metrics_by_type.items():
            metrics_list = [m for k, m in metrics_dict.items() if k != "Human (Ground Truth)"]
            if "Human (Ground Truth)" in metrics_dict:
                metrics_list.append(metrics_dict["Human (Ground Truth)"])

            all_md_parts.append(f"## {scenario_type.upper()} Results\n")
            all_md_parts.append(generator.generate_markdown_table(metrics_list))
            all_md_parts.append("")

        # Combined LaTeX table (for paper)
        all_md_parts.append("## LaTeX Table (for paper)\n")
        all_md_parts.append("```latex")
        for scenario_type, metrics_dict in metrics_by_type.items():
            metrics_list = [m for k, m in metrics_dict.items() if k != "Human (Ground Truth)"]
            if "Human (Ground Truth)" in metrics_dict:
                metrics_list.append(metrics_dict["Human (Ground Truth)"])
            all_md_parts.append(generator.generate_latex_table(
                metrics_list,
                caption=f"Value Risk Identification Results ({scenario_type})",
                label=f"tab:results_{scenario_type}"
            ))
            all_md_parts.append("")
        all_md_parts.append("```\n")

        detailed_file = Path(output_dir) / f"{experiment_name}_full_report.md"
        detailed_file.parent.mkdir(parents=True, exist_ok=True)
        detailed_file.write_text("\n".join(all_md_parts), encoding='utf-8')
        saved_files["full_report"] = str(detailed_file)

        print(f"\nReports saved:")
        for fmt, path in saved_files.items():
            print(f"  - {fmt}: {path}")

        return saved_files

    # ------------------------------------------------------------------
    # Dataset statistics
    # ------------------------------------------------------------------
    def print_dataset_statistics(self) -> None:
        """Print dataset statistics"""
        print("\n" + "=" * 60)
        print("  DATASET STATISTICS")
        print("=" * 60)

        for scenario_type, samples in self.datasets.items():
            print(f"\n--- {scenario_type.upper()} ---")
            print(f"  Total samples: {len(samples)}")

            n_risk = sum(1 for s in samples if s.ground_truth_has_risk)
            n_no_risk = len(samples) - n_risk
            print(f"  With value risk: {n_risk}")
            print(f"  Without value risk: {n_no_risk}")

            # Value distribution
            value_counts = {}
            for s in samples:
                for v in s.ground_truth_values:
                    value_counts[v] = value_counts.get(v, 0) + 1

            if value_counts:
                print(f"  Value label distribution:")
                for vid, cnt in sorted(value_counts.items(), key=lambda x: -x[1]):
                    print(f"    {vid}: {cnt}")

            # Distribution by project (for issues data)
            projects = {}
            for s in samples:
                proj = s.metadata.get("project_name", "N/A")
                projects[proj] = projects.get(proj, 0) + 1
            if len(projects) > 1:
                print(f"  Project distribution:")
                for p, c in sorted(projects.items()):
                    print(f"    {p}: {c}")

        total = sum(len(s) for s in self.datasets.values())
        print(f"\nTotal: {total} samples")
        print("=" * 60 + "\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Value hypothesis identification experiment")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--data", help="Path to scenario data file (JSON or CSV, overrides datasets config)")
    parser.add_argument("--tables-dir", help="Directory containing value model tables")
    parser.add_argument("--parallel", action="store_true", help="Run models in parallel")
    parser.add_argument("--output-name", default="value_risk_experiment", help="Output filename prefix")
    parser.add_argument("--generate-sample-data", help="Generate sample dataset to the specified file")
    parser.add_argument("--stats-only", action="store_true", help="Only print dataset statistics without running the experiment")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    # Generate sample dataset
    if args.generate_sample_data:
        save_sample_dataset(args.generate_sample_data)
        return

    try:
        experiment = ValueRiskExperiment(args.config)
        experiment.setup(tables_dir=args.tables_dir, data_file=args.data)

        # Print dataset statistics
        experiment.print_dataset_statistics()

        if args.stats_only:
            return

        # Run experiment
        metrics_by_type = experiment.run_experiment(
            parallel=args.parallel,
            verbose=not args.quiet
        )

        # Generate report
        experiment.generate_report(metrics_by_type, experiment_name=args.output_name)

    except Exception as e:
        print(f"Experiment execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
