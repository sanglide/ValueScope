#!/usr/bin/env python
"""
Main value-hypothesis identification experiment script.

Evaluates how well different LLMs generate value hypotheses for code and text scenarios.

Supported scenario types:
  - code: code scenarios (code_scenarios/*.json)
  - text: issue discussions (text_scenarios/issues.json)

Results are reported per code / text / overall.
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
    from . import paths as exp_paths
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
    import paths as exp_paths


class ValueRiskExperiment:
    """价值风险识别实验类"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.llm_clients: dict[str, BaseLLMClient] = {}
        self.value_model_loader: Optional[ValueModelLoader] = None
        self.metrics_calculator = MetricsCalculator(
            confidence_threshold=self.config.get("evaluation", {}).get("confidence_threshold", 0.5)
        )
        # Datasets indexed by scenario type ("code" / "text")
        self.datasets: dict[str, list[ValueScenarioSample]] = {}
        # Predictions indexed by (model_key, scenario_type)
        self.results: dict[str, dict[str, list[PredictionResult]]] = {}
        
        # Logging
        project_root = Path(__file__).parent.parent.parent
        self.log_dir = project_root / self.config.get(
            "output", {}
        ).get(
            "logs_dir",
            str(exp_paths.LOGS_DIR.relative_to(exp_paths.PROJECT_ROOT)),
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"experiment_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # 创建聚合数据和LLM输出的存储目录
        self.aggregated_data_dir = self.log_dir / "aggregated_data"
        self.llm_outputs_dir = exp_paths.LLM_OUTPUTS_DIR
        self.aggregated_data_dir.mkdir(exist_ok=True)
        self.llm_outputs_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        config_file = Path(__file__).parent / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def setup(self, tables_dir: str = None, data_file: str = None) -> None:
        """初始化实验环境"""
        print("=" * 60)
        print("初始化实验环境")
        print("=" * 60)

        # 加载价值模型
        if tables_dir is None:
            tables_dir = Path(__file__).parent.parent.parent / "tables"
        self.value_model_loader = ValueModelLoader(str(tables_dir))
        self.value_model_loader.load()
        print(f"已加载 {len(self.value_model_loader.l2_values)} 个L2价值, "
              f"{len(self.value_model_loader.l3_values)} 个L3价值")

        # 加载数据集 —— 优先使用命令行参数，否则按config.yaml配置加载
        if data_file:
            self._load_single_data_file(data_file)
        else:
            self._load_datasets_from_config()

        # 创建LLM客户端
        llm_configs = self.config.get("llm_models", {})
        self.llm_clients = LLMClientFactory.create_all_enabled(llm_configs)
        print(f"已创建 {len(self.llm_clients)} 个LLM客户端: {list(self.llm_clients.keys())}")
        print()

    def _load_single_data_file(self, data_file: str) -> None:
        """从单个文件加载（向后兼容）"""
        loader = ScenarioDataLoader(data_file)
        if data_file.endswith('.json'):
            loader.load_from_json(data_file)
        elif data_file.endswith('.csv'):
            loader.load_from_csv(data_file)
        self.datasets["code"] = loader.get_samples()
        print(f"已加载 {len(self.datasets['code'])} 个场景样本 (from file)")

    def _load_datasets_from_config(self) -> None:
        """根据config.yaml中的datasets配置加载多个数据集"""
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
                # 为每个样本打上 scenario_type
                for s in samples:
                    s.metadata["scenario_type"] = scenario_type

            elif ds_type == "json_dir":
                loader = ScenarioDataLoader()
                loader.load_from_directory(str(ds_path))
                samples = loader.get_samples()
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
                print(f"警告: 未知的数据集类型 '{ds_type}'，跳过 {ds_key}")
                continue

            if scenario_type not in self.datasets:
                self.datasets[scenario_type] = []
            self.datasets[scenario_type].extend(samples)
            print(f"[{ds_key}] 已加载 {len(samples)} 个 {scenario_type} 样本")

        if not self.datasets:
            print("未找到任何数据集配置，使用内置示例数据集")
            for sample in create_sample_dataset():
                sample.metadata["scenario_type"] = "code"
            self.datasets["code"] = create_sample_dataset()
            print(f"已加载 {len(self.datasets['code'])} 个内置示例样本")

    # ------------------------------------------------------------------
    # prompt
    # ------------------------------------------------------------------
    def _get_prompt_name_for_scenario_type(self, scenario_type: str) -> str:
        """根据场景类型选择对应的prompt"""
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
        """构建prompt"""
        prompts_config = self.config.get("prompts", {})
        prompt_config = prompts_config.get(prompt_name, prompts_config.get("value_risk_identification"))

        system_prompt = prompt_config.get("system_prompt", "")
        user_prompt_template = prompt_config.get("user_prompt_template", "")

        value_model_text = self.value_model_loader.format_value_model_for_prompt()

        user_prompt = user_prompt_template.replace("{value_scenario}", sample.scenario_content)
        user_prompt = user_prompt.replace("{value_model}", value_model_text)
        
        # 记录聚合后的数据
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
        
        self.logger.info(f"已记录聚合数据: {sample.sample_id}")

        return system_prompt, user_prompt

    # ------------------------------------------------------------------
    # LLM调用 & 解析
    # ------------------------------------------------------------------
    def _parse_llm_result(self, response: LLMResponse) -> tuple:
        """解析LLM响应"""
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
        """在一组样本上运行单个模型"""
        prompt_name = self._get_prompt_name_for_scenario_type(scenario_type)
        predictions = []

        for i, sample in enumerate(samples):
            if verbose:
                print(f"  [{model_key}][{scenario_type}] {i+1}/{len(samples)}: {sample.sample_id}")

            system_prompt, user_prompt = self._build_prompt(sample, prompt_name)
            response = client.call(system_prompt, user_prompt)
            has_risk, values, confidences = self._parse_llm_result(response)
            
            # 记录LLM输出
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
            
            self.logger.info(f"已记录LLM输出: {model_key} - {sample.sample_id}")

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
                print(f"    警告: {response.error}")

        return predictions

    # ------------------------------------------------------------------
    # 主实验逻辑
    # ------------------------------------------------------------------
    def run_experiment(
        self,
        parallel: bool = False,
        verbose: bool = True
    ) -> dict[str, dict[str, EvaluationMetrics]]:
        """运行完整实验

        Returns:
            嵌套字典: {scenario_type: {model_key: EvaluationMetrics}}
            其中 scenario_type 包含 "code", "text", "overall"
        """
        total_samples = sum(len(v) for v in self.datasets.values())
        if total_samples == 0:
            raise ValueError("没有可用的场景样本")

        print(f"开始实验:")
        for stype, samples in self.datasets.items():
            n_risk = sum(1 for s in samples if s.ground_truth_has_risk)
            print(f"  {stype}: {len(samples)} 个样本 (有风险={n_risk}, 无风险={len(samples)-n_risk})")
        print(f"  模型数量: {len(self.llm_clients)}")
        print()

        # 收集所有 (model_key, scenario_type) -> predictions
        all_predictions: dict[str, dict[str, list[PredictionResult]]] = {}

        for model_key, client in self.llm_clients.items():
            all_predictions[model_key] = {}
            for scenario_type, samples in self.datasets.items():
                print(f"运行: {model_key} on {scenario_type} ({len(samples)} samples)")
                preds = self._run_model_on_samples(
                    model_key, client, samples, scenario_type, verbose
                )
                all_predictions[model_key][scenario_type] = preds

        self.results = all_predictions

        # 计算指标: per scenario_type + overall
        metrics_by_type: dict[str, dict[str, EvaluationMetrics]] = {}

        for scenario_type in list(self.datasets.keys()) + ["overall"]:
            metrics_by_type[scenario_type] = {}

            for model_key in self.llm_clients:
                if scenario_type == "overall":
                    # 合并所有类型的predictions
                    combined = []
                    for st_preds in all_predictions[model_key].values():
                        combined.extend(st_preds)
                else:
                    combined = all_predictions[model_key].get(scenario_type, [])

                m = self.metrics_calculator.calculate(combined, model_key)
                metrics_by_type[scenario_type][model_key] = m

            # 添加人工标注基准
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
    # 报告
    # ------------------------------------------------------------------
    def generate_report(
        self,
        metrics_by_type: dict[str, dict[str, EvaluationMetrics]],
        experiment_name: str = "value_risk_experiment",
        output_formats: list[str] = None
    ) -> dict[str, str]:
        """Generate experiment report."""
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / self.config.get(
            "output", {}
        ).get(
            "results_dir",
            str(exp_paths.RESULTS_DIR.relative_to(exp_paths.PROJECT_ROOT)),
        )
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

        # 生成综合详细报告
        all_md_parts = [
            "# Value Risk Identification Experiment Report\n",
            f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        ]

        # 数据集统计
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

        # LaTeX综合表格（论文用）
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

        print(f"\n报告已保存:")
        for fmt, path in saved_files.items():
            print(f"  - {fmt}: {path}")

        return saved_files

    # ------------------------------------------------------------------
    # 数据集统计
    # ------------------------------------------------------------------
    def print_dataset_statistics(self) -> None:
        """打印数据集统计信息"""
        print("\n" + "=" * 60)
        print("  DATASET STATISTICS")
        print("=" * 60)

        for scenario_type, samples in self.datasets.items():
            print(f"\n--- {scenario_type.upper()} ---")
            print(f"  总样本数: {len(samples)}")

            n_risk = sum(1 for s in samples if s.ground_truth_has_risk)
            n_no_risk = len(samples) - n_risk
            print(f"  有价值风险: {n_risk}")
            print(f"  无价值风险: {n_no_risk}")

            # 价值分布
            value_counts = {}
            for s in samples:
                for v in s.ground_truth_values:
                    value_counts[v] = value_counts.get(v, 0) + 1

            if value_counts:
                print(f"  价值标签分布:")
                for vid, cnt in sorted(value_counts.items(), key=lambda x: -x[1]):
                    print(f"    {vid}: {cnt}")

            # 按项目分布（针对issues数据）
            projects = {}
            for s in samples:
                proj = s.metadata.get("project_name", "N/A")
                projects[proj] = projects.get(proj, 0) + 1
            if len(projects) > 1:
                print(f"  项目分布:")
                for p, c in sorted(projects.items()):
                    print(f"    {p}: {c}")

        total = sum(len(s) for s in self.datasets.values())
        print(f"\n总计: {total} 个样本")
        print("=" * 60 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="价值假说识别实验")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--data", help="场景数据文件路径 (JSON或CSV, 覆盖config中的datasets配置)")
    parser.add_argument("--tables-dir", help="价值模型表格目录")
    parser.add_argument("--parallel", action="store_true", help="并行执行模型")
    parser.add_argument("--output-name", default="value_risk_experiment", help="输出文件名前缀")
    parser.add_argument("--generate-sample-data", help="生成示例数据集到指定文件")
    parser.add_argument("--stats-only", action="store_true", help="仅输出数据集统计信息，不运行实验")
    parser.add_argument("--quiet", action="store_true", help="安静模式")

    args = parser.parse_args()

    # 生成示例数据集
    if args.generate_sample_data:
        save_sample_dataset(args.generate_sample_data)
        return

    try:
        experiment = ValueRiskExperiment(args.config)
        experiment.setup(tables_dir=args.tables_dir, data_file=args.data)

        # 输出数据集统计
        experiment.print_dataset_statistics()

        if args.stats_only:
            return

        # 运行实验
        metrics_by_type = experiment.run_experiment(
            parallel=args.parallel,
            verbose=not args.quiet
        )

        # 生成报告
        experiment.generate_report(metrics_by_type, experiment_name=args.output_name)

    except Exception as e:
        print(f"实验执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
