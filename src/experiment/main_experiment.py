#!/usr/bin/env python3
"""
Main experiment entry point — ValueGuard value-risk identification.

Research question:
  To what extent does profile-guided, evidence-verified value risk identification
  (ValueGuard) improve agreement with human annotations compared to zero-shot
  LLM baselines, and what is the contribution of each pipeline component?

Usage:
  # Mock mode (for debugging)
  uv run python -m experiment.main_experiment --mock

  # Full run with the primary model (qwen-plus)
  uv run python -m experiment.main_experiment

  # Limit samples and specify a model
  uv run python -m experiment.main_experiment --primary-model qwen-plus --max-samples 50

  # Cross-model validation
  uv run python -m experiment.main_experiment --validation-models deepseek-chat grok-4
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# 加载 .env 文件中的环境变量
from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path, override=True)

# 验证关键环境变量是否加载成功
import os
_required_env_vars = ["DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "GPTSAPI_KEY"]
_missing_vars = [v for v in _required_env_vars if not os.getenv(v)]
if _missing_vars:
    import warnings
    warnings.warn(
        f"Missing environment variables: {', '.join(_missing_vars)}. "
        f"Please check {_env_path} exists and has the correct values."
    )

import yaml

# Project paths
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from experiment.benchmark_builder import BenchmarkBuilder, BenchmarkSample, save_benchmark, load_benchmark
from experiment.unified_pipeline import UnifiedPipelineEvaluator, UnifiedSampleResult, compute_unified_metrics
from experiment.stat_tests import compare_methods, bootstrap_ci
from experiment.iaa_metrics import precision_recall_f1_binary, micro_precision_recall_f1
from experiment.llm_client import LLMClientFactory
from experiment.traditional_baselines import run_all_baselines as run_traditional_baselines
from experiment import paths as exp_paths

# ValueGuard Profile 加载
from valueguard.core.models import ValueProfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("main_experiment")

# Value ID 映射
VALUE_NAMES = {
    "HV1": "Conformity", "HV2": "Pleasure", "HV3": "Dignity",
    "HV4": "Inclusiveness", "HV5": "Sense of Belonging", "HV6": "Freedom",
    "HV7": "Independence", "HV8": "Wealth", "HV9": "Privacy", "HV10": "Security",
    "SV1": "Trust", "SV2": "Correctness", "SV3": "Compatibility",
    "SV4": "Portability", "SV5": "Reliability", "SV6": "Efficiency",
    "SV7": "Energy Preservation", "SV8": "Usability", "SV9": "Accessibility",
    "SV10": "Longevity",
}


# ============================================================
# Profile 加载
# ============================================================

def load_profiles_from_cache(cache_dir: str, model_key: str = "qwen-plus") -> dict[str, ValueProfile]:
    """Load all repository profiles from the cache directory."""
    cache_path = Path(cache_dir) / model_key
    profiles = {}

    if not cache_path.exists():
        logger.warning(f"Profile cache not found: {cache_path}")
        # Fallback: load from unified profile results directory
        results_dir = exp_paths.PROFILE_DIR
        for f in results_dir.glob("exp1_*_profile.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                repo_name = f.stem.replace("exp1_", "").replace("_profile", "")
                profiles[repo_name] = _dict_to_profile(data, repo_name)
            except Exception as e:
                logger.warning(f"Failed to load profile from {f}: {e}")
        return profiles

    for f in sorted(cache_path.glob("*_default.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            repo_name = f.stem.replace("_default", "")
            # v2 profile 文件名带 _v2 后缀，剥离使 v2 覆盖 v1
            if repo_name.endswith("_v2"):
                repo_name = repo_name[:-3]
            profiles[repo_name] = _dict_to_profile(data, repo_name)
        except Exception as e:
            logger.warning(f"Failed to load profile {f}: {e}")

    # Track profile versions
    v2_count = sum(1 for f in cache_path.glob("*_v2_default.json"))
    logger.info(f"Loaded {len(profiles)} profiles from {cache_path} ({v2_count} v2 profiles)")
    return profiles


def _dict_to_profile(data: dict, repo_name: str) -> ValueProfile:
    """Convert a JSON dict to a ValueProfile."""
    l2_scores = {}
    for vid in [f"HV{i}" for i in range(1, 11)]:
        l2_scores[vid] = data.get("l2_scores", {}).get(vid, 0.5)

    l3_scores = {}
    for vid in [f"SV{i}" for i in range(1, 11)]:
        l3_scores[vid] = data.get("l3_scores", {}).get(vid, 0.5)

    core_values = data.get("core_values", [])

    return ValueProfile(
        repo=repo_name,
        l2_scores=l2_scores,
        l3_scores=l3_scores,
        core_values=core_values,
    )


# ============================================================
# LLM-only Baseline (reuses IAA cache)
# ============================================================

def load_llm_only_baseline(
    iaa_cache_dir: str,
    model_key: str,
    sample_ids: list[str],
) -> dict[str, dict]:
    """Load LLM-only baseline predictions from the IAA cache.

    Returns:
        {sample_id: {"predicted_has_risk": bool, "predicted_values": list,
                      "predicted_confidences": dict}}
    """
    cache_path = Path(iaa_cache_dir)
    results = {}
    loaded = 0

    for sid in sample_ids:
        cache_file = cache_path / f"{model_key}_{sid}_output.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                results[sid] = {
                    "predicted_has_risk": data.get("predicted_has_risk", False),
                    "predicted_values": data.get("predicted_values", []),
                    "predicted_confidences": data.get("predicted_confidences", {}),
                }
                loaded += 1
                continue
            except Exception:
                pass

        # 缓存不存在 → 默认无风险预测
        results[sid] = {
            "predicted_has_risk": False,
            "predicted_values": [],
            "predicted_confidences": {},
        }

    logger.info(f"LLM-only baseline: loaded {loaded}/{len(sample_ids)} from cache ({model_key})")
    return results


def llm_only_to_unified_results(
    predictions: dict[str, dict],
    benchmark_samples: list[BenchmarkSample],
) -> list[UnifiedSampleResult]:
    """Convert LLM-only baseline predictions to UnifiedSampleResult."""
    results = []
    for bs in benchmark_samples:
        pred = predictions.get(bs.sample_id, {})
        results.append(UnifiedSampleResult(
            sample_id=bs.sample_id,
            scenario_type=bs.scenario_type,
            repo=bs.repo,
            predicted_has_risk=pred.get("predicted_has_risk", False),
            predicted_values=pred.get("predicted_values", []),
            predicted_confidences=pred.get("predicted_confidences", {}),
            ground_truth_has_risk=bs.has_value_risk,
            ground_truth_values=bs.ground_truth_values,
            hypothesis_count=0,
            confirmed_count=0,
            profile_used="none",
            total_time_ms=0.0,
        ))
    return results


# ============================================================
# 主实验类
# ============================================================

class MainExperiment:
    """Main experiment runner."""

    def __init__(self, config_path: str, mock_mode: bool = False,
                 max_samples: int = 0, primary_model: Optional[str] = None,
                 validation_models: Optional[list[str]] = None,
                 parallel_workers: Optional[int] = None,
                 variants_filter: Optional[str] = None,
                 skip_llm_only: bool = False,
                 skip_validation: bool = False,
                 skip_traditional: bool = False):
        self.config = self._load_config(config_path)
        self.mock_mode = mock_mode
        if self.mock_mode:
            logger.warning("=" * 60)
            logger.warning("MOCK MODE ENABLED — RESULTS ARE NOT REAL")
            logger.warning("DO NOT USE THESE RESULTS IN ANY PAPER OR REPORT")
            logger.warning("=" * 60)
        self.max_samples = max_samples
        # Parallel workers: read from config if not provided
        self.parallel_workers = parallel_workers or self.config.get("main_experiment", {}).get("parallel_workers", 1)
        logger.info(f"Parallel workers: {self.parallel_workers}")

        # 覆盖模型配置
        self.primary_model = primary_model or self.config.get("main_experiment", {}).get("primary_model", "qwen-plus")
        self.validation_models = validation_models or self.config.get("main_experiment", {}).get("validation_models", [])

        # Output directory: prefer main_experiment.output_dir, fall back to output.results_dir
        main_cfg = self.config.get("main_experiment", {})
        output_cfg = self.config.get("output", {})
        self.output_dir = project_root / main_cfg.get(
            "output_dir",
            output_cfg.get(
                "results_dir",
                str(exp_paths.MAIN_EXP_DIR.relative_to(exp_paths.PROJECT_ROOT)),
            ),
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logs directory (project-root relative)
        log_dir = project_root / output_cfg.get(
            "logs_dir",
            str(exp_paths.LOGS_DIR.relative_to(exp_paths.PROJECT_ROOT)),
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"main_experiment_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)

        self.variants_filter = variants_filter
        self.skip_llm_only = skip_llm_only
        self.skip_validation = skip_validation
        self.skip_traditional = skip_traditional

        # 数据目录
        self.data_dir = Path(__file__).parent / "data"

    def _load_config(self, config_path: str) -> dict:
        """Load YAML config from the experiment directory."""
        config_file = Path(__file__).parent / config_path
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def run(self):
        """Run the full main experiment."""
        print("=" * 70)
        print("ValueGuard Main Experiment")
        print("=" * 70)
        print(f"Primary model: {self.primary_model}")
        print(f"Validation models: {self.validation_models}")
        if self.mock_mode:
            print("!!! MOCK MODE !!! RESULTS ARE SIMULATED AND NOT FOR PUBLICATION")
        else:
            print("Real API mode: all predictions come from LLM calls on your data")
        print()

        # Step 1: Build benchmark
        print("[1/6] Building benchmark...")
        benchmark = self._build_benchmark()
        stats = BenchmarkBuilder(str(self.data_dir))
        stats.samples = benchmark
        benchmark_stats = stats.get_statistics()
        print(f"  Total: {benchmark_stats['total']} samples "
              f"({benchmark_stats['code']} code + {benchmark_stats['text']} text)")
        print(f"  Pipeline-capable: {benchmark_stats['pipeline_capable']}")
        print()

        # Step 2: Load profiles
        print("[2/6] Loading profiles...")
        profiles = self._load_profiles()
        print(f"  Loaded {len(profiles)} profiles: {list(profiles.keys())}")
        print()

        # Step 3: LLM-only baseline
        llm_only_results = []
        llm_metrics = None
        if not self.skip_llm_only:
            print("[3/6] Loading LLM-only baseline from IAA cache...")
            llm_only_results = self._run_llm_only_baseline(benchmark)
            llm_metrics = compute_unified_metrics(llm_only_results)
            self._print_metrics("LLM-only", llm_metrics)
            # 保存 LLM-only 结果（用于 inter-method Jaccard 矩阵）
            self._save_variant_results("LLM-only", llm_only_results, llm_metrics)
        else:
            print("[3/6] Skipped LLM-only baseline (--skip-llm-only)")
        print()

        # Step 3.5: 传统方法 baseline（TF-IDF / BM25 / BERT Zero-shot）
        traditional_results: dict[str, list[UnifiedSampleResult]] = {}
        if not self.skip_traditional:
            print("[3.5/6] Running traditional baselines (TF-IDF / BM25 / BERT)...")
            # 将 BenchmarkSample 转为 dict 供 traditional_baselines 使用
            bench_dicts = [{
                "sample_id": s.sample_id,
                "content": s.content,
                "scenario_type": s.scenario_type,
                "has_value_risk": s.has_value_risk,
                "ground_truth_values": list(s.ground_truth_values),
                "repo": s.repo,
            } for s in benchmark]
            benchmark_json_path = str(self.output_dir / "benchmark.json")
            traditional_results = run_traditional_baselines(
                benchmark_path=benchmark_json_path,
                output_dir=str(self.output_dir),
                methods=["tfidf", "bm25", "bert"],
            )
            for tname, tresults in traditional_results.items():
                tmetrics = compute_unified_metrics(tresults)
                self._print_metrics(tname, tmetrics)
                self._save_variant_results(tname, tresults, tmetrics)
        else:
            print("[3.5/6] Skipped traditional baselines (--skip-traditional)")
        print()

        # Step 4: Pipeline 变体（消融）
        print("[4/6] Running pipeline variants (ablation)...")
        ablation_config = self.config.get("main_experiment", {}).get("ablation", {})
        all_variants = ablation_config.get("variants", [])
        pipeline_results = {}

        # 按 --variants 过滤
        selected_variants = self._filter_variants(all_variants)
        if not selected_variants:
            print("  No variants selected. Use --list-variants to see available options.")
        else:
            print(f"  Selected {len(selected_variants)}/{len(all_variants)} variants: "
                  f"{[v['name'] for v in selected_variants]}")

        if ablation_config.get("enabled", True):
            for variant in selected_variants:
                name = variant["name"]
                profile_mode = variant.get("profile_mode", "real")
                skip_evidence = variant.get("skip_evidence", False)
                profile_alpha = variant.get("profile_alpha", 1.0)
                search_depth = variant.get("search_depth", 3)
                top_k = variant.get("top_k", 10)
                profile_threshold_mode = variant.get("profile_threshold_mode", "rank")
                profile_prompt_injection = variant.get("profile_prompt_injection", True)
                text_confidence_threshold = variant.get("text_confidence_threshold", 0.5)
                print(f"\n  --- {name} (profile={profile_mode}, skip_evidence={skip_evidence}, "
                      f"alpha={profile_alpha}, depth={search_depth}, top_k={top_k}, "
                      f"threshold_mode={profile_threshold_mode}, "
                      f"prompt_injection={profile_prompt_injection}, "
                      f"text_conf_threshold={text_confidence_threshold}) ---")

                evaluator = UnifiedPipelineEvaluator(
                    llm_provider=self.primary_model,
                    profile_map=profiles if profile_mode == "real" else None,
                    skip_evidence=skip_evidence,
                    profile_mode=profile_mode,
                    mock_mode=self.mock_mode,
                    repo_path=str(self.data_dir / "repos" / "Signal-Android"),
                    llm_configs=self.config.get("llm_models", {}),
                    cache_dir=str(self.output_dir / "pipeline_cache"),
                    profile_alpha=profile_alpha,
                    search_depth=search_depth,
                    top_k=top_k,
                    profile_threshold_mode=profile_threshold_mode,
                    profile_prompt_injection=profile_prompt_injection,
                    text_confidence_threshold=text_confidence_threshold,
                )

                results = evaluator.evaluate_all(
                    benchmark,
                    max_samples=self.max_samples if self.max_samples > 0 else None,
                    parallel_workers=self.parallel_workers,
                )

                # 从当前 benchmark 刷新 GT（pipeline 缓存可能保存了旧 GT）
                bench_map = {s.sample_id: s for s in benchmark}
                for r in results:
                    bs = bench_map.get(r.sample_id)
                    if bs:
                        r.ground_truth_has_risk = bs.has_value_risk
                        r.ground_truth_values = list(bs.ground_truth_values)

                pipeline_results[name] = results
                metrics = compute_unified_metrics(results)
                self._print_metrics(name, metrics)

                # 保存中间结果
                self._save_variant_results(name, results, metrics)
        print()

        # Step 5: 统计检验
        print("[5/6] Computing statistical tests...")
        stat_results = self._compute_statistical_tests(
            llm_only_results, pipeline_results, benchmark
        ) if llm_only_results and pipeline_results else {}
        print()

        # Step 6: 生成论文表格
        print("[6/6] Generating paper tables...")
        all_method_metrics = {}
        if llm_metrics:
            all_method_metrics["LLM-only"] = llm_metrics
        # 传统方法 baseline 纳入对比表
        for tname, tresults in traditional_results.items():
            all_method_metrics[tname] = compute_unified_metrics(tresults)
        for name, results in pipeline_results.items():
            all_method_metrics[name] = compute_unified_metrics(results)

        # 方案三：高质量标注子集（gt_label_count >= 2 的 text 样本）
        # 筛选逻辑：只保留有至少2个GT value的 text 场景样本
        hq_sample_ids = {
            s.sample_id for s in benchmark
            if s.scenario_type == "text" and s.gt_label_count >= 2
        }
        logger.info(f"  [Scheme-3] High-quality text subset: {len(hq_sample_ids)} samples "
                    f"(gt_label_count>=2, out of "
                    f"{sum(1 for s in benchmark if s.scenario_type=='text')} text samples)")

        hq_method_metrics = {}
        if hq_sample_ids:
            hq_llm = [r for r in llm_only_results if r.sample_id in hq_sample_ids]
            if hq_llm:
                hq_method_metrics["LLM-only"] = compute_unified_metrics(hq_llm)
            for name, results in pipeline_results.items():
                hq_results = [r for r in results if r.sample_id in hq_sample_ids]
                if hq_results:
                    hq_method_metrics[name] = compute_unified_metrics(hq_results)

        self._generate_paper_tables(all_method_metrics, stat_results, hq_method_metrics)
        print()

        # 跨模型验证（如果非 mock 模式且未跳过）
        if not self.mock_mode and self.validation_models and not self.skip_validation:
            print("[BONUS] Cross-model validation...")
            self._run_cross_model_validation(benchmark, profiles)
        elif self.skip_validation:
            print("[BONUS] Skipped cross-model validation (--skip-validation)")

        # 汇总保存
        self._save_final_results(all_method_metrics, stat_results, benchmark_stats)
        print("\nExperiment completed! Results saved to:", self.output_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_variants(self, all_variants: list[dict]) -> list[dict]:
        """Filter ablation variants by the --variants argument.

        Supported formats:
          --variants 1,3,5        Select by 1-based row numbers
          --variants 1-3,5        Select by ranges
          --variants Full,w/o     Select by name substring (comma-separated)
          --variants all          Run all
          (default)               Run all
        """
        if not self.variants_filter:
            return all_variants

        filt = self.variants_filter.strip()
        if filt.lower() == "all":
            return all_variants

        selected = []
        import re
        if re.match(r'^[\d,\-\s]+$', filt):
            # Row-number mode
            indices = set()
            for part in filt.split(","):
                part = part.strip()
                if "-" in part:
                    lo, hi = part.split("-", 1)
                    indices.update(range(int(lo), int(hi) + 1))
                else:
                    indices.add(int(part))
            for idx in sorted(indices):
                if 1 <= idx <= len(all_variants):
                    selected.append(all_variants[idx - 1])
                else:
                    print(f"  Warning: row {idx} out of range (1-{len(all_variants)})")
        else:
            # Name substring matching
            keywords = [k.strip().lower() for k in filt.split(",") if k.strip()]
            for v in all_variants:
                name_lower = v["name"].lower()
                if any(k in name_lower for k in keywords):
                    selected.append(v)

        return selected

    @staticmethod
    def list_variants(config_path: str):
        """List available ablation variants with row numbers."""
        import yaml
        config_path_full = Path(__file__).parent / config_path
        with open(config_path_full) as f:
            config = yaml.safe_load(f)
        variants = config.get("main_experiment", {}).get("ablation", {}).get("variants", [])
        print(f"{len(variants)} ablation variants:\n")
        print(f"{'#':>4s}  {'Name':40s} {'Profile':8s} {'SkipEv':8s} {'Alpha':6s} {'ThMode':10s} {'Injection':10s}")
        print("-" * 95)
        for i, v in enumerate(variants, 1):
            print(f"  {i:2d}.  {v['name']:40s} "
                  f"{v.get('profile_mode','real'):8s} {str(v.get('skip_evidence',False)):8s} "
                  f"{str(v.get('profile_alpha',1.0)):6s} {v.get('profile_threshold_mode','rank'):10s} "
                  f"{str(v.get('profile_prompt_injection',True)):10s}")
        print(f"\nUsage:")
        print(f"  --variants 1,3,5        Select by row numbers")
        print(f"  --variants 1-4,6        Select by ranges")
        print(f"  --variants Full,w/o     Select by name substring")
        print(f"  --variants all          Run all")

    def _build_benchmark(self) -> list[BenchmarkSample]:
        """Build the unified benchmark."""
        builder = BenchmarkBuilder(str(self.data_dir))

        # Build from config if datasets are specified, otherwise use legacy discovery
        datasets_config = self.config.get("datasets", {})
        if datasets_config:
            builder.build_from_config(datasets_config)
        else:
            builder.build_all()

        # Optionally override generated GT with LLM annotations
        iaa_cfg = self.config.get("main_experiment", {}).get("iaa", {})
        llm_gt_enabled = iaa_cfg.get("llm_gt_override", True)
        if llm_gt_enabled:
            iaa_cache_dir = iaa_cfg.get(
                "cache_dir",
                str(exp_paths.LLM_OUTPUTS_DIR.relative_to(exp_paths.PROJECT_ROOT)),
            )
            iaa_cache_path = project_root / iaa_cache_dir
            if not iaa_cache_path.exists():
                iaa_cache_path = Path(__file__).parent / iaa_cache_dir
            overridden = builder.override_gt_with_llm(
                str(iaa_cache_path), self.primary_model
            )
            print(f"  [LLM GT] Overrode GT for {overridden} generated samples")

        samples = builder.get_samples()
        if self.max_samples > 0:
            samples = samples[:self.max_samples]

        # 保存 benchmark
        save_benchmark(samples, str(self.output_dir / "benchmark.json"))
        return samples

    def _load_profiles(self) -> dict[str, ValueProfile]:
        """Load repository profiles."""
        profile_cfg = self.config.get("main_experiment", {}).get("profile", {})
        cache_dir = profile_cfg.get(
            "cache_dir",
            str(exp_paths.PROFILE_CACHE_DIR.relative_to(exp_paths.PROJECT_ROOT)),
        )
        cache_path = project_root / cache_dir

        profiles = load_profiles_from_cache(str(cache_path), self.primary_model)

        if self.mock_mode and not profiles:
            # Mock 模式下创建假 profile
            for repo in ["Signal-Android", "focus-android", "kubernetes", "git", "openclaw"]:
                profiles[repo] = ValueProfile(
                    repo=repo,
                    l2_scores={"HV9": 0.9, "HV10": 0.8, "HV6": 0.7,
                               **{f"HV{i}": 0.4 for i in [1, 2, 3, 4, 5, 7, 8]}},
                    l3_scores={"SV1": 0.8, "SV2": 0.7, "SV5": 0.7,
                               **{f"SV{i}": 0.4 for i in [3, 4, 6, 7, 8, 9, 10]}},
                    core_values=["HV9", "HV10", "HV6"],
                )

        return profiles

    def _run_llm_only_baseline(self, benchmark: list[BenchmarkSample]) -> list[UnifiedSampleResult]:
        """Run LLM-only baseline loaded from the IAA cache."""
        llm_cfg = self.config.get("main_experiment", {}).get("llm_only", {})
        iaa_cache_dir = llm_cfg.get(
            "iaa_cache_dir",
            str(exp_paths.LLM_OUTPUTS_DIR.relative_to(exp_paths.PROJECT_ROOT)),
        )
        iaa_cache_path = project_root / iaa_cache_dir
        if not iaa_cache_path.exists():
            iaa_cache_path = Path(__file__).parent / iaa_cache_dir

        sample_ids = [s.sample_id for s in benchmark]
        predictions = load_llm_only_baseline(
            str(iaa_cache_path), self.primary_model, sample_ids
        )

        return llm_only_to_unified_results(predictions, benchmark)

    def _compute_statistical_tests(
        self,
        llm_only_results: list[UnifiedSampleResult],
        pipeline_results: dict[str, list[UnifiedSampleResult]],
        benchmark: list[BenchmarkSample],
    ) -> dict:
        """计算统计检验"""
        stat_results = {}

        gt_risks = [s.has_value_risk for s in benchmark]
        gt_values = [set(s.ground_truth_values) for s in benchmark]

        # LLM-only baseline 数据
        llm_risks = [r.predicted_has_risk for r in llm_only_results]
        llm_jaccards = []
        for r, bs in zip(llm_only_results, benchmark):
            pred_set = set(r.predicted_values)
            gt_set = set(bs.ground_truth_values)
            if pred_set or gt_set:
                llm_jaccards.append(len(pred_set & gt_set) / len(pred_set | gt_set))
            else:
                llm_jaccards.append(1.0)

        # 与每个 Pipeline 变体对比
        for name, results in pipeline_results.items():
            pipe_risks = [r.predicted_has_risk for r in results]
            pipe_jaccards = []
            for r, bs in zip(results, benchmark):
                pred_set = set(r.predicted_values)
                gt_set = set(bs.ground_truth_values)
                if pred_set or gt_set:
                    pipe_jaccards.append(len(pred_set & gt_set) / len(pred_set | gt_set))
                else:
                    pipe_jaccards.append(1.0)

            # 确保长度一致
            n = min(len(gt_risks), len(llm_risks), len(pipe_risks))
            comparison = compare_methods(
                gt_risks[:n], llm_risks[:n], pipe_risks[:n],
                llm_jaccards[:n], pipe_jaccards[:n],
                method_a_name="LLM-only",
                method_b_name=name,
            )
            stat_results[f"LLM-only vs {name}"] = comparison

        # Pipeline 变体之间对比（消融）
        variant_names = list(pipeline_results.keys())
        for i in range(len(variant_names)):
            for j in range(i + 1, len(variant_names)):
                name_a, name_b = variant_names[i], variant_names[j]
                res_a = pipeline_results[name_a]
                res_b = pipeline_results[name_b]
                risks_a = [r.predicted_has_risk for r in res_a]
                risks_b = [r.predicted_has_risk for r in res_b]

                n = min(len(gt_risks), len(risks_a), len(risks_b))
                jaccards_a = []
                jaccards_b = []
                for ra, rb, bs in zip(res_a, res_b, benchmark):
                    gt_set = set(bs.ground_truth_values)
                    for r, jac_list in [(ra, jaccards_a), (rb, jaccards_b)]:
                        pred_set = set(r.predicted_values)
                        if pred_set or gt_set:
                            jac_list.append(len(pred_set & gt_set) / len(pred_set | gt_set))
                        else:
                            jac_list.append(1.0)

                comparison = compare_methods(
                    gt_risks[:n], risks_a[:n], risks_b[:n],
                    jaccards_a[:n], jaccards_b[:n],
                    method_a_name=name_a,
                    method_b_name=name_b,
                )
                stat_results[f"{name_a} vs {name_b}"] = comparison

        return stat_results

    def _run_cross_model_validation(
        self,
        benchmark: list[BenchmarkSample],
        profiles: dict[str, ValueProfile],
    ):
        """Run cross-model validation on the validation model set."""
        for model_key in self.validation_models:
            logger.info(f"  Cross-model validation: {model_key}")

            # LLM-only baseline
            llm_cfg = self.config.get("main_experiment", {}).get("llm_only", {})
            iaa_cache_dir = llm_cfg.get(
                "iaa_cache_dir",
                str(exp_paths.LLM_OUTPUTS_DIR.relative_to(exp_paths.PROJECT_ROOT)),
            )
            iaa_cache_path = project_root / iaa_cache_dir
            if not iaa_cache_path.exists():
                iaa_cache_path = Path(__file__).parent / iaa_cache_dir

            sample_ids = [s.sample_id for s in benchmark]
            predictions = load_llm_only_baseline(str(iaa_cache_path), model_key, sample_ids)
            llm_results = llm_only_to_unified_results(predictions, benchmark)
            llm_metrics = compute_unified_metrics(llm_results)

            # Full Pipeline
            evaluator = UnifiedPipelineEvaluator(
                llm_provider=model_key,
                profile_map=profiles,
                skip_evidence=False,
                profile_mode="real",
                mock_mode=False,
                repo_path=str(self.data_dir / "repos" / "Signal-Android"),
                llm_configs=self.config.get("llm_models", {}),
                cache_dir=str(self.output_dir / "pipeline_cache"),
            )

            pipeline_results = evaluator.evaluate_all(
                benchmark,
                max_samples=self.max_samples if self.max_samples > 0 else None,
                parallel_workers=self.parallel_workers,
            )
            pipe_metrics = compute_unified_metrics(pipeline_results)

            # 保存
            self._save_variant_results(f"validation_{model_key}_llm_only", llm_results, llm_metrics)
            self._save_variant_results(f"validation_{model_key}_pipeline", pipeline_results, pipe_metrics)

    def _watermark(self, content: str) -> str:
        if not self.mock_mode:
            return content
        marker = (
            "\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "!!! WARNING: THIS OUTPUT WAS GENERATED IN MOCK MODE                !!!\n"
            "!!! IT DOES NOT REFLECT REAL LLM PERFORMANCE AND MUST NOT BE USED  !!!\n"
            "!!! FOR PUBLICATION, DECISION-MAKING, OR PAPER SUBMISSION          !!!\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "\n"
        )
        return marker + content
    def _generate_paper_tables(self, all_metrics: dict, stat_results: dict,
                               hq_metrics: dict = None):
        """生成论文级 LaTeX 和 Markdown 表格"""
        hq_metrics = hq_metrics or {}

        # Table 1: 主对比表
        md_table = self._watermark(self._generate_main_comparison_md(all_metrics))
        (self.output_dir / "table1_main_comparison.md").write_text(md_table, encoding="utf-8")

        latex_table = self._watermark(self._generate_main_comparison_latex(all_metrics))
        (self.output_dir / "table1_main_comparison.tex").write_text(latex_table, encoding="utf-8")

        # Table 2: 消融表
        ablation_md = self._watermark(self._generate_ablation_md(all_metrics))
        (self.output_dir / "table2_ablation.md").write_text(ablation_md, encoding="utf-8")

        ablation_tex = self._watermark(self._generate_ablation_latex(all_metrics))
        (self.output_dir / "table2_ablation.tex").write_text(ablation_tex, encoding="utf-8")

        # Table 3: 统计检验表
        stat_md = self._watermark(self._generate_stat_tests_md(stat_results))
        (self.output_dir / "table3_statistical_tests.md").write_text(stat_md, encoding="utf-8")

        # Table 4: Value Identification 一致性表（方案一：全量 + 新指标；方案三：HQ子集）
        value_id_md = self._watermark(self._generate_value_id_detail_md(all_metrics, hq_metrics))
        (self.output_dir / "table4_value_id_detail.md").write_text(value_id_md, encoding="utf-8")

        value_id_tex = self._watermark(self._generate_value_id_detail_latex(all_metrics, hq_metrics))
        (self.output_dir / "table4_value_id_detail.tex").write_text(value_id_tex, encoding="utf-8")

        print(f"  Tables saved to {self.output_dir}/")

    # ------------------------------------------------------------------
    # 表格生成
    # ------------------------------------------------------------------

    def _generate_main_comparison_md(self, all_metrics: dict) -> str:
        """生成 Markdown 主对比表"""
        lines = [
            "# Table 1: Main Comparison — Value Risk Identification Performance",
            "",
            f"Primary model: {self.primary_model}",
            "",
            "| Method | Scenario | N | Prec. | Rec. | F1 | κ | Jaccard | Sym-F1 | Micro-F1 |",
            "|--------|----------|---|-------|------|----|----|---------|--------|----------|",
        ]

        for method_name, metrics in all_metrics.items():
            for scenario in ["code", "text", "overall"]:
                m = metrics.get(scenario, {})
                if not m:
                    continue
                d1 = m.get("dim1_risk_detection", {})
                d2 = m.get("dim2_value_identification", {})
                lines.append(
                    f"| {method_name} | {scenario} | {m.get('n', 0)} "
                    f"| {d1.get('precision', 0):.3f} | {d1.get('recall', 0):.3f} "
                    f"| {d1.get('f1', 0):.3f} | {d1.get('cohen_kappa', 0):.3f} "
                    f"| {d2.get('pairwise_jaccard', 0):.3f} | {d2.get('symmetric_f1', 0):.3f} "
                    f"| {d2.get('micro_f1', 0):.3f} |"
                )

        return "\n".join(lines)

    def _generate_main_comparison_latex(self, all_metrics: dict) -> str:
        """生成 LaTeX 主对比表"""
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Value risk identification performance: Human agreement across method groups"
            f" (primary model: {self.primary_model}). Best results in \\textbf{{bold}}.}}",
            r"\label{tab:main-comparison}",
            r"\begin{tabular}{l l r cccc ccc}",
            r"\toprule",
            r"& & & \multicolumn{4}{c}{\textbf{Dim 1: Risk Detection}}"
            r" & \multicolumn{3}{c}{\textbf{Dim 2: Value ID}} \\",
            r"\cmidrule(lr){4-7}\cmidrule(lr){8-10}",
            r"\textbf{Method} & \textbf{Scenario} & \textbf{N}"
            r" & Prec. & Rec. & F1 & $\kappa$"
            r" & Jaccard & Sym-F1 & Micro-F1 \\",
            r"\midrule",
        ]

        for method_name, metrics in all_metrics.items():
            first = True
            for scenario in ["code", "text", "overall"]:
                m = metrics.get(scenario, {})
                if not m:
                    continue
                d1 = m.get("dim1_risk_detection", {})
                d2 = m.get("dim2_value_identification", {})
                method_label = method_name if first else ""
                first = False
                lines.append(
                    f"{method_label} & {scenario} & {m.get('n', 0)} "
                    f"& {d1.get('precision', 0):.3f} & {d1.get('recall', 0):.3f} "
                    f"& {d1.get('f1', 0):.3f} & {d1.get('cohen_kappa', 0):.3f} "
                    f"& {d2.get('pairwise_jaccard', 0):.3f} & {d2.get('symmetric_f1', 0):.3f} "
                    f"& {d2.get('micro_f1', 0):.3f} \\\\"
                )
            if method_name != list(all_metrics.keys())[-1]:
                lines.append(r"\midrule")

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
        return "\n".join(lines)

    def _generate_ablation_md(self, all_metrics: dict) -> str:
        """生成消融表 Markdown：所有 pipeline 变体相对 ValueGuard (Full) 的 Δ"""
        lines = [
            "# Table 2: Ablation Study — Component Contribution",
            "",
            "Values show $\\Delta$ = ValueGuard (Full) − variant. Positive means the full system outperforms the ablated variant.",
            "",
            "| Variant | Scenario | ΔF1 | Δκ | ΔJaccard | ΔSym-F1 |",
            "|---------|----------|-----|----|----------|---------|",
        ]

        full_key = "ValueGuard (Full)"
        full_metrics = all_metrics.get(full_key, {})

        for variant_name, variant_metrics in all_metrics.items():
            if variant_name == full_key or variant_name == "LLM-only":
                continue
            for scenario in ["code", "text", "overall"]:
                full_m = full_metrics.get(scenario, {})
                var_m = variant_metrics.get(scenario, {})
                if not full_m or not var_m:
                    continue

                d1_full = full_m.get("dim1_risk_detection", {})
                d1_var = var_m.get("dim1_risk_detection", {})
                d2_full = full_m.get("dim2_value_identification", {})
                d2_var = var_m.get("dim2_value_identification", {})

                delta_f1 = d1_full.get("f1", 0) - d1_var.get("f1", 0)
                delta_kappa = d1_full.get("cohen_kappa", 0) - d1_var.get("cohen_kappa", 0)
                delta_jaccard = d2_full.get("pairwise_jaccard", 0) - d2_var.get("pairwise_jaccard", 0)
                delta_sym_f1 = d2_full.get("symmetric_f1", 0) - d2_var.get("symmetric_f1", 0)

                lines.append(
                    f"| {variant_name} | {scenario} "
                    f"| {delta_f1:+.4f} | {delta_kappa:+.4f} "
                    f"| {delta_jaccard:+.4f} | {delta_sym_f1:+.4f} |"
                )

        return "\n".join(lines)

    def _generate_ablation_latex(self, all_metrics: dict) -> str:
        """生成消融表 LaTeX：所有 pipeline 变体相对 ValueGuard (Full)"""
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Ablation study: contribution of pipeline components and hyperparameters"
            rf" ({self.primary_model}). $\Delta$ = ValueGuard (Full) $-$ variant. Positive values indicate the full system benefits from the component/setting.}}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{l l r r r r}",
            r"\toprule",
            r"& & \multicolumn{2}{c}{\textbf{Dim 1}}"
            r" & \multicolumn{2}{c}{\textbf{Dim 2}} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            r"\textbf{Variant} & \textbf{Scenario}"
            r" & $\Delta$F1 & $\Delta\kappa$ & $\Delta$Jac. & $\Delta$Sym-F1 \\",
            r"\midrule",
        ]
    
        full_key = "ValueGuard (Full)"
        full_metrics = all_metrics.get(full_key, {})
    
        for variant_name, variant_metrics in all_metrics.items():
            if variant_name == full_key or variant_name == "LLM-only":
                continue
            for scenario in ["code", "text", "overall"]:
                full_m = full_metrics.get(scenario, {})
                var_m = variant_metrics.get(scenario, {})
                if not full_m or not var_m:
                    continue
    
                d1f = full_m.get("dim1_risk_detection", {})
                d1v = var_m.get("dim1_risk_detection", {})
                d2f = full_m.get("dim2_value_identification", {})
                d2v = var_m.get("dim2_value_identification", {})
    
                df = d1f.get("f1", 0) - d1v.get("f1", 0)
                dk = d1f.get("cohen_kappa", 0) - d1v.get("cohen_kappa", 0)
                dj = d2f.get("pairwise_jaccard", 0) - d2v.get("pairwise_jaccard", 0)
                ds = d2f.get("symmetric_f1", 0) - d2v.get("symmetric_f1", 0)
    
                def fmt_delta(v):
                    s = "+" if v >= 0 else ""
                    b0 = r"\textbf{" if v > 0 else ""
                    b1 = "}" if v > 0 else ""
                    return f"{b0}{s}{v:.4f}{b1}"
    
                # LaTeX 安全化变体名称
                safe_name = variant_name.replace("&", "\\&")
                lines.append(
                    f"{safe_name} & {scenario} "
                    f"& {fmt_delta(df)} & {fmt_delta(dk)} "
                    f"& {fmt_delta(dj)} & {fmt_delta(ds)} \\\\"
                )
    
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
        return "\n".join(lines)
    
    def _generate_value_id_detail_md(self, all_metrics: dict,
                                    hq_metrics: dict = None) -> str:
        """生成 Table 4: Value Identification 一致性表格

        方案一：全量数据 + 新增 Recall-of-GT / Precision-of-GT / Partial-Credit-Jaccard
        方案三：高质量标注子集（gt_label_count>=2）对比
        设计思路：
          4a. 全量：各方法 vs Human Annotation（含方案一新指标）
          4b. Code vs Text 分层（方案一）
          4c. 方案三：高质量子集 vs Human Annotation
          4d. 方法间 Jaccard 矩阵（全量）
        """
        hq_metrics = hq_metrics or {}
        method_names = list(all_metrics.keys())

        # --- 4a: 全量 vs Human（含方案一新指标）---
        lines = [
            "# Table 4: Value Identification Agreement with Human Annotation",
            "",
            "**Scheme 1 (Full benchmark)**: All metrics compare predicted value sets against",
            "ground-truth labels. New metrics (Recall-of-GT, Prec-of-GT, PC-Jaccard) are designed",
            "for sparse GT annotations where only the most salient values are labeled.",
            "",
            "| Metric | Description |",
            "|--------|-------------|",
            "| Jaccard | Standard: |pred∩gt| / |pred∪gt| |",
            "| Sym-F1 | 2|pred∩gt| / (|pred|+|gt|) |",
            "| **Recall-of-GT** | |pred∩gt| / |gt| — covers annotated values? |",
            "| **Prec-of-GT** | |pred∩gt| / |pred| — no spurious values? |",
            "| **PC-Jaccard** | |pred∩gt| / max(|pred|,|gt|) — sparse-GT fair |",
            "",
            "## 4a. Full Benchmark — All Methods vs. Human Annotation",
            "",
            "| Method | N | Jaccard | Sym-F1 | Recall-of-GT | Prec-of-GT | PC-Jaccard | Micro-F1 |",
            "|--------|---|---------|--------|--------------|------------|------------|----------|",
        ]
        for m in method_names:
            d2 = all_metrics[m].get("overall", {}).get("dim2_value_identification", {})
            n  = all_metrics[m].get("overall", {}).get("n", 0)
            lines.append(
                f"| {m} | {n} "
                f"| {d2.get('pairwise_jaccard', 0):.3f} "
                f"| {d2.get('symmetric_f1', 0):.3f} "
                f"| **{d2.get('recall_of_gt', 0):.3f}** "
                f"| {d2.get('precision_of_gt', 0):.3f} "
                f"| {d2.get('partial_credit_jaccard', 0):.3f} "
                f"| {d2.get('f1', 0):.3f} |"
            )

        # --- 4b: Code vs Text 分层 ---
        lines.extend([
            "",
            "## 4b. Code vs. Text Breakdown (Scheme 1)",
            "",
            "| Method | Scenario | N | Jaccard | Recall-of-GT | PC-Jaccard |",
            "|--------|----------|---|---------|--------------|------------|",
        ])
        for m in method_names:
            for scenario in ["code", "text"]:
                sm = all_metrics[m].get(scenario, {})
                if not sm:
                    continue
                d2 = sm.get("dim2_value_identification", {})
                lines.append(
                    f"| {m} | {scenario} | {sm.get('n', 0)} "
                    f"| {d2.get('pairwise_jaccard', 0):.3f} "
                    f"| **{d2.get('recall_of_gt', 0):.3f}** "
                    f"| {d2.get('partial_credit_jaccard', 0):.3f} |"
                )

        # --- 4c: 方案三：高质量子集 ---
        lines.extend([
            "",
            "## 4c. High-Quality Annotation Subset (Scheme 3: gt_label_count ≥ 2)",
            "",
            "Filtering rationale: the original dataset uses a *minimal labeling protocol*",
            "(annotators label only the most salient value). Samples with ≥2 GT labels",
            "have richer annotations, providing a more reliable Dim2 evaluation.",
            "",
        ])
        if hq_metrics:
            hq_methods = list(hq_metrics.keys())
            lines.extend([
                "| Method | N | Jaccard | Sym-F1 | Recall-of-GT | Prec-of-GT | PC-Jaccard |",
                "|--------|---|---------|--------|--------------|------------|------------|",
            ])
            for m in hq_methods:
                d2 = hq_metrics[m].get("overall", {}).get("dim2_value_identification", {})
                n  = hq_metrics[m].get("overall", {}).get("n", 0)
                if n == 0:
                    continue
                lines.append(
                    f"| {m} | {n} "
                    f"| {d2.get('pairwise_jaccard', 0):.3f} "
                    f"| {d2.get('symmetric_f1', 0):.3f} "
                    f"| **{d2.get('recall_of_gt', 0):.3f}** "
                    f"| {d2.get('precision_of_gt', 0):.3f} "
                    f"| {d2.get('partial_credit_jaccard', 0):.3f} |"
                )
        else:
            lines.append("*(No text samples with gt_label_count ≥ 2 in current benchmark)*")

        # --- 4d: 方法间 Jaccard 矩阵 ---
        lines.extend([
            "",
            "## 4d. Inter-Method Jaccard Matrix (Predicted Value Sets, Full Benchmark)",
            "",
            "Shows pairwise agreement *between methods* (not vs. human).",
            "Higher = more similar value detection behavior.",
            "",
        ])
        method_pred_sets = {}
        for m in method_names:
            safe = m.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            results_file = self.output_dir / f"{safe}_results.json"
            if results_file.exists():
                try:
                    raw = json.loads(results_file.read_text(encoding="utf-8"))
                    method_pred_sets[m] = [set(r.get("predicted_values", [])) for r in raw]
                except Exception:
                    method_pred_sets[m] = []
            else:
                method_pred_sets[m] = []

        header = "| | " + " | ".join(method_names) + " |"
        sep    = "|---|" + "---|" * len(method_names)
        lines.extend([header, sep])
        for ma in method_names:
            row_cells = [f"**{ma}**"]
            sets_a = method_pred_sets.get(ma, [])
            for mb in method_names:
                if ma == mb:
                    row_cells.append("**1.000**")
                    continue
                sets_b = method_pred_sets.get(mb, [])
                n = min(len(sets_a), len(sets_b))
                if n == 0:
                    row_cells.append("-")
                    continue
                scores = []
                for sa, sb in zip(sets_a[:n], sets_b[:n]):
                    if not sa and not sb:
                        scores.append(1.0)
                    elif not sa or not sb:
                        scores.append(0.0)
                    else:
                        scores.append(len(sa & sb) / len(sa | sb))
                row_cells.append(f"{sum(scores)/len(scores):.3f}")
            lines.append("| " + " | ".join(row_cells) + " |")

        return "\n".join(lines)

    def _generate_value_id_detail_latex(self, all_metrics: dict,
                                        hq_metrics: dict = None) -> str:
        """生成 Table 4 LaTeX 版本（方案一全量 + 方案三子集）"""
        hq_metrics = hq_metrics or {}
        method_names = list(all_metrics.keys())

        # --- 4a: 全量 vs Human（方案一所有指标）---
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Value identification agreement with human annotation (full benchmark)."
            r" Recall-of-GT, Precision-of-GT, and PC-Jaccard are designed for sparse GT"
            r" annotations where only the most salient values are labeled."
            r" All metrics compare predicted sets against ground-truth.}",
            r"\label{tab:value-id-full}",
            r"\begin{tabular}{l r ccc cc c}",
            r"\toprule",
            r" & & \multicolumn{3}{c}{\textbf{Standard}} & \multicolumn{2}{c}{\textbf{Sparse-GT aware}} & \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
            r"\textbf{Method} & \textbf{N} & Jaccard & Sym-F1 & Micro-F1 & Recall-of-GT & PC-Jaccard & Prec-of-GT \\",
            r"\midrule",
        ]
        for m in method_names:
            d2 = all_metrics[m].get("overall", {}).get("dim2_value_identification", {})
            n  = all_metrics[m].get("overall", {}).get("n", 0)
            lines.append(
                f"{m} & {n} "
                f"& {d2.get('pairwise_jaccard',0):.3f} "
                f"& {d2.get('symmetric_f1',0):.3f} "
                f"& {d2.get('f1',0):.3f} "
                f"& \\textbf{{{d2.get('recall_of_gt',0):.3f}}} "
                f"& {d2.get('partial_credit_jaccard',0):.3f} "
                f"& {d2.get('precision_of_gt',0):.3f} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])

        # --- 4c: 方案三子集 ---
        if hq_metrics:
            hq_methods = [m for m in hq_metrics if hq_metrics[m].get("overall", {}).get("n", 0) > 0]
            if hq_methods:
                lines.extend([
                    "",
                    r"\begin{table}[t]",
                    r"\centering",
                    r"\caption{Value identification agreement on the high-quality annotation subset"
                    r" (gt\_label\_count $\geq$ 2). Samples with at least two GT-labeled values"
                    r" provide richer supervision for Dim2 evaluation.}",
                    r"\label{tab:value-id-hq}",
                    r"\begin{tabular}{l r ccc cc}",
                    r"\toprule",
                    r"\textbf{Method} & \textbf{N} & Jaccard & Sym-F1 & Recall-of-GT & PC-Jaccard & Prec-of-GT \\",
                    r"\midrule",
                ])
                for m in hq_methods:
                    d2 = hq_metrics[m].get("overall", {}).get("dim2_value_identification", {})
                    n  = hq_metrics[m].get("overall", {}).get("n", 0)
                    lines.append(
                        f"{m} & {n} "
                        f"& {d2.get('pairwise_jaccard',0):.3f} "
                        f"& {d2.get('symmetric_f1',0):.3f} "
                        f"& \\textbf{{{d2.get('recall_of_gt',0):.3f}}} "
                        f"& {d2.get('partial_credit_jaccard',0):.3f} "
                        f"& {d2.get('precision_of_gt',0):.3f} \\\\"
                    )
                lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        return "\n".join(lines)

    def _generate_stat_tests_md(self, stat_results: dict) -> str:
        """生成统计检验结果 Markdown"""
        lines = [
            "# Table 3: Statistical Tests",
            "",
            "| Comparison | Dim1 (McNemar p) | Dim2 (Wilcoxon p) | Cohen's h | Rank-biserial r |",
            "|------------|------------------|-------------------|-----------|-----------------|",
        ]
        for comparison_name, result in stat_results.items():
            mcnemar_p = result.get("dim1_mcnemar", {}).get("p_value", 1.0)
            wilcoxon_p = result.get("dim2_wilcoxon", {}).get("p_value", 1.0)
            cohens_h = result.get("effect_size_cohens_h", 0)
            rank_r = result.get("effect_size_rank_biserial", 0)
            sig = "*" if mcnemar_p < 0.05 or wilcoxon_p < 0.05 else ""
            lines.append(
                f"| {comparison_name} | {mcnemar_p:.4f}{sig} | {wilcoxon_p:.4f}{sig} "
                f"| {cohens_h:.4f} | {rank_r:.4f} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _print_metrics(self, method_name: str, metrics: dict):
        """打印指标摘要"""
        for scenario in ["code", "text", "overall"]:
            m = metrics.get(scenario, {})
            if not m:
                continue
            d1 = m.get("dim1_risk_detection", {})
            d2 = m.get("dim2_value_identification", {})
            logger.info(
                f"  [{method_name} | {scenario}] "
                f"F1={d1.get('f1', 0):.3f} κ={d1.get('cohen_kappa', 0):.3f} "
                f"Jac={d2.get('pairwise_jaccard', 0):.3f} Sym-F1={d2.get('symmetric_f1', 0):.3f}"
            )

    def _save_variant_results(self, variant_name: str, results: list[UnifiedSampleResult], metrics: dict):
        """保存单个变体的结果"""
        safe_name = variant_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        results_path = self.output_dir / f"{safe_name}_results.json"
        metrics_path = self.output_dir / f"{safe_name}_metrics.json"

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    def _save_final_results(self, all_metrics: dict, stat_results: dict, benchmark_stats: dict):
        """保存最终汇总结果"""
        final = {
            "experiment_info": {
                "primary_model": self.primary_model,
                "validation_models": self.validation_models,
                "mock_mode": self.mock_mode,
                "timestamp": datetime.now().isoformat(),
            },
            "benchmark_stats": benchmark_stats,
            "all_method_metrics": all_metrics,
            "statistical_tests": stat_results,
        }

        with open(self.output_dir / "final_results.json", "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, ensure_ascii=False, default=str)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ValueGuard Main Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the unified experiment config file",
    )
    # NOTE: --mock 已移除。主实验必须运行真实 LLM。如需 smoke test，请使用单元测试。
    parser.add_argument("--max-samples", type=int, default=0, help="最大样本数（0=全量）")
    parser.add_argument("--primary-model", default=None, help="覆盖主力模型")
    parser.add_argument("--validation-models", nargs="*", default=None, help="覆盖验证模型列表")
    parser.add_argument("--output-dir", default=None, help="覆盖输出目录")
    parser.add_argument("--parallel-workers", type=int, default=None,
                        help="并行工作线程数（覆盖配置文件中的 parallel_workers）")
    parser.add_argument("--variants", default=None,
                        help="选择运行的消融变体。格式：行号(1,3,5)、范围(1-4)、名称子串(Full,w/o)、或 all")
    parser.add_argument("--list-variants", action="store_true",
                        help="列出所有可用的消融变体及其行号，然后退出")
    parser.add_argument("--skip-llm-only", action="store_true",
                        help="跳过 LLM-only baseline（已有缓存时使用）")
    parser.add_argument("--skip-validation", action="store_true",
                        help="跳过跨模型验证")
    parser.add_argument("--skip-traditional", action="store_true",
                        help="跳过传统方法 baseline（TF-IDF / BM25 / BERT）")
    args = parser.parse_args()

    if args.list_variants:
        MainExperiment.list_variants(args.config)
        return

    experiment = MainExperiment(
        config_path=args.config,
        max_samples=args.max_samples,
        primary_model=args.primary_model,
        validation_models=args.validation_models,
        parallel_workers=args.parallel_workers,
        variants_filter=args.variants,
        skip_llm_only=args.skip_llm_only,
        skip_validation=args.skip_validation,
        skip_traditional=args.skip_traditional,
    )

    if args.output_dir:
        experiment.output_dir = Path(args.output_dir)
        experiment.output_dir.mkdir(parents=True, exist_ok=True)

    experiment.run()


if __name__ == "__main__":
    main()