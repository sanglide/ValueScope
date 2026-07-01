#!/usr/bin/env python
"""
Value Profile Evaluation orchestrator.

Runs three evaluation experiments and produces a combined report.

Experiments:
  1. Profile Characterization (case study)
  2. Cross-Model Consistency
  3. Downstream Impact / Ablation (reuses the IAA dataset)

Usage:
    python -m experiment.profile_experiment.run_all --experiments all
    python -m experiment.profile_experiment.run_all --experiments 1,2
    python -m experiment.profile_experiment.run_all --experiments 1 --force-api
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Project paths
project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import yaml
from experiment.llm_client import LLMClientFactory
from experiment.data_loader import ValueModelLoader
from experiment.profile_experiment.profile_generator import ProfileGenerator
from experiment.profile_experiment.profile_visualizer import ProfileVisualizer
from experiment import paths as exp_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("profile_experiment")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, rel: str) -> str:
    """Resolve a relative path against a base directory."""
    p = base / rel
    return str(p)


def main():
    parser = argparse.ArgumentParser(description="Value Profile Evaluation Experiments")
    parser.add_argument(
        "--config",
        default=str(src_path / "experiment" / "config.yaml"),
        help="Config file path",
    )
    parser.add_argument(
        "--experiments",
        default="all",
        help="Experiments to run, comma-separated (1,2,3 or all)",
    )
    parser.add_argument("--force-api", action="store_true", help="Force re-calling APIs")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--tables-dir", default=None, help="Value model tables directory")
    args = parser.parse_args()

    # 解析要运行的实验
    if args.experiments == "all":
        exp_ids = {1, 2, 3}
    else:
        exp_ids = {int(x.strip()) for x in args.experiments.split(",")}

    # 加载配置
    config = load_config(args.config)
    llm_configs = config.get("llm_models", {})
    profile_cfg = config.get("profile_experiment", {})

    # Base directories
    exp_base = src_path / "experiment"

    # Output and cache directories
    output_dir = args.output_dir or resolve_path(
        project_root,
        profile_cfg.get(
            "output_dir",
            str(exp_paths.PROFILE_DIR.relative_to(exp_paths.PROJECT_ROOT)),
        ),
    )
    cache_dir = resolve_path(
        project_root,
        profile_cfg.get(
            "cache_dir",
            str(exp_paths.PROFILE_CACHE_DIR.relative_to(exp_paths.PROJECT_ROOT)),
        ),
    )

    # 价值模型
    tables_dir = args.tables_dir or str(project_root / "tables")
    value_loader = ValueModelLoader(tables_dir)
    value_loader.load()
    value_model_text = value_loader.format_value_model_for_prompt()

    # LLM clients
    logger.info("Initializing LLM clients ...")
    llm_clients = LLMClientFactory.create_all_enabled(llm_configs)
    logger.info(f"Available LLMs: {list(llm_clients.keys())}")

    # 核心组件
    generator = ProfileGenerator(llm_clients, value_model_text, cache_dir=cache_dir)
    viz = ProfileVisualizer(output_dir)

    # Repository configurations
    repos = profile_cfg.get("repos", [])
    for repo in repos:
        repo["path"] = resolve_path(exp_base, repo["path"])

    # 记录开始
    start_time = time.time()
    all_results = {}

    # ====================================================================
    # Experiment 1: Profile Characterization
    # ====================================================================
    if 1 in exp_ids:
        logger.info("=" * 60)
        logger.info("Experiment 1: Profile Characterization")
        logger.info("=" * 60)
        from experiment.profile_experiment.exp1_characterization import run_exp1

        exp1_cfg = profile_cfg.get("exp1_characterization", {})
        model = exp1_cfg.get("representative_model", "qwen-plus")

        results = run_exp1(
            generator, viz, repos, model_key=model,
            output_dir=output_dir, force_api=args.force_api,
        )
        all_results["exp1"] = results

    # ====================================================================
    # Experiment 2: Cross-Model Consistency
    # ====================================================================
    if 2 in exp_ids:
        logger.info("=" * 60)
        logger.info("Experiment 2: Cross-Model Consistency")
        logger.info("=" * 60)
        from experiment.profile_experiment.exp2_cross_model import run_exp2

        exp2_cfg = profile_cfg.get("exp2_cross_model", {})
        models = exp2_cfg.get("models", list(llm_clients.keys()))
        # 只保留可用的模型
        models = [m for m in models if m in llm_clients]

        results = run_exp2(
            generator, viz, repos, model_keys=models,
            output_dir=output_dir, force_api=args.force_api,
        )
        all_results["exp2"] = results

    # ====================================================================
    # Experiment 3: Bayesian Profile Calibration
    # ====================================================================
    if 3 in exp_ids:
        logger.info("=" * 60)
        logger.info("Experiment 3: Bayesian Profile Calibration")
        logger.info("=" * 60)
        from experiment.profile_experiment.exp4_downstream import run_exp3

        exp3_cfg = profile_cfg.get("exp3_downstream", {})
        model = exp3_cfg.get("model", "qwen-plus")

            # IAA cache directory
        iaa_cfg = config.get("iaa_experiment", {})
        iaa_cache_dir = resolve_path(
            project_root,
            iaa_cfg.get(
                "llm_outputs_dir",
                str(exp_paths.LLM_OUTPUTS_DIR.relative_to(exp_paths.PROJECT_ROOT)),
            ),
        )

        # 需要 Exp1 的 Profile
        profiles = {}
        if "exp1" in all_results and "profiles" in all_results["exp1"]:
            profiles = all_results["exp1"]["profiles"]
        else:
            # 尝试为每个项目生成 Profile
            logger.info("[Exp3] Exp1 results unavailable; generating profiles for each repo ...")
            for repo in repos:
                docs = generator.collect_documents(repo["path"], repo.get("doc_sources", []))
                p = generator.generate_profile(
                    repo["name"], model, docs,
                    force_api=args.force_api,
                    repo_path=repo["path"],
                    use_code_evidence=True,
                )
                profiles[repo["name"]] = p

        results = run_exp3(
            viz=viz,
            profiles=profiles,
            config=config,
            model_key=model,
            iaa_cache_dir=iaa_cache_dir,
            output_dir=output_dir,
        )
        all_results["exp3"] = results

    # ====================================================================
    # Combined report
    # ====================================================================
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"All experiments completed in {elapsed:.1f}s")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    # Write summary
    summary = {
        "experiments_run": sorted(exp_ids),
        "elapsed_seconds": round(elapsed, 1),
        "output_dir": output_dir,
    }

    for eid in sorted(exp_ids):
        key = f"exp{eid}"
        if key in all_results:
            r = all_results[key]
            if eid == 1:
                summary[key] = r.get("summary", {})
            elif eid == 2:
                summary[key] = {
                    repo: {"consistency": data["consistency"]}
                    for repo, data in r.get("repos", {}).items()
                }
            elif eid == 3:
                summary[key] = {
                    "optimal_alpha": r.get("optimal_alpha"),
                    "comparison": r.get("comparison", {}),
                }

    summary_path = Path(output_dir) / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"综合摘要: {summary_path}")

    # 打印关键结果
    print("\n" + "=" * 60)
    print("VALUE PROFILE EVALUATION — KEY RESULTS")
    print("=" * 60)

    if "exp1" in all_results:
        print("\n[Exp1] Profile Characterization:")
        for name, info in all_results["exp1"].get("summary", {}).items():
            print(f"  {name}: Core = {info.get('core_values', [])}")

    if "exp2" in all_results:
        print("\n[Exp2] Cross-Model Consistency:")
        for name, data in all_results["exp2"].get("repos", {}).items():
            c = data["consistency"]
            print(f"  {name}: W={c['kendall_w']:.3f}, "
                  f"ρ={c['avg_spearman']:.3f}, cos={c['avg_cosine']:.3f}")

    if "exp3" in all_results:
        print("\n[Exp3] Bayesian Profile Calibration:")
        r3 = all_results["exp3"]
        opt_a = r3.get("optimal_alpha", "?")
        print(f"  Optimal alpha: {opt_a}")
        overall = r3.get("comparison", {}).get("overall", {})
        if overall:
            for k in overall.get("baseline", {}):
                v_base = overall["baseline"][k]
                v_opt = overall["optimal"].get(k, 0.0)
                d = overall["delta"].get(k, 0.0)
                sign = "+" if d >= 0 else ""
                print(f"  {k}: {v_base:.4f} -> {v_opt:.4f} ({sign}{d:.4f})")

    print(f"\nAll outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
