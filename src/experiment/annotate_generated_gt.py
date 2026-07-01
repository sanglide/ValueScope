#!/usr/bin/env python3
"""
Batch LLM annotation script for generated samples.

Replaces keyword-inferred GT with LLM-annotated GT for generated code scenarios.

Usage:
  python -m experiment.annotate_generated_gt --benchmark path/to/benchmark.json
  python -m experiment.annotate_generated_gt --benchmark path/to/benchmark.json --model qwen-plus
  python -m experiment.annotate_generated_gt --benchmark path/to/benchmark.json --max-workers 4
  python -m experiment.annotate_generated_gt --benchmark path/to/benchmark.json --dry-run
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Project paths
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from data_loader import ValueModelLoader
    from llm_client import LLMClientFactory
    import paths as exp_paths
except ImportError:
    from experiment.data_loader import ValueModelLoader
    from experiment.llm_client import LLMClientFactory
    from experiment import paths as exp_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("annotate_gt")

# ============================================================
# Prompt（复用 config.yaml 中的 code_value_risk prompt）
# ============================================================

SYSTEM_PROMPT = (
    "你是一位经验丰富的软件工程师和项目管理者，专注于软件系统中的价值风险分析。\n\n"
    "你的任务是分析给定的代码，识别其中可能存在的价值风险。\n\n"
    "价值风险是指软件系统可能对用户或利益相关者的重要价值"
    "（如隐私、安全、公平性、可用性等）产生负面影响的潜在问题。\n"
)

USER_PROMPT_TEMPLATE = (
    "## 价值场景（代码）\n```\n{value_scenario}\n```\n\n"
    "## 价值模型参考\n以下是可能涉及的价值类型及其定义：\n{value_model}\n\n"
    "## 任务\n请你以一个软件工程师和项目管理者的视角，回答以下两个问题：\n\n"
    "（1）上述代码是否面临价值风险？请回答\"是\"或\"否\"。\n\n"
    "（2）如果面临价值风险，可能涉及哪些价值风险？请从上述价值模型中选择相关的价值类型，并说明原因。\n\n"
    "## 输出格式\n请严格按照以下JSON格式输出：\n"
    "```json\n"
    "{\n"
    '  "has_value_risk": true/false,\n'
    '  "identified_values": [\n'
    "    {\n"
    '      "value_id": "价值ID（如HV1, SV1等）",\n'
    '      "value_name": "价值名称",\n'
    '      "confidence": 0.0-1.0之间的置信度,\n'
    '      "reasoning": "识别该价值风险的理由"\n'
    "    }\n"
    "  ],\n"
    '  "overall_analysis": "整体分析说明"\n'
    "}\n```\n"
)


def load_benchmark(path: str) -> list[dict]:
    """加载 benchmark JSON"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_benchmark(samples: list[dict], path: str):
    """保存 benchmark JSON"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def annotate_one(
    client,
    sample: dict,
    value_model_text: str,
    cache_dir: Path,
    model_key: str,
    dry_run: bool = False,
) -> dict:
    """标注单个样本，返回 {has_value_risk, identified_values, confidences}"""
    sid = sample["sample_id"]
    cache_file = cache_dir / f"{model_key}_{sid}_gt.json"

    # 检查缓存
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return {
                "sample_id": sid,
                "status": "cache",
                "has_value_risk": data.get("has_value_risk", False),
                "identified_values": data.get("identified_values", []),
                "confidences": data.get("confidences", {}),
            }
        except (json.JSONDecodeError, KeyError):
            pass

    if dry_run:
        return {
            "sample_id": sid,
            "status": "dry_run",
            "has_value_risk": False,
            "identified_values": [],
            "confidences": {},
        }

    # 构建 prompt
    content = sample.get("content", "")
    user_prompt = USER_PROMPT_TEMPLATE.replace("{value_scenario}", content)
    user_prompt = user_prompt.replace("{value_model}", value_model_text)

    # 调用 LLM（带重试）
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = client.call(SYSTEM_PROMPT, user_prompt)
            if response.error:
                raise RuntimeError(f"API error: {response.error}")

            result = response.parsed_result or {}
            has_risk = result.get("has_value_risk", False)
            values = []
            confidences = {}
            for item in result.get("identified_values", []):
                vid = item.get("value_id", "")
                if vid:
                    values.append(vid)
                    confidences[vid] = item.get("confidence", 1.0)

            # 保存缓存
            cache_data = {
                "sample_id": sid,
                "model": model_key,
                "has_value_risk": has_risk,
                "identified_values": values,
                "confidences": confidences,
                "overall_analysis": result.get("overall_analysis", ""),
                "timestamp": datetime.now().isoformat(),
            }
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps(cache_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            return {
                "sample_id": sid,
                "status": "api",
                "has_value_risk": has_risk,
                "identified_values": values,
                "confidences": confidences,
            }
        except Exception as e:
            if attempt < max_retries:
                delay = 5 * (2 ** (attempt - 1))
                logger.warning(f"[{sid}] attempt {attempt}/{max_retries} failed: {e}, retry in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"[{sid}] all {max_retries} attempts failed: {e}")
                return {
                    "sample_id": sid,
                    "status": "failed",
                    "has_value_risk": False,
                    "identified_values": [],
                    "confidences": {},
                }


def main():
    parser = argparse.ArgumentParser(description="Batch-annotate generated samples with an LLM")
    parser.add_argument(
        "--benchmark",
        default=str(exp_paths.MAIN_EXP_DIR / "benchmark.json"),
        help="Path to benchmark.json",
    )
    parser.add_argument("--model", default="qwen-plus", help="LLM model key")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--cache-dir", default="", help="Cache directory")
    parser.add_argument("--dry-run", action="store_true", help="Count samples without calling APIs")
    parser.add_argument("--save", action="store_true", help="Update the benchmark file after annotation")
    args = parser.parse_args()

    benchmark_path = args.benchmark
    model_key = args.model

    # 加载 benchmark
    samples = load_benchmark(benchmark_path)
    logger.info(f"Loaded {len(samples)} samples from {benchmark_path}")

    # 筛选需要标注的样本
    to_annotate = [
        s for s in samples
        if s.get("source") == "generated" and s.get("gt_quality") == "keyword_inferred"
    ]
    logger.info(f"Found {len(to_annotate)} generated samples with keyword_inferred GT")

    if not to_annotate:
        logger.info("No samples to annotate. Exiting.")
        return

    # 加载价值模型
    tables_dir = project_root / "tables"
    vml = ValueModelLoader(str(tables_dir))
    vml.load()
    value_model_text = vml.format_value_model_for_prompt()
    logger.info(f"Loaded value model: {len(vml.l2_values)} L2 + {len(vml.l3_values)} L3 values")

    # 缓存目录
    cache_dir = Path(args.cache_dir) if args.cache_dir else exp_paths.GT_ANNOTATION_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cache dir: {cache_dir}")

    # 初始化 LLM 客户端
    import yaml
    config_path = Path(__file__).parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    llm_config = config.get("llm_models", {}).get(model_key, {})
    if not llm_config:
        logger.error(f"Model {model_key} not found in config.yaml")
        return
    client = LLMClientFactory.create(llm_config)
    logger.info(f"LLM client created for {model_key}")

    # 批量标注
    stats = {"cache": 0, "api": 0, "failed": 0, "dry_run": 0}
    results_lock = Lock()
    annotations = {}

    def process(idx: int, sample: dict) -> dict:
        result = annotate_one(client, sample, value_model_text, cache_dir, model_key, args.dry_run)
        with results_lock:
            stats[result["status"]] += 1
            annotations[result["sample_id"]] = result
            total_done = sum(stats.values())
            if total_done % 50 == 0 or result["status"] == "api":
                logger.info(
                    f"  Progress: {total_done}/{len(to_annotate)} "
                    f"(cache={stats['cache']}, api={stats['api']}, failed={stats['failed']})"
                )
        return result

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process, i, s)
            for i, s in enumerate(to_annotate)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

    elapsed = time.time() - start_time
    logger.info(
        f"\nAnnotation complete in {elapsed:.1f}s\n"
        f"  cache={stats['cache']}, api={stats['api']}, failed={stats['failed']}"
    )

    # 统计标注结果
    n_risk = sum(1 for r in annotations.values() if r["has_value_risk"])
    n_no_risk = len(annotations) - n_risk
    logger.info(f"  LLM 标注: {n_risk} 有风险, {n_no_risk} 无风险")

    if args.dry_run:
        logger.info("Dry run mode — no changes made.")
        return

    # 更新 benchmark
    if args.save:
        bench_map = {s["sample_id"]: s for s in samples}
        updated = 0
        for sid, ann in annotations.items():
            if ann["status"] in ("api", "cache"):
                bs = bench_map.get(sid)
                if bs:
                    bs["has_value_risk"] = ann["has_value_risk"]
                    bs["ground_truth_values"] = ann["identified_values"]
                    bs["gt_quality"] = "llm_inferred"
                    bs["gt_label_count"] = len(ann["identified_values"])
                    updated += 1

        save_benchmark(samples, benchmark_path)
        logger.info(f"Updated {updated} samples in {benchmark_path}")
    else:
        logger.info("Use --save to write annotations back to benchmark file.")


if __name__ == "__main__":
    main()
