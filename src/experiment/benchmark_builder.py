#!/usr/bin/env python3
"""
Unified Benchmark Builder — used by the main experiment.

Builds a single BenchmarkSample collection from the unified scenario JSON files,
attaches repo metadata, deduplicates by sample id, and optionally preserves
diff-hunk data so that pipeline-capable samples can run the evidence agent.
"""

import csv
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# 统一数据结构
# ============================================================

@dataclass
class BenchmarkSample:
    """主实验统一样本格式"""
    sample_id: str
    content: str                        # 场景文本（用于 LLM-only baseline）
    scenario_type: str                  # "code" / "text"
    has_value_risk: bool
    ground_truth_values: list[str] = field(default_factory=list)
    repo: str = "unknown"

    # Provenance metadata
    source: str = "unknown"             # "handwritten" / "generated" / "pipeline" / "issues"
    gt_quality: str = "unknown"         # "human" / "keyword_inferred" / "llm_inferred"
    gt_label_count: int = -1            # Number of GT values; -1 = not recorded

    # Pipeline-only fields (filled when diff_hunks_data is available)
    commit_sha: Optional[str] = None
    commit_message: Optional[str] = None
    diff_hunks_data: Optional[list[dict]] = None  # Serialized DiffHunkExtracted

    def to_dict(self) -> dict:
        d = asdict(self)
        # 移除 None 值以节省空间
        return {k: v for k, v in d.items() if v is not None}

    @property
    def supports_pipeline(self) -> bool:
        """是否支持 Pipeline 模式（需要 diff hunks）"""
        return self.diff_hunks_data is not None and len(self.diff_hunks_data) > 0


# ============================================================
# Benchmark 构建器
# ============================================================

class BenchmarkBuilder:
    """从多种数据源构建统一 benchmark"""

    # repo 名称标准化映射
    REPO_ALIASES = {
        "signal-android": "Signal-Android",
        "signal": "Signal-Android",
        "focus-android": "focus-android",
        "focus": "focus-android",
        "k-9": "k-9",
        "k9": "k-9",
        "git": "git",
        "kubernetes": "kubernetes",
        "kube": "kubernetes",
        "openclaw": "openclaw",
        "carmen": "carmen",
        "proposal-type-annotations": "proposal-type-annotations",
    }

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples: list[BenchmarkSample] = []
        self._seen_ids: set[str] = set()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def build_from_config(self, config: dict) -> list[BenchmarkSample]:
        """根据配置文件构建 benchmark

        Args:
            config: 主实验配置中的 datasets 部分
        """
        for ds_key, ds_conf in config.items():
            if not ds_conf.get("enabled", True):
                logger.info(f"  跳过已禁用数据集: {ds_key}")
                continue

            ds_type = ds_conf.get("type", "")
            ds_path = self.data_dir / ds_conf.get("path", "")

            if ds_type in ("json", "scenario_json"):
                # Unified scenario JSON (id, scenario, has_value_risk, metadata, ...)
                self._load_scenario_json(ds_path, source=ds_conf.get("source", ds_key))
            elif ds_type == "issues_json":
                self._load_issues_json(ds_path)
            elif ds_type == "issues_dataset":
                self._load_issues(ds_path, ds_conf)
            else:
                logger.warning(f"  Unknown dataset type: {ds_type} ({ds_key})")

        logger.info(f"Benchmark 构建完成: {len(self.samples)} 个样本 "
                     f"({self._count_by_type('code')} code + "
                     f"{self._count_by_type('text')} text)")
        return self.samples

    def build_all(self) -> list[BenchmarkSample]:
        """Discover and load the unified scenario JSON files under the data directory."""
        code_path = self.data_dir / "code_scenarios.json"
        text_path = self.data_dir / "text_scenarios" / "issues.json"
    
        if code_path.exists():
            self._load_scenario_json(code_path, source="code")
        if text_path.exists():
            self._load_issues_json(text_path)
    
        logger.info(f"Benchmark built: {len(self.samples)} samples "
                     f"({self._count_by_type('code')} code + "
                     f"{self._count_by_type('text')} text)")
        return self.samples

    def override_gt_with_llm(
        self,
        llm_cache_dir: str,
        model: str = "qwen-plus",
        sources: list[str] | None = None,
    ) -> int:
        """用 IAA LLM 缓存覆盖 generated 样本的 GT 标签

        对于 keyword_inferred 的样本，如果 IAA 缓存中有 LLM 预测，
        则用 LLM 预测替代关键词标注作为 GT。

        搜索顺序：
        1. {llm_cache_dir}/{model}_{sid}_output.json  (IAA 缓存格式)
        2. {llm_cache_dir}/gt_annotation_cache/{model}_{sid}_gt.json  (批量标注格式)
        """
        if sources is None:
            sources = ["generated"]

        cache_dir = Path(llm_cache_dir)
        gt_cache_dir = cache_dir / "gt_annotation_cache"
        if not cache_dir.exists():
            logger.warning(f"IAA 缓存目录不存在: {cache_dir}")
            return 0

        overridden = 0
        no_cache = 0
        for s in self.samples:
            if s.source not in sources:
                continue
            if s.gt_quality == "human":
                continue

            # 尝试 IAA 缓存格式
            iaa_file = cache_dir / f"{model}_{s.sample_id}_output.json"
            gt_file = gt_cache_dir / f"{model}_{s.sample_id}_gt.json"

            data = None
            for cf in [iaa_file, gt_file]:
                if cf.exists():
                    try:
                        data = json.loads(cf.read_text(encoding="utf-8"))
                        break
                    except (json.JSONDecodeError, OSError):
                        continue

            if data is None:
                no_cache += 1
                continue

            # 兼容两种字段命名
            llm_pred = data.get("predicted_has_risk", data.get("has_value_risk"))
            llm_values = data.get("predicted_values", data.get("identified_values", []))
            if llm_pred is None:
                continue

            s.has_value_risk = llm_pred
            s.ground_truth_values = llm_values
            s.gt_quality = "llm_inferred"
            s.gt_label_count = len(llm_values)
            overridden += 1

        logger.info(f"  [LLM GT override] 覆盖 {overridden} 个样本 "
                    f"({no_cache} 个无缓存保留 keyword GT)")
        return overridden

    def get_samples(self) -> list[BenchmarkSample]:
        return self.samples

    def get_code_samples(self) -> list[BenchmarkSample]:
        return [s for s in self.samples if s.scenario_type == "code"]

    def get_text_samples(self) -> list[BenchmarkSample]:
        return [s for s in self.samples if s.scenario_type == "text"]

    def get_pipeline_capable_samples(self) -> list[BenchmarkSample]:
        """获取支持 Pipeline 模式的样本（有 diff hunks）"""
        return [s for s in self.samples if s.supports_pipeline]

    def get_statistics(self) -> dict:
        code = self.get_code_samples()
        text = self.get_text_samples()
        pipeline_capable = self.get_pipeline_capable_samples()

        code_risk = sum(1 for s in code if s.has_value_risk)
        text_risk = sum(1 for s in text if s.has_value_risk)

        # repo 分布
        repo_dist = {}
        for s in self.samples:
            repo_dist[s.repo] = repo_dist.get(s.repo, 0) + 1

        # 来源分布
        source_dist = {}
        for s in self.samples:
            source_dist[s.source] = source_dist.get(s.source, 0) + 1

        return {
            "total": len(self.samples),
            "code": len(code),
            "text": len(text),
            "pipeline_capable": len(pipeline_capable),
            "code_with_risk": code_risk,
            "text_with_risk": text_risk,
            "code_risk_rate": round(code_risk / len(code), 3) if code else 0,
            "text_risk_rate": round(text_risk / len(text), 3) if text else 0,
            "repo_distribution": repo_dist,
            "source_distribution": source_dist,
        }

    # ------------------------------------------------------------------
    # 数据加载器
    # ------------------------------------------------------------------

    def _load_handwritten(self, path: Path, conf: dict):
        """加载手写 code 场景"""
        if not path.exists():
            logger.warning(f"文件不存在: {path}")
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        count = 0
        for item in data:
            sid = item.get("id", "")
            if sid in self._seen_ids:
                continue
            self._seen_ids.add(sid)

            repo = self._normalize_repo(item.get("metadata", {}).get("repo", "unknown"))
            self.samples.append(BenchmarkSample(
                sample_id=sid,
                content=item.get("scenario", ""),
                scenario_type="code",
                has_value_risk=item.get("has_value_risk", False),
                ground_truth_values=item.get("ground_truth_values", []),
                repo=repo,
                source="handwritten",
                gt_quality="human",
            ))
            count += 1
        logger.info(f"  [handwritten] 加载 {count} 个样本")

    def _load_generated_dir(self, dir_path: Path, conf: dict):
        """加载 generated 目录下的所有 scenario JSON"""
        if not dir_path.exists():
            logger.warning(f"目录不存在: {dir_path}")
            return

        pattern = conf.get("pattern", "*_scenarios.json")
        total = 0
        for json_file in sorted(dir_path.glob(pattern)):
            if json_file.stat().st_size < 10:
                continue
            count = self._load_scenario_json(json_file, source="generated")
            total += count
        logger.info(f"  [generated] 加载 {total} 个样本")

    def _load_pipeline_dir(self, dir_path: Path, conf: dict):
        """加载 pipeline 目录下的 JSON（两种格式）"""
        if not dir_path.exists():
            logger.warning(f"目录不存在: {dir_path}")
            return

        total = 0
        for json_file in sorted(dir_path.glob("*.json")):
            if json_file.stat().st_size < 10:
                continue

            # 判断格式：pipeline 格式有 commit + diff_hunks，scenario 格式有 id + scenario
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            if not isinstance(data, list) or not data:
                continue

            if "commit" in data[0] and "diff_hunks" in data[0]:
                # Pipeline 格式（CommitSample JSON）
                count = self._load_pipeline_json(json_file)
            elif "id" in data[0] and "scenario" in data[0]:
                # Scenario 格式
                count = self._load_scenario_json(json_file, source="pipeline")
            else:
                logger.warning(f"  未知格式: {json_file.name}")
                continue
            total += count
        logger.info(f"  [pipeline] 加载 {total} 个样本")

    def _load_scenario_json(self, path: Path, source: str = "unknown") -> int:
        """Load the unified scenario JSON format (id, scenario, has_value_risk, metadata, ...)."""
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return 0
        data = json.loads(path.read_text(encoding="utf-8"))
        count = 0
        for item in data:
            sid = item.get("id", "")
            if sid in self._seen_ids:
                continue
            self._seen_ids.add(sid)

            meta = item.get("metadata", {})
            repo = self._normalize_repo(meta.get("repo", "unknown"))
            item_source = meta.get("source", source)
            gt_quality = "keyword_inferred" if meta.get("auto_labeled") else "human"
            gt_values = item.get("ground_truth_values", [])

            self.samples.append(BenchmarkSample(
                sample_id=sid,
                content=item.get("scenario", ""),
                scenario_type=meta.get("scenario_type", "code"),
                has_value_risk=item.get("has_value_risk", False),
                ground_truth_values=gt_values,
                repo=repo,
                source=item_source,
                gt_quality=gt_quality,
                gt_label_count=len(gt_values),
                commit_sha=meta.get("commit_sha"),
                commit_message=meta.get("commit_message"),
                diff_hunks_data=item.get("diff_hunks_data"),
            ))
            count += 1
        return count

    def _load_pipeline_json(self, path: Path) -> int:
        """加载 pipeline CommitSample JSON 格式

        这是唯一会填充 diff_hunks_data 的加载器，
        使这些样本支持 Pipeline 模式运行。
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        count = 0

        # 从文件名推断 repo
        fname = path.stem.lower()
        repo_hint = "unknown"
        for name in ["signal_android", "signal-android", "focus_android", "focus-android"]:
            if name in fname:
                repo_hint = name.replace("_", "-")
                break

        for item in data:
            commit = item.get("commit", {})
            sha = commit.get("sha", "")
            short_sha = commit.get("short_sha", sha[:8])

            sid = f"pipe_{short_sha}"
            if sid in self._seen_ids:
                continue
            self._seen_ids.add(sid)

            repo = self._normalize_repo(commit.get("repo", repo_hint))
            diff_hunks = item.get("diff_hunks", [])

            # 构建场景文本（用于 LLM-only baseline）
            content = self._build_pipeline_content(item)

            has_risk = item.get("ground_truth_has_risk")
            if has_risk is None:
                has_risk = bool(commit.get("inferred_labels", []))
            gt_values = item.get("ground_truth_values", [])
            if not gt_values:
                gt_values = commit.get("inferred_labels", [])

            self.samples.append(BenchmarkSample(
                sample_id=sid,
                content=content,
                scenario_type="code",
                has_value_risk=has_risk,
                ground_truth_values=gt_values,
                repo=repo,
                source="pipeline",
                gt_quality="keyword_inferred" if not item.get("ground_truth_has_risk") else "human",
                commit_sha=sha,
                commit_message=commit.get("message", ""),
                diff_hunks_data=diff_hunks,
            ))
            count += 1
        return count

    def _load_issues(self, dir_path: Path, conf: dict):
        """加载 values_issues_dataset"""
        issues_file = dir_path / "issues.csv"
        posts_file = dir_path / "issue-posts.csv"
        labels_file = dir_path / "values-label.csv"

        if not all(f.exists() for f in [issues_file, posts_file, labels_file]):
            logger.warning(f"Issues 数据集文件不完整: {dir_path}")
            return

        from experiment.data_loader import ISSUES_VALUE_ID_MAPPING

        # 加载 issues
        issues = {}
        with open(issues_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="|"):
                issues[row["issue_id"]] = row

        # 加载 posts
        issue_texts = {}
        with open(posts_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="|"):
                iid = row["issue_id"]
                body = row.get("body_text", "")
                ptype = row.get("type", "post")
                if iid not in issue_texts:
                    issue_texts[iid] = {"title": "", "posts": []}
                if ptype == "title":
                    issue_texts[iid]["title"] = body
                else:
                    issue_texts[iid]["posts"].append(body)

        # 加载 labels
        issue_values = {}
        with open(labels_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="|"):
                iid = row["issue_id"]
                vid = row["proposed_values_id"]
                if iid not in issue_values:
                    issue_values[iid] = set()
                mapped = ISSUES_VALUE_ID_MAPPING.get(vid)
                if mapped:
                    issue_values[iid].add(mapped)

        # 采样配置
        sample_per_project = conf.get("sample_per_project")
        max_text_length = conf.get("max_text_length", 8000)

        # 按 project 分组
        by_project: dict[str, list[dict]] = {}
        for iid, issue in issues.items():
            project = issue.get("project_name", "unknown")
            title = issue.get("title", "")
            texts_data = issue_texts.get(iid, {"title": title, "posts": []})

            full_text = f"[Issue Title]: {texts_data['title']}\n\n"
            for i, pt in enumerate(texts_data["posts"], 1):
                full_text += f"[Post {i}]:\n{pt}\n\n"

            if len(full_text) > max_text_length:
                full_text = full_text[:max_text_length] + "\n\n[... truncated ...]"

            gt_values = sorted(issue_values.get(iid, set()))
            has_risk = len(gt_values) > 0

            record = {
                "sample_id": f"issue_{iid}",
                "content": full_text,
                "has_value_risk": has_risk,
                "ground_truth_values": gt_values,
                "repo": project,
                "project": project,
            }

            if project not in by_project:
                by_project[project] = []
            by_project[project].append(record)

        # 采样
        import random
        rng = random.Random(conf.get("seed", 42))
        all_records = []

        if sample_per_project is not None:
            for project, records in sorted(by_project.items()):
                with_risk = [r for r in records if r["has_value_risk"]]
                without_risk = [r for r in records if not r["has_value_risk"]]
                n_risk = min(len(with_risk), sample_per_project // 2 + sample_per_project % 2)
                n_no_risk = min(len(without_risk), sample_per_project - n_risk)
                all_records.extend(rng.sample(with_risk, min(n_risk, len(with_risk))))
                all_records.extend(rng.sample(without_risk, min(n_no_risk, len(without_risk))))
        else:
            for records in by_project.values():
                all_records.extend(records)

        # 添加到 samples
        count = 0
        for rec in all_records:
            sid = rec["sample_id"]
            if sid in self._seen_ids:
                continue
            self._seen_ids.add(sid)

            self.samples.append(BenchmarkSample(
                sample_id=sid,
                content=rec["content"],
                scenario_type="text",
                has_value_risk=rec["has_value_risk"],
                ground_truth_values=rec["ground_truth_values"],
                repo=self._normalize_repo(rec["repo"]),
                source="issues",
                gt_quality="human",
                gt_label_count=len(rec["ground_truth_values"]),
            ))
            count += 1
        logger.info(f"  [issues] 加载 {count} 个样本")

    def _load_issues_json(self, path: Path):
        """加载统一 JSON 格式的 text 场景"""
        if not path.exists():
            logger.warning(f"文件不存在: {path}")
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        count = 0
        for item in data:
            sid = item.get("id", "")
            if sid in self._seen_ids:
                continue
            self._seen_ids.add(sid)

            meta = item.get("metadata", {})
            repo = self._normalize_repo(meta.get("project_name", "unknown"))

            self.samples.append(BenchmarkSample(
                sample_id=sid,
                content=item.get("scenario", ""),
                scenario_type="text",
                has_value_risk=item.get("has_value_risk", False),
                ground_truth_values=item.get("ground_truth_values", []),
                repo=repo,
                source="issues",
                gt_quality="human",
            ))
            count += 1
        logger.info(f"  [issues.json] 加载 {count} 个样本")

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _normalize_repo(self, raw: str) -> str:
        """标准化 repo 名称"""
        key = raw.lower().replace("-", "").replace("_", "").replace(" ", "")
        for alias, canonical in self.REPO_ALIASES.items():
            alias_key = alias.lower().replace("-", "").replace("_", "").replace(" ", "")
            if alias_key == key or key in alias_key:
                return canonical
        return raw

    def _count_by_type(self, scenario_type: str) -> int:
        return sum(1 for s in self.samples if s.scenario_type == scenario_type)

    def _build_pipeline_content(self, item: dict) -> str:
        """从 pipeline JSON 构建场景文本"""
        commit = item.get("commit", {})
        parts = [
            f"// Repository: {commit.get('repo', 'unknown')}",
            f"// Commit: {commit.get('short_sha', '')} ({commit.get('date', '')})",
            f"// Message: {commit.get('message', '')}",
        ]
        files = commit.get("files_changed", [])
        if files:
            files_str = ", ".join(files[:5])
            if len(files) > 5:
                files_str += f", ... (+{len(files) - 5} more)"
            parts.append(f"// Files: {files_str}")

        for hunk in item.get("diff_hunks", []):
            parts.append("")
            parts.append(f"--- a/{hunk.get('file_path', '?')}")
            parts.append(f"@@ -{hunk.get('old_start', 0)},{hunk.get('old_lines', 0)} "
                        f"+{hunk.get('new_start', 0)},{hunk.get('new_lines', 0)} @@")
            parts.append(hunk.get("content", ""))

        return "\n".join(parts)


# ============================================================
# 保存 / 加载 benchmark
# ============================================================

def save_benchmark(samples: list[BenchmarkSample], output_path: str):
    """保存 benchmark 到 JSON"""
    data = [s.to_dict() for s in samples]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Benchmark 已保存: {len(data)} 个样本 -> {output_path}")


def load_benchmark(input_path: str) -> list[BenchmarkSample]:
    """从 JSON 加载 benchmark"""
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    samples = []
    for item in data:
        samples.append(BenchmarkSample(
            sample_id=item["sample_id"],
            content=item["content"],
            scenario_type=item["scenario_type"],
            has_value_risk=item["has_value_risk"],
            ground_truth_values=item.get("ground_truth_values", []),
            repo=item.get("repo", "unknown"),
            source=item.get("source", "unknown"),
            gt_quality=item.get("gt_quality", "unknown"),
            commit_sha=item.get("commit_sha"),
            commit_message=item.get("commit_message"),
            diff_hunks_data=item.get("diff_hunks_data"),
        ))
    logger.info(f"Benchmark 已加载: {len(samples)} 个样本 <- {input_path}")
    return samples


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    from experiment import paths as exp_paths
    parser = argparse.ArgumentParser(description="构建统一 Benchmark")
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).parent / "data"),
        help="数据根目录",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(exp_paths.MAIN_EXP_DIR / "benchmark.json"),
        help="输出路径",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML 配置文件（可选，不指定则自动发现）",
    )
    args = parser.parse_args()

    builder = BenchmarkBuilder(args.data_dir)

    if args.config:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        builder.build_from_config(config.get("datasets", {}))
    else:
        builder.build_all()

    stats = builder.get_statistics()
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    save_benchmark(builder.samples, args.output)


if __name__ == "__main__":
    main()