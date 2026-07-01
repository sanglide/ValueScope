"""
Pipeline 评估框架 — Hypothesis Generator + Evidence Location Agent 组合评估
从真实 Git 仓库提取 commits，运行完整 pipeline，评估假说质量和证据定位效果
"""

import subprocess
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class CommitInfo:
    """Git commit 信息"""
    sha: str
    short_sha: str
    message: str
    author: str
    date: str
    files_changed: list[str] = field(default_factory=list)

    # 从 commit message 提取的标签
    inferred_labels: list[str] = field(default_factory=list)
    bug_ids: list[str] = field(default_factory=list)

    def has_value_keywords(self) -> bool:
        """检查是否包含价值相关关键词"""
        return len(self.inferred_labels) > 0


@dataclass
class DiffHunkExtracted:
    """从 git diff 提取的代码变更块"""
    file_path: str
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    content: str
    change_type: str = "modified"

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "old_start": self.old_start,
            "old_lines": self.old_lines,
            "new_start": self.new_start,
            "new_lines": self.new_lines,
            "content": self.content,
            "change_type": self.change_type,
        }


@dataclass
class CommitSample:
    """用于评估的 commit 样本"""
    commit: CommitInfo
    diff_hunks: list[DiffHunkExtracted]

    # Ground truth 标注（可选）
    ground_truth_has_risk: Optional[bool] = None
    ground_truth_values: list[str] = field(default_factory=list)

    # 推断的风险（基于 commit message 关键词）
    @property
    def inferred_has_risk(self) -> bool:
        return self.commit.has_value_keywords()


# ============================================================
# Commit Diff 提取器
# ============================================================

# 价值相关关键词，用于从 commit message 推断风险
VALUE_KEYWORDS = {
    # 安全/隐私相关
    "security": ["HV10"],
    "privacy": ["HV9"],
    "sensitive": ["HV9"],
    "encrypt": ["HV10", "SV1"],
    "decrypt": ["HV10"],
    "auth": ["HV10", "SV1"],
    "token": ["HV10"],
    "password": ["HV9", "HV10"],
    "credential": ["HV9", "HV10"],
    "leak": ["HV9"],
    "exposure": ["HV9"],
    "vulnerability": ["HV10"],
    "cve": ["HV10"],
    # 隐私/遥测相关（补充遗漏的关键词）
    "telemetry": ["HV9"],
    "glean": ["HV9"],
    "analytics": ["HV9"],
    "tracking": ["HV9"],
    "screenshot": ["HV9"],
    "workmanager": ["HV9"],
    "probe": ["HV9"],
    "metric": ["HV9"],
    "data collection": ["HV9"],

    # 可靠性相关（移除过宽的 fix/bug/error）
    "crash": ["SV5"],
    "exception": ["SV5"],
    "regression": ["SV2"],
    "incorrect": ["SV2"],
    "miscalculat": ["SV2"],
    "wrong result": ["SV2"],
    "broken": ["SV2"],

    # 用户体验相关
    "accessibility": ["SV9"],
    "a11y": ["SV9"],
    "usability": ["SV8"],
    "performance": ["SV6"],

    # 信任相关
    "trust": ["SV1"],
    "consent": ["HV6"],
    "permission": ["HV6", "HV9"],

    # 包容性
    "inclusiv": ["HV4"],
    "discriminat": ["HV3", "HV4"],
}

# 负向关键词：commit message 中出现这些词时，即使匹配到正向关键词也标记为无风险
# 这些通常是测试、构建、代码风格等非价值偏离的改动
NEGATIVE_KEYWORDS = {
    "test", "lint", "format", "import ordering", "kotlinlinter",
    "flakey", "flaky", "re-enable", "re-enable ui test",
    "verifydownloadedfile", "verifycalendarform", "verifyexternal",
    "verifytextinput", "verifydownload",
    "update to", "bump version", "upgrade", "dependency",
    "changelog", "readme", "typo", "rename", "refactor",
    "merge branch", "merge pull request", "revert",
}


class CommitDiffExtractor:
    """从 Git 仓库提取 commit diff"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Not a git repository: {repo_path}")

    def get_recent_commits(
        self,
        limit: int = 100,
        since: Optional[str] = None,
        until: Optional[str] = None,
        filter_value_related: bool = False,
    ) -> list[CommitInfo]:
        """获取最近的 commits"""
        cmd = [
            "git", "-C", str(self.repo_path), "log",
            f"--max-count={limit}",
            "--format=%H|%h|%s|%an|%ad",
            "--date=short",
        ]
        if since:
            cmd.append(f"--since={since}")
        if until:
            cmd.append(f"--until={until}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"git log failed: {result.stderr}")
            return []

        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 4)
            if len(parts) < 5:
                continue

            sha, short_sha, message, author, date = parts
            commit = CommitInfo(
                sha=sha,
                short_sha=short_sha,
                message=message,
                author=author,
                date=date,
            )

            # 推断标签
            commit.inferred_labels = self._infer_labels(message)
            commit.bug_ids = self._extract_bug_ids(message)

            # 获取变更文件
            commit.files_changed = self._get_changed_files(sha)

            if filter_value_related and not commit.has_value_keywords():
                continue

            commits.append(commit)

        return commits

    def _infer_labels(self, message: str) -> list[str]:
        """从 commit message 推断价值标签
        
        规则：
        1. 先检查负向关键词，若命中则返回空（测试/构建/风格类改动不是价值偏离）
        2. 再匹配正向关键词，推断价值标签
        """
        labels = set()
        message_lower = message.lower()

        # 负向过滤：测试/构建/代码风格等改动不视为价值偏离
        for neg_kw in NEGATIVE_KEYWORDS:
            if neg_kw in message_lower:
                return []

        # 正向匹配
        for keyword, value_ids in VALUE_KEYWORDS.items():
            if keyword in message_lower:
                labels.update(value_ids)

        return list(labels)

    def _extract_bug_ids(self, message: str) -> list[str]:
        """提取 bug/issue ID"""
        patterns = [
            r"Bug\s*(\d+)",
            r"#(\d+)",
            r"issue\s*(\d+)",
            r"fix\s*(\d+)",
        ]
        bug_ids = []
        for pattern in patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            bug_ids.extend(matches)
        return bug_ids

    def _get_changed_files(self, sha: str) -> list[str]:
        """获取 commit 变更的文件列表"""
        cmd = [
            "git", "-C", str(self.repo_path),
            "diff-tree", "--no-commit-id", "--name-only", "-r", sha
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return []
        return [f for f in result.stdout.strip().split("\n") if f]

    def get_commit_diff(self, sha: str) -> list[DiffHunkExtracted]:
        """获取单个 commit 的 diff hunks"""
        cmd = [
            "git", "-C", str(self.repo_path),
            "show", sha, "--format=", "-U3"  # 3行上下文
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"git show failed for {sha}: {result.stderr}")
            return []

        return self._parse_diff(result.stdout)

    def _parse_diff(self, diff_text: str) -> list[DiffHunkExtracted]:
        """解析 git diff 输出为 DiffHunk 列表"""
        hunks = []
        current_file = None
        current_hunk_lines = []
        current_old_start = 0
        current_new_start = 0
        current_old_lines = 0
        current_new_lines = 0

        for line in diff_text.split("\n"):
            # 文件路径
            if line.startswith("diff --git"):
                # 保存上一个 hunk
                if current_file and current_hunk_lines:
                    hunks.append(DiffHunkExtracted(
                        file_path=current_file,
                        old_start=current_old_start,
                        old_lines=current_old_lines,
                        new_start=current_new_start,
                        new_lines=current_new_lines,
                        content="\n".join(current_hunk_lines),
                    ))
                # 重置
                current_hunk_lines = []
                # 提取文件路径
                match = re.search(r"diff --git a/(.*) b/(.*)", line)
                if match:
                    current_file = match.group(2)
                continue

            # Hunk header: @@ -old_start,old_lines +new_start,new_lines @@
            if line.startswith("@@"):
                # 保存上一个 hunk
                if current_file and current_hunk_lines:
                    hunks.append(DiffHunkExtracted(
                        file_path=current_file,
                        old_start=current_old_start,
                        old_lines=current_old_lines,
                        new_start=current_new_start,
                        new_lines=current_new_lines,
                        content="\n".join(current_hunk_lines),
                    ))
                # 解析新 hunk
                match = re.search(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if match:
                    current_old_start = int(match.group(1))
                    current_old_lines = int(match.group(2) or 1)
                    current_new_start = int(match.group(3))
                    current_new_lines = int(match.group(4) or 1)
                current_hunk_lines = []
                continue

            # 跳过 binary 文件和其他元数据
            if line.startswith("Binary files") or line.startswith("index "):
                continue
            if line.startswith("---") or line.startswith("+++"):
                continue
            if line.startswith("new file mode") or line.startswith("deleted file"):
                continue

            # 实际 diff 内容
            if current_file:
                current_hunk_lines.append(line)

        # 保存最后一个 hunk
        if current_file and current_hunk_lines:
            hunks.append(DiffHunkExtracted(
                file_path=current_file,
                old_start=current_old_start,
                old_lines=current_old_lines,
                new_start=current_new_start,
                new_lines=current_new_lines,
                content="\n".join(current_hunk_lines),
            ))

        return hunks

    def build_commit_samples(
        self,
        commits: list[CommitInfo],
        max_hunks_per_commit: int = 10,
    ) -> list[CommitSample]:
        """构建用于评估的 commit 样本"""
        samples = []

        for commit in commits:
            hunks = self.get_commit_diff(commit.sha)

            # 过滤掉太大或非代码文件的 hunks
            code_extensions = {".java", ".kt", ".py", ".js", ".ts", ".go", ".rs", ".cpp", ".c", ".h"}
            filtered_hunks = [
                h for h in hunks
                if any(h.file_path.endswith(ext) for ext in code_extensions)
                and len(h.content) < 5000  # 不要太长的 hunk
            ][:max_hunks_per_commit]

            if filtered_hunks:
                sample = CommitSample(
                    commit=commit,
                    diff_hunks=filtered_hunks,
                    ground_truth_values=commit.inferred_labels,
                    ground_truth_has_risk=commit.has_value_keywords(),
                )
                samples.append(sample)

        return samples


# ============================================================
# 导出为 JSON（供外部使用）
# ============================================================

def export_samples_to_json(samples: list[CommitSample], output_path: str):
    """将样本导出为 JSON 格式"""
    data = []
    for sample in samples:
        data.append({
            "commit": {
                "sha": sample.commit.sha,
                "short_sha": sample.commit.short_sha,
                "message": sample.commit.message,
                "author": sample.commit.author,
                "date": sample.commit.date,
                "files_changed": sample.commit.files_changed,
                "inferred_labels": sample.commit.inferred_labels,
                "bug_ids": sample.commit.bug_ids,
            },
            "diff_hunks": [h.to_dict() for h in sample.diff_hunks],
            "ground_truth_has_risk": sample.ground_truth_has_risk,
            "ground_truth_values": sample.ground_truth_values,
            "inferred_has_risk": sample.inferred_has_risk,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"导出 {len(data)} 个样本到 {output_path}")


def scenario_record_to_commit_sample(record: dict) -> CommitSample:
    """Convert a unified scenario-format record back to a CommitSample.

    The scenario record must contain ``diff_hunks_data`` (the original hunks).
    This lets the standalone pipeline evaluator consume the same unified
    ``code_scenarios.json`` that the benchmark builder uses.
    """
    meta = record.get("metadata", {})
    commit_sha = meta.get("commit_sha", "")
    commit_short = commit_sha[:8] if commit_sha else ""
    commit_message = meta.get("commit_message", "")
    commit_date = meta.get("commit_date", "")

    diff_hunks = [
        DiffHunkExtracted(**h) for h in record.get("diff_hunks_data", [])
    ]

    files_changed = sorted({h.file_path for h in diff_hunks})
    inferred_labels = meta.get("inferred_labels", [])

    commit = CommitInfo(
        sha=commit_sha,
        short_sha=commit_short,
        message=commit_message,
        author="",
        date=commit_date,
        files_changed=files_changed,
        inferred_labels=inferred_labels,
        bug_ids=[],
    )

    sample = CommitSample(
        commit=commit,
        diff_hunks=diff_hunks,
        ground_truth_has_risk=record.get("has_value_risk"),
        ground_truth_values=record.get("ground_truth_values", []),
    )
    sample._repo_name = meta.get("repo", "unknown")
    return sample


def load_samples_from_json(input_path: str) -> list[CommitSample]:
    """Load commit samples from JSON.

    Accepts both the native CommitSample format and the unified scenario format
    produced by ``commit_sample_to_scenario``. Scenario records that contain
    ``diff_hunks_data`` are converted back to ``CommitSample`` objects.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        # Unified scenario format (id, scenario, has_value_risk, metadata, diff_hunks_data)
        if "commit" not in item and item.get("diff_hunks_data"):
            sample = scenario_record_to_commit_sample(item)
            samples.append(sample)
            continue

        if "commit" not in item:
            # Scenario record without diff hunks is not pipeline-evaluable; skip it.
            continue

        # Native CommitSample format
        commit = CommitInfo(
            sha=item["commit"]["sha"],
            short_sha=item["commit"]["short_sha"],
            message=item["commit"]["message"],
            author=item["commit"]["author"],
            date=item["commit"]["date"],
            files_changed=item["commit"].get("files_changed", []),
            inferred_labels=item["commit"].get("inferred_labels", []),
            bug_ids=item["commit"].get("bug_ids", []),
        )
        hunks = [
            DiffHunkExtracted(**h) for h in item["diff_hunks"]
        ]
        samples.append(CommitSample(
            commit=commit,
            diff_hunks=hunks,
            ground_truth_has_risk=item.get("ground_truth_has_risk"),
            ground_truth_values=item.get("ground_truth_values", []),
        ))

    return samples




# ============================================================
# CommitSample -> ValueScenarioSample 格式转换器
# ============================================================

def _build_scenario_text(sample: CommitSample, repo_name: str, max_content_length: int = 4000) -> str:
    """将 CommitSample 的 diff hunks 拼合为结构化场景文本"""
    header_lines = [
        f"// Repository: {repo_name}",
        f"// Commit: {sample.commit.short_sha} ({sample.commit.date})",
        f"// Message: {sample.commit.message}",
    ]
    if sample.commit.files_changed:
        files_str = ", ".join(sample.commit.files_changed[:5])
        if len(sample.commit.files_changed) > 5:
            files_str += f", ... (+{len(sample.commit.files_changed) - 5} more)"
        header_lines.append(f"// Files: {files_str}")

    diff_parts = []
    for hunk in sample.diff_hunks:
        part = (
            f"--- a/{hunk.file_path}\n"
            f"@@ -{hunk.old_start},{hunk.old_lines} +{hunk.new_start},{hunk.new_lines} @@\n"
            f"{hunk.content}"
        )
        diff_parts.append(part)

    scenario_text = "\n".join(header_lines) + "\n\n" + "\n\n".join(diff_parts)

    if len(scenario_text) > max_content_length:
        scenario_text = scenario_text[:max_content_length] + "\n\n[... truncated ...]"

    return scenario_text


def commit_sample_to_scenario(
    sample: CommitSample,
    repo_name: str = "unknown",
    scenario_id_prefix: str = "gen",
    max_content_length: int = 4000,
) -> dict:
    """将单个 CommitSample 转换为 scenario JSON 记录

    Returns:
        与统一格式 {id, scenario, has_value_risk, ground_truth_values, metadata} 一致的 dict
    """
    scenario_text = _build_scenario_text(sample, repo_name, max_content_length)
    sample_id = f"{scenario_id_prefix}_{sample.commit.short_sha}"

    has_risk = sample.ground_truth_has_risk if sample.ground_truth_has_risk is not None else sample.inferred_has_risk
    gt_values = sample.ground_truth_values if sample.ground_truth_values else sample.commit.inferred_labels

    return {
        "id": sample_id,
        "scenario": scenario_text,
        "has_value_risk": has_risk,
        "ground_truth_values": gt_values,
        "metadata": {
            "source": "generated",
            "scenario_type": "code",
            "repo": repo_name,
            "commit_sha": sample.commit.sha,
            "commit_message": sample.commit.message,
            "commit_date": sample.commit.date,
            "inferred_labels": sample.commit.inferred_labels,
            "auto_labeled": True,
        }
    }


def convert_pipeline_file_to_scenario_format(
    input_path: str,
    output_path: str,
    repo_name: str,
    scenario_id_prefix: str = "pipe",
) -> int:
    """将整个 pipeline JSON 文件批量转换为 scenario 格式并保存"""
    samples = load_samples_from_json(input_path)
    scenario_records = []

    for sample in samples:
        record = commit_sample_to_scenario(sample, repo_name, scenario_id_prefix)
        scenario_records.append(record)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scenario_records, f, indent=2, ensure_ascii=False)

    logger.info(f"Converted {len(scenario_records)} pipeline samples -> {output_path}")
    return len(scenario_records)


def generate_scenarios_from_repos(
    repos_dir: str,
    output_dir: str,
    max_commits_per_repo: int = 100,
    max_hunks_per_commit: int = 10,
    code_extensions: Optional[set] = None,
    max_content_length: int = 4000,
    existing_shas: Optional[set] = None,
) -> dict:
    """从多个仓库批量提取 commits 并生成 scenario 格式文件"""
    if code_extensions is None:
        code_extensions = {".java", ".kt", ".py", ".js", ".ts", ".go", ".rs", ".cpp", ".c", ".h", ".rb"}
    if existing_shas is None:
        existing_shas = set()

    repos_path = Path(repos_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}
    repo_dirs = sorted([
        d for d in repos_path.iterdir()
        if d.is_dir() and (d / ".git").exists()
    ])

    for repo_dir in repo_dirs:
        repo_name = repo_dir.name
        logger.info(f"Processing repo: {repo_name}")

        try:
            extractor = CommitDiffExtractor(str(repo_dir))
        except ValueError as e:
            logger.warning(f"Skipping {repo_name}: {e}")
            continue

        commits = extractor.get_recent_commits(
            limit=max_commits_per_repo,
            filter_value_related=False,
        )
        logger.info(f"  Found {len(commits)} commits")

        samples = extractor.build_commit_samples(
            commits,
            max_hunks_per_commit=max_hunks_per_commit,
        )

        filtered_samples = []
        seen_shas = set(existing_shas)
        for s in samples:
            if s.commit.sha in seen_shas:
                continue
            seen_shas.add(s.commit.sha)
            code_hunks = [
                h for h in s.diff_hunks
                if any(h.file_path.endswith(ext) for ext in code_extensions)
                and len(h.content) < 5000
            ][:max_hunks_per_commit]
            if code_hunks:
                s.diff_hunks = code_hunks
                filtered_samples.append(s)

        logger.info(f"  {len(filtered_samples)} valid samples after filtering")

        scenario_records = []
        repo_prefix = re.sub(r'[^a-z]', '', repo_name.lower())[:4]
        for s in filtered_samples:
            record = commit_sample_to_scenario(
                s, repo_name, f"gen_{repo_prefix}", max_content_length
            )
            scenario_records.append(record)

        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', repo_name.lower())
        output_file = output_path / f"{safe_name}_scenarios.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(scenario_records, f, indent=2, ensure_ascii=False)

        risk_count = sum(1 for r in scenario_records if r["has_value_risk"])
        logger.info(
            f"  Saved {len(scenario_records)} scenarios to {output_file} "
            f"(risk={risk_count}, no_risk={len(scenario_records) - risk_count})"
        )
        results[repo_name] = len(scenario_records)

    total = sum(results.values())
    logger.info(f"Total generated: {total} scenarios from {len(results)} repos")
    return results


# ============================================================
# CLI 入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="从 Git 仓库提取 commit diff 样本")
    parser.add_argument("repo_path", help="仓库路径")
    parser.add_argument("--output", "-o", default="commit_samples.json", help="输出文件")
    parser.add_argument("--limit", "-n", type=int, default=50, help="最大 commit 数")
    parser.add_argument("--filter-value", action="store_true", help="只保留价值相关的 commits")
    parser.add_argument("--since", help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--until", help="结束日期 (YYYY-MM-DD)")
    args = parser.parse_args()

    extractor = CommitDiffExtractor(args.repo_path)

    print(f"从 {args.repo_path} 提取 commits...")
    commits = extractor.get_recent_commits(
        limit=args.limit,
        since=args.since,
        until=args.until,
        filter_value_related=args.filter_value,
    )
    print(f"找到 {len(commits)} 个 commits")

    print("构建样本...")
    samples = extractor.build_commit_samples(commits)
    print(f"构建了 {len(samples)} 个有效样本")

    # 统计
    risk_count = sum(1 for s in samples if s.inferred_has_risk)
    print(f"推断有风险的样本: {risk_count}/{len(samples)}")

    # 导出
    export_samples_to_json(samples, args.output)


if __name__ == "__main__":
    main()
