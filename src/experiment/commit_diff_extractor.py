"""
Pipeline Evaluation Framework — Combined evaluation of Hypothesis Generator + Evidence Location Agent
Extracts commits from real Git repositories, runs the full pipeline, and evaluates hypothesis quality and evidence location effectiveness
"""

import subprocess
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Data Structures
# ============================================================

@dataclass
class CommitInfo:
    """Git commit information"""
    sha: str
    short_sha: str
    message: str
    author: str
    date: str
    files_changed: list[str] = field(default_factory=list)

    # Labels inferred from commit message
    inferred_labels: list[str] = field(default_factory=list)
    bug_ids: list[str] = field(default_factory=list)

    def has_value_keywords(self) -> bool:
        """Check whether the commit contains value-related keywords"""
        return len(self.inferred_labels) > 0


@dataclass
class DiffHunkExtracted:
    """Code change hunk extracted from git diff"""
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
    """Commit sample for evaluation"""
    commit: CommitInfo
    diff_hunks: list[DiffHunkExtracted]

    # Ground truth annotations (optional)
    ground_truth_has_risk: Optional[bool] = None
    ground_truth_values: list[str] = field(default_factory=list)

    # Inferred risk (based on commit message keywords)
    @property
    def inferred_has_risk(self) -> bool:
        return self.commit.has_value_keywords()


# ============================================================
# Commit Diff Extractor
# ============================================================

# Value-related keywords used to infer risk from commit messages
VALUE_KEYWORDS = {
    # Security / Privacy related
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

    # Reliability related
    "crash": ["SV5"],
    "bug": ["SV2"],
    "fix": ["SV2"],
    "error": ["SV2"],
    "exception": ["SV5"],

    # User experience related
    "accessibility": ["SV9"],
    "a11y": ["SV9"],
    "usability": ["SV8"],
    "performance": ["SV6"],

    # Trust related
    "trust": ["SV1"],
    "consent": ["HV6"],
    "permission": ["HV6", "HV9"],

    # Inclusiveness
    "inclusiv": ["HV4"],
    "discriminat": ["HV3", "HV4"],
}


class CommitDiffExtractor:
    """Extract commit diffs from a Git repository"""

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
        """Retrieve recent commits"""
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

            # Infer labels
            commit.inferred_labels = self._infer_labels(message)
            commit.bug_ids = self._extract_bug_ids(message)

            # Get changed files
            commit.files_changed = self._get_changed_files(sha)

            if filter_value_related and not commit.has_value_keywords():
                continue

            commits.append(commit)

        return commits

    def _infer_labels(self, message: str) -> list[str]:
        """Infer value labels from commit message"""
        labels = set()
        message_lower = message.lower()

        for keyword, value_ids in VALUE_KEYWORDS.items():
            if keyword in message_lower:
                labels.update(value_ids)

        return list(labels)

    def _extract_bug_ids(self, message: str) -> list[str]:
        """Extract bug/issue IDs from commit message"""
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
        """Get the list of files changed in a commit"""
        cmd = [
            "git", "-C", str(self.repo_path),
            "diff-tree", "--no-commit-id", "--name-only", "-r", sha
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return []
        return [f for f in result.stdout.strip().split("\n") if f]

    def get_commit_diff(self, sha: str) -> list[DiffHunkExtracted]:
        """Get diff hunks for a single commit"""
        cmd = [
            "git", "-C", str(self.repo_path),
            "show", sha, "--format=", "-U3"  # 3 lines of context
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"git show failed for {sha}: {result.stderr}")
            return []

        return self._parse_diff(result.stdout)

    def _parse_diff(self, diff_text: str) -> list[DiffHunkExtracted]:
        """Parse git diff output into a list of DiffHunks"""
        hunks = []
        current_file = None
        current_hunk_lines = []
        current_old_start = 0
        current_new_start = 0
        current_old_lines = 0
        current_new_lines = 0

        for line in diff_text.split("\n"):
            # File path
            if line.startswith("diff --git"):
                # Save previous hunk
                if current_file and current_hunk_lines:
                    hunks.append(DiffHunkExtracted(
                        file_path=current_file,
                        old_start=current_old_start,
                        old_lines=current_old_lines,
                        new_start=current_new_start,
                        new_lines=current_new_lines,
                        content="\n".join(current_hunk_lines),
                    ))
                # Reset
                current_hunk_lines = []
                # Extract file path
                match = re.search(r"diff --git a/(.*) b/(.*)", line)
                if match:
                    current_file = match.group(2)
                continue

            # Hunk header: @@ -old_start,old_lines +new_start,new_lines @@
            if line.startswith("@@"):
                # Save previous hunk
                if current_file and current_hunk_lines:
                    hunks.append(DiffHunkExtracted(
                        file_path=current_file,
                        old_start=current_old_start,
                        old_lines=current_old_lines,
                        new_start=current_new_start,
                        new_lines=current_new_lines,
                        content="\n".join(current_hunk_lines),
                    ))
                # Parse new hunk
                match = re.search(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if match:
                    current_old_start = int(match.group(1))
                    current_old_lines = int(match.group(2) or 1)
                    current_new_start = int(match.group(3))
                    current_new_lines = int(match.group(4) or 1)
                current_hunk_lines = []
                continue

            # Skip binary files and other metadata
            if line.startswith("Binary files") or line.startswith("index "):
                continue
            if line.startswith("---") or line.startswith("+++"):
                continue
            if line.startswith("new file mode") or line.startswith("deleted file"):
                continue

            # Actual diff content
            if current_file:
                current_hunk_lines.append(line)

        # Save the last hunk
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
        """Build commit samples for evaluation"""
        samples = []

        for commit in commits:
            hunks = self.get_commit_diff(commit.sha)

            # Filter out hunks that are too large or from non-code files
            code_extensions = {".java", ".kt", ".py", ".js", ".ts", ".go", ".rs", ".cpp", ".c", ".h"}
            filtered_hunks = [
                h for h in hunks
                if any(h.file_path.endswith(ext) for ext in code_extensions)
                and len(h.content) < 5000  # Skip overly long hunks
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
# Export to JSON (for external use)
# ============================================================

def export_samples_to_json(samples: list[CommitSample], output_path: str):
    """Export samples to JSON format"""
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

    logger.info(f"Exported {len(data)} samples to {output_path}")


def load_samples_from_json(input_path: str) -> list[CommitSample]:
    """Load samples from JSON"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
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
# CLI Entry Point
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract commit diff samples from a Git repository")
    parser.add_argument("repo_path", help="Path to the repository")
    parser.add_argument("--output", "-o", default="commit_samples.json", help="Output file path")
    parser.add_argument("--limit", "-n", type=int, default=50, help="Maximum number of commits")
    parser.add_argument("--filter-value", action="store_true", help="Keep only value-related commits")
    parser.add_argument("--since", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--until", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    extractor = CommitDiffExtractor(args.repo_path)

    print(f"Extracting commits from {args.repo_path}...")
    commits = extractor.get_recent_commits(
        limit=args.limit,
        since=args.since,
        until=args.until,
        filter_value_related=args.filter_value,
    )
    print(f"Found {len(commits)} commits")

    print("Building samples...")
    samples = extractor.build_commit_samples(commits)
    print(f"Built {len(samples)} valid samples")

    # Statistics
    risk_count = sum(1 for s in samples if s.inferred_has_risk)
    print(f"Samples with inferred risk: {risk_count}/{len(samples)}")

    # Export
    export_samples_to_json(samples, args.output)


if __name__ == "__main__":
    main()
