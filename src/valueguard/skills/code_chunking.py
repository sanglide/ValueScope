"""Code chunking skill for parsing git diffs."""

import os
import re
import subprocess
from typing import Any, Optional

from valueguard.core.models import DiffHunk
from valueguard.skills.base_skill import BaseSkill


class CodeChunkingSkill(BaseSkill):
    """Skill for parsing git diffs into structured code chunks.

    Extracts diff hunks from a repository, providing structured access
    to changed code with surrounding context.
    """

    name = "code_chunking"
    description = "Parse git diff into structured code chunks"
    version = "1.0.0"

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        self.context_lines = config.get("context_lines", 3) if config else 3

    def validate_args(self, **kwargs: Any) -> None:
        """Validate arguments."""
        repo_path = kwargs.get("repo_path")
        if repo_path and not os.path.isdir(repo_path):
            raise ValueError(f"Repository path does not exist: {repo_path}")

    def execute(
        self,
        repo_path: str = ".",
        diff_base: str = "HEAD~1",
        diff_target: str = "HEAD",
    ) -> list[DiffHunk]:
        """Parse git diff and return structured DiffHunk objects.

        Args:
            repo_path: Path to the git repository
            diff_base: Base commit for diff comparison (default: HEAD~1)
            diff_target: Target commit for diff (default: HEAD)

        Returns:
            List of DiffHunk objects representing changed code
        """
        # Get list of changed files
        changed_files = self._get_changed_files(repo_path, diff_base, diff_target)

        # Extract hunks from each file
        all_hunks = []
        for file_path in changed_files:
            hunks = self._extract_hunks(repo_path, file_path, diff_base, diff_target)
            all_hunks.extend(hunks)

        return all_hunks

    def _run_git(
        self, args: list[str], cwd: Optional[str] = None
    ) -> tuple[bool, str]:
        """Run a git command and return (success, output)."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0, result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return False, ""

    def _get_changed_files(
        self, repo_path: str, diff_base: str, diff_target: str
    ) -> list[str]:
        """Get list of changed files between two commits."""
        # Try standard diff
        success, output = self._run_git(
            ["diff", "--name-only", f"{diff_base}...{diff_target}"],
            cwd=repo_path,
        )
        if success and output:
            return [f for f in output.splitlines() if f.strip()]

        # Fallback: try without three dots
        success, output = self._run_git(
            ["diff", "--name-only", diff_base, diff_target],
            cwd=repo_path,
        )
        if success and output:
            return [f for f in output.splitlines() if f.strip()]

        # Fallback: list all tracked files
        success, output = self._run_git(
            ["ls-tree", "-r", "--name-only", "HEAD"],
            cwd=repo_path,
        )
        if success and output:
            return [f for f in output.splitlines() if f.strip()]

        return []

    def _extract_hunks(
        self,
        repo_path: str,
        file_path: str,
        diff_base: str,
        diff_target: str,
    ) -> list[DiffHunk]:
        """Extract diff hunks for a specific file."""
        # Get unified diff
        success, diff_output = self._run_git(
            [
                "diff",
                f"-U{self.context_lines}",
                f"{diff_base}...{diff_target}",
                "--",
                file_path,
            ],
            cwd=repo_path,
        )

        if not success or not diff_output:
            # Try without three dots
            success, diff_output = self._run_git(
                [
                    "diff",
                    f"-U{self.context_lines}",
                    diff_base,
                    diff_target,
                    "--",
                    file_path,
                ],
                cwd=repo_path,
            )

        if not diff_output:
            return []

        # Parse hunks
        hunks = []
        current_hunk = None
        lines = diff_output.splitlines()

        for line in lines:
            if line.startswith("@@"):
                # Save previous hunk
                if current_hunk is not None:
                    hunks.append(current_hunk)

                # Parse hunk header: @@ -old_start,old_lines +new_start,new_lines @@
                hunk_match = re.match(
                    r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line
                )
                if hunk_match:
                    old_start = int(hunk_match.group(1))
                    old_lines = int(hunk_match.group(2) or 1)
                    new_start = int(hunk_match.group(3))
                    new_lines = int(hunk_match.group(4) or 1)

                    current_hunk = {
                        "header": line,
                        "old_start": old_start,
                        "old_lines": old_lines,
                        "new_start": new_start,
                        "new_lines": new_lines,
                        "body": [],
                    }
            elif current_hunk is not None:
                current_hunk["body"].append(line)

        # Don't forget the last hunk
        if current_hunk is not None:
            hunks.append(current_hunk)

        # Convert to DiffHunk objects
        diff_hunks = []
        for idx, hunk in enumerate(hunks):
            content = hunk["header"] + "\n" + "\n".join(hunk["body"])

            # Determine change type
            has_additions = any(line.startswith("+") for line in hunk["body"])
            has_deletions = any(line.startswith("-") for line in hunk["body"])

            if has_additions and has_deletions:
                change_type = "modified"
            elif has_additions:
                change_type = "added"
            elif has_deletions:
                change_type = "deleted"
            else:
                change_type = "modified"

            diff_hunk = DiffHunk(
                file_path=file_path,
                old_start=hunk["old_start"],
                old_lines=hunk["old_lines"],
                new_start=hunk["new_start"],
                new_lines=hunk["new_lines"],
                content=content,
                change_type=change_type,
            )
            diff_hunks.append(diff_hunk)

        return diff_hunks

    def get_file_content(
        self, repo_path: str, file_path: str, start_line: int, end_line: int
    ) -> list[dict[str, Any]]:
        """Get file content with line numbers.

        Args:
            repo_path: Repository path
            file_path: Path to file within repo
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (inclusive)

        Returns:
            List of {"line": int, "text": str} dictionaries
        """
        full_path = os.path.join(repo_path, file_path)
        if not os.path.isfile(full_path):
            return []

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()

            result = []
            for i in range(max(0, start_line - 1), min(len(all_lines), end_line)):
                result.append({"line": i + 1, "text": all_lines[i].rstrip("\n")})
            return result
        except (IOError, UnicodeDecodeError):
            return []
