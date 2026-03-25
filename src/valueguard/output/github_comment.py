"""GitHub comment poster for ValueGuard."""

import os
from typing import Optional

from valueguard.core.config import Config
from valueguard.core.models import ValueGuardReport
from valueguard.output.reporter import Reporter


class GitHubCommentPoster:
    """Post ValueGuard analysis results as GitHub PR comments.

    Uses the GitHub API to create or update comments on pull requests.
    """

    COMMENT_MARKER = "<!-- ValueGuard Analysis -->"

    def __init__(
        self,
        config: Optional[Config] = None,
        github_token: Optional[str] = None,
    ):
        """Initialize the comment poster.

        Args:
            config: ValueGuard configuration
            github_token: GitHub API token (falls back to GITHUB_TOKEN env var)
        """
        self.config = config
        self.token = github_token or os.environ.get("GITHUB_TOKEN", "")
        self.reporter = Reporter(config=config)

    def post_comment(
        self,
        report: ValueGuardReport,
        owner: str,
        repo: str,
        pr_number: int,
        update_existing: bool = True,
    ) -> dict:
        """Post analysis report as a PR comment.

        Args:
            report: ValueGuard analysis report
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            update_existing: If True, update existing comment instead of creating new

        Returns:
            API response dict with comment URL and ID
        """
        if not self.token:
            return {
                "success": False,
                "error": "GitHub token not provided",
            }

        # Generate markdown report
        markdown = self.reporter.to_markdown(report)

        # Add marker for identification
        body = f"{self.COMMENT_MARKER}\n{markdown}"

        try:
            import requests

            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
            }

            # Check for existing comment
            existing_comment_id = None
            if update_existing:
                existing_comment_id = self._find_existing_comment(
                    owner, repo, pr_number, headers
                )

            if existing_comment_id:
                # Update existing comment
                url = f"https://api.github.com/repos/{owner}/{repo}/issues/comments/{existing_comment_id}"
                response = requests.patch(
                    url,
                    headers=headers,
                    json={"body": body},
                    timeout=30,
                )
            else:
                # Create new comment
                url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
                response = requests.post(
                    url,
                    headers=headers,
                    json={"body": body},
                    timeout=30,
                )

            if response.status_code in (200, 201):
                data = response.json()
                return {
                    "success": True,
                    "comment_id": data.get("id"),
                    "comment_url": data.get("html_url"),
                    "updated": existing_comment_id is not None,
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "details": response.text,
                }

        except ImportError:
            return {
                "success": False,
                "error": "requests library not installed",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def _find_existing_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        headers: dict,
    ) -> Optional[int]:
        """Find existing ValueGuard comment on a PR.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            headers: API headers

        Returns:
            Comment ID if found, None otherwise
        """
        try:
            import requests

            url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                comments = response.json()
                for comment in comments:
                    if self.COMMENT_MARKER in comment.get("body", ""):
                        return comment.get("id")

        except Exception:
            pass

        return None

    def delete_comment(
        self,
        owner: str,
        repo: str,
        comment_id: int,
    ) -> dict:
        """Delete a ValueGuard comment.

        Args:
            owner: Repository owner
            repo: Repository name
            comment_id: Comment ID to delete

        Returns:
            API response dict
        """
        if not self.token:
            return {
                "success": False,
                "error": "GitHub token not provided",
            }

        try:
            import requests

            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
            }

            url = f"https://api.github.com/repos/{owner}/{repo}/issues/comments/{comment_id}"
            response = requests.delete(url, headers=headers, timeout=30)

            return {
                "success": response.status_code == 204,
                "status_code": response.status_code,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


def post_pr_comment(
    report: ValueGuardReport,
    owner: Optional[str] = None,
    repo: Optional[str] = None,
    pr_number: Optional[int] = None,
    github_token: Optional[str] = None,
) -> dict:
    """Convenience function to post a PR comment.

    Auto-detects owner/repo/pr from environment variables if not provided.

    Args:
        report: ValueGuard analysis report
        owner: Repository owner (default: from GITHUB_REPOSITORY)
        repo: Repository name (default: from GITHUB_REPOSITORY)
        pr_number: PR number (default: from report or GITHUB_PR_NUMBER)
        github_token: GitHub token (default: from GITHUB_TOKEN)

    Returns:
        API response dict
    """
    # Auto-detect from environment
    if not owner or not repo:
        github_repo = os.environ.get("GITHUB_REPOSITORY", "")
        if "/" in github_repo:
            owner, repo = github_repo.split("/", 1)

    if not pr_number:
        pr_number = report.pr_number or int(
            os.environ.get("GITHUB_PR_NUMBER", os.environ.get("PR_NUMBER", 0))
        )

    if not owner or not repo or not pr_number:
        return {
            "success": False,
            "error": "Missing required parameters: owner, repo, or pr_number",
        }

    poster = GitHubCommentPoster(github_token=github_token)
    return poster.post_comment(report, owner, repo, pr_number)
