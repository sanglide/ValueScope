"""GitHub Action adapter for ValueGuard."""

import json
import os
import sys
from typing import Optional


def get_github_context() -> dict:
    """Get GitHub Action context from environment."""
    return {
        "event_name": os.environ.get("GITHUB_EVENT_NAME", ""),
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),
        "sha": os.environ.get("GITHUB_SHA", ""),
        "ref": os.environ.get("GITHUB_REF", ""),
        "workspace": os.environ.get("GITHUB_WORKSPACE", "."),
        "actor": os.environ.get("GITHUB_ACTOR", ""),
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "run_number": os.environ.get("GITHUB_RUN_NUMBER", ""),
    }


def get_pr_number() -> Optional[int]:
    """Get PR number from GitHub context."""
    # Try from environment
    pr_num = os.environ.get("GITHUB_PR_NUMBER") or os.environ.get("PR_NUMBER")
    if pr_num:
        return int(pr_num)

    # Try from event payload
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        try:
            with open(event_path, "r") as f:
                event = json.load(f)
            pr = event.get("pull_request", {})
            if pr.get("number"):
                return int(pr["number"])
        except (json.JSONDecodeError, IOError):
            pass

    # Try from ref
    ref = os.environ.get("GITHUB_REF", "")
    if ref.startswith("refs/pull/") and "/merge" in ref:
        parts = ref.split("/")
        if len(parts) >= 3:
            try:
                return int(parts[2])
            except ValueError:
                pass

    return None


def get_base_ref() -> str:
    """Get base ref for diff comparison."""
    # Check event payload
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        try:
            with open(event_path, "r") as f:
                event = json.load(f)

            # For pull_request events
            pr = event.get("pull_request", {})
            if pr.get("base", {}).get("sha"):
                return pr["base"]["sha"]

            # For push events
            if event.get("before"):
                return event["before"]

        except (json.JSONDecodeError, IOError):
            pass

    # Default
    return "HEAD~1"


def run_action(
    post_comment: bool = True,
    fail_on_high_risk: bool = True,
    output_format: str = "markdown",
) -> int:
    """Run ValueGuard as a GitHub Action.

    Args:
        post_comment: Whether to post a PR comment
        fail_on_high_risk: Whether to fail the workflow on high risk
        output_format: Output format for summary

    Returns:
        Exit code (0=success, 1=high risk, 2=error)
    """
    from valueguard.core.config import load_config
    from valueguard.core.dispatcher import ValueGuardDispatcher
    from valueguard.output.reporter import Reporter
    from valueguard.output.github_comment import post_pr_comment as post_comment_api

    context = get_github_context()

    print(f"::group::ValueGuard Analysis")
    print(f"Repository: {context['repository']}")
    print(f"Event: {context['event_name']}")
    print(f"SHA: {context['sha']}")

    try:
        # Load config from workspace
        workspace = context["workspace"]
        config = load_config(repo_path=workspace)

        # Override with action inputs
        llm_provider = os.environ.get("INPUT_LLM_PROVIDER")
        if llm_provider:
            config.analysis.llm_provider = llm_provider

        # Create dispatcher
        dispatcher = ValueGuardDispatcher(config=config)

        # Get diff parameters
        base_ref = get_base_ref()
        pr_number = get_pr_number()

        print(f"Base ref: {base_ref}")
        print(f"PR number: {pr_number}")

        # Run analysis
        report = dispatcher.analyze_diff(
            repo_path=workspace,
            diff_base=base_ref,
            repo_name=context["repository"],
            pr_number=pr_number,
            commit_sha=context["sha"],
        )

        print(f"::endgroup::")

        # Generate output
        reporter = Reporter(config=config)

        if output_format == "json":
            output = reporter.to_json(report)
        elif output_format == "markdown":
            output = reporter.to_markdown(report)
        else:
            output = reporter.to_console(report)

        # Write to step summary
        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_file:
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(reporter.to_markdown(report))

        # Set outputs
        output_file = os.environ.get("GITHUB_OUTPUT")
        if output_file:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"risk_score={report.overall_risk_score}\n")
                f.write(f"hypothesis_count={len(report.hypotheses)}\n")
                confirmed = sum(1 for e in report.evidences if e.is_confirmed)
                f.write(f"confirmed_count={confirmed}\n")

        # Post PR comment
        if post_comment and pr_number:
            owner, repo = context["repository"].split("/", 1)
            result = post_comment_api(
                report=report,
                owner=owner,
                repo=repo,
                pr_number=pr_number,
            )
            if result.get("success"):
                print(f"::notice::Posted comment: {result.get('comment_url')}")
            else:
                print(f"::warning::Failed to post comment: {result.get('error')}")

        # Print output
        print(output)

        # Determine exit code
        if fail_on_high_risk and report.overall_risk_score >= 0.6:
            print(f"::error::High value risk detected: {report.overall_risk_score:.2f}")
            return 1

        return 0

    except Exception as e:
        print(f"::endgroup::")
        print(f"::error::ValueGuard analysis failed: {e}")
        return 2


def main() -> int:
    """Main entry point for GitHub Action."""
    # Read action inputs from environment
    post_comment = os.environ.get("INPUT_POST_COMMENT", "true").lower() == "true"
    fail_on_risk = os.environ.get("INPUT_FAIL_ON_HIGH_RISK", "true").lower() == "true"
    output_format = os.environ.get("INPUT_OUTPUT_FORMAT", "markdown")

    return run_action(
        post_comment=post_comment,
        fail_on_high_risk=fail_on_risk,
        output_format=output_format,
    )


if __name__ == "__main__":
    sys.exit(main())
