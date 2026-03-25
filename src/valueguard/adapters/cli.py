"""Command-line interface for ValueGuard."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for ValueGuard CLI."""
    parser = argparse.ArgumentParser(
        prog="valueguard",
        description="ValueGuard - Detect value deviations in code commits",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze code changes for value deviations"
    )
    analyze_parser.add_argument(
        "--repo-path",
        "-r",
        default=".",
        help="Path to the repository (default: current directory)",
    )
    analyze_parser.add_argument(
        "--diff-base",
        "-b",
        default="HEAD~1",
        help="Base commit for diff comparison (default: HEAD~1)",
    )
    analyze_parser.add_argument(
        "--diff-target",
        "-t",
        default="HEAD",
        help="Target commit for diff (default: HEAD)",
    )
    analyze_parser.add_argument(
        "--output",
        "-o",
        choices=["json", "markdown", "console"],
        default="console",
        help="Output format (default: console)",
    )
    analyze_parser.add_argument(
        "--output-file",
        "-f",
        help="Write output to file instead of stdout",
    )
    analyze_parser.add_argument(
        "--config",
        "-c",
        help="Path to .valueguard.yml config file",
    )
    analyze_parser.add_argument(
        "--llm-provider",
        choices=["deepseek", "openai", "anthropic", "qwen"],
        help="LLM provider to use",
    )
    analyze_parser.add_argument(
        "--rebuild-profile",
        action="store_true",
        help="Force rebuild of project value profile",
    )

    # profile command
    profile_parser = subparsers.add_parser(
        "profile", help="Manage project value profiles"
    )
    profile_parser.add_argument(
        "action",
        choices=["show", "build", "clear"],
        help="Profile action",
    )
    profile_parser.add_argument(
        "--repo-path",
        "-r",
        default=".",
        help="Path to the repository",
    )

    # memory command
    memory_parser = subparsers.add_parser(
        "memory", help="Manage ValueGuard memory"
    )
    memory_parser.add_argument(
        "action",
        choices=["show", "clear", "stats"],
        help="Memory action",
    )
    memory_parser.add_argument(
        "--repo",
        help="Repository identifier",
    )

    # init command
    init_parser = subparsers.add_parser(
        "init", help="Initialize ValueGuard for a repository"
    )
    init_parser.add_argument(
        "--repo-path",
        "-r",
        default=".",
        help="Path to the repository",
    )

    return parser


def cmd_analyze(args: argparse.Namespace) -> int:
    """Execute analyze command."""
    from valueguard.core.config import load_config
    from valueguard.core.dispatcher import ValueGuardDispatcher
    from valueguard.output.reporter import Reporter

    # Load config
    config = load_config(
        repo_path=args.repo_path,
        config_file=args.config,
    )

    # Override LLM provider if specified
    if args.llm_provider:
        config.analysis.llm_provider = args.llm_provider

    # Create dispatcher
    dispatcher = ValueGuardDispatcher(config=config)

    # Run analysis
    print(f"Analyzing {args.repo_path}...", file=sys.stderr)
    print(f"Diff: {args.diff_base}...{args.diff_target}", file=sys.stderr)

    try:
        report = dispatcher.analyze_diff(
            repo_path=args.repo_path,
            diff_base=args.diff_base,
            diff_target=args.diff_target,
        )
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        return 1

    # Generate output
    reporter = Reporter(config=config)

    if args.output == "json":
        output = reporter.to_json(report)
    elif args.output == "markdown":
        output = reporter.to_markdown(report)
    else:
        output = reporter.to_console(report)

    # Write output
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Output written to {args.output_file}", file=sys.stderr)
    else:
        print(output)

    # Return exit code based on risk
    if report.overall_risk_score >= 0.8:
        return 2  # Critical
    elif report.overall_risk_score >= 0.6:
        return 1  # High
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """Execute profile command."""
    from valueguard.core.config import load_config
    from valueguard.core.models import ProfileTask
    from valueguard.memory.manager import MemoryManager
    from valueguard.skills.registry import SkillRegistry
    from valueguard.agents.profiler_agent import ProfilerAgent

    config = load_config(repo_path=args.repo_path)
    repo_name = os.path.basename(os.path.abspath(args.repo_path))

    memory = MemoryManager(storage_path=config.memory.storage_path)

    if args.action == "show":
        profile = memory.get_profile(repo_name)
        if profile:
            print(f"Profile for: {profile.repo}")
            print(f"Version: {profile.version}")
            print(f"Confidence: {profile.confidence:.2f}")
            print(f"Core values: {', '.join(profile.core_values)}")
            print(f"L2 scores: {profile.l2_scores}")
            print(f"L3 scores: {profile.l3_scores}")
        else:
            print(f"No profile found for {repo_name}")
        return 0

    elif args.action == "build":
        skills = SkillRegistry()
        skills.auto_discover()

        agent_config = {"repo_path": args.repo_path}
        profiler = ProfilerAgent(skills=skills, memory=memory, config=agent_config)

        task = ProfileTask(repo=repo_name, rebuild=True)
        profile = profiler.execute(task)

        print(f"Profile built for: {profile.repo}")
        print(f"Core values: {', '.join(profile.core_values)}")
        return 0

    elif args.action == "clear":
        memory.profile_memory.delete(repo_name)
        print(f"Profile cleared for {repo_name}")
        return 0

    return 1


def cmd_memory(args: argparse.Namespace) -> int:
    """Execute memory command."""
    from valueguard.core.config import load_config
    from valueguard.memory.manager import MemoryManager

    config = load_config()
    memory = MemoryManager(storage_path=config.memory.storage_path)

    if args.action == "show":
        if args.repo:
            summary = memory.get_memory_summary(args.repo)
            print(json.dumps(summary, indent=2))
        else:
            repos = memory.profile_memory.list_repos()
            print(f"Repositories with profiles: {len(repos)}")
            for repo in repos:
                print(f"  - {repo}")
        return 0

    elif args.action == "stats":
        if args.repo:
            stats = memory.get_analysis_statistics(args.repo)
            print(json.dumps(stats, indent=2, default=str))
        else:
            print("Please specify --repo for stats")
            return 1
        return 0

    elif args.action == "clear":
        if args.repo:
            memory.clear_repo(args.repo)
            print(f"Memory cleared for {args.repo}")
        else:
            print("Please specify --repo to clear")
            return 1
        return 0

    return 1


def cmd_init(args: argparse.Namespace) -> int:
    """Execute init command."""
    repo_path = Path(args.repo_path).resolve()

    config_file = repo_path / ".valueguard.yml"
    if config_file.exists():
        print(f"ValueGuard config already exists: {config_file}")
        return 0

    # Create default config
    default_config = """# ValueGuard Configuration
# See documentation for all options

# Core values for this project (auto-detected if not specified)
# core_values:
#   - HV9_Privacy
#   - HV10_Security

# Analysis settings
analysis:
  llm_provider: deepseek
  confidence_threshold: 0.5
  max_hypotheses: 10

# Memory settings
memory:
  storage_path: .valueguard/memory
  profile_ttl_days: 30

# Output settings
output:
  post_pr_comment: true
  mention_reviewers: true
  reviewer_teams:
    security: "@security-team"
"""

    with open(config_file, "w", encoding="utf-8") as f:
        f.write(default_config)

    print(f"Created ValueGuard config: {config_file}")

    # Create memory directory
    memory_dir = repo_path / ".valueguard"
    memory_dir.mkdir(exist_ok=True)

    # Add to .gitignore if exists
    gitignore = repo_path / ".gitignore"
    if gitignore.exists():
        with open(gitignore, "r", encoding="utf-8") as f:
            content = f.read()
        if ".valueguard/" not in content:
            with open(gitignore, "a", encoding="utf-8") as f:
                f.write("\n# ValueGuard\n.valueguard/\n")
            print("Added .valueguard/ to .gitignore")

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "analyze": cmd_analyze,
        "profile": cmd_profile,
        "memory": cmd_memory,
        "init": cmd_init,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        return cmd_func(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
