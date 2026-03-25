"""Central dispatcher for ValueGuard."""

import concurrent.futures
from typing import Any, Optional

from valueguard.agents.evidence_agent import EvidenceAgent
from valueguard.agents.hypothesis_agent import HypothesisAgent
from valueguard.agents.profiler_agent import ProfilerAgent
from valueguard.core.config import Config, load_config
from valueguard.core.models import (
    AnalysisEvent,
    DiffHunk,
    EvidenceResult,
    EvidenceTask,
    HypothesisTask,
    ProfileTask,
    ValueGuardReport,
    ValueHypothesis,
    ValueProfile,
    ValueScore,
)
from valueguard.memory.manager import MemoryManager
from valueguard.output.reporter import Reporter
from valueguard.skills.registry import SkillRegistry


class ValueGuardDispatcher:
    """Central task dispatcher with agent routing and memory integration.

    The dispatcher is the main orchestrator that:
    - Receives analysis requests (PR event, CLI command)
    - Decomposes requests into typed tasks
    - Routes tasks to appropriate agents
    - Manages task dependencies and parallel execution
    - Aggregates results from multiple agents
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        memory: Optional[MemoryManager] = None,
        skill_registry: Optional[SkillRegistry] = None,
    ):
        """Initialize the dispatcher.

        Args:
            config: ValueGuard configuration (loads default if not provided)
            memory: Memory manager (creates default if not provided)
            skill_registry: Skill registry (creates and auto-discovers if not provided)
        """
        self.config = config or load_config()
        
        # Initialize memory
        if memory is None:
            memory = MemoryManager(
                storage_path=self.config.memory.storage_path,
                history_retention=self.config.memory.history_retention,
            )
        self.memory = memory

        # Initialize skill registry
        if skill_registry is None:
            skill_registry = SkillRegistry()
            skill_registry.auto_discover()
        self.skills = skill_registry

        # Initialize specialized agents
        agent_config = {
            "repo_path": self.config.repo_path,
            "llm_provider": self.config.analysis.llm_provider,
            "max_hypotheses": self.config.analysis.max_hypotheses,
        }

        self.agents = {
            "profiler": ProfilerAgent(
                skills=self.skills, memory=self.memory, config=agent_config
            ),
            "hypothesis": HypothesisAgent(
                skills=self.skills, memory=self.memory, config=agent_config
            ),
            "evidence": EvidenceAgent(
                skills=self.skills, memory=self.memory, config=agent_config
            ),
        }

        self.reporter = Reporter(config=self.config)

    def dispatch(self, event: AnalysisEvent) -> ValueGuardReport:
        """Main entry point: dispatch analysis tasks and aggregate results.

        Args:
            event: Analysis event with repo info and diff hunks

        Returns:
            Complete ValueGuardReport
        """
        # 1. Load or build project profile (with memory)
        profile_task = ProfileTask(
            repo=event.repo,
            rebuild=event.force_rebuild,
            sources=["readme", "config"],
        )
        profile = self.agents["profiler"].execute(profile_task)

        # 2. Generate value hypotheses (uses profile as context)
        hypothesis_task = HypothesisTask(
            diff_hunks=event.diff_hunks,
            profile=profile,
            memory_context=self.memory.get_relevant_patterns(repo=event.repo),
            max_hypotheses=self.config.analysis.max_hypotheses,
        )
        hypotheses = self.agents["hypothesis"].execute(hypothesis_task)

        # 3. Locate evidence for each hypothesis (can parallelize)
        evidence_tasks = [
            EvidenceTask(h, event.repo_path, search_depth=3) for h in hypotheses
        ]
        evidences = self._parallel_execute_evidence(evidence_tasks)

        # 4. Update memory with analysis results
        self.memory.record_analysis(
            repo=event.repo,
            hypotheses=hypotheses,
            evidences=evidences,
            event_type=event.event_type,
            pr_number=event.pr_number,
            commit_sha=event.commit_sha,
        )

        # 5. Generate report
        return self._build_report(event, profile, hypotheses, evidences)

    def analyze_diff(
        self,
        repo_path: str,
        diff_base: str = "HEAD~1",
        diff_target: str = "HEAD",
        repo_name: Optional[str] = None,
        pr_number: Optional[int] = None,
        commit_sha: Optional[str] = None,
    ) -> ValueGuardReport:
        """Convenience method to analyze a diff directly.

        Args:
            repo_path: Path to the repository
            diff_base: Base commit for diff
            diff_target: Target commit for diff
            repo_name: Repository name (defaults to repo_path basename)
            pr_number: PR number if applicable
            commit_sha: Commit SHA if applicable

        Returns:
            ValueGuardReport
        """
        import os

        # Get diff hunks using code_chunking skill
        diff_hunks: list[DiffHunk] = []
        if "code_chunking" in self.skills:
            diff_hunks = self.skills.invoke(
                "code_chunking",
                repo_path=repo_path,
                diff_base=diff_base,
                diff_target=diff_target,
            )

        # Create analysis event
        event = AnalysisEvent(
            repo=repo_name or os.path.basename(repo_path),
            repo_path=repo_path,
            diff_hunks=diff_hunks,
            event_type="pr" if pr_number else "manual",
            pr_number=pr_number,
            commit_sha=commit_sha,
        )

        return self.dispatch(event)

    def _parallel_execute_evidence(
        self, tasks: list[EvidenceTask]
    ) -> list[EvidenceResult]:
        """Execute evidence tasks in parallel.

        Args:
            tasks: List of evidence tasks

        Returns:
            List of evidence results
        """
        if not tasks:
            return []

        # Limit parallelism to avoid overwhelming resources
        max_workers = min(4, len(tasks))
        evidence_agent = self.agents["evidence"]

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evidence_agent.execute, task): task for task in tasks
            }

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception:
                    # Create empty result on error
                    task = futures[future]
                    results.append(
                        EvidenceResult(hypothesis_id=task.hypothesis.id)
                    )

        return results

    def _build_report(
        self,
        event: AnalysisEvent,
        profile: ValueProfile,
        hypotheses: list[ValueHypothesis],
        evidences: list[EvidenceResult],
    ) -> ValueGuardReport:
        """Build the final report from analysis results.

        Args:
            event: Original analysis event
            profile: Project value profile
            hypotheses: Generated hypotheses
            evidences: Evidence results

        Returns:
            Complete ValueGuardReport
        """
        from datetime import datetime

        # Build evidence lookup
        evidence_map = {e.hypothesis_id: e for e in evidences}

        # Calculate value scores
        value_scores = self._calculate_value_scores(hypotheses, evidence_map)

        # Calculate overall risk score
        overall_risk = self._calculate_overall_risk(value_scores)

        # Determine mentions based on confirmed hypotheses
        mentions = self._determine_mentions(hypotheses, evidence_map)

        return ValueGuardReport(
            repo=event.repo,
            timestamp=datetime.now(),
            event_type=event.event_type,
            pr_number=event.pr_number,
            commit_sha=event.commit_sha,
            overall_risk_score=overall_risk,
            value_scores=value_scores,
            hypotheses=hypotheses,
            evidences=evidences,
            profile=profile,
            mentions=mentions,
        )

    def _calculate_value_scores(
        self,
        hypotheses: list[ValueHypothesis],
        evidence_map: dict[str, EvidenceResult],
    ) -> list[ValueScore]:
        """Calculate per-value risk scores."""
        # Group hypotheses by value
        value_hypotheses: dict[str, list[ValueHypothesis]] = {}
        for h in hypotheses:
            if h.value_id not in value_hypotheses:
                value_hypotheses[h.value_id] = []
            value_hypotheses[h.value_id].append(h)

        scores = []
        for value_id, hypos in value_hypotheses.items():
            # Count confirmed hypotheses
            confirmed = sum(
                1
                for h in hypos
                if h.id in evidence_map and evidence_map[h.id].is_confirmed
            )

            # Calculate score based on confidence and confirmation
            total_confidence = sum(h.confidence for h in hypos)
            confirmed_confidence = sum(
                h.confidence
                for h in hypos
                if h.id in evidence_map and evidence_map[h.id].is_confirmed
            )

            score = confirmed_confidence / len(hypos) if hypos else 0.0

            # Determine severity based on max hypothesis severity
            severity_order = ["low", "medium", "high", "critical"]
            max_severity = max(
                hypos, key=lambda h: severity_order.index(h.severity)
            ).severity

            scores.append(
                ValueScore(
                    value_id=value_id,
                    value_name=self._get_value_name(value_id),
                    score=min(1.0, score),
                    severity=max_severity,
                    hypothesis_count=len(hypos),
                    evidence_count=confirmed,
                )
            )

        # Sort by score descending
        scores.sort(key=lambda s: s.score, reverse=True)
        return scores

    def _calculate_overall_risk(self, value_scores: list[ValueScore]) -> float:
        """Calculate overall risk score from value scores."""
        if not value_scores:
            return 0.0

        # Weighted average with severity boost
        severity_weights = {"low": 0.5, "medium": 1.0, "high": 1.5, "critical": 2.0}

        weighted_sum = sum(
            s.score * severity_weights.get(s.severity, 1.0) for s in value_scores
        )
        total_weight = sum(
            severity_weights.get(s.severity, 1.0) for s in value_scores
        )

        return min(1.0, weighted_sum / total_weight) if total_weight > 0 else 0.0

    def _determine_mentions(
        self,
        hypotheses: list[ValueHypothesis],
        evidence_map: dict[str, EvidenceResult],
    ) -> list[str]:
        """Determine who should be @mentioned based on findings."""
        if not self.config.output.mention_reviewers:
            return []

        mentions = set()
        reviewer_teams = self.config.output.reviewer_teams

        # Check for confirmed high-severity hypotheses
        for h in hypotheses:
            if h.id in evidence_map and evidence_map[h.id].is_confirmed:
                if h.severity in ("high", "critical"):
                    # Add relevant team
                    value_lower = h.value_id.lower()
                    for keyword, team in reviewer_teams.items():
                        if keyword in value_lower:
                            mentions.add(team)

        return list(mentions)

    def _get_value_name(self, value_id: str) -> str:
        """Get human-readable name for a value ID."""
        # Simple mapping - could be expanded with value_model skill
        names = {
            "HV1": "Conformity",
            "HV2": "Pleasure",
            "HV3": "Dignity",
            "HV4": "Inclusiveness",
            "HV5": "Sense of Belonging",
            "HV6": "Freedom",
            "HV7": "Independence",
            "HV8": "Wealth",
            "HV9": "Privacy",
            "HV10": "Security",
            "SV1": "Trust",
            "SV2": "Correctness",
            "SV3": "Compatibility",
            "SV4": "Portability",
            "SV5": "Reliability",
            "SV6": "Efficiency",
            "SV7": "Energy Preservation",
            "SV8": "Usability",
            "SV9": "Accessibility",
            "SV10": "Longevity",
        }
        return names.get(value_id, value_id)
