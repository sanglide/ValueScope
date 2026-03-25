"""Memory manager for unified memory operations."""

from typing import Any, Optional

from valueguard.core.models import (
    AnalysisRecord,
    EvidenceResult,
    PatternType,
    ValueHypothesis,
    ValuePattern,
    ValueProfile,
)
from valueguard.memory.analysis_history import AnalysisHistoryMemory
from valueguard.memory.pattern_memory import PatternMemory
from valueguard.memory.project_profile import ProjectProfileMemory
from valueguard.memory.storage.json_store import BaseStorage, JsonStorage


class MemoryManager:
    """Unified interface for all memory operations.

    Coordinates access to:
    - Project profiles (value baselines)
    - Analysis history (audit trail)
    - Pattern memory (learned patterns)
    """

    def __init__(
        self,
        storage: Optional[BaseStorage] = None,
        storage_path: str = ".valueguard/memory",
        history_retention: int = 100,
    ):
        """Initialize memory manager.

        Args:
            storage: Storage backend (creates JsonStorage if not provided)
            storage_path: Path for storage (used if storage not provided)
            history_retention: Maximum history records per repo
        """
        if storage is None:
            storage = JsonStorage(storage_path)

        self.storage = storage
        self.profile_memory = ProjectProfileMemory(storage)
        self.history_memory = AnalysisHistoryMemory(storage, history_retention)
        self.pattern_memory = PatternMemory(storage)

    # --- Profile Operations ---

    def get_profile(self, repo: str) -> Optional[ValueProfile]:
        """Get value profile for a repository.

        Args:
            repo: Repository identifier

        Returns:
            ValueProfile or None if not found
        """
        return self.profile_memory.get(repo)

    def store_profile(self, repo: str, profile: ValueProfile) -> None:
        """Store value profile for a repository.

        Args:
            repo: Repository identifier
            profile: Profile to store
        """
        self.profile_memory.store(repo, profile)

    def update_profile(
        self,
        repo: str,
        new_observations: dict[str, Any],
        existing_weight: float = 0.7,
        new_weight: float = 0.3,
    ) -> ValueProfile:
        """Incrementally update profile based on new analysis.

        Args:
            repo: Repository identifier
            new_observations: New value observations
            existing_weight: Weight for existing profile
            new_weight: Weight for new observations

        Returns:
            Updated ValueProfile
        """
        return self.profile_memory.update(
            repo, new_observations, existing_weight, new_weight
        )

    def is_profile_stale(self, repo: str, ttl_days: int = 30) -> bool:
        """Check if a profile needs rebuilding.

        Args:
            repo: Repository identifier
            ttl_days: Time-to-live in days

        Returns:
            True if profile is stale or doesn't exist
        """
        return self.profile_memory.is_stale(repo, ttl_days)

    # --- History Operations ---

    def record_analysis(
        self,
        repo: str,
        hypotheses: list[ValueHypothesis],
        evidences: list[EvidenceResult],
        event_type: str = "pr",
        pr_number: Optional[int] = None,
        commit_sha: Optional[str] = None,
    ) -> AnalysisRecord:
        """Record a new analysis.

        Also extracts and stores patterns from the analysis.

        Args:
            repo: Repository identifier
            hypotheses: Hypotheses generated
            evidences: Evidence results
            event_type: Type of triggering event
            pr_number: PR number if applicable
            commit_sha: Commit SHA if applicable

        Returns:
            The created AnalysisRecord
        """
        # Record the analysis
        record = self.history_memory.record(
            repo=repo,
            hypotheses=hypotheses,
            evidences=evidences,
            event_type=event_type,
            pr_number=pr_number,
            commit_sha=commit_sha,
        )

        # Extract and store patterns
        patterns = self._extract_patterns(repo, hypotheses, evidences)
        if patterns:
            self.pattern_memory.store_patterns(patterns)

        return record

    def get_analysis_history(
        self, repo: str, limit: int = 10
    ) -> list[AnalysisRecord]:
        """Get recent analysis history.

        Args:
            repo: Repository identifier
            limit: Maximum records to return

        Returns:
            List of AnalysisRecord, most recent first
        """
        return self.history_memory.get_recent(repo, limit)

    def get_analysis_statistics(self, repo: str) -> dict[str, Any]:
        """Get statistics about analyses for a repository.

        Args:
            repo: Repository identifier

        Returns:
            Statistics dictionary
        """
        return self.history_memory.get_statistics(repo)

    # --- Pattern Operations ---

    def get_relevant_patterns(
        self,
        repo: Optional[str] = None,
        value_id: Optional[str] = None,
        min_confidence: float = 0.3,
    ) -> list[ValuePattern]:
        """Get patterns relevant to a repo or value.

        Combines repo-specific and universal patterns.

        Args:
            repo: Repository identifier (optional)
            value_id: Value ID to filter by (optional)
            min_confidence: Minimum confidence threshold

        Returns:
            List of relevant ValuePattern objects
        """
        return self.pattern_memory.get_relevant(
            repo=repo,
            value_id=value_id,
            min_confidence=min_confidence,
        )

    def store_pattern(self, pattern: ValuePattern) -> None:
        """Store a single pattern.

        Args:
            pattern: Pattern to store
        """
        self.pattern_memory.store_patterns([pattern])

    # --- Utility Operations ---

    def clear_repo(self, repo: str) -> None:
        """Clear all memory for a repository.

        Args:
            repo: Repository identifier
        """
        self.profile_memory.delete(repo)
        self.history_memory.clear(repo)

    def get_memory_summary(self, repo: str) -> dict[str, Any]:
        """Get a summary of all memory for a repository.

        Args:
            repo: Repository identifier

        Returns:
            Summary dictionary
        """
        profile = self.get_profile(repo)
        history_stats = self.get_analysis_statistics(repo)
        patterns = self.get_relevant_patterns(repo=repo)

        return {
            "has_profile": profile is not None,
            "profile_version": profile.version if profile else 0,
            "profile_confidence": profile.confidence if profile else 0.0,
            "core_values": profile.core_values if profile else [],
            "analysis_count": history_stats.get("total_analyses", 0),
            "pattern_count": len(patterns),
            "confirmation_rate": history_stats.get("confirmation_rate", 0.0),
        }

    def _extract_patterns(
        self,
        repo: str,
        hypotheses: list[ValueHypothesis],
        evidences: list[EvidenceResult],
    ) -> list[ValuePattern]:
        """Extract patterns from analysis results.

        Identifies confirmed hypotheses as potential patterns.

        Args:
            repo: Repository identifier
            hypotheses: Generated hypotheses
            evidences: Evidence results

        Returns:
            List of extracted patterns
        """
        patterns = []

        # Build evidence lookup
        evidence_map = {e.hypothesis_id: e for e in evidences}

        for hypothesis in hypotheses:
            evidence = evidence_map.get(hypothesis.id)

            # Only create patterns for confirmed hypotheses with high confidence
            if (
                evidence
                and evidence.is_confirmed
                and hypothesis.confidence >= 0.6
            ):
                # Determine pattern type
                pattern_type = PatternType.TRUE_POSITIVE

                # Extract code pattern from hypothesis description
                code_pattern = self._extract_code_pattern(hypothesis)

                if code_pattern:
                    pattern = ValuePattern(
                        pattern_type=pattern_type,
                        value_id=hypothesis.value_id,
                        code_pattern=code_pattern,
                        description=hypothesis.description[:200],
                        confidence=hypothesis.confidence,
                        seen_count=1,
                        repos=[repo],
                    )
                    patterns.append(pattern)

        return patterns

    def _extract_code_pattern(self, hypothesis: ValueHypothesis) -> str:
        """Extract a code pattern identifier from a hypothesis.

        Args:
            hypothesis: Hypothesis to extract pattern from

        Returns:
            Code pattern string or empty string
        """
        # Simple extraction based on deviation type and description
        if hypothesis.deviation_type and hypothesis.value_id:
            # Create pattern ID from value + deviation type
            pattern = f"{hypothesis.value_id}_{hypothesis.deviation_type}"

            # Add hint from description if available
            desc_lower = hypothesis.description.lower()
            if "fallback" in desc_lower:
                pattern += "_fallback"
            elif "unencrypt" in desc_lower:
                pattern += "_unencrypted"
            elif "log" in desc_lower and "sensitive" in desc_lower:
                pattern += "_sensitive_logging"
            elif "permission" in desc_lower:
                pattern += "_permission"

            return pattern

        return ""
