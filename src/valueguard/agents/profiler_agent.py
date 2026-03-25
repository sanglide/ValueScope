"""Profiler agent for building project value profiles."""

from typing import Any, Optional

from valueguard.agents.base_agent import BaseAgent
from valueguard.core.models import ProfileEvidence, ProfileTask, ValueProfile
from valueguard.memory.manager import MemoryManager
from valueguard.skills.registry import SkillRegistry


# Prompt for value classification
VALUE_CLASSIFICATION_SYSTEM = """You are a value analyst for software projects.
Your task is to identify human and system values expressed in project documentation.

Given a piece of text from a project (README, config, etc.), identify which values
from the L2 (Human Value Themes) and L3 (System Value Themes) are expressed or implied.

## L2 Human Value Themes:
- HV1 (Conformity): Following rules, meeting expectations
- HV2 (Pleasure): User enjoyment, satisfaction
- HV3 (Dignity): Respect for users, ethical treatment
- HV4 (Inclusiveness): Accessibility, supporting diverse users
- HV5 (Sense of belonging): Community, connection
- HV6 (Freedom): User autonomy, choice
- HV7 (Independence): Self-sufficiency, not locked-in
- HV8 (Wealth): Economic value, efficiency
- HV9 (Privacy): Data protection, user control over information
- HV10 (Security): Safety, protection from harm

## L3 System Value Themes:
- SV1 (Trust): Reliability of the system
- SV2 (Correctness): Accuracy, bug-free operation
- SV3 (Compatibility): Works with other systems
- SV4 (Portability): Works across platforms
- SV5 (Reliability): Consistent operation
- SV6 (Efficiency): Performance, resource usage
- SV7 (Energy Preservation): Green computing
- SV8 (Usability): Ease of use
- SV9 (Accessibility): Support for users with disabilities
- SV10 (Longevity): Long-term maintainability

Respond in JSON format with:
{
  "l2_values": [{"value_id": "HVX", "confidence": 0.0-1.0, "evidence": "quote or reason"}],
  "l3_values": [{"value_id": "SVX", "confidence": 0.0-1.0, "evidence": "quote or reason"}]
}

Only include values with confidence >= 0.5.
"""


class ProfilerAgent(BaseAgent):
    """Agent for building and maintaining project value profiles.

    Responsibilities:
    - Collect documents from project sources
    - Classify documents to identify expressed values
    - Aggregate classifications into a coherent profile
    - Store and update profiles in memory
    """

    name = "profiler"
    role = "Value Profile Analyst"
    goal = "Build accurate project value baselines from historical data"

    def __init__(
        self,
        skills: SkillRegistry,
        memory: MemoryManager,
        config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(skills, memory, config)
        self._llm_provider = self.get_config("llm_provider", "deepseek")

    def execute(self, task: ProfileTask) -> ValueProfile:
        """Build or retrieve a value profile for a repository.

        Args:
            task: ProfileTask with repo and rebuild settings

        Returns:
            ValueProfile for the repository
        """
        # Check cache first (unless rebuild requested)
        if not task.rebuild:
            cached = self.memory.get_profile(task.repo)
            if cached and not self.memory.is_profile_stale(task.repo):
                return cached

        # Collect documents
        docs = self._collect_documents(task)

        # Classify each document
        classifications = []
        for doc in docs:
            result = self._classify_document(doc)
            if result:
                classifications.append(result)

        # Aggregate into profile
        profile = self._aggregate_profile(task.repo, classifications)

        # Store in memory
        self.memory.store_profile(task.repo, profile)

        return profile

    def _collect_documents(self, task: ProfileTask) -> list[dict[str, Any]]:
        """Collect documents from project sources."""
        if not self.has_skill("doc_collector"):
            return []

        # Get repo path from config or use default
        repo_path = self.get_config("repo_path", ".")

        docs = self.invoke_skill(
            "doc_collector",
            repo_path=repo_path,
            sources=task.sources,
            max_length=8000,
        )

        # Convert CollectedDocument objects to dicts
        return [
            {
                "source_type": doc.source_type,
                "file_path": doc.file_path,
                "content": doc.content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]

    def _classify_document(self, doc: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Classify a document to identify expressed values."""
        if not self.has_skill("llm_call"):
            return None

        content = doc.get("content", "")
        if not content or len(content) < 50:
            return None

        try:
            response = self.invoke_skill(
                "llm_call",
                system=VALUE_CLASSIFICATION_SYSTEM,
                user=f"Analyze this project text and identify the values:\n\n{content}",
                provider=self._llm_provider,
                parse_json=True,
            )

            if response.error:
                return None

            parsed = response.parsed_result
            if not parsed:
                # Try parsing with json_parser skill
                if self.has_skill("json_parser"):
                    parsed = self.invoke_skill(
                        "json_parser",
                        response=response.raw_response,
                        strict=False,
                        default={},
                    )

            if parsed:
                return {
                    "source": doc.get("source_type", "unknown"),
                    "file_path": doc.get("file_path", ""),
                    "l2_values": parsed.get("l2_values", []),
                    "l3_values": parsed.get("l3_values", []),
                }

        except Exception:
            pass

        return None

    def _aggregate_profile(
        self, repo: str, classifications: list[dict[str, Any]]
    ) -> ValueProfile:
        """Aggregate classifications into a value profile."""
        # Aggregate L2 scores
        l2_scores: dict[str, list[float]] = {}
        l2_evidence: dict[str, list[ProfileEvidence]] = {}

        for cls in classifications:
            for v in cls.get("l2_values", []):
                vid = v.get("value_id", "")
                conf = v.get("confidence", 0.0)
                if vid and conf >= 0.5:
                    if vid not in l2_scores:
                        l2_scores[vid] = []
                        l2_evidence[vid] = []
                    l2_scores[vid].append(conf)
                    l2_evidence[vid].append(
                        ProfileEvidence(
                            source=cls.get("source", ""),
                            content=v.get("evidence", "")[:200],
                            value_id=vid,
                            confidence=conf,
                        )
                    )

        # Aggregate L3 scores
        l3_scores: dict[str, list[float]] = {}

        for cls in classifications:
            for v in cls.get("l3_values", []):
                vid = v.get("value_id", "")
                conf = v.get("confidence", 0.0)
                if vid and conf >= 0.5:
                    if vid not in l3_scores:
                        l3_scores[vid] = []
                    l3_scores[vid].append(conf)

        # Compute final scores (average of all observations)
        final_l2 = {vid: sum(scores) / len(scores) for vid, scores in l2_scores.items()}
        final_l3 = {vid: sum(scores) / len(scores) for vid, scores in l3_scores.items()}

        # Collect evidence samples (top 2 per value)
        evidence_samples = []
        for vid, samples in l2_evidence.items():
            sorted_samples = sorted(samples, key=lambda e: e.confidence, reverse=True)
            evidence_samples.extend(sorted_samples[:2])

        # Determine core values
        combined = {**final_l2, **final_l3}
        sorted_values = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        core_values = [v[0] for v in sorted_values[:3]]

        # Compute confidence based on data volume
        import math
        data_points = len(classifications)
        confidence = min(0.99, 0.5 + 0.4 * math.log10(data_points + 1))

        from datetime import datetime

        return ValueProfile(
            repo=repo,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            l2_scores=final_l2,
            l3_scores=final_l3,
            core_values=core_values,
            evidence_samples=evidence_samples[:10],
            analysis_count=1,
            confidence=confidence,
        )
