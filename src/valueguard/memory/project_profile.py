"""Project profile memory for storing and evolving value baselines."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from valueguard.core.models import ProfileEvidence, ValueProfile
from valueguard.memory.storage.json_store import BaseStorage


class ProjectProfileMemory:
    """Manages project value profiles with versioning and evolution.

    Key features:
    - Stores value baselines that evolve over time
    - Supports weighted updates for incremental learning
    - Tracks profile confidence based on analysis count
    """

    COLLECTION = "profiles"

    def __init__(self, storage: BaseStorage):
        """Initialize profile memory.

        Args:
            storage: Storage backend to use
        """
        self.storage = storage

    def get(self, repo: str) -> Optional[ValueProfile]:
        """Get profile for a repository.

        Args:
            repo: Repository identifier

        Returns:
            ValueProfile or None if not found
        """
        key = f"{self.COLLECTION}/{self._sanitize_repo(repo)}"
        data = self.storage.get(key)

        if data is None:
            return None

        return self._dict_to_profile(data)

    def store(self, repo: str, profile: ValueProfile) -> None:
        """Store a profile for a repository.

        Args:
            repo: Repository identifier
            profile: Profile to store
        """
        key = f"{self.COLLECTION}/{self._sanitize_repo(repo)}"

        # Update timestamps
        profile.updated_at = datetime.now()

        data = self._profile_to_dict(profile)
        self.storage.store(key, data)

    def update(
        self,
        repo: str,
        new_observations: dict[str, Any],
        existing_weight: float = 0.7,
        new_weight: float = 0.3,
    ) -> ValueProfile:
        """Incrementally update profile based on new analysis.

        Uses weighted merging to combine existing profile with new observations.
        More recent observations can have higher weight.

        Args:
            repo: Repository identifier
            new_observations: New value observations
                - l2_observations: dict[str, float]
                - l3_observations: dict[str, float]
                - evidence_samples: list[ProfileEvidence]
            existing_weight: Weight for existing profile (0-1)
            new_weight: Weight for new observations (0-1)

        Returns:
            Updated ValueProfile
        """
        existing = self.get(repo)

        if existing is None:
            # Create new profile
            return self._create_new(repo, new_observations)

        # Weighted merge of L2 scores
        l2_observations = new_observations.get("l2_observations", {})
        updated_l2 = self._weighted_merge(
            existing.l2_scores,
            l2_observations,
            existing_weight,
            new_weight,
        )

        # Weighted merge of L3 scores
        l3_observations = new_observations.get("l3_observations", {})
        updated_l3 = self._weighted_merge(
            existing.l3_scores,
            l3_observations,
            existing_weight,
            new_weight,
        )

        # Update evidence samples (append new, keep recent)
        new_evidence = new_observations.get("evidence_samples", [])
        updated_evidence = existing.evidence_samples + new_evidence
        # Keep only most recent 20 samples
        updated_evidence = updated_evidence[-20:]

        # Create updated profile
        updated_profile = ValueProfile(
            repo=repo,
            created_at=existing.created_at,
            updated_at=datetime.now(),
            version=existing.version + 1,
            l2_scores=updated_l2,
            l3_scores=updated_l3,
            core_values=self._compute_core_values(updated_l2, updated_l3),
            evidence_samples=updated_evidence,
            analysis_count=existing.analysis_count + 1,
            confidence=self._compute_confidence(existing.analysis_count + 1),
        )

        self.store(repo, updated_profile)
        return updated_profile

    def delete(self, repo: str) -> bool:
        """Delete a profile.

        Args:
            repo: Repository identifier

        Returns:
            True if deleted, False if not found
        """
        key = f"{self.COLLECTION}/{self._sanitize_repo(repo)}"
        return self.storage.delete(key)

    def list_repos(self) -> list[str]:
        """List all repositories with profiles.

        Returns:
            List of repository identifiers
        """
        keys = self.storage.list_keys(f"{self.COLLECTION}/")
        repos = []
        for key in keys:
            # Extract repo name from key
            parts = key.split("/", 1)
            if len(parts) == 2:
                repos.append(parts[1].replace("__", "/"))
        return repos

    def is_stale(self, repo: str, ttl_days: int = 30) -> bool:
        """Check if a profile is stale and needs rebuilding.

        Args:
            repo: Repository identifier
            ttl_days: Time-to-live in days

        Returns:
            True if profile is stale or doesn't exist
        """
        profile = self.get(repo)
        if profile is None:
            return True

        age = datetime.now() - profile.updated_at
        return age > timedelta(days=ttl_days)

    def _create_new(
        self, repo: str, observations: dict[str, Any]
    ) -> ValueProfile:
        """Create a new profile from observations."""
        l2_scores = observations.get("l2_observations", {})
        l3_scores = observations.get("l3_observations", {})
        evidence = observations.get("evidence_samples", [])

        profile = ValueProfile(
            repo=repo,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            l2_scores=l2_scores,
            l3_scores=l3_scores,
            core_values=self._compute_core_values(l2_scores, l3_scores),
            evidence_samples=evidence,
            analysis_count=1,
            confidence=self._compute_confidence(1),
        )

        self.store(repo, profile)
        return profile

    def _weighted_merge(
        self,
        existing: dict[str, float],
        new: dict[str, float],
        existing_weight: float,
        new_weight: float,
    ) -> dict[str, float]:
        """Merge two score dictionaries with weights."""
        result = {}

        # All keys from both dicts
        all_keys = set(existing.keys()) | set(new.keys())

        for key in all_keys:
            existing_val = existing.get(key, 0.0)
            new_val = new.get(key, 0.0)

            if key in existing and key in new:
                # Both have values - weighted average
                result[key] = (
                    existing_val * existing_weight + new_val * new_weight
                )
            elif key in new:
                # Only new has value - use it directly
                result[key] = new_val
            else:
                # Only existing has value - keep it
                result[key] = existing_val

        return result

    def _compute_core_values(
        self,
        l2_scores: dict[str, float],
        l3_scores: dict[str, float],
        top_n: int = 3,
    ) -> list[str]:
        """Compute top-N core values from scores."""
        combined = {**l2_scores, **l3_scores}
        sorted_values = sorted(
            combined.items(), key=lambda x: x[1], reverse=True
        )
        return [v[0] for v in sorted_values[:top_n]]

    def _compute_confidence(self, analysis_count: int) -> float:
        """Compute confidence based on analysis count.

        Confidence grows logarithmically, approaching 1.0.
        """
        import math

        # Logarithmic growth: reaches ~0.9 after 10 analyses
        return min(0.99, 0.5 + 0.4 * math.log10(analysis_count + 1))

    def _sanitize_repo(self, repo: str) -> str:
        """Sanitize repository name for use as key."""
        return repo.replace("/", "__")

    def _profile_to_dict(self, profile: ValueProfile) -> dict[str, Any]:
        """Convert ValueProfile to dictionary for storage."""
        data = asdict(profile)
        # Convert datetime objects
        data["created_at"] = profile.created_at.isoformat()
        data["updated_at"] = profile.updated_at.isoformat()
        return data

    def _dict_to_profile(self, data: dict[str, Any]) -> ValueProfile:
        """Convert dictionary to ValueProfile."""
        # Parse datetime strings
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        # Convert evidence samples
        evidence_samples = []
        for e in data.get("evidence_samples", []):
            if isinstance(e, dict):
                evidence_samples.append(
                    ProfileEvidence(
                        source=e.get("source", ""),
                        content=e.get("content", ""),
                        value_id=e.get("value_id", ""),
                        confidence=e.get("confidence", 0.0),
                    )
                )
            elif isinstance(e, ProfileEvidence):
                evidence_samples.append(e)

        return ValueProfile(
            repo=data.get("repo", ""),
            created_at=data.get("created_at", datetime.now()),
            updated_at=data.get("updated_at", datetime.now()),
            version=data.get("version", 1),
            l2_scores=data.get("l2_scores", {}),
            l3_scores=data.get("l3_scores", {}),
            core_values=data.get("core_values", []),
            evidence_samples=evidence_samples,
            analysis_count=data.get("analysis_count", 0),
            confidence=data.get("confidence", 0.0),
        )
