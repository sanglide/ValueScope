"""Pattern memory for learned value patterns."""

from dataclasses import asdict
from datetime import datetime
from typing import Any, Optional
import uuid

from valueguard.core.models import PatternType, ValuePattern
from valueguard.memory.storage.json_store import BaseStorage


class PatternMemory:
    """Stores learned patterns for improved hypothesis generation.

    Key features:
    - Stores patterns observed during analyses
    - Updates confidence based on repeated observations
    - Provides relevant patterns for context injection
    """

    COLLECTION = "patterns"

    def __init__(self, storage: BaseStorage):
        """Initialize pattern memory.

        Args:
            storage: Storage backend to use
        """
        self.storage = storage

    def store_patterns(self, patterns: list[ValuePattern]) -> None:
        """Store newly observed patterns.

        If a pattern already exists, updates its confidence and seen_count.

        Args:
            patterns: List of patterns to store
        """
        for pattern in patterns:
            existing = self.get(pattern.pattern_id)

            if existing is not None:
                # Update existing pattern
                existing.confidence = min(0.99, existing.confidence + 0.05)
                existing.seen_count += 1
                existing.updated_at = datetime.now()

                # Add repo if not already tracked
                if pattern.repos:
                    for repo in pattern.repos:
                        if repo not in existing.repos:
                            existing.repos.append(repo)

                self._save_pattern(existing)
            else:
                # Store new pattern
                pattern.created_at = datetime.now()
                pattern.updated_at = datetime.now()
                self._save_pattern(pattern)

    def get(self, pattern_id: str) -> Optional[ValuePattern]:
        """Get a pattern by ID.

        Args:
            pattern_id: Pattern identifier

        Returns:
            ValuePattern or None if not found
        """
        key = f"{self.COLLECTION}/{pattern_id}"
        data = self.storage.get(key)

        if data is None:
            return None

        return self._dict_to_pattern(data)

    def get_relevant(
        self,
        repo: Optional[str] = None,
        value_id: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 20,
    ) -> list[ValuePattern]:
        """Get patterns relevant to a repo or value.

        Args:
            repo: Repository to find patterns for (optional)
            value_id: Value ID to filter by (optional)
            min_confidence: Minimum confidence threshold
            limit: Maximum number of patterns to return

        Returns:
            List of relevant ValuePattern objects
        """
        all_patterns = self._get_all_patterns()
        relevant = []

        for pattern in all_patterns:
            # Filter by confidence
            if pattern.confidence < min_confidence:
                continue

            # Filter by value_id if specified
            if value_id and pattern.value_id != value_id:
                continue

            # Score pattern relevance
            score = pattern.confidence

            # Boost if pattern was seen in this repo
            if repo and repo in pattern.repos:
                score += 0.3

            # Boost for high seen_count
            if pattern.seen_count >= 5:
                score += 0.1

            relevant.append((score, pattern))

        # Sort by score and limit
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in relevant[:limit]]

    def get_universal_patterns(
        self,
        min_confidence: float = 0.8,
        min_seen_count: int = 5,
    ) -> list[ValuePattern]:
        """Get high-confidence patterns seen across multiple repos.

        Args:
            min_confidence: Minimum confidence threshold
            min_seen_count: Minimum number of times seen

        Returns:
            List of universal ValuePattern objects
        """
        all_patterns = self._get_all_patterns()

        universal = [
            p
            for p in all_patterns
            if p.confidence >= min_confidence and p.seen_count >= min_seen_count
        ]

        return sorted(universal, key=lambda p: p.confidence, reverse=True)

    def get_patterns_for_value(self, value_id: str) -> list[ValuePattern]:
        """Get all patterns associated with a value.

        Args:
            value_id: Value ID (e.g., "HV9_Privacy")

        Returns:
            List of ValuePattern objects
        """
        all_patterns = self._get_all_patterns()
        return [p for p in all_patterns if p.value_id == value_id]

    def create_pattern(
        self,
        value_id: str,
        code_pattern: str,
        pattern_type: PatternType = PatternType.COMMON_RISK,
        description: str = "",
        repo: Optional[str] = None,
    ) -> ValuePattern:
        """Create and store a new pattern.

        Args:
            value_id: Associated value ID
            code_pattern: Code pattern identifier
            pattern_type: Type of pattern
            description: Human-readable description
            repo: Repository where pattern was found

        Returns:
            Created ValuePattern
        """
        pattern = ValuePattern(
            pattern_id=str(uuid.uuid4())[:8],
            pattern_type=pattern_type,
            value_id=value_id,
            code_pattern=code_pattern,
            description=description,
            confidence=0.5,
            seen_count=1,
            repos=[repo] if repo else [],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        self._save_pattern(pattern)
        return pattern

    def delete(self, pattern_id: str) -> bool:
        """Delete a pattern.

        Args:
            pattern_id: Pattern identifier

        Returns:
            True if deleted, False if not found
        """
        key = f"{self.COLLECTION}/{pattern_id}"
        return self.storage.delete(key)

    def _save_pattern(self, pattern: ValuePattern) -> None:
        """Save a pattern to storage."""
        key = f"{self.COLLECTION}/{pattern.pattern_id}"
        data = self._pattern_to_dict(pattern)
        self.storage.store(key, data)

    def _get_all_patterns(self) -> list[ValuePattern]:
        """Get all stored patterns."""
        keys = self.storage.list_keys(f"{self.COLLECTION}/")
        patterns = []

        for key in keys:
            data = self.storage.get(key)
            if data:
                pattern = self._dict_to_pattern(data)
                patterns.append(pattern)

        return patterns

    def _pattern_to_dict(self, pattern: ValuePattern) -> dict[str, Any]:
        """Convert ValuePattern to dictionary."""
        return {
            "pattern_id": pattern.pattern_id,
            "pattern_type": (
                pattern.pattern_type.value
                if hasattr(pattern.pattern_type, "value")
                else str(pattern.pattern_type)
            ),
            "value_id": pattern.value_id,
            "code_pattern": pattern.code_pattern,
            "description": pattern.description,
            "confidence": pattern.confidence,
            "seen_count": pattern.seen_count,
            "repos": pattern.repos,
            "created_at": pattern.created_at.isoformat(),
            "updated_at": pattern.updated_at.isoformat(),
        }

    def _dict_to_pattern(self, data: dict[str, Any]) -> ValuePattern:
        """Convert dictionary to ValuePattern."""
        # Parse datetime strings
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now()

        # Parse pattern type
        pattern_type_str = data.get("pattern_type", "common_risk")
        try:
            pattern_type = PatternType(pattern_type_str)
        except ValueError:
            pattern_type = PatternType.COMMON_RISK

        return ValuePattern(
            pattern_id=data.get("pattern_id", ""),
            pattern_type=pattern_type,
            value_id=data.get("value_id", ""),
            code_pattern=data.get("code_pattern", ""),
            description=data.get("description", ""),
            confidence=data.get("confidence", 0.5),
            seen_count=data.get("seen_count", 1),
            repos=data.get("repos", []),
            created_at=created_at,
            updated_at=updated_at,
        )
