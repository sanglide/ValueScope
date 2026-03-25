"""Analysis history memory for audit trail of analyses."""

from dataclasses import asdict
from datetime import datetime
from typing import Any, Optional

from valueguard.core.models import (
    AnalysisRecord,
    EvidenceResult,
    ValueHypothesis,
)
from valueguard.memory.storage.json_store import BaseStorage


class AnalysisHistoryMemory:
    """Manages analysis history for audit trail.

    Stores records of all analyses performed, including:
    - Timestamp and event metadata
    - Hypotheses generated
    - Evidence found
    - Overall results summary
    """

    COLLECTION = "history"

    def __init__(self, storage: BaseStorage, retention: int = 100):
        """Initialize history memory.

        Args:
            storage: Storage backend to use
            retention: Maximum number of records to keep per repo
        """
        self.storage = storage
        self.retention = retention

    def record(
        self,
        repo: str,
        hypotheses: list[ValueHypothesis],
        evidences: list[EvidenceResult],
        event_type: str = "pr",
        pr_number: Optional[int] = None,
        commit_sha: Optional[str] = None,
    ) -> AnalysisRecord:
        """Record a new analysis.

        Args:
            repo: Repository identifier
            hypotheses: Hypotheses generated
            evidences: Evidence results
            event_type: Type of event that triggered analysis
            pr_number: PR number if applicable
            commit_sha: Commit SHA if applicable

        Returns:
            The created AnalysisRecord
        """
        # Create record
        confirmed_count = sum(1 for e in evidences if e.is_confirmed)
        value_ids = list(set(h.value_id for h in hypotheses if h.value_id))

        record = AnalysisRecord(
            repo=repo,
            timestamp=datetime.now(),
            event_type=event_type,
            pr_number=pr_number,
            commit_sha=commit_sha,
            hypothesis_count=len(hypotheses),
            confirmed_count=confirmed_count,
            value_ids=value_ids,
            hypotheses=hypotheses,
            evidences=evidences,
        )

        # Store record
        self._store_record(repo, record)

        return record

    def get_recent(
        self, repo: str, limit: int = 10
    ) -> list[AnalysisRecord]:
        """Get recent analysis records for a repository.

        Args:
            repo: Repository identifier
            limit: Maximum number of records to return

        Returns:
            List of AnalysisRecord, most recent first
        """
        key = f"{self.COLLECTION}/{self._sanitize_repo(repo)}"
        data = self.storage.get(key)

        if data is None:
            return []

        records = []
        for record_data in data.get("records", []):
            record = self._dict_to_record(record_data)
            records.append(record)

        # Sort by timestamp (most recent first) and limit
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:limit]

    def get_by_id(self, repo: str, record_id: str) -> Optional[AnalysisRecord]:
        """Get a specific analysis record by ID.

        Args:
            repo: Repository identifier
            record_id: Record ID

        Returns:
            AnalysisRecord or None if not found
        """
        records = self.get_recent(repo, limit=self.retention)
        for record in records:
            if record.id == record_id:
                return record
        return None

    def get_by_pr(
        self, repo: str, pr_number: int
    ) -> list[AnalysisRecord]:
        """Get all analysis records for a specific PR.

        Args:
            repo: Repository identifier
            pr_number: PR number

        Returns:
            List of AnalysisRecord for the PR
        """
        records = self.get_recent(repo, limit=self.retention)
        return [r for r in records if r.pr_number == pr_number]

    def get_statistics(self, repo: str) -> dict[str, Any]:
        """Get statistics about analyses for a repository.

        Args:
            repo: Repository identifier

        Returns:
            Statistics dictionary
        """
        records = self.get_recent(repo, limit=self.retention)

        if not records:
            return {
                "total_analyses": 0,
                "total_hypotheses": 0,
                "total_confirmed": 0,
                "confirmation_rate": 0.0,
                "value_distribution": {},
            }

        total_hypotheses = sum(r.hypothesis_count for r in records)
        total_confirmed = sum(r.confirmed_count for r in records)

        # Value distribution
        value_counts: dict[str, int] = {}
        for record in records:
            for vid in record.value_ids:
                value_counts[vid] = value_counts.get(vid, 0) + 1

        return {
            "total_analyses": len(records),
            "total_hypotheses": total_hypotheses,
            "total_confirmed": total_confirmed,
            "confirmation_rate": (
                total_confirmed / total_hypotheses if total_hypotheses > 0 else 0.0
            ),
            "value_distribution": dict(
                sorted(value_counts.items(), key=lambda x: -x[1])
            ),
            "first_analysis": min(r.timestamp for r in records).isoformat(),
            "last_analysis": max(r.timestamp for r in records).isoformat(),
        }

    def clear(self, repo: str) -> bool:
        """Clear all history for a repository.

        Args:
            repo: Repository identifier

        Returns:
            True if cleared, False if not found
        """
        key = f"{self.COLLECTION}/{self._sanitize_repo(repo)}"
        return self.storage.delete(key)

    def _store_record(self, repo: str, record: AnalysisRecord) -> None:
        """Store a record, maintaining retention limit."""
        key = f"{self.COLLECTION}/{self._sanitize_repo(repo)}"
        data = self.storage.get(key) or {"records": []}

        # Add new record
        record_dict = self._record_to_dict(record)
        data["records"].append(record_dict)

        # Enforce retention limit
        if len(data["records"]) > self.retention:
            # Sort by timestamp and keep most recent
            data["records"].sort(
                key=lambda r: r.get("timestamp", ""),
                reverse=True,
            )
            data["records"] = data["records"][: self.retention]

        self.storage.store(key, data)

    def _sanitize_repo(self, repo: str) -> str:
        """Sanitize repository name for use as key."""
        return repo.replace("/", "__")

    def _record_to_dict(self, record: AnalysisRecord) -> dict[str, Any]:
        """Convert AnalysisRecord to dictionary."""
        # Convert hypotheses and evidences to simple dicts
        hypotheses_data = []
        for h in record.hypotheses:
            h_dict = {
                "id": h.id,
                "value_id": h.value_id,
                "deviation_type": h.deviation_type,
                "confidence": h.confidence,
                "severity": h.severity,
                "description": h.description,
            }
            hypotheses_data.append(h_dict)

        evidences_data = []
        for e in record.evidences:
            e_dict = {
                "hypothesis_id": e.hypothesis_id,
                "status": e.status.value if hasattr(e.status, "value") else str(e.status),
                "evidence_count": len(e.evidence_pieces),
            }
            evidences_data.append(e_dict)

        return {
            "id": record.id,
            "repo": record.repo,
            "timestamp": record.timestamp.isoformat(),
            "event_type": record.event_type,
            "pr_number": record.pr_number,
            "commit_sha": record.commit_sha,
            "hypothesis_count": record.hypothesis_count,
            "confirmed_count": record.confirmed_count,
            "value_ids": record.value_ids,
            "hypotheses": hypotheses_data,
            "evidences": evidences_data,
        }

    def _dict_to_record(self, data: dict[str, Any]) -> AnalysisRecord:
        """Convert dictionary to AnalysisRecord."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return AnalysisRecord(
            id=data.get("id", ""),
            repo=data.get("repo", ""),
            timestamp=timestamp,
            event_type=data.get("event_type", "pr"),
            pr_number=data.get("pr_number"),
            commit_sha=data.get("commit_sha"),
            hypothesis_count=data.get("hypothesis_count", 0),
            confirmed_count=data.get("confirmed_count", 0),
            value_ids=data.get("value_ids", []),
            # Note: Full hypotheses/evidences not restored to save memory
            hypotheses=[],
            evidences=[],
        )
