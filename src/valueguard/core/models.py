"""Data models for ValueGuard."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import uuid


class TaskStatus(str, Enum):
    """Status of a task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvidenceStatus(str, Enum):
    """Status of evidence verification."""

    CONFIRMED = "confirmed"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


class PatternType(str, Enum):
    """Type of learned value pattern."""

    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    COMMON_RISK = "common_risk"


# --- Diff and Code Models ---


@dataclass
class DiffHunk:
    """A chunk of changed code from a git diff."""

    file_path: str
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    content: str
    change_type: str = "modified"  # "added", "deleted", "modified"
    language: Optional[str] = None

    def __post_init__(self):
        if self.language is None:
            self.language = self._detect_language()

    def _detect_language(self) -> str:
        """Detect language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
        }
        for ext, lang in ext_map.items():
            if self.file_path.endswith(ext):
                return lang
        return "unknown"


# --- Value Profile Models ---


@dataclass
class ProfileEvidence:
    """Evidence supporting a value in the profile."""

    source: str  # "readme", "issue", "config", "code"
    content: str
    value_id: str
    confidence: float


@dataclass
class ValueProfile:
    """Project value profile with L2/L3 scores."""

    repo: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1

    # L2/L3 scores (0-1 normalized)
    l2_scores: dict[str, float] = field(default_factory=dict)
    l3_scores: dict[str, float] = field(default_factory=dict)

    # Core values (top-N by score)
    core_values: list[str] = field(default_factory=list)

    # Evidence for profile (for explainability)
    evidence_samples: list[ProfileEvidence] = field(default_factory=list)

    # Learning metadata
    analysis_count: int = 0
    confidence: float = 0.0

    def get_top_values(self, n: int = 3) -> list[str]:
        """Get top N values by combined L2+L3 scores."""
        combined = {**self.l2_scores, **self.l3_scores}
        sorted_values = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [v[0] for v in sorted_values[:n]]


# --- Hypothesis Models ---


@dataclass
class CrossLayerTrace:
    """Trace of value reasoning across layers."""

    l1_value: str  # Schwartz value
    l2_theme: str  # Human value theme
    l3_attribute: str  # System value theme
    l4_indicator: str  # Code artifact indicator
    reasoning: str  # Explanation of the trace


@dataclass
class ValueHypothesis:
    """A hypothesis about value deviation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    value_id: str = ""  # e.g., "HV9_Privacy"
    deviation_type: str = ""  # "violation", "inconsistency", "risk"
    confidence: float = 0.0
    severity: str = "medium"  # "low", "medium", "high", "critical"

    # Cross-layer reasoning trace
    cross_layer_trace: Optional[CrossLayerTrace] = None

    # Context
    diff_hunk: Optional[DiffHunk] = None
    description: str = ""
    suggested_action: str = ""

    # Ranking metadata
    profile_weight: float = 0.0  # Weight from project profile


# --- Evidence Models ---


@dataclass
class ASTTrace:
    """AST-based call chain trace."""

    function_name: str
    file_path: str
    line_number: int
    call_chain: list[str] = field(default_factory=list)


@dataclass
class EvidencePiece:
    """A piece of code evidence."""

    file_path: str
    start_line: int
    end_line: int
    snippet: str
    relevance_score: float = 0.0
    ast_trace: Optional[ASTTrace] = None

    @property
    def location(self) -> str:
        """Format location as file:line."""
        return f"{self.file_path}:{self.start_line}"


@dataclass
class EvidenceResult:
    """Result of evidence search for a hypothesis."""

    hypothesis_id: str
    status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    evidence_pieces: list[EvidencePiece] = field(default_factory=list)
    search_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_confirmed(self) -> bool:
        return self.status == EvidenceStatus.CONFIRMED


# --- Task Models ---


@dataclass
class ProfileTask:
    """Task for building/updating project profile."""

    repo: str
    rebuild: bool = False
    sources: list[str] = field(
        default_factory=lambda: ["readme", "issues", "config"]
    )


@dataclass
class HypothesisTask:
    """Task for generating value hypotheses."""

    diff_hunks: list[DiffHunk]
    profile: ValueProfile
    memory_context: list["ValuePattern"] = field(default_factory=list)
    max_hypotheses: int = 10


@dataclass
class EvidenceTask:
    """Task for locating evidence for a hypothesis."""

    hypothesis: ValueHypothesis
    repo_path: str
    search_depth: int = 3


# --- Analysis Event ---


@dataclass
class AnalysisEvent:
    """Event triggering a value analysis."""

    repo: str
    repo_path: str
    diff_hunks: list[DiffHunk]
    event_type: str = "pr"  # "pr", "push", "manual"
    pr_number: Optional[int] = None
    commit_sha: Optional[str] = None
    force_rebuild: bool = False


# --- Memory Models ---


@dataclass
class ValuePattern:
    """Learned pattern for value detection."""

    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    pattern_type: PatternType = PatternType.COMMON_RISK
    value_id: str = ""
    code_pattern: str = ""  # e.g., "unencrypted_fallback"
    description: str = ""
    confidence: float = 0.5
    seen_count: int = 1
    repos: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class AnalysisRecord:
    """Record of a single analysis run."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    repo: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = "pr"
    pr_number: Optional[int] = None
    commit_sha: Optional[str] = None

    # Results summary
    hypothesis_count: int = 0
    confirmed_count: int = 0
    value_ids: list[str] = field(default_factory=list)

    # Full results (optional, for detailed audit)
    hypotheses: list[ValueHypothesis] = field(default_factory=list)
    evidences: list[EvidenceResult] = field(default_factory=list)


# --- Report Models ---


@dataclass
class ValueScore:
    """Score for a single value in the report."""

    value_id: str
    value_name: str
    score: float  # 0-1, higher = more deviation
    severity: str
    hypothesis_count: int
    evidence_count: int


@dataclass
class ValueGuardReport:
    """Final report from ValueGuard analysis."""

    repo: str
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = "pr"
    pr_number: Optional[int] = None
    commit_sha: Optional[str] = None

    # Overall scores
    overall_risk_score: float = 0.0
    value_scores: list[ValueScore] = field(default_factory=list)

    # Detailed results
    hypotheses: list[ValueHypothesis] = field(default_factory=list)
    evidences: list[EvidenceResult] = field(default_factory=list)
    profile: Optional[ValueProfile] = None

    # Reviewer mentions
    mentions: list[str] = field(default_factory=list)

    def get_top_risks(self, n: int = 5) -> list[ValueHypothesis]:
        """Get top N hypotheses by confidence."""
        confirmed_ids = {e.hypothesis_id for e in self.evidences if e.is_confirmed}
        confirmed = [h for h in self.hypotheses if h.id in confirmed_ids]
        return sorted(confirmed, key=lambda h: h.confidence, reverse=True)[:n]
