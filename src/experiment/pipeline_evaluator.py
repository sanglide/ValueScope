#!/usr/bin/env python
"""
Pipeline Evaluator — Run Hypothesis Generator + Evidence Location Agent and compute evaluation metrics
Compare Pipeline output with 0-shot LLM direct evaluation results
"""

import sys
import json
import time
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

# Add project path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"

# Set PYTHONPATH
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Set environment variables
os.environ['PYTHONPATH'] = str(src_path)

from valueguard.core.models import (
    DiffHunk,
    HypothesisTask,
    EvidenceTask,
    ValueHypothesis,
    EvidenceResult,
    EvidenceStatus,
    ValueProfile,
)
from valueguard.agents.hypothesis_agent import HypothesisAgent
from valueguard.agents.evidence_agent import EvidenceAgent
from valueguard.skills.registry import SkillRegistry
from valueguard.skills.llm_call import LLMCallSkill
from valueguard.memory.manager import MemoryManager

try:
    from .commit_diff_extractor import CommitSample, load_samples_from_json
    from .iaa_metrics import (
        percent_agreement,
        cohen_kappa_binary,
        pabak,
        gwet_ac1,
        pairwise_jaccard,
        pairwise_symmetric_f1,
    )
except ImportError:
    from commit_diff_extractor import CommitSample, load_samples_from_json
    from iaa_metrics import (
        percent_agreement,
        cohen_kappa_binary,
        pabak,
        gwet_ac1,
        pairwise_jaccard,
        pairwise_symmetric_f1,
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Evaluation Result Data Structures
# ============================================================

@dataclass
class HypothesisStats:
    """Hypothesis generation statistics"""
    total_generated: int = 0
    avg_confidence: float = 0.0
    confidence_distribution: dict = field(default_factory=dict)  # Statistics by range
    value_distribution: dict = field(default_factory=dict)  # Statistics by value_id
    severity_distribution: dict = field(default_factory=dict)  # Statistics by severity
    deviation_type_distribution: dict = field(default_factory=dict)


@dataclass
class EvidenceStats:
    """Evidence location statistics"""
    total_hypotheses: int = 0
    confirmed_count: int = 0
    unverified_count: int = 0
    rejected_count: int = 0
    avg_relevance_score: float = 0.0
    confirmation_rate: float = 0.0


@dataclass
class SampleResult:
    """Evaluation result for a single sample"""
    sample_id: str
    commit_sha: str
    commit_message: str

    # Pipeline output
    hypotheses: list[ValueHypothesis] = field(default_factory=list)
    evidence_results: list[EvidenceResult] = field(default_factory=list)

    # Filtered results (CONFIRMED only)
    confirmed_hypotheses: list[ValueHypothesis] = field(default_factory=list)

    # Ground truth
    ground_truth_has_risk: Optional[bool] = None
    ground_truth_values: list[str] = field(default_factory=list)

    # Prediction results
    @property
    def predicted_has_risk(self) -> bool:
        return len(self.confirmed_hypotheses) > 0

    @property
    def predicted_has_risk_before_filter(self) -> bool:
        return len(self.hypotheses) > 0

    @property
    def predicted_values(self) -> list[str]:
        return list(set(h.value_id for h in self.confirmed_hypotheses))

    @property
    def predicted_values_before_filter(self) -> list[str]:
        return list(set(h.value_id for h in self.hypotheses))

    # Timing
    hypothesis_time_ms: float = 0.0
    evidence_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "commit_sha": self.commit_sha,
            "commit_message": self.commit_message,
            "hypotheses_count": len(self.hypotheses),
            "confirmed_count": len(self.confirmed_hypotheses),
            "predicted_has_risk": self.predicted_has_risk,
            "predicted_has_risk_before_filter": self.predicted_has_risk_before_filter,
            "predicted_values": self.predicted_values,
            "predicted_values_before_filter": self.predicted_values_before_filter,
            "ground_truth_has_risk": self.ground_truth_has_risk,
            "ground_truth_values": self.ground_truth_values,
            "hypothesis_time_ms": self.hypothesis_time_ms,
            "evidence_time_ms": self.evidence_time_ms,
        }


@dataclass
class PipelineEvaluationResults:
    """Pipeline overall evaluation results"""
    total_samples: int = 0
    total_time_seconds: float = 0.0

    # Hypothesis statistics
    hypothesis_stats: HypothesisStats = field(default_factory=HypothesisStats)

    # Evidence statistics
    evidence_stats: EvidenceStats = field(default_factory=EvidenceStats)

    # IAA agreement metrics (Human vs Pipeline, before filtering)
    # Dimension 1: Risk detection (binary classification)
    before_filter_percent_agreement: float = 0.0
    before_filter_cohen_kappa: float = 0.0
    before_filter_pabak: float = 0.0
    before_filter_gwet_ac1: float = 0.0

    # Dimension 2: Value identification (multi-label)
    before_filter_jaccard: float = 0.0
    before_filter_symmetric_f1: float = 0.0

    # IAA agreement metrics (Human vs Pipeline, after filtering)
    # Dimension 1: Risk detection (binary classification)
    after_filter_percent_agreement: float = 0.0
    after_filter_cohen_kappa: float = 0.0
    after_filter_pabak: float = 0.0
    after_filter_gwet_ac1: float = 0.0

    # Dimension 2: Value identification (multi-label)
    after_filter_jaccard: float = 0.0
    after_filter_symmetric_f1: float = 0.0

    # Detailed results
    sample_results: list[SampleResult] = field(default_factory=list)


# ============================================================
# Cache Helper Functions
# ============================================================

def _stub_hypothesis(value_id: str) -> ValueHypothesis:
    """Create a minimal ValueHypothesis stub used when restoring from cache."""
    return ValueHypothesis(
        id="cached",
        value_id=value_id,
        deviation_type="risk",
        confidence=1.0,
        severity="medium",
        description="(restored from cache)",
        suggested_action="",
    )


# ============================================================
# Pipeline Evaluator
# ============================================================

class PipelineEvaluator:
    """Hypothesis Generator + Evidence Location Agent Pipeline Evaluator"""

    def __init__(
        self,
        repo_path: str,
        llm_provider: str = "deepseek",
        storage_path: str = ".valueguard/memory",
        mock_mode: bool = False,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ):
        self.repo_path = Path(repo_path)
        self.llm_provider = llm_provider
        self.mock_mode = mock_mode

        # ---------- Cache Configuration ----------
        # Cache directory: defaults to experiment_logs/llm_outputs/pipeline_cache/<provider>/
        if use_cache:
            _base = Path(cache_dir) if cache_dir else (
                Path(__file__).parent / "experiment_logs" / "llm_outputs" / "pipeline_cache"
            )
            self._cache_dir = _base / llm_provider
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache dir: {self._cache_dir}")
        else:
            self._cache_dir = None
        # --------------------------------

        if not mock_mode:
            # Initialize skills
            self.skills = SkillRegistry()
            self._setup_skills()

            # Initialize memory
            self.memory = MemoryManager(storage_path=storage_path)

            # Initialize agents
            self.hypothesis_agent = HypothesisAgent(
                self.skills, self.memory,
                config={"llm_provider": llm_provider}
            )
            self.evidence_agent = EvidenceAgent(
                self.skills, self.memory,
                config={"search_depth": 3, "top_k": 10, "llm_provider": llm_provider}
            )
        else:
            logger.info("Running in MOCK mode - using simulated hypothesis generation")
            self.skills = None
            self.memory = None
            self.hypothesis_agent = None
            self.evidence_agent = None

        # Create empty profile (can be refined later)
        self.default_profile = ValueProfile(
            repo=str(self.repo_path),
            l2_scores={f"HV{i}": 0.5 for i in range(1, 11)},
            l3_scores={f"SV{i}": 0.5 for i in range(1, 11)},
        )

    def _cache_path(self, commit_sha: str) -> Optional[Path]:
        """Return cache file path for the given commit, or None if caching is disabled."""
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{commit_sha}.json"

    def _load_from_cache(self, commit_sha: str) -> Optional["SampleResult"]:
        """Load SampleResult from cache, returns None if cache does not exist."""
        path = self._cache_path(commit_sha)
        if path is None or not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            confirmed_count = data.get("confirmed_count", 0)
            hypotheses_count = data.get("hypotheses_count", 0)

            # Rebuild evidence_results: used by _compute_statistics for confirmation_rate
            # confirmed_count CONFIRMED entries, rest as UNVERIFIED
            evidence_results = []
            for i in range(hypotheses_count):
                status = EvidenceStatus.CONFIRMED if i < confirmed_count else EvidenceStatus.UNVERIFIED
                evidence_results.append(EvidenceResult(
                    hypothesis_id=f"cached_{i}",
                    status=status,
                ))

            return SampleResult(
                sample_id=data["sample_id"],
                commit_sha=data["commit_sha"],
                commit_message=data["commit_message"],
                ground_truth_has_risk=data["ground_truth_has_risk"],
                ground_truth_values=data["ground_truth_values"],
                hypothesis_time_ms=data.get("hypothesis_time_ms", 0.0),
                evidence_time_ms=data.get("evidence_time_ms", 0.0),
                # Rebuild predicted info using placeholder ValueHypothesis list
                hypotheses=[
                    _stub_hypothesis(v) for v in data.get("predicted_values_before_filter", [])
                ],
                confirmed_hypotheses=[
                    _stub_hypothesis(v) for v in data.get("predicted_values", [])
                ],
                evidence_results=evidence_results,
            )
        except Exception as e:
            logger.warning(f"Cache read failed {commit_sha}: {e}")
            return None

    def _save_to_cache(self, result: "SampleResult") -> None:
        """Write SampleResult to cache file, only caches when sample was successfully processed."""
        path = self._cache_path(result.commit_sha)
        if path is None:
            return

        # ---- Success validation logic ----
        # If hypothesis generation or evidence verification completely failed (no hypotheses),
        # indicates a call error; do not cache so the next retry will re-call the API
        has_successful_processing = (
            len(result.hypotheses) > 0 or  # Generated hypotheses (even if all filtered out later)
            len(result.confirmed_hypotheses) > 0 or  # Or has confirmed hypotheses
            result.hypothesis_time_ms > 100  # Or hypothesis generation took >100ms (indicates real LLM call)
        )

        # Additional check: if hypothesis count is 0 and ground_truth indicates risk,
        # the LLM call may have failed; do not cache, allow retry
        if (
            len(result.hypotheses) == 0 and 
            result.ground_truth_has_risk and 
            result.hypothesis_time_ms < 50  # Very short time suggests no real call was made
        ):
            logger.debug(f"Skipping cache {result.commit_sha} - risk sample with no output, possible call failure")
            return

        if not has_successful_processing:
            logger.debug(f"Skipping cache {result.commit_sha} - sample processing failed or no output")
            return
        # ----------------------

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Cache write failed {result.commit_sha}: {e}")

    def _setup_skills(self):
        """Set up required skills"""
        # Register LLM call skill
        llm_skill = LLMCallSkill()
        self.skills.register(llm_skill)
        logger.info(f"✓ Registered LLMCallSkill for {self.llm_provider}")

        # Optional: register other skills (vector_search, ast_analysis, etc.)
        # These need to be configured for full deployment

    def _convert_to_diff_hunks(self, sample: CommitSample) -> list[DiffHunk]:
        """Convert extracted diff hunks to ValueGuard format"""
        hunks = []
        for extracted in sample.diff_hunks:
            hunks.append(DiffHunk(
                file_path=extracted.file_path,
                old_start=extracted.old_start,
                old_lines=extracted.old_lines,
                new_start=extracted.new_start,
                new_lines=extracted.new_lines,
                content=extracted.content,
                change_type=extracted.change_type,
            ))
        return hunks

    def _mock_generate_hypotheses(self, sample: CommitSample, diff_hunks: list[DiffHunk]) -> list[ValueHypothesis]:
        """Mock mode: generate simulated hypotheses based on keywords
        
        Simulates LLM behavioral characteristics:
        1. Tends to generate hypotheses for risky samples (high recall)
        2. May also produce false positives for non-risky samples (lower precision)
        """
        import random
        import uuid
        
        hypotheses = []
        msg_lower = sample.commit.message.lower()
        
        # Generate hypotheses based on keywords in commit message (aligned with ground truth)
        value_mappings = {
            "security": ("HV10", "Security violation risk"),
            "privacy": ("HV9", "Privacy concern detected"),
            "fix": ("SV2", "Bug fix - reliability improvement"),
            "bug": ("SV2", "Bug related change"),
            "crash": ("SV5", "Crash handling code"),
            "permission": ("HV6", "Permission handling change"),
            "encrypt": ("HV10", "Encryption related change"),
            "auth": ("SV1", "Authentication modification"),
            "error": ("SV2", "Error handling change"),
            "exception": ("SV5", "Exception handling"),
            "accessibility": ("SV9", "Accessibility improvement"),
            "a11y": ("SV9", "Accessibility feature"),
        }
        
        # If ground truth has risk, likely generate corresponding hypotheses (simulate high recall)
        if sample.ground_truth_has_risk:
            for keyword, (value_id, desc) in value_mappings.items():
                if keyword in msg_lower:
                    hypotheses.append(ValueHypothesis(
                        id=str(uuid.uuid4())[:8],
                        value_id=value_id,
                        deviation_type="risk",
                        confidence=random.uniform(0.7, 0.95),
                        severity=random.choice(["medium", "high"]),
                        description=f"{desc} in {sample.commit.short_sha}",
                        suggested_action="Review code change",
                        diff_hunk=diff_hunks[0] if diff_hunks else None,
                    ))
        
        # For all samples, LLM has a certain probability of producing false positives (simulate over-sensitivity)
        # This matches the tendency observed in 0-shot evaluation where LLMs tend to flag risk
        if random.random() < 0.4:  # 40% probability of generating extra hypotheses
            num_extra = random.randint(1, 2)
            extra_values = ["HV3", "HV5", "SV3", "SV6", "SV8", "HV7", "SV4"]
            for _ in range(num_extra):
                value_id = random.choice(extra_values)
                hypotheses.append(ValueHypothesis(
                    id=str(uuid.uuid4())[:8],
                    value_id=value_id,
                    deviation_type="risk",
                    confidence=random.uniform(0.5, 0.75),
                    severity="low",
                    description=f"Potential value concern in code change",
                    suggested_action="Manual review recommended",
                    diff_hunk=diff_hunks[0] if diff_hunks else None,
                ))
        
        return hypotheses

    def _mock_verify_hypothesis(self, hypothesis: ValueHypothesis) -> EvidenceResult:
        """Mock mode: simulate evidence verification"""
        import random
        from valueguard.core.models import EvidencePiece
        
        # Higher confidence hypotheses are more likely to be confirmed
        confirm_prob = hypothesis.confidence * 0.8
        
        if random.random() < confirm_prob:
            status = EvidenceStatus.CONFIRMED
        elif random.random() < 0.3:
            status = EvidenceStatus.REJECTED
        else:
            status = EvidenceStatus.UNVERIFIED
        
        return EvidenceResult(
            hypothesis_id=hypothesis.id,
            status=status,
            evidence_pieces=[
                EvidencePiece(
                    file_path="mock/path.java",
                    start_line=1,
                    end_line=10,
                    snippet="Simulated evidence for testing",
                    relevance_score=random.uniform(0.5, 0.9),
                )
            ] if status == EvidenceStatus.CONFIRMED else [],
            search_metadata={"mock": True, "reasoning": "Mock verification result"},
        )

    def evaluate_sample(self, sample: CommitSample) -> SampleResult:
        """Evaluate a single sample"""
        # ---- Cache hit check ----
        cached = self._load_from_cache(sample.commit.sha)
        if cached is not None:
            logger.info(f"  [CACHE HIT] {sample.commit.short_sha}")
            return cached
        # ----------------------

        result = SampleResult(
            sample_id=sample.commit.short_sha,
            commit_sha=sample.commit.sha,
            commit_message=sample.commit.message,
            ground_truth_has_risk=sample.ground_truth_has_risk,
            ground_truth_values=sample.ground_truth_values,
        )

        # Convert diff hunks
        diff_hunks = self._convert_to_diff_hunks(sample)

        if not diff_hunks:
            return result

        # Stage 1: Generate hypotheses
        start_time = time.time()
        if self.mock_mode:
            hypotheses = self._mock_generate_hypotheses(sample, diff_hunks)
            result.hypotheses = hypotheses
        else:
            try:
                logger.info(f"🔧 [DEBUG] Skills registered: {self.skills.list_skills()}")
                logger.info(f"🔧 [DEBUG] HypothesisAgent has llm_call skill: {self.hypothesis_agent.has_skill('llm_call')}")
                
                task = HypothesisTask(
                    diff_hunks=diff_hunks,
                    profile=self.default_profile,
                    memory_context=[],
                    max_hypotheses=10,
                )
                hypotheses = self.hypothesis_agent.execute(task)
                result.hypotheses = hypotheses
            except Exception as e:
                logger.error(f"Hypothesis generation failed {sample.commit.short_sha}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                hypotheses = []
        result.hypothesis_time_ms = (time.time() - start_time) * 1000

        # Stage 2: Evidence location
        start_time = time.time()
        for hypothesis in hypotheses:
            if self.mock_mode:
                evidence_result = self._mock_verify_hypothesis(hypothesis)
                result.evidence_results.append(evidence_result)
                if evidence_result.status == EvidenceStatus.CONFIRMED:
                    result.confirmed_hypotheses.append(hypothesis)
            else:
                try:
                    evidence_task = EvidenceTask(
                        hypothesis=hypothesis,
                        repo_path=str(self.repo_path),
                        search_depth=3,
                    )
                    evidence_result = self.evidence_agent.execute(evidence_task)
                    result.evidence_results.append(evidence_result)

                    # Filter CONFIRMED hypotheses
                    if evidence_result.status == EvidenceStatus.CONFIRMED:
                        result.confirmed_hypotheses.append(hypothesis)
                except Exception as e:
                    logger.warning(f"Evidence location failed {hypothesis.id}: {e}")
        result.evidence_time_ms = (time.time() - start_time) * 1000

        # ---- Write to cache ----
        self._save_to_cache(result)
        # ------------------

        return result

    def evaluate_all(
        self,
        samples: list[CommitSample],
        max_samples: Optional[int] = None,
        parallel_workers: int = 1,
    ) -> PipelineEvaluationResults:
        """Evaluate all samples.

        Args:
            samples: List of samples to evaluate.
            max_samples: Maximum number of samples to evaluate (None = all).
            parallel_workers: Number of parallel threads. 1 = sequential (default), >1 = parallel LLM API calls.
                Recommended to set to the API-allowed concurrency, typically 4~8.
        """
        import concurrent.futures
        import threading

        if max_samples:
            samples = samples[:max_samples]

        results = PipelineEvaluationResults(total_samples=len(samples))
        start_time = time.time()

        if parallel_workers <= 1:
            # Sequential execution (original behavior)
            for i, sample in enumerate(samples):
                logger.info(f"Evaluating sample {i+1}/{len(samples)}: {sample.commit.short_sha}")
                sample_result = self.evaluate_sample(sample)
                results.sample_results.append(sample_result)
        else:
            # Parallel execution
            logger.info(f"Parallel mode: {parallel_workers} worker threads, {len(samples)} samples total")
            completed = [0]
            lock = threading.Lock()

            def _run(sample: CommitSample) -> SampleResult:
                result = self.evaluate_sample(sample)
                with lock:
                    completed[0] += 1
                    logger.info(
                        f"[{completed[0]}/{len(samples)}] Completed: {sample.commit.short_sha}"
                    )
                return result

            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                future_map = {executor.submit(_run, s): s for s in samples}
                sample_order = {s.commit.short_sha: i for i, s in enumerate(samples)}
                finished: list[SampleResult] = []
                for future in concurrent.futures.as_completed(future_map):
                    try:
                        finished.append(future.result())
                    except Exception as e:
                        sample = future_map[future]
                        logger.error(f"Sample {sample.commit.short_sha} failed: {e}")
                        # Insert empty result to ensure consistent sample count
                        finished.append(SampleResult(
                            sample_id=sample.commit.short_sha,
                            commit_sha=sample.commit.sha,
                            commit_message=sample.commit.message,
                            ground_truth_has_risk=sample.ground_truth_has_risk,
                            ground_truth_values=sample.ground_truth_values,
                        ))
                # Sort by original sample order
                finished.sort(key=lambda r: sample_order.get(r.sample_id, 0))
                results.sample_results = finished

        results.total_time_seconds = time.time() - start_time

        # Compute statistics
        self._compute_statistics(results)

        return results

    def _compute_statistics(self, results: PipelineEvaluationResults):
        """Compute various statistics"""
        # Hypothesis statistics
        all_hypotheses = []
        for sr in results.sample_results:
            all_hypotheses.extend(sr.hypotheses)

        h_stats = results.hypothesis_stats
        h_stats.total_generated = len(all_hypotheses)

        if all_hypotheses:
            confidences = [h.confidence for h in all_hypotheses]
            h_stats.avg_confidence = sum(confidences) / len(confidences)

            # Confidence distribution
            h_stats.confidence_distribution = {
                "0.5-0.6": sum(1 for c in confidences if 0.5 <= c < 0.6),
                "0.6-0.7": sum(1 for c in confidences if 0.6 <= c < 0.7),
                "0.7-0.8": sum(1 for c in confidences if 0.7 <= c < 0.8),
                "0.8-0.9": sum(1 for c in confidences if 0.8 <= c < 0.9),
                "0.9-1.0": sum(1 for c in confidences if 0.9 <= c <= 1.0),
            }

            # Value distribution
            for h in all_hypotheses:
                h_stats.value_distribution[h.value_id] = h_stats.value_distribution.get(h.value_id, 0) + 1

            # Severity distribution
            for h in all_hypotheses:
                h_stats.severity_distribution[h.severity] = h_stats.severity_distribution.get(h.severity, 0) + 1

        # Evidence statistics
        all_evidence = []
        for sr in results.sample_results:
            all_evidence.extend(sr.evidence_results)

        e_stats = results.evidence_stats
        e_stats.total_hypotheses = len(all_evidence)
        e_stats.confirmed_count = sum(1 for e in all_evidence if e.status == EvidenceStatus.CONFIRMED)
        e_stats.unverified_count = sum(1 for e in all_evidence if e.status == EvidenceStatus.UNVERIFIED)
        e_stats.rejected_count = sum(1 for e in all_evidence if e.status == EvidenceStatus.REJECTED)

        if e_stats.total_hypotheses > 0:
            e_stats.confirmation_rate = e_stats.confirmed_count / e_stats.total_hypotheses

        # Collect relevance scores from all evidence
        all_scores = []
        for e in all_evidence:
            for piece in e.evidence_pieces:
                all_scores.append(piece.relevance_score)
        if all_scores:
            e_stats.avg_relevance_score = sum(all_scores) / len(all_scores)

        # Compute comparison metrics
        self._compute_comparison_metrics(results)

    def _compute_comparison_metrics(self, results: PipelineEvaluationResults):
        """Compute IAA agreement metrics before and after filtering (Human vs Pipeline)"""
        # Collect ground truth (Human) and predictions (Pipeline)
        human_risks = []
        pipeline_risks_before = []
        pipeline_risks_after = []
        human_values_list = []
        pipeline_values_before_list = []
        pipeline_values_after_list = []

        for sr in results.sample_results:
            if sr.ground_truth_has_risk is not None:
                human_risks.append(sr.ground_truth_has_risk)
                pipeline_risks_before.append(sr.predicted_has_risk_before_filter)
                pipeline_risks_after.append(sr.predicted_has_risk)
                human_values_list.append(set(sr.ground_truth_values))
                pipeline_values_before_list.append(set(sr.predicted_values_before_filter))
                pipeline_values_after_list.append(set(sr.predicted_values))

        if not human_risks:
            return

        # ============================================================
        # Dimension 1: Risk detection agreement (binary classification)
        # ============================================================
        
        # Before filtering
        results.before_filter_percent_agreement = percent_agreement(human_risks, pipeline_risks_before)
        results.before_filter_cohen_kappa = cohen_kappa_binary(human_risks, pipeline_risks_before)
        results.before_filter_pabak = pabak(human_risks, pipeline_risks_before)
        results.before_filter_gwet_ac1 = gwet_ac1(human_risks, pipeline_risks_before)

        # After filtering
        results.after_filter_percent_agreement = percent_agreement(human_risks, pipeline_risks_after)
        results.after_filter_cohen_kappa = cohen_kappa_binary(human_risks, pipeline_risks_after)
        results.after_filter_pabak = pabak(human_risks, pipeline_risks_after)
        results.after_filter_gwet_ac1 = gwet_ac1(human_risks, pipeline_risks_after)

        # ============================================================
        # Dimension 2: Value identification agreement (multi-label)
        # ============================================================
        
        # Before filtering
        results.before_filter_jaccard = pairwise_jaccard(human_values_list, pipeline_values_before_list)
        results.before_filter_symmetric_f1 = pairwise_symmetric_f1(human_values_list, pipeline_values_before_list)

        # After filtering
        results.after_filter_jaccard = pairwise_jaccard(human_values_list, pipeline_values_after_list)
        results.after_filter_symmetric_f1 = pairwise_symmetric_f1(human_values_list, pipeline_values_after_list)


# ============================================================
# Report Generation
# ============================================================

# Value ID to Name mapping
VALUE_NAMES = {
    "HV1": "Conformity", "HV2": "Pleasure", "HV3": "Dignity", "HV4": "Inclusiveness",
    "HV5": "Sense of Belonging", "HV6": "Freedom", "HV7": "Independence", "HV8": "Wealth",
    "HV9": "Privacy", "HV10": "Security",
    "SV1": "Trust", "SV2": "Correctness", "SV3": "Compatibility", "SV4": "Portability",
    "SV5": "Reliability", "SV6": "Efficiency", "SV7": "Energy Preservation", "SV8": "Usability",
    "SV9": "Accessibility", "SV10": "Longevity",
}


def _interpret_kappa(kappa: float) -> str:
    """Interpret Cohen's Kappa value."""
    if kappa < 0:
        return "Poor"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


def _format_change(before: float, after: float) -> str:
    """Format the change between before and after values."""
    diff = after - before
    if abs(diff) < 0.0001:
        return "—"
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.4f}"


def _format_percent(value: float) -> str:
    """Format value as percentage."""
    return f"{value * 100:.1f}%"


def generate_report(results: PipelineEvaluationResults) -> str:
    """Generate comprehensive evaluation report in English (Markdown format)."""
    lines = [
        "# Pipeline Evaluation Report",
        "",
        "## Experiment Overview",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Total Samples | {results.total_samples} |",
        f"| Total Time | {results.total_time_seconds:.2f}s |",
        f"| Hypotheses Generated | {results.hypothesis_stats.total_generated} |",
        f"| Avg Confidence | {results.hypothesis_stats.avg_confidence:.4f} |",
        f"| Evidence Confirmation Rate | {_format_percent(results.evidence_stats.confirmation_rate)} |",
        "",
        "---",
        "",
        "## Hypothesis Generation Statistics",
        "",
        "### Confidence Distribution",
        "",
        "| Range | Count | Percentage |",
        "|---|---|---|",
    ]

    total_hyp = results.hypothesis_stats.total_generated or 1
    for bucket, count in results.hypothesis_stats.confidence_distribution.items():
        pct = count / total_hyp * 100
        lines.append(f"| {bucket} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "### Value Distribution",
        "",
        "| Value ID | Name | Count | Percentage |",
        "|---|---|---|---|",
    ])
    
    for value_id, count in sorted(results.hypothesis_stats.value_distribution.items(), key=lambda x: -x[1]):
        name = VALUE_NAMES.get(value_id, value_id)
        pct = count / total_hyp * 100
        lines.append(f"| {value_id} | {name} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "### Severity Distribution",
        "",
        "| Severity | Count | Percentage |",
        "|---|---|---|",
    ])
    
    for severity, count in sorted(results.hypothesis_stats.severity_distribution.items(), key=lambda x: -x[1]):
        pct = count / total_hyp * 100
        lines.append(f"| {severity} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## Evidence Verification Statistics",
        "",
        "| Status | Count | Percentage |",
        "|---|---|---|",
        f"| CONFIRMED | {results.evidence_stats.confirmed_count} | {_format_percent(results.evidence_stats.confirmed_count / max(1, results.evidence_stats.total_hypotheses))} |",
        f"| UNVERIFIED | {results.evidence_stats.unverified_count} | {_format_percent(results.evidence_stats.unverified_count / max(1, results.evidence_stats.total_hypotheses))} |",
        f"| REJECTED | {results.evidence_stats.rejected_count} | {_format_percent(results.evidence_stats.rejected_count / max(1, results.evidence_stats.total_hypotheses))} |",
        "",
        f"**Average Relevance Score:** {results.evidence_stats.avg_relevance_score:.4f}",
        "",
        "---",
        "",
        "## Human vs Pipeline Agreement (IAA Metrics)",
        "",
        "### Dimension 1: Risk Detection (Binary Classification)",
        "",
        "| Metric | Before Filter | After Filter | Change | Interpretation |",
        "|---|---|---|---|---|",
        f"| % Agreement | {_format_percent(results.before_filter_percent_agreement)} | {_format_percent(results.after_filter_percent_agreement)} | {_format_change(results.before_filter_percent_agreement, results.after_filter_percent_agreement)} | — |",
        f"| Cohen's κ | {results.before_filter_cohen_kappa:.4f} | {results.after_filter_cohen_kappa:.4f} | {_format_change(results.before_filter_cohen_kappa, results.after_filter_cohen_kappa)} | {_interpret_kappa(results.after_filter_cohen_kappa)} |",
        f"| PABAK | {results.before_filter_pabak:.4f} | {results.after_filter_pabak:.4f} | {_format_change(results.before_filter_pabak, results.after_filter_pabak)} | {_interpret_kappa(results.after_filter_pabak)} |",
        f"| Gwet's AC1 | {results.before_filter_gwet_ac1:.4f} | {results.after_filter_gwet_ac1:.4f} | {_format_change(results.before_filter_gwet_ac1, results.after_filter_gwet_ac1)} | {_interpret_kappa(results.after_filter_gwet_ac1)} |",
        "",
        "### Dimension 2: Value Identification (Multi-label)",
        "",
        "| Metric | Before Filter | After Filter | Change |",
        "|---|---|---|---|",
        f"| Pairwise Jaccard | {results.before_filter_jaccard:.4f} | {results.after_filter_jaccard:.4f} | {_format_change(results.before_filter_jaccard, results.after_filter_jaccard)} |",
        f"| Pairwise Symmetric F1 | {results.before_filter_symmetric_f1:.4f} | {results.after_filter_symmetric_f1:.4f} | {_format_change(results.before_filter_symmetric_f1, results.after_filter_symmetric_f1)} |",
        "",
        "---",
        "",
        "## Summary Statistics",
        "",
        "| Dimension | Metric | Before Filter | After Filter |",
        "|---|---|---|---|",
        f"| Risk Detection | Cohen's κ | {results.before_filter_cohen_kappa:.4f} | {results.after_filter_cohen_kappa:.4f} |",
        f"| Risk Detection | % Agreement | {_format_percent(results.before_filter_percent_agreement)} | {_format_percent(results.after_filter_percent_agreement)} |",
        f"| Risk Detection | PABAK | {results.before_filter_pabak:.4f} | {results.after_filter_pabak:.4f} |",
        f"| Risk Detection | Gwet's AC1 | {results.before_filter_gwet_ac1:.4f} | {results.after_filter_gwet_ac1:.4f} |",
        f"| Value ID | Pairwise Jaccard | {results.before_filter_jaccard:.4f} | {results.after_filter_jaccard:.4f} |",
        f"| Value ID | Pairwise F1 | {results.before_filter_symmetric_f1:.4f} | {results.after_filter_symmetric_f1:.4f} |",
        "",
        "---",
        "",
        "## Analysis & Findings",
        "",
    ])

    # Analysis
    findings = []
    
    kappa_change = results.after_filter_cohen_kappa - results.before_filter_cohen_kappa
    if kappa_change > 0.05:
        findings.append(f"- Evidence Agent filtering **improved Cohen's κ** by {kappa_change:.4f}, indicating better alignment with human annotations.")
    elif kappa_change > 0:
        findings.append(f"- Evidence Agent filtering showed slight improvement in Cohen's κ (+{kappa_change:.4f}).")
    else:
        findings.append("- Evidence Agent filtering had limited impact on Cohen's κ.")

    jaccard_change = results.after_filter_jaccard - results.before_filter_jaccard
    if jaccard_change > 0.05:
        findings.append(f"- Value identification **Jaccard similarity improved** by {jaccard_change:.4f} after filtering.")
    elif jaccard_change > 0:
        findings.append(f"- Value identification Jaccard similarity showed slight improvement (+{jaccard_change:.4f}).")
    else:
        findings.append("- Value identification Jaccard similarity remained stable after filtering.")

    # Confirmation rate analysis
    conf_rate = results.evidence_stats.confirmation_rate
    if conf_rate > 0.9:
        findings.append(f"- **High confirmation rate ({_format_percent(conf_rate)})** suggests the Evidence Agent may be too lenient in verification.")
    elif conf_rate < 0.3:
        findings.append(f"- **Low confirmation rate ({_format_percent(conf_rate)})** suggests the Evidence Agent is strict in filtering hypotheses.")
    else:
        findings.append(f"- Confirmation rate of {_format_percent(conf_rate)} indicates balanced hypothesis filtering.")

    # Kappa interpretation
    kappa = results.after_filter_cohen_kappa
    findings.append(f"- Final Cohen's κ = {kappa:.4f} indicates **{_interpret_kappa(kappa)}** agreement with human annotations.")

    lines.extend(findings)

    return "\n".join(lines)


def generate_latex_tables(results: PipelineEvaluationResults) -> str:
    """Generate LaTeX tables for paper inclusion."""
    latex = [
        "% Pipeline Evaluation Results - LaTeX Tables",
        "% Auto-generated",
        "",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Human vs Pipeline Agreement: Risk Detection}",
        "\\label{tab:pipeline-risk}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Metric & Before Filter & After Filter & Change & Interpretation \\\\",
        "\\midrule",
        f"\\% Agreement & {_format_percent(results.before_filter_percent_agreement)} & {_format_percent(results.after_filter_percent_agreement)} & {_format_change(results.before_filter_percent_agreement, results.after_filter_percent_agreement)} & — \\\\",
        f"Cohen's $\\kappa$ & {results.before_filter_cohen_kappa:.4f} & {results.after_filter_cohen_kappa:.4f} & {_format_change(results.before_filter_cohen_kappa, results.after_filter_cohen_kappa)} & {_interpret_kappa(results.after_filter_cohen_kappa)} \\\\",
        f"PABAK & {results.before_filter_pabak:.4f} & {results.after_filter_pabak:.4f} & {_format_change(results.before_filter_pabak, results.after_filter_pabak)} & {_interpret_kappa(results.after_filter_pabak)} \\\\",
        f"Gwet's AC1 & {results.before_filter_gwet_ac1:.4f} & {results.after_filter_gwet_ac1:.4f} & {_format_change(results.before_filter_gwet_ac1, results.after_filter_gwet_ac1)} & {_interpret_kappa(results.after_filter_gwet_ac1)} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Human vs Pipeline Agreement: Value Identification}",
        "\\label{tab:pipeline-value}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Metric & Before Filter & After Filter & Change \\\\",
        "\\midrule",
        f"Pairwise Jaccard & {results.before_filter_jaccard:.4f} & {results.after_filter_jaccard:.4f} & {_format_change(results.before_filter_jaccard, results.after_filter_jaccard)} \\\\",
        f"Pairwise F1 & {results.before_filter_symmetric_f1:.4f} & {results.after_filter_symmetric_f1:.4f} & {_format_change(results.before_filter_symmetric_f1, results.after_filter_symmetric_f1)} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(latex)


# ============================================================
# Multi-Model Comparison
# ============================================================

@dataclass
class ModelRow:
    """Single model row for comparison table."""
    model_name: str
    total_samples: int
    hypotheses_generated: int
    confirmation_rate: float
    avg_confidence: float
    # Before filter
    before_kappa: float
    before_pabak: float
    before_gwet: float
    before_agreement: float
    before_jaccard: float
    before_f1: float
    # After filter
    after_kappa: float
    after_pabak: float
    after_gwet: float
    after_agreement: float
    after_jaccard: float
    after_f1: float


def run_multi_model_comparison(
    samples_files: list[str],
    repo_paths: list[str],
    model_providers: list[str],
    max_samples: Optional[int] = None,
    mock_mode: bool = False,
    parallel_workers: int = 1,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    output_dir: Optional[str] = None,
    output_base_name: str = "pipeline_multimodel",
) -> list[ModelRow]:
    """Run pipeline evaluation across multiple LLM providers and collect comparison rows.

    Results are saved incrementally after each model completes when output_dir is provided,
    so partial results are preserved even if the run is interrupted.
    """
    # --- Merge multiple sample files ---
    all_samples: list[CommitSample] = []
    for i, sf in enumerate(samples_files):
        loaded = load_samples_from_json(sf)
        # If each file corresponds to an independent repo_path, inject it into sample metadata for evidence search
        rp = repo_paths[i] if i < len(repo_paths) else repo_paths[0]
        for s in loaded:
            # Record repo_path to each sample (for selecting the correct repo during evidence location)
            s._repo_path_override = rp
        all_samples.extend(loaded)
        print(f"  Loaded {len(loaded)} samples from {sf} (repo: {rp})")

    print(f"  Total merged samples: {len(all_samples)}")
    if max_samples:
        all_samples = all_samples[:max_samples]
        print(f"  After max_samples limit: {len(all_samples)}")

    # Evidence search needs the correct repo_path for each sample.
    # To achieve this, PipelineEvaluator.evaluate_sample needs repo_path passed manually.
    # Use the first repo_path as default (single-repo mode; multi-repo via _repo_path_override)
    primary_repo = repo_paths[0]

    rows: list[ModelRow] = []

    for provider in model_providers:
        print(f"\n{'='*60}")
        print(f"Running pipeline with LLM provider: {provider}")
        if parallel_workers > 1:
            print(f"Parallel workers: {parallel_workers}")
        print(f"{'='*60}")

        evaluator = PipelineEvaluator(
            repo_path=primary_repo,
            llm_provider=provider,
            mock_mode=mock_mode,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )
        if mock_mode:
            logger.warning(f"⚠️  MOCK MODE for {provider} - no real API calls")
        else:
            logger.info(f"✓ REAL API MODE for {provider} - will call LLM APIs")
        results = evaluator.evaluate_all(
            all_samples,
            parallel_workers=parallel_workers,
        )

        rows.append(ModelRow(
            model_name=provider,
            total_samples=results.total_samples,
            hypotheses_generated=results.hypothesis_stats.total_generated,
            confirmation_rate=results.evidence_stats.confirmation_rate,
            avg_confidence=results.hypothesis_stats.avg_confidence,
            before_kappa=results.before_filter_cohen_kappa,
            before_pabak=results.before_filter_pabak,
            before_gwet=results.before_filter_gwet_ac1,
            before_agreement=results.before_filter_percent_agreement,
            before_jaccard=results.before_filter_jaccard,
            before_f1=results.before_filter_symmetric_f1,
            after_kappa=results.after_filter_cohen_kappa,
            after_pabak=results.after_filter_pabak,
            after_gwet=results.after_filter_gwet_ac1,
            after_agreement=results.after_filter_percent_agreement,
            after_jaccard=results.after_filter_jaccard,
            after_f1=results.after_filter_symmetric_f1,
        ))

        print(f"  [{provider}] κ Before/After: "
              f"{results.before_filter_cohen_kappa:.4f} / {results.after_filter_cohen_kappa:.4f}  "
              f"Jaccard: {results.before_filter_jaccard:.4f} / {results.after_filter_jaccard:.4f}")

        # -- Incrementally write files after each model completes --
        if output_dir:
            save_multi_model_results(
                rows=rows,
                output_dir=output_dir,
                base_name=output_base_name,
            )
            print(f"  [Live write] {output_dir}/{output_base_name}.{{json,md,tex}} "
                  f"({len(rows)}/{len(model_providers)} models completed)")

    return rows


# ============================================================
# Multi-Model Report Generation (Paper Quality)
# ============================================================

def _model_display_name(provider: str) -> str:
    """Convert provider key to a clean display name for tables."""
    mapping = {
        "deepseek": "DeepSeek-V3",
        "deepseek-chat": "DeepSeek-V3",
        "qwen": "Qwen-Plus",
        "qwen-plus": "Qwen-Plus",
        "o4-mini": "o4-mini",
        "claude-sonnet-4-5": "Claude-3.5-Sonnet",
        "gpt-5.2": "GPT-5.2",
        "grok-4": "Grok-4",
        "gemini-2.5-flash": "Gemini-2.5-Flash",
    }
    return mapping.get(provider, provider)


def generate_multi_model_markdown(rows: list[ModelRow]) -> str:
    """Generate a Markdown comparison table for all models."""
    lines = [
        "# Pipeline Evaluation: Multi-Model Comparison",
        "",
        "## Dimension 1: Risk Detection Agreement (Human vs Pipeline)",
        "",
        "| Model | PA (Bef.) | κ (Bef.) | PABAK (Bef.) | AC1 (Bef.) "
        "| PA (Aft.) | κ (Aft.) | PABAK (Aft.) | AC1 (Aft.) | Δκ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        delta = r.after_kappa - r.before_kappa
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {_model_display_name(r.model_name)} "
            f"| {r.before_agreement:.3f} | {r.before_kappa:.3f} | {r.before_pabak:.3f} | {r.before_gwet:.3f} "
            f"| {r.after_agreement:.3f} | {r.after_kappa:.3f} | {r.after_pabak:.3f} | {r.after_gwet:.3f} "
            f"| {sign}{delta:.3f} |"
        )

    lines.extend([
        "",
        "*PA = Percent Agreement, κ = Cohen's Kappa, PABAK = Prevalence-Adjusted Bias-Adjusted Kappa,*",
        "*AC1 = Gwet's AC1, Bef. = Before Evidence Filter, Aft. = After Evidence Filter, Δκ = Improvement*",
        "",
        "## Dimension 2: Value Identification Agreement (Human vs Pipeline)",
        "",
        "| Model | Jaccard (Bef.) | F1 (Bef.) | Jaccard (Aft.) | F1 (Aft.) | ΔJaccard |",
        "|---|---|---|---|---|---|",
    ])
    for r in rows:
        delta = r.after_jaccard - r.before_jaccard
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {_model_display_name(r.model_name)} "
            f"| {r.before_jaccard:.3f} | {r.before_f1:.3f} "
            f"| {r.after_jaccard:.3f} | {r.after_f1:.3f} "
            f"| {sign}{delta:.3f} |"
        )

    lines.extend([
        "",
        "## Pipeline Statistics per Model",
        "",
        "| Model | Samples | Hypotheses | Confirm Rate | Avg Confidence |",
        "|---|---|---|---|---|",
    ])
    for r in rows:
        lines.append(
            f"| {_model_display_name(r.model_name)} "
            f"| {r.total_samples} | {r.hypotheses_generated} "
            f"| {r.confirmation_rate:.1%} | {r.avg_confidence:.3f} |"
        )

    return "\n".join(lines)


def generate_multi_model_latex(rows: list[ModelRow]) -> str:
    """Generate LaTeX tables for paper inclusion (multi-model comparison).

    Table 1 (table*): Dim.1 (Risk Detection) and Dim.2 (Value Identification)
                      merged into one wide table for compactness.
    Table 2 (table):  Pipeline Statistics per LLM.
    """
    lines = [
        "% Auto-generated by pipeline_evaluator.py",
        "% Requires: booktabs, multirow, bm packages",
        "",
        # ===== Table 1: Risk Detection + Value Identification (merged) =====
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Human vs.\ Pipeline Agreement on Risk Detection (Dim.~1) and Value "
        r"Identification (Dim.~2) before/after Evidence Filtering. "
        r"$\Delta\kappa$ = Cohen's $\kappa$ improvement; $\Delta$Jac.\ = Jaccard improvement.}",
        r"\label{tab:pipeline-agreement-multimodel}",
        # 15 columns: model | 4 before-risk | 4 after-risk | Δκ | 2 before-val | 2 after-val | ΔJac
        r"\begin{tabular}{l cccc cccc r cc cc r}",
        r"\toprule",
        # Row 0: top-level dimension headers (no Model cell here, added via multirow below)
        r"& \multicolumn{9}{c}{\textbf{Dimension 1: Risk Detection}}"
        r" & \multicolumn{5}{c}{\textbf{Dimension 2: Value Identification}} \\",
        r"\cmidrule(lr){2-10}\cmidrule(lr){11-15}",
        # Row 1: Before/After sub-headers + delta labels (Model uses multirow spanning rows 1-2)
        r"\multirow{2}{*}{\textbf{Model}}"
        r" & \multicolumn{4}{c}{\textbf{Before Filter}}"
        r" & \multicolumn{4}{c}{\textbf{After Filter}}"
        r" & \multirow{2}{*}{$\bm{\Delta\kappa}$}"
        r" & \multicolumn{2}{c}{\textbf{Before Filter}}"
        r" & \multicolumn{2}{c}{\textbf{After Filter}}"
        r" & \multirow{2}{*}{$\bm{\Delta}$Jac.} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){11-12}\cmidrule(lr){13-14}",
        # Row 2: individual metric names
        r"& PA & $\kappa$ & PABAK & AC1"
        r" & PA & $\kappa$ & PABAK & AC1 &"
        r" & Jaccard & F1 & Jaccard & F1 & \\",
        r"\midrule",
    ]

    for r in rows:
        dk = r.after_kappa - r.before_kappa
        dk_sign = "+" if dk >= 0 else ""
        dk_bold0 = r"\textbf{" if dk > 0 else ""
        dk_bold1 = r"}" if dk > 0 else ""

        dj = r.after_jaccard - r.before_jaccard
        dj_sign = "+" if dj >= 0 else ""
        dj_bold0 = r"\textbf{" if dj > 0 else ""
        dj_bold1 = r"}" if dj > 0 else ""

        lines.append(
            f"{_model_display_name(r.model_name)}"
            f" & {r.before_agreement:.3f} & {r.before_kappa:.3f}"
            f" & {r.before_pabak:.3f} & {r.before_gwet:.3f}"
            f" & {r.after_agreement:.3f} & {r.after_kappa:.3f}"
            f" & {r.after_pabak:.3f} & {r.after_gwet:.3f}"
            f" & {dk_bold0}{dk_sign}{dk:.3f}{dk_bold1}"
            f" & {r.before_jaccard:.3f} & {r.before_f1:.3f}"
            f" & {r.after_jaccard:.3f} & {r.after_f1:.3f}"
            f" & {dj_bold0}{dj_sign}{dj:.3f}{dj_bold1} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
        # ===== Table 2: Pipeline Stats =====
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pipeline Statistics per LLM: number of hypotheses generated, "
        r"evidence confirmation rate, and average hypothesis confidence.}",
        r"\label{tab:pipeline-stats-multimodel}",
        r"\begin{tabular}{l r r r r}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Samples} & \textbf{Hypotheses}"
        r" & \textbf{Confirm Rate} & \textbf{Avg.\ Conf.} \\",
        r"\midrule",
    ])

    for r in rows:
        # confirmation_rate is a float (0-1); format as LaTeX-safe percentage with \%
        confirm_pct = f"{r.confirmation_rate * 100:.1f}\\%"
        lines.append(
            f"{_model_display_name(r.model_name)}"
            f" & {r.total_samples} & {r.hypotheses_generated}"
            f" & {confirm_pct} & {r.avg_confidence:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def save_multi_model_results(
    rows: list[ModelRow],
    output_dir: str,
    base_name: str = "pipeline_multimodel",
) -> dict[str, str]:
    """Save multi-model comparison results to JSON, Markdown, and LaTeX."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = {}

    # JSON
    json_path = out / f"{base_name}.json"
    data = []
    for r in rows:
        data.append({
            "model": r.model_name,
            "display_name": _model_display_name(r.model_name),
            "total_samples": r.total_samples,
            "hypotheses_generated": r.hypotheses_generated,
            "confirmation_rate": r.confirmation_rate,
            "avg_confidence": r.avg_confidence,
            "before_filter": {
                "percent_agreement": r.before_agreement,
                "cohen_kappa": r.before_kappa,
                "pabak": r.before_pabak,
                "gwet_ac1": r.before_gwet,
                "jaccard": r.before_jaccard,
                "symmetric_f1": r.before_f1,
            },
            "after_filter": {
                "percent_agreement": r.after_agreement,
                "cohen_kappa": r.after_kappa,
                "pabak": r.after_pabak,
                "gwet_ac1": r.after_gwet,
                "jaccard": r.after_jaccard,
                "symmetric_f1": r.after_f1,
            },
            "delta": {
                "kappa": r.after_kappa - r.before_kappa,
                "jaccard": r.after_jaccard - r.before_jaccard,
            }
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    saved["json"] = str(json_path)

    # Markdown
    md_path = out / f"{base_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_multi_model_markdown(rows))
    saved["markdown"] = str(md_path)

    # LaTeX
    tex_path = out / f"{base_name}.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(generate_multi_model_latex(rows))
    saved["latex"] = str(tex_path)

    return saved


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline Evaluator - Hypothesis + Evidence Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single repo, single model
  python pipeline_evaluator.py data/focus_android_samples.json data/repo_data/focus-android

  # Single repo, all models, parallel API calls
  python pipeline_evaluator.py data/focus_android_samples.json data/repo_data/focus-android \\
      --all-models deepseek-chat qwen-plus grok-4 --parallel-workers 4

  # Merge two repos (Signal + Focus), all models
  python pipeline_evaluator.py \\
      --samples-files data/focus_android_samples.json data/signal_android_samples.json \\
      --repo-paths data/repo_data/focus-android data/repo_data/Signal-Android \\
      --all-models deepseek-chat qwen-plus --parallel-workers 4
"""
    )
    # Positional arguments (single sample file mode, backward compatible)
    parser.add_argument("samples_file", nargs="?", help="Path to samples JSON file (single-file mode)")
    parser.add_argument("repo_path", nargs="?", help="Path to repository (single-file mode)")
    # Multi-file mode
    parser.add_argument(
        "--samples-files", nargs="+", metavar="FILE",
        help="One or more samples JSON files (merged before evaluation). Overrides positional samples_file."
    )
    parser.add_argument(
        "--repo-paths", nargs="+", metavar="REPO",
        help="Repository paths corresponding to each --samples-files entry. Overrides positional repo_path."
    )
    # Common parameters
    parser.add_argument("--output", "-o", default="pipeline_evaluation_results.json", help="Output JSON file (single-model mode)")
    parser.add_argument("--report", "-r", default="pipeline_evaluation_report.md", help="Output report file (single-model mode)")
    parser.add_argument("--latex", "-l", default=None, help="Output LaTeX file (optional, single-model mode)")
    parser.add_argument("--max-samples", "-n", type=int, default=None, help="Max samples to evaluate (after merging)")
    parser.add_argument("--llm-provider", default="deepseek", help="LLM provider (single-model mode)")
    parser.add_argument("--mock", action="store_true", help="Use mock mode (no real LLM calls)")
    parser.add_argument(
        "--parallel-workers", type=int, default=1, metavar="N",
        help="Number of parallel threads for LLM API calls per model (default: 1 = sequential). Recommended: 4~8."
    )
    # Multi-model mode
    parser.add_argument(
        "--all-models",
        nargs="+",
        metavar="PROVIDER",
        help=(
            "Run multi-model comparison mode. Pass one or more provider keys, e.g.: "
            "--all-models deepseek-chat qwen-plus o4-mini claude-sonnet-4-5 gpt-5.2 grok-4 gemini-2.5-flash"
        ),
    )
    parser.add_argument("--output-dir", default="experiment_results/pipeline", help="Output directory for multi-model results")
    parser.add_argument("--output-name", default="pipeline_multimodel", help="Base filename for multi-model outputs")
    parser.add_argument("--cache-dir", default=None, metavar="DIR",
                        help="Cache directory for per-sample LLM results (default: experiment_logs/llm_outputs/pipeline_cache/<provider>/)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable result caching (re-run all samples from scratch)")
    args = parser.parse_args()
    if args.samples_files:
        resolved_samples_files = args.samples_files
        resolved_repo_paths = args.repo_paths if args.repo_paths else (
            [args.repo_path] if args.repo_path else None
        )
    else:
        if not args.samples_file or not args.repo_path:
            parser.error(
                "Must provide either positional arguments (samples_file repo_path) "
                "or --samples-files and --repo-paths."
            )
        resolved_samples_files = [args.samples_file]
        resolved_repo_paths = [args.repo_path]

    if not resolved_repo_paths:
        parser.error("Must provide at least one repo path via positional arg or --repo-paths.")

    # --------------------------------------------------------
    # Multi-model comparison mode
    # --------------------------------------------------------
    if args.all_models:
        print(f"\n=== Multi-Model Pipeline Comparison ===")
        print(f"Models: {args.all_models}")
        print(f"Samples files: {resolved_samples_files}")
        print(f"Repo paths:    {resolved_repo_paths}")
        if args.parallel_workers > 1:
            print(f"Parallel workers per model: {args.parallel_workers}")
        if args.max_samples:
            print(f"Max samples (after merge): {args.max_samples}")
        print()

        rows = run_multi_model_comparison(
            samples_files=resolved_samples_files,
            repo_paths=resolved_repo_paths,
            model_providers=args.all_models,
            max_samples=args.max_samples,
            mock_mode=args.mock,
            parallel_workers=args.parallel_workers,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            output_dir=args.output_dir,
            output_base_name=args.output_name,
        )

        # Final results already written incrementally in run_multi_model_comparison
        # Only print the final summary table here
        print(f"\n=== Multi-Model Results Saved ===")
        print(f"  Output dir : {args.output_dir}/{args.output_name}.{{json,md,tex}}")

        # Print summary table
        print(f"\n{'Model':<25} {'κ Before':>9} {'κ After':>9} {'Δκ':>7} {'Jac Bef':>9} {'Jac Aft':>9}")
        print("-" * 70)
        for r in rows:
            dk = r.after_kappa - r.before_kappa
            print(
                f"{_model_display_name(r.model_name):<25} "
                f"{r.before_kappa:>9.4f} {r.after_kappa:>9.4f} {dk:>+7.4f} "
                f"{r.before_jaccard:>9.4f} {r.after_jaccard:>9.4f}"
            )
        return

    # --------------------------------------------------------
    # Single-model mode (original behavior)
    # --------------------------------------------------------
    print(f"Loading samples from {resolved_samples_files}...")
    all_samples: list[CommitSample] = []
    for sf in resolved_samples_files:
        loaded = load_samples_from_json(sf)
        print(f"  {len(loaded)} samples from {sf}")
        all_samples.extend(loaded)
    print(f"Total: {len(all_samples)} samples")

    # Create evaluator
    print(f"Initializing evaluator..." + (" [MOCK MODE]" if args.mock else ""))
    evaluator = PipelineEvaluator(
        repo_path=resolved_repo_paths[0],
        llm_provider=args.llm_provider,
        mock_mode=args.mock,
    )

    # Run evaluation
    print(f"Starting evaluation" + (f" ({args.parallel_workers} parallel workers)..." if args.parallel_workers > 1 else "..."))
    results = evaluator.evaluate_all(
        all_samples,
        max_samples=args.max_samples,
        parallel_workers=args.parallel_workers,
    )

    # Save results
    results_dict = {
        "total_samples": results.total_samples,
        "total_time_seconds": results.total_time_seconds,
        "hypothesis_stats": {
            "total_generated": results.hypothesis_stats.total_generated,
            "avg_confidence": results.hypothesis_stats.avg_confidence,
            "confidence_distribution": results.hypothesis_stats.confidence_distribution,
            "value_distribution": results.hypothesis_stats.value_distribution,
            "severity_distribution": results.hypothesis_stats.severity_distribution,
        },
        "evidence_stats": {
            "total_hypotheses": results.evidence_stats.total_hypotheses,
            "confirmed_count": results.evidence_stats.confirmed_count,
            "unverified_count": results.evidence_stats.unverified_count,
            "rejected_count": results.evidence_stats.rejected_count,
            "confirmation_rate": results.evidence_stats.confirmation_rate,
            "avg_relevance_score": results.evidence_stats.avg_relevance_score,
        },
        "iaa_metrics": {
            "before_filter": {
                "risk_percent_agreement": results.before_filter_percent_agreement,
                "risk_cohen_kappa": results.before_filter_cohen_kappa,
                "risk_pabak": results.before_filter_pabak,
                "risk_gwet_ac1": results.before_filter_gwet_ac1,
                "value_jaccard": results.before_filter_jaccard,
                "value_symmetric_f1": results.before_filter_symmetric_f1,
            },
            "after_filter": {
                "risk_percent_agreement": results.after_filter_percent_agreement,
                "risk_cohen_kappa": results.after_filter_cohen_kappa,
                "risk_pabak": results.after_filter_pabak,
                "risk_gwet_ac1": results.after_filter_gwet_ac1,
                "value_jaccard": results.after_filter_jaccard,
                "value_symmetric_f1": results.after_filter_symmetric_f1,
            },
        },
        "sample_results": [sr.to_dict() for sr in results.sample_results],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.output}")

    # Generate markdown report
    report = generate_report(results)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {args.report}")

    # Generate LaTeX tables if requested
    if args.latex:
        latex = generate_latex_tables(results)
        with open(args.latex, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"LaTeX tables saved to {args.latex}")

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary (Human vs Pipeline IAA)")
    print("=" * 60)
    print(f"Total Samples: {results.total_samples}")
    print(f"Hypotheses Generated: {results.hypothesis_stats.total_generated}")
    print(f"Confirmation Rate: {results.evidence_stats.confirmation_rate:.1%}")
    print(f"Total Time: {results.total_time_seconds:.1f}s")
    print()
    print("Risk Detection Agreement (Cohen's κ):")
    print(f"  Before Filter: {results.before_filter_cohen_kappa:.4f} ({_interpret_kappa(results.before_filter_cohen_kappa)})")
    print(f"  After Filter:  {results.after_filter_cohen_kappa:.4f} ({_interpret_kappa(results.after_filter_cohen_kappa)})")
    print()
    print("Value Identification Agreement (Jaccard):")
    print(f"  Before Filter: {results.before_filter_jaccard:.4f}")
    print(f"  After Filter:  {results.after_filter_jaccard:.4f}")


if __name__ == "__main__":
    main()
