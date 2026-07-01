#!/usr/bin/env python3
"""
Unified pipeline evaluator for the main experiment.

Supports both scenarios:
  - Code: Profile -> HypothesisAgent -> EvidenceAgent
  - Text: text-specific prompt + LLM evidence verification

Ablation controls:
  - profile_mode: "real" (use profile_map) / "uniform" (all 0.5)
  - skip_evidence: True (skip verification) / False
"""

import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Project paths
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from valueguard.core.models import (
    ValueProfile,
    EvidenceStatus,
)

try:
    from .benchmark_builder import BenchmarkSample
    from .iaa_metrics import (
        precision_recall_f1_binary,
        micro_precision_recall_f1,
        percent_agreement,
        cohen_kappa_binary,
        pabak,
        gwet_ac1,
        pairwise_jaccard,
        pairwise_symmetric_f1,
        recall_of_gt,
        precision_of_gt,
        partial_credit_jaccard,
    )
    from .llm_client import LLMClientFactory, BaseLLMClient
    from . import paths as exp_paths
except ImportError:
    from benchmark_builder import BenchmarkSample
    from iaa_metrics import (
        precision_recall_f1_binary,
        micro_precision_recall_f1,
        percent_agreement,
        cohen_kappa_binary,
        pabak,
        gwet_ac1,
        pairwise_jaccard,
        pairwise_symmetric_f1,
        recall_of_gt,
        precision_of_gt,
        partial_credit_jaccard,
    )
    from llm_client import LLMClientFactory, BaseLLMClient
    import paths as exp_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Text 场景专用 Prompt
# ============================================================

TEXT_HYPOTHESIS_SYSTEM = """You are a value deviation analyst for software project discussions.
Your task is to analyze GitHub issue discussions and identify potential value deviations using
four-layer cross-layer reasoning:

L1 (Schwartz Values) -> L2 (Human Value Themes) -> L3 (System Value Themes) -> L4 (Discussion Indicators)

## Analysis Process:
1. Examine the issue discussion text
2. Identify any L4 discussion indicators (user complaints, feature requests, privacy concerns, etc.)
3. Trace upward through L3 -> L2 -> L1 to understand the value semantics
4. Determine if there's a deviation from the project's declared value profile
5. Assess confidence and severity

## Output Format (JSON):
{
  "hypotheses": [
    {
      "value_id": "HV9",
      "deviation_type": "violation|inconsistency|risk",
      "confidence": 0.0-1.0,
      "severity": "low|medium|high|critical",
      "description": "Brief description of the deviation",
      "suggested_action": "What should be reviewed or changed",
      "cross_layer_trace": {
        "l1_value": "Security",
        "l2_theme": "HV9 Privacy",
        "l3_attribute": "SV10 Longevity",
        "l4_indicator": "user complaint about data collection",
        "reasoning": "How this discussion relates to the value chain"
      }
    }
  ]
}

Only include hypotheses with confidence >= 0.5.
If no value deviations are found, respond with: {"hypotheses": []}
"""

TEXT_EVIDENCE_SYSTEM = """You are a discussion evidence verifier. Your task is to determine if an issue discussion
actually supports a given value deviation hypothesis.

## Verification Criteria
1. CONFIRMED: The discussion clearly shows the claimed deviation with specific quotes or user statements
2. REJECTED: The discussion does NOT support the hypothesis claim
3. UNVERIFIED: Cannot determine - discussion is ambiguous or lacks specific evidence

## Output Format (JSON):
{
  "status": "CONFIRMED|REJECTED|UNVERIFIED",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation",
  "evidence_snippet": "Most relevant quote from the discussion if CONFIRMED"
}

Be strict! Only CONFIRM when you see clear, specific evidence in the discussion."""


# ============================================================
# 统一评估结果
# ============================================================

@dataclass
class UnifiedSampleResult:
    """统一评估结果（code 和 text 场景共用）"""
    sample_id: str
    scenario_type: str            # "code" / "text"
    repo: str

    # 预测结果
    predicted_has_risk: bool = False
    predicted_values: list[str] = field(default_factory=list)

    # 置信度向量（value_id -> confidence）
    predicted_confidences: dict[str, float] = field(default_factory=dict)

    # Ground truth
    ground_truth_has_risk: bool = False
    ground_truth_values: list[str] = field(default_factory=list)

    # Pipeline 统计
    hypothesis_count: int = 0
    confirmed_count: int = 0
    profile_used: str = "unknown"  # "real" / "uniform"

    # 计时
    total_time_ms: float = 0.0

    # 原始 LLM 输出（可选，用于调试）
    raw_hypothesis_output: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "scenario_type": self.scenario_type,
            "repo": self.repo,
            "predicted_has_risk": self.predicted_has_risk,
            "predicted_values": self.predicted_values,
            "predicted_confidences": self.predicted_confidences,
            "ground_truth_has_risk": self.ground_truth_has_risk,
            "ground_truth_values": self.ground_truth_values,
            "hypothesis_count": self.hypothesis_count,
            "confirmed_count": self.confirmed_count,
            "profile_used": self.profile_used,
            "total_time_ms": self.total_time_ms,
        }


# ============================================================
# Text Pipeline 评估器
# ============================================================

class TextPipelineEvaluator:
    """Text 场景的 Pipeline 评估器（issue text → hypothesis → evidence）"""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        profile_map: Optional[dict[str, ValueProfile]] = None,
        skip_evidence: bool = False,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        profile_alpha: float = 1.0,
        text_confidence_threshold: float = 0.5,
    ):
        self.llm_client = llm_client
        self.profile_map = profile_map or {}
        self.skip_evidence = skip_evidence
        self.profile_alpha = float(profile_alpha)
        self.text_confidence_threshold = float(text_confidence_threshold)

        # Cache configuration
        if use_cache:
            _base = Path(cache_dir) if cache_dir else exp_paths.TEXT_PIPELINE_CACHE_DIR
            suffix_parts = []
            if self.profile_alpha != 1.0:
                suffix_parts.append(f"alpha{self.profile_alpha}")
            if self.text_confidence_threshold != 0.5:
                suffix_parts.append(f"tct{self.text_confidence_threshold}")
            suffix = "_".join(suffix_parts)
            self._cache_dir = _base / suffix if suffix else _base
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_dir = None

        # Uniform prior
        self._uniform_profile = ValueProfile(
            repo="uniform",
            l2_scores={f"HV{i}": 0.5 for i in range(1, 11)},
            l3_scores={f"SV{i}": 0.5 for i in range(1, 11)},
        )

    def _resolve_profile(self, repo: str) -> ValueProfile:
        """匹配 repo 到 profile"""
        if not repo or not self.profile_map:
            return self._uniform_profile

        normalized = repo.lower().replace("-", "").replace("_", "").replace(" ", "")
        for key, profile in self.profile_map.items():
            key_norm = key.lower().replace("-", "").replace("_", "").replace(" ", "")
            if key_norm == normalized or normalized in key_norm or key_norm in normalized:
                return profile
        return self._uniform_profile

    def _cache_path(self, sample_id: str) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{sample_id}.json"

    def _load_from_cache(self, sample_id: str) -> Optional[UnifiedSampleResult]:
        path = self._cache_path(sample_id)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return UnifiedSampleResult(
                sample_id=data["sample_id"],
                scenario_type=data.get("scenario_type", "text"),
                repo=data.get("repo", "unknown"),
                predicted_has_risk=data["predicted_has_risk"],
                predicted_values=data.get("predicted_values", []),
                predicted_confidences=data.get("predicted_confidences", {}),
                ground_truth_has_risk=data.get("ground_truth_has_risk", False),
                ground_truth_values=data.get("ground_truth_values", []),
                hypothesis_count=data.get("hypothesis_count", 0),
                confirmed_count=data.get("confirmed_count", 0),
                profile_used=data.get("profile_used", "unknown"),
                total_time_ms=data.get("total_time_ms", 0.0),
            )
        except Exception as e:
            logger.warning(f"缓存读取失败 {sample_id}: {e}")
            return None

    def _save_to_cache(self, result: UnifiedSampleResult):
        path = self._cache_path(result.sample_id)
        if path is None:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"缓存写入失败 {result.sample_id}: {e}")

    def _build_hypothesis_prompt(self, content: str, profile: ValueProfile) -> str:
        """构建 text 场景的假说生成 prompt"""
        parts = []

        # Profile context
        parts.append("## Project Value Profile")
        if profile.core_values:
            parts.append(f"Core values: {', '.join(profile.core_values)}")
        if profile.l2_scores:
            top_l2 = sorted(profile.l2_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            parts.append("Top L2 values: " + ", ".join(f"{k}({v:.2f})" for k, v in top_l2))
        if profile.l3_scores:
            top_l3 = sorted(profile.l3_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            parts.append("Top L3 values: " + ", ".join(f"{k}({v:.2f})" for k, v in top_l3))

        # Issue text
        parts.append("\n## Issue Discussion to Analyze")
        parts.append(content)

        parts.append("\nAnalyze this issue discussion for potential value deviations.")
        parts.append(
            "\n**IMPORTANT: You MUST respond with ONLY a valid JSON object. No explanations, no markdown, no natural language.**"
            "\nBe generous in your analysis — even minor or potential deviations should be reported with appropriate confidence (0.5-0.7)."
            "\nIf truly no deviations are found, respond with: {\"hypotheses\": []}"
        )

        return "\n".join(parts)

    def _build_evidence_prompt(self, content: str, hypothesis_desc: str, value_id: str) -> str:
        """构建 text 场景的证据验证 prompt"""
        parts = [
            "## Hypothesis to Verify",
            f"Value ID: {value_id}",
            f"Description: {hypothesis_desc}",
            "",
            "## Original Discussion",
            content[:4000],  # 截断过长内容
            "",
            "Verify whether this discussion provides evidence supporting the hypothesis above.",
            "\n**Respond with ONLY a valid JSON object.**",
        ]
        return "\n".join(parts)

    def _parse_hypotheses(self, raw_text: str) -> list[dict]:
        """解析 LLM 返回的假说 JSON"""
        # 尝试提取 JSON
        try:
            # 去除可能的 markdown 标记
            text = raw_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])

            parsed = json.loads(text)
            if isinstance(parsed, dict) and "hypotheses" in parsed:
                return parsed["hypotheses"]
            elif isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            # 尝试从文本中提取 JSON
            import re
            match = re.search(r'\{[\s\S]*"hypotheses"[\s\S]*\}', raw_text)
            if match:
                try:
                    parsed = json.loads(match.group())
                    return parsed.get("hypotheses", [])
                except json.JSONDecodeError:
                    pass
        return []

    def _parse_evidence(self, raw_text: str) -> dict:
        """解析 LLM 返回的证据验证 JSON"""
        try:
            text = raw_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            return json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[\s\S]*"status"[\s\S]*\}', raw_text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {"status": "UNVERIFIED", "confidence": 0.0, "reasoning": "parse_error"}

    def evaluate_sample(self, sample: BenchmarkSample) -> UnifiedSampleResult:
        """评估单个 text 样本"""
        # 缓存检查
        cached = self._load_from_cache(sample.sample_id)
        if cached is not None:
            return cached

        start_time = time.time()
        result = UnifiedSampleResult(
            sample_id=sample.sample_id,
            scenario_type=sample.scenario_type,
            repo=sample.repo,
            ground_truth_has_risk=sample.has_value_risk,
            ground_truth_values=sample.ground_truth_values,
        )

        # 解析 profile
        profile = self._resolve_profile(sample.repo)
        result.profile_used = "real" if profile is not self._uniform_profile else "uniform"

        # Stage 1: Hypothesis generation
        prompt = self._build_hypothesis_prompt(sample.content, profile)
        try:
            response = self.llm_client.send_request([
                {"role": "system", "content": TEXT_HYPOTHESIS_SYSTEM},
                {"role": "user", "content": prompt},
            ])
            raw_hypotheses = self._parse_hypotheses(response)
            result.raw_hypothesis_output = response[:500] if response else ""
        except Exception as e:
            logger.error(f"Text hypothesis generation failed for {sample.sample_id}: {e}")
            raw_hypotheses = []

        # Apply profile-aware confidence adjustment for text scenarios
        adjusted = []
        for h in raw_hypotheses:
            conf = float(h.get("confidence", 0))
            if conf <= 0:
                continue
            vid = h.get("value_id", "")
            profile_score = max(
                profile.l2_scores.get(vid, 0.0),
                profile.l3_scores.get(vid, 0.0),
            )
            if profile_score > 0 and self.profile_alpha != 0.0:
                all_scores = list(profile.l2_scores.values()) + list(profile.l3_scores.values())
                mu = sum(all_scores) / len(all_scores) if all_scores else 0.5
                ratio = profile_score / mu if mu > 0 else 1.0
                conf = conf * (ratio ** self.profile_alpha)
            if conf >= self.text_confidence_threshold:
                h["confidence"] = min(conf, 1.0)
                adjusted.append(h)
        hypotheses = adjusted
        result.hypothesis_count = len(hypotheses)

        # Stage 2: Evidence verification (or skip)
        if self.skip_evidence:
            confirmed = hypotheses
        else:
            confirmed = []
            for h in hypotheses:
                try:
                    ev_prompt = self._build_evidence_prompt(
                        sample.content,
                        h.get("description", ""),
                        h.get("value_id", ""),
                    )
                    ev_response = self.llm_client.send_request([
                        {"role": "system", "content": TEXT_EVIDENCE_SYSTEM},
                        {"role": "user", "content": ev_prompt},
                    ])
                    ev_parsed = self._parse_evidence(ev_response)
                    status = ev_parsed.get("status", "UNVERIFIED").upper()
                    if status == "CONFIRMED":
                        confirmed.append(h)
                except Exception as e:
                    logger.warning(f"Text evidence verification failed: {e}")

        result.confirmed_count = len(confirmed)

        # 最终预测：只有当最高置信度超过阈值时才预测有风险
        max_conf = max((float(h.get("confidence", 0)) for h in confirmed), default=0.0)
        result.predicted_has_risk = max_conf >= self.text_confidence_threshold
        result.predicted_values = list(set(
            h.get("value_id", "") for h in confirmed
            if h.get("value_id") and float(h.get("confidence", 0)) >= self.text_confidence_threshold
        ))

        # 构建置信度向量
        for h in confirmed:
            vid = h.get("value_id", "")
            conf = float(h.get("confidence", 0))
            if vid:
                result.predicted_confidences[vid] = max(
                    result.predicted_confidences.get(vid, 0), conf
                )

        result.total_time_ms = (time.time() - start_time) * 1000

        # 缓存
        self._save_to_cache(result)
        return result

    def evaluate_all(self, samples: list[BenchmarkSample]) -> list[UnifiedSampleResult]:
        """评估所有 text 样本"""
        results = []
        for i, sample in enumerate(samples):
            logger.info(f"  [TEXT {i+1}/{len(samples)}] {sample.sample_id}")
            results.append(self.evaluate_sample(sample))
        return results


# ============================================================
# 统一 Pipeline 评估器
# ============================================================

class UnifiedPipelineEvaluator:
    """统一 Pipeline 评估器：自动路由 code/text 场景"""

    def __init__(
        self,
        llm_provider: str = "qwen-plus",
        profile_map: Optional[dict[str, ValueProfile]] = None,
        skip_evidence: bool = False,
        profile_mode: str = "real",  # "real" or "uniform"
        mock_mode: bool = False,
        repo_path: str = ".",
        llm_configs: Optional[dict] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        profile_alpha: float = 1.0,
        search_depth: int = 3,
        top_k: int = 10,
        profile_threshold_mode: str = "rank",
        profile_prompt_injection: bool = True,
        text_confidence_threshold: float = 0.5,
    ):
        self.llm_provider = llm_provider
        self.mock_mode = mock_mode
        self.repo_path = repo_path
        self.skip_evidence = skip_evidence
        self.profile_mode = profile_mode
        self.profile_alpha = float(profile_alpha)
        self.search_depth = int(search_depth)
        self.top_k = int(top_k)
        self.profile_threshold_mode = profile_threshold_mode
        self.profile_prompt_injection = bool(profile_prompt_injection)
        self.text_confidence_threshold = float(text_confidence_threshold)

        # Profile 配置
        if profile_mode == "uniform" or not profile_map:
            self.profile_map = {}
        else:
            self.profile_map = profile_map

        # 缓存子目录后缀（区分消融变体）
        cache_suffix = ""
        if profile_mode == "uniform":
            cache_suffix = "_wo_profile"
        if skip_evidence:
            cache_suffix += "_wo_evidence"
        if self.profile_alpha != 1.0:
            cache_suffix += f"_alpha{self.profile_alpha}"
        if self.search_depth != 3:
            cache_suffix += f"_sd{self.search_depth}"
        if self.top_k != 10:
            cache_suffix += f"_tk{self.top_k}"
        if self.profile_threshold_mode != "rank":
            cache_suffix += f"_tm{self.profile_threshold_mode}"
        if not self.profile_prompt_injection:
            cache_suffix += "_noinject"
        if self.text_confidence_threshold != 0.5:
            cache_suffix += f"_tct{self.text_confidence_threshold}"

        # Code pipeline evaluator
        from experiment.pipeline_evaluator import PipelineEvaluator
        code_cache_dir = cache_dir or str(exp_paths.PIPELINE_CACHE_DIR)
        self.code_evaluator = PipelineEvaluator(
            repo_path=repo_path,
            llm_provider=llm_provider,
            mock_mode=mock_mode,
            profile_map=self.profile_map if profile_mode == "real" else None,
            skip_evidence=skip_evidence,
            cache_dir=code_cache_dir + cache_suffix if cache_suffix else None,
            use_cache=use_cache,
            profile_alpha=self.profile_alpha,
            search_depth=self.search_depth,
            top_k=self.top_k,
            profile_threshold_mode=self.profile_threshold_mode,
            profile_prompt_injection=self.profile_prompt_injection,
        )

        # Text Pipeline 评估器
        if not mock_mode and llm_configs:
            llm_conf = llm_configs.get(llm_provider, {})
            if not llm_conf:
                raise ValueError(
                    f"LLM provider '{llm_provider}' not found in llm_configs. "
                    f"Available providers: {list(llm_configs.keys())}"
                )
            if not llm_conf.get("api_key_env"):
                raise ValueError(
                    f"LLM provider '{llm_provider}' missing 'api_key_env' in config. "
                    f"Config: {llm_conf}"
                )
            logger.info(f"Creating text_evaluator for {llm_provider}, config: {llm_conf}")
            llm_client = LLMClientFactory.create(llm_conf)
            text_cache_dir = cache_dir or str(exp_paths.TEXT_PIPELINE_CACHE_DIR)
            self.text_evaluator = TextPipelineEvaluator(
                llm_client=llm_client,
                profile_map=self.profile_map if profile_mode == "real" else None,
                skip_evidence=skip_evidence,
                cache_dir=text_cache_dir + cache_suffix if cache_suffix else None,
                use_cache=use_cache,
                profile_alpha=self.profile_alpha,
                text_confidence_threshold=self.text_confidence_threshold,
            )
            logger.info(f"text_evaluator created successfully")
        else:
            logger.warning(f"text_evaluator NOT created: mock_mode={mock_mode}, llm_configs={bool(llm_configs)}")
            self.text_evaluator = None

    def evaluate_all(
        self,
        samples: list[BenchmarkSample],
        max_samples: Optional[int] = None,
        parallel_workers: int = 1,
    ) -> list[UnifiedSampleResult]:
        """评估所有样本，自动路由 code/text"""
        if max_samples:
            samples = samples[:max_samples]

        results = []
        code_samples = [s for s in samples if s.scenario_type == "code" and s.supports_pipeline]
        text_samples = [s for s in samples if s.scenario_type == "text"]
        other_code = [s for s in samples if s.scenario_type == "code" and not s.supports_pipeline]

        logger.info(f"UnifiedPipeline: {len(code_samples)} code-pipeline + "
                     f"{len(other_code)} code-text-only + {len(text_samples)} text")

        # Code 场景（通过 PipelineEvaluator）
        if code_samples:
            from experiment.commit_diff_extractor import CommitSample, CommitInfo, DiffHunkExtracted
            commit_samples = self._convert_to_commit_samples(code_samples)

            logger.info(f"  Running code pipeline on {len(commit_samples)} samples...")
            pipeline_results = self.code_evaluator.evaluate_all(commit_samples, parallel_workers=parallel_workers)

            for pr, bs in zip(pipeline_results.sample_results, code_samples):
                results.append(UnifiedSampleResult(
                    sample_id=bs.sample_id,
                    scenario_type="code",
                    repo=bs.repo,
                    predicted_has_risk=pr.predicted_has_risk,
                    predicted_values=pr.predicted_values,
                    predicted_confidences={},  # Pipeline 模式暂不提供 per-value confidence
                    ground_truth_has_risk=bs.has_value_risk,
                    ground_truth_values=bs.ground_truth_values,
                    hypothesis_count=len(pr.hypotheses),
                    confirmed_count=len(pr.confirmed_hypotheses),
                    profile_used="real" if self.profile_map else "uniform",
                    total_time_ms=pr.hypothesis_time_ms + pr.evidence_time_ms,
                ))

        # Text 场景 + 不支持 pipeline 的 code 场景
        text_and_other = text_samples + other_code
        if text_and_other:
            if self.text_evaluator and not self.mock_mode:
                logger.info(f"  Running text pipeline on {len(text_and_other)} samples...")
                text_results = self.text_evaluator.evaluate_all(text_and_other)
                results.extend(text_results)
            elif self.mock_mode:
                logger.info(f"  Running text pipeline (MOCK) on {len(text_and_other)} samples...")
                for s in text_and_other:
                    results.append(self._mock_text_result(s))

        return results

    def _convert_to_commit_samples(self, samples: list[BenchmarkSample]) -> list:
        """将 BenchmarkSample（含 diff_hunks_data）转换为 CommitSample"""
        from experiment.commit_diff_extractor import CommitSample, CommitInfo, DiffHunkExtracted

        commit_samples = []
        for bs in samples:
            if not bs.diff_hunks_data:
                continue

            commit_info = CommitInfo(
                sha=bs.commit_sha or bs.sample_id,
                short_sha=(bs.commit_sha or bs.sample_id)[:8],
                message=bs.commit_message or "",
                author="",
                date="",
            )

            hunks = []
            for h in bs.diff_hunks_data:
                hunks.append(DiffHunkExtracted(
                    file_path=h.get("file_path", "unknown"),
                    old_start=h.get("old_start", 0),
                    old_lines=h.get("old_lines", 0),
                    new_start=h.get("new_start", 0),
                    new_lines=h.get("new_lines", 0),
                    content=h.get("content", ""),
                    change_type=h.get("change_type", "modified"),
                ))

            cs = CommitSample(
                commit=commit_info,
                diff_hunks=hunks,
                ground_truth_has_risk=bs.has_value_risk,
                ground_truth_values=bs.ground_truth_values,
            )
            # 注入 repo 名称，供 profile 匹配
            cs._repo_name = bs.repo
            commit_samples.append(cs)

        return commit_samples

    def _mock_text_result(self, sample: BenchmarkSample) -> UnifiedSampleResult:
        """Mock 模式下的 text 评估结果"""
        import random
        has_risk = sample.has_value_risk and random.random() < 0.7
        values = sample.ground_truth_values[:1] if has_risk else []

        return UnifiedSampleResult(
            sample_id=sample.sample_id,
            scenario_type=sample.scenario_type,
            repo=sample.repo,
            predicted_has_risk=has_risk,
            predicted_values=values,
            predicted_confidences={v: 0.7 for v in values},
            ground_truth_has_risk=sample.has_value_risk,
            ground_truth_values=sample.ground_truth_values,
            hypothesis_count=len(values) + random.randint(0, 2),
            confirmed_count=len(values),
            profile_used="mock",
            total_time_ms=random.uniform(100, 500),
        )


# ============================================================
# 统一指标计算
# ============================================================

def compute_unified_metrics(
    results: list[UnifiedSampleResult],
) -> dict:
    """计算统一评估指标（按 code/text/overall 三组）"""
    metrics = {}
    for label, filter_fn in [
        ("code", lambda r: r.scenario_type == "code"),
        ("text", lambda r: r.scenario_type == "text"),
        ("overall", lambda r: True),
    ]:
        subset = [r for r in results if filter_fn(r)]
        if not subset:
            continue

        gt_risks = [r.ground_truth_has_risk for r in subset]
        pred_risks = [r.predicted_has_risk for r in subset]
        gt_values = [set(r.ground_truth_values) for r in subset]
        pred_values = [set(r.predicted_values) for r in subset]

        # Dim1: Risk Detection
        dim1 = precision_recall_f1_binary(gt_risks, pred_risks)
        dim1["cohen_kappa"] = round(cohen_kappa_binary(gt_risks, pred_risks), 4)
        dim1["pabak"] = round(pabak(gt_risks, pred_risks), 4)
        dim1["gwet_ac1"] = round(gwet_ac1(gt_risks, pred_risks), 4)

        # Dim2: Value Identification
        dim2 = micro_precision_recall_f1(gt_values, pred_values)
        # 为 Table 4 提供统一的 precision/recall/f1 别名（micro-level）
        dim2["precision"] = dim2.get("micro_precision", 0.0)
        dim2["recall"]    = dim2.get("micro_recall", 0.0)
        dim2["f1"]        = dim2.get("micro_f1", 0.0)
        dim2["pairwise_jaccard"] = round(pairwise_jaccard(gt_values, pred_values), 4)
        dim2["symmetric_f1"] = round(pairwise_symmetric_f1(gt_values, pred_values), 4)
        # 方案一新增指标：针对稀疏GT标注更公平的评估
        dim2["recall_of_gt"] = round(recall_of_gt(gt_values, pred_values), 4)
        dim2["precision_of_gt"] = round(precision_of_gt(gt_values, pred_values), 4)
        dim2["partial_credit_jaccard"] = round(partial_credit_jaccard(gt_values, pred_values), 4)

        # Dim2 per-value precision/recall/F1
        all_value_ids = set()
        for s in gt_values:
            all_value_ids.update(s)
        for s in pred_values:
            all_value_ids.update(s)

        per_value = {}
        for vid in sorted(all_value_ids):
            tp = sum(1 for g, p in zip(gt_values, pred_values) if vid in g and vid in p)
            fp = sum(1 for g, p in zip(gt_values, pred_values) if vid not in g and vid in p)
            fn = sum(1 for g, p in zip(gt_values, pred_values) if vid in g and vid not in p)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            support = sum(1 for g in gt_values if vid in g)
            per_value[vid] = {
                "precision": round(prec, 4),
                "recall":    round(rec,  4),
                "f1":        round(f1,   4),
                "support":   support,
                "tp": tp, "fp": fp, "fn": fn,
            }
        dim2["per_value"] = per_value

        # Pipeline stats
        total_hypotheses = sum(r.hypothesis_count for r in subset)
        total_confirmed = sum(r.confirmed_count for r in subset)
        confirmation_rate = total_confirmed / total_hypotheses if total_hypotheses > 0 else 0.0
        profile_match_rate = sum(1 for r in subset if r.profile_used == "real") / len(subset)

        metrics[label] = {
            "n": len(subset),
            "dim1_risk_detection": dim1,
            "dim2_value_identification": dim2,
            "pipeline_stats": {
                "total_hypotheses": total_hypotheses,
                "total_confirmed": total_confirmed,
                "confirmation_rate": round(confirmation_rate, 4),
                "avg_hypotheses_per_sample": round(total_hypotheses / len(subset), 2),
                "profile_match_rate": round(profile_match_rate, 4),
            },
        }

    return metrics