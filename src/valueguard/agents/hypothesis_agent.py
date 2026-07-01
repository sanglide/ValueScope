"""Hypothesis agent for generating value deviation hypotheses."""

import logging
from typing import Any, Optional
import uuid

logger = logging.getLogger(__name__)

from valueguard.agents.base_agent import BaseAgent
from valueguard.core.models import (
    CrossLayerTrace,
    DiffHunk,
    HypothesisTask,
    ValueHypothesis,
    ValuePattern,
    ValueProfile,
)
from valueguard.memory.manager import MemoryManager
from valueguard.skills.registry import SkillRegistry


# System prompt for cross-layer value reasoning
CROSS_LAYER_REASONING_SYSTEM = """You are a value deviation analyst for software code changes.
Your task is to analyze code changes and identify potential value deviations using
four-layer cross-layer reasoning:

L1 (Schwartz Values) -> L2 (Human Value Themes) -> L3 (System Value Themes) -> L4 (Code Indicators)

## Analysis Process:
1. Examine the code change (diff hunk)
2. Identify any L4 code indicators that might relate to values
3. Trace upward through L3 -> L2 -> L1 to understand the value semantics
4. Determine if there's a deviation from the project's declared value profile
5. Assess confidence and severity

## Output Format (JSON):
{
  "hypotheses": [
    {
      "value_id": "HV9",  // L2 or L3 value ID
      "deviation_type": "violation|inconsistency|risk",
      "confidence": 0.0-1.0,
      "severity": "low|medium|high|critical",
      "description": "Brief description of the deviation",
      "suggested_action": "What should be reviewed or changed",
      "cross_layer_trace": {
        "l1_value": "Security",
        "l2_theme": "HV9 Privacy",
        "l3_attribute": "SV10 Security",
        "l4_indicator": "unencrypted data transmission",
        "reasoning": "How this code relates to the value chain"
      }
    }
  ]
}

## Deviation Types:
- violation: Direct contradiction of a stated value
- inconsistency: Behavior that partially conflicts with values
- risk: Potential future violation if not addressed

Only include hypotheses with confidence >= 0.5.
"""


class HypothesisAgent(BaseAgent):
    """Agent for generating value deviation hypotheses.

    Uses four-layer cross-layer reasoning to trace value semantics
    from code changes (L4) up to normative values (L1).
    """

    name = "hypothesis"
    role = "Cross-Layer Value Reasoner"
    goal = "Generate structured value hypotheses using four-layer reasoning"

    def __init__(
        self,
        skills: SkillRegistry,
        memory: MemoryManager,
        config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(skills, memory, config)
        self._llm_provider = self.get_config("llm_provider", "deepseek")
        self._max_hypotheses = self.get_config("max_hypotheses", 10)
        # profile_alpha 控制 repo value profile 对假说排序的加权强度
        # 0.0 = 无 profile 影响（类似 uniform prior）
        # 1.0 = 默认强度（score *= 1 + profile_weight）
        self._profile_alpha = float(self.get_config("profile_alpha", 1.0))
        # profile 使用模式："rank"（原始乘法排序）或 "threshold"（双向阈值约束）
        # threshold 模式：高 profile value → 降低报告阈值（更敏感）
        #                 低 profile value → 提高报告阈值（更严格）
        self._profile_threshold_mode = self.get_config("profile_threshold_mode", "rank")
        # 阈值敏感度：控制 profile 对阈值的影响幅度
        self._threshold_sensitivity = float(self.get_config("threshold_sensitivity", 0.4))
        # 基础置信度阈值
        self._base_threshold = 0.5
        # 是否在 prompt 中注入 profile（P2: 后验过滤模式）
        # True: 将 profile 信息注入 prompt，引导 LLM 关注高 profile value
        # False: 不注入 profile，LLM 客观分析代码，profile 仅用于后验过滤/排序
        self._profile_prompt_injection = self.get_config("profile_prompt_injection", True)

    def execute(self, task: HypothesisTask) -> list[ValueHypothesis]:
        """Generate value hypotheses for code changes.

        Args:
            task: HypothesisTask with diff hunks and profile context

        Returns:
            List of ValueHypothesis objects
        """
        all_hypotheses = []

        for hunk in task.diff_hunks:
            # Generate hypotheses for each hunk
            hypotheses = self._analyze_hunk(hunk, task.profile, task.memory_context)
            all_hypotheses.extend(hypotheses)

        # Rank and limit hypotheses
        ranked = self._rank_hypotheses(all_hypotheses, task.profile)
        return ranked[: task.max_hypotheses]

    def _analyze_hunk(
        self,
        hunk: DiffHunk,
        profile: ValueProfile,
        memory_patterns: list[ValuePattern],
    ) -> list[ValueHypothesis]:
        """Analyze a single diff hunk for value deviations."""
        if not self.has_skill("llm_call"):
            logger.warning("⚠️  LLM call skill not available, skipping hypothesis generation")
            return []

        # Build context-aware prompt
        prompt = self._build_analysis_prompt(hunk, profile, memory_patterns)

        try:
            logger.info(f"→ Calling LLM provider {self._llm_provider} for file: {hunk.file_path}")
            response = self.invoke_skill(
                "llm_call",
                system=CROSS_LAYER_REASONING_SYSTEM,
                user=prompt,
                provider=self._llm_provider,
                parse_json=True,
            )

            if response.error:
                logger.warning(f"  [HYPOTHESIS] LLM returned error: {response.error}")
                return []

            raw_text = response.raw_response or ""

            # 检测内容安全拒绝（自然语言回复，非JSON）
            REFUSAL_PATTERNS = ["i can't discuss", "i cannot discuss", "i'm unable to", "i can't help"]
            if any(p in raw_text.lower() for p in REFUSAL_PATTERNS):
                logger.warning(f"  [HYPOTHESIS] LLM refused to analyze (safety filter): {raw_text[:100]!r}")
                return []

            logger.debug(f"  [HYPOTHESIS] raw_response (first 500 chars): {raw_text[:500]!r}")
            logger.debug(f"  [HYPOTHESIS] parsed_result: {response.parsed_result}")

            parsed = response.parsed_result
            if not parsed:
                logger.warning(f"  [HYPOTHESIS] parsed_result is None/empty, trying json_parser fallback")
                if self.has_skill("json_parser"):
                    parsed = self.invoke_skill(
                        "json_parser",
                        response=response.raw_response,
                        strict=False,
                        default={},
                    )
                else:
                    logger.warning(f"  [HYPOTHESIS] json_parser skill not available")

            if parsed and "hypotheses" in parsed:
                raw_list = parsed["hypotheses"]
                logger.info(f"  [HYPOTHESIS] Found {len(raw_list)} raw hypotheses before confidence filter")
                if len(raw_list) == 0:
                    logger.debug(f"  [HYPOTHESIS] LLM returned empty hypotheses list. Full parsed: {parsed}")
                result = self._parse_hypotheses(raw_list, hunk, profile)
                logger.info(f"  [HYPOTHESIS] After confidence>=0.5 filter: {len(result)} hypotheses")
                return result
            else:
                logger.warning(f"  [HYPOTHESIS] parsed={parsed!r} — 'hypotheses' key missing or parsed is None")

        except Exception as e:
            logger.error(f"_analyze_hunk failed for {hunk.file_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())

        return []

    def _build_analysis_prompt(
        self,
        hunk: DiffHunk,
        profile: ValueProfile,
        memory_patterns: list[ValuePattern],
    ) -> str:
        """Build the analysis prompt with context."""
        parts = []

        # Project profile context（仅当 profile_prompt_injection=True 时注入）
        if self._profile_prompt_injection:
            parts.append("## Project Value Profile")
            if profile.core_values:
                parts.append(f"Core values: {', '.join(profile.core_values)}")

            if profile.l2_scores:
                top_l2 = sorted(
                    profile.l2_scores.items(), key=lambda x: x[1], reverse=True
                )[:5]
                parts.append(
                    "Top L2 values: " + ", ".join(f"{k}({v:.2f})" for k, v in top_l2)
                )

            if profile.l3_scores:
                top_l3 = sorted(
                    profile.l3_scores.items(), key=lambda x: x[1], reverse=True
                )[:5]
                parts.append(
                    "Top L3 values: " + ", ".join(f"{k}({v:.2f})" for k, v in top_l3)
                )
        else:
            # 后验过滤模式：不注入 profile，让 LLM 客观分析
            logger.debug("Profile prompt injection disabled — objective analysis mode")

        # Memory patterns context
        if memory_patterns:
            parts.append("\n## Known Patterns (from previous analyses)")
            for pattern in memory_patterns[:5]:
                parts.append(
                    f"- {pattern.value_id}: {pattern.code_pattern} "
                    f"(confidence: {pattern.confidence:.2f})"
                )

        # Code change
        parts.append("\n## Code Change to Analyze")
        parts.append(f"File: {hunk.file_path}")
        parts.append(f"Language: {hunk.language}")
        parts.append(f"Change type: {hunk.change_type}")
        parts.append(f"Lines: {hunk.new_start}-{hunk.new_start + hunk.new_lines}")
        parts.append("\n```")
        parts.append(hunk.content)
        parts.append("```")

        parts.append("\nAnalyze this code change for potential value deviations.")
        parts.append(
            "\n**IMPORTANT: You MUST respond with ONLY a valid JSON object. No explanations, no markdown, no natural language.**"
            "\nBe generous in your analysis — even minor or potential deviations should be reported with appropriate confidence (0.5-0.7)."
            "\nIf truly no deviations are found, respond with: {\"hypotheses\": []}"
            "\nFormat:\n{\"hypotheses\": [{\"value_id\": \"...\", \"deviation_type\": \"...\", \"confidence\": 0.0, \"severity\": \"...\", \"description\": \"...\", \"suggested_action\": \"...\", \"cross_layer_trace\": {}}]}"
        )

        return "\n".join(parts)

    def _parse_hypotheses(
        self,
        raw_hypotheses: list[dict[str, Any]],
        hunk: DiffHunk,
        profile: ValueProfile,
    ) -> list[ValueHypothesis]:
        """Parse raw hypothesis data into ValueHypothesis objects."""
        hypotheses = []

        for raw in raw_hypotheses:
            # Skip low confidence
            confidence = float(raw.get("confidence", 0.0))

            # 根据的模式计算动态阈值
            value_id = raw.get("value_id", "")
            profile_weight = max(
                profile.l2_scores.get(value_id, 0.0),
                profile.l3_scores.get(value_id, 0.0),
            )

            if self._profile_threshold_mode == "threshold":
                # 双向阈值约束：
                # high profile → lower threshold（更敏感，低置信度也能报告）
                # low profile → higher threshold（更严格，需要高置信度才报告）
                threshold = self._base_threshold + (0.5 - profile_weight) * self._threshold_sensitivity
                threshold = max(0.1, min(0.9, threshold))  # clamp to [0.1, 0.9]
            else:
                # 原始模式：固定阈值
                threshold = self._base_threshold

            if confidence < threshold:
                continue

            # Parse cross-layer trace
            trace_data = raw.get("cross_layer_trace", {})
            cross_layer_trace = None
            if trace_data:
                cross_layer_trace = CrossLayerTrace(
                    l1_value=trace_data.get("l1_value", ""),
                    l2_theme=trace_data.get("l2_theme", ""),
                    l3_attribute=trace_data.get("l3_attribute", ""),
                    l4_indicator=trace_data.get("l4_indicator", ""),
                    reasoning=trace_data.get("reasoning", ""),
                )

            # profile_weight 已在阈值计算阶段获得，复用

            hypothesis = ValueHypothesis(
                id=str(uuid.uuid4())[:8],
                value_id=value_id,
                deviation_type=raw.get("deviation_type", "risk"),
                confidence=confidence,
                severity=raw.get("severity", "medium"),
                cross_layer_trace=cross_layer_trace,
                diff_hunk=hunk,
                description=raw.get("description", ""),
                suggested_action=raw.get("suggested_action", ""),
                profile_weight=profile_weight,
            )
            hypotheses.append(hypothesis)

        return hypotheses

    def _rank_hypotheses(
        self, hypotheses: list[ValueHypothesis], profile: ValueProfile
    ) -> list[ValueHypothesis]:
        """Rank hypotheses by relevance to project profile."""
        # Score each hypothesis
        scored = []
        for h in hypotheses:
            score = h.confidence

            # Boost by profile weight (values important to the project)
            # profile_alpha 控制强度；0.0 时完全忽略 profile
            if h.profile_weight > 0 and self._profile_alpha != 0.0:
                score *= 1.0 + self._profile_alpha * h.profile_weight

            # Boost by severity
            severity_boost = {
                "critical": 0.4,
                "high": 0.2,
                "medium": 0.0,
                "low": -0.1,
            }
            score += severity_boost.get(h.severity, 0.0)

            scored.append((score, h))

        # Sort by score (descending)
        scored.sort(key=lambda x: x[0], reverse=True)

        return [h for _, h in scored]
