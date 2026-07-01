#!/usr/bin/env python
"""
Pipeline evaluator: run Hypothesis Agent + Evidence Agent and compute metrics.
Compares pipeline output against zero-shot LLM evaluation.
"""

import sys
import json
import time
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

# Project paths
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

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
    from . import paths as exp_paths
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
    import paths as exp_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# 评估结果数据结构
# ============================================================

@dataclass
class HypothesisStats:
    """假说生成统计"""
    total_generated: int = 0
    avg_confidence: float = 0.0
    confidence_distribution: dict = field(default_factory=dict)  # 按区间统计
    value_distribution: dict = field(default_factory=dict)  # 按 value_id 统计
    severity_distribution: dict = field(default_factory=dict)  # 按 severity 统计
    deviation_type_distribution: dict = field(default_factory=dict)


@dataclass
class EvidenceStats:
    """证据定位统计"""
    total_hypotheses: int = 0
    confirmed_count: int = 0
    unverified_count: int = 0
    rejected_count: int = 0
    avg_relevance_score: float = 0.0
    confirmation_rate: float = 0.0


@dataclass
class SampleResult:
    """单个样本的评估结果"""
    sample_id: str
    commit_sha: str
    commit_message: str

    # Pipeline 输出
    hypotheses: list[ValueHypothesis] = field(default_factory=list)
    evidence_results: list[EvidenceResult] = field(default_factory=list)

    # 筛选后的结果（仅 CONFIRMED）
    confirmed_hypotheses: list[ValueHypothesis] = field(default_factory=list)

    # Ground truth
    ground_truth_has_risk: Optional[bool] = None
    ground_truth_values: list[str] = field(default_factory=list)

    # 预测结果
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

    # 计时
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
    """Pipeline 整体评估结果"""
    total_samples: int = 0
    total_time_seconds: float = 0.0

    # 假说统计
    hypothesis_stats: HypothesisStats = field(default_factory=HypothesisStats)

    # 证据统计
    evidence_stats: EvidenceStats = field(default_factory=EvidenceStats)

    # IAA 一致性指标（Human vs Pipeline，筛选前）
    # 维度1: 风险检测 (二分类)
    before_filter_percent_agreement: float = 0.0
    before_filter_cohen_kappa: float = 0.0
    before_filter_pabak: float = 0.0
    before_filter_gwet_ac1: float = 0.0

    # 维度2: 价值识别 (多标签)
    before_filter_jaccard: float = 0.0
    before_filter_symmetric_f1: float = 0.0

    # IAA 一致性指标（Human vs Pipeline，筛选后）
    # 维度1: 风险检测 (二分类)
    after_filter_percent_agreement: float = 0.0
    after_filter_cohen_kappa: float = 0.0
    after_filter_pabak: float = 0.0
    after_filter_gwet_ac1: float = 0.0

    # 维度2: 价值识别 (多标签)
    after_filter_jaccard: float = 0.0
    after_filter_symmetric_f1: float = 0.0

    # 详细结果
    sample_results: list[SampleResult] = field(default_factory=list)


# ============================================================
# 缓存辅助函数
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
# Pipeline 评估器
# ============================================================

class PipelineEvaluator:
    """Hypothesis Generator + Evidence Location Agent Pipeline 评估器"""

    def __init__(
        self,
        repo_path: str,
        llm_provider: str = "deepseek",
        storage_path: str = ".valueguard/memory",
        mock_mode: bool = False,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        profile_map: Optional[dict[str, "ValueProfile"]] = None,
        skip_evidence: bool = False,
        profile_alpha: float = 1.0,
        search_depth: int = 3,
        top_k: int = 10,
        profile_threshold_mode: str = "rank",
        profile_prompt_injection: bool = True,
    ):
        self.repo_path = Path(repo_path)
        self.llm_provider = llm_provider
        self.mock_mode = mock_mode
        self.skip_evidence = skip_evidence
        self.profile_alpha = float(profile_alpha)
        self.search_depth = int(search_depth)
        self.top_k = int(top_k)
        self.profile_threshold_mode = profile_threshold_mode
        self.profile_prompt_injection = profile_prompt_injection

        # ---------- Profile 配置 ----------
        # profile_map: {repo_name: ValueProfile}，用于注入真实 profile
        # 若为 None，则所有样本使用 uniform prior
        self.profile_map = profile_map or {}
        if profile_map:
            logger.info(f"Profile map loaded: {list(profile_map.keys())}")
        # --------------------------------

        # ---------- Cache configuration ----------
        # Default: experiment_outputs/cache/llm_outputs/pipeline_cache/<provider>/
        # Ablation parameters are isolated to avoid stale-cache reuse.
        if use_cache:
            _base = Path(cache_dir) if cache_dir else exp_paths.PIPELINE_CACHE_DIR
            suffix_parts = [llm_provider]
            if self.profile_alpha != 1.0:
                suffix_parts.append(f"alpha{self.profile_alpha}")
            if self.search_depth != 3:
                suffix_parts.append(f"sd{self.search_depth}")
            if self.top_k != 10:
                suffix_parts.append(f"tk{self.top_k}")
            if self.profile_threshold_mode != "rank":
                suffix_parts.append(f"ptm{self.profile_threshold_mode}")
            if not self.profile_prompt_injection:
                suffix_parts.append("noprompt")
            cache_subdir = "_".join(suffix_parts)
            self._cache_dir = _base / cache_subdir
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache dir: {self._cache_dir}")
        else:
            self._cache_dir = None
        # --------------------------------

        if not mock_mode:
            # 初始化 skills
            self.skills = SkillRegistry()
            self._setup_skills()

            # 初始化 memory
            self.memory = MemoryManager(storage_path=storage_path)

            # 初始化 agents
            self.hypothesis_agent = HypothesisAgent(
                self.skills, self.memory,
                config={"llm_provider": llm_provider,
                        "profile_alpha": self.profile_alpha,
                        "profile_threshold_mode": self.profile_threshold_mode,
                        "profile_prompt_injection": self.profile_prompt_injection}
            )
            self.evidence_agent = EvidenceAgent(
                self.skills, self.memory,
                config={"search_depth": self.search_depth,
                        "top_k": self.top_k,
                        "llm_provider": llm_provider}
            )
        else:
            logger.info("运行在 MOCK 模式 - 使用模拟假说生成")
            self.skills = None
            self.memory = None
            self.hypothesis_agent = None
            self.evidence_agent = None

        # 创建 uniform prior profile（当无 profile_map 匹配时回退使用）
        self.default_profile = ValueProfile(
            repo=str(self.repo_path),
            l2_scores={f"HV{i}": 0.5 for i in range(1, 11)},
            l3_scores={f"SV{i}": 0.5 for i in range(1, 11)},
        )

    def _cache_path(self, commit_sha: str) -> Optional[Path]:
        """返回给定 commit 的缓存文件路径，如果未启用缓存则返回 None。"""
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{commit_sha}.json"

    def _load_from_cache(self, commit_sha: str) -> Optional["SampleResult"]:
        """从缓存加载 SampleResult，缓存不存在则返回 None。"""
        path = self._cache_path(commit_sha)
        if path is None or not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            confirmed_count = data.get("confirmed_count", 0)
            hypotheses_count = data.get("hypotheses_count", 0)

            # 重建 evidence_results：用于 _compute_statistics 统计 confirmation_rate
            # confirmed_count 个 CONFIRMED，其余为 UNVERIFIED
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
                # 用占位 ValueHypothesis 列表重建 predicted 信息
                hypotheses=[
                    _stub_hypothesis(v) for v in data.get("predicted_values_before_filter", [])
                ],
                confirmed_hypotheses=[
                    _stub_hypothesis(v) for v in data.get("predicted_values", [])
                ],
                evidence_results=evidence_results,
            )
        except Exception as e:
            logger.warning(f"缓存读取失败 {commit_sha}: {e}")
            return None

    def _save_to_cache(self, result: "SampleResult") -> None:
        """将 SampleResult 写入缓存文件，仅当样本被成功处理时才缓存。"""
        path = self._cache_path(result.commit_sha)
        if path is None:
            return

        # ---- 成功校验逻辑 ----
        # 如果假说生成或证据验证完全失败（没有 hypotheses），说明调用出错，不缓存
        # 这样可以保证下次重试时还会重新调用 API
        has_successful_processing = (
            len(result.hypotheses) > 0 or  # 生成了假说（即使最后都被 filter 掉）
            len(result.confirmed_hypotheses) > 0 or  # 或有确认的假说
            result.hypothesis_time_ms > 100  # 或假说生成耗时>100ms（说明真的调用了 LLM）
        )

        # 额外检查：如果假说数量为 0 且 ground_truth 显示有风险，可能是 LLM 调用失败
        # 这种情况下也不缓存，允许重试
        if (
            len(result.hypotheses) == 0 and 
            result.ground_truth_has_risk and 
            result.hypothesis_time_ms < 50  # 极短时间说明可能没真正调用
        ):
            logger.debug(f"跳过缓存 {result.commit_sha} - 风险样本无输出，可能调用失败")
            return

        if not has_successful_processing:
            logger.debug(f"跳过缓存 {result.commit_sha} - 样本处理失败或无输出")
            return
        # ----------------------

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"缓存写入失败 {result.commit_sha}: {e}")

    def _setup_skills(self):
        """设置必要的 skills"""
        # 注册 LLM call skill
        llm_skill = LLMCallSkill()
        self.skills.register(llm_skill)
        logger.info(f"✓ Registered LLMCallSkill for {self.llm_provider}")

        # 可选：注册其他 skills（vector_search, ast_analysis 等）
        # 这些在完整部署时需要配置

    def _resolve_profile(self, sample: CommitSample) -> ValueProfile:
        """根据样本的 repo 信息匹配 profile_map 中的真实 profile。

        匹配逻辑：
        1. 从 CommitSample 的 _repo_path_override 或 commit 信息提取 repo 名称
        2. 在 profile_map 中查找（大小写不敏感、忽略连字符）
        3. 找不到则回退 uniform prior
        """
        # 尝试从多个来源提取 repo 名称
        repo_name = ""
        if hasattr(sample, "_repo_path_override") and sample._repo_path_override:
            repo_name = Path(sample._repo_path_override).name
        if not repo_name:
            # 从 commit message 或 sample_id 中尝试推断
            repo_name = getattr(sample, "_repo_name", "")

        if repo_name and self.profile_map:
            # 标准化比较
            normalized = repo_name.lower().replace("-", "").replace("_", "").replace(" ", "")
            for key, profile in self.profile_map.items():
                key_norm = key.lower().replace("-", "").replace("_", "").replace(" ", "")
                if key_norm == normalized or normalized in key_norm or key_norm in normalized:
                    logger.debug(f"  Profile matched: {repo_name} -> {key}")
                    return profile

        # 回退到 uniform prior
        return self.default_profile

    def _convert_to_diff_hunks(self, sample: CommitSample) -> list[DiffHunk]:
        """将提取的 diff hunks 转换为 ValueGuard 格式"""
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
        """Mock 模式：基于关键词生成模拟假说
        
        模拟 LLM 的行为特点：
        1. 对有风险的样本倾向于生成假说（高召回）
        2. 对无风险样本也可能产生误报（较低精确率）
        """
        import random
        import uuid
        
        hypotheses = []
        msg_lower = sample.commit.message.lower()
        
        # 根据 commit message 中的关键词生成假说（与 ground truth 对齐）
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
        
        # 如果 ground truth 有风险，大概率生成对应的假说（模拟高召回）
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
        
        # 对所有样本，LLM 有一定概率产生误报（模拟过度敏感）
        # 这符合 0-shot 评估中观察到的 LLM 倾向于判定有风险的行为
        if random.random() < 0.4:  # 40% 概率产生额外假说
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
        """Mock 模式：模拟证据验证"""
        import random
        from valueguard.core.models import EvidencePiece
        
        # 高置信度假说更可能被确认
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
        """评估单个样本"""
        # ---- 缓存命中检查 ----
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

        # 转换 diff hunks
        diff_hunks = self._convert_to_diff_hunks(sample)

        if not diff_hunks:
            return result

        # 解析 profile（使用真实 profile 或 uniform prior）
        resolved_profile = self._resolve_profile(sample)
        profile_source = "real" if resolved_profile is not self.default_profile else "uniform"
        logger.info(f"  [PROFILE] Using {profile_source} profile for {sample.commit.short_sha}")

        # Stage 1: 生成假说
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
                    profile=resolved_profile,  # 使用解析后的 profile
                    memory_context=[],
                    max_hypotheses=10,
                )
                hypotheses = self.hypothesis_agent.execute(task)
                result.hypotheses = hypotheses
            except Exception as e:
                logger.error(f"假说生成失败 {sample.commit.short_sha}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                hypotheses = []
        result.hypothesis_time_ms = (time.time() - start_time) * 1000

        # Stage 2: 证据定位（可跳过，用于消融实验）
        start_time = time.time()
        if self.skip_evidence:
            # 消融：跳过 Evidence 验证，所有假说直接视为 confirmed
            result.confirmed_hypotheses = list(hypotheses)
            result.evidence_results = []
            logger.info(f"  [EVIDENCE] Skipped (ablation mode) — {len(hypotheses)} hypotheses treated as confirmed")
        else:
            for hypothesis in hypotheses:
                if self.mock_mode:
                    evidence_result = self._mock_verify_hypothesis(hypothesis)
                    result.evidence_results.append(evidence_result)
                    if evidence_result.status == EvidenceStatus.CONFIRMED:
                        result.confirmed_hypotheses.append(hypothesis)
                else:
                    try:
                        # 使用样本指定的 repo_path（支持多仓库）和构造时配置的 search_depth
                        repo_path = getattr(sample, "_repo_path_override", None) or str(self.repo_path)
                        evidence_task = EvidenceTask(
                            hypothesis=hypothesis,
                            repo_path=repo_path,
                            search_depth=self.search_depth,
                        )
                        evidence_result = self.evidence_agent.execute(evidence_task)
                        result.evidence_results.append(evidence_result)

                        # 筛选 CONFIRMED 的假说
                        if evidence_result.status == EvidenceStatus.CONFIRMED:
                            result.confirmed_hypotheses.append(hypothesis)
                    except Exception as e:
                        logger.warning(f"证据定位失败 {hypothesis.id}: {e}")
        result.evidence_time_ms = (time.time() - start_time) * 1000

        # ---- 写入缓存 ----
        self._save_to_cache(result)
        # ------------------

        return result

    def evaluate_all(
        self,
        samples: list[CommitSample],
        max_samples: Optional[int] = None,
        parallel_workers: int = 1,
    ) -> PipelineEvaluationResults:
        """评估所有样本。

        Args:
            samples: 待评估样本列表。
            max_samples: 最多评估样本数（None = 全部）。
            parallel_workers: 并行线程数。1 = 串行（默认），>1 = 并行调用 LLM API。
                建议设置为 API 允许的并发数，通常 4~8。
        """
        import concurrent.futures
        import threading

        if max_samples:
            samples = samples[:max_samples]

        results = PipelineEvaluationResults(total_samples=len(samples))
        start_time = time.time()

        if parallel_workers <= 1:
            # 串行执行（原有行为）
            for i, sample in enumerate(samples):
                logger.info(f"评估样本 {i+1}/{len(samples)}: {sample.commit.short_sha}")
                sample_result = self.evaluate_sample(sample)
                results.sample_results.append(sample_result)
        else:
            # 并行执行
            logger.info(f"并行模式: {parallel_workers} 个工作线程，共 {len(samples)} 个样本")
            completed = [0]
            lock = threading.Lock()

            def _run(sample: CommitSample) -> SampleResult:
                result = self.evaluate_sample(sample)
                with lock:
                    completed[0] += 1
                    logger.info(
                        f"[{completed[0]}/{len(samples)}] 完成: {sample.commit.short_sha}"
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
                        logger.error(f"样本 {sample.commit.short_sha} 失败: {e}")
                        # 插入空结果保证样本数一致
                        finished.append(SampleResult(
                            sample_id=sample.commit.short_sha,
                            commit_sha=sample.commit.sha,
                            commit_message=sample.commit.message,
                            ground_truth_has_risk=sample.ground_truth_has_risk,
                            ground_truth_values=sample.ground_truth_values,
                        ))
                # 按原始样本顺序排序
                finished.sort(key=lambda r: sample_order.get(r.sample_id, 0))
                results.sample_results = finished

        results.total_time_seconds = time.time() - start_time

        # 计算统计指标
        self._compute_statistics(results)

        return results

    def _compute_statistics(self, results: PipelineEvaluationResults):
        """计算各类统计指标"""
        # 假说统计
        all_hypotheses = []
        for sr in results.sample_results:
            all_hypotheses.extend(sr.hypotheses)

        h_stats = results.hypothesis_stats
        h_stats.total_generated = len(all_hypotheses)

        if all_hypotheses:
            confidences = [h.confidence for h in all_hypotheses]
            h_stats.avg_confidence = sum(confidences) / len(confidences)

            # 置信度分布
            h_stats.confidence_distribution = {
                "0.5-0.6": sum(1 for c in confidences if 0.5 <= c < 0.6),
                "0.6-0.7": sum(1 for c in confidences if 0.6 <= c < 0.7),
                "0.7-0.8": sum(1 for c in confidences if 0.7 <= c < 0.8),
                "0.8-0.9": sum(1 for c in confidences if 0.8 <= c < 0.9),
                "0.9-1.0": sum(1 for c in confidences if 0.9 <= c <= 1.0),
            }

            # Value 分布
            for h in all_hypotheses:
                h_stats.value_distribution[h.value_id] = h_stats.value_distribution.get(h.value_id, 0) + 1

            # Severity 分布
            for h in all_hypotheses:
                h_stats.severity_distribution[h.severity] = h_stats.severity_distribution.get(h.severity, 0) + 1

        # 证据统计
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

        # 收集所有证据的 relevance score
        all_scores = []
        for e in all_evidence:
            for piece in e.evidence_pieces:
                all_scores.append(piece.relevance_score)
        if all_scores:
            e_stats.avg_relevance_score = sum(all_scores) / len(all_scores)

        # 计算对比指标
        self._compute_comparison_metrics(results)

    def _compute_comparison_metrics(self, results: PipelineEvaluationResults):
        """计算筛选前后的 IAA 一致性指标（Human vs Pipeline）"""
        # 收集 ground truth (Human) 和预测 (Pipeline)
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
        # 维度1: 风险检测一致性 (二分类)
        # ============================================================
        
        # 筛选前
        results.before_filter_percent_agreement = percent_agreement(human_risks, pipeline_risks_before)
        results.before_filter_cohen_kappa = cohen_kappa_binary(human_risks, pipeline_risks_before)
        results.before_filter_pabak = pabak(human_risks, pipeline_risks_before)
        results.before_filter_gwet_ac1 = gwet_ac1(human_risks, pipeline_risks_before)

        # 筛选后
        results.after_filter_percent_agreement = percent_agreement(human_risks, pipeline_risks_after)
        results.after_filter_cohen_kappa = cohen_kappa_binary(human_risks, pipeline_risks_after)
        results.after_filter_pabak = pabak(human_risks, pipeline_risks_after)
        results.after_filter_gwet_ac1 = gwet_ac1(human_risks, pipeline_risks_after)

        # ============================================================
        # 维度2: 价值识别一致性 (多标签)
        # ============================================================
        
        # 筛选前
        results.before_filter_jaccard = pairwise_jaccard(human_values_list, pipeline_values_before_list)
        results.before_filter_symmetric_f1 = pairwise_symmetric_f1(human_values_list, pipeline_values_before_list)

        # 筛选后
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
    repo_dir: Optional[str] = None,
) -> list[ModelRow]:
    """Run pipeline evaluation across multiple LLM providers and collect comparison rows.

    Results are saved incrementally after each model completes when output_dir is provided,
    so partial results are preserved even if the run is interrupted.
    """
    repo_dir_path = Path(repo_dir) if repo_dir else None

    # --- Merge sample files ---
    all_samples: list[CommitSample] = []
    for i, sf in enumerate(samples_files):
        loaded = load_samples_from_json(sf)
        # Resolve repo path per sample when a base repo directory is provided.
        if repo_dir_path is not None:
            for s in loaded:
                repo_name = getattr(s, "_repo_name", None) or Path(repo_paths[i] if i < len(repo_paths) else repo_paths[0]).name
                s._repo_path_override = str(repo_dir_path / repo_name)
        else:
            rp = repo_paths[i] if i < len(repo_paths) else repo_paths[0]
            for s in loaded:
                s._repo_path_override = rp
        all_samples.extend(loaded)
        print(f"  Loaded {len(loaded)} samples from {sf}")

    print(f"  Total merged samples: {len(all_samples)}")
    if max_samples:
        all_samples = all_samples[:max_samples]
        print(f"  After max_samples limit: {len(all_samples)}")

    # Evidence 搜索时需要为每个样本用正确的 repo_path。
    # 为了实现这一点，需要将 PipelineEvaluator.evaluate_sample 手动传入 repo_path。
    # 使用第一个 repo_path 作为默认（单仓库模式，多仓库通过 _repo_path_override 覆盖）
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

        # —— 每个模型跑完后立即增量写入文件 ——
        if output_dir:
            save_multi_model_results(
                rows=rows,
                output_dir=output_dir,
                base_name=output_base_name,
            )
            print(f"  [实时写入] {output_dir}/{output_base_name}.{{json,md,tex}} "
                  f"({len(rows)}/{len(model_providers)} 模型完成)")

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
        # confirmation_rate is a float (0–1); format as LaTeX-safe percentage with \%
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
  # Unified code scenarios (mixed repos) in mock mode
  python pipeline_evaluator.py data/code_scenarios.json --repo-dir data/repos --mock --max-samples 10

  # Single repo, all models, parallel API calls
  python pipeline_evaluator.py data/code_scenarios.json --repo-dir data/repos \\
      --all-models deepseek-chat qwen-plus grok-4 --parallel-workers 4 --max-samples 10

  # Per-file repo mapping (native CommitSample JSON files)
  python pipeline_evaluator.py \\
      --samples-files path/to/focus_commits.json path/to/signal_commits.json
      --repo-paths data/repos/focus-android data/repos/Signal-Android \\
      --all-models deepseek-chat qwen-plus --parallel-workers 4
"""
    )
    # 位置参数（单个样本文件模式，保持向后兼容）
    parser.add_argument("samples_file", nargs="?", help="Path to samples JSON file (single-file mode)")
    parser.add_argument("repo_path", nargs="?", help="Path to repository (single-file mode)")
    # 多文件模式
    parser.add_argument(
        "--samples-files", nargs="+", metavar="FILE",
        help="One or more samples JSON files (merged before evaluation). Overrides positional samples_file."
    )
    parser.add_argument(
        "--repo-paths", nargs="+", metavar="REPO",
        help="Repository paths corresponding to each --samples-files entry. Overrides positional repo_path."
    )
    # 公共参数
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
    # 多模型模式
    parser.add_argument(
        "--all-models",
        nargs="+",
        metavar="PROVIDER",
        help=(
            "Run multi-model comparison mode. Pass one or more provider keys, e.g.: "
            "--all-models deepseek-chat qwen-plus o4-mini claude-sonnet-4-5 gpt-5.2 grok-4 gemini-2.5-flash"
        ),
    )
    parser.add_argument("--output-dir", default=str(exp_paths.PIPELINE_DIR), help="Output directory for multi-model results")
    parser.add_argument("--output-name", default="pipeline_multimodel", help="Base filename for multi-model outputs")
    parser.add_argument(
        "--repo-dir",
        default=str(Path(__file__).parent / "data" / "repos"),
        help="Base directory containing checked-out repositories (used to resolve repo names from scenario metadata)",
    )
    parser.add_argument("--cache-dir", default=None, metavar="DIR",
                        help="Cache directory for per-sample LLM results (default: experiment_outputs/cache/llm_outputs/pipeline_cache/<provider>/)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable result caching (re-run all samples from scratch)")
    args = parser.parse_args()
    repo_dir_path = Path(args.repo_dir)

    if args.samples_files:
        resolved_samples_files = args.samples_files
        resolved_repo_paths = args.repo_paths if args.repo_paths else (
            [args.repo_path] if args.repo_path else [str(repo_dir_path)]
        )
    else:
        if not args.samples_file:
            parser.error("Must provide either a positional samples_file or --samples-files.")
        if not args.repo_path and not repo_dir_path.exists():
            parser.error(
                "Must provide either a positional repo_path or an existing --repo-dir."
            )
        resolved_samples_files = [args.samples_file]
        resolved_repo_paths = [args.repo_path or str(repo_dir_path)]

    if not resolved_repo_paths:
        parser.error("Must provide at least one repo path via positional arg, --repo-paths, or --repo-dir.")

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
            repo_dir=args.repo_dir,
        )

        # 最终结果已在 run_multi_model_comparison 内实时写入
        # 此处只打印最终汇总表格
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
        # Resolve repo path per sample from scenario metadata when --repo-dir is given.
        repo_dir_path = Path(args.repo_dir)
        if repo_dir_path.exists():
            for s in loaded:
                repo_name = getattr(s, "_repo_name", None) or Path(resolved_repo_paths[0]).name
                s._repo_path_override = str(repo_dir_path / repo_name)
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

    # 保存结果
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
