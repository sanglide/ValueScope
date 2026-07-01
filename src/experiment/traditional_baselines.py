#!/usr/bin/env python3
"""
Traditional non-LLM baselines.

Supports three methods, applied uniformly to code and text scenarios:
  1. TF-IDF + value keyword dictionary (stdlib only)
  2. BM25 + value keyword query (requires rank-bm25)
  3. BERT zero-shot classification (requires sentence-transformers)

Input: benchmark.json (list of BenchmarkSample)
Output: JSON result files compatible with UnifiedSampleResult

Standalone usage:
  uv run python -m experiment.traditional_baselines \
      --benchmark experiment_outputs/results/main_exp/benchmark.json \
      --method all \
      --output-dir experiment_outputs/results/main_exp

Module usage:
  from experiment.traditional_baselines import run_all_baselines
"""

import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Optional

# Project paths
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from experiment import paths as exp_paths
except ImportError:
    import paths as exp_paths

logger = logging.getLogger(__name__)

# ============================================================
# Value keyword dictionary (unified for code and text)
# ============================================================

# VALUE_ID -> expanded keywords (value name + synonyms + related technical terms)
VALUE_KEYWORDS: dict[str, list[str]] = {
    "HV1": [  # Conformity
        "conformity", "conform", "convention", "standard", "compliance", "comply",
        "policy", "guideline", "rule", "norm", "regulation", "enforce",
    ],
    "HV2": [  # Pleasure
        "pleasure", "enjoyment", "fun", "delight", "satisfaction", "comfortable",
        "experience", "aesthetic", "entertainment", "happy", "enjoy",
    ],
    "HV3": [  # Dignity
        "dignity", "respect", "honor", "humiliation", "harassment", "abuse",
        "offensive", "insult", "demean", "discriminat",
    ],
    "HV4": [  # Inclusiveness
        "inclusiveness", "inclusive", "diversity", "accessible", "barrier",
        "exclude", "marginalize", "underrepresent", "minorit",
    ],
    "HV5": [  # Sense of Belonging
        "belonging", "community", "social", "collaboration", "team", "group",
        "connect", "member", "together",
    ],
    "HV6": [  # Freedom
        "freedom", "free", "autonomy", "autonomous", "choice", "opt-out", "opt out",
        "control", "restrict", "censor", "block", "ban", "limit",
    ],
    "HV7": [  # Independence
        "independence", "independent", "self-reliant", "standalone", "decoupled",
        "vendor lock", "dependency", "portable",
    ],
    "HV8": [  # Wealth
        "wealth", "cost", "money", "profit", "revenue", "monetize", "fee",
        "subscription", "paid", "premium", "billing", "price",
    ],
    "HV9": [  # Privacy
        "privacy", "private", "personal data", "personal information", "pii",
        "gdpr", "tracking", "surveillance", "leak", "exposure", "confidential",
        "sensitive", "anonymous", "anonymize", "consent", "opt-in", "collect data",
        "log", "telemetry",
    ],
    "HV10": [  # Security
        "security", "secure", "vulnerabilit", "exploit", "attack", "inject",
        "xss", "csrf", "sql injection", "overflow", "authentication", "auth",
        "password", "token", "encrypt", "decrypt", "crypto", "hash", "tls", "ssl",
        "certificate", "unauthorized", "privilege", "permission", "access control",
    ],
    "SV1": [  # Trust
        "trust", "trustworthy", "reliable", "honest", "transparent", "integrity",
        "verify", "verify", "mislead", "fake", "scam",
    ],
    "SV2": [  # Correctness
        "correctness", "correct", "accurate", "accuracy", "bug", "error", "fault",
        "wrong", "incorrect", "fix", "regression", "test", "validation", "assert",
        "expected",
    ],
    "SV3": [  # Compatibility
        "compatibility", "compatible", "interoperable", "backward", "forward",
        "migration", "version", "api break", "breaking change",
    ],
    "SV4": [  # Portability
        "portability", "portable", "cross-platform", "platform-independent",
        "windows", "linux", "macos", "android", "ios", "architecture",
    ],
    "SV5": [  # Reliability
        "reliability", "reliable", "availability", "uptime", "failover", "fault",
        "crash", "hang", "timeout", "retry", "resilience",
    ],
    "SV6": [  # Efficiency
        "efficiency", "efficient", "performance", "speed", "latency", "throughput",
        "bottleneck", "optimize", "slow", "fast", "memory", "cpu", "resource",
    ],
    "SV7": [  # Energy Preservation
        "energy", "battery", "power", "consumption", "green", "carbon", "wakelock",
        "doze", "background process",
    ],
    "SV8": [  # Usability
        "usability", "usable", "user-friendly", "intuitive", "confus", "difficult",
        "workflow", "ux", "ui", "interface", "accessibility", "documentation",
        "unclear", "hard to",
    ],
    "SV9": [  # Accessibility
        "accessibility", "accessible", "screen reader", "aria", "wcag",
        "visually impaired", "disability", "a11y", "blind", "deaf",
    ],
    "SV10": [  # Longevity
        "longevity", "maintenance", "deprecated", "obsolete", "eol", "end of life",
        "legacy", "long-term", "sustain", "archive",
    ],
}

# ============================================================
# 内容预处理：统一处理 code 和 text
# ============================================================

def preprocess_content(content: str, scenario_type: str) -> str:
    """
    预处理 content 字段：
    - code 场景中的 diff 格式：保留 '+' 行（新增代码），去除 '---'/'+++'/metadata 行
    - 其他 code：保留注释（// /* */ # 等含大量信息）
    - text 场景：直接使用原文
    """
    if scenario_type == "code" and "--- a/" in content[:200]:
        # unified diff 格式：只保留新增行（以 '+' 开头但不是 '+++ b/'）
        lines = []
        for line in content.splitlines():
            if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@ "):
                continue
            if line.startswith("+"):
                lines.append(line[1:])  # 去掉前缀 '+'
            elif not line.startswith("-"):
                lines.append(line)
        return " ".join(lines)
    return content


def tokenize(text: str) -> list[str]:
    """简单 tokenizer：小写 + 按非字母数字分割"""
    text = text.lower()
    tokens = re.findall(r'[a-z][a-z0-9_]*', text)
    return tokens


# ============================================================
# 方法一：TF-IDF + 价值词典
# ============================================================

class TFIDFValueBaseline:
    """
    TF-IDF + 价值关键词词典 Baseline

    对每个 value，计算样本文本与该 value 关键词集合的 TF-IDF 加权匹配得分。
    得分超过阈值则认为该 value 存在风险，任意 value 有风险则 has_risk=True。

    核心思想：
      score(value, doc) = Σ_{kw ∈ KW(value)} tf(kw, doc) * idf(kw)
      其中 idf 在整个 benchmark corpus 上计算。
    """

    def __init__(self, threshold: float = 0.0):
        """
        Args:
            threshold: 判定某个 value 存在风险的得分阈值
                      0.0 = 出现任意关键词即为有风险（高 Recall）
                      正数 = 需要关键词出现足够多次
        """
        self.threshold = threshold
        self._idf: dict[str, float] = {}
        self._corpus_size = 0

    def fit(self, documents: list[str]):
        """在 corpus 上计算 IDF"""
        self._corpus_size = len(documents)
        doc_freq: dict[str, int] = {}
        for doc in documents:
            tokens = set(tokenize(doc))
            for t in tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1
        # IDF with smoothing: log((N+1)/(df+1)) + 1
        for term, df in doc_freq.items():
            self._idf[term] = math.log((self._corpus_size + 1) / (df + 1)) + 1.0

    def _score_value(self, tokens: list[str], kws: list[str]) -> float:
        """计算单个 value 的 TF-IDF 得分"""
        if not tokens:
            return 0.0
        token_count: dict[str, int] = {}
        for t in tokens:
            token_count[t] = token_count.get(t, 0) + 1

        total_tokens = len(tokens)
        score = 0.0
        for kw in kws:
            kw_tokens = tokenize(kw)
            if not kw_tokens:
                continue
            # 计算 bigram 或 unigram 的匹配
            for kt in kw_tokens:
                tf = token_count.get(kt, 0) / total_tokens
                idf = self._idf.get(kt, math.log(self._corpus_size + 1) + 1.0)
                score += tf * idf
        return score

    def predict_sample(self, content: str, scenario_type: str) -> dict:
        """对单个样本预测"""
        processed = preprocess_content(content, scenario_type)
        tokens = tokenize(processed)

        predicted_values = []
        confidences = {}

        for value_id, kws in VALUE_KEYWORDS.items():
            score = self._score_value(tokens, kws)
            # 归一化到 [0, 1]：用最大可能分数估计
            max_score = sum(
                math.log(self._corpus_size + 1) + 1.0
                for kw in kws for _ in tokenize(kw)
            )
            normalized = min(score / max_score, 1.0) if max_score > 0 else 0.0
            confidences[value_id] = round(normalized, 4)
            if score > self.threshold:
                predicted_values.append(value_id)

        has_risk = len(predicted_values) > 0
        return {
            "predicted_has_risk": has_risk,
            "predicted_values": predicted_values,
            "predicted_confidences": confidences,
        }

    def predict_all(self, samples: list[dict]) -> list[dict]:
        """预测所有样本，返回 UnifiedSampleResult 格式的 dict 列表"""
        # Fit IDF on corpus
        docs = [preprocess_content(s.get("content", ""), s.get("scenario_type", "code"))
                for s in samples]
        self.fit(docs)

        results = []
        for s in samples:
            pred = self.predict_sample(s.get("content", ""), s.get("scenario_type", "code"))
            results.append({
                "sample_id": s["sample_id"],
                "scenario_type": s["scenario_type"],
                "repo": s.get("repo", "unknown"),
                "predicted_has_risk": pred["predicted_has_risk"],
                "predicted_values": pred["predicted_values"],
                "predicted_confidences": pred["predicted_confidences"],
                "ground_truth_has_risk": s.get("has_value_risk", False),
                "ground_truth_values": s.get("ground_truth_values", []),
                "hypothesis_count": 0,
                "confirmed_count": 0,
                "profile_used": "none",
                "total_time_ms": 0.0,
            })
        return results


# ============================================================
# 方法二：BM25 + 价值关键词查询
# ============================================================

class BM25ValueBaseline:
    """
    BM25 + 价值关键词查询 Baseline

    对 benchmark 中所有样本构建 BM25 索引。
    对每个 value，把其关键词拼接为"查询"，检索 BM25 得分。
    若样本对某个 value 的 BM25 得分 >= 阈值，则认为该 value 有风险。
    """

    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold: BM25 得分归一化后的阈值（0~1）
        """
        self.threshold = threshold
        self._bm25 = None
        self._tokenized_corpus: list[list[str]] = []

    def fit(self, documents: list[str]):
        """构建 BM25 索引"""
        from rank_bm25 import BM25Okapi
        self._tokenized_corpus = [tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def predict_sample(self, doc_idx: int) -> dict[str, float]:
        """对第 doc_idx 个样本，返回每个 value 的 BM25 得分（归一化）"""
        scores = {}
        doc_tokens = self._tokenized_corpus[doc_idx]

        for value_id, kws in VALUE_KEYWORDS.items():
            # 查询 = 所有关键词的 token 列表
            query_tokens = []
            for kw in kws:
                query_tokens.extend(tokenize(kw))

            if not query_tokens or not doc_tokens:
                scores[value_id] = 0.0
                continue

            # BM25 对整个 corpus 打分，取当前样本的分数
            all_scores = self._bm25.get_scores(query_tokens)
            raw_score = float(all_scores[doc_idx])

            # 归一化：用该查询在 corpus 中的最大分数
            max_score = float(max(all_scores)) if max(all_scores) > 0 else 1.0
            normalized = min(raw_score / max_score, 1.0) if max_score > 0 else 0.0
            scores[value_id] = round(normalized, 4)

        return scores

    def predict_all(self, samples: list[dict]) -> list[dict]:
        """预测所有样本"""
        docs = [preprocess_content(s.get("content", ""), s.get("scenario_type", "code"))
                for s in samples]
        self.fit(docs)

        results = []
        for idx, s in enumerate(samples):
            value_scores = self.predict_sample(idx)
            predicted_values = [vid for vid, score in value_scores.items()
                                 if score >= self.threshold]
            has_risk = len(predicted_values) > 0

            results.append({
                "sample_id": s["sample_id"],
                "scenario_type": s["scenario_type"],
                "repo": s.get("repo", "unknown"),
                "predicted_has_risk": has_risk,
                "predicted_values": predicted_values,
                "predicted_confidences": value_scores,
                "ground_truth_has_risk": s.get("has_value_risk", False),
                "ground_truth_values": s.get("ground_truth_values", []),
                "hypothesis_count": 0,
                "confirmed_count": 0,
                "profile_used": "none",
                "total_time_ms": 0.0,
            })
        return results


# ============================================================
# 方法三：BERT 零样本分类（sentence-transformers）
# ============================================================

class BERTZeroShotBaseline:
    """
    BERT 零样本语义匹配 Baseline

    使用 sentence-transformers 模型计算文本与每个 value 描述的余弦相似度。
    相似度超过阈值则认为该 value 有风险。

    模型选择：all-MiniLM-L6-v2（轻量，速度快，无需GPU）
    """

    # 每个 value 的自然语言描述（用于语义匹配）
    VALUE_DESCRIPTIONS: dict[str, str] = {
        "HV1": "conformity, compliance with rules, conventions and social norms",
        "HV2": "pleasure, enjoyment, fun and satisfying user experience",
        "HV3": "dignity, respect, honor and protection from humiliation or harassment",
        "HV4": "inclusiveness, diversity, accessibility for all users regardless of background",
        "HV5": "sense of belonging, community, social connection and team collaboration",
        "HV6": "freedom, autonomy, user control and absence of censorship or restrictions",
        "HV7": "independence, self-reliance, avoiding vendor lock-in and dependency",
        "HV8": "wealth, cost, financial sustainability, monetization and pricing",
        "HV9": "privacy, personal data protection, preventing data leakage and tracking",
        "HV10": "security, preventing vulnerabilities, attacks, authentication and encryption",
        "SV1": "trust, transparency, honesty and reliability of information",
        "SV2": "correctness, accuracy, bug-free behavior and expected functionality",
        "SV3": "compatibility, interoperability, backward compatibility and migration",
        "SV4": "portability, cross-platform support and platform independence",
        "SV5": "reliability, availability, crash prevention and fault tolerance",
        "SV6": "efficiency, performance, speed optimization and resource usage",
        "SV7": "energy preservation, battery life, power consumption reduction",
        "SV8": "usability, user-friendly interface, intuitive workflow and documentation",
        "SV9": "accessibility, screen reader support, disability accommodation",
        "SV10": "longevity, maintainability, avoiding deprecated or obsolete dependencies",
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.35,
        batch_size: int = 64,
        max_seq_length: int = 256,
    ):
        """
        Args:
            model_name: sentence-transformers 模型名称
            threshold: 余弦相似度阈值（0~1），超过则判定为有风险
            batch_size: 批量编码大小
            max_seq_length: 最大序列长度（content 会截断到此长度对应的字符数）
        """
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self._model = None
        self._value_embeddings: Optional[dict] = None

    def _load_model(self):
        """懒加载模型（自动使用 HF 镜像）"""
        if self._model is None:
            import os
            # 若无法访问 HuggingFace，自动切换国内镜像
            if not os.environ.get("HF_ENDPOINT"):
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._model.max_seq_length = self.max_seq_length

    def _encode_values(self):
        """编码所有 value 描述（只做一次）"""
        if self._value_embeddings is None:
            descriptions = [self.VALUE_DESCRIPTIONS[vid]
                            for vid in sorted(self.VALUE_DESCRIPTIONS.keys())]
            value_ids = sorted(self.VALUE_DESCRIPTIONS.keys())
            embeddings = self._model.encode(
                descriptions,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            self._value_embeddings = {
                vid: emb for vid, emb in zip(value_ids, embeddings)
            }

    def predict_all(self, samples: list[dict]) -> list[dict]:
        """预测所有样本"""
        import numpy as np

        self._load_model()
        self._encode_values()

        # 批量编码所有文档
        contents = []
        for s in samples:
            raw = preprocess_content(s.get("content", ""), s.get("scenario_type", "code"))
            # 截断过长文本（按字符数粗略控制，实际 token 截断由模型处理）
            contents.append(raw[:2000])

        logger.info(f"Encoding {len(contents)} documents with BERT...")
        doc_embeddings = self._model.encode(
            contents,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # 计算余弦相似度并预测
        # 先堆叠 value 矩阵（V x D）
        value_ids = sorted(self._value_embeddings.keys())
        value_matrix = np.stack([self._value_embeddings[vid] for vid in value_ids])  # (V, D)

        # 归一化
        doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        val_norms = np.linalg.norm(value_matrix, axis=1, keepdims=True)
        doc_emb_norm = doc_embeddings / np.maximum(doc_norms, 1e-8)
        val_emb_norm = value_matrix / np.maximum(val_norms, 1e-8)

        # cos_sim: (N, V)
        cos_sim = doc_emb_norm @ val_emb_norm.T

        results = []
        for idx, s in enumerate(samples):
            sims = cos_sim[idx]  # shape (V,)
            confidences = {vid: round(float(sims[i]), 4) for i, vid in enumerate(value_ids)}
            predicted_values = [vid for i, vid in enumerate(value_ids)
                                 if float(sims[i]) >= self.threshold]
            has_risk = len(predicted_values) > 0

            results.append({
                "sample_id": s["sample_id"],
                "scenario_type": s["scenario_type"],
                "repo": s.get("repo", "unknown"),
                "predicted_has_risk": has_risk,
                "predicted_values": predicted_values,
                "predicted_confidences": confidences,
                "ground_truth_has_risk": s.get("has_value_risk", False),
                "ground_truth_values": s.get("ground_truth_values", []),
                "hypothesis_count": 0,
                "confirmed_count": 0,
                "profile_used": "none",
                "total_time_ms": 0.0,
            })
        return results


# ============================================================
# 统一结果转换为 UnifiedSampleResult
# ============================================================

def dicts_to_unified_results(result_dicts: list[dict]):
    """将 dict 列表转换为 UnifiedSampleResult 列表"""
    try:
        from experiment.unified_pipeline import UnifiedSampleResult
    except ImportError:
        from unified_pipeline import UnifiedSampleResult

    results = []
    for d in result_dicts:
        results.append(UnifiedSampleResult(
            sample_id=d["sample_id"],
            scenario_type=d["scenario_type"],
            repo=d["repo"],
            predicted_has_risk=d["predicted_has_risk"],
            predicted_values=d["predicted_values"],
            predicted_confidences=d["predicted_confidences"],
            ground_truth_has_risk=d["ground_truth_has_risk"],
            ground_truth_values=d["ground_truth_values"],
            hypothesis_count=0,
            confirmed_count=0,
            profile_used="none",
            total_time_ms=0.0,
        ))
    return results


# ============================================================
# 一键运行所有 baselines
# ============================================================

def run_all_baselines(
    benchmark_path: str,
    output_dir: str,
    methods: Optional[list[str]] = None,
    tfidf_threshold: float = 0.0,
    bm25_threshold: float = 0.1,
    bert_threshold: float = 0.10,
    bert_model: str = "all-MiniLM-L6-v2",
) -> dict:
    """
    运行所有传统 baseline，保存结果文件并返回各方法的 UnifiedSampleResult 列表。

    Args:
        benchmark_path: benchmark.json 路径
        output_dir: 输出目录
        methods: 要运行的方法列表，可选 ["tfidf", "bm25", "bert"]，None 表示全部
        tfidf_threshold: TF-IDF 得分阈值
        bm25_threshold: BM25 得分归一化阈值
        bert_threshold: BERT 余弦相似度阈值

    Returns:
        {method_name: list[UnifiedSampleResult]}
    """
    if methods is None:
        methods = ["tfidf", "bm25", "bert"]

    bench_path = Path(benchmark_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载 benchmark
    samples = json.loads(bench_path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(samples)} samples from {bench_path}")

    all_results = {}

    # ---------- TF-IDF ----------
    if "tfidf" in methods:
        print(f"\n[TF-IDF] Running on {len(samples)} samples...")
        baseline = TFIDFValueBaseline(threshold=tfidf_threshold)
        result_dicts = baseline.predict_all(samples)

        # 保存
        out_path = out_dir / "tfidf_results.json"
        out_path.write_text(json.dumps(result_dicts, ensure_ascii=False, indent=2), encoding="utf-8")

        unified = dicts_to_unified_results(result_dicts)
        all_results["TF-IDF (Value Dict)"] = unified

        # 打印摘要
        _print_summary("TF-IDF (Value Dict)", unified)

    # ---------- BM25 ----------
    if "bm25" in methods:
        print(f"\n[BM25] Running on {len(samples)} samples...")
        baseline = BM25ValueBaseline(threshold=bm25_threshold)
        result_dicts = baseline.predict_all(samples)

        out_path = out_dir / "bm25_results.json"
        out_path.write_text(json.dumps(result_dicts, ensure_ascii=False, indent=2), encoding="utf-8")

        unified = dicts_to_unified_results(result_dicts)
        all_results["BM25 (Value Dict)"] = unified
        _print_summary("BM25 (Value Dict)", unified)

    # ---------- BERT ----------
    if "bert" in methods:
        print(f"\n[BERT] Running on {len(samples)} samples (model: {bert_model})...")
        baseline = BERTZeroShotBaseline(
            model_name=bert_model,
            threshold=bert_threshold,
        )
        result_dicts = baseline.predict_all(samples)

        out_path = out_dir / "bert_zeroshot_results.json"
        out_path.write_text(json.dumps(result_dicts, ensure_ascii=False, indent=2), encoding="utf-8")

        unified = dicts_to_unified_results(result_dicts)
        all_results["BERT Zero-shot"] = unified
        _print_summary("BERT Zero-shot", unified)

    # ---------- 保存 metrics ----------
    try:
        from experiment.unified_pipeline import compute_unified_metrics
    except ImportError:
        from unified_pipeline import compute_unified_metrics

    all_metrics = {}
    for method_name, unified_results in all_results.items():
        metrics = compute_unified_metrics(unified_results)
        all_metrics[method_name] = metrics
        out_key = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        (out_dir / f"{out_key}_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 汇总 metrics 表格
    _print_comparison_table(all_metrics)

    # 保存汇总
    (out_dir / "traditional_baselines_summary.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nResults saved to: {out_dir}")
    return all_results


def _print_summary(method_name: str, unified_results):
    """打印单个方法的 Dim1 + Dim2 指标"""
    try:
        from experiment.unified_pipeline import compute_unified_metrics
    except ImportError:
        from unified_pipeline import compute_unified_metrics

    metrics = compute_unified_metrics(unified_results)
    print(f"\n  [{method_name}]")
    print(f"    {'Scenario':<8}  {'N':>5}  "
          f"{'[Dim1] Prec':>11} {'Rec':>6} {'F1':>6} {'κ':>6}  "
          f"{'[Dim2] mP':>9} {'mR':>6} {'mF1':>6} {'Jaccard':>8} {'Sym-F1':>7}")
    print(f"    {'-'*8}  {'-'*5}  {'-'*11} {'-'*6} {'-'*6} {'-'*6}  "
          f"{'-'*9} {'-'*6} {'-'*6} {'-'*8} {'-'*7}")
    for scenario in ["code", "text", "overall"]:
        m = metrics.get(scenario, {})
        if not m:
            continue
        d1 = m.get("dim1_risk_detection", {})
        d2 = m.get("dim2_value_identification", {})
        print(f"    {scenario:<8}  {m['n']:>5}  "
              f"{d1.get('precision', 0):>11.3f} "
              f"{d1.get('recall', 0):>6.3f} "
              f"{d1.get('f1', 0):>6.3f} "
              f"{d1.get('cohen_kappa', 0):>6.3f}  "
              f"{d2.get('micro_precision', 0):>9.3f} "
              f"{d2.get('micro_recall', 0):>6.3f} "
              f"{d2.get('micro_f1', 0):>6.3f} "
              f"{d2.get('pairwise_jaccard', 0):>8.3f} "
              f"{d2.get('symmetric_f1', 0):>7.3f}")


def _print_comparison_table(all_metrics: dict):
    """打印所有方法的完整对比表（Dim1 + Dim2）"""
    # ----- Dim1 表 -----
    print("\n" + "=" * 100)
    print("[Dim1] Risk Detection")
    print(f"  {'Method':<25} {'Scenario':<8} {'N':>5} {'Prec':>6} {'Rec':>6} {'F1':>6} {'κ':>7}")
    print("  " + "-" * 68)
    for method_name, metrics in all_metrics.items():
        for scenario in ["code", "text", "overall"]:
            m = metrics.get(scenario, {})
            if not m:
                continue
            d1 = m.get("dim1_risk_detection", {})
            print(f"  {method_name:<25} {scenario:<8} {m['n']:>5} "
                  f"{d1.get('precision', 0):>6.3f} "
                  f"{d1.get('recall', 0):>6.3f} "
                  f"{d1.get('f1', 0):>6.3f} "
                  f"{d1.get('cohen_kappa', 0):>7.3f}")

    # ----- Dim2 表 -----
    print()
    print("[Dim2] Value Identification")
    print(f"  {'Method':<25} {'Scenario':<8} {'N':>5} {'micro-P':>8} {'micro-R':>8} {'micro-F1':>9} {'Jaccard':>8} {'Sym-F1':>7}")
    print("  " + "-" * 80)
    for method_name, metrics in all_metrics.items():
        for scenario in ["code", "text", "overall"]:
            m = metrics.get(scenario, {})
            if not m:
                continue
            d2 = m.get("dim2_value_identification", {})
            print(f"  {method_name:<25} {scenario:<8} {m['n']:>5} "
                  f"{d2.get('micro_precision', 0):>8.3f} "
                  f"{d2.get('micro_recall', 0):>8.3f} "
                  f"{d2.get('micro_f1', 0):>9.3f} "
                  f"{d2.get('pairwise_jaccard', 0):>8.3f} "
                  f"{d2.get('symmetric_f1', 0):>7.3f}")
    print("=" * 100)


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(
        description="Traditional baselines (TF-IDF / BM25 / BERT Zero-shot)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark", required=True,
        help="Path to benchmark.json (e.g., experiment_outputs/results/main_exp/benchmark.json)"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--method", default="all",
        choices=["all", "tfidf", "bm25", "bert"],
        help="运行指定方法（默认 all）"
    )
    parser.add_argument("--tfidf-threshold", type=float, default=0.0)
    parser.add_argument("--bm25-threshold", type=float, default=0.1)
    parser.add_argument("--bert-threshold", type=float, default=0.10)
    parser.add_argument("--bert-model", default="all-MiniLM-L6-v2",
                        help="sentence-transformers 模型名称")

    args = parser.parse_args()

    methods = ["tfidf", "bm25", "bert"] if args.method == "all" else [args.method]

    run_all_baselines(
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        methods=methods,
        tfidf_threshold=args.tfidf_threshold,
        bm25_threshold=args.bm25_threshold,
        bert_threshold=args.bert_threshold,
        bert_model=args.bert_model,
    )
