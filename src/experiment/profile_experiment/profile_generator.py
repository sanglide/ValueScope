#!/usr/bin/env python
"""
Value profile generator (lightweight, standalone).
Reuses experiment/llm_client.py for LLM calls.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt (kept in sync with profiler_agent.py)
# ---------------------------------------------------------------------------

VALUE_CLASSIFICATION_SYSTEM = """You are a value analyst for software projects.
Your task is to identify human and system values expressed in project documentation.

Given project documentation, identify which values from the L2 (Human Value Themes)
and L3 (System Value Themes) are expressed or implied.

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

Rate each value on a scale of 0.0 to 1.0 based on how strongly it is expressed.
0.0 means not expressed at all, 1.0 means a core project value.

Respond in JSON format:
{
  "l2_scores": {"HV1": 0.0, "HV2": 0.0, ..., "HV10": 0.0},
  "l3_scores": {"SV1": 0.0, "SV2": 0.0, ..., "SV10": 0.0},
  "core_values": ["top 3 value IDs"],
  "evidence": [
    {"value_id": "HVX", "quote": "relevant text from documentation", "confidence": 0.9}
  ]
}

Important:
- Rate ALL 20 dimensions (HV1-HV10, SV1-SV10), even if 0.0.
- core_values should list the top-3 value IDs by score.
- evidence should contain 3-5 most important supporting quotes.
"""

# All L2 / L3 value IDs
L2_IDS = [f"HV{i}" for i in range(1, 11)]
L3_IDS = [f"SV{i}" for i in range(1, 11)]
ALL_VALUE_IDS = L2_IDS + L3_IDS

# Code file extensions
_CODE_EXTENSIONS = {".java", ".kt", ".py", ".go", ".ts", ".js", ".rs", ".c", ".cpp", ".h", ".swift"}

# ---------------------------------------------------------------------------
# Stage 2 Prompt — 代码证据分析
# ---------------------------------------------------------------------------

CODE_VALUE_ANALYSIS_SYSTEM = """You are a code value analyst for software projects.
Your task is to identify which human and system values are **actually demonstrated** in the
project's source code through implementation patterns, design decisions, and code structure.

IMPORTANT: Focus on what the CODE does, not what documentation CLAIMS.
- High score = the code has concrete, extensive implementation patterns supporting this value
- Low score = the code has minimal or no implementation patterns for this value
- A value that is merely mentioned in docs but has no code evidence should score LOW

## L2 Human Value Themes (what the code does for users):
- HV1 (Conformity): Input validation, schema enforcement, compliance checks
- HV2 (Pleasure): UX-related code, animations, smooth interactions
- HV3 (Dignity): User control features, consent mechanisms, transparent behavior
- HV4 (Inclusiveness): i18n/l10n, accessibility APIs, adaptive UI, multi-platform support
- HV5 (Sense of belonging): Community features, social interactions, user profiles
- HV6 (Freedom): Plugin systems, configurable options, user choice, extensibility
- HV7 (Independence): Offline support, self-contained modules, no vendor lock-in patterns
- HV8 (Wealth): Caching, optimization, resource efficiency, cost-saving patterns
- HV9 (Privacy): Encryption, data minimization, anonymization, access control
- HV10 (Security): Auth mechanisms, input sanitization, secure protocols, vulnerability prevention

## L3 System Value Themes (how the code is built):
- SV1 (Trust): Error handling, logging, audit trails, rollback mechanisms
- SV2 (Correctness): Test coverage patterns, type safety, assertions, formal verification
- SV3 (Compatibility): API versioning, backward compatibility, protocol adapters
- SV4 (Portability): Platform abstractions, cross-platform code, containerization
- SV5 (Reliability): Retry logic, health checks, failover, graceful degradation
- SV6 (Efficiency): Performance-critical paths, algorithms, data structure choices
- SV7 (Energy Preservation): Power management, lazy loading, background task scheduling
- SV8 (Usability): Clear API design, documentation in code, helpful error messages
- SV9 (Accessibility): ARIA attributes, screen reader support, keyboard navigation
- SV10 (Longevity): Code modularity, clean architecture, low coupling, maintainability

Rate each value from 0.0 to 1.0 based on CODE EVIDENCE strength:
- 0.0-0.2: No or negligible code evidence
- 0.3-0.5: Some code patterns but not a major focus
- 0.6-0.8: Significant code investment, clear implementation patterns
- 0.9-1.0: Core architectural feature, extensively implemented

Be DISCRIMINATING. Most projects have 2-4 core values at 0.7+, and many values at 0.2-0.4.
Do NOT rate everything high. Only rate high what the code CLEARLY demonstrates.

Respond in JSON format:
{
  "l2_scores": {"HV1": 0.0, "HV2": 0.0, ..., "HV10": 0.0},
  "l3_scores": {"SV1": 0.0, "SV2": 0.0, ..., "SV10": 0.0},
  "core_values": ["top 3 value IDs based on CODE evidence"],
  "evidence": [
    {"value_id": "HVX", "code_pattern": "brief description of code pattern", "confidence": 0.9}
  ]
}"""

# 文档收集时关注的文件 / 目录后缀
_DOC_GLOBS = [
    "README.md", "README.rst", "README.txt", "README",
    "CONTRIBUTING.md", "CONTRIBUTING.rst",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "LICENSE", "LICENSE.md",
    "CODEOWNERS",
]
_DOC_DIRS = [".github", "docs"]
_DOC_EXTENSIONS = {".md", ".rst", ".txt", ".yml", ".yaml", ".toml", ".cfg", ".ini"}


class ProfileGenerator:
    """Lightweight value profile generator."""

    def __init__(
        self,
        llm_clients: dict,
        value_model_text: str = "",
        cache_dir: str = None,
    ):
        if cache_dir is None:
            from experiment import paths as exp_paths
            cache_dir = str(exp_paths.PROFILE_CACHE_DIR)
        self.llm_clients = llm_clients  # {model_key: BaseLLMClient}
        self.value_model_text = value_model_text
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_absolute():
            self.cache_dir = Path(__file__).parent.parent.parent.parent / self.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Document collection
    # ------------------------------------------------------------------

    def collect_documents(
        self,
        repo_path: str,
        sources: Optional[list[str]] = None,
        max_length: int = 4000,
    ) -> list[dict]:
        """从项目目录递归收集文档。

        Returns:
            [{source_type, file_path, content}]
        """
        repo = Path(repo_path)
        if not repo.exists():
            logger.warning(f"仓库路径不存在: {repo_path}")
            return []

        collected: list[dict] = []

        # 1. 根目录下的已知文档文件
        for name in _DOC_GLOBS:
            fpath = repo / name
            if fpath.is_file():
                content = self._read_file(fpath, max_length)
                if content:
                    collected.append({
                        "source_type": "root_doc",
                        "file_path": str(fpath.relative_to(repo)),
                        "content": content,
                    })

        # 2. 已知文档目录
        for dname in _DOC_DIRS:
            dpath = repo / dname
            if dpath.is_dir():
                for fpath in sorted(dpath.rglob("*")):
                    if fpath.is_file() and fpath.suffix.lower() in _DOC_EXTENSIONS:
                        # 跳过 .git 内部
                        if ".git" in fpath.parts:
                            continue
                        content = self._read_file(fpath, max_length)
                        if content:
                            collected.append({
                                "source_type": "doc_dir",
                                "file_path": str(fpath.relative_to(repo)),
                                "content": content,
                            })

        # 3. 如果指定了额外 sources，也纳入
        if sources:
            for src in sources:
                spath = repo / src
                if spath.is_file() and str(spath.relative_to(repo)) not in {
                    d["file_path"] for d in collected
                }:
                    content = self._read_file(spath, max_length)
                    if content:
                        collected.append({
                            "source_type": "extra",
                            "file_path": str(spath.relative_to(repo)),
                            "content": content,
                        })
                elif spath.is_dir():
                    for fpath in sorted(spath.rglob("*")):
                        if fpath.is_file() and fpath.suffix.lower() in _DOC_EXTENSIONS:
                            if ".git" in fpath.parts:
                                continue
                            rel = str(fpath.relative_to(repo))
                            if rel not in {d["file_path"] for d in collected}:
                                content = self._read_file(fpath, max_length)
                                if content:
                                    collected.append({
                                        "source_type": "extra",
                                        "file_path": rel,
                                        "content": content,
                                    })

        logger.info(f"从 {repo_path} 收集到 {len(collected)} 个文档")
        return collected

    def summarize_documents(
        self,
        documents: list[dict],
        model_key: str,
        max_summary_length: int = 150,  # 降低到 150 字符
        batch_size: int = 10,
        max_docs: int = 100,  # 新增：最多保留的文档数
    ) -> list[dict]:
        """对大量文档进行摘要压缩，减少上下文长度。
        
        Args:
            documents: 原始文档列表 [{source_type, file_path, content}]
            model_key: 用于摘要的 LLM 模型
            max_summary_length: 每个文档摘要的最大字符数
            batch_size: 批处理大小（避免过多 API调用）
            max_docs: 最多保留的文档数量（用于超大项目）
        
        Returns:
            摘要后的文档列表 [{source_type, file_path, content, summary}]
        """
        if not documents:
            return []
        
        # 如果文档数量较少（<20），直接返回原文档
        if len(documents) < 20:
            for doc in documents:
                doc['summary'] = doc['content']  # 不摘要
            return documents
        
        logger.info(f"开始对 {len(documents)} 个文档进行摘要...")
        
        # ========== 第一阶段：文档重要性排序与选择 ==========
        priority_order = {
            'README.md': 1, 'README.rst': 1, 'README.txt': 1, 'README': 1,
            'CONTRIBUTING.md': 2, 'CONTRIBUTING.rst': 2,
            'CODE_OF_CONDUCT.md': 3, 'SECURITY.md': 4,
            'LICENSE': 5, 'LICENSE.md': 5,
        }
        
        def get_priority(doc: dict) -> int:
            filename = Path(doc['file_path']).name
            base_name = Path(doc['file_path']).stem.lower()
            
            # 根目录的核心文档优先级最高
            if doc['file_path'] in priority_order:
                return priority_order[doc['file_path']]
            
            # 根据文件名判断
            for key in priority_order:
                if base_name == Path(key).stem.lower():
                    return priority_order[key] + 10
            
            # .github 目录中的 workflow 文件优先级较低
            if '.github/workflows' in doc['file_path']:
                return 999
            
            # docs 目录按路径深度排序
            if doc['file_path'].startswith('docs/'):
                depth = doc['file_path'].count('/')
                return 100 + depth
            
            # 其他文档
            return 500
        
        # 按优先级排序并选择前 N 个重要文档
        sorted_docs = sorted(documents, key=get_priority)
        selected_docs = sorted_docs[:max_docs]
        
        if len(selected_docs) < len(documents):
            logger.info(f"文档选择：从 {len(documents)} 个中选择最重要的 {len(selected_docs)} 个")
        
        # ========== 第二阶段：对选中的文档进行摘要 ==========
        client = self.llm_clients.get(model_key)
        if client is None:
            logger.warning(f"LLM 客户端不存在：{model_key}，使用原文档")
            for doc in selected_docs:
                doc['summary'] = doc['content']
            return selected_docs
        
        summary_prompt_template = """Summarize the following project documentation into a very brief abstract (max {max_length} characters).
Focus ONLY on values, principles, and guidelines. Ignore technical details.

Text:
{text}

Abstract (2-3 sentences, focus on values):"""
        
        summarized_docs = []
        
        for i, doc in enumerate(selected_docs):
            try:
                # 检查是否已有摘要缓存
                cache_key = f"doc_summary_{hashlib.md5(doc['content'].encode()).hexdigest()}"
                cached_summary = self._load_cache_string(cache_key)
                
                if cached_summary:
                    doc['summary'] = cached_summary
                    summarized_docs.append(doc)
                    continue
                
                # 调用 LLM 生成摘要
                prompt = summary_prompt_template.format(
                    text=doc['content'][:6000],  # 限制输入长度
                    max_length=max_summary_length
                )
                
                response = client.call(
                    system_prompt="You are a technical documentation summarizer specializing in extracting project values.",
                    user_prompt=prompt,
                )
                
                summary = response.raw_response.strip()
                doc['summary'] = summary
                
                # 缓存摘要（作为字符串）
                self._save_cache_string(cache_key, summary)
                
                summarized_docs.append(doc)
                
                if (i + 1) % 20 == 0:
                    logger.info(f"已摘要 {i+1}/{len(selected_docs)} 个文档")
                    
            except Exception as e:
                logger.warning(f"文档摘要失败：{doc['file_path']}，使用原文档 - {e}")
                doc['summary'] = doc['content']
                summarized_docs.append(doc)
        
        total_original = sum(len(d['content']) for d in summarized_docs)
        total_summarized = sum(len(d['summary']) for d in summarized_docs)
        logger.info(f"文档摘要完成！选中 {len(summarized_docs)} 个文档，"
                   f"原始总字符：{total_original:,} → 摘要后总字符：{total_summarized:,}，"
                   f"压缩率：{(1 - total_summarized/total_original)*100:.1f}%")
        
        return summarized_docs

    def collect_documents_incremental(
        self,
        repo_path: str,
        steps: list[list[str]],
        max_length: int = 4000,
    ) -> list[list[dict]]:
        """按递增步骤收集文档子集（用于 Exp5 演化实验）。

        Args:
            steps: 每步新增的文件/目录路径列表，如:
                   [["README.md"], ["CONTRIBUTING.md"], [".github/"], ...]
        Returns:
            累计文档列表的列表，len == len(steps)
        """
        repo = Path(repo_path)
        cumulative: list[dict] = []
        result: list[list[dict]] = []
        seen_paths: set[str] = set()

        for step_sources in steps:
            for src in step_sources:
                spath = repo / src
                if spath.is_file():
                    rel = str(spath.relative_to(repo))
                    if rel not in seen_paths:
                        content = self._read_file(spath, max_length)
                        if content:
                            cumulative.append({
                                "source_type": "incremental",
                                "file_path": rel,
                                "content": content,
                            })
                            seen_paths.add(rel)
                elif spath.is_dir():
                    for fpath in sorted(spath.rglob("*")):
                        if fpath.is_file() and fpath.suffix.lower() in _DOC_EXTENSIONS:
                            if ".git" in fpath.parts:
                                continue
                            rel = str(fpath.relative_to(repo))
                            if rel not in seen_paths:
                                content = self._read_file(fpath, max_length)
                                if content:
                                    cumulative.append({
                                        "source_type": "incremental",
                                        "file_path": rel,
                                        "content": content,
                                    })
                                    seen_paths.add(rel)
            # 每步返回当前累计快照的副本
            result.append(list(cumulative))

        return result

    # ------------------------------------------------------------------
    # Profile 生成
    # ------------------------------------------------------------------

    def collect_code_samples(
        self,
        repo_path: str,
        max_files: int = 30,
        max_length: int = 2000,
    ) -> list[dict]:
        """从项目中收集代表性代码样本，用于 Stage 2 代码分析。

        策略：按目录多样性 + 文件重要性选取代表性文件。

        Returns:
            [{file_path, content, language}]
        """
        repo = Path(repo_path)
        if not repo.exists():
            return []

        # 收集所有代码文件
        all_code_files = []
        for ext in _CODE_EXTENSIONS:
            all_code_files.extend(repo.rglob(f"*{ext}"))

        if not all_code_files:
            return []

        # 过滤掉测试、生成代码、vendor 等
        skip_patterns = {"test", "tests", "__tests__", "vendor", "node_modules",
                         "build", "dist", "generated", "mock", "fixture"}

        def is_useful(f: Path) -> bool:
            parts_lower = [p.lower() for p in f.parts]
            if any(skip in parts_lower for skip in skip_patterns):
                return False
            # 跳过过大的文件（可能是生成的）
            try:
                if f.stat().st_size > 100_000:
                    return False
            except OSError:
                return False
            return True

        useful_files = [f for f in all_code_files if is_useful(f)]

        # 按目录分组，确保多样性
        dir_groups: dict[str, list[Path]] = {}
        for f in useful_files:
            rel_dir = str(f.parent.relative_to(repo))
            dir_groups.setdefault(rel_dir, []).append(f)

        # 从每个目录组中选文件，优先选较短、命名有代表性的
        selected = []
        seen_dirs = set()

        def file_priority(f: Path) -> int:
            name = f.stem.lower()
            # 核心模块文件优先
            for kw in ["main", "core", "base", "service", "manager", "handler",
                       "controller", "model", "config", "util", "helper"]:
                if kw in name:
                    return 0
            return 1

        for dir_name in sorted(dir_groups.keys(), key=lambda d: d.count("/")):
            files = sorted(dir_groups[dir_name], key=file_priority)
            # 每个目录最多取 3 个文件
            for f in files[:3]:
                if len(selected) >= max_files:
                    break
                selected.append(f)
            if len(selected) >= max_files:
                break

        # 读取内容
        code_samples = []
        for f in selected:
            content = self._read_file(f, max_length)
            if content:
                rel = str(f.relative_to(repo))
                ext = f.suffix.lstrip(".")
                lang_map = {"java": "Java", "kt": "Kotlin", "py": "Python",
                            "go": "Go", "ts": "TypeScript", "js": "JavaScript",
                            "rs": "Rust", "c": "C", "cpp": "C++", "swift": "Swift"}
                code_samples.append({
                    "file_path": rel,
                    "content": content,
                    "language": lang_map.get(ext, ext),
                })

        logger.info(f"从 {repo_path} 收集到 {len(code_samples)} 个代码样本 "
                    f"(共 {len(useful_files)} 个有效代码文件)")
        return code_samples

    def generate_profile(
        self,
        repo_name: str,
        model_key: str,
        documents: list[dict],
        run_id: str = "default",
        force_api: bool = False,
        repo_path: Optional[str] = None,
        use_code_evidence: bool = True,
    ) -> dict:
        """用指定 LLM 从文档集 + 代码样本生成 ValueProfile（v2 两阶段）。

        Stage 1: 基于文档的 stated values（原有逻辑）
        Stage 2: 基于代码样本的 demonstrated values（新增）
        Merge: 加权合并两个信号

        Returns:
            {repo, model, l2_scores, l3_scores, core_values, evidence, raw_response}
        """
        # 检查缓存（v2 缓存 key 包含 _v2 后缀以区分）
        cache_key = self._cache_key(repo_name, model_key, documents, run_id)
        if use_code_evidence:
            cache_key = cache_key.replace("_default", "_v2_default")
        if not force_api:
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.info(f"[缓存命中] {model_key}/{repo_name} (run={run_id}, v2={use_code_evidence})")
                return cached

        client = self.llm_clients.get(model_key)
        if client is None:
            raise ValueError(f"LLM 客户端不存在: {model_key}")

        # ========== Stage 1: 文档分析（stated values）==========
        doc_text = self._format_documents(documents)

        user_prompt_s1 = (
            f"## Project: {repo_name}\n\n"
            f"## Project Documentation\n{doc_text}\n\n"
        )
        if self.value_model_text:
            user_prompt_s1 += f"## Value Model Reference\n{self.value_model_text}\n\n"
        user_prompt_s1 += (
            "Please analyze the above project documentation and produce the "
            "value profile as specified in your instructions."
        )

        logger.info(f"[Stage 1/2 - Doc] {model_key} → {repo_name} (run={run_id})")
        response_s1 = client.call(VALUE_CLASSIFICATION_SYSTEM, user_prompt_s1)

        if response_s1.error:
            logger.error(f"Stage 1 LLM 调用失败: {response_s1.error}")
            return self._empty_profile(repo_name, model_key, error=response_s1.error)

        parsed_s1 = response_s1.parsed_result or {}
        profile_s1 = self._build_profile(repo_name, model_key, parsed_s1, response_s1.raw_response or "")

        # 如果不使用代码证据，直接返回文档分析结果
        if not use_code_evidence or not repo_path:
            self._save_cache(cache_key, profile_s1)
            return profile_s1

        # ========== Stage 2: 代码分析（demonstrated values）==========
        code_samples = self.collect_code_samples(repo_path)
        if not code_samples:
            logger.warning(f"未收集到代码样本，跳过 Stage 2")
            self._save_cache(cache_key, profile_s1)
            return profile_s1

        code_text = self._format_code_samples(code_samples)
        user_prompt_s2 = (
            f"## Project: {repo_name}\n\n"
            f"## Representative Source Code\n{code_text}\n\n"
        )
        if self.value_model_text:
            user_prompt_s2 += f"## Value Model Reference\n{self.value_model_text}\n\n"
        user_prompt_s2 += (
            "Analyze the ACTUAL CODE above (not documentation) and rate how strongly "
            "each value is demonstrated through implementation patterns. "
            "Be discriminating — only rate high what the code CLEARLY shows."
        )

        logger.info(f"[Stage 2/2 - Code] {model_key} → {repo_name} "
                    f"({len(code_samples)} code samples)")
        response_s2 = client.call(CODE_VALUE_ANALYSIS_SYSTEM, user_prompt_s2)

        if response_s2.error:
            logger.error(f"Stage 2 LLM 调用失败: {response_s2.error}，使用 Stage 1 结果")
            self._save_cache(cache_key, profile_s1)
            return profile_s1

        parsed_s2 = response_s2.parsed_result or {}
        profile_s2 = self._build_profile(repo_name, model_key, parsed_s2, response_s2.raw_response or "")

        # ========== Merge: 加权合并 ==========
        # 代码证据权重 0.6（代码行为比文档声明更可靠）
        # 文档声明权重 0.4
        CODE_WEIGHT = 0.6
        DOC_WEIGHT = 0.4

        merged_l2 = {}
        for vid in L2_IDS:
            s1 = profile_s1["l2_scores"].get(vid, 0.5)
            s2 = profile_s2["l2_scores"].get(vid, 0.5)
            merged_l2[vid] = round(s1 * DOC_WEIGHT + s2 * CODE_WEIGHT, 4)

        merged_l3 = {}
        for vid in L3_IDS:
            s1 = profile_s1["l3_scores"].get(vid, 0.5)
            s2 = profile_s2["l3_scores"].get(vid, 0.5)
            merged_l3[vid] = round(s1 * DOC_WEIGHT + s2 * CODE_WEIGHT, 4)

        # Score normalization: 将分数映射到 [0.1, 0.95] 区间，避免极端值
        merged_l2 = self._normalize_scores(merged_l2)
        merged_l3 = self._normalize_scores(merged_l3)

        # 重新计算 core values
        combined = {**merged_l2, **merged_l3}
        sorted_vals = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        core_values = [v[0] for v in sorted_vals[:3]]

        # 合并 evidence
        evidence = parsed_s1.get("evidence", [])[:3] + parsed_s2.get("evidence", [])[:3]

        merged_profile = {
            "repo": repo_name,
            "model": model_key,
            "l2_scores": merged_l2,
            "l3_scores": merged_l3,
            "core_values": core_values,
            "evidence": evidence,
            "raw_response": f"[S1]{response_s1.raw_response[:200]}...\n[S2]{response_s2.raw_response[:200]}...",
            "_generation_mode": "v2_doc+code",
            "_doc_scores": profile_s1["l2_scores"] | profile_s1["l3_scores"],
            "_code_scores": profile_s2["l2_scores"] | profile_s2["l3_scores"],
        }

        self._save_cache(cache_key, merged_profile)
        logger.info(f"[Profile v2] {repo_name}: core={core_values}, "
                    f"L2 range=[{min(merged_l2.values()):.2f},{max(merged_l2.values()):.2f}], "
                    f"L3 range=[{min(merged_l3.values()):.2f},{max(merged_l3.values()):.2f}]")
        return merged_profile

    def generate_profiles_multi_model(
        self,
        repo_name: str,
        documents: list[dict],
        model_keys: Optional[list[str]] = None,
        force_api: bool = False,
    ) -> dict[str, dict]:
        """对同一项目用多个 LLM 生成 Profile。"""
        keys = model_keys or list(self.llm_clients.keys())
        results = {}
        for mk in keys:
            try:
                results[mk] = self.generate_profile(
                    repo_name, mk, documents, force_api=force_api
                )
            except Exception as e:
                logger.error(f"{mk} 生成失败: {e}")
                results[mk] = self._empty_profile(repo_name, mk, error=str(e))
        return results

    def generate_profile_n_runs(
        self,
        repo_name: str,
        model_key: str,
        documents: list[dict],
        n_runs: int = 5,
        force_api: bool = False,
    ) -> list[dict]:
        """同一模型多次运行，用于稳定性评估。"""
        profiles = []
        for i in range(n_runs):
            p = self.generate_profile(
                repo_name, model_key, documents,
                run_id=f"run_{i}",
                force_api=force_api,
            )
            profiles.append(p)
        return profiles

    def generate_profile_incremental(
        self,
        repo_name: str,
        model_key: str,
        doc_subsets: list[list[dict]],
        force_api: bool = False,
    ) -> list[dict]:
        """逐步增加文档子集，返回 Profile 序列（用于 Exp5）。"""
        profiles = []
        for i, docs in enumerate(doc_subsets):
            p = self.generate_profile(
                repo_name, model_key, docs,
                run_id=f"step_{i}",
                force_api=force_api,
            )
            profiles.append(p)
        return profiles

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _read_file(path: Path, max_length: int = 4000) -> Optional[str]:
        """安全读取文件，截断到 max_length。"""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) > max_length:
                text = text[:max_length] + "\n\n[... truncated ...]"
            return text.strip() if text.strip() else None
        except Exception:
            return None

    @staticmethod
    def _format_documents(documents: list[dict]) -> str:
        """格式化文档内容，优先使用摘要（如果有）。"""
        parts = []
        for doc in documents:
            # 优先使用 summary 字段（如果已摘要），否则使用 content
            content = doc.get('summary', doc['content'])
            parts.append(f"### {doc['file_path']}\n```\n{content}\n```\n")
        return "\n".join(parts)

    @staticmethod
    def _format_code_samples(code_samples: list[dict]) -> str:
        """格式化代码样本为 LLM 可读的文本。"""
        parts = []
        for sample in code_samples:
            lang = sample.get('language', 'code')
            parts.append(
                f"### {sample['file_path']} ({lang})\n"
                f"```{lang.lower()}\n{sample['content']}\n```\n"
            )
        return "\n".join(parts)

    @staticmethod
    def _normalize_scores(
        scores: dict[str, float],
        lo: float = 0.3,
        hi: float = 0.8,
    ) -> dict[str, float]:
        """将分数归一化到 [lo, hi] 区间，减少极端差异。

        使用 min-max 归一化 + 线性映射，确保分数差异化且不集中在两端。
        默认范围 [0.3, 0.8] 比原来的 [0.1, 0.95] 更紧凑，减少价值偏向性。
        """
        values = list(scores.values())
        if not values:
            return scores

        vmin = min(values)
        vmax = max(values)

        if vmax - vmin < 0.01:
            # 所有分数几乎相同，返回中值
            mid = (lo + hi) / 2
            return {k: round(mid, 4) for k in scores}

        normalized = {}
        for k, v in scores.items():
            # 线性映射到 [lo, hi]
            normalized[k] = round(lo + (v - vmin) / (vmax - vmin) * (hi - lo), 4)

        return normalized

    def _build_profile(
        self, repo_name: str, model_key: str, parsed: dict, raw: str
    ) -> dict:
        l2 = parsed.get("l2_scores", {})
        l3 = parsed.get("l3_scores", {})

        # 确保所有 20 维都有值
        l2_scores = {vid: float(l2.get(vid, 0.0)) for vid in L2_IDS}
        l3_scores = {vid: float(l3.get(vid, 0.0)) for vid in L3_IDS}

        # core values
        combined = {**l2_scores, **l3_scores}
        sorted_vals = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        core_values = parsed.get("core_values", [v[0] for v in sorted_vals[:3]])

        return {
            "repo": repo_name,
            "model": model_key,
            "l2_scores": l2_scores,
            "l3_scores": l3_scores,
            "core_values": core_values,
            "evidence": parsed.get("evidence", []),
            "raw_response": raw,
        }

    def _empty_profile(self, repo_name: str, model_key: str, error: str = "") -> dict:
        return {
            "repo": repo_name,
            "model": model_key,
            "l2_scores": {vid: 0.0 for vid in L2_IDS},
            "l3_scores": {vid: 0.0 for vid in L3_IDS},
            "core_values": [],
            "evidence": [],
            "raw_response": "",
            "error": error,
        }

    # ------------------------------------------------------------------
    # 缓存
    # ------------------------------------------------------------------

    def _cache_key(
        self, repo: str, model: str, documents: list[dict], run_id: str
    ) -> str:
        content_hash = hashlib.md5(
            json.dumps([d["file_path"] for d in documents], sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"{model}/{repo}_{content_hash}_{run_id}"

    def _cache_path(self, key: str) -> Path:
        p = self.cache_dir / f"{key}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _load_cache(self, key: str) -> Optional[dict]:
        p = self._cache_path(key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, data: dict) -> None:
        p = self._cache_path(key)
        # 不缓存 raw_response（太大）
        to_save = {k: v for k, v in data.items() if k != "raw_response"}
        to_save["_cached"] = True
        p.write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")
    
    def _load_cache_string(self, key: str) -> Optional[str]:
        """加载字符串缓存（用于文档摘要）。"""
        p = self._cache_path(key)
        if p.exists():
            try:
                content = json.loads(p.read_text(encoding="utf-8"))
                # 兼容旧格式（字典）和新格式（字符串）
                if isinstance(content, dict):
                    return content.get('summary', content.get('content'))
                return str(content)
            except Exception:
                return None
        return None
    
    def _save_cache_string(self, key: str, text: str) -> None:
        """保存字符串缓存（用于文档摘要）。"""
        p = self._cache_path(key)
        p.write_text(json.dumps(text, ensure_ascii=False), encoding="utf-8")
