#!/usr/bin/env python
"""
Profile generation engine — a lightweight standalone module independent of the full ValueGuard runtime.
Reuses experiment/llm_client.py for LLM calls and the prompt template from profiler_agent.py.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt (consistent with VALUE_CLASSIFICATION_SYSTEM in profiler_agent.py)
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

# File patterns and directory suffixes targeted during document collection
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
    """Lightweight Value Profile generator."""

    def __init__(
        self,
        llm_clients: dict,
        value_model_text: str = "",
        cache_dir: str = "experiment_logs/profile_cache",
    ):
        self.llm_clients = llm_clients  # {model_key: BaseLLMClient}
        self.value_model_text = value_model_text
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Document Collection
    # ------------------------------------------------------------------

    def collect_documents(
        self,
        repo_path: str,
        sources: Optional[list[str]] = None,
        max_length: int = 4000,
    ) -> list[dict]:
        """Recursively collect documents from a project directory.

        Returns:
            [{source_type, file_path, content}]
        """
        repo = Path(repo_path)
        if not repo.exists():
            logger.warning(f"Repository path does not exist: {repo_path}")
            return []

        collected: list[dict] = []

        # 1. Known document files in the root directory
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

        # 2. Known documentation directories
        for dname in _DOC_DIRS:
            dpath = repo / dname
            if dpath.is_dir():
                for fpath in sorted(dpath.rglob("*")):
                    if fpath.is_file() and fpath.suffix.lower() in _DOC_EXTENSIONS:
                        # Skip files inside .git
                        if ".git" in fpath.parts:
                            continue
                        content = self._read_file(fpath, max_length)
                        if content:
                            collected.append({
                                "source_type": "doc_dir",
                                "file_path": str(fpath.relative_to(repo)),
                                "content": content,
                            })

        # 3. Include additional sources if specified
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

        logger.info(f"Collected {len(collected)} documents from {repo_path}")
        return collected

    def summarize_documents(
        self,
        documents: list[dict],
        model_key: str,
        max_summary_length: int = 150,  # Reduced to 150 characters
        batch_size: int = 10,
        max_docs: int = 100,  # Maximum number of documents to retain
    ) -> list[dict]:
        """Summarize and compress a large number of documents to reduce context length.
        
        Args:
            documents: Original document list [{source_type, file_path, content}]
            model_key: LLM model used for summarization
            max_summary_length: Maximum character count per document summary
            batch_size: Batch size (to avoid excessive API calls)
            max_docs: Maximum number of documents to retain (for very large projects)
        
        Returns:
            Summarized document list [{source_type, file_path, content, summary}]
        """
        if not documents:
            return []
        
        # If the number of documents is small (<20), return the original documents directly
        if len(documents) < 20:
            for doc in documents:
                doc['summary'] = doc['content']  # No summarization
            return documents
        
        logger.info(f"Starting summarization of {len(documents)} documents...")
        
        # ========== Phase 1: Document importance ranking and selection ==========
        priority_order = {
            'README.md': 1, 'README.rst': 1, 'README.txt': 1, 'README': 1,
            'CONTRIBUTING.md': 2, 'CONTRIBUTING.rst': 2,
            'CODE_OF_CONDUCT.md': 3, 'SECURITY.md': 4,
            'LICENSE': 5, 'LICENSE.md': 5,
        }
        
        def get_priority(doc: dict) -> int:
            filename = Path(doc['file_path']).name
            base_name = Path(doc['file_path']).stem.lower()
            
            # Root-level core documents have the highest priority
            if doc['file_path'] in priority_order:
                return priority_order[doc['file_path']]
            
            # Determine priority by filename
            for key in priority_order:
                if base_name == Path(key).stem.lower():
                    return priority_order[key] + 10
            
            # Workflow files in the .github directory have lower priority
            if '.github/workflows' in doc['file_path']:
                return 999
            
            # Sort docs directory files by path depth
            if doc['file_path'].startswith('docs/'):
                depth = doc['file_path'].count('/')
                return 100 + depth
            
            # Other documents
            return 500
        
        # Sort by priority and select the top N most important documents
        sorted_docs = sorted(documents, key=get_priority)
        selected_docs = sorted_docs[:max_docs]
        
        if len(selected_docs) < len(documents):
            logger.info(f"Document selection: selected the {len(selected_docs)} most important out of {len(documents)}")
        
        # ========== Phase 2: Summarize the selected documents ==========
        client = self.llm_clients.get(model_key)
        if client is None:
            logger.warning(f"LLM client does not exist: {model_key}, using original documents")
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
                # Check if a cached summary already exists
                cache_key = f"doc_summary_{hashlib.md5(doc['content'].encode()).hexdigest()}"
                cached_summary = self._load_cache_string(cache_key)
                
                if cached_summary:
                    doc['summary'] = cached_summary
                    summarized_docs.append(doc)
                    continue
                
                # Call LLM to generate summary
                prompt = summary_prompt_template.format(
                    text=doc['content'][:6000],  # Limit input length
                    max_length=max_summary_length
                )
                
                response = client.call(
                    system_prompt="You are a technical documentation summarizer specializing in extracting project values.",
                    user_prompt=prompt,
                )
                
                summary = response.raw_response.strip()
                doc['summary'] = summary
                
                # Cache the summary (as a string)
                self._save_cache_string(cache_key, summary)
                
                summarized_docs.append(doc)
                
                if (i + 1) % 20 == 0:
                    logger.info(f"Summarized {i+1}/{len(selected_docs)} documents")
                    
            except Exception as e:
                logger.warning(f"Document summarization failed: {doc['file_path']}, using original document - {e}")
                doc['summary'] = doc['content']
                summarized_docs.append(doc)
        
        total_original = sum(len(d['content']) for d in summarized_docs)
        total_summarized = sum(len(d['summary']) for d in summarized_docs)
        logger.info(f"Document summarization complete! Selected {len(summarized_docs)} documents, "
                   f"original total chars: {total_original:,} -> summarized total chars: {total_summarized:,}, "
                   f"compression ratio: {(1 - total_summarized/total_original)*100:.1f}%")
        
        return summarized_docs

    def collect_documents_incremental(
        self,
        repo_path: str,
        steps: list[list[str]],
        max_length: int = 4000,
    ) -> list[list[dict]]:
        """Collect document subsets in incremental steps (for Exp5 evolution experiment).

        Args:
            steps: List of file/directory paths added at each step, e.g.:
                   [["README.md"], ["CONTRIBUTING.md"], [".github/"], ...]
        Returns:
            List of cumulative document lists, len == len(steps)
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
            # Return a copy of the current cumulative snapshot at each step
            result.append(list(cumulative))

        return result

    # ------------------------------------------------------------------
    # Profile Generation
    # ------------------------------------------------------------------

    def generate_profile(
        self,
        repo_name: str,
        model_key: str,
        documents: list[dict],
        run_id: str = "default",
        force_api: bool = False,
    ) -> dict:
        """Generate a ValueProfile from a document set using the specified LLM.

        Returns:
            {repo, model, l2_scores, l3_scores, core_values, evidence, raw_response}
        """
        # Check cache
        cache_key = self._cache_key(repo_name, model_key, documents, run_id)
        if not force_api:
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.info(f"[Cache hit] {model_key}/{repo_name} (run={run_id})")
                return cached

        client = self.llm_clients.get(model_key)
        if client is None:
            raise ValueError(f"LLM client does not exist: {model_key}")

        # Concatenate document content
        doc_text = self._format_documents(documents)

        user_prompt = (
            f"## Project: {repo_name}\n\n"
            f"## Project Documentation\n{doc_text}\n\n"
        )
        if self.value_model_text:
            user_prompt += f"## Value Model Reference\n{self.value_model_text}\n\n"
        user_prompt += (
            "Please analyze the above project documentation and produce the "
            "value profile as specified in your instructions."
        )

        logger.info(f"[API call] {model_key} -> {repo_name} (run={run_id})")
        response = client.call(VALUE_CLASSIFICATION_SYSTEM, user_prompt)

        if response.error:
            logger.error(f"LLM call failed: {response.error}")
            return self._empty_profile(repo_name, model_key, error=response.error)

        parsed = response.parsed_result or {}
        profile = self._build_profile(repo_name, model_key, parsed, response.raw_response)

        # Write to cache
        self._save_cache(cache_key, profile)
        return profile

    def generate_profiles_multi_model(
        self,
        repo_name: str,
        documents: list[dict],
        model_keys: Optional[list[str]] = None,
        force_api: bool = False,
    ) -> dict[str, dict]:
        """Generate profiles for the same project using multiple LLMs."""
        keys = model_keys or list(self.llm_clients.keys())
        results = {}
        for mk in keys:
            try:
                results[mk] = self.generate_profile(
                    repo_name, mk, documents, force_api=force_api
                )
            except Exception as e:
                logger.error(f"{mk} generation failed: {e}")
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
        """Run the same model multiple times for stability assessment."""
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
        """Incrementally add document subsets and return a sequence of profiles (for Exp5)."""
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
    # Internal Methods
    # ------------------------------------------------------------------

    @staticmethod
    def _read_file(path: Path, max_length: int = 4000) -> Optional[str]:
        """Safely read a file, truncating to max_length."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) > max_length:
                text = text[:max_length] + "\n\n[... truncated ...]"
            return text.strip() if text.strip() else None
        except Exception:
            return None

    @staticmethod
    def _format_documents(documents: list[dict]) -> str:
        """Format document content, preferring summaries if available."""
        parts = []
        for doc in documents:
            # Prefer the summary field (if summarized), otherwise use content
            content = doc.get('summary', doc['content'])
            parts.append(f"### {doc['file_path']}\n```\n{content}\n```\n")
        return "\n".join(parts)

    def _build_profile(
        self, repo_name: str, model_key: str, parsed: dict, raw: str
    ) -> dict:
        l2 = parsed.get("l2_scores", {})
        l3 = parsed.get("l3_scores", {})

        # Ensure all 20 dimensions have values
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
    # Cache
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
        # Do not cache raw_response (too large)
        to_save = {k: v for k, v in data.items() if k != "raw_response"}
        to_save["_cached"] = True
        p.write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")
    
    def _load_cache_string(self, key: str) -> Optional[str]:
        """Load a string cache entry (used for document summaries)."""
        p = self._cache_path(key)
        if p.exists():
            try:
                content = json.loads(p.read_text(encoding="utf-8"))
                # Compatible with old format (dict) and new format (string)
                if isinstance(content, dict):
                    return content.get('summary', content.get('content'))
                return str(content)
            except Exception:
                return None
        return None
    
    def _save_cache_string(self, key: str, text: str) -> None:
        """Save a string cache entry (used for document summaries)."""
        p = self._cache_path(key)
        p.write_text(json.dumps(text, ensure_ascii=False), encoding="utf-8")
