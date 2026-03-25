"""Evidence agent for locating and verifying code evidence."""

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

from valueguard.agents.base_agent import BaseAgent
from valueguard.core.models import (
    ASTTrace,
    EvidencePiece,
    EvidenceResult,
    EvidenceStatus,
    EvidenceTask,
    ValueHypothesis,
)
from valueguard.memory.manager import MemoryManager
from valueguard.skills.registry import SkillRegistry


# System prompt for evidence verification
EVIDENCE_VERIFICATION_SYSTEM = """You are a code evidence verifier. Your task is to determine if a code change actually supports a given value deviation hypothesis.

## Your Role
- Analyze the code change (diff) and the hypothesis claim
- Determine if the code ACTUALLY demonstrates the claimed value deviation
- Be SKEPTICAL - many hypotheses are false positives

## Verification Criteria
1. CONFIRMED: The code clearly shows the claimed deviation with concrete evidence
   - Example: Hypothesis claims "privacy risk from logging user data", code shows `log.info(user.email)`
   
2. REJECTED: The code does NOT support the hypothesis claim
   - Example: Hypothesis claims "security risk", but code is just UI styling changes
   - Example: Hypothesis is too vague or generic with no specific code evidence
   
3. UNVERIFIED: Cannot determine - need more context or code is ambiguous

## Output Format (JSON):
{
  "status": "CONFIRMED|REJECTED|UNVERIFIED",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of your decision",
  "evidence_snippet": "Most relevant code line(s) if CONFIRMED, empty if REJECTED"
}

Be strict! Most generic hypotheses should be REJECTED. Only CONFIRM when you see clear, specific evidence."""


class EvidenceAgent(BaseAgent):
    """Agent for locating and verifying code evidence for hypotheses.

    Uses multiple strategies:
    - LLM-based verification (primary)
    - Vector search for semantic similarity
    - AST analysis for structural verification
    - Pattern matching for known indicators
    """

    name = "evidence"
    role = "Code Evidence Locator"
    goal = "Verify value hypotheses with precise code evidence"

    def __init__(
        self,
        skills: SkillRegistry,
        memory: MemoryManager,
        config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(skills, memory, config)
        self._search_depth = self.get_config("search_depth", 3)
        self._top_k = self.get_config("top_k", 10)
        self._llm_provider = self.get_config("llm_provider", "deepseek")

    def execute(self, task: EvidenceTask) -> EvidenceResult:
        """Locate evidence for a hypothesis.

        Args:
            task: EvidenceTask with hypothesis and repo path

        Returns:
            EvidenceResult with evidence pieces and status
        """
        hypothesis = task.hypothesis
        evidence_pieces = []

        # Strategy 1: Extract code from the diff hunk itself
        if hypothesis.diff_hunk:
            hunk_evidence = self._extract_hunk_evidence(hypothesis)
            evidence_pieces.extend(hunk_evidence)

        # Strategy 2: Vector search for related code (if available)
        if self.has_skill("vector_search"):
            vector_evidence = self._search_vector(hypothesis, task.repo_path)
            evidence_pieces.extend(vector_evidence)

        # Strategy 3: AST analysis for verification (if available)
        if self.has_skill("ast_analysis") and evidence_pieces:
            evidence_pieces = self._verify_with_ast(
                evidence_pieces, hypothesis, task.repo_path, task.search_depth
            )

        # Strategy 4: Pattern-based search
        pattern_evidence = self._search_patterns(hypothesis, task.repo_path)
        evidence_pieces.extend(pattern_evidence)

        # Deduplicate
        unique_evidence = self._deduplicate_evidence(evidence_pieces)

        # PRIMARY VERIFICATION: Use LLM to verify hypothesis against code
        if self.has_skill("llm_call") and hypothesis.diff_hunk:
            status, verification_result = self._verify_with_llm(hypothesis)
            
            # Update evidence based on LLM verification
            if verification_result.get("evidence_snippet"):
                # Create evidence piece from LLM-identified snippet
                snippet_evidence = EvidencePiece(
                    file_path=hypothesis.diff_hunk.file_path,
                    start_line=hypothesis.diff_hunk.new_start,
                    end_line=hypothesis.diff_hunk.new_start + 10,
                    snippet=verification_result.get("evidence_snippet", ""),
                    relevance_score=verification_result.get("confidence", 0.5),
                )
                unique_evidence = [snippet_evidence] + unique_evidence[:2]
            
            return EvidenceResult(
                hypothesis_id=hypothesis.id,
                status=status,
                evidence_pieces=unique_evidence[:3],
                search_metadata={
                    "strategies_used": self._get_strategies_used() + ["llm_verification"],
                    "total_candidates": len(evidence_pieces),
                    "llm_reasoning": verification_result.get("reasoning", ""),
                    "llm_confidence": verification_result.get("confidence", 0.0),
                },
            )

        # Fallback: Use heuristic ranking if no LLM available
        ranked_evidence = self._rank_evidence(unique_evidence, hypothesis)
        status = self._determine_status_heuristic(ranked_evidence)

        return EvidenceResult(
            hypothesis_id=hypothesis.id,
            status=status,
            evidence_pieces=ranked_evidence[:3],
            search_metadata={
                "strategies_used": self._get_strategies_used(),
                "total_candidates": len(evidence_pieces),
            },
        )

    def _verify_with_llm(self, hypothesis: ValueHypothesis) -> tuple[EvidenceStatus, dict]:
        """Use LLM to verify if hypothesis is supported by code evidence."""
        import json
        
        # Build verification prompt
        code_content = hypothesis.diff_hunk.content if hypothesis.diff_hunk else ""
        file_path = hypothesis.diff_hunk.file_path if hypothesis.diff_hunk else ""
        
        logger.info(f"🔍 [LLM Verification] Verifying hypothesis for {file_path}")
        
        user_prompt = f"""## Hypothesis to Verify
- Value ID: {hypothesis.value_id}
- Deviation Type: {hypothesis.deviation_type}
- Description: {hypothesis.description}
- Suggested Action: {hypothesis.suggested_action}

## Code Change to Analyze
File: {file_path}
```diff
{code_content}
```

Analyze this code change and determine if it actually demonstrates the claimed value deviation.
Return your analysis as JSON."""

        try:
            logger.info(f"→ Calling LLM provider: {self._llm_provider}")
            response = self.invoke_skill(
                "llm_call",
                user=user_prompt,
                system=EVIDENCE_VERIFICATION_SYSTEM,
                provider=self._llm_provider,
                temperature=0.0,
                parse_json=True,
            )
            
            if response and response.parsed_result:
                result = response.parsed_result
                status_str = result.get("status", "UNVERIFIED").upper()
                
                if status_str == "CONFIRMED":
                    status = EvidenceStatus.CONFIRMED
                elif status_str == "REJECTED":
                    status = EvidenceStatus.REJECTED
                else:
                    status = EvidenceStatus.UNVERIFIED
                    
                return status, result
            else:
                # Parse raw response if JSON parsing failed
                raw = response.raw_response if response else ""
                if "REJECTED" in raw.upper():
                    return EvidenceStatus.REJECTED, {"reasoning": raw, "confidence": 0.3}
                elif "CONFIRMED" in raw.upper():
                    return EvidenceStatus.CONFIRMED, {"reasoning": raw, "confidence": 0.7}
                else:
                    return EvidenceStatus.UNVERIFIED, {"reasoning": raw, "confidence": 0.5}
                    
        except Exception as e:
            # On error, default to UNVERIFIED
            return EvidenceStatus.UNVERIFIED, {"reasoning": f"Verification error: {str(e)}", "confidence": 0.0}

    def _determine_status_heuristic(self, ranked_evidence: list[EvidencePiece]) -> EvidenceStatus:
        """Determine status using heuristics (fallback when LLM not available)."""
        if not ranked_evidence:
            return EvidenceStatus.UNVERIFIED
            
        # More conservative thresholds
        top_score = ranked_evidence[0].relevance_score if ranked_evidence else 0
        
        # Require AST verification or very high score for confirmation
        has_ast_verification = any(e.ast_trace for e in ranked_evidence)
        
        if has_ast_verification and top_score >= 0.8:
            return EvidenceStatus.CONFIRMED
        elif top_score >= 0.9:
            return EvidenceStatus.CONFIRMED
        elif top_score < 0.4:
            return EvidenceStatus.REJECTED
        else:
            return EvidenceStatus.UNVERIFIED

    def _extract_hunk_evidence(
        self, hypothesis: ValueHypothesis
    ) -> list[EvidencePiece]:
        """Extract evidence from the hypothesis's diff hunk."""
        evidence = []

        if hypothesis.diff_hunk:
            hunk = hypothesis.diff_hunk

            # Find the most relevant lines in the hunk
            lines = hunk.content.split("\n")
            added_lines = []
            for i, line in enumerate(lines):
                if line.startswith("+") and not line.startswith("+++"):
                    # Calculate line number in new file
                    added_lines.append((hunk.new_start + i, line[1:].strip()))

            if added_lines:
                # Create evidence piece from added lines
                start_line = added_lines[0][0]
                end_line = added_lines[-1][0]
                snippet = "\n".join(line for _, line in added_lines[:10])

                evidence.append(
                    EvidencePiece(
                        file_path=hunk.file_path,
                        start_line=start_line,
                        end_line=end_line,
                        snippet=snippet,
                        # Lower default score - LLM verification will adjust
                        relevance_score=0.5,
                    )
                )

        return evidence

    def _search_vector(
        self, hypothesis: ValueHypothesis, repo_path: str
    ) -> list[EvidencePiece]:
        """Search for evidence using vector similarity."""
        evidence = []

        try:
            # Build search query from hypothesis
            query = self._build_search_query(hypothesis)

            # Ensure index exists
            self.invoke_skill(
                "vector_search",
                action="index",
                repo_path=repo_path,
            )

            # Search
            results = self.invoke_skill(
                "vector_search",
                action="search",
                query=query,
                top_k=self._top_k,
            )

            for result in results:
                evidence.append(
                    EvidencePiece(
                        file_path=result.get("file_path", ""),
                        start_line=result.get("start_line", 0),
                        end_line=result.get("end_line", 0),
                        snippet=result.get("content", ""),
                        relevance_score=result.get("score", 0.5),
                    )
                )

        except Exception:
            pass

        return evidence

    def _verify_with_ast(
        self,
        evidence_pieces: list[EvidencePiece],
        hypothesis: ValueHypothesis,
        repo_path: str,
        search_depth: int,
    ) -> list[EvidencePiece]:
        """Verify evidence using AST analysis."""
        verified = []

        for piece in evidence_pieces:
            try:
                result = self.invoke_skill(
                    "ast_analysis",
                    file_path=f"{repo_path}/{piece.file_path}",
                    hypothesis=hypothesis,
                    search_depth=search_depth,
                )

                if result.get("verified", False):
                    # Add AST trace to evidence
                    ast_trace = None
                    if result.get("call_chain"):
                        ast_trace = ASTTrace(
                            function_name=result.get("function_name", ""),
                            file_path=piece.file_path,
                            line_number=piece.start_line,
                            call_chain=result.get("call_chain", []),
                        )

                    verified_piece = EvidencePiece(
                        file_path=piece.file_path,
                        start_line=result.get("lines", [piece.start_line])[0],
                        end_line=result.get("lines", [piece.end_line])[-1],
                        snippet=result.get("snippet", piece.snippet),
                        relevance_score=min(1.0, piece.relevance_score + 0.2),
                        ast_trace=ast_trace,
                    )
                    verified.append(verified_piece)
                else:
                    # Keep unverified but lower score
                    piece.relevance_score = max(0.0, piece.relevance_score - 0.2)
                    verified.append(piece)

            except Exception:
                verified.append(piece)

        return verified

    def _search_patterns(
        self, hypothesis: ValueHypothesis, repo_path: str
    ) -> list[EvidencePiece]:
        """Search for evidence using known patterns."""
        evidence = []

        # Get patterns related to this value
        patterns = self.memory.get_relevant_patterns(value_id=hypothesis.value_id)

        for pattern in patterns[:3]:
            # Build regex from pattern
            regex = self._pattern_to_regex(pattern.code_pattern)
            if regex:
                matches = self._grep_pattern(regex, repo_path, hypothesis)
                for match in matches[:2]:
                    match.relevance_score = pattern.confidence
                    evidence.append(match)

        return evidence

    def _build_search_query(self, hypothesis: ValueHypothesis) -> str:
        """Build a search query from hypothesis."""
        parts = []

        if hypothesis.value_id:
            parts.append(hypothesis.value_id)

        if hypothesis.description:
            # Extract key terms from description
            words = hypothesis.description.split()
            key_words = [
                w
                for w in words
                if len(w) > 4 and w.lower() not in ("should", "could", "would", "this")
            ]
            parts.extend(key_words[:5])

        if hypothesis.cross_layer_trace:
            if hypothesis.cross_layer_trace.l4_indicator:
                parts.append(hypothesis.cross_layer_trace.l4_indicator)

        return " ".join(parts)

    def _pattern_to_regex(self, code_pattern: str) -> Optional[str]:
        """Convert a code pattern to a regex."""
        # Simple pattern mappings
        pattern_regexes = {
            "unencrypted_fallback": r"(fallback|plain|unencrypt)",
            "sensitive_logging": r"(log|print|console)\s*\(.*?(password|secret|key|token)",
            "permission": r"(chmod|permission|access)\s*\(",
            "hardcoded_secret": r"(password|secret|api_key)\s*=\s*['\"]",
        }

        # Check for exact match
        if code_pattern in pattern_regexes:
            return pattern_regexes[code_pattern]

        # Extract suffix
        for suffix, regex in pattern_regexes.items():
            if code_pattern.endswith(suffix):
                return regex

        return None

    def _grep_pattern(
        self, regex: str, repo_path: str, hypothesis: ValueHypothesis
    ) -> list[EvidencePiece]:
        """Search for pattern matches in code."""
        evidence = []

        # Use code_chunking skill if diff hunk is available
        if hypothesis.diff_hunk:
            content = hypothesis.diff_hunk.content
            matches = re.finditer(regex, content, re.IGNORECASE)

            for match in matches:
                # Find line number
                lines_before = content[: match.start()].count("\n")
                line_num = hypothesis.diff_hunk.new_start + lines_before

                evidence.append(
                    EvidencePiece(
                        file_path=hypothesis.diff_hunk.file_path,
                        start_line=line_num,
                        end_line=line_num,
                        snippet=match.group(0),
                        relevance_score=0.7,
                    )
                )

        return evidence

    def _deduplicate_evidence(
        self, evidence: list[EvidencePiece]
    ) -> list[EvidencePiece]:
        """Remove duplicate evidence pieces."""
        seen = set()
        unique = []

        for piece in evidence:
            key = (piece.file_path, piece.start_line, piece.end_line)
            if key not in seen:
                seen.add(key)
                unique.append(piece)

        return unique

    def _rank_evidence(
        self, evidence: list[EvidencePiece], hypothesis: ValueHypothesis
    ) -> list[EvidencePiece]:
        """Rank evidence by relevance."""
        # Sort by relevance score, then by AST verification
        def score(piece: EvidencePiece) -> float:
            s = piece.relevance_score
            if piece.ast_trace:
                s += 0.1
            return s

        return sorted(evidence, key=score, reverse=True)

    def _get_strategies_used(self) -> list[str]:
        """Get list of search strategies that were used."""
        strategies = ["hunk_extraction"]
        if self.has_skill("vector_search"):
            strategies.append("vector_search")
        if self.has_skill("ast_analysis"):
            strategies.append("ast_analysis")
        strategies.append("pattern_matching")
        return strategies
