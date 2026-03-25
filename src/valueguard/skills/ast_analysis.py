"""AST analysis skill for code structure verification."""

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from valueguard.skills.base_skill import BaseSkill


@dataclass
class ASTVerificationResult:
    """Result of AST verification."""

    verified: bool = False
    function_name: str = ""
    file_path: str = ""
    lines: list[int] = field(default_factory=list)
    snippet: str = ""
    call_chain: list[str] = field(default_factory=list)
    error: Optional[str] = None


class ASTAnalysisSkill(BaseSkill):
    """Skill for analyzing code AST using tree-sitter.

    Provides:
    - Function/method extraction
    - Call chain tracing
    - Pattern verification
    """

    name = "ast_analysis"
    description = "Analyze code AST using tree-sitter for verification"
    version = "1.0.0"

    # Language configurations
    LANGUAGE_CONFIGS = {
        "python": {
            "function_query": "(function_definition name: (identifier) @name)",
            "call_query": "(call function: (identifier) @name)",
            "extension": ".py",
        },
        "javascript": {
            "function_query": "(function_declaration name: (identifier) @name)",
            "call_query": "(call_expression function: (identifier) @name)",
            "extension": ".js",
        },
        "typescript": {
            "function_query": "(function_declaration name: (identifier) @name)",
            "call_query": "(call_expression function: (identifier) @name)",
            "extension": ".ts",
        },
    }

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        self._max_call_depth = config.get("max_call_depth", 5) if config else 5
        self._parsers: dict[str, Any] = {}

    def execute(
        self,
        file_path: str,
        hypothesis: Optional[Any] = None,
        search_depth: int = 3,
        action: str = "verify",
    ) -> dict[str, Any]:
        """Execute AST analysis.

        Args:
            file_path: Path to the source file
            hypothesis: ValueHypothesis to verify (optional)
            search_depth: Depth for call chain tracing
            action: "verify" to verify hypothesis, "extract" to extract functions

        Returns:
            Verification result or extracted functions
        """
        if action == "verify":
            return self._verify_hypothesis(file_path, hypothesis, search_depth)
        elif action == "extract":
            return self._extract_functions(file_path)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _verify_hypothesis(
        self,
        file_path: str,
        hypothesis: Any,
        search_depth: int,
    ) -> dict[str, Any]:
        """Verify a hypothesis against code AST."""
        result = ASTVerificationResult(file_path=file_path)

        if not os.path.isfile(file_path):
            result.error = f"File not found: {file_path}"
            return self._result_to_dict(result)

        # Detect language
        language = self._detect_language(file_path)
        if not language:
            result.error = f"Unsupported language for: {file_path}"
            return self._result_to_dict(result)

        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse with tree-sitter
            tree = self._parse(content, language)
            if tree is None:
                # Fallback to simple verification
                return self._simple_verify(file_path, content, hypothesis)

            # Extract relevant information
            functions = self._extract_functions_from_tree(tree, language)
            calls = self._extract_calls_from_tree(tree, language)

            # Try to find evidence for hypothesis
            if hypothesis and hasattr(hypothesis, "cross_layer_trace"):
                trace = hypothesis.cross_layer_trace
                if trace and trace.l4_indicator:
                    # Search for indicator pattern
                    verified, snippet, lines = self._search_pattern(
                        content, trace.l4_indicator
                    )
                    if verified:
                        result.verified = True
                        result.snippet = snippet
                        result.lines = lines

                        # Try to find containing function
                        for func in functions:
                            if func["start_line"] <= lines[0] <= func["end_line"]:
                                result.function_name = func["name"]
                                break

                        # Build call chain (simplified)
                        result.call_chain = self._build_call_chain(
                            result.function_name, calls, search_depth
                        )

            # Default verification based on diff hunk
            if not result.verified and hypothesis and hasattr(hypothesis, "diff_hunk"):
                hunk = hypothesis.diff_hunk
                if hunk and hunk.file_path.endswith(os.path.basename(file_path)):
                    result.verified = True
                    result.lines = list(range(hunk.new_start, hunk.new_start + hunk.new_lines))
                    # Get snippet from file
                    lines_list = content.split("\n")
                    start = max(0, hunk.new_start - 1)
                    end = min(len(lines_list), hunk.new_start + hunk.new_lines)
                    result.snippet = "\n".join(lines_list[start:end])

        except Exception as e:
            result.error = str(e)

        return self._result_to_dict(result)

    def _extract_functions(self, file_path: str) -> dict[str, Any]:
        """Extract all functions from a file."""
        if not os.path.isfile(file_path):
            return {"error": f"File not found: {file_path}", "functions": []}

        language = self._detect_language(file_path)
        if not language:
            return {"error": f"Unsupported language: {file_path}", "functions": []}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = self._parse(content, language)
            if tree is None:
                return self._extract_functions_simple(content, language)

            functions = self._extract_functions_from_tree(tree, language)
            return {"functions": functions}

        except Exception as e:
            return {"error": str(e), "functions": []}

    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
        }
        _, ext = os.path.splitext(file_path)
        return ext_map.get(ext.lower())

    def _parse(self, content: str, language: str) -> Optional[Any]:
        """Parse content with tree-sitter."""
        try:
            import tree_sitter_python
            import tree_sitter_javascript
            from tree_sitter import Language, Parser

            # Get or create parser for language
            if language not in self._parsers:
                if language == "python":
                    lang = Language(tree_sitter_python.language())
                elif language in ("javascript", "typescript"):
                    lang = Language(tree_sitter_javascript.language())
                else:
                    return None

                parser = Parser(lang)
                self._parsers[language] = parser

            parser = self._parsers[language]
            tree = parser.parse(bytes(content, "utf8"))
            return tree

        except ImportError:
            # tree-sitter not installed, fall back to simple analysis
            return None
        except Exception:
            return None

    def _extract_functions_from_tree(
        self, tree: Any, language: str
    ) -> list[dict[str, Any]]:
        """Extract function definitions from AST tree."""
        functions = []

        try:
            root = tree.root_node

            def visit(node: Any) -> None:
                # Check for function definitions
                if node.type in ("function_definition", "function_declaration", "method_definition"):
                    name_node = None
                    for child in node.children:
                        if child.type == "identifier" or child.type == "name":
                            name_node = child
                            break

                    if name_node:
                        functions.append({
                            "name": name_node.text.decode("utf8"),
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "type": node.type,
                        })

                # Recurse into children
                for child in node.children:
                    visit(child)

            visit(root)

        except Exception:
            pass

        return functions

    def _extract_calls_from_tree(
        self, tree: Any, language: str
    ) -> list[dict[str, Any]]:
        """Extract function calls from AST tree."""
        calls = []

        try:
            root = tree.root_node

            def visit(node: Any) -> None:
                if node.type in ("call", "call_expression"):
                    name = None
                    for child in node.children:
                        if child.type == "identifier":
                            name = child.text.decode("utf8")
                            break
                        elif child.type == "attribute":
                            # Method call: obj.method()
                            for attr_child in child.children:
                                if attr_child.type == "identifier":
                                    name = attr_child.text.decode("utf8")
                                    break

                    if name:
                        calls.append({
                            "name": name,
                            "line": node.start_point[0] + 1,
                        })

                for child in node.children:
                    visit(child)

            visit(root)

        except Exception:
            pass

        return calls

    def _search_pattern(
        self, content: str, pattern: str
    ) -> tuple[bool, str, list[int]]:
        """Search for a pattern in content."""
        import re

        # Convert pattern to regex-friendly form
        pattern_words = pattern.lower().replace("_", "|").replace("-", "|")
        patterns_to_try = [
            pattern,  # Exact match
            pattern.replace("_", ""),  # No underscores
            f".*({'|'.join(pattern.split('_'))})+.*",  # Any word from pattern
        ]

        lines = content.split("\n")
        for pat in patterns_to_try:
            try:
                regex = re.compile(pat, re.IGNORECASE)
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        # Get context (3 lines before/after)
                        start = max(0, i - 4)
                        end = min(len(lines), i + 3)
                        snippet = "\n".join(lines[start:end])
                        return True, snippet, list(range(start + 1, end + 1))
            except re.error:
                continue

        return False, "", []

    def _build_call_chain(
        self, function_name: str, calls: list[dict[str, Any]], depth: int
    ) -> list[str]:
        """Build a simplified call chain."""
        if not function_name or depth <= 0:
            return []

        chain = [function_name]

        # Find calls made by this function (simplified - just return called functions)
        called = [c["name"] for c in calls[:depth]]
        chain.extend(called)

        return chain[:depth]

    def _simple_verify(
        self, file_path: str, content: str, hypothesis: Any
    ) -> dict[str, Any]:
        """Simple verification without tree-sitter."""
        result = ASTVerificationResult(file_path=file_path)

        if hypothesis and hasattr(hypothesis, "cross_layer_trace"):
            trace = hypothesis.cross_layer_trace
            if trace and trace.l4_indicator:
                verified, snippet, lines = self._search_pattern(
                    content, trace.l4_indicator
                )
                if verified:
                    result.verified = True
                    result.snippet = snippet
                    result.lines = lines

        return self._result_to_dict(result)

    def _extract_functions_simple(
        self, content: str, language: str
    ) -> dict[str, Any]:
        """Extract functions using simple regex patterns."""
        import re

        functions = []

        if language == "python":
            pattern = r"^(\s*)def\s+(\w+)\s*\("
            for i, line in enumerate(content.split("\n"), 1):
                match = re.match(pattern, line)
                if match:
                    functions.append({
                        "name": match.group(2),
                        "start_line": i,
                        "end_line": i,  # Simplified
                        "type": "function_definition",
                    })

        elif language in ("javascript", "typescript"):
            patterns = [
                r"function\s+(\w+)\s*\(",
                r"const\s+(\w+)\s*=\s*(?:async\s+)?\(",
                r"(\w+)\s*:\s*(?:async\s+)?function\s*\(",
            ]
            for pattern in patterns:
                for i, line in enumerate(content.split("\n"), 1):
                    match = re.search(pattern, line)
                    if match:
                        functions.append({
                            "name": match.group(1),
                            "start_line": i,
                            "end_line": i,
                            "type": "function_declaration",
                        })

        return {"functions": functions}

    def _result_to_dict(self, result: ASTVerificationResult) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "verified": result.verified,
            "function_name": result.function_name,
            "file_path": result.file_path,
            "lines": result.lines,
            "snippet": result.snippet,
            "call_chain": result.call_chain,
            "error": result.error,
        }
