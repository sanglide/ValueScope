"""Document collector skill for extracting text from project sources."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from valueguard.skills.base_skill import BaseSkill


@dataclass
class CollectedDocument:
    """A document collected from a project source."""

    source_type: str  # "readme", "issue", "config", "code"
    file_path: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocCollectorSkill(BaseSkill):
    """Skill for collecting documents from project sources.

    Extracts text from:
    - README files
    - Configuration files (package.json, pyproject.toml, etc.)
    - Code comments and docstrings
    - Issue/PR descriptions (from local files)
    """

    name = "doc_collector"
    description = "Extract text from README, configs, and other project sources"
    version = "1.0.0"

    # Patterns for different source types
    README_PATTERNS = [
        "README.md",
        "README.rst",
        "README.txt",
        "README",
        "readme.md",
    ]

    CONFIG_PATTERNS = [
        "package.json",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "Cargo.toml",
        "go.mod",
        ".valueguard.yml",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
    ]

    def execute(
        self,
        repo_path: str,
        sources: Optional[list[str]] = None,
        max_length: int = 10000,
    ) -> list[CollectedDocument]:
        """Collect documents from project sources.

        Args:
            repo_path: Path to the repository
            sources: List of source types to collect ("readme", "config", "code")
            max_length: Maximum length per document (truncate if longer)

        Returns:
            List of CollectedDocument objects
        """
        sources = sources or ["readme", "config"]
        repo_path = Path(repo_path)
        documents = []

        if "readme" in sources:
            docs = self._collect_readme(repo_path, max_length)
            documents.extend(docs)

        if "config" in sources:
            docs = self._collect_configs(repo_path, max_length)
            documents.extend(docs)

        if "code" in sources:
            docs = self._collect_code_docs(repo_path, max_length)
            documents.extend(docs)

        return documents

    def _collect_readme(
        self, repo_path: Path, max_length: int
    ) -> list[CollectedDocument]:
        """Collect README files."""
        documents = []

        for pattern in self.README_PATTERNS:
            readme_path = repo_path / pattern
            if readme_path.exists():
                content = self._read_file(readme_path, max_length)
                if content:
                    documents.append(
                        CollectedDocument(
                            source_type="readme",
                            file_path=str(pattern),
                            content=content,
                            metadata={"file_type": readme_path.suffix or ".txt"},
                        )
                    )
                break  # Only get the first README found

        return documents

    def _collect_configs(
        self, repo_path: Path, max_length: int
    ) -> list[CollectedDocument]:
        """Collect configuration files."""
        documents = []

        for pattern in self.CONFIG_PATTERNS:
            config_path = repo_path / pattern
            if config_path.exists():
                content = self._read_file(config_path, max_length)
                if content:
                    # Extract relevant sections based on file type
                    extracted = self._extract_config_values(
                        content, pattern
                    )
                    documents.append(
                        CollectedDocument(
                            source_type="config",
                            file_path=str(pattern),
                            content=extracted or content,
                            metadata={"file_type": config_path.suffix or ".txt"},
                        )
                    )

        return documents

    def _collect_code_docs(
        self, repo_path: Path, max_length: int
    ) -> list[CollectedDocument]:
        """Collect documentation from code files (docstrings, comments)."""
        documents = []
        code_extensions = {".py", ".js", ".ts", ".java", ".go", ".rs"}

        # Only scan top-level source directories
        src_dirs = ["src", "lib", "app", "."]

        for src_dir in src_dirs:
            src_path = repo_path / src_dir
            if not src_path.exists():
                continue

            for root, _, files in os.walk(src_path):
                # Limit depth
                rel_path = Path(root).relative_to(repo_path)
                if len(rel_path.parts) > 3:
                    continue

                for file in files[:10]:  # Limit files per directory
                    file_path = Path(root) / file
                    if file_path.suffix not in code_extensions:
                        continue

                    content = self._read_file(file_path, max_length)
                    if content:
                        docstrings = self._extract_docstrings(
                            content, file_path.suffix
                        )
                        if docstrings:
                            documents.append(
                                CollectedDocument(
                                    source_type="code",
                                    file_path=str(
                                        file_path.relative_to(repo_path)
                                    ),
                                    content=docstrings,
                                    metadata={
                                        "file_type": file_path.suffix,
                                        "language": self._get_language(
                                            file_path.suffix
                                        ),
                                    },
                                )
                            )

                    if len(documents) >= 20:  # Limit total code docs
                        break

        return documents

    def _read_file(self, file_path: Path, max_length: int) -> str:
        """Read file content with length limit."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read(max_length + 100)
                if len(content) > max_length:
                    content = content[:max_length] + "\n\n[... truncated ...]"
                return content
        except (IOError, UnicodeDecodeError):
            return ""

    def _extract_config_values(
        self, content: str, filename: str
    ) -> Optional[str]:
        """Extract value-relevant sections from config files."""
        if filename == "package.json":
            # Extract description, keywords, license
            try:
                import json
                data = json.loads(content)
                parts = []
                if "description" in data:
                    parts.append(f"Description: {data['description']}")
                if "keywords" in data:
                    parts.append(f"Keywords: {', '.join(data['keywords'])}")
                if "license" in data:
                    parts.append(f"License: {data['license']}")
                if "repository" in data:
                    if isinstance(data["repository"], dict):
                        parts.append(f"Repository: {data['repository'].get('url', '')}")
                    else:
                        parts.append(f"Repository: {data['repository']}")
                return "\n".join(parts) if parts else None
            except json.JSONDecodeError:
                return None

        elif filename == "pyproject.toml":
            # Extract relevant TOML sections
            parts = []
            # Simple regex extraction (avoid toml dependency)
            desc_match = re.search(r'description\s*=\s*"([^"]*)"', content)
            if desc_match:
                parts.append(f"Description: {desc_match.group(1)}")

            keywords_match = re.search(r'keywords\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if keywords_match:
                parts.append(f"Keywords: {keywords_match.group(1)}")

            license_match = re.search(r'license\s*=\s*"([^"]*)"', content)
            if license_match:
                parts.append(f"License: {license_match.group(1)}")

            return "\n".join(parts) if parts else content

        return content

    def _extract_docstrings(self, content: str, suffix: str) -> str:
        """Extract docstrings and significant comments from code."""
        docstrings = []

        if suffix == ".py":
            # Python docstrings (triple quotes)
            pattern = r'"""([\s\S]*?)"""|\'\'\'([\s\S]*?)\'\'\''
            matches = re.findall(pattern, content)
            for match in matches[:5]:  # Limit number of docstrings
                doc = match[0] or match[1]
                if doc.strip() and len(doc.strip()) > 20:
                    docstrings.append(doc.strip())

        elif suffix in (".js", ".ts"):
            # JSDoc comments
            pattern = r'/\*\*([\s\S]*?)\*/'
            matches = re.findall(pattern, content)
            for match in matches[:5]:
                if match.strip() and len(match.strip()) > 20:
                    # Clean up JSDoc formatting
                    cleaned = re.sub(r'^\s*\*\s?', '', match, flags=re.MULTILINE)
                    docstrings.append(cleaned.strip())

        elif suffix == ".java":
            # Javadoc comments
            pattern = r'/\*\*([\s\S]*?)\*/'
            matches = re.findall(pattern, content)
            for match in matches[:5]:
                if match.strip() and len(match.strip()) > 20:
                    cleaned = re.sub(r'^\s*\*\s?', '', match, flags=re.MULTILINE)
                    docstrings.append(cleaned.strip())

        return "\n\n---\n\n".join(docstrings)

    def _get_language(self, suffix: str) -> str:
        """Get language name from file suffix."""
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
        }
        return mapping.get(suffix, "unknown")
