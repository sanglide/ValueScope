"""Configuration loading for ValueGuard."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .exceptions import ConfigError


@dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider."""

    model: str
    api_key_env: str
    temperature: float = 0.0
    max_tokens: int = 2048
    base_url: Optional[str] = None

    def get_api_key(self) -> str:
        """Get API key from environment."""
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ConfigError(f"API key not found: {self.api_key_env}")
        return key


@dataclass
class LLMConfig:
    """LLM configuration."""

    default_provider: str = "deepseek"
    providers: dict[str, LLMProviderConfig] = field(default_factory=dict)


@dataclass
class VectorSearchConfig:
    """Vector search configuration."""

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    index_type: str = "faiss"
    chunk_size: int = 512
    overlap: int = 64


@dataclass
class ASTConfig:
    """AST analysis configuration."""

    languages: list[str] = field(
        default_factory=lambda: ["python", "javascript", "typescript", "java"]
    )
    max_call_depth: int = 5


@dataclass
class AnalysisConfig:
    """Analysis parameters configuration."""

    llm_provider: str = "deepseek"
    confidence_threshold: float = 0.5
    max_hypotheses: int = 10


@dataclass
class MemoryConfig:
    """Memory system configuration."""

    storage_type: str = "json"  # "json" or "sqlite"
    storage_path: str = ".valueguard/memory"
    profile_ttl_days: int = 30
    history_retention: int = 100


@dataclass
class OutputConfig:
    """Output configuration."""

    post_pr_comment: bool = True
    include_radar_chart: bool = True
    mention_reviewers: bool = True
    reviewer_teams: dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration for ValueGuard."""

    # Project-specific
    core_values: list[str] = field(default_factory=list)
    value_weights: dict[str, float] = field(default_factory=dict)

    # Component configs
    llm: LLMConfig = field(default_factory=LLMConfig)
    vector_search: VectorSearchConfig = field(default_factory=VectorSearchConfig)
    ast: ASTConfig = field(default_factory=ASTConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Paths
    repo_path: str = "."
    tables_path: str = "tables"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        config = cls()

        # Simple fields
        config.core_values = data.get("core_values", [])
        config.value_weights = data.get("value_weights", {})
        config.repo_path = data.get("repo_path", ".")
        config.tables_path = data.get("tables_path", "tables")

        # LLM config
        if "llm" in data or "llm_call" in data:
            llm_data = data.get("llm", data.get("llm_call", {}))
            providers = {}
            for name, pdata in llm_data.get("providers", {}).items():
                providers[name] = LLMProviderConfig(
                    model=pdata.get("model", ""),
                    api_key_env=pdata.get("api_key_env", ""),
                    temperature=pdata.get("temperature", 0.0),
                    max_tokens=pdata.get("max_tokens", 2048),
                    base_url=pdata.get("base_url"),
                )
            config.llm = LLMConfig(
                default_provider=llm_data.get("default_provider", "deepseek"),
                providers=providers,
            )

        # Vector search config
        if "vector_search" in data:
            vs = data["vector_search"]
            config.vector_search = VectorSearchConfig(
                embedding_model=vs.get("embedding_model", "BAAI/bge-small-en-v1.5"),
                index_type=vs.get("index_type", "faiss"),
                chunk_size=vs.get("chunk_size", 512),
                overlap=vs.get("overlap", 64),
            )

        # AST config
        if "ast_analysis" in data:
            ast = data["ast_analysis"]
            config.ast = ASTConfig(
                languages=ast.get(
                    "languages", ["python", "javascript", "typescript", "java"]
                ),
                max_call_depth=ast.get("max_call_depth", 5),
            )

        # Analysis config
        if "analysis" in data:
            analysis = data["analysis"]
            config.analysis = AnalysisConfig(
                llm_provider=analysis.get("llm_provider", "deepseek"),
                confidence_threshold=analysis.get("confidence_threshold", 0.5),
                max_hypotheses=analysis.get("max_hypotheses", 10),
            )

        # Memory config
        if "memory" in data:
            mem = data["memory"]
            config.memory = MemoryConfig(
                storage_type=mem.get("storage_type", "json"),
                storage_path=mem.get("storage_path", ".valueguard/memory"),
                profile_ttl_days=mem.get("profile_ttl_days", 30),
                history_retention=mem.get("history_retention", 100),
            )

        # Output config
        if "output" in data:
            out = data["output"]
            config.output = OutputConfig(
                post_pr_comment=out.get("post_pr_comment", True),
                include_radar_chart=out.get("include_radar_chart", True),
                mention_reviewers=out.get("mention_reviewers", True),
                reviewer_teams=out.get("reviewer_teams", {}),
            )

        return config


def load_config(
    repo_path: str = ".",
    config_file: Optional[str] = None,
    skills_file: Optional[str] = None,
) -> Config:
    """Load configuration from files and environment.

    Priority (highest to lowest):
    1. Environment variables
    2. .valueguard.yml in repo root
    3. Default skills.yaml from package
    4. Built-in defaults
    """
    repo_path = Path(repo_path).resolve()
    merged_data: dict[str, Any] = {}

    # 1. Load default skills config from package
    package_dir = Path(__file__).parent.parent
    default_skills = package_dir / "config" / "skills.yaml"
    if default_skills.exists():
        with open(default_skills) as f:
            merged_data.update(yaml.safe_load(f) or {})

    # 2. Load custom skills file if provided
    if skills_file:
        skills_path = Path(skills_file)
        if skills_path.exists():
            with open(skills_path) as f:
                merged_data.update(yaml.safe_load(f) or {})

    # 3. Load repo-specific .valueguard.yml
    if config_file:
        config_path = Path(config_file)
    else:
        config_path = repo_path / ".valueguard.yml"

    if config_path.exists():
        with open(config_path) as f:
            repo_config = yaml.safe_load(f) or {}
            merged_data.update(repo_config)

    # 4. Set repo path
    merged_data["repo_path"] = str(repo_path)

    # 5. Check for tables path
    tables_path = repo_path / "tables"
    if tables_path.exists():
        merged_data["tables_path"] = str(tables_path)

    # Create config
    config = Config.from_dict(merged_data)

    # 6. Apply environment variable overrides
    if os.environ.get("VALUEGUARD_LLM_PROVIDER"):
        config.analysis.llm_provider = os.environ["VALUEGUARD_LLM_PROVIDER"]

    if os.environ.get("VALUEGUARD_CONFIDENCE_THRESHOLD"):
        config.analysis.confidence_threshold = float(
            os.environ["VALUEGUARD_CONFIDENCE_THRESHOLD"]
        )

    return config
