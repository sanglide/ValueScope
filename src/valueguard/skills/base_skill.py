"""Base skill interface for ValueGuard."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SkillMetadata:
    """Metadata for a skill."""

    name: str
    description: str
    version: str = "1.0.0"
    author: str = "ValueGuard"
    tags: list[str] = field(default_factory=list)


class BaseSkill(ABC):
    """Base class for all ValueGuard skills.

    Skills are modular, stateless functions that agents invoke on-demand.
    They enable composability, reusability, testability, and extensibility.
    """

    # Class attributes to be overridden by subclasses
    name: str = "base_skill"
    description: str = "Base skill description"
    version: str = "1.0.0"

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize skill with optional configuration.

        Args:
            config: Skill-specific configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the skill with given arguments.

        Args:
            **kwargs: Skill-specific arguments

        Returns:
            Skill-specific result
        """
        pass

    def validate_args(self, **kwargs: Any) -> None:
        """Validate input arguments before execution.

        Override this method to add argument validation.
        Raise ValueError if arguments are invalid.

        Args:
            **kwargs: Arguments to validate
        """
        pass

    def get_metadata(self) -> SkillMetadata:
        """Get skill metadata."""
        return SkillMetadata(
            name=self.name,
            description=self.description,
            version=self.version,
        )

    def __repr__(self) -> str:
        return f"<Skill:{self.name} v{self.version}>"
