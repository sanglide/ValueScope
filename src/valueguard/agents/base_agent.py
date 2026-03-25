"""Base agent interface for ValueGuard."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from valueguard.memory.manager import MemoryManager
from valueguard.skills.registry import SkillRegistry


class BaseAgent(ABC):
    """Base class for all ValueGuard agents.

    Agents are specialized workers that:
    - Receive typed tasks from the dispatcher
    - Invoke skills to perform subtasks
    - Use memory for context and learning
    - Return structured results
    """

    name: str = "base_agent"
    role: str = "Base Agent"
    goal: str = "Execute tasks"

    def __init__(
        self,
        skills: SkillRegistry,
        memory: MemoryManager,
        config: Optional[dict[str, Any]] = None,
    ):
        """Initialize agent with skills and memory.

        Args:
            skills: Skill registry for invoking skills
            memory: Memory manager for context and storage
            config: Agent-specific configuration
        """
        self.skills = skills
        self.memory = memory
        self.config = config or {}

    @abstractmethod
    def execute(self, task: Any) -> Any:
        """Execute the given task and return results.

        Args:
            task: Task dataclass with execution parameters

        Returns:
            Task-specific result
        """
        pass

    def invoke_skill(self, skill_name: str, **kwargs: Any) -> Any:
        """Invoke a skill by name with arguments.

        Args:
            skill_name: Name of skill to invoke
            **kwargs: Arguments to pass to skill

        Returns:
            Skill execution result

        Raises:
            SkillNotFoundError: If skill not registered
            SkillExecutionError: If skill execution fails
        """
        return self.skills.invoke(skill_name, **kwargs)

    def has_skill(self, skill_name: str) -> bool:
        """Check if a skill is available.

        Args:
            skill_name: Name of skill to check

        Returns:
            True if skill is registered
        """
        return skill_name in self.skills

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.role}>"
