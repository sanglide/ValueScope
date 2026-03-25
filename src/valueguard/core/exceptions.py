"""Custom exceptions for ValueGuard."""


class ValueGuardError(Exception):
    """Base exception for all ValueGuard errors."""

    pass


class ConfigError(ValueGuardError):
    """Configuration loading or validation error."""

    pass


class SkillNotFoundError(ValueGuardError):
    """Raised when a requested skill is not registered."""

    def __init__(self, skill_name: str):
        self.skill_name = skill_name
        super().__init__(f"Skill not found: {skill_name}")


class SkillExecutionError(ValueGuardError):
    """Raised when a skill fails during execution."""

    def __init__(self, skill_name: str, message: str):
        self.skill_name = skill_name
        super().__init__(f"Skill '{skill_name}' failed: {message}")


class MemoryError(ValueGuardError):
    """Memory system error."""

    pass


class AgentError(ValueGuardError):
    """Agent execution error."""

    def __init__(self, agent_name: str, message: str):
        self.agent_name = agent_name
        super().__init__(f"Agent '{agent_name}' failed: {message}")
