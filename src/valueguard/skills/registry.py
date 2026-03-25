"""Skill registry for ValueGuard."""

import importlib
import inspect
import pkgutil
from typing import Any, Optional

from valueguard.core.exceptions import SkillExecutionError, SkillNotFoundError

from .base_skill import BaseSkill


class SkillRegistry:
    """Central registry for skill discovery and invocation.

    The registry maintains a collection of skills that can be:
    - Manually registered
    - Auto-discovered from a package
    - Invoked by name with arguments
    """

    def __init__(self):
        """Initialize empty skill registry."""
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        """Register a skill instance.

        Args:
            skill: Skill instance to register

        Raises:
            ValueError: If skill with same name already exists
        """
        if skill.name in self._skills:
            raise ValueError(f"Skill already registered: {skill.name}")
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> None:
        """Unregister a skill by name.

        Args:
            name: Name of skill to unregister
        """
        if name in self._skills:
            del self._skills[name]

    def get(self, name: str) -> Optional[BaseSkill]:
        """Get a skill by name without raising.

        Args:
            name: Name of skill to get

        Returns:
            Skill instance or None if not found
        """
        return self._skills.get(name)

    def invoke(self, name: str, **kwargs: Any) -> Any:
        """Invoke a skill by name with arguments.

        Args:
            name: Name of skill to invoke
            **kwargs: Arguments to pass to skill

        Returns:
            Result from skill execution

        Raises:
            SkillNotFoundError: If skill not registered
            SkillExecutionError: If skill execution fails
        """
        if name not in self._skills:
            raise SkillNotFoundError(name)

        skill = self._skills[name]

        try:
            skill.validate_args(**kwargs)
            return skill.execute(**kwargs)
        except SkillNotFoundError:
            raise
        except Exception as e:
            raise SkillExecutionError(name, str(e)) from e

    def auto_discover(self, package: str = "valueguard.skills") -> int:
        """Auto-discover and register skills from a package.

        Scans the package for BaseSkill subclasses and registers them.

        Args:
            package: Package path to scan for skills

        Returns:
            Number of skills discovered and registered
        """
        count = 0

        try:
            pkg = importlib.import_module(package)
        except ImportError:
            return 0

        pkg_path = getattr(pkg, "__path__", None)
        if pkg_path is None:
            return 0

        for _, module_name, _ in pkgutil.iter_modules(pkg_path):
            # Skip base_skill and registry modules
            if module_name in ("base_skill", "registry", "__init__"):
                continue

            try:
                module = importlib.import_module(f"{package}.{module_name}")

                # Find all BaseSkill subclasses in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseSkill)
                        and obj is not BaseSkill
                        and obj.name != "base_skill"
                    ):
                        # Check if already registered
                        if obj.name not in self._skills:
                            skill = obj()
                            self.register(skill)
                            count += 1
            except Exception:
                # Skip modules that fail to import
                continue

        return count

    def list_skills(self) -> list[str]:
        """List all registered skill names.

        Returns:
            List of skill names
        """
        return list(self._skills.keys())

    def get_skill_info(self) -> list[dict[str, str]]:
        """Get information about all registered skills.

        Returns:
            List of skill metadata dictionaries
        """
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "version": skill.version,
            }
            for skill in self._skills.values()
        ]

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __repr__(self) -> str:
        return f"<SkillRegistry: {len(self)} skills>"
