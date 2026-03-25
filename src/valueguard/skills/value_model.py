"""Value model skill for loading L1-L4 value definitions."""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from valueguard.skills.base_skill import BaseSkill


@dataclass
class ValueDefinition:
    """Definition of a value in the four-layer model."""

    layer: str  # L1, L2, L3, L4
    value_id: str  # e.g., "HV9", "SV10"
    value_name: str  # e.g., "Privacy", "Longevity"
    schwartz_mapping: str  # Mapping to L1 Schwartz value
    definition: str  # Full definition
    notes: str = ""
    indicators: list[str] = field(default_factory=list)  # L4 code indicators


@dataclass
class ValueModel:
    """Complete four-layer value model."""

    l1_values: dict[str, ValueDefinition] = field(default_factory=dict)
    l2_values: dict[str, ValueDefinition] = field(default_factory=dict)
    l3_values: dict[str, ValueDefinition] = field(default_factory=dict)
    l4_mappings: dict[str, list[str]] = field(default_factory=dict)  # L3 -> L4 indicators

    def get_all_values(self) -> dict[str, ValueDefinition]:
        """Get all value definitions."""
        return {**self.l1_values, **self.l2_values, **self.l3_values}

    def get_value(self, value_id: str) -> Optional[ValueDefinition]:
        """Get a value definition by ID."""
        all_values = self.get_all_values()
        return all_values.get(value_id)


class ValueModelSkill(BaseSkill):
    """Skill for loading and formatting the four-layer value model.

    Loads value definitions from CSV files and provides formatted
    representations for use in LLM prompts.
    """

    name = "value_model"
    description = "Load and format L1-L4 value definitions"
    version = "1.0.0"

    # Default L1 Schwartz values
    DEFAULT_L1_VALUES = {
        "Self-Direction": "Independent thought and action—choosing, creating, exploring",
        "Stimulation": "Excitement, novelty, and challenge in life",
        "Hedonism": "Pleasure and sensuous gratification for oneself",
        "Achievement": "Personal success through demonstrating competence according to social standards",
        "Power": "Social status and prestige, control or dominance over people and resources",
        "Security": "Safety, harmony, and stability of society, of relationships, and of self",
        "Conformity": "Restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms",
        "Tradition": "Respect, commitment, and acceptance of the customs and ideas that traditional culture or religion provide",
        "Benevolence": "Preserving and enhancing the welfare of those with whom one is in frequent personal contact",
        "Universalism": "Understanding, appreciation, tolerance, and protection for the welfare of all people and for nature",
    }

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        self._tables_path = config.get("tables_path", "tables") if config else "tables"
        self._cached_model: Optional[ValueModel] = None

    def execute(
        self,
        tables_path: Optional[str] = None,
        include_l1: bool = True,
        include_l2: bool = True,
        include_l3: bool = True,
        include_mappings: bool = False,
        format_for_prompt: bool = False,
        value_ids: Optional[list[str]] = None,
    ) -> Any:
        """Load and return the value model.

        Args:
            tables_path: Path to tables directory (overrides config)
            include_l1: Include L1 Schwartz values
            include_l2: Include L2 Human Value Themes
            include_l3: Include L3 System Value Themes
            include_mappings: Include cross-layer mappings
            format_for_prompt: Return formatted string for LLM prompt
            value_ids: Filter to specific value IDs

        Returns:
            ValueModel object or formatted string
        """
        tables_path = tables_path or self._tables_path
        model = self._load_model(tables_path)

        # Filter if value_ids specified
        if value_ids:
            model = self._filter_model(model, value_ids)

        # Remove layers if not requested
        if not include_l1:
            model.l1_values = {}
        if not include_l2:
            model.l2_values = {}
        if not include_l3:
            model.l3_values = {}
        if not include_mappings:
            model.l4_mappings = {}

        if format_for_prompt:
            return self._format_for_prompt(model)

        return model

    def _load_model(self, tables_path: str) -> ValueModel:
        """Load the complete value model from files."""
        if self._cached_model is not None:
            return self._cached_model

        tables_dir = Path(tables_path)
        model = ValueModel()

        # Load L1 defaults
        for name, definition in self.DEFAULT_L1_VALUES.items():
            model.l1_values[name] = ValueDefinition(
                layer="L1",
                value_id=name,
                value_name=name,
                schwartz_mapping=name,
                definition=definition,
            )

        # Load L2 values
        l2_file = tables_dir / "L2_Value_Themes.csv"
        if l2_file.exists():
            model.l2_values = self._load_csv_values(l2_file, "L2")

        # Load L3 values
        l3_file = tables_dir / "L3_system_value_themes.csv"
        if l3_file.exists():
            model.l3_values = self._load_csv_values(l3_file, "L3")

        self._cached_model = model
        return model

    def _load_csv_values(
        self, file_path: Path, layer: str
    ) -> dict[str, ValueDefinition]:
        """Load values from a CSV file."""
        values = {}

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                value_id = row.get("value_id", "")
                if not value_id:
                    continue

                value_def = ValueDefinition(
                    layer=row.get("layer", layer),
                    value_id=value_id,
                    value_name=row.get("value_name", ""),
                    schwartz_mapping=row.get("schwartz_mapping", ""),
                    definition=row.get("paper_definition", row.get("definition", "")),
                    notes=row.get("notes", row.get("mapping_notes", "")),
                )
                values[value_id] = value_def

        return values

    def _filter_model(
        self, model: ValueModel, value_ids: list[str]
    ) -> ValueModel:
        """Filter model to only include specified value IDs."""
        filtered = ValueModel()
        value_ids_set = set(value_ids)

        for vid, vdef in model.l1_values.items():
            if vid in value_ids_set:
                filtered.l1_values[vid] = vdef

        for vid, vdef in model.l2_values.items():
            if vid in value_ids_set:
                filtered.l2_values[vid] = vdef

        for vid, vdef in model.l3_values.items():
            if vid in value_ids_set:
                filtered.l3_values[vid] = vdef

        return filtered

    def _format_for_prompt(self, model: ValueModel) -> str:
        """Format the value model for use in LLM prompts."""
        lines = []

        if model.l1_values:
            lines.append("## L1: Schwartz Universal Values")
            for vid, vdef in model.l1_values.items():
                lines.append(f"- **{vid}**: {vdef.definition}")
            lines.append("")

        if model.l2_values:
            lines.append("## L2: Human Value Themes in Software")
            for vid, vdef in model.l2_values.items():
                lines.append(f"- **{vid} ({vdef.value_name})**: {vdef.definition}")
                if vdef.schwartz_mapping:
                    lines.append(f"  - Schwartz mapping: {vdef.schwartz_mapping}")
            lines.append("")

        if model.l3_values:
            lines.append("## L3: System Value Themes")
            for vid, vdef in model.l3_values.items():
                lines.append(f"- **{vid} ({vdef.value_name})**: {vdef.definition}")
                if vdef.schwartz_mapping:
                    lines.append(f"  - Schwartz mapping: {vdef.schwartz_mapping}")
            lines.append("")

        if model.l4_mappings:
            lines.append("## L4: Code Artifact Indicators")
            for l3_id, indicators in model.l4_mappings.items():
                lines.append(f"- **{l3_id}**: {', '.join(indicators)}")

        return "\n".join(lines)

    def get_cross_layer_mapping(
        self, value_id: str, tables_path: Optional[str] = None
    ) -> dict[str, str]:
        """Get the cross-layer mapping for a value.

        Returns mapping from L1 -> L2 -> L3 for the given value ID.
        """
        tables_path = tables_path or self._tables_path
        model = self._load_model(tables_path)

        mapping = {}
        vdef = model.get_value(value_id)

        if vdef:
            mapping["value_id"] = vdef.value_id
            mapping["value_name"] = vdef.value_name
            mapping["layer"] = vdef.layer
            mapping["definition"] = vdef.definition

            if vdef.schwartz_mapping:
                mapping["l1_schwartz"] = vdef.schwartz_mapping

        return mapping
