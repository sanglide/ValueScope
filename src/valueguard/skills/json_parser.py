"""JSON parser skill for parsing and validating JSON responses."""

import json
import re
from typing import Any, Optional

from valueguard.skills.base_skill import BaseSkill


class JsonParserSkill(BaseSkill):
    """Skill for parsing and validating JSON from LLM responses.

    Handles various JSON formats including:
    - Raw JSON
    - JSON in markdown code blocks
    - Partial JSON extraction
    """

    name = "json_parser"
    description = "Parse and validate JSON from LLM responses"
    version = "1.0.0"

    def execute(
        self,
        response: str,
        schema: Optional[dict[str, Any]] = None,
        strict: bool = False,
        default: Optional[Any] = None,
    ) -> Any:
        """Parse JSON from a response string.

        Args:
            response: Raw response text (may contain JSON)
            schema: Optional JSON schema for validation
            strict: If True, raise on parse/validation errors
            default: Default value if parsing fails (when not strict)

        Returns:
            Parsed JSON object or default value

        Raises:
            ValueError: If strict=True and parsing/validation fails
        """
        # Try to parse JSON
        parsed = self._parse_json(response)

        if parsed is None:
            if strict:
                raise ValueError("Failed to parse JSON from response")
            return default

        # Validate against schema if provided
        if schema is not None:
            is_valid, error = self._validate_schema(parsed, schema)
            if not is_valid:
                if strict:
                    raise ValueError(f"JSON validation failed: {error}")
                return default

        return parsed

    def _parse_json(self, text: str) -> Optional[Any]:
        """Parse JSON from text using multiple strategies."""
        # Strategy 1: Direct parsing
        result = self._try_direct_parse(text)
        if result is not None:
            return result

        # Strategy 2: Extract from markdown code blocks
        result = self._try_code_block_parse(text)
        if result is not None:
            return result

        # Strategy 3: Find JSON object/array pattern
        result = self._try_pattern_parse(text)
        if result is not None:
            return result

        # Strategy 4: Try to fix common issues and parse again
        result = self._try_fix_and_parse(text)
        return result

    def _try_direct_parse(self, text: str) -> Optional[Any]:
        """Try to parse text directly as JSON."""
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None

    def _try_code_block_parse(self, text: str) -> Optional[Any]:
        """Try to extract and parse JSON from markdown code blocks."""
        # Match ```json ... ``` or ``` ... ```
        patterns = [
            r"```json\s*([\s\S]*?)```",
            r"```\s*([\s\S]*?)```",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue

        return None

    def _try_pattern_parse(self, text: str) -> Optional[Any]:
        """Try to find and parse JSON object/array patterns."""
        # Try to find JSON object
        obj_match = re.search(r"\{[\s\S]*\}", text)
        if obj_match:
            try:
                return json.loads(obj_match.group())
            except json.JSONDecodeError:
                pass

        # Try to find JSON array
        arr_match = re.search(r"\[[\s\S]*\]", text)
        if arr_match:
            try:
                return json.loads(arr_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _try_fix_and_parse(self, text: str) -> Optional[Any]:
        """Try to fix common JSON issues and parse."""
        # Extract potential JSON content
        obj_match = re.search(r"\{[\s\S]*\}", text)
        if not obj_match:
            arr_match = re.search(r"\[[\s\S]*\]", text)
            if not arr_match:
                return None
            json_text = arr_match.group()
        else:
            json_text = obj_match.group()

        # Fix 1: Remove trailing commas
        json_text = re.sub(r",\s*([}\]])", r"\1", json_text)

        # Fix 2: Replace single quotes with double quotes
        # (Be careful not to replace quotes inside strings)
        json_text = self._fix_quotes(json_text)

        # Fix 3: Add missing commas between elements
        json_text = re.sub(r'"\s*\n\s*"', '",\n"', json_text)

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return None

    def _fix_quotes(self, text: str) -> str:
        """Fix single quotes to double quotes (simple heuristic)."""
        # This is a simple fix that works for most cases
        # A more robust solution would require proper parsing
        result = []
        in_string = False
        string_char = None

        i = 0
        while i < len(text):
            char = text[i]

            if not in_string:
                if char == '"':
                    in_string = True
                    string_char = '"'
                    result.append(char)
                elif char == "'":
                    # Check if this looks like a string start
                    # (followed by content and closing quote)
                    in_string = True
                    string_char = "'"
                    result.append('"')  # Replace with double quote
                else:
                    result.append(char)
            else:
                if char == string_char:
                    in_string = False
                    if string_char == "'":
                        result.append('"')  # Replace with double quote
                    else:
                        result.append(char)
                    string_char = None
                elif char == "\\" and i + 1 < len(text):
                    result.append(char)
                    result.append(text[i + 1])
                    i += 1
                else:
                    result.append(char)

            i += 1

        return "".join(result)

    def _validate_schema(
        self, data: Any, schema: dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate data against a JSON schema (simplified validation)."""
        # Simple type validation
        schema_type = schema.get("type")

        if schema_type == "object":
            if not isinstance(data, dict):
                return False, f"Expected object, got {type(data).__name__}"

            # Check required fields
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    return False, f"Missing required field: {field}"

            # Check properties
            properties = schema.get("properties", {})
            for prop_name, prop_schema in properties.items():
                if prop_name in data:
                    is_valid, error = self._validate_schema(
                        data[prop_name], prop_schema
                    )
                    if not is_valid:
                        return False, f"{prop_name}: {error}"

        elif schema_type == "array":
            if not isinstance(data, list):
                return False, f"Expected array, got {type(data).__name__}"

            items_schema = schema.get("items")
            if items_schema:
                for i, item in enumerate(data):
                    is_valid, error = self._validate_schema(item, items_schema)
                    if not is_valid:
                        return False, f"[{i}]: {error}"

        elif schema_type == "string":
            if not isinstance(data, str):
                return False, f"Expected string, got {type(data).__name__}"

        elif schema_type == "number":
            if not isinstance(data, (int, float)):
                return False, f"Expected number, got {type(data).__name__}"

        elif schema_type == "integer":
            if not isinstance(data, int):
                return False, f"Expected integer, got {type(data).__name__}"

        elif schema_type == "boolean":
            if not isinstance(data, bool):
                return False, f"Expected boolean, got {type(data).__name__}"

        return True, None

    def extract_multiple(self, text: str) -> list[Any]:
        """Extract multiple JSON objects from text.

        Useful when response contains multiple JSON blocks.
        """
        results = []

        # Find all code blocks
        code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
        for block in code_blocks:
            try:
                results.append(json.loads(block.strip()))
            except json.JSONDecodeError:
                pass

        # If no code blocks found, try to find all JSON objects
        if not results:
            for match in re.finditer(r"\{[^{}]*\}", text):
                try:
                    results.append(json.loads(match.group()))
                except json.JSONDecodeError:
                    pass

        return results
