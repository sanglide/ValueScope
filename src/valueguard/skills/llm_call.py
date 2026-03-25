"""LLM call skill for unified LLM invocation."""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from valueguard.skills.base_skill import BaseSkill


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    model_name: str
    raw_response: str
    parsed_result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class LLMCallSkill(BaseSkill):
    """Skill for unified LLM invocation.

    Supports multiple LLM providers through a common interface:
    - OpenAI (and OpenAI-compatible APIs like DeepSeek)
    - Anthropic Claude
    """

    name = "llm_call"
    description = "Invoke LLM with system/user prompts"
    version = "1.0.0"

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        self._clients: dict[str, Any] = {}
        self._default_provider = (
            config.get("default_provider", "deepseek") if config else "deepseek"
        )

    def validate_args(self, **kwargs: Any) -> None:
        """Validate arguments."""
        if "user" not in kwargs:
            raise ValueError("'user' prompt is required")

    def execute(
        self,
        user: str,
        system: str = "You are a helpful assistant.",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        parse_json: bool = True,
    ) -> LLMResponse:
        """Invoke an LLM with the given prompts.

        Args:
            user: User prompt (required)
            system: System prompt (default: helpful assistant)
            provider: LLM provider name. Supported values:
                - "deepseek" or "deepseek-chat"  -> api.deepseek.com
                - "qwen" or "qwen-plus"          -> dashscope
                - "openai", "o4-mini", "gpt-5.2" -> api.openai.com
                - "anthropic", "claude-sonnet-4-5"-> api.anthropic.com
                - "gemini-2.5-flash"             -> generativelanguage.googleapis.com
                - "grok-4"                       -> api.x.ai
            model: Model name override (provider-specific)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            parse_json: Whether to attempt JSON parsing of response

        Returns:
            LLMResponse with raw and optionally parsed response
        """
        provider = provider or self._default_provider
        start_time = time.time()

        # Determine which API path to use
        ANTHROPIC_PROVIDERS = {"anthropic", "claude-sonnet-4-5"}

        try:
            if provider in ANTHROPIC_PROVIDERS:
                response = self._call_anthropic(
                    user=user,
                    system=system,
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                response = self._call_openai_compatible(
                    user=user,
                    system=system,
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            latency_ms = (time.time() - start_time) * 1000

            # Parse JSON if requested
            parsed = None
            if parse_json:
                parsed = self._parse_json_response(response)

            return LLMResponse(
                model_name=model or provider,
                raw_response=response,
                parsed_result=parsed,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            import logging
            logger = logging.getLogger(__name__)
            
            # Enhanced error reporting for common issues
            error_msg = str(e)
            if "Unexpected role \"system\"" in error_msg and provider in ("claude-sonnet-4-5", "anthropic"):
                logger.error(
                    f"Claude API format issue detected!\n"
                    f"   Provider: {provider}\n"
                    f"   Error: {error_msg}\n\n"
                    f"   Claude does not accept 'system' as a message role in the messages array.\n"
                    f"   The system prompt should be passed via the 'system' parameter instead."
                )
            
            logger.error(f"LLM call failed (provider={provider}): {e}")
            return LLMResponse(
                model_name=model or provider,
                raw_response="",
                error=str(e),
                latency_ms=latency_ms,
            )

    def _get_api_key(self, env_var: str) -> str:
        """Get API key from environment variable."""
        key = os.environ.get(env_var, "")
        if not key:
            raise ValueError(f"API key not found: {env_var}")
        return key

    def _call_openai_compatible(
        self,
        user: str,
        system: str,
        provider: str,
        model: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call OpenAI-compatible API (OpenAI, DeepSeek, Qwen, Gemini, Grok, etc.)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        # Provider-specific settings
        if provider in ("deepseek", "deepseek-chat"):
            api_key = self._get_api_key("DEEPSEEK_API_KEY")
            base_url = "https://api.deepseek.com/v1"
            default_model = "deepseek-chat"
        elif provider in ("qwen", "qwen-plus"):
            api_key = self._get_api_key("DASHSCOPE_API_KEY")
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            default_model = "qwen-plus"
        elif provider in ("o4-mini", "gpt-5.2"):
            api_key = self._get_api_key("OPENAI_API_KEY")
            base_url = "https://api.openai.com/v1"
            default_model = provider
        elif provider == "gemini-2.5-flash":
            api_key = self._get_api_key("GOOGLE_API_KEY")
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
            default_model = "gemini-2.5-flash"
        elif provider == "grok-4":
            api_key = self._get_api_key("XAI_API_KEY")
            base_url = "https://api.x.ai/v1"
            default_model = "grok-4"
        else:  # openai direct
            api_key = self._get_api_key("OPENAI_API_KEY")
            base_url = None
            default_model = "gpt-4o"

        # Get config overrides
        if self.config and provider in self.config.get("providers", {}):
            provider_config = self.config["providers"][provider]
            if "api_key_env" in provider_config:
                api_key = self._get_api_key(provider_config["api_key_env"])
            if "base_url" in provider_config:
                base_url = provider_config["base_url"]
            if "model" in provider_config and not model:
                default_model = provider_config["model"]

        model = model or default_model

        import logging
        logging.getLogger(__name__).info(f"Calling {provider} -> model={model}, base_url={base_url}")

        # Create client
        client = OpenAI(api_key=api_key, base_url=base_url)

        # Build messages
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Model-specific temperature handling
        # o4-mini only supports temperature=1 (default)
        if model == "o4-mini":
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                # temperature omitted (defaults to 1)
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return response.choices[0].message.content or ""

    def _call_anthropic(
        self,
        user: str,
        system: str,
        provider: str = "anthropic",
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Call Anthropic Claude API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

        api_key = self._get_api_key("ANTHROPIC_API_KEY")

        # Map provider name to model if no explicit model given
        PROVIDER_MODEL_MAP = {
            "claude-sonnet-4-5": "claude-sonnet-4-5-20250514",
        }
        default_model = PROVIDER_MODEL_MAP.get(provider, "claude-sonnet-4-5-20250514")

        if self.config and "anthropic" in self.config.get("providers", {}):
            provider_config = self.config["providers"]["anthropic"]
            if "api_key_env" in provider_config:
                api_key = self._get_api_key(provider_config["api_key_env"])
            if "model" in provider_config and not model:
                default_model = provider_config["model"]

        model = model or default_model

        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        return response.content[0].text

    def _parse_json_response(self, response_text: str) -> Optional[dict[str, Any]]:
        """Parse JSON from LLM response.

        Tries multiple strategies:
        1. Direct JSON parsing
        2. Extract from markdown code blocks
        3. Find JSON object pattern
        4. Strip JS-style comments and retry
        """
        import logging
        _log = logging.getLogger(__name__)

        if not response_text:
            _log.warning("[_parse_json_response] response_text is empty")
            return None

        _log.debug(f"[_parse_json_response] input (first 600 chars): {response_text[:600]!r}")

        # Strategy 1: Direct parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code blocks
        json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(json_pattern, response_text)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError as e:
                _log.debug(f"[_parse_json_response] code-block match failed: {e} | match={match[:200]!r}")
                continue

        # Strategy 3: Find JSON object pattern
        json_obj_pattern = r"\{[\s\S]*\}"
        matches = re.findall(json_obj_pattern, response_text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # Strategy 4: Strip // line comments and /* block comments */ then retry
        try:
            cleaned = re.sub(r'//[^\n]*', '', response_text)
            cleaned = re.sub(r'/\*[\s\S]*?\*/', '', cleaned)
            # Try extracting JSON block again after stripping comments
            matches = re.findall(json_obj_pattern, cleaned)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass

        _log.warning(f"[_parse_json_response] All strategies failed for response: {response_text[:300]!r}")
        return None
