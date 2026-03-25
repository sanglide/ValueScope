"""
LLM Client Module
Unified API interface supporting multiple large language model providers
"""

import os
import json
import re
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """LLM response data class"""
    model_name: str
    raw_response: str
    parsed_result: Optional[dict] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class BaseLLMClient(ABC):
    """LLM client base class"""
    
    def __init__(self, model_config: dict):
        self.model_name = model_config.get("model_name")
        self.temperature = model_config.get("temperature", 0.0)
        self.max_tokens = model_config.get("max_tokens", 2048)
        self.api_key = self._get_api_key(model_config.get("api_key_env"))
        # base_url supports direct configuration or reading from environment variables
        self.base_url = model_config.get("base_url") or self._get_api_key(model_config.get("base_url_env"))
    
    def _get_api_key(self, env_var: str) -> Optional[str]:
        """Get API Key from environment variable"""
        if env_var:
            return os.getenv(env_var)
        return None
    
    @abstractmethod
    def call(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call LLM API"""
        pass
    
    def parse_json_response(self, response_text: str) -> Optional[dict]:
        """Parse JSON from response"""
        # Try direct parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON from markdown code blocks
        json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(json_pattern, response_text)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        # Try finding JSON object
        json_obj_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_obj_pattern, response_text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None


class OpenAIClient(BaseLLMClient):
    """OpenAI API client (also supports OpenAI API-compatible services)"""
    
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        try:
            # Azure OpenAI requires special handling
            if model_config.get("azure"):
                from openai import AzureOpenAI
                self.client = AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.base_url,
                    api_version=model_config.get("api_version", "2024-02-01"),
                )
            else:
                from openai import OpenAI
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self.client = OpenAI(**kwargs)
        except ImportError:
            raise ImportError("Please install the openai library: pip install openai")
    
    def call(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        import time
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            raw_response = response.choices[0].message.content
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                model_name=self.model_name,
                raw_response=raw_response,
                parsed_result=self.parse_json_response(raw_response),
                latency_ms=latency_ms
            )
        except Exception as e:
            return LLMResponse(
                model_name=self.model_name,
                raw_response="",
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client"""
    
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        try:
            import anthropic
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = anthropic.Anthropic(**kwargs)
        except ImportError:
            raise ImportError("Please install the anthropic library: pip install anthropic")
    
    def call(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        import time
        start_time = time.time()
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            raw_response = response.content[0].text
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                model_name=self.model_name,
                raw_response=raw_response,
                parsed_result=self.parse_json_response(raw_response),
                latency_ms=latency_ms
            )
        except Exception as e:
            return LLMResponse(
                model_name=self.model_name,
                raw_response="",
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )


class GeminiClient(BaseLLMClient):
    """Google Gemini API client (using the latest google-genai SDK)"""

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types

            # Proxy support: prioritize proxy from model_config, then environment variables
            proxy = (
                model_config.get("proxy")
                or os.getenv("HTTPS_PROXY")
                or os.getenv("https_proxy")
                or os.getenv("HTTP_PROXY")
                or os.getenv("http_proxy")
            )
            if proxy:
                import httpx
                httpx_client = httpx.Client(proxy=proxy)
                http_options = types.HttpOptions(httpxClient=httpx_client)
                self.client = genai.Client(api_key=self.api_key, http_options=http_options)
            else:
                self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install the google-genai library: pip install google-genai")

    def call(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        import time
        start_time = time.time()

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=self.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            raw_response = response.text
            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                model_name=self.model_name,
                raw_response=raw_response,
                parsed_result=self.parse_json_response(raw_response),
                latency_ms=latency_ms
            )
        except Exception as e:
            return LLMResponse(
                model_name=self.model_name,
                raw_response="",
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )


class LLMClientFactory:
    """LLM client factory class"""

    _providers = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "gemini": GeminiClient,
    }
    
    @classmethod
    def register_provider(cls, provider_name: str, client_class: type):
        """Register a new LLM provider"""
        cls._providers[provider_name] = client_class
    
    @classmethod
    def create(cls, model_config: dict) -> BaseLLMClient:
        """Create an LLM client based on configuration"""
        provider = model_config.get("provider", "openai")
        if provider not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        return cls._providers[provider](model_config)
    
    @classmethod
    def create_all_enabled(cls, llm_configs: dict) -> dict[str, BaseLLMClient]:
        """Create all enabled LLM clients"""
        clients = {}
        for model_key, config in llm_configs.items():
            if config.get("enabled", True):
                try:
                    clients[model_key] = cls.create(config)
                except Exception as e:
                    print(f"Warning: Unable to create {model_key} client: {e}")
        return clients
