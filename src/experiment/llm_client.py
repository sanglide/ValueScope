"""
LLM客户端模块
支持多种大模型API的统一调用接口
"""

import os
import json
import re
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """LLM响应数据类"""
    model_name: str
    raw_response: str
    parsed_result: Optional[dict] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    def __init__(self, model_config: dict):
        self.model_name = model_config.get("model_name")
        self.temperature = model_config.get("temperature", 0.0)
        self.max_tokens = model_config.get("max_tokens", 2048)
        self.api_key = self._get_api_key(model_config.get("api_key_env"))
        # base_url 支持直接配置或从环境变量读取
        self.base_url = model_config.get("base_url") or self._get_api_key(model_config.get("base_url_env"))
    
    def _get_api_key(self, env_var: str) -> Optional[str]:
        """从环境变量获取API Key"""
        if env_var:
            return os.getenv(env_var)
        return None
    
    @abstractmethod
    def call(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """调用LLM API"""
        pass
    
    def parse_json_response(self, response_text: str) -> Optional[dict]:
        """从响应中解析JSON"""
        # 尝试直接解析
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # 尝试从markdown代码块中提取JSON
        json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(json_pattern, response_text)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        # 尝试找到JSON对象
        json_obj_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_obj_pattern, response_text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None


class OpenAIClient(BaseLLMClient):
    """OpenAI API客户端（也支持兼容OpenAI API的服务）"""
    
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        try:
            # Azure OpenAI 需要特殊处理
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
            raise ImportError("请安装openai库: pip install openai")
    
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
    
    def send_request(self, messages: list) -> str:
        """发送消息列表并返回原始文本响应"""
        import time
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"LLM request failed: {str(e)}")


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API客户端"""
    
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        try:
            import anthropic
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = anthropic.Anthropic(**kwargs)
        except ImportError:
            raise ImportError("请安装anthropic库: pip install anthropic")
    
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
    
    def send_request(self, messages: list) -> str:
        """发送消息列表并返回原始文本响应"""
        import time
        start_time = time.time()
        
        try:
            # Anthropic 需要分离 system prompt
            system_msg = None
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    user_messages.append(msg)
            
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                system=system_msg or "",
                messages=user_messages
            )
            
            return response.content[0].text
        except Exception as e:
            raise Exception(f"LLM request failed: {str(e)}")


class GeminiClient(BaseLLMClient):
    """Google Gemini API客户端（使用最新 google-genai SDK）"""

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types

            # 代理支持：优先读取 model_config 中的 proxy，其次读取环境变量
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
            raise ImportError("请安装google-genai库: pip install google-genai")

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
    
    def send_request(self, messages: list) -> str:
        """发送消息列表并返回原始文本响应"""
        import time
        start_time = time.time()
        
        try:
            # Gemini 需要分离 system prompt
            system_msg = None
            contents = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    contents.append(msg["content"])
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents="\n".join(contents),
                config=self.types.GenerateContentConfig(
                    system_instruction=system_msg or "",
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            
            return response.text
        except Exception as e:
            raise Exception(f"LLM request failed: {str(e)}")


class LLMClientFactory:
    """LLM客户端工厂类"""

    _providers = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "gemini": GeminiClient,
    }
    
    @classmethod
    def register_provider(cls, provider_name: str, client_class: type):
        """注册新的LLM提供商"""
        cls._providers[provider_name] = client_class
    
    @classmethod
    def create(cls, model_config: dict) -> BaseLLMClient:
        """根据配置创建LLM客户端"""
        provider = model_config.get("provider", "openai")
        if provider not in cls._providers:
            raise ValueError(f"不支持的LLM提供商: {provider}")
        return cls._providers[provider](model_config)
    
    @classmethod
    def create_all_enabled(cls, llm_configs: dict) -> dict[str, BaseLLMClient]:
        """创建所有启用的LLM客户端"""
        clients = {}
        for model_key, config in llm_configs.items():
            if config.get("enabled", True):
                try:
                    clients[model_key] = cls.create(config)
                except Exception as e:
                    print(f"警告: 无法创建{model_key}客户端: {e}")
        return clients
