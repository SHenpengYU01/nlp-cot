"""
OpenAI-compatible API wrapper for the COT project.

Supports any OpenAI-compatible endpoint, including:
- DeepSeek Official (https://api.deepseek.com/v1)
- DMXAPI (https://www.dmxapi.cn/v1)
- Official OpenAI API
- Other third-party compatible services
"""
import os
import httpx
from typing import List, Dict, Any, Optional

from openai import OpenAI

from .base import BaseModel

# Default API config
DEFAULT_API_KEY = "sk-56da081510c3491f93b749edda4ac1e0"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class OpenAIModel(BaseModel):
    """OpenAI-compatible API model wrapper."""

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or DEFAULT_API_KEY
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        if not self.api_key:
            raise ValueError(
                "API key is required. Set OPENAI_API_KEY env var or pass --api_key."
            )
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=httpx.Client(trust_env=False),
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        n: int = 1,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            n=n,
            **kwargs
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        n: int = 1,
        **kwargs
    ) -> List[str]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            n=n,
            **kwargs
        )
        return [choice.message.content for choice in response.choices]

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info["provider"] = "openai_compatible"
        info["base_url"] = self.base_url
        return info
