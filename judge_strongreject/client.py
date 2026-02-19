"""
Minimal LLM client for StrongREJECT judge (OpenAI only).
Same interface and behavior as judge_cat client.
"""
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict

# Optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LLMClient(ABC):
    """Base interface for Judge LLM clients."""

    @abstractmethod
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call the LLM; return response text."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. pip install openai")
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )
        endpoint = endpoint or "https://api.openai.com/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=endpoint)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content


# Judge-supported models: model_name -> (client_class, deployment_name for API)
JUDGE_MODELS: Dict[str, tuple] = {
    "gpt-4o-mini": (OpenAIClient, "gpt-4o-mini"),
    "gpt-4o": (OpenAIClient, "gpt-4o"),
    "gpt-4": (OpenAIClient, "gpt-4"),
    "gpt-3.5-turbo": (OpenAIClient, "gpt-3.5-turbo"),
}

_client_cache: Dict[str, LLMClient] = {}


def get_judge_client(model_name: Optional[str] = None) -> LLMClient:
    """
    Create (or return cached) Judge LLM client.
    Default model: gpt-4o-mini.
    """
    if model_name is None:
        model_name = "gpt-4o-mini"
    if model_name not in JUDGE_MODELS:
        raise ValueError(
            f"Unknown judge model: {model_name}. "
            f"Available: {list(JUDGE_MODELS.keys())}"
        )
    if model_name in _client_cache:
        return _client_cache[model_name]
    client_cls, deployment = JUDGE_MODELS[model_name]
    client = client_cls(deployment)
    _client_cache[model_name] = client
    return client
