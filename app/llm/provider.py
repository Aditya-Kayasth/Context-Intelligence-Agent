"""LLM provider factory — returns the correct async client based on settings."""
from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from app.config import settings

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class BaseLLMProvider(ABC):
    """Abstract base for all LLM provider implementations."""

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request and return the assistant message text."""


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI chat-completion provider via httpx."""

    def __init__(self) -> None:
        if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
            raise ValueError("Azure OpenAI configuration is missing in settings.")
        self._api_key = settings.azure_openai_api_key
        base_url = settings.azure_openai_endpoint.rstrip("/")
        deployment = settings.azure_openai_deployment_name
        api_version = settings.azure_openai_api_version
        self._url = f"{base_url}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Azure OpenAI chat-completions endpoint and return the response text."""
        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(self._url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class GroqProvider(BaseLLMProvider):
    """Groq chat-completion provider via httpx (fully async, no SDK dependency)."""

    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in configuration.")
        self._api_key = settings.groq_api_key
        self._model = settings.groq_model

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Groq chat-completions endpoint and return the response text."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(_GROQ_CHAT_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class OllamaProvider(BaseLLMProvider):
    """Ollama chat-completion provider via httpx."""

    def __init__(self) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Ollama generate endpoint and return the response text."""
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]


class OpenAIProvider(BaseLLMProvider):
    """OpenAI chat-completion provider via httpx."""

    def __init__(self) -> None:
        if not getattr(settings, "openai_api_key", None):
            raise ValueError("OPENAI_API_KEY is not set in configuration.")
        self._api_key = settings.openai_api_key
        self._model = getattr(settings, "openai_model", "gpt-4o")

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call the OpenAI chat-completions endpoint and return the response text."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class StubProvider(BaseLLMProvider):
    """Fallback provider that returns canned JSON when real LLMs are unavailable."""

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return a generic but valid JSON response for profiling."""
        import json
        
        # Try to extract column names from the prompt to make the mock response look real
        cols = []
        try:
            prompt_data = json.loads(user_prompt)
            if isinstance(prompt_data, list):
                cols = [c.get("name", "unknown") for c in prompt_data]
        except Exception:
            pass

        mock_response = {
            "semantic_types": {c: "category" for c in cols},
            "suggested_analyses": [
                "Analyse general distribution of data",
                "Explore correlations between numeric columns",
                "Review data quality and null patterns"
            ]
        }
        return json.dumps(mock_response)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic chat-completion provider (stub)."""

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Raise NotImplementedError until Anthropic support is added."""
        raise NotImplementedError("Anthropic provider not yet implemented.")


def get_llm_provider() -> BaseLLMProvider:
    """Instantiate and return the provider configured in settings. 
    Falls back to StubProvider if configuration is missing.
    """
    provider = settings.llm_provider
    try:
        if provider == "azure_openai":
            return AzureOpenAIProvider()
        if provider == "groq":
            return GroqProvider()
        if provider == "ollama":
            return OllamaProvider()
        if provider == "openai":
            return OpenAIProvider()
        if provider == "anthropic":
            return AnthropicProvider()
    except ValueError as exc:
        import logging
        logging.getLogger(__name__).warning("LLM provider %s misconfigured: %s -- using StubProvider fallback", provider, exc)
    
    return StubProvider()
