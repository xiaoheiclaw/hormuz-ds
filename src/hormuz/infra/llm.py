"""LLM backend — Protocol + Claude API / OpenClaw implementations.

Factory function selects backend based on config.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

import httpx


@runtime_checkable
class LLMBackend(Protocol):
    async def extract(self, text: str, prompt: str) -> dict: ...


class ClaudeAPIBackend:
    """Claude API backend via httpx."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str = "",
        base_url: str = "https://api.anthropic.com",
        proxy: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.proxy = proxy

    async def extract(self, text: str, prompt: str) -> dict:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": f"{prompt}\n\n---\n{text}"}],
        }
        async with httpx.AsyncClient(proxy=self.proxy, timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/v1/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            content = resp.json()["content"][0]["text"]
            # Extract JSON from response
            return json.loads(content)


class OpenClawBackend:
    """OpenClaw agent backend."""

    def __init__(self, endpoint: str = "http://localhost:8080"):
        self.endpoint = endpoint

    async def extract(self, text: str, prompt: str) -> dict:
        payload = {"text": text, "prompt": prompt}
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.endpoint}/extract", json=payload)
            resp.raise_for_status()
            return resp.json()


def create_llm_backend(backend_type: str, **kwargs) -> LLMBackend:
    """Factory function to create LLM backend."""
    if backend_type == "claude_api":
        return ClaudeAPIBackend(
            model=kwargs.get("model", "claude-sonnet-4-6"),
            api_key=kwargs.get("api_key", ""),
            base_url=kwargs.get("base_url", "https://api.anthropic.com"),
            proxy=kwargs.get("proxy"),
        )
    elif backend_type == "openclaw":
        return OpenClawBackend(endpoint=kwargs.get("endpoint", "http://localhost:8080"))
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
