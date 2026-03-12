"""Claude API direct backend using httpx (no SDK dependency)."""
from __future__ import annotations

import json
import os

import httpx

_DEFAULT_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class ClaudeAPIBackend:
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        proxy: str | None = None,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.proxy = proxy
        self.max_tokens = max_tokens
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
        self.api_url = f"{base_url}/v1/messages" if base_url else _DEFAULT_API_URL

    async def complete(self, prompt: str, system: str | None = None) -> str:
        """Send a single-turn completion request to the Claude API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }
        body: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        async with httpx.AsyncClient(
            proxy=self.proxy,
            timeout=60.0,
        ) as client:
            resp = await client.post(self.api_url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        # Extract text from first content block
        content = data.get("content", [])
        if content and content[0].get("type") == "text":
            return content[0]["text"]
        return ""
