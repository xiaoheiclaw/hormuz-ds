"""Claude API direct backend."""
from __future__ import annotations
import httpx


class ClaudeAPIBackend:
    def __init__(self, model: str, proxy: str | None = None):
        self.model = model
        self.proxy = proxy

    async def complete(self, prompt: str, system: str | None = None) -> str:
        raise NotImplementedError("Claude API backend: implement when needed")
