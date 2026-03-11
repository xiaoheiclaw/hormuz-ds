"""OpenClaw agent backend."""
from __future__ import annotations
import httpx


class OpenClawBackend:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def complete(self, prompt: str, system: str | None = None) -> str:
        raise NotImplementedError("OpenClaw backend: implement when needed")
