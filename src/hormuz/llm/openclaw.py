"""OpenClaw agent backend — sends prompt to OpenClaw gateway."""
from __future__ import annotations

import os

import httpx


class OpenClawBackend:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def complete(self, prompt: str, system: str | None = None) -> str:
        """Send prompt to OpenClaw gateway and return response text."""
        payload: dict = {"prompt": prompt}
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(self.endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return data.get("response", data.get("text", ""))
