"""LLM backend abstraction."""
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    async def complete(self, prompt: str, system: str | None = None) -> str: ...


def get_backend(config: dict) -> LLMBackend:
    backend_name = config["llm"]["backend"]
    if backend_name == "claude_api":
        from hormuz.llm.claude_api import ClaudeAPIBackend
        cfg = config["llm"]["claude_api"]
        return ClaudeAPIBackend(model=cfg["model"], proxy=cfg.get("proxy"))
    elif backend_name == "openclaw":
        from hormuz.llm.openclaw import OpenClawBackend
        cfg = config["llm"]["openclaw"]
        return OpenClawBackend(endpoint=cfg["endpoint"])
    else:
        raise ValueError(f"Unknown LLM backend: {backend_name}")
