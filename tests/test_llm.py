"""Tests for LLM backend abstraction."""
from hormuz.llm import LLMBackend, get_backend


class TestBackendSelection:
    def test_get_claude_api_backend(self):
        config = {"llm": {"backend": "claude_api", "claude_api": {"model": "claude-sonnet-4-6", "proxy": None}}}
        backend = get_backend(config)
        assert isinstance(backend, LLMBackend)

    def test_get_openclaw_backend(self):
        config = {"llm": {"backend": "openclaw", "openclaw": {"endpoint": "http://localhost:8080"}}}
        backend = get_backend(config)
        assert isinstance(backend, LLMBackend)

    def test_invalid_backend_raises(self):
        import pytest
        config = {"llm": {"backend": "invalid"}}
        with pytest.raises(ValueError):
            get_backend(config)
