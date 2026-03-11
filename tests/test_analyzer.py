"""Tests for LLM-based observation extractor."""
import json

import pytest
from datetime import datetime, UTC
from unittest.mock import AsyncMock

from hormuz.analyzer import Analyzer
from hormuz.llm import LLMBackend
from hormuz.models import Observation


@pytest.fixture
def mock_backend():
    backend = AsyncMock(spec=LLMBackend)
    return backend


@pytest.fixture
def analyzer(mock_backend):
    return Analyzer(mock_backend)


class TestExtraction:
    @pytest.mark.asyncio
    async def test_extracts_attack_event(self, analyzer, mock_backend):
        """CENTCOM article -> attack observation extracted."""
        mock_backend.complete.return_value = json.dumps({
            "observations": [{
                "category": "q1_attack",
                "key": "attack_event",
                "value": 1.0,
                "metadata": {"type": "missile", "target": "tanker", "region": "strait"},
            }]
        })

        articles = [{"title": "CENTCOM: Missile strike on tanker", "summary": "...", "content": "...", "source_url": "..."}]
        obs = await analyzer.extract(articles)
        assert len(obs) == 1
        assert obs[0].category == "q1_attack"
        assert obs[0].metadata["type"] == "missile"

    @pytest.mark.asyncio
    async def test_extracts_multiple_observations(self, analyzer, mock_backend):
        mock_backend.complete.return_value = json.dumps({
            "observations": [
                {"category": "q1_attack", "key": "attack_event", "value": 1.0, "metadata": {"type": "drone"}},
                {"category": "q1_attack", "key": "attack_frequency", "value": 3.5, "metadata": {}},
            ]
        })

        obs = await analyzer.extract([{"title": "...", "summary": "...", "content": "...", "source_url": "..."}])
        assert len(obs) == 2

    @pytest.mark.asyncio
    async def test_handles_code_fence_response(self, analyzer, mock_backend):
        """LLM wraps JSON in markdown code fence."""
        mock_backend.complete.return_value = '```json\n{"observations": [{"category": "market", "key": "transit_volume", "value": 5.0}]}\n```'

        obs = await analyzer.extract([{"title": "...", "summary": "...", "content": "...", "source_url": "..."}])
        assert len(obs) == 1
        assert obs[0].key == "transit_volume"

    @pytest.mark.asyncio
    async def test_graceful_on_llm_failure(self, analyzer, mock_backend):
        """LLM returns garbage -> empty list, no crash."""
        mock_backend.complete.return_value = "I don't understand"
        obs = await analyzer.extract([{"title": "...", "summary": "...", "content": "...", "source_url": "..."}])
        assert obs == []

    @pytest.mark.asyncio
    async def test_graceful_on_exception(self, analyzer, mock_backend):
        """LLM throws exception -> empty list, no crash."""
        mock_backend.complete.side_effect = Exception("API error")
        obs = await analyzer.extract([{"title": "...", "summary": "...", "content": "...", "source_url": "..."}])
        assert obs == []

    @pytest.mark.asyncio
    async def test_empty_articles_returns_empty(self, analyzer, mock_backend):
        obs = await analyzer.extract([])
        assert obs == []
        mock_backend.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_invalid_category(self, analyzer, mock_backend):
        """Observations with invalid category are skipped."""
        mock_backend.complete.return_value = json.dumps({
            "observations": [
                {"category": "invalid_cat", "key": "foo", "value": 1.0},
                {"category": "q1_attack", "key": "attack_event", "value": 1.0, "metadata": {}},
            ]
        })
        obs = await analyzer.extract([{"title": "...", "summary": "...", "content": "...", "source_url": "..."}])
        assert len(obs) == 1
        assert obs[0].category == "q1_attack"

    @pytest.mark.asyncio
    async def test_multiple_articles(self, analyzer, mock_backend):
        """Each article gets its own LLM call."""
        mock_backend.complete.side_effect = [
            json.dumps({"observations": [{"category": "q1_attack", "key": "attack_event", "value": 1.0}]}),
            json.dumps({"observations": [{"category": "market", "key": "war_risk_premium", "value": 2.5}]}),
        ]
        obs = await analyzer.extract([
            {"title": "A1", "summary": "...", "content": "...", "source_url": "..."},
            {"title": "A2", "summary": "...", "content": "...", "source_url": "..."},
        ])
        assert len(obs) == 2
        assert mock_backend.complete.call_count == 2


class TestPromptBuilding:
    def test_prompt_contains_article_content(self, analyzer):
        prompt = analyzer._build_prompt({
            "title": "Test Title",
            "summary": "Test Summary",
            "content": "Full content here",
            "source_url": "https://...",
        })
        assert "Test Title" in prompt
        assert "Full content here" in prompt

    def test_prompt_defines_output_schema(self, analyzer):
        prompt = analyzer._build_prompt({
            "title": "...",
            "summary": "...",
            "content": "...",
            "source_url": "...",
        })
        assert "q1_attack" in prompt
        assert "q2_mine" in prompt
        assert "observations" in prompt
