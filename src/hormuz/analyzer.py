"""LLM-based observation extractor.

Converts raw articles into structured Observation objects via LLM extraction.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, UTC

from hormuz.llm import LLMBackend
from hormuz.models import Observation, VALID_CATEGORIES

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an intelligence analyst specializing in Strait of Hormuz maritime security.
Extract structured observations from the given article.

Return a JSON object with an "observations" array. Each observation has:
- "category": one of "q1_attack", "q2_mine", "q3_buffer", "market", "schelling"
- "key": observation key (see below)
- "value": numeric value (float)
- "metadata": object with additional context (can be empty {})

Valid keys by category:
- q1_attack: attack_event (value=1.0, metadata: type/target/region/coordinated), attack_frequency (value: attacks per day)
- q2_mine: mine_strike, new_area_mining, mine_in_cleared_lane (value=1.0, metadata: location/details)
- q3_buffer: spr_release, pipeline_flow (value: numeric amount, metadata: details)
- market: transit_volume, war_risk_premium (value: numeric, metadata: source/details)
- schelling: t1_platform_movement, t2_multi_region, t3_deliberate_corridor, e1_civilian_casualty, e2_environmental, e3_coalition_split, e4_domestic_backlash, c1_ceasefire, c2_un_resolution (value=1.0, metadata: details)

Only extract observations that are clearly supported by the article.
If the article contains no relevant observations, return {"observations": []}.
Return ONLY the JSON object, no other text.
"""


class Analyzer:
    """Extracts structured observations from articles using an LLM backend."""

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    async def extract(self, articles: list[dict]) -> list[Observation]:
        """Extract structured observations from articles using LLM.

        Processes each article independently. On any LLM failure,
        logs a warning and continues (graceful degradation).
        """
        if not articles:
            return []

        all_obs: list[Observation] = []
        now = datetime.now(UTC)

        for article in articles:
            try:
                prompt = self._build_prompt(article)
                response = await self._backend.complete(prompt, system=SYSTEM_PROMPT)
                obs = self._parse_response(response, now)
                all_obs.extend(obs)
            except Exception:
                logger.warning(
                    "Failed to extract observations from article: %s",
                    article.get("title", "<no title>"),
                    exc_info=True,
                )

        return all_obs

    def _build_prompt(self, article: dict) -> str:
        """Build extraction prompt with article content."""
        title = article.get("title", "")
        summary = article.get("summary", "")
        content = article.get("content", "")
        source_url = article.get("source_url", "")

        return (
            f"# Article\n\n"
            f"**Title:** {title}\n"
            f"**Source:** {source_url}\n"
            f"**Summary:** {summary}\n\n"
            f"## Content\n\n{content}\n\n"
            f"---\n"
            f"Extract all observations from this article.\n"
            f"Return JSON with an \"observations\" array.\n"
            f"Categories: q1_attack, q2_mine, q3_buffer, market, schelling\n"
        )

    def _parse_response(self, response: str, timestamp: datetime) -> list[Observation]:
        """Parse LLM JSON response into Observation objects.

        Handles JSON wrapped in markdown code fences, missing fields,
        and invalid values. Returns empty list on parse failure.
        """
        # Strip markdown code fences
        text = response.strip()
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON: %.200s", response)
            return []

        raw_obs = data.get("observations", [])
        if not isinstance(raw_obs, list):
            logger.warning("'observations' field is not a list")
            return []

        results: list[Observation] = []
        for item in raw_obs:
            try:
                category = item.get("category", "")
                if category not in VALID_CATEGORIES:
                    logger.warning("Skipping observation with invalid category: %s", category)
                    continue

                obs = Observation(
                    timestamp=timestamp,
                    source="llm",
                    category=category,
                    key=item.get("key", ""),
                    value=float(item.get("value", 0.0)),
                    metadata=item.get("metadata") or None,
                )
                results.append(obs)
            except Exception:
                logger.warning("Skipping invalid observation: %s", item, exc_info=True)

        return results
