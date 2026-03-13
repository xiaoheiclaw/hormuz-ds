"""Observation analyzer — extract structured observations from articles via LLM."""

from __future__ import annotations

from datetime import datetime

from hormuz.core.types import Observation
from hormuz.infra.llm import LLMBackend

_EXTRACTION_PROMPT = """You are an intelligence analyst for the Hormuz Strait crisis monitoring system.

Extract observations from the following articles. For each observation, provide:
- id: one of O01-O13 (attack frequency, coordination, ammo substitution, GPS spoofing, etc.)
- value: numeric value (0-1 scale for qualitative, actual value for quantitative)
- confidence: "high", "medium", or "low"
- direction: "H1" (depletion) or "H2" (preservation)

Return JSON: {"observations": [{"id": "O01", "value": 3.0, "confidence": "high", "direction": "H1"}, ...]}

Observation IDs:
- O01: attack_frequency (attacks per day)
- O02: attack_frequency_2nd_derivative (acceleration of decline)
- O03: attack_coordination (0-1, degrading=H1, maintains sync=H2)
- O04: ammo_substitution_ratio (0-1, high-end extinct=H1, retains=H2)
- O05: gps_spoofing_complexity (0-1, degrades=H1, maintains=H2)
- O06: mosaic_fragmentation (0-1, isolated=H1, multi-node=H2)
- O07: ap_premium_pct (insurance premium percentage)
- O08: pni_status (P&I club status indicator)
- O09: vlcc_td3_rate (freight rate)
- O10: strait_daily_transit (daily vessel count)
- O11: yanbu_ais_loading (pipeline flow indicator)
- O12: fujairah_sg_spread (fuel oil spread)
- O13: spr_actual_release (DOE weekly report mbd)
"""


async def extract_observations(
    articles: list[dict],
    llm: LLMBackend,
    timestamp: datetime | None = None,
) -> list[Observation]:
    """Extract O01-O13 observations from articles using LLM."""
    if not articles:
        return []

    ts = timestamp or datetime.now()

    # Build text from articles
    text_parts = []
    for a in articles:
        text_parts.append(f"[{a.get('source', 'unknown')}] {a.get('title', '')}\n{a.get('summary', '')}")
    text = "\n\n---\n\n".join(text_parts)

    result = await llm.extract(text, _EXTRACTION_PROMPT)

    observations = []
    for item in result.get("observations", []):
        observations.append(Observation(
            id=item["id"],
            timestamp=ts,
            value=float(item["value"]),
            source=f"llm:{item.get('confidence', 'unknown')}",
            noise_note=item.get("direction"),
        ))

    return observations
