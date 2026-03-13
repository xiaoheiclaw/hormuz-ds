"""Observation analyzer — extract structured observations + Schelling signals from articles via LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from hormuz.core.types import Observation
from hormuz.core.m5_game import SignalEvidence
from hormuz.infra.llm import LLMBackend

_EVIDENCE_MAP = {"high": 1.0, "medium": 0.5, "low": 0.2}

_EXTRACTION_PROMPT_TEMPLATE = """You are an intelligence analyst for the Hormuz Strait crisis monitoring system.

## SITUATION CONTEXT
{context}

Extract observations AND Schelling game-theory signals from the following articles.
Give VALUES that reflect the CURRENT situation described in the articles.
If an article describes ongoing conflict/attacks, the values should reflect that — do NOT default to 0.
If articles don't mention a specific observation, OMIT it rather than guessing 0.

Return JSON only:
{{
  "observations": [{{"id": "O01", "value": 0.8, "confidence": "high"}}, ...],
  "signals": [{{"key": "external_mediation", "evidence": "high"}}, ...]
}}

## A-GROUP: Threat Status (feed ACH engine, 0-1 scale)
## IMPORTANT: These cover ALL Iran/IRGC military activity in the Gulf region,
## not just attacks on strait shipping. Include attacks on oil infrastructure,
## ports, Gulf states, and general Iran military operations.

O01 — attack_frequency: Iran/IRGC military attacks in the Gulf/Hormuz region in past 24h.
  Include: attacks on shipping, oil infrastructure, Gulf state territory, drone/missile strikes on ports.
  0=no attacks anywhere in region, 0.3=sporadic (1-2/day), 0.5=moderate (3-5/day), 0.8=heavy (6+/day), 1.0=saturated/continuous
  News phrases: "CENTCOM reported X attacks", "no incidents reported", "surge in attacks",
  "drone strike on port", "oil terminal attacked", "missiles fired at Saudi/UAE/Oman",
  "attacks on merchant vessels", "Gulf under wave of attacks"

O02 — attack_trend_change: are attacks rising, stable, or declining vs recent days?
  0=sharp rise/new escalation, 0.5=stable/no change, 1.0=sharp decline
  News phrases: "attacks dropped 50%", "lull in hostilities", "3 days without incident",
  "renewed wave of attacks", "escalation in frequency", "both sides dig in"

O03 — attack_coordination: tactical complexity and sophistication of recent Iran/IRGC operations.
  0=no organized attacks/amateur, 0.5=some coordination, 1.0=multi-platform synchronized operations
  News phrases: "coordinated multi-axis attack", "simultaneous strikes from multiple directions",
  "isolated lone-wolf attack", "sophisticated swarming tactics",
  "multi-vector assault", "combined drone and missile strike"

O04 — advanced_weapon_use: ratio of high-end weapons (ASCM/ASBM/cruise missiles/ballistic) vs low-end (drones/FIAC/unguided).
  0=only crude/cheap weapons, 0.5=mixed, 1.0=exclusively advanced missiles
  News phrases: "anti-ship missile fired", "only suicide drones used", "cruise missile intercepted",
  "ballistic missiles launched", "switched to cheap UAVs", "Iranian missile strikes"

O05 — gps_spoofing_complexity: electronic warfare sophistication in Gulf/strait waters.
  0=no EW activity, 0.3=basic jamming, 0.7=complex geometric spoofing, 1.0=advanced multi-frequency
  News phrases: "GPS spoofing cluster detected", "AIS anomalies", "phantom vessel tracks",
  "electronic warfare activity ceased", "jamming signals reported",
  "ships must coordinate with Iran's navy" (implies EW/control)

O06 — network_fragmentation: geographic distribution of IRGC operational capability.
  0=collapsed to few inland sites, 0.5=partial fragmentation, 1.0=distributed multi-node network active
  News phrases: "attacks only from mountain positions", "coastal launch sites destroyed",
  "multiple firing positions along coastline", "dispersed mobile launchers",
  "Iran's allies step up strikes", "Hezbollah/Houthi coordinated action"

## B-GROUP: Blockade/Recovery (0-1 scale except O07, O09)

O07 — war_risk_insurance_premium: hull war risk additional premium as % of vessel value.
  Give actual percentage (e.g., 2.5 = 2.5%). Normal peacetime <0.1%, crisis 1-5%.
  News phrases: "war risk premium surged to X%", "insurance costs soared", "AP quoted at X%",
  "hull war risk", "underwriters raised rates"

O08 — pni_exclusion: P&I club war risk coverage status.
  0=normal coverage, 0.3=surcharges added, 0.7=72h cancellation notice triggered, 1.0=full exclusion/per-voyage only
  News phrases: "P&I clubs withdrew coverage", "war risk exclusion", "per-voyage approval only",
  "insurance restored to normal", "clubs reinstated coverage"

O09 — vlcc_freight_rate: VLCC spot rate on TD3 route (Middle East→Far East), Worldscale points.
  Give WS number (e.g., 250). Normal 40-80, crisis 200-500+.
  News phrases: "VLCC rates hit WS400", "tanker rates surged", "freight market frozen",
  "no fixtures reported", "rates returning to normal"

O10 — strait_daily_transit: commercial vessel traffic through Hormuz.
  0=zero transits (full blockade), 0.5=heavily reduced (~30 ships), 1.0=normal (~60+ ships/day)
  News phrases: "only X vessels transited", "shipping traffic resumed", "strait effectively closed",
  "AIS shows near-zero traffic", "convoy of X ships passed through"

## C-GROUP: Buffer Arrival (verify alternative supply)

O11 — yanbu_ais_loading: Saudi Yanbu port crude loading activity (pipeline diversion proxy).
  0=no loading activity, 0.5=partial operations, 1.0=full capacity loading
  News phrases: "Yanbu loadings increased", "tankers queuing at Yanbu", "Red Sea exports surging",
  "Saudi East-West pipeline at capacity", "Yanbu terminal operating normally"

O12 — fujairah_singapore_spread: fuel oil price spread between Fujairah and Singapore ($/mt).
  Give actual spread value. Normal <$20, crisis >$100 signals logistics breakdown.
  News phrases: "Fujairah premium surged", "fuel oil spread widened to $X",
  "Fujairah storage hub disrupted", "bunkering prices spiked"

O13 — spr_release_rate: actual SPR release rate in million barrels/day.
  Give mbd value (e.g., 1.5). Zero until release order + 13-day pump delay.
  News phrases: "DOE released X million barrels", "SPR drawdown of X mbd",
  "strategic reserve release authorized", "emergency oil release"

## A6: H3 UNFREEZE MONITOR

O14 — unknown_weapon_type: has an unknown/new weapon type been observed that is NOT in IRGC's known inventory?
  0=no unknown weapons (normal), 1.0=confirmed new weapon type (e.g., Russian-origin missile not previously seen)
  News phrases: "previously unseen weapon", "new missile type identified", "weapon not in known Iranian arsenal",
  "foreign-supplied munitions confirmed", "debris analysis reveals unknown origin"
  NOTE: This is rare. Most days should be 0. Only report 1.0 if articles explicitly describe a NEW weapon type.

## SCHELLING SIGNALS (game theory — include in "signals" array if detected)

Only include a signal if articles contain clear evidence. Do NOT fabricate.

external_mediation — Third-party mediation effort detected (Oman, Qatar, China, UN envoy).
  News phrases: "Omani envoy", "back-channel talks", "Qatar mediation", "diplomatic shuttle",
  "UN special envoy", "China offered to mediate", "secret negotiations"

us_inconsistency — Contradictory US messaging suggesting internal policy conflict.
  News phrases: "Pentagon denied State Department claim", "mixed signals from Washington",
  "internal debate over response", "White House contradicted CENTCOM",
  "policy rift between hawks and doves"

costly_self_binding — One side makes costly commitment to de-escalation.
  News phrases: "unilateral ceasefire announced", "withdrew forces as goodwill gesture",
  "opened humanitarian corridor", "released detained vessels",
  "suspended enrichment activities", "pulled back naval assets"

irgc_escalation — IRGC escalation against oil infrastructure (cross-layer with E1).
  News phrases: "oil terminal attacked", "pipeline sabotaged", "Ras Tanura struck",
  "Fujairah port hit", "infrastructure targeted", "storage facility ablaze",
  "IRGC claims attack on Saudi oil facility"

irgc_fragmentation — Signs of IRGC internal disagreement.
  News phrases: "IRGC commander defected", "internal power struggle", "Quds Force vs Navy dispute",
  "hardliners vs pragmatists split", "reports of insubordination", "IRGC leadership shake-up"
  NOTE: Only include if US inconsistency also present (validates interpretation).

## RULES
- Extract observations you can infer from the articles. Reasonable inference is OK — do not require exact quotes.
- If articles describe active conflict but don't give exact attack counts, estimate based on intensity described (e.g., "heavy fighting" → O01 ≈ 0.7-0.8).
- If articles don't mention a specific observation AT ALL, OMIT it from the output (don't guess 0).
- Give confidence: "high" (explicit numbers/quotes), "medium" (reasonable inference), "low" (vague mention).
- O01-O06, O08, O10, O11, O14: stay within [0, 1].
- O07: percentage points. O09: WS points. O12: $/mt. O13: mbd.
- signals: array of objects with "key" and "evidence" (high/medium/low). Empty array if no signals detected. Most days will be empty.
"""


def _build_context(conflict_day: int | None = None, previous_obs: dict[str, float] | None = None) -> str:
    """Build situation context string for extraction prompt."""
    parts = []
    if conflict_day is not None and conflict_day > 0:
        parts.append(f"This is DAY {conflict_day} of the US-Israel vs Iran conflict in the Gulf region.")
        parts.append("There is an active military conflict. Attacks on shipping, oil infrastructure, and Gulf states are ongoing.")
    if previous_obs:
        lines = []
        for oid, val in sorted(previous_obs.items()):
            lines.append(f"  {oid}={val:.2f}")
        parts.append("Previous observation values (use as baseline, update based on new articles):\n" + "\n".join(lines))
    if not parts:
        parts.append("Monitor the Hormuz Strait region for crisis-related developments.")
    return "\n".join(parts)


def build_extraction_prompt(conflict_day: int | None = None, previous_obs: dict[str, float] | None = None) -> str:
    """Build the full extraction prompt with dynamic context."""
    context = _build_context(conflict_day, previous_obs)
    return _EXTRACTION_PROMPT_TEMPLATE.format(context=context)


@dataclass
class ExtractionResult:
    """Combined extraction output: observations + Schelling signals."""
    observations: list[Observation] = field(default_factory=list)
    signals: list[SignalEvidence] = field(default_factory=list)


async def _extract_batch(
    articles: list[dict],
    llm: LLMBackend,
    ts: datetime,
    conflict_day: int | None = None,
    previous_obs: dict[str, float] | None = None,
) -> tuple[list[Observation], list[SignalEvidence]]:
    """Extract observations and signals from a single batch of articles."""
    text_parts = []
    for a in articles:
        text_parts.append(f"[{a.get('source', 'unknown')}] {a.get('title', '')}\n{a.get('summary', '')}")
    text = "\n\n---\n\n".join(text_parts)

    prompt = build_extraction_prompt(conflict_day=conflict_day, previous_obs=previous_obs)
    result = await llm.extract(text, prompt)

    observations = []
    for item in result.get("observations", []):
        observations.append(Observation(
            id=item["id"],
            timestamp=ts,
            value=float(item["value"]),
            source=f"llm:{item.get('confidence', 'unknown')}",
        ))

    raw_signals = result.get("signals", [])
    if not isinstance(raw_signals, list):
        raw_signals = []

    valid_keys = {
        "external_mediation", "us_inconsistency", "costly_self_binding",
        "irgc_escalation", "irgc_fragmentation",
    }
    signals: list[SignalEvidence] = []
    for s in raw_signals:
        if isinstance(s, dict):
            key = s.get("key", "")
            evidence = _EVIDENCE_MAP.get(s.get("evidence", "low"), 0.2)
        elif isinstance(s, str):
            # Backwards compat: plain string → default medium evidence
            key = s
            evidence = 0.5
        else:
            continue
        if key in valid_keys:
            signals.append(SignalEvidence(key=key, evidence=evidence))

    return observations, signals


async def extract_observations(
    articles: list[dict],
    llm: LLMBackend,
    timestamp: datetime | None = None,
    batch_size: int = 5,
    conflict_day: int | None = None,
    previous_obs: dict[str, float] | None = None,
) -> ExtractionResult:
    """Extract O01-O14 observations + Schelling signals from articles in batches.

    Processes articles in batches of batch_size, merges results,
    and keeps the highest-confidence observation per ID.
    Returns ExtractionResult with both observations and signals.
    """
    if not articles:
        return ExtractionResult()

    ts = timestamp or datetime.now()

    # Process in batches
    all_obs: list[Observation] = []
    all_signals: list[SignalEvidence] = []
    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]
        try:
            batch_obs, batch_sigs = await _extract_batch(
                batch, llm, ts,
                conflict_day=conflict_day,
                previous_obs=previous_obs,
            )
            all_obs.extend(batch_obs)
            all_signals.extend(batch_sigs)
        except Exception:
            continue  # skip failed batch

    # Deduplicate observations: keep highest-confidence per obs ID
    _conf_rank = {"high": 3, "medium": 2, "low": 1, "unknown": 0}
    best: dict[str, tuple[Observation, int]] = {}
    for o in all_obs:
        conf = o.source.split(":")[-1] if ":" in o.source else "unknown"
        rank = _conf_rank.get(conf, 0)
        if o.id not in best:
            best[o.id] = (o, rank)
        elif rank > best[o.id][1]:
            best[o.id] = (o, rank)

    # Deduplicate signals: keep highest evidence per key
    best_sigs: dict[str, SignalEvidence] = {}
    for s in all_signals:
        if s.key not in best_sigs or s.evidence > best_sigs[s.key].evidence:
            best_sigs[s.key] = s

    return ExtractionResult(
        observations=[obs for obs, _ in best.values()],
        signals=list(best_sigs.values()),
    )
