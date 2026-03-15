"""M1: ACH Bayesian inference engine — PRD §3.2.

Pure functions. Log-odds space with correlation grouping.
T1a/T1b unbinding: O05 (GPS spoofing) interpretation depends on O01 trend.
Schelling/game-theory signals handled by M5 only (credibility framework).
"""

from __future__ import annotations

import math

from hormuz.core.types import ACHPosterior, Observation


# ── Likelihood ratio table ────────────────────────────────────────────
# LR range: {0.77, 0.95, 1.0, 1.05, 1.3}
# "strong" = 1.3/0.77 (odds ratio ~1.7:1), "moderate" = 1.05/0.95 (~1.1:1)
#
# Weight by PREDICTIVE POWER for blockade persistence, not current severity:
#   STRONG  — capability (O03/O04/O06), trend (O02)
#   MODERATE — snapshot (O01/O10), market (O07-O09/O12)
#
# O11 (pipeline diversion) and O13 (SPR release) are EXCLUDED from ACH.
# They are buffer/response variables — belong in M3, not causal evidence.

_LR_TABLE: dict[str, dict[str, dict[str, float]]] = {
    # ── CAPABILITY indicators (predict "can they persist?") — STRONG ──
    "O03": {  # attack_coordination — intact C2 = can sustain
        "high": {"H1": 0.77, "H2": 1.3},
        "low":  {"H1": 1.3, "H2": 0.77},
    },
    "O04": {  # advanced_weapon_use — arsenal depth
        "high": {"H1": 0.77, "H2": 1.3},
        "low":  {"H1": 1.3, "H2": 0.77},
    },
    "O06": {  # network_fragmentation — resilient = sustainable
        "high": {"H1": 0.77, "H2": 1.3},
        "low":  {"H1": 1.3, "H2": 0.77},
    },
    # ── TREND indicator — STRONG ──────────────────────────────────────
    "O02": {  # attack_decline_rate — declining = depleting
        "high": {"H1": 1.3, "H2": 0.77},
        "low":  {"H1": 0.77, "H2": 1.3},
    },
    # ── SNAPSHOT indicators — MODERATE ────────────────────────────────
    "O01": {  # attack_frequency — current state, not persistence predictor
        "high": {"H1": 0.95, "H2": 1.05},
        "low":  {"H1": 1.05, "H2": 0.95},
    },
    # O05: GPS spoofing — handled specially via T1a/T1b unbinding
    "O10": {  # strait_daily_transit — blockade result, not cause
        "high": {"H1": 1.05, "H2": 0.95},
        "low":  {"H1": 0.95, "H2": 1.05},
    },
    # ── MARKET indicators — MODERATE (lagging) ────────────────────────
    "O07": {  # war_risk_insurance
        "high": {"H1": 0.95, "H2": 1.05},
        "low":  {"H1": 1.05, "H2": 0.95},
    },
    "O08": {  # pni_exclusion
        "high": {"H1": 0.95, "H2": 1.05},
        "low":  {"H1": 1.05, "H2": 0.95},
    },
    "O09": {  # vlcc_freight_rate
        "high": {"H1": 0.95, "H2": 1.05},
        "low":  {"H1": 1.05, "H2": 0.95},
    },
    "O12": {  # fujairah_singapore_spread
        "high": {"H1": 0.95, "H2": 1.05},
        "low":  {"H1": 1.05, "H2": 0.95},
    },
    # ── H3 indicator (external resupply) ─────────────────────────────
    "O14": {  # unknown_weapon_type — novel weapons suggest external supply
        "high": {"H1": 0.77, "H2": 0.95, "H3": 1.3},
        "low":  {"H1": 1.05, "H2": 1.0, "H3": 0.77},
    },
}


# ── Correlation groups — prevent double-counting ─────────────────────
# Correlated observations vote as a GROUP, not individually.
# capability: take top 2 of 3 (allow one to be an outlier)
# market: 5 indicators measure ~same thing (strait closed) → single vote (max |delta|)
_CORRELATION_GROUPS: dict[str, dict] = {
    "capability": {"ids": ["O03", "O04", "O06"], "top_n": 2},
    "market":     {"ids": ["O07", "O08", "O09", "O12"], "top_n": 1},
}
# Independent (not grouped): O01, O02, O05, O10, O14
# O10 (strait transit) is a physical snapshot, not a lagging market indicator.


# ── Custom thresholds for non-0-1 observations ───────────────────────
_CUSTOM_THRESHOLDS: dict[str, float] = {
    "O07": 0.5,    # AP >= 0.5% = crisis level (peacetime 0.05-0.07%)
    "O09": 150.0,  # WS >= 150 = elevated freight
    "O12": 50.0,   # spread >= $50/mt = logistics stress
}


# Max log-odds magnitude: |log(95/5)| ≈ 2.94 → posterior capped at ~95%
_MAX_LOG_ODDS = math.log(95.0 / 5.0)


def compute_prior(h3_suspended: bool, h3_prior: float) -> dict[str, float | None]:
    """Compute ACH prior distribution.

    When H3 suspended: H1 = H2 = 0.5 (clean 50:50).
    When H3 active: H1 = H2 = (1 - h3_prior) / 2.
    """
    if h3_suspended:
        return {"H1": 0.5, "H2": 0.5, "H3": None}
    else:
        half = (1.0 - h3_prior) / 2
        return {"H1": half, "H2": half, "H3": h3_prior}


def get_likelihood_ratio(obs_id: str, value: float, context: dict) -> dict[str, float]:
    """Get LR for a single observation.

    O05 (GPS spoofing) uses T1a/T1b unbinding:
    - O05 high + O01_trend rising  → strong H2 (offensive EW)
    - O05 high + O01_trend falling → moderate H2 (defensive EW)
    - O05 low → strong H1 (EW capability degrading)
    """
    if obs_id == "O05":
        if value > 0.5:
            trend = context.get("O01_trend", "unknown")
            if trend == "rising":
                return {"H1": 0.77, "H2": 1.3}
            elif trend == "unknown":
                return {"H1": 1.0, "H2": 1.0}  # no trend data → neutral
            else:
                return {"H1": 0.95, "H2": 1.05}
        else:
            return {"H1": 1.3, "H2": 0.77}

    if obs_id in _CUSTOM_THRESHOLDS:
        threshold = _CUSTOM_THRESHOLDS[obs_id]
        direction = "high" if value >= threshold else "low"
    else:
        direction = "high" if value > 0.5 else "low"

    if obs_id in _LR_TABLE:
        return _LR_TABLE[obs_id][direction]

    # Unknown observation — neutral
    return {"H1": 1.0, "H2": 1.0}


def _compute_log_lr(lr: dict[str, float]) -> float:
    """Compute log(LR_H1 / LR_H2) for a single LR dict."""
    h1 = lr.get("H1", 1.0)
    h2 = lr.get("H2", 1.0)
    if h1 > 0 and h2 > 0:
        return math.log(h1 / h2)
    return 0.0


def _apply_correlation_grouping(
    observations: list[Observation],
    context: dict,
) -> tuple[list[float], list[dict[str, float]]]:
    """Apply correlation grouping, return (log-LR deltas, raw LR dicts).

    Returns two parallel lists:
    - log_deltas: log(LR_H1/LR_H2) for 2-way path
    - lr_dicts: original LR dicts with all hypothesis keys for 3-way path

    Grouped observations: only top_n strongest from each group contribute.
    Independent observations: all contribute directly.
    """
    # Collect (delta, lr_dict) pairs by group
    group_items: dict[str, list[tuple[float, dict]]] = {
        name: [] for name in _CORRELATION_GROUPS
    }
    independent_items: list[tuple[float, dict]] = []

    for obs in observations:
        lr = get_likelihood_ratio(obs.id, obs.value, context)
        delta = _compute_log_lr(lr)
        # Check if any hypothesis has non-neutral LR (not just H1/H2)
        has_h3_evidence = any(abs(math.log(v)) > 1e-10 for k, v in lr.items() if k == "H3" and v > 0)
        if abs(delta) < 1e-10 and not has_h3_evidence:
            continue  # neutral across all hypotheses, skip

        # Check if belongs to a group
        found_group = None
        for name, g in _CORRELATION_GROUPS.items():
            if obs.id in g["ids"]:
                found_group = name
                break

        item = (delta, lr)
        if found_group:
            group_items[found_group].append(item)
        else:
            independent_items.append(item)

    # Take top_n from each group (by absolute log-delta magnitude)
    result_items: list[tuple[float, dict]] = list(independent_items)
    for name, items in group_items.items():
        if not items:
            continue
        top_n = _CORRELATION_GROUPS[name]["top_n"]
        sorted_items = sorted(items, key=lambda x: abs(x[0]), reverse=True)
        result_items.extend(sorted_items[:top_n])

    log_deltas = [d for d, _ in result_items]
    lr_dicts = [lr for _, lr in result_items]
    return log_deltas, lr_dicts


def run_ach(
    observations: list[Observation],
    h3_suspended: bool = True,
    h3_prior: float = 0.10,
    max_log_odds: float = _MAX_LOG_ODDS,
    o01_trend: str = "stable",
) -> ACHPosterior:
    """Run ACH in log-odds space with correlation grouping.

    1. Compute prior (50:50 when H3 suspended)
    2. Compute observation LRs with correlation grouping (prevent double-counting)
    3. Accumulate in log-odds space, clamp to 95% cap
    4. Convert back to probabilities

    Schelling/game-theory signals are handled by M5 (adjust_path_weights),
    which has the mature credibility × focal convergence framework.
    """
    prior = compute_prior(h3_suspended, h3_prior)
    ach_context = {"O01_trend": o01_trend}

    active = {k: v for k, v in prior.items() if v is not None}
    keys = sorted(active.keys())

    if len(keys) == 2:
        k0, k1 = keys  # H1, H2
        log_odds = math.log(active[k0] / active[k1]) if active[k1] > 0 else 0.0

        # 1. Observation evidence with correlation grouping
        deltas, _ = _apply_correlation_grouping(observations, ach_context)
        for d in deltas:
            log_odds += d

        # 2. Clamp
        log_odds = max(-max_log_odds, min(max_log_odds, log_odds))

        # 3. Convert back
        odds = math.exp(log_odds)
        p0 = odds / (1.0 + odds)
        p1 = 1.0 / (1.0 + odds)

        return ACHPosterior(
            h1=p0 if k0 == "H1" else p1,
            h2=p1 if k0 == "H1" else p0,
            h3=None if h3_suspended else prior.get("H3"),
        )
    else:
        # 3-way (H3 active): correlation grouping + cap
        posterior_dict = dict(active)

        # Apply grouped observation evidence (use raw LR dicts with H3 keys)
        _, lr_dicts = _apply_correlation_grouping(observations, ach_context)
        for lr in lr_dicts:
            # Ensure all hypothesis keys present (default neutral for missing)
            full_lr = {k: lr.get(k, 1.0) for k in posterior_dict}
            posterior_dict = bayesian_update(posterior_dict, full_lr)

        # Cap: no single hypothesis > 95%
        for k in posterior_dict:
            posterior_dict[k] = max(0.02, min(0.95, posterior_dict[k]))
        total = sum(posterior_dict.values())
        posterior_dict = {k: v / total for k, v in posterior_dict.items()}

        return ACHPosterior(
            h1=posterior_dict.get("H1", 0.0),
            h2=posterior_dict.get("H2", 0.0),
            h3=posterior_dict.get("H3"),
        )


def bayesian_update(prior: dict[str, float], lr: dict[str, float]) -> dict[str, float]:
    """Single-step Bayes update + normalize (used for 3-way H3 case)."""
    active_keys = [k for k in prior if prior[k] is not None and k in lr]
    unnormalized = {k: prior[k] * lr[k] for k in active_keys}
    total = sum(unnormalized.values())
    if total == 0:
        return {k: 1.0 / len(active_keys) for k in active_keys}
    return {k: v / total for k, v in unnormalized.items()}
