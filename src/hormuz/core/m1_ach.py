"""M1: ACH Bayesian inference engine — PRD §3.2.

Pure functions. Likelihood ratios ∈ {0.2, 0.5, 1.0, 2.0, 5.0}.
T1a/T1b unbinding: O05 (GPS spoofing) interpretation depends on O01 trend.
"""

from __future__ import annotations

from hormuz.core.types import ACHPosterior, Observation


# ── Likelihood ratio table ────────────────────────────────────────────
# Mapping: (obs_id, direction) -> {"H1": lr, "H2": lr}
# direction: "high" = value > 0.5, "low" = value <= 0.5

# LR range: {0.77, 0.95, 1.0, 1.05, 1.3}
# "strong" = 1.3/0.77 (odds ratio ~1.7:1), "moderate" = 1.05/0.95 (odds ratio ~1.1:1)
# Single moderate: 50% → 53% (+3pp). Single strong: 50% → 63% (+13pp).
# 3 strong concordant → ~83%. 5 strong concordant → ~92%. Cap at 95%.

_LR_TABLE: dict[str, dict[str, dict[str, float]]] = {
    # O01: attack_frequency — high=frequent (H2), low=declining (H1). Moderate.
    "O01": {
        "high": {"H1": 0.95, "H2": 1.05},
        "low":  {"H1": 1.05, "H2": 0.95},
    },
    # O02: attack_decline_rate — high=rapid decline (H1), low=stable (H2). Strong.
    "O02": {
        "high": {"H1": 1.3, "H2": 0.77},
        "low":  {"H1": 0.95, "H2": 1.05},
    },
    # O03: attack_coordination — high=coordinated (H2), low=fragmented (H1). Strong.
    "O03": {
        "high": {"H1": 0.77, "H2": 1.3},
        "low":  {"H1": 1.3, "H2": 0.77},
    },
    # O04: advanced_weapon_use — high=advanced (H2), low=crude only (H1). Strong.
    "O04": {
        "high": {"H1": 0.77, "H2": 1.3},
        "low":  {"H1": 1.3, "H2": 0.77},
    },
    # O05: GPS spoofing — handled specially via T1a/T1b unbinding
    # O06: network_fragmentation — high=intact (H2), low=fragmented (H1). Moderate.
    "O06": {
        "high": {"H1": 0.95, "H2": 1.05},
        "low":  {"H1": 1.05, "H2": 0.95},
    },
    # O07: war_risk_insurance — high=elevated (H2), low=normal (H1). Moderate (lagging).
    "O07": {
        "high": {"H1": 0.95, "H2": 1.05},
        "low":  {"H1": 1.05, "H2": 0.95},
    },
    # O08: pni_exclusion — high=excluded (H2), low=normal (H1). Moderate.
    "O08": {
        "high": {"H1": 0.95, "H2": 1.05},
        "low":  {"H1": 1.05, "H2": 0.95},
    },
    # O10: strait_daily_transit — high=normal traffic (H1), low=halted (H2). Moderate.
    "O10": {
        "high": {"H1": 1.05, "H2": 0.95},
        "low":  {"H1": 0.95, "H2": 1.05},
    },
    # O11: pipeline_diversion — high=active (H2), low=none (H1). Moderate.
    "O11": {
        "high": {"H1": 0.95, "H2": 1.05},
        "low":  {"H1": 1.05, "H2": 0.95},
    },
}


def compute_prior(h3_suspended: bool, h3_prior: float) -> dict[str, float | None]:
    """Compute ACH prior distribution.

    When H3 suspended: redistribute h3_prior equally to H1/H2.
    When H3 active: H1 = H2 = (1 - h3_prior) / 2.
    """
    if h3_suspended:
        # H1=H2 = (1 - h3_prior) / 2 = 0.45, then +h3_prior/4 each = 0.475
        # Remaining 5% is "suspended mass" — absorbed during normalization in update
        base = (1.0 - h3_prior) / 2 + h3_prior / 4
        return {"H1": base, "H2": base, "H3": None}
    else:
        half = (1.0 - h3_prior) / 2
        return {"H1": half, "H2": half, "H3": h3_prior}


def get_likelihood_ratio(obs_id: str, value: float, context: dict) -> dict[str, float]:
    """Get LR for a single observation.

    O05 (GPS spoofing) uses T1a/T1b unbinding:
    - O05 high + O01_trend rising -> T1a: offensive H2, LR(H2)=5.0
    - O05 high + O01_trend falling -> T1b: defensive H2, LR(H2)=2.0
    - O05 low -> LR(H1)=3.0 regardless of O01
    """
    if obs_id == "O05":
        if value > 0.5:
            trend = context.get("O01_trend", "unknown")
            if trend == "rising":
                # T1a: offensive H2
                return {"H1": 0.77, "H2": 1.3}
            elif trend == "falling":
                # T1b: defensive H2
                return {"H1": 0.95, "H2": 1.05}
            else:
                # Unknown trend, moderate H2
                return {"H1": 0.95, "H2": 1.05}
        else:
            # GPS degrading -> H1
            return {"H1": 1.3, "H2": 0.8}

    # Non-0-1 observations need custom thresholds for high/low direction
    # O07: AP premium (%), >1% = "high" (blockade effective)
    # O09: VLCC freight (WS points), >150 = "high" (logistics frozen)
    # O12: Fujairah-Singapore spread ($/mt), >50 = "high" (logistics breakdown)
    # O13: SPR release (mbd), >1.0 = "high" (active release)
    _CUSTOM_THRESHOLDS: dict[str, float] = {
        "O07": 1.0,    # AP > 1% = blockade confirmed
        "O09": 150.0,  # WS > 150 = elevated freight
        "O12": 50.0,   # spread > $50/mt = logistics stress
        "O13": 1.0,    # > 1 mbd = meaningful release
    }

    if obs_id in _CUSTOM_THRESHOLDS:
        threshold = _CUSTOM_THRESHOLDS[obs_id]
        direction = "high" if value > threshold else "low"
    else:
        direction = "high" if value > 0.5 else "low"

    if obs_id in _LR_TABLE:
        return _LR_TABLE[obs_id][direction]

    # Unknown observation — neutral
    return {"H1": 1.0, "H2": 1.0}


import math

# Max log-odds magnitude: |log(95/5)| ≈ 2.94 → posterior capped at ~95%
_MAX_LOG_ODDS = math.log(95.0 / 5.0)


def bayesian_update(prior: dict[str, float], lr: dict[str, float]) -> dict[str, float]:
    """Single-step Bayes update + normalize.

    P(Hi|E) ∝ P(Hi) × LR(Hi)
    Only updates keys present in both prior and lr (skips None/H3 when suspended).
    """
    active_keys = [k for k in prior if prior[k] is not None and k in lr]
    unnormalized = {k: prior[k] * lr[k] for k in active_keys}
    total = sum(unnormalized.values())
    if total == 0:
        return {k: 1.0 / len(active_keys) for k in active_keys}
    return {k: v / total for k, v in unnormalized.items()}


def run_ach(
    observations: list[Observation],
    h3_suspended: bool = True,
    h3_prior: float = 0.10,
    max_log_odds: float = _MAX_LOG_ODDS,
    o01_trend: str = "stable",
) -> ACHPosterior:
    """Run ACH in log-odds space with saturation cap.

    Accumulates log(LR) per hypothesis, clamps total magnitude to max_log_odds,
    then converts back to probabilities. This prevents any single update cycle
    from pushing posterior past ~95% regardless of observation count.

    o01_trend: "rising"/"falling"/"stable" — used for T1a/T1b O05 unbinding.
    """
    prior = compute_prior(h3_suspended, h3_prior)

    # Build context for O05 T1a/T1b unbinding
    ach_context = {"O01_trend": o01_trend}

    # Work in log-odds space (H1 vs H2 only when H3 suspended)
    active = {k: v for k, v in prior.items() if v is not None}
    keys = sorted(active.keys())

    if len(keys) == 2:
        # Binary case: track log-odds of H1 vs H2
        k0, k1 = keys  # H1, H2
        log_odds = math.log(active[k0] / active[k1]) if active[k1] > 0 else 0.0

        for obs in observations:
            lr = get_likelihood_ratio(obs.id, obs.value, ach_context)
            # log(LR_H1 / LR_H2)
            lr0 = lr.get(k0, 1.0)
            lr1 = lr.get(k1, 1.0)
            if lr0 > 0 and lr1 > 0:
                log_odds += math.log(lr0 / lr1)

        # Clamp
        log_odds = max(-max_log_odds, min(max_log_odds, log_odds))

        # Convert back
        odds = math.exp(log_odds)
        p0 = odds / (1.0 + odds)
        p1 = 1.0 / (1.0 + odds)

        return ACHPosterior(
            h1=p0 if k0 == "H1" else p1,
            h2=p1 if k0 == "H1" else p0,
            h3=None if h3_suspended else prior.get("H3"),
        )
    else:
        # 3-way: fall back to sequential multiply (rare, only when H3 active)
        posterior = dict(active)
        for obs in observations:
            lr = get_likelihood_ratio(obs.id, obs.value, ach_context)
            posterior = bayesian_update(posterior, lr)
        return ACHPosterior(
            h1=posterior.get("H1", 0.0),
            h2=posterior.get("H2", 0.0),
            h3=posterior.get("H3"),
        )


def map_to_decay_rate(posterior: ACHPosterior) -> float:
    """Map ACH posterior to capability decay rate.

    Linear interpolation in [0.02, 0.08] based on P(H1).
    H1 dominant (depletion) -> high decay rate.
    H2 dominant (preserved) -> low decay rate.
    """
    return 0.02 + (0.08 - 0.02) * posterior.h1
