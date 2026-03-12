"""M1: ACH Bayesian inference engine — PRD §3.2.

Pure functions. Likelihood ratios ∈ {0.2, 0.5, 1.0, 2.0, 5.0}.
T1a/T1b unbinding: O05 (GPS spoofing) interpretation depends on O01 trend.
"""

from __future__ import annotations

from hormuz.core.types import ACHPosterior, Observation


# ── Likelihood ratio table ────────────────────────────────────────────
# Mapping: (obs_id, direction) -> {"H1": lr, "H2": lr}
# direction: "high" = value > 0.5, "low" = value <= 0.5

_LR_TABLE: dict[str, dict[str, dict[str, float]]] = {
    # O01: attack_frequency — high = many attacks (H2 preserved), low = declining (H1)
    "O01": {
        "high": {"H1": 0.5, "H2": 2.0},
        "low":  {"H1": 2.0, "H2": 0.5},
    },
    # O02: attack_frequency_2nd_derivative — high = accelerating decline (H1)
    "O02": {
        "high": {"H1": 5.0, "H2": 0.2},
        "low":  {"H1": 0.5, "H2": 2.0},
    },
    # O03: attack_coordination — high = degrading (H1), low = maintains sync (H2)
    "O03": {
        "high": {"H1": 5.0, "H2": 0.2},
        "low":  {"H1": 0.2, "H2": 5.0},
    },
    # O04: ammo_substitution_ratio — high = high-end extinct (H1)
    "O04": {
        "high": {"H1": 5.0, "H2": 0.2},
        "low":  {"H1": 0.2, "H2": 5.0},
    },
    # O05: GPS spoofing — handled specially via T1a/T1b unbinding
    # O06: mosaic_fragmentation — high = deep mountain only (H1)
    "O06": {
        "high": {"H1": 2.0, "H2": 0.5},
        "low":  {"H1": 0.5, "H2": 2.0},
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
                return {"H1": 0.2, "H2": 5.0}
            elif trend == "falling":
                # T1b: defensive H2
                return {"H1": 0.5, "H2": 2.0}
            else:
                # Unknown trend, moderate H2
                return {"H1": 0.5, "H2": 2.0}
        else:
            # GPS degrading -> H1
            return {"H1": 3.0, "H2": 0.5}

    direction = "high" if value > 0.5 else "low"
    if obs_id in _LR_TABLE:
        return _LR_TABLE[obs_id][direction]

    # Unknown observation — neutral
    return {"H1": 1.0, "H2": 1.0}


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
) -> ACHPosterior:
    """Run full ACH: sequential Bayes updates over all observations."""
    prior = compute_prior(h3_suspended, h3_prior)

    # Working posterior (only active hypotheses)
    posterior = {k: v for k, v in prior.items() if v is not None}

    for obs in observations:
        context = {}  # Could be enriched by caller with trend data
        lr = get_likelihood_ratio(obs.id, obs.value, context)
        posterior = bayesian_update(posterior, lr)

    return ACHPosterior(
        h1=posterior.get("H1", 0.0),
        h2=posterior.get("H2", 0.0),
        h3=posterior.get("H3") if not h3_suspended else None,
    )


def map_to_decay_rate(posterior: ACHPosterior) -> float:
    """Map ACH posterior to capability decay rate.

    Linear interpolation in [0.02, 0.08] based on P(H1).
    H1 dominant (depletion) -> high decay rate.
    H2 dominant (preserved) -> low decay rate.
    """
    return 0.02 + (0.08 - 0.02) * posterior.h1
