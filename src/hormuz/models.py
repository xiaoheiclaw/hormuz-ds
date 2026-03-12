"""Data models for the Hormuz decision support system.

These Pydantic models are the contracts between all pipeline stages.
"""
from datetime import datetime
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, model_validator


# --- Enums ---

class SignalStatus(StrEnum):
    inactive = "inactive"
    triggered = "triggered"
    confirmed = "confirmed"
    reverted = "reverted"


class RegimeType(StrEnum):
    wide = "wide"
    lean_h1 = "lean_h1"
    lean_h2 = "lean_h2"
    confirmed_h3 = "confirmed_h3"


# --- Valid sets ---

VALID_SOURCES = {"centcom", "readwise", "yfinance", "manual", "llm"}
VALID_CATEGORIES = {"q1_attack", "q2_mine", "q3_buffer", "market", "schelling"}
VALID_QUESTIONS = {"q1", "q2"}
VALID_CONFIDENCES = {"high", "medium", "low"}
Q1_DIRECTIONS = {"h1", "h2", "h3", "neutral"}
Q2_DIRECTIONS = {"h1", "h2", "neutral"}


# --- Models ---

class Observation(BaseModel):
    """Raw data from any source."""
    id: Optional[int] = None
    timestamp: datetime
    source: str
    category: str
    key: str
    value: float
    metadata: Optional[dict] = None

    @model_validator(mode="after")
    def _validate_enums(self):
        if self.source not in VALID_SOURCES:
            raise ValueError(f"source must be one of {VALID_SOURCES}, got '{self.source}'")
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {VALID_CATEGORIES}, got '{self.category}'")
        return self


class ACHEvidence(BaseModel):
    """Evidence for ACH matrix."""
    id: Optional[int] = None
    timestamp: datetime
    question: str
    evidence_id: int
    direction: str
    confidence: str
    notes: Optional[str] = None
    source_observation_id: Optional[int] = None

    @model_validator(mode="after")
    def _validate_fields(self):
        if self.question not in VALID_QUESTIONS:
            raise ValueError(f"question must be one of {VALID_QUESTIONS}")
        if self.confidence not in VALID_CONFIDENCES:
            raise ValueError(f"confidence must be one of {VALID_CONFIDENCES}")
        allowed = Q1_DIRECTIONS if self.question == "q1" else Q2_DIRECTIONS
        if self.direction not in allowed:
            raise ValueError(
                f"direction '{self.direction}' not allowed for {self.question}; "
                f"valid: {allowed}"
            )
        return self


class PathWeights(BaseModel):
    """Probability weights for three paths (a, b, c)."""
    a: float
    b: float
    c: float

    @model_validator(mode="after")
    def _validate_sum(self):
        total = self.a + self.b + self.c
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"weights must sum to ~1.0 (tolerance 0.02), got {total}")
        return self

    def apply_delta(self, a_delta: float = 0.0, c_delta: float = 0.0) -> "PathWeights":
        """Adjust weights: clamp a/c to [0,1], b is residual."""
        new_a = max(0.0, min(1.0, self.a + a_delta))
        new_c = max(0.0, min(1.0, self.c + c_delta))
        new_b = 1.0 - new_a - new_c
        # Clamp b and redistribute if needed
        if new_b < 0.0:
            # Scale a and c down proportionally
            total_ac = new_a + new_c
            new_a = new_a / total_ac
            new_c = new_c / total_ac
            new_b = 0.0
        return PathWeights(a=new_a, b=new_b, c=new_c)


class Regime(BaseModel):
    """Regime judgment record."""
    id: Optional[int] = None
    timestamp: datetime
    question: str
    regime: RegimeType
    trigger: str


class Signal(BaseModel):
    """Tripwire/event/confirmation signal state."""
    id: Optional[int] = None
    timestamp: datetime
    signal_id: str
    status: SignalStatus
    revert_deadline: Optional[datetime] = None
    action_taken: str


class MCParams(BaseModel):
    """Complete MC parameter snapshot.

    v5.4: disruption_range removed. Gross gap is constant 16 mbd (step function).
    irgc_decay_mean now only affects T1 duration, not disruption percentage.
    """
    id: Optional[int] = None
    timestamp: datetime
    irgc_decay_mean: float
    convoy_start_mean: float
    pipeline_max: float
    pipeline_ramp_weeks: float
    spr_rate_mean: float
    spr_delay_weeks: float
    surplus_buffer: float
    path_weights: PathWeights
    trigger: str


class MCResult(BaseModel):
    """MC model output."""
    id: Optional[int] = None
    timestamp: datetime
    params_id: Optional[int] = None
    price_mean: float
    price_p10: float
    price_p50: float
    price_p90: float
    path_a_price: float
    path_b_price: float
    path_c_price: float
    key_dates: Optional[dict] = None


class PositionSignal(BaseModel):
    """Position action record."""
    id: Optional[int] = None
    timestamp: datetime
    trigger: str
    action: str
    executed: bool = False
