# Hormuz-DS Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a structured scenario analysis + position management system for Hormuz Strait crisis energy investment decisions.

**Architecture:** Pipeline-style (ingester → analyzer → engine → reporter), 4h auto-update cycle, SQLite storage, OpenClaw/Claude API dual LLM backend. Six-layer information flow: physical → observation → institutional → game theory → MC → positions.

**Tech Stack:** Python 3.12+, SQLite, Pydantic, Jinja2, matplotlib, yfinance, httpx, click

**Reference docs:**
- Design: `docs/plans/2026-03-11-hormuz-ds-design.md`
- PRD: `~/Downloads/Hormuz Decision System v4.md`
- Analysis: `~/Downloads/霍尔木兹危机：投资决策操作系统 v4.0.md`
- GeoPulse (pattern reference): `~/Projects/geopulse/`

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.structure.yml`
- Create: `src/hormuz/__init__.py`
- Create: `src/hormuz/engine/__init__.py`
- Create: `src/hormuz/llm/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `configs/config.yaml`
- Create: `configs/constants.yaml`
- Create: `configs/parameters.yaml`
- Create: `templates/` (dir)
- Create: `data/` (dir)
- Create: `reports/` (dir)
- Create: `CLAUDE.md`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "hormuz-ds"
version = "0.1.0"
description = "Hormuz Strait crisis investment decision support system"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "jinja2>=3.1",
    "matplotlib>=3.8",
    "yfinance>=0.2",
    "httpx>=0.27",
    "click>=8.1",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]

[project.scripts]
hormuz = "hormuz.cli:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.backends"

[tool.hatch.build.targets.wheel]
packages = ["src/hormuz"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

**Step 2: Create .structure.yml**

```yaml
version: "1.0"
rules:
  configs:
    allow: ["configs/"]
    deny: ["."]
  sources:
    allow: ["src/hormuz/"]
  tests:
    allow: ["tests/"]
    deny: ["src/"]
  templates:
    allow: ["templates/"]
  data:
    allow: ["data/"]
  reports:
    allow: ["reports/"]
  docs:
    allow: ["docs/"]
```

**Step 3: Create configs/config.yaml**

```yaml
# Hormuz-DS Configuration
readwise:
  token: "${READWISE_TOKEN}"
  tag: "hormuz"
  sources:
    - "CENTCOM"
    - "Reuters"
    - "Lloyd's List"
    - "TradeWinds"
    - "IMO"
    - "OilPrice.com"
    - "Iran International"
  proxy: "http://127.0.0.1:7890"
  timeout: 30

llm:
  backend: "openclaw"  # "openclaw" | "claude_api"
  openclaw:
    endpoint: ""  # TBD: OpenClaw channel API
  claude_api:
    model: "claude-sonnet-4-6"
    proxy: "http://127.0.0.1:7890"

conflict:
  start_date: "2026-03-01"  # W1 起点

update:
  interval_hours: 4

db:
  path: "data/hormuz.db"
```

**Step 4: Create configs/constants.yaml**

从 PRD §4 提取所有常数和校准参照：

```yaml
# 物理常数（不可变）
physical:
  q1:
    C1_coastline: "沿岸山地，海峡北岸全为伊朗领土"
    C2_channel_width_km: 9  # 两条3km航道 + 3km缓冲
  q2:
    C3_sweep_area: "航道地理常数"
    C4_mine_types: ["contact", "magnetic", "acoustic"]
    C5_sweep_rate_ceiling: "单舰技术物理极限"

# ACH 证据定义（判别力和预测方向为常数）
ach:
  q1:
    hypotheses: ["h1_depletion", "h2_preservation", "h3_resupply"]
    evidence:
      - id: 1
        name: "总袭击频率二阶导数"
        discriminating_power: "high"
        h1_prediction: "加速下降（凸曲线）"
        h2_prediction: "匀速或可控下降"
        h3_prediction: "不可解释的平台期"
        source: "centcom"
        frequency: "weekly"
      - id: 2
        name: "攻击协调性"
        discriminating_power: "high"
        h1_prediction: "从协调→零散"
        h2_prediction: "可随时恢复"
        h3_prediction: "协调性维持"
        source: "centcom"
        frequency: "weekly"
      - id: 3
        name: "补给线活动"
        discriminating_power: "high"
        h1_prediction: "应出现紧急补给尝试"
        h2_prediction: "不需要补给"
        h3_prediction: "应观测到外部补给迹象"
        source: "satellite_osint"
        frequency: "weekly"
      - id: 4
        name: "发射平台部署活跃度"
        discriminating_power: "high"
        h1_prediction: "平台遗弃/不再移动"
        h2_prediction: "平台定期转移"
        h3_prediction: "平台活跃"
        source: "satellite"
        frequency: "weekly"
      - id: 5
        name: "攻击精度/命中率"
        discriminating_power: "medium"
        h1_prediction: "精度下降"
        h2_prediction: "精度稳定"
        h3_prediction: "精度稳定或新弹药特征"
        source: "shipping_reports"
        frequency: "weekly"
      - id: 6
        name: "美军打击C2/后勤密度"
        discriminating_power: "medium"
        h1_prediction: "约束收紧→持续力受限"
        h2_prediction: "约束收紧但不影响隐蔽库存"
        h3_prediction: "约束收紧但补给可绕过"
        source: "centcom_satellite"
        frequency: "weekly"
      - id: 7
        name: "连续无攻击天数"
        discriminating_power: "medium"
        h1_prediction: "沉默窗口单调递增"
        h2_prediction: "短期沉默是纪律"
        h3_prediction: "沉默窗口不应系统性增长"
        source: "centcom"
        frequency: "daily"
      - id: 8
        name: "武器类型分布变化"
        discriminating_power: "low"
        h1_prediction: "单调降级"
        h2_prediction: "策略性选择"
        h3_prediction: "出现非伊制武器"
        source: "centcom"
        frequency: "weekly"
      - id: 9
        name: "高价值目标窗口反应"
        discriminating_power: "low"
        h1_prediction: "只有低成本手段"
        h2_prediction: "选择性克制"
        h3_prediction: "反应能力维持"
        source: "ais_attack"
        frequency: "event_driven"
  q2:
    hypotheses: ["h1_destroyed", "h2_hidden"]
    evidence:
      - id: 1
        name: "布雷出击频率衰减曲线"
        discriminating_power: "high"
        h1_prediction: "加速下降"
        h2_prediction: "匀速或波动"
        source: "centcom_ais"
        frequency: "weekly"
      - id: 2
        name: "CENTCOM反布雷战果累计"
        discriminating_power: "medium"
        h1_prediction: "占比上升"
        h2_prediction: "占比低+未暴露储备"
        source: "centcom"
        frequency: "daily"
      - id: 3
        name: "商业卫星水雷存储设施"
        discriminating_power: "medium"
        h1_prediction: "多数设施被摧毁"
        h2_prediction: "设施被毁但水雷已转移"
        source: "satellite"
        frequency: "weekly"
      - id: 4
        name: "保险分航线报价差异"
        discriminating_power: "medium"
        h1_prediction: "保费趋降"
        h2_prediction: "保费高位/分化加剧"
        source: "lloyds"
        frequency: "daily"
      - id: 5
        name: "AIS航线偏离模式"
        discriminating_power: "medium"
        h1_prediction: "偏离区域缩小"
        h2_prediction: "偏离区域扩大"
        source: "vortexa_kpler"
        frequency: "daily"
      - id: 6
        name: "漂雷/可疑漂浮物频率"
        discriminating_power: "low"
        h1_prediction: "频率下降"
        h2_prediction: "频率下降但转入隐蔽区域"
        source: "imo_navarea"
        frequency: "event_driven"
      - id: 7
        name: "布雷手段降级路径"
        discriminating_power: "low"
        h1_prediction: "专用艇→民用小艇→消失"
        h2_prediction: "民用化是主动选择"
        source: "centcom_ais"
        frequency: "weekly"

# 信号定义
signals:
  grabo_tripwires:
    - id: "T1"
      name: "导弹发射平台向预设阵地移动"
      revert_hours: 48
      action: "波动率头寸加倍"
    - id: "T2"
      name: "多区域沿海阵地同时活跃"
      revert_hours: 48
      action: "路径C权重大幅上调"
    - id: "T3"
      name: "多艘布雷船同时出港"
      revert_hours: 48
      action: "convoyStartMean上调2周"
  event_triggers:
    - id: "E1"
      name: "IRGC目标升级至能源基础设施"
      revert: false
      action: "波动率加倍+衰退对冲5%"
      cross_layer: "schelling_4"
    - id: "E2"
      name: "IRGC集中攻击扫雷舰"
      revert: false
      action: "convoyStartMean上调2周"
      cross_layer: "ach_h2_diagnostic"
    - id: "E3"
      name: "触雷事件"
      revert: false
      action: "convoyStartMean上调1周"
    - id: "E4"
      name: "新区域发现布雷迹象"
      revert_hours: 48
      action: "Q2时间线延长"
  confirmations:
    - id: "C1"
      name: "战场出现非伊制武器系统"
      action: "ACH→H3确认，持续力无上界"
    - id: "C2"
      name: "已清扫航道再次发现水雷"
      action: "convoyStartMean上调3周，H2终极确认"

# Schelling 信号表
schelling:
  class_a:  # 硬 observable，单独触发
    - id: 1
      name: "外部协调者下场"
      diagnosis: "focal_point"
      direction: {a: "+", c: "-"}
    - id: 2
      name: "美方信号包不一致"
      diagnosis: "commitment_loosening"
      direction: {a: "+"}
    - id: 3
      name: "高成本自我绑定动作"
      diagnosis: "commitment_loosening"
      direction: {a: "+"}
    - id: 4
      name: "IRGC目标升级至能源基础设施"
      diagnosis: "escalation"
      direction: {c: "++", a: "-"}
  class_b:  # 模式推断，需组合
    - id: 5
      name: "谈判/停火窗口被公开化"
      diagnosis: "focal_point"
      direction: {a: "+"}
      requires_combo: [1, 3]
    - id: 6
      name: "IRGC区域行为碎片化"
      diagnosis: "commitment_loosening"
      direction: {a: "+"}
      requires_combo: [2]

# 校准参照
calibration:
  - case: "1988 Praying Mantis"
    lesson: "单枚水雷即可重创薄壳油轮"
    calibrates: "severity"
  - case: "1991 海湾战争"
    lesson: "布雷方不需主动作战能力即可让水雷持续发挥"
    calibrates: "Q1→Q2独立存续"
  - case: "1950 元山港"
    lesson: "水雷密度与清除时间呈凸函数关系"
    calibrates: "MC非线性"
```

**Step 5: Create configs/parameters.yaml**

```yaml
# 可调参数（战争时间尺度内锁定，重大修正时才改）
physical:
  q1:
    irgc_decay_mean_weeks: 6.0    # 主动威胁半衰期
    disruption_range: [0.55, 0.90] # 通行中断比例范围
    attack_threshold_per_day: 2.0  # Q2启动的Q1门控阈值
  q2:
    mine_initial_stock: [2000, 6000]  # 水雷初始总库存范围
    convoy_start_mean_weeks: 5.0      # 从Q1降至阈值开始计
    sea_mines_current: 50             # 当前水中水雷估计
    sweep_ships_available: 6          # 3 LCS + 3-6盟军
  q3:
    pipeline_max_mbd: 4.0         # 延布港瓶颈
    pipeline_ramp_weeks: 2.5
    spr_rate_mean_mbd: 2.5
    spr_delay_weeks: 2.5          # 13天物理 + 政治协商
    surplus_buffer_mbd: 2.5       # 冲击前全球过剩

# 路径初始权重
paths:
  a_fast_deescalation: 0.30
  b_gradual_attrition: 0.50
  c_prolonged_standoff: 0.20

# 仓位规则参数
positions:
  energy_base_pct: 15
  volatility_base_pct: 3
  recession_hedge_pct: 2
  stop_loss:
    brent_below: 80
    brent_days: 3
    max_portfolio_loss_pct: 8

# Schelling delta 约束
schelling:
  max_delta_pp: 10  # 单次信号调整上限
```

**Step 6: Create CLAUDE.md**

（使用设计 §5 中已定义的内容）

**Step 7: Create directory stubs and __init__.py files**

```bash
mkdir -p src/hormuz/engine src/hormuz/llm tests templates data reports
touch src/hormuz/__init__.py src/hormuz/engine/__init__.py src/hormuz/llm/__init__.py tests/__init__.py
```

**Step 8: Create tests/conftest.py**

```python
"""Shared test fixtures for hormuz-ds."""

import sqlite3
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Provide a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def constants() -> dict:
    """Load constants from config."""
    path = Path(__file__).parent.parent / "configs" / "constants.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def parameters() -> dict:
    """Load parameters from config."""
    path = Path(__file__).parent.parent / "configs" / "parameters.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
```

**Step 9: Install project in dev mode**

Run: `cd ~/Projects/hormuz-ds && uv pip install -e ".[dev]"`

**Step 10: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with configs, structure, and test fixtures"
```

---

## Task 2: Data Models (models.py)

**Files:**
- Create: `src/hormuz/models.py`
- Test: `tests/test_models.py`

**Step 1: Write failing tests**

```python
"""Tests for Pydantic data models."""

from datetime import datetime

import pytest

from hormuz.models import (
    ACHEvidence,
    MCParams,
    MCResult,
    Observation,
    PathWeights,
    PositionSignal,
    Regime,
    Signal,
    SignalStatus,
)


class TestObservation:
    def test_create_market_observation(self):
        obs = Observation(
            timestamp=datetime(2026, 3, 11, 8, 0),
            source="yfinance",
            category="market",
            key="brent_price",
            value=95.5,
        )
        assert obs.source == "yfinance"
        assert obs.value == 95.5

    def test_create_attack_observation_with_metadata(self):
        obs = Observation(
            timestamp=datetime(2026, 3, 11, 8, 0),
            source="readwise",
            category="q1_attack",
            key="attack_event",
            value=1.0,
            metadata={"type": "missile", "target": "tanker", "region": "strait"},
        )
        assert obs.metadata["type"] == "missile"

    def test_observation_source_validation(self):
        with pytest.raises(ValueError):
            Observation(
                timestamp=datetime(2026, 3, 11),
                source="invalid_source",
                category="market",
                key="brent_price",
                value=95.5,
            )


class TestACHEvidence:
    def test_create_q1_evidence(self):
        ev = ACHEvidence(
            timestamp=datetime(2026, 3, 11),
            question="q1",
            evidence_id=1,
            direction="h1",
            confidence="high",
            notes="袭击频率二阶导数加速下降",
        )
        assert ev.question == "q1"
        assert ev.direction == "h1"

    def test_q1_allows_h3(self):
        ev = ACHEvidence(
            timestamp=datetime(2026, 3, 11),
            question="q1",
            evidence_id=3,
            direction="h3",
            confidence="high",
        )
        assert ev.direction == "h3"

    def test_q2_rejects_h3(self):
        with pytest.raises(ValueError):
            ACHEvidence(
                timestamp=datetime(2026, 3, 11),
                question="q2",
                evidence_id=1,
                direction="h3",
                confidence="high",
            )


class TestPathWeights:
    def test_weights_must_sum_to_one(self):
        pw = PathWeights(a=0.30, b=0.50, c=0.20)
        assert abs(pw.a + pw.b + pw.c - 1.0) < 0.01

    def test_invalid_weights_rejected(self):
        with pytest.raises(ValueError):
            PathWeights(a=0.50, b=0.50, c=0.50)

    def test_apply_delta(self):
        pw = PathWeights(a=0.30, b=0.50, c=0.20)
        new_pw = pw.apply_delta(a_delta=0.05, c_delta=-0.05)
        assert abs(new_pw.a - 0.35) < 0.01
        assert abs(new_pw.c - 0.15) < 0.01
        assert abs(new_pw.b - 0.50) < 0.01  # b = residual

    def test_delta_clamped_to_bounds(self):
        pw = PathWeights(a=0.05, b=0.90, c=0.05)
        new_pw = pw.apply_delta(a_delta=-0.10, c_delta=0.10)
        assert new_pw.a >= 0.0
        assert new_pw.c <= 1.0


class TestSignal:
    def test_grabo_tripwire_has_revert_deadline(self):
        sig = Signal(
            timestamp=datetime(2026, 3, 11),
            signal_id="T1",
            status=SignalStatus.TRIGGERED,
            revert_deadline=datetime(2026, 3, 13),
            action_taken="波动率头寸加倍",
        )
        assert sig.revert_deadline is not None

    def test_event_trigger_no_revert(self):
        sig = Signal(
            timestamp=datetime(2026, 3, 11),
            signal_id="E3",
            status=SignalStatus.TRIGGERED,
            action_taken="convoyStartMean上调1周",
        )
        assert sig.revert_deadline is None


class TestMCParams:
    def test_create_params_with_weights(self):
        params = MCParams(
            timestamp=datetime(2026, 3, 11),
            irgc_decay_mean=6.0,
            convoy_start_mean=5.0,
            disruption_range=(0.55, 0.90),
            pipeline_max=4.0,
            pipeline_ramp_weeks=2.5,
            spr_rate_mean=2.5,
            spr_delay_weeks=2.5,
            surplus_buffer=2.5,
            path_weights=PathWeights(a=0.30, b=0.50, c=0.20),
            trigger="initial",
        )
        assert params.irgc_decay_mean == 6.0
        assert params.path_weights.b == 0.50
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/hormuz-ds && python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hormuz.models'`

**Step 3: Implement models.py**

```python
"""Pydantic data models — contracts between pipeline stages."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, field_validator, model_validator

VALID_SOURCES = {"centcom", "readwise", "yfinance", "manual", "llm"}
VALID_CATEGORIES = {"q1_attack", "q2_mine", "q3_buffer", "market", "schelling"}
VALID_QUESTIONS = {"q1", "q2"}
Q1_DIRECTIONS = {"h1", "h2", "h3", "neutral"}
Q2_DIRECTIONS = {"h1", "h2", "neutral"}
VALID_CONFIDENCES = {"high", "medium", "low"}


class SignalStatus(StrEnum):
    INACTIVE = "inactive"
    TRIGGERED = "triggered"
    CONFIRMED = "confirmed"
    REVERTED = "reverted"


class RegimeType(StrEnum):
    WIDE = "wide"
    LEAN_H1 = "lean_h1"
    LEAN_H2 = "lean_h2"
    CONFIRMED_H3 = "confirmed_h3"


class Observation(BaseModel):
    id: int | None = None
    timestamp: datetime
    source: str
    category: str
    key: str
    value: float
    metadata: dict[str, Any] | None = None

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if v not in VALID_SOURCES:
            raise ValueError(f"source must be one of {VALID_SOURCES}")
        return v

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if v not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {VALID_CATEGORIES}")
        return v


class ACHEvidence(BaseModel):
    id: int | None = None
    timestamp: datetime
    question: str  # "q1" | "q2"
    evidence_id: int
    direction: str  # "h1" | "h2" | "h3" | "neutral"
    confidence: str  # "high" | "medium" | "low"
    notes: str | None = None
    source_observation_id: int | None = None

    @model_validator(mode="after")
    def validate_direction_for_question(self) -> ACHEvidence:
        valid = Q1_DIRECTIONS if self.question == "q1" else Q2_DIRECTIONS
        if self.direction not in valid:
            raise ValueError(
                f"question={self.question} only allows directions {valid}, "
                f"got '{self.direction}'"
            )
        return self


class PathWeights(BaseModel):
    a: float
    b: float
    c: float

    @model_validator(mode="after")
    def validate_sum(self) -> PathWeights:
        total = self.a + self.b + self.c
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"path weights must sum to ~1.0, got {total}")
        return self

    def apply_delta(self, a_delta: float = 0.0, c_delta: float = 0.0) -> PathWeights:
        new_a = max(0.0, min(1.0, self.a + a_delta))
        new_c = max(0.0, min(1.0, self.c + c_delta))
        new_b = max(0.0, 1.0 - new_a - new_c)
        return PathWeights(a=new_a, b=new_b, c=new_c)


class Regime(BaseModel):
    id: int | None = None
    timestamp: datetime
    question: str
    regime: RegimeType
    trigger: str


class Signal(BaseModel):
    id: int | None = None
    timestamp: datetime
    signal_id: str  # T1-T3, E1-E4, C1-C2
    status: SignalStatus
    revert_deadline: datetime | None = None
    action_taken: str


class MCParams(BaseModel):
    id: int | None = None
    timestamp: datetime
    irgc_decay_mean: float
    convoy_start_mean: float
    disruption_range: tuple[float, float]
    pipeline_max: float
    pipeline_ramp_weeks: float
    spr_rate_mean: float
    spr_delay_weeks: float
    surplus_buffer: float
    path_weights: PathWeights
    trigger: str


class MCResult(BaseModel):
    id: int | None = None
    timestamp: datetime
    params_id: int | None = None
    price_mean: float
    price_p10: float
    price_p50: float
    price_p90: float
    path_a_price: float
    path_b_price: float
    path_c_price: float
    key_dates: dict[str, str] | None = None  # q1_threshold, q2_sweep_done, transit_recovery


class PositionSignal(BaseModel):
    id: int | None = None
    timestamp: datetime
    trigger: str  # "mc_update" | "tripwire_T1" | "event_E3" | ...
    action: str   # "energy_add_22%" | "vol_double" | ...
    executed: bool = False
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/hormuz-ds && python -m pytest tests/test_models.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/hormuz/models.py tests/test_models.py
git commit -m "feat: Pydantic data models with validation and path weight delta logic"
```

---

## Task 3: Database Layer (db.py)

**Files:**
- Create: `src/hormuz/db.py`
- Test: `tests/test_db.py`

**Step 1: Write failing tests**

```python
"""Tests for SQLite database layer."""

from datetime import datetime

import pytest

from hormuz.db import HormuzDB
from hormuz.models import (
    ACHEvidence,
    MCParams,
    Observation,
    PathWeights,
    PositionSignal,
    Regime,
    RegimeType,
    Signal,
    SignalStatus,
)


class TestDBInit:
    def test_creates_tables(self, tmp_db):
        db = HormuzDB(tmp_db)
        tables = db.list_tables()
        assert "observations" in tables
        assert "ach_evidence" in tables
        assert "regimes" in tables
        assert "signals" in tables
        assert "mc_params" in tables
        assert "mc_results" in tables
        assert "position_signals" in tables


class TestObservations:
    def test_insert_and_query(self, tmp_db):
        db = HormuzDB(tmp_db)
        obs = Observation(
            timestamp=datetime(2026, 3, 11, 8, 0),
            source="yfinance",
            category="market",
            key="brent_price",
            value=95.5,
        )
        obs_id = db.insert_observation(obs)
        assert obs_id > 0

        fetched = db.get_observations_since(datetime(2026, 3, 11))
        assert len(fetched) == 1
        assert fetched[0].value == 95.5

    def test_query_by_category(self, tmp_db):
        db = HormuzDB(tmp_db)
        for cat, val in [("market", 95.0), ("q1_attack", 3.0), ("market", 96.0)]:
            db.insert_observation(Observation(
                timestamp=datetime(2026, 3, 11),
                source="yfinance",
                category=cat,
                key="test",
                value=val,
            ))
        market = db.get_observations_since(datetime(2026, 3, 10), category="market")
        assert len(market) == 2


class TestACHEvidence:
    def test_insert_and_query_by_question(self, tmp_db):
        db = HormuzDB(tmp_db)
        ev = ACHEvidence(
            timestamp=datetime(2026, 3, 11),
            question="q1",
            evidence_id=1,
            direction="h1",
            confidence="high",
            notes="test",
        )
        db.insert_ach_evidence(ev)
        results = db.get_ach_evidence("q1")
        assert len(results) == 1
        assert results[0].direction == "h1"

    def test_get_recent_high_confidence(self, tmp_db):
        db = HormuzDB(tmp_db)
        for direction, conf in [("h1", "high"), ("h2", "low"), ("h1", "high"), ("h1", "high")]:
            db.insert_ach_evidence(ACHEvidence(
                timestamp=datetime(2026, 3, 11),
                question="q1",
                evidence_id=1,
                direction=direction,
                confidence=conf,
            ))
        high = db.get_ach_evidence("q1", confidence="high")
        assert len(high) == 3


class TestSignals:
    def test_insert_and_get_active(self, tmp_db):
        db = HormuzDB(tmp_db)
        sig = Signal(
            timestamp=datetime(2026, 3, 11),
            signal_id="T1",
            status=SignalStatus.TRIGGERED,
            revert_deadline=datetime(2026, 3, 13),
            action_taken="波动率加倍",
        )
        db.insert_signal(sig)
        active = db.get_active_signals()
        assert len(active) == 1
        assert active[0].signal_id == "T1"

    def test_update_signal_status(self, tmp_db):
        db = HormuzDB(tmp_db)
        sig_id = db.insert_signal(Signal(
            timestamp=datetime(2026, 3, 11),
            signal_id="T1",
            status=SignalStatus.TRIGGERED,
            revert_deadline=datetime(2026, 3, 13),
            action_taken="test",
        ))
        db.update_signal_status(sig_id, SignalStatus.REVERTED)
        active = db.get_active_signals()
        assert len(active) == 0


class TestRegimes:
    def test_get_latest_regime(self, tmp_db):
        db = HormuzDB(tmp_db)
        db.insert_regime(Regime(
            timestamp=datetime(2026, 3, 10),
            question="q1",
            regime=RegimeType.WIDE,
            trigger="initial",
        ))
        db.insert_regime(Regime(
            timestamp=datetime(2026, 3, 11),
            question="q1",
            regime=RegimeType.LEAN_H1,
            trigger="3 high-disc evidence converged",
        ))
        latest = db.get_latest_regime("q1")
        assert latest.regime == RegimeType.LEAN_H1


class TestMCParams:
    def test_insert_and_get_latest(self, tmp_db):
        db = HormuzDB(tmp_db)
        params = MCParams(
            timestamp=datetime(2026, 3, 11),
            irgc_decay_mean=6.0,
            convoy_start_mean=5.0,
            disruption_range=(0.55, 0.90),
            pipeline_max=4.0,
            pipeline_ramp_weeks=2.5,
            spr_rate_mean=2.5,
            spr_delay_weeks=2.5,
            surplus_buffer=2.5,
            path_weights=PathWeights(a=0.3, b=0.5, c=0.2),
            trigger="initial",
        )
        db.insert_mc_params(params)
        latest = db.get_latest_mc_params()
        assert latest.irgc_decay_mean == 6.0
        assert latest.path_weights.a == 0.3


class TestPositionSignals:
    def test_get_unexecuted(self, tmp_db):
        db = HormuzDB(tmp_db)
        db.insert_position_signal(PositionSignal(
            timestamp=datetime(2026, 3, 11),
            trigger="tripwire_T1",
            action="vol_double",
        ))
        db.insert_position_signal(PositionSignal(
            timestamp=datetime(2026, 3, 11),
            trigger="mc_update",
            action="energy_add",
            executed=True,
        ))
        pending = db.get_unexecuted_position_signals()
        assert len(pending) == 1
        assert pending[0].trigger == "tripwire_T1"
```

**Step 2: Run tests to verify fail**

Run: `cd ~/Projects/hormuz-ds && python -m pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement db.py**

实现 `HormuzDB` 类，包含：
- `__init__(path)`: 创建/打开数据库，建表
- `list_tables() → list[str]`
- 每张表的 insert / query 方法（见测试中使用的方法签名）
- JSON 序列化/反序列化用 `json.dumps`/`json.loads` 处理 metadata、params、path_weights 等字段
- 所有时间戳存储为 ISO8601 字符串

**Step 4: Run tests to verify pass**

Run: `cd ~/Projects/hormuz-ds && python -m pytest tests/test_db.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/hormuz/db.py tests/test_db.py
git commit -m "feat: SQLite database layer with full CRUD for all tables"
```

---

## Task 4: LLM Backend Abstraction

**Files:**
- Create: `src/hormuz/llm/__init__.py` (with Protocol)
- Create: `src/hormuz/llm/claude_api.py`
- Create: `src/hormuz/llm/openclaw.py`
- Test: `tests/test_llm.py`

**Step 1: Write failing tests**

```python
"""Tests for LLM backend abstraction."""

from unittest.mock import AsyncMock, patch

import pytest

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
        config = {"llm": {"backend": "invalid"}}
        with pytest.raises(ValueError):
            get_backend(config)
```

**Step 2: Run tests to verify fail**

Run: `cd ~/Projects/hormuz-ds && python -m pytest tests/test_llm.py -v`

**Step 3: Implement LLM backends**

`src/hormuz/llm/__init__.py`:
```python
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
```

`src/hormuz/llm/claude_api.py`:
```python
"""Claude API direct backend."""

from __future__ import annotations

import httpx


class ClaudeAPIBackend:
    def __init__(self, model: str, proxy: str | None = None):
        self.model = model
        self.proxy = proxy

    async def complete(self, prompt: str, system: str | None = None) -> str:
        # httpx 调用 Anthropic Messages API
        # 实际实现在集成测试阶段补全
        raise NotImplementedError("Claude API backend not yet implemented")
```

`src/hormuz/llm/openclaw.py`:
```python
"""OpenClaw agent backend."""

from __future__ import annotations

import httpx


class OpenClawBackend:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def complete(self, prompt: str, system: str | None = None) -> str:
        # httpx 调用 OpenClaw channel API
        # 实际实现在集成测试阶段补全
        raise NotImplementedError("OpenClaw backend not yet implemented")
```

**Step 4: Run tests to verify pass**

Run: `cd ~/Projects/hormuz-ds && python -m pytest tests/test_llm.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/hormuz/llm/ tests/test_llm.py
git commit -m "feat: LLM backend abstraction with OpenClaw and Claude API stubs"
```

---

## Task 5: Signals Engine (signals.py)

**Files:**
- Create: `src/hormuz/engine/signals.py`
- Test: `tests/test_signals.py`

**Step 1: Write failing tests**

```python
"""Tests for Grabo tripwire / event trigger / confirmation signal system."""

from datetime import datetime, timedelta

import pytest

from hormuz.db import HormuzDB
from hormuz.engine.signals import SignalEngine
from hormuz.models import Observation, Signal, SignalStatus


@pytest.fixture
def signal_engine(tmp_db) -> SignalEngine:
    db = HormuzDB(tmp_db)
    return SignalEngine(db)


class TestTripwireDetection:
    def test_e3_mine_strike_detected(self, signal_engine):
        """触雷事件（E3）→ 立即触发，不回退"""
        obs = [Observation(
            timestamp=datetime(2026, 3, 11),
            source="readwise",
            category="q2_mine",
            key="mine_strike",
            value=1.0,
            metadata={"vessel": "MT Pacific", "location": "strait"},
        )]
        triggered = signal_engine.scan(obs)
        assert any(s.signal_id == "E3" for s in triggered)

        # E3 不回退
        e3 = next(s for s in triggered if s.signal_id == "E3")
        assert e3.revert_deadline is None

    def test_e1_infrastructure_attack(self, signal_engine):
        """IRGC 目标升级至能源基础设施（E1）"""
        obs = [Observation(
            timestamp=datetime(2026, 3, 11),
            source="readwise",
            category="q1_attack",
            key="attack_event",
            value=1.0,
            metadata={"target": "pipeline", "type": "missile"},
        )]
        triggered = signal_engine.scan(obs)
        assert any(s.signal_id == "E1" for s in triggered)

    def test_no_false_positive_on_normal_attack(self, signal_engine):
        """普通油轮攻击不触发 E1"""
        obs = [Observation(
            timestamp=datetime(2026, 3, 11),
            source="readwise",
            category="q1_attack",
            key="attack_event",
            value=1.0,
            metadata={"target": "tanker", "type": "missile"},
        )]
        triggered = signal_engine.scan(obs)
        assert not any(s.signal_id == "E1" for s in triggered)


class TestRevert:
    def test_tripwire_reverts_after_48h(self, signal_engine):
        """T类信号48h无确认→回退"""
        # 模拟 T1 已触发49小时前
        db = signal_engine.db
        db.insert_signal(Signal(
            timestamp=datetime(2026, 3, 9, 8, 0),
            signal_id="T1",
            status=SignalStatus.TRIGGERED,
            revert_deadline=datetime(2026, 3, 11, 8, 0),
            action_taken="波动率加倍",
        ))

        now = datetime(2026, 3, 11, 9, 0)  # 超过 revert_deadline
        reverted = signal_engine.check_reverts(now)
        assert len(reverted) == 1
        assert reverted[0].signal_id == "T1"

        # 确认数据库已更新
        active = db.get_active_signals()
        assert len(active) == 0

    def test_event_trigger_never_reverts(self, signal_engine):
        """E类信号不回退"""
        db = signal_engine.db
        db.insert_signal(Signal(
            timestamp=datetime(2026, 3, 9),
            signal_id="E3",
            status=SignalStatus.TRIGGERED,
            revert_deadline=None,
            action_taken="convoyStartMean上调1周",
        ))

        reverted = signal_engine.check_reverts(datetime(2026, 3, 20))
        assert len(reverted) == 0


class TestPositionSignalGeneration:
    def test_e3_generates_position_signal(self, signal_engine):
        """触雷 → 自动生成仓位指令"""
        obs = [Observation(
            timestamp=datetime(2026, 3, 11),
            source="readwise",
            category="q2_mine",
            key="mine_strike",
            value=1.0,
        )]
        signal_engine.scan(obs)
        pending = signal_engine.db.get_unexecuted_position_signals()
        assert len(pending) >= 1
        assert any("convoy" in p.action.lower() or "E3" in p.trigger for p in pending)
```

**Step 2: Run tests to verify fail**

Run: `cd ~/Projects/hormuz-ds && python -m pytest tests/test_signals.py -v`

**Step 3: Implement signals.py**

核心结构：
- `SIGNAL_DEFINITIONS`: 从 constants.yaml 加载或硬编码信号定义 + detection_fn
- `SignalEngine.__init__(db)`: 持有数据库引用
- `SignalEngine.scan(observations) → list[Signal]`: 遍历定义，检测触发，写入 signals + position_signals
- `SignalEngine.check_reverts(now) → list[Signal]`: 检查超时回退

每个信号的 detection_fn 基于 observation 的 category + key + metadata 匹配：
- E1: `category="q1_attack"` + `metadata.target in {"pipeline", "refinery", "desalination"}`
- E2: `category="q1_attack"` + `metadata.target == "minesweeper"`
- E3: `category="q2_mine"` + `key="mine_strike"`
- E4: `category="q2_mine"` + `key="new_area_mining"`
- T1-T3: `category` + `key` 匹配对应的 LLM 提取标签

**Step 4: Run tests to verify pass**

Run: `cd ~/Projects/hormuz-ds && python -m pytest tests/test_signals.py -v`

**Step 5: Commit**

```bash
git add src/hormuz/engine/signals.py tests/test_signals.py
git commit -m "feat: signal engine with Grabo tripwires, event triggers, and auto-revert"
```

---

## Task 6: ACH Matrix Engine (ach.py)

**Files:**
- Create: `src/hormuz/engine/ach.py`
- Test: `tests/test_ach.py`

**Step 1: Write failing tests**

```python
"""Tests for ACH (Analysis of Competing Hypotheses) matrix engine."""

from datetime import datetime, timedelta

import pytest

from hormuz.db import HormuzDB
from hormuz.engine.ach import ACHEngine
from hormuz.models import ACHEvidence, Regime, RegimeType


@pytest.fixture
def ach_engine(tmp_db, constants) -> ACHEngine:
    db = HormuzDB(tmp_db)
    return ACHEngine(db, constants["ach"])


class TestConvergenceRules:
    def test_three_high_disc_same_direction_updates_regime(self, ach_engine):
        """≥3条高判别力同向 → 更新 regime"""
        for eid in [1, 2, 4]:  # evidence IDs with high discriminating power
            ach_engine.add_evidence(ACHEvidence(
                timestamp=datetime(2026, 3, 11),
                question="q1",
                evidence_id=eid,
                direction="h1",
                confidence="high",
            ))

        regime = ach_engine.evaluate_regime("q1")
        assert regime == RegimeType.LEAN_H1

    def test_only_medium_low_does_not_update(self, ach_engine):
        """仅中/低判别力 → 不更新（保持宽分布）"""
        for eid in [5, 6, 7, 8, 9]:  # medium + low
            ach_engine.add_evidence(ACHEvidence(
                timestamp=datetime(2026, 3, 11),
                question="q1",
                evidence_id=eid,
                direction="h1",
                confidence="high",
            ))

        regime = ach_engine.evaluate_regime("q1")
        assert regime == RegimeType.WIDE

    def test_single_contrary_high_disc_reverts(self, ach_engine):
        """单条反向高判别力 → 回退宽分布"""
        # 先建立偏 H1
        for eid in [1, 2, 4]:
            ach_engine.add_evidence(ACHEvidence(
                timestamp=datetime(2026, 3, 11),
                question="q1",
                evidence_id=eid,
                direction="h1",
                confidence="high",
            ))
        assert ach_engine.evaluate_regime("q1") == RegimeType.LEAN_H1

        # 单条反向高判别力
        ach_engine.add_evidence(ACHEvidence(
            timestamp=datetime(2026, 3, 12),
            question="q1",
            evidence_id=3,
            direction="h3",
            confidence="high",
        ))
        regime = ach_engine.evaluate_regime("q1")
        assert regime == RegimeType.WIDE

    def test_stale_evidence_decays(self, ach_engine):
        """2周无确认 → 权重下调"""
        old = datetime(2026, 2, 25)  # >2 weeks ago
        for eid in [1, 2, 4]:
            ach_engine.add_evidence(ACHEvidence(
                timestamp=old,
                question="q1",
                evidence_id=eid,
                direction="h1",
                confidence="high",
            ))

        regime = ach_engine.evaluate_regime("q1", as_of=datetime(2026, 3, 12))
        assert regime == RegimeType.WIDE  # stale evidence decayed


class TestQ2ACH:
    def test_q2_two_hypotheses_only(self, ach_engine):
        """Q2 只有 H1/H2 两个假设"""
        ach_engine.add_evidence(ACHEvidence(
            timestamp=datetime(2026, 3, 11),
            question="q2",
            evidence_id=1,
            direction="h1",
            confidence="high",
        ))
        # Q2 高判别力证据只有 #1
        # 需要再加中判别力来测试不触发
        regime = ach_engine.evaluate_regime("q2")
        # 只有1条高判别力，不够3条 → WIDE
        assert regime == RegimeType.WIDE
```

**Step 2: Run tests to verify fail**

Run: `cd ~/Projects/hormuz-ds && python -m pytest tests/test_ach.py -v`

**Step 3: Implement ach.py**

核心逻辑：
- `ACHEngine.__init__(db, ach_config)`: 加载证据定义（判别力、假设预测方向）
- `ACHEngine.add_evidence(evidence)`: 写入数据库
- `ACHEngine.evaluate_regime(question, as_of=None) → RegimeType`:
  1. 从数据库拉取该 question 的所有证据
  2. 过滤 stale（>2 周且无新确认 → 权重 ×0.5 或忽略）
  3. 按判别力分组
  4. 检查收敛：≥3 高判别力同向 → lean_hX
  5. 检查反向：任何 1 条高判别力反向 → 回退 wide
  6. 返回 regime

**Step 4: Run tests to verify pass**

Run: `cd ~/Projects/hormuz-ds && python -m pytest tests/test_ach.py -v`

**Step 5: Commit**

```bash
git add src/hormuz/engine/ach.py tests/test_ach.py
git commit -m "feat: ACH matrix engine with convergence rules and stale evidence decay"
```

---

## Task 7: Physical Layer (physical.py)

**Files:**
- Create: `src/hormuz/engine/physical.py`
- Test: `tests/test_physical.py`

**Step 1: Write failing tests**

```python
"""Tests for physical layer state equations."""

from datetime import datetime

import pytest

from hormuz.engine.physical import PhysicalLayer
from hormuz.models import MCParams, PathWeights, RegimeType


@pytest.fixture
def physical(parameters) -> PhysicalLayer:
    return PhysicalLayer(parameters["physical"])


class TestRegimeToParams:
    def test_wide_keeps_defaults(self, physical):
        params = physical.update_params(
            q1_regime=RegimeType.WIDE,
            q2_regime=RegimeType.WIDE,
        )
        assert params["irgc_decay_mean"] == 6.0
        assert params["convoy_start_mean"] == 5.0

    def test_lean_h1_speeds_decay(self, physical):
        params = physical.update_params(
            q1_regime=RegimeType.LEAN_H1,
            q2_regime=RegimeType.WIDE,
        )
        assert params["irgc_decay_mean"] < 6.0  # 下调

    def test_lean_h2_no_change(self, physical):
        params = physical.update_params(
            q1_regime=RegimeType.LEAN_H2,
            q2_regime=RegimeType.WIDE,
        )
        assert params["irgc_decay_mean"] == 6.0  # 不动

    def test_confirmed_h3_extends(self, physical):
        params = physical.update_params(
            q1_regime=RegimeType.CONFIRMED_H3,
            q2_regime=RegimeType.WIDE,
        )
        assert params["irgc_decay_mean"] > 12.0  # 大幅上调

    def test_q2_lean_h1_shortens_convoy(self, physical):
        params = physical.update_params(
            q1_regime=RegimeType.WIDE,
            q2_regime=RegimeType.LEAN_H1,
        )
        assert params["convoy_start_mean"] < 5.0

    def test_q2_lean_h2_extends_convoy(self, physical):
        params = physical.update_params(
            q1_regime=RegimeType.WIDE,
            q2_regime=RegimeType.LEAN_H2,
        )
        assert params["convoy_start_mean"] > 5.0


class TestQ1Decay:
    def test_exponential_decay(self, physical):
        """capability(t) = initial * exp(-t / decay_mean)"""
        cap_w0 = physical.q1_capability(week=0, decay_mean=6.0)
        cap_w6 = physical.q1_capability(week=6, decay_mean=6.0)
        assert abs(cap_w6 / cap_w0 - 0.368) < 0.01  # e^(-1) ≈ 0.368


class TestQ3Buffer:
    def test_buffer_ramp(self, physical):
        """缓冲按时序到位"""
        buf_d1 = physical.q3_buffer(day=1)
        buf_d14 = physical.q3_buffer(day=14)
        buf_d30 = physical.q3_buffer(day=30)
        assert buf_d1 < buf_d14 < buf_d30
        assert buf_d30 <= 10.0  # 稳态上限 ~7-10 mbd
```

**Step 2: Run tests to verify fail**

Run: `cd ~/Projects/hormuz-ds && python -m pytest tests/test_physical.py -v`

**Step 3: Implement physical.py**

核心：
- `PhysicalLayer.__init__(params_config)`: 加载参数
- `update_params(q1_regime, q2_regime) → dict`: PRD §5.2/5.3 的 regime→参数映射
- `q1_capability(week, decay_mean) → float`: 指数衰减
- `q2_stock_flow(...)`: 双池模型（MVP 可简化为净 flow 方向判断）
- `q3_buffer(day) → float`: 分段累积函数（管道 + SPR + 绕行）

**Step 4: Run tests to verify pass**

**Step 5: Commit**

```bash
git add src/hormuz/engine/physical.py tests/test_physical.py
git commit -m "feat: physical layer with Q1 decay, Q2 stock-flow, Q3 buffer ramp"
```

---

## Task 8: Schelling Signal Sheet (schelling.py)

**Files:**
- Create: `src/hormuz/engine/schelling.py`
- Test: `tests/test_schelling.py`

**Step 1: Write failing tests**

```python
"""Tests for Schelling Signal Sheet (game theory layer)."""

from datetime import datetime

import pytest

from hormuz.engine.schelling import SchellingSheet
from hormuz.models import PathWeights


@pytest.fixture
def schelling(constants) -> SchellingSheet:
    return SchellingSheet(constants["schelling"])


class TestDeltaOutput:
    def test_class_a_signal_triggers_alone(self, schelling):
        """A类信号单独触发即可调整"""
        signals_active = {1: True}  # 外部协调者下场
        delta = schelling.compute_delta(signals_active)
        assert delta["a"] > 0
        assert delta["c"] < 0

    def test_class_b_needs_combo(self, schelling):
        """B类信号需要组合才触发"""
        # #5 单独出现 → 不调整
        delta = schelling.compute_delta({5: True})
        assert delta["a"] == 0

        # #5 + #1 组合 → 调整
        delta = schelling.compute_delta({5: True, 1: True})
        assert delta["a"] > 0

    def test_delta_capped_at_10pp(self, schelling):
        """单次信号调整上限 ±10pp"""
        # 所有 A 类同时触发
        delta = schelling.compute_delta({1: True, 2: True, 3: True, 4: True})
        assert abs(delta["a"]) <= 0.10
        assert abs(delta["c"]) <= 0.10

    def test_no_signals_no_delta(self, schelling):
        delta = schelling.compute_delta({})
        assert delta["a"] == 0
        assert delta["c"] == 0


class TestWeekGating:
    def test_before_w4_only_records(self, schelling):
        """W4 前只记录基线，不输出 delta"""
        delta = schelling.compute_delta(
            {1: True},
            current_week=2,
        )
        assert delta["a"] == 0  # gated
```

**Step 2: Run tests to verify fail**

**Step 3: Implement schelling.py**

- `SchellingSheet.__init__(config)`: 加载信号定义
- `compute_delta(active_signals, current_week=None) → dict`:
  - W4 前 → `{a: 0, c: 0}`
  - A 类：单独触发
  - B 类：检查 requires_combo
  - 累加 delta，cap 到 ±10pp

**Step 4: Run tests, Step 5: Commit**

```bash
git add src/hormuz/engine/schelling.py tests/test_schelling.py
git commit -m "feat: Schelling Signal Sheet with class A/B logic and W4 gating"
```

---

## Task 9: MC Model Phase 1 (mc.py)

**Files:**
- Create: `src/hormuz/engine/mc.py`
- Test: `tests/test_mc.py`

**Step 1: Write failing tests**

```python
"""Tests for Monte Carlo model (Phase 1: analytical approximation)."""

from datetime import datetime

import pytest

from hormuz.engine.mc import MCModel
from hormuz.models import MCParams, MCResult, PathWeights


@pytest.fixture
def mc() -> MCModel:
    return MCModel()


@pytest.fixture
def base_params() -> MCParams:
    return MCParams(
        timestamp=datetime(2026, 3, 11),
        irgc_decay_mean=6.0,
        convoy_start_mean=5.0,
        disruption_range=(0.55, 0.90),
        pipeline_max=4.0,
        pipeline_ramp_weeks=2.5,
        spr_rate_mean=2.5,
        spr_delay_weeks=2.5,
        surplus_buffer=2.5,
        path_weights=PathWeights(a=0.3, b=0.5, c=0.2),
        trigger="test",
    )


class TestPhase1:
    def test_produces_result(self, mc, base_params):
        result = mc.run(base_params)
        assert isinstance(result, MCResult)
        assert result.price_p10 < result.price_p50 < result.price_p90

    def test_path_a_lower_than_c(self, mc, base_params):
        """快速降级的油价应低于长期对峙"""
        result = mc.run(base_params)
        assert result.path_a_price < result.path_c_price

    def test_weighted_mean_between_paths(self, mc, base_params):
        """加权均值在三条路径之间"""
        result = mc.run(base_params)
        assert result.path_a_price <= result.price_mean <= result.path_c_price

    def test_higher_c_weight_raises_mean(self, mc, base_params):
        """路径C权重上调 → 均价上升"""
        result_base = mc.run(base_params)

        high_c = MCParams(
            **{**base_params.model_dump(), "path_weights": PathWeights(a=0.1, b=0.4, c=0.5)}
        )
        result_high_c = mc.run(high_c)
        assert result_high_c.price_mean > result_base.price_mean

    def test_longer_convoy_raises_prices(self, mc, base_params):
        """convoyStartMean 延长 → 价格上升"""
        result_base = mc.run(base_params)

        long_convoy = MCParams(
            **{**base_params.model_dump(), "convoy_start_mean": 10.0}
        )
        result_long = mc.run(long_convoy)
        assert result_long.price_mean > result_base.price_mean
```

**Step 2: Run tests to verify fail**

**Step 3: Implement mc.py Phase 1**

Phase 1 解析近似：
- 每条路径定义一个参数化的布伦特价格曲线（冲击峰值 → 恢复路径）
- 路径 A：峰值后快速回落（irgcDecay 短 + convoy 短）
- 路径 B：峰值后缓慢回落（基准参数）
- 路径 C：峰值后高位平台（长期缺口 = 17mbd 通行量 - buffer 到位量）
- 加权混合产生分布统计量

油价方程的大致形式（待调参）：
```python
def path_price(week, base_price, disruption_pct, buffer_ramp, recovery_week):
    supply_gap = 17.0 * disruption_pct - buffer_ramp(week)
    price_premium = supply_gap * PRICE_SENSITIVITY  # ~$5-10 per mbd gap
    if week > recovery_week:
        price_premium *= exp(-(week - recovery_week) / RECOVERY_HALFLIFE)
    return base_price + price_premium
```

**Step 4: Run tests, Step 5: Commit**

```bash
git add src/hormuz/engine/mc.py tests/test_mc.py
git commit -m "feat: MC model Phase 1 analytical approximation with three path curves"
```

---

## Task 10: Position Rules Engine (positions.py)

**Files:**
- Create: `src/hormuz/engine/positions.py`
- Test: `tests/test_positions.py`

**Step 1: Write failing tests**

```python
"""Tests for position rules engine."""

from datetime import datetime

import pytest

from hormuz.db import HormuzDB
from hormuz.engine.positions import PositionEngine
from hormuz.models import MCResult, Observation, PathWeights, PositionSignal


@pytest.fixture
def pos_engine(tmp_db, parameters) -> PositionEngine:
    db = HormuzDB(tmp_db)
    return PositionEngine(db, parameters["positions"])


class TestHardStopLoss:
    def test_brent_below_80_three_days(self, pos_engine):
        """布伦特 < $80 连续3天 → 平掉全部能源超配"""
        observations = [
            Observation(timestamp=datetime(2026, 3, 9), source="yfinance", category="market", key="brent_price", value=79.0),
            Observation(timestamp=datetime(2026, 3, 10), source="yfinance", category="market", key="brent_price", value=78.0),
            Observation(timestamp=datetime(2026, 3, 11), source="yfinance", category="market", key="brent_price", value=77.0),
        ]
        signals = pos_engine.evaluate(observations=observations)
        assert any("平掉全部能源超配" in s.action for s in signals)

    def test_brent_above_80_no_stop(self, pos_engine):
        observations = [
            Observation(timestamp=datetime(2026, 3, 9), source="yfinance", category="market", key="brent_price", value=85.0),
            Observation(timestamp=datetime(2026, 3, 10), source="yfinance", category="market", key="brent_price", value=83.0),
            Observation(timestamp=datetime(2026, 3, 11), source="yfinance", category="market", key="brent_price", value=81.0),
        ]
        signals = pos_engine.evaluate(observations=observations)
        assert not any("平掉" in s.action for s in signals)


class TestMCDrivenRules:
    def test_high_attack_freq_adds_energy(self, pos_engine):
        """袭击频率连续7天 > 3次/天 + 通行量 < 3mbd → 能源加仓至22%"""
        obs = []
        for day in range(7):
            obs.append(Observation(
                timestamp=datetime(2026, 3, 5 + day),
                source="centcom",
                category="q1_attack",
                key="attack_frequency",
                value=4.0,
            ))
            obs.append(Observation(
                timestamp=datetime(2026, 3, 5 + day),
                source="readwise",
                category="market",
                key="transit_volume",
                value=2.5,
            ))
        signals = pos_engine.evaluate(observations=obs)
        assert any("22%" in s.action for s in signals)
```

**Step 2: Run tests, Step 3: Implement, Step 4: Run tests, Step 5: Commit**

```bash
git add src/hormuz/engine/positions.py tests/test_positions.py
git commit -m "feat: position rules engine with hard stop-loss and MC-driven rules"
```

---

## Task 11: Ingester (ingester.py)

**Files:**
- Create: `src/hormuz/ingester.py`
- Test: `tests/test_ingester.py`

**Step 1: Write failing tests**

测试要点：
- `ReadwiseIngester.fetch()` → 返回文章列表（mock httpx）
- `MarketIngester.fetch()` → 返回市场数据观测（mock yfinance）
- 过滤逻辑：source 白名单 + tag 过滤（仿 GeoPulse 模式）
- 写入 observations 表

**Step 2-5: Implement, test, commit**

```bash
git commit -m "feat: ingester with Readwise and yfinance data fetching"
```

---

## Task 12: Analyzer (analyzer.py)

**Files:**
- Create: `src/hormuz/analyzer.py`
- Test: `tests/test_analyzer.py`

**Step 1: Write failing tests**

测试要点：
- `Analyzer.extract(articles) → list[Observation]`
- LLM 提取：文章 → 结构化观测（袭击事件、触雷报告、Schelling 信号等）
- Mock LLM backend，测试提取逻辑和 Pydantic 解析
- 提取的 prompt template 需要精确匹配 PRD 的观测分类

**Step 2-5: Implement, test, commit**

```bash
git commit -m "feat: analyzer with LLM-based structured observation extraction"
```

---

## Task 13: Reporter (reporter.py)

**Files:**
- Create: `src/hormuz/reporter.py`
- Create: `templates/status.html.jinja`
- Test: `tests/test_reporter.py`

**Step 1: Write failing tests**

测试要点：
- `Reporter.update_status()` → 生成 status.html
- HTML 包含五个面板的内容
- matplotlib 图表生成为 base64
- `Reporter.archive_weekly()` → 生成 `reports/YYYY-WNN.html`

**Step 2-5: Implement, test, commit**

```bash
git commit -m "feat: HTML reporter with status dashboard and weekly archive"
```

---

## Task 14: Pipeline Orchestrator (pipeline.py)

**Files:**
- Create: `src/hormuz/pipeline.py`
- Test: `tests/test_pipeline.py`

**Step 1: Write failing tests**

测试要点（集成测试）：
- `Pipeline.run()` 按正确顺序执行7个步骤
- 绊线（步骤4）在 ACH/MC（步骤5）之前执行
- 周三自动触发周报归档
- 错误不中断管道（graceful degradation）

**Step 2-5: Implement, test, commit**

```bash
git commit -m "feat: pipeline orchestrator with 7-step execution flow"
```

---

## Task 15: CLI Entry Points (cli.py)

**Files:**
- Create: `src/hormuz/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write failing tests**

测试要点：
- `hormuz run` → 调用 pipeline.run()
- `hormuz status` → 输出终端摘要
- `hormuz record` → 交互式录入
- `hormuz mc` → 单独重跑 MC
- `hormuz init-db` → 创建数据库

**Step 2-5: Implement, test, commit**

```bash
git commit -m "feat: CLI entry points with click"
```

---

## Task 16: LaunchAgent Setup

**Files:**
- Create: `scripts/run.sh`
- Create: `scripts/com.hormuzds.update.plist`（不安装，只生成）

**Step 1: Create run.sh**

参考 GeoPulse 的 run 脚本，包含：
- 代理环境变量（`http_proxy=http://127.0.0.1:7890`）
- Python 路径设置
- `hormuz run` 调用
- 日志重定向

**Step 2: Create plist 模板**

每 4 小时触发，参考 chain-monitor 的 daily plist。

**Step 3: Commit**

```bash
git add scripts/
git commit -m "feat: LaunchAgent scripts for 4h auto-update cycle"
```

---

## Task 17: Integration Test + Initial Data Seed

**Files:**
- Create: `tests/test_integration.py`
- Create: `scripts/seed.py`

测试要点：
- 完整管道端到端（mock LLM + mock Readwise）
- seed.py 写入初始参数快照和 regime（wide/wide）到数据库

```bash
git commit -m "feat: integration test and initial data seed script"
```

---

## Task 18: Push to Remote

```bash
cd ~/Projects/hormuz-ds
git remote add origin git@github.com:xiaoheiclaw/hormuz-ds.git
git push -u origin main
```

---

## Execution Summary

| Task | 模块 | 优先级 | 预计测试数 |
|------|------|--------|-----------|
| 1 | 项目脚手架 | P0 | 0 |
| 2 | 数据模型 | P0 | ~12 |
| 3 | 数据库层 | P0 | ~10 |
| 4 | LLM 后端 | P0 | ~3 |
| 5 | 绊线系统 | P0 | ~6 |
| 6 | ACH 矩阵 | P0 | ~5 |
| 7 | 物理层 | P0 | ~6 |
| 8 | Schelling | P1 | ~5 |
| 9 | MC 模型 | P0 | ~5 |
| 10 | 仓位引擎 | P1 | ~3 |
| 11 | Ingester | P0 | ~4 |
| 12 | Analyzer | P0 | ~4 |
| 13 | Reporter | P1 | ~4 |
| 14 | Pipeline | P0 | ~4 |
| 15 | CLI | P1 | ~5 |
| 16 | LaunchAgent | P2 | 0 |
| 17 | 集成测试 | P1 | ~3 |
| 18 | Push | P2 | 0 |
| **Total** | | | **~79** |
