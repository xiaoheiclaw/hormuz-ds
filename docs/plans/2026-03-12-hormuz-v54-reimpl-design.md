# Hormuz Decision OS v5.4 — 重新实现设计文档

**日期**：2026-03-12
**状态**：Approved
**依赖文档**：docs/ref-v54-backend-prd.md (PRD), docs/ref-v54-analysis.md, docs/ref-v54-dashboard.html
**决策**：推倒重来，基于 v5.4 PRD 从头实现

---

## 1. 范围

PRD 计算引擎（M1-M5 + MC N=10000）+ 数据采集（Readwise + yfinance）+ LLM 观测提取 + CLI + HTML 看板（含参数设置说明）+ 仓位规则。一次交付，不分阶段。

### 不含

- 油价预测模型（总缺口是油价的输入，油价建模是下游）
- 自动交易执行（所有仓位由人确认）
- LaunchAgent 定时任务配置（后续单独配）

---

## 2. 架构：引擎核心独立 + 薄外围

```
hormuz/
  core/              ← 纯计算，零 IO，零副作用
    types.py             # Pydantic 数据契约（PRD §2 + §7）
    variables.py         # 六类变量容器，YAML 加载
    m1_ach.py            # M1: ACH 贝叶斯推断（PRD §3.2）
    m2_duration.py       # M2: T 分布估算器（PRD §3.3）
    m3_buffer.py         # M3: Buffer 爬坡函数（PRD §3.4）
    m4_gap.py            # M4: 缺口积分器（PRD §3.5）
    m5_game.py           # M5: 博弈层路径调节（PRD §3.6）
    mc.py                # 蒙特卡洛模拟（PRD §5）
  infra/             ← IO 层
    db.py                # SQLite 存储
    ingester.py          # Readwise + yfinance 数据采集
    analyzer.py          # LLM 观测提取
    llm.py               # LLM 后端（Protocol + factory）
  app/               ← 产品层
    pipeline.py          # 管道编排
    signals.py           # Grabo 绊线（穿透语义）
    positions.py         # 仓位规则引擎
    reporter.py          # HTML 报告生成
    cli.py               # Click CLI
configs/
  config.yaml            # 运行时配置
  constants.yaml         # C01-C05 不变常数
  parameters.yaml        # P01-P10 可调参数
templates/
  status.html.jinja      # 看板模板
data/
  hormuz.db              # SQLite
reports/
  YYYY-WNN.html          # 周报归档
tests/
  test_core/             # 纯函数测试，零 mock
  test_infra/            # IO 层测试
  test_app/              # 集成测试
```

### 分层原则

- **core/**：纯函数，输入数据结构 → 输出数据结构。可独立测试，不依赖 IO
- **infra/**：处理网络、数据库、LLM 调用
- **app/**：组装 core + infra，面向用户

---

## 3. 数据契约（core/types.py）

对应 PRD §2 六类变量 + §7 SystemOutput。

### 六类变量

| 类 | Pydantic Model | 来源 | 可变性 |
|---|---|---|---|
| 常数 C01-C05 | `Constants` | constants.yaml 硬编码 | 不变 |
| 参数 P01-P10 | `Parameters` | parameters.yaml + 人工覆写 | 重大情报修正时 |
| 状态变量 S01-S11 | `StateVector` | 引擎动态更新 | 每轮 |
| 观测 O01-O13 | `Observation` | ingester + analyzer | 每日 |
| 控制 D01-D05 | `Control` | 人工录入 | 事件驱动 |
| 校准参照 R01-R04 | `CalibrationRef` | constants.yaml | 不变 |

### 核心输出

```python
class ACHPosterior:
    h1: float           # 耗竭概率
    h2: float           # 保留概率
    h3: float | None    # 补给（挂起时 None）
    dominant: str       # "H1" | "H2" | "inconclusive"

class PathWeights:
    a: float = 0.30
    b: float = 0.50
    c: float = 0.20
    # normalize() + clip [0.05, 0.85]

class SystemOutput:
    timestamp: datetime
    ach_posterior: ACHPosterior
    t1_percentiles: dict       # {p10, p25, p50, p75, p90}
    t2_percentiles: dict
    t_total_percentiles: dict
    buffer_trajectory: list[tuple[int, float]]
    gross_gap_mbd: float
    net_gap_trajectories: dict[str, list[tuple[int, float]]]
    path_probabilities: PathWeights
    path_total_gaps: dict[str, float]
    expected_total_gap: float
    consistency_flags: list[str]
```

### 变量容器（core/variables.py）

从 YAML 加载 Constants + Parameters，提供只读访问。Parameters 支持人工覆写接口（记录到 DB）。StateVector 作为可变状态在管道中传递。

---

## 4. 计算引擎（core/ M1-M5）

严格对应 PRD §3.2-3.6。

### M1: ACH 贝叶斯推断（m1_ach.py）

- 输入：`list[Observation]`（O01-O06）, `Parameters`
- 输出：`ACHPosterior`
- 先验：H1=47.5%, H2=47.5%, H3=5%（挂起→平分给 H1/H2）
- 似然表：每条观测 → LR ∈ {0.2, 0.5, 1.0, 2.0, 5.0}
- T1a/T1b 解绑：O05(GPS) × O01(频率) 共现
  - GPS↑ + 频率↑ → LR(H2)=5.0（攻击性 H2）
  - GPS↑ + 频率↓ → LR(H2)=2.0（防御性 H2）
  - GPS↓ → LR(H1)=3.0
- H3 解挂条件：梅赫拉巴德恢复 或 新型号武器 → 三元推断重启
- 输出映射：P(H1)>0.7 → irgcDecayRate 高；P(H2)>0.7 → 低

### M2: T 分布估算（m2_duration.py）

- 输入：`ACHPosterior`, `Parameters`, `list[Control]`, 事件标志
- 输出：T₁/T₂/T 合成的经验分布数组

**T₁**：
- irgcDecayRate ∈ [0.02, 0.08]/天，由 ACH 后验参数化
- S04(t+1) = S04(t) × (1 - decay) - externalKill(t)
- 首达 attack_freq < 2次/天 = T₁ 结束
- 分布：对数正态，H1 主导→中位 2-3w，H2 主导→中位 5-7w

**T₂**：
- stock-flow：S06(t+1) = S06(t) - sweep_rate(t)
- 采样 mines_in_water ~ Uniform(20, 100)
- 声学雷修正系数
- 事件跳变：E3→+7天, C2→+21天, E2→+14天

**T 合成**：T = T₁ + Uniform(7,14) + T₂（卷积）

### M3: Buffer 爬坡函数（m3_buffer.py）

- 输入：day(int), `Parameters`, `list[Control]`, `list[Observation]`(O11-O13)
- 输出：buffer_mbd(float)

Buffer(t) = pipeline(t) + spr(t) + cape_reroute(t)

- pipeline(t)：ADCOP 先到（D3-5）→ 沙特管道爬坡（D5-14）→ 稳态
- spr(t)：D02 触发后 13 天不可压缩延迟 → 线性爬坡
- cape_reroute(t)：D14 首批到货 → D21 渐进增加
- 校验：O11(延布 AIS) 校准 pipeline, O13(DOE 周报) 校准 SPR
- 路径 C 脆弱性：O12(富查伊拉价差) 极端 → buffer 骤降至 ~2 mbd

### M4: 缺口积分器（m4_gap.py）

- 输入：T 分布, Buffer 轨迹, `Constants`, `StateVector`
- 输出：每路径 NetGap(t) 轨迹 + TotalGap 标量

GrossGap = C05(20.1) × S11(0.80) ≈ 16 mbd（T 期间恒定）
NetGap(t) = GrossGap - Buffer(t)
TotalGap = Σ segments [days × NetGap]

S11 阶跃条件：O10 连续↑3天 且 O07 < 1% → T 结束

### M5: 博弈层路径调节（m5_game.py）

- 输入：`PathWeights`(基准), `list[Control]`, 事件标志
- 输出：`PathWeights`(调整后)
- 基准：A=0.30, B=0.50, C=0.20
- 信号规则：
  - D03(调停) → A+=0.15, C-=0.10, B-=0.05
  - 承诺松动 → A+=0.10, B-=0.10
  - D05(扫雷令) → A+=0.05, B→A 部分迁移
  - E1(目标溢出) → C+=0.15, A-=0.10, B-=0.05
- 每次 delta 后归一化 sum=1.0, clip [0.05, 0.85]
- 硬约束：只改概率，不碰 T/Buffer

---

## 5. 蒙特卡洛模拟（core/mc.py）

对应 PRD §5。

```
N = 10000

每轮采样:
  mines_in_water   ~ Uniform(20, 100)
  pipeline_incr    ~ Uniform(1.9, 3.0)
  cape_days        ~ Uniform(10, 13)
  insurance_weeks  ~ Uniform(7, 14) 天
  irgcDecayRate    ~ ACH 后验参数化
  deployment_gap   ~ Uniform(7, 14) 天

每轮计算:
  1. M2 → T_i
  2. M3 → Buffer_i(t)
  3. M4 → TotalGap_i

路径分类:
  T < 35天 → A
  35 ≤ T ≤ 120天 → B
  T > 120天 → C

输出:
  T 经验分布 (histogram + percentiles)
  每路径 TotalGap 分布
  路径频率 (MC) vs 路径概率 (M5 调整后)
  最终权重使用 M5 调整后的贝叶斯权重

并行: concurrent.futures.ProcessPoolExecutor
```

---

## 6. 外围层

### 存储（infra/db.py）

SQLite 8 表：

| 表 | 用途 |
|---|---|
| observations | O01-O13 原始观测时间序列 |
| ach_evidence | ACH 证据 + 似然比 |
| state_snapshots | StateVector 每轮快照（JSON） |
| controls | D01-D05 决策输入 + 触发时间 |
| mc_runs | MC 输出摘要统计 |
| system_outputs | SystemOutput 完整输出 |
| position_signals | 仓位指令（trigger, action, executed） |
| parameters_override | 人工覆写参数历史（who, when, old→new） |

### 数据采集（infra/ingester.py）

- Readwise API：按 tag 过滤，分类映射到 O01-O13
- yfinance：Brent/OVX/VLCC → O07/O09/O10 代理

### LLM 观测提取（infra/analyzer.py + llm.py）

- Protocol-based 后端切换（Claude API / OpenClaw）
- 结构化提示 → 输出 observation_id, value, confidence, evidence_direction

### 管道编排（app/pipeline.py）

```
1. ingester.fetch()           # 数据采集
2. analyzer.extract()         # LLM 提取观测
3. signals.scan()             # 绊线穿透（在 ACH 前！）
4. engine_run():
   a. m1 → b. m2 → c. m3 → d. m4 → e. m5 → f. mc
5. positions.evaluate()       # 仓位规则
6. reporter.render()          # HTML 输出
7. db.snapshot()              # 持久化
```

### 绊线（app/signals.py）

- T1a/T1b, T2, T3（48h 回退）
- E1-E4（持久）
- C1-C2（确认）
- 穿透语义：在 ACH 前执行，即时影响仓位

### 仓位规则（app/positions.py）

- MC 驱动常规仓位：能源 15% / 波动率 3% / 衰退 2%
- 退出规则：T 结束（通航↑3天+AP<1%）/ $150 需求破坏 / $80 系统失效
- 绊线穿透即时调整
- executed 标志 = 人机边界

### HTML 看板（app/reporter.py + templates/status.html.jinja）

九个区域：

1. **状态总览** — 冲突天数 / 当前路径权重 / 布伦特 / 通航量
2. **核心公式** — T × NetGap = TotalGap 公式条
3. **物理层** — Q1(ACH 热力图) / Q2(水雷 stock) / Q3(Buffer 爬坡图)
4. **博弈层** — Schelling 三诊断 + 信号表 + delta
5. **路径卡片** — A/B/C 概率 + TotalGap + 分段明细
6. **MC 分布** — T 直方图 + TotalGap 直方图（base64 matplotlib）
7. **仓位** — 持仓结构 / 未执行指令 / 止损状态
8. **参数设置说明** — 六类变量当前值 + 分布类型 + 覆写历史 + MC 配置 + 路径边界
9. **校验层** — 内部预测 vs 外部观测一致性警告

### CLI（app/cli.py）

```
hormuz run          # 完整管道
hormuz status       # 终端摘要
hormuz record       # 交互录入
hormuz mc           # 单独重跑 MC
hormuz report       # 手动周报
hormuz override     # 人工覆写参数
hormuz init-db      # 初始化 DB
hormuz validate     # 校验回路
```

---

## 7. 校验回路

对应 PRD §4.3。

| 校验 | 内部预测 | 外部观测 | 不一致处理 |
|---|---|---|---|
| 通航量 vs S11 | S11=0.80 → <4 艘/天 | O10 实际 | O10 持续 > 预测 → S11 下调 |
| AP vs S06 | S06 高 → AP > 1% | O07 实际 | AP 回落但 S06 未清 → 标记精算滞后 |
| 运费 vs Buffer | Buffer 爬坡 → 运费回落 | O09 实际 | 运费维持极端 → Buffer 实际偏低 |

校验结果写入 SystemOutput.consistency_flags，在看板 ⑨ 区域展示。

---

## 8. 已知薄弱点

直接继承 PRD §6 的 W1-W8，作为系统文档一部分，不试图在代码中解决。

---

## 9. 技术栈

```toml
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "numpy>=1.26",       # MC 数值计算
    "scipy>=1.12",       # 对数正态分布 + 卷积
    "jinja2>=3.1",
    "matplotlib>=3.8",
    "yfinance>=0.2",
    "httpx>=0.27",
    "click>=8.1",
    "pyyaml>=6.0",
    "python-dotenv>=1.2",
]
```

新增 numpy + scipy（MC 和 T 分布计算必需）。

---

## 10. 测试策略

- **core/**：纯函数测试，零 mock。给定输入 → 验证输出。覆盖所有模块
- **infra/**：mock 外部 IO（Readwise API, yfinance, LLM）
- **app/**：集成测试，用 fixture 数据跑完整管道
- MC 测试：小 N（100）验证分布形状和路径分类逻辑
