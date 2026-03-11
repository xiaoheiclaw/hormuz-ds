# 霍尔木兹危机投资决策系统 — 设计文档

**日期**：2026-03-11
**状态**：Approved
**依赖文档**：Red_Team_Analysis_Report_v4.md, Hormuz Decision System v4.md (PRD)

---

## 1. 目标

构建结构化情景分析 + 仓位管理系统，用于霍尔木兹海峡危机期间的能源投资决策辅助。不是自动交易系统，所有仓位决策由人做出。

**核心输出**：三条路径（A 快速降级 / B 渐进消耗 / C 长期对峙）的概率权重 + 油价路径概率分布 → 仓位建议。

---

## 2. 关键决策

| 决策 | 结论 | 理由 |
|------|------|------|
| 项目关系 | 独立项目 `~/Projects/hormuz-ds/` | 六层架构和 MC 模型与 GeoPulse Bayesian DAG 方法论不同 |
| 架构 | 管道式（仿 GeoPulse 三阶段流水线） | 批处理性质，4h 一轮，不需要事件驱动复杂度 |
| 交互方式 | CLI + HTML 报告 | CLI 为主操作入口，status.html 实时覆盖 + 周报归档 |
| 数据采集 | Readwise + API + 人工输入 | Readwise 做策展层（CENTCOM/航运/新闻），yfinance 拉市场数据，人工填判断 |
| MC 模型 | 分阶段：先解析近似，后 full MC | 油价方程未定义，先验证参数体系 |
| 存储 | SQLite 单文件 | 时间序列查询需求重，比纯 JSON 强 |
| 更新频率 | 每 4 小时自动（LaunchAgent） | 跟 GeoPulse heartbeat 节奏一致 |
| LLM 后端 | OpenClaw agent / Claude API 可切换 | 自动运行用 OpenClaw，手动操作用 Claude Code |
| 输出 | 实时 status.html + 每周三归档周报 | status.html 每 4h 覆盖，周报额外归档快照 |

---

## 3. 项目结构

```
~/Projects/hormuz-ds/
├── configs/
│   ├── config.yaml          # Readwise token, API keys, proxy, llm_backend, 更新间隔
│   ├── constants.yaml        # 常数 C1-C5 + 校准参照
│   └── parameters.yaml       # 可调参数初始值 + MC 参数
├── src/hormuz/
│   ├── __init__.py
│   ├── models.py             # Pydantic 数据合约
│   ├── db.py                 # SQLite schema + CRUD
│   ├── ingester.py           # Readwise 拉取 + yfinance/OVX API
│   ├── analyzer.py           # 统一接口：analyze(articles) → list[Observation]
│   ├── llm/
│   │   ├── __init__.py       # LLMBackend protocol + get_backend(config)
│   │   ├── openclaw.py       # OpenClaw agent 调用
│   │   └── claude_api.py     # Claude API 直接调用
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── ach.py            # ACH 矩阵（Q1: 3假设×9证据, Q2: 2假设×7证据）
│   │   ├── physical.py       # 物理层状态方程 + stock-flow
│   │   ├── signals.py        # Grabo 绊线 T1-T3 + 事件触发器 E1-E4 + 确认信号 C1-C2
│   │   ├── schelling.py      # 博弈层 6 项信号表 + delta 输出
│   │   ├── mc.py             # MC 模型（Phase1 解析近似 / Phase2 full MC）
│   │   └── positions.py      # 仓位规则引擎
│   ├── reporter.py           # status.html + 周报
│   └── pipeline.py           # 编排器
├── templates/
│   └── status.html.jinja     # Jinja2 模板
├── data/                     # hormuz.db（运行时生成）
├── reports/                  # 周报归档 YYYY-WNN.html
├── tests/
├── pyproject.toml
└── .structure.yml
```

---

## 4. SQLite Schema

```sql
-- 原始观测
observations (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    source TEXT,              -- 'centcom' | 'readwise' | 'yfinance' | 'manual'
    category TEXT,            -- 'q1_attack' | 'q2_mine' | 'q3_buffer' | 'market' | 'schelling'
    key TEXT,                 -- 'attack_frequency' | 'brent_price' | ...
    value REAL,
    metadata JSON
)

-- ACH 证据累积
ach_evidence (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    question TEXT,            -- 'q1' | 'q2'
    evidence_id INTEGER,
    direction TEXT,           -- 'h1' | 'h2' | 'h3' | 'neutral'
    confidence TEXT,          -- 'high' | 'medium' | 'low'
    notes TEXT,
    source_observation_id INTEGER REFERENCES observations(id)
)

-- Regime 判断历史
regimes (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    question TEXT,
    regime TEXT,              -- 'wide' | 'lean_h1' | 'lean_h2' | 'confirmed_h3'
    trigger TEXT
)

-- 即时响应信号
signals (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    signal_id TEXT,           -- 'T1'-'T3', 'E1'-'E4', 'C1'-'C2'
    status TEXT,              -- 'inactive' | 'triggered' | 'confirmed' | 'reverted'
    revert_deadline TEXT,
    action_taken TEXT
)

-- MC 参数快照
mc_params (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    params JSON,
    path_weights JSON,        -- {a: 0.30, b: 0.50, c: 0.20}
    trigger TEXT
)

-- MC 输出
mc_results (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    params_id INTEGER REFERENCES mc_params(id),
    output JSON
)

-- 仓位指令
position_signals (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    trigger TEXT,
    action TEXT,
    executed BOOLEAN DEFAULT 0
)
```

---

## 5. 管道流程（每 4 小时）

```
pipeline.run()
│
├─ 1. signals.check_reverts()          # 绊线 48h 回退检查
├─ 2. ingester.fetch()                 # Readwise + yfinance/OVX → observations
├─ 3. analyzer.extract(articles)       # LLM 提取结构化观测 → observations
├─ 4. signals.scan(new_observations)   # 绊线穿透扫描 → position_signals（立即）
├─ 5. engine.update()
│   ├─ ach.update()                    # 新证据 → 收敛检查 → regime
│   ├─ physical.update(regime)         # regime → MC 参数调整
│   ├─ schelling.scan()                # 6 项信号 → delta（W4 前只记录）
│   ├─ mc.run(params, weights)         # 解析近似 / full MC
│   └─ positions.evaluate()            # MC → 常规仓位指令
├─ 6. reporter.update_status()         # 覆盖 status.html
│   └─ if 周三: reporter.archive()     # 额外归档周报
└─ 7. notify()                         # 绊线触发/regime变更 → 推送
```

**关键**：步骤 4（绊线）在步骤 5（ACH/MC）之前——穿透语义。

---

## 6. 引擎模块逻辑

### 6.1 ACH 矩阵 (ach.py)

- Q1: 3 假设（H1 耗竭 / H2 保留 / H3 补给）× 9 条证据
- Q2: 2 假设（H1 摧毁 / H2 隐藏）× 7 条证据
- 证据判别力和预测方向为常数（constants.yaml）
- LLM 判断新观测偏向哪个假设
- 收敛规则：≥3 高判别力同向 → 更新 regime；单条反向 → 回退宽分布；2 周无确认 → 权重下调

### 6.2 物理层 (physical.py)

- Q1: 指数衰减 `capability(t) = initial * exp(-t / decay_mean)`
- Q2: 双池 stock-flow（陆上库存 → 水中水雷 → 清除），扫雷速率受 Q1 门控
- Q3: 缓冲到位分段函数（管道 + SPR + 绕行 + surplus）
- Regime → MC 参数映射表（PRD §5.2/5.3 定义）

### 6.3 绊线系统 (signals.py)

- 3 类信号：Grabo 前兆（T1-T3, 48h 回退）、事件触发器（E1-E4, 不回退）、假设确认（C1-C2）
- detection_fn(observations) → bool
- 触发 → 立即写入 position_signals + signals 表

### 6.4 博弈层 (schelling.py)

- 6 项固定检查：A 类 (1-4) 单独触发，B 类 (5-6) 需组合
- LLM 辅助判断 6 个 yes/no
- 输出 delta：±10pp 上限，物理层 base=0 不可调正
- W4 前只记录基线

### 6.5 MC 模型 (mc.py)

- Phase 1（MVP）：三条参数化油价曲线加权混合 → 均值/P10/P50/P90
- Phase 2（后续）：参数采样 N=10000 → 完整分布，含元山港凸函数校准

### 6.6 仓位规则 (positions.py)

- 两类输入：MC 常规 + 绊线穿透
- 硬止损：布伦特 < $80 连续 3 天 / 总亏损 > 8%
- executed=False 等人工确认

---

## 7. HTML 报告

### status.html（每 4h 覆盖）

| 面板 | 内容 |
|------|------|
| 状态总览 | 当前周数、路径权重图、布伦特、通行量 |
| 物理层 | Q1 攻击频率趋势 + regime、Q2 stock-flow 图、Q3 缓冲进度 |
| 观测层 | ACH 热力图、绊线状态灯、H3 watchlist |
| 博弈层 | Schelling 6 项状态 + delta |
| 仓位 | 持仓结构、未执行指令、止损状态 |

Jinja2 + matplotlib 静态图（base64 嵌入），单文件无外部依赖。

### 周报（每周三归档 reports/YYYY-WNN.html）

status.html 全部内容 + ACH 证据回溯 + MC 参数变更审计。

---

## 8. CLI 入口

```
hormuz run          # 手动触发完整管道
hormuz status       # 终端输出状态摘要
hormuz record       # 交互式手动录入（ACH 判断、regime 决策）
hormuz mc           # 单独重跑 MC
hormuz report       # 手动生成周报
hormuz init-db      # 初始化数据库
```

---

## 9. 技术栈

- Python 3.12+
- SQLite（标准库 sqlite3）
- Pydantic（数据合约）
- Jinja2（HTML 模板）
- matplotlib（图表）
- yfinance（市场数据）
- httpx（Readwise API / OpenClaw API）
- click（CLI）

---

## 10. 待明确

- OpenClaw agent 的具体调用接口（channel API 格式）
- 通知机制（Telegram via OpenClaw？LaunchAgent 推送？）
- MC Phase 1 的油价方程具体形式（需单独设计）
- Readwise 订阅源列表（CENTCOM RSS 是否可用）
