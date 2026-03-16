# Hormuz Decision OS v5.4

霍尔木兹海峡危机投资决策操作系统。

三个物理问题（主动威胁衰减 / 水雷演化 / 缓冲到位）→ 贝叶斯推断 → MC 模拟 → 路径概率 + 总缺口。

**[在线看板](https://xiaoheiclaw.github.io/hormuz-ds/)**

## 核心公式

```
总缺口 = ∫₀ᵀ [ 16 mbd − Buffer(t) ] dt
T = T₁(攻击阶段) + gap(7-14天) + T₂(扫雷)
```

## 架构

```
src/hormuz/
├── core/           # 纯计算，零 IO
│   ├── types.py    # 六类变量 + SystemOutput
│   ├── m1_ach.py   # ACH 贝叶斯（14 O 系列 → log-odds → 后验）
│   ├── m2_duration.py  # T₁ 对数正态混合 + T₂ 数据驱动扫雷
│   ├── m3_buffer.py    # 管道 + SPR + 绕行爬坡
│   ├── m4_gap.py       # 分段积分 TotalGap
│   ├── m5_game.py      # Schelling 博弈调整路径权重
│   └── mc.py           # MC N=10000 + 8% regime jump
├── infra/          # IO 层
│   ├── db.py       # SQLite 存储
│   ├── ingester.py # Readwise + yfinance + Ship&Bunker + EIA
│   ├── analyzer.py # LLM 提取 O 系列 + Schelling 信号
│   └── llm.py      # Claude API / OpenClaw 后端
└── app/            # 产品层
    ├── pipeline.py # 6 步编排
    ├── cli.py      # Click CLI
    ├── positions.py# 仓位规则 + 退出信号
    └── reporter.py # HTML 看板生成
```

## 快速开始

```bash
# 安装
uv sync

# 初始化 DB + 种子数据
hormuz init-db
python scripts/load_seed.py

# 运行
hormuz run          # 全链路：拉数据 → LLM 提取 → 引擎 → 看板
hormuz status       # 查看最新输出
hormuz validate     # 一致性检查
hormuz mc           # 独立 MC 模拟
hormuz record O01 0.7 --source manual  # 手动录入观测
```

## 自动运行

LaunchAgent `com.hormuz-ds.run`，每天 08/12/16/20 四次。

```bash
# 手动触发
./scripts/run.sh

# 查看日志
tail -f data/logs/pipeline.log | python3 -m json.tool
```

## 数据源

| 观测 | 来源 | 方式 |
|------|------|------|
| O01-O08, O10-O11, O14 | Readwise 新闻 | LLM 提取（交叉信源校准） |
| O02 攻击趋势 | DB 历史 O01 | 自动计算 |
| O09 VLCC 运费 | yfinance BWET ETF | WS 换算代理 |
| O12 富查伊拉价差 | Ship & Bunker | 网页爬取 |
| O13 SPR 释放 | EIA API v2 | 需免费 key |
| Brent / OVX | yfinance | 校验变量，不进 ACH |

## 配置

```bash
configs/
├── config.yaml       # 主配置（Readwise token, LLM, 冲突起始日, MC 参数）
├── constants.yaml    # 物理常数
└── parameters.yaml   # 可调参数（扫雷舰=4, SPR=2.0mbd, regime_jump=8%）
```

## 关键设计决策

- **物理 > 制度 > 博弈**（因果优先级硬编码）
- **ACH 无状态/幂等**：每次从 DB 快照重算，不跨 run 累积
- **T2 数据驱动**：水雷数量由 O03/O10/E3/C2 观测调整，不靠 ACH 猜
- **T 报加权均值**：路径加权期望（含尾部风险），不是中位数
- **TotalGap 用 MC 物理权重**：不用 M5 调整后的权重，避免错配
- **sweep_ships=4**：2025.9 Avenger 退役，LLM 可从新闻自动更新
- **M5 信号去重**：同一事件多篇文章只算一次
- **core/ 零 IO**：纯函数可独立测试
