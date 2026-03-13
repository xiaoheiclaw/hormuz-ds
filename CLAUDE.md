# Hormuz Decision OS v5.4

## WHAT
霍尔木兹海峡危机投资决策操作系统。状态空间模型 + 贝叶斯推断引擎，
输出 T 分布、净缺口轨迹、三路径总缺口与概率权重。

## WHY
三个物理问题（主动威胁衰减/水雷演化/缓冲到位）决定油价路径，
需要系统化跟踪 14 类观测、贝叶斯更新假设、MC 模拟不确定性。

## HOW
- Python 3.12+, numpy, scipy, SQLite, Jinja2, matplotlib
- 三层架构: core/(纯计算) + infra/(IO) + app/(产品)
- core/ 映射 PRD M1-M5: ACH → T分布 → Buffer → 缺口积分 → 博弈调节
- MC N=10000 联合采样 6 个不确定参数，含 regime jumps (8%)
- 绊线穿透语义: signals.scan() 在 ACH 前执行

## 架构
- core/types.py: PRD §2 六类变量 + §7 SystemOutput
- core/m1_ach.py: log-odds 贝叶斯推断 (LR ∈ {0.77..1.3}, 95%cap)
- core/m2_duration.py: T₁对数正态混合 + T₂ stock-flow + regime jumps
- core/m3_buffer.py: pipeline(t) + spr(t) + cape(t) 三子缓冲
- core/m4_gap.py: 分段积分 ∫₀ᵀ [16 - Buffer(t)] dt
- core/m5_game.py: MC path_frequencies 基础权重 + Schelling delta
- core/mc.py: N=10000 蒙特卡洛，调用 estimate_t_total (含 regime jumps)

## 数据源
- A组(O01-O06,O14): LLM 从 Readwise 新闻批量提取（5篇/batch, 30篇上限）
- O09 VLCC运费: BWET ETF via yfinance（TD3代理）
- O12 富查伊拉价差: Ship & Bunker 网页抓取
- O13 SPR释放: EIA API v2（需免费key）
- O02 趋势: DB 历史 O01 计算（替代 LLM 猜测）
- O07/O08/O10/O11: LLM提取 + `hormuz record` 手动（无免费API）
- Brent/OVX: yfinance 校验变量（不进 O 系列）

## 关键约定
- 物理 > 制度 > 博弈（因果优先级硬编码）
- core/ 零 IO 零副作用，纯函数可独立测试
- PathWeights 由 MC path_frequencies 驱动，clip [0.05, 0.85]
- position_signals.executed 是人机边界
- H3（外部补给）挂起，O14 高置信触发自动解挂
- B组非 0-1 指标有专门 ACH 阈值：O07>1%, O09>WS150, O12>$50

# currentDate
Today's date is 2026-03-13.

      IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.
