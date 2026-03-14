# 数据架构文档 — Hormuz-DS v5.4

**日期**：2026-03-14
**状态**：基于实际运行经验补写（PRD 补充件）

---

## 1. 数据源清单

### 1.1 新闻/情报层（→ O01-O06, O08, O10, O11, O14 + Schelling 信号）

| 数据源 | 接口 | 认证 | 频率 | 产出 |
|--------|------|------|------|------|
| Readwise Reader | REST API v3 | Token | 每 4h | 原始文章（标题+全文 HTML） |
| → LLM 提取 | Claude API | API Key | 每 4h | 结构化观测 + Schelling 信号 |

**实际数据特征**（探针结论）：
- Readwise 每次拉取 ~500 篇，主要来源：Al Jazeera(45%), OilPrice(16%), Al-Monitor(13%), gCaptain/The Cradle/Splash247(各 ~7%)
- 经关键词过滤后 ~60-70 篇相关，去重后全部处理（不设硬限）
- 全文 HTML 转 text 后截断 3000 字/篇，5 篇/batch，单 batch 上下文 ~15k tokens
- LLM 需要 conflict_day + previous_obs 基线才能准确提取，否则缺失观测默认 0

### 1.2 市场数据层（→ O09, 校准变量）

| 数据源 | 接口 | 认证 | 频率 | 产出 |
|--------|------|------|------|------|
| yfinance: BZ=F | Python lib | 无 | 每 4h | Brent 价格（校准，非 O 系列） |
| yfinance: ^OVX | Python lib | 无 | 每 4h | 原油波动率（校准） |
| yfinance: BWET | Python lib | 无 | 每 4h | O09 VLCC 运费代理（ETF→WS 映射） |

**限制**：BWET 是 ETF 价格不是真实 WS 点数，映射为线性插值近似。

### 1.3 专业数据层（→ O12, O13）

| 数据源 | 接口 | 认证 | 频率 | 产出 |
|--------|------|------|------|------|
| Ship & Bunker | 网页抓取 | 无 | 每 4h | O12 Fujairah-Singapore VLSFO 价差 |
| EIA API v2 | REST | 免费 Key | 每周 | O13 SPR 释放率 |

### 1.4 低频观测层（→ O07, O08, O10, O11）

这 4 个观测没有免费实时 API。非危机时期变动低频，LLM 提取 + baseline 校准已够用。

| 观测 | 现状 | 免费补充方案 | 付费兜底 |
|------|------|-------------|---------|
| O07 保费率 | LLM 提取 | + hellenicshippingnews.com 抓取（经常引用保费数字） | Beazley 邮件列表（免费注册） |
| O08 P&I 状态 | LLM 提取（已够用） | + IGP&I/Gard RSS 低频检查（每天 1 次） | 不需要 |
| O10 海峡通行量 | LLM 提取 | + EIA chokepoint API 月度 baseline（已有 key） | Kpler $500/月 |
| O11 延布装载 | LLM 提取 | + EIA 沙特出口 API 月度 baseline | Kpler 或 TankerTrackers $99/月 |

**设计原则**：无新闻 = 无变化，保持上次值。变了一定有新闻（尤其 O08 排除覆盖是头条级事件）。

---

## 2. 数据流

```
Readwise API ──fetch──→ 原始文章 (500+)
                           │
                      关键词过滤 (STRONG/WEAK 二级)
                           │
                      ≤60-70 篇相关
                           │
                      去重 (DB articles 表)
                           │
                      新文章 (≤30 篇)
                           │
                      ┌────┴────┐
                      │ 存 articles 表 │
                      └────┬────┘
                           │
                      LLM 提取 (5篇/batch × 6 batch)
                      context: conflict_day + previous_obs
                           │
                      ┌────┴────┐
                      │ observations │──→ ACH → MC → SystemOutput
                      │ signals      │──→ M5 博弈调节
                      │ provenance   │──→ article_observations 表
                      └─────────────┘

yfinance ──→ Brent/OVX (校准变量，不进 O 系列)
         ──→ BWET → O09 (ETF→WS 映射)

Ship&Bunker ──scrape──→ O12 (Fujairah-SG spread)

EIA API ──→ O13 (SPR release rate)

手动 `hormuz record` ──→ O07/O08/O10/O11
```

### 2.1 错误处理与降级

| 步骤 | 失败场景 | 降级策略 |
|------|---------|---------|
| Readwise 拉取 | 429/网络错误 | 3 次指数退避重试，全失败则跳过，用 DB 历史观测 |
| LLM 提取 | 单 batch 解析失败 | 指数退避重试 3 次（1s/2s/4s），全失败则跳过该 batch |
| yfinance | ticker 无数据 | 该观测缺失，不影响其他 |
| Ship & Bunker | HTML 结构变化 | O12 缺失，LLM 提取作为 fallback |
| EIA API | 数据延迟 | O13 缺失，不影响引擎（SPR 通常为 0） |

---

## 3. 存储设计

### 3.1 数据库选型

**SQLite** — 单文件，零运维，对当前规模（<10k 行/月）绰绰有余。

### 3.2 Schema（10 张表）

```
── 原始层 ──
articles                    # 原始文章去重存储
  id TEXT PRIMARY KEY       # Readwise doc ID
  title, source, url, summary, published_date, fetched_at

── 提取层 ──
observations                # O01-O14 时间序列
  id, timestamp, value, source, noise_note, created_at
article_observations        # 溯源：文章→观测映射
  article_id, obs_id, confidence, batch_run, created_at
ach_evidence                # ACH 推断证据链
  obs_id, direction, lr_h1, lr_h2, timestamp

── 控制层 ──
controls                    # D01-D05 决策输入
  id, actor, triggered, trigger_time, effect

── 输出层 ──
system_outputs              # SystemOutput JSON 快照
  timestamp, data_json
mc_runs                     # MC 模拟结果
  timestamp, n_samples, seed, result_json
state_snapshots             # 状态快照
  timestamp, data_json

── 操作层 ──
position_signals            # 仓位信号（含 executed 标记）
  timestamp, signal_type, action, executed
parameters_override         # 参数覆写审计
  param, old_value, new_value, created_at
```

### 3.3 容量估算

| 表 | 增量 | 月累积 | 年累积 |
|----|------|--------|--------|
| articles | ~60/次（去重后递减）× 6次/天 | ~5,000（首月，后续递减） | ~20k |
| observations | ~14/次 × 6次/天 = ~84/天 | ~2,500 | ~30k |
| article_observations | ~200/次 × 6次/天 = ~1200/天 | ~36k | ~432k |
| system_outputs | 6/天 | 180 | ~2.2k |

**预计 DB 文件大小**：<50MB/年。SQLite 舒适区（<1GB）。

### 3.4 清理策略

- articles.summary: 截断 3000 字，不存完整全文
- 超过 6 个月的 article_observations: 可归档删除（观测值已在 observations 表独立存在）
- system_outputs: 保留全部（审计需要）

---

## 4. 数据质量

### 4.1 已知问题

| 问题 | 影响 | 缓解 |
|------|------|------|
| LLM 对缺失观测默认 0 | ACH 偏向 H1 | conflict_day + previous_obs 基线注入 |
| Readwise 无关文章（体育/娱乐） | 浪费 LLM token，可能引入噪声 | 二级关键词过滤（STRONG/WEAK） |
| BWET→WS 映射不精确 | O09 有系统偏差 | 线性插值，标注 source=yfinance:bwet_proxy |
| O07/O08/O10/O11 无免费数据源 | 依赖 LLM 从新闻推断，精度低 | 手动 `hormuz record` 覆写 |
| 关键词过滤可能漏掉相关文章 | 观测覆盖不全 | STRONG 集合持续扩充 |

### 4.2 未实现

- [ ] 文章 embedding + 语义去重（目前只按 Readwise ID 去重）
- [ ] LLM 提取结果的置信度校验（多 batch 交叉验证）
- [ ] O09/O12 数据源切换（当 BWET/Ship&Bunker 不可用时）
- [ ] 看板展示溯源链（"O01=0.7 来自哪几篇文章"）
