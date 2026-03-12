# 系统 PRD：霍尔木兹危机决策操作系统 v5.4

**日期**：2026-03-12
**系统定位**：状态空间模型 + 贝叶斯推断引擎
**目标输出**：T 分布（中断持续时间）、净缺口轨迹 NetGap(t)、三条路径的总缺口与概率权重

---

## 1. 系统边界

### 1.1 输入

| 输入类 | 来源 | 频率 | 举例 |
|---|---|---|---|
| 军事观测 | CENTCOM公报、OSINT | 每日/每周 | 袭击频率、武器型号、发射源定位 |
| 海事商业数据 | 海事AI平台、AIS | 实时/每日 | GPS欺骗集群、通航量、装载量 |
| 保险/金融市场 | 劳合社、Baltic Exchange、ICE | 每日 | AP保费率、VLCC运费、P&I条款状态 |
| 政府/机构公告 | DOE、IEA、CENTCOM | 每周/事件驱动 | SPR释放量、护航命令、调停框架 |
| 事件信号 | 多源交叉 | 事件驱动 | 触雷(E3)、扫雷舰遇袭(E2)、目标溢出(E1) |

### 1.2 输出

| 输出 | 类型 | 含义 |
|---|---|---|
| **T 分布** | 概率分布（离散路径加权） | P(T \| path_k) × P(path_k) 的混合分布 |
| **NetGap(t) 轨迹** | 时间序列（每路径一条） | 每日净供应缺口 = 毛缺口 − Buffer(t) |
| **TotalGap_k** | 标量 × 3 | 每条路径的累积总缺口 ∫₀ᵀ NetGap(t) dt |
| **路径概率** | [P_A, P_B, P_C] | 三路径的当前概率权重 |
| **ACH 后验** | [P(H1), P(H2)] | Q1 主动威胁状态的贝叶斯后验 |

### 1.3 不含

- 油价预测模型（总缺口是油价的输入，但油价建模是下游系统）
- 仓位计算 / 风控规则（属于 portfolio layer，不在本系统内）
- 实时数据采集管道（本系统假设输入已到位）

---

## 2. 变量体系

系统将所有信息严格分为六类，每类有不同的更新规则和存储要求。

### 2.1 常数（Constants）

所有情景、所有时间不变。硬编码，不暴露修改接口。

| ID | 名称 | 值 | 用途 |
|---|---|---|---|
| C01 | strait_width_km | 9 | 可通航航道宽度（2×3km航道 + 3km缓冲） |
| C02 | sweep_area | [航道地理常数] | Q2 扫雷任务物理规模 |
| C03 | sweep_rate_per_ship_max | [技术极限值] | 单舰日清扫面积上限 |
| C04 | mine_type_mix | {contact: x%, magnetic: y%, acoustic: z%} | 决定扫雷难度系数 |
| C05 | exposed_supply_mbd | 20.1 | 海峡日均原油+成品油通行量（2025Q1 EIA） |

### 2.2 参数（Parameters）

战争时间尺度内锁定。仅在重大情报修正或制度变更时人工覆写。

| ID | 名称 | 初始值 | 分布 | 更新条件 |
|---|---|---|---|---|
| P01 | mine_initial_stock | 2000–6000 | — | 情报修正 |
| P02 | mines_in_water | [20, 100] | Uniform | 情报修正 |
| P03 | spr_pump_min_days | 13 | 固定 | 制度变更 |
| P04 | spr_rate_max_mbd | 4.4 | 固定 | — |
| P05 | pipeline_max_mbd | 4.0 | 固定（延布+ADCOP合计理论值） | Q3观测校验可下调 |
| P06 | adcop_spare_mbd | 0.7 | 固定（1.8 − 1.07） | — |
| P07 | saudi_pipeline_increment | [1.9, 3.0] | Uniform | Q3观测校验可下调 |
| P08 | cape_reroute_days | [10, 13] | Uniform | — |
| P09 | attack_threshold | 2 | 固定：次/天 | — |
| P10 | insurance_observation_weeks | [1, 2] | Uniform | — |

### 2.3 状态变量（State Variables）

随 stock-flow 动态演化，每日/每周更新。这是系统的核心动态层。

| ID | 名称 | 初始值 | inflow | outflow | 更新频率 |
|---|---|---|---|---|---|
| S01 | ascm_remaining | unknown | 0（封闭系统，H3挂起） | 发射消耗 + 美军打击 | 每日 |
| S02 | uav_remaining | unknown | 0 | 发射消耗 + 美军打击 | 每日 |
| S03 | fast_boat_remaining | ~数百，已损失~20 | 0 | 美军猎杀速率 | 每日 |
| S04 | irgc_composite_capability | f(S01,S02,S03,C2) | — | 弹药消耗+平台损失+C2断裂 | 每日（综合指标） |
| S05 | mines_on_land | P01 | 0 | 布设速率 + 美军摧毁 | 每周 |
| S06 | mines_in_water | P02 | 布雷速率（from S05） | 扫雷速率（from S08） | 每日 |
| S07 | minesweepers_available | 3 LCS + 3–6 allied | 增援部署 | 损失 | 每周 |
| S08 | sweep_rate | S07 × C03 | — | 受 C03 上限约束 | 每日 |
| S09 | buffer_mbd | 0 | 管道切换+SPR入市+绕行到货 | — | 每日 |
| S10 | spr_released_mbd | 0 | DOE释放令 | 物理泵送约束 P03 | 每周 |
| S11 | disruption_rate | ~0.80 | — | T结束时阶跃→<0.10 | 每日 |

### 2.4 观测（Observables）

状态变量的带噪声测量值。系统通过观测反推状态变量。

| ID | 名称 | 数据源 | 频率 | 测量对象 | 噪声来源 |
|---|---|---|---|---|---|
| O01 | attack_frequency | CENTCOM公报 | 每日 | S04 | 报告延迟、未记录事件 |
| O02 | attack_frequency_2nd_derivative | O01 导出 | 每周 | S04 变化趋势 | 短期波动 |
| O03 | attack_coordination | 战报交叉分析 | 每周 | S04（C2完整度） | 判断主观性 |
| O04 | ammo_substitution_ratio | OSINT武器识别 | 每日 | S01 | 型号误判 |
| O05 | gps_spoofing_complexity | 海事AI平台 | 实时 | S04（C2完整度） | 民用平台精度 |
| O06 | mosaic_fragmentation | OSINT发射源定位 | 每周 | S04（地理分布） | 卫星14天延迟 |
| O07 | ap_premium_pct | 劳合社/经纪 | 每日 | S06（水雷威胁） + S11 | 精算滞后 |
| O08 | pni_status | P&I俱乐部 | 每日 | S11 | 制度惯性 |
| O09 | vlcc_td3_rate | Baltic Exchange | 每日 | S11 + S09 | 投机性波动 |
| O10 | strait_daily_transit | Vortexa/Kpler | 每日 | S11 | AIS欺骗/延迟 |
| O11 | yanbu_ais_loading | AIS追踪 | 每日 | S09（管道实际流量） | 装卸周期 |
| O12 | fujairah_sg_spread | 市场数据 | 每日 | S09（物流健康度） | 投机溢价 |
| O13 | spr_actual_release | DOE周报 | 每周 | S10 | 报告延迟 |

### 2.5 控制/决策输入（Controls）

各方 actor 的决策。事件驱动，触发状态变量或参数的离散跳变。

| ID | 名称 | actor | 触发效果 |
|---|---|---|---|
| D01 | convoy_order | 美军 | S07 ↑，T₂ 时钟可能提前启动 |
| D02 | spr_release_order | 白宫/IEA | S10 启动，受 P03 延迟 |
| D03 | mediation_framework | 第三方 | 路径概率调整（博弈层信号） |
| D04 | ceasefire_arrangement | 多方 | S11 → 阶跃下降 |
| D05 | sweep_priority_order | 美军 | 承诺锁定加强信号 |

### 2.6 校准参照（Calibration References）

固定历史事实，用于校准模型参数，不参与动态更新。

| ID | 事件 | 用途 |
|---|---|---|
| R01 | 1984 Tanker War | 中断率校准锚（~70%在24h内） |
| R02 | 2026-03 实际通行量 | 中断率校准锚（13艘 ≈ 8%通行 = 92%中断） |
| R03 | 1991 海湾战争扫雷 | T₂ 持续时间参照 |
| R04 | 1950 元山港 | 水雷拒止极端情景参照 |

---

## 3. 计算引擎

### 3.1 模块总览

```
输入（观测+事件）
    │
    ▼
┌─────────────┐
│ M1: ACH 引擎 │ ← O01–O06 每日/每周
│  贝叶斯推断   │
│  输出: P(H1), P(H2)
└──────┬──────┘
       │ H1/H2 后验
       ▼
┌─────────────┐
│ M2: T 分布    │ ← ACH后验 + 事件信号(E1–E3,T1a/b)
│  估算器       │
│  输出: T₁分布, T₂分布, T=T₁+T₂分布
└──────┬──────┘
       │ T 分布
       ▼
┌─────────────┐
│ M3: Buffer   │ ← O11–O13 + P03–P08 + 制度约束
│  爬坡函数     │
│  输出: Buffer(t) 轨迹
└──────┬──────┘
       │ Buffer(t)
       ▼
┌─────────────────────┐
│ M4: 缺口积分器        │ ← T分布 × NetGap(t) = 16 − Buffer(t)
│  输出: 每路径 TotalGap  │
│  + 路径概率加权期望      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ M5: 博弈层调节器      │ ← 信号表（D01–D05 + E1/E2）
│  输出: [P_A, P_B, P_C] │ → 回写 M4 路径权重
│  (delta 调整，非独立生成) │
└─────────────────────┘
```

### 3.2 M1: ACH 贝叶斯推断引擎

**目的**：根据持续输入的观测数据，维护 Q1 主动威胁的状态后验 P(H1|evidence) vs P(H2|evidence)。

**状态**：
- 先验：P(H1)=47.5%, P(H2)=47.5%, P(H3)=5%（挂起状态，H3的10%先验平分给H1/H2各+5%）
- H3 恢复条件：梅赫拉巴德机场恢复运营 或 观测到新型号武器入列。触发后 H3 解挂，先验重置为 H1=45%, H2=45%, H3=10%，三元贝叶斯推断启动。

**更新逻辑**：
```
对每条新观测 Oᵢ:
    1. 查询似然表: L(Oᵢ | H1), L(Oᵢ | H2)
    2. 贝叶斯更新:
       P(Hk | O₁..ₙ) ∝ P(Hk | O₁..ₙ₋₁) × L(Oₙ | Hk)
    3. 归一化
```

**似然表（核心配置）**：

每条观测对 H1/H2 的诊断力方向已在 md 中定义。似然比需要标定。

| 观测 | H1 方向特征 | H2 方向特征 | 判别力权重 |
|---|---|---|---|
| O02 袭击频率二阶导 | 加速下降（凸曲线） | 匀速或平台期 | 高 |
| O03 攻击协调性 | 退化为零散盲射 | 维持蜂群同步 | 高（难伪装） |
| O04 弹药替代率 | 高端绝迹→廉价消耗 | 降频但保持高端比例 | 高 |
| O05 GPS欺骗复杂度 | 退化为暴力干扰/消失 | 维持复杂几何欺骗 | 高（指C2存活） |
| O06 马赛克碎裂度 | 仅剩深山孤立据点 | 多节点交替开火 | 中/高 |

**似然比量化方案**：

每条证据按方向匹配度映射为似然比 LR ∈ {0.2, 0.5, 1.0, 2.0, 5.0}：
- LR=5.0：证据强烈支持该假设
- LR=2.0：证据中度支持
- LR=1.0：证据无判别力
- LR=0.5：证据中度反对
- LR=0.2：证据强烈反对

**T1a/T1b 解绑规则**：

O05（GPS欺骗）的解读依赖于 O01（袭击频率）的共现状态：
- O05 复杂度↑ **且** O01 频率↑ → T1a：攻击性 H2，LR(H2) = 5.0
- O05 复杂度↑ **但** O01 频率↓ → T1b：防御性 H2，LR(H2) = 2.0（C2存活但蛰伏）
- O05 复杂度↓ → 无论 O01 状态，LR(H1) = 3.0

**输出**：
- P(H1), P(H2) 当前后验
- ACH → irgcDecayRate 映射：P(H1) > 0.7 → decay 快；P(H2) > 0.7 → decay 慢

### 3.3 M2: T 分布估算器

**目的**：输出 T₁ 和 T₂ 的概率分布，进而得到 T = T₁ + T₂ 的混合分布。

#### T₁ 子模型（主动威胁消除耗时）

**驱动变量**：irgcDecayRate（由 M1 ACH 后验映射）

**逻辑**：
```
T₁ = 使 S04(t) 衰减至 attack_frequency < P09(2次/天) 的首达时间

S04(t+1) = S04(t) × (1 − irgcDecayRate) − externalKill(t)

其中:
  irgcDecayRate ∈ [0.02, 0.08]/天  ← 由 ACH 后验决定
    P(H1) 高 → 取高值（快速衰减）
    P(H2) 高 → 取低值（缓慢衰减）
  externalKill(t) = 美军日打击效果（观测驱动）
```

**分布输出**：T₁ ~ 对数正态，参数由 ACH 后验参数化
- P(H1)主导：T₁ 中位数 ~2–3周
- P(H2)主导：T₁ 中位数 ~5–7周
- 混合：T₁ = P(H1) × T₁|H1 + P(H2) × T₁|H2

**离散事件跳变**：
| 事件 | 效果 |
|---|---|
| T1a 触发 | T₁ 分布右移（延长），波动率放大 |
| E2（攻击扫雷舰） | T₁ 不直接影响，但 H2 后验跳升 → 间接延长 |

#### T₂ 子模型（扫雷清除耗时）

**前置条件**：T₁ 完成（O01 < P09 连续 N 天确认）后 T₂ 时钟启动。T₁→T₂ 之间存在 1–2 周部署缺口（固定延迟）。

**核心状态方程**：
```
S06(t+1) = S06(t) − S08(t)         # 水中水雷 stock-flow
S08(t) = min(S07(t) × C03, S06(t))  # 扫雷速率 = min(产能, 剩余量)

T₂ = 使 S06(t) → 安全通航阈值 的首达时间
```

**关键不确定性**：S06 初始值 ~ Uniform[20, 100]

蒙特卡洛采样：
```
for each sample:
    s06_init ~ Uniform(20, 100)
    s07 = current minesweeper count
    t2 = s06_init / (s07 × C03)  # 简化；实际需考虑补雷、类型混合
    # 声学雷修正：acoustic_fraction × acoustic_penalty_factor
    t2_adjusted = t2 × (1 + mine_type_penalty)
```

**离散事件跳变**：
| 事件 | 效果 |
|---|---|
| E3（触雷） | convoyStartMean += 7天 |
| C2（已扫区再雷） | convoyStartMean += 21天 |
| E2（攻击扫雷舰） | convoyStartMean += 14天；S07 −= 损失数 |

**分布输出**：T₂ ~ 蒙特卡洛采样分布，当前中位数 ~5周（35天）

#### T 合成

```
T = T₁ + deployment_gap + T₂

其中 deployment_gap ~ Uniform(7, 14) 天

T 的分布 = T₁分布 ⊕ gap分布 ⊕ T₂分布  （卷积）
```

### 3.4 M3: Buffer 爬坡函数

**目的**：计算 Buffer(t)——在毛缺口持续期间，每天有多少替代供应到位。

**结构**：Buffer(t) 是多个子缓冲的加总，每个子缓冲有各自的启动延迟和爬坡曲线。

```
Buffer(t) = pipeline(t) + spr(t) + cape_reroute(t)
```

#### 子缓冲定义

**pipeline(t)**：沙特东西管道 + ADCOP
```
if t < 3:  pipeline = 0
elif t < 5:  pipeline = ramp(0, P06, t-3, 2)       # ADCOP 先到位
elif t < 14: pipeline = P06 + ramp(0, P07_low, t-5, 9)  # 沙特管道爬坡
else: pipeline = P06 + P07_actual                    # 稳态

# P07_actual 由 O11（延布AIS装载量）校验：
#   若 O11 连续1周卡在 2.2 mbd → P07_actual = max(2.2 - P06, 0)
#   否则 P07_actual = P07 采样值
```

**spr(t)**：SPR 释放
```
if D02 未触发: spr = 0
else:
    t_since_order = t - D02.trigger_time
    if t_since_order < P03(13天): spr = 0   # 物理泵送不可压缩延迟
    else: spr = min(spr_ramp(t_since_order - P03), P04)

# spr_ramp: 线性爬坡至 sprRateMean
# sprRateMean 由 O13（DOE周报）校验：
#   O13 稳态 < 1.5 mbd → sprRateMean 下调
#   O13 稳态 > 3.5 mbd → sprRateMean 上调
```

**cape_reroute(t)**：好望角绕行
```
if t < 14: cape = 0
elif t < 21: cape = ramp(0, 0.5, t-14, 7)   # 首批到货
else: cape = ramp(0.5, cape_steady, t-21, 30) # 渐进增加
# cape_steady ~ 1–2 mbd，受全球运力约束
```

**Buffer(t) 分段近似**：
| 时段 | Buffer(t) 近似值 | 净缺口 |
|---|---|---|
| D1–D3 | ~0 mbd | ~16 mbd |
| D3–D14 | ~1.5 mbd（管道切换初期） | ~14.5 mbd |
| D14–D30 | 爬坡中，~3–5 mbd | ~11–13 mbd |
| D30+ | ~7 mbd（稳态上限） | ~9 mbd |

**路径C脆弱性**：
- 事件：富查伊拉遇袭 → pipeline(t) 骤降（物流枢纽瘫痪）→ Buffer(t) ≈ 2 mbd → NetGap ≈ 14 mbd
- 检测代理：O12（富查伊拉 vs 新加坡燃料油价差）极端拉开 → 触发 Buffer 下调

#### 制度约束（嵌入 Buffer 时间常数）

制度约束不是独立模块，而是 Buffer 各子缓冲的参数约束：

| 约束 | 影响的子缓冲 | 机制 |
|---|---|---|
| SPR 行政流程（P03=13天） | spr(t) | 启动延迟下界，不可压缩 |
| 保险精算观察期（P10=1–2周） | disruption_rate S11 的恢复时点 | AP 回调永远滞后于军事扫雷 |
| IEA 协调审批（1–2周） | spr(t) 的全球总规模 | 影响 sprRateMean 上界 |

### 3.5 M4: 缺口积分器

**目的**：对每条路径，计算净缺口轨迹和累积总缺口。

**核心公式**：
```
GrossGap = C05 × S11 = 20.1 × 0.80 ≈ 16 mbd  （T期间近似恒定）

NetGap(t) = GrossGap − Buffer(t)

TotalGap_k = ∫₀^{T_k} NetGap(t) dt
           ≈ Σ_segments [ days_in_segment × NetGap_segment ]
```

**中断率 S11 的行为**：
- T 期间：S11 ≈ 0.80（由 Q1+Q2 物理威胁锁定）
- T 结束时：S11 → 阶跃下降至 < 0.10（非渐变）
- 阶跃条件：O10（通航量）连续↑3天 **且** 此后 O07（AP）< 1%（滞后确认1–2周）

**分段积分**（替代均值×T，修正缓冲爬坡偏差）：
```
TotalGap = Σᵢ [ (t_{i+1} − tᵢ) × NetGap(tᵢ) ]

典型分段:
  segment_1: [D0, D3]   NetGap ≈ 16     → 3 × 16 = 48
  segment_2: [D3, D14]  NetGap ≈ 14.5   → 11 × 14.5 = 159.5
  segment_3: [D14, D30] NetGap ≈ 11     → 16 × 11 = 176
  segment_4: [D30, T]   NetGap ≈ 9      → (T-30) × 9
```

**三路径输出**：

| 路径 | T | 关键假设 | TotalGap |
|---|---|---|---|
| A | ~28天 | Q1快速衰减+Q2受控+聚点 | ~270 mbd·天 |
| B | ~84天 | Q1慢衰减~6w+Q2~5w | ~833 mbd·天 |
| C | >180天 | Q2持续拒止+Buffer崩溃 | >2500 mbd·天 |

### 3.6 M5: 博弈层路径调节器

**目的**：根据 Schelling 信号对 [P_A, P_B, P_C] 进行 delta 调整。不生成物理量，只改变路径概率。

**基准概率**：P_A=0.30, P_B=0.50, P_C=0.20

**信号触发规则**：

| 信号 | 检测条件 | delta 调整 | 归一化 |
|---|---|---|---|
| 聚点注入 | D03 触发（调停框架出现） | P_A += 0.15, P_C −= 0.10, P_B −= 0.05 | 是 |
| 承诺松动 | 美方信号包不一致（定性判断） | P_A += 0.10, P_B −= 0.10 | 是 |
| 承诺锁定 | D05 触发（扫雷优先令/SPR释放） | P_A += 0.05, P_B → 部分迁移至A | 是 |
| 失控实现 | E1 触发（目标溢出区域能源网） | P_C += 0.15, P_A −= 0.10, P_B −= 0.05 | 是 |

**归一化**：每次 delta 调整后，[P_A, P_B, P_C] 归一化至 sum = 1.0，并 clip 每个值至 [0.05, 0.85] 防止极端。

**层级边界硬约束**：
- 博弈层只改变路径概率，不直接修改 T₁/T₂ 分布或 Buffer(t)
- 物理时间常数设定 T 的硬下界，博弈层无法突破
- 路径C中 Buffer 崩溃（如富查伊拉遇袭）是物理事件，与博弈失控相关但因果独立

---

## 4. 数据流与更新周期

### 4.1 更新频率矩阵

| 模块 | 实时 | 每日 | 每周 | 事件驱动 |
|---|---|---|---|---|
| M1 ACH | O05 GPS | O01,O04 | O02,O03,O06 | H3 解挂判断 |
| M2 T分布 | — | irgcDecayRate 更新 | T₁/T₂ 重采样 | E2/E3/T1a 跳变 |
| M3 Buffer | — | O11 延布装载 | O13 SPR释放 | 富查伊拉遇袭 |
| M4 缺口积分 | — | NetGap(t) 更新 | TotalGap 重算 | S11 阶跃 |
| M5 博弈调节 | — | — | — | 信号触发 |

### 4.2 因果优先级

严格序：**物理 > 制度 > 博弈**

- 物理层（M1/M2/M3）的输出可以覆写博弈层（M5）的假设
- 博弈层只能调整概率权重，不能修改物理常数或状态方程
- 制度约束是物理层子缓冲的参数，不是独立决策层

### 4.3 校验回路

系统用 §3 结果变量（O07/O09/O10 + 布伦特结构）与内部状态做一致性校验：

| 校验 | 内部状态预测 | 外部观测 | 不一致处理 |
|---|---|---|---|
| 通航量 vs S11 | S11=0.80 → 通航应 < 4艘/天 | O10 实际 | 若 O10 持续 > 预测 → S11 下调 |
| AP vs S06 | S06 高 → AP 应 > 1% | O07 实际 | 若 AP 回落但 S06 未清 → 标记精算滞后 |
| VLCC运费 vs Buffer | Buffer 爬坡 → 运费应回落 | O09 实际 | 若运费维持极端 → Buffer 实际值可能偏低 |

---

## 5. 蒙特卡洛模拟规格

### 5.1 采样参数

| 参数 | 分布 | 采样 |
|---|---|---|
| P02 水中水雷 | Uniform(20, 100) | 每轮采样 |
| P07 沙特管道增量 | Uniform(1.9, 3.0) | 每轮采样 |
| P08 绕行天数 | Uniform(10, 13) | 每轮采样 |
| P10 保险观察期 | Uniform(7, 14) 天 | 每轮采样 |
| irgcDecayRate | 由 ACH 后验参数化 | 每轮采样 |
| deployment_gap | Uniform(7, 14) 天 | 每轮采样 |

### 5.2 模拟流程

```
N = 10000 轮

for i in 1..N:
    1. 采样所有不确定参数
    2. 运行 M2 → 得到 T₁_i, T₂_i, T_i
    3. 运行 M3 → 得到 Buffer_i(t) 轨迹
    4. 运行 M4 → 得到 NetGap_i(t) 轨迹, TotalGap_i
    5. 根据 T_i 落入区间分类路径:
       T < 35天 → 路径A
       35 ≤ T ≤ 120天 → 路径B
       T > 120天 → 路径C

输出:
    - T 的经验分布 (histogram + percentiles)
    - 每路径的 TotalGap 分布
    - 路径概率 (蒙特卡洛频率 vs M5 贝叶斯调整后的权重)
```

### 5.3 路径分类边界

| 路径 | T 区间 | 定性特征 |
|---|---|---|
| A | T < 35天 | 快速收敛，库存无断裂 |
| B | 35–120天 | 结构性中期缺口，库存严重消耗 |
| C | > 120天 | 极端持久，触发需求破坏 |

注：边界值可配置。蒙特卡洛给出的路径频率与 M5 博弈调节给出的概率会存在差异——前者是纯物理模拟的频率，后者融合了博弈判断。最终输出使用 M5 调整后的权重。

---

## 6. 系统已知薄弱点

诚实列出模型已知的盲区和简化假设，防止过度信任输出精度。

| # | 薄弱点 | 影响 | 缓解 |
|---|---|---|---|
| W1 | 袭击频率 < 2次/天 阈值无物理推导 | T₁ 结束判断可能偏早或偏晚 | 配合 O03（协调性）多信号确认 |
| W2 | S01–S03 初始存量完全未知 | T₁ 分布前半段高度不确定 | ACH 持续收窄，但早期误差大 |
| W3 | T₁→T₂ 部署缺口为固定估计 | 可能低估实际部署摩擦 | 采样 Uniform(7,14) 吸收部分不确定性 |
| W4 | S11 中断率在T期间假设恒定80% | 实际可能随封锁强度波动 | 用 O10（通航量）做运行时校验 |
| W5 | Buffer 爬坡函数为线性近似 | 实际爬坡可能有阶梯/瓶颈 | O11/O13 实测数据做参数校准 |
| W6 | 博弈层 delta 调整为人工标定 | 主观性高 | 保持 delta 幅度保守，clip 防极端 |
| W7 | H3（外部补给）挂起假设可能失效 | 若俄/朝补给路径存在但未观测到 → H2 被高估 | 持续监控 O04 是否出现未知型号武器 |
| W8 | 路径分类边界（35天/120天）为人设 | 真实情景连续谱而非离散路径 | 输出完整 T 分布而非仅路径标签 |

---

## 7. 输出规格

### 7.1 主输出对象

```python
@dataclass
class SystemOutput:
    timestamp: datetime
    
    # ACH 状态
    ach_posterior: dict  # {"H1": float, "H2": float, "H3": float|"suspended"}
    ach_dominant: str    # "H1" | "H2" | "inconclusive"
    
    # T 分布
    t1_distribution: Distribution  # 参数化分布或经验分布
    t2_distribution: Distribution
    t_total_distribution: Distribution
    t_percentiles: dict  # {p10, p25, p50, p75, p90}
    
    # Buffer 轨迹
    buffer_trajectory: TimeSeries  # Buffer(t) for t in [0, T_p90]
    buffer_current_mbd: float
    
    # 净缺口
    gross_gap_mbd: float  # ≈ 16（恒定）
    net_gap_trajectory: dict  # {path_A: TimeSeries, path_B: ..., path_C: ...}
    net_gap_current_mbd: float
    
    # 路径
    path_probabilities: dict  # {"A": float, "B": float, "C": float}
    path_total_gaps: dict     # {"A": float, "B": float, "C": float}  mbd·天
    expected_total_gap: float  # 概率加权期望
    
    # 校验
    consistency_flags: list  # 内部预测 vs 外部观测的不一致警告
```

### 7.2 前端消费方式

前端（HTML 仪表盘）从 SystemOutput 读取渲染：
- 公式条数值：从 t_percentiles、gross_gap_mbd、net_gap_current_mbd 取
- T panel 内容：从 ach_posterior、t1/t2 分布取
- Gap panel 内容：从 buffer_trajectory、buffer_current_mbd 取
- 博弈层：从 path_probabilities 取
- 路径卡片：从 path_total_gaps、path_probabilities 取
- 校验层：从 consistency_flags 取

---

## 8. 实现优先级

| 阶段 | 模块 | 理由 |
|---|---|---|
| P0 | M4 缺口积分器 + M3 Buffer 静态版 | 可用固定参数立即输出三路径 TotalGap |
| P1 | M2 T 分布 + 蒙特卡洛 | T 的不确定性量化是核心价值 |
| P2 | M1 ACH 引擎 | 需要积累观测数据才有意义 |
| P3 | M5 博弈调节 + 校验回路 | 依赖前三个模块运行 |
