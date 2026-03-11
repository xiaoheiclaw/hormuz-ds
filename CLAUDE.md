# Hormuz Decision System (hormuz-ds)

## WHAT
霍尔木兹海峡危机投资决策辅助系统。结构化情景分析 + 仓位管理，
不是自动交易系统，所有仓位决策由人做出。

## WHY
三个物理问题（主动威胁衰减/水雷演化/缓冲到位）决定油价路径，
需要系统化跟踪观测、更新概率、生成仓位建议。

## HOW
- Python 3.12+, SQLite, Jinja2, matplotlib
- 管道式架构：ingester → analyzer → engine → reporter
- 每 4h 自动运行（LaunchAgent），每周三人工深度分析
- LLM 后端：OpenClaw agent（自动）/ Claude API（手动），config.yaml 切换
- 数据源：Readwise（新闻/公报）+ yfinance（市场数据）+ 人工输入（判断类）

## 架构
- 六层信息流：物理层 → 观测层 → 制度层 → 博弈层 → MC模型 → 仓位规则
- Grabo 绊线穿透机制：T1-T3/E1-E4/C1-C2 跳过正常管道直达仓位
- ACH 收敛规则：≥3高判别力同向→更新regime，单条反向→回退宽分布
- MC 分阶段：Phase1 解析近似，Phase2 full MC

## 项目结构
- configs/：config.yaml + constants.yaml + parameters.yaml
- src/hormuz/engine/：ach.py, physical.py, signals.py, schelling.py, mc.py, positions.py
- data/hormuz.db：SQLite 单文件存储
- templates/：Jinja2 HTML 模板
- reports/：周报归档

## 关键约定
- 变量分类法：常数/参数/状态变量/观测/控制/校准参照，更新规则不同
- 物理层优先：低层信号和高层信号矛盾时信低层
- pipeline 执行顺序：绊线扫描 → ACH → 物理层 → 博弈层 → MC → 仓位
- position_signals.executed 是人机边界
