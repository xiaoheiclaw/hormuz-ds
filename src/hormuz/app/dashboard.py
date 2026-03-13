"""Framework reference dashboard — system architecture + parameters.

Generates a self-contained HTML page showing the Hormuz-DS framework:
pipeline flow, model parameters, signal definitions, position rules.
"""

from __future__ import annotations

from pathlib import Path

from hormuz.core.m5_game import _SIGNAL_DEFS, _COMBO_REQUIRES, BASE_SENSITIVITY, FOCAL_BONUS
from hormuz.core.m1_ach import _LR_TABLE


def _signal_rows() -> str:
    rows = []
    for key, sdef in _SIGNAL_DEFS.items():
        cred = sdef.credibility
        combo = _COMBO_REQUIRES.get(key)
        combo_str = f'需要: {", ".join(combo)}' if combo else "—"
        dir_class = {"A": "green", "B": "amber", "C": "red"}[sdef.direction]
        rows.append(
            f'<tr><td>{key}</td>'
            f'<td class="{dir_class}">{sdef.direction}</td>'
            f'<td>{sdef.cost:.1f}</td>'
            f'<td>{sdef.observability:.1f}</td>'
            f'<td class="mono">{cred:.2f}</td>'
            f'<td class="dim">{combo_str}</td></tr>'
        )
    return "\n".join(rows)


def _lr_rows() -> str:
    obs_names = {
        "O01": "攻击频率", "O02": "频率变化趋势", "O03": "攻击协调性",
        "O04": "高级武器使用", "O06": "马赛克防御覆盖",
        "O07": "战争险溢价", "O08": "P&I 72h取消",
        "O10": "海峡通行量", "O11": "管道替代流量",
    }
    rows = []
    for obs_id, dirs in sorted(_LR_TABLE.items()):
        name = obs_names.get(obs_id, obs_id)
        h = dirs["high"]
        l = dirs["low"]

        def _fmt(v: float) -> str:
            if v > 1.0:
                return f'<span class="green">{v:.2f}</span>'
            elif v < 1.0:
                return f'<span class="red">{v:.2f}</span>'
            return f'{v:.2f}'

        rows.append(
            f'<tr><td>{obs_id}</td><td>{name}</td>'
            f'<td>{_fmt(h["H1"])}</td><td>{_fmt(h["H2"])}</td>'
            f'<td>{_fmt(l["H1"])}</td><td>{_fmt(l["H2"])}</td></tr>'
        )
    return "\n".join(rows)


_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hormuz-DS 框架参考</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #07090f; font-family: 'Courier New', monospace; color: #e2e8f0; }
.wrap { max-width: 1000px; margin: 0 auto; padding: 24px 20px; }
h1 { font-size: 20px; text-align: center; margin-bottom: 4px; }
.subtitle { text-align: center; color: #475569; font-size: 11px; margin-bottom: 20px; }

/* Tabs */
.tabs { display: flex; gap: 0; border-bottom: 1px solid #1e293b; margin-bottom: 20px; }
.tab { padding: 8px 16px; font-size: 12px; color: #64748b; cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.2s; }
.tab:hover { color: #94a3b8; }
.tab.active { color: #f59e0b; border-bottom-color: #f59e0b; }
.panel { display: none; }
.panel.active { display: block; }

/* Components */
.section { margin-bottom: 24px; }
.sec-label { font-size: 11px; color: #475569; letter-spacing: 2px; margin-bottom: 8px; text-transform: uppercase; }
.card { background: #0a0e1a; border: 1px solid #1e293b; border-radius: 8px; padding: 14px 16px; margin-bottom: 8px; }
.card-title { font-size: 12px; color: #94a3b8; margin-bottom: 6px; }
table { width: 100%; border-collapse: collapse; font-size: 11px; }
th, td { padding: 5px 8px; border-bottom: 1px solid #1e293b; text-align: left; }
th { color: #475569; font-weight: 600; font-size: 10px; text-transform: uppercase; }
td { color: #cbd5e1; }
.amber { color: #f59e0b; }
.green { color: #10b981; }
.red { color: #ef4444; }
.blue { color: #3b82f6; }
.dim { color: #64748b; }
.mono { font-family: 'Courier New', monospace; }
.param-val { color: #f59e0b; font-weight: 600; }
.flow { display: flex; align-items: center; gap: 0; flex-wrap: wrap; justify-content: center; margin: 16px 0; }
.flow-box { background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 8px 12px; text-align: center; min-width: 100px; }
.flow-box .label { font-size: 10px; color: #475569; }
.flow-box .name { font-size: 13px; color: #e2e8f0; font-weight: 600; }
.flow-arrow { color: #334155; font-size: 18px; padding: 0 4px; }
.formula { background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 10px 14px; font-size: 13px; text-align: center; margin: 8px 0; }
.row { display: flex; gap: 8px; flex-wrap: wrap; }
.col { flex: 1; min-width: 280px; }
.sec-group { font-size: 10px; color: #64748b; padding: 4px 8px; background: #0f172a; font-weight: 600; }
.tag { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; margin-right: 4px; }
.tag-strong { background: #065f46; color: #6ee7b7; }
.tag-moderate { background: #713f12; color: #fcd34d; }
</style>
</head>
<body>
<div class="wrap">
<h1>Hormuz-DS 框架参考</h1>
<div class="subtitle">v5.4 · Schelling credibility-based game theory · 5-step pipeline</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('arch')">系统架构</div>
  <div class="tab" onclick="switchTab('m1')">M1 ACH</div>
  <div class="tab" onclick="switchTab('m2m4')">M2-M4 物理层</div>
  <div class="tab" onclick="switchTab('m5')">M5 博弈层</div>
  <div class="tab" onclick="switchTab('pos')">仓位规则</div>
</div>

<!-- ════ Tab 1: Architecture ════ -->
<div id="arch" class="panel active">

<div class="section">
<div class="sec-label">Pipeline 流程</div>
<div class="flow">
  <div class="flow-box"><div class="label">STEP 1</div><div class="name">数据获取</div></div>
  <div class="flow-arrow">→</div>
  <div class="flow-box"><div class="label">STEP 2</div><div class="name">LLM 提取</div></div>
  <div class="flow-arrow">→</div>
  <div class="flow-box"><div class="label">STEP 3</div><div class="name">引擎计算</div></div>
  <div class="flow-arrow">→</div>
  <div class="flow-box"><div class="label">STEP 4</div><div class="name">仓位评估</div></div>
  <div class="flow-arrow">→</div>
  <div class="flow-box"><div class="label">STEP 5</div><div class="name">DB 快照</div></div>
</div>
</div>

<div class="section">
<div class="sec-label">引擎内部 (Step 3)</div>
<div class="flow">
  <div class="flow-box" style="border-color:#10b981;">
    <div class="label">M1</div><div class="name green">ACH</div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-box" style="border-color:#3b82f6;">
    <div class="label">M2</div><div class="name blue">T 分布</div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-box" style="border-color:#3b82f6;">
    <div class="label">M3</div><div class="name blue">Buffer</div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-box" style="border-color:#3b82f6;">
    <div class="label">M4</div><div class="name blue">Gap</div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-box" style="border-color:#f59e0b;">
    <div class="label">MC</div><div class="name amber">模拟</div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-box" style="border-color:#ef4444;">
    <div class="label">M5</div><div class="name red">博弈调整</div>
  </div>
</div>
</div>

<div class="section">
<div class="sec-label">核心公式</div>
<div class="formula">
  <span class="amber">TotalGap</span> = ∫₀<sup>T</sup> [<span class="blue">GrossGap</span> − Buffer(t)] dt
</div>
<div class="formula">
  <span class="blue">GrossGap</span> = C01 × P10 = <span class="param-val">20.1 × 0.80 = 16.0 mbd</span>
</div>
<div class="formula">
  <span class="red">PathWeight</span> = MC频率 × (1 + Σ credibility × evidence × sensitivity × focal)
</div>
</div>

<div class="section">
<div class="sec-label">变量分类法</div>
<div class="card">
<table>
<tr><th>类型</th><th>符号</th><th>特征</th><th>更新频率</th></tr>
<tr><td class="dim">常数</td><td>C01-C05</td><td>物理定律/地理，永不变</td><td>—</td></tr>
<tr><td class="amber">参数</td><td>P01-P10</td><td>可调，校准或人工覆盖</td><td>周/月</td></tr>
<tr><td class="blue">状态</td><td>S01-S11</td><td>运行时可变</td><td>每次运行</td></tr>
<tr><td class="green">观测</td><td>O01-O14</td><td>外部数据，LLM/API 提取</td><td>每 4h</td></tr>
<tr><td class="red">控制</td><td>D01-D05</td><td>行为者决策</td><td>事件驱动</td></tr>
<tr><td>校准</td><td>R01-R03</td><td>历史参照</td><td>—</td></tr>
</table>
</div>
</div>
</div>

<!-- ════ Tab 2: M1 ACH ════ -->
<div id="m1" class="panel">

<div class="section">
<div class="sec-label">M1 · ACH 贝叶斯推断</div>
<div class="card">
<div class="card-title">假设</div>
<table>
<tr><td class="green">H1</td><td>能力衰竭</td><td>封闭系统，不可逆衰减 → 短危机</td></tr>
<tr><td class="red">H2</td><td>能力保存</td><td>战略配给，C2 完整 → 长危机</td></tr>
<tr><td class="dim">H3</td><td>外部补给</td><td class="dim">暂停 — 梅赫拉巴德机场已摧毁</td></tr>
</table>
</div>
</div>

<div class="section">
<div class="sec-label">似然比表 (LR)</div>
<div class="card">
<div class="card-title">
  LR 范围: <span class="tag tag-strong">strong 1.3/0.77</span>
  <span class="tag tag-moderate">moderate 1.05/0.95</span>
  · 单条 strong: 50%→63% (+13pp) · 3条同向: →~83% · 上限 95%
</div>
<table>
<tr><th>ID</th><th>观测</th><th colspan="2">High (value&gt;0.5)</th><th colspan="2">Low (value≤0.5)</th></tr>
<tr><th></th><th></th><th>H1</th><th>H2</th><th>H1</th><th>H2</th></tr>
%%LR_ROWS%%
<tr class="sec-group"><td colspan="6">O05 GPS欺骗 — 特殊处理: T1a/T1b 解绑 (依赖 O01 趋势)</td></tr>
</table>
</div>
</div>

<div class="section">
<div class="sec-label">关键参数</div>
<div class="card">
<table>
<tr><td>先验分布</td><td>H3暂停时: H1=H2=<span class="param-val">0.475</span></td></tr>
<tr><td>对数赔率上限</td><td><span class="param-val">log(95/5) ≈ 2.94</span> → 后验 cap 95%</td></tr>
<tr><td>O05 T1a</td><td>GPS高 + O01上升 → LR(H2)=<span class="param-val">1.30</span> (进攻性H2)</td></tr>
<tr><td>O05 T1b</td><td>GPS高 + O01下降 → LR(H2)=<span class="param-val">1.05</span> (防御性H2)</td></tr>
<tr><td>衰减率映射</td><td>decay = 0.02 + 0.06 × P(H1)，范围 <span class="param-val">[0.02, 0.08]</span></td></tr>
</table>
</div>
</div>
</div>

<!-- ════ Tab 3: M2-M4 Physical ════ -->
<div id="m2m4" class="panel">

<div class="section">
<div class="sec-label">M2 · T 分布</div>
<div class="card">
<div class="card-title">T1 (主动威胁持续时间) — 对数正态混合</div>
<table>
<tr><td>H1 分量</td><td>中位数 <span class="param-val">17</span> 天, σ=<span class="param-val">0.6</span></td><td class="dim">能力衰竭 → 快结束</td></tr>
<tr><td>H2 分量</td><td>中位数 <span class="param-val">42</span> 天, σ=<span class="param-val">0.55</span></td><td class="dim">能力保存 → 拉锯</td></tr>
<tr><td>混合权重</td><td>w_H1 = P(H1)/(P(H1)+P(H2))</td><td class="dim">ACH 后验</td></tr>
</table>
</div>
<div class="card">
<div class="card-title">T2 (扫雷持续时间) — stock-flow</div>
<table>
<tr><td>水中水雷</td><td>Uniform(<span class="param-val">20, 100</span>)</td><td class="dim">P02</td></tr>
<tr><td>扫雷舰</td><td><span class="param-val">6</span> 艘</td><td class="dim">P03 多国MCM</td></tr>
</table>
</div>
<div class="card">
<div class="card-title">事件跳跃</div>
<table>
<tr><td>E2 扫雷舰被击</td><td class="red">+14 天</td></tr>
<tr><td>E3 触雷</td><td class="red">+7 天</td></tr>
<tr><td>C2 再布雷</td><td class="red">+21 天</td></tr>
</table>
</div>
<div class="card">
<div class="card-title">路径分类</div>
<table>
<tr><td class="green">路径 A (快速)</td><td>T &lt; <span class="param-val">35</span> 天</td></tr>
<tr><td class="amber">路径 B (拉锯)</td><td><span class="param-val">35</span>–<span class="param-val">120</span> 天</td></tr>
<tr><td class="red">路径 C (长期)</td><td>T &gt; <span class="param-val">120</span> 天</td></tr>
</table>
</div>
</div>

<div class="section">
<div class="sec-label">M3 · Buffer 爬坡</div>
<div class="card">
<div class="formula">Buffer(t) = Pipeline(t) + SPR(t) + Cape(t)</div>
<table>
<tr><th>子系统</th><th>启动延迟</th><th>爬坡</th><th>峰值</th></tr>
<tr><td>ADCOP 管道</td><td>D3</td><td>2 天</td><td><span class="param-val">0.5</span> mbd</td></tr>
<tr><td>沙特东西管线</td><td>D5</td><td>~10 天</td><td><span class="param-val">3.5</span> mbd</td></tr>
<tr><td>SPR 释放</td><td>D<span class="param-val">13</span></td><td>~7 天</td><td><span class="param-val">2.5</span> mbd</td></tr>
<tr><td>好望角改道</td><td>D10</td><td>~30 天</td><td><span class="param-val">2.0</span> mbd</td></tr>
<tr class="sec-group"><td colspan="4">合计峰值 ≈ 7.0 mbd (D14+) → NetGap ≈ 9.0 mbd</td></tr>
</table>
</div>
</div>

<div class="section">
<div class="sec-label">M4 · Gap 积分</div>
<div class="card">
<table>
<tr><td>GrossGap</td><td>C01 × P10 = <span class="param-val">20.1 × 0.80 = 16.0</span> mbd</td><td class="dim">中断期间恒定</td></tr>
<tr><td>NetGap(t)</td><td>= max(0, GrossGap − Buffer(t))</td><td class="dim">逐日递减</td></tr>
<tr><td>TotalGap</td><td>= ∫₀ᵀ NetGap(t) dt</td><td class="dim">梯形积分, mbd·天</td></tr>
</table>
</div>
</div>
</div>

<!-- ════ Tab 4: M5 Game Theory ════ -->
<div id="m5" class="panel">

<div class="section">
<div class="sec-label">M5 · Schelling 博弈层</div>
<div class="card">
<div class="card-title">设计原理</div>
<table>
<tr><td>核心理论</td><td>Schelling focal point — 可信度来自成本结构，不是声明本身</td></tr>
<tr><td>可信度公式</td><td class="mono">credibility = cost × 0.6 + observability × 0.4</td></tr>
<tr><td>有效强度</td><td class="mono">strength = credibility × evidence × <span class="param-val">BASE_SENSITIVITY</span></td></tr>
<tr><td>焦点收敛</td><td class="mono">focal = 1 + <span class="param-val">FOCAL_BONUS</span> × (N − 1)，N=同方向信号数</td></tr>
</table>
</div>
</div>

<div class="section">
<div class="sec-label">全局调参旋钮</div>
<div class="card">
<table>
<tr><td>BASE_SENSITIVITY</td><td class="param-val">%%BASE_SENSITIVITY%%</td><td class="dim">单位可信度×证据的路径偏移量</td></tr>
<tr><td>FOCAL_BONUS</td><td class="param-val">%%FOCAL_BONUS%%</td><td class="dim">每多一条同方向信号的放大系数</td></tr>
</table>
</div>
</div>

<div class="section">
<div class="sec-label">信号定义</div>
<div class="card">
<table>
<tr><th>信号</th><th>方向</th><th>cost</th><th>observ.</th><th>credibility</th><th>前置</th></tr>
%%SIGNAL_ROWS%%
</table>
</div>
</div>

<div class="section">
<div class="sec-label">证据等级 (LLM 提取)</div>
<div class="card">
<table>
<tr><td class="green">high</td><td class="param-val">1.0</td><td>明确数据/引文</td></tr>
<tr><td class="amber">medium</td><td class="param-val">0.5</td><td>合理推断</td></tr>
<tr><td class="red">low</td><td class="param-val">0.2</td><td>模糊提及</td></tr>
</table>
</div>
</div>

<div class="section">
<div class="sec-label">情景速查</div>
<div class="card">
<div class="card-title">base A=30% B=50% C=20%</div>
<table>
<tr><th>情景</th><th>信号</th><th>ΔA</th><th>ΔC</th></tr>
<tr><td>阿曼秘密传话</td><td>mediation, low</td><td class="green">+0.3pp</td><td>−0.1pp</td></tr>
<tr><td>阿曼公开斡旋</td><td>mediation, high</td><td class="green">+1.5pp</td><td>−0.4pp</td></tr>
<tr><td>中国公开担保</td><td>self_binding, high</td><td class="green">+3.1pp</td><td>−0.9pp</td></tr>
<tr><td>IRGC 攻击油港</td><td>escalation, high</td><td>−0.9pp</td><td class="red">+2.4pp</td></tr>
<tr><td>IRGC 传言</td><td>escalation, low</td><td>−0.2pp</td><td class="red">+0.5pp</td></tr>
<tr><td>焦点收敛: 斡旋+承诺</td><td>mediation+binding, high</td><td class="green">+6.5pp</td><td>−2.3pp</td></tr>
<tr><td>矛盾: 斡旋+升级</td><td>mediation+escalation</td><td class="green">+0.5pp</td><td class="red">+1.6pp</td></tr>
<tr><td>日常噪音</td><td>mixed, low-med</td><td colspan="2" class="dim">总变化 &lt;2pp</td></tr>
</table>
</div>
</div>
</div>

<!-- ════ Tab 5: Positions ════ -->
<div id="pos" class="panel">

<div class="section">
<div class="sec-label">仓位规则</div>
<div class="card">
<div class="card-title">基础仓位</div>
<table>
<tr><td class="amber">能源多头</td><td class="param-val">15%</td><td class="dim">油价上行敞口</td></tr>
<tr><td class="blue">波动率</td><td class="param-val">3%</td><td class="dim">OVX/VIX 对冲</td></tr>
<tr><td class="green">衰退对冲</td><td class="param-val">2%</td><td class="dim">需求崩塌保护</td></tr>
</table>
</div>
</div>

<div class="section">
<div class="sec-label">退出规则 (覆盖基础仓位)</div>
<div class="card">
<table>
<tr><th>规则</th><th>触发条件</th><th>动作</th></tr>
<tr>
  <td class="red">系统失效</td>
  <td>Brent &lt; $<span class="param-val">80</span> × <span class="param-val">3</span> 天</td>
  <td>FORCE CLOSE ALL (能源+波动+衰退=0)</td>
</tr>
<tr>
  <td class="amber">需求毁灭</td>
  <td>Brent &gt; $<span class="param-val">150</span></td>
  <td>清空能源, 衰退×2 = <span class="param-val">4%</span></td>
</tr>
<tr>
  <td class="green">T-end 确认</td>
  <td>通行恢复 + 保费正常化</td>
  <td>能源→5%, 关闭波动率</td>
</tr>
</table>
</div>
</div>

<div class="section">
<div class="sec-label">人机边界</div>
<div class="card" style="border-left: 3px solid #f59e0b;">
  <div style="font-size: 12px;">
    <span class="amber">position_signals.executed</span> 是确认边界。<br>
    系统只生成建议，所有仓位决策由人做出。
  </div>
</div>
</div>
</div>

<div style="text-align:center;color:#334155;font-size:10px;margin-top:20px;">
  hormuz-ds v5.4 · Schelling credibility-based framework
</div>
</div>

<script>
function switchTab(id) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}
</script>
</body>
</html>"""


def generate_dashboard(output_path: Path | str = "docs/index.html") -> Path:
    """Generate framework reference dashboard HTML."""
    output_path = Path(output_path)

    html = _TEMPLATE
    html = html.replace("%%SIGNAL_ROWS%%", _signal_rows())
    html = html.replace("%%LR_ROWS%%", _lr_rows())
    html = html.replace("%%BASE_SENSITIVITY%%", str(BASE_SENSITIVITY))
    html = html.replace("%%FOCAL_BONUS%%", str(FOCAL_BONUS))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    path = generate_dashboard()
    print(f"Dashboard generated: {path}")
