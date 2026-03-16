"""HTML reporter — status dashboard + framework reference tabs.

Renders SystemOutput + MCResult into self-contained HTML.
Two tabs: 实时状态 (pipeline output) and 框架参考 (static architecture).
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
from jinja2 import Environment, BaseLoader

from hormuz.core.types import Parameters, SystemOutput
from hormuz.core.mc import MCResult
from hormuz.core.m5_game import _SIGNAL_DEFS, _COMBO_REQUIRES, BASE_SENSITIVITY, FOCAL_BONUS, SignalEvidence


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#0a0e1a", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_mc_chart(mc_result: MCResult) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax in (ax1, ax2):
        ax.set_facecolor("#0a0e1a")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
    ax1.hist(mc_result.t_samples, bins=40, color="#f59e0b", alpha=0.7, edgecolor="#0a0e1a")
    ax1.set_title("T 分布 (天)", color="#e2e8f0", fontsize=11)
    ax1.axvline(35, color="#ef4444", linestyle="--", alpha=0.5, label="A|B=35d")
    ax1.axvline(120, color="#ef4444", linestyle="--", alpha=0.5, label="B|C=120d")
    ax1.legend(fontsize=8, facecolor="#0a0e1a", edgecolor="#1e293b", labelcolor="#94a3b8")
    ax2.hist(mc_result.total_gap_samples, bins=40, color="#3b82f6", alpha=0.7, edgecolor="#0a0e1a")
    ax2.set_title("TotalGap 分布 (mbd·天)", color="#e2e8f0", fontsize=11)
    fig.patch.set_facecolor("#0a0e1a")
    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_buffer_chart(buffer_trajectory: list[tuple[int, float]]) -> str:
    days = [d for d, _ in buffer_trajectory]
    vals = [v for _, v in buffer_trajectory]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_facecolor("#0a0e1a")
    ax.plot(days, vals, color="#10b981", linewidth=2)
    ax.fill_between(days, vals, alpha=0.15, color="#10b981")
    ax.set_title("Buffer 爬坡 (mbd)", color="#e2e8f0", fontsize=11)
    ax.set_xlabel("天", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    fig.patch.set_facecolor("#0a0e1a")
    fig.tight_layout()
    return _fig_to_base64(fig)


# Signal display names
_SIGNAL_NAMES = {
    "external_mediation": "第三方斡旋",
    "us_inconsistency": "美国信号矛盾",
    "costly_self_binding": "高成本自我约束",
    "irgc_escalation": "IRGC 基础设施升级",
    "irgc_fragmentation": "IRGC 内部分裂",
}


def _build_signal_display(game_signals: list[SignalEvidence]) -> list[dict]:
    """Build signal display data for template."""
    rows = []
    for sig in game_signals:
        sdef = _SIGNAL_DEFS.get(sig.key)
        if sdef is None:
            continue
        dir_class = {"A": "green", "B": "amber", "C": "red"}[sdef.direction]
        evidence_label = "high" if sig.evidence >= 0.8 else "medium" if sig.evidence >= 0.4 else "low"
        strength = sdef.credibility * sig.evidence * BASE_SENSITIVITY
        rows.append({
            "name": _SIGNAL_NAMES.get(sig.key, sig.key),
            "direction": sdef.direction,
            "dir_class": dir_class,
            "credibility": f"{sdef.credibility:.2f}",
            "evidence": f"{sig.evidence:.1f}",
            "evidence_label": evidence_label,
            "strength": f"{strength:.3f}",
        })
    return rows


_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>霍尔木兹决策系统 v5.4</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #07090f; font-family: 'Courier New', monospace; color: #e2e8f0; padding: 24px 20px; }
.wrap { max-width: 960px; margin: 0 auto; }
h1 { font-size: 22px; text-align: center; margin-bottom: 8px; }
.subtitle { text-align: center; color: #475569; font-size: 12px; margin-bottom: 20px; }
.confidence-badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: bold; margin-left: 8px; }
.conf-normal { background: #065f46; color: #6ee7b7; }
.conf-low { background: #713f12; color: #fcd34d; }
.conf-burn_in { background: #7f1d1d; color: #fca5a5; }

/* Tabs */
.tabs { display: flex; gap: 0; border-bottom: 1px solid #1e293b; margin-bottom: 20px; }
.tab { padding: 8px 16px; font-size: 12px; color: #64748b; cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.2s; }
.tab:hover { color: #94a3b8; }
.tab.active { color: #f59e0b; border-bottom-color: #f59e0b; }
.panel { display: none; }
.panel.active { display: block; }

.section { margin-bottom: 24px; }
.sec-label { font-size: 12px; color: #475569; letter-spacing: 3px; margin-bottom: 10px; text-transform: uppercase; }
.card { background: #0a0e1a; border: 1px solid #1e293b; border-radius: 10px; padding: 16px 20px; margin-bottom: 10px; }
.card-title { font-size: 13px; color: #94a3b8; margin-bottom: 8px; }
.big-num { font-size: 28px; font-weight: 800; }
.amber { color: #f59e0b; }
.blue { color: #3b82f6; }
.green { color: #10b981; }
.red { color: #ef4444; }
.grey { color: #64748b; }
.dim { color: #64748b; }
.mono { font-family: 'Courier New', monospace; }
.param-val { color: #f59e0b; font-weight: 600; }
.row { display: flex; gap: 10px; flex-wrap: wrap; }
.col { flex: 1; min-width: 200px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { padding: 6px 10px; border-bottom: 1px solid #1e293b; text-align: left; }
th { color: #475569; font-weight: 600; }
td { color: #cbd5e1; }
.bar { height: 8px; border-radius: 4px; margin-top: 4px; }
.flag { background: #7f1d1d22; border-left: 3px solid #ef4444; padding: 8px 12px; margin: 4px 0; font-size: 12px; color: #fca5a5; border-radius: 0 6px 6px 0; }
img.chart { width: 100%; border-radius: 8px; margin: 8px 0; }
.pct-bar { display: inline-block; height: 6px; border-radius: 3px; }
.pos-action { font-size: 11px; color: #94a3b8; padding: 2px 0; }
.pos-action:before { content: "→ "; color: #475569; }
.sec-group { font-size: 11px; color: #64748b; padding: 4px 10px; background: #0f172a; font-weight: 600; }
.flow { display: flex; align-items: center; gap: 0; flex-wrap: wrap; justify-content: center; margin: 16px 0; }
.flow-box { background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 8px 12px; text-align: center; min-width: 90px; }
.flow-box .label { font-size: 10px; color: #475569; }
.flow-box .name { font-size: 13px; color: #e2e8f0; font-weight: 600; }
.flow-arrow { color: #334155; font-size: 18px; padding: 0 4px; }
.formula { background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 10px 14px; font-size: 13px; text-align: center; margin: 8px 0; }
.tag { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; margin-right: 4px; }
.tag-high { background: #065f46; color: #6ee7b7; }
.tag-medium { background: #713f12; color: #fcd34d; }
.tag-low { background: #7f1d1d; color: #fca5a5; }
</style>
</head>
<body>
<div class="wrap">
<h1>霍尔木兹决策系统 v5.4</h1>
<div class="subtitle">{{ timestamp }} · 冲突第 {{ conflict_day }} 天 · Brent ${{ brent_price }}
  <span class="confidence-badge conf-{{ confidence }}">{{ confidence_zh }}</span>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('status')">实时状态</div>
  <div class="tab" onclick="switchTab('framework')">框架参考</div>
</div>

<!-- ════════════════ TAB 1: 实时状态 ════════════════ -->
<div id="status" class="panel active">

<!-- 1. 状态总览 -->
<div class="section">
<div class="sec-label">1 · 状态总览</div>
<div class="row">
  <div class="col"><div class="card">
    <div class="card-title">ACH 主导假设</div>
    <div class="big-num {{ 'green' if dominant == 'H1' else 'red' if dominant == 'H2' else 'amber' }}">{{ dominant }}</div>
    <div style="margin-top:8px;font-size:12px;">H1={{ "%.0f"|format(h1*100) }}% H2={{ "%.0f"|format(h2*100) }}%{% if h3 is not none %} H3={{ "%.0f"|format(h3*100) }}%{% endif %}</div>
  </div></div>
  <div class="col"><div class="card">
    <div class="card-title">T 中位数</div>
    <div class="big-num amber">{{ t_p50 }} 天</div>
    <div style="margin-top:8px;font-size:12px;">p10={{ t_p10 }} p90={{ t_p90 }}</div>
  </div></div>
  <div class="col"><div class="card">
    <div class="card-title">路径权重</div>
    <div style="font-size:14px;">
      <span class="green">A {{ "%.0f"|format(pa*100) }}%</span> ·
      <span class="amber">B {{ "%.0f"|format(pb*100) }}%</span> ·
      <span class="red">C {{ "%.0f"|format(pc*100) }}%</span>
    </div>
    <div style="margin-top:8px;">
      <span class="pct-bar green" style="width:{{ (pa*200)|int }}px;background:#10b981;"></span>
      <span class="pct-bar amber" style="width:{{ (pb*200)|int }}px;background:#f59e0b;"></span>
      <span class="pct-bar red" style="width:{{ (pc*200)|int }}px;background:#ef4444;"></span>
    </div>
  </div></div>
</div>
</div>

<!-- 2. 核心公式 -->
<div class="section">
<div class="sec-label">2 · 核心公式</div>
<div class="card" style="text-align:center;">
  <div style="font-size:16px;">
    <span class="amber">TotalGap</span> = ∫₀<sup>T</sup> [<span class="blue">{{ gross_gap }}</span> − Buffer(t)] dt
    = <span class="amber big-num" style="font-size:24px;">{{ "%.0f"|format(expected_gap) }}</span> mbd·天
  </div>
</div>
</div>

<!-- 3. 物理层 -->
<div class="section">
<div class="sec-label">3 · 物理层</div>
<div class="row">
  <div class="col"><div class="card">
    <div class="card-title">Q1 主动威胁 (ACH)</div>
    <div style="margin:8px 0;">
      <div style="font-size:12px;">H1 衰竭 <span class="green">{{ "%.0f"|format(h1*100) }}%</span></div>
      <div class="bar green" style="width:{{ (h1*100)|int }}%;background:#10b981;"></div>
      <div style="font-size:12px;margin-top:6px;">H2 保存 <span class="red">{{ "%.0f"|format(h2*100) }}%</span></div>
      <div class="bar red" style="width:{{ (h2*100)|int }}%;background:#ef4444;"></div>
    </div>
  </div></div>
  <div class="col"><div class="card">
    <div class="card-title">Q3 缓冲爬坡</div>
    {% if buffer_chart %}
    <img class="chart" src="data:image/png;base64,{{ buffer_chart }}" alt="buffer">
    {% endif %}
  </div></div>
</div>
</div>

<!-- 4. 博弈层 -->
<div class="section">
<div class="sec-label">4 · 博弈层 (Schelling)</div>
<div class="card">
  <div class="card-title">本期活跃信号</div>
  <div style="font-size:11px;color:#64748b;margin-bottom:8px;">strength = credibility × evidence × {{ base_sensitivity }}</div>
  {% if signal_display %}
  <table>
    <tr><th>信号</th><th>方向</th><th>可信度</th><th>证据</th><th>强度</th></tr>
    {% for s in signal_display %}
    <tr>
      <td>{{ s.name }}</td>
      <td class="{{ s.dir_class }}">{{ s.direction }}</td>
      <td class="mono">{{ s.credibility }}</td>
      <td><span class="tag tag-{{ s.evidence_label }}">{{ s.evidence_label }}</span> {{ s.evidence }}</td>
      <td class="mono param-val">{{ s.strength }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <div style="font-size:12px;color:#475569;">本期无博弈信号触发</div>
  {% endif %}
</div>
</div>

<!-- 5. 路径 -->
<div class="section">
<div class="sec-label">5 · 路径卡片</div>
<div class="row">
  {% for path, data in paths.items() %}
  <div class="col"><div class="card" style="border-top:3px solid {{ data.color }};">
    <div class="card-title">路径 {{ path }}</div>
    <div class="big-num" style="color:{{ data.color }};">{{ "%.0f"|format(data.prob*100) }}%</div>
    <div style="font-size:12px;margin-top:6px;">TotalGap: {{ "%.0f"|format(data.gap) }} mbd·天</div>
    <div style="font-size:11px;color:#475569;">{{ data.desc }}</div>
  </div></div>
  {% endfor %}
</div>
</div>

<!-- 6. MC -->
<div class="section">
<div class="sec-label">6 · MC 分布</div>
<div class="card">
  {% if mc_chart %}
  <img class="chart" src="data:image/png;base64,{{ mc_chart }}" alt="MC distributions">
  {% endif %}
  <div style="margin-top:8px;">
    <table>
      <tr><th></th><th>A (快速)</th><th>B (拉锯)</th><th>C (长期)</th></tr>
      <tr>
        <td style="color:#64748b;">MC 物理频率</td>
        <td>{{ "%.0f"|format(mc_freq_a*100) }}%</td>
        <td>{{ "%.0f"|format(mc_freq_b*100) }}%</td>
        <td>{{ "%.0f"|format(mc_freq_c*100) }}%</td>
      </tr>
      <tr>
        <td style="color:#64748b;">M5 博弈调整</td>
        <td>{{ "%.0f"|format(pa*100) }}%</td>
        <td>{{ "%.0f"|format(pb*100) }}%</td>
        <td>{{ "%.0f"|format(pc*100) }}%</td>
      </tr>
      <tr>
        <td style="color:#64748b;">差异</td>
        <td class="{{ 'green' if pa > mc_freq_a else 'red' if pa < mc_freq_a else 'grey' }}">{{ "%+.0f"|format((pa - mc_freq_a)*100) }}pp</td>
        <td class="{{ 'green' if pb > mc_freq_b else 'red' if pb < mc_freq_b else 'grey' }}">{{ "%+.0f"|format((pb - mc_freq_b)*100) }}pp</td>
        <td class="{{ 'green' if pc < mc_freq_c else 'red' if pc > mc_freq_c else 'grey' }}">{{ "%+.0f"|format((pc - mc_freq_c)*100) }}pp</td>
      </tr>
    </table>
  </div>
</div>
</div>

<!-- 7. 本轮信息源 -->
{% if recent_events %}
{% set obs_zh = {
  "O01": "攻击频率", "O02": "攻击趋势", "O03": "攻击协调",
  "O04": "高端武器", "O05": "GPS欺骗", "O06": "网络分布",
  "O07": "战争险", "O08": "P&I", "O09": "VLCC运费",
  "O10": "通行量", "O11": "延布装载", "O12": "富查伊拉价差",
  "O13": "SPR释放", "O14": "外部补给"
} %}
<div class="section">
<div class="sec-label">7 · 本轮信息源（{{ recent_events|length }} 篇）</div>
<div class="card">
  <table style="width:100%;font-size:12px;">
    <tr><th style="width:50%;">文章</th><th>来源</th><th>提取观测</th></tr>
    {% for ev in recent_events %}
    <tr>
      <td style="color:#e2e8f0;" title="{{ ev.title }}">{% if ev.title_zh %}{{ ev.title_zh }}{% else %}{{ ev.title[:55] }}{% if ev.title|length > 55 %}…{% endif %}{% endif %}</td>
      <td style="color:#64748b;font-size:11px;">{{ ev.source[:12] }}</td>
      <td>{% for oid in ev.obs_ids.split(',') %}<span style="background:#1e293b;padding:1px 5px;border-radius:3px;margin-right:2px;font-size:10px;" title="{{ obs_zh.get(oid, oid) }}">{{ oid }}</span>{% endfor %}</td>
    </tr>
    {% endfor %}
  </table>
</div>
</div>
{% endif %}

<!-- 8. 仓位 -->
<div class="section">
<div class="sec-label">8 · 仓位建议</div>
<div class="row">
  <div class="col"><div class="card">
    <div class="card-title">能源多头</div>
    <div class="big-num amber">{{ pos_energy }}%</div>
  </div></div>
  <div class="col"><div class="card">
    <div class="card-title">波动率</div>
    <div class="big-num blue">{{ pos_vol }}%</div>
  </div></div>
  <div class="col"><div class="card">
    <div class="card-title">衰退对冲</div>
    <div class="big-num green">{{ pos_recession }}%</div>
  </div></div>
</div>
{% if pos_actions %}
<div class="card">
  <div class="card-title">仓位调整动作</div>
  {% for a in pos_actions %}
  <div class="pos-action">{{ a }}</div>
  {% endfor %}
</div>
{% endif %}
<div style="font-size:11px;color:#475569;margin-top:4px;">* 仅为建议，position_signals.executed 为人工确认边界</div>
</div>

<!-- 8. 校验 -->
<div class="section">
<div class="sec-label">8 · 校验层</div>
{% if flags %}
  {% for f in flags %}
  <div class="flag">⚠ {{ f }}</div>
  {% endfor %}
{% else %}
  <div class="card" style="color:#10b981;">✓ 无一致性警告</div>
{% endif %}
</div>

</div><!-- end status panel -->

<!-- ════════════════ TAB 2: 框架参考 ════════════════ -->
<div id="framework" class="panel">

<div class="section">
<div class="sec-label">引擎流程</div>
<div class="flow">
  <div class="flow-box" style="border-color:#10b981;"><div class="label">M1</div><div class="name green">ACH</div></div>
  <div class="flow-arrow">→</div>
  <div class="flow-box" style="border-color:#3b82f6;"><div class="label">M2</div><div class="name blue">T 分布</div></div>
  <div class="flow-arrow">→</div>
  <div class="flow-box" style="border-color:#3b82f6;"><div class="label">M3</div><div class="name blue">Buffer</div></div>
  <div class="flow-arrow">→</div>
  <div class="flow-box" style="border-color:#3b82f6;"><div class="label">M4</div><div class="name blue">Gap</div></div>
  <div class="flow-arrow">→</div>
  <div class="flow-box" style="border-color:#f59e0b;"><div class="label">MC</div><div class="name amber">模拟</div></div>
  <div class="flow-arrow">→</div>
  <div class="flow-box" style="border-color:#ef4444;"><div class="label">M5</div><div class="name red">博弈</div></div>
</div>
</div>

<div class="section">
<div class="sec-label">参数总表</div>
<div class="card">
<table>
<tr><th>ID</th><th>参数</th><th>当前值</th><th>来源</th></tr>
<tr class="sec-group"><td colspan="4">常数 (C) — 物理定律/地理，永不变</td></tr>
<tr><td>C01</td><td>正常通行流量</td><td>20.1 mbd</td><td>EIA/OPEC</td></tr>
<tr><td>C02</td><td>海峡可航宽度</td><td>9.0 km</td><td>海图</td></tr>
<tr><td>C03</td><td>扫雷面积</td><td>航道总雷区面积</td><td>海图</td></tr>
<tr><td>C04</td><td>水雷类型</td><td>触发/磁性/声学</td><td>伊朗已知库存</td></tr>
<tr><td>C05</td><td>单舰扫雷上限</td><td>每天最大清扫面积</td><td>MCM 规格</td></tr>
<tr class="sec-group"><td colspan="4">参数 (P) — 可调，校准或人工覆盖</td></tr>
<tr><td>P01</td><td>总缺口 (GrossGap)</td><td>{{ gross_gap }} mbd</td><td>C01 × {{ disruption_rate }}</td></tr>
<tr><td>P02</td><td>水中水雷</td><td>{{ mines_range }}</td><td>Uniform 分布</td></tr>
<tr><td>P03</td><td>扫雷舰</td><td>{{ sweep_ships }} 艘</td><td>多国 MCM</td></tr>
<tr><td>P04</td><td>管道替代</td><td>{{ pipeline_max }} mbd</td><td>ADCOP+沙特</td></tr>
<tr><td>P05</td><td>管道爬坡周期</td><td>{{ pipeline_ramp }} 周</td><td>物理约束</td></tr>
<tr><td>P06</td><td>SPR 释放率</td><td>{{ spr_rate }} mbd</td><td>DOE</td></tr>
<tr><td>P07</td><td>SPR 延迟</td><td>{{ spr_delay }} 天</td><td>物理约束</td></tr>
<tr><td>P08</td><td>H3 悬置</td><td class="param-val">{{ h3_suspended }}</td><td>梅赫拉巴德机场摧毁</td></tr>
<tr><td>P09</td><td>H3 先验</td><td>{{ h3_prior }}</td><td>重分配给 H1/H2</td></tr>
<tr><td>P10</td><td>有效中断率</td><td>{{ disruption_rate }}</td><td>校准 (1984~70%, 2026~92%)</td></tr>
<tr class="sec-group"><td colspan="4">M1 ACH — 贝叶斯推理</td></tr>
<tr><td>—</td><td>先验</td><td>H1=H2=50% (H3 悬置时)</td><td>均匀先验+H3重分配</td></tr>
<tr><td>—</td><td>似然比范围</td><td>{0.77, 0.95, 1.0, 1.05, 1.3}</td><td>强=1.3/0.77, 中=1.05/0.95</td></tr>
<tr><td>—</td><td>后验上限</td><td>~95% (log-odds clamp)</td><td>防过度自信</td></tr>
<tr><td>—</td><td>O05 T1a/T1b</td><td>GPS高+攻击↑=进攻H2 / GPS高+攻击↓=防御H2</td><td>解绑规则</td></tr>
<tr class="sec-group"><td colspan="4">M3 Buffer 爬坡</td></tr>
<tr><td>—</td><td>D1-D14 缓冲</td><td>1.5 mbd → 净缺口 ~14.5</td><td>管道初期</td></tr>
<tr><td>—</td><td>D14+ 缓冲</td><td>7.0 mbd → 净缺口 ~9.0</td><td>管道+SPR+富查伊拉</td></tr>
<tr><td>—</td><td>路径C崩溃</td><td>2.0 mbd → 净缺口 ~14</td><td>富查伊拉被击中</td></tr>
<tr class="sec-group"><td colspan="4">M5 博弈层</td></tr>
<tr><td>—</td><td>BASE_SENSITIVITY</td><td class="param-val">{{ base_sensitivity }}</td><td>全局灵敏度</td></tr>
<tr><td>—</td><td>FOCAL_BONUS</td><td class="param-val">{{ focal_bonus }}</td><td>焦点收敛系数</td></tr>
<tr class="sec-group"><td colspan="4">MC / 路径</td></tr>
<tr><td>—</td><td>采样数</td><td>N={{ mc_n }}</td><td>—</td></tr>
<tr><td>—</td><td>路径边界</td><td>A&lt;35d / B=35-120d / C&gt;120d</td><td>PRD</td></tr>
<tr class="sec-group"><td colspan="4">仓位规则</td></tr>
<tr><td>—</td><td>基础仓位</td><td>能源 15% / 波动 3% / 衰退 2%</td><td>PRD</td></tr>
<tr><td>—</td><td>系统失效</td><td>Brent &lt; $80 × 3天 → 清仓</td><td>退出规则</td></tr>
<tr><td>—</td><td>需求毁灭</td><td>Brent &gt; $150 → 清能源</td><td>退出规则</td></tr>
<tr><td>—</td><td>最大亏损</td><td>8%</td><td>止损线</td></tr>
</table>
</div>
</div>

<div class="section">
<div class="sec-label">观测变量 (O01-O14)</div>
<div class="card">
<table>
<tr><th>ID</th><th>名称</th><th>范围</th><th>ACH 作用</th></tr>
<tr class="sec-group"><td colspan="4">A组 — 威胁状态 (喂入 ACH)</td></tr>
<tr><td>O01</td><td>攻击频率</td><td>0-1</td><td>高→H2(0.95/1.05), 低→H1(1.05/0.95)</td></tr>
<tr><td>O02</td><td>攻击趋势变化</td><td>0-1</td><td><b>强</b>: 高→H1(1.3/0.77), 低→H2(0.95/1.05)</td></tr>
<tr><td>O03</td><td>攻击协调性</td><td>0-1</td><td><b>强</b>: 高→H2(0.77/1.3), 低→H1(1.3/0.77)</td></tr>
<tr><td>O04</td><td>高端武器使用</td><td>0-1</td><td><b>强</b>: 高→H2(0.77/1.3), 低→H1(1.3/0.77)</td></tr>
<tr><td>O05</td><td>GPS 欺骗复杂度</td><td>0-1</td><td>T1a/T1b 解绑（依赖 O01 趋势）</td></tr>
<tr><td>O06</td><td>网络碎片化</td><td>0-1</td><td>高→H2(0.95/1.05), 低→H1(1.05/0.95)</td></tr>
<tr class="sec-group"><td colspan="4">B组 — 封锁/恢复</td></tr>
<tr><td>O07</td><td>战争险附加费</td><td>%</td><td>&gt;1%→H2, &lt;1%→H1</td></tr>
<tr><td>O08</td><td>P&I 除外条款</td><td>0-1</td><td>高→H2(0.95/1.05), 低→H1(1.05/0.95)</td></tr>
<tr><td>O09</td><td>VLCC 即期运费</td><td>WS点</td><td>&gt;150→elevated（不入ACH，校准参照）</td></tr>
<tr><td>O10</td><td>海峡日通行量</td><td>0-1</td><td>高→H1(1.05/0.95), 低→H2(0.95/1.05)</td></tr>
<tr class="sec-group"><td colspan="4">C组 — 缓冲到位</td></tr>
<tr><td>O11</td><td>延布港装船量</td><td>0-1</td><td>高→H2(0.95/1.05)（管道分流代理）</td></tr>
<tr><td>O12</td><td>富查伊拉-新加坡价差</td><td>$/mt</td><td>&gt;$50→物流崩溃（不入ACH）</td></tr>
<tr><td>O13</td><td>SPR 释放率</td><td>mbd</td><td>&gt;1mbd→active（不入ACH）</td></tr>
<tr class="sec-group"><td colspan="4">H3 解冻监控</td></tr>
<tr><td>O14</td><td>未知武器类型</td><td>0-1</td><td>high confidence → 解冻H3（3-way ACH）</td></tr>
</table>
</div>
</div>

<div class="section">
<div class="sec-label">控制变量 (D01-D05)</div>
<div class="card">
<div style="font-size:12px;color:#94a3b8;">
  行为者决策变量，事件驱动触发。triggered=true 时 effect 字段映射到 Schelling 信号（evidence=0.5）。
</div>
<table>
<tr><th>字段</th><th>说明</th></tr>
<tr><td>id</td><td>D01-D05 标识符</td></tr>
<tr><td>actor</td><td>行为者（US / Iran / IRGC / mediator / …）</td></tr>
<tr><td>triggered</td><td>是否激活（DB 手动标记）</td></tr>
<tr><td>effect</td><td>→ Schelling 信号 key（如 costly_self_binding）</td></tr>
</table>
</div>
</div>

<div class="section">
<div class="sec-label">M5 Schelling 信号定义</div>
<div class="card">
<div class="card-title">credibility = cost × 0.6 + observability × 0.4 · strength = credibility × evidence × {{ base_sensitivity }} · focal = 1 + {{ focal_bonus }} × (N−1)</div>
<table>
<tr><th>信号</th><th>方向</th><th>cost</th><th>observ.</th><th>credibility</th><th>前置要求</th></tr>
{% for key, sdef in signal_defs.items() %}
{% set combo = combo_requires.get(key) %}
<tr>
  <td>{{ signal_names.get(key, key) }}</td>
  <td class="{{ {'A':'green','B':'amber','C':'red'}[sdef.direction] }}">{{ sdef.direction }}</td>
  <td>{{ "%.1f"|format(sdef.cost) }}</td>
  <td>{{ "%.1f"|format(sdef.observability) }}</td>
  <td class="mono param-val">{{ "%.2f"|format(sdef.credibility) }}</td>
  <td class="dim">{{ combo|join(', ') if combo else '—' }}</td>
</tr>
{% endfor %}
</table>
</div>
</div>

<div class="section">
<div class="sec-label">校准参照 (R01-R03)</div>
<div class="card">
<table>
<tr><th>ID</th><th>事件</th><th>年份</th><th>校准用途</th></tr>
<tr><td>R01</td><td>Operation Praying Mantis</td><td>1988</td><td>Q1 主动干扰场景时长基线</td></tr>
<tr><td>R02</td><td>海湾战争扫雷</td><td>1991</td><td>Q2 扫雷时间线校准</td></tr>
<tr><td>R03</td><td>元山水雷封锁</td><td>1950</td><td>密集水雷场景上界（3000枚，2周延迟）</td></tr>
</table>
</div>
</div>

</div><!-- end framework panel -->

<div style="text-align:center;color:#334155;font-size:10px;margin-top:16px;">
  hormuz-ds v5.4 · {{ timestamp }}
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


def render_status(
    system_output: SystemOutput,
    mc_result: MCResult,
    params: Parameters,
    output_path: Path,
    conflict_start: str = "2026-03-01",
    brent_price: float = 95.0,
    position_result: object | None = None,
    game_signals: list[SignalEvidence] | None = None,
    mc_n: int = 10000,
    recent_events: list[dict] | None = None,
) -> None:
    """Render two-tab HTML dashboard: status + framework reference."""
    so = system_output

    start = datetime.strptime(conflict_start, "%Y-%m-%d")
    conflict_day = (so.timestamp - start).days

    mc_chart = generate_mc_chart(mc_result)
    buffer_chart = generate_buffer_chart(so.buffer_trajectory) if so.buffer_trajectory else ""

    paths = {
        "A": {"prob": so.path_probabilities.a, "gap": so.path_total_gaps.get("A", 0),
               "color": "#10b981", "desc": "快速解决 (<35天)"},
        "B": {"prob": so.path_probabilities.b, "gap": so.path_total_gaps.get("B", 0),
               "color": "#f59e0b", "desc": "中等拖延 (35-120天)"},
        "C": {"prob": so.path_probabilities.c, "gap": so.path_total_gaps.get("C", 0),
               "color": "#ef4444", "desc": "长期危机 (>120天)"},
    }

    pos_energy = pos_vol = pos_recession = 0
    pos_actions: list[str] = []
    if position_result is not None:
        pos_energy = getattr(position_result, "energy_pct", 15)
        pos_vol = getattr(position_result, "vol_pct", 3)
        pos_recession = getattr(position_result, "recession_pct", 2)
        pos_actions = getattr(position_result, "actions", [])
    else:
        pos_energy, pos_vol, pos_recession = 15, 3, 2

    conf = so.confidence_level
    conf_zh = {"burn_in": "BURN-IN", "low": "LOW", "normal": "NORMAL"}.get(conf, conf)

    signal_display = _build_signal_display(game_signals or [])

    env = Environment(loader=BaseLoader())
    template = env.from_string(_TEMPLATE)

    html = template.render(
        timestamp=so.timestamp.strftime("%Y-%m-%d %H:%M"),
        conflict_day=conflict_day,
        brent_price=f"{brent_price:.1f}",
        confidence=conf,
        confidence_zh=conf_zh,
        dominant=so.ach_posterior.dominant,
        h1=so.ach_posterior.h1,
        h2=so.ach_posterior.h2,
        h3=so.ach_posterior.h3,
        t_p50=int(so.t_total_percentiles.get("p50", 0)),
        t_p10=int(so.t_total_percentiles.get("p10", 0)),
        t_p90=int(so.t_total_percentiles.get("p90", 0)),
        pa=so.path_probabilities.a,
        pb=so.path_probabilities.b,
        pc=so.path_probabilities.c,
        gross_gap=f"{so.gross_gap_mbd:.1f}",
        expected_gap=so.expected_total_gap,
        buffer_chart=buffer_chart,
        mc_chart=mc_chart,
        mc_freq_a=mc_result.path_frequencies.get("A", 0),
        mc_freq_b=mc_result.path_frequencies.get("B", 0),
        mc_freq_c=mc_result.path_frequencies.get("C", 0),
        paths=paths,
        # Section 4: Schelling signals
        signal_display=signal_display,
        base_sensitivity=BASE_SENSITIVITY,
        # Framework tab
        signal_defs=_SIGNAL_DEFS,
        combo_requires=_COMBO_REQUIRES,
        signal_names=_SIGNAL_NAMES,
        focal_bonus=FOCAL_BONUS,
        # Section 7: positions
        pos_energy=pos_energy,
        pos_vol=pos_vol,
        pos_recession=pos_recession,
        pos_actions=pos_actions,
        # Params table
        mines_range=f"{params.mines_in_water_range}",
        sweep_ships=params.sweep_ships,
        pipeline_max=f"{params.pipeline_max_mbd}",
        pipeline_ramp=f"{params.pipeline_ramp_weeks}",
        spr_rate=f"{params.spr_rate_mean_mbd}",
        spr_delay=f"{params.spr_pump_min_days}",
        disruption_rate=f"{params.effective_disruption_rate}",
        h3_suspended="是" if params.h3_suspended else "否",
        h3_prior=f"{params.h3_prior}",
        mc_n=mc_n,
        flags=so.consistency_flags,
        recent_events=recent_events or [],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
