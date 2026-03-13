"""HTML reporter — 9-section dashboard with parameter docs.

Renders SystemOutput + MCResult into self-contained HTML with base64 charts.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader, BaseLoader

from hormuz.core.types import Parameters, SystemOutput
from hormuz.core.mc import MCResult


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#0a0e1a", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_mc_chart(mc_result: MCResult) -> str:
    """T histogram + TotalGap histogram as base64 PNG."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    for ax in (ax1, ax2):
        ax.set_facecolor("#0a0e1a")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#1e293b")

    # T distribution
    ax1.hist(mc_result.t_samples, bins=40, color="#f59e0b", alpha=0.7, edgecolor="#0a0e1a")
    ax1.set_title("T 分布 (天)", color="#e2e8f0", fontsize=11)
    ax1.axvline(35, color="#ef4444", linestyle="--", alpha=0.5, label="A|B=35d")
    ax1.axvline(120, color="#ef4444", linestyle="--", alpha=0.5, label="B|C=120d")
    ax1.legend(fontsize=8, facecolor="#0a0e1a", edgecolor="#1e293b", labelcolor="#94a3b8")

    # TotalGap distribution
    ax2.hist(mc_result.total_gap_samples, bins=40, color="#3b82f6", alpha=0.7, edgecolor="#0a0e1a")
    ax2.set_title("TotalGap 分布 (mbd·天)", color="#e2e8f0", fontsize=11)

    fig.patch.set_facecolor("#0a0e1a")
    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_buffer_chart(buffer_trajectory: list[tuple[int, float]]) -> str:
    """Buffer ramp line chart as base64 PNG."""
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


_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>霍尔木兹决策系统 v5.4</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #07090f; font-family: 'Courier New', monospace; color: #e2e8f0; padding: 32px 20px; }
.wrap { max-width: 960px; margin: 0 auto; }
h1 { font-size: 22px; text-align: center; margin-bottom: 8px; }
.subtitle { text-align: center; color: #475569; font-size: 12px; margin-bottom: 32px; }
.section { margin-bottom: 28px; }
.sec-label { font-size: 12px; color: #475569; letter-spacing: 3px; margin-bottom: 10px; text-transform: uppercase; }
.card { background: #0a0e1a; border: 1px solid #1e293b; border-radius: 10px; padding: 16px 20px; margin-bottom: 10px; }
.card-title { font-size: 13px; color: #94a3b8; margin-bottom: 8px; }
.big-num { font-size: 28px; font-weight: 800; }
.amber { color: #f59e0b; }
.blue { color: #3b82f6; }
.green { color: #10b981; }
.red { color: #ef4444; }
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
</style>
</head>
<body>
<div class="wrap">
<h1>霍尔木兹决策系统 v5.4</h1>
<div class="subtitle">{{ timestamp }} · 冲突第 {{ conflict_day }} 天 · Brent ${{ brent_price }}</div>

<!-- 1. 状态总览 -->
<div class="section">
<div class="sec-label">1 · 状态总览</div>
<div class="row">
  <div class="col"><div class="card">
    <div class="card-title">ACH 主导假设</div>
    <div class="big-num {{ 'green' if dominant == 'H1' else 'red' if dominant == 'H2' else 'amber' }}">{{ dominant }}</div>
    <div style="margin-top:8px;font-size:12px;">H1={{ "%.0f"|format(h1*100) }}% H2={{ "%.0f"|format(h2*100) }}%</div>
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
<div class="sec-label">4 · 博弈层</div>
<div class="card">
  <div class="card-title">Schelling 信号 & 路径调节</div>
  <div style="font-size:12px;color:#94a3b8;">路径权重通过博弈信号 delta 调节后归一化 + clip [5%, 85%]</div>
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
  <div style="font-size:12px;margin-top:8px;">
    路径频率: A={{ "%.0f"|format(mc_freq_a*100) }}% B={{ "%.0f"|format(mc_freq_b*100) }}% C={{ "%.0f"|format(mc_freq_c*100) }}%
  </div>
</div>
</div>

<!-- 7. 仓位 -->
<div class="section">
<div class="sec-label">7 · 仓位建议</div>
<div class="row">
  <div class="col"><div class="card"><div class="card-title">能源多头</div><div class="big-num amber">15%</div></div></div>
  <div class="col"><div class="card"><div class="card-title">波动率</div><div class="big-num blue">3%</div></div></div>
  <div class="col"><div class="card"><div class="card-title">衰退对冲</div><div class="big-num green">2%</div></div></div>
</div>
</div>

<!-- 8. 参数 -->
<div class="section">
<div class="sec-label">8 · 参数设置说明</div>
<div class="card">
<table>
<tr><th>ID</th><th>参数</th><th>当前值</th><th>来源/分布</th></tr>
<tr><td>C01</td><td>正常流量</td><td>20.1 mbd</td><td>EIA/OPEC 常数</td></tr>
<tr><td>C02</td><td>海峡宽度</td><td>9.0 km</td><td>海图 常数</td></tr>
<tr><td>P01</td><td>总缺口</td><td>{{ gross_gap }} mbd</td><td>C01 × 0.80</td></tr>
<tr><td>P02</td><td>水中水雷</td><td>{{ mines_range }}</td><td>Uniform 分布</td></tr>
<tr><td>P04</td><td>管道最大</td><td>{{ pipeline_max }} mbd</td><td>ADCOP + 沙特管线</td></tr>
<tr><td>P06</td><td>SPR 释放率</td><td>{{ spr_rate }} mbd</td><td>DOE 均值</td></tr>
<tr><td>P07</td><td>SPR 延迟</td><td>{{ spr_delay }} 天</td><td>物理约束</td></tr>
<tr><td>MC</td><td>采样数</td><td>N=10000</td><td>—</td></tr>
<tr><td>—</td><td>路径边界</td><td>A&lt;35d / B=35-120d / C&gt;120d</td><td>PRD 定义</td></tr>
</table>
</div>
</div>

<!-- 9. 校验 -->
<div class="section">
<div class="sec-label">9 · 校验层</div>
{% if flags %}
  {% for f in flags %}
  <div class="flag">⚠ {{ f }}</div>
  {% endfor %}
{% else %}
  <div class="card" style="color:#10b981;">✓ 无一致性警告</div>
{% endif %}
</div>

</div>
</body>
</html>"""


def render_status(
    system_output: SystemOutput,
    mc_result: MCResult,
    params: Parameters,
    output_path: Path,
    conflict_start: str = "2026-03-01",
    brent_price: float = 95.0,
    overrides: list[dict] | None = None,
) -> None:
    """Render full 9-section HTML dashboard."""
    so = system_output

    # Conflict day count
    start = datetime.strptime(conflict_start, "%Y-%m-%d")
    conflict_day = (so.timestamp - start).days

    # Generate charts
    mc_chart = generate_mc_chart(mc_result)
    buffer_chart = generate_buffer_chart(so.buffer_trajectory) if so.buffer_trajectory else ""

    # Path data
    paths = {
        "A": {"prob": so.path_probabilities.a, "gap": so.path_total_gaps.get("A", 0),
               "color": "#10b981", "desc": "快速解决 (<35天)"},
        "B": {"prob": so.path_probabilities.b, "gap": so.path_total_gaps.get("B", 0),
               "color": "#f59e0b", "desc": "中等拖延 (35-120天)"},
        "C": {"prob": so.path_probabilities.c, "gap": so.path_total_gaps.get("C", 0),
               "color": "#ef4444", "desc": "长期危机 (>120天)"},
    }

    env = Environment(loader=BaseLoader())
    template = env.from_string(_TEMPLATE)

    html = template.render(
        timestamp=so.timestamp.strftime("%Y-%m-%d %H:%M"),
        conflict_day=conflict_day,
        brent_price=f"{brent_price:.1f}",
        dominant=so.ach_posterior.dominant,
        h1=so.ach_posterior.h1,
        h2=so.ach_posterior.h2,
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
        mines_range=f"{params.mines_in_water_range}",
        pipeline_max=f"{params.pipeline_max_mbd}",
        spr_rate=f"{params.spr_rate_mean_mbd}",
        spr_delay=f"{params.spr_pump_min_days}",
        flags=so.consistency_flags,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
