#!/usr/bin/env python3
"""Load seed_all.yaml into DB, compute O02, run engine, print results."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml

from hormuz.core.types import Observation, Constants, Parameters
from hormuz.core.variables import load_constants, load_parameters
from hormuz.infra.db import (
    init_db,
    insert_observations,
    get_observations,
    compute_o02_from_history,
    compute_o01_trend,
    compute_confidence_level,
    save_system_output,
)
from hormuz.app.pipeline import engine_run


def main():
    db_path = ROOT / "data" / "hormuz.db"
    seed_path = ROOT / "data" / "seed_all.yaml"
    configs_dir = ROOT / "configs"

    # 1. Parse seed YAML
    data = yaml.safe_load(seed_path.read_text())
    observations: list[Observation] = []
    for entry in data["seed"]:
        date_str = entry["date"]
        ts = datetime.fromisoformat(date_str + "T12:00:00")
        for key, val in entry.items():
            if key == "date":
                continue
            if not key.startswith("O"):
                continue
            observations.append(Observation(
                id=key,
                timestamp=ts,
                value=float(val),
                source="seed:wikipedia",
            ))

    print(f"Parsed {len(observations)} observations from seed_all.yaml")

    # 2. Clear old observations and re-init DB
    import sqlite3
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    old_count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    conn.execute("DELETE FROM observations")
    conn.commit()
    conn.close()
    print(f"Cleared {old_count} old observations from DB")

    # 3. Insert seed observations
    insert_observations(db_path, observations)
    print(f"Inserted {len(observations)} seed observations")

    # 4. Compute O02 from O01 history
    # Since timestamps are historical (2026-02/03), we need to temporarily
    # compute O02 manually rather than using compute_o02_from_history
    # (which uses datetime.now() as cutoff)
    o01_by_date: dict[str, float] = {}
    for o in observations:
        if o.id == "O01":
            o01_by_date[o.timestamp.isoformat()[:10]] = o.value

    o01_dates = sorted(o01_by_date.keys())
    o02_obs: list[Observation] = []
    for i, date_str in enumerate(o01_dates):
        if i == 0:
            o02_val = 0.50  # first day, no trend
        else:
            # Compare current to 3-day rolling avg of previous days
            window = o01_dates[max(0, i - 3):i]
            prev_avg = sum(o01_by_date[d] for d in window) / len(window)
            current = o01_by_date[date_str]
            if prev_avg == 0:
                o02_val = 0.50
            else:
                ratio = (prev_avg - current) / prev_avg
                o02_val = max(0.0, min(1.0, 0.5 + ratio))
        o02_obs.append(Observation(
            id="O02",
            timestamp=datetime.fromisoformat(date_str + "T12:00:00"),
            value=round(o02_val, 2),
            source="seed:computed",
        ))

    insert_observations(db_path, o02_obs)
    print(f"Computed and inserted {len(o02_obs)} O02 values:")
    for o in o02_obs:
        print(f"  {o.timestamp.isoformat()[:10]}: O02={o.value:.2f}")

    # 5. Load all observations for engine
    all_obs = get_observations(db_path)
    # Keep latest per obs_id (use last day's values for engine)
    latest: dict[str, Observation] = {}
    for o in sorted(all_obs, key=lambda x: x.timestamp):
        latest[o.id] = o

    engine_obs = list(latest.values())
    obs_summary = {o.id: o.value for o in sorted(engine_obs, key=lambda x: x.id)}
    print(f"\nEngine input ({len(engine_obs)} latest observations):")
    for oid, val in obs_summary.items():
        print(f"  {oid} = {val}")

    # 6. Determine O01 trend manually (since DB timestamps are historical)
    if len(o01_dates) >= 4:
        mid = len(o01_dates) // 2
        older = [o01_by_date[d] for d in o01_dates[:mid]]
        recent = [o01_by_date[d] for d in o01_dates[mid:]]
        older_avg = sum(older) / len(older)
        recent_avg = sum(recent) / len(recent)
        diff = recent_avg - older_avg
        if diff > 0.05:
            o01_trend = "rising"
        elif diff < -0.05:
            o01_trend = "falling"
        else:
            o01_trend = "stable"
    else:
        o01_trend = "stable"
    print(f"\nO01 trend: {o01_trend}")

    # 7. Extract events from observations
    events: dict[str, bool] = {}
    o06_val = obs_summary.get("O06")
    o10_val = obs_summary.get("O10")
    o01_latest = obs_summary.get("O01")
    if o06_val is not None and o06_val < 0.4:
        events["E2"] = True
    if o10_val is not None and o10_val < 0.1 and o01_latest is not None and o01_latest > 0.6:
        events["E3"] = True
    if events:
        print(f"Events detected: {events}")

    # 8. Infer SPR trigger day from O13
    spr_trigger_day = None
    o13_val = obs_summary.get("O13")
    if o13_val is not None and o13_val > 0:
        spr_trigger_day = 1
        print(f"SPR trigger day: {spr_trigger_day} (O13={o13_val})")

    # 9. Run engine
    constants = load_constants(configs_dir / "constants.yaml")
    params = load_parameters(configs_dir / "parameters.yaml")

    print(f"\nRunning engine (MC N=10000)...")
    so, mc_result = engine_run(
        constants=constants,
        params=params,
        observations=engine_obs,
        controls=[],
        events=events,
        mc_n=10000,
        seed=42,
        o01_trend=o01_trend,
    )

    # Confidence based on history depth (15 days = normal)
    so.confidence_level = "normal"

    # 8. Save to DB
    save_system_output(db_path, so)

    # 9. Print results
    print("\n" + "=" * 60)
    print("  HORMUZ DECISION OS — SEED RUN RESULTS")
    print("=" * 60)
    print(f"\n  ACH Posterior:")
    print(f"    H1 (快速解决) = {so.ach_posterior.h1:.1%}")
    print(f"    H2 (长期封锁) = {so.ach_posterior.h2:.1%}")
    print(f"    Dominant: {so.ach_posterior.dominant}")

    print(f"\n  T Distribution (days):")
    print(f"    T1 p50={so.t1_percentiles.get('p50', '?'):.0f}")
    print(f"    T2 p50={so.t2_percentiles.get('p50', '?'):.0f}")
    print(f"    T_total p10={so.t_total_percentiles['p10']:.0f}  p50={so.t_total_percentiles['p50']:.0f}  p90={so.t_total_percentiles['p90']:.0f}")

    print(f"\n  Path Probabilities:")
    print(f"    A (短期<35天) = {so.path_probabilities.a:.1%}")
    print(f"    B (中期35-120天) = {so.path_probabilities.b:.1%}")
    print(f"    C (长期>120天) = {so.path_probabilities.c:.1%}")

    print(f"\n  Gap Analysis:")
    print(f"    Gross Gap: {so.gross_gap_mbd:.1f} mbd")
    print(f"    Path A TotalGap: {so.path_total_gaps['A']:.0f} mbd·天")
    print(f"    Path B TotalGap: {so.path_total_gaps['B']:.0f} mbd·天")
    print(f"    Path C TotalGap: {so.path_total_gaps['C']:.0f} mbd·天")
    print(f"    Expected TotalGap: {so.expected_total_gap:.0f} mbd·天")

    print(f"\n  MC Results:")
    print(f"    Path frequencies: A={mc_result.path_frequencies['A']:.1%}  B={mc_result.path_frequencies['B']:.1%}  C={mc_result.path_frequencies['C']:.1%}")
    print(f"    Path gap means: A={mc_result.path_total_gap_means.get('A', 0):.0f}  B={mc_result.path_total_gap_means.get('B', 0):.0f}  C={mc_result.path_total_gap_means.get('C', 0):.0f}")

    if so.consistency_flags:
        print(f"\n  Flags:")
        for f in so.consistency_flags:
            print(f"    ⚠ {f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
