from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PORTFOLIO_METHOD = "learned_portfolio"
SINGLE_PROFILE_METHODS = ("fast_only", "balanced_only", "safe_only")
PRIMARY_METHODS = (PORTFOLIO_METHOD, *SINGLE_PROFILE_METHODS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize lazy rolling dispatch mission/event outputs.")
    parser.add_argument("--rolling-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    missions = pd.read_csv(args.rolling_dir / "lazy_rolling_missions.csv")
    events = pd.read_csv(args.rolling_dir / "lazy_rolling_events.csv")
    rng = np.random.default_rng(args.seed)

    common = common_complete_summary(missions)
    paired = paired_gains(missions)
    paired_ci = paired_gain_ci(missions, rng, args.bootstrap)
    modes = mode_mix(events)
    provenance = provenance_summary(missions)

    common.to_csv(args.rolling_dir / "rolling_common_complete_summary.csv", index=False)
    paired.to_csv(args.rolling_dir / "rolling_paired_gains.csv", index=False)
    paired_ci.to_csv(args.rolling_dir / "rolling_paired_gain_ci.csv", index=False)
    modes.to_csv(args.rolling_dir / "rolling_mode_mix.csv", index=False)
    (args.rolling_dir / "rolling_provenance_summary.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"rolling_dir": str(args.rolling_dir), **provenance}, indent=2))


def common_complete_summary(missions: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mission_true_length",
        "mission_true_risk",
        "mean_flow_time",
        "total_planner_calls",
        "planner_runtime_sec",
        "prediction_runtime_sec",
        "decision_wall_time_sec",
        "planner_call_reduction_vs_exhaustive",
        "exhaustive_candidate_planner_call_estimate",
    ]
    rows: list[dict[str, Any]] = []
    for survival, survival_group in missions.groupby("minimum_mean_route_survival", sort=True):
        complete_ids = common_complete_ids(survival_group, PRIMARY_METHODS)
        subset = survival_group[survival_group["mission_id"].astype(int).isin(complete_ids)].copy()
        for method, group in subset.groupby("method", sort=False):
            row: dict[str, Any] = {
                "minimum_mean_route_survival": float(survival),
                "method": method,
                "missions": int(group["mission_id"].nunique()),
            }
            for metric in metrics:
                if metric in group.columns:
                    row[f"{metric}_mean"] = float(group[metric].astype(float).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def paired_gains(missions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for survival, group in missions.groupby("minimum_mean_route_survival", sort=True):
        for reference in SINGLE_PROFILE_METHODS:
            row = paired_gain_row(group, f"portfolio_vs_{reference}", PORTFOLIO_METHOD, reference)
            row["minimum_mean_route_survival"] = float(survival)
            rows.append(row)
    return pd.DataFrame(rows)


def paired_gain_ci(missions: pd.DataFrame, rng: np.random.Generator, bootstrap: int) -> pd.DataFrame:
    rows = []
    for survival, group in missions.groupby("minimum_mean_route_survival", sort=True):
        for reference in SINGLE_PROFILE_METHODS:
            values = paired_gain_values(group, PORTFOLIO_METHOD, reference).to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row: dict[str, Any] = {
                "minimum_mean_route_survival": float(survival),
                "comparison": f"portfolio_vs_{reference}",
                "n": int(len(values)),
            }
            if len(values):
                boot = np.empty(int(bootstrap), dtype=float)
                for index in range(int(bootstrap)):
                    sample = rng.integers(0, len(values), size=len(values))
                    boot[index] = float(np.mean(values[sample]))
                row.update(
                    mean=float(np.mean(values)),
                    median=float(np.median(values)),
                    ci95_low=float(np.quantile(boot, 0.025)),
                    ci95_high=float(np.quantile(boot, 0.975)),
                    p10=float(np.quantile(values, 0.10)),
                    p90=float(np.quantile(values, 0.90)),
                )
            rows.append(row)
    return pd.DataFrame(rows)


def paired_gain_row(missions: pd.DataFrame, label: str, portfolio: str, reference: str) -> dict[str, Any]:
    gains = paired_gain_values(missions, portfolio, reference)
    values = gains.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "comparison": label,
            "missions": 0,
            "mean_length_gain": math.nan,
            "median_length_gain": math.nan,
            "win_rate": math.nan,
            "tie_rate": math.nan,
            "loss_rate": math.nan,
            "p10_length_gain": math.nan,
            "p90_length_gain": math.nan,
        }
    return {
        "comparison": label,
        "missions": int(len(values)),
        "mean_length_gain": float(np.mean(values)),
        "median_length_gain": float(np.median(values)),
        "win_rate": float(np.mean(values > 1e-6)),
        "tie_rate": float(np.mean(np.abs(values) <= 1e-6)),
        "loss_rate": float(np.mean(values < -1e-6)),
        "p10_length_gain": float(np.quantile(values, 0.10)),
        "p90_length_gain": float(np.quantile(values, 0.90)),
    }


def paired_gain_values(missions: pd.DataFrame, portfolio: str, reference: str) -> pd.Series:
    complete_ids = common_complete_ids(missions, (portfolio, reference))
    subset = missions[missions["mission_id"].astype(int).isin(complete_ids)].copy()
    pivot = subset.pivot_table(index="mission_id", columns="method", values="mission_true_length", aggfunc="first")
    if portfolio not in pivot.columns or reference not in pivot.columns:
        return pd.Series(dtype=float)
    return pivot[reference].astype(float) - pivot[portfolio].astype(float)


def common_complete_ids(missions: pd.DataFrame, methods: tuple[str, ...]) -> set[int]:
    pivot = missions.pivot_table(index="mission_id", columns="method", values="completion_rate", aggfunc="first")
    if any(method not in pivot.columns for method in methods):
        return set()
    complete = np.ones(len(pivot), dtype=bool)
    for method in methods:
        complete &= pivot[method].astype(float).to_numpy() >= 1.0 - 1e-12
    return {int(item) for item in pivot.index[complete].tolist()}


def mode_mix(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (survival, method), group in events.groupby(["minimum_mean_route_survival", "method"], sort=True):
        counts = {"fast": 0, "balanced": 0, "safe": 0}
        active_events = 0
        ge2 = 0
        all3 = 0
        for raw in group["modes"].dropna().astype(str):
            modes = [item for item in raw.split(";") if item]
            if not modes:
                continue
            active_events += 1
            active = set(modes)
            ge2 += int(len(active) >= 2)
            all3 += int(len(active) == 3)
            for mode in modes:
                if mode in counts:
                    counts[mode] += 1
        total = sum(counts.values())
        rows.append(
            {
                "minimum_mean_route_survival": float(survival),
                "method": method,
                "executed_routes": int(total),
                "fast_ratio": counts["fast"] / total if total else math.nan,
                "balanced_ratio": counts["balanced"] / total if total else math.nan,
                "safe_ratio": counts["safe"] / total if total else math.nan,
                "fast_count": counts["fast"],
                "balanced_count": counts["balanced"],
                "safe_count": counts["safe"],
                "events_with_modes": int(active_events),
                "events_using_ge2_modes_rate": ge2 / active_events if active_events else math.nan,
                "events_using_all3_modes_rate": all3 / active_events if active_events else math.nan,
            }
        )
    return pd.DataFrame(rows)


def provenance_summary(missions: pd.DataFrame) -> dict[str, Any]:
    unique = missions[["mission_id", "seed", "local_mission_id", "scenario_id"]].drop_duplicates()
    return {
        "missions": int(unique["mission_id"].nunique()),
        "unique_scenarios": int(unique["scenario_id"].nunique()),
        "scenario_id_min": int(unique["scenario_id"].min()),
        "scenario_id_max": int(unique["scenario_id"].max()),
        "seeds": [int(item) for item in sorted(unique["seed"].unique().tolist())],
        "local_mission_ids": [int(item) for item in sorted(unique["local_mission_id"].unique().tolist())],
    }


if __name__ == "__main__":
    main()
