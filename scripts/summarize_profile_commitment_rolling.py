from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHODS = (
    "exact_portfolio",
    "learned_portfolio",
    "dispatch_global",
    "fast_only",
    "balanced_only",
    "safe_only",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize rolling profile-commitment controls and serial timing.")
    parser.add_argument("--learned-dir", type=Path, required=True)
    parser.add_argument("--global-dir", type=Path, required=True)
    parser.add_argument("--exact-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--global-runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    missions = combine_missions(args.learned_dir, args.global_dir, args.exact_dir)
    validate_missions(missions)
    events = combine_events(args.learned_dir, args.global_dir, args.exact_dir)
    rng = np.random.default_rng(args.seed)

    population = population_summary(missions)
    paired = paired_summary(missions, rng, args.bootstrap)
    modes = mode_summary(events)
    runtime = runtime_summary(args.runtime_dir, args.global_runtime_dir)

    missions.to_csv(args.output_dir / "rolling_commitment_missions.csv", index=False)
    population.to_csv(args.output_dir / "rolling_commitment_population.csv", index=False)
    paired.to_csv(args.output_dir / "rolling_commitment_paired.csv", index=False)
    modes.to_csv(args.output_dir / "rolling_commitment_modes.csv", index=False)
    runtime.to_csv(args.output_dir / "rolling_commitment_runtime.csv", index=False)
    payload = {
        "methods": list(METHODS),
        "missions_per_method": int(missions["mission_id"].nunique()),
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
    }
    (args.output_dir / "rolling_commitment_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


def combine_missions(learned_dir: Path, global_dir: Path, exact_dir: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(learned_dir / "lazy_rolling_missions.csv").query(
            "method in ['learned_portfolio', 'fast_only', 'balanced_only', 'safe_only']"
        ),
        pd.read_csv(global_dir / "lazy_rolling_missions.csv").query("method == 'dispatch_global'"),
        pd.read_csv(exact_dir / "lazy_rolling_missions.csv").query("method == 'exact_portfolio'"),
    ]
    return pd.concat(frames, ignore_index=True)


def combine_events(learned_dir: Path, global_dir: Path, exact_dir: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(learned_dir / "lazy_rolling_events.csv").query(
            "method in ['learned_portfolio', 'fast_only', 'balanced_only', 'safe_only']"
        ),
        pd.read_csv(global_dir / "lazy_rolling_events.csv").query("method == 'dispatch_global'"),
        pd.read_csv(exact_dir / "lazy_rolling_events.csv").query("method == 'exact_portfolio'"),
    ]
    return pd.concat(frames, ignore_index=True)


def validate_missions(missions: pd.DataFrame) -> None:
    counts = missions.groupby(["method", "minimum_mean_route_survival"])["mission_id"].nunique()
    if set(counts.index.get_level_values("method")) != set(METHODS):
        raise ValueError(f"rolling methods do not match protocol: {counts}")
    if counts.nunique() != 1:
        raise ValueError(f"rolling method mission counts differ: {counts}")
    duplicates = missions.duplicated(["mission_id", "minimum_mean_route_survival", "method"])
    if bool(duplicates.any()):
        raise ValueError("rolling inputs contain duplicate mission/method rows")
    provenance = missions.groupby("mission_id")[["seed", "local_mission_id", "scenario_id"]].nunique()
    if bool((provenance > 1).any().any()):
        raise ValueError("rolling methods do not share mission provenance")


def population_summary(missions: pd.DataFrame) -> pd.DataFrame:
    return (
        missions.groupby(["minimum_mean_route_survival", "method"], as_index=False)
        .agg(
            missions=("mission_id", "nunique"),
            completed_tasks_mean=("completed_tasks", "mean"),
            completion_rate_mean=("completion_rate", "mean"),
            mission_true_length_mean=("mission_true_length", "mean"),
            mission_true_risk_mean=("mission_true_risk", "mean"),
            total_planner_calls_mean=("total_planner_calls", "mean"),
            prediction_runtime_sec_mean=("prediction_runtime_sec", "mean"),
            assignment_runtime_sec_mean=("assignment_runtime_sec", "mean"),
            decision_wall_time_sec_mean=("decision_wall_time_sec", "mean"),
            infeasible_events_mean=("infeasible_events", "mean"),
            executed_violation_events_mean=("executed_violation_events", "mean"),
        )
        .sort_values(["minimum_mean_route_survival", "method"])
    )


def paired_summary(
    missions: pd.DataFrame, rng: np.random.Generator, bootstrap: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for survival, group in missions.groupby("minimum_mean_route_survival", sort=True):
        for reference in (
            "dispatch_global",
            "fast_only",
            "exact_portfolio",
            "balanced_only",
            "safe_only",
        ):
            values = paired_gain_values(group, "learned_portfolio", reference)
            rows.append(
                {
                    "minimum_mean_route_survival": float(survival),
                    "comparison": f"learned_portfolio_vs_{reference}",
                    **distribution_summary(values, rng, bootstrap),
                }
            )
    return pd.DataFrame(rows)


def paired_gain_values(missions: pd.DataFrame, method: str, reference: str) -> np.ndarray:
    subset = missions[missions["method"].isin([method, reference])]
    pivot_completion = subset.pivot(index="mission_id", columns="method", values="completion_rate")
    if method not in pivot_completion or reference not in pivot_completion:
        return np.empty(0, dtype=float)
    complete = (pivot_completion[method] >= 1.0 - 1e-12) & (pivot_completion[reference] >= 1.0 - 1e-12)
    ids = pivot_completion.index[complete]
    pivot_length = subset[subset["mission_id"].isin(ids)].pivot(
        index="mission_id", columns="method", values="mission_true_length"
    )
    return (pivot_length[reference].astype(float) - pivot_length[method].astype(float)).to_numpy(dtype=float)


def mode_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in events.groupby("method", sort=True):
        counts = {mode: 0 for mode in ("fast", "balanced", "safe")}
        mixed = 0
        active = 0
        for raw in group["modes"].dropna().astype(str):
            modes = [mode for mode in raw.split(";") if mode]
            if not modes:
                continue
            active += 1
            mixed += int(len(set(modes)) >= 2)
            for mode in modes:
                counts[mode] += 1
        total = sum(counts.values())
        rows.append(
            {
                "method": method,
                "executed_routes": total,
                "fast_rate": counts["fast"] / total if total else math.nan,
                "balanced_rate": counts["balanced"] / total if total else math.nan,
                "safe_rate": counts["safe"] / total if total else math.nan,
                "mixed_profile_event_rate": mixed / active if active else math.nan,
            }
        )
    return pd.DataFrame(rows)


def runtime_summary(runtime_dir: Path, global_runtime_dir: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(runtime_dir / "lazy_rolling_missions.csv").query("method in ['learned_portfolio', 'fast_only']"),
        pd.read_csv(global_runtime_dir / "lazy_rolling_missions.csv").query("method == 'dispatch_global'"),
    ]
    runtime = pd.concat(frames, ignore_index=True)
    return (
        runtime.groupby("method", as_index=False)
        .agg(
            missions=("mission_id", "nunique"),
            prediction_runtime_sec_mean=("prediction_runtime_sec", "mean"),
            assignment_runtime_sec_mean=("assignment_runtime_sec", "mean"),
            decision_wall_time_sec_mean=("decision_wall_time_sec", "mean"),
            total_planner_calls_mean=("total_planner_calls", "mean"),
        )
        .sort_values("method")
    )


def distribution_summary(
    values: np.ndarray, rng: np.random.Generator, bootstrap: int
) -> dict[str, float | int]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"common_complete_n": 0}
    boot = np.empty(bootstrap, dtype=float)
    for index in range(bootstrap):
        sample = rng.integers(0, len(values), size=len(values))
        boot[index] = float(np.mean(values[sample]))
    return {
        "common_complete_n": int(len(values)),
        "gain_mean": float(np.mean(values)),
        "gain_median": float(np.median(values)),
        "gain_ci95_low": float(np.quantile(boot, 0.025)),
        "gain_ci95_high": float(np.quantile(boot, 0.975)),
        "win_rate": float(np.mean(values > 1e-6)),
        "tie_rate": float(np.mean(np.abs(values) <= 1e-6)),
        "loss_rate": float(np.mean(values < -1e-6)),
    }


if __name__ == "__main__":
    main()
