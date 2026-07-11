from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Pool held-out scale-assignment results with paired bootstrap intervals.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    frames = []
    mode_frames = []
    for path in args.inputs:
        pool = path.name
        frame = pd.read_csv(path / "scale_assignment_results.csv")
        frame.insert(0, "pool", pool)
        frame["case_id"] = frame["pool"].astype(str) + ":" + frame["instance_id"].astype(str)
        frames.append(frame)
        modes = pd.read_csv(path / "scale_assignment_mode_activation.csv")
        modes.insert(0, "pool", pool)
        modes["case_id"] = modes["pool"].astype(str) + ":" + modes["instance_id"].astype(str)
        mode_frames.append(modes)

    results = pd.concat(frames, ignore_index=True)
    modes = pd.concat(mode_frames, ignore_index=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    for consumer in sorted(results["consumer"].unique()):
        objective = "true_sum_length" if consumer == "sum_length" else "true_makespan"
        for level in sorted(results["budget_level"].astype(float).unique()):
            group = results[
                (results["consumer"] == consumer)
                & np.isclose(results["budget_level"].astype(float), level)
            ]
            exact_edge = select(group, "exact", "edge_wise")
            exact_global = select(group, "exact", "dispatch_global")
            learned_edge = select(group, "learned", "edge_wise")
            learned_global = select(group, "learned", "dispatch_global")

            exact_gain = paired_difference(exact_global, exact_edge, objective)
            learned_mask = valid(learned_edge) & valid(learned_global)
            learned_gain = paired_difference(learned_global, learned_edge, objective, learned_mask)
            fidelity_mask = valid(learned_edge)
            learned_gap = paired_difference(learned_edge, exact_edge, objective, fidelity_mask)
            exact_objective = exact_edge.loc[learned_gap.index, objective].astype(float)
            relative_gap = learned_gap / exact_objective.replace(0.0, np.nan)
            exact_gain_ci = bootstrap_mean_ci(exact_gain, args.bootstrap, rng)
            learned_gain_ci = bootstrap_mean_ci(learned_gain, args.bootstrap, rng)

            activation = modes[
                (modes["consumer"] == consumer)
                & np.isclose(modes["budget_level"].astype(float), level)
                & (modes["acquisition"] == "learned")
                & (modes["commitment"] == "edge_wise")
            ].set_index("case_id")
            activation = activation.loc[activation.index.intersection(learned_edge.index[fidelity_mask])]

            row = {
                "consumer": consumer,
                "budget_level": float(level),
                "instances": int(len(exact_edge)),
                "exact_edge_wise_gain_mean": float(exact_gain.mean()),
                "exact_edge_wise_gain_ci95_low": exact_gain_ci[0],
                "exact_edge_wise_gain_ci95_high": exact_gain_ci[1],
                "exact_edge_wise_win_rate": float((exact_gain > 1e-6).mean()),
                "learned_common_nonviolating_rate": float(learned_mask.mean()),
                "learned_edge_wise_gain_mean": mean_or_nan(learned_gain),
                "learned_edge_wise_gain_ci95_low": learned_gain_ci[0],
                "learned_edge_wise_gain_ci95_high": learned_gain_ci[1],
                "learned_edge_wise_win_rate": rate_or_nan(learned_gain > 1e-6),
                "learned_edge_wise_loss_rate": rate_or_nan(learned_gain < -1e-6),
                "learned_edge_predicted_feasible_rate": float(
                    learned_edge["predicted_feasible"].astype(int).eq(1).mean()
                ),
                "learned_edge_true_violation_rate_given_proposal": float(
                    learned_edge.loc[
                        learned_edge["predicted_feasible"].astype(int).eq(1), "true_violation"
                    ].astype(float).mean()
                ),
                "learned_to_exact_objective_gap_mean": mean_or_nan(learned_gap),
                "learned_to_exact_relative_gap_mean": mean_or_nan(relative_gap),
                "learned_using_at_least_two_modes": float((activation["active_modes"] >= 2).mean()),
                "learned_using_all_three_modes": float((activation["active_modes"] == 3).mean()),
                "learned_solver_sec_mean": float(learned_edge["solver_sec"].dropna().mean()),
                "learned_solver_sec_p95": float(learned_edge["solver_sec"].dropna().quantile(0.95)),
            }
            rows.append(row)

    summary = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "scale_assignment_pooled_summary.csv", index=False)
    payload = {
        "test_pools": [path.name for path in args.inputs],
        "instances_per_pool": int(results.groupby("pool")["case_id"].nunique().iloc[0]),
        "total_instances": int(results["case_id"].nunique()),
        "bootstrap_resamples": int(args.bootstrap),
        "bootstrap_seed": int(args.seed),
        "rows": summary.to_dict("records"),
    }
    (args.output / "scale_assignment_pooled_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


def select(frame: pd.DataFrame, acquisition: str, commitment: str) -> pd.DataFrame:
    out = frame[
        (frame["acquisition"] == acquisition) & (frame["commitment"] == commitment)
    ].copy()
    if out["case_id"].duplicated().any():
        raise ValueError(f"duplicate cases for {acquisition}/{commitment}")
    return out.set_index("case_id").sort_index()


def valid(frame: pd.DataFrame) -> pd.Series:
    return frame["predicted_feasible"].astype(int).eq(1) & frame["true_violation"].astype(float).eq(0.0)


def paired_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str,
    mask: pd.Series | None = None,
) -> pd.Series:
    common = left.index.intersection(right.index)
    left = left.loc[common]
    right = right.loc[common]
    if mask is None:
        mask = pd.Series(True, index=common)
    else:
        mask = mask.reindex(common).fillna(False)
    return left.loc[mask, column].astype(float) - right.loc[mask, column].astype(float)


def bootstrap_mean_ci(values: pd.Series, count: int, rng: np.random.Generator) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) == 0:
        return math.nan, math.nan
    means = np.empty(count, dtype=float)
    for index in range(count):
        means[index] = rng.choice(array, size=len(array), replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def mean_or_nan(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else math.nan


def rate_or_nan(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else math.nan


if __name__ == "__main__":
    main()
