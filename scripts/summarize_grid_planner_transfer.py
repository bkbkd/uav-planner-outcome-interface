from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


MODES = ("fast", "balanced", "safe")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize held-out exact commitment transfer for the grid planner.")
    parser.add_argument("--benchmark", nargs=2, action="append", metavar=("POOL", "DIR"), required=True)
    parser.add_argument("--evaluation", nargs=2, action="append", metavar=("POOL", "DIR"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    benchmark_paths = {name: Path(path) for name, path in args.benchmark}
    evaluation_paths = {name: Path(path) for name, path in args.evaluation}
    if set(benchmark_paths) != set(evaluation_paths):
        raise ValueError("benchmark and evaluation pool names must match")

    edge_frames = []
    instance_frames = []
    result_frames = []
    for pool in sorted(benchmark_paths):
        root = benchmark_paths[pool]
        for mode in MODES:
            pairs = pd.read_csv(root / mode / "pairs.csv")
            pairs.insert(0, "pool", pool)
            pairs["mode"] = mode
            edge_frames.append(pairs)
            instances = pd.read_csv(root / mode / "instances.csv")
            instances.insert(0, "pool", pool)
            instances["mode"] = mode
            instance_frames.append(instances)
        results = pd.read_csv(evaluation_paths[pool] / "budget_assignment_results.csv")
        results.insert(0, "pool", pool)
        results["case_id"] = results["pool"] + ":" + results["instance_id"].astype(str)
        result_frames.append(results)

    edges = pd.concat(edge_frames, ignore_index=True)
    instances = pd.concat(instance_frames, ignore_index=True)
    results = pd.concat(result_frames, ignore_index=True)
    validate(edges, results)

    key = ["pool", "instance_id", "agent_id", "task_id"]
    wide = edges.pivot(index=key, columns="mode", values=["length", "risk"])
    monotone = (
        (wide["length"]["fast"] <= wide["length"]["balanced"] + 1e-7)
        & (wide["length"]["balanced"] <= wide["length"]["safe"] + 1e-7)
        & (wide["risk"]["fast"] >= wide["risk"]["balanced"] - 1e-7)
        & (wide["risk"]["balanced"] >= wide["risk"]["safe"] - 1e-7)
    )
    effective_counts = []
    for _, group in edges.groupby(key, sort=False):
        effective_counts.append(len(group[["length", "risk"]].round(8).drop_duplicates()))
    effective = pd.Series(effective_counts)

    rng = np.random.default_rng(args.seed)
    budget_rows = []
    for quantile in sorted(results["budget_quantile"].astype(float).unique()):
        frame = results[np.isclose(results["budget_quantile"].astype(float), quantile)]
        edge = frame[frame["method"] == "portfolio"].set_index("case_id").sort_index()
        global_profile = frame[frame["method"] == "global_profile"].set_index("case_id").sort_index()
        gain = global_profile["selected_length"].astype(float) - edge["selected_length"].astype(float)
        low, high = bootstrap_mean_ci(gain.to_numpy(dtype=float), args.bootstrap, rng)
        budget_rows.append(
            {
                "budget_quantile": float(quantile),
                "instances": int(len(gain)),
                "edge_wise_gain_mean": float(gain.mean()),
                "edge_wise_gain_median": float(gain.median()),
                "edge_wise_gain_ci95_low": low,
                "edge_wise_gain_ci95_high": high,
                "positive_rate": float((gain > 1e-7).mean()),
                "gain_over_1m_rate": float((gain > 1.0).mean()),
            }
        )

    pair_runtime = edges.groupby(["pool", "mode", "instance_id"], sort=False)["runtime_sec"].sum()
    graph_runtime = instances["graph_build_runtime_sec"].astype(float)
    shifts = {
        "balanced_minus_fast_length_mean": float((wide["length"]["balanced"] - wide["length"]["fast"]).mean()),
        "balanced_minus_fast_risk_mean": float((wide["risk"]["balanced"] - wide["risk"]["fast"]).mean()),
        "safe_minus_balanced_length_mean": float((wide["length"]["safe"] - wide["length"]["balanced"]).mean()),
        "safe_minus_balanced_risk_mean": float((wide["risk"]["safe"] - wide["risk"]["balanced"]).mean()),
    }
    summary = {
        "planner_family": "holonomic_8_connected_risk_aware_grid_dijkstra",
        "test_pools": sorted(benchmark_paths),
        "held_out_instances": int(results["case_id"].nunique()),
        "held_out_edges": int(len(wide)),
        "monotone_profile_fraction": float(monotone.mean()),
        "effective_profile_distribution": {
            str(int(value)): float((effective == value).mean()) for value in sorted(effective.unique())
        },
        "mean_profile_shifts": shifts,
        "runtime": {
            "graph_build_sec_per_map_profile_mean": float(graph_runtime.mean()),
            "graph_build_sec_per_map_profile_p95": float(graph_runtime.quantile(0.95)),
            "query_sec_per_instance_profile_mean": float(pair_runtime.mean()),
            "query_sec_per_edge_mean": float(edges["runtime_sec"].astype(float).mean()),
        },
        "commitment": budget_rows,
        "bootstrap_resamples": int(args.bootstrap),
        "bootstrap_seed": int(args.seed),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(budget_rows).to_csv(args.output / "grid_transfer_commitment_summary.csv", index=False)
    (args.output / "grid_transfer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def validate(edges: pd.DataFrame, results: pd.DataFrame) -> None:
    expected_modes = set(MODES)
    if set(edges["mode"]) != expected_modes:
        raise ValueError("incomplete profile modes")
    keys = ["pool", "instance_id", "agent_id", "task_id", "mode"]
    if edges.duplicated(keys).any():
        raise ValueError("duplicate edge-profile rows")
    if not results[results["method"] == "portfolio"]["feasible"].astype(int).eq(1).all():
        raise ValueError("unexpected infeasible exact portfolio")


def bootstrap_mean_ci(
    values: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if not len(values):
        return math.nan, math.nan
    means = np.empty(count, dtype=float)
    for index in range(count):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


if __name__ == "__main__":
    main()
