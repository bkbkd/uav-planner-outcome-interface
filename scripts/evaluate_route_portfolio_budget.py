from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import bisect
import sys
from functools import lru_cache
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.evaluate_assignment import assignment_to_string


PROFILE_MODES = ("fast", "balanced", "safe")
PROFILE_SUBSETS = (
    ("fast", "balanced"),
    ("fast", "safe"),
    ("balanced", "safe"),
)


def profile_frontier_specs(n_agents: int) -> dict[str, tuple[tuple[str, ...], ...]]:
    specs = {
        "portfolio": tuple(itertools.product(PROFILE_MODES, repeat=n_agents)),
        "global_profile": tuple((mode,) * n_agents for mode in PROFILE_MODES),
    }
    for subset in PROFILE_SUBSETS:
        specs[f"portfolio_{'_'.join(subset)}"] = tuple(itertools.product(subset, repeat=n_agents))
    for mode in PROFILE_MODES:
        specs[f"{mode}_only"] = ((mode,) * n_agents,)
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare risk-budget assignment with a multi-route portfolio against single-mode route candidates."
    )
    parser.add_argument("--fast", type=Path, required=True, help="Benchmark directory for the fast route mode.")
    parser.add_argument("--balanced", type=Path, required=True, help="Benchmark directory for the balanced route mode.")
    parser.add_argument("--safe", type=Path, required=True, help="Benchmark directory for the safe route mode.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--budget-quantiles",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    )
    args = parser.parse_args()
    for quantile in args.budget_quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"budget quantile must be in [0, 1], got {quantile}")

    args.output.mkdir(parents=True, exist_ok=True)
    benchmarks = {
        "fast": load_benchmark(args.fast),
        "balanced": load_benchmark(args.balanced),
        "safe": load_benchmark(args.safe),
    }
    validate_same_instances(benchmarks)

    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    instances = benchmarks["balanced"]["instances"]
    for instance in instances.itertuples(index=False):
        instance_id = int(instance.instance_id)
        n_agents = int(instance.n_agents)
        n_tasks = int(instance.n_tasks)
        mode_matrices = {
            mode: {
                "length": matrix_from_pairs(data["pairs"], instance_id, "length", n_agents, n_tasks),
                "risk": matrix_from_pairs(data["pairs"], instance_id, "risk", n_agents, n_tasks),
            }
            for mode, data in benchmarks.items()
        }
        assignments = tuple(itertools.permutations(range(n_tasks), n_agents))
        frontiers = {
            method: build_candidate_frontier(mode_matrices, assignments, route_choices)
            for method, route_choices in profile_frontier_specs(n_agents).items()
        }
        portfolio_frontier = frontiers["portfolio"]
        portfolio_risks = portfolio_frontier["risks"]
        for quantile in args.budget_quantiles:
            budget = float(np.quantile(portfolio_risks, quantile))
            portfolio = solve_frontier(portfolio_frontier, budget)
            if portfolio is None:
                continue
            methods = {
                method: (portfolio if method == "portfolio" else solve_frontier(frontier, budget))
                for method, frontier in frontiers.items()
            }

            oracle_length = portfolio["length"]
            oracle_risk = portfolio["risk"]
            for method, result in methods.items():
                row = {
                    "instance_id": instance_id,
                    "scenario_id": int(instance.scenario_id),
                    "budget_quantile": float(quantile),
                    "risk_budget": budget,
                    "method": method,
                    "oracle_portfolio_length": oracle_length,
                    "oracle_portfolio_risk": oracle_risk,
                    "oracle_assignment": assignment_to_string(portfolio["assignment"]),
                    "oracle_modes": ";".join(portfolio["modes"]),
                }
                if result is None:
                    row.update(
                        {
                            "feasible": 0,
                            "selected_length": math.nan,
                            "selected_risk": math.nan,
                            "length_regret_to_portfolio": math.nan,
                            "risk_slack": math.nan,
                            "assignment": "",
                            "modes": "",
                        }
                    )
                else:
                    row.update(
                        {
                            "feasible": 1,
                            "selected_length": result["length"],
                            "selected_risk": result["risk"],
                            "length_regret_to_portfolio": result["length"] - oracle_length,
                            "risk_slack": budget - result["risk"],
                            "assignment": assignment_to_string(result["assignment"]),
                            "modes": ";".join(result["modes"]),
                        }
                    )
                    if method == "portfolio":
                        counts = Counter(result["modes"])
                        detail_rows.append(
                            {
                                "instance_id": instance_id,
                                "scenario_id": int(instance.scenario_id),
                                "budget_quantile": float(quantile),
                                "risk_budget": budget,
                                "fast_edges": counts.get("fast", 0),
                                "balanced_edges": counts.get("balanced", 0),
                                "safe_edges": counts.get("safe", 0),
                                "selected_length": result["length"],
                                "selected_risk": result["risk"],
                                "assignment": assignment_to_string(result["assignment"]),
                                "modes": ";".join(result["modes"]),
                            }
                        )
                rows.append(row)

    write_rows(args.output / "budget_assignment_results.csv", rows)
    write_rows(args.output / "portfolio_mode_selection.csv", detail_rows)
    summary = summarize(rows, detail_rows)
    compact_rows = compact_budget_summary(rows, detail_rows)
    write_rows(args.output / "budget_assignment_compact.csv", compact_rows)
    summary["compact_budget_summary"] = compact_rows
    summary["args"] = {
        "fast": str(args.fast),
        "balanced": str(args.balanced),
        "safe": str(args.safe),
        "budget_quantiles": args.budget_quantiles,
    }
    (args.output / "budget_assignment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved: {args.output}")


def load_benchmark(path: Path) -> dict[str, pd.DataFrame]:
    return {
        "instances": sort_instances(pd.read_csv(path / "instances.csv")),
        "pairs": sort_pairs(pd.read_csv(path / "pairs.csv")),
    }


def sort_instances(df: pd.DataFrame) -> pd.DataFrame:
    if "instance_id" not in df.columns:
        return df
    out = df.copy()
    out["_instance_id_sort"] = out["instance_id"].astype(int)
    out = out.sort_values("_instance_id_sort").drop(columns=["_instance_id_sort"])
    return out.reset_index(drop=True)


def sort_pairs(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["instance_id", "agent_id", "task_id"]
    if not set(keys).issubset(df.columns):
        return df
    out = df.copy()
    for key in keys:
        out[f"_{key}_sort"] = out[key].astype(int)
    sort_keys = [f"_{key}_sort" for key in keys]
    out = out.sort_values(sort_keys).drop(columns=sort_keys)
    return out.reset_index(drop=True)


def validate_same_instances(benchmarks: dict[str, dict[str, pd.DataFrame]]) -> None:
    reference_instances = benchmarks["balanced"]["instances"][["instance_id", "scenario_id", "agents_json", "tasks_json"]].reset_index(drop=True)
    reference_pair_keys = benchmarks["balanced"]["pairs"][["instance_id", "agent_id", "task_id"]].reset_index(drop=True)
    reference_pair_xy = benchmarks["balanced"]["pairs"][["start_x", "start_y", "goal_x", "goal_y"]].reset_index(drop=True)
    for mode, data in benchmarks.items():
        instances = data["instances"][["instance_id", "scenario_id", "agents_json", "tasks_json"]].reset_index(drop=True)
        pair_keys = data["pairs"][["instance_id", "agent_id", "task_id"]].reset_index(drop=True)
        pair_xy = data["pairs"][["start_x", "start_y", "goal_x", "goal_y"]].reset_index(drop=True)
        if not reference_instances.equals(instances):
            raise ValueError(f"{mode} instances do not match balanced benchmark")
        if not reference_pair_keys.equals(pair_keys):
            raise ValueError(f"{mode} pair keys do not match balanced benchmark")
        if not np.allclose(reference_pair_xy.to_numpy(dtype=float), pair_xy.to_numpy(dtype=float), rtol=0.0, atol=1e-6):
            raise ValueError(f"{mode} pair geometry does not match balanced benchmark")


def matrix_from_pairs(pairs: pd.DataFrame, instance_id: int, column: str, n_agents: int, n_tasks: int) -> np.ndarray:
    df = pairs[pairs["instance_id"].astype(int) == instance_id]
    matrix = np.zeros((n_agents, n_tasks), dtype=float)
    for row in df.itertuples(index=False):
        matrix[int(row.agent_id), int(row.task_id)] = float(getattr(row, column))
    return matrix


def solve_portfolio(
    mode_matrices: dict[str, dict[str, np.ndarray]],
    assignments: tuple[tuple[int, ...], ...],
    route_choices: tuple[tuple[str, ...], ...],
    budget: float,
) -> dict[str, Any] | None:
    return solve_frontier(build_candidate_frontier(mode_matrices, assignments, route_choices), budget)


def build_candidate_frontier(
    mode_matrices: dict[str, dict[str, np.ndarray]],
    assignments: tuple[tuple[int, ...], ...],
    route_choices: tuple[tuple[str, ...], ...],
) -> dict[str, Any]:
    if not assignments or not route_choices:
        return {
            "risks": np.asarray([], dtype=float),
            "lengths": np.asarray([], dtype=float),
            "assignment_indices": np.zeros((0, 0), dtype=int),
            "mode_indices": np.zeros((0, 0), dtype=int),
            "mode_names": tuple(mode_matrices.keys()),
            "prefix_best_idx": np.asarray([], dtype=int),
        }

    mode_names = tuple(mode_matrices.keys())
    assignment_idx, mode_idx, agent_idx = decision_index_arrays(assignments, route_choices, mode_names)
    total = assignment_idx.shape[0]

    length_stack = np.stack([mode_matrices[mode]["length"] for mode in mode_names], axis=0)
    risk_stack = np.stack([mode_matrices[mode]["risk"] for mode in mode_names], axis=0)
    lengths = length_stack[mode_idx, agent_idx, assignment_idx].sum(axis=1)
    risks = risk_stack[mode_idx, agent_idx, assignment_idx].sum(axis=1)

    order = np.lexsort((lengths, risks))
    risks = risks[order]
    lengths = lengths[order]
    assignment_idx = assignment_idx[order]
    mode_idx = mode_idx[order]

    previous_best = np.empty_like(lengths)
    previous_best[0] = np.inf
    if total > 1:
        previous_best[1:] = np.minimum.accumulate(lengths[:-1])
    is_new_best = lengths < previous_best - 1e-9
    prefix_best_idx = np.maximum.accumulate(np.where(is_new_best, np.arange(total, dtype=int), 0))
    return {
        "risks": risks,
        "lengths": lengths,
        "assignment_indices": assignment_idx,
        "mode_indices": mode_idx,
        "mode_names": mode_names,
        "prefix_best_idx": prefix_best_idx,
    }


@lru_cache(maxsize=32)
def decision_index_arrays(
    assignments: tuple[tuple[int, ...], ...],
    route_choices: tuple[tuple[str, ...], ...],
    mode_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode_to_idx = {mode: idx for idx, mode in enumerate(mode_names)}
    assignments_arr = np.asarray(assignments, dtype=int)
    route_choices_arr = np.asarray([[mode_to_idx[mode] for mode in modes] for modes in route_choices], dtype=int)
    n_assignments, n_agents = assignments_arr.shape
    n_choices = route_choices_arr.shape[0]
    total = n_assignments * n_choices
    assignment_idx = np.repeat(assignments_arr, n_choices, axis=0)
    mode_idx = np.tile(route_choices_arr, (n_assignments, 1))
    agent_idx = np.broadcast_to(np.arange(n_agents, dtype=int), (total, n_agents))
    return assignment_idx, mode_idx, agent_idx


def solve_frontier(frontier: dict[str, Any], budget: float) -> dict[str, Any] | None:
    risks = frontier["risks"]
    idx = bisect.bisect_right(risks, budget + 1e-9) - 1
    if idx < 0:
        return None
    best_idx = int(frontier["prefix_best_idx"][idx])
    risk = float(frontier["risks"][best_idx])
    length = float(frontier["lengths"][best_idx])
    assignment = tuple(int(x) for x in frontier["assignment_indices"][best_idx])
    mode_names = frontier["mode_names"]
    modes = tuple(mode_names[int(x)] for x in frontier["mode_indices"][best_idx])
    return {"assignment": assignment, "modes": modes, "length": length, "risk": risk}


def route_assignment_cost(
    mode_matrices: dict[str, dict[str, np.ndarray]],
    assignment: tuple[int, ...],
    modes: tuple[str, ...],
    column: str,
) -> float:
    return float(sum(mode_matrices[mode][column][agent, task] for agent, (task, mode) in enumerate(zip(assignment, modes))))


def summarize(rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for (method, quantile), group in df.groupby(["method", "budget_quantile"], sort=True):
        feasible = group[group["feasible"].astype(int) == 1]
        item: dict[str, Any] = {
            "instances": int(len(group)),
            "feasible_rate": float(group["feasible"].astype(float).mean()),
        }
        if len(feasible):
            item.update(
                {
                    "selected_length_mean": float(feasible["selected_length"].astype(float).mean()),
                    "selected_risk_mean": float(feasible["selected_risk"].astype(float).mean()),
                    "length_regret_to_portfolio_mean": float(feasible["length_regret_to_portfolio"].astype(float).mean()),
                    "risk_slack_mean": float(feasible["risk_slack"].astype(float).mean()),
                }
            )
        summary[f"{method}_q{float(quantile):.2f}"] = item

    detail = pd.DataFrame(detail_rows)
    if len(detail):
        mode_summary = {}
        for quantile, group in detail.groupby("budget_quantile", sort=True):
            total_edges = max(
                1,
                int(group[["fast_edges", "balanced_edges", "safe_edges"]].astype(float).to_numpy().sum()),
            )
            mode_summary[f"q{float(quantile):.2f}"] = {
                "fast_rate": float(group["fast_edges"].astype(float).sum() / total_edges),
                "balanced_rate": float(group["balanced_edges"].astype(float).sum() / total_edges),
                "safe_rate": float(group["safe_edges"].astype(float).sum() / total_edges),
            }
        summary["portfolio_mode_mix"] = mode_summary
    return summary


def compact_budget_summary(rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    detail = pd.DataFrame(detail_rows)
    out: list[dict[str, Any]] = []
    single_methods = ["fast_only", "balanced_only", "safe_only"]
    for quantile in sorted(df["budget_quantile"].astype(float).unique()):
        qdf = df[df["budget_quantile"].astype(float) == quantile]
        portfolio = method_stats(qdf, "portfolio")
        balanced = paired_method_stats(qdf, "balanced_only")
        best_single = per_instance_best_single_stats(qdf, single_methods)
        qdetail = detail[detail["budget_quantile"].astype(float) == quantile]
        total_edges = max(
            1,
            int(qdetail[["fast_edges", "balanced_edges", "safe_edges"]].astype(float).to_numpy().sum()),
        )
        row = {
            "budget_quantile": float(quantile),
            "portfolio_length_mean": portfolio["selected_length_mean"],
            "portfolio_risk_mean": portfolio["selected_risk_mean"],
            "portfolio_risk_slack_mean": portfolio["risk_slack_mean"],
            "portfolio_violation_rate": 0.0,
            "balanced_length_mean": balanced["selected_length_mean"],
            "balanced_feasible_rate": balanced["feasible_rate"],
            "portfolio_vs_balanced_length_reduction": balanced["paired_length_reduction"],
            "portfolio_vs_balanced_length_reduction_pct": (
                balanced["paired_length_reduction"]
                / max(balanced["selected_length_mean"], 1e-9)
            ),
            "best_single_method": best_single["method"],
            "best_single_length_mean": best_single["selected_length_mean"],
            "best_single_feasible_rate": best_single["feasible_rate"],
            "portfolio_vs_best_single_length_reduction": best_single["paired_length_reduction"],
            "portfolio_vs_best_single_length_reduction_pct": (
                best_single["paired_length_reduction"]
                / max(best_single["selected_length_mean"], 1e-9)
            ),
            "portfolio_fast_rate": float(qdetail["fast_edges"].astype(float).sum() / total_edges) if len(qdetail) else math.nan,
            "portfolio_balanced_rate": float(qdetail["balanced_edges"].astype(float).sum() / total_edges) if len(qdetail) else math.nan,
            "portfolio_safe_rate": float(qdetail["safe_edges"].astype(float).sum() / total_edges) if len(qdetail) else math.nan,
        }
        out.append(row)
    return out


def method_stats(qdf: pd.DataFrame, method: str) -> dict[str, Any]:
    group = qdf[qdf["method"] == method]
    feasible = group[group["feasible"].astype(int) == 1]
    return {
        "method": method,
        "instances": int(len(group)),
        "feasible_rate": float(group["feasible"].astype(float).mean()) if len(group) else 0.0,
        "selected_length_mean": float(feasible["selected_length"].astype(float).mean()) if len(feasible) else math.inf,
        "selected_risk_mean": float(feasible["selected_risk"].astype(float).mean()) if len(feasible) else math.nan,
        "risk_slack_mean": float(feasible["risk_slack"].astype(float).mean()) if len(feasible) else math.nan,
    }


def paired_method_stats(qdf: pd.DataFrame, method: str) -> dict[str, Any]:
    portfolio = qdf[qdf["method"] == "portfolio"].set_index("instance_id")
    group = qdf[qdf["method"] == method].set_index("instance_id")
    feasible = group[group["feasible"].astype(int) == 1]
    common = portfolio.index.intersection(feasible.index)
    selected_length = feasible.loc[common, "selected_length"].astype(float)
    portfolio_length = portfolio.loc[common, "selected_length"].astype(float)
    return {
        "method": method,
        "instances": int(len(group)),
        "feasible_rate": float(len(common) / len(group)) if len(group) else 0.0,
        "selected_length_mean": float(selected_length.mean()) if len(common) else math.inf,
        "paired_length_reduction": float((selected_length - portfolio_length).mean()) if len(common) else math.nan,
    }


def per_instance_best_single_stats(qdf: pd.DataFrame, methods: list[str]) -> dict[str, Any]:
    portfolio = qdf[qdf["method"] == "portfolio"].set_index("instance_id")
    singles = qdf[(qdf["method"].isin(methods)) & (qdf["feasible"].astype(int) == 1)].copy()
    if singles.empty:
        return {
            "method": "per_instance_best_feasible",
            "instances": int(len(portfolio)),
            "feasible_rate": 0.0,
            "selected_length_mean": math.inf,
            "paired_length_reduction": math.nan,
        }
    best_idx = singles.groupby("instance_id")["selected_length"].idxmin()
    best = singles.loc[best_idx].set_index("instance_id")
    common = portfolio.index.intersection(best.index)
    selected_length = best.loc[common, "selected_length"].astype(float)
    portfolio_length = portfolio.loc[common, "selected_length"].astype(float)
    return {
        "method": "per_instance_best_feasible",
        "instances": int(len(portfolio)),
        "feasible_rate": float(len(common) / len(portfolio)) if len(portfolio) else 0.0,
        "selected_length_mean": float(selected_length.mean()) if len(common) else math.inf,
        "paired_length_reduction": float((selected_length - portfolio_length).mean()) if len(common) else math.nan,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
