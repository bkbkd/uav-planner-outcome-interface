from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


MODES = ("fast", "balanced", "safe")
CONSUMERS = ("sum_length", "makespan")
COMMITMENTS = ("edge_wise", "dispatch_global")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate scalable risk-budgeted profile commitment with exact MILP solves."
    )
    parser.add_argument("predictions", type=Path, help="Directory produced by export_scale_assignment_predictions.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-levels", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-rel-gap", type=float, default=1e-8)
    args = parser.parse_args()
    if any(not 0.0 <= value <= 1.0 for value in args.budget_levels):
        raise ValueError("budget levels must lie in [0, 1].")

    instances = pd.read_csv(args.predictions / "instances.csv")
    pairs = pd.read_csv(args.predictions / "portfolio_predictions.csv")
    validate_table(instances, pairs)
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    for instance in instances.itertuples(index=False):
        instance_id = int(instance.instance_id)
        n_agents = int(instance.n_agents)
        n_tasks = int(instance.n_tasks)
        frame = pairs[pairs["instance_id"].astype(int) == instance_id]
        matrices = build_matrices(frame, n_agents, n_tasks)

        for consumer in CONSUMERS:
            risk_floor = solve_assignment(
                matrices,
                length_source="true",
                risk_source="true",
                consumer="sum_risk",
                budget=None,
                allowed_modes=MODES,
                time_limit=args.time_limit,
                mip_rel_gap=args.mip_rel_gap,
            )
            unconstrained = solve_assignment(
                matrices,
                length_source="true",
                risk_source="true",
                consumer=consumer,
                budget=None,
                allowed_modes=MODES,
                time_limit=args.time_limit,
                mip_rel_gap=args.mip_rel_gap,
            )
            require_optimal(risk_floor, "risk endpoint")
            require_optimal(unconstrained, f"{consumer} endpoint")
            low = float(risk_floor["true_risk"])
            high = max(low, float(unconstrained["true_risk"]))

            for level in args.budget_levels:
                budget = low + float(level) * (high - low)
                for acquisition, length_source, risk_source in (
                    ("exact", "true", "true"),
                    ("learned", "pred", "pred"),
                ):
                    for commitment in COMMITMENTS:
                        result = solve_commitment(
                            matrices,
                            length_source=length_source,
                            risk_source=risk_source,
                            consumer=consumer,
                            budget=budget,
                            commitment=commitment,
                            time_limit=args.time_limit,
                            mip_rel_gap=args.mip_rel_gap,
                        )
                        row = {
                            "instance_id": instance_id,
                            "scenario_id": int(instance.scenario_id),
                            "n_agents": n_agents,
                            "n_tasks": n_tasks,
                            "consumer": consumer,
                            "budget_level": float(level),
                            "risk_floor": low,
                            "unconstrained_true_risk": high,
                            "risk_budget": budget,
                            "acquisition": acquisition,
                            "commitment": commitment,
                        }
                        row.update(result_row(result, budget))
                        rows.append(row)
                        if result is not None:
                            counts = {mode: result["modes"].count(mode) for mode in MODES}
                            mode_rows.append(
                                {
                                    **{key: row[key] for key in (
                                        "instance_id", "scenario_id", "consumer", "budget_level",
                                        "acquisition", "commitment",
                                    )},
                                    **{f"{mode}_edges": counts[mode] for mode in MODES},
                                    "active_modes": sum(value > 0 for value in counts.values()),
                                }
                            )

    result_frame = pd.DataFrame(rows)
    mode_frame = pd.DataFrame(mode_rows)
    result_frame.to_csv(args.output / "scale_assignment_results.csv", index=False)
    mode_frame.to_csv(args.output / "scale_assignment_mode_activation.csv", index=False)
    summary = summarize(result_frame, mode_frame)
    summary["protocol"] = {
        "budget_definition": "R_min + level * (R_unconstrained_objective - R_min)",
        "budget_levels": args.budget_levels,
        "consumers": list(CONSUMERS),
        "commitments": list(COMMITMENTS),
        "solver": "scipy.optimize.milp (HiGHS)",
        "time_limit_sec": args.time_limit,
        "mip_rel_gap": args.mip_rel_gap,
    }
    (args.output / "scale_assignment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def validate_table(instances: pd.DataFrame, pairs: pd.DataFrame) -> None:
    required = {
        "instance_id", "scenario_id", "agent_id", "task_id", "mode",
        "true_length", "true_risk", "pred_length", "pred_risk",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"prediction table missing columns: {sorted(missing)}")
    for instance in instances.itertuples(index=False):
        frame = pairs[pairs["instance_id"].astype(int) == int(instance.instance_id)]
        expected = int(instance.n_agents) * int(instance.n_tasks) * len(MODES)
        if len(frame) != expected:
            raise ValueError(f"instance {instance.instance_id} has {len(frame)} rows, expected {expected}")
        keys = frame[["agent_id", "task_id", "mode"]].drop_duplicates()
        if len(keys) != expected or set(frame["mode"]) != set(MODES):
            raise ValueError(f"instance {instance.instance_id} has incomplete or duplicate edge-profile keys")


def build_matrices(frame: pd.DataFrame, n_agents: int, n_tasks: int) -> dict[str, dict[str, np.ndarray]]:
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for source in ("true", "pred"):
        matrices[source] = {}
        for field in ("length", "risk"):
            array = np.zeros((n_agents, n_tasks, len(MODES)), dtype=float)
            for mode_index, mode in enumerate(MODES):
                mode_frame = frame[frame["mode"] == mode]
                array[
                    mode_frame["agent_id"].to_numpy(dtype=int),
                    mode_frame["task_id"].to_numpy(dtype=int),
                    mode_index,
                ] = mode_frame[f"{source}_{field}"].to_numpy(dtype=float)
            matrices[source][field] = array
    return matrices


def solve_commitment(
    matrices: dict[str, dict[str, np.ndarray]],
    *,
    length_source: str,
    risk_source: str,
    consumer: str,
    budget: float,
    commitment: str,
    time_limit: float,
    mip_rel_gap: float,
) -> dict[str, Any] | None:
    if commitment == "edge_wise":
        return solve_assignment(
            matrices,
            length_source=length_source,
            risk_source=risk_source,
            consumer=consumer,
            budget=budget,
            allowed_modes=MODES,
            time_limit=time_limit,
            mip_rel_gap=mip_rel_gap,
        )
    candidates = [
        solve_assignment(
            matrices,
            length_source=length_source,
            risk_source=risk_source,
            consumer=consumer,
            budget=budget,
            allowed_modes=(mode,),
            time_limit=time_limit,
            mip_rel_gap=mip_rel_gap,
        )
        for mode in MODES
    ]
    feasible = [item for item in candidates if item is not None]
    if not feasible:
        return None
    key = "pred_sum_length" if consumer == "sum_length" else "pred_makespan"
    if length_source == "true":
        key = "true_sum_length" if consumer == "sum_length" else "true_makespan"
    return min(feasible, key=lambda item: (float(item[key]), float(item[f"{risk_source}_risk"])))


def solve_assignment(
    matrices: dict[str, dict[str, np.ndarray]],
    *,
    length_source: str,
    risk_source: str,
    consumer: str,
    budget: float | None,
    allowed_modes: Iterable[str],
    time_limit: float,
    mip_rel_gap: float,
) -> dict[str, Any] | None:
    mode_indices = tuple(MODES.index(mode) for mode in allowed_modes)
    n_agents, n_tasks, _ = matrices["true"]["length"].shape
    variables = [(i, j, k) for i in range(n_agents) for j in range(n_tasks) for k in mode_indices]
    x_count = len(variables)
    has_t = consumer == "makespan"
    variable_count = x_count + int(has_t)
    t_index = x_count if has_t else None

    if consumer == "sum_risk":
        c = np.asarray([matrices[risk_source]["risk"][i, j, k] for i, j, k in variables], dtype=float)
    elif consumer == "sum_length":
        c = np.asarray([matrices[length_source]["length"][i, j, k] for i, j, k in variables], dtype=float)
    elif consumer == "makespan":
        c = np.zeros(variable_count, dtype=float)
        c[t_index] = 1.0
    else:
        raise ValueError(f"Unknown consumer: {consumer}")

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coefficients: Iterable[tuple[int, float]], lb: float, ub: float) -> None:
        row = len(lower)
        for column, value in coefficients:
            rows.append(row)
            cols.append(column)
            values.append(float(value))
        lower.append(float(lb))
        upper.append(float(ub))

    by_agent = [[] for _ in range(n_agents)]
    by_task = [[] for _ in range(n_tasks)]
    for index, (agent, task, _) in enumerate(variables):
        by_agent[agent].append((index, 1.0))
        by_task[task].append((index, 1.0))
    for coefficients in by_agent:
        add(coefficients, 1.0, 1.0)
    for coefficients in by_task:
        add(coefficients, 1.0, 1.0)

    if budget is not None:
        add(
            (
                (index, matrices[risk_source]["risk"][agent, task, mode])
                for index, (agent, task, mode) in enumerate(variables)
            ),
            -np.inf,
            float(budget),
        )
    if has_t:
        for agent in range(n_agents):
            coefficients = [
                (index, matrices[length_source]["length"][i, task, mode])
                for index, (i, task, mode) in enumerate(variables)
                if i == agent
            ]
            coefficients.append((t_index, -1.0))
            add(coefficients, -np.inf, 0.0)

    matrix = coo_matrix((values, (rows, cols)), shape=(len(lower), variable_count)).tocsr()
    bounds = Bounds(np.zeros(variable_count), np.full(variable_count, np.inf))
    integrality = np.zeros(variable_count, dtype=int)
    integrality[:x_count] = 1
    upper_bounds = np.ones(variable_count)
    if has_t:
        upper_bounds[t_index] = np.inf
    bounds = Bounds(np.zeros(variable_count), upper_bounds)

    tic = time.perf_counter()
    result = milp(
        c,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": float(time_limit), "mip_rel_gap": float(mip_rel_gap)},
    )
    elapsed = time.perf_counter() - tic
    if result.status == 2:
        return None
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed with status={result.status}: {result.message}")

    selected_indices = np.flatnonzero(result.x[:x_count] > 0.5)
    if len(selected_indices) != n_agents:
        raise RuntimeError(f"MILP selected {len(selected_indices)} edges for {n_agents} agents")
    assignment = [-1] * n_agents
    modes = [""] * n_agents
    for index in selected_indices:
        agent, task, mode = variables[int(index)]
        assignment[agent] = task
        modes[agent] = MODES[mode]

    out: dict[str, Any] = {
        "assignment": tuple(assignment),
        "modes": tuple(modes),
        "solver_sec": elapsed,
        "mip_gap": float(getattr(result, "mip_gap", math.nan)),
    }
    for source in ("true", "pred"):
        lengths = [
            matrices[source]["length"][agent, task, MODES.index(mode)]
            for agent, (task, mode) in enumerate(zip(assignment, modes))
        ]
        risks = [
            matrices[source]["risk"][agent, task, MODES.index(mode)]
            for agent, (task, mode) in enumerate(zip(assignment, modes))
        ]
        out[f"{source}_sum_length"] = float(np.sum(lengths))
        out[f"{source}_makespan"] = float(np.max(lengths))
        out[f"{source}_risk"] = float(np.sum(risks))
    return out


def require_optimal(result: dict[str, Any] | None, name: str) -> None:
    if result is None:
        raise RuntimeError(f"Unexpected infeasible {name}")


def result_row(result: dict[str, Any] | None, budget: float) -> dict[str, Any]:
    if result is None:
        return {
            "predicted_feasible": 0,
            "true_sum_length": math.nan,
            "true_makespan": math.nan,
            "true_risk": math.nan,
            "pred_sum_length": math.nan,
            "pred_makespan": math.nan,
            "pred_risk": math.nan,
            "true_violation": math.nan,
            "true_risk_slack": math.nan,
            "assignment": "",
            "modes": "",
            "solver_sec": math.nan,
            "mip_gap": math.nan,
        }
    return {
        "predicted_feasible": 1,
        "true_sum_length": result["true_sum_length"],
        "true_makespan": result["true_makespan"],
        "true_risk": result["true_risk"],
        "pred_sum_length": result["pred_sum_length"],
        "pred_makespan": result["pred_makespan"],
        "pred_risk": result["pred_risk"],
        "true_violation": int(result["true_risk"] > budget + 1e-9),
        "true_risk_slack": budget - result["true_risk"],
        "assignment": ";".join(f"{agent}->{task}" for agent, task in enumerate(result["assignment"])),
        "modes": ";".join(result["modes"]),
        "solver_sec": result["solver_sec"],
        "mip_gap": result["mip_gap"],
    }


def summarize(results: pd.DataFrame, modes: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    group_keys = ["consumer", "budget_level", "acquisition", "commitment"]
    for key, group in results.groupby(group_keys, sort=True):
        feasible = group[group["predicted_feasible"].astype(int) == 1]
        label = "/".join(str(item) for item in key)
        item: dict[str, Any] = {
            "instances": int(len(group)),
            "predicted_feasible_rate": float(group["predicted_feasible"].mean()),
        }
        if len(feasible):
            item.update(
                {
                    "true_sum_length_mean": float(feasible["true_sum_length"].mean()),
                    "true_makespan_mean": float(feasible["true_makespan"].mean()),
                    "true_risk_mean": float(feasible["true_risk"].mean()),
                    "true_violation_rate": float(feasible["true_violation"].mean()),
                    "solver_sec_mean": float(feasible["solver_sec"].mean()),
                    "solver_sec_p95": float(feasible["solver_sec"].quantile(0.95)),
                    "mip_gap_max": float(feasible["mip_gap"].fillna(0.0).max()),
                }
            )
        summary[label] = item

    paired = {}
    for (consumer, level, acquisition), group in results.groupby(
        ["consumer", "budget_level", "acquisition"], sort=True
    ):
        edge = group[group["commitment"] == "edge_wise"].set_index("instance_id")
        global_profile = group[group["commitment"] == "dispatch_global"].set_index("instance_id")
        common_ids = edge.index.intersection(global_profile.index)
        edge = edge.loc[common_ids]
        global_profile = global_profile.loc[common_ids]
        mask = (
            edge["predicted_feasible"].astype(int).eq(1)
            & global_profile["predicted_feasible"].astype(int).eq(1)
            & edge["true_violation"].astype(float).eq(0.0)
            & global_profile["true_violation"].astype(float).eq(0.0)
        )
        objective = "true_sum_length" if consumer == "sum_length" else "true_makespan"
        gain = global_profile.loc[mask, objective].astype(float) - edge.loc[mask, objective].astype(float)
        tolerance = 1e-6
        label = f"{consumer}/{level}/{acquisition}"
        paired[label] = {
            "all_instances": int(len(group) // 2),
            "common_nonviolating_instances": int(mask.sum()),
            "common_nonviolating_rate": float(mask.mean()),
            "edge_wise_gain_mean": float(gain.mean()) if len(gain) else math.nan,
            "edge_wise_gain_median": float(gain.median()) if len(gain) else math.nan,
            "edge_wise_win_rate": float((gain > tolerance).mean()) if len(gain) else math.nan,
            "tie_rate": float((gain.abs() <= tolerance).mean()) if len(gain) else math.nan,
            "edge_wise_loss_rate": float((gain < -tolerance).mean()) if len(gain) else math.nan,
        }
    summary["paired_edge_wise_vs_dispatch_global"] = paired

    if len(modes):
        edge_modes = modes[modes["commitment"] == "edge_wise"]
        activation = {}
        for key, group in edge_modes.groupby(["consumer", "budget_level", "acquisition"], sort=True):
            label = "/".join(str(item) for item in key)
            total = max(1, int(group[[f"{mode}_edges" for mode in MODES]].to_numpy().sum()))
            activation[label] = {
                "instances": int(len(group)),
                "using_at_least_two_modes": float((group["active_modes"] >= 2).mean()),
                "using_all_three_modes": float((group["active_modes"] == 3).mean()),
                **{f"{mode}_rate": float(group[f"{mode}_edges"].sum() / total) for mode in MODES},
            }
        summary["edge_wise_activation"] = activation
    return summary


if __name__ == "__main__":
    main()
