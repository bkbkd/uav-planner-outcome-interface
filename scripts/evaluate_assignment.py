from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_generator import (
    DatasetConfig,
    PlannerConfig,
    make_base_record,
    run_planner,
    sample_uniform_valid_point,
)
from src.envs.scenario import Scenario
from src.learning.bid_targets import baseline_bid as planner_baseline_bid
from src.learning.bid_targets import true_bid_from_planner_metrics
from src.utils.assignment import solve_linear_assignment


def sample_assignment_instance(
    scenario: Scenario,
    n_agents: int,
    n_tasks: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float]]]:
    agents = [(*sample_uniform_point(scenario, rng), 0.0) for _ in range(n_agents)]
    tasks = [sample_uniform_point(scenario, rng) for _ in range(n_tasks)]
    return agents, tasks


def sample_uniform_point(scenario: Scenario, rng: np.random.Generator) -> tuple[float, float]:
    return sample_uniform_valid_point(scenario, rng)


def evaluate_true_pair_costs(
    scenario: Scenario,
    metadata: dict,
    scenario_id: int,
    instance_id: int,
    agents: list[tuple[float, float, float]],
    tasks: list[tuple[float, float]],
    planner_config: PlannerConfig,
    base_seed: int,
) -> tuple[pd.DataFrame | None, dict[str, float]]:
    pending_records = []
    runtime_sec = 0.0
    dataset_config = DatasetConfig(planner=planner_config)
    for i, agent in enumerate(agents):
        for j, task in enumerate(tasks):
            pair_id = i * len(tasks) + j
            record = make_base_record(
                sample_id=pair_id,
                scenario_id=scenario_id,
                seed=base_seed + pair_id,
                start_goal_mode="assignment_pair",
                start=agent,
                goal=task,
                scenario=scenario,
                config=dataset_config,
            )
            record.update({"instance_id": instance_id, "agent_id": i, "task_id": j, "pair_id": pair_id})
            pending_records.append(record)

    records = []
    for record in sorted(pending_records, key=planner_failure_priority, reverse=True):
        pair_id = int(record["pair_id"])
        agent = agents[int(record["agent_id"])]
        task = tasks[int(record["task_id"])]
        result, elapsed = run_planner(scenario, agent, task, planner_config)
        runtime_sec += elapsed
        if result is None:
            return None, {"runtime_sec": runtime_sec}
        record.update(
            {
                "length": result.metrics["length"],
                "time": result.metrics["time"],
                "risk": result.metrics["risk"],
                "turning": result.metrics["turning"],
                "objective": result.metrics["objective"],
                "true_bid": true_bid_from_planner_metrics(result.metrics, metadata),
                "baseline_bid": float(planner_baseline_bid(pd.DataFrame([record]), metadata)[0]),
                "runtime_sec": elapsed,
                "feasible": 1,
            }
        )
        records.append(record)
    records.sort(key=lambda item: (int(item["agent_id"]), int(item["task_id"])))
    return pd.DataFrame(records), {"runtime_sec": runtime_sec}


def planner_failure_priority(record: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(record["straight_line_collision"]),
        -float(record["corridor_collision_free_count"]),
        float(record["euclidean_distance"]),
        float(record["straight_line_p90_threat"]),
        float(record["straight_line_max_threat"]),
        float(record["straight_line_risk"]),
    )


def matrix_from_column(df: pd.DataFrame, column: str, n_agents: int, n_tasks: int) -> np.ndarray:
    matrix = np.zeros((n_agents, n_tasks), dtype=float)
    for _, row in df.iterrows():
        matrix[int(row["agent_id"]), int(row["task_id"])] = float(row[column])
    return matrix


def solve_assignment(cost_matrix: np.ndarray) -> tuple[tuple[int, ...], float]:
    return solve_linear_assignment(cost_matrix)


def assignment_cost(cost_matrix: np.ndarray, assignment: tuple[int, ...]) -> float:
    return float(sum(cost_matrix[i, task] for i, task in enumerate(assignment)))


def assignment_to_string(assignment: tuple[int, ...]) -> str:
    return ";".join(f"{i}->{task}" for i, task in enumerate(assignment))


def validate_near_edge_policy(zero_threshold: float, oracle_threshold: float) -> None:
    if zero_threshold < 0.0 or oracle_threshold < 0.0:
        raise ValueError("Near-edge Euclidean thresholds must be non-negative.")
    if oracle_threshold > 0.0 and zero_threshold > oracle_threshold:
        raise ValueError("near-zero threshold must be no larger than near-oracle threshold.")


def apply_near_edge_policy(
    pair_df: pd.DataFrame,
    column: str = "learned_bid",
    zero_threshold: float = 0.0,
    oracle_threshold: float = 0.0,
    raw_column: str | None = None,
) -> None:
    if zero_threshold <= 0.0 and oracle_threshold <= 0.0:
        return
    validate_near_edge_policy(zero_threshold, oracle_threshold)
    if column not in pair_df.columns:
        raise ValueError(f"near-edge policy requires a {column} column.")
    if "true_bid" not in pair_df.columns or "euclidean_distance" not in pair_df.columns:
        raise ValueError("near-edge policy requires true_bid and euclidean_distance columns.")
    if raw_column and raw_column not in pair_df.columns:
        pair_df[raw_column] = pair_df[column]

    distance = pair_df["euclidean_distance"].to_numpy(dtype=float)
    bid = pair_df[column].to_numpy(dtype=float, copy=True)
    true_bid = pair_df["true_bid"].to_numpy(dtype=float)

    if oracle_threshold > 0.0:
        oracle_mask = distance <= oracle_threshold
        if zero_threshold > 0.0:
            oracle_mask &= distance > zero_threshold
        bid[oracle_mask] = true_bid[oracle_mask]
    if zero_threshold > 0.0:
        bid[distance <= zero_threshold] = 0.0

    pair_df[column] = bid


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
