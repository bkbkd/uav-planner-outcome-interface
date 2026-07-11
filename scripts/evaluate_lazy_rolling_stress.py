from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.evaluate_assignment import sample_assignment_instance
from scripts.evaluate_assignment_benchmark import load_model, predict_costs, validate_model_bid_weights
from scripts.interface_protocol import validation_risk_buffers
from src.data.dataset_generator import DatasetConfig, PlannerConfig, make_base_record, run_planner
from src.experiment_config import (
    CURRENT_NEAR_ORACLE_EUCLIDEAN_THRESHOLD,
    CURRENT_NEAR_ZERO_EUCLIDEAN_THRESHOLD,
    CURRENT_PLANNER,
)
from src.learning.bid_targets import baseline_bid as planner_baseline_bid
from src.learning.features import scenario_from_metadata


MODES = ("fast", "balanced", "safe")
DEFAULT_EVAL_METHODS = ("learned_portfolio", "fast_only", "balanced_only", "safe_only")
ALL_EVAL_METHODS = ("exact_portfolio", "dispatch_global", *DEFAULT_EVAL_METHODS)
MODE_BETA = {"fast": 150.0, "balanced": 650.0, "safe": 1500.0}
_WORKER: dict[str, Any] = {}


@dataclass(frozen=True)
class Candidate:
    agent: int
    task: int
    mode: str
    length: float
    risk: float
    objective: float
    key: tuple[Any, ...]
    distance: float


@dataclass(frozen=True)
class TrueRoute:
    length: float
    risk: float
    runtime_sec: float
    success: bool
    end_x: float
    end_y: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Lazy rolling stress test for the multi-profile route interface. "
            "Candidate route outcomes are predicted for all pending edges; the planner is called only "
            "for selected routes and cached selected-route verification."
        )
    )
    parser.add_argument("--scenario-pool", type=Path, required=True)
    parser.add_argument("--fast-model", type=Path, required=True)
    parser.add_argument("--balanced-model", type=Path, required=True)
    parser.add_argument("--safe-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--missions", type=int, default=20)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--mission-seed", type=int, required=True)
    parser.add_argument("--wave-size", type=int, default=5)
    parser.add_argument("--arrival-interval", type=float, default=5.0)
    parser.add_argument(
        "--minimum-mean-route-survivals",
        nargs="+",
        type=float,
        default=[0.80],
        help=(
            "Externally specified minimum geometric-mean survival across committed routes. "
            "For m routes, the additive-risk budget is -m*log(s)."
        ),
    )
    parser.add_argument("--risk-buffer-quantile", type=float, default=0.75)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_EVAL_METHODS,
        default=list(DEFAULT_EVAL_METHODS),
        help="Methods to evaluate. exact_portfolio uses true candidate outcomes before allocation.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--image-cache-dir",
        type=Path,
        default=None,
        help="Optional persistent image cache. Leave unset for measured final runs.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if any(survival <= 0.0 or survival > 1.0 for survival in args.minimum_mean_route_survivals):
        parser.error("--minimum-mean-route-survivals values must lie in (0, 1]")

    args.output.mkdir(parents=True, exist_ok=True)
    source_metadata = json.loads((args.scenario_pool / "metadata.json").read_text(encoding="utf-8"))
    scenario_ids = [int(item["scenario_id"]) for item in source_metadata["scenarios"]]
    if len(scenario_ids) < int(args.missions):
        parser.error("The rolling protocol requires one distinct scenario per mission.")
    mission_specs = build_mission_specs(args)
    mission_path = args.output / "lazy_rolling_missions.csv"
    event_path = args.output / "lazy_rolling_events.csv"
    summary_path = args.output / "lazy_rolling_summary.csv"
    mission_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    completed: set[tuple[int, int, int]] = set()
    methods = tuple(str(method) for method in args.methods)
    if args.resume and mission_path.exists():
        old_missions = pd.read_csv(mission_path)
        old_events = pd.read_csv(event_path) if event_path.exists() else pd.DataFrame()
        mission_rows = old_missions.to_dict("records")
        event_rows = old_events.to_dict("records")
        completed = completed_specs(old_missions, args.minimum_mean_route_survivals, methods)
        print(f"[resume] loaded {len(completed)}/{len(mission_specs)} completed mission specs", flush=True)
    jobs = [(idx, spec) for idx, spec in enumerate(mission_specs) if spec_key(spec) not in completed]
    worker_init = (
        source_metadata,
        scenario_ids,
        {
            "fast": str(args.fast_model),
            "balanced": str(args.balanced_model),
            "safe": str(args.safe_model),
        },
        {
            "agents": int(args.agents),
            "tasks": int(args.tasks),
            "wave_size": int(args.wave_size),
            "arrival_interval": float(args.arrival_interval),
            "minimum_mean_route_survivals": [float(x) for x in args.minimum_mean_route_survivals],
            "risk_buffer_quantile": float(args.risk_buffer_quantile),
            "batch_size": int(args.batch_size),
            "image_cache_dir": None if args.image_cache_dir is None else str(args.image_cache_dir),
            "methods": list(methods),
        },
    )
    expected_rows_per_spec = len(args.minimum_mean_route_survivals) * len(methods)
    if int(args.workers) > 1:
        with Pool(processes=int(args.workers), initializer=init_worker, initargs=worker_init) as pool:
            for idx, missions, events in pool.imap_unordered(run_mission_job, jobs, chunksize=1):
                mission_rows.extend(missions)
                event_rows.extend(events)
                write_outputs(args, mission_rows, event_rows, mission_path, event_path, summary_path)
                print(f"[{len(mission_rows) // expected_rows_per_spec}/{len(mission_specs)}] mission complete", flush=True)
    else:
        init_worker(*worker_init)
        for job in jobs:
            _idx, missions, events = run_mission_job(job)
            mission_rows.extend(missions)
            event_rows.extend(events)
            write_outputs(args, mission_rows, event_rows, mission_path, event_path, summary_path)
            print(f"[{len(mission_rows) // expected_rows_per_spec}/{len(mission_specs)}] mission complete", flush=True)

    summary_df = write_outputs(args, mission_rows, event_rows, mission_path, event_path, summary_path)
    metadata = {
        "scenario_pool": str(args.scenario_pool),
        "missions": int(len(mission_specs)),
        "mission_seed": int(args.mission_seed),
        "agents": int(args.agents),
        "tasks": int(args.tasks),
        "mode_beta": MODE_BETA,
        "wave_size": int(args.wave_size),
        "arrival_interval": float(args.arrival_interval),
        "risk_budget_source": "external_minimum_geometric_mean_route_survival",
        "minimum_mean_route_survivals": [float(x) for x in args.minimum_mean_route_survivals],
        "risk_buffer_quantile": float(args.risk_buffer_quantile),
        "methods": list(methods),
        "workers": int(args.workers),
    }
    (args.output / "lazy_rolling_summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary_df.to_dict("records")}, indent=2),
        encoding="utf-8",
    )
    print(summary_df.to_string(index=False))
    print(f"saved: {args.output}")


def build_mission_specs(args: argparse.Namespace) -> list[dict[str, int]]:
    return [
        {"mission_id": mission_id, "seed": int(args.mission_seed), "local_mission_id": mission_id}
        for mission_id in range(int(args.missions))
    ]


def spec_key(spec: dict[str, int]) -> tuple[int, int, int]:
    return (int(spec["mission_id"]), int(spec["seed"]), int(spec["local_mission_id"]))


def completed_specs(
    missions: pd.DataFrame,
    survivals: list[float],
    methods: tuple[str, ...],
) -> set[tuple[int, int, int]]:
    if missions.empty:
        return set()
    required = {(method, round(float(survival), 12)) for method in methods for survival in survivals}
    completed = set()
    for key, group in missions.groupby(["mission_id", "seed", "local_mission_id"], dropna=False):
        observed = {
            (str(row.method), round(float(row.minimum_mean_route_survival), 12))
            for row in group.itertuples(index=False)
        }
        if required.issubset(observed):
            completed.add(tuple(int(item) for item in key))
    return completed


def write_outputs(
    args: argparse.Namespace,
    mission_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    mission_path: Path,
    event_path: Path,
    summary_path: Path,
) -> pd.DataFrame:
    mission_df = pd.DataFrame(mission_rows)
    event_df = pd.DataFrame(event_rows)
    summary_df = summarize(mission_df) if len(mission_df) else pd.DataFrame()
    mission_df.to_csv(mission_path, index=False)
    event_df.to_csv(event_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    progress = {
        "completed_mission_rows": int(len(mission_df)),
        "completed_event_rows": int(len(event_df)),
        "updated_at_unix": time.time(),
    }
    (args.output / "rolling_progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
    return summary_df


def init_worker(
    source_metadata: dict[str, Any],
    scenario_ids: list[int],
    model_paths: dict[str, str],
    config: dict[str, Any],
) -> None:
    mode_metadata = {mode: metadata_with_beta(source_metadata, MODE_BETA[mode]) for mode in MODES}
    loaded_models: dict[Path, dict[str, Any]] = {}
    models: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        path = Path(model_paths[mode]).resolve()
        if path not in loaded_models:
            loaded_models[path] = load_model(path)
            loaded_models[path]["residual_scale"] = 1.0
        validate_model_bid_weights(loaded_models[path], mode_metadata[mode], source=str(path))
        models[mode] = loaded_models[path]
    buffers = {
        mode: risk_buffer_for_model(
            Path(model_paths[mode]),
            float(config["risk_buffer_quantile"]),
            MODE_BETA[mode],
            models[mode]["model_type"],
        )
        for mode in MODES
    }
    _WORKER.clear()
    _WORKER.update(
        {
            "source_metadata": source_metadata,
            "scenario_ids": scenario_ids,
            "mode_metadata": mode_metadata,
            "models": models,
            "buffers": buffers,
            "config": config,
        }
    )


def run_mission_job(job: tuple[int, dict[str, int]]) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    idx, spec = job
    source_metadata = _WORKER["source_metadata"]
    scenario_ids = _WORKER["scenario_ids"]
    config = _WORKER["config"]
    mission_id = int(spec["mission_id"])
    seed = int(spec["seed"])
    local_id = int(spec["local_mission_id"])
    scenario_id = int(scenario_ids[mission_id])
    scenario = scenario_from_metadata(source_metadata, scenario_id)
    mission_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(local_id,)))
    agents, tasks = sample_assignment_instance(
        scenario,
        int(config["agents"]),
        int(config["tasks"]),
        mission_rng,
    )
    mission_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for minimum_mean_route_survival in config["minimum_mean_route_survivals"]:
        for method in config["methods"]:
            route_cache: dict[tuple[Any, ...], TrueRoute] = {}
            exact_cache: set[tuple[Any, ...]] = set()
            mission, events = simulate_mission(
                mission_id=mission_id,
                seed=seed,
                local_mission_id=local_id,
                scenario_id=scenario_id,
                scenario=scenario,
                agents=list(agents),
                tasks=tasks,
                method=method,
                minimum_mean_route_survival=float(minimum_mean_route_survival),
                mode_metadata=_WORKER["mode_metadata"],
                models=_WORKER["models"],
                buffers=_WORKER["buffers"],
                route_cache=route_cache,
                exact_cache=exact_cache,
                config=config,
            )
            mission_rows.append(mission)
            event_rows.extend(events)
    return idx, mission_rows, event_rows


def metadata_with_beta(source_metadata: dict[str, Any], beta: float) -> dict[str, Any]:
    metadata = deepcopy(source_metadata)
    planner = dict(metadata["config"]["planner"])
    planner["beta"] = float(beta)
    metadata["config"]["planner"] = planner
    return metadata


def risk_buffer_for_model(
    model_path: Path,
    quantile: float,
    profile_beta: float,
    model_type: str,
) -> float:
    buffers = validation_risk_buffers(
        model_path / "predictions.csv",
        [float(quantile)],
        profile_beta=profile_beta if model_type == "profile_onehot_cnn" else None,
    )
    return float(next(iter(buffers.values()))) if buffers else 0.0


def simulate_mission(
    *,
    mission_id: int,
    seed: int,
    local_mission_id: int,
    scenario_id: int,
    scenario: Any,
    agents: list[tuple[float, float, float]],
    tasks: list[tuple[float, float]],
    method: str,
    minimum_mean_route_survival: float,
    mode_metadata: dict[str, dict[str, Any]],
    models: dict[str, dict[str, Any]],
    buffers: dict[str, float],
    route_cache: dict[tuple[Any, ...], TrueRoute],
    exact_cache: set[tuple[Any, ...]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    speed = float(mode_metadata["balanced"]["config"]["planner"].get("speed", 28.0))
    arrival_times = {
        task: math.floor(task / max(int(config["wave_size"]), 1)) * float(config["arrival_interval"])
        for task in range(len(tasks))
    }
    released: set[int] = set()
    pending: list[int] = []
    source_task: list[int | None] = [None for _ in agents]
    active_task: list[int | None] = [None for _ in agents]
    active_endpoint: list[tuple[float, float, float] | None] = [None for _ in agents]
    finish_time = [math.inf for _ in agents]
    task_flow_times: list[float] = []
    completed = 0
    event_idx = 0
    current_time = 0.0
    total_length = 0.0
    total_risk = 0.0
    proposal_verification_calls = 0
    near_candidate_planner_calls = 0
    near_candidate_planner_runtime = 0.0
    planner_runtime = 0.0
    cache_hits = 0
    executable_commitments = 0
    prediction_runtime = 0.0
    assignment_runtime = 0.0
    decision_wall_times: list[float] = []
    pre_verification_violations = 0
    infeasible_events = 0
    executed_violation_events = 0
    exhaustive_call_estimate = 0
    event_rows: list[dict[str, Any]] = []
    scenario_cache = {scenario_id: scenario}
    base_image_cache: dict[tuple, np.ndarray] = {}

    while completed < len(tasks):
        next_arrival = min((time for task, time in arrival_times.items() if task not in released), default=math.inf)
        next_finish = min(finish_time)
        current_time = min(next_arrival, next_finish)
        if not math.isfinite(current_time):
            break

        completed_now = []
        for agent in range(len(agents)):
            if active_task[agent] is not None and finish_time[agent] <= current_time + 1e-9:
                task = int(active_task[agent])
                active_task[agent] = None
                source_task[agent] = task
                if active_endpoint[agent] is None:
                    raise RuntimeError("Completed route is missing its planner endpoint.")
                agents[agent] = active_endpoint[agent]
                active_endpoint[agent] = None
                finish_time[agent] = math.inf
                completed += 1
                completed_now.append(task)
                task_flow_times.append(current_time - arrival_times[task])

        arrived = [task for task, time in arrival_times.items() if task not in released and time <= current_time + 1e-9]
        if arrived:
            pending.extend(sorted(arrived))
            released.update(arrived)
        idle_agents = [idx for idx, task in enumerate(active_task) if task is None]
        if not pending or not idle_agents:
            event_idx += 1
            continue

        decision_tic = time.perf_counter()
        prediction_tic = time.perf_counter()
        candidates = predict_event_candidates(
            mission_id=mission_id,
            scenario_id=scenario_id,
            scenario=scenario,
            agents=agents,
            tasks=tasks,
            source_task=source_task,
            idle_agents=idle_agents,
            pending=pending,
            mode_metadata=mode_metadata,
            models=models,
            buffers=buffers,
            method=method,
            cache_namespace=f"{method}/survival_{minimum_mean_route_survival:.4f}",
            config=config,
            scenario_cache=scenario_cache,
            base_image_cache=base_image_cache,
        )
        event_prediction_runtime = time.perf_counter() - prediction_tic
        prediction_runtime += event_prediction_runtime
        if method == "exact_portfolio":
            candidates, calls, hits, runtime_sec = apply_exact_candidate_policy(
                candidates,
                scenario,
                agents,
                tasks,
                mode_metadata,
                route_cache,
            )
        else:
            candidates, calls, hits, runtime_sec = apply_near_candidate_policy(
                candidates,
                scenario,
                agents,
                tasks,
                mode_metadata,
                route_cache,
            )
        event_near_calls = calls
        event_near_runtime = runtime_sec
        near_candidate_planner_calls += calls
        near_candidate_planner_runtime += runtime_sec
        cache_hits += hits
        planner_runtime += runtime_sec
        exhaustive_call_estimate += count_new_candidate_keys(candidates, exact_cache)
        routes_to_commit = min(len(idle_agents), len(pending))
        risk_budget_per_route = -math.log(float(minimum_mean_route_survival))
        budget = risk_budget_per_route * routes_to_commit
        solve_tic = time.perf_counter()
        choice = solve_event(
            candidates,
            idle_agents,
            pending,
            budget,
            method,
        )
        event_assignment_runtime = time.perf_counter() - solve_tic
        assignment_runtime += event_assignment_runtime
        if not choice:
            infeasible_events += 1
            decision_wall_times.append(time.perf_counter() - decision_tic)
            break
        proposal_predicted_feasible = sum(c.risk for c in choice) <= budget + 1e-9

        true_routes, calls, hits, runtime_sec = verify_choice(
            choice,
            scenario,
            agents,
            tasks,
            mode_metadata,
            route_cache,
        )
        proposal_verification_calls += calls
        cache_hits += hits
        planner_runtime += runtime_sec
        true_risk = sum(route.risk for route in true_routes)
        if (not proposal_predicted_feasible) or true_risk > budget + 1e-9:
            pre_verification_violations += 1
        if not all(route.success for route in true_routes):
            infeasible_events += 1
            decision_wall_times.append(time.perf_counter() - decision_tic)
            break

        true_length = sum(route.length for route in true_routes)
        executable_commitments += len(choice)
        executed_violation_events += int(true_risk > budget + 1e-9)
        total_length += true_length
        total_risk += true_risk
        for candidate, route in zip(choice, true_routes):
            active_task[candidate.agent] = candidate.task
            active_endpoint[candidate.agent] = (route.end_x, route.end_y, 0.0)
            finish_time[candidate.agent] = current_time + route.length / max(speed, 1e-9)
            if candidate.task in pending:
                pending.remove(candidate.task)
        decision_wall_time = time.perf_counter() - decision_tic
        decision_wall_times.append(decision_wall_time)

        event_rows.append(
            {
                "mission_id": mission_id,
                "seed": seed,
                "local_mission_id": local_mission_id,
                "scenario_id": scenario_id,
                "method": method,
                "minimum_mean_route_survival": minimum_mean_route_survival,
                "risk_budget_per_route": risk_budget_per_route,
                "event": event_idx,
                "event_time": current_time,
                "arrived": len(arrived),
                "completed_at_event": len(completed_now),
                "pending_before": len(pending) + len(choice),
                "idle_agents": len(idle_agents),
                "assigned": len(choice),
                "candidate_edges": len(candidates),
                "risk_budget": budget,
                "true_length": true_length,
                "true_risk": true_risk,
                "violation": int(true_risk > budget + 1e-9),
                "proposal_predicted_feasible": int(proposal_predicted_feasible),
                "prediction_runtime_sec": event_prediction_runtime,
                "near_candidate_planner_calls": event_near_calls,
                "near_candidate_planner_runtime_sec": event_near_runtime,
                "assignment_runtime_sec": event_assignment_runtime,
                "decision_wall_time_sec": decision_wall_time,
                "modes": ";".join(c.mode for c in choice),
                "assignment": ";".join(f"{c.agent}->{c.task}" for c in choice),
            }
        )
        event_idx += 1

    completion_time = max([current_time] + [t for t in finish_time if math.isfinite(t)])
    mission = {
        "mission_id": mission_id,
        "seed": seed,
        "local_mission_id": local_mission_id,
        "scenario_id": scenario_id,
        "method": method,
        "minimum_mean_route_survival": minimum_mean_route_survival,
        "risk_budget_per_route": -math.log(float(minimum_mean_route_survival)),
        "completed_tasks": completed,
        "events": event_idx,
        "mission_true_length": total_length,
        "mission_true_risk": total_risk,
        "mean_flow_time": float(np.mean(task_flow_times)) if task_flow_times else math.nan,
        "p95_flow_time": float(np.quantile(task_flow_times, 0.95)) if task_flow_times else math.nan,
        "mission_completion_time": completion_time,
        "completion_rate": completed / max(len(tasks), 1),
        "executable_commitments": executable_commitments,
        "proposal_verification_planner_calls": proposal_verification_calls,
        "near_candidate_planner_calls": near_candidate_planner_calls,
        "total_planner_calls": near_candidate_planner_calls + proposal_verification_calls,
        "near_candidate_planner_runtime_sec": near_candidate_planner_runtime,
        "planner_runtime_sec": planner_runtime,
        "selected_cache_hits": cache_hits,
        "prediction_runtime_sec": prediction_runtime,
        "assignment_runtime_sec": assignment_runtime,
        "decision_wall_time_sec": float(np.sum(decision_wall_times)),
        "decision_wall_time_p95_sec": float(np.quantile(decision_wall_times, 0.95)) if decision_wall_times else math.nan,
        "pre_verification_violation_events": pre_verification_violations,
        "infeasible_events": infeasible_events,
        "executed_violation_events": executed_violation_events,
        "exhaustive_candidate_planner_call_estimate": exhaustive_call_estimate,
        "planner_call_reduction_vs_exhaustive": exhaustive_call_estimate
        / max(near_candidate_planner_calls + proposal_verification_calls, 1),
    }
    return mission, event_rows


def predict_event_candidates(
    *,
    mission_id: int,
    scenario_id: int,
    scenario: Any,
    agents: list[tuple[float, float, float]],
    tasks: list[tuple[float, float]],
    source_task: list[int | None],
    idle_agents: list[int],
    pending: list[int],
    mode_metadata: dict[str, dict[str, Any]],
    models: dict[str, dict[str, Any]],
    buffers: dict[str, float],
    method: str,
    cache_namespace: str,
    config: dict[str, Any],
    scenario_cache: dict[int, Any],
    base_image_cache: dict[tuple, np.ndarray],
) -> list[Candidate]:
    rows_by_mode: dict[str, pd.DataFrame] = {}
    prediction_modes = (
        MODES
        if method in {"learned_portfolio", "dispatch_global", "exact_portfolio"}
        else (method.removesuffix("_only"),)
    )
    for mode in prediction_modes:
        planner_config = PlannerConfig(**mode_metadata[mode]["config"]["planner"])
        dataset_config = DatasetConfig(planner=planner_config)
        rows = []
        for agent in idle_agents:
            for task in pending:
                start_xy = agents[agent][:2]
                goal_xy = tasks[task]
                edge_seed = mission_id * 1_000_000 + agent * 1000 + task
                record = make_base_record(
                    sample_id=mission_id * 100000 + agent * 1000 + task,
                    scenario_id=scenario_id,
                    seed=edge_seed,
                    start_goal_mode="lazy_rolling_candidate",
                    start=(float(start_xy[0]), float(start_xy[1]), 0.0),
                    goal=(float(goal_xy[0]), float(goal_xy[1])),
                    scenario=scenario,
                    config=dataset_config,
                )
                record.update(
                    {
                        "instance_id": mission_id,
                        "edge_type": "agent_task" if source_task[agent] is None else "task_task",
                        "agent_id": agent,
                        "from_task_id": "" if source_task[agent] is None else int(source_task[agent]),
                        "task_id": task,
                        "pair_id": agent * len(tasks) + task,
                    }
                )
                record["baseline_bid"] = float(planner_baseline_bid(pd.DataFrame([record]), mode_metadata[mode])[0])
                rows.append(record)
        frame = pd.DataFrame(rows)
        frame["pred_length"] = 0.0
        frame["pred_risk"] = 0.0
        far_mask = frame["euclidean_distance"].astype(float) > CURRENT_NEAR_ORACLE_EUCLIDEAN_THRESHOLD
        if method != "exact_portfolio" and bool(far_mask.any()):
            predictions = predict_costs(
                frame.loc[far_mask].reset_index(drop=True),
                mode_metadata[mode],
                models[mode],
                batch_size=int(config["batch_size"]),
                image_cache_dir=(
                    None
                    if config["image_cache_dir"] is None
                    else Path(str(config["image_cache_dir"])) / cache_namespace
                ),
                scenario_cache=scenario_cache,
                base_image_cache=base_image_cache,
            )
            frame.loc[far_mask, "pred_length"] = np.maximum(0.0, predictions["length"].astype(float))
            frame.loc[far_mask, "pred_risk"] = np.maximum(0.0, predictions["risk"].astype(float)) + buffers[mode]
        rows_by_mode[mode] = frame

    candidates = []
    for mode, frame in rows_by_mode.items():
        for row in frame.itertuples(index=False):
            length = float(row.pred_length)
            risk = float(row.pred_risk)
            candidates.append(
                Candidate(
                    agent=int(row.agent_id),
                    task=int(row.task_id),
                    mode=mode,
                    length=length,
                    risk=risk,
                    objective=length,
                    key=route_key(scenario_id, mode, int(row.task_id), agents[int(row.agent_id)]),
                    distance=float(row.euclidean_distance),
                )
            )
    return candidates


def route_key(scenario_id: int, mode: str, task_id: int, agent: tuple[float, float, float]) -> tuple[Any, ...]:
    return (
        int(scenario_id),
        str(mode),
        int(task_id),
        round(float(agent[0]), 4),
        round(float(agent[1]), 4),
    )


def apply_near_candidate_policy(
    candidates: list[Candidate],
    scenario: Any,
    agents: list[tuple[float, float, float]],
    tasks: list[tuple[float, float]],
    mode_metadata: dict[str, dict[str, Any]],
    route_cache: dict[tuple[Any, ...], TrueRoute],
) -> tuple[list[Candidate], int, int, float]:
    exact_candidates = [
        candidate
        for candidate in candidates
        if CURRENT_NEAR_ZERO_EUCLIDEAN_THRESHOLD < candidate.distance <= CURRENT_NEAR_ORACLE_EUCLIDEAN_THRESHOLD
    ]
    exact_routes, calls, hits, runtime_sec = verify_choice(
        exact_candidates,
        scenario,
        agents,
        tasks,
        mode_metadata,
        route_cache,
    )
    exact_values = {
        candidate.key: replace(candidate, length=route.length, risk=route.risk, objective=route.length)
        for candidate, route in zip(exact_candidates, exact_routes)
    }
    updated = []
    for candidate in candidates:
        if candidate.distance <= CURRENT_NEAR_ZERO_EUCLIDEAN_THRESHOLD:
            route_cache.setdefault(
                candidate.key,
                TrueRoute(
                    length=0.0,
                    risk=0.0,
                    runtime_sec=0.0,
                    success=True,
                    end_x=float(agents[candidate.agent][0]),
                    end_y=float(agents[candidate.agent][1]),
                ),
            )
            updated.append(replace(candidate, length=0.0, risk=0.0, objective=0.0))
        else:
            updated.append(exact_values.get(candidate.key, candidate))
    return updated, calls, hits, runtime_sec


def apply_exact_candidate_policy(
    candidates: list[Candidate],
    scenario: Any,
    agents: list[tuple[float, float, float]],
    tasks: list[tuple[float, float]],
    mode_metadata: dict[str, dict[str, Any]],
    route_cache: dict[tuple[Any, ...], TrueRoute],
) -> tuple[list[Candidate], int, int, float]:
    exact_candidates = [
        candidate
        for candidate in candidates
        if candidate.distance > CURRENT_NEAR_ZERO_EUCLIDEAN_THRESHOLD
    ]
    exact_routes, calls, hits, runtime_sec = verify_choice(
        exact_candidates,
        scenario,
        agents,
        tasks,
        mode_metadata,
        route_cache,
    )
    exact_values = {
        candidate.key: replace(candidate, length=route.length, risk=route.risk, objective=route.length)
        for candidate, route in zip(exact_candidates, exact_routes)
    }
    updated = []
    for candidate in candidates:
        if candidate.distance <= CURRENT_NEAR_ZERO_EUCLIDEAN_THRESHOLD:
            route_cache.setdefault(
                candidate.key,
                TrueRoute(
                    length=0.0,
                    risk=0.0,
                    runtime_sec=0.0,
                    success=True,
                    end_x=float(agents[candidate.agent][0]),
                    end_y=float(agents[candidate.agent][1]),
                ),
            )
            updated.append(replace(candidate, length=0.0, risk=0.0, objective=0.0))
        else:
            updated.append(exact_values[candidate.key])
    return updated, calls, hits, runtime_sec


def solve_event(
    candidates: list[Candidate],
    idle_agents: list[int],
    pending: list[int],
    budget: float,
    method: str,
) -> list[Candidate]:
    routes_to_commit = min(len(idle_agents), len(pending))
    if routes_to_commit <= 0:
        return []
    if method == "dispatch_global":
        choices = [
            solve_event(candidates, idle_agents, pending, budget, f"{mode}_only")
            for mode in MODES
        ]
        feasible = [choice for choice in choices if choice]
        if not feasible:
            return []
        return min(
            feasible,
            key=lambda choice: (
                sum(candidate.length for candidate in choice),
                sum(candidate.risk for candidate in choice),
            ),
        )
    by_agent: dict[int, list[Candidate]] = {}
    for agent in idle_agents:
        agent_candidates = [c for c in candidates if c.agent == agent]
        if method == "balanced_only":
            agent_candidates = [c for c in agent_candidates if c.mode == "balanced"]
        elif method.endswith("_only"):
            selected_mode = method.removesuffix("_only")
            agent_candidates = [c for c in agent_candidates if c.mode == selected_mode]
        by_agent[agent] = agent_candidates

    states: dict[int, list[tuple[float, float, list[Candidate]]]] = {0: [(0.0, 0.0, [])]}
    for agent in idle_agents:
        next_states: dict[int, list[tuple[float, float, list[Candidate]]]] = {
            mask: list(frontier) for mask, frontier in states.items()
        }
        for mask, frontier in states.items():
            used_tasks = mask_to_tasks(mask)
            for length, risk, choice in frontier:
                for candidate in by_agent.get(agent, []):
                    if candidate.task in used_tasks:
                        continue
                    new_mask = mask | (1 << candidate.task)
                    new_item = (length + candidate.length, risk + candidate.risk, choice + [candidate])
                    next_states.setdefault(new_mask, []).append(new_item)
        states = {mask: prune_frontier(frontier) for mask, frontier in next_states.items()}
        if not states:
            break

    best: tuple[float, float, list[Candidate]] | None = None
    for frontier in states.values():
        for item in frontier:
            length, risk, _choice = item
            if len(_choice) != routes_to_commit:
                continue
            if risk > budget + 1e-9:
                continue
            if best is None or length < best[0] - 1e-9 or (abs(length - best[0]) <= 1e-9 and risk < best[1]):
                best = item
    if best is None:
        return []
    return best[2]


def mask_to_tasks(mask: int) -> set[int]:
    tasks = set()
    bit = 0
    while mask:
        if mask & 1:
            tasks.add(bit)
        mask >>= 1
        bit += 1
    return tasks


def prune_frontier(frontier: list[tuple[float, float, list[Candidate]]]) -> list[tuple[float, float, list[Candidate]]]:
    frontier = sorted(frontier, key=lambda item: (item[1], item[0]))
    kept: list[tuple[float, float, list[Candidate]]] = []
    best_length = math.inf
    for item in frontier:
        length, _risk, _choice = item
        if length < best_length - 1e-9:
            kept.append(item)
            best_length = length
    return kept


def count_new_candidate_keys(candidates: list[Candidate], seen: set[tuple[Any, ...]]) -> int:
    new_keys = {
        candidate.key
        for candidate in candidates
        if candidate.distance > CURRENT_NEAR_ZERO_EUCLIDEAN_THRESHOLD and candidate.key not in seen
    }
    seen.update(new_keys)
    return len(new_keys)


def verify_choice(
    choice: list[Candidate],
    scenario: Any,
    agents: list[tuple[float, float, float]],
    tasks: list[tuple[float, float]],
    mode_metadata: dict[str, dict[str, Any]],
    route_cache: dict[tuple[Any, ...], TrueRoute],
) -> tuple[list[TrueRoute], int, int, float]:
    routes: list[TrueRoute] = []
    calls = 0
    hits = 0
    runtime_sec = 0.0
    for candidate in choice:
        key = candidate.key
        if key in route_cache:
            routes.append(route_cache[key])
            hits += 1
            continue
        route = run_selected_route(
            candidate,
            scenario,
            agents,
            tasks,
            mode_metadata[candidate.mode],
        )
        route_cache[key] = route
        routes.append(route)
        calls += 1
        runtime_sec += route.runtime_sec
    return routes, calls, hits, runtime_sec


def run_selected_route(
    candidate: Candidate,
    scenario: Any,
    agents: list[tuple[float, float, float]],
    tasks: list[tuple[float, float]],
    metadata: dict[str, Any],
) -> TrueRoute:
    start_xy = agents[candidate.agent][:2]
    goal_xy = tasks[candidate.task]
    start = (float(start_xy[0]), float(start_xy[1]), 0.0)
    goal = (float(goal_xy[0]), float(goal_xy[1]))
    planner_config = PlannerConfig(**metadata["config"]["planner"])
    tic = time.perf_counter()
    result, elapsed = run_planner(scenario, start, goal, planner_config)
    runtime = float(elapsed if elapsed is not None else time.perf_counter() - tic)
    if result is None:
        return TrueRoute(
            length=1e6,
            risk=1e6,
            runtime_sec=runtime,
            success=False,
            end_x=float(start_xy[0]),
            end_y=float(start_xy[1]),
        )
    return TrueRoute(
        length=float(result.metrics["length"]),
        risk=float(result.metrics["risk"]),
        runtime_sec=runtime,
        success=True,
        end_x=float(result.path[-1, 0]),
        end_y=float(result.path[-1, 1]),
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["minimum_mean_route_survival", "risk_budget_per_route", "method"], dropna=False)
        .agg(
            missions=("mission_id", "count"),
            completed_tasks_mean=("completed_tasks", "mean"),
            completion_rate_mean=("completion_rate", "mean"),
            executable_commitments_mean=("executable_commitments", "mean"),
            mission_true_length_mean=("mission_true_length", "mean"),
            mission_true_risk_mean=("mission_true_risk", "mean"),
            mean_flow_time_mean=("mean_flow_time", "mean"),
            p95_flow_time_mean=("p95_flow_time", "mean"),
            mission_completion_time_mean=("mission_completion_time", "mean"),
            near_candidate_planner_calls_mean=("near_candidate_planner_calls", "mean"),
            proposal_verification_planner_calls_mean=("proposal_verification_planner_calls", "mean"),
            total_planner_calls_mean=("total_planner_calls", "mean"),
            near_candidate_planner_runtime_sec_mean=("near_candidate_planner_runtime_sec", "mean"),
            planner_runtime_sec_mean=("planner_runtime_sec", "mean"),
            selected_cache_hits_mean=("selected_cache_hits", "mean"),
            prediction_runtime_sec_mean=("prediction_runtime_sec", "mean"),
            assignment_runtime_sec_mean=("assignment_runtime_sec", "mean"),
            decision_wall_time_sec_mean=("decision_wall_time_sec", "mean"),
            decision_wall_time_p95_sec_mean=("decision_wall_time_p95_sec", "mean"),
            pre_verification_violation_events_mean=("pre_verification_violation_events", "mean"),
            infeasible_events_mean=("infeasible_events", "mean"),
            executed_violation_events_mean=("executed_violation_events", "mean"),
            exhaustive_candidate_planner_call_estimate_mean=("exhaustive_candidate_planner_call_estimate", "mean"),
            planner_call_reduction_vs_exhaustive_mean=("planner_call_reduction_vs_exhaustive", "mean"),
        )
        .reset_index()
    )


if __name__ == "__main__":
    main()
