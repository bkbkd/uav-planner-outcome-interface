from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.evaluate_assignment_benchmark import load_model, predict_costs, validate_model_bid_weights
from scripts.evaluate_route_portfolio_budget import build_candidate_frontier, solve_frontier
from scripts.interface_protocol import validation_risk_buffers
from src.data.dataset_generator import PlannerConfig, run_planner
from src.experiment_config import CURRENT_NEAR_ORACLE_EUCLIDEAN_THRESHOLD, CURRENT_NEAR_ZERO_EUCLIDEAN_THRESHOLD
from src.learning.features import scenario_from_metadata


MODES = ("fast", "balanced", "safe")
MODE_BETA = {"fast": 150.0, "balanced": 650.0, "safe": 1500.0}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Serial wall-clock profiler for the static 5x5 route-portfolio protocol. "
            "It reruns the planner for sampled instances instead of extrapolating from cached pair runtimes."
        )
    )
    parser.add_argument(
        "--benchmark-set",
        nargs=4,
        action="append",
        metavar=("POOL", "FAST_DIR", "BALANCED_DIR", "SAFE_DIR"),
        required=True,
    )
    parser.add_argument("--fast-model", type=Path, required=True)
    parser.add_argument("--balanced-model", type=Path, required=True)
    parser.add_argument("--safe-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances-per-pool", type=int, default=30)
    parser.add_argument("--budget-quantiles", nargs="+", type=float, default=[0.10, 0.50, 0.90])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["exact_portfolio", "learned_portfolio", "dispatch_global", "fast_only", "balanced_only", "safe_only"],
    )
    parser.add_argument("--near-zero-euclidean-threshold", type=float, default=CURRENT_NEAR_ZERO_EUCLIDEAN_THRESHOLD)
    parser.add_argument("--near-oracle-euclidean-threshold", type=float, default=CURRENT_NEAR_ORACLE_EUCLIDEAN_THRESHOLD)
    parser.add_argument("--risk-buffer-quantile", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--image-cache-dir",
        type=Path,
        default=None,
        help="Optional corridor-image cache. Omit for end-to-end online matrix-construction timing.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "static_runtime_serial_results.csv"
    completed = load_completed(result_path) if args.resume else set()

    rows: list[dict[str, Any]] = []
    if args.resume and result_path.exists():
        rows = pd.read_csv(result_path).to_dict("records")

    model_paths = {"fast": args.fast_model, "balanced": args.balanced_model, "safe": args.safe_model}
    loaded_models: dict[Path, dict[str, Any]] = {}
    models: dict[str, dict[str, Any]] = {}
    for mode, raw_path in model_paths.items():
        path = raw_path.resolve()
        if path not in loaded_models:
            loaded_models[path] = load_model(path)
        models[mode] = loaded_models[path]
    for bundle in models.values():
        bundle["residual_scale"] = 1.0
    buffers = {
        mode: float(
            next(
                iter(
                    validation_risk_buffers(
                        path / "predictions.csv",
                        [args.risk_buffer_quantile],
                        profile_beta=MODE_BETA[mode]
                        if models[mode]["model_type"] == "profile_onehot_cnn"
                        else None,
                    ).values()
                )
            )
        )
        for mode, path in model_paths.items()
    }
    warmed = False
    for raw_pool, fast_dir, balanced_dir, safe_dir in args.benchmark_set:
        pool = str(raw_pool)
        data = load_pool({"fast": Path(fast_dir), "balanced": Path(balanced_dir), "safe": Path(safe_dir)})
        mode_metadata = {mode: metadata_with_beta(data[mode]["metadata"], MODE_BETA[mode]) for mode in MODES}
        for mode in MODES:
            validate_model_bid_weights(models[mode], mode_metadata[mode], source=str(model_paths[mode]))
        instance_ids = sorted(data["balanced"]["instances"]["instance_id"].astype(int).unique())
        instance_ids = instance_ids[: max(0, int(args.instances_per_pool))]
        for instance_id in instance_ids:
            instance = data["balanced"]["instances"][data["balanced"]["instances"]["instance_id"].astype(int) == int(instance_id)].iloc[0]
            scenario_id = int(instance["scenario_id"])
            n_agents = int(instance["n_agents"])
            n_tasks = int(instance["n_tasks"])
            pair_frames = {
                mode: data[mode]["pairs"][data[mode]["pairs"]["instance_id"].astype(int) == int(instance_id)].copy()
                for mode in MODES
            }
            if not warmed:
                warm_models(
                    pair_frames,
                    mode_metadata,
                    models,
                    batch_size=int(args.batch_size),
                    image_cache_dir=args.image_cache_dir,
                    scenario_cache={},
                    oracle_threshold=float(args.near_oracle_euclidean_threshold),
                )
                warmed = True
            assignments = tuple(tuple(perm) for perm in itertools.permutations(range(n_tasks), n_agents))
            route_choices = tuple(tuple(choice) for choice in itertools.product(MODES, repeat=n_agents))
            true_matrices = build_true_matrices(pair_frames, n_agents, n_tasks)
            frontier = build_candidate_frontier(true_matrices, assignments, route_choices)
            for quantile in args.budget_quantiles:
                budget = float(np.quantile(frontier["risks"], float(quantile)))
                if solve_frontier(frontier, budget) is None:
                    continue
                for method in args.methods:
                    key = (pool, int(instance_id), float(quantile), str(method))
                    if key in completed:
                        continue
                    cache: dict[tuple[Any, ...], dict[str, Any]] = {}
                    row = profile_method(
                        pool=pool,
                        instance_id=instance_id,
                        scenario_id=scenario_id,
                        quantile=float(quantile),
                        budget=budget,
                        method=str(method),
                        scenario_metadata=data["balanced"]["metadata"],
                        pair_frames=pair_frames,
                        mode_metadata=mode_metadata,
                        n_agents=n_agents,
                        n_tasks=n_tasks,
                        assignments=assignments,
                        route_choices=route_choices,
                        models=models,
                        buffers=buffers,
                        batch_size=int(args.batch_size),
                        image_cache_dir=args.image_cache_dir,
                        cache=cache,
                        zero_threshold=float(args.near_zero_euclidean_threshold),
                        oracle_threshold=float(args.near_oracle_euclidean_threshold),
                    )
                    rows.append(row)
                    write_rows(result_path, rows)
                    write_summary(args.output, rows, args)
                    print(
                        f"[pool={pool} inst={instance_id} q={quantile:.2f} {method}] "
                        f"calls={row['total_planner_calls']} wall={row['decision_wall_time_sec']:.2f}s",
                        flush=True,
                    )

    write_rows(result_path, rows)
    write_summary(args.output, rows, args)
    print(f"saved: {args.output}", flush=True)


def load_completed(path: Path) -> set[tuple[str, int, float, str]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return {
        (str(row.pool), int(row.instance_id), float(row.budget_quantile), str(row.method))
        for row in df.itertuples(index=False)
    }


def load_pool(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    out = {}
    for mode, path in paths.items():
        out[mode] = {
            "instances": pd.read_csv(path / "instances.csv"),
            "pairs": pd.read_csv(path / "pairs.csv"),
            "metadata": json.loads((path / "source_metadata.json").read_text(encoding="utf-8")),
        }
    return out


def metadata_with_beta(metadata: dict[str, Any], beta: float) -> dict[str, Any]:
    out = deepcopy(metadata)
    planner = dict(out["config"]["planner"])
    planner["beta"] = float(beta)
    out["config"]["planner"] = planner
    return out


def build_true_matrices(pair_frames: dict[str, pd.DataFrame], n_agents: int, n_tasks: int) -> dict[str, dict[str, np.ndarray]]:
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for mode, frame in pair_frames.items():
        length = np.zeros((n_agents, n_tasks), dtype=float)
        risk = np.zeros((n_agents, n_tasks), dtype=float)
        for row in frame.itertuples(index=False):
            length[int(row.agent_id), int(row.task_id)] = float(row.length)
            risk[int(row.agent_id), int(row.task_id)] = float(row.risk)
        matrices[mode] = {"length": length, "risk": risk}
    return matrices


def profile_method(
    *,
    pool: str,
    instance_id: int,
    scenario_id: int,
    quantile: float,
    budget: float,
    method: str,
    scenario_metadata: dict[str, Any],
    pair_frames: dict[str, pd.DataFrame],
    mode_metadata: dict[str, dict[str, Any]],
    n_agents: int,
    n_tasks: int,
    assignments: tuple[tuple[int, ...], ...],
    route_choices: tuple[tuple[str, ...], ...],
    models: dict[str, dict[str, Any]],
    buffers: dict[str, float],
    batch_size: int,
    image_cache_dir: Path | None,
    cache: dict[tuple[Any, ...], dict[str, Any]],
    zero_threshold: float,
    oracle_threshold: float,
) -> dict[str, Any]:
    tic = time.perf_counter()
    map_tic = time.perf_counter()
    scenario = scenario_from_metadata(scenario_metadata, scenario_id)
    map_preprocessing_runtime = time.perf_counter() - map_tic
    scenario_cache: dict[int, Any] = {scenario_id: scenario}
    prediction_runtime = 0.0
    assignment_runtime = 0.0
    near_calls = 0
    near_runtime = 0.0
    selected_calls = 0
    selected_runtime = 0.0
    predicted_feasible = True

    if method == "exact_portfolio":
        candidates = all_route_specs(pair_frames)
        near_calls, near_runtime = plan_specs(candidates, scenario, mode_metadata, cache, zero_threshold)
        solve_tic = time.perf_counter()
        exact_matrices = matrices_from_cache(pair_frames, cache, n_agents, n_tasks, zero_threshold)
        selected = solve_frontier(build_candidate_frontier(exact_matrices, assignments, route_choices), budget)
        assignment_runtime += time.perf_counter() - solve_tic
        if selected is None:
            raise RuntimeError(f"exact portfolio infeasible for pool={pool} instance={instance_id} q={quantile}")
        selected_specs = choice_specs(pair_frames, selected["assignment"], selected["modes"])
    elif method in {"learned_portfolio", "dispatch_global", "fast_only", "balanced_only", "safe_only"}:
        selected_mode = None if method == "dispatch_global" else method.removesuffix("_only")
        prediction_modes = MODES if method in {"learned_portfolio", "dispatch_global"} else (str(selected_mode),)
        base_image_cache: dict[tuple, np.ndarray] = {}
        predicted, prediction_runtime = predict_far_outcomes(
            pair_frames,
            mode_metadata,
            models,
            buffers,
            prediction_modes,
            batch_size=batch_size,
            image_cache_dir=image_cache_dir,
            scenario_cache=scenario_cache,
            base_image_cache=base_image_cache,
            oracle_threshold=oracle_threshold,
        )
        candidates = near_edge_specs({mode: pair_frames[mode] for mode in prediction_modes}, zero_threshold, oracle_threshold)
        near_calls, near_runtime = plan_specs(candidates, scenario, mode_metadata, cache, zero_threshold)
        solve_tic = time.perf_counter()
        pred_matrices = matrices_from_interface(
            pair_frames,
            predicted,
            cache,
            n_agents,
            n_tasks,
            prediction_modes,
            zero_threshold,
            oracle_threshold,
        )
        if method == "learned_portfolio":
            choices = route_choices
        elif method == "dispatch_global":
            choices = tuple((mode,) * n_agents for mode in MODES)
        else:
            choices = ((str(selected_mode),) * n_agents,)
        selected = solve_frontier(build_candidate_frontier(pred_matrices, assignments, choices), budget)
        assignment_runtime += time.perf_counter() - solve_tic
        predicted_feasible = selected is not None
        selected_specs = None if selected is None else choice_specs(pair_frames, selected["assignment"], selected["modes"])
        if selected_specs:
            selected_calls, selected_runtime = plan_specs(selected_specs, scenario, mode_metadata, cache, zero_threshold)
    else:
        raise ValueError(f"unsupported method: {method}")

    selected_specs = selected_specs or []
    true_length = sum(float(cache[spec_key(spec)]["length"]) for spec in selected_specs)
    true_risk = sum(float(cache[spec_key(spec)]["risk"]) for spec in selected_specs)
    true_violation = bool(selected_specs) and true_risk > budget + 1e-9
    total_calls = near_calls + selected_calls
    wall = time.perf_counter() - tic
    return {
        "pool": pool,
        "instance_id": int(instance_id),
        "scenario_id": int(scenario_id),
        "budget_quantile": float(quantile),
        "risk_budget": float(budget),
        "method": method,
        "prediction_runtime_sec": float(prediction_runtime),
        "map_preprocessing_runtime_sec": float(map_preprocessing_runtime),
        "assignment_runtime_sec": float(assignment_runtime),
        "near_oracle_planner_calls": int(near_calls),
        "selected_route_planner_calls": int(selected_calls),
        "total_planner_calls": int(total_calls),
        "near_oracle_runtime_sec": float(near_runtime),
        "selected_route_runtime_sec": float(selected_runtime),
        "planner_runtime_sec": float(near_runtime + selected_runtime),
        "decision_wall_time_sec": float(wall),
        "cache_entries": int(len(cache)),
        "predicted_feasible": int(predicted_feasible),
        "true_risk_violation": int(true_violation),
        "true_selected_length": float(true_length) if selected_specs else math.nan,
        "true_selected_risk": float(true_risk) if selected_specs else math.nan,
        "selected_routes": int(len(selected_specs)),
        "selected_assignment": ";".join(
            f"{int(spec['agent_id'])}->{int(spec['task_id'])}" for spec in sorted(selected_specs, key=lambda item: item["agent_id"])
        ),
        "selected_modes": ";".join(str(spec["mode"]) for spec in sorted(selected_specs, key=lambda item: item["agent_id"])),
    }


def all_route_specs(pair_frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for mode, frame in pair_frames.items():
        specs.extend(row_to_spec(mode, row) for row in frame.itertuples(index=False))
    return specs


def near_edge_specs(pair_frames: dict[str, pd.DataFrame], zero_threshold: float, oracle_threshold: float) -> list[dict[str, Any]]:
    specs = []
    for mode, frame in pair_frames.items():
        for row in frame.itertuples(index=False):
            distance = float(row.euclidean_distance)
            if zero_threshold < distance <= oracle_threshold:
                specs.append(row_to_spec(mode, row))
    return specs


def warm_models(
    pair_frames: dict[str, pd.DataFrame],
    mode_metadata: dict[str, dict[str, Any]],
    models: dict[str, dict[str, Any]],
    *,
    batch_size: int,
    image_cache_dir: Path | None,
    scenario_cache: dict[int, Any],
    oracle_threshold: float,
) -> None:
    for mode in MODES:
        far = pair_frames[mode][pair_frames[mode]["euclidean_distance"].astype(float) > oracle_threshold].head(1)
        if not far.empty:
            predict_costs(
                far,
                mode_metadata[mode],
                models[mode],
                batch_size=batch_size,
                image_cache_dir=image_cache_dir,
                scenario_cache=scenario_cache,
            )


def predict_far_outcomes(
    pair_frames: dict[str, pd.DataFrame],
    mode_metadata: dict[str, dict[str, Any]],
    models: dict[str, dict[str, Any]],
    buffers: dict[str, float],
    modes: tuple[str, ...],
    *,
    batch_size: int,
    image_cache_dir: Path | None,
    scenario_cache: dict[int, Any],
    base_image_cache: dict[tuple, np.ndarray],
    oracle_threshold: float,
) -> tuple[dict[str, dict[tuple[int, int], tuple[float, float]]], float]:
    outcomes: dict[str, dict[tuple[int, int], tuple[float, float]]] = {}
    elapsed = 0.0
    for mode in modes:
        far = pair_frames[mode][pair_frames[mode]["euclidean_distance"].astype(float) > oracle_threshold].copy()
        tic = time.perf_counter()
        predictions = predict_costs(
            far,
            mode_metadata[mode],
            models[mode],
            batch_size=batch_size,
            image_cache_dir=image_cache_dir,
            scenario_cache=scenario_cache,
            base_image_cache=base_image_cache,
        )
        elapsed += time.perf_counter() - tic
        mode_outcomes: dict[tuple[int, int], tuple[float, float]] = {}
        for idx, row in enumerate(far.itertuples(index=False)):
            mode_outcomes[(int(row.agent_id), int(row.task_id))] = (
                max(0.0, float(predictions["length"][idx])),
                max(0.0, float(predictions["risk"][idx]) + float(buffers[mode])),
            )
        outcomes[mode] = mode_outcomes
    return outcomes, elapsed


def matrices_from_interface(
    pair_frames: dict[str, pd.DataFrame],
    predicted: dict[str, dict[tuple[int, int], tuple[float, float]]],
    cache: dict[tuple[Any, ...], dict[str, Any]],
    n_agents: int,
    n_tasks: int,
    modes: tuple[str, ...],
    zero_threshold: float,
    oracle_threshold: float,
) -> dict[str, dict[str, np.ndarray]]:
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for mode in modes:
        length = np.zeros((n_agents, n_tasks), dtype=float)
        risk = np.zeros((n_agents, n_tasks), dtype=float)
        for row in pair_frames[mode].itertuples(index=False):
            agent = int(row.agent_id)
            task = int(row.task_id)
            distance = float(row.euclidean_distance)
            if distance <= zero_threshold:
                values = (0.0, 0.0)
            elif distance <= oracle_threshold:
                result = cache[spec_key(row_to_spec(mode, row))]
                values = (float(result["length"]), float(result["risk"]))
            else:
                values = predicted[mode][(agent, task)]
            length[agent, task], risk[agent, task] = values
        matrices[mode] = {"length": length, "risk": risk}
    return matrices


def matrices_from_cache(
    pair_frames: dict[str, pd.DataFrame],
    cache: dict[tuple[Any, ...], dict[str, Any]],
    n_agents: int,
    n_tasks: int,
    zero_threshold: float,
) -> dict[str, dict[str, np.ndarray]]:
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for mode in MODES:
        length = np.zeros((n_agents, n_tasks), dtype=float)
        risk = np.zeros((n_agents, n_tasks), dtype=float)
        for row in pair_frames[mode].itertuples(index=False):
            agent = int(row.agent_id)
            task = int(row.task_id)
            if float(row.euclidean_distance) <= zero_threshold:
                values = (0.0, 0.0)
            else:
                result = cache[spec_key(row_to_spec(mode, row))]
                values = (float(result["length"]), float(result["risk"]))
            length[agent, task], risk[agent, task] = values
        matrices[mode] = {"length": length, "risk": risk}
    return matrices


def choice_specs(
    pair_frames: dict[str, pd.DataFrame],
    assignment: tuple[int, ...],
    modes: tuple[str, ...],
) -> list[dict[str, Any]]:
    specs = []
    for agent, (task, mode) in enumerate(zip(assignment, modes)):
        frame = pair_frames[mode]
        rows = frame[(frame["agent_id"].astype(int) == agent) & (frame["task_id"].astype(int) == int(task))]
        if len(rows) != 1:
            raise ValueError(f"missing pair row for agent={agent} task={task} mode={mode}")
        specs.append(row_to_spec(mode, rows.iloc[0]))
    return specs


def row_to_spec(mode: str, row: Any) -> dict[str, Any]:
    return {
        "mode": mode,
        "agent_id": int(row.agent_id),
        "task_id": int(row.task_id),
        "start": (float(row.start_x), float(row.start_y), float(row.start_theta)),
        "goal": (float(row.goal_x), float(row.goal_y)),
        "distance": float(row.euclidean_distance),
        "seed": int(row.seed),
    }


def spec_key(spec: dict[str, Any]) -> tuple[Any, ...]:
    return (
        spec["mode"],
        spec["agent_id"],
        spec["task_id"],
        round(float(spec["start"][0]), 4),
        round(float(spec["start"][1]), 4),
        round(float(spec["goal"][0]), 4),
        round(float(spec["goal"][1]), 4),
    )


def spec_distance(spec: dict[str, Any]) -> float:
    return float(spec["distance"])


def plan_specs(
    specs: list[dict[str, Any]],
    scenario: Any,
    mode_metadata: dict[str, dict[str, Any]],
    cache: dict[tuple[Any, ...], dict[str, Any]],
    zero_threshold: float,
) -> tuple[int, float]:
    calls = 0
    runtime = 0.0
    for spec in specs:
        if spec_distance(spec) <= zero_threshold:
            cache.setdefault(spec_key(spec), {"length": 0.0, "risk": 0.0, "runtime_sec": 0.0, "success": True})
            continue
        key = spec_key(spec)
        if key in cache:
            continue
        planner_config = PlannerConfig(**mode_metadata[spec["mode"]]["config"]["planner"])
        result, elapsed = run_planner(
            scenario,
            spec["start"],
            spec["goal"],
            planner_config,
        )
        elapsed = float(elapsed)
        calls += 1
        runtime += elapsed
        if result is None:
            cache[key] = {"length": 1e6, "risk": 1e6, "runtime_sec": elapsed, "success": False}
        else:
            cache[key] = {
                "length": float(result.metrics["length"]),
                "risk": float(result.metrics["risk"]),
                "runtime_sec": elapsed,
                "success": True,
            }
    return calls, runtime


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_summary(output: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["budget_quantile", "method"], as_index=False)
        .agg(
            rows=("method", "size"),
            total_planner_calls_mean=("total_planner_calls", "mean"),
            total_planner_calls_median=("total_planner_calls", "median"),
            prediction_runtime_sec_mean=("prediction_runtime_sec", "mean"),
            map_preprocessing_runtime_sec_mean=("map_preprocessing_runtime_sec", "mean"),
            assignment_runtime_sec_mean=("assignment_runtime_sec", "mean"),
            planner_runtime_sec_mean=("planner_runtime_sec", "mean"),
            decision_wall_time_sec_mean=("decision_wall_time_sec", "mean"),
            decision_wall_time_sec_median=("decision_wall_time_sec", "median"),
            decision_wall_time_sec_q25=("decision_wall_time_sec", lambda values: values.quantile(0.25)),
            decision_wall_time_sec_q75=("decision_wall_time_sec", lambda values: values.quantile(0.75)),
            predicted_feasible_rate=("predicted_feasible", "mean"),
            true_violation_rate=("true_risk_violation", "mean"),
            near_oracle_planner_calls_mean=("near_oracle_planner_calls", "mean"),
            selected_route_planner_calls_mean=("selected_route_planner_calls", "mean"),
        )
        .sort_values(["budget_quantile", "method"])
    )
    summary.to_csv(output / "static_runtime_serial_summary.csv", index=False)
    metadata = {
        "benchmark_sets": [list(x) for x in args.benchmark_set],
        "instances_per_pool": int(args.instances_per_pool),
        "budget_quantiles": [float(x) for x in args.budget_quantiles],
        "methods": list(args.methods),
        "near_zero_euclidean_threshold": float(args.near_zero_euclidean_threshold),
        "near_oracle_euclidean_threshold": float(args.near_oracle_euclidean_threshold),
        "risk_buffer_quantile": float(args.risk_buffer_quantile),
        "batch_size": int(args.batch_size),
        "image_cache_dir": str(args.image_cache_dir) if args.image_cache_dir is not None else None,
        "runtime_protocol": (
            "serial end-to-end dispatch rebuild with warmed, preloaded models; includes prediction, "
            "candidate matrix construction, assignment solve, and exact execution of selected routes; "
            "route cache is method-local and never shared across methods"
        ),
    }
    (output / "static_runtime_serial_summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary.to_dict("records")}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
