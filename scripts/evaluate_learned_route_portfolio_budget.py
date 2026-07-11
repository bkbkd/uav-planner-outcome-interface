from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.evaluate_assignment import assignment_to_string, validate_near_edge_policy
from scripts.evaluate_assignment_benchmark import (
    load_model,
    predict_costs,
    validate_cached_bid_semantics,
    validate_model_bid_weights,
)
from scripts.evaluate_route_portfolio_budget import (
    build_candidate_frontier,
    profile_frontier_specs,
    route_assignment_cost,
    solve_frontier,
)
from scripts.evaluate_route_portfolio_budget import sort_instances, sort_pairs
from scripts.interface_protocol import apply_near_target_policy, validation_risk_buffers
from src.experiment_config import CURRENT_NEAR_ORACLE_EUCLIDEAN_THRESHOLD, CURRENT_NEAR_ZERO_EUCLIDEAN_THRESHOLD


MODES = ("fast", "balanced", "safe")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate risk-budget route-portfolio assignment using CNN-predicted length/risk "
            "for fast, balanced, and safe route candidates, with true planner length/risk used only for evaluation."
        )
    )
    parser.add_argument("--fast", type=Path, required=True, help="Benchmark directory for the fast route mode.")
    parser.add_argument("--fast-model", type=Path, required=True)
    parser.add_argument("--balanced", type=Path, required=True, help="Benchmark directory for the balanced route mode.")
    parser.add_argument("--balanced-model", type=Path, required=True)
    parser.add_argument("--safe", type=Path, required=True, help="Benchmark directory for the safe route mode.")
    parser.add_argument("--safe-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--image-cache-dir", type=Path, default=Path("outputs/cache/corridor_images"))
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--risk-buffer-quantile",
        type=float,
        default=None,
        help="Optional edge-only validation quantile added to each mode's predicted risk before budget assignment.",
    )
    parser.add_argument(
        "--budget-quantiles",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument(
        "--near-zero-euclidean-threshold",
        type=float,
        default=CURRENT_NEAR_ZERO_EUCLIDEAN_THRESHOLD,
    )
    parser.add_argument(
        "--near-oracle-euclidean-threshold",
        type=float,
        default=CURRENT_NEAR_ORACLE_EUCLIDEAN_THRESHOLD,
    )
    args = parser.parse_args()
    validate_near_edge_policy(args.near_zero_euclidean_threshold, args.near_oracle_euclidean_threshold)
    for quantile in args.budget_quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"budget quantile must be in [0, 1], got {quantile}")
    if args.risk_buffer_quantile is not None and not 0.0 <= args.risk_buffer_quantile <= 1.0:
        raise ValueError(f"risk buffer quantile must be in [0, 1], got {args.risk_buffer_quantile}")

    args.output.mkdir(parents=True, exist_ok=True)
    specs = {
        "fast": (args.fast, args.fast_model),
        "balanced": (args.balanced, args.balanced_model),
        "safe": (args.safe, args.safe_model),
    }
    scenario_cache: dict[int, Any] = {}
    base_image_cache: dict[tuple, np.ndarray] = {}
    mode_data = {
        mode: load_mode_predictions(
            benchmark_dir=benchmark_dir,
            model_dir=model_dir,
            batch_size=args.batch_size,
            image_cache_dir=args.image_cache_dir,
            residual_scale=args.residual_scale,
            zero_threshold=args.near_zero_euclidean_threshold,
            oracle_threshold=args.near_oracle_euclidean_threshold,
            risk_buffer_quantile=args.risk_buffer_quantile,
            scenario_cache=scenario_cache,
            base_image_cache=base_image_cache,
        )
        for mode, (benchmark_dir, model_dir) in specs.items()
    }
    validate_same_instances(mode_data)

    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    instances = mode_data["balanced"]["instances"]
    assignments_cache: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {}
    route_choices_cache: dict[int, dict[str, tuple[tuple[str, ...], ...]]] = {}
    pair_groups = {
        mode: {
            int(instance_id): group
            for instance_id, group in data["pairs"].groupby(data["pairs"]["instance_id"].astype(int), sort=False)
        }
        for mode, data in mode_data.items()
    }
    for instance in instances.itertuples(index=False):
        instance_id = int(instance.instance_id)
        n_agents = int(instance.n_agents)
        n_tasks = int(instance.n_tasks)
        true_matrices = build_mode_matrices_from_groups(pair_groups, instance_id, n_agents, n_tasks, prefix="true")
        pred_matrices = build_mode_matrices_from_groups(pair_groups, instance_id, n_agents, n_tasks, prefix="pred")
        cache_key = (n_agents, n_tasks)
        assignments = assignments_cache.setdefault(cache_key, tuple(itertools.permutations(range(n_tasks), n_agents)))
        frontier_specs = route_choices_cache.setdefault(n_agents, profile_frontier_specs(n_agents))
        true_frontier = build_candidate_frontier(true_matrices, assignments, frontier_specs["portfolio"])
        pred_frontiers = {
            learned_method_name(method): build_candidate_frontier(pred_matrices, assignments, route_choices)
            for method, route_choices in frontier_specs.items()
        }
        portfolio_true_risks = true_frontier["risks"]
        for quantile in args.budget_quantiles:
            budget = float(np.quantile(portfolio_true_risks, quantile))
            oracle = solve_frontier(true_frontier, budget)
            if oracle is None:
                continue
            rows.append(
                build_row(
                    instance=instance,
                    quantile=quantile,
                    budget=budget,
                    method="oracle_true_portfolio",
                    selected=oracle,
                    oracle=oracle,
                    true_matrices=true_matrices,
                    predicted=0,
                )
            )
            selected_by_method = {
                method: solve_frontier(frontier, budget)
                for method, frontier in pred_frontiers.items()
            }
            for method, selected in selected_by_method.items():
                row = build_row(
                    instance=instance,
                    quantile=quantile,
                    budget=budget,
                    method=method,
                    selected=selected,
                    oracle=oracle,
                    true_matrices=true_matrices,
                    predicted=1,
                )
                rows.append(row)
                if method == "learned_portfolio" and selected is not None:
                    counts = Counter(selected["modes"])
                    detail_rows.append(
                        {
                            "instance_id": instance_id,
                            "scenario_id": int(instance.scenario_id),
                            "budget_quantile": float(quantile),
                            "risk_budget": budget,
                            "fast_edges": counts.get("fast", 0),
                            "balanced_edges": counts.get("balanced", 0),
                            "safe_edges": counts.get("safe", 0),
                            "pred_selected_length": selected["length"],
                            "pred_selected_risk": selected["risk"],
                            "true_selected_length": row["selected_true_length"],
                            "true_selected_risk": row["selected_true_risk"],
                            "assignment": assignment_to_string(selected["assignment"]),
                            "modes": ";".join(selected["modes"]),
                        }
                    )

    write_rows(args.output / "learned_budget_assignment_results.csv", rows)
    write_rows(args.output / "learned_portfolio_mode_selection.csv", detail_rows)
    compact_rows = compact_summary(rows, detail_rows)
    write_rows(args.output / "learned_budget_assignment_compact.csv", compact_rows)
    summary = {
        "compact_budget_summary": compact_rows,
        "edge_prediction_summary": {mode: data["edge_summary"] for mode, data in mode_data.items()},
        "model_summary": {mode: data["model_summary"] for mode, data in mode_data.items()},
        "args": {
            "fast": str(args.fast),
            "fast_model": str(args.fast_model),
            "balanced": str(args.balanced),
            "balanced_model": str(args.balanced_model),
            "safe": str(args.safe),
            "safe_model": str(args.safe_model),
            "budget_quantiles": args.budget_quantiles,
            "risk_buffer_quantile": args.risk_buffer_quantile,
            "near_zero_euclidean_threshold": args.near_zero_euclidean_threshold,
            "near_oracle_euclidean_threshold": args.near_oracle_euclidean_threshold,
        },
    }
    (args.output / "learned_budget_assignment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved: {args.output}")


def learned_method_name(method: str) -> str:
    if method == "portfolio":
        return "learned_portfolio"
    if method == "global_profile":
        return "dispatch_global_learned"
    if method.endswith("_only"):
        return f"{method}_learned"
    return f"learned_{method}"


def load_mode_predictions(
    benchmark_dir: Path,
    model_dir: Path,
    batch_size: int,
    image_cache_dir: Path,
    residual_scale: float,
    zero_threshold: float,
    oracle_threshold: float,
    risk_buffer_quantile: float | None,
    scenario_cache: dict[int, Any],
    base_image_cache: dict[tuple, np.ndarray],
) -> dict[str, Any]:
    metadata = json.loads((benchmark_dir / "source_metadata.json").read_text(encoding="utf-8"))
    instances = sort_instances(pd.read_csv(benchmark_dir / "instances.csv"))
    pairs = sort_pairs(pd.read_csv(benchmark_dir / "pairs.csv"))
    validate_cached_bid_semantics(pairs, metadata, source=str(benchmark_dir))
    bundle = load_model(model_dir)
    validate_model_bid_weights(bundle, metadata, source=str(model_dir))
    missing_targets = {"length", "risk"} - set(bundle["target_names"])
    if missing_targets:
        raise ValueError(f"{model_dir} missing required heads: {sorted(missing_targets)}")
    bundle["residual_scale"] = residual_scale

    tic = time.perf_counter()
    predictions = predict_costs(
        pairs,
        metadata,
        bundle,
        batch_size=batch_size,
        image_cache_dir=image_cache_dir,
        scenario_cache=scenario_cache,
        base_image_cache=base_image_cache,
    )
    inference_sec = time.perf_counter() - tic
    risk_buffer = 0.0
    if risk_buffer_quantile is not None:
        buffers = validation_risk_buffers(
            model_dir / "predictions.csv",
            [float(risk_buffer_quantile)],
            profile_beta=float(metadata["config"]["planner"]["beta"])
            if bundle["model_type"] == "profile_onehot_cnn"
            else None,
        )
        risk_buffer = next(iter(buffers.values()))
    pred_pairs = pairs.copy()
    pred_pairs["pred_length"] = np.maximum(0.0, predictions["length"].astype(float))
    pred_pairs["pred_risk"] = np.maximum(0.0, predictions["risk"].astype(float) + risk_buffer)
    apply_near_target_policy(pred_pairs, "pred_length", "length", zero_threshold, oracle_threshold)
    apply_near_target_policy(pred_pairs, "pred_risk", "risk", zero_threshold, oracle_threshold)
    pred_pairs["true_length"] = pred_pairs["length"].astype(float)
    pred_pairs["true_risk"] = pred_pairs["risk"].astype(float)

    return {
        "instances": instances,
        "pairs": pred_pairs,
        "metadata": metadata,
        "edge_summary": summarize_edges(pred_pairs),
        "model_summary": {
            "benchmark_dir": str(benchmark_dir),
            "model_dir": str(model_dir),
            "model_type": bundle["model_type"],
            "target_mode": bundle["target_mode"],
            "target_names": bundle["target_names"],
            "inference_sec": inference_sec,
            "inference_sec_per_edge": inference_sec / max(len(pred_pairs), 1),
            "risk_buffer_quantile": risk_buffer_quantile,
            "risk_buffer": risk_buffer,
        },
    }


def validate_same_instances(mode_data: dict[str, dict[str, Any]]) -> None:
    reference_instances = mode_data["balanced"]["instances"][
        ["instance_id", "scenario_id", "agents_json", "tasks_json"]
    ].reset_index(drop=True)
    reference_pairs = mode_data["balanced"]["pairs"][["instance_id", "agent_id", "task_id"]].reset_index(drop=True)
    reference_xy = mode_data["balanced"]["pairs"][["start_x", "start_y", "goal_x", "goal_y"]].reset_index(drop=True)
    for mode, data in mode_data.items():
        instances = data["instances"][["instance_id", "scenario_id", "agents_json", "tasks_json"]].reset_index(drop=True)
        pairs = data["pairs"][["instance_id", "agent_id", "task_id"]].reset_index(drop=True)
        xy = data["pairs"][["start_x", "start_y", "goal_x", "goal_y"]].reset_index(drop=True)
        if not reference_instances.equals(instances):
            raise ValueError(f"{mode} instances do not match balanced benchmark")
        if not reference_pairs.equals(pairs):
            raise ValueError(f"{mode} pair keys do not match balanced benchmark")
        if not np.allclose(reference_xy.to_numpy(dtype=float), xy.to_numpy(dtype=float), rtol=0.0, atol=1e-6):
            raise ValueError(f"{mode} pair geometry does not match balanced benchmark")


def build_mode_matrices(
    mode_data: dict[str, dict[str, Any]],
    instance_id: int,
    n_agents: int,
    n_tasks: int,
    prefix: str,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        mode: {
            "length": matrix_from_pairs(data["pairs"], instance_id, f"{prefix}_length", n_agents, n_tasks),
            "risk": matrix_from_pairs(data["pairs"], instance_id, f"{prefix}_risk", n_agents, n_tasks),
        }
        for mode, data in mode_data.items()
    }


def build_mode_matrices_from_groups(
    pair_groups: dict[str, dict[int, pd.DataFrame]],
    instance_id: int,
    n_agents: int,
    n_tasks: int,
    prefix: str,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        mode: {
            "length": matrix_from_pair_frame(groups[instance_id], f"{prefix}_length", n_agents, n_tasks),
            "risk": matrix_from_pair_frame(groups[instance_id], f"{prefix}_risk", n_agents, n_tasks),
        }
        for mode, groups in pair_groups.items()
    }


def matrix_from_pairs(pairs: pd.DataFrame, instance_id: int, column: str, n_agents: int, n_tasks: int) -> np.ndarray:
    df = pairs[pairs["instance_id"].astype(int) == instance_id]
    return matrix_from_pair_frame(df, column, n_agents, n_tasks)


def matrix_from_pair_frame(df: pd.DataFrame, column: str, n_agents: int, n_tasks: int) -> np.ndarray:
    matrix = np.zeros((n_agents, n_tasks), dtype=float)
    agents = df["agent_id"].to_numpy(dtype=int)
    tasks = df["task_id"].to_numpy(dtype=int)
    values = df[column].to_numpy(dtype=float)
    matrix[agents, tasks] = values
    return matrix


def build_row(
    instance: Any,
    quantile: float,
    budget: float,
    method: str,
    selected: dict[str, Any] | None,
    oracle: dict[str, Any],
    true_matrices: dict[str, dict[str, np.ndarray]],
    predicted: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "instance_id": int(instance.instance_id),
        "scenario_id": int(instance.scenario_id),
        "budget_quantile": float(quantile),
        "risk_budget": budget,
        "method": method,
        "uses_predicted_interface": predicted,
        "oracle_true_length": oracle["length"],
        "oracle_true_risk": oracle["risk"],
        "oracle_assignment": assignment_to_string(oracle["assignment"]),
        "oracle_modes": ";".join(oracle["modes"]),
    }
    if selected is None:
        row.update(
            {
                "predicted_feasible": 0,
                "selected_assignment": "",
                "selected_modes": "",
                "selected_pred_length": math.nan,
                "selected_pred_risk": math.nan,
                "selected_true_length": math.nan,
                "selected_true_risk": math.nan,
                "true_length_regret_to_portfolio": math.nan,
                "true_risk_violation": math.nan,
                "true_risk_violation_amount": math.nan,
                "matches_oracle": 0,
            }
        )
        return row

    true_length = route_assignment_cost(true_matrices, selected["assignment"], selected["modes"], "length")
    true_risk = route_assignment_cost(true_matrices, selected["assignment"], selected["modes"], "risk")
    violation_amount = max(0.0, true_risk - budget)
    row.update(
        {
            "predicted_feasible": 1,
            "selected_assignment": assignment_to_string(selected["assignment"]),
            "selected_modes": ";".join(selected["modes"]),
            "selected_pred_length": selected["length"],
            "selected_pred_risk": selected["risk"],
            "selected_true_length": true_length,
            "selected_true_risk": true_risk,
            "true_length_regret_to_portfolio": true_length - oracle["length"],
            "true_risk_violation": int(violation_amount > 1e-9),
            "true_risk_violation_amount": violation_amount,
            "matches_oracle": int(selected["assignment"] == oracle["assignment"] and selected["modes"] == oracle["modes"]),
        }
    )
    return row

def summarize_edges(pairs: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"edges": int(len(pairs))}
    for target in ["length", "risk"]:
        err = pairs[f"pred_{target}"].to_numpy(dtype=float) - pairs[f"true_{target}"].to_numpy(dtype=float)
        out[target] = {
            "mae": float(np.mean(np.abs(err))) if len(err) else math.nan,
            "bias": float(np.mean(err)) if len(err) else math.nan,
            "rmse": float(np.sqrt(np.mean(err**2))) if len(err) else math.nan,
        }
    return out


def compact_summary(rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    detail = pd.DataFrame(detail_rows)
    out: list[dict[str, Any]] = []
    for quantile in sorted(df["budget_quantile"].astype(float).unique()):
        qdf = df[df["budget_quantile"].astype(float) == quantile]
        oracle = method_stats(qdf, "oracle_true_portfolio")
        learned = method_stats(qdf, "learned_portfolio")
        balanced = method_stats(qdf, "balanced_only_learned")
        qdetail = detail[detail["budget_quantile"].astype(float) == quantile] if len(detail) else pd.DataFrame()
        total_edges = max(
            1,
            int(qdetail[["fast_edges", "balanced_edges", "safe_edges"]].astype(float).to_numpy().sum()),
        )
        row = {
            "budget_quantile": float(quantile),
            "oracle_true_length_mean": oracle["selected_true_length_mean"],
            "oracle_true_risk_mean": oracle["selected_true_risk_mean"],
            "learned_portfolio_true_length_mean": learned["selected_true_length_mean"],
            "learned_portfolio_true_risk_mean": learned["selected_true_risk_mean"],
            "learned_portfolio_predicted_feasible_rate": learned["predicted_feasible_rate"],
            "learned_portfolio_violation_rate": learned["true_risk_violation_rate"],
            "learned_portfolio_violation_amount_mean": learned["true_risk_violation_amount_mean"],
            "learned_portfolio_regret_to_true_portfolio_mean": learned["true_length_regret_mean"],
            "balanced_only_true_length_mean": balanced["selected_true_length_mean"],
            "balanced_only_predicted_feasible_rate": balanced["predicted_feasible_rate"],
            "balanced_only_violation_rate": balanced["true_risk_violation_rate"],
            "portfolio_vs_balanced_true_length_reduction": (
                balanced["selected_true_length_mean"] - learned["selected_true_length_mean"]
            ),
            "portfolio_vs_balanced_true_length_reduction_pct": (
                (balanced["selected_true_length_mean"] - learned["selected_true_length_mean"])
                / max(balanced["selected_true_length_mean"], 1e-9)
            ),
            "learned_portfolio_fast_rate": (
                float(qdetail["fast_edges"].astype(float).sum() / total_edges) if len(qdetail) else math.nan
            ),
            "learned_portfolio_balanced_rate": (
                float(qdetail["balanced_edges"].astype(float).sum() / total_edges) if len(qdetail) else math.nan
            ),
            "learned_portfolio_safe_rate": (
                float(qdetail["safe_edges"].astype(float).sum() / total_edges) if len(qdetail) else math.nan
            ),
        }
        out.append(row)
    return out


def method_stats(qdf: pd.DataFrame, method: str) -> dict[str, Any]:
    group = qdf[qdf["method"] == method]
    feasible = group[group["predicted_feasible"].astype(int) == 1]
    return {
        "method": method,
        "instances": int(len(group)),
        "predicted_feasible_rate": float(group["predicted_feasible"].astype(float).mean()) if len(group) else 0.0,
        "selected_true_length_mean": float(feasible["selected_true_length"].astype(float).mean()) if len(feasible) else math.inf,
        "selected_true_risk_mean": float(feasible["selected_true_risk"].astype(float).mean()) if len(feasible) else math.nan,
        "true_length_regret_mean": float(feasible["true_length_regret_to_portfolio"].astype(float).mean()) if len(feasible) else math.inf,
        "true_risk_violation_rate": float(feasible["true_risk_violation"].astype(float).mean()) if len(feasible) else math.nan,
        "true_risk_violation_amount_mean": (
            float(feasible["true_risk_violation_amount"].astype(float).mean()) if len(feasible) else math.nan
        ),
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
