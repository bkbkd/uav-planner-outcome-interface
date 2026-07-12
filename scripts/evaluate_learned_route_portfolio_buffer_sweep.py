from __future__ import annotations

import argparse
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
from scripts.evaluate_learned_route_portfolio_budget import (
    MODES,
    build_mode_matrices_from_groups,
    build_row,
    compact_summary,
    summarize_edges,
    validate_same_instances,
    write_rows,
)
from scripts.evaluate_route_portfolio_budget import build_candidate_frontier, solve_frontier, sort_instances, sort_pairs
from scripts.interface_protocol import apply_near_target_policy, validation_risk_buffers
from src.experiment_config import CURRENT_NEAR_ORACLE_EUCLIDEAN_THRESHOLD, CURRENT_NEAR_ZERO_EUCLIDEAN_THRESHOLD


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate learned route-portfolio assignment across multiple one-sided risk buffers "
            "while sharing each mode's CNN predictions."
        )
    )
    parser.add_argument("--fast", type=Path, required=True)
    parser.add_argument("--fast-model", type=Path, required=True)
    parser.add_argument("--balanced", type=Path, required=True)
    parser.add_argument("--balanced-model", type=Path, required=True)
    parser.add_argument("--safe", type=Path, required=True)
    parser.add_argument("--safe-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--image-cache-dir", type=Path, default=Path("outputs/cache/corridor_images"))
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--risk-buffer-quantiles",
        nargs="+",
        default=["raw", "0.50", "0.75", "0.90", "0.95"],
        help='Use "raw" for no buffer, otherwise quantiles in [0, 1].',
    )
    parser.add_argument(
        "--budget-quantiles",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument("--near-zero-euclidean-threshold", type=float, default=CURRENT_NEAR_ZERO_EUCLIDEAN_THRESHOLD)
    parser.add_argument("--near-oracle-euclidean-threshold", type=float, default=CURRENT_NEAR_ORACLE_EUCLIDEAN_THRESHOLD)
    args = parser.parse_args()

    validate_near_edge_policy(args.near_zero_euclidean_threshold, args.near_oracle_euclidean_threshold)
    buffer_specs = parse_buffer_specs(args.risk_buffer_quantiles)
    for quantile in args.budget_quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"budget quantile must be in [0, 1], got {quantile}")

    specs = {
        "fast": (args.fast, args.fast_model),
        "balanced": (args.balanced, args.balanced_model),
        "safe": (args.safe, args.safe_model),
    }
    scenario_cache: dict[int, Any] = {}
    base_image_cache: dict[tuple, np.ndarray] = {}
    requested_quantiles = [q for _, q in buffer_specs if q is not None]
    mode_base = {
        mode: load_mode_base_predictions(
            benchmark_dir=benchmark_dir,
            model_dir=model_dir,
            batch_size=args.batch_size,
            image_cache_dir=args.image_cache_dir,
            residual_scale=args.residual_scale,
            requested_quantiles=requested_quantiles,
            scenario_cache=scenario_cache,
            base_image_cache=base_image_cache,
        )
        for mode, (benchmark_dir, model_dir) in specs.items()
    }

    for label, quantile in buffer_specs:
        mode_data = {
            mode: materialize_mode_predictions(
                base=data,
                risk_buffer_quantile=quantile,
                zero_threshold=args.near_zero_euclidean_threshold,
                oracle_threshold=args.near_oracle_euclidean_threshold,
            )
            for mode, data in mode_base.items()
        }
        validate_same_instances(mode_data)
        output = args.output_root / f"{args.output_prefix}_{label}"
        evaluate_buffer(output=output, mode_data=mode_data, args=args, risk_buffer_label=label, risk_buffer_quantile=quantile)
        print(f"saved: {output}", flush=True)


def parse_buffer_specs(values: list[str]) -> list[tuple[str, float | None]]:
    specs: list[tuple[str, float | None]] = []
    seen: set[str] = set()
    for value in values:
        if str(value).lower() == "raw":
            label = "raw"
            quantile = None
        else:
            quantile = float(value)
            if not 0.0 <= quantile <= 1.0:
                raise ValueError(f"risk buffer quantile must be in [0, 1], got {value}")
            label = f"q{int(round(quantile * 100)):02d}"
        if label in seen:
            continue
        seen.add(label)
        specs.append((label, quantile))
    return specs


def load_mode_base_predictions(
    benchmark_dir: Path,
    model_dir: Path,
    batch_size: int,
    image_cache_dir: Path,
    residual_scale: float,
    requested_quantiles: list[float],
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
    buffers = (
        validation_risk_buffers(
            model_dir / "predictions.csv",
            requested_quantiles,
            profile_beta=float(metadata["config"]["planner"]["beta"])
            if bundle["model_type"] == "profile_onehot_cnn"
            else None,
        )
        if requested_quantiles
        else {}
    )
    base_pairs = pairs.copy()
    base_pairs["pred_length_base"] = np.maximum(0.0, predictions["length"].astype(float))
    # Preserve the raw prediction so every sweep point applies the same
    # deployed rule as the main evaluation: max(0, risk_hat + buffer).
    base_pairs["pred_risk_base"] = predictions["risk"].astype(float)
    base_pairs["true_length"] = base_pairs["length"].astype(float)
    base_pairs["true_risk"] = base_pairs["risk"].astype(float)
    return {
        "instances": instances,
        "pairs": base_pairs,
        "metadata": metadata,
        "model_summary": {
            "benchmark_dir": str(benchmark_dir),
            "model_dir": str(model_dir),
            "model_type": bundle["model_type"],
            "target_mode": bundle["target_mode"],
            "target_names": bundle["target_names"],
            "inference_sec": inference_sec,
            "inference_sec_per_edge": inference_sec / max(len(base_pairs), 1),
        },
        "buffers": {str(key): float(value) for key, value in buffers.items()},
    }


def materialize_mode_predictions(
    base: dict[str, Any],
    risk_buffer_quantile: float | None,
    zero_threshold: float,
    oracle_threshold: float,
) -> dict[str, Any]:
    risk_buffer = 0.0
    if risk_buffer_quantile is not None:
        risk_buffer = float(base["buffers"][buffer_key(risk_buffer_quantile)])
    pred_pairs = base["pairs"].copy()
    pred_pairs["pred_length"] = pred_pairs["pred_length_base"].astype(float)
    pred_pairs["pred_risk"] = np.maximum(0.0, pred_pairs["pred_risk_base"].astype(float) + risk_buffer)
    apply_near_target_policy(pred_pairs, "pred_length", "length", zero_threshold, oracle_threshold)
    apply_near_target_policy(pred_pairs, "pred_risk", "risk", zero_threshold, oracle_threshold)
    model_summary = dict(base["model_summary"])
    model_summary["risk_buffer_quantile"] = risk_buffer_quantile
    model_summary["risk_buffer"] = risk_buffer
    return {
        "instances": base["instances"],
        "pairs": pred_pairs,
        "metadata": base["metadata"],
        "edge_summary": summarize_edges(pred_pairs),
        "model_summary": model_summary,
    }


def buffer_key(quantile: float) -> str:
    return f"buffer_q{int(round(float(quantile) * 100)):02d}"


def evaluate_buffer(
    output: Path,
    mode_data: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    risk_buffer_label: str,
    risk_buffer_quantile: float | None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    instances = mode_data["balanced"]["instances"]
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
        assignments = tuple(itertools.permutations(range(n_tasks), n_agents))
        route_choices = tuple(itertools.product(MODES, repeat=n_agents))
        true_frontier = build_candidate_frontier(true_matrices, assignments, route_choices)
        pred_frontiers = {"learned_portfolio": build_candidate_frontier(pred_matrices, assignments, route_choices)}
        for mode in MODES:
            pred_frontiers[f"{mode}_only_learned"] = build_candidate_frontier(
                pred_matrices,
                assignments,
                ((mode,) * n_agents,),
            )
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
            for method, frontier in pred_frontiers.items():
                selected = solve_frontier(frontier, budget)
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

    write_rows(output / "learned_budget_assignment_results.csv", rows)
    write_rows(output / "learned_portfolio_mode_selection.csv", detail_rows)
    compact_rows = compact_summary(rows, detail_rows)
    write_rows(output / "learned_budget_assignment_compact.csv", compact_rows)
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
            "risk_buffer_label": risk_buffer_label,
            "risk_buffer_quantile": risk_buffer_quantile,
            "near_zero_euclidean_threshold": args.near_zero_euclidean_threshold,
            "near_oracle_euclidean_threshold": args.near_oracle_euclidean_threshold,
        },
    }
    (output / "learned_budget_assignment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
