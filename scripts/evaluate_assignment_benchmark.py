from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.evaluate_assignment import apply_near_edge_policy, assignment_cost, assignment_to_string, solve_assignment, validate_near_edge_policy
from src.learning.bid_targets import bid_cost_weights, planner_cost_weights
from src.learning.cnn import BidCostCNN
from src.experiment_config import CURRENT_NEAR_ORACLE_EUCLIDEAN_THRESHOLD, CURRENT_NEAR_ZERO_EUCLIDEAN_THRESHOLD
from src.learning.features import build_feature_frame, load_or_build_corridor_images
from src.learning.mlp import BidCostMLP, BidCostScalarFusionMLP


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a learned bid model on a cached assignment benchmark.")
    parser.add_argument("benchmark_dir", type=Path, help="Directory containing instances.csv, pairs.csv, source_metadata.json.")
    parser.add_argument("model_dir", type=Path, help="Trained MLP or CNN model directory.")
    parser.add_argument("--output", type=Path, default=Path("outputs/assignment_benchmark_eval"))
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4096, help="Inference batch size.")
    parser.add_argument("--image-cache-dir", type=Path, default=Path("outputs/cache/corridor_images"))
    parser.add_argument(
        "--near-zero-euclidean-threshold",
        type=float,
        default=CURRENT_NEAR_ZERO_EUCLIDEAN_THRESHOLD,
        help="Set learned bid to zero when Euclidean distance is at or below this threshold. Disabled at 0.",
    )
    parser.add_argument(
        "--near-oracle-euclidean-threshold",
        type=float,
        default=CURRENT_NEAR_ORACLE_EUCLIDEAN_THRESHOLD,
        help=(
            "Use planner true_bid for learned bid when Euclidean distance is above the zero threshold "
            "and at or below this threshold. Disabled at 0."
        ),
    )
    args = parser.parse_args()
    validate_near_edge_policy(args.near_zero_euclidean_threshold, args.near_oracle_euclidean_threshold)

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.benchmark_dir / "source_metadata.json").read_text(encoding="utf-8"))
    instances = pd.read_csv(args.benchmark_dir / "instances.csv")
    pairs = pd.read_csv(args.benchmark_dir / "pairs.csv")
    validate_cached_bid_semantics(pairs, metadata, source=str(args.benchmark_dir))
    bundle = load_model(args.model_dir)
    validate_model_bid_weights(
        bundle,
        metadata,
        source=str(args.model_dir),
    )
    bundle["residual_scale"] = args.residual_scale

    tic = time.perf_counter()
    learned_preds = predict_costs(pairs, metadata, bundle, batch_size=args.batch_size, image_cache_dir=args.image_cache_dir)
    inference_sec = time.perf_counter() - tic
    pairs = pairs.copy()
    if "bid" in learned_preds:
        pairs["learned_bid"] = learned_preds["bid"]
    else:
        pairs["learned_length"] = learned_preds["length"]
        pairs["learned_risk"] = learned_preds["risk"]
        length_weight, risk_weight, _ = bid_cost_weights(metadata)
        pairs["learned_bid"] = length_weight * pairs["learned_length"] + risk_weight * pairs["learned_risk"]
    apply_near_edge_policy(
        pairs,
        column="baseline_bid",
        zero_threshold=args.near_zero_euclidean_threshold,
        oracle_threshold=args.near_oracle_euclidean_threshold,
        raw_column="raw_baseline_bid",
    )
    apply_near_edge_policy(
        pairs,
        column="learned_bid",
        zero_threshold=args.near_zero_euclidean_threshold,
        oracle_threshold=args.near_oracle_euclidean_threshold,
        raw_column="raw_learned_bid",
    )

    result_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for instance in instances.itertuples(index=False):
        instance_pairs = pairs[pairs["instance_id"] == int(instance.instance_id)].copy()
        n_agents = int(instance.n_agents)
        n_tasks = int(instance.n_tasks)
        true_matrix = matrix_from_column(instance_pairs, "true_bid", n_agents, n_tasks)
        baseline_matrix = matrix_from_column(instance_pairs, "baseline_bid", n_agents, n_tasks)
        learned_matrix = matrix_from_column(instance_pairs, "learned_bid", n_agents, n_tasks)

        oracle_assignment, oracle_true_cost = solve_assignment(true_matrix)
        baseline_assignment, _ = solve_assignment(baseline_matrix)
        learned_assignment, _ = solve_assignment(learned_matrix)
        baseline_true_cost = assignment_cost(true_matrix, baseline_assignment)
        learned_true_cost = assignment_cost(true_matrix, learned_assignment)
        baseline_pred_selected_cost = assignment_cost(baseline_matrix, baseline_assignment)
        baseline_pred_oracle_cost = assignment_cost(baseline_matrix, oracle_assignment)
        learned_pred_selected_cost = assignment_cost(learned_matrix, learned_assignment)
        learned_pred_oracle_cost = assignment_cost(learned_matrix, oracle_assignment)

        result_rows.append(
            {
                "instance_id": int(instance.instance_id),
                "scenario_id": int(instance.scenario_id),
                "oracle_true_cost": oracle_true_cost,
                "baseline_true_cost": baseline_true_cost,
                "learned_true_cost": learned_true_cost,
                "baseline_regret": baseline_true_cost - oracle_true_cost,
                "learned_regret": learned_true_cost - oracle_true_cost,
                "baseline_regret_pct": (baseline_true_cost - oracle_true_cost) / max(oracle_true_cost, 1e-6),
                "learned_regret_pct": (learned_true_cost - oracle_true_cost) / max(oracle_true_cost, 1e-6),
                "baseline_matches_oracle": int(baseline_assignment == oracle_assignment),
                "learned_matches_oracle": int(learned_assignment == oracle_assignment),
                "case_type": case_type(baseline_true_cost - oracle_true_cost, learned_true_cost - oracle_true_cost),
                "rescue_gain": max(0.0, baseline_true_cost - learned_true_cost),
                "harm_loss": max(0.0, learned_true_cost - baseline_true_cost),
                "baseline_pred_gap_to_oracle": baseline_pred_oracle_cost - baseline_pred_selected_cost,
                "learned_pred_gap_to_oracle": learned_pred_oracle_cost - learned_pred_selected_cost,
                "oracle_assignment": assignment_to_string(oracle_assignment),
                "baseline_assignment": assignment_to_string(baseline_assignment),
                "learned_assignment": assignment_to_string(learned_assignment),
            }
        )

        selected_oracle = set(enumerate(oracle_assignment))
        selected_baseline = set(enumerate(baseline_assignment))
        selected_learned = set(enumerate(learned_assignment))
        for row in instance_pairs.itertuples(index=False):
            key = (int(row.agent_id), int(row.task_id))
            pair_rows.append(
                {
                    **row._asdict(),
                    "baseline_error": float(row.baseline_bid - row.true_bid),
                    "learned_error": float(row.learned_bid - row.true_bid),
                    "selected_by_oracle_eval": int(key in selected_oracle),
                    "selected_by_baseline_eval": int(key in selected_baseline),
                    "selected_by_learned_eval": int(key in selected_learned),
                }
            )

    write_rows(args.output / "assignment_results.csv", result_rows)
    write_rows(args.output / "assignment_pairs.csv", pair_rows)
    summary = summarize(result_rows, pair_rows)
    summary["model"] = {
        "model_dir": str(args.model_dir),
        "model_type": bundle["model_type"],
        "target_mode": bundle["target_mode"],
        "target_names": bundle["target_names"],
        "residual_scale": args.residual_scale,
        "inference_sec": inference_sec,
        "inference_sec_per_edge": inference_sec / max(len(pairs), 1),
    }
    summary["benchmark_dir"] = str(args.benchmark_dir)
    summary["near_edge_policy"] = {
        "applies_to": ["baseline_bid", "learned_bid"],
        "zero_euclidean_threshold": args.near_zero_euclidean_threshold,
        "oracle_euclidean_threshold": args.near_oracle_euclidean_threshold,
    }
    (args.output / "assignment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved: {args.output}")


def load_model(model_dir: Path) -> dict[str, Any]:
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu")
    feature_names = json.loads((model_dir / "feature_names.json").read_text(encoding="utf-8"))
    target_names = json.loads((model_dir / "target_names.json").read_text(encoding="utf-8"))
    model_type = str(checkpoint.get("model_type", ""))
    if model_type in {"cnn", "profile_onehot_cnn"}:
        scalers = np.load(model_dir / "scalers.npz")
        model = BidCostCNN(
            scalar_dim=int(checkpoint["scalar_dim"]),
            output_dim=int(checkpoint["output_dim"]),
            image_channels=int(checkpoint["image_channels"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            dropout=float(checkpoint["dropout"]),
            image_pool=str(checkpoint["image_pool"]),
            fusion=str(checkpoint["fusion"]),
            image_encoder=str(checkpoint.get("image_encoder", "simple")),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return {
            "model_type": model_type,
            "model": model,
            "scalers": scalers,
            "feature_names": feature_names,
            "target_names": target_names,
            "target_mode": checkpoint["target_mode"],
            "density_scale": checkpoint["density_scale"],
            "bid_cost_weights": checkpoint.get("bid_cost_weights"),
            "image_size": int(checkpoint["image_size"]),
            "lateral_width": float(checkpoint["lateral_width"]),
            "corridor_scale_mode": str(checkpoint["corridor_scale_mode"]),
            "anchor_betas": [float(value) for value in checkpoint.get("anchor_betas", [])],
        }
    if model_type != "mlp":
        raise ValueError(f"Unsupported deployable model_type={model_type!r} in {model_dir}.")
    scalers = np.load(model_dir / "scaler.npz")
    if checkpoint["architecture"] == "cnn_scalar":
        model = BidCostScalarFusionMLP(
            input_dim=int(checkpoint["input_dim"]),
            output_dim=int(checkpoint["output_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            dropout=float(checkpoint["dropout"]),
        )
        model_type = "mlp_cnn_scalar"
    else:
        model = BidCostMLP(
            input_dim=int(checkpoint["input_dim"]),
            output_dim=int(checkpoint["output_dim"]),
            hidden_dims=tuple(checkpoint["hidden_dims"]),
            dropout=float(checkpoint["dropout"]),
        )
        model_type = "mlp"
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {
        "model_type": model_type,
        "model": model,
        "scalers": scalers,
        "feature_names": feature_names,
        "target_names": target_names,
        "target_mode": checkpoint["target_mode"],
        "density_scale": checkpoint["density_scale"],
        "bid_cost_weights": checkpoint["bid_cost_weights"],
    }


def validate_model_bid_weights(
    bundle: dict[str, Any],
    metadata: dict,
    source: str = "model",
) -> None:
    expected = dict(zip(["alpha", "beta", "gamma"], bid_cost_weights(metadata)))
    if bundle["model_type"] == "profile_onehot_cnn":
        beta = float(expected["beta"])
        if not any(np.isclose(beta, anchor) for anchor in bundle["anchor_betas"]):
            raise ValueError(
                f"{source} has profile anchors {bundle['anchor_betas']}, but evaluation requires beta={beta}."
            )
        return
    actual = bundle["bid_cost_weights"]
    if actual is None:
        raise ValueError(
            f"{source} is missing saved bid_cost_weights. Refusing to assume it matches the benchmark "
            f"bid weights {expected}. Retrain the model with current training code."
        )
    for key in ["alpha", "beta", "gamma"]:
        if abs(float(actual[key]) - float(expected[key])) > 1e-9:
            raise ValueError(
                f"{source} was trained with bid {key}={float(actual[key])}, "
                f"but evaluation metadata requires {key}={float(expected[key])}."
            )


def predict_costs(
    pair_df: pd.DataFrame,
    metadata: dict,
    bundle: dict[str, Any],
    batch_size: int = 4096,
    image_cache_dir: Path | None = None,
    scenario_cache: dict[int, Any] | None = None,
    base_image_cache: dict[tuple, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    features = build_feature_frame(pair_df, metadata)
    if bundle["model_type"] == "profile_onehot_cnn":
        beta = float(metadata["config"]["planner"]["beta"])
        matched = False
        for anchor in bundle["anchor_betas"]:
            column = f"profile_beta_{int(anchor) if float(anchor).is_integer() else anchor}"
            is_active = bool(np.isclose(beta, anchor))
            features[column] = float(is_active)
            matched = matched or is_active
        if not matched:
            raise ValueError(f"No profile anchor in {bundle['anchor_betas']} matches beta={beta}.")
    missing_features = [column for column in bundle["feature_names"] if column not in features.columns]
    if missing_features:
        raise ValueError(
            "Model requests features outside the deployable feature schema: "
            f"{missing_features}. Retrain it with the current feature builder."
        )
    x = features[bundle["feature_names"]].to_numpy(dtype=np.float32)
    scalers = bundle["scalers"]
    x_scaled = (x - scalers["x_mean"]) / scalers["x_std"]
    if bundle["model_type"] in {"cnn", "profile_onehot_cnn"}:
        image_size = (bundle["image_size"], bundle["image_size"])
        images = load_or_build_corridor_images(
            pair_df,
            metadata,
            image_size=image_size,
            lateral_width=bundle["lateral_width"],
            scale_mode=bundle["corridor_scale_mode"],
            cache_dir=image_cache_dir,
            scenario_cache=scenario_cache,
            base_image_cache=base_image_cache,
        )
        image_x = (images - scalers["image_mean"]) / scalers["image_std"]
        pred_batches = []
        batch_size = max(1, int(batch_size))
        with torch.no_grad():
            for start in range(0, len(x_scaled), batch_size):
                end = min(start + batch_size, len(x_scaled))
                pred_batches.append(
                    bundle["model"](
                        torch.tensor(x_scaled[start:end], dtype=torch.float32),
                        torch.tensor(image_x[start:end], dtype=torch.float32),
                    ).numpy()
                )
        pred_scaled = np.vstack(pred_batches)
    else:
        pred_batches = []
        batch_size = max(1, int(batch_size))
        with torch.no_grad():
            for start in range(0, len(x_scaled), batch_size):
                end = min(start + batch_size, len(x_scaled))
                pred_batches.append(bundle["model"](torch.tensor(x_scaled[start:end], dtype=torch.float32)).numpy())
        pred_scaled = np.vstack(pred_batches)

    pred_target = pred_scaled * scalers["target_std"] + scalers["target_mean"]
    baseline = baseline_predictions(pair_df, metadata, bundle["target_names"])
    if bundle["target_mode"] == "residual":
        pred = baseline + float(bundle["residual_scale"]) * pred_target
    elif bundle["target_mode"] == "residual_density":
        target_scales = target_scale_predictions(pair_df, metadata, bundle["target_names"], bundle["density_scale"])
        pred = baseline + float(bundle["residual_scale"]) * target_scales * pred_target
    elif bundle["target_mode"] == "density":
        target_scales = target_scale_predictions(pair_df, metadata, bundle["target_names"], bundle["density_scale"])
        pred = np.maximum(0.0, target_scales * pred_target)
    elif bundle["target_mode"] == "log_density":
        target_scales = target_scale_predictions(pair_df, metadata, bundle["target_names"], bundle["density_scale"])
        pred = np.maximum(0.0, target_scales * np.expm1(pred_target))
    else:
        pred = pred_target
    return {target: pred[:, idx] for idx, target in enumerate(bundle["target_names"])}


def baseline_predictions(pair_df: pd.DataFrame, metadata: dict, target_names: list[str]) -> np.ndarray:
    speed = float(metadata["config"]["planner"]["speed"])
    columns = []
    for target in target_names:
        if target == "length":
            columns.append(pair_df["euclidean_distance"].to_numpy(dtype=np.float32))
        elif target == "time":
            columns.append((pair_df["euclidean_distance"] / speed).to_numpy(dtype=np.float32))
        elif target == "risk":
            columns.append(pair_df["straight_line_risk"].to_numpy(dtype=np.float32))
        elif target == "bid":
            columns.append(pair_df["baseline_bid"].to_numpy(dtype=np.float32))
        else:
            raise ValueError(f"Unsupported target for baseline prediction: {target}")
    return np.column_stack(columns)


def validate_cached_bid_semantics(
    pairs: pd.DataFrame,
    metadata: dict,
    source: str = "cached benchmark",
    tolerance: float = 1e-4,
) -> None:
    alpha, beta, _ = bid_cost_weights(metadata)
    required_baseline = {"baseline_bid", "euclidean_distance", "straight_line_risk"}
    missing_baseline = required_baseline - set(pairs.columns)
    if missing_baseline:
        raise ValueError(f"{source} missing cached baseline bid columns: {sorted(missing_baseline)}")
    expected_baseline = alpha * pairs["euclidean_distance"].astype(float) + beta * pairs["straight_line_risk"].astype(float)
    baseline_diff = float(np.max(np.abs(pairs["baseline_bid"].astype(float) - expected_baseline))) if len(pairs) else 0.0
    if baseline_diff > tolerance:
        raise ValueError(
            f"{source} baseline_bid is inconsistent with source_metadata bid weights "
            f"(alpha={alpha}, beta={beta}); max_abs_diff={baseline_diff:.6g}. "
            "Regenerate the assignment benchmark with matching planner metadata."
        )

    required_true = {"true_bid", "length", "risk"}
    if not required_true.issubset(pairs.columns):
        return
    expected_true = alpha * pairs["length"].astype(float) + beta * pairs["risk"].astype(float)
    true_diff = float(np.max(np.abs(pairs["true_bid"].astype(float) - expected_true))) if len(pairs) else 0.0
    if true_diff > tolerance:
        raise ValueError(
            f"{source} true_bid is inconsistent with source_metadata bid weights "
            f"(alpha={alpha}, beta={beta}); max_abs_diff={true_diff:.6g}. "
            "Regenerate the assignment benchmark with matching planner metadata."
        )


def target_scale_predictions(pair_df: pd.DataFrame, metadata: dict, target_names: list[str], density_scale: str = "euclidean") -> np.ndarray:
    speed = float(metadata["config"]["planner"]["speed"])
    distance = pair_df["euclidean_distance"].to_numpy(dtype=np.float32)
    if density_scale == "effective_euclidean":
        goal_tolerance = float(metadata["config"]["planner"]["goal_tolerance"])
        distance = distance - goal_tolerance
    elif density_scale != "euclidean":
        raise ValueError(f"Unsupported density scale: {density_scale}")
    distance = np.maximum(distance, 1.0)
    columns = []
    for target in target_names:
        if target == "time":
            columns.append(distance / speed)
        else:
            columns.append(distance)
    return np.column_stack(columns).astype(np.float32)


def matrix_from_column(df: pd.DataFrame, column: str, n_agents: int, n_tasks: int) -> np.ndarray:
    matrix = np.zeros((n_agents, n_tasks), dtype=float)
    for _, row in df.iterrows():
        matrix[int(row["agent_id"]), int(row["task_id"])] = float(row[column])
    return matrix


def summarize(result_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not result_rows:
        return {"instances": 0}
    baseline_regrets = np.asarray([row["baseline_regret"] for row in result_rows], dtype=float)
    learned_regrets = np.asarray([row["learned_regret"] for row in result_rows], dtype=float)
    delta = learned_regrets - baseline_regrets
    baseline_err = np.asarray([row["baseline_error"] for row in pair_rows], dtype=float)
    learned_err = np.asarray([row["learned_error"] for row in pair_rows], dtype=float)
    rescue_gain = np.maximum(0.0, baseline_regrets - learned_regrets)
    harm_loss = np.maximum(0.0, learned_regrets - baseline_regrets)
    baseline_pred_gaps = np.asarray([row["baseline_pred_gap_to_oracle"] for row in result_rows], dtype=float)
    learned_pred_gaps = np.asarray([row["learned_pred_gap_to_oracle"] for row in result_rows], dtype=float)
    baseline_wrong_mask = baseline_regrets > 1e-9
    return {
        "instances": len(result_rows),
        "edges": len(pair_rows),
        "baseline_regret": distribution_summary(baseline_regrets),
        "learned_regret": distribution_summary(learned_regrets),
        "learned_minus_baseline_regret": distribution_summary(delta),
        "baseline_oracle_match_rate": float(np.mean([row["baseline_matches_oracle"] for row in result_rows])),
        "learned_oracle_match_rate": float(np.mean([row["learned_matches_oracle"] for row in result_rows])),
        "learned_better_rate": float(np.mean(learned_regrets < baseline_regrets - 1e-9)),
        "learned_equal_rate": float(np.mean(np.abs(learned_regrets - baseline_regrets) <= 1e-9)),
        "learned_worse_rate": float(np.mean(learned_regrets > baseline_regrets + 1e-9)),
        "case_categories": case_category_summary(result_rows),
        "rescue_harm": {
            "total_rescue_gain": float(np.sum(rescue_gain)),
            "total_harm_loss": float(np.sum(harm_loss)),
            "net_gain": float(np.sum(rescue_gain) - np.sum(harm_loss)),
            "mean_rescue_gain": float(np.mean(rescue_gain)),
            "mean_harm_loss": float(np.mean(harm_loss)),
            "rescue_rate": float(np.mean(rescue_gain > 1e-9)),
            "harm_rate": float(np.mean(harm_loss > 1e-9)),
        },
        "predicted_gap_to_oracle": {
            "baseline": distribution_summary(baseline_pred_gaps),
            "learned": distribution_summary(learned_pred_gaps),
            "learned_minus_baseline": distribution_summary(learned_pred_gaps - baseline_pred_gaps),
            "baseline_wrong_only": {
                "baseline": distribution_summary(baseline_pred_gaps[baseline_wrong_mask]),
                "learned": distribution_summary(learned_pred_gaps[baseline_wrong_mask]),
                "learned_minus_baseline": distribution_summary(
                    learned_pred_gaps[baseline_wrong_mask] - baseline_pred_gaps[baseline_wrong_mask]
                ),
            }
            if bool(np.any(baseline_wrong_mask))
            else None,
        },
        "baseline_edge_error": error_summary(baseline_err),
        "learned_edge_error": error_summary(learned_err),
        "selected_edge_error": selected_edge_summary(pair_rows),
    }


def case_type(baseline_regret: float, learned_regret: float, tol: float = 1e-9) -> str:
    baseline_correct = baseline_regret <= tol
    learned_correct = learned_regret <= tol
    if baseline_correct and learned_correct:
        return "both_correct"
    if (not baseline_correct) and learned_correct:
        return "baseline_wrong_learned_correct"
    if baseline_correct and (not learned_correct):
        return "baseline_correct_learned_wrong"
    return "both_wrong"


def case_category_summary(result_rows: list[dict[str, Any]]) -> dict[str, float | int]:
    labels = [
        "both_correct",
        "baseline_wrong_learned_correct",
        "baseline_correct_learned_wrong",
        "both_wrong",
    ]
    total = max(len(result_rows), 1)
    counts = {label: sum(1 for row in result_rows if row.get("case_type") == label) for label in labels}
    summary: dict[str, float | int] = {}
    for label in labels:
        summary[f"{label}_count"] = counts[label]
        summary[f"{label}_rate"] = counts[label] / total
    return summary


def selected_edge_summary(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_baseline = [row for row in pair_rows if int(row["selected_by_baseline_eval"]) == 1]
    selected_learned = [row for row in pair_rows if int(row["selected_by_learned_eval"]) == 1]
    return {
        "baseline_selected": error_summary(np.asarray([row["baseline_error"] for row in selected_baseline], dtype=float)),
        "learned_selected": error_summary(np.asarray([row["learned_error"] for row in selected_learned], dtype=float)),
    }


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def error_summary(error: np.ndarray) -> dict[str, float]:
    abs_error = np.abs(error)
    return {
        "bias": float(np.mean(error)),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_abs": float(np.median(abs_error)),
        "p75_abs": float(np.quantile(abs_error, 0.75)),
        "p90_abs": float(np.quantile(abs_error, 0.90)),
        "p95_abs": float(np.quantile(abs_error, 0.95)),
        "max_abs": float(np.max(abs_error)),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
