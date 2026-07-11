from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_bid_cnn import (
    apply_image_scaler,
    build_target_scales,
    fit_image_scaler,
    make_loader,
    train,
    transform_targets,
)
from scripts.train_bid_mlp import (
    add_bid_target_if_needed,
    apply_scaler,
    build_baseline_predictions,
    build_loss,
    compute_prediction_metrics,
    fit_scalers,
    fit_target_scaler,
    scenario_split_summary,
    train_val_split,
    write_history,
)
from src.learning.cnn import BidCostCNN
from src.learning.features import feasible_regression_data, load_dataset_table, load_or_build_corridor_images


TARGETS = ["length", "risk"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one CNN shared by discrete planner profiles using one-hot beta conditioning.")
    parser.add_argument("--profile", nargs=2, action="append", metavar=("BETA", "DATASET_DIR"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--image-cache-dir", type=Path, default=Path("outputs/cache/corridor_images"))
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.05)
    args = parser.parse_args()

    profiles = parse_profiles(args.profile)
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    profile_data = [load_profile(beta, path, profiles, args) for beta, path in profiles]
    validate_paired_profiles(profile_data)

    scalar_x = np.concatenate([item["x"] for item in profile_data], axis=0)
    images = np.concatenate([item["images"] for item in profile_data], axis=0)
    y = np.concatenate([item["y"] for item in profile_data], axis=0)
    baselines = np.concatenate([item["baseline"] for item in profile_data], axis=0)
    target_scales = np.concatenate([item["target_scales"] for item in profile_data], axis=0)
    transformed = transform_targets(y, baselines, target_scales, "residual")
    split = concatenate_splits(profile_data)

    x_scaler = fit_scalers(scalar_x[split["train"]], transformed[split["train"]])
    y_scaler = fit_target_scaler(transformed[split["train"]])
    image_scaler = fit_image_scaler(images[split["train"]])
    x_scaled = apply_scaler(scalar_x, x_scaler["x_mean"], x_scaler["x_std"])
    y_scaled = apply_scaler(transformed, y_scaler["mean"], y_scaler["std"])
    images_scaled = apply_image_scaler(images, image_scaler)

    model = BidCostCNN(
        scalar_dim=scalar_x.shape[1],
        output_dim=y.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        image_pool="grid",
        fusion="concat",
        image_encoder="simple",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = build_loss("mae", 1.0)
    history = train(
        model,
        optimizer,
        loss_fn,
        make_loader(x_scaled, images_scaled, y_scaled, split["train"], args.batch_size, shuffle=True),
        make_loader(x_scaled, images_scaled, y_scaled, split["val"], args.batch_size, shuffle=False),
        args.epochs,
        args.patience,
    )
    save_recovery_checkpoint(args.output / "model.pt", model, args, profiles, scalar_x.shape[1], y.shape[1])
    predictions = predict_batched(
        model,
        x_scaled,
        images_scaled,
        y_scaler,
        baselines,
        batch_size=max(args.batch_size, 256),
    )
    save_results(args, profiles, profile_data, model, history, predictions, baselines, x_scaler, y_scaler, image_scaler)


def parse_profiles(raw: list[list[str]]) -> list[tuple[float, Path]]:
    profiles = [(float(beta), Path(path)) for beta, path in raw]
    betas = [beta for beta, _ in profiles]
    if len(profiles) < 2 or len(set(betas)) != len(betas):
        raise ValueError("At least two profiles with unique beta values are required.")
    return sorted(profiles)


def load_profile(beta: float, dataset_dir: Path, profiles: list[tuple[float, Path]], args: argparse.Namespace) -> dict[str, Any]:
    df, metadata = load_dataset_table(dataset_dir)
    actual_beta = float(metadata["config"]["planner"]["beta"])
    if not np.isclose(actual_beta, beta):
        raise ValueError(f"{dataset_dir} has planner beta={actual_beta}, expected {beta}.")
    add_bid_target_if_needed(df, TARGETS, metadata)
    x, y, feature_names, target_names, filtered = feasible_regression_data(df, metadata, TARGETS)
    onehot = np.zeros((len(filtered), len(profiles)), dtype=np.float32)
    profile_index = [item[0] for item in profiles].index(beta)
    onehot[:, profile_index] = 1.0
    x = np.concatenate([x, onehot], axis=1)
    feature_names = feature_names + [f"profile_beta_{format_beta(item[0])}" for item in profiles]
    images = load_or_build_corridor_images(
        filtered,
        metadata,
        image_size=(args.image_size, args.image_size),
        lateral_width=320.0,
        scale_mode="square_edge",
        cache_dir=args.image_cache_dir,
    )
    split = train_val_split(filtered)
    return {
        "beta": beta,
        "dataset_dir": dataset_dir,
        "metadata": metadata,
        "filtered": filtered,
        "x": x,
        "y": y,
        "images": images,
        "baseline": build_baseline_predictions(filtered, metadata, target_names),
        "target_scales": build_target_scales(filtered, metadata, target_names),
        "feature_names": feature_names,
        "target_names": target_names,
        "split": split,
    }


def validate_paired_profiles(items: list[dict[str, Any]]) -> None:
    reference = items[0]
    exact_keys = ["sample_id", "scenario_id", "split"]
    coordinate_keys = ["start_x", "start_y", "goal_x", "goal_y"]
    ref = reference["filtered"].reset_index(drop=True)
    for item in items[1:]:
        current = item["filtered"].reset_index(drop=True)
        exact_match = ref[exact_keys].equals(current[exact_keys])
        coordinate_match = np.allclose(
            ref[coordinate_keys].to_numpy(dtype=float),
            current[coordinate_keys].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-9,
        )
        if not exact_match or not coordinate_match:
            raise ValueError(f"Profile beta={item['beta']} is not row-paired with beta={reference['beta']}.")
        if item["feature_names"] != reference["feature_names"] or item["target_names"] != reference["target_names"]:
            raise ValueError("Profiles do not share the same feature and target schema.")


def concatenate_splits(items: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    combined: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    offset = 0
    for item in items:
        for name in combined:
            if name in item["split"]:
                combined[name].append(item["split"][name] + offset)
        offset += len(item["filtered"])
    return {name: np.concatenate(parts).astype(int) for name, parts in combined.items() if parts}


def predict_batched(
    model: BidCostCNN,
    x_scaled: np.ndarray,
    images_scaled: np.ndarray,
    y_scaler: dict[str, np.ndarray],
    baselines: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_scaled), batch_size):
            end = min(start + batch_size, len(x_scaled))
            batches.append(
                model(
                    torch.tensor(x_scaled[start:end], dtype=torch.float32),
                    torch.tensor(images_scaled[start:end], dtype=torch.float32),
                ).numpy()
            )
    pred_scaled = np.concatenate(batches, axis=0)
    pred_residual = pred_scaled * y_scaler["std"] + y_scaler["mean"]
    return baselines + pred_residual


def save_recovery_checkpoint(
    path: Path,
    model: BidCostCNN,
    args: argparse.Namespace,
    profiles: list[tuple[float, Path]],
    scalar_dim: int,
    output_dim: int,
) -> None:
    torch.save(
        {
            "model_type": "profile_onehot_cnn",
            "model_state_dict": model.state_dict(),
            "scalar_dim": scalar_dim,
            "output_dim": output_dim,
            "hidden_dim": args.hidden_dim,
            "conditioning": "discrete_beta_onehot",
            "anchor_betas": [beta for beta, _ in profiles],
            "training_complete": True,
            "evaluation_complete": False,
        },
        path,
    )


def save_results(
    args: argparse.Namespace,
    profiles: list[tuple[float, Path]],
    items: list[dict[str, Any]],
    model: BidCostCNN,
    history: list[dict[str, float]],
    predictions: np.ndarray,
    baselines: np.ndarray,
    x_scaler: dict[str, np.ndarray],
    y_scaler: dict[str, np.ndarray],
    image_scaler: dict[str, np.ndarray],
) -> None:
    feature_names = items[0]["feature_names"]
    target_names = items[0]["target_names"]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics: dict[str, Any] = {
        "conditioning": "discrete_beta_onehot",
        "anchor_betas": [beta for beta, _ in profiles],
        "parameter_count": parameter_count,
        "profiles": {},
    }
    offset = 0
    all_prediction_frames = []
    for item in items:
        count = len(item["filtered"])
        local_predictions = predictions[offset : offset + count]
        local_baselines = baselines[offset : offset + count]
        metrics["profiles"][format_beta(item["beta"])] = {
            "dataset_dir": str(item["dataset_dir"]),
            "targets": compute_prediction_metrics(item["y"], local_predictions, target_names, item["split"]),
            "risk_error": risk_error_summary(item["y"], local_predictions, target_names, item["split"]),
            "scenario_split": scenario_split_summary(item["filtered"], item["split"]),
        }
        frame = prediction_frame(item, local_predictions, local_baselines)
        all_prediction_frames.append(frame)
        offset += count

    torch.save(
        {
            "model_type": "profile_onehot_cnn",
            "model_state_dict": model.state_dict(),
            "scalar_dim": len(feature_names),
            "output_dim": len(target_names),
            "image_channels": 2,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "image_pool": "grid",
            "image_encoder": "simple",
            "fusion": "concat",
            "target_mode": "residual",
            "density_scale": "euclidean",
            "conditioning": "discrete_beta_onehot",
            "anchor_betas": [beta for beta, _ in profiles],
            "loss": "mae",
            "image_size": args.image_size,
            "lateral_width": 320.0,
            "corridor_scale_mode": "square_edge",
        },
        args.output / "model.pt",
    )
    np.savez(
        args.output / "scalers.npz",
        x_mean=x_scaler["x_mean"],
        x_std=x_scaler["x_std"],
        target_mean=y_scaler["mean"],
        target_std=y_scaler["std"],
        image_mean=image_scaler["mean"],
        image_std=image_scaler["std"],
    )
    (args.output / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    (args.output / "target_names.json").write_text(json.dumps(target_names, indent=2), encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.concat(all_prediction_frames, ignore_index=True).to_csv(args.output / "predictions.csv", index=False)
    write_history(args.output / "history.csv", history)
    print(json.dumps(metrics, indent=2))
    print(f"saved: {args.output}")


def prediction_frame(item: dict[str, Any], predictions: np.ndarray, baselines: np.ndarray) -> pd.DataFrame:
    target_names = item["target_names"]
    out = item["filtered"][["sample_id", "scenario_id", "split"] + target_names].copy()
    out.insert(0, "profile_beta", float(item["beta"]))
    for index, target in enumerate(target_names):
        out[f"baseline_{target}"] = baselines[:, index]
        out[f"pred_{target}"] = predictions[:, index]
        out[f"err_{target}"] = predictions[:, index] - out[target]
    return out


def risk_error_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    split: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    risk_index = target_names.index("risk")
    summary = {}
    for name, indices in split.items():
        error = y_pred[indices, risk_index] - y_true[indices, risk_index]
        summary[name] = {
            "bias": float(np.mean(error)),
            "under_rate": float(np.mean(error < 0.0)),
            "under_p05": float(np.quantile(error, 0.05)),
            "under_p01": float(np.quantile(error, 0.01)),
        }
    return summary


def format_beta(beta: float) -> str:
    return str(int(beta)) if float(beta).is_integer() else str(beta).replace(".", "p")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
