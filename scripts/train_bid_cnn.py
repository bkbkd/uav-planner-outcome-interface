from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.learning.cnn import BidCostCNN
from src.learning.features import (
    feasible_regression_data,
    load_or_build_corridor_images,
    load_dataset_table,
)
from src.learning.bid_targets import bid_cost_weights, planner_cost_weights
from scripts.train_bid_mlp import (
    add_bid_target_if_needed,
    apply_scaler,
    build_loss,
    build_baseline_predictions,
    compute_metrics,
    compute_prediction_metrics,
    fit_scalers,
    fit_target_scaler,
    plot_history,
    plot_parity,
    print_metrics,
    scenario_split_summary,
    train_val_split,
    write_history,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a corridor-image CNN for learned UAV bid costs.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--target-mode",
        choices=["direct", "residual", "density", "residual_density", "log_density"],
        default="residual",
    )
    parser.add_argument(
        "--density-scale",
        choices=["euclidean", "effective_euclidean"],
        default="euclidean",
        help="Scale used by density/log_density target modes. effective_euclidean subtracts planner goal_tolerance.",
    )
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--lateral-width", type=float, default=320.0)
    parser.add_argument("--corridor-scale-mode", choices=["fixed_width", "square_edge"], default="fixed_width")
    parser.add_argument("--image-cache-dir", type=Path, default=Path("outputs/cache/corridor_images"))
    parser.add_argument("--image-pool", choices=["global", "grid"], default="global")
    parser.add_argument("--image-encoder", choices=["simple", "wide", "residual"], default="simple")
    parser.add_argument("--fusion", choices=["concat", "scalar_residual"], default="concat")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--targets", nargs="+", default=["length", "risk"])
    parser.add_argument("--loss", choices=["mse", "mae", "smooth_l1", "asymmetric_mae", "asymmetric_smooth_l1"], default="mse")
    parser.add_argument("--huber-beta", type=float, default=1.0, help="Beta parameter for SmoothL1 loss on scaled targets.")
    parser.add_argument("--asym-under-weight", type=float, default=2.0, help="Weight applied to underestimation in asymmetric SmoothL1.")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = args.output or args.dataset_dir / "models" / "bid_cnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    df, metadata = load_dataset_table(args.dataset_dir)
    add_bid_target_if_needed(df, args.targets, metadata)
    x, y, feature_names, target_names, filtered = feasible_regression_data(df, metadata, args.targets)
    images = load_or_build_corridor_images(
        filtered,
        metadata,
        image_size=(args.image_size, args.image_size),
        lateral_width=args.lateral_width,
        scale_mode=args.corridor_scale_mode,
        cache_dir=args.image_cache_dir,
    )
    split = train_val_split(filtered)

    baseline_predictions = build_baseline_predictions(filtered, metadata, target_names)
    target_scales = build_target_scales(filtered, metadata, target_names, args.density_scale)
    train_targets = transform_targets(y, baseline_predictions, target_scales, args.target_mode)

    x_scaler = fit_scalers(x[split["train"]], train_targets[split["train"]])
    y_scaler = fit_target_scaler(train_targets[split["train"]])
    image_scaler = fit_image_scaler(images[split["train"]])
    x_scaled = apply_scaler(x, x_scaler["x_mean"], x_scaler["x_std"])
    images_scaled = apply_image_scaler(images, image_scaler)
    y_scaled = apply_scaler(train_targets, y_scaler["mean"], y_scaler["std"])

    train_loader = make_loader(
        x_scaled,
        images_scaled,
        y_scaled,
        split["train"],
        args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        x_scaled,
        images_scaled,
        y_scaled,
        split["val"],
        args.batch_size,
        shuffle=False,
    )

    model = BidCostCNN(
        scalar_dim=x.shape[1],
        output_dim=y.shape[1],
        image_channels=images.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        image_pool=args.image_pool,
        fusion=args.fusion,
        image_encoder=args.image_encoder,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = build_loss(args.loss, args.huber_beta, args.asym_under_weight)

    history = train(
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        args.epochs,
        args.patience,
    )
    predictions = predict(model, x_scaled, images_scaled, y_scaler, baseline_predictions, target_scales, args.target_mode)
    metrics = compute_metrics(y, predictions, target_names, split)
    metrics["baselines"] = {"straight_line": compute_prediction_metrics(y, baseline_predictions, target_names, split)}
    metrics["target_mode"] = args.target_mode
    metrics["density_scale"] = args.density_scale
    metrics["planner_cost_weights"] = dict(zip(["alpha", "beta", "gamma"], planner_cost_weights(metadata)))
    metrics["bid_cost_weights"] = dict(zip(["alpha", "beta", "gamma"], bid_cost_weights(metadata)))
    metrics["loss"] = args.loss
    metrics["huber_beta"] = args.huber_beta
    metrics["asym_under_weight"] = args.asym_under_weight
    metrics["image_size"] = args.image_size
    metrics["lateral_width"] = args.lateral_width
    metrics["corridor_scale_mode"] = args.corridor_scale_mode
    metrics["image_cache_dir"] = str(args.image_cache_dir) if args.image_cache_dir else None
    metrics["image_pool"] = args.image_pool
    metrics["fusion"] = args.fusion
    metrics["scenario_split"] = scenario_split_summary(filtered, split)

    save_artifacts(
        output_dir,
        model,
        x_scaler,
        y_scaler,
        image_scaler,
        feature_names,
        target_names,
        args,
        history,
        metrics,
        filtered,
        predictions,
        baseline_predictions,
        split,
    )
    print_metrics(metrics)
    print(f"saved: {output_dir}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_image_scaler(images: np.ndarray) -> dict[str, np.ndarray]:
    mean = images.mean(axis=(0, 2, 3), keepdims=True)
    std = images.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_image_scaler(images: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    return ((images - scaler["mean"]) / scaler["std"]).astype(np.float32)


def make_loader(
    x: np.ndarray,
    images: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tensors = TensorDataset(
        torch.tensor(x[indices], dtype=torch.float32),
        torch.tensor(images[indices], dtype=torch.float32),
        torch.tensor(y[indices], dtype=torch.float32),
    )
    return DataLoader(tensors, batch_size=min(batch_size, len(indices)), shuffle=shuffle)


def train(
    model: BidCostCNN,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    patience: int,
) -> list[dict[str, float]]:
    history = []
    best_state = None
    best_val = float("inf")
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for scalar_x, image_x, yb in train_loader:
            optimizer.zero_grad()
            pred = model(scalar_x, image_x)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))

        val_loss = evaluate_scaled_loss(model, loss_fn, val_loader)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})
        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate_scaled_loss(
    model: BidCostCNN,
    loss_fn: nn.Module,
    loader: DataLoader,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for scalar_x, image_x, yb in loader:
            pred = model(scalar_x, image_x)
            loss = loss_fn(pred, yb)
            losses.append(float(loss))
    return float(np.mean(losses)) if losses else float("nan")


def predict(
    model: BidCostCNN,
    x_scaled: np.ndarray,
    images_scaled: np.ndarray,
    y_scaler: dict[str, np.ndarray],
    baseline_predictions: np.ndarray,
    target_scales: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_scaled = model(
            torch.tensor(x_scaled, dtype=torch.float32),
            torch.tensor(images_scaled, dtype=torch.float32),
        ).numpy()
    pred_target = pred_scaled * y_scaler["std"] + y_scaler["mean"]
    if target_mode == "residual":
        return baseline_predictions + pred_target
    if target_mode == "residual_density":
        return baseline_predictions + target_scales * pred_target
    if target_mode == "density":
        return np.maximum(0.0, target_scales * pred_target)
    if target_mode == "log_density":
        return np.maximum(0.0, target_scales * np.expm1(pred_target))
    return pred_target


def transform_targets(
    y: np.ndarray,
    baseline_predictions: np.ndarray,
    target_scales: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    if target_mode == "residual":
        return y - baseline_predictions
    if target_mode == "residual_density":
        return ((y - baseline_predictions) / np.maximum(target_scales, 1e-6)).astype(np.float32)
    if target_mode == "density":
        return (np.maximum(y, 0.0) / np.maximum(target_scales, 1e-6)).astype(np.float32)
    if target_mode == "log_density":
        return np.log1p(np.maximum(y, 0.0) / np.maximum(target_scales, 1e-6)).astype(np.float32)
    return y


def build_target_scales(filtered, metadata: dict, target_names: list[str], density_scale: str = "euclidean") -> np.ndarray:
    speed = float(metadata["config"]["planner"]["speed"])
    distance = filtered["euclidean_distance"].to_numpy(dtype=np.float32)
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


def save_artifacts(
    output_dir: Path,
    model: BidCostCNN,
    x_scaler: dict[str, np.ndarray],
    y_scaler: dict[str, np.ndarray],
    image_scaler: dict[str, np.ndarray],
    feature_names: list[str],
    target_names: list[str],
    args: argparse.Namespace,
    history: list[dict[str, float]],
    metrics: dict,
    filtered,
    predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    split: dict[str, np.ndarray],
) -> None:
    torch.save(
        {
            "model_type": "cnn",
            "model_state_dict": model.state_dict(),
            "scalar_dim": len(feature_names),
            "output_dim": len(target_names),
            "image_channels": 2,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "image_pool": args.image_pool,
            "image_encoder": args.image_encoder,
            "fusion": args.fusion,
            "target_mode": args.target_mode,
            "density_scale": args.density_scale,
            "planner_cost_weights": metrics["planner_cost_weights"],
            "bid_cost_weights": metrics["bid_cost_weights"],
            "loss": args.loss,
            "huber_beta": args.huber_beta,
            "asym_under_weight": args.asym_under_weight,
            "image_size": args.image_size,
            "lateral_width": args.lateral_width,
            "corridor_scale_mode": args.corridor_scale_mode,
        },
        output_dir / "model.pt",
    )
    np.savez(
        output_dir / "scalers.npz",
        x_mean=x_scaler["x_mean"],
        x_std=x_scaler["x_std"],
        target_mean=y_scaler["mean"],
        target_std=y_scaler["std"],
        image_mean=image_scaler["mean"],
        image_std=image_scaler["std"],
    )
    (output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    (output_dir / "target_names.json").write_text(json.dumps(target_names, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "scenario_split.json").write_text(json.dumps(metrics["scenario_split"], indent=2), encoding="utf-8")
    write_history(output_dir / "history.csv", history)
    write_predictions(output_dir / "predictions.csv", filtered, predictions, baseline_predictions, target_names, split)
    plot_history(output_dir / "loss_curve.png", history)
    plot_parity(output_dir / "parity.png", filtered[target_names].to_numpy(dtype=np.float32), predictions, target_names)
    plot_sample_patches(output_dir / "corridor_patches.png", filtered, target_names, predictions)


def write_predictions(
    path: Path,
    filtered,
    predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    target_names: list[str],
    split: dict[str, np.ndarray],
) -> None:
    out = filtered[["sample_id", "scenario_id"] + target_names].copy()
    split_name = np.full(len(filtered), "unused", dtype=object)
    for name, indices in split.items():
        split_name[indices] = name
    out["split"] = split_name
    for i, target in enumerate(target_names):
        out[f"baseline_{target}"] = baseline_predictions[:, i]
        out[f"baseline_err_{target}"] = baseline_predictions[:, i] - out[target]
        out[f"pred_{target}"] = predictions[:, i]
        out[f"err_{target}"] = predictions[:, i] - out[target]
    out.to_csv(path, index=False)


def plot_sample_patches(path: Path, filtered, target_names: list[str], predictions: np.ndarray) -> None:
    # Lightweight placeholder figure: records target ranges alongside CNN artifacts.
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, target in enumerate(target_names):
        ax.scatter(filtered[target], predictions[:, i], s=20, label=target)
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    ax.set_title("CNN predictions")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
