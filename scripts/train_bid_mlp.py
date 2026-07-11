from __future__ import annotations

import argparse
import csv
import json
import math
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

from src.learning.features import TARGET_COLUMNS, feasible_regression_data, load_dataset_table
from src.learning.bid_targets import (
    add_bid_target_if_needed as add_planner_bid_target_if_needed,
    bid_cost_weights,
    planner_cost_weights,
)
from src.learning.losses import (
    AsymmetricL1Loss,
    AsymmetricSmoothL1Loss,
)
from src.learning.mlp import BidCostMLP, BidCostScalarFusionMLP


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a scalar-feature MLP for learned UAV bid costs.")
    parser.add_argument("dataset_dir", type=Path, help="Dataset directory containing samples.csv and metadata.json.")
    parser.add_argument("--output", type=Path, default=None, help="Training output directory.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-mode", choices=["direct", "residual", "density", "residual_density"], default="residual")
    parser.add_argument(
        "--density-scale",
        choices=["euclidean", "effective_euclidean"],
        default="euclidean",
        help="Scale used by density target mode. effective_euclidean subtracts planner goal_tolerance.",
    )
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--architecture", choices=["standard", "cnn_scalar"], default="standard")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--targets", nargs="+", default=["length", "risk"])
    parser.add_argument("--loss", choices=["mse", "mae", "smooth_l1", "asymmetric_mae", "asymmetric_smooth_l1"], default="mse")
    parser.add_argument("--huber-beta", type=float, default=1.0, help="Beta parameter for SmoothL1 loss on scaled targets.")
    parser.add_argument("--asym-under-weight", type=float, default=2.0, help="Weight applied to underestimation in asymmetric SmoothL1.")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = args.output or args.dataset_dir / "models" / "bid_mlp"
    output_dir.mkdir(parents=True, exist_ok=True)

    df, metadata = load_dataset_table(args.dataset_dir)
    add_bid_target_if_needed(df, args.targets, metadata)
    x, y, feature_names, target_names, filtered = feasible_regression_data(df, metadata, args.targets)
    split = train_val_split(filtered)

    scaler = fit_scalers(x[split["train"]], y[split["train"]])
    baseline_predictions = build_baseline_predictions(filtered, metadata, target_names)
    target_scales = build_target_scales(filtered, metadata, target_names, args.density_scale)
    train_targets = transform_targets(y, baseline_predictions, target_scales, args.target_mode)
    x_scaled = apply_scaler(x, scaler["x_mean"], scaler["x_std"])
    y_scaler = fit_target_scaler(train_targets[split["train"]])
    y_scaled = apply_scaler(train_targets, y_scaler["mean"], y_scaler["std"])
    scaler["target_mean"] = y_scaler["mean"]
    scaler["target_std"] = y_scaler["std"]

    train_loader = make_loader(
        x_scaled,
        y_scaled,
        split["train"],
        args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        x_scaled,
        y_scaled,
        split["val"],
        args.batch_size,
        shuffle=False,
    )

    if args.architecture == "cnn_scalar":
        model = BidCostScalarFusionMLP(
            input_dim=x.shape[1],
            output_dim=y.shape[1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    else:
        model = BidCostMLP(
            input_dim=x.shape[1],
            output_dim=y.shape[1],
            hidden_dims=tuple(args.hidden),
            dropout=args.dropout,
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
    predictions = predict(model, x_scaled, scaler, baseline_predictions, target_scales, args.target_mode)
    metrics = compute_metrics(y, predictions, target_names, split)
    metrics["baselines"] = {
        "straight_line": compute_prediction_metrics(y, baseline_predictions, target_names, split)
    }
    metrics["target_mode"] = args.target_mode
    metrics["planner_cost_weights"] = dict(zip(["alpha", "beta", "gamma"], planner_cost_weights(metadata)))
    metrics["bid_cost_weights"] = dict(zip(["alpha", "beta", "gamma"], bid_cost_weights(metadata)))
    metrics["loss"] = args.loss
    metrics["huber_beta"] = args.huber_beta
    metrics["asym_under_weight"] = args.asym_under_weight
    metrics["scenario_split"] = scenario_split_summary(filtered, split)

    save_artifacts(
        output_dir,
        model,
        scaler,
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


def build_loss(loss_name: str, huber_beta: float, asym_under_weight: float = 2.0) -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss(beta=float(huber_beta))
    if loss_name == "asymmetric_mae":
        return AsymmetricL1Loss(under_weight=float(asym_under_weight))
    if loss_name == "asymmetric_smooth_l1":
        return AsymmetricSmoothL1Loss(beta=float(huber_beta), under_weight=float(asym_under_weight))
    raise ValueError(f"Unsupported loss: {loss_name}")


def train_val_split(filtered) -> dict[str, np.ndarray]:
    if "split" not in filtered.columns:
        raise ValueError("samples.csv must contain the fixed map-disjoint split column")
    split_values = filtered["split"].astype(str).str.lower()
    train_idx = np.flatnonzero(split_values == "train")
    val = np.flatnonzero(split_values == "val")
    test = np.flatnonzero(split_values == "test")
    if len(train_idx) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError("fixed split column must contain non-empty train, val, and test rows")
    return {"train": train_idx.astype(int), "val": val.astype(int), "test": test.astype(int)}


def scenario_split_summary(filtered, split: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for split_name, indices in split.items():
        scenario_ids = sorted({int(item) for item in filtered.iloc[indices]["scenario_id"].tolist()})
        summary[split_name] = {
            "sample_count": int(len(indices)),
            "scenario_count": int(len(scenario_ids)),
            "scenario_ids": scenario_ids,
        }
    return summary


def fit_scalers(x_train: np.ndarray, y_train: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "x_mean": x_train.mean(axis=0, keepdims=True),
        "x_std": x_train.std(axis=0, keepdims=True) + 1e-6,
    }


def fit_target_scaler(y_train: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": y_train.mean(axis=0, keepdims=True),
        "std": y_train.std(axis=0, keepdims=True) + 1e-6,
    }


def apply_scaler(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = torch.tensor(x[indices], dtype=torch.float32)
    y_tensor = torch.tensor(y[indices], dtype=torch.float32)
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=min(batch_size, len(indices)), shuffle=shuffle)


def train(
    model: BidCostMLP,
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
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))

        val_loss = evaluate_scaled_loss(model, loss_fn, val_loader)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
            }
        )
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
    model: BidCostMLP,
    loss_fn: nn.Module,
    loader: DataLoader,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            losses.append(float(loss))
    return float(np.mean(losses)) if losses else math.nan


def predict(
    model: BidCostMLP,
    x_scaled: np.ndarray,
    scaler: dict[str, np.ndarray],
    baseline_predictions: np.ndarray,
    target_scales: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(x_scaled, dtype=torch.float32)).numpy()
    pred_target = pred_scaled * scaler["target_std"] + scaler["target_mean"]
    if target_mode == "residual":
        return baseline_predictions + pred_target
    if target_mode == "residual_density":
        return baseline_predictions + target_scales * pred_target
    if target_mode == "density":
        return np.maximum(0.0, target_scales * pred_target)
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
    return y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str], split: dict[str, np.ndarray]) -> dict:
    metrics = {"n_train": int(len(split["train"])), "n_val": int(len(split["val"])), "targets": {}}
    metrics["targets"] = compute_prediction_metrics(y_true, y_pred, target_names, split)
    return metrics


def compute_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str], split: dict[str, np.ndarray]) -> dict:
    metrics = {}
    for name, col in zip(target_names, range(y_true.shape[1])):
        target_metrics = {}
        for split_name, indices in split.items():
            true = y_true[indices, col]
            pred = y_pred[indices, col]
            err = pred - true
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))
            rel_mae = float(np.mean(np.abs(err) / np.maximum(np.abs(true), 1e-6)))
            target_metrics[split_name] = {"mae": mae, "rmse": rmse, "rel_mae": rel_mae}
        metrics[name] = target_metrics
    return metrics


def build_baseline_predictions(filtered, metadata: dict, target_names: list[str]) -> np.ndarray:
    speed = float(metadata["config"]["planner"]["speed"])
    columns = []
    for target in target_names:
        if target == "length":
            columns.append(filtered["euclidean_distance"].to_numpy(dtype=np.float32))
        elif target == "time":
            columns.append((filtered["euclidean_distance"] / speed).to_numpy(dtype=np.float32))
        elif target == "risk":
            columns.append(filtered["straight_line_risk"].to_numpy(dtype=np.float32))
        elif target == "bid":
            columns.append(filtered["baseline_bid"].to_numpy(dtype=np.float32))
        else:
            raise ValueError(f"Unsupported target for baseline prediction: {target}")
    return np.column_stack(columns).astype(np.float32)


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


def add_bid_target_if_needed(
    df,
    target_names: list[str],
    metadata: dict,
) -> None:
    add_planner_bid_target_if_needed(df, target_names, metadata)


def save_artifacts(
    output_dir: Path,
    model: BidCostMLP,
    scaler: dict[str, np.ndarray],
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
            "model_type": "mlp",
            "model_state_dict": model.state_dict(),
            "input_dim": len(feature_names),
            "output_dim": len(target_names),
            "architecture": args.architecture,
            "hidden_dims": tuple(args.hidden),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "target_mode": args.target_mode,
            "density_scale": args.density_scale,
            "planner_cost_weights": metrics["planner_cost_weights"],
            "bid_cost_weights": metrics["bid_cost_weights"],
            "loss": args.loss,
            "huber_beta": args.huber_beta,
            "asym_under_weight": args.asym_under_weight,
        },
        output_dir / "model.pt",
    )
    np.savez(
        output_dir / "scaler.npz",
        x_mean=scaler["x_mean"],
        x_std=scaler["x_std"],
        target_mean=scaler["target_mean"],
        target_std=scaler["target_std"],
    )
    (output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    (output_dir / "target_names.json").write_text(json.dumps(target_names, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "scenario_split.json").write_text(json.dumps(metrics["scenario_split"], indent=2), encoding="utf-8")
    write_history(output_dir / "history.csv", history)
    write_predictions(output_dir / "predictions.csv", filtered, predictions, baseline_predictions, target_names, split)
    plot_history(output_dir / "loss_curve.png", history)
    plot_parity(output_dir / "parity.png", filtered[target_names].to_numpy(dtype=np.float32), predictions, target_names)


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)


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


def plot_history(path: Path, history: list[dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = [item["epoch"] for item in history]
    ax.plot(epochs, [item["train_loss"] for item in history], label="train")
    ax.plot(epochs, [item["val_loss"] for item in history], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("scaled loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_parity(path: Path, y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> None:
    n = len(target_names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, name, col in zip(axes, target_names, range(n)):
        true = y_true[:, col]
        pred = y_pred[:, col]
        ax.scatter(true, pred, s=30, c="#2563eb", edgecolors="black", linewidths=0.35)
        lo = float(min(true.min(), pred.min()))
        hi = float(max(true.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel(f"true {name}")
        ax.set_ylabel(f"pred {name}")
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_metrics(metrics: dict) -> None:
    print(f"train/val: {metrics['n_train']}/{metrics['n_val']}")
    for target, target_metrics in metrics["targets"].items():
        val = target_metrics["val"]
        baseline = metrics["baselines"]["straight_line"].get(target, {}).get("val", {})
        baseline_text = ""
        if baseline:
            baseline_text = f", straight_line_val_mae={baseline['mae']:.4f}"
        print(
            f"{target}: val_mae={val['mae']:.4f}, val_rmse={val['rmse']:.4f}, "
            f"val_rel_mae={val['rel_mae']:.3f}{baseline_text}"
        )


if __name__ == "__main__":
    main()
