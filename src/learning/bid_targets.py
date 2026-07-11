from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def planner_cost_weights(metadata: dict) -> tuple[float, float, float]:
    planner = metadata["config"]["planner"]
    alpha = float(planner["alpha"])
    beta = float(planner["beta"])
    gamma = float(planner["gamma"])
    return alpha, beta, gamma


def bid_cost_weights(metadata: dict) -> tuple[float, float, float]:
    alpha, beta, _ = planner_cost_weights(metadata)
    return alpha, beta, 0.0


def add_bid_target_if_needed(
    df: pd.DataFrame,
    target_names: Iterable[str],
    metadata: dict,
) -> None:
    if "bid" not in target_names:
        return
    alpha, beta, _ = bid_cost_weights(metadata)
    df["bid"] = alpha * df["length"].astype(float) + beta * df["risk"].astype(float)
    df["baseline_bid"] = baseline_bid(df, metadata)


def baseline_bid(df: pd.DataFrame, metadata: dict) -> np.ndarray:
    alpha, beta, _ = bid_cost_weights(metadata)
    return (
        alpha * df["euclidean_distance"].astype(float).to_numpy()
        + beta * df["straight_line_risk"].astype(float).to_numpy()
    )


def true_bid_from_planner_metrics(metrics: dict, metadata: dict) -> float:
    alpha, beta, _ = bid_cost_weights(metadata)
    return alpha * float(metrics["length"]) + beta * float(metrics["risk"])
