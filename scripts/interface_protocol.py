from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def validation_risk_buffers(
    predictions_path: Path,
    quantiles: list[float],
    profile_beta: float | None = None,
) -> dict[str, float]:
    if not quantiles:
        return {}
    if not predictions_path.exists():
        raise ValueError(f"risk buffer calibration requires {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    if profile_beta is not None:
        if "profile_beta" not in predictions.columns:
            raise ValueError(f"profile-specific calibration requires profile_beta in {predictions_path}")
        predictions = predictions[np.isclose(predictions["profile_beta"].astype(float), float(profile_beta))]
    val = predictions[predictions["split"] == "val"]
    if len(val) == 0:
        raise ValueError(f"risk buffer calibration found no validation rows in {predictions_path}")
    under_residual = -val["err_risk"].to_numpy(dtype=float)
    return {
        f"buffer_q{int(round(quantile * 100)):02d}": max(0.0, float(np.quantile(under_residual, quantile)))
        for quantile in quantiles
    }


def apply_near_target_policy(
    df: pd.DataFrame,
    column: str,
    true_column: str,
    zero_threshold: float,
    oracle_threshold: float,
) -> None:
    if zero_threshold <= 0.0 and oracle_threshold <= 0.0:
        return
    distance = df["euclidean_distance"].to_numpy(dtype=float)
    values = df[column].to_numpy(dtype=float, copy=True)
    true_values = df[true_column].to_numpy(dtype=float)
    if oracle_threshold > 0.0:
        oracle_mask = distance <= oracle_threshold
        if zero_threshold > 0.0:
            oracle_mask &= distance > zero_threshold
        values[oracle_mask] = true_values[oracle_mask]
    if zero_threshold > 0.0:
        values[distance <= zero_threshold] = 0.0
    df[column] = values
