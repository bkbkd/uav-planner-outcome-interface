from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a direct scalar-bid CNN with bid reconstructed from a semantic length-risk CNN. "
            "All inputs are frozen model predictions or scalar-assignment evaluation outputs."
        )
    )
    parser.add_argument(
        "--edge-result",
        nargs=4,
        action="append",
        metavar=("PROFILE", "BETA", "SEMANTIC_MODEL", "DIRECT_BID_MODEL"),
        required=True,
    )
    parser.add_argument(
        "--assignment-result",
        nargs=4,
        action="append",
        metavar=("PROFILE", "POOL", "SEMANTIC_EVAL", "DIRECT_BID_EVAL"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    edge_pairs, edge_summary = summarize_edges(args.edge_result)
    assignment_pairs, assignment_summary = summarize_assignments(args.assignment_result)
    edge_pairs.to_csv(args.output_dir / "bid_reuse_edge_pairs.csv", index=False)
    edge_summary.to_csv(args.output_dir / "bid_reuse_edge_summary.csv", index=False)
    assignment_pairs.to_csv(args.output_dir / "bid_reuse_assignment_pairs.csv", index=False)
    assignment_summary.to_csv(args.output_dir / "bid_reuse_assignment_summary.csv", index=False)
    print(f"saved: {args.output_dir}")


def summarize_edges(specs: list[list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_frames = []
    summary_rows: list[dict[str, Any]] = []
    for profile, raw_beta, semantic_dir, direct_dir in specs:
        beta = float(raw_beta)
        semantic = pd.read_csv(Path(semantic_dir) / "predictions.csv")
        direct = pd.read_csv(Path(direct_dir) / "predictions.csv")
        if "profile_beta" in semantic.columns:
            semantic = semantic[np.isclose(semantic["profile_beta"].astype(float), beta)].copy()
        require_columns(semantic, {"sample_id", "scenario_id", "split", "length", "risk", "pred_length", "pred_risk", "baseline_length", "baseline_risk"}, semantic_dir)
        require_columns(direct, {"sample_id", "scenario_id", "split", "bid", "pred_bid", "baseline_bid"}, direct_dir)
        semantic = semantic[semantic["split"].astype(str) == "test"].copy()
        direct = direct[direct["split"].astype(str) == "test"].copy()
        keys = ["sample_id", "scenario_id", "split"]
        merged = semantic.merge(direct, on=keys, how="inner", validate="one_to_one", suffixes=("_semantic", "_direct"))
        if len(merged) != len(semantic) or len(merged) != len(direct):
            raise ValueError(f"{profile}: semantic and direct-bid test rows are not exactly paired")

        true_bid = merged["length"].astype(float) + beta * merged["risk"].astype(float)
        derived_bid = merged["pred_length"].astype(float) + beta * merged["pred_risk"].astype(float)
        analytic_bid = merged["baseline_length"].astype(float) + beta * merged["baseline_risk"].astype(float)
        assert_close(true_bid, merged["bid"], f"{profile} true bid")
        assert_close(analytic_bid, merged["baseline_bid"], f"{profile} analytic bid")
        frame = pd.DataFrame(
            {
                "profile": profile,
                "beta": beta,
                "sample_id": merged["sample_id"].astype(int),
                "scenario_id": merged["scenario_id"].astype(int),
                "true_bid": true_bid,
                "analytic_bid": analytic_bid,
                "direct_bid": merged["pred_bid"].astype(float),
                "semantic_derived_bid": derived_bid,
            }
        )
        frame["analytic_abs_error"] = (frame["analytic_bid"] - frame["true_bid"]).abs()
        frame["direct_abs_error"] = (frame["direct_bid"] - frame["true_bid"]).abs()
        frame["semantic_abs_error"] = (frame["semantic_derived_bid"] - frame["true_bid"]).abs()
        pair_frames.append(frame)
        summary_rows.extend(edge_metric_rows(profile, beta, frame))

    pairs = pd.concat(pair_frames, ignore_index=True)
    summary_rows.extend(edge_metric_rows("all_profiles", math.nan, pairs))
    return pairs, pd.DataFrame(summary_rows)


def edge_metric_rows(profile: str, beta: float, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for method, column in (
        ("analytic", "analytic_bid"),
        ("direct_bid_cnn", "direct_bid"),
        ("semantic_derived_bid", "semantic_derived_bid"),
    ):
        true = frame["true_bid"].to_numpy(dtype=float)
        pred = frame[column].to_numpy(dtype=float)
        error = pred - true
        rows.append(
            {
                "profile": profile,
                "beta": beta,
                "method": method,
                "edges": int(len(frame)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
            }
        )
    direct = frame["direct_abs_error"].to_numpy(dtype=float)
    semantic = frame["semantic_abs_error"].to_numpy(dtype=float)
    rows.append(
        {
            "profile": profile,
            "beta": beta,
            "method": "semantic_minus_direct",
            "edges": int(len(frame)),
            "mae": float(np.mean(semantic) - np.mean(direct)),
            "rmse": math.nan,
            "bias": math.nan,
            "semantic_better_rate": float(np.mean(semantic < direct - 1e-9)),
            "tie_rate": float(np.mean(np.abs(semantic - direct) <= 1e-9)),
            "semantic_worse_rate": float(np.mean(semantic > direct + 1e-9)),
        }
    )
    return rows


def summarize_assignments(specs: list[list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_frames = []
    for profile, pool, semantic_dir, direct_dir in specs:
        semantic = pd.read_csv(Path(semantic_dir) / "assignment_results.csv")
        direct = pd.read_csv(Path(direct_dir) / "assignment_results.csv")
        columns = {"instance_id", "oracle_true_cost", "baseline_true_cost", "learned_true_cost", "baseline_regret", "learned_regret", "learned_matches_oracle"}
        require_columns(semantic, columns, semantic_dir)
        require_columns(direct, columns, direct_dir)
        merged = semantic.merge(direct, on="instance_id", how="inner", validate="one_to_one", suffixes=("_semantic", "_direct"))
        if len(merged) != len(semantic) or len(merged) != len(direct):
            raise ValueError(f"{profile}/{pool}: semantic and direct-bid assignment rows are not exactly paired")
        assert_close(merged["oracle_true_cost_semantic"], merged["oracle_true_cost_direct"], f"{profile}/{pool} oracle")
        assert_close(merged["baseline_true_cost_semantic"], merged["baseline_true_cost_direct"], f"{profile}/{pool} analytic")
        frame = pd.DataFrame(
            {
                "profile": profile,
                "pool": pool,
                "instance_id": merged["instance_id"].astype(int),
                "oracle_true_cost": merged["oracle_true_cost_semantic"].astype(float),
                "analytic_true_cost": merged["baseline_true_cost_semantic"].astype(float),
                "semantic_true_cost": merged["learned_true_cost_semantic"].astype(float),
                "direct_true_cost": merged["learned_true_cost_direct"].astype(float),
                "analytic_regret": merged["baseline_regret_semantic"].astype(float),
                "semantic_regret": merged["learned_regret_semantic"].astype(float),
                "direct_regret": merged["learned_regret_direct"].astype(float),
                "semantic_oracle_match": merged["learned_matches_oracle_semantic"].astype(int),
                "direct_oracle_match": merged["learned_matches_oracle_direct"].astype(int),
            }
        )
        frame["direct_minus_semantic_true_cost"] = frame["direct_true_cost"] - frame["semantic_true_cost"]
        pair_frames.append(frame)

    pairs = pd.concat(pair_frames, ignore_index=True)
    rows = []
    for profile, group in list(pairs.groupby("profile", sort=True)) + [("all_profiles", pairs)]:
        delta = group["direct_minus_semantic_true_cost"].to_numpy(dtype=float)
        rows.append(
            {
                "profile": profile,
                "instances": int(len(group)),
                "analytic_regret_mean": float(group["analytic_regret"].mean()),
                "direct_regret_mean": float(group["direct_regret"].mean()),
                "semantic_regret_mean": float(group["semantic_regret"].mean()),
                "semantic_minus_direct_regret_mean": float(group["semantic_regret"].mean() - group["direct_regret"].mean()),
                "direct_oracle_match_rate": float(group["direct_oracle_match"].mean()),
                "semantic_oracle_match_rate": float(group["semantic_oracle_match"].mean()),
                "semantic_win_rate": float(np.mean(delta > 1e-9)),
                "tie_rate": float(np.mean(np.abs(delta) <= 1e-9)),
                "semantic_loss_rate": float(np.mean(delta < -1e-9)),
            }
        )
    return pairs, pd.DataFrame(rows)


def require_columns(frame: pd.DataFrame, required: set[str], source: str | Path) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def assert_close(left: pd.Series, right: pd.Series, label: str, tolerance: float = 1e-3) -> None:
    difference = np.abs(left.to_numpy(dtype=float) - right.to_numpy(dtype=float))
    maximum = float(difference.max()) if len(difference) else 0.0
    if maximum > tolerance:
        raise ValueError(f"{label} semantics differ; max_abs_diff={maximum:.6g}")


if __name__ == "__main__":
    main()
