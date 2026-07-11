from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_route_portfolio_budget import compact_budget_summary


METHODS = (
    "learned_portfolio",
    "fast_only_learned",
    "balanced_only_learned",
    "safe_only_learned",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pool final static portfolio results without repair or fallback.")
    parser.add_argument("--exact-result", nargs=2, action="append", metavar=("POOL", "DIR"), required=True)
    parser.add_argument("--learned-result", nargs=2, action="append", metavar=("POOL", "DIR"), required=True)
    parser.add_argument(
        "--buffer-result",
        nargs=3,
        action="append",
        metavar=("POOL", "BUFFER", "DIR"),
        default=[],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exact_rows, exact_details = load_results(args.exact_result, "budget_assignment_results.csv", "portfolio_mode_selection.csv")
    learned_rows, learned_details = load_results(
        args.learned_result,
        "learned_budget_assignment_results.csv",
        "learned_portfolio_mode_selection.csv",
    )

    pd.DataFrame(compact_budget_summary(exact_rows.to_dict("records"), exact_details.to_dict("records"))).to_csv(
        args.output_dir / "true_portfolio_summary.csv", index=False
    )
    learned_proposal_summary(learned_rows).to_csv(args.output_dir / "learned_proposal_summary.csv", index=False)
    learned_common_quality(learned_rows).to_csv(args.output_dir / "learned_common_quality.csv", index=False)
    mode_activation_summary(learned_details).to_csv(args.output_dir / "learned_mode_activation_summary.csv", index=False)
    if args.buffer_result:
        buffer_sensitivity(args.buffer_result).to_csv(args.output_dir / "risk_buffer_sensitivity.csv", index=False)
    print(f"saved: {args.output_dir}")


def load_results(
    specs: list[list[str]],
    result_name: str,
    detail_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    details = []
    for pool, raw_dir in specs:
        directory = Path(raw_dir)
        result = pd.read_csv(directory / result_name)
        detail = pd.read_csv(directory / detail_name)
        result.insert(0, "pool", str(pool))
        detail.insert(0, "pool", str(pool))
        result["instance_id"] = result["pool"].astype(str) + ":" + result["instance_id"].astype(str)
        detail["instance_id"] = detail["pool"].astype(str) + ":" + detail["instance_id"].astype(str)
        results.append(result)
        details.append(detail)
    return pd.concat(results, ignore_index=True), pd.concat(details, ignore_index=True)


def learned_proposal_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (q, method), group in results[results["method"].isin(METHODS)].groupby(
        ["budget_quantile", "method"], sort=True
    ):
        feasible = group[group["predicted_feasible"].astype(int) == 1]
        violations = feasible[feasible["true_risk_violation"].fillna(0).astype(int) == 1]
        rows.append(
            {
                "budget_quantile": float(q),
                "method": method,
                "instances": int(len(group)),
                "predicted_feasible_rate": rate(group["predicted_feasible"]),
                "true_length_mean": mean(feasible["selected_true_length"]),
                "true_risk_mean": mean(feasible["selected_true_risk"]),
                "regret_to_true_portfolio_mean": mean(feasible["true_length_regret_to_portfolio"]),
                "true_violation_rate": rate(feasible["true_risk_violation"]),
                "mean_violation_excess": mean(violations["true_risk_violation_amount"]),
                "p95_violation_excess": quantile(violations["true_risk_violation_amount"], 0.95),
            }
        )
    return pd.DataFrame(rows)


def learned_common_quality(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for q, qdf in results.groupby("budget_quantile", sort=True):
        groups = {method: qdf[qdf["method"] == method].set_index("instance_id") for method in METHODS}
        portfolio = groups["learned_portfolio"]
        portfolio_valid = valid_index(portfolio)
        for method, group in groups.items():
            if method == "learned_portfolio":
                continue
            common = portfolio_valid.intersection(valid_index(group))
            portfolio_length = portfolio.loc[common, "selected_true_length"].astype(float)
            reference_length = group.loc[common, "selected_true_length"].astype(float)
            gains = reference_length - portfolio_length
            rows.append(
                {
                    "budget_quantile": float(q),
                    "reference_method": method,
                    "common_n": int(len(common)),
                    "common_rate": float(len(common) / len(group)) if len(group) else math.nan,
                    "portfolio_true_length_mean": mean(portfolio_length),
                    "reference_true_length_mean": mean(reference_length),
                    "portfolio_advantage": mean(gains),
                    "win_rate": rate(gains > 1e-6),
                    "tie_rate": rate(gains.abs() <= 1e-6),
                    "loss_rate": rate(gains < -1e-6),
                }
            )
    return pd.DataFrame(rows)


def mode_activation_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q, group in details.groupby("budget_quantile", sort=True):
        counts = group[["fast_edges", "balanced_edges", "safe_edges"]].astype(int)
        totals = counts.sum(axis=1)
        active = (counts > 0).sum(axis=1)
        denominator = int(totals.sum())
        rows.append(
            {
                "budget_quantile": float(q),
                "predicted_feasible_instances": int(len(group)),
                "instances_using_ge2_modes_rate": float((active >= 2).mean()),
                "instances_using_all3_modes_rate": float((active == 3).mean()),
                "mean_active_modes": float(active.mean()),
                "fast_rate": float(counts["fast_edges"].sum() / denominator),
                "balanced_rate": float(counts["balanced_edges"].sum() / denominator),
                "safe_rate": float(counts["safe_edges"].sum() / denominator),
            }
        )
    return pd.DataFrame(rows)


def buffer_sensitivity(specs: list[list[str]]) -> pd.DataFrame:
    frames = []
    for pool, label, raw_dir in specs:
        frame = pd.read_csv(Path(raw_dir) / "learned_budget_assignment_results.csv")
        frame = frame[frame["method"] == "learned_portfolio"].copy()
        frame["pool"] = str(pool)
        frame["buffer"] = str(label)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    rows = []
    for (q, label), group in data.groupby(["budget_quantile", "buffer"], sort=True):
        feasible = group[group["predicted_feasible"].astype(int) == 1]
        violations = feasible[feasible["true_risk_violation"].fillna(0).astype(int) == 1]
        rows.append(
            {
                "budget_quantile": float(q),
                "buffer": label,
                "instances": int(len(group)),
                "predicted_feasible_rate": rate(group["predicted_feasible"]),
                "true_length_mean": mean(feasible["selected_true_length"]),
                "true_violation_rate": rate(feasible["true_risk_violation"]),
                "mean_violation_excess": mean(violations["true_risk_violation_amount"]),
                "p95_violation_excess": quantile(violations["true_risk_violation_amount"], 0.95),
            }
        )
    return pd.DataFrame(rows)


def mean(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else math.nan


def quantile(values: pd.Series, q: float) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q)) if len(array) else math.nan


def rate(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else math.nan


def valid_index(group: pd.DataFrame) -> pd.Index:
    return group.index[
        (group["predicted_feasible"].astype(int) == 1)
        & (group["true_risk_violation"].fillna(0).astype(int) == 0)
    ]


if __name__ == "__main__":
    main()
