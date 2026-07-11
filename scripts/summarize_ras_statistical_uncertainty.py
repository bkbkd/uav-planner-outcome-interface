from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHOD_PORTFOLIO = "learned_portfolio"
METHOD_BALANCED = "balanced_only_learned"
METHOD_FAST = "fast_only_learned"
METHOD_SAFE = "safe_only_learned"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired bootstrap uncertainty summaries for the RAS interface paper "
            "from already-generated static and rolling evaluation CSVs."
        )
    )
    parser.add_argument("--static-result", nargs=2, action="append", metavar=("POOL", "DIR"), required=True)
    parser.add_argument("--rolling-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    static_pairs = build_static_pairs(args.static_result)
    static_summary = summarize_static(static_pairs, rng, args.bootstrap)
    rolling_pairs = build_rolling_pairs(args.rolling_dir)
    rolling_summary = summarize_rolling(rolling_pairs, rng, args.bootstrap)

    write_csv(static_pairs, args.output_dir / "static_paired_differences.csv")
    write_csv(static_summary, args.output_dir / "static_paired_bootstrap_summary.csv")
    write_csv(rolling_pairs, args.output_dir / "rolling_paired_differences.csv")
    write_csv(rolling_summary, args.output_dir / "rolling_paired_bootstrap_summary.csv")
    (args.output_dir / "statistical_uncertainty_summary.json").write_text(
        json.dumps(
            {
                "bootstrap": args.bootstrap,
                "seed": args.seed,
                "static_rows": int(len(static_pairs)),
                "rolling_rows": int(len(rolling_pairs)),
                "outputs": {
                    "static_paired_bootstrap_summary": str(args.output_dir / "static_paired_bootstrap_summary.csv"),
                    "rolling_paired_bootstrap_summary": str(args.output_dir / "rolling_paired_bootstrap_summary.csv"),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "static_rows": len(static_pairs), "rolling_rows": len(rolling_pairs)}, indent=2))


def build_static_pairs(static_results: list[list[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pool, raw_dir in static_results:
        results = pd.read_csv(Path(raw_dir) / "learned_budget_assignment_results.csv")
        methods_available = set(results["method"].astype(str))
        for q in sorted(results["budget_quantile"].astype(float).unique()):
            methods = {"portfolio": METHOD_PORTFOLIO, "balanced": METHOD_BALANCED, "fast": METHOD_FAST, "safe": METHOD_SAFE}
            missing = [method for method in methods.values() if method not in methods_available]
            if missing:
                raise ValueError(f"{pool} q={q}: missing methods {missing}")
            groups = {
                label: select_method(results, q, method).set_index("instance_id")
                for label, method in methods.items()
            }
            portfolio_good = valid_index(groups["portfolio"])
            for reference in ("balanced", "fast", "safe"):
                common_index = portfolio_good.intersection(valid_index(groups[reference]))
                portfolio = groups["portfolio"].loc[common_index, "selected_true_length"].astype(float)
                oracle = groups["portfolio"].loc[common_index, "oracle_true_length"].astype(float)
                for instance_id in common_index:
                    p_len = float(portfolio.loc[instance_id])
                    rows.append(
                        {
                            "pool": pool,
                            "budget_quantile": float(q),
                            "comparison": f"portfolio_vs_{reference}",
                            "instance_id": int(instance_id),
                            "portfolio_regret": p_len - float(oracle.loc[instance_id]),
                            "portfolio_gain": float(groups[reference].loc[instance_id, "selected_true_length"]) - p_len,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_static(pairs: pd.DataFrame, rng: np.random.Generator, bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "portfolio_regret",
        "portfolio_gain",
    ]
    for (q, comparison), group in pairs.groupby(["budget_quantile", "comparison"], sort=True):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "budget_quantile": float(q),
                    "comparison": comparison,
                    "metric": metric,
                    **summary_stats(values, rng, bootstrap),
                }
            )
    return pd.DataFrame(rows)


def build_rolling_pairs(rolling_dir: Path) -> pd.DataFrame:
    missions = pd.read_csv(rolling_dir / "lazy_rolling_missions.csv")
    keys = ["minimum_mean_route_survival", "mission_id", "seed", "local_mission_id", "scenario_id"]
    pivot = missions.pivot_table(index=keys, columns="method", values=["mission_true_length", "completion_rate"], aggfunc="first")
    pivot.columns = [f"{left}:{right}" for left, right in pivot.columns]
    pivot = pivot.reset_index()
    references = ["fast_only", "balanced_only", "safe_only"]
    required_methods = ["learned_portfolio", *references]
    for method in required_methods:
        if f"mission_true_length:{method}" not in pivot.columns:
            raise ValueError(f"rolling results missing {method}")
    rows: list[dict[str, Any]] = []
    portfolio_complete = pivot["completion_rate:learned_portfolio"].astype(float).to_numpy() >= 1.0 - 1e-12
    for reference in references:
        reference_complete = pivot[f"completion_rate:{reference}"].astype(float).to_numpy() >= 1.0 - 1e-12
        subset = pivot[portfolio_complete & reference_complete].copy()
        reference_length = subset[f"mission_true_length:{reference}"].astype(float).to_numpy()
        portfolio_length = subset["mission_true_length:learned_portfolio"].astype(float).to_numpy()
        for idx, item in enumerate(subset.itertuples(index=False)):
            rows.append(
                {
                    "minimum_mean_route_survival": float(item.minimum_mean_route_survival),
                    "comparison": f"portfolio_vs_{reference}",
                    "mission_id": int(item.mission_id),
                    "seed": int(item.seed),
                    "local_mission_id": int(item.local_mission_id),
                    "portfolio_gain": float(reference_length[idx] - portfolio_length[idx]),
                }
            )
    return pd.DataFrame(rows)


def summarize_rolling(pairs: pd.DataFrame, rng: np.random.Generator, bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (survival, comparison), group in pairs.groupby(["minimum_mean_route_survival", "comparison"], sort=True):
        for metric in ["portfolio_gain"]:
            values = group[metric].to_numpy(dtype=float)
            row = {
                "minimum_mean_route_survival": float(survival),
                "comparison": str(comparison),
                "metric": metric,
                **summary_stats(values, rng, bootstrap),
            }
            row["win_rate"] = float(np.mean(values > 1e-6)) if len(values) else math.nan
            row["tie_rate"] = float(np.mean(np.abs(values) <= 1e-6)) if len(values) else math.nan
            row["loss_rate"] = float(np.mean(values < -1e-6)) if len(values) else math.nan
            rows.append(row)
    return pd.DataFrame(rows)


def summary_stats(values: np.ndarray, rng: np.random.Generator, bootstrap: int) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0, "mean": math.nan, "median": math.nan, "ci95_low": math.nan, "ci95_high": math.nan, "p10": math.nan, "p90": math.nan}
    boot = np.empty(bootstrap, dtype=float)
    for idx in range(bootstrap):
        sample_idx = rng.integers(0, len(values), size=len(values))
        boot[idx] = float(np.mean(values[sample_idx]))
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def select_method(results: pd.DataFrame, quantile: float, method: str) -> pd.DataFrame:
    return results[(results["budget_quantile"].astype(float) == float(quantile)) & (results["method"].astype(str) == method)].copy()


def valid_index(group: pd.DataFrame) -> pd.Index:
    return group.index[
        (group["predicted_feasible"].astype(int) == 1)
        & (group["true_risk_violation"].fillna(0).astype(int) == 0)
    ]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
