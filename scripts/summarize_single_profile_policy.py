from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROFILE_METHODS = {
    "fast": "fast_only_learned",
    "balanced": "balanced_only_learned",
    "safe": "safe_only_learned",
}
PORTFOLIO_METHOD = "learned_portfolio"
BALANCED_METHOD = "balanced_only_learned"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select deployable single-profile policies on a development pool and evaluate them on held-out pools."
    )
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--test-result", nargs=2, action="append", metavar=("POOL", "DIR"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv(args.development_result / "learned_budget_assignment_results.csv")
    policy = select_policy(dev)
    test_rows: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    for pool, raw_dir in args.test_result:
        results = pd.read_csv(Path(raw_dir) / "learned_budget_assignment_results.csv")
        test_rows.extend(population_rows(pool, results, policy))
        common_rows.extend(common_quality_rows(pool, results, policy))

    population = pd.DataFrame(test_rows)
    common = pd.DataFrame(common_rows)
    population_main = aggregate_population(population)
    common_main = aggregate_common(common)

    pd.DataFrame(policy["selection_rows"]).to_csv(args.output_dir / "single_profile_policy_selection.csv", index=False)
    population.to_csv(args.output_dir / "single_profile_policy_population_per_pool.csv", index=False)
    population_main.to_csv(args.output_dir / "single_profile_policy_population_main.csv", index=False)
    common.to_csv(args.output_dir / "single_profile_policy_common_per_pool.csv", index=False)
    common_main.to_csv(args.output_dir / "single_profile_policy_common_main.csv", index=False)
    (args.output_dir / "single_profile_policy_summary.json").write_text(
        json.dumps(
            {
                "development_result": str(args.development_result),
                "selection_metric": "proposal_regret_on_common_predicted_feasible_instances",
                "global_profile": policy["global_profile"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "global_profile": policy["global_profile"]}, indent=2))


def select_policy(dev: pd.DataFrame) -> dict[str, Any]:
    stats = []
    for q in sorted(dev["budget_quantile"].astype(float).unique()):
        qdf = dev[dev["budget_quantile"].astype(float) == q]
        groups = {
            profile: qdf[qdf["method"].astype(str) == method].set_index("instance_id")
            for profile, method in PROFILE_METHODS.items()
        }
        common_index = None
        for group in groups.values():
            feasible = group.index[group["predicted_feasible"].astype(int) == 1]
            common_index = feasible if common_index is None else common_index.intersection(feasible)
        assert common_index is not None
        for profile, method in PROFILE_METHODS.items():
            full_group = groups[profile]
            group = full_group.loc[common_index]
            stats.append(
                {
                    "budget_quantile": float(q),
                    "profile": profile,
                    "method": method,
                    "development_common_n": int(len(common_index)),
                    "predicted_feasible_rate": float(full_group["predicted_feasible"].astype(int).mean()),
                    "true_violation_rate": float(group["true_risk_violation"].fillna(0).astype(int).mean()),
                    "common_proposal_true_length": float(group["selected_true_length"].astype(float).mean()),
                    "common_proposal_regret": float(group["true_length_regret_to_portfolio"].astype(float).mean()),
                }
            )
    table = pd.DataFrame(stats)
    global_profile = (
        table.groupby("profile")["common_proposal_regret"]
        .mean()
        .sort_values()
        .index[0]
    )
    selection_rows = []
    for q in sorted(table["budget_quantile"].astype(float).unique()):
        row = table[(table["budget_quantile"].astype(float) == q) & (table["profile"].astype(str) == str(global_profile))].iloc[0]
        selection_rows.append(
            {
                "policy": "global_fixed",
                "budget_quantile": float(q),
                "selected_profile": str(global_profile),
                "selected_method": PROFILE_METHODS[str(global_profile)],
                "development_common_n": int(row["development_common_n"]),
                "development_predicted_feasible_rate": float(row["predicted_feasible_rate"]),
                "development_true_violation_rate": float(row["true_violation_rate"]),
                "development_proposal_length": float(row["common_proposal_true_length"]),
                "development_proposal_regret": float(row["common_proposal_regret"]),
            }
        )
    return {
        "global_profile": str(global_profile),
        "selection_rows": selection_rows,
    }


def population_rows(pool: str, results: pd.DataFrame, policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for q in sorted(results["budget_quantile"].astype(float).unique()):
        policies = {
            "portfolio": PORTFOLIO_METHOD,
            "global_fixed_single": PROFILE_METHODS[policy["global_profile"]],
            "balanced_only": BALANCED_METHOD,
        }
        for label, method in policies.items():
            group = select_method(results, q, method)
            rows.append(
                {
                    "pool": pool,
                    "budget_quantile": float(q),
                    "policy": label,
                    "selected_profile": profile_for_method(method),
                    "instances": int(len(group)),
                    **proposal_stats(group),
                }
            )
    return rows


def common_quality_rows(pool: str, results: pd.DataFrame, policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for q in sorted(results["budget_quantile"].astype(float).unique()):
        references = {
            "global_fixed_single": PROFILE_METHODS[policy["global_profile"]],
            "balanced_only": BALANCED_METHOD,
        }
        portfolio = select_method(results, q, PORTFOLIO_METHOD).set_index("instance_id")
        portfolio_good = valid_proposal_index(portfolio)
        for label, method in references.items():
            reference = select_method(results, q, method).set_index("instance_id")
            common_index = portfolio_good.intersection(valid_proposal_index(reference))
            portfolio_length = portfolio.loc[common_index, "selected_true_length"].astype(float)
            reference_length = reference.loc[common_index, "selected_true_length"].astype(float)
            gains = reference_length - portfolio_length
            rows.append(
                {
                    "pool": pool,
                    "budget_quantile": float(q),
                    "reference_policy": label,
                    "selected_profile": profile_for_method(method),
                    "common_feasible_n": int(len(common_index)),
                    "instances": int(len(portfolio)),
                    "common_feasible_rate": float(len(common_index) / len(portfolio)) if len(portfolio) else math.nan,
                    "portfolio_true_length_mean": float(portfolio_length.mean()) if len(common_index) else math.nan,
                    "reference_true_length_mean": float(reference_length.mean()) if len(common_index) else math.nan,
                    "portfolio_advantage": float(gains.mean()) if len(common_index) else math.nan,
                    "win_rate": float((gains > 1e-6).mean()) if len(common_index) else math.nan,
                    "tie_rate": float((gains.abs() <= 1e-6).mean()) if len(common_index) else math.nan,
                    "loss_rate": float((gains < -1e-6).mean()) if len(common_index) else math.nan,
                }
            )
    return rows


def aggregate_population(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(["budget_quantile", "policy", "selected_profile"], dropna=False):
        weights = group["instances"].to_numpy(dtype=float)
        feasible_weights = group["predicted_feasible_n"].to_numpy(dtype=float)
        rows.append(
            {
                "budget_quantile": float(keys[0]),
                "policy": keys[1],
                "selected_profile": keys[2],
                "instances": int(weights.sum()),
                "predicted_feasible_n": int(feasible_weights.sum()),
                "predicted_feasible_rate": weighted_mean(group["predicted_feasible_rate"], weights),
                "true_violation_rate": weighted_mean(group["true_violation_rate"], feasible_weights),
                "proposal_true_length": weighted_mean(group["proposal_true_length"], feasible_weights),
                "proposal_regret_to_true_portfolio": weighted_mean(group["proposal_regret_to_true_portfolio"], feasible_weights),
            }
        )
    grouped = pd.DataFrame(rows)
    return add_portfolio_advantage(grouped, "proposal_regret_to_true_portfolio")


def aggregate_common(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(["budget_quantile", "reference_policy", "selected_profile"], dropna=False):
        weights = group["common_feasible_n"].to_numpy(dtype=float)
        instances = int(group["instances"].astype(int).sum())
        common_n = int(weights.sum())
        rows.append(
            {
                "budget_quantile": float(keys[0]),
                "reference_policy": keys[1],
                "selected_profile": keys[2],
                "common_feasible_n": common_n,
                "common_feasible_rate": common_n / instances if instances else math.nan,
                "portfolio_true_length_mean": weighted_mean(group["portfolio_true_length_mean"], weights),
                "reference_true_length_mean": weighted_mean(group["reference_true_length_mean"], weights),
                "portfolio_advantage": weighted_mean(group["portfolio_advantage"], weights),
                "win_rate": weighted_mean(group["win_rate"], weights),
                "tie_rate": weighted_mean(group["tie_rate"], weights),
                "loss_rate": weighted_mean(group["loss_rate"], weights),
            }
        )
    return pd.DataFrame(rows)


def add_portfolio_advantage(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = []
    for q, group in df.groupby("budget_quantile", sort=True):
        portfolio_metric = float(group[group["policy"].astype(str) == "portfolio"][metric].iloc[0])
        for row in group.to_dict("records"):
            row["portfolio_advantage"] = float(row[metric]) - portfolio_metric
            out.append(row)
    return pd.DataFrame(out)


def proposal_stats(group: pd.DataFrame) -> dict[str, float]:
    predicted_feasible = group["predicted_feasible"].astype(int).to_numpy() == 1
    violation = group["true_risk_violation"].fillna(0).astype(int).to_numpy() == 1
    selected_length = group["selected_true_length"].astype(float).to_numpy()
    oracle_length = group["oracle_true_length"].astype(float).to_numpy()
    return {
        "predicted_feasible_n": int(predicted_feasible.sum()),
        "predicted_feasible_rate": rate(predicted_feasible),
        "true_violation_rate": rate(violation[predicted_feasible]),
        "proposal_true_length": safe_mean(selected_length[predicted_feasible]),
        "proposal_regret_to_true_portfolio": safe_mean(
            selected_length[predicted_feasible] - oracle_length[predicted_feasible]
        ),
    }


def profile_for_method(method: str) -> str:
    for profile, profile_method in PROFILE_METHODS.items():
        if method == profile_method:
            return profile
    return "mixed"


def select_method(results: pd.DataFrame, quantile: float, method: str) -> pd.DataFrame:
    return results[(results["budget_quantile"].astype(float) == float(quantile)) & (results["method"].astype(str) == method)].copy()


def valid_proposal_index(group: pd.DataFrame) -> pd.Index:
    return group.index[
        (group["predicted_feasible"].astype(int) == 1)
        & (group["true_risk_violation"].fillna(0).astype(int) == 0)
    ]


def safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan


def rate(mask: np.ndarray) -> float:
    return float(np.mean(mask.astype(float))) if mask.size else math.nan


def weighted_mean(values: pd.Series, weights: np.ndarray) -> float:
    array = values.to_numpy(dtype=float)
    valid = np.isfinite(array) & np.isfinite(weights) & (weights > 0)
    return float(np.average(array[valid], weights=weights[valid])) if np.any(valid) else math.nan


if __name__ == "__main__":
    main()
