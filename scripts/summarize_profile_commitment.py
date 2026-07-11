from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXACT_PORTFOLIO = "portfolio"
LEARNED_PORTFOLIO = "learned_portfolio"
SINGLE_EXACT = ("fast_only", "balanced_only", "safe_only")
REGIME_METRICS = (
    "effective_alternative_rate",
    "exchange_rate_log_iqr",
    "assignment_disagreement",
    "budget_utilization",
)
OUTCOME_EQ_ATOL = 1e-8


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize profile-commitment granularity, portfolio cardinality, and operating regimes."
    )
    parser.add_argument("--exact-result", nargs=2, action="append", metavar=("POOL", "DIR"), required=True)
    parser.add_argument("--learned-result", nargs=2, action="append", metavar=("POOL", "DIR"), required=True)
    parser.add_argument(
        "--benchmark-set",
        nargs=4,
        action="append",
        metavar=("POOL", "FAST_DIR", "BALANCED_DIR", "SAFE_DIR"),
        required=True,
    )
    parser.add_argument("--development-pool", default="development")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exact = load_result_sets(args.exact_result, "budget_assignment_results.csv")
    learned = load_result_sets(args.learned_result, "learned_budget_assignment_results.csv")
    rng = np.random.default_rng(args.seed)

    heldout_exact = exact[exact["pool"].astype(str) != str(args.development_pool)].copy()
    heldout_learned = learned[learned["pool"].astype(str) != str(args.development_pool)].copy()
    exact_summary = summarize_exact(heldout_exact)
    learned_proposals = summarize_learned_proposals(heldout_learned)
    learned_pairs = summarize_learned_pairs(heldout_learned, rng, args.bootstrap)
    operating = build_operating_rows(exact, args.benchmark_set)
    thresholds = development_thresholds(operating, args.development_pool)
    operating = assign_regime_bins(operating, thresholds)
    regime_summary = summarize_regimes(operating, args.development_pool)
    joint_regime_summary = summarize_joint_regimes(operating, args.development_pool)
    regime_correlations = summarize_regime_correlations(operating, args.development_pool)
    effective_cardinality = summarize_effective_cardinality(operating, args.development_pool)
    example = select_development_example(operating, exact, args.development_pool)

    exact_summary.to_csv(args.output_dir / "exact_commitment_summary.csv", index=False)
    learned_proposals.to_csv(args.output_dir / "learned_commitment_proposals.csv", index=False)
    learned_pairs.to_csv(args.output_dir / "learned_commitment_paired.csv", index=False)
    operating.to_csv(args.output_dir / "operating_regime_instances.csv", index=False)
    regime_summary.to_csv(args.output_dir / "operating_regime_summary.csv", index=False)
    joint_regime_summary.to_csv(args.output_dir / "operating_regime_joint_summary.csv", index=False)
    regime_correlations.to_csv(args.output_dir / "operating_regime_correlations.csv", index=False)
    effective_cardinality.to_csv(args.output_dir / "effective_outcome_cardinality.csv", index=False)
    (args.output_dir / "operating_regime_thresholds.json").write_text(
        json.dumps(thresholds, indent=2), encoding="utf-8"
    )
    (args.output_dir / "fleet_example.json").write_text(json.dumps(example, indent=2), encoding="utf-8")
    summary = {
        "development_pool": args.development_pool,
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "exact_rows": int(len(exact)),
        "learned_rows": int(len(learned)),
        "heldout_exact_rows": int(len(heldout_exact)),
        "heldout_learned_rows": int(len(heldout_learned)),
        "operating_rows": int(len(operating)),
        "outcome_equivalence_atol": OUTCOME_EQ_ATOL,
        "fleet_example": example,
    }
    (args.output_dir / "profile_commitment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def load_result_sets(specs: list[list[str]], filename: str) -> pd.DataFrame:
    frames = []
    for pool, directory in specs:
        frame = pd.read_csv(Path(directory) / filename)
        frame.insert(0, "pool", str(pool))
        frame["instance_key"] = frame["pool"] + ":" + frame["instance_id"].astype(str)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_exact(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (q, method), group in results.groupby(["budget_quantile", "method"], sort=True):
        portfolio = results[
            (results["budget_quantile"].astype(float) == float(q))
            & (results["method"].astype(str) == EXACT_PORTFOLIO)
        ].set_index("instance_key")
        candidate = group.set_index("instance_key")
        feasible = candidate[candidate["feasible"].astype(int) == 1]
        common = portfolio.index.intersection(feasible.index)
        gains = (
            feasible.loc[common, "selected_length"].astype(float)
            - portfolio.loc[common, "selected_length"].astype(float)
        ).to_numpy(dtype=float)
        rows.append(
            {
                "budget_quantile": float(q),
                "method": str(method),
                "library_size": library_size(str(method)),
                "commitment_level": commitment_level(str(method)),
                "instances": int(len(candidate)),
                "feasible_rate": float(len(feasible) / len(candidate)) if len(candidate) else math.nan,
                "common_n": int(len(common)),
                "selected_length_mean": safe_mean(feasible["selected_length"]),
                "gain_to_k3_mean": safe_mean_array(gains),
                "gain_to_k3_median": safe_quantile(gains, 0.5),
                "gain_to_k3_p90": safe_quantile(gains, 0.9),
                "gain_gt_1m_rate": safe_rate(gains > 1.0),
            }
        )
    return pd.DataFrame(rows)


def summarize_learned_proposals(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    predicted = results[results["uses_predicted_interface"].astype(int) == 1]
    for (q, method), group in predicted.groupby(["budget_quantile", "method"], sort=True):
        proposals = group[group["predicted_feasible"].astype(int) == 1]
        rows.append(
            {
                "budget_quantile": float(q),
                "method": str(method),
                "library_size": library_size(str(method)),
                "commitment_level": commitment_level(str(method)),
                "instances": int(len(group)),
                "predicted_feasible_rate": float(len(proposals) / len(group)) if len(group) else math.nan,
                "proposal_n": int(len(proposals)),
                "true_length_mean": safe_mean(proposals["selected_true_length"]),
                "true_risk_mean": safe_mean(proposals["selected_true_risk"]),
                "true_violation_rate": safe_mean(proposals["true_risk_violation"]),
                "true_regret_mean": safe_mean(proposals["true_length_regret_to_portfolio"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_learned_pairs(
    results: pd.DataFrame, rng: np.random.Generator, bootstrap: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    predicted = results[results["uses_predicted_interface"].astype(int) == 1]
    for q in sorted(predicted["budget_quantile"].astype(float).unique()):
        qdf = predicted[predicted["budget_quantile"].astype(float) == q]
        portfolio = qdf[qdf["method"] == LEARNED_PORTFOLIO].set_index("instance_key")
        references = sorted(set(qdf["method"].astype(str)) - {LEARNED_PORTFOLIO})
        for reference in references:
            control = qdf[qdf["method"] == reference].set_index("instance_key")
            for scope in ("common_predicted_feasible", "common_true_nonviolating"):
                p_valid = portfolio["predicted_feasible"].astype(int) == 1
                c_valid = control["predicted_feasible"].astype(int) == 1
                if scope == "common_true_nonviolating":
                    p_valid &= portfolio["true_risk_violation"].fillna(1).astype(int) == 0
                    c_valid &= control["true_risk_violation"].fillna(1).astype(int) == 0
                common = portfolio.index[p_valid].intersection(control.index[c_valid])
                gains = (
                    control.loc[common, "selected_true_length"].astype(float)
                    - portfolio.loc[common, "selected_true_length"].astype(float)
                ).to_numpy(dtype=float)
                rows.append(
                    {
                        "budget_quantile": float(q),
                        "reference_method": reference,
                        "scope": scope,
                        "common_n": int(len(common)),
                        "common_rate": float(len(common) / len(portfolio)) if len(portfolio) else math.nan,
                        **distribution_summary(gains, rng, bootstrap),
                    }
                )
    return pd.DataFrame(rows)


def build_operating_rows(results: pd.DataFrame, benchmark_specs: list[list[str]]) -> pd.DataFrame:
    edge_metrics = []
    for pool, fast_dir, balanced_dir, safe_dir in benchmark_specs:
        pairs = {
            "fast": load_pairs(Path(fast_dir)),
            "balanced": load_pairs(Path(balanced_dir)),
            "safe": load_pairs(Path(safe_dir)),
        }
        validate_pair_keys(pairs)
        for instance_id in sorted(pairs["fast"]["instance_id"].astype(int).unique()):
            frames = {
                mode: frame[frame["instance_id"].astype(int) == instance_id].reset_index(drop=True)
                for mode, frame in pairs.items()
            }
            metrics = edge_profile_metrics(frames)
            edge_metrics.append({"pool": str(pool), "instance_id": int(instance_id), **metrics})
    edge_df = pd.DataFrame(edge_metrics)

    rows: list[dict[str, Any]] = []
    for (pool, instance_id, q), group in results.groupby(
        ["pool", "instance_id", "budget_quantile"], sort=True
    ):
        indexed = group.set_index("method")
        if EXACT_PORTFOLIO not in indexed.index or "global_profile" not in indexed.index:
            continue
        portfolio = indexed.loc[EXACT_PORTFOLIO]
        global_profile = indexed.loc["global_profile"]
        singles = indexed.loc[[method for method in SINGLE_EXACT if method in indexed.index]]
        feasible_singles = singles[singles["feasible"].astype(int) == 1]
        assignments = feasible_singles["assignment"].astype(str).tolist()
        disagreement = assignment_disagreement(assignments)
        budget = float(portfolio["risk_budget"])
        slack = float(portfolio["risk_slack"])
        rows.append(
            {
                "pool": str(pool),
                "instance_id": int(instance_id),
                "instance_key": f"{pool}:{int(instance_id)}",
                "budget_quantile": float(q),
                "risk_budget": budget,
                "portfolio_length": float(portfolio["selected_length"]),
                "global_profile_length": float(global_profile["selected_length"]),
                "mixing_gain": float(global_profile["selected_length"] - portfolio["selected_length"]),
                "normalized_slack": slack / max(abs(budget), 1e-9),
                "budget_utilization": 1.0 - slack / max(abs(budget), 1e-9),
                "feasible_global_profiles": int(len(feasible_singles)),
                "assignment_disagreement": disagreement,
                "portfolio_assignment": str(portfolio["assignment"]),
                "portfolio_modes": str(portfolio["modes"]),
                "global_assignment": str(global_profile["assignment"]),
                "global_modes": str(global_profile["modes"]),
            }
        )
    operating = pd.DataFrame(rows)
    return operating.merge(edge_df, on=["pool", "instance_id"], how="left", validate="many_to_one")


def load_pairs(directory: Path) -> pd.DataFrame:
    pairs = pd.read_csv(directory / "pairs.csv")
    return pairs.sort_values(["instance_id", "agent_id", "task_id"], kind="mergesort").reset_index(drop=True)


def validate_pair_keys(pairs: dict[str, pd.DataFrame]) -> None:
    reference = pairs["fast"][["instance_id", "agent_id", "task_id"]].reset_index(drop=True)
    for mode, frame in pairs.items():
        current = frame[["instance_id", "agent_id", "task_id"]].reset_index(drop=True)
        if not reference.equals(current):
            raise ValueError(f"profile pair keys differ for {mode}")


def edge_profile_metrics(frames: dict[str, pd.DataFrame]) -> dict[str, float | int]:
    outcomes = np.stack(
        [frames[mode][["length", "risk"]].to_numpy(dtype=float) for mode in ("fast", "balanced", "safe")],
        axis=1,
    )
    cardinalities = []
    for edge_outcomes in outcomes:
        classes: list[np.ndarray] = []
        for outcome in edge_outcomes:
            if not any(np.allclose(outcome, observed, rtol=0.0, atol=OUTCOME_EQ_ATOL) for observed in classes):
                classes.append(outcome)
        cardinalities.append(len(classes))
    cardinalities_array = np.asarray(cardinalities, dtype=int)
    values = []
    for lower, higher in (("fast", "balanced"), ("balanced", "safe")):
        delta_length = frames[higher]["length"].to_numpy(dtype=float) - frames[lower]["length"].to_numpy(dtype=float)
        delta_risk = frames[lower]["risk"].to_numpy(dtype=float) - frames[higher]["risk"].to_numpy(dtype=float)
        valid = (delta_length > 1e-9) & (delta_risk > 1e-9)
        values.extend(np.log1p(delta_length[valid] / delta_risk[valid]).tolist())
    array = np.asarray(values, dtype=float)
    return {
        "effective_k1_rate": float(np.mean(cardinalities_array == 1)),
        "effective_k2_rate": float(np.mean(cardinalities_array == 2)),
        "effective_k3_rate": float(np.mean(cardinalities_array == 3)),
        "effective_alternative_rate": float(np.mean(cardinalities_array >= 2)),
        "effective_outcome_count_mean": float(np.mean(cardinalities_array)),
        "valid_exchange_count": int(len(array)),
        "exchange_rate_log_iqr": (
            float(np.quantile(array, 0.75) - np.quantile(array, 0.25)) if len(array) >= 2 else math.nan
        ),
    }


def assignment_disagreement(assignments: list[str]) -> float:
    pairs = list(combinations(assignments, 2))
    return float(np.mean([left != right for left, right in pairs])) if pairs else 0.0


def development_thresholds(rows: pd.DataFrame, development_pool: str) -> dict[str, dict[str, list[float]]]:
    development = rows[rows["pool"].astype(str) == str(development_pool)]
    thresholds: dict[str, dict[str, list[float]]] = {}
    for q, group in development.groupby("budget_quantile", sort=True):
        thresholds[str(float(q))] = {}
        for metric in REGIME_METRICS:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                raise ValueError(f"no finite development values for {metric} at q={q}")
            thresholds[str(float(q))][metric] = [
                float(np.quantile(values, 1.0 / 3.0)),
                float(np.quantile(values, 2.0 / 3.0)),
            ]
    return thresholds


def assign_regime_bins(
    rows: pd.DataFrame, thresholds: dict[str, dict[str, list[float]]]
) -> pd.DataFrame:
    out = rows.copy()
    for metric in REGIME_METRICS:
        labels = []
        for row in out.itertuples(index=False):
            low, high = thresholds[str(float(row.budget_quantile))][metric]
            value = float(getattr(row, metric))
            labels.append("missing" if not math.isfinite(value) else "low" if value <= low else "medium" if value <= high else "high")
        out[f"{metric}_bin"] = labels
    return out


def summarize_regimes(rows: pd.DataFrame, development_pool: str) -> pd.DataFrame:
    test = rows[rows["pool"].astype(str) != str(development_pool)]
    summaries = []
    for metric in REGIME_METRICS:
        bin_column = f"{metric}_bin"
        for (q, label), group in test.groupby(["budget_quantile", bin_column], sort=True):
            gain = group["mixing_gain"].to_numpy(dtype=float)
            summaries.append(
                {
                    "budget_quantile": float(q),
                    "regime_metric": metric,
                    "regime_bin": str(label),
                    "n": int(len(group)),
                    "mixing_gain_mean": safe_mean_array(gain),
                    "mixing_gain_median": safe_quantile(gain, 0.5),
                    "mixing_gain_p90": safe_quantile(gain, 0.9),
                    "mixing_gain_gt_1m_rate": safe_rate(gain > 1.0),
                }
            )
    return pd.DataFrame(summaries)


def summarize_joint_regimes(rows: pd.DataFrame, development_pool: str) -> pd.DataFrame:
    test = rows[rows["pool"].astype(str) != str(development_pool)]
    summaries = []
    for (q, disagreement, utilization), group in test.groupby(
        ["budget_quantile", "assignment_disagreement_bin", "budget_utilization_bin"], sort=True
    ):
        gain = group["mixing_gain"].to_numpy(dtype=float)
        summaries.append(
            {
                "budget_quantile": float(q),
                "assignment_disagreement_bin": str(disagreement),
                "budget_utilization_bin": str(utilization),
                "n": int(len(group)),
                "mixing_gain_mean": safe_mean_array(gain),
                "mixing_gain_median": safe_quantile(gain, 0.5),
                "mixing_gain_p90": safe_quantile(gain, 0.9),
                "mixing_gain_gt_1m_rate": safe_rate(gain > 1.0),
            }
        )
    return pd.DataFrame(summaries)


def summarize_regime_correlations(rows: pd.DataFrame, development_pool: str) -> pd.DataFrame:
    test = rows[rows["pool"].astype(str) != str(development_pool)]
    summaries = []
    for q, group in test.groupby("budget_quantile", sort=True):
        for metric in REGIME_METRICS:
            valid = group[[metric, "mixing_gain"]].replace([np.inf, -np.inf], np.nan).dropna()
            summaries.append(
                {
                    "budget_quantile": float(q),
                    "metric": metric,
                    "n": int(len(valid)),
                    "spearman_rho": (
                        float(valid[metric].rank().corr(valid["mixing_gain"].rank()))
                        if len(valid) >= 2
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(summaries)


def summarize_effective_cardinality(rows: pd.DataFrame, development_pool: str) -> pd.DataFrame:
    per_instance = rows.drop_duplicates(["pool", "instance_id"])
    summaries = []
    groups = [(pool, group) for pool, group in per_instance.groupby("pool", sort=True)]
    groups.append(("heldout_all", per_instance[per_instance["pool"].astype(str) != str(development_pool)]))
    for pool, group in groups:
        summaries.append(
            {
                "pool": str(pool),
                "instances": int(len(group)),
                "edges": int(len(group) * 25),
                "effective_k1_rate": float(group["effective_k1_rate"].mean()),
                "effective_k2_rate": float(group["effective_k2_rate"].mean()),
                "effective_k3_rate": float(group["effective_k3_rate"].mean()),
                "effective_alternative_rate": float(group["effective_alternative_rate"].mean()),
                "effective_outcome_count_mean": float(group["effective_outcome_count_mean"].mean()),
            }
        )
    return pd.DataFrame(summaries)


def select_development_example(
    operating: pd.DataFrame, exact: pd.DataFrame, development_pool: str
) -> dict[str, Any]:
    candidates = operating[
        (operating["pool"].astype(str) == str(development_pool))
        & np.isclose(operating["budget_quantile"].astype(float), 0.5)
        & (operating["mixing_gain"].astype(float) > 1.0)
        & (operating["assignment_disagreement_bin"].astype(str) == "high")
        & (operating["budget_utilization_bin"].astype(str) == "high")
    ].copy()
    if candidates.empty:
        candidates = operating[
            (operating["pool"].astype(str) == str(development_pool))
            & np.isclose(operating["budget_quantile"].astype(float), 0.5)
            & (operating["mixing_gain"].astype(float) > 1.0)
        ].copy()
    if candidates.empty:
        return {}
    target = float(candidates["mixing_gain"].median())
    selected = candidates.iloc[(candidates["mixing_gain"] - target).abs().to_numpy().argmin()]
    key = str(selected["instance_key"])
    q = float(selected["budget_quantile"])
    rows = exact[(exact["instance_key"] == key) & np.isclose(exact["budget_quantile"].astype(float), q)]
    methods = {}
    for row in rows.itertuples(index=False):
        methods[str(row.method)] = {
            "feasible": int(row.feasible),
            "length": finite_or_none(row.selected_length),
            "risk": finite_or_none(row.selected_risk),
            "assignment": str(row.assignment),
            "modes": str(row.modes),
        }
    return {
        "selection_rule": "development q=0.5, gain>1m, high assignment disagreement and budget utilization, closest to median gain",
        "pool": str(selected["pool"]),
        "instance_id": int(selected["instance_id"]),
        "budget_quantile": q,
        "risk_budget": float(rows["risk_budget"].iloc[0]),
        "mixing_gain": float(selected["mixing_gain"]),
        "exchange_rate_log_iqr": float(selected["exchange_rate_log_iqr"]),
        "assignment_disagreement": float(selected["assignment_disagreement"]),
        "methods": methods,
    }


def distribution_summary(values: np.ndarray, rng: np.random.Generator, bootstrap: int) -> dict[str, float | int]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "gain_mean": math.nan,
            "gain_median": math.nan,
            "gain_ci95_low": math.nan,
            "gain_ci95_high": math.nan,
            "win_rate": math.nan,
            "tie_rate": math.nan,
            "loss_rate": math.nan,
        }
    boot = np.empty(bootstrap, dtype=float)
    for index in range(bootstrap):
        sample = rng.integers(0, len(values), size=len(values))
        boot[index] = float(np.mean(values[sample]))
    return {
        "gain_mean": float(np.mean(values)),
        "gain_median": float(np.median(values)),
        "gain_ci95_low": float(np.quantile(boot, 0.025)),
        "gain_ci95_high": float(np.quantile(boot, 0.975)),
        "win_rate": float(np.mean(values > 1e-6)),
        "tie_rate": float(np.mean(np.abs(values) <= 1e-6)),
        "loss_rate": float(np.mean(values < -1e-6)),
    }


def library_size(method: str) -> int:
    if method in {"portfolio", "learned_portfolio", "global_profile", "dispatch_global_learned"}:
        return 3
    if "portfolio_" in method:
        return 2
    return 1


def commitment_level(method: str) -> str:
    if method in {"global_profile", "dispatch_global_learned"}:
        return "dispatch_global"
    if "portfolio" in method:
        return "edgewise"
    return "fixed"


def safe_mean(values: pd.Series) -> float:
    return safe_mean_array(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float))


def safe_mean_array(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else math.nan


def safe_quantile(values: np.ndarray, quantile: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.quantile(values, quantile)) if len(values) else math.nan


def safe_rate(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else math.nan


def finite_or_none(value: Any) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


if __name__ == "__main__":
    main()
