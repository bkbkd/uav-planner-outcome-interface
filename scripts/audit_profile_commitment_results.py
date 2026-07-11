from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXACT_SINGLES = ("fast_only", "balanced_only", "safe_only")
EXACT_SUBSETS = ("portfolio_fast_balanced", "portfolio_fast_safe", "portfolio_balanced_safe")
LEARNED_SINGLES = ("fast_only_learned", "balanced_only_learned", "safe_only_learned")
LEARNED_SUBSETS = (
    "learned_portfolio_fast_balanced",
    "learned_portfolio_fast_safe",
    "learned_portfolio_balanced_safe",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit profile-commitment result-set containment and selection semantics.")
    parser.add_argument("--result-set", nargs=3, action="append", metavar=("POOL", "EXACT_DIR", "LEARNED_DIR"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checks = []
    for pool, exact_dir, learned_dir in args.result_set:
        exact = pd.read_csv(Path(exact_dir) / "budget_assignment_results.csv")
        learned = pd.read_csv(Path(learned_dir) / "learned_budget_assignment_results.csv")
        checks.extend(audit_pool(str(pool), exact, learned))
    failures = [check for check in checks if not check["passed"]]
    payload = {
        "passed": not failures,
        "checks": checks,
        "failure_count": len(failures),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


def audit_pool(pool: str, exact: pd.DataFrame, learned: pd.DataFrame) -> list[dict[str, object]]:
    exact_keyed = exact.set_index(["instance_id", "budget_quantile", "method"]).sort_index()
    learned_keyed = learned.set_index(["instance_id", "budget_quantile", "method"]).sort_index()
    exact_keys = exact[["instance_id", "budget_quantile"]].drop_duplicates().itertuples(index=False, name=None)
    failures = {
        "exact_global_selection": 0,
        "exact_containment": 0,
        "learned_global_selection": 0,
        "learned_predicted_containment": 0,
        "shared_budget_and_oracle": 0,
    }
    instances = set()
    quantiles = set()
    for instance_id, q in exact_keys:
        instance_id = int(instance_id)
        q = float(q)
        instances.add(instance_id)
        quantiles.add(q)
        exact_rows = exact_keyed.loc[(instance_id, q)]
        learned_rows = learned_keyed.loc[(instance_id, q)]

        feasible_singles = exact_rows.loc[list(EXACT_SINGLES)]
        feasible_singles = feasible_singles[feasible_singles["feasible"].astype(int) == 1]
        global_row = exact_rows.loc["global_profile"]
        if len(feasible_singles) == 0 or not close(
            float(global_row["selected_length"]), float(feasible_singles["selected_length"].astype(float).min())
        ):
            failures["exact_global_selection"] += 1
        portfolio_length = float(exact_rows.loc["portfolio", "selected_length"])
        for method in (*EXACT_SUBSETS, *EXACT_SINGLES, "global_profile"):
            row = exact_rows.loc[method]
            if int(row["feasible"]) == 1 and portfolio_length > float(row["selected_length"]) + 1e-7:
                failures["exact_containment"] += 1

        predicted_singles = learned_rows.loc[list(LEARNED_SINGLES)]
        predicted_singles = predicted_singles[predicted_singles["predicted_feasible"].astype(int) == 1]
        learned_global = learned_rows.loc["dispatch_global_learned"]
        if len(predicted_singles) == 0:
            if int(learned_global["predicted_feasible"]) != 0:
                failures["learned_global_selection"] += 1
        elif int(learned_global["predicted_feasible"]) != 1 or not close(
            float(learned_global["selected_pred_length"]),
            float(predicted_singles["selected_pred_length"].astype(float).min()),
        ):
            failures["learned_global_selection"] += 1
        learned_portfolio = learned_rows.loc["learned_portfolio"]
        if int(learned_portfolio["predicted_feasible"]) == 1:
            portfolio_pred_length = float(learned_portfolio["selected_pred_length"])
            for method in (*LEARNED_SUBSETS, *LEARNED_SINGLES, "dispatch_global_learned"):
                row = learned_rows.loc[method]
                if int(row["predicted_feasible"]) == 1 and portfolio_pred_length > float(row["selected_pred_length"]) + 1e-7:
                    failures["learned_predicted_containment"] += 1

        exact_portfolio = exact_rows.loc["portfolio"]
        learned_oracle = learned_rows.loc["oracle_true_portfolio"]
        if not (
            close(float(exact_portfolio["risk_budget"]), float(learned_oracle["risk_budget"]))
            and close(float(exact_portfolio["selected_length"]), float(learned_oracle["selected_true_length"]))
            and close(float(exact_portfolio["selected_risk"]), float(learned_oracle["selected_true_risk"]))
        ):
            failures["shared_budget_and_oracle"] += 1

    checks = []
    for name, count in failures.items():
        checks.append(
            {
                "pool": pool,
                "check": name,
                "instances": len(instances),
                "budget_quantiles": len(quantiles),
                "failure_count": int(count),
                "passed": count == 0,
            }
        )
    return checks


def close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1e-10, atol=1e-8))


if __name__ == "__main__":
    main()
