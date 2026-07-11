from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.evaluate_route_portfolio_budget import load_benchmark, validate_same_instances, write_rows


MODES = ("fast", "balanced", "safe")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose whether multi-profile true planner outcomes form a non-degenerate length-risk portfolio."
    )
    parser.add_argument("--fast", type=Path, required=True)
    parser.add_argument("--balanced", type=Path, required=True)
    parser.add_argument("--safe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length-rel-eps", type=float, default=0.01)
    parser.add_argument("--risk-rel-eps", type=float, default=0.01)
    parser.add_argument("--risk-floor", type=float, default=1e-6)
    args = parser.parse_args()

    benchmarks = {
        "fast": load_benchmark(args.fast),
        "balanced": load_benchmark(args.balanced),
        "safe": load_benchmark(args.safe),
    }
    validate_same_instances(benchmarks)
    args.output.mkdir(parents=True, exist_ok=True)

    edge_rows = diagnose_edges(
        benchmarks,
        length_rel_eps=args.length_rel_eps,
        risk_rel_eps=args.risk_rel_eps,
        risk_floor=args.risk_floor,
    )
    summary_rows = summarize(edge_rows, args)
    pair_rows = summarize_pairs(edge_rows)

    write_rows(args.output / "profile_nondegeneracy_edges.csv", edge_rows)
    write_rows(args.output / "profile_nondegeneracy_summary.csv", summary_rows)
    write_rows(args.output / "profile_tradeoff_pairs.csv", pair_rows)
    metadata = {
        "fast": str(args.fast),
        "balanced": str(args.balanced),
        "safe": str(args.safe),
        "length_rel_eps": args.length_rel_eps,
        "risk_rel_eps": args.risk_rel_eps,
        "risk_floor": args.risk_floor,
        "summary": summary_rows,
    }
    (args.output / "profile_nondegeneracy_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_markdown(args.output / "profile_nondegeneracy_summary.md", summary_rows, pair_rows)
    print(json.dumps(metadata, indent=2))
    print(f"saved: {args.output}")


def diagnose_edges(
    benchmarks: dict[str, dict[str, pd.DataFrame]],
    *,
    length_rel_eps: float,
    risk_rel_eps: float,
    risk_floor: float,
) -> list[dict[str, Any]]:
    pairs_by_mode = {mode: data["pairs"].reset_index(drop=True) for mode, data in benchmarks.items()}
    reference = pairs_by_mode["balanced"]
    rows: list[dict[str, Any]] = []
    for idx, ref in reference.iterrows():
        outcomes = {
            mode: {
                "length": float(pairs_by_mode[mode].iloc[idx]["length"]),
                "risk": float(pairs_by_mode[mode].iloc[idx]["risk"]),
            }
            for mode in MODES
        }
        dominated = {
            mode: is_dominated(mode, outcomes, length_rel_eps=length_rel_eps, risk_rel_eps=risk_rel_eps, risk_floor=risk_floor)
            for mode in MODES
        }
        nondominated = [mode for mode in MODES if not dominated[mode]]
        lengths = np.asarray([outcomes[mode]["length"] for mode in MODES], dtype=float)
        risks = np.asarray([outcomes[mode]["risk"] for mode in MODES], dtype=float)
        row: dict[str, Any] = {
            "instance_id": int(ref["instance_id"]),
            "agent_id": int(ref["agent_id"]),
            "task_id": int(ref["task_id"]),
            "nondominated_count": int(len(nondominated)),
            "nondominated_modes": ";".join(nondominated),
            "length_range": float(lengths.max() - lengths.min()),
            "risk_range": float(risks.max() - risks.min()),
        }
        for mode in MODES:
            row[f"{mode}_length"] = outcomes[mode]["length"]
            row[f"{mode}_risk"] = outcomes[mode]["risk"]
            row[f"{mode}_dominated"] = int(dominated[mode])
            row[f"{mode}_nondominated"] = int(not dominated[mode])
        rows.append(row)
    return rows


def is_dominated(
    mode: str,
    outcomes: dict[str, dict[str, float]],
    *,
    length_rel_eps: float,
    risk_rel_eps: float,
    risk_floor: float,
) -> bool:
    length = outcomes[mode]["length"]
    risk = outcomes[mode]["risk"]
    eps_length = max(0.0, length_rel_eps * max(length, 1e-9))
    eps_risk = max(0.0, risk_rel_eps * max(risk, risk_floor))
    for other in MODES:
        if other == mode:
            continue
        other_length = outcomes[other]["length"]
        other_risk = outcomes[other]["risk"]
        no_worse = other_length <= length + eps_length and other_risk <= risk + eps_risk
        meaningfully_better = other_length < length - eps_length or other_risk < risk - eps_risk
        if no_worse and meaningfully_better:
            return True
    return False


def summarize(edge_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    df = pd.DataFrame(edge_rows)
    if df.empty:
        return []
    row: dict[str, Any] = {
        "edges": int(len(df)),
        "length_rel_eps": float(args.length_rel_eps),
        "risk_rel_eps": float(args.risk_rel_eps),
        "risk_floor": float(args.risk_floor),
        "mean_nondominated_count": float(df["nondominated_count"].mean()),
        "p_nondominated_ge2": float((df["nondominated_count"] >= 2).mean()),
        "p_nondominated_eq3": float((df["nondominated_count"] == 3).mean()),
        "length_range_mean": float(df["length_range"].mean()),
        "length_range_p50": float(df["length_range"].quantile(0.50)),
        "length_range_p90": float(df["length_range"].quantile(0.90)),
        "risk_range_mean": float(df["risk_range"].mean()),
        "risk_range_p50": float(df["risk_range"].quantile(0.50)),
        "risk_range_p90": float(df["risk_range"].quantile(0.90)),
    }
    for mode in MODES:
        row[f"{mode}_dominated_rate"] = float(df[f"{mode}_dominated"].mean())
        row[f"{mode}_nondominated_rate"] = float(df[f"{mode}_nondominated"].mean())
    return [row]


def summarize_pairs(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(edge_rows)
    rows: list[dict[str, Any]] = []
    if df.empty:
        return rows
    for left, right in [("fast", "balanced"), ("balanced", "safe"), ("fast", "safe")]:
        d_length = df[f"{right}_length"].astype(float) - df[f"{left}_length"].astype(float)
        d_risk = df[f"{right}_risk"].astype(float) - df[f"{left}_risk"].astype(float)
        tradeoff = (d_length > 0.0) & (d_risk < 0.0)
        rows.append(
            {
                "pair": f"{left}_to_{right}",
                "delta_length_mean": float(d_length.mean()),
                "delta_length_p50": float(d_length.quantile(0.50)),
                "delta_length_p90": float(d_length.quantile(0.90)),
                "delta_risk_mean": float(d_risk.mean()),
                "delta_risk_p50": float(d_risk.quantile(0.50)),
                "delta_risk_p10": float(d_risk.quantile(0.10)),
                "tradeoff_rate_longer_and_safer": float(tradeoff.mean()),
            }
        )
    return rows


def write_markdown(path: Path, summary_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> None:
    lines = ["# Profile Non-Degeneracy Diagnostic", ""]
    if summary_rows:
        row = summary_rows[0]
        lines.extend(
            [
                f"- Edges: {row['edges']}",
                f"- Mean nondominated profiles per edge: {row['mean_nondominated_count']:.3f}",
                f"- P(nondominated >= 2): {row['p_nondominated_ge2']:.3f}",
                f"- P(nondominated = 3): {row['p_nondominated_eq3']:.3f}",
                "",
                "| Mode | Dominated rate | Nondominated rate |",
                "|---|---:|---:|",
            ]
        )
        for mode in MODES:
            lines.append(f"| {mode} | {row[f'{mode}_dominated_rate']:.3f} | {row[f'{mode}_nondominated_rate']:.3f} |")
    if pair_rows:
        lines.extend(["", "| Pair | dL mean | dR mean | Longer and safer |", "|---|---:|---:|---:|"])
        for row in pair_rows:
            lines.append(
                f"| {row['pair']} | {row['delta_length_mean']:.3f} | {row['delta_risk_mean']:.4f} | "
                f"{row['tradeoff_rate_longer_and_safer']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
