from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.dataset_generator import (
    PlannerConfig,
    _failed_label,
    planner_iterations,
    planner_nodes,
    run_planner,
)
from src.learning.features import scenario_from_metadata

_METADATA: dict[str, Any] | None = None
_PLANNER_CONFIG: PlannerConfig | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replan an existing edge dataset with a different planner risk weight while preserving scenarios and start-goal pairs."
    )
    parser.add_argument("source_dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planner-beta", type=float, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    source_samples_path = args.source_dataset / "samples.csv"
    source_metadata_path = args.source_dataset / "metadata.json"
    if not source_samples_path.exists() or not source_metadata_path.exists():
        raise FileNotFoundError(f"Expected samples.csv and metadata.json under {args.source_dataset}")

    args.output.mkdir(parents=True, exist_ok=True)
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    output_metadata = metadata_with_beta(source_metadata, args)
    (args.output / "metadata.json").write_text(json.dumps(output_metadata, indent=2), encoding="utf-8")

    samples = pd.read_csv(source_samples_path)
    if args.limit_samples is not None:
        samples = samples.head(int(args.limit_samples)).copy()

    output_samples_path = args.output / "samples.csv"
    existing = load_existing_records(output_samples_path) if args.resume else {}
    pending_df = samples[~samples["sample_id"].astype(int).isin(existing)].copy()
    jobs = [group.to_dict("records") for _, group in pending_df.groupby("scenario_id", sort=False)]
    records = [existing[idx] for idx in sorted(existing)]
    planner_config = PlannerConfig(**output_metadata["config"]["planner"])
    next_checkpoint = next_threshold(len(records), args.checkpoint_every)
    next_progress = next_threshold(len(records), args.progress_every)

    if jobs:
        if int(args.workers) > 1:
            with Pool(
                processes=int(args.workers),
                initializer=init_worker,
                initargs=(output_metadata, asdict(planner_config)),
            ) as pool:
                for batch in pool.imap(replan_scenario, jobs, chunksize=1):
                    records.extend(batch)
                    next_checkpoint = maybe_checkpoint(
                        output_samples_path, records, args.checkpoint_every, next_checkpoint
                    )
                    next_progress = maybe_progress(
                        len(records), len(samples), args.progress_every, next_progress
                    )
        else:
            init_worker(output_metadata, asdict(planner_config))
            for job in jobs:
                records.extend(replan_scenario(job))
                next_checkpoint = maybe_checkpoint(
                    output_samples_path, records, args.checkpoint_every, next_checkpoint
                )
                next_progress = maybe_progress(
                    len(records), len(samples), args.progress_every, next_progress
                )

    records.sort(key=lambda item: int(item["sample_id"]))
    write_records_csv(output_samples_path, records, list(samples.columns), required=True)
    write_summary(args.output / "summary.json", records, args, source=str(args.source_dataset))
    feasible = sum(int(record["feasible"]) for record in records)
    print(f"done: beta={args.planner_beta} feasible={feasible}/{len(records)}")
    print(f"samples: {output_samples_path}")


def metadata_with_beta(source_metadata: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    metadata = json.loads(json.dumps(source_metadata))
    allowed = set(PlannerConfig.__dataclass_fields__)
    planner = {key: value for key, value in metadata["config"]["planner"].items() if key in allowed}
    planner["beta"] = float(args.planner_beta)
    metadata["config"]["planner"] = planner
    protocol = dict(metadata.get("protocol", {}))
    protocol["profile_replan_source_dataset"] = str(args.source_dataset)
    protocol["profile_replan_label"] = args.label or f"beta_{args.planner_beta:g}"
    protocol["profile_replan_preserves"] = "scenarios, start-goal pairs, seeds, split, and distribution tags"
    metadata["protocol"] = protocol
    return metadata


def init_worker(metadata: dict[str, Any], planner_config_dict: dict[str, Any]) -> None:
    global _METADATA, _PLANNER_CONFIG
    _METADATA = metadata
    _PLANNER_CONFIG = PlannerConfig(**planner_config_dict)


def replan_scenario(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if _METADATA is None or _PLANNER_CONFIG is None:
        raise RuntimeError("worker is not initialized")
    if not rows:
        return []
    scenario_id = int(rows[0]["scenario_id"])
    if any(int(row["scenario_id"]) != scenario_id for row in rows):
        raise ValueError("A replan job must contain exactly one scenario.")
    scenario = scenario_from_metadata(_METADATA, scenario_id)
    return [replan_row(row, scenario) for row in rows]


def replan_row(row: dict[str, Any], scenario: Any) -> dict[str, Any]:
    if _METADATA is None or _PLANNER_CONFIG is None:
        raise RuntimeError("worker is not initialized")

    start = (float(row["start_x"]), float(row["start_y"]), float(row["start_theta"]))
    goal = (float(row["goal_x"]), float(row["goal_y"]))
    record = dict(row)
    result, runtime_sec = run_planner(scenario, start, goal, _PLANNER_CONFIG)
    record["runtime_sec"] = runtime_sec
    if result is None:
        record.update(_failed_label(_PLANNER_CONFIG))
        return normalize_record(record)

    record.update(
        {
            "feasible": 1,
            "length": result.metrics["length"],
            "time": result.metrics["time"],
            "risk": result.metrics["risk"],
            "survival_prob": result.metrics["survival_prob"],
            "turning": result.metrics["turning"],
            "objective": result.metrics["objective"],
            "planner_iterations": planner_iterations(result.stats),
            "planner_nodes": planner_nodes(result.stats),
        }
    )
    return normalize_record(record)


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    for key, value in list(record.items()):
        if isinstance(value, float) and math.isnan(value):
            record[key] = ""
    return record


def next_threshold(done: int, interval: int) -> int:
    if interval <= 0:
        return 0
    return ((int(done) // int(interval)) + 1) * int(interval)


def maybe_checkpoint(path: Path, records: list[dict[str, Any]], checkpoint_every: int, next_checkpoint: int) -> int:
    done = len(records)
    if checkpoint_every > 0 and next_checkpoint > 0 and done >= next_checkpoint:
        write_records_csv(path, records, list(records[0].keys()), required=False)
        while next_checkpoint <= done:
            next_checkpoint += int(checkpoint_every)
    return next_checkpoint


def maybe_progress(done: int, total: int, progress_every: int, next_progress: int) -> int:
    if progress_every > 0 and (next_progress > 0 and done >= next_progress or done == total):
        print(f"[{done}/{total}] replanned", flush=True)
        while next_progress <= done:
            next_progress += int(progress_every)
    return next_progress


def load_existing_records(path: Path) -> dict[int, dict[str, Any]]:
    candidates = [path, pending_path(path)]
    existing_paths = [candidate for candidate in candidates if candidate.exists()]
    if not existing_paths:
        return {}
    source = max(existing_paths, key=count_csv_rows)
    if source != path:
        print(f"resuming from checkpoint fallback: {source}", flush=True)
    with source.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {int(row["sample_id"]): row for row in rows}


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def pending_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.pending{path.suffix}")


def write_records_csv(path: Path, records: list[dict[str, Any]], columns: list[str], required: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in sorted(records, key=lambda item: int(item["sample_id"])):
            writer.writerow({column: record.get(column, "") for column in columns})
    for attempt in range(6 if required else 1):
        try:
            tmp_path.replace(path)
            return
        except PermissionError:
            if attempt < 5 and required:
                time.sleep(2.0)
                continue
            fallback = pending_path(path)
            tmp_path.replace(fallback)
            message = f"warning: could not replace locked {path}; wrote checkpoint fallback {fallback}"
            print(message, flush=True)
            if required:
                raise PermissionError(f"{message}. Close the CSV and rerun with --resume.")
            return


def write_summary(path: Path, records: list[dict[str, Any]], args: argparse.Namespace, source: str) -> None:
    df = pd.DataFrame(records)
    summary = {
        "source_dataset": source,
        "label": args.label or f"beta_{args.planner_beta:g}",
        "planner_beta": float(args.planner_beta),
        "samples": int(len(df)),
        "feasible": int(pd.to_numeric(df["feasible"], errors="coerce").fillna(0).sum()) if len(df) else 0,
        "mean_length": float(pd.to_numeric(df["length"], errors="coerce").mean()) if len(df) else math.nan,
        "mean_risk": float(pd.to_numeric(df["risk"], errors="coerce").mean()) if len(df) else math.nan,
        "mean_objective": float(pd.to_numeric(df["objective"], errors="coerce").mean()) if len(df) else math.nan,
        "mean_runtime_sec": float(pd.to_numeric(df["runtime_sec"], errors="coerce").mean()) if len(df) else math.nan,
    }
    if len(df) and {"split", "distribution"}.issubset(df.columns):
        groups = []
        for (split, distribution), group in df.groupby(["split", "distribution"], sort=True):
            groups.append(
                {
                    "split": split,
                    "distribution": distribution,
                    "samples": int(len(group)),
                    "feasible": int(pd.to_numeric(group["feasible"], errors="coerce").fillna(0).sum()),
                    "mean_length": float(pd.to_numeric(group["length"], errors="coerce").mean()),
                    "mean_risk": float(pd.to_numeric(group["risk"], errors="coerce").mean()),
                }
            )
        summary["by_split_distribution"] = groups
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
