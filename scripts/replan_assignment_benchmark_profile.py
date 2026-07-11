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

import numpy as np
import pandas as pd

from src.data.dataset_generator import PlannerConfig, run_planner
from src.learning.bid_targets import baseline_bid as planner_baseline_bid
from src.learning.bid_targets import true_bid_from_planner_metrics
from src.learning.features import scenario_from_metadata

_METADATA: dict[str, Any] | None = None
_PLANNER_CONFIG: PlannerConfig | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replan cached assignment benchmark edges with a different planner risk weight while preserving instances."
    )
    parser.add_argument("source_benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planner-beta", type=float, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-instances", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    source_instances_path = args.source_benchmark / "instances.csv"
    source_pairs_path = args.source_benchmark / "pairs.csv"
    source_metadata_path = args.source_benchmark / "source_metadata.json"
    for path in [source_instances_path, source_pairs_path, source_metadata_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    args.output.mkdir(parents=True, exist_ok=True)
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    metadata = metadata_with_beta(source_metadata, args)
    (args.output / "source_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    source_instances = pd.read_csv(source_instances_path)
    source_pairs = pd.read_csv(source_pairs_path)
    if args.limit_instances is not None:
        keep = set(source_instances["instance_id"].astype(int).head(int(args.limit_instances)))
        source_instances = source_instances[source_instances["instance_id"].astype(int).isin(keep)].copy()
        source_pairs = source_pairs[source_pairs["instance_id"].astype(int).isin(keep)].copy()
    planner_config = PlannerConfig(**metadata["config"]["planner"])

    output_pairs_path = args.output / "pairs.csv"
    existing_pairs = load_existing_rows(output_pairs_path) if args.resume else []
    done = {
        (int(row["instance_id"]), int(row["agent_id"]), int(row["task_id"]))
        for row in existing_pairs
    }
    pending_df = source_pairs[
        ~source_pairs.apply(
            lambda row: (int(row["instance_id"]), int(row["agent_id"]), int(row["task_id"])) in done,
            axis=1,
        )
    ].copy()
    jobs = [group.to_dict("records") for _, group in pending_df.groupby("instance_id", sort=False)]
    pair_rows = list(existing_pairs)
    next_checkpoint = next_threshold(len(pair_rows), args.checkpoint_every)
    next_progress = next_threshold(len(pair_rows), args.progress_every)

    if jobs:
        if int(args.workers) > 1:
            with Pool(
                processes=int(args.workers),
                initializer=init_worker,
                initargs=(metadata, asdict(planner_config)),
            ) as pool:
                for batch in pool.imap(replan_instance, jobs, chunksize=1):
                    pair_rows.extend(batch)
                    next_checkpoint = maybe_checkpoint(
                        output_pairs_path, pair_rows, args.checkpoint_every, next_checkpoint
                    )
                    next_progress = maybe_progress(
                        len(pair_rows), len(source_pairs), args.progress_every, next_progress
                    )
        else:
            init_worker(metadata, asdict(planner_config))
            for job in jobs:
                pair_rows.extend(replan_instance(job))
                next_checkpoint = maybe_checkpoint(
                    output_pairs_path, pair_rows, args.checkpoint_every, next_checkpoint
                )
                next_progress = maybe_progress(
                    len(pair_rows), len(source_pairs), args.progress_every, next_progress
                )

    pair_rows.sort(key=lambda row: (int(row["instance_id"]), int(row["agent_id"]), int(row["task_id"])))
    pair_df = pd.DataFrame(pair_rows)
    instance_rows, pair_rows = rebuild_instances(source_instances, pair_df)
    write_rows_atomic(args.output / "pairs.csv", pair_rows, required=True)
    write_rows_atomic(args.output / "instances.csv", instance_rows, required=True)
    write_summary(args.output / "benchmark_summary.json", instance_rows, pair_rows, args, source=str(args.source_benchmark))
    print(f"done: beta={args.planner_beta} instances={len(instance_rows)} edges={len(pair_rows)}")
    print(f"saved: {args.output}")


def metadata_with_beta(source_metadata: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    metadata = json.loads(json.dumps(source_metadata))
    allowed = set(PlannerConfig.__dataclass_fields__)
    planner = {key: value for key, value in metadata["config"]["planner"].items() if key in allowed}
    planner["beta"] = float(args.planner_beta)
    metadata["config"]["planner"] = planner
    protocol = dict(metadata.get("protocol", {}))
    protocol["profile_replan_source_benchmark"] = str(args.source_benchmark)
    protocol["profile_replan_label"] = args.label or f"beta_{args.planner_beta:g}"
    protocol["profile_replan_preserves"] = "instances, agents, tasks, pair keys, and scenario pool"
    metadata["protocol"] = protocol
    return metadata


def init_worker(metadata: dict[str, Any], planner_config_dict: dict[str, Any]) -> None:
    global _METADATA, _PLANNER_CONFIG
    _METADATA = metadata
    _PLANNER_CONFIG = PlannerConfig(**planner_config_dict)


def replan_instance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if _METADATA is None or _PLANNER_CONFIG is None:
        raise RuntimeError("worker is not initialized")
    if not rows:
        return []
    instance_id = int(rows[0]["instance_id"])
    scenario_id = int(rows[0]["scenario_id"])
    if any(int(row["instance_id"]) != instance_id or int(row["scenario_id"]) != scenario_id for row in rows):
        raise ValueError("A replan job must contain exactly one instance and scenario.")
    scenario = scenario_from_metadata(_METADATA, scenario_id)
    return [replan_pair(row, scenario) for row in rows]


def replan_pair(row: dict[str, Any], scenario: Any) -> dict[str, Any]:
    if _METADATA is None or _PLANNER_CONFIG is None:
        raise RuntimeError("worker is not initialized")

    start = (float(row["start_x"]), float(row["start_y"]), float(row["start_theta"]))
    goal = (float(row["goal_x"]), float(row["goal_y"]))
    result, elapsed = run_planner(scenario, start, goal, _PLANNER_CONFIG)
    if result is None:
        raise RuntimeError(
            f"planner failed for instance={row['instance_id']} agent={row['agent_id']} task={row['task_id']} beta={_PLANNER_CONFIG.beta}"
        )

    out = dict(row)
    out.update(
        {
            "length": result.metrics["length"],
            "time": result.metrics["time"],
            "risk": result.metrics["risk"],
            "turning": result.metrics["turning"],
            "objective": result.metrics["objective"],
            "true_bid": true_bid_from_planner_metrics(result.metrics, _METADATA),
            "feasible": 1,
            "runtime_sec": elapsed,
        }
    )
    out["baseline_bid"] = float(planner_baseline_bid(pd.DataFrame([out]), _METADATA)[0])
    return normalize_record(out)


def rebuild_instances(source_instances: pd.DataFrame, pair_df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pair_df = pair_df.copy()
    for instance in source_instances.itertuples(index=False):
        instance_id = int(instance.instance_id)
        instance_pairs = pair_df[pair_df["instance_id"].astype(int) == instance_id].copy()
        out = instance._asdict()
        out["true_planner_runtime_sec"] = float(instance_pairs["runtime_sec"].astype(float).sum())
        rows.append(normalize_record(out))

    sort_df = pair_df.copy()
    for column in ["instance_id", "agent_id", "task_id"]:
        sort_df[f"_{column}_sort"] = sort_df[column].astype(int)
    updated_pairs = (
        sort_df.sort_values(["_instance_id_sort", "_agent_id_sort", "_task_id_sort"])
        .drop(columns=["_instance_id_sort", "_agent_id_sort", "_task_id_sort"])
        .to_dict("records")
    )
    return rows, [normalize_record(row) for row in updated_pairs]


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in record.items():
        if isinstance(value, float) and math.isnan(value):
            out[key] = ""
        else:
            out[key] = value
    return out


def next_threshold(done: int, interval: int) -> int:
    if interval <= 0:
        return 0
    return ((int(done) // int(interval)) + 1) * int(interval)


def maybe_checkpoint(path: Path, rows: list[dict[str, Any]], checkpoint_every: int, next_checkpoint: int) -> int:
    done = len(rows)
    if checkpoint_every > 0 and next_checkpoint > 0 and done >= next_checkpoint:
        write_rows_atomic(path, rows, required=False)
        while next_checkpoint <= done:
            next_checkpoint += int(checkpoint_every)
    return next_checkpoint


def maybe_progress(done: int, total: int, progress_every: int, next_progress: int) -> int:
    if progress_every > 0 and (next_progress > 0 and done >= next_progress or done == total):
        print(f"[{done}/{total}] replanned assignment edges", flush=True)
        while next_progress <= done:
            next_progress += int(progress_every)
    return next_progress


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    candidates = [path, pending_path(path)]
    existing = [candidate for candidate in candidates if candidate.exists()]
    if not existing:
        return []
    source = max(existing, key=count_csv_rows)
    if source != path:
        print(f"resuming from checkpoint fallback: {source}", flush=True)
    with source.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def pending_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.pending{path.suffix}")


def write_rows_atomic(path: Path, rows: list[dict[str, Any]], required: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(6 if required else 1):
        try:
            tmp_path.replace(path)
            return
        except PermissionError:
            if required and attempt < 5:
                time.sleep(2.0)
                continue
            fallback = pending_path(path)
            tmp_path.replace(fallback)
            print(f"warning: could not replace locked {path}; wrote checkpoint fallback {fallback}", flush=True)
            if required:
                raise
            return


def write_summary(path: Path, instance_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], args: argparse.Namespace, source: str) -> None:
    instances = pd.DataFrame(instance_rows)
    pairs = pd.DataFrame(pair_rows)
    summary = {
        "source_benchmark": source,
        "label": args.label or f"beta_{args.planner_beta:g}",
        "planner_beta": float(args.planner_beta),
        "instances": int(len(instances)),
        "edges": int(len(pairs)),
        "mean_true_planner_runtime_sec": float(instances["true_planner_runtime_sec"].astype(float).mean()),
        "mean_edge_length": float(pairs["length"].astype(float).mean()),
        "mean_edge_risk": float(pairs["risk"].astype(float).mean()),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
