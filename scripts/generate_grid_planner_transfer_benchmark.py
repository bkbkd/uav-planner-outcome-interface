from __future__ import annotations

import argparse
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

from src.learning.features import scenario_from_metadata
from src.planners.grid_risk_planner import GridRiskPlanner, GridRiskPlannerConfig


PROFILES = (("fast", 150.0), ("balanced", 650.0), ("safe", 1500.0))
_METADATA: dict[str, Any] | None = None
_CONFIG: dict[str, Any] | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replan a locked assignment subset with a holonomic risk-aware grid Dijkstra planner."
    )
    parser.add_argument("source_benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=50)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--resolution", type=float, default=12.0)
    parser.add_argument("--collision-check-step", type=float, default=2.0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args()

    metadata = json.loads((args.source_benchmark / "source_metadata.json").read_text(encoding="utf-8"))
    instances = pd.read_csv(args.source_benchmark / "instances.csv").head(args.instances).copy()
    pairs = pd.read_csv(args.source_benchmark / "pairs.csv")
    keep = set(instances["instance_id"].astype(int))
    pairs = pairs[pairs["instance_id"].astype(int).isin(keep)].copy()
    expected = len(instances) * int(instances.iloc[0].n_agents) * int(instances.iloc[0].n_tasks)
    if len(pairs) != expected:
        raise ValueError(f"source subset has {len(pairs)} pairs, expected {expected}")

    config = GridRiskPlannerConfig(
        resolution=args.resolution,
        collision_check_step=args.collision_check_step,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_protocol(args.output, args, config, instances, pairs)

    completed = completed_instance_ids(args.output)
    jobs = []
    for instance in instances.itertuples(index=False):
        instance_id = int(instance.instance_id)
        if instance_id in completed:
            continue
        frame = pairs[pairs["instance_id"].astype(int) == instance_id]
        jobs.append((instance._asdict(), frame.to_dict("records")))

    existing = load_existing(args.output)
    results = list(existing)
    if jobs:
        with Pool(
            processes=max(1, int(args.workers)),
            initializer=init_worker,
            initargs=(metadata, asdict(config)),
        ) as pool:
            for index, batch in enumerate(pool.imap(generate_instance, jobs, chunksize=1), start=1):
                results.append(batch)
                if index % max(1, args.checkpoint_every) == 0:
                    write_outputs(args.output, instances, metadata, config, results)
                print(
                    f"[{len(completed) + index}/{len(instances)}] "
                    f"instance={batch['instance_id']} complete",
                    flush=True,
                )

    write_outputs(args.output, instances, metadata, config, results)
    print(f"saved grid-planner transfer benchmark: {args.output}")


def init_worker(metadata: dict[str, Any], config: dict[str, Any]) -> None:
    global _METADATA, _CONFIG
    _METADATA = metadata
    _CONFIG = config


def generate_instance(job: tuple[dict[str, Any], list[dict[str, Any]]]) -> dict[str, Any]:
    if _METADATA is None or _CONFIG is None:
        raise RuntimeError("worker is not initialized")
    instance, rows = job
    instance_id = int(instance["instance_id"])
    scenario_id = int(instance["scenario_id"])
    scenario = scenario_from_metadata(_METADATA, scenario_id)
    output: dict[str, Any] = {"instance_id": instance_id, "profiles": {}}

    for label, beta in PROFILES:
        tic = time.perf_counter()
        planner = GridRiskPlanner(
            scenario,
            GridRiskPlannerConfig(**{**_CONFIG, "beta": beta}),
        )
        build_sec = time.perf_counter() - tic
        profile_rows = []
        query_sec = 0.0
        for row in rows:
            start = (float(row["start_x"]), float(row["start_y"]), float(row["start_theta"]))
            goal = (float(row["goal_x"]), float(row["goal_y"]))
            result = planner.plan(start, goal)
            if result is None:
                raise RuntimeError(
                    f"grid planner failed: instance={instance_id} agent={row['agent_id']} "
                    f"task={row['task_id']} beta={beta:g}"
                )
            query_sec += result.runtime_sec
            updated = dict(row)
            updated.update(
                {
                    "length": result.metrics["length"],
                    "time": result.metrics["time"],
                    "risk": result.metrics["risk"],
                    "turning": result.metrics["turning"],
                    "objective": result.metrics["objective"],
                    "true_bid": result.metrics["length"] + beta * result.metrics["risk"],
                    "baseline_bid": float(row["euclidean_distance"]) + beta * float(row["straight_line_risk"]),
                    "runtime_sec": result.runtime_sec,
                    "expanded_nodes": result.expanded_nodes,
                    "feasible": 1,
                }
            )
            profile_rows.append(normalize_record(updated))
        output["profiles"][label] = {
            "beta": beta,
            "build_sec": build_sec,
            "query_sec": query_sec,
            "rows": profile_rows,
        }
    return output


def completed_instance_ids(output: Path) -> set[int]:
    path = output / "checkpoint.jsonl"
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(int(json.loads(line)["instance_id"]))
    return completed


def load_existing(output: Path) -> list[dict[str, Any]]:
    path = output / "checkpoint.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_outputs(
    output: Path,
    source_instances: pd.DataFrame,
    metadata: dict[str, Any],
    config: GridRiskPlannerConfig,
    results: list[dict[str, Any]],
) -> None:
    ordered = sorted(results, key=lambda item: int(item["instance_id"]))
    checkpoint = output / "checkpoint.jsonl"
    atomic_text(checkpoint, "".join(json.dumps(item, sort_keys=True) + "\n" for item in ordered))

    for label, beta in PROFILES:
        profile_dir = output / label
        profile_dir.mkdir(parents=True, exist_ok=True)
        pair_rows = [row for item in ordered for row in item["profiles"][label]["rows"]]
        pair_rows.sort(key=lambda row: (int(row["instance_id"]), int(row["agent_id"]), int(row["task_id"])))
        instance_rows = []
        for instance in source_instances.itertuples(index=False):
            matching = next((item for item in ordered if int(item["instance_id"]) == int(instance.instance_id)), None)
            if matching is None:
                continue
            row = normalize_record(instance._asdict())
            timings = matching["profiles"][label]
            row["true_planner_runtime_sec"] = float(timings["build_sec"] + timings["query_sec"])
            row["graph_build_runtime_sec"] = float(timings["build_sec"])
            instance_rows.append(row)
        pd.DataFrame(instance_rows).to_csv(profile_dir / "instances.csv", index=False)
        pd.DataFrame(pair_rows).to_csv(profile_dir / "pairs.csv", index=False)
        profile_metadata = json.loads(json.dumps(metadata))
        profile_metadata["planner_family"] = "holonomic_8_connected_risk_aware_grid_dijkstra"
        profile_metadata["grid_planner"] = {**asdict(config), "beta": beta, "label": label}
        (profile_dir / "source_metadata.json").write_text(
            json.dumps(profile_metadata, indent=2), encoding="utf-8"
        )
        summary = {
            "instances": len(instance_rows),
            "pairs": len(pair_rows),
            "planner_family": profile_metadata["planner_family"],
            "profile": label,
            "beta": beta,
        }
        (profile_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_protocol(
    output: Path,
    args: argparse.Namespace,
    config: GridRiskPlannerConfig,
    instances: pd.DataFrame,
    pairs: pd.DataFrame,
) -> None:
    payload = {
        "status": "exact_transfer_gate",
        "source_benchmark": str(args.source_benchmark),
        "instance_selection": "first_n_predeclared_instances",
        "instances": int(len(instances)),
        "pairs_per_profile": int(len(pairs)),
        "profiles": [{"label": label, "beta": beta} for label, beta in PROFILES],
        "planner_family": "holonomic_8_connected_risk_aware_grid_dijkstra",
        "config": asdict(config),
        "worker_count_changes_content": False,
    }
    (output / "protocol.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in record.items():
        if isinstance(value, float) and math.isnan(value):
            out[key] = ""
        elif hasattr(value, "item"):
            out[key] = value.item()
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    main()
