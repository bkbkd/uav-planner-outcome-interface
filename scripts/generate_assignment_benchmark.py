from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_assignment import evaluate_true_pair_costs, sample_assignment_instance
from src.data.dataset_generator import PlannerConfig
from src.experiment_config import CURRENT_PLANNER
from src.learning.bid_targets import bid_cost_weights, planner_cost_weights
from src.learning.features import scenario_from_metadata


def metadata_with_effective_planner(metadata: dict, planner_config: PlannerConfig) -> dict:
    effective = deepcopy(metadata)
    effective.setdefault("config", {})["planner"] = asdict(planner_config)
    return effective


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a natural assignment benchmark with cached planner outcomes.")
    parser.add_argument("dataset_dir", type=Path, help="Scenario-pool directory containing metadata.json.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario-ids",
        required=True,
        help="Text file or comma-separated ids. The first N ids map one-to-one to the N benchmark instances.",
    )
    parser.add_argument("--instances", type=int, default=1000)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--point-seed", type=int, required=True)
    parser.add_argument("--instance-offset", type=int, default=0)
    parser.add_argument("--max-point-sampling-attempts", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.tasks < args.agents:
        raise ValueError("--tasks must be >= --agents.")
    if args.instances <= 0 or args.max_point_sampling_attempts <= 0:
        raise ValueError("--instances and --max-point-sampling-attempts must be positive.")

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    scenario_ids = parse_scenario_ids(args.scenario_ids, metadata)
    if len(scenario_ids) < args.instances:
        raise ValueError(f"Need at least {args.instances} scenario ids for one-map-one-instance generation; got {len(scenario_ids)}.")
    scenario_ids = scenario_ids[: args.instances]
    planner_config = PlannerConfig(**CURRENT_PLANNER)
    metadata = metadata_with_effective_planner(metadata, planner_config)

    instance_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    attempted_pair_sets = 0
    if args.resume:
        instance_rows, pair_rows, attempted_pair_sets = load_existing_benchmark(args.output, metadata, scenario_ids)
        if len(instance_rows) > args.instances:
            raise ValueError(f"Existing benchmark has {len(instance_rows)} instances, more than requested {args.instances}.")
        print(f"resuming {args.output}: {len(instance_rows)}/{args.instances} instances", flush=True)

    for local_instance_id in range(len(instance_rows), args.instances):
        instance_id = int(args.instance_offset + local_instance_id)
        scenario_id = int(scenario_ids[local_instance_id])
        scenario = scenario_from_metadata(metadata, scenario_id)
        rng = np.random.default_rng(np.random.SeedSequence(int(args.point_seed), spawn_key=(instance_id,)))
        accepted = False
        for point_attempt in range(1, args.max_point_sampling_attempts + 1):
            attempted_pair_sets += 1
            agents, tasks = sample_assignment_instance(scenario, args.agents, args.tasks, rng)
            base_seed = int(rng.integers(0, 2**31 - 1))
            pair_df, true_metrics = evaluate_true_pair_costs(
                scenario=scenario,
                metadata=metadata,
                scenario_id=scenario_id,
                instance_id=instance_id,
                agents=agents,
                tasks=tasks,
                planner_config=planner_config,
                base_seed=base_seed,
            )
            if pair_df is None:
                continue

            pair_rows.extend(pair_df.to_dict("records"))
            instance_rows.append(
                {
                    "instance_id": instance_id,
                    "scenario_id": scenario_id,
                    "point_mode": "uniform",
                    "n_agents": args.agents,
                    "n_tasks": args.tasks,
                    "agents_json": json.dumps(agents),
                    "tasks_json": json.dumps(tasks),
                    "point_sampling_attempt": point_attempt,
                    "true_planner_runtime_sec": true_metrics["runtime_sec"],
                    "base_seed": base_seed,
                }
            )
            accepted = True
            break

        if not accepted:
            write_benchmark_files(
                args.output, metadata, instance_rows, pair_rows, scenario_ids, args, attempted_pair_sets, partial=True
            )
            raise RuntimeError(
                f"Scenario {scenario_id} did not yield a complete {args.agents}x{args.tasks} planner-feasible matrix "
                f"after {args.max_point_sampling_attempts} natural point samples."
            )

        if args.checkpoint_every > 0 and len(instance_rows) % args.checkpoint_every == 0:
            write_benchmark_files(
                args.output, metadata, instance_rows, pair_rows, scenario_ids, args, attempted_pair_sets, partial=True
            )
        print(
            f"[{local_instance_id + 1:04d}/{args.instances:04d}] scenario={scenario_id} "
            f"point_attempt={instance_rows[-1]['point_sampling_attempt']} "
            f"runtime={instance_rows[-1]['true_planner_runtime_sec']:.2f}s",
            flush=True,
        )

    summary = write_benchmark_files(
        args.output, metadata, instance_rows, pair_rows, scenario_ids, args, attempted_pair_sets, partial=False
    )
    print(json.dumps(summary, indent=2))
    print(f"saved: {args.output}")


def write_benchmark_files(
    output_dir: Path,
    metadata: dict,
    instance_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    scenario_ids: list[int],
    args: argparse.Namespace,
    attempted_pair_sets: int,
    partial: bool,
) -> dict[str, Any]:
    write_rows(output_dir / "instances.csv", instance_rows)
    write_rows(output_dir / "pairs.csv", pair_rows)
    (output_dir / "source_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary = {
        "instances": len(instance_rows),
        "pairs": len(pair_rows),
        "attempted_pair_sets": attempted_pair_sets,
        "scenario_ids": scenario_ids,
        "scenario_protocol": {
            "distribution": "natural_random",
            "mapping": "one_map_per_instance",
            "assignment_test_scenario_ids": scenario_ids,
        },
        "point_sampling_protocol": {
            "distribution": "uniform_over_valid_free_space",
            "boundary_margin": 0.0,
            "minimum_euclidean_distance": 0.0,
            "point_seed": int(args.point_seed),
            "global_instance_offset": int(args.instance_offset),
        },
        "args": {
            **vars(args),
            "dataset_dir": str(args.dataset_dir),
            "output": str(args.output),
        },
        "planner_cost_weights": dict(zip(["alpha", "beta", "gamma"], planner_cost_weights(metadata))),
        "bid_cost_weights": dict(zip(["alpha", "beta", "gamma"], bid_cost_weights(metadata))),
        "point_sampling_attempt": distribution([row["point_sampling_attempt"] for row in instance_rows]),
        "true_planner_runtime_sec": distribution([row["true_planner_runtime_sec"] for row in instance_rows]),
    }
    path = output_dir / ("benchmark_summary_partial.json" if partial else "benchmark_summary.json")
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_scenario_ids(spec: str, metadata: dict) -> list[int]:
    path = Path(spec)
    if path.exists():
        tokens: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                tokens.extend(line.replace(",", " ").split())
    else:
        tokens = spec.replace(",", " ").split()
    scenario_ids = [int(token) for token in tokens]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Scenario ids must be unique for one-map-one-instance generation.")
    available = {int(item["scenario_id"]) for item in metadata["scenarios"]}
    missing = [item for item in scenario_ids if item not in available]
    if missing:
        raise ValueError(f"Scenario ids are not present in metadata: {missing}")
    if not scenario_ids:
        raise ValueError("No scenario ids were provided.")
    return scenario_ids


def load_existing_benchmark(
    output_dir: Path,
    metadata: dict,
    scenario_ids: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    instances_path = output_dir / "instances.csv"
    pairs_path = output_dir / "pairs.csv"
    summary_path = output_dir / "benchmark_summary.json"
    if not summary_path.exists():
        summary_path = output_dir / "benchmark_summary_partial.json"
    if not instances_path.exists() or not pairs_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Cannot resume because benchmark files are missing in {output_dir}.")
    instances = pd.read_csv(instances_path)
    pairs = pd.read_csv(pairs_path)
    instance_offset = int(summary_args(summary_path).get("instance_offset", 0))
    expected_ids = list(range(instance_offset, instance_offset + len(instances)))
    if instances["instance_id"].astype(int).tolist() != expected_ids:
        raise ValueError("Cannot resume a benchmark with non-contiguous instance ids.")
    if instances["scenario_id"].astype(int).tolist() != scenario_ids[: len(instances)]:
        raise ValueError("Cannot resume: existing one-map-one-instance scenario mapping differs from this command.")
    validate_existing_pair_rows(pairs, metadata, output_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return instances.to_dict("records"), pairs.to_dict("records"), int(summary.get("attempted_pair_sets", len(instances)))


def summary_args(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")).get("args", {})


def validate_existing_pair_rows(pairs: pd.DataFrame, metadata: dict, output_dir: Path) -> None:
    required = {"baseline_bid", "euclidean_distance", "straight_line_risk", "true_bid", "length", "risk"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Cannot resume {output_dir}; pairs are missing columns: {sorted(missing)}")
    alpha, beta, _ = bid_cost_weights(metadata)
    expected_baseline = alpha * pairs["euclidean_distance"].astype(float) + beta * pairs["straight_line_risk"].astype(float)
    expected_true = alpha * pairs["length"].astype(float) + beta * pairs["risk"].astype(float)
    if float(np.max(np.abs(pairs["baseline_bid"].astype(float) - expected_baseline))) > 1e-4:
        raise ValueError(f"Cannot resume {output_dir}; baseline bid semantics differ.")
    if float(np.max(np.abs(pairs["true_bid"].astype(float) - expected_true))) > 1e-4:
        raise ValueError(f"Cannot resume {output_dir}; true bid semantics differ.")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


if __name__ == "__main__":
    main()
