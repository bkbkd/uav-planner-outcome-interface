from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_generator import CSV_COLUMNS
from src.experiment_config import CURRENT_MIN_START_GOAL_DISTANCE, CURRENT_PLANNER
from src.utils.protocol_seeds import FINAL_MASTER_SEED, derive_seed


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    ratio: float
    threat_layout: str
    start_goal_mode: str
    map_type: str


@dataclass(frozen=True)
class ShardJob:
    split: str
    distribution: DistributionSpec
    shard_index: int
    samples: int
    scenarios: int
    map_seed: int
    sample_seed: int
    output: Path


DISTRIBUTIONS = [
    DistributionSpec(
        name="natural_random",
        ratio=0.60,
        threat_layout="random",
        start_goal_mode="uniform",
        map_type="natural",
    ),
    DistributionSpec(
        name="hard_random",
        ratio=0.25,
        threat_layout="central_barrier",
        start_goal_mode="uniform",
        map_type="structured_hard",
    ),
    DistributionSpec(
        name="hard_crossing",
        ratio=0.15,
        threat_layout="central_barrier",
        start_goal_mode="barrier_crossing",
        map_type="structured_hard",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the final mixed edge dataset with clean split/distribution tags."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/datasets/final_edge_lattice20k_v1"))
    parser.add_argument("--train-samples", type=int, default=16_000)
    parser.add_argument("--val-samples", type=int, default=2_000)
    parser.add_argument("--test-samples", type=int, default=2_000)
    parser.add_argument("--edges-per-scenario", type=int, default=10)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--master-seed", type=int, default=FINAL_MASTER_SEED)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    write_plan(args.output / "generation_plan.json", args, jobs)

    print(f"output: {args.output}")
    print(f"jobs: {len(jobs)}")
    print(f"total samples: {sum(job.samples for job in jobs)}")
    print(f"workers: {args.workers}")
    if args.dry_run:
        for job in jobs:
            print(" ".join(command_for_job(job, args)))
        return

    run_jobs(jobs, args)
    merge_jobs(args.output, jobs, master_seed=args.master_seed)
    print(f"done: {args.output / 'samples.csv'}")
    print(f"metadata: {args.output / 'metadata.json'}")


def build_jobs(args: argparse.Namespace) -> list[ShardJob]:
    split_sizes = {
        "train": args.train_samples,
        "val": args.val_samples,
        "test": args.test_samples,
    }
    jobs: list[ShardJob] = []
    split_indices = {"train": 0, "val": 1, "test": 2}
    for split, split_samples in split_sizes.items():
        counts = allocate_counts(split_samples, [item.ratio for item in DISTRIBUTIONS])
        for dist_index, (distribution, total) in enumerate(zip(DISTRIBUTIONS, counts)):
            if total <= 0:
                continue
            shard_counts = shard_counts_for(total, args.shard_size)
            for shard_index, shard_samples in enumerate(shard_counts):
                scenarios = max(1, math.ceil(shard_samples / args.edges_per_scenario))
                seed_indices = (split_indices[split], dist_index, shard_index)
                output = args.output / "shards" / f"{split}_{distribution.name}_{shard_index:03d}"
                jobs.append(
                    ShardJob(
                        split=split,
                        distribution=distribution,
                        shard_index=shard_index,
                        samples=shard_samples,
                        scenarios=scenarios,
                        map_seed=derive_seed("edge_map", *seed_indices, master_seed=args.master_seed),
                        sample_seed=derive_seed("edge_pair", *seed_indices, master_seed=args.master_seed),
                        output=output,
                    )
                )
    return jobs


def allocate_counts(total: int, ratios: list[float]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda idx: raw[idx] - counts[idx], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def shard_counts_for(total: int, shard_size: int) -> list[int]:
    if shard_size <= 0:
        return [total]
    counts = []
    remaining = total
    while remaining > 0:
        count = min(shard_size, remaining)
        counts.append(count)
        remaining -= count
    return counts


def run_jobs(jobs: list[ShardJob], args: argparse.Namespace) -> None:
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_one_job, job, args): job for job in jobs}
        completed = 0
        for future in as_completed(futures):
            job = futures[future]
            future.result()
            completed += 1
            print(
                f"[{completed:03d}/{len(jobs):03d}] completed "
                f"{job.split}/{job.distribution.name}/shard{job.shard_index:03d}",
                flush=True,
            )


def run_one_job(job: ShardJob, args: argparse.Namespace) -> None:
    job.output.mkdir(parents=True, exist_ok=True)
    samples_path = job.output / "samples.csv"
    if samples_path.exists() and count_csv_rows(samples_path) >= job.samples:
        return

    log_path = job.output / "generation.log"
    cmd = command_for_job(job, args)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n\n=== command ===\n")
        log.write(" ".join(cmd) + "\n")
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Shard failed ({result.returncode}): {job.output}. See {log_path}")


def command_for_job(job: ShardJob, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_dataset.py"),
        "--output",
        str(job.output),
        "--samples",
        str(job.samples),
        "--scenarios",
        str(job.scenarios),
        "--map-seed",
        str(job.map_seed),
        "--sample-seed",
        str(job.sample_seed),
        "--threat-layout",
        job.distribution.threat_layout,
        "--start-goal-mode",
        job.distribution.start_goal_mode,
        "--min-start-goal-distance",
        str(CURRENT_MIN_START_GOAL_DISTANCE),
        "--lattice-priority-mode",
        CURRENT_PLANNER["lattice_priority_mode"],
        "--max-iters",
        str(CURRENT_PLANNER["max_iters"]),
        "--omega-max",
        str(CURRENT_PLANNER["omega_max"]),
        "--primitive-dt-values",
        *(str(value) for value in CURRENT_PLANNER["primitive_dt_values"]),
        "--goal-tolerance",
        str(CURRENT_PLANNER["goal_tolerance"]),
        "--lattice-post-solution-expansion-limit",
        str(CURRENT_PLANNER["lattice_post_solution_expansion_limit"]),
        "--lattice-xy-resolution",
        str(CURRENT_PLANNER["lattice_xy_resolution"]),
        "--lattice-heading-bins",
        str(CURRENT_PLANNER["lattice_heading_bins"]),
        "--progress-every",
        str(args.progress_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
    ]
    if not args.no_resume:
        cmd.append("--resume")
    if CURRENT_PLANNER["free_start_heading"]:
        cmd.append("--free-start-heading")
    return cmd


def merge_jobs(output: Path, jobs: list[ShardJob], master_seed: int) -> None:
    merged_samples: list[dict[str, Any]] = []
    merged_scenarios: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    root_config: dict[str, Any] | None = None
    sample_offset = 0
    scenario_offset = 0

    for job in jobs:
        samples_path = job.output / "samples.csv"
        metadata_path = job.output / "metadata.json"
        if not samples_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Missing shard artifacts: {job.output}")
        samples = pd.read_csv(samples_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if root_config is None:
            root_config = dict(metadata["config"])
            root_config.pop("map_seed", None)
            root_config.pop("sample_seed", None)
        scenario_map = {
            int(item["scenario_id"]): scenario_offset + idx for idx, item in enumerate(metadata["scenarios"])
        }
        sample_map = {
            int(sample_id): sample_offset + idx
            for idx, sample_id in enumerate(samples["sample_id"].astype(int).tolist())
        }

        for item in metadata["scenarios"]:
            old_id = int(item["scenario_id"])
            scenario = dict(item)
            scenario["scenario_id"] = scenario_map[old_id]
            scenario["source_scenario_id"] = old_id
            scenario["source_shard"] = str(job.output.relative_to(output))
            scenario["split"] = job.split
            scenario["distribution"] = job.distribution.name
            scenario["map_type"] = job.distribution.map_type
            merged_scenarios.append(scenario)

        for record in samples.to_dict("records"):
            old_sample_id = int(record["sample_id"])
            old_scenario_id = int(record["scenario_id"])
            record["source_sample_id"] = old_sample_id
            record["source_scenario_id"] = old_scenario_id
            record["source_shard"] = str(job.output.relative_to(output))
            record["sample_id"] = sample_map[old_sample_id]
            record["scenario_id"] = scenario_map[old_scenario_id]
            record["split"] = job.split
            record["distribution"] = job.distribution.name
            record["map_type"] = job.distribution.map_type
            merged_samples.append(record)

        components.append(
            {
                "split": job.split,
                "distribution": job.distribution.name,
                "map_type": job.distribution.map_type,
                "threat_layout": job.distribution.threat_layout,
                "start_goal_mode": job.distribution.start_goal_mode,
                "shard_index": job.shard_index,
                "samples": int(len(samples)),
                "scenarios": int(len(metadata["scenarios"])),
                "map_seed": job.map_seed,
                "sample_seed": job.sample_seed,
                "path": str(job.output),
            }
        )
        sample_offset += len(samples)
        scenario_offset += len(metadata["scenarios"])

    write_merged_samples(output / "samples.csv", merged_samples)
    metadata = {
        "config": root_config or {},
        "protocol": {
            "name": "final_edge_mixed_lattice",
            "split_policy": "scenario_disjoint_by_independent_shards",
            "master_seed": int(master_seed),
            "train_ratio": {"natural_random": 0.60, "hard_random": 0.25, "hard_crossing": 0.15},
            "planner_teacher": "deterministic_lattice",
            "natural_point_sampling": {
                "distribution": "uniform_over_valid_free_space",
                "boundary_margin": 0.0,
                "minimum_euclidean_distance": float(CURRENT_MIN_START_GOAL_DISTANCE),
            },
        },
        "components": components,
        "scenarios": merged_scenarios,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(output / "summary.json", merged_samples, components)


def write_merged_samples(path: Path, records: list[dict[str, Any]]) -> None:
    extra_columns = ["split", "distribution", "map_type", "source_shard", "source_sample_id", "source_scenario_id"]
    columns = CSV_COLUMNS + [column for column in extra_columns if column not in CSV_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in columns})


def write_summary(path: Path, samples: list[dict[str, Any]], components: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(samples)
    groups = []
    for (split, distribution), group in df.groupby(["split", "distribution"], sort=True):
        groups.append(
            {
                "split": split,
                "distribution": distribution,
                "samples": int(len(group)),
                "scenarios": int(group["scenario_id"].nunique()),
                "feasible": int(pd.to_numeric(group["feasible"], errors="coerce").fillna(0).sum()),
                "mean_runtime_sec": float(pd.to_numeric(group["runtime_sec"], errors="coerce").mean()),
            }
        )
    summary = {
        "samples": int(len(df)),
        "scenarios": int(df["scenario_id"].nunique()),
        "feasible": int(pd.to_numeric(df["feasible"], errors="coerce").fillna(0).sum()),
        "by_split_distribution": groups,
        "components": components,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_plan(path: Path, args: argparse.Namespace, jobs: list[ShardJob]) -> None:
    plan = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "distributions": [distribution.__dict__ for distribution in DISTRIBUTIONS],
        "jobs": [
            {
                "split": job.split,
                "distribution": job.distribution.name,
                "samples": job.samples,
                "scenarios": job.scenarios,
                "map_seed": job.map_seed,
                "sample_seed": job.sample_seed,
                "output": str(job.output),
            }
            for job in jobs
        ],
    }
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


if __name__ == "__main__":
    main()
