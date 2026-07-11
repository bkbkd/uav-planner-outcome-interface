from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable sharded assignment benchmark.")
    parser.add_argument("--scenario-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=1000)
    parser.add_argument("--shards", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--point-seed", type=int, required=True)
    parser.add_argument("--scenario-offset", type=int, default=0)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--max-point-sampling-attempts", type=int, default=10)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    shard_dirs = [args.output / "shards" / f"shard_{idx:03d}" for idx in range(args.shards)]
    scenario_files = write_shard_scenario_files(args, shard_dirs)

    if not args.merge_only:
        counts = shard_counts(args.instances, args.shards)
        offsets = [sum(counts[:idx]) for idx in range(len(counts))]
        jobs = [
            (idx, shard_dirs[idx], count, scenario_files[idx], offsets[idx])
            for idx, count in enumerate(counts)
        ]
        run_jobs(args, jobs)

    completed = [path for path in shard_dirs if (path / "benchmark_summary.json").exists()]
    if len(completed) == args.shards:
        merge(args.output, shard_dirs)
    else:
        print(f"not merging yet: {len(completed)}/{args.shards} shards complete")
        print(f"resume with the same command, or merge later with --merge-only")


def shard_counts(total: int, shards: int) -> list[int]:
    base = total // shards
    extra = total % shards
    return [base + (1 if idx < extra else 0) for idx in range(shards)]


def write_shard_scenario_files(args: argparse.Namespace, shard_dirs: list[Path]) -> list[Path]:
    metadata = json.loads((args.scenario_pool / "metadata.json").read_text(encoding="utf-8"))
    scenario_ids = [int(item["scenario_id"]) for item in metadata["scenarios"]]
    if args.scenario_offset < 0:
        raise ValueError("--scenario-offset must be non-negative.")
    if args.scenario_offset + args.instances > len(scenario_ids):
        raise ValueError(
            f"One-map-one-instance benchmark needs at least {args.instances} scenarios; "
            f"with offset {args.scenario_offset}; found {len(scenario_ids)} in {args.scenario_pool}."
        )
    scenario_ids = scenario_ids[args.scenario_offset : args.scenario_offset + args.instances]
    counts = shard_counts(args.instances, args.shards)
    files: list[Path] = []
    offset = 0
    for idx, (shard_dir, count) in enumerate(zip(shard_dirs, counts)):
        shard_dir.mkdir(parents=True, exist_ok=True)
        ids = scenario_ids[offset : offset + count]
        offset += count
        path = shard_dir / "scenario_ids.txt"
        path.write_text("\n".join(str(item) for item in ids) + "\n", encoding="utf-8")
        files.append(path)
    return files


def run_jobs(args: argparse.Namespace, jobs: list[tuple[int, Path, int, Path, int]]) -> None:
    stop_monitor = threading.Event()
    monitor = threading.Thread(target=monitor_progress, args=(jobs, stop_monitor), daemon=True)
    monitor.start()
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(run_one, args, idx, path, count, scenario_file, instance_offset): (idx, path, count)
                for idx, path, count, scenario_file, instance_offset in jobs
            }
            completed = 0
            for future in as_completed(futures):
                idx, path, count = futures[future]
                future.result()
                completed += 1
                print_assignment_progress(jobs, prefix=f"[{completed:03d}/{len(jobs):03d}] shard_{idx:03d} complete")
    finally:
        stop_monitor.set()
        monitor.join(timeout=2.0)


def monitor_progress(jobs: list[tuple[int, Path, int, Path, int]], stop: threading.Event) -> None:
    while not stop.wait(60.0):
        print_assignment_progress(jobs)


def print_assignment_progress(jobs: list[tuple[int, Path, int, Path, int]], prefix: str = "assignment progress") -> None:
    total = sum(count for _, _, count, _, _ in jobs)
    done = sum(shard_progress(path, count) for _, path, count, _, _ in jobs)
    complete = sum(1 for _, path, _, _, _ in jobs if (path / "benchmark_summary.json").exists())
    print(f"{prefix}: instances={done}/{total} complete_shards={complete}/{len(jobs)}", flush=True)


def shard_progress(path: Path, target: int) -> int:
    if (path / "benchmark_summary.json").exists():
        return target
    log_path = path / "generation.log"
    if not log_path.exists():
        return 0
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    matches = re.findall(r"\[(\d+)/(\d+)\]", text)
    if not matches:
        return 0
    done, _ = matches[-1]
    return min(target, max(0, int(done)))


def run_one(
    args: argparse.Namespace,
    shard_index: int,
    output: Path,
    instances: int,
    scenario_file: Path,
    instance_offset: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    final_summary = output / "benchmark_summary.json"
    if final_summary.exists():
        summary = json.loads(final_summary.read_text(encoding="utf-8"))
        if int(summary.get("instances", -1)) != instances:
            raise ValueError(
                f"Completed shard {output} contains {summary.get('instances')} instances; expected {instances}."
            )
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_assignment_benchmark.py"),
        str(args.scenario_pool),
        "--output",
        str(output),
        "--instances",
        str(instances),
        "--agents",
        str(args.agents),
        "--tasks",
        str(args.tasks),
        "--point-seed",
        str(args.point_seed),
        "--instance-offset",
        str(instance_offset),
        "--scenario-ids",
        str(scenario_file),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--max-point-sampling-attempts",
        str(args.max_point_sampling_attempts),
    ]
    if (output / "instances.csv").exists() and (output / "benchmark_summary_partial.json").exists():
        cmd.append("--resume")
    log_path = output / "generation.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n\n=== command ===\n")
        log.write(" ".join(cmd) + "\n")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Shard {shard_index} failed with {result.returncode}; see {log_path}")


def merge(output: Path, shard_dirs: list[Path]) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "merge_assignment_benchmark_shards.py"),
        *[str(path) for path in shard_dirs],
        "--output",
        str(output / "merged"),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
