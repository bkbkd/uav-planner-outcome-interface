from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.protocol_seeds import FINAL_MASTER_SEED, derive_seed


POOLS = ("development", "test_1", "test_2", "test_3")
PROFILES = (("beta150", "150", "fast"), ("beta1500", "1500", "safe"))


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    complete: Callable[[], bool]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the locked 10x10 assignment-scale benchmark with resumable planner calls."
    )
    parser.add_argument("--root", type=Path, default=Path("outputs/final_ras"))
    parser.add_argument("--instances-per-pool", type=int, default=50)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--master-seed", type=int, default=FINAL_MASTER_SEED)
    args = parser.parse_args()

    if args.agents != args.tasks:
        raise ValueError("The locked scale benchmark uses square assignment matrices.")
    if args.instances_per_pool <= 0 or args.instances_per_pool > 500:
        raise ValueError("instances-per-pool must be between 1 and the 500 predeclared pool maps.")

    scale_root = args.root / "assignment_scale" / f"{args.agents}x{args.tasks}"
    log_root = scale_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stages = build_stages(args, scale_root)
    event_log = log_root / "events.jsonl"

    for index, stage in enumerate(stages):
        if stage.complete():
            write_event(event_log, index, len(stages), stage.name, "skipped_complete")
            continue
        write_event(event_log, index, len(stages), stage.name, "running")
        log_path = log_root / f"{index:02d}_{stage.name}.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log.write(" ".join(stage.command) + "\n")
            log.flush()
            result = subprocess.run(
                stage.command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            write_event(event_log, index, len(stages), stage.name, "failed", result.returncode)
            raise SystemExit(f"Stage failed: {stage.name}; see {log_path}")
        if not stage.complete():
            write_event(event_log, index, len(stages), stage.name, "incomplete_after_exit")
            raise SystemExit(f"Stage did not produce its locked artifact: {stage.name}")
        write_event(event_log, index, len(stages), stage.name, "completed")

    write_event(event_log, len(stages), len(stages), "all", "completed")


def build_stages(args: argparse.Namespace, scale_root: Path) -> list[Stage]:
    py = sys.executable
    expected_pairs = args.instances_per_pool * args.agents * args.tasks
    stages: list[Stage] = []
    for pool_index, pool in enumerate(POOLS):
        scenario_pool = args.root / "scenario_pools" / pool
        if not (scenario_pool / "metadata.json").exists():
            raise FileNotFoundError(f"Missing locked scenario pool: {scenario_pool}")

        point_seed = derive_seed("scale_assignment_point", args.agents, pool_index, master_seed=args.master_seed)
        base = scale_root / pool / "beta650"
        stages.append(
            Stage(
                f"{pool}_beta650",
                [
                    py,
                    str(PROJECT_ROOT / "scripts/run_sharded_assignment_benchmark.py"),
                    "--scenario-pool",
                    str(scenario_pool),
                    "--output",
                    str(base),
                    "--instances",
                    str(args.instances_per_pool),
                    "--shards",
                    str(min(args.workers, args.instances_per_pool)),
                    "--workers",
                    str(args.workers),
                    "--point-seed",
                    str(point_seed),
                    "--agents",
                    str(args.agents),
                    "--tasks",
                    str(args.tasks),
                    "--checkpoint-every",
                    "1",
                    "--max-point-sampling-attempts",
                    "10",
                ],
                lambda base=base: benchmark_complete(base / "merged", args.instances_per_pool, expected_pairs),
            )
        )
        for profile, beta, label in PROFILES:
            output = scale_root / pool / profile
            stages.append(
                Stage(
                    f"{pool}_{profile}",
                    [
                        py,
                        str(PROJECT_ROOT / "scripts/replan_assignment_benchmark_profile.py"),
                        str(base / "merged"),
                        "--output",
                        str(output),
                        "--planner-beta",
                        beta,
                        "--label",
                        label,
                        "--workers",
                        str(args.workers),
                        "--resume",
                        "--checkpoint-every",
                        "100",
                        "--progress-every",
                        "100",
                    ],
                    lambda output=output: benchmark_complete(output, args.instances_per_pool, expected_pairs),
                )
            )
    return stages


def benchmark_complete(path: Path, expected_instances: int, expected_pairs: int) -> bool:
    summary_path = path / "benchmark_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        int(summary.get("instances", -1)) == expected_instances
        and int(summary.get("pairs", summary.get("edges", -1))) == expected_pairs
    )


def write_event(
    path: Path,
    index: int,
    count: int,
    stage: str,
    state: str,
    returncode: int | None = None,
) -> None:
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage_index": index,
        "stage_count": count,
        "stage": stage,
        "state": state,
        "returncode": returncode,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
