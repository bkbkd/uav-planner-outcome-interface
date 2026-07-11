from __future__ import annotations

import argparse
import csv
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


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    complete: Callable[[], bool]


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-safe serial orchestrator for all final data assets.")
    parser.add_argument("--root", type=Path, default=Path("outputs/final_ras"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--master-seed", type=int, default=FINAL_MASTER_SEED)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    logs = args.root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = args.root / "data_generation_status.json"
    stages = build_stages(args)

    for index, stage in enumerate(stages):
        if stage.complete():
            write_status(status_path, stages, index, stage.name, "skipped_complete")
            continue
        write_status(status_path, stages, index, stage.name, "running")
        log_path = logs / f"{index:02d}_{stage.name}.log"
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
            write_status(status_path, stages, index, stage.name, "failed", returncode=result.returncode)
            raise SystemExit(f"Stage failed: {stage.name}; see {log_path}")
        if not stage.complete():
            write_status(status_path, stages, index, stage.name, "incomplete_after_exit")
            raise SystemExit(f"Stage exited without complete artifacts: {stage.name}; see {log_path}")
        write_status(status_path, stages, index, stage.name, "completed")

    write_status(status_path, stages, len(stages), "all", "completed")
    print(f"all final data assets completed: {args.root}")


def build_stages(args: argparse.Namespace) -> list[Stage]:
    py = sys.executable
    root = args.root
    workers = str(max(1, int(args.workers)))
    stages = [
        Stage(
            "scenario_pools",
            [py, str(PROJECT_ROOT / "scripts/generate_final_scenario_pools.py"), "--output", str(root / "scenario_pools"), "--master-seed", str(args.master_seed)],
            lambda: scenario_pools_complete(root / "scenario_pools", args.master_seed),
        ),
        Stage(
            "edges_beta650",
            [
                py,
                str(PROJECT_ROOT / "scripts/generate_final_edge_dataset.py"),
                "--output",
                str(root / "edges_beta650"),
                "--train-samples",
                "16000",
                "--val-samples",
                "2000",
                "--test-samples",
                "2000",
                "--edges-per-scenario",
                "10",
                "--shard-size",
                "500",
                "--workers",
                workers,
                "--master-seed",
                str(args.master_seed),
                "--progress-every",
                "25",
                "--checkpoint-every",
                "100",
            ],
            lambda: edge_complete(root / "edges_beta650", 20_000),
        ),
    ]
    for profile, beta, label in (("beta150", "150", "fast"), ("beta1500", "1500", "safe")):
        output = root / f"edges_{profile}"
        stages.append(
            Stage(
                f"edges_{profile}",
                [
                    py,
                    str(PROJECT_ROOT / "scripts/replan_edge_dataset_profile.py"),
                    str(root / "edges_beta650"),
                    "--output",
                    str(output),
                    "--planner-beta",
                    beta,
                    "--label",
                    label,
                    "--workers",
                    workers,
                    "--resume",
                    "--checkpoint-every",
                    "100",
                    "--progress-every",
                    "100",
                ],
                lambda output=output: edge_complete(output, 20_000),
            )
        )

    for pool_index, pool in enumerate(("development", "test_1", "test_2", "test_3")):
        scenario_pool = root / "scenario_pools" / pool
        point_seed = derive_seed("assignment_point", pool_index, master_seed=args.master_seed)
        base = root / "assignment" / pool / "beta650"
        stages.append(
            Stage(
                f"assignment_{pool}_beta650",
                [
                    py,
                    str(PROJECT_ROOT / "scripts/run_sharded_assignment_benchmark.py"),
                    "--scenario-pool",
                    str(scenario_pool),
                    "--output",
                    str(base),
                    "--instances",
                    "500",
                    "--shards",
                    "20",
                    "--workers",
                    workers,
                    "--point-seed",
                    str(point_seed),
                    "--agents",
                    "5",
                    "--tasks",
                    "5",
                    "--checkpoint-every",
                    "1",
                    "--max-point-sampling-attempts",
                    "10",
                ],
                lambda base=base: assignment_complete(base / "merged", 500, 12_500),
            )
        )
        for profile, beta, label in (("beta150", "150", "fast"), ("beta1500", "1500", "safe")):
            output = root / "assignment" / pool / profile
            stages.append(
                Stage(
                    f"assignment_{pool}_{profile}",
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
                        workers,
                        "--resume",
                        "--checkpoint-every",
                        "100",
                        "--progress-every",
                        "100",
                    ],
                    lambda output=output: assignment_complete(output, 500, 12_500),
                )
            )

    audit_output = root / "final_data_audit.json"
    stages.append(
        Stage(
            "final_audit",
            [py, str(PROJECT_ROOT / "scripts/audit_final_data_protocol.py"), "--root", str(root), "--output", str(audit_output)],
            lambda: audit_complete(audit_output),
        )
    )
    return stages


def scenario_pools_complete(root: Path, master_seed: int) -> bool:
    manifest = load_json(root / "seed_manifest.json")
    if not manifest or int(manifest.get("master_seed", -1)) != int(master_seed):
        return False
    expected = {"development": 500, "test_1": 500, "test_2": 500, "test_3": 500, "rolling_test": 100}
    for name, count in expected.items():
        metadata = load_json(root / name / "metadata.json")
        if not metadata or len(metadata.get("scenarios", [])) != count:
            return False
    return True


def edge_complete(path: Path, expected_rows: int) -> bool:
    if not (path / "metadata.json").exists() or not (path / "summary.json").exists():
        return False
    return csv_rows(path / "samples.csv") == expected_rows


def assignment_complete(path: Path, expected_instances: int, expected_pairs: int) -> bool:
    summary = load_json(path / "benchmark_summary.json")
    if not summary:
        return False
    pair_count = summary.get("pairs", summary.get("edges", -1))
    return int(summary.get("instances", -1)) == expected_instances and int(pair_count) == expected_pairs


def audit_complete(path: Path) -> bool:
    report = load_json(path)
    return bool(report and report.get("ok"))


def csv_rows(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open(newline="", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_status(
    path: Path,
    stages: list[Stage],
    index: int,
    stage: str,
    state: str,
    returncode: int | None = None,
) -> None:
    payload = {
        "state": state,
        "stage_index": int(index),
        "stage_count": len(stages),
        "stage": stage,
        "returncode": returncode,
        "updated_at_unix": time.time(),
    }
    try:
        append_status_event(path.with_name("data_generation_events.jsonl"), payload)
    except OSError as exc:
        print(f"warning: could not write status update for {stage}/{state}: {exc}", flush=True)


def append_status_event(path: Path, payload: dict) -> None:
    line = json.dumps(payload, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


if __name__ == "__main__":
    main()
