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
        description="Generate paired grid-planner edge labels on an existing map-disjoint geometry split."
    )
    parser.add_argument("source_dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--resolution", type=float, default=12.0)
    parser.add_argument("--collision-check-step", type=float, default=2.0)
    parser.add_argument("--checkpoint-every-scenarios", type=int, default=25)
    parser.add_argument("--limit-scenarios", type=int)
    args = parser.parse_args()

    metadata = json.loads((args.source_dataset / "metadata.json").read_text(encoding="utf-8"))
    samples = pd.read_csv(args.source_dataset / "samples.csv")
    scenario_ids = list(dict.fromkeys(samples["scenario_id"].astype(int).tolist()))
    if args.limit_scenarios is not None:
        scenario_ids = scenario_ids[: int(args.limit_scenarios)]
        keep = set(scenario_ids)
        samples = samples[samples["scenario_id"].astype(int).isin(keep)].copy()
    config = GridRiskPlannerConfig(
        resolution=args.resolution,
        collision_check_step=args.collision_check_step,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_metadata(args.output, args.source_dataset, metadata, config, len(samples), len(scenario_ids))

    profile_rows = load_existing(args.output)
    completed = completed_scenarios(profile_rows)
    profile_rows = {
        label: [row for row in rows if int(row["scenario_id"]) in completed]
        for label, rows in profile_rows.items()
    }
    jobs = [
        (scenario_id, samples[samples["scenario_id"].astype(int) == scenario_id].to_dict("records"))
        for scenario_id in scenario_ids
        if scenario_id not in completed
    ]

    if jobs:
        with Pool(
            processes=max(1, int(args.workers)),
            initializer=init_worker,
            initargs=(metadata, asdict(config)),
        ) as pool:
            for index, batch in enumerate(pool.imap(generate_scenario, jobs, chunksize=1), start=1):
                for label in profile_rows:
                    profile_rows[label].extend(batch["profiles"][label])
                if index % max(1, int(args.checkpoint_every_scenarios)) == 0:
                    write_samples(args.output, profile_rows)
                print(
                    f"[{len(completed) + index}/{len(scenario_ids)}] "
                    f"scenario={batch['scenario_id']} complete",
                    flush=True,
                )

    write_samples(args.output, profile_rows)
    write_summaries(args.output, profile_rows)
    print(f"saved paired grid edge profiles: {args.output}")


def init_worker(metadata: dict[str, Any], config: dict[str, Any]) -> None:
    global _METADATA, _CONFIG
    _METADATA = metadata
    _CONFIG = config


def generate_scenario(job: tuple[int, list[dict[str, Any]]]) -> dict[str, Any]:
    if _METADATA is None or _CONFIG is None:
        raise RuntimeError("worker is not initialized")
    scenario_id, rows = job
    scenario = scenario_from_metadata(_METADATA, int(scenario_id))
    output: dict[str, Any] = {"scenario_id": int(scenario_id), "profiles": {}}
    for label, beta in PROFILES:
        build_tic = time.perf_counter()
        planner = GridRiskPlanner(
            scenario,
            GridRiskPlannerConfig(**{**_CONFIG, "beta": beta}),
        )
        build_sec = time.perf_counter() - build_tic
        generated = []
        for row in rows:
            start = (float(row["start_x"]), float(row["start_y"]), float(row["start_theta"]))
            goal = (float(row["goal_x"]), float(row["goal_y"]))
            result = planner.plan(start, goal)
            if result is None:
                raise RuntimeError(
                    f"grid planner failed: scenario={scenario_id} sample={row['sample_id']} beta={beta:g}"
                )
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
                    "graph_build_runtime_sec": build_sec / max(1, len(rows)),
                    "expanded_nodes": result.expanded_nodes,
                    "feasible": 1,
                    "failure_reason": "",
                }
            )
            generated.append(normalize_record(updated))
        output["profiles"][label] = generated
    return output


def load_existing(output: Path) -> dict[str, list[dict[str, Any]]]:
    existing = {}
    for label, _ in PROFILES:
        path = output / label / "samples.csv"
        existing[label] = pd.read_csv(path).to_dict("records") if path.exists() else []
    return existing


def completed_scenarios(profile_rows: dict[str, list[dict[str, Any]]]) -> set[int]:
    per_profile = []
    for rows in profile_rows.values():
        per_profile.append({int(row["scenario_id"]) for row in rows})
    if not per_profile:
        return set()
    return set.intersection(*per_profile)


def write_samples(output: Path, profile_rows: dict[str, list[dict[str, Any]]]) -> None:
    for label, rows in profile_rows.items():
        ordered = sorted(rows, key=lambda row: int(row["sample_id"]))
        frame = pd.DataFrame(ordered)
        path = output / label / "samples.csv"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)


def write_metadata(
    output: Path,
    source_dataset: Path,
    source_metadata: dict[str, Any],
    config: GridRiskPlannerConfig,
    samples: int,
    scenarios: int,
) -> None:
    for label, beta in PROFILES:
        profile_dir = output / label
        profile_dir.mkdir(parents=True, exist_ok=True)
        metadata = json.loads(json.dumps(source_metadata))
        metadata["config"]["planner"].update(
            {
                "speed": config.speed,
                "alpha": config.alpha,
                "beta": beta,
                "gamma": 0.0,
            }
        )
        protocol = dict(metadata.get("protocol", {}))
        protocol.update(
            {
                "planner_teacher": "holonomic_8_connected_risk_aware_grid_dijkstra",
                "transfer_source_dataset": str(source_dataset),
                "profile_geometry": "paired_across_all_grid_profiles",
                "grid_planner": {**asdict(config), "beta": beta, "label": label},
                "requested_samples": samples,
                "requested_scenarios": scenarios,
            }
        )
        metadata["protocol"] = protocol
        (profile_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_summaries(output: Path, profile_rows: dict[str, list[dict[str, Any]]]) -> None:
    for label, beta in PROFILES:
        frame = pd.DataFrame(profile_rows[label])
        summary = {
            "samples": int(len(frame)),
            "scenarios": int(frame["scenario_id"].nunique()) if len(frame) else 0,
            "feasible": int(frame["feasible"].astype(int).sum()) if len(frame) else 0,
            "planner_family": "holonomic_8_connected_risk_aware_grid_dijkstra",
            "profile": label,
            "beta": beta,
            "split_counts": frame["split"].value_counts().sort_index().to_dict() if len(frame) else {},
        }
        (output / label / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


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
