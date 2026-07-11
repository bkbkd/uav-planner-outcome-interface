from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_generator import (
    DatasetConfig,
    MapConfig,
    ObstacleConfig,
    PlannerConfig,
    ThreatConfig,
    random_scenario,
    scenario_metadata,
)
from src.experiment_config import CURRENT_PLANNER


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a scenario-only metadata pool without edge labels.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=1000)
    parser.add_argument("--map-seed", type=int, required=True)
    parser.add_argument("--threat-layout", choices=["random", "central_barrier"], default="random")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.map_seed)
    config = DatasetConfig(
        n_samples=0,
        n_scenarios=args.scenarios,
        start_goal_mode="uniform",
        min_start_goal_distance=0.0,
        map_seed=args.map_seed,
        sample_seed=0,
        map=MapConfig(),
        threat=ThreatConfig(layout=args.threat_layout),
        obstacles=ObstacleConfig(),
        planner=PlannerConfig(**CURRENT_PLANNER),
    )
    scenario_records = []
    for scenario_id in range(args.scenarios):
        scenario = random_scenario(
            scenario_id=scenario_id,
            rng=rng,
            map_config=config.map,
            threat_config=config.threat,
            obstacle_config=config.obstacles,
        )
        scenario_records.append(scenario_metadata(scenario, scenario_id))
    metadata = {
        "config": asdict(config),
        "protocol": {"type": "scenario_pool", "contains_start_goal_samples": False},
        "scenarios": scenario_records,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"scenarios: {len(scenario_records)}")
    print(f"metadata: {args.output / 'metadata.json'}")


if __name__ == "__main__":
    main()
