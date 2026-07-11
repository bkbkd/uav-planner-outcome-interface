from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.obstacle_map import ObstacleMap, RectangleObstacle
from src.envs.scenario import Scenario
from src.envs.threat_field import GaussianThreat, ThreatField
from src.planners.grid_risk_planner import GridRiskPlanner, GridRiskPlannerConfig


def main() -> None:
    open_scenario = Scenario(
        240.0,
        240.0,
        ThreatField(240.0, 240.0, 2.0, background=0.0),
        ObstacleMap(),
    )
    start = (15.0, 20.0, 1.2)
    goal = (215.0, 190.0)
    result = GridRiskPlanner(open_scenario, GridRiskPlannerConfig(beta=650.0)).plan(start, goal)
    assert result is not None
    assert abs(result.metrics["length"] - math.dist(start[:2], goal)) < 1e-9
    assert result.metrics["risk"] == 0.0

    obstacle_scenario = Scenario(
        240.0,
        240.0,
        ThreatField(
            240.0,
            240.0,
            2.0,
            sources=[GaussianThreat(120.0, 120.0, 0.08, 35.0)],
        ),
        ObstacleMap(rectangles=[RectangleObstacle(90.0, 70.0, 150.0, 170.0)]),
    )
    results = []
    for beta in (150.0, 650.0, 1500.0):
        planner = GridRiskPlanner(obstacle_scenario, GridRiskPlannerConfig(beta=beta))
        planned = planner.plan((20.0, 120.0, 0.0), (220.0, 120.0))
        assert planned is not None
        assert not obstacle_scenario.obstacle_map.path_collision(planned.path)
        assert abs(planned.metrics["objective"] - (
            planned.metrics["length"] + beta * planned.metrics["risk"]
        )) < 1e-8
        results.append(planned)

    lengths = np.asarray([item.metrics["length"] for item in results])
    risks = np.asarray([item.metrics["risk"] for item in results])
    assert np.all(np.diff(lengths) >= -1e-7), (lengths, risks)
    assert np.all(np.diff(risks) <= 1e-7), (lengths, risks)
    print("grid risk planner smoke passed")
    for beta, item in zip((150, 650, 1500), results):
        print(beta, item.metrics["length"], item.metrics["risk"], item.runtime_sec, item.expanded_nodes)


if __name__ == "__main__":
    main()
