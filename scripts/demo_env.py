from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import matplotlib.pyplot as plt

from src.envs.obstacle_map import CircleObstacle, ObstacleMap, RectangleObstacle
from src.envs.scenario import Scenario
from src.envs.threat_field import ThreatField
from src.utils.visualization import plot_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=Path, default=None, help="Optional path for saving the figure.")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive Matplotlib window.")
    args = parser.parse_args()

    width, height = 1000.0, 1000.0
    threat = ThreatField.random_gaussian(width, height, resolution=2.0, n_sources=6, seed=7)
    obstacles = ObstacleMap(
        circles=[
            CircleObstacle(430.0, 470.0, 70.0),
            CircleObstacle(680.0, 260.0, 55.0),
        ],
        rectangles=[
            RectangleObstacle(180.0, 660.0, 320.0, 780.0),
            RectangleObstacle(690.0, 650.0, 830.0, 780.0),
        ],
    )
    scenario = Scenario(width, height, threat, obstacles)

    fig, ax = plt.subplots(figsize=(8, 7))
    _, image = plot_scenario(scenario, ax=ax)
    fig.colorbar(image, ax=ax, label="threat rate")
    ax.set_title("Threat field and obstacles")
    plt.tight_layout()
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=180)
        print(f"saved: {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
