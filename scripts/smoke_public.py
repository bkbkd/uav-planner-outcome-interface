"""Run public, data-free checks for the core repository components."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.assignment import solve_linear_assignment


def run_script(*arguments: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        [sys.executable, *arguments],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    with tempfile.TemporaryDirectory(prefix="uav-interface-smoke-") as directory:
        figure = Path(directory) / "scenario.png"
        run_script("scripts/demo_env.py", "--no-show", "--save", str(figure), env=env)
        if not figure.is_file() or figure.stat().st_size == 0:
            raise RuntimeError("Environment rendering did not produce a figure.")

    run_script("scripts/smoke_grid_risk_planner.py", env=env)

    costs = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0]])
    assignment, total = solve_linear_assignment(costs)
    if assignment != (1, 0) or not np.isclose(total, 3.0):
        raise RuntimeError(f"Unexpected assignment smoke result: {assignment}, {total}")

    print("public smoke checks passed")


if __name__ == "__main__":
    main()

