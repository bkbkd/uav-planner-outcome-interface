from __future__ import annotations

import numpy as np

from src.envs.scenario import Scenario


def plot_scenario(scenario: Scenario, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    image = scenario.threat_field.plot(ax=ax)
    scenario.obstacle_map.plot(ax=ax)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    return ax, image


def plot_path(path: np.ndarray, ax=None, color: str = "#00e5ff", linewidth: float = 2.2):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.plot(path[:, 0], path[:, 1], color=color, linewidth=linewidth)
    return ax

