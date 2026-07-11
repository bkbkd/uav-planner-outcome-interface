from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CircleObstacle:
    x: float
    y: float
    radius: float

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= (self.radius + margin) ** 2


@dataclass(frozen=True)
class RectangleObstacle:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (
            self.x_min - margin <= x <= self.x_max + margin
            and self.y_min - margin <= y <= self.y_max + margin
        )


class ObstacleMap:
    def __init__(
        self,
        circles: Iterable[CircleObstacle] | None = None,
        rectangles: Iterable[RectangleObstacle] | None = None,
    ) -> None:
        self.circles = list(circles or [])
        self.rectangles = list(rectangles or [])

    def is_collision(self, x: float, y: float, margin: float = 0.0) -> bool:
        return any(obs.contains(x, y, margin) for obs in self.circles) or any(
            obs.contains(x, y, margin) for obs in self.rectangles
        )

    def path_collision(self, path: np.ndarray, margin: float = 0.0) -> bool:
        return any(self.is_collision(float(x), float(y), margin) for x, y in path[:, :2])

    def segment_collision(
        self,
        p0: tuple[float, float],
        p1: tuple[float, float],
        step: float = 5.0,
        margin: float = 0.0,
    ) -> bool:
        distance = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        n = max(2, int(np.ceil(distance / step)) + 1)
        xs = np.linspace(p0[0], p1[0], n)
        ys = np.linspace(p0[1], p1[1], n)
        return any(self.is_collision(float(x), float(y), margin) for x, y in zip(xs, ys))

    def plot(self, ax=None, facecolor: str = "#4b5563", edgecolor: str = "white", alpha: float = 0.8):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))

        for obstacle in self.circles:
            ax.add_patch(
                Circle(
                    (obstacle.x, obstacle.y),
                    obstacle.radius,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=1.0,
                    alpha=alpha,
                )
            )
        for obstacle in self.rectangles:
            ax.add_patch(
                Rectangle(
                    (obstacle.x_min, obstacle.y_min),
                    obstacle.x_max - obstacle.x_min,
                    obstacle.y_max - obstacle.y_min,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=1.0,
                    alpha=alpha,
                )
            )
        return ax

