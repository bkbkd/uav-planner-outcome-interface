from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GaussianThreat:
    x: float
    y: float
    amplitude: float
    sigma: float


class ThreatField:
    """Continuous threat-rate field backed by a high-resolution heat map."""

    def __init__(
        self,
        width: float,
        height: float,
        resolution: float,
        sources: Iterable[GaussianThreat] | None = None,
        background: float = 0.0,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.resolution = float(resolution)
        self.sources = list(sources or [])
        self.background = float(background)

        self.xs = np.arange(0.0, self.width + self.resolution, self.resolution)
        self.ys = np.arange(0.0, self.height + self.resolution, self.resolution)
        self.grid = self._build_grid()

    @classmethod
    def random_gaussian(
        cls,
        width: float,
        height: float,
        resolution: float,
        n_sources: int = 5,
        amplitude_range: tuple[float, float] = (0.01, 0.08),
        sigma_range: tuple[float, float] = (50.0, 140.0),
        seed: int | None = None,
        background: float = 0.0,
    ) -> "ThreatField":
        rng = np.random.default_rng(seed)
        margin = max(sigma_range[1], min(width, height) * 0.05)
        sources = [
            GaussianThreat(
                x=float(rng.uniform(margin, width - margin)),
                y=float(rng.uniform(margin, height - margin)),
                amplitude=float(rng.uniform(*amplitude_range)),
                sigma=float(rng.uniform(*sigma_range)),
            )
            for _ in range(n_sources)
        ]
        return cls(width, height, resolution, sources, background=background)

    def _build_grid(self) -> np.ndarray:
        xx, yy = np.meshgrid(self.xs, self.ys)
        grid = np.full_like(xx, self.background, dtype=float)
        for source in self.sources:
            dist2 = (xx - source.x) ** 2 + (yy - source.y) ** 2
            grid += source.amplitude * np.exp(-dist2 / (2.0 * source.sigma**2))
        return grid

    def value_at(self, x: float, y: float) -> float:
        """Bilinearly interpolate threat rate at a continuous coordinate."""
        x = float(np.clip(x, 0.0, self.width))
        y = float(np.clip(y, 0.0, self.height))

        gx = x / self.resolution
        gy = y / self.resolution
        x0 = int(np.floor(gx))
        y0 = int(np.floor(gy))
        x1 = min(x0 + 1, self.grid.shape[1] - 1)
        y1 = min(y0 + 1, self.grid.shape[0] - 1)
        x0 = min(x0, self.grid.shape[1] - 1)
        y0 = min(y0, self.grid.shape[0] - 1)

        tx = gx - x0
        ty = gy - y0
        v00 = self.grid[y0, x0]
        v10 = self.grid[y0, x1]
        v01 = self.grid[y1, x0]
        v11 = self.grid[y1, x1]
        return float((1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11)

    def values_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized bilinear interpolation for arrays of continuous coordinates."""
        x = np.clip(np.asarray(x, dtype=float), 0.0, self.width)
        y = np.clip(np.asarray(y, dtype=float), 0.0, self.height)

        gx = x / self.resolution
        gy = y / self.resolution
        x0 = np.floor(gx).astype(int)
        y0 = np.floor(gy).astype(int)
        x1 = np.minimum(x0 + 1, self.grid.shape[1] - 1)
        y1 = np.minimum(y0 + 1, self.grid.shape[0] - 1)
        x0 = np.minimum(x0, self.grid.shape[1] - 1)
        y0 = np.minimum(y0, self.grid.shape[0] - 1)

        tx = gx - x0
        ty = gy - y0
        v00 = self.grid[y0, x0]
        v10 = self.grid[y0, x1]
        v01 = self.grid[y1, x0]
        v11 = self.grid[y1, x1]
        return (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11

    def plot(self, ax=None, cmap: str = "inferno", alpha: float = 0.85):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(
            self.grid,
            extent=(0.0, self.width, 0.0, self.height),
            origin="lower",
            cmap=cmap,
            alpha=alpha,
            interpolation="bilinear",
        )
        ax.set_xlim(0.0, self.width)
        ax.set_ylim(0.0, self.height)
        ax.set_aspect("equal", adjustable="box")
        return image
