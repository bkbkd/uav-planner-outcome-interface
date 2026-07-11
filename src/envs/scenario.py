from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.envs.obstacle_map import ObstacleMap
from src.envs.threat_field import ThreatField


@dataclass
class Scenario:
    width: float
    height: float
    threat_field: ThreatField
    obstacle_map: ObstacleMap

    def in_bounds(self, x: float, y: float, margin: float = 0.0) -> bool:
        return margin <= x <= self.width - margin and margin <= y <= self.height - margin

    def risk_at(self, x: float, y: float) -> float:
        return self.threat_field.value_at(x, y)

    def risk_values(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.threat_field.values_at(x, y)

    def is_state_valid(self, x: float, y: float, margin: float = 0.0) -> bool:
        return self.in_bounds(x, y, margin=margin) and not self.obstacle_map.is_collision(x, y, margin=margin)
