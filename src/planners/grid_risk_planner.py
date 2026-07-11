from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.envs.scenario import Scenario
from src.planners.path_utils import evaluate_path, path_objective


@dataclass(frozen=True)
class GridRiskPlannerConfig:
    """Configuration for an exact shortest-path search on a fixed 2D grid graph."""

    resolution: float = 12.0
    speed: float = 28.0
    alpha: float = 1.0
    beta: float = 650.0
    obstacle_margin: float = 0.0
    collision_check_step: float = 2.0
    connector_radius_cells: float = 3.0
    connector_limit: int = 16

    def __post_init__(self) -> None:
        if self.resolution <= 0.0 or self.speed <= 0.0:
            raise ValueError("resolution and speed must be positive")
        if self.alpha < 0.0 or self.beta < 0.0:
            raise ValueError("length and risk weights must be non-negative")
        if self.collision_check_step <= 0.0:
            raise ValueError("collision_check_step must be positive")
        if self.connector_radius_cells <= 0.0 or self.connector_limit <= 0:
            raise ValueError("connector settings must be positive")


@dataclass(frozen=True)
class GridPlanResult:
    path: np.ndarray
    metrics: dict[str, float]
    runtime_sec: float
    expanded_nodes: int


@dataclass
class _GoalTree:
    goal: tuple[float, float]
    distances: np.ndarray
    next_nodes: np.ndarray
    expanded_nodes: int


class GridRiskPlanner:
    """Holonomic risk-aware grid planner with exact Dijkstra search on its graph."""

    _FORWARD_NEIGHBORS = ((1, 0), (0, 1), (1, 1), (-1, 1))

    def __init__(self, scenario: Scenario, config: GridRiskPlannerConfig | None = None) -> None:
        self.scenario = scenario
        self.config = config or GridRiskPlannerConfig()
        self.nx = int(math.floor(scenario.width / self.config.resolution)) + 1
        self.ny = int(math.floor(scenario.height / self.config.resolution)) + 1
        self.positions = self._build_positions()
        self.valid = self._build_valid_mask()
        self.adjacency = self._build_adjacency()
        self._goal_cache: dict[tuple[float, float], _GoalTree] = {}

    def plan(self, start: tuple[float, float, float], goal: tuple[float, float]) -> GridPlanResult | None:
        tic = time.perf_counter()
        if not self.scenario.is_state_valid(start[0], start[1], margin=self.config.obstacle_margin):
            return None
        if not self.scenario.is_state_valid(goal[0], goal[1], margin=self.config.obstacle_margin):
            return None

        direct = self._direct_path(start, goal)
        tree = self._goal_tree(goal)
        connectors = self._visible_connectors((start[0], start[1]))
        best: tuple[float, int] | None = None
        for node, length, risk in connectors:
            remaining = float(tree.distances[node])
            if not math.isfinite(remaining):
                continue
            objective = self.config.alpha * length + self.config.beta * risk + remaining
            candidate = (objective, node)
            if best is None or candidate < best:
                best = candidate

        candidates: list[tuple[float, np.ndarray]] = []
        if direct is not None:
            candidates.append((self._objective(direct), direct))
        if best is not None:
            path = self._reconstruct(start, goal, best[1], tree)
            candidates.append((self._objective(path), path))
        if not candidates:
            return None

        _, path = min(candidates, key=lambda item: (item[0], len(item[1]), tuple(item[1][:, :2].ravel())))
        metrics = evaluate_path(path, self.scenario, self.config.speed)
        metrics["objective"] = path_objective(
            path,
            self.scenario,
            self.config.speed,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=0.0,
        )
        return GridPlanResult(
            path=path,
            metrics=metrics,
            runtime_sec=time.perf_counter() - tic,
            expanded_nodes=tree.expanded_nodes,
        )

    def _build_positions(self) -> np.ndarray:
        positions = np.zeros((self.nx * self.ny, 2), dtype=float)
        for iy in range(self.ny):
            for ix in range(self.nx):
                positions[self._node_id(ix, iy)] = self._cell_position(ix, iy)
        return positions

    def _build_valid_mask(self) -> np.ndarray:
        return np.asarray(
            [
                self.scenario.is_state_valid(float(x), float(y), margin=self.config.obstacle_margin)
                for x, y in self.positions
            ],
            dtype=bool,
        )

    def _build_adjacency(self) -> tuple[tuple[tuple[int, float], ...], ...]:
        adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(self.positions))]
        for iy in range(self.ny):
            for ix in range(self.nx):
                current = self._node_id(ix, iy)
                if not self.valid[current]:
                    continue
                for dx, dy in self._FORWARD_NEIGHBORS:
                    jx, jy = ix + dx, iy + dy
                    if not (0 <= jx < self.nx and 0 <= jy < self.ny):
                        continue
                    neighbor = self._node_id(jx, jy)
                    if not self.valid[neighbor] or not self._segment_valid(current, neighbor):
                        continue
                    objective = self._segment_objective(self.positions[current], self.positions[neighbor])
                    adjacency[current].append((neighbor, objective))
                    adjacency[neighbor].append((current, objective))
        return tuple(tuple(sorted(edges)) for edges in adjacency)

    def _goal_tree(self, goal: tuple[float, float]) -> _GoalTree:
        key = (float(goal[0]), float(goal[1]))
        cached = self._goal_cache.get(key)
        if cached is not None:
            return cached

        distances = np.full(len(self.positions), np.inf, dtype=float)
        next_nodes = np.full(len(self.positions), -2, dtype=int)
        heap: list[tuple[float, int]] = []
        for node, length, risk in self._visible_connectors(goal):
            cost = self.config.alpha * length + self.config.beta * risk
            if cost + 1e-12 < distances[node]:
                distances[node] = cost
                next_nodes[node] = -1
                heapq.heappush(heap, (cost, node))

        expanded = 0
        while heap:
            cost, current = heapq.heappop(heap)
            if cost > distances[current] + 1e-12:
                continue
            expanded += 1
            for neighbor, edge_cost in self.adjacency[current]:
                new_cost = cost + edge_cost
                if new_cost + 1e-12 < distances[neighbor]:
                    distances[neighbor] = new_cost
                    next_nodes[neighbor] = current
                    heapq.heappush(heap, (new_cost, neighbor))
                elif abs(new_cost - distances[neighbor]) <= 1e-12 and current < next_nodes[neighbor]:
                    next_nodes[neighbor] = current

        tree = _GoalTree(key, distances, next_nodes, expanded)
        self._goal_cache[key] = tree
        return tree

    def _visible_connectors(self, point: tuple[float, float]) -> list[tuple[int, float, float]]:
        radius = self.config.connector_radius_cells * self.config.resolution
        distances = np.linalg.norm(self.positions - np.asarray(point, dtype=float), axis=1)
        candidates = np.flatnonzero(self.valid & (distances <= radius + 1e-9))
        order = sorted((float(distances[node]), int(node)) for node in candidates)
        connectors = []
        for distance, node in order:
            if self.scenario.obstacle_map.segment_collision(
                point,
                tuple(self.positions[node]),
                step=self.config.collision_check_step,
                margin=self.config.obstacle_margin,
            ):
                continue
            risk = self._segment_risk(np.asarray(point, dtype=float), self.positions[node])
            connectors.append((node, distance, risk))
            if len(connectors) >= self.config.connector_limit:
                break
        return connectors

    def _direct_path(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float],
    ) -> np.ndarray | None:
        if self.scenario.obstacle_map.segment_collision(
            (start[0], start[1]),
            goal,
            step=self.config.collision_check_step,
            margin=self.config.obstacle_margin,
        ):
            return None
        return self._path_with_headings([(start[0], start[1]), goal])

    def _reconstruct(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float],
        first_node: int,
        tree: _GoalTree,
    ) -> np.ndarray:
        points = [(float(start[0]), float(start[1]))]
        node = first_node
        visited: set[int] = set()
        while node >= 0:
            if node in visited:
                raise RuntimeError("cycle in Dijkstra successor tree")
            visited.add(node)
            points.append(tuple(float(value) for value in self.positions[node]))
            node = int(tree.next_nodes[node])
        points.append((float(goal[0]), float(goal[1])))
        deduplicated = [points[0]]
        for point in points[1:]:
            if math.dist(point, deduplicated[-1]) > 1e-9:
                deduplicated.append(point)
        return self._path_with_headings(deduplicated)

    def _path_with_headings(self, points: Iterable[tuple[float, float]]) -> np.ndarray:
        xy = np.asarray(tuple(points), dtype=float)
        headings = np.zeros(len(xy), dtype=float)
        if len(xy) > 1:
            delta = np.diff(xy, axis=0)
            headings[:-1] = np.arctan2(delta[:, 1], delta[:, 0])
            headings[-1] = headings[-2]
        return np.column_stack([xy, headings])

    def _objective(self, path: np.ndarray) -> float:
        return path_objective(
            path,
            self.scenario,
            self.config.speed,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=0.0,
        )

    def _segment_objective(self, start: np.ndarray, goal: np.ndarray) -> float:
        length = float(np.linalg.norm(goal - start))
        risk = self._segment_risk(start, goal)
        return self.config.alpha * length + self.config.beta * risk

    def _segment_risk(self, start: np.ndarray, goal: np.ndarray) -> float:
        midpoint = 0.5 * (start + goal)
        length = float(np.linalg.norm(goal - start))
        return float(self.scenario.risk_at(float(midpoint[0]), float(midpoint[1])) * length / self.config.speed)

    def _segment_valid(self, start: int, goal: int) -> bool:
        return not self.scenario.obstacle_map.segment_collision(
            tuple(self.positions[start]),
            tuple(self.positions[goal]),
            step=self.config.collision_check_step,
            margin=self.config.obstacle_margin,
        )

    def _node_id(self, ix: int, iy: int) -> int:
        return iy * self.nx + ix

    def _cell_position(self, ix: int, iy: int) -> tuple[float, float]:
        return (
            float(min(ix * self.config.resolution, self.scenario.width)),
            float(min(iy * self.config.resolution, self.scenario.height)),
        )
