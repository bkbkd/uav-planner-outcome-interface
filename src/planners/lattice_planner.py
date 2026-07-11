from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math
import time
from typing import Any

import numpy as np

from src.dynamics.dubins_model import rollout_arc
from src.envs.scenario import Scenario
from src.experiment_config import CURRENT_PLANNER
from src.planners.path_utils import evaluate_path, path_length, path_objective, path_risk
from src.utils.geometry import euclidean_distance, wrap_angle


@dataclass
class LatticePlannerConfig:
    """Deterministic SE(2) state-lattice planner configuration."""

    speed: float = CURRENT_PLANNER["speed"]
    omega_max: float = CURRENT_PLANNER["omega_max"]
    primitive_dt_values: tuple[float, ...] = CURRENT_PLANNER["primitive_dt_values"]
    primitive_samples: int = CURRENT_PLANNER["primitive_samples"]
    n_actions: int = CURRENT_PLANNER["n_actions"]
    xy_resolution: float = CURRENT_PLANNER["lattice_xy_resolution"]
    heading_bins: int = CURRENT_PLANNER["lattice_heading_bins"]
    goal_tolerance: float = CURRENT_PLANNER["goal_tolerance"]
    max_expansions: int = CURRENT_PLANNER["max_iters"]
    post_solution_expansion_limit: int = CURRENT_PLANNER["lattice_post_solution_expansion_limit"]
    obstacle_margin: float = 0.0
    alpha: float = CURRENT_PLANNER["alpha"]
    beta: float = CURRENT_PLANNER["beta"]
    gamma: float = CURRENT_PLANNER["gamma"]
    use_grid_guidance: bool = True
    grid_guidance_resolution: float = 24.0
    grid_guidance_weight: float = CURRENT_PLANNER["lattice_grid_guidance_weight"]
    grid_guidance_risk_weight: float = CURRENT_PLANNER["beta"]
    grid_guidance_edge_samples: int = 5
    straight_line_heuristic_samples: int = 96
    clipped_straight_risk_rho0: float = 0.5
    capped_straight_risk_rho: float = 0.5
    continue_after_first_solution: bool = True
    free_start_heading: bool = CURRENT_PLANNER["free_start_heading"]
    priority_mode: str = CURRENT_PLANNER["lattice_priority_mode"]

    def __post_init__(self) -> None:
        self.primitive_dt_values = tuple(float(value) for value in self.primitive_dt_values)
        if not self.primitive_dt_values:
            raise ValueError("At least one lattice primitive dt must be provided.")
        if any(value <= 0.0 for value in self.primitive_dt_values):
            raise ValueError("Lattice primitive dt values must be positive.")


@dataclass
class LatticePlanningResult:
    path: np.ndarray
    metrics: dict[str, float]
    stats: dict[str, Any]


@dataclass
class _LatticeNode:
    key: tuple[int, int, int]
    x: float
    y: float
    theta: float
    cost: float
    parent: tuple[int, int, int] | None
    edge_path: np.ndarray | None

    @property
    def state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta], dtype=float)


@dataclass
class _GridGuidance:
    status: str
    resolution: float
    nx: int
    ny: int
    values: np.ndarray | None
    goal_cell: tuple[int, int] | None


class DeterministicLatticePlanner:
    """A deterministic A* search over Dubins-like motion primitives.

    The graph is implicit: each expanded SE(2) lattice state rolls out a fixed
    ordered set of constant-curvature primitives. The returned path is still a
    continuous primitive sequence, and all labels are computed by the shared
    physical path evaluator.
    """

    def __init__(self, scenario: Scenario, config: LatticePlannerConfig | None = None) -> None:
        self.scenario = scenario
        self.config = config or LatticePlannerConfig()
        self.omega_candidates = self._ordered_omega_candidates()
        self.primitive_dts = self._primitive_dt_values()
        self.last_runtime_sec = 0.0
        self.last_failure_reason = ""
        self.last_expansions = 0
        self.last_discovered = 0
        self.last_guidance_status = ""
        self.last_search_stats: dict[str, Any] = {}

    def plan(
        self,
        start: tuple[float, float, float | None],
        goal: tuple[float, float],
    ) -> LatticePlanningResult | None:
        tic = time.perf_counter()
        if not self.scenario.is_state_valid(start[0], start[1], margin=self.config.obstacle_margin):
            self._record_failure(tic, "invalid_start", expansions=0, discovered=0)
            return None
        if not self.scenario.is_state_valid(goal[0], goal[1], margin=self.config.obstacle_margin):
            self._record_failure(tic, "invalid_goal", expansions=0, discovered=0)
            return None

        guidance = self._build_guidance_if_needed(goal)
        self.last_guidance_status = guidance.status
        start_states = self._initial_states(start)
        start_goal_distance = euclidean_distance(start[0], start[1], goal[0], goal[1])
        start_goal_risk = self._straight_line_risk(np.array([start[0], start[1], 0.0], dtype=float), goal)
        nodes: dict[tuple[int, int, int], _LatticeNode] = {}
        best_cost: dict[tuple[int, int, int], float] = {}
        closed: set[tuple[int, int, int]] = set()
        open_heap: list[tuple[float, float, int, tuple[int, int, int]]] = []
        counter = 0
        for start_state in start_states:
            start_key = self._state_key(start_state)
            if start_key in nodes:
                continue
            nodes[start_key] = _LatticeNode(
                key=start_key,
                x=float(start_state[0]),
                y=float(start_state[1]),
                theta=float(start_state[2]),
                cost=0.0,
                parent=None,
                edge_path=None,
            )
            best_cost[start_key] = 0.0
            start_priority, start_tie_breaker = self._priority(
                start_state,
                goal,
                g_cost=0.0,
                guidance=guidance,
                start_goal_distance=start_goal_distance,
                start_goal_risk=start_goal_risk,
            )
            heapq.heappush(open_heap, (start_priority, start_tie_breaker, counter, start_key))
            counter += 1

        best_goal_key: tuple[int, int, int] | None = None
        best_goal_cost = float("inf")
        first_solution_expansion: int | None = None
        expansions = 0
        expansions_at_first_solution: int | None = None
        generated_successors = 0
        same_key_successors = 0
        closed_key_successors = 0
        dominated_successors = 0
        accepted_successors = 0

        while open_heap and expansions < self.config.max_expansions:
            _, _, _, key = heapq.heappop(open_heap)
            node = nodes[key]
            if key in closed:
                continue
            if (
                expansions_at_first_solution is not None
                and expansions - expansions_at_first_solution >= self.config.post_solution_expansion_limit
            ):
                break
            closed.add(key)
            expansions += 1

            if self._is_goal(node, goal):
                if first_solution_expansion is None:
                    first_solution_expansion = expansions
                    expansions_at_first_solution = expansions
                if node.cost < best_goal_cost:
                    best_goal_key = key
                    best_goal_cost = node.cost
                if not self.config.continue_after_first_solution:
                    break

            successors = self._successors(node)
            generated_successors += len(successors)
            for edge_path, edge_cost in successors:
                end = edge_path[-1]
                next_key = self._state_key(end)
                if next_key == key:
                    same_key_successors += 1
                    continue
                if next_key in closed:
                    closed_key_successors += 1
                    continue
                new_cost = node.cost + edge_cost
                if new_cost + 1e-9 >= best_cost.get(next_key, float("inf")):
                    dominated_successors += 1
                    continue
                best_cost[next_key] = new_cost
                nodes[next_key] = _LatticeNode(
                    key=next_key,
                    x=float(end[0]),
                    y=float(end[1]),
                    theta=float(end[2]),
                    cost=new_cost,
                    parent=key,
                    edge_path=edge_path,
                )
                counter += 1
                accepted_successors += 1
                priority, tie_breaker = self._priority(
                    end,
                    goal,
                    g_cost=new_cost,
                    guidance=guidance,
                    start_goal_distance=start_goal_distance,
                    start_goal_risk=start_goal_risk,
                )
                heapq.heappush(open_heap, (priority, tie_breaker, counter, next_key))

        self.last_runtime_sec = time.perf_counter() - tic
        self.last_expansions = expansions
        self.last_discovered = len(nodes)
        search_stats = {
            "expansions": expansions,
            "discovered_states": len(nodes),
            "closed_states": len(closed),
            "generated_successors": generated_successors,
            "same_key_successors": same_key_successors,
            "closed_key_successors": closed_key_successors,
            "dominated_successors": dominated_successors,
            "accepted_successors": accepted_successors,
            "same_key_successor_rate": same_key_successors / generated_successors if generated_successors else 0.0,
            "accepted_successor_rate": accepted_successors / generated_successors if generated_successors else 0.0,
            "guidance_status": guidance.status,
            "guidance_start_cost": self._grid_guidance_value(start_states[0], guidance),
            "free_start_heading": self._free_start_heading_enabled(start),
            "initial_heading_count": len(start_states),
            "priority_mode": self.config.priority_mode,
        }
        self.last_search_stats = search_stats

        if best_goal_key is None:
            if expansions == 1 and generated_successors == 0:
                reason = "start_kinodynamic_dead_end"
            else:
                reason = "search_budget_exhausted" if open_heap else "discrete_open_set_exhausted"
            self.last_failure_reason = reason
            return None

        path = self._reconstruct_path(nodes, best_goal_key)
        metrics = evaluate_path(path, self.scenario, self.config.speed)
        metrics["objective"] = path_objective(
            path,
            self.scenario,
            self.config.speed,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
        )
        stats = {
            "success": True,
            "planner": "deterministic_lattice_guided_astar",
            "runtime_sec": self.last_runtime_sec,
            "expansions": expansions,
            "discovered_states": len(nodes),
            "closed_states": len(closed),
            "generated_successors": generated_successors,
            "same_key_successors": same_key_successors,
            "closed_key_successors": closed_key_successors,
            "dominated_successors": dominated_successors,
            "accepted_successors": accepted_successors,
            "same_key_successor_rate": search_stats["same_key_successor_rate"],
            "accepted_successor_rate": search_stats["accepted_successor_rate"],
            "first_solution_expansion": first_solution_expansion,
            "post_solution_expansions": (
                expansions - first_solution_expansion if first_solution_expansion is not None else 0
            ),
            "guidance_status": guidance.status,
            "guidance_start_cost": self._grid_guidance_value(start_states[0], guidance),
            "free_start_heading": self._free_start_heading_enabled(start),
            "initial_heading_count": len(start_states),
            "priority_mode": self.config.priority_mode,
            "goal_key": best_goal_key,
            "tree_cost": best_goal_cost,
            "config": asdict(self.config),
        }
        self.last_failure_reason = ""
        return LatticePlanningResult(path=path, metrics=metrics, stats=stats)

    def _initial_states(self, start: tuple[float, float, float | None]) -> list[np.ndarray]:
        if not self._free_start_heading_enabled(start):
            return [np.array([start[0], start[1], wrap_angle(float(start[2]))], dtype=float)]
        return [
            np.array([start[0], start[1], theta], dtype=float)
            for theta in self._heading_bin_centers()
        ]

    def _free_start_heading_enabled(self, start: tuple[float, float, float | None]) -> bool:
        theta = start[2]
        if self.config.free_start_heading or theta is None:
            return True
        try:
            return math.isnan(float(theta))
        except (TypeError, ValueError):
            return False

    def _heading_bin_centers(self) -> list[float]:
        return [
            -math.pi + (2.0 * math.pi * idx / self.config.heading_bins)
            for idx in range(self.config.heading_bins)
        ]

    def _successors(self, node: _LatticeNode) -> list[tuple[np.ndarray, float]]:
        successors = []
        for dt in self.primitive_dts:
            for omega in self.omega_candidates:
                edge_path = rollout_arc(
                    node.state,
                    omega=float(omega),
                    speed=self.config.speed,
                    dt=float(dt),
                    num_samples=self.config.primitive_samples,
                )
                if not self._edge_valid(edge_path):
                    continue
                edge_cost = self._edge_cost(edge_path, float(omega), float(dt))
                successors.append((edge_path, edge_cost))
        return successors

    def _edge_valid(self, edge_path: np.ndarray) -> bool:
        return all(
            self.scenario.is_state_valid(float(x), float(y), margin=self.config.obstacle_margin)
            for x, y, _ in edge_path
        )

    def _edge_cost(self, edge_path: np.ndarray, omega: float, dt: float) -> float:
        length = path_length(edge_path)
        risk = path_risk(edge_path, self.scenario, self.config.speed)
        turn = abs(omega) * dt
        return self.config.alpha * length + self.config.beta * risk + self.config.gamma * turn

    def _priority(
        self,
        state: np.ndarray,
        goal: tuple[float, float],
        g_cost: float,
        guidance: _GridGuidance,
        start_goal_distance: float,
        start_goal_risk: float,
    ) -> tuple[float, float]:
        h_euclid = self._euclidean_lower_bound(state, goal)
        lower_bound = g_cost + h_euclid
        if self.config.priority_mode == "euclidean":
            return lower_bound, h_euclid
        if self.config.priority_mode == "straight":
            straight_value = self._straight_line_value(state, goal)
            return g_cost + straight_value, straight_value
        if self.config.priority_mode == "scaled_straight":
            straight_value = self._straight_line_value(state, goal, risk_scale=0.5)
            return g_cost + straight_value, straight_value
        if self.config.priority_mode == "clipped_straight":
            straight_value = self._clipped_straight_line_value(state, goal, start_goal_distance)
            return g_cost + straight_value, straight_value
        if self.config.priority_mode == "capped_straight":
            straight_value = self._capped_straight_line_value(state, goal, start_goal_risk)
            return g_cost + straight_value, straight_value
        grid_value = self._grid_guidance_value(state, guidance)
        if self.config.priority_mode == "dijkstra":
            if grid_value is None:
                return lower_bound, h_euclid
            return g_cost + grid_value, grid_value
        if self.config.priority_mode != "guided":
            raise ValueError(f"Unsupported lattice priority_mode: {self.config.priority_mode}")
        if grid_value is None:
            return lower_bound, lower_bound
        priority = lower_bound + self.config.grid_guidance_weight * grid_value
        return priority, lower_bound

    def _euclidean_lower_bound(self, state: np.ndarray, goal: tuple[float, float]) -> float:
        distance = euclidean_distance(float(state[0]), float(state[1]), goal[0], goal[1])
        return self.config.alpha * max(0.0, distance - self.config.goal_tolerance)

    def _straight_line_value(
        self,
        state: np.ndarray,
        goal: tuple[float, float],
        risk_scale: float = 1.0,
    ) -> float:
        distance = euclidean_distance(float(state[0]), float(state[1]), goal[0], goal[1])
        if distance <= self.config.goal_tolerance:
            return 0.0
        n = max(2, int(self.config.straight_line_heuristic_samples))
        xs = np.linspace(float(state[0]), float(goal[0]), n)
        ys = np.linspace(float(state[1]), float(goal[1]), n)
        theta = math.atan2(float(goal[1]) - float(state[1]), float(goal[0]) - float(state[0]))
        path = np.column_stack([xs, ys, np.full(n, theta, dtype=float)])
        risk = self._path_risk_for_straight_line(path)
        return self.config.alpha * distance + float(risk_scale) * self.config.beta * risk

    def _clipped_straight_line_value(
        self,
        state: np.ndarray,
        goal: tuple[float, float],
        start_goal_distance: float,
    ) -> float:
        distance = euclidean_distance(float(state[0]), float(state[1]), goal[0], goal[1])
        if distance <= self.config.goal_tolerance:
            return 0.0
        n = max(2, int(self.config.straight_line_heuristic_samples))
        xs = np.linspace(float(state[0]), float(goal[0]), n)
        ys = np.linspace(float(state[1]), float(goal[1]), n)
        theta = math.atan2(float(goal[1]) - float(state[1]), float(goal[0]) - float(state[0]))
        path = np.column_stack([xs, ys, np.full(n, theta, dtype=float)])
        risk = self._path_risk_for_straight_line(path)
        eps = 1e-9
        rho = distance / max(float(start_goal_distance), eps)
        risk_weight_scale = min(1.0, float(self.config.clipped_straight_risk_rho0) / max(rho, eps))
        return self.config.alpha * distance + risk_weight_scale * self.config.beta * risk

    def _capped_straight_line_value(
        self,
        state: np.ndarray,
        goal: tuple[float, float],
        start_goal_risk: float,
    ) -> float:
        distance = euclidean_distance(float(state[0]), float(state[1]), goal[0], goal[1])
        if distance <= self.config.goal_tolerance:
            return 0.0
        risk = self._straight_line_risk(state, goal)
        capped_risk = min(risk, float(self.config.capped_straight_risk_rho) * float(start_goal_risk))
        return self.config.alpha * distance + self.config.beta * capped_risk

    def _straight_line_risk(self, state: np.ndarray, goal: tuple[float, float]) -> float:
        n = max(2, int(self.config.straight_line_heuristic_samples))
        xs = np.linspace(float(state[0]), float(goal[0]), n)
        ys = np.linspace(float(state[1]), float(goal[1]), n)
        theta = math.atan2(float(goal[1]) - float(state[1]), float(goal[0]) - float(state[0]))
        path = np.column_stack([xs, ys, np.full(n, theta, dtype=float)])
        return self._path_risk_for_straight_line(path)

    def _path_risk_for_straight_line(self, path: np.ndarray) -> float:
        return path_risk(path, self.scenario, self.config.speed)

    def _build_guidance_if_needed(self, goal: tuple[float, float]) -> _GridGuidance:
        if self.config.priority_mode not in {"guided", "dijkstra"}:
            return _GridGuidance(
                status="not_needed",
                resolution=self.config.grid_guidance_resolution,
                nx=0,
                ny=0,
                values=None,
                goal_cell=None,
            )
        return self._build_grid_guidance(goal)

    def _build_grid_guidance(self, goal: tuple[float, float]) -> _GridGuidance:
        if not self.config.use_grid_guidance:
            return _GridGuidance(
                status="disabled",
                resolution=self.config.grid_guidance_resolution,
                nx=0,
                ny=0,
                values=None,
                goal_cell=None,
            )

        resolution = self.config.grid_guidance_resolution
        nx = int(math.floor(self.scenario.width / resolution)) + 1
        ny = int(math.floor(self.scenario.height / resolution)) + 1
        goal_cell = self._point_to_guidance_cell(goal[0], goal[1], nx, ny, resolution)
        if not self._guidance_cell_valid(goal_cell, resolution):
            return _GridGuidance(
                status="invalid_goal_cell",
                resolution=resolution,
                nx=nx,
                ny=ny,
                values=None,
                goal_cell=goal_cell,
            )

        values = np.full((ny, nx), np.inf, dtype=float)
        values[goal_cell[1], goal_cell[0]] = 0.0
        heap: list[tuple[float, tuple[int, int]]] = [(0.0, goal_cell)]
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        visited = 0
        while heap:
            cost, current = heapq.heappop(heap)
            if cost > values[current[1], current[0]] + 1e-9:
                continue
            visited += 1
            cx, cy = self._guidance_cell_center(current, resolution)
            for dx, dy in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny):
                    continue
                if not self._guidance_cell_valid(nxt, resolution):
                    continue
                if not self._guidance_edge_valid(current, nxt, resolution):
                    continue
                nxp, nyp = self._guidance_cell_center(nxt, resolution)
                step = math.hypot(dx, dy) * resolution
                mid_x = 0.5 * (cx + nxp)
                mid_y = 0.5 * (cy + nyp)
                risk = self.scenario.risk_at(mid_x, mid_y)
                edge_cost = self.config.alpha * step + self.config.grid_guidance_risk_weight * risk * step / self.config.speed
                new_cost = cost + edge_cost
                if new_cost + 1e-9 >= values[nxt[1], nxt[0]]:
                    continue
                values[nxt[1], nxt[0]] = new_cost
                heapq.heappush(heap, (new_cost, nxt))

        status = "grid_guidance_ready" if visited > 1 else "grid_guidance_goal_isolated"
        return _GridGuidance(status=status, resolution=resolution, nx=nx, ny=ny, values=values, goal_cell=goal_cell)

    def _grid_guidance_value(self, state: np.ndarray, guidance: _GridGuidance) -> float | None:
        if guidance.values is None:
            return None
        cell = self._point_to_guidance_cell(state[0], state[1], guidance.nx, guidance.ny, guidance.resolution)
        value = float(guidance.values[cell[1], cell[0]])
        if not math.isfinite(value):
            return None
        return value

    def _is_goal(self, node: _LatticeNode, goal: tuple[float, float]) -> bool:
        return euclidean_distance(node.x, node.y, goal[0], goal[1]) <= self.config.goal_tolerance

    def _state_key(self, state: np.ndarray) -> tuple[int, int, int]:
        ix = int(round(float(state[0]) / self.config.xy_resolution))
        iy = int(round(float(state[1]) / self.config.xy_resolution))
        theta = wrap_angle(float(state[2]))
        normalized = (theta + math.pi) / (2.0 * math.pi)
        ih = int(math.floor(normalized * self.config.heading_bins)) % self.config.heading_bins
        return ix, iy, ih

    def _ordered_omega_candidates(self) -> np.ndarray:
        candidates = np.linspace(-self.config.omega_max, self.config.omega_max, self.config.n_actions)
        return np.asarray(sorted(candidates, key=lambda omega: (abs(float(omega)), float(omega))), dtype=float)

    def _primitive_dt_values(self) -> tuple[float, ...]:
        values = tuple(float(value) for value in self.config.primitive_dt_values)
        return tuple(sorted(set(values)))

    def _point_to_guidance_cell(self, x: float, y: float, nx: int, ny: int, resolution: float) -> tuple[int, int]:
        ix = int(np.clip(round(float(x) / resolution), 0, nx - 1))
        iy = int(np.clip(round(float(y) / resolution), 0, ny - 1))
        return ix, iy

    def _guidance_cell_center(self, cell: tuple[int, int], resolution: float) -> tuple[float, float]:
        return (
            float(np.clip(cell[0] * resolution, 0.0, self.scenario.width)),
            float(np.clip(cell[1] * resolution, 0.0, self.scenario.height)),
        )

    def _guidance_cell_valid(self, cell: tuple[int, int], resolution: float) -> bool:
        x, y = self._guidance_cell_center(cell, resolution)
        return self.scenario.is_state_valid(x, y, margin=self.config.obstacle_margin)

    def _guidance_edge_valid(
        self,
        cell_a: tuple[int, int],
        cell_b: tuple[int, int],
        resolution: float,
    ) -> bool:
        ax, ay = self._guidance_cell_center(cell_a, resolution)
        bx, by = self._guidance_cell_center(cell_b, resolution)
        n = max(2, int(self.config.grid_guidance_edge_samples))
        for t in np.linspace(0.0, 1.0, n):
            x = (1.0 - t) * ax + t * bx
            y = (1.0 - t) * ay + t * by
            if not self.scenario.is_state_valid(float(x), float(y), margin=self.config.obstacle_margin):
                return False
        return True

    def _record_failure(self, tic: float, reason: str, expansions: int, discovered: int) -> None:
        self.last_runtime_sec = time.perf_counter() - tic
        self.last_failure_reason = reason
        self.last_guidance_status = reason
        self.last_expansions = expansions
        self.last_discovered = discovered
        self.last_search_stats = {
            "expansions": expansions,
            "discovered_states": discovered,
            "guidance_status": reason,
        }

    @staticmethod
    def _reconstruct_path(nodes: dict[tuple[int, int, int], _LatticeNode], goal_key: tuple[int, int, int]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        key: tuple[int, int, int] | None = goal_key
        while key is not None:
            node = nodes[key]
            if node.edge_path is not None:
                chunks.append(node.edge_path)
            else:
                chunks.append(node.state.reshape(1, 3))
            key = node.parent

        chunks.reverse()
        path_parts = [chunks[0]]
        for chunk in chunks[1:]:
            path_parts.append(chunk[1:])
        return np.vstack(path_parts)
