from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.envs.obstacle_map import CircleObstacle, ObstacleMap, RectangleObstacle
from src.envs.scenario import Scenario
from src.envs.threat_field import GaussianThreat, ThreatField
from src.experiment_config import CURRENT_MIN_START_GOAL_DISTANCE, CURRENT_PLANNER
from src.planners.lattice_planner import DeterministicLatticePlanner, LatticePlannerConfig
from src.planners.path_utils import path_length, path_risk, survival_prob


@dataclass
class MapConfig:
    width: float = 1000.0
    height: float = 1000.0
    resolution: float = 2.0


@dataclass
class ThreatConfig:
    layout: str = "random"
    n_sources_range: tuple[int, int] = (4, 8)
    amplitude_range: tuple[float, float] = (0.005, 0.055)
    sigma_range: tuple[float, float] = (55.0, 150.0)
    barrier_sources_range: tuple[int, int] = (5, 8)
    barrier_amplitude_range: tuple[float, float] = (0.035, 0.085)
    barrier_sigma_range: tuple[float, float] = (75.0, 135.0)


@dataclass
class ObstacleConfig:
    n_circles_range: tuple[int, int] = (1, 4)
    n_rectangles_range: tuple[int, int] = (1, 4)
    circle_radius_range: tuple[float, float] = (45.0, 90.0)
    rectangle_size_range: tuple[float, float] = (90.0, 170.0)


@dataclass
class PlannerConfig:
    speed: float = CURRENT_PLANNER["speed"]
    omega_max: float = CURRENT_PLANNER["omega_max"]
    primitive_dt_values: tuple[float, ...] = CURRENT_PLANNER["primitive_dt_values"]
    primitive_samples: int = CURRENT_PLANNER["primitive_samples"]
    n_actions: int = CURRENT_PLANNER["n_actions"]
    max_iters: int = CURRENT_PLANNER["max_iters"]
    goal_tolerance: float = CURRENT_PLANNER["goal_tolerance"]
    alpha: float = CURRENT_PLANNER["alpha"]
    beta: float = CURRENT_PLANNER["beta"]
    gamma: float = CURRENT_PLANNER["gamma"]
    lattice_xy_resolution: float = CURRENT_PLANNER["lattice_xy_resolution"]
    lattice_heading_bins: int = CURRENT_PLANNER["lattice_heading_bins"]
    lattice_post_solution_expansion_limit: int = CURRENT_PLANNER["lattice_post_solution_expansion_limit"]
    lattice_grid_guidance_weight: float = CURRENT_PLANNER["lattice_grid_guidance_weight"]
    free_start_heading: bool = CURRENT_PLANNER["free_start_heading"]
    lattice_priority_mode: str = CURRENT_PLANNER["lattice_priority_mode"]

    def __post_init__(self) -> None:
        self.primitive_dt_values = tuple(float(value) for value in self.primitive_dt_values)
        if not self.primitive_dt_values:
            raise ValueError("At least one lattice primitive dt must be provided.")
        if any(value <= 0.0 for value in self.primitive_dt_values):
            raise ValueError("Primitive dt values must be positive.")


@dataclass
class DatasetConfig:
    n_samples: int = 20
    n_scenarios: int = 5
    start_goal_mode: str = "uniform"
    mixed_barrier_crossing_prob: float = 0.5
    min_start_goal_distance: float = CURRENT_MIN_START_GOAL_DISTANCE
    max_sample_tries: int = 500
    straight_line_samples: int = 96
    map_seed: int = 0
    sample_seed: int = 1
    map: MapConfig = field(default_factory=MapConfig)
    threat: ThreatConfig = field(default_factory=ThreatConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)


CSV_COLUMNS = [
    "sample_id",
    "scenario_id",
    "start_goal_mode",
    "seed",
    "start_x",
    "start_y",
    "start_theta",
    "goal_x",
    "goal_y",
    "euclidean_distance",
    "start_risk",
    "goal_risk",
    "straight_line_risk",
    "straight_line_mean_threat",
    "straight_line_max_threat",
    "straight_line_p90_threat",
    "straight_line_survival_prob",
    "straight_line_collision",
    "midpoint_risk",
    "corridor_min_risk",
    "corridor_min_collision_free_risk",
    "corridor_collision_free_count",
    "feasible",
    "length",
    "time",
    "risk",
    "survival_prob",
    "turning",
    "objective",
    "runtime_sec",
    "planner_iterations",
    "planner_nodes",
]


def generate_dataset(
    output_dir: Path,
    config: DatasetConfig,
    save_threat_grids: bool = False,
    save_paths: bool = False,
    progress_every: int = 1,
    checkpoint_every: int = 0,
    resume: bool = False,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    map_rng = np.random.default_rng(config.map_seed)

    scenarios = [
        random_scenario(
            scenario_id=i,
            rng=map_rng,
            map_config=config.map,
            threat_config=config.threat,
            obstacle_config=config.obstacles,
        )
        for i in range(config.n_scenarios)
    ]
    if checkpoint_every > 0:
        write_metadata(output_dir / "metadata.json", config, scenarios)
        if save_threat_grids:
            save_scenario_grids(output_dir / "scenario_grids.npz", scenarios)

    existing_records = load_existing_records(output_dir / "samples.csv") if resume else {}
    if existing_records:
        print(f"resuming dataset generation from {len(existing_records)} existing samples", flush=True)

    records: list[dict[str, Any]] = []
    path_arrays: dict[str, np.ndarray] = {}
    for sample_id in range(config.n_samples):
        scenario_id = sample_id % config.n_scenarios
        scenario = scenarios[scenario_id]
        sample_seed = int(
            np.random.SeedSequence(int(config.sample_seed), spawn_key=(sample_id,)).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        if sample_id in existing_records:
            records.append(existing_records[sample_id])
            continue

        sample_rng = np.random.default_rng(sample_seed)
        start, goal, actual_start_goal_mode = sample_start_goal_with_mode(
            scenario,
            rng=sample_rng,
            mode=config.start_goal_mode,
            mixed_barrier_crossing_prob=config.mixed_barrier_crossing_prob,
            min_distance=config.min_start_goal_distance,
            max_tries=config.max_sample_tries,
        )

        record = make_base_record(
            sample_id=sample_id,
            scenario_id=scenario_id,
            seed=sample_seed,
            start_goal_mode=actual_start_goal_mode,
            start=start,
            goal=goal,
            scenario=scenario,
            config=config,
        )
        result, runtime_sec = run_planner(scenario, start, goal, config.planner)
        record["runtime_sec"] = runtime_sec

        if result is None:
            record.update(_failed_label(config.planner))
        else:
            if save_paths:
                path_arrays[f"path_{sample_id}"] = result.path.astype(np.float32)
            record.update(
                {
                    "feasible": 1,
                    "length": result.metrics["length"],
                    "time": result.metrics["time"],
                    "risk": result.metrics["risk"],
                    "survival_prob": result.metrics["survival_prob"],
                    "turning": result.metrics["turning"],
                    "objective": result.metrics["objective"],
                    "planner_iterations": planner_iterations(result.stats),
                    "planner_nodes": planner_nodes(result.stats),
                }
            )
        records.append(record)
        if progress_every > 0 and ((sample_id + 1) % progress_every == 0 or sample_id + 1 == config.n_samples):
            feasible_count = sum(int(item["feasible"]) for item in records)
            print(
                f"[{sample_id + 1:04d}/{config.n_samples:04d}] "
                f"feasible={feasible_count}/{len(records)} "
                f"last_scenario={scenario_id} last_L={_fmt(record['length'])} "
                f"last_R={_fmt(record['risk'])} last_runtime={runtime_sec:.2f}s",
                flush=True,
            )
        if checkpoint_every > 0 and (
            (sample_id + 1) % checkpoint_every == 0 or sample_id + 1 == config.n_samples
        ):
            write_records_csv(output_dir / "samples.csv", records)

    write_records_csv(output_dir / "samples.csv", records)
    write_metadata(output_dir / "metadata.json", config, scenarios)
    if save_threat_grids:
        save_scenario_grids(output_dir / "scenario_grids.npz", scenarios)
    if save_paths:
        save_sample_paths(output_dir / "paths.npz", path_arrays)
    return records


def load_existing_records(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    existing: dict[int, dict[str, Any]] = {}
    for expected_id, record in enumerate(records):
        sample_id = int(record["sample_id"])
        if sample_id != expected_id:
            raise ValueError(f"Cannot resume non-contiguous samples.csv: expected sample_id {expected_id}, got {sample_id}.")
        existing[sample_id] = record
    return existing


def random_scenario(
    scenario_id: int,
    rng: np.random.Generator,
    map_config: MapConfig,
    threat_config: ThreatConfig,
    obstacle_config: ObstacleConfig,
) -> Scenario:
    n_sources = int(rng.integers(threat_config.n_sources_range[0], threat_config.n_sources_range[1] + 1))
    barrier_orientation = None
    barrier_gap = None
    if threat_config.layout == "central_barrier":
        threat, barrier_orientation, barrier_gap = central_barrier_threat_field(
            map_config=map_config,
            threat_config=threat_config,
            rng=rng,
        )
    else:
        threat = ThreatField.random_gaussian(
            map_config.width,
            map_config.height,
            map_config.resolution,
            n_sources=n_sources,
            amplitude_range=threat_config.amplitude_range,
            sigma_range=threat_config.sigma_range,
            seed=int(rng.integers(0, 2**31 - 1)),
        )

    n_circles = int(rng.integers(obstacle_config.n_circles_range[0], obstacle_config.n_circles_range[1] + 1))
    n_rectangles = int(rng.integers(obstacle_config.n_rectangles_range[0], obstacle_config.n_rectangles_range[1] + 1))
    circles = [
        CircleObstacle(
            x=float(rng.uniform(120.0, map_config.width - 120.0)),
            y=float(rng.uniform(120.0, map_config.height - 120.0)),
            radius=float(rng.uniform(*obstacle_config.circle_radius_range)),
        )
        for _ in range(n_circles)
    ]
    rectangles = []
    for _ in range(n_rectangles):
        w = float(rng.uniform(*obstacle_config.rectangle_size_range))
        h = float(rng.uniform(*obstacle_config.rectangle_size_range))
        x_min = float(rng.uniform(60.0, map_config.width - w - 60.0))
        y_min = float(rng.uniform(60.0, map_config.height - h - 60.0))
        rectangles.append(RectangleObstacle(x_min=x_min, y_min=y_min, x_max=x_min + w, y_max=y_min + h))

    scenario = Scenario(map_config.width, map_config.height, threat, ObstacleMap(circles=circles, rectangles=rectangles))
    scenario.scenario_id = scenario_id  # Convenient for debugging without changing the dataclass API.
    scenario.threat_layout = threat_config.layout
    scenario.barrier_orientation = barrier_orientation
    scenario.barrier_gap = barrier_gap if threat_config.layout == "central_barrier" else None
    return scenario


def central_barrier_threat_field(
    map_config: MapConfig,
    threat_config: ThreatConfig,
    rng: np.random.Generator,
) -> tuple[ThreatField, str, float]:
    """Build a high-risk band with one softer gap for detour-friendly labels."""
    orientation = str(rng.choice(["vertical", "horizontal"]))
    n_barrier = int(rng.integers(threat_config.barrier_sources_range[0], threat_config.barrier_sources_range[1] + 1))
    sources: list[GaussianThreat] = []
    gap_index = int(rng.integers(1, max(2, n_barrier - 1)))

    if orientation == "vertical":
        x_center = float(rng.uniform(0.42 * map_config.width, 0.58 * map_config.width))
        ys = np.linspace(0.18 * map_config.height, 0.82 * map_config.height, n_barrier)
        for idx, y in enumerate(ys):
            if abs(idx - gap_index) <= 0:
                continue
            sources.append(
                GaussianThreat(
                    x=float(x_center + rng.normal(0.0, 25.0)),
                    y=float(y + rng.normal(0.0, 35.0)),
                    amplitude=float(rng.uniform(*threat_config.barrier_amplitude_range)),
                    sigma=float(rng.uniform(*threat_config.barrier_sigma_range)),
                )
            )
        gap_position = float(ys[gap_index])
    else:
        y_center = float(rng.uniform(0.42 * map_config.height, 0.58 * map_config.height))
        xs = np.linspace(0.18 * map_config.width, 0.82 * map_config.width, n_barrier)
        for idx, x in enumerate(xs):
            if abs(idx - gap_index) <= 0:
                continue
            sources.append(
                GaussianThreat(
                    x=float(x + rng.normal(0.0, 35.0)),
                    y=float(y_center + rng.normal(0.0, 25.0)),
                    amplitude=float(rng.uniform(*threat_config.barrier_amplitude_range)),
                    sigma=float(rng.uniform(*threat_config.barrier_sigma_range)),
                )
            )
        gap_position = float(xs[gap_index])

    n_background = int(rng.integers(threat_config.n_sources_range[0], threat_config.n_sources_range[1] + 1))
    for _ in range(n_background):
        sources.append(
            GaussianThreat(
                x=float(rng.uniform(0.1 * map_config.width, 0.9 * map_config.width)),
                y=float(rng.uniform(0.1 * map_config.height, 0.9 * map_config.height)),
                amplitude=float(rng.uniform(*threat_config.amplitude_range)),
                sigma=float(rng.uniform(*threat_config.sigma_range)),
            )
        )
    return ThreatField(map_config.width, map_config.height, map_config.resolution, sources=sources), orientation, gap_position


def sample_start_goal(
    scenario: Scenario,
    rng: np.random.Generator,
    mode: str,
    min_distance: float,
    max_tries: int,
) -> tuple[tuple[float, float, float], tuple[float, float]]:
    start, goal, _ = sample_start_goal_with_mode(
        scenario,
        rng=rng,
        mode=mode,
        mixed_barrier_crossing_prob=0.5,
        min_distance=min_distance,
        max_tries=max_tries,
    )
    return start, goal


def sample_start_goal_with_mode(
    scenario: Scenario,
    rng: np.random.Generator,
    mode: str,
    mixed_barrier_crossing_prob: float,
    min_distance: float,
    max_tries: int,
) -> tuple[tuple[float, float, float], tuple[float, float], str]:
    if mode not in {"uniform", "barrier_crossing", "mixed"}:
        raise ValueError(f"Unsupported start-goal mode: {mode}")
    for _ in range(max_tries):
        actual_mode = mode
        if mode == "mixed":
            actual_mode = "barrier_crossing" if rng.random() < mixed_barrier_crossing_prob else "uniform"

        if actual_mode == "barrier_crossing":
            sx, sy, gx, gy = sample_barrier_crossing_pair(scenario, rng)
        else:
            sx, sy = sample_uniform_valid_point(scenario, rng)
            gx, gy = sample_uniform_valid_point(scenario, rng)
        if math.hypot(gx - sx, gy - sy) < min_distance:
            continue
        return (sx, sy, 0.0), (gx, gy), actual_mode
    raise RuntimeError("Failed to sample a valid start-goal pair; reduce obstacles or min distance.")


def sample_uniform_valid_point(
    scenario: Scenario,
    rng: np.random.Generator,
    boundary_margin: float = 0.0,
    max_tries: int = 500,
) -> tuple[float, float]:
    for _ in range(max_tries):
        x = float(rng.uniform(boundary_margin, scenario.width - boundary_margin))
        y = float(rng.uniform(boundary_margin, scenario.height - boundary_margin))
        if scenario.is_state_valid(x, y):
            return x, y
    raise RuntimeError("Failed to sample a valid point.")


def sample_barrier_crossing_pair(scenario: Scenario, rng: np.random.Generator) -> tuple[float, float, float, float]:
    orientation = getattr(scenario, "barrier_orientation", None)
    if orientation not in {"vertical", "horizontal"}:
        orientation = str(rng.choice(["vertical", "horizontal"]))
    flip = bool(rng.integers(0, 2))
    if orientation == "vertical":
        left_x = float(rng.uniform(30.0, 0.30 * scenario.width))
        right_x = float(rng.uniform(0.70 * scenario.width, scenario.width - 30.0))
        sx, gx = (right_x, left_x) if flip else (left_x, right_x)
        sy = float(rng.uniform(0.12 * scenario.height, 0.88 * scenario.height))
        gy = float(np.clip(sy + rng.normal(0.0, 0.22 * scenario.height), 30.0, scenario.height - 30.0))
    else:
        bottom_y = float(rng.uniform(30.0, 0.30 * scenario.height))
        top_y = float(rng.uniform(0.70 * scenario.height, scenario.height - 30.0))
        sy, gy = (top_y, bottom_y) if flip else (bottom_y, top_y)
        sx = float(rng.uniform(0.12 * scenario.width, 0.88 * scenario.width))
        gx = float(np.clip(sx + rng.normal(0.0, 0.22 * scenario.width), 30.0, scenario.width - 30.0))
    return sx, sy, gx, gy


def make_base_record(
    sample_id: int,
    scenario_id: int,
    seed: int,
    start_goal_mode: str,
    start: tuple[float, float, float],
    goal: tuple[float, float],
    scenario: Scenario,
    config: DatasetConfig,
) -> dict[str, Any]:
    line_path = straight_line_path(start, goal, config.straight_line_samples)
    line_stats = path_threat_stats(line_path, scenario, config.planner.speed)
    corridor_stats = offset_corridor_stats(start, goal, scenario, config)
    midpoint_x = 0.5 * (start[0] + goal[0])
    midpoint_y = 0.5 * (start[1] + goal[1])
    return {
        "sample_id": sample_id,
        "scenario_id": scenario_id,
        "start_goal_mode": start_goal_mode,
        "seed": seed,
        "start_x": start[0],
        "start_y": start[1],
        "start_theta": start[2],
        "goal_x": goal[0],
        "goal_y": goal[1],
        "euclidean_distance": math.hypot(goal[0] - start[0], goal[1] - start[1]),
        "start_risk": scenario.risk_at(start[0], start[1]),
        "goal_risk": scenario.risk_at(goal[0], goal[1]),
        "straight_line_risk": line_stats["risk"],
        "straight_line_mean_threat": line_stats["mean_threat"],
        "straight_line_max_threat": line_stats["max_threat"],
        "straight_line_p90_threat": line_stats["p90_threat"],
        "straight_line_survival_prob": survival_prob(line_stats["risk"]),
        "straight_line_collision": int(scenario.obstacle_map.path_collision(line_path)),
        "midpoint_risk": scenario.risk_at(midpoint_x, midpoint_y),
        "corridor_min_risk": corridor_stats["min_risk"],
        "corridor_min_collision_free_risk": corridor_stats["min_collision_free_risk"],
        "corridor_collision_free_count": corridor_stats["collision_free_count"],
    }


def straight_line_path(
    start: tuple[float, float, float],
    goal: tuple[float, float],
    n_samples: int,
) -> np.ndarray:
    xs = np.linspace(start[0], goal[0], n_samples)
    ys = np.linspace(start[1], goal[1], n_samples)
    theta = math.atan2(goal[1] - start[1], goal[0] - start[0])
    return np.column_stack((xs, ys, np.full(n_samples, theta)))


def offset_line_path(
    start: tuple[float, float, float],
    goal: tuple[float, float],
    n_samples: int,
    offset: float,
) -> np.ndarray:
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return straight_line_path(start, goal, n_samples)
    nx = -dy / norm
    ny = dx / norm
    shifted_start = (start[0] + offset * nx, start[1] + offset * ny, start[2])
    shifted_goal = (goal[0] + offset * nx, goal[1] + offset * ny)
    return straight_line_path(shifted_start, shifted_goal, n_samples)


def path_threat_stats(path: np.ndarray, scenario: Scenario, speed: float) -> dict[str, float]:
    rates = scenario.risk_values(path[:, 0], path[:, 1])
    return {
        "risk": path_risk(path, scenario, speed),
        "mean_threat": float(np.mean(rates)),
        "max_threat": float(np.max(rates)),
        "p90_threat": float(np.quantile(rates, 0.9)),
    }


def offset_corridor_stats(
    start: tuple[float, float, float],
    goal: tuple[float, float],
    scenario: Scenario,
    config: DatasetConfig,
) -> dict[str, float]:
    offsets = [-120.0, -80.0, -40.0, 0.0, 40.0, 80.0, 120.0]
    risks = []
    collision_free_risks = []
    for offset in offsets:
        path = offset_line_path(start, goal, config.straight_line_samples, offset)
        in_bounds = all(scenario.in_bounds(float(x), float(y)) for x, y in path[:, :2])
        collision = (not in_bounds) or scenario.obstacle_map.path_collision(path)
        risk = path_risk(path, scenario, config.planner.speed)
        risks.append(risk)
        if not collision:
            collision_free_risks.append(risk)
    return {
        "min_risk": float(np.min(risks)),
        "min_collision_free_risk": float(np.min(collision_free_risks)) if collision_free_risks else float(np.min(risks)),
        "collision_free_count": int(len(collision_free_risks)),
    }


def run_planner(
    scenario: Scenario,
    start: tuple[float, float, float],
    goal: tuple[float, float],
    planner_config: PlannerConfig,
    return_diagnostics: bool = False,
):
    config = LatticePlannerConfig(
        speed=planner_config.speed,
        omega_max=planner_config.omega_max,
        primitive_dt_values=planner_config.primitive_dt_values,
        primitive_samples=planner_config.primitive_samples,
        n_actions=planner_config.n_actions,
        xy_resolution=planner_config.lattice_xy_resolution,
        heading_bins=planner_config.lattice_heading_bins,
        goal_tolerance=planner_config.goal_tolerance,
        max_expansions=planner_config.max_iters,
        post_solution_expansion_limit=planner_config.lattice_post_solution_expansion_limit,
        alpha=planner_config.alpha,
        beta=planner_config.beta,
        gamma=planner_config.gamma,
        grid_guidance_weight=planner_config.lattice_grid_guidance_weight,
        grid_guidance_risk_weight=planner_config.beta,
        free_start_heading=planner_config.free_start_heading,
        priority_mode=planner_config.lattice_priority_mode,
    )
    planner = DeterministicLatticePlanner(scenario, config)
    tic = time.perf_counter()
    result = planner.plan(start, goal)
    runtime_sec = time.perf_counter() - tic
    if return_diagnostics:
        diagnostics = {
            "failure_reason": "" if result is not None else planner.last_failure_reason,
            "guide_status": planner.last_guidance_status,
            **planner.last_search_stats,
        }
        return result, runtime_sec, diagnostics
    return result, runtime_sec


def planner_iterations(stats: dict[str, Any]) -> int | float:
    return stats.get("iterations", stats.get("expansions", -1))


def planner_nodes(stats: dict[str, Any]) -> int | float:
    return stats.get("nodes", stats.get("discovered_states", -1))


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in CSV_COLUMNS})


def write_metadata(path: Path, config: DatasetConfig, scenarios: list[Scenario]) -> None:
    metadata = {
        "config": _dataclass_to_jsonable(config),
        "scenarios": [scenario_metadata(scenario, i) for i, scenario in enumerate(scenarios)],
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def scenario_metadata(scenario: Scenario, scenario_id: int) -> dict[str, Any]:
    return {
        "scenario_id": int(scenario_id),
        "threat_layout": getattr(scenario, "threat_layout", "random"),
        "barrier_orientation": getattr(scenario, "barrier_orientation", None),
        "barrier_gap": getattr(scenario, "barrier_gap", None),
        "threat_sources": [asdict(source) for source in scenario.threat_field.sources],
        "circle_obstacles": [asdict(obstacle) for obstacle in scenario.obstacle_map.circles],
        "rectangle_obstacles": [asdict(obstacle) for obstacle in scenario.obstacle_map.rectangles],
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_scenario_grids(path: Path, scenarios: list[Scenario]) -> None:
    grids = np.stack([scenario.threat_field.grid.astype(np.float32) for scenario in scenarios], axis=0)
    np.savez_compressed(path, threat_grids=grids)


def save_sample_paths(path: Path, path_arrays: dict[str, np.ndarray]) -> None:
    if path_arrays:
        np.savez_compressed(path, **path_arrays)


def _failed_label(planner_config: PlannerConfig) -> dict[str, Any]:
    return {
        "feasible": 0,
        "length": math.nan,
        "time": math.nan,
        "risk": math.nan,
        "survival_prob": math.nan,
        "turning": math.nan,
        "objective": math.nan,
        "planner_iterations": planner_config.max_iters,
        "planner_nodes": 0,
    }


def _dataclass_to_jsonable(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {key: _dataclass_to_jsonable(value) for key, value in asdict(obj).items()}
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, list):
        return [_dataclass_to_jsonable(value) for value in obj]
    if isinstance(obj, dict):
        return {key: _dataclass_to_jsonable(value) for key, value in obj.items()}
    return obj


def _fmt(value: Any) -> str:
    try:
        if value is None or math.isnan(float(value)):
            return "nan"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
