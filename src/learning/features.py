from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import numpy as np
import pandas as pd

from src.envs.obstacle_map import CircleObstacle, ObstacleMap, RectangleObstacle
from src.envs.scenario import Scenario
from src.envs.threat_field import GaussianThreat, ThreatField
from src.learning.bid_targets import bid_cost_weights, planner_cost_weights


TARGET_COLUMNS = ["length", "time", "risk"]


def load_dataset_table(dataset_dir: Path) -> tuple[pd.DataFrame, dict]:
    samples_path = dataset_dir / "samples.csv"
    metadata_path = dataset_dir / "metadata.json"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples.csv: {samples_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {metadata_path}")
    df = pd.read_csv(samples_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return df, metadata


def build_feature_frame(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Build compact scalar features for the first bid-learning baseline."""
    width = float(metadata["config"]["map"]["width"])
    height = float(metadata["config"]["map"]["height"])
    diagonal = math.hypot(width, height)
    planner_alpha, planner_beta, _ = planner_cost_weights(metadata)

    dx = df["goal_x"] - df["start_x"]
    dy = df["goal_y"] - df["start_y"]
    distance = np.sqrt(dx**2 + dy**2)
    goal_tolerance = float(metadata["config"]["planner"]["goal_tolerance"])
    effective_distance = np.maximum(distance - goal_tolerance, 1.0)
    bearing = np.arctan2(dy, dx)
    straight_line_bid = planner_alpha * distance + planner_beta * df["straight_line_risk"]
    corridor_min_bid = planner_alpha * distance + planner_beta * df["corridor_min_risk"]
    corridor_min_collision_free_bid = planner_alpha * distance + planner_beta * df["corridor_min_collision_free_risk"]

    features = pd.DataFrame(
        {
            "start_x_norm": df["start_x"] / width,
            "start_y_norm": df["start_y"] / height,
            "goal_x_norm": df["goal_x"] / width,
            "goal_y_norm": df["goal_y"] / height,
            "dx_norm": dx / width,
            "dy_norm": dy / height,
            "euclidean_norm": distance / diagonal,
            "effective_euclidean_norm": effective_distance / diagonal,
            "euclidean_to_effective_ratio": distance / effective_distance,
            "bearing_sin": np.sin(bearing),
            "bearing_cos": np.cos(bearing),
            "start_cost_density": planner_alpha + planner_beta * df["start_risk"],
            "goal_cost_density": planner_alpha + planner_beta * df["goal_risk"],
            "straight_line_risk_cost": planner_beta * df["straight_line_risk"],
            "straight_line_planner_value": straight_line_bid,
            "straight_line_bid_density": straight_line_bid / effective_distance,
            "straight_line_survival_prob": df["straight_line_survival_prob"],
            "straight_line_collision": df["straight_line_collision"],
            "straight_line_mean_cost_density": planner_alpha + planner_beta * df["straight_line_mean_threat"],
            "straight_line_max_cost_density": planner_alpha + planner_beta * df["straight_line_max_threat"],
            "straight_line_p90_cost_density": planner_alpha + planner_beta * df["straight_line_p90_threat"],
            "midpoint_cost_density": planner_alpha + planner_beta * df["midpoint_risk"],
            "corridor_min_line_cost": corridor_min_bid,
            "corridor_min_line_density": corridor_min_bid / effective_distance,
            "corridor_min_collision_free_line_cost": corridor_min_collision_free_bid,
            "corridor_min_collision_free_line_density": corridor_min_collision_free_bid / effective_distance,
            "corridor_collision_free_count": df["corridor_collision_free_count"],
        }
    )
    return features.astype(np.float32)


def feasible_regression_data(
    df: pd.DataFrame,
    metadata: dict,
    target_columns: Iterable[str] = TARGET_COLUMNS,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], pd.DataFrame]:
    target_columns = list(target_columns)
    mask = df["feasible"].astype(int) == 1
    for column in target_columns:
        mask &= np.isfinite(df[column].astype(float))
    filtered = df.loc[mask].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No feasible samples with finite regression targets.")

    features = build_feature_frame(filtered, metadata)
    targets = filtered[target_columns].astype(np.float32)
    return (
        features.to_numpy(dtype=np.float32),
        targets.to_numpy(dtype=np.float32),
        list(features.columns),
        target_columns,
        filtered,
    )


def build_corridor_images(
    df: pd.DataFrame,
    metadata: dict,
    image_size: tuple[int, int] = (48, 48),
    lateral_width: float = 320.0,
    scale_mode: str = "fixed_width",
    scenario_cache: dict[int, Scenario] | None = None,
    base_image_cache: dict[tuple, np.ndarray] | None = None,
) -> np.ndarray:
    """Rasterize local start-goal corridor patches with cost-density and obstacle channels."""
    if scale_mode not in {"fixed_width", "square_edge"}:
        raise ValueError("scale_mode must be 'fixed_width' or 'square_edge'.")
    height_px, width_px = image_size
    bid_alpha, bid_beta, _ = bid_cost_weights(metadata)
    retain_scenarios = scenario_cache is not None and base_image_cache is None
    scenario_cache = scenario_cache if scenario_cache is not None else {}
    images = np.zeros((len(df), 2, height_px, width_px), dtype=np.float32)
    u_values = np.linspace(0.0, 1.0, width_px, dtype=np.float32)
    v_unit_values = np.linspace(-0.5, 0.5, height_px, dtype=np.float32)
    uu, vv_unit = np.meshgrid(u_values, v_unit_values)

    for idx, row in df.reset_index(drop=True).iterrows():
        scenario_id = int(row["scenario_id"])
        sx, sy = float(row["start_x"]), float(row["start_y"])
        gx, gy = float(row["goal_x"]), float(row["goal_y"])
        dx = gx - sx
        dy = gy - sy
        distance = float(np.hypot(dx, dy))
        if distance < 1e-6:
            tx, ty = 1.0, 0.0
        else:
            tx, ty = dx / distance, dy / distance
        nx, ny = -ty, tx
        row_lateral_width = distance if scale_mode == "square_edge" else float(lateral_width)
        vv = vv_unit * float(row_lateral_width)

        cache_key = (
            int(row["scenario_id"]),
            round(sx, 6),
            round(sy, 6),
            round(gx, 6),
            round(gy, 6),
            int(height_px),
            int(width_px),
            round(float(lateral_width), 6),
            str(scale_mode),
        )
        if base_image_cache is not None and cache_key in base_image_cache:
            base_image = base_image_cache[cache_key]
            threat = base_image[0]
            obstacle = base_image[1]
        else:
            if scenario_id not in scenario_cache:
                if not retain_scenarios:
                    scenario_cache.clear()
                scenario_cache[scenario_id] = scenario_from_metadata(metadata, scenario_id)
            scenario = scenario_cache[scenario_id]
            xs = sx + uu * dx + vv * nx
            ys = sy + uu * dy + vv * ny
            in_bounds = (xs >= 0.0) & (xs <= scenario.width) & (ys >= 0.0) & (ys <= scenario.height)
            threat = scenario.risk_values(xs, ys).astype(np.float32)
            obstacle = np.ones_like(threat, dtype=np.float32)
            flat_obs = obstacle.ravel()
            flat_x = xs.ravel()
            flat_y = ys.ravel()
            flat_bounds = in_bounds.ravel()
            for j, (x, y, ok) in enumerate(zip(flat_x, flat_y, flat_bounds)):
                if ok and not scenario.obstacle_map.is_collision(float(x), float(y)):
                    flat_obs[j] = 0.0
            obstacle = flat_obs.reshape(threat.shape)
            if base_image_cache is not None:
                base_image_cache[cache_key] = np.stack([threat, obstacle]).astype(np.float32)
        cost_density = bid_alpha + bid_beta * threat

        images[idx, 0] = cost_density.astype(np.float32)
        images[idx, 1] = obstacle.astype(np.float32)
    return images


def load_or_build_corridor_images(
    df: pd.DataFrame,
    metadata: dict,
    image_size: tuple[int, int] = (48, 48),
    lateral_width: float = 320.0,
    scale_mode: str = "fixed_width",
    cache_dir: Path | None = None,
    scenario_cache: dict[int, Scenario] | None = None,
    base_image_cache: dict[tuple, np.ndarray] | None = None,
) -> np.ndarray:
    """Load cached corridor images, or rasterize and cache them.

    Channel 0 is physical bid-cost density alpha + beta * threat_rate, and
    channel 1 is obstacle occupancy. Model-specific normalization is applied
    by callers using each checkpoint's image scaler.
    """
    if cache_dir is None:
        return build_corridor_images(
            df,
            metadata,
            image_size=image_size,
            lateral_width=lateral_width,
            scale_mode=scale_mode,
            scenario_cache=scenario_cache,
            base_image_cache=base_image_cache,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"corridor_images_{corridor_image_cache_key(df, metadata, image_size, lateral_width, scale_mode)}.npy"
    if cache_path.exists():
        cached = np.load(cache_path, mmap_mode=None)
        expected_shape = (len(df), 2, int(image_size[0]), int(image_size[1]))
        if cached.shape == expected_shape:
            return cached.astype(np.float32, copy=False)

    images = build_corridor_images(
        df,
        metadata,
        image_size=image_size,
        lateral_width=lateral_width,
        scale_mode=scale_mode,
        scenario_cache=scenario_cache,
        base_image_cache=base_image_cache,
    )
    tmp_path = cache_path.with_suffix(".tmp.npy")
    np.save(tmp_path, images)
    try:
        tmp_path.replace(cache_path)
    except PermissionError:
        # Windows can briefly hold a just-written large .npy file; cache failure
        # must not change the feature tensor returned to the caller.
        pass
    return images


def corridor_image_cache_key(
    df: pd.DataFrame,
    metadata: dict,
    image_size: tuple[int, int],
    lateral_width: float,
    scale_mode: str = "fixed_width",
) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"corridor_images_v3_bid_cost_density")
    hasher.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    hasher.update(str(tuple(int(x) for x in image_size)).encode("ascii"))
    hasher.update(f"{float(lateral_width):.8f}".encode("ascii"))
    hasher.update(str(scale_mode).encode("ascii"))
    columns = ["scenario_id", "start_x", "start_y", "goal_x", "goal_y"]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot cache corridor images; missing columns: {missing}")
    row_hash = pd.util.hash_pandas_object(df[columns].reset_index(drop=True), index=False).to_numpy(dtype=np.uint64)
    hasher.update(str(len(df)).encode("ascii"))
    hasher.update(row_hash.tobytes())
    return hasher.hexdigest()[:24]


def scenario_from_metadata(metadata: dict, scenario_id: int) -> Scenario:
    config = metadata["config"]["map"]
    scenario_meta = next(item for item in metadata["scenarios"] if int(item["scenario_id"]) == scenario_id)
    sources = [GaussianThreat(**source) for source in scenario_meta["threat_sources"]]
    threat = ThreatField(config["width"], config["height"], config["resolution"], sources=sources)
    circles = [CircleObstacle(**obstacle) for obstacle in scenario_meta["circle_obstacles"]]
    rectangles = [RectangleObstacle(**obstacle) for obstacle in scenario_meta["rectangle_obstacles"]]
    return Scenario(config["width"], config["height"], threat, ObstacleMap(circles=circles, rectangles=rectangles))


def wrap_angle_array(angle) -> np.ndarray:
    angle = np.asarray(angle, dtype=np.float64)
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
