from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_scenario_protocol import scenario_hashes


PROFILES = {"beta150": 150.0, "beta650": 650.0, "beta1500": 1500.0}
EDGE_GEOMETRY = [
    "sample_id",
    "scenario_id",
    "seed",
    "start_x",
    "start_y",
    "start_theta",
    "goal_x",
    "goal_y",
    "split",
    "distribution",
    "map_type",
]
PAIR_GEOMETRY = [
    "instance_id",
    "scenario_id",
    "agent_id",
    "task_id",
    "start_x",
    "start_y",
    "start_theta",
    "goal_x",
    "goal_y",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all final edge, assignment, and rolling data assets.")
    parser.add_argument("--root", type=Path, default=Path("outputs/final_ras"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = {
        "edge": audit_edge_profiles(args.root),
        "assignment": audit_assignment_profiles(args.root),
    }
    all_groups = report["edge"]["map_hash_groups"] | report["assignment"]["map_hash_groups"]
    report["map_isolation"] = assert_disjoint_hash_groups(all_groups)
    report["ok"] = True
    report["edge"].pop("map_hash_groups")
    report["assignment"].pop("map_hash_groups")

    output = args.output or args.root / "final_data_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def audit_edge_profiles(root: Path) -> dict[str, Any]:
    frames = {}
    metadata = {}
    for profile, beta in PROFILES.items():
        directory = root / f"edges_{profile}"
        frames[profile] = pd.read_csv(directory / "samples.csv").sort_values("sample_id").reset_index(drop=True)
        metadata[profile] = load_json(directory / "metadata.json")
        if len(frames[profile]) != 20_000:
            raise ValueError(f"{profile}: expected 20000 edge rows, got {len(frames[profile])}")
        assert_planner_beta(metadata[profile], beta, profile)

    reference = frames["beta650"]
    for profile, frame in frames.items():
        assert_frame_equal(reference, frame, EDGE_GEOMETRY, f"edge geometry beta650 vs {profile}")
        assert_planners_equal_except_beta(metadata["beta650"], metadata[profile], profile)

    split_hashes: dict[str, set[str]] = {}
    hashes = scenario_hashes(metadata["beta650"])
    scenarios = pd.DataFrame(metadata["beta650"]["scenarios"])
    for split, group in scenarios.groupby("split"):
        split_hashes[f"edge/{split}"] = {hashes[int(item)] for item in group["scenario_id"]}
    return {
        "rows_per_profile": {profile: int(len(frame)) for profile, frame in frames.items()},
        "shared_geometry": True,
        "map_hash_groups": split_hashes,
    }


def audit_assignment_profiles(root: Path) -> dict[str, Any]:
    map_groups: dict[str, set[str]] = {}
    pool_report = {}
    for pool in ("development", "test_1", "test_2", "test_3"):
        frames = {}
        instances = {}
        metadata = {}
        for profile, beta in PROFILES.items():
            directory = root / "assignment" / pool / profile
            if profile == "beta650":
                directory = directory / "merged"
            frames[profile] = pd.read_csv(directory / "pairs.csv").sort_values(
                ["instance_id", "agent_id", "task_id"]
            ).reset_index(drop=True)
            instances[profile] = pd.read_csv(directory / "instances.csv").sort_values("instance_id").reset_index(drop=True)
            metadata[profile] = load_json(directory / "source_metadata.json")
            if len(instances[profile]) != 500 or len(frames[profile]) != 12_500:
                raise ValueError(
                    f"{pool}/{profile}: expected 500 instances and 12500 pairs; "
                    f"got {len(instances[profile])} and {len(frames[profile])}"
                )
            assert_planner_beta(metadata[profile], beta, f"{pool}/{profile}")
            assert_bid_formulas(frames[profile], metadata[profile], f"{pool}/{profile}")

        for profile in PROFILES:
            assert_frame_equal(frames["beta650"], frames[profile], PAIR_GEOMETRY, f"{pool} pair geometry")
            assert_frame_equal(
                instances["beta650"],
                instances[profile],
                ["instance_id", "scenario_id", "agents_json", "tasks_json"],
                f"{pool} instances",
            )
            assert_planners_equal_except_beta(metadata["beta650"], metadata[profile], f"{pool}/{profile}")

        map_groups[f"assignment/{pool}"] = set(scenario_hashes(metadata["beta650"]).values())
        pool_report[pool] = {"instances": 500, "pairs_per_profile": 12_500, "shared_geometry": True}

    rolling_metadata = load_json(root / "scenario_pools" / "rolling_test" / "metadata.json")
    map_groups["rolling/test"] = set(scenario_hashes(rolling_metadata).values())
    return {"pools": pool_report, "map_hash_groups": map_groups}


def assert_bid_formulas(frame: pd.DataFrame, metadata: dict, label: str) -> None:
    planner = metadata["config"]["planner"]
    alpha, beta = float(planner["alpha"]), float(planner["beta"])
    true_expected = alpha * frame["length"].astype(float) + beta * frame["risk"].astype(float)
    base_expected = alpha * frame["euclidean_distance"].astype(float) + beta * frame["straight_line_risk"].astype(float)
    if float(np.max(np.abs(frame["true_bid"].astype(float) - true_expected))) > 1e-4:
        raise ValueError(f"{label}: true_bid formula mismatch")
    if float(np.max(np.abs(frame["baseline_bid"].astype(float) - base_expected))) > 1e-4:
        raise ValueError(f"{label}: baseline_bid formula mismatch")


def assert_planner_beta(metadata: dict, expected: float, label: str) -> None:
    observed = float(metadata["config"]["planner"]["beta"])
    if observed != float(expected):
        raise ValueError(f"{label}: expected beta={expected}, got {observed}")


def assert_planners_equal_except_beta(left: dict, right: dict, label: str) -> None:
    left_planner = dict(left["config"]["planner"])
    right_planner = dict(right["config"]["planner"])
    left_planner.pop("beta", None)
    right_planner.pop("beta", None)
    if left_planner != right_planner:
        raise ValueError(f"{label}: planner settings differ beyond beta")


def assert_frame_equal(left: pd.DataFrame, right: pd.DataFrame, columns: list[str], label: str) -> None:
    left_view = left[columns].reset_index(drop=True)
    right_view = right[columns].reset_index(drop=True)
    numeric_columns = [
        column
        for column in columns
        if column in left_view.columns
        and column in right_view.columns
        and pd.api.types.is_numeric_dtype(left_view[column])
        and pd.api.types.is_numeric_dtype(right_view[column])
    ]
    text_columns = [column for column in columns if column not in numeric_columns]
    if text_columns:
        pd.testing.assert_frame_equal(
            left_view[text_columns],
            right_view[text_columns],
            check_dtype=False,
            check_exact=True,
            obj=f"{label} categorical columns",
        )
    if numeric_columns:
        pd.testing.assert_frame_equal(
            left_view[numeric_columns],
            right_view[numeric_columns],
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-9,
            obj=f"{label} numeric columns",
        )


def assert_disjoint_hash_groups(groups: dict[str, set[str]]) -> dict[str, Any]:
    names = sorted(groups)
    comparisons = []
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(f"Map-content overlap between {left} and {right}: {len(overlap)}")
            comparisons.append({"left": left, "right": right, "overlap": 0})
    return {"groups": {name: len(groups[name]) for name in names}, "comparisons": comparisons}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
