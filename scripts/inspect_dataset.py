from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.learning.features import scenario_from_metadata
from src.utils.visualization import plot_path, plot_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect generated path-cost dataset quality.")
    parser.add_argument("dataset_dir", type=Path, help="Directory containing samples.csv and metadata.json.")
    parser.add_argument("--sample-id", type=int, default=None, help="Sample id to render. Defaults to first feasible sample.")
    parser.add_argument("--save", type=Path, default=None, help="Output directory for figures. Defaults to dataset_dir/qc.")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive Matplotlib windows.")
    args = parser.parse_args()

    records = read_records(args.dataset_dir / "samples.csv")
    metadata = json.loads((args.dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    output_dir = args.save or args.dataset_dir / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    print_summary(records)
    save_summary_figure(records, output_dir / "summary.png")
    print(f"summary figure: {output_dir / 'summary.png'}")

    sample_id = args.sample_id
    if sample_id is None:
        sample_id = first_feasible_sample(records)
    if sample_id is not None:
        rendered = render_sample(args.dataset_dir, output_dir, metadata, records, sample_id)
        if rendered:
            print(f"sample figure: {output_dir / f'sample_{sample_id}.png'}")
    else:
        print("no feasible sample available for path rendering")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def print_summary(records: list[dict[str, Any]]) -> None:
    n = len(records)
    feasible = [record for record in records if int(float(record["feasible"])) == 1]
    print(f"samples: {n}")
    print(f"feasible: {len(feasible)}/{n} ({len(feasible) / max(n, 1):.1%})")
    for key in ["length", "risk", "survival_prob", "turning", "runtime_sec"]:
        values = numeric_values(feasible, key)
        if len(values):
            print(f"{key}: mean={values.mean():.3f}, median={np.median(values):.3f}, min={values.min():.3f}, max={values.max():.3f}")
    straight_collision = numeric_values(records, "straight_line_collision")
    if len(straight_collision):
        print(f"straight-line collision rate: {straight_collision.mean():.1%}")


def save_summary_figure(records: list[dict[str, Any]], path: Path) -> None:
    feasible = [record for record in records if int(float(record["feasible"])) == 1]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    plot_hist(axes[0, 0], numeric_values(feasible, "length"), "Planned path length / m")
    plot_hist(axes[0, 1], numeric_values(feasible, "risk"), "Cumulative risk")
    plot_hist(axes[1, 0], numeric_values(feasible, "runtime_sec"), "Runtime / s")

    euclidean = numeric_values(feasible, "euclidean_distance")
    length = numeric_values(feasible, "length")
    risk = numeric_values(feasible, "risk")
    ax = axes[1, 1]
    if len(euclidean) and len(length):
        scatter = ax.scatter(euclidean, length, c=risk, cmap="viridis", s=36, edgecolors="black", linewidths=0.35)
        fig.colorbar(scatter, ax=ax, label="risk")
    ax.set_xlabel("Euclidean distance / m")
    ax.set_ylabel("Planned length / m")
    ax.set_title("Geometry vs. planned cost")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180)


def plot_hist(ax, values: np.ndarray, title: str) -> None:
    if len(values):
        ax.hist(values, bins=min(20, max(4, int(math.sqrt(len(values)) + 1))), color="#3b82f6", edgecolor="white")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def render_sample(
    dataset_dir: Path,
    output_dir: Path,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
    sample_id: int,
) -> bool:
    record = next((item for item in records if int(item["sample_id"]) == sample_id), None)
    if record is None:
        print(f"sample {sample_id} not found")
        return False
    if int(float(record["feasible"])) != 1:
        print(f"sample {sample_id} is infeasible; no path to render")
        return False

    paths_file = dataset_dir / "paths.npz"
    if not paths_file.exists():
        print("paths.npz not found; regenerate with --save-paths to render trajectories")
        return False
    paths = np.load(paths_file)
    key = f"path_{sample_id}"
    if key not in paths:
        print(f"{key} not found in paths.npz")
        return False

    scenario = scenario_from_metadata(metadata, int(record["scenario_id"]))
    path = paths[key]
    start = (float(record["start_x"]), float(record["start_y"]))
    goal = (float(record["goal_x"]), float(record["goal_y"]))

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    _, image = plot_scenario(scenario, ax=ax)
    fig.colorbar(image, ax=ax, label="threat rate")
    plot_path(path, ax=ax)
    ax.scatter([start[0]], [start[1]], marker="o", s=90, c="#7cff6b", edgecolors="black", label="start", zorder=5)
    ax.scatter([goal[0]], [goal[1]], marker="*", s=180, c="#ffffff", edgecolors="black", label="goal", zorder=5)
    ax.legend(loc="upper left")
    ax.set_title(
        f"sample {sample_id} | L={float(record['length']):.1f} m, "
        f"R={float(record['risk']):.3f}, P={float(record['survival_prob']):.3f}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"sample_{sample_id}.png", dpi=180)
    return True


def first_feasible_sample(records: list[dict[str, Any]]) -> int | None:
    for record in records:
        if int(float(record["feasible"])) == 1:
            return int(record["sample_id"])
    return None


def numeric_values(records: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for record in records:
        try:
            value = float(record[key])
        except (KeyError, ValueError, TypeError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


if __name__ == "__main__":
    main()
