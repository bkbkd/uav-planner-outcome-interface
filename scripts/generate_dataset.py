from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_generator import DatasetConfig, PlannerConfig, ThreatConfig, generate_dataset
from src.experiment_config import CURRENT_MIN_START_GOAL_DISTANCE, CURRENT_PLANNER


def primitive_dt_values(args: argparse.Namespace) -> tuple[float, ...]:
    return tuple(float(value) for value in args.primitive_dt_values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate start-goal path-cost labels with the fixed lattice planner.")
    parser.add_argument("--output", type=Path, default=Path("outputs/datasets/debug"), help="Output directory.")
    parser.add_argument("--samples", type=int, default=8, help="Number of start-goal samples.")
    parser.add_argument("--scenarios", type=int, default=2, help="Number of random scenarios.")
    parser.add_argument("--map-seed", type=int, required=True, help="Seed used only for scenario generation.")
    parser.add_argument("--sample-seed", type=int, required=True, help="Seed used only for start-goal sampling.")
    parser.add_argument("--threat-layout", choices=["random", "central_barrier"], default="random")
    parser.add_argument("--start-goal-mode", choices=["uniform", "barrier_crossing", "mixed"], default="uniform")
    parser.add_argument("--mixed-barrier-crossing-prob", type=float, default=0.5, help="Probability of barrier_crossing samples when --start-goal-mode=mixed.")
    parser.add_argument("--min-start-goal-distance", type=float, default=CURRENT_MIN_START_GOAL_DISTANCE)
    parser.add_argument("--max-iters", type=int, default=CURRENT_PLANNER["max_iters"])
    parser.add_argument("--omega-max", type=float, default=CURRENT_PLANNER["omega_max"])
    parser.add_argument("--n-actions", type=int, default=CURRENT_PLANNER["n_actions"])
    parser.add_argument(
        "--primitive-dt-values",
        nargs="+",
        type=float,
        default=list(CURRENT_PLANNER["primitive_dt_values"]),
        help="Motion primitive dt values. A single value is represented as a one-element list.",
    )
    parser.add_argument("--goal-tolerance", type=float, default=CURRENT_PLANNER["goal_tolerance"])
    parser.add_argument("--planner-alpha", type=float, default=CURRENT_PLANNER["alpha"], help="Planner objective weight for path length.")
    parser.add_argument("--planner-beta", type=float, default=CURRENT_PLANNER["beta"], help="Planner objective weight for path risk.")
    parser.add_argument("--planner-gamma", type=float, default=CURRENT_PLANNER["gamma"], help="Planner objective weight for turning.")
    parser.add_argument("--lattice-xy-resolution", type=float, default=CURRENT_PLANNER["lattice_xy_resolution"])
    parser.add_argument("--lattice-heading-bins", type=int, default=CURRENT_PLANNER["lattice_heading_bins"])
    parser.add_argument("--lattice-post-solution-expansion-limit", type=int, default=CURRENT_PLANNER["lattice_post_solution_expansion_limit"])
    parser.add_argument("--lattice-guidance-weight", type=float, default=CURRENT_PLANNER["lattice_grid_guidance_weight"])
    parser.add_argument("--free-start-heading", action="store_true", default=CURRENT_PLANNER["free_start_heading"])
    parser.add_argument(
        "--lattice-priority-mode",
        choices=[
            "straight",
        ],
        default=CURRENT_PLANNER["lattice_priority_mode"],
    )
    parser.add_argument("--save-threat-grids", action="store_true", help="Save threat grids to scenario_grids.npz.")
    parser.add_argument("--save-paths", action="store_true", help="Save successful planned paths to paths.npz.")
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N samples; 0 disables progress.")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Write partial CSV/JSONL outputs every N samples; 0 disables checkpointing.")
    parser.add_argument("--resume", action="store_true", help="Continue from an existing samples.csv in the output directory.")
    args = parser.parse_args()

    config = DatasetConfig(
        n_samples=args.samples,
        n_scenarios=args.scenarios,
        start_goal_mode=args.start_goal_mode,
        mixed_barrier_crossing_prob=args.mixed_barrier_crossing_prob,
        min_start_goal_distance=args.min_start_goal_distance,
        map_seed=args.map_seed,
        sample_seed=args.sample_seed,
        threat=ThreatConfig(layout=args.threat_layout),
        planner=PlannerConfig(
            max_iters=args.max_iters,
            omega_max=args.omega_max,
            n_actions=args.n_actions,
            primitive_dt_values=primitive_dt_values(args),
            goal_tolerance=args.goal_tolerance,
            alpha=args.planner_alpha,
            beta=args.planner_beta,
            gamma=args.planner_gamma,
            lattice_xy_resolution=args.lattice_xy_resolution,
            lattice_heading_bins=args.lattice_heading_bins,
            lattice_post_solution_expansion_limit=args.lattice_post_solution_expansion_limit,
            lattice_grid_guidance_weight=args.lattice_guidance_weight,
            free_start_heading=args.free_start_heading,
            lattice_priority_mode=args.lattice_priority_mode,
        ),
    )
    records = generate_dataset(
        args.output,
        config,
        save_threat_grids=args.save_threat_grids,
        save_paths=args.save_paths,
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    feasible = sum(int(record["feasible"]) for record in records)
    print(f"done: {feasible}/{len(records)} feasible samples")
    print(f"samples: {args.output / 'samples.csv'}")
    print(f"metadata: {args.output / 'metadata.json'}")


if __name__ == "__main__":
    main()
