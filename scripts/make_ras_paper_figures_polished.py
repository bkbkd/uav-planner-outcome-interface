from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D

from src.data.dataset_generator import PlannerConfig, run_planner
from src.learning.features import scenario_from_metadata


PROFILE_COLORS = {"fast": "#d95f02", "balanced": "#1b9e77", "safe": "#386cb0"}
PROFILE_HATCHES = {"fast": "///", "balanced": "\\\\", "safe": "..."}
EDGE_COLOR = "#7c3aed"
GLOBAL_COLOR = "#64748b"
EXACT_COLOR = "#991b1b"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the current RAS manuscript figures from frozen results.")
    parser.add_argument("--commitment-summary-dir", type=Path, required=True)
    parser.add_argument("--rolling-commitment-dir", type=Path, required=True)
    parser.add_argument(
        "--rolling-exact-dir",
        type=Path,
        default=Path("outputs/final_ras/results/shared_h192/rolling_exact_reference"),
    )
    parser.add_argument("--static-runtime-dir", type=Path, required=True)
    parser.add_argument(
        "--grid-transfer-summary",
        type=Path,
        default=Path("outputs/planner_transfer/grid12/summary/grid_transfer_commitment_summary.csv"),
    )
    parser.add_argument(
        "--scale-10-dir",
        type=Path,
        default=Path("outputs/final_ras/results/shared_h192/scale_10x10/evaluation"),
    )
    parser.add_argument(
        "--scale-20-dir",
        type=Path,
        default=Path("outputs/final_ras/results/shared_h192/scale_20x20/evaluation"),
    )
    parser.add_argument(
        "--development-benchmark-dir",
        type=Path,
        default=Path("outputs/final_ras/assignment/development/beta650/merged"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures/ras_interface"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    make_commitment_example(
        args.commitment_summary_dir,
        args.development_benchmark_dir,
        args.output_dir,
    )
    make_commitment_evidence(args.commitment_summary_dir, args.output_dir)
    make_operating_regime_appendix(args.commitment_summary_dir, args.output_dir)
    make_portability(
        args.commitment_summary_dir,
        args.grid_transfer_summary,
        args.scale_10_dir,
        args.scale_20_dir,
        args.output_dir,
    )
    make_runtime_accounting(args.static_runtime_dir, args.output_dir)
    make_rolling_stress(args.rolling_commitment_dir, args.rolling_exact_dir, args.output_dir)
    print(f"saved figures to: {args.output_dir}")


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 8.7,
            "axes.titlesize": 9.3,
            "axes.labelsize": 8.8,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.linewidth": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.6,
            "lines.markersize": 5.0,
            "patch.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    for suffix in (".png", ".pdf"):
        fig.savefig(output_dir / f"{name}{suffix}", bbox_inches="tight")
    plt.close(fig)


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    resamples: int = 5000,
) -> tuple[float, float, float]:
    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty sample")
    boot = np.mean(rng.choice(values, size=(resamples, len(values)), replace=True), axis=1)
    return (
        float(np.mean(values)),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def make_commitment_example(
    summary_dir: Path,
    benchmark_dir: Path,
    output_dir: Path,
) -> None:
    example = json.loads((summary_dir / "fleet_example.json").read_text(encoding="utf-8"))
    instance_id = int(example["instance_id"])
    instances = pd.read_csv(benchmark_dir / "instances.csv")
    instance = instances[instances["instance_id"].astype(int) == instance_id]
    if len(instance) != 1:
        raise ValueError(f"Expected one development instance {instance_id}, found {len(instance)}")
    instance = instance.iloc[0]
    agents = np.asarray(json.loads(instance["agents_json"]), dtype=float)
    tasks = np.asarray(json.loads(instance["tasks_json"]), dtype=float)

    profile_dirs = {
        "fast": benchmark_dir.parents[1] / "beta150",
        "balanced": benchmark_dir,
        "safe": benchmark_dir.parents[1] / "beta1500",
    }
    metadata = {
        profile: json.loads((path / "source_metadata.json").read_text(encoding="utf-8"))
        for profile, path in profile_dirs.items()
    }
    scenario = scenario_from_metadata(metadata["balanced"], int(instance["scenario_id"]))
    routes = load_or_build_fleet_example_routes(
        summary_dir,
        example,
        scenario,
        agents,
        tasks,
        metadata,
    )

    methods = ("global_profile", "portfolio")
    titles = ("Hindsight best uniform profile", "Per-edge profile selection")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.55), sharex=True, sharey=True)
    image = None
    for ax, method, title in zip(axes, methods, titles):
        row = example["methods"][method]
        modes = row["modes"].split(";")
        ax.grid(False)
        image = scenario.threat_field.plot(ax=ax, cmap="YlOrRd", alpha=0.58)
        scenario.obstacle_map.plot(
            ax=ax,
            facecolor="#374151",
            edgecolor="white",
            alpha=0.9,
        )
        for agent_index, mode in enumerate(modes):
            path = routes[f"{method}_{agent_index}"]
            ax.plot(
                path[:, 0],
                path[:, 1],
                color="white",
                linewidth=3.6,
                alpha=0.85,
                zorder=3,
            )
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=PROFILE_COLORS[mode],
                linewidth=2.25,
                zorder=4,
            )
        draw_fleet_endpoints(ax, agents, tasks)
        ax.set_title(title, fontweight="bold", pad=7)
        ax.text(
            0.02,
            0.02,
            f"L = {float(row['length']):.1f} m\n"
            f"R = {float(row['risk']):.4f} / B = {float(example['risk_budget']):.4f}\n"
            f"profiles: {format_mode_mix(modes)}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.4,
            bbox={"facecolor": "white", "edgecolor": "#475569", "alpha": 0.86, "pad": 3.0},
            zorder=8,
        )
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    gain = float(example["mixing_gain"])
    fig.suptitle(
        f"Same dispatch, matching, and exposure budget: mixed profiles save {gain:.2f} m",
        fontsize=10.0,
        fontweight="bold",
        y=0.995,
    )
    handles = [
        Line2D([0], [0], color=PROFILE_COLORS["fast"], lw=2.5, label="fast route"),
        Line2D([0], [0], color=PROFILE_COLORS["safe"], lw=2.5, label="safe route"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#111827", markeredgecolor="white", label="UAV"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="white", markeredgecolor="#111827", markersize=8, label="task"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.46, -0.005))
    if image is not None:
        colorbar_axis = fig.add_axes([0.92, 0.18, 0.014, 0.62])
        colorbar_axis.grid(False)
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("threat exposure rate")
    fig.subplots_adjust(left=0.065, right=0.91, bottom=0.13, top=0.89, wspace=0.08)
    save(fig, output_dir, "fig2_commitment_example")


def load_or_build_fleet_example_routes(
    summary_dir: Path,
    example: dict[str, object],
    scenario: object,
    agents: np.ndarray,
    tasks: np.ndarray,
    metadata: dict[str, dict[str, object]],
) -> dict[str, np.ndarray]:
    cache_path = summary_dir / "fleet_example_routes.npz"
    if cache_path.exists():
        with np.load(cache_path) as cached:
            return {key: cached[key].copy() for key in cached.files}

    route_cache: dict[tuple[int, int, str], tuple[np.ndarray, float, float]] = {}
    routes: dict[str, np.ndarray] = {}
    for method in ("global_profile", "portfolio"):
        row = example["methods"][method]
        assignment = parse_assignment(row["assignment"])
        modes = row["modes"].split(";")
        total_length = 0.0
        total_risk = 0.0
        for agent_index, (task_index, mode) in enumerate(zip(assignment, modes)):
            key = (agent_index, task_index, mode)
            if key not in route_cache:
                planner_config = PlannerConfig(**metadata[mode]["config"]["planner"])
                start = tuple(float(value) for value in agents[agent_index, :3])
                goal = tuple(float(value) for value in tasks[task_index, :2])
                result, _ = run_planner(scenario, start, goal, planner_config)
                if result is None:
                    raise RuntimeError(f"Failed to reconstruct route {key}")
                route_cache[key] = (
                    result.path.astype(np.float32),
                    float(result.metrics["length"]),
                    float(result.metrics["risk"]),
                )
            path, length, risk = route_cache[key]
            routes[f"{method}_{agent_index}"] = path
            total_length += length
            total_risk += risk
        if not np.isclose(total_length, float(row["length"]), atol=1e-6):
            raise RuntimeError(f"{method} reconstructed length {total_length} != frozen {row['length']}")
        if not np.isclose(total_risk, float(row["risk"]), atol=1e-9):
            raise RuntimeError(f"{method} reconstructed risk {total_risk} != frozen {row['risk']}")
    np.savez_compressed(cache_path, **routes)
    return routes


def parse_assignment(value: str) -> list[int]:
    pairs = [item.split("->") for item in str(value).split(";") if item]
    ordered = sorted((int(agent), int(task)) for agent, task in pairs)
    return [task for _, task in ordered]


def format_mode_mix(modes: list[str]) -> str:
    counts = {mode: modes.count(mode) for mode in ("fast", "balanced", "safe")}
    return " + ".join(f"{count} {mode}" for mode, count in counts.items() if count)


def draw_fleet_endpoints(ax: plt.Axes, agents: np.ndarray, tasks: np.ndarray) -> None:
    ax.scatter(
        agents[:, 0],
        agents[:, 1],
        marker="o",
        s=34,
        c="#111827",
        edgecolors="white",
        linewidths=0.8,
        zorder=6,
    )
    ax.scatter(
        tasks[:, 0],
        tasks[:, 1],
        marker="*",
        s=72,
        c="white",
        edgecolors="#111827",
        linewidths=0.8,
        zorder=6,
    )
    for index, (x, y, _) in enumerate(agents):
        ax.annotate(f"U{index + 1}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6.4, fontweight="bold", zorder=7)
    for index, (x, y) in enumerate(tasks):
        ax.annotate(f"T{index + 1}", (x, y), xytext=(4, -8), textcoords="offset points", fontsize=6.4, fontweight="bold", zorder=7)


def make_commitment_evidence(summary_dir: Path, output_dir: Path) -> None:
    exact = pd.read_csv(summary_dir / "exact_commitment_summary.csv")
    exact = exact[exact["method"] == "global_profile"].sort_values("budget_quantile")
    learned = pd.read_csv(summary_dir / "learned_commitment_paired.csv")
    learned = learned[
        (learned["reference_method"] == "dispatch_global_learned")
        & (learned["scope"] == "common_true_nonviolating")
    ].sort_values("budget_quantile")
    regimes = pd.read_csv(summary_dir / "operating_regime_instances.csv")
    regimes = regimes[
        (regimes["pool"] != "development")
        & (regimes["budget_quantile"].isin([0.1, 0.5, 0.9]))
    ]

    fig, (exact_ax, learned_ax, regime_ax) = plt.subplots(1, 3, figsize=(10.4, 3.25))
    q = exact["budget_quantile"].to_numpy(float)
    exact_ax.plot(q, exact["gain_to_k3_mean"], marker="o", color=EXACT_COLOR)
    exact_ax.axhline(0, color="#475569", linewidth=0.9)
    exact_ax.set_xlabel("budget quantile")
    exact_ax.set_ylabel("per-edge gain (m)")
    exact_ax.set_title("(a) Planner-generated outcomes")

    ql = learned["budget_quantile"].to_numpy(float)
    mean = learned["gain_mean"].to_numpy(float)
    lo = learned["gain_ci95_low"].to_numpy(float)
    hi = learned["gain_ci95_high"].to_numpy(float)
    learned_ax.plot(ql, mean, marker="o", color=EDGE_COLOR)
    learned_ax.fill_between(ql, lo, hi, color=EDGE_COLOR, alpha=0.15, linewidth=0)
    learned_ax.axhline(0, color="#475569", linewidth=0.9)
    learned_ax.set_xlabel("budget quantile")
    learned_ax.set_ylabel("per-edge gain (m)")
    learned_ax.set_title("(b) Predicted outcomes, same predictions")

    budgets = np.asarray([0.1, 0.5, 0.9])
    x = np.arange(len(budgets), dtype=float)
    regime_specs = (
        ("low disagreement + low utilization", "low", "low", GLOBAL_COLOR, "o", -0.07),
        ("high disagreement + high utilization", "high", "high", EDGE_COLOR, "s", 0.07),
    )
    rng = np.random.default_rng(20260712)
    for label, disagreement, utilization, color, marker, offset in regime_specs:
        means = []
        lows = []
        highs = []
        counts = []
        for budget in budgets:
            values = regimes[
                np.isclose(regimes["budget_quantile"], budget)
                & (regimes["assignment_disagreement_bin"] == disagreement)
                & (regimes["budget_utilization_bin"] == utilization)
            ]["mixing_gain"].to_numpy(float)
            if len(values) == 0:
                raise ValueError(f"No held-out operating-regime samples for q={budget}, {label}")
            sample_mean, sample_low, sample_high = bootstrap_mean_ci(values, rng)
            means.append(sample_mean)
            lows.append(sample_low)
            highs.append(sample_high)
            counts.append(len(values))
        means_array = np.asarray(means)
        yerr = np.vstack((means_array - np.asarray(lows), np.asarray(highs) - means_array))
        regime_ax.errorbar(
            x + offset,
            means_array,
            yerr=yerr,
            color=color,
            marker=marker,
            capsize=2.5,
            label=label,
        )
        for xpos, value, count in zip(x + offset, means_array, counts):
            regime_ax.annotate(
                f"$n={count}$",
                (xpos, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                color=color,
            )
    regime_ax.axhline(0, color="#475569", linewidth=0.9)
    regime_ax.set_xticks(x, ["tight\n$q=0.1$", "medium\n$q=0.5$", "loose\n$q=0.9$"])
    regime_ax.set_ylabel("per-edge gain (m)")
    regime_ax.set_title("(c) When per-edge selection matters")
    regime_ax.legend(loc="upper right", frameon=False, fontsize=6.5)

    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.2, top=0.9, wspace=0.34)
    save(fig, output_dir, "fig3_commitment_evidence")


def make_operating_regime_appendix(summary_dir: Path, output_dir: Path) -> None:
    regimes = pd.read_csv(summary_dir / "operating_regime_joint_summary.csv")
    regimes = regimes[regimes["budget_quantile"].isin([0.1, 0.5, 0.9])]
    bin_order_x = ["low", "medium", "high"]
    bin_order_y = ["high", "medium", "low"]
    vmax = float(regimes["mixing_gain_mean"].max())
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.75))
    heat_image = None
    for index, (ax, budget) in enumerate(zip(axes, (0.1, 0.5, 0.9))):
        subset = regimes[np.isclose(regimes["budget_quantile"], budget)].set_index(
            ["budget_utilization_bin", "assignment_disagreement_bin"]
        )
        values = np.asarray(
            [[subset.loc[(row, col), "mixing_gain_mean"] for col in bin_order_x] for row in bin_order_y],
            dtype=float,
        )
        counts = np.asarray(
            [[subset.loc[(row, col), "n"] for col in bin_order_x] for row in bin_order_y],
            dtype=int,
        )
        ax.grid(False)
        heat_image = ax.imshow(values, cmap="Purples", vmin=0.0, vmax=vmax, aspect="auto")
        for row in range(3):
            for col in range(3):
                color = "white" if values[row, col] > 0.52 * vmax else "#111827"
                ax.text(col, row, f"{values[row, col]:.1f}\n$n={counts[row, col]}$", ha="center", va="center", fontsize=6.8, color=color)
        ax.set_xticks(range(3), bin_order_x)
        ax.set_yticks(range(3), bin_order_y if index == 0 else [])
        ax.set_xlabel("assignment disagreement")
        if index == 0:
            ax.set_ylabel("budget utilization")
        ax.set_title(f"({chr(ord('a') + index)}) $q={budget:.1f}$")
    assert heat_image is not None
    colorbar_axis = fig.add_axes([0.925, 0.2, 0.014, 0.63])
    colorbar_axis.grid(False)
    colorbar = fig.colorbar(heat_image, cax=colorbar_axis)
    colorbar.set_label("mean per-edge gain (m)")
    fig.subplots_adjust(left=0.075, right=0.9, bottom=0.22, top=0.86, wspace=0.22)
    save(fig, output_dir, "figB1_operating_regimes")


def make_portability(
    commitment_summary_dir: Path,
    grid_summary_path: Path,
    scale_10_dir: Path,
    scale_20_dir: Path,
    output_dir: Path,
) -> None:
    rng = np.random.default_rng(20260712)
    lattice = load_lattice_exact_gains(commitment_summary_dir)
    grid = pd.read_csv(grid_summary_path).sort_values("budget_quantile")

    fig, (planner_ax, scale_ax) = plt.subplots(1, 2, figsize=(9.6, 3.45))
    lattice_summary = []
    for budget, subset in lattice.groupby("budget_quantile", sort=True):
        mean, low, high = bootstrap_mean_ci(subset["gain"].to_numpy(float), rng)
        lattice_summary.append((float(budget), mean, low, high))
    lattice_summary = np.asarray(lattice_summary, dtype=float)
    planner_ax.plot(
        lattice_summary[:, 0],
        lattice_summary[:, 1],
        marker="o",
        color=EXACT_COLOR,
        label="nonholonomic lattice planner",
    )
    planner_ax.fill_between(
        lattice_summary[:, 0],
        lattice_summary[:, 2],
        lattice_summary[:, 3],
        color=EXACT_COLOR,
        alpha=0.13,
        linewidth=0,
    )
    planner_ax.plot(
        grid["budget_quantile"],
        grid["edge_wise_gain_mean"],
        marker="s",
        color="#0f766e",
        label="holonomic grid planner",
    )
    planner_ax.fill_between(
        grid["budget_quantile"].to_numpy(float),
        grid["edge_wise_gain_ci95_low"].to_numpy(float),
        grid["edge_wise_gain_ci95_high"].to_numpy(float),
        color="#0f766e",
        alpha=0.13,
        linewidth=0,
    )
    planner_ax.axhline(0, color="#475569", linewidth=0.9)
    planner_ax.set_xlabel("budget quantile")
    planner_ax.set_ylabel("per-edge gain (m)")
    planner_ax.set_title("(a) Different planners")
    planner_ax.legend(frameon=False, loc="upper right")

    scale_frames = {
        "5$\\times$5\n75 outcomes\n$n=1{,}500$": scale_gain_frame(lattice, 5, {0.1: "tight", 0.5: "medium", 0.9: "loose"}),
        "10$\\times$10\n300 outcomes\n$n=150$": scale_gain_frame(load_scale_exact_gains(scale_10_dir), 10, {0.25: "tight", 0.5: "medium", 0.75: "loose"}),
        "20$\\times$20\n1,200 outcomes\n$n=30$": scale_gain_frame(load_scale_exact_gains(scale_20_dir), 20, {0.25: "tight", 0.5: "medium", 0.75: "loose"}),
    }
    scale_x = np.arange(len(scale_frames), dtype=float)
    level_specs = (
        ("tight", "#7c3aed", "o", -0.16),
        ("medium", "#2563eb", "s", 0.0),
        ("loose", "#0f766e", "^", 0.16),
    )
    for level, color, marker, offset in level_specs:
        means = []
        lows = []
        highs = []
        for frame in scale_frames.values():
            values = frame.loc[frame["level"] == level, "gain_per_route"].to_numpy(float)
            mean, low, high = bootstrap_mean_ci(values, rng)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        means_array = np.asarray(means)
        scale_ax.errorbar(
            scale_x + offset,
            means_array,
            yerr=np.vstack((means_array - np.asarray(lows), np.asarray(highs) - means_array)),
            color=color,
            marker=marker,
            linestyle="none",
            capsize=2.8,
            label=level,
        )
    scale_ax.axhline(0, color="#475569", linewidth=0.9)
    scale_ax.set_xticks(scale_x, list(scale_frames))
    scale_ax.set_ylabel("gain per selected route (m)")
    scale_ax.set_title("(b) Larger assignments")
    scale_ax.legend(frameon=False, ncol=3, loc="upper right")

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.25, top=0.9, wspace=0.28)
    save(fig, output_dir, "fig4_portability")


def load_lattice_exact_gains(summary_dir: Path) -> pd.DataFrame:
    rows = []
    exact_dir = summary_dir.parent / "exact"
    for pool in ("test_1", "test_2", "test_3"):
        results = pd.read_csv(exact_dir / pool / "budget_assignment_results.csv")
        selected = results[results["method"].isin(["portfolio", "global_profile"])].copy()
        pivot = selected.pivot(index=["instance_id", "budget_quantile"], columns="method", values="selected_length")
        gains = (pivot["global_profile"] - pivot["portfolio"]).rename("gain").reset_index()
        gains["pool"] = pool
        rows.append(gains)
    return pd.concat(rows, ignore_index=True)


def load_scale_exact_gains(evaluation_dir: Path) -> pd.DataFrame:
    rows = []
    for pool in ("test_1", "test_2", "test_3"):
        results = pd.read_csv(evaluation_dir / pool / "scale_assignment_results.csv")
        selected = results[
            (results["consumer"] == "sum_length")
            & (results["acquisition"] == "exact")
            & (results["commitment"].isin(["edge_wise", "dispatch_global"]))
        ].copy()
        pivot = selected.pivot(index=["instance_id", "budget_level"], columns="commitment", values="true_sum_length")
        gains = (pivot["dispatch_global"] - pivot["edge_wise"]).rename("gain").reset_index()
        gains["pool"] = pool
        rows.append(gains)
    return pd.concat(rows, ignore_index=True)


def scale_gain_frame(gains: pd.DataFrame, selected_routes: int, levels: dict[float, str]) -> pd.DataFrame:
    budget_column = "budget_quantile" if "budget_quantile" in gains.columns else "budget_level"
    selected = gains[gains[budget_column].isin(levels)].copy()
    selected["level"] = selected[budget_column].map(levels)
    selected["gain_per_route"] = selected["gain"] / float(selected_routes)
    return selected


def make_runtime_accounting(static_runtime_dir: Path, output_dir: Path) -> None:
    summary = pd.read_csv(static_runtime_dir / "static_runtime_serial_summary.csv")
    q50 = summary[np.isclose(summary["budget_quantile"], 0.5)].set_index("method")
    methods = ["exact_portfolio", "fast_only", "dispatch_global", "learned_portfolio"]
    labels = ["plan all\ncandidates", "fixed\nfast", "predict,\none profile", "predict,\nper edge"]
    colors = [EXACT_COLOR, PROFILE_COLORS["fast"], GLOBAL_COLOR, EDGE_COLOR]
    ordered = q50.loc[methods]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.35))
    calls = axes[0].bar(labels, ordered["total_planner_calls_mean"], color=colors, width=0.68)
    axes[0].set_ylabel("measured planner calls")
    axes[0].set_title("(a) Planner calls")
    axes[0].bar_label(calls, labels=[f"{v:.1f}" for v in ordered["total_planner_calls_mean"]], padding=2, fontsize=7.6)

    means = ordered["decision_wall_time_sec_mean"].to_numpy(float)
    yerr = np.vstack(
        [
            means - ordered["decision_wall_time_sec_q25"].to_numpy(float),
            ordered["decision_wall_time_sec_q75"].to_numpy(float) - means,
        ]
    )
    times = axes[1].bar(labels, means, color=colors, width=0.68, yerr=yerr, capsize=3)
    axes[1].set_ylabel("decision wall time (s)")
    axes[1].set_title("(b) End-to-end dispatch timing")
    axes[1].bar_label(times, labels=[f"{v:.1f}" for v in means], padding=2, fontsize=7.6)
    axes[1].annotate(
        "same prediction workload\n+9 ms assignment time",
        xy=(3, q50.loc["learned_portfolio", "decision_wall_time_sec_mean"]),
        xytext=(2.45, 56),
        ha="center",
        fontsize=7.4,
        color=EDGE_COLOR,
        arrowprops={"arrowstyle": "->", "color": EDGE_COLOR, "linewidth": 0.9},
    )
    fig.tight_layout()
    save(fig, output_dir, "fig5_acquisition_economics")


def make_rolling_stress(rolling_dir: Path, exact_dir: Path, output_dir: Path) -> None:
    population = pd.read_csv(rolling_dir / "rolling_commitment_population.csv").set_index("method")
    exact_missions = pd.read_csv(exact_dir / "lazy_rolling_missions.csv").query("method == 'exact_portfolio'")
    learned_missions = pd.read_csv(
        rolling_dir.parent.parent / "rolling" / "lazy_rolling_missions.csv"
    ).query("method == 'learned_portfolio'")
    missions = pd.concat([exact_missions, learned_missions], ignore_index=True)
    order = ["exact_portfolio", "learned_portfolio", "fast_only"]
    labels = ["plan all\ncandidates", "predict,\nper edge", "fixed\nfast"]
    colors = [EXACT_COLOR, EDGE_COLOR, PROFILE_COLORS["fast"]]
    rows = population.loc[order]

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.35))
    complete_values = rows["completion_rate_mean"].to_numpy(float) * 100.0
    method_x = np.arange(len(labels), dtype=float)
    axes[0].scatter(method_x, complete_values, color=colors, s=54, zorder=3)
    axes[0].set_ylim(85, 101)
    axes[0].set_xticks(method_x, labels)
    axes[0].set_ylabel("completed tasks (%)")
    axes[0].set_title("(a) Task completion")
    for xpos, value, color in zip(method_x, complete_values, colors):
        axes[0].annotate(
            f"{value + 1e-9:.1f}%",
            (xpos, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.4,
            color=color,
        )

    call_values = rows["total_planner_calls_mean"].to_numpy(float)
    calls = axes[1].bar(labels, call_values, color=colors, width=0.68)
    axes[1].set_ylabel("planner calls / mission")
    axes[1].set_title("(b) Planner workload")
    axes[1].bar_label(calls, labels=[f"{value:.1f}" for value in call_values], padding=2, fontsize=7.4)

    paired = missions.pivot(
        index="mission_id",
        columns="method",
        values=["completed_tasks", "mission_true_length"],
    )
    common_complete = (
        (paired["completed_tasks"]["exact_portfolio"] == 20)
        & (paired["completed_tasks"]["learned_portfolio"] == 20)
    )
    exact_length = paired["mission_true_length"]["exact_portfolio"][common_complete].to_numpy(float)
    learned_length = paired["mission_true_length"]["learned_portfolio"][common_complete].to_numpy(float)
    if len(exact_length) != 94:
        raise ValueError(f"Expected 94 common-complete rolling missions, found {len(exact_length)}")
    axes[2].scatter(exact_length, learned_length, s=18, color=EDGE_COLOR, alpha=0.68, edgecolors="none")
    lower = float(min(np.min(exact_length), np.min(learned_length)))
    upper = float(max(np.max(exact_length), np.max(learned_length)))
    margin = 0.035 * (upper - lower)
    axes[2].plot([lower - margin, upper + margin], [lower - margin, upper + margin], color="#475569", linestyle="--", linewidth=1.0, label="identity")
    axes[2].set_xlim(lower - margin, upper + margin)
    axes[2].set_ylim(lower - margin, upper + margin)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel("planner-outcome system length (m)")
    axes[2].set_ylabel("predicted-outcome system length (m)")
    axes[2].set_title("(c) Cumulative route length")
    mard = float(np.mean(np.abs(learned_length - exact_length) / exact_length) * 100.0)
    axes[2].text(
        0.04,
        0.96,
        f"completed by both: $n={len(exact_length)}$\nMARD = {mard:.2f}%",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
    )
    axes[2].legend(frameon=False, loc="lower right")

    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.2, top=0.89, wspace=0.34)
    save(fig, output_dir, "fig6_rolling_stress")


if __name__ == "__main__":
    main()
