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

from src.data.dataset_generator import DatasetConfig, ThreatConfig, random_scenario, sample_start_goal_with_mode
from src.planners.lattice_planner import DeterministicLatticePlanner, LatticePlannerConfig


PROFILE_COLORS = {"fast": "#d95f02", "balanced": "#1b9e77", "safe": "#386cb0"}
PROFILE_HATCHES = {"fast": "///", "balanced": "\\\\", "safe": "..."}
EDGE_COLOR = "#7c3aed"
GLOBAL_COLOR = "#64748b"
EXACT_COLOR = "#991b1b"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the current RAS manuscript figures from frozen results.")
    parser.add_argument("--commitment-summary-dir", type=Path, required=True)
    parser.add_argument("--rolling-commitment-dir", type=Path, required=True)
    parser.add_argument("--static-runtime-dir", type=Path, required=True)
    parser.add_argument("--risk-summary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures/ras_interface"))
    parser.add_argument("--route-seed", type=int, default=17)
    parser.add_argument("--route-samples", type=int, default=60)
    parser.add_argument("--make-route-profile-example", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    make_commitment_example(args.commitment_summary_dir, args.output_dir)
    make_commitment_evidence(args.commitment_summary_dir, args.output_dir)
    make_runtime_accounting(args.static_runtime_dir, args.output_dir)
    make_rolling_stress(args.rolling_commitment_dir, args.output_dir)
    make_risk_buffer_frontier(args.risk_summary_dir, args.output_dir)
    if args.make_route_profile_example:
        make_route_profiles(args.output_dir, seed=args.route_seed, samples=args.route_samples)
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


def make_commitment_example(summary_dir: Path, output_dir: Path) -> None:
    example = json.loads((summary_dir / "fleet_example.json").read_text(encoding="utf-8"))
    methods = ["fast_only", "balanced_only", "safe_only", "portfolio"]
    portfolio_modes = example["methods"]["portfolio"]["modes"].split(";")
    mode_counts = {name: portfolio_modes.count(name) for name in ("fast", "balanced", "safe")}
    composition = " + ".join(f"{count} {name}" for name, count in mode_counts.items() if count)
    labels = ["uniform fast", "uniform balanced", "uniform safe", f"edge-wise\n{composition}"]
    colors = [PROFILE_COLORS["fast"], PROFILE_COLORS["balanced"], PROFILE_COLORS["safe"], EDGE_COLOR]
    markers = ["o", "o", "o", "D"]

    fig, ax = plt.subplots(figsize=(5.1, 3.55))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        row = example["methods"][method]
        ax.scatter(row["risk"], row["length"], s=58, marker=marker, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        offset = (6, 5) if method != "safe_only" else (6, -13)
        ax.annotate(label, (row["risk"], row["length"]), xytext=offset, textcoords="offset points", color=color, fontsize=7.5)
    budget = float(example["risk_budget"])
    ax.axvline(budget, color="#111827", linestyle="--", linewidth=1.2, label=f"budget B={budget:.4f}")
    ax.annotate(
        "41.98 m",
        xy=(example["methods"]["portfolio"]["risk"], example["methods"]["portfolio"]["length"]),
        xytext=(example["methods"]["safe_only"]["risk"] - 0.004, example["methods"]["safe_only"]["length"] - 18),
        arrowprops={"arrowstyle": "->", "color": EDGE_COLOR, "linewidth": 1.0},
        color=EDGE_COLOR,
        ha="right",
        fontsize=8.0,
    )
    ax.set_xlabel("total exposure")
    ax.set_ylabel("total route length (m)")
    ax.set_title("One dispatch: uniform versus edge-wise profile commitment")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, output_dir, "fig2_commitment_example")


def make_commitment_evidence(summary_dir: Path, output_dir: Path) -> None:
    exact = pd.read_csv(summary_dir / "exact_commitment_summary.csv")
    exact = exact[exact["method"] == "global_profile"].sort_values("budget_quantile")
    learned = pd.read_csv(summary_dir / "learned_commitment_paired.csv")
    learned = learned[
        (learned["reference_method"] == "dispatch_global_learned")
        & (learned["scope"] == "common_true_nonviolating")
    ].sort_values("budget_quantile")
    regimes = pd.read_csv(summary_dir / "operating_regime_joint_summary.csv")
    regimes = regimes[regimes["budget_quantile"].isin([0.1, 0.5, 0.9])]

    fig = plt.figure(figsize=(10.0, 6.15))
    grid = fig.add_gridspec(2, 6, height_ratios=[0.88, 1.12], hspace=0.38, wspace=0.42)
    exact_ax = fig.add_subplot(grid[0, :3])
    learned_ax = fig.add_subplot(grid[0, 3:])
    q = exact["budget_quantile"].to_numpy(float)
    exact_ax.plot(q, exact["gain_to_k3_mean"], marker="o", color=EXACT_COLOR)
    exact_ax.axhline(0, color="#475569", linewidth=0.9)
    exact_ax.set_xlabel("budget quantile")
    exact_ax.set_ylabel("edge-wise gain (m)")
    exact_ax.set_title("(a) Reference-outcome commitment value")

    ql = learned["budget_quantile"].to_numpy(float)
    mean = learned["gain_mean"].to_numpy(float)
    lo = learned["gain_ci95_low"].to_numpy(float)
    hi = learned["gain_ci95_high"].to_numpy(float)
    learned_ax.plot(ql, mean, marker="o", color=EDGE_COLOR)
    learned_ax.fill_between(ql, lo, hi, color=EDGE_COLOR, alpha=0.15, linewidth=0)
    learned_ax.axhline(0, color="#475569", linewidth=0.9)
    learned_ax.set_xlabel("budget quantile")
    learned_ax.set_ylabel("edge-wise gain (m)")
    learned_ax.set_title("(b) Learned outcomes, same predictions")

    bin_order_x = ["low", "medium", "high"]
    bin_order_y = ["high", "medium", "low"]
    heat_axes = []
    heat_image = None
    vmax = float(regimes["mixing_gain_mean"].max())
    for index, budget in enumerate((0.1, 0.5, 0.9)):
        ax = fig.add_subplot(grid[1, 2 * index : 2 * index + 2])
        heat_axes.append(ax)
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
        ax.set_title(f"({chr(ord('c') + index)}) Reference-outcome regimes, $q={budget:.1f}$")
    assert heat_image is not None
    colorbar_axis = fig.add_axes([0.945, 0.105, 0.012, 0.335])
    colorbar_axis.grid(False)
    colorbar = fig.colorbar(heat_image, cax=colorbar_axis)
    colorbar.set_label("mean edge-wise gain (m)")
    fig.subplots_adjust(left=0.07, right=0.925, bottom=0.09, top=0.95)
    save(fig, output_dir, "fig3_commitment_evidence")


def make_runtime_accounting(static_runtime_dir: Path, output_dir: Path) -> None:
    summary = pd.read_csv(static_runtime_dir / "static_runtime_serial_summary.csv")
    q50 = summary[np.isclose(summary["budget_quantile"], 0.5)].set_index("method")
    methods = ["exact_portfolio", "fast_only", "dispatch_global", "learned_portfolio"]
    labels = ["planner-acquired\nedge-wise", "fixed\nfast", "learned\nglobal", "learned\nedge-wise"]
    colors = [EXACT_COLOR, PROFILE_COLORS["fast"], GLOBAL_COLOR, EDGE_COLOR]
    ordered = q50.loc[methods]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.35))
    calls = axes[0].bar(labels, ordered["total_planner_calls_mean"], color=colors, width=0.68)
    axes[0].set_ylabel("measured planner calls")
    axes[0].set_title("(a) Candidate and selected-route planning")
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
        "same predicted library\n+9 ms allocator solve time",
        xy=(3, q50.loc["learned_portfolio", "decision_wall_time_sec_mean"]),
        xytext=(2.45, 56),
        ha="center",
        fontsize=7.4,
        color=EDGE_COLOR,
        arrowprops={"arrowstyle": "->", "color": EDGE_COLOR, "linewidth": 0.9},
    )
    fig.tight_layout()
    save(fig, output_dir, "fig4_runtime_accounting")


def make_rolling_stress(rolling_dir: Path, output_dir: Path) -> None:
    population = pd.read_csv(rolling_dir / "rolling_commitment_population.csv").set_index("method")
    paired = pd.read_csv(rolling_dir / "rolling_commitment_paired.csv").set_index("comparison")
    modes = pd.read_csv(rolling_dir / "rolling_commitment_modes.csv").set_index("method")
    order = ["exact_portfolio", "fast_only", "dispatch_global", "learned_portfolio"]
    labels = ["planner-acquired\nedge-wise", "fixed\nfast", "learned\nglobal", "learned\nedge-wise"]
    colors = [EXACT_COLOR, PROFILE_COLORS["fast"], GLOBAL_COLOR, EDGE_COLOR]
    rows = population.loc[order]

    fig, axes = plt.subplots(2, 2, figsize=(9.3, 5.8))
    axes = axes.ravel()
    complete = axes[0].bar(labels, rows["completed_tasks_mean"], color=colors, width=0.68)
    axes[0].set_ylim(0, 20.5)
    axes[0].set_ylabel("completed tasks / mission")
    axes[0].set_title("(a) Task completion")
    axes[0].bar_label(complete, labels=[f"{v:.2f}" for v in rows["completed_tasks_mean"]], padding=2, fontsize=7.4)

    comparisons = ["learned_portfolio_vs_dispatch_global", "learned_portfolio_vs_fast_only"]
    comp_labels = ["vs learned global", "vs fixed fast"]
    means = paired.loc[comparisons, "gain_mean"].to_numpy(float)
    low = paired.loc[comparisons, "gain_ci95_low"].to_numpy(float)
    high = paired.loc[comparisons, "gain_ci95_high"].to_numpy(float)
    axes[1].bar(np.arange(2), means, yerr=np.vstack([means - low, high - means]), capsize=4, color=[GLOBAL_COLOR, PROFILE_COLORS["fast"]], width=0.58)
    axes[1].axhline(0, color="#334155", linewidth=1.0)
    axes[1].set_xticks(np.arange(2), comp_labels)
    axes[1].set_ylabel("learned edge-wise gain (m)")
    axes[1].set_title("(b) Common-complete quality")

    calls = axes[2].bar(labels, rows["total_planner_calls_mean"], color=colors, width=0.68)
    axes[2].set_ylabel("planner calls / mission")
    axes[2].set_title("(c) Actual planner workload")
    axes[2].bar_label(calls, labels=[f"{v:.1f}" for v in rows["total_planner_calls_mean"]], padding=2, fontsize=7.4)

    mix_methods = ["exact_portfolio", "dispatch_global", "learned_portfolio"]
    mix_labels = ["planner-acquired edge-wise", "learned global", "learned edge-wise"]
    y = np.arange(3)
    left = np.zeros(3)
    for profile in ("fast", "balanced", "safe"):
        values = modes.loc[mix_methods, f"{profile}_rate"].to_numpy(float)
        axes[3].barh(y, values, left=left, color=PROFILE_COLORS[profile], hatch=PROFILE_HATCHES[profile], edgecolor="white", label=profile)
        left += values
    axes[3].set_yticks(y, mix_labels)
    axes[3].set_xlim(0, 1)
    axes[3].set_xlabel("executed route share")
    axes[3].set_title("(d) Executed profile mix")
    axes[3].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    save(fig, output_dir, "fig5_rolling_stress")


def make_risk_buffer_frontier(summary_dir: Path, output_dir: Path) -> None:
    df = pd.read_csv(summary_dir / "risk_buffer_sensitivity.csv")
    q50 = df[np.isclose(df["budget_quantile"], 0.5)].copy()
    order = ["raw", "q50", "q75", "q90", "q95"]
    q50["buffer"] = pd.Categorical(q50["buffer"], order, ordered=True)
    q50 = q50.sort_values("buffer")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.35))
    axes[0].plot(q50["true_violation_rate"] * 100, q50["true_length_mean"], marker="o", color=EDGE_COLOR)
    for row in q50.itertuples(index=False):
        axes[0].annotate(str(row.buffer), (row.true_violation_rate * 100, row.true_length_mean), xytext=(4, 3), textcoords="offset points")
    axes[0].set_xlabel("true violation rate (%)")
    axes[0].set_ylabel("proposal true length (m)")
    axes[0].set_title("(a) Empirical exposure-buffer frontier")
    axes[1].bar(q50["buffer"].astype(str), q50["predicted_feasible_rate"] * 100, color="#64748b")
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("predicted feasible rate (%)")
    axes[1].set_title("(b) Proposal coverage")
    fig.tight_layout()
    save(fig, output_dir, "figB1_risk_buffer_frontier")


def make_route_profiles(output_dir: Path, *, seed: int, samples: int) -> None:
    scenario, start, goal, results = find_route_profile_example(seed=seed, samples=samples)
    fig, ax = plt.subplots(figsize=(5.8, 5.35))
    image = scenario.threat_field.plot(ax=ax, cmap="inferno", alpha=0.82)
    ax.grid(False)
    scenario.obstacle_map.plot(ax=ax, facecolor="#111827", edgecolor="white", alpha=0.86)
    ax.plot([start[0], goal[0]], [start[1], goal[1]], color="#e5e7eb", lw=1.25, ls="--", label="straight")
    ax.scatter([start[0]], [start[1]], s=58, color="#22c55e", edgecolor="black", linewidth=0.75, zorder=8)
    ax.scatter([goal[0]], [goal[1]], s=82, color="#38bdf8", marker="*", edgecolor="black", linewidth=0.75, zorder=8)
    ax.annotate("start", (start[0], start[1]), xytext=(5, 4), textcoords="offset points", fontsize=7.4, color="white")
    ax.annotate("goal", (goal[0], goal[1]), xytext=(5, 5), textcoords="offset points", fontsize=7.4, color="white")
    metric_rows = []
    for name, result in results:
        metric_rows.append(f"{name:8s}  L={result.metrics['length']:.0f},  R={result.metrics['risk']:.3f}")
        ax.plot(result.path[:, 0], result.path[:, 1], color=PROFILE_COLORS[name], lw=1.65, label=name)
    ax.set_xlim(0, scenario.width)
    ax.set_ylim(0, scenario.height)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", ncol=4, frameon=True, framealpha=0.72, fontsize=6.7)
    ax.text(0.035, 0.035, "\n".join(metric_rows), transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, family="monospace", bbox={"facecolor": "white", "edgecolor": "#475569", "linewidth": 0.55, "alpha": 0.78, "pad": 2.6})
    cbar = fig.colorbar(image, ax=ax, shrink=0.65, pad=0.023, aspect=34)
    cbar.set_label("threat rate", fontsize=8.3)
    fig.tight_layout()
    save(fig, output_dir, "fig6_route_profiles")


def find_route_profile_example(*, seed: int, samples: int):
    config = DatasetConfig(n_samples=samples, n_scenarios=1, start_goal_mode="mixed", mixed_barrier_crossing_prob=0.7, map_seed=seed, sample_seed=seed + 1, threat=ThreatConfig(layout="central_barrier"))
    rng = np.random.default_rng(seed)
    scenario = random_scenario(scenario_id=0, rng=rng, map_config=config.map, threat_config=config.threat, obstacle_config=config.obstacles)
    best = None
    for _ in range(samples):
        sample_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        start, goal, _ = sample_start_goal_with_mode(scenario, rng=sample_rng, mode=config.start_goal_mode, mixed_barrier_crossing_prob=config.mixed_barrier_crossing_prob, min_distance=config.min_start_goal_distance, max_tries=config.max_sample_tries)
        results = []
        for name, beta in (("fast", 150.0), ("balanced", 650.0), ("safe", 1500.0)):
            result = DeterministicLatticePlanner(scenario, LatticePlannerConfig(beta=beta, grid_guidance_risk_weight=beta)).plan(start, goal)
            if result is None:
                results = []
                break
            results.append((name, result))
        if len(results) != 3:
            continue
        lengths = np.asarray([result.metrics["length"] for _, result in results])
        risks = np.asarray([result.metrics["risk"] for _, result in results])
        score = float(np.ptp(lengths) + 500.0 * np.ptp(risks))
        if best is None or score > best[0]:
            best = (score, start, goal, results)
    if best is None:
        raise RuntimeError("Could not find a successful route-profile example.")
    _, start, goal, results = best
    return scenario, start, goal, results


if __name__ == "__main__":
    main()
