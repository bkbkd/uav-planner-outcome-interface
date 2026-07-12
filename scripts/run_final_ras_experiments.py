from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROFILES = {
    "fast": {"beta": "150", "dataset": "edges_beta150", "assignment": "beta150", "model": "beta150_specialist", "direct": "beta150_direct_bid"},
    "balanced": {"beta": "650", "dataset": "edges_beta650", "assignment": "beta650/merged", "model": "beta650_specialist", "direct": "beta650_direct_bid"},
    "safe": {"beta": "1500", "dataset": "edges_beta1500", "assignment": "beta1500", "model": "beta1500_specialist", "direct": "beta1500_direct_bid"},
}
TEST_POOLS = ("test_1", "test_2", "test_3")
ALL_POOLS = ("development", *TEST_POOLS)
BUDGET_QUANTILES = ("0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90")
BUFFER_LABELS = ("raw", "q50", "q75", "q90", "q95")
MAIN_MODEL = "profile_onehot_h192"
MAIN_RESULT_NAMESPACE = "shared_h192"
STATIC_RUNTIME_INSTANCES_PER_POOL = 10
STATIC_INSTANCES_PER_POOL = 500
ROLLING_MISSIONS = 100
ROLLING_SERIAL_MISSIONS = 30


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    complete: Callable[[], bool]


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-safe final RAS experiment runner.")
    parser.add_argument("--root", type=Path, default=Path("outputs/final_ras"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    log_dir = root / "experiment_logs" / MAIN_RESULT_NAMESPACE
    log_dir.mkdir(parents=True, exist_ok=True)
    event_log = log_dir / "events.jsonl"
    stages = build_stages(root, max(1, int(args.workers)), int(args.bootstrap))

    for index, stage in enumerate(stages):
        if stage.complete():
            write_event(event_log, index, len(stages), stage.name, "skipped_complete")
            continue
        write_event(event_log, index, len(stages), stage.name, "running")
        log_path = log_dir / f"{index:02d}_{stage.name}.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log.write(" ".join(stage.command) + "\n")
            log.flush()
            result = subprocess.run(
                stage.command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            write_event(event_log, index, len(stages), stage.name, "failed", result.returncode)
            raise SystemExit(f"Stage failed: {stage.name}; see {log_path}")
        if not stage.complete():
            write_event(event_log, index, len(stages), stage.name, "incomplete_after_exit")
            raise SystemExit(f"Stage exited without complete artifacts: {stage.name}; see {log_path}")
        write_event(event_log, index, len(stages), stage.name, "completed")

    write_event(event_log, len(stages), len(stages), "all", "completed")
    print(f"all final RAS experiments completed: {root}")


def build_stages(root: Path, workers: int, bootstrap: int) -> list[Stage]:
    py = sys.executable
    models = root / "models"
    results = root / "results"
    main_results = results / MAIN_RESULT_NAMESPACE
    cache = root / "cache" / "corridor_images"
    stages: list[Stage] = []

    for profile, spec in PROFILES.items():
        model_dir = models / spec["model"]
        stages.append(
            Stage(
                f"train_{profile}_semantic",
                train_cnn_command(py, root / spec["dataset"], model_dir, ["length", "risk"], cache),
                lambda model_dir=model_dir: model_complete(model_dir, ["length", "risk"]),
            )
        )
    for profile, spec in PROFILES.items():
        model_dir = models / spec["direct"]
        stages.append(
            Stage(
                f"train_{profile}_direct_bid",
                train_cnn_command(py, root / spec["dataset"], model_dir, ["bid"], cache),
                lambda model_dir=model_dir: model_complete(model_dir, ["bid"]),
            )
        )

    onehot_h192 = models / "profile_onehot_h192"
    onehot_h476 = models / "profile_onehot_h476"
    stages.append(
        Stage(
            "train_profile_onehot_h192",
            onehot_command(py, root, onehot_h192, 192, cache),
            lambda: model_complete(onehot_h192, ["length", "risk"]),
        )
    )
    stages.append(
        Stage(
            "train_profile_onehot_h476",
            onehot_command(py, root, onehot_h476, 476, cache),
            lambda: model_complete(onehot_h476, ["length", "risk"]),
        )
    )

    for pool in ALL_POOLS:
        exact_dir = results / "static_exact" / pool
        stages.append(
            Stage(
                f"exact_static_{pool}",
                [
                    py,
                    "scripts/evaluate_route_portfolio_budget.py",
                    "--fast",
                    str(assignment_dir(root, pool, "fast")),
                    "--balanced",
                    str(assignment_dir(root, pool, "balanced")),
                    "--safe",
                    str(assignment_dir(root, pool, "safe")),
                    "--output",
                    str(exact_dir),
                ],
                lambda exact_dir=exact_dir: (exact_dir / "budget_assignment_results.csv").exists(),
            )
        )
        learned_dir = main_results / "static_learned_q75" / pool
        stages.append(
            Stage(
                f"learned_static_q75_{pool}",
                learned_static_command(py, root, pool, learned_dir, cache, risk_buffer="0.75"),
                lambda learned_dir=learned_dir: (learned_dir / "learned_budget_assignment_results.csv").exists(),
            )
        )

        commitment_exact_dir = main_results / "profile_commitment" / "exact" / pool
        stages.append(
            Stage(
                f"profile_commitment_exact_{pool}",
                [
                    py,
                    "scripts/evaluate_route_portfolio_budget.py",
                    "--fast",
                    str(assignment_dir(root, pool, "fast")),
                    "--balanced",
                    str(assignment_dir(root, pool, "balanced")),
                    "--safe",
                    str(assignment_dir(root, pool, "safe")),
                    "--output",
                    str(commitment_exact_dir),
                ],
                lambda commitment_exact_dir=commitment_exact_dir: commitment_result_complete(
                    commitment_exact_dir / "budget_assignment_results.csv",
                    exact=True,
                ),
            )
        )
        commitment_learned_dir = main_results / "profile_commitment" / "learned_q75" / pool
        stages.append(
            Stage(
                f"profile_commitment_learned_{pool}",
                learned_static_command(py, root, pool, commitment_learned_dir, cache, risk_buffer="0.75"),
                lambda commitment_learned_dir=commitment_learned_dir: commitment_result_complete(
                    commitment_learned_dir / "learned_budget_assignment_results.csv",
                    exact=False,
                ),
            )
        )
        sweep_root = main_results / "buffer_sweep"
        stages.append(
            Stage(
                f"buffer_sweep_{pool}",
                [
                    py,
                    "scripts/evaluate_learned_route_portfolio_buffer_sweep.py",
                    *learned_static_common_args(root, pool),
                    "--output-root",
                    str(sweep_root),
                    "--output-prefix",
                    pool,
                    "--image-cache-dir",
                    str(cache),
                    "--risk-buffer-quantiles",
                    "raw",
                    "0.50",
                    "0.75",
                    "0.90",
                    "0.95",
                ],
                lambda sweep_root=sweep_root, pool=pool: all(
                    (sweep_root / f"{pool}_{label}" / "learned_budget_assignment_results.csv").exists()
                    for label in BUFFER_LABELS
                ),
            )
        )

    for pool in TEST_POOLS:
        for profile, spec in PROFILES.items():
            semantic_dir = main_results / "scalar_assignment" / profile / pool
            direct_dir = results / "scalar_assignment" / "direct_bid" / profile / pool
            stages.append(
                Stage(
                    f"scalar_assignment_semantic_{profile}_{pool}",
                    [
                        py,
                        "scripts/evaluate_assignment_benchmark.py",
                        str(assignment_dir(root, pool, profile)),
                        str(models / MAIN_MODEL),
                        "--output",
                        str(semantic_dir),
                        "--image-cache-dir",
                        str(cache),
                    ],
                    lambda semantic_dir=semantic_dir: (semantic_dir / "assignment_results.csv").exists(),
                )
            )
            stages.append(
                Stage(
                    f"scalar_assignment_direct_{profile}_{pool}",
                    [
                        py,
                        "scripts/evaluate_assignment_benchmark.py",
                        str(assignment_dir(root, pool, profile)),
                        str(models / spec["direct"]),
                        "--output",
                        str(direct_dir),
                        "--image-cache-dir",
                        str(cache),
                    ],
                    lambda direct_dir=direct_dir: (direct_dir / "assignment_results.csv").exists(),
                )
            )

    for pool in TEST_POOLS:
        diag_dir = results / "profile_nondegeneracy" / pool
        stages.append(
            Stage(
                f"profile_nondegeneracy_{pool}",
                [
                    py,
                    "scripts/diagnose_profile_nondegeneracy.py",
                    "--fast",
                    str(assignment_dir(root, pool, "fast")),
                    "--balanced",
                    str(assignment_dir(root, pool, "balanced")),
                    "--safe",
                    str(assignment_dir(root, pool, "safe")),
                    "--output",
                    str(diag_dir),
                ],
                lambda diag_dir=diag_dir: (diag_dir / "profile_nondegeneracy_summary.json").exists(),
            )
        )

    runtime_dir = main_results / "static_runtime"
    runtime_command = [
        py,
        "scripts/profile_static_runtime_serial.py",
        *sum(
            (
                [
                    "--benchmark-set",
                    pool,
                    str(assignment_dir(root, pool, "fast")),
                    str(assignment_dir(root, pool, "balanced")),
                    str(assignment_dir(root, pool, "safe")),
                ]
                for pool in TEST_POOLS
            ),
            [],
        ),
        "--fast-model",
        str(models / MAIN_MODEL),
        "--balanced-model",
        str(models / MAIN_MODEL),
        "--safe-model",
        str(models / MAIN_MODEL),
        "--output",
        str(runtime_dir),
        "--instances-per-pool",
        str(STATIC_RUNTIME_INSTANCES_PER_POOL),
        "--budget-quantiles",
        "0.50",
        "--methods",
        "exact_portfolio",
        "learned_portfolio",
        "dispatch_global",
        "fast_only",
        "balanced_only",
        "safe_only",
        "--resume",
    ]
    stages.append(
        Stage(
            "static_runtime_serial",
            runtime_command,
            lambda: runtime_complete(
                runtime_dir / "static_runtime_serial_results.csv",
                STATIC_RUNTIME_INSTANCES_PER_POOL,
            ),
        )
    )

    rolling_dir = main_results / "rolling"
    stages.append(
        Stage(
            "rolling_stress",
            rolling_command(
                py, root, rolling_dir, ROLLING_MISSIONS,
                ("learned_portfolio", "fast_only", "balanced_only", "safe_only"), workers,
            ),
            lambda: rolling_complete(rolling_dir / "lazy_rolling_missions.csv", ROLLING_MISSIONS),
        )
    )

    commitment_rolling_dir = main_results / "profile_commitment" / "rolling_global"
    stages.append(
        Stage(
            "rolling_dispatch_global",
            rolling_command(py, root, commitment_rolling_dir, ROLLING_MISSIONS, ("dispatch_global",), workers),
            lambda: rolling_complete(
                commitment_rolling_dir / "lazy_rolling_missions.csv",
                ROLLING_MISSIONS,
                ("dispatch_global",),
            ),
        )
    )

    rolling_exact_reference_dir = main_results / "rolling_exact_reference"
    stages.append(
        Stage(
            "rolling_exact_reference",
            rolling_command(
                py, root, rolling_exact_reference_dir, ROLLING_MISSIONS,
                ("exact_portfolio", "learned_portfolio"), workers,
            ),
            lambda: rolling_complete(
                rolling_exact_reference_dir / "lazy_rolling_missions.csv",
                ROLLING_MISSIONS,
                ("exact_portfolio", "learned_portfolio"),
            ),
        )
    )

    rolling_runtime_dir = main_results / "rolling_runtime_serial"
    stages.append(
        Stage(
            "rolling_runtime_serial",
            rolling_command(
                py, root, rolling_runtime_dir, ROLLING_SERIAL_MISSIONS,
                ("learned_portfolio", "fast_only", "balanced_only", "safe_only"), 1,
            ),
            lambda: rolling_complete(
                rolling_runtime_dir / "lazy_rolling_missions.csv",
                ROLLING_SERIAL_MISSIONS,
            ),
        )
    )

    commitment_rolling_runtime_dir = main_results / "profile_commitment" / "rolling_global_runtime_serial"
    stages.append(
        Stage(
            "rolling_dispatch_global_runtime_serial",
            rolling_command(
                py, root, commitment_rolling_runtime_dir, ROLLING_SERIAL_MISSIONS,
                ("dispatch_global",), 1,
            ),
            lambda: rolling_complete(
                commitment_rolling_runtime_dir / "lazy_rolling_missions.csv",
                ROLLING_SERIAL_MISSIONS,
                ("dispatch_global",),
            ),
        )
    )

    stages.extend(summary_stages(py, root, results, main_results, bootstrap))
    return stages


def summary_stages(
    py: str,
    root: Path,
    results: Path,
    main_results: Path,
    bootstrap: int,
) -> list[Stage]:
    stages: list[Stage] = []
    commitment_root = main_results / "profile_commitment"
    commitment_summary_dir = commitment_root / "summary"
    benchmark_args = sum(
        (
            [
                "--benchmark-set",
                pool,
                str(assignment_dir(root, pool, "fast")),
                str(assignment_dir(root, pool, "balanced")),
                str(assignment_dir(root, pool, "safe")),
            ]
            for pool in ALL_POOLS
        ),
        [],
    )
    stages.append(
        Stage(
            "summarize_profile_commitment",
            [
                py,
                "scripts/summarize_profile_commitment.py",
                *sum(
                    (
                        ["--exact-result", pool, str(commitment_root / "exact" / pool)]
                        for pool in ALL_POOLS
                    ),
                    [],
                ),
                *sum(
                    (
                        ["--learned-result", pool, str(commitment_root / "learned_q75" / pool)]
                        for pool in ALL_POOLS
                    ),
                    [],
                ),
                *benchmark_args,
                "--development-pool",
                "development",
                "--output-dir",
                str(commitment_summary_dir),
                "--bootstrap",
                str(bootstrap),
                "--seed",
                "0",
            ],
            lambda: (commitment_summary_dir / "profile_commitment_summary.json").exists(),
        )
    )
    commitment_audit = commitment_root / "audit.json"
    stages.append(
        Stage(
            "audit_profile_commitment",
            [
                py,
                "scripts/audit_profile_commitment_results.py",
                *sum(
                    (
                        [
                            "--result-set",
                            pool,
                            str(commitment_root / "exact" / pool),
                            str(commitment_root / "learned_q75" / pool),
                        ]
                        for pool in ALL_POOLS
                    ),
                    [],
                ),
                "--output",
                str(commitment_audit),
            ],
            lambda: commitment_audit.exists()
            and json.loads(commitment_audit.read_text(encoding="utf-8")).get("passed") is True,
        )
    )
    rolling_commitment_summary_dir = commitment_root / "rolling_summary"
    stages.append(
        Stage(
            "summarize_profile_commitment_rolling",
            [
                py,
                "scripts/summarize_profile_commitment_rolling.py",
                "--learned-dir",
                str(main_results / "rolling"),
                "--global-dir",
                str(commitment_root / "rolling_global"),
                "--exact-dir",
                str(main_results / "rolling_exact_reference"),
                "--runtime-dir",
                str(main_results / "rolling_runtime_serial"),
                "--global-runtime-dir",
                str(commitment_root / "rolling_global_runtime_serial"),
                "--output-dir",
                str(rolling_commitment_summary_dir),
                "--bootstrap",
                str(bootstrap),
                "--seed",
                "0",
            ],
            lambda: (rolling_commitment_summary_dir / "rolling_commitment_summary.json").exists(),
        )
    )

    single_dir = main_results / "single_profile_policy"
    stages.append(
        Stage(
            "summarize_single_profile_policy",
            [
                py,
                "scripts/summarize_single_profile_policy.py",
                "--development-result",
                str(main_results / "static_learned_q75" / "development"),
                *sum(
                    (
                        ["--test-result", pool, str(main_results / "static_learned_q75" / pool)]
                        for pool in TEST_POOLS
                    ),
                    [],
                ),
                "--output-dir",
                str(single_dir),
            ],
            lambda: (single_dir / "single_profile_policy_summary.json").exists(),
        )
    )

    static_summary_dir = main_results / "static_summary"
    buffer_args: list[str] = []
    for pool in TEST_POOLS:
        for label in BUFFER_LABELS:
            buffer_args.extend(
                ["--buffer-result", pool, label, str(main_results / "buffer_sweep" / f"{pool}_{label}")]
            )
    stages.append(
        Stage(
            "summarize_static_portfolio",
            [
                py,
                "scripts/summarize_static_portfolio_eval.py",
                *sum((["--exact-result", pool, str(results / "static_exact" / pool)] for pool in TEST_POOLS), []),
                *sum(
                    (["--learned-result", pool, str(main_results / "static_learned_q75" / pool)] for pool in TEST_POOLS),
                    [],
                ),
                *buffer_args,
                "--output-dir",
                str(static_summary_dir),
            ],
            lambda: (static_summary_dir / "learned_common_quality.csv").exists(),
        )
    )

    bid_reuse_dir = main_results / "bid_reuse"
    edge_args: list[str] = []
    assignment_args: list[str] = []
    for profile, spec in PROFILES.items():
        edge_args.extend(
            [
                "--edge-result",
                profile,
                spec["beta"],
                str(root / "models" / MAIN_MODEL),
                str(root / "models" / spec["direct"]),
            ]
        )
        for pool in TEST_POOLS:
            assignment_args.extend(
                [
                    "--assignment-result",
                    profile,
                    pool,
                    str(main_results / "scalar_assignment" / profile / pool),
                    str(results / "scalar_assignment" / "direct_bid" / profile / pool),
                ]
            )
    stages.append(
        Stage(
            "summarize_bid_reuse",
            [
                py,
                "scripts/summarize_bid_reuse_eval.py",
                *edge_args,
                *assignment_args,
                "--output-dir",
                str(bid_reuse_dir),
            ],
            lambda: (bid_reuse_dir / "bid_reuse_assignment_summary.csv").exists(),
        )
    )

    rolling_dir = main_results / "rolling"
    stages.append(
        Stage(
            "summarize_rolling",
            [py, "scripts/summarize_rolling_dispatch_eval.py", "--rolling-dir", str(rolling_dir)],
            lambda: (rolling_dir / "rolling_paired_gains.csv").exists(),
        )
    )

    stats_dir = main_results / "stats"
    stages.append(
        Stage(
            "summarize_uncertainty",
            [
                py,
                "scripts/summarize_ras_statistical_uncertainty.py",
                *sum(
                    (["--static-result", pool, str(main_results / "static_learned_q75" / pool)] for pool in TEST_POOLS),
                    [],
                ),
                "--rolling-dir",
                str(rolling_dir),
                "--output-dir",
                str(stats_dir),
                "--bootstrap",
                str(bootstrap),
                "--seed",
                "0",
            ],
            lambda: (stats_dir / "static_paired_bootstrap_summary.csv").exists()
            and (stats_dir / "rolling_paired_bootstrap_summary.csv").exists(),
        )
    )

    figures_dir = root / "paper_figures" / MAIN_RESULT_NAMESPACE
    stages.append(
        Stage(
            "make_figures",
            [
                py,
                "scripts/make_ras_paper_figures_polished.py",
                "--commitment-summary-dir",
                str(main_results / "profile_commitment" / "summary"),
                "--rolling-commitment-dir",
                str(main_results / "profile_commitment" / "rolling_summary"),
                "--static-runtime-dir",
                str(main_results / "static_runtime"),
                "--output-dir",
                str(figures_dir),
            ],
            lambda: all(
                (figures_dir / name).exists()
                for name in (
                    "fig2_commitment_example.png",
                    "fig3_commitment_evidence.png",
                    "fig4_portability.png",
                    "fig5_acquisition_economics.png",
                    "fig6_rolling_stress.png",
                    "figB1_operating_regimes.png",
                )
            ),
        )
    )
    return stages


def train_cnn_command(py: str, dataset: Path, output: Path, targets: list[str], cache: Path) -> list[str]:
    return [
        py,
        "scripts/train_bid_cnn.py",
        str(dataset),
        "--output",
        str(output),
        "--epochs",
        "300",
        "--batch-size",
        "32",
        "--lr",
        "1e-3",
        "--weight-decay",
        "1e-4",
        "--target-mode",
        "residual",
        "--patience",
        "50",
        "--seed",
        "0",
        "--image-size",
        "48",
        "--lateral-width",
        "320.0",
        "--corridor-scale-mode",
        "square_edge",
        "--image-cache-dir",
        str(cache),
        "--image-pool",
        "grid",
        "--image-encoder",
        "simple",
        "--fusion",
        "concat",
        "--hidden-dim",
        "192",
        "--dropout",
        "0.05",
        "--targets",
        *targets,
        "--loss",
        "mae",
    ]


def onehot_command(py: str, root: Path, output: Path, hidden_dim: int, cache: Path) -> list[str]:
    return [
        py,
        "scripts/train_profile_onehot_cnn.py",
        "--profile",
        "150",
        str(root / "edges_beta150"),
        "--profile",
        "650",
        str(root / "edges_beta650"),
        "--profile",
        "1500",
        str(root / "edges_beta1500"),
        "--output",
        str(output),
        "--epochs",
        "300",
        "--batch-size",
        "32",
        "--lr",
        "1e-3",
        "--weight-decay",
        "1e-4",
        "--patience",
        "50",
        "--seed",
        "0",
        "--image-size",
        "48",
        "--image-cache-dir",
        str(cache),
        "--hidden-dim",
        str(hidden_dim),
        "--dropout",
        "0.05",
    ]


def learned_static_command(py: str, root: Path, pool: str, output: Path, cache: Path, *, risk_buffer: str | None) -> list[str]:
    command = [
        py,
        "scripts/evaluate_learned_route_portfolio_budget.py",
        *learned_static_common_args(root, pool),
        "--output",
        str(output),
        "--image-cache-dir",
        str(cache),
    ]
    if risk_buffer is not None:
        command.extend(["--risk-buffer-quantile", risk_buffer])
    return command


def learned_static_common_args(root: Path, pool: str) -> list[str]:
    models = root / "models"
    return [
        "--fast",
        str(assignment_dir(root, pool, "fast")),
        "--fast-model",
        str(models / MAIN_MODEL),
        "--balanced",
        str(assignment_dir(root, pool, "balanced")),
        "--balanced-model",
        str(models / MAIN_MODEL),
        "--safe",
        str(assignment_dir(root, pool, "safe")),
        "--safe-model",
        str(models / MAIN_MODEL),
    ]


def rolling_command(
    py: str,
    root: Path,
    output: Path,
    missions: int,
    methods: tuple[str, ...],
    workers: int,
) -> list[str]:
    model = root / "models" / MAIN_MODEL
    return [
        py,
        "scripts/evaluate_lazy_rolling_stress.py",
        "--scenario-pool",
        str(root / "scenario_pools" / "rolling_test"),
        "--fast-model",
        str(model),
        "--balanced-model",
        str(model),
        "--safe-model",
        str(model),
        "--output",
        str(output),
        "--missions",
        str(missions),
        "--agents",
        "5",
        "--tasks",
        "20",
        "--mission-seed",
        "3450876602",
        "--wave-size",
        "5",
        "--arrival-interval",
        "5.0",
        "--minimum-mean-route-survivals",
        "0.50",
        "--methods",
        *methods,
        "--risk-buffer-quantile",
        "0.75",
        "--workers",
        str(workers),
        "--resume",
    ]


def assignment_dir(root: Path, pool: str, profile: str) -> Path:
    spec = PROFILES[profile]
    return root / "assignment" / pool / spec["assignment"]


def model_complete(path: Path, targets: list[str]) -> bool:
    target_path = path / "target_names.json"
    if not (path / "model.pt").exists() or not (path / "predictions.csv").exists() or not target_path.exists():
        return False
    try:
        observed = json.loads(target_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return list(observed) == list(targets)


def runtime_complete(path: Path, instances_per_pool: int) -> bool:
    if not path.exists() or not path.with_name("static_runtime_serial_summary.csv").exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return False
    expected = {
        (pool, instance_id, 0.5, method)
        for pool in TEST_POOLS
        for instance_id in range(instances_per_pool)
        for method in ("exact_portfolio", "learned_portfolio", "dispatch_global", "fast_only", "balanced_only", "safe_only")
    }
    observed = {
        (str(row["pool"]), int(row["instance_id"]), float(row["budget_quantile"]), str(row["method"]))
        for row in rows
    }
    return observed == expected


def commitment_result_complete(path: Path, *, exact: bool) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return False
    expected_methods = {
        "portfolio",
        "global_profile",
        "portfolio_fast_balanced",
        "portfolio_fast_safe",
        "portfolio_balanced_safe",
        "fast_only",
        "balanced_only",
        "safe_only",
    }
    if not exact:
        expected_methods = {
            "oracle_true_portfolio",
            "learned_portfolio",
            "dispatch_global_learned",
            "learned_portfolio_fast_balanced",
            "learned_portfolio_fast_safe",
            "learned_portfolio_balanced_safe",
            "fast_only_learned",
            "balanced_only_learned",
            "safe_only_learned",
        }
    observed_methods = {str(row.get("method", "")) for row in rows}
    observed_quantiles = {float(row["budget_quantile"]) for row in rows}
    group_counts: dict[tuple[str, float], int] = {}
    for row in rows:
        key = (str(row.get("method", "")), float(row["budget_quantile"]))
        group_counts[key] = group_counts.get(key, 0) + 1
    return (
        observed_methods == expected_methods
        and observed_quantiles == {float(q) for q in BUDGET_QUANTILES}
        and set(group_counts.values()) == {STATIC_INSTANCES_PER_POOL}
    )


def rolling_complete(
    path: Path,
    missions: int,
    methods: tuple[str, ...] = ("learned_portfolio", "fast_only", "balanced_only", "safe_only"),
) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return False
    expected = {
        (mission_id, method, 0.5)
        for mission_id in range(missions)
        for method in methods
    }
    observed = {
        (int(row["mission_id"]), str(row["method"]), float(row["minimum_mean_route_survival"]))
        for row in rows
    }
    return observed == expected


def write_event(path: Path, index: int, count: int, stage: str, state: str, returncode: int | None = None) -> None:
    payload = {
        "updated_at_unix": time.time(),
        "stage_index": int(index),
        "stage_count": int(count),
        "stage": stage,
        "state": state,
        "returncode": returncode,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
