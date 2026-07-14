"""Build modular Zenodo archives for the paper data, models, and results."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "final_ras"
DEFAULT_RELEASE = ROOT / "zenodo_release"
POOLS = ("development", "test_1", "test_2", "test_3")
PROFILES = {
    "fast": "beta150",
    "balanced": "beta650/merged",
    "safe": "beta1500",
}
CANONICAL_BENCHMARK_FILES = (
    "benchmark_summary.json",
    "instances.csv",
    "pairs.csv",
    "source_metadata.json",
)
MODEL_DIRS = (
    "profile_onehot_h192",
    "profile_onehot_h476",
    "beta150_specialist",
    "beta650_specialist",
    "beta1500_specialist",
    "beta150_direct_bid",
    "beta650_direct_bid",
    "beta1500_direct_bid",
)


@dataclass(frozen=True)
class ArchiveEntry:
    source: Path
    target: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RELEASE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release = args.output.resolve()
    root = ROOT.resolve()
    if release == root or root not in release.parents:
        raise ValueError("Release directory must be inside the repository.")
    reset_release_directory(release)

    readme = archive_readme()
    license_text = data_license()
    (release / "README.md").write_text(readme, encoding="utf-8")
    (release / "LICENSE_DATA.md").write_text(license_text, encoding="utf-8")
    (release / "zenodo_metadata_draft.json").write_text(
        json.dumps(zenodo_metadata_draft(), indent=2), encoding="utf-8"
    )

    archives = {
        "planner_outcome_data_v1.zip": data_entries(),
        "planner_outcome_models_v1.zip": model_entries(),
        "planner_outcome_results_v1.zip": result_entries(),
    }
    inventory: dict[str, dict[str, int | str]] = {}
    for name, entries in archives.items():
        destination = release / name
        write_archive(destination, entries)
        inventory[name] = {
            "entries": len(entries),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }

    validation = semantic_validation(release)
    (release / "VALIDATION_REPORT.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    for name in ("README.md", "LICENSE_DATA.md", "VALIDATION_REPORT.json"):
        path = release / name
        inventory[name] = {
            "entries": 1,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    (release / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "code_repository": "https://github.com/bkbkd/uav-planner-outcome-interface",
                "code_commit": public_code_commit(),
                "files": inventory,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_checksums(release)
    verify_release(release, archives)
    print(f"Zenodo draft prepared: {release}")


def reset_release_directory(release: Path) -> None:
    if release.exists():
        shutil.rmtree(release)
    release.mkdir(parents=True)


def data_entries() -> list[ArchiveEntry]:
    entries: list[ArchiveEntry] = [
        ArchiveEntry(ROOT / "paper" / "protocol_lock.yaml", Path("protocol/protocol_lock.yaml"))
    ]
    for beta in ("150", "650", "1500"):
        source = OUTPUT_ROOT / f"edges_beta{beta}"
        for name in ("metadata.json", "samples.csv", "summary.json"):
            entries.append(ArchiveEntry(source / name, Path(f"edges/beta{beta}/{name}")))

    entries.extend(benchmark_entries(OUTPUT_ROOT / "assignment", Path("assignment_5x5")))
    entries.extend(
        benchmark_entries(
            OUTPUT_ROOT / "assignment_scale" / "10x10", Path("assignment_10x10")
        )
    )
    entries.extend(
        benchmark_entries(
            OUTPUT_ROOT / "assignment_scale" / "20x20", Path("assignment_20x20")
        )
    )
    entries.extend(tree_entries(OUTPUT_ROOT / "scenario_pools", Path("scenario_pools")))

    grid = ROOT / "outputs" / "planner_transfer" / "grid12"
    entries.extend(tree_entries(grid / "edge_profiles", Path("grid_planner/edge_profiles")))
    for pool in POOLS:
        for profile in PROFILES:
            source = grid / pool / profile
            for name in CANONICAL_BENCHMARK_FILES:
                entries.append(
                    ArchiveEntry(source / name, Path(f"grid_planner/{pool}/{profile}/{name}"))
                )
        protocol = grid / pool / "protocol.json"
        entries.append(ArchiveEntry(protocol, Path(f"grid_planner/{pool}/protocol.json")))
    return validated(entries)


def benchmark_entries(source_root: Path, target_root: Path) -> list[ArchiveEntry]:
    entries: list[ArchiveEntry] = []
    for pool in POOLS:
        for profile, relative in PROFILES.items():
            source = source_root / pool / Path(relative)
            for name in CANONICAL_BENCHMARK_FILES:
                entries.append(
                    ArchiveEntry(source / name, target_root / pool / profile / name)
                )
    return entries


def model_entries() -> list[ArchiveEntry]:
    entries: list[ArchiveEntry] = []
    for directory in MODEL_DIRS:
        entries.extend(
            tree_entries(
                OUTPUT_ROOT / "models" / directory,
                Path("models") / directory,
                exclude_suffixes={".png"},
            )
        )
    return validated(entries)


def result_entries() -> list[ArchiveEntry]:
    entries = tree_entries(
        OUTPUT_ROOT / "results" / "shared_h192",
        Path("results/shared_h192"),
        exclude_names={"rolling_exact_subset_smoke"},
        exclude_suffixes={".log"},
    )
    entries.extend(
        tree_entries(
            OUTPUT_ROOT / "results" / "profile_nondegeneracy",
            Path("results/profile_nondegeneracy"),
            exclude_suffixes={".log"},
        )
    )
    entries.extend(
        tree_entries(
            OUTPUT_ROOT / "results" / "scalar_assignment" / "direct_bid",
            Path("results/direct_bid_assignment"),
            exclude_suffixes={".log"},
        )
    )
    grid = ROOT / "outputs" / "planner_transfer" / "grid12"
    entries.extend(tree_entries(grid / "evaluation", Path("results/grid_planner/evaluation")))
    entries.extend(tree_entries(grid / "summary", Path("results/grid_planner/summary")))
    return validated(entries)


def tree_entries(
    source: Path,
    target: Path,
    *,
    exclude_names: set[str] | None = None,
    exclude_suffixes: set[str] | None = None,
) -> list[ArchiveEntry]:
    excluded_names = exclude_names or set()
    excluded_suffixes = exclude_suffixes or set()
    entries = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part in excluded_names for part in relative.parts):
            continue
        if path.suffix.lower() in excluded_suffixes:
            continue
        if path.name.endswith(".tmp") or ".tmp." in path.name:
            continue
        entries.append(ArchiveEntry(path, target / relative))
    return entries


def validated(entries: list[ArchiveEntry]) -> list[ArchiveEntry]:
    missing = [entry.source for entry in entries if not entry.source.is_file()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Archive inputs are missing:\n{joined}")
    targets = [entry.target.as_posix() for entry in entries]
    if len(targets) != len(set(targets)):
        raise ValueError("Archive target paths are not unique.")
    return entries


def write_archive(destination: Path, entries: list[ArchiveEntry]) -> None:
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for index, entry in enumerate(entries, start=1):
            archive.write(entry.source, entry.target.as_posix())
            if index % 100 == 0 or index == len(entries):
                print(f"{destination.name}: {index}/{len(entries)}")


def public_code_commit() -> str:
    release_repo = ROOT / "public_release"
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=release_repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def archive_readme() -> str:
    return """# Route-Profile Selection Data, Models, and Results

This record accompanies *Delaying Route-Profile Selection in Risk-Budgeted
Multi-UAV Assignment with a Learned Planner Interface*.

## Files

- `planner_outcome_data_v1.zip`: map-disjoint edge supervision, canonical 5x5,
  10x10, and 20x20 assignment benchmarks, grid-planner replication data, and the
  locked experiment protocol.
- `planner_outcome_models_v1.zip`: the main shared profile-conditioned model
  and the specialist, capacity, and direct-bid models used in the paper.
- `planner_outcome_results_v1.zip`: final learned/reference outcomes,
  statistical summaries, runtime records, rolling missions, and external
  validity results used for paper tables and figures.
- `manifest.json` and `SHA256SUMS.txt`: sizes, archive entry counts, code
  revision, and integrity checks.

The source code and command-level documentation are available at:
https://github.com/bkbkd/uav-planner-outcome-interface

Feature caches, worker shards, checkpoints, and logs are excluded because they
are mechanical intermediates that can be reconstructed from the archived data.
No post hoc repair data or unpublished alternative experiment namespace is
included.

## Restoring the expected layout

Extract each ZIP into the same directory. The resulting top-level directories
are `edges`, `assignment_5x5`, `assignment_10x10`, `assignment_20x20`,
`grid_planner`, `models`, `results`, `scenario_pools`, and `protocol`.

Integrity can be checked with any SHA-256 utility against `SHA256SUMS.txt`.
"""


def data_license() -> str:
    return """# Data and model license

The datasets, trained model weights, and derived result files in this Zenodo
record are licensed under the Creative Commons Attribution 4.0 International
license (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/

The separately hosted source code is licensed under the MIT License.
"""


def zenodo_metadata_draft() -> dict[str, object]:
    return {
        "status": "draft-ready",
        "title": "Route-Profile Selection Data, Models, and Results for Risk-Budgeted Multi-UAV Assignment",
        "resource_type": "dataset",
        "publication_date": "2026-07-14",
        "creators": [
            {
                "name": "Xu, Zhouning",
                "affiliation": "National Elite Institute of Engineering, Northwestern Polytechnical University",
            },
            {
                "name": "Wu, Qiupeng",
                "affiliation": "National Elite Institute of Engineering, Northwestern Polytechnical University",
            },
            {
                "name": "Chen, Bolin",
                "affiliation": "School of Computer Science, Northwestern Polytechnical University",
            },
        ],
        "description": (
            "Map-disjoint planner supervision, fleet assignment benchmarks, trained models, "
            "and final results accompanying a study of when route profiles should be selected "
            "in risk-budgeted multi-UAV assignment."
        ),
        "license": "cc-by-4.0",
        "keywords": [
            "multi-robot task allocation",
            "multi-UAV assignment",
            "risk-aware planning",
            "route-profile selection",
            "learned planner interface",
        ],
        "related_identifiers": [
            {
                "identifier": "https://github.com/bkbkd/uav-planner-outcome-interface",
                "relation": "isSupplementedBy",
                "resource_type": "software",
            }
        ],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(release: Path) -> None:
    names = sorted(
        path.name
        for path in release.iterdir()
        if path.is_file() and path.name not in {"SHA256SUMS.txt", "zenodo_metadata_draft.json"}
    )
    lines = [f"{sha256(release / name)}  {name}" for name in names]
    (release / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


def verify_release(release: Path, archives: dict[str, list[ArchiveEntry]]) -> None:
    for name, entries in archives.items():
        with zipfile.ZipFile(release / name) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError(f"Corrupt ZIP member in {name}: {bad}")
            archived = set(archive.namelist())
            expected = {entry.target.as_posix() for entry in entries}
            if archived != expected:
                raise RuntimeError(f"ZIP inventory mismatch: {name}")


def semantic_validation(release: Path) -> dict[str, object]:
    data_archive = release / "planner_outcome_data_v1.zip"
    with zipfile.ZipFile(data_archive) as archive:
        row_counts = {
            "edges_fast": csv_rows(archive, "edges/beta150/samples.csv"),
            "edges_balanced": csv_rows(archive, "edges/beta650/samples.csv"),
            "edges_safe": csv_rows(archive, "edges/beta1500/samples.csv"),
            "assignment_5x5_test1_pairs": csv_rows(
                archive, "assignment_5x5/test_1/fast/pairs.csv"
            ),
            "assignment_10x10_test1_pairs": csv_rows(
                archive, "assignment_10x10/test_1/fast/pairs.csv"
            ),
            "assignment_20x20_test1_pairs": csv_rows(
                archive, "assignment_20x20/test_1/fast/pairs.csv"
            ),
            "grid_test1_pairs": csv_rows(
                archive, "grid_planner/test_1/fast/pairs.csv"
            ),
        }
    expected_counts = {
        "edges_fast": 20_000,
        "edges_balanced": 20_000,
        "edges_safe": 20_000,
        "assignment_5x5_test1_pairs": 12_500,
        "assignment_10x10_test1_pairs": 5_000,
        "assignment_20x20_test1_pairs": 4_000,
        "grid_test1_pairs": 1_250,
    }
    if row_counts != expected_counts:
        raise RuntimeError(f"Unexpected canonical row counts: {row_counts}")

    model_members = {
        "models/profile_onehot_h192/model.pt",
        "models/profile_onehot_h192/scalers.npz",
        "models/profile_onehot_h192/feature_names.json",
        "models/profile_onehot_h192/target_names.json",
    }
    with zipfile.ZipFile(release / "planner_outcome_models_v1.zip") as archive:
        if not model_members.issubset(archive.namelist()):
            raise RuntimeError("Main model inference artifacts are incomplete.")
        with tempfile.TemporaryDirectory(prefix="zenodo-model-check-") as directory:
            member = "models/profile_onehot_h192/model.pt"
            archive.extract(member, directory)
            import torch

            checkpoint = torch.load(Path(directory) / member, map_location="cpu")
            if not checkpoint:
                raise RuntimeError("Main model checkpoint is empty.")

    result_members = {
        "results/shared_h192/profile_commitment/summary/learned_commitment_paired.csv",
        "results/shared_h192/scale_10x10/summary/scale_assignment_pooled_summary.csv",
        "results/shared_h192/scale_20x20/summary/scale_assignment_pooled_summary.csv",
        "results/shared_h192/rolling/lazy_rolling_missions.csv",
        "results/shared_h192/static_runtime/static_runtime_serial_results.csv",
        "results/grid_planner/summary/grid_transfer_commitment_summary.csv",
    }
    with zipfile.ZipFile(release / "planner_outcome_results_v1.zip") as archive:
        if not result_members.issubset(archive.namelist()):
            raise RuntimeError("Key result files are incomplete.")

    scan_archives_for_local_paths(release)
    return {
        "status": "passed",
        "canonical_row_counts": row_counts,
        "main_model_checkpoint_loaded": True,
        "key_result_files_present": sorted(result_members),
        "zip_crc_and_inventory_verified": True,
        "local_path_scan_passed": True,
    }


def csv_rows(archive: zipfile.ZipFile, member: str) -> int:
    with archive.open(member) as raw:
        return sum(1 for _ in io.TextIOWrapper(raw, encoding="utf-8")) - 1


def scan_archives_for_local_paths(release: Path) -> None:
    markers = ("E:\\academic", "C:\\Users")
    text_suffixes = (".json", ".csv", ".txt", ".yaml", ".md")
    for path in release.glob("*.zip"):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if not info.filename.lower().endswith(text_suffixes):
                    continue
                if info.file_size >= 20_000_000:
                    continue
                text = archive.read(info).decode("utf-8", errors="ignore")
                if any(marker in text for marker in markers):
                    raise RuntimeError(f"Local path found in {path.name}:{info.filename}")


if __name__ == "__main__":
    main()
