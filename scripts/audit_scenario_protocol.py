from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit train/val/assignment-test scenario separation for cost-model experiments."
    )
    parser.add_argument("--train-dataset-dir", type=Path, required=True, help="Dataset used to train/select the model.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Trained model directory.")
    parser.add_argument("--benchmark-dir", type=Path, required=True, help="Cached assignment benchmark directory.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    train_metadata = load_json(args.train_dataset_dir / "metadata.json")
    benchmark_metadata = load_json(args.benchmark_dir / "source_metadata.json")
    benchmark_summary = load_json(args.benchmark_dir / "benchmark_summary.json")

    model_split = load_model_scenario_split(args.model_dir)
    train_hashes = scenario_hashes(train_metadata)
    benchmark_hashes = scenario_hashes(benchmark_metadata)
    assignment_test_ids = sorted(int(item) for item in benchmark_summary.get("scenario_ids", []))
    assignment_test_hashes = {benchmark_hashes[item] for item in assignment_test_ids}

    split_report = {}
    for split_name, split_ids in model_split.items():
        ids = sorted(int(item) for item in split_ids)
        hashes = {train_hashes[item] for item in ids if item in train_hashes}
        split_report[split_name] = {
            "scenario_ids": ids,
            "scenario_count": len(ids),
            "id_overlap_with_assignment_test": sorted(set(ids) & set(assignment_test_ids)),
            "map_hash_overlap_with_assignment_test": sorted(hashes & assignment_test_hashes),
        }

    report = {
        "train_dataset_dir": str(args.train_dataset_dir),
        "model_dir": str(args.model_dir),
        "benchmark_dir": str(args.benchmark_dir),
        "assignment_test": {
            "scenario_ids": assignment_test_ids,
            "scenario_count": len(assignment_test_ids),
            "distribution": benchmark_summary.get("scenario_protocol", {}).get("distribution"),
            "mapping": benchmark_summary.get("scenario_protocol", {}).get("mapping"),
        },
        "model_splits": split_report,
        "clean_assignment_test": all(
            not item["map_hash_overlap_with_assignment_test"] for item in split_report.values()
        ),
    }

    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_scenario_split(model_dir: Path) -> dict[str, list[int]]:
    split_path = model_dir / "scenario_split.json"
    raw = load_json(split_path)
    return {
        split_name: [int(item) for item in split_info.get("scenario_ids", [])]
        for split_name, split_info in raw.items()
    }


def scenario_hashes(metadata: dict) -> dict[int, str]:
    hashes = {}
    for scenario in metadata["scenarios"]:
        scenario_id = int(scenario["scenario_id"])
        physical = {
            "threat_layout": scenario.get("threat_layout", "random"),
            "barrier_orientation": scenario.get("barrier_orientation"),
            "barrier_gap": scenario.get("barrier_gap"),
            "threat_sources": scenario["threat_sources"],
            "circle_obstacles": scenario["circle_obstacles"],
            "rectangle_obstacles": scenario["rectangle_obstacles"],
        }
        canonical = json.dumps(physical, sort_keys=True, separators=(",", ":"))
        hashes[scenario_id] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return hashes


if __name__ == "__main__":
    main()
