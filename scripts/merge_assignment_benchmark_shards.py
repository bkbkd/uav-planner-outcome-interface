from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge cached assignment benchmark shards.")
    parser.add_argument("shards", type=Path, nargs="+", help="Shard directories with instances.csv, pairs.csv, and benchmark_summary.json.")
    parser.add_argument("--output", type=Path, required=True, help="Merged benchmark directory.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    instances, pairs, source_metadata, summary = merge_shards(args.shards)
    write_rows(args.output / "instances.csv", instances)
    write_rows(args.output / "pairs.csv", pairs)
    (args.output / "source_metadata.json").write_text(json.dumps(source_metadata, indent=2), encoding="utf-8")
    (args.output / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"merged {len(args.shards)} shards")
    print(f"instances: {len(instances)}")
    print(f"pairs: {len(pairs)}")
    print(f"output: {args.output}")


def merge_shards(shards: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict, dict]:
    merged_instances: list[dict[str, Any]] = []
    merged_pairs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    source_metadata: dict | None = None
    instance_offset = 0

    for shard in shards:
        instances_path = shard / "instances.csv"
        pairs_path = shard / "pairs.csv"
        summary_path = shard / "benchmark_summary.json"
        metadata_path = shard / "source_metadata.json"
        if not instances_path.exists() or not pairs_path.exists() or not summary_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Shard is missing benchmark files: {shard}")

        shard_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if source_metadata is None:
            source_metadata = shard_metadata
        elif shard_metadata != source_metadata:
            raise ValueError(f"Shard source metadata does not match first shard: {shard}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(summary)
        instances_df = pd.read_csv(instances_path)
        pairs_df = pd.read_csv(pairs_path)
        if instances_df["instance_id"].duplicated().any():
            raise ValueError(f"Duplicate instance_id inside shard: {shard}")

        instance_map = {
            int(instance_id): instance_offset + idx
            for idx, instance_id in enumerate(instances_df["instance_id"].astype(int).tolist())
        }

        for record in instances_df.to_dict("records"):
            record = normalize_record(record)
            record["instance_id"] = instance_map[int(record["instance_id"])]
            merged_instances.append(record)

        for record in pairs_df.to_dict("records"):
            record = normalize_record(record)
            record["instance_id"] = instance_map[int(record["instance_id"])]
            merged_pairs.append(record)

        instance_offset += len(instances_df)

    if source_metadata is None:
        raise ValueError("No shards provided.")
    summary = build_summary(shards, summaries, merged_instances, merged_pairs)
    return merged_instances, merged_pairs, source_metadata, summary


def build_summary(shards: list[Path], summaries: list[dict[str, Any]], instances: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    first = deepcopy(summaries[0])
    scenario_ids = sorted({int(item) for summary in summaries for item in summary.get("scenario_ids", [])})
    attempted_pair_sets = int(sum(int(summary.get("attempted_pair_sets", 0)) for summary in summaries))
    first.update(
        {
            "instances": len(instances),
            "pairs": len(pairs),
            "attempted_pair_sets": attempted_pair_sets,
            "scenario_ids": scenario_ids,
            "merged_from": [str(path) for path in shards],
            "point_sampling_attempt": distribution([float(row["point_sampling_attempt"]) for row in instances]),
            "true_planner_runtime_sec": distribution([float(row["true_planner_runtime_sec"]) for row in instances]),
        }
    )
    if "scenario_protocol" in first:
        first["scenario_protocol"]["assignment_test_scenario_ids"] = scenario_ids
    return first


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in record.items():
        if pd.isna(value):
            normalized[key] = ""
        elif isinstance(value, np.generic):
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


if __name__ == "__main__":
    main()
