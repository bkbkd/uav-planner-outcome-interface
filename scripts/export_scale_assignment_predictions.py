from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from evaluate_learned_route_portfolio_budget import (
    load_mode_predictions,
    validate_same_instances,
)


MODES = ("fast", "balanced", "safe")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export semantic portfolio predictions for a scale benchmark.")
    for mode in MODES:
        parser.add_argument(f"--{mode}", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--risk-buffer-quantile", type=float, default=0.75)
    parser.add_argument("--near-zero-threshold", type=float, default=24.0)
    parser.add_argument("--near-oracle-threshold", type=float, default=120.0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    args = parser.parse_args()

    scenario_cache: dict[int, Any] = {}
    base_image_cache: dict[tuple, Any] = {}
    mode_data = {
        mode: load_mode_predictions(
            benchmark_dir=getattr(args, mode),
            model_dir=args.model,
            batch_size=args.batch_size,
            image_cache_dir=args.image_cache_dir,
            residual_scale=args.residual_scale,
            zero_threshold=args.near_zero_threshold,
            oracle_threshold=args.near_oracle_threshold,
            risk_buffer_quantile=args.risk_buffer_quantile,
            scenario_cache=scenario_cache,
            base_image_cache=base_image_cache,
        )
        for mode in MODES
    }
    validate_same_instances(mode_data)

    args.output.mkdir(parents=True, exist_ok=True)
    mode_data["balanced"]["instances"].to_csv(args.output / "instances.csv", index=False)
    frames = []
    for mode in MODES:
        frame = mode_data[mode]["pairs"][
            [
                "instance_id",
                "scenario_id",
                "agent_id",
                "task_id",
                "true_length",
                "true_risk",
                "pred_length",
                "pred_risk",
            ]
        ].copy()
        frame.insert(4, "mode", mode)
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(args.output / "portfolio_predictions.csv", index=False)
    summary = {
        "rows": int(len(predictions)),
        "instances": int(len(mode_data["balanced"]["instances"])),
        "modes": list(MODES),
        "risk_buffer_quantile": float(args.risk_buffer_quantile),
        "near_edge_policy": {
            "zero_threshold": float(args.near_zero_threshold),
            "oracle_threshold": float(args.near_oracle_threshold),
        },
        "model": str(args.model),
        "mode_summaries": {mode: mode_data[mode]["model_summary"] for mode in MODES},
    }
    (args.output / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
