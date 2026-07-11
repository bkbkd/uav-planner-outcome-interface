from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.protocol_seeds import FINAL_MASTER_SEED, derive_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the locked natural-map pools for final fleet experiments.")
    parser.add_argument("--output", type=Path, default=Path("outputs/final_ras/scenario_pools"))
    parser.add_argument("--master-seed", type=int, default=FINAL_MASTER_SEED)
    args = parser.parse_args()

    specs = []
    for pool_index, name in enumerate(("development", "test_1", "test_2", "test_3")):
        specs.append(
            {
                "consumer": "assignment",
                "name": name,
                "scenarios": 500,
                "map_seed": derive_seed("assignment_map", pool_index, master_seed=args.master_seed),
                "point_seed": derive_seed("assignment_point", pool_index, master_seed=args.master_seed),
            }
        )
    specs.append(
        {
            "consumer": "rolling",
            "name": "rolling_test",
            "scenarios": 100,
            "map_seed": derive_seed("rolling_map", 0, master_seed=args.master_seed),
            "mission_seed": derive_seed("rolling_mission", 0, master_seed=args.master_seed),
        }
    )

    args.output.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        output = args.output / spec["name"]
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_scenario_pool.py"),
            "--output",
            str(output),
            "--scenarios",
            str(spec["scenarios"]),
            "--map-seed",
            str(spec["map_seed"]),
            "--threat-layout",
            "random",
        ]
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    manifest = {
        "master_seed": int(args.master_seed),
        "derivation": "numpy.SeedSequence(master_seed, spawn_key=(namespace_id, pool_index))",
        "pools": specs,
    }
    (args.output / "seed_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
