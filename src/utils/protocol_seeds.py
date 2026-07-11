from __future__ import annotations

import numpy as np


FINAL_MASTER_SEED = 20_260_704

SEED_NAMESPACES = {
    "edge_map": 1,
    "edge_pair": 2,
    "assignment_map": 3,
    "assignment_point": 4,
    "rolling_map": 5,
    "rolling_mission": 6,
    "scale_assignment_point": 7,
}


def derive_seed(namespace: str, *indices: int, master_seed: int = FINAL_MASTER_SEED) -> int:
    if namespace not in SEED_NAMESPACES:
        raise ValueError(f"Unknown seed namespace: {namespace}")
    sequence = np.random.SeedSequence(
        int(master_seed),
        spawn_key=(SEED_NAMESPACES[namespace], *(int(index) for index in indices)),
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])
