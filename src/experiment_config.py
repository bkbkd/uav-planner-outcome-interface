CURRENT_MIN_START_GOAL_DISTANCE = 120.0
CURRENT_NEAR_ZERO_EUCLIDEAN_THRESHOLD = 24.0
CURRENT_NEAR_ORACLE_EUCLIDEAN_THRESHOLD = 120.0

CURRENT_PLANNER = {
    "speed": 28.0,
    "omega_max": 0.85,
    "primitive_dt_values": (0.45, 0.8, 1.15),
    "primitive_samples": 8,
    "n_actions": 11,
    "max_iters": 180000,
    "goal_tolerance": 24.0,
    "alpha": 1.0,
    "beta": 650.0,
    "gamma": 5.0,
    "lattice_xy_resolution": 24.0,
    "lattice_heading_bins": 24,
    "lattice_post_solution_expansion_limit": 200,
    "lattice_grid_guidance_weight": 0.25,
    "free_start_heading": True,
    "lattice_priority_mode": "straight",
}
