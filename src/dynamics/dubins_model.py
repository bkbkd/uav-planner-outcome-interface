from __future__ import annotations

import numpy as np

from src.utils.geometry import wrap_angle


def rollout_arc(
    state: tuple[float, float, float] | np.ndarray,
    omega: float,
    speed: float,
    dt: float,
    num_samples: int = 12,
    omega_eps: float = 1e-6,
) -> np.ndarray:
    """Roll out one constant-curvature motion primitive with analytic arc equations."""
    x0, y0, theta0 = map(float, state)
    ts = np.linspace(0.0, dt, num_samples + 1)

    if abs(omega) < omega_eps:
        xs = x0 + speed * ts * np.cos(theta0)
        ys = y0 + speed * ts * np.sin(theta0)
        thetas = np.full_like(ts, theta0)
    else:
        theta_t = theta0 + omega * ts
        xs = x0 + speed / omega * (np.sin(theta_t) - np.sin(theta0))
        ys = y0 - speed / omega * (np.cos(theta_t) - np.cos(theta0))
        thetas = theta_t

    wrapped = np.vectorize(wrap_angle)(thetas)
    return np.column_stack((xs, ys, wrapped))

