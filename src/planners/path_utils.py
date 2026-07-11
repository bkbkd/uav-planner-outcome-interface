from __future__ import annotations

import math

import numpy as np

from src.envs.scenario import Scenario


def path_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    deltas = np.diff(path[:, :2], axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def path_time(path: np.ndarray, speed: float) -> float:
    return path_length(path) / float(speed)


def path_risk(path: np.ndarray, scenario: Scenario, speed: float) -> float:
    if len(path) < 2:
        return 0.0
    midpoints = 0.5 * (path[:-1, :2] + path[1:, :2])
    ds = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    rates = scenario.risk_values(midpoints[:, 0], midpoints[:, 1])
    return float(np.sum(rates * ds / float(speed)))


def path_turning(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    dtheta = np.diff(path[:, 2])
    dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.abs(dtheta).sum())


def path_objective(
    path: np.ndarray,
    scenario: Scenario,
    speed: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
) -> float:
    return alpha * path_length(path) + beta * path_risk(path, scenario, speed) + gamma * path_turning(path)


def survival_prob(risk: float) -> float:
    return float(math.exp(-max(0.0, risk)))


def evaluate_path(path: np.ndarray, scenario: Scenario, speed: float) -> dict[str, float]:
    length = path_length(path)
    time = length / float(speed)
    risk = path_risk(path, scenario, speed)
    return {
        "length": length,
        "time": time,
        "risk": risk,
        "survival_prob": survival_prob(risk),
        "turning": path_turning(path),
    }
