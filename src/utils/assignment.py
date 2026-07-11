from __future__ import annotations

import numpy as np


def solve_linear_assignment(cost_matrix: np.ndarray) -> tuple[tuple[int, ...], float]:
    """Solve a rectangular minimum-cost assignment with n_rows <= n_cols.

    This is the Hungarian/Kuhn-Munkres algorithm in the shortest augmenting
    path form. It returns one selected column per row.
    """
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be a 2D array.")
    n_rows, n_cols = cost.shape
    if n_cols < n_rows:
        raise ValueError("Assignment requires at least as many columns as rows.")
    if not np.all(np.isfinite(cost)):
        raise ValueError("cost_matrix must contain only finite costs.")
    if n_rows == 0:
        return (), 0.0

    # 1-indexed implementation following the standard rectangular Hungarian
    # algorithm for minimization. p[j] is the row matched to column j.
    u = np.zeros(n_rows + 1, dtype=float)
    v = np.zeros(n_cols + 1, dtype=float)
    p = np.zeros(n_cols + 1, dtype=int)
    way = np.zeros(n_cols + 1, dtype=int)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf, dtype=float)
        used = np.zeros(n_cols + 1, dtype=bool)
        way.fill(0)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = np.full(n_rows, -1, dtype=int)
    for j in range(1, n_cols + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    if np.any(assignment < 0):
        raise RuntimeError("Hungarian solver failed to assign every row.")
    assignment_tuple = tuple(int(item) for item in assignment)
    total = float(cost[np.arange(n_rows), assignment].sum())
    return assignment_tuple, total
