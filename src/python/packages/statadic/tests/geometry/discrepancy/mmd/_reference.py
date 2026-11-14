"""Small independent empirical functionals for discrepancy and mixture tests."""

from __future__ import annotations

import numpy as np


def mean_power_distance(
    left: np.ndarray,
    right: np.ndarray,
    exponent: float = 1.0,
    *,
    angular: bool = False,
) -> float:
    """Average ordered point pairs, including coincident observations."""
    total = 0.0
    for x in left:
        for y in right:
            if angular:
                cosine = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
                distance = np.arccos(np.clip(cosine, -1.0, 1.0))
            else:
                distance = np.linalg.norm(x - y)
            total += float(distance**exponent)
    return total / (len(left) * len(right))


def energy_v(
    left: np.ndarray,
    right: np.ndarray,
    exponent: float = 1.0,
    *,
    angular: bool = False,
) -> float:
    """Calculate the raw V-statistic without calling a STATADIC primitive."""
    return (
        2 * mean_power_distance(left, right, exponent, angular=angular)
        - mean_power_distance(left, left, exponent, angular=angular)
        - mean_power_distance(right, right, exponent, angular=angular)
    )


def rbf_block(left: np.ndarray, right: np.ndarray, bandwidth: float) -> np.ndarray:
    """Build a Gaussian block directly from row differences and a stated sigma."""
    return np.asarray(
        [
            [np.exp(-np.dot(x - y, x - y) / (2 * bandwidth**2)) for y in right]
            for x in left
        ]
    )


def mmd_from_blocks(
    xx: np.ndarray,
    yy: np.ndarray,
    xy: np.ndarray,
    *,
    unbiased: bool,
) -> float:
    """Reduce independently supplied Gram blocks under explicit U/V divisors."""
    if unbiased:
        within_x = sum(
            xx[i, j] for i in range(len(xx)) for j in range(len(xx)) if i != j
        )
        within_y = sum(
            yy[i, j] for i in range(len(yy)) for j in range(len(yy)) if i != j
        )
        within_x /= len(xx) * (len(xx) - 1)
        within_y /= len(yy) * (len(yy) - 1)
    else:
        within_x = xx.sum() / xx.size
        within_y = yy.sum() / yy.size
    return float(within_x + within_y - 2 * xy.sum() / xy.size)
