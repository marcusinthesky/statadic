"""Independent tiny Euclidean-energy, scalar-mixture and jackknife witnesses."""

from __future__ import annotations

import numpy as np


def mean_pair_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Average raw Euclidean distances, including all V-statistic self pairs."""
    return sum(
        float(np.linalg.norm(left - right)) for left in first for right in second
    ) / (len(first) * len(second))


def mixture_energy(
    x: np.ndarray, candidates: list[np.ndarray], weights: np.ndarray
) -> float:
    """Evaluate the unrooted two-sample energy of a weighted empirical measure."""
    cross = sum(
        weight * mean_pair_distance(x, group)
        for weight, group in zip(weights, candidates, strict=True)
    )
    within = sum(
        weights[first] * weights[second] * mean_pair_distance(left, right)
        for first, left in enumerate(candidates)
        for second, right in enumerate(candidates)
    )
    return float(2.0 * cross - mean_pair_distance(x, x) - within)


def two_candidate_optimum(x: np.ndarray, candidates: list[np.ndarray]) -> np.ndarray:
    """Minimize the independent scalar quadratic on [0,1], with no native solver."""
    first, second = candidates
    cross_first, cross_second = (
        mean_pair_distance(x, first),
        mean_pair_distance(x, second),
    )
    first_within = mean_pair_distance(first, first)
    second_within = mean_pair_distance(second, second)
    between = mean_pair_distance(first, second)
    curvature = 2 * between - first_within - second_within
    cost = 2 * (cross_first - cross_second - between + second_within)
    weight = float(np.clip(-cost / (2 * curvature), 0.0, 1.0))
    return np.asarray([weight, 1.0 - weight])


def studentized_energy(
    x: np.ndarray,
    candidates: list[np.ndarray],
    weights: np.ndarray,
    group_variances: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Sum separate group jackknives and independently calculate Kish effective size."""
    groups = [x, *candidates]
    variance = 0.0
    for group, values in enumerate(groups):
        replicates = []
        for dropped in range(len(values)):
            reduced = list(groups)
            reduced[group] = np.delete(values, dropped, axis=0)
            replicates.append(mixture_energy(reduced[0], reduced[1:], weights))
        replicates = np.asarray(replicates)
        variance += (
            (len(values) - 1)
            / len(values)
            * np.sum((replicates - replicates.mean()) ** 2)
        )
    sizes = np.asarray([len(group) for group in candidates])
    effective_size = 1.0 / np.sum(weights**2 / sizes)
    if group_variances is not None:
        effective_size = float(
            np.sum(weights * group_variances)
            / np.sum(weights**2 * group_variances / sizes)
        )
    outer_scale = len(x) * effective_size / (len(x) + effective_size)
    statistic = (
        np.sqrt(outer_scale)
        * mixture_energy(x, candidates, weights)
        / np.sqrt(max(variance, 1e-8))
    )
    return float(statistic), float(variance), float(effective_size)
