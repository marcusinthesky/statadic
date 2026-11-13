"""Pure data generation, numerical witnesses and Monte Carlo summaries.

These helpers generate synthetic teaching data and independent reference values.
They do NOT implement a substitute backend for any interactive STATADIC result.
All random generators are local. Paired prefixes make slider experiments stable.
"""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Literal

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.floating]


def bounded_integer(value: int, name: str, low: int, high: int) -> int:
    """Validate a bounded integer without silently truncating a float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}.")
    return result


def finite_scalar(value: float, name: str, low: float, high: float) -> float:
    """Validate a finite, bounded numerical setting."""
    result = float(value)
    if not np.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be finite and between {low} and {high}.")
    return result


def cloud_pair(
    n: int = 64,
    shift: float = 1.0,
    scale: float = 1.0,
    seed: int = 42,
    mode: Literal["independent", "scaled-copy"] = "independent",
) -> tuple[FloatArray, FloatArray]:
    """Create two 2D clouds, preserving prefixes when sample size changes.

    Independent mode draws two independent standard-normal bases. Scaled-copy
    mode is a deliberately dependent geometric experiment, NOT two IID samples.
    Shift is along coordinate one. The maximum sample capacity is fixed at 128.
    """
    n = bounded_integer(n, "n", 8, 128)
    seed = bounded_integer(seed, "seed", 0, 2**32 - 1)
    shift = finite_scalar(shift, "shift", -3, 3)
    scale = finite_scalar(scale, "scale", 0.25, 3)
    left_seed, right_seed = np.random.SeedSequence(seed).spawn(2)
    x = np.random.default_rng(left_seed).normal(size=(128, 2))[:n]
    if mode == "independent":
        base_y = np.random.default_rng(right_seed).normal(size=(128, 2))[:n]
    elif mode == "scaled-copy":
        base_y = x.copy()
    else:
        raise ValueError("mode must be independent or scaled-copy.")
    y = scale * base_y + np.array([shift, 0.0])
    return x, y


def vector_pair(radius: float, angle_degrees: float) -> tuple[FloatArray, FloatArray]:
    """A fixed unit vector and a vector with adjustable length and direction."""
    radius = finite_scalar(radius, "radius", 0, 3)
    angle = np.deg2rad(finite_scalar(angle_degrees, "angle", -180, 180))
    return np.array([[1.0, 0.0]]), np.array(
        [[radius * np.cos(angle), radius * np.sin(angle)]]
    )


def reference_raw_energy(x: FloatArray, y: FloatArray) -> float:
    """Independent Euclidean, exponent-one, empirical V-energy witness.

    Explicit broadcasting deliberately avoids STATADIC/JAX centering or geometry.
    Self-pairs are retained. This value is for tests, not a STATADIC UI fallback.
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Expected compatible two-dimensional clouds.")
    if min(len(x), len(y)) == 0 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Clouds must be nonempty and finite.")

    def mean_distance(a: FloatArray, b: FloatArray) -> float:
        return float(np.sqrt(np.sum((a[:, None] - b[None, :]) ** 2, axis=-1)).mean())

    return 2 * mean_distance(x, y) - mean_distance(x, x) - mean_distance(y, y)


def measure_dataset(spread: float = 1.0, seed: int = 7) -> dict:
    """Twenty-four whole units, each represented by an unequal-sized cloud.

    This is a synthetic workflow example, not a performance benchmark. The split
    is determined before fitting: units 0..17 train; units 18..23 are held out.
    """
    spread = finite_scalar(spread, "spread", 0.5, 2)
    seed = bounded_integer(seed, "seed", 0, 2**32 - 1)
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-1.5, 1.5, 24)
    price = rng.normal(size=24)
    outcome_noise = rng.normal(size=24)
    clouds = []
    for i in range(24):
        size = 8 + i % 5
        # Equal deterministic quantiles within a unit keep spread changes stable.
        z = np.linspace(-1.4, 1.4, size)
        clouds.append((centres[i] + spread * (0.5 + 0.1 * (i % 3)) * z)[:, None])
    return {
        "clouds": clouds,
        "price": price,
        "outcome": 1 + 0.5 * price + centres**2 + 0.15 * outcome_noise,
        "row_ids": tuple(f"unit-{i:02d}" for i in range(24)),
    }


def wilson_interval(
    successes: int, trials: int, level: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for an IID Bernoulli rejection frequency.

    This quantifies Monte Carlo uncertainty, not estimator coverage. It is an
    approximate interval; it does not certify a test's size across other designs.
    """
    trials = bounded_integer(trials, "trials", 1, 10**7)
    successes = bounded_integer(successes, "successes", 0, trials)
    level = finite_scalar(level, "level", 0.5, 0.999)
    z = NormalDist().inv_cdf((1 + level) / 2)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / denominator
    # Analytic endpoints are exact at 0/n and n/n; retain them despite cancellation.
    lower = 0.0 if successes == 0 else max(0.0, centre - half)
    upper = 1.0 if successes == trials else min(1.0, centre + half)
    return lower, upper


def add_one_pvalue(observed: float, null: FloatArray) -> float:
    """Independent upper-tail Monte Carlo add-one calculation for tests."""
    null = np.asarray(null)
    if (
        null.ndim != 1
        or len(null) == 0
        or not np.isfinite(null).all()
        or not np.isfinite(observed)
    ):
        raise ValueError(
            "Observed and null statistics must be finite; null must be nonempty."
        )
    return float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))


def calibration_budget(n: int, permutations: int, trials: int) -> tuple[int, int, int]:
    """Bound all three dimensions before any potentially expensive native call."""
    n = bounded_integer(n, "n", 8, 32)
    permutations = bounded_integer(permutations, "permutations", 19, 99)
    trials = bounded_integer(trials, "trials", 5, 100)
    if 3 * trials * permutations * (2 * n) ** 2 > 125_000_000:
        raise ValueError(
            "Teaching-run budget exceeded; lower n, trials or permutations."
        )
    return n, permutations, trials


def reject_at_five_percent(exceedances: int, permutations: int) -> bool:
    """Compare the add-one fraction with 1/20 using integers, not rounded floats.

    In particular, a float32 representation of 0.05 must not change whether
    exactly 1/20 qualifies as p <= 0.05 after conversion to a Python float.
    """
    permutations = bounded_integer(permutations, "permutations", 1, 1000000)
    exceedances = bounded_integer(exceedances, "exceedances", 0, permutations)
    return 20 * (1 + exceedances) <= 1 + permutations
