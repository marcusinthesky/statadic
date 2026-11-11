"""Construct shrinkable full-rank linear cases without conditioning on a fit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp


@dataclass(frozen=True, eq=False, slots=True)
class LinearCase:
    """Own immutable represented inputs for independent linear test routes.

    Attributes:
        design: Full-column-rank design with a nonsingular triangular anchor.
        outcome: Outcome with a resolved nonzero component outside the design.
        precision: Positive, finite relative precision weights.

    """

    design: np.ndarray
    outcome: np.ndarray
    precision: np.ndarray


@st.composite
def linear_cases(
    draw: st.DrawFn,
    *,
    n: int,
    p: int,
    intercept: bool = False,
    exact: bool = False,
    weighted: bool = True,
) -> LinearCase:
    """Construct a fixed-shape rank-guaranteed linear case.

    Args:
        draw: Hypothesis value supplier.
        n: Observation count, strictly larger than parameter count.
        p: Positive parameter count.
        intercept: Reserve the first design column for a constant.
        exact: Omit residual noise for exact-model cases.
        weighted: Draw positive precision weights; otherwise use unit weights.

    Returns:
        Fresh readonly arrays with positive precision and guaranteed residual df.

    Raises:
        ValueError: Dimensions cannot support positive residual df.

    """
    if not 0 < p < n:
        message = "linear_cases requires 0 < p < n."
        raise ValueError(message)
    entries = st.integers(-4, 4)
    design = (
        np.asarray(
            draw(st.lists(entries, min_size=n * p, max_size=n * p)), dtype=float
        ).reshape(n, p)
        / 8
    )
    design[:p] = np.tril(design[:p], k=-1) + np.eye(p)
    if intercept:
        design[:p] = np.eye(p)
        design[:, 0] = 1
    beta = np.asarray(draw(st.lists(entries, min_size=p, max_size=p))) / 4
    noise = np.zeros(n)
    noise[p:] = np.asarray(draw(st.lists(entries, min_size=n - p, max_size=n - p))) / 8
    noise[-1] = 0.5
    if exact:
        noise[:] = 0
    outcome = design @ beta + noise
    precision = (
        np.asarray(
            draw(st.lists(st.integers(1, 12), min_size=n, max_size=n)), dtype=float
        )
        / 4
    )
    if not weighted:
        precision[:] = 1
    dtype = np.dtype(jnp.result_type(float))
    design, outcome, precision = (
        np.asarray(array, dtype=dtype) for array in (design, outcome, precision)
    )
    for array in (design, outcome, precision):
        array.setflags(write=False)
    return LinearCase(design, outcome, precision)


def ols_oracle(case: LinearCase, *, weighted: bool = False) -> np.ndarray:
    """Solve represented inputs independently by NumPy's reduced QR."""
    root = (
        np.sqrt(np.asarray(case.precision, dtype=float))
        if weighted
        else np.ones(len(case.outcome))
    )
    q, r = np.linalg.qr(root[:, None] * case.design)
    return np.linalg.solve(r, q.T @ (root * case.outcome))


@st.composite
def perturbed_linear_cases(draw: st.DrawFn, baseline: LinearCase) -> LinearCase:
    """Vary noise and weights while preserving a reference design and precision."""
    n = len(baseline.outcome)
    width = 32 if baseline.outcome.dtype == np.float32 else 64
    noise = draw(
        hnp.arrays(
            baseline.outcome.dtype, n, elements=st.floats(-0.5, 0.5, width=width)
        )
    )
    scale = draw(
        hnp.arrays(
            baseline.precision.dtype, n, elements=st.floats(0.5, 2.0, width=width)
        )
    )
    arrays = (
        baseline.design.copy(),
        baseline.outcome + noise,
        baseline.precision * scale,
    )
    for array in arrays:
        array.setflags(write=False)
    return LinearCase(*arrays)


def wald_oracle(
    restriction: np.ndarray, null: np.ndarray, beta: np.ndarray, covariance: np.ndarray
) -> float:
    """Evaluate an identified Wald restriction without a production test kernel."""
    delta = restriction @ beta - null
    return float(
        delta @ np.linalg.solve(restriction @ covariance @ restriction.T, delta)
    )


def information_criteria(log_likelihood: float, k: int, n: int) -> dict[str, float]:
    """Independent penalties under STATADIC's declared parameter-count convention."""
    deviance = -2 * log_likelihood
    return {
        "aic": deviance + 2 * k,
        "bic": deviance + k * math.log(n),
        "hqic": deviance + 2 * k * math.log(math.log(n)),
    }
