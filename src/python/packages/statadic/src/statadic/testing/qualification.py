"""Pure trial accounting for prespecified library qualification studies.

This module neither generates data nor calls the implementation being qualified.
Families own their DGP, complete procedure and definition of a valid trial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc

__all__ = [
    "BOUNDS_METHOD",
    "DEFAULT_DRAWS",
    "DEFAULT_TRIALS",
    "TRIAL_KEY_SCHEME",
    "Qualification",
    "QualificationReport",
    "TrialOutcome",
    "summarize_trials",
    "trial_key",
]

type Qualification = Literal["qualified", "excess-error", "inconclusive"]

DEFAULT_TRIALS = 10_000
DEFAULT_DRAWS = 999
BOUNDS_METHOD = "clopper-pearson-betainc-v2"
TRIAL_KEY_SCHEME = "trial-key-v2"
_KEY_SEED = 20261028
_WORD_BITS = 32
_WORD_MASK = 2**_WORD_BITS - 1
_BOUND_ITERATIONS = 64


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    """Retain a complete trial, including unavailable inference."""

    valid: bool
    rejected: bool = False
    covered: bool | None = None

    def __post_init__(self) -> None:
        """Reject contradictory trial evidence."""
        if not self.valid and (self.rejected or self.covered is True):
            message = "An unavailable trial cannot reject or cover."
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class QualificationReport:
    """Failure-inclusive counts and one-sided exact binomial bounds."""

    requested: int
    valid: int
    failed: int
    rejected: int
    covered: int | None
    conditional_rejection: float | None
    size_lower: float
    size_upper: float
    failure_lower: float
    failure_upper: float
    coverage_lower: float | None
    coverage_upper: float | None
    size_status: Qualification
    coverage_status: Qualification | None
    bounds_method: str
    bounds_dtype: str


def trial_key(scenario: int, trial: int, *, stream: int = 0) -> jax.Array:
    """Return a trial-key-v2 typed Threefry stream without integer narrowing."""
    key = jax.random.key(_KEY_SEED, impl="threefry2x32")
    for tag, value in enumerate((scenario, trial, stream)):
        if type(value) is not int or value < 0:
            msg = "Scenario, trial and stream IDs must be nonnegative integers"
            raise ValueError(msg)
        words = []
        remaining = value
        while remaining:
            words.append(remaining & _WORD_MASK)
            remaining >>= _WORD_BITS
        key = jax.random.fold_in(key, tag)
        key = jax.random.fold_in(key, len(words))
        for word in words:
            key = jax.random.fold_in(key, word)
    return key


@jax.jit
def _bounds_kernel(
    events: jax.Array, total: jax.Array, confidence: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Invert the two beta CDFs defining one-sided Clopper-Pearson bounds."""
    # Clopper-Pearson construction, evaluated at ordinary caller precision.
    probabilities = jnp.asarray([1 - confidence, confidence])
    a = jnp.asarray([jnp.maximum(events, 1), events + 1], dtype=probabilities.dtype)
    b = jnp.asarray(
        [total - events + 1, jnp.maximum(total - events, 1)], dtype=probabilities.dtype
    )

    def bisect(
        _index: jax.Array, state: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        lo, hi = state
        mid = lo + (hi - lo) / 2
        below = betainc(a, b, mid) < probabilities
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = jax.lax.fori_loop(
        0, _BOUND_ITERATIONS, bisect, (jnp.zeros_like(a), jnp.ones_like(a))
    )
    lower = jnp.where(events == 0, 0, lo[0])
    upper = jnp.where(events == total, 1, hi[1])
    return lower, upper


def _bounds(events: int, total: int, confidence: float) -> tuple[float, float]:
    """Transfer one binomial interval at the eager reporting boundary."""
    lower, upper = _bounds_kernel(
        jnp.asarray(events), jnp.asarray(total), jnp.asarray(confidence)
    )
    return float(lower), float(upper)


def _status(
    lower: float,
    upper: float,
    limit: float,
    failure: tuple[float, float],
    budget: float,
) -> Qualification:
    if lower > limit or failure[0] > budget:
        return "excess-error"
    if upper <= limit and failure[1] <= budget:
        return "qualified"
    return "inconclusive"


def summarize_trials(
    outcomes: list[TrialOutcome],
    *,
    alpha: float = 0.05,
    margin: float = 0.01,
    failure_budget: float = 0.005,
    confidence: float = 0.95,
    coverage: bool = False,
) -> QualificationReport:
    """Assess size and optional coverage without discarding failed trials.

    Size treats every failure as a rejection; coverage treats it as uncovered.
    Both confidence bounds are one-sided at the declared confidence level.
    """
    if not outcomes:
        message = "At least one attempted trial is required."
        raise ValueError(message)
    if not (0 < alpha < 1 and 0 <= margin < min(alpha, 1 - alpha)):
        message = "Alpha and margin must define interior acceptance limits."
        raise ValueError(message)
    if not (0 <= failure_budget < 1 and 0 < confidence < 1):
        message = "Invalid failure budget or confidence level."
        raise ValueError(message)
    if coverage and any(o.valid and o.covered is None for o in outcomes):
        message = "Every valid coverage trial needs a coverage outcome."
        raise ValueError(message)
    total = len(outcomes)
    valid = sum(o.valid for o in outcomes)
    failed = total - valid
    rejected = sum(o.rejected for o in outcomes)
    size_bounds = _bounds(rejected + failed, total, confidence)
    failures = _bounds(failed, total, confidence)
    covered = sum(o.covered is True for o in outcomes) if coverage else None
    cov_bounds = _bounds(covered, total, confidence) if covered is not None else None
    coverage_status = None
    if cov_bounds is not None:
        coverage_status = _status(
            1 - cov_bounds[1],
            1 - cov_bounds[0],
            alpha + margin,
            failures,
            failure_budget,
        )
    return QualificationReport(
        total,
        valid,
        failed,
        rejected,
        covered,
        rejected / valid if valid else None,
        *size_bounds,
        *failures,
        cov_bounds[0] if cov_bounds else None,
        cov_bounds[1] if cov_bounds else None,
        _status(*size_bounds, alpha + margin, failures, failure_budget),
        coverage_status,
        BOUNDS_METHOD,
        jnp.asarray(confidence).dtype.name,
    )
