# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""Paired Sharpe comparisons and stationary-bootstrap Sharpe intervals.

The difference comparison uses circular blocks of paired observations; the
interval uses the stationary bootstrap. Both require an appropriate resampling
law for the actual serial dependence. Numerical execution does not establish
calibration. The Sharpe calculation is owned by :mod:`statadic.estimation.sharpe`.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from statadic.core.typing import Array, ArrayLike, PRNGKey  # noqa: TC001
from statadic.estimation.sharpe import sharpe_ratio_axis
from statadic.inference.resampling.block_length import (
    heuristic_block_length,
    heuristic_block_length_result,
    stationary_bootstrap_ci,
)
from statadic.inference.resampling.results import ConfidenceInterval  # noqa: TC001
from statadic.operators.longrun import (
    circular_block_bootstrap_indices,
    circular_block_bootstrap_indices_kernel,
)

__all__ = [
    "FloatScalar",
    "ReturnVector",
    "SharpeDifferenceKernelResult",
    "SharpeDifferenceTest",
    "sharpe_difference_test",
    "sharpe_difference_test_kernel",
    "sharpe_ratio_ci",
]

type ReturnVector = Float[Array, " observations"]  # jaxtyping
type _BootstrapMatrix = Int[Array, "bootstrap observations"]
type FloatScalar = Float[Array, ""]  # jaxtyping scalar


class SharpeDifferenceTest(NamedTuple):
    """Eager paired-Sharpe statistic and bootstrap p-value."""

    statistic: float
    pvalue: float


class SharpeDifferenceKernelResult(NamedTuple):
    """Array-only carrier for a keyed paired-Sharpe test."""

    statistic: FloatScalar
    pvalue: FloatScalar
    block_length: Int[Array, ""]


@jax.jit
def _sharpe_difference_kernel(
    returns_a: ReturnVector,
    returns_b: ReturnVector,
    indices: _BootstrapMatrix,
) -> tuple[FloatScalar, FloatScalar]:
    """Evaluate the paired test on a fixed bootstrap index matrix.

    Args:
        returns_a: First return series, shape ``(T,)``.
        returns_b: Second return series, shape ``(T,)``.
        indices: Fixed bootstrap indices, shape ``(B, T)``.

    Returns:
        Studentized Sharpe difference and two-sided bootstrap p-value.

    """
    delta_hat = sharpe_ratio_axis(returns_a, axis=0) - sharpe_ratio_axis(
        returns_b, axis=0
    )
    delta_boot = sharpe_ratio_axis(returns_a[indices], axis=1) - sharpe_ratio_axis(
        returns_b[indices], axis=1
    )
    standard_error = jnp.std(delta_boot, ddof=1)
    statistic = jnp.where(standard_error > 0.0, delta_hat / standard_error, 0.0)
    centered = delta_boot - jnp.mean(delta_boot)
    pvalue = jnp.mean(
        jnp.abs(centered) >= jnp.abs(delta_hat),
        dtype=returns_a.dtype,
    )
    # The p-value uses only ``centered`` and ``delta_hat``; the raw returns are
    # checked too because the Sharpe ratio's degenerate-series fallback can hide
    # a non-finite input.
    valid = (
        jnp.all(jnp.isfinite(returns_a))
        & jnp.all(jnp.isfinite(returns_b))
        & jnp.isfinite(delta_hat)
        & jnp.all(jnp.isfinite(centered))
    )
    return statistic, jnp.where(valid, jnp.clip(pvalue, 0.0, 1.0), jnp.nan)


@partial(jax.jit, static_argnames=("n_boot",))
def sharpe_difference_test_kernel(
    key: PRNGKey,
    returns_a: ReturnVector,
    returns_b: ReturnVector,
    *,
    n_boot: int,
) -> SharpeDifferenceKernelResult:
    """Run the paired-Sharpe test from an explicit bootstrap key."""
    difference = returns_a - returns_b
    block_length = heuristic_block_length_result(difference)
    indices = circular_block_bootstrap_indices_kernel(
        key,
        returns_a.shape[0],
        block_length,
        n_boot,
    )
    statistic, pvalue = _sharpe_difference_kernel(returns_a, returns_b, indices)
    return SharpeDifferenceKernelResult(statistic, pvalue, block_length)


def sharpe_difference_test(
    returns_a: ArrayLike,
    returns_b: ArrayLike,
    n_boot: int = 2000,
    block_length: int | None = None,
    seed: int = 0,
) -> SharpeDifferenceTest:
    """Compare paired Sharpes using centered circular-bootstrap differences.

    Tests ``H0: SR(returns_a) == SR(returns_b)`` for two paired strategy
    return series. Circular blocks resample paired rows; block length is chosen by
    :func:`statadic.inference.resampling.block_length.heuristic_block_length`
    applied to the
    return differential ``d_t = a_t - b_t`` unless overridden. Bootstrap
    differences are centered at their bootstrap mean. The returned statistic
    uses one bootstrap standard deviation; each resample is not separately
    studentized. This differs from Ledoit-Wolf's (2008) studentized procedure.
    The p-value is a raw finite-bootstrap exceedance proportion, without an
    add-one correction. Validity requires suitable resampling assumptions.

    Args:
        returns_a: Strategy-A return series, shape ``(T,)``.
        returns_b: Strategy-B return series, shape ``(T,)``, paired with
            ``returns_a`` (same dates).
        n_boot: Number of circular-block-bootstrap resamples.
        block_length: Block length ``l``; ``None`` uses the first-lag
            heuristic on the differential series.
        seed: RNG seed for the bootstrap.

    Returns:
        Tuple ``(stat, pvalue)`` where ``stat`` is the studentized statistic
        ``dSR / se_boot`` (``dSR = SR(a) - SR(b)``, ``se_boot`` the bootstrap
        standard deviation of ``dSR``) and ``pvalue`` is the two-sided
        bootstrap p-value for ``H0: dSR = 0``.

    Raises:
        ValueError: If the two return series have different shapes.

    """
    a = jnp.asarray(returns_a).ravel()
    b = jnp.asarray(returns_b).ravel()
    if a.shape != b.shape:
        message = "returns_a and returns_b must have the same shape"
        raise ValueError(message)
    n_observations = a.size
    difference = a - b
    ell = (
        heuristic_block_length(jax.device_get(difference))
        if block_length is None
        else int(block_length)
    )
    ell = max(1, min(ell, n_observations))

    indices = jnp.asarray(
        circular_block_bootstrap_indices(
            n_observations,
            ell,
            n_boot,
            seed=seed,
        ),
        dtype=jnp.int64,
    )
    statistic, pvalue = _sharpe_difference_kernel(a, b, indices)
    return SharpeDifferenceTest(float(statistic), float(pvalue))


def _sharpe_bootstrap_statistic(values: Array) -> Array:
    """Stable scalar callable shared by repeated stationary-bootstrap reports."""
    column = values[:, 0] if values.ndim > 1 else values
    deviation = jnp.std(column, ddof=1)
    varying = jnp.ptp(column) > 0.0 if column.size else jnp.zeros((), dtype=bool)
    return jnp.where(varying & (deviation > 0), jnp.mean(column) / deviation, 0.0)


def sharpe_ratio_ci(
    returns: ArrayLike,
    confidence_level: float = 0.95,
    num_bootstrap: int = 999,
    seed: int = 0,
) -> ConfidenceInterval:
    """Stationary-bootstrap CI for the per-observation Sharpe ratio.

    Thin, parity-preserving wrapper: routes through
    :func:`statadic.inference.resampling.block_length.stationary_bootstrap_ci` with the
    Politis-White automatic block length (its default), so overlapping-window
    autocorrelation is handled by the block bootstrap rather than an iid
    assumption.

    Args:
        returns: Return series, shape ``(n,)``.
        confidence_level: CI level.
        num_bootstrap: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        ``(low, high)`` percentile CI for the Sharpe ratio.

    """
    values = jnp.asarray(returns).ravel()
    return stationary_bootstrap_ci(
        values,
        _sharpe_bootstrap_statistic,
        confidence_level=confidence_level,
        num_bootstrap=num_bootstrap,
        seed=seed,
    )
