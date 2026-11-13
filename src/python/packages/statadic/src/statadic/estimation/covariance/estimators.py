# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""Sample-covariance and constant-correlation Ledoit-Wolf estimators."""

# jaxtyping shape strings are runtime contracts.
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, SupportsIndex

import jax
import jax.numpy as jnp
from jaxtyping import Float as JaxFloat  # noqa: TC002

from statadic.core.typing import (  # noqa: TC001
    Array,
    ArrayLike,
    Static,
    as_index,
)
from statadic.estimation.covariance._validation import validate_observation_panel

__all__ = [
    "CovarianceProblem",
    "LedoitWolf",
    "LedoitWolfKernelResult",
    "SampleCovariance",
    "ledoit_wolf_sample_kernel",
    "sample_covariance",
    "sample_covariance_kernel",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CovarianceProblem:
    """Observation-by-variable values for a declared covariance estimator.

    Host preparation establishes finite values and estimator-specific domains;
    the native bricks retain the existing kernels' transformation boundary.
    """

    values: ArrayLike


@dataclass(frozen=True, init=False)
class SampleCovariance:
    """Ordinary covariance with the explicit divisor ``n - ddof``."""

    ddof: int = 1

    def __init__(self, ddof: SupportsIndex = 1) -> None:
        """Require a static nonnegative integer divisor adjustment."""
        normalized = as_index(ddof)
        if normalized is None or normalized < 0:
            message = "ddof must be a nonnegative integer."
            raise ValueError(message)
        object.__setattr__(self, "ddof", normalized)

    def estimate(self, problem: CovarianceProblem) -> Array:
        """Apply the existing sample-covariance kernel to prepared values."""
        return sample_covariance_kernel(jnp.asarray(problem.values), self.ddof)


@dataclass(frozen=True)
class LedoitWolf:
    """Constant-correlation shrinkage of the ``ddof=0`` sample covariance."""

    @staticmethod
    def estimate(problem: CovarianceProblem) -> LedoitWolfKernelResult:
        """Apply the existing Ledoit-Wolf kernel to prepared values."""
        return ledoit_wolf_sample_kernel(jnp.asarray(problem.values))


@partial(jax.jit, static_argnames=("ddof",))
def sample_covariance_kernel(
    returns: JaxFloat[Array, "t n"],
    ddof: Static[int] = 1,
) -> JaxFloat[Array, "n n"]:
    """Return a fixed-rank sample covariance entirely on the JAX graph."""
    centered = returns - jnp.mean(returns, axis=0, keepdims=True)
    return centered.T @ centered / (returns.shape[0] - ddof)


def sample_covariance(returns: ArrayLike, ddof: int = 1) -> JaxFloat[Array, "n n"]:
    """Return the ordinary sample covariance of an observation panel."""
    panel = validate_observation_panel(returns, ddof=ddof)
    return sample_covariance_kernel(panel, ddof)


class LedoitWolfKernelResult(NamedTuple):
    """Fixed-structure JAX constant-correlation shrinkage result."""

    sigma: JaxFloat[Array, "n n"]
    shrinkage: JaxFloat[Array, ""]
    target: JaxFloat[Array, "n n"]


def _constant_correlation_target_kernel(
    sample: JaxFloat[Array, "n n"],
) -> JaxFloat[Array, "n n"]:
    """Build the constant-correlation target without traced boolean slicing."""
    n_variables = sample.shape[0]
    variance = jnp.diag(sample)
    standard_deviation = jnp.sqrt(variance)
    correlation = sample / jnp.outer(standard_deviation, standard_deviation)
    mask = 1.0 - jnp.eye(n_variables, dtype=sample.dtype)
    mean_correlation = jnp.sum(correlation * mask) / (n_variables * (n_variables - 1))
    target = mean_correlation * jnp.outer(standard_deviation, standard_deviation)
    diagonal = jnp.diag_indices(n_variables)
    return target.at[diagonal].set(variance)


def _rho_hat_kernel(
    sample: JaxFloat[Array, "n n"],
    centered: JaxFloat[Array, "t n"],
    deviations: JaxFloat[Array, "t n n"],
    entry_variance: JaxFloat[Array, "n n"],
) -> JaxFloat[Array, ""]:
    """Accumulate the constant-correlation asymptotic covariance term."""
    n_variables = sample.shape[0]
    variance = jnp.diag(sample)
    standard_deviation = jnp.sqrt(variance)
    correlation = sample / jnp.outer(standard_deviation, standard_deviation)
    mask = 1.0 - jnp.eye(n_variables, dtype=sample.dtype)
    mean_correlation = jnp.sum(correlation * mask) / (n_variables * (n_variables - 1))
    squared = centered**2
    theta_ii = jnp.mean(
        (squared[:, :, None] - variance[None, :, None]) * deviations,
        axis=0,
    )
    theta_jj = jnp.mean(
        (squared[:, None, :] - variance[None, None, :]) * deviations,
        axis=0,
    )
    coefficient = (
        0.5
        * mean_correlation
        * (
            jnp.sqrt(jnp.outer(1.0 / variance, variance)) * theta_ii
            + jnp.sqrt(jnp.outer(variance, 1.0 / variance)) * theta_jj
        )
    )
    return jnp.trace(entry_variance) + jnp.sum(coefficient * mask)


@jax.jit
def ledoit_wolf_sample_kernel(
    returns: JaxFloat[Array, "t n"],
) -> LedoitWolfKernelResult:
    """Estimate constant-correlation Ledoit-Wolf shrinkage on graph."""
    n_observations, n_variables = returns.shape
    sample = sample_covariance_kernel(returns, ddof=0)
    if n_variables == 1:
        return LedoitWolfKernelResult(
            sigma=sample,
            shrinkage=jnp.asarray(0.0, dtype=returns.dtype),
            target=sample,
        )
    target = _constant_correlation_target_kernel(sample)
    centered = returns - jnp.mean(returns, axis=0, keepdims=True)
    products = centered[:, :, None] * centered[:, None, :]
    deviations = products - sample[None, :, :]
    entry_variance = jnp.mean(deviations**2, axis=0)
    pi_hat = jnp.sum(entry_variance)
    rho_hat = _rho_hat_kernel(sample, centered, deviations, entry_variance)
    gamma_hat = jnp.sum((target - sample) ** 2)
    kappa = jnp.where(gamma_hat > 0.0, (pi_hat - rho_hat) / gamma_hat, 0.0)
    shrinkage = jnp.clip(kappa / n_observations, 0.0, 1.0)
    sigma = shrinkage * target + (1.0 - shrinkage) * sample
    return LedoitWolfKernelResult(
        sigma=sigma,
        shrinkage=shrinkage,
        target=target,
    )
