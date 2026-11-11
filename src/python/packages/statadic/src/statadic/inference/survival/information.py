"""Observed-information covariance for actual fitted survival candidates.

These calculations establish numerical availability only. They do not establish
existence of a global optimum, censoring assumptions, or inferential calibration.
Spatial likelihoods retain dependence within each period; no rowwise sandwich
is constructed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from statadic.core.results import InferenceStatus
from statadic.core.typing import Array  # noqa: TC001
from statadic.estimation.survival.fit import SurvivalFit
from statadic.estimation.survival.spatial import SpatialSurvivalFit

__all__ = [
    "SurvivalCovariance",
    "SurvivalUncertainty",
    "observed_information",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SurvivalCovariance:
    """Full free-parameter covariance and observed spell count.

    The count is descriptive; spatial spells within a period are dependent.
    """

    values: Array
    n_observations: Array
    rank: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SurvivalUncertainty:
    """Numerical covariance attached to the actual local fitted candidate."""

    fit: SurvivalFit | SpatialSurvivalFit
    covariance: SurvivalCovariance
    valid: Array
    status: Array

    @property
    def sample_capacity(self) -> int:
        """Number of observed spells retained by this fit."""
        if isinstance(self.fit, SpatialSurvivalFit):
            return sum(
                len(period.graph.observed_indices)
                for period in self.fit.problem.periods
            )
        return jnp.asarray(self.fit.problem.duration).shape[0]


def observed_information(
    fit: SurvivalFit | SpatialSurvivalFit,
) -> SurvivalUncertainty:
    """Invert the total observed Hessian in the fit's free-parameter order.

    Reject nonfinite, asymmetric, rank-deficient or non-positive-definite
    information. The fit's numerical validity and, for spatial fits, its
    explicit information availability remain separate prerequisites. This
    covariance does not choose or validate a reference distribution.
    """
    if not isinstance(fit, (SurvivalFit, SpatialSurvivalFit)):
        msg = "Expected a retained independent or spatial survival fit."
        raise TypeError(msg)
    info = fit.observed_information
    p = fit.parameters.shape[-1]
    if info.shape[-2:] != (p, p):
        msg = "Observed information must match the fitted free parameters."
        raise ValueError(msg)
    count = (
        sum(len(period.graph.observed_indices) for period in fit.problem.periods)
        if isinstance(fit, SpatialSurvivalFit)
        else jnp.asarray(fit.problem.duration).shape[0]
    )
    eligible = fit.valid
    if isinstance(fit, SpatialSurvivalFit):
        eligible &= fit.information_available
    values, rank, available = _invert_information(info, eligible)
    covariance = SurvivalCovariance(values, jnp.asarray(count, dtype=jnp.int32), rank)
    status = jnp.where(
        available, InferenceStatus.AVAILABLE, InferenceStatus.INVALID_COVARIANCE
    )
    status = jnp.where(fit.valid, status, InferenceStatus.INVALID_FIT)
    return SurvivalUncertainty(
        fit, covariance, available, jnp.asarray(status, dtype=jnp.int32)
    )


@jax.jit
def _invert_information(info: Array, eligible: Array) -> tuple[Array, Array, Array]:
    """Use one symmetric spectrum for positive definiteness and rank."""
    p = info.shape[-1]
    finite = jnp.all(jnp.isfinite(info), axis=(-2, -1))
    safe = jnp.where(jnp.isfinite(info), info, 0.0)
    transpose = jnp.swapaxes(safe, -1, -2)
    tolerance = (
        10 * p * jnp.finfo(safe.dtype).eps * jnp.max(jnp.abs(safe), axis=(-2, -1))
    )
    symmetric = jnp.max(jnp.abs(safe - transpose), axis=(-2, -1)) <= tolerance
    sym = (safe + transpose) / 2
    eigenvalues = jnp.linalg.eigvalsh(sym)
    magnitudes = jnp.abs(eigenvalues)
    cutoff = 10 * p * jnp.finfo(sym.dtype).eps * magnitudes.max(axis=-1, keepdims=True)
    rank = jnp.sum(magnitudes > cutoff, axis=-1)
    available = finite & symmetric & (rank == p) & (eigenvalues[..., 0] > 0) & eligible
    matrix = jnp.where(available[..., None, None], sym, jnp.eye(p, dtype=sym.dtype))
    inverse = jnp.linalg.solve(
        matrix, jnp.broadcast_to(jnp.eye(p, dtype=sym.dtype), matrix.shape)
    )
    available &= jnp.all(jnp.isfinite(inverse), axis=(-2, -1))
    values = jnp.where(available[..., None, None], inverse, jnp.nan)
    return values, jnp.where(finite, rank, -1), available
