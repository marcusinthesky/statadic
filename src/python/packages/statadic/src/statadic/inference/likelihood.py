"""Observed-information and independent-row sandwich likelihood covariance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp

from statadic.core.results import InferenceStatus
from statadic.core.typing import Array  # noqa: TC001
from statadic.estimation.likelihood import LikelihoodFit  # noqa: TC001

__all__ = ["LikelihoodCovariance", "LikelihoodUncertainty", "likelihood_covariance"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LikelihoodCovariance:
    """Covariance and its effective number of independent observations."""

    values: Array
    n_observations: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LikelihoodUncertainty:
    """Fit-attached covariance accepted by the shared inference interface."""

    fit: LikelihoodFit
    covariance: LikelihoodCovariance
    valid: Array
    status: Array
    method: str = field(metadata={"static": True})

    @property
    def sample_capacity(self) -> int:
        """Return retained row capacity."""
        return self.fit.scores.shape[0]


def likelihood_covariance(
    fit: LikelihoodFit,
    *,
    method: Literal["model", "HC0", "HC1"] = "model",
    finite_sample_correction: float = 1.0,
) -> LikelihoodUncertainty:
    """Use the joint observed Hessian and, for HC, full joint score rows."""
    if method not in {"model", "HC0", "HC1"}:
        message = "Covariance method must be model, HC0 or HC1."
        raise ValueError(message)
    correction = jnp.asarray(finite_sample_correction)
    inverse = jnp.linalg.inv(fit.hessian)
    covariance = (
        inverse if method == "model" else inverse @ fit.scores.T @ fit.scores @ inverse
    )
    if method == "HC1":
        covariance = covariance * fit.n_observations / jnp.maximum(fit.df_resid, 1)
    covariance = covariance * correction
    valid = (
        fit.valid
        & jnp.isfinite(correction)
        & (correction > 0)
        & jnp.all(jnp.isfinite(covariance))
        & jnp.all(jnp.diag(covariance) > 0)
    )
    if method == "HC1":
        valid &= fit.df_resid > 0
    status = jnp.where(
        valid, InferenceStatus.AVAILABLE, InferenceStatus.INVALID_COVARIANCE
    )
    status = jnp.where(fit.valid, status, InferenceStatus.INVALID_FIT)
    return LikelihoodUncertainty(
        fit,
        LikelihoodCovariance(jnp.where(valid, covariance, jnp.nan), fit.n_observations),
        valid,
        status,
        method,
    )
