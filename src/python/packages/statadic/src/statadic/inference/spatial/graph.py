"""Observed information conditional on retained, fixed Gaussian graph roles."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from statadic.core.results import InferenceStatus
from statadic.core.typing import Array  # noqa: TC001
from statadic.estimation.spatial.graph_likelihood import (
    GraphMLFit,  # noqa: TC001
    graph_log_density,
)
from statadic.inference.likelihood import LikelihoodCovariance

__all__: list[str] = []


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _GraphUncertainty:
    """Actual fixed-graph fit with its independently selected uncertainty."""

    fit: GraphMLFit
    covariance: LikelihoodCovariance
    valid: Array
    status: Array
    sample_capacity: int = field(metadata={"static": True})


def _loss(
    theta: Array,
    lag: tuple[Array, ...],
    error: tuple[Array, ...],
    design: tuple[Array, ...],
    outcome: tuple[Array, ...],
    observed: tuple[Array, ...] | None,
) -> Array:
    width = design[0].shape[1]
    return -graph_log_density(
        lag,
        error,
        design,
        outcome,
        theta[:width],
        theta[width:-1],
        jnp.exp(theta[-1]),
        observed=observed,
    )


_INFORMATION = jax.jit(jax.hessian(_loss))


def _graph_covariance(
    fit: GraphMLFit,
    lag: tuple[Array, ...],
    error: tuple[Array, ...],
    design: tuple[Array, ...],
    outcome: tuple[Array, ...],
    *,
    observed: tuple[Array, ...] | None = None,
) -> _GraphUncertainty:
    """Marginalize log innovation variance from total observed information.

    Graph matrices and the supplied design, including WX columns, are fixed.
    This does not infer uncertainty for estimated graph recipe parameters.
    """
    theta = jnp.append(fit.parameters, jnp.log(fit.sigma2))
    info = _INFORMATION(theta, lag, error, design, outcome, observed)
    sym = (info + info.T) / 2
    finite = jnp.all(jnp.isfinite(sym))
    safe = jnp.where(finite, sym, jnp.eye(theta.size, dtype=theta.dtype))
    valid = (
        fit.valid
        & finite
        & (jnp.linalg.eigvalsh(safe)[0] > 0)
        & (jnp.linalg.matrix_rank(safe) == theta.size)
    )
    inverse = jnp.linalg.inv(safe)[:-1, :-1]
    valid &= jnp.all(jnp.isfinite(inverse))
    n = sum(y.size for y in outcome)
    status = jnp.where(
        fit.valid,
        jnp.where(valid, InferenceStatus.AVAILABLE, InferenceStatus.INVALID_COVARIANCE),
        InferenceStatus.INVALID_FIT,
    )
    return _GraphUncertainty(
        fit,
        LikelihoodCovariance(jnp.where(valid, inverse, jnp.nan), jnp.asarray(n)),
        valid,
        status,
        n,
    )
