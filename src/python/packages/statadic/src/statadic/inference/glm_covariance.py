# ruff: noqa: F722  # jaxtyping numerical carrier contracts.
"""Fit-attached canonical GLM covariance through the shared mean-equation composer."""

# jaxtyping numerical carrier contracts.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal, overload

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Int  # noqa: TC002

from statadic.core.equations import (  # noqa: TC001
    IndependentRows,
    MeanEquationJacobian,
    ParameterCovariance,
    VariabilityMatrix,
    parameter_covariance,
    unsafe_assume_variability_matrix,
)
from statadic.core.results import InferenceStatus
from statadic.core.typing import Array  # noqa: TC001
from statadic.estimation.glm import (
    GLMFit,  # noqa: TC001
)
from statadic.model.glm import GLMEquation

__all__ = ["GLMUncertainty", "glm_covariance"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GLMUncertainty[EquationT: GLMEquation]:
    """A sampling covariance and availability evidence attached to its actual fit."""

    fit: GLMFit[EquationT]
    covariance: ParameterCovariance[EquationT, IndependentRows]
    valid: Bool[Array, "*batch"]
    status: Int[Array, "*batch"]
    method: str = field(metadata={"static": True})
    n_clusters: Int[Array, "*batch"]

    @property
    def sample_capacity(self) -> int:
        """Return static retained capacity, distinct from effective contributions."""
        return self.fit.scores.values.shape[-2]


@overload
def glm_covariance[EquationT: GLMEquation](
    fit: GLMFit[EquationT],
    *,
    method: Literal["model", "HC0", "HC1"] = "model",
    finite_sample_correction: float | Array = 1.0,
) -> GLMUncertainty[EquationT]: ...


@overload
def glm_covariance[LeftT: GLMEquation, RightT: GLMEquation](
    fit: GLMFit[LeftT] | GLMFit[RightT],
    *,
    method: Literal["model", "HC0", "HC1"] = "model",
    finite_sample_correction: float | Array = 1.0,
) -> GLMUncertainty[LeftT] | GLMUncertainty[RightT]: ...


def glm_covariance[EquationT: GLMEquation](
    fit: GLMFit[EquationT],
    *,
    method: Literal["model", "HC0", "HC1"] = "model",
    finite_sample_correction: float | Array = 1.0,
) -> GLMUncertainty[EquationT]:
    """Compose canonical model or independent-row HC parameter sampling covariance.

    Model covariance uses the information equality with fixed dispersion one.
    HC0 uses the uncentered outer product of the actual fitted score rows,
    divided by retained row count. HC1 adds ``n/(n-p)``. The same shared
    aggregate-Jacobian sandwich divides limiting covariance by ``n``; a supplied
    positive finite-sample multiplier remains separate from reference choice.
    No normal/t reference or likelihood-ratio validity follows from this call.
    """
    if method not in {"model", "HC0", "HC1"}:
        message = "method must be model, HC0 or HC1."
        raise ValueError(message)
    count = fit.n_observations
    correction = jnp.asarray(finite_sample_correction, dtype=fit.parameters.dtype)
    if correction.ndim != 0:
        message = "finite_sample_correction must be scalar within one fit."
        raise ValueError(message)
    covariance, valid, status = _glm_covariance(
        fit.jacobian,
        fit.scores.values,
        count,
        fit.df_resid,
        fit.valid,
        correction,
        method=method,
    )
    return GLMUncertainty(
        fit=fit,
        covariance=covariance,
        valid=valid,
        status=jnp.asarray(status, dtype=jnp.int32),
        method=method,
        n_clusters=jnp.asarray(0, dtype=jnp.int32),
    )


@partial(jax.jit, static_argnames=("method",))
def _glm_covariance[EquationT: GLMEquation](
    jacobian: MeanEquationJacobian[EquationT],
    scores: Array,
    count: Array,
    df_resid: Array,
    fit_valid: Array,
    correction: Array,
    *,
    method: Literal["model", "HC0", "HC1"],
) -> tuple[ParameterCovariance[EquationT, IndependentRows], Array, Array]:
    """Compute covariance evidence without passing the complete fit through XLA."""
    omega = -jacobian.values if method == "model" else (scores.T @ scores / count)
    if method == "HC1":
        correction = correction * count / jnp.maximum(df_resid, 1)
    meat: VariabilityMatrix[EquationT, IndependentRows] = (
        unsafe_assume_variability_matrix(omega)
    )
    covariance = parameter_covariance(
        jacobian,
        meat,
        count,
        jnp.eye(jacobian.values.shape[-1], dtype=jacobian.values.dtype),
        finite_sample_correction=correction,
    )
    eligible = fit_valid & ((df_resid > 0) if method == "HC1" else True)
    valid = (
        eligible
        & covariance.numerically_valid
        & jnp.all(jnp.diag(covariance.values) > 0)
    )
    covariance = replace(
        covariance,
        values=jnp.where(valid, covariance.values, jnp.nan),
        numerically_valid=valid,
    )
    status = jnp.where(
        valid, InferenceStatus.AVAILABLE, InferenceStatus.INVALID_COVARIANCE
    )
    if method == "HC1":
        status = jnp.where(df_resid > 0, status, InferenceStatus.INSUFFICIENT_DF)
    status = jnp.where(fit_valid, status, InferenceStatus.INVALID_FIT)
    return covariance, valid, status
