# ruff: noqa: F722  # jaxtyping numerical carrier contracts.
"""Uncertainty attached to the complete linear fit that produced its influences.

These preview providers preserve the actual retained rows, parameter order and
precision weights by consuming a :class:`LinearFit` and returning it inside the
result. Covariance correction and reference-distribution choices are separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Generic, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Int  # noqa: TC002

from statadic.core.equations import (  # noqa: TC001
    IndependentRows,
    OneWayCluster,
    ParameterCovariance,
    PlainLongRun,
    VariabilityT,
    _parameter_covariance_from_limiting,
)
from statadic.core.results import InferenceStatus
from statadic.core.typing import Array, ArrayLike, static_jaxtyping  # noqa: TC001
from statadic.estimation.linear import (  # noqa: TC001
    LinearEquation,
    LinearFit,
)
from statadic.inference._row_order import _validated_row_order
from statadic.operators.parameters import (
    cluster_parameter_covariance,
    coefficient_influence_rows,
    hac_parameter_covariance,
    independent_parameter_covariance,
)

__all__ = [
    "LinearUncertainty",
    "classical_covariance",
    "cluster_covariance",
    "hac_covariance",
    "hc_covariance",
]

_MIN_CLUSTERS = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LinearUncertainty(
    Generic[VariabilityT]  # noqa: UP046
):  # retain scheme variance
    """A sampling covariance and availability evidence tied to one actual fit.

    ``valid`` requires positive marginal variances. A singular covariance with
    positive diagonal may support marginal intervals; joint restrictions still
    require a separate projected-covariance rank check. ``n_clusters`` is the
    occupied cluster count, or zero for a noncluster provider. No reference
    distribution or automatic reference degrees of freedom are selected here.
    """

    fit: LinearFit
    covariance: ParameterCovariance[LinearEquation, VariabilityT]
    status: Int[Array, "*batch"]
    valid: Bool[Array, "*batch"]
    method: str = field(metadata={"static": True})
    n_clusters: Int[Array, "*batch"]

    @property
    def sample_capacity(self) -> int:
        """Return static retained capacity, distinct from effective contributions."""
        return self.fit.outcome.shape[-1]


def _weighted_residuals(fit: LinearFit) -> Array:
    """Return the whitened residuals used by the fitted QR system."""
    return jnp.sqrt(fit.relative_precision.astype(fit.parameters.dtype)) * fit.residuals


def _exact_fit(
    fit: LinearFit,
) -> Bool[Array, ""]:
    """Classify residuals indistinguishable from backward-error roundoff.

    Each prediction is a width-``p`` dot product followed by one subtraction.
    The floor is ``8 * gamma_(p+1) * ||sqrt(w) * (|y| + |X| |beta|)||_2``,
    with ``gamma_k = k*eps/(1-k*eps)``. It follows dtype, rowwise prediction
    magnitudes and relative-precision scale. Observation count enters only
    through the norm, rather than an additional multiplier that could classify
    resolved small noise as an exact fit in a large sample. A point fit can
    remain valid when ordinary covariance is unavailable at this resolution.
    """
    root_weights = jnp.sqrt(fit.relative_precision.astype(fit.parameters.dtype))
    magnitude = jnp.abs(fit.outcome) + jnp.abs(fit.design) @ jnp.abs(fit.parameters)
    relative_error = (fit.design.shape[1] + 1) * jnp.finfo(fit.parameters.dtype).eps
    gamma = relative_error / (1.0 - relative_error)
    floor = 8.0 * gamma * jnp.linalg.norm(root_weights * magnitude)
    return jnp.linalg.norm(_weighted_residuals(fit)) <= floor


def _assemble_uncertainty[
    VariabilityA: IndependentRows | PlainLongRun | OneWayCluster,
](
    fit: LinearFit,
    sampling_covariance: Array,
    correction: Array,
    method: str,
    *,
    extra_valid: Array,
    n_clusters: Array,
    too_few_clusters: Array,
) -> LinearUncertainty[VariabilityA]:
    """Attach scaling and statuses without inventing fit-instance identifiers."""
    exact = _exact_fit(fit)
    eligible = fit.valid & (fit.df_resid > 0) & ~exact & extra_valid
    covariance: ParameterCovariance[LinearEquation, VariabilityA] = (
        _parameter_covariance_from_limiting(
            sampling_covariance * fit.n_observations,
            fit.n_observations,
            correction,
            fit.rank,
            fit.identified,
            eligible,
        )
    )
    positive_variances = jnp.all(jnp.diag(covariance.values) > 0.0)
    valid = covariance.numerically_valid & positive_variances
    status = jnp.where(
        valid, InferenceStatus.AVAILABLE, InferenceStatus.INVALID_COVARIANCE
    )
    status = jnp.where(too_few_clusters, InferenceStatus.TOO_FEW_CLUSTERS, status)
    status = jnp.where(exact, InferenceStatus.EXACT_FIT, status)
    status = jnp.where(fit.df_resid > 0, status, InferenceStatus.INSUFFICIENT_DF)
    status = jnp.where(fit.valid, status, InferenceStatus.INVALID_FIT)
    return LinearUncertainty(
        fit=fit,
        covariance=covariance,
        status=jnp.asarray(status, dtype=jnp.int32),
        valid=valid,
        method=method,
        n_clusters=jnp.asarray(n_clusters, dtype=jnp.int32),
    )


def _correction(fit: LinearFit, correction: float | Array) -> Array:
    """Keep a positive finite-sample multiplier separate from covariance shape."""
    value = jnp.asarray(correction, dtype=fit.parameters.dtype)
    if value.ndim != 0:
        message = "finite_sample_correction must be scalar within one fit."
        raise ValueError(message)
    return value


def classical_covariance(
    fit: LinearFit,
    *,
    finite_sample_correction: float | Array = 1.0,
) -> LinearUncertainty[IndependentRows]:
    """Estimate classical OLS/relative-precision WLS sampling covariance.

    Assumes independent rows with ``Var(error_i | X) = sigma² / w_i``. The
    estimated common scale is weighted SSE divided by ``n-p``; covariance is
    ``fit.common_scale * fit.normalized_covariance``. The recorded correction
    is ``n/(n-p)`` times any explicitly supplied additional multiplier. This
    scale model does not mean frequency or survey weights.
    """
    count = fit.n_observations.astype(fit.parameters.dtype)
    correction = (count / jnp.maximum(fit.df_resid, 1)) * _correction(
        fit, finite_sample_correction
    )
    sampling = (fit.weighted_sse / jnp.maximum(count, 1)) * fit.normalized_covariance
    return _assemble_uncertainty(
        fit,
        sampling,
        correction,
        "classical",
        extra_valid=jnp.ones((), dtype=bool),
        n_clusters=jnp.asarray(0),
        too_few_clusters=jnp.zeros((), dtype=bool),
    )


def hc_covariance(
    fit: LinearFit,
    *,
    kind: Literal["HC0", "HC1", "HC2", "HC3"] = "HC0",
    finite_sample_correction: float | Array = 1.0,
) -> LinearUncertainty[IndependentRows]:
    """Estimate HC0/HC1/HC2/HC3 covariance from the fit's whitened influences.

    HC1 records the explicit ``n/(n-p)`` multiplier on HC0. HC2 divides row
    outer products by ``1-h_i`` and HC3 by ``(1-h_i)²``. Any additional positive
    correction is recorded separately from count normalization; no normal or
    Student-t reference choice follows from selecting HC1.
    """
    if kind not in {"HC0", "HC1", "HC2", "HC3"}:
        message = "kind must be HC0, HC1, HC2 or HC3."
        raise ValueError(message)
    covariance, status, valid = _hc_fields(fit, kind, finite_sample_correction)
    return LinearUncertainty(
        fit, covariance, status, valid, kind, jnp.asarray(0, dtype=jnp.int32)
    )


@partial(jax.jit, static_argnames=("kind",))
def _hc_fields(
    fit: LinearFit,
    kind: Literal["HC0", "HC1", "HC2", "HC3"],
    finite_sample_correction: float | Array,
) -> tuple[ParameterCovariance[LinearEquation, IndependentRows], Array, Array]:
    """Fuse influences, their Gram and diagnostics; retain fit identity in the host."""
    influences = coefficient_influence_rows(
        fit.coefficient_map, _weighted_residuals(fit)
    )
    sampling = independent_parameter_covariance(
        influences, fit.leverage, kind="HC0" if kind == "HC1" else kind
    )
    correction = _correction(fit, finite_sample_correction)
    if kind == "HC1":
        correction = correction * fit.n_observations / jnp.maximum(fit.df_resid, 1)
    valid_leverage = (
        jnp.all((fit.leverage >= 0.0) & (fit.leverage < 1.0))
        if kind in {"HC2", "HC3"}
        else jnp.ones((), dtype=bool)
    )
    result: LinearUncertainty[IndependentRows] = _assemble_uncertainty(
        fit,
        sampling,
        correction,
        kind,
        extra_valid=valid_leverage,
        n_clusters=jnp.asarray(0),
        too_few_clusters=jnp.zeros((), dtype=bool),
    )
    return result.covariance, result.status, result.valid


def hac_covariance(
    fit: LinearFit,
    bandwidth: int,
    *,
    row_order: ArrayLike | None = None,
    finite_sample_correction: float | Array = 1.0,
) -> LinearUncertainty[PlainLongRun]:
    """Apply the existing uncentered Bartlett HAC to ordered fit influences.

    One contribution is one retained observation row. ``row_order`` optionally
    selects their adjacency; otherwise the retained order is used. Lags count
    observation positions, not elapsed calendar intervals.
    ``bandwidth`` is an explicit static integer in ``[0,n)``. Every lag uses
    the same denominator ``n`` inside the shared kernel; the influence scaling
    converts that kernel to parameter sampling covariance. The default
    finite-sample correction is one.
    """
    influences = coefficient_influence_rows(
        fit.coefficient_map, _weighted_residuals(fit)
    )
    order, order_valid = _validated_row_order(row_order, influences.shape[0])
    sampling = hac_parameter_covariance(influences[order], bandwidth)
    return _assemble_uncertainty(
        fit,
        sampling,
        _correction(fit, finite_sample_correction),
        "HAC",
        extra_valid=order_valid,
        n_clusters=jnp.asarray(0),
        too_few_clusters=jnp.zeros((), dtype=bool),
    )


@static_jaxtyping("observations")
def cluster_covariance(
    fit: LinearFit,
    groups: Int[Array, " observations"],
    *,
    n_clusters: int,
    finite_sample_correction: float | Array = 1.0,
) -> LinearUncertainty[OneWayCluster]:
    """Estimate one-way clustered covariance with explicit contiguous labels.

    Rows share a cluster exactly when their integer labels match. All labels
    ``0,...,n_clusters-1`` must occur, and at least two clusters must be
    occupied. Invalid labels/groups retain unavailable status under JIT/vmap.
    ``n_clusters`` is a positive static shape. The default correction is one;
    e.g. ``G/(G-1)`` is an explicit caller choice, separate from a reference
    distribution and its degrees of freedom.
    """
    if (
        isinstance(n_clusters, bool)
        or not isinstance(n_clusters, int)
        or n_clusters < 1
    ):
        message = "n_clusters must be a positive static integer."
        raise ValueError(message)
    if groups.shape != (fit.design.shape[0],) or not jnp.issubdtype(
        groups.dtype, jnp.integer
    ):
        message = "Integer group labels must align with the fit rows."
        raise ValueError(message)
    labels_valid = jnp.all((groups >= 0) & (groups < n_clusters))
    safe_groups = jnp.clip(groups, 0, n_clusters - 1)
    counts = jnp.bincount(safe_groups, length=n_clusters)
    occupied = jnp.sum(counts > 0)
    groups_valid = labels_valid & jnp.all(counts > 0)
    influences = coefficient_influence_rows(
        fit.coefficient_map, _weighted_residuals(fit)
    )
    sampling = cluster_parameter_covariance(
        influences, safe_groups, n_clusters=n_clusters
    )
    return _assemble_uncertainty(
        fit,
        sampling,
        _correction(fit, finite_sample_correction),
        "cluster",
        extra_valid=groups_valid & (occupied >= _MIN_CLUSTERS),
        n_clusters=occupied,
        too_few_clusters=labels_valid & (occupied < _MIN_CLUSTERS),
    )
