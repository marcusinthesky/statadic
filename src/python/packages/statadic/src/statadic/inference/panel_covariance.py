# ruff: noqa: F722  # jaxtyping numerical carrier contracts.
"""Static-panel sampling covariance with explicit dependence and corrections.

Covariance consumes the actual transformed fit; no coefficient refit occurs.
Effect degrees of freedom, slope debiasing and cluster debiasing are separate
choices. Kernels use actual integer period distances, never compressed gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Int  # noqa: TC002

from statadic.core.equations import (  # noqa: TC001
    EquationDescriptor,
    ParameterCovariance,
    VariabilityScheme,
    _parameter_covariance_from_limiting,
)
from statadic.core.results import InferenceStatus
from statadic.core.typing import Array  # noqa: TC001
from statadic.estimation.panel import (  # noqa: TC001
    FamaMacBethFit,
    PanelFit,
)
from statadic.operators.parameters import cluster_parameter_covariance

__all__ = [
    "PanelEquation",
    "PanelUncertainty",
    "fama_macbeth_covariance",
    "panel_covariance",
]

_MIN_CLUSTERS = 2


_PARZEN_BREAKPOINT = 0.5


class PanelEquation(EquationDescriptor):
    """Fit-owned panel coefficient equation after its declared transformation."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PanelUncertainty:
    """Sampling covariance attached to a panel fit and explicit availability."""

    fit: PanelFit | FamaMacBethFit
    covariance: ParameterCovariance[PanelEquation, VariabilityScheme]
    valid: Bool[Array, "*batch"]
    status: Int[Array, "*batch"]
    n_clusters: Int[Array, "*batch"]
    method: str = field(metadata={"static": True})

    @property
    def sample_capacity(self) -> int:
        """Return static retained capacity, distinct from effective contributions."""
        if isinstance(self.fit, FamaMacBethFit):
            return self.fit.period_parameters.shape[-2]
        return self.fit.linear.outcome.shape[-1]


def _kernel(distances: Array, bandwidth: int, kernel: str) -> Array:
    """Evaluate a named HAC window on nonnegative integer time distances."""
    if kernel not in ("bartlett", "parzen", "qs"):
        message = "kernel must be bartlett, parzen or qs."
        raise ValueError(message)
    if bandwidth < 0:
        message = "bandwidth must be nonnegative."
        raise ValueError(message)
    if bandwidth == 0:
        return (distances == 0).astype(jnp.result_type(distances, 1.0))
    if kernel == "bartlett":
        return jnp.maximum(1 - distances / (bandwidth + 1), 0)
    if kernel == "parzen":
        x = distances / (bandwidth + 1)
        return jnp.where(
            x <= _PARZEN_BREAKPOINT,
            1 - 6 * x * x + 6 * x * x * x,
            jnp.where(x <= 1, 2 * (1 - x) ** 3, 0),
        )
    if kernel == "qs":
        x = distances / bandwidth
        z = 6 * jnp.pi * jnp.where(x == 0, 1, x) / 5
        return jnp.where(x == 0, 1, 3 * (jnp.sin(z) / z - jnp.cos(z)) / (z * z))
    message = "kernel must be 'bartlett', 'parzen' or 'qs'."
    raise ValueError(message)


def _cluster_meat(
    influences: Array, groups: Array, n_groups: int, *, group_debias: bool
) -> Array:
    """Sum coefficient influences by group with an optional finite-G correction."""
    counts = jnp.bincount(groups, length=n_groups)
    occupied = jnp.sum(counts > 0).astype(influences.dtype)
    correction = (
        occupied
        / jnp.maximum(occupied - 1, 1)
        * (influences.shape[0] - 1)
        / influences.shape[0]
        if group_debias
        else 1
    )
    return (
        cluster_parameter_covariance(influences, groups, n_clusters=n_groups)
        * correction
    )


def _wrap(
    fit: PanelFit | FamaMacBethFit,
    sampling: Array,
    correction: Array,
    *,
    method: str,
    count: Array,
    rank: Array,
    valid: Array,
    n_clusters: Array,
) -> PanelUncertainty:
    """Brand one actual-fit sampling covariance without selecting a reference law."""
    covariance: ParameterCovariance[PanelEquation, VariabilityScheme] = (
        _parameter_covariance_from_limiting(
            sampling * count,
            count,
            correction,
            rank,
            fit.valid,
            valid,
        )
    )
    available = (
        valid
        & fit.valid
        & covariance.numerically_valid
        & jnp.all(jnp.diag(covariance.values) > 0)
    )
    status = jnp.where(
        available, InferenceStatus.AVAILABLE, InferenceStatus.INVALID_COVARIANCE
    )
    status = jnp.where(fit.valid, status, InferenceStatus.INVALID_FIT)
    return PanelUncertainty(
        fit,
        covariance,
        available,
        jnp.asarray(status, dtype=jnp.int32),
        n_clusters,
        method,
    )


def _panel_cluster_covariance(
    fit: PanelFit, influences: Array, *, method: str, group_debias: bool
) -> tuple[Array, Array]:
    """Combine entity/time clusters using the unique retained intersection rows."""
    entity_count = fit.index.n_entities
    time_count = fit.index.n_periods
    entities = jnp.sum(jnp.bincount(fit.row_entities, length=entity_count) > 0)
    periods = jnp.sum(jnp.bincount(fit.row_times, length=time_count) > 0)
    if method == "cluster-entity":
        return _cluster_meat(
            influences, fit.row_entities, entity_count, group_debias=group_debias
        ), entities
    if method == "cluster-time":
        return _cluster_meat(
            influences, fit.row_times, time_count, group_debias=group_debias
        ), periods
    if method != "cluster-two-way":
        message = "Unknown panel clustering method."
        raise ValueError(message)
    # Prepared panel keys are unique, so intersection clusters are single rows.
    sampling = _cluster_meat(
        influences, fit.row_entities, entity_count, group_debias=group_debias
    )
    sampling += _cluster_meat(
        influences, fit.row_times, time_count, group_debias=group_debias
    )
    return sampling - influences.T @ influences, jnp.minimum(entities, periods)


def _panel_time_covariance(
    fit: PanelFit, influences: Array, *, method: str, bandwidth: int, kernel: str
) -> Array:
    """Apply actual-period HAC either after pooling or within each entity."""
    if fit.method == "between":
        message = "Between estimation has no within-entity time covariance."
        raise ValueError(message)
    times = fit.index.time_ids
    if method == "driscoll-kraay":
        sums = (
            jnp.zeros(
                (fit.index.n_periods, influences.shape[1]), dtype=influences.dtype
            )
            .at[fit.row_times]
            .add(influences)
        )
        return _time_cross_product(sums, times, bandwidth, kernel)
    # Keep only integer row positions on the entity/time grid. Each observed
    # row gathers its entity's contributions; there is no E*T*p score cube or
    # T*T window. QS still evaluates every retained period distance.
    positions = jnp.full(
        (fit.index.n_entities, fit.index.n_periods), -1, dtype=jnp.int32
    )
    positions = positions.at[fit.row_entities, fit.row_times].set(
        jnp.arange(influences.shape[0], dtype=positions.dtype)
    )

    def add_row(row: Int[Array, ""] | int, total: Array) -> Array:
        other = positions[fit.row_entities[row]]
        weights = _kernel(
            jnp.abs(times - times[fit.row_times[row]]), bandwidth, kernel
        ).astype(influences.dtype)
        weights = jnp.where(other >= 0, weights, 0.0)
        weighted_sum = weights @ influences[jnp.maximum(other, 0)]
        return total + jnp.outer(influences[row], weighted_sum)

    initial = jnp.zeros(
        (influences.shape[1], influences.shape[1]), dtype=influences.dtype
    )
    return jax.lax.fori_loop(0, influences.shape[0], add_row, initial)


def _time_cross_product(
    values: Array, times: Array, bandwidth: int, kernel: str
) -> Array:
    """Stream exact actual-time HAC products without a square time window."""

    def add_row(row: Int[Array, ""] | int, total: Array) -> Array:
        weights = _kernel(jnp.abs(times - times[row]), bandwidth, kernel).astype(
            values.dtype
        )
        return total + jnp.outer(values[row], weights @ values)

    initial = jnp.zeros((values.shape[1], values.shape[1]), dtype=values.dtype)
    return jax.lax.fori_loop(0, values.shape[0], add_row, initial)


def panel_covariance(
    fit: PanelFit,
    *,
    method: Literal[
        "classical",
        "heteroskedastic",
        "cluster-entity",
        "cluster-time",
        "cluster-two-way",
        "driscoll-kraay",
        "autocorrelation",
    ] = "classical",
    debiased: bool = True,
    count_effects: bool = True,
    group_debias: bool = False,
    bandwidth: int = 0,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
) -> PanelUncertainty:
    """Estimate covariance with explicit slope/effect/cluster corrections.

    Driscoll-Kraay aggregates influences by time before HAC. Autocorrelation
    covariance sums within-entity HAC score products without reweighting short
    entities. The declared bandwidth is retained even for short entities.
    One-/two-way clustering uses the retained entity/time grouping. Between
    models have no time covariance, and first differences have transformed
    row groups; methods unavailable for those samples are rejected explicitly.
    """
    linear = fit.linear
    n, p = linear.design.shape
    count = jnp.asarray(n, dtype=jnp.int32)
    absorbed = fit.absorbed_df if count_effects else jnp.asarray(0)
    denominator = n - absorbed - (p if debiased else 0)
    correction = jnp.asarray(n, dtype=linear.parameters.dtype) / jnp.maximum(
        denominator, 1
    ).astype(linear.parameters.dtype)
    residual = (
        jnp.sqrt(linear.relative_precision.astype(linear.parameters.dtype))
        * linear.residuals
    )
    influences = linear.coefficient_map.T * residual[:, None]
    clusters = jnp.asarray(0, dtype=jnp.int32)
    valid = fit.valid & (denominator > 0)
    if method == "classical":
        sampling = jnp.sum(residual**2) / n * linear.normalized_covariance
    elif method == "heteroskedastic":
        sampling = influences.T @ influences
    elif method.startswith("cluster"):
        sampling, clusters = _panel_cluster_covariance(
            fit, influences, method=method, group_debias=group_debias
        )
        valid &= clusters >= _MIN_CLUSTERS
    elif method in ("driscoll-kraay", "autocorrelation"):
        sampling = _panel_time_covariance(
            fit, influences, method=method, bandwidth=bandwidth, kernel=kernel
        )
    else:
        message = "Unknown panel covariance method."
        raise ValueError(message)
    return _wrap(
        fit,
        sampling,
        correction,
        method=method,
        count=count,
        rank=linear.rank,
        valid=valid,
        n_clusters=clusters,
    )


def fama_macbeth_covariance(
    fit: FamaMacBethFit,
    *,
    bandwidth: int = 0,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    debiased: bool = True,
) -> PanelUncertainty:
    """Estimate time-series covariance of identified cross-section coefficients."""
    count = jnp.sum(fit.period_valid).astype(jnp.int32)
    residuals = jnp.where(
        fit.period_valid[:, None], fit.period_parameters - fit.parameters, 0
    )
    periods = fit.index.time_ids
    sampling = (
        _time_cross_product(residuals, periods, bandwidth, kernel)
        / jnp.maximum(count, 1) ** 2
    )
    floating_count = count.astype(sampling.dtype)
    correction = (
        floating_count / jnp.maximum(floating_count - 1, 1)
        if debiased
        else jnp.asarray(1.0, dtype=sampling.dtype)
    )
    return _wrap(
        fit,
        sampling,
        correction,
        method="Fama-MacBeth-HAC",
        count=count,
        rank=jnp.asarray(fit.parameters.shape[-1], dtype=jnp.int32),
        valid=fit.valid & (count >= _MIN_CLUSTERS),
        n_clusters=jnp.asarray(0, dtype=jnp.int32),
    )
