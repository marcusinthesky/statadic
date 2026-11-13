# jaxtyping dimensions are runtime contracts.
"""Fit-derived spatial uncertainty and exact dense covariate impacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from statadic.core.equations import (  # noqa: TC001
    EquationDescriptor,
    ParameterCovariance,
    VariabilityScheme,
    _parameter_covariance_from_limiting,
)
from statadic.core.results import InferenceStatus
from statadic.core.typing import Array  # noqa: TC001
from statadic.estimation.spatial import (
    SpatialFit,
    SpatialIVFit,
)
from statadic.estimation.spatial.gmm import SpatialGMMFit
from statadic.estimation.spatial.observed_panel import ObservedSpatialPanelFit
from statadic.estimation.spatial.panel import SpatialPanelFit
from statadic.inference.gmm.covariance import gmm_covariance
from statadic.operators.spatial.moments import spatial_moment_variability

__all__ = [
    "SpatialEquation",
    "SpatialImpacts",
    "SpatialPanelImpacts",
    "SpatialUncertainty",
    "spatial_covariance",
    "spatial_impacts",
    "spatial_panel_impacts",
]


type _SpatialFit = (
    SpatialFit
    | SpatialIVFit
    | SpatialGMMFit
    | SpatialPanelFit
    | ObservedSpatialPanelFit
)
type _CovarianceMethod = Literal[
    "expected", "observed", "homoskedastic", "heteroskedastic"
]


class SpatialEquation(EquationDescriptor):
    """Actual spatial fit and its explicitly declared uncertainty calculation."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpatialUncertainty:
    """Sampling covariance attached to the actual spatial fit."""

    fit: _SpatialFit
    covariance: ParameterCovariance[SpatialEquation, VariabilityScheme]
    valid: Array
    status: Array
    n_clusters: Array
    method: str = field(metadata={"static": True})

    @property
    def sample_capacity(self) -> int:
        """Return static retained capacity, distinct from effective contributions."""
        if isinstance(self.fit, ObservedSpatialPanelFit):
            return sum(
                len(period.observed_indices) for period in self.fit.problem.periods
            )
        if isinstance(self.fit, SpatialPanelFit):
            return self.fit.outcome.shape[-2] * self.fit.outcome.shape[-1]
        return self.fit.outcome.shape[-1]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpatialImpacts:
    """Average direct, indirect and total impacts and optional delta uncertainty.

    Rows follow original design columns before Durbin expansion; columns are
    direct, indirect, total. Uncertainty is separate from point availability.
    The covariance flattens that row-major layout.
    """

    values: Array
    covariance: Array
    standard_errors: Array
    valid: Array
    uncertainty_valid: Array


def _spatial_gmm_sampling(  # noqa: PLR0914
    fit: SpatialGMMFit,
) -> Array:  # named scientific fit/covariance state
    """Construct the matched joint coefficient/lambda GMM covariance."""
    w = fit.weights.values
    if isinstance(w, BCOO):
        message = "Spatial-error quadratic moments require dense weights."
        raise TypeError(message)
    lam = fit.parameters[-1]
    n = fit.outcome.shape[0]
    hetero = fit.variance_model == "heteroskedastic"
    psi, adjustments, mapping = spatial_moment_variability(
        fit.residuals,
        lam,
        w,
        fit.moment_matrices,
        fit.design,
        fit.instruments,
        heteroskedastic=hetero,
        endogenous=fit.endogenous,
    )
    innovations = fit.innovations
    derivative = fit.moment_derivative
    weighted = jnp.linalg.solve(psi, derivative)
    lambda_variance = 1 / (n * (derivative @ weighted))
    sigma2 = jnp.mean(innovations**2)
    if hetero:
        beta_covariance = (mapping * innovations[None, :]) @ (
            mapping * innovations[None, :]
        ).T
        cross = (
            mapping
            @ (innovations[:, None] ** 2 * adjustments)
            @ weighted
            * lambda_variance
        )
    else:
        beta_covariance = sigma2 * mapping @ mapping.T
        mu3 = jnp.mean(innovations**3)
        diagonal = jnp.diagonal(fit.moment_matrices, axis1=1, axis2=2).T
        if fit.endogenous:
            cross = (
                mapping
                @ (sigma2 * adjustments + mu3 * diagonal)
                @ weighted
                * lambda_variance
            )
        else:
            filtered = fit.design - lam * (w @ fit.design)
            raw_map = jnp.linalg.solve(filtered.T @ filtered, fit.design.T)
            cross = raw_map @ (mu3 * diagonal) @ weighted * lambda_variance
    output = jnp.zeros(
        (fit.parameters.shape[0], fit.parameters.shape[0]), dtype=fit.outcome.dtype
    )
    output = output.at[:-1, :-1].set(beta_covariance)
    output = output.at[:-1, -1].set(cross)
    output = output.at[-1, :-1].set(cross)
    return output.at[-1, -1].set(lambda_variance)


def spatial_covariance(
    fit: _SpatialFit, *, method: _CovarianceMethod = "expected", debiased: bool = False
) -> SpatialUncertainty:
    """Compute conventional Gaussian information or actual-fit spatial IV covariance.

    Expected ML information jointly includes the profiled scale nuisance;
    eliminate it only after inversion. IV robust covariance assumes independent
    structural innovations. These methods do not supply weak-IV-robust inference.
    """
    covariance, available, status = _spatial_covariance_state(
        fit, method=method, debiased=debiased
    )
    return SpatialUncertainty(
        fit, covariance, available, status, jnp.asarray(0, dtype=jnp.int32), method
    )


@partial(jax.jit, static_argnames=("method", "debiased"))
def _spatial_covariance_state(  # noqa: C901, PLR0912, PLR0914, PLR0915
    fit: _SpatialFit,
    *,
    method: _CovarianceMethod,
    debiased: bool,
) -> tuple[ParameterCovariance[SpatialEquation, VariabilityScheme], Array, Array]:
    """Compute and validate covariance; the host retains the original fit identity."""
    if isinstance(fit, ObservedSpatialPanelFit):
        n = sum(len(period.observed_indices) for period in fit.problem.periods)
        dtype = fit.parameters.dtype
    else:
        n = fit.outcome.size
        dtype = fit.outcome.dtype
    p = fit.parameters.shape[-1]
    correction = jnp.asarray(n / (n - p) if debiased and n > p else 1.0, dtype=dtype)
    if isinstance(fit, ObservedSpatialPanelFit):
        if method != "observed" or debiased:
            message = (
                "Observed panel covariance requires method='observed' and no debiasing."
            )
            raise ValueError(message)
        sampling = jnp.linalg.inv(fit.information)
    elif isinstance(fit, SpatialPanelFit):
        if method != "expected":
            message = (
                "Spatial panel covariance uses the fit's retained information_method."
            )
            raise ValueError(message)
        nuisance = 2 if fit.effect_type == "random" else 1
        sampling = jnp.linalg.inv(fit.information)[:-nuisance, :-nuisance]
    elif isinstance(fit, SpatialGMMFit):
        if method not in ("expected", fit.variance_model):
            message = "Spatial GMM covariance must match its fitted variance_model."
            raise ValueError(message)
        sampling = _spatial_gmm_sampling(fit)
    elif isinstance(fit, SpatialFit):
        if method != "expected":
            message = (
                "ML uncertainty requires expected information; "
                "heteroskedastic models need valid spatial GMM."
            )
            raise ValueError(message)
        sampling = jnp.linalg.inv(fit.information)[:-1, :-1]
    elif method == "heteroskedastic":
        sampling = gmm_covariance(fit.gmm).covariance.values
    elif method in ("homoskedastic", "expected"):
        z = fit.instruments
        projected = z @ jnp.linalg.lstsq(z, fit.design, rcond=None)[0]
        q, r = jnp.linalg.qr(projected, mode="reduced")
        influence = jnp.linalg.solve(r, q.T)
        sampling = jnp.sum(fit.residuals**2) / n * (influence @ influence.T)
    else:
        message = "Unknown spatial covariance method."
        raise ValueError(message)
    valid = fit.valid & jnp.all(jnp.isfinite(sampling)) & (n > p)
    if isinstance(fit, ObservedSpatialPanelFit):
        valid &= fit.information_valid & ~fit.variance_boundary
    if isinstance(fit, SpatialPanelFit):
        # A variance component on its boundary does not have the ordinary
        # interior joint reference; retain the valid point fit separately.
        valid &= ~fit.variance_boundary
    count = jnp.asarray(n, dtype=jnp.int32)
    covariance: ParameterCovariance[SpatialEquation, VariabilityScheme] = (
        _parameter_covariance_from_limiting(
            sampling * count,
            count,
            correction,
            jnp.asarray(p, dtype=jnp.int32),
            fit.valid,
            valid,
        )
    )
    available = (
        valid & covariance.numerically_valid & jnp.all(jnp.diag(covariance.values) > 0)
    )
    status = jnp.where(
        available, InferenceStatus.AVAILABLE, InferenceStatus.INVALID_COVARIANCE
    )
    status = jnp.where(fit.valid, status, InferenceStatus.INVALID_FIT)
    return covariance, available, jnp.asarray(status, dtype=jnp.int32)


def spatial_impacts(fit: SpatialFit | SpatialUncertainty) -> SpatialImpacts:
    """Compute exact average derivatives and optional actual-fit delta covariance.

    W is fixed. This differentiates the explicit impact map, not the fitted
    solution. No symmetry, row-standardization or scalar-multiplier shortcut
    is assumed. Supply a fit for point impacts or its uncertainty carrier for
    delta covariance. That carrier owns the actual fit under JIT/vmap too.
    """
    uncertainty = fit if isinstance(fit, SpatialUncertainty) else None
    if uncertainty is not None:
        if not isinstance(uncertainty.fit, SpatialFit):
            message = (
                "Exact impact covariance requires a cross-sectional likelihood fit."
            )
            raise TypeError(message)
        fit = uncertainty.fit
    if not isinstance(fit, SpatialFit):
        message = "Exact impacts require a cross-sectional likelihood fit."
        raise TypeError(message)
    if isinstance(fit.weights.values, BCOO):
        message = "Exact impacts require explicitly dense weights."
        raise TypeError(message)
    w = fit.weights.values
    n = w.shape[0]
    base = fit.design.shape[1] - len(fit.slx_columns)
    p = fit.design.shape[1]
    identity = jnp.eye(n, dtype=w.dtype)

    def impact_map(parameters: Array) -> Array:
        rho = parameters[p] if fit.lag else jnp.asarray(0, dtype=w.dtype)
        inverse = jnp.linalg.solve(identity - rho * w, identity)
        trace = jnp.trace(inverse) / n
        total = jnp.sum(inverse) / n
        lag_trace = jnp.trace(inverse @ w) / n
        lag_total = jnp.sum(inverse @ w) / n
        theta = jnp.zeros(base, dtype=w.dtype)
        if fit.slx_columns:
            theta = theta.at[jnp.asarray(fit.slx_columns)].set(parameters[base:p])
        direct = parameters[:base] * trace + theta * lag_trace
        all_effects = parameters[:base] * total + theta * lag_total
        return jnp.stack([direct, all_effects - direct, all_effects], axis=1)

    values = impact_map(fit.parameters)
    if uncertainty is None:
        covariance = jnp.full((base * 3, base * 3), jnp.nan, dtype=w.dtype)
        errors = jnp.full_like(values, jnp.nan)
        uncertainty_valid = jnp.zeros((), dtype=bool)
    else:

        def flat_impacts(parameters: Array) -> Array:
            return impact_map(parameters).reshape(-1)

        jacobian = jax.jacrev(flat_impacts)(fit.parameters)
        covariance = jacobian @ uncertainty.covariance.values @ jacobian.T
        uncertainty_valid = (
            uncertainty.valid
            & jnp.all(jnp.isfinite(covariance))
            & jnp.all(jnp.diag(covariance) >= 0)
        )
        errors = jnp.where(
            uncertainty_valid,
            jnp.sqrt(jnp.maximum(jnp.diag(covariance), 0)).reshape(base, 3),
            jnp.nan,
        )
    return SpatialImpacts(
        values,
        covariance,
        errors,
        fit.valid & jnp.all(jnp.isfinite(values)),
        uncertainty_valid,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpatialPanelImpacts:
    """Per-period active-graph impacts and optional actual-fit delta covariance.

    Rows in each period are base design columns; the final axis is direct,
    indirect, total. Graphs and entity effects are held fixed. These are mean
    derivatives, not causal effects or derivatives of fitted random effects.
    """

    values: Array
    covariance: Array
    standard_errors: Array
    valid: Array
    uncertainty_valid: Array


def spatial_panel_impacts(
    fit: ObservedSpatialPanelFit | SpatialUncertainty,
) -> SpatialPanelImpacts:
    """Evaluate impacts on each actual active graph, with optional delta SEs."""
    uncertainty = fit if isinstance(fit, SpatialUncertainty) else None
    if uncertainty is not None:
        if not isinstance(uncertainty.fit, ObservedSpatialPanelFit):
            message = "Panel impacts require an observed spatial panel fit."
            raise TypeError(message)
        fit = uncertainty.fit
    if not isinstance(fit, ObservedSpatialPanelFit):
        message = "Panel impacts require an observed spatial panel fit."
        raise TypeError(message)
    base = jnp.asarray(fit.problem.design[0]).shape[1]
    width = base + len(fit.slx_columns)

    def impact_map(parameters: Array) -> Array:
        rho = parameters[width] if fit.lag else jnp.zeros((), dtype=parameters.dtype)
        theta = jnp.zeros(base, dtype=parameters.dtype)
        if fit.slx_columns:
            theta = theta.at[jnp.asarray(fit.slx_columns)].set(parameters[base:width])
        outputs = []
        for period in fit.problem.periods:
            w = jnp.asarray(period.weights.values, dtype=parameters.dtype)
            n = w.shape[0]
            inverse = jnp.linalg.solve(
                jnp.eye(n, dtype=parameters.dtype) - rho * w,
                jnp.eye(n, dtype=parameters.dtype),
            )
            trace = jnp.trace(inverse) / n
            total = jnp.sum(inverse) / n
            lag_trace = jnp.trace(inverse @ w) / n
            lag_total = jnp.sum(inverse @ w) / n
            direct = parameters[:base] * trace + theta * lag_trace
            total_effect = parameters[:base] * total + theta * lag_total
            outputs.append(
                jnp.stack((direct, total_effect - direct, total_effect), axis=1)
            )
        return jnp.stack(outputs)

    values = impact_map(fit.parameters)
    size = values.size
    if uncertainty is None:
        covariance = jnp.full((size, size), jnp.nan, dtype=values.dtype)
        errors = jnp.full_like(values, jnp.nan)
        uncertainty_valid = jnp.zeros((), dtype=bool)
    else:

        def flat_impacts(parameters: Array) -> Array:
            return impact_map(parameters).reshape(-1)

        jacobian = jax.jacrev(flat_impacts)(fit.parameters)
        covariance = jacobian @ uncertainty.covariance.values @ jacobian.T
        uncertainty_valid = (
            uncertainty.valid
            & jnp.all(jnp.isfinite(covariance))
            & jnp.all(jnp.diag(covariance) >= 0)
        )
        errors = jnp.where(
            uncertainty_valid,
            jnp.sqrt(jnp.maximum(jnp.diag(covariance), 0)).reshape(values.shape),
            jnp.nan,
        )
    return SpatialPanelImpacts(
        values,
        covariance,
        errors,
        fit.valid & jnp.all(jnp.isfinite(values)),
        uncertainty_valid,
    )
