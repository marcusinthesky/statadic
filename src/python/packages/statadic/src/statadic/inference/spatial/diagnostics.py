# jaxtyping shape strings are runtime contracts.
"""Conventional OLS spatial specification diagnostics with matched nulls.

Robust LM means adjustment for the competing lag/error alternative, not general
robustness to heteroskedasticity, estimated W, weak instruments or misspecification.
All diagnostics condition on one fixed exogenous graph and complete OLS sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jax.scipy.special import ndtr
from jax.scipy.stats import chi2

from statadic.core.results import (
    TestResult,
)
from statadic.core.typing import Array  # noqa: TC001
from statadic.estimation.linear import (
    LinearFit,  # noqa: TC001
)
from statadic.estimation.spatial.iv import (
    SpatialIVFit,  # noqa: TC001
)
from statadic.estimation.spatial.likelihood import SpatialFit
from statadic.estimation.spatial.panel import SpatialPanelFit
from statadic.inference.spatial.covariance import (
    SpatialUncertainty,  # noqa: TC001
)
from statadic.operators.spatial import (
    SpatialWeights,  # noqa: TC001
)
from statadic.operators.spatial.moments import iv_response_map

__all__ = [
    "SpatialDiagnostics",
    "spatial_ak_test",
    "spatial_common_factor_test",
    "spatial_diagnostics",
    "spatial_likelihood_ratio",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpatialDiagnostics:
    """Moran residual normal reference and lag/error chi-square LM diagnostics."""

    moran: TestResult[Array]
    moran_expectation: Array
    moran_variance: Array
    lm_lag: TestResult[Array]
    lm_error: TestResult[Array]
    robust_lm_lag: TestResult[Array]
    robust_lm_error: TestResult[Array]
    lm_sarma: TestResult[Array]
    valid: Array


def _test(statistic: Array, pvalue: Array, valid: Array) -> TestResult[Array]:
    """Retain unavailable calibrations without a fabricated null sample."""
    return TestResult(
        jnp.where(valid, statistic, jnp.nan),
        jnp.where(valid, pvalue, jnp.nan),
        jnp.empty(0, dtype=statistic.dtype),
        0,
    )


def spatial_ak_test(fit: SpatialIVFit) -> TestResult[Array]:
    """Anselin-Kelejian residual spatial-correlation test after spatial IV.

    Uses the general endogenous-regressor/spatial-lag correction, independent
    structural innovations and the actual 2SLS instrument projection. It is
    not a test robust to weak instruments or arbitrary spatial errors.
    """
    w = fit.weights.values
    if isinstance(w, BCOO):
        message = "Exact AK trace calculations require explicitly dense weights."
        raise TypeError(message)
    n = fit.outcome.shape[0]
    u = fit.residuals
    mapping, _ = iv_response_map(fit.design, fit.instruments)
    cross = u @ w @ fit.design @ mapping
    sigma2 = jnp.sum(u * u) / n
    total = jnp.sum(w)
    trace = jnp.trace(w @ w) + jnp.sum(w * w)
    phi2 = (trace + 4 * jnp.sum(cross * cross) / sigma2) / ((total / n) ** 2 * n)
    moran = n / total * (u @ w @ u) / (u @ u)
    statistic = n * moran * moran / phi2
    valid = fit.valid & (phi2 > 0) & jnp.isfinite(statistic)
    return _test(statistic, chi2.sf(statistic, 1), valid)


def spatial_common_factor_test(uncertainty: SpatialUncertainty) -> TestResult[Array]:
    """Test theta_j + rho beta_j = 0 for a fitted Spatial Durbin ML model."""
    fit = uncertainty.fit
    if (
        not isinstance(fit, SpatialFit)
        or not fit.lag
        or fit.error
        or not fit.slx_columns
    ):
        message = (
            "Common-factor restrictions require an SDM likelihood fit "
            "and its uncertainty."
        )
        raise ValueError(message)
    p = fit.design.shape[1]
    base = p - len(fit.slx_columns)
    columns = jnp.asarray(fit.slx_columns)

    def restrictions(parameters: Array) -> Array:
        return parameters[base:p] + parameters[p] * parameters[columns]

    values = restrictions(fit.parameters)
    gradient = jax.jacfwd(restrictions)(fit.parameters)
    covariance = gradient @ uncertainty.covariance.values @ gradient.T
    statistic = values @ jnp.linalg.solve(covariance, values)
    valid = (
        uncertainty.valid
        & (
            jnp.linalg.matrix_rank(jnp.where(jnp.isfinite(covariance), covariance, 0))
            == len(fit.slx_columns)
        )
        & (statistic >= 0)
        & jnp.isfinite(statistic)
    )
    return _test(statistic, chi2.sf(statistic, len(fit.slx_columns)), valid)


def spatial_likelihood_ratio(
    unrestricted: SpatialFit | SpatialPanelFit,
    restricted: SpatialFit | SpatialPanelFit,
    *,
    degrees_of_freedom: int,
    assume_nested: bool = False,
) -> TestResult[Array]:
    """Compare explicitly nested interior Gaussian likelihoods on the same data.

    The caller must declare nesting; matching model labels alone cannot verify
    restrictions, exogeneity or regularity. Panel effect types must agree.
    Different W values, sample ordering or parameter counts invalidate the test.
    """
    if not assume_nested or degrees_of_freedom < 1:
        message = "Declare a nested model comparison and positive restriction degrees."
        raise ValueError(message)
    if (
        type(unrestricted) is not type(restricted)
        or unrestricted.outcome.shape != restricted.outcome.shape
        or unrestricted.weights.values.shape != restricted.weights.values.shape
    ):
        message = "Likelihood comparisons require the same model/data layout."
        raise ValueError(message)
    if (
        isinstance(unrestricted, SpatialPanelFit)
        and isinstance(restricted, SpatialPanelFit)
        and unrestricted.effect_type != restricted.effect_type
    ):
        message = "Spatial panel effect specifications must agree."
        raise ValueError(message)
    difference = 2 * (unrestricted.log_likelihood - restricted.log_likelihood)
    floor = (
        jnp.finfo(difference.dtype).eps
        * 16
        * (
            jnp.abs(unrestricted.log_likelihood)
            + jnp.abs(restricted.log_likelihood)
            + 1
        )
    )
    statistic = jnp.maximum(difference, 0)
    valid = (
        unrestricted.valid
        & restricted.valid
        & jnp.all(unrestricted.outcome == restricted.outcome)
        & jnp.all(unrestricted.weights.values == restricted.weights.values)
    )
    valid &= (difference >= -floor) & (
        (unrestricted.parameters.shape[-1] - restricted.parameters.shape[-1])
        == degrees_of_freedom
    )
    return _test(statistic, chi2.sf(statistic, degrees_of_freedom), valid)


def spatial_diagnostics(  # noqa: PLR0914
    fit: LinearFit, weights: SpatialWeights
) -> SpatialDiagnostics:  # named scientific fit/covariance state
    """Compute Gaussian OLS residual Moran and Anselin lag/error LM statistics.

    Requires ordinary unweighted OLS, an explicit intercept and a zero-diagonal
    dense W. Residual Moran is regression-adjusted, not the unconditional Moran
    reference applied to fitted residuals. Coefficients do not acquire causal
    interpretation from passing these specification tests.
    """
    if isinstance(weights.values, BCOO):
        message = "Exact diagnostic trace calculations require explicit dense weights."
        raise TypeError(message)
    x, y, u = fit.design, fit.outcome, fit.residuals
    w = weights.values
    if w.shape != (y.shape[0], y.shape[0]):
        message = "Diagnostic graph must align with the fitted sample."
        raise ValueError(message)
    n, p = x.shape
    q, _ = jnp.linalg.qr(x, mode="reduced")
    m = jnp.eye(n, dtype=x.dtype) - q @ q.T
    trace_term = jnp.trace(w @ w) + jnp.sum(w * w)
    variance = jnp.sum(u * u) / n
    err = u @ w @ u / variance
    lag = u @ w @ y / variance
    wfitted = w @ fit.fitted_values
    nj = wfitted @ m @ wfitted / variance + trace_term
    lm_error = err * err / trace_term
    lm_lag = lag * lag / nj
    robust_error = (err - trace_term * lag / nj) ** 2 / (
        trace_term * (1 - trace_term / nj)
    )
    robust_lag = (lag - err) ** 2 / (nj - trace_term)
    joint = robust_lag + lm_error
    total = jnp.sum(w)
    d = n - p
    sym = (w + w.T) / 2
    mw = m @ sym @ m
    expectation = n / total * jnp.trace(mw) / d
    moran = n / total * (u @ w @ u) / (u @ u)
    moran_variance = (
        (n / total) ** 2
        * 2
        / (d * (d + 2))
        * (jnp.trace(mw @ mw) - jnp.trace(mw) ** 2 / d)
    )
    z = (moran - expectation) / jnp.sqrt(moran_variance)
    valid = (
        fit.valid
        & jnp.all(fit.relative_precision == 1)
        & (d > 0)
        & (variance > 0)
        & (total != 0)
        & jnp.all(jnp.diag(w) == 0)
    )
    valid &= jnp.any(jnp.all(x == 1, axis=0)) & (trace_term > 0)
    robust_valid = valid & (nj > trace_term)
    moran_valid = valid & (moran_variance > 0)
    return SpatialDiagnostics(
        _test(moran, 2 * ndtr(-jnp.abs(z)), moran_valid),
        expectation,
        moran_variance,
        _test(lm_lag, chi2.sf(lm_lag, 1), valid),
        _test(lm_error, chi2.sf(lm_error, 1), valid),
        _test(robust_lag, chi2.sf(robust_lag, 1), robust_valid),
        _test(robust_error, chi2.sf(robust_error, 1), robust_valid),
        _test(joint, chi2.sf(joint, 2), robust_valid),
        robust_valid & moran_valid,
    )
