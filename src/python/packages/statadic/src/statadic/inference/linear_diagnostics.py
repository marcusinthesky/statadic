# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""Residual specification and influence diagnostics for an unweighted OLS fit.

Every kernel consumes the actual :class:`~statadic.estimation.linear.LinearFit`: its
original-order residuals, retained design, leverage and coefficient map. Nothing
refits the model, repairs it or replaces its estimates. A diagnostic is
unavailable, with NaN statistics, unless the fit is valid, identified, has
positive residual degrees of freedom and a non-zero residual sum of squares,
and every relative precision is equal. Relative precisions that differ define a
WLS fit, whose raw residuals are not homoskedastic under its own model; these
diagnostics are not defined for it.

The statistics follow the conventions of ``statsmodels.stats.diagnostic`` and
``statsmodels.stats.outliers_influence``. Test references are asymptotic:

* Jarque-Bera uses biased sample moments of the residuals and a chi-square(2)
  reference (Jarque and Bera 1987).
* Breusch-Pagan regresses squared residuals on a variance design containing a
  constant. The studentized (Koenker 1981) form reports ``n R²``; the original
  form (Breusch and Pagan 1979) reports half the explained sum of squares of
  ``e²/mean(e²)``, which assumes Gaussian errors. Both use chi-square(rank-1).
* White (1980) is the studentized Breusch-Pagan statistic on the products
  ``x_j x_k`` (``j <= k``) of the mean design, which must contain a constant.
* Durbin-Watson (1950) has no universal p-value: its null distribution depends
  on the design. It is reported as a statistic only.
* Breusch-Godfrey (Breusch 1978; Godfrey 1978) regresses residuals on the
  design and ``L`` zero-filled residual lags; ``n R²`` is chi-square(L).
* RESET (Ramsey 1969) adds powers of the fitted values and Wald-tests them
  with the classical covariance of the augmented fit, with either a
  chi-square or an F reference.

Durbin-Watson and Breusch-Godfrey use retained row order unless an explicit
``row_order`` permutation selects adjacency. Lags count ordered observations,
not elapsed calendar intervals.
Auxiliary regressions are projections onto the SVD range of their design, so a
rank-deficient auxiliary design (for example squares of dummy columns) loses
only redundant columns; degrees of freedom count the identified rank.

Influence quantities (Cook 1977; Belsley, Kuh and Welsch 1980) are descriptive
sensitivities of the actual fit to deleting each row. They are not tests and do
not mark an observation as invalid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.scipy.stats import chi2
from jaxtyping import Bool, Float  # noqa: TC002

from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.estimation.linear import LinearFit  # noqa: TC001
from statadic.inference._parameter_reference import _f_survival
from statadic.inference._row_order import _validated_row_order

__all__ = [
    "LinearInfluence",
    "ReferenceLaw",
    "ResidualTest",
    "breusch_godfrey",
    "breusch_pagan",
    "durbin_watson",
    "jarque_bera",
    "ols_influence",
    "reset_test",
    "residual_moments",
    "white_test",
]

type ReferenceLaw = Literal["chi2", "F", "none"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ResidualTest:
    """A residual diagnostic statistic with its explicit asymptotic reference.

    ``df`` is the chi-square or numerator F degrees of freedom and
    ``df_denominator`` the F denominator degrees of freedom (NaN otherwise).
    ``reference`` is ``"none"`` for a statistic without a calibrated p-value.
    Unavailable diagnostics hold NaN statistics and p-values with ``valid``
    false; nothing is fabricated for them.
    """

    __test__ = False

    statistic: Float[Array, "*batch"]
    pvalue: Float[Array, "*batch"]
    df: Float[Array, "*batch"]
    df_denominator: Float[Array, "*batch"]
    valid: Bool[Array, "*batch"]
    reference: ReferenceLaw = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LinearInfluence:
    """Row-deletion sensitivities of an actual OLS fit, in original row order.

    ``studentized_internal`` scales by the full-sample residual variance and
    ``studentized_external`` by the variance with the row deleted. ``dfbetas``
    has shape ``(observations, parameters)``. ``valid`` records full-fit
    residual-diagnostic eligibility, not availability of every deletion.
    Each quantity has its own validity mask and unavailable entries are NaN.
    The DFBETAS mask has the same row/parameter axes as its values; other masks
    have one entry per row. Leverage can remain available for an exact fit.
    """

    leverage: Float[Array, "*batch observations"]
    studentized_internal: Float[Array, "*batch observations"]
    studentized_external: Float[Array, "*batch observations"]
    cooks_distance: Float[Array, "*batch observations"]
    dffits: Float[Array, "*batch observations"]
    dfbetas: Float[Array, "*batch observations parameters"]
    valid: Bool[Array, "*batch"]
    leverage_valid: Bool[Array, "*batch observations"]
    studentized_internal_valid: Bool[Array, "*batch observations"]
    studentized_external_valid: Bool[Array, "*batch observations"]
    cooks_distance_valid: Bool[Array, "*batch observations"]
    dffits_valid: Bool[Array, "*batch observations"]
    dfbetas_valid: Bool[Array, "*batch observations parameters"]


def _eligible(fit: LinearFit) -> Array:
    """Return whether residual diagnostics are defined for this actual fit."""
    weights = fit.relative_precision
    n = fit.residuals.shape[-1]
    eps = jnp.finfo(fit.residuals.dtype).eps
    scale = jnp.sum(jnp.square(fit.outcome))
    nonzero_residuals = fit.sse > n * eps * scale
    return (
        fit.valid
        & fit.identified
        & (fit.df_resid > 0)
        & jnp.all(weights == weights[..., :1])
        & nonzero_residuals
    )


def _projection(design: Array, response: Array) -> tuple[Array, Array]:
    """Project onto the identified SVD range of ``design``; return fit and rank."""
    u, singular, _ = jnp.linalg.svd(design, full_matrices=False)
    tolerance = (
        jnp.max(singular, initial=0.0) * max(design.shape) * jnp.finfo(design.dtype).eps
    )
    keep = singular > tolerance
    fitted = u @ jnp.where(keep, u.T @ response, 0.0)
    return fitted, jnp.sum(keep)


def _spans_constant(design: Array) -> Array:
    """Return whether the column space of ``design`` contains the constant."""
    ones = jnp.ones(design.shape[0], dtype=design.dtype)
    fitted, _ = _projection(design, ones)
    tolerance = jnp.sqrt(jnp.finfo(design.dtype).eps) * jnp.sqrt(design.shape[0])
    return jnp.linalg.norm(ones - fitted) <= tolerance


def _auxiliary(design: Array, response: Array) -> tuple[Array, Array, Array]:
    """Return centered total and residual sums of squares and the auxiliary rank."""
    fitted, rank = _projection(design, response)
    total = jnp.sum(jnp.square(response - jnp.mean(response)))
    residual = jnp.sum(jnp.square(response - fitted))
    return total, residual, rank


def _chi2_test(statistic: Array, df: Array, valid: Array) -> ResidualTest:
    safe_df = jnp.where(valid, df, 1)
    return ResidualTest(
        jnp.where(valid, statistic, jnp.nan),
        jnp.where(valid, chi2.sf(statistic, safe_df), jnp.nan),
        jnp.where(valid, df, jnp.nan).astype(statistic.dtype),
        jnp.full_like(statistic, jnp.nan),
        valid,
        "chi2",
    )


def residual_moments(fit: LinearFit) -> tuple[Array, Array]:
    """Return biased residual skewness and (non-excess) kurtosis."""
    centered = fit.residuals - jnp.mean(fit.residuals, axis=-1, keepdims=True)
    m2 = jnp.mean(jnp.square(centered), axis=-1)
    m3 = jnp.mean(centered**3, axis=-1)
    m4 = jnp.mean(centered**4, axis=-1)
    valid = _eligible(fit) & (m2 > 0)
    safe = jnp.where(valid, m2, 1)
    skewness = jnp.where(valid, m3 / safe**1.5, jnp.nan)
    kurtosis = jnp.where(valid, m4 / jnp.square(safe), jnp.nan)
    return skewness, kurtosis


@jax.jit
def jarque_bera(fit: LinearFit) -> ResidualTest:
    """Test residual skewness and kurtosis against a Gaussian chi-square(2) null."""
    skewness, kurtosis = residual_moments(fit)
    n = fit.residuals.shape[-1]
    statistic = n / 6.0 * (jnp.square(skewness) + jnp.square(kurtosis - 3.0) / 4.0)
    valid = jnp.isfinite(statistic)
    return _chi2_test(statistic, jnp.asarray(2), valid)


@partial(jax.jit, static_argnames=("studentized",))
def breusch_pagan(
    fit: LinearFit,
    variance_design: Array | None = None,
    *,
    studentized: bool = True,
) -> ResidualTest:
    """Test whether the error variance changes with the supplied variance design.

    ``variance_design`` defaults to the fit's mean design and must contain a
    constant. ``studentized=False`` gives the original statistic, whose
    chi-square reference additionally requires Gaussian errors.
    """
    design = fit.design if variance_design is None else jnp.asarray(variance_design)
    squared = jnp.square(fit.residuals)
    n = squared.shape[-1]
    if design.ndim != 2 or design.shape[0] != n:  # noqa: PLR2004
        message = "The variance design must be a matrix with one row per residual."
        raise ValueError(message)
    response = squared if studentized else squared / jnp.mean(squared)
    total, residual, rank = _auxiliary(design.astype(squared.dtype), response)
    explained = total - residual
    statistic = n * explained / total if studentized else explained / 2.0
    valid = (
        _eligible(fit)
        & _spans_constant(design.astype(squared.dtype))
        & (rank >= 2)  # noqa: PLR2004 -- a constant and one variance regressor
        & (total > 0)
    )
    return _chi2_test(statistic, rank - 1, valid)


@jax.jit
def white_test(fit: LinearFit) -> ResidualTest:
    """Studentized Breusch-Pagan test on all products of the mean design columns."""
    first, second = jnp.triu_indices(fit.design.shape[-1])
    products = fit.design[:, first] * fit.design[:, second]
    return breusch_pagan(fit, products, studentized=True)


@jax.jit
def durbin_watson(
    fit: LinearFit, *, row_order: ArrayLike | None = None
) -> ResidualTest:
    """Return Durbin-Watson over the supplied permutation, without a p-value."""
    order, order_valid = _validated_row_order(row_order, fit.residuals.shape[-1])
    residuals = fit.residuals[order]
    statistic = jnp.sum(jnp.square(jnp.diff(residuals))) / jnp.sum(
        jnp.square(residuals)
    )
    valid = _eligible(fit) & order_valid & (residuals.shape[-1] >= 2)  # noqa: PLR2004
    return ResidualTest(
        jnp.where(valid, statistic, jnp.nan),
        jnp.full_like(statistic, jnp.nan),
        jnp.full_like(statistic, jnp.nan),
        jnp.full_like(statistic, jnp.nan),
        valid,
        "none",
    )


@partial(jax.jit, static_argnames=("lags",))
def breusch_godfrey(
    fit: LinearFit, lags: int = 1, *, row_order: ArrayLike | None = None
) -> ResidualTest:
    """Test serial correlation through ``lags`` in the selected observation order.

    Pre-sample residual lags are zero, as in ``statsmodels``; ``n R²`` of the
    auxiliary regression on the design and the lags is chi-square(lags),
    provided all requested added-lag restrictions are identified.
    """
    if isinstance(lags, bool) or not isinstance(lags, int) or lags < 1:
        message = "Breusch-Godfrey lags must be a positive integer."
        raise ValueError(message)
    order, order_valid = _validated_row_order(row_order, fit.residuals.shape[-1])
    residuals = fit.residuals[order]
    n = residuals.shape[-1]
    padded = jnp.concatenate((jnp.zeros(lags, dtype=residuals.dtype), residuals))
    lagged = jnp.stack(
        [padded[lags - lag : lags - lag + n] for lag in range(1, lags + 1)], axis=1
    )
    mean_design = fit.design[order].astype(residuals.dtype)
    design = jnp.concatenate((mean_design, lagged), axis=1)
    total, residual, rank = _auxiliary(design, residuals)
    statistic = n * (total - residual) / total
    valid = (
        _eligible(fit)
        & order_valid
        & (total > 0)
        & (rank < n)
        & (rank == fit.rank + lags)
        & _spans_constant(mean_design)
    )
    return _chi2_test(statistic, jnp.asarray(lags), valid)


@partial(jax.jit, static_argnames=("powers", "reference"))
def reset_test(  # noqa: PLR0914 -- restricted and augmented fit evidence
    fit: LinearFit,
    powers: tuple[int, ...] = (2, 3),
    reference: Literal["chi2", "F"] = "chi2",
) -> ResidualTest:
    """Wald-test added powers of the fitted values with classical covariance.

    The chi-square statistic is ``(SSR_restricted - SSR_augmented) / s²`` with
    ``s²`` from the augmented fit; the F statistic divides it by the number of
    identified added columns.
    """
    if not powers or any(
        isinstance(power, bool) or not isinstance(power, int) or power < 2  # noqa: PLR2004
        for power in powers
    ):
        message = "RESET powers must be integers of at least two."
        raise ValueError(message)
    if reference not in {"chi2", "F"}:
        message = "RESET reference must be 'chi2' or 'F'."
        raise ValueError(message)
    fitted = fit.fitted_values
    added = jnp.stack([fitted**power for power in powers], axis=1)
    design = jnp.concatenate((fit.design.astype(fitted.dtype), added), axis=1)
    augmented, rank = _projection(design, fit.outcome.astype(fitted.dtype))
    n = fitted.shape[-1]
    ssr_augmented = jnp.sum(jnp.square(fit.outcome - augmented))
    restrictions = rank - fit.rank
    df_resid = n - rank
    scale = ssr_augmented / jnp.where(df_resid > 0, df_resid, 1)
    wald = (fit.sse - ssr_augmented) / scale
    valid = _eligible(fit) & (restrictions >= 1) & (df_resid > 0) & (ssr_augmented > 0)
    if reference == "chi2":
        return _chi2_test(wald, restrictions, valid)
    safe_q = jnp.where(valid, restrictions, 1).astype(fitted.dtype)
    safe_df = jnp.where(valid, df_resid, 1).astype(fitted.dtype)
    statistic = wald / safe_q
    return ResidualTest(
        jnp.where(valid, statistic, jnp.nan),
        jnp.where(valid, _f_survival(statistic, safe_q, safe_df), jnp.nan),
        jnp.where(valid, safe_q, jnp.nan),
        jnp.where(valid, safe_df, jnp.nan),
        valid,
        "F",
    )


@jax.jit
def ols_influence(fit: LinearFit) -> LinearInfluence:  # noqa: PLR0914
    """Return leverage, studentized residuals, Cook's distance, DFFITS and DFBETAS."""
    residuals = fit.residuals
    leverage = fit.leverage
    n = residuals.shape[-1]
    p = fit.rank.astype(residuals.dtype)
    complement = 1.0 - leverage
    eps = jnp.finfo(residuals.dtype).eps
    valid = _eligible(fit)
    leverage_valid = (
        fit.valid
        & fit.identified
        & jnp.all(fit.relative_precision == fit.relative_precision[..., :1])
        & jnp.isfinite(leverage)
        & (leverage >= 0)
        & (leverage <= 1.0 + jnp.sqrt(eps))
    )
    row_valid = valid & leverage_valid & (complement > jnp.sqrt(eps))
    safe_complement = jnp.where(row_valid, complement, 1.0)
    scale = fit.sse / jnp.maximum(n - p, 1.0)
    deleted_sse = fit.sse - jnp.square(residuals) / safe_complement
    # Cancellation cannot establish a positive deleted variance at roundoff.
    deleted_valid = (
        row_valid
        & (n - p - 1 > 0)
        & jnp.isfinite(deleted_sse)
        & (deleted_sse > n * eps * fit.sse)
    )
    deleted_scale = jnp.where(
        deleted_valid, deleted_sse / jnp.maximum(n - p - 1.0, 1.0), 1.0
    )
    internal = residuals / jnp.sqrt(scale * safe_complement)
    external = residuals / jnp.sqrt(deleted_scale * safe_complement)
    cooks = jnp.square(internal) * leverage / (p * safe_complement)
    dffits = external * jnp.sqrt(jnp.maximum(leverage, 0.0) / safe_complement)
    change = (fit.coefficient_map * (residuals / safe_complement)).T
    diagonal = jnp.diag(fit.normalized_covariance)
    coefficient_valid = jnp.isfinite(diagonal) & (diagonal > 0)
    dfbetas = change / jnp.sqrt(
        deleted_scale[:, None] * jnp.where(coefficient_valid, diagonal, 1.0)[None, :]
    )
    internal_valid = row_valid & jnp.isfinite(internal)
    external_valid = deleted_valid & jnp.isfinite(external)
    cooks_valid = row_valid & jnp.isfinite(cooks)
    dffits_valid = deleted_valid & jnp.isfinite(dffits)
    dfbetas_valid = (
        deleted_valid[:, None] & coefficient_valid[None, :] & jnp.isfinite(dfbetas)
    )

    return LinearInfluence(
        jnp.where(leverage_valid, leverage, jnp.nan),
        jnp.where(internal_valid, internal, jnp.nan),
        jnp.where(external_valid, external, jnp.nan),
        jnp.where(cooks_valid, cooks, jnp.nan),
        jnp.where(dffits_valid, dffits, jnp.nan),
        jnp.where(dfbetas_valid, dfbetas, jnp.nan),
        valid,
        leverage_valid,
        internal_valid,
        external_valid,
        cooks_valid,
        dffits_valid,
        dfbetas_valid,
    )
