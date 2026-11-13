"""Residual and influence diagnostics against statsmodels' public APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.inference.linear_covariance import hac_covariance
from statadic.inference.linear_diagnostics import (
    breusch_godfrey,
    breusch_pagan,
    durbin_watson,
    jarque_bera,
    ols_influence,
    reset_test,
    white_test,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

pytestmark = [
    pytest.mark.integration,
    pytest.mark.reference,
    pytest.mark.external_reference,
    pytest.mark.requires_optional("statsmodels"),
]


def _case(seed: int, *, dummies: bool) -> tuple[np.ndarray, np.ndarray]:
    """Heteroskedastic, serially correlated, nonlinear errors with t tails."""
    rng = np.random.default_rng(seed)
    n = 150
    columns = [np.ones(n), rng.normal(size=n), rng.uniform(size=n)]
    if dummies:
        columns.append(rng.binomial(1, 0.4, size=n).astype(float))
    x = np.column_stack(columns)
    errors = rng.standard_t(5, size=n) * (1 + 0.5 * np.abs(x[:, 1]))
    errors[1:] += 0.3 * errors[:-1]
    y = x @ np.linspace(1.0, -1.0, x.shape[1]) + 0.3 * x[:, 1] ** 2 + errors
    return x, y


def _tolerance() -> float:
    return 1e-8 if jax.config.x64_enabled else 5e-4


@pytest.mark.parametrize("dummies", [False, True])
def test_residual_tests_match_statsmodels(
    optional_reference: Callable[[str], ModuleType], *, dummies: bool
) -> None:
    """Statistics and p-values match statsmodels.

    The dummy makes White's auxiliary design rank deficient: a squared indicator
    equals itself. The matched peer convention counts identified auxiliary rank;
    this test compares the resulting statistic and tail, not a separate df field.
    """
    sm = optional_reference("statsmodels.api")
    smd = optional_reference("statsmodels.stats.diagnostic")
    stattools = optional_reference("statsmodels.stats.stattools")
    x, y = _case(17 if dummies else 16, dummies=dummies)
    reference = sm.OLS(y, x).fit()
    fit = LeastSquares().estimate(LinearProblem(jnp.asarray(x), jnp.asarray(y)))
    tol = _tolerance()

    def check(actual: object, statistic: float, pvalue: float) -> None:
        assert bool(actual.valid)  # type: ignore[attr-defined]
        np.testing.assert_allclose(
            [actual.statistic, actual.pvalue],  # type: ignore[attr-defined]
            [statistic, pvalue],
            rtol=tol,
            atol=1e-12,
        )

    check(jarque_bera(fit), *stattools.jarque_bera(reference.resid)[:2])
    check(breusch_pagan(fit), *smd.het_breuschpagan(reference.resid, x)[:2])
    check(
        breusch_pagan(fit, studentized=False),
        *smd.het_breuschpagan(reference.resid, x, robust=False)[:2],
    )
    check(white_test(fit), *smd.het_white(reference.resid, x)[:2])
    for lags in (1, 4):
        check(
            breusch_godfrey(fit, lags),
            *smd.acorr_breusch_godfrey(reference, nlags=lags)[:2],
        )
    chi2 = smd.linear_reset(reference, power=3, test_type="fitted", use_f=False)
    check(reset_test(fit), float(chi2.statistic), float(chi2.pvalue))
    f_test = smd.linear_reset(reference, power=3, test_type="fitted", use_f=True)
    check(reset_test(fit, reference="F"), float(f_test.fvalue), float(f_test.pvalue))
    np.testing.assert_allclose(
        durbin_watson(fit).statistic,
        stattools.durbin_watson(reference.resid),
        rtol=tol,
    )
    assert np.isnan(durbin_watson(fit).pvalue)


def test_ordered_serial_statistics_match_statsmodels(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Order complete BG rows, retain zero padding, and order DW residuals."""
    sm = optional_reference("statsmodels.api")
    diagnostics = optional_reference("statsmodels.stats.diagnostic")
    stattools = optional_reference("statsmodels.stats.stattools")
    design, outcome = _case(49, dummies=False)
    shuffled = np.random.default_rng(9).permutation(len(outcome))
    order = np.argsort(shuffled)
    fit = LeastSquares().estimate(
        LinearProblem(jnp.asarray(design[shuffled]), jnp.asarray(outcome[shuffled]))
    )
    reference = sm.OLS(outcome, design).fit()
    lags = 4
    expected = diagnostics.acorr_breusch_godfrey(reference, nlags=lags, store=True)
    actual = breusch_godfrey(fit, lags, row_order=jnp.asarray(order))
    assert actual.valid
    np.testing.assert_allclose(
        [actual.statistic, actual.pvalue], expected[:2], rtol=_tolerance()
    )
    assert int(actual.df) == expected[4].usedlag
    assert expected[4].resols.nobs == len(outcome)
    np.testing.assert_allclose(
        durbin_watson(fit, row_order=jnp.asarray(order)).statistic,
        stattools.durbin_watson(reference.resid),
        rtol=_tolerance(),
    )
    # Trimming the first L rows is a different auxiliary regression.
    residuals = np.asarray(reference.resid)
    lagged = np.column_stack(
        [residuals[lags - lag : -lag] for lag in range(1, lags + 1)]
    )
    trimmed = sm.OLS(residuals[lags:], np.column_stack((design[lags:], lagged))).fit()
    assert not np.isclose(float(actual.statistic), trimmed.nobs * trimmed.rsquared)


@pytest.mark.parametrize("weighted", [False, True], ids=["ols", "wls"])
def test_ordered_hac_matches_statsmodels(
    optional_reference: Callable[[str], ModuleType], *, weighted: bool
) -> None:
    """Complete ordered influence rows match Bartlett HAC for OLS and WLS."""
    sm = optional_reference("statsmodels.api")
    sandwich = optional_reference("statsmodels.stats.sandwich_covariance")
    design, outcome = _case(38, dummies=False)
    precision = 0.5 + np.arange(len(outcome)) % 7
    shuffled = np.random.default_rng(18).permutation(len(outcome))
    fit = LeastSquares().estimate(
        LinearProblem(
            jnp.asarray(design[shuffled]),
            jnp.asarray(outcome[shuffled]),
            jnp.asarray(precision[shuffled]) if weighted else None,
        )
    )
    reference = (
        sm.WLS(outcome, design, weights=precision).fit()
        if weighted
        else sm.OLS(outcome, design).fit()
    )
    bandwidth = 3
    uncertainty = hac_covariance(
        fit, bandwidth, row_order=jnp.asarray(np.argsort(shuffled))
    )
    assert uncertainty.valid
    assert uncertainty.fit is fit
    np.testing.assert_allclose(
        uncertainty.covariance.values,
        sandwich.cov_hac_simple(reference, nlags=bandwidth, use_correction=False),
        rtol=_tolerance() * 10,
        atol=_tolerance() * 0.01,
    )


def test_breusch_pagan_variance_design_matches_statsmodels(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """A separate variance design with a constant reproduces het_breuschpagan."""
    sm = optional_reference("statsmodels.api")
    smd = optional_reference("statsmodels.stats.diagnostic")
    x, y = _case(5, dummies=False)
    variance = np.column_stack([np.ones(len(y)), np.abs(x[:, 1]), x[:, 2] ** 2])
    reference = sm.OLS(y, x).fit()
    fit = LeastSquares().estimate(LinearProblem(jnp.asarray(x), jnp.asarray(y)))
    actual = breusch_pagan(fit, jnp.asarray(variance))
    expected = smd.het_breuschpagan(reference.resid, variance)
    np.testing.assert_allclose(
        [actual.statistic, actual.pvalue], expected[:2], rtol=_tolerance()
    )


def test_influence_matches_statsmodels(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Leverage, studentized residuals, Cook's distance, DFFITS and DFBETAS."""
    sm = optional_reference("statsmodels.api")
    outliers = optional_reference("statsmodels.stats.outliers_influence")
    x, y = _case(8, dummies=True)
    reference = outliers.OLSInfluence(sm.OLS(y, x).fit())
    actual = ols_influence(
        LeastSquares().estimate(LinearProblem(jnp.asarray(x), jnp.asarray(y)))
    )
    tol = _tolerance()
    assert bool(actual.valid)
    for native, expected in (
        (actual.leverage, reference.hat_matrix_diag),
        (actual.studentized_internal, reference.resid_studentized_internal),
        (actual.studentized_external, reference.resid_studentized_external),
        (actual.cooks_distance, reference.cooks_distance[0]),
        (actual.dffits, reference.dffits[0]),
        (actual.dfbetas, reference.dfbetas),
    ):
        np.testing.assert_allclose(native, expected, rtol=tol * 100, atol=tol)
