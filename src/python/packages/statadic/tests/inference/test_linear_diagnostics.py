"""Residual and influence diagnostics against independent NumPy witnesses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.inference.linear_diagnostics import (
    breusch_godfrey,
    breusch_pagan,
    durbin_watson,
    jarque_bera,
    ols_influence,
    reset_test,
    residual_moments,
    white_test,
)
from scipy import stats

if TYPE_CHECKING:
    from statadic.estimation.linear import LinearFit

pytestmark = pytest.mark.unit
_DURBIN_WATSON_UPPER = 4.0


def _data(n: int = 80, seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), rng.normal(size=n), rng.uniform(size=n)])
    errors = rng.standard_t(6, size=n) * (1.0 + 0.4 * np.abs(x[:, 1]))
    y = x @ np.array([0.5, 1.5, -1.0]) + 0.2 * x[:, 1] ** 2 + errors
    return x, y


def _fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> LinearFit:
    return LeastSquares().estimate(
        LinearProblem(
            jnp.asarray(x),
            jnp.asarray(y),
            None if weights is None else jnp.asarray(weights),
        )
    )


def _tolerance() -> float:
    return 1e-9 if jax.config.x64_enabled else 2e-4


def _r2(design: np.ndarray, response: np.ndarray) -> tuple[float, int]:
    coef, *_ = np.linalg.lstsq(design, response, rcond=None)
    fitted = design @ coef
    total = np.sum((response - response.mean()) ** 2)
    return 1.0 - np.sum((response - fitted) ** 2) / total, np.linalg.matrix_rank(design)


def _residuals() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the NumPy least-squares design, response and residuals."""
    x, y = _data()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return x, y, y - x @ beta


@pytest.mark.reference
def test_jarque_bera_matches_biased_moment_formula() -> None:
    """Skewness, kurtosis and the chi-square(2) tail match NumPy and SciPy."""
    x, y, e = _residuals()
    fit = _fit(x, y)
    centered = e - e.mean()
    skew = np.mean(centered**3) / np.mean(centered**2) ** 1.5
    kurt = np.mean(centered**4) / np.mean(centered**2) ** 2
    jb = len(e) / 6 * (skew**2 + (kurt - 3) ** 2 / 4)
    actual = jarque_bera(fit)
    np.testing.assert_allclose(residual_moments(fit), (skew, kurt), rtol=_tolerance())
    np.testing.assert_allclose(
        [actual.statistic, actual.pvalue],
        [jb, stats.chi2.sf(jb, 2)],
        rtol=_tolerance(),
    )


@pytest.mark.reference
def test_breusch_pagan_and_white_match_auxiliary_regressions() -> None:
    """Studentized n R², half the explained sum and identified-rank df."""
    x, y, e = _residuals()
    fit = _fit(x, y)
    n = len(e)
    r2, rank = _r2(x, e**2)
    koenker = breusch_pagan(fit)
    np.testing.assert_allclose(koenker.statistic, n * r2, rtol=_tolerance())
    assert int(koenker.df) == rank - 1
    scaled = e**2 / np.mean(e**2)
    coef, *_ = np.linalg.lstsq(x, scaled, rcond=None)
    explained = np.sum((x @ coef - scaled.mean()) ** 2)
    np.testing.assert_allclose(
        breusch_pagan(fit, studentized=False).statistic,
        explained / 2,
        rtol=_tolerance(),
    )
    i, j = np.triu_indices(x.shape[1])
    r2, rank = _r2(x[:, i] * x[:, j], e**2)
    white = white_test(fit)
    np.testing.assert_allclose(white.statistic, n * r2, rtol=_tolerance())
    assert int(white.df) == rank - 1


@pytest.mark.reference
def test_serial_correlation_statistics_match_lagged_regressions() -> None:
    """Durbin-Watson and zero-padded Breusch-Godfrey match their formulas."""
    x, y, e = _residuals()
    fit = _fit(x, y)
    np.testing.assert_allclose(
        durbin_watson(fit).statistic,
        np.sum(np.diff(e) ** 2) / np.sum(e**2),
        rtol=_tolerance(),
    )
    lags = 3
    lagged = np.column_stack([np.r_[np.zeros(k), e[:-k]] for k in range(1, lags + 1)])
    r2, _ = _r2(np.column_stack([x, lagged]), e)
    np.testing.assert_allclose(
        breusch_godfrey(fit, lags).statistic, len(e) * r2, rtol=_tolerance()
    )


@pytest.mark.reference
def test_reset_matches_augmented_least_squares() -> None:
    """The Wald and F forms use the augmented fit's classical scale."""
    x, y, e = _residuals()
    fit = _fit(x, y)
    n = len(e)
    fitted = y - e
    augmented = np.column_stack([x, fitted**2, fitted**3])
    coef, *_ = np.linalg.lstsq(augmented, y, rcond=None)
    ssr_u = np.sum((y - augmented @ coef) ** 2)
    wald = (np.sum(e**2) - ssr_u) / (ssr_u / (n - augmented.shape[1]))
    tol = _tolerance() * 10
    np.testing.assert_allclose(reset_test(fit).statistic, wald, rtol=tol)
    f_test = reset_test(fit, reference="F")
    np.testing.assert_allclose(f_test.statistic, wald / 2, rtol=tol)
    np.testing.assert_allclose(
        f_test.pvalue, stats.f.sf(wald / 2, 2, n - augmented.shape[1]), rtol=tol
    )


@pytest.mark.reference
def test_influence_matches_explicit_leave_one_out_refits() -> None:  # noqa: PLR0914
    """Cook's distance, DFFITS and DFBETAS match actually deleting each row."""
    x, y = _data(n=30, seed=9)
    fit = _fit(x, y)
    influence = ols_influence(fit)
    n, p = x.shape
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    s2 = np.sum((y - x @ beta) ** 2) / (n - p)
    xtx_inv = np.linalg.inv(x.T @ x)
    tol = 1e-8 if jax.config.x64_enabled else 2e-3
    for row in range(n):
        keep = np.arange(n) != row
        deleted, *_ = np.linalg.lstsq(x[keep], y[keep], rcond=None)
        s2_deleted = np.sum((y[keep] - x[keep] @ deleted) ** 2) / (n - p - 1)
        change = beta - deleted
        cooks = change @ x.T @ x @ change / (p * s2)
        dffits = x[row] @ change / np.sqrt(s2_deleted * x[row] @ xtx_inv @ x[row])
        dfbetas = change / np.sqrt(s2_deleted * np.diag(xtx_inv))
        np.testing.assert_allclose(influence.cooks_distance[row], cooks, rtol=tol)
        np.testing.assert_allclose(influence.dffits[row], dffits, rtol=tol)
        np.testing.assert_allclose(influence.dfbetas[row], dfbetas, rtol=tol, atol=tol)


@pytest.mark.property
def test_statistics_are_invariant_to_response_scale() -> None:
    """Rescaling the response leaves every scale-free statistic unchanged."""
    x, y = _data()
    base, scaled = _fit(x, y), _fit(x, 7.5 * y)
    for kernel in (jarque_bera, breusch_pagan, white_test, durbin_watson, reset_test):
        np.testing.assert_allclose(
            kernel(scaled).statistic, kernel(base).statistic, rtol=_tolerance() * 10
        )
    np.testing.assert_allclose(
        breusch_pagan(scaled, studentized=False).statistic,
        breusch_pagan(base, studentized=False).statistic,
        rtol=_tolerance() * 10,
    )


@pytest.mark.property
def test_durbin_watson_lies_in_zero_to_four() -> None:
    """The bound proved as STATADIC-DW-001 holds on alternating and smooth residuals."""
    x = np.column_stack([np.ones(40), np.arange(40.0)])
    for y in (np.cos(np.arange(40.0) * np.pi), np.sin(np.arange(40.0) / 3)):
        value = float(durbin_watson(_fit(x, y)).statistic)
        assert 0.0 <= value <= _DURBIN_WATSON_UPPER


@pytest.mark.contract
def test_unavailable_diagnostics_hold_no_statistic() -> None:
    """Weighted, rank-deficient, exact and constant-free fits are unavailable."""
    x, y = _data(n=20)
    weighted = _fit(x, y, np.linspace(0.5, 2.0, 20))
    rank_deficient = _fit(np.column_stack([x, x[:, 1]]), y)
    exact = _fit(x, x @ np.array([1.0, 2.0, 3.0]))
    no_constant = _fit(x[:, 1:], y)
    for fit in (weighted, rank_deficient, exact):
        for test in (jarque_bera(fit), breusch_pagan(fit), durbin_watson(fit)):
            assert not bool(test.valid)
            assert np.isnan(test.statistic)
        assert not bool(ols_influence(fit).valid)
    assert bool(jarque_bera(no_constant).valid)
    assert not bool(breusch_pagan(no_constant).valid)
    assert not bool(white_test(no_constant).valid)
    assert not bool(breusch_godfrey(no_constant).valid)


@pytest.mark.contract
def test_equal_relative_precisions_are_ordinary_least_squares() -> None:
    """A common positive precision leaves residuals and diagnostics unchanged."""
    x, y = _data()
    base = breusch_pagan(_fit(x, y))
    common = breusch_pagan(_fit(x, y, np.full(len(y), 3.0)))
    assert bool(common.valid)
    np.testing.assert_allclose(common.statistic, base.statistic, rtol=_tolerance())


@pytest.mark.contract
def test_invalid_static_options_raise_before_tracing() -> None:
    """Lags, RESET powers and variance designs are checked eagerly."""
    x, y = _data(n=20)
    fit = _fit(x, y)
    with pytest.raises(ValueError, match="positive integer"):
        breusch_godfrey(fit, 0)
    with pytest.raises(ValueError, match="at least two"):
        reset_test(fit, (1,))
    with pytest.raises(ValueError, match="one row per residual"):
        breusch_pagan(fit, jnp.ones((3, 2)))


@pytest.mark.property
def test_kernels_batch_under_vmap() -> None:
    """Stacked fits give the same statistics as separate calls."""
    x, y = _data(n=40)
    rng = np.random.default_rng(2)
    ys = jnp.asarray(np.stack([y, y + rng.normal(size=y.size)]))

    def estimate(response: jax.Array) -> LinearFit:
        return LeastSquares().estimate(LinearProblem(jnp.asarray(x), response))

    fits = jax.vmap(estimate)(ys)
    batched = jax.vmap(breusch_pagan)(fits)
    for row in range(2):
        single = breusch_pagan(_fit(x, np.asarray(ys[row])))
        np.testing.assert_allclose(
            batched.statistic[row], single.statistic, rtol=_tolerance() * 10
        )
