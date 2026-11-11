# standalone package-maintenance entry point
"""Residual diagnostics against spreg's regression-adjusted public APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from statadic.estimation.linear import (
    LeastSquares,
    LinearProblem,
)
from statadic.estimation.spatial import ML, SAR, SpatialIV, SpatialProblem
from statadic.inference.spatial import (
    spatial_ak_test,
    spatial_common_factor_test,
    spatial_covariance,
)
from statadic.inference.spatial.diagnostics import spatial_diagnostics
from statadic.operators.spatial import prepare_spatial_weights
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
def test_moran_and_lm_match_spreg(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Match residual Moran moments and classical/robust lag/error LM tests."""
    spreg = optional_reference("spreg")
    libpysal = optional_reference("libpysal")
    rng = np.random.default_rng(126)
    n = 30
    w = 0.8 * np.roll(np.eye(n), 1, axis=1) + 0.2 * np.roll(np.eye(n), -1, axis=1)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    y = rng.normal(size=n)
    actual = spatial_diagnostics(
        LeastSquares().estimate(LinearProblem(x, y)), prepare_spatial_weights(w)
    )
    weights = libpysal.weights.full2W(w)
    oracle = spreg.OLS(y[:, None], x[:, 1:], w=weights, spat_diag=True, moran=True)
    assert actual.valid
    for native, reference in [
        (actual.lm_lag, oracle.lm_lag),
        (actual.lm_error, oracle.lm_error),
        (actual.robust_lm_lag, oracle.rlm_lag),
        (actual.robust_lm_error, oracle.rlm_error),
        (actual.lm_sarma, oracle.lm_sarma),
    ]:
        np.testing.assert_allclose(
            [native.statistic, native.pvalue], reference, rtol=1e-8, atol=1e-10
        )
        degrees = 2 if native is actual.lm_sarma else 1
        np.testing.assert_allclose(
            native.pvalue,
            stats.chi2.sf(float(native.statistic), degrees),
            rtol=1e-10,
            atol=1e-12,
        )
    moran = spreg.MoranRes(oracle, weights, z=True)
    np.testing.assert_allclose(actual.moran.statistic, moran.I, atol=1e-10)
    np.testing.assert_allclose(actual.moran_expectation, moran.eI, atol=1e-10)
    np.testing.assert_allclose(actual.moran_variance, moran.vI, atol=1e-10)
    np.testing.assert_allclose(actual.moran.pvalue, moran.p_norm, atol=1e-10)


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
def test_ak_and_common_factor_match_spreg(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Both diagnostics consume the actual fitted model and its specification."""
    spreg = optional_reference("spreg")
    libpysal = optional_reference("libpysal")
    rng = np.random.default_rng(126)
    n = 40
    w = 0.6 * np.roll(np.eye(n), 1, axis=1) + 0.4 * np.roll(np.eye(n), -1, axis=1)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    y = np.linalg.solve(
        np.eye(n) - 0.3 * w, x @ np.array([0.3, -0.5, 0.8]) + rng.normal(size=n) * 0.5
    )
    graph = prepare_spatial_weights(w)
    oracle_graph = libpysal.weights.full2W(w)
    iv = SpatialIV(instrument_columns=(1, 2)).estimate(
        SpatialProblem(x, y, graph, process=SAR(), slx_columns=())
    )
    oracle_iv = spreg.GM_Lag(y[:, None], x[:, 1:], w=oracle_graph, w_lags=1)
    actual = spatial_ak_test(iv)
    oracle = spreg.AKtest(oracle_iv, oracle_graph, case="gen")
    np.testing.assert_allclose(
        [actual.statistic, actual.pvalue], [oracle.ak, oracle.p], rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        actual.pvalue, stats.chi2.sf(float(actual.statistic), 1), rtol=1e-10, atol=1e-12
    )
    sdm = ML().estimate(SpatialProblem(x, y, graph, process=SAR(), slx_columns=(1, 2)))
    actual = spatial_common_factor_test(spatial_covariance(sdm))
    oracle_sdm = spreg.ML_Lag(
        y[:, None], x[:, 1:], oracle_graph, slx_lags=1, epsilon=1e-9
    )
    np.testing.assert_allclose(
        [actual.statistic, actual.pvalue], oracle_sdm.cfh_test, rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        actual.pvalue,
        stats.chi2.sf(float(actual.statistic), len(sdm.slx_columns)),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.component
@pytest.mark.contract
def test_intercept_only_keeps_identified_diagnostics() -> None:
    """An unidentified robust LM alternative does not suppress Moran or LM error."""
    n = 20
    w = np.roll(np.eye(n), 1, axis=1)
    y = np.random.default_rng(126).normal(size=n)
    actual = spatial_diagnostics(
        LeastSquares().estimate(LinearProblem(np.ones((n, 1)), y)),
        prepare_spatial_weights(w),
    )
    assert np.isfinite(actual.moran.pvalue)
    assert np.isfinite(actual.lm_error.pvalue)
    assert not actual.valid
    assert np.isnan(actual.robust_lm_lag.pvalue)


@pytest.mark.component
@pytest.mark.property
def test_diagnostics_joint_node_permutation_preserves_all_reported_outputs() -> None:
    """Reorder both graph axes together with the complete fitted OLS sample."""
    rng = np.random.default_rng(738)
    n = 25
    w = rng.uniform(size=(n, n))
    np.fill_diagonal(w, 0)
    w /= w.sum(axis=1, keepdims=True)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    y = rng.normal(size=n)
    order = rng.permutation(n)
    original = spatial_diagnostics(
        LeastSquares().estimate(LinearProblem(x, y)), prepare_spatial_weights(w)
    )
    actual = spatial_diagnostics(
        LeastSquares().estimate(LinearProblem(x[order], y[order])),
        prepare_spatial_weights(w[np.ix_(order, order)]),
    )
    assert original.valid
    assert actual.valid
    for field in (
        "moran",
        "lm_lag",
        "lm_error",
        "robust_lm_lag",
        "robust_lm_error",
        "lm_sarma",
    ):
        expected_test, actual_test = getattr(original, field), getattr(actual, field)
        np.testing.assert_allclose(
            [actual_test.statistic, actual_test.pvalue],
            [expected_test.statistic, expected_test.pvalue],
            rtol=1e-9,
            atol=1e-11,
        )
    np.testing.assert_allclose(
        [actual.moran_expectation, actual.moran_variance],
        [original.moran_expectation, original.moran_variance],
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "cause", ["weighted", "no-intercept", "nonzero-diagonal", "unidentified"]
)
def test_unavailable_spatial_diagnostic_assumptions_do_not_yield_pvalues(
    cause: str,
) -> None:
    """Unsupported OLS/graph contracts cannot produce seemingly valid tests."""
    rng = np.random.default_rng(739)
    n = 20
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = rng.normal(size=n)
    w = np.roll(np.eye(n), 1, axis=1)
    if cause == "no-intercept":
        x = x[:, 1:]
    if cause == "unidentified":
        x[:, 1] = 1
    if cause == "nonzero-diagonal":
        w += np.eye(n) * 0.1
    fit = (
        LeastSquares().estimate(LinearProblem(x, y, np.linspace(0.5, 2.0, n)))
        if cause == "weighted"
        else LeastSquares().estimate(LinearProblem(x, y))
    )
    actual = spatial_diagnostics(
        fit, prepare_spatial_weights(w, allow_diagonal=cause == "nonzero-diagonal")
    )
    assert not actual.valid
    for test in (
        actual.moran,
        actual.lm_lag,
        actual.lm_error,
        actual.robust_lm_lag,
        actual.robust_lm_error,
        actual.lm_sarma,
    ):
        assert np.isnan(test.statistic)
        assert np.isnan(test.pvalue)
