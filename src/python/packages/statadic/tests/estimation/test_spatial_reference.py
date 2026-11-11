# standalone package-maintenance entry point
"""Spatial fitted-state and covariance parity against actual spreg estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.core.results import InferenceStatus
from statadic.estimation.spatial import (
    ML,
    SAC,
    SAR,
    SEM,
    SLXLeastSquares,
    SpatialIV,
    SpatialIVProblem,
    SpatialProblem,
)
from statadic.inference.parameters import coefficient_intervals, linear_restriction_test
from statadic.inference.spatial import spatial_likelihood_ratio
from statadic.inference.spatial.covariance import spatial_covariance, spatial_impacts
from statadic.operators.spatial import prepare_spatial_weights
from scipy import stats

from .._budgets import examples

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from statadic.estimation.spatial.iv import SpatialIVFit
    from statadic.estimation.spatial.likelihood import SpatialFit

_PROCESS = {"sar": SAR, "sem": SEM}


def _sample(seed: int = 126) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 40
    w = 0.8 * np.roll(np.eye(n), 1, axis=1) + 0.2 * np.roll(np.eye(n), -1, axis=1)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    y = np.linalg.solve(
        np.eye(n) - 0.35 * w, x @ np.array([0.4, -0.5, 0.8]) + rng.normal(size=n) * 0.4
    )
    return x, y, w


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
@pytest.mark.parametrize("model", ["sar", "sem"])
@pytest.mark.parametrize("durbin", [False, True])
@examples(6)
@example(seed=126)
@given(seed=st.integers(0, 2**32 - 1))
def test_spatial_ml_state_and_information_match_spreg(
    optional_reference: Callable[[str], ModuleType],
    *,
    model: Literal["sar", "sem"],
    durbin: bool,
    seed: int,
) -> None:
    """Match full Gaussian likelihood and expected-information covariance."""
    spreg = optional_reference("spreg")
    libpysal = optional_reference("libpysal")
    x, y, w = _sample(seed)
    fit = ML().estimate(
        SpatialProblem(
            x,
            y,
            prepare_spatial_weights(w),
            process=_PROCESS[model](),
            slx_columns=(1, 2) if durbin else (),
        )
    )
    cls = spreg.ML_Lag if model == "sar" else spreg.ML_Error
    oracle = cls(
        y[:, None],
        x[:, 1:],
        libpysal.weights.full2W(w),
        slx_lags=1 if durbin else 0,
        epsilon=1e-9,
    )
    assert fit.valid
    np.testing.assert_allclose(
        fit.parameters, np.asarray(oracle.betas).ravel(), atol=2e-6
    )
    np.testing.assert_allclose(fit.log_likelihood, oracle.logll, atol=2e-7)
    np.testing.assert_allclose(
        fit.sigma2, np.asarray(oracle.sig2).ravel()[0], atol=2e-7
    )
    covariance = spatial_covariance(fit)
    assert covariance.valid
    np.testing.assert_allclose(
        covariance.covariance.values, oracle.vm, rtol=1e-5, atol=1e-7
    )
    intervals = coefficient_intervals(covariance, reference="normal")
    assert intervals.valid
    np.testing.assert_allclose(
        intervals.estimate.se, np.asarray(oracle.std_err).ravel(), rtol=1e-5, atol=1e-7
    )
    restriction = np.zeros((1, fit.parameters.shape[-1]))
    restriction[0, -1] = 1
    test = linear_restriction_test(covariance, restriction, 0.0, reference="chi2")
    np.testing.assert_allclose(
        test.test.statistic, oracle.z_stat[-1][0] ** 2, rtol=1e-5
    )
    assert test.valid
    assert test.numerator_df == 1
    np.testing.assert_allclose(
        test.test.pvalue, oracle.z_stat[-1][1], rtol=1e-5, atol=1e-9
    )
    expected_ci = (
        np.asarray(oracle.betas).ravel()[None, :]
        + np.array([-1.0, 1.0])[:, None]
        * stats.norm.ppf(0.975)
        * np.asarray(oracle.std_err).ravel()[None, :]
    )
    np.testing.assert_allclose(intervals.estimate.ci, expected_ci, rtol=1e-5, atol=2e-6)
    impacts = spatial_impacts(covariance)
    assert impacts.valid
    np.testing.assert_allclose(
        impacts.values[:, 0] + impacts.values[:, 1], impacts.values[:, 2], atol=1e-12
    )
    np.testing.assert_allclose(
        fit.fitted_values, np.asarray(oracle.predy).ravel(), atol=2e-6
    )
    np.testing.assert_allclose(fit.residuals, np.asarray(oracle.u).ravel(), atol=2e-6)
    np.testing.assert_allclose(
        fit.innovations,
        fit.residuals
        - (float(fit.parameters[-1]) * (w @ fit.residuals) if model == "sem" else 0),
        atol=1e-10,
    )
    if model == "sar":
        np.testing.assert_allclose(
            fit.reduced_form, np.asarray(oracle.predy_e).ravel(), atol=3e-6
        )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
@pytest.mark.parametrize("robust", [False, True])
@examples(6)
@example(seed=126)
@given(seed=st.integers(0, 2**32 - 1))
def test_identified_spatial_iv_matches_spreg(
    optional_reference: Callable[[str], ModuleType],
    *,
    robust: bool,
    seed: int,
) -> None:
    """Match ordinary and heteroskedastic 2SLS covariance with actual instruments."""
    spreg = optional_reference("spreg")
    libpysal = optional_reference("libpysal")
    x, y, w = _sample(seed)
    fit = SpatialIV(instrument_columns=(1, 2)).estimate(
        SpatialProblem(x, y, prepare_spatial_weights(w), process=SAR(), slx_columns=())
    )
    oracle = spreg.GM_Lag(
        y[:, None],
        x[:, 1:],
        w=libpysal.weights.full2W(w),
        w_lags=1,
        robust="white" if robust else None,
        sig2n_k=False,
    )
    assert fit.valid
    assert fit.first_stage_rank == fit.design.shape[1]
    np.testing.assert_allclose(
        fit.parameters, np.asarray(oracle.betas).ravel(), atol=1e-8
    )
    covariance = spatial_covariance(
        fit, method="heteroskedastic" if robust else "homoskedastic"
    )
    np.testing.assert_allclose(
        covariance.covariance.values, oracle.vm, rtol=1e-7, atol=1e-8
    )
    assert covariance.valid
    np.testing.assert_allclose(fit.residuals, np.asarray(oracle.u).ravel(), atol=1e-8)
    intervals = coefficient_intervals(covariance, reference="normal")
    assert intervals.valid
    np.testing.assert_allclose(
        intervals.estimate.se, np.asarray(oracle.std_err).ravel(), rtol=1e-7, atol=1e-8
    )


@pytest.mark.component
@pytest.mark.contract
@examples(6)
@example(seed=126)
@given(seed=st.integers(0, 2**32 - 1))
def test_spatial_ml_jit_vmap_and_iv_rank_failure(seed: int) -> None:
    """Qualify transformed fitting and reject intercept-only instruments."""
    x, y, w = _sample(seed)
    weights = prepare_spatial_weights(w)

    def evaluate_fit(z: jax.Array) -> SpatialFit:
        return ML().estimate(
            SpatialProblem(jnp.asarray(x), z, weights, process=SAR(), slx_columns=())
        )

    evaluate_fit = jax.jit(evaluate_fit)(jnp.asarray(y))
    assert evaluate_fit.valid

    def evaluate_fit_3(z: jax.Array) -> SpatialFit:
        return ML().estimate(
            SpatialProblem(jnp.asarray(x), z, weights, process=SAR(), slx_columns=())
        )

    batch = jax.jit(jax.vmap(evaluate_fit_3))(jnp.asarray(np.stack([y, y * 2])))
    assert np.all(batch.valid)
    np.testing.assert_allclose(
        batch.parameters[:, -1], evaluate_fit.parameters[-1], atol=1e-6
    )
    failed = SpatialIV().estimate(
        SpatialProblem(np.ones((len(y), 1)), y, weights, process=SAR(), slx_columns=())
    )
    assert not failed.valid
    assert np.all(np.isnan(failed.parameters))


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
@examples(6)
@example(seed=126)
@given(seed=st.integers(0, 2**32 - 1))
def test_slx_fit_covariance_prediction_and_intervals_match_statsmodels(
    optional_reference: Callable[[str], ModuleType],
    seed: int,
) -> None:
    """Exercise public SLX through the shared linear inference path."""
    statsmodels = optional_reference("statsmodels.api")
    from statadic.inference.linear_covariance import hc_covariance  # noqa: PLC0415

    x, y, w = _sample(seed)
    fit = SLXLeastSquares().estimate(
        SpatialProblem(
            x, y, prepare_spatial_weights(w), process=None, slx_columns=(1, 2)
        )
    )
    oracle = statsmodels.OLS(y, np.column_stack([x, w @ x[:, 1:]])).fit(
        cov_type="HC1", use_t=True
    )
    uncertainty = hc_covariance(fit, kind="HC1")
    intervals = coefficient_intervals(
        uncertainty, reference="t", degrees_of_freedom=fit.df_resid
    )
    assert intervals.valid
    np.testing.assert_allclose(fit.parameters, oracle.params, atol=1e-10)
    np.testing.assert_allclose(
        uncertainty.covariance.values, oracle.cov_params(), atol=1e-10
    )
    np.testing.assert_allclose(
        intervals.estimate.ci, np.asarray(oracle.conf_int()).T, atol=1e-8
    )


@pytest.mark.integration
@pytest.mark.reference
@examples(6)
@example(seed=126)
@given(seed=st.integers(0, 2**32 - 1))
def test_combined_ml_matches_independent_scipy_joint_likelihood(seed: int) -> None:
    """Compare GNS beta/rho/lambda to an independent Gaussian objective."""
    from scipy.optimize import minimize  # noqa: PLC0415

    x, y, w = _sample(seed)
    n = len(y)
    design = np.column_stack([x, w @ x[:, 1:]])
    fit = ML().estimate(
        SpatialProblem(
            x, y, prepare_spatial_weights(w), process=SAC(), slx_columns=(1, 2)
        )
    )

    def objective(spatial: np.ndarray) -> float:
        a = np.eye(n) - spatial[0] * w
        b = np.eye(n) - spatial[1] * w
        xx = b @ design
        yy = b @ a @ y
        coefficient = np.linalg.lstsq(xx, yy, rcond=None)[0]
        residual = yy - xx @ coefficient
        return float(
            n / 2 * (np.log(2 * np.pi * np.mean(residual**2)) + 1)
            - np.linalg.slogdet(a)[1]
            - np.linalg.slogdet(b)[1]
        )

    candidates = [
        minimize(
            objective,
            np.array([a, b]),
            bounds=[(-0.999, 0.999)] * 2,
            method="L-BFGS-B",
            options={"ftol": 1e-14, "gtol": 1e-7},
        )
        for a in (-0.5, 0.0, 0.5)
        for b in (-0.5, 0.0, 0.5)
    ]
    oracle = min(candidates, key=lambda result: result.fun)
    assert fit.valid
    np.testing.assert_allclose(fit.parameters[-2:], oracle.x, atol=3e-5)
    np.testing.assert_allclose(fit.log_likelihood, -oracle.fun, atol=1e-7)


@pytest.mark.integration
@pytest.mark.reference
@examples(6)
@example(seed=126)
@given(seed=st.integers(0, 2**32 - 1))
def test_named_prediction_preserves_graph_and_schema(seed: int) -> None:
    """Reduced-form prediction uses new X and explicit spatial links."""
    x, y, w = _sample(seed)
    weights = prepare_spatial_weights(w)
    fit = ML().estimate(
        SpatialProblem(
            x,
            y,
            weights,
            parameter_names=("intercept", "a", "b"),
            process=SAR(),
            slx_columns=(),
        )
    )
    new = x.copy()
    new[:, 1] *= 2
    actual = fit.predict(new, parameter_names=("intercept", "a", "b"))
    expected = np.linalg.solve(
        np.eye(len(y)) - float(fit.parameters[-1]) * w,
        new @ np.asarray(fit.parameters[:-1]),
    )
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    with pytest.raises(ValueError, match="names"):
        fit.predict(new, parameter_names=("intercept", "b", "a"))


@pytest.mark.integration
@pytest.mark.reference
@examples(6)
@example(seed=126)
@given(seed=st.integers(0, 2**32 - 1))
def test_impacts_and_nested_likelihood_against_explicit_maps(seed: int) -> None:
    """Check exact derivative maps, finite-difference delta covariance and LR."""
    chi2 = pytest.importorskip("scipy.stats").chi2

    x, y, w = _sample(seed)
    graph = prepare_spatial_weights(w)
    fit = ML().estimate(SpatialProblem(x, y, graph, process=SAR(), slx_columns=(1, 2)))
    uncertainty = spatial_covariance(fit)
    actual = spatial_impacts(uncertainty)

    def oracle(parameters: np.ndarray) -> np.ndarray:
        inverse = np.linalg.inv(np.eye(len(y)) - parameters[-1] * w)
        effects = []
        for column in range(x.shape[1]):
            theta = 0.0 if column == 0 else parameters[x.shape[1] + column - 1]
            derivative = inverse @ (parameters[column] * np.eye(len(y)) + theta * w)
            direct, total = np.trace(derivative) / len(y), derivative.sum() / len(y)
            effects.append([direct, total - direct, total])
        return np.asarray(effects)

    parameters = np.asarray(fit.parameters)
    step = 1e-5
    perturbations = np.eye(len(parameters)) * step
    gradient = np.column_stack(
        [
            (
                (oracle(parameters + delta) - oracle(parameters - delta)) / (2 * step)
            ).ravel()
            for delta in perturbations
        ]
    )
    np.testing.assert_allclose(actual.values, oracle(parameters), atol=1e-10)
    np.testing.assert_allclose(
        actual.covariance,
        gradient @ uncertainty.covariance.values @ gradient.T,
        rtol=1e-7,
        atol=1e-10,
    )
    assert actual.uncertainty_valid
    np.testing.assert_allclose(
        actual.standard_errors,
        np.sqrt(np.diag(gradient @ uncertainty.covariance.values @ gradient.T)).reshape(
            actual.values.shape
        ),
        rtol=1e-7,
        atol=1e-10,
    )
    restricted = ML().estimate(
        SpatialProblem(x, y, graph, process=SAR(), slx_columns=())
    )
    result = spatial_likelihood_ratio(
        fit, restricted, degrees_of_freedom=2, assume_nested=True
    )
    statistic = 2 * (float(fit.log_likelihood) - float(restricted.log_likelihood))
    np.testing.assert_allclose(result.statistic, statistic, atol=1e-10)
    np.testing.assert_allclose(result.pvalue, chi2.sf(statistic, 2), atol=1e-10)


@pytest.mark.component
@pytest.mark.property
@pytest.mark.parametrize("model", ["sar", "sem"])
def test_spatial_likelihood_and_impacts_follow_both_node_axes(
    model: Literal["sar", "sem"],
) -> None:
    """Joint relabeling preserves coefficients and covariance and reorders row state."""
    x, y, w = _sample()
    order = np.random.default_rng(738).permutation(len(y))
    brick = ML()
    original = brick.estimate(
        SpatialProblem(
            x,
            y,
            prepare_spatial_weights(w),
            process=_PROCESS[model](),
            slx_columns=(1, 2),
        )
    )
    permuted = brick.estimate(
        SpatialProblem(
            x[order],
            y[order],
            prepare_spatial_weights(w[np.ix_(order, order)]),
            process=_PROCESS[model](),
            slx_columns=(1, 2),
        )
    )
    assert original.valid
    assert permuted.valid
    np.testing.assert_allclose(
        permuted.parameters, original.parameters, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        permuted.log_likelihood, original.log_likelihood, atol=1e-9
    )
    np.testing.assert_allclose(
        permuted.residuals, original.residuals[order], rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        permuted.reduced_form, original.reduced_form[order], rtol=1e-6, atol=1e-7
    )
    actual, expected = spatial_covariance(permuted), spatial_covariance(original)
    assert actual.valid
    np.testing.assert_allclose(
        actual.covariance.values, expected.covariance.values, rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        spatial_impacts(actual).values,
        spatial_impacts(expected).values,
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.component
@pytest.mark.contract
def test_failed_spatial_fits_cannot_report_available_covariance() -> None:
    """An exhausted optimizer and unidentified instruments retain failure evidence."""
    x, y, w = _sample()
    weights = prepare_spatial_weights(w)
    failed_ml = ML(max_steps=1).estimate(
        SpatialProblem(x, y, weights, process=SAR(), slx_columns=())
    )
    failed_iv = SpatialIV().estimate(
        SpatialProblem(np.ones((len(y), 1)), y, weights, process=SAR(), slx_columns=())
    )
    for fit in (failed_ml, failed_iv):
        assert not fit.valid
        uncertainty = spatial_covariance(fit)
        assert not uncertainty.valid
        assert uncertainty.status == InferenceStatus.INVALID_FIT
        assert np.all(np.isnan(uncertainty.covariance.values))


@pytest.mark.component
@pytest.mark.contract
def test_spatial_iv_batched_parameters_keep_invalid_lanes_unavailable() -> None:
    """Two fit lanes broadcast availability across three coefficient columns."""
    x, y, w = _sample()
    x = x[:, :2]
    graph = prepare_spatial_weights(w)
    excluded = w @ x[:, 1:]

    def evaluate_fit(instruments: jax.Array) -> SpatialIVFit:
        return SpatialIV().estimate(
            SpatialIVProblem(
                jnp.asarray(x),
                jnp.asarray(y),
                graph,
                excluded_instruments=instruments,
                process=SAR(),
                slx_columns=(),
            )
        )

    batch = jax.jit(jax.vmap(evaluate_fit))(
        jnp.asarray(np.stack([excluded, np.zeros_like(excluded)]))
    )
    np.testing.assert_array_equal(batch.valid, [True, False])
    assert np.all(np.isfinite(batch.parameters[0]))
    assert np.all(np.isnan(batch.parameters[1]))
