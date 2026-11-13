"""Independent graph derivatives, filters and conventional diagnostic formulas."""

import os

import jax.numpy as jnp
import numpy as np
import pytest
from _spatial_diagnostic import run_with_diagnostic
from scipy.stats import chi2, norm
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.inference.spatial import spatial_diagnostics
from statadic.operators.spatial import prepare_spatial_weights
from statadic_studies.adapters import run_trial as run_spatial_remaining
from statadic_studies.adapters.dependence.spatial_durbin import durbin_data
from statadic_studies.adapters.dependence.spatial_impacts import exact_impacts
from statadic_studies.adapters.dependence.spatial_panel import varying_panel_data
from statadic_studies.contracts import STUDIES

SPATIAL_REMAINING_STUDIES = {
    key: value
    for key, value in STUDIES.items()
    if key
    in (
        "spatial.durbin-likelihood",
        "spatial.period-varying-panel",
        "spatial.observed-sem",
        "spatial.cross-section-impacts",
        "spatial.observed-panel-impacts",
        "spatial.ols-diagnostics",
        "spatial.ak-diagnostic",
        "spatial.common-factor",
        "spatial.nested-likelihood-ratio",
    )
}


def test_full_graph_impact_matches_independent_finite_difference_of_mean():
    w = np.array([[0.0, 1.0, 0.0], [0.2, 0.0, 0.8], [0.7, 0.3, 0.0]])
    beta, theta, rho = -0.5, 0.2, 0.25
    x = np.array([1.0, -0.3, 2.0])

    def mean(values):
        return np.linalg.solve(np.eye(3) - rho * w, beta * values + theta * w @ values)

    derivative = np.column_stack(
        [(mean(x + delta) - mean(x - delta)) / (2e-5) for delta in np.eye(3) * 1e-5]
    )
    direct, total = np.trace(derivative) / 3, derivative.sum() / 3
    np.testing.assert_allclose(
        exact_impacts(w, beta, theta, rho), [direct, total - direct, total], atol=2e-10
    )


def test_durbin_and_varying_panel_dgp_recover_exact_structural_innovations():
    class ZeroNoise:
        @staticmethod
        def normal(size):
            return np.zeros(size)

    x, y, w = durbin_data(ZeroNoise(), "GNS")
    np.testing.assert_allclose(
        (np.eye(len(w)) - 0.25 * w) @ y,
        x @ np.array([0.3, -0.5, 0.8]) + (w @ x[:, 1:]) @ np.array([0.2, -0.1]),
        atol=1e-14,
    )
    x, y, graphs = varying_panel_data(ZeroNoise(), "fixed")
    assert not np.array_equal(graphs[0], graphs[-1])
    for xx, yy, w in zip(x, y, graphs, strict=True):
        np.testing.assert_allclose(
            (np.eye(len(w)) - 0.25 * w) @ yy, xx @ np.array([-0.5, 0.8]), atol=1e-14
        )


def test_native_delta_impacts_match_independent_matrix_map_and_difference_jacobian():
    from statadic.estimation.spatial import ML, SAR, SpatialProblem
    from statadic.inference.spatial import spatial_covariance, spatial_impacts

    x, y, w = durbin_data(np.random.default_rng(571), "SDM")
    fit = ML().estimate(
        SpatialProblem(
            jnp.asarray(x),
            jnp.asarray(y),
            prepare_spatial_weights(w),
            process=SAR(),
            slx_columns=(1, 2),
        )
    )
    uncertainty = spatial_covariance(fit)
    actual = spatial_impacts(uncertainty)
    assert bool(actual.valid & actual.uncertainty_valid)
    parameters = np.asarray(fit.parameters, dtype=float)

    def effects(parameters):
        return exact_impacts(w, parameters[1], parameters[3], parameters[5])

    steps = np.eye(len(parameters)) * 1e-5
    jacobian = np.column_stack(
        [
            (effects(parameters + step) - effects(parameters - step)) / (2e-5)
            for step in steps
        ]
    )
    expected = jacobian @ np.asarray(uncertainty.covariance.values) @ jacobian.T
    np.testing.assert_allclose(
        actual.values[1], effects(parameters), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        actual.standard_errors[1], np.sqrt(np.diag(expected)), rtol=2e-4, atol=1e-6
    )


def test_native_ols_lm_and_moran_match_independent_projection_formulas():
    generator = np.random.default_rng(6)
    n = 17
    x = np.column_stack([np.ones(n), generator.normal(size=(n, 2))])
    y = x @ np.array([0.2, -0.4, 0.7]) + generator.normal(size=n)
    w = 0.6 * np.roll(np.eye(n), 1, axis=1) + 0.4 * np.roll(np.eye(n), -1, axis=1)
    fit = LeastSquares().estimate(LinearProblem(jnp.asarray(x), jnp.asarray(y)))
    actual = spatial_diagnostics(fit, prepare_spatial_weights(w))
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    projection = np.eye(n) - x @ np.linalg.solve(x.T @ x, x.T)
    sigma2 = residual @ residual / n
    trace = np.trace(w @ w) + np.sum(w * w)
    error = residual @ w @ residual / sigma2
    lag = residual @ w @ y / sigma2
    fitted_lag = w @ x @ beta
    nj = fitted_lag @ projection @ fitted_lag / sigma2 + trace
    expected = [
        lag**2 / nj,
        error**2 / trace,
        (lag - error) ** 2 / (nj - trace),
        (error - trace * lag / nj) ** 2 / (trace * (1 - trace / nj)),
    ]
    tests = [
        actual.lm_lag,
        actual.lm_error,
        actual.robust_lm_lag,
        actual.robust_lm_error,
    ]
    for result, value in zip(tests, expected, strict=True):
        np.testing.assert_allclose(result.statistic, value, rtol=1e-4, atol=2e-6)
        np.testing.assert_allclose(
            result.pvalue, chi2.sf(value, 1), rtol=1e-4, atol=2e-6
        )
    symmetric = (w + w.T) / 2
    mw = projection @ symmetric @ projection
    d = n - x.shape[1]
    expectation = np.trace(mw) / d
    variance = 2 / (d * (d + 2)) * (np.trace(mw @ mw) - np.trace(mw) ** 2 / d)
    moran = residual @ w @ residual / (residual @ residual)
    np.testing.assert_allclose(
        actual.moran.pvalue,
        2 * norm.sf(abs((moran - expectation) / np.sqrt(variance))),
        rtol=1e-4,
        atol=2e-6,
    )


@pytest.mark.parametrize(
    "study,scenario",
    [
        (study.id, scenario)
        for study in SPATIAL_REMAINING_STUDIES.values()
        for scenario in study.scenarios
    ],
)
def test_actual_spatial_procedures(study, scenario):
    records = run_spatial_remaining(study, scenario, 0, 42)
    from statadic_studies.campaign import _validate_outcomes

    _validate_outcomes(records, SPATIAL_REMAINING_STUDIES[study].record(), scenario)
    assert tuple(records) == SPATIAL_REMAINING_STUDIES[study].methods
    assert any(record["valid"] for record in records.values()), records
    for record in records.values():
        assert record["valid"] or record["failure"]
        if record["valid"]:
            if "pvalue" in record:
                assert 0 <= record["pvalue"] <= 1
            if "estimate" in record:
                assert np.isfinite(
                    [record["estimate"], record["truth"], record["width"]]
                ).all()
                assert record["width"] >= 0


def test_native_runtime_failure_is_not_reported_as_statistical_outcome(monkeypatch):
    import statadic_studies.adapters.dependence.spatial_diagnostics as adapters

    def failure(*args, **kwargs):
        raise RuntimeError("injected spatial backend failure")

    monkeypatch.setattr(adapters, "_ak_kernel", failure)
    with pytest.raises(RuntimeError, match="injected spatial backend"):
        run_spatial_remaining("spatial.ak-diagnostic", "independent-errors", 0, 42)


@pytest.mark.parametrize("x64", ["0", "1"])
def test_period_varying_random_panel_retains_precision_and_trial_outcomes(x64):
    """Retain the positive-variance fit and finite inference in both precisions."""
    code = """
import jax.numpy as jnp
import numpy as np
from statadic.estimation.spatial import PanelML, SAR, SpatialPanelProblem
from statadic.inference.spatial import spatial_covariance
from statadic.operators.spatial import prepare_spatial_panel_weights
from statadic_studies.adapters import run_trial
from statadic_studies.adapters._random import rng
from statadic_studies.adapters.dependence.spatial_panel import varying_panel_data
from statadic_studies.campaign import _validate_outcomes
from statadic_studies.contracts import STUDIES

study = STUDIES['spatial.period-varying-panel']
records = run_trial(study.id, 'random', 0, 42)
_validate_outcomes(records, study.record(), 'random')
assert tuple(records) == study.methods
for record in records.values():
    if record['valid']:
        values = [record['estimate'], record['width'], record['pvalue']]
        assert np.isfinite(values).all()
        assert record['width'] >= 0
        assert 0 <= record['pvalue'] <= 1
    else:
        assert record['failure'] == 'fit-or-inference-unavailable'
assert all(record['valid'] for record in records.values()), records
x, y, graphs = varying_panel_data(
    rng(study.id, 'random', 0, seed=42), 'random'
)
weights = prepare_spatial_panel_weights(graphs, period_ids=np.arange(len(x)))
fit = PanelML().estimate(SpatialPanelProblem(
    jnp.asarray(x), jnp.asarray(y), weights,
    process=SAR(), slx_columns=(), effects='random'
))
assert bool(fit.valid & fit.solver.converged)
assert -90.32 < float(fit.log_likelihood) < -90.30
assert 5.3 < float(fit.variance_ratio) < 5.4
assert np.isfinite(spatial_covariance(fit).covariance.values).all()
"""
    environment = os.environ | {
        "JAX_ENABLE_X64": x64,
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    }
    run_with_diagnostic(code, environment)
