"""Independent finite sandwich formulas and dependent-procedure smoke runs."""

import numpy as np
import pytest
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.estimation.panel import FamaMacBeth, PanelProblem, Within
from statadic.inference.linear_covariance import cluster_covariance, hac_covariance
from statadic.inference.panel_covariance import (
    fama_macbeth_covariance,
    panel_covariance,
)
from statadic.operators.panel import prepare_panel_index
from statadic_studies.adapters.dependence.covariance import (
    fama_data,
    linear_covariance_data,
    panel_covariance_data,
    run_covariance_dependence,
)
from statadic_studies.definitions.dependence.covariance import (
    COVARIANCE_DEPENDENCE_STUDIES,
)


def _window(times, bandwidth, kernel):
    distance = np.abs(times[:, None] - times[None, :])
    if bandwidth == 0:
        return (distance == 0).astype(float)
    if kernel == "bartlett":
        return np.maximum(1 - distance / (bandwidth + 1), 0)
    if kernel == "parzen":
        u = distance / (bandwidth + 1)
        return np.where(
            u <= 0.5, 1 - 6 * u**2 + 6 * u**3, np.where(u <= 1, 2 * (1 - u) ** 3, 0)
        )
    z = 6 * np.pi * np.where(distance == 0, 1, distance) / bandwidth / 5
    return np.where(distance == 0, 1, 3 * (np.sin(z) / z - np.cos(z)) / z**2)


def test_linear_bartlett_and_cluster_against_numpy_sandwich():
    x, y, groups = linear_covariance_data(np.random.default_rng(15), "serial-AR1")
    fit = LeastSquares().estimate(LinearProblem(x, y))
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    influence = (x @ np.linalg.inv(x.T @ x)) * (y - x @ beta)[:, None]
    uncertainty = hac_covariance(fit, 4)
    assert uncertainty.valid
    expected = influence.T @ _window(np.arange(len(x)), 4, "bartlett") @ influence
    np.testing.assert_allclose(
        uncertainty.covariance.values, expected, rtol=3e-5, atol=3e-8
    )
    totals = np.stack([influence[groups == g].sum(axis=0) for g in np.unique(groups)])
    uncertainty = cluster_covariance(
        fit, groups, n_clusters=32, finite_sample_correction=32 / 31
    )
    assert uncertainty.valid
    np.testing.assert_allclose(
        uncertainty.covariance.values, totals.T @ totals * 32 / 31, rtol=3e-5, atol=3e-8
    )


@pytest.fixture(scope="module")
def panel_sample():
    x, y, entity, times = panel_covariance_data(
        np.random.default_rng(16), "gapped-time"
    )
    fit = Within(tolerance=1e-6).estimate(
        PanelProblem(
            x, y, prepare_panel_index(entity, times), effects="unit", intercept_column=0
        )
    )
    xx = (
        x.reshape(24, 24, 2)
        - x.reshape(24, 24, 2).mean(axis=1, keepdims=True)
        + x.mean(axis=0)
    ).reshape(-1, 2)
    yy = (
        y.reshape(24, 24) - y.reshape(24, 24).mean(axis=1, keepdims=True) + y.mean()
    ).ravel()
    beta = np.linalg.lstsq(xx, yy, rcond=None)[0]
    influence = (xx @ np.linalg.inv(xx.T @ xx)) * (yy - xx @ beta)[:, None]
    assert fit.valid
    np.testing.assert_allclose(fit.parameters, beta, rtol=3e-5, atol=3e-7)
    return fit, influence, entity, times


@pytest.mark.parametrize(
    "method", ["cluster-entity", "cluster-time", "cluster-two-way"]
)
def test_panel_cluster_inclusion_exclusion(panel_sample, method):
    fit, influence, entity, times = panel_sample
    entity_totals = np.stack(
        [influence[entity == g].sum(axis=0) for g in np.unique(entity)]
    )
    time_totals = np.stack(
        [influence[times == t].sum(axis=0) for t in np.unique(times)]
    )
    left, right = entity_totals.T @ entity_totals, time_totals.T @ time_totals
    expected = (
        left
        if method == "cluster-entity"
        else right
        if method == "cluster-time"
        else left + right - influence.T @ influence
    )
    uncertainty = panel_covariance(
        fit, method=method, debiased=False, count_effects=False
    )
    assert uncertainty.valid
    np.testing.assert_allclose(
        uncertainty.covariance.values, expected, rtol=4e-5, atol=3e-8
    )


@pytest.mark.parametrize("method", ["autocorrelation", "driscoll-kraay"])
@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "qs"])
def test_panel_kernel_uses_actual_integer_time_distance(panel_sample, method, kernel):
    fit, influence, _, times = panel_sample
    period_ids = np.unique(times)
    windows = _window(period_ids, 4, kernel)
    cube = influence.reshape(24, 24, 2)
    if method == "driscoll-kraay":
        totals = cube.sum(axis=0)
        expected = totals.T @ windows @ totals
    else:
        expected = sum(rows.T @ windows @ rows for rows in cube)
    uncertainty = panel_covariance(
        fit,
        method=method,
        bandwidth=4,
        kernel=kernel,
        debiased=False,
        count_effects=False,
    )
    assert uncertainty.valid
    np.testing.assert_allclose(
        uncertainty.covariance.values, expected, rtol=4e-5, atol=3e-8
    )


def test_fama_cross_sections_and_iid_sampling_variance_against_numpy():
    x, y, entity, times = fama_data(np.random.default_rng(17), "independent-slopes")
    expected = np.stack(
        [
            np.linalg.lstsq(x[times == t], y[times == t], rcond=None)[0]
            for t in np.unique(times)
        ]
    )
    fit = FamaMacBeth().estimate(
        PanelProblem(x, y, prepare_panel_index(entity, times), effects="pooled")
    )
    uncertainty = fama_macbeth_covariance(fit)
    assert fit.valid and uncertainty.valid
    np.testing.assert_allclose(
        fit.parameters, expected.mean(axis=0), rtol=4e-5, atol=3e-7
    )
    np.testing.assert_allclose(
        uncertainty.covariance.values,
        np.cov(expected, rowvar=False, ddof=1) / len(expected),
        rtol=4e-5,
        atol=3e-8,
    )


@pytest.mark.parametrize(
    "study,scenario",
    [
        (study.id, scenario)
        for study in COVARIANCE_DEPENDENCE_STUDIES.values()
        for scenario in study.scenarios
    ],
)
def test_native_dependent_inference_contracts(study, scenario):
    records = run_covariance_dependence(study, scenario, 0, 42)
    assert tuple(records) == COVARIANCE_DEPENDENCE_STUDIES[study].methods
    assert records == run_covariance_dependence(study, scenario, 0, 42)
    for record in records.values():
        assert record["valid"] or record["failure"]
        if record["valid"]:
            assert all(
                np.isfinite(value)
                for value in record.values()
                if isinstance(value, float)
            )
