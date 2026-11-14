"""Independent weighted/panel/spatial witnesses and complete design smoke runs."""

from dataclasses import replace

import jax
import numpy as np
import pytest
from statadic.estimation.dyadic import DyadicEstimator
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.estimation.panel import PanelProblem, Within
from statadic.estimation.spatial import SAR, SLXLeastSquares, SpatialProblem
from statadic.model.dyadic import DyadicModel, DyadicSample
from statadic.operators.panel import prepare_panel_index
from statadic.operators.spatial import prepare_spatial_weights
from statadic_studies.adapters.dependence._dispatch import run_dependence
from statadic_studies.adapters.dependence.dyadic import dyadic_data
from statadic_studies.adapters.dependence.linear_panel import panel_design, wls_data
from statadic_studies.adapters.dependence.spatial import spatial_design
from statadic_studies.definitions.dependence.core import DEPENDENCE_STUDIES


def test_wls_against_numpy_whitening():
    x, y, precision = wls_data(np.random.default_rng(14))
    fit = LeastSquares().estimate(LinearProblem(x, y, precision))
    reference = np.linalg.lstsq(
        np.sqrt(precision[:, None]) * x, np.sqrt(precision) * y, rcond=None
    )[0]
    assert fit.valid
    np.testing.assert_allclose(fit.parameters, reference, rtol=3e-5)


def test_within_slope_against_entity_demeaning():
    x, entity, time = panel_design()
    y = x @ np.array([0.4, 0, -0.7]) + np.random.default_rng(51).normal(
        size=len(entity)
    )
    fit = Within(tolerance=1e-6).estimate(
        PanelProblem(
            x, y, prepare_panel_index(entity, time), effects="unit", intercept_column=0
        )
    )
    dx = (
        x[:, 1:].reshape(24, 6, 2)
        - x[:, 1:].reshape(24, 6, 2).mean(axis=1, keepdims=True)
    ).reshape(-1, 2)
    dy = (y.reshape(24, 6) - y.reshape(24, 6).mean(axis=1, keepdims=True)).ravel()
    expected = np.linalg.lstsq(dx, dy, rcond=None)[0]
    assert fit.valid
    np.testing.assert_allclose(fit.parameters[1:], expected, rtol=3e-5, atol=2e-7)


def test_spatial_graph_direction_and_slx_against_numpy():
    x, w = spatial_design()
    for i in range(len(w)):
        assert w[i, (i + 1) % len(w)] == 0.7
        assert w[i, (i - 2) % len(w)] == 0.3
    design = np.column_stack((x, w @ x[:, 1:]))
    beta = np.array([0.4, -0.5, 0.8, 0.3, -0.2])
    y = design @ beta + np.random.default_rng(81).normal(size=len(x))
    problem = SpatialProblem(x, y, prepare_spatial_weights(w), process=SAR())
    fit = SLXLeastSquares().estimate(replace(problem, process=None, slx_columns=(1, 2)))
    assert fit.valid
    np.testing.assert_allclose(
        fit.parameters, np.linalg.lstsq(design, y, rcond=None)[0], rtol=3e-5, atol=2e-7
    )


@pytest.mark.skipif(
    not jax.config.x64_enabled, reason="Host dyadic rank selection requires x64"
)
def test_dyadic_pooled_coefficient_against_numpy_edges():
    focal, y, left, right = dyadic_data(np.random.default_rng(34), 16)
    fit = DyadicEstimator().estimate(
        DyadicModel(
            ["focal"], {"focal": focal}, DyadicSample.from_arrays(y, left, right)
        )
    )
    expected = np.linalg.lstsq(
        np.column_stack((np.ones(len(y)), focal)), y, rcond=None
    )[0]
    np.testing.assert_allclose(fit.coefficients["focal"], expected[1], rtol=1e-10)


@pytest.mark.parametrize(
    "study,scenario",
    [
        (study.id, scenario)
        for study in DEPENDENCE_STUDIES.values()
        for scenario in study.scenarios
    ],
)
def test_complete_dependence_scenarios(study, scenario):
    result = run_dependence(study, scenario, 0, 18)
    assert tuple(result) == DEPENDENCE_STUDIES[study].methods
    for record in result.values():
        assert record["valid"] or record["failure"]
        if record["valid"]:
            assert all(
                np.isfinite(value)
                for value in record.values()
                if isinstance(value, float)
            )


@pytest.mark.skipif(jax.config.x64_enabled, reason="Only tests unsupported precision")
def test_dyadic_float32_retains_unavailability():
    records = run_dependence("dyadic.point-estimation", "n16", 0, 18)
    assert all(
        not value["valid"] and value["failure"] == "x64-required"
        for value in records.values()
    )
