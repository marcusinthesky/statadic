"""Independent spatial algebra witnesses and native procedure smoke checks."""

import numpy as np
import pytest
from statadic.estimation.spatial import (
    KKP,
    SAC,
    SAR,
    SEM,
    SpatialGMM,
    SpatialIV,
    SpatialPanelProblem,
    SpatialProblem,
)
from statadic.operators.spatial import (
    prepare_spatial_panel_weights,
    prepare_spatial_weights,
)
from statadic_studies.adapters.dependence.spatial_extensions import (
    kkp_data,
    observed_data,
    observed_fixture,
    run_spatial_extensions,
    spatial_fixture,
    spatial_outcome,
)
from statadic_studies.definitions.dependence.spatial_extensions import (
    SPATIAL_EXTENSION_STUDIES,
)


def test_spatial_iv_matches_independent_projection():
    x, w = spatial_fixture()
    y = spatial_outcome(np.random.default_rng(72), x, w, rho=0.25)
    fit = SpatialIV(instrument_columns=(1, 2)).estimate(
        SpatialProblem(x, y, prepare_spatial_weights(w), process=SAR(), slx_columns=())
    )
    design = np.column_stack((x, w @ y))
    instruments = np.column_stack((x, w @ x[:, 1:]))
    projected = instruments @ np.linalg.lstsq(instruments, design, rcond=None)[0]
    expected = np.linalg.lstsq(projected, y, rcond=None)[0]
    assert fit.valid
    np.testing.assert_allclose(fit.parameters, expected, rtol=5e-5, atol=3e-7)


def test_spatial_gmm_actual_filter_matches_numpy():
    x, w = spatial_fixture()
    y = spatial_outcome(np.random.default_rng(73), x, w, lam=0.3)
    fit = SpatialGMM().estimate(
        SpatialProblem(
            x,
            y,
            prepare_spatial_weights(w),
            process=SAC() if False else SEM(),
            slx_columns=(),
        )
    )
    assert fit.valid
    operator = np.eye(len(w)) - float(fit.coefficient_lambda) * w
    expected = np.linalg.lstsq(operator @ x, operator @ y, rcond=None)[0]
    np.testing.assert_allclose(fit.parameters[:-1], expected, rtol=5e-5, atol=3e-7)


def test_kkp_actual_whitening_matches_numpy():
    x, y, w = kkp_data(np.random.default_rng(74))
    fit = KKP().estimate(
        SpatialPanelProblem(
            x,
            y,
            prepare_spatial_panel_weights(w, period_ids=np.arange(len(x))),
            process=SEM(),
            effects="random",
            slx_columns=(),
        )
    )
    assert fit.valid
    operator = np.eye(len(w)) - float(fit.spatial_lambda) * w
    xx = np.einsum("ij,tjp->tip", operator, x)
    yy = y @ operator.T
    theta = 1 - np.sqrt(float(fit.variance_components[0] / fit.variance_components[1]))
    xx = (xx - theta * xx.mean(axis=0)).reshape(-1, x.shape[-1])
    yy = (yy - theta * yy.mean(axis=0)).ravel()
    np.testing.assert_allclose(
        fit.parameters, np.linalg.lstsq(xx, yy, rcond=None)[0], rtol=5e-5, atol=3e-7
    )


def test_observed_data_preserves_graph_not_selected_subgraph():
    problem = observed_data(np.random.default_rng(75))
    x, graphs, selections, _ = observed_fixture()
    for period, design, outcome, xx, graph, selected in zip(
        problem.periods,
        problem.design,
        problem.outcome,
        x,
        graphs,
        selections,
        strict=True,
    ):
        assert design.shape == (12, 2)
        assert outcome.shape == (10,)
        assert tuple(period.observed_indices) == selected
        np.testing.assert_allclose(period.weights.values, graph, rtol=1e-6)
        np.testing.assert_allclose(design, xx, rtol=1e-6)


@pytest.mark.parametrize(
    "study,scenario",
    [
        (study.id, scenario)
        for study in SPATIAL_EXTENSION_STUDIES.values()
        for scenario in study.scenarios
    ],
)
def test_native_spatial_procedure_contracts(study, scenario):
    records = run_spatial_extensions(study, scenario, 0, 123)
    assert tuple(records) == SPATIAL_EXTENSION_STUDIES[study].methods
    for record in records.values():
        assert record["valid"] or record["failure"]
        if record["valid"]:
            assert all(
                np.isfinite(value)
                for value in record.values()
                if isinstance(value, float)
            )
