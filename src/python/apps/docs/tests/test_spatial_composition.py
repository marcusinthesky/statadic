"""Exercise the documented spatial survival composition on the real package."""

import runpy
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.native
EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "packages/statadic/docs/tutorials/spatial-panel-survival.py"
)


def test_spatial_composition_keeps_baseline_and_inference_contracts():
    from statadic.api.design import Linear
    from statadic.api.inference import ModelBased
    from statadic.api.regression import Regression
    from statadic.api.spatial.graphs import SpatialLag
    from statadic.api.survival import RightCensored, SpatialSurvivalData
    from statadic.estimation.survival import SpatialSurvivalML
    from statadic.model.survival import LogNormalAFT

    _, values = runpy.run_path(str(EXAMPLE))["app"].run()
    pooled, panel = values["pooled"], values["time_fit"]
    data, streams = values["panel_data"], values["streams"]
    baseline_model = Regression(
        RightCensored("time", "event"),
        Linear("x"),
        structure=data.panel,
        law=LogNormalAFT(),
        dependence=SpatialLag("neighbors"),
        estimator=SpatialSurvivalML(rho_fixed=0.0),
        effects="time",
    )
    baseline = baseline_model.estimate(
        SpatialSurvivalData(data, streams)
    ).require_valid()
    spatial = {
        0.0: (
            baseline,
            baseline.infer(covariance=ModelBased()),
            baseline.predict(data, np.array(1.0), kind="survival"),
        ),
        0.2: (values["fit"], values["inference"], values["prediction"]),
    }
    assert bool(pooled.core.valid)
    assert bool(panel.core.valid)
    spells = panel.context.observations
    events = np.asarray(spells.numeric("event"))
    times = np.asarray(spells.numeric("time"))
    assert np.count_nonzero(events == 0) == 1
    assert events[0] == 0
    assert times[0] == 1.0
    baseline, baseline_inference, baseline_prediction = spatial[0.0]
    dependent, dependent_inference, dependent_prediction = spatial[0.2]
    for fit, inference, prediction in (spatial[0.0], spatial[0.2]):
        assert bool(fit.core.valid)
        assert bool(fit.core.solver.converged)
        assert bool(fit.core.integration_valid)
        assert bool(inference.uncertainty.valid)
        assert inference.uncertainty.fit is fit.core
        assert len(prediction.values) == 3
        for values in prediction.values:
            assert np.all((np.asarray(values) >= 0) & (np.asarray(values) <= 1))
    # The zero-lag spatial model reduces to the independent lognormal panel
    # likelihood for the same censoring pattern, data, and effects.
    np.testing.assert_allclose(
        baseline.core.parameters, panel.core.parameters, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        baseline.core.log_likelihood, panel.core.log_likelihood, rtol=1e-3, atol=1e-4
    )
    assert not np.allclose(
        baseline_prediction.values[0], dependent_prediction.values[0]
    )
    assert baseline.core.rho_fixed == 0.0
    assert dependent.core.rho_fixed == 0.2
    assert baseline_inference.report().metadata["reference"] is None
    assert dependent_inference.report().metadata["reference"] is None
