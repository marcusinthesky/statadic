"""Independent design, paired estimand and failure-accounting contracts."""

import numpy as np
from statadic_studies.adapters import run_trial
from statadic_studies.adapters.dependence.seasonal_survival import _covariates, _data
from statadic_studies.definitions.dependence.composition import SEASONAL, SLX


def test_cardinal_reference_and_prespecified_censoring():
    phase = np.array([0.0, 0.25, 0.75, 1.0])
    columns = _covariates(phase)
    seasonal = 0.3 * (columns["k1"] - columns["k3"])
    np.testing.assert_allclose(seasonal, [0, 0.3, -0.3, 0], atol=1e-15)
    data, graphs, _, counts = _data("administrative", 0, 20261104, 16)
    assert len(graphs) == len(counts) == 3
    event = np.asarray(data.outcomes.numeric("event")).reshape(3, 8)
    np.testing.assert_array_equal(8 - event.sum(axis=1), counts)
    assert np.all(event[:, 1:] == 1)


def test_named_spatial_lag_matches_external_least_squares():
    for scenario in SLX.scenarios:
        result = run_trial(SLX.id, scenario, 0, 42)
        left, right = (result[m] for m in SLX.methods)
        assert left["valid"] and right["valid"]
        np.testing.assert_allclose(left["estimate"], right["estimate"], atol=1e-5)


def test_two_way_seasonal_contrast_agrees_in_two_coordinate_systems():
    results = run_trial(SEASONAL.id, "exact", 0, 20261104, draws=16)
    left, right = (results[m] for m in SEASONAL.methods)
    assert left["valid"] and right["valid"]
    for field in ("estimate", "width", "log_likelihood", "rho_hat"):
        np.testing.assert_allclose(left[field], right[field], atol=5e-3)
