"""Independent checks for the numerical examples in the published notebooks."""

import runpy
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.design import DistanceToLandmarks, Intercept, Linear
from statadic.api.inference import HC, ModelBased
from statadic.api.ordination import DistanceGram, Ordination, PrincipalCoordinates
from statadic.api.regression import Regression
from statadic.api.representation import MMD, Energy, Wasserstein1D
from statadic.discrepancy.mmd.energy import energy_distance
from statadic.geometry.ground.metrics import angular_distance
from statadic.geometry.ground.similarities import RBFKernel
from statadic.sample.packed import pack

DOCS = Path(__file__).resolve().parents[3] / "packages/statadic/docs"
pytestmark = pytest.mark.native


def notebook(path):
    """Load the same reusable functions that the rendered notebook calls."""
    return runpy.run_path(str(DOCS / path))


def reference_energy(x, y):
    """Compute Euclidean V-energy independently with NumPy."""
    x, y = np.asarray(x), np.asarray(y)

    def distance(a, b):
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)

    return 2 * distance(x, y).mean() - distance(x, x).mean() - distance(y, y).mean()


def test_energy_has_analytic_value():
    x = jnp.asarray([[0.0], [1.0]])
    y = jnp.asarray([[2.0], [3.0]])
    assert float(
        energy_distance(pack([x]), pack([y]), metric="euclidean")[0]
    ) == pytest.approx(3, abs=2e-5)


def test_energy_interfaces_match_numpy():
    cloud = notebook("tutorials/energy-and-scale.py")["make_cloud"](42, 16)
    other = 2 * cloud
    raw = float(energy_distance(pack([cloud]), pack([other]), metric="euclidean")[0])
    rooted = float(Energy(ground="euclidean").cross(pack([cloud]), pack([other]))[0, 0])
    assert raw == pytest.approx(reference_energy(cloud, other), abs=2e-5, rel=2e-5)
    assert rooted**2 == pytest.approx(raw, abs=2e-5, rel=2e-5)


def test_geometry_preserves_identity_and_distinguishes_scale():
    measures = notebook("guides/change-geometry.py")["make_measures"]()
    ids = ("original", "scaled")
    euclidean = Energy(ground="euclidean").prepare(measures, row_ids=ids).pairwise()
    angular = Energy(ground="angular").prepare(measures, row_ids=ids).pairwise()
    np.testing.assert_allclose(euclidean.values, [[0, 2], [2, 0]], atol=2e-5)
    np.testing.assert_allclose(angular.values, 0, atol=2e-5)
    assert euclidean.row_ids == angular.row_ids == ids
    assert bool(jnp.isnan(angular_distance(jnp.zeros((1, 2)), jnp.ones((1, 2)))[0, 0]))


def test_permutation_add_one_and_repeatability():
    energy_test = notebook("tutorials/permutation-tests.py")["energy_test"]
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(16, 2)), rng.normal(loc=0.5, size=(16, 2))
    result = energy_test(x, y, 3, 19)
    draws = np.asarray(result.null_distribution)
    expected_p = (1 + np.count_nonzero(draws >= float(result.statistic))) / 20
    assert float(result.pvalue) == pytest.approx(expected_p, abs=1e-6)
    assert float(result.statistic) == pytest.approx(
        8 * reference_energy(x, y), abs=5e-4, rel=5e-4
    )
    np.testing.assert_array_equal(draws, energy_test(x, y, 3, 19).null_distribution)


def test_regression_matches_independent_least_squares():
    chapter = notebook("tutorials/ols/fit.py")
    data = chapter["make_data"]()
    fit = chapter["fit_data"](data)
    design = np.column_stack([np.ones(len(data["x"])), data["x"]])
    expected = np.linalg.lstsq(design, data["y"], rcond=None)[0]
    np.testing.assert_allclose(fit.parameters, expected, rtol=2e-5, atol=2e-5)
    robust = fit.infer(covariance=HC("HC3"))
    classical = fit.infer(covariance=ModelBased())
    np.testing.assert_array_equal(
        robust.uncertainty.fit.parameters, classical.uncertainty.fit.parameters
    )


@pytest.mark.parametrize("geometry", ["Energy", "RBF MMD", "1D W2"])
def test_measure_workflow_separates_training_and_held_out(geometry):
    data = notebook("tutorials/distributions.py")["make_observations"](7, 1.0)
    training, held_out = data.take(tuple(range(18))), data.take(tuple(range(18, 24)))
    comparison = {
        "Energy": Energy(),
        "RBF MMD": MMD(kernel=RBFKernel(bandwidth=1.0)),
        "1D W2": Wasserstein1D(p=2),
    }[geometry]
    prepared = comparison.prepare(
        training.measure("distribution"), row_ids=training.row_ids
    )
    distances = np.asarray(prepared.pairwise().values)
    embedding = Ordination(DistanceGram(), PrincipalCoordinates(2)).estimate(prepared)
    query = embedding.transform(held_out.measure("distribution"))
    landmarks = ("unit-1", "unit-10")
    terms = (
        Intercept()
        + Linear("price")
        + DistanceToLandmarks("distribution", geometry=comparison, landmarks=landmarks)
    )
    fit = Regression(response="outcome", terms=terms).estimate(training).require_valid()
    predicted = fit.predict(held_out)
    assert distances.shape == (18, 18)
    np.testing.assert_allclose(distances, distances.T, atol=2e-5)
    assert embedding.coordinates.shape == (18, 2) and query.shape == (6, 2)
    assert predicted.shape == (6,) and np.isfinite(predicted).all()
    assert set(landmarks).issubset(training.row_ids)
    assert set(training.row_ids).isdisjoint(held_out.row_ids)


def test_calibration_reports_all_scenarios():
    calibration = notebook("tutorials/calibration.py")["calibration"]
    rows = calibration((0.0, 0.75, 1.5), trials=5, permutations=19, seed=123)
    assert rows.shape == (3, 3)
    assert np.all((rows[:, 0] >= 0) & (rows[:, 0] <= 1))
    assert np.all(rows[:, 1] <= rows[:, 0]) and np.all(rows[:, 0] <= rows[:, 2])
