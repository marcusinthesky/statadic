"""Must run against the REAL branch-matched STATADIC. No import-or-skip fallback."""

import numpy as np
import pytest
from statadic_docs_labs.experiments import add_one_pvalue, reference_raw_energy

pytestmark = pytest.mark.native


@pytest.fixture(scope="module")
def native():
    from statadic_docs_labs import native

    return native


def test_native_energy_has_analytic_value(native):
    x = native.jnp.asarray([[0.0], [1.0]])
    y = native.jnp.asarray([[2.0], [3.0]])
    raw = float(
        native.energy_distance(
            native.PackedClouds.from_array(x),
            native.PackedClouds.from_array(y),
            metric="euclidean",
            exponent=1.0,
        )
    )
    assert raw == pytest.approx(3, abs=2e-5)


def test_both_energy_interfaces_agree_on_explicit_euclidean_convention(native):
    result = native.energy_result(n=16, ground="euclidean")
    # Quantize the independent witness to the represented native input dtype.
    dtype = np.dtype(result["dtype"])
    expected = reference_raw_energy(
        result["x"].astype(dtype), result["y"].astype(dtype)
    )
    assert result["raw"] == pytest.approx(expected, abs=2e-5, rel=2e-5)
    assert result["rooted"] ** 2 == pytest.approx(result["raw"], abs=2e-5, rel=2e-5)


def test_native_geometry_and_domain(native):
    result = native.geometry_result(3, 0)
    assert result["euclidean"] == pytest.approx(2)
    assert result["angular"] == pytest.approx(0, abs=1e-5)
    with pytest.raises(ValueError, match="zero vector"):
        native.geometry_result(0, 90)


def test_native_permutation_pvalue_and_scaled_statistic(native):
    result = native.permutation_result(n=16, permutations=19, seed=3)
    assert result["pvalue"] == pytest.approx(
        add_one_pvalue(result["observed"], result["null"]), abs=1e-6
    )
    expected = 8 * reference_raw_energy(result["x"], result["y"])
    assert result["observed"] == pytest.approx(expected, abs=5e-4, rel=5e-4)
    repeat = native.permutation_result(n=16, permutations=19, seed=3)
    np.testing.assert_array_equal(result["null"], repeat["null"])


def test_native_regression_matches_independent_least_squares(native):
    result = native.regression_result()
    design = np.column_stack([np.ones(len(result["x"])), result["x"]])
    expected = np.linalg.lstsq(design, result["y"], rcond=None)[0]
    np.testing.assert_allclose(result["coefficients"], expected, rtol=2e-5, atol=2e-5)
    classical = native.regression_result(covariance="Model-based")
    np.testing.assert_array_equal(result["coefficients"], classical["coefficients"])
    assert result["valid"] and classical["valid"]


@pytest.mark.parametrize("geometry", ["Energy", "RBF MMD", "1D W2"])
def test_native_measure_workflow_separates_training_and_held_out(native, geometry):
    result = native.distribution_result(geometry=geometry)
    assert result["distances"].shape == (18, 18)
    np.testing.assert_allclose(result["distances"], result["distances"].T, atol=2e-5)
    assert result["coordinates"].shape == (18, 2)
    assert result["query_coordinates"].shape == (6, 2)
    assert result["predicted"].shape == (6,)
    assert set(result["landmarks"]).issubset(result["training_ids"])
    assert set(result["training_ids"]).isdisjoint(result["held_out_ids"])


def test_native_small_calibration_run_has_all_trials(native):
    rows = native.calibration_result(n=8, permutations=19, trials=5)
    assert len(rows) == 3
    for row in rows:
        assert row["trials"] == 5
        assert 0 <= row["lower"] <= row["rate"] <= row["upper"] <= 1
