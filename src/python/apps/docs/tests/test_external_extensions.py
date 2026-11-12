"""Exercise public-only downstream extensions from the published guide."""

import runpy
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.artifacts import ArtifactError
from statadic.api.data import Observations
from statadic.api.design import Intercept, RowDifference
from statadic.api.ordination import DistanceGram, Ordination, PrincipalCoordinates
from statadic.api.regression import Binomial, Observed, Regression
from statadic.sample.packed import PackedClouds

pytestmark = pytest.mark.native
EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "packages/statadic/docs/guides/extend-a-term.py"
)
extensions = runpy.run_path(str(EXAMPLE))


@pytest.mark.parametrize("family", ["ols", "binomial"])
def test_centered_term_retains_training_state_across_families(family):
    term = extensions["Centered"]("age")
    data = Observations(
        {
            "age": [10.0, 20.0, 30.0, 40.0, 50.0],
            "y": [0.0, 1.0, 0.0, 1.0, 1.0],
        }
    )
    terms = Intercept() + term
    model = (
        Regression("y", terms)
        if family == "ols"
        else Regression(Observed("y"), terms, law=Binomial(link="logit"))
    )
    fit = model.estimate(data)
    assert fit.valid
    assert fit.design.column_names == ("intercept", "centered(age)")
    batch = Observations({"age": [50.0, 20.0]}, row_ids=("b", "a"))
    reverse = Observations({"age": [20.0, 50.0]}, row_ids=("a", "b"))
    single = Observations({"age": [50.0]})
    np.testing.assert_array_equal(fit.design.transform(single), [[1.0, 20.0]])
    np.testing.assert_array_equal(
        fit.design.transform(batch), [[1.0, 20.0], [1.0, -10.0]]
    )
    np.testing.assert_allclose(fit.predict(batch)[::-1], fit.predict(reverse))
    np.testing.assert_allclose(fit.predict(single), fit.predict(reverse)[1:])


def test_term_supports_live_prediction_independently_of_other_operations(tmp_path):
    term = extensions["Centered"]("age")
    estimator = extensions["OverdeterminedLeastSquares"]()
    model = Regression("y", Intercept() + term, estimator=estimator)
    training = Observations({"age": [20.0, 30.0, 40.0], "y": [1.0, 2.0, 3.0]})
    fit = model.estimate(training)
    np.testing.assert_allclose(fit.predict({"age": [50.0]}), [4.0], rtol=1e-5)
    delta = RowDifference({"age": 50.0}, {"age": 30.0}).compile(fit.design)
    np.testing.assert_allclose(delta.coefficients, [0.0, 20.0])
    with pytest.raises(ValueError, match="matching additive fitted term"):
        fit.design.design_contrast(term, at=50.0, versus=30.0)
    with pytest.raises(ArtifactError, match="allowlisted prediction codec"):
        fit.save_predictor(tmp_path / "age")
    with pytest.raises(ValueError, match="more rows than design columns"):
        model.estimate({"age": [20.0, 30.0], "y": [1.0, 2.0]})


def test_external_geometry_preserves_axes_and_requires_an_embedding_warrant():
    geometry = extensions["MeanDistance"]()
    reference = PackedClouds.from_clouds(
        [
            jnp.array([[2.0], [4.0]]),
            jnp.array([[10.0]]),
        ]
    )
    prepared = geometry.prepare(reference, row_ids=("left", "right"))
    query = PackedClouds.from_clouds([jnp.array([[1.0]])])
    np.testing.assert_allclose(prepared.cross(query), [[2.0, 9.0]])
    pairwise = prepared.pairwise()
    assert pairwise.row_ids == ("left", "right")
    np.testing.assert_allclose(pairwise.values, [[0.0, 7.0], [7.0, 0.0]])
    with pytest.raises(ValueError, match="squared-negative-type"):
        Ordination(DistanceGram(), PrincipalCoordinates()).estimate(pairwise)
