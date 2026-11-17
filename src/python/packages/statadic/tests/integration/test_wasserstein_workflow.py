"""Empirical measures compose through frozen Wasserstein coordinates and regression."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.geometry.ground.metrics import EUCLIDEAN
from statadic.geometry.ground.similarities import LINEAR_KERNEL
from statadic.measure.clouds import PackedClouds
from statadic.observation.data import MeasureColumn, Observations
from statadic.observation.design import Intercept
from statadic.observation.geometry import MMD, Wasserstein
from statadic.ordination.fit import Ordination
from statadic.prediction.artifacts import load_ordination_predictor, load_predictor
from statadic.regression import LinearResult, Regression

if TYPE_CHECKING:
    from pathlib import Path


def _translated_clouds(centers: list[list[float]], *, weighted: bool) -> PackedClouds:
    """Represent translated measures with uniform or unequal weighted supports."""
    clouds = []
    masses = []
    for index, center in enumerate(centers):
        if weighted and index % 2:
            offsets = [[-0.5, 0.0], [0.5, 0.0], [0.5, 0.0]]
            probabilities = [0.25, 0.25, 0.5]
        else:
            offsets = [[-0.5, 0.0], [0.5, 0.0]]
            probabilities = [0.25, 0.75] if weighted else [0.5, 0.5]
        clouds.append(jnp.asarray(offsets) + jnp.asarray(center))
        masses.append(jnp.asarray(probabilities))
    return PackedClouds.from_clouds(clouds, masses=masses)


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.numerical
def test_centroids_and_wasserstein_share_inputs_but_differ_in_information() -> None:
    """A mean-preserving spread vanishes under linear MMD and has W2 equal to one."""
    clouds = PackedClouds.from_clouds(
        [[[-1.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
    )
    centroids = MMD(LINEAR_KERNEL).cross(clouds, clouds)
    distributions = Wasserstein(p=2, ground=EUCLIDEAN)
    distances = distributions.cross(clouds, clouds)
    np.testing.assert_allclose(centroids, 0.0, atol=1e-6)
    np.testing.assert_allclose(distances, [[0.0, 1.0], [1.0, 0.0]], atol=1e-6)
    assert distributions.matrix(clouds) is None


@pytest.mark.e2e
@pytest.mark.contract
@pytest.mark.parametrize("weighted", [False, True], ids=["uniform", "weighted-unequal"])
def test_wasserstein_rotation_and_nested_regression_keep_frozen_references(
    tmp_path: Path, *, weighted: bool
) -> None:
    """Held-out placement survives alignment and standalone/nested persistence."""
    ids = ("a", "b", "c", "d", "e", "f")
    clouds = _translated_clouds(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0], [3.0, -1.0], [-1.0, -1.0]],
        weighted=weighted,
    )
    training = Observations(
        {"y": [1.0, 2.0, 1.5, 3.2, 4.1, 0.7]},
        row_ids=ids,
        measures={"articles": MeasureColumn(ids, clouds)},
    )
    query_ids = ("new-a", "new-b")
    query = Observations(
        {},
        row_ids=query_ids,
        measures={
            "articles": MeasureColumn(
                query_ids,
                _translated_clouds([[0.5, 1.0], [2.5, -0.5]], weighted=weighted),
            )
        },
    )
    geometry = Wasserstein(p=2, ground=EUCLIDEAN)
    source = Ordination.pcoa(
        geometry,
        column="articles",
        dim=2,
        negative_inertia="positive-subspace",
        name="news",
    ).estimate(training)
    centroid = Ordination.pcoa(
        MMD(LINEAR_KERNEL), column="articles", dim=2, name="centroid"
    ).estimate(training)
    np.testing.assert_allclose(
        centroid.transform(training), centroid.coordinates, rtol=2e-5, atol=2e-5
    )
    rotated = source.rotate(jnp.asarray([[0.0, -1.0], [1.0, 0.0]]))
    aligned = source.align_to(rotated.coordinates[::-1], row_ids=ids[::-1])
    expected = rotated.transform(query)
    np.testing.assert_allclose(aligned.transform(query), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(aligned.eigenvalues, source.eigenvalues)
    cross = geometry.cross(query.measure("articles"), clouds)
    np.testing.assert_allclose(
        aligned.transform_distances(cross, reference_ids=ids),
        expected,
        rtol=2e-5,
        atol=2e-5,
    )
    with pytest.raises(ValueError, match="reference IDs"):
        aligned.transform_distances(cross, reference_ids=ids[::-1])
    aligned.save_predictor(tmp_path / "coordinates")
    restored = load_ordination_predictor(tmp_path / "coordinates")
    np.testing.assert_allclose(
        restored.transform(query), expected, rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        restored.transform(query.take([1])), expected[1:], rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        restored.transform_distances(cross, reference_ids=ids),
        expected,
        rtol=2e-5,
        atol=2e-5,
    )
    # The regression sample is smaller than the representation-learning sample.
    # as_term() must retain that map instead of estimating new principal axes.
    regression_data = training.take([0, 2, 3, 4])
    regression = Regression("y", Intercept() + aligned.as_term()).estimate(
        regression_data
    )
    assert isinstance(regression, LinearResult)
    np.testing.assert_allclose(
        regression.design.transform(regression_data)[:, 1:],
        aligned.transform(regression_data),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_array_equal(
        regression.design.ordination("news").spectrum.eigenvalues, source.eigenvalues
    )
    regression.save_predictor(tmp_path / "regression")
    predictor = load_predictor(tmp_path / "regression")
    np.testing.assert_allclose(
        predictor.predict(query), regression.predict(query), rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        predictor.design.transform(query)[:, 1:], expected, rtol=2e-5, atol=2e-5
    )


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.numerical
def test_wasserstein_signed_spectrum_requires_explicit_policy() -> None:
    """Centered two-atom clouds provide a non-Hilbertian W2 witness."""
    clouds = PackedClouds.from_clouds(
        [
            [[-1.0, 0.0], [1.0, 0.0]],
            [[0.0, -1.0], [0.0, 1.0]],
            [[-1.0, -1.0], [1.0, 1.0]],
            [[-1.0, 1.0], [1.0, -1.0]],
        ]
    )
    geometry = Wasserstein(p=2, ground=EUCLIDEAN)
    with pytest.raises(ValueError, match="qualified squared-negative-type"):
        Ordination.pcoa(geometry, dim=2).estimate(clouds)
    fit = Ordination.pcoa(
        geometry, dim=2, negative_inertia="positive-subspace"
    ).estimate(clouds)
    assert fit.spectrum.material_negative_inertia
    np.testing.assert_allclose(fit.eigenvalues, [2.0, 1.0, 0.0, -0.5], atol=2e-5)
    np.testing.assert_allclose(fit.transform(clouds), fit.coordinates, atol=2e-5)
