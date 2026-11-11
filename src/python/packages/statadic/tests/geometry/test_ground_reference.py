"""Independent ground-geometry witnesses with explicit domains and axes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.geometry.ground.config import resolve_ground_distance
from statadic.geometry.ground.metrics import (
    EUCLIDEAN,
    NORMALIZED_EUCLIDEAN,
    cdist,
    pdist,
    projective_angular_distance,
)
from statadic.geometry.ground.similarities import (
    cosine_gram,
    distance_induced_gram,
    linear_gram,
    rbf_gram,
)
from scipy.spatial import distance

from .._budgets import examples
from .._strategies import float_arrays, nonzero_rows

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from jax import Array
    from statadic.geometry.ground.config import GroundDistanceName


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
@pytest.mark.parametrize(
    "metric", ["euclidean", "cosine", "angular", "normalized_euclidean"]
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_rectangular_distances_match_scipy_on_the_declared_domain(
    metric: GroundDistanceName, dtype: str
) -> None:
    """Compare radians, normalized chords and raw distances on nonzero rows."""
    left = np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0], [-2.0, 1.0, 2.0]], dtype=dtype)
    right = np.asarray([[1.0, 0.0, 0.0], [0.0, -3.0, 4.0]], dtype=dtype)
    if metric == "normalized_euclidean":
        expected = distance.cdist(
            left / np.linalg.norm(left, axis=1, keepdims=True),
            right / np.linalg.norm(right, axis=1, keepdims=True),
        )
    elif metric == "angular":
        expected = np.arccos(np.clip(1 - distance.cdist(left, right, "cosine"), -1, 1))
    else:
        expected = distance.cdist(left, right, metric)
    # Noncoincident, nonantipodal angles avoid the singular arccos endpoints.
    tolerance = 8e-7 if dtype == "float32" else 5e-13

    actual = cdist(
        jnp.asarray(left), jnp.asarray(right), metric=resolve_ground_distance(metric)
    )

    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
@examples(20)
@example(cloud=jnp.asarray([[0.0, 0.0], [3.0, 4.0], [-2.0, 1.0], [3.0, 4.0]]))
@given(cloud=float_arrays((4, 2)))
def test_square_pdist_matches_condensed_scipy_after_explicit_conversion(
    cloud: Array,
) -> None:
    """STATADIC's square result is aligned explicitly with SciPy's condensed vector."""
    points = np.asarray(cloud)
    expected = distance.squareform(distance.pdist(points, "euclidean"))

    actual = pdist(jnp.asarray(points), metric=EUCLIDEAN)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(np.diag(actual), np.zeros(len(points)))


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("metric", ["angular", "cosine", "normalized_euclidean"])
def test_zero_direction_mask_does_not_poison_valid_cross_pairs(
    metric: GroundDistanceName,
) -> None:
    """An undefined direction affects its own row/column rather than other pairs."""
    left = jnp.asarray([[0.0, 0.0], [1.0, 0.0]])
    right = jnp.asarray([[0.0, 1.0], [0.0, 0.0]])

    actual = cdist(left, right, metric=resolve_ground_distance(metric))

    np.testing.assert_array_equal(np.isnan(actual), [[True, True], [False, True]])
    assert np.isfinite(actual[1, 0])


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scikit-learn")
@examples(20)
@example(
    family="linear",
    left_rows=jnp.asarray([[3.0, 4.0], [-2.0, 1.0], [1.0, -3.0]]),
    right_rows=jnp.asarray([[1.0, 0.0], [2.0, 3.0], [-2.0, -1.0]]),
)
@example(
    family="cosine",
    left_rows=jnp.asarray([[3.0, 4.0], [-2.0, 1.0], [1.0, -3.0]]),
    right_rows=jnp.asarray([[1.0, 0.0], [2.0, 3.0], [-2.0, -1.0]]),
)
@given(
    family=st.sampled_from(["linear", "cosine"]),
    left_rows=nonzero_rows(3, 2),
    right_rows=nonzero_rows(3, 2),
)
def test_gram_blocks_match_sklearn_without_forcing_cross_diagonals(
    optional_reference: Callable[[str], ModuleType],
    family: str,
    left_rows: Array,
    right_rows: Array,
) -> None:
    """Cross-block diagonals need not be self-similarities, even for square blocks."""
    peer = optional_reference("sklearn.metrics.pairwise")
    left = np.asarray(left_rows)
    right = np.asarray(right_rows)
    expected = (
        peer.linear_kernel(left, right)
        if family == "linear"
        else peer.cosine_similarity(left, right)
    )

    actual = (
        linear_gram(jnp.asarray(left), jnp.asarray(right))
        if family == "linear"
        else cosine_gram(jnp.asarray(left), jnp.asarray(right))
    )

    assert np.isfinite(actual).all()
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scikit-learn")
@examples(20)
@example(
    bandwidth=0.5,
    left_rows=jnp.asarray([[0.0, 0.0], [1.0, -2.0], [1.0, -2.0]]),
    right_rows=jnp.asarray([[1.0, -2.0], [0.5, 0.25]]),
)
@example(
    bandwidth=2.0,
    left_rows=jnp.asarray([[0.0, 0.0], [1.0, -2.0], [1.0, -2.0]]),
    right_rows=jnp.asarray([[1.0, -2.0], [0.5, 0.25]]),
)
@given(
    bandwidth=st.sampled_from([0.5, 2.0]),
    left_rows=float_arrays((3, 2)),
    right_rows=float_arrays((2, 2)),
)
def test_rbf_bandwidth_matches_sklearn_gamma(
    optional_reference: Callable[[str], ModuleType],
    bandwidth: float,
    left_rows: Array,
    right_rows: Array,
) -> None:
    """The Gaussian convention is gamma=1/(2*sigma**2), including coincident rows."""
    peer = optional_reference("sklearn.metrics.pairwise")
    left = np.asarray(left_rows)
    right = np.asarray(right_rows)
    expected = peer.rbf_kernel(left, right, gamma=1 / (2 * bandwidth**2))

    actual = rbf_gram(jnp.asarray(left), jnp.asarray(right), bandwidth=bandwidth)

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.unit
@pytest.mark.reference
@examples(20)
@example(
    exponent=0.5,
    left_rows=jnp.asarray([[0.0, 0.0], [3.0, 4.0], [-1.0, 2.0]]),
    right_rows=jnp.asarray([[1.0, 1.0], [4.0, -2.0]]),
    reference_row=jnp.asarray([2.0, -3.0]),
)
@example(
    exponent=1.0,
    left_rows=jnp.asarray([[0.0, 0.0], [3.0, 4.0], [-1.0, 2.0]]),
    right_rows=jnp.asarray([[1.0, 1.0], [4.0, -2.0]]),
    reference_row=jnp.asarray([2.0, -3.0]),
)
@example(
    exponent=1.5,
    left_rows=jnp.asarray([[0.0, 0.0], [3.0, 4.0], [-1.0, 2.0]]),
    right_rows=jnp.asarray([[1.0, 1.0], [4.0, -2.0]]),
    reference_row=jnp.asarray([2.0, -3.0]),
)
@given(
    exponent=st.sampled_from([0.5, 1.0, 1.5]),
    left_rows=float_arrays((3, 2)),
    right_rows=float_arrays((2, 2)),
    reference_row=float_arrays((2,)),
)
def test_distance_induced_cross_gram_uses_one_supplied_reference(
    exponent: float, left_rows: Array, right_rows: Array, reference_row: Array
) -> None:
    """Independent raw norms protect the unhalved kernel and powered reference."""
    left = np.asarray(left_rows)
    right = np.asarray(right_rows)
    reference = np.asarray(reference_row)
    expected = (
        np.linalg.norm(left - reference, axis=1)[:, None] ** exponent
        + np.linalg.norm(right - reference, axis=1)[None, :] ** exponent
        - np.linalg.norm(left[:, None] - right[None, :], axis=-1) ** exponent
    )

    actual = distance_induced_gram(
        jnp.asarray(left),
        jnp.asarray(right),
        metric="euclidean",
        exponent=exponent,
        reference=jnp.asarray(reference),
    )

    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.unit
@pytest.mark.reference
@examples(15)
@given(hnp.arrays(np.float64, (4, 2), elements=st.integers(-4, 4).map(float)))
def test_generated_euclidean_isometries_and_direction_quotients(
    raw: np.ndarray,
) -> None:
    """Finite fixed-shape clouds exercise each lawful transform independently."""
    # A positive first coordinate excludes zero rows without filtering a result.
    points = np.column_stack([np.ones(len(raw)), raw])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    moved = points @ rotation + np.asarray([2.0, -3.0, 1.0])
    expected_euclidean = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    scaled = points * np.asarray([0.5, 2.0, 3.0, 4.0])[:, None]

    original = pdist(jnp.asarray(points), metric=EUCLIDEAN)
    transformed = pdist(jnp.asarray(moved), metric=EUCLIDEAN)
    chord = cdist(jnp.asarray(points), jnp.asarray(scaled), metric=NORMALIZED_EUCLIDEAN)
    projective = projective_angular_distance(jnp.asarray(points), jnp.asarray(-points))

    np.testing.assert_allclose(original, expected_euclidean, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(transformed, expected_euclidean, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.diag(chord), 0.0, atol=5e-8)
    np.testing.assert_allclose(np.diag(projective), 0.0, atol=5e-8)
