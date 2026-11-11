# jaxtyping shape strings are runtime contracts.
"""Principal coordinates — the precondition, and the agreement of two bodies.

Two things are pinned here that no other suite can see:

* the **precondition** :mod:`statadic.geometry.embedding` was created to make
  explicit — a materially negative Gram spectrum raises rather than being
  clipped away;
* the **agreement of the two PCoA callables**, which exist separately only
  because ``DMat.values`` is a JAX array and Paper 1's artifact path is
  NumPy/float64 (see the module docstring). Nothing else would notice the two
  bodies drifting apart.

``pcoa``'s ``@requires`` declaration and the wiring that may legally feed it are
in ``tests/core/test_axioms.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.api.ordination import DistanceGram, Ordination, PrincipalCoordinates
from statadic.api.representation import ObservedDissimilarity
from statadic.geometry.embedding import NonEuclideanEmbeddingError, pcoa_kernel

from .._budgets import examples
from .._clouds import euclidean

_POSITIVE_SPECTRUM_FLOOR = 1e-10
_COORDINATE_BOUND = 10.0
_MINIMUM_SPREAD = 0.1


def _has_spread(points: NDArray[np.float64]) -> bool:
    """Require a nonconstant cloud, since PCoA needs one positive eigenvalue."""
    return bool(np.ptp(points, axis=0).max() >= _MINIMUM_SPREAD)


def _point_clouds(rows: int, columns: int) -> SearchStrategy[NDArray[np.float64]]:
    """Draw finite point clouds of one fixed shape."""
    return hnp.arrays(
        np.float64,
        (rows, columns),
        elements=st.floats(
            -_COORDINATE_BOUND,
            _COORDINATE_BOUND,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from hypothesis.strategies import SearchStrategy
    from numpy.typing import NDArray


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scikit-bio")
@pytest.mark.parametrize("shape", ["square", "tetrahedron"])
def test_pcoa_tied_eigenspaces_match_skbio_and_independent_gram(
    optional_reference: Callable[[str], ModuleType], shape: str
) -> None:
    """Compare a complete repeated-eigenvalue block through its projector."""
    peer = optional_reference("skbio")
    points = (
        np.asarray([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
        if shape == "square"
        else np.asarray(
            [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
        )
    )
    points = points + 7.0
    distances = euclidean(points)
    squared = distances**2
    expected_gram = -0.5 * (
        squared
        - squared.mean(axis=0)[None, :]
        - squared.mean(axis=1)[:, None]
        + squared.mean()
    )
    spectrum, vectors = np.linalg.eigh(expected_gram)
    positive = spectrum > _POSITIVE_SPECTRUM_FLOOR
    expected_projector = vectors[:, positive] @ vectors[:, positive].T
    identifiers = ["d", "a", "c", "b"]
    reference = peer.stats.ordination.pcoa(
        peer.DistanceMatrix(distances, ids=identifiers), method="eigh"
    )

    result = pcoa_kernel(jnp.asarray(distances))

    assert bool(result.valid)
    assert np.isfinite(result.eigenvalues).all()
    np.testing.assert_allclose(
        result.eigenvalues, spectrum[::-1], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(result.active, positive[::-1])
    np.testing.assert_allclose(
        result.coordinates @ result.coordinates.T,
        expected_gram,
        rtol=1e-12,
        atol=1e-12,
    )
    normalized = np.asarray(result.coordinates)[:, positive[::-1]] / np.sqrt(
        np.asarray(result.eigenvalues)[positive[::-1]]
    )
    np.testing.assert_allclose(
        normalized @ normalized.T, expected_projector, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        result.coordinates @ result.coordinates.T,
        np.asarray(reference.samples) @ np.asarray(reference.samples).T,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        reference.eigvals, spectrum[::-1], rtol=1e-12, atol=1e-12
    )
    assert reference.samples.index.tolist() == identifiers


@pytest.mark.unit
@pytest.mark.reference
def test_non_euclidean_spectrum_matches_independent_double_centering() -> None:
    """A negative eigenvalue is retained as evidence rather than clipped away."""
    distances = np.abs(np.arange(4.0)[:, None] - np.arange(4.0)[None, :]) ** 2
    squared = distances**2
    gram = -0.5 * (
        squared
        - squared.mean(axis=0)[None, :]
        - squared.mean(axis=1)[:, None]
        + squared.mean()
    )
    expected = np.linalg.eigvalsh(gram)[::-1]

    result = pcoa_kernel(jnp.asarray(distances))

    assert not bool(result.valid)
    np.testing.assert_allclose(result.eigenvalues, expected, rtol=1e-12, atol=1e-12)
    assert float(result.minimum_eigenvalue) == pytest.approx(expected[-1], rel=1e-12)
    assert float(result.spectral_scale) == pytest.approx(
        np.max(np.abs(expected)), rel=1e-12
    )


@pytest.mark.component
@pytest.mark.reference
@examples(20)
@example(points=np.random.default_rng(1).normal(size=(12, 4)))
@example(
    points=np.vstack(
        (
            [0.0, 1e-6, 1e-6, 1e-6],
            [1e-6, 1.0, 1e-6, 1e-6],
            np.full((10, 4), 1e-6),
        )
    )
)
@given(points=_point_clouds(12, 4).filter(_has_spread))
def test_pcoa_matrix_reconstructs_a_euclidean_cloud(
    points: NDArray[np.float64],
) -> None:
    """Recover distances within the independently bounded discarded spectrum."""
    dist = euclidean(points)
    fit = Ordination(DistanceGram(), PrincipalCoordinates(None)).estimate(
        ObservedDissimilarity.from_array(dist)
    )
    coords, eigvals = fit.coordinates, fit.eigenvalues
    assert eigvals.min() >= -1e-8 * max(abs(eigvals.max()), 1.0)
    # Form the reference Gram directly from points, independently of STATADIC's
    # distance double-centering. The default cut intentionally drops tiny axes.
    centered = points - points.mean(axis=0)
    spectrum = np.linalg.eigvalsh(centered @ centered.T)
    scale = max(float(np.max(np.abs(spectrum))), 1.0)
    roundoff = 8 * len(points) * np.finfo(points.dtype).eps * scale
    np.testing.assert_allclose(eigvals, spectrum[::-1], rtol=0.0, atol=roundoff)
    cutoff = 1e-8 * scale
    discarded = spectrum[spectrum <= cutoff + roundoff]
    discarded_norm = float(np.max(np.abs(discarded), initial=0.0))
    # Each squared-distance error is (e_i-e_j)' E (e_i-e_j), at most
    # 2 ||E||_2, plus dense-centering/eigensolver backward error.
    np.testing.assert_allclose(
        euclidean(np.asarray(coords)) ** 2,
        dist**2,
        rtol=0.0,
        atol=2 * (discarded_norm + roundoff),
    )


@pytest.mark.component
@pytest.mark.contract
@examples(20)
@example(
    clouds=(
        np.random.default_rng(9).normal(size=(8, 3)),
        np.random.default_rng(10).normal(size=(8, 4)),
    )
)
@given(clouds=st.tuples(_point_clouds(8, 3), _point_clouds(8, 4)))
def test_pcoa_kernel_has_static_jit_and_vmap_structure(
    clouds: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """Keep every axis and report active coordinates with a mask under transforms."""
    distances = np.stack([euclidean(clouds[0]), euclidean(clouds[1])])
    direct = pcoa_kernel(jnp.asarray(distances[0]))
    compiled = jax.jit(pcoa_kernel)(jnp.asarray(distances[0]))
    batched = jax.vmap(pcoa_kernel)(jnp.asarray(distances))

    assert direct.coordinates.shape == (8, 8)
    assert direct.active.shape == (8,)
    assert compiled.coordinates.shape == (8, 8)
    assert batched.coordinates.shape == (2, 8, 8)
    assert batched.active.shape == (2, 8)
    assert bool(jnp.all(batched.valid))
    inactive_values = jnp.where(
        batched.active[..., None, :],
        0.0,
        batched.coordinates,
    )
    assert bool(jnp.allclose(inactive_values, 0.0, rtol=0.0, atol=0.0))


@pytest.mark.unit
@pytest.mark.contract
def test_pcoa_kernel_reports_non_euclidean_input_without_traced_raise() -> None:
    """Return a fixed-shape false diagnostic for an invalid branded computation."""
    x = jnp.arange(4.0)
    squared = jnp.abs(x[:, None] - x[None, :]) ** 2

    result = jax.jit(pcoa_kernel)(squared)

    assert not bool(result.valid)
    assert result.coordinates.shape == (4, 4)


@pytest.mark.component
@pytest.mark.regression
def test_pcoa_raises_on_a_non_negative_type_matrix() -> None:
    """Raise instead of clipping when the Gram spectrum is materially negative.

    Squared Euclidean distances on four collinear points: squaring destroys the
    negative type that classical scaling needs, and the leading negative
    eigenvalue is O(30%) of the spectral scale — not rounding. The pre-t46 code
    would have dropped those axes silently.
    """
    x = np.arange(4.0)
    squared = np.abs(x[:, None] - x[None, :]) ** 2
    with pytest.raises(NonEuclideanEmbeddingError, match="negative eigenvalue"):
        Ordination(DistanceGram(), PrincipalCoordinates(None)).estimate(
            ObservedDissimilarity.from_array(squared)
        )
