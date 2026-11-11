"""Graph-distance identity, duplicate-point and transformed numerical controls."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.geometry.shortest_paths import geodesic_cross_distances, knn_shortest_paths

pytestmark = pytest.mark.unit


@pytest.mark.reference
def test_zero_length_edges_and_chain_distances_survive_jit() -> None:
    """Distinct coincident vertices remain connected by zero-length edges."""
    positions = jnp.asarray([0.0, 0.0, 1.0, 3.0])
    distances = jnp.abs(positions[:, None] - positions[None])
    codes = jnp.arange(4)
    solve = partial(knn_shortest_paths, neighbors=2, tie_codes=codes)
    actual = jax.jit(solve)(distances)
    np.testing.assert_allclose(actual, distances, atol=1e-12)
    np.testing.assert_allclose(
        jax.vmap(solve)(jnp.stack([distances, distances * 2])),
        jnp.stack([distances, distances * 2]),
        atol=1e-12,
    )


@pytest.mark.reference
def test_query_paths_use_training_only_intermediates() -> None:
    """A second query cannot shorten the first query's route through training."""
    training_paths = jnp.asarray([[0.0, 2.0, 4.0], [2.0, 0.0, 2.0], [4.0, 2.0, 0.0]])
    cross = jnp.asarray([[1.0, 3.0, 10.0], [10.0, 3.0, 1.0]])
    expected = np.asarray([[1.0, 3.0, 5.0], [5.0, 3.0, 1.0]])
    operation = jax.jit(
        partial(
            geodesic_cross_distances,
            training_paths=training_paths,
            neighbors=1,
            tie_codes=jnp.arange(3),
        )
    )
    np.testing.assert_allclose(operation(cross), expected, atol=1e-12)
    np.testing.assert_allclose(operation(cross[:1]), expected[:1], atol=1e-12)


@pytest.mark.contract
def test_disconnected_paths_remain_infinite() -> None:
    """Native kernels retain unavailable pairs for the host's rejection policy."""
    positions = jnp.asarray([0.0, 1.0, 100.0, 101.0])
    distances = jnp.abs(positions[:, None] - positions[None])
    actual = knn_shortest_paths(distances, neighbors=1, tie_codes=jnp.arange(4))
    assert bool(jnp.isinf(actual[0, 2]))
