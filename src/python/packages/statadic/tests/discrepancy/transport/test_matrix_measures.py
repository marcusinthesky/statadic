"""A weighted quantile Hilbert witness qualifies the pairwise W2 producer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.discrepancy.transport.matrix import (
    finite_sliced_w2_matrix,
    quantile_w2_matrix,
)
from statadic.discrepancy.transport.sliced import ProjectionDesign
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    from jax import Array
    from jax.typing import DTypeLike


@pytest.mark.component
@pytest.mark.numerical
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_weighted_w2_matches_common_quantile_embedding(dtype: DTypeLike) -> None:
    """Four unequal supports share quarter-interval quantile coordinates exactly."""
    measures = PackedClouds.from_clouds(
        [
            jnp.asarray(x, dtype=dtype)[:, None]
            for x in ([0.0], [0.0, 2.0], [1.0, 2.0, 4.0], [2.0, 3.0])
        ],
        [
            jnp.asarray(w, dtype=dtype)
            for w in ([1.0], [0.25, 0.75], [0.5, 0.25, 0.25], [0.5, 0.5])
        ],
    )
    embedding = 0.5 * np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 4.0],
            [2.0, 2.0, 3.0, 3.0],
        ]
    )
    expected = np.linalg.norm(embedding[:, None] - embedding[None, :], axis=-1)

    def evaluate_get_values(m: PackedClouds) -> Array:
        return quantile_w2_matrix(m).values

    actual = jax.jit(evaluate_get_values)(measures)
    tolerance = 2e-6 if dtype == jnp.float32 else 2e-13
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    design = ProjectionDesign(jax.random.PRNGKey(4), n_projections=3, p=2)
    np.testing.assert_allclose(
        finite_sliced_w2_matrix(measures, design).values,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.unit
@pytest.mark.contract
def test_other_sliced_orders_do_not_get_hilbertian_brand() -> None:
    """A valid transport order does not automatically imply squared CND."""
    measures = PackedClouds.from_clouds(
        [jnp.asarray([[0.0], [1.0]]), jnp.asarray([[2.0]])]
    )
    with pytest.raises(ValueError, match="order-two"):
        finite_sliced_w2_matrix(measures, ProjectionDesign(jax.random.PRNGKey(1), p=3))
