"""Probability masses, unequal supports and selection retain their meanings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.sample.packed import PackedClouds, checked_packed_clouds

if TYPE_CHECKING:
    from jax.typing import DTypeLike


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_probabilities_counts_and_measure_selection(dtype: DTypeLike) -> None:
    """Unequal supports keep distinct probabilities and zero mass on padding."""
    clouds = [
        jnp.asarray([[1.0], [4.0]], dtype=dtype),
        jnp.asarray([[2.0]], dtype=dtype),
    ]
    measures = PackedClouds.from_clouds(
        clouds, [jnp.asarray([0.25, 0.75], dtype=dtype), jnp.ones(1, dtype=dtype)]
    )
    selected = measures.take([1, 0, 1])
    np.testing.assert_array_equal(measures.counts, [2, 1])
    np.testing.assert_array_equal(measures.masses, [[0.25, 0.75], [1.0, 0.0]])
    np.testing.assert_array_equal(selected.counts, [1, 2, 1])
    np.testing.assert_array_equal(selected.masses, [[1, 0], [0.25, 0.75], [1, 0]])

    def copy_measures(value: PackedClouds) -> PackedClouds:
        return value

    copied = jax.jit(copy_measures)(measures)
    np.testing.assert_array_equal(copied.masses, measures.masses)
    assert copied.data.dtype == dtype


@pytest.mark.component
@pytest.mark.contract
def test_mass_normalization_is_explicit_and_stable() -> None:
    """Unnormalized inputs fail; explicit normalization avoids overflow."""
    cloud = [jnp.asarray([[0.0], [2.0]])]
    with pytest.raises(ValueError, match="sum to one"):
        PackedClouds.from_clouds(cloud, [[2.0, 6.0]])
    # The direct sum overflows float64; scaling before summing must not.
    normalized = PackedClouds.from_clouds(cloud, [[6e307, 1.2e308]], normalize=True)
    np.testing.assert_allclose(normalized.masses, [[1 / 3, 2 / 3]], rtol=1e-14)
    uniform = PackedClouds.from_clouds(cloud)
    np.testing.assert_array_equal(uniform.masses, [[0.5, 0.5]])


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "mass", [[-1.0, 2.0], [0.0, 0.0], [np.nan, 1.0], [np.inf, 1.0], [1.0]]
)
def test_invalid_mass_vectors_fail(mass: list[float]) -> None:
    """Reject invalid probabilities before they reach numerical producers."""
    with pytest.raises(
        ValueError, match=r"masses|mass|supports|cloud|Measure|support|selection"
    ):
        PackedClouds.from_clouds([jnp.asarray([[0.0], [1.0]])], [mass])


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "clouds",
    [
        [],
        [jnp.empty((0, 1))],
        [jnp.asarray([[np.nan]])],
        [jnp.ones((2, 1)), jnp.ones((2, 2))],
    ],
)
def test_invalid_supports_fail(clouds: list[jax.Array]) -> None:
    """Finite nonempty compatible supports are an admission requirement."""
    with pytest.raises(
        ValueError, match=r"masses|mass|supports|cloud|Measure|support|selection"
    ):
        PackedClouds.from_clouds(clouds)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("indices", [[-1], [2], [], [0.5]])
def test_selection_does_not_use_gather_clipping(indices: list[float]) -> None:
    """Invalid observation indices raise instead of silently selecting a row."""
    measures = PackedClouds.from_clouds([jnp.ones((1, 1)), jnp.zeros((2, 1))])
    with pytest.raises(
        ValueError, match=r"masses|mass|supports|cloud|Measure|support|selection"
    ):
        measures.take(indices)


@pytest.mark.component
@pytest.mark.contract
def test_concrete_constructor_rejects_probability_on_padding() -> None:
    """Raw concrete construction cannot bypass the probability-domain guard."""
    supports = PackedClouds(
        jnp.asarray([[[1.0], [jnp.nan]]]), jnp.asarray([1], dtype=jnp.int32)
    )
    with pytest.raises(ValueError, match="padding"):
        checked_packed_clouds(supports.data, supports.counts, jnp.asarray([[0.5, 0.5]]))
    valid = checked_packed_clouds(
        supports.data, supports.counts, jnp.asarray([[1.0, 0.0]])
    )
    assert valid.n_observations == 1


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("traced", [False, True])
def test_complex_support_cannot_bypass_the_factory(*, traced: bool) -> None:
    """Real distance-law producers reject complex carriers before tracing values."""

    def construct(data: jax.Array) -> jax.Array:
        measures = PackedClouds(
            data, jnp.asarray([1], dtype=jnp.int32), jnp.asarray([[1.0]])
        )
        return measures.masses

    data = jnp.asarray([[[1.0 + 2.0j]]])
    evaluate = jax.jit(construct) if traced else construct
    with pytest.raises(TypeError, match="real"):
        evaluate(data)


@pytest.mark.component
@pytest.mark.contract
def test_zero_feature_width_cannot_bypass_the_factory() -> None:
    """The structural domain needs a nonempty coordinate space."""
    supports = PackedClouds(jnp.empty((1, 1, 0)), jnp.asarray([1], dtype=jnp.int32))
    with pytest.raises(ValueError, match="feature"):
        PackedClouds.from_clouds([supports.data[0]], [jnp.asarray([1.0])])


@pytest.mark.unit
@pytest.mark.contract
def test_full_support_construction_preserves_uniformity_in_caller_precision() -> None:
    """The integer support size rounds before division, as in the ragged carrier."""
    data = jnp.zeros((2049, 1), dtype=jnp.float16)
    cloud = PackedClouds.from_array(data)
    ragged = PackedClouds(data, jnp.asarray(data.shape[0], dtype=jnp.int32))
    np.testing.assert_array_equal(cloud.masses, ragged.masses)
    assert bool(cloud.is_uniform)
