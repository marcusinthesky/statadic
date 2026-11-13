"""Independent weighted expectation witnesses for rooted V-MMD and energy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.api.representation import Energy
from statadic.core.domains import FiniteValues
from statadic.discrepancy.mmd import mmd_matrix
from statadic.discrepancy.mmd.energy import energy_distance_matrix
from statadic.discrepancy.mmd.energy.weighted import (
    weighted_energy_cross,
    weighted_energy_matrix,
    weighted_energy_squared_cross,
)
from statadic.discrepancy.mmd.weighted import (
    _linear_mmd_cross,
    weighted_mmd_cross,
    weighted_mmd_matrix,
    weighted_mmd_squared_cross,
)
from statadic.geometry.ground.similarities import (
    LINEAR_KERNEL,
    PositiveSemidefiniteKernelProperties,
    RBFKernel,
    declare_similarity,
)
from statadic.sample.packed import PackedClouds, pack

from ..._budgets import examples
from ..._strategies import float_arrays


@pytest.mark.unit
@pytest.mark.numerical
def test_squared_linear_cross_has_correct_curvature_at_zero() -> None:
    """The squared learning objective retains its analytic Hessian at equality."""
    target = PackedClouds.from_clouds([jnp.asarray([[1.0], [3.0]])])

    def objective(shift: jax.Array) -> jax.Array:
        left = target.with_coordinates(target.data + shift)
        return weighted_mmd_squared_cross(left, target)[0, 0]

    origin = jnp.asarray(0.0)
    assert float(jax.grad(objective)(origin)) == pytest.approx(0)
    assert float(jax.hessian(objective)(origin)) == pytest.approx(2)


if TYPE_CHECKING:
    from jax import Array
    from jax.typing import DTypeLike
    from statadic.geometry.ground.similarities import (
        PositiveSemidefiniteKernelLaw,
        SimilarityStrategy,
    )


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("integer_input", [False, True])
def test_linear_mean_route_against_gram_witness(
    dtype: DTypeLike, *, integer_input: bool
) -> None:
    """Witness equal-mass Gram/mean agreement; this is not a floating-point proof."""
    coords = np.asarray([[1000000, 1000002], [1000004, 999996], [9000, -9000]])
    other = np.asarray([[999996, 1000000], [1000008, 1000004], [-9000, 9000]])
    if not integer_input:
        coords = coords.astype(np.float64)
        other = other.astype(np.float64)
    weights = np.asarray([0.25, 0.75, 0.0])
    other_weights = np.asarray([0.5, 0.5, 0.0])
    reference = np.asarray([1000000, 1000000])
    expected = (
        weights @ (coords @ coords.T) @ weights
        + other_weights @ (other @ other.T) @ other_weights
        - 2 * weights @ (coords @ other.T) @ other_weights
    )
    with jax.enable_x64(dtype == jnp.float64):
        actual = _linear_mmd_cross(
            jnp.asarray(coords)[None, ...],
            jnp.asarray(weights, dtype=dtype)[None, ...],
            jnp.asarray(other)[None, ...],
            jnp.asarray(other_weights, dtype=dtype)[None, ...],
            jnp.asarray(reference),
        )
        assert float(actual[0, 0]) == pytest.approx(expected, rel=1e-5, abs=1e-5)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("family", ["rbf", "linear", "energy"])
def test_weighted_expectations_match_independent_sums(
    dtype: DTypeLike, family: str
) -> None:
    """Explicit ordered atom sums protect mass normalization and energy's factor."""
    x = np.asarray([[0.0, 1.0], [2.0, -1.0]])
    y = np.asarray([[1.0, 2.0], [-1.0, 0.0], [3.0, 1.0]])
    a, b = np.asarray([0.25, 0.75]), np.asarray([0.5, 0.25, 0.25])
    measures = PackedClouds.from_clouds(
        [jnp.asarray(x, dtype=dtype), jnp.asarray(y, dtype=dtype)],
        [jnp.asarray(a, dtype=dtype), jnp.asarray(b, dtype=dtype)],
    )

    def expectation(
        left: np.ndarray, wa: np.ndarray, right: np.ndarray, wb: np.ndarray
    ) -> float:
        total = 0.0
        for i in range(len(left)):
            for k in range(len(right)):
                if family == "energy":
                    value = np.linalg.norm(left[i] - right[k]) ** 1.5
                elif family == "rbf":
                    value = np.exp(-np.sum((left[i] - right[k]) ** 2) / (2 * 1.7**2))
                else:
                    value = left[i] @ right[k]
                total += wa[i] * wb[k] * value
        return float(total)

    cross = expectation(x, a, y, b)
    within = expectation(x, a, x, a) + expectation(y, b, y, b)
    squared = 2 * cross - within if family == "energy" else within - 2 * cross
    if family == "energy":

        def evaluate_get_values(m: PackedClouds) -> Array:
            return weighted_energy_matrix(m, exponent=1.5).values

        actual = jax.jit(evaluate_get_values)(measures)
    else:
        kernel = (
            RBFKernel(jnp.asarray(1.7, dtype=dtype))
            if family == "rbf"
            else LINEAR_KERNEL
        )

        def evaluate_kernel_values(m: PackedClouds) -> Array:
            return weighted_mmd_matrix(m, kernel=kernel).values

        actual = jax.jit(evaluate_kernel_values)(measures)
    tolerance = 4e-6 if dtype == jnp.float32 else 2e-13
    np.testing.assert_allclose(
        actual,
        [[0.0, np.sqrt(squared)], [np.sqrt(squared), 0.0]],
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.component
@pytest.mark.property
def test_uniform_weighted_mmd_matches_existing_v_convention() -> None:
    """Uniform masses recover the existing unequal-support V-statistic."""
    clouds = [jnp.asarray([[0.0], [2.0]]), jnp.asarray([[1.0], [3.0], [6.0]])]
    measures = PackedClouds.from_clouds(clouds)
    kernel = RBFKernel(jnp.asarray(1.2))
    np.testing.assert_allclose(
        weighted_mmd_matrix(measures, kernel=kernel).values,
        mmd_matrix(pack(clouds, require_positive_counts=True), kernel=kernel).values,
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.unit
@pytest.mark.property
@examples(20)
@example(family="rbf", first=jnp.asarray([[1.0], [3.0]]), second=jnp.asarray([[2.0]]))
@example(
    family="energy", first=jnp.asarray([[1.0], [3.0]]), second=jnp.asarray([[2.0]])
)
@given(
    family=st.sampled_from(["rbf", "energy"]),
    first=float_arrays((2, 1)),
    second=float_arrays((1, 1)),
)
def test_inactive_padding_cannot_change_values(
    family: str, first: Array, second: Array
) -> None:
    """An inactive NaN must not contaminate weighted expectations."""
    clean = PackedClouds.from_clouds([first, second])
    poisoned = PackedClouds(
        clean.data.at[1, 1, 0].set(jnp.nan),
        clean.counts,
        clean.masses,
    )
    if family == "energy":
        expected = weighted_energy_cross(clean, clean)
        actual = jax.jit(weighted_energy_cross)(poisoned, poisoned)
    else:
        kernel = RBFKernel(jnp.asarray(1.0))
        expected = weighted_mmd_cross(clean, clean, kernel=kernel)

        def evaluate_weighted_mmd_cross(m: PackedClouds) -> Array:
            return weighted_mmd_cross(m, m, kernel=kernel)

        actual = jax.jit(evaluate_weighted_mmd_cross)(poisoned)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("amplitude", [1e-20, 1.0, 1e20])
def test_material_negative_mmd_is_unavailable_instead_of_clipped(
    amplitude: float,
) -> None:
    """An adversarial false PSD declaration must not manufacture zero distance."""

    def negative_dot(
        x: jax.Array, y: jax.Array, *, reference: jax.Array | None = None
    ) -> jax.Array:
        del reference
        return -amplitude * (x @ y.T)

    kernel = declare_similarity(
        negative_dot,
        law=PositiveSemidefiniteKernelProperties(),
        required_evidence=FiniteValues,
    )
    measures = PackedClouds.from_clouds([jnp.asarray([[0.0]]), jnp.asarray([[2.0]])])

    def evaluate_weighted_mmd_cross(m: PackedClouds) -> Array:
        return weighted_mmd_cross(m, m, kernel=kernel)

    actual = jax.jit(evaluate_weighted_mmd_cross)(measures)
    assert np.isnan(actual[0, 1])
    assert np.isnan(actual[1, 0])


@pytest.mark.unit
@pytest.mark.numerical
def test_zero_mass_origin_is_not_used_as_angular_kernel_anchor() -> None:
    """A zero-mass atom at the origin cannot invalidate positive-mass geometry."""
    measures = PackedClouds.from_clouds(
        [jnp.asarray([[0.0, 0.0], [1.0, 0.0]]), jnp.asarray([[0.0, 1.0]])],
        [[0.0, 1.0], [1.0]],
    )

    def evaluate_get_values(m: PackedClouds) -> Array:
        return weighted_energy_matrix(m, ground="angular").values

    actual = jax.jit(evaluate_get_values)(measures)
    np.testing.assert_allclose(
        actual, [[0.0, np.sqrt(np.pi)], [np.sqrt(np.pi), 0.0]], rtol=1e-13, atol=1e-13
    )


@pytest.mark.component
@pytest.mark.contract
def test_weighted_numerics_support_outer_vmap_and_jit() -> None:
    """Fixed-capacity numerical fits batch without host probability checks."""
    measures = PackedClouds.from_clouds(
        [jnp.asarray([[0.0], [2.0]]), jnp.asarray([[4.0]])]
    )
    kernel = RBFKernel(jnp.asarray(1.2))

    def evaluate(data: jax.Array) -> jax.Array:
        current = PackedClouds(data, measures.counts, measures.masses)
        return weighted_mmd_matrix(current, kernel=kernel).values

    actual = jax.jit(jax.vmap(evaluate))(jnp.stack([measures.data, measures.data + 3]))
    expected = weighted_mmd_matrix(measures, kernel=kernel).values
    np.testing.assert_allclose(
        actual, jnp.stack([expected, expected]), rtol=1e-13, atol=1e-13
    )


@pytest.mark.unit
@pytest.mark.property
@pytest.mark.parametrize("kernel", [LINEAR_KERNEL, RBFKernel(jnp.asarray(1.0))])
def test_weighted_geometry_preserves_large_translation(
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
) -> None:
    """Resolved unit differences cannot collapse inside Gram expectation algebra."""
    original = PackedClouds.from_clouds([jnp.asarray([[0.0]]), jnp.asarray([[1.0]])])
    translated = PackedClouds.from_clouds(
        [jnp.asarray([[1e10]]), jnp.asarray([[1e10 + 1.0]])]
    )
    expected = weighted_mmd_matrix(original, kernel=kernel).values
    actual = weighted_mmd_matrix(translated, kernel=kernel).values
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.unit
@pytest.mark.property
@pytest.mark.parametrize("kernel", [LINEAR_KERNEL, RBFKernel(jnp.asarray(1.0))])
def test_equal_measures_in_mixed_precision_have_zero_distance(
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
) -> None:
    """All self and cross terms must share the promoted working precision."""
    points = jnp.asarray(
        np.random.default_rng(0).normal(size=(4, 2)), dtype=jnp.float32
    )
    low = PackedClouds.from_clouds([points])
    high = PackedClouds.from_clouds([points.astype(jnp.float64)])
    actual = weighted_mmd_cross(low, high, kernel=kernel)
    np.testing.assert_allclose(actual, [[0.0]], rtol=0, atol=1e-14)


@pytest.mark.unit
@pytest.mark.property
def test_equal_measures_in_mixed_precision_have_zero_energy() -> None:
    """All energy blocks use the same promoted input precision."""
    points = jnp.asarray(
        np.random.default_rng(0).normal(size=(4, 2)), dtype=jnp.float32
    )
    low = PackedClouds.from_clouds([points])
    high = PackedClouds.from_clouds([points.astype(jnp.float64)])
    actual = weighted_energy_cross(low, high)
    np.testing.assert_allclose(actual, [[0.0]], rtol=0, atol=1e-14)


@pytest.mark.unit
@pytest.mark.regression
@pytest.mark.parametrize("offset", [0.0, 1e2, 1e4])
def test_float32_weighted_energy_has_no_far_anchor_cancellation(offset: float) -> None:
    """A distant first measure cannot degrade energy between two nearby measures.

    The former distance-induced-kernel route anchored every Gram entry at the
    first positive-mass atom; at a 1e4 offset float32 lost 42% of this value.
    """
    rng = np.random.default_rng(0)
    base = 1e-2 * rng.normal(size=(6, 1))
    near_a, near_b = base, base[:5] + 5e-2
    far = 1e-2 * rng.normal(size=(3, 1)) + offset
    measures = PackedClouds.from_clouds(
        [jnp.asarray(cloud, dtype=jnp.float32) for cloud in (far, near_a, near_b)]
    )

    def mean_distance(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.abs(u[:, None, 0] - v[None, :, 0]).mean())

    expected = np.sqrt(
        2 * mean_distance(near_a, near_b)
        - mean_distance(near_a, near_a)
        - mean_distance(near_b, near_b)
    )
    actual = weighted_energy_matrix(measures).values

    assert actual.dtype == jnp.float32
    assert float(actual[1, 2]) == pytest.approx(expected, rel=1e-5)


@pytest.mark.component
@pytest.mark.property
@pytest.mark.parametrize("exponent", [1.0, 1.5])
def test_uniform_weighted_energy_matches_native_unweighted_matrix(
    exponent: float,
) -> None:
    """Uniform masses reduce to the existing unequal-support packed energy core."""
    clouds = [
        jnp.asarray([[0.0, 1.0], [2.0, -1.0]]),
        jnp.asarray([[1.0, 2.0], [-1.0, 0.0], [3.0, 1.0]]),
        jnp.asarray([[4.0, 4.0]]),
    ]
    native = energy_distance_matrix(
        pack(clouds, require_positive_counts=True),
        exponent=exponent,
        metric="euclidean",
    )
    actual = weighted_energy_matrix(
        PackedClouds.from_clouds(clouds), ground="euclidean", exponent=exponent
    ).values
    np.testing.assert_allclose(actual, np.sqrt(native.values), rtol=1e-12, atol=1e-12)


@pytest.mark.component
@pytest.mark.regression
def test_float32_energy_zero_mass_extreme_padding_and_common_offset() -> None:
    """Zero-mass extremes and padding must not hide small, representable separation."""
    measures = PackedClouds.from_clouds(
        [
            jnp.asarray([[1e20], [10000.0]], dtype=jnp.float32),
            jnp.asarray([[10000.0625], [1e20], [10000.0625]], dtype=jnp.float32),
        ],
        [
            jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32),
        ],
    )
    expected = np.sqrt(2 * 0.0625)
    eager = weighted_energy_matrix(measures).values

    def evaluate_get_values(value: PackedClouds) -> Array:
        return weighted_energy_matrix(value).values

    compiled = jax.jit(evaluate_get_values)(measures)

    def evaluate_batched_energy_values(data: jax.Array, masses: jax.Array) -> Array:
        return weighted_energy_matrix(
            PackedClouds(data, measures.counts, masses)
        ).values

    batched = jax.vmap(evaluate_batched_energy_values)(
        jnp.stack([measures.data, measures.data]),
        jnp.stack([measures.masses, measures.masses]),
    )
    prepared = Energy().prepare(measures)
    for value in (eager, compiled, batched[0], batched[1], prepared.pairwise().values):
        assert value.dtype == jnp.float32
        np.testing.assert_allclose(value, [[0, expected], [expected, 0]], rtol=1e-6)


def _invalid_measure_collection(invalid: str) -> PackedClouds:
    """Keep the bad first lane beside two independent point-mass controls."""
    data = jnp.asarray([[[0.0], [0.0]], [[1.0], [0.0]], [[3.0], [0.0]]])
    counts = jnp.ones(3, dtype=jnp.int32)
    masses = jnp.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    if invalid == "total":
        masses = masses.at[0, 0].set(2)
    elif invalid == "negative":
        masses = masses.at[0].set(jnp.asarray([-1.0, 2.0]))
    elif invalid == "nonfinite-mass":
        masses = masses.at[0, 0].set(jnp.nan)
    elif invalid == "padding":
        masses = masses.at[0].set(jnp.asarray([0.75, 0.25]))
    elif invalid == "empty":
        counts = counts.at[0].set(0)
    elif invalid == "count":
        counts = counts.at[0].set(3)
    else:
        data = data.at[0, 0, 0].set(jnp.nan)
    return PackedClouds(data, counts, masses)


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("family", ["linear", "rbf", "energy"])
@pytest.mark.parametrize(
    "invalid",
    ["total", "negative", "nonfinite-mass", "padding", "empty", "count", "coordinate"],
)
def test_invalid_measures_remain_unavailable_without_poisoning_neighbors(
    family: str, invalid: str
) -> None:
    """Every matrix route retains NaN lanes and independent valid point distances."""
    measures = _invalid_measure_collection(invalid)
    np.testing.assert_array_equal(measures.valid, [False, True, True])
    kernel = RBFKernel(jnp.asarray(1.0)) if family == "rbf" else LINEAR_KERNEL

    def evaluate(value: PackedClouds) -> tuple[Array, Array, Array]:
        if family == "energy":
            return (
                weighted_energy_squared_cross(value, value),
                weighted_energy_matrix(value).values,
                jnp.sqrt(energy_distance_matrix(value, metric="euclidean").values),
            )
        return (
            weighted_mmd_squared_cross(value, value, kernel=kernel),
            weighted_mmd_matrix(value, kernel=kernel).values,
            mmd_matrix(value, kernel=kernel).values,
        )

    def duplicate_leaf(leaf: jax.Array) -> jax.Array:
        return jnp.stack([leaf, leaf])

    batched = jax.tree.map(duplicate_leaf, measures)
    executions = [evaluate(measures), jax.jit(evaluate)(measures)]
    executions.extend(zip(*jax.jit(jax.vmap(evaluate))(batched), strict=True))
    expected_squared = 2 * (1 - np.exp(-2)) if family == "rbf" else 4.0
    expected = np.asarray([[0.0, expected_squared], [expected_squared, 0.0]])
    for result in executions:
        for index, matrix in enumerate(result):
            assert np.isnan(matrix[0]).all()
            assert np.isnan(matrix[:, 0]).all()
            np.testing.assert_allclose(
                matrix[1:, 1:],
                expected if index == 0 else np.sqrt(expected),
                rtol=1e-6,
                atol=1e-12,
            )


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("family", ["linear", "rbf", "energy"])
def test_unavailable_singleton_matrix_does_not_acquire_a_zero_diagonal(
    family: str,
) -> None:
    """Matrix assembly cannot make an invalid lone probability a valid distance."""
    measures = PackedClouds(
        jnp.asarray([[[1.0]]]),
        jnp.asarray([1], dtype=jnp.int32),
        jnp.asarray([[2.0]]),
    )

    def evaluate(value: PackedClouds) -> Array:
        if family == "energy":
            return energy_distance_matrix(value, metric="euclidean").values
        kernel = RBFKernel(jnp.asarray(1.0)) if family == "rbf" else LINEAR_KERNEL
        return mmd_matrix(value, kernel=kernel).values

    assert np.isnan(evaluate(measures)[0, 0])
    assert np.isnan(jax.jit(evaluate)(measures)[0, 0])
