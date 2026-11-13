"""Independent unequal-count energy functionals and packed reductions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.discrepancy.mmd.energy import (
    energy_distance,
    energy_test_statistic,
    energy_test_statistic_from_blocks,
    energy_test_statistic_from_distances,
    mixture_energy_distance,
    mixture_energy_distance_from_blocks,
)
from statadic.discrepancy.mmd.energy.balancing import energy_distance_kernel_mixture
from statadic.discrepancy.mmd.energy.matrix import (
    energy_distance_matrix,
)
from statadic.geometry.ground.metrics import ANGULAR, EUCLIDEAN
from statadic.sample.packed import PackedClouds, pack

from ..._budgets import examples
from ._reference import energy_v, mean_power_distance


@pytest.mark.unit
@pytest.mark.reference
def test_weighted_mixture_uses_atom_probabilities() -> None:
    """Flattening the finite mixture gives an independent bilinear witness."""
    x, p = np.asarray([0.0, 3.0]), np.asarray([0.2, 0.8])
    y, q = np.asarray([1.0, 2.0, 5.0]), np.asarray([0.4, 0.42, 0.18])
    expected = (
        2 * p @ np.abs(x[:, None] - y) @ q
        - p @ np.abs(x[:, None] - x) @ p
        - q @ np.abs(y[:, None] - y) @ q
    )
    target = PackedClouds.from_points(x[:, None], p)
    candidates = PackedClouds.from_clouds(
        [jnp.asarray([[1.0]]), jnp.asarray([[2.0], [5.0]])],
        [jnp.asarray([1.0]), jnp.asarray([0.7, 0.3])],
    )
    weights = jnp.asarray([0.4, 0.6])
    np.testing.assert_allclose(
        mixture_energy_distance(target, candidates, weights), expected
    )
    np.testing.assert_allclose(
        energy_distance_kernel_mixture(target, candidates, weights), expected
    )
    candidate_sites = [np.asarray([1.0]), np.asarray([2.0, 5.0])]
    candidate_masses = [np.asarray([1.0]), np.asarray([0.7, 0.3])]
    cross = np.asarray(
        [
            p @ np.abs(x[:, None] - sites) @ mass
            for sites, mass in zip(candidate_sites, candidate_masses, strict=True)
        ]
    )
    gram = np.asarray(
        [
            [
                left_mass @ np.abs(left[:, None] - right) @ right_mass
                for right, right_mass in zip(
                    candidate_sites, candidate_masses, strict=True
                )
            ]
            for left, left_mass in zip(candidate_sites, candidate_masses, strict=True)
        ]
    )
    target_self = p @ np.abs(x[:, None] - x) @ p

    def precomputed(value: jax.Array) -> jax.Array:
        return energy_distance_kernel_mixture(
            target,
            candidates,
            value,
            precomputed_components=(jnp.asarray(gram), jnp.asarray(cross)),
            precomputed_within=jnp.asarray(target_self),
        )

    np.testing.assert_allclose(jax.jit(precomputed)(weights), expected)
    np.testing.assert_allclose(
        jax.grad(precomputed)(weights), 2 * cross - 2 * gram @ np.asarray(weights)
    )
    np.testing.assert_allclose(jax.hessian(precomputed)(weights), -2 * gram)


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


def _clouds() -> list[np.ndarray]:
    """Unequal supports include repeated observations and a singleton."""
    return [
        np.asarray(values)[:, None] for values in ([0.0, 2.0, 2.0], [-1.0, 3.0], [4.0])
    ]


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("exponent", [0.5, 1.0, 1.5])
def test_scalar_energy_matches_nested_unequal_count_reference(exponent: float) -> None:
    """True empirical counts, repeats and open-interval powers define the value."""
    left, right = _clouds()[:2]
    expected = energy_v(left, right, exponent)

    actual = energy_distance(
        PackedClouds.from_array(jnp.asarray(jnp.asarray(left))),
        PackedClouds.from_array(jnp.asarray(jnp.asarray(right))),
        exponent,
        metric="euclidean",
    )

    assert np.isfinite(actual)
    assert float(actual) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("dcor")
@pytest.mark.parametrize("exponent", [0.5, 1.0, 1.5])
def test_euclidean_v_energy_matches_dcor_at_multiple_powers(
    optional_reference: Callable[[str], ModuleType],
    exponent: float,
) -> None:
    """Only dcor's matched Euclidean V-statistic is claimed here."""
    peer = optional_reference("dcor")
    left, right = _clouds()[:2]
    expected = peer.energy_distance(left, right, exponent=exponent, estimation_stat="V")

    actual = energy_distance(
        PackedClouds.from_array(jnp.asarray(jnp.asarray(left))),
        PackedClouds.from_array(jnp.asarray(jnp.asarray(right))),
        exponent,
        metric="euclidean",
    )

    assert float(actual) == pytest.approx(float(expected), rel=1e-12, abs=1e-12)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("dcor")
@pytest.mark.parametrize("exponent", [0.5, 1.0, 1.5])
def test_euclidean_v_test_statistic_matches_dcor_scaling(
    optional_reference: Callable[[str], ModuleType], exponent: float
) -> None:
    """Unequal sample counts distinguish the statistic from the raw discrepancy."""
    peer = optional_reference("dcor.homogeneity")
    left, right = _clouds()[:2]
    expected = peer.energy_test_statistic(
        left, right, exponent=exponent, estimation_stat="V"
    )
    actual = energy_test_statistic(
        jnp.asarray(left), jnp.asarray(right), exponent, metric="euclidean"
    )
    assert float(actual) == pytest.approx(float(expected), rel=1e-12, abs=1e-12)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("exponent", [0.5, 1.0, 1.5])
def test_ragged_energy_matrix_matches_independent_raw_functionals(
    exponent: float,
) -> None:
    """Every cell has its own true count; self-functionals remain zero."""
    clouds = _clouds()
    expected = np.asarray([[energy_v(x, y, exponent) for y in clouds] for x in clouds])

    actual = energy_distance_matrix(
        pack([jnp.asarray(x) for x in clouds], require_positive_counts=True),
        exponent,
        metric="euclidean",
    )

    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.diag(actual), np.zeros(len(clouds)))


@pytest.mark.component
@pytest.mark.reference
def test_pooled_and_block_energy_statistics_match_independent_sums() -> None:
    """Pooled distances and separately supplied block sums use the same scaling."""
    left, right = _clouds()[:2]
    pooled = np.concatenate([left, right])
    distances = np.asarray([[abs(float(x[0] - y[0])) for y in pooled] for x in pooled])
    means = np.asarray(
        [[mean_power_distance(x, y) for y in (left, right)] for x in (left, right)]
    )
    counts = np.asarray([len(left), len(right)])
    sums = means * counts[:, None] * counts[None, :]
    expected = len(left) * len(right) / len(pooled) * energy_v(left, right)

    scalar = energy_test_statistic(
        jnp.asarray(left), jnp.asarray(right), metric="euclidean"
    )
    matrix = energy_test_statistic_from_distances(jnp.asarray(distances), len(left))
    blocks = energy_test_statistic_from_blocks(jnp.asarray(sums), len(left), len(right))

    np.testing.assert_allclose(
        [scalar, matrix, blocks], expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("weights", [[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]])
def test_mixture_energy_matches_independent_candidate_means(
    weights: list[float],
) -> None:
    """A zero mixture coefficient does not change the retained candidate counts."""
    clouds = _clouds()
    means = np.asarray([[mean_power_distance(x, y) for y in clouds] for x in clouds])
    counts = np.asarray([len(x) for x in clouds])
    weights_array = np.asarray(weights)
    expected = (
        2 * weights_array @ means[0, 1:]
        - means[0, 0]
        - weights_array @ means[1:, 1:] @ weights_array
    )

    scalar = mixture_energy_distance(
        PackedClouds.from_array(jnp.asarray(jnp.asarray(clouds[0]))),
        pack([jnp.asarray(x) for x in clouds[1:]], require_positive_counts=True),
        jnp.asarray(weights),
    )
    blocks = mixture_energy_distance_from_blocks(
        jnp.asarray(means * counts[:, None] * counts[None, :]),
        jnp.asarray(counts),
        jnp.asarray(weights),
    )

    np.testing.assert_allclose([scalar, blocks], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("angular", [False, True])
def test_poisoned_inactive_energy_storage_uses_true_counts(*, angular: bool) -> None:
    """Nonfinite padding never contributes mass or contaminates active pairs."""
    clouds = [np.asarray([[1.0, 0.0], [0.0, 1.0]]), np.asarray([[-1.0, 0.0]])]
    packed = pack([jnp.asarray(x) for x in clouds])
    poisoned = packed.data.at[1, 1].set(jnp.asarray([jnp.nan, jnp.inf]))
    expected = np.asarray(
        [[energy_v(x, y, angular=angular) for y in clouds] for x in clouds]
    )
    ground = ANGULAR if angular else EUCLIDEAN

    def evaluate_energy_distance_matrix_packed_core(data: jax.Array) -> jax.Array:
        return energy_distance_matrix(
            PackedClouds(data, packed.counts), 1.0, metric=ground
        )

    actual = jax.jit(evaluate_energy_distance_matrix_packed_core)(poisoned)

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.component
@pytest.mark.property
@examples(12)
@given(hnp.arrays(np.float64, (5, 2), elements=st.integers(-3, 3).map(float)))
def test_generated_energy_permutation_isometry_and_homogeneity(raw: np.ndarray) -> None:
    """Euclidean laws are checked against raw functionals under known power scaling."""
    left, right = raw[:2], raw[2:]
    expected = energy_v(left, right, 1.5)
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])

    actual = energy_distance(
        PackedClouds.from_array(jnp.asarray(jnp.asarray(left[::-1]))),
        PackedClouds.from_array(jnp.asarray(jnp.asarray(right[::-1]))),
        1.5,
        metric="euclidean",
    )
    transformed = energy_distance(
        PackedClouds.from_array(jnp.asarray(jnp.asarray(3 * left @ rotation + 2))),
        PackedClouds.from_array(jnp.asarray(jnp.asarray(3 * right @ rotation + 2))),
        1.5,
        metric="euclidean",
    )

    assert float(actual) == pytest.approx(expected, rel=1e-11, abs=1e-11)
    assert float(transformed) == pytest.approx(3**1.5 * expected, rel=1e-11, abs=1e-11)


@pytest.mark.component
@pytest.mark.reference
def test_large_euclidean_reduction_matches_weighted_dense_reference() -> None:
    """Streamed rows retain finite-mass weights and ignore inactive storage."""
    rng = np.random.default_rng(812)
    x = rng.normal(size=(135, 8))
    y = rng.normal(size=(130, 8))
    wx = rng.dirichlet(np.ones(len(x)))
    wy = rng.dirichlet(np.ones(len(y)))
    exponent = 1.25

    def block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1) ** exponent

    expected = 2 * wx @ block(x, y) @ wy - wx @ block(x, x) @ wx - wy @ block(y, y) @ wy
    padded = PackedClouds(
        jnp.asarray(np.pad(x, ((0, 1), (0, 0)), constant_values=np.nan)),
        jnp.asarray(len(x), dtype=jnp.int32),
        jnp.asarray(np.append(wx, 0.0)),
    )
    result = energy_distance(
        padded,
        PackedClouds.from_array(jnp.asarray(y), jnp.asarray(wy)),
        exponent,
        metric="euclidean",
    )
    np.testing.assert_allclose(result, expected, atol=2e-12, rtol=2e-12)


def _coordinate_gradient(
    x: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray, exponent: float
) -> np.ndarray:
    """Independent derivative of the finite weighted Euclidean energy sum."""
    cross = x[:, None, :] - y[None, :, :]
    within = x[:, None, :] - x[None, :, :]

    def weighted_directions(differences: np.ndarray, mass: np.ndarray) -> np.ndarray:
        squared = np.sum(differences**2, axis=-1)
        positive = squared > 0
        powers = np.where(
            positive, np.where(positive, squared, 1) ** (exponent / 2 - 1), 0
        )
        return np.einsum("ijd,ij,j->id", differences, powers, mass)

    return (
        2
        * exponent
        * a[:, None]
        * (weighted_directions(cross, b) - weighted_directions(within, a))
    )


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("exponent", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("shape", [(4, 5, 2), (129, 131, 8)])
def test_weighted_energy_coordinate_derivatives_match_analytic_reference(  # noqa: PLR0914  # independent coordinate/JVP/HVP witnesses share one fixture
    exponent: float, shape: tuple[int, int, int]
) -> None:
    """Dense and streamed self-pairs retain zero derivatives at all three powers."""
    n, m, d = shape
    generator = np.random.default_rng(293)
    x, y = generator.normal(size=(n, d)), generator.normal(size=(m, d))
    a, b = generator.dirichlet(np.ones(n)), generator.dirichlet(np.ones(m))
    left, right = PackedClouds.from_points(x, a), PackedClouds.from_points(y, b)

    def objective(coordinates: jax.Array) -> jax.Array:
        return energy_distance(
            left.with_coordinates(coordinates), right, exponent, metric="euclidean"
        )

    coordinates = jnp.asarray(x)
    expected = _coordinate_gradient(x, y, a, b, exponent)
    gradient = jax.jit(jax.grad(objective))(coordinates)
    np.testing.assert_allclose(gradient, expected, rtol=2e-10, atol=2e-11)
    direction_host = generator.normal(size=x.shape)
    direction = jnp.asarray(direction_host)
    _, tangent = jax.jvp(objective, (coordinates,), (direction,))
    np.testing.assert_allclose(
        tangent, np.sum(expected * direction_host), rtol=2e-10, atol=2e-11
    )
    small_support = 4
    if n == small_support:
        step = 1e-5
        finite_hvp = (
            _coordinate_gradient(x + step * direction_host, y, a, b, exponent)
            - _coordinate_gradient(x - step * direction_host, y, a, b, exponent)
        ) / (2 * step)
        _, hvp = jax.jvp(jax.grad(objective), (coordinates,), (direction,))
        np.testing.assert_allclose(hvp, finite_hvp, rtol=2e-7, atol=2e-8)
        hessian = jax.jit(jax.hessian(objective))(coordinates)
        np.testing.assert_allclose(
            np.einsum("ijkl,kl->ij", np.asarray(hessian), direction_host),
            finite_hvp,
            rtol=2e-7,
            atol=2e-8,
        )


@pytest.mark.unit
@pytest.mark.reference
def test_weighted_energy_batch_scan_and_mass_logit_derivatives() -> None:
    """Changing coordinates and masses remain live through their JAX consumers."""
    x = np.asarray([[0.2, -0.3], [1.1, 0.7], [-0.4, 0.2]])
    y = np.asarray([[0.8, 0.4], [1.4, -0.2]])
    a, b = np.asarray([0.2, 0.5, 0.3]), np.asarray([0.6, 0.4])
    left, right = PackedClouds.from_points(x, a), PackedClouds.from_points(y, b)
    changed = np.stack([x, 1.2 * x + 0.1])

    def objective(coordinates: jax.Array) -> jax.Array:
        return energy_distance(
            left.with_coordinates(coordinates), right, metric="euclidean"
        )

    expected = np.stack(
        [_coordinate_gradient(sites, y, a, b, 1.0) for sites in changed]
    )
    gradients = jax.jit(jax.vmap(jax.grad(objective)))(jnp.asarray(changed))
    np.testing.assert_allclose(gradients, expected, rtol=1e-11, atol=1e-12)

    def accumulate(coordinates: jax.Array) -> jax.Array:
        def step(total: jax.Array, sites: jax.Array) -> tuple[jax.Array, jax.Array]:
            value = objective(sites)
            return total + value, value

        return jax.lax.scan(step, jnp.zeros(()), coordinates)[0]

    np.testing.assert_allclose(
        jax.jit(jax.grad(accumulate))(jnp.asarray(changed)),
        expected,
        rtol=1e-11,
        atol=1e-12,
    )

    def mass_objective(logits: jax.Array) -> jax.Array:
        current = PackedClouds(left.data, left.counts, jax.nn.softmax(logits))
        return energy_distance(current, right, metric="euclidean")

    cross = np.linalg.norm(x[:, None] - y[None, :], axis=-1)
    within = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    derivative = 2 * cross @ b - 2 * within @ a
    mass_gradient = a * (derivative - a @ derivative)
    np.testing.assert_allclose(
        jax.jit(jax.grad(mass_objective))(jnp.log(jnp.asarray(a))),
        mass_gradient,
        rtol=1e-11,
        atol=1e-12,
    )
