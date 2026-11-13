"""Independent weighted transport and transformation witnesses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from scipy.optimize import linprog
from scipy.stats import wasserstein_distance
from statadic.discrepancy.transport.sliced import (
    ProjectionDesign,
    sliced_wasserstein_under_design,
    w1_1d,
    w2_1d,
)
from statadic.discrepancy.transport.univariate import (
    wasserstein_1d,
    wasserstein_1d_power,
)
from statadic.sample.packed import PackedClouds

from ..._budgets import examples
from ..._strategies import float_arrays

if TYPE_CHECKING:
    from jax import Array
    from jax.typing import DTypeLike


_FAST_X = jnp.array([2.0, 0.1, -1.0])
_FAST_Y = jnp.array([0.5, -2.0, 1.2])


@pytest.mark.unit
@pytest.mark.property
@pytest.mark.parametrize("masses", [[0.3] * 7, [0.1] * 10, [1.0, 2.0, 0.5]])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_trailing_padding_receives_no_cumulative_roundoff(
    masses: list[float], dtype: DTypeLike
) -> None:
    """Identical laws stay at zero when sorted padding follows every genuine atom."""
    weights = jnp.asarray([*masses, 0.0], dtype=dtype)
    padded = PackedClouds.from_array(
        jnp.asarray([*([-1e3] * len(masses)), 0.0], dtype=dtype)[..., None],
        weights / jnp.sum(weights),
    )
    target = PackedClouds.from_array(jnp.asarray([-1e3], dtype=dtype)[..., None])
    assert float(wasserstein_1d_power(padded, target, p=2)) == pytest.approx(
        0.0, abs=0.0
    )


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("order", [1, 1.5, 2, 3])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_weighted_unequal_supports_match_independent_linear_program(
    order: float, dtype: DTypeLike
) -> None:
    """A transportation LP independently checks the quantile integral."""
    left = np.array([2.0, -1.0, 0.5])
    right = np.array([3.0, -0.2, 1.1, 0.4, -2.0])
    wx, wy = np.array([2.0, 3.0, 5.0]), np.array([1.0, 2.0, 3.0, 2.0, 2.0])
    cost = np.abs(left[:, None] - right[None, :]) ** order
    constraints = np.concatenate(
        (
            np.kron(np.eye(len(left)), np.ones((1, len(right)))),
            np.tile(np.eye(len(right)), (1, len(left))),
        ),
        axis=0,
    )
    result = linprog(
        cost.ravel(),
        A_eq=constraints,
        b_eq=np.concatenate((wx / wx.sum(), wy / wy.sum())),
        bounds=(0, None),
        method="highs",
    )
    assert result.success

    def evaluate_wasserstein_1d_power(
        x: jax.Array, y: jax.Array, a: jax.Array, b: jax.Array
    ) -> Array:
        return wasserstein_1d_power(
            PackedClouds.from_array(
                jnp.asarray(x)[..., None],
                jnp.asarray(a) / jnp.sum(jnp.asarray(a), axis=-1, keepdims=True),
            ),
            PackedClouds.from_array(
                jnp.asarray(y)[..., None],
                jnp.asarray(b) / jnp.sum(jnp.asarray(b), axis=-1, keepdims=True),
            ),
            p=order,
        )

    actual = jax.jit(evaluate_wasserstein_1d_power)(
        jnp.asarray(left, dtype=dtype),
        jnp.asarray(right, dtype=dtype),
        jnp.asarray(wx, dtype=dtype),
        jnp.asarray(wy, dtype=dtype),
    )
    tolerance = 2e-6 if dtype == jnp.float32 else 2e-12
    assert float(actual) == pytest.approx(result.fun, rel=tolerance, abs=tolerance)


@pytest.mark.unit
@pytest.mark.reference
def test_w1_unequal_uniform_matches_scipy() -> None:
    """Uniform unequal samples agree with an independent public W1 API."""
    x, y = jnp.array([-1.0, 0.0, 3.0]), jnp.array([0.2, 1.0])
    assert float(
        w1_1d(
            PackedClouds.from_array(jnp.asarray(x)[..., None]),
            PackedClouds.from_array(jnp.asarray(y)[..., None]),
        )
    ) == pytest.approx(wasserstein_distance(x, y))
    assert float(
        w2_1d(
            PackedClouds.from_array(jnp.asarray(x)[..., None]),
            PackedClouds.from_array(jnp.asarray(y)[..., None]),
        )
    ) ** 2 == pytest.approx(
        float(
            wasserstein_1d_power(
                PackedClouds.from_array(jnp.asarray(x)[..., None]),
                PackedClouds.from_array(jnp.asarray(y)[..., None]),
            )
        )
    )


@pytest.mark.unit
@pytest.mark.numerical
@examples(40)
@example(order=1, x=_FAST_X, y=_FAST_Y)
@example(order=2, x=_FAST_X, y=_FAST_Y)
@example(order=3, x=_FAST_X, y=_FAST_Y)
@given(order=st.sampled_from([1, 2, 3]), x=float_arrays((3,)), y=float_arrays((3,)))
def test_uniform_fast_path_retains_sorted_difference_result(
    order: int, x: jax.Array, y: jax.Array
) -> None:
    """Existing equal-cardinality formulas retain their arithmetic."""
    expected = jnp.mean(jnp.abs(jnp.sort(x) - jnp.sort(y)) ** order)
    np.testing.assert_allclose(
        wasserstein_1d_power(
            PackedClouds.from_array(jnp.asarray(x)[..., None]),
            PackedClouds.from_array(jnp.asarray(y)[..., None]),
            p=order,
        ),
        expected,
        rtol=4 * np.finfo(np.float64).eps,
        atol=0,
    )


@pytest.mark.unit
@pytest.mark.contract
def test_atom_weights_and_support_batch_axes_broadcast() -> None:
    """Shared sites and independently batched masses produce a pair grid."""
    x, y = jnp.array([0.0, 2.0]), jnp.array([0.1, 1.0, 3.0])
    a = jnp.array([[[0.2, 0.8]], [[0.7, 0.3]]])
    b = jnp.array([[[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.1, 0.1, 0.8]]])
    actual = jax.jit(wasserstein_1d_power)(
        PackedClouds.from_array(
            jnp.asarray(x)[..., None],
            jnp.asarray(a) / jnp.sum(jnp.asarray(a), axis=-1, keepdims=True),
        ),
        PackedClouds.from_array(
            jnp.asarray(y)[..., None],
            jnp.asarray(b) / jnp.sum(jnp.asarray(b), axis=-1, keepdims=True),
        ),
    )
    expected = jnp.stack(
        [
            jnp.stack(
                [
                    wasserstein_1d_power(
                        PackedClouds.from_array(
                            jnp.asarray(x)[..., None],
                            jnp.asarray(row[0])
                            / jnp.sum(jnp.asarray(row[0]), axis=-1, keepdims=True),
                        ),
                        PackedClouds.from_array(
                            jnp.asarray(y)[..., None],
                            jnp.asarray(column)
                            / jnp.sum(jnp.asarray(column), axis=-1, keepdims=True),
                        ),
                    )
                    for column in b[0]
                ]
            )
            for row in a
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.unit
@pytest.mark.property
def test_zero_mass_coordinates_are_inert_in_values_and_gradients() -> None:
    """Inactive padding can contain NaN without contaminating differentiation."""
    x = jnp.array([-1.0, 2.0, jnp.nan])
    y = jnp.array([0.0, 3.0])
    weights = jnp.array([0.3, 0.7, 0.0])

    def evaluate(locations: jax.Array) -> jax.Array:
        return wasserstein_1d_power(
            PackedClouds.from_array(
                jnp.asarray(locations)[..., None],
                jnp.asarray(weights)
                / jnp.sum(jnp.asarray(weights), axis=-1, keepdims=True),
            ),
            PackedClouds.from_array(jnp.asarray(y)[..., None]),
        )

    actual = evaluate(x)
    expected = wasserstein_1d_power(
        PackedClouds.from_array(
            jnp.asarray(x[:2])[..., None],
            jnp.asarray(weights[:2])
            / jnp.sum(jnp.asarray(weights[:2]), axis=-1, keepdims=True),
        ),
        PackedClouds.from_array(jnp.asarray(y)[..., None]),
    )
    assert float(actual) == pytest.approx(float(expected))
    gradient = jax.jit(jax.grad(evaluate))(x)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(gradient[-1]) == pytest.approx(0.0, abs=0.0)


@pytest.mark.unit
@pytest.mark.reference
def test_parameter_gradients_match_finite_differences_away_from_quantile_ties() -> None:
    """Both location and mass parameters retain piecewise derivatives."""
    x, y = jnp.array([-1.0, 0.5, 2.0]), jnp.array([-0.7, 1.0, 3.0])
    target = jnp.array([0.17, 0.26, 0.57])

    def objective(parameters: jax.Array) -> jax.Array:
        positions = x + parameters[0] * jnp.array([0.2, -0.1, 0.3])
        weights = jax.nn.softmax(parameters[1:])
        return wasserstein_1d_power(
            PackedClouds.from_array(
                jnp.asarray(positions)[..., None],
                jnp.asarray(weights)
                / jnp.sum(jnp.asarray(weights), axis=-1, keepdims=True),
            ),
            PackedClouds.from_array(
                jnp.asarray(y)[..., None],
                jnp.asarray(target)
                / jnp.sum(jnp.asarray(target), axis=-1, keepdims=True),
            ),
        )

    theta = jnp.concatenate((jnp.array([0.1]), jnp.log(jnp.array([0.21, 0.33, 0.46]))))
    actual = jax.jit(jax.grad(objective))(theta)
    step = 1e-5
    expected = jnp.array(
        [
            (objective(theta + step * direction) - objective(theta - step * direction))
            / (2 * step)
            for direction in jnp.eye(theta.size)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    _, tangent = jax.jvp(objective, (theta,), (jnp.ones_like(theta),))
    assert float(tangent) == pytest.approx(float(actual.sum()), abs=1e-12)


@pytest.mark.unit
@pytest.mark.contract
def test_invalid_numeric_lanes_stay_visible_under_vmap() -> None:
    """A failed mass vector cannot spoil or disappear from its valid neighbor."""
    weights = jnp.array([[0.2, 0.8], [-0.1, 1.1], [0.0, 0.0]])

    def evaluate_wasserstein_1d_power(w: jax.Array) -> Array:
        return wasserstein_1d_power(
            PackedClouds.from_array(
                jnp.asarray(jnp.array([0.0, 1.0]))[..., None],
                jnp.asarray(w) / jnp.sum(jnp.asarray(w), axis=-1, keepdims=True),
            ),
            PackedClouds.from_array(jnp.asarray(jnp.array([0.5]))[..., None]),
        )

    actual = jax.jit(jax.vmap(evaluate_wasserstein_1d_power))(weights)
    assert float(actual[0]) == pytest.approx(0.25)
    assert bool(jnp.all(jnp.isnan(actual[1:])))


@pytest.mark.component
@pytest.mark.property
@pytest.mark.parametrize("order", [1, 2])
def test_sliced_d1_uses_weighted_unequal_quantiles(order: int) -> None:
    """Every one-dimensional unit projection gives the same exact distance."""
    x, y = jnp.array([[-1.0], [2.0]]), jnp.array([[0.1], [1.0], [3.0]])
    a, b = jnp.array([0.4, 0.6]), jnp.array([0.2, 0.3, 0.5])
    design = ProjectionDesign(jax.random.key(19), n_projections=8, p=order)

    def evaluate_sliced_wasserstein_under_design(
        left: jax.Array, right: jax.Array
    ) -> Array:
        return sliced_wasserstein_under_design(
            design,
            PackedClouds.from_array(
                jnp.asarray(left),
                jnp.asarray(a) / jnp.sum(jnp.asarray(a), axis=-1, keepdims=True),
            ),
            PackedClouds.from_array(
                jnp.asarray(right),
                jnp.asarray(b) / jnp.sum(jnp.asarray(b), axis=-1, keepdims=True),
            ),
        )

    actual = jax.jit(evaluate_sliced_wasserstein_under_design)(x, y)
    expected = wasserstein_1d(
        PackedClouds.from_array(
            jnp.asarray(x[:, 0])[..., None],
            jnp.asarray(a) / jnp.sum(jnp.asarray(a), axis=-1, keepdims=True),
        ),
        PackedClouds.from_array(
            jnp.asarray(y[:, 0])[..., None],
            jnp.asarray(b) / jnp.sum(jnp.asarray(b), axis=-1, keepdims=True),
        ),
        p=order,
    )
    assert float(actual) == pytest.approx(float(expected), abs=1e-12)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("order", [0, float("inf"), float("nan"), True])
def test_invalid_static_orders_raise(order: float) -> None:
    """Invalid transport orders fail before any compiled execution."""
    with pytest.raises((TypeError, ValueError), match="order"):
        wasserstein_1d_power(
            PackedClouds.from_array(jnp.asarray(jnp.ones(2))[..., None]),
            PackedClouds.from_array(jnp.asarray(jnp.ones(3))[..., None]),
            p=order,
        )


@pytest.mark.unit
@pytest.mark.contract
def test_malformed_support_and_atom_axes_raise() -> None:
    """Empty supports and incompatible atom axes are structural errors."""
    with pytest.raises(ValueError, match="nonempty"):
        wasserstein_1d_power(
            PackedClouds.from_array(jnp.asarray(jnp.ones(0))[..., None]),
            PackedClouds.from_array(jnp.asarray(jnp.ones(3))[..., None]),
        )
    with pytest.raises(ValueError, match="Atom weights"):
        wasserstein_1d_power(
            PackedClouds.from_array(
                jnp.asarray(jnp.ones(2))[..., None],
                jnp.asarray(jnp.ones(4))
                / jnp.sum(jnp.asarray(jnp.ones(4)), axis=-1, keepdims=True),
            ),
            PackedClouds.from_array(jnp.asarray(jnp.ones(3))[..., None]),
        )


@pytest.mark.component
@pytest.mark.reference
def test_sliced_retains_explicit_cloud_batch() -> None:
    """A mass batch produces separate distances, checked against transport LPs."""
    design = ProjectionDesign(jax.random.key(19), n_projections=8)
    x, y = np.array([0.0, 2.0]), np.array([0.1, 1.0, 3.0])
    weights = np.array([[0.2, 0.8], [0.7, 0.3]])
    left = PackedClouds.from_array(jnp.asarray(x[:, None]), jnp.asarray(weights))
    right = PackedClouds.from_points(jnp.asarray(y[:, None]))
    actual = sliced_wasserstein_under_design(design, left, right)
    constraints = np.concatenate(
        (np.kron(np.eye(2), np.ones((1, 3))), np.tile(np.eye(3), (1, 2)))
    )
    expected = []
    for row in weights:
        result = linprog(
            ((x[:, None] - y[None, :]) ** 2).ravel(),
            A_eq=constraints,
            b_eq=np.concatenate((row, np.full(3, 1 / 3))),
            bounds=(0, None),
            method="highs",
        )
        assert result.success
        expected.append(np.sqrt(result.fun))
    np.testing.assert_allclose(actual, expected)


@pytest.mark.unit
@pytest.mark.reference
def test_w1_cdf_integral_retains_weight_gradients_and_ignores_padding() -> None:
    """Independent SciPy values and a finite difference on a variable mass."""
    left = jnp.array([2.0, 1.0, jnp.nan, 1.0])
    right = jnp.array([3.0, -2.0, -0.5])
    right_mass = jnp.array([0.2, 0.4, 0.4])

    def value(q: jax.Array) -> jax.Array:
        weights = jnp.array([q, 0.5 - q, 0.0, 0.5])
        return wasserstein_1d(
            PackedClouds.from_array(left[:, None], weights),
            PackedClouds.from_array(right[:, None], right_mass),
            p=1,
        )

    def oracle(q: float) -> float:
        return float(
            wasserstein_distance([2.0, 1.0, 1.0], right, [q, 0.5 - q, 0.5], right_mass)
        )

    q = 0.3
    result, derivative = jax.jit(jax.value_and_grad(value))(jnp.asarray(q))
    assert float(result) == pytest.approx(oracle(q), abs=1e-12)
    step = 1e-5
    assert float(derivative) == pytest.approx(
        (oracle(q + step) - oracle(q - step)) / (2 * step)
    )


@pytest.mark.unit
@pytest.mark.numerical
def test_w1_zero_imbalance_ignores_overflowing_coordinate_span() -> None:
    """The common gap of identical measures has zero area even at dtype limits."""
    limits = PackedClouds.from_array(
        jnp.array([[-1e38], [1e38]], dtype=jnp.float32) * 2
    )
    assert float(wasserstein_1d(limits, limits, p=1)) == pytest.approx(0.0)


@pytest.mark.unit
@pytest.mark.regression
@pytest.mark.parametrize(
    ("dtype", "extent"), [(jnp.float32, 1e20), (jnp.float64, 1e200)]
)
@pytest.mark.parametrize("order", [1.5, 2, 3])
def test_zero_width_quantile_intervals_cannot_overflow_self_transport(
    dtype: DTypeLike, extent: float, order: float
) -> None:
    """Identical probability laws cost zero even if an unused gap power overflows."""
    with jax.enable_x64(dtype == jnp.float64):
        cloud = PackedClouds.from_array(
            jnp.asarray([[-extent], [extent], [jnp.nan]], dtype=dtype),
            jnp.asarray([0.5, 0.5, 0.0], dtype=dtype),
        )

        def objective(value: PackedClouds) -> Array:
            return wasserstein_1d_power(value, value, p=order)

        np.testing.assert_array_equal(objective(cloud), 0.0)
        np.testing.assert_array_equal(jax.jit(objective)(cloud), 0.0)

        def duplicate_leaf(leaf: jax.Array) -> jax.Array:
            return jnp.stack([leaf, leaf])

        batched = jax.tree.map(duplicate_leaf, cloud)
        np.testing.assert_array_equal(jax.jit(jax.vmap(objective))(batched), [0, 0])
