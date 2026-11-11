# standalone package-maintenance entry point
"""Axiom battery for the promoted exact balanced W1 (t46.9) and W2 (t88).

The parent contract predicted ``METRIC``. Measurement says otherwise, and the
witness that shows it is fixed rather than random — the same method note t46.1
recorded in Finding 2: property draws essentially never produce two samples that
differ by a pure rescaling, so a random battery reports ``METRIC`` and misses
the failure entirely.

The W2 rows added by t88 carry one further fixed witness, for the same reason:
a random cost matrix usually has one permutation that is optimal for both
orders, so a random battery cannot tell a genuine W2 solve from a W1 solve whose
matched costs were squared afterwards.
"""

from __future__ import annotations

from itertools import pairwise, permutations
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.axioms import (
    MARKER_AXIOMS,
    SEMIMETRIC,
    Axioms,
    Premetric,
    Semimetric,
)
from statadic.discrepancy.transport.exact import (
    EXACT_BALANCED_W1_AXIOMS,
    EXACT_BALANCED_W1_BRAND,
    EXACT_BALANCED_W2_AXIOMS,
    EXACT_BALANCED_W2_BRAND,
    InvalidCostMatrixError,
    InvalidWassersteinOrderError,
    NonfiniteCostError,
    balanced_wasserstein1_from_cost,
    balanced_wasserstein1_from_cost_jax,
    balanced_wasserstein2_from_cost,
    balanced_wasserstein2_from_cost_jax,
    balanced_wasserstein_from_cost,
    balanced_wasserstein_from_cost_jax,
    balanced_wasserstein_from_cost_result,
    exact_balanced_wasserstein,
    exact_balanced_wasserstein1,
    exact_balanced_wasserstein1_jax,
    exact_balanced_wasserstein2,
    exact_balanced_wasserstein2_jax,
    exact_balanced_wasserstein_jax,
)
from statadic.geometry.ground.metrics import sphere_chord_distance
from statadic.sample.packed import PackedClouds
from statadic.testing.axioms import assert_axioms, measure_axioms
from scipy.optimize import linprog


@pytest.mark.component
@pytest.mark.numerical
def test_exact_cloud_axes_broadcast_directly() -> None:
    """Exact transport returns array lanes for direct cloud-axis broadcasting."""
    points = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]], [[-1.0, 0.0], [0.0, -1.0]]])
    clouds = PackedClouds.from_array(points)
    reference = PackedClouds.from_array(points[0])
    expected = np.asarray([0.0, np.sqrt(2.0)])
    for order in (1.0, 2.0, 3.0):
        np.testing.assert_allclose(
            jax.jit(exact_balanced_wasserstein_jax)(clouds, reference, order),
            expected,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        exact_balanced_wasserstein1_jax(clouds, reference), expected, atol=1e-12
    )
    np.testing.assert_allclose(
        exact_balanced_wasserstein2_jax(clouds, reference), expected, atol=1e-12
    )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("order", [1.0, 2.0, 3.0])
def test_exact_padded_uniform_clouds_under_jit_and_vmap(order: float) -> None:
    """Inactive support and differing capacities do not alter assignment."""
    left = PackedClouds(
        jnp.array([[1.0, 0.0], [0.0, 1.0], [jnp.nan, jnp.nan]]), jnp.array(2)
    )
    right = PackedClouds(
        jnp.array([[0.0, 1.0], [-1.0, 0.0], [99.0, 99.0], [0.0, 0.0]]),
        jnp.array(2),
    )
    expected = exact_balanced_wasserstein(left, right, order)
    compiled = jax.jit(exact_balanced_wasserstein_jax)(left, right, order)
    np.testing.assert_allclose(compiled, expected, atol=1e-7)
    batch = PackedClouds(jnp.stack([left.data, left.data]), jnp.array([2, 1]))
    observed = jax.jit(exact_balanced_wasserstein_jax)(batch, right, order)
    np.testing.assert_allclose(observed[0], expected, atol=1e-7)
    assert np.isnan(observed[1])
    right_batch = PackedClouds(jnp.stack([right.data, right.data]), jnp.array([2, 1]))
    paired = jax.jit(exact_balanced_wasserstein_jax)(batch, right_batch, order)
    expected_second = exact_balanced_wasserstein(batch[1], right_batch[1], order)
    np.testing.assert_allclose(paired, [expected, expected_second], atol=1e-7)
    nonuniform = PackedClouds(left.data, left.counts, jnp.array([0.9, 0.1, 0.0]))
    assert np.isnan(jax.jit(exact_balanced_wasserstein_jax)(nonuniform, right, order))


if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )
    from types import ModuleType

    from statadic.core.typing import (
        Array,
    )
    from statadic.discrepancy.transport.exact import (
        ExactTransportResult,
    )

_N_ATOMS = 5
_N_FEATURES = 3
_N_OBJECTS = 4
_RESCALE = 3.0
_ZERO_FLOOR = 1e-8


def _w1(left: Array, right: Array) -> Array:
    """Adapt the NumPy kernel to the battery's JAX two-object signature.

    The battery is JAX-typed and this stage is NumPy; the two ``asarray`` calls
    are the whole adapter, and they are the reason the t46.9 modules could keep
    their NumPy annotations without the battery having to learn about them.
    """
    return jnp.asarray(
        exact_balanced_wasserstein1(
            PackedClouds.from_array(jnp.asarray(np.asarray(left))),
            PackedClouds.from_array(jnp.asarray(np.asarray(right))),
        )
    )


def _generic_objects() -> list[Array]:
    """Return structure-free samples of equal cardinality and width."""
    rng = np.random.default_rng(20260731)
    objects = [rng.normal(size=(_N_ATOMS, _N_FEATURES)) for _ in range(_N_OBJECTS)]
    return [
        jnp.asarray(value / np.linalg.norm(value, axis=1, keepdims=True))
        for value in objects
    ]


@pytest.mark.unit
@pytest.mark.property
def test_declared_axioms_hold_on_generic_objects() -> None:
    """Satisfy every declared flag on structure-free samples."""
    assert_axioms(_w1, EXACT_BALANCED_W1_AXIOMS, _generic_objects())


@pytest.mark.unit
@pytest.mark.property
def test_random_objects_alone_would_have_reported_a_metric() -> None:
    """Show why the parent contract's METRIC prediction looked correct."""
    assert measure_axioms(_w1, _generic_objects()) == Axioms.NONNEGATIVE | (
        (Axioms.ZERO_DIAGONAL | Axioms.SEPARATION) | Axioms.SYMMETRY | Axioms.TRIANGLE
    )


@pytest.mark.unit
@pytest.mark.contract
def test_exact_w1_rejects_a_pipeline_boundary_rescaling() -> None:
    """Reject off-sphere rows instead of silently changing the estimand."""
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(_N_ATOMS, _N_FEATURES))
    sample = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    other_raw = rng.normal(size=(_N_ATOMS, _N_FEATURES))
    other = other_raw / np.linalg.norm(other_raw, axis=1, keepdims=True)
    with pytest.raises(ValueError, match="already be L2-normalized"):
        exact_balanced_wasserstein1(
            PackedClouds.from_array(jnp.asarray(sample)),
            PackedClouds.from_array(jnp.asarray(_RESCALE * sample)),
        )
    assert exact_balanced_wasserstein1(
        PackedClouds.from_array(jnp.asarray(sample)),
        PackedClouds.from_array(jnp.asarray(sample)),
    ) == pytest.approx(0.0, abs=_ZERO_FLOOR)
    assert (
        exact_balanced_wasserstein1(
            PackedClouds.from_array(jnp.asarray(sample)),
            PackedClouds.from_array(jnp.asarray(other)),
        )
        > _ZERO_FLOOR
    )


@pytest.mark.unit
@pytest.mark.contract
def test_array_level_identity_is_not_declared() -> None:
    """Independent row permutations identify one empirical measure."""
    assert (Axioms.ZERO_DIAGONAL | Axioms.SEPARATION) not in EXACT_BALANCED_W1_AXIOMS


@pytest.mark.component
@pytest.mark.contract
def test_jax_hungarian_core_transforms_and_matches_eager() -> None:
    """Pin the OTT/Optax exact core under jit and vmap."""
    objects = jnp.stack(_generic_objects()[:2])
    eager = exact_balanced_wasserstein1(
        PackedClouds.from_array(jnp.asarray(np.asarray(objects[0]))),
        PackedClouds.from_array(jnp.asarray(np.asarray(objects[1]))),
    )
    compiled = jax.jit(exact_balanced_wasserstein1_jax)(
        PackedClouds.from_array(jnp.asarray(objects[0])),
        PackedClouds.from_array(jnp.asarray(objects[1])),
    )
    batched = jax.vmap(exact_balanced_wasserstein1_jax, in_axes=(0, None))(
        PackedClouds.from_array(jnp.asarray(objects)),
        PackedClouds.from_array(jnp.asarray(objects[1])),
    )
    assert float(compiled) == pytest.approx(eager, abs=1e-6)
    assert batched.shape == (2,)
    cost = sphere_chord_distance(objects[0], objects[1])
    assert float(jax.jit(balanced_wasserstein1_from_cost_jax)(cost)) == pytest.approx(
        eager, abs=1e-6
    )


@pytest.mark.unit
@pytest.mark.contract
def test_brand_is_implied_by_the_declared_axioms() -> None:
    """Promise no axiom through the static marker that the flags lack."""
    implied = MARKER_AXIOMS[EXACT_BALANCED_W1_BRAND]
    assert implied & ~EXACT_BALANCED_W1_AXIOMS == Axioms(0)


@pytest.mark.unit
@pytest.mark.contract
def test_brand_is_the_strongest_available() -> None:
    """Reject the next marker down the chain, which would promise IDENTITY."""
    assert EXACT_BALANCED_W1_BRAND is Premetric
    assert MARKER_AXIOMS[Semimetric] & ~EXACT_BALANCED_W1_AXIOMS == Axioms.SEPARATION
    assert SEMIMETRIC & ~EXACT_BALANCED_W1_AXIOMS == Axioms.SEPARATION


def _w2(left: Array, right: Array) -> Array:
    """Adapt the W2 door to the battery's JAX two-object signature."""
    return jnp.asarray(
        exact_balanced_wasserstein2(
            PackedClouds.from_array(jnp.asarray(np.asarray(left))),
            PackedClouds.from_array(jnp.asarray(np.asarray(right))),
        )
    )


@pytest.mark.unit
@pytest.mark.property
def test_declared_w2_axioms_hold_on_generic_objects() -> None:
    """Satisfy every declared W2 flag on structure-free samples."""
    assert_axioms(_w2, EXACT_BALANCED_W2_AXIOMS, _generic_objects())


@pytest.mark.unit
@pytest.mark.contract
def test_w2_brand_is_implied_and_strongest() -> None:
    """Promise no axiom the W2 flags lack, and none weaker than they support."""
    assert EXACT_BALANCED_W2_BRAND is Premetric
    assert MARKER_AXIOMS[EXACT_BALANCED_W2_BRAND] & ~EXACT_BALANCED_W2_AXIOMS == Axioms(
        0
    )
    assert MARKER_AXIOMS[Semimetric] & ~EXACT_BALANCED_W2_AXIOMS == Axioms.SEPARATION


@pytest.mark.unit
@pytest.mark.reference
def test_w2_matches_exhaustive_assignment() -> None:
    """Match a brute-force minimum over every permutation at solver precision."""
    left, right = _generic_objects()[:2]
    cost = np.asarray(sphere_chord_distance(left, right))
    squared = cost**2
    rows = np.arange(cost.shape[0])
    expected = np.sqrt(
        min(squared[rows, np.asarray(order)].mean() for order in permutations(rows))
    )
    observed = exact_balanced_wasserstein2(
        PackedClouds.from_array(jnp.asarray(np.asarray(left))),
        PackedClouds.from_array(jnp.asarray(np.asarray(right))),
    )
    assert observed == pytest.approx(float(expected), abs=1e-14)


@pytest.mark.unit
@pytest.mark.property
def test_w2_dominates_w1() -> None:
    """Order the two by Cauchy--Schwarz on the matched cost vector."""
    for left, right in pairwise(_generic_objects()):
        w1 = exact_balanced_wasserstein1(
            PackedClouds.from_array(jnp.asarray(np.asarray(left))),
            PackedClouds.from_array(jnp.asarray(np.asarray(right))),
        )
        w2 = exact_balanced_wasserstein2(
            PackedClouds.from_array(jnp.asarray(np.asarray(left))),
            PackedClouds.from_array(jnp.asarray(np.asarray(right))),
        )
        assert w1 <= w2 + _ZERO_FLOOR


@pytest.mark.unit
@pytest.mark.numerical
def test_w2_solves_its_own_assignment_not_w1s() -> None:
    """Reject squaring a W1-optimal matching in place of a W2 solve.

    The witness is the whole point: on this cost matrix the W1 optimum is the
    identity permutation and the W2 optimum is the swap, so a W2 reported from
    the W1 plan would be feasible, larger, and wrong.
    """
    cost = np.asarray([[0.0, 2.0], [2.0, 3.0]])
    assert balanced_wasserstein1_from_cost(cost) == pytest.approx(1.5, abs=1e-14)
    assert balanced_wasserstein2_from_cost(cost) == pytest.approx(2.0, abs=1e-14)
    w1_plan_rooted = float(np.sqrt(np.mean(np.diag(cost) ** 2)))
    assert w1_plan_rooted > balanced_wasserstein2_from_cost(cost)


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize(
    ("p", "specialized"),
    [
        (1.0, balanced_wasserstein1_from_cost),
        (2.0, balanced_wasserstein2_from_cost),
    ],
)
def test_generalized_cost_core_preserves_w1_w2_wrappers(
    p: float,
    specialized: Callable[[np.ndarray], float],
) -> None:
    """Keep the historical order-specific doors as exact projections of Wp."""
    cost = np.asarray([[0.0, 2.0], [2.0, 3.0]])
    assert balanced_wasserstein_from_cost(cost, p) == pytest.approx(
        specialized(cost),
        abs=1e-14,
    )


@pytest.mark.unit
@pytest.mark.reference
def test_generalized_w3_matches_exhaustive_assignment() -> None:
    """Verify a non-wrapper order against every feasible uniform assignment."""
    cost = np.asarray(
        [
            [0.5, 1.0, 2.0],
            [1.5, 0.25, 1.0],
            [0.75, 2.0, 1.25],
        ]
    )
    rows = np.arange(cost.shape[0])
    expected = min(
        np.mean(cost[rows, np.asarray(order)] ** 3.0) ** (1.0 / 3.0)
        for order in permutations(rows)
    )
    assert balanced_wasserstein_from_cost(cost, 3.0) == pytest.approx(
        expected,
        abs=1e-14,
    )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("p", [True, 0.5, 0.0, -1.0, np.inf, np.nan])
def test_generalized_eager_door_rejects_invalid_orders(p: object) -> None:
    """Reserve the Wasserstein name for finite metric orders p at least one."""
    with pytest.raises(InvalidWassersteinOrderError):
        balanced_wasserstein_from_cost(np.eye(2), p)


@pytest.mark.unit
@pytest.mark.numerical
def test_integer_costs_are_promoted_before_w2_squaring() -> None:
    """Integral costs near 50000 remain finite under the W2 objective."""
    cost = np.asarray([[50000, 50001], [50002, 50003]], dtype=np.int64)

    observed = balanced_wasserstein2_from_cost(cost)

    expected = float(np.sqrt(np.mean(np.asarray([50000**2, 50003**2]))))
    assert observed == pytest.approx(expected, rel=1e-6)


@pytest.mark.unit
@pytest.mark.numerical
def test_w2_scales_before_powering_large_float32_costs() -> None:
    """Avoid overflow without imposing x64 on a finite float32 W2 problem."""
    maximum = np.finfo(np.float32).max
    cost = np.asarray([[maximum]], dtype=np.float32)

    observed = balanced_wasserstein2_from_cost(cost)

    assert np.isfinite(observed)
    assert observed == pytest.approx(float(maximum), rel=1e-6)


@pytest.mark.unit
@pytest.mark.numerical
@pytest.mark.parametrize(
    ("dtype", "cost", "order"),
    [
        (np.float32, 1e-3, 20.0),
        (np.float32, 1.4757394e19, 2.0),
        (np.float64, 1e-200, 2.0),
        (np.float64, 1e200, 2.0),
    ],
)
def test_constant_cost_extreme_ranges_transform(
    dtype: type, cost: float, order: float
) -> None:
    """The exact constant-cost oracle survives powers and their reduction."""
    if dtype is np.float64 and not jax.config.x64_enabled:
        pytest.skip("float64 cases require caller-enabled JAX x64")
    matrix = np.full((2, 2), cost, dtype=dtype)
    array = jnp.asarray(matrix)

    def evaluate_balanced_wasserstein_from_cost_result(
        c: jax.Array,
    ) -> ExactTransportResult:
        return balanced_wasserstein_from_cost_result(c, order)

    results = [
        balanced_wasserstein_from_cost_result(array, order),
        jax.jit(balanced_wasserstein_from_cost_result)(array, order),
        jax.jit(jax.vmap(evaluate_balanced_wasserstein_from_cost_result))(
            jnp.stack([array, array])
        ),
    ]
    expected = float(matrix[0, 0])
    assert balanced_wasserstein_from_cost(matrix, order) == pytest.approx(
        expected, rel=2e-5, abs=0
    )
    for result in results:
        assert np.all(result.input_valid)
        assert np.all(result.finite)
        assert not np.any(result.power_underflow)
        np.testing.assert_allclose(result.value, expected, rtol=2e-5, atol=0)


@pytest.mark.unit
@pytest.mark.numerical
def test_mixed_range_preserves_selected_cost_scale_and_warns() -> None:
    """Unmatched large costs cannot erase the selected-cost mean."""
    matrix = jnp.asarray([[1e-10, 1e10], [1e10, 2e-10]], dtype=jnp.float32)
    result = balanced_wasserstein_from_cost_result(matrix, 2.0)
    expected = np.sqrt((1e-20 + 4e-20) / 2)
    assert bool(result.input_valid)
    assert bool(result.finite)
    assert bool(result.power_underflow)
    assert float(result.value) == pytest.approx(expected, rel=2e-6, abs=0)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("order", [1.0, 2.0, 3.0, 20.0])
def test_exact_cost_scale_and_exhaustive_assignment(order: float) -> None:
    """Compare the powered assignment against an independent enumeration."""
    cost = np.asarray([[0.5, 1.0, 2.0], [1.5, 0.25, 1.0], [0.75, 2.0, 1.25]])
    rows = np.arange(len(cost))
    expected = min(
        np.mean(cost[rows, np.asarray(assignment)] ** order) ** (1 / order)
        for assignment in permutations(rows)
    )
    reference = balanced_wasserstein_from_cost(cost, order)
    assert reference == pytest.approx(expected, rel=2e-6, abs=1e-7)
    assert balanced_wasserstein_from_cost(cost * 1e-3, order) == pytest.approx(
        reference * 1e-3, rel=2e-6, abs=0
    )


@pytest.mark.unit
@pytest.mark.contract
def test_exact_result_zero_and_invalid_numerical_lanes() -> None:
    """Zeros are valid; invalid lanes return finite-state evidence under vmap."""
    costs = jnp.asarray([[[0.0, 0.0], [0.0, 0.0]], [[1.0, -1.0], [0.0, 1.0]]])
    batched = jax.jit(
        jax.vmap(balanced_wasserstein_from_cost_result, in_axes=(0, None))
    )
    result = batched(costs, 2.0)
    np.testing.assert_array_equal(result.input_valid, [True, False])
    np.testing.assert_array_equal(result.finite, [True, False])
    np.testing.assert_array_equal(result.value[0], 0.0)
    assert np.isnan(result.value[1])
    assert not np.any(result.power_underflow)


@pytest.mark.unit
@pytest.mark.contract
def test_exact_result_invalid_orders_are_not_reported_as_valid() -> None:
    """Numeric order validity is retained for each transformed lane."""
    orders = jnp.asarray([1.0, 0.5, jnp.inf, jnp.nan])
    results = jax.jit(jax.vmap(balanced_wasserstein_from_cost_result, (None, 0)))(
        jnp.ones((2, 2)), orders
    )
    np.testing.assert_array_equal(results.input_valid, [True, False, False, False])
    np.testing.assert_array_equal(results.finite, [True, False, False, False])
    np.testing.assert_allclose(results.value[0], 1.0)
    assert np.all(np.isnan(results.value[1:]))


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (0, 0), (2,)])
def test_exact_result_rejects_invalid_cost_shapes(shape: tuple[int, ...]) -> None:
    """A uniform assignment requires one nonempty square matrix per lane."""
    cost = jnp.ones(shape)
    for solve in (
        balanced_wasserstein_from_cost_result,
        jax.jit(balanced_wasserstein_from_cost_result),
    ):
        with pytest.raises(InvalidCostMatrixError):
            solve(cost, 2.0)


@pytest.mark.unit
@pytest.mark.contract
def test_exact_result_rejects_nonscalar_order() -> None:
    """Batching orders requires vmap instead of a nonscalar single-lane order."""
    for solve in (
        balanced_wasserstein_from_cost_result,
        jax.jit(balanced_wasserstein_from_cost_result),
    ):
        with pytest.raises(InvalidWassersteinOrderError):
            solve(jnp.ones((2, 2)), jnp.ones(2))


@pytest.mark.unit
@pytest.mark.contract
def test_generalized_eager_door_still_rejects_nonfinite_costs() -> None:
    """Scaling stabilizes finite values but does not legalize infinite costs."""
    with pytest.raises(NonfiniteCostError):
        balanced_wasserstein_from_cost(np.asarray([[np.inf]]), 3.0)


@pytest.mark.component
@pytest.mark.contract
def test_jax_w2_core_transforms_and_matches_eager() -> None:
    """Pin the W2 assignment core under jit and vmap."""
    objects = jnp.stack(_generic_objects()[:2])
    eager = exact_balanced_wasserstein2(
        PackedClouds.from_array(jnp.asarray(np.asarray(objects[0]))),
        PackedClouds.from_array(jnp.asarray(np.asarray(objects[1]))),
    )
    compiled = jax.jit(exact_balanced_wasserstein2_jax)(
        PackedClouds.from_array(jnp.asarray(objects[0])),
        PackedClouds.from_array(jnp.asarray(objects[1])),
    )
    batched = jax.vmap(exact_balanced_wasserstein2_jax, in_axes=(0, None))(
        PackedClouds.from_array(jnp.asarray(objects)),
        PackedClouds.from_array(jnp.asarray(objects[1])),
    )
    assert float(compiled) == pytest.approx(eager, abs=1e-6)
    assert batched.shape == (2,)
    cost = sphere_chord_distance(objects[0], objects[1])
    assert float(jax.jit(balanced_wasserstein2_from_cost_jax)(cost)) == pytest.approx(
        eager, abs=1e-6
    )


@pytest.mark.component
@pytest.mark.contract
def test_generalized_jax_core_transforms_and_reuses_ground_metric() -> None:
    """Pin arbitrary p under jit/vmap and share the governed chord producer."""
    objects = jnp.stack(_generic_objects()[:2])
    cost = sphere_chord_distance(objects[0], objects[1])
    eager = exact_balanced_wasserstein(
        PackedClouds.from_array(jnp.asarray(np.asarray(objects[0]))),
        PackedClouds.from_array(jnp.asarray(np.asarray(objects[1]))),
        3.0,
    )
    compiled = jax.jit(exact_balanced_wasserstein_jax)(
        PackedClouds.from_array(jnp.asarray(objects[0])),
        PackedClouds.from_array(jnp.asarray(objects[1])),
        3.0,
    )
    batched = jax.vmap(exact_balanced_wasserstein_jax, in_axes=(0, None, None))(
        PackedClouds.from_array(jnp.asarray(objects)),
        PackedClouds.from_array(jnp.asarray(objects[1])),
        3.0,
    )
    from_cost = jax.jit(balanced_wasserstein_from_cost_jax)(cost, 3.0)
    assert float(compiled) == pytest.approx(eager, abs=1e-6)
    assert float(from_cost) == pytest.approx(eager, abs=1e-6)
    assert batched.shape == (2,)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("POT")
def test_w2_matches_pot_exact_emd(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Agree with POT's exact EMD on the squared ground cost."""
    ot = optional_reference("ot")
    left, right = _generic_objects()[:2]
    cost = np.asarray(sphere_chord_distance(left, right)) ** 2
    weights = np.full(cost.shape[0], 1.0 / cost.shape[0])
    expected = float(np.sqrt(ot.emd2(weights, weights, np.ascontiguousarray(cost))))
    observed = exact_balanced_wasserstein2(
        PackedClouds.from_array(jnp.asarray(np.asarray(left))),
        PackedClouds.from_array(jnp.asarray(np.asarray(right))),
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("order", [1.0, 2.0, 3.0])
def test_powered_assignment_attains_independent_transport_dual(order: float) -> None:
    """Compare the native result to feasible LP duals in original cost units."""
    slack_tolerance = 1e-10

    cost = np.array([[2.0, 1.0, 5.0], [2.0, 1.0, 5.0], [4.0, 7.0, 1.0]])
    powered = cost**order
    n = len(cost)
    masses = np.full(n, 1.0 / n)
    constraints = np.vstack(
        [np.kron(np.eye(n), np.ones(n)), np.kron(np.ones(n), np.eye(n))]
    )
    oracle = linprog(
        powered.ravel(),
        A_eq=constraints,
        b_eq=np.tile(masses, 2),
        bounds=(0, None),
        method="highs",
    )
    assert oracle.success
    row_dual, col_dual = np.split(oracle.eqlin.marginals, 2)
    slack = powered - row_dual[:, None] - col_dual[None, :]
    assert np.min(slack) >= -slack_tolerance
    np.testing.assert_allclose(oracle.x.reshape(n, n) * slack, 0, atol=slack_tolerance)

    fit = balanced_wasserstein_from_cost_result(jnp.asarray(cost), order)
    assert bool(fit.input_valid & fit.finite & ~fit.power_underflow)
    dual_value = masses @ row_dual + masses @ col_dual
    np.testing.assert_allclose(float(fit.value) ** order, dual_value, rtol=2e-6)
