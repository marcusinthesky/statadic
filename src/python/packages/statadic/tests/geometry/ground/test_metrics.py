# jaxtyping shape strings are runtime contracts.
"""Ground distances and their scipy-shaped public API.

Two surfaces in one module because they are one module: the four dissimilarity
callables (``angular``, ``cosine``, ``euclidean``, ``normalized_euclidean``) and
the ``cdist``/``pdist`` dispatchers that apply them.

What is *not* here: whether each callable satisfies the axioms it declares. That
is measured in ``tests/testing/test_axioms.py`` against the shared ``CASES``
table, and the agreement between that measurement and this module's shipped
``DECLARED_AXIOMS`` is asserted in
``tests/migration/test_ground_discrepancy_migration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from jax import random
from statadic.core.axioms import (
    METRIC,
    LawProperties,
)
from statadic.core.domains import (
    FiniteValues,
)
from statadic.geometry.ground.metrics import (
    ANGULAR,
    COSINE,
    EUCLIDEAN,
    NORMALIZED_EUCLIDEAN,
    angular_distance,
    cdist,
    cosine_distance,
    declare_ground_distance,
    euclidean_distance,
    normalized_euclidean_distance,
    pdist,
)

from ..._budgets import examples
from ..._strategies import float_arrays, nonzero_rows
from ..._typing import ill_typed

if TYPE_CHECKING:
    from jax import Array
    from statadic.core.axioms import (
        Axioms,
        Law,
    )
    from statadic.geometry.ground.metrics import (
        GroundDistance,
    )


@dataclass(frozen=True, slots=True)
class _MetricOnlyLaw:
    """Honest custom metric law without Euclidean-specific capabilities."""

    axioms: Axioms = field(default=METRIC, init=False)
    properties: LawProperties = field(default=LawProperties(0), init=False)
    nonnegative: Literal[True] = field(default=True, init=False)
    zero_diagonal: Literal[True] = field(default=True, init=False)
    separates_points: Literal[True] = field(default=True, init=False)
    symmetric: Literal[True] = field(default=True, init=False)
    triangle_inequality: Literal[True] = field(default=True, init=False)


class TestAngularDistance:
    """Tests for angular_distance function."""

    @pytest.mark.unit
    @pytest.mark.property
    def test_identical_vectors(self) -> None:
        """Test angular distance between identical vectors is zero."""
        x = jnp.array([[1.0, 0.0, 0.0]])
        y = jnp.array([[2.0, 0.0, 0.0]])  # Same direction, different magnitude
        dist = angular_distance(x, y)
        assert jnp.allclose(dist, 0.0, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_orthogonal_vectors(self) -> None:
        """Test angular distance between orthogonal vectors is π/2."""
        x = jnp.array([[1.0, 0.0]])
        y = jnp.array([[0.0, 1.0]])
        dist = angular_distance(x, y)
        assert jnp.allclose(dist, jnp.pi / 2, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_opposite_vectors(self) -> None:
        """Test angular distance between opposite vectors is π."""
        x = jnp.array([[1.0, 0.0]])
        y = jnp.array([[-1.0, 0.0]])
        dist = angular_distance(x, y)
        assert jnp.allclose(dist, jnp.pi, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.contract
    def test_matrix_output_shape(self) -> None:
        """Test output shape is correct for multiple vectors."""
        x = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
        dist = angular_distance(x, y)
        assert dist.shape == (2, 3)

    @pytest.mark.unit
    @pytest.mark.property
    @examples(40)
    @example(x=jnp.array([[1.0, 2.0, 3.0]]), y=jnp.array([[4.0, 5.0, 6.0]]))
    @given(x=nonzero_rows(1, 3), y=nonzero_rows(1, 3))
    def test_symmetry(self, x: Array, y: Array) -> None:
        """Test angular distance is symmetric on nonzero rows."""
        dist_xy = angular_distance(x, y)
        dist_yx = angular_distance(y, x)
        assert jnp.allclose(dist_xy, dist_yx.T, atol=1e-6)


class TestCosineDistance:
    """Tests for cosine_distance function."""

    @pytest.mark.unit
    @pytest.mark.property
    def test_identical_vectors(self) -> None:
        """Test cosine distance between identical directions is zero."""
        x = jnp.array([[1.0, 0.0]])
        y = jnp.array([[2.0, 0.0]])  # Same direction, different magnitude
        dist = cosine_distance(x, y)
        assert jnp.allclose(dist, 0.0, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_orthogonal_vectors(self) -> None:
        """Test cosine distance between orthogonal vectors is 1."""
        x = jnp.array([[1.0, 0.0]])
        y = jnp.array([[0.0, 1.0]])
        dist = cosine_distance(x, y)
        assert jnp.allclose(dist, 1.0, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_opposite_vectors(self) -> None:
        """Test cosine distance between opposite vectors is 2."""
        x = jnp.array([[1.0, 0.0]])
        y = jnp.array([[-1.0, 0.0]])
        dist = cosine_distance(x, y)
        assert jnp.allclose(dist, 2.0, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.property
    @pytest.mark.slow
    def test_range_bounds(self) -> None:
        """Test cosine distance is in [0, 2]."""
        key = random.PRNGKey(42)
        x = random.normal(key, (10, 5))
        y = random.normal(random.PRNGKey(43), (10, 5))
        dist = cosine_distance(x, y)
        assert jnp.all(dist >= 0.0)
        maximum_cosine_distance = 2.0
        assert jnp.all(dist <= maximum_cosine_distance)


class TestNormalizedEuclideanDistance:
    """Tests for normalized_euclidean_distance function."""

    @pytest.mark.unit
    @pytest.mark.property
    def test_identical_vectors(self) -> None:
        """Test distance between identical vectors is zero."""
        x = jnp.array([[1.0, 2.0, 3.0]])
        y = jnp.array([[2.0, 4.0, 6.0]])  # Same direction, different magnitude
        dist = normalized_euclidean_distance(x, y)
        assert jnp.allclose(dist, 0.0, atol=1e-3)

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_orthogonal_vectors(self) -> None:
        """Test distance between orthogonal vectors is sqrt(2)."""
        x = jnp.array([[1.0, 0.0]])
        y = jnp.array([[0.0, 1.0]])
        dist = normalized_euclidean_distance(x, y)
        assert jnp.allclose(dist, jnp.sqrt(2.0), atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_opposite_vectors(self) -> None:
        """Test distance between opposite vectors is 2."""
        x = jnp.array([[1.0, 0.0, 0.0]])
        y = jnp.array([[-1.0, 0.0, 0.0]])
        dist = normalized_euclidean_distance(x, y)
        assert jnp.allclose(dist, 2.0, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.property
    @pytest.mark.slow
    def test_relationship_to_cosine(self) -> None:
        """Test relationship: normalized_euclidean = sqrt(2 * cosine_distance)."""
        key = random.PRNGKey(42)
        x = random.normal(key, (5, 10))
        y = random.normal(random.PRNGKey(43), (5, 10))
        norm_euclidean = normalized_euclidean_distance(x, y)
        cosine_dist = cosine_distance(x, y)

        expected = jnp.sqrt(2.0 * cosine_dist)
        assert jnp.allclose(norm_euclidean, expected, atol=1e-5)


class TestMetricDispatch:
    """Tests for cdist and pdist dispatch to declared strategies."""

    @pytest.mark.unit
    def test_angular_metric(self) -> None:
        """Test dispatch to angular_distance."""
        x = jnp.array([[1.0, 0.0]])
        y = jnp.array([[0.0, 1.0]])
        dist1 = cdist(x, y, metric=ANGULAR)
        dist2 = angular_distance(x, y)
        assert jnp.allclose(dist1, dist2)

    @pytest.mark.unit
    def test_cosine_metric(self) -> None:
        """Test dispatch to cosine_distance."""
        x = jnp.array([[1.0, 0.0]])
        y = jnp.array([[0.0, 1.0]])
        dist1 = cdist(x, y, metric=COSINE)
        dist2 = cosine_distance(x, y)
        assert jnp.allclose(dist1, dist2)

    @pytest.mark.unit
    def test_normalized_euclidean_metric(self) -> None:
        """Test dispatch to normalized_euclidean_distance."""
        x = jnp.array([[1.0, 0.0]])
        y = jnp.array([[0.0, 1.0]])
        dist1 = cdist(x, y, metric=NORMALIZED_EUCLIDEAN)
        dist2 = normalized_euclidean_distance(x, y)
        assert jnp.allclose(dist1, dist2)

    @pytest.mark.unit
    @pytest.mark.property
    def test_self_distance_when_y_none(self) -> None:
        """Test that pdist computes distances within x."""
        x = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        dist = pdist(x, metric=ANGULAR)
        assert dist.shape == (3, 3)
        # Diagonal should be zero (or very close due to numerical precision)
        assert jnp.allclose(jnp.diag(dist), 0.0, atol=1e-3)

    @pytest.mark.unit
    @pytest.mark.contract
    def test_invalid_metric(self) -> None:
        """Test error on invalid metric."""
        x = jnp.array([[1.0, 0.0]])
        with pytest.raises(TypeError):
            cdist(
                x,
                x,
                metric=ill_typed("invalid"),
            )


# --- the scipy.spatial.distance-shaped surface ----------------------------
#
# `cdist`/`pdist`/`euclidean_distance` exist to be
# drop-in-familiar to a scipy caller, so what is pinned below is the *shape* of
# that API — argument order, output shape, the symmetric/zero-diagonal
# invariants — rather than the axioms, which `tests/testing/test_axioms.py`
# measures. (t49 B2: these came from `test_scipy_api.py`, which named a
# borrowed convention rather than the module it covers.)


class TestCdist:
    """Tests for cdist function (scipy-like API)."""

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_euclidean_metric(self) -> None:
        """Test cdist with euclidean metric."""
        x = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        y = jnp.array([[0.0, 1.0], [1.0, 1.0]])

        result = cdist(x, y, metric=EUCLIDEAN)

        assert result.shape == (2, 2)
        assert jnp.allclose(result[0, 0], 1.0)  # dist((0,0), (0,1))
        assert jnp.allclose(result[0, 1], jnp.sqrt(2.0))  # dist((0,0), (1,1))
        assert jnp.allclose(result[1, 0], jnp.sqrt(2.0))  # dist((1,0), (0,1))
        assert jnp.allclose(result[1, 1], 1.0)  # dist((1,0), (1,1))

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_angular_metric(self) -> None:
        """Test cdist with angular metric."""
        x = jnp.array([[1.0, 0.0]])
        y = jnp.array([[0.0, 1.0]])

        result = cdist(x, y, metric=ANGULAR)

        assert result.shape == (1, 1)
        assert jnp.allclose(result[0, 0], jnp.pi / 2.0, atol=1e-5)  # 90 degrees

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_cosine_metric(self) -> None:
        """Test cdist with cosine metric."""
        x = jnp.array([[1.0, 0.0]])
        y = jnp.array([[0.0, 1.0]])

        result = cdist(x, y, metric=COSINE)

        assert result.shape == (1, 1)
        assert jnp.allclose(result[0, 0], 1.0)  # orthogonal vectors

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_normalized_euclidean_metric(self) -> None:
        """Test cdist with normalized_euclidean metric."""
        x = jnp.array([[3.0, 4.0]])  # magnitude 5
        y = jnp.array([[0.6, 0.8]])  # same direction, magnitude 1

        result = cdist(x, y, metric=NORMALIZED_EUCLIDEAN)

        assert result.shape == (1, 1)
        assert jnp.allclose(result[0, 0], 0.0, atol=1e-3)  # same direction

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_callable_metric(self) -> None:
        """Test cdist with custom callable metric."""

        def manhattan(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            """Manhattan (L1) distance."""
            return jnp.sum(jnp.abs(x[:, None, :] - y[None, :, :]), axis=-1)

        x = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        y = jnp.array([[0.0, 1.0], [1.0, 1.0]])

        declared = declare_ground_distance(
            manhattan,
            law=_MetricOnlyLaw(),
            name="manhattan",
            required_evidence=FiniteValues,
        )
        result = cdist(x, y, metric=declared)

        assert result.shape == (2, 2)
        assert jnp.allclose(result[0, 0], 1.0)  # |0-0| + |0-1| = 1
        assert jnp.allclose(result[0, 1], 2.0)  # |0-1| + |0-1| = 2
        assert jnp.allclose(result[1, 0], 2.0)  # |1-0| + |0-1| = 2
        assert jnp.allclose(result[1, 1], 1.0)  # |1-1| + |0-1| = 1

    @pytest.mark.unit
    @pytest.mark.contract
    def test_invalid_string_metric(self) -> None:
        """Test cdist with invalid string metric."""
        x = jnp.array([[1.0, 0.0]])
        y = jnp.array([[0.0, 1.0]])

        with pytest.raises(TypeError):
            cdist(
                x,
                y,
                metric=ill_typed("invalid_metric"),
            )

    @pytest.mark.unit
    @pytest.mark.contract
    def test_bare_callable_is_rejected(self) -> None:
        """Custom callables must cross the declaration factory."""

        def undeclared(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return euclidean_distance(x, y)

        points = jnp.array([[0.0], [1.0]])
        with pytest.raises(TypeError):
            cdist(
                points,
                points,
                metric=ill_typed(undeclared),
            )

    @pytest.mark.unit
    @pytest.mark.contract
    def test_invalid_metric_type(self) -> None:
        """Test cdist with invalid metric type."""
        x = jnp.array([[1.0, 0.0]])
        y = jnp.array([[0.0, 1.0]])

        with pytest.raises(TypeError):
            cdist(x, y, metric=ill_typed(123))

    @pytest.mark.unit
    @pytest.mark.property
    def test_high_dimensional(self) -> None:
        """Test cdist with high-dimensional data."""
        key = random.PRNGKey(42)
        x = random.normal(key, (10, 50))
        y = random.normal(random.PRNGKey(43), (15, 50))

        result = cdist(x, y, metric=EUCLIDEAN)

        assert result.shape == (10, 15)
        assert jnp.all(result >= 0)  # distances are non-negative


class TestPdist:
    """Tests for pdist function (scipy-like API)."""

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_euclidean_metric(self) -> None:
        """Test pdist with euclidean metric."""
        x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        result = pdist(x, metric=EUCLIDEAN)

        assert result.shape == (3, 3)
        # Diagonal should be zero
        assert jnp.allclose(jnp.diag(result), 0.0, atol=1e-6)
        # Should be symmetric
        assert jnp.allclose(result, result.T, atol=1e-6)
        # Check specific distances
        assert jnp.allclose(result[0, 1], 1.0)  # dist((0,0), (1,0))
        assert jnp.allclose(result[0, 2], 1.0)  # dist((0,0), (0,1))
        assert jnp.allclose(result[1, 2], jnp.sqrt(2.0))  # dist((1,0), (0,1))

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_callable_metric(self) -> None:
        """Test pdist with custom callable metric."""

        def chebyshev(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            """Chebyshev (L-infinity) distance."""
            return jnp.max(jnp.abs(x[:, None, :] - y[None, :, :]), axis=-1)

        x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])

        declared = declare_ground_distance(
            chebyshev,
            law=_MetricOnlyLaw(),
            name="chebyshev",
            required_evidence=FiniteValues,
        )
        result = pdist(x, metric=declared)

        assert result.shape == (3, 3)
        assert jnp.allclose(result[0, 1], 1.0)  # max(|0-1|, |0-0|) = 1
        assert jnp.allclose(result[0, 2], 2.0)  # max(|0-0|, |0-2|) = 2
        assert jnp.allclose(result[1, 2], 2.0)  # max(|1-0|, |0-2|) = 2


class TestEuclideanDistance:
    """Tests for euclidean_distance function."""

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_basic_distance(self) -> None:
        """Test basic Euclidean distance computation."""
        x = jnp.array([[0.0, 0.0]])
        y = jnp.array([[3.0, 4.0]])

        result = euclidean_distance(x, y)

        assert result.shape == (1, 1)
        assert jnp.allclose(result[0, 0], 5.0)  # 3-4-5 triangle

    @pytest.mark.unit
    @pytest.mark.property
    @examples(40)
    @example(x=jnp.array([[1.0, 2.0, 3.0]]))
    @given(x=float_arrays((1, 3)))
    def test_zero_distance(self, x: Array) -> None:
        """Test distance to self is zero."""
        result = euclidean_distance(x, x)

        assert jnp.allclose(result[0, 0], 0.0, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.property
    @examples(40)
    @example(
        x=random.normal(random.PRNGKey(42), (5, 10)),
        y=random.normal(random.PRNGKey(43), (7, 10)),
    )
    @given(x=float_arrays((5, 10)), y=float_arrays((7, 10)))
    def test_symmetry(self, x: Array, y: Array) -> None:
        """Test that distance is symmetric."""
        dxy = euclidean_distance(x, y)
        dyx = euclidean_distance(y, x)

        assert jnp.allclose(dxy, dyx.T, atol=1e-5)


class TestBackwardCompatibility:
    """Tests for the pdist/cdist convenience entry points."""

    @pytest.mark.unit
    @pytest.mark.regression
    def test_pdist_angular_diagonal(self) -> None:
        """Test pdist with the angular strategy."""
        x = jnp.array([[1.0, 0.0], [0.0, 1.0]])

        result = pdist(x, metric=ANGULAR)

        assert result.shape == (2, 2)
        assert jnp.allclose(jnp.diag(result), 0.0, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.regression
    def test_cdist_angular_pair(self) -> None:
        """Test cdist with two arguments."""
        x = jnp.array([[1.0, 0.0]])
        y = jnp.array([[0.0, 1.0]])

        result = cdist(x, y, metric=ANGULAR)

        assert result.shape == (1, 1)
        assert jnp.allclose(result[0, 0], jnp.pi / 2.0, atol=1e-5)


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(
    strategy=EUCLIDEAN,
    x=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
    y=jnp.array([[1.0, 1.0], [-1.0, 1.0]]),
)
@example(
    strategy=ANGULAR,
    x=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
    y=jnp.array([[1.0, 1.0], [-1.0, 1.0]]),
)
@example(
    strategy=COSINE,
    x=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
    y=jnp.array([[1.0, 1.0], [-1.0, 1.0]]),
)
@example(
    strategy=NORMALIZED_EUCLIDEAN,
    x=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
    y=jnp.array([[1.0, 1.0], [-1.0, 1.0]]),
)
@given(
    strategy=st.sampled_from([EUCLIDEAN, ANGULAR, COSINE, NORMALIZED_EUCLIDEAN]),
    x=nonzero_rows(2, 2),
    y=nonzero_rows(2, 2),
)
def test_builtin_strategy_is_static_and_transformable(
    strategy: GroundDistance[Law], x: jax.Array, y: jax.Array
) -> None:
    """Built-ins add no array leaves and preserve eager/JIT/vmap results."""
    assert jax.tree_util.tree_leaves(strategy) == []
    eager = cdist(x, y, metric=strategy)

    def evaluate_cdist(a: jax.Array, b: jax.Array) -> Array:
        return cdist(a, b, metric=strategy)

    jitted = jax.jit(evaluate_cdist)(x, y)

    def evaluate_cdist_against_y(a: jax.Array) -> Array:
        return cdist(a, y, metric=strategy)

    vmapped = jax.vmap(evaluate_cdist_against_y)(x[:, None, :])
    assert jnp.allclose(jitted, eager, equal_nan=True)
    assert jnp.allclose(vmapped, eager[:, None, :], equal_nan=True)
