# jaxtyping numerical carrier contracts.
"""Tests for distributional energy-balancing functions."""

from dataclasses import dataclass, field
from typing import Literal

import jax  # runtime annotations  # noqa: TC002
import jax.numpy as jnp
import pytest
from jax import config, random

# `Axioms`/`LawProperties` are runtime, not annotation-only: this module has no
# `from __future__ import annotations`, and both name dataclass *field* types
# (`axioms:`/`properties:` below), which are evaluated when the class body runs.
# A `TYPE_CHECKING` guard would raise `NameError` at collection.
from statadic.core.axioms import (
    METRIC,
    NEGATIVE_TYPE,
    Axioms,  # noqa: TC002
    LawProperties,  # noqa: TC002
)
from statadic.core.domains import (
    FiniteValues,
)
from statadic.discrepancy.mmd.energy.balancing import (
    energy_distance_kernel_mixture,
    energy_distance_kernel_pairwise,
    energy_test_kernel,
    energy_test_kernel_pairwise_max,
    inverse_variance_m_eff,
    kernel_balancing_weights,
)
from statadic.geometry.ground.metrics import declare_ground_distance
from statadic.sample.packed import PackedClouds, pack

# Enable 64-bit precision for tests
config.update("jax_enable_x64", val=True)

_DOMINANT_WEIGHT = 0.95
_NEGLIGIBLE_WEIGHT = 0.05
_MAJORITY_WEIGHT = 0.5
_SIMPLEX_DIAGNOSTIC_TOLERANCE = 1e-7
_SIMPLEX_SUM_TOLERANCE = 1e-12
_MIXTURE_ZERO_TOLERANCE = 0.01
_CLOSEST_WEIGHT = 0.8
_LARGE_DISTANCE = 10.0


@dataclass(frozen=True, slots=True)
class _ManhattanLaw:
    """Negative-type metric declaration without Euclidean-only capabilities."""

    axioms: Axioms = field(default=METRIC, init=False)
    properties: LawProperties = field(default=NEGATIVE_TYPE, init=False)
    nonnegative: Literal[True] = field(default=True, init=False)
    zero_diagonal: Literal[True] = field(default=True, init=False)
    separates_points: Literal[True] = field(default=True, init=False)
    symmetric: Literal[True] = field(default=True, init=False)
    triangle_inequality: Literal[True] = field(default=True, init=False)
    negative_type: Literal[True] = field(default=True, init=False)


class TestEnergyDistanceKernel:
    """Test energy distance kernel optimization.

    The QP-solving methods are individually marked ``slow``; the cheap argument-
    validation tests (``*_raises``) stay in the default gate.
    """

    @pytest.mark.component
    @pytest.mark.numerical
    @pytest.mark.slow
    def test_single_candidate_gets_full_weight(self) -> None:
        """Test that single candidate receives weight 1.0."""
        target = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        candidate = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        candidates = candidate[None, :, :]  # Shape (1, 2, 2)

        weights = kernel_balancing_weights(target, candidates)

        assert weights.shape == (1,)
        assert jnp.allclose(weights[0], 1.0)

    @pytest.mark.component
    @pytest.mark.property
    @pytest.mark.slow
    def test_weights_sum_to_one(self) -> None:
        """Test that weights sum to 1."""
        target = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand1 = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand2 = jnp.array([[10.0, 10.0], [11.0, 11.0]])
        candidates = jnp.stack([cand1, cand2])

        weights = kernel_balancing_weights(target, candidates)

        assert jnp.allclose(jnp.sum(weights), 1.0)

    @pytest.mark.component
    @pytest.mark.property
    @pytest.mark.slow
    def test_weights_in_valid_range(self) -> None:
        """Test that weights are in [0, 1]."""
        target = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand1 = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand2 = jnp.array([[5.0, 5.0], [6.0, 6.0]])
        cand3 = jnp.array([[10.0, 10.0], [11.0, 11.0]])
        candidates = jnp.stack([cand1, cand2, cand3])

        weights = kernel_balancing_weights(target, candidates)

        assert jnp.all(weights >= 0.0)
        assert jnp.all(weights <= 1.0)

    @pytest.mark.component
    @pytest.mark.property
    @pytest.mark.slow
    def test_perfect_match_gets_full_weight(self) -> None:
        """Test that perfect match gets nearly all weight."""
        target = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        perfect = jnp.array([[0.0, 0.0], [1.0, 1.0]])  # Same as target
        far = jnp.array([[100.0, 100.0], [101.0, 101.0]])
        candidates = jnp.stack([perfect, far])

        weights = kernel_balancing_weights(target, candidates)

        # Perfect match should get most/all weight
        assert weights[0] > _DOMINANT_WEIGHT
        assert weights[1] < _NEGLIGIBLE_WEIGHT

    @pytest.mark.component
    @pytest.mark.property
    @pytest.mark.slow
    def test_symmetric_mixture(self) -> None:
        """Test that symmetric setup gives equal weights."""
        # Target at origin
        target = jnp.array([[0.0, 0.0], [0.1, 0.1]])
        # Two candidates equidistant from target
        cand1 = jnp.array([[1.0, 0.0], [1.1, 0.1]])
        cand2 = jnp.array([[-1.0, 0.0], [-1.1, 0.1]])
        candidates = jnp.stack([cand1, cand2])

        weights = kernel_balancing_weights(target, candidates, metric="euclidean")

        # Should get approximately equal weights due to symmetry
        assert jnp.allclose(weights[0], weights[1], atol=0.1)

    @pytest.mark.component
    @pytest.mark.property
    @pytest.mark.slow
    def test_different_metrics(self) -> None:
        """Test kernel with different distance metrics."""
        target = jnp.array([[0.1, 0.0], [1.0, 1.0]])
        cand1 = jnp.array([[0.1, 0.0], [1.0, 1.0]])
        cand2 = jnp.array([[10.0, 10.0], [11.0, 11.0]])
        candidates = jnp.stack([cand1, cand2])

        metrics: tuple[Literal["euclidean", "angular", "cosine"], ...] = (
            "euclidean",
            "angular",
            "cosine",
        )
        for metric in metrics:
            weights = kernel_balancing_weights(target, candidates, metric=metric)
            assert jnp.allclose(jnp.sum(weights), 1.0)
            assert jnp.all(weights >= 0.0)

    @pytest.mark.component
    @pytest.mark.property
    @pytest.mark.slow
    def test_different_exponents(self) -> None:
        """Test kernel with different distance exponents."""
        target = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand1 = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand2 = jnp.array([[10.0, 10.0], [11.0, 11.0]])
        candidates = jnp.stack([cand1, cand2])

        for exp in [0.5, 1.0, 1.5]:
            weights = kernel_balancing_weights(target, candidates, exponent=exp)
            assert jnp.allclose(jnp.sum(weights), 1.0)
            assert jnp.all(weights >= 0.0)

    @pytest.mark.unit
    @pytest.mark.contract
    def test_invalid_exponent_raises(self) -> None:
        """Test that invalid exponent raises ValueError."""
        target = jnp.array([[0.0, 0.0]])
        candidates = jnp.array([[[1.0, 1.0]]])

        with pytest.raises(ValueError, match="exponent must be in"):
            kernel_balancing_weights(target, candidates, exponent=0.0)

        with pytest.raises(ValueError, match="exponent must be in"):
            kernel_balancing_weights(target, candidates, exponent=2.0)

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.parametrize("solver", ["invalid", "boxosqp", "eqcp"])
    def test_invalid_solver_raises(self, solver: str) -> None:
        """Test that invalid solver raises ValueError."""
        target = jnp.array([[0.0, 0.0]])
        candidates = jnp.array([[[1.0, 1.0]]])

        with pytest.raises(ValueError, match="Unknown solver"):
            kernel_balancing_weights(target, candidates, solver=solver)

    @pytest.mark.component
    @pytest.mark.property
    @pytest.mark.slow
    def test_many_candidates(self) -> None:
        """Test kernel with many candidates."""
        target = jnp.array([[0.0, 0.0], [1.0, 1.0]])

        # Create 10 candidates spread around target
        candidates_list = []
        for i in range(10):
            offset = i * 1.0
            candidates_list.append(target + offset)
        candidates = jnp.stack(candidates_list)

        weights = kernel_balancing_weights(target, candidates)

        assert weights.shape == (10,)
        assert jnp.allclose(jnp.sum(weights), 1.0)
        assert jnp.all(weights >= 0.0)
        # Closest candidate (offset=0) should get most weight
        assert weights[0] > _MAJORITY_WEIGHT


@pytest.mark.slow
class TestEnergyDistanceKernelPairwise:
    """Test pairwise energy distance kernel."""

    @pytest.mark.component
    @pytest.mark.contract
    def test_pairwise_shape(self) -> None:
        """Test that pairwise kernel returns correct shape."""
        dist1 = jnp.array([[0.0, 0.0], [0.1, 0.1]])
        dist2 = jnp.array([[1.0, 1.0], [1.1, 1.1]])
        dist3 = jnp.array([[10.0, 10.0], [10.1, 10.1]])
        distributions = jnp.stack([dist1, dist2, dist3])

        weights = energy_distance_kernel_pairwise(distributions)

        # Each distribution gets weights for kernel_matrix-1 others
        assert weights.shape == (3, 2)

    @pytest.mark.component
    @pytest.mark.property
    def test_pairwise_weights_valid(self) -> None:
        """Test that pairwise weights are valid."""
        dist1 = jnp.array([[0.0, 0.0], [0.1, 0.1]])
        dist2 = jnp.array([[1.0, 1.0], [1.1, 1.1]])
        dist3 = jnp.array([[10.0, 10.0], [10.1, 10.1]])
        distributions = jnp.stack([dist1, dist2, dist3])

        weights = energy_distance_kernel_pairwise(distributions)

        # Each row should sum to 1
        assert jnp.allclose(jnp.sum(weights, axis=1), 1.0)
        # All weights should be in [0, 1]
        assert jnp.all(weights >= 0.0)
        assert jnp.all(weights <= 1.0)

    @pytest.mark.component
    @pytest.mark.numerical
    def test_pairwise_two_distributions(self) -> None:
        """Test pairwise with two distributions."""
        dist1 = jnp.array([[0.0, 0.0], [0.1, 0.1]])
        dist2 = jnp.array([[10.0, 10.0], [10.1, 10.1]])
        distributions = jnp.stack([dist1, dist2])

        weights = energy_distance_kernel_pairwise(distributions)

        # Each gets weight 1.0 for the other
        assert weights.shape == (2, 1)
        assert jnp.allclose(weights, 1.0)


class TestEnergyDistanceKernelMixture:
    """Test energy distance to mixture evaluation."""

    @pytest.mark.unit
    @pytest.mark.property
    def test_mixture_perfect_match(self) -> None:
        """Test that perfect match has zero distance."""
        target = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        perfect = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        candidates = perfect[None, :, :]
        weights = jnp.array([1.0])

        distance = energy_distance_kernel_mixture(
            PackedClouds.from_array(jnp.asarray(target)),
            pack(candidates, require_positive_counts=True),
            weights,
            metric="euclidean",
        )

        # Should be very close to zero
        assert jnp.abs(distance) < _MIXTURE_ZERO_TOLERANCE

    @pytest.mark.unit
    @pytest.mark.property
    def test_mixture_increases_with_distance(self) -> None:
        """Test that mixture distance increases with target-candidate distance."""
        target = jnp.array([[0.0, 0.0], [1.0, 1.0]])

        # Two candidates at different distances
        close = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        far = jnp.array([[10.0, 10.0], [11.0, 11.0]])

        candidates_close = close[None, :, :]
        candidates_far = far[None, :, :]
        weights = jnp.array([1.0])

        dist_close = energy_distance_kernel_mixture(
            PackedClouds.from_array(jnp.asarray(target)),
            pack(candidates_close, require_positive_counts=True),
            weights,
            metric="euclidean",
        )
        dist_far = energy_distance_kernel_mixture(
            PackedClouds.from_array(jnp.asarray(target)),
            pack(candidates_far, require_positive_counts=True),
            weights,
            metric="euclidean",
        )

        assert dist_close < dist_far

    @pytest.mark.unit
    @pytest.mark.property
    def test_mixture_equal_weights(self) -> None:
        """Test mixture with equal weights."""
        target = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand1 = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand2 = jnp.array([[10.0, 10.0], [11.0, 11.0]])
        candidates = jnp.stack([cand1, cand2])
        weights = jnp.array([0.5, 0.5])

        distance = energy_distance_kernel_mixture(
            PackedClouds.from_array(jnp.asarray(target)),
            pack(candidates, require_positive_counts=True),
            weights,
            metric="euclidean",
        )

        # Should be positive (mixture doesn't perfectly match target)
        assert distance > 0.0

    @pytest.mark.unit
    @pytest.mark.contract
    def test_mixture_invalid_exponent_raises(self) -> None:
        """Test that invalid exponent raises ValueError."""
        target = jnp.array([[0.0, 0.0]])
        candidates = jnp.array([[[1.0, 1.0]]])
        weights = jnp.array([1.0])

        with pytest.raises(ValueError, match="exponent must be in"):
            energy_distance_kernel_mixture(
                PackedClouds.from_array(jnp.asarray(target)),
                pack(candidates, require_positive_counts=True),
                weights,
                exponent=0.0,
            )

    @pytest.mark.integration
    @pytest.mark.property
    def test_mixture_consistency_with_kernel(self) -> None:
        """Test that optimal weights minimize mixture distance."""
        target = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand1 = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand2 = jnp.array([[10.0, 10.0], [11.0, 11.0]])
        candidates = jnp.stack([cand1, cand2])

        # Get optimal weights
        optimal_weights = kernel_balancing_weights(
            target, candidates, metric="euclidean"
        )

        # Get distance with optimal weights
        dist_optimal = energy_distance_kernel_mixture(
            PackedClouds.from_array(jnp.asarray(target)),
            pack(candidates, require_positive_counts=True),
            optimal_weights,
            metric="euclidean",
        )

        # Try suboptimal weights (equal weights)
        equal_weights = jnp.array([0.5, 0.5])
        dist_equal = energy_distance_kernel_mixture(
            PackedClouds.from_array(jnp.asarray(target)),
            pack(candidates, require_positive_counts=True),
            equal_weights,
            metric="euclidean",
        )

        # Optimal should be better (lower or equal distance)
        assert dist_optimal <= dist_equal + 1e-3  # Small tolerance for numerical error


class TestCustomMetrics:
    """Test kernels with custom distance metrics."""

    @pytest.mark.component
    @pytest.mark.property
    def test_kernel_with_callable_metric(self) -> None:
        """Test kernel with custom callable metric."""

        def manhattan(x: jax.Array, y: jax.Array) -> jax.Array:
            """Manhattan distance."""
            return jnp.sum(jnp.abs(x[:, None, :] - y[None, :, :]), axis=-1)

        target = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand1 = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand2 = jnp.array([[10.0, 10.0], [11.0, 11.0]])
        candidates = jnp.stack([cand1, cand2])

        declared = declare_ground_distance(
            manhattan,
            law=_ManhattanLaw(),
            name="manhattan",
            required_evidence=FiniteValues,
        )
        weights = kernel_balancing_weights(target, candidates, metric=declared)

        assert jnp.allclose(jnp.sum(weights), 1.0)
        assert jnp.all(weights >= 0.0)
        # Perfect match should get most weight
        assert weights[0] > _DOMINANT_WEIGHT


@pytest.mark.slow
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.component
    @pytest.mark.property
    def test_identical_candidates(self) -> None:
        """Test kernel with identical candidates."""
        target = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand = jnp.array([[5.0, 5.0], [6.0, 6.0]])
        # Two identical candidates
        candidates = jnp.stack([cand, cand])

        weights = kernel_balancing_weights(target, candidates)

        # Should get equal weights since candidates are identical
        assert jnp.allclose(weights[0], weights[1], atol=_MIXTURE_ZERO_TOLERANCE)
        assert jnp.allclose(jnp.sum(weights), 1.0)

    @pytest.mark.component
    @pytest.mark.property
    def test_high_dimensional(self) -> None:
        """Test kernel in higher dimensions."""
        # 10-dimensional space
        target = jnp.ones((5, 10)) * 0.0
        cand1 = jnp.ones((5, 10)) * 0.0
        cand2 = jnp.ones((5, 10)) * 10.0
        candidates = jnp.stack([cand1, cand2])

        weights = kernel_balancing_weights(target, candidates, metric="euclidean")

        assert jnp.allclose(jnp.sum(weights), 1.0)
        assert weights[0] > _DOMINANT_WEIGHT  # Perfect match gets most weight

    @pytest.mark.component
    @pytest.mark.property
    def test_single_sample_distributions(self) -> None:
        """Test kernel with single-sample distributions."""
        target = jnp.array([[0.0, 0.0]])
        cand1 = jnp.array([[0.0, 0.0]])
        cand2 = jnp.array([[10.0, 10.0]])
        candidates = jnp.stack([cand1, cand2])

        weights = kernel_balancing_weights(target, candidates, metric="euclidean")

        assert jnp.allclose(jnp.sum(weights), 1.0)
        assert weights[0] > _DOMINANT_WEIGHT


@pytest.mark.slow
class TestIrregularSizes:
    """Test kernels with irregular-sized distributions (list API)."""

    @pytest.mark.component
    @pytest.mark.property
    def test_irregular_sizes_basic(self) -> None:
        """Test that irregular-sized distributions work correctly."""
        target = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand1 = jnp.array([[0.0, 0.0], [1.0, 1.0]])  # 2 samples
        cand2 = jnp.array([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]])  # 3 samples
        candidates = [cand1, cand2]  # List instead of stacked array

        weights = kernel_balancing_weights(target, candidates)

        assert weights.shape == (2,)
        assert jnp.allclose(jnp.sum(weights), 1.0)
        assert weights[0] > _DOMINANT_WEIGHT  # Perfect match gets most weight

    @pytest.mark.component
    @pytest.mark.property
    def test_irregular_sizes_many_candidates(self) -> None:
        """Test with many candidates of different sizes."""
        target = jnp.array([[0.0, 0.0], [0.1, 0.1]])
        cand1 = jnp.array([[0.0, 0.0], [0.1, 0.1]])  # 2 samples
        cand2 = jnp.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]])  # 3 samples
        cand3 = jnp.array([[5.0, 5.0]])  # 1 sample
        cand4 = jnp.array(
            [
                [10.0, 10.0],
                [10.1, 10.1],
                [10.2, 10.2],
                [10.3, 10.3],
            ]
        )  # 4 samples
        candidates = [cand1, cand2, cand3, cand4]

        weights = kernel_balancing_weights(target, candidates)

        assert weights.shape == (4,)
        assert jnp.allclose(jnp.sum(weights), 1.0)
        assert jnp.all(weights >= 0.0)
        assert jnp.all(weights <= 1.0)
        assert weights[0] > _CLOSEST_WEIGHT  # First candidate is closest

    @pytest.mark.component
    @pytest.mark.property
    def test_irregular_sizes_pairwise(self) -> None:
        """Test pairwise kernel with irregular sizes."""
        dist1 = jnp.array([[0.0, 0.0], [0.1, 0.1]])  # 2 samples
        dist2 = jnp.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]])  # 3 samples
        dist3 = jnp.array([[10.0, 10.0]])  # 1 sample
        distributions = [dist1, dist2, dist3]

        weights = energy_distance_kernel_pairwise(distributions)

        assert weights.shape == (3, 2)
        # Check that all rows sum to 1
        assert jnp.allclose(jnp.sum(weights, axis=1), jnp.ones(3))

    @pytest.mark.component
    @pytest.mark.property
    def test_irregular_sizes_mixture(self) -> None:
        """Test mixture evaluation with irregular sizes."""
        target = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand1 = jnp.array([[0.0, 0.0], [1.0, 1.0]])  # 2 samples
        cand2 = jnp.array([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]])  # 3 samples
        candidates = [cand1, cand2]

        # Perfect match weight
        weights = jnp.array([1.0, 0.0])
        dist = energy_distance_kernel_mixture(
            PackedClouds.from_array(jnp.asarray(target)),
            pack(candidates, require_positive_counts=True),
            weights,
        )
        assert jnp.isclose(dist, 0.0, atol=1e-6)

        # All weight on far candidate
        weights = jnp.array([0.0, 1.0])
        dist = energy_distance_kernel_mixture(
            PackedClouds.from_array(jnp.asarray(target)),
            pack(candidates, require_positive_counts=True),
            weights,
        )
        assert dist > _LARGE_DISTANCE  # Should be large

    @pytest.mark.component
    @pytest.mark.regression
    def test_irregular_backward_compatible(self) -> None:
        """Test that stacked arrays still work (backward compatibility)."""
        target = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand1 = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        cand2 = jnp.array([[10.0, 10.0], [11.0, 11.0]])

        # Test with stacked array (old API)
        candidates_stacked = jnp.stack([cand1, cand2])
        weights_stacked = kernel_balancing_weights(target, candidates_stacked)

        # Test with list (new API)
        candidates_list = [cand1, cand2]
        weights_list = kernel_balancing_weights(target, candidates_list)

        # Should get same results
        assert jnp.allclose(weights_stacked, weights_list)


class TestInverseVarianceMEff:
    """Pure-math tests for the inverse-variance (Kish) effective size.

    No QP solves — these stay in the default gate as fast unit tests of the
    ``inverse_variance_m_eff`` closed form.
    """

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_equal_groups_pools(self) -> None:
        """Verify that equal groups at equal weights pool their effective size.

        This is the worked counterexample to w^T m (which returns just m).
        """
        candidate_count, m = 4, 25.0
        w = jnp.ones(candidate_count) / candidate_count
        m_sizes = jnp.full((candidate_count,), m)
        m_eff = inverse_variance_m_eff(w, m_sizes)
        assert jnp.allclose(
            m_eff, candidate_count * m, rtol=1e-4
        )  # = 100, not 25 (= w^T m)
        assert jnp.allclose(jnp.dot(w, m_sizes), m)  # legacy w^T m = 25

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_vertex_agrees(self) -> None:
        """At a simplex vertex both m_eff definitions agree (= m_k)."""
        m_sizes = jnp.array([10.0, 100.0, 40.0])
        w = jnp.array([0.0, 1.0, 0.0])
        assert jnp.allclose(inverse_variance_m_eff(w, m_sizes), 100.0)
        assert jnp.allclose(jnp.dot(w, m_sizes), 100.0)

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_heteroscedastic(self) -> None:
        """Heteroscedastic inverse-variance form uses group variances s_k²."""
        w = jnp.array([0.5, 0.5])
        m_sizes = jnp.array([50.0, 50.0])
        s2 = jnp.array([1.0, 4.0])
        m_eff = inverse_variance_m_eff(w, m_sizes, group_variances=s2)
        # Closed form: 1/m_eff = (Σ w²s²/m)/(Σ w s²)
        inv = (0.25 * 1.0 / 50 + 0.25 * 4.0 / 50) / (0.5 * 1.0 + 0.5 * 4.0)
        assert jnp.allclose(m_eff, 1.0 / inv, rtol=1e-4)


@pytest.mark.slow
class TestEnergyTestKernel:
    """Test energy test statistic kernel optimization."""

    @pytest.mark.component
    @pytest.mark.numerical
    def test_single_candidate_full_weight(self) -> None:
        """Single candidate should get full weight."""
        x = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        y = jnp.array([[5.0, 5.0], [6.0, 6.0]])
        candidates = [y]

        weights, stat = energy_test_kernel(x, candidates, metric="euclidean")

        assert jnp.allclose(weights, jnp.array([1.0]))
        assert stat > 0.0

    @pytest.mark.component
    @pytest.mark.property
    def test_chooses_most_different(self) -> None:
        """Should put more weight on most different candidate."""
        key = random.PRNGKey(42)
        x = random.normal(key, (50, 2))

        # Candidate 1: similar to x
        cand1 = random.normal(random.PRNGKey(1), (50, 2))

        # Candidate 2: very different from x
        cand2 = random.normal(random.PRNGKey(2), (50, 2)) + jnp.array([10.0, 10.0])

        candidates = [cand1, cand2]
        weights, stat = energy_test_kernel(x, candidates, metric="euclidean")

        # Should put more weight on cand2 (more different)
        assert weights[1] > weights[0]
        assert stat > 0.0

    @pytest.mark.component
    @pytest.mark.property
    def test_weights_sum_to_one(self) -> None:
        """Weights should sum to one."""
        key = random.PRNGKey(42)
        x = random.normal(key, (30, 2))

        candidates = [
            random.normal(random.PRNGKey(i), (30, 2)) + jnp.array([i * 2.0, 0.0])
            for i in range(3)
        ]

        weights, _ = energy_test_kernel(x, candidates, metric="euclidean")

        assert jnp.allclose(jnp.sum(weights), 1.0)

    @pytest.mark.component
    @pytest.mark.property
    def test_irregular_sizes(self) -> None:
        """Should handle irregular-sized candidates."""
        key = random.PRNGKey(42)
        x = random.normal(key, (50, 2))

        candidates = [
            random.normal(random.PRNGKey(1), (30, 2)),
            random.normal(random.PRNGKey(2), (60, 2)),
            random.normal(random.PRNGKey(3), (40, 2)),
        ]

        weights, stat = energy_test_kernel(x, candidates, metric="euclidean")

        assert weights.shape == (3,)
        assert jnp.allclose(jnp.sum(weights), 1.0)
        assert jnp.isfinite(stat)

    @pytest.mark.component
    @pytest.mark.regression
    def test_backward_compatible(self) -> None:
        """Should work with stacked arrays."""
        key = random.PRNGKey(42)
        x = random.normal(key, (50, 2))

        # Stacked array format
        candidates = jnp.stack(
            [
                random.normal(random.PRNGKey(1), (30, 2)),
                random.normal(random.PRNGKey(2), (30, 2)),
            ]
        )

        weights, _stat = energy_test_kernel(x, candidates, metric="euclidean")

        assert weights.shape == (2,)
        assert jnp.allclose(jnp.sum(weights), 1.0)

    @pytest.mark.component
    @pytest.mark.property
    def test_effective_sample_size_inverse_variance(self) -> None:
        """Corrected m_eff is the inverse-variance (Kish) size, not w^T m.

        The old code used m_eff = w^T m, which is refuted (over-optimistic; only
        correct at a simplex vertex). The default is now the inverse-variance
        effective size 1/m_eff = Σ_k w_k²/m_k, which can EXCEED max(m_k) when
        weight is spread across candidates.
        """
        key = random.PRNGKey(42)
        x = random.normal(key, (50, 2))

        # Create candidates with very different sizes
        cand1 = random.normal(random.PRNGKey(1), (10, 2)) + jnp.array([5.0, 0.0])
        cand2 = random.normal(random.PRNGKey(2), (100, 2)) + jnp.array([5.0, 0.0])

        candidates = [cand1, cand2]
        weights, stat = energy_test_kernel(x, candidates, metric="euclidean")

        # Verify weights account for sample sizes
        assert weights[1] > weights[0]  # More weight on larger candidate

        # The default (inverse-variance) m_eff differs from the legacy w^T m.
        m_sizes = jnp.array([len(cand1), len(cand2)], dtype=jnp.float32)
        m_eff_iv = inverse_variance_m_eff(weights, m_sizes)
        m_eff_linear = jnp.dot(weights, m_sizes)

        # Inverse-variance m_eff is >= the linear one whenever weight is spread,
        # and equals it only at a vertex.
        assert m_eff_iv >= m_eff_linear - 1e-3
        # Verify stat is finite and positive
        assert jnp.isfinite(stat)
        assert stat > 0.0

    @pytest.mark.component
    @pytest.mark.numerical
    def test_different_exponents(self) -> None:
        """Test with different distance exponents."""
        key = random.PRNGKey(42)
        x = random.normal(key, (30, 2))
        y = random.normal(random.PRNGKey(1), (30, 2)) + jnp.array([2.0, 2.0])
        candidates = [y]

        # Test different exponents
        for exp in [0.5, 1.0, 1.5]:
            weights, stat = energy_test_kernel(
                x, candidates, exponent=exp, metric="euclidean"
            )
            assert jnp.allclose(weights, jnp.array([1.0]))
            assert stat > 0.0

    @pytest.mark.unit
    @pytest.mark.contract
    def test_invalid_exponent_raises(self) -> None:
        """Test that invalid exponent raises error."""
        x = jnp.array([[0.0, 0.0]])
        y = jnp.array([[1.0, 1.0]])
        candidates = [y]

        with pytest.raises(ValueError, match="exponent must be in"):
            energy_test_kernel(x, candidates, exponent=2.5)


@pytest.mark.slow
class TestEnergyTestKernelPairwiseMax:
    """Test pairwise maximum energy test statistic."""

    @pytest.mark.component
    @pytest.mark.contract
    @pytest.mark.smoke
    def test_outlier_detection(self) -> None:
        """Return finite statistics and correctly shaped outputs for all groups."""
        # Two similar distributions
        dist1 = random.normal(random.PRNGKey(0), (50, 2))
        dist2 = random.normal(random.PRNGKey(1), (50, 2)) + jnp.array([0.1, 0.1])

        # One outlier - much farther away
        dist3 = random.normal(random.PRNGKey(2), (50, 2)) + jnp.array([20.0, 20.0])

        distributions = [dist1, dist2, dist3]
        weights, stats = energy_test_kernel_pairwise_max(
            distributions, metric="euclidean"
        )

        # All test statistics should be positive and finite
        assert jnp.all(stats > 0)
        assert jnp.all(jnp.isfinite(stats))

        assert stats.shape == (3,)
        assert weights.shape == (3, 2)

    @pytest.mark.component
    @pytest.mark.contract
    def test_pairwise_shapes(self) -> None:
        """Test output shapes are correct."""
        distributions = [random.normal(random.PRNGKey(i), (30, 2)) for i in range(4)]

        weights, stats = energy_test_kernel_pairwise_max(
            distributions, metric="euclidean"
        )

        assert weights.shape == (4, 3)  # Each vs 3 others
        assert stats.shape == (4,)
        assert jnp.all(jnp.isfinite(stats))

    @pytest.mark.component
    @pytest.mark.property
    def test_irregular_sizes_pairwise(self) -> None:
        """Should handle irregular sizes in pairwise comparison."""
        distributions = [
            random.normal(random.PRNGKey(1), (30, 2)),
            random.normal(random.PRNGKey(2), (50, 2)),
            random.normal(random.PRNGKey(3), (40, 2)),
        ]

        weights, stats = energy_test_kernel_pairwise_max(
            distributions, metric="euclidean"
        )

        assert weights.shape == (3, 2)
        assert stats.shape == (3,)

        # Each row should sum to 1
        for i in range(3):
            assert jnp.allclose(jnp.sum(weights[i]), 1.0)

    @pytest.mark.component
    @pytest.mark.property
    def test_symmetric_distributions_similar_stats(self) -> None:
        """Symmetric distributions should have similar test statistics."""
        # Three similar distributions
        distributions = [random.normal(random.PRNGKey(i), (50, 2)) for i in range(3)]

        _weights, stats = energy_test_kernel_pairwise_max(
            distributions, metric="euclidean"
        )

        # Statistics should be similar (no clear outlier)
        assert jnp.std(stats) < 0.5 * jnp.mean(stats)

    @pytest.mark.component
    @pytest.mark.property
    def test_effective_sample_sizes_across_distributions(self) -> None:
        """Test that effective sample sizes are properly computed for all pairs."""
        # Create distributions with varied sizes
        distributions = [
            random.normal(random.PRNGKey(1), (20, 2)),
            random.normal(random.PRNGKey(2), (50, 2)),
            random.normal(random.PRNGKey(3), (100, 2)),
        ]

        weights, stats = energy_test_kernel_pairwise_max(
            distributions, metric="euclidean"
        )

        # All statistics should be positive and finite
        assert jnp.all(stats > 0)
        assert jnp.all(jnp.isfinite(stats))

        # Verify each row of weights sums to 1
        row_sums = jnp.sum(weights, axis=1)
        assert jnp.allclose(row_sums, jnp.ones(3))
