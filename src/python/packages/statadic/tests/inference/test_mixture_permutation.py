# jaxtyping shape strings are runtime contracts.
"""Permutation resampling and selected-mixture weight routes.

Two blocks:

* **the engine** — a permutation test over an arbitrary statistic. The two
  statistics below (difference of means, difference of variances) are
  deliberately trivial: what is under test is the resampling, the p-value
  convention and the returned container, not the statistic.
* **re-optimization (t05 correction 3)** — whether ``w*`` is re-solved inside
  each permutation replicate or plugged in from the observed sample. Plugging in
  is a look-ahead, so it is disallowed rather than merely discouraged, and the
  exact/split routes must agree once both re-optimize.

The bootstrap CI and the studentized statistic moved to their own modules
(``test_bootstrap.py``, ``test_studentized.py``) in t49 B2, mirroring the
sub-modules of ``statadic/inference/`` they exercise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from jax import random
from statadic.inference.mixture import (
    corrected_mixture_test,
)
from statadic.inference.resampling.results import (
    HypothesisTest,
)

from .._budgets import examples
from .._permutation import host_permutation_test
from .._strategies import float_arrays
from ._hedge import hedge_samples, original_hedge_data

if TYPE_CHECKING:
    from jax import Array

    from ._hedge import HedgeData

_SIGNIFICANCE_LEVEL = 0.05
_SMOKE_PERMUTATIONS = 19
_STANDARD_PERMUTATIONS = 99
_WEAK_SIGNIFICANCE_LEVEL = 0.1
_SEEDS = st.integers(0, 2**31 - 1)


def _refinement_permutation_data() -> HedgeData:
    """Retain the exact full-sample fixture whose second draw needs refinement."""
    target = jnp.full((60, 4), 3.0)
    first = jnp.full((30, 4), 2.0).at[jnp.array([0, 9]), 0].set(3.0)
    second = jnp.full((50, 4), 9.0).at[7, 0].set(10.0)
    return target, [first, second]


def simple_mean_difference(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute a simple difference-of-means statistic.

    Args:
        x: First sample, shape (n, d).
        y: Second sample, shape (m, d).

    Returns:
        Absolute difference of means across samples.

    """
    return jnp.abs(jnp.mean(x) - jnp.mean(y))


def variance_difference(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Test statistic: difference in variance.

    Args:
        x: First sample, shape (n, d).
        y: Second sample, shape (m, d).

    Returns:
        Absolute difference of variances.

    """
    return jnp.abs(jnp.var(x) - jnp.var(y))


class TestPermutationTest:
    """Tests for the permutation test result and its host report."""

    pytestmark = pytest.mark.slow

    @pytest.mark.component
    @pytest.mark.property
    @examples(10)
    @example(x=random.normal(random.PRNGKey(42), (20, 3)), seed=42)
    @given(x=float_arrays((20, 3)), seed=_SEEDS)
    def test_identical_distributions(self, x: Array, seed: int) -> None:
        """Test with identical distributions should give high p-value.

        Args:
            x: Generated sample, used as both groups so the observed statistic is zero.
            seed: Permutation seed.

        """
        y = x

        result = host_permutation_test(
            x, y, simple_mean_difference, num_permutations=19, seed=seed
        )

        # With identical samples, every relabelling remains consistent with H0.
        assert result.pvalue > _SIGNIFICANCE_LEVEL

    @pytest.mark.component
    def test_different_distributions(self) -> None:
        """Test with different distributions should give low p-value."""
        key = random.PRNGKey(42)
        x = random.normal(key, (50, 3))
        y = random.normal(random.PRNGKey(43), (50, 3)) + 5.0  # Shifted mean

        result = host_permutation_test(
            x, y, simple_mean_difference, num_permutations=99, seed=42
        )

        # With clearly different distributions, p-value should be very low
        assert result.pvalue < _SIGNIFICANCE_LEVEL

    @pytest.mark.component
    @pytest.mark.contract
    @examples(10)
    @example(
        x=jnp.array([[1.0, 2.0], [3.0, 4.0]]), y=jnp.array([[5.0, 6.0], [7.0, 8.0]])
    )
    @given(x=float_arrays((2, 2)), y=float_arrays((2, 2)))
    def test_result_structure(self, x: Array, y: Array) -> None:
        """Test that result has correct structure.

        Args:
            x: Generated first sample.
            y: Generated second sample.

        """
        result = host_permutation_test(
            x, y, simple_mean_difference, num_permutations=19
        )

        # Check result is HypothesisTest
        assert isinstance(result, HypothesisTest)
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")
        assert hasattr(result, "num_permutations")
        assert hasattr(result, "null_distribution")

        # Check properties
        assert result.num_permutations == _SMOKE_PERMUTATIONS
        assert len(result.null_distribution) == _SMOKE_PERMUTATIONS
        assert 0.0 <= result.pvalue <= 1.0
        assert result.statistic >= 0.0

    @pytest.mark.component
    @pytest.mark.numerical
    def test_pvalue_add_one_floor(self) -> None:
        """Zero extreme null draws map exactly to ``1 / (B + 1)``."""
        x = jnp.array([[1.0], [2.0], [3.0]])
        y = jnp.array([[10.0], [11.0], [12.0]])

        num_perms = 19
        result = host_permutation_test(
            x, y, simple_mean_difference, num_permutations=num_perms, seed=8
        )

        min_pvalue = 1.0 / (num_perms + 1)
        extreme_count = jnp.sum(
            jnp.abs(result.null_distribution) >= jnp.abs(result.statistic)
        )
        assert int(extreme_count) == 0
        assert result.pvalue == pytest.approx(min_pvalue)

    @pytest.mark.component
    @pytest.mark.property
    @examples(10)
    @example(
        x=random.normal(random.PRNGKey(42), (15, 4)),
        y=random.normal(random.PRNGKey(43), (15, 4)),
        seed=42,
    )
    @given(x=float_arrays((15, 4)), y=float_arrays((15, 4)), seed=_SEEDS)
    def test_seed_reproducibility(self, x: Array, y: Array, seed: int) -> None:
        """Test that same seed gives same results.

        Args:
            x: Generated first sample.
            y: Generated second sample.
            seed: The repeated permutation seed.

        """
        result1 = host_permutation_test(
            x, y, simple_mean_difference, num_permutations=19, seed=seed
        )
        result2 = host_permutation_test(
            x, y, simple_mean_difference, num_permutations=19, seed=seed
        )

        assert jnp.allclose(result1.statistic, result2.statistic)
        assert result1.pvalue == result2.pvalue
        assert jnp.allclose(result1.null_distribution, result2.null_distribution)

    @pytest.mark.component
    @pytest.mark.property
    def test_different_seeds_give_different_results(self) -> None:
        """Test that different seeds give different null distributions."""
        key = random.PRNGKey(42)
        x = random.normal(key, (15, 4))
        y = random.normal(random.PRNGKey(43), (15, 4))

        result1 = host_permutation_test(
            x, y, simple_mean_difference, num_permutations=19, seed=42
        )
        result2 = host_permutation_test(
            x, y, simple_mean_difference, num_permutations=19, seed=123
        )

        # Statistic should be the same (same observed data)
        assert jnp.allclose(result1.statistic, result2.statistic)

        # But null distributions should differ (different random permutations)
        assert not jnp.allclose(result1.null_distribution, result2.null_distribution)

        # P-values might differ slightly
        # (no strict assertion, as they could be similar by chance)

    @pytest.mark.component
    @pytest.mark.property
    @examples(10)
    @example(low=0.0, high=10.0, seed=42)
    @given(low=st.floats(-10, 10), high=st.floats(-10, 10), seed=_SEEDS)
    def test_statistic_in_null_distribution_context(
        self, low: float, high: float, seed: int
    ) -> None:
        """Test that observed statistic makes sense relative to null.

        Args:
            low: Value of the first constant group.
            high: Value of the second constant group.
            seed: Permutation seed.

        """
        x = jnp.full((3, 1), low)
        y = jnp.full((3, 1), high)

        result = host_permutation_test(
            x, y, simple_mean_difference, num_permutations=19, seed=seed
        )

        # Observed statistic should be at or above most null statistics
        # (since distributions are clearly different)
        assert result.statistic >= jnp.mean(result.null_distribution)

    @pytest.mark.component
    @pytest.mark.contract
    @examples(10)
    @example(
        x=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        y=jnp.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
    )
    @given(x=float_arrays((2, 2)), y=float_arrays((3, 2)))
    def test_unequal_sample_sizes(self, x: Array, y: Array) -> None:
        """Test permutation test works with unequal sample sizes.

        Args:
            x: Generated first sample of two rows.
            y: Generated second sample of three rows.

        """
        result = host_permutation_test(
            x, y, simple_mean_difference, num_permutations=99
        )

        assert isinstance(result, HypothesisTest)
        assert result.num_permutations == _STANDARD_PERMUTATIONS
        assert 0.0 <= result.pvalue <= 1.0

    @pytest.mark.component
    def test_alternative_statistic(self) -> None:
        """Test with a different test statistic."""
        key = random.PRNGKey(42)
        # Create samples with same mean but different variance
        x = random.normal(key, (50, 2)) * 0.5
        y = random.normal(random.PRNGKey(43), (50, 2)) * 2.0

        result = host_permutation_test(
            x, y, variance_difference, num_permutations=999, seed=42
        )

        # Should detect variance difference
        assert result.statistic > 0.0
        # Likely to be significant
        assert result.pvalue < _WEAK_SIGNIFICANCE_LEVEL


# ---------------------------------------------------------------------------
# Correction 3: re-optimize w* in permutation / selection split
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Correction 3: re-optimize w* in permutation / selection split
# ---------------------------------------------------------------------------


class TestReoptimization:
    """Check weight reoptimization across permutation routes."""

    @pytest.mark.component
    @pytest.mark.contract
    @examples(5)
    @example(data=original_hedge_data())
    @given(data=hedge_samples())
    def test_plugin_permutation_is_disallowed(self, data: HedgeData) -> None:
        """Verify plugin permutation is disallowed.

        Args:
            data: Generated target and candidate samples.

        """
        x, cands = data
        with pytest.raises(ValueError, match="double-dipping"):
            corrected_mixture_test(
                x, cands, selection_split=None, reoptimize_in_permutation=False
            )

    @pytest.mark.component
    @pytest.mark.contract
    @examples(5)
    @example(data=original_hedge_data())
    @given(data=hedge_samples())
    def test_split_route_flags(self, data: HedgeData) -> None:
        """Verify split route flags.

        Args:
            data: Generated target and candidate samples.

        """
        # Flag-only assertions: a handful of permutations/bootstraps exercises the
        # split-route branch; no statistical power is required here.
        x, cands = data
        r = corrected_mixture_test(
            x,
            cands,
            num_permutations=9,
            metric="euclidean",
            num_bootstrap=9,
            seed=1,
        )
        assert r.reoptimized_in_permutation is False  # split route
        assert r.studentized is True
        assert r.weight_method == "kernel_balance"

    @pytest.mark.component
    @examples(5)
    @example(data=original_hedge_data())
    @given(data=hedge_samples())
    def test_split_route_supports_exact_energy_barycentre(
        self, data: HedgeData
    ) -> None:
        """Exercise the exact barycentre on the selection-split route.

        Args:
            data: Generated target and candidate samples.

        """
        x, cands = data
        r = corrected_mixture_test(
            x,
            cands,
            num_permutations=3,
            metric="euclidean",
            num_bootstrap=3,
            seed=11,
            weight_method="energy_barycentre",
            tol=1e-7,
            maxiter=10_000,
        )

        assert r.weight_method == "energy_barycentre"
        assert r.solver_used == "simplex_qp"
        assert r.solver_converged
        assert r.solver_frank_wolfe_gap is not None
        assert r.solver_cnd_tangent_max_eigenvalue is not None
        assert r.solver_simplex_sum_error is not None
        assert jnp.allclose(jnp.sum(r.weights), 1.0)
        assert jnp.all(r.weights >= 0.0)

    @pytest.mark.component
    @examples(5)
    @example(data=original_hedge_data())
    @example(data=_refinement_permutation_data())
    @given(data=hedge_samples())
    def test_full_sample_exact_route_reoptimizes_permutations(
        self, data: HedgeData
    ) -> None:
        """Reoptimize exact barycentre weights for full-sample permutations.

        Args:
            data: Generated target and candidate samples.

        """
        x, cands = data
        result = corrected_mixture_test(
            x,
            cands,
            num_permutations=2,
            metric="euclidean",
            num_bootstrap=2,
            seed=7,
            selection_split=None,
            reoptimize_in_permutation=True,
            weight_method="energy_barycentre",
            tol=1e-7,
            maxiter=10_000,
        )

        assert result.reoptimized_in_permutation
        assert result.solver_used == "simplex_qp"
        assert result.solver_converged

    @pytest.mark.component
    @pytest.mark.contract
    @examples(5)
    @example(data=original_hedge_data())
    @given(data=hedge_samples())
    def test_unknown_weight_method_is_rejected(self, data: HedgeData) -> None:
        """Verify unknown weight method is rejected.

        Args:
            data: Generated target and candidate samples.

        """
        x, cands = data
        with pytest.raises(ValueError, match="weight_method"):
            corrected_mixture_test(
                x,
                cands,
                num_permutations=1,
                num_bootstrap=1,
                weight_method="unknown",
            )

    @pytest.mark.component
    @pytest.mark.contract
    @pytest.mark.slow
    @examples(5)
    @example(data=original_hedge_data())
    @given(data=hedge_samples())
    def test_reoptimize_route_flags(self, data: HedgeData) -> None:
        """Verify reoptimize route flags.

        Args:
            data: Generated target and candidate samples.

        """
        # Flag-only assertions on the (expensive) re-optimize-in-permutation
        # branch. The QP-per-permutation path costs >2s even at 9 perms, so this
        # stays behind the slow gate; the fast `test_split_route_flags` covers
        # the studentized/non-reoptimize flags in the default gate.
        x, cands = data
        r = corrected_mixture_test(
            x,
            cands,
            num_permutations=9,
            metric="euclidean",
            selection_split=None,
            reoptimize_in_permutation=True,
            num_bootstrap=9,
            seed=2,
        )
        assert r.reoptimized_in_permutation is True
        assert r.studentized is True

    @pytest.mark.component
    @pytest.mark.slow
    def test_far_mixture_rejects(self, hedge_data: HedgeData) -> None:
        """Verify far mixture rejects."""
        # Genuine power assertion (pvalue < 0.1) — needs enough permutations for
        # the null-distribution resolution, so this stays behind the slow gate.
        x, cands = hedge_data
        r = corrected_mixture_test(
            x,
            cands,
            num_permutations=199,
            metric="euclidean",
            num_bootstrap=99,
            seed=3,
        )
        # Target clearly differs from the far-dominated optimal mixture.
        assert r.pvalue < _WEAK_SIGNIFICANCE_LEVEL
