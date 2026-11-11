# jaxtyping numerical carrier contracts.
"""Parity tests for the typed energy-permutation distance hoist.

Compares the typed energy path (one compatible geometry -> pooled
distance matrix ``D`` computed once, replicates driven by reindexing ``D``)
against a naive oracle that recomputes ``cdist`` from scratch for every
replicate (the pre-t05.9 behaviour), on a small deterministic seeded input.
Tolerance + decision-invariant bar (see ``tests/_parity.py`` module docstring):
``allclose`` on the null/observed statistics, plus an identical reject/accept
decision at a fixed alpha.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import random
from statadic.discrepancy.mmd.energy import (
    UNIT_EXPONENT,
    energy_geometry,
    energy_test_statistic,
)
from statadic.geometry.ground.metrics import (
    EUCLIDEAN,
)
from statadic.inference.resampling.permutation import (
    energy_permutation_test,
)

from .._permutation import host_permutation_test

# See tests/_parity.py: reindexing a precomputed distance matrix vs
# recomputing cdist from scratch drifts ~1e-8 from XLA matmul-tiling
# nondeterminism, not from any algebraic difference.
ATOL = 1e-6
ALPHA = 0.05
_EUCLIDEAN_ENERGY = energy_geometry(EUCLIDEAN, UNIT_EXPONENT)


def _energy_statistic_euclidean(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Energy test statistic with euclidean metric (matches the size_power caller)."""
    return energy_test_statistic(x, y, exponent=1.0, metric="euclidean")


@pytest.fixture
def two_sample_data() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Deterministic seeded two-sample fixture (shifted-mean alternative)."""
    x = random.normal(random.PRNGKey(0), (12, 3))
    y = random.normal(random.PRNGKey(1), (10, 3)) + 1.5
    return x, y


class TestDistanceHoistParity:
    """Check parity after hoisting reusable distance computations."""

    @pytest.mark.component
    @pytest.mark.regression
    def test_hoisted_matches_naive_oracle(
        self, two_sample_data: tuple[jnp.ndarray, jnp.ndarray]
    ) -> None:
        """Match the greater-tail naive permutation oracle."""
        x, y = two_sample_data
        num_permutations = 49
        seed = 7

        naive = host_permutation_test(
            x,
            y,
            statistic_fn=_energy_statistic_euclidean,
            num_permutations=num_permutations,
            seed=seed,
            alternative="greater",
        )
        hoisted = energy_permutation_test(
            x,
            y,
            _EUCLIDEAN_ENERGY,
            num_permutations=num_permutations,
            seed=seed,
            alternative="greater",
        )

        assert jnp.allclose(hoisted.statistic, naive.statistic, atol=ATOL)
        assert jnp.allclose(
            jnp.asarray(hoisted.null_distribution),
            jnp.asarray(naive.null_distribution),
            atol=ATOL,
        )
        max_abs_diff = float(
            jnp.max(
                jnp.abs(
                    jnp.asarray(hoisted.null_distribution)
                    - jnp.asarray(naive.null_distribution)
                )
            )
        )
        assert max_abs_diff < ATOL

        # Decision invariance: same reject/accept call at alpha, and exact
        # p-value match since permutation keys/order are identical so the
        # null-vs-observed ordering (and hence the fixed (1+count)/(1+B)
        # formula) is unchanged.
        assert (naive.pvalue < ALPHA) == (hoisted.pvalue < ALPHA)
        assert hoisted.pvalue == naive.pvalue

    @pytest.mark.component
    @pytest.mark.regression
    def test_hoisted_matches_naive_oracle_two_sided(
        self, two_sample_data: tuple[jnp.ndarray, jnp.ndarray]
    ) -> None:
        """Match the two-sided naive permutation oracle."""
        x, y = two_sample_data
        num_permutations = 29
        seed = 11

        naive = host_permutation_test(
            x,
            y,
            statistic_fn=_energy_statistic_euclidean,
            num_permutations=num_permutations,
            seed=seed,
        )
        hoisted = energy_permutation_test(
            x,
            y,
            _EUCLIDEAN_ENERGY,
            num_permutations=num_permutations,
            seed=seed,
        )

        assert jnp.allclose(hoisted.statistic, naive.statistic, atol=ATOL)
        assert hoisted.pvalue == naive.pvalue
        assert (naive.pvalue < ALPHA) == (hoisted.pvalue < ALPHA)

    @pytest.mark.component
    @pytest.mark.regression
    def test_generic_statistic_fn_path_is_unaffected(
        self, two_sample_data: tuple[jnp.ndarray, jnp.ndarray]
    ) -> None:
        """The non-hoisted generic path (arbitrary statistic_fn) is untouched."""
        x, y = two_sample_data

        def mean_diff(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            return jnp.abs(jnp.mean(a) - jnp.mean(b))

        expected_permutations = 19
        result = host_permutation_test(
            x, y, mean_diff, num_permutations=expected_permutations, seed=3
        )
        assert 0.0 <= result.pvalue <= 1.0
        assert result.num_permutations == expected_permutations
