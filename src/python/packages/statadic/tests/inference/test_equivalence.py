"""Tests for equivalence/relevance framing, margin selection, and multiplicity control.

Covers Correction #2 (equivalence null H0: E >= delta vs H1: E < delta), margin
selection, FWER multiplicity control, and the headline integration test asserting
the corrected statistic is all-three-at-once: studentized ∧ equivalence-framed ∧
re-optimized.
"""

import math
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from jax import Array, random
from statadic.discrepancy.mmd.energy.balancing import inverse_variance_m_eff
from statadic.inference.equivalence import (
    equivalence_pvalue_from_bootstrap,
    equivalence_test,
)
from statadic.inference.mixture import MixtureTestResult

from .._budgets import examples
from ._hedge import hedge_samples, original_hedge_data

HedgeData = tuple[Array, list[Array]]


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_bootstrap_or_margin_returns_nan(bad: float) -> None:
    """Any undefined replicate or margin makes the p-value unavailable."""
    assert math.isnan(equivalence_pvalue_from_bootstrap(jnp.array([0.0, bad]), 1.0))
    assert math.isnan(equivalence_pvalue_from_bootstrap(jnp.array([0.0]), bad))
    assert math.isnan(equivalence_pvalue_from_bootstrap(jnp.empty(0), bad))


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "interval",
    [
        (math.nan, 0.0),
        (-math.inf, 0.0),
        (0.0, math.nan),
        (0.0, -math.inf),
        (0.0, math.inf),
    ],
)
def test_nonfinite_interval_cannot_declare_equivalence(
    interval: tuple[float, float],
) -> None:
    """Neither the main verdict nor its curve accepts an undefined interval."""
    base = MixtureTestResult(
        energy_point_estimate=0.0,
        energy_ci=interval,
        studentized_statistic=0.0,
        pvalue=1.0,
        weights=jnp.array([1.0]),
        m_eff=1.0,
        null_distribution=jnp.array([0.0]),
        reoptimized_in_permutation=False,
        studentized=True,
        equivalence_framed=False,
        num_permutations=1,
        bootstrap_distribution=jnp.array([0.0]),
    )
    with patch(
        "statadic.inference.equivalence.corrected_mixture_test", return_value=base
    ):
        result = equivalence_test(
            jnp.zeros((1, 1)),
            [jnp.zeros((1, 1))],
            1.0,
            delta_grid=jnp.array([1.0, 2.0]),
        )
    assert result.equivalence is False
    assert result.equivalence_curve == [(1.0, False), (2.0, False)]


@pytest.mark.component
@pytest.mark.contract
def test_nonfinite_replicate_precludes_verdict_with_finite_interval() -> None:
    """A percentile interval can omit an infinite replicate in its tails."""
    bootstrap = jnp.concatenate([jnp.zeros(99), jnp.array([jnp.inf])])
    interval = tuple(
        float(v) for v in jnp.quantile(bootstrap, jnp.array([0.025, 0.975]))
    )
    assert all(math.isfinite(v) for v in interval)
    base = MixtureTestResult(
        energy_point_estimate=0.0,
        energy_ci=(interval[0], interval[1]),
        studentized_statistic=0.0,
        pvalue=1.0,
        weights=jnp.array([1.0]),
        m_eff=1.0,
        null_distribution=jnp.array([0.0]),
        reoptimized_in_permutation=False,
        studentized=True,
        equivalence_framed=False,
        num_permutations=1,
        bootstrap_distribution=bootstrap,
    )
    with patch(
        "statadic.inference.equivalence.corrected_mixture_test", return_value=base
    ):
        result = equivalence_test(
            jnp.zeros((1, 1)), [jnp.zeros((1, 1))], 1.0, delta_grid=jnp.array([1.0])
        )
    assert result.equivalence_pvalue is not None
    assert math.isnan(result.equivalence_pvalue)
    assert result.equivalence is False
    assert result.equivalence_curve == [(1.0, False)]


@pytest.fixture
def hedge_data() -> HedgeData:
    """Target with a close candidate and a far candidate."""
    x = random.normal(random.PRNGKey(0), (60, 4)) + 3.0
    close = random.normal(random.PRNGKey(1), (30, 4)) + 3.0
    far = random.normal(random.PRNGKey(2), (50, 4)) + 9.0
    return x, [close, far]


# ---------------------------------------------------------------------------
# Correction 2: equivalence / relevance framing
# ---------------------------------------------------------------------------


class TestEquivalence:
    """Check equivalence-test estimation and decision contracts."""

    @pytest.mark.component
    @examples(5)
    @example(data=original_hedge_data())
    @given(data=hedge_samples())
    def test_declares_equivalence_below_margin(self, data: HedgeData) -> None:
        """Verify declares equivalence below margin.

        Args:
            data: Generated target and candidate samples.

        """
        x, cands = data
        # Ê(X, Y_{w*}) is small (close candidate available); a big δ ⇒ equivalent.
        generous_margin = 50.0
        eq = equivalence_test(
            x,
            cands,
            delta=generous_margin,
            num_permutations=9,
            metric="euclidean",
            num_bootstrap=49,
            seed=3,
        )
        assert eq.equivalence is True
        assert eq.equivalence_framed is True
        assert eq.delta == pytest.approx(generous_margin)

    @pytest.mark.component
    def test_rejects_equivalence_below_tiny_margin(self, hedge_data: HedgeData) -> None:
        """Verify rejects equivalence below tiny margin."""
        x, cands = hedge_data
        eq = equivalence_test(
            x,
            cands,
            delta=1e-6,
            num_permutations=9,
            metric="euclidean",
            num_bootstrap=49,
            seed=3,
        )
        assert eq.equivalence is False

    @pytest.mark.component
    @pytest.mark.property
    @examples(5)
    @example(data=original_hedge_data())
    @given(data=hedge_samples())
    def test_equivalence_curve_monotone(self, data: HedgeData) -> None:
        """Verify equivalence curve monotone.

        Args:
            data: Generated target and candidate samples.

        """
        x, cands = data
        grid = jnp.array([0.01, 1.0, 100.0])
        eq = equivalence_test(
            x,
            cands,
            delta=1.0,
            num_permutations=9,
            metric="euclidean",
            num_bootstrap=49,
            seed=3,
            delta_grid=grid,
        )
        rejected = [r for _, r in eq.equivalence_curve]
        # Larger δ ⇒ easier to declare equivalence (monotone non-decreasing).
        assert rejected == sorted(rejected)

    @pytest.mark.component
    @pytest.mark.contract
    @examples(5)
    @example(data=original_hedge_data())
    @given(data=hedge_samples())
    def test_point_estimate_is_primary_metric(self, data: HedgeData) -> None:
        """Verify point estimate is primary metric.

        Args:
            data: Generated target and candidate samples.

        """
        x, cands = data
        eq = equivalence_test(
            x,
            cands,
            delta=10.0,
            num_permutations=9,
            metric="euclidean",
            num_bootstrap=49,
            seed=3,
        )
        lo, hi = eq.energy_ci
        # CI is well-ordered and non-negative; the plug-in Ê need not lie inside
        # a percentile bootstrap CI for a boundary/skewed energy statistic.
        assert lo <= hi
        numerical_floor = -1e-4
        assert lo >= numerical_floor
        assert eq.energy_point_estimate >= numerical_floor

    @pytest.mark.unit
    @pytest.mark.regression
    def test_equivalence_pvalue_direction(self) -> None:
        """BUG #5: small equivalence p ⇒ evidence FOR equivalence (Ê below δ)."""
        boot = jnp.linspace(0.0, 1.0, 100)  # Ê* in [0, 1]
        # δ well above the bootstrap mass ⇒ almost no Ê* ≥ δ ⇒ small p ⇒ equivalent.
        p_equiv = equivalence_pvalue_from_bootstrap(boot, delta=2.0)
        # δ below the bootstrap mass ⇒ most Ê* ≥ δ ⇒ large p ⇒ not equivalent.
        p_not = equivalence_pvalue_from_bootstrap(boot, delta=0.0)
        small_pvalue_limit = 0.05
        large_pvalue_limit = 0.9
        assert p_equiv < small_pvalue_limit
        assert p_not > large_pvalue_limit
        assert p_equiv < p_not

    @pytest.mark.component
    @pytest.mark.regression
    def test_equivalence_pvalue_agrees_with_flag(self, hedge_data: HedgeData) -> None:
        """BUG #5: equivalence_pvalue direction agrees with the CI-based flag."""
        x, cands = hedge_data
        # Big δ ⇒ equivalent (flag True) ⇒ small equivalence p.
        eq_big = equivalence_test(
            x, cands, delta=50.0, num_permutations=9, num_bootstrap=49, seed=3
        )
        assert eq_big.equivalence is True
        assert eq_big.equivalence_pvalue is not None
        decision_midpoint = 0.5
        assert eq_big.equivalence_pvalue < decision_midpoint
        # Tiny δ ⇒ not equivalent (flag False) ⇒ large equivalence p.
        eq_tiny = equivalence_test(
            x, cands, delta=1e-6, num_permutations=9, num_bootstrap=49, seed=3
        )
        assert eq_tiny.equivalence is False
        assert eq_tiny.equivalence_pvalue is not None
        assert eq_tiny.equivalence_pvalue > decision_midpoint
        # The DIFFERENCE p (result.pvalue) is a separate quantity, kept intact.
        assert 0.0 < eq_big.pvalue <= 1.0


# ---------------------------------------------------------------------------
# HEADLINE INTEGRATION TEST: all-three-at-once
# ---------------------------------------------------------------------------


class TestComposedCorrection:
    """Check the composed hedge-inference correction path."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_studentized_and_equivalence_and_reoptimized(
        self, hedge_data: HedgeData
    ) -> None:
        """Verify that the corrected statistic composes all validity fixes.

        studentized ∧ equivalence-framed ∧ re-optimized-in-permutation.

        Only structural flags / m_eff / CI-ordering are asserted (no power), so
        a handful of permutations suffices; but the re-optimize-in-permutation
        QP path is intrinsically expensive (>2s even at 9 perms), so this
        integration test stays behind the slow gate.
        """
        x, cands = hedge_data
        r = equivalence_test(
            x,
            cands,
            delta=10.0,
            num_permutations=9,
            metric="euclidean",
            selection_split=None,
            reoptimize_in_permutation=True,
            num_bootstrap=9,
            seed=7,
        )
        assert r.studentized is True
        assert r.equivalence_framed is True
        assert r.reoptimized_in_permutation is True
        # And it also carries the corrected inverse-variance effective size (#1).
        ms = jnp.array([len(c) for c in cands], dtype=jnp.float32)
        relative_tolerance = 1e-4
        assert jnp.allclose(
            r.m_eff,
            float(inverse_variance_m_eff(r.weights, ms)),
            rtol=relative_tolerance,
        )
        # Primary metric present with a well-ordered, non-negative CI. (The
        # percentile bootstrap CI need not bracket the plug-in Ê for a boundary/
        # skewed statistic — only ordering and non-negativity are guaranteed.)
        assert r.energy_ci[0] <= r.energy_ci[1]
        numerical_floor = -1e-4
        assert r.energy_ci[0] >= numerical_floor
        assert r.energy_point_estimate >= numerical_floor


@pytest.mark.component
@pytest.mark.numerical
@examples(100)
@example(margin=-0.1, bootstrap=np.asarray([0.0, 0.25, 0.25, 0.5, 1.0]))
@example(margin=0.0, bootstrap=np.asarray([0.0, 0.25, 0.25, 0.5, 1.0]))
@example(margin=0.25, bootstrap=np.asarray([0.0, 0.25, 0.25, 0.5, 1.0]))
@example(margin=0.5, bootstrap=np.asarray([0.0, 0.25, 0.25, 0.5, 1.0]))
@example(margin=1.0, bootstrap=np.asarray([0.0, 0.25, 0.25, 0.5, 1.0]))
@example(margin=2.0, bootstrap=np.asarray([0.0, 0.25, 0.25, 0.5, 1.0]))
@given(
    margin=st.sampled_from([-0.1, 0.0, 0.25, 0.5, 1.0, 2.0]),
    bootstrap=hnp.arrays(
        np.float64, 5, elements=st.sampled_from([0.0, 0.25, 0.5, 1.0])
    ),
)
def test_equivalence_tail_counts_include_margin_ties_and_add_one(
    margin: float, bootstrap: np.ndarray
) -> None:
    """Small upper-tail mass supports equivalence; equal-to-margin draws count null.

    Margins sit on the grid of the draws so that ties are exercised, not avoided.

    Args:
        margin: Equivalence margin, on or off the draw grid.
        bootstrap: Generated bootstrap energies drawn from the grid.

    """
    expected = (1.0 + np.count_nonzero(bootstrap >= margin)) / (1.0 + len(bootstrap))
    assert equivalence_pvalue_from_bootstrap(
        jnp.asarray(bootstrap), margin
    ) == pytest.approx(expected)
    assert equivalence_pvalue_from_bootstrap(jnp.empty(0), margin) == pytest.approx(1.0)


@pytest.mark.integration
@pytest.mark.reference
def test_equivalence_direct_interval_and_curve_match_bernoulli_energy_resamples() -> (
    None
):
    """Binary distances give unrooted E=2(p-q)^2 without any native energy oracle."""
    target = np.asarray([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).reshape(-1, 1)
    candidate = np.asarray([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
    testing_x, testing_y = target[4:], candidate[4:]
    bootstrap = []
    for key in random.split(random.PRNGKey(181), 19):
        first_key, second_key = random.split(key)
        first_rows = np.asarray(random.choice(first_key, 4, shape=(4,), replace=True))
        second_rows = np.asarray(random.choice(second_key, 4, shape=(4,), replace=True))
        bootstrap.append(
            2.0 * (testing_x[first_rows].mean() - testing_y[second_rows].mean()) ** 2
        )
    expected_interval = np.quantile(bootstrap, [0.025, 0.975])
    margin = 0.18
    grid = np.asarray([0.125, margin, 0.5])
    result = equivalence_test(
        jnp.asarray(target),
        [jnp.asarray(candidate)],
        delta=margin,
        num_permutations=7,
        num_bootstrap=19,
        seed=181,
        delta_grid=jnp.asarray(grid),
    )
    np.testing.assert_array_equal(result.bootstrap_distribution, bootstrap)
    np.testing.assert_allclose(result.energy_ci, expected_interval, atol=1e-14)
    assert result.energy_point_estimate == pytest.approx(
        2.0 * (testing_x.mean() - testing_y.mean()) ** 2
    )
    assert result.equivalence is bool(expected_interval[1] < margin)
    assert result.equivalence_pvalue == pytest.approx(
        (1.0 + np.count_nonzero(np.asarray(bootstrap) >= margin)) / 20.0
    )
    assert result.equivalence_curve == [
        (float(value), bool(expected_interval[1] < value)) for value in grid
    ]
