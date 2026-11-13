# jaxtyping shape strings are runtime contracts.
"""``statadic.inference.studentized`` — correction 4+1, the variance-scaled statistic.

The studentized mixture statistic divides the energy point estimate by a
bootstrap standard error, which makes it scale-invariant and gives it a null
that does not move with the units of the data. Both properties are asserted
directly (a rescaled input must give the same statistic), because either could
break while the statistic stayed finite and positive.

``studentized_mixture_kernel`` is addressed directly, not only through the public entry
point: the jit/vmap-safety of the core under a *static* epsilon is a separate
contract from the value the wrapper returns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from jax import random
from statadic.discrepancy.mmd.energy import (
    mixture_energy_distance as _med,
)
from statadic.discrepancy.mmd.energy.balancing import inverse_variance_m_eff
from statadic.geometry.ground.metrics import (
    EUCLIDEAN,
    cdist,
)
from statadic.inference.resampling.bootstrap import bootstrap_mixture_energy_interval
from statadic.inference.studentized import (
    studentized_mixture_kernel,
    studentized_mixture_statistic,
)
from statadic.sample.packed import PackedClouds, pack

from .._budgets import examples
from .._strategies import float_arrays
from .._validation.mixture import mixture_energy, studentized_energy
from ._hedge import hedge_samples, original_hedge_data

if TYPE_CHECKING:
    from hypothesis.strategies import DrawFn
    from jax import Array

    from ._hedge import HedgeData

_SEEDS = st.integers(0, 2**31 - 1)

# ---------------------------------------------------------------------------
# Correction 4 + 1: studentized statistic
# ---------------------------------------------------------------------------


def _group_jackknife_variance(replicates: jnp.ndarray) -> jnp.ndarray:
    """Return the leave-one-out variance contribution of one drop group."""
    n_g = replicates.shape[0]
    return (n_g - 1.0) / n_g * jnp.sum((replicates - jnp.mean(replicates)) ** 2)


def _jackknife_variance(*groups: jnp.ndarray) -> float:
    """Sum the per-group leave-one-out variances of the given replicate blocks."""
    return float(sum(_group_jackknife_variance(group) for group in groups))


def _drop_x_replicates(
    x: jnp.ndarray, cand: list[jnp.ndarray], w: jnp.ndarray
) -> jnp.ndarray:
    """Return the leave-one-out energy replicates dropping each row of ``x``."""
    return jnp.array(
        [
            _med(
                PackedClouds.from_array(jnp.asarray(jnp.delete(x, i, axis=0))),
                pack(cand, require_positive_counts=True),
                w,
                1.0,
                "euclidean",
            )
            for i in range(x.shape[0])
        ]
    )


def _drop_candidate_replicates(
    x: jnp.ndarray, cand: list[jnp.ndarray], w: jnp.ndarray, index: int
) -> jnp.ndarray:
    """Return the replicates dropping each row of candidate group ``index``."""
    target: jnp.ndarray = cand[index]
    return jnp.array(
        [
            _med(
                PackedClouds.from_array(jnp.asarray(x)),
                pack(
                    [*cand[:index], jnp.delete(target, j, axis=0), *cand[index + 1 :]],
                    require_positive_counts=True,
                ),
                w,
                1.0,
                "euclidean",
            )
            for j in range(target.shape[0])
        ]
    )


def _expected_studentized(
    x: jnp.ndarray,
    cand: list[jnp.ndarray],
    w: jnp.ndarray,
    sizes: list[float],
    sigma: float,
) -> float:
    """Rebuild the studentized statistic from a hand-computed jackknife sigma."""
    e_hat = float(
        _med(
            PackedClouds.from_array(jnp.asarray(x)),
            pack(cand, require_positive_counts=True),
            w,
            1.0,
            "euclidean",
        )
    )
    m_eff = float(inverse_variance_m_eff(w, jnp.array(sizes)))
    scale = (x.shape[0] * m_eff) / (x.shape[0] + m_eff)
    return (scale**0.5) * e_hat / sigma


def _with_spread(data: HedgeData) -> HedgeData:
    """Add a ramp wider than the draw bound to every group, by construction.

    A constant sample has zero jackknife variance, so the studentizing epsilon floor
    (which is not scale-covariant) would decide the result: outside the statistic's
    domain. The ramp rules that out without filtering.
    """
    target, candidates = data

    def ramped(group: Array) -> Array:
        return group + jnp.linspace(-5.0, 5.0, group.shape[0])[:, None]

    return ramped(target), [ramped(group) for group in candidates]


def _translated_empirical_groups(points: Array) -> HedgeData:
    """Repeat one nonconstant empirical distribution and translate the far copy."""
    points = points + jnp.linspace(-5.0, 5.0, 10)[:, None]
    return jnp.tile(points, (6, 1)), [
        jnp.tile(points, (3, 1)),
        jnp.tile(points + 6.0, (5, 1)),
    ]


class TestStudentized:
    """Check studentized mixture-statistic behavior."""

    @pytest.mark.component
    @pytest.mark.property
    @examples(10)
    @example(
        sizes=(3, 4, 5),
        points=random.normal(random.PRNGKey(12), (12, 2)),
        seed=19,
    )
    @example(
        sizes=(4, 3, 5),
        points=random.normal(random.PRNGKey(12), (12, 2)),
        seed=19,
    )
    @given(
        sizes=st.sampled_from([(3, 4, 5), (4, 3, 5)]),
        points=float_arrays((12, 2)),
        seed=_SEEDS,
    )
    def test_energy_bootstrap_is_seeded_for_multiple_static_shape_tuples(
        self, sizes: tuple[int, ...], points: Array, seed: int
    ) -> None:
        """`lax.map` bootstrap is repeatable for distinct pooled-group shapes.

        Args:
            sizes: Group sizes, in two distinct static orders.
            points: Generated pooled points.
            seed: Bootstrap seed.

        """
        n_groups = len(sizes)
        distances = cdist(points, points, metric=EUCLIDEAN)
        group_ids = jnp.concatenate(
            [jnp.full(size, group, dtype=jnp.int32) for group, size in enumerate(sizes)]
        )
        membership = jax.nn.one_hot(group_ids, n_groups, dtype=distances.dtype)
        size_arr = membership.sum(0)
        weights = jnp.ones(n_groups - 1) / (n_groups - 1)

        first_ci, first_dist = bootstrap_mixture_energy_interval(
            distances,
            membership,
            size_arr,
            list(sizes),
            weights,
            confidence_level=0.9,
            num_bootstrap=5,
            rng=random.PRNGKey(seed),
        )
        second_ci, second_dist = bootstrap_mixture_energy_interval(
            distances,
            membership,
            size_arr,
            list(sizes),
            weights,
            confidence_level=0.9,
            num_bootstrap=5,
            rng=random.PRNGKey(seed),
        )

        assert jnp.allclose(first_dist, second_dist)
        assert first_ci == pytest.approx(second_ci)
        assert jnp.all(jnp.isfinite(first_dist))

    @pytest.mark.component
    @pytest.mark.contract
    @examples(10)
    @example(points=random.normal(random.PRNGKey(7), (12, 2)))
    @given(
        points=float_arrays((12, 2)).map(
            # A ramp wider than the draw bound rules out identical points, whose
            # zero jackknife variance is outside the statistic's domain.
            lambda a: a + jnp.linspace(-5.0, 5.0, 12)[:, None]
        )
    )
    def test_studentized_kernel_supports_jit_vmap_with_static_epsilon(
        self, points: Array
    ) -> None:
        """The prevalidated core agrees with its wrapper under `jit` and `vmap`.

        Args:
            points: Generated pooled points with non-degenerate spread.

        """
        sizes = (4, 3, 5)
        n_groups = len(sizes)
        distances = cdist(points, points, metric=EUCLIDEAN)
        group_ids = jnp.concatenate(
            [jnp.full(size, group, dtype=jnp.int32) for group, size in enumerate(sizes)]
        )
        weights = jnp.array([0.4, 0.6])

        def core(d: jnp.ndarray) -> jnp.ndarray:
            return studentized_mixture_kernel(
                d, group_ids, weights, None, n_groups, eps=1e-8
            )

        candidates = [points[4:7], points[7:]]
        expected = studentized_mixture_statistic(points[:4], candidates, weights)
        batched_distances = jnp.stack([distances, distances * 1.1])

        assert jnp.allclose(core(distances), expected)
        assert jnp.allclose(jax.jit(core)(distances), expected)
        assert jnp.allclose(
            jax.vmap(core)(batched_distances),
            jnp.array([core(distances), core(distances * 1.1)]),
        )

    @pytest.mark.component
    @pytest.mark.property
    @examples(5)
    @example(data=original_hedge_data())
    @given(data=hedge_samples())
    def test_positive_and_finite(self, data: HedgeData) -> None:
        """Verify positive and finite.

        Args:
            data: Generated target and candidate samples.

        """
        x, cands = data
        w = jnp.array([0.5, 0.5])
        t = studentized_mixture_statistic(x, cands, w, metric="euclidean")
        assert jnp.isfinite(t)
        assert t > 0.0

    @pytest.mark.component
    @pytest.mark.property
    @examples(5)
    @example(data=original_hedge_data())
    @given(data=float_arrays((10, 4), bound=1.5).map(_translated_empirical_groups))
    def test_larger_for_farther_mixture(self, data: HedgeData) -> None:
        """A translated empirical distribution exceeds an identical one.

        Studentization does not preserve energy ordering in general because
        the jackknife variance and effective size depend on the candidate.
        Here the close empirical measure equals the target, so its energy is
        zero; translating the nonconstant far measure gives positive energy.

        Args:
            data: Repeated target and candidate samples, far group 6 units away.

        """
        x, cands = data
        t_far = studentized_mixture_statistic(
            x, cands, jnp.array([0.0, 1.0]), metric="euclidean"
        )
        t_close = studentized_mixture_statistic(
            x, cands, jnp.array([1.0, 0.0]), metric="euclidean"
        )
        assert t_far > t_close

    @pytest.mark.component
    @pytest.mark.regression
    def test_jackknife_sigma_shrinks_vs_pooled(self) -> None:
        """BUG #6: per-group jackknife σ̂ < the buggy pooled σ̂ (asymmetric case).

        The X-drop and Y-drop leave-one-out replicates have different centres, so
        pooling them injects a spurious between-group term. On an asymmetric
        example the corrected (sum of per-group variances) σ̂ must be strictly
        smaller than the pooled version. We reconstruct both from the SAME LOO
        replicates to isolate the variance formula.
        """
        x = random.normal(random.PRNGKey(0), (25, 3)) + 5.0
        cand = [random.normal(random.PRNGKey(1), (12, 3)) + 5.2]
        w = jnp.array([1.0])

        e_x = _drop_x_replicates(x, cand, w)
        e_y = _drop_candidate_replicates(x, cand, w, 0)

        fixed_sigma = _jackknife_variance(e_x, e_y) ** 0.5
        pooled_sigma = _jackknife_variance(jnp.concatenate([e_x, e_y])) ** 0.5
        # The between-group offset (different X- vs Y-drop centres) makes the
        # corrected per-group σ̂ strictly smaller.
        assert fixed_sigma < pooled_sigma
        # And the statistic's σ̂ matches the corrected per-group formula: because
        # a smaller σ̂ yields a LARGER studentized statistic than the pooled one.
        t = float(studentized_mixture_statistic(x, cand, w, metric="euclidean"))
        t_fixed: float = _expected_studentized(x, cand, w, [12.0], fixed_sigma)
        assert jnp.allclose(t, t_fixed, rtol=1e-4)

    @pytest.mark.component
    @pytest.mark.regression
    def test_jackknife_sums_per_candidate_groups(self) -> None:
        """jack_var = jack_X + Σ_k jack_{Y_k} (K≥2 unequal sizes), not pooled Y."""
        x = random.normal(random.PRNGKey(0), (20, 2)) + 1.0
        c0 = random.normal(random.PRNGKey(1), (8, 2)) + 1.1
        c1 = random.normal(random.PRNGKey(2), (15, 2)) + 0.9
        cand = [c0, c1]
        w = jnp.array([0.4, 0.6])

        e_x = _drop_x_replicates(x, cand, w)
        e_y0 = _drop_candidate_replicates(x, cand, w, 0)
        e_y1 = _drop_candidate_replicates(x, cand, w, 1)

        multi = _jackknife_variance(e_x, e_y0, e_y1)
        # Buggy single stacked-Y group (pre-fix formula for K>1).
        buggy = _jackknife_variance(e_x, jnp.concatenate([e_y0, e_y1]))
        assert multi != pytest.approx(buggy, rel=1e-6)

        t = float(studentized_mixture_statistic(x, cand, w, metric="euclidean"))
        t_gold: float = _expected_studentized(x, cand, w, [8.0, 15.0], multi**0.5)
        assert jnp.allclose(t, t_gold, rtol=1e-4)

    @pytest.mark.component
    @pytest.mark.property
    @examples(5)
    @example(data=original_hedge_data(), c=5.0)
    @given(
        data=hedge_samples().map(_with_spread),
        c=st.floats(0.5, 5.0),
    )
    def test_scale_invariance_property(self, data: HedgeData, c: float) -> None:
        """Studentization ⇒ statistic is invariant to a common energy rescale.

        Args:
            data: Generated target and candidate samples.
            c: Common coordinate scale.

        """
        x, cands = data
        w = jnp.array([0.3, 0.7])
        t1 = studentized_mixture_statistic(x, cands, w, metric="euclidean")
        # Scale all coordinates by c: energy and its std scale by c (exponent=1),
        # so the studentized ratio is unchanged up to the sqrt(scale) factor which
        # is size-only, hence unchanged.
        t2 = studentized_mixture_statistic(
            c * x, [c * cc for cc in cands], w, metric="euclidean"
        )
        assert jnp.allclose(t1, t2, rtol=1e-2)


@pytest.mark.component
@pytest.mark.reference
@examples(12)
@example(
    entries=[0, 1, 2, 3, -1, 0, 4, 2, 1, 0, -3, 2], numerator=3, heterogeneous=False
)
@given(
    entries=st.lists(st.integers(-4, 4), min_size=12, max_size=12),
    numerator=st.integers(1, 7),
    heterogeneous=st.booleans(),
)
def test_unequal_group_studentization_matches_raw_distance_delete_one_oracle(
    entries: list[int],
    numerator: int,
    heterogeneous: bool,  # noqa: FBT001
) -> None:
    """The oracle shares neither native energy, jackknife nor effective-size code."""
    raw = np.asarray(entries, dtype=np.float64).reshape(-1, 1)
    target, candidates = raw[:3], [raw[3:7], raw[7:]]
    weights = np.asarray([numerator / 8.0, 1.0 - numerator / 8.0])
    variances = np.asarray([1.0, 4.0]) if heterogeneous else None
    expected, _, effective_size = studentized_energy(
        target, candidates, weights, variances
    )
    actual = studentized_mixture_statistic(
        jnp.asarray(target),
        [jnp.asarray(group) for group in candidates],
        jnp.asarray(weights),
        group_variances=None if variances is None else jnp.asarray(variances),
    )
    # The existing effective-size API deliberately accumulates in float32.
    assert float(actual) == pytest.approx(expected, rel=2e-6, abs=1e-8)
    assert float(
        inverse_variance_m_eff(
            jnp.asarray(weights),
            jnp.asarray([4, 5]),
            None if variances is None else jnp.asarray(variances),
        )
    ) == pytest.approx(effective_size, rel=2e-6)


_GROUP_SIZES = (3, 4, 5)
_ORIGINAL_GROUPS = (
    np.asarray([0.0, 1.0, 2.0]),
    np.asarray([-1.0, 0.0, 1.0, 3.0]),
    np.asarray([1.0, 2.0, 4.0, 5.0, 7.0]),
)


@st.composite
def raw_groups(draw: DrawFn) -> tuple[np.ndarray, ...]:
    """Draw one-dimensional sample groups of fixed unequal sizes.

    Args:
        draw: Hypothesis draw function.

    Returns:
        Three float64 vectors of sizes 3, 4 and 5.

    """
    return tuple(
        draw(
            hnp.arrays(
                np.float64,
                size,
                elements=st.floats(-10, 10, allow_nan=False),
            )
        )
        for size in _GROUP_SIZES
    )


@pytest.mark.component
@pytest.mark.reference
@examples(10)
@example(values=_ORIGINAL_GROUPS, weight=0.25, seed=92)
@given(values=raw_groups(), weight=st.floats(0.0, 1.0), seed=_SEEDS)
def test_energy_bootstrap_each_draw_matches_independent_raw_cloud_resampling(
    values: tuple[np.ndarray, ...], weight: float, seed: int
) -> None:
    """Unequal counts, duplicate draws and fixed weights keep their V denominators.

    Args:
        values: Generated groups of sizes 3, 4 and 5.
        weight: Weight on the first candidate group.
        seed: Bootstrap seed.

    """
    groups = [group.reshape(-1, 1) for group in values]
    sizes = [len(group) for group in groups]
    weights = np.asarray([weight, 1.0 - weight])
    pooled = np.concatenate(groups)
    distances = np.abs(pooled[:, 0, None] - pooled[:, 0][None, :])
    ids = np.repeat(np.arange(len(groups)), sizes)
    membership = jax.nn.one_hot(jnp.asarray(ids), len(groups), dtype=jnp.float64)
    key = random.PRNGKey(seed)
    expected = []
    for draw_key in random.split(key, 7):
        keys = random.split(draw_key, len(groups))
        resampled = [
            group[
                np.asarray(
                    random.choice(
                        group_key, len(group), shape=(len(group),), replace=True
                    )
                )
            ]
            for group, group_key in zip(groups, keys, strict=True)
        ]
        expected.append(mixture_energy(resampled[0], resampled[1:], weights))
    interval, distribution = bootstrap_mixture_energy_interval(
        jnp.asarray(distances),
        membership,
        membership.sum(axis=0),
        sizes,
        jnp.asarray(weights),
        0.95,
        7,
        key,
    )
    np.testing.assert_allclose(distribution, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        interval, np.quantile(expected, [0.025, 0.975]), rtol=1e-12
    )
