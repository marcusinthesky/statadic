# jaxtyping shape strings are runtime contracts.
"""Permutation result composition through nested JIT and vmap."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from jax import random
from statadic.core.results import TestResult
from statadic.discrepancy.mmd.energy import (
    UNIT_EXPONENT,
    energy_geometry,
    energy_test_statistic,
)
from statadic.geometry.ground.metrics import (
    EUCLIDEAN,
)
from statadic.inference.resampling.permutation import (
    EnergyPermutationResult,
    energy_permutation_test_result,
    permutation_test_kernel,
    permutation_test_result,
)
from statadic.inference.resampling.results import (
    HostAdapterError,
    to_host_test_result,
)
from scipy import stats

from ..._budgets import examples
from ..._permutation import host_permutation_test
from ..._strategies import float_arrays

if TYPE_CHECKING:
    from jax import Array
    from statadic.discrepancy.mmd.energy import (
        SeparatingDistributionEnergyLaw,
        UnitExponent,
    )
    from statadic.geometry.ground.metrics import EuclideanLaw
    from statadic.inference.resampling.permutation.kernel import (
        Alternative,
    )

# --- the vmap contract (t46.7) --------------------------------------------


_CELLS = 4
_RESAMPLES = 16
_N = 12
_DIM = 2
_EUCLIDEAN_ENERGY = energy_geometry(EUCLIDEAN, UNIT_EXPONENT)
_SEEDS = st.integers(0, 2**31 - 1)
_SHIFTS = float_arrays(_CELLS, bound=6.0)
_TIED_GRID = st.integers(-3, 3).map(float)
_ALTERNATIVES = st.sampled_from(["greater", "less", "two-sided"])


def _energy_permutation_cell(
    key: jnp.ndarray, shift: jnp.ndarray
) -> EnergyPermutationResult[
    EuclideanLaw,
    UnitExponent,
    SeparatingDistributionEnergyLaw,
]:
    """One Monte-Carlo cell: draw two samples at ``shift``, permute, test.

    Everything here is a JAX op on traced values -- no ``.item()``, no
    ``float()`` -- which is what lets the whole cell be vmapped.

    Args:
        key: PRNG key for the two draws and the permutations.
        shift: Location offset applied to the second sample.

    Returns:
        The cell's ``TestResult``.

    """
    k_x, k_y, k_perm = random.split(key, 3)
    x = random.normal(k_x, (_N, _DIM))
    y = random.normal(k_y, (_N, _DIM)) + shift
    return energy_permutation_test_result(
        x,
        y,
        _EUCLIDEAN_ENERGY,
        k_perm,
        num_permutations=_RESAMPLES,
        alternative="greater",
    )


class TestResultsSurviveAGrid:
    """``vmap`` over a Monte-Carlo grid of permutation tests."""

    @pytest.mark.integration
    @pytest.mark.contract
    @examples(10)
    @example(seed=0, shifts=jnp.linspace(0.0, 3.0, _CELLS))
    @given(seed=_SEEDS, shifts=_SHIFTS)
    def test_grid_returns_one_batched_test_result(
        self, seed: int, shifts: Array
    ) -> None:
        """One container comes back, not ``_CELLS`` of them.

        Args:
            seed: Seed for the per-cell keys.
            shifts: Generated per-cell location offsets.

        """
        keys = random.split(random.PRNGKey(seed), _CELLS)

        batched = jax.vmap(_energy_permutation_cell)(keys, shifts)

        assert isinstance(batched, EnergyPermutationResult)
        assert batched.statistic.shape == (_CELLS,)
        assert batched.pvalue.shape == (_CELLS,)
        assert batched.null_distribution.shape == (_CELLS, _RESAMPLES)

    @pytest.mark.integration
    @pytest.mark.contract
    @examples(10)
    @example(seed=1, shifts=jnp.linspace(0.0, 3.0, _CELLS))
    @given(seed=_SEEDS, shifts=_SHIFTS)
    def test_grid_leaf_shapes_carry_the_batch_axis(
        self, seed: int, shifts: Array
    ) -> None:
        """``tree_leaves`` sees exactly three batched leaves, in field order.

        Args:
            seed: Seed for the per-cell keys.
            shifts: Generated per-cell location offsets.

        """
        keys = random.split(random.PRNGKey(seed), _CELLS)

        batched = jax.vmap(_energy_permutation_cell)(keys, shifts)

        assert [leaf.shape for leaf in jax.tree_util.tree_leaves(batched)] == [
            (_CELLS,),
            (_CELLS,),
            (_CELLS, _RESAMPLES),
        ]

    @pytest.mark.integration
    @pytest.mark.contract
    @examples(10)
    @example(seed=2, shifts=jnp.zeros(_CELLS))
    @given(seed=_SEEDS, shifts=_SHIFTS)
    def test_n_resamples_stays_static_across_the_grid(
        self, seed: int, shifts: Array
    ) -> None:
        """The static field is a Python ``int``, never a batched leaf.

        Args:
            seed: Seed for the per-cell keys.
            shifts: Generated per-cell location offsets.

        """
        keys = random.split(random.PRNGKey(seed), _CELLS)

        batched = jax.vmap(_energy_permutation_cell)(keys, shifts)

        assert batched.n_resamples == _RESAMPLES
        assert isinstance(batched.n_resamples, int)

    @pytest.mark.integration
    @pytest.mark.property
    def test_grid_pvalues_fall_with_separation(self) -> None:
        """The batched result is a real test, not a shape-correct placeholder."""
        keys = random.split(random.PRNGKey(3), _CELLS)
        shifts = jnp.linspace(0.0, 6.0, _CELLS)

        batched = jax.vmap(_energy_permutation_cell)(keys, shifts)

        assert float(batched.pvalue[0]) > float(batched.pvalue[-1])

    @pytest.mark.integration
    @pytest.mark.contract
    @examples(10)
    @example(seed=4, shifts=jnp.linspace(0.0, 3.0, _CELLS))
    @given(seed=_SEEDS, shifts=_SHIFTS)
    def test_grid_is_jittable(self, seed: int, shifts: Array) -> None:
        """The same grid closes under ``jit`` — the container is not a barrier.

        Args:
            seed: Seed for the per-cell keys.
            shifts: Generated per-cell location offsets.

        """
        keys = random.split(random.PRNGKey(seed), _CELLS)

        jitted = jax.jit(jax.vmap(_energy_permutation_cell))(keys, shifts)
        eager = jax.vmap(_energy_permutation_cell)(keys, shifts)

        assert jnp.allclose(jitted.pvalue, eager.pvalue)

    @pytest.mark.integration
    @pytest.mark.contract
    @examples(10)
    @example(seed=5, shifts=jnp.linspace(0.0, 2.0, _CELLS))
    @given(seed=_SEEDS, shifts=_SHIFTS)
    def test_full_permutation_result_survives_nested_jit_and_vmap(
        self, seed: int, shifts: Array
    ) -> None:
        """The PRNG-key core, not only result assembly, closes under transforms.

        Args:
            seed: Seed for the per-cell keys.
            shifts: Generated per-cell location offsets.

        """

        def cell(key: jax.Array, shift: jax.Array) -> TestResult[jax.Array]:
            x = jnp.arange(_N * _DIM, dtype=jnp.float32).reshape(_N, _DIM)
            y = x + shift
            return permutation_test_result(
                x,
                y,
                lambda left, right: jnp.abs(jnp.mean(left) - jnp.mean(right)),
                key,
                num_permutations=_RESAMPLES,
                alternative="greater",
            )

        keys = random.split(random.PRNGKey(seed), _CELLS)
        transformed = jax.jit(jax.jit(jax.vmap(cell)))(keys, shifts)

        assert isinstance(transformed, TestResult)
        assert transformed.statistic.shape == (_CELLS,)
        assert transformed.pvalue.shape == (_CELLS,)
        assert transformed.null_distribution.shape == (_CELLS, _RESAMPLES)
        assert transformed.n_resamples == _RESAMPLES

    @pytest.mark.integration
    @pytest.mark.contract
    @examples(20)
    @example(observed=1.0)
    @given(observed=st.floats(-10, 10))
    def test_host_test_adapter_fails_explicitly_under_tracing(
        self, observed: float
    ) -> None:
        """Compiled code cannot accidentally concretize a legacy report.

        Args:
            observed: Generated observed statistic.

        """

        def invalid_host_boundary(statistic: jax.Array) -> jax.Array:
            result = permutation_test_kernel(
                statistic, jnp.zeros(_RESAMPLES), _RESAMPLES, "greater"
            )
            return jnp.asarray(to_host_test_result(result).pvalue)

        with pytest.raises(HostAdapterError, match="host-only"):
            jax.jit(invalid_host_boundary)(jnp.asarray(observed))


class TestKernelContract:
    """The kernel's own contract, independent of any grid."""

    @pytest.mark.component
    @pytest.mark.numerical
    @examples(100)
    @example(
        alternative="greater",
        statistic=2.0,
        null=np.array([-3.0, -2.0, -2.0, 0.0, 2.0, 2.0, 4.0]),
    )
    @example(
        alternative="greater",
        statistic=-2.0,
        null=np.array([-3.0, -2.0, -2.0, 0.0, 2.0, 2.0, 4.0]),
    )
    @example(
        alternative="less",
        statistic=2.0,
        null=np.array([-3.0, -2.0, -2.0, 0.0, 2.0, 2.0, 4.0]),
    )
    @example(
        alternative="less",
        statistic=-2.0,
        null=np.array([-3.0, -2.0, -2.0, 0.0, 2.0, 2.0, 4.0]),
    )
    @example(
        alternative="two-sided",
        statistic=2.0,
        null=np.array([-3.0, -2.0, -2.0, 0.0, 2.0, 2.0, 4.0]),
    )
    @example(
        alternative="two-sided",
        statistic=-2.0,
        null=np.array([-3.0, -2.0, -2.0, 0.0, 2.0, 2.0, 4.0]),
    )
    @given(
        alternative=_ALTERNATIVES,
        statistic=_TIED_GRID,
        null=hnp.arrays(np.float64, 7, elements=_TIED_GRID),
    )
    def test_literal_tail_counts_include_ties_for_every_alternative(
        self, alternative: Alternative, statistic: float, null: np.ndarray
    ) -> None:
        """Interior tails protect direction, absolute two-sidedness and ties.

        Integer-valued draws make ties common; the oracle counts them explicitly.

        Args:
            alternative: Tail direction.
            statistic: Observed statistic, possibly tied with null draws.
            null: Generated null draws.

        """
        extreme_count = {
            "greater": np.count_nonzero(null >= statistic),
            "less": np.count_nonzero(null <= statistic),
            "two-sided": np.count_nonzero(np.abs(null) >= abs(statistic)),
        }[alternative]

        result = permutation_test_kernel(
            jnp.asarray(statistic), jnp.asarray(null), null.size, alternative
        )

        assert float(result.pvalue) == pytest.approx((1 + extreme_count) / 8)

    @pytest.mark.component
    @pytest.mark.numerical
    @examples(100)
    @example(
        alternative="two-sided", beyond=7.0, null=np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    )
    @example(
        alternative="greater", beyond=7.0, null=np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    )
    @example(alternative="less", beyond=7.0, null=np.array([-3.0, -1.0, 0.0, 1.0, 3.0]))
    @given(
        alternative=_ALTERNATIVES,
        beyond=st.floats(0.5, 20.0),
        null=hnp.arrays(np.float64, 5, elements=st.floats(-3.0, 3.0)),
    )
    def test_no_extreme_null_draw_uses_add_one_floor(
        self, alternative: Alternative, beyond: float, null: np.ndarray
    ) -> None:
        """No extreme draw gives ``1 / (B + 1)``, never zero.

        Args:
            alternative: Tail direction.
            beyond: Distance by which the statistic clears every null draw.
            null: Generated null draws.

        """
        reach = float(np.max(np.abs(null))) + beyond
        statistic = -reach if alternative == "less" else reach

        result = permutation_test_kernel(
            jnp.asarray(statistic), jnp.asarray(null), null.size, alternative
        )

        assert float(result.pvalue) == pytest.approx(1.0 / (null.size + 1))

    @pytest.mark.component
    @pytest.mark.numerical
    def test_add_one_count_includes_ties(self) -> None:
        """The numerator adds one after counting equally extreme draws."""
        null = jnp.asarray([0.0, 1.0, 2.0, 2.0])

        result = permutation_test_kernel(jnp.asarray(2.0), null, null.size, "greater")

        assert float(result.pvalue) == pytest.approx(3.0 / 5.0)

    @pytest.mark.integration
    @pytest.mark.contract
    @examples(40)
    @example(
        statistics=np.array([10.0, 1.0]),
        nulls=np.array([[-3.0, -1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
    )
    @given(
        statistics=hnp.arrays(np.float64, 2, elements=_TIED_GRID),
        nulls=hnp.arrays(np.float64, (2, 4), elements=_TIED_GRID),
    )
    def test_add_one_floor_survives_jit_and_vmap(
        self, statistics: np.ndarray, nulls: np.ndarray
    ) -> None:
        """A batched compiled kernel retains one floor per Monte-Carlo cell.

        Args:
            statistics: Generated per-cell statistics.
            nulls: Generated per-cell null draws.

        """

        def evaluate_permutation_test_kernel(
            statistic: jax.Array, null: jax.Array
        ) -> TestResult[Array]:
            return permutation_test_kernel(statistic, null, nulls.shape[-1], "greater")

        batched_kernel = jax.jit(jax.vmap(evaluate_permutation_test_kernel))

        result = batched_kernel(jnp.asarray(statistics), jnp.asarray(nulls))
        counts = np.count_nonzero(nulls >= statistics[:, None], axis=1)

        assert isinstance(result, TestResult)
        assert result.pvalue.shape == (2,)
        assert jnp.allclose(result.pvalue, (1.0 + counts) / 5.0)
        assert result.null_distribution.shape == nulls.shape
        assert result.n_resamples == nulls.shape[-1]

    @pytest.mark.component
    @pytest.mark.property
    @examples(5)
    @example(
        x=random.normal(random.PRNGKey(10), (20, 3)),
        y=random.normal(random.PRNGKey(11), (20, 3)) + 1.5,
        seed=7,
    )
    @given(x=float_arrays((20, 3)), y=float_arrays((20, 3)), seed=_SEEDS)
    def test_single_run_matches_the_eager_entry_point(
        self, x: Array, y: Array, seed: int
    ) -> None:
        """The host report is the kernel plus a concretizing boundary.

        Args:
            x: Generated first sample.
            y: Generated second sample.
            seed: Permutation seed.

        """
        eager = host_permutation_test(
            x,
            y,
            statistic_fn=lambda a, b: energy_test_statistic(a, b, metric="euclidean"),
            num_permutations=31,
            seed=seed,
            alternative="greater",
        )

        direct = permutation_test_kernel(
            jnp.asarray(eager.statistic),
            jnp.asarray(eager.null_distribution),
            eager.num_permutations,
            "greater",
        )

        assert float(direct.pvalue) == pytest.approx(eager.pvalue)
        assert direct.n_resamples == eager.num_permutations

    @pytest.mark.component
    @pytest.mark.contract
    @examples(40)
    @example(alternative="sideways")
    @given(
        alternative=st.text(max_size=10).filter(
            lambda a: a not in {"greater", "less", "two-sided"}
        )
    )
    def test_unknown_alternative_raises(self, alternative: str) -> None:
        """A bad alternative fails in either runtime-enforcement plane.

        The jaxtyping import hook rejects the ``Literal`` at entry with a
        ``TypeError``. Without that opt-in hook, the body retains its precise
        ``ValueError`` for untyped production callers.

        Args:
            alternative: Any string that is not a supported tail.

        """
        with pytest.raises(
            (TypeError, ValueError),
            match=r"alternative must be|Static\[Alternative\]",
        ):
            permutation_test_kernel(
                jnp.asarray(1.0),
                jnp.zeros(4),
                4,
                cast("Alternative", alternative),
            )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("invalid", ["nan_stat", "inf_stat", "nan_draw", "inf_draw"])
def test_nonfinite_permutation_lanes(
    alternative: Literal["two-sided", "greater", "less"], invalid: str
) -> None:
    """Invalid lanes are unavailable across transformations without changing peers."""
    stats = jnp.array([0.5, 0.5])
    draws = jnp.array([[0.0, 1.0, -1.0], [0.0, 1.0, -1.0]])
    if invalid.endswith("stat"):
        stats = stats.at[0].set(jnp.nan if invalid == "nan_stat" else jnp.inf)
    else:
        draws = draws.at[0, 0].set(jnp.nan if invalid == "nan_draw" else -jnp.inf)

    def evaluate(stat: jax.Array, null: jax.Array) -> jax.Array:
        return permutation_test_kernel(stat, null, 3, alternative).pvalue

    baseline = evaluate(stats[1], draws[1])
    for run in (jax.vmap(evaluate), jax.jit(jax.vmap(evaluate))):
        result = run(stats, draws)
        assert jnp.isnan(result[0])
        np.testing.assert_array_equal(result[1], baseline)
    assert jnp.isnan(evaluate(stats[0], draws[0]))
    assert jnp.isnan(jax.jit(evaluate)(stats[0], draws[0]))


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf])
def test_unavailable_permutation_pvalue_survives_host_adapter(bad: float) -> None:
    """The legacy float result carries an unavailable kernel p-value as NaN."""
    result = permutation_test_kernel(jnp.asarray(bad), jnp.array([0.0, 1.0]), 2)
    assert np.isnan(to_host_test_result(result).pvalue)


# --- independent p-value witnesses ----------------------------------------


def _witness_pvalue(
    statistic: float, null: np.ndarray, alternative: str, num_permutations: int
) -> float:
    """Count by the documented rule: ``(1 + b) / (1 + B)``.

    ``b`` is the number of null draws as or more extreme than the observation:
    ``abs(null) >= abs(statistic)`` two-sided, ``null >= statistic`` greater and
    ``null <= statistic`` less. Plain Python loops, no statadic or JAX helper.
    """
    extreme = 0
    for draw in null.tolist():
        if alternative == "two-sided":
            extreme += abs(draw) >= abs(statistic)
        elif alternative == "greater":
            extreme += draw >= statistic
        else:
            extreme += draw <= statistic
    return (1 + extreme) / (1 + num_permutations)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_kernel_pvalue_matches_numpy_count_on_tied_draws(
    alternative: Literal["two-sided", "greater", "less"], seed: int
) -> None:
    """The kernel p-value equals an independent count of the stated rule.

    Draws are integer-valued so ties with the observation occur and the ``>=``
    and ``<=`` boundaries (and the absolute value for two-sided) are exercised.
    The numpy witness is the docstring definition ``(1 + b) / (1 + B)``; a peer
    that doubles the smaller tail would disagree on the asymmetric draws.
    """
    rng = np.random.default_rng(seed)
    resamples = 40
    null = rng.integers(-4, 6, size=resamples).astype(float)  # asymmetric, tied
    for observed in (-3.0, 0.0, 2.0, 5.0, -7.0):
        result = permutation_test_kernel(
            jnp.asarray(observed), jnp.asarray(null), resamples, alternative
        )
        expected = _witness_pvalue(observed, null, alternative, resamples)
        np.testing.assert_allclose(float(result.pvalue), expected, rtol=1e-12)
        np.testing.assert_array_equal(np.asarray(result.null_distribution), null)


def _two_sample_statistic(values: np.ndarray, group: tuple[int, ...]) -> float:
    """Difference in group means for the relabeling ``group`` (first sample)."""
    mask = np.zeros(values.size, dtype=bool)
    mask[list(group)] = True
    return float(values[mask].mean() - values[~mask].mean())


_ENUM_VALUES = np.array([1.0, 4.0, 2.0, 8.0, 3.0, 9.0, 7.0, 5.0])
_ENUM_OBSERVED_GROUP = (0, 1, 2, 3)


def _enumerated_split() -> tuple[float, np.ndarray]:
    """Observed statistic and the other 69 of the 70 relabelings of 4+4."""
    observed = _two_sample_statistic(_ENUM_VALUES, _ENUM_OBSERVED_GROUP)
    others = [
        _two_sample_statistic(_ENUM_VALUES, group)
        for group in combinations(range(8), 4)
        if group != _ENUM_OBSERVED_GROUP
    ]
    assert len(others) == 69  # noqa: PLR2004
    return observed, np.asarray(others)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_kernel_matches_itertools_exact_enumeration_of_4_plus_4(
    alternative: Literal["two-sided", "greater", "less"],
) -> None:
    """Supplying the other 69 relabelings makes the add-one p-value exact.

    With the observed arrangement counted once by the ``+1`` and the remaining
    ``B = 69`` of the ``C(8, 4) = 70`` relabelings as null draws, ``(1 + b) /
    (1 + B) = (1 + b) / 70`` is the exact enumeration p-value ``#{extreme of
    all 70} / 70``. The witness enumerates all 70 splits with ``itertools``
    directly, without the kernel.
    """
    observed, others = _enumerated_split()
    everything = np.append(others, observed)  # all 70 relabelings
    if alternative == "two-sided":
        exact = np.sum(np.abs(everything) >= abs(observed)) / 70
    elif alternative == "greater":
        exact = np.sum(everything >= observed) / 70
    else:
        exact = np.sum(everything <= observed) / 70

    result = permutation_test_kernel(
        jnp.asarray(observed), jnp.asarray(others), 69, alternative
    )
    np.testing.assert_allclose(float(result.pvalue), exact, rtol=1e-12)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.requires_optional("scipy")
@pytest.mark.external_reference
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_kernel_matches_scipy_exact_permutation_test_where_definitions_coincide(
    alternative: Literal["two-sided", "greater", "less"],
) -> None:
    """The kernel agrees with ``scipy.stats.permutation_test`` on exact enumeration.

    The definitions coincide only when scipy enumerates every relabeling
    (``n_resamples=np.inf``, giving ``#{extreme} / 70`` with the observed
    arrangement among the 70) and the kernel is fed the other 69 relabelings, so
    that its ``+1`` stands for the observed arrangement. scipy's randomized mode
    uses ``(1 + b) / (1 + B)`` over *random* draws that may repeat the observed
    arrangement, and its two-sided rule is not guaranteed to be
    ``abs(null) >= abs(observed)`` across versions, so those are not compared.
    The values below are integers-over-a-small-grid, so tie handling is exact.
    """
    observed, others = _enumerated_split()
    first = _ENUM_VALUES[list(_ENUM_OBSERVED_GROUP)]
    second = np.delete(_ENUM_VALUES, list(_ENUM_OBSERVED_GROUP))

    def mean_difference(x: np.ndarray, y: np.ndarray, axis: int) -> np.ndarray:
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    reference = stats.permutation_test(
        (first, second),
        mean_difference,
        permutation_type="independent",
        n_resamples=np.inf,
        alternative=alternative,
        vectorized=True,
    )
    result = permutation_test_kernel(
        jnp.asarray(observed), jnp.asarray(others), 69, alternative
    )
    np.testing.assert_allclose(float(reference.statistic), observed, rtol=1e-12)
    np.testing.assert_allclose(float(result.pvalue), reference.pvalue, rtol=1e-12)
