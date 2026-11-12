# standalone package-maintenance entry point
"""``statadic.inference.mantel`` and the Mantel workflow — statistic, null and p-value.

The p-value convention (one-sided right tail, add-one smoothed) is pinned in its
own section at the bottom rather than assumed, because both halves of it are
choices a reimplementation could silently change while every correlation
assertion above kept passing.
"""

from __future__ import annotations

import subprocess
import sys
from itertools import permutations
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy import stats
from statadic.api.association import (
    Mantel,
    MantelInput,
    MantelTestEvidence,
    MatrixExchangeability,
)
from statadic.api.comparison import PermutationPlan
from statadic.api.hypotheses import Hypothesis
from statadic.association.mantel import mantel_correlation_kernel
from statadic.inference.mantel import (
    PairedMantelBootstrapResult,
    mantel_bootstrap_ci,
    mantel_bootstrap_kernel,
    mantel_permutation_kernel,
    paired_mantel_bootstrap,
)
from statadic.inference.resampling.permutation import permutation_test_kernel

from .._budgets import examples
from .._strategies import float_arrays
from .._typing import ill_typed

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy
    from jax import Array
    from statadic.api.association import MantelResult

_SEEDS = st.integers(0, 2**31 - 1)
_RAMP_SPAN = 10.0


def _mantel_evidence(result: MantelResult, plan: PermutationPlan) -> MantelTestEvidence:
    """Retain native schedules after testing explicit object exchangeability."""
    outcome = result.infer(resampling=plan).test(Hypothesis(MatrixExchangeability()))
    assert outcome.evidence is not None
    return outcome.evidence


def distance_matrices(n: int) -> SearchStrategy[np.ndarray]:
    """Draw Euclidean distance matrices of ``n`` distinct points in the plane.

    A ramp on the first coordinate, wider than the generated jitter, keeps the
    points distinct by construction, so the off-diagonal entries are never
    constant and every correlation is defined.

    Args:
        n: Number of nodes, fixed per test.

    Returns:
        A strategy for symmetric hollow float64 matrices.

    """

    def to_distances(jitter: Array) -> np.ndarray:
        points = np.asarray(jitter).copy()
        points[:, 0] += np.linspace(0.0, _RAMP_SPAN, n)
        diff = points[:, None, :] - points[None, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))

    return float_arrays((n, 3), bound=1.5).map(to_distances)


def _is_not_constant(values: np.ndarray) -> bool:
    """Report whether a vector has a defined correlation (it varies)."""
    return bool(np.ptp(values) > 0)


def node_resamples(n: int, draws: int) -> SearchStrategy[np.ndarray]:
    """Draw ``(draws, n)`` node resamples whose first row is the identity.

    The identity row is a nondegenerate draw by construction, so a bootstrap
    summary always has at least one usable replicate.

    Args:
        n: Number of nodes.
        draws: Number of resamples.

    Returns:
        A strategy for integer index arrays.

    """
    return hnp.arrays(np.int64, (draws, n), elements=st.integers(0, n - 1)).map(
        lambda rows: np.vstack([np.arange(n), rows[1:]])
    )


class _Outcome(NamedTuple):
    """Observed coefficient and add-one calibration of the workflow door."""

    correlation: float
    pvalue: float
    method: str
    num_permutations: int
    null_distribution: jax.Array


def _mantel(
    x: np.ndarray,
    y: np.ndarray,
    method: Literal["pearson", "spearman"] = "pearson",
    num_permutations: int = 10_000,
    seed: int = 42,
) -> _Outcome:
    """Run the native-stream random Mantel relabeling through the workflow."""
    fit = Mantel(method).estimate(MantelInput(x, y))
    inference = _mantel_evidence(fit, PermutationPlan(n=num_permutations, seed=seed))
    assert inference.pvalue is not None
    return _Outcome(
        float(fit.statistic), inference.pvalue, method, num_permutations, inference.null
    )


def _hollow_matrix(triangle: list[float]) -> np.ndarray:
    """Build a labeled four-node matrix from six literal edge values."""
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[np.triu_indices(4, 1)] = triangle
    return matrix + matrix.T


def _reference_correlation(
    left: np.ndarray, right: np.ndarray, method: Literal["pearson", "spearman"]
) -> float:
    """Calculate the independent compact-vector SciPy correlation."""
    result = (
        stats.pearsonr(left, right)
        if method == "pearson"
        else stats.spearmanr(left, right)
    )
    return float(result.statistic)


def _reference_compacted_correlations(
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray],
    nodes: np.ndarray,
    method: Literal["pearson", "spearman"],
) -> list[float]:
    """Remove resampled self-edges before independently ranking/correlating."""
    triangle = np.triu_indices(len(nodes), 1)
    rows, columns = nodes[triangle[0]], nodes[triangle[1]]
    distinct = rows != columns
    compact = [matrix[rows[distinct], columns[distinct]] for matrix in matrices]
    return [
        _reference_correlation(compact[arm], compact[2], method) for arm in range(2)
    ]


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_all_node_permutations_match_independent_tied_mantel_null(
    method: Literal["pearson", "spearman"],
) -> None:
    """Enumerate node relabelings rather than exchanging individual edges."""
    x = _hollow_matrix([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    y = _hollow_matrix([3.0, 1.0, 2.0, 4.0, 1.0, 1.0])
    schedule = np.asarray(list(permutations(range(4))), dtype=np.int32)
    triangle = np.triu_indices(4, 1)
    expected = np.asarray(
        [
            _reference_correlation(x[triangle], y[np.ix_(rows, rows)][triangle], method)
            for rows in schedule
        ]
    )
    expected_extreme_count = np.count_nonzero(expected >= expected[0])

    observed, null = mantel_permutation_kernel(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(schedule), method
    )
    calibrated = permutation_test_kernel(observed, null, len(schedule), "greater")

    # Float64 correlations lie in [-1, 1]; this permits reduction roundoff,
    # while the exact tail-count assertion separately protects ties.
    np.testing.assert_allclose(observed, expected[0], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(null, expected, rtol=1e-13, atol=1e-13)
    assert np.isfinite(expected).all()
    assert (
        np.count_nonzero(np.asarray(null) >= float(observed)) == expected_extreme_count
    )
    assert np.count_nonzero(expected == expected[0]) > 1
    # Exhaustive input does not change the numerical kernel's Monte Carlo rule.
    assert float(calibrated.pvalue) == pytest.approx(
        (1 + expected_extreme_count) / (1 + len(schedule))
    )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
@pytest.mark.parametrize("method", ["pearson", "spearman"])
@pytest.mark.parametrize("chunk", [1, 4, 64])
def test_paired_mantel_bootstrap_matches_compacted_node_draws(
    method: Literal["pearson", "spearman"], chunk: int
) -> None:
    """Duplicated nodes and invalid draws retain their declared paired policy."""
    focal = _hollow_matrix([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    comparator = _hollow_matrix([3.0, 1.0, 2.0, 4.0, 1.0, 1.0])
    target = _hollow_matrix([2.0, 4.0, 1.0, 1.0, 3.0, 2.0])
    indices = np.asarray(
        [
            [0, 1, 2, 3],
            [0, 0, 1, 2],
            [2, 3, 1, 1],
            [3, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 1],
        ],
        dtype=np.int32,
    )
    expected_values = np.asarray(
        [
            _reference_compacted_correlations(
                (focal, comparator, target), nodes, method
            )
            for nodes in indices[:4]
        ]
    )
    expected_gaps = expected_values[:, 1] - expected_values[:, 0]
    expected_interval = np.percentile(expected_gaps, [10, 90], method="linear")

    values, valid = mantel_bootstrap_kernel(
        jnp.asarray(np.stack([focal, comparator, target])),
        jnp.asarray(indices),
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray([2, 2], dtype=jnp.int32),
        method,
        chunk=chunk,
    )
    paired = paired_mantel_bootstrap(
        focal,
        comparator,
        target,
        indices,
        method,
        lower_percentile=10.0,
        upper_percentile=90.0,
    )
    interval = mantel_bootstrap_ci(
        focal, target, indices, method, lower_percentile=10.0, upper_percentile=90.0
    )

    np.testing.assert_array_equal(valid, [1, 1, 1, 1, 0, 0])
    assert values.shape == (len(indices), 2)
    assert np.isfinite(expected_values).all()
    np.testing.assert_allclose(values[:4], expected_values, rtol=1e-13, atol=1e-13)
    assert paired.gap_n_boot == len(expected_gaps)
    assert paired.gap_mean == pytest.approx(expected_gaps.mean(), abs=1e-13)
    np.testing.assert_allclose(
        [paired.gap_ci_low, paired.gap_ci_high],
        expected_interval,
        rtol=1e-13,
        atol=1e-13,
    )
    assert paired.gap_prob_le_zero == np.mean(expected_gaps <= 0)
    np.testing.assert_allclose(
        interval,
        np.percentile(expected_values[:, 0], [10, 90], method="linear"),
        rtol=1e-13,
        atol=1e-13,
    )


def _random_distance_matrix(rng: np.random.Generator, n: int, d: int = 3) -> np.ndarray:
    points = rng.normal(size=(n, d))
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


@pytest.mark.component
@pytest.mark.numerical
@examples(10)
@example(x=_random_distance_matrix(np.random.default_rng(0), n=10))
@given(x=distance_matrices(10))
def test_self_correlation_is_one(x: np.ndarray) -> None:
    """Verify self correlation is one.

    Args:
        x: Generated distance matrix.

    """
    result = _mantel(x, x, num_permutations=200)
    assert result.correlation == pytest.approx(1.0, abs=1e-10)


@pytest.mark.component
def test_independent_matrices_not_strongly_correlated() -> None:
    """Distance matrices from independent point sets should show weak correlation.

    Checks the correlation magnitude rather than the permutation p-value
    directly, since a strict ``pvalue > alpha`` assertion is a type-I-error
    boundary check and would flake at the nominal rate under a true null.
    """
    rng = np.random.default_rng(2)
    x = _random_distance_matrix(rng, n=20)
    y = _random_distance_matrix(rng, n=20)
    result = _mantel(x, y, num_permutations=500, seed=2)
    weak_correlation_limit = 0.5
    assert abs(result.correlation) < weak_correlation_limit


@pytest.mark.component
@pytest.mark.numerical
@examples(10)
@example(x=_random_distance_matrix(np.random.default_rng(3), n=15))
@given(x=distance_matrices(15))
def test_spearman_method(x: np.ndarray) -> None:
    """Verify spearman method.

    Args:
        x: Generated distance matrix.

    """
    result = _mantel(x, x, method="spearman", num_permutations=100)
    assert result.method == "spearman"
    assert result.correlation == pytest.approx(1.0, abs=1e-10)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.requires_optional("scipy")
@pytest.mark.external_reference
@examples(100)
@example(
    x=np.array([1.0, 2.0, 2.0, 4.0, 5.0]),
    y=np.array([3.0, 1.0, 1.0, 4.0, 6.0]),
    method="pearson",
)
@example(
    x=np.array([1.0, 2.0, 2.0, 4.0, 5.0]),
    y=np.array([3.0, 1.0, 1.0, 4.0, 6.0]),
    method="spearman",
)
@given(
    x=hnp.arrays(np.float64, 5, elements=st.integers(0, 4).map(float)).filter(
        _is_not_constant
    ),
    y=hnp.arrays(np.float64, 5, elements=st.integers(0, 4).map(float)).filter(
        _is_not_constant
    ),
    method=st.sampled_from(["pearson", "spearman"]),
)
def test_mantel_jax_correlation_matches_scipy_with_ties(
    x: np.ndarray,
    y: np.ndarray,
    method: Literal["pearson", "spearman"],
) -> None:
    """Pin the retired SciPy semantics, including average ranks for ties.

    Integer-valued draws from a small range make ties common; constant vectors
    have no correlation, so they are outside the domain.

    Args:
        x: Generated vector with ties.
        y: Generated vector with ties.
        method: Correlation method.

    """
    expected = (
        stats.pearsonr(x, y).statistic
        if method == "pearson"
        else stats.spearmanr(x, y).statistic
    )

    actual = mantel_correlation_kernel(jnp.asarray(x), jnp.asarray(y), method)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.integration
@pytest.mark.contract
@examples(10)
@example(
    x_np=_random_distance_matrix(np.random.default_rng(31), n=6),
    y_np=_random_distance_matrix(np.random.default_rng(32), n=6),
)
@given(x_np=distance_matrices(6), y_np=distance_matrices(6))
def test_mantel_permutation_kernel_jit_vmap_and_jaxpr(
    x_np: np.ndarray, y_np: np.ndarray
) -> None:
    """Keep the complete relabeling null on graph with static result shape.

    Args:
        x_np: Generated distance matrix.
        y_np: Generated distance matrix.

    """
    x = jnp.asarray(x_np)
    y = jnp.asarray(y_np)
    relabelings = jnp.stack([jnp.arange(6), jnp.arange(5, -1, -1)])

    def compute(
        left: jax.Array,
        right: jax.Array,
        draws: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return mantel_permutation_kernel(left, right, draws, "pearson")

    observed, null = compute(x, y, relabelings)
    compiled = jax.jit(compute)(x, y, relabelings)
    triangle = jnp.triu_indices(6, 1)
    batched = jax.vmap(mantel_correlation_kernel)(
        jnp.stack([x[triangle], y[triangle]]),
        jnp.stack([y[triangle], x[triangle]]),
    )

    np.testing.assert_allclose(compiled[0], observed)
    np.testing.assert_allclose(compiled[1], null)
    assert null.shape == (2,)
    assert batched.shape == (2,)
    assert "host_callback" not in str(jax.make_jaxpr(compute)(x, y, relabelings))


@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.subprocess
def test_invalid_method_is_rejected_without_test_instrumentation() -> None:
    """Plain production imports reject unsupported eager and compiled methods."""
    source = """
import jax  # noqa: PLC0415, TC002
import jax.numpy as jnp
from statadic.api.association import Mantel, MantelInput
from statadic.inference.mantel import mantel_permutation_kernel
from statadic.association.mantel import (
    mantel_correlation_kernel,
)

def mantel_fit(x, y, method="pearson"):
    return Mantel(method).estimate(MantelInput(x, y))

vector = jnp.array([1.0, 2.0, 3.0])
matrix = jnp.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
draws = jnp.array([[0, 1, 2], [2, 1, 0]])
for disable_jit in (True, False):
    with jax.disable_jit(disable_jit):
        for function, arguments in (
            (mantel_correlation_kernel, (vector, vector)),
            (mantel_permutation_kernel, (matrix, matrix, draws)),
            (mantel_fit, (matrix, matrix)),
        ):
            try:
                function(*arguments, method="kendall")
            except ValueError as error:
                assert "Unknown method" in str(error)
            else:
                raise AssertionError(
                    f"{function.__name__} accepted kendall; disable_jit={disable_jit}"
                )
"""
    completed = subprocess.run(  # Fixed interpreter and local source.  # noqa: S603
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.component
@pytest.mark.contract
@examples(10)
@example(
    x=_random_distance_matrix(np.random.default_rng(4), n=8),
    y=_random_distance_matrix(np.random.default_rng(14), n=8),
)
@given(x=distance_matrices(8), y=distance_matrices(8))
def test_null_distribution_length(x: np.ndarray, y: np.ndarray) -> None:
    """Verify null distribution length.

    Args:
        x: Generated distance matrix.
        y: Generated distance matrix.

    """
    expected_permutations = 333
    result = _mantel(x, y, num_permutations=expected_permutations)
    assert result.num_permutations == expected_permutations
    assert result.null_distribution.shape == (expected_permutations,)


@pytest.mark.component
@pytest.mark.property
@examples(10)
@example(
    x=_random_distance_matrix(np.random.default_rng(5), n=10),
    y=_random_distance_matrix(np.random.default_rng(15), n=10),
    seed=7,
)
@given(x=distance_matrices(10), y=distance_matrices(10), seed=_SEEDS)
def test_reproducible_with_same_seed(x: np.ndarray, y: np.ndarray, seed: int) -> None:
    """Verify reproducible with same seed.

    Args:
        x: Generated distance matrix.
        y: Generated distance matrix.
        seed: The repeated seed.

    """
    r1 = _mantel(x, y, num_permutations=100, seed=seed)
    r2 = _mantel(x, y, num_permutations=100, seed=seed)
    assert np.array_equal(r1.null_distribution, r2.null_distribution)
    assert r1.pvalue == r2.pvalue


@pytest.mark.component
@pytest.mark.contract
@examples(10)
@example(
    matrix=_random_distance_matrix(np.random.default_rng(17), n=8),
    indices=np.random.default_rng(17).integers(0, 8, size=(40, 8)),
)
@given(matrix=distance_matrices(8), indices=node_resamples(8, 40))
def test_mantel_bootstrap_ci_uses_caller_owned_node_schedule(
    matrix: np.ndarray, indices: np.ndarray
) -> None:
    """Self-correlation stays one under every nondegenerate node resample.

    Args:
        matrix: Generated distance matrix.
        indices: Generated node resamples, the first being the identity.

    """
    lower, upper = mantel_bootstrap_ci(matrix, matrix, indices)

    assert lower == pytest.approx(1.0, abs=1e-12)
    assert upper == pytest.approx(1.0, abs=1e-12)


@pytest.mark.component
@pytest.mark.property
@examples(10)
@example(
    focal=_random_distance_matrix(np.random.default_rng(18), n=9),
    target=_random_distance_matrix(np.random.default_rng(28), n=9),
    indices=np.random.default_rng(38).integers(0, 9, size=(50, 9)),
)
@given(
    focal=distance_matrices(9),
    target=distance_matrices(9),
    indices=node_resamples(9, 50),
)
def test_paired_mantel_bootstrap_uses_the_same_draw_for_both_arms(
    focal: np.ndarray, target: np.ndarray, indices: np.ndarray
) -> None:
    """Identical focal/comparator matrices have an exactly zero paired gap.

    Args:
        focal: Generated distance matrix, also used as the comparator.
        target: Generated target distance matrix.
        indices: Generated node resamples, the first being the identity.

    """
    result = paired_mantel_bootstrap(focal, focal.copy(), target, indices)

    assert isinstance(result, PairedMantelBootstrapResult)
    assert result.gap_n_boot > 0
    assert result.gap_mean == pytest.approx(0.0, abs=1e-12)
    assert result.gap_ci_low == pytest.approx(0.0, abs=1e-12)
    assert result.gap_ci_high == pytest.approx(0.0, abs=1e-12)


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(
    matrix=_random_distance_matrix(np.random.default_rng(19), n=5),
    bad=5,
    position=4,
)
@given(
    matrix=distance_matrices(5),
    bad=st.one_of(st.integers(5, 10**6), st.integers(-(10**6), -1)),
    position=st.integers(0, 4),
)
def test_mantel_bootstrap_rejects_an_invalid_node_schedule(
    matrix: np.ndarray, bad: int, position: int
) -> None:
    """Reject indices outside the matrix's node domain before resampling.

    Args:
        matrix: Generated distance matrix of five nodes.
        bad: An index outside ``[0, 5)``.
        position: Where in the draw the invalid index sits.

    """
    indices = np.arange(5)[None, :].copy()
    indices[0, position] = bad

    with pytest.raises(ValueError, match="values in"):
        mantel_bootstrap_ci(matrix, matrix, indices)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.needs_import_hook
def test_invalid_method_raises() -> None:
    """Reject an unsupported correlation method, in either enforcement plane.

    Two spellings of one contract, because the two planes disagree about which
    guard fires first. This session installs the jaxtyping/beartype import hook
    (``conftest.py``), so ``MantelMethod = Literal['pearson', 'spearman']`` is
    enforced on entry and raises ``jaxtyping.TypeCheckError`` — a ``TypeError``
    — before the body runs. Every production caller imports ``statadic`` *without*
    the hook, where ``_require_method``'s ``ValueError`` is the only thing
    standing between an untyped caller and a silent wrong answer. t46.6 kept
    the runtime guard for exactly that reason, and this assertion accepts
    whichever of the two is live.

    That guard used to live in the private ``_correlate`` helper, which
    the retired eager Mantel door never called — so without the hook it returned
    a Pearson coefficient instead of raising. ``_correlate`` died with the
    traced bootstrap rewrite and the guard moved to ``_require_method``, which
    all three public doors now call.
    """
    x = np.eye(4)
    with pytest.raises((TypeError, ValueError), match=r"Unknown method|Literal"):
        _mantel(
            x,
            x,
            method=ill_typed("kendall"),
            num_permutations=10,
        )


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(sizes=(4, 5))
@given(
    sizes=st.tuples(st.integers(2, 8), st.integers(2, 8)).filter(lambda t: t[0] != t[1])
)
def test_mismatched_shape_raises(sizes: tuple[int, int]) -> None:
    """Verify mismatched shape raises.

    Args:
        sizes: Two different square-matrix sizes.

    """
    x = np.zeros((sizes[0], sizes[0]))
    y = np.zeros((sizes[1], sizes[1]))
    with pytest.raises(ValueError, match=r"equal shapes|square matrix"):
        _mantel(x, y)


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(shape=(4, 5))
@given(
    shape=st.tuples(st.integers(2, 8), st.integers(2, 8)).filter(lambda t: t[0] != t[1])
)
def test_non_square_raises(shape: tuple[int, int]) -> None:
    """Verify non square raises.

    Args:
        shape: A non-square matrix shape.

    """
    x = np.zeros(shape)
    y = np.zeros(shape)
    with pytest.raises(ValueError, match=r"equal shapes|square matrix"):
        _mantel(x, y)


@pytest.mark.component
@pytest.mark.contract
@examples(10)
@example(x=_random_distance_matrix(np.random.default_rng(6), n=5))
@given(x=distance_matrices(5))
def test_returns_named_tuple_fields(x: np.ndarray) -> None:
    """Verify returns named tuple fields.

    Args:
        x: Generated distance matrix.

    """
    result = _mantel(x, x, num_permutations=20)
    fit = Mantel().estimate(MantelInput(x, x))
    inference = _mantel_evidence(fit, PermutationPlan(n=20))
    assert isinstance(inference, MantelTestEvidence)
    assert result.method == "pearson"
    assert inference.pvalue is not None


# --- the p-value convention -----------------------------------------------
#
# One-sided right-tail with add-one smoothing, matching `methodology.typ`
# eq-mantel-pvalue (a directional positive-association hypothesis). The three
# tests below pin the two tails and the floor: a two-sided implementation would
# still pass "small p for positive association" but would fail the p≈1 case, and
# an unsmoothed one would return exactly 0 rather than 1/(1+B).
# (t49 B2: from `test_mantel_pvalue_convention.py`, whose name described a
# property rather than the module it covers.)


@pytest.mark.component
def test_strongly_positive_association_gives_small_pvalue() -> None:
    """Verify strongly positive association gives small pvalue."""
    rng = np.random.default_rng(2)
    x = _random_distance_matrix(rng, n=25)
    noise = rng.normal(scale=0.01, size=x.shape)
    y = x + 0.5 * (noise + noise.T)
    result = _mantel(x, y, num_permutations=500, seed=2)
    strong_correlation = 0.9
    significance_level = 0.05
    assert result.correlation > strong_correlation
    assert result.pvalue < significance_level


@pytest.mark.component
def test_strongly_negative_association_gives_pvalue_near_one() -> None:
    """Verify strongly negative association gives pvalue near one."""
    rng = np.random.default_rng(3)
    x = _random_distance_matrix(rng, n=25)
    y = -x + x.max()
    np.fill_diagonal(y, 0.0)
    result = _mantel(x, y, num_permutations=500, seed=3)
    strong_negative_correlation = -0.9
    assert result.correlation < strong_negative_correlation
    assert result.pvalue == pytest.approx(1.0, abs=1e-6)


@pytest.mark.component
@pytest.mark.numerical
def test_pvalue_floor_when_no_null_exceeds_observed() -> None:
    """When no permuted null correlation is >= observed, p is exactly the floor."""
    rng = np.random.default_rng(4)
    x = _random_distance_matrix(rng, n=20)
    noise = rng.normal(scale=1e-4, size=x.shape)
    y = x + (noise + noise.T)
    num_permutations = 200
    result = _mantel(x, y, num_permutations=num_permutations, seed=4)
    assert np.sum(result.null_distribution >= result.correlation) == 0
    assert result.pvalue == pytest.approx(1.0 / (1 + num_permutations))
