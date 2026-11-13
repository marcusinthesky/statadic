"""Contracts for rank-clean and symmetric dyadic design operators."""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, Any, override

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.operators.design import (
    NodeIndex,
    _rank_profile_gram_schmidt,
    drop_collinear_columns,
    fwl_coefficient,
    symmetric_dyadic_effects,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from jax.typing import DTypeLike


_RANK_TOL = 1e-10


class _CompileCounter(logging.Handler):
    """Count XLA compilations emitted under :func:`jax.log_compiles`."""

    def __init__(self) -> None:
        super().__init__()
        self.compiles = 0

    @override
    def emit(self, record: logging.LogRecord) -> None:
        if "Compiling" in record.getMessage():
            self.compiles += 1


def _rank_profile_greedy(xf: jax.Array, threshold: float) -> list[int]:
    """Return the declared-order rank profile by masked greedy rank growth.

    Keep a column when adding it to the already-kept prefix raises the rank.
    That definition, not any particular factorization, is the contract.

    Columns are masked to zero rather than sliced out.  Zeroing column ``k``
    leaves ``xf @ xf.T`` -- and therefore every nonzero singular value --
    identical to deleting it, so the rank is exact.  Holding the width fixed is
    what makes it *fast*: the sliced candidate changed shape on every
    iteration, so the eager ``matrix_rank`` kernel missed its shape-keyed cache
    and recompiled once per column.  Measured on one rank-deficient 1326x74
    dyadic design, sliced against masked: 26.4 s and 742 compilations against
    0.67 s and 4, falling to 0.48 s and 0 once the two shapes are warm.

    Rank decisions use the supplied design precision.
    """
    width = xf.shape[1]
    flags = [0.0] * width
    keep: list[int] = []
    rank = 0
    for column in range(width):
        flags[column] = 1.0
        candidate = xf * jnp.asarray(flags, dtype=xf.dtype)[None, :]
        candidate_rank = int(jnp.linalg.matrix_rank(candidate, tol=threshold))
        if candidate_rank > rank:
            keep.append(column)
            rank = candidate_rank
        else:
            flags[column] = 0.0
    return keep


def _reference_rank_profile(design: np.ndarray, tol: float = _RANK_TOL) -> list[int]:
    """Return the declared-order rank profile from an independent NumPy oracle.

    Deliberately the *sliced* formulation on NumPy's own SVD: it shares neither
    the array library nor the masking trick with the implementation, so it can
    witness a regression in either.
    """
    threshold = tol * max(float(np.linalg.norm(design, ord=2)), 1.0)
    keep: list[int] = []
    rank = 0
    for column in range(design.shape[1]):
        candidate = design[:, [*keep, column]]
        candidate_rank = int(np.linalg.matrix_rank(candidate, tol=threshold))
        if candidate_rank > rank:
            keep.append(column)
            rank = candidate_rank
    return keep


def _dyadic_effects(nodes: int) -> np.ndarray:
    """Return the undirected endpoint-effect block for a saturated node panel."""
    pairs = list(itertools.combinations(range(nodes), 2))
    effects = np.zeros((len(pairs), nodes))
    for row, (left, right) in enumerate(pairs):
        effects[row, left] += 1.0
        effects[row, right] += 1.0
    return effects


def _rank_profile_cases() -> Iterator[tuple[str, np.ndarray]]:
    """Yield the collinearity shapes a dyadic bootstrap actually produces."""
    rng = np.random.default_rng(7)
    for nodes in (5, 8):
        effects = _dyadic_effects(nodes)
        rows = effects.shape[0]
        intercept = np.ones(rows)
        zeros = np.zeros(rows)
        yield f"dummy-trap-{nodes}", np.column_stack([intercept, effects])
        yield f"identified-{nodes}", np.column_stack([intercept, effects[:, 1:]])
        yield (
            f"trailing-zero-columns-{nodes}",
            np.column_stack([intercept, effects[:, 1:], zeros, zeros]),
        )
        yield (
            f"interleaved-zero-columns-{nodes}",
            np.column_stack(
                [intercept, zeros, effects[:, 1:], zeros, rng.normal(size=(rows, 2))]
            ),
        )
        yield (
            f"duplicated-effects-{nodes}",
            np.column_stack(
                [intercept, effects[:, 1:], effects[:, 2:4], 2.0 * effects[:, 1:2]]
            ),
        )
        yield (
            f"bootstrap-inactive-nodes-{nodes}",
            np.column_stack(
                [intercept, effects[:, 1:] * rng.integers(0, 2, size=nodes - 1)]
            ),
        )
        yield (
            f"named-regressors-first-{nodes}",
            np.column_stack([intercept, rng.normal(size=(rows, 3)), effects[:, 1:]]),
        )
    for index in range(6):
        height, width = 12 + index, 9
        rank = 1 + index % width
        factored = rng.normal(size=(height, rank)) @ rng.normal(size=(rank, width))
        yield f"low-rank-{height}x{width}r{rank}", factored
    # Wide, near-full-row-rank panels with interior dependencies: the family
    # that exposed the QR-diagonal shortcut, and the one a curated dyadic
    # battery does not reach on its own.
    for index in range(6):
        height, width = 8 + index, 22 + index
        rank = height - index % 3
        yield (
            f"wide-interior-dependence-{height}x{width}r{rank}",
            rng.normal(size=(height, rank)) @ rng.normal(size=(rank, width)),
        )
    yield "wider-than-tall", rng.normal(size=(4, 11))


_PROFILE_CASES = list(_rank_profile_cases())


def _threshold(design: jax.Array, tol: float = _RANK_TOL) -> float:
    """Return the absolute rank tolerance the public entry point would use."""
    return tol * max(float(jnp.linalg.norm(design, ord=2)), 1.0)


@pytest.mark.unit
def test_drop_collinear_columns_preserves_declared_order() -> None:
    """Keep the first representative of a collinear column family."""
    x = np.column_stack(
        [
            np.ones(5),
            np.arange(5, dtype=float),
            np.zeros(5),
            2.0 * np.arange(5, dtype=float),
            np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
        ]
    )

    kept = drop_collinear_columns(x)

    assert kept.tolist() == [0, 1, 4]
    assert np.linalg.matrix_rank(x[:, kept]) == len(kept)


@pytest.mark.unit
def test_drop_collinear_columns_handles_integer_design() -> None:
    """Integer input must not truncate the normalized rank basis."""
    x = jnp.asarray([[1, 2], [2, 4], [3, 6]])
    assert drop_collinear_columns(x).tolist() == [0]


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_drop_collinear_columns_uses_dtype_aware_default(dtype: DTypeLike) -> None:
    """Exact dependence agrees with an independent singular-value rank."""
    design = jnp.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=dtype)
    expected_rank = np.linalg.matrix_rank(np.asarray(design, dtype=np.float64))
    kept = drop_collinear_columns(design)
    assert len(kept) == expected_rank
    assert kept.tolist() == [0]


@pytest.mark.unit
@pytest.mark.contract
def test_drop_collinear_columns_respects_explicit_tolerance() -> None:
    """An explicit threshold can retain a near-dependent float64 column."""
    design = jnp.asarray([[1.0, 1.0], [0.0, 1e-8]], dtype=jnp.float64)
    assert drop_collinear_columns(design).tolist() == [0, 1]
    assert drop_collinear_columns(design, tol=1e-6).tolist() == [0]


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize(
    ("label", "design"),
    _PROFILE_CASES,
    ids=[label for label, _ in _PROFILE_CASES],
)
def test_rank_profile_matches_an_independent_oracle(
    label: str, design: np.ndarray
) -> None:
    """Both backends select exactly what NumPy's sliced greedy growth selects.

    The oracle shares neither the array library nor the algorithm with either
    implementation, so it witnesses a regression in the shipped single-pass
    Gram-Schmidt and in the `matrix_rank` reference alike.
    """
    del label
    expected = _reference_rank_profile(design)

    with jax.enable_x64(new_val=True):
        matrix = jnp.asarray(design, dtype=jnp.float64)
        threshold = _threshold(matrix)
        reference = _rank_profile_greedy(matrix, threshold)
        accepted = _rank_profile_gram_schmidt(
            matrix, jnp.asarray(threshold, dtype=jnp.float64)
        )

    assert reference == expected
    assert jnp.flatnonzero(accepted).tolist() == expected
    assert drop_collinear_columns(design).tolist() == expected
    assert np.linalg.matrix_rank(design[:, expected]) == len(expected)


@pytest.mark.unit
@pytest.mark.regression
def test_drop_collinear_columns_handles_degenerate_shapes() -> None:
    """Degenerate designs return empty or singleton profiles, never raise.

    A zero-width design is the regression that matters: `scan` traces its body
    even at zero length, so an unguarded write into an empty basis raises
    instead of yielding the empty profile the callers branch on.
    """
    assert drop_collinear_columns(np.zeros((4, 3))).tolist() == []
    assert drop_collinear_columns(np.zeros((4, 0))).tolist() == []
    assert drop_collinear_columns(np.ones((4, 1))).tolist() == [0]
    assert drop_collinear_columns(np.array([[1.0, 2.0, 0.0]])).tolist() == [0]
    assert drop_collinear_columns(np.zeros((4, 3))).dtype == jnp.int64


@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.regression
def test_gram_schmidt_profile_compiles_once_per_shape() -> None:
    """The rank tolerance must stay a traced argument, not a baked constant.

    As a closed-over Python float the threshold lands in the jaxpr and re-keys
    the compilation cache on every call -- measured 20 compilations across 20
    calls, which gave back most of what the single pass buys. Only a new shape
    may add an entry.
    """
    rng = np.random.default_rng(3)
    design = rng.normal(size=(20, 6))
    counter = _CompileCounter()
    logger = logging.getLogger("jax")
    logger.addHandler(counter)
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        with jax.log_compiles():
            drop_collinear_columns(design)
            after_first = counter.compiles
            for scale in (1.0, 2.0, 7.5):
                drop_collinear_columns(design * scale)
            after_repeats = counter.compiles
            drop_collinear_columns(rng.normal(size=(31, 9)))
            after_new_shape = counter.compiles
    finally:
        logger.removeHandler(counter)
        logger.setLevel(previous_level)

    assert after_first > 0
    assert after_repeats == after_first
    assert after_new_shape > after_repeats


@pytest.mark.unit
@pytest.mark.regression
def test_backends_diverge_only_where_a_singular_value_meets_the_threshold() -> None:
    """Record the one regime where the two predicates disagree.

    Greedy asks whether `sigma_{k+1}` clears the threshold; Gram-Schmidt asks
    whether the orthogonal residual norm does. Since
    `sigma_min([A a]) <= dist(a, span(A))`, Gram-Schmidt is the more permissive,
    and on a spectrum placed deliberately astride the threshold it keeps one
    column more. Real designs are not in this regime -- their collinearity is
    structural, so a singular value is either many orders above the threshold or
    exactly zero -- which is what licenses the faster backend. This test exists
    so the difference stays recorded rather than being rediscovered.
    """
    rng = np.random.default_rng(0)
    left, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    right, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    astride = left[:, :4] @ np.diag(np.logspace(0, -10, 4)) @ right.T

    with jax.enable_x64(new_val=True):
        matrix = jnp.asarray(astride, dtype=jnp.float64)
        threshold = _threshold(matrix)
        reference = _rank_profile_greedy(matrix, threshold)
        accepted = _rank_profile_gram_schmidt(
            matrix, jnp.asarray(threshold, dtype=jnp.float64)
        )

    assert reference == [0, 1, 2]
    assert jnp.flatnonzero(accepted).tolist() == [0, 1, 2, 3]


@pytest.mark.unit
@pytest.mark.regression
def test_rank_profile_survives_zero_column_before_active_columns() -> None:
    """A zero column never suppresses an independent column that follows it."""
    x = np.column_stack(
        [
            np.ones(6),
            np.zeros(6),
            np.arange(6, dtype=float),
            np.zeros(6),
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        ]
    )

    kept = drop_collinear_columns(x)

    assert kept.tolist() == [0, 2, 4]


@pytest.mark.unit
@pytest.mark.regression
def test_neither_qr_variant_can_replace_greedy_rank_growth() -> None:
    """Pin both shortcuts, because both look correct and both lose a column.

    Unpivoted: a dependent column consumes its diagonal row whether or not it
    adds rank, so every later independent column is shifted off the diagonal.
    Pivoted: ``geqp3`` reorders by norm, so it returns *a* maximal independent
    set rather than *the* declared-order one, which would silently drop a named
    regressor in favour of a generated effect that follows it.
    """
    shifted = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    reordered = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    with jax.enable_x64(new_val=True):
        diagonal = jnp.abs(
            jnp.diagonal(jnp.linalg.qr(jnp.asarray(shifted, dtype=jnp.float64), "r"))
        )
        _q, _r, pivots = jax.lax.linalg.qr(
            jnp.asarray(reordered, dtype=jnp.float64),
            full_matrices=False,
            pivoting=True,
        )

    assert drop_collinear_columns(shifted).tolist() == [0, 2]
    assert jnp.flatnonzero(diagonal > _RANK_TOL).tolist() == [0]

    assert drop_collinear_columns(reordered).tolist() == [0, 1]
    assert sorted(np.asarray(pivots)[:2].tolist()) == [0, 2]


@pytest.mark.unit
@pytest.mark.regression
def test_greedy_rank_profile_holds_its_candidate_shape_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Masking, not slicing, is what keeps the loop off the compile path.

    Slicing the candidate gave `matrix_rank` a new shape on every column, so
    the eager kernel missed its shape-keyed cache and recompiled once per
    column. On one production 1326x74 dyadic design that cost 74 `matrix_rank`
    compilations (10.1 s) plus 130 more for the reslicing gather (10.3 s).
    """
    observed: list[tuple[int, ...]] = []
    original = jnp.linalg.matrix_rank

    def _record(matrix: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        observed.append(tuple(matrix.shape))
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(jnp.linalg, "matrix_rank", _record)

    effects = _dyadic_effects(6)
    design = np.column_stack([np.ones(effects.shape[0]), effects])
    with jax.enable_x64(new_val=True):
        matrix = jnp.asarray(design, dtype=jnp.float64)
        _rank_profile_greedy(matrix, _threshold(matrix))

    assert len(observed) == design.shape[1]
    assert set(observed) == {design.shape}


@pytest.mark.unit
@pytest.mark.property
def test_symmetric_dyadic_effects_are_endpoint_order_invariant() -> None:
    """Swapping endpoints leaves every undirected effect row unchanged."""
    endpoint_i = ("a", "a", "b")
    endpoint_j = ("b", "c", "c")

    effects, names = symmetric_dyadic_effects(endpoint_i, endpoint_j)
    swapped, swapped_names = symmetric_dyadic_effects(endpoint_j, endpoint_i)

    assert names == ["node_effect[b]", "node_effect[c]"]
    assert swapped_names == names
    np.testing.assert_array_equal(swapped, effects)


@pytest.mark.component
@pytest.mark.contract
def test_node_index_canonicalises_unlike_hashable_id_types() -> None:
    """Canonical effect columns do not require cross-type comparison operators."""
    endpoint_i = (1, "a", 1)
    endpoint_j = ("a", 2, 2)

    direct = NodeIndex.from_endpoints(endpoint_i, endpoint_j)
    swapped = NodeIndex.from_endpoints(endpoint_j, endpoint_i)

    assert direct == swapped
    assert direct.node_ids == (1, 2, "a")
    np.testing.assert_array_equal(
        direct.positions_for(endpoint_i),
        np.array([0, 2, 0]),
    )


@pytest.mark.unit
@pytest.mark.contract
def test_symmetric_dyadic_effect_names_are_adapter_configurable() -> None:
    """A domain adapter can retain its own effect-label vocabulary."""
    endpoint_i = ("a", "a", "b")
    endpoint_j = ("b", "c", "c")

    _effects, names = symmetric_dyadic_effects(
        endpoint_i,
        endpoint_j,
        name_prefix="firm_fe",
    )

    assert names == ["firm_fe[b]", "firm_fe[c]"]


@pytest.mark.unit
@pytest.mark.reference
def test_fwl_coefficient_matches_full_ols() -> None:
    """Residualization reproduces the corresponding full-OLS coefficient."""
    nuisance = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    focal = np.array([0.2, 0.8, 0.1, 1.3, 0.7])
    x = np.column_stack([np.ones(focal.size), focal, nuisance])
    y = 0.4 + 1.7 * focal - 0.3 * nuisance

    expected = np.linalg.lstsq(x, y, rcond=None)[0][1]

    assert fwl_coefficient(x, y, 1) == pytest.approx(expected, abs=1e-12)
