"""Independent multinomial node-multiplicity and retained-draw witnesses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.inference.dyadic import (
    DyadicBootstrapDesign,
    NodeCountSchedule,
    bootstrap_coefficient_inference,
    bootstrap_dyadic,
)
from statadic.model.dyadic import DyadicSample
from statadic.operators.design import NodeIndex

from .._budgets import examples

if TYPE_CHECKING:
    from collections.abc import Hashable

_MIN_ACTIVE_DYADS = 3
# Below float32 range, draws underflow; below this spread the standard error is nil.
_MIN_DRAW_MAGNITUDE = 1e-6
_MIN_DRAW_SPREAD = 1e-2


def _case() -> tuple[np.ndarray, DyadicSample, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1102)
    left, right = np.triu_indices(6, k=1)
    labels = np.array(list("abcdef"))
    focal = rng.normal(size=len(left))
    error = rng.normal(size=6)
    y = (
        0.4
        + 0.7 * focal
        + error[left]
        + error[right]
        + rng.normal(size=len(left)) * 0.2
    )
    counts = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [2, 0, 1, 1, 2, 0],
            [6, 0, 0, 0, 0, 0],
            [3, 3, 0, 0, 0, 0],
            [2, 2, 2, 0, 0, 0],
        ]
    )
    return (
        np.column_stack([np.ones(len(left)), focal]),
        DyadicSample.from_arrays(y, labels[left], labels[right]),
        counts,
        labels,
    )


def _rank_profile(matrix: np.ndarray) -> list[int]:
    """Independent declared-order NumPy rank on the well-separated domain."""
    threshold = 1e-10 * max(np.linalg.norm(matrix, ord=2), 1)
    selected: list[int] = []
    for column in range(matrix.shape[1]):
        if np.linalg.matrix_rank(matrix[:, [*selected, column]], tol=threshold) > len(
            selected
        ):
            selected.append(column)
    return selected


_ORIGINAL_COUNTS = _case()[2]
_N_NODES = _ORIGINAL_COUNTS.shape[1]
# Each draw is a multinomial resample: n_nodes picks, tallied per node, so every
# row sums to the sampled node count as the schedule's documented domain requires.
_COUNTS = hnp.arrays(
    np.int64,
    (*_ORIGINAL_COUNTS.shape[:1], _N_NODES),
    elements=st.integers(0, _N_NODES - 1),
).map(lambda picks: np.stack([np.bincount(row, minlength=_N_NODES) for row in picks]))


@pytest.mark.component
@pytest.mark.reference
@examples(20)
@example(counts=_ORIGINAL_COUNTS, node_effects=False)
@example(counts=_ORIGINAL_COUNTS, node_effects=True)
@given(counts=_COUNTS, node_effects=st.booleans())
def test_supplied_node_counts_match_independent_expanded_edge_refits(  # noqa: PLR0914
    counts: np.ndarray, *, node_effects: bool
) -> None:
    """Replicate each distinct-node edge count_i*count_j times and retain failures.

    Args:
        counts: Generated multinomial node-multiplicity schedule, one draw per row.
        node_effects: Whether node dummies enter the design.

    """
    design, sample, _, labels = _case()
    names = ["intercept", "focal"]
    if node_effects:
        dummy = np.column_stack(
            [
                (np.asarray(sample.endpoint_i) == label).astype(float)
                + (np.asarray(sample.endpoint_j) == label).astype(float)
                for label in labels[1:]
            ]
        )
        design = np.column_stack([design, dummy])
        names.extend(f"node_effect[{label}]" for label in labels[1:])
    index = NodeIndex(tuple(labels.tolist()))
    actual = bootstrap_dyadic(
        DyadicBootstrapDesign(
            design,
            names,
            sample,
            NodeCountSchedule(index, counts),
            node_effects=node_effects,
            required_columns=("focal",),
        )
    )
    np.testing.assert_array_equal(actual.draw_id, np.arange(len(counts)))
    np.testing.assert_array_equal(actual.active_nodes, np.count_nonzero(counts, axis=1))
    for draw, node_counts in enumerate(counts):
        multiplicity = np.array(
            [
                node_counts[list(labels).index(left)]
                * node_counts[list(labels).index(right)]
                for left, right in zip(
                    sample.endpoint_i, sample.endpoint_j, strict=True
                )
            ]
        )
        active = np.count_nonzero(multiplicity)
        assert actual.active_dyads[draw] == active
        if active < _MIN_ACTIVE_DYADS:
            assert not actual.solve_ok[draw]
            assert np.isnan(actual.sse[draw])
            assert all(np.isnan(actual.coefficient(name)[draw]) for name in names)
            continue
        kept = _rank_profile(design * np.sqrt(multiplicity)[:, None])
        expanded = np.repeat(np.arange(len(multiplicity)), multiplicity)
        x, y = design[expanded][:, kept], np.asarray(sample.outcome)[expanded]
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        assert actual.solve_ok[draw]
        assert actual.effective_columns[draw] == len(kept)
        assert actual.effective_rank[draw] == np.linalg.matrix_rank(x)
        for column, name in enumerate(names):
            if column in kept:
                assert float(actual.coefficient(name)[draw]) == pytest.approx(
                    beta[kept.index(column)], rel=1e-10, abs=1e-12
                )
            else:
                assert np.isnan(actual.coefficient(name)[draw])
        residual = y - x @ beta
        assert float(actual.sse[draw]) == pytest.approx(residual @ residual, abs=1e-12)
        assert float(actual.overall_r2[draw]) == pytest.approx(
            1 - residual @ residual / np.sum((y - y.mean()) ** 2), abs=1e-12
        )
        if node_effects and active > np.linalg.matrix_rank(x):  # not saturated
            effect_columns = [
                position
                for position, name in enumerate(names)
                if name == "intercept" or name.startswith("node_effect[")
            ]
            effects = design[expanded][:, effect_columns]
            effect_residual = y - effects @ np.linalg.lstsq(effects, y, rcond=None)[0]
            assert float(actual.within_r2[draw]) == pytest.approx(
                1 - residual @ residual / (effect_residual @ effect_residual),
                rel=1e-10,
                abs=1e-12,
            )


@pytest.mark.component
@pytest.mark.property
def test_bijective_mixed_node_relabeling_requires_matching_count_columns() -> None:
    """A non-order-preserving bijection changes the canonical schedule axis."""
    design, sample, counts, labels = _case()
    names = ["intercept", "focal"]
    original = bootstrap_dyadic(
        DyadicBootstrapDesign(
            design,
            names,
            sample,
            NodeCountSchedule(NodeIndex(tuple(labels.tolist())), counts),
        )
    )
    mapping = dict(zip(labels, [("z",), 8, "m", -2, ("a",), "b"], strict=True))
    left, right = (
        np.empty(len(sample.endpoint_i), dtype=object),
        np.empty(len(sample.endpoint_j), dtype=object),
    )
    left[:] = [mapping[value] for value in sample.endpoint_j]
    right[:] = [mapping[value] for value in sample.endpoint_i]
    renamed = DyadicSample.from_arrays(sample.outcome, left, right)
    index = NodeIndex.from_arrays(left, right)
    inverse: dict[Hashable, str] = {value: key for key, value in mapping.items()}
    columns = [list(labels).index(inverse[node]) for node in index.node_ids]
    aligned = bootstrap_dyadic(
        DyadicBootstrapDesign(
            design, names, renamed, NodeCountSchedule(index, counts[:, columns])
        )
    )
    np.testing.assert_array_equal(aligned.draw_id, original.draw_id)
    np.testing.assert_array_equal(aligned.solve_ok, original.solve_ok)
    np.testing.assert_array_equal(aligned.active_dyads, original.active_dyads)
    np.testing.assert_allclose(
        aligned.coefficient("focal"),
        original.coefficient("focal"),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )
    wrong = bootstrap_dyadic(
        DyadicBootstrapDesign(design, names, renamed, NodeCountSchedule(index, counts))
    )
    assert not np.allclose(
        wrong.coefficient("focal"), original.coefficient("focal"), equal_nan=True
    )


@pytest.mark.component
@pytest.mark.reference
@examples(100)
@example(draws=np.array([-2.0, -0.4, 0.2, 0.8, 1.5]), coefficient=0.3)
@given(
    draws=hnp.arrays(
        np.float64,
        5,
        elements=st.floats(-5, 5, allow_nan=False).filter(
            lambda v: v == 0 or abs(v) > _MIN_DRAW_MAGNITUDE
        ),
        unique=True,
    ).filter(lambda d: np.std(d, ddof=1) > _MIN_DRAW_SPREAD),
    coefficient=st.floats(-5, 5, allow_nan=False),
)
def test_node_coefficient_summary_matches_independent_order_statistics(
    draws: np.ndarray, coefficient: float
) -> None:
    """The bootstrap tail and percentile interval use draws, never edge-IID df.

    Args:
        draws: Generated bootstrap coefficient draws with non-degenerate spread.
        coefficient: The point estimate being summarised.

    """
    actual = bootstrap_coefficient_inference(draws, coefficient)
    standard_error = np.std(draws, ddof=1)
    tails = 2 * min(
        (1 + np.count_nonzero(draws <= 0)) / (len(draws) + 1),
        (1 + np.count_nonzero(draws >= 0)) / (len(draws) + 1),
    )
    expected = [
        standard_error,
        coefficient / standard_error,
        min(1, tails),
        *np.percentile(draws, [2.5, 97.5]),
    ]
    np.testing.assert_allclose(
        [
            actual.standard_error,
            actual.t_statistic,
            actual.pvalue,
            actual.ci_lower,
            actual.ci_upper,
        ],
        np.asarray(expected, dtype=float),
        rtol=1e-12,
        atol=1e-13,
    )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    ("coefficient", "draws"),
    [
        (float("nan"), [0.0, 1.0]),
        (float("inf"), [0.0, 1.0]),
        (0.0, [0.0, float("nan")]),
        (0.0, [0.0, -float("inf")]),
    ],
)
def test_nonfinite_bootstrap_coefficient_pvalue_is_unavailable(
    coefficient: float, draws: list[float]
) -> None:
    """A non-finite coefficient or draw makes the p-value unavailable."""
    assert np.isnan(
        bootstrap_coefficient_inference(np.asarray(draws), coefficient).pvalue
    )
