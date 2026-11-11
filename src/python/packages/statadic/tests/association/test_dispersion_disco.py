"""Independent group-count and label-null witnesses for DISCO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.association.decomposition import group_design
from statadic.association.dispersion import disco_decomposition
from statadic.core.random import cell_key, permutation_batch
from statadic.inference.disco import disco_permutation_null

from .._budgets import examples
from .._strategies import distinct_points
from .._validation.numeric import Tolerance, assert_finite_close

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy

N_POINTS = 6
GROUP_SIZES = [0, 1, 1, 2, 2, 2]
HAND_POINTS = np.asarray([0.0, 1.0, 3.0, 5.0, 8.0, 9.0])
HAND_SCHEDULES = np.asarray(
    [[0, 1, 1, 2, 2, 2], [2, 0, 1, 2, 1, 2], [2, 2, 2, 1, 1, 0]], dtype=np.int32
)


def _group_reference(distances: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Count actual rows and ordered within-group pairs independently."""
    within = 0.0
    groups = np.unique(codes)
    for group in groups:
        indices = np.flatnonzero(codes == group)
        within += sum(distances[i, j] for i in indices for j in indices) / (
            2 * len(indices)
        )
    total = float(distances.sum() / (2 * len(codes)))
    between = total - within
    ratio = (between / (len(groups) - 1)) / (within / (len(codes) - len(groups)))
    return np.asarray([within, between, total, between / total, ratio])


def _relabelings() -> SearchStrategy[np.ndarray]:
    """Draw three label schedules that keep the unequal sizes and the singleton."""
    schedule = st.permutations(GROUP_SIZES).map(
        lambda codes: np.asarray(codes, dtype=np.int32)
    )
    return st.lists(schedule, min_size=3, max_size=3).map(np.stack)


@pytest.mark.unit
@pytest.mark.reference
@examples(40)
@given(points=distinct_points(N_POINTS), schedules=_relabelings())
@example(points=HAND_POINTS, schedules=HAND_SCHEDULES)
def test_explicit_group_relabelings_and_singletons_match_independent_sums(
    points: np.ndarray, schedules: np.ndarray
) -> None:
    """Unequal sizes and a singleton retain their V-statistic normalization.

    Args:
        points: Distinct real-line points; their separation keeps every group's
            within-group dispersion positive, so the F ratio is defined.
        schedules: Three relabelings of the fixed unequal group sizes.

    """
    distances = np.abs(points[:, None] - points[None, :])
    expected = np.stack([_group_reference(distances, codes) for codes in schedules])
    design = group_design(schedules[0], n_groups=3)

    values = jax.vmap(disco_decomposition, in_axes=(None, 0, None))(
        jnp.asarray(distances), jnp.asarray(schedules), design.n_groups
    )

    assert np.isfinite(values).all()
    assert_finite_close(
        np.stack(values, axis=1),
        expected,
        tolerance=_tolerance(),
        label="DISCO decomposition",
    )


@pytest.mark.component
@pytest.mark.reference
@examples(20)
@given(points=distinct_points(N_POINTS), seed=st.integers(0, 2**31 - 1))
@example(points=HAND_POINTS, seed=42)
def test_label_permutation_null_matches_independent_per_draw_decompositions(
    points: np.ndarray, seed: int
) -> None:
    """Fix the key schedule and calculate each statistic independently.

    Args:
        points: Distinct real-line points, so every group has positive dispersion.
        seed: Seed that fixes the permutation schedule.

    """
    distances = np.abs(points[:, None] - points[None, :])
    codes = jnp.asarray(GROUP_SIZES)
    key = cell_key(seed, "association", "disco_permutation_null")
    schedules = np.asarray(permutation_batch(key, codes, 11))
    expected = np.asarray([_group_reference(distances, row)[-1] for row in schedules])

    actual = disco_permutation_null(distances, codes, 3, 11, seed)

    assert np.isfinite(actual).all()
    assert_finite_close(
        actual, expected, tolerance=_tolerance(), label="permutation null"
    )


def _tolerance() -> Tolerance:
    """Budget small dyadic reductions at the actual process precision."""
    error = 3e-5 if not jax.config.x64_enabled else 1e-12
    return Tolerance(
        rtol=error,
        atol=error,
        rationale="Small dyadic distance reductions at the active precision.",
    )
