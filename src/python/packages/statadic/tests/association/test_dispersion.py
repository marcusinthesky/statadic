# standalone package-maintenance entry point
"""DISCO — the distance-components decomposition, against hand-computed values.

The definition moved out of ``pipeline/stages/papers/paper1`` in t46.6, and a
decomposition is exactly the kind of code a refactor can leave subtly wrong: the
V-statistic convention counts the zero diagonal, so an implementation that
switched to the U-statistic would still return plausible, positive, finite
numbers. ``_hand_checked`` fixes that by supplying four points on a line whose
``S_W``, ``S_B``, ``T``, ``R2_E`` and ``F`` are all closed-form.
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
from statadic.association.decomposition import disco_components, group_design
from statadic.association.dispersion import disco_decomposition, dispersion_f_ratio
from statadic.core.matrices import unsafe_assume_dmat
from statadic.inference.calibration import add_one_p_kernel
from statadic.inference.disco import disco_permutation_null
from scipy.stats import f_oneway

from .._budgets import examples
from .._clouds import clustered

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

_TWO_GROUPS = 2
MIN_GAP = 0.1
_HAND_R2 = 9.0 / 11.0
_HAND_F = 9.0


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
def test_supplied_permdisp_label_draws_match_independent_radius_anova() -> None:
    """Paired signs make each scheduled group's median exactly zero."""
    coordinates = np.asarray([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
    schedules = np.asarray(
        [[1, 1, 0, 0, 0, 0, 1, 1], [1, 0, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1, 0]],
        dtype=np.int32,
    )
    expected = np.asarray(
        [
            f_oneway(
                np.abs(coordinates[codes == 0]), np.abs(coordinates[codes == 1])
            ).statistic
            for codes in schedules
        ]
    )

    actual = jax.jit(
        jax.vmap(dispersion_f_ratio, in_axes=(None, 0, None)), static_argnums=2
    )(jnp.asarray(coordinates[:, None]), jnp.asarray(schedules), 2)

    np.testing.assert_allclose(expected, [24.0, 1.5, 0.0], rtol=1e-13, atol=1e-13)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scikit-bio")
def test_median_permdisp_matches_skbio_and_independent_radius_anova(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Symmetric groups give exact medians and a nondegenerate ANOVA on radii."""
    peer = optional_reference("skbio")
    shape = np.asarray([[-1.0, 0.0], [1.0, 0.0], [0.0, -2.0], [0.0, 2.0]])
    clouds = [shape, 2 * shape + [10.0, 0.0], 3 * shape + [0.0, 10.0]]
    points = np.concatenate(clouds)
    codes = np.repeat(np.arange(3), len(shape))
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    radii = np.asarray([1.0, 1.0, 2.0, 2.0])
    expected = f_oneway(radii, 2 * radii, 3 * radii).statistic
    reference = peer.stats.distance.permdisp(
        peer.DistanceMatrix(distances),
        codes,
        test="median",
        permutations=0,
        method="eigh",
    )

    actual = dispersion_f_ratio(jnp.asarray(points), jnp.asarray(codes), 3)

    assert np.isfinite(actual)
    assert float(actual) == pytest.approx(float(expected), rel=1e-10, abs=1e-10)
    assert float(actual) == pytest.approx(
        float(reference["test statistic"]), rel=1e-10, abs=1e-10
    )


def _hand_checked() -> tuple[np.ndarray, np.ndarray]:
    """Build two hand-checkable groups ``A = {0, 1}``, ``B = {5, 6}`` on a line.

    Pairwise distances give ``abar_AA = abar_BB = 1/2`` (V-statistic: the zero
    diagonal counts), so ``S_W = 1``, ``T = 11/2``, ``S_B = 9/2``,
    ``R2_E = 9/11`` and ``F = 9``.

    Returns:
        ``(distances, codes)`` for the four points.

    """
    dist = np.array(
        [
            [0.0, 1.0, 5.0, 6.0],
            [1.0, 0.0, 4.0, 5.0],
            [5.0, 4.0, 0.0, 1.0],
            [6.0, 5.0, 1.0, 0.0],
        ]
    )
    return dist, np.array([0, 0, 1, 1], dtype=np.int32)


@pytest.mark.unit
@pytest.mark.numerical
def test_disco_decomposition_matches_hand_computed_v_statistic() -> None:
    """Reproduce the hand-checked DISCO quantities after the move into statadic."""
    dist, codes = _hand_checked()
    s_w, s_b, t_total, r2_e, f_ratio = disco_decomposition(
        jnp.asarray(dist), jnp.asarray(codes), _TWO_GROUPS
    )
    assert float(s_w) == pytest.approx(1.0, rel=1e-5)
    assert float(t_total) == pytest.approx(11.0 / 2.0, rel=1e-5)
    assert float(s_b) == pytest.approx(9.0 / 2.0, rel=1e-5)
    assert float(r2_e) == pytest.approx(_HAND_R2, rel=1e-5)
    assert float(f_ratio) == pytest.approx(_HAND_F, rel=1e-5)
    assert float(t_total) == pytest.approx(float(s_w + s_b), abs=1e-12)


def _well_separated(values: np.ndarray) -> bool:
    """Require gaps wide enough that every group's dispersion is positive."""
    return bool(np.diff(np.sort(values)).min() > MIN_GAP)


@pytest.mark.unit
@pytest.mark.property
@examples(40)
@given(
    points=hnp.arrays(
        np.float64, 4, elements=st.floats(-10.0, 10.0, width=64), unique=True
    ).filter(_well_separated),
    codes=st.permutations([0, 0, 1, 1]).map(
        lambda labels: np.asarray(labels, dtype=np.int32)
    ),
)
@example(points=np.array([0.0, 1.0, 5.0, 6.0]), codes=np.array([0, 0, 1, 1]))
def test_disco_is_invariant_to_group_name_permutation(
    points: np.ndarray, codes: np.ndarray
) -> None:
    """Group identifiers are labels, not an ordered numerical covariate.

    Args:
        points: Distinct real-line points, so both groups have positive dispersion.
        codes: Two equal-sized groups, assigned to the points in any order.

    """
    dist = np.abs(points[:, None] - points[None, :])
    original = disco_decomposition(jnp.asarray(dist), jnp.asarray(codes), 2)
    renamed = disco_decomposition(jnp.asarray(dist), jnp.asarray(1 - codes), 2)

    np.testing.assert_allclose(np.asarray(original), np.asarray(renamed))


@pytest.mark.unit
@pytest.mark.numerical
def test_alpha_two_squared_euclidean_disco_matches_classical_anova() -> None:
    """At alpha=2 DISCO is the classical mean-only ANOVA decomposition."""
    values = np.array([0.0, 1.0, 3.0, 5.0, 8.0, 9.0], dtype=np.float64)
    codes = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    distances = (values[:, None] - values[None, :]) ** 2

    within, between, total, _index, f_ratio = disco_decomposition(
        jnp.asarray(distances), jnp.asarray(codes), 2
    )
    grand_mean = values.mean()
    group_means = np.array([values[codes == group].mean() for group in range(2)])
    classical_within = sum(
        np.sum((values[codes == group] - group_means[group]) ** 2) for group in range(2)
    )
    classical_between = sum(
        np.count_nonzero(codes == group) * (group_means[group] - grand_mean) ** 2
        for group in range(2)
    )
    classical_total = np.sum((values - grand_mean) ** 2)
    classical_f = (classical_between / 1) / (classical_within / 4)

    assert float(within) == pytest.approx(classical_within)
    assert float(between) == pytest.approx(classical_between)
    assert float(total) == pytest.approx(classical_total)
    assert float(f_ratio) == pytest.approx(classical_f)


@pytest.mark.component
@pytest.mark.property
def test_typed_disco_test_is_deterministic_and_reports_the_null() -> None:
    """The only composite path preserves typed decomposition and calibration."""
    dist, codes = clustered()
    matrix = unsafe_assume_dmat(jnp.asarray(dist))
    design = group_design(codes, n_groups=3)
    observed = disco_components(matrix, design).f_ratio.value
    first = disco_permutation_null(matrix.values, design.codes, 3, 200, 11)
    second = disco_permutation_null(matrix.values, design.codes, 3, 200, 11)
    np.testing.assert_array_equal(first, second)
    first_p = float(add_one_p_kernel(observed, first))
    second_p = float(add_one_p_kernel(observed, second))
    assert first_p == pytest.approx(second_p)
    assert 0.0 < first_p <= 1.0
    assert first_p == pytest.approx(1.0 / 201.0)
    assert np.isfinite(np.asarray(first).std(ddof=1))
