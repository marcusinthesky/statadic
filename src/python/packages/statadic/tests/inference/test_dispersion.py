# standalone package-maintenance entry point
"""PERMDISP statistic and its residual-permutation null keep scientific contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.association.dispersion import (
    dispersion_f_ratio,
    dispersion_radii,
    radius_anova_f,
)
from statadic.core.random import cell_key, permutation_batch
from statadic.geometry.median import spatial_medians, weiszfeld_medians
from statadic.inference.dispersion import (
    dispersion_permutation_null,
    dispersion_residual_null,
)

from .._budgets import examples
from .._strategies import float_arrays


@pytest.mark.integration
@pytest.mark.numerical
def test_dispersion_ratio_matches_symmetric_cloud_anova() -> None:
    """Known group medians give an independent one-way ANOVA on radii."""
    coords = jnp.asarray([[-1.0], [1.0], [-2.0], [2.0], [8.0], [12.0], [6.0], [14.0]])
    codes = jnp.repeat(jnp.arange(2), 4)
    radii = np.asarray([1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0])
    means = radii.reshape(2, 4).mean(axis=1)
    between = 4.0 * np.sum((means - radii.mean()) ** 2)
    within = np.sum((radii - np.repeat(means, 4)) ** 2)
    expected = between / (within / 6.0)
    np.testing.assert_allclose(
        dispersion_f_ratio(coords, codes, 2), expected, rtol=1e-10
    )


def _hand_anova(values: np.ndarray, codes: np.ndarray) -> float:
    groups = [values[codes == group] for group in np.unique(codes)]
    grand = values.mean()
    between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    return (between / (len(groups) - 1)) / (within / (len(values) - len(groups)))


_PERMUTATIONS = np.asarray(
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [8, 7, 6, 5, 4, 3, 2, 1, 0],
        [5, 6, 7, 8, 0, 1, 2, 3, 4],
    ]
)


@pytest.mark.integration
@pytest.mark.numerical
@examples(40)
@given(
    # A within-group ramp keeps every group non-constant, by construction: a
    # constant group has no residual variation and the null is unavailable.
    radii=float_arrays(9, bound=3.0).map(
        lambda noise: np.asarray(noise) + np.asarray([0, 2, 4, 6, 8, 0, 2, 4, 6.0])
    )
)
def test_residual_null_matches_the_hand_anova_of_permuted_residuals(
    radii: np.ndarray,
) -> None:
    """Each draw is the one-way ANOVA of permuted within-group residuals.

    Args:
        radii: Generated radii with within-group variation.

    """
    codes = np.repeat([0, 1], [5, 4])
    residuals = radii - np.asarray([radii[codes == g].mean() for g in codes])
    expected = [_hand_anova(residuals[row], codes) for row in _PERMUTATIONS]
    actual = dispersion_residual_null(
        jnp.asarray(radii), jnp.asarray(codes), 2, jnp.asarray(_PERMUTATIONS)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.integration
@pytest.mark.reference
def test_residual_null_refits_permuted_full_model_residuals() -> None:
    """Each draw is the one-way ANOVA of permuted within-group residuals."""
    radii = np.asarray([4.0, 1.0, 0.0, 4.0, 5.0, 3.0, 2.0, 0.0, 7.0])
    codes = np.repeat([0, 1], [5, 4])
    residuals = radii - np.asarray([radii[codes == g].mean() for g in codes])
    schedules = np.asarray(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1, 0],
            [5, 6, 7, 8, 0, 1, 2, 3, 4],
        ]
    )
    expected = [_hand_anova(residuals[row], codes) for row in schedules]

    actual = dispersion_residual_null(
        jnp.asarray(radii), jnp.asarray(codes), 2, jnp.asarray(schedules)
    )

    assert expected[0] == pytest.approx(0.0, abs=1e-12)
    assert expected[1] > 0
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.unit
@pytest.mark.contract
@examples(40)
@example(low=1.0, high=3.0)
@given(low=st.floats(-10, 10), high=st.floats(-10, 10))
def test_residual_null_is_unavailable_without_within_group_variation(
    low: float, high: float
) -> None:
    """Constant within-group radii carry no residual reference distribution.

    Args:
        low: Constant radius of the first group.
        high: Constant radius of the second group.

    """
    radii = jnp.asarray([low, low, low, high, high, high])
    codes = jnp.repeat(jnp.arange(2), 3)
    schedules = permutation_batch(jax.random.PRNGKey(0), jnp.arange(6), 4)

    assert bool(
        jnp.all(jnp.isnan(dispersion_residual_null(radii, codes, 2, schedules)))
    )


@pytest.mark.integration
@pytest.mark.regression
def test_dispersion_null_retains_key_and_permutes_residuals_of_its_statistic() -> None:
    """The door keeps its RNG stream and permutes residuals of its own radii."""
    coords = jnp.asarray([[-1.0], [1.0], [-2.0], [2.0], [8.0], [12.0], [6.0], [14.0]])
    codes = jnp.repeat(jnp.arange(2), 4)
    seed, draws = 19, 13
    schedules = permutation_batch(
        cell_key(seed, "geometry", "dispersion_permutation_null"),
        jnp.arange(8, dtype=jnp.int32),
        draws,
    )
    medians = weiszfeld_medians(coords, codes, 2)
    radii = jnp.sqrt(jnp.sum((coords - medians[codes]) ** 2, axis=1) + 1e-12)
    expected = dispersion_residual_null(radii, codes, 2, schedules)

    actual = dispersion_permutation_null(coords, codes, 2, draws, seed)

    np.testing.assert_array_equal(actual, expected)
    assert float(radius_anova_f(radii, codes, 2)) == pytest.approx(
        float(dispersion_f_ratio(coords, codes, 2))
    )


_SIZE_REPLICATIONS = 1_000
_SIZE_PERMUTATIONS = 99


@jax.jit
def _rejection_rate(shift: jax.Array, scale: jax.Array) -> jax.Array:
    """Monte Carlo 5% rejection rate for 15 + 15 bivariate normal points."""
    per_group = 15
    codes = jnp.repeat(jnp.arange(2), per_group)

    def replicate(key: jax.Array) -> jax.Array:
        data_key, permutation_key = jax.random.split(key)
        points = jax.random.normal(data_key, (2 * per_group, 2))
        points = points.at[per_group:].multiply(scale).at[per_group:, 0].add(shift)
        medians, _ = spatial_medians(points, codes, 2)
        radii = dispersion_radii(points, codes, medians, bias_adjust=True)
        observed = radius_anova_f(radii, codes, 2)
        schedules = permutation_batch(
            permutation_key, jnp.arange(2 * per_group), _SIZE_PERMUTATIONS
        )
        null = dispersion_residual_null(radii, codes, 2, schedules)
        pvalue = (jnp.sum(null >= observed) + 1) / (_SIZE_PERMUTATIONS + 1)
        return pvalue <= 0.05  # noqa: PLR2004

    keys = jax.random.split(jax.random.PRNGKey(11), _SIZE_REPLICATIONS)
    return jnp.mean(jax.vmap(replicate)(keys))


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.statistical
def test_residual_permutation_holds_size_under_pure_location_shift() -> None:
    """Equal dispersion with shifted locations is PERMDISP's null hypothesis.

    Relabeling with median refits rejected 24.5% at a 6 SD shift. The band is
    about 3.6 Monte Carlo standard errors (0.7 points each) around 5%.
    """
    rates = {
        shift: float(_rejection_rate(jnp.asarray(shift), jnp.asarray(1.0)))
        for shift in (0.0, 6.0)
    }

    for rate in rates.values():
        assert 0.025 <= rate <= 0.075  # noqa: PLR2004
    assert rates[0.0] == rates[6.0]
    power = float(_rejection_rate(jnp.asarray(0.0), jnp.asarray(1.5)))
    assert power > 0.2  # noqa: PLR2004
