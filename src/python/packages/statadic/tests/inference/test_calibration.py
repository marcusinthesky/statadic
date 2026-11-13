"""Shared eager calibration rules."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.inference.calibration import add_one_p, add_one_p_kernel

from .._budgets import examples

# Integer-valued grid, so ties with the observed value are common.
_GRID = st.integers(-3, 3).map(float)
_NONFINITE = st.sampled_from([np.nan, np.inf, -np.inf])
_DRAWS = st.one_of(_GRID, _NONFINITE)


@pytest.mark.component
@pytest.mark.numerical
@examples(100)
@example(observed=1.5, null=np.asarray([0.0, 1.5, 2.0, np.nan, np.inf]))
@given(observed=_GRID, null=hnp.arrays(np.float64, 6, elements=_DRAWS))
def test_add_one_p_counts_ties_and_rejects_nonfinite_draws(
    observed: float, null: np.ndarray
) -> None:
    """The right-tail rule includes ties and requires every draw to be finite.

    Args:
        observed: Generated observed statistic, possibly tied with draws.
        null: Generated draws, some of them nonfinite.

    """
    if not np.all(np.isfinite(null)):
        assert np.isnan(add_one_p(observed, null))
    else:
        expected = (1.0 + np.count_nonzero(null >= observed)) / (1.0 + null.size)
        assert add_one_p(observed, null) == pytest.approx(expected)


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(observed=np.nan, null=np.asarray([0.0, 1.0]))
@example(observed=np.inf, null=np.asarray([0.0, 1.0]))
@example(observed=-np.inf, null=np.asarray([0.0, 1.0]))
@given(observed=_NONFINITE, null=hnp.arrays(np.float64, 4, elements=_GRID))
def test_add_one_p_rejects_nonfinite_observed_values(
    observed: float, null: np.ndarray
) -> None:
    """An undefined observed statistic has no calibrated p-value.

    Args:
        observed: A nonfinite observed statistic.
        null: Generated finite draws.

    """
    assert np.isnan(add_one_p(observed, null))


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(observed=1.0, null=np.asarray([np.nan, np.inf]))
@given(
    observed=_GRID,
    null=st.lists(_NONFINITE, min_size=1, max_size=6).map(np.asarray),
)
def test_add_one_p_requires_at_least_one_finite_null_draw(
    observed: float, null: np.ndarray
) -> None:
    """An empty effective reference distribution returns NaN.

    Args:
        observed: Generated observed statistic.
        null: Generated draws, none of them finite.

    """
    assert np.isnan(add_one_p(observed, null))


@pytest.mark.integration
@pytest.mark.contract
@examples(60)
@example(
    observed=np.array([2.0, 1.0, np.nan]),
    null=np.array(
        [
            [0.0, 2.0, 2.0, np.nan, np.inf],
            [np.nan, np.inf, -np.inf, np.nan, np.nan],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ]
    ),
)
@given(
    observed=hnp.arrays(np.float64, 3, elements=st.one_of(_GRID, st.just(np.nan))),
    null=hnp.arrays(np.float64, (3, 5), elements=_DRAWS),
)
def test_compiled_calibration_keeps_per_cell_validity(
    observed: np.ndarray, null: np.ndarray
) -> None:
    """Batching retains all-draw calibration and undefined-result cases.

    Args:
        observed: Generated per-cell statistics, possibly NaN.
        null: Generated per-cell draws, some nonfinite.

    """
    actual = np.asarray(
        jax.jit(jax.vmap(add_one_p_kernel))(jnp.asarray(observed), jnp.asarray(null))
    )

    for cell in range(3):
        valid = null[cell]
        if not np.all(np.isfinite(valid)) or not np.isfinite(observed[cell]):
            assert np.isnan(actual[cell])
        else:
            expected = (1.0 + np.count_nonzero(valid >= observed[cell])) / (
                1.0 + valid.size
            )
            assert actual[cell] == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("invalid", ["nan_stat", "inf_stat", "nan_draw", "inf_draw"])
def test_mixed_lanes_preserve_valid_calibration(invalid: str) -> None:
    """Eager and transformed lanes agree while invalid lanes return NaN."""
    stats = jnp.array([0.5, 0.5])
    draws = jnp.array([[0.0, 1.0, -1.0], [0.0, 1.0, -1.0]])
    if invalid.endswith("stat"):
        stats = stats.at[0].set(jnp.nan if invalid == "nan_stat" else jnp.inf)
    else:
        draws = draws.at[0, 0].set(jnp.nan if invalid == "nan_draw" else -jnp.inf)
    baseline = add_one_p_kernel(stats[1], draws[1])
    assert jnp.isnan(add_one_p_kernel(stats[0], draws[0]))
    assert jnp.isnan(jax.jit(add_one_p_kernel)(stats[0], draws[0]))
    for run in (jax.vmap(add_one_p_kernel), jax.jit(jax.vmap(add_one_p_kernel))):
        result = run(stats, draws)
        assert jnp.isnan(result[0])
        np.testing.assert_array_equal(result[1], baseline)
