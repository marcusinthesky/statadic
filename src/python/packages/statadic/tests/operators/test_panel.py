"""Panel absorption against explicit weighted dummy projections."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.operators.panel import (
    absorb_panel_effects,
    prepare_panel_index,
    prepare_panel_sample,
)

from .._budgets import examples

if TYPE_CHECKING:
    from statadic.operators.panel import (
        PanelProjection,
    )


_ENTITY = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3])
_TIME = np.array([0, 1, 2, 0, 2, 0, 1, 2, 1, 2])
_N_OBS = len(_ENTITY)
_OUTCOMES = hnp.arrays(
    np.float64,
    (_N_OBS, 3),
    elements=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False, width=64),
)
# Positive weights within the original 0.5-2 range keep the solver well conditioned.
_WEIGHTS = hnp.arrays(
    np.float64,
    (_N_OBS,),
    elements=st.floats(0.5, 2.0, allow_nan=False, allow_infinity=False),
)
_RNG = np.random.default_rng(126)
_ORIGINAL_Y = _RNG.normal(size=(_N_OBS, 3))
_ORIGINAL_WEIGHTS = _RNG.uniform(0.5, 2.0, size=_N_OBS)


@pytest.mark.unit
@pytest.mark.reference
@examples(20)
@example(y=_ORIGINAL_Y, weights=_ORIGINAL_WEIGHTS)
@given(y=_OUTCOMES, weights=_WEIGHTS)
def test_weighted_unbalanced_two_way_absorption_matches_dummy_projection(
    y: np.ndarray, weights: np.ndarray
) -> None:
    """Alternating reductions reproduce an independently solved full projection."""
    entity, time = _ENTITY, _TIME
    index = prepare_panel_index(entity, time)
    dummy = np.column_stack([np.eye(4)[entity], np.eye(3)[time]])
    coef = np.linalg.lstsq(
        dummy * np.sqrt(weights[:, None]), y * np.sqrt(weights[:, None]), rcond=None
    )[0]
    expected = y - dummy @ coef

    def evaluate_absorb_panel_effects(v: jax.Array) -> PanelProjection:
        return absorb_panel_effects(
            v, index, weights=jnp.asarray(weights), time_effects=True, tolerance=1e-11
        )

    actual = jax.jit(evaluate_absorb_panel_effects)(jnp.asarray(y))
    assert actual.converged
    np.testing.assert_allclose(actual.values, expected, atol=1e-10)

    def evaluate_unweighted_absorption(v: jax.Array) -> PanelProjection:
        return absorb_panel_effects(v, index, time_effects=True)

    batch = jax.vmap(evaluate_unweighted_absorption)(
        jnp.stack([jnp.asarray(y), jnp.asarray(y) * 2])
    )
    assert np.all(batch.converged)


@pytest.mark.unit
@pytest.mark.contract
def test_duplicate_panel_keys_are_rejected() -> None:
    """Preparation never silently aggregates repeated observations."""
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_panel_index(np.array([0, 0]), np.array([1, 1]))


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("keys", [[True, False], [1.0, 2.0]])
def test_panel_keys_require_integer_dtype(keys: list[bool] | list[float]) -> None:
    """Numerically integral floats and booleans are not panel keys."""
    with pytest.raises(ValueError, match="as integers"):
        prepare_panel_index(np.array(keys), np.array([0, 1]))


@pytest.mark.unit
@pytest.mark.contract
def test_panel_mixed_signed_unsigned_keys_do_not_round_for_duplicates() -> None:
    """Duplicate detection compares encoded keys rather than a promoted matrix."""
    with jax.enable_x64(False):  # noqa: FBT003
        index = prepare_panel_index(
            np.array([0, 0], dtype=np.int32),
            np.array([16777216, 16777217], dtype=np.uint32),
        )
    assert index.n_periods == 2  # noqa: PLR2004
    np.testing.assert_array_equal(index.difference_rows, [1])


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("x64", [False, True])
def test_panel_keys_never_narrow_before_grouping_or_pruning(*, x64: bool) -> None:
    """Preserve representable keys and reject lossy identity conversions."""
    entities = np.array([7, 4294967303], dtype=np.int64)
    times = np.array([0, 1], dtype=np.int64)
    with jax.enable_x64(x64):
        if x64:
            index = prepare_panel_index(entities, times)
            assert index.n_entities == len(entities)
            assert index.difference_rows.size == 0
            sample = prepare_panel_sample(
                np.ones((3, 1)),
                np.ones(3),
                np.array([7, 7, 4294967303], dtype=np.int64),
                np.array([0, 1, 1]),
                singletons="drop",
            )
            np.testing.assert_array_equal(sample.retained_rows, [0, 1])
        else:
            with pytest.raises(ValueError, match="active JAX integer dtype"):
                prepare_panel_index(entities, times)
            with pytest.raises(ValueError, match="active JAX integer dtype"):
                prepare_panel_sample(
                    np.ones((2, 1)), np.ones(2), entities, times, singletons="drop"
                )
            with pytest.raises(ValueError, match="active JAX integer dtype"):
                prepare_panel_index(times, entities)


@pytest.mark.component
@pytest.mark.contract
def test_absorbed_column_removal_retains_original_schema() -> None:
    """Numerical projection noise cannot retain a fully absorbed regressor."""
    entity = np.repeat(np.arange(4), 3)
    time = np.tile(np.arange(3), 4)
    x = np.column_stack([entity, np.sin(np.arange(len(entity)))])
    sample = prepare_panel_sample(
        x, np.cos(np.arange(len(entity))), entity, time, drop_absorbed=True
    )
    np.testing.assert_array_equal(sample.retained_columns, [1])
    np.testing.assert_array_equal(sample.retained_rows, np.arange(len(entity)))


@pytest.mark.unit
@pytest.mark.numerical
def test_converged_absorption_has_the_projected_gradient() -> None:
    """The idle loop tail keeps reverse AD and its first convergence evidence."""
    values = jnp.array([1.0, 3.0, 2.0, 8.0])
    index = prepare_panel_index(jnp.array([0, 0, 1, 1]), jnp.array([0, 1, 0, 1]))

    def objective(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        result = absorb_panel_effects(x, index, max_steps=20)
        return jnp.sum(result.values**2), result.iterations

    gradient, iterations = jax.jit(jax.grad(objective, has_aux=True))(values)
    np.testing.assert_array_equal(gradient, [-2.0, 2.0, -6.0, 6.0])
    assert int(iterations) == 2  # noqa: PLR2004 -- second sweep witnesses convergence
