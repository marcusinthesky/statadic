# standalone package-maintenance entry point
"""Independent fixed-draw witnesses for schedule, refit and calibration semantics."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from jax import random
from statadic.estimation.linear import (
    LeastSquares,
    LinearProblem,
)
from statadic.inference.resampling.block_length import (
    stationary_bootstrap_estimate,
)
from statadic.inference.resampling.bootstrap import bootstrap_estimate
from statadic.inference.resampling.schedule import (
    ReplicateStatus,
    ReplicateValue,
    evaluate_replicates,
    iid_row_schedule,
    linear_pairs_replicates,
    percentile_summary,
    refit_replicates,
    row_resample_schedule,
    stationary_row_schedule,
)

from .._budgets import examples
from .._strategies import float_arrays, xps

if TYPE_CHECKING:
    from jax import Array
    from statadic.estimation.linear import LinearFit
    from statadic.inference.resampling.schedule import (
        PercentileSummary,
        ReplicateResult,
    )


_ROWS = np.asarray(
    [[0, 1, 2, 3], [3, 2, 1, 0], [0, 0, 2, 2], [1, 2, 2, 3], [0, 1, 1, 3]],
    dtype=np.int32,
)
_DRAW_IDS = np.asarray([41, 9, 12, 83, 1], dtype=np.int32)
_SEEDS = st.integers(0, 2**31 - 1)
_ORIGINAL_DATA = jnp.asarray([1.0, 3.0, -2.0, 8.0])
_DATA4 = float_arrays(4)


def _valid(values: jax.Array) -> ReplicateValue[jax.Array]:
    """Declare a traceable numerical value available before finite checks."""
    return ReplicateValue(
        values,
        jnp.ones((), dtype=bool),
        jnp.asarray(ReplicateStatus.VALID, dtype=jnp.int32),
    )


@pytest.mark.component
@pytest.mark.reference
@examples(20)
@example(values=_ORIGINAL_DATA, chunk=1)
@example(values=_ORIGINAL_DATA, chunk=2)
@example(values=_ORIGINAL_DATA, chunk=4)
@example(values=_ORIGINAL_DATA, chunk=64)
@given(values=float_arrays(4, bound=3.0), chunk=st.sampled_from([1, 2, 4, 64]))
def test_paired_vector_draws_match_explicit_numpy_and_remainder(
    values: Array, chunk: int
) -> None:
    """The same IDs and row draws align all components, including a final slab.

    Args:
        values: Generated observations.
        chunk: Draw slab size, including a final remainder slab.

    """
    data = np.asarray(values)
    schedule = row_resample_schedule(_ROWS, n_observations=4, draw_ids=_DRAW_IDS)

    def statistic(indices: jax.Array) -> ReplicateValue[jax.Array]:
        sample = values[indices]
        return _valid(jnp.asarray([jnp.mean(sample), jnp.mean(sample**2)]))

    result = jax.jit(lambda: evaluate_replicates(schedule, statistic, chunk=chunk))()
    expected = np.asarray(
        [[np.mean(data[row]), np.mean(data[row] ** 2)] for row in _ROWS]
    )
    np.testing.assert_array_equal(result.draw_ids, _DRAW_IDS)
    np.testing.assert_allclose(result.values, expected, atol=1e-14)
    assert int(result.requested_count) == len(_ROWS)
    assert int(result.valid_count) == len(_ROWS)
    assert int(result.failed_count) == 0
    np.testing.assert_array_equal(result.status, 0)


@pytest.mark.component
@pytest.mark.numerical
@examples(20)
@example(data=jnp.asarray([1.0, 3.0, -2.0, 8.0], dtype=jnp.float32))
@given(
    data=xps.arrays(
        jnp.float32,
        4,
        elements={
            "min_value": -10.0,
            "max_value": 10.0,
            "allow_nan": False,
            "allow_infinity": False,
        },
    )
)
def test_nested_pytree_summary_uses_supplied_draws(data: Array) -> None:
    """Calibration is isolated from random generation and uses marginal axes.

    Args:
        data: Generated float32 observations.

    """
    schedule = row_resample_schedule(_ROWS, n_observations=4, draw_ids=_DRAW_IDS)

    def statistic(
        indices: jax.Array,
    ) -> ReplicateValue[dict[str, tuple[jax.Array, jax.Array]]]:
        sample = data[indices]
        return ReplicateValue(
            {
                "moments": (
                    jnp.mean(sample),
                    jnp.asarray([jnp.min(sample), jnp.max(sample)]),
                )
            },
            jnp.ones((), dtype=bool),
            jnp.asarray(0, dtype=jnp.int32),
        )

    result = evaluate_replicates(schedule, statistic, chunk=2)

    def evaluate_percentile_summary(
        draws: ReplicateResult[jax.Array],
    ) -> PercentileSummary[Array]:
        return percentile_summary(draws, confidence_level=0.8)

    summary = jax.jit(evaluate_percentile_summary)(result)
    assert bool(summary.valid)
    for draws, lower, upper, se in zip(
        jax.tree.leaves(result.values),
        jax.tree.leaves(summary.lower),
        jax.tree.leaves(summary.upper),
        jax.tree.leaves(summary.standard_error),
        strict=True,
    ):
        np.testing.assert_allclose(lower, np.percentile(draws, 10, axis=0), atol=2e-6)
        np.testing.assert_allclose(upper, np.percentile(draws, 90, axis=0), atol=2e-6)
        np.testing.assert_allclose(se, np.std(draws, axis=0, ddof=1), atol=2e-6)
        assert lower.dtype == jnp.float32


@pytest.mark.component
@pytest.mark.regression
@examples(20)
@example(seed=17, data=_ORIGINAL_DATA, typed_key=False)
@example(seed=17, data=_ORIGINAL_DATA, typed_key=True)
@given(seed=_SEEDS, data=_DATA4, typed_key=st.booleans())
def test_iid_schedule_preserves_historical_random_stream(
    seed: int, data: Array, *, typed_key: bool
) -> None:
    """Changing chunk sizes or key representation does not redraw a given stream.

    Args:
        seed: Stream seed.
        data: Generated observations resampled by the bootstrap.
        typed_key: Whether the key is a typed key or a legacy uint32 pair.

    """
    key = random.key(seed) if typed_key else random.PRNGKey(seed)

    def evaluate_choice(k: jax.Array) -> Array:
        return random.choice(k, 4, shape=(4,), replace=True)

    expected = jax.vmap(evaluate_choice)(random.split(key, 5))
    for chunk in (1, 2, 64):
        schedule = iid_row_schedule(key, 4, 5, chunk=chunk)
        np.testing.assert_array_equal(schedule.indices, expected)
    estimate = bootstrap_estimate(
        data, jnp.mean, key, num_bootstrap=5, confidence_level=0.8
    )
    draws = np.asarray(data)[np.asarray(expected)].mean(axis=1)
    np.testing.assert_allclose(estimate.ci, np.percentile(draws, [10, 90]), atol=1e-14)
    np.testing.assert_allclose(estimate.se, np.std(draws, ddof=1), atol=1e-14)


def _legacy_stationary_indices(
    key: jax.Array, n_rows: int, n_draws: int, length: float
) -> np.ndarray:
    """Independent host recurrence with the former split and transition order."""
    rows = []
    for draw_key in random.split(key, n_draws):
        key_initial, key_scan = random.split(draw_key)
        previous = int(random.randint(key_initial, (), 0, n_rows))
        indices = [previous]
        for scan_key in random.split(key_scan, n_rows - 1):
            key_jump, key_index = random.split(scan_key)
            jump = float(random.uniform(key_jump, ())) < 1.0 / max(length, 1.0)
            index = int(random.randint(key_index, (), 0, n_rows))
            previous = index if jump else (previous + 1) % n_rows
            indices.append(previous)
        rows.append(indices)
    return np.asarray(rows)


@pytest.mark.component
@pytest.mark.regression
@examples(20)
@example(length=0.5, seed=18, data=_ORIGINAL_DATA)
@example(length=1.0, seed=18, data=_ORIGINAL_DATA)
@example(length=3.5, seed=18, data=_ORIGINAL_DATA)
@given(length=st.floats(0.5, 10.0), seed=_SEEDS, data=_DATA4)
def test_stationary_schedule_and_scalar_adapter_preserve_stream(
    length: float, seed: int, data: Array
) -> None:
    """A host recurrence detects altered restarts, circular continuation and keys.

    Args:
        length: Mean block length.
        seed: Stream seed.
        data: Generated observations.

    """
    key = random.PRNGKey(seed)
    expected = _legacy_stationary_indices(key, 4, 5, length)
    for chunk in (1, 2, 64):
        schedule = stationary_row_schedule(key, 4, 5, length, chunk=chunk)
        np.testing.assert_array_equal(schedule.indices, expected)
    estimate = stationary_bootstrap_estimate(
        data, jnp.mean, key, length, num_bootstrap=5, confidence_level=0.8
    )
    draws = np.asarray(data)[expected].mean(axis=1)
    np.testing.assert_allclose(estimate.ci, np.percentile(draws, [10, 90]), atol=1e-14)
    np.testing.assert_allclose(estimate.se, np.std(draws, ddof=1), atol=1e-14)


@pytest.mark.component
@pytest.mark.contract
@examples(20)
@example(data=jnp.arange(12.0).reshape(3, 4), seed=19)
@given(data=float_arrays((3, 4)), seed=_SEEDS)
def test_outer_batch_and_compiled_summary_retain_counts(data: Array, seed: int) -> None:
    """Each batch cell has its own evidence with fixed schedule identity.

    Args:
        data: Generated batch of observations.
        seed: Seed for the per-cell keys.

    """

    def cell(
        data: jax.Array, key: jax.Array
    ) -> tuple[ReplicateResult[jax.Array], jax.Array]:
        schedule = iid_row_schedule(key, 4, 5, chunk=2)
        result = evaluate_replicates(
            schedule, lambda idx: _valid(jnp.mean(data[idx])), chunk=2
        )
        return result, percentile_summary(result).lower

    keys = random.split(random.key(seed), 3)
    transformed, lower = jax.jit(jax.vmap(cell))(data, keys)
    assert transformed.values.shape == (3, 5)
    assert transformed.draw_ids.shape == (3, 5)
    np.testing.assert_array_equal(transformed.valid_count, [5, 5, 5])
    assert lower.shape == (3,)


@pytest.mark.component
@pytest.mark.reference
def test_linear_pairs_refits_aligned_weighted_rows_against_numpy() -> None:
    """Weights, response and retained columns move together with one row index."""
    x = np.column_stack([np.ones(4), [-1.0, 0.0, 2.0, 3.0]])
    y = np.asarray([2.0, 4.0, 7.0, 8.0])
    weights = np.asarray([1.0, 2.0, 0.75, 3.0])
    fit = LeastSquares().estimate(
        LinearProblem(x, y, weights, parameter_names=("intercept", "slope"))
    )
    schedule = row_resample_schedule(_ROWS, n_observations=4, draw_ids=_DRAW_IDS)

    def evaluate_linear_pairs_replicates(state: LinearFit) -> ReplicateResult[Array]:
        return linear_pairs_replicates(state, schedule, chunk=2)

    result = jax.jit(evaluate_linear_pairs_replicates)(fit)
    expected = []
    for rows in _ROWS:
        root_weights = np.sqrt(weights[rows])
        expected.append(
            np.linalg.lstsq(
                x[rows] * root_weights[:, None], y[rows] * root_weights, rcond=None
            )[0]
        )
    assert bool(jnp.all(result.valid))
    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(result.draw_ids, _DRAW_IDS)


@pytest.mark.component
@pytest.mark.contract
def test_rank_failed_refit_is_retained_and_invalidates_summary() -> None:
    """A failed draw cannot disappear into a plausible conditional interval."""
    x = jnp.column_stack([jnp.ones(4), jnp.arange(4.0)])
    fit = LeastSquares().estimate(LinearProblem(x, jnp.asarray([2.0, 3.0, 4.0, 7.0])))
    schedule = row_resample_schedule(
        jnp.asarray([[0, 1, 2, 3], [1, 1, 1, 1]]),
        n_observations=4,
        draw_ids=jnp.asarray([7, 14]),
    )
    result = linear_pairs_replicates(fit, schedule, chunk=1)
    np.testing.assert_array_equal(result.valid, [True, False])
    assert int(result.status[1]) == ReplicateStatus.INVALID_REFIT
    assert int(result.requested_count) == len(schedule.draw_ids)
    assert int(result.valid_count) == 1
    assert int(result.failed_count) == 1
    assert bool(jnp.all(jnp.isnan(result.values[1])))
    summary = percentile_summary(result)
    assert not bool(summary.valid)
    assert int(summary.status) == ReplicateStatus.FAILED_DRAWS
    assert bool(jnp.all(jnp.isnan(summary.lower)))


@pytest.mark.component
@pytest.mark.contract
def test_generic_refit_preserves_solver_failure_reason() -> None:
    """Backend failure and a scientifically invalid successful solve differ."""
    fit = LeastSquares().estimate(LinearProblem(jnp.ones((4, 1)), jnp.arange(4.0)))
    failed = replace(
        fit,
        valid=jnp.zeros((), dtype=bool),
        solver=replace(fit.solver, converged=jnp.zeros((), dtype=bool)),
    )
    schedule = row_resample_schedule(_ROWS, n_observations=4)
    result = refit_replicates(schedule, lambda _: failed, chunk=2)
    np.testing.assert_array_equal(result.status, ReplicateStatus.SOLVER_FAILURE)
    assert int(result.failed_count) == len(_ROWS)


@pytest.mark.component
@pytest.mark.contract
@examples(20)
@example(chunk=1, finite=jnp.asarray([2.0, 11.0, 13.0]))
@example(chunk=3, finite=jnp.asarray([2.0, 11.0, 13.0]))
@example(chunk=64, finite=jnp.asarray([2.0, 11.0, 13.0]))
@given(chunk=st.sampled_from([1, 3, 64]), finite=float_arrays(3))
def test_mixed_failure_ledger_preserves_every_draw_and_invalidates_summary(
    chunk: int, finite: Array
) -> None:
    """Mixed callbacks cannot turn four attempted draws into one successful fit.

    Args:
        chunk: Draw slab size.
        finite: Generated values for the three draws whose value is finite.

    """
    indices = jnp.asarray([[0, 1, 2, 3], [1, 1, 2, 3], [2, 1, 2, 3], [3, 1, 2, 3]])
    identities = jnp.asarray([41, 9, 83, 12])
    schedule = row_resample_schedule(indices, n_observations=4, draw_ids=identities)
    values = jnp.concatenate([finite, jnp.asarray([jnp.inf])])
    statuses = jnp.asarray(
        [
            ReplicateStatus.VALID,
            ReplicateStatus.INVALID_REFIT,
            ReplicateStatus.SOLVER_FAILURE,
            ReplicateStatus.VALID,
        ],
        dtype=jnp.int32,
    )

    def statistic(rows: jax.Array) -> ReplicateValue[jax.Array]:
        draw = rows[0]
        return ReplicateValue(values[draw], statuses[draw] == 0, statuses[draw])

    result = jax.jit(lambda: evaluate_replicates(schedule, statistic, chunk=chunk))()
    summary = percentile_summary(result)

    np.testing.assert_array_equal(result.draw_ids, identities)
    np.testing.assert_array_equal(result.values, [*np.asarray(finite), np.inf])
    np.testing.assert_array_equal(result.valid, [True, False, False, False])
    np.testing.assert_array_equal(
        result.status,
        [
            ReplicateStatus.VALID,
            ReplicateStatus.INVALID_REFIT,
            ReplicateStatus.SOLVER_FAILURE,
            ReplicateStatus.NONFINITE_VALUE,
        ],
    )
    assert int(result.requested_count) == len(identities)
    assert int(result.valid_count) == 1
    assert int(result.failed_count) == len(identities) - 1
    assert not bool(summary.valid)
    assert int(summary.status) == ReplicateStatus.FAILED_DRAWS
    assert np.isnan([summary.lower, summary.upper, summary.standard_error]).all()
    np.testing.assert_array_equal(schedule.indices, indices)
    np.testing.assert_array_equal(schedule.draw_ids, identities)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "rows",
    [
        np.asarray([[0.0, 1, 2, 3]]),
        np.asarray([[0, 1, 2]]),
        np.empty((0, 4), dtype=np.int32),
    ],
)
def test_schedule_rejects_wrong_representation_or_alignment(rows: np.ndarray) -> None:
    """Float multiplicities, wrong row counts and empty requests are not row draws."""
    with pytest.raises(ValueError, match="indices must be"):
        row_resample_schedule(rows, n_observations=4)


@pytest.mark.component
@pytest.mark.contract
def test_schedule_dynamic_bounds_and_duplicate_identity_are_unavailable() -> None:
    """JAX gather clipping and duplicate IDs cannot make invalid draws available."""

    def run(rows: jax.Array, ids: jax.Array) -> ReplicateResult[jax.Array]:
        schedule = row_resample_schedule(rows, n_observations=4, draw_ids=ids)
        return evaluate_replicates(
            schedule, lambda idx: _valid(jnp.mean(idx.astype(float)))
        )

    rows = jnp.asarray([[0, 1, 2, 3], [0, 1, 2, 4]])
    bounds = jax.jit(run)(rows, jnp.asarray([2, 3]))
    np.testing.assert_array_equal(bounds.valid, [True, False])
    assert int(bounds.status[1]) == ReplicateStatus.INVALID_SCHEDULE
    duplicate = jax.jit(run)(rows, jnp.asarray([2, 2]))
    np.testing.assert_array_equal(duplicate.status, ReplicateStatus.INVALID_SCHEDULE)


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("large_field", ["indices", "draw_ids"])
def test_host_integer_narrowing_cannot_change_bounds_or_identity(
    large_field: str,
) -> None:
    """Default-width JAX must not wrap an invalid row or a valid large draw ID."""
    rows = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    ids = np.asarray([41], dtype=np.int64)
    if large_field == "indices":
        rows[0, -1] = 2**32 + 3
    else:
        ids[0] = 2**32 + 41
    with (
        jax.enable_x64(new_val=False),
        pytest.raises(ValueError, match="without narrowing"),
    ):
        row_resample_schedule(rows, n_observations=4, draw_ids=ids)
    # Caller-selected x64 retains identity and labels the original row bound.
    schedule = row_resample_schedule(rows, n_observations=4, draw_ids=ids)
    np.testing.assert_array_equal(schedule.draw_ids, ids)
    assert bool(schedule.valid[0]) == (large_field == "draw_ids")


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.subprocess
def test_callback_dtypes_are_checked_without_annotation_hook() -> None:
    """A fractional reason must not truncate to VALID during ordinary imports."""
    source = """
import jax  # noqa: PLC0415, TC002
import jax.numpy as jnp
from statadic.inference.resampling.schedule import (
    ReplicateValue, evaluate_replicates, row_resample_schedule,
)
schedule = row_resample_schedule(jnp.array([[0, 1], [1, 0]]), n_observations=2)
for validity, status in ((jnp.array(False), jnp.array(0.5)),
                         (jnp.array(1.0), jnp.array(0))):
    for transformed in (False, True):
        def evaluate():
            return evaluate_replicates(
                schedule,
                lambda _: ReplicateValue(jnp.array(1.0), validity, status),
            )
        try:
            (jax.jit(evaluate) if transformed else evaluate)()
        except ValueError as error:
            assert "boolean/signed-integer scalars" in str(error)
        else:
            raise AssertionError("Invalid callback dtype was accepted")
"""
    completed = subprocess.run(  # fixed interpreter and local source  # noqa: S603
        [sys.executable, "-I", "-c", source],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.component
@pytest.mark.contract
def test_node_axis_cannot_be_substituted_for_retained_row_axis() -> None:
    """An entity-sized schedule cannot silently gather a row-sized fitted sample."""
    fit = LeastSquares().estimate(LinearProblem(jnp.ones((4, 1)), jnp.arange(4.0)))
    schedule = row_resample_schedule(jnp.asarray([[0, 1, 2]]), n_observations=3)
    with pytest.raises(ValueError, match="retained observation axis"):
        linear_pairs_replicates(fit, schedule)


@pytest.mark.component
@pytest.mark.contract
def test_nonfinite_draw_and_nonfinite_block_length_have_explicit_reasons() -> None:
    """Numerical availability is checked independently of a user success flag."""
    schedule = row_resample_schedule(_ROWS, n_observations=4)
    result = evaluate_replicates(schedule, lambda _: _valid(jnp.asarray(jnp.inf)))
    np.testing.assert_array_equal(result.status, ReplicateStatus.NONFINITE_VALUE)
    stationary = stationary_row_schedule(random.key(20), 4, 5, jnp.inf)
    invalid = evaluate_replicates(stationary, lambda idx: _valid(idx.astype(float)))
    np.testing.assert_array_equal(invalid.status, ReplicateStatus.INVALID_SCHEDULE)


@pytest.mark.component
@pytest.mark.contract
def test_single_draw_strict_summary_and_legacy_scalar_policy_differ_explicitly() -> (
    None
):
    """New strict summaries need variance degrees of freedom; old scalar SE stays0."""
    schedule = iid_row_schedule(random.key(21), 4, 1)
    result = evaluate_replicates(
        schedule, lambda idx: _valid(jnp.mean(idx.astype(float)))
    )
    summary = percentile_summary(result)
    assert int(summary.status) == ReplicateStatus.INSUFFICIENT_DRAWS
    assert bool(jnp.isnan(summary.standard_error))
    legacy = bootstrap_estimate(
        jnp.arange(4.0), jnp.mean, random.key(21), num_bootstrap=1
    )
    np.testing.assert_array_equal(legacy.se, 0.0)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "value", [jnp.ones((2, 2)), [jnp.asarray(1.0)], {}, jnp.asarray(1 + 2j)]
)
def test_result_boundary_rejects_unsupported_leaf_or_container(value: object) -> None:
    """The promised result contract stays smaller than arbitrary JAX pytrees."""
    schedule = row_resample_schedule(_ROWS, n_observations=4)
    with pytest.raises(ValueError, match="Replicate values"):
        evaluate_replicates(
            schedule,
            lambda _: ReplicateValue(
                value, jnp.ones((), dtype=bool), jnp.asarray(0, dtype=jnp.int32)
            ),
        )


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(confidence=0.0)
@example(confidence=1.0)
@example(confidence=float("nan"))
@given(confidence=st.floats().filter(lambda c: not 0 < c < 1))
def test_percentile_configuration_is_explicitly_validated(confidence: float) -> None:
    """Invalid interval levels do not enter percentile interpolation.

    Args:
        confidence: Any float outside the open unit interval, including NaN.

    """
    schedule = row_resample_schedule(_ROWS, n_observations=4)
    result = evaluate_replicates(
        schedule, lambda idx: _valid(jnp.mean(idx.astype(float)))
    )
    with pytest.raises(ValueError, match="confidence_level"):
        percentile_summary(result, confidence_level=confidence)
