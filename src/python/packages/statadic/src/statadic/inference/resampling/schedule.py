# ruff: noqa: F722  # jaxtyping dimensions are runtime contracts.
"""Preview fixed-schedule row resampling and explicit replicate validity.

Schedules contain row indices, never node multiplicities, sampled cluster IDs or
wild multipliers. Preprocessing, model columns and tuning are frozen by the
caller's evaluation/refit closure. The executor does not infer a resampling law
or establish bootstrap validity. Percentile intervals are marginal and require
every requested draw to succeed; no conditional-on-success summary is provided.
"""

# jaxtyping dimensions are runtime contracts.

from __future__ import annotations

from collections.abc import (
    Callable,  # runtime callable annotations  # noqa: TC003
)
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal, Protocol

import jax
import jax.numpy as jnp
from jax import lax, random
from jaxtyping import Bool, Int  # noqa: TC002

from statadic.core.results import (
    SolverResult,  # runtime protocol  # noqa: TC001
)
from statadic.core.typing import Array, ArrayLike, PRNGKey  # noqa: TC001
from statadic.estimation.linear import (  # runtime carrier  # noqa: TC001
    LeastSquares,
    LinearFit,
    LinearProblem,
)
from statadic.inference.resampling.blocks import DEFAULT_REPLICATE_CHUNK, map_replicates

__all__ = [
    "PercentileSummary",
    "RefitResult",
    "ReplicateResult",
    "ReplicateStatus",
    "ReplicateValue",
    "RowResampleSchedule",
    "evaluate_replicates",
    "iid_row_schedule",
    "linear_pairs_replicates",
    "percentile_summary",
    "refit_replicates",
    "row_resample_schedule",
    "stationary_row_schedule",
]

_MIN_SUMMARY_DRAWS = 2
_MATRIX_RANK = 2


class ReplicateStatus(IntEnum):
    """Availability of one draw or a strict all-draw percentile summary."""

    VALID = 0
    INVALID_SCHEDULE = 1
    NONFINITE_VALUE = 2
    INVALID_REFIT = 3
    SOLVER_FAILURE = 4
    INSUFFICIENT_DRAWS = 5
    FAILED_DRAWS = 6


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RowResampleSchedule:
    """Fixed-shape paired row-index draws with retained identity.

    Shape/schema errors raise at construction through the named factories.
    Dynamic bounds, duplicate IDs and invalid block lengths produce unavailable
    draws, including under JIT. Every replicate retains exactly the original
    number of rows, in its scheduled order. This carrier is a row design only.
    """

    indices: Int[Array, "*batch draws observations"]
    draw_ids: Int[Array, "*batch draws"]
    valid: Bool[Array, "*batch draws"]
    n_observations: int = field(metadata={"static": True})
    scheme: Literal["supplied", "iid", "stationary"] = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ReplicateValue[T]:
    """One scalar/vector or tuple/dict pytree and its explicit validity.

    Leaves must be real floating JAX arrays with at most one axis before the
    replicate/batch transformations. Tuples and string-key dictionaries may
    nest; lists, custom objects, empty trees and complex leaves are unsupported.
    ``status`` is a scalar integer reason, with zero reserved for valid draws.
    Nonfinite leaves are rejected independently of the supplied validity flag.
    """

    values: T
    valid: Bool[Array, "*batch"]
    status: Int[Array, "*batch"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ReplicateResult[T]:
    """Every requested draw, in schedule order, including failures.

    ``values`` retain raw evaluated leaves, even for failed draws; consumers
    must inspect ``valid``. Status codes and draw IDs remain aligned. Counts
    are numerical leaves so outer JIT/vmap operations retain their evidence.
    """

    values: T
    draw_ids: Int[Array, "*batch draws"]
    valid: Bool[Array, "*batch draws"]
    status: Int[Array, "*batch draws"]
    requested_count: Int[Array, "*batch"]
    valid_count: Int[Array, "*batch"]
    failed_count: Int[Array, "*batch"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PercentileSummary[T]:
    """Componentwise marginal bootstrap SE and percentile bounds.

    All leaves are NaN when any draw failed or fewer than two draws exist.
    These are neither simultaneous intervals nor studentized/BCa intervals.
    ``valid`` only certifies numerical availability, not statistical coverage.
    """

    standard_error: T
    lower: T
    upper: T
    valid: Bool[Array, "*batch"]
    status: Int[Array, "*batch"]
    confidence_level: float = field(metadata={"static": True})


class RefitResult(Protocol):
    """Minimal actual-fit state consumed by a scheduled refit adapter."""

    @property
    def parameters(self) -> Array:
        """Return parameters in a frozen, caller-owned order."""
        ...

    @property
    def valid(self) -> Array:
        """Return the fit's scientific availability flag."""
        ...

    @property
    def solver(self) -> SolverResult[Array]:
        """Return numerical diagnostics for this actual refit."""
        ...


def row_resample_schedule(
    indices: ArrayLike,
    *,
    n_observations: int,
    draw_ids: ArrayLike | None = None,
) -> RowResampleSchedule:
    """Own supplied integer row draws and label invalid dynamic values.

    IDs default to zero-based draw order. Supplied IDs must be unique
    nonnegative integers; duplicates invalidate the complete paired schedule.
    Out-of-range rows invalidate their draw rather than relying on JAX gather
    clipping. An explicit observation count prevents entity/row-axis confusion.
    """
    rows = _schedule_integer_array(indices)
    matrix_rank = 2
    if (
        n_observations < 1
        or rows.ndim != matrix_rank
        or rows.shape[0] < 1
        or rows.shape[1] != n_observations
        or not jnp.issubdtype(rows.dtype, jnp.signedinteger)
    ):
        message = "indices must be a nonempty integer (draws, n_observations) array."
        raise ValueError(message)
    ids = (
        jnp.arange(rows.shape[0], dtype=jnp.int32)
        if draw_ids is None
        else _schedule_integer_array(draw_ids)
    )
    if (
        ids.ndim != 1
        or ids.shape[0] != rows.shape[0]
        or not jnp.issubdtype(ids.dtype, jnp.signedinteger)
    ):
        message = "draw_ids must be an integer vector matching the draws."
        raise ValueError(message)
    sorted_ids = jnp.sort(ids)
    id_valid = jnp.all(ids >= 0) & jnp.all(sorted_ids[1:] != sorted_ids[:-1])
    valid = jnp.all((rows >= 0) & (rows < n_observations), axis=1) & id_valid
    return RowResampleSchedule(
        indices=jnp.array(rows, copy=True),
        draw_ids=jnp.array(ids, copy=True),
        valid=valid,
        n_observations=n_observations,
        scheme="supplied",
    )


def _schedule_integer_array(value: ArrayLike) -> Array:
    """Reject host integer narrowing before JAX could change bounds or draw IDs."""
    if not isinstance(value, (jax.Array, jax.core.Tracer)):
        host = value.tolist() if hasattr(value, "tolist") else value

        def integers(item: object) -> list[int]:
            if isinstance(item, (list, tuple)):
                return [i for child in item for i in integers(child)]
            return [item] if type(item) is int else []

        exact = integers(host)
        if exact:
            dtype = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
            bounds = jnp.iinfo(dtype)
            if min(exact) < bounds.min or max(exact) > bounds.max:
                message = (
                    "Host schedule integers must fit the caller's active JAX "
                    "signed integer precision without narrowing."
                )
                raise ValueError(message)
            return jnp.asarray(host)
    return jnp.asarray(value)


def _validate_counts(n_observations: int, num_replicates: int, chunk: int) -> None:
    """Validate static shape and executor configuration."""
    if n_observations < 1 or num_replicates < 1 or chunk < 1:
        message = "n_observations, num_replicates and chunk must be positive."
        raise ValueError(message)


def iid_row_schedule(
    key: PRNGKey,
    n_observations: int,
    num_replicates: int,
    *,
    chunk: int = DEFAULT_REPLICATE_CHUNK,
) -> RowResampleSchedule:
    """Generate paired iid row draws preserving the legacy split/choice stream."""
    _validate_counts(n_observations, num_replicates, chunk)
    keys = random.split(key, num_replicates)

    def draw(draw_key: PRNGKey) -> Array:
        """Draw the historical with-replacement row schedule."""
        return random.choice(
            draw_key, n_observations, shape=(n_observations,), replace=True
        )

    rows = map_replicates(draw, keys, chunk=chunk)
    return RowResampleSchedule(
        indices=rows,
        draw_ids=jnp.arange(num_replicates, dtype=jnp.int32),
        valid=jnp.ones(num_replicates, dtype=bool),
        n_observations=n_observations,
        scheme="iid",
    )


def stationary_row_schedule(
    key: PRNGKey,
    n_observations: int,
    num_replicates: int,
    block_length: Array | float,
    *,
    chunk: int = DEFAULT_REPLICATE_CHUNK,
) -> RowResampleSchedule:
    """Generate legacy stationary draws with circular continuation.

    Finite block lengths below one retain the historical clamp to one.
    Nonfinite values invalidate every draw. Typed and legacy Threefry keys keep
    their own JAX streams; an equal seed across PRNG implementations is not a
    stream guarantee. The block-length array owns the probability dtype.
    """
    _validate_counts(n_observations, num_replicates, chunk)
    length = jnp.asarray(block_length, dtype=jnp.result_type(block_length, 1.0))
    if length.ndim != 0 or not jnp.issubdtype(length.dtype, jnp.floating):
        message = "block_length must be a real floating scalar."
        raise ValueError(message)
    probability = 1.0 / jnp.maximum(length, 1.0)

    def draw(draw_key: PRNGKey) -> Array:
        """Retain split order and scan recurrence from the scalar resampler."""
        key_initial, key_scan = random.split(draw_key)
        start = random.randint(key_initial, (), 0, n_observations)

        def step(previous: Array, scan_key: PRNGKey) -> tuple[Array, Array]:
            """Continue circularly or restart at a uniformly sampled row."""
            key_jump, key_index = random.split(scan_key)
            jump = random.uniform(key_jump, ()) < probability
            new_index = random.randint(key_index, (), 0, n_observations)
            next_index = jnp.where(jump, new_index, (previous + 1) % n_observations)
            return next_index, next_index

        _, tail = lax.scan(step, start, random.split(key_scan, n_observations - 1))
        return jnp.concatenate([jnp.reshape(start, (1,)), tail])

    rows = map_replicates(draw, random.split(key, num_replicates), chunk=chunk)
    return RowResampleSchedule(
        indices=rows,
        draw_ids=jnp.arange(num_replicates, dtype=jnp.int32),
        valid=jnp.full((num_replicates,), jnp.isfinite(length), dtype=bool),
        n_observations=n_observations,
        scheme="stationary",
    )


def _validate_value_tree(values: object) -> None:
    """Reject unsupported containers and non-scalar/vector real array leaves."""
    if isinstance(values, tuple) and values:
        for value in values:
            _validate_value_tree(value)
        return
    if (
        isinstance(values, dict)
        and values
        and all(isinstance(key, str) for key in values)
    ):
        for value in values.values():
            _validate_value_tree(value)
        return
    if (
        isinstance(values, jax.Array)
        and values.ndim <= 1
        and jnp.issubdtype(values.dtype, jnp.floating)
    ):
        return
    message = (
        "Replicate values must be nonempty tuple/dict pytrees "
        "of real scalar/vector JAX arrays."
    )
    raise ValueError(message)


def evaluate_replicates[T](
    schedule: RowResampleSchedule,
    statistic_fn: Callable[[Array], ReplicateValue[T]],
    *,
    chunk: int = DEFAULT_REPLICATE_CHUNK,
) -> ReplicateResult[T]:
    """Evaluate a frozen row-index closure through the shared chunk executor.

    Each call receives one index vector, never resampled data implicitly.
    Schedules, including paired IDs, are reusable across different statistics.
    Invalid indices are clipped only to keep failed lanes numerically safe;
    their results always retain ``INVALID_SCHEDULE`` and are never summaries.
    """
    _validate_counts(schedule.n_observations, schedule.indices.shape[0], chunk)

    def evaluate(indices: Array) -> ReplicateValue[T]:
        """Evaluate and establish the promised limited result boundary."""
        result = statistic_fn(jnp.clip(indices, 0, schedule.n_observations - 1))
        _validate_value_tree(result.values)
        if (
            result.valid.ndim != 0
            or result.status.ndim != 0
            or result.valid.dtype != jnp.bool_
            or not jnp.issubdtype(result.status.dtype, jnp.signedinteger)
        ):
            message = (
                "Replicate validity/status must be boolean/signed-integer scalars."
            )
            raise ValueError(message)
        finite = jnp.all(
            jnp.stack(
                [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(result.values)]
            )
        )
        supplied_valid = result.valid & (result.status == ReplicateStatus.VALID)
        valid = supplied_valid & finite
        reason = jnp.where(
            supplied_valid,
            ReplicateStatus.NONFINITE_VALUE,
            jnp.where(result.status == 0, ReplicateStatus.INVALID_REFIT, result.status),
        )
        return ReplicateValue(
            result.values,
            valid,
            jnp.where(valid, ReplicateStatus.VALID, reason),
        )

    evaluated = lax.map(evaluate, schedule.indices, batch_size=chunk)
    valid = evaluated.valid & schedule.valid
    status = jnp.where(
        schedule.valid, evaluated.status, ReplicateStatus.INVALID_SCHEDULE
    )
    count = jnp.asarray(schedule.indices.shape[0], dtype=jnp.int32)
    valid_count = jnp.sum(valid, dtype=jnp.int32)
    return ReplicateResult(
        evaluated.values,
        schedule.draw_ids,
        valid,
        status,
        count,
        valid_count,
        count - valid_count,
    )


def percentile_summary[T](
    result: ReplicateResult[T], *, confidence_level: float = 0.95
) -> PercentileSummary[T]:
    """Calibrate supplied draws without regenerating or discarding any draw.

    Percentiles use JAX's linear interpolation, independently per real output
    component. The standard error uses sample variance (ddof=1). Configuration
    is static under JIT; map over an outer grid before calling this operation.
    """
    if not 0.0 < confidence_level < 1.0:
        message = "confidence_level must be in (0, 1)."
        raise ValueError(message)
    valid = (result.requested_count >= _MIN_SUMMARY_DRAWS) & (result.failed_count == 0)
    status = jnp.where(
        result.requested_count < _MIN_SUMMARY_DRAWS,
        ReplicateStatus.INSUFFICIENT_DRAWS,
        jnp.where(
            result.failed_count > 0, ReplicateStatus.FAILED_DRAWS, ReplicateStatus.VALID
        ),
    ).astype(jnp.int32)
    alpha = 1.0 - confidence_level

    def se(values: Array) -> Array:
        """Return a sample SE only when every requested draw is valid."""
        return jnp.where(valid, jnp.std(values, axis=0, ddof=1), jnp.nan)

    def lower(values: Array) -> Array:
        """Return componentwise lower percentile bounds."""
        return jnp.where(
            valid, jnp.percentile(values, 100 * alpha / 2, axis=0), jnp.nan
        )

    def upper(values: Array) -> Array:
        """Return componentwise upper percentile bounds."""
        return jnp.where(
            valid, jnp.percentile(values, 100 * (1 - alpha / 2), axis=0), jnp.nan
        )

    return PercentileSummary(
        jax.tree.map(se, result.values),
        jax.tree.map(lower, result.values),
        jax.tree.map(upper, result.values),
        valid,
        status,
        confidence_level,
    )


def refit_replicates(
    schedule: RowResampleSchedule,
    refit_fn: Callable[[Array], RefitResult],
    *,
    chunk: int = DEFAULT_REPLICATE_CHUNK,
) -> ReplicateResult[Array]:
    """Refit one frozen specification and retain all parameter draws.

    The closure owns aligned arrays and states which nuisance/preprocessing
    steps are frozen or repeated. No exceptions are swallowed: shape/program
    errors raise, while fit/solver failures use their numerical result flags.
    This adapter composes any actual fit exposing the minimal result protocol.
    """

    def evaluate(indices: Array) -> ReplicateValue[Array]:
        """Separate numerical solve success from scientific fit availability."""
        fit = refit_fn(indices)
        valid = fit.valid & fit.solver.converged
        status = jnp.where(
            fit.solver.converged,
            jnp.where(fit.valid, ReplicateStatus.VALID, ReplicateStatus.INVALID_REFIT),
            ReplicateStatus.SOLVER_FAILURE,
        ).astype(jnp.int32)
        return ReplicateValue(fit.parameters, valid, status)

    return evaluate_replicates(schedule, evaluate, chunk=chunk)


def linear_pairs_replicates(
    fit: LinearFit,
    schedule: RowResampleSchedule,
    *,
    chunk: int = DEFAULT_REPLICATE_CHUNK,
) -> ReplicateResult[Array]:
    """Refit aligned complete rows with columns, preprocessing and weights frozen.

    Design, outcome and analytic relative precisions use exactly the same row
    indices. An OLS fit has unit precisions and follows the same QR point solve.
    Named column order is retained; no column selection, imputation or nuisance
    tuning is repeated. Row-pairs bootstrap assumptions remain caller-owned.
    """
    if (
        fit.design.ndim != _MATRIX_RANK
        or schedule.n_observations != fit.design.shape[0]
    ):
        message = "The row schedule must match the fit's retained observation axis."
        raise ValueError(message)

    def refit(indices: Array) -> LinearFit:
        """Refit all aligned components under the exact retained schema."""
        return LeastSquares().estimate(
            LinearProblem(
                fit.design[indices],
                fit.outcome[indices],
                fit.relative_precision[indices],
                parameter_names=fit.parameter_names,
            )
        )

    result = refit_replicates(schedule, refit, chunk=chunk)
    valid = result.valid & fit.valid
    status = jnp.where(fit.valid, result.status, ReplicateStatus.INVALID_REFIT).astype(
        jnp.int32
    )
    valid_count = jnp.sum(valid, dtype=jnp.int32)
    return ReplicateResult(
        result.values,
        result.draw_ids,
        valid,
        status,
        result.requested_count,
        valid_count,
        result.requested_count - valid_count,
    )
