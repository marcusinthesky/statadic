"""Panel indexing and weighted absorption without dense dummy matrices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Int, Integer  # noqa: TC002

from statadic.core.typing import Array, ArrayLike  # noqa: TC001

__all__ = [
    "PanelIndex",
    "PanelProjection",
    "PanelSample",
    "absorb_panel_effects",
    "panel_group_mean",
    "prepare_panel_index",
    "prepare_panel_sample",
]


_DESIGN_DIMENSIONS = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PanelIndex:
    """Complete row alignment with contiguous labels and original numeric keys.

    Cardinalities are static shape metadata. Row labels and original IDs are
    dynamic arrays, so two datasets of the same shape share compilation.
    Preparation rejects duplicate entity/time keys; neither sorting nor missing
    row deletion occurs inside a fitter.
    """

    entity: Int[Array, "*batch rows"]  # noqa: F722
    time: Int[Array, "*batch rows"]  # noqa: F722
    entity_ids: Integer[Array, "*batch entities"]  # noqa: F722
    time_ids: Integer[Array, "*batch periods"]  # noqa: F722
    n_entities: int = field(metadata={"static": True})
    n_periods: int = field(metadata={"static": True})
    effect_rank: Int[Array, "*batch"]  # noqa: F722
    difference_rows: Int[Array, "*batch differences"]  # noqa: F722
    previous_rows: Int[Array, "*batch differences"]  # noqa: F722


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PanelProjection:
    """Weighted residual projection with independently retained convergence."""

    values: Array
    effects: Array
    converged: Bool[Array, "*batch"]  # noqa: F722
    iterations: Int[Array, "*batch"]  # noqa: F722
    error: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PanelSample:
    """Explicit eager row/column decisions before fixed-shape estimation."""

    design: Array
    outcome: Array
    relative_precision: Array
    index: PanelIndex
    retained_rows: Int[Array, "*batch rows"]  # noqa: F722
    retained_columns: Int[Array, "*batch columns"]  # noqa: F722


def _singleton_mask(
    entity: jnp.ndarray,
    time: jnp.ndarray,
    keep: jnp.ndarray,
    *,
    entity_effects: bool,
    time_effects: bool,
) -> jnp.ndarray:
    """Find the retained two-core using explicit selected effect groups."""
    while True:
        previous = keep.copy()
        for labels, selected in ((entity, entity_effects), (time, time_effects)):
            if selected:
                values, counts = jnp.unique(labels[keep], return_counts=True)
                keep = keep & ~jnp.isin(labels, values[counts == 1])
        if jnp.array_equal(previous, keep):
            return keep


def _unabsorbed_columns(
    x: Array,
    weights: Array,
    index: PanelIndex,
    *,
    entity_effects: bool,
    time_effects: bool,
    intercept_column: int | None,
) -> jnp.ndarray:
    """Select an ordered independent basis only after declaring effect absorption."""
    if jnp.linalg.matrix_rank(jnp.asarray(x)) < x.shape[1]:
        message = (
            "Resolve original design rank deficiency before dropping absorbed columns."
        )
        raise ValueError(message)
    projection = absorb_panel_effects(
        x,
        index,
        weights=weights,
        entity_effects=entity_effects,
        time_effects=time_effects,
    )
    if not bool(projection.converged):
        message = "Panel absorption did not converge during sample preparation."
        raise ValueError(message)
    projected = jnp.asarray(projection.values)
    if intercept_column is not None:
        if not 0 <= intercept_column < x.shape[1] or not jnp.all(
            jnp.asarray(x)[:, intercept_column] == 1
        ):
            message = "intercept_column must identify an explicit unit column."
            raise ValueError(message)
        projected = projected + jnp.average(
            jnp.asarray(x), axis=0, weights=jnp.asarray(weights)
        )
    cutoff = (
        max(x.shape)
        * jnp.finfo(projected.dtype).eps
        * jnp.linalg.norm(jnp.asarray(x), ord=2)
    )
    retained: list[int] = []
    for column in range(x.shape[1]):
        if jnp.linalg.matrix_rank(
            projected[:, jnp.asarray([*retained, column])], tol=cutoff
        ) > len(retained):
            retained.append(column)
    if not retained:
        message = "Every regressor is absorbed by the selected effects."
        raise ValueError(message)
    return jnp.asarray(retained, dtype=jnp.int32)


def prepare_panel_sample(
    design: ArrayLike,
    outcome: ArrayLike,
    entity_ids: ArrayLike,
    time_ids: ArrayLike,
    *,
    relative_precision: ArrayLike | None = None,
    missing: Literal["raise", "drop"] = "raise",
    singletons: Literal["keep", "drop"] = "keep",
    entity_effects: bool = True,
    time_effects: bool = False,
    intercept_column: int | None = None,
    drop_absorbed: bool = False,
) -> PanelSample:
    """Make optional missing-row, singleton and absorbed-column choices explicit.

    Defaults retain singletons and reject missing/absorbed input. The fitter
    itself never changes the sample. Singleton removal iterates both selected
    effects to their two-core. Absorbed-column removal preserves input order
    and records every retained column; it does not repair an originally
    rank-deficient design. Keys must be complete unique integer pairs.
    """
    x, y = jnp.asarray(design), jnp.asarray(outcome)
    keys = prepare_panel_index(entity_ids, time_ids)
    e, t = jnp.asarray(entity_ids), jnp.asarray(time_ids)
    if x.ndim != _DESIGN_DIMENSIONS or y.shape != (x.shape[0],) or e.shape != y.shape:
        message = "Panel arrays and keys must have aligned rows."
        raise ValueError(message)
    w = (
        jnp.ones(len(y), dtype=jnp.result_type(x, y, 1.0))
        if relative_precision is None
        else jnp.asarray(relative_precision)
    )
    if w.shape != y.shape or jnp.any(w <= 0):
        message = "Panel weights must be aligned positive relative precisions."
        raise ValueError(message)
    keep = jnp.all(jnp.isfinite(x), axis=1) & jnp.isfinite(y) & jnp.isfinite(w)
    if missing == "raise" and not jnp.all(keep):
        message = "Missing panel data require explicit missing='drop'."
        raise ValueError(message)
    if missing not in ("raise", "drop") or singletons not in ("keep", "drop"):
        message = "Unknown missing or singleton preparation policy."
        raise ValueError(message)
    if singletons == "drop":
        keep = _singleton_mask(
            e, t, keep, entity_effects=entity_effects, time_effects=time_effects
        )
    rows = jnp.flatnonzero(keep)
    if not len(rows):
        message = "Panel preparation retained no observations."
        raise ValueError(message)
    index = keys if jnp.all(keep) else prepare_panel_index(e[rows], t[rows])
    xx, yy, ww = jnp.asarray(x[rows]), jnp.asarray(y[rows]), jnp.asarray(w[rows])
    columns = jnp.arange(x.shape[1])
    if drop_absorbed:
        columns = _unabsorbed_columns(
            xx,
            ww,
            index,
            entity_effects=entity_effects,
            time_effects=time_effects,
            intercept_column=intercept_column,
        )
        xx = xx[:, jnp.asarray(columns)]
    return PanelSample(xx, yy, ww, index, jnp.asarray(rows), jnp.asarray(columns))


def _lossless_keys(keys: ArrayLike) -> tuple[Array, list[int]]:
    """Read integer keys once and detect any narrowing by the active JAX dtype."""
    message = (
        "Panel keys cannot be represented in the active JAX integer dtype; "
        "enable JAX x64 or encode keys within its range."
    )
    try:
        converted = jnp.asarray(keys)
    except OverflowError as exc:
        raise ValueError(message) from exc
    if converted.ndim != 1 or not converted.size:
        msg = "Panel keys must be nonempty aligned vectors."
        raise ValueError(msg)
    if not jnp.issubdtype(converted.dtype, jnp.integer):
        msg = "Encode entity and actual time keys as integers before preparation."
        raise ValueError(msg)
    host = converted.tolist()
    if converted is not keys:
        original = keys.tolist() if hasattr(keys, "tolist") else keys
        if isinstance(original, tuple):
            original = list(original)
        if host != original:
            raise ValueError(message)
    return converted, host


def prepare_panel_index(entity_ids: ArrayLike, time_ids: ArrayLike) -> PanelIndex:
    """Encode complete integer entity/time keys while preserving row order."""
    entities, entity_values = _lossless_keys(entity_ids)
    periods, period_values = _lossless_keys(time_ids)
    if periods.shape != entities.shape:
        message = "Panel keys must be nonempty aligned vectors."
        raise ValueError(message)
    # Keys are host metadata. Transfer each vector once instead of dispatching
    # a device scalar read per row while constructing the predecessor pairs.
    rows = tuple(zip(entity_values, period_values, strict=True))
    lookup = {pair: row for row, pair in enumerate(rows)}
    if len(lookup) != len(rows):
        message = "Duplicate entity/time observations are not permitted."
        raise ValueError(message)
    entity_codes = {key: i for i, key in enumerate(sorted(set(entity_values)))}
    time_codes = {key: i for i, key in enumerate(sorted(set(period_values)))}
    entity = [entity_codes[e] for e, _ in rows]
    time = [time_codes[t] for _, t in rows]
    parents = list(range(len(entity_codes) + len(time_codes)))

    def root(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for left, right in zip(entity, time, strict=True):
        parents[root(left)] = root(len(entity_codes) + right)
    components = len({root(node) for node in range(len(parents))})
    pairs = [
        (row, lookup[(e, t - 1)])
        for row, (e, t) in enumerate(rows)
        if (e, t - 1) in lookup
    ]
    return PanelIndex(
        entity=jnp.asarray(entity),
        time=jnp.asarray(time),
        entity_ids=jnp.asarray(tuple(entity_codes), dtype=entities.dtype),
        time_ids=jnp.asarray(tuple(time_codes), dtype=periods.dtype),
        n_entities=len(entity_codes),
        n_periods=len(time_codes),
        effect_rank=jnp.asarray(len(entity_codes) + len(time_codes) - components),
        difference_rows=jnp.asarray([p[0] for p in pairs], dtype=jnp.int32),
        previous_rows=jnp.asarray([p[1] for p in pairs], dtype=jnp.int32),
    )


def panel_group_mean(
    values: Array, groups: Array, weights: Array, *, n_groups: int
) -> Array:
    """Return weighted group means; unoccupied groups retain NaN."""
    sums = jnp.zeros((n_groups, *values.shape[1:]), dtype=values.dtype)
    expand = (slice(None),) + (None,) * (values.ndim - 1)
    sums = sums.at[groups].add(values * weights[expand])
    counts = jnp.zeros(n_groups, dtype=weights.dtype).at[groups].add(weights)
    return sums / counts[expand]


def _sweep_panel_effects(
    current: Array,
    index: PanelIndex,
    precision: Array,
    *,
    entity_effects: bool,
    time_effects: bool,
) -> Array:
    """Subtract weighted entity means, then weighted time means, once."""
    result = current
    if entity_effects:
        result = (
            result
            - panel_group_mean(
                result, index.entity, precision, n_groups=index.n_entities
            )[index.entity]
        )
    if time_effects:
        result = (
            result
            - panel_group_mean(result, index.time, precision, n_groups=index.n_periods)[
                index.time
            ]
        )
    return result


def absorb_panel_effects(
    values: Array,
    index: PanelIndex,
    *,
    weights: Array | None = None,
    entity_effects: bool = True,
    time_effects: bool = False,
    tolerance: float = 1e-8,
    max_steps: int = 512,
) -> PanelProjection:
    """Project out selected effects by weighted alternating group reductions.

    Works with unbalanced panels and multiple response/design columns. The
    caller decides whether to restore grand means for an explicit intercept.
    Nonfinite inputs, nonpositive weights or exhausted absorption invalidate
    the result; a converged numerical projection does not establish slope rank.
    """
    if values.shape[0] != index.entity.shape[0]:
        message = "Panel projection rows must match the prepared index."
        raise ValueError(message)
    precision = (
        jnp.ones(values.shape[0], dtype=values.dtype) if weights is None else weights
    )
    if precision.shape != (values.shape[0],):
        message = "Panel weights must match observation rows."
        raise ValueError(message)
    if max_steps < 1 or tolerance <= 0:
        message = "Positive absorption budget and tolerance are required."
        raise ValueError(message)

    scale = jnp.maximum(jnp.max(jnp.abs(values)), 1)

    def sweep(
        state: tuple[Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array]:
        current, _, iterations, _ = state
        candidate = _sweep_panel_effects(
            current,
            index,
            precision,
            entity_effects=entity_effects,
            time_effects=time_effects,
        )
        change = jnp.max(jnp.abs(candidate - current))
        return candidate, change <= tolerance * scale, iterations + 1, change

    def step(
        _: Array, state: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array]:
        # Fixed trip count preserves reverse-mode AD. In an unbatched call
        # the remaining sweeps skip group reductions once the fit converges.
        return jax.lax.cond(state[1], lambda: state, lambda: sweep(state))

    transformed, converged, iterations, error = jax.lax.fori_loop(
        0,
        max_steps,
        step,
        (
            values,
            jnp.zeros((), dtype=bool),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=values.dtype),
        ),
    )
    valid = (
        jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(precision))
        & jnp.all(precision > 0)
    )
    valid &= jnp.all(jnp.isfinite(transformed)) & converged
    return PanelProjection(
        values=jnp.where(valid, transformed, jnp.nan),
        effects=jnp.where(valid, values - transformed, jnp.nan),
        converged=valid,
        iterations=iterations,
        error=error,
    )
