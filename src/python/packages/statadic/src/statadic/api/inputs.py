"""Adapt named regression inputs to validated observations."""

from __future__ import annotations

from collections.abc import (  # noqa: TC003 -- runtime annotations
    Iterable,
    Mapping,
    Sequence,
    Sized,
)
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from statadic.api.data import (
    MeasureColumn,
    Observations,
    RowIds,  # noqa: TC001 -- runtime annotations
    _check_array_precision,
    _numeric_column,
)
from statadic.api.roles import InputRole  # noqa: TC001 -- runtime annotations

__all__ = [
    "ArrowStream",
    "RegressionInput",
    "as_observations",
]


@runtime_checkable
class ArrowStream(Protocol):
    """A table exposing the Arrow C stream protocol."""

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        """Return an Arrow C stream capsule."""
        ...


type RegressionInput = Observations | Mapping[str, object] | ArrowStream


def _node_predictor(
    nodes: Observations,
    name: str,
    role: InputRole,
    positions: tuple[int, ...],
    row_ids: RowIds,
) -> jax.Array | RowIds | MeasureColumn:
    """Gather a full-node predictor in outcome-row order without changing its role."""
    if role == "numeric":
        return nodes.numeric(name)[jnp.asarray(positions, dtype=jnp.int32)]
    if role == "categorical":
        labels = nodes.categorical(name)
        return tuple(labels[position] for position in positions)
    if role == "measure":
        return MeasureColumn(row_ids, nodes.measures[name].take(positions).measures)
    message = f"Unsupported predictor role {role!r}."
    raise TypeError(message)


def _matching_predictor(
    observed: Observations,
    name: str,
    expected: jax.Array | RowIds | MeasureColumn,
) -> bool:
    """Compare duplicate columns without coercing numeric, label or mass domains."""
    if isinstance(expected, MeasureColumn):
        if name not in observed.measures:
            return False
        actual = observed.measures[name].measures
        target = expected.measures
        return all(
            bool(jnp.array_equal(a, b))
            for a, b in (
                (actual.data, target.data),
                (actual.counts, target.counts),
                (actual.masses, target.masses),
            )
        )
    if isinstance(expected, tuple):
        return (
            name not in observed.columns
            and name not in observed.measures
            and (observed.resolve_grouping(name) == expected)
        )
    return (
        name not in observed.labels
        and name not in observed.measures
        and bool(jnp.array_equal(observed.resolve_numeric(name), expected))
    )


def _validate_duplicate_predictors(
    nodes: Observations,
    observed: Observations,
    roles: Mapping[str, InputRole],
    positions: tuple[int, ...],
) -> None:
    """Reject conflicting copies of required predictors; omitted copies are allowed."""
    for name, role in roles.items():
        if name not in observed.available_columns:
            continue
        expected = _node_predictor(nodes, name, role, positions, observed.row_ids)
        if not _matching_predictor(observed, name, expected):
            message = f"Outcome predictor {name!r} conflicts with its full-node values."
            raise ValueError(message)


@runtime_checkable
class _ArrayLikeColumn(Protocol):
    """A host array column that can be copied and listed (for example NumPy)."""

    def copy(self) -> _ArrayLikeColumn:
        """Return an independent copy."""
        ...

    def tolist(self) -> object:
        """Return the values as Python objects."""
        ...


def _host_values(value: object) -> tuple[object, ...]:
    """Materialize one host vector without asserting the type of its elements."""
    if isinstance(value, (jax.Array, _ArrayLikeColumn)):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        message = "Column values must form a one-dimensional sequence."
        raise TypeError(message)
    return tuple(value)


class _MappingColumns:
    """A fit-time snapshot of undeclared mapping columns, converted only on use.

    NumPy-like values are copied so later caller mutation cannot change what
    inference resolves; JAX arrays are immutable and other values become tuples.
    """

    def __init__(self, values: Mapping[str, object], n_rows: int) -> None:
        self._values: dict[str, jax.Array | _ArrayLikeColumn | tuple[object, ...]] = {}
        for name, value in values.items():
            if isinstance(value, jax.Array):
                self._values[name] = value
            elif isinstance(value, _ArrayLikeColumn):
                self._values[name] = value.copy()
            else:
                self._values[name] = _host_values(value)
        self._n_rows = n_rows

    @property
    def names(self) -> tuple[str, ...]:
        """Return the snapshotted column names."""
        return tuple(self._values)

    @property
    def n_rows(self) -> int:
        """Return the fit-time row count."""
        return self._n_rows

    def labels(self, name: str) -> tuple[object, ...]:
        """Return one snapshotted column as labels for role validation."""
        return _host_values(self._values[name])

    def _numeric_rows(
        self, name: str, *, rows: tuple[int, ...] | None = None
    ) -> object:
        """Return an owned numeric candidate without losing its source dtype."""
        value = self._values[name]
        if rows is None:
            return value.copy() if isinstance(value, _ArrayLikeColumn) else value
        if isinstance(value, tuple):
            return tuple(value[row] for row in rows)
        _check_array_precision(value, name)
        return jnp.asarray(value)[jnp.asarray(rows, dtype=jnp.int32)]


def as_observations(  # noqa: C901, PLR0912, PLR0914, PLR0915 -- explicit host adaptation
    data: object, roles: Mapping[str, InputRole], *, broadcast_scalars: bool
) -> Observations:
    """Validate a regression input using its declared column roles.

    At fit time (``broadcast_scalars=False``) undeclared columns are kept
    unconverted as the observations' source, so inference can resolve one by
    name later; an unused unsupported column cannot break fitting. Prediction
    inputs keep no source.
    """
    if isinstance(data, Observations):
        return data
    if hasattr(data, "__arrow_c_stream__"):
        import pyarrow as pa  # noqa: PLC0415

        from statadic.api.tables import ArrowColumns, from_arrow  # noqa: PLC0415

        table = pa.table(data)
        selected = {
            name: role for name, role in roles.items() if name in table.column_names
        }
        if "measure" in roles.values():
            message = (
                "Measure roles require measures_from_arrow and explicit Observations."
            )
            raise TypeError(message)
        return from_arrow(
            table,
            numeric=tuple(name for name, role in selected.items() if role == "numeric"),
            categorical=tuple(
                name for name, role in selected.items() if role == "categorical"
            ),
            copy=None if broadcast_scalars else True,
            source=None
            if broadcast_scalars or len(selected) == table.num_columns
            else ArrowColumns(
                table.select(
                    [name for name in table.column_names if name not in selected]
                )
            ),
        )
    if hasattr(data, "collect"):
        message = "Collect the LazyFrame with collect() before fitting or predicting."
        raise TypeError(message)
    if not isinstance(data, Mapping):
        message = "Expected Observations, a mapping, or an Arrow-compatible frame."
        raise TypeError(message)
    values: dict[str, object] = {}
    for name, value in data.items():
        if not isinstance(name, str):
            message = "Column names must be strings."
            raise TypeError(message)
        values[name] = value
    unknown = set(values) - set(roles)
    if unknown and broadcast_scalars:
        message = f"Unknown columns {sorted(unknown)}; expected {sorted(roles)}."
        raise ValueError(message)
    lengths: dict[str, int] = {}
    for name, value in values.items():
        if isinstance(value, MeasureColumn):
            lengths[name] = len(value.row_ids)
        elif (
            isinstance(value, (str, int, float, bool))
            or getattr(value, "ndim", None) == 0
        ):
            if not broadcast_scalars:
                message = f"Fit column {name!r} must not be scalar."
                raise ValueError(message)
        else:
            if not isinstance(value, Sized):
                message = f"Column {name!r} must be a sized vector."
                raise TypeError(message)
            lengths[name] = len(value)
    if len(set(lengths.values())) > 1:
        message = f"Column lengths do not match: {lengths}."
        raise ValueError(message)
    n_rows = next(iter(lengths.values()), 1)
    numeric: dict[str, object] = {}
    labels: dict[str, Sequence[object]] = {}
    measures: dict[str, MeasureColumn] = {}
    undeclared: dict[str, object] = {}
    for name, value in values.items():
        role = roles.get(name)
        if role is None and not isinstance(value, MeasureColumn):
            # Only fits reach here (prediction rejects unknown names above).
            undeclared[name] = value
            continue
        if role == "measure" or isinstance(value, MeasureColumn):
            if role not in (None, "measure"):
                message = f"Column {name!r} is declared {role}, not measure."
                raise TypeError(message)
            if not isinstance(value, MeasureColumn):
                message = f"Measure column {name!r} requires a MeasureColumn."
                raise TypeError(message)
            measures[name] = value
            continue
        scalar = name not in lengths
        if role == "categorical":
            if scalar:
                scalar_value = (
                    value.tolist()
                    if isinstance(value, (jax.Array, _ArrayLikeColumn))
                    else value
                )
                items = [scalar_value] * n_rows
            else:
                items = _host_values(value)
            labels[name] = items
        elif scalar:
            # Validate the one host scalar, then broadcast it inside JAX.
            numeric[name] = jnp.broadcast_to(_numeric_column(name, [value]), (n_rows,))
        else:
            numeric[name] = value
    identities = (
        next(iter(measures.values())).row_ids if measures else tuple(range(n_rows))
    )
    return Observations(
        numeric,
        labels=labels,
        measures=measures,
        row_ids=identities,
        source=_MappingColumns(undeclared, n_rows) if undeclared else None,
    )
