"""Aligned host observations with separate scalar and empirical-measure columns.

Observation identities describe whole statistical units. Atoms within an
empirical measure are not extra regression rows. Preparation is eager; numeric
payloads remain immutable JAX arrays for the scientific owners to consume.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence  # noqa: TC003  # runtime annotations
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from statadic.api.reporting import MetadataValue, _row_identity  # noqa: TC001
from statadic.core.typing import Array  # noqa: TC001
from statadic.sample.packed import PackedClouds, checked_packed_clouds  # noqa: TC001

__all__ = [
    "ColumnSource",
    "MeasureColumn",
    "Observations",
    "PanelLayout",
    "RelativePrecision",
    "RowId",
    "RowIds",
    "SpatialPanelData",
]

type RowId = int | str
type RowIds = tuple[RowId, ...]


_DEFAULT_WIDTH = 4


@runtime_checkable
class ColumnSource(Protocol):
    """Fit-time columns kept unconverted until an inference step names one.

    Model roles are converted eagerly into :class:`Observations`; a source keeps
    the remaining fit-time columns so a later decision, such as clustering, can
    resolve one by name in the role it needs. Rows follow the source's order.
    """

    @property
    def names(self) -> tuple[str, ...]:
        """Return every column name the source can resolve."""
        ...

    @property
    def n_rows(self) -> int:
        """Return the number of source rows."""
        ...

    def labels(self, name: str) -> Sequence[object]:
        """Return raw host values for validation when a grouping role is named."""
        ...


@runtime_checkable
class _NumericColumnSource(Protocol):
    """Optional source capability preserving explicit numeric payload dtypes."""

    def _numeric_rows(
        self, name: str, *, rows: tuple[int, ...] | None = None
    ) -> object:
        """Return owned selected source rows before numeric value validation."""
        ...


@dataclass(frozen=True)
class RelativePrecision:
    """A column representing inverse relative error variance."""

    column: str


@dataclass(frozen=True)
class PanelLayout:
    """Entity labels and actual integer time keys, separately from effects."""

    entity: str
    time: str


@dataclass(frozen=True)
class SpatialPanelData:
    """Full graph-node covariates and separately observed outcome records.

    The two row axes may differ. Both carry entity/time roles selected by a
    ``PanelLayout``; outcome row identities are retained as spell identities.
    """

    nodes: Observations
    outcomes: Observations


def _label_values(values: Sequence[object], context: str) -> tuple[RowId, ...]:
    """Decode integer/string labels without coercion or boolean identities."""
    result: list[RowId] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, Integral)):
            message = (
                f"{context} requires nonmissing integer/string labels, excluding bool."
            )
            raise TypeError(message)
        result.append(value if isinstance(value, str) else int(value))
    return tuple(result)


def _identities(values: Sequence[object]) -> tuple[RowId, ...]:
    """Validate unique typed identities without string coercion."""
    result = _label_values(values, "Observation identities")
    if len(set(result)) != len(result):
        message = "Observation identities must be unique."
        raise ValueError(message)
    return result


def _check_array_precision(value: object, name: str) -> None:
    """Reject explicit wide inputs before JAX can silently narrow them."""
    if isinstance(value, (list, tuple)):
        for item in value:
            _check_array_precision(item, name)
        return
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return
    try:
        original = jnp.dtype(dtype)
    except TypeError as exc:
        message = f"Column {name!r} has an unsupported dtype; use from_polars."
        raise TypeError(message) from exc
    if (
        original.kind in "fiu"
        and original.itemsize > _DEFAULT_WIDTH
        and not jax.config.x64_enabled
    ):
        message = f"Column {name!r} requires JAX x64 or an explicit cast."
        raise ValueError(message)


def _numeric_column(name: str, values: object) -> Array:
    """Normalize one complete real vector, checking precision before conversion."""
    _check_array_precision(values, name)
    array = jnp.asarray(values)
    if array.ndim != 1:
        message = f"Numeric column {name!r} must have a one-dimensional vector shape."
        raise ValueError(message)
    if not jnp.issubdtype(array.dtype, jnp.number) or jnp.iscomplexobj(array):
        message = f"Numeric column {name!r} must contain real numbers."
        raise TypeError(message)
    if not bool(jnp.all(jnp.isfinite(array))):
        message = f"Numeric column {name!r} contains missing/nonfinite values."
        raise ValueError(message)
    return array


def _resolve_row_order(data: Observations, order_by: str | None) -> Array | None:
    """Resolve one unique finite real ordering key from retained observations."""
    if order_by is None:
        return None
    if not isinstance(order_by, str) or not order_by:
        message = "order_by must be a nonempty column name or None."
        raise TypeError(message)
    keys = data.resolve_numeric(order_by)
    order = jnp.argsort(keys)
    sorted_keys = keys[order]
    if bool(jnp.any(sorted_keys[1:] == sorted_keys[:-1])):
        message = "order_by requires unique numeric keys; ties do not define lag order."
        raise ValueError(message)
    return order


def _row_order_metadata(
    data: Observations, order_by: str | None, indices: Array | None
) -> dict[str, MetadataValue]:
    """Describe an already-resolved permutation without sorting or fitting again.

    Metadata stays fixed-size: row identities are digests, never id tuples.
    """
    rows = data.row_ids
    metadata: dict[str, MetadataValue] = {
        "rows": _row_identity(rows),
        "order_by": order_by,
        "direction": "retained" if order_by is None else "ascending",
        "lag_unit": "adjacent ordered observations",
    }
    if indices is not None:
        metadata["calculation_order"] = _row_identity(
            [rows[int(index)] for index in jax.device_get(indices).tolist()]
        )
    return metadata


def _category(name: str, values: Sequence[object], n_rows: int) -> tuple[RowId, ...]:
    """Own categorical values without coercing label types."""
    labels = _label_values(values, f"Categorical column {name!r}")
    if len(labels) != n_rows:
        message = f"Categorical column {name!r} is not observation-aligned."
        raise ValueError(message)
    return labels


def _source_positions(
    source: ColumnSource | None, positions: Sequence[int] | None, n_rows: int
) -> tuple[int, ...] | None:
    """Validate the retained observation-to-source row mapping."""
    rows = None if positions is None else tuple(positions)
    if source is None:
        if rows is not None:
            message = "Source rows require a column source."
            raise ValueError(message)
        return None
    if (len(rows) if rows is not None else source.n_rows) != n_rows:
        message = "A column source must be aligned with the observation rows."
        raise ValueError(message)
    if rows is not None and any(
        isinstance(row, bool)
        or not isinstance(row, int)
        or row < 0
        or row >= source.n_rows
        for row in rows
    ):
        message = "Source rows require valid integer positions in the source."
        raise ValueError(message)
    return rows


@dataclass(frozen=True)
class MeasureColumn:
    """Empirical measures paired with the ordered identities of their observations.

    Identity is host context owned by the API layer; :class:`PackedClouds`
    stays a purely numeric pytree. ``measures`` row ``i`` is observation
    ``row_ids[i]``. :class:`Observations` accepts the column only when
    ``row_ids`` equals its own identities exactly (same ids, same order).
    """

    row_ids: tuple[RowId, ...]
    measures: PackedClouds

    def __post_init__(self) -> None:
        """Validate unique typed identities aligned with the measure rows."""
        identities = _identities(self.row_ids)
        if self.measures.counts.ndim != 1:
            message = "A measure column requires one collection axis."
            raise ValueError(message)
        checked_packed_clouds(
            self.measures.data,
            self.measures.counts,
            self.measures.masses,
            require_positive_counts=True,
        )
        if len(identities) != self.measures.n_observations:
            message = "Measure column identities must match its number of measures."
            raise ValueError(message)
        object.__setattr__(self, "row_ids", identities)

    def take(self, indices: Sequence[int]) -> MeasureColumn:
        """Select whole measures and their identities together, in order."""
        rows = tuple(indices)
        return MeasureColumn(
            tuple(self.row_ids[item] for item in rows),
            self.measures.take(jnp.asarray(rows, dtype=jnp.int32)),
        )


@dataclass(frozen=True, init=False)
class Observations:
    """One explicit outer observation axis shared by all input roles.

    Construct from named numeric columns, categorical labels and empirical-measure
    columns. All column lengths must agree. Missing/nonfinite
    numeric values raise; no independent per-column row dropping is performed.
    Mapping containers are copied and made read-only during construction. Each
    measure column carries its own observation identities, which must equal
    ``row_ids`` exactly; mismatches raise and nothing is reordered or dropped.

    An optional ``source`` keeps the remaining fit-time columns unconverted;
    :meth:`resolve_grouping` reads one only when an inference step names it.
    ``source_rows`` maps observation rows to source rows after :meth:`take`.
    """

    row_ids: tuple[RowId, ...]
    columns: Mapping[str, Array]
    labels: Mapping[str, tuple[RowId, ...]]
    measures: Mapping[str, MeasureColumn]
    source: ColumnSource | None = field(default=None, repr=False, compare=False)
    source_rows: tuple[int, ...] | None = field(default=None, repr=False, compare=False)

    def __init__(
        self,
        columns: Mapping[str, object],
        *,
        row_ids: Sequence[RowId] | None = None,
        labels: Mapping[str, Sequence[object]] | None = None,
        measures: Mapping[str, MeasureColumn] | None = None,
        source: ColumnSource | None = None,
        source_rows: Sequence[int] | None = None,
    ) -> None:
        """Normalize raw columns once and retain an immutable aligned sample."""
        numeric = {
            name: _numeric_column(name, value) for name, value in columns.items()
        }
        categories: dict[str, tuple[object, ...]] = (
            {}
            if labels is None
            else {name: tuple(value) for name, value in labels.items()}
        )
        empirical = {} if measures is None else dict(measures)
        identities: tuple[RowId, ...]
        if row_ids is None:
            lengths = [array.shape[0] for array in numeric.values()]
            lengths.extend(len(value) for value in categories.values())
            lengths.extend(len(value.row_ids) for value in empirical.values())
            if not lengths:
                message = (
                    "Supply at least one column or explicit observation identities."
                )
                raise ValueError(message)
            identities = tuple(range(lengths[0]))
        else:
            identities = _identities(row_ids)
        n_rows = len(identities)
        if not n_rows:
            message = "Observations require at least one row."
            raise ValueError(message)
        rows = _source_positions(source, source_rows, n_rows)
        for name, array in numeric.items():
            if array.shape[0] != n_rows:
                message = f"Numeric column {name!r} must have shape ({n_rows},)."
                raise ValueError(message)
        categories = {
            name: _category(name, value, n_rows) for name, value in categories.items()
        }
        for name, values in empirical.items():
            if values.row_ids != identities:
                message = (
                    f"Measure column {name!r} identities must equal the "
                    "observation row_ids exactly (same ids, same order)."
                )
                raise ValueError(message)
        names = (*numeric, *categories, *empirical)
        if len(set(names)) != len(names) or any(
            not isinstance(name, str) or not name for name in names
        ):
            message = "Column names must be nonempty and unique across input kinds."
            raise ValueError(message)
        object.__setattr__(self, "row_ids", identities)
        object.__setattr__(self, "columns", MappingProxyType(numeric))
        object.__setattr__(self, "labels", MappingProxyType(categories))
        object.__setattr__(self, "measures", MappingProxyType(empirical))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_rows", rows)

    @property
    def n_rows(self) -> int:
        """Return the number of whole observations, independently of atom counts."""
        return len(self.row_ids)

    def numeric(self, name: str) -> Array:
        """Select a numeric role without changing observation order."""
        return self.columns[name]

    def resolve_numeric(self, name: str) -> Array:
        """Resolve a numeric training role from the immutable fit-time snapshot.

        Unused source columns are checked only when requested. Selected rows
        retain their observation-to-source mapping and explicit source precision.
        """
        if name in self.columns:
            return self.numeric(name)
        if self.source is None or name not in self.source.names:
            available = ", ".join(self.available_columns)
            message = (
                f"Column {name!r} is not in the fitted data; available: {available}."
            )
            raise KeyError(message)
        if isinstance(self.source, _NumericColumnSource):
            # This hook is intentionally outside the public ColumnSource API.
            values = self.source._numeric_rows(  # noqa: SLF001 -- private source protocol
                name, rows=self.source_rows
            )
        else:
            labels = self.source.labels(name)
            values = (
                labels
                if self.source_rows is None
                else tuple(labels[row] for row in self.source_rows)
            )
        array = _numeric_column(name, values)
        if array.shape != (self.n_rows,):
            message = f"Numeric column {name!r} is not observation-aligned."
            raise ValueError(message)
        return array

    def categorical(self, name: str) -> tuple[RowId, ...]:
        """Select already-aligned categorical labels."""
        if name not in self.labels:
            message = (
                f"Categorical column {name!r} requires labels supplied through "
                "Observations(labels=...); numeric columns are separate roles."
            )
            raise ValueError(message)
        return self.labels[name]

    def grouping_labels(self, name: str) -> tuple[RowId, ...]:
        """Select categorical or integer numeric keys in observation order."""
        if name in self.labels:
            return self.categorical(name)
        values = self.numeric(name)
        if not jnp.issubdtype(values.dtype, jnp.integer):
            message = "Numeric grouping/unit/time columns require integer keys."
            raise ValueError(message)
        return tuple(int(value) for value in jax.device_get(values).tolist())

    @property
    def available_columns(self) -> tuple[str, ...]:
        """Return sample columns, then source columns an inference step can name."""
        names = (*self.columns, *self.labels, *self.measures)
        extra = () if self.source is None else self.source.names
        return tuple(dict.fromkeys((*names, *extra)))

    def resolve_grouping(self, name: str) -> tuple[RowId, ...]:
        """Resolve grouping keys from the sample, or else from the fit-time source.

        Inference-only columns, such as cluster labels, are read and validated
        here rather than when fitting; prepared model roles stay strict.
        """
        if name in self.labels or name in self.columns:
            return self.grouping_labels(name)
        if self.source is None or name not in self.source.names:
            available = ", ".join(self.available_columns)
            message = (
                f"Column {name!r} is not in the fitted data; available: {available}."
            )
            raise KeyError(message)
        values = tuple(self.source.labels(name))
        if self.source_rows is not None:
            values = tuple(values[row] for row in self.source_rows)
        return _category(name, values, self.n_rows)

    def measure(self, name: str) -> PackedClouds:
        """Select a distribution-valued column without pooling its atoms."""
        return self.measures[name].measures

    def take(self, indices: Sequence[int]) -> Observations:
        """Select a unique ordered subset; resampling uses numerical schedules."""
        rows = tuple(indices)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in rows):
            message = (
                "Observation selection requires integer positions, excluding bool."
            )
            raise TypeError(message)
        if any(index < 0 or index >= self.n_rows for index in rows):
            message = "Observation selection is outside the retained sample."
            raise IndexError(message)
        if len(set(rows)) != len(rows):
            message = (
                "Host observation selection requires unique rows; "
                "use a resample schedule for repeated draws."
            )
            raise ValueError(message)
        index = jnp.asarray(rows, dtype=jnp.int32)
        return Observations(
            {name: values[index] for name, values in self.columns.items()},
            row_ids=tuple(self.row_ids[item] for item in rows),
            labels={
                name: tuple(values[item] for item in rows)
                for name, values in self.labels.items()
            },
            measures={
                name: values.take(rows) for name, values in self.measures.items()
            },
            source=self.source,
            source_rows=None
            if self.source is None
            else tuple(
                rows
                if self.source_rows is None
                else (self.source_rows[item] for item in rows)
            ),
        )
