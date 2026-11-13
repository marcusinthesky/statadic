"""Optional Polars/Arrow preparation into the same aligned observation contract.

Import this module after installing ``statadic[data]``. Primitive numeric columns
cross Arrow/DLPack directly; labels remain on the host. Chunk consolidation and
the scientific fit's deliberate input ownership may still allocate memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence  # noqa: TC003  # runtime annotations
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import pyarrow as pa

from statadic.api.data import (  # noqa: TC001  # runtime annotations
    ColumnSource,
    MeasureColumn,
    Observations,
    RowId,
    _label_values,
)
from statadic.core.typing import Array  # noqa: TC001
from statadic.sample.packed import PackedClouds

__all__ = [
    "ArrowColumns",
    "ArrowFrame",
    "from_arrow",
    "from_polars",
    "measures_from_arrow",
    "measures_from_polars",
]

_WIDE_BITS = 64


@runtime_checkable
class ArrowFrame(Protocol):
    """A frame that exports an Arrow table."""

    def to_arrow(self) -> pa.Table:
        """Return the frame's Arrow representation."""
        ...


def _labels(table: pa.Table, name: str) -> tuple[RowId, ...]:
    """Read integer/string identities without coercion or missing values."""
    values: list[object] = table.column(name).to_pylist()
    return _label_values(values, f"Label column {name!r}")


class ArrowColumns:
    """A fit-time Arrow table whose extra columns resolve only when named.

    Arrow buffers may alias writable caller memory. An IPC snapshot owns the
    retained bytes without interpreting unused columns or changing their schema.
    """

    def __init__(self, table: pa.Table) -> None:
        """Copy the table's buffers before a caller can mutate their backing store."""
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        self._table = pa.ipc.open_stream(sink.getvalue()).read_all()

    @property
    def names(self) -> tuple[str, ...]:
        """Return every column name in the retained table."""
        return tuple(self._table.column_names)

    @property
    def n_rows(self) -> int:
        """Return the retained table's row count."""
        return int(self._table.num_rows)

    def labels(self, name: str) -> tuple[object, ...]:
        """Return fresh host values for the requested role's own validation."""
        return tuple(self._table.column(name).to_pylist())

    def _numeric_rows(self, name: str, *, rows: tuple[int, ...] | None = None) -> Array:
        """Read a numeric candidate with its original precision and null checks."""
        selected = self._table.select([name])
        if rows is not None:
            selected = selected.take(pa.array(rows, type=pa.int64()))
        return _dlpack_column(selected, name, "Numeric", copy=True)


def _real_column(table: pa.Table, name: str, role: str) -> pa.ChunkedArray:
    """Return a complete primitive numeric column or raise."""
    column = table.column(name)
    if column.null_count:
        message = f"{role} column {name!r} contains nulls; missing='raise'."
        raise ValueError(message)
    dtype = column.type
    if not (pa.types.is_floating(dtype) or pa.types.is_integer(dtype)):
        message = f"{role} column {name!r} must have primitive integer/float type."
        raise TypeError(message)
    if dtype.bit_width == _WIDE_BITS and not jax.config.x64_enabled:
        message = (
            f"{role} column {name!r} requires caller-enabled JAX x64 "
            "or an explicit source cast."
        )
        raise ValueError(message)
    return column


def _dlpack_column(
    table: pa.Table, name: str, role: str, *, copy: bool | None
) -> Array:
    """Import one primitive real column through DLPack, without host conversion."""
    column = _real_column(table, name, role)
    array = column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()
    if copy:
        # Some CPU jaxlib releases alias aligned DLPack buffers even with
        # copy=True. Complete an explicit device copy before caller mutation
        # can race fit-time ownership. Keep the other DLPack policies intact.
        return jnp.array(jnp.from_dlpack(array), copy=True).block_until_ready()
    return jnp.from_dlpack(array, copy=copy)


def from_arrow(
    table: pa.Table,
    *,
    numeric: tuple[str, ...],
    categorical: tuple[str, ...] = (),
    row_id: str | None = None,
    measures: Mapping[str, MeasureColumn] | None = None,
    copy: bool | None = None,
    source: ColumnSource | None = None,
) -> Observations:
    """Prepare selected complete numeric columns through Arrow/DLPack.

    Numeric roles are explicit. Float64/int64 imports require caller-enabled
    JAX x64 rather than silently truncating. ``copy`` applies to the DLPack
    import only, independently of chunk consolidation and fitting ownership.

    Args:
        table: An Arrow table with aligned rows.
        numeric: Names of primitive integer/float numeric columns.
        categorical: Names of integer/string categorical columns.
        row_id: Explicit identity column, or positional identities when absent.
        measures: Empirical-measure columns carrying their identities; they
            must equal the observation identities exactly.
        copy: DLPack import copy policy.
        source: Unconverted fit-time columns, such as :class:`ArrowColumns`,
            that later inference may resolve by name.

    Returns:
        Validated observations preserving the table's selected row order.

    Raises:
        TypeError: If a numeric or label column has an unsupported type.
        ValueError: If columns overlap, contain nulls or require disabled x64.

    """
    selected = (*numeric, *categorical)
    if len(set(selected)) != len(selected):
        message = "Numeric and categorical roles must use distinct columns."
        raise ValueError(message)
    columns = {
        name: _dlpack_column(table, name, "Numeric", copy=copy) for name in numeric
    }
    identities = (
        tuple(range(table.num_rows)) if row_id is None else _labels(table, row_id)
    )
    return Observations(
        columns,
        row_ids=identities,
        labels={name: _labels(table, name) for name in categorical},
        measures=measures,
        source=source,
    )


def from_polars(
    frame: ArrowFrame,
    *,
    numeric: tuple[str, ...],
    categorical: tuple[str, ...] = (),
    row_id: str | None = None,
    measures: Mapping[str, MeasureColumn] | None = None,
    copy: bool | None = None,
) -> Observations:
    """Export Polars through Arrow using the shared explicit-role adapter.

    Args:
        frame: A Polars dataframe whose row order is retained.
        numeric: Names of primitive numeric columns.
        categorical: Names of integer/string categorical columns.
        row_id: Identity column, or positional identities when absent.
        measures: Empirical-measure columns carrying their identities; they
            must equal the observation identities exactly.
        copy: DLPack import copy policy.

    Returns:
        Validated observations sharing the array workflow contract.

    """
    return from_arrow(
        frame.to_arrow(),
        numeric=numeric,
        categorical=categorical,
        row_id=row_id,
        measures=measures,
        copy=copy,
    )


def measures_from_arrow(
    table: pa.Table,
    *,
    row_id: str,
    values: tuple[str, ...],
    mass: str | None = None,
    normalize: bool = False,
    order: Sequence[RowId] | None = None,
    copy: bool | None = None,
) -> MeasureColumn:
    """Build empirical measures from a long table with one row per atom.

    Rows sharing a ``row_id`` value are the atoms of one observation's measure,
    kept in table order; ``values`` name the support coordinates. Identity order
    is deterministic: ``order`` when supplied (it must list exactly the ids in
    the table), otherwise ascending sorted ids (integers or strings, not mixed).
    Masses follow :meth:`PackedClouds.from_clouds`: omitted ``mass`` means
    uniform atoms; supplied masses must sum to one per observation unless
    ``normalize=True`` is requested.

    Args:
        table: Long-format Arrow table, one row per atom.
        row_id: Observation identity column (integer/string, no nulls).
        values: One or more primitive numeric support-coordinate columns.
        mass: Optional atom-mass column.
        normalize: Explicitly normalize masses within each observation.
        order: Explicit identity order, otherwise sorted.
        copy: DLPack import copy policy for value and mass columns.

    Returns:
        The measures with their ordered identities, for ``from_arrow(measures=)``.

    Raises:
        TypeError: If a column has an unsupported type or ids cannot be sorted.
        ValueError: If columns are missing/null, ``values`` is empty or ``order``
            differs from the table's identities.

    """
    if not values:
        message = "Long-format measures require at least one value column."
        raise ValueError(message)
    roles = (row_id, *values, *(() if mass is None else (mass,)))
    if len(set(roles)) != len(roles):
        message = "Identity, value and mass roles must use distinct columns."
        raise ValueError(message)
    ids = _labels(table, row_id)
    support = jnp.column_stack(
        [_dlpack_column(table, name, "Value", copy=copy) for name in values]
    )
    weights = None if mass is None else _dlpack_column(table, mass, "Mass", copy=copy)
    positions: dict[RowId, list[int]] = {}
    for position, identity in enumerate(ids):
        positions.setdefault(identity, []).append(position)
    if order is None:
        integers = sorted(key for key in positions if isinstance(key, int))
        strings = sorted(key for key in positions if isinstance(key, str))
        if integers and strings:
            message = "Mixed integer/string identities require an explicit order."
            raise TypeError(message)
        ordered: tuple[RowId, ...] = (*integers, *strings)
    else:
        ordered = tuple(order)
        if len(ordered) != len(set(ordered)) or set(ordered) != set(positions):
            message = "Explicit order must list exactly the table's identities once."
            raise ValueError(message)
    rows = [jnp.asarray(positions[identity]) for identity in ordered]
    return MeasureColumn(
        ordered,
        PackedClouds.from_clouds(
            [support[index] for index in rows],
            None if weights is None else [weights[index] for index in rows],
            normalize=normalize,
        ),
    )


def measures_from_polars(
    frame: ArrowFrame,
    *,
    row_id: str,
    values: tuple[str, ...],
    mass: str | None = None,
    normalize: bool = False,
    order: Sequence[RowId] | None = None,
) -> MeasureColumn:
    """Build identity-carrying empirical measures from a long Polars frame.

    See :func:`measures_from_arrow` for the atom, mass and identity-order rules.
    """
    return measures_from_arrow(
        frame.to_arrow(),
        row_id=row_id,
        values=values,
        mass=mass,
        normalize=normalize,
        order=order,
    )
