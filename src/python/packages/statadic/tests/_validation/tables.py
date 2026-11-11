"""Caller-owned table buffers with deliberate CPU alignment."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st


def aligned_array(values: np.ndarray, *, offset: int = 0) -> np.ndarray:
    """Copy values into storage aligned to 64 bytes, plus an element offset."""
    storage = np.empty(values.size + 64 // values.itemsize + 1, dtype=values.dtype)
    start = (-storage.ctypes.data % 64) // values.itemsize + offset
    result = storage[start : start + values.size].reshape(values.shape)
    result[:] = values
    return result


@dataclass(frozen=True)
class TableCase:
    """Fresh caller-owned columns and a complete two-chunk row partition."""

    columns: dict[str, np.ndarray]
    split: int


@st.composite
def table_cases(draw: st.DrawFn, *, n: int = 8) -> TableCase:
    """Draw identified regressions with writable, deliberately aligned buffers."""
    dtype = draw(
        st.sampled_from(
            [np.float32, np.int32, *([np.float64] if jax.config.x64_enabled else [])]
        )
    )
    offset = draw(st.integers(0, 1))
    slope = draw(st.integers(1, 4))
    shift = draw(st.integers(-4, 4))
    x = np.arange(n, dtype=dtype) + shift
    columns = {
        "x": aligned_array(x, offset=offset),
        "y": aligned_array(
            2 + slope * x + np.arange(n, dtype=dtype) % 2, offset=offset
        ),
        "z": aligned_array(x + 2, offset=offset),
        "group": aligned_array(np.arange(n, dtype=np.int32) // 2, offset=offset),
    }
    return TableCase(columns, draw(st.integers(1, n - 1)))


def arrow_source(case: TableCase, *, sliced: bool = False) -> object:
    """Export chunks and optional leading padding without copying numeric storage."""
    import pyarrow as pa  # noqa: PLC0415 -- optional table test dependency

    table = pa.table(
        {
            name: pa.chunked_array(
                [pa.array(values[: case.split]), pa.array(values[case.split :])]
            )
            for name, values in case.columns.items()
        }
    )
    if sliced:
        table = pa.concat_tables([table.slice(0, 1), table]).slice(1)
    return table


def as_source(case: TableCase, kind: str) -> object:
    """Adapt the same represented values without silently collecting LazyFrames."""
    if kind == "mapping-numpy":
        return case.columns
    if kind == "mapping-jax":
        return {name: jnp.asarray(values) for name, values in case.columns.items()}
    if kind in {"arrow", "arrow-sliced"}:
        return arrow_source(case, sliced=kind == "arrow-sliced")
    import polars as pl  # noqa: PLC0415 -- optional table test dependency

    frame = pl.DataFrame(case.columns)
    if kind == "polars":
        return frame
    if kind == "polars-lazy":
        return frame.lazy()
    message = f"Unknown table source: {kind}"
    raise ValueError(message)


def scribble(case: TableCase) -> None:
    """Overwrite every mutable caller-owned numerical backing buffer."""
    for values in case.columns.values():
        values[:] = 0
