# standalone package-maintenance entry point
"""Typed Array API strategies shared by property tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from hypothesis.extra.array_api import make_strategies_namespace

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy
    from jax import Array

xps = make_strategies_namespace(jnp)


def numpy_float_arrays(
    shape: int
    | tuple[int, ...]
    | SearchStrategy[int]
    | SearchStrategy[tuple[int, ...]],
    *,
    lower: float = -10.0,
    upper: float = 10.0,
) -> SearchStrategy[np.ndarray]:
    """Draw bounded NumPy inputs at the active JAX floating precision."""
    dtype = np.dtype(jnp.result_type(float))
    return hnp.arrays(
        dtype,
        shape,
        elements=st.floats(
            lower,
            upper,
            width=64 if jax.config.x64_enabled else 32,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


def distinct_points(n: int) -> SearchStrategy[np.ndarray]:
    """Draw dyadic real-line points with gaps of at least one quarter."""
    return st.lists(st.integers(-32, 32), min_size=n, max_size=n, unique=True).map(
        lambda values: np.asarray(values, dtype=np.dtype(jnp.result_type(float))) / 4
    )


def float_arrays(
    shape: int | tuple[int, ...] | SearchStrategy[tuple[int, ...]],
    *,
    bound: float = 10.0,
) -> SearchStrategy[Array]:
    """Draw finite arrays of JAX's default float dtype, within ``±bound``.

    Keep ``shape`` fixed unless shape is the property under test, because each new
    shape retraces and recompiles JAX code.

    Args:
        shape: Array shape, or a strategy for it.
        bound: Largest allowed magnitude.

    Returns:
        A strategy for JAX arrays.

    """
    return xps.arrays(
        jnp.result_type(float),
        shape,
        elements={
            "min_value": -bound,
            "max_value": bound,
            "allow_nan": False,
            "allow_infinity": False,
        },
    )


def prng_keys() -> SearchStrategy[Array]:
    """Draw JAX PRNG keys from the full 32-bit seed range.

    Returns:
        A strategy for legacy uint32 keys, as ``jax.random.PRNGKey`` makes them.

    """
    return st.integers(0, 2**32 - 1).map(jax.random.PRNGKey)


_ROW_FLOOR = 0.5


def nonzero_rows(rows: int, columns: int) -> SearchStrategy[Array]:
    """Draw finite matrices whose rows are nonzero by construction.

    Directional functions such as cosine and angular distances document nonzero
    rows as their domain. The first coordinate of every row is pushed out to
    magnitude at least one half, so no row can be zero and nothing is filtered.

    Args:
        rows: Number of rows.
        columns: Number of columns.

    Returns:
        A strategy for JAX matrices of shape ``(rows, columns)``.

    """

    def keep_rows_nonzero(matrix: Array) -> Array:
        head = matrix[:, 0]
        return matrix.at[:, 0].set(jnp.where(jnp.abs(head) < _ROW_FLOOR, 1.0, head))

    return float_arrays((rows, columns)).map(keep_rows_nonzero)
