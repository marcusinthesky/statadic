# jaxtyping shape strings are runtime contracts.
"""Small deterministic fixtures, explicitly loaded by consuming test suites.

Set ``pytest_plugins = ["statadic.testing.fixtures"]`` in the initial conftest,
after installing any runtime annotation hook. Merely depending on STATADIC does
not register fixtures or import JAX into unrelated pytest sessions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    import jax

__all__ = ["prng_key", "seeded_key", "tolerances"]

#: Seed every ``prng_key`` consumer shares. A constant, not a per-test draw:
#: these suites assert numerical identities, and a fixture that varied the seed
#: would turn a tolerance failure into an unreproducible one.
SEED = 42

#: Absolute/relative tolerances for comparing float64 statistics accumulated over
#: O(n^2) terms. 1e-9 absolute would report summation noise as a real difference;
#: see ``tests/_cases.py``, which sets the same pair for the axiom battery.
ATOL = 1e-9
RTOL = 1e-7


@pytest.fixture
def prng_key() -> jax.Array:
    """Return a deterministic PRNG key for reproducible tests.

    Returns:
        ``jax.random.PRNGKey(SEED)``.

    """
    import jax  # see the module docstring: startup cost  # noqa: PLC0415

    return jax.random.PRNGKey(SEED)


@pytest.fixture
def seeded_key() -> Callable[[int], jax.Array]:
    """Return a factory for independent, reproducible PRNG keys.

    Use this where one test needs several unrelated draws — folding a per-draw
    integer into the shared seed keeps them independent without introducing a
    second literal seed that has to be kept in step with :data:`SEED`.

    Returns:
        ``make(index)`` returning a key derived from ``SEED`` and ``index``.

    """
    import jax  # see the module docstring: startup cost  # noqa: PLC0415

    def make(index: int) -> jax.Array:
        return jax.random.fold_in(jax.random.PRNGKey(SEED), index)

    return make


@pytest.fixture
def tolerances() -> tuple[float, float]:
    """Return the shared ``(atol, rtol)`` for float64 statistical comparisons.

    Returns:
        ``(ATOL, RTOL)``.

    """
    return ATOL, RTOL
