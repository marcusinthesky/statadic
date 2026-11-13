"""Hypothesis strategy for the hedge-inference suites: a target and two candidates.

A plain module rather than a ``conftest.py``, following ``_moments.py``: it is a
strategy, not lifecycle. It replaces the data half of the old ``hedge_data``
fixture for properties that hold for every valid draw (flags, orderings,
validation). Power assertions such as ``test_far_mixture_rejects`` need one fixed
draw and still read the ``hedge_data`` fixture.

Draws are bounded to ``+-HALF_WIDTH`` around the sample centres 3, 3 and 9, so the
far candidate is always further than the width of the close one — the same
"far from the tie case" guarantee the fixture's six-standard-deviation gap gave.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from hypothesis import strategies as st
from jax import random

from .._strategies import float_arrays

if TYPE_CHECKING:
    from hypothesis.strategies import DrawFn
    from jax import Array

HedgeData = tuple["Array", list["Array"]]

HALF_WIDTH = 1.5
_TARGET_ROWS, _CLOSE_ROWS, _FAR_ROWS, _DIM = 60, 30, 50, 4


def original_hedge_data() -> HedgeData:
    """Return the seeded draw the former fixture used, as a regression witness.

    Returns:
        ``(target, [close, far])`` with shapes ``(60, 4)``, ``(30, 4)``, ``(50, 4)``.

    """
    x = random.normal(random.PRNGKey(0), (_TARGET_ROWS, _DIM)) + 3.0
    close = random.normal(random.PRNGKey(1), (_CLOSE_ROWS, _DIM)) + 3.0
    far = random.normal(random.PRNGKey(2), (_FAR_ROWS, _DIM)) + 9.0
    return x, [close, far]


@st.composite
def hedge_samples(draw: DrawFn) -> HedgeData:
    """Draw a target with a close candidate and a far candidate.

    Shapes are fixed so JAX traces once; only the values vary.

    Args:
        draw: Hypothesis draw function.

    Returns:
        ``(target, [close, far])`` shaped as :func:`original_hedge_data`.

    """
    x = draw(float_arrays((_TARGET_ROWS, _DIM), bound=HALF_WIDTH)) + 3.0
    close = draw(float_arrays((_CLOSE_ROWS, _DIM), bound=HALF_WIDTH)) + 3.0
    far = draw(float_arrays((_FAR_ROWS, _DIM), bound=HALF_WIDTH)) + 9.0
    return x, [jnp.asarray(close), jnp.asarray(far)]
