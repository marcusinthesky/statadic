"""Fixtures shared by the inference suites, and no shallower (t49 C4).

``hedge_data`` sits here rather than in ``tests/conftest.py`` because
``test_permutation.py`` and ``test_studentized.py`` are its only consumers, and
a fixture at the tests root is an import cost every ``-k`` invocation pays —
including the ones that never touch inference.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import random

#: Target sample plus the candidate list a mixture test selects over.
HedgeData = tuple[jnp.ndarray, list[jnp.ndarray]]


@pytest.fixture
def hedge_data() -> HedgeData:
    """Target with a close candidate and a far candidate.

    The two candidates are separated by six standard deviations of the draw, so
    every selection assertion built on this fixture is far from the tie case —
    a test that fails here failed because the code changed, not because the
    seeds landed badly.

    Returns:
        ``(target, [close, far])`` with shapes ``(60, 4)``, ``(30, 4)`` and
        ``(50, 4)``.

    """
    x = random.normal(random.PRNGKey(0), (60, 4)) + 3.0
    close = random.normal(random.PRNGKey(1), (30, 4)) + 3.0
    far = random.normal(random.PRNGKey(2), (50, 4)) + 9.0
    return x, [close, far]
