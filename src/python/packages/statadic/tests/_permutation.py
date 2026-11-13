# jaxtyping shape strings are runtime contracts.
"""Host report of the on-device two-sample permutation result.

``permutation_test_result`` is the numerical door; the eager ``HypothesisTest``
view these suites assert on is its ``to_host_test_result`` adapter on the
seed-zero default key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import random
from statadic.inference.resampling.permutation import permutation_test_result
from statadic.inference.resampling.results import to_host_test_result

if TYPE_CHECKING:
    from statadic.core.typing import ArrayLike
    from statadic.inference.resampling.permutation.kernel import Alternative
    from statadic.inference.resampling.permutation.test import StatisticFn
    from statadic.inference.resampling.results import HypothesisTest


def host_permutation_test(
    x: ArrayLike,
    y: ArrayLike,
    statistic_fn: StatisticFn,
    num_permutations: int = 999,
    seed: int | None = None,
    alternative: Alternative = "two-sided",
) -> HypothesisTest:
    """Return ``to_host_test_result(permutation_test_result(...))`` for ``seed``."""
    return to_host_test_result(
        permutation_test_result(
            jnp.asarray(x),
            jnp.asarray(y),
            statistic_fn,
            random.PRNGKey(seed if seed is not None else 0),
            num_permutations=num_permutations,
            alternative=alternative,
        )
    )
