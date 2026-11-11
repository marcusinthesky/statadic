# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""Traceable permutation-result assembly from a statistic and its null draws.

``permutation_test_kernel`` compares an observed scalar with a supplied null
array using an explicit alternative and Monte-Carlo add-one calibration. It
returns a numerical ``TestResult`` without host concretization. It is also
exported from ``statadic.inference.resampling.permutation``.

The caller owns how the null draws were generated and the assumptions licensing
them. Generic and energy-specific data-to-test procedures live in the sibling
``test`` module.
"""

from __future__ import annotations

from typing import Literal  # runtime type alias

import jax.numpy as jnp
from jaxtyping import Float  # noqa: TC002

from statadic.core.results import (
    TestResult,
)
from statadic.core.typing import Array, Static, static_jaxtyping  # noqa: TC001

__all__ = ["Alternative", "permutation_test_kernel"]

type Alternative = Literal["two-sided", "greater", "less"]

_ALTERNATIVES: tuple[Alternative, ...] = ("two-sided", "greater", "less")


def _alternative_message(alternative: str) -> str:
    """Build the shared bad-``alternative`` error message.

    Returns:
        The ValueError message naming the three accepted alternatives.

    """
    return f"alternative must be 'two-sided', 'greater', or 'less', got '{alternative}'"


@static_jaxtyping("resamples")
def permutation_test_kernel(
    statistic: Float[Array, ""],  # jaxtyping scalar shape
    null_distribution: Float[Array, " resamples"],  # jaxtyping
    num_permutations: Static[int],
    alternative: Static[Alternative] = "two-sided",
) -> TestResult[Float[Array, ""]]:  # jaxtyping scalar shape
    """Assemble a :class:`~statadic.core.results.TestResult` from a statistic and its null.

    The traceable half of the permutation machinery: every operation is a JAX op
    on ``statistic``/``null_distribution``, and ``num_permutations``/
    ``alternative`` are Python-level (``n_resamples`` is a *static* pytree field
    and the alternative selects a branch at trace time). Nothing is concretized,
    so this composes under ``jit``/``vmap``:

        >>> grid = jax.vmap(lambda s, nd: permutation_test_kernel(s, nd, 999))
        >>> batched = grid(stats, nulls)  # stats (cells,) nulls (cells, 999)
        >>> [leaf.shape for leaf in jax.tree_util.tree_leaves(batched)]
        [(cells,), (cells,), (cells, 999)]

    One ``TestResult`` comes back, not ``cells`` of them — that is the payoff of
    pytree-registering the result types, and it is why
    ``TestResult.null_distribution`` carries a ``*batch`` shape prefix.

    Batching is via a transform, never by hand: passing an already-batched
    ``null_distribution`` would make the extreme-count reduction include cells
    as well as resamples.

    Args:
        statistic: Observed test statistic (scalar under a single run).
        null_distribution: Statistics under the null, shape ``(resamples,)``.
        num_permutations: Number of resamples; recorded as ``n_resamples``.
        alternative: ``"two-sided"``, ``"greater"`` or ``"less"``.

    Returns:
        ``TestResult`` carrying the statistic, the finite-sample add-one
        p-value ``(1 + b) / (1 + B)`` (where ``b`` is the number of null
        draws as or more extreme than the observation and ``B`` is
        ``num_permutations``), and the null draws.  The smallest attainable
        p-value is therefore ``1 / (B + 1)``, never zero. The p-value is
        NaN if the observed statistic or any null draw is non-finite.

    Raises:
        ValueError: If ``alternative`` is not recognized or the declared
            resample count does not match the null-distribution shape.

    """
    if null_distribution.shape[0] != num_permutations:
        message = (
            "null_distribution length "
            f"({null_distribution.shape[0]}) must equal num_permutations "
            f"({num_permutations})"
        )
        raise ValueError(message)

    if alternative == "two-sided":
        as_or_more_extreme = jnp.abs(null_distribution) >= jnp.abs(statistic)
    elif alternative == "greater":
        as_or_more_extreme = null_distribution >= statistic
    elif alternative == "less":
        as_or_more_extreme = null_distribution <= statistic
    else:
        raise ValueError(_alternative_message(alternative))

    # Monte-Carlo permutation tests must include the observed arrangement in
    # the reference set.  With B randomly sampled permutations this is the
    # finite-sample-valid add-one estimate (1 + b) / (1 + B), not b / B; the
    # latter can report an impossible p-value of zero.  Keep this entirely in
    # JAX so the kernel remains closed under jit/vmap.
    pvalue = (1 + jnp.sum(as_or_more_extreme)) / (1 + num_permutations)
    valid = jnp.isfinite(statistic) & jnp.all(jnp.isfinite(null_distribution))
    pvalue = jnp.where(valid, pvalue, jnp.nan)

    return TestResult(
        statistic=statistic,
        pvalue=pvalue,
        null_distribution=null_distribution,
        n_resamples=num_permutations,
    )
