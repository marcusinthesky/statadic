# jaxtyping dimensions are runtime contracts.
"""Label-permutation null of a pure DISCO decomposition."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from statadic.association.dispersion import disco_decomposition
from statadic.core.random import cell_key, permutation_batch
from statadic.core.typing import ArrayLike  # noqa: TC001

__all__ = [
    "disco_permutation_key",
    "disco_permutation_null",
]


@jax.jit(static_argnames=("n_groups", "n_permutations"))
def _disco_permutation_null_kernel(
    dist: jax.Array,
    group_codes: jax.Array,
    key: jax.Array,
    n_groups: int,
    n_permutations: int,
) -> jax.Array:
    """Reuse one pooled matrix while permuting labels only."""
    permuted_codes = permutation_batch(key, group_codes, n_permutations)
    _, _, _, _, f_ratio = jax.vmap(disco_decomposition, in_axes=(None, 0, None))(
        dist, permuted_codes, n_groups
    )
    return f_ratio


def disco_permutation_key(random_state: int) -> jax.Array:
    """Return the key owning the label-permutation stream for ``random_state``.

    Consumers that need the index schedule behind :func:`disco_permutation_null`
    draw ``permutation_batch(key, arange(n), m)``; permuting codes directly is
    the same stream.
    """
    return cell_key(random_state, "association", "disco_permutation_null")


def disco_permutation_null(
    dist: ArrayLike,
    group_codes: ArrayLike,
    n_groups: int,
    n_permutations: int,
    random_state: int,
) -> jax.Array:
    """Draw the label-permutation null while holding the matrix fixed."""
    if isinstance(n_permutations, bool) or n_permutations <= 0:
        message = "n_permutations must be a positive integer"
        raise ValueError(message)
    key = disco_permutation_key(random_state)
    return _disco_permutation_null_kernel(
        jnp.asarray(dist),
        jnp.asarray(group_codes),
        key,
        n_groups,
        n_permutations,
    )
