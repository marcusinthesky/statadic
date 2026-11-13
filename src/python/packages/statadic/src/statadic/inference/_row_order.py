"""Validate a fixed-shape row permutation without changing retained fit state."""

import jax.numpy as jnp

from statadic.core.typing import Array, ArrayLike  # noqa: TC001

__all__: list[str] = []


def _validated_row_order(
    row_order: ArrayLike | None, observations: int
) -> tuple[Array, Array]:
    """Return safe gather indices and numerical permutation validity.

    Shape and dtype are structural contracts. Repeated or out-of-range integer
    entries make a numerical lane unavailable, including under JIT and vmap.
    """
    identity = jnp.arange(observations)
    if row_order is None:
        return identity, jnp.ones((), dtype=bool)
    order = jnp.asarray(row_order)
    if order.shape != (observations,) or not jnp.issubdtype(order.dtype, jnp.integer):
        message = "row_order must be an observation-aligned integer vector."
        raise ValueError(message)
    valid = jnp.all(jnp.sort(order) == identity)
    # Signed/unsigned 64-bit promotion would otherwise produce floating indices.
    # Validity is checked before this structural cast, including out-of-range keys.
    return jnp.where(valid, order.astype(identity.dtype), identity), valid
