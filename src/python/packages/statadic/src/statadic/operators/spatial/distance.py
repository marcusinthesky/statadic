"""Differentiable spatial weights from nonnegative pairwise costs."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from statadic.core.typing import Array  # noqa: TC001

__all__ = ["rbf_weights"]


def rbf_weights(cost: Array, bandwidth: Array) -> Array:
    """Return zero-diagonal row-softmax weights for cost/h, with at least two nodes.

    The input may carry leading period or draw axes. Subtracting each row's
    off-diagonal minimum before division preserves a neighbor at tiny h.
    Identity, finiteness and positivity are established by the host recipe.
    """
    diagonal = jnp.eye(cost.shape[-1], dtype=bool)
    masked = jnp.where(diagonal, jnp.inf, cost)
    centered = jnp.where(diagonal, 0, cost - jnp.min(masked, axis=-1, keepdims=True))
    return jax.nn.softmax(jnp.where(diagonal, -jnp.inf, -centered / bandwidth), axis=-1)
