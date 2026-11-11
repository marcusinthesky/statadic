# jaxtyping shape strings are runtime contracts.
"""Shared eager calibration rules for inferential procedures."""

from __future__ import annotations

import jax.numpy as jnp

from statadic.core.typing import (
    ArrayLike,  # noqa: TC001
)

__all__ = ["add_one_p", "add_one_p_kernel"]


def add_one_p_kernel(observed: ArrayLike, null: ArrayLike) -> jnp.ndarray:
    """Right-tail Monte-Carlo p-value with add-one correction, transformable.

    Every reference draw must be finite; no draws are discarded.

    Args:
        observed: Observed statistic.
        null: Reference draws of any shape; flattened before counting.

    Returns:
        The p-value as a JAX scalar, or ``nan`` when the observed statistic is
        non-finite, any reference draw is non-finite, or there are no draws.

    """
    statistic = jnp.asarray(observed)
    draws = jnp.asarray(null).ravel()
    exceed = jnp.count_nonzero(draws >= statistic)
    pvalue = (1 + exceed) / (draws.size + 1)
    defined = jnp.isfinite(statistic) & jnp.all(jnp.isfinite(draws)) & (draws.size > 0)
    return jnp.where(defined, pvalue, jnp.nan)


def add_one_p(observed: float, null: ArrayLike) -> float:
    """Return a right-tail Monte-Carlo p-value with add-one correction.

    A non-finite statistic or draw, or an empty reference set, returns ``nan``.

    This is the host adapter over :func:`add_one_p_kernel`; it exists because
    callers in ``apps/pipeline`` put the result straight into a result record.
    Call the kernel directly from anything that needs to stay transformable.
    """
    return float(add_one_p_kernel(observed, null))
