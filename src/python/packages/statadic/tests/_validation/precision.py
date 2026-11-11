"""Discover backend precision capabilities independently of STATADIC's estimators."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def bfloat16_qr_error() -> type[Exception] | None:
    """Return the backend's unsupported-QR exception type, or None if supported."""
    design = jnp.asarray([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=jnp.bfloat16)
    try:
        jax.block_until_ready(jnp.linalg.qr(design))
    except (NotImplementedError, TypeError, ValueError) as error:
        # Only recognize dtype rejection from the direct backend probe.
        if "bfloat16" not in str(error):
            raise
        return type(error)
    return None
