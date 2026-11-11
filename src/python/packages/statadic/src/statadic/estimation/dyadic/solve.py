# jaxtyping shape strings are runtime contracts.
"""Fixed-shape JAX solve behind dyadic host-side rank selection."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from statadic.core.typing import (
    ArrayLike,  # noqa: TC001
)
from statadic.estimation.dyadic.contracts import DyadicComputationError
from statadic.optimize.backends import least_squares

__all__: list[str] = []


def _least_squares_host(
    design: ArrayLike,
    outcome: ArrayLike,
    *,
    context: str,
) -> jax.Array:
    """Solve one rank-selected design through the shared Lineax QR kernel.

    Column/rank selection is deliberately host-only and may change width from
    draw to draw. Once that decision is made, this named eager door delegates
    the sole least-squares implementation to the fixed-shape JAX backend.
    """
    if isinstance(design, jax.core.Tracer) or isinstance(outcome, jax.core.Tracer):
        message = (
            "dyadic rank selection is a host-only boundary; call least_squares "
            "directly only after selecting a fixed-width design"
        )
        raise RuntimeError(  # noqa: TRY004
            message
        )  # tracing is an execution mode
    try:
        return least_squares(
            jnp.asarray(design),
            jnp.asarray(outcome),
        )
    except (RuntimeError, ValueError) as error:
        message = f"dyadic {context} failed for a validated finite design"
        raise DyadicComputationError(message) from error
