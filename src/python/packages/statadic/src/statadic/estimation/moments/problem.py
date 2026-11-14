"""Numeric moment problems shared by GMM and GEL estimators."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Float  # noqa: TC002

from statadic.core.typing import Array, ArrayLike, static_jaxtyping  # noqa: TC001

__all__ = ["AffineMomentProblem", "FixedMomentProblem"]

_MATRIX_NDIM = 2
_MIN_MOMENT_OBSERVATIONS = 2


@static_jaxtyping("t m")
def _validate_moment_panel(
    g: ArrayLike,
    *,
    name: str = "g",
) -> tuple[Float[Array, "t m"], int, int]:  # noqa: F722
    """Check eager finite moment rows without changing the caller's precision.

    GEL and eager inference share this numerical input contract. GMM's
    traceable preparation separately retains nonfinite lanes as unavailable;
    it must not synchronize values through this eager boundary.
    """
    moments: Float[Array, "t m"] = jnp.asarray(g)  # noqa: F722
    if moments.ndim != _MATRIX_NDIM:
        message = f"{name} must have shape (n_observations, n_moments)."
        raise ValueError(message)
    n_observations, n_moments = moments.shape
    if n_observations < _MIN_MOMENT_OBSERVATIONS or n_moments < 1:
        message = f"{name} requires at least two observations and one moment."
        raise ValueError(message)
    if not bool(jnp.all(jnp.isfinite(moments))):
        message = f"{name} must contain only finite values."
        raise ValueError(message)
    return moments, int(n_observations), int(n_moments)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FixedMomentProblem:
    """Complete supplied moment rows, in their supplied order."""

    moments: ArrayLike


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AffineMomentProblem:
    """Rows and derivative for ``g_i(theta) = g0_i + D_i @ theta``.

    An optional weight belongs to the numeric problem and is consumed only by
    fixed-weight GMM. It remains a dynamic JAX input.
    """

    moments0: ArrayLike
    jacobian: ArrayLike
    weight: ArrayLike | None = None
