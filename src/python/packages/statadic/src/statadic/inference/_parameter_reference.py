# ruff: noqa: F722  # jaxtyping shape strings are runtime contracts.
"""JAX distribution primitives used by the parameter-inference adapters.

The pinned JAX version supplies incomplete beta, normal and chi-square tails,
but no Student-t CDF/quantile or F distribution. The identities here use those
maintained special functions; Student critical values delegate inversion to
Optimistix's maintained scalar bisection solver. No distribution solver or
SciPy runtime dependency is introduced.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optimistix as ox
from jax.scipy.special import betainc
from jaxtyping import Bool, Float  # noqa: TC002

from statadic.core.typing import Array  # noqa: TC001

_MAX_VALIDATED_DEGREES = 10000.0


def _outside_validated_range(degrees_of_freedom: Array) -> Array:
    """Flag degrees or dtypes outside the independently checked reference range."""
    return (degrees_of_freedom > _MAX_VALIDATED_DEGREES) | (
        jnp.finfo(degrees_of_freedom.dtype).bits < 32  # noqa: PLR2004
    )


def _student_two_sided_pvalue(statistic: Array, degrees_of_freedom: Array) -> Array:
    """Return ``P(|T_df| >= |statistic|)`` via the regularized incomplete beta."""
    squared = statistic**2
    upper = betainc(
        degrees_of_freedom / 2.0,
        0.5,
        degrees_of_freedom / (degrees_of_freedom + squared),
    )
    # Near the centre, form the small complementary argument directly. Forming
    # df/(df+t²) first loses the informative digits when df is large. Far into
    # the tail, retain the direct upper-tail identity to avoid cancellation.
    central = 1.0 - betainc(
        0.5, degrees_of_freedom / 2.0, squared / (degrees_of_freedom + squared)
    )
    return jnp.where(squared <= degrees_of_freedom, central, upper)


def _f_survival(statistic: Array, numerator_df: Array, denominator_df: Array) -> Array:
    """Use complementary beta arguments for the central and far upper F tails."""
    scaled_statistic = numerator_df * statistic
    upper = betainc(
        denominator_df / 2.0,
        numerator_df / 2.0,
        denominator_df / (denominator_df + scaled_statistic),
    )
    central = 1.0 - betainc(
        numerator_df / 2.0,
        denominator_df / 2.0,
        scaled_statistic / (denominator_df + scaled_statistic),
    )
    return jnp.where(scaled_statistic <= denominator_df, central, upper)


def _student_critical_scalar(
    confidence: Float[Array, ""],  # jaxtyping scalar
    degrees_of_freedom: Float[Array, ""],
) -> tuple[Float[Array, ""], Bool[Array, ""]]:
    """Invert the normalized upper tail, retaining bisection failure evidence."""
    valid = (
        jnp.isfinite(confidence)
        & (confidence > 0.0)
        & (confidence < 1.0)
        & jnp.isfinite(degrees_of_freedom)
        & (degrees_of_freedom > 0.0)
    )
    safe_confidence = jnp.where(valid, confidence, 0.95)
    safe_df = jnp.where(valid, degrees_of_freedom, 1.0)

    def objective(value: Array, args: tuple[Array, Array]) -> Array:
        df, tail_probability = args
        relative_error = _student_two_sided_pvalue(value, df) / tail_probability - 1.0
        # Incomplete-beta evaluation loses accuracy as its shape grows. Keep
        # parameter-width convergence tight while allowing a dtype-relative
        # probability residual at that scale; independent tail tests cover it.
        return relative_error / jnp.sqrt(jnp.maximum(df, 1.0))

    # Both the parameter interval and normalized tail residual must converge.
    # A zero absolute tolerance would make Optimistix's strict residual check
    # impossible. Scaling the tail equation keeps heavy-tail intervals honest.
    tolerance = float(64.0 * jnp.finfo(confidence.dtype).eps)
    # Optimistix 0.1 declares `norm` as a ClassVar; its runtime constructor does
    # not accept it. The pinned checker incorrectly treats it as an init field.
    solver = ox.Bisection(  # type: ignore[missing-argument]
        rtol=tolerance, atol=tolerance, expand_if_necessary=True
    )
    solution = ox.root_find(
        objective,
        solver,
        jnp.asarray(0.5, dtype=confidence.dtype),
        args=(safe_df, 1.0 - safe_confidence),
        options={
            "lower": jnp.asarray(0.0, dtype=confidence.dtype),
            "upper": jnp.asarray(1.0, dtype=confidence.dtype),
        },
        max_steps=256,
        throw=False,
    )
    converged = (
        valid
        & (solution.result == ox.RESULTS.successful)
        & jnp.isfinite(solution.value)
        & (solution.value >= 0.0)
    )
    return jnp.where(converged, solution.value, jnp.nan), converged


def _student_critical_value(
    confidence: Array, degrees_of_freedom: Array
) -> tuple[Array, Array]:
    """Apply the scalar maintained solve to scalar or broadcast batch arguments."""
    levels, degrees = jnp.broadcast_arrays(confidence, degrees_of_freedom)
    levels = levels + jnp.zeros_like(degrees)
    degrees = degrees + jnp.zeros_like(levels)
    critical, converged = jax.vmap(_student_critical_scalar)(
        levels.reshape(-1), degrees.reshape(-1)
    )
    return critical.reshape(levels.shape), converged.reshape(levels.shape)
