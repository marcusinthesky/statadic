"""Fixed-shape parameter-influence covariance arithmetic.

Rows supplied here are already scaled influences on the coefficient estimate,
not raw scores. Results are sampling covariance arrays before a separate
finite-sample correction. Fit ownership and inferential status belong above
this numerical layer.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Float, Int  # noqa: TC002

from statadic.core.typing import Array, static_jaxtyping  # noqa: TC001
from statadic.operators.longrun import newey_west_kernel

__all__ = [
    "cluster_parameter_covariance",
    "coefficient_influence_rows",
    "hac_parameter_covariance",
    "independent_parameter_covariance",
]


@static_jaxtyping("parameters observations")
def coefficient_influence_rows(
    coefficient_map: Float[Array, "parameters observations"],  # noqa: F722
    weighted_residuals: Float[Array, "observations"],  # noqa: F821, UP037
) -> Float[Array, "observations parameters"]:  # noqa: F722
    """Map whitened residuals through the fitted QR coefficient map.

    ``coefficient_map`` is ``(sqrt(w) X)^+``. The result's row ``i`` is
    ``coefficient_map[:, i] * sqrt(w_i) * residual_i``. Complete aligned
    finite arrays are the producer's precondition.
    """
    if coefficient_map.shape[-1] != weighted_residuals.shape[-1]:
        message = "The coefficient map and residual rows must align."
        raise ValueError(message)
    return coefficient_map.T * weighted_residuals[:, None]


@static_jaxtyping("observations parameters")
def independent_parameter_covariance(
    influence_rows: Float[Array, "observations parameters"],  # noqa: F722
    leverage: Float[Array, "observations"],  # noqa: F821, UP037
    *,
    kind: Literal["HC0", "HC2", "HC3"] = "HC0",
) -> Float[Array, "parameters parameters"]:  # noqa: F722
    """Sum row-influence outer products, optionally dividing by ``1-h_i``.

    HC2 divides each outer product by ``1-h_i``; HC3 divides it by its square.
    For HC2/HC3, unit or invalid leverage yields NaNs. HC1 is the caller's count
    correction applied to HC0; it does not change row arithmetic here.
    """
    if kind not in {"HC0", "HC2", "HC3"}:
        message = "kind must be HC0, HC2 or HC3."
        raise ValueError(message)
    if leverage.shape != influence_rows.shape[:1]:
        message = "Leverage must align with influence rows."
        raise ValueError(message)
    if kind == "HC0":
        adjusted = influence_rows
    else:
        remaining = 1.0 - leverage
        divisor = jnp.sqrt(remaining) if kind == "HC2" else remaining
        valid_leverage = jnp.isfinite(leverage) & (leverage >= 0.0) & (remaining > 0.0)
        adjusted = influence_rows / jnp.where(valid_leverage, divisor, jnp.nan)[:, None]
    return adjusted.T @ adjusted


@static_jaxtyping("observations parameters")
def hac_parameter_covariance(
    influence_rows: Float[Array, "observations parameters"],  # noqa: F722
    bandwidth: int,
) -> Float[Array, "parameters parameters"]:  # noqa: F722
    """Sum Bartlett-weighted influence cross-products without mean centering.

    ``bandwidth`` is static under JIT and must satisfy ``0 <= b < n``. The
    existing HAC kernel divides by ``n``; multiplying by ``n`` returns the
    sampling covariance of these already-scaled coefficient influences.
    """
    if isinstance(bandwidth, bool) or not isinstance(bandwidth, int):
        message = "bandwidth must be an integer static lag count."
        raise TypeError(message)
    if not 0 <= bandwidth < influence_rows.shape[0]:
        message = "bandwidth must satisfy 0 <= bandwidth < observation count."
        raise ValueError(message)
    return influence_rows.shape[0] * newey_west_kernel(influence_rows, bandwidth)


@static_jaxtyping("observations parameters clusters")
def cluster_parameter_covariance(
    influence_rows: Float[Array, "observations parameters"],  # noqa: F722
    groups: Int[Array, "observations"],  # noqa: F821, UP037
    *,
    n_clusters: int,
) -> Float[Array, "parameters parameters"]:  # noqa: F722
    """Sum outer products of one-way cluster-summed coefficient influences.

    Labels must be contiguous integers in ``[0, n_clusters)`` and every group
    must be occupied. ``n_clusters`` is a positive static shape. The fit-aware
    caller checks those value conditions and the number of occupied groups;
    this primitive supplies only the fixed-shape array reduction.
    """
    if (
        isinstance(n_clusters, bool)
        or not isinstance(n_clusters, int)
        or n_clusters < 1
    ):
        message = "n_clusters must be a positive static integer."
        raise ValueError(message)
    if groups.shape != influence_rows.shape[:1] or not jnp.issubdtype(
        groups.dtype, jnp.integer
    ):
        message = "Integer group labels must align with influence rows."
        raise ValueError(message)
    sums = jnp.zeros((n_clusters, influence_rows.shape[1]), dtype=influence_rows.dtype)
    summed: Float[Array, "clusters parameters"] = sums.at[groups].add(  # noqa: F722
        influence_rows
    )
    return summed.T @ summed
