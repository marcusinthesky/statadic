# jaxtyping shape strings are runtime contracts.
"""Sample Sharpe estimation and reusable sample-moment reductions.

``sharpe_ratio`` returns the eager per-observation mean divided by sample
standard deviation (ddof=1), with zero for nonpositive dispersion.
``sharpe_ratio_axis`` is the compiled axis reduction used by bootstrap callers;
``sample_skew_kurtosis`` returns unadjusted moment skewness and non-excess
kurtosis. Sharpe uncertainty and performance benchmarks belong to
:mod:`statadic.inference.sharpe`.

Precision follows the supplied arrays and caller configuration.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from statadic.core.typing import (  # runtime  # noqa: TC001
    Array,
    ArrayLike,
    Static,
)


def sharpe_ratio(returns: ArrayLike) -> float:
    """Per-observation sample Sharpe ratio ``mean / std`` (ddof=1).

    Args:
        returns: 1-D return series.

    Returns:
        Sharpe ratio (0.0 if the series has non-positive dispersion).

    """
    r = jnp.asarray(returns).ravel()
    sd = jnp.std(r, ddof=1) if r.size > 1 else jnp.asarray(0.0)
    varying = jnp.ptp(r) > 0.0 if r.size else jnp.zeros((), dtype=bool)
    value = jnp.where(varying & (sd > 0.0), jnp.mean(r) / sd, 0.0)
    return float(value)


@partial(jax.jit, static_argnames=("axis",))
def sharpe_ratio_axis(x: ArrayLike, axis: Static[int]) -> Array:
    """Vectorised Sharpe ratio along ``axis`` (0.0 where dispersion vanishes).

    Args:
        x: 2-D array of return series.
        axis: Time axis to reduce over.

    Returns:
        Sharpe ratios with ``axis`` removed.

    """
    values = jnp.asarray(x)
    mean = jnp.mean(values, axis=axis)
    sd = jnp.std(values, axis=axis, ddof=1)
    varying = (
        jnp.ptp(values, axis=axis) > 0.0
        if values.shape[axis]
        else jnp.zeros_like(mean, dtype=bool)
    )
    return jnp.where(varying & (sd > 0.0), mean / sd, 0.0)


def sample_skew_kurtosis(returns: ArrayLike) -> tuple[float, float]:
    """Sample skewness and NON-excess kurtosis (normal -> kurtosis 3).

    Both moments are undefined (NaN) for an exactly constant series.

    Args:
        returns: 1-D return series.

    Returns:
        ``(skew, kurtosis)`` with kurtosis on the non-excess scale.

    """
    r = jnp.asarray(returns).ravel()
    centered = r - jnp.mean(r)
    second = jnp.mean(centered**2)
    varying = jnp.ptp(r) > 0.0 if r.size else jnp.zeros((), dtype=bool)
    skew = jnp.where(varying, jnp.mean(centered**3) / second**1.5, jnp.nan)
    kurt = jnp.where(varying, jnp.mean(centered**4) / second**2, jnp.nan)
    return float(skew), float(kurt)


__all__ = ["sample_skew_kurtosis", "sharpe_ratio", "sharpe_ratio_axis"]
