"""Scientific adapter functions grouped by computational ownership."""

from __future__ import annotations

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.discrepancy.mmd.scalar import mmd_squared
from statadic.sample.packed import PackedClouds

from statadic_studies.adapters._random import rng


@cache
def _mmd_kernel():
    return jax.jit(
        lambda x, y: (
            mmd_squared(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
                unbiased=True,
            ),
            mmd_squared(
                PackedClouds.from_array(jnp.asarray(x)),
                PackedClouds.from_array(jnp.asarray(y)),
            ),
        )
    )


def finite_mmd_truth(p, q):
    """Enumerate all population Gram entries, independently of STATADIC's reduction."""
    support = (-1.0, 0.0, 2.0)
    xx = sum(
        p[i] * p[j] * a * b
        for i, a in enumerate(support)
        for j, b in enumerate(support)
    )
    yy = sum(
        q[i] * q[j] * a * b
        for i, a in enumerate(support)
        for j, b in enumerate(support)
    )
    xy = sum(
        p[i] * q[j] * a * b
        for i, a in enumerate(support)
        for j, b in enumerate(support)
    )
    return xx + yy - 2 * xy


def _mmd(scenario, replicate, seed):
    generator = rng("mmd.finite-bias", scenario, replicate, seed=seed)
    p = (0.2, 0.5, 0.3)
    q = (0.5, 0.2, 0.3) if scenario.startswith("alternative") else p
    m = 13 if scenario.endswith("unequal") else 8
    x = generator.choice((-1.0, 0.0, 2.0), size=(8, 1), p=p)
    y = generator.choice((-1.0, 0.0, 2.0), size=(m, 1), p=q)
    outputs = jax.device_get(_mmd_kernel()(jnp.asarray(x), jnp.asarray(y)))
    target = finite_mmd_truth(p, q)
    return {
        method: {"valid": True, "estimate": float(value), "truth": target}
        if np.isfinite(value)
        else {"valid": False, "failure": "nonfinite"}
        for method, value in zip(("U", "V"), outputs, strict=True)
    }
