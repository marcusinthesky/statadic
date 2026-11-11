"""Scientific adapter functions grouped by computational ownership."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from statadic.estimation.sharpe import sharpe_ratio

from statadic_studies.adapters._random import rng


def _sharpe(scenario, replicate, seed):
    generator = rng("sharpe.point-estimation", scenario, replicate, seed=seed)
    values = generator.normal(size=128)
    if scenario == "skewed":
        sigma = 0.6
        values = (np.exp(sigma * values) - np.exp(sigma**2 / 2)) / np.sqrt(
            np.expm1(sigma**2) * np.exp(sigma**2)
        )
    elif scenario == "dependent":
        for i in range(1, len(values)):
            values[i] = 0.5 * values[i - 1] + np.sqrt(0.75) * values[i]
    value = float(sharpe_ratio(jnp.asarray(values + 0.2)))
    return {
        "sample-sharpe": {"valid": True, "estimate": value, "truth": 0.2}
        if np.isfinite(value)
        else {"valid": False, "failure": "nonfinite"}
    }
