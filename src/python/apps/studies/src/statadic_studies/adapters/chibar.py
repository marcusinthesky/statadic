"""Native population operating characteristics with independent targets."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from statadic.inference.wolak import chi_bar_squared_weights_mc_kernel

from statadic_studies.adapters._outcomes import _record
from statadic_studies.adapters._random import rng


def _chibar(scenario, replicate, seed, draws):
    study = "inequalities.chibar-weight-accuracy"
    generator = rng(study, scenario, replicate, seed=seed, stream="calibration")
    key = jax.random.PRNGKey(int(generator.integers(0, 2**32, dtype=np.uint32)))
    result = jax.device_get(
        chi_bar_squared_weights_mc_kernel(key, jnp.eye(3), n_mc=draws)
    )
    weights = np.asarray(result.weights)
    if (
        not bool(result.converged)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1)
    ):
        return {
            f"weight-{index}": {
                "valid": False,
                "failure": "projection-or-weight-unavailable",
                "inner_draws": draws,
            }
            for index in range(4)
        }
    return {
        f"weight-{index}": _record(value, truth, draws)
        for index, (value, truth) in enumerate(
            zip(weights, (0.125, 0.375, 0.375, 0.125), strict=True)
        )
    }
