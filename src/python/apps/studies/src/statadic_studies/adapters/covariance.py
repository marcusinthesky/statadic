"""Scientific adapter functions grouped by computational ownership."""

from __future__ import annotations

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.estimation.covariance import (
    ledoit_wolf_sample_kernel,
    sample_covariance_kernel,
)

from statadic_studies.adapters._random import rng


@cache
def _covariance_kernel():
    return jax.jit(
        lambda data: (
            sample_covariance_kernel(data, ddof=0),
            sample_covariance_kernel(data, ddof=1),
            ledoit_wolf_sample_kernel(data).sigma,
        )
    )


def _covariance(scenario, replicate, seed):
    generator = rng("covariance.population-risk", scenario, replicate, seed=seed)
    sd = np.array([0.5, 1.0, 1.5, 2.0])
    correlation = 0.3 * np.ones((4, 4)) + 0.7 * np.eye(4)
    if scenario == "misspecified-target":
        correlation = np.array(
            [[1, 0.7, 0, 0], [0.7, 1, 0, 0], [0, 0, 1, -0.5], [0, 0, -0.5, 1]]
        )
    truth = correlation * np.outer(sd, sd)
    data = generator.normal(size=(64, 4)) @ np.linalg.cholesky(truth).T + np.arange(4)
    values = jax.device_get(_covariance_kernel()(jnp.asarray(data)))
    return {
        method: {
            "valid": True,
            "estimate": float(np.trace(value)),
            "truth": float(np.trace(truth)),
            "frobenius_loss": float(np.sum((value - truth) ** 2)),
        }
        if np.isfinite(value).all()
        else {"valid": False, "failure": "nonfinite"}
        for method, value in zip(
            ("sample-ddof0", "sample-ddof1", "ledoit-wolf-constant-correlation"),
            values,
            strict=True,
        )
    }
