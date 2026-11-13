"""Native population operating characteristics with independent targets."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from statadic.estimation.sharpe import sample_skew_kurtosis

from statadic_studies.adapters._outcomes import _record
from statadic_studies.adapters._random import rng


def lognormal_moments(sigma):
    """Population skew and non-excess kurtosis from exact lognormal moments."""
    v = np.exp(sigma**2)
    return ((v + 2) * np.sqrt(v - 1), v**4 + 2 * v**3 + 3 * v**2 - 3)


def _moments(scenario, replicate, seed, draws):
    study = "sharpe.moment-risk"
    law, size = scenario.split("-")
    generator = rng(study, scenario, replicate, seed=seed)
    normal = generator.normal(size=int(size[1:]))
    if law == "normal":
        values, truths = normal, (0.0, 3.0)
    else:
        values, truths = np.exp(0.6 * normal), lognormal_moments(0.6)
        values = (values - np.exp(0.6**2 / 2)) / np.sqrt(
            (np.exp(0.6**2) - 1) * np.exp(0.6**2)
        )
    observed = sample_skew_kurtosis(jnp.asarray(values))
    return {
        name: _record(value, truth)
        for name, value, truth in zip(
            ("sample-skewness", "sample-kurtosis"), observed, truths, strict=True
        )
    }
