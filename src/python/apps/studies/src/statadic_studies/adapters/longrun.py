"""Native population operating characteristics with independent targets."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from statadic.operators.longrun import hall_centered_hac, newey_west

from statadic_studies.adapters._outcomes import _record
from statadic_studies.adapters._random import rng


def _longrun(scenario, replicate, seed, draws):
    study = "longrun.gaussian-ar1-variability"
    rho_string, n_string = scenario.split("-")
    rho, n = float(rho_string[3:]), int(n_string[1:])
    generator = rng(study, scenario, replicate, seed=seed)
    values = generator.normal(size=n)
    for i in range(1, n):
        values[i] = rho * values[i - 1] + np.sqrt(1 - rho**2) * values[i]
    data, truth = jnp.asarray(values[:, None]), (1 + rho) / (1 - rho)
    return {
        "plain-fixed4": _record(newey_west(data, 4)[0, 0], truth),
        "centered-fixed4": _record(hall_centered_hac(data, 4)[0, 0], truth),
        "plain-auto": _record(newey_west(data)[0, 0], truth),
        "centered-auto": _record(hall_centered_hac(data)[0, 0], truth),
    }
