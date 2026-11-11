"""Common-data comparison of a named spatial term and an external OLS oracle."""

import jax.numpy as jnp
import numpy as np
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression
from statadic.api.spatial.graphs import SpatialData, SpatialX
from statadic.operators.spatial import prepare_spatial_weights

from statadic_studies.adapters._random import rng
from statadic_studies.definitions.dependence.composition import SLX


def slx_trial(scenario, replicate, seed):
    """Simulate a spatial exposure and retain paired estimates of its coefficient."""
    theta = {"zero": 0.0, "spillover": 0.5}[scenario]
    gen = rng(SLX.id, scenario, replicate, seed=seed)
    n = 64
    w = 0.7 * np.roll(np.eye(n), 1, axis=1) + 0.3 * np.roll(np.eye(n), -2, axis=1)
    x = gen.normal(size=n)
    design = np.column_stack((np.ones(n), x, w @ x))
    y = design @ [0.2, 0.4, theta] + 0.6 * gen.normal(size=n)
    ids = np.arange(n, dtype=np.int32)
    weights = prepare_spatial_weights(w, unit_ids=ids)
    data = SpatialData(
        Observations({"unit": ids, "x": jnp.asarray(x), "y": jnp.asarray(y)}),
        {"neighbors": weights},
        unit="unit",
    )
    model = Regression(
        "y", Intercept() + Linear("x") + SpatialX("x", graph="neighbors")
    )
    fit = model.estimate(data)
    explicit, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    return {
        "named-SLX": {
            "valid": True,
            "estimate": float(
                fit.parameters[fit.parameter_names.index("W[neighbors]:x")]
            ),
            "truth": theta,
        }
        if fit.valid
        else {"valid": False, "failure": "fit-unavailable"},
        "explicit-NumPy": {
            "valid": True,
            "estimate": float(explicit[2]),
            "truth": theta,
        }
        if rank == 3
        else {"valid": False, "failure": "rank-deficient"},
    }
