"""Native spatial durbin studies."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.estimation.spatial import (
    ML,
    SAC,
    SAR,
    SEM,
    SpatialProblem,
)
from statadic.inference.spatial import (
    spatial_covariance,
)
from statadic.operators.spatial import (
    prepare_spatial_weights,
)

from statadic_studies.adapters._random import rng
from statadic_studies.adapters.dependence._common import _inference, _record
from statadic_studies.adapters.dependence.spatial_extensions import (
    spatial_fixture,
)

BETA = np.array([0.3, -0.5, 0.8])
THETA = np.array([0.2, -0.1])


def durbin_data(generator, model="SDM", theta=THETA):
    x, w = spatial_fixture()
    rho = 0.25 if model != "SDEM" else 0.0
    lam = 0.3 if model != "SDM" else 0.0
    error = np.linalg.solve(
        np.eye(len(w)) - lam * w, 0.5 * generator.normal(size=len(w))
    )
    mean = x @ BETA + (w @ x[:, 1:]) @ theta
    return x, np.linalg.solve(np.eye(len(w)) - rho * w, mean + error), w


@cache
def _durbin_kernel(model):
    process = {"SDM": SAR, "SDEM": SEM, "GNS": SAC}[model]
    x, w = spatial_fixture()
    graph = prepare_spatial_weights(w)

    @jax.jit
    def one(y):
        fit = ML().estimate(
            SpatialProblem(
                jnp.asarray(x), y, graph, process=process(), slx_columns=(1, 2)
            )
        )
        uncertainty = spatial_covariance(fit)
        target = 0.25 if model == "SDM" else 0.3
        results = tuple(
            _inference(fit, uncertainty, position, truth, reference="normal")
            for position, truth in (
                (1, -0.5),
                (3, 0.2),
                (fit.parameters.shape[0] - 1, target),
            )
        )
        return results, fit.valid, fit.solver.converged, fit.boundary

    return one


def _durbin(scenario, replicate, seed):
    study = "spatial.durbin-likelihood"
    _, y, _ = durbin_data(rng(study, scenario, replicate, seed=seed), scenario)
    results, valid, converged, boundary = jax.device_get(
        _durbin_kernel(scenario)(jnp.asarray(y))
    )
    records = {}
    for name, values, truth in zip(
        ("beta", "theta", "last-spatial"),
        results,
        (-0.5, 0.2, 0.25 if scenario == "SDM" else 0.3),
        strict=True,
    ):
        records[name] = _record(values, truth)
        records[name].update(
            {
                "fit_valid": bool(valid),
                "solver_converged": bool(converged),
                "boundary": bool(boundary),
            }
        )
    return records
