"""Scientific adapter functions grouped by computational ownership."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.estimation.spatial import ML, SAR, SLXLeastSquares, SpatialProblem
from statadic.inference.linear_covariance import classical_covariance, hc_covariance
from statadic.inference.spatial import spatial_covariance
from statadic.operators.spatial import prepare_spatial_weights

from statadic_studies.adapters._random import rng
from statadic_studies.adapters.dependence._common import _inference, _record


def spatial_design(n=48):
    """Directed row-stochastic graph; row i points to i+1 and i-2."""
    w = 0.7 * np.roll(np.eye(n), 1, axis=1) + 0.3 * np.roll(np.eye(n), -2, axis=1)
    generator = rng("spatial", "fixed-design", 0, seed=220 + n)
    x = np.column_stack((np.ones(n), generator.normal(size=(n, 2))))
    return x, w


@cache
def _sar_kernel(rho):
    x, w = spatial_design()
    graph = prepare_spatial_weights(w)

    def one(y):
        fit = ML().estimate(
            SpatialProblem(jnp.asarray(x), y, graph, process=SAR(), slx_columns=())
        )
        return _inference(fit, spatial_covariance(fit), 3, rho, reference="normal")

    return jax.jit(one)


def _sar(scenario, replicate, seed):
    rho = 0.0 if scenario == "rho0" else 0.4
    x, w = spatial_design()
    generator = rng("spatial.sar-inference", scenario, replicate, seed=seed)
    y = np.linalg.solve(
        np.eye(len(x)) - rho * w,
        x @ np.array([0.4, -0.5, 0.8]) + 0.5 * generator.normal(size=len(x)),
    )
    return {
        "Gaussian-ML-information": _record(
            jax.device_get(_sar_kernel(rho)(jnp.asarray(y))), rho
        )
    }


@cache
def _slx_kernel():
    x, w = spatial_design()
    graph = prepare_spatial_weights(w)

    def one(y):
        fit = SLXLeastSquares().estimate(
            SpatialProblem(jnp.asarray(x), y, graph, process=None, slx_columns=(1, 2))
        )
        return _inference(fit, classical_covariance(fit), 3, 0.3), _inference(
            fit, hc_covariance(fit, kind="HC3"), 3, 0.3
        )

    return jax.jit(one)


def _slx(scenario, replicate, seed):
    x, w = spatial_design()
    generator = rng("spatial.slx-inference", scenario, replicate, seed=seed)
    design = np.column_stack((x, w @ x[:, 1:]))
    scale = 0.5 if scenario == "homoskedastic" else 0.3 + np.abs(x[:, 1])
    y = design @ np.array([0.4, -0.5, 0.8, 0.3, -0.2]) + scale * generator.normal(
        size=len(x)
    )
    outputs = jax.device_get(_slx_kernel()(jnp.asarray(y)))
    return {
        method: _record(value, 0.3)
        for method, value in zip(("SLX-classical", "SLX-HC3"), outputs, strict=True)
    }
