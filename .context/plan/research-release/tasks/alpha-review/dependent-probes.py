"""Bounded survival and current graph-import witnesses."""

import importlib
import json

import jax
import jax.numpy as jnp

from statadic.model.survival import LogLogisticAFT, restricted_mean, survival_probability


def run():
    imports = {}
    for name in ("statadic.api.graph_selection", "statadic.api.graph_resampling"):
        try:
            importlib.import_module(name)
            imports[name] = "imported"
        except ImportError as error:
            imports[name] = str(error)
    quadrature = []
    for dtype in (jnp.float32, jnp.float64):
        design = jnp.ones((1, 1), dtype=dtype)
        theta = jnp.array([0.0, -20.0], dtype=dtype)
        horizon = jnp.array(1.0, dtype=dtype)
        family = LogLogisticAFT()

        def calculate(parameter):
            return restricted_mean(parameter, design, horizon, family)

        lower_bound = horizon * survival_probability(theta, design, horizon, family)
        quadrature.append(
            {
                "dtype": str(jnp.dtype(dtype)),
                "eager": calculate(theta).tolist(),
                "jit": jax.jit(calculate)(theta).tolist(),
                "vmap": jax.jit(jax.vmap(calculate))(
                    jnp.stack([theta, theta])
                ).tolist(),
                "monotonic_survival_lower_bound": lower_bound.tolist(),
            }
        )
    return {
        "graph_imports": imports,
        "restricted_mean": quadrature,
        "limits": "Bounded finite-horizon survival and actual current imports; no qualification campaign.",
    }


if __name__ == "__main__":
    with jax.enable_x64(new_val=True):
        print(json.dumps(run(), indent=2))
