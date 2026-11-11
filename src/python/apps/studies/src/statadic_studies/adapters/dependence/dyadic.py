"""Scientific adapter functions grouped by computational ownership."""

import jax
import numpy as np
from statadic.estimation.dyadic import DyadicComputationError, DyadicEstimator
from statadic.model.dyadic import DyadicInputError, DyadicModel, DyadicSample

from statadic_studies.adapters._random import rng


def dyadic_data(generator, nodes):
    left, right = np.triu_indices(nodes, k=1)
    covariate = generator.normal(size=nodes)
    focal = np.abs(covariate[left] - covariate[right])
    effects = generator.normal(size=nodes)
    y = (
        0.4
        + 0.6 * focal
        + effects[left]
        + effects[right]
        + 0.3 * generator.normal(size=len(left))
    )
    labels = np.array([f"node{i:02d}" for i in range(nodes)])
    return focal, y, labels[left], labels[right]


def _dyadic(scenario, replicate, seed):
    if not jax.config.x64_enabled:
        return {
            name: {"valid": False, "failure": "x64-required"}
            for name in ("pooled", "node-fixed-effects")
        }
    generator = rng("dyadic.point-estimation", scenario, replicate, seed=seed)
    focal, y, left, right = dyadic_data(generator, int(scenario[1:]))
    records = {}
    for name, effects in (("pooled", False), ("node-fixed-effects", True)):
        try:
            fit = DyadicEstimator().estimate(
                DyadicModel(
                    ["focal"],
                    {"focal": focal},
                    DyadicSample.from_arrays(y, left, right),
                    node_effects=effects,
                )
            )
            estimate = fit.coefficients["focal"]
            valid = "focal" not in fit.dropped_collinear_columns and np.isfinite(
                estimate
            )
            records[name] = (
                {"valid": True, "estimate": float(estimate), "truth": 0.6}
                if valid
                else {"valid": False, "failure": "focal-dropped-or-nonfinite"}
            )
        except (DyadicComputationError, DyadicInputError, ValueError) as exc:
            records[name] = {"valid": False, "failure": type(exc).__name__}
    return records
