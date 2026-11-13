"""CPU workload for a one-dimensional projected empirical-Wasserstein SAR profile."""

import jax
import jax.numpy as jnp
import numpy as np
from statadic.estimation.spatial.graph_likelihood import graph_profile
from statadic.operators.spatial.distance import rbf_weights


def prepare_graph_profile(rng, dtype, smoke):
    n, articles, periods = (8, 4, 2) if smoke else (100, 128, 4)
    clouds = rng.normal(size=(n, articles, 2)).astype(dtype)
    x = rng.normal(size=(periods, n, 1)).astype(dtype)
    design = np.concatenate((np.ones_like(x), x), axis=-1)
    y = rng.normal(size=(periods, n)).astype(dtype)
    angle = np.asarray(0.38, dtype=dtype)

    def objective(theta, points, xx, yy):
        direction = jnp.stack((jnp.cos(theta), jnp.sin(theta)))
        z = jnp.sort(points @ direction, axis=1)
        cost = jnp.mean((z[:, None] - z[None, :]) ** 2, axis=-1)
        w = rbf_weights(cost, jnp.asarray(0.17, dtype=yy.dtype))
        empty = jnp.empty((0, n, n), dtype=yy.dtype)
        return graph_profile(
            (w[None],) * periods,
            (empty,) * periods,
            tuple(xx),
            tuple(yy),
            jnp.asarray([0.35], dtype=yy.dtype),
        )[2]

    inputs = (angle, clouds, design, y)
    return {
        "n": n,
        "articles": articles,
        "input_dimension": 2,
        "rank": 1,
        "periods": periods,
        "rho": 0.35,
        "bandwidth": 0.17,
        "geometry": "sorted equal-mass 1D W2 squared; unit angle parameter",
    }, [
        ("profile-value", objective, inputs),
        ("profile-value-angle-gradient", jax.value_and_grad(objective), inputs),
    ]
