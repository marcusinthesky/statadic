"""Out-of-sample size and power after choosing a projection on training dates."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import chdtrc, softmax
from statadic.estimation.spatial.graph_likelihood import (
    estimate_graph_ml,
    graph_profile,
)

from statadic_studies.adapters._random import rng

STUDY_ID = "spatial.graph-selection-power"
_PERIODS = 4


@cache
def _design():
    generator = rng(STUDY_ID, "fixed-design", 0, seed=718)
    n = 20
    articles = generator.normal(size=(n, 1, 2)) + 0.25 * generator.normal(
        size=(n, 4, 2)
    )
    graphs = []
    for dimension in (0, 1):
        # Exact W2 squared for equal-mass empirical laws on a line.
        z = np.sort(articles[:, :, dimension], axis=1)
        cost = np.square(z[:, None, :] - z[None, :, :]).mean(axis=2)
        h = 0.25 * np.median(cost[np.triu_indices(n, k=1)])
        logits = -cost / h
        np.fill_diagonal(logits, -np.inf)
        graphs.append(softmax(logits, axis=1))
    x = generator.normal(size=(2 * _PERIODS, n))
    # A distinct supplied graph for the WX term: row i receives x[i+1].
    design = np.stack((np.ones_like(x), x, np.roll(x, -1, axis=1)), axis=-1)
    return np.stack(graphs), design


@jax.jit
def _estimate(w, design, y):
    periods, n = y.shape
    errors = (jnp.empty((0, n, n), dtype=y.dtype),) * periods
    lag = (w[None],) * periods
    xx, yy = tuple(design), tuple(y)
    fit = estimate_graph_ml(lag, errors, xx, yy, ("intercept", "x", "WX", "rho"))
    null = graph_profile(errors, errors, xx, yy, jnp.empty(0, dtype=y.dtype))[2]
    return fit.log_likelihood, null, fit.valid


def graph_selection_trial(scenario, replicate, seed):
    """Score each decision on fresh periods after fitting a shared training sample."""
    rho = {"rho-zero": 0.0, "rho-quarter": 0.25, "rho-half": 0.5}[scenario]
    graphs, design = _design()
    generator = rng(STUDY_ID, scenario, replicate, seed=seed)
    innovations = generator.normal(scale=0.6, size=design.shape[:2])
    mean = design @ np.array([0.2, 0.5, -0.3])
    y = np.linalg.solve(
        np.eye(graphs.shape[1]) - rho * graphs[0], (mean + innovations).T
    ).T

    def run(graph, window):
        return jax.device_get(_estimate(graph, design[window], y[window]))

    def decision(result):
        ll, null, valid = result
        if not valid or not np.isfinite(null):
            return {"valid": False, "failure": "invalid-likelihood-fit"}
        pvalue = float(chdtrc(1, max(0.0, 2.0 * (ll - null))))
        return {"valid": True, "pvalue": pvalue, "rejected": pvalue < 0.05}

    train = [run(w, slice(0, _PERIODS)) for w in graphs]
    test = slice(_PERIODS, None)
    prespecified = decision(run(graphs[0], test))
    if all(bool(v[2]) for v in train):
        chosen = int(np.argmax([v[0] for v in train]))
        selected = prespecified if chosen == 0 else decision(run(graphs[1], test))
    else:
        selected = {"valid": False, "failure": "invalid-training-candidate"}
    return {"fixed-projection": prespecified, "training-selected": selected}
