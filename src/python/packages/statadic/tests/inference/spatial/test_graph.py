"""Fixed-graph observed information against the independent OLS special case."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.estimation.spatial.graph_likelihood import estimate_graph_ml
from statadic.inference.spatial.graph import _graph_covariance

if TYPE_CHECKING:
    from statadic.estimation.spatial.graph_likelihood import GraphMLFit
    from statadic.inference.spatial.graph import _GraphUncertainty


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("partial", [False, True])
def test_fixed_graph_covariance_recovers_observed_row_ols(*, partial: bool) -> None:
    """Mean covariance profiles the actual ML variance on the selected rows."""
    rng = np.random.default_rng(714)
    x = np.column_stack((np.ones(12), rng.normal(size=12)))
    y = x @ np.array([0.4, -0.7]) + rng.normal(size=12)
    rows = np.array([9, 2, 0, 7, 10, 4]) if partial else np.arange(12)
    design, outcome = (jnp.asarray(x),), (jnp.asarray(y[rows]),)
    observed = (jnp.asarray(rows),) if partial else None
    empty = (jnp.empty((0, len(x), len(x))),)
    fit = estimate_graph_ml(
        empty, empty, design, outcome, ("intercept", "x"), observed=observed
    )
    actual = _graph_covariance(fit, empty, empty, design, outcome, observed=observed)
    selected = x[rows]
    beta = np.linalg.lstsq(selected, y[rows], rcond=None)[0]
    sigma2 = np.mean((y[rows] - selected @ beta) ** 2)
    expected = sigma2 * np.linalg.inv(selected.T @ selected)
    assert actual.fit is fit
    assert actual.valid
    assert actual.sample_capacity == len(rows)
    np.testing.assert_allclose(actual.covariance.values, expected, atol=1e-11)

    # The native uncertainty remains an array pytree after the owner move.
    def covariance(state: GraphMLFit) -> _GraphUncertainty:
        return _graph_covariance(
            state, empty, empty, design, outcome, observed=observed
        )

    compiled = jax.jit(covariance)(fit)
    np.testing.assert_allclose(compiled.covariance.values, expected, atol=1e-11)
