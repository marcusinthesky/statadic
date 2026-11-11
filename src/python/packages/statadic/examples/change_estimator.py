"""Replace a panel estimator while preserving observations and design."""

# ruff: noqa: INP001, T201  # Standalone executable documentation.
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from statadic.api.data import Observations
from statadic.api.design import Linear
from statadic.api.data import PanelLayout
from statadic.api.regression import Regression
from statadic.estimation.panel import Pooled, Within

if TYPE_CHECKING:
    from statadic.api.panel import PanelResult


def compare() -> tuple[PanelResult, PanelResult]:
    """Return pooled and unit-within fits for a known synthetic relationship."""
    x = jnp.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    effects = jnp.array([-1.0, 0.0, 1.0])[:, None]
    noise = jnp.array([-0.1, 0.2, -0.1])
    data = Observations(
        {
            "x": x.reshape(-1),
            "y": (2 * x + effects + noise).reshape(-1),
            "period": jnp.tile(jnp.arange(3), 3),
        },
        labels={"unit": ("a",) * 3 + ("b",) * 3 + ("c",) * 3},
    )
    model = Regression(
        "y", Linear("x"), structure=PanelLayout("unit", "period"), estimator=Pooled()
    )
    pooled = model.estimate(data)
    # Replacement creates a new preparation owner: fit raw observations again.
    within = replace(model, effects="unit", estimator=Within()).estimate(data)
    if not pooled.valid or not within.valid:
        message = "Inspect the unavailable panel fit before interpretation."
        raise RuntimeError(message)
    # Noise is orthogonal to x in each unit. Unit effects shift the pooled slope.
    np.testing.assert_allclose(pooled.parameters, [2.125], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(within.parameters, [2.0], rtol=2e-5, atol=2e-5)
    return pooled, within


if __name__ == "__main__":
    for name, fit in zip(("Pooled", "Unit within"), compare(), strict=True):
        print(name, fit.report().to_text(), sep="\n")
    print(
        "Changing effects changes the estimating assumptions; no inference is selected."
    )
