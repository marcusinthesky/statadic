"""Compose training-fitted principal coordinates directly into regression."""

# ruff: noqa: INP001, T201  # Standalone executable documentation prints its result.

import jax.numpy as jnp
from statadic.api.data import Observations
from statadic.api.design import Intercept
from statadic.api.ordination import LinearGram, Ordination, PrincipalCoordinates
from statadic.api.regression import LinearResult, Regression


def main() -> None:
    """Fit the full composition on training rows and predict held-out rows."""
    x = jnp.linspace(-2.0, 2.0, 12)
    z = x**2 + 0.15 * jnp.sin(3 * x)
    data = Observations(
        {"x": x, "z": z, "y": 1 + 0.8 * x - 0.2 * z + 0.1 * jnp.cos(2 * x)}
    )
    training = data.take(tuple(range(10)))
    held_out = data.take((10, 11))
    components = Ordination(
        LinearGram(features=("x", "z"), scale=True),
        PrincipalCoordinates(dim=1),
        name="principal",
    )
    result = Regression("y", Intercept() + components).estimate(training)
    if not isinstance(result, LinearResult) or not result.valid:
        message = "The synthetic linear fit is unavailable."
        raise RuntimeError(message)
    print(result.report().to_text())
    print("Held-out predictions:", result.predict(held_out))
    print(
        "Synthetic illustration; no inference or predictive calibration is established."
    )


if __name__ == "__main__":
    main()
