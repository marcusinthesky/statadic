"""Change uncertainty estimation while preserving the point fit."""

import jax.numpy as jnp
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC
from statadic.api.regression import Regression


def compare():
    """Return HC0 and HC3 normal-reference reports from one OLS fit."""
    data = Observations(
        {
            "x": jnp.arange(8, dtype=float),
            "y": jnp.array([1.0, 2.4, 2.8, 4.2, 4.9, 5.5, 7.3, 7.7]),
        }
    )
    fit = Regression(response='y', terms=Intercept() + Linear('x')).estimate(data)
    if not fit.valid:
        raise RuntimeError("Point fit unavailable")
    reports = tuple(
        fit.infer(covariance=HC(method)).report(reference="normal")
        for method in ("HC0", "HC3")
    )
    for report in reports:
        report.require_available("interval")
    return reports


if __name__ == "__main__":
    for method, report in zip(("HC0", "HC3"), compare(), strict=True):
        print(method, report.to_text(), sep="\n")
