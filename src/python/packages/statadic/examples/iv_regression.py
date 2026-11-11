# Standalone documentation example.
# ruff: noqa: INP001, T201
"""Fit a structural index with explicit excluded instruments and covariance."""

import jax
import jax.numpy as jnp
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC
from statadic.api.regression import Regression
from statadic.estimation.iv import TwoStageLeastSquares

z, shock, noise = jax.random.normal(jax.random.key(412), (3, 128))
treatment = 0.8 * z + 0.5 * shock
y = 0.2 + 0.7 * treatment + 0.3 * shock + noise
data = Observations({"y": y, "treatment": treatment, "assignment": z})
fit = Regression(
    "y",
    Intercept() + Linear("treatment"),
    estimator=TwoStageLeastSquares(),
    endogenous=("treatment",),
    instruments=("assignment",),
).estimate(data)
print(fit.report())
intervals = fit.infer(covariance=HC("HC0")).intervals(reference="normal")
structural_index = fit.predict(data)
if (
    not bool(fit.valid)
    or not bool(intervals.valid)
    or structural_index.shape != (128,)
    or not bool(jnp.all(jnp.isfinite(structural_index)))
):
    message = "The synthetic IV fit or its inference/predictions are unavailable."
    raise RuntimeError(message)
