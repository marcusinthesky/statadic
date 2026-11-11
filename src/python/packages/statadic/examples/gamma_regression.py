# Standalone documentation example.
# ruff: noqa: INP001, T201
"""Choose the dispersion convention and retain the native likelihood result."""

import jax
import jax.numpy as jnp
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.inference import ModelBased
from statadic.api.regression import Regression, Gamma, Observed
from statadic.estimation.likelihood import FixedDispersion

x = jnp.linspace(-1, 1, 128)
mean = jnp.exp(0.2 + 0.4 * x)
y = 0.6 * mean * jax.random.gamma(jax.random.key(413), 1 / 0.6, shape=(128,))
data = Observations({"x": x, "y": y})
fit = Regression(Observed('y'), Intercept() + Linear('x'), law=Gamma(dispersion=FixedDispersion(0.6))).estimate(data)
print(fit.report())
prediction = fit.predict(data)
intervals = fit.infer(covariance=ModelBased()).intervals(reference="normal")
assert bool(fit.valid) and bool(intervals.valid)
assert prediction.shape == (128,)
