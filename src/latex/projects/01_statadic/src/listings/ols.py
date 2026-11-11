import jax.numpy as jnp
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC
from statadic.api.regression import Regression

data = Observations({
    "x": jnp.arange(8.0),
    "y": jnp.array([1.0, 2.4, 2.8, 4.2, 4.9, 5.5, 7.3, 7.7]),
})
fit = Regression.ols("y", Intercept() + Linear("x")).estimate(data)

hc0 = fit.infer(covariance=HC("HC0"))   # one retained fit,
hc3 = fit.infer(covariance=HC("HC3"))   # two covariance procedures
report = hc3.report(reference="normal", confidence=0.95)
