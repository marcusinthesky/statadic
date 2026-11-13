"""Scientific adapter functions grouped by computational ownership."""

from __future__ import annotations

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.inference.linear_covariance import classical_covariance, hc_covariance
from statadic.inference.parameters import coefficient_intervals, linear_restriction_test

from statadic_studies.adapters._random import rng


@cache
def _linear_kernel(n, reference):
    x = jnp.column_stack([jnp.ones(n), jnp.linspace(-2, 2, n)])

    def one(y):
        fit = LeastSquares().estimate(LinearProblem(x, y, jnp.ones(n)))
        outputs = []
        for method in ("classical", "HC0", "HC1", "HC2", "HC3"):
            covariance = (
                classical_covariance(fit)
                if method == "classical"
                else hc_covariance(fit, kind=method)
            )
            interval = coefficient_intervals(
                covariance,
                reference=reference,
                degrees_of_freedom=fit.df_resid if reference == "t" else None,
            )
            test = linear_restriction_test(
                covariance,
                jnp.array([[0.0, 1.0]]),
                0.0,
                reference="f" if reference == "t" else "chi2",
                degrees_of_freedom=fit.df_resid if reference == "t" else None,
            )
            low, high = interval.estimate.ci
            valid = fit.valid & covariance.valid & jnp.all(interval.valid) & test.valid
            outputs.append(
                (valid, fit.parameters[1], low[1], high[1], test.test.pvalue)
            )
        return tuple(outputs)

    return jax.jit(one)


def _linear(scenario, replicate, seed):
    law, size, reference = scenario.split("-")
    n = int(size[1:])
    # Reference-law comparisons share the identical generated data.
    generator = rng("linear.ols-covariance", f"{law}-{size}", replicate, seed=seed)
    predictor = np.linspace(-2, 2, n)
    sigma = 1.0 if law == "homoskedastic" else 0.3 + np.abs(predictor)
    y = 0.4 + generator.normal(size=n) * sigma
    outputs = jax.device_get(_linear_kernel(n, reference)(jnp.asarray(y)))
    records = {}
    for method, values in zip(
        ("classical", "HC0", "HC1", "HC2", "HC3"), outputs, strict=True
    ):
        ok, estimate, low, high, pvalue = values
        finite = np.isfinite([estimate, low, high, pvalue]).all()
        if not bool(ok and finite):
            records[method] = {"valid": False, "failure": "unavailable-or-nonfinite"}
        else:
            records[method] = {
                "valid": True,
                "estimate": float(estimate),
                "truth": 0.0,
                "covered": bool(low <= 0 <= high),
                "width": float(high - low),
                "rejected": bool(pvalue < 0.05),
                "pvalue": float(pvalue),
            }
    return records
