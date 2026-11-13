"""Native spatial diagnostics studies."""

from dataclasses import replace
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.estimation.spatial import (
    ML,
    SAR,
    SpatialIV,
    SpatialProblem,
)
from statadic.inference.spatial import (
    spatial_ak_test,
    spatial_common_factor_test,
    spatial_covariance,
    spatial_diagnostics,
    spatial_likelihood_ratio,
)
from statadic.operators.spatial import (
    prepare_spatial_weights,
)

from statadic_studies.adapters._random import rng
from statadic_studies.adapters.dependence.spatial_durbin import BETA, THETA, durbin_data
from statadic_studies.adapters.dependence.spatial_extensions import (
    spatial_fixture,
    spatial_outcome,
)


def _test_record(test):
    if not np.isfinite([test.statistic, test.pvalue]).all():
        return {"valid": False, "failure": "spatial-diagnostic-unavailable"}
    return {
        "valid": True,
        "statistic": float(test.statistic),
        "pvalue": float(test.pvalue),
        "rejected": bool(test.pvalue <= 0.05),
    }


@cache
def _diagnostic_kernel():
    x, w = spatial_fixture()
    weights = prepare_spatial_weights(w)

    @jax.jit
    def one(y):
        fit = LeastSquares().estimate(LinearProblem(jnp.asarray(x), y))
        result = spatial_diagnostics(fit, weights)
        return (
            result.moran,
            result.lm_lag,
            result.lm_error,
            result.robust_lm_lag,
            result.robust_lm_error,
            result.lm_sarma,
        )

    return one


def _diagnostics(scenario, replicate, seed):
    x, w = spatial_fixture()
    rho = 0.3 if scenario == "lag-alternative" else 0.0
    lam = 0.3 if scenario == "error-alternative" else 0.0
    y = spatial_outcome(
        rng("spatial.ols-diagnostics", scenario, replicate, seed=seed),
        x,
        w,
        rho=rho,
        lam=lam,
    )
    results = _diagnostic_kernel()(jnp.asarray(y))
    return {
        name: _test_record(result)
        for name, result in zip(
            (
                "Moran-normal",
                "lag-LM",
                "error-LM",
                "robust-lag-LM",
                "robust-error-LM",
                "joint-SARMA-LM",
            ),
            results,
            strict=True,
        )
    }


@cache
def _ak_kernel():
    x, w = spatial_fixture()
    weights = prepare_spatial_weights(w)

    @jax.jit
    def one(y):
        fit = SpatialIV(instrument_columns=(1, 2)).estimate(
            SpatialProblem(jnp.asarray(x), y, weights, process=SAR(), slx_columns=())
        )
        return spatial_ak_test(fit)

    return one


def _ak(scenario, replicate, seed):
    x, w = spatial_fixture()
    y = spatial_outcome(
        rng("spatial.ak-diagnostic", scenario, replicate, seed=seed),
        x,
        w,
        rho=0.25,
        lam=0.3 if scenario == "error-alternative" else 0.0,
    )
    return {"Anselin-Kelejian": _test_record(_ak_kernel()(jnp.asarray(y)))}


@cache
def _restriction_kernel(likelihood_ratio):
    x, w = spatial_fixture()
    weights = prepare_spatial_weights(w)

    @jax.jit
    def one(y):
        problem = SpatialProblem(jnp.asarray(x), y, weights, process=SAR())
        unrestricted = ML().estimate(
            replace(problem, process=SAR(), slx_columns=(1, 2))
        )
        if likelihood_ratio:
            restricted = ML().estimate(replace(problem, process=SAR(), slx_columns=()))
            return spatial_likelihood_ratio(
                unrestricted, restricted, degrees_of_freedom=2, assume_nested=True
            )
        return spatial_common_factor_test(spatial_covariance(unrestricted))

    return one


def _restrictions(study, scenario, replicate, seed):
    lr = study == "spatial.nested-likelihood-ratio"
    theta = np.zeros(2) if lr else -0.25 * BETA[1:]
    if scenario == "restriction-alternative":
        theta = THETA
    _, y, _ = durbin_data(rng(study, scenario, replicate, seed=seed), "SDM", theta)
    return {
        "interior-LR" if lr else "common-factor-Wald": _test_record(
            _restriction_kernel(lr)(jnp.asarray(y))
        )
    }
