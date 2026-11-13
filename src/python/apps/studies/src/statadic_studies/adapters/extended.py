"""Actual numerical producers for the additional descriptive designs."""

from __future__ import annotations

import math
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr

from statadic_studies.adapters._random import rng
from statadic_studies.adapters.estimators import normal_record
from statadic_studies.definitions.extended import EXTENDED_STUDIES


@cache
def _likelihood_kernel(family, estimated, covariance):
    from statadic.estimation.likelihood import (
        EstimatedDispersion,
        FixedDispersion,
        GammaMLE,
        LikelihoodProblem,
        NegativeBinomialMLE,
        ProbitMLE,
    )
    from statadic.inference.likelihood import likelihood_covariance

    policy = EstimatedDispersion() if estimated else FixedDispersion(0.6)
    estimator = (
        GammaMLE(policy)
        if family == "gamma"
        else NegativeBinomialMLE(policy)
        if family == "nb2"
        else ProbitMLE()
    )

    def one(x, y, trials):
        fit = estimator.estimate(LikelihoodProblem(x, y, trials=trials))
        uncertainty = likelihood_covariance(fit, method=covariance)
        return (
            fit.parameters[1],
            uncertainty.covariance.values[1, 1],
            fit.valid & uncertainty.valid,
        )

    return jax.jit(one)


def _likelihood(study, scenario, replicate, seed):
    generator = rng(study, scenario, replicate, seed=seed)
    family = study.split(".")[1].removesuffix("-inference")
    x = np.column_stack((np.ones(128), np.linspace(-1, 1, 128)))
    eta = x @ np.array([0.2, 0.4])
    trials = 5 if scenario == "grouped" else 1
    if family == "gamma":
        y = generator.gamma(1 / 0.6, 0.6 * np.exp(eta))
    elif family == "nb2":
        y = generator.negative_binomial(1 / 0.6, 1 / (1 + 0.6 * np.exp(eta)))
    else:
        y = generator.binomial(trials, ndtr(eta))
    result = {}
    for method in EXTENDED_STUDIES[study].methods:
        estimated = method.startswith("estimated")
        covariance = method.rsplit("-", 1)[-1]
        estimate, variance, valid = _likelihood_kernel(family, estimated, covariance)(
            jnp.asarray(x), jnp.asarray(y), jnp.asarray(trials)
        )
        result[method] = normal_record(estimate, 0.4, variance, valid)
    return result


@cache
def _survival_kernel(name):
    from statadic.estimation.survival import SurvivalML, SurvivalProblem
    from statadic.model.survival import (
        LogLogisticAFT,
        LogNormalAFT,
        WeibullAFT,
        survival_mean,
    )

    family = {
        "weibull": WeibullAFT,
        "lognormal": LogNormalAFT,
        "loglogistic": LogLogisticAFT,
    }[name]()
    estimator = SurvivalML(family)

    def one(x, duration, event, entry):
        fit = estimator.estimate(SurvivalProblem(x, duration, event, entry))
        value = survival_mean(fit.parameters, jnp.asarray([[1.0, 0.0]]), family)[0]
        return value, fit.valid & jnp.isfinite(value)

    return jax.jit(one)


def _survival(study, scenario, replicate, seed):
    generator = rng(study, scenario, replicate, seed=seed)
    family = scenario.split("-")[0]
    x = np.column_stack((np.ones(128), generator.uniform(-1, 1, 128)))
    eta = x @ np.array([0.2, 0.4])
    entry = (
        generator.uniform(0, 0.3, 128)
        if scenario.endswith("delayed")
        else np.zeros(128)
    )
    u = generator.uniform(size=128)
    if family == "weibull":
        duration = np.exp(eta) * (-np.log1p(-u)) ** (1 / 1.8)
        truth = math.exp(0.2) * math.gamma(1 + 1 / 1.8)
    elif family == "lognormal":
        duration = np.exp(eta + 0.6 * generator.normal(size=128))
        truth = math.exp(0.2 + 0.6**2 / 2)
    else:
        cdf_entry = 1 / (1 + (np.exp(eta) / np.where(entry > 0, entry, 1)) ** 1.8)
        cdf_entry = np.where(entry > 0, cdf_entry, 0)
        q = cdf_entry + (1 - cdf_entry) * u
        duration = np.exp(eta) * (q / (1 - q)) ** (1 / 1.8)
        truth = math.exp(0.2) * (math.pi / 1.8) / math.sin(math.pi / 1.8)
    censor = entry + generator.exponential(3, 128)
    estimate, valid = _survival_kernel(family)(
        jnp.asarray(x),
        jnp.asarray(np.minimum(duration, censor)),
        jnp.asarray(duration <= censor),
        jnp.asarray(entry),
    )
    return {
        "fitted-mean": {"valid": True, "estimate": float(estimate), "truth": truth}
        if valid
        else {"valid": False, "failure": "fit-or-mean-unavailable"}
    }


def _iv(study, scenario, replicate, seed):
    from statadic.core.equations import OneWayCluster, PlainLongRun
    from statadic.estimation.iv import (
        IteratedIVGMM,
        IVProblem,
        TwoStageLeastSquares,
        TwoStepIVGMM,
    )
    from statadic.inference.iv import iv_covariance

    generator = rng(study, scenario, replicate, seed=seed)
    z = generator.normal(size=(128, 4))
    shock, noise = generator.normal(size=(2, 128))
    groups = np.repeat(np.arange(16), 8)
    if scenario == "hac":
        noise[0] /= math.sqrt(1 - 0.3**2)
        for i in range(1, 128):
            noise[i] += 0.3 * noise[i - 1]
    elif scenario == "clustered":
        noise += generator.normal(size=16)[groups]
    elif scenario == "heteroskedastic":
        noise *= 0.3 + np.abs(z[:, 1])
    strength = 0.03 if scenario == "weak" else 1.0
    x = (strength * (0.8 * z[:, 0] + 0.4 * z[:, 1]) + 0.5 * shock)[:, None]
    y = 0.7 * x[:, 0] + 0.3 * shock + noise
    cluster = scenario == "clustered"
    scheme = OneWayCluster if cluster else PlainLongRun
    bandwidth = 4 if scenario == "hac" else 0
    problem = IVProblem(
        jnp.asarray(x),
        jnp.asarray(y),
        jnp.asarray(z),
        groups=jnp.asarray(groups) if cluster else None,
        n_clusters=16 if cluster else 0,
    )
    estimators = (
        TwoStageLeastSquares(),
        TwoStepIVGMM(scheme, bandwidth),
        IteratedIVGMM(scheme, bandwidth),
    )
    results = {}
    for method, estimator in zip(
        EXTENDED_STUDIES[study].methods, estimators, strict=True
    ):
        fit = estimator.estimate(problem)
        uncertainty = iv_covariance(
            fit,
            kind="cluster" if cluster else "HAC" if bandwidth else "HC0",
            bandwidth=bandwidth,
            groups=problem.groups,
            n_clusters=problem.n_clusters,
        )
        results[method] = normal_record(
            fit.parameters[0],
            0.7,
            uncertainty.covariance.values[0, 0],
            fit.valid
            & uncertainty.valid
            & (fit.converged if method == "iterated" else True),
        )
    return results


def run_extended(study, scenario, replicate, seed):
    """Execute the explicitly registered family without inferring applicability."""
    if (
        study not in EXTENDED_STUDIES
        or scenario not in EXTENDED_STUDIES[study].scenarios
    ):
        raise ValueError("Study and scenario must be explicitly registered together.")
    if study.startswith("likelihood."):
        return _likelihood(study, scenario, replicate, seed)
    if study == "survival.aft-mean-prediction":
        return _survival(study, scenario, replicate, seed)
    return _iv(study, scenario, replicate, seed)
