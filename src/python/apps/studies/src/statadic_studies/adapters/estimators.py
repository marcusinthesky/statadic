"""Complete estimator procedures with known population targets.

These descriptive studies measure finite-sample operating characteristics. They
never promote normal approximations or finite solver iterates to exact validity.
"""

from __future__ import annotations

import math
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.core.equations import MeanCenteredLongRun
from statadic.estimation.gel import FixedGEL
from statadic.estimation.glm import (
    BinomialLogitMLE,
    BinomialProblem,
    PoissonMLE,
    PoissonProblem,
)
from statadic.estimation.gmm import FixedWeightGMM, IteratedGMM
from statadic.estimation.moments import AffineMomentProblem, FixedMomentProblem
from statadic.estimation.survival import SurvivalML, SurvivalProblem
from statadic.inference.gel import gel_test
from statadic.inference.glm_covariance import glm_covariance
from statadic.inference.gmm import hansen_j_from_fit
from statadic.inference.gmm.covariance import gmm_covariance
from statadic.inference.survival import observed_information
from statadic.model.survival import LogNormalAFT, WeibullAFT
from scipy.stats import norm

from statadic_studies.adapters._random import rng


def normal_record(estimate, truth, variance, valid):
    """Retain normal-Wald performance at the declared population parameter."""
    estimate, variance = float(estimate), float(variance)
    if not bool(valid) or not np.isfinite([estimate, variance]).all() or variance <= 0:
        return {"valid": False, "failure": "fit-or-covariance-unavailable"}
    se = math.sqrt(variance)
    statistic = (estimate - truth) / se
    pvalue = math.erfc(abs(statistic) / math.sqrt(2))
    return {
        "valid": True,
        "estimate": estimate,
        "truth": truth,
        "se": se,
        "covered": abs(statistic) <= 1.959963984540054,
        "width": 2 * 1.959963984540054 * se,
        "pvalue": pvalue,
        "rejected": pvalue < 0.05,
    }


def iv_data(generator, n=256, *, heteroskedastic=False):
    """Strong instruments; structural error correlated with endogenous regressors."""
    instruments = generator.normal(size=(n, 4))
    shocks = generator.normal(size=(n, 3))
    design = instruments @ np.array([[1, 0.1], [0.3, 0.4], [0.1, 1.2], [0.5, -0.3]])
    design += 0.5 * shocks[:, :2]
    scale = 0.3 + np.abs(instruments[:, 1]) if heteroskedastic else 1.0
    error = 0.3 * shocks[:, 0] + shocks[:, 2] * scale
    outcome = design @ np.array([0.7, -0.4]) + error
    return design, outcome, instruments


@cache
def _gmm_kernel(iv=False):
    def one(g0, derivative, iv_weight):
        outputs = []
        weights = (jnp.eye(g0.shape[1]), iv_weight) if iv else (jnp.eye(g0.shape[1]),)
        for weight in weights:
            fit = FixedWeightGMM(MeanCenteredLongRun).estimate(
                AffineMomentProblem(g0, derivative, weight)
            )
            uncertainty = gmm_covariance(fit)
            outputs.append(
                (
                    fit.parameters[0],
                    uncertainty.covariance.values[0, 0],
                    fit.valid & uncertainty.valid,
                )
            )
        problem = AffineMomentProblem(g0, derivative)
        fit = IteratedGMM(MeanCenteredLongRun).estimate(problem)
        uncertainty = gmm_covariance(fit)
        tested = hansen_j_from_fit(fit, assume_correct_specification=True)
        outputs.append(
            (
                fit.parameters[0],
                uncertainty.covariance.values[0, 0],
                fit.valid & uncertainty.valid,
            )
        )
        return tuple(outputs), (tested.test.pvalue, tested.valid)

    return jax.jit(one)


def _gmm(study, scenario, replicate, seed):
    generator = rng(study, scenario, replicate, seed=seed)
    iv = study == "iv.structural-estimation"
    if iv:
        x, y, z = iv_data(generator, heteroskedastic=scenario == "heteroskedastic")
        g0 = z * y[:, None]
        derivative = -z[:, :, None] * x[:, None, :]
        weight = np.linalg.inv(z.T @ z / len(z))
        truth, methods = 0.7, ("identity-GMM", "2SLS", "iterated-GMM")
    else:
        d = np.array([[1.0], [0.5], [-0.25]])
        shift = np.array([0, 0.2, 0.2]) if scenario == "misspecified" else np.zeros(3)
        g0 = generator.normal(size=(256, 3)) - d[:, 0] * 0.4 + shift
        derivative, weight = d, np.eye(3)
        truth, methods = 0.4, ("identity-GMM", "iterated-GMM")
    outputs, (pvalue, valid) = jax.device_get(
        _gmm_kernel(iv)(jnp.asarray(g0), jnp.asarray(derivative), jnp.asarray(weight))
    )
    records = {
        method: normal_record(values[0], truth, values[1], values[2])
        for method, values in zip(methods, outputs, strict=True)
    }
    # J calibration is attached to the actual efficient fit, separately from Wald.
    records["Hansen-J"] = (
        {"valid": True, "pvalue": float(pvalue), "rejected": bool(pvalue < 0.05)}
        if bool(valid) and np.isfinite(pvalue)
        else {"valid": False, "failure": "J-reference-unavailable"}
    )
    return records


@cache
def _glm_kernel(family):
    def one(x, y):
        fit = (
            BinomialLogitMLE().estimate(BinomialProblem(x, y, trials=5))
            if family == "binomial"
            else PoissonMLE().estimate(PoissonProblem(x, y))
        )
        return tuple(
            (
                fit.parameters[1],
                (u := glm_covariance(fit, method=method)).covariance.values[1, 1],
                fit.valid & u.valid,
            )
            for method in ("model", "HC0", "HC1")
        )

    return jax.jit(one)


def _glm(scenario, replicate, seed):
    generator = rng("glm.canonical-inference", scenario, replicate, seed=seed)
    x = np.column_stack((np.ones(128), np.linspace(-1.5, 1.5, 128)))
    eta = x @ np.array([0.2, 0.4])
    y = (
        generator.binomial(5, 1 / (1 + np.exp(-eta)))
        if scenario == "binomial"
        else generator.poisson(np.exp(eta))
    )
    outputs = jax.device_get(_glm_kernel(scenario)(jnp.asarray(x), jnp.asarray(y)))
    return {
        method: normal_record(values[0], 0.4, values[1], values[2])
        for method, values in zip(("model", "HC0", "HC1"), outputs, strict=True)
    }


def _gel(scenario, replicate, seed):
    generator = rng("gel.fixed-moments", scenario, replicate, seed=seed)
    if scenario == "rare-null":
        moments = generator.binomial(1, 0.05, size=(512, 1)) - 0.05
    else:
        moments = generator.normal(size=(256, 2)) + (
            0 if scenario == "normal-null" else 0.2
        )
    records = {}
    for criterion in ("cue", "el", "et"):
        try:
            fit = FixedGEL(criterion=criterion).estimate(
                FixedMomentProblem(jnp.asarray(moments))
            )
            tested = gel_test(fit, assume_iid=True)
            valid = bool(
                tested.converged
                and tested.reference_valid
                and np.isfinite(tested.pvalue)
            )
            records[criterion] = (
                {
                    "valid": True,
                    "pvalue": float(tested.pvalue),
                    "rejected": bool(tested.pvalue < 0.05),
                }
                if valid
                else {"valid": False, "failure": "dual-or-reference-unavailable"}
            )
        except (ValueError, FloatingPointError) as exc:
            records[criterion] = {"valid": False, "failure": type(exc).__name__}
    return records


def survival_data(generator, family, delayed, n=128):
    """Independent censoring and conditional-on-entry event-time sampling."""
    x = np.column_stack((np.ones(n), generator.normal(size=(n, 2))))
    mu = x @ np.array([0.2, 0.4, -0.3])
    entry = generator.uniform(0, 1, size=n) if delayed else np.zeros(n)
    uniform = generator.uniform(size=n)
    if family == "weibull":
        event_time = np.exp(mu) * ((entry / np.exp(mu)) ** 1.5 - np.log(uniform)) ** (
            1 / 1.5
        )
    else:
        at_entry = np.where(
            entry > 0, norm.sf((np.log(np.where(entry > 0, entry, 1)) - mu) / 0.7), 1
        )
        event_time = np.exp(mu + 0.7 * norm.isf(uniform * at_entry))
    censor_time = entry + generator.exponential(scale=2, size=n)
    return x, np.minimum(event_time, censor_time), event_time <= censor_time, entry


@cache
def _survival_kernel(family):
    def one(x, duration, event, entry):
        aft = WeibullAFT() if family == "weibull" else LogNormalAFT()
        fit = SurvivalML(aft).estimate(SurvivalProblem(x, duration, event, entry))
        uncertainty = observed_information(fit)
        return (
            fit.parameters[1],
            uncertainty.covariance.values[1, 1],
            fit.valid & uncertainty.valid,
        )

    return jax.jit(one)


def _survival(scenario, replicate, seed):
    family, entry = scenario.split("-")
    generator = rng("survival.independent-aft", scenario, replicate, seed=seed)
    data = survival_data(generator, family, entry == "delayed")
    estimate, variance, valid = jax.device_get(
        _survival_kernel(family)(*(jnp.asarray(v) for v in data))
    )
    return {"observed-information-Wald": normal_record(estimate, 0.4, variance, valid)}


def run_estimators(study, scenario, replicate, seed):
    """Dispatch only reviewed concrete estimator designs."""
    if study in {"gmm.affine-mean", "iv.structural-estimation"}:
        return _gmm(study, scenario, replicate, seed)
    if study == "glm.canonical-inference":
        return _glm(scenario, replicate, seed)
    if study == "gel.fixed-moments":
        return _gel(scenario, replicate, seed)
    if study == "survival.independent-aft":
        return _survival(scenario, replicate, seed)
    raise ValueError(f"No estimator adapter for {study}")
