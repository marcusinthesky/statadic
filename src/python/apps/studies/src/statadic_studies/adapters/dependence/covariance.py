"""Native HAC, clustering, panel kernel and Fama-MacBeth inference studies."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.estimation.panel import FamaMacBeth, PanelProblem, Within
from statadic.inference.linear_covariance import (
    classical_covariance,
    cluster_covariance,
    hac_covariance,
)
from statadic.inference.panel_covariance import fama_macbeth_covariance, panel_covariance
from statadic.operators.panel import prepare_panel_index

from statadic_studies.adapters._random import rng
from statadic_studies.adapters.dependence._common import _inference, _record

LINEAR_COVARIANCE_METHODS = (
    "classical",
    "HAC-b0",
    "HAC-b4",
    "HAC-b12",
    "cluster-uncorrected",
    "cluster-G-corrected",
)
PANEL_COVARIANCE_METHODS = (
    "classical",
    "heteroskedastic",
    "cluster-entity",
    "cluster-time",
    "cluster-two-way",
    "autocorrelation-bartlett",
    "autocorrelation-parzen",
    "autocorrelation-qs",
    "driscoll-kraay-bartlett",
    "driscoll-kraay-parzen",
    "driscoll-kraay-qs",
)
FAMA_COVARIANCE_METHODS = ("iid-b0", "HAC-bartlett", "HAC-parzen", "HAC-qs")


def stationary_ar(generator, length, rho, shape=()):
    """Unit marginal variance AR(1), initialized from its stationary law."""
    values = generator.normal(size=(length, *shape))
    for t in range(1, length):
        values[t] = rho * values[t - 1] + np.sqrt(1 - rho**2) * values[t]
    return values


def linear_covariance_data(generator, scenario):
    n = 256
    fixed = rng("linear.dependence-covariance", "fixed-design", 0, seed=350)
    x = np.column_stack((np.ones(n), fixed.normal(size=n)))
    groups = np.repeat(np.arange(32), 8)
    if scenario == "serial-AR1":
        error = stationary_ar(generator, n, 0.7)
    else:
        error = 0.8 * generator.normal(size=32)[groups] + 0.6 * generator.normal(size=n)
    return x, x @ np.array([0.2, 0.4]) + error, groups


@cache
def _linear_kernel():
    groups = jnp.repeat(jnp.arange(32), 8)

    def one(x, y):
        fit = LeastSquares().estimate(LinearProblem(x, y))
        covariances = (
            classical_covariance(fit),
            *(hac_covariance(fit, b) for b in (0, 4, 12)),
            cluster_covariance(fit, groups, n_clusters=32),
            cluster_covariance(
                fit, groups, n_clusters=32, finite_sample_correction=32 / 31
            ),
        )
        return tuple(
            _inference(fit, uncertainty, 1, 0.4, reference="normal")
            for uncertainty in covariances
        )

    return jax.jit(one)


def _linear(scenario, replicate, seed):
    generator = rng("linear.dependence-covariance", scenario, replicate, seed=seed)
    x, y, _ = linear_covariance_data(generator, scenario)
    outputs = jax.device_get(_linear_kernel()(jnp.asarray(x), jnp.asarray(y)))
    return {
        method: _record(value, 0.4)
        for method, value in zip(LINEAR_COVARIANCE_METHODS, outputs, strict=True)
    }


def panel_covariance_fixture(scenario):
    entities, periods = 24, 24
    entity = np.repeat(np.arange(entities), periods)
    actual_times = np.arange(periods) * (2 if scenario == "gapped-time" else 1)
    times = np.tile(actual_times, entities)
    fixed = rng("panel.dependence-covariance", "fixed-design", 0, seed=351)
    x = np.column_stack((np.ones(len(entity)), fixed.normal(size=len(entity))))
    return x, entity, times


def panel_covariance_data(generator, scenario):
    x, entity, times = panel_covariance_fixture(scenario)
    period_ids = np.unique(times)
    all_times = int(period_ids[-1]) + 1
    idiosyncratic = stationary_ar(generator, all_times, 0.6, (24,))[
        period_ids
    ].T.ravel()
    common = stationary_ar(generator, all_times, 0.5)[period_ids]
    error = 0.6 * idiosyncratic + 0.5 * np.tile(common, 24)
    y = x @ np.array([0.2, 0.4]) + 0.7 * generator.normal(size=24)[entity] + error
    return x, y, entity, times


@cache
def _panel_kernel(scenario):
    x, entity, times = panel_covariance_fixture(scenario)
    index = prepare_panel_index(entity, times)

    def one(y):
        fit = Within(tolerance=1e-6).estimate(
            PanelProblem(jnp.asarray(x), y, index, effects="unit", intercept_column=0)
        )
        covariances = [
            panel_covariance(fit, method=method, group_debias=True)
            for method in PANEL_COVARIANCE_METHODS[:5]
        ]
        for method in ("autocorrelation", "driscoll-kraay"):
            covariances.extend(
                panel_covariance(fit, method=method, bandwidth=4, kernel=kernel)
                for kernel in ("bartlett", "parzen", "qs")
            )
        return tuple(
            _inference(fit, uncertainty, 1, 0.4, reference="normal")
            for uncertainty in covariances
        )

    return jax.jit(one)


def _panel(scenario, replicate, seed):
    generator = rng("panel.dependence-covariance", scenario, replicate, seed=seed)
    _, y, _, _ = panel_covariance_data(generator, scenario)
    outputs = jax.device_get(_panel_kernel(scenario)(jnp.asarray(y)))
    return {
        method: _record(value, 0.4)
        for method, value in zip(PANEL_COVARIANCE_METHODS, outputs, strict=True)
    }


def fama_fixture():
    entity, times = np.repeat(np.arange(32), 32), np.tile(np.arange(32), 32)
    fixed = rng("panel.fama-macbeth", "fixed-design", 0, seed=352)
    x = np.column_stack((np.ones(len(entity)), fixed.normal(size=len(entity))))
    return x, entity, times


def fama_data(generator, scenario):
    x, entity, times = fama_fixture()
    slope = 0.4 + 0.2 * stationary_ar(
        generator, 32, 0.6 if scenario == "serial-slopes" else 0
    )
    y = 0.2 + slope[times] * x[:, 1] + 0.5 * generator.normal(size=len(entity))
    return x, y, entity, times


@cache
def _fama_kernel():
    x, entity, times = fama_fixture()
    index = prepare_panel_index(entity, times)

    def one(y):
        fit = FamaMacBeth().estimate(
            PanelProblem(jnp.asarray(x), y, index, effects="pooled")
        )
        covariances = (
            fama_macbeth_covariance(fit, bandwidth=0),
            *(
                fama_macbeth_covariance(fit, bandwidth=4, kernel=kernel)
                for kernel in ("bartlett", "parzen", "qs")
            ),
        )
        return tuple(
            _inference(fit, uncertainty, 1, 0.4, reference="normal")
            for uncertainty in covariances
        )

    return jax.jit(one)


def _fama(scenario, replicate, seed):
    generator = rng("panel.fama-macbeth", scenario, replicate, seed=seed)
    _, y, _, _ = fama_data(generator, scenario)
    outputs = jax.device_get(_fama_kernel()(jnp.asarray(y)))
    return {
        method: _record(value, 0.4)
        for method, value in zip(FAMA_COVARIANCE_METHODS, outputs, strict=True)
    }


def run_covariance_dependence(study, scenario, replicate, seed):
    adapters = {
        "linear.dependence-covariance": _linear,
        "panel.dependence-covariance": _panel,
        "panel.fama-macbeth": _fama,
    }
    if study not in adapters:
        raise ValueError(f"No dependent covariance adapter for {study}")
    return adapters[study](scenario, replicate, seed)
