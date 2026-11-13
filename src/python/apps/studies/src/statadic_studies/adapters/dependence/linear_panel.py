"""Scientific adapter functions grouped by computational ownership."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.estimation.panel import (
    Between,
    FirstDifference,
    PanelProblem,
    Pooled,
    RandomEffects,
    Within,
)
from statadic.inference.linear_covariance import classical_covariance, hc_covariance
from statadic.inference.panel_covariance import panel_covariance
from statadic.operators.panel import prepare_panel_index

from statadic_studies.adapters._random import rng
from statadic_studies.adapters.dependence._common import _inference, _record

PANEL_METHODS = (
    "pooled-classical",
    "pooled-cluster-entity",
    "within-classical",
    "within-cluster-entity",
    "random-effects-classical",
    "random-effects-cluster-entity",
    "between-classical",
    "first-difference-classical",
)


@cache
def _wls_kernel():
    def one(x, y, precision):
        ols = LeastSquares().estimate(LinearProblem(x, y))
        wls = LeastSquares().estimate(LinearProblem(x, y, precision))
        return (
            _inference(ols, classical_covariance(ols), 1, 0.4),
            _inference(ols, hc_covariance(ols, kind="HC3"), 1, 0.4),
            _inference(wls, classical_covariance(wls), 1, 0.4),
            _inference(wls, hc_covariance(wls, kind="HC3"), 1, 0.4),
        )

    return jax.jit(one)


def wls_data(generator):
    x = np.column_stack((np.ones(128), np.linspace(-2, 2, 128)))
    sigma = 0.3 + np.abs(x[:, 1])
    return (
        x,
        x @ np.array([0.2, 0.4]) + sigma * generator.normal(size=128),
        1 / sigma**2,
    )


def _wls(scenario, replicate, seed):
    generator = rng(
        "linear.weighted-inference", "heteroskedastic", replicate, seed=seed
    )
    x, y, precision = wls_data(generator)
    if scenario == "reversed-precision":
        precision = 1 / precision
    outputs = jax.device_get(
        _wls_kernel()(jnp.asarray(x), jnp.asarray(y), jnp.asarray(precision))
    )
    return {
        method: _record(value, 0.4)
        for method, value in zip(
            ("OLS-classical", "OLS-HC3", "WLS-classical", "WLS-HC3"),
            outputs,
            strict=True,
        )
    }


def panel_design():
    """Fixed complete balanced observation order: entity major, then actual time."""
    entity, time = np.repeat(np.arange(24), 6), np.tile(np.arange(6), 24)
    generator = rng("panel.static-estimation", "fixed-design", 0, seed=211)
    x = np.column_stack((np.ones(len(entity)), generator.normal(size=(len(entity), 2))))
    return x, entity, time


@cache
def _panel_kernel():
    x, entity, time = panel_design()
    index = prepare_panel_index(entity, time)

    def one(y):
        outputs = []
        for estimator, effects in (
            (Pooled(), "pooled"),
            (Within(tolerance=1e-6), "unit"),
            (RandomEffects(), "random"),
            (Between(), "pooled"),
        ):
            problem = PanelProblem(
                jnp.asarray(x), y, index, effects=effects, intercept_column=0
            )
            fit = estimator.estimate(problem)
            outputs.append(_inference(fit, panel_covariance(fit), 1, 0.0))
            if isinstance(estimator, (Pooled, Within, RandomEffects)):
                outputs.append(
                    _inference(
                        fit,
                        panel_covariance(
                            fit, method="cluster-entity", group_debias=True
                        ),
                        1,
                        0.0,
                    )
                )
        # Differencing removes the intercept, so it is excluded explicitly.
        fit = FirstDifference().estimate(
            PanelProblem(jnp.asarray(x[:, 1:]), y, index, effects="pooled")
        )
        outputs.append(_inference(fit, panel_covariance(fit), 0, 0.0))
        return tuple(outputs)

    return jax.jit(one)


def _panel(scenario, replicate, seed):
    generator = rng("panel.static-estimation", scenario, replicate, seed=seed)
    x, entity, _ = panel_design()
    effect = 0.7 * generator.normal(size=24)
    if scenario == "correlated-effects":
        effect += 0.8 * x[:, 1].reshape(24, 6).mean(axis=1)
    y = (
        x @ np.array([0.4, 0, -0.7])
        + effect[entity]
        + generator.normal(size=len(entity))
    )
    return {
        method: _record(value, 0.0)
        for method, value in zip(
            PANEL_METHODS, jax.device_get(_panel_kernel()(jnp.asarray(y))), strict=True
        )
    }
