"""Native spatial panel studies."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.estimation.spatial import (
    SAR,
    SEM,
    ObservedPanelML,
    ObservedSpatialPanelProblem,
    PanelML,
    SpatialPanelProblem,
)
from statadic.inference.spatial import (
    spatial_covariance,
)
from statadic.operators.spatial import (
    prepare_spatial_panel_weights,
)

from statadic_studies.adapters._random import rng
from statadic_studies.adapters.dependence._common import _inference, _record
from statadic_studies.adapters.dependence.spatial_durbin import BETA
from statadic_studies.adapters.dependence.spatial_extensions import (
    observed_fixture,
)


def varying_panel_fixture(effects):
    t, n = 6, 16
    generator = rng("spatial.period-varying-panel", "fixed-design", 0, seed=673)
    x = generator.normal(size=(t, n, 2))
    if effects != "fixed":
        x = np.concatenate([np.ones((t, n, 1)), x], axis=2)
    graphs = np.stack(
        [
            (0.55 + 0.04 * i) * np.roll(np.eye(n), 1, axis=1)
            + (0.45 - 0.04 * i) * np.roll(np.eye(n), -1, axis=1)
            for i in range(t)
        ]
    )
    return x, graphs


def varying_panel_data(generator, effects):
    x, graphs = varying_panel_fixture(effects)
    beta = np.array([-0.5, 0.8]) if effects == "fixed" else BETA
    alpha = (
        np.zeros(x.shape[1])
        if effects == "pooled"
        else generator.normal(size=x.shape[1])
    )
    y = np.stack(
        [
            np.linalg.solve(
                np.eye(len(w)) - 0.25 * w,
                xx @ beta + alpha + 0.5 * generator.normal(size=len(w)),
            )
            for xx, w in zip(x, graphs, strict=True)
        ]
    )
    return x, y, graphs


@cache
def _panel_kernel(effects):
    x, graphs = varying_panel_fixture(effects)
    weights = prepare_spatial_panel_weights(graphs, period_ids=np.arange(len(x)))

    @jax.jit
    def one(y):
        fit = PanelML().estimate(
            SpatialPanelProblem(
                jnp.asarray(x),
                y,
                weights,
                process=SAR(),
                slx_columns=(),
                effects="unit" if effects == "fixed" else effects,
            )
        )
        uncertainty = spatial_covariance(fit)
        return tuple(
            _inference(fit, uncertainty, position, truth, reference="normal")
            for position, truth in (
                (0 if effects == "fixed" else 1, -0.5),
                (fit.parameters.shape[0] - 1, 0.25),
            )
        )

    return one


def _panel(scenario, replicate, seed):
    _, y, _ = varying_panel_data(
        rng("spatial.period-varying-panel", scenario, replicate, seed=seed), scenario
    )
    results = jax.device_get(_panel_kernel(scenario)(jnp.asarray(y)))
    return {
        name: _record(values, truth)
        for name, values, truth in zip(
            ("beta", "rho"), results, (-0.5, 0.25), strict=True
        )
    }


def observed_remaining_data(generator, process):
    x, graphs, selections, _ = observed_fixture()
    outcomes = []
    for xx, w, selected in zip(x, graphs, selections, strict=True):
        error = 0.5 * generator.normal(size=len(w))
        full = (
            np.linalg.solve(
                np.eye(len(w)) - 0.25 * w, xx @ np.array([0.3, -0.5]) + error
            )
            if process == "SAR"
            else xx @ np.array([0.3, -0.5])
            + np.linalg.solve(np.eye(len(w)) - 0.3 * w, error)
        )
        outcomes.append(full[list(selected)])
    return tuple(outcomes)


@cache
def _observed_sem_kernel():
    x, _, _, periods = observed_fixture()

    @jax.jit
    def one(outcomes):
        fit = ObservedPanelML().estimate(
            ObservedSpatialPanelProblem(
                periods,
                tuple(jnp.asarray(xx) for xx in x),
                outcomes,
                12,
                process=SEM(),
                slx_columns=(),
                effects="pooled",
            )
        )
        uncertainty = spatial_covariance(fit, method="observed")
        return tuple(
            _inference(fit, uncertainty, position, truth, reference="normal")
            for position, truth in ((1, -0.5), (2, 0.3))
        )

    return one


def _observed_sem(scenario, replicate, seed):
    outcomes = observed_remaining_data(
        rng("spatial.observed-sem", scenario, replicate, seed=seed), "SEM"
    )
    results = jax.device_get(
        _observed_sem_kernel()(tuple(jnp.asarray(y) for y in outcomes))
    )
    return {
        name: _record(values, truth)
        for name, values, truth in zip(
            ("beta", "lambda"), results, (-0.5, 0.3), strict=True
        )
    }
