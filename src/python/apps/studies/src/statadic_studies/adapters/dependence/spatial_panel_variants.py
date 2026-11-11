"""Actual balanced and changing-roster spatial ML, with independent full DGPs."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.estimation.spatial import (
    SAC,
    SAR,
    SEM,
    ObservedPanelML,
    ObservedSpatialPanelProblem,
    PanelML,
    SpatialPanelProblem,
)
from statadic.inference.spatial import spatial_covariance
from statadic.operators.spatial import (
    ObservedSpatialPeriod,
    prepare_spatial_panel_weights,
    prepare_spatial_weights,
)

from statadic_studies.adapters._random import rng
from statadic_studies.adapters.dependence._common import _inference, _record
from statadic_studies.definitions.dependence.spatial_panel_variants import (
    PANEL_VARIANT_STUDIES,
)


def variant_options(study, scenario):
    if (
        study not in PANEL_VARIANT_STUDIES
        or scenario not in PANEL_VARIANT_STUDIES[study].scenarios
    ):
        raise ValueError("Unknown spatial panel variant")
    effect, _, convention = scenario.rpartition("-")
    return (
        "-observed-" in study,
        study.rsplit("-", 1)[-1].upper(),
        effect,
        convention == "durbin",
    )


@cache
def variant_fixture(study, scenario):
    """Freeze exogenous inputs, identities and observation selection before outcomes."""
    observed, process, effect, durbin = variant_options(study, scenario)
    generator = rng(study, "fixed-design", 0, seed=9521)
    n_units, n_periods = 24, 8
    designs, graphs, rosters, selected = [], [], [], []
    intercept = effect in ("pooled", "random")
    for t in range(n_periods):
        ids = tuple(i for i in range(n_units) if not observed or (i + 2 * t) % 11 != 0)
        n = len(ids)
        x = generator.normal(size=(n, 2))
        if intercept:
            x = np.column_stack((np.ones(n), x))
        raw = generator.uniform(0.05, 1, (n, n))
        np.fill_diagonal(raw, 0)
        w = raw / raw.sum(axis=1, keepdims=True)
        keep = tuple(
            i for i, unit in enumerate(ids) if not observed or (unit + t) % 5 != 0
        )
        designs.append(x)
        graphs.append(w)
        rosters.append(ids)
        selected.append(keep)
    return tuple(designs), tuple(graphs), tuple(rosters), tuple(selected)


def variant_data(study, scenario, generator):
    """Simulate the full graph law before selecting any absent outcomes."""
    observed, process, effect, durbin = variant_options(study, scenario)
    designs, graphs, rosters, selected = variant_fixture(study, scenario)
    alpha = (
        generator.normal(scale=0.5, size=24)
        if effect in ("unit", "two-way", "random")
        else np.zeros(24)
    )
    delta = np.linspace(-0.2, 0.2, 8) if effect in ("time", "two-way") else np.zeros(8)
    slope = 1 if effect in ("pooled", "random") else 0
    beta = np.array([0.3, -0.5, 0.8]) if slope else np.array([-0.5, 0.8])
    outcomes = []
    for t, (x, w, ids, keep) in enumerate(
        zip(designs, graphs, rosters, selected, strict=True)
    ):
        mean = x @ beta + alpha[list(ids)] + delta[t]
        if durbin:
            mean = mean + 0.2 * (w @ x[:, slope])
        error = 0.5 * generator.normal(size=len(w))
        if process != "SAR":
            error = np.linalg.solve(np.eye(len(w)) + 0.15 * w, error)
        outcome = mean + error
        if process != "SEM":
            outcome = np.linalg.solve(np.eye(len(w)) - 0.2 * w, outcome)
        outcomes.append(outcome[list(keep)] if observed else outcome)
    return tuple(outcomes)


@cache
def _variant_kernel(study, scenario):
    observed, name, effect, durbin = variant_options(study, scenario)
    designs, graphs, rosters, selected = variant_fixture(study, scenario)
    slope = 1 if effect in ("pooled", "random") else 0
    process = {"SAR": SAR, "SEM": SEM, "SAC": SAC}[name]()
    slx_columns = (slope,) if durbin else ()
    width = designs[0].shape[1] + int(durbin)
    targets = [(slope, -0.5)]
    if name != "SEM":
        targets.append((width, 0.2))
    if name != "SAR":
        targets.append((width + int(name == "SAC"), -0.15))
    if observed:
        periods = tuple(
            ObservedSpatialPeriod(prepare_spatial_weights(w), ids, keep)
            for w, ids, keep in zip(graphs, rosters, selected, strict=True)
        )
        estimator = ObservedPanelML()
    else:
        weights = prepare_spatial_panel_weights(
            np.stack(graphs), period_ids=np.arange(len(graphs))
        )
        estimator = PanelML()

    @jax.jit
    def one(outcomes):
        problem = (
            ObservedSpatialPanelProblem(
                periods,
                tuple(jnp.asarray(x) for x in designs),
                outcomes,
                24,
                process=process,
                effects=effect,
                slx_columns=slx_columns,
            )
            if observed
            else SpatialPanelProblem(
                jnp.asarray(np.stack(designs)),
                jnp.stack(outcomes),
                weights,
                process=process,
                effects=effect,
                slx_columns=slx_columns,
            )
        )
        fit = estimator.estimate(problem)
        uncertainty = (
            spatial_covariance(fit, method="observed")
            if observed
            else spatial_covariance(fit)
        )
        return tuple(
            _inference(fit, uncertainty, position, truth, reference="normal")
            for position, truth in targets
        )

    return one, tuple(truth for _, truth in targets)


def run_panel_variant(study, scenario, replicate, seed):
    outcomes = variant_data(study, scenario, rng(study, scenario, replicate, seed=seed))
    kernel, truths = _variant_kernel(study, scenario)
    results = jax.device_get(kernel(tuple(jnp.asarray(y) for y in outcomes)))
    return {
        name: _record(value, truth)
        for name, value, truth in zip(
            PANEL_VARIANT_STUDIES[study].methods, results, truths, strict=True
        )
    }
