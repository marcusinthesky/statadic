"""Native spatial IV/GMM, component models and selected-observation likelihood."""

from dataclasses import replace
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.estimation.spatial import (
    KKP,
    ML,
    SAC,
    SAR,
    SEM,
    ObservedPanelML,
    ObservedSpatialPanelProblem,
    SpatialGMM,
    SpatialIV,
    SpatialPanelProblem,
    SpatialProblem,
)
from statadic.inference.linear_covariance import classical_covariance
from statadic.inference.spatial import spatial_covariance
from statadic.operators.spatial import (
    ObservedSpatialPeriod,
    prepare_spatial_panel_weights,
    prepare_spatial_weights,
)

from statadic_studies.adapters._random import rng
from statadic_studies.adapters.dependence._common import _inference, _record


def spatial_fixture(n=48):
    """Fixed directional graph, aligned with every generated outcome row."""
    w = 0.6 * np.roll(np.eye(n), 1, axis=1) + 0.4 * np.roll(np.eye(n), -1, axis=1)
    generator = rng("spatial.extensions", f"fixed-n{n}", 0, seed=126)
    x = np.column_stack((np.ones(n), generator.normal(size=(n, 2))))
    return x, w


def spatial_outcome(generator, x, w, *, rho=0.0, lam=0.0, heteroskedastic=False):
    scale = 0.3 + 0.2 * np.abs(x[:, 1]) if heteroskedastic else 0.5
    shock = generator.normal(size=len(x)) * scale
    error = np.linalg.solve(np.eye(len(x)) - lam * w, shock)
    return np.linalg.solve(
        np.eye(len(x)) - rho * w, x @ np.array([0.3, -0.5, 0.8]) + error
    )


@cache
def _iv_kernel():
    x, w = spatial_fixture()
    graph = prepare_spatial_weights(w)

    def one(y):
        fit = SpatialIV(instrument_columns=(1, 2)).estimate(
            SpatialProblem(jnp.asarray(x), y, graph, process=SAR(), slx_columns=())
        )
        return tuple(
            _inference(
                fit, spatial_covariance(fit, method=kind), 3, 0.25, reference="normal"
            )
            for kind in ("homoskedastic", "heteroskedastic")
        )

    return jax.jit(one)


def _iv(scenario, replicate, seed):
    x, w = spatial_fixture()
    generator = rng("spatial.iv-inference", scenario, replicate, seed=seed)
    y = spatial_outcome(
        generator, x, w, rho=0.25, heteroskedastic=scenario == "heteroskedastic"
    )
    outputs = jax.device_get(_iv_kernel()(jnp.asarray(y)))
    return {
        method: _record(value, 0.25)
        for method, value in zip(
            ("2SLS-homoskedastic", "2SLS-heteroskedastic"), outputs, strict=True
        )
    }


@cache
def _gmm_kernel(lag):
    x, w = spatial_fixture()
    graph = prepare_spatial_weights(w)

    def one(y):
        outputs = []
        for variance in ("homoskedastic", "heteroskedastic"):
            fit = SpatialGMM(
                variance_model=variance, instrument_columns=(1, 2)
            ).estimate(
                SpatialProblem(
                    jnp.asarray(x),
                    y,
                    graph,
                    process=SAC() if lag else SEM(),
                    slx_columns=(),
                )
            )
            outputs.append(
                _inference(
                    fit,
                    spatial_covariance(fit),
                    fit.parameters.shape[0] - 1,
                    0.3,
                    reference="normal",
                )
            )
        return tuple(outputs)

    return jax.jit(one)


def _gmm(scenario, replicate, seed):
    model, variance = scenario.split("-")
    lag = model == "combo"
    x, w = spatial_fixture()
    generator = rng("spatial.error-gmm", scenario, replicate, seed=seed)
    y = spatial_outcome(
        generator,
        x,
        w,
        rho=0.25 if lag else 0.0,
        lam=0.3,
        heteroskedastic=variance == "heteroskedastic",
    )
    outputs = jax.device_get(_gmm_kernel(lag)(jnp.asarray(y)))
    return {
        method: _record(value, 0.3)
        for method, value in zip(
            ("homoskedastic-moments", "heteroskedastic-moments"), outputs, strict=True
        )
    }


@cache
def _ml_kernel(combo):
    x, w = spatial_fixture()
    graph = prepare_spatial_weights(w)

    def one(y):
        fit = ML().estimate(
            SpatialProblem(
                jnp.asarray(x),
                y,
                graph,
                process=SAC() if combo else SEM(),
                slx_columns=(),
            )
        )
        return _inference(
            fit,
            spatial_covariance(fit),
            fit.parameters.shape[0] - 1,
            0.3,
            reference="normal",
        )

    return jax.jit(one)


def _ml(scenario, replicate, seed):
    combo = scenario == "SAC"
    x, w = spatial_fixture()
    generator = rng("spatial.error-ml", scenario, replicate, seed=seed)
    y = spatial_outcome(generator, x, w, rho=0.25 if combo else 0.0, lam=0.3)
    return {
        "Gaussian-ML-information": _record(
            jax.device_get(_ml_kernel(combo)(jnp.asarray(y))), 0.3
        )
    }


def kkp_fixture():
    periods, units = 6, 16
    _, w = spatial_fixture(units)
    generator = rng("spatial.kkp-components", "fixed-design", 0, seed=126)
    x = np.concatenate(
        (np.ones((periods, units, 1)), generator.normal(size=(periods, units, 2))),
        axis=2,
    )
    return x, w


def kkp_data(generator):
    """Filter the entity effect AND innovation, unlike marginal spatial panels."""
    x, w = kkp_fixture()
    raw = generator.normal(size=x.shape[1]) + 0.5 * generator.normal(size=x.shape[:2])
    y = (
        x @ np.array([0.3, -0.5, 0.8])
        + np.linalg.solve(np.eye(len(w)) - 0.3 * w, raw.T).T
    )
    return x, y, w


@cache
def _kkp_kernel():
    x, w = kkp_fixture()
    graph = prepare_spatial_panel_weights(w, period_ids=np.arange(len(x)))

    def one(y):
        outputs = []
        for full in (False, True):
            fit = KKP(full_weights=full).estimate(
                SpatialPanelProblem(
                    jnp.asarray(x),
                    y,
                    graph,
                    process=SEM(),
                    effects="random",
                    slx_columns=(),
                )
            )
            beta = _inference(fit.linear, classical_covariance(fit.linear), 1, -0.5)
            outputs.append((beta, fit.spatial_lambda, fit.valid))
        return tuple(outputs)

    return jax.jit(one)


def _kkp(scenario, replicate, seed):
    generator = rng("spatial.kkp-components", scenario, replicate, seed=seed)
    _, y, _ = kkp_data(generator)
    outputs = jax.device_get(_kkp_kernel()(jnp.asarray(y)))
    records = {}
    for name, (beta, lam, valid) in zip(
        ("identity-moments", "full-weights"), outputs, strict=True
    ):
        records[name + "-beta"] = _record(beta, -0.5)
        if not bool(valid):
            records[name + "-beta"] = {
                "valid": False,
                "failure": "component-fit-unavailable",
            }
        records[name + "-lambda"] = (
            {"valid": True, "estimate": float(lam), "truth": 0.3}
            if bool(valid) and np.isfinite(lam)
            else {"valid": False, "failure": "component-fit-unavailable"}
        )
    return records


def observed_fixture():
    """Full graph and full covariates survive outcome selection in each period."""
    n, periods = 12, 4
    generator = rng("spatial.observed-panel", "fixed-design", 0, seed=603)
    x = tuple(
        np.column_stack((np.ones(n), generator.normal(size=n))) for _ in range(periods)
    )
    graphs = []
    selections = []
    declarations = []
    for t in range(periods):
        w = generator.uniform(size=(n, n))
        np.fill_diagonal(w, 0)
        w /= w.sum(axis=1, keepdims=True)
        selected = tuple(i for i in range(n) if i not in (t, t + 4))
        declarations.append(
            ObservedSpatialPeriod(prepare_spatial_weights(w), tuple(range(n)), selected)
        )
        graphs.append(w)
        selections.append(selected)
    return x, tuple(graphs), tuple(selections), tuple(declarations)


def observed_data(generator):
    x, graphs, selections, declarations = observed_fixture()
    outcomes = tuple(
        np.linalg.solve(
            np.eye(len(w)) - 0.25 * w,
            xx @ np.array([0.3, -0.5]) + 0.5 * generator.normal(size=len(w)),
        )[list(selected)]
        for xx, w, selected in zip(x, graphs, selections, strict=True)
    )
    return ObservedSpatialPanelProblem(
        declarations,
        tuple(jnp.asarray(xx) for xx in x),
        tuple(jnp.asarray(y) for y in outcomes),
        12,
        process=SAR(),
        effects="pooled",
    )


@cache
def _observed_kernel():
    x, _, _, periods = observed_fixture()

    def one(outcomes):
        problem = ObservedSpatialPanelProblem(
            periods,
            tuple(jnp.asarray(xx) for xx in x),
            outcomes,
            12,
            process=SAR(),
            effects="pooled",
        )
        fit = ObservedPanelML().estimate(
            replace(problem, process=SAR(), slx_columns=(), effects="pooled")
        )
        return _inference(
            fit, spatial_covariance(fit, method="observed"), 2, 0.25, reference="normal"
        )

    return jax.jit(one)


def _observed(scenario, replicate, seed):
    generator = rng("spatial.observed-panel", scenario, replicate, seed=seed)
    problem = observed_data(generator)
    return {
        "selected-Gaussian-ML": _record(
            jax.device_get(_observed_kernel()(problem.outcome)), 0.25
        )
    }


def run_spatial_extensions(study, scenario, replicate, seed):
    adapters = {
        "spatial.iv-inference": _iv,
        "spatial.error-gmm": _gmm,
        "spatial.error-ml": _ml,
        "spatial.kkp-components": _kkp,
        "spatial.observed-panel": _observed,
    }
    if study not in adapters:
        raise ValueError(f"No spatial extension adapter for {study}")
    return adapters[study](scenario, replicate, seed)
