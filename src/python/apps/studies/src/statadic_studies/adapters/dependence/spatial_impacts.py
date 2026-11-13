"""Native spatial impacts studies."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from statadic.estimation.spatial import (
    ML,
    SAR,
    ObservedPanelML,
    ObservedSpatialPanelProblem,
    SpatialProblem,
)
from statadic.inference.spatial import (
    spatial_covariance,
    spatial_impacts,
    spatial_panel_impacts,
)
from statadic.operators.spatial import (
    prepare_spatial_weights,
)

from statadic_studies.adapters._random import rng
from statadic_studies.adapters.dependence.spatial_durbin import THETA, durbin_data
from statadic_studies.adapters.dependence.spatial_extensions import (
    observed_fixture,
    spatial_fixture,
)
from statadic_studies.adapters.dependence.spatial_panel import observed_remaining_data


def exact_impacts(w, beta, theta, rho):
    """Independent full derivative matrix; no row-standardized scalar shortcut."""
    derivative = np.linalg.solve(
        np.eye(len(w)) - rho * w, beta * np.eye(len(w)) + theta * w
    )
    direct, total = np.trace(derivative) / len(w), derivative.sum() / len(w)
    return np.array([direct, total - direct, total])


def _impact_record(estimate, se, truth, valid):
    if not valid or not np.isfinite([estimate, se]).all() or se < 0:
        return {"valid": False, "failure": "impact-or-delta-uncertainty-unavailable"}
    radius = 1.959963984540054 * float(se)
    return {
        "valid": True,
        "estimate": float(estimate),
        "truth": float(truth),
        "covered": bool(estimate - radius <= truth <= estimate + radius),
        "width": 2 * radius,
        "standard_error": float(se),
    }


@cache
def _impact_kernel(durbin):
    x, w = spatial_fixture()
    graph = prepare_spatial_weights(w)

    @jax.jit
    def one(y):
        fit = ML().estimate(
            SpatialProblem(
                jnp.asarray(x),
                y,
                graph,
                process=SAR(),
                slx_columns=(1, 2) if durbin else (),
            )
        )
        impact = spatial_impacts(spatial_covariance(fit))
        return (
            impact.values[1],
            impact.standard_errors[1],
            impact.valid & impact.uncertainty_valid,
        )

    return one


def _impacts(scenario, replicate, seed):
    durbin = scenario == "Durbin"
    x, y, w = durbin_data(
        rng("spatial.cross-section-impacts", scenario, replicate, seed=seed),
        "SDM",
        THETA if durbin else np.zeros(2),
    )
    del x
    values, errors, valid = jax.device_get(_impact_kernel(durbin)(jnp.asarray(y)))
    truths = exact_impacts(w, -0.5, 0.2 if durbin else 0.0, 0.25)
    return {
        name: _impact_record(value, error, truth, valid)
        for name, value, error, truth in zip(
            ("direct", "indirect", "total"), values, errors, truths, strict=True
        )
    }


@cache
def _panel_impact_kernel():
    x, _, _, periods = observed_fixture()

    @jax.jit
    def one(outcomes):
        fit = ObservedPanelML().estimate(
            ObservedSpatialPanelProblem(
                periods,
                tuple(jnp.asarray(xx) for xx in x),
                outcomes,
                12,
                process=SAR(),
                slx_columns=(),
                effects="pooled",
            )
        )
        impacts = spatial_panel_impacts(spatial_covariance(fit, method="observed"))
        # Report actual first active-graph-period derivatives, not mean SEs
        # incorrectly substituted for uncertainty of an average derivative.
        return (
            impacts.values[0, 1],
            impacts.standard_errors[0, 1],
            impacts.valid & impacts.uncertainty_valid,
        )

    return one


def _panel_impacts(scenario, replicate, seed):
    outcomes = observed_remaining_data(
        rng("spatial.observed-panel-impacts", scenario, replicate, seed=seed), "SAR"
    )
    values, errors, valid = jax.device_get(
        _panel_impact_kernel()(tuple(jnp.asarray(y) for y in outcomes))
    )
    _, graphs, _, _ = observed_fixture()
    truths = exact_impacts(graphs[0], -0.5, 0.0, 0.25)
    return {
        name: _impact_record(value, error, truth, valid)
        for name, value, error, truth in zip(
            ("direct", "indirect", "total"), values, errors, truths, strict=True
        )
    }
