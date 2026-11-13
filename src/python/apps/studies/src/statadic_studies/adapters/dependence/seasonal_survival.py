"""Recovery of an independently generated periodic spatial log-time contrast."""

from __future__ import annotations

from statistics import NormalDist

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
from statadic.api.data import Observations, PanelLayout, SpatialPanelData
from statadic.api.design import Linear
from statadic.api.inference import ModelBased
from statadic.api.survival import RightCensored, SpatialSurvivalData
from statadic.estimation.survival import SpatialSurvivalML
from statadic.operators.spatial import prepare_spatial_weights
from statadic.optimize.integration import IntegrationDesign

from statadic_studies.adapters._random import rng
from statadic_studies.adapters.dependence.spatial_survival import _key
from statadic_studies.definitions.dependence.composition import SEASONAL


def _observations(columns):
    # Select the campaign's declared JAX precision at the host boundary.
    return Observations({name: jnp.asarray(values) for name, values in columns.items()})


def _covariates(phase):
    # The SciPy reference is parameterized by values at the four knots.
    # Knot zero supplies the baseline, absorbed by the model's intercept.
    eye = np.eye(4)
    interpolator = CubicSpline(
        np.linspace(0, 1, 5), np.vstack((eye, eye[0])), bc_type="periodic"
    )
    basis = interpolator(phase)
    return {"phase": phase, **{f"k{i}": basis[:, i] for i in range(1, 4)}}


def _data(scenario, replicate, seed, draws):
    if scenario not in SEASONAL.scenarios:
        raise ValueError(f"Unknown seasonal survival scenario: {scenario}")
    generator = rng(SEASONAL.id, scenario, replicate, seed=seed)
    n, periods = 8, 3
    weights = 0.7 * np.roll(np.eye(n), 1, axis=1) + 0.3 * np.roll(np.eye(n), -2, axis=1)
    graph = prepare_spatial_weights(weights, unit_ids=np.arange(n))
    phase = generator.uniform(size=n * periods)
    columns = _covariates(phase)
    seasonal = 0.3 * (columns["k1"] - columns["k3"]).reshape(periods, n)
    mean = (
        0.2
        + seasonal
        + np.linspace(-0.2, 0.2, n)
        + np.linspace(-0.15, 0.15, periods)[:, None]
    )
    logs = np.linalg.solve(
        np.eye(n) - 0.25 * weights,
        (mean + 0.4 * generator.normal(size=(periods, n))).T,
    ).T
    event_time = np.exp(logs)
    caps = np.full(n, np.inf)
    if scenario == "administrative":
        caps[0] = 1.0
    event = event_time <= caps
    keys = {
        "unit": np.tile(np.arange(n), periods),
        "period": np.repeat(np.arange(periods), n),
    }
    data = SpatialPanelData(
        _observations({**columns, **keys}),
        _observations(
            {
                **keys,
                "duration": np.minimum(caps, event_time).ravel(),
                "event": event.ravel().astype(int),
            }
        ),
    )
    graphs = tuple((t, graph) for t in range(periods))
    designs = tuple(
        IntegrationDesign.from_key(
            _key(SEASONAL.id, scenario, replicate, seed, f"integration-{t}"),
            replicates=2,
            draws=max(2, draws + draws % 2),
        )
        for t in range(periods)
    )
    return data, graphs, designs, (~event).sum(axis=1).tolist()


def _wald(result, censored):
    fit = result.core
    evidence = {
        "fit_valid": bool(fit.valid),
        "integration_valid": bool(fit.integration_valid),
        "identified": bool(fit.identified),
        "information_available": bool(fit.information_available),
        "solver_status": int(fit.solver.status),
        "censored_per_period": censored,
    }
    if not fit.valid or not fit.integration_valid:
        return {"valid": False, "failure": "fit-unavailable", **evidence}
    evidence.update(
        {"rho_hat": float(fit.rho), "log_likelihood": float(fit.log_likelihood)}
    )
    design = result.panel.design
    values = design.transform(_observations(_covariates(np.array([0.25, 0.75]))))
    contrast = np.zeros(len(result.parameters))
    contrast[: len(design.column_names)] = values[0] - values[1]
    estimate = float(contrast @ np.asarray(result.parameters))
    info = result.infer(covariance=ModelBased()).uncertainty
    variance = float(contrast @ np.asarray(info.covariance.values) @ contrast)
    if not info.valid or not np.isfinite(variance) or variance <= 0:
        return {
            "valid": False,
            "failure": "contrast-variance-unavailable",
            "point": estimate,
            **evidence,
        }
    half_width = NormalDist().inv_cdf(0.975) * np.sqrt(variance)
    return {
        "valid": True,
        "estimate": estimate,
        "truth": 0.6,
        "covered": bool(abs(estimate - 0.6) <= half_width),
        "width": float(2 * half_width),
        **evidence,
    }


def seasonal_trial(scenario, replicate, seed, draws):
    """Fit both public interfaces to the same simulated panel."""
    from statadic.api.design import CyclicSpline
    from statadic.api.regression import Regression
    from statadic.api.spatial.graphs import PanelData, SpatialLag
    from statadic.model.survival import LogNormalAFT

    data, graphs, designs, censored = _data(scenario, replicate, seed, draws)
    layout = PanelLayout("unit", "period")
    common = dict(
        response=RightCensored("duration", "event"),
        structure=layout,
        effects="two-way",
        estimator=SpatialSurvivalML(),
    )
    composed = Regression(
        **common,
        law=LogNormalAFT(),
        dependence=SpatialLag("neighbors"),
        terms=CyclicSpline("phase", df=3, period=1),
    )
    explicit = Regression(
        **common,
        law=LogNormalAFT(),
        dependence=SpatialLag("neighbors"),
        terms=Linear("k1") + Linear("k2") + Linear("k3"),
    )
    panel = PanelData(
        nodes=data.nodes,
        outcomes=data.outcomes,
        panel=layout,
        graphs={"neighbors": graphs},
    )
    return {
        "compositional-Wald": _wald(
            composed.estimate(SpatialSurvivalData(panel, designs)), censored
        ),
        "explicit-basis-Wald": _wald(
            explicit.estimate(SpatialSurvivalData(panel, designs)), censored
        ),
    }
