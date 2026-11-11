"""Compose a repeated-spell survival model and fixed spatial-lag examples."""

# ruff: noqa: INP001, T201  # Standalone executable documentation.
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.inference import ModelBased
from statadic.api.data import PanelLayout
from statadic.api.regression import Regression
from statadic.api.spatial.graphs import PanelData, SpatialLag
from statadic.api.survival import RightCensored, SpatialSurvivalData
from statadic.estimation.survival import SpatialSurvivalML, SurvivalML
from statadic.model.survival import LogNormalAFT
from statadic.operators.spatial import prepare_spatial_weights
from statadic.optimize.integration import IntegrationDesign

if TYPE_CHECKING:
    from statadic.api.inference import InferenceResult
    from statadic.api.spatial.prediction import SpatialPanelPrediction
    from statadic.api.survival import SpatialSurvivalResult, SurvivalResult


def run() -> tuple[  # noqa: PLR0914 -- one fixed dataset and three matched model specifications
    SurvivalResult,
    SurvivalResult,
    dict[float, tuple[SpatialSurvivalResult, InferenceResult, SpatialPanelPrediction]],
]:
    """Fit pooled, time-effect and fixed-rho spatial models to one spell panel."""
    rng = np.random.default_rng(241)
    n, periods_count = 6, 3
    # The application chooses source precision before Observations validates it.
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    x = rng.normal(size=(periods_count, n)).astype(dtype)
    times = np.exp(0.4 + 0.3 * x + 0.5 * rng.normal(size=(periods_count, n)))
    # Administrative censoring is specified before observing the durations.
    limits = np.full_like(times, np.inf)
    limits[0, 0] = 1.0
    observed = np.minimum(times, limits).astype(dtype)
    events = (times <= limits).astype(np.int32)
    units = np.tile(np.arange(n, dtype=np.int32), periods_count)
    periods = np.repeat(np.arange(periods_count, dtype=np.int32), n)
    nodes = Observations({"x": x.ravel(), "unit": units, "period": periods})
    spells = Observations(
        {
            "time": observed.ravel(),
            "event": events.ravel(),
            "unit": units,
            "period": periods,
            "x": x.ravel(),
        }
    )
    response = RightCensored("time", "event")
    terms = Intercept() + Linear("x")
    layout = PanelLayout("unit", "period")
    estimator = SurvivalML(family=LogNormalAFT())
    pooled = Regression(
        response, terms, law=LogNormalAFT(), estimator=estimator
    ).estimate(spells)
    panel = Regression(
        response,
        Linear("x"),
        law=LogNormalAFT(),
        estimator=estimator,
        structure=layout,
        effects="time",
    ).estimate(spells)
    for label, fit in (("pooled", pooled), ("panel time effects", panel)):
        if not fit.valid:
            message = f"{label} fit unavailable: {fit.report()}"
            raise RuntimeError(message)
    weights = (np.ones((n, n)) - np.eye(n)) / (n - 1)
    graphs = tuple(
        (t, prepare_spatial_weights(weights, unit_ids=np.arange(n)))
        for t in range(periods_count)
    )
    integrations = tuple(
        IntegrationDesign.from_key(jax.random.key(t), replicates=2, draws=16)
        for t in range(periods_count)
    )
    data = PanelData(nodes, layout, {"neighbors": graphs}, spells)
    spatial = {}
    for rho in (0.0, 0.2):
        model = Regression(
            response,
            Linear("x"),
            structure=layout,
            law=LogNormalAFT(),
            dependence=SpatialLag("neighbors"),
            estimator=SpatialSurvivalML(rho_fixed=rho),
            effects="time",
        )
        fit = model.estimate(SpatialSurvivalData(data, integrations))
        if not fit.valid:
            message = f"Spatial fit unavailable at fixed rho={rho}: {fit.report()}"
            raise RuntimeError(message)
        inference = fit.infer(covariance=ModelBased())
        prediction = fit.predict(data, np.array(1.0), kind="survival")
        spatial[rho] = (fit, inference, prediction)
    return pooled, panel, spatial


def main() -> None:
    """Print retained fits, covariance availability and graph-ordered predictions."""
    pooled, panel, spatial = run()
    for label, fit in (("pooled", pooled), ("panel time effects", panel)):
        print(label, fit.report())
    for rho, (fit, inference, prediction) in spatial.items():
        print(f"spatial, rho fixed at {rho}", fit.report())
        print("model-based covariance available:", bool(inference.uncertainty.valid))
        print("model-based inference:", inference.report())
        print("period-0 node survival at time 1:", np.asarray(prediction.values[0]))


if __name__ == "__main__":
    main()
