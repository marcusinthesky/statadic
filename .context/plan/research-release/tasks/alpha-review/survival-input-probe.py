"""Compare covariate authority at independent and zero-lag survival preparation."""

import json

import jax
import numpy as np

from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.graphs import PanelData, SpatialLag
from statadic.api.panel import PanelLayout
from statadic.api.survival import RightCensored, SurvivalRegression
from statadic.estimation.survival import SpatialSurvivalML
from statadic.model.survival import LogNormalAFT
from statadic.operators.spatial import prepare_spatial_weights
from statadic.optimize.integration import IntegrationDesign


def run():
    ids = np.tile(np.arange(3), 2)
    times = np.repeat(np.arange(2), 3)
    x = np.arange(6.0)
    order = np.array([5, 0, 3, 1, 4, 2])
    nodes = Observations({"unit": ids, "year": times, "x": x})
    outcomes = Observations(
        {
            "unit": ids[order],
            "year": times[order],
            "x": x[order] + 100,
            "duration": np.exp(x[order] / 10),
            "event": np.ones(6),
        },
        row_ids=order.tolist(),
    )
    layout = PanelLayout("unit", "year")
    weights = prepare_spatial_weights(
        np.roll(np.eye(3), 1, axis=1), unit_ids=np.arange(3)
    )
    data = PanelData(nodes, layout, {"neighbors": weights}, outcomes)
    response = RightCensored("duration", "event")
    terms = Intercept() + Linear("x")
    independent = SurvivalRegression(
        response, terms, panel=layout, family=LogNormalAFT()
    ).prepare(data)
    streams = tuple(
        IntegrationDesign.from_key(jax.random.key(t), replicates=2, draws=8)
        for t in range(2)
    )
    spatial = SurvivalRegression(
        response,
        terms,
        panel=layout,
        family=LogNormalAFT(),
        dependence=SpatialLag("neighbors"),
        estimator=SpatialSurvivalML(rho_fixed=0.0),
    ).prepare(data, integrations=streams)
    return {
        "independent_design": independent.values.tolist(),
        "spatial_node_designs": [value.tolist() for value in spatial.panel.values],
        "limit": "Preparation only; same keys and deliberately conflicting duplicated covariate, no fitting.",
    }


if __name__ == "__main__":
    with jax.enable_x64(new_val=True):
        print(json.dumps(run(), indent=2))
