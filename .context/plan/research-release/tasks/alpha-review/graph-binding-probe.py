"""Isolate a second graph defect after simulating only the known import migration."""

import json

import jax
import numpy as np

import statadic.api.composition as composition
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.graph_learning import GraphML, LearnBandwidth
from statadic.api.graph_recipes import BandwidthRef, DistanceGraph, RBFWeights
from statadic.api.graphs import PanelData, SpatialX
from statadic.api.panel import PanelLayout


def run():
    parameter = BandwidthRef("smooth")
    ids = tuple(range(7))
    layout = PanelLayout("unit", "year")
    costs = np.abs(np.subtract.outer(np.arange(7), np.arange(7)))
    graph = DistanceGraph.from_distances(
        costs, unit_ids=ids, weighting=RBFWeights(parameter)
    )
    x = np.arange(14.0)
    rows = Observations(
        {
            "x": x,
            "y": 1 + x + np.sin(x),
            "unit": np.tile(ids, 2),
            "year": np.repeat([0, 1], 7),
        }
    )
    data = PanelData(rows, layout, {"information": graph})
    procedure = GraphML(bandwidths={parameter: LearnBandwidth(1.0)})
    model = composition.Equation(
        "y",
        Intercept() + Linear("x") + SpatialX("x", "information"),
        estimator=procedure,
    )
    unbound = procedure._initial(model.prepare(data))
    # This temporary process-local name allows inspection past F015. It is not
    # an edit, supported alias or evidence that the candidate module imports.
    composition.Regression = composition.Equation
    try:
        from statadic.api.graph_selection import _bind

        bound = _bind(data, parameter, 1.3)
        try:
            procedure._initial(model.prepare(bound))
            observed = "accepted"
        except ValueError as error:
            observed = str(error)
    finally:
        del composition.Regression
    return {
        "unbound_initial": {key: value.tolist() for key, value in unbound.items()},
        "fixed_candidate_with_retained_binding": observed,
        "limits": "Conditional on completing F015; only a process-local import name was simulated, no source or production API changed.",
    }


if __name__ == "__main__":
    with jax.enable_x64(new_val=True):
        print(json.dumps(run(), indent=2))
