"""Reduced-form graph predictors bound to their serialized unit and period axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from statadic.api.data import Observations, PanelLayout
from statadic.api.panel import prepare_panel_effect_design
from statadic.api.spatial.graph_regression import _align, _arrays, _GraphDesign
from statadic.api.spatial.graphs import DistanceGraph, PanelData, SpatialData
from statadic.api.spatial.prediction import SpatialPanelPrediction
from statadic.estimation.spatial.graph_likelihood import _graph_mean
from statadic.operators.spatial.weights import prepare_spatial_weights

from ._design import decode_design, decode_design_record, encode_design
from ._store import (
    Arrays,
    ArtifactError,
    items,
    labels,
    optional_text,
    read_bundle,
    record,
    text,
    write_bundle,
)

if TYPE_CHECKING:
    from pathlib import Path

    from statadic.api.design import FittedDesign
    from statadic.api.spatial.graph_regression import GraphRegressionResult
    from statadic.core.typing import Array
    from statadic.operators.spatial.weights import SpatialWeights

_FAMILY = "frozen-graph-regression"
_CAP = "predict-reduced-form"
_WX_FIELDS = 2


@dataclass(frozen=True)
class FrozenGraphPredictor:
    """Reduced-form predictions on the fitted graph population with new covariates."""

    design: _GraphDesign
    parameters: Array
    unit: str | None
    periods: tuple[int, ...]
    weights: tuple[tuple[SpatialWeights, ...], ...]

    def predict(self, observations: Observations) -> Array | SpatialPanelPrediction:
        """Align new covariates to saved graph identities and solve each SAR mean."""
        names = self.design.names
        if self.design.effects is None:
            data = SpatialData(
                observations, dict(zip(names, self.weights[0], strict=True)), self.unit
            )
        else:
            graphs = {
                name: tuple(
                    (t, w[i]) for t, w in zip(self.periods, self.weights, strict=True)
                )
                for i, name in enumerate(names)
            }
            data = PanelData(observations, self.design.effects.layout, graphs)
        rows = _align(data, self.design)
        lag, _, designs = _arrays(self.design, rows, None)
        width = designs[0].shape[1]
        rho = self.parameters[width : width + len(self.design.lag)]
        means = tuple(
            _graph_mean(w, rho, x, self.parameters[:width])
            for w, x in zip(lag, designs, strict=True)
        )
        return (
            means[0]
            if isinstance(data, SpatialData)
            else SpatialPanelPrediction(rows.periods, rows.rosters, means)
        )


def save_graph_predictor(path: Path, result: GraphRegressionResult) -> None:
    """Freeze graph realizations, roles, fitted transformations and coefficients."""
    if not bool(result.core.valid):
        msg = "A frozen predictor requires a valid fitted graph regression."
        raise ArtifactError(msg)
    p = result.prepared
    s = p.schema
    arrays = Arrays()
    blocks = []
    for t, ids, graph in zip(
        p.rows.periods, p.rows.rosters, p.rows.graphs, strict=True
    ):
        weights = {
            name: arrays.add(
                src.weights(result.graph_parameters)
                if isinstance(src, DistanceGraph)
                else src.values
            )
            for name, src in graph.items()
        }
        blocks.append({"period": t, "units": list(ids), "weights": weights})
    panel = None
    if s.effects is not None:
        e = s.effects
        mode = {
            (False, False): "pooled",
            (True, False): "unit",
            (False, True): "time",
            (True, True): "two-way",
        }[bool(e.entity_levels), bool(e.time_levels)]
        panel = {"entity": e.layout.entity, "time": e.layout.time, "effects": mode}
    state = {
        "design": encode_design(s.base, arrays),
        "panel": panel,
        "unit": p.data.unit if isinstance(p.data, SpatialData) else None,
        "lag": list(s.lag),
        "error": list(s.error),
        "wx": [[name, s.base.column_names[i]] for name, i in s.wx],
        "parameter_names": list(s.parameter_names),
        "coefficients": arrays.add(result.parameters),
        "blocks": blocks,
    }
    write_bundle(
        path,
        purpose="prediction",
        family=_FAMILY,
        capabilities=(_CAP,),
        state=state,
        arrays=arrays,
        metadata={},
    )


def _names(value: object) -> tuple[str, ...]:
    return tuple(text(n, "name") for n in labels(value, "names"))


def _decode_wx(raw: object, base: FittedDesign) -> tuple[tuple[str, int], ...]:
    columns = base.column_names
    wx = []
    for pair in items(raw, "spatial covariates"):
        fields = items(pair, "spatial covariate")
        if (
            len(fields) != _WX_FIELDS
            or fields[1] not in columns
            or columns.index(text(fields[1], "column")) in base.intercept_columns
        ):
            msg = "Invalid graph covariate column."
            raise ArtifactError(msg)
        wx.append(
            (
                text(fields[0], "graph"),
                columns.index(text(fields[1], "column")),
            )
        )
    return tuple(wx)


def _decode_blocks(
    raw_blocks: object, names: tuple[str, ...], arrays: Arrays
) -> tuple[list[int], list[tuple[SpatialWeights, ...]], list[int | str], list[int]]:
    periods: list[int] = []
    weights: list[tuple[SpatialWeights, ...]] = []
    units: list[int | str] = []
    times: list[int] = []
    for raw in items(raw_blocks, "graph blocks"):
        block = record(raw, {"period", "units", "weights"}, "graph block")
        t = block["period"]
        ids = labels(block["units"], "unit roster")
        if (
            type(t) is not int
            or (periods and t <= periods[-1])
            or not ids
            or any(type(n) is not int for n in ids)
        ):
            msg = "Invalid graph period or unit roster."
            raise ArtifactError(msg)
        periods.append(t)
        matrices = record(block["weights"], set(names), "named weights")
        weights.append(
            tuple(
                prepare_spatial_weights(
                    arrays.get(matrices[n]),
                    unit_ids=jnp.asarray(ids),
                    allow_isolates=True,
                    allow_diagonal=True,
                )
                for n in names
            )
        )
        units.extend(ids)
        times.extend([t] * len(ids))
    return periods, weights, units, times


def load_graph_predictor(path: Path) -> FrozenGraphPredictor:  # noqa: PLR0914
    """Verify and restore a predictor with only the frozen reduced-form capability."""
    saved, arrays = read_bundle(path, purpose="prediction")
    if saved["family"] != _FAMILY or saved["capabilities"] != [_CAP]:
        msg = "Expected a frozen reduced-form graph predictor."
        raise ArtifactError(msg)
    s = record(
        saved["state"],
        {
            "design",
            "panel",
            "unit",
            "lag",
            "error",
            "wx",
            "parameter_names",
            "coefficients",
            "blocks",
        },
        "graph predictor",
    )
    base = decode_design(decode_design_record(s["design"]), arrays)
    lag, error = _names(s["lag"]), _names(s["error"])
    wx = _decode_wx(s["wx"], base)
    names = tuple(dict.fromkeys((*lag, *error, *(n for n, _ in wx))))
    if not names or len(set(wx)) != len(wx):
        msg = "Expected unique graph covariate pairs and at least one graph."
        raise ArtifactError(msg)
    periods, weights, units, times = _decode_blocks(s["blocks"], names, arrays)
    unit = optional_text(s["unit"], "unit column")
    effects = None
    if not periods:
        msg = "The predictor requires at least one graph period."
        raise ArtifactError(msg)
    if s["panel"] is None:
        if periods != [0]:
            msg = "A cross-sectional predictor requires a single block."
            raise ArtifactError(msg)
    else:
        panel = record(s["panel"], {"entity", "time", "effects"}, "panel roles")
        layout = PanelLayout(
            text(panel["entity"], "entity"), text(panel["time"], "time")
        )
        mode = panel["effects"]
        if (
            unit is not None
            or layout.entity == layout.time
            or mode not in ("pooled", "unit", "time", "two-way")
        ):
            msg = "Invalid panel layout or effect specification."
            raise ArtifactError(msg)
        roster = Observations({}, labels={layout.entity: units, layout.time: times})
        effects = prepare_panel_effect_design(
            roster, layout, mode, has_intercept=base.has_intercept
        )
    design = _GraphDesign(base, effects, tuple(wx), lag, error)
    coeff = arrays.get(s["coefficients"])
    if design.parameter_names != _names(s["parameter_names"]) or coeff.shape != (
        len(design.parameter_names),
    ):
        msg = "Inconsistent graph coefficient inventory."
        raise ArtifactError(msg)
    arrays.require_consumed()
    return FrozenGraphPredictor(design, coeff, unit, tuple(periods), tuple(weights))
