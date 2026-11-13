"""Chronological bandwidth selection by frozen Gaussian joint log score."""

# ruff: noqa: TC001 -- public annotations are resolved by beartype.
from __future__ import annotations

import math
from collections.abc import Sequence  # noqa: TC003 -- runtime annotations
from dataclasses import dataclass, replace

from statadic.api.regression import Regression
from statadic.api.spatial.graph_regression import GraphRegressionResult
from statadic.api.spatial.graphs import (
    BandwidthRef,
    DistanceGraph,
    GraphSource,
    PanelData,
    PanelGraph,
    RBFWeights,
)
from statadic.api.spatial.learning import GraphML

__all__ = ["BandwidthSelection", "select_bandwidth"]


@dataclass(frozen=True)
class BandwidthSelection:
    """Scores in candidate order, chosen setting, windows and frozen/refitted result."""

    bandwidth: float
    scores: tuple[tuple[float, float | None], ...]
    training_periods: tuple[int, ...]
    validation_periods: tuple[int, ...]
    refit_periods: tuple[int, ...]
    result: GraphRegressionResult


def _bind(data: PanelData, reference: BandwidthRef, value: float) -> PanelData:
    matched = False

    def bind(source: GraphSource) -> GraphSource:
        nonlocal matched
        if (
            isinstance(source, DistanceGraph)
            and source.weighting.bandwidth == reference
        ):
            matched = True
            return replace(source, weighting=RBFWeights(value))
        return source

    graphs: dict[str, PanelGraph] = {}
    for name, source in data.graphs.items():
        if isinstance(source, tuple):
            graphs[name] = tuple((period, bind(graph)) for period, graph in source)
        elif isinstance(source, DistanceGraph):
            graphs[name] = bind(source)
        else:
            graphs[name] = source
    if not matched:
        msg = f"No distance graph uses the requested bandwidth {reference.name!r}."
        raise ValueError(msg)
    return replace(data, graphs=graphs)


def _periods(data: PanelData) -> tuple[int, ...]:
    dates = set(data.nodes.grouping_labels(data.panel.time))
    if not dates or any(type(t) is not int for t in dates):
        msg = "Bandwidth selection needs nonempty panels with integer periods."
        raise ValueError(msg)
    return tuple(sorted(int(t) for t in dates))


def select_bandwidth(  # noqa: C901 -- split validation, candidate scoring and refit
    model: Regression,
    training: PanelData,
    validation: PanelData,
    *,
    parameter: BandwidthRef,
    candidates: Sequence[float],
    refit: PanelData | None = None,
) -> BandwidthSelection:
    """Select the highest frozen validation log density, breaking ties by input order.

    Candidate h values are absolute cost-scale bandwidths. Researchers can
    form multipliers from a training-only median. A supplied refit panel must
    span exactly the union of training and validation periods.
    """
    train_dates, val_dates = _periods(training), _periods(validation)
    if train_dates[-1] >= val_dates[0] or training.panel != validation.panel:
        msg = "Training periods must precede validation on the same panel layout."
        raise ValueError(msg)
    if refit is not None and (
        refit.panel != training.panel or _periods(refit) != (*train_dates, *val_dates)
    ):
        msg = "Refit must cover exactly the training and validation periods."
        raise ValueError(msg)
    settings = tuple(float(h) for h in candidates)
    if (
        not settings
        or len(set(settings)) != len(settings)
        or not all(math.isfinite(h) and h > 0 for h in settings)
    ):
        msg = "Bandwidth candidates must be distinct, finite and positive."
        raise ValueError(msg)
    if model.estimator is not None and not isinstance(model.estimator, GraphML):
        msg = "Bandwidth selection requires the graph likelihood estimator."
        raise TypeError(msg)
    estimator = model.estimator or GraphML()
    # A grid candidate fixes this reference. Keep other learned bindings, but
    # do not ask GraphML to learn a reference removed from the candidate graph.
    estimator = replace(
        estimator,
        bandwidths={
            reference: binding
            for reference, binding in estimator.bandwidths.items()
            if reference != parameter
        },
    )
    procedure = replace(model, estimator=estimator)
    scores: list[tuple[float, float | None]] = []
    best = None
    highest = -math.inf
    selected = settings[0]
    for h in settings:
        fit = procedure.estimate(_bind(training, parameter, h))
        if not isinstance(fit, GraphRegressionResult):
            msg = "Expected a graph regression result."
            raise TypeError(msg)
        score = (
            float(fit.log_score(_bind(validation, parameter, h)))
            if fit.valid
            else math.nan
        )
        scores.append((h, score if math.isfinite(score) else None))
        if math.isfinite(score) and score > highest:
            best, highest, selected = fit, score, h
    if best is None:
        msg = "No candidate produced a finite held-out score from a valid fit."
        raise ValueError(msg)
    if refit is not None:
        final = procedure.estimate(_bind(refit, parameter, selected))
        if not isinstance(final, GraphRegressionResult) or not final.valid:
            msg = "Selected bandwidth failed the requested final refit."
            raise ValueError(msg)
        best = final
    return BandwidthSelection(
        selected,
        tuple(scores),
        train_dates,
        val_dates,
        () if refit is None else _periods(refit),
        best,
    )
