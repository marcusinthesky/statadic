"""Circular time-block pairs bootstrap for balanced Gaussian graph panels."""

# ruff: noqa: TC001 -- host annotations are resolved at runtime.
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from itertools import pairwise

import jax.numpy as jnp

from statadic.api.data import MeasureColumn, Observations
from statadic.api.design import Term
from statadic.api.panel import spatial_panel_rows
from statadic.api.regression import Regression, RegressionContext
from statadic.api.reporting import ParameterRow, Report
from statadic.api.spatial.graph_regression import (
    GraphRegressionResult,
    PreparedGraphRegression,
)
from statadic.api.spatial.graphs import PanelData
from statadic.api.spatial.learning import GraphML
from statadic.core.typing import Array
from statadic.operators.longrun import circular_block_bootstrap_indices

__all__ = ["GraphBootstrap", "TimeBlockPairs", "bootstrap_graph_regression"]

_MIN_REFITS = 2


@dataclass(frozen=True)
class TimeBlockPairs:
    """Circular whole-date refits of the retained canonical graph specification.

    This supplies percentile uncertainty, not a null-calibrated hypothesis test.
    The original fit is reused; each sampled panel repeats the declared estimator
    from its original initial values, including any fitted graph parameters.
    """

    block_length: int
    n_boot: int
    seed: int

    def infer(self, result: GraphRegressionResult, /) -> GraphBootstrap:
        """Resample the retained panel without recomputing its original fit."""
        if not isinstance(result.context, RegressionContext):
            msg = "Time-block refits require a canonical Regression fit context."
            raise TypeError(msg)
        model = result.context.specification
        model.inspect(result.prepared)
        data = _validate(
            model, result.prepared, self.block_length, self.n_boot, self.seed
        )
        return _bootstrap(
            model,
            data,
            result,
            block_length=self.block_length,
            n_boot=self.n_boot,
            seed=self.seed,
        )


@dataclass(frozen=True)
class GraphBootstrap:
    """Original point fit, source-date schedules and outcomes of every refit."""

    original: GraphRegressionResult
    periods: tuple[int, ...]
    indices: Array
    draws: tuple[tuple[float, ...] | None, ...]
    failures: tuple[str | None, ...]
    bandwidth_names: tuple[str, ...]
    block_length: int
    seed: int

    @property
    def n_eff(self) -> int:
        """Number of valid refits among all attempted bootstrap draws."""
        return sum(draw is not None for draw in self.draws)

    def report(self, *, confidence: float = 0.95) -> Report:
        """Summarize refit variation when every requested draw succeeded."""
        if not 0 < confidence < 1:
            msg = "Confidence must be between zero and one."
            raise ValueError(msg)
        report = self.original.report()
        rows = list(report.parameters)
        for name in self.bandwidth_names:
            label = f"bandwidth[{name}]"
            rows.append(
                ParameterRow(
                    label,
                    float(self.original.graph_parameters[name]),
                    parameter_id=label,
                )
            )
        if self.n_eff == len(self.draws):
            values = jnp.asarray(self.draws)
            se = jnp.std(values, axis=0, ddof=1).tolist()
            tail = (1 - confidence) / 2
            lower, upper = jnp.quantile(
                values, jnp.array([tail, 1 - tail]), axis=0
            ).tolist()
            resolve_tails = self.n_eff * tail >= 1 - 1e-12
            rows = [
                replace(
                    row,
                    std_error=err,
                    ci_lower=lo if resolve_tails else None,
                    ci_upper=hi if resolve_tails else None,
                    interval_status="available" if resolve_tails else "unavailable",
                    interval_unavailable_reason=None
                    if resolve_tails
                    else "too-few-refits-for-tail",
                )
                for row, err, lo, hi in zip(rows, se, lower, upper, strict=True)
            ]
        else:
            rows = [
                replace(
                    row,
                    interval_status="unavailable",
                    interval_unavailable_reason="failed-refits",
                )
                for row in rows
            ]
        return replace(
            report,
            parameters=tuple(rows),
            metadata={
                **report.metadata,
                "inference": "circular time-block pairs bootstrap, percentile",
                "confidence": confidence,
                "original_periods": self.periods,
                "units_per_date": len(self.original.prepared.rows.rosters[0]),
                "sampling_assumption": (
                    "independent identically distributed dates"
                    if self.block_length == 1
                    else "weakly dependent stationary regularly spaced dates"
                ),
                "block_length": self.block_length,
                "seed": self.seed,
                "attempted_refits": len(self.draws),
                "n_eff": self.n_eff,
                "refit_failures": dict(Counter(f for f in self.failures if f)),
                "graph_refits": tuple(self.original.graph_parameters),
            },
            notes=(
                *report.notes,
                (
                    "Conditional on the unit roster, supplied upstream geometry "
                    "and declared estimation procedure."
                ),
            ),
        )


def _take_periods(
    data: Observations, time: str, indices: tuple[tuple[int, ...], ...]
) -> Observations:
    """Make unique bootstrap spells, carrying each selected date's complete rows."""
    positions = tuple(i for period in indices for i in period)
    take = jnp.array(positions, dtype=jnp.int32)
    dates = tuple(t for t, period in enumerate(indices) for _ in period)
    ids = tuple(range(len(positions)))
    columns = {
        k: jnp.array(dates, dtype=jnp.int32) if k == time else v[take]
        for k, v in data.columns.items()
    }
    labels = {
        k: dates if k == time else tuple(v[i] for i in positions)
        for k, v in data.labels.items()
    }
    return Observations(
        columns,
        row_ids=ids,
        labels=labels,
        measures={
            k: MeasureColumn(ids, v.measures.take(take))
            for k, v in data.measures.items()
        },
        source=data.source,
        source_rows=None
        if data.source is None
        else (
            positions
            if data.source_rows is None
            else tuple(data.source_rows[i] for i in positions)
        ),
    )


def _prepare(
    model: Regression, data: PanelData, block_length: int, n_boot: int, seed: int
) -> PreparedGraphRegression:
    prepared = model.prepare(data)
    if not isinstance(prepared, PreparedGraphRegression):
        msg = "Supply a graph regression (use GraphML for a single supplied graph)."
        raise TypeError(msg)
    _validate(model, prepared, block_length, n_boot, seed)
    return prepared


def _validate(
    model: Regression,
    prepared: PreparedGraphRegression,
    block_length: int,
    n_boot: int,
    seed: int,
) -> PanelData:
    data = prepared.data
    if not isinstance(data, PanelData):
        msg = "Time-block refits require retained PanelData."
        raise TypeError(msg)
    if model.effects not in (None, "pooled", "unit"):
        msg = "Time-block graph refits require pooled or unit fixed effects."
        raise ValueError(msg)
    time = data.panel.time
    if not isinstance(model.terms, Term):
        msg = "Time-block graph refits require one predictor design."
        raise TypeError(msg)
    if time in model.terms.input_columns or time == model.response:
        msg = "The time key is reassigned in each draw and cannot enter the model."
        raise ValueError(msg)
    rows = prepared.rows
    periods = rows.periods
    if prepared.observed is not None or len(set(rows.rosters)) != 1:
        msg = "Time-block refits require full outcomes on one balanced ordered roster."
        raise ValueError(msg)
    if type(block_length) is not int or not 1 <= block_length < len(periods):
        msg = (
            "Block length must be a positive integer smaller than the number of dates."
        )
        raise ValueError(msg)
    if block_length > 1 and len({b - a for a, b in pairwise(periods)}) != 1:
        msg = "Multiple-date blocks require regularly spaced period keys."
        raise ValueError(msg)
    if type(n_boot) is not int or n_boot < _MIN_REFITS or type(seed) is not int:
        msg = "Supply at least two refits and an explicit integer seed."
        raise ValueError(msg)
    return data


def bootstrap_graph_regression(
    model: Regression,
    data: PanelData,
    *,
    block_length: int,
    n_boot: int,
    seed: int,
) -> GraphBootstrap:
    """Refit a declared graph model after resampling whole contiguous date blocks.

    Length one assumes independent dates. Longer circular blocks require a
    regularly spaced, approximately stationary date process with weak dependence.
    The roster stays fixed and external training or model selection is conditional.
    """
    prepared = _prepare(model, data, block_length, n_boot, seed)
    original = model.estimate_prepared(prepared)
    if not isinstance(original, GraphRegressionResult):
        msg = "Expected a graph regression result."
        raise TypeError(msg)
    return _bootstrap(
        model, data, original, block_length=block_length, n_boot=n_boot, seed=seed
    )


def _bootstrap(  # noqa: PLR0914 -- retain all aligned graph and date state per draw
    model: Regression,
    data: PanelData,
    original: GraphRegressionResult,
    *,
    block_length: int,
    n_boot: int,
    seed: int,
) -> GraphBootstrap:
    prepared = original.prepared
    rows = prepared.rows
    periods = rows.periods
    time = data.panel.time
    original.require_valid()
    if not all(
        bool(jnp.all(jnp.isfinite(v))) for v in original.graph_parameters.values()
    ):
        msg = "Original fit has nonfinite graph parameters."
        raise ValueError(msg)
    names = (
        tuple(ref.name for ref in model.estimator.bandwidths)
        if isinstance(model.estimator, GraphML)
        else ()
    )
    if any(f"bandwidth[{name}]" in original.parameter_names for name in names):
        msg = (
            "Bootstrap bandwidth labels must differ from regression coefficient names."
        )
        raise ValueError(msg)
    mapping = spatial_panel_rows(
        data.spatial, data.panel, tuple(zip(periods, rows.rosters, strict=True))
    )
    schedule = circular_block_bootstrap_indices(
        len(periods), block_length, n_boot, seed=seed
    )
    draws: list[tuple[float, ...] | None] = []
    failures: list[str | None] = []
    for sample in schedule.tolist():
        nodes = _take_periods(
            data.nodes, time, tuple(mapping.node_rows[t] for t in sample)
        )
        outcomes = (
            None
            if data.outcomes is None
            else _take_periods(
                data.outcomes, time, tuple(mapping.outcome_rows[t] for t in sample)
            )
        )
        graphs = {
            name: tuple((i, rows.graphs[t][name]) for i, t in enumerate(sample))
            for name in rows.graphs[0]
        }
        try:
            result = model.estimate(PanelData(nodes, data.panel, graphs, outcomes))
        except ValueError as exc:
            # Preparation may reject a draw that omits a required design level.
            failures.append(f"{type(exc).__name__}: {exc}")
            draws.append(None)
            continue
        if not isinstance(result, GraphRegressionResult):
            msg = "Expected a graph regression refit."
            raise TypeError(msg)
        valid = result.valid and all(
            bool(jnp.all(jnp.isfinite(v))) for v in result.graph_parameters.values()
        )
        if not valid or result.parameter_names != original.parameter_names:
            failures.append("invalid-fit" if not valid else "changed-parameter-schema")
            draws.append(None)
            continue
        values = (
            *result.parameters.tolist(),
            *(float(result.graph_parameters[k]) for k in names),
        )
        draws.append(values)
        failures.append(None)
    return GraphBootstrap(
        original,
        periods,
        schedule,
        tuple(draws),
        tuple(failures),
        names,
        block_length,
        seed,
    )
