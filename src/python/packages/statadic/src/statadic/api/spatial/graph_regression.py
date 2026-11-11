"""Gaussian regression with individually named SAR, disturbance and WX graphs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
    Protocol,
    overload,
    override,
    runtime_checkable,
)

import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from statadic.api.base import FitContext, ParameterResult
from statadic.api.design import FittedDesign
from statadic.api.fit_statistics import LikelihoodSummary
from statadic.api.inference import InferenceResult
from statadic.api.panel import (
    prepare_panel_effect_design,
    spatial_panel_rows,
)
from statadic.api.prediction import (  # noqa: TC001
    PredictionOptions,
    _spatial_prediction_options,
)
from statadic.api.reporting import Metric, ParameterRow, Report
from statadic.api.spatial.cross_section import (
    _SPATIAL_LIKELIHOOD_REASON,
    SpatialInformation,
)
from statadic.api.spatial.graphs import (
    BandwidthRef,
    DistanceGraph,
    MeasureDistances,
    PanelData,
    SpatialData,
)
from statadic.api.spatial.prediction import SpatialPanelPrediction
from statadic.core.matrices import DMat
from statadic.core.typing import Array
from statadic.estimation.spatial.graph_likelihood import (
    GraphMLFit,
    _graph_mean,
    estimate_graph_ml,
    graph_log_density,
)
from statadic.inference.spatial.graph import _graph_covariance
from statadic.operators.spatial import SpatialPanelWeights

if TYPE_CHECKING:
    from collections.abc import Mapping

    from statadic.api.data import PanelLayout
    from statadic.api.design import Term
    from statadic.api.panel import (
        PanelEffectDesign,
    )
    from statadic.api.reporting import MetadataValue
    from statadic.api.spatial.graphs import GraphSource, SpatialX

__all__ = ["GraphRegressionResult", "GraphResampling", "PreparedGraphRegression"]


@runtime_checkable
class GraphResampling[OutputT](Protocol):
    """A declared refit procedure consuming the exact retained graph fit."""

    def infer(self, result: GraphRegressionResult, /) -> OutputT:
        """Evaluate supported refits without replacing the original point fit."""
        ...


type _Data = PanelData | SpatialData
type _Stacks = tuple[tuple[Array, ...], tuple[Array, ...], tuple[Array, ...]]


def _unit_ids(source: GraphSource) -> tuple[int, ...]:
    return tuple(jnp.asarray(source.unit_ids).tolist())


@dataclass(frozen=True)
class _GraphDesign:
    base: FittedDesign
    effects: PanelEffectDesign | None
    wx: tuple[tuple[str, int], ...]
    lag: tuple[str, ...]
    error: tuple[str, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys((*self.lag, *self.error, *(name for name, _ in self.wx)))
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        local = self.base.column_names
        return (
            *local,
            *(f"W[{graph}]:{local[column]}" for graph, column in self.wx),
            *(self.effects.column_names if self.effects is not None else ()),
            *(f"rho[{name}]" for name in self.lag),
            *(f"lambda[{name}]" for name in self.error),
        )


@dataclass(frozen=True)
class _Rows:
    graphs: tuple[Mapping[str, GraphSource], ...]
    rosters: tuple[tuple[int, ...], ...]
    periods: tuple[int, ...]
    x: tuple[Array, ...]
    effects: tuple[Array, ...]
    outcome_rows: tuple[tuple[int, ...], ...]
    observed: tuple[Array, ...] | None


def _align(
    data: _Data, design: _GraphDesign, *, observed_likelihood: bool = False
) -> _Rows:
    names = design.names
    if not set(names) <= set(data.graphs):
        msg = f"Missing named graphs: {sorted(set(names) - set(data.graphs))}."
        raise ValueError(msg)
    if isinstance(data, SpatialData):
        graph = MappingProxyType({name: data.graphs[name] for name in names})
        units = _unit_ids(graph[names[0]])
        rows = data.observations
        ids = rows.row_ids if data.unit is None else rows.grouping_labels(data.unit)
        if ids != units or any(_unit_ids(src) != units for src in graph.values()):
            msg = "Spatial rows must match each graph's ordered numeric unit IDs."
            raise ValueError(msg)
        x = design.base.transform(rows)
        return _Rows(
            (graph,),
            (units,),
            (0,),
            (x,),
            (x[:, :0],),
            (tuple(range(len(units))),),
            None,
        )

    sources = {
        name: sorted(data.graph_sources(name), key=lambda pair: pair[0])
        for name in names
    }
    ref = sources[names[0]]
    periods = tuple(t for t, _ in ref)
    rosters = tuple(_unit_ids(src) for _, src in ref)
    if any(
        tuple(t for t, _ in graph) != periods
        or tuple(_unit_ids(src) for _, src in graph) != rosters
        for graph in sources.values()
    ):
        msg = "Every graph must have matching ordered periods and unit rosters."
        raise ValueError(msg)
    mapped = spatial_panel_rows(
        data.spatial, data.panel, tuple(zip(periods, rosters, strict=True))
    )
    graphs = tuple(
        MappingProxyType({name: sources[name][t][1] for name in names})
        for t in range(len(ref))
    )
    x = design.base.transform(data.nodes)
    effects = (
        x[:, :0] if design.effects is None else design.effects.transform(data.nodes)
    )
    ids = tuple(jnp.asarray(rows, dtype=jnp.int32) for rows in mapped.node_rows)
    observed = (
        None
        if not observed_likelihood
        and all(
            len(pos) == len(roster)
            for pos, roster in zip(mapped.observed, rosters, strict=True)
        )
        else tuple(jnp.asarray(pos, dtype=jnp.int32) for pos in mapped.observed)
    )
    return _Rows(
        graphs,
        rosters,
        periods,
        tuple(x[i] for i in ids),
        tuple(effects[i] for i in ids),
        mapped.outcome_rows,
        observed,
    )


def _arrays(
    design: _GraphDesign, rows: _Rows, parameters: Mapping[str, Array] | None
) -> _Stacks:
    lags: list[Array] = []
    errors: list[Array] = []
    designs: list[Array] = []
    cache: dict[int, Array] = {}
    for graph, x, effects in zip(rows.graphs, rows.x, rows.effects, strict=True):
        values = {}
        for name, source in graph.items():
            key = id(source)
            if key not in cache:
                value = (
                    source.weights(parameters)
                    if isinstance(source, DistanceGraph)
                    else source.values
                )
                if isinstance(value, BCOO):
                    msg = "Multi-graph Gaussian ML requires dense operators."
                    raise TypeError(msg)
                cache[key] = value
            values[name] = cache[key]
        empty = jnp.empty((0, x.shape[0], x.shape[0]), dtype=x.dtype)
        lags.append(
            jnp.stack([values[name] for name in design.lag]) if design.lag else empty
        )
        errors.append(
            jnp.stack([values[name] for name in design.error])
            if design.error
            else empty
        )
        designs.append(
            jnp.column_stack(
                (x, *(values[name] @ x[:, col] for name, col in design.wx), effects)
            )
        )
    return tuple(lags), tuple(errors), tuple(designs)


@dataclass(frozen=True)
class PreparedGraphRegression:
    """Bound graph roles, identities and fitted local design before solving."""

    schema: _GraphDesign
    rows: _Rows
    response: str
    outcome: tuple[Array, ...]
    data: _Data
    model_token: object = field(repr=False)

    @property
    def lag_names(self) -> tuple[str, ...]:
        """Return the declared response-lag graph names in coefficient order."""
        return self.schema.lag

    @property
    def error_names(self) -> tuple[str, ...]:
        """Return the declared disturbance graph names in coefficient order."""
        return self.schema.error

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the complete qualified mean and dependence coefficient schema."""
        return self.schema.parameter_names

    @property
    def graphs(self) -> tuple[Mapping[str, GraphSource], ...]:
        """Return the graph sources aligned to each retained period."""
        return self.rows.graphs

    @property
    def observed(self) -> tuple[Array, ...] | None:
        """Return observed-node selectors, or None for complete outcomes."""
        return self.rows.observed

    def arrays(self, parameters: Mapping[str, Array] | None = None) -> _Stacks:
        """Evaluate matrix values and rebuild each named WX column."""
        return _arrays(self.schema, self.rows, parameters)

    def make_result(
        self,
        fit: GraphMLFit,
        *,
        parameters: Mapping[str, Array] | None = None,
        sensitivity: Mapping[str, bool] | None = None,
        availability: Mapping[str, bool] | None = None,
        free_dimensions: Mapping[str, int] | None = None,
    ) -> GraphRegressionResult:
        """Freeze numerical graph state with this exact fit and observation axis."""
        observations = (
            self.data.spatial.outcomes
            if isinstance(self.data, PanelData)
            else self.data.observations
        )
        return GraphRegressionResult(
            fit,
            FitContext(observations, "Gaussian spatial regression", self.response),
            self,
            MappingProxyType({} if parameters is None else dict(parameters)),
            MappingProxyType({} if sensitivity is None else dict(sensitivity)),
            MappingProxyType({} if availability is None else dict(availability)),
            MappingProxyType({} if free_dimensions is None else dict(free_dimensions)),
        )

    def estimate(self) -> GraphRegressionResult:
        """Estimate the frozen graph problem with its supplied graph parameters."""
        if any(
            isinstance(src, DistanceGraph)
            and (
                src.projection_ref is not None
                or isinstance(src.weighting.bandwidth, BandwidthRef)
            )
            for period in self.graphs
            for src in period.values()
        ):
            msg = "Unbound graph parameters require GraphML."
            raise ValueError(msg)
        lag, error, design = self.arrays()
        return self.make_result(
            estimate_graph_ml(
                lag,
                error,
                design,
                self.outcome,
                self.parameter_names,
                observed=self.observed,
            )
        )


def _prepare_graph_regression(
    data: _Data,
    response: str,
    base: Term,
    lags: tuple[SpatialX, ...],
    lag: tuple[str, ...],
    error: tuple[str, ...],
    effects: Literal["pooled", "unit", "time", "two-way", "random"],
    panel: PanelLayout | None,
    token: object,
    *,
    observed_likelihood: bool = False,
) -> PreparedGraphRegression:
    """Bind graph sources without evaluating any parameter-dependent distances."""
    if effects == "random" or (
        not isinstance(data, PanelData) and (effects != "pooled" or panel is not None)
    ):
        msg = "Multi-graph regression supports pooled or fixed panel effects only."
        raise ValueError(msg)
    if isinstance(data, PanelData) and panel is not None and data.panel != panel:
        msg = "Model and data panel layouts must agree."
        raise ValueError(msg)
    nodes = data.nodes if isinstance(data, PanelData) else data.observations
    fitted = FittedDesign(base._prepare_term(nodes))
    if effects in ("unit", "time", "two-way") and fitted.has_intercept:
        msg = "Fixed effects require a design without an explicit intercept."
        raise ValueError(msg)
    fixed = (
        prepare_panel_effect_design(
            nodes, data.panel, effects, has_intercept=fitted.has_intercept
        )
        if isinstance(data, PanelData)
        else None
    )
    covariates = {
        name
        for i, name in enumerate(fitted.column_names)
        if i not in fitted.intercept_columns
    }
    if any(term.column not in covariates for term in lags):
        msg = "SpatialX requires a named, non-intercept column from the local design."
        raise ValueError(msg)
    wx = tuple((term.graph, fitted.column_names.index(term.column)) for term in lags)
    if len(set(wx)) != len(wx):
        msg = "Each SpatialX column and graph pair must be unique."
        raise ValueError(msg)
    schema = _GraphDesign(fitted, fixed, wx, lag, error)
    if len(set(schema.parameter_names)) != len(schema.parameter_names):
        msg = "Design and graph-qualified parameter names must be unique."
        raise ValueError(msg)
    rows = _align(data, schema, observed_likelihood=observed_likelihood)
    outcomes = (
        data.spatial.outcomes if isinstance(data, PanelData) else nodes
    ).numeric(response)
    y = tuple(outcomes[jnp.asarray(r, dtype=jnp.int32)] for r in rows.outcome_rows)
    return PreparedGraphRegression(schema, rows, response, y, data, token)


_REPORT_COMPONENTS = 8
_PANEL_MATRIX_AXES = 3


def _graph_parameter_summary(value: Array) -> dict[str, MetadataValue]:
    """Keep fitted loadings on the result, reporting scalar and spectral summaries."""
    finite = bool(jnp.all(jnp.isfinite(value)))
    if not value.ndim:
        return {"shape": (), "value": float(value) if finite else None}
    summary: dict[str, MetadataValue] = {"shape": tuple(value.shape), "finite": finite}
    if finite:
        spectrum = jnp.linalg.svd(value, compute_uv=False)
        if not bool(jnp.all(jnp.isfinite(spectrum))):
            return summary
        norm = jnp.linalg.norm(spectrum)
        cutoff = jnp.finfo(spectrum.dtype).eps * max(value.shape) * spectrum[0]
        summary.update(
            {
                "rank": int(jnp.sum(spectrum > cutoff)),
                "frobenius_norm": float(norm) if bool(jnp.isfinite(norm)) else None,
                "singular_values": tuple(spectrum[:_REPORT_COMPONENTS].tolist()),
                "singular_values_omitted": max(0, spectrum.size - _REPORT_COMPONENTS),
            }
        )
    return summary


class _GraphRecipe(NamedTuple):
    source: str
    geometry: str | None = None
    weighting: str | None = None
    bandwidth: float | None = None
    bandwidth_ref: str | None = None
    projection: str | None = None


def _recipe_label(graph: GraphSource) -> _GraphRecipe:
    if not isinstance(graph, DistanceGraph):
        return _GraphRecipe("weights")
    h = graph.weighting.bandwidth
    source = graph.source
    geometry = None
    projection = None
    if isinstance(source, MeasureDistances):
        source_kind = "measures"
        geometry = type(source.geometry).__name__
        if graph.projection_ref is not None:
            projection = graph.projection_ref.name
        elif source.projection is not None:
            projection = "fixed"
    else:
        source_kind = "distances" if isinstance(source, DMat) else "costs"
    return _GraphRecipe(
        source_kind,
        geometry,
        type(graph.weighting).__name__,
        None if isinstance(h, BandwidthRef) else float(h),
        h.name if isinstance(h, BandwidthRef) else None,
        projection,
    )


def _graph_provenance(prepared: PreparedGraphRegression) -> dict[str, MetadataValue]:
    provenance: dict[str, MetadataValue] = {}
    for name in prepared.schema.names:
        declared = prepared.data.graphs[name]
        indexed = isinstance(declared, tuple) or (
            isinstance(declared, SpatialPanelWeights)
            and declared.values.ndim == _PANEL_MATRIX_AXES
        )
        counts: dict[_GraphRecipe, int] = {}
        for period in prepared.graphs:
            recipe = _recipe_label(period[name])
            counts[recipe] = counts.get(recipe, 0) + 1
        provenance[name] = {
            "binding": "period-indexed" if indexed else "fixed",
            "period_count": len(prepared.graphs),
            "recipes": tuple(
                {**recipe._asdict(), "period_count": count}
                for recipe, count in tuple(counts.items())[:_REPORT_COMPONENTS]
            ),
            "recipes_omitted": max(0, len(counts) - _REPORT_COMPONENTS),
        }
    return provenance


@dataclass(frozen=True)
class GraphRegressionResult(ParameterResult[GraphMLFit]):
    """Gaussian spatial fit with graph-qualified parameters and frozen graph state."""

    prepared: PreparedGraphRegression = field(repr=False)
    graph_parameters: Mapping[str, Array] = field(repr=False)
    graph_sensitivity: Mapping[str, bool] = field(repr=False)
    graph_availability: Mapping[str, bool] = field(default_factory=dict, repr=False)
    graph_free_dimensions: Mapping[str, int] = field(default_factory=dict, repr=False)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the actual retained native coefficient names."""
        return self.core.parameter_names

    @override
    def likelihood_summary(self) -> LikelihoodSummary:
        """Return the Gaussian graph likelihood summary.

        ``log_likelihood`` is the full Gaussian likelihood with the ``2 pi`` term
        and the summed graph-filter Jacobians (the observed-node marginal when
        outcomes are partly observed). ``k`` is the mean coefficients, every
        graph spatial coefficient, the scale, and every coordinate of a learned
        graph parameter (bandwidth or projection entry). ``n`` counts observed
        outcomes.
        """
        graph_coordinates = sum(
            self.graph_free_dimensions.get(name, int(jnp.size(value)))
            for name, value in self.graph_parameters.items()
        )
        return LikelihoodSummary(
            float(self.core.log_likelihood) if self.valid else math.nan,
            len(self.core.parameters) + 1 + graph_coordinates,
            sum(int(y.size) for y in self.prepared.outcome),
            _SPATIAL_LIKELIHOOD_REASON,
        )

    def report(self) -> Report:
        """Describe retained coefficients, graph state and numerical status."""
        core = self.core
        statistics = self.fit_statistics()
        return Report(
            "Gaussian spatial regression",
            parameters=tuple(
                ParameterRow(
                    name,
                    float(value) if self.valid else None,
                    parameter_id=name,
                    valid=self.valid,
                    status="available" if self.valid else "unavailable",
                    unavailable_reason=None if self.valid else "invalid-fit",
                )
                for name, value in zip(
                    self.parameter_names, core.parameters, strict=True
                )
            ),
            metrics=(
                Metric("n_observations", int(self.context.observations.n_rows)),
                *statistics.metrics,
            ),
            notes=statistics.notes,
            metadata={
                "outcome_lag_graphs": self.prepared.lag_names,
                "error_graphs": self.prepared.error_names,
                "spatial_x": tuple(
                    (name, self.prepared.schema.base.column_names[col])
                    for name, col in self.prepared.schema.wx
                ),
                "graph_provenance": _graph_provenance(self.prepared),
                "graph_parameters": {
                    name: _graph_parameter_summary(value)
                    for name, value in self.graph_parameters.items()
                },
                "inference": "joint graph parameter covariance unavailable"
                if self.graph_parameters
                else "conditional on fixed graphs; request SpatialInformation",
                "identification_scope": (
                    "observed mean design and spatial profile curvature "
                    "conditional on realized graphs"
                ),
                "graph_profile_perturbation_detected": self.graph_sensitivity,
                "graph_construction_checks": self.graph_availability,
                "graph_construction_unavailable": tuple(
                    name
                    for name, available in self.graph_availability.items()
                    if not available
                ),
                "graph_parameter_identification": "unverified"
                if self.graph_parameters
                else "fixed",
                "solver_converged": bool(core.converged),
                "solver_status": int(core.solver.status),
                "solver_residual_norm": (
                    float(core.solver.residual_norm)
                    if bool(jnp.isfinite(core.solver.residual_norm))
                    else None
                ),
                "solver_iterations": int(core.solver.iterations),
                "identified": bool(core.identified),
                "boundary": bool(core.boundary),
            },
        )

    def _evaluation(self, data: _Data) -> tuple[_Rows, _Stacks]:
        # NaN coefficients of an invalid fit would otherwise fail the domain check.
        self.require_valid()
        training = self.prepared
        if type(data) is not type(training.data):
            msg = "Prediction must supply the fitted spatial data layout."
            raise TypeError(msg)
        if (
            isinstance(data, PanelData)
            and isinstance(training.data, PanelData)
            and data.panel != training.data.panel
        ):
            msg = "Prediction panel roles must match the training layout."
            raise ValueError(msg)
        rows = _align(data, training.schema)
        if isinstance(data, SpatialData) and rows.rosters != training.rows.rosters:
            msg = "A changed cross-sectional unit roster requires a new model."
            raise ValueError(msg)
        stacks = _arrays(training.schema, rows, self.graph_parameters)
        lag, error, design = stacks
        spatial = self.parameters[design[0].shape[1] :]
        n_lag = len(training.lag_names)
        for period, w, v in zip(rows.periods, lag, error, strict=True):
            for roles, coefficients in ((w, spatial[:n_lag]), (v, spatial[n_lag:])):
                norms = jnp.max(jnp.sum(jnp.abs(roles), axis=-1), axis=-1)
                if not bool(jnp.abs(coefficients) @ norms < 1):
                    msg = (
                        "Fitted spatial dependence is outside the admissible range "
                        f"for prediction period {period}."
                    )
                    raise ValueError(msg)
        return rows, stacks

    def log_score(self, data: _Data) -> Array:
        """Sum held-out joint Gaussian log densities, marginalizing missing outcomes."""
        training = self.prepared
        rows, (lag, error, design) = self._evaluation(data)
        width = design[0].shape[1]
        outcomes = (
            data.spatial.outcomes if isinstance(data, PanelData) else data.observations
        ).numeric(training.response)
        y = tuple(outcomes[jnp.asarray(r, dtype=jnp.int32)] for r in rows.outcome_rows)
        return graph_log_density(
            lag,
            error,
            design,
            y,
            self.parameters[:width],
            self.parameters[width:],
            self.core.sigma2,
            observed=rows.observed,
        )

    @overload
    def infer(self, *, covariance: SpatialInformation) -> InferenceResult: ...

    @overload
    def infer[OutputT](self, *, resampling: GraphResampling[OutputT]) -> OutputT: ...

    def infer[OutputT](
        self,
        *,
        covariance: SpatialInformation | None = None,
        resampling: GraphResampling[OutputT] | None = None,
    ) -> InferenceResult | OutputT:
        """Attach conditional covariance or run an explicit graph refit design."""
        if (covariance is None) == (resampling is None):
            msg = "Supply exactly one covariance or resampling request."
            raise TypeError(msg)
        if resampling is not None:
            if not isinstance(resampling, GraphResampling):
                msg = "Graph resampling requires a supported refit request."
                raise TypeError(msg)
            return resampling.infer(self)
        if self.graph_parameters:
            msg = "Joint graph parameter uncertainty is unavailable for learned graphs."
            raise ValueError(msg)
        if not isinstance(covariance, SpatialInformation) or covariance.debiased:
            msg = "Fixed graph ML requires non-debiased SpatialInformation."
            raise TypeError(msg)
        p = self.prepared
        lag, error, design = p.arrays()
        uncertainty = _graph_covariance(
            self.core, lag, error, design, p.outcome, observed=p.observed
        )
        return InferenceResult(
            uncertainty,
            self.context,
            covariance,
            design=self.prepared.schema.base,
            fit_statistics=self.fit_statistics(),
        )

    def prediction_options(self) -> PredictionOptions:
        """Describe keyed predictions using the fitted named graph roles."""
        training = self.prepared
        data = training.data
        keys = (
            (data.panel.entity, data.panel.time)
            if isinstance(data, PanelData)
            else (() if data.unit is None else (data.unit,))
        )
        return _spatial_prediction_options(
            tuple(dict.fromkeys((*training.schema.base.input_columns, *keys))),
            response=training.response if training.lag_names else None,
            condition=f"Aligned graph rosters: {', '.join(training.schema.names)}.",
            valid=self.valid,
        )

    @overload
    def predict(
        self,
        data: PanelData,
        *,
        kind: Literal["reduced-form", "structural"] = "reduced-form",
    ) -> SpatialPanelPrediction: ...

    @overload
    def predict(
        self,
        data: SpatialData,
        *,
        kind: Literal["reduced-form", "structural"] = "reduced-form",
    ) -> Array: ...

    def predict(
        self,
        data: _Data,
        *,
        kind: Literal["reduced-form", "structural"] = "reduced-form",
    ) -> Array | SpatialPanelPrediction:
        """Predict on the supplied roster using the fitted graph parameters."""
        if kind not in ("structural", "reduced-form"):
            msg = "Unknown spatial prediction kind."
            raise ValueError(msg)
        training = self.prepared
        rows, (lag, _, designs) = self._evaluation(data)
        width = designs[0].shape[1]
        rho = self.parameters[width : width + len(training.lag_names)]
        outcomes = None
        if kind == "structural" and rho.size:
            outcomes = (
                data.spatial.outcomes
                if isinstance(data, PanelData)
                else data.observations
            ).numeric(training.response)
        values = []
        for t, x in enumerate(designs):
            y = None
            if outcomes is not None:
                y = outcomes[jnp.asarray(rows.outcome_rows[t], dtype=jnp.int32)]
                if rows.observed is not None:
                    present = (
                        jnp.zeros((x.shape[0],), dtype=bool)
                        .at[rows.observed[t]]
                        .set(True)
                    )
                    if bool(jnp.any(lag[t][:, :, ~present] != 0)):
                        msg = "Structural prediction requires every neighbor outcome."
                        raise ValueError(msg)
                    y = jnp.zeros(x.shape[0], dtype=y.dtype).at[rows.observed[t]].set(y)
            values.append(
                _graph_mean(lag[t], rho, x, self.parameters[:width], outcome=y)
            )
        return (
            values[0]
            if isinstance(data, SpatialData)
            else SpatialPanelPrediction(rows.periods, rows.rosters, tuple(values))
        )
