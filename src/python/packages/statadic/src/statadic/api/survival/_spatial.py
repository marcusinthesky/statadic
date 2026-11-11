"""Host workflow for endogenous spatial log-time survival."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, override

import jax.numpy as jnp
from jax.scipy.special import log_ndtr, ndtri

from statadic.api.base import (
    FitContext,
    Model,
    ParameterResult,
)
from statadic.api.design import DesignContrast  # noqa: TC001
from statadic.api.fit_statistics import LikelihoodSummary
from statadic.api.inference import InferenceResult, ModelBased  # noqa: TC001
from statadic.api.panel import (
    prepare_panel_effect_design,
    prepare_spatial_panel_data,
    select_lag_columns,
)
from statadic.api.prediction import PredictionOptions  # noqa: TC001
from statadic.api.reporting import Metric, ParameterRow, Report, _finite_metric
from statadic.api.spatial.graphs import PanelData
from statadic.api.spatial.prediction import SpatialPanelPrediction
from statadic.api.survival._inputs import SpatialSurvivalData, _SpatialSurvivalInput
from statadic.api.survival._prediction import survival_prediction_options
from statadic.api.survival.inference import survival_inference
from statadic.estimation.survival import (
    SpatialSurvivalFit,
    SpatialSurvivalML,
    SpatialSurvivalProblem,
    prepare_spatial_survival_period,
)
from statadic.estimation.survival.refinement import (
    IntegrationRefinement,
    refine_spatial_survival,
)
from statadic.model.survival import LaggedLogTime

if TYPE_CHECKING:
    from statadic.api.data import PanelLayout, SpatialPanelData
    from statadic.api.design import Term
    from statadic.api.panel import (
        PanelEffectDesign,
        PreparedSpatialPanelData,
    )
    from statadic.api.survival._inputs import RightCensored
    from statadic.core.typing import Array, ArrayLike
    from statadic.estimation.survival.refinement import RefinedSpatialSurvivalFit
    from statadic.operators.spatial import SpatialWeights
    from statadic.optimize.integration import IntegrationDesign


@dataclass(frozen=True)
class PreparedSpatialSurvival:
    """Full graph design and separately aligned observed spell inputs."""

    data: _SpatialSurvivalInput
    panel: PreparedSpatialPanelData
    effects: PanelEffectDesign | None
    problem: SpatialSurvivalProblem
    estimator: SpatialSurvivalML
    layout: PanelLayout
    response: str
    model_token: object = field(repr=False)


@dataclass(frozen=True)
class RefinedSpatialSurvivalResult:
    """Host result and the complete retained numerical-refinement diagnostics."""

    result: SpatialSurvivalResult
    refinement: RefinedSpatialSurvivalFit = field(repr=False)


@dataclass(frozen=True)
class SpatialSurvivalResult(ParameterResult[SpatialSurvivalFit]):
    """Retained joint fit, full graph identities and design state."""

    panel: PreparedSpatialPanelData = field(repr=False)
    layout: PanelLayout
    effects: PanelEffectDesign | None = field(repr=False)

    graph_name: str | None = None

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the coefficient names used by ``report``."""
        return self.core.parameter_names

    @override
    def likelihood_summary(self) -> LikelihoodSummary:
        """Return the joint spatial log-time likelihood summary.

        ``k`` counts the coefficients (including spatially lagged columns and
        fixed-effect columns), the spatial parameter ``rho`` unless the caller
        fixed it, and the log scale. The censored-spell terms are numerically
        integrated, so the log likelihood carries the reported integration error.
        ``n`` is the number of spells.
        """
        return LikelihoodSummary(
            float(self.core.log_likelihood),
            int(self.core.parameters.shape[-1]),
            int(self.context.observations.n_rows),
            "The spatial survival fit is invalid, so its log likelihood "
            "is unavailable.",
        )

    @override
    def report(self) -> Report:
        valid = bool(self.core.valid)
        statistics = self.fit_statistics()
        return Report(
            title=f"{self.context.model_name} fit",
            parameters=tuple(
                ParameterRow(
                    name,
                    float(self.core.parameters[i]) if valid else None,
                    valid=valid,
                    status="available" if valid else "unavailable",
                    unavailable_reason=None if valid else "invalid-fit",
                )
                for i, name in enumerate(self.core.parameter_names)
            ),
            metrics=(
                Metric("n_observations", int(self.context.observations.n_rows)),
                *statistics.metrics,
                _finite_metric(
                    "integration_error",
                    self.core.integration_error,
                    "Numerical integration diagnostic.",
                ),
            ),
            notes=statistics.notes,
            warnings=("Integration accuracy remains unqualified.",),
            metadata={
                "fit_valid": valid,
                "graph": self.graph_name,
                "period_ids": self.panel.period_ids,
                "design_columns": self.panel.design.column_names,
                "effect_columns": ()
                if self.effects is None
                else self.effects.column_names,
                "integration_draws": tuple(
                    p.integration.draws for p in self.core.problem.periods
                ),
                "integration_replicates": tuple(
                    p.integration.replicates for p in self.core.problem.periods
                ),
                "identified": bool(self.core.identified),
                "integration_valid": bool(self.core.integration_valid),
                "information_available": bool(self.core.information_available),
                "solver_converged": bool(self.core.solver.converged),
                "solver_status": int(self.core.solver.status),
                "rho_fixed": self.core.rho_fixed,
            },
        )

    def infer(self, *, covariance: ModelBased) -> InferenceResult:
        """Attach model-based covariance subject to its assumptions and diagnostics."""
        return survival_inference(
            self.core,
            self.context,
            covariance,
            design=self.panel.design,
            fit_statistics=self.fit_statistics(),
        )

    def design_contrast(
        self, term: Term, *, at: float, versus: float
    ) -> DesignContrast:
        """Use the retained basis to contrast two structural predictor values."""
        return self.panel.design.design_contrast(term, at=at, versus=versus)

    def prediction_options(self) -> PredictionOptions:
        """Describe marginal node-level quantities on aligned period graphs."""
        columns = tuple(
            dict.fromkeys(
                (*self.panel.design.input_columns, self.layout.entity, self.layout.time)
            )
        )
        return survival_prediction_options(
            columns,
            spatial=True,
            valid=self.valid,
            condition=(
                "Marginal for each node. Supply aligned period graphs "
                "and known fixed effects."
            ),
        )

    def predict(  # noqa: C901, PLR0912, PLR0914, PLR0915 -- per-period prediction
        self,
        data: SpatialPanelData | PanelData | SpatialSurvivalData,
        value: ArrayLike | None = None,
        *,
        graphs: tuple[tuple[int, SpatialWeights], ...] | None = None,
        kind: Literal[
            "survival",
            "cumulative_hazard",
            "quantile",
            "median",
            "restricted_mean",
            "mean",
        ] = "median",
        grid: bool = False,
    ) -> SpatialPanelPrediction:
        """Predict unconditional node marginals, not conditional on observed spells."""
        if kind not in (
            "survival",
            "cumulative_hazard",
            "quantile",
            "median",
            "restricted_mean",
            "mean",
        ):
            msg = "Unknown spatial survival prediction kind."
            raise ValueError(msg)
        if kind in ("mean", "median"):
            if value is not None or grid:
                msg = f"{kind.capitalize()} prediction does not accept value or grid."
                raise ValueError(msg)
        elif value is None:
            msg = "This prediction kind requires a value."
            raise ValueError(msg)
        if isinstance(data, SpatialSurvivalData):
            data = data.panel
        if isinstance(data, PanelData):
            if (
                data.panel != self.layout
                or self.graph_name is None
                or graphs is not None
            ):
                msg = "Named prediction data requires the fitted layout and graph name."
                raise ValueError(msg)
            graphs = data.period_graphs(self.graph_name)
            data = data.spatial
        if graphs is None:
            graphs = tuple(
                zip(
                    self.panel.period_ids,
                    (p.weights for p in self.panel.periods),
                    strict=True,
                )
            )
        aligned = prepare_spatial_panel_data(
            data, self.layout, graphs, self.panel.design
        )
        effects = None if self.effects is None else self.effects.transform(data.nodes)
        predictions: list[Array] = []
        unit_ids: list[tuple[int, ...]] = []
        for period, design_values, rows in zip(
            aligned.periods, aligned.values, aligned.node_rows, strict=True
        ):
            w = period.weights.values
            if not bool(
                jnp.isfinite(self.core.rho)
                & (jnp.abs(self.core.rho) < period.weights.limit)
            ):
                period_id = aligned.period_ids[len(predictions)]
                msg = (
                    "Fitted rho is outside the admissible spatial parameter "
                    f"range for prediction period {period_id}."
                )
                raise ValueError(msg)
            x = design_values
            if effects is not None:
                x = jnp.column_stack((x, effects[jnp.asarray(rows, dtype=jnp.int32)]))
            columns = self.core.process.slx_columns
            if columns:
                x = jnp.column_stack((x, w @ x[:, jnp.asarray(columns)]))
            p = x.shape[1]
            identity = jnp.eye(w.shape[0], dtype=x.dtype)
            propagation = jnp.linalg.solve(identity - self.core.rho * w, identity)
            mean = propagation @ (x @ self.core.parameters[:p])
            scale = jnp.exp(self.core.parameters[-1]) * jnp.sqrt(
                jnp.sum(propagation**2, axis=1)
            )
            q = jnp.asarray(0.0 if value is None else value)
            if grid:
                if q.ndim != 1:
                    msg = "Prediction grids require a one-dimensional value array."
                    raise ValueError(msg)
                q, mean, scale = q[None, :], mean[:, None], scale[:, None]
            if kind == "mean":
                output = jnp.exp(mean + scale**2 / 2)
                output = jnp.where(jnp.isfinite(output), output, jnp.nan)
            elif kind in ("survival", "cumulative_hazard", "restricted_mean"):
                safe_time = jnp.where(q > 0, q, 1)
                z = (mean - jnp.log(safe_time)) / scale
                log_s = jnp.where(q == 0, 0, log_ndtr(z))
                if kind == "survival":
                    output = jnp.exp(log_s)
                elif kind == "cumulative_hazard":
                    output = -log_s
                else:
                    log_first = mean + scale**2 / 2 + log_ndtr(-z - scale)
                    log_second = jnp.log(safe_time) + log_ndtr(z)
                    output = jnp.where(
                        q == 0, 0, jnp.exp(jnp.logaddexp(log_first, log_second))
                    )
                output = jnp.where(
                    jnp.isfinite(q)
                    & (q >= 0)
                    & ((kind != "restricted_mean") | jnp.isfinite(output)),
                    output,
                    jnp.nan,
                )
            else:
                q = jnp.asarray(0.5, dtype=mean.dtype) if kind == "median" else q
                safe_q = jnp.where((q > 0) & (q < 1), q, 0.5)
                quantile = jnp.exp(mean + scale * ndtri(safe_q))
                output = jnp.where(
                    (q > 0) & (q < 1) & jnp.isfinite(quantile), quantile, jnp.nan
                )
            predictions.append(jnp.where(self.core.valid, output, jnp.nan))
            unit_ids.append(tuple(aligned.unit_ids[i] for i in period.entity_indices))
        return SpatialPanelPrediction(
            aligned.period_ids, tuple(unit_ids), tuple(predictions)
        )


@dataclass(frozen=True)
class _SpatialSurvivalBackend(
    Model[_SpatialSurvivalInput, PreparedSpatialSurvival, SpatialSurvivalResult]
):
    """Lognormal endogenous spatial-lag survival over explicit period graphs."""

    response: RightCensored
    terms: Term
    panel: PanelLayout
    graphs: tuple[tuple[int, SpatialWeights], ...]
    estimator: SpatialSurvivalML = field(default_factory=SpatialSurvivalML)
    effects: Literal["pooled", "unit", "time", "two-way"] = "pooled"
    lag_columns: tuple[str, ...] | None = None
    graph_name: str | None = None

    @override
    def prepare(self, data: _SpatialSurvivalInput, /) -> PreparedSpatialSurvival:
        aligned = prepare_spatial_panel_data(
            data.data, self.panel, self.graphs, self.terms
        )
        if len(data.integrations) != len(aligned.periods):
            msg = "Provide one explicit integration design per period."
            raise ValueError(msg)
        observations = data.data.outcomes
        duration, event = (
            observations.numeric(self.response.time),
            observations.numeric(self.response.event),
        )
        entry = (
            None
            if self.response.entry is None
            else observations.numeric(self.response.entry)
        )
        if entry is not None and bool(jnp.any(entry != 0)):
            msg = "Positive entry is not supported for spatial survival."
            raise ValueError(msg)
        design = aligned.design
        if self.effects in ("unit", "time", "two-way") and design.has_intercept:
            msg = "Fixed effects require a design without an explicit intercept."
            raise ValueError(msg)
        effect_design = prepare_panel_effect_design(
            data.data.nodes,
            self.panel,
            self.effects,
            has_intercept=design.has_intercept,
        )
        effect_values = effect_design.transform(data.data.nodes)
        names = (*design.column_names, *effect_design.column_names)
        if len(set(names)) != len(names):
            msg = "Predictor and effect names must be unique."
            raise ValueError(msg)
        estimator = self.estimator
        columns = (
            select_lag_columns(design.column_names, self.lag_columns, ())
            if self.lag_columns is not None
            else ()
        )
        if len(set(columns)) != len(columns) or any(
            index < 0 or index >= len(names) for index in columns
        ):
            msg = "Positional lag columns must select unique fitted design columns."
            raise ValueError(msg)
        prefix = "W" if self.graph_name is None else f"W[{self.graph_name}]"
        names = (*names, *(f"{prefix}:{names[index]}" for index in columns))
        if len(set(names)) != len(names):
            msg = "Predictor and lag names must be unique."
            raise ValueError(msg)
        periods = tuple(
            prepare_spatial_survival_period(
                graph,
                jnp.column_stack(
                    (values, effect_values[jnp.asarray(node_rows, dtype=jnp.int32)])
                ),
                duration[jnp.asarray(outcome_rows, dtype=jnp.int32)],
                event[jnp.asarray(outcome_rows, dtype=jnp.int32)],
                integration=integration,
                entry_integration=None,
            )
            for graph, values, node_rows, outcome_rows, integration in zip(
                aligned.periods,
                aligned.values,
                aligned.node_rows,
                aligned.outcome_rows,
                data.integrations,
                strict=True,
            )
        )
        problem = SpatialSurvivalProblem(
            periods,
            process=LaggedLogTime(slx_columns=columns),
            graph_name=self.graph_name,
            parameter_names=names,
        )
        return PreparedSpatialSurvival(
            data,
            aligned,
            effect_design,
            problem,
            estimator,
            self.panel,
            self.response.time,
            self._token,
        )

    @override
    def _estimate_prepared(
        self, prepared: PreparedSpatialSurvival
    ) -> SpatialSurvivalResult:
        return _spatial_result(
            prepared,
            prepared.estimator.estimate(prepared.problem),
            self.panel,
            self.response.time,
        )


def refine(
    prepared: PreparedSpatialSurvival,
    /,
    *,
    validation_integrations: tuple[IntegrationDesign, ...],
    config: IntegrationRefinement = IntegrationRefinement(),  # noqa: B008 -- immutable policy
) -> RefinedSpatialSurvivalResult:
    """Refine one retained spatial problem with independent replayable draws."""
    if not isinstance(prepared, PreparedSpatialSurvival):
        msg = "refine requires PreparedSpatialSurvival."
        raise TypeError(msg)
    refinement = refine_spatial_survival(
        prepared.estimator,
        prepared.problem,
        validation_designs=validation_integrations,
        config=config,
    )
    result = _spatial_result(
        prepared,
        refinement.fit,
        prepared.layout,
        prepared.response,
        prepared.problem.graph_name,
    )
    return RefinedSpatialSurvivalResult(result, refinement)


def _spatial_result(
    prepared: PreparedSpatialSurvival,
    fit: SpatialSurvivalFit,
    layout: PanelLayout,
    response: str,
    graph_name: str | None = None,
) -> SpatialSurvivalResult:
    return SpatialSurvivalResult(
        fit,
        FitContext(prepared.data.data.outcomes, "Regression", response),
        prepared.panel,
        layout,
        prepared.effects,
        graph_name,
    )
