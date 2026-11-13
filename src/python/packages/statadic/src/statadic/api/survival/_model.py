"""Right-censored spell regression using STATADIC's independent AFT estimator."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, overload, override

import jax.numpy as jnp

from statadic.api.base import (
    FitContext,
    Model,
    ParameterResult,
)
from statadic.api.data import MeasureColumn, Observations
from statadic.api.design import FittedDesign
from statadic.api.fit_statistics import LikelihoodSummary
from statadic.api.inference import InferenceResult, ModelBased  # noqa: TC001
from statadic.api.inputs import (  # noqa: TC001 -- public runtime annotation
    RegressionInput,
    _matching_predictor,
    _node_predictor,
    as_observations,
)
from statadic.api.panel import (
    panel_input_roles,
    prepare_panel_effect_design,
)
from statadic.api.prediction import PredictionOptions  # noqa: TC001
from statadic.api.reporting import Metric, ParameterRow, Report
from statadic.api.spatial.graphs import (  # noqa: TC001 -- checked constructor annotations
    PanelData,
    SpatialError,
    SpatialLag,
    SpatialLagError,
    split_spatial_terms,
)
from statadic.api.survival._inputs import (
    RightCensored,
    SpatialSurvivalData,
    _SpatialSurvivalInput,
)
from statadic.api.survival._prediction import survival_prediction_options
from statadic.api.survival._spatial import (
    PreparedSpatialSurvival,
    SpatialSurvivalResult,
    _spatial_result,
    _SpatialSurvivalBackend,
)
from statadic.api.survival.inference import survival_inference
from statadic.estimation.survival import (
    SpatialSurvivalML,
    SurvivalFit,
    SurvivalML,
    SurvivalProblem,
)
from statadic.model.survival import (
    LogLogisticAFT,
    LogNormalAFT,
    cumulative_hazard,
    restricted_mean,
    survival_mean,
    survival_probability,
    survival_quantile,
)

if TYPE_CHECKING:
    from statadic.api.data import PanelLayout
    from statadic.api.design import DesignContrast, Term
    from statadic.api.panel import (
        PanelEffectDesign,
    )
    from statadic.core.typing import Array, ArrayLike
    from statadic.model.survival import AFTFamily


@dataclass(frozen=True)
class PreparedSurvival:
    """Training feature state, spell outcomes and panel effects."""

    values: Array
    design: FittedDesign
    effects: PanelEffectDesign | None
    context: FitContext
    problem: SurvivalProblem
    model_token: object = field(repr=False)


@dataclass(frozen=True)
class SurvivalResult(ParameterResult[SurvivalFit]):
    """AFT candidate with retained design; inference is a separate operation."""

    design: FittedDesign = field(repr=False)
    effects: PanelEffectDesign | None = field(repr=False)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the coefficient names used by ``report``."""
        return self.core.parameter_names or (
            *self.design.column_names,
            "log_ancillary",
        )

    @override
    def likelihood_summary(self) -> LikelihoodSummary:
        """Return the conditional spell likelihood summary behind the criteria.

        The log likelihood keeps every density constant (the log-normal ``2 pi``
        term and the original-time Jacobian). ``k`` counts the coefficients,
        including any fixed-effect columns, plus the one estimated log shape or
        log scale. ``n`` is the number of spells.
        """
        return LikelihoodSummary(
            float(self.core.log_likelihood),
            int(self.core.parameters.shape[-1]),
            int(self.context.observations.n_rows),
            "The survival fit is invalid, so its log likelihood is unavailable.",
        )

    @override
    def report(self) -> Report:
        """Describe numerical fit evidence without claiming global optimality."""
        fit = self.core
        statistics = self.fit_statistics()
        valid = bool(fit.valid)
        names = fit.parameter_names or (*self.design.column_names, "log_ancillary")
        return Report(
            title=f"{self.context.model_name} fit",
            parameters=tuple(
                ParameterRow(
                    name,
                    float(fit.parameters[i]) if valid else None,
                    valid=valid,
                    status="available" if valid else "unavailable",
                    unavailable_reason=None if valid else "invalid-fit",
                )
                for i, name in enumerate(names)
            ),
            metrics=(
                Metric("n_observations", int(self.context.observations.n_rows)),
                *statistics.metrics,
            ),
            notes=statistics.notes,
            metadata={
                "fit_valid": valid,
                "identified": bool(fit.identified),
                "solver_converged": bool(fit.solver.converged),
                "solver_status": int(fit.solver.status),
                "existence_status": int(fit.existence_status),
                "family": type(fit.family).__name__,
            },
            warnings=("A locally valid fit does not certify a finite global optimum.",),
        )

    def infer(self, *, covariance: ModelBased) -> InferenceResult:
        """Attach model-based covariance subject to its assumptions and diagnostics."""
        return survival_inference(
            self.core,
            self.context,
            covariance,
            design=self.design,
            fit_statistics=self.fit_statistics(),
        )

    def design_contrast(
        self, term: Term, *, at: float, versus: float
    ) -> DesignContrast:
        """Use the retained basis to contrast two structural predictor values."""
        return self.design.design_contrast(term, at=at, versus=versus)

    def prediction_options(self) -> PredictionOptions:
        """Describe each fitted duration target and its conditioning requirements."""
        columns = tuple(self.design.input_columns)
        condition = ""
        if self.effects is not None:
            columns = tuple(
                dict.fromkeys(
                    (*columns, self.effects.layout.entity, self.effects.layout.time)
                )
            )
            condition = "Supply trained fixed-effect levels."
        divergent = (
            isinstance(self.core.family, LogLogisticAFT)
            and self.core.parameters[-1] <= 0
        )
        return survival_prediction_options(
            columns,
            spatial=False,
            valid=self.valid,
            condition=condition,
            mean_reason="The fitted log-logistic shape is at most one."
            if divergent
            else None,
        )

    def predict(  # noqa: C901, PLR0912, PLR0915 -- explicit prediction and grid roles
        self,
        data: RegressionInput | PanelData,
        value: ArrayLike | None = None,
        *,
        kind: Literal[
            "survival",
            "cumulative_hazard",
            "quantile",
            "restricted_mean",
            "mean",
            "median",
        ] = "median",
        entry: ArrayLike | None = None,
        grid: bool = False,
    ) -> Array:
        """Predict at elapsed time, quantile probability or restricted-mean horizon.

        Entry conditioning is supported for survival and cumulative hazard.
        Fixed effects require levels observed during preparation.
        """
        if kind == "median":
            if value is not None or grid or entry is not None:
                msg = "Median prediction does not accept value, grid or entry."
                raise ValueError(msg)
            kind, value = "quantile", jnp.asarray(0.5)
        if isinstance(data, PanelData):
            data = _independent_panel_rows(data, self.design)
        else:
            roles = (
                self.design.input_columns
                if self.effects is None
                else panel_input_roles(self.design.input_columns, self.effects.layout)
            )
            data = as_observations(data, roles, broadcast_scalars=True)
        if kind == "mean":
            if value is not None or grid or entry is not None:
                msg = "Mean prediction does not accept value, grid or entry."
                raise ValueError(msg)
        elif value is None:
            msg = "This prediction kind requires a value."
            raise ValueError(msg)
        design = self.design.transform(data)
        if self.effects is not None:
            design = jnp.column_stack((design, self.effects.transform(data)))
        if grid:
            times = jnp.asarray(value)
            if times.ndim != 1:
                msg = "Prediction grids require a one-dimensional value array."
                raise ValueError(msg)
            value = times[None, :]
            design = design[:, None, :]
            if entry is not None:
                entry_array = jnp.asarray(entry)
                if entry_array.ndim == 1:
                    if entry_array.shape != (data.n_rows,):
                        msg = "Entry values must align with prediction observations."
                        raise ValueError(msg)
                    entry = entry_array[:, None]
                elif entry_array.ndim != 0:
                    msg = "Entry values must be scalar or one per observation."
                    raise ValueError(msg)
        fit = self.core
        if entry is not None and kind not in ("survival", "cumulative_hazard"):
            msg = (
                "Entry conditioning is only defined for survival and cumulative hazard."
            )
            raise ValueError(msg)
        if kind == "mean":
            result = survival_mean(fit.parameters, design, fit.family)
        elif value is None:
            msg = "This prediction kind requires a value."
            raise ValueError(msg)
        elif kind == "survival":
            result = survival_probability(
                fit.parameters, design, value, fit.family, entry=entry
            )
        elif kind == "cumulative_hazard":
            result = cumulative_hazard(
                fit.parameters, design, value, fit.family, entry=entry
            )
        elif kind == "quantile":
            result = survival_quantile(fit.parameters, design, value, fit.family)
        elif kind == "restricted_mean":
            result = restricted_mean(fit.parameters, design, value, fit.family)
        else:
            msg = "Unknown survival prediction kind."
            raise ValueError(msg)
        return jnp.where(fit.valid, result, jnp.nan)


def _panel_node_positions(data: PanelData) -> tuple[int, ...]:
    """Match unique observed entity/period keys to their full-node positions."""

    def keys(rows: Observations) -> tuple[tuple[int | str, int | str], ...]:
        return tuple(
            zip(
                rows.grouping_labels(data.panel.entity),
                rows.grouping_labels(data.panel.time),
                strict=True,
            )
        )

    node_keys = keys(data.nodes)
    observed_keys = keys(data.spatial.outcomes)
    index = {key: i for i, key in enumerate(node_keys)}
    if len(index) != len(node_keys) or len(set(observed_keys)) != len(observed_keys):
        message = "Node and observed spell entity/period keys must be unique."
        raise ValueError(message)
    if any(key not in index for key in observed_keys):
        message = "Every independent spell must match a full-node entity/period key."
        raise ValueError(message)
    return tuple(index[key] for key in observed_keys)


def _independent_panel_rows(
    data: PanelData, terms: Term | FittedDesign
) -> Observations:
    """Align node predictors and reject conflicting duplicates on observed spells.

    Outcomes own response/event/entry roles and their row identities. Required
    predictors always come from full nodes, as they do for spatial survival.
    """
    observed = data.spatial.outcomes
    positions = _panel_node_positions(data)
    numeric = dict(observed.columns)
    categorical = dict(observed.labels)
    measures = dict(observed.measures)
    for name, role in terms.input_columns.items():
        value = _node_predictor(data.nodes, name, role, positions, observed.row_ids)
        duplicate = name in observed.available_columns
        if duplicate and not _matching_predictor(observed, name, value):
            message = f"Outcome predictor {name!r} conflicts with its full-node values."
            raise ValueError(message)
        if isinstance(value, MeasureColumn):
            measures[name] = value
        elif isinstance(value, tuple):
            categorical[name] = value
        else:
            numeric[name] = value
    return Observations(
        numeric,
        row_ids=observed.row_ids,
        labels=categorical,
        measures=measures,
        source=observed.source,
        source_rows=observed.source_rows,
    )


@dataclass(frozen=True)
class _SurvivalBackend(
    Model[
        Observations | PanelData | SpatialSurvivalData,
        PreparedSurvival | PreparedSpatialSurvival,
        SurvivalResult | SpatialSurvivalResult,
    ]
):
    """Right-censored AFT regression with optional endogenous spatial lag."""

    response: RightCensored
    terms: Term
    estimator: SurvivalML | SpatialSurvivalML | None = None
    panel: PanelLayout | None = None
    effects: Literal["pooled", "unit", "time", "two-way"] = "pooled"
    family: AFTFamily | None = field(default=None, kw_only=True)
    dependence: SpatialLag | SpatialError | SpatialLagError | None = field(
        default=None, kw_only=True
    )

    if TYPE_CHECKING:
        # At runtime the frozen dataclass builds this initializer. Its two
        # supported call forms require either an estimator or a named family.
        @overload
        def __init__(
            self,
            response: RightCensored,
            terms: Term,
            estimator: SurvivalML | SpatialSurvivalML,
            panel: PanelLayout | None = ...,
            effects: Literal["pooled", "unit", "time", "two-way"] = ...,
            *,
            family: AFTFamily | None = ...,
            dependence: SpatialLag | SpatialError | SpatialLagError | None = ...,
        ) -> None: ...

        @overload
        def __init__(
            self,
            response: RightCensored,
            terms: Term,
            estimator: SurvivalML | SpatialSurvivalML | None = ...,
            panel: PanelLayout | None = ...,
            effects: Literal["pooled", "unit", "time", "two-way"] = ...,
            *,
            family: AFTFamily,
            dependence: SpatialLag | SpatialError | SpatialLagError | None = ...,
        ) -> None: ...

        def __init__(self, *args: object, **kwargs: object) -> None: ...

    def __post_init__(self) -> None:
        """Resolve the estimation method and reject unsupported likelihoods."""
        if self.dependence is not None:
            if not isinstance(self.dependence, SpatialLag):
                msg = "Spatial survival supports SpatialLag only."
                raise ValueError(msg)
            if self.panel is None or not isinstance(self.family, LogNormalAFT):
                msg = "SpatialLag survival requires a PanelLayout and LogNormalAFT."
                raise ValueError(msg)
            if self.estimator is not None and not isinstance(
                self.estimator, SpatialSurvivalML
            ):
                msg = "SpatialLag survival requires SpatialSurvivalML."
                raise TypeError(msg)
        elif self.estimator is None:
            if self.family is None:
                msg = "Specify an AFT family or a SurvivalML estimator."
                raise TypeError(msg)
        elif not isinstance(self.estimator, SurvivalML) or (
            self.family is not None and self.family != self.estimator.family
        ):
            msg = "Independent survival requires a matching SurvivalML family."
            raise ValueError(msg)

    @overload
    def prepare(self, data: Observations, /) -> PreparedSurvival: ...

    @overload
    def prepare(self, data: SpatialSurvivalData, /) -> PreparedSpatialSurvival: ...

    @overload
    def prepare(
        self,
        data: Observations | PanelData | SpatialSurvivalData,
        /,
    ) -> PreparedSurvival | PreparedSpatialSurvival: ...

    @override
    def prepare(  # noqa: C901 -- independent and graph-aligned input branches
        self,
        data: Observations | PanelData | SpatialSurvivalData,
        /,
    ) -> PreparedSurvival | PreparedSpatialSurvival:
        """Resolve named weights and prepare the existing family-specific problem."""
        integrations = ()
        if isinstance(data, SpatialSurvivalData):
            if self.dependence is None:
                message = "Independent survival does not consume integration designs."
                raise ValueError(message)
            integrations = data.integrations
            data = data.panel
        elif self.dependence is not None:
            message = "Spatial survival requires SpatialSurvivalData."
            raise TypeError(message)
        if isinstance(data, PanelData):
            independent_rows = _independent_panel_rows(data, self.terms)
            if self.dependence is None:
                if self.panel != data.panel:
                    msg = "Model and data panel layouts must match."
                    raise ValueError(msg)
                return self._prepare_independent(independent_rows)
            if self.panel != data.panel:
                msg = "Model and data panel layouts must match."
                raise ValueError(msg)
            graph = self.dependence.graph
            terms, lags = split_spatial_terms(self.terms, graph)
            estimator = self.estimator or SpatialSurvivalML()
            if not isinstance(estimator, SpatialSurvivalML):
                msg = "Spatial survival dependence requires a SpatialSurvivalML."
                raise TypeError(msg)
            backend = _SpatialSurvivalBackend(
                self.response,
                terms,
                data.panel,
                data.period_graphs(graph),
                estimator,
                self.effects,
                lags or None,
                graph_name=graph,
            )
            prepared = backend.prepare(
                _SpatialSurvivalInput(data.spatial, integrations)
            )
            # Propagation by invertible (I - rho W_t) cannot recover a
            # direction already lost on the full graph. Censoring and missing
            # outcomes can reduce information further; the fit checks those.
            columns = prepared.problem.process.slx_columns
            blocks = tuple(
                jnp.column_stack(
                    (
                        p.design,
                        p.graph.weights.values @ p.design[:, jnp.asarray(columns)],
                    )
                )
                if columns
                else p.design
                for p in prepared.problem.periods
            )
            full_design = jnp.concatenate(blocks)
            if jnp.linalg.matrix_rank(full_design) < full_design.shape[1]:
                msg = "Spatial survival design is rank deficient on the full graph."
                raise ValueError(msg)
            return replace(prepared, model_token=self._token)
        if self.dependence is not None or integrations:
            msg = (
                "Spatial survival requires named PanelData and per-period integrations."
            )
            raise ValueError(msg)
        return self._prepare_independent(data)

    def _prepare_independent(self, data: Observations) -> PreparedSurvival:
        design = FittedDesign(self.terms._prepare_term(data))
        values = design.transform(data)
        duration = data.numeric(self.response.time)
        event = data.numeric(self.response.event)
        entry = (
            None if self.response.entry is None else data.numeric(self.response.entry)
        )
        if not bool(jnp.all((event == 0) | (event == 1))):
            msg = "Event indicators must be 0 or 1."
            raise ValueError(msg)
        if not bool(jnp.all(duration > 0)) or (
            entry is not None and not bool(jnp.all((entry >= 0) & (entry < duration)))
        ):
            msg = "Durations must be positive; require 0 <= entry < duration."
            raise ValueError(msg)
        if self.panel is None and self.effects != "pooled":
            msg = "Fixed panel effects require a PanelLayout."
            raise ValueError(msg)
        if self.effects in ("unit", "time", "two-way") and design.has_intercept:
            msg = "Fixed effects require a design without an explicit intercept."
            raise ValueError(msg)
        effect_design = None
        names = design.column_names
        if self.panel is not None:
            effect_design = prepare_panel_effect_design(
                data, self.panel, self.effects, has_intercept=design.has_intercept
            )
            values = jnp.column_stack((values, effect_design.transform(data)))
            names = (*names, *effect_design.column_names)
        if len(set(names)) != len(names):
            msg = "Predictor and effect names must be unique."
            raise ValueError(msg)
        if jnp.linalg.matrix_rank(jnp.asarray(values)) != values.shape[1]:
            msg = "Prepared survival design is rank deficient."
            raise ValueError(msg)
        problem = SurvivalProblem(values, duration, event, entry, parameter_names=names)
        return PreparedSurvival(
            values,
            design,
            effect_design,
            FitContext(data, "Regression", self.response.time),
            problem,
            self._token,
        )

    @overload
    def _estimate_prepared(self, prepared: PreparedSurvival) -> SurvivalResult: ...

    @overload
    def _estimate_prepared(
        self, prepared: PreparedSpatialSurvival
    ) -> SpatialSurvivalResult: ...

    @overload
    def _estimate_prepared(
        self, prepared: PreparedSurvival | PreparedSpatialSurvival
    ) -> SurvivalResult | SpatialSurvivalResult: ...

    @override
    def _estimate_prepared(
        self, prepared: PreparedSurvival | PreparedSpatialSurvival
    ) -> SurvivalResult | SpatialSurvivalResult:
        """Estimate the exact prepared problem using its resolved estimator."""
        if isinstance(prepared, PreparedSpatialSurvival):
            if self.panel is None or self.dependence is None:
                msg = "Spatial preparation requires a spatial specification."
                raise ValueError(msg)
            return _spatial_result(
                prepared,
                prepared.estimator.estimate(prepared.problem),
                self.panel,
                self.response.time,
                self.dependence.graph,
            )
        if self.estimator is not None:
            estimator = self.estimator
        elif self.family is not None:
            estimator = SurvivalML(self.family)
        else:
            msg = "An independent AFT fit requires a family."
            raise TypeError(msg)
        if not isinstance(estimator, SurvivalML):
            msg = "Independent preparation requires a SurvivalML estimator."
            raise TypeError(msg)
        return SurvivalResult(
            estimator.estimate(prepared.problem),
            prepared.context,
            prepared.design,
            prepared.effects,
        )


__all__ = [
    "PreparedSurvival",
    "RightCensored",
    "SurvivalResult",
    "_SpatialSurvivalInput",
]
