"""Host lifecycle for observed-data spatial Gaussian panels."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, override

import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from statadic.api.base import FitContext, Model, ParameterResult
from statadic.api.data import SpatialPanelData
from statadic.api.fit_statistics import LikelihoodSummary
from statadic.api.inference import InferenceResult
from statadic.api.panel import (
    prepare_spatial_panel_data,
    select_lag_columns,
)
from statadic.api.prediction import (  # noqa: TC001
    PredictionOptions,
    _spatial_prediction_options,
)
from statadic.api.reporting import Metric, ParameterRow, Report, _finite_metric
from statadic.api.spatial.cross_section import (
    _SPATIAL_LIKELIHOOD_REASON,
    SpatialInformation,
)
from statadic.api.spatial.graphs import PanelData
from statadic.api.spatial.prediction import SpatialPanelPrediction
from statadic.estimation.spatial import (
    SAR,  # noqa: TC001 -- runtime dataclass annotations
    SEM,  # noqa: TC001 -- runtime dataclass annotations
    ObservedPanelML,
    ObservedSpatialPanelFit,
    ObservedSpatialPanelProblem,
)
from statadic.inference.spatial import spatial_covariance, spatial_panel_impacts

if TYPE_CHECKING:
    from statadic.api.data import PanelLayout
    from statadic.api.design import Term
    from statadic.api.panel import (
        PreparedSpatialPanelData,
    )
    from statadic.inference.spatial import SpatialPanelImpacts
    from statadic.operators.spatial import SpatialWeights


__all__ = ["ObservedSpatialPanelResult", "PreparedObservedSpatialPanel"]


@dataclass(frozen=True)
class PreparedObservedSpatialPanel:
    """Aligned graph and spell rows with producing model identity."""

    data: SpatialPanelData
    panel: PreparedSpatialPanelData
    model_token: object = field(repr=False)


@dataclass(frozen=True)
class ObservedSpatialPanelResult(ParameterResult[ObservedSpatialPanelFit]):
    """Fitted actual graphs and observed-outcome likelihood."""

    panel: PreparedSpatialPanelData = field(repr=False)
    layout: PanelLayout
    response: str
    graph_name: str | None = None

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the coefficient names used by ``report``."""
        return self.core.parameter_names or tuple(
            f"parameter_{i}" for i in range(len(self.core.parameters))
        )

    @override
    def likelihood_summary(self) -> LikelihoodSummary:
        """Return the observed-data Gaussian likelihood summary.

        ``log_likelihood`` is the full marginal Gaussian likelihood over the
        observed nodes with the ``2 pi`` term and covariance log-determinant.
        ``k`` is the mean design rank (base, SLX and any explicit unit or time
        effect columns), the spatial parameter, the scale and, for random
        effects, the variance ratio. ``n`` counts observed outcomes.
        """
        core = self.core
        problem = core.problem
        periods = len(problem.periods)
        coefficients = (
            int(jnp.asarray(problem.design[0]).shape[-1])
            + len(problem.slx_columns)
            + (problem.n_units if core.effect_type in ("unit", "two-way") else 0)
            + (
                periods
                if core.effect_type == "time"
                else periods - 1
                if core.effect_type == "two-way"
                else 0
            )
        )
        return LikelihoodSummary(
            float(core.log_likelihood) if bool(core.valid) else math.nan,
            coefficients + 2 + (core.effect_type == "random"),
            sum(len(period.observed_indices) for period in problem.periods),
            _SPATIAL_LIKELIHOOD_REASON,
        )

    @override
    def report(self) -> Report:
        valid = bool(self.core.valid)
        statistics = self.fit_statistics()
        names = self.core.parameter_names or tuple(
            f"parameter_{i}" for i in range(len(self.core.parameters))
        )
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
                for i, name in enumerate(names)
            ),
            metrics=(
                Metric("n_observations", int(self.context.observations.n_rows)),
                _finite_metric(
                    "sigma2", self.core.sigma2, "Profiled innovation variance."
                ),
                *statistics.metrics,
            ),
            notes=statistics.notes,
            metadata={
                "fit_valid": valid,
                "identified": bool(self.core.identified),
                "solver_converged": bool(self.core.solver.converged),
                "solver_status": int(self.core.solver.status),
                "effects": self.core.effect_type,
                "information_valid": bool(self.core.information_valid),
            },
        )

    def infer(self, *, covariance: SpatialInformation) -> InferenceResult:
        """Attach observed information to the actual fitted panel."""
        if not isinstance(covariance, SpatialInformation):
            msg = "Observed panel inference requires SpatialInformation."
            raise TypeError(msg)
        return InferenceResult(
            spatial_covariance(
                self.core, method="observed", debiased=covariance.debiased
            ),
            self.context,
            covariance,
            design=self.panel.design,
            fit_statistics=self.fit_statistics(),
        )

    def impacts(self) -> SpatialPanelImpacts:
        """Return direct, indirect and total effects on each actual graph."""
        return spatial_panel_impacts(self.core)

    def prediction_options(self) -> PredictionOptions:
        """Describe targets across actual graphs and observed neighbor outcomes."""
        return _spatial_prediction_options(
            tuple(
                dict.fromkeys(
                    (
                        *self.panel.design.input_columns,
                        self.layout.entity,
                        self.layout.time,
                    )
                )
            ),
            response=self.response if self.core.lag else None,
            condition=(
                "Supply aligned node and outcome tables, dense period graphs "
                "and known fixed effects."
            ),
            valid=self.valid,
        )

    def predict(  # noqa: C901, PLR0912, PLR0914, PLR0915 -- explicit panel roles
        self,
        data: SpatialPanelData | PanelData,
        *,
        graphs: tuple[tuple[int, SpatialWeights], ...] | None = None,
        kind: Literal["reduced-form", "structural"] = "reduced-form",
    ) -> SpatialPanelPrediction:
        """Predict full graph means; structural means require observed neighbours.

        Random-effect reduced-form prediction is the marginal population mean
        (zero random-effect mean), not a conditional empirical-Bayes predictor.
        """
        if isinstance(data, PanelData):
            if (
                data.panel != self.layout
                or self.graph_name is None
                or graphs is not None
            ):
                msg = (
                    "Named prediction requires the fitted panel layout and graph name."
                )
                raise ValueError(msg)
            graphs = data.period_graphs(self.graph_name)
            data = data.spatial
        if kind not in ("reduced-form", "structural"):
            msg = "Prediction kind must be reduced-form or structural."
            raise ValueError(msg)
        if graphs is None:
            graphs = tuple(
                zip(
                    self.panel.period_ids,
                    (p.weights for p in self.panel.periods),
                    strict=True,
                )
            )
        # Apply the fitted feature transform; do not refit categories/landmarks.
        aligned = prepare_spatial_panel_data(
            data, self.layout, graphs, self.panel.design
        )
        values = aligned.values
        fit = self.core
        beta = fit.parameters[:-1]
        rho = fit.parameters[-1] if fit.lag else jnp.zeros_like(fit.parameters[-1])
        fitted_units = {key: i for i, key in enumerate(self.panel.unit_ids)}
        fitted_periods = {key: i for i, key in enumerate(self.panel.period_ids)}
        outcome = (
            data.outcomes.numeric(self.response)
            if kind == "structural" and fit.lag
            else None
        )
        predictions = []
        unit_ids = []
        for t, (period, x, rows) in enumerate(
            zip(aligned.periods, values, aligned.outcome_rows, strict=True)
        ):
            w = period.weights.values
            if not bool(
                jnp.isfinite(fit.parameters[-1])
                & (jnp.abs(fit.parameters[-1]) < period.weights.limit)
            ):
                msg = (
                    "Fitted spatial dependence is outside the admissible "
                    f"range for prediction period {aligned.period_ids[t]}."
                )
                raise ValueError(msg)
            if isinstance(w, BCOO):
                msg = "Prediction requires dense period weights."
                raise TypeError(msg)
            ids = tuple(aligned.unit_ids[i] for i in period.entity_indices)
            unit_ids.append(ids)
            augmented = (
                jnp.column_stack((x, w @ x[:, jnp.asarray(fit.slx_columns)]))
                if fit.slx_columns
                else x
            )
            mean = augmented @ beta
            if fit.effect_type in ("unit", "two-way"):
                if any(unit not in fitted_units for unit in ids):
                    msg = "Fixed unit prediction requires known unit levels."
                    raise ValueError(msg)
                mean += fit.effects[jnp.asarray([fitted_units[unit] for unit in ids])]
            if fit.effect_type in ("time", "two-way"):
                period_id = aligned.period_ids[t]
                if period_id not in fitted_periods:
                    msg = "Fixed time prediction requires known period levels."
                    raise ValueError(msg)
                index = fitted_periods[period_id]
                if fit.effect_type == "time":
                    mean += fit.effects[index]
                elif index:
                    mean += fit.effects[len(fitted_units) + index - 1]
            if kind == "reduced-form":
                mean = jnp.linalg.solve(
                    jnp.eye(w.shape[0], dtype=mean.dtype) - rho * w, mean
                )
            elif fit.lag:
                if outcome is None:
                    msg = "Structural prediction requires observed outcomes."
                    raise ValueError(msg)
                present = jnp.zeros(len(ids), dtype=bool)
                present = present.at[jnp.asarray(period.observed_indices)].set(True)
                if jnp.any(jnp.asarray(w)[:, ~present] != 0):
                    msg = "Structural prediction requires every neighbour outcome."
                    raise ValueError(msg)
                y = (
                    jnp.zeros((len(ids),), dtype=mean.dtype)
                    .at[jnp.asarray(period.observed_indices, dtype=jnp.int32)]
                    .set(outcome[jnp.asarray(rows, dtype=jnp.int32)])
                )
                mean += rho * (w @ y)
            predictions.append(jnp.where(fit.valid, mean, jnp.nan))
        return SpatialPanelPrediction(
            aligned.period_ids, tuple(unit_ids), tuple(predictions)
        )


@dataclass(frozen=True)
class _ObservedSpatialPanelBackend(
    Model[SpatialPanelData, PreparedObservedSpatialPanel, ObservedSpatialPanelResult]
):
    """Fit Gaussian spatial panels with full graphs and observed outcome subsets."""

    response: str
    terms: Term
    panel: PanelLayout
    graphs: tuple[tuple[int, SpatialWeights], ...]
    process: SAR | SEM
    effects: Literal["pooled", "unit", "time", "two-way", "random"]
    graph_name: str | None = None
    estimator: ObservedPanelML = field(default_factory=ObservedPanelML)
    lag_columns: tuple[str, ...] | None = None

    @override
    def prepare(self, data: SpatialPanelData, /) -> PreparedObservedSpatialPanel:
        aligned = prepare_spatial_panel_data(data, self.panel, self.graphs, self.terms)
        if self.response not in data.outcomes.columns:
            msg = "Outcome column is missing."
            raise ValueError(msg)
        if self.effects in ("unit", "time", "two-way") and aligned.design.has_intercept:
            msg = "Fixed spatial effects require a design without intercept."
            raise ValueError(msg)
        return PreparedObservedSpatialPanel(data, aligned, self._token)

    @override
    def _estimate_prepared(
        self, prepared: PreparedObservedSpatialPanel
    ) -> ObservedSpatialPanelResult:
        aligned = prepared.panel
        columns = (
            select_lag_columns(aligned.design.column_names, self.lag_columns, ())
            if self.lag_columns is not None
            else ()
        )
        outcome = prepared.data.outcomes.numeric(self.response)
        problem = ObservedSpatialPanelProblem(
            aligned.periods,
            aligned.values,
            tuple(
                outcome[jnp.asarray(rows, dtype=jnp.int32)]
                for rows in aligned.outcome_rows
            ),
            len(aligned.unit_ids),
            process=self.process,
            effects=self.effects,
            slx_columns=columns,
            graph_name=self.graph_name,
            parameter_names=aligned.design.column_names,
        )
        return ObservedSpatialPanelResult(
            self.estimator.estimate(problem),
            FitContext(prepared.data.outcomes, "Observed spatial panel", self.response),
            aligned,
            self.panel,
            self.response,
            self.graph_name,
        )
