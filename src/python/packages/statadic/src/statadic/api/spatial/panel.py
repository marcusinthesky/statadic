"""Host workflows for balanced spatial panel regression."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, override

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from statadic.api.base import (
    ParameterResult,
    PreparedRegression,  # noqa: TC001
    RegressionResult,
    _RegressionPreparation,
)
from statadic.api.data import (  # noqa: TC001 -- runtime annotations
    Observations,
    PanelLayout,
)
from statadic.api.design import FittedDesign  # noqa: TC001
from statadic.api.fit_statistics import LikelihoodSummary
from statadic.api.inference import (
    InferenceResult,
    ModelBased,
)
from statadic.api.panel import (
    panel_input_roles,
    select_lag_columns,
    spatial_panel_rows,
)
from statadic.api.prediction import (  # noqa: TC001
    PredictionOptions,
    _spatial_prediction_options,
)
from statadic.api.reporting import (
    Metric,
    ParameterRow,
    Report,
    _finite_metric,
)
from statadic.api.roles import InputRole  # noqa: TC001 -- runtime annotations
from statadic.api.spatial.cross_section import (
    _SPATIAL_LIKELIHOOD_REASON,
    SpatialInformation,
)
from statadic.api.spatial.graphs import PanelData
from statadic.api.spatial.prediction import SpatialPanelPrediction
from statadic.estimation.spatial import (
    SAC,  # noqa: TC001 -- runtime dataclass annotations
    SAR,  # noqa: TC001 -- runtime dataclass annotations
    SEM,  # noqa: TC001 -- runtime dataclass annotations
    KKPPanelFit,
    SpatialPanelFit,
    SpatialPanelProblem,
)
from statadic.inference.linear_covariance import (
    classical_covariance,
)
from statadic.inference.spatial import (
    spatial_covariance,
)
from statadic.operators.spatial import (
    SpatialPanelWeights,
    spatial_panel_lag,
)

if TYPE_CHECKING:
    from statadic.api.base import Estimator

__all__ = ["KKPPanelResult", "SpatialPanelResult"]


_RESERVED = ("rho", "lambda")


@dataclass(frozen=True)
class SpatialPanelResult(RegressionResult[SpatialPanelFit]):
    """Gaussian SAR panel result with its native fitted state."""

    panel: PanelLayout | None = None
    graph_name: str | None = None

    def prediction_options(self) -> PredictionOptions:
        """Describe targets on the fitted time-major graph roster."""
        keys = () if self.panel is None else (self.panel.entity, self.panel.time)
        return _spatial_prediction_options(
            tuple(dict.fromkeys((*self.design.input_columns, *keys))),
            response=self.context.response if self.core.lag else None,
            unavailable_reason="No retained panel layout."
            if self.panel is None
            else None,
            condition="Use the fitted time-major graph identities.",
            valid=self.valid,
        )

    def predict(  # noqa: C901, PLR0912, PLR0914, PLR0915 -- explicit graph-keyed panel prediction
        self,
        data: Observations | PanelData,
        *,
        kind: Literal["reduced-form", "structural"] = "reduced-form",
    ) -> SpatialPanelPrediction:
        """Predict in retained time-major graph order with explicit mean semantics.

        Structural prediction uses all supplied neighbour outcomes. Random
        effects use their zero population mean, not a conditional BLUP.
        """
        weights = self.core.weights
        if isinstance(data, PanelData):
            if data.panel != self.panel or self.graph_name is None:
                msg = (
                    "Named prediction requires the fitted panel layout and graph name."
                )
                raise ValueError(msg)
            graphs = data.period_graphs(self.graph_name)
            period_ids = tuple(
                int(t) for t in jax.device_get(weights.period_ids).tolist()
            )
            unit_ids = tuple(int(u) for u in jax.device_get(weights.unit_ids).tolist())
            if tuple(t for t, _ in graphs) != period_ids or any(
                tuple(int(u) for u in jax.device_get(graph.unit_ids).tolist())
                != unit_ids
                for _, graph in graphs
            ):
                msg = (
                    "Prediction rows must match the fitted graph periods "
                    "and unit identities."
                )
                raise ValueError(msg)
            if any(isinstance(graph.values, BCOO) for _, graph in graphs):
                msg = "Balanced spatial panel prediction requires dense graphs."
                raise TypeError(msg)
            weights = SpatialPanelWeights(
                jnp.stack([jnp.asarray(graph.values) for _, graph in graphs]),
                weights.unit_ids,
                weights.period_ids,
                jnp.min(jnp.stack([graph.limit for _, graph in graphs])),
            )
            rosters = tuple((period, unit_ids) for period in period_ids)
            rows = spatial_panel_rows(data.spatial, data.panel, rosters)
            order = tuple(i for block in rows.node_rows for i in block)
            nodes = data.nodes.take(order)
            if kind == "structural" and self.core.lag and data.outcomes is not None:
                if any(len(block) != len(unit_ids) for block in rows.outcome_rows):
                    msg = "Structural prediction requires every neighbour outcome."
                    raise ValueError(msg)
                outcome_order = jnp.asarray(
                    [i for block in rows.outcome_rows for i in block]
                )
                nodes = Observations(
                    {
                        **nodes.columns,
                        self.context.response: data.outcomes.numeric(
                            self.context.response
                        )[outcome_order],
                    },
                    row_ids=nodes.row_ids,
                    labels=nodes.labels,
                    measures=nodes.measures,
                )
            data = nodes
        if self.panel is None:
            msg = "Prediction requires a retained panel layout."
            raise ValueError(msg)
        if kind not in ("reduced-form", "structural"):
            msg = "Prediction kind must be reduced-form or structural."
            raise ValueError(msg)
        periods = int(weights.period_ids.shape[0])
        units = int(weights.unit_ids.shape[0])
        entity = tuple(jax.device_get(weights.unit_ids).tolist()) * periods
        time = tuple(jax.device_get(jnp.repeat(weights.period_ids, units)).tolist())
        if (
            data.grouping_labels(self.panel.entity) != entity
            or data.grouping_labels(self.panel.time) != time
        ):
            msg = "Prediction rows must match the fitted time-major graph identities."
            raise ValueError(msg)
        x = self.design.transform(data).reshape(periods, units, -1)
        if self.core.slx_columns:
            x = jnp.concatenate(
                (
                    x,
                    spatial_panel_lag(
                        weights.values, x[..., jnp.asarray(self.core.slx_columns)]
                    ),
                ),
                axis=-1,
            )
        width = x.shape[-1]
        mean = jnp.einsum("tnp,p->tn", x, self.core.parameters[:width])
        if self.core.effect_type == "unit":
            mean = mean + self.core.effects[None, :]
        rho = (
            self.core.parameters[width]
            if self.core.lag
            else jnp.zeros((), dtype=mean.dtype)
        )
        if kind == "structural" and self.core.lag:
            outcome = data.numeric(self.context.response).reshape(periods, units)
            mean = mean + rho * spatial_panel_lag(weights.values, outcome)
        elif kind == "reduced-form" and self.core.lag:
            graph = jnp.broadcast_to(weights.values, (periods, units, units))
            mean = jnp.linalg.solve(
                jnp.eye(units, dtype=mean.dtype) - rho * graph, mean[..., None]
            )[..., 0]
        admissible = jnp.all(jnp.abs(self.core.parameters[width:]) < weights.limit)
        predicted = jnp.where(self.core.valid & admissible, mean, jnp.nan)
        period_ids = tuple(int(p) for p in jax.device_get(weights.period_ids).tolist())
        unit_ids = tuple(int(u) for u in jax.device_get(weights.unit_ids).tolist())
        return SpatialPanelPrediction(
            period_ids,
            tuple(unit_ids for _ in period_ids),
            tuple(predicted[t] for t in range(periods)),
        )

    def infer(self, *, covariance: SpatialInformation) -> InferenceResult:
        """Attach the panel fit's retained Gaussian information covariance."""
        if not isinstance(covariance, SpatialInformation):
            message = "Spatial panel ML inference requires SpatialInformation."
            raise TypeError(message)
        return InferenceResult(
            spatial_covariance(self.core, debiased=covariance.debiased),
            self.context,
            covariance,
            design=self.design,
            fit_statistics=self.fit_statistics(),
        )

    @override
    def likelihood_summary(self) -> LikelihoodSummary:
        """Return the Gaussian spatial panel likelihood summary.

        ``log_likelihood`` is the full Gaussian likelihood with the ``2 pi``
        term and the per-period Jacobians. ``k`` is the design coefficients, the
        active ``rho`` and ``lambda``, the scale and, for random effects, the
        variance ratio. Unit effects profiled out by the within projection
        count, one per entity, as the explicit-dummy routes do. ``n`` is ``T N``.
        """
        core = self.core
        return LikelihoodSummary(
            float(core.log_likelihood) if bool(core.valid) else math.nan,
            len(core.parameters)
            + 1
            + (core.effect_type == "random")
            + (int(core.effects.shape[0]) if core.effect_type == "unit" else 0),
            int(core.outcome.size),
            _SPATIAL_LIKELIHOOD_REASON,
        )

    @override
    def report(self) -> Report:
        """Report coefficients, likelihood, profiled scale and boundary evidence."""
        report = super().report()
        statistics = self.fit_statistics()
        return replace(
            report,
            metrics=(
                *report.metrics,
                _finite_metric(
                    "sigma2",
                    self.core.sigma2,
                    "Profiled Gaussian innovation variance, SSE/(T N).",
                ),
                *statistics.metrics,
            ),
            notes=(*report.notes, *statistics.notes),
            metadata={
                **dict(report.metadata),
                "identified": bool(self.core.identified),
                "boundary": bool(self.core.boundary),
                "effects": self.core.effect_type,
            },
        )


@dataclass(frozen=True)
class KKPPanelResult(ParameterResult[KKPPanelFit]):
    """KKP coefficients with separately retained spatial variance components."""

    design: FittedDesign = field(repr=False)
    panel: PanelLayout | None = None
    graph_name: str | None = None

    def prediction_options(self) -> PredictionOptions:
        """Report the lack of qualified spatial prediction for a KKP fit."""
        return _spatial_prediction_options(
            (),
            response=None,
            condition="",
            valid=self.valid,
            unavailable_reason="KKP estimates have no native prediction contract.",
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the coefficient names used by ``report``."""
        return self.core.linear.parameter_names or self.design.column_names

    @override
    def report(self) -> Report:
        valid = bool(self.core.valid)
        names = self.core.linear.parameter_names or self.design.column_names
        return Report(
            title=f"{self.context.model_name} fit",
            parameters=tuple(
                ParameterRow(
                    name,
                    float(self.core.parameters[index]) if valid else None,
                    valid=valid,
                    status="available" if valid else "unavailable",
                    unavailable_reason=None if valid else "invalid-fit",
                )
                for index, name in enumerate(names)
            ),
            metrics=(
                Metric("n_observations", int(self.context.observations.n_rows)),
                _finite_metric(
                    "lambda"
                    if self.graph_name is None
                    else f"lambda[{self.graph_name}]",
                    self.core.spatial_lambda,
                    "KKP spatial error coefficient.",
                ),
            ),
            metadata={
                "response": self.context.response,
                "fit_valid": valid,
                "estimator": "KKP",
                "beta_solver_converged": bool(self.core.linear.solver.converged),
                "beta_solver_status": int(self.core.linear.solver.status),
                "lambda_solver_converged": bool(self.core.lambda_solver.converged),
                "lambda_solver_status": int(self.core.lambda_solver.status),
            },
            warnings=()
            if valid
            else ("Point fitting is unavailable; inspect retained core diagnostics.",),
        )

    def infer(self, *, covariance: ModelBased) -> InferenceResult:
        """Attach plug-in covariance for beta only, conditional on fitted components."""
        if not isinstance(covariance, ModelBased):
            message = "KKP beta inference requires ModelBased covariance."
            raise TypeError(message)
        if not bool(self.core.valid):
            message = "KKP beta inference requires a valid component fit."
            raise ValueError(message)
        return InferenceResult(
            classical_covariance(
                self.core.linear,
                finite_sample_correction=covariance.finite_sample_correction,
            ),
            self.context,
            covariance,
            design=self.design,
        )


@dataclass(frozen=True)
class _SpatialPanelBackend(_RegressionPreparation[SpatialPanelResult | KKPPanelResult]):
    """Balanced spatial panel with a declared likelihood or KKP estimator.

    Rows must be time-major: period blocks in the graph's period order, units in
    its unit order within each block.
    """

    weights: SpatialPanelWeights
    panel: PanelLayout
    estimator: Estimator[SpatialPanelProblem, SpatialPanelFit | KKPPanelFit]
    process: SAR | SEM | SAC
    effects: Literal["pooled", "unit", "random"]
    graph_name: str | None = None
    lag_columns: tuple[str, ...] | None = None

    @property
    @override
    def input_columns(self) -> dict[str, InputRole]:
        """Declare predictor, response, entity and integer time columns."""
        return panel_input_roles(super().input_columns, self.panel)

    @override
    def _estimate_prepared(
        self, prepared: PreparedRegression
    ) -> SpatialPanelResult | KKPPanelResult:
        """Estimate matching prepared state with the native panel estimator."""
        columns = (
            select_lag_columns(prepared.design.column_names, self.lag_columns, ())
            if self.lag_columns is not None
            else ()
        )
        return _fit_panel(
            prepared,
            self.weights,
            self.panel,
            self.estimator,
            process=self.process,
            effects=self.effects,
            slx_columns=columns,
            graph_name=self.graph_name,
        )


def _fit_panel(
    prepared: PreparedRegression,
    weights: SpatialPanelWeights,
    panel: PanelLayout,
    estimator: Estimator[SpatialPanelProblem, SpatialPanelFit | KKPPanelFit],
    *,
    process: SAR | SEM | SAC,
    effects: Literal["pooled", "unit", "random"],
    slx_columns: tuple[int, ...],
    graph_name: str | None,
) -> SpatialPanelResult | KKPPanelResult:
    data = prepared.context.observations
    if any(name in _RESERVED for name in prepared.design.column_names):
        message = (
            "The coefficient names rho and lambda are reserved spatial parameters."
        )
        raise ValueError(message)
    periods = int(weights.period_ids.shape[0])
    units = int(weights.unit_ids.shape[0])
    entity = tuple(jax.device_get(weights.unit_ids).tolist()) * periods
    time = tuple(jax.device_get(jnp.repeat(weights.period_ids, units)).tolist())
    if (
        data.grouping_labels(panel.entity) != entity
        or data.grouping_labels(panel.time) != time
    ):
        message = "Panel rows must be time-major and match the graph's keys."
        raise ValueError(message)
    problem = SpatialPanelProblem(
        prepared.values.reshape(periods, units, -1),
        data.numeric(prepared.context.response).reshape(periods, units),
        weights,
        process=process,
        effects=effects,
        slx_columns=slx_columns,
        graph_name=graph_name,
        parameter_names=prepared.design.column_names,
    )
    core = estimator.estimate(problem)
    if isinstance(core, KKPPanelFit):
        return KKPPanelResult(
            core=core,
            context=prepared.context,
            design=prepared.design,
            panel=panel,
            graph_name=graph_name,
        )
    if not isinstance(core, SpatialPanelFit):
        message = "Spatial panel estimator returned an unsupported fit."
        raise TypeError(message)
    return SpatialPanelResult(
        core=core,
        context=prepared.context,
        design=prepared.design,
        panel=panel,
        graph_name=graph_name,
    )
