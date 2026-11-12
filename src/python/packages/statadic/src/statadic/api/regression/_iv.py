"""Aligned host workflow for non-spatial instrumental-variable regression."""

from dataclasses import dataclass, field, replace
from typing import override

import jax.numpy as jnp

from statadic.api.base import (  # noqa: TC001
    Estimator,
    FitContext,
    Model,
    RegressionResult,
)
from statadic.api.data import _resolve_row_order, _row_order_metadata
from statadic.api.design import FittedDesign, Term, level_codes  # noqa: TC001
from statadic.api.diagnostics import Diagnostics, IVDiagnostic, _diagnostic_requests
from statadic.api.inference import HAC, HC, Clustered, InferenceResult, ModelBased
from statadic.api.inputs import RegressionInput, as_observations
from statadic.api.prediction import (
    PredictionOption,
    PredictionOptions,
    _prediction_inputs,
)
from statadic.api.reporting import MetadataValue, Report, _finite_metric  # noqa: TC001
from statadic.api.roles import InputRole, merge_input_roles  # noqa: TC001
from statadic.core.typing import Array  # noqa: TC001
from statadic.estimation.iv import (
    IVFit,
    IVProblem,
    TwoStageLeastSquares,
    iv_first_stage_diagnostics,
)
from statadic.inference.iv import IVUncertainty, iv_covariance  # noqa: TC001


@dataclass(frozen=True)
class PreparedIV:
    """Frozen model-owned aligned IV inputs and learned design."""

    design: FittedDesign
    context: FitContext
    problem: IVProblem
    model_token: object = field(repr=False)


@dataclass(frozen=True)
class IVResult(RegressionResult[IVFit]):
    """Structural IV result with explicit uncertainty and prediction roles."""

    endogenous: tuple[str, ...]
    exogenous_width: int

    def prediction_options(self) -> PredictionOptions:
        """Describe the structural index available from predict(data)."""
        columns = tuple(dict.fromkeys((*self.design.input_columns, *self.endogenous)))
        return PredictionOptions.for_fit(
            (
                PredictionOption(
                    "structural", "Linear index at supplied endogenous values.", columns
                ),
            ),
            valid=self.valid,
        )

    def predict(self, data: RegressionInput) -> Array:
        """Evaluate the structural linear index using supplied endogenous values."""
        data = _prediction_inputs(data, self.design, "mean", additional=self.endogenous)
        return self.design.transform(data) @ self.core.parameters

    def infer(
        self, *, covariance: ModelBased | HC | HAC | Clustered
    ) -> InferenceResult[IVUncertainty]:
        """Attach the requested covariance to the retained actual-weight fit."""
        if not isinstance(covariance, (ModelBased, HC, HAC, Clustered)):
            message = "Unsupported IV covariance"
            raise TypeError(message)
        correction = covariance.finite_sample_correction
        calculation: dict[str, MetadataValue] = {}
        if isinstance(covariance, ModelBased):
            uncertainty = iv_covariance(
                self.core, kind="model", finite_sample_correction=correction
            )
        elif isinstance(covariance, HC):
            if covariance.kind not in {"HC0", "HC1"}:
                message = "IV supports HC0 and HC1"
                raise ValueError(message)
            uncertainty = iv_covariance(
                self.core, kind=covariance.kind, finite_sample_correction=correction
            )
        elif isinstance(covariance, HAC):
            order = _resolve_row_order(self.context.observations, covariance.order_by)
            calculation = _row_order_metadata(
                self.context.observations, covariance.order_by, order
            )
            uncertainty = iv_covariance(
                self.core,
                kind="HAC",
                bandwidth=covariance.bandwidth,
                row_order=order,
                finite_sample_correction=correction,
            )
        else:
            labels = self.context.observations.resolve_grouping(covariance.column)
            levels, codes = level_codes(labels)
            uncertainty = iv_covariance(
                self.core,
                kind="cluster",
                groups=codes,
                n_clusters=len(levels),
                finite_sample_correction=correction,
            )
        return InferenceResult(
            uncertainty,
            self.context,
            covariance,
            design=self.design,
            calculation=calculation,
        )

    def diagnose(self, *requests: IVDiagnostic) -> Diagnostics:
        """Evaluate explicit IV diagnostics without replacing the fitted weight."""
        return Diagnostics(
            self.context.model_name,
            tuple(
                request.evaluate_iv(self)
                for request in _diagnostic_requests(requests, IVDiagnostic)
            ),
        )

    @override
    def report(self) -> Report:
        """Report fit status and descriptive first-stage partial R²."""
        report = super().report()
        r2, rank = iv_first_stage_diagnostics(self.core, self.exogenous_width)
        metrics = tuple(
            _finite_metric(
                f"first_stage_partial_r2.{name}",
                value,
                "Partial R² conditional on exogenous design; "
                "not a weak-identification test.",
            )
            for name, value in zip(self.endogenous, r2, strict=True)
        )
        return replace(
            report,
            metrics=(*report.metrics, *metrics),
            metadata={
                **dict(report.metadata),
                "instrument_rank": int(rank),
                "weight_iteration_converged": bool(self.core.converged),
                "weight_iterations": int(self.core.iterations),
                "clusters": self.core.n_clusters,
            },
        )


@dataclass(frozen=True)
class _IVBackend(Model[RegressionInput, PreparedIV, IVResult]):
    """Prepare and fit aligned structural and excluded-instrument columns."""

    response: str
    terms: "Term"
    endogenous: tuple[str, ...]
    instruments: tuple[str, ...]
    estimator: "Estimator[IVProblem, IVFit]" = field(
        default_factory=TwoStageLeastSquares, kw_only=True
    )
    clusters: str | None = None

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Declare structural, instrument and fit-time clustering input roles."""
        roles = merge_input_roles(
            self.terms.input_columns,
            dict.fromkeys(
                (self.response, *self.endogenous, *self.instruments), "numeric"
            ),
        )
        if self.clusters is not None and roles.get(self.clusters) != "numeric":
            roles = merge_input_roles(roles, {self.clusters: "categorical"})
        return roles

    def prepare(self, data: RegressionInput) -> PreparedIV:
        """Retain observations and construct structural and instrument matrices."""
        if (
            not self.endogenous
            or not self.instruments
            or len(set(self.endogenous)) != len(self.endogenous)
            or len(set(self.instruments)) != len(self.instruments)
        ):
            msg = "Supply unique endogenous and excluded instrument columns"
            raise ValueError(msg)
        data = as_observations(data, self.input_columns, broadcast_scalars=False)
        design = FittedDesign(self.terms._prepare_term(data))
        x = design.transform(data)
        width = len(self.endogenous)
        if design.column_names[-width:] != self.endogenous:
            message = (
                "Endogenous scalar terms must be the final structural columns, "
                "in declared order."
            )
            raise ValueError(message)
        exog = x[:, :-width]
        z = jnp.column_stack((exog, *(data.numeric(name) for name in self.instruments)))
        groups, count = None, 0
        if self.clusters is not None:
            levels, groups = level_codes(data.grouping_labels(self.clusters))
            count = len(levels)
        context = FitContext(data, type(self).__name__, self.response)
        problem = IVProblem(
            x,
            data.numeric(self.response),
            z,
            groups,
            count,
            design.column_names,
        )
        return PreparedIV(design, context, problem, self._token)

    def _estimate_prepared(self, prepared: PreparedIV) -> IVResult:
        """Estimate only preparation owned by this model instance."""
        fit = self.estimator.estimate(prepared.problem)
        _check_iv_fit_schema(fit, prepared.problem)
        return IVResult(
            fit,
            prepared.context,
            prepared.design,
            self.endogenous,
            len(prepared.design.column_names) - len(self.endogenous),
        )


def _check_iv_fit_schema(fit: object, problem: IVProblem) -> None:
    """Require the custom estimator result to retain this prepared schema."""
    if not isinstance(fit, IVFit):
        message = "IV estimator must return IVFit"
        raise TypeError(message)
    n, p = jnp.asarray(problem.design).shape
    m = jnp.asarray(problem.instruments).shape[1]
    if (
        fit.parameter_names != problem.parameter_names  # noqa: PLR0916
        or fit.parameters.shape != (p,)
        or fit.design.shape != (n, p)
        or fit.outcome.shape != (n,)
        or fit.instruments.shape != (n, m)
        or fit.affine.scores.values.shape != (n, m)
        or fit.affine.jacobian.values.shape != (m, p)
        or fit.affine.moment_map.shape != (p, m)
        or fit.affine.weight.shape != (m, m)
        or fit.preliminary_variability.shape != (m, m)
        or fit.final_variability.shape != (m, m)
        or fit.n_clusters != problem.n_clusters
        or (fit.groups is None) != (problem.groups is None)
        or (fit.groups is not None and fit.groups.shape != (n,))
    ):
        message = "IV estimator result does not match prepared schema"
        raise ValueError(message)
