# ruff: noqa: TC001  # Runtime annotations are resolved by the import hook.
"""Host lifecycle for complete-array linear regression."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, override

from statadic.api.artifacts import save_predictor
from statadic.api.base import (
    Estimator,
    PreparedRegression,
    RegressionResult,
    _RegressionPreparation,
)
from statadic.api.data import RelativePrecision, _resolve_row_order, _row_order_metadata
from statadic.api.design import level_codes
from statadic.api.diagnostics import Diagnostics, LinearDiagnostic, _diagnostic_requests
from statadic.api.fit_statistics import (
    FitStatistics,
    LikelihoodSummary,
    gaussian_least_squares_summary,
    linear_fit_statistics,
)
from statadic.api.inference import HAC, HC, Clustered, InferenceResult, ModelBased
from statadic.api.inputs import RegressionInput
from statadic.api.prediction import (
    PredictionOptions,
    _fitted_regression_options,
    _prediction_design,
    _prediction_inputs,
    _validate_prediction_kind,
)
from statadic.api.regression._schema import check_parameter_schema
from statadic.api.reporting import MetadataValue, Report, _finite_metric
from statadic.api.roles import InputRole, merge_input_roles
from statadic.core.typing import Array
from statadic.estimation.linear import LeastSquares, LinearFit, LinearProblem
from statadic.inference.contracts import FittedUncertainty
from statadic.inference.linear_covariance import (
    classical_covariance,
    cluster_covariance,
    hac_covariance,
    hc_covariance,
)


@dataclass(frozen=True)
class LinearResult(RegressionResult[LinearFit]):
    """OLS/WLS result preserving the native fit and its retained feature state."""

    weighted: bool = field(default=False, kw_only=True, repr=False)

    @property
    def df_resid(self) -> Array:
        """Return the native residual degrees of freedom."""
        return self.core.df_resid

    @override
    def likelihood_summary(self) -> LikelihoodSummary:
        """Return the Gaussian likelihood at the profiled scale.

        ``k`` counts the design rank plus the scale and ``n`` is the number of
        observations; precision weights enter the log likelihood.
        """
        return gaussian_least_squares_summary(self.core)

    @override
    def fit_statistics(self) -> FitStatistics:
        """Return the Gaussian information criteria and R-squared of this fit.

        See ``statadic.api.fit_statistics.linear_fit_statistics``; precision weights
        enter both the log likelihood and R-squared when this result is weighted.
        """
        return linear_fit_statistics(self.core, weighted=self.weighted)

    @override
    def report(self) -> Report:
        """Add native objectives and residual degrees of freedom to the fit report."""
        report = super().report()
        statistics = self.fit_statistics()
        return replace(
            report,
            notes=(*report.notes, *statistics.notes),
            metrics=(
                *report.metrics,
                _finite_metric(
                    "sse", self.core.sse, "Sum of squared original-row residuals."
                ),
                _finite_metric(
                    "weighted_sse",
                    self.core.weighted_sse,
                    "Analytic-precision weighted residual sum of squares.",
                ),
                _finite_metric(
                    "common_scale",
                    self.core.common_scale,
                    "Weighted SSE / residual degrees of freedom.",
                ),
                _finite_metric(
                    "df_resid",
                    self.core.df_resid,
                    "Retained rows minus identified design rank.",
                ),
                *statistics.metrics,
            ),
            metadata={
                **dict(report.metadata),
                "identified": bool(self.core.identified),
            },
        )

    def prediction_options(self) -> PredictionOptions:
        """Describe prediction targets and required columns for this fitted design."""
        return _fitted_regression_options("linear", self.design, valid=self.valid)

    def predict(
        self, data: RegressionInput, *, kind: Literal["mean"] = "mean"
    ) -> Array:
        """Predict from retained training features and model-specific input roles.

        Args:
            data: New rows containing the fitted design's predictor columns.
            kind: Prediction target, which is ``"mean"`` for least squares.

        Returns:
            Fitted conditional means in the order of the supplied rows.

        """
        _validate_prediction_kind("linear", kind)
        data = _prediction_inputs(data, self.design, kind)
        return self.core.predict(
            _prediction_design(data, self.design, kind),
            parameter_names=self.design.column_names,
        )

    def diagnose(self, *requests: LinearDiagnostic) -> Diagnostics:
        """Evaluate explicit diagnostic requests against this exact fit.

        Each request names its procedure, options and null hypothesis, for
        example ``JarqueBera()``, ``BreuschPagan()`` or ``Influence()``.
        Nothing is refitted; unavailable diagnostics stay visible with a reason.

        Args:
            *requests: Diagnostic specifications to evaluate.

        Returns:
            A collection of outcomes with their availability and references.

        """
        return Diagnostics(
            self.context.model_name,
            tuple(
                request.evaluate_linear(self)
                for request in _diagnostic_requests(requests, LinearDiagnostic)
            ),
        )

    def infer(
        self, *, covariance: ModelBased | HC | HAC | Clustered
    ) -> InferenceResult:
        """Attach supported covariance to this exact core fit without refitting.

        Args:
            covariance: Model-based, HC, HAC or clustered covariance request.

        Returns:
            An inference result for intervals and hypothesis tests.

        Raises:
            TypeError: If the covariance request is unsupported.

        """
        if not isinstance(covariance, (ModelBased, HC, HAC, Clustered)):
            message = "Unsupported linear covariance specification."
            raise TypeError(message)
        correction = covariance.finite_sample_correction
        calculation: dict[str, MetadataValue] = {}
        if isinstance(covariance, ModelBased):
            uncertainty: FittedUncertainty = classical_covariance(
                self.core, finite_sample_correction=correction
            )
        elif isinstance(covariance, HC):
            uncertainty = hc_covariance(
                self.core, kind=covariance.kind, finite_sample_correction=correction
            )
        elif isinstance(covariance, HAC):
            order = _resolve_row_order(self.context.observations, covariance.order_by)
            calculation = _row_order_metadata(
                self.context.observations, covariance.order_by, order
            )
            uncertainty = hac_covariance(
                self.core,
                covariance.bandwidth,
                row_order=order,
                finite_sample_correction=correction,
            )
        elif isinstance(covariance, Clustered):
            labels = self.context.observations.resolve_grouping(covariance.column)
            levels, codes = level_codes(labels)
            uncertainty = cluster_covariance(
                self.core,
                codes,
                n_clusters=len(levels),
                finite_sample_correction=correction,
            )
        else:
            message = "Unsupported linear covariance specification."
            raise TypeError(message)
        return InferenceResult(
            uncertainty,
            self.context,
            covariance,
            design=self.design,
            calculation=calculation,
            fit_statistics=self.fit_statistics(),
        )

    def save_predictor(self, path: str | Path) -> None:
        """Write a restricted prediction artifact with retained feature state."""
        save_predictor(
            Path(path),
            design=self.design,
            parameters=self.core.parameters,
            family="linear",
            metadata={
                "model": self.context.model_name,
                "response": self.context.response,
            },
        )


@dataclass(frozen=True)
class _LinearBackend(_RegressionPreparation[LinearResult]):
    """Reusable OLS specification with explicit predictor terms."""

    _accepts_regression_inputs = True

    estimator: Estimator[LinearProblem, LinearFit] = field(
        default_factory=LeastSquares, kw_only=True
    )

    @override
    def _estimate_prepared(self, prepared: PreparedRegression) -> LinearResult:
        """Estimate matching prepared state with the least-squares estimator brick."""
        core = self.estimator.estimate(
            LinearProblem(
                prepared.values,
                prepared.context.observations.numeric(self.response),
                parameter_names=prepared.design.column_names,
            )
        )
        check_parameter_schema(core, prepared.design.column_names)
        return LinearResult(core=core, context=prepared.context, design=prepared.design)


@dataclass(frozen=True)
class _WeightedLinearBackend(_LinearBackend):
    """Analytic relative-precision WLS; no frequency or survey weighting."""

    precision: RelativePrecision

    @property
    @override
    def input_columns(self) -> dict[str, InputRole]:
        """Add the analytic precision column to the ordinary regression roles."""
        return merge_input_roles(
            super().input_columns, {self.precision.column: "numeric"}
        )

    @override
    def _estimate_prepared(self, prepared: PreparedRegression) -> LinearResult:
        """Estimate matching prepared state with the least-squares estimator brick."""
        data = prepared.context.observations
        core = self.estimator.estimate(
            LinearProblem(
                prepared.values,
                data.numeric(self.response),
                data.numeric(self.precision.column),
                parameter_names=prepared.design.column_names,
            )
        )
        check_parameter_schema(core, prepared.design.column_names)
        return LinearResult(
            core=core,
            context=prepared.context,
            design=prepared.design,
            weighted=True,
        )
