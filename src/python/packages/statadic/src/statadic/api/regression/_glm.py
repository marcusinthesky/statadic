# ruff: noqa: TC001  # Runtime annotations are resolved by the import hook.
"""Host lifecycle for Poisson and binomial-logit regression."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, override

import jax.numpy as jnp

from statadic.api.artifacts import save_predictor
from statadic.api.base import (
    Estimator,
    PreparedRegression,
    RegressionResult,
    _RegressionPreparation,
)
from statadic.api.fit_statistics import LikelihoodSummary
from statadic.api.inference import HC, InferenceResult, ModelBased, _independent_covariance
from statadic.api.inputs import RegressionInput
from statadic.api.prediction import (
    PredictionOptions,
    _fitted_regression_options,
    _prediction_design,
    _prediction_inputs,
    _prediction_role,
    _validate_prediction_kind,
)
from statadic.api.regression._schema import (
    check_binomial_observations,
    check_parameter_schema,
)
from statadic.api.reporting import Report, _finite_metric
from statadic.api.roles import InputRole, merge_input_roles
from statadic.core.typing import Array
from statadic.estimation.glm import (
    BinomialLogitMLE,
    BinomialProblem,
    GLMFit,
    PoissonMLE,
    PoissonProblem,
)
from statadic.inference.glm_covariance import GLMUncertainty, glm_covariance
from statadic.model.glm import BinomialLogitEquation, GLMEquation, PoissonLogEquation


@dataclass(frozen=True)
class _GLMResult[EquationT: GLMEquation](RegressionResult[GLMFit[EquationT]]):
    """Shared diagnostics and actual-fit covariance for canonical GLM results."""

    @property
    def df_resid(self) -> Array:
        """Return the native residual degrees of freedom."""
        return self.core.df_resid

    @override
    def likelihood_summary(self) -> LikelihoodSummary:
        """Return the complete log likelihood with its parameter and row counts.

        The log likelihood keeps the count and binomial-coefficient constants.
        Dispersion is fixed at one, so it is not estimated and ``k`` is the
        identified design rank.
        """
        core = self.core
        rank = int(core.rank)
        return LikelihoodSummary(
            float(core.log_likelihood),
            rank if rank >= 0 else int(core.parameters.shape[-1]),
            int(core.n_observations),
            "The GLM fit is invalid, so its log likelihood is unavailable.",
        )

    @override
    def report(self) -> Report:
        """Add full native likelihood, deviance and finite-optimum evidence."""
        report = super().report()
        statistics = self.fit_statistics()
        return replace(
            report,
            notes=(*report.notes, *statistics.notes),
            metrics=(
                *report.metrics,
                _finite_metric(
                    "deviance",
                    self.core.deviance,
                    "Twice saturated minus fitted log likelihood.",
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
                "finite_optimum": bool(self.core.existence.finite),
                "existence_status": int(self.core.existence.status),
            },
        )

    def infer(
        self, *, covariance: ModelBased | HC
    ) -> InferenceResult[GLMUncertainty[EquationT]]:
        """Attach supported covariance to this exact core fit without refitting."""
        options = _independent_covariance(covariance)
        return InferenceResult(
            glm_covariance(
                self.core,
                method=options.method,
                finite_sample_correction=options.finite_sample_correction,
            ),
            self.context,
            covariance,
            design=self.design,
            fit_statistics=self.fit_statistics(),
        )


@dataclass(frozen=True)
class PoissonResult(_GLMResult[PoissonLogEquation]):
    """Poisson result retaining explicit offset/exposure prediction roles."""

    exposure: str | None = None
    offset: str | None = None

    @override
    def report(self) -> Report:
        """Add retained offset and exposure identities to the shared diagnostics."""
        report = super().report()
        return replace(
            report,
            metadata={
                **dict(report.metadata),
                "offset": self.offset,
                "exposure": self.exposure,
            },
        )

    def prediction_options(self) -> PredictionOptions:
        """Describe prediction targets and required columns for this fitted design."""
        return _fitted_regression_options(
            "poisson",
            self.design,
            valid=self.valid,
            offset=self.offset,
            exposure=self.exposure,
        )

    def predict(
        self, data: RegressionInput, *, kind: Literal["mean", "link", "rate"] = "mean"
    ) -> Array:
        """Predict from retained training features and model-specific input roles."""
        _validate_prediction_kind("poisson", kind)
        data = _prediction_inputs(
            data, self.design, kind, offset=self.offset, exposure=self.exposure
        )
        # Even rate prediction validates any retained exposure role explicitly.
        exposure = _prediction_role(
            data, self.exposure, "exposure", kind, required=kind != "rate"
        )
        if exposure is not None and not bool(jnp.all(exposure > 0)):
            message = "Prediction exposure must be positive."
            raise ValueError(message)
        offset = _prediction_role(data, self.offset, "offset", kind)
        return self.core.predict(
            _prediction_design(data, self.design, kind),
            scale=kind,
            exposure=exposure,
            offset=offset,
            parameter_names=self.design.column_names,
        )

    def save_predictor(self, path: str | Path) -> None:
        """Write a restricted prediction artifact with retained feature state."""
        save_predictor(
            Path(path),
            design=self.design,
            parameters=self.core.parameters,
            family="poisson",
            exposure=self.exposure,
            offset=self.offset,
            metadata={
                "model": self.context.model_name,
                "response": self.context.response,
            },
        )


@dataclass(frozen=True)
class LogisticResult(_GLMResult[BinomialLogitEquation]):
    """Binomial-logit result retaining explicit trials/offset prediction roles."""

    trials: str | None = None
    offset: str | None = None

    @override
    def report(self) -> Report:
        """Add retained offset and trials identities to the shared diagnostics."""
        report = super().report()
        return replace(
            report,
            metadata={
                **dict(report.metadata),
                "offset": self.offset,
                "trials": self.trials,
            },
        )

    def prediction_options(self) -> PredictionOptions:
        """Describe prediction targets and required columns for this fitted design."""
        return _fitted_regression_options(
            "logistic",
            self.design,
            valid=self.valid,
            offset=self.offset,
            trials=self.trials,
        )

    def predict(
        self,
        data: RegressionInput,
        *,
        kind: Literal["probability", "mean", "link"] = "probability",
    ) -> Array:
        """Predict probabilities, expected successes or the linear predictor.

        Expected successes use the retained trials role on the new rows (one
        trial per row when the model declared none).
        """
        _validate_prediction_kind("logistic", kind)
        data = _prediction_inputs(
            data, self.design, kind, offset=self.offset, trials=self.trials
        )
        role_trials = (
            _prediction_role(data, self.trials, "trials", kind)
            if kind == "mean"
            else None
        )
        trials = 1 if role_trials is None else role_trials
        offset = _prediction_role(data, self.offset, "offset", kind)
        scale: Literal["link", "response", "mean"] = (
            "response" if kind == "probability" else kind
        )
        return self.core.predict(
            _prediction_design(data, self.design, kind),
            scale=scale,
            trials=trials,
            offset=offset,
            parameter_names=self.design.column_names,
        )

    def save_predictor(self, path: str | Path) -> None:
        """Write a restricted prediction artifact with retained feature state."""
        save_predictor(
            Path(path),
            design=self.design,
            parameters=self.core.parameters,
            family="logistic",
            trials=self.trials,
            offset=self.offset,
            metadata={
                "model": self.context.model_name,
                "response": self.context.response,
            },
        )


@dataclass(frozen=True)
class _PoissonHost(_RegressionPreparation[PoissonResult]):
    """Poisson/log MLE using retained named exposure and additive offset roles.

    An explicitly supplied estimator owns its solver configuration; ``max_steps``
    and ``tolerance`` configure the default estimator only.
    """

    _accepts_regression_inputs = True

    exposure: str | None = None
    offset: str | None = None
    max_steps: int = 256
    tolerance: float | None = None
    estimator: Estimator[PoissonProblem, GLMFit[PoissonLogEquation]] | None = field(
        default=None, kw_only=True
    )

    @property
    @override
    def input_columns(self) -> dict[str, InputRole]:
        """Declare predictor, response, offset and exposure roles explicitly."""
        return merge_input_roles(
            super().input_columns,
            {
                name: "numeric"
                for name in (self.offset, self.exposure)
                if name is not None
            },
        )

    @override
    def _estimate_prepared(self, prepared: PreparedRegression) -> PoissonResult:
        """Estimate matching prepared state with the Poisson MLE brick."""
        data = prepared.context.observations
        estimator = (
            self.estimator
            if self.estimator is not None
            else PoissonMLE(max_steps=self.max_steps, tol=self.tolerance)
        )
        core = estimator.estimate(
            PoissonProblem(
                prepared.values,
                data.numeric(self.response),
                exposure=None if self.exposure is None else data.numeric(self.exposure),
                offset=None if self.offset is None else data.numeric(self.offset),
                parameter_names=prepared.design.column_names,
            )
        )
        check_parameter_schema(core, prepared.design.column_names)
        return PoissonResult(
            core=core,
            context=prepared.context,
            design=prepared.design,
            exposure=self.exposure,
            offset=self.offset,
        )


@dataclass(frozen=True)
class _LogitHost(_RegressionPreparation[LogisticResult]):
    """Binomial-logit MLE with retained named trials and additive offset roles.

    ``response`` holds integer successes; ``trials=None`` means one trial per
    row (binary outcomes). Dispersion is fixed at one. An explicitly supplied
    estimator owns its solver configuration; ``max_steps`` and ``tolerance``
    configure the default estimator only.
    """

    _accepts_regression_inputs = True

    trials: str | None = None
    offset: str | None = None
    max_steps: int = 256
    tolerance: float | None = None
    estimator: Estimator[BinomialProblem, GLMFit[BinomialLogitEquation]] | None = field(
        default=None, kw_only=True
    )

    @override
    def prepare(self, data: RegressionInput) -> PreparedRegression:
        """Validate observed successes and trials on the retained training rows."""
        prepared = super().prepare(data)
        check_binomial_observations(prepared, self.trials)
        return prepared

    @property
    @override
    def input_columns(self) -> dict[str, InputRole]:
        """Declare predictor, response, offset and binomial-trials roles."""
        return merge_input_roles(
            super().input_columns,
            {
                name: "numeric"
                for name in (self.offset, self.trials)
                if name is not None
            },
        )

    @override
    def _estimate_prepared(self, prepared: PreparedRegression) -> LogisticResult:
        """Estimate matching prepared state with the binomial-logit MLE brick."""
        data = prepared.context.observations
        estimator = (
            self.estimator
            if self.estimator is not None
            else BinomialLogitMLE(max_steps=self.max_steps, tol=self.tolerance)
        )
        core = estimator.estimate(
            BinomialProblem(
                prepared.values,
                data.numeric(self.response),
                trials=1 if self.trials is None else data.numeric(self.trials),
                offset=None if self.offset is None else data.numeric(self.offset),
                parameter_names=prepared.design.column_names,
            )
        )
        check_parameter_schema(core, prepared.design.column_names)
        return LogisticResult(
            core=core,
            context=prepared.context,
            design=prepared.design,
            trials=self.trials,
            offset=self.offset,
        )
