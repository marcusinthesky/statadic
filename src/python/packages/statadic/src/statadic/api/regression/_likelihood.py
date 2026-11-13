# ruff: noqa: TC001  # Runtime annotations are inspected by the import hook.
"""Host workflows for Gamma, NB2 and grouped probit likelihoods."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import override

import jax.numpy as jnp

from statadic.api.base import (
    Estimator,
    PreparedRegression,
    RegressionResult,
    _RegressionPreparation,
)
from statadic.api.fit_statistics import LikelihoodSummary
from statadic.api.inference import (
    HC,
    InferenceResult,
    ModelBased,
    _independent_covariance,
)
from statadic.api.inputs import RegressionInput
from statadic.api.prediction import (
    PredictionOptions,
    _default_prediction_kind,
    _fitted_regression_options,
    _prediction_design,
    _prediction_inputs,
    _prediction_role,
    _validate_prediction_kind,
)
from statadic.api.regression._schema import check_binomial_observations
from statadic.api.reporting import Report, _finite_metric
from statadic.api.roles import InputRole, merge_input_roles
from statadic.core.typing import Array
from statadic.estimation.likelihood import (
    Dispersion,
    FixedDispersion,
    LikelihoodFit,
    LikelihoodProblem,
    ProbitMLE,
)
from statadic.inference.likelihood import likelihood_covariance


@dataclass(frozen=True)
class LikelihoodResult(RegressionResult[LikelihoodFit]):
    """A retained likelihood fit with explicit new-row roles."""

    offset: str | None = None
    exposure: str | None = None
    trials: str | None = None

    def prediction_options(self) -> PredictionOptions:
        """Describe prediction targets and required columns for this fitted design."""
        return _fitted_regression_options(
            self.core.family,
            self.design,
            valid=self.valid,
            offset=self.offset,
            exposure=self.exposure,
            trials=self.trials,
        )

    def predict(self, data: RegressionInput, *, kind: str | None = None) -> Array:
        family = self.core.family
        kind = _default_prediction_kind(family) if kind is None else kind
        _validate_prediction_kind(family, kind)
        data = _prediction_inputs(
            data,
            self.design,
            kind,
            offset=self.offset,
            exposure=self.exposure,
            trials=self.trials,
        )
        exposure = _prediction_role(
            data, self.exposure, "exposure", kind, required=kind != "rate"
        )
        if exposure is not None and not bool(jnp.all(exposure > 0)):
            message = "Prediction exposure must be positive."
            raise ValueError(message)
        offset = _prediction_role(data, self.offset, "offset", kind)
        trials = (
            _prediction_role(data, self.trials, "trials", kind)
            if kind == "mean"
            else None
        )
        return self.core.predict(
            _prediction_design(data, self.design, kind),
            kind=kind,
            offset=0 if offset is None else offset,
            exposure=1 if exposure is None else exposure,
            trials=1 if trials is None else trials,
        )

    def infer(self, *, covariance: ModelBased | HC) -> InferenceResult:
        options = _independent_covariance(covariance)
        return InferenceResult(
            likelihood_covariance(
                self.core,
                method=options.method,
                finite_sample_correction=options.finite_sample_correction,
            ),
            self.context,
            covariance,
            design=self.design,
            fit_statistics=self.fit_statistics(),
        )

    @override
    def likelihood_summary(self) -> LikelihoodSummary:
        """Return the complete log likelihood with its parameter and row counts.

        ``k`` counts every estimated parameter: the coefficients plus the
        dispersion when it is estimated (gamma ``phi``, NB2 ``alpha``). A
        caller-fixed dispersion and the probit link have none.
        """
        core = self.core
        return LikelihoodSummary(
            float(core.log_likelihood),
            int(core.parameters.shape[-1]),
            int(core.n_observations),
            "The likelihood fit is invalid, so its log likelihood is unavailable.",
        )

    @override
    def report(self) -> Report:
        report = super().report()
        statistics = self.fit_statistics()
        dispersion_name = "phi" if self.core.family == "gamma" else "alpha"
        dispersion_metrics = (
            ()
            if self.core.family == "probit"
            else (
                _finite_metric(
                    dispersion_name, self.core.dispersion, "Fitted or fixed dispersion."
                ),
            )
        )
        return replace(
            report,
            notes=(*report.notes, *statistics.notes),
            metrics=(
                *report.metrics,
                *dispersion_metrics,
                _finite_metric(
                    "df_resid", self.core.df_resid, "Rows minus estimated parameters."
                ),
                *statistics.metrics,
            ),
            metadata={
                **dict(report.metadata),
                "offset": self.offset,
                "exposure": self.exposure,
                "trials": self.trials,
                "input_valid": bool(self.core.input_valid),
                "stationary": bool(self.core.stationary),
                "dispersion_boundary": bool(self.core.dispersion_boundary),
                "numerical_resolved": bool(self.core.numerical_resolved),
            },
        )


def _problem(
    prepared: PreparedRegression,
    response: str,
    *,
    offset: str | None,
    exposure: str | None = None,
    trials: str | None = None,
) -> LikelihoodProblem:
    data = prepared.context.observations
    return LikelihoodProblem(
        prepared.values,
        data.numeric(response),
        offset=0 if offset is None else data.numeric(offset),
        exposure=1 if exposure is None else data.numeric(exposure),
        trials=1 if trials is None else data.numeric(trials),
        parameter_names=prepared.design.column_names,
    )


def _check_fit(
    fit: LikelihoodFit,
    family: str,
    names: tuple[str, ...],
    dispersion: Dispersion | None = None,
) -> None:
    """Require injected estimators to retain this model's family and schema."""
    if fit.family != family:
        message = f"{family} estimator returned a different family."
        raise ValueError(message)
    if (
        fit.n_coefficients != len(names)
        or fit.parameter_names not in (names, (*names, "dispersion"))
        or fit.parameters.shape != (len(fit.parameter_names or ()),)
    ):
        message = "Estimator returned a different parameter schema."
        raise ValueError(message)
    if dispersion is not None:
        expected_names = (
            names if isinstance(dispersion, FixedDispersion) else (*names, "dispersion")
        )
        if fit.parameter_names != expected_names:
            message = "Estimator returned a different dispersion policy."
            raise ValueError(message)
        if isinstance(dispersion, FixedDispersion) and bool(
            jnp.isfinite(fit.dispersion)
        ):
            expected = jnp.asarray(dispersion.value, dtype=fit.dispersion.dtype)
            if not bool(fit.dispersion == expected):
                message = "Estimator returned a different fixed dispersion."
                raise ValueError(message)


@dataclass(frozen=True)
class _GammaHost(_RegressionPreparation[LikelihoodResult]):
    """Gamma log-mean regression with an explicit dispersion policy."""

    _accepts_regression_inputs = True

    estimator: Estimator[LikelihoodProblem, LikelihoodFit]
    dispersion: Dispersion
    offset: str | None = None

    @property
    @override
    def input_columns(self) -> dict[str, InputRole]:
        """Declare the offset beside the response and predictor roles."""
        return merge_input_roles(
            super().input_columns,
            {} if self.offset is None else {self.offset: "numeric"},
        )

    def _estimate_prepared(self, prepared: PreparedRegression) -> LikelihoodResult:
        fit = self.estimator.estimate(
            _problem(prepared, self.response, offset=self.offset)
        )
        _check_fit(fit, "gamma", prepared.design.column_names, self.dispersion)
        return LikelihoodResult(
            fit, prepared.context, prepared.design, offset=self.offset
        )


@dataclass(frozen=True)
class _NB2Host(_RegressionPreparation[LikelihoodResult]):
    """NB2 regression with explicit dispersion and optional exposure."""

    _accepts_regression_inputs = True

    estimator: Estimator[LikelihoodProblem, LikelihoodFit]
    dispersion: Dispersion
    offset: str | None = None
    exposure: str | None = None

    @property
    @override
    def input_columns(self) -> dict[str, InputRole]:
        """Declare predictor, response, offset and exposure input roles."""
        return merge_input_roles(
            super().input_columns,
            {
                name: "numeric"
                for name in (self.offset, self.exposure)
                if name is not None
            },
        )

    def _estimate_prepared(self, prepared: PreparedRegression) -> LikelihoodResult:
        fit = self.estimator.estimate(
            _problem(
                prepared, self.response, offset=self.offset, exposure=self.exposure
            )
        )
        _check_fit(fit, "nb2", prepared.design.column_names, self.dispersion)
        return LikelihoodResult(
            fit,
            prepared.context,
            prepared.design,
            offset=self.offset,
            exposure=self.exposure,
        )


@dataclass(frozen=True)
class _ProbitHost(_RegressionPreparation[LikelihoodResult]):
    """Binary or grouped-binomial probit regression."""

    _accepts_regression_inputs = True

    offset: str | None = None
    trials: str | None = None
    estimator: Estimator[LikelihoodProblem, LikelihoodFit] = field(
        default_factory=ProbitMLE
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
        """Declare predictor, response, offset and grouped-trials input roles."""
        return merge_input_roles(
            super().input_columns,
            {
                name: "numeric"
                for name in (self.offset, self.trials)
                if name is not None
            },
        )

    def _estimate_prepared(self, prepared: PreparedRegression) -> LikelihoodResult:
        fit = self.estimator.estimate(
            _problem(prepared, self.response, offset=self.offset, trials=self.trials)
        )
        _check_fit(fit, "probit", prepared.design.column_names)
        return LikelihoodResult(
            fit,
            prepared.context,
            prepared.design,
            offset=self.offset,
            trials=self.trials,
        )
