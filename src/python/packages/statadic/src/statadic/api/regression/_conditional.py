"""Named predictors for conditional response laws, compiled to native estimators."""

from __future__ import annotations

from collections.abc import Mapping, Sized
from dataclasses import dataclass, field
from typing import Literal, cast, override

import jax.numpy as jnp
from jax.nn import log_sigmoid, sigmoid
from jax.scipy.special import log_ndtr, ndtr

from statadic.api.base import (
    Estimator,
    Model,
)
from statadic.api.data import MeasureColumn, Observations
from statadic.api.design import FittedDesign, Term
from statadic.api.fit_statistics import FitStatistics, LikelihoodSummary
from statadic.api.inputs import RegressionInput, as_observations
from statadic.api.prediction import PredictionOption, PredictionOptions
from statadic.api.regression._glm import (
    LogisticResult,
    PoissonResult,
    _LogitHost,
    _PoissonHost,
)
from statadic.api.regression._likelihood import (
    LikelihoodResult,
    _GammaHost,
    _NB2Host,
    _ProbitHost,
)
from statadic.api.reporting import ParameterRow, Report
from statadic.api.roles import InputRole, merge_input_roles  # noqa: TC001
from statadic.core.typing import Array  # noqa: TC001
from statadic.estimation.glm import (
    BinomialLogitMLE,
    BinomialProblem,
    GLMFit,
    PoissonProblem,
)
from statadic.estimation.likelihood import (
    Dispersion,  # noqa: TC001 -- checked public law annotations
    GammaMLE,
    LikelihoodFit,
    LikelihoodProblem,
    NegativeBinomialMLE,
    PositivePoissonMLE,
    ProbitMLE,
)
from statadic.model.glm import BinomialLogitEquation, PoissonLogEquation
from statadic.model.likelihood import positive_poisson_log_mean


@dataclass(frozen=True)
class Observed:
    """Observed outcome and optional binomial trial totals."""

    response: str
    trials: str | None = None


@dataclass(frozen=True)
class Gaussian:
    """Gaussian response law; predictor terms belong to Regression."""


@dataclass(frozen=True)
class Binomial[LinkT: Literal["logit", "probit"] = Literal["logit"]]:
    """Binomial response with an explicit probability link."""

    link: LinkT = cast("LinkT", "logit")  # noqa: RUF009 -- immutable literal generic default

    def __post_init__(self) -> None:
        if self.link not in ("logit", "probit"):
            message = "Binomial supports logit or probit links."
            raise ValueError(message)


@dataclass(frozen=True)
class Poisson:
    """Poisson response with its log-rate parametrization."""


@dataclass(frozen=True)
class Gamma:
    """Gamma log-mean response with an explicit dispersion policy."""

    dispersion: Dispersion


@dataclass(frozen=True)
class NegativeBinomial2:
    """NB2 log-mean response with an explicit dispersion policy."""

    dispersion: Dispersion


@dataclass(frozen=True)
class HurdlePoisson:
    """Zero-probability gate and zero-truncated log-rate count law."""

    zero_link: Literal["logit", "probit"] = "logit"

    def __post_init__(self) -> None:
        if self.zero_link not in ("logit", "probit"):
            message = "HurdlePoisson supports logit or probit zero links."
            raise ValueError(message)


@dataclass(frozen=True)
class _Predictor:
    """Resolved private predictor; declared terms remain on Regression."""

    terms: Term
    link: str = "log"
    offset: str | None = None
    exposure: str | None = None


@dataclass(frozen=True)
class _PreparedHurdle:
    """Gate rows and positive-count rows with their own trained designs."""

    data: Observations = field(repr=False)
    gate: FittedDesign
    positive: Observations = field(repr=False)
    count: FittedDesign
    model_token: object = field(repr=False)


type GateFit = GLMFit[BinomialLogitEquation] | LikelihoodFit
type HurdleTarget = Literal["zero_probability", "rate", "positive_mean", "mean"]


@dataclass(frozen=True, repr=False)
class HurdleResult:
    """Retained two-block fit. Each prediction uses only the blocks it needs."""

    gate_fit: GateFit
    count_fit: LikelihoodFit
    gate: FittedDesign
    count: FittedDesign
    offset: str | None
    count_offset: str | None
    exposure: str | None

    @property
    def valid(self) -> bool:
        """Whether both component parameters have resolved finite estimates."""
        return bool(self.gate_fit.valid & self.count_fit.valid)

    def prediction_options(self) -> PredictionOptions:
        """Identify the fitted blocks and covariates needed for each target."""
        gate = tuple(self.gate.input_columns) + (
            () if self.offset is None else (self.offset,)
        )
        rate = tuple(self.count.input_columns) + (
            () if self.count_offset is None else (self.count_offset,)
        )
        count = rate + (() if self.exposure is None else (self.exposure,))
        rows = (
            (
                "zero_probability",
                "Probability of a zero response.",
                gate,
                self.gate_fit.valid,
            ),
            (
                "rate",
                "Untruncated Poisson intensity per unit exposure.",
                rate,
                self.count_fit.valid,
            ),
            (
                "positive_mean",
                "Expected count given a positive response.",
                count,
                self.count_fit.valid,
            ),
            ("mean", "Unconditional expected count.", (*gate, *count), self.valid),
        )
        return PredictionOptions(
            tuple(
                PredictionOption(
                    kind,
                    meaning,
                    tuple(dict.fromkeys(columns)),
                    unavailable_reason=None
                    if bool(valid)
                    else "Required component fit is invalid.",
                )
                for kind, meaning, columns, valid in rows
            )
        )

    def predict(self, data: RegressionInput, *, kind: HurdleTarget = "mean") -> Array:
        """Predict a gate probability, unit rate, positive mean or marginal mean."""
        if kind not in ("zero_probability", "rate", "positive_mean", "mean"):
            message = "Choose zero_probability, rate, positive_mean or mean."
            raise ValueError(message)
        rows = self._prediction_rows(data, kind)
        log_nonzero = 0.0
        if kind in ("zero_probability", "mean"):
            x = self.gate.transform(rows)
            offset = 0 if self.offset is None else rows.numeric(self.offset)
            # Evaluate the complementary tail directly so near-certain zeros
            # retain their small, scientifically meaningful positive mean.
            eta = x @ self.gate_fit.parameters + offset
            inverse_link = sigmoid if isinstance(self.gate_fit, GLMFit) else ndtr
            if kind == "zero_probability":
                return inverse_link(eta)
            log_nonzero = (
                log_sigmoid(-eta)
                if isinstance(self.gate_fit, GLMFit)
                else log_ndtr(-eta)
            )
        exposure = (
            1
            if self.exposure is None or kind == "rate"
            else rows.numeric(self.exposure)
        )
        if not bool(jnp.all(jnp.asarray(exposure) > 0)):
            message = "Prediction exposure must be positive."
            raise ValueError(message)
        values = self.count.transform(rows)
        count_offset = (
            0 if self.count_offset is None else rows.numeric(self.count_offset)
        )
        if kind == "rate":
            return self.count_fit.predict(values, kind="rate", offset=count_offset)
        eta_count = self.count_fit.predict(
            values, kind="link", exposure=exposure, offset=count_offset
        )
        return jnp.exp(log_nonzero + positive_poisson_log_mean(eta_count))

    def _prediction_rows(
        self, data: RegressionInput, kind: HurdleTarget
    ) -> Observations:
        gate = kind in ("zero_probability", "mean")
        count = kind != "zero_probability"
        names = (
            self.offset if gate else None,
            self.count_offset if count else None,
            self.exposure if kind in ("positive_mean", "mean") else None,
        )
        roles = merge_input_roles(
            self.gate.input_columns if gate else {},
            self.count.input_columns if count else {},
            {name: "numeric" for name in names if name is not None},
        )
        if isinstance(data, Mapping):
            declared = {
                *self.gate.input_columns,
                *self.count.input_columns,
                *(
                    name
                    for name in (self.offset, self.count_offset, self.exposure)
                    if name is not None
                ),
            }
            supplied = {
                name: value
                for name, value in data.items()
                if name in roles or name not in declared
            }
            if not roles and not supplied:
                # An intercept-only target still needs the caller's row axis.
                # Inspect lengths, without validating an unused block's values.
                sizes = {
                    len(value.row_ids)
                    if isinstance(value, MeasureColumn)
                    else len(value)
                    for value in data.values()
                    if isinstance(value, MeasureColumn)
                    or (
                        isinstance(value, Sized)
                        and not isinstance(value, str)
                        and getattr(value, "ndim", None) != 0
                    )
                }
                if len(sizes) > 1:
                    message = (
                        "Intercept-only prediction needs one unambiguous row count."
                    )
                    raise ValueError(message)
                return Observations({}, row_ids=tuple(range(next(iter(sizes), 1))))
            data = supplied
        return as_observations(data, roles, broadcast_scalars=True)

    def likelihood_summary(self) -> LikelihoodSummary:
        """Return the two-part log likelihood with its parameter and row counts.

        The log likelihood is the gate log likelihood over every row plus the
        positive-count log likelihood over the positive rows, each with its
        constants. ``k`` adds the gate and count coefficients; the truncated
        Poisson count part has no dispersion and the gate has none. ``n`` is the
        number of rows entering the gate.
        """
        return LikelihoodSummary(
            float(self.gate_fit.log_likelihood + self.count_fit.log_likelihood),
            int(self.gate_fit.parameters.shape[-1])
            + int(self.count_fit.parameters.shape[-1]),
            int(self.gate_fit.n_observations),
            "A component fit is invalid, so the hurdle log likelihood is unavailable.",
        )

    def fit_statistics(self) -> FitStatistics:
        """Return the information criteria derived from the likelihood summary."""
        return FitStatistics.from_likelihood(self.likelihood_summary())

    def report(self) -> Report:
        """Label coefficients by their parameter slot and retain failure evidence."""
        statistics = self.fit_statistics()
        parameters = tuple(
            ParameterRow(
                f"{slot}.{name}",
                float(value) if bool(fit.valid) else None,
                valid=bool(fit.valid),
                status="available" if bool(fit.valid) else "unavailable",
                unavailable_reason=None
                if bool(fit.valid)
                else "Component estimate unavailable.",
            )
            for slot, fit, design in (
                ("zero_probability", self.gate_fit, self.gate),
                ("log_rate", self.count_fit, self.count),
            )
            for name, value in zip(design.column_names, fit.parameters, strict=True)
        )
        return Report(
            "Hurdle Poisson",
            parameters=parameters,
            metrics=statistics.metrics,
            notes=statistics.notes,
            metadata={
                "valid": self.valid,
                "exposure": self.exposure,
                "gate_offset": self.offset,
                "count_offset": self.count_offset,
            },
        )

    def _repr_html_(self) -> str:
        return self.report()._repr_html_()


@dataclass(frozen=True)
class _HurdleHost(Model[RegressionInput, _PreparedHurdle, HurdleResult]):
    """Independent gate and positive-count likelihood blocks with frozen designs."""

    response: str
    law: HurdlePoisson
    terms: Mapping[str, Term]
    offsets: Mapping[str, str]
    exposure: str | None = None

    def _predictors(self) -> tuple[_Predictor, _Predictor]:
        return (
            _Predictor(
                self.terms["zero_probability"],
                self.law.zero_link,
                self.offsets.get("zero_probability"),
            ),
            _Predictor(
                self.terms["count"], "log", self.offsets.get("count"), self.exposure
            ),
        )

    @override
    def prepare(self, data: RegressionInput) -> _PreparedHurdle:
        gate, count = self._predictors()
        observed: dict[str, InputRole] = {
            name: "numeric"
            for name in (self.response, gate.offset, count.offset, count.exposure)
            if name is not None
        }
        rows = as_observations(
            data,
            merge_input_roles(
                gate.terms.input_columns, count.terms.input_columns, observed
            ),
            broadcast_scalars=False,
        )
        y = rows.numeric(self.response)
        if not bool(jnp.all((y >= 0) & (y == jnp.floor(y)))):
            message = "A hurdle response requires nonnegative integer counts."
            raise ValueError(message)
        if count.exposure is not None and not bool(
            jnp.all(rows.numeric(count.exposure) > 0)
        ):
            message = "Count exposure must be positive."
            raise ValueError(message)
        positive_indices = jnp.flatnonzero(y > 0).tolist()
        if not positive_indices:
            message = "A positive-count design requires observed positive counts."
            raise ValueError(message)
        positive = rows.take(positive_indices)
        return _PreparedHurdle(
            rows,
            FittedDesign(gate.terms._prepare_term(rows)),
            positive,
            FittedDesign(count.terms._prepare_term(positive)),
            self._token,
        )

    @override
    def _estimate_prepared(self, prepared: _PreparedHurdle) -> HurdleResult:
        gate, count = self._predictors()
        rows, positive = prepared.data, prepared.positive
        response = rows.numeric(self.response)
        x = prepared.gate.transform(rows)
        x = x.astype(jnp.result_type(x, response, 1.0))
        zeros = (response == 0).astype(x.dtype)
        offset = 0 if gate.offset is None else rows.numeric(gate.offset)
        names = prepared.gate.column_names
        gate_fit: GateFit
        if gate.link == "logit":
            gate_fit = BinomialLogitMLE().estimate(
                BinomialProblem(x, zeros, offset=offset, parameter_names=names)
            )
        else:
            gate_fit = ProbitMLE().estimate(
                LikelihoodProblem(x, zeros, offset=offset, parameter_names=names)
            )
        count_fit = PositivePoissonMLE().estimate(
            LikelihoodProblem(
                prepared.count.transform(positive),
                positive.numeric(self.response),
                exposure=1
                if count.exposure is None
                else positive.numeric(count.exposure),
                offset=0 if count.offset is None else positive.numeric(count.offset),
                parameter_names=prepared.count.column_names,
            )
        )
        return HurdleResult(
            gate_fit,
            count_fit,
            prepared.gate,
            prepared.count,
            gate.offset,
            count.offset,
            count.exposure,
        )


type ConditionalLaw = (
    Binomial[Literal["logit", "probit"]]
    | Poisson
    | Gamma
    | NegativeBinomial2
    | HurdlePoisson
)
type ConditionalResult = (
    LogisticResult | PoissonResult | LikelihoodResult | HurdleResult
)
type _BinomialEstimator = Estimator[BinomialProblem, GLMFit[BinomialLogitEquation]]
type ConditionalEstimator = (
    _BinomialEstimator
    | Estimator[PoissonProblem, GLMFit[PoissonLogEquation]]
    | Estimator[LikelihoodProblem, LikelihoodFit]
)
type _ConditionalHost = (
    _LogitHost | _PoissonHost | _ProbitHost | _GammaHost | _NB2Host | _HurdleHost
)

__all__ = [
    "Binomial",
    "ConditionalEstimator",
    "ConditionalLaw",
    "ConditionalResult",
    "Gamma",
    "Gaussian",
    "HurdlePoisson",
    "HurdleResult",
    "NegativeBinomial2",
    "Observed",
    "Poisson",
]


def _conditional_host(
    observed: Observed,
    terms: Term | Mapping[str, Term],
    law: ConditionalLaw,
    estimator: ConditionalEstimator | None,
    offset: str | Mapping[str, str] | None,
    exposure: str | None,
) -> _ConditionalHost:
    response = observed.response
    if not isinstance(law, Binomial) and observed.trials is not None:
        message = "Only Binomial accepts trial totals."
        raise ValueError(message)
    if isinstance(law, HurdlePoisson):
        return _hurdle_host(observed.response, terms, law, estimator, offset, exposure)
    terms, offset = _single_predictor(terms, law, offset, exposure)
    if isinstance(law, Binomial):
        p = _Predictor(terms, law.link, offset)
        if p.link == "logit":
            return _LogitHost(
                response,
                p.terms,
                trials=observed.trials,
                offset=p.offset,
                estimator=cast(
                    "_BinomialEstimator | None",
                    estimator,
                ),
            )
        return _ProbitHost(
            response,
            p.terms,
            trials=observed.trials,
            offset=p.offset,
            estimator=cast(
                "Estimator[LikelihoodProblem, LikelihoodFit]",
                estimator or ProbitMLE(),
            ),
        )
    if isinstance(law, Poisson):
        rate = _Predictor(terms, offset=offset, exposure=exposure)
        return _PoissonHost(
            response,
            rate.terms,
            exposure=rate.exposure,
            offset=rate.offset,
            estimator=cast(
                "Estimator[PoissonProblem, GLMFit[PoissonLogEquation]] | None",
                estimator,
            ),
        )
    if isinstance(law, Gamma):
        mean = _Predictor(terms, offset=offset, exposure=exposure)
        estimator = estimator or GammaMLE(law.dispersion)
        if isinstance(estimator, GammaMLE) and estimator.dispersion != law.dispersion:
            message = "Estimator dispersion policy must match the Gamma law."
            raise ValueError(message)
        return _GammaHost(
            response,
            mean.terms,
            estimator=cast(
                "Estimator[LikelihoodProblem, LikelihoodFit]",
                estimator,
            ),
            dispersion=law.dispersion,
            offset=mean.offset,
        )
    if isinstance(law, NegativeBinomial2):
        mean = _Predictor(terms, offset=offset, exposure=exposure)
        estimator = estimator or NegativeBinomialMLE(law.dispersion)
        if (
            isinstance(estimator, NegativeBinomialMLE)
            and estimator.dispersion != law.dispersion
        ):
            message = "Estimator dispersion policy must match the NB2 law."
            raise ValueError(message)
        return _NB2Host(
            response,
            mean.terms,
            estimator=cast(
                "Estimator[LikelihoodProblem, LikelihoodFit]",
                estimator,
            ),
            dispersion=law.dispersion,
            offset=mean.offset,
            exposure=mean.exposure,
        )
    message = "Unsupported conditional response law."
    raise TypeError(message)


def _hurdle_host(
    response: str,
    terms: Term | Mapping[str, Term],
    law: HurdlePoisson,
    estimator: ConditionalEstimator | None,
    offset: str | Mapping[str, str] | None,
    exposure: str | None,
) -> _HurdleHost:
    if estimator is not None:
        message = "HurdlePoisson uses its qualified native estimators."
        raise ValueError(message)
    if not isinstance(terms, Mapping) or set(terms) != {
        "zero_probability",
        "count",
    }:
        message = "Hurdle terms must name zero_probability and count."
        raise ValueError(message)
    if offset is not None and not isinstance(offset, Mapping):
        message = "Hurdle offsets must name their predictor blocks."
        raise TypeError(message)
    offsets = {} if offset is None else dict(offset)
    if not set(offsets) <= {"zero_probability", "count"}:
        message = "Unknown hurdle offset block."
        raise ValueError(message)
    return _HurdleHost(response, law, dict(terms), offsets, exposure)


def _single_predictor(
    terms: Term | Mapping[str, Term],
    law: ConditionalLaw,
    offset: str | Mapping[str, str] | None,
    exposure: str | None,
) -> tuple[Term, str | None]:
    if not isinstance(terms, Term):
        message = "A single-predictor law requires one Term expression."
        raise TypeError(message)
    if offset is not None and not isinstance(offset, str):
        message = "A single-predictor offset must name one column."
        raise TypeError(message)
    if exposure is not None and isinstance(law, (Binomial, Gamma)):
        message = "This law does not consume exposure."
        raise ValueError(message)
    return terms, offset
