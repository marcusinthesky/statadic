"""Explicit diagnostic requests evaluated against an estimated result.

``result.diagnose(*requests)`` is the single diagnostic entry point. Each request
names one procedure, its options and its null hypothesis; a result family
accepts only requests that implement its capability (``evaluate_linear`` for
OLS). Nothing runs implicitly during estimation or inference, and no request
refits, repairs or replaces the retained estimate.

A diagnostic answers a narrower question than "are the assumptions met?".
Failing to reject a null is not evidence that it holds, residuals are not the
unobserved errors, and exogeneity cannot be checked from OLS residuals at all.
Influence quantities describe sensitivity to deleting a row; they do not mark
an observation as wrong. Unavailable diagnostics remain in the collection with
their reason rather than disappearing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from statadic.api.data import _resolve_row_order, _row_order_metadata
from statadic.api.hypotheses import (
    ChiSquare,
    ConstantVariance,
    FisherF,
    GaussianMoments,
    Hypothesis,
    NoAddedFittedPowerEffect,
    NoSerialCorrelation,
    OveridentifyingRestrictions,
    Proposition,
    TestOutcome,
)
from statadic.api.reporting import Metric, Report, _row_identity
from statadic.core.results import InferenceStatus
from statadic.inference.gmm import hansen_j_from_fit
from statadic.inference.iv import hansen_j
from statadic.inference.linear_diagnostics import (
    breusch_godfrey,
    breusch_pagan,
    durbin_watson,
    jarque_bera,
    ols_influence,
    reset_test,
    residual_moments,
    white_test,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from statadic.api.base import FitContext
    from statadic.api.design import FittedDesign
    from statadic.api.inference import InferenceContext
    from statadic.api.reporting import MetadataValue
    from statadic.core.equations import VariabilityScheme
    from statadic.core.typing import Array
    from statadic.estimation.gmm import AffineGMMFit
    from statadic.estimation.iv import IVFit
    from statadic.estimation.linear import LinearFit
    from statadic.inference.linear_diagnostics import LinearInfluence, ResidualTest

__all__ = [
    "RESET",
    "BreuschGodfrey",
    "BreuschPagan",
    "Diagnostic",
    "Diagnostics",
    "DurbinWatson",
    "HansenJ",
    "IVDiagnostic",
    "Influence",
    "InfluenceDiagnostic",
    "JarqueBera",
    "LinearDiagnostic",
    "MomentsDiagnostic",
    "RetainedIVFit",
    "RetainedLinearFit",
    "RetainedMomentsFit",
    "White",
]


@runtime_checkable
class RetainedLinearFit(Protocol):
    """The retained state of an OLS result that linear diagnostics consume."""

    @property
    def core(self) -> LinearFit:
        """Return the actual native least-squares fit."""
        ...

    @property
    def context(self) -> FitContext:
        """Return the aligned training observations and model identity."""
        ...

    @property
    def design(self) -> FittedDesign:
        """Return the retained fitted design."""
        ...


@dataclass(frozen=True)
class InfluenceDiagnostic:
    """Per-row deletion sensitivities of the actual fit, aligned to ``row_ids``."""

    name: str
    row_ids: tuple[int | str, ...]
    parameter_names: tuple[str, ...]
    leverage: Array = field(repr=False)
    studentized_internal: Array = field(repr=False)
    studentized_external: Array = field(repr=False)
    cooks_distance: Array = field(repr=False)
    dffits: Array = field(repr=False)
    dfbetas: Array = field(repr=False)
    leverage_valid: Array = field(repr=False)
    studentized_internal_valid: Array = field(repr=False)
    studentized_external_valid: Array = field(repr=False)
    cooks_distance_valid: Array = field(repr=False)
    dffits_valid: Array = field(repr=False)
    dfbetas_valid: Array = field(repr=False)
    available: bool
    unavailable_reason: str | None = None
    sources: tuple[str, ...] = ()

    def metrics(self) -> tuple[Metric, ...]:
        """Summarize each quantity over its own available observation rows."""
        metrics: list[Metric] = []
        for name, label, values, valid in (
            (
                "cooks_distance",
                "Cook's distance",
                self.cooks_distance,
                self.cooks_distance_valid,
            ),
            ("leverage", "leverage", self.leverage, self.leverage_valid),
        ):
            if not bool(jnp.any(valid)):
                metrics.append(
                    Metric(
                        f"{self.name}.max_{name}",
                        None,
                        f"Largest {label}.",
                        available=False,
                        unavailable_reason=self.unavailable_reason
                        or f"{name}-unavailable",
                        status="unavailable",
                    )
                )
                continue
            index = int(jnp.argmax(jnp.where(valid, values, -jnp.inf)))
            metrics.append(
                Metric(
                    f"{self.name}.max_{name}",
                    float(values[index]),
                    f"Largest {label}, at row {self.row_ids[index]!r}.",
                )
            )
        return tuple(metrics)


type Diagnostic = TestOutcome | InfluenceDiagnostic


def _diagnostic_requests[RequestT](
    requests: tuple[RequestT, ...], capability: type[RequestT]
) -> tuple[RequestT, ...]:
    """Validate a family's explicit request list without discovering procedures."""
    if not requests:
        message = "Request at least one diagnostic."
        raise ValueError(message)
    for request in requests:
        if not isinstance(request, capability):
            message = (
                f"{type(request).__name__} does not support {capability.__name__}."
            )
            raise TypeError(message)
    return requests


@runtime_checkable
class LinearDiagnostic(Protocol):
    """A diagnostic request that an OLS result can evaluate."""

    def evaluate_linear(self, fitted: RetainedLinearFit) -> Diagnostic:
        """Evaluate this request against the retained OLS fit."""
        ...


@runtime_checkable
class RetainedIVFit(Protocol):
    """Native IV state and the identities to which it belongs."""

    @property
    def core(self) -> IVFit:
        """Return the actual IV fit."""
        ...

    @property
    def context(self) -> InferenceContext:
        """Return the retained observation identities."""
        ...


@runtime_checkable
class RetainedMomentsFit(Protocol):
    """Native affine GMM state and its retained observation identities."""

    @property
    def core(self) -> AffineGMMFit[VariabilityScheme]:
        """Return the actual affine GMM fit."""
        ...

    @property
    def context(self) -> InferenceContext:
        """Return the retained observation identities."""
        ...


@runtime_checkable
class IVDiagnostic(Protocol):
    """A diagnostic that a structural IV result can evaluate."""

    def evaluate_iv(self, fitted: RetainedIVFit) -> TestOutcome:
        """Evaluate against the retained IV state."""
        ...


@runtime_checkable
class MomentsDiagnostic(Protocol):
    """A diagnostic that an affine GMM result can evaluate."""

    def evaluate_moments(self, fitted: RetainedMomentsFit) -> TestOutcome:
        """Evaluate against the retained moment state."""
        ...


@dataclass(frozen=True)
class Diagnostics:
    """Ordered diagnostic outcomes for one estimated result, by unique name."""

    model_name: str
    results: tuple[Diagnostic, ...]

    def __post_init__(self) -> None:
        """Require unique names so every outcome is addressable."""
        names = [result.name for result in self.results]
        if len(set(names)) != len(names):
            message = "Each diagnostic in one request must have a distinct name."
            raise ValueError(message)

    def __getitem__(self, name: str) -> Diagnostic:
        """Return the outcome with ``name``."""
        for result in self.results:
            if result.name == name:
                return result
        message = f"No diagnostic named {name!r}; available: {self.names}."
        raise KeyError(message)

    def __iter__(self) -> Iterator[Diagnostic]:
        """Iterate over outcomes in request order."""
        return iter(self.results)

    def __len__(self) -> int:
        """Return the number of requested diagnostics."""
        return len(self.results)

    @property
    def names(self) -> tuple[str, ...]:
        """Return outcome names in request order."""
        return tuple(result.name for result in self.results)

    def report(self) -> Report:
        """Report every outcome, keeping unavailable diagnostics visible."""
        reports = {
            result.name: result.report()
            for result in self.results
            if isinstance(result, TestOutcome)
        }
        unavailable = tuple(
            f"{result.name}: {result.unavailable_reason}"
            for result in self.results
            if not result.available and isinstance(result, InfluenceDiagnostic)
        )
        return Report(
            title=f"{self.model_name} diagnostics",
            tests=tuple(
                result.test_row()
                for result in self.results
                if isinstance(result, TestOutcome)
            ),
            metrics=tuple(
                metric
                for result in self.results
                for metric in (
                    result.components()
                    if isinstance(result, TestOutcome)
                    else result.metrics()
                )
            ),
            metadata={
                "diagnostics": self.names,
                "outcomes": {name: report.metadata for name, report in reports.items()},
            },
            warnings=tuple(
                dict.fromkeys(
                    (
                        *(
                            warning
                            for report in reports.values()
                            for warning in report.warnings
                        ),
                        *(f"Unavailable diagnostic {item}." for item in unavailable),
                    )
                )
            ),
            notes=(
                "Failing to reject a null hypothesis is not evidence that it holds.",
                (
                    "p-values use each outcome's stated reference law and assumptions; "
                    "statistics without a calibrated reference have no p-value."
                ),
            ),
        )

    def _repr_html_(self) -> str:
        """Render the report for notebook display."""
        return self.report()._repr_html_()


def _common_reason(fit: LinearFit) -> str | None:
    """Explain why no residual diagnostic is defined for this fit, if any."""
    if not bool(fit.valid):
        return "invalid-fit"
    weights = fit.relative_precision
    if not bool(jnp.all(weights == weights[..., :1])):
        return "weighted-fit"
    if int(fit.df_resid) <= 0:
        return "no-residual-degrees-of-freedom"
    return None


def _test_result[NullT: Proposition](
    name: str,
    null: NullT,
    test: ResidualTest,
    fit: LinearFit,
    sources: tuple[str, ...],
    *,
    specific_reason: str,
    details: Mapping[str, float] | None = None,
    construction: Mapping[str, MetadataValue] | None = None,
    assumptions: tuple[str, ...] = (),
) -> TestOutcome[NullT]:
    available = bool(test.valid)
    reference = None
    pvalue = None
    if available and test.reference == "chi2":
        reference = ChiSquare(float(test.df))
        pvalue = float(test.pvalue)
    elif available and test.reference == "F":
        reference = FisherF(float(test.df), float(test.df_denominator))
        pvalue = float(test.pvalue)
    return TestOutcome(
        name,
        Hypothesis(null),
        float(test.statistic) if available else None,
        pvalue,
        reference,
        available=available,
        unavailable_reason=None
        if available
        else (_common_reason(fit) or specific_reason),
        details={} if details is None else dict(details),
        construction={
            **({} if construction is None else construction),
            "native_valid": available,
            "reference_requested": test.reference,
            "reference_degrees": float(test.df)
            if available and test.reference != "none"
            else None,
            "denominator_degrees": float(test.df_denominator)
            if available and test.reference == "F"
            else None,
        },
        assumptions=assumptions,
        sources=sources,
        evidence=test,
    )


def _linear_construction(fitted: RetainedLinearFit) -> dict[str, MetadataValue]:
    """Describe an order-insensitive diagnostic's sample by fixed-size identity."""
    rows = fitted.context.row_ids
    return {
        "rows": _row_identity(rows),
        "parameter_names": fitted.design.column_names,
        "sample_count": len(rows),
    }


def _ordered_construction(
    fitted: RetainedLinearFit, row_order: Array | None, order_by: str | None
) -> dict[str, MetadataValue]:
    """Add the explicit ordering choice, still without sample-sized id tuples."""
    return {
        **_linear_construction(fitted),
        **_row_order_metadata(fitted.context.observations, order_by, row_order),
    }


def _varying_columns(fitted: RetainedLinearFit) -> tuple[str, ...]:
    """Return mean-design columns that vary, so a constant is never a variance effect.

    A design without varying columns keeps its names; its test is unavailable.
    """
    design = fitted.core.design
    varying = jax.device_get(jnp.any(design != design[:1], axis=0)).tolist()
    names = tuple(
        name
        for name, varies in zip(fitted.design.column_names, varying, strict=False)
        if varies
    )
    return names or fitted.design.column_names


@dataclass(frozen=True)
class JarqueBera:
    """Residual skewness and kurtosis against Gaussian values, chi-square(2)."""

    name: str = "jarque_bera"

    def evaluate_linear(
        self, fitted: RetainedLinearFit
    ) -> TestOutcome[GaussianMoments]:
        """Evaluate against the actual OLS residuals."""
        fit = fitted.core
        skewness, kurtosis = residual_moments(fit)
        test = jarque_bera(fit)
        return _test_result(
            self.name,
            GaussianMoments(),
            test,
            fit,
            ("jarque-bera-1987",),
            specific_reason="zero-residual-variance",
            details={"skewness": float(skewness), "kurtosis": float(kurtosis)}
            if bool(test.valid)
            else None,
            construction=_linear_construction(fitted),
            assumptions=(
                (
                    "Independent identically distributed Gaussian errors under "
                    "the null; "
                    "asymptotic chi-square calibration."
                ),
            ),
        )


@dataclass(frozen=True)
class BreuschPagan:
    """Error variance that changes linearly with a variance design.

    ``variables`` names numeric training columns; a constant is added. By
    default the mean design is used and must contain a constant.
    ``studentized=False`` selects the original statistic, whose reference
    additionally requires Gaussian errors.
    """

    variables: tuple[str, ...] | None = None
    studentized: bool = True
    name: str = "breusch_pagan"

    def evaluate_linear(
        self, fitted: RetainedLinearFit
    ) -> TestOutcome[ConstantVariance]:
        """Evaluate against the actual OLS residuals and retained observations."""
        fit = fitted.core
        variance_design = None
        if self.variables is not None:
            if not self.variables:
                message = "BreuschPagan variables must name at least one column."
                raise ValueError(message)
            observations = fitted.context.observations
            columns = [observations.resolve_numeric(name) for name in self.variables]
            dtype = fit.residuals.dtype
            variance_design = jnp.column_stack(
                [jnp.ones(observations.n_rows, dtype=dtype)]
                + [jnp.asarray(column, dtype=dtype) for column in columns]
            )
        test = breusch_pagan(fit, variance_design, studentized=self.studentized)
        variance_columns = self.variables or _varying_columns(fitted)
        form = "Koenker studentized" if self.studentized else "original Gaussian"
        sources = (
            ("breusch-pagan-1979", "koenker-1981")
            if self.studentized
            else ("breusch-pagan-1979",)
        )
        return _test_result(
            self.name,
            ConstantVariance(variance_columns),
            test,
            fit,
            sources,
            specific_reason="variance-design-lacks-constant-or-variation",
            construction={
                **_linear_construction(fitted),
                "formulation": form,
                "auxiliary_response": "squared residuals"
                if self.studentized
                else "squared residuals divided by mean squared residual",
                "variance_columns": variance_columns,
                "auxiliary_rank": int(test.df) + 1 if bool(test.valid) else None,
            },
            assumptions=(
                (
                    "Correct conditional mean, independent observations and valid "
                    "score-test asymptotics."
                ),
                "Studentized Koenker variance formulation."
                if self.studentized
                else "Gaussian errors for the original Breusch-Pagan formulation.",
            ),
        )


@dataclass(frozen=True)
class White:
    """Studentized variance test on all products of the mean design columns."""

    name: str = "white"

    def evaluate_linear(
        self, fitted: RetainedLinearFit
    ) -> TestOutcome[ConstantVariance]:
        """Evaluate against the actual OLS residuals and retained design."""
        names = _varying_columns(fitted)
        products = (
            *names,
            *(
                f"{first}*{second}"
                for index, first in enumerate(names)
                for second in names[index:]
            ),
        )
        test = white_test(fitted.core)
        return _test_result(
            self.name,
            ConstantVariance(products),
            test,
            fitted.core,
            ("white-1980",),
            specific_reason="design-lacks-constant-or-variation",
            construction={
                **_linear_construction(fitted),
                "auxiliary_response": "squared residuals",
                "variance_columns": products,
                "auxiliary_rank": int(test.df) + 1 if bool(test.valid) else None,
            },
            assumptions=(
                (
                    "Correct conditional mean, independent observations and "
                    "studentized variance-score asymptotics."
                ),
            ),
        )


@dataclass(frozen=True)
class DurbinWatson:
    """First-order residual autocorrelation statistic in the selected row order.

    Its null distribution depends on the design, so no p-value is reported.
    """

    name: str = "durbin_watson"
    order_by: str | None = field(kw_only=True)

    def evaluate_linear(
        self, fitted: RetainedLinearFit
    ) -> TestOutcome[NoSerialCorrelation]:
        """Evaluate the actual OLS residuals under the explicit ordering choice."""
        order = _resolve_row_order(fitted.context.observations, self.order_by)
        return _test_result(
            self.name,
            NoSerialCorrelation(1, self.order_by),
            durbin_watson(fitted.core, row_order=order),
            fitted.core,
            ("durbin-watson-1950",),
            specific_reason="zero-residual-variance",
            construction={
                **_ordered_construction(fitted, order, self.order_by),
                "calibration": "statistic only; design-specific null distribution",
            },
            assumptions=(
                (
                    "Chosen row adjacency defines positional lags; sorting establishes "
                    "neither equal time spacing nor a calibrated reference."
                ),
            ),
        )


@dataclass(frozen=True)
class BreuschGodfrey:
    """Residual serial correlation through ``lags`` in the selected row order."""

    lags: int = 1
    name: str = "breusch_godfrey"
    order_by: str | None = field(kw_only=True)

    def evaluate_linear(
        self, fitted: RetainedLinearFit
    ) -> TestOutcome[NoSerialCorrelation]:
        """Evaluate the actual OLS residuals under the explicit ordering choice."""
        order = _resolve_row_order(fitted.context.observations, self.order_by)
        return _test_result(
            self.name,
            NoSerialCorrelation(self.lags, self.order_by),
            breusch_godfrey(fitted.core, self.lags, row_order=order),
            fitted.core,
            ("breusch-1978", "godfrey-1978"),
            specific_reason="auxiliary-lag-design-unavailable",
            construction={
                **_ordered_construction(fitted, order, self.order_by),
                "auxiliary_response": "residuals",
                "pre_sample_lags": "zero-filled; all original observations retained",
            },
            assumptions=(
                (
                    "Correct conditional mean and serial-correlation LM asymptotics "
                    "in the declared order; sorting establishes neither equal "
                    "spacing nor stationarity."
                ),
            ),
        )


@dataclass(frozen=True)
class RESET:
    """Ramsey's test for omitted powers of the fitted values."""

    powers: tuple[int, ...] = (2, 3)
    reference: Literal["chi2", "F"] = "chi2"
    name: str = "reset"

    def evaluate_linear(
        self, fitted: RetainedLinearFit
    ) -> TestOutcome[NoAddedFittedPowerEffect]:
        """Evaluate by augmenting the retained design, without refitting the model."""
        return _test_result(
            self.name,
            NoAddedFittedPowerEffect(self.powers),
            reset_test(fitted.core, self.powers, self.reference),
            fitted.core,
            ("ramsey-1969",),
            specific_reason="added-powers-not-identified",
            construction={
                **_linear_construction(fitted),
                "powers": self.powers,
                "covariance": "classical augmented-fit covariance",
                "restriction_rank": "identified added fitted-power span",
            },
            assumptions=(
                (
                    "Correct conditional mean and homoskedastic independent errors; "
                    "the reference needs its classical F or asymptotic chi-square "
                    "justification."
                ),
            ),
        )


@dataclass(frozen=True)
class HansenJ:
    """Conventional overidentification reference for an actual efficient IV/GMM fit.

    Acknowledgement covers correct moments, strong identification, an interior
    true parameter and the applicable dependence/bandwidth asymptotic regime.
    Those assumptions cannot be established from the fitted arrays.
    """

    assume_correct_specification: bool
    name: str = "hansen_j"

    def __post_init__(self) -> None:
        """Require an explicit boolean acknowledgement, including explicit False."""
        if type(self.assume_correct_specification) is not bool:
            message = "HansenJ requires a boolean assumption acknowledgement."
            raise TypeError(message)

    def evaluate_iv(
        self, fitted: RetainedIVFit
    ) -> TestOutcome[OveridentifyingRestrictions]:
        """Preserve the native IV weight-provenance and reference guards."""
        fit = fitted.core
        native = hansen_j(
            fit, assume_correct_specification=self.assume_correct_specification
        )
        available = bool(native.valid)
        return TestOutcome(
            self.name,
            Hypothesis(
                OveridentifyingRestrictions(
                    fit.instruments.shape[1], fit.design.shape[1]
                )
            ),
            float(native.statistic) if available else None,
            float(native.pvalue) if available else None,
            ChiSquare(float(native.degrees_of_freedom)) if available else None,
            available,
            unavailable_reason=None
            if available
            else _iv_hansen_reason(fit, acknowledged=self.assume_correct_specification),
            details={"degrees_of_freedom": int(native.degrees_of_freedom)},
            construction={
                "rows": _row_identity(fitted.context.row_ids),
                "native_valid": available,
                "weight_provenance": native.provenance,
                "weight_iteration_converged": bool(fit.converged),
                "variability_scheme": fit.scheme.__name__,
                "bandwidth": fit.bandwidth,
                "weight_iterations": int(fit.iterations),
                "clusters": fit.n_clusters,
                "conventional_assumptions": self.assume_correct_specification,
            },
            assumptions=_HANSEN_ASSUMPTIONS,
            sources=("hansen-1982",),
        )

    def evaluate_moments(
        self, fitted: RetainedMomentsFit
    ) -> TestOutcome[OveridentifyingRestrictions]:
        """Preserve actual retained GMM weighting, convergence and native status."""
        fit = fitted.core
        native = hansen_j_from_fit(
            fit, assume_correct_specification=self.assume_correct_specification
        )
        available = bool(native.valid)
        error = float(fit.weight_fixed_point_error)
        return TestOutcome(
            self.name,
            Hypothesis(
                OveridentifyingRestrictions(
                    fit.scores.values.shape[1], fit.parameters.shape[0]
                )
            ),
            float(native.test.statistic) if available else None,
            float(native.test.pvalue) if available else None,
            ChiSquare(float(native.degrees_of_freedom)) if available else None,
            available,
            unavailable_reason=None
            if available
            else InferenceStatus(int(native.status)).name,
            details={"degrees_of_freedom": int(native.degrees_of_freedom)},
            construction={
                "rows": _row_identity(fitted.context.row_ids),
                "native_status": int(native.status),
                "native_valid": available,
                "overidentified": bool(native.overidentified),
                "efficient_weight": bool(native.efficient_weight),
                "conventional_assumptions": bool(native.conventional_assumptions),
                "fixed_weight": bool(fit.fixed_weight),
                "solver_converged": bool(fit.solver.converged),
                "weight_fixed_point_error": error
                if bool(jnp.isfinite(error))
                else None,
                "bandwidth": fit.bandwidth,
                "variability_scheme": fit.scheme.__name__,
            },
            assumptions=_HANSEN_ASSUMPTIONS,
            sources=("hansen-1982",),
        )


def _iv_hansen_reason(fit: IVFit, *, acknowledged: bool) -> str:
    """Name the first failed native IV Hansen J eligibility condition.

    Mirrors ``statadic.inference.iv.hansen_j``; the kernel decides availability.
    """
    if not acknowledged:
        return "conventional-assumptions-not-acknowledged"
    if fit.method not in {"two-step", "iterated"}:
        return "weight-not-efficient"
    if fit.instruments.shape[1] <= fit.design.shape[1]:
        return "not-overidentified"
    if not bool(fit.valid):
        return "invalid-fit"
    if fit.method == "iterated" and not bool(fit.converged):
        return "weight-iteration-not-converged"
    return "variability-not-positive-definite"


_HANSEN_ASSUMPTIONS = (
    "Correctly specified moments at a strongly identified interior true parameter.",
    "Valid dependence and bandwidth asymptotics for the retained moment scores.",
    (
        "Conventional efficient-weight Hansen J calibration; array checks do not "
        "establish these assumptions."
    ),
)


@dataclass(frozen=True)
class Influence:
    """Leverage, studentized residuals, Cook's distance, DFFITS and DFBETAS."""

    name: str = "influence"

    def evaluate_linear(self, fitted: RetainedLinearFit) -> InfluenceDiagnostic:
        """Evaluate row-deletion sensitivities of the actual OLS fit."""
        fit = fitted.core
        influence: LinearInfluence = ols_influence(fit)
        available = bool(influence.valid)
        return InfluenceDiagnostic(
            self.name,
            fitted.context.row_ids,
            fit.parameter_names or fitted.design.column_names,
            influence.leverage,
            influence.studentized_internal,
            influence.studentized_external,
            influence.cooks_distance,
            influence.dffits,
            influence.dfbetas,
            influence.leverage_valid,
            influence.studentized_internal_valid,
            influence.studentized_external_valid,
            influence.cooks_distance_valid,
            influence.dffits_valid,
            influence.dfbetas_valid,
            available=available,
            unavailable_reason=None
            if available
            else (_common_reason(fit) or "linear-influence-ineligible"),
            sources=("cook-1977", "belsley-kuh-welsch-1980"),
        )
