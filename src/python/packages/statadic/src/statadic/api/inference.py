"""Shared host covariance requests and fitted-uncertainty reporting."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol, runtime_checkable

import jax.numpy as jnp

from statadic.api.design import (  # noqa: TC001
    DesignContrast,
    FittedDesign,
    LinearPredictorContrast,
    RowDifference,
)
from statadic.api.fit_statistics import FitStatistics
from statadic.api.hypotheses import (
    ChiSquare,
    FisherF,
    Hypothesis,
    LinearRestriction,
    TestOutcome,
)
from statadic.api.reporting import (
    MetadataValue,
    Metric,
    ParameterRow,
    Report,
    ReportNumber,  # noqa: TC001 -- public detail annotation
    _confidence_label,
    _row_identity,
    freeze_metadata,
)
from statadic.core.results import InferenceStatus
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.inference.contracts import FittedUncertainty
from statadic.inference.parameters import (
    CoefficientIntervals,  # noqa: TC001
    coefficient_intervals,
    linear_restriction_test,
    normal_critical_value,
)

__all__ = [
    "HAC",
    "HC",
    "Clustered",
    "ContrastResult",
    "CovarianceSpec",
    "InferenceContext",
    "InferenceResult",
    "ModelBased",
]


def _null_value(value: object) -> float:
    """Return one finite real null, rejecting booleans and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = "A coefficient null must be a finite real number."
        raise TypeError(message)
    if not math.isfinite(value):
        message = "A coefficient null must be finite."
        raise ValueError(message)
    return float(value)


def _null_note(null: float | Mapping[str, float], reference: str | None) -> str:
    """State the null, the two-sided alternative and the reference."""
    if isinstance(null, Mapping):
        stated = ", ".join(f"{key} = {value:g}" for key, value in null.items())
        return f"Two-sided tests of {stated}; {reference} reference."
    return f"Two-sided tests against {null:g}; {reference} reference."


@dataclass(frozen=True)
class CovarianceSpec:
    """Base of every explicit covariance request a result's ``infer`` accepts.

    Each model family subclasses it with its supported options, including
    ``statadic.api.panel.PanelCovariance``. It selects no reference distribution.
    """


@dataclass(frozen=True)
class ModelBased(CovarianceSpec):
    """Request the supported family-specific model covariance.

    Args:
        finite_sample_correction: Multiplier applied to the covariance.

    Examples:
        >>> ModelBased().finite_sample_correction
        1.0

    """

    finite_sample_correction: float = 1.0


@dataclass(frozen=True)
class HC(CovarianceSpec):
    """Request an explicit supported independent-row HC variant.

    Args:
        kind: HC0, HC1, HC2 or HC3, subject to the model family's support.
        finite_sample_correction: Additional covariance multiplier.

    Examples:
        >>> HC("HC3").kind
        'HC3'

    """

    kind: Literal["HC0", "HC1", "HC2", "HC3"] = "HC0"
    finite_sample_correction: float = 1.0


@dataclass(frozen=True)
class HAC(CovarianceSpec):
    """Request Bartlett HAC covariance for an ordered sequence.

    Args:
        bandwidth: Maximum lag used by the Bartlett weighting scheme.
        finite_sample_correction: Additional covariance multiplier.
        order_by: Numeric ordering column, or ``None`` to use retained order.

    Examples:
        >>> HAC(2, order_by="time").bandwidth
        2

    """

    bandwidth: int
    finite_sample_correction: float = 1.0
    order_by: str | None = field(kw_only=True)


@dataclass(frozen=True)
class Clustered(CovarianceSpec):
    """Request covariance from one-way clusters of retained observations.

    Args:
        column: Name of the aligned cluster-label column.
        finite_sample_correction: Additional covariance multiplier.

    Examples:
        >>> Clustered("firm").column
        'firm'

    """

    column: str
    finite_sample_correction: float = 1.0


def _covariance_options(specification: CovarianceSpec) -> dict[str, MetadataValue]:
    """Describe invalid numeric options without hiding native unavailable status."""
    options = dict[str, MetadataValue](asdict(specification))
    for name, value in tuple(options.items()):
        if isinstance(value, float) and not math.isfinite(value):
            options[name] = None
            options[f"{name}_reason"] = "nonfinite-option"
    return options


@dataclass(frozen=True)
class _IndependentCovariance:
    """Validated covariance options shared by independent-row likelihood fits."""

    method: Literal["model", "HC0", "HC1"]
    finite_sample_correction: float


def _independent_covariance(
    covariance: ModelBased | HC,
) -> _IndependentCovariance:
    if isinstance(covariance, ModelBased):
        method: Literal["model", "HC0", "HC1"] = "model"
    elif isinstance(covariance, HC) and covariance.kind == "HC0":
        method = "HC0"
    elif isinstance(covariance, HC) and covariance.kind == "HC1":
        method = "HC1"
    else:
        message = "Independent-row covariance supports only model, HC0 and HC1."
        raise TypeError(message)
    return _IndependentCovariance(method, covariance.finite_sample_correction)


@runtime_checkable
class InferenceContext(Protocol):
    """Context naming the model whose inference is reported."""

    @property
    def model_name(self) -> str:
        """Return the model name."""
        ...

    @property
    def row_ids(self) -> tuple[int | str, ...]:
        """Return the actual retained observation identities."""
        ...


@dataclass(frozen=True)
class ContrastResult:
    """A structural contrast with explicitly available normal uncertainty."""

    estimand: LinearPredictorContrast
    estimate: float | None
    standard_error: float | None
    ci_lower: float | None
    ci_upper: float | None
    valid: bool
    unavailable_reason: str | None = None
    confidence: float = field(default=0.95, kw_only=True)
    reference: Literal["normal"] = field(default="normal", kw_only=True)

    def report(self) -> Report:
        """Present the retained contrast without recalculating its uncertainty."""
        return Report(
            self.estimand.describe(),
            metrics=tuple(
                Metric(
                    name,
                    value,
                    available=value is not None,
                    unavailable_reason=None
                    if value is not None
                    else self.unavailable_reason or "not-computed",
                )
                for name, value in (
                    ("estimate", self.estimate),
                    ("standard_error", self.standard_error),
                    ("ci_lower", self.ci_lower),
                    ("ci_upper", self.ci_upper),
                )
            ),
            metadata={
                "valid": self.valid,
                "unavailable_reason": self.unavailable_reason,
                "confidence": self.confidence,
                "reference": self.reference,
            },
            notes=(
                f"{_confidence_label(self.confidence)} intervals; normal reference.",
            ),
        )


@dataclass(frozen=True)
class InferenceResult[U: FittedUncertainty = FittedUncertainty]:
    """Conveniences around actual-fit uncertainty, without automatic calibration."""

    uncertainty: U = field(repr=False)
    context: InferenceContext
    covariance_spec: CovarianceSpec
    design: FittedDesign | None = field(default=None, repr=False, kw_only=True)
    calculation: Mapping[str, MetadataValue] = field(
        default_factory=dict, repr=False, kw_only=True
    )
    fit_statistics: FitStatistics = field(default=FitStatistics(), kw_only=True)

    def __post_init__(self) -> None:
        """Own calculation-order metadata independently of retained fit identity."""
        if not isinstance(self.calculation, Mapping):
            message = "Inference calculation metadata must be a mapping."
            raise TypeError(message)
        object.__setattr__(self, "calculation", freeze_metadata(self.calculation))

    def _repr_html_(self) -> str:
        """Render the report for notebook display."""
        return self.report()._repr_html_()

    def contrast(  # noqa: C901  # distinct schema, covariance and interval failures
        self,
        contrast: DesignContrast | RowDifference,
        *,
        reference: Literal["normal"],
        confidence: float = 0.95,
    ) -> ContrastResult:
        """Infer one structural coefficient contrast without refitting."""
        if reference != "normal":
            message = "Contrasts currently support the normal reference only."
            raise ValueError(message)
        if not 0 < confidence < 1:
            message = "Confidence must lie strictly between zero and one."
            raise ValueError(message)
        fit = self.uncertainty.fit
        if self.design is None:
            message = "This inference does not retain a design for contrasts."
            raise ValueError(message)
        if isinstance(contrast, RowDifference):
            contrast = contrast.compile(self.design)
        if contrast.design is not self.design:
            message = "Contrast belongs to a different fitted design."
            raise ValueError(message)
        n = len(contrast.parameter_names)
        if (
            fit.parameter_names is None
            or tuple(fit.parameter_names[:n]) != contrast.parameter_names
        ):
            message = "Contrast coefficient schema does not match the fit."
            raise ValueError(message)
        if n > len(fit.parameters) or contrast.coefficients.shape != (n,):
            message = "Contrast coefficients must match the retained design."
            raise ValueError(message)
        vector = jnp.pad(
            jnp.asarray(contrast.coefficients), (0, len(fit.parameters) - n)
        )
        point = vector @ fit.parameters
        estimate = float(point) if bool(jnp.isfinite(point)) else None
        if not bool(self.uncertainty.valid):
            return ContrastResult(
                contrast.estimand,
                estimate,
                None,
                None,
                None,
                valid=False,
                unavailable_reason="INVALID_COVARIANCE",
                confidence=confidence,
            )
        variance = vector @ self.uncertainty.covariance.values @ vector
        if estimate is None or not bool(jnp.isfinite(variance) & (variance > 0)):
            return ContrastResult(
                contrast.estimand,
                estimate,
                None,
                None,
                None,
                valid=False,
                unavailable_reason="INVALID_CONTRAST_VARIANCE",
                confidence=confidence,
            )
        se = float(jnp.sqrt(variance))
        critical = float(
            normal_critical_value(jnp.asarray(confidence, dtype=point.dtype))
        )
        lower, upper = estimate - critical * se, estimate + critical * se
        if not bool(jnp.all(jnp.isfinite(jnp.asarray((lower, upper))))):
            return ContrastResult(
                contrast.estimand,
                estimate,
                se,
                None,
                None,
                valid=False,
                unavailable_reason="INVALID_REFERENCE_INTERVAL",
                confidence=confidence,
            )
        return ContrastResult(
            contrast.estimand,
            estimate,
            se,
            lower,
            upper,
            valid=True,
            confidence=confidence,
        )

    def intervals(
        self,
        *,
        reference: Literal["normal", "t"],
        degrees_of_freedom: ArrayLike | None = None,
        confidence: float = 0.95,
    ) -> CoefficientIntervals:
        """Compute intervals under the explicitly selected reference law.

        Args:
            reference: Normal or Student reference distribution.
            degrees_of_freedom: Degrees of freedom for a Student reference.
            confidence: Requested marginal confidence level.

        Returns:
            Coefficient intervals with numerical availability information.

        """
        return coefficient_intervals(
            self.uncertainty,
            confidence=confidence,
            reference=reference,
            degrees_of_freedom=degrees_of_freedom,
        )

    def _compile_hypothesis(
        self,
        hypothesis: Hypothesis[LinearRestriction],
    ) -> tuple[Array, Array]:
        """Bind a supported named question to the actual retained fit schema."""
        if not isinstance(hypothesis, Hypothesis) or not isinstance(
            hypothesis.null, LinearRestriction
        ):
            message = "Coefficient tests require Hypothesis[LinearRestriction]."
            raise TypeError(message)
        if hypothesis.alternative is not None:
            message = "Coefficient tests support the complement alternative only."
            raise ValueError(message)
        fit = self.uncertainty.fit
        if fit.parameter_names is None:
            message = "Named restrictions require actual fitted parameter names."
            raise ValueError(message)
        if fit.parameters.ndim != 1:
            message = "Host tests require one fit; use the native kernel for batches."
            raise ValueError(message)
        restriction = hypothesis.null
        if restriction.design is not None:
            if restriction.design is not self.design:
                message = "Restriction belongs to a different fitted design."
                raise ValueError(message)
            names = restriction.design.column_names
            if tuple(fit.parameter_names[: len(names)]) != names or any(
                not set(row) <= set(names) for row in restriction.rows
            ):
                message = (
                    "Restriction coefficient schema does not match the fitted design."
                )
                raise ValueError(message)
        return restriction.compile(fit.parameter_names, fit.parameters.dtype)

    def test(
        self,
        hypothesis: Hypothesis[LinearRestriction],
        *,
        reference: Literal["chi2", "f"],
        degrees_of_freedom: ArrayLike | None = None,
    ) -> TestOutcome[LinearRestriction]:
        """Test a named hypothesis against this fit without refitting.

        Positional or batched numerical callers use the native restriction
        kernel. Reference selection remains the caller's explicit assumption.

        Args:
            hypothesis: Named linear restriction to test.
            reference: Chi-square or F reference distribution.
            degrees_of_freedom: Denominator degrees of freedom for an F reference.

        Returns:
            A test outcome carrying the claim, calibration and availability.

        """
        matrix, target = self._compile_hypothesis(hypothesis)
        fit = self.uncertainty.fit
        native = linear_restriction_test(
            self.uncertainty,
            matrix,
            target,
            parameter_names=fit.parameter_names,
            reference=reference,
            degrees_of_freedom=degrees_of_freedom,
        )
        available = bool(native.valid)
        calibration: ChiSquare | FisherF | None = None
        if available:
            calibration = (
                FisherF(float(native.numerator_df), float(native.denominator_df))
                if reference == "f"
                else ChiSquare(float(native.numerator_df))
            )
        details: dict[str, ReportNumber] = {
            "restriction_rank": int(native.restriction_rank),
            "covariance_rank": int(native.covariance_rank),
            "numerator_df": int(native.numerator_df),
        }
        if bool(jnp.isfinite(native.wald_statistic)):
            details["wald_statistic"] = float(native.wald_statistic)
        if bool(jnp.isfinite(native.denominator_df)):
            details["denominator_df"] = float(native.denominator_df)
        return TestOutcome(
            "wald",
            hypothesis,
            float(native.test.statistic) if available else None,
            float(native.test.pvalue) if available else None,
            calibration,
            available,
            unavailable_reason=None
            if available
            else InferenceStatus(int(native.status)).name,
            details=details,
            construction={
                "rows": _row_identity(self.context.row_ids),
                **self.calculation,
                "parameter_names": fit.parameter_names,
                "covariance": type(self.covariance_spec).__name__,
                "covariance_options": _covariance_options(self.covariance_spec),
                "requested_reference": reference,
                "status": InferenceStatus(int(native.status)).name,
                "finite_reference_outside_validated_range": bool(
                    native.finite_reference_outside_validated_range
                ),
                "integration_accuracy_unqualified": hasattr(fit, "integration_valid"),
            },
            evidence=native,
            assumptions=(
                (
                    "The selected reference is the caller's statistical assumption; "
                    "a covariance method or numerical rank does not establish it."
                ),
            ),
        )

    def _null_targets(
        self, null: float | Mapping[str, float] | None
    ) -> dict[str, float]:
        """Validate a stated null and bind it to retained coefficient names."""
        if null is None:
            return {}
        fit = self.uncertainty.fit
        names = fit.parameter_names or tuple(
            f"parameter_{i}" for i in range(len(fit.parameters))
        )
        if isinstance(null, Mapping):
            if not null:
                message = "A null mapping must name at least one coefficient."
                raise ValueError(message)
            unknown = sorted(str(key) for key in null if key not in names)
            if unknown:
                message = f"Null names unknown coefficients: {', '.join(unknown)}."
                raise ValueError(message)
            return {key: _null_value(value) for key, value in null.items()}
        value = _null_value(null)
        return dict.fromkeys(names, value)

    def _coefficient_test(
        self,
        index: int,
        null: float,
        *,
        se_available: bool,
        reference: Literal["normal", "t"] | None,
        degrees_of_freedom: ArrayLike | None,
        covariance_reason: str,
    ) -> tuple[float | None, float | None, bool, str | None, float]:
        """Run one single-restriction Wald test for coefficient ``index``.

        The null is canonicalised once in the fit's dtype and that value, the
        effective null, is what the native test uses. Returns the signed root of
        the native Wald quadratic, the native two-sided p-value of the same call,
        the native outside-range flag, the unavailable reason (``None`` when the
        test is available) and the effective null.
        """
        fit = self.uncertainty.fit
        dtype = fit.parameters.dtype
        selector = jnp.zeros((1, fit.parameters.shape[-1]), dtype=dtype)
        selector = selector.at[0, index].set(1.0)
        target = jnp.asarray([null], dtype=dtype)
        effective = float(target[0])
        native = linear_restriction_test(
            self.uncertainty,
            selector,
            target,
            parameter_names=fit.parameter_names,
            reference="f" if reference == "t" else "chi2",
            degrees_of_freedom=degrees_of_freedom,
        )
        outside = bool(native.finite_reference_outside_validated_range)
        estimate = fit.parameters[index]
        if not (se_available and bool(native.valid) and bool(jnp.isfinite(estimate))):
            reason = (
                InferenceStatus(int(native.status)).name
                if se_available
                else covariance_reason
            )
            return None, None, outside, reason, effective
        wald = float(native.wald_statistic)
        statistic = (
            0.0
            if wald <= 0.0
            else math.copysign(math.sqrt(wald), float(estimate) - effective)
        )
        return statistic, float(native.test.pvalue), outside, None, effective

    def report(  # noqa: PLR0914 -- one reported row per retained parameter
        self,
        *,
        reference: Literal["normal", "t"] | None = None,
        degrees_of_freedom: ArrayLike | None = None,
        confidence: float = 0.95,
        null: float | Mapping[str, float] | None = None,
    ) -> Report:
        """Report SEs and explicitly calibrated intervals.

        No p-value is reported unless the caller states a ``null`` value (one
        float for every coefficient, or a mapping naming the tested ones) and a
        ``reference``. Each tested row then holds the signed root
        ``sign(estimate - null) * sqrt(wald)`` and the two-sided p-value of the
        single-restriction Wald test that :meth:`test` performs. The null is
        converted once to the fit's dtype; when that changes it, metadata
        ``effective_null`` and a warning record the value actually tested.

        Args:
            reference: Optional normal or Student reference for intervals and tests.
            degrees_of_freedom: Degrees of freedom for a Student reference.
            confidence: Requested marginal confidence level.
            null: Optional scalar or named coefficient targets to test.

        Returns:
            A report containing the retained fit and requested inference.

        Raises:
            ValueError: If a null or degrees of freedom lacks a reference.

        """
        if reference is None and degrees_of_freedom is not None:
            message = "Degrees of freedom require an explicitly selected reference."
            raise ValueError(message)
        if null is not None and reference is None:
            message = "Coefficient tests require an explicitly selected reference."
            raise ValueError(message)
        tested = self._null_targets(null)
        intervals = (
            None
            if reference is None
            else self.intervals(
                reference=reference,
                degrees_of_freedom=degrees_of_freedom,
                confidence=confidence,
            )
        )
        fit = self.uncertainty.fit
        names = fit.parameter_names or tuple(
            f"parameter_{i}" for i in range(len(fit.parameters))
        )
        variances = jnp.diag(self.uncertainty.covariance.values)
        covariance_valid = bool(self.uncertainty.valid)
        covariance_reason = InferenceStatus(int(self.uncertainty.status)).name
        interval_valid = intervals is not None and bool(intervals.valid)
        interval_reason = (
            None if intervals is None else InferenceStatus(int(intervals.status)).name
        )
        outside_reference_range = intervals is not None and bool(
            intervals.finite_reference_outside_validated_range
        )
        rows = []
        warnings: list[str] = []
        effective_nulls: dict[str, float] = {}
        for index, name in enumerate(names):
            variance = variances[index]
            # The covariance owns the SE; a calibration failure withholds only CIs.
            se_available = covariance_valid and bool(
                jnp.isfinite(variance) & (variance > 0.0)
            )
            show_interval = se_available and interval_valid
            interval_failed = (
                intervals is not None and se_available and not show_interval
            )
            statistic = None
            pvalue = None
            if name in tested:
                statistic, pvalue, outside, problem, effective = self._coefficient_test(
                    index,
                    tested[name],
                    se_available=se_available,
                    reference=reference,
                    degrees_of_freedom=degrees_of_freedom,
                    covariance_reason=covariance_reason,
                )
                outside_reference_range = outside_reference_range or outside
                if effective != tested[name]:
                    effective_nulls[name] = effective
                    warnings.append(
                        f"Null {tested[name]!r} for {name} is {effective!r} in the "
                        f"fit's {fit.parameters.dtype.name} precision; the test "
                        f"uses {effective!r}."
                    )
                if problem is not None:
                    warnings.append(
                        f"Coefficient test unavailable for {name}: {problem}."
                    )
            rows.append(
                ParameterRow(
                    name,
                    float(fit.parameters[index])
                    if bool(jnp.isfinite(fit.parameters[index]))
                    else None,
                    std_error=float(jnp.sqrt(variance)) if se_available else None,
                    statistic=statistic,
                    pvalue=pvalue,
                    ci_lower=float(intervals.estimate.ci[0][index])
                    if show_interval and intervals is not None
                    else None,
                    ci_upper=float(intervals.estimate.ci[1][index])
                    if show_interval and intervals is not None
                    else None,
                    valid=se_available,
                    status="available" if se_available else "unavailable",
                    unavailable_reason=None
                    if se_available
                    else (
                        covariance_reason
                        if not covariance_valid
                        else InferenceStatus.INVALID_COVARIANCE.name
                    ),
                    interval_status=None
                    if intervals is None or not se_available
                    else "available"
                    if show_interval
                    else "unavailable",
                    interval_unavailable_reason=interval_reason
                    if interval_failed
                    else None,
                )
            )
        if hasattr(fit, "integration_valid"):
            warnings.append("Integration accuracy remains unqualified.")
        if outside_reference_range:
            warnings.append(
                "Finite-reference numerical accuracy is outside the independently "
                "validated precision and degrees-of-freedom range."
            )
        return Report(
            title=f"{self.context.model_name} inference",
            parameters=tuple(rows),
            metrics=(
                Metric(
                    "n_observations", int(self.uncertainty.covariance.n_observations)
                ),
                *self.fit_statistics.metrics,
            ),
            metadata={
                "rows": _row_identity(self.context.row_ids),
                **self.calculation,
                "covariance": type(self.covariance_spec).__name__,
                "covariance_options": _covariance_options(self.covariance_spec),
                "n_observations": int(self.uncertainty.covariance.n_observations),
                "reference": reference,
                "confidence": confidence if intervals is not None else None,
                "degrees_of_freedom": None
                if degrees_of_freedom is None
                else float(jnp.asarray(degrees_of_freedom)),
                "inference_columns": {self.covariance_spec.column: "grouping"}
                if isinstance(self.covariance_spec, Clustered)
                else {},
                "inference_status": covariance_reason,
                "interval_status": interval_reason,
                "finite_reference_outside_validated_range": outside_reference_range,
                **({} if null is None else {"null": null}),
                **(
                    {}
                    if not effective_nulls
                    else {
                        "effective_null": next(iter(effective_nulls.values()))
                        if not isinstance(null, Mapping)
                        else effective_nulls
                    }
                ),
            },
            notes=(
                *(
                    (
                        (
                            f"{_confidence_label(confidence)} intervals; "
                            f"{reference} reference; covariance "
                            f"{getattr(self.covariance_spec, 'kind', type(self.covariance_spec).__name__)}"  # noqa: E501
                        ),
                    )
                    if intervals is not None
                    else ()
                ),
                *(() if null is None else (_null_note(null, reference),)),
                *self.fit_statistics.notes,
            ),
            warnings=tuple(warnings),
        )
