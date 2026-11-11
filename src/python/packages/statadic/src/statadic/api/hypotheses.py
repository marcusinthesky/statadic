# ruff: noqa: TC001  # Public annotations resolve under the runtime import hook.
"""Immutable host hypotheses and reportable outcomes above numerical procedures.

Propositions describe a scientific null; they neither choose a procedure nor
establish its assumptions. Numerical kernels remain at their existing owners.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import ClassVar, Literal, Protocol, final, runtime_checkable

import jax.numpy as jnp

from statadic.api.design import DesignContrast, FittedDesign
from statadic.api.reporting import (
    FrozenMetadataValue,
    MetadataValue,
    Metric,
    Report,
    ReportNumber,
    TestRow,
    freeze_metadata,
)
from statadic.core.typing import Array

__all__ = [
    "ChiSquare",
    "ConstantVariance",
    "FisherF",
    "GaussianMoments",
    "Hypothesis",
    "LinearRestriction",
    "NoAddedFittedPowerEffect",
    "NoSerialCorrelation",
    "Normal",
    "OveridentifyingRestrictions",
    "Proposition",
    "Reference",
    "ResampledReference",
    "TestOutcome",
]


def _number(value: ReportNumber, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        message = f"{name} must be a finite real number."
        raise ValueError(message)
    return float(value)


def _component(value: ReportNumber, name: str) -> ReportNumber:
    """Keep an integral component exact and every other component a float."""
    number = _number(value, name)
    return int(value) if isinstance(value, Integral) else number


def _names(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    owned = tuple(values)
    if not owned or any(not isinstance(value, str) or not value for value in owned):
        message = f"{name} requires nonempty names."
        raise ValueError(message)
    return owned


@runtime_checkable
class Proposition(Protocol):
    """An immutable scientific proposition with explicit finite metadata.

    Implementations own their inputs. The wrapper snapshots descriptions and
    metadata; it never serializes arbitrary attributes or a live fitted design.
    """

    def describe(self) -> str:
        """State the proposition without asserting that it holds."""
        ...

    def metadata(self) -> Mapping[str, MetadataValue]:
        """Return explicit JSON-compatible construction fields."""
        ...


@final
@dataclass(frozen=True)
class Hypothesis[NullT: Proposition = Proposition]:
    """A null proposition and an optional explicit alternative.

    ``None`` denotes the complement. First-slice procedures require that
    default; accepting another alternative needs its own supported procedure.
    """

    null: NullT
    alternative: Proposition | None = None
    _description: str = field(init=False, repr=False, compare=False)
    _metadata: Mapping[str, FrozenMetadataValue] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Snapshot the stated question independently of mutable metadata inputs."""
        if not isinstance(self.null, Proposition) or (
            self.alternative is not None
            and not isinstance(self.alternative, Proposition)
        ):
            message = "Hypotheses require propositions with describe() and metadata()."
            raise TypeError(message)
        description = self.null.describe()
        if not isinstance(description, str) or not description:
            message = "A proposition requires a nonempty description."
            raise ValueError(message)
        if self.alternative is not None:
            alternative = self.alternative.describe()
            if not isinstance(alternative, str) or not alternative:
                message = "An alternative requires a nonempty description."
                raise ValueError(message)
            description += f"; alternative: {alternative}"
        null_metadata = self.null.metadata()
        alternative_metadata = (
            None if self.alternative is None else self.alternative.metadata()
        )
        if not isinstance(null_metadata, Mapping) or (
            alternative_metadata is not None
            and not isinstance(alternative_metadata, Mapping)
        ):
            message = "Proposition metadata must be a mapping."
            raise TypeError(message)
        object.__setattr__(self, "_description", description)
        object.__setattr__(
            self,
            "_metadata",
            freeze_metadata(
                {
                    "null": null_metadata,
                    "alternative": alternative_metadata,
                }
            ),
        )

    def describe(self) -> str:
        """Return the frozen statement of the question."""
        return self._description

    def metadata(self) -> Mapping[str, FrozenMetadataValue]:
        """Return the immutable metadata snapshot, without live design identity."""
        return self._metadata


@final
@dataclass(frozen=True)
class LinearRestriction:
    """Named rows of ``R theta = r``, optionally bound to one learned design."""

    rows: tuple[Mapping[str, ReportNumber], ...]
    values: tuple[ReportNumber, ...] | ReportNumber = 0.0
    design: FittedDesign | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Own finite coefficients and one target per restriction row."""
        if not self.rows:
            message = "A linear restriction requires at least one row."
            raise ValueError(message)
        rows = []
        for row in self.rows:
            _names(tuple(row), "Restriction rows")
            rows.append(
                MappingProxyType(
                    {
                        name: _number(value, "Restriction coefficient")
                        for name, value in row.items()
                    }
                )
            )
        values = (
            tuple(_number(value, "Restriction target") for value in self.values)
            if isinstance(self.values, tuple)
            else (_number(self.values, "Restriction target"),) * len(rows)
        )
        if len(values) != len(rows):
            message = "Restriction targets must match the number of rows."
            raise ValueError(message)
        object.__setattr__(self, "rows", tuple(rows))
        object.__setattr__(self, "values", values)

    @classmethod
    def zero(cls, *names: str) -> LinearRestriction:
        """State one zero-coefficient restriction for each supplied name."""
        names = _names(names, "Zero restrictions")
        if len(set(names)) != len(names):
            message = "Zero restrictions require unique coefficient names."
            raise ValueError(message)
        return cls(tuple({name: 1.0} for name in names))

    @classmethod
    def from_contrasts(
        cls,
        *contrasts: DesignContrast,
        values: tuple[ReportNumber, ...] | ReportNumber = 0.0,
    ) -> LinearRestriction:
        """Bind structural contrasts from exactly one fitted coefficient basis."""
        if not contrasts:
            message = "Supply at least one design contrast."
            raise ValueError(message)
        first = contrasts[0]
        rows = []
        for contrast in contrasts:
            if (
                contrast.design is not first.design
                or contrast.parameter_names != first.parameter_names
            ):
                message = (
                    "Restriction contrasts must share one fitted design and schema."
                )
                raise ValueError(message)
            if contrast.coefficients.shape != (len(contrast.parameter_names),):
                message = "Contrast coefficients must match their named schema."
                raise ValueError(message)
            rows.append(
                dict(
                    zip(
                        contrast.parameter_names,
                        map(float, contrast.coefficients),
                        strict=True,
                    )
                )
            )
        return cls(tuple(rows), values, first.design)

    def compile(
        self, parameter_names: tuple[str, ...], dtype: jnp.dtype
    ) -> tuple[Array, Array]:
        """Resolve names in the actual fit order, including ancillary zero columns."""
        names = _names(parameter_names, "Fitted parameters")
        if len(set(names)) != len(names):
            message = "Fitted parameter names must be unique."
            raise ValueError(message)
        unknown = set().union(*(row.keys() for row in self.rows)) - set(names)
        if unknown:
            message = f"Unknown restricted parameter names: {sorted(unknown)}."
            raise ValueError(message)
        matrix = jnp.asarray(
            [[row.get(name, 0.0) for name in names] for row in self.rows], dtype=dtype
        )
        return matrix, jnp.asarray(self.values, dtype=dtype)

    def describe(self) -> str:
        """State each named equality without selecting a reference law."""
        targets = (
            self.values
            if isinstance(self.values, tuple)
            else (self.values,) * len(self.rows)
        )
        return "; ".join(
            " + ".join(f"{value:g}*{name}" for name, value in row.items())
            + f" = {target:g}"
            for row, target in zip(self.rows, targets, strict=True)
        )

    def metadata(self) -> Mapping[str, MetadataValue]:
        """Describe coefficient rows without serializing the fitted design."""
        return {
            "kind": "linear-restriction",
            "rows": self.rows,
            "values": self.values,
            "design_bound": self.design is not None,
        }


@dataclass(frozen=True)
class ConstantVariance:
    """No variance effects represented by the specified variance-design columns."""

    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        """Own the names of the tested variance design."""
        object.__setattr__(self, "columns", _names(self.columns, "Variance design"))

    def describe(self) -> str:
        """State the bounded variance-design null."""
        return f"No error-variance effects in the specified design {self.columns}."

    def metadata(self) -> Mapping[str, MetadataValue]:
        """Return the tested variance-column names."""
        return {"kind": "constant-variance", "columns": self.columns}


@dataclass(frozen=True)
class GaussianMoments:
    """Gaussian skewness and kurtosis, without an omnibus normality claim."""

    @staticmethod
    def describe() -> str:
        """State the two tested Gaussian moments."""
        return "Errors have Gaussian skewness 0 and kurtosis 3."

    @staticmethod
    def metadata() -> Mapping[str, MetadataValue]:
        """Name the two tested moments."""
        return {"kind": "gaussian-moments", "skewness": 0.0, "kurtosis": 3.0}


@dataclass(frozen=True)
class NoSerialCorrelation:
    """No serial correlation through the requested positional lag count."""

    lags: int
    order_by: str | None

    def __post_init__(self) -> None:
        """Validate the explicit lag and ordering choices."""
        if (
            isinstance(self.lags, bool)
            or not isinstance(self.lags, int)
            or self.lags < 1
        ):
            message = "Serial-correlation lags must be a positive integer."
            raise ValueError(message)
        if self.order_by is not None and (
            not isinstance(self.order_by, str) or not self.order_by
        ):
            message = "order_by must name a column or be explicit None."
            raise ValueError(message)

    def describe(self) -> str:
        """Keep retained-row ordering distinct from an asserted time scale."""
        order = (
            "retained row order"
            if self.order_by is None
            else f"ascending {self.order_by!r} order"
        )
        return (
            f"No residual serial correlation at positional lags 1 to {self.lags} "
            f"in {order}."
        )

    def metadata(self) -> Mapping[str, MetadataValue]:
        """Retain ordering and lag interpretation."""
        return {
            "kind": "no-serial-correlation",
            "lags": self.lags,
            "order_by": self.order_by,
            "lag_unit": "adjacent ordered observations",
        }


@dataclass(frozen=True)
class NoAddedFittedPowerEffect:
    """No added mean effect from the specified fitted-value power span."""

    powers: tuple[int, ...]

    def __post_init__(self) -> None:
        """Own the power sequence, preserving redundant powers for rank accounting."""
        powers = tuple(self.powers)
        if not powers or any(
            isinstance(power, bool) or not isinstance(power, int) or power < 2  # noqa: PLR2004 -- nonlinear fitted powers
            for power in powers
        ):
            message = "Fitted powers must be integers of at least two."
            raise ValueError(message)
        object.__setattr__(self, "powers", powers)

    def describe(self) -> str:
        """State the tested added span rather than all mean specification."""
        return f"No added mean effect from fitted-value powers {self.powers}."

    def metadata(self) -> Mapping[str, MetadataValue]:
        """Retain the actual requested powers."""
        return {"kind": "no-added-fitted-power-effect", "powers": self.powers}


@dataclass(frozen=True)
class OveridentifyingRestrictions:
    """Correct specification of the supplied moments at one common parameter."""

    moment_count: int
    parameter_count: int

    def __post_init__(self) -> None:
        """Own dimensions even when they cannot support a calibrated J test."""
        for value, minimum in ((self.moment_count, 1), (self.parameter_count, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                message = (
                    "Moment and parameter counts must be valid nonnegative dimensions."
                )
                raise ValueError(message)

    def describe(self) -> str:
        """State moment compatibility without asserting identification."""
        return (
            f"All {self.moment_count} supplied moments are correctly specified "
            f"at one common {self.parameter_count}-parameter value."
        )

    def metadata(self) -> Mapping[str, MetadataValue]:
        """Retain dimensions rather than confusing moment count with test degrees."""
        return {
            "kind": "overidentifying-restrictions",
            "moment_count": self.moment_count,
            "parameter_count": self.parameter_count,
        }


@dataclass(frozen=True)
class ChiSquare:
    """The chi-square law used to calibrate one returned p-value."""

    df: ReportNumber

    def __post_init__(self) -> None:
        """Require finite positive degrees of freedom for an actual reference."""
        degrees = _number(self.df, "Chi-square degrees of freedom")
        if degrees <= 0:
            message = "Chi-square degrees of freedom must be positive."
            raise ValueError(message)
        object.__setattr__(self, "df", degrees)

    def __str__(self) -> str:
        """Return the familiar report notation."""
        return f"chi2({self.df:g})"

    def metadata(self) -> Mapping[str, MetadataValue]:
        """Return the actual reference parameters."""
        return {"kind": "chi-square", "df": self.df}


@dataclass(frozen=True)
class FisherF:
    """The F law used to calibrate one returned p-value."""

    df: ReportNumber
    df_denominator: ReportNumber

    def __post_init__(self) -> None:
        """Require both finite positive reference degrees."""
        for name in ("df", "df_denominator"):
            value = _number(getattr(self, name), "F degrees of freedom")
            if value <= 0:
                message = "F degrees of freedom must be positive."
                raise ValueError(message)
            object.__setattr__(self, name, value)

    def __str__(self) -> str:
        """Return the familiar report notation."""
        return f"F({self.df:g}, {self.df_denominator:g})"

    def metadata(self) -> Mapping[str, MetadataValue]:
        """Return the actual reference parameters."""
        return {"kind": "f", "df": self.df, "df_denominator": self.df_denominator}


@dataclass(frozen=True)
class Normal:
    """A normal reference with its actual statistic location and scale."""

    location: ReportNumber = 0.0
    scale: ReportNumber = 1.0

    def __post_init__(self) -> None:
        """Require finite location and strictly positive scale."""
        location = _number(self.location, "Normal location")
        scale = _number(self.scale, "Normal scale")
        if scale <= 0:
            message = "Normal scale must be positive."
            raise ValueError(message)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "scale", scale)

    def __str__(self) -> str:
        """Return the calibrated statistic's reference notation."""
        return f"Normal({self.location:g}, {self.scale:g})"

    def metadata(self) -> Mapping[str, MetadataValue]:
        """Retain the statistic's reference parameters."""
        return {"kind": "normal", "location": self.location, "scale": self.scale}


@dataclass(frozen=True)
class ResampledReference:
    """A completed retained schedule, without upgrading its calibration claim.

    ``exact`` describes enumeration only when ``approximate`` is false. A
    supplied schedule retains its actual finite-sample tail correction.
    Numerical draws and their identities stay in the outcome's typed evidence.
    """

    method: Literal["exact", "random", "supplied"]
    draws: int
    correction: Literal["none", "add-one"] = field(kw_only=True)
    tail: Literal["greater", "less", "two-sided"] = field(
        default="greater", kw_only=True
    )
    approximate: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        """Reject empty schedules and contradictory enumeration corrections."""
        if (
            self.method not in ("exact", "random", "supplied")
            or self.correction not in ("none", "add-one")
            or self.tail not in ("greater", "less", "two-sided")
            or type(self.approximate) is not bool
        ):
            message = "Invalid resampled reference specification."
            raise ValueError(message)
        if type(self.draws) is not int or self.draws < 1:
            message = "Invalid resampled reference draw count."
            raise ValueError(message)
        if self.method == "exact" and self.correction != "none":
            message = "An enumerated reference uses its uncorrected exact count."
            raise ValueError(message)

    def __str__(self) -> str:
        """Distinguish approximate schedules from exact randomization references."""
        qualifier = "approximate " if self.approximate else ""
        return f"{qualifier}{self.method} resampling ({self.draws} draws)"

    def metadata(self) -> Mapping[str, MetadataValue]:
        """Retain the actual schedule and tail conventions."""
        return {
            "kind": "resampled",
            "method": self.method,
            "draws": self.draws,
            "correction": self.correction,
            "tail": self.tail,
            "approximate": self.approximate,
        }


type Reference = ChiSquare | FisherF | Normal | ResampledReference


@dataclass(frozen=True)
class TestOutcome[NullT: Proposition = Proposition, EvidenceT = object]:
    """One host statistic, its stated hypothesis and optional calibrated tail."""

    __test__: ClassVar[bool] = False

    name: str
    hypothesis: Hypothesis[NullT]
    statistic: ReportNumber | None
    pvalue: ReportNumber | None
    reference: Reference | None
    available: bool
    unavailable_reason: str | None = None
    details: Mapping[str, ReportNumber] = field(default_factory=dict, repr=False)
    construction: Mapping[str, MetadataValue] = field(default_factory=dict, repr=False)
    assumptions: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    evidence: EvidenceT | None = field(default=None, repr=False, kw_only=True)

    def __post_init__(self) -> None:
        """Freeze provenance and reject contradictory availability metadata."""
        if (
            not isinstance(self.name, str)
            or not self.name
            or type(self.available) is not bool
        ):
            message = "Test outcomes require a name and boolean availability."
            raise ValueError(message)
        if not isinstance(self.hypothesis, Hypothesis):
            message = "Test outcomes require a Hypothesis wrapper."
            raise TypeError(message)
        if self.reference is not None and not isinstance(
            self.reference, (ChiSquare, FisherF, Normal, ResampledReference)
        ):
            message = "Test references must be supported immutable reference values."
            raise TypeError(message)
        if self.unavailable_reason is not None and (
            not isinstance(self.unavailable_reason, str) or not self.unavailable_reason
        ):
            message = "An unavailable reason must be nonempty text."
            raise ValueError(message)
        self._validate_numbers()
        details = {
            name: _component(value, "Test component")
            for name, value in self.details.items()
        }
        if any(not isinstance(name, str) or not name for name in details):
            message = "Test components require nonempty names."
            raise ValueError(message)
        if {"statistic", "pvalue"} & details.keys():
            message = "Test components cannot replace statistic or pvalue metrics."
            raise ValueError(message)
        object.__setattr__(self, "details", MappingProxyType(details))
        object.__setattr__(self, "construction", freeze_metadata(self.construction))
        for name in ("assumptions", "sources"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) or not value for value in values):
                message = "Test assumptions and sources require nonempty strings."
                raise ValueError(message)
            object.__setattr__(self, name, values)

    def _validate_numbers(self) -> None:
        """Keep statistic availability distinct from an optional calibrated tail."""
        if self.available:
            if self.statistic is None:
                message = "Available test outcomes require a finite statistic."
                raise ValueError(message)
            object.__setattr__(
                self, "statistic", _number(self.statistic, "Test statistic")
            )
            if self.unavailable_reason is not None:
                message = "Available test outcomes cannot carry an unavailable reason."
                raise ValueError(message)
        elif (
            not self.unavailable_reason
            or self.statistic is not None
            or self.pvalue is not None
        ):
            message = (
                "Unavailable test outcomes require a reason and absent "
                "statistic/p-value."
            )
            raise ValueError(message)
        if (self.pvalue is None) != (self.reference is None):
            message = "A reference and calibrated p-value must be supplied together."
            raise ValueError(message)
        if self.pvalue is not None:
            probability = _number(self.pvalue, "P-value")
            if not 0 <= probability <= 1:
                message = "P-values must lie in [0, 1]."
                raise ValueError(message)
            object.__setattr__(self, "pvalue", probability)

    def test_row(self) -> TestRow:
        """Return the typed row of tests for this statistic and its reference."""
        reference = self.reference
        law: Literal["chi-square", "f", "normal", "resampled"] | None
        parameters: tuple[ReportNumber, ...]
        match reference:
            case ChiSquare():
                law, parameters = "chi-square", (reference.df,)
            case FisherF():
                law = "f"
                parameters = (reference.df, reference.df_denominator)
            case Normal():
                law, parameters = "normal", (reference.location, reference.scale)
            case ResampledReference():
                law, parameters = "resampled", (reference.draws,)
            case _:
                law, parameters = None, ()
        return TestRow(
            self.name,
            self.statistic,
            self.pvalue,
            law,
            parameters,
            "" if reference is None else str(reference),
            self.hypothesis.describe(),
            available=self.available,
            unavailable_reason=self.unavailable_reason,
        )

    def components(self) -> tuple[Metric, ...]:
        """Keep finite diagnostic components visible as metrics."""
        return tuple(
            Metric(f"{self.name}.{name}", value) for name, value in self.details.items()
        )

    def report(self) -> Report:
        """Render retained evidence without repeating any statistical calculation."""
        warnings = (
            [] if self.available else [f"Unavailable test: {self.unavailable_reason}."]
        )
        if self.construction.get("finite_reference_outside_validated_range") is True:
            warnings.append(
                "Finite-reference numerical accuracy is outside the independently "
                "validated precision and degrees-of-freedom range."
            )
        if self.construction.get("integration_accuracy_unqualified") is True:
            warnings.append("Integration accuracy remains unqualified.")
        return Report(
            self.name,
            tests=(self.test_row(),),
            metrics=self.components(),
            metadata={
                "hypothesis": self.hypothesis.metadata(),
                "reference": None
                if self.reference is None
                else self.reference.metadata(),
                "construction": self.construction,
                "assumptions": self.assumptions,
                "sources": self.sources,
            },
            warnings=tuple(warnings),
            notes=(
                "Failing to reject a null hypothesis is not evidence that it holds.",
            ),
        )

    def _repr_html_(self) -> str:
        """Use the bounded notebook report renderer."""
        return self.report()._repr_html_()
