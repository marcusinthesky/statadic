"""Immutable numerical report snapshots and renderers with no statistical work."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Literal, cast, overload

import jax.numpy as jnp

__all__ = [
    "FIT_STATISTIC_NAMES",
    "FitStatisticName",
    "FrozenMetadataValue",
    "JsonValue",
    "MetadataValue",
    "Metric",
    "ParameterRow",
    "PlotSpec",
    "ReferenceLine",
    "Report",
    "ReportNumber",
    "TestRow",
    "freeze_metadata",
    "metadata_to_json",
]

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.figure import Figure

    from statadic.core.typing import Array

type ReportNumber = int | float

type JsonValue = (
    bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
)
type FrozenMetadataValue = (
    bool
    | int
    | float
    | str
    | tuple[FrozenMetadataValue, ...]
    | Mapping[str, FrozenMetadataValue]
    | None
)
type MetadataValue = (
    JsonValue
    | FrozenMetadataValue
    | list[MetadataValue]
    | tuple[MetadataValue, ...]
    | Mapping[str, MetadataValue]
    | None
)
_REFERENCE_ENDPOINT_COUNT = 2
_NOTEBOOK_BUDGET = 12_000
_PREVIEW_ITEMS = 12
_PREVIEW_STRING = 160
_PREVIEW_DEPTH = 3


def _metadata_key(value: object) -> str:
    if not isinstance(value, str):
        message = "Report metadata keys must be strings."
        raise TypeError(message)
    return value


def _preview_metadata(
    value: FrozenMetadataValue, depth: int = 0
) -> FrozenMetadataValue:
    """Bound notebook metadata without changing retained export data."""
    if isinstance(value, str):
        return value[:_PREVIEW_STRING] + ("…" if len(value) > _PREVIEW_STRING else "")
    if isinstance(value, (Mapping, tuple)):
        if depth >= _PREVIEW_DEPTH:
            return f"… ({len(value)} items)"
        if isinstance(value, Mapping):
            result: dict[str, FrozenMetadataValue] = {}
            for index, (key, child) in enumerate(value.items()):
                if index >= _PREVIEW_ITEMS:
                    result["…"] = f"{len(value) - index} more items"
                    break
                result[key[:_PREVIEW_STRING]] = _preview_metadata(child, depth + 1)
            return result
        children = [
            _preview_metadata(child, depth + 1) for child in value[:_PREVIEW_ITEMS]
        ]
        if len(value) > _PREVIEW_ITEMS:
            children.append(f"… ({len(value) - _PREVIEW_ITEMS} more items)")
        return tuple(children)
    return value


@overload
def freeze_metadata(
    value: Mapping[str, object],
) -> Mapping[str, FrozenMetadataValue]: ...


@overload
def freeze_metadata(
    value: list[object] | tuple[object, ...],
) -> tuple[FrozenMetadataValue, ...]: ...


@overload
def freeze_metadata(value: object) -> FrozenMetadataValue: ...


def freeze_metadata(value: object) -> FrozenMetadataValue:
    """Own a recursively immutable copy of JSON-compatible metadata."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            message = "Report metadata must use null and a reason for nonfinite values."
            raise ValueError(message)
        return number
    if isinstance(value, Mapping):
        return MappingProxyType(
            {_metadata_key(key): freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_metadata(item) for item in value)
    message = f"Unsupported report metadata type {type(value).__name__!r}."
    raise TypeError(message)


@overload
def metadata_to_json(value: Mapping[str, object]) -> dict[str, JsonValue]: ...


@overload
def metadata_to_json(value: list[object] | tuple[object, ...]) -> list[JsonValue]: ...


@overload
def metadata_to_json(value: object) -> JsonValue: ...


def metadata_to_json(value: object) -> JsonValue:
    """Return ordinary JSON containers for formatting or persistence."""
    if isinstance(value, Mapping):
        return {
            _metadata_key(key): metadata_to_json(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [metadata_to_json(item) for item in value]
    frozen = freeze_metadata(value)
    if isinstance(frozen, (Mapping, tuple)):
        return metadata_to_json(frozen)
    return frozen


def _finite(value: ReportNumber | None, name: str) -> None:
    """Reject nonfinite canonical numbers instead of hiding them in formatting."""
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        message = f"{name} must be a finite number or None."
        raise ValueError(message)


def _row_identity(rows: Sequence[int | str]) -> dict[str, MetadataValue]:
    """Return a fixed-size identity of ordered row ids for report metadata.

    The digest covers the canonical JSON list of ids, so ``1`` and ``"1"``
    differ and order matters. Readers verify a report by re-hashing the ids;
    reports never carry sample-sized id tuples.
    """
    ids = [int(row) if isinstance(row, Integral) else row for row in rows]
    payload = json.dumps(ids, separators=(",", ":"), ensure_ascii=False)
    return {
        "count": len(ids),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _status(status: str | int, reason: str | None) -> None:
    """Require literal status metadata and a textual optional explanation."""
    if (
        isinstance(status, bool)
        or not isinstance(status, (str, int))
        or (reason is not None and not isinstance(reason, str))
    ):
        message = "Report status must be an integer/string and its reason must be text."
        raise TypeError(message)


@dataclass(frozen=True)
class ParameterRow:
    """Retain unrounded coefficient evidence, including unavailable inference.

    ``valid``, ``status`` and ``unavailable_reason`` describe the estimate and its
    standard error. ``interval_status`` and ``interval_unavailable_reason``
    describe only a requested interval, so a failed calibration can withhold the
    interval while the covariance-based standard error stays visible. ``None``
    means no interval was requested.
    """

    name: str
    estimate: ReportNumber | None
    std_error: ReportNumber | None = None
    statistic: ReportNumber | None = None
    pvalue: ReportNumber | None = None
    ci_lower: ReportNumber | None = None
    ci_upper: ReportNumber | None = None
    valid: bool = True
    status: str | int = "available"
    unavailable_reason: str | None = None
    parameter_id: str | None = None
    interval_status: str | int | None = None
    interval_unavailable_reason: str | None = None

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Require honest availability and finite supplied numerical evidence."""
        if type(self.valid) is not bool:
            message = "Parameter validity must be boolean."
            raise TypeError(message)
        _status(self.status, self.unavailable_reason)
        if self.interval_status is not None or self.interval_unavailable_reason:
            _status(
                "unavailable" if self.interval_status is None else self.interval_status,
                self.interval_unavailable_reason,
            )
        if (
            self.interval_status == "available"
            and self.interval_unavailable_reason is not None
        ):
            message = "An available interval cannot have an unavailable reason."
            raise ValueError(message)
        interval_declared_unavailable = bool(self.interval_unavailable_reason) or (
            self.interval_status is not None and self.interval_status != "available"
        )
        if interval_declared_unavailable:
            if self.ci_lower is not None or self.ci_upper is not None:
                message = "An unavailable interval cannot report endpoints."
                raise ValueError(message)
            if not self.interval_unavailable_reason:
                message = "An unavailable interval requires a reason."
                raise ValueError(message)
        elif self.interval_status == "available" and (
            self.ci_lower is None or self.ci_upper is None
        ):
            message = "An available interval requires both endpoints."
            raise ValueError(message)
        if self.parameter_id is not None and not isinstance(self.parameter_id, str):
            message = "Parameter identities must be strings."
            raise TypeError(message)
        if not isinstance(self.name, str) or not self.name:
            message = "Parameter names must be nonempty strings."
            raise ValueError(message)
        for name in (
            "estimate",
            "std_error",
            "statistic",
            "pvalue",
            "ci_lower",
            "ci_upper",
        ):
            _finite(getattr(self, name), name)
        if not self.valid and not self.unavailable_reason:
            message = "Unavailable parameters require a reason."
            raise ValueError(message)
        if not self.valid and self.status == "available":
            object.__setattr__(self, "status", "unavailable")
        if self.valid and self.estimate is None:
            message = "Available parameter rows require an estimate."
            raise ValueError(message)
        if self.std_error is not None and self.std_error < 0:
            message = "Standard errors cannot be negative."
            raise ValueError(message)
        if self.pvalue is not None and not 0 <= self.pvalue <= 1:
            message = "P-values must lie in [0, 1]."
            raise ValueError(message)
        if (self.ci_lower is None) != (self.ci_upper is None):
            message = "Both interval endpoints must be supplied together."
            raise ValueError(message)
        if (
            self.ci_lower is not None
            and self.ci_upper is not None
            and self.ci_lower > self.ci_upper
        ):
            message = "Interval endpoints must be ordered."
            raise ValueError(message)

    @property
    def interval_available(self) -> bool:
        """Whether both endpoints are present and not declared unavailable."""
        return (
            self.ci_lower is not None
            and self.ci_upper is not None
            and self.interval_status in (None, "available")
        )


@dataclass(frozen=True)
class Metric:
    """Retain one named fit, diagnostic or test quantity and its interpretation."""

    name: str
    value: ReportNumber | None
    definition: str = ""
    available: bool = True
    unavailable_reason: str | None = None
    status: str | int = "available"

    def __post_init__(self) -> None:
        """Require a value or an explicit unavailable reason."""
        if (
            type(self.available) is not bool
            or not isinstance(self.name, str)
            or not isinstance(self.definition, str)
        ):
            message = "Metric name/definition and availability have invalid types."
            raise TypeError(message)
        _status(self.status, self.unavailable_reason)
        _finite(self.value, self.name)
        if not self.name or (self.available and self.value is None):
            message = "Available metrics require a name and value."
            raise ValueError(message)
        if not self.available and not self.unavailable_reason:
            message = "Unavailable metrics require a reason."
            raise ValueError(message)
        if self.available and self.unavailable_reason:
            message = "Available metrics cannot have an unavailable reason."
            raise ValueError(message)
        if not self.available and self.value is not None:
            message = "Unavailable metrics report no value."
            raise ValueError(message)
        if not self.available and self.status == "available":
            object.__setattr__(self, "status", "unavailable")


_LAW_ARITY = {"chi-square": 1, "f": 2, "normal": 2, "resampled": 1}


def _law_parameters(law: str | None, values: object) -> tuple[ReportNumber, ...]:
    """Check that the parameters of a calibrating law match its kind."""
    if not isinstance(values, (tuple, list)):
        message = "Test law parameters must be a sequence of numbers."
        raise TypeError(message)
    parameters = tuple(values)
    for value in parameters:
        if isinstance(value, bool) or not isinstance(value, Real):
            message = "Test law parameters must be numbers."
            raise TypeError(message)
        if not math.isfinite(value):
            message = "Test law parameters must be finite."
            raise ValueError(message)
    if law is None:
        if parameters:
            message = "A test without a law has no law parameters."
            raise ValueError(message)
        return parameters
    if len(parameters) != _LAW_ARITY[law]:
        message = f"The {law} law takes {_LAW_ARITY[law]} parameter(s)."
        raise ValueError(message)
    positive = parameters[1:] if law == "normal" else parameters
    if any(value <= 0 for value in positive):
        message = "Degrees of freedom, scales and draws must be positive."
        raise ValueError(message)
    if law == "resampled" and not isinstance(parameters[0], Integral):
        message = "The number of resampling draws must be an integer."
        raise ValueError(message)
    return parameters


@dataclass(frozen=True)
class TestRow:
    """One calibrated or descriptive test statistic, ready for a table of tests."""

    __test__: ClassVar[bool] = False  # not a pytest class

    name: str
    statistic: ReportNumber | None
    pvalue: ReportNumber | None = None
    law: Literal["chi-square", "f", "normal", "resampled"] | None = None
    law_parameters: tuple[ReportNumber, ...] = ()
    reference: str = ""
    hypothesis: str = ""
    available: bool = True
    unavailable_reason: str | None = None
    status: str | int = "available"

    def __post_init__(self) -> None:  # noqa: C901
        """Require honest availability, a coherent law and finite evidence."""
        if (
            not isinstance(self.name, str)
            or not isinstance(self.reference, str)
            or not isinstance(self.hypothesis, str)
            or type(self.available) is not bool
        ):
            message = (
                "Test name/reference/hypothesis and availability have invalid types."
            )
            raise TypeError(message)
        if not self.name:
            message = "Test names must be nonempty strings."
            raise ValueError(message)
        _status(self.status, self.unavailable_reason)
        _finite(self.statistic, "statistic")
        _finite(self.pvalue, "pvalue")
        if self.pvalue is not None and not 0 <= self.pvalue <= 1:
            message = "P-values must lie in [0, 1]."
            raise ValueError(message)
        if self.available:
            if self.statistic is None or self.unavailable_reason is not None:
                message = "Available tests require a statistic and no reason."
                raise ValueError(message)
        else:
            if not self.unavailable_reason:
                message = "Unavailable tests require a reason."
                raise ValueError(message)
            if self.statistic is not None or self.pvalue is not None:
                message = "Unavailable tests report no statistic or p-value."
                raise ValueError(message)
            if self.status == "available":
                object.__setattr__(self, "status", "unavailable")
        if self.law is not None and self.law not in _LAW_ARITY:
            message = f"Unknown test law {self.law!r}."
            raise ValueError(message)
        if (self.pvalue is None) != (self.law is None):
            message = "A test has a p-value exactly when it has a law."
            raise ValueError(message)
        object.__setattr__(
            self, "law_parameters", _law_parameters(self.law, self.law_parameters)
        )


def _finite_metric(name: str, value: Array, definition: str) -> Metric:
    """Keep undefined native summaries unavailable instead of rendering NaNs."""
    available = bool(jnp.isfinite(value))
    number: ReportNumber | None = None
    if available:
        # Integer native summaries, such as degrees of freedom, stay exact counts.
        number = (
            int(value) if jnp.issubdtype(value.dtype, jnp.integer) else float(value)
        )
    return Metric(
        name,
        number,
        definition,
        available=available,
        unavailable_reason=None if available else "Unavailable in the retained fit.",
    )


@dataclass(frozen=True)
class ReferenceLine:
    """Retain already-computed endpoints of one plotting reference line."""

    x: tuple[ReportNumber, ReportNumber]
    y: tuple[ReportNumber, ReportNumber]
    label: str = ""

    def __post_init__(self) -> None:
        """Own finite endpoint tuples."""
        for name in ("x", "y"):
            values = tuple(getattr(self, name))
            if len(values) != _REFERENCE_ENDPOINT_COUNT:
                message = "Reference lines require two endpoints."
                raise ValueError(message)
            for value in values:
                if value is None:
                    message = "Reference endpoints cannot be missing."
                    raise ValueError(message)
                _finite(value, name)
            object.__setattr__(self, name, values)
        if not isinstance(self.label, str):
            message = "Reference labels must be strings."
            raise TypeError(message)


@dataclass(frozen=True, init=False)
class PlotSpec:
    """Precomputed plotting coordinates with explicit interval interpretation."""

    kind: Literal["scatter", "line", "coefficient_interval"]
    x: tuple[ReportNumber, ...]
    y: tuple[ReportNumber, ...]
    xlabel: str
    ylabel: str
    title: str
    lower: tuple[ReportNumber, ...] | None
    upper: tuple[ReportNumber, ...] | None
    interval_meaning: str | None
    labels: tuple[str, ...]
    reference_lines: tuple[ReferenceLine, ...]
    metadata: Mapping[str, FrozenMetadataValue]
    label: str
    interval_style: Literal["bars", "ribbon"]
    overlays: tuple[PlotSpec, ...]
    markup: Literal["plain", "tex"]

    def __init__(
        self,
        kind: Literal["scatter", "line", "coefficient_interval"],
        x: tuple[ReportNumber, ...],
        y: tuple[ReportNumber, ...],
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        lower: tuple[ReportNumber, ...] | None = None,
        upper: tuple[ReportNumber, ...] | None = None,
        interval_meaning: str | None = None,
        labels: tuple[str, ...] = (),
        reference_lines: tuple[ReferenceLine, ...] = (),
        metadata: Mapping[str, MetadataValue] | None = None,
        label: str = "",
        interval_style: Literal["bars", "ribbon"] = "bars",
        overlays: tuple[PlotSpec, ...] = (),
        markup: Literal["plain", "tex"] = "plain",
    ) -> None:
        """Own caller metadata while retaining only immutable report state."""
        if metadata is not None and not isinstance(metadata, Mapping):
            message = "Plot metadata must be a mapping."
            raise TypeError(message)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "xlabel", xlabel)
        object.__setattr__(self, "ylabel", ylabel)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "interval_meaning", interval_meaning)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "reference_lines", reference_lines)
        object.__setattr__(
            self, "metadata", freeze_metadata({} if metadata is None else metadata)
        )
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "interval_style", interval_style)
        object.__setattr__(self, "overlays", overlays)
        object.__setattr__(self, "markup", markup)
        self.__post_init__()

    def __post_init__(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Validate supplied coordinates without calculating a diagnostic."""
        for value in (self.xlabel, self.ylabel, self.title, *self.labels):
            if not isinstance(value, str):
                message = "Plot display labels must be strings."
                raise TypeError(message)
        if any(not isinstance(line, ReferenceLine) for line in self.reference_lines):
            message = "Plot references must be precomputed ReferenceLine records."
            raise TypeError(message)
        if self.interval_meaning is not None and not isinstance(
            self.interval_meaning, str
        ):
            message = "Interval interpretation must be ordinary text."
            raise TypeError(message)
        if self.kind not in ("scatter", "line", "coefficient_interval"):
            message = f"Unsupported plot kind {self.kind!r}."
            raise ValueError(message)
        for name in ("x", "y", "labels", "reference_lines"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.x or len(self.x) != len(self.y):
            message = "Plot coordinates must be nonempty aligned vectors."
            raise ValueError(message)
        for value in (*self.x, *self.y):
            if value is None:
                message = "Plot coordinates cannot be missing."
                raise ValueError(message)
            _finite(value, "plot coordinate")
        if self.labels and len(self.labels) != len(self.x):
            message = "Plot labels must match the supplied coordinates."
            raise ValueError(message)
        if (self.lower is None) != (self.upper is None):
            message = "Both plotting interval endpoints are required."
            raise ValueError(message)
        if self.lower is not None and self.upper is not None:
            lower, upper = tuple(self.lower), tuple(self.upper)
            if (
                len(lower) != len(self.x)
                or len(upper) != len(self.x)
                or not self.interval_meaning
            ):
                message = (
                    "Plot intervals require aligned endpoints and an explicit meaning."
                )
                raise ValueError(message)
            for low, high in zip(lower, upper, strict=True):
                if low is None or high is None:
                    message = "Plot interval endpoints cannot be missing."
                    raise ValueError(message)
                _finite(low, "lower")
                _finite(high, "upper")
                if low > high:
                    message = "Plot interval endpoints must be ordered."
                    raise ValueError(message)
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)
        if self.kind == "coefficient_interval" and self.lower is None:
            message = "Coefficient-interval plots require interval endpoints."
            raise ValueError(message)
        if self.markup not in ("plain", "tex") or self.interval_style not in (
            "bars",
            "ribbon",
        ):
            msg = "Invalid plot markup or interval style."
            raise ValueError(msg)
        if not isinstance(self.label, str):
            msg = "Plot label must be text."
            raise TypeError(msg)
        object.__setattr__(self, "overlays", tuple(self.overlays))
        if any(
            not isinstance(p, PlotSpec)
            or p.overlays
            or p.xlabel
            or p.ylabel
            or p.title
            or p.markup != self.markup
            for p in self.overlays
        ):
            msg = (
                "Overlays must share markup and have no axes, title or nested overlays."
            )
            raise ValueError(msg)
        if self.markup == "tex" and any(
            "\n" in value or "\r" in value
            for value in (
                self.label,
                self.xlabel,
                self.ylabel,
                self.title,
                *self.labels,
                *(line.label for line in self.reference_lines),
            )
        ):
            msg = "TeX labels cannot contain newlines."
            raise ValueError(msg)
        if self.interval_style == "ribbon" and (
            self.kind != "line"
            or self.lower is None
            or any(a >= b for a, b in zip(self.x, self.x[1:], strict=False))
        ):
            msg = "Ribbons require line intervals and strictly increasing x."
            raise ValueError(msg)

    def to_pgfplots(
        self, *, width_cm: ReportNumber = 12, height_cm: ReportNumber = 7
    ) -> str:
        """Render native PGFplots without fitting or calculating statistics."""
        return _render_pgfplots(self, width_cm=width_cm, height_cm=height_cm)

    def save_pgfplots(
        self,
        path: str | Path,
        *,
        width_cm: ReportNumber = 12,
        height_cm: ReportNumber = 7,
    ) -> None:
        """Write exactly ``to_pgfplots(...)`` to ``path`` as UTF-8."""
        text = self.to_pgfplots(width_cm=width_cm, height_cm=height_cm)
        Path(path).write_text(text, encoding="utf-8", newline="\n")

    def to_matplotlib(self) -> Figure:  # noqa: C901
        """Draw the identical supplied coordinates on the caller's plotting backend."""
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.text import Text  # noqa: PLC0415

        figure, axis = plt.subplots()
        for spec in (self, *self.overlays):
            if spec.kind == "line":
                axis.plot(spec.x, spec.y, label=spec.label or None)
            else:
                axis.scatter(spec.x, spec.y, label=spec.label or None)
            if spec.lower is not None and spec.upper is not None:
                if spec.interval_style == "ribbon":
                    axis.fill_between(spec.x, spec.lower, spec.upper, alpha=0.2)
                else:
                    axis.vlines(spec.x, spec.lower, spec.upper)
            for line in spec.reference_lines:
                axis.plot(line.x, line.y, linestyle="--", label=line.label or None)
        if any(
            spec.label or any(line.label for line in spec.reference_lines)
            for spec in (self, *self.overlays)
        ):
            axis.legend()
        axis.set(xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)
        if self.labels:
            axis.set_xticks(self.x, self.labels)
        if self.markup == "plain":
            for text in figure.findobj(match=Text):
                if isinstance(text, Text):
                    text.set_parse_math(False)
                    text.set_usetex(False)
        return figure


_TEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "%": r"\%",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\u2014": "---",
    "\u2013": "--",
    "\u2026": r"\ldots{}",
    "\u2212": "$-$",
    "\u00d7": r"$\times$",
    "\u00b7": r"$\cdot$",
    "\u00b1": r"$\pm$",
    "\u2264": r"$\leq$",
    "\u2265": r"$\geq$",
    "\u2260": r"$\neq$",
    "\u2248": r"$\approx$",
    "\u221e": r"$\infty$",
    "\u2192": r"$\to$",
    "\u00b2": "$^{2}$",
    "\u00b3": "$^{3}$",
}
_GREEK_LOWER = (
    "alpha,beta,gamma,delta,epsilon,zeta,eta,theta,iota,kappa,lambda,mu,nu,xi,o,"
    "pi,rho,varsigma,sigma,tau,upsilon,phi,chi,psi,omega"
)
# U+03A2 is unassigned, so its slot is empty and skipped.
_GREEK_UPPER = (
    "A,B,Gamma,Delta,E,Z,H,Theta,I,K,Lambda,M,N,Xi,O,Pi,P,,Sigma,T,Upsilon,Phi,X,"
    "Psi,Omega"
)


def _greek_tex(name: str) -> str:
    """Return ASCII for omicron, upright math for Latin lookalikes, else a macro."""
    if name == "o":
        return name
    if len(name) == 1:
        return rf"$\mathrm{{{name}}}$"
    return "$\\" + name + "$"


_TEX_ESCAPE.update(
    {
        chr(0x03B1 + index): _greek_tex(name)
        for index, name in enumerate(_GREEK_LOWER.split(","))
    }
)
_TEX_ESCAPE.update(
    {
        chr(0x0391 + index): _greek_tex(name)
        for index, name in enumerate(_GREEK_UPPER.split(","))
        if name
    }
)


def _tex(text: str) -> str:
    """Escape TeX-special characters and spell glyphs the pinned fonts lack."""
    return "".join(
        _TEX_ESCAPE.get(character, character)
        for character in " ".join(text.splitlines())
    )


def _display(value: ReportNumber | None) -> str:
    """Apply presentation precision without modifying the canonical value."""
    if value is None:
        return "unavailable"
    if isinstance(value, Integral):
        return str(int(value))
    return format(value, ".6g")


def _confidence_label(confidence: float) -> str:
    """Format the supplied confidence level without rounding it to 100%."""
    with localcontext() as context:
        context.prec = 50
        percentage = format(Decimal(str(confidence)) * 100, "f")
    if "." in percentage:
        percentage = percentage.rstrip("0").rstrip(".")
    return percentage + "%"


@dataclass(frozen=True)
class _Cell:
    """One table cell: ``plain`` cells are decimal numbers an ``S`` column aligns."""

    text: str
    kind: Literal["plain", "sci", "none", "raw"]


_PARAMETER_COLUMNS = (
    ("estimate", "Estimate"),
    ("std_error", "SE"),
    ("statistic", "Statistic"),
    ("pvalue", "$p$"),
    ("ci_lower", "Lower"),
    ("ci_upper", "Upper"),
)
_PVALUE_FLOOR = 0.001


def _check_table_options(digits: int, decimals: int | None) -> None:
    if isinstance(digits, bool) or not isinstance(digits, int) or digits < 1:
        msg = "digits must be a positive integer"
        raise ValueError(msg)
    if decimals is not None and (
        isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0
    ):
        msg = "decimals must be a nonnegative integer or None"
        raise ValueError(msg)


def _number_cell(
    value: ReportNumber | None, digits: int, decimals: int | None
) -> _Cell:
    """Format one retained number; integers are exact and zeros are kept."""
    if value is None:
        return _Cell("", "none")
    if isinstance(value, Integral):
        return _Cell(str(int(value)), "plain")
    if decimals is not None:
        return _Cell(f"{value:.{decimals}f}", "plain")
    text = format(value, f"#.{digits}g")
    if "e" not in text:
        return _Cell(text.removesuffix("."), "plain")
    mantissa, exponent = text.split("e")
    if int(exponent) < 0:
        return _Cell(rf"${mantissa} \times 10^{{{int(exponent)}}}$", "sci")
    return _Cell(f"{value:.0f}", "plain")


def _pvalue_cell(value: ReportNumber | None) -> _Cell:
    if value is None:
        return _Cell("---", "raw")
    if value < _PVALUE_FLOOR:
        return _Cell("$<0.001$", "raw")
    return _Cell(f"${value:.3f}$", "raw")


def _footer_cell(
    value: ReportNumber | None, digits: int, decimals: int | None
) -> _Cell:
    """Footer counts are ``num`` so a count never shifts decimal alignment."""
    if isinstance(value, Integral):
        return _Cell(rf"{{\num{{{int(value)}}}}}", "raw")
    return _number_cell(value, digits, decimals)


def _column_spec(cells: Sequence[_Cell]) -> str:
    """Compute the siunitx ``S`` column from the plain decimal cells."""
    plain = [cell.text for cell in cells if cell.kind == "plain"]
    if not plain:
        return "c"
    negative = "-" if any(text.startswith("-") for text in plain) else ""
    whole = max(len(text.lstrip("-").partition(".")[0]) for text in plain)
    fraction = max(len(text.partition(".")[2]) for text in plain)
    fmt = negative + str(whole) + (f".{fraction}" if fraction else "")
    return f"S[table-format={fmt},group-digits=integer]"


def _finish_cell(cell: _Cell, spec: str) -> str:
    aligned = spec.startswith("S")
    if cell.kind == "none":
        return "{---}" if aligned else "---"
    if cell.kind == "sci" and aligned:
        return "{" + cell.text + "}"
    return cell.text


def _row_text(cells: Sequence[str]) -> str:
    """Join cells with ``&`` and no padding around empty ones."""
    text = ""
    for index, cell in enumerate(cells):
        text += (" &" if index else "") + (" " if cell and index else "") + cell
    return text + r" \\"


def _label_text(
    name: str, labels: Mapping[str, str] | None, markup: Literal["plain", "tex"]
) -> str:
    if labels is None or name not in labels:
        return _tex(name)
    label = labels[name]
    if markup == "plain":
        return _tex(label)
    if "\n" in label or "\r" in label:
        msg = "TeX labels cannot contain newlines."
        raise ValueError(msg)
    return label


type FitStatisticName = Literal[
    "n_observations",
    "n_parameters",
    "r_squared",
    "adjusted_r_squared",
    "uncentred_r_squared",
    "log_likelihood",
    "aic",
    "bic",
    "hqic",
]

FIT_STATISTIC_NAMES: tuple[FitStatisticName, ...] = (
    "n_observations",
    "n_parameters",
    "r_squared",
    "adjusted_r_squared",
    "uncentred_r_squared",
    "log_likelihood",
    "aic",
    "bic",
    "hqic",
)

_METRIC_LABELS: Mapping[FitStatisticName, str] = MappingProxyType(
    {
        "n_observations": "Observations",
        "n_parameters": "Parameters",
        "r_squared": "$R^2$",
        "adjusted_r_squared": "Adjusted $R^2$",
        "uncentred_r_squared": "Uncentred $R^2$",
        "log_likelihood": "Log likelihood",
        "aic": "AIC",
        "bic": "BIC",
        "hqic": "HQIC",
    }
)


def _metric_label(
    name: str, labels: Mapping[str, str] | None, markup: Literal["plain", "tex"]
) -> str:
    """Label a metric row: caller label first, then the package TeX default."""
    if labels is not None and name in labels:
        return _label_text(name, labels, markup)
    return _METRIC_LABELS.get(cast("FitStatisticName", name)) or _tex(name)


def _select_metrics(report: Report, names: Sequence[str] | None) -> tuple[Metric, ...]:
    """Pick footer metrics by name, in the requested order."""
    if names is None:
        return report.metrics
    if isinstance(names, str):
        msg = "metrics must be a sequence of metric names, not a string"
        raise TypeError(msg)
    by_name = {metric.name: metric for metric in report.metrics}
    chosen: list[Metric] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            msg = f"Duplicate metric {name!r} in metrics"
            raise ValueError(msg)
        if name not in by_name:
            msg = f"Unknown metric {name!r}; available: {sorted(by_name)}"
            raise ValueError(msg)
        seen.add(name)
        chosen.append(by_name[name])
    return tuple(chosen)


def _parameter_table(
    report: Report,
    digits: int,
    decimals: int | None,
    labels: Mapping[str, str] | None,
    markup: Literal["plain", "tex"],
    metrics: tuple[Metric, ...],
) -> list[str]:
    rows = report.parameters
    names = [
        name
        for index, (name, _) in enumerate(_PARAMETER_COLUMNS)
        if index == 0 or any(getattr(row, name) is not None for row in rows)
    ]
    footers = [_footer_cell(metric.value, digits, decimals) for metric in metrics]
    specs: list[str] = []
    columns: list[list[_Cell]] = []
    for name in names:
        if name == "pvalue":
            cells = [_pvalue_cell(row.pvalue) for row in rows]
            specs.append("r")
        else:
            cells = [_number_cell(getattr(row, name), digits, decimals) for row in rows]
            extra = footers if name == "estimate" else []
            specs.append(_column_spec([*cells, *extra]))
        columns.append(cells)
    heads = {**dict(_PARAMETER_COLUMNS), "statistic": _statistic_head(report)}
    spec_text = "@{}l" + "".join(specs) + "@{}"
    lines = [rf"\begin{{tabular}}{{{spec_text}}}", r"\toprule"]
    lines.extend(_header_lines(report, names, [heads[name] for name in names]))
    lines.append(r"\midrule")
    for index, row in enumerate(rows):
        label = _label_text(row.name, labels, markup)
        lines.append(
            _row_text(
                [
                    label,
                    *(
                        _finish_cell(col[index], sp)
                        for col, sp in zip(columns, specs, strict=True)
                    ),
                ]
            )
        )
    if metrics:
        lines.append(r"\midrule")
        for metric, cell in zip(metrics, footers, strict=True):
            blanks = [""] * (len(names) - 1)
            label = _metric_label(metric.name, labels, markup)
            lines.append(_row_text([label, _finish_cell(cell, specs[0]), *blanks]))
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return lines


def _statistic_head(report: Report, *, plain: bool = False) -> str:
    """Name the coefficient statistic after the law that calibrates it."""
    reference = report.metadata.get("reference")
    letter = (
        {"normal": "z", "t": "t"}.get(reference) if isinstance(reference, str) else None
    )
    if letter is None:
        return "Statistic"
    return letter if plain else f"${letter}$"


def _head(text: str) -> str:
    return text if text == "$p$" else "{" + text + "}"


def _header_lines(report: Report, names: list[str], heads: list[str]) -> list[str]:
    confidence = report.metadata.get("confidence")
    spans = (
        "ci_lower" in names
        and "ci_upper" in names
        and isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
    )
    if not spans:
        return [_row_text(["Parameter", *(_head(h) for h in heads)])]
    cut = names.index("ci_lower")
    lead = [_head(h) for h in heads[:cut]]
    level = _tex(_confidence_label(float(confidence))) + " CI"
    first = _row_text(["Parameter", *lead, rf"\multicolumn{{2}}{{c}}{{{level}}}"])
    return [
        first,
        rf"\cmidrule(lr){{{cut + 2}-{cut + 3}}}",
        _row_text([""] * (cut + 1) + ["{Lower}", "{Upper}"]),
    ]


def _metric_table(
    digits: int,
    decimals: int | None,
    labels: Mapping[str, str] | None,
    markup: Literal["plain", "tex"],
    metrics: tuple[Metric, ...],
) -> list[str]:
    cells = [_number_cell(metric.value, digits, decimals) for metric in metrics]
    spec = _column_spec(cells)
    lines = [
        rf"\begin{{tabular}}{{@{{}}l{spec}@{{}}}}",
        r"\toprule",
        _row_text(["Metric", "{Value}"]),
        r"\midrule",
    ]
    lines.extend(
        _row_text(
            [_metric_label(metric.name, labels, markup), _finish_cell(cell, spec)]
        )
        for metric, cell in zip(metrics, cells, strict=True)
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return lines


def _count_text(value: ReportNumber) -> str:
    if isinstance(value, Integral) or float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def _law_number_text(value: ReportNumber, digits: int) -> str:
    """Print a law parameter with significant digits and a short exponent."""
    if isinstance(value, Integral) or float(value).is_integer():
        return str(int(value))
    text = format(value, f".{digits}g")
    if "e" in text:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}e{int(exponent)}"
    return text


def _reference_cell(row: TestRow, digits: int) -> str:
    params = [_count_text(value) for value in row.law_parameters]
    match row.law:
        case "chi-square":
            return rf"$\chi^2({params[0]})$"
        case "f":
            return f"$F({params[0]}, {params[1]})$"
        case "normal":
            if row.law_parameters == (0, 1):
                return "$N(0, 1)$"
            location, scale = (
                rf"\num{{{_law_number_text(value, digits)}}}"
                for value in row.law_parameters
            )
            return f"$N({location}, {scale}^2)$"
        case "resampled":
            return _tex(row.reference)
        case _:
            return "---"


def _test_lines(tests: Sequence[TestRow]) -> list[str]:
    """Format the plain-text table of tests with its availability and null notes."""
    lines = ["Test\tStatistic\tReference\tp-value"] if tests else []
    for test in tests:
        reference = test.reference or "unavailable"
        cells = [test.name, _display(test.statistic), reference, _display(test.pvalue)]
        lines.append("\t".join(cells))
        if test.unavailable_reason:
            lines.append(
                f"  {test.name}: {test.unavailable_reason} (status={test.status})"
            )
        if test.hypothesis:
            lines.append(f"  {test.name}: {test.hypothesis}")
    return lines


def _test_table(
    report: Report,
    digits: int,
    decimals: int | None,
    labels: Mapping[str, str] | None,
    markup: Literal["plain", "tex"],
) -> list[str]:
    rows = report.tests
    cells = [_number_cell(row.statistic, digits, decimals) for row in rows]
    spec = _column_spec(cells)
    calibrated = any(row.law is not None for row in rows)
    columns = "cr" if calibrated else ""
    heads = ["Test", "{Statistic}", *(["Reference", "$p$"] if calibrated else [])]
    lines = [
        rf"\begin{{tabular}}{{@{{}}l{spec}{columns}@{{}}}}",
        r"\toprule",
        _row_text(heads),
        r"\midrule",
    ]
    for row, cell in zip(rows, cells, strict=True):
        parts = [_label_text(row.name, labels, markup), _finish_cell(cell, spec)]
        if calibrated:
            parts.extend(
                [
                    _reference_cell(row, digits),
                    _finish_cell(_pvalue_cell(row.pvalue), "r"),
                ]
            )
        lines.append(_row_text(parts))
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return lines


def _table_notes(
    report: Report,
    labels: Mapping[str, str] | None,
    markup: Literal["plain", "tex"],
) -> list[str]:
    notes = [
        row.name + ": " + row.unavailable_reason
        for row in report.parameters
        if row.unavailable_reason
    ]
    notes.extend(
        row.name + " interval: " + row.interval_unavailable_reason
        for row in report.parameters
        if row.interval_unavailable_reason
    )
    head = len(notes)
    metric_notes = [
        _metric_label(metric.name, labels, markup)
        + ": "
        + _tex(metric.unavailable_reason or metric.definition)
        for metric in report.metrics
        if metric.unavailable_reason or metric.definition
    ]
    for test in report.tests:
        if test.hypothesis:
            notes.append(test.name + ": " + test.hypothesis)
        if test.unavailable_reason:
            notes.append(test.name + ": " + test.unavailable_reason)
    notes.extend(report.notes)
    notes.extend(report.warnings)
    return [
        *(r"\par " + _tex(note) for note in notes[:head]),
        *(r"\par " + note for note in metric_notes),
        *(r"\par " + _tex(note) for note in notes[head:]),
    ]


_VALUES_HEADER = (
    (
        r"\providecommand{\jcDeclareValue}[2]"
        r"{\expandafter\gdef\csname jc@value@#1\endcsname{#2}}"
    ),
    (
        r"\providecommand{\jcvalue}[1]{\ifcsname jc@value@#1\endcsname"
        r"\csname jc@value@#1\endcsname\else\PackageError{statadic}"
        r"{Undefined value '#1'}{Export it with Report.save_values.}\fi}"
    ),
    r"\providecommand{\jcnum}[1]{\num{\jcvalue{#1}}}",
)
_VALUES_PREFIX = re.compile(r"[a-z][a-z0-9-]*")
_PARAMETER_QUANTITIES = (
    ("estimate", "estimate"),
    ("se", "std_error"),
    ("statistic", "statistic"),
    ("p", "pvalue"),
    ("lower", "ci_lower"),
    ("upper", "ci_upper"),
)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        msg = f"{text!r} has no letters or digits to build a value key from"
        raise ValueError(msg)
    return slug


def _value_text(value: ReportNumber, digits: int, decimals: int | None) -> str:
    """Format one siunitx input: exact integers, table significant digits."""
    if isinstance(value, Integral):
        return str(int(value))
    if decimals is not None:
        return f"{value:.{decimals}f}"
    text = format(value, f"#.{digits}g")
    if "e" not in text:
        return text.removesuffix(".")
    mantissa, exponent = text.split("e")
    if int(exponent) < 0:
        return f"{mantissa}e{int(exponent)}"
    return f"{value:.0f}"


def _pvalue_text(value: ReportNumber) -> str:
    if value < _PVALUE_FLOOR:
        return "< 0.001"
    return f"= {value:.3f}"


def _value_entries(  # noqa: C901
    report: Report, prefix: str, digits: int, decimals: int | None
) -> list[tuple[str, str, str, str | None]]:
    """Return ``(key, source, text, unavailable reason)`` in report order."""
    entries: list[tuple[str, str, str, str | None]] = []

    def add(
        key: str, source: str, value: ReportNumber | None, *, p: bool = False
    ) -> None:
        if value is not None:
            text = _pvalue_text(value) if p else _value_text(value, digits, decimals)
            entries.append((key, source, text, None))

    def missing(key: str, source: str, reason: str | None) -> None:
        entries.append((key, source, "", reason or "unavailable"))

    for row in report.parameters:
        base = f"{prefix}-{_slug(row.name)}"
        source = f"parameter {row.name!r}"
        for suffix, field in _PARAMETER_QUANTITIES:
            add(
                f"{base}-{suffix}",
                source,
                getattr(row, field),
                p=field == "pvalue",
            )
        if not row.valid and row.estimate is None:
            missing(f"{base}-estimate", source, row.unavailable_reason)
        if not row.interval_available and (
            row.interval_unavailable_reason or row.interval_status is not None
        ):
            for suffix in ("lower", "upper"):
                missing(f"{base}-{suffix}", source, row.interval_unavailable_reason)
    for metric in report.metrics:
        key = f"{prefix}-{_slug(metric.name)}"
        source = f"metric {metric.name!r}"
        if metric.available:
            add(key, source, metric.value)
        else:
            missing(key, source, metric.unavailable_reason)
    for test in report.tests:
        base = f"{prefix}-{_slug(test.name)}"
        source = f"test {test.name!r}"
        if not test.available:
            missing(f"{base}-statistic", source, test.unavailable_reason)
            continue
        add(f"{base}-statistic", source, test.statistic)
        add(f"{base}-p", source, test.pvalue, p=True)
        names = {"chi-square": ("df",), "f": ("df1", "df2")}.get(test.law or "", ())
        for name, df in zip(names, test.law_parameters, strict=False):
            entries.append((f"{base}-{name}", source, _count_text(df), None))
    return entries


def _pgf_number(value: ReportNumber) -> str:
    """Print an integer exactly, or the shortest text that reads back to a float."""
    if isinstance(value, int):
        return str(value)
    return repr(float(value)).removesuffix(".0")


def _pgf_points(xs: Sequence[ReportNumber], ys: Sequence[ReportNumber]) -> str:
    return " ".join(
        f"({_pgf_number(x)},{_pgf_number(y)})" for x, y in zip(xs, ys, strict=True)
    )


def _render_pgfplots(
    spec: PlotSpec, *, width_cm: ReportNumber = 12, height_cm: ReportNumber = 7
) -> str:
    """Serialize supplied coordinates as an includable native PGFplots fragment."""
    for value in (width_cm, height_cm):
        _finite(value, "plot dimension")
        if not math.isfinite(value) or value <= 0:
            message = "Plot dimensions must be finite positive centimetres."
            raise ValueError(message)

    def label(value: str) -> str:
        return _tex(value) if spec.markup == "plain" else value

    options = [f"width={width_cm:g}cm", f"height={height_cm:g}cm"]
    options.extend(
        f"{name}={{{label(value)}}}"
        for name, value in (
            ("xlabel", spec.xlabel),
            ("ylabel", spec.ylabel),
            ("title", spec.title),
        )
        if value
    )
    if spec.labels:
        options.extend(
            [
                "xtick={" + ",".join(_pgf_number(value) for value in spec.x) + "}",
                "xticklabels={"
                + ",".join("{" + label(item) + "}" for item in spec.labels)
                + "}",
            ]
        )
    lines = [r"\begin{tikzpicture}", r"\begin{axis}[" + ",".join(options) + "]"]
    series = (spec, *spec.overlays)
    unlabelled = 0
    for index, item in enumerate(series):
        _pgf_series(lines, item, label)
        if not item.label:
            unlabelled += 1
            if index < len(series) - 1:
                lines.append(rf"\pgfplotsset{{cycle list shift={unlabelled}}}")
    lines.extend([r"\end{axis}", r"\end{tikzpicture}"])
    return "\n".join(lines) + "\n"


def _pgf_series(lines: list[str], spec: PlotSpec, label: Callable[[str], str]) -> None:
    if spec.lower is not None and spec.upper is not None:
        lines.append(
            "% Interval meaning: "
            + " ".join((spec.interval_meaning or "").splitlines())
        )
        if spec.interval_style == "ribbon":
            coordinates = _pgf_points(
                [*spec.x, *reversed(spec.x)], [*spec.lower, *reversed(spec.upper)]
            )
            lines.append(
                r"\addplot+[draw=none,mark=none,fill=.,fill opacity=0.2,forget plot] "
                r"coordinates {" + coordinates + "} -- cycle;"
            )
        else:
            for x, low, high in zip(spec.x, spec.lower, spec.upper, strict=True):
                lines.append(
                    r"\addplot+[no marks,forget plot] coordinates "
                    "{" + _pgf_points((x, x), (low, high)) + "};"
                )
    style = "no marks" if spec.kind == "line" else "only marks"
    options = style + ("" if spec.label else ",forget plot")
    lines.append(
        rf"\addplot+[{options}] coordinates {{{_pgf_points(spec.x, spec.y)}}};"
    )
    if spec.label:
        lines.append(r"\addlegendentry{" + label(spec.label) + "}")
    for line in spec.reference_lines:
        options = "black,dashed,no marks" + ("" if line.label else ",forget plot")
        lines.append(
            rf"\addplot[{options}] coordinates {{{_pgf_points(line.x, line.y)}}};"
        )
        if line.label:
            lines.append(r"\addlegendentry{" + label(line.label) + "}")


@dataclass(frozen=True, init=False)
class Report:
    """Immutable cross-family report whose renderers format retained quantities.

    Args:
        title: Heading for the rendered report.
        parameters: Ordered parameter estimates and optional uncertainty.
        metrics: Named fit or diagnostic metrics.
        metadata: Descriptive metadata copied into immutable retained state.
        warnings: Warnings about interpretation or availability.
        plots: Plot specifications attached to the report.
        notes: Explanatory notes shown with the report.
        tests: Named hypothesis-test rows.

    Raises:
        TypeError: If metadata or a report record has an unsupported type.
        ValueError: If parameter identities or test names are repeated.

    Examples:
        >>> Report("Example").to_text()
        'Example'

    """

    title: str
    parameters: tuple[ParameterRow, ...]
    metrics: tuple[Metric, ...]
    metadata: Mapping[str, FrozenMetadataValue]
    warnings: tuple[str, ...]
    plots: tuple[PlotSpec, ...]
    notes: tuple[str, ...]
    tests: tuple[TestRow, ...]

    def __init__(
        self,
        title: str,
        parameters: tuple[ParameterRow, ...] = (),
        metrics: tuple[Metric, ...] = (),
        metadata: Mapping[str, MetadataValue] | None = None,
        warnings: tuple[str, ...] = (),
        plots: tuple[PlotSpec, ...] = (),
        notes: tuple[str, ...] = (),
        tests: tuple[TestRow, ...] = (),
    ) -> None:
        """Normalize accepted metadata into immutable retained state once."""
        if metadata is not None and not isinstance(metadata, Mapping):
            message = "Report metadata must be a mapping."
            raise TypeError(message)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(
            self, "metadata", freeze_metadata({} if metadata is None else metadata)
        )
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "plots", plots)
        object.__setattr__(self, "notes", notes)
        object.__setattr__(self, "tests", tests)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Own immutable nested metadata and ordered report sections."""
        if not isinstance(self.title, str):
            message = "Report title must be text."
            raise TypeError(message)
        for name, expected in (
            ("parameters", ParameterRow),
            ("metrics", Metric),
            ("tests", TestRow),
            ("plots", PlotSpec),
            ("warnings", str),
            ("notes", str),
        ):
            if any(not isinstance(item, expected) for item in getattr(self, name)):
                message = f"Invalid {name} report records."
                raise TypeError(message)
        for name in ("parameters", "metrics", "tests", "warnings", "plots", "notes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        ids = [row.parameter_id or row.name for row in self.parameters]
        if len(set(ids)) != len(ids):
            message = "Report parameter identities must be unique."
            raise ValueError(message)
        names = [row.name for row in self.tests]
        if len(set(names)) != len(names):
            message = "Report test names must be unique."
            raise ValueError(message)

    def to_text(self) -> str:
        """Format a plain-text snapshot without triggering post-estimation work."""
        lines = [self.title]
        if self.parameters:
            head = _statistic_head(self, plain=True)
            lines.append(f"Parameter\tEstimate\tSE\t{head}\tp-value\tLower\tUpper")
        for row in self.parameters:
            lines.append(
                "\t".join(
                    [
                        row.name,
                        *[
                            _display(value)
                            for value in (
                                row.estimate,
                                row.std_error,
                                row.statistic,
                                row.pvalue,
                                row.ci_lower,
                                row.ci_upper,
                            )
                        ],
                    ]
                )
            )
            if row.unavailable_reason:
                lines.append(
                    f"  {row.name}: {row.unavailable_reason} (status={row.status})"
                )
            if row.interval_unavailable_reason:
                lines.append(
                    f"  {row.name} interval: {row.interval_unavailable_reason} "
                    f"(status={row.interval_status})"
                )
        lines.extend(_test_lines(self.tests))
        for metric in self.metrics:
            lines.append(f"{metric.name}: {_display(metric.value)}")
            if metric.definition:
                lines.append(f"  {metric.definition}")
            if metric.unavailable_reason:
                lines.append(f"  {metric.unavailable_reason} (status={metric.status})")
        if self.metadata:
            lines.append(
                json.dumps(
                    metadata_to_json(self.metadata), ensure_ascii=False, sort_keys=True
                )
            )
        lines.extend(self.notes)
        lines.extend(f"Warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return the already-computed plain report."""
        return self.to_text()

    def coefficient_plot(self, *, reference_value: float = 0.0) -> PlotSpec:
        """Plot retained coefficient intervals without recomputing inference."""
        confidence = self.metadata.get("confidence")
        if confidence is None or isinstance(confidence, bool):
            msg = "Coefficient plots require retained confidence metadata."
            raise ValueError(msg)
        if not isinstance(confidence, (int, float)):
            msg = "Coefficient plot confidence must be numeric."
            raise TypeError(msg)
        if not self.parameters or any(
            row.estimate is None or row.ci_lower is None or row.ci_upper is None
            for row in self.parameters
        ):
            msg = "Coefficient plots require estimates and intervals for every row."
            raise ValueError(msg)
        x = tuple(float(i) for i in range(len(self.parameters)))
        covariance = self.metadata.get("covariance")
        options = self.metadata.get("covariance_options")
        if isinstance(options, Mapping) and "kind" in options:
            covariance = options["kind"]
        meaning = (
            f"{_confidence_label(confidence)} intervals; "
            f"{self.metadata.get('reference')} reference; "
            f"covariance {covariance}"
        )
        return PlotSpec(
            "coefficient_interval",
            x,
            tuple(
                float(row.estimate)
                for row in self.parameters
                if row.estimate is not None
            ),
            lower=tuple(
                float(row.ci_lower)
                for row in self.parameters
                if row.ci_lower is not None
            ),
            upper=tuple(
                float(row.ci_upper)
                for row in self.parameters
                if row.ci_upper is not None
            ),
            labels=tuple(row.name for row in self.parameters),
            interval_meaning=meaning,
            reference_lines=(
                ReferenceLine(
                    (x[0] - 0.5, x[-1] + 0.5),
                    (reference_value, reference_value),
                    "reference",
                ),
            ),
        )

    def require_available(
        self,
        quantity: Literal["estimate", "std_error", "interval", "statistic", "pvalue"],
        *,
        parameters: Sequence[str] | None = None,
    ) -> Report:
        """Require a retained quantity for every selected parameter.

        Args:
            quantity: Estimate, standard error, interval, statistic or p-value.
            parameters: Selected parameter names, or ``None`` for all rows.

        Returns:
            This report, so the check can precede an export.

        Raises:
            ValueError: If the quantity or names are invalid or a value is unavailable.

        """
        if quantity not in ("estimate", "std_error", "interval", "statistic", "pvalue"):
            message = f"Unknown report quantity {quantity!r}."
            raise ValueError(message)
        selected = (
            {row.name for row in self.parameters}
            if parameters is None
            else set(parameters)
        )
        unknown = selected.difference(row.name for row in self.parameters)
        if unknown or not selected:
            message = f"Expected existing parameter names; unknown: {sorted(unknown)}."
            raise ValueError(message)
        for row in self.parameters:
            if row.name not in selected:
                continue
            if quantity == "interval":
                available = row.interval_available
                reason = row.interval_unavailable_reason or row.unavailable_reason
            else:
                available = getattr(row, quantity) is not None
                reason = row.unavailable_reason
            if not available:
                message = (
                    f"{row.name}: {quantity} unavailable ({reason or 'not reported'})."
                )
                raise ValueError(message)
        return self

    def _repr_html_(self) -> str:
        """Render a bounded preview; explicit exports retain the complete report."""
        preview = replace(
            self,
            parameters=self.parameters[:_PREVIEW_ITEMS],
            metrics=self.metrics[:_PREVIEW_ITEMS],
            tests=self.tests[:_PREVIEW_ITEMS],
            metadata={
                key: _preview_metadata(value)
                for key, value in list(self.metadata.items())[:_PREVIEW_ITEMS]
            },
            warnings=self.warnings[:_PREVIEW_ITEMS],
        )
        rendered = preview.to_html()
        if preview == self and len(rendered) <= _NOTEBOOK_BUDGET:
            return rendered
        notice = "Notebook preview; use report.to_html() for the complete report."
        if len(rendered) > _NOTEBOOK_BUDGET:
            # Slice text before escaping so truncation never breaks HTML markup.
            text = "\n".join((*preview.warnings, preview.to_text()))
            rendered = "<section><pre>" + html.escape(text[:1_500]) + "</pre></section>"
        return rendered + "<p>" + notice + "</p>"

    def to_html(self) -> str:
        """Escape labels and render the same retained numerical sections as HTML."""
        lines = ["<section><h2>" + html.escape(self.title) + "</h2>"]
        if self.parameters:
            head = html.escape(_statistic_head(self, plain=True))
            lines.append(
                "<table><thead><tr><th>Parameter</th><th>Estimate</th><th>SE</th><th>"
                + head
                + "</th><th>p-value</th><th>Lower</th><th>Upper</th>"
                + "<th>Status</th></tr></thead><tbody>"
            )
        for row in self.parameters:
            cells = [
                row.name,
                *[
                    _display(value)
                    for value in (
                        row.estimate,
                        row.std_error,
                        row.statistic,
                        row.pvalue,
                        row.ci_lower,
                        row.ci_upper,
                    )
                ],
                (row.unavailable_reason or str(row.status))
                + (
                    f"; interval: {row.interval_unavailable_reason}"
                    if row.interval_unavailable_reason
                    else ""
                ),
            ]
            lines.append(
                "<tr>"
                + "".join("<td>" + html.escape(cell) + "</td>" for cell in cells)
                + "</tr>"
            )
        if self.parameters:
            lines.append("</tbody></table>")
        if self.tests:
            lines.append(
                "<table><thead><tr><th>Test</th><th>Statistic</th><th>Reference</th><th>p-value</th><th>Status</th></tr></thead><tbody>"
            )
        for test in self.tests:
            cells = [
                test.name,
                _display(test.statistic),
                test.reference or "unavailable",
                _display(test.pvalue),
                test.unavailable_reason or str(test.status),
            ]
            lines.append(
                "<tr>"
                + "".join("<td>" + html.escape(cell) + "</td>" for cell in cells)
                + "</tr>"
            )
        if self.tests:
            lines.append("</tbody></table>")
        lines.extend(
            "<p>" + html.escape(f"{test.name}: {test.hypothesis}") + "</p>"
            for test in self.tests
            if test.hypothesis
        )
        for metric in self.metrics:
            label = f"{metric.name}: {_display(metric.value)}"
            detail = metric.unavailable_reason or metric.definition
            lines.append(
                "<p>"
                + html.escape(label)
                + (" — " + html.escape(detail) if detail else "")
                + "</p>"
            )
        if self.metadata:
            lines.append(
                "<pre>"
                + html.escape(
                    json.dumps(
                        metadata_to_json(self.metadata),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                + "</pre>"
            )
        lines.extend("<p>" + html.escape(note) + "</p>" for note in self.notes)
        lines.extend(
            "<p>Warning: " + html.escape(warning) + "</p>" for warning in self.warnings
        )
        lines.append("</section>")
        return "\n".join(lines)

    def to_latex(
        self,
        *,
        digits: int = 6,
        decimals: int | None = None,
        labels: Mapping[str, str] | None = None,
        label_markup: Literal["plain", "tex"] = "plain",
        notes: bool = True,
        metrics: Sequence[str] | None = None,
    ) -> str:
        """Render booktabs with siunitx columns; counts stay exact.

        ``digits`` keeps significant digits; ``decimals`` fixes decimal places.
        ``labels`` maps parameter or metric names to display labels and
        overrides the default TeX labels of the standard fit statistics.
        ``metrics`` names the retained metrics shown in the footer (or the
        metric-only table), in that order; ``None`` shows all and ``()`` none.
        Unknown or duplicated names raise ``ValueError``. Notes are unaffected;
        pass ``notes=False`` to omit them.

        Args:
            digits: Significant digits used for numbers.
            decimals: Optional fixed number of decimal places.
            labels: Optional display labels for parameters and metrics.
            label_markup: Interpret labels as plain text or trusted TeX.
            notes: Include retained report notes.
            metrics: Ordered metric names to render, or ``None`` for all.

        Returns:
            A LaTeX string containing the report tables.

        Raises:
            ValueError: If formatting options or metric names are invalid.

        """
        _check_table_options(digits, decimals)
        selected = _select_metrics(self, metrics)
        if label_markup not in ("plain", "tex"):
            msg = "label_markup must be 'plain' or 'tex'"
            raise ValueError(msg)
        lines = ["% " + " ".join(self.title.splitlines())]
        tables: list[list[str]] = []
        if self.parameters:
            tables.append(
                _parameter_table(self, digits, decimals, labels, label_markup, selected)
            )
        if self.tests:
            tables.append(_test_table(self, digits, decimals, labels, label_markup))
        if selected and not self.parameters:
            tables.append(
                _metric_table(digits, decimals, labels, label_markup, selected)
            )
        for index, table in enumerate(tables):
            if index:
                lines.append(r"\par\medskip")
            lines.extend(table)
        if notes:
            lines.extend(_table_notes(self, labels, label_markup))
        if self.metadata:
            lines.append(
                "% "
                + " ".join(
                    json.dumps(
                        metadata_to_json(self.metadata),
                        ensure_ascii=False,
                        sort_keys=True,
                    ).splitlines()
                )
            )
        return "\n".join(lines) + "\n"

    def save_latex(
        self,
        path: str | Path,
        *,
        digits: int = 6,
        decimals: int | None = None,
        labels: Mapping[str, str] | None = None,
        label_markup: Literal["plain", "tex"] = "plain",
        notes: bool = True,
        metrics: Sequence[str] | None = None,
    ) -> None:
        """Write exactly ``to_latex(...)`` to ``path`` as UTF-8."""
        text = self.to_latex(
            digits=digits,
            decimals=decimals,
            labels=labels,
            label_markup=label_markup,
            notes=notes,
            metrics=metrics,
        )
        Path(path).write_text(text, encoding="utf-8", newline="\n")

    def to_values(
        self, *, prefix: str, digits: int = 6, decimals: int | None = None
    ) -> str:
        """Return a TeX file declaring each available number as a named value.

        Keys are ``{prefix}-{slug(name)}-{quantity}`` for parameters and tests and
        ``{prefix}-{slug(name)}`` for metrics. Unavailable quantities are listed
        in comments and not declared.
        """
        _check_table_options(digits, decimals)
        if not isinstance(prefix, str) or not _VALUES_PREFIX.fullmatch(prefix):
            msg = "prefix must match ^[a-z][a-z0-9-]*$"
            raise ValueError(msg)
        seen: dict[str, str] = {}
        lines = [
            "% Generated by statadic Report.save_values from: "
            + " ".join(self.title.split()),
            *_VALUES_HEADER,
        ]
        for key, source, text, reason in _value_entries(self, prefix, digits, decimals):
            if key in seen:
                msg = f"Value key {key!r} is produced by both {seen[key]} and {source}"
                raise ValueError(msg)
            seen[key] = source
            if reason is None:
                lines.append(rf"\jcDeclareValue{{{key}}}{{{text}}}")
            else:
                lines.append(f"% unavailable: {key} ({' '.join(reason.split())})")
        return "\n".join(lines) + "\n"

    def save_values(
        self,
        path: str | Path,
        *,
        prefix: str,
        digits: int = 6,
        decimals: int | None = None,
    ) -> None:
        """Write exactly ``to_values(...)`` to ``path`` as UTF-8."""
        text = self.to_values(prefix=prefix, digits=digits, decimals=decimals)
        Path(path).write_text(text, encoding="utf-8", newline="\n")
