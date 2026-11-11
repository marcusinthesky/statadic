"""Shared host prediction preparation, independent of fitting and persistence."""

from __future__ import annotations

from collections.abc import Iterable  # noqa: TC003 -- runtime annotations
from dataclasses import dataclass, replace
from html import escape
from typing import Literal

from statadic.api.data import Observations  # noqa: TC001
from statadic.api.design import FittedDesign  # noqa: TC001
from statadic.api.inputs import as_observations
from statadic.api.roles import (  # noqa: TC001 -- runtime annotations
    InputRole,
    merge_input_roles,
)
from statadic.core.typing import Array  # noqa: TC001

__all__ = [
    "PredictionFamily",
    "PredictionKind",
    "PredictionOption",
    "PredictionOptions",
]

type PredictionFamily = Literal[
    "linear", "poisson", "logistic", "gamma", "nb2", "probit", "positive_poisson"
]
type PredictionKind = Literal["mean", "link", "rate", "probability"]

_KINDS: dict[PredictionFamily, tuple[PredictionKind, ...]] = {
    "linear": ("mean",),
    "poisson": ("mean", "link", "rate"),
    "positive_poisson": ("mean", "link", "rate"),
    "logistic": ("probability", "mean", "link"),
    "gamma": ("mean", "link"),
    "nb2": ("mean", "link", "rate"),
    "probit": ("probability", "mean", "link"),
}


@dataclass(frozen=True)
class PredictionOption:
    """One fitted target, the columns it consumes and its conditions of use."""

    kind: str
    meaning: str
    required_columns: tuple[str, ...]
    condition: str = ""
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        """Whether this fit supports the target for inputs meeting its conditions."""
        return self.unavailable_reason is None


@dataclass(frozen=True)
class PredictionOptions:
    """Inspectable fitted prediction contract. Inspection never evaluates new rows."""

    rows: tuple[PredictionOption, ...]

    @classmethod
    def for_fit(
        cls, rows: Iterable[PredictionOption], *, valid: bool
    ) -> PredictionOptions:
        """Qualify each supported target against the fitted numerical status."""
        return cls(
            tuple(
                row
                if valid or row.unavailable_reason
                else replace(row, unavailable_reason="Invalid fitted parameters.")
                for row in rows
            )
        )

    def _repr_html_(self) -> str:
        labels = ("Target", "Meaning", "Required columns", "Conditions", "Availability")
        head = "".join(f"<th>{label}</th>" for label in labels)
        rows = []
        for row in self.rows:
            values = (
                row.kind,
                row.meaning,
                ", ".join(row.required_columns),
                row.condition,
                row.unavailable_reason or "Available",
            )
            rows.append(
                "<tr>" + "".join(f"<td>{escape(v)}</td>" for v in values) + "</tr>"
            )
        body = "".join(rows)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _fitted_regression_options(
    family: PredictionFamily,
    design: FittedDesign,
    *,
    valid: bool,
    offset: str | None = None,
    exposure: str | None = None,
    trials: str | None = None,
) -> PredictionOptions:
    """Bind each supported target to the names actually declared by this fit."""
    meanings = {
        "mean": "Expected successes."
        if family in ("probit", "logistic")
        else "Expected response given a positive count."
        if family == "positive_poisson"
        else "Expected response.",
        "probability": "Probability of success on one trial.",
        "rate": "Count intensity per unit exposure.",
        "link": "Linear predictor including declared offsets and log exposure.",
    }
    rows = []
    for kind in _supported_prediction_kinds(family):
        roles = (
            offset,
            exposure if kind != "rate" else None,
            trials if kind == "mean" else None,
        )
        columns = tuple(
            dict.fromkeys((*design.input_columns, *(s for s in roles if s is not None)))
        )
        rows.append(PredictionOption(kind, meanings[kind], columns))
    return PredictionOptions.for_fit(rows, valid=valid)


def _spatial_prediction_options(
    columns: tuple[str, ...],
    *,
    response: str | None,
    condition: str,
    valid: bool,
    unavailable_reason: str | None = None,
) -> PredictionOptions:
    """Describe fitted spatial means and observed neighbor requirements."""
    reduced = PredictionOption(
        "reduced-form",
        "Mean accounting for fitted response feedback.",
        columns,
        condition,
        unavailable_reason,
    )
    structural = PredictionOption(
        "structural",
        "Structural predictor evaluated at observed neighbor responses."
        if response
        else "Structural predictor without response feedback.",
        tuple(dict.fromkeys((*columns, response))) if response is not None else columns,
        condition,
        unavailable_reason,
    )
    return PredictionOptions.for_fit((reduced, structural), valid=valid)


def _supported_prediction_kinds(family: PredictionFamily) -> tuple[PredictionKind, ...]:
    """Return the supported prediction scales for a host regression family."""
    return _KINDS[family]


def _default_prediction_kind(family: PredictionFamily) -> PredictionKind:
    """Return the family's default prediction scale."""
    return _supported_prediction_kinds(family)[0]


def _validate_prediction_kind(family: PredictionFamily, kind: str) -> None:
    """Require a prediction scale supported by the supplied family."""
    allowed = _supported_prediction_kinds(family)
    if kind not in allowed:
        message = f"Prediction kind for {family} must be one of {allowed}."
        raise ValueError(message)


def _prediction_role(
    data: Observations, name: str | None, role: str, kind: str, *, required: bool = True
) -> Array | None:
    """Read a declared prediction role only when supplied or required."""
    if name is None:
        return None
    if name not in data.columns:
        if not required:
            return None
        message = f"Column {name!r} required for {role} in {kind!r} prediction."
        raise KeyError(message)
    return data.numeric(name)


def _prediction_design(data: Observations, design: FittedDesign, kind: str) -> Array:
    """Transform the retained design, identifying missing prediction columns."""
    try:
        return design.transform(data)
    except KeyError as exc:
        message = f"Column {exc.args[0]!r} required for design in {kind!r} prediction."
        raise KeyError(message) from exc


def _prediction_inputs(
    data: object,
    design: FittedDesign,
    kind: str,
    *,
    offset: str | None = None,
    exposure: str | None = None,
    trials: str | None = None,
    additional: tuple[str, ...] = (),
) -> Observations:
    """Adapt named prediction columns using retained design and family roles."""
    names = (offset, exposure, trials if kind == "mean" else None, *additional)
    numeric: dict[str, InputRole] = {
        name: "numeric" for name in names if name is not None
    }
    roles = merge_input_roles(design.input_columns, numeric)
    return as_observations(data, roles, broadcast_scalars=True)
