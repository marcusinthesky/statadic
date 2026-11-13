"""Closed codecs for retained predictor expressions and coordinate maps."""

from __future__ import annotations

from statadic.api.design import (
    CyclicSpline,
    FittedCategorical,
    FittedDesign,
    FittedDistanceToLandmarks,
    FittedInteraction,
    FittedIntercept,
    FittedLinear,
    FittedSpline,
    FittedSumTerm,
    FittedTerm,  # noqa: TC001 -- runtime artifact annotations
    Fourier,
    Spline,
)
from statadic.api.ordination_terms import (
    FittedOrdination,
    GeodesicCoordinateMap,
    KernelCoordinateMap,
    LinearCoordinateMap,
)

from ._geometry import decode_geometry, decode_geometry_record, encode_geometry
from ._ordination import (
    decode_coordinate_map,
    decode_coordinate_record,
    encode_coordinate_map,
)
from ._schema import (  # noqa: TC001 -- runtime artifact annotations
    DesignRecord,
    OrdinationTerm,
    TermRecord,
)
from ._store import (
    Arrays,  # noqa: TC001 -- runtime artifact annotations
    ArtifactError,
    integer,
    items,
    labels,
    number,
    record,
    text,
)

_TERMS = {
    CyclicSpline,
    FittedSpline,
    Fourier,
    FittedIntercept,
    FittedLinear,
    FittedCategorical,
    FittedSumTerm,
    FittedInteraction,
    FittedDistanceToLandmarks,
    FittedOrdination,
}


def encode_design(design: FittedDesign, arrays: Arrays) -> DesignRecord:
    """Encode the exact retained feature expression and ordered coefficient names."""
    return {
        "terms": _encode_term(design.terms, arrays),
        "column_names": list(design.column_names),
    }


def _encode_term(term: FittedTerm, arrays: Arrays) -> TermRecord:  # noqa: C901, PLR0911
    if type(term) not in _TERMS:
        message = "Custom fitted terms have no allowlisted prediction codec."
        raise ArtifactError(message)
    if isinstance(term, FittedIntercept):
        return {"kind": "intercept"}
    if isinstance(term, FittedLinear):
        return {"kind": "linear", "column": term.column}
    if isinstance(term, FittedOrdination):
        return {
            "kind": "ordination",
            "mapping": encode_coordinate_map(term.mapping, arrays),
            "column_names": list(term.column_names),
        }
    if isinstance(term, CyclicSpline):
        return {
            "kind": "cyclic_spline",
            "column": term.column,
            "df": term.df,
            "period": float(term.period),
        }
    if isinstance(term, Fourier):
        return {
            "kind": "fourier",
            "column": term.column,
            "harmonics": term.harmonics,
            "period": float(term.period),
        }
    if isinstance(term, FittedSpline):
        return {
            "kind": "spline",
            "column": term.spec.column,
            "df": term.spec.df,
            "degree": term.spec.degree,
            "lower": term.lower,
            "upper": term.upper,
        }
    if isinstance(term, FittedCategorical):
        return {
            "kind": "categorical",
            "column": term.column,
            "levels": list(term.levels),
            "reference": term.reference,
        }
    if isinstance(term, FittedSumTerm):
        return {
            "kind": "sum",
            "terms": [_encode_term(child, arrays) for child in term.terms],
        }
    if isinstance(term, FittedInteraction):
        return {
            "kind": "interaction",
            "left": _encode_term(term.left, arrays),
            "right": _encode_term(term.right, arrays),
        }
    if not isinstance(term, FittedDistanceToLandmarks):
        message = "Unsupported fitted-term codec."
        raise ArtifactError(message)
    return {
        "kind": "distance_to_landmarks",
        "column": term.column,
        "geometry": encode_geometry(term.geometry, arrays),
    }


def _decode_term_record(value: object) -> TermRecord:  # noqa: C901, PLR0911, PLR0912 -- closed wire variants
    if not isinstance(value, dict):
        message = "Fitted term must be a declarative record."
        raise ArtifactError(message)
    kind = value.get("kind")
    if kind == "ordination":
        saved = record(value, {"kind", "mapping", "column_names"}, "ordination term")
        names = [
            text(name, "coordinate name")
            for name in items(saved["column_names"], "coordinate names")
        ]
        if not names or len(set(names)) != len(names):
            message = "An ordination term requires unique fixed coordinate names."
            raise ArtifactError(message)
        return {
            "kind": "ordination",
            "mapping": decode_coordinate_record(saved["mapping"]),
            "column_names": names,
        }
    if kind == "intercept":
        record(value, {"kind"}, "intercept")
        return {"kind": "intercept"}
    if kind == "linear":
        saved = record(value, {"kind", "column"}, "linear term")
        return {"kind": "linear", "column": text(saved["column"], "column")}
    if kind == "cyclic_spline":
        saved = record(value, {"kind", "column", "df", "period"}, "cyclic spline")
        return {
            "kind": "cyclic_spline",
            "column": text(saved["column"], "column"),
            "df": integer(saved["df"], "df"),
            "period": number(saved["period"], "period"),
        }
    if kind == "fourier":
        saved = record(value, {"kind", "column", "harmonics", "period"}, "Fourier")
        return {
            "kind": "fourier",
            "column": text(saved["column"], "column"),
            "harmonics": integer(saved["harmonics"], "harmonics"),
            "period": number(saved["period"], "period"),
        }
    if kind == "spline":
        saved = record(
            value, {"kind", "column", "df", "degree", "lower", "upper"}, "spline"
        )
        return {
            "kind": "spline",
            "column": text(saved["column"], "column"),
            "df": integer(saved["df"], "df"),
            "degree": integer(saved["degree"], "degree"),
            "lower": number(saved["lower"], "lower endpoint"),
            "upper": number(saved["upper"], "upper endpoint"),
        }
    if kind == "categorical":
        saved = record(
            value, {"kind", "column", "levels", "reference"}, "categorical term"
        )
        levels = labels(saved["levels"], "categorical levels")
        reference = saved["reference"]
        if (
            not levels
            or isinstance(reference, bool)
            or not isinstance(reference, (str, int))
            or reference not in levels
        ):
            message = (
                "Persisted categorical reference must belong to its nonempty levels."
            )
            raise ArtifactError(message)
        return {
            "kind": "categorical",
            "column": text(saved["column"], "column"),
            "levels": list(levels),
            "reference": reference,
        }
    if kind == "sum":
        saved = record(value, {"kind", "terms"}, "sum term")
        return {
            "kind": "sum",
            "terms": [
                _decode_term_record(child)
                for child in items(saved["terms"], "sum terms")
            ],
        }
    if kind == "interaction":
        saved = record(value, {"kind", "left", "right"}, "interaction term")
        return {
            "kind": "interaction",
            "left": _decode_term_record(saved["left"]),
            "right": _decode_term_record(saved["right"]),
        }
    if kind == "distance_to_landmarks":
        saved = record(value, {"kind", "column", "geometry"}, "landmark term")
        return {
            "kind": "distance_to_landmarks",
            "column": text(saved["column"], "column"),
            "geometry": decode_geometry_record(saved["geometry"]),
        }
    message = f"Unknown fitted term kind {kind!r}; executable classes are unsupported."
    raise ArtifactError(message)


def _restore_ordination(saved: OrdinationTerm, arrays: Arrays) -> FittedOrdination:
    mapping = decode_coordinate_map(saved["mapping"], arrays)
    if isinstance(mapping, LinearCoordinateMap):
        width = mapping.projection.components.shape[1]
    elif isinstance(mapping, KernelCoordinateMap):
        width = mapping.projection.axes.shape[1]
    elif isinstance(mapping, GeodesicCoordinateMap):
        width = mapping.distance_map.projection.axes.shape[1]
    else:
        width = mapping.projection.axes.shape[1]
    if width != len(saved["column_names"]) or not mapping.input_columns:
        message = (
            "Ordination term names or source roles disagree with its retained map."
        )
        raise ArtifactError(message)
    return FittedOrdination(mapping, tuple(saved["column_names"]))


def _restore_term(saved: TermRecord, arrays: Arrays) -> FittedTerm:  # noqa: PLR0911
    if saved["kind"] == "ordination":
        return _restore_ordination(saved, arrays)
    if saved["kind"] == "intercept":
        return FittedIntercept()
    if saved["kind"] == "linear":
        return FittedLinear(saved["column"])
    if saved["kind"] == "cyclic_spline":
        return CyclicSpline(saved["column"], saved["df"], saved["period"])
    if saved["kind"] == "fourier":
        return Fourier(saved["column"], saved["harmonics"], saved["period"])
    if saved["kind"] == "spline":
        return FittedSpline(
            Spline(saved["column"], saved["df"], saved["degree"]),
            saved["lower"],
            saved["upper"],
        )
    if saved["kind"] == "categorical":
        return FittedCategorical(
            saved["column"], tuple(saved["levels"]), saved["reference"]
        )
    if saved["kind"] == "sum":
        return FittedSumTerm(
            tuple(_restore_term(child, arrays) for child in saved["terms"])
        )
    if saved["kind"] == "interaction":
        return FittedInteraction(
            _restore_term(saved["left"], arrays), _restore_term(saved["right"], arrays)
        )
    return FittedDistanceToLandmarks(
        saved["column"], decode_geometry(saved["geometry"], arrays)
    )


def decode_design_record(value: object) -> DesignRecord:
    """Decode the recursive fitted expression and required feature-name list."""
    saved = record(value, {"terms", "column_names"}, "fitted design")
    return {
        "terms": _decode_term_record(saved["terms"]),
        "column_names": [
            text(name, "feature name")
            for name in items(saved["column_names"], "feature names")
        ],
    }


def decode_design(saved: DesignRecord, arrays: Arrays) -> FittedDesign:
    """Restore a decoded design and require it to reproduce its saved names.

    Wire input is parsed once by :func:`decode_design_record`.
    """
    design = FittedDesign(_restore_term(saved["terms"], arrays))
    names = saved["column_names"]
    if tuple(names) != design.column_names:
        message = (
            "Persisted design column names disagree with its fitted transformations."
        )
        raise ArtifactError(message)
    return design
