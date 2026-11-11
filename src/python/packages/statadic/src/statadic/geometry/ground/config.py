# jaxtyping shape strings are runtime contracts.
"""Serialized ground-distance selection and eager compatibility parsing.

Strings and ``TypedDict`` records stop here. Numerical kernels receive only a
:class:`~statadic.geometry.ground.metrics.GroundDistance`, so dynamic configuration loses
precision once, at an explicit boundary, instead of at every ``cdist`` call.
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from typing import Literal, TypedDict, TypeIs, overload

from statadic.core.axioms import Law
from statadic.geometry.ground._strategy import (
    GroundDistance,
)
from statadic.geometry.ground.metrics import (
    ANGULAR,
    COSINE,
    EUCLIDEAN,
    NORMALIZED_EUCLIDEAN,
    AngularLaw,
    CosineLaw,
    EuclideanLaw,
    NormalizedEuclideanLaw,
)

__all__ = [
    "AngularDistanceConfig",
    "BuiltinGroundDistance",
    "CosineDistanceConfig",
    "EuclideanDistanceConfig",
    "GroundDistanceConfig",
    "GroundDistanceName",
    "GroundDistanceSelection",
    "InvalidGroundDistanceConfigError",
    "NormalizedEuclideanDistanceConfig",
    "is_euclidean",
    "parse_ground_distance",
    "parse_ground_distance_name",
    "resolve_ground_distance",
]

type GroundDistanceName = Literal[
    "euclidean",
    "angular",
    "cosine",
    "normalized_euclidean",
]
"""Closed serialized names for shipped ground strategies."""


class InvalidGroundDistanceConfigError(ValueError):
    """Raised when a serialized ground-distance record has invalid structure."""

    def __init__(self, reason: str) -> None:
        """Describe the rejected mapping without exposing implementation detail."""
        super().__init__(f"invalid ground-distance config: {reason}")


class EuclideanDistanceConfig(TypedDict):
    """Serialized Euclidean strategy selection."""

    kind: Literal["euclidean"]


class AngularDistanceConfig(TypedDict):
    """Serialized angular strategy selection."""

    kind: Literal["angular"]


class CosineDistanceConfig(TypedDict):
    """Serialized cosine strategy selection."""

    kind: Literal["cosine"]


class NormalizedEuclideanDistanceConfig(TypedDict):
    """Serialized normalized-Euclidean strategy selection."""

    kind: Literal["normalized_euclidean"]


type GroundDistanceConfig = (
    EuclideanDistanceConfig
    | AngularDistanceConfig
    | CosineDistanceConfig
    | NormalizedEuclideanDistanceConfig
)
"""Discriminated serialized ground-distance configuration."""

type BuiltinGroundDistance = (
    GroundDistance[EuclideanLaw]
    | GroundDistance[AngularLaw]
    | GroundDistance[CosineLaw]
    | GroundDistance[NormalizedEuclideanLaw]
)
"""Narrowable union of shipped typed strategies."""

type GroundDistanceSelection = (
    GroundDistance[Law] | GroundDistanceName | GroundDistanceConfig
)
"""Declared strategy or closed serialized selection accepted by eager APIs."""


def parse_ground_distance_name(name: str) -> BuiltinGroundDistance:
    """Validate a serialized built-in name and return its singleton strategy.

    Args:
        name: Dynamic serialized name, validated at this boundary.

    Returns:
        The corresponding immutable strategy singleton.

    Raises:
        ValueError: If the name does not identify a shipped strategy.

    """
    if name == "euclidean":
        return EUCLIDEAN
    if name == "angular":
        return ANGULAR
    if name == "cosine":
        return COSINE
    if name == "normalized_euclidean":
        return NORMALIZED_EUCLIDEAN
    message = (
        f"unknown ground distance {name!r}; expected euclidean, angular, cosine, "
        "or normalized_euclidean"
    )
    raise ValueError(message)


def parse_ground_distance(
    config: object,
) -> BuiltinGroundDistance:
    """Parse a discriminated config record into a typed strategy union.

    Args:
        config: Serialized record. It must contain exactly one ``kind`` field
            whose value is a shipped ground-distance name.

    Returns:
        The selected immutable strategy singleton.

    """
    if not isinstance(config, Mapping):
        message = f"expected a mapping, got {type(config).__name__}"
        raise InvalidGroundDistanceConfigError(message)
    keys = set(config)
    if keys != {"kind"}:
        rendered_keys = sorted(repr(key) for key in keys)
        message = f"expected exactly {{'kind'}}, got {rendered_keys!r}"
        raise InvalidGroundDistanceConfigError(message)
    kind = config["kind"]
    if not isinstance(kind, str):
        message = f"'kind' must be a string, got {type(kind).__name__}"
        raise InvalidGroundDistanceConfigError(message)
    try:
        return parse_ground_distance_name(kind)
    except ValueError as exc:
        message = f"unknown 'kind' {kind!r}"
        raise InvalidGroundDistanceConfigError(message) from exc


@overload
def resolve_ground_distance[GroundLawT: Law](
    selection: GroundDistance[GroundLawT],
) -> GroundDistance[GroundLawT]: ...


@overload
def resolve_ground_distance(
    selection: GroundDistanceName | GroundDistanceConfig,
) -> BuiltinGroundDistance: ...


@overload
def resolve_ground_distance(
    selection: Mapping[object, object],
) -> BuiltinGroundDistance: ...


def resolve_ground_distance(
    selection: (
        GroundDistance[Law]
        | GroundDistanceName
        | GroundDistanceConfig
        | Mapping[object, object]
    ),
) -> GroundDistance[Law] | BuiltinGroundDistance:
    """Resolve configuration once while preserving an existing strategy's law.

    Args:
        selection: A declared strategy or closed serialized name/config union.

    Returns:
        ``selection`` unchanged when already declared, otherwise its built-in
        singleton.

    """
    if isinstance(selection, str):
        return parse_ground_distance_name(selection)
    if isinstance(selection, Mapping):
        return parse_ground_distance(selection)
    if isinstance(selection, GroundDistance) and selection.backend == "jax":
        return selection
    message = (
        "ground distance must be a declared strategy or a validated built-in "
        "name/config; bare callables require declare_ground_distance"
    )
    raise TypeError(message)


def is_euclidean(
    distance: BuiltinGroundDistance,
) -> TypeIs[GroundDistance[EuclideanLaw]]:
    """Narrow a parsed built-in union to the exact Euclidean law.

    Args:
        distance: Parsed built-in strategy union.

    Returns:
        Whether ``distance`` is the Euclidean singleton.

    """
    return distance is EUCLIDEAN
