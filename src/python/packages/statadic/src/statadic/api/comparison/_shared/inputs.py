"""Aligned geometry and group labels for an observed comparison."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from statadic.api.data import Observations
from statadic.api.design import level_codes
from statadic.api.embedding import RowIds  # noqa: TC001 -- runtime annotations
from statadic.api.representation import (
    ObservedDissimilarity,  # noqa: TC001 -- checked dataclass annotations
    PairwiseGeometry,
    PreparedGeometry,
)
from statadic.core.typing import Array, ArrayLike  # noqa: TC001 -- runtime annotations

_MIN_GROUPS = 2
type GroupLabels = Sequence[int | str] | ArrayLike | Observations


def _grouping(
    groups: GroupLabels,
    row_ids: RowIds,
    *,
    grouping: str | None,
    group_ids: RowIds | None,
) -> tuple[Array, tuple[int | str, ...]]:
    """Encode labels stably and reject identity changes before computing statistics."""
    if group_ids is not None and tuple(group_ids) != row_ids:
        message = "Group IDs must exactly match geometry row order."
        raise ValueError(message)
    if isinstance(groups, Observations):
        if grouping is None:
            message = "Specify a categorical grouping column for Observations."
            raise ValueError(message)
        if groups.row_ids != row_ids:
            message = "Grouping observations must exactly match geometry row IDs."
            raise ValueError(message)
        labels = groups.categorical(grouping)
    else:
        if group_ids is None:
            message = (
                "Group arrays require group_ids exactly matching geometry row order."
            )
            raise ValueError(message)
        raw: object = (
            list(groups)
            if isinstance(groups, Sequence)
            else getattr(groups, "tolist", lambda: None)()
        )
        if not isinstance(raw, list) or len(raw) != len(row_ids):
            message = "Group labels must have one entry per outer observation."
            raise ValueError(message)
        labels = tuple(raw)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, str)) for value in labels
    ):
        message = "Groups must be nonmissing integer or string labels."
        raise ValueError(message)
    levels, codes = level_codes(labels)
    if not _MIN_GROUPS <= len(levels) < len(row_ids):
        message = (
            "Require at least two groups and positive residual degrees of freedom."
        )
        raise ValueError(message)
    return codes, levels


def _geometry(value: PairwiseGeometry | PreparedGeometry) -> PairwiseGeometry:
    """Consume actual construction evidence rather than interpreting type arguments."""
    geometry = value.pairwise() if isinstance(value, PreparedGeometry) else value
    if not isinstance(geometry, PairwiseGeometry):
        message = "Comparison requires prepared empirical-measure geometry."
        raise TypeError(message)
    return geometry


@dataclass(frozen=True)
class ComparisonInput:
    """Geometry and aligned group labels for an observed comparison."""

    geometry: PairwiseGeometry | PreparedGeometry | ObservedDissimilarity
    groups: GroupLabels
    group_ids: RowIds | None = None


@dataclass(frozen=True)
class PreparedComparison:
    """Inputs retained for one comparison specification."""

    geometry: PairwiseGeometry | ObservedDissimilarity
    groups: GroupLabels
    group_ids: RowIds | None
    model_token: object
