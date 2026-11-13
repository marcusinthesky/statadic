"""Aligned graph-node prediction values shared by spatial model families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from statadic.core.typing import Array


__all__ = ["SpatialPanelPrediction"]


@dataclass(frozen=True)
class SpatialPanelPrediction:
    """Full graph predictions in each declared period and node order."""

    period_ids: tuple[int, ...]
    unit_ids: tuple[tuple[int, ...], ...]
    values: tuple[Array, ...]
