"""Column roles and graph-aligned numerical inputs for survival workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from statadic.api.data import SpatialPanelData
    from statadic.api.spatial.graphs import PanelData
    from statadic.optimize.integration import IntegrationDesign


@dataclass(frozen=True)
class RightCensored:
    """Column roles for elapsed duration, event indicator and optional entry."""

    time: str
    event: str
    entry: str | None = None


@dataclass(frozen=True)
class _SpatialSurvivalInput:
    """Distinct full graph nodes and observed outcomes with explicit integration."""

    data: SpatialPanelData
    integrations: tuple[IntegrationDesign, ...]


@dataclass(frozen=True)
class SpatialSurvivalData:
    """Named full graph panel and one explicit integration design per period."""

    panel: PanelData
    integrations: tuple[IntegrationDesign, ...]
