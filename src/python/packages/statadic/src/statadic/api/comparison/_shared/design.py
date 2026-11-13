"""The relabelling group a permutation plan declares, and its resolution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from statadic.api.data import (  # noqa: TC001 -- runtime annotations
    Observations,
    _label_values,
)
from statadic.api.embedding import RowIds  # noqa: TC001 -- runtime annotations
from statadic.core.typing import Array, ArrayLike  # noqa: TC001 -- runtime annotations

type DesignLabels = str | Sequence[int | str] | ArrayLike


@dataclass(frozen=True)
class ExchangeabilityDesign:
    """The resolved relabelling group a permutation plan draws from.

    Rows map to assignment units and units to exchangeability blocks. Under the
    free design every row is its own unit and there is one block.
    """

    unit: str | None
    blocks: str | None
    row_units: tuple[int, ...]
    unit_blocks: tuple[int, ...]
    unit_ids: tuple[int | str, ...]
    block_ids: tuple[int | str, ...]

    @property
    def free(self) -> bool:
        """Whether this is unrestricted relabelling of rows."""
        return self.unit is None and self.blocks is None

    def units_in(self, block: int) -> tuple[int, ...]:
        """Return the unit indices of one block, in unit order."""
        return tuple(u for u, b in enumerate(self.unit_blocks) if b == block)

    def metadata(self, arrangements: int) -> dict[str, int | str | None]:
        """Describe the design for reports."""
        return {
            "unit": self.unit,
            "blocks": self.blocks,
            "n_units": len(self.unit_ids),
            "n_blocks": len(self.block_ids),
            "arrangements": arrangements,
        }

    def description(self) -> str:
        """Name the declared relabelling group in words."""
        units = "rows" if self.unit is None else f"whole units ({self.unit})"
        within = "" if self.blocks is None else f" within blocks ({self.blocks})"
        return f"{units}{within}"


class PlannedDraws(NamedTuple):
    """Draws to evaluate, the design they came from and its group size."""

    draws: Array
    design: ExchangeabilityDesign
    admissible: int


def _role_name(value: DesignLabels | None) -> str | None:
    """Name a design role in reports: its column, or that labels were given."""
    if value is None:
        return None
    return value if isinstance(value, str) else "supplied labels"


def _design_labels(
    role: str,
    value: DesignLabels,
    row_ids: RowIds,
    plan_row_ids: RowIds | None,
    observations: Observations | None,
) -> tuple[int | str, ...]:
    """Resolve one design role by column name or aligned explicit labels."""
    if isinstance(value, str):
        if observations is None:
            message = (
                f"PermutationPlan {role}={value!r} names a column, but this result "
                "keeps no grouping Observations; pass labels aligned with row_ids."
            )
            raise ValueError(message)
        labels = tuple(observations.resolve_grouping(value))
    else:
        if plan_row_ids is None:
            message = f"Explicit PermutationPlan {role} labels require row_ids."
            raise ValueError(message)
        if not isinstance(value, Sequence):
            message = f"PermutationPlan {role} labels must be a sequence."
            raise ValueError(message)  # noqa: TRY004 -- invalid design specification
        labels = value
        if len(labels) != len(row_ids):
            message = f"PermutationPlan {role} needs one label per observation."
            raise ValueError(message)
    try:
        return _label_values(labels, f"PermutationPlan {role}")
    except TypeError as error:
        message = f"PermutationPlan {role} labels must be integers or strings."
        raise ValueError(message) from error
