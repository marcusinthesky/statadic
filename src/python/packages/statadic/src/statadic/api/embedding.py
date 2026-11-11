"""Scientific classical-scaling spectrum carriers.

Model specification belongs to :mod:`statadic.api.ordination`; observed matrices
belong to :mod:`statadic.api.representation`. This module has no alternative model
constructors or compatibility aliases.
"""

from __future__ import annotations

from dataclasses import dataclass

from statadic.core.typing import Array  # noqa: TC001 -- runtime annotations

__all__ = ["PCoASpectrum", "RowIds"]

type RowIds = tuple[int | str, ...]


@dataclass(frozen=True)
class PCoASpectrum:
    """Signed classical-scaling spectrum retained by historical scientific users.

    Coordinates contain the positive axes selected by the producer. The full
    signed spectrum and absolute threshold remain independent of coordinate
    truncation; a material negative eigenvalue precludes a Euclidean claim.
    """

    coordinates: Array
    eigenvalues: Array
    row_ids: RowIds
    threshold: float
    n_positive: int
    n_negative: int

    @property
    def material_negative_inertia(self) -> bool:
        """Whether any eigenvalue lies materially below zero."""
        return self.n_negative > 0
