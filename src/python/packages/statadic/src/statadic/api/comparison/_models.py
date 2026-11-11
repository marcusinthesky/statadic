"""Group comparison specifications."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, override

import jax.numpy as jnp

from statadic.api.base import Model
from statadic.api.comparison._results import GroupComparisonResult
from statadic.api.comparison._shared import (
    _MIN_GROUPS,
    ComparisonInput,
    GroupLabels,  # noqa: TC001
    PreparedComparison,
    RowIds,  # noqa: TC001
    _geometry,
    _grouping,
)
from statadic.api.data import Observations
from statadic.api.ordination import (
    DistanceGram,
    Ordination,
    OrdinationSpectrum,  # noqa: TC001
    PrincipalCoordinates,
)
from statadic.api.representation import (  # noqa: TC001
    ObservedDissimilarity,
    PairwiseGeometry,
    PreparedGeometry,
)
from statadic.association.dispersion import (
    dispersion_radii,
    radius_anova_f,
)
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.geometry.median import spatial_medians


def _kept(groups: GroupLabels) -> Observations | None:
    """Retain grouping Observations for name resolution at inference."""
    return groups if isinstance(groups, Observations) else None


@dataclass(frozen=True)
class _GroupComparison(
    Model[ComparisonInput, PreparedComparison, "GroupComparisonResult"]
):
    """Shared aligned preparation and model-owned fitting lifecycle."""

    grouping: str | None = None

    @override
    def prepare(self, data: ComparisonInput) -> PreparedComparison:
        """Validate and snapshot geometry and aligned group labels."""
        geometry = (
            data.geometry
            if isinstance(data.geometry, ObservedDissimilarity)
            else _geometry(data.geometry)
        )
        codes, levels = _grouping(
            data.groups,
            geometry.row_ids,
            grouping=self.grouping,
            group_ids=data.group_ids,
        )
        # Observations are immutable; keep them so inference can resolve
        # design columns (assignment units, blocks) by name.
        groups = (
            data.groups
            if isinstance(data.groups, Observations)
            else tuple(levels[int(code)] for code in codes)
        )
        return PreparedComparison(geometry, groups, geometry.row_ids, self._token)

    @override
    def _estimate_prepared(self, prepared: PreparedComparison) -> GroupComparisonResult:
        return self._estimate(
            prepared.geometry, groups=prepared.groups, group_ids=prepared.group_ids
        )

    @abstractmethod
    def _estimate(
        self,
        geometry: PairwiseGeometry | PreparedGeometry | ObservedDissimilarity,
        *,
        groups: GroupLabels,
        group_ids: RowIds | None = None,
    ) -> GroupComparisonResult:
        """Estimate this method's statistic on aligned inputs."""
        ...


@dataclass(frozen=True)
class PERMDISP(_GroupComparison):
    """Dispersion about converged group spatial medians (Anderson 2006).

    ``bias_adjust`` scales radii by ``sqrt(n_g / (n_g - 1))`` as in vegan's
    ``betadisper(bias.adjust = TRUE)``; it then requires two points per group.
    """

    center: Literal["median"] = "median"
    bias_adjust: bool = False

    @override
    def _estimate(
        self,
        geometry: PairwiseGeometry | PreparedGeometry | ObservedDissimilarity,
        *,
        groups: GroupLabels,
        group_ids: RowIds | None = None,
    ) -> GroupComparisonResult:
        """Retain the strict full-rank embedding and the existing median statistic."""
        if isinstance(geometry, ObservedDissimilarity):
            return self._observed_result(
                geometry, groups=groups, group_ids=group_ids, tolerance=None
            )
        if self.center != "median":
            message = "The initial PERMDISP workflow supports spatial medians only."
            raise ValueError(message)
        pairwise = _geometry(geometry)
        codes, levels = _grouping(
            groups, pairwise.row_ids, grouping=self.grouping, group_ids=group_ids
        )
        count = len(levels)
        if (
            self.bias_adjust
            and int(jnp.min(jnp.bincount(codes, length=count))) < _MIN_GROUPS
        ):
            message = "Bias-adjusted PERMDISP requires two observations per group."
            raise ValueError(message)
        if not pairwise.squared_negative_type:
            message = "PERMDISP requires qualified squared-negative-type geometry."
            raise ValueError(message)
        coordinates = (
            Ordination(DistanceGram(), PrincipalCoordinates(None))
            .estimate(pairwise)
            .coordinates
        )
        return self._radii_result(
            pairwise, codes, levels, coordinates, None, _kept(groups)
        )

    def from_distances(
        self,
        distances: ArrayLike,
        *,
        groups: GroupLabels,
        row_ids: RowIds,
        group_ids: RowIds | None = None,
        tolerance: float | None = None,
    ) -> GroupComparisonResult:
        """Run PERMDISP on one observed dissimilarity via signed-spectrum PCoA.

        Coordinates are the positive axes of the supplied matrix. When the
        spectrum has material negative inertia the result is withheld with
        ``reason="material_negative_inertia"`` instead of silently truncating
        the geometry. ``tolerance`` is the relative eigenvalue cut of
        :class:`statadic.api.ordination.PrincipalCoordinates`.
        """
        observed = ObservedDissimilarity.from_array(distances, row_ids)
        return self._observed_result(
            observed, groups=groups, group_ids=group_ids, tolerance=tolerance
        )

    def _observed_result(
        self,
        observed: ObservedDissimilarity,
        *,
        groups: GroupLabels,
        group_ids: RowIds | None,
        tolerance: float | None,
    ) -> GroupComparisonResult:
        """Preserve the observed-matrix spectrum and withholding policy."""
        if self.center != "median":
            message = "The initial PERMDISP workflow supports spatial medians only."
            raise ValueError(message)
        codes, levels = _grouping(
            groups, observed.row_ids, grouping=self.grouping, group_ids=group_ids
        )
        count = len(levels)
        if (
            self.bias_adjust
            and int(jnp.min(jnp.bincount(codes, length=count))) < _MIN_GROUPS
        ):
            message = "Bias-adjusted PERMDISP requires two observations per group."
            raise ValueError(message)
        ordination = Ordination(
            DistanceGram(),
            PrincipalCoordinates(None, "positive-subspace", tolerance),
        ).estimate(observed)
        spectrum = ordination.spectrum
        if spectrum.material_negative_inertia:
            return GroupComparisonResult(
                spectrum,
                observed,
                "PERMDISP",
                codes,
                levels,
                None,
                None,
                jnp.asarray(jnp.nan, dtype=observed.values.dtype),
                median_converged=None,
                bias_adjust=self.bias_adjust,
                reason="material_negative_inertia",
                spectrum=spectrum,
                groups=_kept(groups),
            )
        return self._radii_result(
            observed, codes, levels, ordination.coordinates, spectrum, _kept(groups)
        )

    def _radii_result(
        self,
        geometry: PairwiseGeometry | ObservedDissimilarity,
        codes: Array,
        levels: tuple[int | str, ...],
        coordinates: Array,
        spectrum: OrdinationSpectrum | None,
        groups: Observations | None,
    ) -> GroupComparisonResult:
        """Fit medians and radii once; shared by the geometry and observed doors."""
        count = len(levels)
        medians, converged = spatial_medians(coordinates, codes, count)
        radii = dispersion_radii(
            coordinates, codes, medians, bias_adjust=self.bias_adjust
        )
        median_converged = bool(jnp.all(converged))
        statistic = (
            radius_anova_f(radii, codes, count)
            if median_converged
            else jnp.asarray(jnp.nan, dtype=radii.dtype)
        )
        return GroupComparisonResult(
            radii,
            geometry,
            "PERMDISP",
            codes,
            levels,
            None,
            coordinates,
            statistic,
            dispersion_radii=radii,
            median_converged=median_converged,
            bias_adjust=self.bias_adjust,
            reason=None if median_converged else "spatial_median_not_converged",
            spectrum=spectrum,
            groups=groups,
        )
