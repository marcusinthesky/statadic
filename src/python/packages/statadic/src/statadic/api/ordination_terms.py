"""Frozen ordination maps shared by workflow results, designs and artifacts.

This owner has no dependency on model constructors, reports or persistence.
Numeric ``project_*`` methods apply retained arrays without host validation;
``transform`` checks named host inputs and never fits a new representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from statadic.api.data import Observations
from statadic.api.representation import (
    PreparedGeometry,  # noqa: TC001 -- runtime annotations
)
from statadic.api.roles import InputRole  # noqa: TC001
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.geometry.embedding import (  # noqa: TC001 -- runtime annotations
    CrossGramMap,
    OrdinationFit,
)
from statadic.geometry.ground.config import (
    GroundDistanceSelection,  # noqa: TC001 -- runtime annotations
    resolve_ground_distance,
)
from statadic.geometry.ground.metrics import cdist
from statadic.geometry.ground.similarities import (
    PositiveSemidefiniteKernelLaw,  # noqa: TC001 -- runtime annotations
    SimilarityStrategy,  # noqa: TC001 -- runtime annotations
    gram,
)
from statadic.geometry.shortest_paths import geodesic_cross_distances
from statadic.sample.packed import PackedClouds
from statadic.sample.reduction import Projection  # noqa: TC001 -- runtime annotations

__all__ = [
    "CoordinateMap",
    "DistanceCoordinateMap",
    "FeatureDistanceMap",
    "FittedOrdination",
    "GeodesicCoordinateMap",
    "KernelCoordinateMap",
    "LinearCoordinateMap",
    "OrdinationContext",
    "OrdinationEvidence",
    "OrdinationSpectrum",
    "coordinate_capabilities",
    "feature_values",
    "transform_coordinates",
    "transform_cross_distances",
]
_MATRIX_NDIM = 2


def feature_values(data: Observations | ArrayLike, names: tuple[str, ...]) -> Array:
    """Extract finite real feature rows in the retained named order."""
    values = (
        jnp.column_stack([data.numeric(name) for name in names])
        if isinstance(data, Observations)
        else jnp.array(data, copy=True)
    )
    if values.ndim != _MATRIX_NDIM or values.shape[1] != len(names) or not len(names):
        message = "Ordination features must match the retained named feature axis."
        raise ValueError(message)
    if not jnp.issubdtype(values.dtype, jnp.number) or jnp.iscomplexobj(values):
        message = "Ordination features must be real numeric values."
        raise TypeError(message)
    values = values.astype(jnp.result_type(values, 1.0))
    if not bool(jnp.all(jnp.isfinite(values))):
        message = "Ordination requires finite features; select missing rows explicitly."
        raise ValueError(message)
    return values


def _finite(values: Array) -> Array:
    if not bool(jnp.all(jnp.isfinite(values))):
        message = "Ordination transformation is not numerically representable."
        raise ValueError(message)
    return values


@dataclass(frozen=True)
class LinearCoordinateMap:
    """A feature projection with frozen training centering and sample scaling."""

    feature_names: tuple[str, ...]
    mean: Array
    scale: Array
    projection: Projection

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return required numeric feature roles."""
        return dict.fromkeys(self.feature_names, "numeric")

    def project_values(self, values: Array) -> Array:
        """Apply numerical frozen preparation on the JAX graph."""
        return self.projection.transform((values - self.mean) / self.scale)

    def transform(self, data: Observations | ArrayLike) -> Array:
        """Transform new rows using the training projection only."""
        return _finite(self.project_values(feature_values(data, self.feature_names)))


@dataclass(frozen=True)
class KernelCoordinateMap:
    """A training-centered kernel projection against frozen feature references."""

    feature_names: tuple[str, ...]
    reference: Array
    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw]
    projection: CrossGramMap

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return required numeric feature roles."""
        return dict.fromkeys(self.feature_names, "numeric")

    def project_values(self, values: Array) -> Array:
        """Evaluate the retained kernel and numerical cross-Gram map."""
        return self.projection.transform(
            gram(values, self.reference, kernel=self.kernel)
        )

    def transform(self, data: Observations | ArrayLike) -> Array:
        """Transform new rows without re-centering on their batch."""
        return _finite(self.project_values(feature_values(data, self.feature_names)))


@dataclass(frozen=True)
class FeatureDistanceMap:
    """Frozen feature references and a declared ground distance."""

    feature_names: tuple[str, ...]
    reference: Array
    ground: GroundDistanceSelection

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the required numeric source features."""
        return dict.fromkeys(self.feature_names, "numeric")

    def cross(self, data: Observations | ArrayLike) -> Array:
        """Evaluate new-by-training distances under the frozen ground."""
        return cdist(
            feature_values(data, self.feature_names),
            self.reference,
            metric=resolve_ground_distance(self.ground),
        )


@dataclass(frozen=True)
class DistanceCoordinateMap:
    """Fixed-axis distance extension with optional frozen measure references."""

    row_ids: tuple[int | str, ...]
    projection: CrossGramMap
    geometry: PreparedGeometry | None
    measure_column: str | None = None

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return a measure role only when a named measure column is bound."""
        return {} if self.measure_column is None else {self.measure_column: "measure"}

    def project_distances(self, distances: Array) -> Array:
        """Project numerical cross distances on the retained positive axes."""
        distances = distances.astype(jnp.result_type(distances, 1.0))
        return self.projection.transform(-0.5 * distances**2)

    def transform(self, data: Observations | PackedClouds) -> Array:
        """Evaluate frozen measure geometry, requiring its actual source."""
        if self.geometry is None:
            message = (
                "Observed matrix ordination requires explicit cross distances "
                "and reference IDs."
            )
            raise ValueError(message)
        if isinstance(data, Observations):
            if self.measure_column is None:
                message = "Bind a named measure column for table transformation."
                raise ValueError(message)
            data = data.measure(self.measure_column)
        return _finite(self.project_distances(self.geometry.cross(data)))


@dataclass(frozen=True)
class GeodesicCoordinateMap:
    """Independent query attachments to a frozen training-only graph."""

    distance_map: DistanceCoordinateMap
    training_paths: Array
    neighbors: int
    tie_codes: Array
    feature_distances: FeatureDistanceMap | None = None

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the actual source's named roles."""
        return (
            self.distance_map.input_columns
            if self.feature_distances is None
            else self.feature_distances.input_columns
        )

    def project_distances(self, distances: Array) -> Array:
        """Attach independent numerical queries and apply the frozen axes."""
        paths = geodesic_cross_distances(
            distances,
            self.training_paths,
            neighbors=self.neighbors,
            tie_codes=self.tie_codes,
        )
        return self.distance_map.project_distances(paths)

    def transform(self, data: Observations | ArrayLike | PackedClouds) -> Array:
        """Evaluate original query distances without rebuilding training edges."""
        if self.feature_distances is not None:
            if isinstance(data, PackedClouds):
                message = "Feature graph transformation requires real feature rows."
                raise TypeError(message)
            values = self.feature_distances.cross(data)
        else:
            geometry = self.distance_map.geometry
            if geometry is None:
                message = (
                    "Observed graph ordination requires explicit cross distances "
                    "and reference IDs."
                )
                raise ValueError(message)
            if isinstance(data, Observations):
                column = self.distance_map.measure_column
                if column is None:
                    message = "Bind a named measure column for table transformation."
                    raise ValueError(message)
                data = data.measure(column)
            if not isinstance(data, PackedClouds):
                message = "Measure graph transformation requires empirical measures."
                raise TypeError(message)
            values = geometry.cross(data)
        return _finite(self.project_distances(values))


type CoordinateMap = (
    LinearCoordinateMap
    | KernelCoordinateMap
    | DistanceCoordinateMap
    | GeodesicCoordinateMap
)


@dataclass(frozen=True)
class OrdinationContext:
    """Training identities and construction evidence independent of the spectrum."""

    row_ids: tuple[int | str, ...]
    origin: str
    construction_warrant: str | None
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class OrdinationSpectrum:
    """Full signed centered-operator spectrum, in Gram rather than variance units."""

    eigenvalues: Array
    threshold: float
    n_positive: int
    n_negative: int

    @property
    def material_negative_inertia(self) -> bool:
        """Whether the observed operator has a materially negative eigenvalue."""
        return self.n_negative > 0


@dataclass(frozen=True)
class OrdinationEvidence:
    """Actual fitted coordinates, frozen map and training construction evidence.

    This host snapshot retains array references, not a second decomposition.
    Prediction artifacts may retain only the map and do not restore this state.
    """

    name: str
    fit: OrdinationFit
    mapping: CoordinateMap
    context: OrdinationContext

    @property
    def spectrum(self) -> OrdinationSpectrum:
        """Inspect the complete signed spectrum without attaching a population law."""
        threshold = float(self.fit.threshold)
        return OrdinationSpectrum(
            self.fit.eigenvalues,
            threshold,
            int(jnp.sum(self.fit.eigenvalues > threshold)),
            int(jnp.sum(self.fit.eigenvalues < -threshold)),
        )

    @property
    def loadings(self) -> Array:
        """Return feature loadings only for a linear feature decomposition."""
        if not isinstance(self.mapping, LinearCoordinateMap):
            message = "Feature loadings are available only for linear feature geometry."
            raise TypeError(message)
        return self.mapping.projection.components

    def _selected_variance(self) -> Array:
        if not isinstance(self.mapping, LinearCoordinateMap):
            message = "Feature variance is available only for linear feature geometry."
            raise TypeError(message)
        return self.fit.eigenvalues[: self.fit.coordinates.shape[1]]

    @property
    def explained_variance(self) -> Array:
        """Return sample variances of the selected linear feature axes."""
        return self._selected_variance() / (len(self.context.row_ids) - 1)

    @property
    def explained_variance_ratio(self) -> Array:
        """Return selected-axis fractions using the complete feature variance."""
        selected = self._selected_variance()
        total = jnp.sum(jnp.maximum(self.fit.eigenvalues, 0))
        return jnp.where(total > 0, selected / jnp.where(total > 0, total, 1), jnp.nan)


def coordinate_capabilities(mapping: CoordinateMap) -> tuple[str, ...]:
    """Describe only operations supplied by the actual frozen query source."""
    if isinstance(mapping, (LinearCoordinateMap, KernelCoordinateMap)):
        return ("transform",)
    base = (
        mapping.distance_map if isinstance(mapping, GeodesicCoordinateMap) else mapping
    )
    automatic = base.geometry is not None or (
        isinstance(mapping, GeodesicCoordinateMap)
        and mapping.feature_distances is not None
    )
    return (
        ("transform", "transform_distances") if automatic else ("transform_distances",)
    )


def transform_coordinates(
    mapping: CoordinateMap, data: Observations | ArrayLike | PackedClouds
) -> Array:
    """Apply a frozen source-specific host transformation without new fitting."""
    if isinstance(mapping, DistanceCoordinateMap):
        if not isinstance(data, (Observations, PackedClouds)):
            message = "Distance transformation requires measures or named observations."
            raise TypeError(message)
        return mapping.transform(data)
    if isinstance(mapping, GeodesicCoordinateMap):
        return mapping.transform(data)
    if isinstance(data, PackedClouds):
        message = "Feature transformation requires a real feature matrix."
        raise TypeError(message)
    return mapping.transform(data)


def transform_cross_distances(
    mapping: CoordinateMap,
    distances: ArrayLike,
    *,
    reference_ids: tuple[int | str, ...],
) -> Array:
    """Check exact reference identities and apply a retained distance extension."""
    if not isinstance(mapping, (DistanceCoordinateMap, GeodesicCoordinateMap)):
        message = "This ordination does not consume cross distances."
        raise TypeError(message)
    base = (
        mapping.distance_map if isinstance(mapping, GeodesicCoordinateMap) else mapping
    )
    values = jnp.asarray(distances)
    if (
        tuple(reference_ids) != base.row_ids
        or values.ndim != _MATRIX_NDIM
        or values.shape[1] != len(base.row_ids)
    ):
        message = "Cross-distance reference IDs and columns must match training order."
        raise ValueError(message)
    if jnp.iscomplexobj(values) or not bool(
        jnp.all(jnp.isfinite(values) & (values >= 0))
    ):
        message = "Cross distances must be finite nonnegative real values."
        raise ValueError(message)
    return _finite(mapping.project_distances(values))


@dataclass(frozen=True)
class FittedOrdination:
    """A fitted design term retaining the exact training coordinate map."""

    mapping: CoordinateMap
    column_names: tuple[str, ...]
    evidence: OrdinationEvidence | None = None

    def __post_init__(self) -> None:
        """Keep optional live inspection bound to the same retained map and axes."""
        if self.evidence is not None and (
            self.evidence.mapping is not self.mapping
            or self.evidence.fit.coordinates.shape[1] != len(self.column_names)
        ):
            message = "Ordination inspection must match the retained map and columns."
            raise ValueError(message)

    @property
    def inspection(self) -> OrdinationEvidence:
        """Return live fitting evidence, absent from predictive-only artifacts."""
        if self.evidence is None:
            message = "This predictive-only term does not retain ordination inspection."
            raise ValueError(message)
        return self.evidence

    @property
    def spectrum(self) -> OrdinationSpectrum:
        """Return the retained full signed spectrum without fitting again."""
        return self.inspection.spectrum

    @property
    def loadings(self) -> Array:
        """Return retained linear feature loadings when supported."""
        return self.inspection.loadings

    @property
    def explained_variance(self) -> Array:
        """Return retained selected linear-axis sample variances."""
        return self.inspection.explained_variance

    @property
    def explained_variance_ratio(self) -> Array:
        """Return retained selected linear-axis fractions of complete variance."""
        return self.inspection.explained_variance_ratio

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Return the source roles retained by the fitted map."""
        return self.mapping.input_columns

    def transform(self, data: Observations) -> Array:
        """Produce design columns without learning from prediction rows."""
        return self.mapping.transform(data)
