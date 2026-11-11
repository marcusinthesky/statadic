"""Composable principal coordinates from feature, kernel and distance geometry.

Named methods emerge from the operator construction: feature factors use SVD,
kernel and distance Grams use symmetric eigenpairs, and graph paths change the
distance source. Geometry laws, observed spectra and inferential calibration
remain separate. Ordination is also a training-fitted regression term.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, override

import jax.numpy as jnp

from statadic.api.artifacts import save_ordination_predictor
from statadic.api.base import Model, Result
from statadic.api.data import MeasureColumn, Observations
from statadic.api.design import Term
from statadic.api.ordination_terms import (
    CoordinateMap,  # noqa: TC001 -- runtime public annotations
    DistanceCoordinateMap,
    FeatureDistanceMap,
    FittedOrdination,
    GeodesicCoordinateMap,
    KernelCoordinateMap,
    LinearCoordinateMap,
    OrdinationContext,
    OrdinationEvidence,
    OrdinationSpectrum,
    coordinate_capabilities,
    feature_values,
    transform_coordinates,
    transform_cross_distances,
)
from statadic.api.reporting import Metric, PlotSpec, Report
from statadic.api.representation import (
    GeometrySpec,
    ObservedDissimilarity,
    PairwiseGeometry,
    PreparedGeometry,
)
from statadic.api.roles import InputRole  # noqa: TC001
from statadic.core.typing import Array, ArrayLike  # noqa: TC001
from statadic.geometry.embedding import (
    CrossGramMap,
    NonEuclideanEmbeddingError,
    OrdinationFit,
    gram_coordinates,
)
from statadic.geometry.ground.config import (
    GroundDistanceSelection,  # noqa: TC001 -- runtime public annotations
    resolve_ground_distance,
)
from statadic.geometry.ground.metrics import cdist
from statadic.geometry.ground.similarities import (
    PositiveSemidefiniteKernelLaw,  # noqa: TC001 -- runtime public annotations
    RBFKernel,
    SimilarityStrategy,
    gram,
)
from statadic.geometry.shortest_paths import knn_shortest_paths
from statadic.sample.packed import PackedClouds
from statadic.sample.reduction import Projection, fit_projection

__all__ = [
    "DistanceGram",
    "KernelGram",
    "LinearGram",
    "Ordination",
    "OrdinationContext",
    "OrdinationResult",
    "OrdinationSpectrum",
    "PreparedOrdination",
    "PrincipalCoordinates",
    "ShortestPathDistances",
]

type _Input = (
    Observations
    | ArrayLike
    | PackedClouds
    | MeasureColumn
    | PreparedGeometry
    | PairwiseGeometry
    | ObservedDissimilarity
)
type _Geometry = LinearGram | KernelGram | DistanceGram
_MATRIX_NDIM = 2
_MIN_ROWS = 2


def _names(names: tuple[str, ...] | None) -> None:
    if names is not None and (
        not names
        or len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name for name in names)
    ):
        message = "Feature names must be nonempty unique strings."
        raise ValueError(message)


@dataclass(frozen=True)
class LinearGram:
    """Centered feature geometry, represented as a factor rather than dense Gram."""

    features: tuple[str, ...] | None = None
    scale: bool = False

    def __post_init__(self) -> None:
        """Freeze selected feature names and require an explicit scaling flag."""
        if self.features is not None:
            object.__setattr__(self, "features", tuple(self.features))
        _names(self.features)
        if not isinstance(self.scale, bool):
            message = "Feature scaling must be a bool."
            raise TypeError(message)


@dataclass(frozen=True)
class KernelGram:
    """Training-centered Gram construction from a declared PSD point kernel."""

    kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw]
    features: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Require a declared kernel and freeze the feature selection."""
        if not isinstance(self.kernel, SimilarityStrategy):
            message = "KernelGram requires a declared PSD SimilarityStrategy."
            raise TypeError(message)
        if isinstance(self.kernel, RBFKernel):
            object.__setattr__(
                self, "kernel", RBFKernel(jnp.array(self.kernel.bandwidth, copy=True))
            )
        if self.features is not None:
            object.__setattr__(self, "features", tuple(self.features))
        _names(self.features)


@dataclass(frozen=True)
class ShortestPathDistances:
    """Undirected union-kNN paths with original edge lengths and a fixed roster.

    Supply named ``features`` for ground-distance evaluation, or ``distances``
    for an empirical-measure geometry. Without either, input is an explicit
    observed/prepared distance carrier. Disconnected graphs are rejected.
    """

    neighbors: int
    features: tuple[str, ...] | None = None
    ground: GroundDistanceSelection = "euclidean"
    distances: GeometrySpec | None = None
    column: str | None = None

    def __post_init__(self) -> None:
        """Freeze graph choices without selecting a neighborhood from outcomes."""
        if type(self.neighbors) is not int or self.neighbors < 1:
            message = "neighbors must be a positive integer."
            raise ValueError(message)
        if self.features is not None:
            object.__setattr__(self, "features", tuple(self.features))
        _names(self.features)
        if self.features is not None and (
            self.distances is not None or self.column is not None
        ):
            message = "Choose feature or measure distances for a graph, not both."
            raise ValueError(message)
        object.__setattr__(self, "ground", resolve_ground_distance(self.ground))


@dataclass(frozen=True)
class DistanceGram:
    """Double-centered squared distances, with an optional distance producer."""

    distances: GeometrySpec | ShortestPathDistances | None = None
    column: str | None = None

    def __post_init__(self) -> None:
        """Require a concrete distance specification and unambiguous source."""
        if self.distances is not None and not isinstance(
            self.distances, (GeometrySpec, ShortestPathDistances)
        ):
            message = "DistanceGram requires a geometry or shortest-path construction."
            raise TypeError(message)
        if self.column is not None and (
            not self.column or not isinstance(self.column, str)
        ):
            message = "A measure column must have a nonempty name."
            raise ValueError(message)
        if (
            isinstance(self.distances, ShortestPathDistances)
            and self.column is not None
        ):
            message = "Bind the measure column inside ShortestPathDistances."
            raise ValueError(message)


@dataclass(frozen=True)
class PrincipalCoordinates:
    """Positive principal coordinates with an explicit indefinite approximation.

    ``tolerance`` is relative to the full spectrum's largest absolute value;
    the default is ``8 n eps``. ``positive-subspace`` retains signed evidence
    and reports approximation without asserting Euclidean embeddability.
    """

    dim: int | None = 2
    negative_inertia: Literal["raise", "positive-subspace"] = "raise"
    tolerance: float | None = None

    def __post_init__(self) -> None:
        """Validate dimension and the explicit spectral policy."""
        if self.dim is not None and (type(self.dim) is not int or self.dim < 1):
            message = "dim must be a positive integer or None for all positive axes."
            raise ValueError(message)
        if self.negative_inertia not in ("raise", "positive-subspace"):
            message = "Unknown negative-inertia policy."
            raise ValueError(message)
        if self.tolerance is not None and not 0 <= self.tolerance < 1:
            message = "Spectrum tolerance must be finite and in [0, 1)."
            raise ValueError(message)


@dataclass(frozen=True)
class PreparedOrdination:
    """Owned numeric source, aligned identities and actual frozen query source."""

    values: Array
    row_ids: tuple[int | str, ...]
    feature_names: tuple[str, ...]
    geometry: PreparedGeometry | None
    feature_distances: FeatureDistanceMap | None
    origin: str
    construction_warrant: str | None
    model_token: object = field(repr=False)


def _feature_source(
    data: _Input, names: tuple[str, ...] | None
) -> tuple[Array, tuple[str, ...], tuple[int | str, ...]]:
    if isinstance(
        data,
        (
            MeasureColumn,
            PackedClouds,
            PreparedGeometry,
            PairwiseGeometry,
            ObservedDissimilarity,
        ),
    ):
        message = "Feature ordination requires a feature matrix or named observations."
        raise TypeError(message)
    if isinstance(data, Observations):
        if names is None:
            message = "Ordination on Observations requires explicit feature names."
            raise ValueError(message)
        ids = data.row_ids
    else:
        values = jnp.asarray(data)
        if values.ndim != _MATRIX_NDIM:
            message = "Features must be an observation-by-feature matrix."
            raise ValueError(message)
        names = names or tuple(f"feature_{index}" for index in range(values.shape[1]))
        ids = tuple(range(values.shape[0]))
    values = feature_values(data, names)
    if len(values) < _MIN_ROWS:
        message = "Ordination requires at least two training rows."
        raise ValueError(message)
    return values, names, ids


def _distance_source(
    data: _Input, specification: GeometrySpec | None, column: str | None
) -> tuple[ObservedDissimilarity, PreparedGeometry | None, str | None]:
    prepared = None
    if specification is not None:
        if isinstance(data, Observations):
            if column is None:
                message = "A measure geometry on Observations requires a named column."
                raise ValueError(message)
            data = MeasureColumn(data.row_ids, data.measure(column))
        if not isinstance(data, (PackedClouds, MeasureColumn)):
            message = "A geometry specification requires empirical measures."
            raise TypeError(message)
        prepared = (
            specification.prepare(data.measures, row_ids=data.row_ids)
            if isinstance(data, MeasureColumn)
            else specification.prepare(data)
        )
        data = prepared
    if isinstance(data, PreparedGeometry):
        prepared = data
        data = prepared.pairwise()
    if isinstance(data, PairwiseGeometry):
        observed = ObservedDissimilarity.from_array(data.values, data.row_ids)
        return observed, prepared, data.construction_warrant
    if isinstance(data, ObservedDissimilarity):
        return ObservedDissimilarity.from_array(data.values, data.row_ids), None, None
    message = (
        "Distance ordination requires observed or prepared distances "
        "with row identities."
    )
    raise TypeError(message)


@dataclass(frozen=True)
class Ordination(Model[_Input, PreparedOrdination, "OrdinationResult"], Term):
    """One coordinate-model specification, also usable directly as a design term."""

    geometry: _Geometry
    coordinates: PrincipalCoordinates = field(default_factory=PrincipalCoordinates)
    name: str = "ordination"

    @staticmethod
    def pca(
        features: tuple[str, ...] | None = None,
        *,
        dim: int | None = 2,
        scale: bool = False,
        name: str = "ordination",
    ) -> Ordination:
        """Compose centered feature SVD without adding another model identity."""
        return Ordination(LinearGram(features, scale), PrincipalCoordinates(dim), name)

    @staticmethod
    def pcoa(
        distances: GeometrySpec | None = None,
        *,
        column: str | None = None,
        dim: int | None = 2,
        negative_inertia: Literal["raise", "positive-subspace"] = "raise",
        tolerance: float | None = None,
        name: str = "ordination",
    ) -> Ordination:
        """Compose squared-distance principal coordinates with an explicit policy."""
        return Ordination(
            DistanceGram(distances, column),
            PrincipalCoordinates(dim, negative_inertia, tolerance),
            name,
        )

    def __post_init__(self) -> None:
        """Require mathematical components rather than whole-method wrappers."""
        if not isinstance(self.geometry, (LinearGram, KernelGram, DistanceGram)):
            message = (
                "Ordination requires a feature, kernel or distance Gram construction."
            )
            raise TypeError(message)
        if not isinstance(self.coordinates, PrincipalCoordinates):
            message = "Ordination requires PrincipalCoordinates."
            raise TypeError(message)
        if not isinstance(self.name, str) or not self.name:
            message = "An ordination name must be a nonempty string."
            raise ValueError(message)

    @property
    def input_columns(self) -> dict[str, InputRole]:
        """Declare only the actual named feature or measure source."""
        geometry = self.geometry
        if isinstance(geometry, (LinearGram, KernelGram)):
            return dict.fromkeys(geometry.features or (), "numeric")
        source = geometry.distances
        if isinstance(source, ShortestPathDistances):
            if source.features is not None:
                return dict.fromkeys(source.features, "numeric")
            column = source.column
        else:
            column = geometry.column
        return {} if column is None else {column: "measure"}

    @override
    def prepare(self, data: _Input) -> PreparedOrdination:
        """Own the training source without solving or learning from new rows."""
        geometry = self.geometry
        if isinstance(geometry, (LinearGram, KernelGram)):
            values, names, ids = _feature_source(data, geometry.features)
            return PreparedOrdination(
                values,
                ids,
                names,
                None,
                None,
                "linear feature geometry"
                if isinstance(geometry, LinearGram)
                else "declared kernel geometry",
                "Centered Euclidean feature Gram."
                if isinstance(geometry, LinearGram)
                else "Centered Gram from a declared PSD kernel.",
                self._token,
            )
        source = geometry.distances
        if isinstance(source, ShortestPathDistances) and source.features is not None:
            values, names, ids = _feature_source(data, source.features)
            distance_map = FeatureDistanceMap(names, values, source.ground)
            distances = cdist(
                values, values, metric=resolve_ground_distance(source.ground)
            )
            observed = ObservedDissimilarity.from_array(distances, ids)
            return PreparedOrdination(
                observed.values,
                ids,
                names,
                None,
                distance_map,
                "union-kNN feature paths",
                None,
                self._token,
            )
        specification = (
            source.distances if isinstance(source, ShortestPathDistances) else source
        )
        column = (
            source.column
            if isinstance(source, ShortestPathDistances)
            else geometry.column
        )
        observed, prepared, warrant = _distance_source(data, specification, column)
        if isinstance(source, ShortestPathDistances):
            origin = "union-kNN paths; no Hilbertian path-distance law asserted"
            warrant = None
        else:
            if (
                warrant is None
                and not isinstance(data, ObservedDissimilarity)
                and self.coordinates.negative_inertia == "raise"
            ):
                message = (
                    "Strict constructed ordination requires qualified "
                    "squared-negative-type geometry. Use an explicit observed "
                    "matrix or positive-subspace analysis for descriptive geometry."
                )
                raise ValueError(message)
            origin = (
                "observed matrix only" if warrant is None else "constructed geometry"
            )
        return PreparedOrdination(
            observed.values,
            observed.row_ids,
            (),
            prepared,
            None,
            origin,
            warrant,
            self._token,
        )

    @override
    def _estimate_prepared(self, prepared: PreparedOrdination) -> OrdinationResult:
        """Solve the selected geometry once and retain its actual extension state."""
        if isinstance(self.geometry, LinearGram):
            return self._linear(prepared)
        if isinstance(self.geometry, KernelGram):
            values = gram(prepared.values, prepared.values, kernel=self.geometry.kernel)
            fit, projection = self._spectral(values)
            mapping: CoordinateMap = KernelCoordinateMap(
                prepared.feature_names,
                prepared.values,
                self.geometry.kernel,
                projection,
            )
        else:
            values = prepared.values
            source = self.geometry.distances
            tie_codes = None
            if isinstance(source, ShortestPathDistances):
                if source.neighbors >= len(values):
                    message = "neighbors must be smaller than the training roster."
                    raise ValueError(message)
                ordered = sorted(
                    prepared.row_ids, key=lambda key: (isinstance(key, str), key)
                )
                tie_codes = jnp.asarray(
                    [ordered.index(key) for key in prepared.row_ids]
                )
                values = knn_shortest_paths(
                    values, neighbors=source.neighbors, tie_codes=tie_codes
                )
                if not bool(jnp.all(jnp.isfinite(values))):
                    message = (
                        "The union-kNN graph is disconnected or path lengths "
                        "are not representable; increase neighbors explicitly."
                    )
                    raise ValueError(message)
            raw_gram = -0.5 * values**2
            fit, projection = self._spectral(raw_gram)
            column = (
                source.column
                if isinstance(source, ShortestPathDistances)
                else self.geometry.column
            )
            base = DistanceCoordinateMap(
                prepared.row_ids, projection, prepared.geometry, column
            )
            mapping = (
                GeodesicCoordinateMap(
                    base,
                    values,
                    source.neighbors,
                    tie_codes,
                    prepared.feature_distances,
                )
                if isinstance(source, ShortestPathDistances) and tie_codes is not None
                else base
            )
        return self._result(prepared, fit, mapping)

    def _spectral(self, raw_gram: Array) -> tuple[OrdinationFit, CrossGramMap]:
        if raw_gram.ndim != _MATRIX_NDIM or raw_gram.shape[0] != raw_gram.shape[1]:
            message = "A Gram construction must produce a square training matrix."
            raise ValueError(message)
        if not bool(jnp.all(jnp.isfinite(raw_gram))):
            message = "The training Gram is not numerically representable."
            raise ValueError(message)
        slack = (
            32
            * jnp.finfo(raw_gram.dtype).eps
            * jnp.maximum(jnp.max(jnp.abs(raw_gram)), 1)
        )
        if not bool(jnp.all(jnp.abs(raw_gram - raw_gram.T) <= slack)):
            message = "A Gram construction must produce a symmetric training matrix."
            raise ValueError(message)
        native = gram_coordinates(raw_gram, 0.0)
        eigenvalues = native.eigenvalues
        indices, threshold, euclidean = self._axes(eigenvalues, len(raw_gram))
        coordinates = native.coordinates[:, indices]
        axes = coordinates / eigenvalues[indices]
        if not bool(jnp.all(jnp.isfinite(axes))):
            message = "The coordinate extension is not numerically representable."
            raise ValueError(message)
        fit = OrdinationFit(
            coordinates, eigenvalues, jnp.asarray(threshold), jnp.asarray(euclidean)
        )
        return fit, CrossGramMap(raw_gram.mean(axis=0), raw_gram.mean(), axes)

    def _axes(self, eigenvalues: Array, count: int) -> tuple[Array, float, bool]:
        if not bool(jnp.all(jnp.isfinite(eigenvalues))):
            message = "The ordination spectrum is not finite."
            raise ValueError(message)
        relative = self.coordinates.tolerance
        if relative is None:
            relative = 8 * count * float(jnp.finfo(eigenvalues.dtype).eps)
        scale = float(jnp.max(jnp.abs(eigenvalues)))
        threshold = relative * scale
        euclidean = bool(jnp.all(eigenvalues >= -threshold))
        if not euclidean and self.coordinates.negative_inertia == "raise":
            raise NonEuclideanEmbeddingError(float(jnp.min(eigenvalues)), scale)
        active = jnp.flatnonzero(eigenvalues > threshold)
        dim = len(active) if self.coordinates.dim is None else self.coordinates.dim
        if dim > len(active):
            message = "dim exceeds the supported positive spectral rank."
            raise ValueError(message)
        return active[:dim], threshold, euclidean

    def _linear(self, prepared: PreparedOrdination) -> OrdinationResult:
        geometry = self.geometry
        if not isinstance(geometry, LinearGram):
            message = "A feature factor requires LinearGram."
            raise TypeError(message)
        values = prepared.values
        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=1) if geometry.scale else jnp.ones_like(mean)
        if not bool(jnp.all(jnp.isfinite(scale) & (scale > 0))):
            message = "Scaling requires positive finite training feature variances."
            raise ValueError(message)
        standardized = (values - mean) / scale
        if not bool(jnp.all(jnp.isfinite(standardized))):
            message = "Training centering/scaling is not numerically representable."
            raise ValueError(message)
        full = fit_projection(standardized, d_out=min(values.shape[1], len(values) - 1))
        scores = full.transform(standardized)
        eigenvalues = jnp.pad(
            jnp.sum(scores**2, axis=0), (0, len(values) - scores.shape[1])
        )
        indices, threshold, euclidean = self._axes(eigenvalues, len(values))
        projection = Projection(full.mean, full.components[:, indices], full.kind)
        fit = OrdinationFit(
            scores[:, indices],
            eigenvalues,
            jnp.asarray(threshold),
            jnp.asarray(euclidean),
        )
        mapping = LinearCoordinateMap(prepared.feature_names, mean, scale, projection)
        return self._result(prepared, fit, mapping)

    def _result(
        self, prepared: PreparedOrdination, fit: OrdinationFit, mapping: CoordinateMap
    ) -> OrdinationResult:
        return OrdinationResult(
            fit,
            OrdinationContext(
                prepared.row_ids,
                prepared.origin,
                prepared.construction_warrant,
                prepared.feature_names,
            ),
            mapping,
            self.name,
        )

    @override
    def _prepare_term(self, training: Observations) -> FittedOrdination:
        """Fit on training predictors only and retain the actual frozen map."""
        if self.coordinates.dim is None or not self.input_columns:
            message = (
                "An ordination predictor requires fixed dim and "
                "a named transformable source."
            )
            raise ValueError(message)
        fitted = self.estimate(training)
        return FittedOrdination(
            fitted.mapping,
            tuple(
                f"{self.name}_{axis + 1}" for axis in range(fitted.coordinates.shape[1])
            ),
            fitted.evidence,
        )


@dataclass(frozen=True)
class OrdinationResult(Result[OrdinationFit, OrdinationContext]):
    """Selected coordinates, full signed evidence and the actual frozen map."""

    mapping: CoordinateMap
    name: str = "ordination"

    @property
    def evidence(self) -> OrdinationEvidence:
        """Share actual fitted state with nested designs without decomposing again."""
        return OrdinationEvidence(self.name, self.core, self.mapping, self.context)

    @property
    def valid(self) -> bool:
        """Expose numerical availability independently of negative-inertia policy."""
        return bool(
            jnp.all(jnp.isfinite(self.core.coordinates))
            & jnp.all(jnp.isfinite(self.core.eigenvalues))
        )

    @property
    def coordinates(self) -> Array:
        """Return retained training coordinates in their original row order."""
        return self.core.coordinates

    @property
    def row_ids(self) -> tuple[int | str, ...]:
        """Return the exact training reference identities."""
        return self.context.row_ids

    @property
    def eigenvalues(self) -> Array:
        """Return full signed eigenvalues in centered-Gram units."""
        return self.core.eigenvalues

    @property
    def loadings(self) -> Array:
        """Return feature loadings only for the retained linear projection."""
        return self.evidence.loadings

    @property
    def spectrum(self) -> OrdinationSpectrum:
        """Return signed-spectrum diagnostics without inferring a population law."""
        return self.evidence.spectrum

    @property
    def explained_variance(self) -> Array:
        """Return sample variances of the selected linear feature axes."""
        return self.evidence.explained_variance

    @property
    def explained_variance_ratio(self) -> Array:
        """Return linear-feature variance fractions; zero total yields NaN."""
        return self.evidence.explained_variance_ratio

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Expose only transformations defined by the retained source."""
        return coordinate_capabilities(self.mapping)

    def transform(self, data: Observations | ArrayLike | PackedClouds) -> Array:
        """Apply the retained training transformation without any new fitting."""
        return transform_coordinates(self.mapping, data)

    def transform_distances(
        self, distances: ArrayLike, *, reference_ids: tuple[int | str, ...]
    ) -> Array:
        """Project supplied original distances with exact training reference order."""
        return transform_cross_distances(
            self.mapping, distances, reference_ids=reference_ids
        )

    @override
    def report(self) -> Report:
        """Report observed spectral evidence and explicit approximation limits."""
        spectrum = self.spectrum
        warnings = (
            ()
            if not spectrum.material_negative_inertia
            else (
                (
                    "Coordinates approximate only the positive subspace; "
                    "material negative inertia remains."
                ),
            )
        )
        return Report(
            "Ordination",
            metrics=(
                Metric("observations", len(self.row_ids)),
                Metric("components", int(self.coordinates.shape[1])),
                Metric("negative_axes", int(spectrum.n_negative)),
            ),
            metadata={
                "geometry_evidence": self.context.origin,
                "construction_warrant": self.context.construction_warrant,
                "spectrum_units": "centered Gram",
                "threshold": spectrum.threshold,
                "capabilities": self.capabilities,
            },
            warnings=warnings,
            plots=(
                PlotSpec(
                    "line",
                    tuple(range(1, self.eigenvalues.size + 1)),
                    tuple(float(value) for value in self.eigenvalues),
                    xlabel="Axis",
                    ylabel="Signed centered-Gram eigenvalue",
                    title="Ordination spectrum",
                ),
            ),
        )

    def save_predictor(self, path: str | Path) -> None:
        """Persist the actual frozen transformation, without restoring inference."""
        save_ordination_predictor(
            Path(path),
            mapping=self.mapping,
            metadata={
                "row_ids": self.row_ids,
                "origin": self.context.origin,
                "spectrum_units": "centered Gram",
                "negative_inertia": self.spectrum.material_negative_inertia,
            },
        )
