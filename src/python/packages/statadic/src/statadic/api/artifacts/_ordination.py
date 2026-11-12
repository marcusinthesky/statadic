"""Closed codecs for frozen ordination maps, independently of model constructors."""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003 -- runtime artifact annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax.numpy as jnp

from statadic.api.data import (
    Observations,  # noqa: TC001 -- runtime artifact annotations
)
from statadic.api.ordination_terms import (
    CoordinateMap,  # noqa: TC001 -- runtime artifact annotations
    DistanceCoordinateMap,
    FeatureDistanceMap,
    GeodesicCoordinateMap,
    KernelCoordinateMap,
    LinearCoordinateMap,
    coordinate_capabilities,
    transform_coordinates,
    transform_cross_distances,
)
from statadic.api.reporting import (
    FrozenMetadataValue,  # noqa: TC001 -- runtime artifact annotations
    MetadataValue,  # noqa: TC001 -- runtime artifact annotations
    freeze_metadata,
    metadata_to_json,
)
from statadic.core.typing import Array, ArrayLike  # noqa: TC001 -- runtime annotations
from statadic.geometry.embedding import CrossGramMap
from statadic.geometry.ground.config import resolve_ground_distance
from statadic.geometry.ground.metrics import (
    ANGULAR,
    COSINE,
    EUCLIDEAN,
    NORMALIZED_EUCLIDEAN,
)
from statadic.sample.packed import PackedClouds  # noqa: TC001 -- runtime annotations
from statadic.sample.reduction import Projection

from ._geometry import decode_geometry, decode_geometry_record, encode_geometry
from ._kernel import decode_kernel, decode_kernel_record, encode_kernel
from ._schema import (  # noqa: TC001 -- runtime artifact annotations
    CoordinateMapRecord,
    CrossGramRecord,
    FeatureDistanceRecord,
    GeodesicMapRecord,
)
from ._store import (
    Arrays,
    ArtifactError,
    integer,
    items,
    labels,
    optional_text,
    read_bundle,
    record,
    text,
    write_bundle,
)

_GROUNDS = {
    "euclidean": EUCLIDEAN,
    "angular": ANGULAR,
    "cosine": COSINE,
    "normalized_euclidean": NORMALIZED_EUCLIDEAN,
}
type _GroundName = Literal["euclidean", "angular", "cosine", "normalized_euclidean"]
_MATRIX_NDIM = 2
_MIN_ROWS = 2


def _ground_name(value: object) -> _GroundName:
    if value == "euclidean":
        return "euclidean"
    if value == "angular":
        return "angular"
    if value == "cosine":
        return "cosine"
    if value == "normalized_euclidean":
        return "normalized_euclidean"
    msg = "Unknown retained ground distance."
    raise ArtifactError(msg)


def _names(value: object) -> list[str]:
    names = [text(name, "feature name") for name in items(value, "feature names")]
    if not names or len(set(names)) != len(names):
        msg = "Feature names must be nonempty and unique."
        raise ArtifactError(msg)
    return names


def _encode_cross(value: CrossGramMap, arrays: Arrays) -> CrossGramRecord:
    return {
        "column_mean": arrays.add(value.column_mean),
        "grand_mean": arrays.add(value.grand_mean),
        "axes": arrays.add(value.axes),
    }


def _cross_record(value: object) -> CrossGramRecord:
    saved = record(value, {"column_mean", "grand_mean", "axes"}, "cross-Gram map")
    return {
        "column_mean": text(saved["column_mean"], "column mean"),
        "grand_mean": text(saved["grand_mean"], "grand mean"),
        "axes": text(saved["axes"], "axes"),
    }


def _decode_cross(value: CrossGramRecord, arrays: Arrays) -> CrossGramMap:
    mean, grand, axes = (
        arrays.get(value[name]) for name in ("column_mean", "grand_mean", "axes")
    )
    if (
        mean.ndim != 1
        or grand.shape != ()
        or axes.ndim != _MATRIX_NDIM
        or any(a.dtype.kind != "f" for a in (mean, grand, axes))
    ):
        msg = "Invalid retained cross-Gram centering or projection shape."
        raise ArtifactError(msg)
    if (
        mean.size < _MIN_ROWS
        or axes.shape[0] != mean.size
        or axes.shape[1] >= mean.size
    ):
        msg = "Cross-Gram axes must align with at least two training references."
        raise ArtifactError(msg)
    return CrossGramMap(mean, grand, axes)


def encode_coordinate_map(value: CoordinateMap, arrays: Arrays) -> CoordinateMapRecord:
    """Persist only known frozen scientific transformations."""
    if type(value) is LinearCoordinateMap:
        return {
            "kind": "linear",
            "feature_names": list(value.feature_names),
            "mean": arrays.add(value.mean),
            "scale": arrays.add(value.scale),
            "projection_mean": arrays.add(value.projection.mean),
            "components": arrays.add(value.projection.components),
        }
    if type(value) is KernelCoordinateMap:
        return {
            "kind": "kernel",
            "feature_names": list(value.feature_names),
            "reference": arrays.add(value.reference),
            "kernel": encode_kernel(value.kernel, arrays),
            "projection": _encode_cross(value.projection, arrays),
        }
    if type(value) is DistanceCoordinateMap:
        return {
            "kind": "distance",
            "row_ids": list(value.row_ids),
            "projection": _encode_cross(value.projection, arrays),
            "geometry": None
            if value.geometry is None
            else encode_geometry(value.geometry, arrays),
            "measure_column": value.measure_column,
        }
    if type(value) is GeodesicCoordinateMap:
        distance = encode_coordinate_map(value.distance_map, arrays)
        if distance["kind"] != "distance":
            msg = "Geodesic extension requires a distance map."
            raise ArtifactError(msg)
        features = value.feature_distances
        feature_record: FeatureDistanceRecord | None = None
        if features is not None:
            ground = resolve_ground_distance(features.ground)
            name = next(
                (name for name, candidate in _GROUNDS.items() if ground is candidate),
                None,
            )
            if name is None:
                msg = "Custom ground distances have no declarative prediction codec."
                raise ArtifactError(msg)
            feature_record = {
                "feature_names": list(features.feature_names),
                "reference": arrays.add(features.reference),
                "ground": _ground_name(name),
            }
        return {
            "kind": "geodesic",
            "distance_map": distance,
            "training_paths": arrays.add(value.training_paths),
            "neighbors": value.neighbors,
            "tie_codes": arrays.add(value.tie_codes),
            "feature_distances": feature_record,
        }
    msg = "Custom coordinate maps have no declarative prediction codec."
    raise ArtifactError(msg)


def decode_coordinate_record(value: object) -> CoordinateMapRecord:
    """Validate the closed recursive wire schema before allocating a map."""
    if not isinstance(value, dict):
        msg = "A coordinate map must be a declarative record."
        raise ArtifactError(msg)
    kind = value.get("kind")
    if kind == "linear":
        saved = record(
            value,
            {"kind", "feature_names", "mean", "scale", "projection_mean", "components"},
            "linear coordinate map",
        )
        return {
            "kind": "linear",
            "feature_names": _names(saved["feature_names"]),
            "mean": text(saved["mean"], "mean"),
            "scale": text(saved["scale"], "scale"),
            "projection_mean": text(saved["projection_mean"], "projection mean"),
            "components": text(saved["components"], "components"),
        }
    if kind == "kernel":
        saved = record(
            value,
            {"kind", "feature_names", "reference", "kernel", "projection"},
            "kernel coordinate map",
        )
        return {
            "kind": "kernel",
            "feature_names": _names(saved["feature_names"]),
            "reference": text(saved["reference"], "reference"),
            "kernel": decode_kernel_record(saved["kernel"]),
            "projection": _cross_record(saved["projection"]),
        }
    if kind == "distance":
        saved = record(
            value,
            {"kind", "row_ids", "projection", "geometry", "measure_column"},
            "distance coordinate map",
        )
        return {
            "kind": "distance",
            "row_ids": list(labels(saved["row_ids"], "reference identities")),
            "projection": _cross_record(saved["projection"]),
            "geometry": None
            if saved["geometry"] is None
            else decode_geometry_record(saved["geometry"]),
            "measure_column": optional_text(saved["measure_column"], "measure column"),
        }
    if kind == "geodesic":
        saved = record(
            value,
            {
                "kind",
                "distance_map",
                "training_paths",
                "neighbors",
                "tie_codes",
                "feature_distances",
            },
            "geodesic coordinate map",
        )
        distance = decode_coordinate_record(saved["distance_map"])
        if distance["kind"] != "distance":
            msg = "Geodesic extension requires a distance map."
            raise ArtifactError(msg)
        feature_record: FeatureDistanceRecord | None = None
        if saved["feature_distances"] is not None:
            features = record(
                saved["feature_distances"],
                {"feature_names", "reference", "ground"},
                "feature distance map",
            )
            ground = features["ground"]
            if ground not in _GROUNDS:
                msg = "Unknown retained ground distance."
                raise ArtifactError(msg)
            feature_record = {
                "feature_names": _names(features["feature_names"]),
                "reference": text(features["reference"], "feature reference"),
                "ground": _ground_name(ground),
            }
        return {
            "kind": "geodesic",
            "distance_map": distance,
            "training_paths": text(saved["training_paths"], "training paths"),
            "neighbors": integer(saved["neighbors"], "neighbors"),
            "tie_codes": text(saved["tie_codes"], "tie codes"),
            "feature_distances": feature_record,
        }
    msg = "Unknown coordinate map; executable classes are unsupported."
    raise ArtifactError(msg)


def decode_coordinate_map(saved: CoordinateMapRecord, arrays: Arrays) -> CoordinateMap:
    """Restore and validate dimensions, identities and numerical domains."""
    if saved["kind"] == "linear":
        names = tuple(saved["feature_names"])
        mean, scale, center, components = (
            arrays.get(saved[name])
            for name in ("mean", "scale", "projection_mean", "components")
        )
        if (
            any(a.dtype.kind != "f" for a in (mean, scale, center, components))
            or any(a.shape != (len(names),) for a in (mean, scale, center))
            or components.ndim != _MATRIX_NDIM
        ):
            msg = "Invalid retained feature centering, scale or projection."
            raise ArtifactError(msg)
        if (
            components.shape[0] != len(names)
            or components.shape[1] > len(names)
            or not bool(jnp.all(scale > 0))
        ):
            msg = "Invalid feature projection width or nonpositive scale."
            raise ArtifactError(msg)
        return LinearCoordinateMap(
            names, mean, scale, Projection(center, components, kind="pca")
        )
    if saved["kind"] == "kernel":
        reference = arrays.get(saved["reference"])
        names = tuple(saved["feature_names"])
        projection = _decode_cross(saved["projection"], arrays)
        if (
            reference.ndim != _MATRIX_NDIM
            or reference.shape != (projection.column_mean.size, len(names))
            or reference.dtype.kind != "f"
        ):
            msg = "Kernel reference rows do not align with retained axes."
            raise ArtifactError(msg)
        return KernelCoordinateMap(
            names, reference, decode_kernel(saved["kernel"], arrays), projection
        )
    if saved["kind"] == "distance":
        ids = tuple(saved["row_ids"])
        projection = _decode_cross(saved["projection"], arrays)
        geometry = (
            None
            if saved["geometry"] is None
            else decode_geometry(saved["geometry"], arrays)
        )
        if len(ids) != projection.column_mean.size or (
            geometry is not None and geometry.row_ids != ids
        ):
            msg = "Distance reference identities disagree with retained axes."
            raise ArtifactError(msg)
        if saved["measure_column"] is not None and geometry is None:
            msg = "A measure-column transformation requires retained geometry."
            raise ArtifactError(msg)
        return DistanceCoordinateMap(ids, projection, geometry, saved["measure_column"])
    return _decode_geodesic(saved, arrays)


def _decode_geodesic(saved: GeodesicMapRecord, arrays: Arrays) -> GeodesicCoordinateMap:
    distance = decode_coordinate_map(saved["distance_map"], arrays)
    if not isinstance(distance, DistanceCoordinateMap):
        msg = "Geodesic extension requires a distance map."
        raise ArtifactError(msg)
    paths, ties = arrays.get(saved["training_paths"]), arrays.get(saved["tie_codes"])
    n = len(distance.row_ids)
    if (
        paths.shape != (n, n)
        or paths.dtype.kind != "f"
        or not bool(jnp.all(paths >= 0))
        or ties.shape != (n,)
        or ties.dtype.kind not in "iu"
    ):
        msg = "Invalid retained geodesic path or tie-code dimensions."
        raise ArtifactError(msg)
    path_slack = (
        32 * jnp.finfo(paths.dtype).eps * jnp.maximum(jnp.max(jnp.abs(paths)), 1)
    )
    if (
        not bool(jnp.all(jnp.abs(paths - paths.T) <= path_slack))
        or not bool(jnp.all(jnp.diag(paths) == 0))
        or not bool(jnp.array_equal(jnp.sort(ties), jnp.arange(n)))
        or not 1 <= saved["neighbors"] < n
    ):
        msg = "Invalid retained geodesic paths, neighbours or identity tie codes."
        raise ArtifactError(msg)
    features = saved["feature_distances"]
    feature_map = None
    if features is not None:
        reference = arrays.get(features["reference"])
        names = tuple(features["feature_names"])
        if (
            reference.shape != (n, len(names))
            or reference.dtype.kind != "f"
            or distance.geometry is not None
        ):
            msg = "Geodesic feature references do not align."
            raise ArtifactError(msg)
        feature_map = FeatureDistanceMap(names, reference, features["ground"])
    return GeodesicCoordinateMap(distance, paths, saved["neighbors"], ties, feature_map)


@dataclass(frozen=True)
class OrdinationArtifact:
    """A frozen transformation; estimation and inference are not restored."""

    mapping: CoordinateMap
    metadata: Mapping[str, FrozenMetadataValue]

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Expose only transformations supported by retained state."""
        return coordinate_capabilities(self.mapping)

    def transform(self, data: Observations | ArrayLike | PackedClouds) -> Array:
        """Apply the retained transformation without fitting."""
        return transform_coordinates(self.mapping, data)

    def transform_distances(
        self, distances: ArrayLike, *, reference_ids: tuple[int | str, ...]
    ) -> Array:
        """Project supplied new-by-training distances in the exact reference order."""
        return transform_cross_distances(
            self.mapping, distances, reference_ids=reference_ids
        )


def save_ordination_predictor(
    path: Path, *, mapping: CoordinateMap, metadata: Mapping[str, MetadataValue]
) -> None:
    """Store a closed, transform-only retained coordinate map."""
    arrays = Arrays()
    saved = decode_coordinate_record(encode_coordinate_map(mapping, arrays))
    restored = decode_coordinate_map(saved, arrays)
    plain = metadata_to_json(freeze_metadata(metadata))
    artifact = OrdinationArtifact(restored, freeze_metadata(plain))
    write_bundle(
        Path(path),
        purpose="prediction",
        family="ordination",
        capabilities=artifact.capabilities,
        state=saved,
        arrays=arrays,
        metadata=plain,
    )


def load_ordination_predictor(path: Path) -> OrdinationArtifact:
    """Restore declared transformations without manufacturing a fitted model."""
    manifest, arrays = read_bundle(Path(path), purpose="prediction")
    if manifest["family"] != "ordination":
        msg = "This artifact does not declare an ordination transformation."
        raise ArtifactError(msg)
    try:
        mapping = decode_coordinate_map(
            decode_coordinate_record(manifest["state"]), arrays
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, ArtifactError):
            raise
        msg = "Invalid retained ordination state."
        raise ArtifactError(msg) from error
    artifact = OrdinationArtifact(mapping, freeze_metadata(manifest["metadata"]))
    if manifest["capabilities"] != list(artifact.capabilities):
        msg = "Ordination capability declaration disagrees with retained state."
        raise ArtifactError(msg)
    arrays.require_consumed()
    return artifact
