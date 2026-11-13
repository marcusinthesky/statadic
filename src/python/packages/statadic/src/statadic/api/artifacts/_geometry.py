"""Closed codecs for retained empirical-measure geometry."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from statadic.api.representation import (
    MMD,
    Energy,
    PreparedGeometry,  # runtime artifact annotations  # noqa: TC001
    SinkhornDivergence,
    SlicedWasserstein,
    Wasserstein1D,
)
from statadic.discrepancy.transport.sliced import ProjectionDesign
from statadic.geometry.ground.config import resolve_ground_distance
from statadic.geometry.ground.metrics import (
    ANGULAR,
    COSINE,
    EUCLIDEAN,
    NORMALIZED_EUCLIDEAN,
)
from statadic.sample.packed import PackedClouds

from ._kernel import decode_kernel, decode_kernel_record, encode_kernel
from ._schema import (
    EnergyGeometry,  # noqa: TC001
    GeometryRecord,  # noqa: TC001
    MMDGeometry,  # noqa: TC001
    SinkhornGeometry,  # noqa: TC001
    SlicedGeometry,  # noqa: TC001
    WassersteinGeometry,  # noqa: TC001
    _GeometryState,  # noqa: TC001
)
from ._store import (
    Arrays,  # noqa: TC001
    ArtifactError,
    _require_artifact_precision,
    integer,
    labels,
    number,
    record,
    text,
)

_SUPPORT_AXES = 3
_W2_ORDER = 2

_GROUNDS = {
    "euclidean": EUCLIDEAN,
    "angular": ANGULAR,
    "cosine": COSINE,
    "normalized_euclidean": NORMALIZED_EUCLIDEAN,
}


def encode_geometry(value: PreparedGeometry, arrays: Arrays) -> GeometryRecord:
    """Serialize only known numerical constructions, never diagnostic class names."""
    _require_artifact_precision(value.reference.data)
    _require_artifact_precision(value.reference.counts)
    _require_artifact_precision(value.reference.masses)
    spec = value.spec
    if type(spec) not in {
        MMD,
        Energy,
        Wasserstein1D,
        SlicedWasserstein,
        SinkhornDivergence,
    }:
        message = "No prediction codec is registered for this geometry."
        raise ArtifactError(message)
    config: dict[str, object]
    kind: str
    if isinstance(spec, MMD):
        kind = "mmd"
        config = {"kernel": encode_kernel(spec.kernel, arrays)}
    elif isinstance(spec, Energy):
        resolved = resolve_ground_distance(spec.ground)
        names = [name for name, strategy in _GROUNDS.items() if resolved is strategy]
        if not names:
            message = "Custom ground strategies have no declarative prediction codec."
            raise ArtifactError(message)
        kind = "energy"
        config = {"ground": names[0], "exponent": spec.exponent}
    elif isinstance(spec, SinkhornDivergence):
        kind = "sinkhorn_divergence"
        config = {
            "epsilon": spec.epsilon,
            "iterations": spec.iterations,
            "tolerance": spec.tolerance,
        }
    elif isinstance(spec, Wasserstein1D):
        kind = "wasserstein_1d"
        config = {"p": spec.p}
    else:
        if not isinstance(spec, SlicedWasserstein):
            message = "Unsupported geometry codec."
            raise ArtifactError(message)
        key = spec.design.key
        directions = spec.design.directions(
            value.reference.data.shape[-1], value.reference.data.dtype
        )
        kind = "sliced_wasserstein"
        config = {
            "p": float(spec.design.p),
            "n_projections": spec.design.n_projections,
            "key": arrays.add(jax.random.key_data(key)),
            "key_implementation": str(jax.random.key_impl(key)),
            "directions": arrays.add(directions),
        }
    data, counts, masses = jax.device_get(
        (
            value.reference.data,
            value.reference.counts,
            value.reference.masses,
        )
    )
    active = jnp.arange(data.shape[1])[None, :] < counts[:, None]
    # Zero-mass and inactive coordinates do not contribute to the measure.
    data = jnp.where((active & (masses > 0))[..., None], data, 0)
    masses = jnp.where(active, masses, 0)
    return decode_geometry_record(
        {
            "kind": kind,
            "config": config,
            "data": arrays.add(data),
            "counts": arrays.add(counts),
            "masses": arrays.add(masses),
            "row_ids": list(value.row_ids),
        }
    )


def _ground_name(
    value: object,
) -> Literal["euclidean", "angular", "cosine", "normalized_euclidean"]:
    if value == "euclidean":
        return "euclidean"
    if value == "angular":
        return "angular"
    if value == "cosine":
        return "cosine"
    if value == "normalized_euclidean":
        return "normalized_euclidean"
    message = "Unknown persisted ground distance."
    raise ArtifactError(message)


def decode_geometry_record(value: object) -> GeometryRecord:
    """Decode a discriminated geometry schema without consuming numeric arrays."""
    saved = record(
        value, {"kind", "config", "data", "counts", "masses", "row_ids"}, "geometry"
    )
    common: _GeometryState = {
        "data": text(saved["data"], "support reference"),
        "counts": text(saved["counts"], "counts reference"),
        "masses": text(saved["masses"], "masses reference"),
        "row_ids": list(labels(saved["row_ids"], "landmark row IDs")),
    }
    if saved["kind"] == "mmd":
        config = record(saved["config"], {"kernel"}, "MMD configuration")
        kernel = decode_kernel_record(config["kernel"])
        mmd: MMDGeometry = {
            "kind": "mmd",
            "config": {"kernel": kernel},
            "data": common["data"],
            "counts": common["counts"],
            "masses": common["masses"],
            "row_ids": common["row_ids"],
        }
        return mmd
    if saved["kind"] == "energy":
        config = record(saved["config"], {"ground", "exponent"}, "energy configuration")
        energy: EnergyGeometry = {
            "kind": "energy",
            "config": {
                "ground": _ground_name(config["ground"]),
                "exponent": number(config["exponent"], "energy exponent"),
            },
            "data": common["data"],
            "counts": common["counts"],
            "masses": common["masses"],
            "row_ids": common["row_ids"],
        }
        return energy
    if saved["kind"] == "sinkhorn_divergence":
        settings = record(
            saved["config"],
            {"epsilon", "iterations", "tolerance"},
            "Sinkhorn configuration",
        )
        sinkhorn: SinkhornGeometry = {
            "kind": "sinkhorn_divergence",
            "config": {
                "epsilon": number(settings["epsilon"], "regularization"),
                "iterations": integer(settings["iterations"], "iterations"),
                "tolerance": number(settings["tolerance"], "marginal tolerance"),
            },
            "data": common["data"],
            "counts": common["counts"],
            "masses": common["masses"],
            "row_ids": common["row_ids"],
        }
        return sinkhorn
    if saved["kind"] == "wasserstein_1d":
        config = record(saved["config"], {"p"}, "Wasserstein configuration")
        order = number(config["p"], "Wasserstein order")
        if order != _W2_ORDER:
            message = "Only the qualified real-line W2 codec is supported."
            raise ArtifactError(message)
        wasserstein: WassersteinGeometry = {
            "kind": "wasserstein_1d",
            "config": {"p": order},
            "data": common["data"],
            "counts": common["counts"],
            "masses": common["masses"],
            "row_ids": common["row_ids"],
        }
        return wasserstein
    if saved["kind"] == "sliced_wasserstein":
        config = record(
            saved["config"],
            {"p", "n_projections", "key", "key_implementation", "directions"},
            "sliced configuration",
        )
        count = integer(config["n_projections"], "projection count")
        if count < 1:
            message = "Invalid sliced projection count/order."
            raise ArtifactError(message)
        sliced: SlicedGeometry = {
            "kind": "sliced_wasserstein",
            "config": {
                "p": number(config["p"], "projection order"),
                "n_projections": count,
                "key": text(config["key"], "PRNG key reference"),
                "key_implementation": text(
                    config["key_implementation"], "PRNG implementation"
                ),
                "directions": text(
                    config["directions"], "projection directions reference"
                ),
            },
            "data": common["data"],
            "counts": common["counts"],
            "masses": common["masses"],
            "row_ids": common["row_ids"],
        }
        return sliced
    message = f"Unknown prediction geometry {saved['kind']!r}."
    raise ArtifactError(message)


def decode_geometry(saved: GeometryRecord, arrays: Arrays) -> PreparedGeometry:
    """Reconstruct support, masses and the specification from a decoded record.

    Wire input is parsed once by :func:`decode_geometry_record`.
    """
    data, counts, masses = (
        arrays.get(saved[name]) for name in ("data", "counts", "masses")
    )
    if (
        data.ndim != _SUPPORT_AXES
        or counts.shape != (data.shape[0],)
        or masses.shape != data.shape[:2]
        or counts.dtype.kind not in "iu"
        or jnp.any((counts < 1) | (counts > data.shape[1]))
    ):
        message = "Landmark supports, true counts and atom masses do not align."
        raise ArtifactError(message)
    empirical = PackedClouds.from_clouds(
        [data[i, : int(count)] for i, count in enumerate(counts)],
        [masses[i, : int(count)] for i, count in enumerate(counts)],
    )
    ids = tuple(saved["row_ids"])
    if saved["kind"] == "mmd":
        spec = MMD(decode_kernel(saved["config"]["kernel"], arrays))
    elif saved["kind"] == "energy":
        config = saved["config"]
        spec = Energy(_GROUNDS[config["ground"]], exponent=config["exponent"])
    elif saved["kind"] == "sinkhorn_divergence":
        settings = saved["config"]
        spec = SinkhornDivergence(
            settings["epsilon"], settings["iterations"], settings["tolerance"]
        )
    elif saved["kind"] == "wasserstein_1d":
        spec = Wasserstein1D(saved["config"]["p"])
    else:
        config = saved["config"]
        key = jax.random.wrap_key_data(
            jnp.asarray(arrays.get(config["key"])),
            impl=config["key_implementation"],
        )
        design = ProjectionDesign(
            key, n_projections=config["n_projections"], p=config["p"]
        )
        directions = arrays.get(config["directions"])
        regenerated = jnp.asarray(design.directions(data.shape[-1], data.dtype))
        if not jnp.array_equal(directions, regenerated):
            message = (
                "Persisted projection directions differ in this numerical environment."
            )
            raise ArtifactError(message)
        spec = SlicedWasserstein(design)
    return spec.prepare(empirical, row_ids=ids)
