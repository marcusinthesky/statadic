"""Portable registered distance graphs and their shared numerical parameter state."""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003 -- checked runtime signature
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 -- checked runtime signature
from types import MappingProxyType

from statadic.api.data import MeasureColumn
from statadic.api.spatial.graphs import (
    BandwidthRef,
    DistanceGraph,
    MeasureDistances,
    ProjectionRef,
    RBFWeights,
)
from statadic.core.matrices import DMat
from statadic.core.typing import Array  # noqa: TC001
from statadic.sample.reduction import Projection

from ._geometry import decode_geometry, decode_geometry_record, encode_geometry
from ._store import (
    Arrays,
    ArtifactError,
    labels,
    number,
    read_bundle,
    record,
    string_record,
    text,
    write_bundle,
)

_FAMILY = "distance_graphs"
_CAPABILITY = "rebuild_weights"


@dataclass(frozen=True)
class GraphArchive:
    """Independently re-evaluable recipes with shared fitted parameters."""

    graphs: Mapping[str, DistanceGraph]
    parameters: Mapping[str, Array]


def _keys(graphs: Mapping[str, DistanceGraph]) -> set[str]:
    keys = set()
    for graph in graphs.values():
        h = graph.weighting.bandwidth
        if isinstance(h, BandwidthRef):
            keys.add(h.name)
        if graph.projection_ref is not None:
            keys.add(graph.projection_ref.name)
    return keys


def _encode(
    graph: DistanceGraph, params: Mapping[str, Array], arrays: Arrays
) -> dict[str, object]:
    h = graph.weighting.bandwidth
    bandwidth = {"ref": h.name} if isinstance(h, BandwidthRef) else h
    common = {"ids": list(graph.unit_ids), "bandwidth": bandwidth}
    source = graph.source
    if isinstance(source, DMat):
        return {**common, "kind": "distances", "values": arrays.add(source.values)}
    if not isinstance(source, MeasureDistances):
        return {**common, "kind": "costs", "values": arrays.add(source)}
    projection = source.projection
    if isinstance(projection, ProjectionRef):
        transform = {"ref": projection.name}
    elif isinstance(projection, Projection):
        transform = {
            "mean": arrays.add(projection.mean),
            "components": arrays.add(projection.components),
            "kind": projection.kind,
        }
    elif projection is not None:
        transform = {"matrix": arrays.add(projection)}
    else:
        transform = None
    mapped = source.projected(params)
    geometry = encode_geometry(
        source.geometry.prepare(mapped, row_ids=graph.unit_ids), arrays
    )
    original = (
        arrays.add(source.column.measures.coordinates)
        if projection is not None
        else None
    )
    return {
        **common,
        "kind": "measures",
        "geometry": geometry,
        "transform": transform,
        "input": original,
    }


def save_graphs(
    path: Path,
    graphs: Mapping[str, DistanceGraph],
    *,
    parameters: Mapping[str, Array] | None = None,
) -> None:
    """Persist recipes, slicing designs and one copy of each shared leaf."""
    state = {} if parameters is None else parameters
    if set(state) != _keys(graphs):
        msg = "Graph archive parameter names must exactly match the graph references."
        raise ArtifactError(msg)
    arrays = Arrays()
    payload = {
        "graphs": {
            name: _encode(graph, state, arrays) for name, graph in graphs.items()
        },
        "parameters": {name: arrays.add(value) for name, value in state.items()},
    }
    write_bundle(
        path,
        purpose="prediction",
        family=_FAMILY,
        capabilities=(_CAPABILITY,),
        state=payload,
        arrays=arrays,
        metadata={},
    )


def _decode(value: object, arrays: Arrays) -> DistanceGraph:
    data = string_record(value, "graph")
    is_matrix = data.get("kind") in ("costs", "distances")
    keys = {"ids", "bandwidth", "kind"} | (
        {"values"} if is_matrix else {"geometry", "transform", "input"}
    )
    data = record(data, keys, "distance graph")
    roster = labels(data["ids"], "unit IDs")
    if any(type(unit) is not int for unit in roster):
        msg = "Spatial graph artifacts require integer unit IDs."
        raise ArtifactError(msg)
    ids = tuple(int(unit) for unit in roster)
    h = data["bandwidth"]
    weighting = RBFWeights(
        BandwidthRef(text(record(h, {"ref"}, "bandwidth")["ref"], "parameter"))
        if isinstance(h, dict)
        else number(h, "bandwidth")
    )
    if is_matrix:
        values = arrays.get(data["values"])
        if data["kind"] == "distances":
            return DistanceGraph.from_distances(
                values, unit_ids=ids, weighting=weighting
            )
        return DistanceGraph.from_costs(values, unit_ids=ids, weighting=weighting)
    if data["kind"] != "measures":
        msg = "Unknown distance graph kind."
        raise ArtifactError(msg)
    geometry = decode_geometry(decode_geometry_record(data["geometry"]), arrays)
    if geometry.row_ids != ids:
        msg = "Stored measure identities disagree with the graph roster."
        raise ArtifactError(msg)
    clouds = geometry.reference
    transform = data["transform"]
    projection = None
    if transform is not None:
        config = string_record(transform, "coordinate map")
        if set(config) == {"ref"}:
            projection = ProjectionRef(text(config["ref"], "projection parameter"))
        elif set(config) == {"matrix"}:
            projection = arrays.get(config["matrix"])
        else:
            config = record(config, {"mean", "components", "kind"}, "fitted projection")
            projection = Projection(
                arrays.get(config["mean"]),
                arrays.get(config["components"]),
                text(config["kind"], "kind"),
            )
        clouds = clouds.with_coordinates(arrays.get(data["input"]))
    elif data["input"] is not None:
        msg = "An identity map must not contain a separate source array."
        raise ArtifactError(msg)
    return DistanceGraph.from_measures(
        MeasureColumn(tuple(ids), clouds),
        geometry=geometry.spec,
        projection=projection,
        weighting=weighting,
    )


def load_graphs(path: Path) -> GraphArchive:
    """Restore and validate supported graph reconstruction capabilities."""
    saved, arrays = read_bundle(path, purpose="prediction")
    if saved["family"] != _FAMILY or saved["capabilities"] != [_CAPABILITY]:
        msg = "This loader requires a distance graph reconstruction artifact."
        raise ArtifactError(msg)
    state = record(saved["state"], {"graphs", "parameters"}, "graph archive")
    params = {
        k: arrays.get(v)
        for k, v in string_record(state["parameters"], "parameters").items()
    }
    graphs = {
        k: _decode(v, arrays)
        for k, v in string_record(state["graphs"], "graphs").items()
    }
    if _keys(graphs) != set(params):
        msg = "Stored parameter references do not match the parameter inventory."
        raise ArtifactError(msg)
    for graph in graphs.values():
        graph.realize(params)
    arrays.require_consumed()
    return GraphArchive(MappingProxyType(graphs), MappingProxyType(params))
