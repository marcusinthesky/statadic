"""Contracts for graph construction, JAX derivatives, and portable graph state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.artifacts import load_graphs, save_graphs
from statadic.api.data import MeasureColumn
from statadic.api.representation import (
    Energy,
    SinkhornDivergence,
    SlicedWasserstein,
    Wasserstein1D,
)
from statadic.api.spatial.graphs import (
    BandwidthRef,
    DistanceGraph,
    ProjectionRef,
    RBFWeights,
)
from statadic.core.matrices import DMat
from statadic.discrepancy.transport.sliced import ProjectionDesign
from statadic.operators.spatial.distance import rbf_weights
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    from pathlib import Path

_IDS = (9, 2, 5, 8)


def _clouds() -> MeasureColumn:
    clouds = PackedClouds.from_clouds(
        [
            jnp.array([[0.1, 0.2], [0.5, 0.3]]),
            jnp.array([[1.1, 0.3], [1.1, 0.3], [1.4, 0.1]]),
            jnp.array([[0.8, 1.0]]),
            jnp.array([[-0.4, 0.2], [0.3, 1.2]]),
        ]
    )
    return MeasureColumn(_IDS, clouds)


@pytest.mark.unit
def test_distance_and_cost_recipes_share_one_weight_law() -> None:
    """Directed normalized weights match a manual Gaussian affinity."""
    d = np.abs(np.subtract.outer([0.0, 0.5, 1.2, 1.4], [0.0, 0.5, 1.2, 1.4]))
    aff = np.exp(-(d**2) / 0.7)
    np.fill_diagonal(aff, 0)
    rule = RBFWeights(0.7)
    for graph in (
        DistanceGraph.from_distances(d, unit_ids=_IDS, weighting=rule),
        DistanceGraph.from_costs(d**2, unit_ids=_IDS, weighting=rule),
    ):
        np.testing.assert_allclose(
            graph.realize().values, aff / aff.sum(1, keepdims=True)
        )
    with pytest.raises(ValueError, match="negative"):
        DistanceGraph.from_distances(-d, unit_ids=_IDS, weighting=rule)
    with pytest.raises(ValueError, match="positive"):
        RBFWeights(0)
    np.testing.assert_allclose(
        rbf_weights(jnp.array(d * d), jnp.array(1e-30)).sum(-1), 1
    )


@pytest.mark.integration
@pytest.mark.parametrize("family", ["energy", "sliced", "sinkhorn"])
def test_projected_measure_gradient_and_graph_roundtrip(
    family: str, tmp_path: Path
) -> None:
    """Independent directional differences on weighted, ragged and coincident atoms."""
    geometry = {
        "energy": Energy(),
        "sliced": SlicedWasserstein(
            ProjectionDesign(jax.random.key(12), n_projections=3)
        ),
        "sinkhorn": SinkhornDivergence(1.0, iterations=48, tolerance=1e-5),
    }[family]
    graph = DistanceGraph.from_measures(
        _clouds(),
        geometry=geometry,
        projection=ProjectionRef("P"),
        weighting=RBFWeights(BandwidthRef("h")),
    )
    mat = jnp.array([[0.8, -0.1], [0.2, 0.6]])
    step = 1e-4
    direction = jnp.array([[0.6, 0.3], [-0.2, 0.4]])

    def loss(p: jax.Array, h: jax.Array) -> jax.Array:
        return graph.weights({"P": p, "h": h})[0, 2]

    h = jnp.array(0.7)
    dp, dh = jax.jit(jax.grad(loss, argnums=(0, 1)))(mat, h)
    np.testing.assert_allclose(
        (dp * direction).sum(),
        (loss(mat + step * direction, h) - loss(mat - step * direction, h))
        / (2 * step),
        rtol=2e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        dh,
        (loss(mat, h + step) - loss(mat, h - step)) / (2 * step),
        rtol=2e-4,
        atol=1e-5,
    )
    assert abs(float(dh)) > step / 10
    state = {"P": mat, "h": h}
    save_graphs(tmp_path, {"first": graph, "second": graph}, parameters=state)
    saved = load_graphs(tmp_path)
    np.testing.assert_allclose(
        saved.graphs["second"].weights(saved.parameters),
        graph.weights(state),
        atol=1e-9,
    )
    assert set(saved.parameters) == set(state)


@pytest.mark.component
def test_signed_ids_and_single_atom_sinkhorn_cost(tmp_path: Path) -> None:
    """Dirac transport has the analytic squared ground cost for every epsilon."""
    points = jnp.array([[-1.2, 0.5], [0.25, 2.0], [0.75, 1.0]])
    source = PackedClouds.from_clouds([point[None, :] for point in points])
    cost = jnp.sum((points[:, None] - points[None, :]) ** 2, axis=-1)
    geometry = SinkhornDivergence(epsilon=0.4)
    np.testing.assert_allclose(geometry.squared_cross(source, source), cost)
    ids = (-3, 0, 4)
    graph = DistanceGraph.from_costs(cost, weighting=RBFWeights(1.0), unit_ids=ids)
    rooted = DistanceGraph.from_distances(
        jnp.sqrt(cost), weighting=RBFWeights(1.0), unit_ids=ids
    )
    save_graphs(tmp_path, {"g": graph, "rooted": rooted})
    assert isinstance(load_graphs(tmp_path).graphs["rooted"].source, DMat)
    restored = load_graphs(tmp_path).graphs["g"]
    assert restored.unit_ids == ids
    np.testing.assert_allclose(restored.weights(), graph.weights())


@pytest.mark.component
@pytest.mark.parametrize(
    "geometry", [Wasserstein1D(p=2), SinkhornDivergence(epsilon=1.0)]
)
def test_keyed_geometry_matches_measure_recipe(
    geometry: Wasserstein1D | SinkhornDivergence, tmp_path: Path
) -> None:
    """Prepared and direct graph construction retain the same signed roster."""
    points = jnp.array([0.0, 2.0, 3.0])
    source = PackedClouds.from_clouds([point.reshape(1, 1) for point in points])
    column = MeasureColumn((1, -2, 4), source)
    pairwise = geometry.prepare(column).pairwise()
    weighting = RBFWeights(2.0)
    rooted = DistanceGraph.from_distances(pairwise, weighting=weighting)
    recipe = DistanceGraph.from_measures(column, geometry=geometry, weighting=weighting)
    save_graphs(tmp_path, {"recipe": recipe, "rooted": rooted})
    saved = load_graphs(tmp_path)
    for graph in saved.graphs.values():
        assert graph.unit_ids == column.row_ids
        np.testing.assert_allclose(graph.weights(), rooted.weights(), atol=1e-7)
    with pytest.raises(ValueError, match="integer"):
        DistanceGraph.from_distances(
            pairwise, unit_ids=(True, -2, 4), weighting=weighting
        )
