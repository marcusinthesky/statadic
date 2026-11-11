"""Fixed-roster union-neighbor graphs and frozen graph-distance extension.

Edges carry original distances, including zero-length edges between distinct
objects. Missing edges are infinity. No disconnected-component completion or
query-to-query shortcuts are introduced. Neighborhood changes and shortest-path
ties have no general smooth derivative contract.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from statadic.core.typing import Array, Int  # noqa: TC001 -- runtime annotations

__all__ = ["geodesic_cross_distances", "knn_shortest_paths"]


def _relax_paths(index: Int[Array, ""], paths: Array) -> Array:  # noqa: F722
    """Relax paths through one integer-array vertex inside the native loop."""
    return jnp.minimum(paths, paths[:, index, None] + paths[None, index, :])


def knn_shortest_paths(distances: Array, *, neighbors: int, tie_codes: Array) -> Array:
    """Return all-pairs paths in the undirected union-kNN distance graph.

    ``neighbors`` is a static shape choice in transformed calls. Identity codes
    resolve equal-distance neighbors, and the diagonal is excluded by index.
    Disconnected pairs remain infinity for the eager caller to diagnose.
    """
    distances = distances / 2 + distances.T / 2
    count = distances.shape[0]
    diagonal = jnp.eye(count, dtype=bool)
    codes = jnp.broadcast_to(tie_codes, distances.shape)
    order = jnp.lexsort((codes, distances, diagonal), axis=-1)[:, :neighbors]
    edges = jnp.zeros_like(diagonal).at[jnp.arange(count)[:, None], order].set(True)
    edges = edges | edges.T
    graph = jnp.where(edges, distances, jnp.inf)
    graph = graph.at[jnp.diag_indices(count)].set(0)
    return jax.lax.fori_loop(
        0,
        count,
        _relax_paths,
        graph,
    )


def geodesic_cross_distances(
    distances: Array, training_paths: Array, *, neighbors: int, tie_codes: Array
) -> Array:
    """Attach queries independently to frozen training paths.

    Each output row minimizes its neighbor-edge length plus a training-only
    path. Mapping queries separately avoids a queries-by-neighbors-by-training
    temporary and prevents one query from changing another's coordinates.
    """

    def extend(row: Array) -> Array:
        order = jnp.lexsort((tie_codes, row))[:neighbors]
        return jnp.min(training_paths[order] + row[order, None], axis=0)

    return jax.lax.map(extend, distances)
