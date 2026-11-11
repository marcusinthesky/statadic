"""Change empirical-measure geometry while preserving atoms and row identities."""

# ruff: noqa: INP001, T201  # Standalone executable documentation.
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from statadic.api.representation import Energy
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    from statadic.api.representation import PairwiseGeometry


def compare() -> tuple[PairwiseGeometry, PairwiseGeometry]:
    """Return rooted V-energy matrices on two positively rescaled clouds."""
    x = jnp.array([[1.0], [2.0]])
    measures = PackedClouds.from_clouds([x, 3 * x])
    identities = ("original", "scaled")
    euclidean = Energy(ground="euclidean").prepare(measures, row_ids=identities)
    angular = Energy(ground="angular").prepare(measures, row_ids=identities)
    matrices = euclidean.pairwise(), angular.pairwise()
    # Raw Euclidean V-energy is 4, so the host's rooted geometry returns 2.
    np.testing.assert_allclose(matrices[0].values, [[0, 2], [2, 0]], atol=2e-5)
    np.testing.assert_allclose(matrices[1].values, 0, atol=2e-5)
    for matrix in matrices:
        np.testing.assert_equal(matrix.row_ids, identities)
    return matrices


if __name__ == "__main__":
    for name, matrix in zip(("Euclidean", "Angular"), compare(), strict=True):
        print(name, matrix.values, sep="\n")
    print("Angular ground describes direction; it cannot distinguish positive scale.")
