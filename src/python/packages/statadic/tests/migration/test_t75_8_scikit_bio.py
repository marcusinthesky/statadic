# standalone package-maintenance entry point
"""T75.8 optional scikit-bio Mantel coefficient conformance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.association import Mantel, MantelInput
from statadic.geometry.ground.metrics import pdist

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )
    from types import ModuleType


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scikit-bio")
@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_mantel_coefficient_matches_scikit_bio(
    optional_reference: Callable[[str], ModuleType],
    reference_clouds: tuple[np.ndarray, np.ndarray],
    method: Literal["pearson", "spearman"],
) -> None:
    """STATADIC and scikit-bio agree on the observed upper-triangle coefficient."""
    skbio = optional_reference("skbio")
    x_cloud, y_cloud = reference_clouds
    # Three objects per cloud give matched rows while preserving non-constant
    # Euclidean distances; rows are observations in both implementations.
    x_matrix = np.array(pdist(jnp.asarray(x_cloud[:3])), dtype=np.float64, copy=True)
    y_matrix = np.array(pdist(jnp.asarray(y_cloud)), dtype=np.float64, copy=True)
    # Both APIs require an exactly hollow distance matrix; remove the measured
    # JAX norm diagonal residual before comparing the shared Mantel contract.
    np.fill_diagonal(x_matrix, 0.0)
    np.fill_diagonal(y_matrix, 0.0)
    ids = ["a", "b", "c"]
    x_distance = skbio.DistanceMatrix(x_matrix, ids=ids)
    y_distance = skbio.DistanceMatrix(y_matrix, ids=ids)

    expected, _pvalue, n_objects = skbio.stats.distance.mantel(
        x_distance,
        y_distance,
        method=method,
        permutations=0,
        seed=75,
    )
    actual = float(Mantel(method).estimate(MantelInput(x_matrix, y_matrix)).statistic)

    assert n_objects == len(ids)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
