# jaxtyping shape strings are runtime contracts.
"""T75.7: pairwise Euclidean distances compared with scikit-learn."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.geometry.ground.metrics import (
    EUCLIDEAN,
    cdist,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )
    from types import ModuleType


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scikit-learn")
def test_euclidean_cdist_matches_sklearn_pairwise_distances(
    optional_reference: Callable[[str], ModuleType],
    reference_clouds: tuple[np.ndarray, np.ndarray],
) -> None:
    """STATADIC's declared Euclidean strategy matches sklearn's pairwise contract."""
    optional_reference("sklearn")
    metrics = optional_reference("sklearn.metrics")
    x, y = reference_clouds

    expected = metrics.pairwise_distances(x, y, metric="euclidean")  # sklearn reference
    observed = np.asarray(cdist(jnp.asarray(x), jnp.asarray(y), metric=EUCLIDEAN))

    assert observed.shape == (x.shape[0], y.shape[0])
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
