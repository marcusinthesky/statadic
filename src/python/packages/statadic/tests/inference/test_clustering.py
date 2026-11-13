"""Tests for :mod:`statadic.inference.clustering` — moment-column clustering."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.inference.clustering import (
    cluster_pairs,
)

from .._budgets import examples
from ._moments import (
    SECOND_NONEMPTY_CLUSTER_LABEL,
    TWO_MOMENT_DOF,
)

# ---------------------------------------------------------------------------
# cluster_pairs
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.numerical
@examples(100)
@example(
    pair_moments=np.random.default_rng(4).standard_normal((100, 12)),
    assignments=np.repeat(np.arange(3), 4),
)
@given(
    pair_moments=hnp.arrays(
        np.float64,
        (20, 12),
        elements={"min_value": -1e3, "max_value": 1e3, "allow_nan": False},
    ),
    assignments=st.permutations(np.repeat(np.arange(3), 4).tolist()).map(np.array),
)
def test_cluster_pairs_averages_assigned_columns(
    pair_moments: np.ndarray, assignments: np.ndarray
) -> None:
    """Clustered moments average exactly the columns assigned to each pair.

    Args:
        pair_moments: Generated moment matrix, one column per pair.
        assignments: Cluster label per pair, four pairs in each of three clusters.

    """
    out = cluster_pairs(pair_moments, assignments, n_clusters=3)
    assert out.shape == (pair_moments.shape[0], 3)
    for label in range(3):
        expected = pair_moments[:, assignments == label].mean(axis=1)
        np.testing.assert_allclose(out[:, label], expected, rtol=1e-9, atol=1e-9)


@pytest.mark.component
@pytest.mark.contract
def test_cluster_pairs_drops_empty_bins() -> None:
    """Empty cluster bins must not create artificial zero moment columns.

    A cluster label in ``[0, n_clusters)`` that receives no pairs must
    NOT produce an all-zero moment column (which silently rank-deficits the
    downstream HAC).  Empty bins are dropped and the result re-indexed.
    """
    rng = np.random.default_rng(40)
    projection = 8
    pair_moments = rng.standard_normal((100, projection))
    # Only labels 0 and 2 used; label 1 and 3 are EMPTY (n_clusters=4).
    assignments = np.array([0, 0, 0, 0, 2, 2, 2, 2])
    out = cluster_pairs(pair_moments, assignments, n_clusters=4)
    # Two non-empty clusters -> exactly two columns, none all-zero.
    assert out.shape == (100, 2)
    assert not np.any(np.all(np.equal(out, 0.0), axis=0)), "no all-zero column allowed"
    # Column order follows ascending non-empty label (0 then 2).
    assert np.allclose(out[:, 0], pair_moments[:, assignments == 0].mean(axis=1))
    assert np.allclose(
        out[:, 1],
        pair_moments[:, assignments == SECOND_NONEMPTY_CLUSTER_LABEL].mean(axis=1),
    )
    # A full-rank HAC-ready matrix: no zero column so covariance is non-singular.
    cov = np.cov(out, rowvar=False)
    assert np.linalg.matrix_rank(cov) == TWO_MOMENT_DOF


@pytest.mark.component
@pytest.mark.contract
@examples(100)
@example(assignments=np.array([5, 5, 6, 6]))
@given(assignments=st.lists(st.integers(3, 50), min_size=4, max_size=4).map(np.array))
def test_cluster_pairs_all_empty_raises(assignments: np.ndarray) -> None:
    """Clustering omits empty bins instead of introducing zero moment columns.

    Args:
        assignments: Labels all outside ``[0, n_clusters)``, so no bin is filled.

    """
    pair_moments = np.random.default_rng(41).standard_normal((10, 4))
    with pytest.raises(ValueError, match="no cluster received any pair"):
        cluster_pairs(pair_moments, assignments, n_clusters=3)
