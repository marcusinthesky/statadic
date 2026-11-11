"""Independent one-dimensional barycentre objectives and eager domain failures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.geometry.barycentre.wasserstein import (
    solve_wasserstein_free_support_barycentre,
    solve_wasserstein_measure_barycentre,
)
from statadic.sample.packed import PackedClouds, pack

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("POT")
def test_three_source_free_support_matches_quantile_barycentre_and_pot(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """For 1D uniform W2, sorted quantile means give a global analytic optimum."""
    peer = optional_reference("ot")
    sources = np.asarray(
        [[[-1.0], [2.0], [0.0]], [[1.0], [4.0], [-2.0]], [[6.0], [-3.0], [1.0]]]
    )
    weights = np.asarray([0.2, 0.3, 0.5])
    sorted_sources = np.sort(sources[:, :, 0], axis=1)
    expected_support = weights @ sorted_sources
    expected_objective = float(
        np.sum(weights[:, None] * (sorted_sources - expected_support) ** 2) / 3
    )
    reference = peer.lp.free_support_barycenter(
        [source.copy() for source in sources],
        [np.full(3, 1 / 3)] * 3,
        sources[0].copy(),
        weights=weights,
        stopThr=1e-12,
    )

    result = solve_wasserstein_free_support_barycentre(
        pack(jnp.asarray(sources), require_positive_counts=True),
        weights,
        p=2.0,
        tol=1e-10,
    )

    assert result.converged
    assert np.isfinite(result.support_values).all()
    np.testing.assert_allclose(
        np.sort(np.asarray(result.support_values)[:, 0]),
        expected_support,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.sort(reference[:, 0]), expected_support, rtol=1e-10, atol=1e-10
    )
    assert result.objective == pytest.approx(expected_objective, rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(result.source_weights, weights, rtol=1e-13, atol=1e-13)
    # Sorted quantile matching makes the fixed-assignment first derivative zero.
    residual = weights @ (
        np.sort(np.asarray(result.support_values)[:, 0]) - sorted_sources
    )
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("weights", [[-0.1, 1.1], [0.0, 0.0], [np.nan, 1.0], [1.0]])
@pytest.mark.parametrize(
    "solver",
    [solve_wasserstein_measure_barycentre, solve_wasserstein_free_support_barycentre],
)
def test_measure_barycentre_rejects_invalid_source_weights(
    weights: list[float],
    solver: Callable[..., object],
) -> None:
    """Source-mixture weights have a simplex contract, separate from atom masses."""
    sources = np.asarray([[[-1.0], [2.0]], [[0.0], [3.0]]])

    with pytest.raises(ValueError, match="source weights"):
        solver(
            PackedClouds.from_array(jnp.asarray(sources)), np.asarray(weights), p=2.0
        )
