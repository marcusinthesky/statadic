"""Independent entropic objectives with explicit product-mass KL conventions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.representation import SinkhornDivergence
from statadic.discrepancy.transport.sinkhorn import sinkhorn_cost
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("POT")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_product_mass_kl_cost_and_debiased_geometry_match_pot(  # noqa: PLR0914 -- independent objectives and matched host/native controls
    optional_reference: Callable[[str], ModuleType], dtype: type
) -> None:
    """POT supplies the plan; explicit KL terms match STATADIC's dual objective.

    POT's ``sinkhorn2`` omits entropy, so its transport-only scalar would compare
    different quantities. Fixed marginals make the product-mass correction a
    constant in the plan optimization, but not in the reported objective.
    """
    peer = optional_reference("ot")
    left = np.asarray([[-1.0], [0.1], [1.2]], dtype=dtype)
    right = np.asarray([[-0.7], [0.4], [1.8], [2.3]], dtype=dtype)
    mass_left = np.asarray([0.2, 0.3, 0.5], dtype=dtype)
    mass_right = np.asarray([0.15, 0.25, 0.35, 0.25], dtype=dtype)
    epsilon = 0.7
    tolerance = 3e-5 if dtype is np.float32 else 2e-10
    residual_tolerance = 3e-6 if dtype is np.float32 else 1e-11

    def independent_cost(
        x: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray
    ) -> float:
        # The peer solves in float64 on exactly the caller's rounded inputs.
        x, y, a, b = (value.astype(np.float64) for value in (x, y, a, b))
        a, b = a / a.sum(), b / b.sum()
        costs = np.sum((x[:, None] - y[None, :]) ** 2, axis=-1)
        plan = peer.sinkhorn(
            a,
            b,
            costs,
            epsilon,
            method="sinkhorn_log",
            numItermax=2000,
            stopThr=1e-13,
            warn=True,
        )
        np.testing.assert_allclose(plan.sum(axis=1), a, atol=1e-12, rtol=0)
        np.testing.assert_allclose(plan.sum(axis=0), b, atol=1e-12, rtol=0)
        product = a[:, None] * b[None, :]
        entropy = np.sum(plan * np.log(plan / product) - plan + product)
        return float(np.sum(plan * costs) + epsilon * entropy)

    cross = independent_cost(left, right, mass_left, mass_right)
    expected = (
        cross
        - (
            independent_cost(left, left, mass_left, mass_left)
            + independent_cost(right, right, mass_right, mass_right)
        )
        / 2
    )
    with jax.enable_x64():
        a, b = jnp.asarray(mass_left), jnp.asarray(mass_right)
        a, b = a / a.sum(), b / b.sum()
        source = PackedClouds.from_points(jnp.asarray(left), a)
        target = PackedClouds.from_points(jnp.asarray(right), b)
        actual = sinkhorn_cost(
            source,
            target,
            epsilon=epsilon,
            iterations=256,
            tolerance=residual_tolerance,
        )
        assert actual.dtype == source.data.dtype
        np.testing.assert_allclose(actual, cross, atol=tolerance, rtol=tolerance)
        geometry = SinkhornDivergence(
            epsilon, iterations=256, tolerance=residual_tolerance
        )
        source_batch = PackedClouds.from_clouds([source.data], [a])
        target_batch = PackedClouds.from_clouds([target.data], [b])
        squared = geometry.squared_cross(source_batch, target_batch)
        rooted = geometry.cross(source_batch, target_batch)
        np.testing.assert_allclose(
            squared, [[expected]], atol=tolerance, rtol=tolerance
        )
        np.testing.assert_allclose(
            rooted, [[np.sqrt(expected)]], atol=tolerance, rtol=tolerance
        )
