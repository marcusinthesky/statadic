"""Numerical contracts for target-directed kernel-MMD barycentre weights."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import config
from statadic.discrepancy.mmd import distance_induced_kernel
from statadic.discrepancy.mmd.energy import (
    UNIT_EXPONENT,
    energy_geometry,
)
from statadic.geometry.barycentre.mmd import (
    KernelBarycentreDiagnostics,
    kernel_barycentre_weights,
)
from statadic.geometry.ground.metrics import (
    EUCLIDEAN,
)
from statadic.geometry.ground.similarities import LINEAR_KERNEL, rbf_kernel
from statadic.sample.packed import PackedClouds, pack

config.update("jax_enable_x64", val=True)


_TARGET = jnp.array([[0.0]])
_CANDIDATES = jnp.array([[[-1.0]], [[1.0]]])

_MMD_TOLERANCE = 1e-10
_SIMPLEX_TOLERANCE = 1e-12


@pytest.mark.component
@pytest.mark.numerical
def test_linear_kernel_barycentre_is_symmetric() -> None:
    """The linear mean embedding gives equal weights for symmetric sources."""
    weights, diagnostics = kernel_barycentre_weights(
        PackedClouds.from_array(jnp.asarray(_TARGET)),
        pack(_CANDIDATES, require_positive_counts=True),
        kernel=LINEAR_KERNEL,
        return_diagnostics=True,
    )

    assert jnp.allclose(weights, jnp.array([0.5, 0.5]), atol=1e-7)
    assert isinstance(diagnostics, KernelBarycentreDiagnostics)
    assert diagnostics.converged
    assert diagnostics.mmd_squared <= _MMD_TOLERANCE
    assert diagnostics.psd_tangent_min_eigenvalue > 0.0
    assert diagnostics.simplex_sum_error <= _SIMPLEX_TOLERANCE


@pytest.mark.component
@pytest.mark.numerical
def test_rbf_kernel_barycentre_is_symmetric() -> None:
    """A declared RBF strategy uses the same simplex-QP contract."""
    weights = kernel_barycentre_weights(
        PackedClouds.from_array(jnp.asarray(_TARGET)),
        pack(_CANDIDATES, require_positive_counts=True),
        kernel=rbf_kernel(1.0),
    )

    assert jnp.allclose(weights, jnp.array([0.5, 0.5]), atol=1e-7)


@pytest.mark.component
@pytest.mark.numerical
def test_distance_induced_energy_kernel_uses_mmd_objective() -> None:
    """The Euclidean energy arm is admitted through its typed PSD strategy."""
    energy_kernel = distance_induced_kernel(energy_geometry(EUCLIDEAN, UNIT_EXPONENT))
    weights = kernel_barycentre_weights(
        PackedClouds.from_array(jnp.asarray(_TARGET)),
        pack(_CANDIDATES, require_positive_counts=True),
        kernel=energy_kernel,
    )

    assert jnp.allclose(weights, jnp.array([0.5, 0.5]), atol=1e-7)
