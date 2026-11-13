"""Numerical perturbation evidence for joint Gaussian graph-profile searches."""

import jax.numpy as jnp
import pytest
from statadic.estimation.spatial.graph_learning import _profile_sensitivity


@pytest.mark.unit
@pytest.mark.reference
def test_profile_screen_distinguishes_flat_and_active_bindings() -> None:
    """A known quadratic and a constant direction have different evidence."""
    coordinates = {"active": jnp.asarray(0.2), "flat": jnp.asarray(0.7)}
    result = _profile_sensitivity(
        jnp.empty(0),
        coordinates,
        jnp.asarray(5 - 0.2**2),
        20,
        lambda vector: 5 - vector[0] ** 2,
    )
    assert result == {"active": True, "flat": False}
