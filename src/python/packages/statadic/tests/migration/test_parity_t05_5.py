# jaxtyping shape strings are runtime contracts.
"""Parity test for t05.5 — fused scalar ``energy_distance`` vs the frozen oracle.

Confirms the fused single-jit scalar path (``statistics.energy_distance``,
which now dispatches to the fused ``_energy_distance_core``) still matches the
frozen t05.1 oracle's ``energy_distance_x_c0`` field within ``DEFAULT_ATOL``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import pytest
from statadic.discrepancy.mmd.energy import (
    energy_distance,
)
from statadic.sample.packed import PackedClouds

from ._parity import DEFAULT_ATOL, load_oracle

if TYPE_CHECKING:
    from statadic.geometry.ground.config import (
        GroundDistanceName,
    )


@pytest.mark.component
@pytest.mark.regression
def test_fused_energy_distance_matches_oracle() -> None:
    """Verify fused energy distance matches oracle."""
    oracle = load_oracle()
    x = jnp.asarray(oracle["x"])
    c0 = jnp.asarray(oracle["candidates"][0])
    exponent = float(oracle["exponent"])
    metric = cast("GroundDistanceName", str(oracle["metric"]))

    result = energy_distance(
        PackedClouds.from_array(jnp.asarray(x)),
        PackedClouds.from_array(jnp.asarray(c0)),
        exponent=exponent,
        metric=metric,
    )

    assert jnp.allclose(
        result, jnp.asarray(oracle["energy_distance_x_c0"]), atol=DEFAULT_ATOL
    ), "fused scalar energy_distance drifted beyond atol vs oracle"
