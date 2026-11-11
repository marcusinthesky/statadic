# jaxtyping shape strings are runtime contracts.
"""T75.2: Euclidean energy-distance parity with dcor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.discrepancy.mmd.energy import (
    energy_distance,
)
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )
    from types import ModuleType


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("dcor")
def test_energy_distance_matches_dcor(
    optional_reference: Callable[[str], ModuleType],
    reference_clouds: tuple[np.ndarray, np.ndarray],
) -> None:
    """STATADIC's V-statistic energy distance agrees with dcor's default."""
    dcor = optional_reference("dcor")
    x, y = reference_clouds

    expected = float(dcor.energy_distance(x, y, exponent=1.0))
    observed = float(
        energy_distance(
            PackedClouds.from_array(jnp.asarray(jnp.asarray(x))),
            PackedClouds.from_array(jnp.asarray(jnp.asarray(y))),
            exponent=1.0,
            metric="euclidean",
        )
    )

    # JAX's pairwise norm leaves a measured 4.2e-8 diagonal residual on this
    # non-integer fixture; after the V-statistic reduction the observed
    # float64 difference is 3.8e-9. Keep the bound tight on the 0.34 result
    # while accounting for that backend-specific structural roundoff.
    np.testing.assert_allclose(observed, expected, rtol=2e-8, atol=1e-8)
