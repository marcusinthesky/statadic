# jaxtyping shape strings are runtime contracts.
"""T75.6: exact one-dimensional transport parity with POT."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.discrepancy.transport import (
    w1_1d,
    w2_1d,
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
@pytest.mark.requires_optional("POT")
def test_w1_1d_matches_pot_exact_emd(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """STATADIC W1 agrees with POT's unregularized empirical transport cost."""
    ot = optional_reference("ot")
    x = np.asarray([-2.0, 0.5, 1.25, 4.0], dtype=np.float64)
    y = np.asarray([-1.0, 1.0, 2.5, 3.0], dtype=np.float64)
    weights = np.full(x.size, 1.0 / x.size, dtype=np.float64)
    cost = np.abs(x[:, None] - y[None, :])

    expected = float(ot.emd2(weights, weights, cost))
    observed = float(
        w1_1d(
            PackedClouds.from_array(jnp.asarray(jnp.asarray(x))[..., None]),
            PackedClouds.from_array(jnp.asarray(jnp.asarray(y))[..., None]),
        )
    )

    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("POT")
def test_w2_1d_matches_sqrt_pot_exact_squared_emd(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """STATADIC W2 agrees with the square root of POT's squared transport cost."""
    ot = optional_reference("ot")
    x = np.asarray([-1.5, -0.25, 2.0, 3.75], dtype=np.float64)
    y = np.asarray([-2.0, 0.5, 1.25, 4.5], dtype=np.float64)
    weights = np.full(x.size, 1.0 / x.size, dtype=np.float64)
    cost = (x[:, None] - y[None, :]) ** 2

    expected = float(np.sqrt(ot.emd2(weights, weights, cost)))
    observed = float(
        w2_1d(
            PackedClouds.from_array(jnp.asarray(jnp.asarray(x))[..., None]),
            PackedClouds.from_array(jnp.asarray(jnp.asarray(y))[..., None]),
        )
    )

    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
