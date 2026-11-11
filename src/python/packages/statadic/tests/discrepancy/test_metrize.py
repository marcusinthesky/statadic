"""Checked NumPy conversion for persisted energy-functional values.

Typed JAX metrization is tested at :func:`statadic.core.matrices.metrize_energy`,
where the ground, moment, estimator scheme, and origin cannot be erased.
"""

from __future__ import annotations

import numpy as np
import pytest
from statadic.discrepancy.metrize import (
    NegativeEnergyFunctionalError,
    NonfiniteEnergyFunctionalError,
    sqrt_energy_functional,
)

_TOLERANCE = 1e-12


@pytest.mark.unit
@pytest.mark.contract
def test_sqrt_energy_functional_retains_float32() -> None:
    """The boundary respects the input's floating precision."""
    assert sqrt_energy_functional(np.array([[0.0, 4.0]], dtype=np.float32)).dtype == (
        np.float32
    )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_sqrt_energy_functional_rejects_non_finite_values(bad: float) -> None:
    """A persisted functional with NaN/inf is corrupt, not merely out of domain.

    Args:
        bad: The non-finite value under test.

    """
    with pytest.raises(NonfiniteEnergyFunctionalError):
        sqrt_energy_functional(np.array([[0.0, bad], [bad, 0.0]]))


@pytest.mark.unit
@pytest.mark.contract
def test_sqrt_energy_functional_rejects_a_material_sign_error() -> None:
    """A materially negative entry is a sign error and must raise, not clamp."""
    with pytest.raises(NegativeEnergyFunctionalError):
        sqrt_energy_functional(np.array([[0.0, -1.0], [-1.0, 0.0]]))


@pytest.mark.unit
@pytest.mark.contract
def test_sqrt_energy_functional_clamps_within_the_relative_tolerance() -> None:
    """Rounding-scale undershoot is clamped; the tolerance is *relative*.

    ``-1e-12 * max|S|`` is the boundary, so the same absolute undershoot is
    rounding on a large functional and a sign error on a small one. Both sides
    are asserted, because a tolerance that had silently become absolute would
    pass the first case alone.
    """
    large = np.array([[0.0, 1e6], [-_TOLERANCE * 1e6 * 0.5, 0.0]])
    np.testing.assert_allclose(sqrt_energy_functional(large)[1, 0], 0.0)

    small = np.array([[0.0, 1e-6], [-1e-3, 0.0]])
    with pytest.raises(NegativeEnergyFunctionalError):
        sqrt_energy_functional(small)
