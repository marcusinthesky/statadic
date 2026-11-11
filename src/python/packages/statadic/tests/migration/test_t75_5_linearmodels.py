# standalone package-maintenance entry point
"""T75.5: linearmodels HAC moment-covariance conformance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from statadic.operators.longrun import hall_centered_hac, newey_west

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )
    from types import ModuleType


_MOMENTS = np.asarray(
    [
        [1.20, -0.40],
        [-0.30, 0.90],
        [0.80, 0.20],
        [1.70, -1.10],
        [-0.60, 1.40],
        [0.10, -0.80],
        [1.30, 0.60],
        [-1.00, 1.10],
        [0.50, -0.20],
        [1.90, 0.30],
        [-0.20, 1.70],
        [0.70, -0.90],
    ],
    dtype=np.float64,
)
_BANDWIDTH = 2


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("linearmodels")
@pytest.mark.parametrize("centered", [False, True], ids=["plain", "centered"])
def test_hac_moment_covariance_matches_linearmodels(
    centered: Literal[False, True],
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Match linearmodels' fixed-Bartlett HAC score covariance.

    The conformance contract is deliberately limited to the HAC moment
    covariance (``s``), not fitted coefficients: STATADIC exposes the HAC/GMM
    operator while linearmodels' panel and IV estimators are different model
    contracts.  For the centered branch, the moments are pre-centered before
    passing them to linearmodels because its kernel covariance path does not
    apply the public ``center`` flag; this is equivalent to STATADIC's
    ``hall_centered_hac`` contract.
    """
    linearmodels = optional_reference("linearmodels")
    covariance_module = optional_reference(
        f"{linearmodels.__name__}.asset_pricing.covariance"
    )
    moments = _MOMENTS - _MOMENTS.mean(axis=0, keepdims=True) if centered else _MOMENTS
    covariance = covariance_module.KernelCovariance(
        moments,
        jacobian=np.eye(_MOMENTS.shape[1]),
        kernel="bartlett",
        bandwidth=_BANDWIDTH,
        center=False,
        debiased=False,
    )
    expected = np.asarray(covariance.s)
    actual = np.asarray(
        hall_centered_hac(_MOMENTS, bandwidth=_BANDWIDTH)
        if centered
        else newey_west(_MOMENTS, bandwidth=_BANDWIDTH)
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
