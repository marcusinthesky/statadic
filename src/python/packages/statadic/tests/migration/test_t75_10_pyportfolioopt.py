# standalone package-maintenance entry point
"""T75.10: STATADIC sample covariance against PyPortfolioOpt's risk matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from statadic.estimation.covariance import (
    sample_covariance,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )
    from types import ModuleType


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("PyPortfolioOpt")
def test_sample_covariance_matches_pyportfolioopt_risk_matrix(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """The named ``sample_cov`` risk model preserves returns covariance semantics."""
    pypfopt = optional_reference("pypfopt")
    risk_models = pypfopt.risk_models

    # Paired full-rank rows give a deterministic, positive-definite covariance;
    # the explicit scale keeps this in the range of ordinary asset returns.
    base = np.asarray(
        [
            [1.0, 0.2, -0.1],
            [0.0, 1.2, 0.3],
            [0.4, -0.2, 1.1],
            [-0.7, 0.5, -0.4],
        ],
        dtype=np.float64,
    )
    returns = np.vstack((base, -base)) * 1e-2

    expected = np.asarray(
        risk_models.risk_matrix(
            returns,
            method="sample_cov",
            returns_data=True,
            frequency=1,
        )
    )
    got = np.asarray(sample_covariance(returns, ddof=1))

    # Both paths are float64 sample covariance with no annualisation; retain a
    # tight tolerance so a divisor, centering, or feature-order regression fails.
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)
