# jaxtyping shape strings are runtime contracts.
"""T75.9 optional arch Bartlett HAC conformance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from statadic.operators.longrun import hall_centered_hac, newey_west

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )
    from types import ModuleType


_BANDWIDTH = 2


def _matched_series(reference_clouds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Build one deterministic two-moment process from the shared fixture."""
    return np.vstack(reference_clouds).astype(np.float64, copy=False)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("arch")
def test_centered_hac_matches_arch_bartlett(
    optional_reference: Callable[[str], ModuleType],
    reference_clouds: tuple[np.ndarray, np.ndarray],
) -> None:
    """Match arch's centered Bartlett long-run covariance at fixed bandwidth.

    Both public estimators use the sample-mean-centered moment process, divide
    lag products by ``T`` (``df_adjust=0``), and apply Bartlett weights
    ``1 - h / (b + 1)`` through lag ``b``.  Automatic bandwidth selection is
    intentionally out of scope: arch's plug-in selector is not STATADIC's
    Newey--West rule.
    """
    arch = optional_reference("arch")
    bartlett = optional_reference(f"{arch.__name__}.covariance.kernel").Bartlett
    series = _matched_series(reference_clouds)
    expected_estimator = bartlett(
        series,
        bandwidth=_BANDWIDTH,
        df_adjust=0,
        center=True,
    )

    actual = np.asarray(hall_centered_hac(series, bandwidth=_BANDWIDTH))
    expected = np.asarray(expected_estimator.cov.long_run)

    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("arch")
def test_uncentered_hac_matches_arch_bartlett(
    optional_reference: Callable[[str], ModuleType],
    reference_clouds: tuple[np.ndarray, np.ndarray],
) -> None:
    """Match arch's uncentered Bartlett covariance for mean-zero moments."""
    arch = optional_reference("arch")
    bartlett = optional_reference(f"{arch.__name__}.covariance.kernel").Bartlett
    series = _matched_series(reference_clouds)
    expected_estimator = bartlett(
        series,
        bandwidth=_BANDWIDTH,
        df_adjust=0,
        center=False,
    )

    actual = np.asarray(newey_west(series, bandwidth=_BANDWIDTH))
    expected = np.asarray(expected_estimator.cov.long_run)

    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
