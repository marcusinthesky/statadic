# standalone package-maintenance entry point
"""Caller-owned precision and numerical failure evidence."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from statadic.inference.dyadic import DyadicFitRequest, DyadicInference
from statadic.inference.wolak import (
    wolak_test,
)
from statadic.model.dyadic import DyadicModel, DyadicSample

from .._dyadic import fit_dyadic_report


def _dyadic_request() -> DyadicFitRequest[int]:
    """Build a minimal well-posed two-node dyadic fit request."""
    rng = np.random.default_rng(7)
    n = 24
    endpoint_i = tuple(node for node in range(4) for _ in range(6))
    endpoint_j = tuple(node for _ in range(6) for node in range(4, 8))
    regressor = rng.normal(size=n)
    return DyadicFitRequest(
        model=DyadicModel(
            ["energy"],
            {"energy": regressor},
            DyadicSample(
                endpoint_i=endpoint_i,
                endpoint_j=endpoint_j,
                outcome=0.3 * regressor + rng.normal(size=n),
            ),
        ),
        inference=DyadicInference(bootstrap_iters=4, bootstrap_seed=1),
    )


@pytest.mark.component
@pytest.mark.contract
def test_dyadic_fit_attempts_a_float32_caller() -> None:
    """A well-conditioned fit runs with caller-selected float32."""
    request = _dyadic_request()
    with jax.enable_x64(new_val=False):
        result = fit_dyadic_report(request)
        assert np.isfinite(result.sse_full)


@pytest.mark.component
@pytest.mark.contract
def test_wolak_test_attempts_a_float32_caller() -> None:
    """An attempted solve exposes its convergence failure, not a dtype veto."""
    moments = np.random.default_rng(11).normal(size=(64, 3))
    with (
        jax.enable_x64(new_val=False),
        pytest.raises(RuntimeError, match="nonnegative-QP solver rejected"),
    ):
        wolak_test(moments)
