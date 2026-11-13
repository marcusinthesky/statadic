# standalone package-maintenance entry point
"""T75.3: STATADIC HAC parity with statsmodels' public sandwich API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.equations import (
    mean_equation_jacobian,
    parameter_covariance,
    unsafe_assume_equation_jacobian,
    unsafe_assume_estimating_function,
)
from statadic.operators.longrun import (
    bartlett_weights,
    newey_west,
    newey_west_variability,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )
    from types import ModuleType


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
def test_bartlett_weights_match_statsmodels(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Both public APIs use ``w_l = 1 - l / (bandwidth + 1)``."""
    statsmodels = optional_reference("statsmodels")
    sandwich = optional_reference(f"{statsmodels.__name__}.stats.sandwich_covariance")

    bandwidth = 3
    observed = np.asarray(bartlett_weights(bandwidth))
    expected = sandwich.weights_bartlett(bandwidth)

    np.testing.assert_array_equal(observed, expected)


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
def test_newey_west_matches_statsmodels_hac_sandwich(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """STATADIC meat plus OLS bread matches statsmodels HAC without correction.

    The matched contract is the uncentered Bartlett HAC of OLS score moments
    ``x_t * residual_t`` at a fixed bandwidth.  STATADIC normalizes its meat by
    ``T`` while statsmodels' public ``cov_hac`` uses raw cross-products, so the
    comparison multiplies STATADIC's meat by ``T`` before applying the same OLS
    bread.  Both finite-sample correction and mean-centering are deliberately
    disabled; neither result is compared to STATADIC's Hall-centered estimator.
    """
    statsmodels = optional_reference("statsmodels")
    api = optional_reference(f"{statsmodels.__name__}.api")
    sandwich = optional_reference(f"{statsmodels.__name__}.stats.sandwich_covariance")

    design = np.asarray(
        [
            [1.0, -1.4],
            [1.0, -0.8],
            [1.0, -0.1],
            [1.0, 0.4],
            [1.0, 1.2],
            [1.0, 1.8],
            [1.0, 2.7],
            [1.0, 3.1],
        ],
        dtype=np.float64,
    )
    response = np.asarray([1.2, 0.1, 1.8, 0.7, 2.9, 1.5, 4.1, 2.6], dtype=np.float64)
    bandwidth = 2

    fit = api.OLS(response, design).fit()
    moments = np.asarray(fit.resid)[:, None] * design
    meat = np.asarray(newey_west(jnp.asarray(moments), bandwidth=bandwidth))
    bread = np.linalg.inv(design.T @ design)
    expected = bread @ (moments.shape[0] * meat) @ bread
    observed = sandwich.cov_hac(
        fit,
        nlags=bandwidth,
        weights_func=sandwich.weights_bartlett,
        use_correction=False,
    )

    # Both paths use float64; the bound allows only JAX/NumPy reduction-order
    # differences after the identical fixed-lag arithmetic.
    np.testing.assert_allclose(observed, expected, rtol=1e-11, atol=1e-13)

    # This is equation-contract validation using a statsmodels fit, before a
    # public STATADIC fitter exists. Scores and derivatives use the same retained
    # rows and the reported OLS parameter; neither bread route rebuilds T rows
    # from an already averaged derivative.
    derivative_panel = unsafe_assume_equation_jacobian(
        jnp.asarray(-np.einsum("ni,nj->nij", design, design))
    )
    typed_meat = newey_west_variability(
        unsafe_assume_estimating_function(jnp.asarray(moments)), bandwidth
    )
    for derivatives in (derivative_panel, mean_equation_jacobian(derivative_panel)):
        result = parameter_covariance(derivatives, typed_meat, len(response))
        assert result.identified
        assert result.numerically_valid
        np.testing.assert_allclose(result.values, observed, rtol=1e-11, atol=1e-13)

        np.testing.assert_allclose(
            parameter_covariance(
                derivatives,
                typed_meat,
                len(response),
                finite_sample_correction=len(response)
                / (len(response) - design.shape[1]),
            ).values,
            sandwich.cov_hac(
                fit,
                nlags=bandwidth,
                weights_func=sandwich.weights_bartlett,
                use_correction=True,
            ),
            rtol=1e-11,
            atol=1e-13,
        )
