"""Shared numerical permutations preserve fitted state and invalid-lane gates."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.inference._row_order import _validated_row_order
from statadic.inference.linear_covariance import hac_covariance
from statadic.inference.linear_diagnostics import breusch_godfrey, durbin_watson

if TYPE_CHECKING:
    from statadic.core.equations import PlainLongRun
    from statadic.estimation.linear import LinearFit
    from statadic.inference.linear_covariance import LinearUncertainty
    from statadic.inference.linear_diagnostics import ResidualTest

pytestmark = pytest.mark.unit


@pytest.mark.contract
def test_unsigned_permutations_keep_integer_gather_indices() -> None:
    """Unsigned indices cannot promote to floats when combined with signed identity."""
    dtype = jnp.uint64 if jax.config.x64_enabled else jnp.uint32
    order, valid = jax.jit(partial(_validated_row_order, observations=2))(
        jnp.asarray([1, 0], dtype=dtype)
    )
    assert bool(valid)
    assert jnp.issubdtype(order.dtype, jnp.integer)
    np.testing.assert_array_equal(jnp.asarray([2.0, 3.0])[order], [3.0, 2.0])
    rejected, valid = _validated_row_order(jnp.asarray([0, 3], dtype=dtype), 2)
    assert not bool(valid)
    np.testing.assert_array_equal(rejected, [0, 1])


def _fit() -> LinearFit:
    x = jnp.column_stack((jnp.ones(12), jnp.arange(12.0)))
    y = jnp.asarray([2.0, 0.0, 4.0, 1.0, 3.0, 7.0, 4.0, 8.0, 9.0, 6.0, 12.0, 9.0])
    return LeastSquares().estimate(LinearProblem(x, y, jnp.linspace(0.5, 2.0, 12)))


@pytest.mark.reference
def test_hac_orders_complete_weighted_influences_without_replacing_fit() -> None:
    """Bartlett products use reordered WLS influences, not reordered residuals alone."""
    fit = _fit()
    order = jnp.asarray([7, 2, 11, 0, 5, 4, 1, 10, 3, 9, 6, 8])
    result = hac_covariance(fit, 2, row_order=order)
    x, y, weights = map(np.asarray, (fit.design, fit.outcome, fit.relative_precision))
    inverse = np.linalg.inv(x.T @ (weights[:, None] * x))
    beta = np.linalg.solve(x.T @ (weights[:, None] * x), x.T @ (weights * y))
    influences = ((weights * (y - x @ beta))[:, None] * x) @ inverse
    influences = influences[np.asarray(order)]
    expected = influences.T @ influences
    for lag in (1, 2):
        cross = influences[lag:].T @ influences[:-lag]
        expected += (1.0 - lag / 3.0) * (cross + cross.T)
    tolerance = 2e-4 if not jax.config.x64_enabled else 2e-10
    np.testing.assert_allclose(result.covariance.values, expected, rtol=tolerance)
    assert bool(result.valid)
    assert result.fit is fit
    np.testing.assert_array_equal(result.fit.design, x)
    np.testing.assert_allclose(
        hac_covariance(fit, 0, row_order=order).covariance.values,
        hac_covariance(fit, 0).covariance.values,
        rtol=tolerance,
    )


@pytest.mark.contract
def test_invalid_permutation_is_unavailable_under_jit_and_vmap() -> None:
    """Repeated, negative and out-of-range indices cannot become a calibrated lane."""
    weighted = _fit()
    fit = LeastSquares().estimate(LinearProblem(weighted.design, weighted.outcome))
    orders = jnp.stack(
        (
            jnp.arange(12),
            jnp.zeros(12, dtype=int),
            jnp.arange(12) - 1,
            jnp.arange(12) + 1,
        )
    )

    def ordered_bg(order: jax.Array) -> ResidualTest:
        return breusch_godfrey(fit, row_order=order)

    def ordered_dw(order: jax.Array) -> ResidualTest:
        return durbin_watson(fit, row_order=order)

    def ordered_hac(order: jax.Array) -> LinearUncertainty[PlainLongRun]:
        return hac_covariance(fit, 1, row_order=order)

    bg = jax.jit(jax.vmap(ordered_bg))(orders)
    dw = jax.jit(jax.vmap(ordered_dw))(orders)
    hac = jax.jit(jax.vmap(ordered_hac))(orders)
    for result in (bg, dw, hac):
        np.testing.assert_array_equal(result.valid, [True, False, False, False])
    assert bool(jnp.all(jnp.isnan(bg.statistic[1:])))
    assert bool(jnp.all(jnp.isnan(dw.statistic[1:])))
    assert bool(jnp.all(jnp.isnan(hac.covariance.values[1:])))


@pytest.mark.contract
@pytest.mark.parametrize(
    "order",
    [jnp.arange(11), jnp.ones((12, 1)), jnp.arange(12.0), jnp.ones(12, dtype=bool)],
)
def test_permutation_structure_raises(order: jax.Array) -> None:
    """Indices have one integer axis; data-domain invalidity is handled separately."""
    fit = _fit()
    with pytest.raises(ValueError, match="integer vector"):
        durbin_watson(fit, row_order=order)
    with pytest.raises(ValueError, match="integer vector"):
        hac_covariance(fit, 1, row_order=order)
