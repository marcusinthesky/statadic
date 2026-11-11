# standalone package-maintenance entry point
"""Independent finite-sum witnesses for parameter-influence covariance arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.operators.parameters import (
    cluster_parameter_covariance,
    coefficient_influence_rows,
    hac_parameter_covariance,
    independent_parameter_covariance,
)

from .._budgets import examples

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy
    from jax import Array


_N_ROWS = 5
_SMALLEST_FACTOR = 1e-100


def _normal_floats(bound: float) -> SearchStrategy[float]:
    """Draw floats whose pairwise products stay out of the subnormal range.

    XLA's CPU backend flushes subnormal results to zero while NumPy keeps them,
    so a product such as ``1e-136 * 1e-178`` differs by construction. Zero or a
    magnitude of at least ``1e-100`` keeps every product normal.
    """
    return st.floats(-bound, bound, allow_nan=False, allow_infinity=False).filter(
        lambda value: not value or abs(value) >= _SMALLEST_FACTOR
    )


_UNIT_ROWS = hnp.arrays(
    np.float64,
    (_N_ROWS, 2),
    elements=_normal_floats(1.0),
)
# Hat-matrix diagonals lie in [0, 1); keep them away from the singular end.
_LEVERAGE = hnp.arrays(
    np.float64,
    (_N_ROWS,),
    elements=_normal_floats(0.5).filter(lambda value: value >= 0.0),
)


def _rows() -> np.ndarray:
    """Return small, deliberately noncentered coefficient-influence rows."""
    return np.asarray([[0.2, -0.1], [0.5, 0.2], [-0.3, 0.6], [0.1, 0.4], [0.7, -0.2]])


@pytest.mark.unit
@pytest.mark.numerical
@examples(100)
@example(
    mapping=np.asarray([[0.2, 0.3, -0.1], [0.4, -0.2, 0.8]]),
    residuals=np.asarray([1.5, -0.7, 0.4]),
)
@given(
    mapping=hnp.arrays(
        np.float64,
        (2, 3),
        elements=_normal_floats(1.0),
    ),
    residuals=hnp.arrays(
        np.float64,
        (3,),
        elements=_normal_floats(2.0),
    ),
)
def test_coefficient_map_applies_each_whitened_residual_to_its_own_column(
    mapping: np.ndarray, residuals: np.ndarray
) -> None:
    """A row/parameter transpose error must change this nonsquare witness."""
    expected = np.asarray([mapping[:, i] * residuals[i] for i in range(3)])
    np.testing.assert_allclose(
        coefficient_influence_rows(jnp.asarray(mapping), jnp.asarray(residuals)),
        expected,
    )


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize(("kind", "power"), [("HC0", 0), ("HC2", 1), ("HC3", 2)])
@examples(100)
@example(rows=_rows(), leverage=np.asarray([0.1, 0.3, 0.2, 0.4, 0.15]))
@given(rows=_UNIT_ROWS, leverage=_LEVERAGE)
def test_independent_covariance_matches_explicit_outer_product_sum(
    rows: np.ndarray,
    leverage: np.ndarray,
    kind: Literal["HC0", "HC2", "HC3"],
    power: int,
) -> None:
    """Leverage adjustments apply to covariance, not to its squared scale twice."""
    expected = sum(
        np.outer(row, row) / (1.0 - h) ** power
        for row, h in zip(rows, leverage, strict=True)
    )
    actual = independent_parameter_covariance(
        jnp.asarray(rows), jnp.asarray(leverage), kind=kind
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


@pytest.mark.unit
@pytest.mark.reference
@examples(100)
@example(rows=_rows(), bandwidth=0)
@example(rows=_rows(), bandwidth=2)
@example(rows=_rows(), bandwidth=4)
@given(rows=_UNIT_ROWS, bandwidth=st.integers(0, _N_ROWS - 1))
def test_hac_covariance_matches_explicit_uncentered_bartlett_sum(
    rows: np.ndarray, bandwidth: int
) -> None:
    """Use raw rows and every finite lag pair with one declared Bartlett weight."""
    expected = sum(np.outer(row, row) for row in rows)
    for lag in range(1, bandwidth + 1):
        for current, previous in zip(rows[lag:], rows[:-lag], strict=True):
            cross = np.outer(current, previous)
            expected += (1.0 - lag / (bandwidth + 1)) * (cross + cross.T)
    actual = hac_parameter_covariance(jnp.asarray(rows), bandwidth)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


@pytest.mark.unit
@pytest.mark.reference
@examples(100)
@example(rows=_rows())
@given(rows=_UNIT_ROWS)
def test_cluster_covariance_matches_outer_products_of_group_sums(
    rows: np.ndarray,
) -> None:
    """Cluster contribution units are whole summed rows, with no hidden correction."""
    groups = np.asarray([0, 0, 1, 1, 2])
    expected = sum(
        np.outer(rows[groups == group].sum(axis=0), rows[groups == group].sum(axis=0))
        for group in range(3)
    )
    actual = cluster_parameter_covariance(
        jnp.asarray(rows), jnp.asarray(groups), n_clusters=3
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


@pytest.mark.integration
@pytest.mark.property
@examples(40)
@example(rows=_rows())
@given(rows=_UNIT_ROWS)
def test_influence_covariance_arrays_support_fixed_shape_jit_and_vmap(
    rows: np.ndarray,
) -> None:
    """Each lane keeps its own influence rows through the existing HAC primitive."""
    panels = jnp.stack([jnp.asarray(rows), 2.0 * jnp.asarray(rows)])

    def evaluate_hac_parameter_covariance(rows: jax.Array) -> Array:
        return hac_parameter_covariance(rows, 2)

    actual = jax.jit(jax.vmap(evaluate_hac_parameter_covariance))(panels)
    np.testing.assert_allclose(actual[1], 4.0 * actual[0], rtol=1e-13, atol=1e-14)


@pytest.mark.unit
@pytest.mark.contract
@examples(100)
@example(rows=_rows(), bandwidth=-1)
@example(rows=_rows(), bandwidth=5)
@given(
    rows=_UNIT_ROWS,
    bandwidth=st.one_of(
        st.integers(max_value=-1), st.integers(min_value=_N_ROWS, max_value=10**6)
    ),
)
def test_hac_rejects_a_lag_outside_the_retained_row_range(
    rows: np.ndarray, bandwidth: int
) -> None:
    """An invalid bandwidth cannot silently act as a different HAC recipe."""
    with pytest.raises(ValueError, match="bandwidth"):
        hac_parameter_covariance(jnp.asarray(rows), bandwidth)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("kind", ["HC2", "HC3"])
@examples(100)
@example(rows=_rows(), invalid_leverage=1.0)
@example(rows=_rows(), invalid_leverage=-0.1)
@example(rows=_rows(), invalid_leverage=float("nan"))
@given(
    rows=_UNIT_ROWS,
    invalid_leverage=st.one_of(
        st.just(float("nan")),
        st.floats(max_value=-0.1, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0, allow_nan=False, allow_infinity=False),
    ),
)
def test_invalid_adjusted_leverage_does_not_produce_finite_covariance(
    rows: np.ndarray, invalid_leverage: float, kind: Literal["HC2", "HC3"]
) -> None:
    """An undefined leverage denominator cannot silently become uncertainty."""
    leverage = jnp.asarray([invalid_leverage, 0.2, 0.3, 0.15, 0.25])
    covariance = independent_parameter_covariance(
        jnp.asarray(rows), leverage, kind=kind
    )
    assert np.isnan(np.asarray(covariance)).all()
