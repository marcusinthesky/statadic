"""Cross-owner covariance, diagnostic and reference composition contracts."""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.equations import (
    EquationDescriptor,  # noqa: TC002  # evaluated return annotation
    ParameterCovariance,  # noqa: TC002  # evaluated return annotation
    VariabilityScheme,  # noqa: TC002  # evaluated return annotation
    parameter_covariance,
    unsafe_assume_mean_equation_jacobian,
    unsafe_assume_variability_matrix,
)
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.inference.linear_diagnostics import breusch_godfrey, ols_influence
from statadic.inference.parameters import normal_critical_value
from scipy import stats

if TYPE_CHECKING:
    from statadic.core.equations import MeanEquationJacobian, VariabilityMatrix

pytestmark = [pytest.mark.unit, pytest.mark.reference]


def test_composed_covariance_rejects_amplified_negative_meat_lane() -> None:
    """Input-relative PSD tolerance cannot certify a negative output variance."""
    dtype = jnp.asarray(1.0).dtype
    eps = np.finfo(dtype).eps
    derivative = jnp.asarray([[0.0], [np.sqrt(eps) / 10]], dtype=dtype)
    matrices = jnp.asarray(
        [np.diag([1.0, -eps / 8]), np.diag([1.0, eps / 8]), np.diag([1.0, 0.0])],
        dtype=dtype,
    )

    def compose(
        omega: jax.Array,
    ) -> ParameterCovariance[EquationDescriptor, VariabilityScheme]:
        bread: MeanEquationJacobian[EquationDescriptor] = (
            unsafe_assume_mean_equation_jacobian(derivative)
        )
        meat: VariabilityMatrix[EquationDescriptor, VariabilityScheme] = (
            unsafe_assume_variability_matrix(omega)
        )
        return parameter_covariance(
            bread,
            meat,
            1,
            jnp.eye(2, dtype=dtype),
        )

    actual = jax.jit(jax.vmap(compose))(matrices)
    np.testing.assert_array_equal(actual.identified, [True, True, True])
    np.testing.assert_array_equal(actual.numerically_valid, [False, True, True])
    assert np.isnan(np.asarray(actual.values[0])).all()
    np.testing.assert_allclose(actual.values[1:], [[[12.5]], [[0.0]]], rtol=2e-5)


def test_bg_requires_every_requested_added_lag_restriction() -> None:
    """The independent augmented design adds one rank, not two restrictions."""
    design = np.column_stack((np.ones(4), [0.0, -1.0, 0.0, 1.0]))
    response = np.asarray([-1.0, 0.0, 1.0, 0.0])
    beta = np.linalg.lstsq(design, response, rcond=None)[0]
    residuals = response - design @ beta
    augmented = np.column_stack(
        (design, np.r_[0.0, residuals[:-1]], np.r_[0.0, 0.0, residuals[:-2]])
    )
    assert np.linalg.matrix_rank(augmented) - np.linalg.matrix_rank(design) == 1
    fit = LeastSquares().estimate(
        LinearProblem(jnp.asarray(design), jnp.asarray(response))
    )
    actual = breusch_godfrey(fit, 2)
    assert not bool(actual.valid)
    assert np.isnan(actual.statistic)
    assert np.isnan(actual.pvalue)


def test_degenerate_deletion_preserves_other_rows_and_full_fit_quantities() -> None:
    """Deleting the only nonzero response has no positive residual variance."""
    response = np.asarray([0.0, 0.0, 0.0, 1.0])
    design = np.ones((4, 1))
    fit = LeastSquares().estimate(
        LinearProblem(jnp.asarray(design), jnp.asarray(response))
    )
    actual = ols_influence(fit)
    assert bool(actual.valid)
    assert np.asarray(actual.leverage_valid).all()
    assert np.asarray(actual.studentized_internal_valid).all()
    assert np.asarray(actual.cooks_distance_valid).all()
    np.testing.assert_array_equal(
        actual.studentized_external_valid, [True, True, True, False]
    )
    np.testing.assert_array_equal(actual.dffits_valid, [True, True, True, False])
    np.testing.assert_array_equal(actual.dfbetas_valid[:, 0], [True, True, True, False])
    assert np.isnan(actual.studentized_external[-1])
    assert np.isnan(actual.dffits[-1])
    assert np.isnan(actual.dfbetas[-1]).all()
    # Independent leave-one-out variance for an available neighboring row.
    deleted = response[1:] - response[1:].mean()
    variance = deleted @ deleted / 2
    expected = (response[0] - response.mean()) / np.sqrt(variance * 0.75)
    np.testing.assert_allclose(actual.studentized_external[0], expected, rtol=2e-5)


def test_normal_critical_value_uses_representable_lower_tail() -> None:
    """The greatest representable confidence below one has a finite quantile."""
    one = jnp.asarray(1.0)
    confidence = jnp.nextafter(one, jnp.zeros_like(one))
    actual = jax.jit(normal_critical_value)(confidence)
    expected = stats.norm.isf((1 - float(confidence)) / 2)
    assert np.isfinite(actual)
    assert actual.dtype == confidence.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    invalid = normal_critical_value(jnp.asarray([0.0, 1.0, np.nan]))
    assert np.isnan(np.asarray(invalid)).all()
