"""Independent references in isolated caller-selected JAX precision processes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.association.mantel import mantel_correlation_kernel
from statadic.core.equations import PlainLongRun
from statadic.discrepancy.mmd.energy.weighted import weighted_energy_cross
from statadic.discrepancy.transport.exact import balanced_wasserstein_from_cost_result
from statadic.estimation.glm import PoissonMLE, PoissonProblem
from statadic.estimation.gmm import FixedWeightGMM
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.estimation.moments import AffineMomentProblem
from statadic.inference.gmm.covariance import gmm_covariance
from statadic.inference.linear_covariance import classical_covariance, hc_covariance
from statadic.inference.parameters import coefficient_intervals
from statadic.sample.packed import PackedClouds

from .._validation.precision import bfloat16_qr_error

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

pytestmark = [pytest.mark.external_reference, pytest.mark.requires_optional("scipy")]


@pytest.fixture(autouse=True)
def record_precision(record_property: Callable[[str, object], None]) -> None:
    """Retain the effective process configuration beside each comparison."""
    record_property("jax_enable_x64", jax.config.x64_enabled)
    record_property("comparison_dtype", jnp.dtype(_dtype()).name)
    record_property("jax_backend", jax.default_backend())


def _dtype() -> type:
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


def _tolerance() -> float:
    # These well-conditioned, unit-scale calculations use fewer than 20 rows.
    return 2e-5 if not jax.config.x64_enabled else 2e-11


@pytest.mark.integration
@pytest.mark.reference
def test_linear_fit_and_student_interval_match_scipy(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Match coefficients, inference and the interval's zero-exclusion decision."""
    stats = optional_reference("scipy.stats")
    x = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.asarray([-1.1, -0.3, 0.2, 1.2, 1.7, 2.7, 3.3, 4.2])
    dtype = _dtype()
    design = jnp.asarray(np.column_stack([np.ones_like(x), x]), dtype=dtype)
    outcome = jnp.asarray(y, dtype=dtype)
    fit = LeastSquares().estimate(LinearProblem(design, outcome))
    reference = stats.linregress(x, y)
    uncertainty = classical_covariance(fit)
    interval = coefficient_intervals(
        uncertainty,
        reference="t",
        degrees_of_freedom=jnp.asarray(len(x) - 2, dtype=dtype),
    )
    assert bool(fit.valid)
    assert bool(fit.solver.converged)
    assert bool(interval.valid)
    assert fit.parameters.dtype == design.dtype
    expected = np.asarray([reference.intercept, reference.slope])
    np.testing.assert_allclose(
        fit.parameters, expected, rtol=_tolerance(), atol=_tolerance()
    )
    np.testing.assert_allclose(
        interval.estimate.se[1], reference.stderr, rtol=_tolerance(), atol=_tolerance()
    )
    critical = stats.t.ppf(0.975, len(x) - 2)
    expected_lower = reference.slope - critical * reference.stderr
    np.testing.assert_allclose(
        interval.estimate.ci[0][1], expected_lower, rtol=_tolerance(), atol=_tolerance()
    )
    assert bool(interval.estimate.ci[0][1] > 0) == bool(expected_lower > 0)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_tied_correlation_matches_scipy_in_caller_precision(
    optional_reference: Callable[[str], ModuleType], method: str
) -> None:
    """Tied ranks and correlation signs survive each supported process dtype."""
    stats = optional_reference("scipy.stats")
    x = np.asarray([1.0, 2.0, 2.0, 4.0, 5.0])
    y = np.asarray([3.0, 1.0, 1.0, 4.0, 6.0])
    expected = (stats.pearsonr if method == "pearson" else stats.spearmanr)(
        x, y
    ).statistic
    a, b = jnp.asarray(x, dtype=_dtype()), jnp.asarray(y, dtype=_dtype())

    def evaluate_mantel_correlation_kernel(
        left: jax.Array, right: jax.Array
    ) -> jax.Array:
        return mantel_correlation_kernel(left, right, method)

    actual = jax.jit(evaluate_mantel_correlation_kernel)(a, b)
    assert actual.dtype == a.dtype
    np.testing.assert_allclose(actual, expected, rtol=_tolerance(), atol=_tolerance())
    assert bool(actual > 0) == bool(expected > 0)


@pytest.mark.unit
@pytest.mark.reference
def test_weighted_energy_matches_scipy_in_caller_precision(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Independent weighted V-statistic uses the same masses and root convention."""
    stats = optional_reference("scipy.stats")
    x, y = np.asarray([-1.0, 0.5, 2.0]), np.asarray([0.0, 1.5, 4.0])
    a, b = np.asarray([0.25, 0.5, 0.25]), np.asarray([0.5, 0.25, 0.25])
    dtype = _dtype()
    left = PackedClouds.from_clouds(
        [jnp.asarray(x[:, None], dtype=dtype)], [jnp.asarray(a, dtype=dtype)]
    )
    right = PackedClouds.from_clouds(
        [jnp.asarray(y[:, None], dtype=dtype)], [jnp.asarray(b, dtype=dtype)]
    )
    actual = jax.jit(weighted_energy_cross)(left, right)
    expected = stats.energy_distance(x, y, a, b)
    assert actual.dtype == left.data.dtype
    np.testing.assert_allclose(
        actual, [[expected]], rtol=_tolerance(), atol=_tolerance()
    )
    assert bool(actual[0, 0] > 0)


@pytest.mark.integration
@pytest.mark.reference
def test_bfloat16_pure_correlation_and_backend_solve(
    optional_reference: Callable[[str], ModuleType],
    record_property: Callable[[str, object], None],
) -> None:
    """Attempt bf16 pure arithmetic and QR; a backend may reject bf16 QR."""
    stats = optional_reference("scipy.stats")
    x = jnp.asarray([0.0, 1.0, 2.0, 3.0], dtype=jnp.bfloat16)
    y = jnp.asarray([0.0, 1.0, 1.0, 3.0], dtype=jnp.bfloat16)
    expected = stats.pearsonr(
        np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    ).statistic

    def evaluate_mantel_correlation_kernel(a: jax.Array, b: jax.Array) -> jax.Array:
        return mantel_correlation_kernel(a, b, "pearson")

    actual = jax.jit(evaluate_mantel_correlation_kernel)(x, y)
    assert actual.dtype == jnp.bfloat16
    assert bool(jnp.isfinite(actual))
    np.testing.assert_allclose(float(actual), expected, rtol=0.03, atol=0.03)
    design = jnp.stack([jnp.ones_like(x), x], axis=1)
    unsupported = bfloat16_qr_error()
    if unsupported is not None:
        with pytest.raises(unsupported, match="bfloat16") as rejected:
            LeastSquares().estimate(LinearProblem(design, y))
        record_property("bf16_backend", str(rejected.value))
    else:
        fit = LeastSquares().estimate(LinearProblem(design, y))
        record_property("bf16_backend", "accepted")
        assert bool(fit.valid)
        assert bool(fit.solver.converged)
        assert fit.parameters.dtype == jnp.bfloat16
        expected_parameters = np.linalg.lstsq(
            np.asarray(design, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
            rcond=None,
        )[0]
        np.testing.assert_allclose(
            fit.parameters, expected_parameters, rtol=0.08, atol=0.08
        )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.requires_optional("statsmodels")
def test_weighted_least_squares_hc3_matches_statsmodels(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Compare WLS and HC3 on the same quantized inputs and retained sample."""
    sm = optional_reference("statsmodels.api")
    dtype = _dtype()
    # In the x64 lane these roles expose the former mixed WLS QR mismatch.
    x = jnp.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    design = jnp.stack([jnp.ones_like(x), x], axis=1)
    y = jnp.asarray([-1.1, -0.3, 0.2, 1.2, 1.7, 2.7, 3.3, 4.2], dtype=dtype)
    weights = jnp.asarray([0.7, 1.0, 0.8, 1.2, 1.5, 0.9, 1.1, 1.3], dtype=jnp.float32)
    fit = LeastSquares().estimate(LinearProblem(design, y, weights))
    uncertainty = hc_covariance(fit, kind="HC3")
    reference = sm.WLS(
        np.asarray(y, dtype=np.float64),
        np.asarray(design, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
    ).fit(cov_type="HC3")
    assert bool(fit.valid)
    assert bool(fit.solver.converged)
    assert bool(uncertainty.valid)
    assert fit.parameters.dtype == jnp.result_type(design, y, weights, 1.0)
    np.testing.assert_allclose(
        fit.parameters, reference.params, rtol=_tolerance(), atol=_tolerance()
    )
    np.testing.assert_allclose(
        uncertainty.covariance.values,
        reference.cov_params(),
        rtol=_tolerance(),
        atol=_tolerance(),
    )
    assert bool(fit.parameters[1] > 0) == bool(reference.params[1] > 0)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("regime", ["tiny", "large", "high_order"])
def test_exact_transport_scale_matches_scipy_assignment(
    optional_reference: Callable[[str], ModuleType], regime: str
) -> None:
    """Power and mean reductions cannot destroy representable positive distances."""
    optimize = optional_reference("scipy.optimize")
    if regime == "tiny":
        scale, order = (1e-200 if jax.config.x64_enabled else 1e-20), 2.0
    elif regime == "large":
        scale, order = (1e200 if jax.config.x64_enabled else 1.4757394e19), 2.0
    else:
        scale, order = 1e-3, 20.0
    costs = jnp.asarray(np.asarray([[1.0, 0.5], [0.75, 1.0]]) * scale, dtype=_dtype())
    represented = np.asarray(costs, dtype=np.float64)
    normalized = represented / represented.max()
    rows, columns = optimize.linear_sum_assignment(normalized**order)
    selected = represented[rows, columns]
    selected_scale = selected.max()
    expected = selected_scale * np.mean((selected / selected_scale) ** order) ** (
        1.0 / order
    )
    result = jax.jit(balanced_wasserstein_from_cost_result)(costs, order)
    assert bool(result.input_valid)
    assert bool(result.finite)
    assert float(result.value) > 0
    assert result.value.dtype == costs.dtype
    np.testing.assert_allclose(result.value, expected, rtol=_tolerance(), atol=0)


@pytest.mark.integration
@pytest.mark.reference
def test_scaled_gmm_fit_and_covariance_match_scipy(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """A previously accepted truncated coefficient also has coherent uncertainty."""
    linalg = optional_reference("scipy.linalg")
    dtype = _dtype()
    derivative = jnp.diag(jnp.asarray([1.0, 0.001], dtype=dtype))
    deviations = jnp.asarray(
        [
            [0.125, 0.000125],
            [-0.125, 0.000125],
            [0.125, -0.000125],
            [-0.125, -0.000125],
        ],
        dtype=dtype,
    )
    moments = deviations - jnp.asarray([1.0, 0.001], dtype=dtype)
    fit = jax.jit(FixedWeightGMM(PlainLongRun).estimate)(
        AffineMomentProblem(moments, derivative, jnp.eye(2, dtype=dtype))
    )
    uncertainty = jax.jit(gmm_covariance)(fit)
    source = np.asarray(moments, dtype=np.float64)
    jacobian = np.asarray(derivative, dtype=np.float64)
    expected = linalg.lstsq(jacobian, -source.mean(axis=0))[0]
    influence_map = linalg.lstsq(jacobian, np.eye(2))[0]
    residuals = source + expected @ jacobian.T
    omega = residuals.T @ residuals / len(source)
    covariance = influence_map @ omega @ influence_map.T / len(source)
    assert bool(fit.valid)
    assert bool(uncertainty.valid)
    assert fit.parameters.dtype == derivative.dtype
    np.testing.assert_allclose(fit.parameters, expected, rtol=_tolerance(), atol=0)
    np.testing.assert_allclose(
        uncertainty.covariance.values, covariance, rtol=_tolerance(), atol=1e-10
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.requires_optional("statsmodels")
def test_poisson_prediction_matches_statsmodels(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Match likelihood coefficients and offset/exposure predictions at both dtypes."""
    sm = optional_reference("statsmodels.api")
    dtype = _dtype()
    x = jnp.linspace(-1.5, 1.5, 12, dtype=dtype)
    design = jnp.stack([jnp.ones_like(x), x], axis=1)
    counts = jnp.asarray([0, 1, 2, 1, 3, 2, 4, 3, 6, 4, 5, 7], dtype=dtype)
    offset = jnp.linspace(-0.15, 0.2, 12, dtype=dtype)
    exposure = jnp.linspace(0.75, 1.75, 12, dtype=dtype)
    fit = PoissonMLE().estimate(
        PoissonProblem(design, counts, offset=offset, exposure=exposure)
    )
    model = sm.GLM(
        np.asarray(counts, dtype=np.float64),
        np.asarray(design, dtype=np.float64),
        family=sm.families.Poisson(),
        offset=np.asarray(offset, dtype=np.float64),
        exposure=np.asarray(exposure, dtype=np.float64),
    )
    reference = model.fit(maxiter=200, tol=1e-12)
    assert bool(fit.valid)
    assert bool(fit.solver.converged)
    assert fit.parameters.dtype == design.dtype
    tolerance = 1e-4 if dtype == jnp.float32 else 1e-6
    np.testing.assert_allclose(
        fit.parameters, reference.params, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        fit.fitted_means, reference.fittedvalues, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        fit.predict(design, offset=offset, exposure=exposure),
        reference.fittedvalues,
        rtol=tolerance,
        atol=tolerance,
    )
    assert np.all(np.asarray(fit.fitted_means) > 0)
