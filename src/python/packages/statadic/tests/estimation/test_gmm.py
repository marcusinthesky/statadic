# jaxtyping shape strings are runtime contracts.
"""Independent affine-GMM solve/weight/state and transformation witnesses."""

from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.core.equations import MeanCenteredLongRun, PlainLongRun
from statadic.core.results import InferenceStatus, SolverStatus
from statadic.estimation.gmm import FixedWeightGMM, IteratedGMM
from statadic.estimation.moments import AffineMomentProblem
from statadic.inference.gmm import (
    hansen_j_from_fit,
)
from statadic.inference.gmm.covariance import gmm_covariance

from .._budgets import examples
from .._typing import ill_typed

if TYPE_CHECKING:
    from statadic.estimation.gmm import AffineGMMFit

if TYPE_CHECKING:
    from statadic.inference.gmm.hypothesis import GMMHansenJResult


@pytest.mark.component
@pytest.mark.property
@pytest.mark.parametrize("small", [1e-3, 1e-4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_equilibrated_fit_and_covariance_do_not_truncate_parameter_units(
    small: float, dtype: type
) -> None:
    """Fit and sandwich share a full-rank solve under disparate units."""
    derivative = np.diag(np.asarray([1.0, small], dtype=dtype))
    rows = np.asarray(
        [
            [-0.9, -0.9 * small],
            [-0.9, -1.1 * small],
            [-1.1, -0.9 * small],
            [-1.1, -1.1 * small],
        ],
        dtype=dtype,
    )
    weight = np.asarray([[2.0, 0.5], [0.5, 1.0]], dtype=dtype)
    fit = FixedWeightGMM(PlainLongRun).estimate(
        AffineMomentProblem(rows, derivative, weight)
    )
    covariance = gmm_covariance(fit)
    d = derivative.astype(np.float64)
    w = weight.astype(np.float64)
    g = rows.astype(np.float64)
    factor = np.linalg.cholesky(w)
    reference_map = np.linalg.lstsq(factor.T @ d, factor.T, rcond=None)[0]
    expected = -reference_map @ g.mean(axis=0)
    fitted = g + expected @ d.T
    omega = fitted.T @ fitted / len(g)
    reference_covariance = reference_map @ omega @ reference_map.T / len(g)
    assert fit.valid
    assert fit.identified
    assert fit.weighted_rank == derivative.shape[1]
    assert covariance.valid
    tolerance = 4e-3 if dtype is np.float32 else 1e-10
    np.testing.assert_allclose(fit.parameters, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        covariance.covariance.values,
        reference_covariance,
        rtol=tolerance,
        atol=tolerance * 0.0025,
    )


@pytest.mark.component
@pytest.mark.contract
def test_supplied_weight_payload_and_effective_symmetric_weight() -> None:
    """Accepted asymmetry never changes which weight the criterion uses."""
    moments, derivative = _sample()
    weight = np.eye(3)
    weight[0, 1] = np.finfo(weight.dtype).eps
    fit = FixedWeightGMM(PlainLongRun).estimate(
        AffineMomentProblem(moments, derivative, weight)
    )
    effective = (weight + weight.T) / 2
    assert fit.valid
    np.testing.assert_array_equal(fit.weight, weight)
    np.testing.assert_array_equal(fit.effective_weight, effective)
    gbar = np.asarray(fit.scores.values).mean(axis=0)
    np.testing.assert_allclose(
        fit.first_order_residual,
        derivative.mean(axis=0).T @ effective @ gbar,
        atol=1e-12,
    )


@pytest.mark.component
@pytest.mark.contract
def test_singular_intermediate_weight_fails_without_returning_prior_fit() -> None:
    """A failed covariance update cannot silently publish the initial iterate."""
    moments = jnp.zeros((8, 2), dtype=jnp.float32)
    problem = AffineMomentProblem(moments, jnp.eye(2, dtype=jnp.float32))
    fit = IteratedGMM(PlainLongRun, max_iter=1).estimate(problem)
    assert not fit.valid
    assert fit.solver.status == SolverStatus.FAILED
    assert fit.solver.iterations == 1
    assert jnp.isnan(fit.parameters).all()


@pytest.mark.component
@pytest.mark.contract
def test_weighted_solve_mixed_precision_and_batched_status() -> None:
    """One invalid lane leaves the promoted solve and input payloads intact."""
    moments, derivative = _sample()
    moments = jnp.asarray(moments, dtype=jnp.float32)
    derivative = jnp.asarray(derivative, dtype=jnp.float64)
    weights = jnp.stack(
        (
            jnp.eye(3, dtype=jnp.float32),
            jnp.diag(jnp.asarray([1.0, 1.0, -1.0], dtype=jnp.float32)),
        )
    )

    def evaluate_fit(w: jax.Array) -> AffineGMMFit[PlainLongRun]:
        return FixedWeightGMM(PlainLongRun).estimate(
            AffineMomentProblem(moments, derivative, w)
        )

    evaluate_fit = jax.jit(jax.vmap(evaluate_fit))(weights)
    np.testing.assert_array_equal(evaluate_fit.valid, [True, False])
    assert evaluate_fit.parameters.dtype == jnp.result_type(
        moments, derivative, weights
    )
    assert evaluate_fit.weight.dtype == weights.dtype
    assert evaluate_fit.moments0.dtype == moments.dtype
    assert evaluate_fit.derivative_panel.dtype == derivative.dtype


@pytest.mark.component
@pytest.mark.contract
def test_covariance_correction_follows_caller_promotion() -> None:
    """A higher-precision correction is not narrowed to fit precision."""
    if not jax.config.jax_enable_x64:
        pytest.skip("mixed float32/float64 requires JAX x64")
    moments, derivative = _sample()
    fit = FixedWeightGMM(PlainLongRun).estimate(
        AffineMomentProblem(
            jnp.asarray(moments, dtype=jnp.float32),
            jnp.asarray(derivative, dtype=jnp.float32),
            jnp.eye(3, dtype=jnp.float32),
        )
    )
    correction = jnp.asarray(1.25, dtype=jnp.float64)
    covariance = gmm_covariance(fit, finite_sample_correction=correction)
    assert covariance.valid
    assert covariance.covariance.values.dtype == jnp.float64


@pytest.mark.component
@pytest.mark.reference
def test_mixed_precision_panel_uses_working_precision_for_mean_jacobian() -> None:
    """Fit, retained map, and covariance use the represented-input mean."""
    if not jax.config.jax_enable_x64:
        pytest.skip("mixed float32/float64 requires JAX x64")
    derivative = np.asarray([1, 1, 1 + np.finfo(np.float32).eps], dtype=np.float32)
    panel = derivative[:, None, None]
    moments = np.asarray([[-0.8], [-1.0], [-1.2]], dtype=np.float64)
    fit = FixedWeightGMM(PlainLongRun).estimate(
        AffineMomentProblem(moments, panel, np.ones((1, 1)))
    )
    mean_d = panel.astype(np.float64).mean(axis=0)
    expected_map = np.linalg.lstsq(mean_d, np.ones((1, 1)), rcond=None)[0]
    expected_theta = -expected_map @ moments.mean(axis=0)
    fitted = moments + np.einsum("tmk,k->tm", panel, expected_theta)
    expected_cov = expected_map @ (fitted.T @ fitted / len(fitted)) @ expected_map.T
    assert fit.valid
    np.testing.assert_allclose(fit.jacobian.values, mean_d, rtol=1e-14)
    np.testing.assert_allclose(fit.moment_map, expected_map, rtol=1e-14)
    np.testing.assert_allclose(fit.parameters, expected_theta, rtol=1e-14)
    np.testing.assert_allclose(
        fit.first_order_residual, mean_d.T @ fitted.mean(axis=0), atol=1e-14
    )
    np.testing.assert_allclose(
        gmm_covariance(fit).covariance.values, expected_cov / len(fitted), rtol=1e-14
    )


@pytest.mark.component
@pytest.mark.contract
def test_integer_moment_and_derivative_payloads_retain_dtype() -> None:
    """Working promotion does not change retained integer input arrays."""
    moments = jnp.asarray([[-1], [-2], [-3]], dtype=jnp.int32)
    derivative = jnp.ones((1, 1), dtype=jnp.int32)
    fit = FixedWeightGMM(PlainLongRun).estimate(
        AffineMomentProblem(moments, derivative, jnp.ones((1, 1)))
    )
    assert fit.valid
    assert fit.moments0.dtype == moments.dtype
    assert fit.derivative_panel.dtype == derivative.dtype
    np.testing.assert_allclose(fit.parameters, [2.0])


@pytest.mark.component
@pytest.mark.performance
def test_iterated_fit_reuses_compilation_for_changed_moments(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Different observations of the same shape reuse the GMM iteration."""
    rng = np.random.default_rng(819)
    derivative = jnp.asarray([[1.0], [0.5], [-0.25]])
    samples = [jnp.asarray(rng.normal(size=(37, 3))) for _ in range(2)]
    estimator = IteratedGMM(variability=PlainLongRun, max_iter=7)
    with caplog.at_level(logging.WARNING), jax.log_compiles():
        first = estimator.estimate(AffineMomentProblem(samples[0], derivative))
        first.parameters.block_until_ready()
        compiled = [r for r in caplog.records if "Compiling jit(_iterate)" in r.message]
        caplog.clear()
        second = estimator.estimate(AffineMomentProblem(samples[1], derivative))
        second.parameters.block_until_ready()
    assert compiled
    assert not [r for r in caplog.records if "Compiling jit(_iterate)" in r.message]
    assert not np.allclose(first.parameters, second.parameters)


@pytest.mark.component
@pytest.mark.contract
@examples(12)
@example(seed=820)
@given(seed=st.integers(0, 2**32 - 1))
def test_fixed_weight_uses_normal_mixed_input_promotion(seed: int) -> None:
    """A higher-precision caller weight is not truncated to the moments dtype."""
    rng = np.random.default_rng(seed)
    moments = jnp.asarray(rng.normal(size=(31, 2)), dtype=jnp.float32)
    derivative = jnp.asarray([[1.0], [0.5]], dtype=jnp.float32)
    weight = jnp.eye(2, dtype=jnp.float64)
    fit = FixedWeightGMM(PlainLongRun).estimate(
        AffineMomentProblem(moments, derivative, weight)
    )
    assert fit.parameters.dtype == jnp.result_type(moments, derivative, weight)
    assert fit.weight.dtype == weight.dtype
    assert fit.moments0.dtype == moments.dtype
    assert fit.derivative_panel.dtype == derivative.dtype


@pytest.mark.component
@pytest.mark.contract
@examples(12)
@example(seed=821)
@given(seed=st.integers(0, 2**32 - 1))
def test_iterated_gmm_preserves_input_dtypes_with_mixed_derivative(seed: int) -> None:
    """The common solver state does not rewrite retained input payloads."""
    rng = np.random.default_rng(seed)
    moments = jnp.asarray(rng.normal(size=(29, 2)), dtype=jnp.float32)
    derivative = jnp.asarray([[1.0], [0.5]], dtype=jnp.float64)
    fit = IteratedGMM(PlainLongRun, max_iter=2).estimate(
        AffineMomentProblem(moments, derivative)
    )
    assert fit.moments0.dtype == moments.dtype
    assert fit.derivative_panel.dtype == derivative.dtype
    assert fit.parameters.dtype == jnp.result_type(moments, derivative)


@pytest.mark.component
@pytest.mark.property
@given(
    scales=st.lists(st.integers(2, 6), min_size=3, max_size=3),
    shear=st.integers(-3, 3),
)
def test_moment_coordinates_transform_actual_weight_and_full_sandwich(
    scales: list[int], shear: int
) -> None:
    """Invertible moment coordinates preserve the criterion only with congruent W."""
    moments, derivative = _sample()
    transform = np.diag(np.asarray(scales) / 4)
    transform[0, 1] = shear / 4
    inverse = np.linalg.inv(transform)
    weight = np.asarray([[4.0, 0.6, 0.0], [0.6, 0.5, 0.1], [0.0, 0.1, 2.0]])
    fit = FixedWeightGMM(variability=PlainLongRun, bandwidth=2).estimate(
        AffineMomentProblem(moments, derivative, weight)
    )
    changed = FixedWeightGMM(variability=PlainLongRun, bandwidth=2).estimate(
        AffineMomentProblem(
            moments @ transform.T,
            np.einsum("ij,tjk->tik", transform, derivative),
            inverse.T @ weight @ inverse,
        )
    )
    assert fit.valid
    assert changed.valid
    np.testing.assert_allclose(changed.parameters, fit.parameters, rtol=1e-11)
    np.testing.assert_allclose(changed.criterion, fit.criterion, rtol=1e-11)
    np.testing.assert_allclose(
        gmm_covariance(changed).covariance.values,
        gmm_covariance(fit).covariance.values,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        changed.scores.values, np.asarray(fit.scores.values) @ transform.T, atol=1e-11
    )


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_estimated_inverse_weight_preserves_symmetry_and_actual_solve(
    dtype: type,
) -> None:
    """SVD roundoff in a well-conditioned inverse cannot invalidate its own fit."""
    derivative = np.asarray([[1.0], [0.5], [-0.25]], dtype=dtype)
    rng = np.random.default_rng(np.random.SeedSequence([20261028, 901, 5, 0]))
    moments = (rng.normal(size=(512, 3)) - derivative[:, 0] * 0.4).astype(dtype)
    centered = moments - moments.mean(0)
    expected_weight = np.linalg.inv(centered.T @ centered / len(moments))
    expected = -np.linalg.solve(
        derivative.T @ expected_weight @ derivative,
        derivative.T @ expected_weight @ moments.mean(0),
    )

    # The caller's update tolerance must resolve coefficient changes in its
    # dtype; the production default remains unchanged.
    update_tolerance = max(1e-10, 10 * float(np.finfo(dtype).eps))
    fit = IteratedGMM(
        variability=MeanCenteredLongRun, tolerance=update_tolerance
    ).estimate(AffineMomentProblem(moments, derivative))

    assert fit.valid
    assert fit.identified
    assert fit.solver.converged
    assert fit.parameters.dtype == dtype
    np.testing.assert_array_equal(fit.weight, np.asarray(fit.weight).T)
    tolerance = 100 * np.finfo(dtype).eps
    np.testing.assert_allclose(
        fit.weight, expected_weight, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(fit.parameters, expected, rtol=tolerance, atol=tolerance)
    inverse = np.linalg.inv(derivative.T @ expected_weight @ derivative)
    omega = centered.T @ centered / len(moments)
    expected_covariance = (
        inverse
        @ derivative.T
        @ expected_weight
        @ omega
        @ expected_weight
        @ derivative
        @ inverse
        / len(moments)
    )
    uncertainty = gmm_covariance(fit)
    assert uncertainty.valid
    np.testing.assert_allclose(
        uncertainty.covariance.values,
        expected_covariance,
        rtol=tolerance,
        atol=tolerance * np.max(np.abs(expected_covariance)),
    )


def _sample(seed: int = 498) -> tuple[np.ndarray, np.ndarray]:
    """Use varying derivatives and serial heterogeneous moments."""
    rng = np.random.default_rng(seed)
    derivative = rng.normal(size=(80, 3, 2)) + np.asarray(
        [[1.0, 0.4], [0.2, 1.2], [-0.5, 0.3]]
    )
    noise = rng.normal(size=(80, 3)) @ np.asarray(
        [[1.0, 0.7, 0.2], [0.0, 1.4, 0.4], [0.0, 0.0, 0.6]]
    )
    moments = noise - derivative @ np.asarray([0.7, -0.2])
    return moments, derivative


# Samples drawn by seed: the properties below hold for any such draw, and the
# retained seed 498 stays as an explicit example.
_SAMPLES = st.integers(0, 2**32 - 1).map(_sample)


def _hac(panel: np.ndarray, lag: int, *, centered: bool) -> np.ndarray:
    """Independent explicit lag products with the same retained n denominator."""
    rows = panel - panel.mean(0) if centered else panel
    omega = rows.T @ rows / len(rows)
    for offset in range(1, lag + 1):
        product = rows[offset:].T @ rows[:-offset] / len(rows)
        omega += (1 - offset / (lag + 1)) * (product + product.T)
    return omega


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("scheme", [PlainLongRun, MeanCenteredLongRun])
def test_last_solve_weight_and_final_state_match_independent_update(
    scheme: type[PlainLongRun | MeanCenteredLongRun],
) -> None:
    """A post-solve Omega inverse must not be substituted for the actual W."""
    moments, derivative = _sample()
    dbar, gbar0 = derivative.mean(0), moments.mean(0)
    initial = -np.linalg.solve(dbar.T @ dbar, dbar.T @ gbar0)
    actual_weight = np.linalg.inv(
        _hac(moments + derivative @ initial, 2, centered=scheme is MeanCenteredLongRun)
    )
    expected = -np.linalg.solve(
        dbar.T @ actual_weight @ dbar, dbar.T @ actual_weight @ gbar0
    )
    fitted = moments + derivative @ expected
    final_omega = _hac(fitted, 2, centered=scheme is MeanCenteredLongRun)
    fit = IteratedGMM(variability=scheme, bandwidth=2, max_iter=1).estimate(
        AffineMomentProblem(moments, derivative)
    )
    assert fit.valid
    assert not fit.solver.converged
    assert fit.solver.status == SolverStatus.MAX_STEPS
    assert fit.solver.iterations == 1
    np.testing.assert_allclose(fit.parameters, expected, rtol=1e-12)
    np.testing.assert_allclose(fit.weight, actual_weight, rtol=1e-12)
    minimum_weight_difference = 1e-5
    assert (
        np.linalg.norm(actual_weight - np.linalg.inv(final_omega))
        > minimum_weight_difference
    )
    np.testing.assert_allclose(fit.scores.values, fitted, atol=1e-12)
    np.testing.assert_allclose(fit.variability.values, final_omega, rtol=1e-12)
    np.testing.assert_allclose(fit.jacobian.values, dbar, rtol=1e-12)
    np.testing.assert_allclose(
        fit.criterion, fitted.mean(0) @ actual_weight @ fitted.mean(0), rtol=1e-12
    )
    np.testing.assert_allclose(
        fit.first_order_residual, dbar.T @ actual_weight @ fitted.mean(0), atol=1e-14
    )


@pytest.mark.component
@pytest.mark.reference
def test_non_efficient_weight_uses_full_sandwich_and_explicit_count() -> None:
    """A deliberately non-efficient W makes the inverse-bread shortcut fail."""
    moments, derivative = _sample()
    weight = np.asarray([[4.0, 0.6, 0.0], [0.6, 0.5, 0.1], [0.0, 0.1, 2.0]])
    fit = FixedWeightGMM(
        variability=PlainLongRun, bandwidth=2, parameter_names=("level", "slope")
    ).estimate(AffineMomentProblem(moments, derivative, weight))
    observed = gmm_covariance(fit, finite_sample_correction=1.2)
    assert fit.valid
    assert fit.solver.converged
    assert observed.valid
    assert observed.fit is fit
    dbar = derivative.mean(0)
    omega = _hac(moments + derivative @ np.asarray(fit.parameters), 2, centered=False)
    inverse = np.linalg.inv(dbar.T @ weight @ dbar)
    expected = inverse @ dbar.T @ weight @ omega @ weight @ dbar @ inverse / 80 * 1.2
    np.testing.assert_allclose(observed.covariance.values, expected, rtol=1e-12)
    efficient_shortcut = np.linalg.inv(dbar.T @ np.linalg.inv(omega) @ dbar) / 80 * 1.2
    minimum_covariance_difference = 1e-4
    assert np.linalg.norm(expected - efficient_shortcut) > minimum_covariance_difference
    assert observed.covariance.n_observations == len(moments)
    np.testing.assert_allclose(observed.covariance.finite_sample_correction, 1.2)
    tested = hansen_j_from_fit(fit, assume_correct_specification=True)
    assert not tested.valid
    assert not tested.efficient_weight
    assert tested.conventional_assumptions
    assert np.isnan(tested.test.pvalue)


@pytest.mark.component
@pytest.mark.contract
def test_exact_identification_estimates_without_fabricated_j() -> None:
    """m=p supports fitting and covariance, but has no overidentification test."""
    moments, derivative = _sample()
    fit = IteratedGMM(variability=MeanCenteredLongRun).estimate(
        AffineMomentProblem(moments[:, :2], derivative[:, :2])
    )
    assert fit.valid
    assert fit.identified
    np.testing.assert_allclose(fit.scores.values.mean(0), 0, atol=1e-12)
    assert gmm_covariance(fit).valid
    result = hansen_j_from_fit(fit, assume_correct_specification=True)
    assert not result.valid
    assert not result.overidentified
    assert result.degrees_of_freedom == 0
    assert np.isnan(result.test.statistic)
    assert np.isnan(result.test.pvalue)


@pytest.mark.component
@pytest.mark.contract
def test_conventional_j_requires_acknowledgement_and_final_efficiency() -> None:
    """A common derivative makes the centered variability fixed point explicit."""
    moments, _ = _sample()
    derivative = np.asarray([[1.0], [0.5], [-0.25]])
    fit = IteratedGMM(variability=MeanCenteredLongRun).estimate(
        AffineMomentProblem(moments, derivative)
    )
    result = hansen_j_from_fit(fit, assume_correct_specification=True)
    assert fit.valid
    assert fit.solver.converged
    assert result.valid
    assert result.efficient_weight
    assert result.conventional_assumptions
    assert result.degrees_of_freedom == moments.shape[1] - derivative.shape[1]
    expected = (
        len(moments)
        * np.asarray(fit.scores.values).mean(0)
        @ np.asarray(fit.weight)
        @ np.asarray(fit.scores.values).mean(0)
    )
    np.testing.assert_allclose(result.test.statistic, expected, rtol=1e-12)
    unacknowledged = hansen_j_from_fit(fit, assume_correct_specification=False)
    assert unacknowledged.efficient_weight
    assert not unacknowledged.conventional_assumptions
    assert not unacknowledged.valid
    unfinished = IteratedGMM(variability=MeanCenteredLongRun, max_iter=0).estimate(
        AffineMomentProblem(moments, derivative)
    )
    assert unfinished.valid
    unfinished_test = hansen_j_from_fit(unfinished, assume_correct_specification=True)
    assert not unfinished_test.efficient_weight
    assert unfinished_test.conventional_assumptions
    assert not unfinished_test.valid


@pytest.mark.component
@pytest.mark.regression
@pytest.mark.parametrize("scheme", [PlainLongRun, MeanCenteredLongRun])
def test_finite_budget_fit_retains_independent_loop_weight(
    scheme: type[PlainLongRun | MeanCenteredLongRun],
) -> None:
    """Preserve the coefficient loop without fabricating an efficient J."""
    moments, derivative = _sample()
    dbar = derivative.mean(0)
    theta = -np.linalg.pinv(dbar.T @ dbar) @ dbar.T @ moments.mean(0)
    for _ in range(3):
        omega = _hac(
            moments + derivative @ theta, 1, centered=scheme is MeanCenteredLongRun
        )
        weight = np.linalg.pinv(omega)
        theta = (
            -np.linalg.pinv(dbar.T @ weight @ dbar) @ dbar.T @ weight @ moments.mean(0)
        )
    fitted = moments + derivative @ theta
    omega = _hac(fitted, 1, centered=scheme is MeanCenteredLongRun)
    result = IteratedGMM(scheme, bandwidth=1, max_iter=3, tolerance=1e-30).estimate(
        AffineMomentProblem(jnp.asarray(moments), jnp.asarray(derivative))
    )
    np.testing.assert_allclose(result.parameters, theta, rtol=1e-12)
    np.testing.assert_allclose(result.variability.values, omega, rtol=1e-12)
    np.testing.assert_allclose(result.weight, weight, rtol=1e-12)
    tested = hansen_j_from_fit(result, assume_correct_specification=True)
    assert not bool(tested.valid)
    assert np.isnan(tested.test.statistic)


@pytest.mark.component
@pytest.mark.contract
@examples(12)
@example(sample=_sample(), case="rank")
@example(sample=_sample(), case="nonfinite")
@example(sample=_sample(), case="indefinite")
@example(sample=_sample(), case="asymmetric")
@given(
    sample=_SAMPLES,
    case=st.sampled_from(["rank", "nonfinite", "indefinite", "asymmetric"]),
)
def test_invalid_numerical_inputs_never_authorize_inference(
    sample: tuple[np.ndarray, np.ndarray], case: str
) -> None:
    """Identification and numerical validity survive independently from pinv."""
    moments, derivative = (array.copy() for array in sample)
    weight = np.eye(3)
    if case == "rank":
        derivative[:, :, 1] = 2 * derivative[:, :, 0]
    elif case == "nonfinite":
        moments[0, 0] = np.nan
    elif case == "indefinite":
        weight[0, 0] = -1
    else:
        weight[0, 1] = 0.5
    fit = FixedWeightGMM(variability=PlainLongRun).estimate(
        AffineMomentProblem(moments, derivative, weight)
    )
    assert not fit.valid
    assert not fit.identified if case == "rank" else not fit.input_valid
    assert np.isnan(fit.parameters).all()
    assert not gmm_covariance(fit).valid
    assert (
        hansen_j_from_fit(fit, assume_correct_specification=True).status
        == InferenceStatus.INVALID_FIT
    )


@pytest.mark.component
@pytest.mark.contract
@examples(8)
@example(sample=_sample())
@given(sample=_SAMPLES)
def test_fixed_shape_jit_batch_preserves_float32_and_lane_statuses(
    sample: tuple[np.ndarray, np.ndarray],
) -> None:
    """Transforms retain a failed lane instead of replacing or dropping it."""
    moments, derivative = sample
    rows = jnp.asarray(moments, dtype=jnp.float32)
    jacobian = jnp.asarray(derivative, dtype=jnp.float32)
    batch = jnp.stack((rows, rows.at[0, 0].set(jnp.nan)))

    def evaluate_fit(g: jax.Array) -> AffineGMMFit[PlainLongRun]:
        return IteratedGMM(
            variability=PlainLongRun, max_iter=2, tolerance=1e-06
        ).estimate(AffineMomentProblem(g, jacobian))

    fitted = jax.jit(jax.vmap(evaluate_fit))(batch)
    assert fitted.parameters.dtype == jnp.float32
    np.testing.assert_array_equal(fitted.valid, [True, False])
    uncertainty = jax.jit(jax.vmap(gmm_covariance))(fitted)
    np.testing.assert_array_equal(uncertainty.valid, [True, False])
    assert uncertainty.covariance.values.dtype == jnp.float32


@pytest.mark.component
@pytest.mark.contract
@examples(8)
@example(sample=_sample())
@given(sample=_SAMPLES)
def test_fit_derived_j_supports_fixed_shape_jit_and_batching(
    sample: tuple[np.ndarray, np.ndarray],
) -> None:
    """Analytic calibration preserves each lane's assumptions/fit evidence."""
    moments, _ = sample
    rows = jnp.asarray(moments)
    derivative = jnp.asarray([[1.0], [0.5], [-0.25]])

    def evaluate_fit(g: jax.Array) -> AffineGMMFit[MeanCenteredLongRun]:
        return IteratedGMM(variability=MeanCenteredLongRun).estimate(
            AffineMomentProblem(g, derivative)
        )

    fitted = jax.jit(jax.vmap(evaluate_fit))(
        jnp.stack((rows, rows.at[0, 0].set(jnp.nan)))
    )

    def evaluate_hansen_j_from_fit(fit: AffineGMMFit[PlainLongRun]) -> GMMHansenJResult:
        return hansen_j_from_fit(fit, assume_correct_specification=True)

    result = jax.jit(jax.vmap(evaluate_hansen_j_from_fit))(fitted)
    np.testing.assert_array_equal(result.valid, [True, False])
    assert result.test.null_distribution.shape == (2, 0)


@pytest.mark.component
@pytest.mark.contract
@examples(12)
@example(sample=_sample())
@given(sample=_SAMPLES)
def test_owned_arrays_names_and_fixed_weight_are_immutable(
    sample: tuple[np.ndarray, np.ndarray],
) -> None:
    """Later mutation of host inputs cannot change the fitted sample."""
    moments, derivative = (array.copy() for array in sample)
    fit = FixedWeightGMM(variability=PlainLongRun, parameter_names=("a", "b")).estimate(
        AffineMomentProblem(moments, derivative, np.eye(3))
    )
    retained = np.asarray(fit.moments0).copy()
    moments[:] = 0
    derivative[:] = 0
    np.testing.assert_array_equal(fit.moments0, retained)
    assert fit.parameter_names == ("a", "b")
    with pytest.raises(FrozenInstanceError):
        ill_typed(fit).bandwidth = 1


@pytest.mark.component
@pytest.mark.contract
@examples(12)
@example(sample=_sample(), case="rows")
@example(sample=_sample(), case="underidentified")
@example(sample=_sample(), case="derivative")
@example(sample=_sample(), case="bandwidth")
@example(sample=_sample(), case="updates")
@example(sample=_sample(), case="names")
@example(sample=_sample(), case="weight")
@given(
    sample=_SAMPLES,
    case=st.sampled_from(
        [
            "rows",
            "underidentified",
            "derivative",
            "bandwidth",
            "updates",
            "names",
            "weight",
        ]
    ),
)
def test_static_boundary_rejects_malformed_inputs(
    sample: tuple[np.ndarray, np.ndarray], case: str
) -> None:
    """Shape and schema errors are explicit API errors before numerical solving."""
    moments, derivative = sample
    if case == "rows":
        moments, derivative = moments[:1], derivative[:1]
    elif case == "underidentified":
        moments, derivative = moments[:, :1], derivative[:, :1]
    elif case == "derivative":
        derivative = derivative[:-1]
    expected = {
        "rows": "g0 must have shape",
        "underidentified": "m >= p",
        "derivative": "jacobian must have shape",
        "bandwidth": "bandwidth must be",
        "updates": "max_iter must be",
        "names": "parameter_names must be",
        "weight": "weight must have shape",
    }
    estimator = (
        FixedWeightGMM(PlainLongRun)
        if case == "weight"
        else IteratedGMM(
            PlainLongRun,
            bandwidth=80 if case == "bandwidth" else 0,
            max_iter=-1 if case == "updates" else 25,
            parameter_names=("a", "a") if case == "names" else None,
        )
    )
    problem = AffineMomentProblem(
        moments, derivative, np.eye(2) if case == "weight" else None
    )
    with pytest.raises(ValueError, match=expected[case]):
        estimator.estimate(problem)
