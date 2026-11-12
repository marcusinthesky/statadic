# standalone package-maintenance entry point
"""Fit-aware restriction and interval semantics against independent references."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from scipy import stats
from statadic.core.results import InferenceStatus
from statadic.estimation.linear import (
    LeastSquares,
    LinearProblem,
)
from statadic.inference import parameters as parameter_module
from statadic.inference._parameter_reference import _f_survival, _student_critical_value
from statadic.inference.linear_covariance import classical_covariance, hc_covariance
from statadic.inference.parameters import (
    coefficient_intervals,
    linear_restriction_test,
)

from .._budgets import examples

if TYPE_CHECKING:
    from statadic.core.equations import (
        IndependentRows,
    )
    from statadic.inference.linear_covariance import LinearUncertainty
    from statadic.inference.parameters import (
        CoefficientIntervals,
        LinearRestrictionResult,
    )

_FLOAT32_QUALIFIED_DF = 10000


def _example(
    names: tuple[str, ...] | None = None,
) -> LinearUncertainty[IndependentRows]:
    """Return one ordinary fit with residual noise and a full-rank design."""
    design = jnp.stack([jnp.ones(8), jnp.linspace(-2.0, 3.0, 8)], axis=1)
    outcome = design @ jnp.array([1.2, 0.7]) + jnp.array(
        [0.2, -0.4, 0.1, 0.3, -0.2, 0.5, -0.1, 0.1]
    )
    return classical_covariance(
        LeastSquares().estimate(LinearProblem(design, outcome, parameter_names=names))
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
def test_restriction_calibration_matches_chi_square_and_f_without_refitting() -> None:
    """An independently formed full-rank quadratic has distinct F and chi2 scales."""
    uncertainty = _example()
    matrix = np.asarray([[1.0, 0.5], [-0.2, 1.0]])
    target = np.asarray([0.8, 0.1])
    delta = matrix @ np.asarray(uncertainty.fit.parameters) - target
    restricted = matrix @ np.asarray(uncertainty.covariance.values) @ matrix.T
    expected = delta @ np.linalg.solve(restricted, delta)
    numerator_df = matrix.shape[0]
    denominator_df = 4.0  # An explicit choice, independent from n-p and correction.
    chi_square = linear_restriction_test(uncertainty, matrix, target, reference="chi2")
    fisher = linear_restriction_test(
        uncertainty,
        matrix,
        target,
        reference="f",
        degrees_of_freedom=denominator_df,
    )
    assert bool(chi_square.valid)
    assert bool(fisher.valid)
    assert float(chi_square.test.statistic) == pytest.approx(expected, rel=1e-12)
    assert float(chi_square.test.pvalue) == pytest.approx(
        stats.chi2.sf(expected, numerator_df), rel=1e-12
    )
    assert float(fisher.wald_statistic) == pytest.approx(expected, rel=1e-12)
    assert float(fisher.test.statistic) == pytest.approx(
        expected / numerator_df, rel=1e-12
    )
    assert float(fisher.test.pvalue) == pytest.approx(
        stats.f.sf(expected / numerator_df, numerator_df, denominator_df), rel=1e-12
    )
    assert int(fisher.restriction_rank) == numerator_df
    assert int(fisher.covariance_rank) == numerator_df
    assert int(fisher.numerator_df) == numerator_df
    np.testing.assert_array_equal(fisher.denominator_df, denominator_df)
    assert np.isinf(chi_square.denominator_df)
    assert chi_square.test.n_resamples == 0
    assert chi_square.test.null_distribution.shape == (0,)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
def test_coefficient_intervals_match_scipy_and_keep_reference_separate() -> None:
    """Covariance stays fixed while an explicit reference/df changes interval widths."""
    uncertainty = _example()
    confidence = 0.95
    degrees = 3.0
    normal = coefficient_intervals(
        uncertainty, confidence=confidence, reference="normal"
    )
    student = coefficient_intervals(
        uncertainty, confidence=confidence, reference="t", degrees_of_freedom=degrees
    )
    standard_errors = np.sqrt(np.diag(np.asarray(uncertainty.covariance.values)))
    parameters = np.asarray(uncertainty.fit.parameters)
    for actual, critical in (
        (normal, stats.norm.isf((1.0 - confidence) / 2.0)),
        (student, stats.t.isf((1.0 - confidence) / 2.0, degrees)),
    ):
        assert bool(actual.valid)
        assert int(actual.status) == InferenceStatus.AVAILABLE
        np.testing.assert_array_equal(actual.estimate.estimate, parameters)
        np.testing.assert_allclose(actual.estimate.se, standard_errors, atol=1e-14)
        np.testing.assert_allclose(
            actual.estimate.ci[0], parameters - critical * standard_errors, atol=1e-12
        )
        np.testing.assert_allclose(
            actual.estimate.ci[1], parameters + critical * standard_errors, atol=1e-12
        )
        assert bool(actual.estimate.converged)
        assert actual.estimate.n_obs == uncertainty.fit.design.shape[0]
    np.testing.assert_array_equal(student.degrees_of_freedom, degrees)
    assert np.isinf(normal.degrees_of_freedom)
    robust = hc_covariance(uncertainty.fit, kind="HC0")
    robust_interval = coefficient_intervals(robust, reference="normal")
    np.testing.assert_array_equal(robust_interval.estimate.estimate, parameters)
    assert not np.allclose(robust_interval.estimate.se, normal.estimate.se)


@pytest.mark.component
@pytest.mark.contract
def test_named_restrictions_require_exact_parameter_order() -> None:
    """Equal width cannot certify a permuted or omitted name layout."""
    uncertainty = _example(("intercept", "slope"))
    for names in (None, ("slope", "intercept")):
        with pytest.raises(ValueError, match="exactly match"):
            linear_restriction_test(
                uncertainty,
                jnp.array([[0.0, 1.0]]),
                0.0,
                reference="chi2",
                parameter_names=names,
            )
    accepted = linear_restriction_test(
        uncertainty,
        jnp.array([[0.0, 1.0]]),
        0.0,
        reference="chi2",
        parameter_names=("intercept", "slope"),
    )
    assert bool(accepted.valid)
    assert (
        coefficient_intervals(uncertainty, reference="normal").parameter_names
        == uncertainty.fit.parameter_names
    )
    with pytest.raises(ValueError, match="positional fit"):
        linear_restriction_test(
            _example(),
            jnp.array([[0.0, 1.0]]),
            0.0,
            reference="chi2",
            parameter_names=("intercept", "slope"),
        )


@pytest.mark.component
@pytest.mark.contract
def test_redundant_restrictions_and_singular_covariance_fail_closed() -> None:
    """Finite pseudoinverse statistics cannot license a reduced-rank reference."""
    uncertainty = _example()
    redundant = linear_restriction_test(
        uncertainty, jnp.array([[0.0, 1.0], [0.0, 1.0]]), 0.0, reference="chi2"
    )
    assert not bool(redundant.valid)
    assert int(redundant.status) == InferenceStatus.INVALID_RESTRICTION
    assert int(redundant.restriction_rank) == 1
    assert np.isnan(redundant.test.statistic)
    assert np.isnan(redundant.test.pvalue)
    singular = replace(
        uncertainty,
        covariance=replace(uncertainty.covariance, values=jnp.ones((2, 2))),
    )
    projected = linear_restriction_test(singular, jnp.eye(2), 0.0, reference="chi2")
    assert not bool(projected.valid)
    assert int(projected.status) == InferenceStatus.INVALID_COVARIANCE
    assert int(projected.covariance_rank) == 1
    assert np.isnan(projected.test.pvalue)
    zero_variance = replace(
        uncertainty,
        covariance=replace(
            uncertainty.covariance, values=jnp.diag(jnp.array([1.0, 0.0]))
        ),
    )
    interval = coefficient_intervals(zero_variance, reference="normal")
    assert not bool(interval.valid)
    assert int(interval.status) == InferenceStatus.INVALID_COVARIANCE
    assert np.isnan(np.asarray(interval.estimate.se)).all()


@pytest.mark.component
@pytest.mark.contract
def test_mixed_interval_statuses_survive_jit_and_vmap() -> None:
    """Numerical convergence and valid intervals remain separate in each lane."""
    ordinary = _example().fit
    exact = ordinary.design @ jnp.array([1.2, 0.7])
    outcomes = jnp.stack(
        [ordinary.outcome, exact, ordinary.outcome.at[0].set(jnp.nan), ordinary.outcome]
    )
    confidence = jnp.array([0.95, 0.95, 0.95, 1.0])

    def run(outcome: jax.Array, level: jax.Array) -> CoefficientIntervals:
        uncertainty = classical_covariance(
            LeastSquares().estimate(LinearProblem(ordinary.design, outcome))
        )
        return coefficient_intervals(
            uncertainty, confidence=level, reference="t", degrees_of_freedom=6.0
        )

    result = jax.jit(jax.vmap(run))(outcomes, confidence)
    np.testing.assert_array_equal(result.valid, [True, False, False, False])
    np.testing.assert_array_equal(
        result.status,
        [
            InferenceStatus.AVAILABLE,
            InferenceStatus.EXACT_FIT,
            InferenceStatus.INVALID_FIT,
            InferenceStatus.INVALID_REFERENCE,
        ],
    )
    np.testing.assert_array_equal(result.estimate.converged[:2], [True, True])
    assert np.isnan(np.asarray(result.estimate.ci[0][1:])).all()


@pytest.mark.component
@pytest.mark.contract
def test_mixed_restriction_statuses_survive_jit_and_vmap() -> None:
    """Invalid restrictions cannot alter a neighboring lane's F calibration."""
    uncertainty = _example()
    matrices = jnp.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[jnp.nan, 0.0], [0.0, 1.0]],
        ]
    )

    def evaluate_linear_restriction_test(matrix: jax.Array) -> LinearRestrictionResult:
        return linear_restriction_test(
            uncertainty, matrix, 0.0, reference="f", degrees_of_freedom=6.0
        )

    result = jax.jit(jax.vmap(evaluate_linear_restriction_test))(matrices)
    np.testing.assert_array_equal(result.valid, [True, False, False])
    np.testing.assert_array_equal(
        result.status,
        [
            InferenceStatus.AVAILABLE,
            InferenceStatus.INVALID_RESTRICTION,
            InferenceStatus.INVALID_RESTRICTION,
        ],
    )
    assert np.isfinite(result.test.pvalue[0])
    assert np.isnan(np.asarray(result.test.pvalue[1:])).all()


@pytest.mark.component
@pytest.mark.contract
@examples(12)
@example(degrees=0.0)
@example(degrees=-1.0)
@example(degrees=float("nan"))
@example(degrees=float("inf"))
@given(degrees=st.floats().filter(lambda d: not 0 < d < float("inf")))
def test_invalid_reference_degrees_are_unavailable(degrees: float) -> None:
    """Nonpositive/nonfinite denominator degrees cannot produce calibrated output.

    Args:
        degrees: Any float that is not a positive finite degrees of freedom.

    """
    uncertainty = _example()
    restriction = linear_restriction_test(
        uncertainty,
        jnp.array([[0.0, 1.0]]),
        0.0,
        reference="f",
        degrees_of_freedom=degrees,
    )
    interval = coefficient_intervals(
        uncertainty, reference="t", degrees_of_freedom=degrees
    )
    for result in (restriction, interval):
        assert not bool(result.valid)
        assert int(result.status) == InferenceStatus.INVALID_REFERENCE


@pytest.mark.component
@pytest.mark.contract
@examples(60)
@example(confidence=0.0)
@example(confidence=1.0)
@example(confidence=-0.1)
@example(confidence=float("nan"))
@example(confidence=float("inf"))
@given(confidence=st.floats().filter(lambda c: not 0 < c < 1))
def test_invalid_confidence_level_is_unavailable(confidence: float) -> None:
    """An invalid probability cannot become a zero-width or infinite interval.

    Args:
        confidence: Any float outside the open unit interval, including NaN.

    """
    result = coefficient_intervals(
        _example(), confidence=confidence, reference="normal"
    )
    assert not bool(result.valid)
    assert int(result.status) == InferenceStatus.INVALID_REFERENCE
    assert np.isnan(np.asarray(result.estimate.ci[0])).all()


@pytest.mark.component
@pytest.mark.contract
def test_reference_arguments_and_restriction_shapes_are_checked() -> None:
    """Missing df and inconsistent shapes fail at the public configuration door."""
    uncertainty = _example()
    with pytest.raises(ValueError, match="required"):
        coefficient_intervals(uncertainty, reference="t")
    with pytest.raises(ValueError, match="required"):
        linear_restriction_test(
            uncertainty, jnp.array([[0.0, 1.0]]), 0.0, reference="f"
        )
    with pytest.raises(ValueError, match="does not use"):
        coefficient_intervals(uncertainty, reference="normal", degrees_of_freedom=6.0)
    with pytest.raises(ValueError, match="shape"):
        linear_restriction_test(uncertainty, jnp.array([[1.0]]), 0.0, reference="chi2")
    with pytest.raises(ValueError, match="one entry per row"):
        linear_restriction_test(
            uncertainty, jnp.eye(2), jnp.array([0.0, 1.0, 2.0]), reference="chi2"
        )
    with pytest.raises(ValueError, match="batch axes"):
        coefficient_intervals(
            uncertainty, confidence=jnp.array([0.9, 0.95]), reference="normal"
        )
    with pytest.raises(ValueError, match="real-valued"):
        linear_restriction_test(
            uncertainty, jnp.array([[0.0, 1.0j]]), 0.0, reference="chi2"
        )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
def test_student_critical_values_preserve_caller_precision(
    enable_x64: bool,  # noqa: FBT001
) -> None:  # pytest parameter
    """Tail accuracy and nearby decisions qualify the explicit precision boundary."""
    previous = jax.config.x64_enabled
    with jax.enable_x64(enable_x64):
        levels = jnp.concatenate(
            [jnp.tile(jnp.array([0.95, 0.99]), 6), jnp.array([0.9999])]
        )
        degrees = jnp.concatenate(
            [
                jnp.repeat(jnp.array([1.0, 5.0, 6.0, 100.0, 1000.0, 10000.0]), 2),
                jnp.array([1.0]),
            ]
        )
        critical, converged = jax.jit(_student_critical_value)(levels, degrees)
        expected = stats.t.isf((1.0 - np.asarray(levels)) / 2.0, np.asarray(degrees))
        supported = (
            np.ones(len(degrees), dtype=bool)
            if enable_x64
            else np.asarray(degrees) <= _FLOAT32_QUALIFIED_DF
        )
        np.testing.assert_array_equal(converged, supported)
        assert np.isnan(np.asarray(critical)[~supported]).all()
        tolerance = 2e-11 if enable_x64 else 1e-3
        np.testing.assert_allclose(
            np.asarray(critical)[supported],
            expected[supported],
            rtol=tolerance,
            atol=tolerance,
        )
        # At df<=10000 the tested float32 critical-value width error is <0.1%.
        # Coverage error is <0.0005 absolute (0.05 percentage point), and
        # decisions a 0.1% critical-value margin away agree.
        actual = np.asarray(critical)[supported]
        actual_df = np.asarray(degrees)[supported]
        alpha = 1.0 - np.asarray(levels)[supported]
        tail_tolerance = 2e-11 if enable_x64 else 5e-4
        np.testing.assert_allclose(
            2.0 * stats.t.sf(actual, actual_df), alpha, rtol=0.0, atol=tail_tolerance
        )
        decision_margin = 0.001
        assert np.all(
            2.0 * stats.t.sf(actual * (1.0 - decision_margin), actual_df) > alpha
        )
        assert np.all(
            2.0 * stats.t.sf(actual * (1.0 + decision_margin), actual_df) < alpha
        )
        assert critical.dtype == levels.dtype
        assert jax.config.x64_enabled == enable_x64
    assert jax.config.x64_enabled == previous


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
def test_f_tail_accuracy_uses_the_same_precision_boundary(enable_x64: bool) -> None:  # noqa: FBT001
    """Finite F tails retain SciPy agreement or report unsupported float32 df."""
    with jax.enable_x64(enable_x64):
        statistic = jnp.array([0.1, 1.0, 4.0, 10.0, 4.0])
        numerator = jnp.array([1.0, 2.0, 5.0, 2.0, 2.0])
        denominator = jnp.array([1.0, 6.0, 1000.0, 1000.0, 10000.0])
        actual = jax.jit(_f_survival)(statistic, numerator, denominator)
        expected = stats.f.sf(
            np.asarray(statistic), np.asarray(numerator), np.asarray(denominator)
        )
        supported = (
            np.ones(len(statistic), dtype=bool)
            if enable_x64
            else np.asarray(denominator) <= _FLOAT32_QUALIFIED_DF
        )
        assert np.isnan(np.asarray(actual)[~supported]).all()
        if enable_x64:
            np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=0.0)
        else:
            # Smaller-df small tails retain relative accuracy; the large-df
            # check has a separate stated absolute calibration budget.
            np.testing.assert_allclose(actual[:-1], expected[:-1], rtol=4e-4, atol=0.0)
            np.testing.assert_allclose(actual[-1], expected[-1], rtol=0.0, atol=5e-4)


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
def test_float32_f_reference_preserves_decisions_away_from_tested_thresholds() -> None:
    """A 0.001 p-value margin exceeds the measured qualified-domain tail error."""
    with jax.enable_x64(new_val=False):
        denominator = np.repeat(np.asarray([1000.0, 10000.0]), 6)
        numerator = np.tile(np.repeat(np.asarray([1.0, 2.0, 5.0]), 2), 2)
        thresholds = np.tile(np.asarray([0.05, 0.01]), 6)
        decision_margin = 0.001
        for sign in (-1.0, 1.0):
            intended = thresholds + sign * decision_margin
            statistic = stats.f.isf(intended, numerator, denominator)
            actual = np.asarray(
                jax.jit(_f_survival)(
                    jnp.asarray(statistic),
                    jnp.asarray(numerator),
                    jnp.asarray(denominator),
                )
            )
            np.testing.assert_allclose(actual, intended, rtol=0.0, atol=5e-4)
            np.testing.assert_array_equal(actual < thresholds, intended < thresholds)


@pytest.mark.component
@pytest.mark.contract
def test_large_df_float32_retains_results_with_unqualified_reference() -> None:
    """Numerical results outside the independently tested range remain visible."""
    with jax.enable_x64(new_val=False):
        uncertainty = _example()
        student = coefficient_intervals(
            uncertainty, reference="t", degrees_of_freedom=100000.0
        )
        fisher = linear_restriction_test(
            uncertainty, jnp.eye(2), 0.0, reference="f", degrees_of_freedom=100000.0
        )
        for result in (student, fisher):
            assert bool(result.valid)
            assert bool(result.finite_reference_outside_validated_range)
        normal = coefficient_intervals(uncertainty, reference="normal")
        chi2 = linear_restriction_test(uncertainty, jnp.eye(2), 0.0, reference="chi2")
        assert bool(normal.valid)
        assert bool(chi2.valid)
        assert not bool(normal.finite_reference_outside_validated_range)
        assert not bool(chi2.finite_reference_outside_validated_range)


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
def test_extreme_float32_f_tail_is_retained_but_unqualified() -> None:
    """A finite output can differ materially from an independent F reference."""
    with jax.enable_x64(new_val=False):
        uncertainty = _example()
        beta = uncertainty.fit.parameters[0]
        se = jnp.sqrt(uncertainty.covariance.values[0, 0])
        result = linear_restriction_test(
            uncertainty,
            jnp.asarray([[1.0, 0.0]]),
            beta - 2.0 * se,
            reference="f",
            degrees_of_freedom=1e8,
        )
        assert bool(result.valid)
        assert bool(result.finite_reference_outside_validated_range)
        pvalue = float(result.test.pvalue)
        expected = stats.f.sf(float(result.test.statistic), 1, 1e8)
        assert np.isfinite(pvalue)
        material_error = 0.01
        assert abs(pvalue - expected) > material_error


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
def test_mixed_precision_student_inputs_promote_without_narrowing() -> None:
    """The finite-reference solver accepts mixed precision under JIT."""
    levels = jnp.asarray(0.95, dtype=jnp.float32)
    degrees = jnp.asarray(5.0, dtype=jnp.float64)
    critical, converged = jax.jit(_student_critical_value)(levels, degrees)
    assert bool(converged)
    assert critical.dtype == jnp.float64
    assert float(critical) == pytest.approx(
        stats.t.isf((1.0 - float(levels)) / 2.0, 5), rel=1e-11
    )


@pytest.mark.integration
@pytest.mark.contract
def test_float64_restrictions_preserve_precision_of_float32_fit() -> None:
    """Restriction and target precision participate in the Wald calculation."""
    x = jnp.stack(
        [jnp.ones(8, dtype=jnp.float32), jnp.arange(8, dtype=jnp.float32)], axis=1
    )
    y = x @ jnp.asarray([1.2, 0.7], dtype=jnp.float32) + jnp.asarray(
        [0.2, -0.4, 0.1, 0.3, -0.2, 0.5, -0.1, 0.1], dtype=jnp.float32
    )
    uncertainty = classical_covariance(LeastSquares().estimate(LinearProblem(x, y)))
    result = linear_restriction_test(
        uncertainty,
        jnp.asarray([[1.0, 0.0]], dtype=jnp.float64),
        jnp.asarray([0.5], dtype=jnp.float64),
        reference="chi2",
    )
    assert bool(result.valid)
    assert result.test.statistic.dtype == jnp.float64


@pytest.mark.component
@pytest.mark.contract
def test_failed_critical_value_inversion_cannot_authorize_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The interval door preserves a numerical failure from its maintained solver."""

    def failed_critical(
        level: jax.Array, degrees: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        del degrees
        return jnp.full_like(level, jnp.nan), jnp.zeros_like(level, dtype=bool)

    monkeypatch.setattr(parameter_module, "_student_critical_value", failed_critical)
    result = coefficient_intervals(_example(), reference="t", degrees_of_freedom=6.0)
    assert not bool(result.valid)
    assert int(result.status) == InferenceStatus.NUMERICAL_FAILURE
    assert np.isnan(np.asarray(result.estimate.ci[0])).all()
