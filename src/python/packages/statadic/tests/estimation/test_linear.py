# standalone package-maintenance entry point
"""Complete-data linear fit contracts, against independent NumPy references."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array
    from statadic.estimation.linear import LinearFit


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.estimation.linear import (
    LeastSquares,
    LinearFit,
    LinearProblem,
)
from statadic.inference.linear_covariance import classical_covariance
from statadic.inference.parameters import coefficient_intervals, linear_restriction_test

from .._budgets import examples
from .._validation.linear import LinearCase, linear_cases
from .._validation.numeric import Tolerance, assert_finite_close


def _sample() -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(
        [
            [1.0, -2.0, 1.0],
            [1.0, -1.0, -0.5],
            [1.0, 0.0, 0.5],
            [1.0, 0.5, 1.5],
            [1.0, 1.5, -1.0],
            [1.0, 2.5, 0.0],
            [1.0, 3.5, 1.0],
        ]
    )
    return design, np.asarray([-1.0, 0.5, 0.0, 2.0, 1.0, 3.5, 2.5])


_SAMPLE_WEIGHTS = np.asarray([0.5, 1.0, 3.0, 0.8, 2.0, 1.2, 5.0])
# Full-rank cases with a triangular anchor and nonzero residual noise (n=7, p=3).
_CASES = linear_cases(n=7, p=3)


def _sample_case() -> LinearCase:
    design, outcome = _sample()
    return LinearCase(design, outcome, _SAMPLE_WEIGHTS)


@pytest.mark.component
@pytest.mark.reference
@examples(40)
@example(case=_sample_case())
@given(case=_CASES)
def test_ols_fit_retains_complete_state_and_stable_qr_covariance_map(
    case: LinearCase,
) -> None:
    """Parameters, residuals and maps match independent NumPy SVD calculations."""
    design, outcome = case.design, case.outcome
    expected = np.linalg.lstsq(design, outcome, rcond=None)[0]
    result = LeastSquares().estimate(
        LinearProblem(design, outcome, parameter_names=("intercept", "x", "z"))
    )
    assert isinstance(result, LinearFit)
    assert result.valid
    assert result.identified
    assert result.input_valid
    assert result.rank == design.shape[1]
    assert result.n_observations == len(outcome)
    assert result.df_resid == len(outcome) - design.shape[1]
    np.testing.assert_allclose(result.parameters, expected, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result.fitted_values, design @ expected, atol=1e-13)
    np.testing.assert_allclose(
        result.residuals, outcome - design @ expected, atol=1e-13
    )
    np.testing.assert_allclose(
        result.scores.values, design * np.asarray(result.residuals)[:, None], atol=1e-13
    )
    np.testing.assert_allclose(
        result.jacobian.values, -(design.T @ design) / len(outcome), atol=1e-13
    )
    np.testing.assert_allclose(
        result.coefficient_map, np.linalg.pinv(design), atol=1e-13
    )
    np.testing.assert_allclose(
        result.normalized_covariance, np.linalg.inv(design.T @ design), atol=1e-13
    )
    np.testing.assert_allclose(
        result.leverage, np.diag(design @ np.linalg.pinv(design)), atol=1e-13
    )
    expected_sse = np.sum((outcome - design @ expected) ** 2)
    assert result.sse == pytest.approx(expected_sse)
    assert result.weighted_sse == pytest.approx(expected_sse)
    assert result.common_scale == pytest.approx(expected_sse / int(result.df_resid))
    np.testing.assert_allclose(
        result.predict(design, parameter_names=("intercept", "x", "z")),
        result.fitted_values,
    )


@pytest.mark.component
@pytest.mark.property
@examples(40)
@example(case=_sample_case())
@given(case=_CASES)
def test_wls_uses_unequal_relative_precisions_and_has_rescaling_invariance(
    case: LinearCase,
) -> None:
    """Positive common rescaling changes estimated scale, not beta or covariance."""
    design, outcome, weights = case.design, case.outcome, case.precision
    root_weights = np.sqrt(weights)
    expected = np.linalg.lstsq(
        root_weights[:, None] * design, root_weights * outcome, rcond=None
    )[0]
    fit = LeastSquares().estimate(LinearProblem(design, outcome, weights))
    multiplier = 7.3
    rescaled = LeastSquares().estimate(
        LinearProblem(design, outcome, multiplier * weights)
    )
    assert fit.valid
    assert rescaled.valid
    np.testing.assert_allclose(fit.parameters, expected, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        rescaled.parameters, fit.parameters, rtol=1e-12, atol=1e-13
    )
    residuals = outcome - design @ expected
    np.testing.assert_allclose(
        fit.scores.values, weights[:, None] * design * residuals[:, None], atol=1e-12
    )
    np.testing.assert_allclose(
        fit.jacobian.values,
        -(design.T @ (weights[:, None] * design)) / len(outcome),
        atol=1e-13,
    )
    assert fit.weighted_sse == pytest.approx(np.sum(weights * residuals**2))
    assert fit.sse == pytest.approx(np.sum(residuals**2))
    assert rescaled.common_scale == pytest.approx(multiplier * float(fit.common_scale))
    np.testing.assert_allclose(
        rescaled.common_scale * rescaled.normalized_covariance,
        fit.common_scale * fit.normalized_covariance,
        rtol=1e-12,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        fit.coefficient_map, np.linalg.pinv(root_weights[:, None] * design), atol=1e-13
    )
    np.testing.assert_allclose(
        fit.coefficient_map @ (root_weights * outcome), fit.parameters, atol=1e-13
    )


@pytest.mark.component
@pytest.mark.property
def test_unequal_precisions_change_the_sample_fit() -> None:
    """The retained sample's weights are informative, not accidentally uniform."""
    case = _sample_case()
    weighted = LeastSquares().estimate(
        LinearProblem(case.design, case.outcome, case.precision)
    )
    unweighted = LeastSquares().estimate(LinearProblem(case.design, case.outcome))
    assert not np.allclose(weighted.parameters, unweighted.parameters)


@pytest.mark.component
@pytest.mark.numerical
def test_intercept_is_an_explicit_design_column() -> None:
    """Neither fit nor prediction silently adds or removes an intercept."""
    design, outcome = _sample()
    without = LeastSquares().estimate(LinearProblem(design[:, 1:], outcome))
    with_intercept = LeastSquares().estimate(LinearProblem(design, outcome))
    expected = np.linalg.lstsq(design[:, 1:], outcome, rcond=None)[0]
    np.testing.assert_allclose(without.parameters, expected, atol=1e-13)
    assert without.parameters.shape == (design.shape[1] - 1,)
    assert not np.allclose(without.fitted_values, with_intercept.fitted_values)
    assert not np.allclose(without.residuals, with_intercept.residuals)


@pytest.mark.component
@pytest.mark.reference
@examples(40)
@example(case=_sample_case(), dtype=np.float32)
@example(case=_sample_case(), dtype=np.float64)
@given(case=_CASES, dtype=st.sampled_from([np.float32, np.float64]))
def test_linear_fitting_preserves_caller_dtypes_and_configuration(
    case: LinearCase, dtype: type
) -> None:
    """Float32 and float64 retain rank, parameter scale and precision ownership."""
    design = case.design.astype(dtype)
    outcome = case.outcome.astype(dtype)
    weights = case.precision.astype(dtype)
    precision_before = jax.config.x64_enabled
    expected = np.linalg.lstsq(
        np.sqrt(weights)[:, None] * design, np.sqrt(weights) * outcome, rcond=None
    )[0]
    # The well-conditioned fixture uses a small QR solve and matrix products.
    # 100 eps allows float32 rounding while preserving rank and fitted behavior.
    tolerance = 100 * np.finfo(dtype).eps
    for fit in (
        LeastSquares().estimate(LinearProblem(design, outcome)),
        LeastSquares().estimate(LinearProblem(design, outcome, weights)),
    ):
        assert fit.parameters.dtype == dtype
        assert fit.design.dtype == dtype
        assert fit.normalized_covariance.dtype == dtype
        assert fit.valid
        assert fit.predict(design).dtype == dtype
    np.testing.assert_allclose(
        LeastSquares().estimate(LinearProblem(design, outcome, weights)).parameters,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    assert jax.config.x64_enabled == precision_before


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(case=_sample_case())
@given(case=_CASES)
def test_prediction_preserves_higher_precision_inputs_and_rejects_complex_design(
    case: LinearCase,
) -> None:
    """Prediction follows real dtype promotion without downcasting new covariates."""
    design, outcome = case.design, case.outcome
    fit = LeastSquares().estimate(
        LinearProblem(design.astype(np.float32), outcome.astype(np.float32))
    )
    assert fit.predict(design).dtype == design.dtype
    with pytest.raises(ValueError, match="real numeric design"):
        fit.predict(design.astype(np.complex128) + 1j)


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(case=_sample_case())
@given(case=_CASES)
def test_fit_owns_immutable_copies_of_host_arrays(case: LinearCase) -> None:
    """Mutating caller buffers cannot change the stored sample or fitted scores."""
    design, outcome = case.design.copy(), case.outcome.copy()
    weights = np.ones_like(outcome)
    expected_design, expected_outcome, expected_weights = (
        design.copy(),
        outcome.copy(),
        weights.copy(),
    )
    fit = LeastSquares().estimate(LinearProblem(design, outcome, weights))
    fit.parameters.block_until_ready()
    expected_scores = np.asarray(fit.scores.values).copy()
    design[:] = 99.0
    outcome[:] = -99.0
    weights[:] = 7.0
    np.testing.assert_array_equal(fit.design, expected_design)
    np.testing.assert_array_equal(fit.outcome, expected_outcome)
    np.testing.assert_array_equal(fit.relative_precision, expected_weights)
    np.testing.assert_array_equal(fit.scores.values, expected_scores)


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(case=_sample_case(), location="design")
@example(case=_sample_case(), location="outcome")
@example(case=_sample_case(), location="weights_nan")
@example(case=_sample_case(), location="weights_zero")
@example(case=_sample_case(), location="weights_negative")
@given(
    case=_CASES,
    location=st.sampled_from(
        ["design", "outcome", "weights_nan", "weights_zero", "weights_negative"]
    ),
)
def test_invalid_inputs_remain_unavailable_without_row_deletion(
    case: LinearCase, location: str
) -> None:
    """Nonfinite data and invalid relative weights cannot create ordinary fits."""
    design, outcome = case.design.copy(), case.outcome.copy()
    weights = np.ones_like(outcome)
    if location == "design":
        design[1, 0] = np.nan
    elif location == "outcome":
        outcome[1] = np.inf
    else:
        weights[1] = {
            "weights_nan": np.nan,
            "weights_zero": 0.0,
            "weights_negative": -1.0,
        }[location]

    def evaluate_fit(x: jax.Array, y: jax.Array, w: jax.Array) -> LinearFit:
        return LeastSquares().estimate(LinearProblem(x, y, w))

    evaluate_fit = jax.jit(evaluate_fit)(design, outcome, weights)
    assert not evaluate_fit.input_valid
    assert not evaluate_fit.identified
    assert not evaluate_fit.valid
    assert evaluate_fit.n_observations == len(outcome)
    assert np.isnan(np.asarray(evaluate_fit.parameters)).all()
    assert np.isnan(np.asarray(evaluate_fit.scores.values)).all()
    assert np.isnan(np.asarray(evaluate_fit.jacobian.values)).all()
    assert np.isnan(np.asarray(evaluate_fit.normalized_covariance)).all()


@pytest.mark.component
@pytest.mark.contract
def test_singular_and_wide_designs_keep_width_without_valid_parameters() -> None:
    """Rank cleaning or minimum-norm answers cannot substitute for identification."""
    for design in (np.ones((5, 2)), np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])):
        fit = LeastSquares().estimate(
            LinearProblem(design, np.arange(design.shape[0], dtype=float))
        )
        assert fit.input_valid
        assert not fit.identified
        assert not fit.valid
        assert fit.rank < design.shape[1]
        assert fit.parameters.shape == (design.shape[1],)
        assert fit.coefficient_map.shape == (design.shape[1], design.shape[0])
        assert np.isnan(np.asarray(fit.parameters)).all()
        assert np.isnan(np.asarray(fit.coefficient_map)).all()
        assert np.isnan(np.asarray(fit.leverage)).all()


@pytest.mark.component
@pytest.mark.contract
def test_saturated_point_fit_has_no_estimated_common_scale() -> None:
    """Identification is independent of positive residual degrees of freedom."""
    fit = LeastSquares().estimate(LinearProblem(jnp.eye(2), jnp.asarray([1.5, -2.0])))
    assert fit.identified
    assert fit.valid
    assert fit.df_resid == 0
    assert jnp.isnan(fit.common_scale)
    np.testing.assert_array_equal(fit.parameters, [1.5, -2.0])


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(case=_sample_case())
@given(case=_CASES)
def test_fit_results_retain_mixed_statuses_under_jit_and_vmap(
    case: LinearCase,
) -> None:
    """Full-rank, nonfinite and collinear lanes retain independent array state."""
    design, outcome = case.design.copy(), case.outcome.copy()
    singular = design.copy()
    singular[:, 2] = singular[:, 1]
    outcomes = np.stack([outcome, outcome.copy(), outcome])
    outcomes[1, 0] = np.nan
    designs = np.stack([design, design, singular])

    def evaluate_fit(x: jax.Array, y: jax.Array) -> LinearFit:
        return LeastSquares().estimate(LinearProblem(x, y))

    result = jax.jit(jax.vmap(evaluate_fit))(designs, outcomes)
    np.testing.assert_array_equal(result.valid, [True, False, False])
    np.testing.assert_array_equal(result.identified, [True, False, False])
    np.testing.assert_array_equal(result.input_valid, [True, False, True])
    np.testing.assert_array_equal(
        result.rank, [design.shape[1], -1, design.shape[1] - 1]
    )
    np.testing.assert_allclose(
        result.parameters[0],
        LeastSquares().estimate(LinearProblem(design, outcome)).parameters,
        atol=1e-13,
    )
    assert result.parameters.shape == (len(outcomes), design.shape[1])

    # A stacked problem is itself a pytree: mapping the brick over it is the same.
    def evaluate_fit_4(problem: LinearProblem) -> LinearFit:
        return LeastSquares().estimate(problem)

    stacked = jax.jit(jax.vmap(evaluate_fit_4))(
        LinearProblem(jnp.asarray(designs), jnp.asarray(outcomes))
    )
    np.testing.assert_array_equal(stacked.valid, result.valid)
    np.testing.assert_array_equal(stacked.parameters, result.parameters)
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(result))

    def evaluate_predict(fit: LinearFit, x: jax.Array) -> Array:
        return fit.predict(x)

    predicted = jax.jit(jax.vmap(evaluate_predict))(result, designs)
    np.testing.assert_allclose(predicted[0], result.fitted_values[0], atol=1e-13)
    assert np.isnan(np.asarray(predicted[1:])).all()


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(case=_sample_case())
@given(case=_CASES)
def test_named_prediction_rejects_missing_or_permuted_column_layouts(
    case: LinearCase,
) -> None:
    """Named prediction checks the schema; unnamed prediction stays positional."""
    design, outcome = case.design, case.outcome
    names = ("intercept", "x", "z")
    fit = LeastSquares().estimate(LinearProblem(design, outcome, parameter_names=names))
    for invalid_names in (None, ("intercept", "z", "x")):
        with pytest.raises(ValueError, match="column order"):
            fit.predict(design, parameter_names=invalid_names)
    with pytest.raises(ValueError, match="fitted width"):
        fit.predict(design[:, :2], parameter_names=names)
    np.testing.assert_allclose(
        fit.predict(design, parameter_names=names), fit.fitted_values
    )


@pytest.mark.component
@pytest.mark.contract
@examples(40)
@example(case=_sample_case(), names=("x",))
@example(case=_sample_case(), names=("x", "x", "z"))
@example(case=_sample_case(), names=("", "x", "z"))
@given(
    case=_CASES,
    names=st.one_of(
        st.lists(st.sampled_from(["x", "z"]), min_size=1, max_size=2).map(tuple),
        st.sampled_from(["x", "z", "w"]).map(lambda name: (name, name, "q")),
        st.just(("", "x", "z")),
    ),
)
def test_parameter_names_require_a_complete_unique_layout(
    case: LinearCase, names: tuple[str, ...]
) -> None:
    """Invalid schemas are rejected before constructing a fitted state."""
    design, outcome = case.design, case.outcome
    with pytest.raises(ValueError, match="unique nonempty names"):
        LeastSquares().estimate(LinearProblem(design, outcome, parameter_names=names))


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("design", [np.zeros((0, 2)), np.zeros((3, 0)), np.ones(3)])
def test_fit_rejects_empty_or_nonmatrix_designs(design: np.ndarray) -> None:
    """Structural input errors cannot silently alter sample or design width."""
    with pytest.raises(ValueError, match="nonempty rank-2"):
        LeastSquares().estimate(LinearProblem(design, np.ones(3)))


@pytest.mark.component
@pytest.mark.contract
def test_fit_rejects_response_and_weight_shape_mismatches() -> None:
    """Every supplied vector shares the exact complete design row axis."""
    design, outcome = _sample()
    with pytest.raises(ValueError, match="outcome must be a vector"):
        LeastSquares().estimate(LinearProblem(design, outcome[:-1]))
    with pytest.raises(ValueError, match="relative_precision must be a vector"):
        LeastSquares().estimate(
            LinearProblem(design, outcome, np.ones((len(outcome), 1)))
        )


@pytest.mark.component
@pytest.mark.contract
def test_finite_qr_output_cannot_certify_overflowing_original_scale_state() -> None:
    """Whitening may hide an unrepresentable original-scale prediction."""
    designs = jnp.asarray([[[1.0], [2.0]], [[1.0], [1e20]]], dtype=jnp.float32)
    outcomes = jnp.asarray([[1.0, 4.0], [1e30, 0.0]], dtype=jnp.float32)
    weights = jnp.asarray([[1.0, 1.0], [1.0, 1e-36]], dtype=jnp.float32)
    eager = LeastSquares().estimate(LinearProblem(designs[1], outcomes[1], weights[1]))

    def evaluate_fit(x: jax.Array, y: jax.Array, w: jax.Array) -> LinearFit:
        return LeastSquares().estimate(LinearProblem(x, y, w))

    result = jax.jit(jax.vmap(evaluate_fit))(designs, outcomes, weights)
    np.testing.assert_array_equal(result.input_valid, [True, True])
    np.testing.assert_array_equal(result.identified, [True, True])
    np.testing.assert_array_equal(result.solver.converged, [True, True])
    np.testing.assert_array_equal(result.valid, [True, False])
    assert eager.input_valid
    assert eager.identified
    assert eager.solver.converged
    assert not eager.valid
    assert np.isfinite(np.asarray(eager.solver.value)).all()

    def second_lane(value: jax.Array) -> Array:
        return value[1]

    for evaluate_fit in (eager, jax.tree.map(second_lane, result)):
        for unavailable in (
            evaluate_fit.parameters,
            evaluate_fit.fitted_values,
            evaluate_fit.residuals,
            evaluate_fit.scores.values,
            evaluate_fit.jacobian.values,
            evaluate_fit.sse,
            evaluate_fit.weighted_sse,
            evaluate_fit.common_scale,
        ):
            assert np.isnan(np.asarray(unavailable)).all()
    np.testing.assert_array_equal(result.design, designs)
    np.testing.assert_array_equal(result.outcome, outcomes)
    np.testing.assert_array_equal(result.relative_precision, weights)


@pytest.mark.component
@pytest.mark.reference
@examples(20)
@pytest.mark.parametrize(("n", "p"), [(8, 2), (12, 3), (7, 5)])
@pytest.mark.parametrize("weighted", [False, True])
@given(data=st.data())
def test_generated_complete_linear_state_matches_independent_qr(
    n: int, p: int, *, weighted: bool, data: st.DataObject
) -> None:
    """Constructive rank/noise guarantees make every fitted output an obligation."""
    case = data.draw(linear_cases(n=n, p=p))
    x, y = case.design, case.outcome
    weights = case.precision if weighted else np.ones(n)
    roots = np.sqrt(weights)
    q, r = np.linalg.qr(roots[:, None] * x)
    beta = np.linalg.solve(r, q.T @ (roots * y))
    residuals = y - x @ beta
    weighted_sse = np.sum(weights * residuals**2)
    inverse_r = np.linalg.solve(r, np.eye(p))
    covariance = weighted_sse / (n - p) * inverse_r @ inverse_r.T
    tolerance = Tolerance(1e-10, 1e-11, "Bounded triangular anchor and float64 QR.")

    fit = (
        LeastSquares().estimate(LinearProblem(x, y, weights))
        if weighted
        else LeastSquares().estimate(LinearProblem(x, y))
    )
    uncertainty = classical_covariance(fit)

    assert fit.valid
    assert fit.identified
    assert uncertainty.valid
    assert int(fit.rank) == p
    assert int(fit.n_observations) == n
    assert int(fit.df_resid) == n - p
    for label, actual, expected in (
        ("coefficients", fit.parameters, beta),
        ("predictions", fit.fitted_values, x @ beta),
        ("residuals", fit.residuals, residuals),
        ("scores", fit.scores.values, x * (weights * residuals)[:, None]),
        ("raw SSE", fit.sse, np.sum(residuals**2)),
        ("weighted SSE", fit.weighted_sse, weighted_sse),
        ("common scale", fit.common_scale, weighted_sse / (n - p)),
        ("covariance", uncertainty.covariance.values, covariance),
        ("weighted score orthogonality", x.T @ (weights * residuals), np.zeros(p)),
    ):
        assert_finite_close(actual, expected, tolerance=tolerance, label=label)


@pytest.mark.component
@pytest.mark.property
@examples(20)
@given(data=st.data(), multiplier=st.sampled_from([0.25, 3.0, 16.0]))
def test_generated_precision_scaling_and_named_permutations_preserve_inference(
    data: st.DataObject, multiplier: float
) -> None:
    """Only scale-dependent outputs change when precision units and layout change."""
    case = data.draw(linear_cases(n=10, p=3))
    rows = np.asarray(data.draw(st.permutations(tuple(range(10)))))
    columns = np.asarray(data.draw(st.permutations((0, 1, 2))))
    names = ("a", "b", "c")
    permuted_names = tuple(names[column] for column in columns)
    restriction = np.asarray([[1.0, -0.3, 0.5], [0.2, 1.0, -0.7]])
    target = np.asarray([0.1, -0.2])

    fit = LeastSquares().estimate(
        LinearProblem(case.design, case.outcome, case.precision, parameter_names=names)
    )
    changed = LeastSquares().estimate(
        LinearProblem(
            case.design[rows][:, columns],
            case.outcome[rows],
            multiplier * case.precision[rows],
            parameter_names=permuted_names,
        )
    )
    original_covariance, changed_covariance = (
        classical_covariance(result) for result in (fit, changed)
    )
    original_test = linear_restriction_test(
        original_covariance,
        restriction,
        target,
        parameter_names=names,
        reference="chi2",
    )
    changed_test = linear_restriction_test(
        changed_covariance,
        restriction[:, columns],
        target,
        parameter_names=permuted_names,
        reference="chi2",
    )
    original_intervals = coefficient_intervals(original_covariance, reference="normal")
    changed_intervals = coefficient_intervals(changed_covariance, reference="normal")

    assert fit.valid
    assert changed.valid
    assert original_test.valid
    assert changed_test.valid
    assert changed.parameter_names == permuted_names
    np.testing.assert_allclose(
        changed.parameters, np.asarray(fit.parameters)[columns], atol=1e-11
    )
    np.testing.assert_allclose(
        changed.fitted_values, np.asarray(fit.fitted_values)[rows], atol=1e-11
    )
    np.testing.assert_allclose(changed.sse, fit.sse, atol=1e-11)
    np.testing.assert_allclose(
        changed.weighted_sse, multiplier * fit.weighted_sse, atol=1e-11
    )
    np.testing.assert_allclose(
        changed.common_scale, multiplier * fit.common_scale, atol=1e-11
    )
    np.testing.assert_allclose(
        changed_covariance.covariance.values,
        np.asarray(original_covariance.covariance.values)[np.ix_(columns, columns)],
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        changed_test.test.statistic,
        original_test.test.statistic,
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        changed_test.test.pvalue, original_test.test.pvalue, rtol=1e-10, atol=1e-11
    )
    np.testing.assert_allclose(
        changed_intervals.estimate.ci,
        np.asarray(original_intervals.estimate.ci)[:, columns],
        rtol=1e-10,
        atol=1e-11,
    )


@pytest.mark.component
@pytest.mark.reference
def test_mixed_precision_fit_preserves_input_arrays() -> None:
    """The retained design and response keep their own input precision."""
    design = jnp.asarray([[1, 0], [1, 1], [1, 2]], dtype=jnp.float32)
    outcome = jnp.asarray([1, 2, 4], dtype=jnp.float64)
    fit = LeastSquares().estimate(LinearProblem(design, outcome))
    assert fit.design.dtype == jnp.float32
    assert fit.outcome.dtype == jnp.float64
    assert fit.parameters.dtype == jnp.float64
    np.testing.assert_allclose(
        fit.parameters, np.linalg.lstsq(design, outcome, rcond=None)[0]
    )
