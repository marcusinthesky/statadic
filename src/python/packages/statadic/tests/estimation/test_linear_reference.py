# standalone package-maintenance entry point
"""Matched public fit, covariance and parameter inference against statsmodels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from statadic.estimation.linear import (
    LeastSquares,
    LinearProblem,
)
from statadic.inference.linear_covariance import (
    classical_covariance,
    cluster_covariance,
    hac_covariance,
    hc_covariance,
)
from statadic.inference.parameters import coefficient_intervals, linear_restriction_test

from .._budgets import examples
from .._validation.linear import LinearCase, linear_cases, perturbed_linear_cases

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )
    from types import ModuleType


def _sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Supply complete nonspherical data with deliberately unequal precisions."""
    time = np.arange(18, dtype=float)
    design = np.column_stack((np.ones_like(time), time / 7, np.cos(time * 0.7)))
    outcome = design @ np.asarray([0.7, 1.3, -0.4]) + (0.2 + time / 30) * np.sin(
        time * 1.9
    )
    precision = 0.3 + (time % 5 + 1) ** 2
    return design, outcome, precision


def _sample_case() -> LinearCase:
    design, outcome, precision = _sample()
    return LinearCase(design, outcome, precision)


_CASES = perturbed_linear_cases(_sample_case())


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
@examples(20)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("kind", ["classical", "HC0", "HC1", "HC2", "HC3", "HAC"])
@example(case=_sample_case())
@given(case=_CASES)
def test_public_linear_workflow_matches_statsmodels(  # noqa: PLR0914
    case: LinearCase,
    weighted: bool,  # pytest parameter  # noqa: FBT001
    kind: Literal["classical", "HC0", "HC1", "HC2", "HC3", "HAC"],
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """The actual fit supplies every covariance and the reported t/F inference."""
    api = optional_reference("statsmodels.api")
    design, outcome, precision = case.design, case.outcome, case.precision
    names = ("intercept", "trend", "cycle")
    if weighted:
        observed_fit = LeastSquares().estimate(
            LinearProblem(design, outcome, precision, parameter_names=names)
        )
        reference_fit = api.WLS(outcome, design, weights=precision).fit()
    else:
        observed_fit = LeastSquares().estimate(
            LinearProblem(design, outcome, parameter_names=names)
        )
        reference_fit = api.OLS(outcome, design).fit()
    assert observed_fit.parameter_names == names
    assert int(observed_fit.df_resid) == reference_fit.df_resid
    assert int(observed_fit.rank) == design.shape[1]
    np.testing.assert_allclose(
        observed_fit.parameters, reference_fit.params, rtol=1e-11
    )
    np.testing.assert_allclose(
        observed_fit.fitted_values, reference_fit.fittedvalues, atol=1e-12
    )
    np.testing.assert_allclose(observed_fit.residuals, reference_fit.resid, atol=1e-12)
    np.testing.assert_allclose(
        observed_fit.sse, reference_fit.resid @ reference_fit.resid, rtol=1e-11
    )
    np.testing.assert_allclose(observed_fit.weighted_sse, reference_fit.ssr, rtol=1e-11)
    np.testing.assert_allclose(
        observed_fit.common_scale, reference_fit.scale, rtol=1e-11
    )
    np.testing.assert_allclose(
        observed_fit.predict(design, parameter_names=names),
        reference_fit.predict(design),
        atol=1e-12,
    )
    held_out = np.asarray([[1.0, -0.8, 0.4], [1.0, 1.4, -0.6]])
    np.testing.assert_allclose(
        observed_fit.predict(held_out, parameter_names=names),
        reference_fit.predict(held_out),
        atol=1e-12,
    )
    if kind == "classical":
        uncertainty = classical_covariance(observed_fit)
        reference = reference_fit
    elif kind == "HAC":
        bandwidth = 2
        correction = len(outcome) / (len(outcome) - design.shape[1])
        uncertainty = hac_covariance(
            observed_fit, bandwidth, finite_sample_correction=correction
        )
        reference = reference_fit.get_robustcov_results(
            cov_type="HAC", maxlags=bandwidth, use_correction=True, use_t=True
        )
    else:
        uncertainty = hc_covariance(observed_fit, kind=kind)
        reference = reference_fit.get_robustcov_results(cov_type=kind, use_t=True)
    assert uncertainty.valid
    np.testing.assert_allclose(
        uncertainty.covariance.values, reference.cov_params(), rtol=1e-10, atol=1e-13
    )
    # Changing covariance consumes the same immutable fit rather than refitting it.
    assert uncertainty.fit is observed_fit
    intervals = coefficient_intervals(
        uncertainty, reference="t", degrees_of_freedom=observed_fit.df_resid
    )
    assert intervals.valid
    np.testing.assert_allclose(intervals.estimate.se, reference.bse, rtol=1e-10)
    np.testing.assert_allclose(
        jnp.stack(intervals.estimate.ci, axis=-1),
        reference.conf_int(),
        rtol=1e-9,
        atol=1e-11,
    )
    restriction = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tested = linear_restriction_test(
        uncertainty,
        restriction,
        np.zeros(2),
        parameter_names=names,
        reference="f",
        degrees_of_freedom=observed_fit.df_resid,
    )
    matched = reference.f_test((restriction, np.zeros(2)))
    assert tested.valid
    np.testing.assert_allclose(tested.test.statistic, matched.fvalue, rtol=1e-10)
    np.testing.assert_allclose(tested.test.pvalue, matched.pvalue, rtol=1e-9)
    assert tested.numerator_df == matched.df_num
    assert tested.denominator_df == matched.df_denom


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("statsmodels")
@examples(20)
@pytest.mark.parametrize("weighted", [False, True])
@example(case=_sample_case())
@given(case=_CASES)
def test_one_way_cluster_matches_explicit_paper6_correction(
    case: LinearCase,
    weighted: bool,  # pytest parameter  # noqa: FBT001
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Match uncorrected cluster covariance times only G/(G-1)."""
    api = optional_reference("statsmodels.api")
    sandwich = optional_reference("statsmodels.stats.sandwich_covariance")
    design, outcome, precision = case.design, case.outcome, case.precision
    groups = np.arange(len(outcome)) % 6
    n_clusters = 6
    if weighted:
        observed_fit = LeastSquares().estimate(
            LinearProblem(design, outcome, precision)
        )
        reference_fit = api.WLS(outcome, design, weights=precision).fit()
    else:
        observed_fit = LeastSquares().estimate(LinearProblem(design, outcome))
        reference_fit = api.OLS(outcome, design).fit()
    multiplier = n_clusters / (n_clusters - 1)
    uncertainty = cluster_covariance(
        observed_fit,
        jnp.asarray(groups),
        n_clusters=n_clusters,
        finite_sample_correction=multiplier,
    )
    expected = (
        sandwich.cov_cluster(reference_fit, groups, use_correction=False) * multiplier
    )
    assert uncertainty.valid
    np.testing.assert_allclose(uncertainty.covariance.values, expected, rtol=1e-10)
    assert uncertainty.n_clusters == n_clusters
    # Normal calibration is a separate choice for this same covariance.
    intervals = coefficient_intervals(uncertainty, reference="normal")
    matched = reference_fit.get_robustcov_results(
        cov_type="cluster",
        groups=groups,
        use_correction=False,
        df_correction=False,
        use_t=False,
    )
    np.testing.assert_allclose(
        intervals.estimate.se, matched.bse * np.sqrt(multiplier), rtol=1e-10
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("kind", ["HC0", "HC1", "HC2", "HC3", "HAC"])
@examples(40)
@given(case=linear_cases(n=12, p=3))
def test_generated_wls_covariance_matches_independent_score_products(
    kind: Literal["HC0", "HC1", "HC2", "HC3", "HAC"], case: LinearCase
) -> None:
    """Replicated rank anchors keep leverage and residual variation well defined."""
    anchor = case.design[:3]
    x = np.tile(anchor, (4, 1))
    y = x @ np.linalg.solve(anchor, case.outcome[:3]) + np.repeat(
        [-0.4, -0.1, 0.2, 0.6], 3
    )
    weights = case.precision
    bread = np.linalg.inv(x.T @ (weights[:, None] * x))
    beta = bread @ x.T @ (weights * y)
    residual = y - x @ beta
    scores = x * (weights * residual)[:, None]
    leverage = weights * np.einsum("ij,jk,ik->i", x, bread, x)
    if kind in {"HC2", "HC3"}:
        exponent = 0.5 if kind == "HC2" else 1.0
        scores = scores / (1 - leverage[:, None]) ** exponent
    meat = scores.T @ scores
    if kind == "HAC":
        for lag in (1, 2):
            cross = scores[lag:].T @ scores[:-lag]
            meat += (1 - lag / 3) * (cross + cross.T)
    if kind == "HC1":
        meat *= 12 / 9
    expected = bread @ meat @ bread

    fit = LeastSquares().estimate(LinearProblem(x, y, weights))
    actual = hac_covariance(fit, 2) if kind == "HAC" else hc_covariance(fit, kind=kind)

    assert actual.valid
    np.testing.assert_allclose(
        actual.covariance.values, expected, rtol=1e-10, atol=1e-12
    )
