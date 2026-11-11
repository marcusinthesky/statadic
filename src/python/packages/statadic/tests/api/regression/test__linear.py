"""Shared fit statistics: the information helper and the linear family's flow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from statadic.api.data import RelativePrecision
from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC, ModelBased
from statadic.api.regression import Regression

from ..._budgets import examples
from ..._validation.linear import linear_cases, ols_oracle
from ..._validation.numeric import Tolerance, assert_finite_close

if TYPE_CHECKING:
    from statadic.api.capabilities import SummarizesLikelihood
    from statadic.api.regression import LinearResult
    from statadic.api.reporting import Metric

    from ..._validation.linear import LinearCase

N = 30
K_EXPLICIT = 3
K_INDICATORS = 4
K_AIC = 3
_STATISTICS = ("n_parameters", "log_likelihood", "aic", "bic", "hqic")


def _data() -> dict[str, jax.Array]:
    x = jnp.linspace(0.0, 4.0, N)
    noise = 0.3 * jax.random.normal(jax.random.key(5), x.shape)
    group = (jnp.arange(N) % 2).astype(x.dtype)
    return {
        "x": x,
        "y": 1.0 + 0.7 * x + 0.5 * group + noise,
        "g1": group,
        "g0": 1.0 - group,
        "w": 0.5 + x / 2,
    }


def _by_name(metrics: tuple[Metric, ...]) -> dict[str, Metric]:
    return {metric.name: metric for metric in metrics}


@pytest.mark.unit
@pytest.mark.contract
def test_explicit_intercept_reports_centred_r_squared() -> None:
    """An explicit intercept gives centred R-squared and the adjusted value."""
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(_data())
    statistics = fit.fit_statistics()
    metrics = _by_name(statistics.metrics)
    assert tuple(metrics) == (
        "n_parameters",
        "r_squared",
        "adjusted_r_squared",
        *_STATISTICS[1:],
    )
    assert metrics["n_parameters"].value == K_EXPLICIT
    r2 = metrics["r_squared"].value
    assert r2 is not None
    assert metrics["adjusted_r_squared"].value == pytest.approx(
        1.0 - (1.0 - r2) * (N - 1) / (N - 2)
    )
    assert statistics.notes[1].startswith("R² is centred and unadjusted")
    assert "precision weights" not in statistics.notes[1]
    assert "uncentred_r_squared" not in metrics


@pytest.mark.unit
@pytest.mark.contract
def test_implicit_constant_from_indicators_is_centred() -> None:
    """Full indicator columns span a constant without an explicit intercept."""
    fit = Regression.ols("y", Linear("g0") + Linear("g1") + Linear("x")).estimate(
        _data()
    )
    metrics = _by_name(fit.fit_statistics().metrics)
    assert "r_squared" in metrics
    assert "adjusted_r_squared" in metrics
    assert "uncentred_r_squared" not in metrics
    assert metrics["n_parameters"].value == K_INDICATORS


@pytest.mark.unit
@pytest.mark.contract
def test_no_constant_reports_uncentred_r_squared() -> None:
    """A design that spans no constant reports uncentred R-squared only."""
    fit = Regression.ols("y", Linear("x")).estimate(_data())
    statistics = fit.fit_statistics()
    metrics = _by_name(statistics.metrics)
    assert "uncentred_r_squared" in metrics
    assert "r_squared" not in metrics
    assert "adjusted_r_squared" not in metrics
    assert statistics.notes[1].startswith("R² is uncentred")


@pytest.mark.unit
@pytest.mark.contract
def test_wls_note_and_reports_carry_statistics() -> None:
    """WLS adds the weights sentence; fit and inference reports append metrics."""
    fit = Regression.wls(
        "y", Intercept() + Linear("x"), precision=RelativePrecision("w")
    ).estimate(_data())
    statistics = fit.fit_statistics()
    assert statistics.notes[1].endswith("Both use the precision weights.")
    fit_report = fit.report()
    names = [m.name for m in fit_report.metrics]
    assert names[-len(statistics.metrics) :] == [m.name for m in statistics.metrics]
    assert fit_report.notes[-2:] == statistics.notes
    inference = fit.infer(covariance=HC("HC3")).report(reference="normal", null=0.0)
    inference_names = [m.name for m in inference.metrics]
    assert inference_names[0] == "n_observations"
    assert inference_names[1:] == [m.name for m in statistics.metrics]
    assert inference.notes[-2:] == statistics.notes
    assert inference.notes[-3] == "Two-sided tests against 0; normal reference."


@pytest.mark.unit
@pytest.mark.contract
def test_inference_default_has_no_fit_statistics() -> None:
    """Model-based inference keeps the same statistics as the fit."""
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(_data())
    inferred = fit.infer(covariance=ModelBased())
    assert inferred.fit_statistics == fit.fit_statistics()


@pytest.mark.unit
@pytest.mark.contract
def test_linear_likelihood_summary_counts_rank_plus_scale() -> None:
    """The summary states k as rank plus scale and n as the rows."""
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(_data())
    summarizer: SummarizesLikelihood = fit
    summary = summarizer.likelihood_summary()
    assert summary is not None
    assert summary.n_parameters == K_EXPLICIT
    assert summary.n_observations == N
    metrics = _by_name(fit.fit_statistics().metrics)
    assert metrics["log_likelihood"].value == pytest.approx(summary.log_likelihood)


def _assert_propagated(fit: LinearResult) -> dict[str, Metric]:
    statistics = fit.fit_statistics()
    assert fit.report().notes[-2:] == statistics.notes
    assert [m.name for m in fit.report().metrics][-len(statistics.metrics) :] == [
        m.name for m in statistics.metrics
    ]
    inferred = fit.infer(covariance=ModelBased())
    assert inferred.fit_statistics == statistics
    return _by_name(statistics.metrics)


def _assert_likelihood_unavailable(metrics: dict[str, Metric]) -> None:
    assert metrics["n_parameters"].available
    for name in _STATISTICS[1:]:
        assert not metrics[name].available


@pytest.mark.unit
@pytest.mark.contract
def test_exact_ols_fit_reports_unit_r_squared() -> None:
    """A valid exact fit keeps R-squared of one; likelihood stays unavailable."""
    x = jnp.arange(4.0)
    data = {"x": x, "y": 1.0 + x}
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(data)
    metrics = _assert_propagated(fit)
    assert metrics["r_squared"].value == pytest.approx(1.0)
    assert metrics["adjusted_r_squared"].value == pytest.approx(1.0)
    _assert_likelihood_unavailable(metrics)


@pytest.mark.unit
@pytest.mark.contract
def test_exact_uncentred_fit_reports_unit_r_squared() -> None:
    """An exact fit without a constant reports uncentred R-squared of one."""
    data = {"x": jnp.array([0.0, 1.0, 0.0]), "y": jnp.array([0.0, 2.0, 0.0])}
    fit = Regression.ols("y", Linear("x")).estimate(data)
    metrics = _assert_propagated(fit)
    assert metrics["uncentred_r_squared"].value == pytest.approx(1.0)
    _assert_likelihood_unavailable(metrics)


@pytest.mark.unit
@pytest.mark.contract
def test_exact_wls_fit_reports_unit_r_squared() -> None:
    """A weighted exact fit reports R-squared of one."""
    x = jnp.arange(4.0)
    data = {"x": x, "y": x, "w": jnp.ones(4)}
    fit = Regression.wls(
        "y", Intercept() + Linear("x"), precision=RelativePrecision("w")
    ).estimate(data)
    metrics = _assert_propagated(fit)
    assert metrics["r_squared"].value == pytest.approx(1.0)
    _assert_likelihood_unavailable(metrics)


@pytest.mark.unit
@pytest.mark.contract
def test_constant_response_r_squared_is_unavailable() -> None:
    """A constant response has no variation, so centred R-squared is undefined."""
    data = {"x": jnp.arange(3.0), "y": jnp.ones(3)}
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(data)
    metrics = _assert_propagated(fit)
    for name in ("r_squared", "adjusted_r_squared"):
        assert not metrics[name].available
        assert metrics[name].unavailable_reason == (
            "The response has no variation about its weighted mean."
        )
    _assert_likelihood_unavailable(metrics)


@pytest.mark.unit
@pytest.mark.contract
def test_saturated_fit_has_unit_r_squared_and_no_adjusted() -> None:
    """With n equal to rank R-squared is one and adjusted R-squared is undefined."""
    data = {"x": jnp.array([0.0, 1.0]), "y": jnp.array([0.0, 1.0])}
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(data)
    metrics = _assert_propagated(fit)
    assert metrics["r_squared"].value == pytest.approx(1.0)
    assert not metrics["adjusted_r_squared"].available
    assert metrics["adjusted_r_squared"].unavailable_reason == (
        "Needs positive residual degrees of freedom (n > rank)."
    )
    _assert_likelihood_unavailable(metrics)


@pytest.mark.component
@pytest.mark.property
@examples(12)
@given(
    case=linear_cases(n=10, p=3, intercept=True),
    order=st.permutations(tuple(range(10))),
    scale=st.sampled_from([0.25, 2.0, 4.0]),
    shift=st.integers(-3, 3),
)
def test_linear_workflow_equivariance(
    case: LinearCase, order: tuple[int, ...], scale: float, shift: int
) -> None:
    """Row/term permutations and affine outcomes preserve the fitted model."""
    data = {"x": case.design[:, 1], "z": case.design[:, 2], "y": case.outcome}
    model = Regression.ols("y", Intercept() + Linear("x") + Linear("z"))
    fit = model.estimate(data)
    tolerance = Tolerance(
        rtol=2e-4 if case.design.dtype == np.float32 else 2e-10,
        atol=2e-5 if case.design.dtype == np.float32 else 2e-11,
        rationale="Small full-rank dyadic design with bounded triangular anchor.",
    )
    assert_finite_close(
        fit.core.parameters,
        ols_oracle(case),
        tolerance=tolerance,
        label="QR coefficients",
    )
    indices = np.asarray(order)
    from statadic.api.data import Observations  # noqa: PLC0415

    permuted = model.estimate(
        Observations(
            {name: value[indices] for name, value in data.items()}, row_ids=order
        )
    )
    assert permuted.context.observations.row_ids == tuple(order)
    assert_finite_close(
        permuted.core.parameters,
        fit.core.parameters,
        tolerance=tolerance,
        label="row permutation",
    )
    assert_finite_close(
        permuted.core.residuals,
        np.asarray(fit.core.residuals)[indices],
        tolerance=tolerance,
        label="residual permutation",
    )
    reordered = Regression.ols("y", Intercept() + Linear("z") + Linear("x")).estimate(
        data
    )
    assert_finite_close(
        reordered.core.parameters,
        np.asarray(fit.core.parameters)[[0, 2, 1]],
        tolerance=tolerance,
        label="term permutation",
    )
    covariance = np.asarray(
        fit.infer(covariance=ModelBased()).uncertainty.covariance.values
    )
    other_covariance = reordered.infer(
        covariance=ModelBased()
    ).uncertainty.covariance.values
    assert_finite_close(
        other_covariance,
        covariance[np.ix_([0, 2, 1], [0, 2, 1])],
        tolerance=tolerance,
        label="covariance permutation",
    )
    affine = model.estimate({**data, "y": scale * case.outcome + shift})
    expected = scale * np.asarray(fit.core.parameters)
    expected[0] += shift
    assert_finite_close(
        affine.core.parameters, expected, tolerance=tolerance, label="affine outcome"
    )
    for left, right in zip(
        fit.fit_statistics().metrics, permuted.fit_statistics().metrics, strict=True
    ):
        assert left.available == right.available
        if left.available:
            assert left.value == pytest.approx(
                right.value, rel=tolerance.rtol, abs=tolerance.atol
            )


@pytest.mark.component
@pytest.mark.property
@examples(12)
@given(
    case=linear_cases(n=10, p=2, intercept=True),
    multiplier=st.sampled_from([0.25, 2.0, 4.0]),
    weights=st.permutations((1, 1, 1, 1, 1, 3, 3, 3, 3, 3)),
)
def test_analytic_precision_scaling_and_integer_replication(
    case: LinearCase, multiplier: float, weights: list[int]
) -> None:
    """Precision scaling is invariant; replication shares coefficients only."""
    data = {"x": case.design[:, 1], "y": case.outcome, "w": case.precision}
    model = Regression.wls(
        "y", Intercept() + Linear("x"), precision=RelativePrecision("w")
    )
    fit = model.estimate(data)
    scaled = model.estimate({**data, "w": multiplier * case.precision})
    tolerance = Tolerance(
        rtol=2e-4 if case.design.dtype == np.float32 else 2e-10,
        atol=2e-5 if case.design.dtype == np.float32 else 2e-11,
        rationale="Positive-weight QR; common precision scale cancels.",
    )
    assert_finite_close(
        fit.core.parameters,
        ols_oracle(case, weighted=True),
        tolerance=tolerance,
        label="weighted QR",
    )
    assert_finite_close(
        scaled.core.parameters,
        fit.core.parameters,
        tolerance=tolerance,
        label="common weight scale",
    )
    assert_finite_close(
        scaled.infer(covariance=ModelBased()).uncertainty.covariance.values,
        fit.infer(covariance=ModelBased()).uncertainty.covariance.values,
        tolerance=tolerance,
        label="precision covariance",
    )
    # Permuting a fixed-sum integer schedule keeps the replicated shape fixed.
    repeats = np.asarray(weights, dtype=np.int32)
    integer_weighted = model.estimate({**data, "w": repeats})
    repeated = Regression.ols("y", Intercept() + Linear("x")).estimate(
        {"x": np.repeat(data["x"], repeats), "y": np.repeat(data["y"], repeats)}
    )
    assert_finite_close(
        repeated.core.parameters,
        integer_weighted.core.parameters,
        tolerance=tolerance,
        label="integer replication",
    )


@pytest.mark.component
@pytest.mark.property
@pytest.mark.contract
@examples(15)
@given(case=linear_cases(n=8, p=2, intercept=True, exact=True), constant=st.booleans())
def test_exact_model_and_constant_response_r_squared(
    case: LinearCase, *, constant: bool
) -> None:
    """R-squared depends on response variation, independently of likelihood."""
    outcome = np.zeros_like(case.outcome) if constant else case.outcome
    fit = Regression.ols("y", Intercept() + Linear("x")).estimate(
        {"x": case.design[:, 1], "y": outcome}
    )
    metrics = _by_name(fit.fit_statistics().metrics)
    variation = float(np.sum((outcome - outcome.mean()) ** 2))
    assert bool(fit.core.valid)
    assert metrics["r_squared"].available == (variation > 0)
    assert metrics["adjusted_r_squared"].available == (variation > 0)
    if variation > 0:
        assert metrics["r_squared"].value == pytest.approx(1.0, abs=2e-5)
    else:
        assert metrics["r_squared"].unavailable_reason
    if float(fit.core.weighted_sse) == 0:  # noqa: RUF069 -- zero is the exact likelihood boundary
        assert not metrics["log_likelihood"].available
        assert not metrics["aic"].available


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("residual", [0.0, 1e-7])
def test_gaussian_likelihood_distinguishes_zero_and_small_positive_sse(
    residual: float,
) -> None:
    """An axis-aligned design reaches the exact zero-SSE boundary without QR noise."""
    dtype = jnp.result_type(float)
    fit = Regression.ols("y", Linear("x")).estimate(
        {
            "x": jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=dtype),
            "y": jnp.asarray([1.0, 0.0, 0.0, residual], dtype=dtype),
        }
    )
    assert bool(fit.core.valid)
    sse = float(fit.core.weighted_sse)
    metrics = _by_name(fit.fit_statistics().metrics)
    if residual == 0:
        assert sse == 0
        assert metrics["uncentred_r_squared"].value == 1
        for name in _STATISTICS[1:]:
            assert not metrics[name].available
            assert metrics[name].unavailable_reason
    else:
        assert sse > 0
        expected = -2 * (np.log(2 * np.pi) + np.log(sse / 4) + 1)
        assert metrics["log_likelihood"].value == pytest.approx(expected)
        assert all(metrics[name].available for name in _STATISTICS)
